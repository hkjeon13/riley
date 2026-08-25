#!/usr/bin/env python3
"""Fail-closed, CPU-only final release-candidate evidence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

from release_common import ReleaseContractError
from verify_release_bundle import verify_bundle


MANIFEST_VERSION = "rustinfer.release-candidate-manifest.v1"
ATTESTATION_VERSION = "rustinfer.release-gate-attestation.v1"
REPORT_VERSION = "rustinfer.release-candidate-report.v1"
PERFORMANCE_VERSION = "rustinfer.release-performance-report.v1"
SOAK_VERSION = "rustinfer.reliability-soak-report.v1"
CORRECTNESS_VERSION = "1.0.0"
CORRECTNESS_GATE = "smollm2-fp32-bf16-native-e0-v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
PLACEHOLDER_RE = re.compile(
    r"(?:placeholder|replace[-_ ]?me|sha256[-_ ]?of|\btodo\b|<[^>]+>)",
    re.IGNORECASE,
)
MAX_JSON_BYTES = 64 * 1024 * 1024

PYTHON_FREE_CHECKS = {
    "release_bundle_verified",
    "no_python_executable",
    "no_python_child",
    "no_forbidden_runtime_artifact",
    "native_dependencies_verified",
    "model_load",
    "prefill",
    "decode",
    "greedy_golden",
    "sampling",
    "streaming",
    "cancellation",
    "graceful_shutdown",
}
CUDA_FAULT_CHECKS = {
    "test_inventory_exact",
    "create_rollback_ambiguity",
    "explicit_close_ambiguity",
    "confirmed_completion_deferred_error",
    "unconfirmed_completion_retained",
    "subprocess_isolation",
    "production_fault_symbols_absent",
}


class CandidateError(ValueError):
    """Release evidence is malformed, failed, unsafe, or inconsistently bound."""


def _fail(path: str, message: str) -> NoReturn:
    raise CandidateError(f"{path}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise CandidateError(f"non-finite JSON number {value!r} is forbidden")


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        if metadata.st_size > MAX_JSON_BYTES:
            _fail(label, f"exceeds the {MAX_JSON_BYTES}-byte JSON bound")
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except FileNotFoundError:
        _fail(label, f"does not exist: {path}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"cannot read strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        _fail(label, "root must be an object")
    _reject_placeholders(value, label)
    return value, raw


def _reject_placeholders(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if PLACEHOLDER_RE.search(value):
            _fail(path, "contains a placeholder marker")
        if value in {"0" * 40, "0" * 64, f"sha256:{'0' * 64}"}:
            _fail(path, "all-zero placeholder value is forbidden")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _exact(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    result = _object(value, path)
    missing = sorted(keys - set(result))
    extra = sorted(set(result) - keys)
    if missing or extra:
        _fail(path, f"closed object mismatch; missing={missing}, unexpected={extra}")
    return result


def _string(value: Any, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    if PLACEHOLDER_RE.search(value):
        _fail(path, "contains a placeholder marker")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, "has invalid format")
    return value


def _sha256(value: Any, path: str) -> str:
    digest = _string(value, path, SHA256_RE)
    if digest == "0" * 64:
        _fail(path, "all-zero placeholder digest is forbidden")
    return digest


def _revision(value: Any, path: str) -> str:
    revision = _string(value, path, GIT_RE)
    if revision == "0" * 40:
        _fail(path, "all-zero placeholder revision is forbidden")
    return revision


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        _fail(str(path), f"cannot hash artifact: {error}")
    return digest.hexdigest()


def _resolve_artifact(
    value: Any,
    path: str,
    evidence_root: Path,
    seen_paths: set[str],
) -> tuple[Path, str, str]:
    artifact = _exact(value, {"path", "sha256"}, path)
    relative = _string(artifact["path"], f"{path}.path")
    if "\\" in relative or "//" in relative:
        _fail(f"{path}.path", "must use a normalized POSIX relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"{path}.path", "path traversal and absolute paths are forbidden")
    normalized = pure.as_posix()
    if normalized in seen_paths:
        _fail(f"{path}.path", "artifact path is duplicated")
    seen_paths.add(normalized)
    candidate = evidence_root.joinpath(*pure.parts)
    current = evidence_root
    for part in pure.parts:
        current = current / part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                _fail(f"{path}.path", "symlink path components are forbidden")
        except OSError as error:
            _fail(f"{path}.path", f"cannot inspect artifact path component: {error}")
    try:
        metadata = candidate.lstat()
    except OSError as error:
        _fail(f"{path}.path", f"cannot inspect artifact: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{path}.path", "artifact must be a regular file, not a link or device")
    try:
        candidate.resolve(strict=True).relative_to(evidence_root)
    except (OSError, ValueError):
        _fail(f"{path}.path", "artifact resolves outside the evidence root")
    declared = _sha256(artifact["sha256"], f"{path}.sha256")
    actual = _file_sha256(candidate)
    if actual != declared:
        _fail(f"{path}.sha256", f"artifact digest mismatch: {actual}")
    return candidate, declared, normalized


def _all_checks_pass(checks: Any, path: str) -> None:
    if not isinstance(checks, list) or not checks:
        _fail(path, "must be a non-empty array")
    for index, raw in enumerate(checks):
        check = _object(raw, f"{path}[{index}]")
        if check.get("passed") is not True:
            _fail(f"{path}[{index}].passed", "must be true")


def _validate_attestation(
    report: dict[str, Any],
    path: str,
    *,
    gate: str,
    required_checks: set[str],
    source: dict[str, Any],
    release: dict[str, Any],
    raw_sha256: str,
) -> None:
    row = _exact(
        report,
        {"schema_version", "gate", "status", "source", "raw_evidence_sha256", "checks"},
        path,
    )
    if row["schema_version"] != ATTESTATION_VERSION:
        _fail(f"{path}.schema_version", f"must be {ATTESTATION_VERSION}")
    if row["gate"] != gate:
        _fail(f"{path}.gate", f"must be {gate}")
    if row["status"] != "passed":
        _fail(f"{path}.status", "must be passed")
    binding = _exact(
        row["source"],
        {
            "git_revision", "git_dirty", "source_archive_sha256",
            "release_binary_sha256", "release_bundle_sha256", "release_image_sha256",
        },
        f"{path}.source",
    )
    expected = {
        "git_revision": source["git_revision"],
        "git_dirty": False,
        "source_archive_sha256": source["archive_sha256"],
        "release_binary_sha256": release["binary_sha256"],
        "release_bundle_sha256": release["bundle_sha256"],
        "release_image_sha256": release["image_sha256"],
    }
    if binding != expected:
        _fail(f"{path}.source", "does not exactly match candidate bindings")
    if row["raw_evidence_sha256"] != raw_sha256:
        _fail(f"{path}.raw_evidence_sha256", "does not bind the raw evidence artifact")
    checks = row["checks"]
    if not isinstance(checks, list):
        _fail(f"{path}.checks", "must be an array")
    observed: set[str] = set()
    for index, raw in enumerate(checks):
        check = _exact(raw, {"id", "passed"}, f"{path}.checks[{index}]")
        check_id = _string(check["id"], f"{path}.checks[{index}].id", ID_RE)
        if check_id in observed:
            _fail(f"{path}.checks[{index}].id", "duplicate check id")
        observed.add(check_id)
        if check["passed"] is not True:
            _fail(f"{path}.checks[{index}].passed", "must be true")
    if observed != required_checks:
        _fail(f"{path}.checks", f"required check set mismatch: {sorted(observed)}")


def _validate_correctness(
    report: dict[str, Any], path: str, revision: str
) -> None:
    row = _exact(
        report,
        {
            "schema_version", "gate_id", "created_at", "status", "roles",
            "gate_contract", "inputs", "bindings", "summary", "cases",
        },
        path,
    )
    if row["schema_version"] != CORRECTNESS_VERSION or row["gate_id"] != CORRECTNESS_GATE:
        _fail(path, "must be the reviewed native E0 correctness report v2")
    if row["status"] != "pass":
        _fail(f"{path}.status", "must be pass")
    bindings = _object(row["bindings"], f"{path}.bindings")
    if bindings.get("candidate_git_revision") != revision:
        _fail(f"{path}.bindings.candidate_git_revision", "source revision mismatch")
    if bindings.get("candidate_git_status_sha256") != hashlib.sha256(b"").hexdigest():
        _fail(f"{path}.bindings.candidate_git_status_sha256", "source tree was not clean")
    _sha256(bindings.get("candidate_executable_sha256"), f"{path}.bindings.candidate_executable_sha256")
    summary = _object(row["summary"], f"{path}.summary")
    expected_summary = {
        "case_count": 31,
        "candidate_variant_count": 2,
        "failure_count": 0,
        "numeric_pass": True,
        "semantic_pass": True,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            _fail(f"{path}.summary.{key}", f"must be {expected!r}")
    variants = _object(summary.get("variants"), f"{path}.summary.variants")
    if set(variants) != {"canonical-v1", "fixed-contiguous-37-balanced-v1"}:
        _fail(f"{path}.summary.variants", "required E0 variant set mismatch")
    for name, variant in variants.items():
        variant_row = _object(variant, f"{path}.summary.variants.{name}")
        expected = {
            "case_count": 31,
            "failure_count": 0,
            "numeric_pass": True,
            "semantic_pass": True,
            "pass": True,
        }
        if any(variant_row.get(key) != value for key, value in expected.items()):
            _fail(f"{path}.summary.variants.{name}", "variant did not pass")
    cases = row["cases"]
    if not isinstance(cases, list) or len(cases) != 31:
        _fail(f"{path}.cases", "must contain exactly 31 cases")
    prompt_ids: set[str] = set()
    for index, raw in enumerate(cases):
        case_path = f"{path}.cases[{index}]"
        case = _exact(raw, {"prompt_id", "variants", "pass"}, case_path)
        prompt_id = _string(case.get("prompt_id"), f"{path}.cases[{index}].prompt_id", ID_RE)
        if prompt_id in prompt_ids:
            _fail(f"{path}.cases[{index}].prompt_id", "duplicate prompt id")
        prompt_ids.add(prompt_id)
        if case.get("pass") is not True:
            _fail(f"{path}.cases[{index}].pass", "must be true")
        case_variants = _exact(
            case["variants"],
            {"canonical-v1", "fixed-contiguous-37-balanced-v1"},
            f"{case_path}.variants",
        )
        for name, raw_variant in case_variants.items():
            variant = _exact(
                raw_variant, {"numeric", "semantic", "pass"}, f"{case_path}.variants.{name}"
            )
            if variant["pass"] is not True:
                _fail(f"{case_path}.variants.{name}.pass", "must be true")
            semantic = _object(variant["semantic"], f"{case_path}.variants.{name}.semantic")
            if semantic.get("pass") is not True:
                _fail(f"{case_path}.variants.{name}.semantic.pass", "must be true")
            numeric = _object(variant["numeric"], f"{case_path}.variants.{name}.numeric")
            for metric_name in ("first_layer_hidden", "final_logits", "final_log_probs"):
                metric = _object(
                    numeric.get(metric_name), f"{case_path}.variants.{name}.numeric.{metric_name}"
                )
                if metric.get("pass") is not True:
                    _fail(
                        f"{case_path}.variants.{name}.numeric.{metric_name}.pass",
                        "must be true",
                    )


def _validate_performance(
    report: dict[str, Any],
    path: str,
    *,
    revision: str,
    archive_sha256: str,
    binary_sha256: str,
    image_sha256: str,
    correctness_sha256: str,
) -> None:
    row = _exact(
        report,
        {"schema_version", "status", "passed", "baseline", "candidate", "ratios", "checks", "errors"},
        path,
    )
    if row["schema_version"] != PERFORMANCE_VERSION or row["status"] != "passed" or row["passed"] is not True:
        _fail(path, "performance gate did not pass")
    if row["errors"] != []:
        _fail(f"{path}.errors", "must be empty")
    candidate = _exact(
        row["candidate"],
        {"candidate_id", "recorded_at_utc", "source", "metrics", "run_summary", "raw_runs"},
        f"{path}.candidate",
    )
    binding = _exact(
        candidate["source"],
        {
            "git_commit", "git_dirty", "source_archive_sha256", "profile_binary_sha256",
            "release_binary_sha256", "profile_image_sha256", "release_image_sha256",
            "semantic_class", "correctness_gate_id", "correctness_report_sha256",
        },
        f"{path}.candidate.source",
    )
    expected = {
        "git_commit": revision,
        "git_dirty": False,
        "source_archive_sha256": archive_sha256,
        "release_binary_sha256": binary_sha256,
        "release_image_sha256": image_sha256,
        "correctness_report_sha256": correctness_sha256,
        "semantic_class": "E0",
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            _fail(f"{path}.candidate.source.{key}", "candidate binding mismatch")
    _sha256(binding["profile_binary_sha256"], f"{path}.candidate.source.profile_binary_sha256")
    _sha256(binding["profile_image_sha256"], f"{path}.candidate.source.profile_image_sha256")
    _string(binding["correctness_gate_id"], f"{path}.candidate.source.correctness_gate_id")
    _all_checks_pass(row["checks"], f"{path}.checks")


def _validate_soak(
    report: dict[str, Any],
    path: str,
    *,
    revision: str,
    archive_sha256: str,
    binary_sha256: str,
    image_sha256: str,
) -> None:
    row = _exact(
        report,
        {"schema_version", "status", "passed", "bindings", "scenario_summaries", "observations", "checks", "errors"},
        path,
    )
    if row["schema_version"] != SOAK_VERSION or row["status"] != "passed" or row["passed"] is not True:
        _fail(path, "reliability soak did not pass")
    if row["errors"] != []:
        _fail(f"{path}.errors", "must be empty")
    bindings = _exact(
        row["bindings"], {"manifest_sha256", "binding_sha256", "source"}, f"{path}.bindings"
    )
    _sha256(bindings["manifest_sha256"], f"{path}.bindings.manifest_sha256")
    _sha256(bindings["binding_sha256"], f"{path}.bindings.binding_sha256")
    source = _exact(
        bindings["source"],
        {
            "git_commit", "git_dirty", "source_archive_sha256", "binary_sha256",
            "image_sha256", "model_sha256", "model_id", "model_revision",
        },
        f"{path}.bindings.source",
    )
    expected = {
        "git_commit": revision,
        "git_dirty": False,
        "source_archive_sha256": archive_sha256,
        "binary_sha256": binary_sha256,
        "image_sha256": image_sha256,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            _fail(f"{path}.bindings.source.{key}", "candidate binding mismatch")
    _sha256(source["model_sha256"], f"{path}.bindings.source.model_sha256")
    _string(source["model_id"], f"{path}.bindings.source.model_id")
    _string(source["model_revision"], f"{path}.bindings.source.model_revision")
    _all_checks_pass(row["checks"], f"{path}.checks")


def _verify_bundle_binding(bundle: Path, binary_sha256: str, revision: str) -> None:
    try:
        verify_bundle(bundle)
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            release_members = [member for member in members if member.name.endswith("/manifest/release.json")]
            binary_members = [member for member in members if member.name.endswith("/bin/rustinfer")]
            if len(release_members) != 1 or len(binary_members) != 1:
                _fail("manifest.release.bundle", "cannot locate unique manifest and binary")
            release_file = archive.extractfile(release_members[0])
            binary_file = archive.extractfile(binary_members[0])
            if release_file is None or binary_file is None:
                _fail("manifest.release.bundle", "cannot read manifest or binary")
            release_manifest = json.loads(
                release_file.read(), object_pairs_hook=_pairs, parse_constant=_nonfinite
            )
            internal_binary_sha256 = hashlib.sha256(binary_file.read()).hexdigest()
    except (OSError, tarfile.TarError, json.JSONDecodeError, ReleaseContractError) as error:
        _fail("manifest.release.bundle", f"release bundle verification failed: {error}")
    if not isinstance(release_manifest, dict):
        _fail("manifest.release.bundle", "embedded release manifest is not an object")
    artifact = _object(release_manifest.get("artifact"), "release manifest.artifact")
    if artifact.get("source_revision") != revision:
        _fail("release manifest.artifact.source_revision", "candidate revision mismatch")
    if internal_binary_sha256 != binary_sha256:
        _fail("manifest.release.binary", "standalone binary differs from bundle binary")


def _empty_report() -> dict[str, Any]:
    return {
        "schema_version": REPORT_VERSION,
        "status": "error",
        "passed": False,
        "candidate_id": None,
        "manifest_sha256": None,
        "bindings": None,
        "checks": [],
        "errors": [],
    }


def evaluate(manifest_path: Path, evidence_root: Path) -> dict[str, Any]:
    """Validate a final tag candidate without executing the release or CUDA."""

    report = _empty_report()
    try:
        evidence_root = evidence_root.resolve(strict=True)
        if not evidence_root.is_dir():
            _fail("--evidence-root", "must be a directory")
        manifest, manifest_raw = _load_json(manifest_path, "manifest")
        row = _exact(
            manifest,
            {"schema_version", "candidate_id", "source", "release", "evidence"},
            "manifest",
        )
        if row["schema_version"] != MANIFEST_VERSION:
            _fail("manifest.schema_version", f"must be {MANIFEST_VERSION}")
        candidate_id = _string(row["candidate_id"], "manifest.candidate_id", ID_RE)
        source_row = _exact(row["source"], {"git_revision", "git_dirty", "archive"}, "manifest.source")
        revision = _revision(source_row["git_revision"], "manifest.source.git_revision")
        if source_row["git_dirty"] is not False:
            _fail("manifest.source.git_dirty", "release source must be clean")
        release_row = _exact(row["release"], {"binary", "bundle", "image_digest"}, "manifest.release")
        image_digest = _string(release_row["image_digest"], "manifest.release.image_digest")
        if not image_digest.startswith("sha256:"):
            _fail("manifest.release.image_digest", "must be sha256:<lowercase digest>")
        image_sha256 = _sha256(image_digest.removeprefix("sha256:"), "manifest.release.image_digest")
        evidence_row = _exact(
            row["evidence"],
            {"python_free_e2e", "cuda_fault", "correctness", "performance", "reliability_soak"},
            "manifest.evidence",
        )
        seen_paths: set[str] = set()
        _, archive_sha256, _ = _resolve_artifact(
            source_row["archive"], "manifest.source.archive", evidence_root, seen_paths
        )
        binary_path, binary_sha256, _ = _resolve_artifact(
            release_row["binary"], "manifest.release.binary", evidence_root, seen_paths
        )
        bundle_path, bundle_sha256, _ = _resolve_artifact(
            release_row["bundle"], "manifest.release.bundle", evidence_root, seen_paths
        )
        if not os.access(binary_path, os.X_OK):
            _fail("manifest.release.binary", "must be executable")
        _verify_bundle_binding(bundle_path, binary_sha256, revision)
        source = {"git_revision": revision, "archive_sha256": archive_sha256}
        release = {
            "binary_sha256": binary_sha256,
            "bundle_sha256": bundle_sha256,
            "image_sha256": image_sha256,
        }

        loaded: dict[str, tuple[dict[str, Any], str]] = {}
        raw_hashes: dict[str, str] = {}
        for gate_name in ("python_free_e2e", "cuda_fault"):
            gate = _exact(evidence_row[gate_name], {"report", "raw_evidence"}, f"manifest.evidence.{gate_name}")
            report_path, report_sha, _ = _resolve_artifact(
                gate["report"], f"manifest.evidence.{gate_name}.report", evidence_root, seen_paths
            )
            _, raw_sha, _ = _resolve_artifact(
                gate["raw_evidence"], f"manifest.evidence.{gate_name}.raw_evidence", evidence_root, seen_paths
            )
            gate_report, _ = _load_json(report_path, f"{gate_name} report")
            loaded[gate_name] = (gate_report, report_sha)
            raw_hashes[gate_name] = raw_sha
        for gate_name in ("correctness", "performance", "reliability_soak"):
            gate = _exact(evidence_row[gate_name], {"report"}, f"manifest.evidence.{gate_name}")
            report_path, report_sha, _ = _resolve_artifact(
                gate["report"], f"manifest.evidence.{gate_name}.report", evidence_root, seen_paths
            )
            gate_report, _ = _load_json(report_path, f"{gate_name} report")
            loaded[gate_name] = (gate_report, report_sha)

        _validate_attestation(
            loaded["python_free_e2e"][0], "python_free_e2e",
            gate="python-free-clean-runtime-e2e", required_checks=PYTHON_FREE_CHECKS,
            source=source, release=release, raw_sha256=raw_hashes["python_free_e2e"],
        )
        _validate_attestation(
            loaded["cuda_fault"][0], "cuda_fault",
            gate="cuda-fault-injection", required_checks=CUDA_FAULT_CHECKS,
            source=source, release=release, raw_sha256=raw_hashes["cuda_fault"],
        )
        correctness_sha256 = loaded["correctness"][1]
        _validate_correctness(loaded["correctness"][0], "correctness", revision)
        _validate_performance(
            loaded["performance"][0], "performance", revision=revision,
            archive_sha256=archive_sha256, binary_sha256=binary_sha256,
            image_sha256=image_sha256, correctness_sha256=correctness_sha256,
        )
        _validate_soak(
            loaded["reliability_soak"][0], "reliability_soak", revision=revision,
            archive_sha256=archive_sha256, binary_sha256=binary_sha256,
            image_sha256=image_sha256,
        )
        evidence_hashes = {name: digest for name, (_, digest) in loaded.items()}
        evidence_hashes.update({f"{name}_raw": digest for name, digest in raw_hashes.items()})
        report.update(
            {
                "status": "passed",
                "passed": True,
                "candidate_id": candidate_id,
                "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "bindings": {
                    "git_revision": revision,
                    "source_archive_sha256": archive_sha256,
                    "release_binary_sha256": binary_sha256,
                    "release_bundle_sha256": bundle_sha256,
                    "release_image_sha256": image_sha256,
                    "evidence_sha256": dict(sorted(evidence_hashes.items())),
                },
                "checks": [
                    {"name": name, "passed": True}
                    for name in (
                        "release_bundle", "python_free_e2e", "cuda_fault",
                        "correctness", "performance", "reliability_soak", "cross_bindings",
                    )
                ],
            }
        )
    except (CandidateError, OSError) as error:
        report["errors"] = [str(error)]
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="create without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(args.manifest, args.evidence_root)
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            with args.report.open("x", encoding="utf-8", newline="") as handle:
                handle.write(encoded)
        except (FileExistsError, OSError) as error:
            print(f"cannot create report {args.report}: {error}", file=sys.stderr)
            return 2
    sys.stdout.write(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
