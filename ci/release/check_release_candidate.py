#!/usr/bin/env python3
"""Fail-closed, CPU-only final release-candidate evidence gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

from release_common import ReleaseContractError
from verify_release_bundle import verify_bundle


_PERFORMANCE_CHECKER_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/scripts/check_release_performance.py"
)
_PERFORMANCE_SPEC = importlib.util.spec_from_file_location(
    "rustinfer_final_candidate_performance_contract",
    _PERFORMANCE_CHECKER_PATH,
)
if _PERFORMANCE_SPEC is None or _PERFORMANCE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load performance contract: {_PERFORMANCE_CHECKER_PATH}")
release_performance = importlib.util.module_from_spec(_PERFORMANCE_SPEC)
sys.modules[_PERFORMANCE_SPEC.name] = release_performance
_PERFORMANCE_SPEC.loader.exec_module(release_performance)

_PYTHON_FREE_E2E_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/scripts/check_python_free_release_e2e.py"
)
_PYTHON_FREE_E2E_SPEC = importlib.util.spec_from_file_location(
    "rustinfer_final_candidate_python_free_e2e_contract",
    _PYTHON_FREE_E2E_PATH,
)
if _PYTHON_FREE_E2E_SPEC is None or _PYTHON_FREE_E2E_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load Python-free E2E contract: {_PYTHON_FREE_E2E_PATH}")
python_free_e2e = importlib.util.module_from_spec(_PYTHON_FREE_E2E_SPEC)
sys.modules[_PYTHON_FREE_E2E_SPEC.name] = python_free_e2e
_PYTHON_FREE_E2E_SPEC.loader.exec_module(python_free_e2e)


MANIFEST_VERSION = "rustinfer.release-candidate-manifest.v1"
ATTESTATION_VERSION = "rustinfer.release-gate-attestation.v1"
REPORT_VERSION = "rustinfer.release-candidate-report.v1"
PERFORMANCE_VERSION = "rustinfer.release-performance-report.v1"
SOAK_VERSION = "rustinfer.reliability-soak-report.v1"
CORRECTNESS_VERSION = "1.0.0"
CORRECTNESS_GATE = "smollm2-fp32-bf16-native-e0-v2"
NATIVE_REPLAY_VERSION = "rustinfer.native-correctness-replay-validation.v1"
OPTIMIZATION_GATE = "pr15-iteration-command-batch-exact-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
PLACEHOLDER_RE = re.compile(
    r"(?:placeholder|replace[-_ ]?me|sha256[-_ ]?of|\btodo\b|<[^>]+>)",
    re.IGNORECASE,
)
MAX_JSON_BYTES = 64 * 1024 * 1024
PERFORMANCE_BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/release/performance-baseline-v1.json"
)
PERFORMANCE_RAW_FILES = {f"candidate-{index}.json" for index in range(1, 6)}
PERFORMANCE_CHECKS = {
    "ttft_p95_regression": ("ttft_p95_ms", "<=", 1.05),
    "tpot_p95_regression": ("tpot_p95_ms", "<=", 1.05),
    "e2e_median_regression": ("e2e_median_ms", "<=", 1.05),
    "throughput_median_regression": (
        "throughput_median_output_tokens_per_second",
        ">=",
        0.95,
    ),
}

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
OPTIMIZATION_LOGS = {
    "cuda-compile-only": "cuda-compile-only.log",
    "workspace-all-features-all-targets": "workspace-all-features-all-targets.log",
    "command-batch-lifecycle": "command-batch-lifecycle-gpu.log",
    "command-batch-resource-ledger": "command-batch-primitives-gpu.log",
    "smollm2-multi-step-greedy-exact": "iteration-command-batch-model-parity-gpu.log",
}
EXPECTED_OPTIMIZATION_TOKENS = [
    4052, 2025, 284, 965, 6497, 288, 1492, 418,
    260, 16438, 30, 198, 198, 504, 16438, 314,
]
SOAK_CONTRACT_ID = "pr16-release-soak-v1"
SOAK_TEMPLATE_CANONICAL_SHA256 = (
    "5ef79434e79e6ac36e6fab4a54b2466572a62b98f972059a20fa59d4f8e7a096"
)
SOAK_SCENARIOS = {
    "steady": ("steady", 14_400),
    "burst-idle": ("burst-idle", 3_600),
    "mixed-short-long": ("mixed", 3_600),
    "invalid": ("invalid", 300),
    "overload": ("overload", 600),
    "cancellation-disconnect": ("cancellation-disconnect", 900),
    "near-kv": ("near-kv", 1_800),
    "graceful-restart": ("graceful-restart", 300),
    "rollback-iteration-batch": ("rollback", 300),
    "rollback-per-operation": ("rollback", 300),
}
SOAK_COMMON_SCENARIO_CHECKS = {
    "complete",
    "execution_completion",
    "service_counters_monotonic",
    "samples",
    "requests",
    "duration_seconds",
    "sample_coverage_seconds",
    "sample_gap_ms",
    "rss_plateau_growth",
    "rss_slope_per_hour",
    "vram_plateau_growth",
    "vram_slope_per_hour",
    "request_outcomes",
}
SOAK_GOLDEN_SCENARIOS = {
    "steady",
    "burst-idle",
    "graceful-restart",
    "rollback-iteration-batch",
    "rollback-per-operation",
}
SOAK_GLOBAL_CHECKS = {
    "run_boundaries",
    "final_sample_position",
    "initial_target_pid_binding",
    "final_quiescence",
    "no_python_children",
    "no_dropped_samples",
    "no_failure_events",
    "cancellations_observed",
    "disconnects_observed",
    "overloads_observed",
    "service_cancellations_observed",
    "service_disconnects_observed",
    "service_overloads_observed",
    "graceful_restart_golden_parity",
    "rollback_golden_parity",
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


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        _fail(path, "is outside the finite reviewed range")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        _fail(str(path), f"cannot hash artifact: {error}")
    return digest.hexdigest()


def _safe_tar_members(archive: tarfile.TarFile, path: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        _fail(path, "archive is empty")
    names: set[str] = set()
    for member in members:
        name = member.name
        pure = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or "//" in name
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            _fail(path, f"unsafe archive member path: {name!r}")
        if name in names:
            _fail(path, f"duplicate archive member path: {name}")
        names.add(name)
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            _fail(path, f"links and special archive members are forbidden: {name}")
        if not member.isdir() and not member.isreg():
            _fail(path, f"unsupported archive member type: {name}")
    return members


def _verify_source_archive(path: Path, revision: str) -> None:
    """Require the commit marker emitted by `git archive --format=tar`."""

    try:
        with tarfile.open(path, "r:") as archive:
            members = _safe_tar_members(archive, "manifest.source.archive")
            if archive.pax_headers != {"comment": revision}:
                _fail(
                    "manifest.source.archive",
                    "missing or mismatched git-archive pax global comment",
                )
            for member in members:
                if member.pax_headers.get("comment") != revision:
                    _fail(
                        "manifest.source.archive",
                        f"member lacks the exact git commit marker: {member.name}",
                    )
    except (OSError, tarfile.TarError) as error:
        _fail("manifest.source.archive", f"not a readable git tar archive: {error}")


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


def _validate_python_free_e2e_replay(
    report: dict[str, Any],
    raw_evidence_path: Path,
    *,
    revision: str,
    archive_sha256: str,
    binary_sha256: str,
    bundle_sha256: str,
    image_sha256: str,
    native_correctness: dict[str, Any],
    native_correctness_sha256: str,
    optimization_correctness: dict[str, Any],
) -> None:
    try:
        archive = python_free_e2e.load_raw_evidence_archive(raw_evidence_path)
    except (python_free_e2e.EvidenceError, OSError) as error:
        _fail("python_free_e2e.raw_evidence", str(error))
    replayed, diagnostic = python_free_e2e.validate_bound_raw_archive(
        archive,
        source_revision=revision,
        source_archive_sha256=archive_sha256,
        release_binary_sha256=binary_sha256,
        release_bundle_sha256=bundle_sha256,
        image_id=f"sha256:{image_sha256}",
        correctness_report=native_correctness,
        correctness_report_sha256=native_correctness_sha256,
    )
    if diagnostic is not None or replayed.get("status") != "passed":
        _fail("python_free_e2e.raw_evidence", diagnostic or "raw replay failed")
    if replayed != report:
        _fail("python_free_e2e", "submitted attestation differs from raw replay")

    raw_model = _object(
        _object(archive["raw"], "python_free_e2e.raw").get("model"),
        "python_free_e2e.raw.model",
    )
    optimizer_model = _object(
        optimization_correctness.get("model"), "optimization_correctness.model"
    )
    cross_bindings = {
        "model_id": (raw_model.get("model_id"), optimizer_model.get("model_id")),
        "model_revision": (
            raw_model.get("model_revision"),
            optimizer_model.get("revision"),
        ),
        "weights_sha256": (
            raw_model.get("weights_sha256"),
            optimizer_model.get("weights_sha256"),
        ),
        "tokenizer_json_sha256": (
            raw_model.get("tokenizer_json_sha256"),
            optimizer_model.get("tokenizer_sha256"),
        ),
    }
    for field, (e2e_value, optimizer_value) in cross_bindings.items():
        if e2e_value != optimizer_value:
            _fail(
                f"python_free_e2e.raw.model.{field}",
                "does not match optimizer/model provenance",
            )


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
    expected_model_bindings = {
        "model_id": python_free_e2e.MODEL_ID,
        "model_revision": python_free_e2e.MODEL_REVISION,
        "config_sha256": python_free_e2e.MODEL_CONFIG_SHA256,
        "weights_sha256": python_free_e2e.MODEL_WEIGHTS_SHA256,
        "tokenizer_sha256": python_free_e2e.TOKENIZER_AGGREGATE_SHA256,
    }
    for field, expected in expected_model_bindings.items():
        if bindings.get(field) != expected:
            _fail(f"{path}.bindings.{field}", "reviewed model binding mismatch")
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


def _validate_native_replay(
    report: dict[str, Any],
    path: str,
    *,
    revision: str,
    archive_sha256: str,
    correctness_sha256: str,
    raw_replay_sha256: str,
) -> None:
    row = _exact(
        report,
        {
            "schema_version", "status", "source", "correctness_report_sha256",
            "raw_replay_sha256", "case_count", "failure_count", "checks",
        },
        path,
    )
    if row["schema_version"] != NATIVE_REPLAY_VERSION or row["status"] != "passed":
        _fail(path, "native correctness replay validation did not pass")
    source = _exact(
        row["source"],
        {"git_revision", "git_dirty", "source_archive_sha256"},
        f"{path}.source",
    )
    if source != {
        "git_revision": revision,
        "git_dirty": False,
        "source_archive_sha256": archive_sha256,
    }:
        _fail(f"{path}.source", "does not exactly match candidate source")
    if row["correctness_report_sha256"] != correctness_sha256:
        _fail(f"{path}.correctness_report_sha256", "native report digest mismatch")
    if row["raw_replay_sha256"] != raw_replay_sha256:
        _fail(f"{path}.raw_replay_sha256", "raw replay bundle digest mismatch")
    if row["case_count"] != 31 or row["failure_count"] != 0:
        _fail(path, "native replay must pass exactly 31 cases with zero failures")
    checks = row["checks"]
    if not isinstance(checks, list):
        _fail(f"{path}.checks", "must be an array")
    required = {
        "schema-closed-validation",
        "raw-input-hashes-replayed",
        "all-cases-replayed",
        "summary-recomputed",
    }
    observed: set[str] = set()
    for index, raw in enumerate(checks):
        check = _exact(raw, {"id", "passed"}, f"{path}.checks[{index}]")
        check_id = _string(check["id"], f"{path}.checks[{index}].id", ID_RE)
        if check_id in observed:
            _fail(f"{path}.checks[{index}].id", "duplicate check id")
        observed.add(check_id)
        if check["passed"] is not True:
            _fail(f"{path}.checks[{index}].passed", "must be true")
    if observed != required:
        _fail(f"{path}.checks", f"required replay check set mismatch: {sorted(observed)}")


def _optimization_test(
    value: Any, path: str, test_id: str, expected: dict[str, Any]
) -> str:
    row = _exact(value, {"id", "result", "log_sha256", *expected}, path)
    if row["id"] != test_id or row["result"] != "passed":
        _fail(path, "test id/result mismatch")
    for key, expected_value in expected.items():
        if row[key] != expected_value:
            _fail(f"{path}.{key}", f"must be {expected_value!r}")
    return _sha256(row["log_sha256"], f"{path}.log_sha256")


def _optimization_log_hashes(path: Path) -> dict[str, str]:
    try:
        with tarfile.open(path, "r:*") as archive:
            members = _safe_tar_members(archive, "optimization_correctness.raw_evidence")
            files = [member for member in members if member.isreg()]
            by_basename: dict[str, tarfile.TarInfo] = {}
            for member in files:
                basename = PurePosixPath(member.name).name
                if basename in by_basename:
                    _fail(
                        "optimization_correctness.raw_evidence",
                        f"duplicate log basename: {basename}",
                    )
                by_basename[basename] = member
            expected_files = set(OPTIMIZATION_LOGS.values())
            if set(by_basename) != expected_files:
                _fail(
                    "optimization_correctness.raw_evidence",
                    f"exact log inventory mismatch: {sorted(by_basename)}",
                )
            result: dict[str, str] = {}
            for test_id, filename in OPTIMIZATION_LOGS.items():
                source = archive.extractfile(by_basename[filename])
                if source is None:
                    _fail("optimization_correctness.raw_evidence", f"cannot read {filename}")
                result[test_id] = hashlib.sha256(source.read()).hexdigest()
            return result
    except (OSError, tarfile.TarError) as error:
        _fail("optimization_correctness.raw_evidence", f"cannot read log archive: {error}")


def _performance_raw_payloads(path: Path) -> list[tuple[str, bytes]]:
    evidence_path = "performance.raw_evidence"
    try:
        with tarfile.open(path, "r:*") as archive:
            members = _safe_tar_members(archive, evidence_path)
            files: dict[str, tarfile.TarInfo] = {}
            for member in members:
                if not member.isreg() or member.name not in PERFORMANCE_RAW_FILES:
                    _fail(
                        evidence_path,
                        "must contain only candidate-1.json through candidate-5.json",
                    )
                if member.size <= 0 or member.size > release_performance.native_profile.MAX_EVIDENCE_BYTES:
                    _fail(
                        evidence_path,
                        f"raw run is empty or exceeds the evidence bound: {member.name}",
                    )
                files[member.name] = member
            if set(files) != PERFORMANCE_RAW_FILES:
                _fail(
                    evidence_path,
                    f"exact raw run inventory mismatch: {sorted(files)}",
                )
            payloads: list[tuple[str, bytes]] = []
            for name in sorted(files):
                source = archive.extractfile(files[name])
                if source is None:
                    _fail(evidence_path, f"cannot read {name}")
                raw = source.read(
                    release_performance.native_profile.MAX_EVIDENCE_BYTES + 1
                )
                if len(raw) != files[name].size:
                    _fail(evidence_path, f"truncated or oversized raw run: {name}")
                payloads.append((f"{evidence_path}:{name}", raw))
            return payloads
    except CandidateError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail(evidence_path, f"cannot read raw run archive: {error}")


def _reviewed_performance_baseline() -> dict[str, Any]:
    try:
        document, raw = release_performance._load_json_bytes(
            PERFORMANCE_BASELINE_PATH, "reviewed performance baseline"
        )
        return release_performance._validate_baseline(document, raw)
    except (release_performance.InputError, OSError) as error:
        _fail("performance.baseline", str(error))


def _verify_native_replay_archive(path: Path) -> None:
    try:
        with tarfile.open(path, "r:*") as archive:
            members = _safe_tar_members(archive, "native_correctness.raw_replay")
            if not any(member.isreg() for member in members):
                _fail("native_correctness.raw_replay", "must contain replay evidence files")
    except (OSError, tarfile.TarError) as error:
        _fail("native_correctness.raw_replay", f"cannot read replay archive: {error}")


def _validate_optimization_correctness(
    report: dict[str, Any],
    path: str,
    *,
    revision: str,
    archive_sha256: str,
    raw_evidence_path: Path,
) -> str:
    row = _exact(
        report,
        {
            "schema_version", "gate_id", "recorded_at_utc", "status", "semantic_class",
            "source", "build", "gpu", "model", "implementations", "tests",
        },
        path,
    )
    if row["schema_version"] != 1 or row["gate_id"] != OPTIMIZATION_GATE:
        _fail(path, "optimizer equivalence schema/gate mismatch")
    if row["status"] != "passed" or row["semantic_class"] != "E0":
        _fail(path, "optimizer equivalence must be a passed E0 gate")
    _string(row["recorded_at_utc"], f"{path}.recorded_at_utc")
    source = _exact(
        row["source"], {"git_commit", "git_dirty", "archive_sha256"}, f"{path}.source"
    )
    if source != {
        "git_commit": revision,
        "git_dirty": False,
        "archive_sha256": archive_sha256,
    }:
        _fail(f"{path}.source", "does not exactly match candidate source")
    build = _exact(
        row["build"],
        {
            "container_image_sha256", "network", "cargo_locked", "cargo_offline",
            "rustc", "cuda_toolkit", "cuda_architecture",
        },
        f"{path}.build",
    )
    profile_image_sha256 = _sha256(
        build["container_image_sha256"], f"{path}.build.container_image_sha256"
    )
    expected_build = {
        "network": "none",
        "cargo_locked": True,
        "cargo_offline": True,
        "cuda_architecture": "89",
    }
    for key, expected in expected_build.items():
        if build[key] != expected:
            _fail(f"{path}.build.{key}", f"must be {expected!r}")
    _string(build["rustc"], f"{path}.build.rustc")
    _string(build["cuda_toolkit"], f"{path}.build.cuda_toolkit")
    gpu = _exact(
        row["gpu"],
        {"model", "uuid", "pci_bus_id", "compute_capability", "vram_mib", "driver_version"},
        f"{path}.gpu",
    )
    for key in ("model", "uuid", "pci_bus_id", "compute_capability", "driver_version"):
        _string(gpu[key], f"{path}.gpu.{key}")
    if not isinstance(gpu["vram_mib"], int) or isinstance(gpu["vram_mib"], bool) or gpu["vram_mib"] <= 0:
        _fail(f"{path}.gpu.vram_mib", "must be a positive integer")
    model = _exact(
        row["model"],
        {"model_id", "revision", "dtype", "manifest_sha256", "weights_sha256", "tokenizer_sha256"},
        f"{path}.model",
    )
    expected_model = {
        "model_id": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "dtype": "bf16",
        "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
        "tokenizer_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    }
    for key, expected in expected_model.items():
        if model[key] != expected:
            _fail(f"{path}.model.{key}", f"must be {expected!r}")
    _sha256(model["manifest_sha256"], f"{path}.model.manifest_sha256")
    implementations = _exact(
        row["implementations"],
        {"baseline", "candidate", "residual_rmsnorm", "rollback"},
        f"{path}.implementations",
    )
    expected_implementations = {
        "baseline": "per-operation",
        "candidate": "iteration-batch",
        "residual_rmsnorm": "separate",
        "rollback": "--execution-completion per-operation",
    }
    if implementations != expected_implementations:
        _fail(f"{path}.implementations", "runtime flag/rollback contract mismatch")
    tests = row["tests"]
    if not isinstance(tests, list) or len(tests) != len(OPTIMIZATION_LOGS):
        _fail(f"{path}.tests", "exact optimizer test inventory is required")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tests):
        test = _object(raw, f"{path}.tests[{index}]")
        test_id = _string(test.get("id"), f"{path}.tests[{index}].id", ID_RE)
        if test_id in by_id:
            _fail(f"{path}.tests[{index}].id", "duplicate test id")
        by_id[test_id] = test
    if set(by_id) != set(OPTIMIZATION_LOGS):
        _fail(f"{path}.tests", f"test id set mismatch: {sorted(by_id)}")
    declared = {
        "cuda-compile-only": _optimization_test(
            by_id["cuda-compile-only"], f"{path}.tests.cuda-compile-only",
            "cuda-compile-only", {},
        ),
        "workspace-all-features-all-targets": _optimization_test(
            by_id["workspace-all-features-all-targets"],
            f"{path}.tests.workspace-all-features-all-targets",
            "workspace-all-features-all-targets", {},
        ),
        "command-batch-lifecycle": _optimization_test(
            by_id["command-batch-lifecycle"], f"{path}.tests.command-batch-lifecycle",
            "command-batch-lifecycle",
            {"one_shot_finish": True, "drop_restores_stream": True},
        ),
        "command-batch-resource-ledger": _optimization_test(
            by_id["command-batch-resource-ledger"],
            f"{path}.tests.command-batch-resource-ledger",
            "command-batch-resource-ledger",
            {
                "validation_fail_closed": True,
                "queued_chain_raw_byte_mismatches": 0,
                "hot_loop_allocation_delta": 0,
                "stream_reuse_after_finish": True,
                "owner_close_allocation_count": 0,
            },
        ),
        "smollm2-multi-step-greedy-exact": _optimization_test(
            by_id["smollm2-multi-step-greedy-exact"],
            f"{path}.tests.smollm2-multi-step-greedy-exact",
            "smollm2-multi-step-greedy-exact",
            {
                "decode_steps": 16,
                "committed_iterations": 16,
                "raw_logit_mismatches": 0,
                "generated_token_ids": EXPECTED_OPTIMIZATION_TOKENS,
                "token_id_mismatches": 0,
                "hot_loop_allocation_delta": 0,
                "owner_close_allocation_count": 0,
            },
        ),
    }
    actual = _optimization_log_hashes(raw_evidence_path)
    if declared != actual:
        _fail(f"{path}.tests", "declared log hashes do not match exact raw log inventory")
    return profile_image_sha256


def _validate_performance(
    report: dict[str, Any],
    path: str,
    *,
    revision: str,
    archive_sha256: str,
    binary_sha256: str,
    image_sha256: str,
    optimization_sha256: str,
    optimization_gate_id: str,
    optimization_profile_image_sha256: str,
    raw_evidence_path: Path,
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
        {
            "candidate_id", "recorded_at_utc", "source", "model",
            "environment", "workload", "metrics", "run_summary", "raw_runs",
        },
        f"{path}.candidate",
    )
    try:
        validated_candidate = release_performance._validate_candidate(
            {
                "schema_version": release_performance.CANDIDATE_SCHEMA,
                "baseline_sha256": release_performance.BASELINE_SHA256,
                "status": "success",
                **candidate,
            }
        )
    except (release_performance.InputError, release_performance.ComparabilityError) as error:
        _fail(f"{path}.candidate", str(error))
    binding = _exact(
        validated_candidate["source"],
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
        "correctness_report_sha256": optimization_sha256,
        "correctness_gate_id": optimization_gate_id,
        "profile_image_sha256": optimization_profile_image_sha256,
        "semantic_class": "E0",
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            _fail(f"{path}.candidate.source.{key}", "candidate binding mismatch")
    _sha256(binding["profile_binary_sha256"], f"{path}.candidate.source.profile_binary_sha256")

    baseline = _reviewed_performance_baseline()
    declared_baseline = _exact(
        row["baseline"], {"baseline_id", "sha256", "metrics"}, f"{path}.baseline"
    )
    expected_baseline = {
        "baseline_id": baseline["baseline_id"],
        "sha256": release_performance.BASELINE_SHA256,
        "metrics": baseline["metrics"],
    }
    if declared_baseline != expected_baseline:
        _fail(f"{path}.baseline", "does not equal the reviewed baseline")
    for field in ("model", "environment", "workload"):
        if validated_candidate[field] != baseline[field]:
            _fail(
                f"{path}.candidate.{field}",
                "differs from the reviewed release lane",
            )

    payloads = _performance_raw_payloads(raw_evidence_path)
    try:
        release_performance.validate_raw_run_payloads(
            payloads, validated_candidate
        )
    except (release_performance.InputError, release_performance.ComparabilityError) as error:
        _fail(f"{path}.raw_evidence", str(error))

    ratios = _exact(
        row["ratios"], set(release_performance.METRIC_FIELDS), f"{path}.ratios"
    )
    expected_ratios: dict[str, float] = {}
    for metric in release_performance.METRIC_FIELDS:
        expected_ratios[metric] = (
            validated_candidate["metrics"][metric] / baseline["metrics"][metric]
        )
        observed = _finite_number(ratios[metric], f"{path}.ratios.{metric}", minimum=0)
        if observed != expected_ratios[metric]:
            _fail(f"{path}.ratios.{metric}", "does not equal the raw-derived ratio")

    checks = row["checks"]
    if not isinstance(checks, list) or len(checks) != len(PERFORMANCE_CHECKS):
        _fail(f"{path}.checks", "exact four-check inventory is required")
    by_name: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(checks):
        check_path = f"{path}.checks[{index}]"
        check = _exact(
            raw, {"name", "passed", "observed", "operator", "limit"}, check_path
        )
        name = _string(check["name"], f"{check_path}.name", ID_RE)
        if name in by_name:
            _fail(f"{check_path}.name", "duplicate performance check")
        by_name[name] = check
    if set(by_name) != set(PERFORMANCE_CHECKS):
        _fail(f"{path}.checks", f"performance check set mismatch: {sorted(by_name)}")
    for name, (metric, operator, limit) in PERFORMANCE_CHECKS.items():
        check = by_name[name]
        observed = _finite_number(
            check["observed"], f"{path}.checks.{name}.observed", minimum=0
        )
        if observed != expected_ratios[metric]:
            _fail(f"{path}.checks.{name}.observed", "raw-derived ratio mismatch")
        if check["operator"] != operator or check["limit"] != limit:
            _fail(f"{path}.checks.{name}", "reviewed threshold contract mismatch")
        passed = observed <= limit if operator == "<=" else observed >= limit
        if check["passed"] is not True or not passed:
            _fail(f"{path}.checks.{name}.passed", "performance threshold did not pass")


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
        row["bindings"],
        {
            "contract_id",
            "reviewed_manifest_template_canonical_sha256",
            "manifest_sha256",
            "binding_sha256",
            "source",
        },
        f"{path}.bindings",
    )
    if bindings["contract_id"] != SOAK_CONTRACT_ID:
        _fail(f"{path}.bindings.contract_id", "reviewed soak contract mismatch")
    if (
        bindings["reviewed_manifest_template_canonical_sha256"]
        != SOAK_TEMPLATE_CANONICAL_SHA256
    ):
        _fail(
            f"{path}.bindings.reviewed_manifest_template_canonical_sha256",
            "reviewed soak manifest digest mismatch",
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

    summaries = row["scenario_summaries"]
    if not isinstance(summaries, list) or len(summaries) != len(SOAK_SCENARIOS):
        _fail(f"{path}.scenario_summaries", "exact reviewed scenario inventory required")
    observed_scenarios: set[str] = set()
    for index, value in enumerate(summaries):
        summary_path = f"{path}.scenario_summaries[{index}]"
        summary = _exact(
            value,
            {
                "scenario_id",
                "kind",
                "events",
                "samples",
                "requests",
                "maximum_sample_gap_ms",
                "observed_duration_seconds",
                "sample_span_seconds",
                "rss_slope_bytes_per_hour",
                "vram_slope_bytes_per_hour",
            },
            summary_path,
        )
        scenario_id = _string(summary["scenario_id"], f"{summary_path}.scenario_id")
        if scenario_id in observed_scenarios or scenario_id not in SOAK_SCENARIOS:
            _fail(f"{summary_path}.scenario_id", "duplicate or unreviewed scenario")
        observed_scenarios.add(scenario_id)
        expected_kind, expected_duration = SOAK_SCENARIOS[scenario_id]
        if summary["kind"] != expected_kind:
            _fail(f"{summary_path}.kind", "reviewed scenario kind mismatch")
        for field in ("events", "samples", "requests"):
            value = summary[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                _fail(f"{summary_path}.{field}", "must be a positive integer")
        maximum_gap = _finite_number(
            summary["maximum_sample_gap_ms"],
            f"{summary_path}.maximum_sample_gap_ms",
            minimum=0,
        )
        if maximum_gap > 2_500:
            _fail(f"{summary_path}.maximum_sample_gap_ms", "exceeds reviewed bound")
        observed_duration = _finite_number(
            summary["observed_duration_seconds"],
            f"{summary_path}.observed_duration_seconds",
            minimum=0,
        )
        if observed_duration < expected_duration:
            _fail(f"{summary_path}.observed_duration_seconds", "scenario was truncated")
        sample_span = _finite_number(
            summary["sample_span_seconds"],
            f"{summary_path}.sample_span_seconds",
            minimum=0,
        )
        if sample_span < expected_duration - 2.5:
            _fail(f"{summary_path}.sample_span_seconds", "samples do not span scenario")
        for field in ("rss_slope_bytes_per_hour", "vram_slope_bytes_per_hour"):
            slope = _finite_number(summary[field], f"{summary_path}.{field}")
            if slope > 33_554_432:
                _fail(f"{summary_path}.{field}", "exceeds reviewed slope bound")
    if observed_scenarios != set(SOAK_SCENARIOS):
        _fail(f"{path}.scenario_summaries", "reviewed scenario set mismatch")

    expected_checks = set(SOAK_GLOBAL_CHECKS)
    for scenario_id in SOAK_SCENARIOS:
        expected_checks.update(
            f"{scenario_id}.{suffix}" for suffix in SOAK_COMMON_SCENARIO_CHECKS
        )
        if scenario_id in SOAK_GOLDEN_SCENARIOS:
            expected_checks.add(f"{scenario_id}.golden_parity")
    checks = row["checks"]
    if not isinstance(checks, list):
        _fail(f"{path}.checks", "must be an array")
    observed_checks: set[str] = set()
    for index, value in enumerate(checks):
        check_path = f"{path}.checks[{index}]"
        check = _exact(value, {"name", "passed", "observed", "threshold"}, check_path)
        name = _string(check["name"], f"{check_path}.name")
        if name in observed_checks:
            _fail(f"{check_path}.name", "duplicate check")
        observed_checks.add(name)
        if check["passed"] is not True:
            _fail(f"{check_path}.passed", "must be true")
    if observed_checks != expected_checks:
        _fail(f"{path}.checks", "exact reviewed soak check inventory mismatch")

    observations = _exact(
        row["observations"],
        {"event_count", "outcome_counts", "service_counter_maxima", "final"},
        f"{path}.observations",
    )
    event_count = observations["event_count"]
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1:
        _fail(f"{path}.observations.event_count", "must be a positive integer")
    for field, minimums in (
        (
            "outcome_counts",
            {"cancelled": 100, "disconnected": 100, "overload": 20},
        ),
        (
            "service_counter_maxima",
            {"cancellations": 100, "disconnects": 100, "overloads": 20},
        ),
    ):
        counters = _object(observations[field], f"{path}.observations.{field}")
        for name, minimum in minimums.items():
            value = counters.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                _fail(
                    f"{path}.observations.{field}.{name}",
                    f"must be an integer >= {minimum}",
                )
    final = _exact(
        observations["final"],
        {
            "active_requests",
            "waiting_requests",
            "kv_allocated_blocks",
            "device_live_count",
            "device_live_bytes",
            "pinned_live_count",
            "pinned_live_bytes",
        },
        f"{path}.observations.final",
    )
    if any(value != 0 for value in final.values()):
        _fail(f"{path}.observations.final", "all final resource values must be zero")


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
            {
                "python_free_e2e", "cuda_fault", "native_correctness",
                "optimization_correctness", "performance", "reliability_soak",
            },
            "manifest.evidence",
        )
        seen_paths: set[str] = set()
        archive_path, archive_sha256, _ = _resolve_artifact(
            source_row["archive"], "manifest.source.archive", evidence_root, seen_paths
        )
        _verify_source_archive(archive_path, revision)
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
        raw_paths: dict[str, Path] = {}
        for gate_name in ("python_free_e2e", "cuda_fault"):
            gate = _exact(evidence_row[gate_name], {"report", "raw_evidence"}, f"manifest.evidence.{gate_name}")
            report_path, report_sha, _ = _resolve_artifact(
                gate["report"], f"manifest.evidence.{gate_name}.report", evidence_root, seen_paths
            )
            raw_path, raw_sha, _ = _resolve_artifact(
                gate["raw_evidence"], f"manifest.evidence.{gate_name}.raw_evidence", evidence_root, seen_paths
            )
            gate_report, _ = _load_json(report_path, f"{gate_name} report")
            loaded[gate_name] = (gate_report, report_sha)
            raw_hashes[gate_name] = raw_sha
            raw_paths[gate_name] = raw_path

        performance = _exact(
            evidence_row["performance"],
            {"report", "raw_evidence"},
            "manifest.evidence.performance",
        )
        performance_report_path, performance_report_sha, _ = _resolve_artifact(
            performance["report"],
            "manifest.evidence.performance.report",
            evidence_root,
            seen_paths,
        )
        performance_raw_path, performance_raw_sha, _ = _resolve_artifact(
            performance["raw_evidence"],
            "manifest.evidence.performance.raw_evidence",
            evidence_root,
            seen_paths,
        )
        performance_report, _ = _load_json(
            performance_report_path, "performance report"
        )
        loaded["performance"] = (performance_report, performance_report_sha)
        raw_hashes["performance"] = performance_raw_sha
        raw_paths["performance"] = performance_raw_path

        for gate_name in ("reliability_soak",):
            gate = _exact(evidence_row[gate_name], {"report"}, f"manifest.evidence.{gate_name}")
            report_path, report_sha, _ = _resolve_artifact(
                gate["report"], f"manifest.evidence.{gate_name}.report", evidence_root, seen_paths
            )
            gate_report, _ = _load_json(report_path, f"{gate_name} report")
            loaded[gate_name] = (gate_report, report_sha)

        native = _exact(
            evidence_row["native_correctness"],
            {"report", "raw_replay", "replay_validation"},
            "manifest.evidence.native_correctness",
        )
        for field in ("report", "raw_replay", "replay_validation"):
            native_path, native_sha, _ = _resolve_artifact(
                native[field], f"manifest.evidence.native_correctness.{field}",
                evidence_root, seen_paths,
            )
            if field == "raw_replay":
                raw_hashes["native_correctness"] = native_sha
                _verify_native_replay_archive(native_path)
            else:
                native_document, _ = _load_json(native_path, f"native_correctness {field}")
                loaded[f"native_correctness_{field}"] = (native_document, native_sha)

        optimization = _exact(
            evidence_row["optimization_correctness"],
            {"report", "raw_evidence"},
            "manifest.evidence.optimization_correctness",
        )
        optimization_report_path, optimization_report_sha, _ = _resolve_artifact(
            optimization["report"], "manifest.evidence.optimization_correctness.report",
            evidence_root, seen_paths,
        )
        optimization_raw_path, optimization_raw_sha, _ = _resolve_artifact(
            optimization["raw_evidence"],
            "manifest.evidence.optimization_correctness.raw_evidence",
            evidence_root, seen_paths,
        )
        optimization_report, _ = _load_json(
            optimization_report_path, "optimization_correctness report"
        )
        loaded["optimization_correctness"] = (
            optimization_report, optimization_report_sha
        )
        raw_hashes["optimization_correctness"] = optimization_raw_sha

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
        native_correctness_sha256 = loaded["native_correctness_report"][1]
        _validate_correctness(
            loaded["native_correctness_report"][0], "native_correctness", revision
        )
        _validate_native_replay(
            loaded["native_correctness_replay_validation"][0],
            "native_correctness.replay_validation",
            revision=revision,
            archive_sha256=archive_sha256,
            correctness_sha256=native_correctness_sha256,
            raw_replay_sha256=raw_hashes["native_correctness"],
        )
        optimization_profile_image_sha256 = _validate_optimization_correctness(
            loaded["optimization_correctness"][0],
            "optimization_correctness",
            revision=revision,
            archive_sha256=archive_sha256,
            raw_evidence_path=optimization_raw_path,
        )
        _validate_python_free_e2e_replay(
            loaded["python_free_e2e"][0],
            raw_paths["python_free_e2e"],
            revision=revision,
            archive_sha256=archive_sha256,
            binary_sha256=binary_sha256,
            bundle_sha256=bundle_sha256,
            image_sha256=image_sha256,
            native_correctness=loaded["native_correctness_report"][0],
            native_correctness_sha256=native_correctness_sha256,
            optimization_correctness=loaded["optimization_correctness"][0],
        )
        _validate_performance(
            loaded["performance"][0], "performance", revision=revision,
            archive_sha256=archive_sha256, binary_sha256=binary_sha256,
            image_sha256=image_sha256,
            optimization_sha256=optimization_report_sha,
            optimization_gate_id=OPTIMIZATION_GATE,
            optimization_profile_image_sha256=optimization_profile_image_sha256,
            raw_evidence_path=raw_paths["performance"],
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
                        "native_correctness", "optimization_correctness",
                        "performance", "reliability_soak", "cross_bindings",
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
