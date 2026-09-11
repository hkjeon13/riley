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
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

import check_cuda_fault_evidence as cuda_fault_evidence
import check_native_correctness_evidence as native_correctness_evidence
import check_optimization_evidence as optimization_evidence
import check_reproducible_build as reproducible_build_evidence
import optimizer_e0_semantic_contract as optimizer_contract
from release_common import ReleaseContractError, canonical_json_bytes
from verify_release_bundle import verify_bundle


_PERFORMANCE_CHECKER_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/scripts/check_release_performance.py"
)
_PERFORMANCE_SPEC = importlib.util.spec_from_file_location(
    "riley_final_candidate_performance_contract",
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
    "riley_final_candidate_python_free_e2e_contract",
    _PYTHON_FREE_E2E_PATH,
)
if _PYTHON_FREE_E2E_SPEC is None or _PYTHON_FREE_E2E_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load Python-free E2E contract: {_PYTHON_FREE_E2E_PATH}")
python_free_e2e = importlib.util.module_from_spec(_PYTHON_FREE_E2E_SPEC)
sys.modules[_PYTHON_FREE_E2E_SPEC.name] = python_free_e2e
_PYTHON_FREE_E2E_SPEC.loader.exec_module(python_free_e2e)

_RELIABILITY_SOAK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/scripts/check_reliability_soak.py"
)
_RELIABILITY_SOAK_SPEC = importlib.util.spec_from_file_location(
    "riley_final_candidate_reliability_soak_contract",
    _RELIABILITY_SOAK_PATH,
)
if _RELIABILITY_SOAK_SPEC is None or _RELIABILITY_SOAK_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load reliability soak contract: {_RELIABILITY_SOAK_PATH}")
reliability_soak = importlib.util.module_from_spec(_RELIABILITY_SOAK_SPEC)
sys.modules[_RELIABILITY_SOAK_SPEC.name] = reliability_soak
_RELIABILITY_SOAK_SPEC.loader.exec_module(reliability_soak)


MANIFEST_VERSION = "riley.release-candidate-manifest.v2"
ATTESTATION_VERSION = "riley.release-gate-attestation.v1"
REPORT_VERSION = "riley.release-candidate-report.v2"
PERFORMANCE_VERSION = "riley.release-performance-report.v1"
SOAK_VERSION = "riley.reliability-soak-report.v2"
CORRECTNESS_VERSION = "1.0.0"
CORRECTNESS_GATE = "smollm2-fp32-bf16-native-e0-v3"
OPTIMIZATION_GATE = "pr15-iteration-command-batch-exact-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
CANDIDATE_ID_RE = re.compile(
    r"^riley-((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*))-rc([1-9][0-9]*)$"
)
PLACEHOLDER_RE = re.compile(
    r"(?:placeholder|replace[-_ ]?me|sha256[-_ ]?of|\btodo\b|<[^>]+>)",
    re.IGNORECASE,
)
MAX_JSON_BYTES = 64 * 1024 * 1024
PERFORMANCE_BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/release/performance-baseline-v1.json"
)
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
OPTIMIZATION_LOGS = optimization_evidence.LOG_FILES
SOAK_CONTRACT_ID = "pr16-release-soak-v1"
SOAK_TEMPLATE_CANONICAL_SHA256 = (
    "ef8d50d07aba2e7b8c0c3f3f157bf242452ac62be9dc22080baff8023278e0f3"
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


@dataclass(frozen=True)
class _ArtifactSnapshotContext:
    directory: Path
    evidence_root_fd: int
    seen_file_ids: set[tuple[int, int]]


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_json(
    path: Path,
    label: str,
    *,
    file_fd: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    try:
        if file_fd is None:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                _fail(label, "must be a regular file, not a link or device")
            if metadata.st_size > MAX_JSON_BYTES:
                _fail(label, f"exceeds the {MAX_JSON_BYTES}-byte JSON bound")
            raw = path.read_bytes()
        else:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                _fail(label, "held FD must be a regular file")
            named_fd = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK,
            )
            try:
                named = os.fstat(named_fd)
                if named.st_dev != before.st_dev or named.st_ino != before.st_ino:
                    _fail(label, "manifest path does not name the held FD")
            finally:
                os.close(named_fd)
            if before.st_size > MAX_JSON_BYTES:
                _fail(label, f"exceeds the {MAX_JSON_BYTES}-byte JSON bound")
            raw = os.pread(file_fd, MAX_JSON_BYTES + 1, 0)
            after = os.fstat(file_fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if (
                len(raw) != before.st_size
                or len(raw) > MAX_JSON_BYTES
                or any(
                    getattr(before, field) != getattr(after, field)
                    for field in stable_fields
                )
            ):
                _fail(label, "held FD changed while it was read")
        text = raw.decode("utf-8")
        value = json.loads(
            text,
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


def _image_id(value: Any, path: str) -> str:
    image_id = _string(value, path)
    if not image_id.startswith("sha256:"):
        _fail(path, "must be sha256:<lowercase digest>")
    _sha256(image_id.removeprefix("sha256:"), path)
    return image_id


def _candidate_id(value: Any, path: str) -> tuple[str, str]:
    candidate_id = _string(value, path)
    match = CANDIDATE_ID_RE.fullmatch(candidate_id)
    if match is None:
        _fail(path, "must be riley-<major>.<minor>.<patch>-rc<positive integer>")
    return candidate_id, match.group(1)


def _source_date_epoch(value: Any, path: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 0xFFFFFFFF
    ):
        _fail(path, "must fit an unsigned 32-bit timestamp")
    return value


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        _fail(path, "is outside the finite reviewed range")
    return result


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
    snapshot_context: _ArtifactSnapshotContext,
) -> tuple[Path, str, str]:
    artifact = _exact(value, {"path", "sha256"}, path)
    relative = _string(artifact["path"], f"{path}.path")
    if "\x00" in relative or "\\" in relative or "//" in relative:
        _fail(f"{path}.path", "must use a normalized POSIX relative path")
    pure = PurePosixPath(relative)
    normalized = pure.as_posix()
    if (
        not pure.parts
        or relative != normalized
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"{path}.path", "path traversal and absolute paths are forbidden")
    if normalized in seen_paths:
        _fail(f"{path}.path", "artifact path is duplicated")
    seen_paths.add(normalized)
    candidate = evidence_root.joinpath(*pure.parts)
    declared = _sha256(artifact["sha256"], f"{path}.sha256")
    directory_fd = snapshot_context.evidence_root_fd
    file_fd = -1
    try:
        directory_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        for part in pure.parts[:-1]:
            next_fd = os.open(
                part,
                directory_flags,
                dir_fd=directory_fd,
            )
            if directory_fd != snapshot_context.evidence_root_fd:
                os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            pure.parts[-1],
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            _fail(
                f"{path}.path",
                "artifact must be a regular file, not a link or device",
            )
        file_id = (before.st_dev, before.st_ino)
        if file_id in snapshot_context.seen_file_ids:
            _fail(
                f"{path}.path",
                "artifact must not be a hard-link alias of another manifest path",
            )
        snapshot_context.seen_file_ids.add(file_id)
        snapshot = (
            snapshot_context.directory / f"{len(seen_paths):03d}-{pure.name}"
        )
        digest = hashlib.sha256()
        with os.fdopen(os.dup(file_fd), "rb") as source, snapshot.open("xb") as output:
            while block := source.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        after = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            _fail(f"{path}.path", "artifact changed while it was snapshotted")
        snapshot.chmod(stat.S_IMODE(before.st_mode) & 0o777)
        actual = digest.hexdigest()
    except CandidateError:
        raise
    except (OSError, ValueError) as error:
        if candidate.is_symlink():
            _fail(f"{path}.path", "symlink path components are forbidden")
        _fail(f"{path}.path", f"cannot open and snapshot artifact: {error}")
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if (
            directory_fd >= 0
            and directory_fd != snapshot_context.evidence_root_fd
        ):
            os.close(directory_fd)
    if actual != declared:
        _fail(f"{path}.sha256", f"artifact digest mismatch: {actual}")
    return snapshot, declared, normalized


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
    correctness_golden_sha256: str,
) -> dict[str, Any]:
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
        correctness_golden_sha256=correctness_golden_sha256,
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
        "model_tree_sha256": (
            raw_model.get("model_tree_sha256"),
            optimizer_model.get("manifest_sha256"),
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
    return raw_model


def _validate_cuda_fault_replay(
    report: dict[str, Any],
    raw_evidence_path: Path,
    *,
    revision: str,
    source_archive: Path,
    build_image_id: str,
    release_binary: Path,
    release_bundle: Path,
    release_image_id: str,
) -> None:
    try:
        replayed = cuda_fault_evidence.replay_raw_evidence(
            raw_evidence_path,
            source_revision=revision,
            source_archive=source_archive,
            build_image_id=build_image_id,
            release_binary=release_binary,
            release_bundle=release_bundle,
            release_image_id=release_image_id,
        )
    except (cuda_fault_evidence.CudaFaultEvidenceError, OSError) as error:
        _fail("cuda_fault.raw_evidence", str(error))
    if replayed != report:
        _fail("cuda_fault", "submitted attestation differs from raw replay")


def _validate_correctness(
    report: dict[str, Any], path: str, revision: str, candidate_executable_sha256: str
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
        _fail(path, "must be the reviewed native E0 correctness report v3")
    if row["status"] != "pass":
        _fail(f"{path}.status", "must be pass")
    bindings = _object(row["bindings"], f"{path}.bindings")
    if bindings.get("candidate_git_revision") != revision:
        _fail(f"{path}.bindings.candidate_git_revision", "source revision mismatch")
    if bindings.get("candidate_git_status_sha256") != hashlib.sha256(b"").hexdigest():
        _fail(f"{path}.bindings.candidate_git_status_sha256", "source tree was not clean")
    if (
        _sha256(
            bindings.get("candidate_executable_sha256"),
            f"{path}.bindings.candidate_executable_sha256",
        )
        != candidate_executable_sha256
    ):
        _fail(
            f"{path}.bindings.candidate_executable_sha256",
            "does not match the replayed native executable artifact",
        )
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
        "candidate_variant_count": 1,
        "failure_count": 0,
        "numeric_pass": True,
        "semantic_pass": True,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            _fail(f"{path}.summary.{key}", f"must be {expected!r}")
    variants = _object(summary.get("variants"), f"{path}.summary.variants")
    if set(variants) != {"canonical-v1"}:
        _fail(f"{path}.summary.variants", "required release variant set mismatch")
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
            {"canonical-v1"},
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


def _performance_raw_replay(
    path: Path,
) -> tuple[list[tuple[str, bytes]], dict[str, Any], str]:
    evidence_path = "performance.raw_evidence"
    try:
        replay = release_performance.replay_raw_evidence_archive(path)
    except (release_performance.InputError, OSError) as error:
        _fail(evidence_path, str(error))
    if not isinstance(replay, dict):
        _fail(evidence_path, "raw replay must return an object")
    payloads = replay.get("payloads")
    runner_manifest = replay.get("runner_manifest")
    archive_sha256 = replay.get("archive_sha256")
    if (
        not isinstance(payloads, list)
        or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], bytes)
            for item in payloads
        )
        or not isinstance(runner_manifest, dict)
    ):
        _fail(evidence_path, "raw replay returned malformed payload/manifest bindings")
    archive_sha256 = _sha256(
        archive_sha256, f"{evidence_path}.archive_sha256"
    )
    return (
        [(f"{evidence_path}:{name}", raw) for name, raw in payloads],
        runner_manifest,
        archive_sha256,
    )


def _reviewed_performance_baseline() -> dict[str, Any]:
    try:
        document, raw = release_performance._load_json_bytes(
            PERFORMANCE_BASELINE_PATH, "reviewed performance baseline"
        )
        return release_performance._validate_baseline(document, raw)
    except (release_performance.InputError, OSError) as error:
        _fail("performance.baseline", str(error))


def _validate_optimization_correctness(
    report: dict[str, Any],
    path: str,
    *,
    revision: str,
    archive_sha256: str,
) -> str:
    """Apply the shared closed optimizer E0 report contract."""

    try:
        return optimizer_contract.validate_final_candidate_report(
            report,
            source_revision=revision,
            source_archive_sha256=archive_sha256,
        )
    except optimizer_contract.OptimizerE0SemanticContractError as error:
        _fail(path, str(error))


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
    optimization_correctness: dict[str, Any],
    profile_binary_sha256: str,
    raw_evidence_path: Path,
    raw_evidence_sha256: str,
    candidate_id: str,
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
    if validated_candidate["candidate_id"] != candidate_id:
        _fail(
            f"{path}.candidate.candidate_id",
            "does not match the trusted final candidate ID",
        )
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
        "profile_binary_sha256": profile_binary_sha256,
        "semantic_class": "E0",
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            _fail(f"{path}.candidate.source.{key}", "candidate binding mismatch")
    _sha256(
        binding["profile_binary_sha256"],
        f"{path}.candidate.source.profile_binary_sha256",
    )

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

    payloads, runner_manifest, replayed_archive_sha256 = _performance_raw_replay(
        raw_evidence_path
    )
    if replayed_archive_sha256 != raw_evidence_sha256:
        _fail(
            f"{path}.raw_evidence.archive_sha256",
            "raw replay digest does not match the final candidate artifact binding",
        )
    try:
        request_identity = release_performance.derive_raw_run_payloads(payloads)
        release_performance._require_request_identity_sha256(
            request_identity,
            baseline["request_identity_sha256"],
            f"{path}.raw_evidence.request_identity_sha256",
        )
    except (
        release_performance.InputError,
        release_performance.ComparabilityError,
    ) as error:
        _fail(f"{path}.raw_evidence.request_identity_sha256", str(error))
    optimization_model = _object(
        optimization_correctness.get("model"),
        "optimization_correctness.model",
    )
    optimizer_model_tree_sha256 = _sha256(
        optimization_model.get("manifest_sha256"),
        "optimization_correctness.model.manifest_sha256",
    )
    runner_candidate = _exact(
        runner_manifest.get("candidate"),
        {
            "source_revision",
            "source_archive_sha256",
            "profile_binary_sha256",
            "model_tree_sha256",
            "optimizer_correctness_report_sha256",
            "optimizer_image_id",
        },
        f"{path}.raw_evidence.runner_manifest.candidate",
    )
    expected_runner_candidate = {
        "source_revision": revision,
        "source_archive_sha256": archive_sha256,
        "profile_binary_sha256": profile_binary_sha256,
        "model_tree_sha256": optimizer_model_tree_sha256,
        "optimizer_correctness_report_sha256": optimization_sha256,
        "optimizer_image_id": f"sha256:{optimization_profile_image_sha256}",
    }
    if runner_candidate != expected_runner_candidate:
        _fail(
            f"{path}.raw_evidence.runner_manifest.candidate",
            "does not exactly bind the final candidate and submitted optimizer model manifest",
        )
    runner = _object(
        runner_manifest.get("runner"),
        f"{path}.raw_evidence.runner_manifest.runner",
    )
    if runner.get("tools") != release_performance.RUNNER_REVIEWED_TOOLS:
        _fail(
            f"{path}.raw_evidence.runner_manifest.runner.tools",
            "does not equal the reviewed server-4096 tool map",
        )
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
    model: dict[str, Any],
    raw_evidence_path: Path,
    raw_evidence_sha256: str,
    correctness_golden_path: Path,
    correctness_golden_sha256: str,
    generated_text_sha256: str,
    native_correctness_report_path: Path,
    native_correctness_report_sha256: str,
) -> None:
    try:
        replay = reliability_soak.replay_raw_evidence_archive(
            raw_evidence_path,
            correctness_golden=correctness_golden_path,
            native_correctness_report=native_correctness_report_path,
        )
    except (reliability_soak.InputError, OSError) as error:
        _fail(f"{path}.raw_evidence", str(error))
    if not isinstance(replay, dict) or "report" not in replay:
        _fail(f"{path}.raw_evidence", "raw replay returned no report binding")
    replayed_report = replay["report"]
    if _canonical_json_bytes(replayed_report) != _canonical_json_bytes(report):
        _fail(path, "submitted report differs from the raw-replayed report")
    expected_replay_keys = {
        "report",
        "archive_sha256",
        *{
            f"{filename}_sha256"
            for filename in reliability_soak.RAW_ARCHIVE_PAYLOADS
        },
    }
    if set(replay) != expected_replay_keys:
        _fail(
            f"{path}.raw_evidence",
            "raw replay returned a non-canonical binding inventory",
        )
    for field in expected_replay_keys - {"report"}:
        _sha256(replay[field], f"{path}.raw_evidence.{field}")
    if replay["archive_sha256"] != raw_evidence_sha256:
        _fail(
            f"{path}.raw_evidence.archive_sha256",
            "does not match the snapshotted manifest raw-evidence artifact",
        )
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
            "trusted_correctness",
            "runtime_provenance",
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
    trusted_correctness = _exact(
        bindings["trusted_correctness"],
        {
            "correctness_gate_id",
            "e2e_correctness_golden_sha256",
            "generated_text_sha256",
            "native_correctness_report_sha256",
        },
        f"{path}.bindings.trusted_correctness",
    )
    expected_trusted_correctness = {
        "correctness_gate_id": CORRECTNESS_GATE,
        "e2e_correctness_golden_sha256": correctness_golden_sha256,
        "generated_text_sha256": generated_text_sha256,
        "native_correctness_report_sha256": native_correctness_report_sha256,
    }
    if trusted_correctness != expected_trusted_correctness:
        _fail(
            f"{path}.bindings.trusted_correctness",
            "does not match the submitted E2E golden and native correctness report",
        )
    runtime_provenance = _exact(
        bindings["runtime_provenance"],
        {
            "host_gpu_sha256",
            "launcher_receipt_sha256",
            "release_image_inspect_sha256",
            "test_layer_image_inspect_sha256",
            "container_inspect_pre_sha256",
            "container_inspect_post_sha256",
            "release_runtime_closure_sha256",
            "run_json_sha256",
            "events_jsonl_sha256",
            "hostname",
            "gpu_uuid",
            "release_image_id",
            "test_layer_image_id",
            "container_id",
            "container_name",
        },
        f"{path}.bindings.runtime_provenance",
    )
    receipt_hashes = {
        "host_gpu_sha256": "host-gpu.csv",
        "launcher_receipt_sha256": "launcher-receipt.json",
        "release_image_inspect_sha256": "release-image-inspect.json",
        "test_layer_image_inspect_sha256": "test-layer-image-inspect.json",
        "container_inspect_pre_sha256": "container-inspect-pre.json",
        "container_inspect_post_sha256": "container-inspect-post.json",
        "release_runtime_closure_sha256": "release-runtime-closure.tsv",
    }
    for field, filename in receipt_hashes.items():
        receipt_sha256 = _sha256(
            runtime_provenance[field],
            f"{path}.bindings.runtime_provenance.{field}",
        )
        if replay.get(f"{filename}_sha256") != receipt_sha256:
            _fail(
                f"{path}.bindings.runtime_provenance.{field}",
                f"does not hash replayed {filename}",
            )
    raw_stream_hashes = {
        "run_json_sha256": "run.json",
        "events_jsonl_sha256": "events.jsonl",
    }
    for field, filename in raw_stream_hashes.items():
        raw_sha256 = _sha256(
            runtime_provenance[field],
            f"{path}.bindings.runtime_provenance.{field}",
        )
        if replay.get(f"{filename}_sha256") != raw_sha256:
            _fail(
                f"{path}.bindings.runtime_provenance.{field}",
                f"does not hash replayed {filename}",
            )
    if runtime_provenance["hostname"] != reliability_soak.DESIGNATED_HOSTNAME:
        _fail(
            f"{path}.bindings.runtime_provenance.hostname",
            "is not the designated soak host",
        )
    if runtime_provenance["gpu_uuid"] != reliability_soak.DESIGNATED_GPU["gpu_uuid"]:
        _fail(
            f"{path}.bindings.runtime_provenance.gpu_uuid",
            "is not the designated soak GPU",
        )
    release_runtime_image = _image_id(
        runtime_provenance["release_image_id"],
        f"{path}.bindings.runtime_provenance.release_image_id",
    )
    if release_runtime_image != f"sha256:{image_sha256}":
        _fail(
            f"{path}.bindings.runtime_provenance.release_image_id",
            "does not match the candidate release image",
        )
    test_layer_image = _image_id(
        runtime_provenance["test_layer_image_id"],
        f"{path}.bindings.runtime_provenance.test_layer_image_id",
    )
    if test_layer_image == release_runtime_image:
        _fail(
            f"{path}.bindings.runtime_provenance.test_layer_image_id",
            "must be a distinct inspected derivative image",
        )
    _sha256(
        runtime_provenance["container_id"],
        f"{path}.bindings.runtime_provenance.container_id",
    )
    container_name = _string(
        runtime_provenance["container_name"],
        f"{path}.bindings.runtime_provenance.container_name",
    )
    if re.fullmatch(
        rf"riley-soak-{re.escape(revision[:12])}-[0-9]{{8}}T[0-9]{{6}}Z",
        container_name,
    ) is None:
        _fail(
            f"{path}.bindings.runtime_provenance.container_name",
            "does not bind the candidate revision prefix and UTC run stamp",
        )
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
        "model_sha256": model.get("model_tree_sha256"),
        "model_id": model.get("model_id"),
        "model_revision": model.get("model_revision"),
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
            "process_pid",
            "process_rss_bytes",
            "process_hwm_bytes",
            "process_fd_count",
            "process_thread_count",
            "process_children",
            "gpu_vram_bytes",
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
    if final["process_children"] != []:
        _fail(
            f"{path}.observations.final.process_children",
            "must be an empty array",
        )
    for field, value in final.items():
        if field == "process_children":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            _fail(
                f"{path}.observations.final.{field}",
                "must be integer zero",
            )


def _verify_bundle_binding(bundle: Path, binary_sha256: str, revision: str) -> str:
    try:
        verify_bundle(bundle)
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            release_members = [member for member in members if member.name.endswith("/manifest/release.json")]
            binary_members = [member for member in members if member.name.endswith("/bin/riley")]
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
    version = artifact.get("version")
    if not isinstance(version, str) or not version:
        _fail("release manifest.artifact.version", "must be a semantic release version")
    if internal_binary_sha256 != binary_sha256:
        _fail("manifest.release.binary", "standalone binary differs from bundle binary")
    return version


def _validate_reproducibility_replay(
    replay: dict[str, Any],
    *,
    revision: str,
    archive_sha256: str,
    source_date_epoch: int,
    build_image_id: str,
    evidence_a_sha256: str,
    evidence_b_sha256: str,
    binary_sha256: str,
    profile_binary_sha256: str,
    bundle_sha256: str,
    native_manifest_sha256: str,
) -> str:
    row = _exact(
        replay,
        {
            "schema_version",
            "gate_id",
            "status",
            "source",
            "build",
            "evidence",
            "artifacts",
            "comparisons",
        },
        "reproducible_build.replay",
    )
    if (
        row["schema_version"] != reproducible_build_evidence.SCHEMA_VERSION
        or row["gate_id"] != reproducible_build_evidence.GATE_ID
        or row["status"] != "passed"
    ):
        _fail("reproducible_build.replay", "schema/gate/status mismatch")
    if _exact(
        row["source"],
        {"revision", "archive_sha256", "source_date_epoch"},
        "reproducible_build.replay.source",
    ) != {
        "revision": revision,
        "archive_sha256": archive_sha256,
        "source_date_epoch": source_date_epoch,
    }:
        _fail("reproducible_build.replay.source", "trusted source binding mismatch")
    build = _exact(
        row["build"],
        {
            "image_id",
            "image_inspect_sha256",
            "platform",
            "network",
            "cargo_command",
            "profile_cargo_command",
            "rust_toolchain",
            "cuda_toolkit",
            "nvcc_version",
            "cuda_architectures",
            "independent_clean_containers",
        },
        "reproducible_build.replay.build",
    )
    if (
        build["image_id"] != build_image_id
        or build["platform"] != reproducible_build_evidence.PLATFORM
        or build["network"] != "none"
        or build["independent_clean_containers"] != 2
    ):
        _fail("reproducible_build.replay.build", "reviewed build binding mismatch")
    _sha256(
        build["image_inspect_sha256"],
        "reproducible_build.replay.build.image_inspect_sha256",
    )
    evidence = _exact(
        row["evidence"],
        {
            "a_sha256",
            "b_sha256",
            "a_container_id",
            "b_container_id",
            "a_workspace_volume",
            "b_workspace_volume",
            "a_workspace_source",
            "b_workspace_source",
            "a_started_at",
            "a_finished_at",
            "b_started_at",
            "b_finished_at",
        },
        "reproducible_build.replay.evidence",
    )
    if evidence["a_sha256"] != evidence_a_sha256 or evidence["b_sha256"] != evidence_b_sha256:
        _fail("reproducible_build.replay.evidence", "A/B artifact digest mismatch")
    artifacts = _exact(
        row["artifacts"],
        {
            "binary_sha256",
            "profile_binary_sha256",
            "bundle_sha256",
            "native_manifest_sha256",
        },
        "reproducible_build.replay.artifacts",
    )
    if artifacts != {
        "binary_sha256": binary_sha256,
        "profile_binary_sha256": profile_binary_sha256,
        "bundle_sha256": bundle_sha256,
        "native_manifest_sha256": native_manifest_sha256,
    }:
        _fail("reproducible_build.replay.artifacts", "final artifact binding mismatch")
    comparisons = _exact(
        row["comparisons"],
        {
            "binary_a_b_final_byte_exact",
            "profile_binary_a_b_final_byte_exact",
            "bundle_a_b_final_byte_exact",
            "native_manifest_a_b_final_byte_exact",
            "source_archive_a_b_final_byte_exact",
        },
        "reproducible_build.replay.comparisons",
    )
    if not all(value is True for value in comparisons.values()):
        _fail("reproducible_build.replay.comparisons", "all byte comparisons must pass")
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()


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


def evaluate(
    manifest_path: Path,
    evidence_root: Path,
    *,
    manifest_fd: int | None = None,
    expected_candidate_id: str,
    expected_revision: str,
    expected_source_archive_sha256: str,
    expected_release_image_id: str,
    expected_reproducible_build_image_id: str,
    expected_cuda_build_image_id: str,
    expected_optimization_build_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    """Validate a final tag candidate without executing the release or CUDA."""

    report = _empty_report()
    snapshot_temporary: tempfile.TemporaryDirectory[str] | None = None
    evidence_root_fd = -1
    try:
        trusted_candidate_id, trusted_release_version = _candidate_id(
            expected_candidate_id,
            "--expected-candidate-id",
        )
        trusted_revision = _revision(expected_revision, "--expected-revision")
        trusted_archive_sha256 = _sha256(
            expected_source_archive_sha256,
            "--expected-source-archive-sha256",
        )
        trusted_release_image_id = _image_id(
            expected_release_image_id,
            "--expected-release-image-id",
        )
        trusted_reproducible_build_image_id = _image_id(
            expected_reproducible_build_image_id,
            "--expected-reproducible-build-image-id",
        )
        trusted_cuda_build_image_id = _image_id(
            expected_cuda_build_image_id,
            "--expected-cuda-build-image-id",
        )
        trusted_optimization_build_image_id = _image_id(
            expected_optimization_build_image_id,
            "--expected-optimization-build-image-id",
        )
        trusted_correctness_golden_sha256 = _sha256(
            expected_correctness_golden_sha256,
            "--expected-correctness-golden-sha256",
        )
        evidence_root = evidence_root.resolve(strict=True)
        if not all(
            hasattr(os, flag)
            for flag in (
                "O_CLOEXEC",
                "O_DIRECTORY",
                "O_NOFOLLOW",
                "O_NONBLOCK",
            )
        ) or os.open not in getattr(os, "supports_dir_fd", set()):
            _fail("--evidence-root", "platform lacks required no-follow open flags")
        root_before = evidence_root.lstat()
        if not stat.S_ISDIR(root_before.st_mode):
            _fail("--evidence-root", "must be a directory")
        evidence_root_fd = os.open(
            evidence_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        root_after = os.fstat(evidence_root_fd)
        if (
            not stat.S_ISDIR(root_after.st_mode)
            or root_before.st_dev != root_after.st_dev
            or root_before.st_ino != root_after.st_ino
        ):
            _fail("--evidence-root", "directory changed while it was opened")
        snapshot_temporary = tempfile.TemporaryDirectory(
            prefix="riley-candidate-snapshot-"
        )
        snapshot_root = _ArtifactSnapshotContext(
            directory=Path(snapshot_temporary.name),
            evidence_root_fd=evidence_root_fd,
            seen_file_ids=set(),
        )
        manifest, manifest_raw = _load_json(
            manifest_path,
            "manifest",
            file_fd=manifest_fd,
        )
        row = _exact(
            manifest,
            {"schema_version", "candidate_id", "source", "release", "evidence"},
            "manifest",
        )
        if row["schema_version"] != MANIFEST_VERSION:
            _fail("manifest.schema_version", f"must be {MANIFEST_VERSION}")
        candidate_id, candidate_release_version = _candidate_id(
            row["candidate_id"], "manifest.candidate_id"
        )
        if candidate_id != trusted_candidate_id:
            _fail(
                "manifest.candidate_id",
                "differs from the trusted expected candidate ID",
            )
        if candidate_release_version != trusted_release_version:
            _fail("manifest.candidate_id", "release version binding differs")
        source_row = _exact(row["source"], {"git_revision", "git_dirty", "archive"}, "manifest.source")
        revision = _revision(source_row["git_revision"], "manifest.source.git_revision")
        if revision != trusted_revision:
            _fail(
                "manifest.source.git_revision",
                "differs from the trusted expected revision",
            )
        if source_row["git_dirty"] is not False:
            _fail("manifest.source.git_dirty", "release source must be clean")
        release_row = _exact(row["release"], {"binary", "bundle", "image_digest"}, "manifest.release")
        image_digest = _image_id(
            release_row["image_digest"], "manifest.release.image_digest"
        )
        if image_digest != trusted_release_image_id:
            _fail(
                "manifest.release.image_digest",
                "differs from the trusted expected release image ID",
            )
        image_sha256 = image_digest.removeprefix("sha256:")
        evidence_row = _exact(
            row["evidence"],
            {
                "python_free_e2e", "cuda_fault", "native_correctness",
                "optimization_correctness", "performance", "reliability_soak",
                "reproducible_build",
            },
            "manifest.evidence",
        )
        seen_paths: set[str] = set()
        archive_path, archive_sha256, _ = _resolve_artifact(
            source_row["archive"],
            "manifest.source.archive",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        if archive_sha256 != trusted_archive_sha256:
            _fail(
                "manifest.source.archive.sha256",
                "differs from the trusted expected source archive SHA-256",
            )
        _verify_source_archive(archive_path, revision)
        binary_path, binary_sha256, _ = _resolve_artifact(
            release_row["binary"],
            "manifest.release.binary",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        bundle_path, bundle_sha256, _ = _resolve_artifact(
            release_row["bundle"],
            "manifest.release.bundle",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        if not os.access(binary_path, os.X_OK):
            _fail("manifest.release.binary", "must be executable")
        bundle_release_version = _verify_bundle_binding(
            bundle_path, binary_sha256, revision
        )
        if bundle_release_version != trusted_release_version:
            _fail(
                "release manifest.artifact.version",
                "does not match the trusted candidate ID release version",
            )
        source = {"git_revision": revision, "archive_sha256": archive_sha256}
        release = {
            "binary_sha256": binary_sha256,
            "bundle_sha256": bundle_sha256,
            "image_sha256": image_sha256,
        }

        loaded: dict[str, tuple[dict[str, Any], str]] = {}
        raw_hashes: dict[str, str] = {}
        raw_paths: dict[str, Path] = {}

        reproducible = _exact(
            evidence_row["reproducible_build"],
            {
                "build_image_id",
                "source_date_epoch",
                "build_a",
                "build_b",
                "profile_binary",
                "native_manifest",
            },
            "manifest.evidence.reproducible_build",
        )
        reproducible_build_image_id = _image_id(
            reproducible["build_image_id"],
            "manifest.evidence.reproducible_build.build_image_id",
        )
        if reproducible_build_image_id != trusted_reproducible_build_image_id:
            _fail(
                "manifest.evidence.reproducible_build.build_image_id",
                "differs from the trusted expected reproducible-build image ID",
            )
        source_date_epoch = _source_date_epoch(
            reproducible["source_date_epoch"],
            "manifest.evidence.reproducible_build.source_date_epoch",
        )
        reproducible_a_path, reproducible_a_sha, _ = _resolve_artifact(
            reproducible["build_a"],
            "manifest.evidence.reproducible_build.build_a",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        reproducible_b_path, reproducible_b_sha, _ = _resolve_artifact(
            reproducible["build_b"],
            "manifest.evidence.reproducible_build.build_b",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        profile_binary_path, profile_binary_sha, _ = _resolve_artifact(
            reproducible["profile_binary"],
            "manifest.evidence.reproducible_build.profile_binary",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        if not os.access(profile_binary_path, os.X_OK):
            _fail(
                "manifest.evidence.reproducible_build.profile_binary",
                "must be executable",
            )
        native_manifest_path, native_manifest_sha, _ = _resolve_artifact(
            reproducible["native_manifest"],
            "manifest.evidence.reproducible_build.native_manifest",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        raw_hashes["reproducible_build_a"] = reproducible_a_sha
        raw_hashes["reproducible_build_b"] = reproducible_b_sha
        raw_hashes["reproducible_build_native_manifest"] = native_manifest_sha

        for gate_name in ("python_free_e2e",):
            gate = _exact(
                evidence_row[gate_name],
                {"report", "raw_evidence", "correctness_golden"},
                f"manifest.evidence.{gate_name}",
            )
            report_path, report_sha, _ = _resolve_artifact(
                gate["report"],
                f"manifest.evidence.{gate_name}.report",
                evidence_root,
                seen_paths,
                snapshot_root,
            )
            raw_path, raw_sha, _ = _resolve_artifact(
                gate["raw_evidence"],
                f"manifest.evidence.{gate_name}.raw_evidence",
                evidence_root,
                seen_paths,
                snapshot_root,
            )
            correctness_golden_path, correctness_golden_sha, _ = _resolve_artifact(
                gate["correctness_golden"],
                f"manifest.evidence.{gate_name}.correctness_golden",
                evidence_root,
                seen_paths,
                snapshot_root,
            )
            if correctness_golden_sha != trusted_correctness_golden_sha256:
                _fail(
                    f"manifest.evidence.{gate_name}.correctness_golden.sha256",
                    "differs from the trusted expected correctness golden SHA-256",
                )
            correctness_golden_document, _ = _load_json(
                correctness_golden_path,
                f"{gate_name} correctness golden",
            )
            trusted_generated_text_sha256 = _sha256(
                correctness_golden_document.get("expected_greedy_text_sha256"),
                (
                    f"manifest.evidence.{gate_name}.correctness_golden"
                    ".expected_greedy_text_sha256"
                ),
            )
            gate_report, _ = _load_json(report_path, f"{gate_name} report")
            loaded[gate_name] = (gate_report, report_sha)
            raw_hashes[gate_name] = raw_sha
            raw_hashes[f"{gate_name}_correctness_golden"] = correctness_golden_sha
            raw_paths[gate_name] = raw_path

        cuda_gate = _exact(
            evidence_row["cuda_fault"],
            {"build_image_id", "report", "raw_evidence"},
            "manifest.evidence.cuda_fault",
        )
        cuda_build_image_id = _image_id(
            cuda_gate["build_image_id"],
            "manifest.evidence.cuda_fault.build_image_id",
        )
        if cuda_build_image_id != trusted_cuda_build_image_id:
            _fail(
                "manifest.evidence.cuda_fault.build_image_id",
                "differs from the trusted expected CUDA-build image ID",
            )
        cuda_report_path, cuda_report_sha, _ = _resolve_artifact(
            cuda_gate["report"],
            "manifest.evidence.cuda_fault.report",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        cuda_raw_path, cuda_raw_sha, _ = _resolve_artifact(
            cuda_gate["raw_evidence"],
            "manifest.evidence.cuda_fault.raw_evidence",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        cuda_report, _ = _load_json(cuda_report_path, "cuda_fault report")
        loaded["cuda_fault"] = (cuda_report, cuda_report_sha)
        raw_hashes["cuda_fault"] = cuda_raw_sha
        raw_paths["cuda_fault"] = cuda_raw_path

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
            snapshot_root,
        )
        performance_raw_path, performance_raw_sha, _ = _resolve_artifact(
            performance["raw_evidence"],
            "manifest.evidence.performance.raw_evidence",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        performance_report, _ = _load_json(
            performance_report_path, "performance report"
        )
        loaded["performance"] = (performance_report, performance_report_sha)
        raw_hashes["performance"] = performance_raw_sha
        raw_paths["performance"] = performance_raw_path

        for gate_name in ("reliability_soak",):
            gate = _exact(
                evidence_row[gate_name],
                {"report", "raw_evidence"},
                f"manifest.evidence.{gate_name}",
            )
            report_path, report_sha, _ = _resolve_artifact(
                gate["report"],
                f"manifest.evidence.{gate_name}.report",
                evidence_root,
                seen_paths,
                snapshot_root,
            )
            raw_path, raw_sha, _ = _resolve_artifact(
                gate["raw_evidence"],
                f"manifest.evidence.{gate_name}.raw_evidence",
                evidence_root,
                seen_paths,
                snapshot_root,
            )
            gate_report, _ = _load_json(report_path, f"{gate_name} report")
            loaded[gate_name] = (gate_report, report_sha)
            raw_hashes[gate_name] = raw_sha
            raw_paths[gate_name] = raw_path

        native = _exact(
            evidence_row["native_correctness"],
            {"report", "raw_replay", "candidate_executable"},
            "manifest.evidence.native_correctness",
        )
        native_report_path, native_report_sha, _ = _resolve_artifact(
            native["report"],
            "manifest.evidence.native_correctness.report",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        native_raw_path, native_raw_sha, _ = _resolve_artifact(
            native["raw_replay"],
            "manifest.evidence.native_correctness.raw_replay",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        native_executable_path, native_executable_sha, _ = _resolve_artifact(
            native["candidate_executable"],
            "manifest.evidence.native_correctness.candidate_executable",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        if not os.access(native_executable_path, os.X_OK):
            _fail(
                "manifest.evidence.native_correctness.candidate_executable",
                "must be executable",
            )
        native_document, _ = _load_json(
            native_report_path, "native_correctness report"
        )
        loaded["native_correctness_report"] = (
            native_document,
            native_report_sha,
        )
        raw_hashes["native_correctness"] = native_raw_sha
        raw_paths["native_correctness"] = native_raw_path

        optimization = _exact(
            evidence_row["optimization_correctness"],
            {"build_image_id", "report", "raw_evidence"},
            "manifest.evidence.optimization_correctness",
        )
        optimization_build_image_id = _image_id(
            optimization["build_image_id"],
            "manifest.evidence.optimization_correctness.build_image_id",
        )
        if optimization_build_image_id != trusted_optimization_build_image_id:
            _fail(
                "manifest.evidence.optimization_correctness.build_image_id",
                "differs from the trusted expected optimization-build image ID",
            )
        optimization_report_path, optimization_report_sha, _ = _resolve_artifact(
            optimization["report"], "manifest.evidence.optimization_correctness.report",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        optimization_raw_path, optimization_raw_sha, _ = _resolve_artifact(
            optimization["raw_evidence"],
            "manifest.evidence.optimization_correctness.raw_evidence",
            evidence_root,
            seen_paths,
            snapshot_root,
        )
        optimization_report, _ = _load_json(
            optimization_report_path, "optimization_correctness report"
        )
        loaded["optimization_correctness"] = (
            optimization_report, optimization_report_sha
        )
        raw_hashes["optimization_correctness"] = optimization_raw_sha
        raw_paths["optimization_correctness"] = optimization_raw_path

        try:
            reproducibility_replay = (
                reproducible_build_evidence.check_reproducible_build(
                    evidence_a=reproducible_a_path,
                    evidence_b=reproducible_b_path,
                    source_archive=archive_path,
                    expected_source_archive_sha256=trusted_archive_sha256,
                    source_revision=trusted_revision,
                    source_date_epoch=source_date_epoch,
                    build_image_id=trusted_reproducible_build_image_id,
                    final_binary=binary_path,
                    final_profile_binary=profile_binary_path,
                    final_bundle=bundle_path,
                    final_native_manifest=native_manifest_path,
                )
            )
        except (ReleaseContractError, OSError) as error:
            _fail("reproducible_build", str(error))
        reproducibility_report_sha256 = _validate_reproducibility_replay(
            reproducibility_replay,
            revision=revision,
            archive_sha256=archive_sha256,
            source_date_epoch=source_date_epoch,
            build_image_id=trusted_reproducible_build_image_id,
            evidence_a_sha256=reproducible_a_sha,
            evidence_b_sha256=reproducible_b_sha,
            binary_sha256=binary_sha256,
            profile_binary_sha256=profile_binary_sha,
            bundle_sha256=bundle_sha256,
            native_manifest_sha256=native_manifest_sha,
        )

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
        _validate_cuda_fault_replay(
            loaded["cuda_fault"][0],
            raw_paths["cuda_fault"],
            revision=revision,
            source_archive=archive_path,
            build_image_id=cuda_build_image_id,
            release_binary=binary_path,
            release_bundle=bundle_path,
            release_image_id=image_digest,
        )
        native_correctness_sha256 = loaded["native_correctness_report"][1]
        try:
            native_replay = native_correctness_evidence.replay_raw_evidence(
                raw_paths["native_correctness"],
                source_revision=revision,
                source_archive=archive_path,
                correctness_report=native_report_path,
                candidate_executable=native_executable_path,
            )
        except (
            native_correctness_evidence.NativeCorrectnessEvidenceError,
            OSError,
        ) as error:
            _fail("native_correctness.raw_replay", str(error))
        if (
            native_replay.source_archive_sha256 != archive_sha256
            or native_replay.correctness_report_sha256 != native_correctness_sha256
            or native_replay.candidate_executable_sha256 != native_executable_sha
            or native_replay.case_count != 31
            or native_replay.failure_count != 0
        ):
            _fail(
                "native_correctness.raw_replay",
                "replayed source/report/executable/count bindings differ",
            )
        _validate_correctness(
            loaded["native_correctness_report"][0],
            "native_correctness",
            revision,
            native_replay.candidate_executable_sha256,
        )
        optimization_profile_image_sha256 = _validate_optimization_correctness(
            loaded["optimization_correctness"][0],
            "optimization_correctness",
            revision=revision,
            archive_sha256=archive_sha256,
        )
        if optimization_profile_image_sha256 != (
            trusted_optimization_build_image_id.removeprefix("sha256:")
        ):
            _fail(
                "optimization_correctness.build.container_image_sha256",
                "differs from the trusted expected optimization-build image ID",
            )
        try:
            optimization_replay = optimization_evidence.replay_raw_evidence(
                raw_paths["optimization_correctness"],
                report=optimization_report_path,
                source_revision=revision,
                source_archive_sha256=archive_sha256,
                build_image_id=trusted_optimization_build_image_id,
                profile_binary=profile_binary_path,
            )
        except (optimization_evidence.OptimizationEvidenceError, OSError) as error:
            _fail("optimization_correctness.raw_evidence", str(error))
        expected_optimization_replay = {
            "report": loaded["optimization_correctness"][0],
            "report_sha256": optimization_report_sha,
            "raw_evidence_sha256": optimization_raw_sha,
            "profile_binary_sha256": profile_binary_sha,
            "build_image_sha256": (
                trusted_optimization_build_image_id.removeprefix("sha256:")
            ),
        }
        for field, expected in expected_optimization_replay.items():
            if optimization_replay.get(field) != expected:
                _fail(
                    f"optimization_correctness.raw_evidence.{field}",
                    "does not match the final candidate binding",
                )
        fixed37_report_test = next(
            test
            for test in loaded["optimization_correctness"][0]["tests"]
            if test["id"] == "fixed37-production-batch-e0"
        )
        replayed_log_sha256 = optimization_replay.get("log_sha256")
        replayed_test_binary_sha256 = optimization_replay.get("test_binary_sha256")
        if (
            not isinstance(replayed_log_sha256, dict)
            or replayed_log_sha256.get("fixed37-production-batch-e0")
            != fixed37_report_test["log_sha256"]
            or not isinstance(replayed_test_binary_sha256, dict)
            or replayed_test_binary_sha256.get(
                "fixed37-production-batch-gpu-test"
            )
            != fixed37_report_test["test_binary_sha256"]
        ):
            _fail(
                "optimization_correctness.raw_evidence.fixed37-production-batch-e0",
                "replayed fixed37 log/test ELF differs from the submitted diagnostic row",
            )
        python_free_model = _validate_python_free_e2e_replay(
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
            correctness_golden_sha256=trusted_correctness_golden_sha256,
        )
        _validate_performance(
            loaded["performance"][0], "performance", revision=revision,
            archive_sha256=archive_sha256, binary_sha256=binary_sha256,
            image_sha256=image_sha256,
            optimization_sha256=optimization_report_sha,
            optimization_gate_id=OPTIMIZATION_GATE,
            optimization_profile_image_sha256=optimization_profile_image_sha256,
            optimization_correctness=loaded["optimization_correctness"][0],
            profile_binary_sha256=profile_binary_sha,
            raw_evidence_path=raw_paths["performance"],
            raw_evidence_sha256=raw_hashes["performance"],
            candidate_id=trusted_candidate_id,
        )
        _validate_soak(
            loaded["reliability_soak"][0], "reliability_soak", revision=revision,
            archive_sha256=archive_sha256, binary_sha256=binary_sha256,
            image_sha256=image_sha256,
            model=python_free_model,
            raw_evidence_path=raw_paths["reliability_soak"],
            raw_evidence_sha256=raw_hashes["reliability_soak"],
            correctness_golden_path=correctness_golden_path,
            correctness_golden_sha256=trusted_correctness_golden_sha256,
            generated_text_sha256=trusted_generated_text_sha256,
            native_correctness_report_path=native_report_path,
            native_correctness_report_sha256=native_correctness_sha256,
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
                    "build_image_ids": {
                        "reproducible_build": trusted_reproducible_build_image_id,
                        "cuda_fault": trusted_cuda_build_image_id,
                        "optimization_correctness": (
                            trusted_optimization_build_image_id
                        ),
                    },
                    "native_correctness_executable_sha256": native_executable_sha,
                    "profile_binary_sha256": profile_binary_sha,
                    "reproducibility_report_sha256": reproducibility_report_sha256,
                    "correctness_golden_sha256": trusted_correctness_golden_sha256,
                    "evidence_sha256": dict(sorted(evidence_hashes.items())),
                },
                "checks": [
                    {"name": name, "passed": True}
                    for name in (
                        "release_bundle", "reproducible_build", "python_free_e2e", "cuda_fault",
                        "native_correctness", "optimization_correctness",
                        "fixed37_production_batch_e0",
                        "performance", "reliability_soak", "cross_bindings",
                    )
                ],
            }
        )
    except (OSError, ValueError) as error:
        report["errors"] = [str(error)]
    finally:
        if evidence_root_fd >= 0:
            os.close(evidence_root_fd)
        if snapshot_temporary is not None:
            snapshot_temporary.cleanup()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--expected-candidate-id", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--expected-release-image-id", required=True)
    parser.add_argument("--expected-reproducible-build-image-id", required=True)
    parser.add_argument("--expected-cuda-build-image-id", required=True)
    parser.add_argument("--expected-optimization-build-image-id", required=True)
    parser.add_argument("--expected-correctness-golden-sha256", required=True)
    parser.add_argument("--report", type=Path, help="create without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.manifest,
        args.evidence_root,
        expected_candidate_id=args.expected_candidate_id,
        expected_revision=args.expected_revision,
        expected_source_archive_sha256=args.expected_source_archive_sha256,
        expected_release_image_id=args.expected_release_image_id,
        expected_reproducible_build_image_id=(
            args.expected_reproducible_build_image_id
        ),
        expected_cuda_build_image_id=args.expected_cuda_build_image_id,
        expected_optimization_build_image_id=(
            args.expected_optimization_build_image_id
        ),
        expected_correctness_golden_sha256=(
            args.expected_correctness_golden_sha256
        ),
    )
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
