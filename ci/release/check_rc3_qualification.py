#!/usr/bin/env python3
"""Fail-closed RC3 qualification envelope for an externally frozen candidate.

This is deliberately an *outer* checker.  ``check_release_candidate.py``
continues to own the current Gate E evidence semantics; this checker binds that
passed report and the C02-only receipts to one immutable RC3 freeze document.
It never starts a model, CUDA process, container, SSH session, or network
request.

The freeze document is intentionally outside the Git checkout.  Putting its
own commit/archive digest in a tracked file would make a clean candidate
self-referential and therefore non-reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

# Child C02 checkers import this file by its stable module name for the shared
# fail-closed primitives.  When this outer checker is executed as a script,
# register that name before any lazy child import so their exception classes
# are the same objects caught by ``evaluate`` below.
if __name__ == "__main__":
    sys.modules["check_rc3_qualification"] = sys.modules[__name__]

import check_release_candidate as release_candidate


FREEZE_VERSION = "riley.rc3-qualification-candidate.v1"
RECEIPT_VERSION = "riley.rc3-qualification-receipt.v1"
REPORT_VERSION = "riley.rc3-qualification-report.v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CUDA_ABI_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){0,2}$")
RELATIVE_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
NON_EMPTY_TEXT_RE = re.compile(r"^\S(?:[^\r\n]*\S)?$")
ENVIRONMENT_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
REQUIRED_GATES = (
    "startup_configuration",
    "qwen_multistep",
    "routing",
    "fault_extension",
    "soak_v2",
    "rollback",
)
BASE_CHECKS = (
    "release_bundle",
    "reproducible_build",
    "python_free_e2e",
    "cuda_fault",
    "native_correctness",
    "optimization_correctness",
    "fixed37_production_batch_e0",
    "performance",
    "reliability_soak",
    "cross_bindings",
)
BASE_EVIDENCE_SHA256_KEYS = (
    "cuda_fault",
    "cuda_fault_raw",
    "native_correctness_report",
    "native_correctness_raw",
    "optimization_correctness",
    "optimization_correctness_raw",
    "performance",
    "performance_raw",
    "python_free_e2e",
    "python_free_e2e_correctness_golden_raw",
    "python_free_e2e_raw",
    "reliability_soak",
    "reliability_soak_raw",
    "reproducible_build_a_raw",
    "reproducible_build_b_raw",
    "reproducible_build_native_manifest_raw",
)
_OPEN_COMMON = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
_OPEN_DIRECTORY = _OPEN_COMMON | getattr(os, "O_DIRECTORY", 0)


class QualificationError(ValueError):
    """An RC3 input is malformed or cannot qualify."""


class IncomparableError(QualificationError):
    """Evidence belongs to a different candidate or immutable binding."""


class GateFailure(QualificationError):
    """A required gate explicitly did not pass."""


@dataclass(frozen=True)
class Descriptor:
    path: str
    sha256: str | None = None


@dataclass(frozen=True)
class FrozenCandidate:
    candidate_id: str
    source: dict[str, str]
    release: dict[str, str]
    images: dict[str, str]
    toolchain: dict[str, str]
    models: dict[str, dict[str, str]]
    arms: dict[str, dict[str, Any]]
    rollback: dict[str, str]
    final_manifest: Descriptor
    final_report: Descriptor
    receipts: dict[str, Descriptor]


def _fail(code: str, message: str) -> NoReturn:
    error = QualificationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _incomparable(message: str) -> NoReturn:
    error = IncomparableError(message)
    error.reason_code = "incomparable-binding"  # type: ignore[attr-defined]
    raise error


def _gate_failure(message: str) -> NoReturn:
    error = GateFailure(message)
    error.reason_code = "gate-failed"  # type: ignore[attr-defined]
    raise error


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate-json-key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail("non-finite-json-number", f"non-finite JSON number {value!r} is forbidden")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _read_fd(fd: int, label: str) -> bytes:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        _fail("unsafe-evidence-path", f"{label} must be a regular file")
    if metadata.st_size > MAX_JSON_BYTES:
        _fail("input-too-large", f"{label} exceeds {MAX_JSON_BYTES} bytes")
    chunks: list[bytes] = []
    remaining = metadata.st_size
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            _fail("truncated-input", f"{label} changed while it was read")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        _fail("mutated-input", f"{label} grew while it was read")
    return b"".join(chunks)


def _read_regular_path(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        _fail("missing-input", f"{label} cannot be inspected: {error}")
    if not stat.S_ISREG(before.st_mode):
        _fail("unsafe-evidence-path", f"{label} must be a regular non-link file")
    try:
        fd = os.open(path, _OPEN_COMMON)
    except OSError as error:
        _fail("missing-input", f"{label} cannot be opened safely: {error}")
    try:
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            _fail("raced-input", f"{label} changed while it was opened")
        return _read_fd(fd, label)
    finally:
        os.close(fd)


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("invalid-relative-path", f"{label} must be a non-empty relative path")
    if "\x00" in value or "\\" in value or "//" in value or not RELATIVE_PATH_RE.fullmatch(value):
        _fail("invalid-relative-path", f"{label} must be normalized POSIX text")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail("invalid-relative-path", f"{label} must not contain traversal or aliases")
    return value


def _read_relative(root: Path, relative: str, label: str) -> bytes:
    """Read a regular evidence file through no-follow directory FDs."""

    parts = PurePosixPath(relative).parts
    try:
        root_metadata = root.lstat()
    except OSError as error:
        _fail("missing-evidence-root", f"evidence root cannot be inspected: {error}")
    if not stat.S_ISDIR(root_metadata.st_mode):
        _fail("unsafe-evidence-root", "evidence root must be a real directory")
    try:
        current_fd = os.open(root, _OPEN_DIRECTORY)
    except OSError as error:
        _fail("unsafe-evidence-root", f"evidence root cannot be opened safely: {error}")
    try:
        for component in parts[:-1]:
            try:
                next_fd = os.open(component, _OPEN_DIRECTORY, dir_fd=current_fd)
            except OSError as error:
                _fail("unsafe-evidence-path", f"{label}: unsafe directory component: {error}")
            os.close(current_fd)
            current_fd = next_fd
        try:
            file_fd = os.open(parts[-1], _OPEN_COMMON, dir_fd=current_fd)
        except OSError as error:
            _fail("missing-input", f"{label} cannot be opened safely: {error}")
        try:
            return _read_fd(file_fd, label)
        finally:
            os.close(file_fd)
    finally:
        os.close(current_fd)


def _parse_document(raw: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except UnicodeDecodeError as error:
        _fail("invalid-json", f"{label} is not UTF-8: {error}")
    except json.JSONDecodeError as error:
        _fail("invalid-json", f"{label} is not JSON: {error}")
    if not isinstance(decoded, dict):
        _fail("invalid-json", f"{label} root must be an object")
    return decoded


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("invalid-shape", f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; missing={sorted(fields - actual)}, extra={sorted(actual - fields)}",
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail("invalid-string", f"{label} must be a non-empty string")
    return value


def _text(value: Any, label: str) -> str:
    value = _string(value, label)
    if not NON_EMPTY_TEXT_RE.fullmatch(value):
        _fail("invalid-string", f"{label} must not have surrounding or line whitespace")
    return value


def _sha256(value: Any, label: str) -> str:
    value = _string(value, label)
    if not SHA256_RE.fullmatch(value) or value == "0" * 64:
        _fail("invalid-sha256", f"{label} must be a lowercase SHA-256")
    return value


def _image(value: Any, label: str) -> str:
    value = _string(value, label)
    if not IMAGE_RE.fullmatch(value) or value == "sha256:" + "0" * 64:
        _fail("invalid-image-id", f"{label} must be sha256:<lowercase digest>")
    return value


def _descriptor(value: Any, label: str, *, require_sha256: bool) -> Descriptor:
    fields = {"path", "sha256"} if require_sha256 else {"path"}
    row = _exact(value, fields, label)
    return Descriptor(
        path=_relative_path(row["path"], f"{label}.path"),
        sha256=_sha256(row["sha256"], f"{label}.sha256") if require_sha256 else None,
    )


def _configuration_arm(value: Any, label: str) -> dict[str, Any]:
    row = _exact(value, {"argv", "environment", "configuration_sha256"}, label)
    argv = row["argv"]
    if not isinstance(argv, list) or not argv or not all(
        isinstance(argument, str) and NON_EMPTY_TEXT_RE.fullmatch(argument)
        for argument in argv
    ):
        _fail("invalid-configuration", f"{label}.argv must be a non-empty string array")
    environment = row["environment"]
    if not isinstance(environment, dict) or not all(
        isinstance(key, str)
        and ENVIRONMENT_KEY_RE.fullmatch(key)
        and isinstance(item, str)
        and "\r" not in item
        and "\n" not in item
        for key, item in environment.items()
    ):
        _fail("invalid-configuration", f"{label}.environment must be a string map")
    expected = hashlib.sha256(
        canonical_json_bytes({"argv": argv, "environment": environment})
    ).hexdigest()
    supplied = _sha256(row["configuration_sha256"], f"{label}.configuration_sha256")
    if supplied != expected:
        _fail("configuration-hash-mismatch", f"{label} canonical configuration SHA-256 mismatch")
    return {"argv": argv, "environment": environment, "configuration_sha256": supplied}


def _model(value: Any, label: str) -> dict[str, str]:
    row = _exact(
        value,
        {
            "model_id",
            "model_revision",
            "config_sha256",
            "weights_sha256",
            "tokenizer_revision",
            "tokenizer_files_sha256",
        },
        label,
    )
    model_revision = _string(row["model_revision"], f"{label}.model_revision")
    tokenizer_revision = _string(row["tokenizer_revision"], f"{label}.tokenizer_revision")
    if not GIT_RE.fullmatch(model_revision) or model_revision == "0" * 40:
        _fail("invalid-model-revision", f"{label}.model_revision must be a full lowercase SHA")
    if not GIT_RE.fullmatch(tokenizer_revision) or tokenizer_revision == "0" * 40:
        _fail("invalid-tokenizer-revision", f"{label}.tokenizer_revision must be a full lowercase SHA")
    return {
        "model_id": _text(row["model_id"], f"{label}.model_id"),
        "model_revision": model_revision,
        "config_sha256": _sha256(row["config_sha256"], f"{label}.config_sha256"),
        "weights_sha256": _sha256(row["weights_sha256"], f"{label}.weights_sha256"),
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_files_sha256": _sha256(
            row["tokenizer_files_sha256"], f"{label}.tokenizer_files_sha256"
        ),
    }


def _validate_freeze(document: dict[str, Any]) -> FrozenCandidate:
    row = _exact(
        document,
        {
            "schema_version",
            "candidate_id",
            "created_at_utc",
            "status",
            "source",
            "release",
            "images",
            "toolchain",
            "models",
            "arms",
            "rollback",
            "outputs",
            "required_gates",
        },
        "freeze",
    )
    if row["schema_version"] != FREEZE_VERSION:
        _fail("unsupported-freeze-version", "freeze.schema_version is unsupported")
    candidate_id = _string(row["candidate_id"], "freeze.candidate_id")
    if not release_candidate.CANDIDATE_ID_RE.fullmatch(candidate_id):
        _fail("invalid-candidate-id", "freeze.candidate_id is not a valid RC candidate")
    if not isinstance(row["created_at_utc"], str) or not UTC_RE.fullmatch(row["created_at_utc"]):
        _fail("invalid-created-at", "freeze.created_at_utc must be UTC second precision")
    if row["status"] != "frozen":
        _fail("invalid-freeze-status", "freeze.status must be frozen")

    source_row = _exact(
        row["source"],
        {
            "git_revision",
            "archive_sha256",
            "cargo_lock_sha256",
            "extension_registry_sha256",
            "correctness_golden_sha256",
        },
        "freeze.source",
    )
    revision = _string(source_row["git_revision"], "freeze.source.git_revision")
    if not GIT_RE.fullmatch(revision) or revision == "0" * 40:
        _fail("invalid-git-revision", "freeze.source.git_revision must be a full lowercase SHA")
    source = {
        "git_revision": revision,
        "archive_sha256": _sha256(source_row["archive_sha256"], "freeze.source.archive_sha256"),
        "cargo_lock_sha256": _sha256(source_row["cargo_lock_sha256"], "freeze.source.cargo_lock_sha256"),
        "extension_registry_sha256": _sha256(
            source_row["extension_registry_sha256"],
            "freeze.source.extension_registry_sha256",
        ),
        "correctness_golden_sha256": _sha256(
            source_row["correctness_golden_sha256"],
            "freeze.source.correctness_golden_sha256",
        ),
    }
    release_row = _exact(
        row["release"],
        {"binary_sha256", "bundle_sha256", "image_id", "cuda_c_abi_version"},
        "freeze.release",
    )
    release = {
        "binary_sha256": _sha256(release_row["binary_sha256"], "freeze.release.binary_sha256"),
        "bundle_sha256": _sha256(release_row["bundle_sha256"], "freeze.release.bundle_sha256"),
        "image_id": _image(release_row["image_id"], "freeze.release.image_id"),
        "cuda_c_abi_version": _string(
            release_row["cuda_c_abi_version"], "freeze.release.cuda_c_abi_version"
        ),
    }
    if not CUDA_ABI_RE.fullmatch(release["cuda_c_abi_version"]):
        _fail("invalid-cuda-abi-version", "freeze.release.cuda_c_abi_version is invalid")
    images_row = _exact(row["images"], {"reproducible", "cuda", "optimization"}, "freeze.images")
    images = {name: _image(images_row[name], f"freeze.images.{name}") for name in sorted(images_row)}
    toolchain_row = _exact(
        row["toolchain"],
        {"rustc", "nvcc", "driver", "cuda_runtime", "cuda_toolkit", "cublas"},
        "freeze.toolchain",
    )
    toolchain = {
        name: _text(toolchain_row[name], f"freeze.toolchain.{name}")
        for name in sorted(toolchain_row)
    }
    models_row = _exact(row["models"], {"smollm2", "qwen"}, "freeze.models")
    models = {name: _model(models_row[name], f"freeze.models.{name}") for name in sorted(models_row)}
    arms_row = _exact(row["arms"], {"stable_default", "max_performance_exact"}, "freeze.arms")
    arms = {name: _configuration_arm(arms_row[name], f"freeze.arms.{name}") for name in sorted(arms_row)}
    if arms["stable_default"]["configuration_sha256"] == arms["max_performance_exact"]["configuration_sha256"]:
        _fail("indistinguishable-arms", "stable-default and max-performance arms must be distinct")
    rollback_row = _exact(
        row["rollback"], {"binary_sha256", "bundle_sha256", "image_id"}, "freeze.rollback"
    )
    rollback = {
        "binary_sha256": _sha256(rollback_row["binary_sha256"], "freeze.rollback.binary_sha256"),
        "bundle_sha256": _sha256(rollback_row["bundle_sha256"], "freeze.rollback.bundle_sha256"),
        "image_id": _image(rollback_row["image_id"], "freeze.rollback.image_id"),
    }
    outputs_row = _exact(
        row["outputs"],
        {"final_release_candidate_manifest", "final_release_candidate", "receipts"},
        "freeze.outputs",
    )
    final_manifest = _descriptor(
        outputs_row["final_release_candidate_manifest"],
        "freeze.outputs.final_release_candidate_manifest",
        require_sha256=False,
    )
    final_report = _descriptor(outputs_row["final_release_candidate"], "freeze.outputs.final_release_candidate", require_sha256=False)
    receipt_rows = _exact(outputs_row["receipts"], set(REQUIRED_GATES), "freeze.outputs.receipts")
    receipts = {
        gate: _descriptor(receipt_rows[gate], f"freeze.outputs.receipts.{gate}", require_sha256=False)
        for gate in REQUIRED_GATES
    }
    all_paths = [
        final_manifest.path,
        final_report.path,
        *(descriptor.path for descriptor in receipts.values()),
    ]
    if len(all_paths) != len(set(all_paths)):
        _fail("duplicate-output-path", "freeze output paths must be unique")
    if row["required_gates"] != list(REQUIRED_GATES):
        _fail("wrong-required-gates", "freeze.required_gates must be the canonical ordered C02 list")
    return FrozenCandidate(
        candidate_id,
        source,
        release,
        images,
        toolchain,
        models,
        arms,
        rollback,
        final_manifest,
        final_report,
        receipts,
    )


def _validate_base_report_shape(document: dict[str, Any], frozen: FrozenCandidate) -> None:
    row = _exact(
        document,
        {"schema_version", "status", "passed", "candidate_id", "manifest_sha256", "bindings", "checks", "errors"},
        "final release candidate report",
    )
    if row["schema_version"] != release_candidate.REPORT_VERSION:
        _fail("unsupported-base-report", "final release candidate report schema changed")
    if row["status"] != "passed" or row["passed"] is not True:
        _gate_failure("existing final release candidate gate did not pass")
    if row["candidate_id"] != frozen.candidate_id:
        _incomparable("final release candidate report belongs to another candidate")
    if row["errors"] != []:
        _gate_failure("a passed final release candidate report must have no errors")
    _sha256(row["manifest_sha256"], "final release candidate report.manifest_sha256")
    bindings = _exact(
        row["bindings"],
        {
            "git_revision", "source_archive_sha256", "release_binary_sha256", "release_bundle_sha256",
            "release_image_sha256", "build_image_ids", "native_correctness_executable_sha256",
            "profile_binary_sha256", "reproducibility_report_sha256", "correctness_golden_sha256",
            "evidence_sha256",
        },
        "final release candidate report.bindings",
    )
    expected = {
        "git_revision": frozen.source["git_revision"],
        "source_archive_sha256": frozen.source["archive_sha256"],
        "release_binary_sha256": frozen.release["binary_sha256"],
        "release_bundle_sha256": frozen.release["bundle_sha256"],
        "release_image_sha256": frozen.release["image_id"].removeprefix("sha256:"),
    }
    for name, value in expected.items():
        if bindings[name] != value:
            _incomparable(f"final release candidate report binding drift: {name}")
    images = _exact(
        bindings["build_image_ids"],
        {"reproducible_build", "cuda_fault", "optimization_correctness"},
        "final release candidate report.bindings.build_image_ids",
    )
    expected_images = {
        "reproducible_build": frozen.images["reproducible"],
        "cuda_fault": frozen.images["cuda"],
        "optimization_correctness": frozen.images["optimization"],
    }
    if images != expected_images:
        _incomparable("final release candidate report build image binding drift")
    for field in (
        "native_correctness_executable_sha256",
        "profile_binary_sha256",
        "reproducibility_report_sha256",
    ):
        _sha256(bindings[field], f"final release candidate report.bindings.{field}")
    if _sha256(
        bindings["correctness_golden_sha256"],
        "final release candidate report.bindings.correctness_golden_sha256",
    ) != frozen.source["correctness_golden_sha256"]:
        _incomparable("final release candidate report correctness golden binding drift")
    evidence_hashes = _exact(
        bindings["evidence_sha256"],
        set(BASE_EVIDENCE_SHA256_KEYS),
        "final release candidate report.bindings.evidence_sha256",
    )
    for name in BASE_EVIDENCE_SHA256_KEYS:
        _sha256(evidence_hashes[name], f"final release candidate report.bindings.evidence_sha256.{name}")
    checks = row["checks"]
    if not isinstance(checks, list) or len(checks) != len(BASE_CHECKS):
        _fail("invalid-base-report", "final release candidate report has an invalid check list")
    observed_checks: list[str] = []
    for index, check in enumerate(checks):
        item = _exact(check, {"name", "passed"}, f"final release candidate report.checks[{index}]")
        if item["passed"] is not True:
            _gate_failure(f"base check {item['name']!r} did not pass")
        observed_checks.append(_string(item["name"], f"final release candidate report.checks[{index}].name"))
    if tuple(observed_checks) != BASE_CHECKS:
        _fail("invalid-base-report", "final release candidate report check inventory drifted")


def _write_snapshot(path: Path, raw: bytes) -> None:
    """Publish immutable bytes into a private temporary replay path."""

    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("snapshot-write-failed", "could not snapshot final candidate manifest")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def revalidate_base_release_candidate(
    frozen: FrozenCandidate,
    freeze_sha256: str,
    evidence_root: Path,
) -> tuple[bytes, str]:
    """Replay Gate E from the freeze-declared manifest and compare its report.

    The submitted final report is not evidence on its own.  This function first
    reads the manifest through the outer checker's no-follow evidence reader,
    snapshots those exact bytes, and asks the existing Gate E checker to replay
    every referenced artifact.  A stale, hand-authored, or merely structural
    ``passed`` report therefore cannot qualify an RC3 candidate.
    """

    manifest_raw = _read_relative(
        evidence_root,
        frozen.final_manifest.path,
        "final release candidate manifest",
    )
    base_raw = _read_relative(
        evidence_root,
        frozen.final_report.path,
        "final release candidate report",
    )
    base_sha256 = hashlib.sha256(base_raw).hexdigest()
    base_document = _parse_document(base_raw, "final release candidate report")

    with tempfile.TemporaryDirectory(prefix="riley-rc3-gate-e-") as temporary:
        snapshot = Path(temporary) / "final-release-candidate-manifest.json"
        _write_snapshot(snapshot, manifest_raw)
        replayed = release_candidate.evaluate(
            snapshot,
            evidence_root,
            expected_candidate_id=frozen.candidate_id,
            expected_revision=frozen.source["git_revision"],
            expected_source_archive_sha256=frozen.source["archive_sha256"],
            expected_release_image_id=frozen.release["image_id"],
            expected_reproducible_build_image_id=frozen.images["reproducible"],
            expected_cuda_build_image_id=frozen.images["cuda"],
            expected_optimization_build_image_id=frozen.images["optimization"],
            expected_correctness_golden_sha256=frozen.source["correctness_golden_sha256"],
        )

    if replayed.get("passed") is not True:
        _gate_failure("replayed final release candidate Gate E did not pass")
    _validate_base_report_shape(base_document, frozen)
    if base_document != replayed:
        _fail(
            "base-report-replay-mismatch",
            "submitted final release candidate report differs from Gate E replay",
        )
    if base_document["manifest_sha256"] != hashlib.sha256(manifest_raw).hexdigest():
        _fail(
            "base-report-manifest-mismatch",
            "submitted final release candidate report does not bind the freeze-declared manifest",
        )
    # The parameter is intentionally consumed here rather than trusting a
    # receipt to quote it.  This protects callers of the public helper from
    # accidentally replaying a manifest after switching to a different freeze.
    _sha256(freeze_sha256, "freeze SHA-256")
    return base_raw, base_sha256


def _validate_receipt(
    document: dict[str, Any],
    *,
    gate: str,
    freeze_path: Path,
    frozen: FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    evidence_root: Path,
) -> None:
    """Validate one future C02 gate through its own semantic verifier."""

    if gate == "startup_configuration":
        # Import lazily: the config checker imports this module for the common
        # immutable-evidence primitives, so a top-level import would create a
        # partially initialized circular module.
        runtime_config = _load_semantic_checker("check_effective_runtime_config_receipt")

        if document.get("schema_version") == runtime_config.CHECK_REPORT_VERSION:
            _validate_startup_configuration_receipt(
                document,
                freeze_path=freeze_path,
                frozen=frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_report_sha256,
                evidence_root=evidence_root,
                runtime_config=runtime_config,
            )
            return

    if gate == "qwen_multistep":
        # Like the configuration checker, the Qwen semantic checker imports
        # this module for the immutable-evidence primitives.  Keep its import
        # lazy to avoid a partially initialized circular import.
        qwen_multistep = _load_semantic_checker("check_qwen_multistep_receipt")

        if document.get("schema_version") == qwen_multistep.CHECK_REPORT_VERSION:
            _validate_qwen_multistep_receipt(
                document,
                freeze_path=freeze_path,
                frozen=frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_report_sha256,
                evidence_root=evidence_root,
                qwen_multistep=qwen_multistep,
            )
            return

    if gate == "routing":
        # The C02 routing receipt is a finite release-binary corpus, not C03
        # fuzz input.  Its checker imports this module for no-follow and Gate E
        # primitives, so keep the registration lazy as with the other C02
        # semantic validators.
        routing = _load_semantic_checker("check_rc3_routing_receipt")

        if document.get("schema_version") == routing.CHECK_REPORT_VERSION:
            _validate_routing_receipt(
                document,
                freeze_path=freeze_path,
                frozen=frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_report_sha256,
                evidence_root=evidence_root,
                routing=routing,
            )
            return

    if gate == "fault_extension":
        # The fault extension is a semantic report whose ``receipt`` field
        # names a distinct raw input.  Re-run it through the child checker;
        # accepting the report bytes or the frozen report path as raw input
        # would turn a self-authored pass envelope into proof.
        fault_extension = _load_semantic_checker("check_fault_extension_receipt")

        if document.get("schema_version") == fault_extension.CHECK_REPORT_VERSION:
            _validate_fault_extension_receipt(
                document,
                freeze_path=freeze_path,
                frozen=frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_report_sha256,
                evidence_root=evidence_root,
                fault_extension=fault_extension,
            )
            return

    if gate == "soak_v2":
        # The C02 soak-v2 checker replays the source-controlled extended-soak
        # contract and Gate E archive.  Load it through the common identity
        # guard so script-mode failures still reach this outer fail-closed path.
        soak_v2 = _load_semantic_checker("check_soak_v2_receipt")

        if document.get("schema_version") == soak_v2.CHECK_REPORT_VERSION:
            _validate_soak_v2_receipt(
                document,
                freeze_path=freeze_path,
                frozen=frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_report_sha256,
                evidence_root=evidence_root,
                soak_v2=soak_v2,
            )
            return

    if gate == "rollback":
        # The rollback checker uses the same frozen Gate E/no-follow primitives
        # as this outer envelope.  Import it lazily to avoid a circular import.
        rollback = _load_semantic_checker("check_rc3_rollback_receipt")

        if document.get("schema_version") == rollback.CHECK_REPORT_VERSION:
            _validate_rollback_receipt(
                document,
                freeze_path=freeze_path,
                frozen=frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_report_sha256,
                evidence_root=evidence_root,
                rollback=rollback,
            )
            return

    # A digest of arbitrary bytes is provenance, not semantic proof.  C02
    # gates must register a closed, gate-specific verifier that replays or
    # validates their actual report shape.  Until that implementation exists,
    # a candidate is intentionally not qualifiable.
    _fail(
        "unimplemented-gate-validator",
        f"receipt.{gate} has no registered semantic validator",
    )


def _load_semantic_checker(module_name: str) -> Any:
    """Load a child checker against this outer module's exception hierarchy.

    A fresh script process naturally imports children after the ``__main__``
    alias above.  An embedded host can pre-import a child, however, leaving it
    with classes derived from a previous named outer module.  Reload exactly
    that closed child module in the mismatch case, so malformed evidence still
    reaches this outer checker's fail-closed handler.
    """

    try:
        checker = importlib.import_module(module_name)
        if getattr(checker, "qualification", None) is not sys.modules[__name__]:
            checker = importlib.reload(checker)
    except (ImportError, AttributeError) as error:
        _fail("semantic-checker-load-failed", f"cannot load semantic checker {module_name}: {error}")
    if getattr(checker, "qualification", None) is not sys.modules[__name__]:
        _fail("semantic-checker-module-identity", f"semantic checker {module_name} has an incompatible outer module")
    return checker


def _validate_startup_configuration_receipt(
    document: dict[str, Any],
    *,
    freeze_path: Path,
    frozen: FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    evidence_root: Path,
    runtime_config: Any,
) -> None:
    """Replay the startup-config checker rather than trust its passed report."""

    receipt = runtime_config.validate_check_report(document)
    if receipt.candidate_id != frozen.candidate_id:
        _incomparable("startup configuration receipt belongs to another candidate")
    if receipt.freeze_sha256 != freeze_sha256:
        _incomparable("startup configuration receipt does not bind this freeze")
    if (
        receipt.base_release_candidate_report.path != frozen.final_report.path
        or receipt.base_release_candidate_report.sha256 != base_report_sha256
    ):
        _incomparable("startup configuration receipt base Gate E binding drifted")
    if receipt.stable_promotion_profile != runtime_config.STABLE_DEFAULT_PROFILE:
        _fail(
            "invalid-stable-promotion-profile",
            "only stable-default may be the startup gate's promotion profile",
        )
    arm_paths: list[str] = []
    for profile in runtime_config.ARM_PROFILES:
        arm = receipt.arms.get(profile)
        if arm is None:
            _fail(
                "missing-configuration-arm",
                f"startup configuration receipt omits required {profile} evidence",
            )
        expected_arm = runtime_config.PROFILE_TO_FREEZE_ARM[profile]
        if arm.configuration_profile != profile:
            _fail(
                "invalid-configuration-arm-profile",
                f"startup configuration receipt {profile} evidence declares another profile",
            )
        if arm.configuration_sha256 != frozen.arms[expected_arm]["configuration_sha256"]:
            _incomparable(
                f"startup configuration receipt {profile} configuration hash drifted"
            )
        arm_paths.extend((arm.endpoint_payload.path, arm.startup_artifact.path))
    if len(arm_paths) != len(set(arm_paths)):
        _fail(
            "duplicate-startup-configuration-input",
            "configuration endpoint and startup artifact inputs must not reuse a path",
        )
    replayed = runtime_config.evaluate(
        freeze_path,
        evidence_root,
        receipt.arms[runtime_config.STABLE_DEFAULT_PROFILE].endpoint_payload.path,
        receipt.arms[runtime_config.STABLE_DEFAULT_PROFILE].startup_artifact.path,
        receipt.arms[runtime_config.MAX_PERFORMANCE_EXACT_PROFILE].endpoint_payload.path,
        receipt.arms[runtime_config.MAX_PERFORMANCE_EXACT_PROFILE].startup_artifact.path,
        expected_freeze_sha256=freeze_sha256,
    )
    if replayed.get("status") != "passed" or replayed.get("passed") is not True:
        _gate_failure("startup configuration semantic replay did not pass")
    if document != replayed:
        _fail(
            "startup-configuration-replay-mismatch",
            "submitted startup configuration receipt differs from semantic replay",
        )


def _validate_qwen_multistep_receipt(
    document: dict[str, Any],
    *,
    freeze_path: Path,
    frozen: FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    evidence_root: Path,
    qwen_multistep: Any,
) -> None:
    """Replay frozen Qwen evidence instead of trusting its check report."""

    receipt = qwen_multistep.validate_check_report(document)
    if receipt.candidate_id != frozen.candidate_id:
        _incomparable("Qwen multi-step receipt belongs to another candidate")
    if receipt.freeze_sha256 != freeze_sha256:
        _incomparable("Qwen multi-step receipt does not bind this freeze")
    if (
        receipt.base_release_candidate_report.path != frozen.final_report.path
        or receipt.base_release_candidate_report.sha256 != base_report_sha256
    ):
        _incomparable("Qwen multi-step receipt base Gate E binding drifted")
    expected_bindings = {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": qwen_multistep.STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }
    if receipt.bindings != expected_bindings:
        _incomparable("Qwen multi-step receipt does not bind frozen stable-default")
    if receipt.model != frozen.models["qwen"]:
        _incomparable("Qwen multi-step receipt model identity drifted")
    if receipt.receipt.path in {
        frozen.final_manifest.path,
        frozen.final_report.path,
        *(descriptor.path for descriptor in frozen.receipts.values()),
    }:
        _fail(
            "reserved-output-path-collision",
            "Qwen raw receipt must not reuse a freeze-declared output path",
        )
    replayed = qwen_multistep.evaluate(
        freeze_path,
        evidence_root,
        receipt.receipt.path,
        expected_freeze_sha256=freeze_sha256,
    )
    if replayed.get("status") != "passed" or replayed.get("passed") is not True:
        _gate_failure("Qwen multi-step semantic replay did not pass")
    if document != replayed:
        _fail(
            "qwen-multistep-replay-mismatch",
            "submitted Qwen multi-step receipt differs from semantic replay",
        )


def _validate_routing_receipt(
    document: dict[str, Any],
    *,
    freeze_path: Path,
    frozen: FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    evidence_root: Path,
    routing: Any,
) -> None:
    """Replay C02's fixed release-binary routing receipt before later gates."""

    receipt = routing.validate_check_report(document)
    if receipt.candidate_id != frozen.candidate_id:
        _incomparable("routing receipt belongs to another candidate")
    if receipt.freeze_sha256 != freeze_sha256:
        _incomparable("routing receipt does not bind this freeze")
    if (
        receipt.base_release_candidate_report.path != frozen.final_report.path
        or receipt.base_release_candidate_report.sha256 != base_report_sha256
    ):
        _incomparable("routing receipt base Gate E binding drifted")
    expected_bindings = {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": routing.STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }
    if receipt.bindings != expected_bindings:
        _incomparable("routing receipt does not bind frozen stable-default")
    if receipt.model != frozen.models["smollm2"]:
        _incomparable("routing receipt model identity drifted")
    if receipt.receipt.path in {
        frozen.final_manifest.path,
        frozen.final_report.path,
        *(descriptor.path for descriptor in frozen.receipts.values()),
    }:
        _fail(
            "reserved-output-path-collision",
            "routing raw receipt must not reuse a freeze-declared output path",
        )
    replayed = routing.evaluate(
        freeze_path,
        evidence_root,
        receipt.receipt.path,
        expected_freeze_sha256=freeze_sha256,
    )
    if replayed.get("status") != "passed" or replayed.get("passed") is not True:
        _gate_failure("routing semantic replay did not pass")
    if document != replayed:
        _fail(
            "routing-replay-mismatch",
            "submitted routing receipt differs from semantic replay",
        )


def _validate_fault_extension_receipt(
    document: dict[str, Any],
    *,
    freeze_path: Path,
    frozen: FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    evidence_root: Path,
    fault_extension: Any,
) -> None:
    """Replay C02 fault evidence from the report's distinct raw receipt."""

    receipt = fault_extension.validate_check_report(document)
    if receipt.candidate_id != frozen.candidate_id:
        _incomparable("fault-extension receipt belongs to another candidate")
    if receipt.freeze_sha256 != freeze_sha256:
        _incomparable("fault-extension receipt does not bind this freeze")
    if (
        receipt.base_release_candidate_report.path != frozen.final_report.path
        or receipt.base_release_candidate_report.sha256 != base_report_sha256
    ):
        _incomparable("fault-extension receipt base Gate E binding drifted")
    expected_bindings = {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": fault_extension.STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }
    if receipt.bindings != expected_bindings:
        _incomparable("fault-extension receipt does not bind frozen stable-default")
    if receipt.receipt.path in {
        frozen.final_manifest.path,
        frozen.final_report.path,
        *(descriptor.path for descriptor in frozen.receipts.values()),
    }:
        _fail(
            "reserved-output-path-collision",
            "fault-extension raw receipt must not reuse a freeze-declared output path",
        )
    replayed = fault_extension.evaluate(
        freeze_path,
        evidence_root,
        receipt.receipt.path,
        expected_freeze_sha256=freeze_sha256,
    )
    if replayed.get("status") != "passed" or replayed.get("passed") is not True:
        _gate_failure("fault-extension semantic replay did not pass")
    if document != replayed:
        _fail(
            "fault-extension-replay-mismatch",
            "submitted fault-extension receipt differs from semantic replay",
        )


def _validate_soak_v2_receipt(
    document: dict[str, Any],
    *,
    freeze_path: Path,
    frozen: FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    evidence_root: Path,
    soak_v2: Any,
) -> None:
    """Replay the C02 extended-soak receipt from its raw descriptor."""

    receipt = soak_v2.validate_check_report(document)
    if receipt.candidate_id != frozen.candidate_id:
        _incomparable("soak-v2 receipt belongs to another candidate")
    if receipt.freeze_sha256 != freeze_sha256:
        _incomparable("soak-v2 receipt does not bind this freeze")
    if (
        receipt.base_release_candidate_report.path != frozen.final_report.path
        or receipt.base_release_candidate_report.sha256 != base_report_sha256
    ):
        _incomparable("soak-v2 receipt base Gate E binding drifted")
    expected_bindings = {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": soak_v2.STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }
    if receipt.bindings != expected_bindings:
        _incomparable("soak-v2 receipt does not bind frozen stable-default")
    if receipt.receipt.path in {
        frozen.final_manifest.path,
        frozen.final_report.path,
        *(descriptor.path for descriptor in frozen.receipts.values()),
    }:
        _fail(
            "reserved-output-path-collision",
            "soak-v2 raw receipt must not reuse a freeze-declared output path",
        )
    replayed = soak_v2.evaluate(
        freeze_path,
        evidence_root,
        receipt.receipt.path,
        expected_freeze_sha256=freeze_sha256,
    )
    if replayed.get("status") != "passed" or replayed.get("passed") is not True:
        _gate_failure("soak-v2 semantic replay did not pass")
    if document != replayed:
        _fail(
            "soak-v2-replay-mismatch",
            "submitted soak-v2 receipt differs from semantic replay",
        )


def _validate_rollback_receipt(
    document: dict[str, Any],
    *,
    freeze_path: Path,
    frozen: FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    evidence_root: Path,
    rollback: Any,
) -> None:
    """Replay a prior-artifact rollback drill rather than trust its report."""

    receipt = rollback.validate_check_report(document)
    if receipt.candidate_id != frozen.candidate_id:
        _incomparable("rollback receipt belongs to another candidate")
    if receipt.freeze_sha256 != freeze_sha256:
        _incomparable("rollback receipt does not bind this freeze")
    if (
        receipt.base_release_candidate_report.path != frozen.final_report.path
        or receipt.base_release_candidate_report.sha256 != base_report_sha256
    ):
        _incomparable("rollback receipt base Gate E binding drifted")
    expected_bindings = {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": rollback.STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }
    if receipt.bindings != expected_bindings:
        _incomparable("rollback receipt does not bind frozen stable-default")
    expected_candidate_artifacts = rollback.ArtifactSet(
        binary_sha256=frozen.release["binary_sha256"],
        bundle_sha256=frozen.release["bundle_sha256"],
        image_id=frozen.release["image_id"],
    )
    expected_rollback_artifacts = rollback.ArtifactSet(
        binary_sha256=frozen.rollback["binary_sha256"],
        bundle_sha256=frozen.rollback["bundle_sha256"],
        image_id=frozen.rollback["image_id"],
    )
    if receipt.candidate_artifacts != expected_candidate_artifacts:
        _incomparable("rollback receipt candidate artifacts drifted from frozen release")
    if receipt.rollback_artifacts != expected_rollback_artifacts:
        _incomparable("rollback receipt prior artifacts drifted from frozen rollback")
    if receipt.receipt.path in {
        frozen.final_manifest.path,
        frozen.final_report.path,
        *(descriptor.path for descriptor in frozen.receipts.values()),
    }:
        _fail(
            "reserved-output-path-collision",
            "rollback raw receipt must not reuse a freeze-declared output path",
        )
    replayed = rollback.evaluate(
        freeze_path,
        evidence_root,
        receipt.receipt.path,
        expected_freeze_sha256=freeze_sha256,
    )
    if replayed.get("status") != "passed" or replayed.get("passed") is not True:
        _gate_failure("rollback semantic replay did not pass")
    if document != replayed:
        _fail(
            "rollback-replay-mismatch",
            "submitted rollback receipt differs from semantic replay",
        )


def _validate_repository(repository_root: Path, frozen: FrozenCandidate) -> None:
    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        _fail("missing-repository", f"repository root cannot be resolved: {error}")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        _fail("invalid-repository", f"cannot inspect candidate checkout: {error}")
    if head != frozen.source["git_revision"]:
        _incomparable("candidate checkout HEAD differs from the frozen revision")
    if status:
        _fail("dirty-candidate-source", "candidate checkout must be clean")
    for relative, expected in (
        ("Cargo.lock", frozen.source["cargo_lock_sha256"]),
        ("deploy/extensions/registry.json", frozen.source["extension_registry_sha256"]),
    ):
        actual = hashlib.sha256(_read_regular_path(root / relative, relative)).hexdigest()
        if actual != expected:
            _incomparable(f"candidate checkout hash drift: {relative}")


def _empty_report() -> dict[str, Any]:
    return {
        "schema_version": REPORT_VERSION,
        "candidate_id": None,
        "candidate_manifest_sha256": None,
        "status": "failed",
        "qualified": False,
        "final_release_candidate_manifest": None,
        "final_release_candidate": None,
        "gate_e_evidence_sha256": None,
        "receipts": None,
        "checks": [],
        "reason_codes": [],
    }


def evaluate(
    freeze_path: Path,
    evidence_root: Path,
    *,
    expected_candidate_sha256: str,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Cross-bind passed Gate E plus C02 receipts without executing a gate."""

    report = _empty_report()
    try:
        expected_freeze_sha256 = _sha256(expected_candidate_sha256, "--expected-candidate-sha256")
        freeze_raw = _read_regular_path(freeze_path, "freeze manifest")
        freeze_sha256 = hashlib.sha256(freeze_raw).hexdigest()
        report["candidate_manifest_sha256"] = freeze_sha256
        if freeze_sha256 != expected_freeze_sha256:
            _fail("candidate-sha-mismatch", "freeze manifest SHA-256 differs from trusted input")
        frozen = _validate_freeze(_parse_document(freeze_raw, "freeze manifest"))
        report["candidate_id"] = frozen.candidate_id
        if repository_root is None:
            _fail(
                "missing-repository-root",
                "RC3 qualification requires the exact clean candidate checkout",
            )
        _validate_repository(repository_root, frozen)
        base_raw, base_sha256 = revalidate_base_release_candidate(
            frozen,
            freeze_sha256,
            evidence_root,
        )
        base_document = _parse_document(
            base_raw,
            "final release candidate report",
        )
        base_manifest_sha256 = base_document["manifest_sha256"]
        base_gate_evidence_sha256 = base_document["bindings"]["evidence_sha256"]
        resolved_receipts: dict[str, dict[str, str]] = {}
        for gate in REQUIRED_GATES:
            receipt_descriptor = frozen.receipts[gate]
            receipt_raw = _read_relative(evidence_root, receipt_descriptor.path, f"receipt.{gate}")
            receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
            _validate_receipt(
                _parse_document(receipt_raw, f"receipt.{gate}"),
                gate=gate,
                freeze_path=freeze_path,
                frozen=frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_sha256,
                evidence_root=evidence_root,
            )
            resolved_receipts[gate] = {
                "path": receipt_descriptor.path,
                "sha256": receipt_sha256,
            }
        report.update(
            {
                "status": "passed",
                "qualified": True,
                "final_release_candidate_manifest": {
                    "path": frozen.final_manifest.path,
                    "sha256": base_manifest_sha256,
                },
                "final_release_candidate": {
                    "path": frozen.final_report.path,
                    "sha256": base_sha256,
                },
                "gate_e_evidence_sha256": base_gate_evidence_sha256,
                "receipts": resolved_receipts,
                "checks": [
                    *[{"name": f"base-{name.replace('_', '-')}", "passed": True} for name in BASE_CHECKS],
                    *[{"name": gate, "passed": True} for gate in REQUIRED_GATES],
                    {"name": "cross-bindings", "passed": True},
                ],
            }
        )
    except IncomparableError as error:
        report["status"] = "incomparable"
        report["reason_codes"] = [getattr(error, "reason_code", "incomparable-binding")]
    except GateFailure as error:
        report["reason_codes"] = [getattr(error, "reason_code", "gate-failed")]
    except (OSError, QualificationError) as error:
        report["reason_codes"] = [getattr(error, "reason_code", "invalid-input")]
    return report


def _write_create_only_bytes(path: Path, encoded: bytes) -> None:
    try:
        parent_metadata = path.parent.lstat()
        if not stat.S_ISDIR(parent_metadata.st_mode):
            _fail("unsafe-output-path", "report parent must be a real directory")
        if path.name in {"", ".", ".."}:
            _fail("unsafe-output-path", "report must have one regular filename")
        parent_fd = os.open(path.parent, _OPEN_DIRECTORY)
    except (FileExistsError, OSError) as error:
        raise QualificationError(f"cannot create report {path}: {error}") from error
    try:
        try:
            output_fd = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                0o644,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise QualificationError(f"cannot create report {path}: {error}") from error
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(output_fd, encoded[offset:])
                if written <= 0:
                    raise QualificationError(f"cannot create report {path}: short write")
                offset += written
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
    finally:
        os.close(parent_fd)


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _write_create_only_bytes(path, encoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="create without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.freeze,
        args.evidence_root,
        expected_candidate_sha256=args.expected_candidate_sha256,
        repository_root=args.repository_root,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            _write_create_only(args.report, report)
        except QualificationError as error:
            print(str(error), file=sys.stderr)
            return 2
    sys.stdout.write(encoded)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
