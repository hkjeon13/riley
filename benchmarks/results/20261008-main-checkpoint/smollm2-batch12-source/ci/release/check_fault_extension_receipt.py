#!/usr/bin/env python3
"""Fail-closed C02 semantic verifier for stable-default fault evidence.

``fault_extension`` is deliberately not a second self-authored ``passed``
envelope.  It binds the frozen RC3 candidate and stable-default arm to the
already-reviewed Gate E CUDA fault artifacts, then snapshots and replays the
closed raw CUDA evidence with :mod:`check_cuda_fault_evidence`.  It also
parses a closed raw trace for C02 §8's post-KV, output-status, scheduler, and
worker/channel faults, while explicitly separating that injectable synthetic
evidence from actual GPU compute-sanitizer evidence.  This checker is
CPU-only: it never starts CUDA, Riley, a container, SSH, or a network request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

import check_cuda_fault_evidence as cuda_fault
import check_rc3_qualification as qualification


RECEIPT_VERSION = "riley.fault-extension-receipt.v2"
CHECK_REPORT_VERSION = "riley.fault-extension-check.v2"
RAW_TRACE_VERSION = "riley.fault-extension-raw-trace.v1"
STABLE_DEFAULT_PROFILE = "stable-default"
INJECTABLE_BACKEND_ID = "riley-fault-injectable-subprocess-v1"
SANITIZER_LOGS = tuple(sorted(cuda_fault.SANITIZER_FILES))
MAX_JSON_BYTES = 8 * 1024 * 1024

# Gate E supplies real-device CUDA coverage for these four cases.  It is
# replayed from the raw archive rather than accepted from an attestation.
GATE_E_FAULT_CASES = (
    ("create-rollback-ambiguous", "create_rollback_ambiguity"),
    ("explicit-close-ambiguous", "explicit_close_ambiguity"),
    ("deferred-submission-error", "confirmed_completion_deferred_error"),
    ("completion-restore-ambiguous", "unconfirmed_completion_retained"),
)


@dataclass(frozen=True)
class ExtendedFaultCase:
    """One C02 §8 engine-level fault trace that Gate E does not cover."""

    case_id: str
    injection_point: str
    events: tuple[str, ...]
    terminal: dict[str, Any]


# These are intentionally raw state transitions, not ``passed`` claims.  A
# trace cannot qualify by merely listing the case IDs: every event and final
# externally visible state has to be present in this exact order.
EXTENDED_FAULT_CASES = (
    ExtendedFaultCase(
        case_id="post-kv-write-runtime-error",
        injection_point="post-kv-write-runtime-error",
        events=(
            "case-start",
            "kv-write-applied",
            "runtime-error-injected",
            "scheduler-commit-skipped",
            "output-suppressed",
            "kv-write-rolled-back",
            "terminal-error-emitted",
        ),
        terminal={
            "request_state": "failed",
            "output_published": False,
            "scheduler_commit": "not-attempted",
            "kv_state": "rolled-back",
            "worker_channel": "open",
            "terminal_events": 1,
        },
    ),
    ExtendedFaultCase(
        case_id="output-status-corruption-test-double",
        injection_point="output-status-corruption-test-double",
        events=(
            "case-start",
            "output-status-corrupted",
            "output-rejected",
            "scheduler-commit-skipped",
            "output-suppressed",
            "terminal-error-emitted",
        ),
        terminal={
            "request_state": "failed",
            "output_published": False,
            "scheduler_commit": "not-attempted",
            "kv_state": "not-retained",
            "worker_channel": "open",
            "terminal_events": 1,
        },
    ),
    ExtendedFaultCase(
        case_id="scheduler-commit-failure",
        injection_point="scheduler-commit-failure",
        events=(
            "case-start",
            "kv-write-applied",
            "scheduler-commit-attempted",
            "scheduler-commit-failed",
            "output-suppressed",
            "kv-write-rolled-back",
            "terminal-error-emitted",
        ),
        terminal={
            "request_state": "failed",
            "output_published": False,
            "scheduler_commit": "failed",
            "kv_state": "rolled-back",
            "worker_channel": "open",
            "terminal_events": 1,
        },
    ),
    ExtendedFaultCase(
        case_id="worker-channel-close-race",
        injection_point="worker-channel-close-race",
        events=(
            "case-start",
            "worker-channel-close-race-injected",
            "worker-channel-closed",
            "scheduler-commit-skipped",
            "output-suppressed",
            "terminal-error-emitted",
        ),
        terminal={
            "request_state": "failed",
            "output_published": False,
            "scheduler_commit": "not-attempted",
            "kv_state": "not-retained",
            "worker_channel": "closed",
            "terminal_events": 1,
        },
    ),
)

# ``fault_cases`` is a closed, explicit inventory in the semantic result.
# The first four are genuine GPU/sanitizer evidence; the engine-state cases
# are deliberately labelled as injectable synthetic evidence rather than
# being misrepresented as device-loss tests.
FAULT_CASES = tuple(
    (case_id, "real-gpu-sanitizer", check_id)
    for case_id, check_id in GATE_E_FAULT_CASES
) + tuple(
    (case.case_id, "injectable-synthetic", case.injection_point)
    for case in EXTENDED_FAULT_CASES
)
CHECK_NAMES = (
    "freeze-binding",
    "gate-e-replay",
    "stable-default-arm-binding",
    "gate-e-cuda-fault-binding",
    "cuda-fault-attestation-replay",
    "real-gpu-sanitizer-binding",
    "extended-fault-trace-binding",
    "extended-fault-trace-replay",
    "expanded-fault-case-coverage",
)

_OPEN_COMMON = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
_OPEN_DIRECTORY = _OPEN_COMMON | getattr(os, "O_DIRECTORY", 0)


class FaultExtensionReceiptError(qualification.QualificationError):
    """A fault-extension input cannot establish its semantic gate."""


class FaultExtensionReceiptIncomparable(qualification.IncomparableError):
    """An otherwise well-formed input binds a different immutable candidate."""


@dataclass(frozen=True)
class Descriptor:
    path: str
    sha256: str


@dataclass(frozen=True)
class ReplayInputs:
    source_archive: Descriptor
    release_binary: Descriptor
    release_bundle: Descriptor


@dataclass(frozen=True)
class FaultExtensionReceipt:
    candidate_id: str
    bindings: dict[str, str]
    replay_inputs: ReplayInputs
    cuda_fault_report: Descriptor
    cuda_fault_raw: Descriptor
    extended_faults_raw_trace: Descriptor


@dataclass(frozen=True)
class FaultExtensionCheckReport:
    """Closed evidence descriptors exposed for a later outer-gate replay."""

    candidate_id: str
    freeze_sha256: str
    base_release_candidate_report: Descriptor
    receipt: Descriptor
    bindings: dict[str, str]
    replay_inputs: ReplayInputs
    cuda_fault_report: Descriptor
    cuda_fault_raw: Descriptor
    extended_faults_raw_trace: Descriptor


def _raise(error_type: type[qualification.QualificationError], code: str, message: str) -> NoReturn:
    error = error_type(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _fail(code: str, message: str) -> NoReturn:
    _raise(FaultExtensionReceiptError, code, message)


def _incomparable(message: str) -> NoReturn:
    _raise(FaultExtensionReceiptIncomparable, "incomparable-binding", message)


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    return qualification._exact(value, fields, label)


def _sha256(value: Any, label: str) -> str:
    return qualification._sha256(value, label)


def _candidate_id(value: Any, label: str) -> str:
    candidate_id = qualification._string(value, label)
    if not qualification.release_candidate.CANDIDATE_ID_RE.fullmatch(candidate_id):
        _fail("invalid-candidate-id", f"{label} is not a valid RC candidate")
    return candidate_id


def _descriptor(value: Any, label: str) -> Descriptor:
    row = _exact(value, {"path", "sha256"}, label)
    return Descriptor(
        path=qualification._relative_path(row["path"], f"{label}.path"),
        sha256=_sha256(row["sha256"], f"{label}.sha256"),
    )


def _bindings(value: Any, label: str) -> dict[str, str]:
    row = _exact(
        value,
        {
            "freeze_sha256",
            "base_release_candidate_report_sha256",
            "configuration_profile",
            "configuration_sha256",
        },
        label,
    )
    if row["configuration_profile"] != STABLE_DEFAULT_PROFILE:
        _incomparable(f"{label}.configuration_profile is not {STABLE_DEFAULT_PROFILE}")
    return {
        "freeze_sha256": _sha256(row["freeze_sha256"], f"{label}.freeze_sha256"),
        "base_release_candidate_report_sha256": _sha256(
            row["base_release_candidate_report_sha256"],
            f"{label}.base_release_candidate_report_sha256",
        ),
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": _sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
    }


def _distinct_paths(paths: Sequence[str], label: str) -> None:
    if len(paths) != len(set(paths)):
        _fail("duplicate-evidence-path", f"{label} must not reuse a file path")


def validate_receipt(
    document: dict[str, Any],
    label: str = "fault extension receipt",
) -> FaultExtensionReceipt:
    """Parse a non-authoritative descriptor for a semantic raw replay.

    The input has no ``status`` or ``passed`` field.  A descriptor becomes
    evidence only after :func:`evaluate` has revalidated Gate E, replayed the
    referenced raw CUDA archive, and parsed the expanded C02 §8 raw trace.
    """

    row = _exact(
        document,
        {
            "schema_version",
            "candidate_id",
            "bindings",
            "replay_inputs",
            "cuda_fault",
            "extended_faults",
        },
        label,
    )
    if row["schema_version"] != RECEIPT_VERSION:
        _fail("unsupported-fault-extension-receipt-version", f"{label}.schema_version is unsupported")
    replay_row = _exact(
        row["replay_inputs"],
        {"source_archive", "release_binary", "release_bundle"},
        f"{label}.replay_inputs",
    )
    cuda_row = _exact(
        row["cuda_fault"],
        {"report", "raw_evidence"},
        f"{label}.cuda_fault",
    )
    extended_row = _exact(
        row["extended_faults"],
        {"raw_trace"},
        f"{label}.extended_faults",
    )
    replay_inputs = ReplayInputs(
        source_archive=_descriptor(
            replay_row["source_archive"], f"{label}.replay_inputs.source_archive"
        ),
        release_binary=_descriptor(
            replay_row["release_binary"], f"{label}.replay_inputs.release_binary"
        ),
        release_bundle=_descriptor(
            replay_row["release_bundle"], f"{label}.replay_inputs.release_bundle"
        ),
    )
    report = _descriptor(cuda_row["report"], f"{label}.cuda_fault.report")
    raw = _descriptor(cuda_row["raw_evidence"], f"{label}.cuda_fault.raw_evidence")
    raw_trace = _descriptor(
        extended_row["raw_trace"], f"{label}.extended_faults.raw_trace"
    )
    _distinct_paths(
        (
            replay_inputs.source_archive.path,
            replay_inputs.release_binary.path,
            replay_inputs.release_bundle.path,
            report.path,
            raw.path,
            raw_trace.path,
        ),
        f"{label} replay inputs",
    )
    return FaultExtensionReceipt(
        candidate_id=_candidate_id(row["candidate_id"], f"{label}.candidate_id"),
        bindings=_bindings(row["bindings"], f"{label}.bindings"),
        replay_inputs=replay_inputs,
        cuda_fault_report=report,
        cuda_fault_raw=raw,
        extended_faults_raw_trace=raw_trace,
    )


def _write_all(descriptor: int, contents: bytes, label: str) -> None:
    offset = 0
    while offset < len(contents):
        try:
            written = os.write(descriptor, contents[offset:])
        except OSError as error:
            _fail("snapshot-write-failed", f"{label} could not be snapshotted: {error}")
        if written <= 0:
            _fail("snapshot-write-failed", f"{label} could not be snapshotted")
        offset += written


def _safe_snapshot(
    evidence_root: Path,
    relative: str,
    *,
    expected_sha256: str | None,
    label: str,
    destination_root: Path,
    sequence: int,
    maximum_bytes: int,
    seen_paths: set[str],
    seen_file_ids: set[tuple[int, int]],
) -> tuple[Path, str]:
    """Snapshot one no-follow evidence file and bind its immutable digest."""

    if relative in seen_paths:
        _fail("duplicate-evidence-path", f"{label} reuses another evidence path")
    seen_paths.add(relative)
    pure = PurePosixPath(relative)
    try:
        root_before = evidence_root.lstat()
    except OSError as error:
        _fail("missing-evidence-root", f"cannot inspect evidence root: {error}")
    if not stat.S_ISDIR(root_before.st_mode):
        _fail("unsafe-evidence-root", "evidence root must be a real directory")

    root_fd = -1
    current_fd = -1
    file_fd = -1
    output_fd = -1
    try:
        root_fd = os.open(evidence_root, _OPEN_DIRECTORY)
        root_after = os.fstat(root_fd)
        if (root_before.st_dev, root_before.st_ino) != (root_after.st_dev, root_after.st_ino):
            _fail("raced-evidence-root", "evidence root changed while it was opened")
        current_fd = root_fd
        for component in pure.parts[:-1]:
            next_fd = os.open(component, _OPEN_DIRECTORY, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(pure.parts[-1], _OPEN_COMMON, dir_fd=current_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            _fail("unsafe-evidence-path", f"{label} must be a regular file")
        if before.st_size > maximum_bytes:
            _fail("input-too-large", f"{label} exceeds its {maximum_bytes}-byte bound")
        file_id = (before.st_dev, before.st_ino)
        if file_id in seen_file_ids:
            _fail("hard-link-evidence-alias", f"{label} aliases another evidence file")
        seen_file_ids.add(file_id)

        snapshot = destination_root / f"{sequence:02d}-{pure.name}"
        output_fd = os.open(
            snapshot,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                _fail("truncated-evidence", f"{label} changed while it was snapshotted")
            digest.update(chunk)
            _write_all(output_fd, chunk, label)
            remaining -= len(chunk)
        if os.read(file_fd, 1):
            _fail("mutated-evidence", f"{label} grew while it was snapshotted")
        os.fsync(output_fd)
        after = os.fstat(file_fd)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            _fail("mutated-evidence", f"{label} changed while it was snapshotted")
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            _fail("evidence-sha256-mismatch", f"{label} does not match its declared SHA-256")
        return snapshot, actual_sha256
    except FaultExtensionReceiptError:
        raise
    except OSError as error:
        _fail("unsafe-evidence-path", f"{label} cannot be opened and snapshotted safely: {error}")
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        if file_fd >= 0:
            os.close(file_fd)
        if current_fd >= 0 and current_fd != root_fd:
            os.close(current_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _parse_json_snapshot(
    snapshot: Path,
    label: str,
    *,
    require_canonical: bool,
) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = snapshot.read_bytes()
    except OSError as error:
        _fail("snapshot-read-failed", f"cannot read {label} snapshot: {error}")
    document = qualification._parse_document(raw, label)
    if require_canonical and raw != qualification.canonical_json_bytes(document):
        _fail("noncanonical-evidence-json", f"{label} must use exact canonical JSON bytes")
    return raw, document


def _expected_bindings(
    frozen: qualification.FrozenCandidate,
    *,
    freeze_sha256: str,
    base_report_sha256: str,
) -> dict[str, str]:
    return {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }


def _validate_gate_e_bindings(
    receipt: FaultExtensionReceipt,
    *,
    frozen: qualification.FrozenCandidate,
    base_report: dict[str, Any],
) -> None:
    if receipt.replay_inputs.source_archive.sha256 != frozen.source["archive_sha256"]:
        _incomparable("fault receipt source archive differs from the frozen candidate")
    if receipt.replay_inputs.release_binary.sha256 != frozen.release["binary_sha256"]:
        _incomparable("fault receipt release binary differs from the frozen candidate")
    if receipt.replay_inputs.release_bundle.sha256 != frozen.release["bundle_sha256"]:
        _incomparable("fault receipt release bundle differs from the frozen candidate")
    evidence_hashes = base_report["bindings"]["evidence_sha256"]
    if receipt.cuda_fault_report.sha256 != evidence_hashes["cuda_fault"]:
        _incomparable("fault receipt CUDA report differs from replayed Gate E evidence")
    if receipt.cuda_fault_raw.sha256 != evidence_hashes["cuda_fault_raw"]:
        _incomparable("fault receipt CUDA raw evidence differs from replayed Gate E evidence")


def _reject_frozen_output_aliases(
    receipt: FaultExtensionReceipt,
    frozen: qualification.FrozenCandidate,
) -> None:
    """Raw evidence must not masquerade as a freeze-declared decision file."""

    reserved_paths = _frozen_output_paths(frozen)
    descriptors = (
        ("fault replay source archive", receipt.replay_inputs.source_archive),
        ("fault replay release binary", receipt.replay_inputs.release_binary),
        ("fault replay release bundle", receipt.replay_inputs.release_bundle),
        ("Gate E CUDA fault attestation", receipt.cuda_fault_report),
        ("Gate E CUDA fault raw evidence", receipt.cuda_fault_raw),
        ("expanded C02 fault raw trace", receipt.extended_faults_raw_trace),
    )
    for label, descriptor in descriptors:
        if descriptor.path in reserved_paths:
            _fail(
                "reserved-output-path-collision",
                f"{label} reuses a freeze-declared final report or semantic receipt path",
            )


def _frozen_output_paths(frozen: qualification.FrozenCandidate) -> set[str]:
    return {
        frozen.final_manifest.path,
        frozen.final_report.path,
        *(descriptor.path for descriptor in frozen.receipts.values()),
    }


def _validate_replayed_cuda_attestation(
    attestation: dict[str, Any]) -> None:
    row = _exact(
        attestation,
        {"schema_version", "gate", "status", "source", "raw_evidence_sha256", "checks"},
        "replayed CUDA fault attestation",
    )
    if row["schema_version"] != cuda_fault.ATTESTATION_VERSION:
        _fail("invalid-cuda-fault-attestation", "replayed CUDA attestation version drifted")
    if row["gate"] != cuda_fault.GATE or row["status"] != "passed":
        _fail("invalid-cuda-fault-attestation", "replayed CUDA attestation did not pass its reviewed gate")
    checks = row["checks"]
    if not isinstance(checks, list) or len(checks) != len(cuda_fault.CHECK_IDS):
        _fail("invalid-cuda-fault-attestation", "replayed CUDA check inventory drifted")
    observed: list[str] = []
    for index, check in enumerate(checks):
        item = _exact(check, {"id", "passed"}, f"replayed CUDA fault attestation.checks[{index}]")
        if item["passed"] is not True:
            _fail("cuda-fault-check-failed", f"CUDA check {item['id']!r} did not pass")
        observed.append(qualification._string(item["id"], f"replayed CUDA fault attestation.checks[{index}].id"))
    if tuple(observed) != tuple(sorted(cuda_fault.CHECK_IDS)):
        _fail("invalid-cuda-fault-attestation", "replayed CUDA check inventory drifted")
    for _, check_id in GATE_E_FAULT_CASES:
        if check_id not in observed:
            _fail("fault-case-coverage-missing", f"CUDA attestation omits {check_id}")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail("invalid-raw-trace", f"{label} must be a positive integer")
    return value


def _raw_trace_bindings(value: Any, label: str) -> dict[str, str]:
    row = _exact(
        value,
        {
            "freeze_sha256",
            "base_release_candidate_report_sha256",
            "configuration_profile",
            "configuration_sha256",
            "source_revision",
            "source_archive_sha256",
            "release_binary_sha256",
            "release_bundle_sha256",
            "release_image_id",
            "cuda_build_image_id",
            "gate_e_cuda_fault_raw_sha256",
        },
        label,
    )
    if row["configuration_profile"] != STABLE_DEFAULT_PROFILE:
        _incomparable(f"{label}.configuration_profile is not {STABLE_DEFAULT_PROFILE}")
    source_revision = qualification._string(row["source_revision"], f"{label}.source_revision")
    if not qualification.GIT_RE.fullmatch(source_revision) or source_revision == "0" * 40:
        _fail("invalid-raw-trace", f"{label}.source_revision must be a full lowercase Git SHA")
    return {
        "freeze_sha256": _sha256(row["freeze_sha256"], f"{label}.freeze_sha256"),
        "base_release_candidate_report_sha256": _sha256(
            row["base_release_candidate_report_sha256"],
            f"{label}.base_release_candidate_report_sha256",
        ),
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": _sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
        "source_revision": source_revision,
        "source_archive_sha256": _sha256(
            row["source_archive_sha256"], f"{label}.source_archive_sha256"
        ),
        "release_binary_sha256": _sha256(
            row["release_binary_sha256"], f"{label}.release_binary_sha256"
        ),
        "release_bundle_sha256": _sha256(
            row["release_bundle_sha256"], f"{label}.release_bundle_sha256"
        ),
        "release_image_id": qualification._image(
            row["release_image_id"], f"{label}.release_image_id"
        ),
        "cuda_build_image_id": qualification._image(
            row["cuda_build_image_id"], f"{label}.cuda_build_image_id"
        ),
        "gate_e_cuda_fault_raw_sha256": _sha256(
            row["gate_e_cuda_fault_raw_sha256"], f"{label}.gate_e_cuda_fault_raw_sha256"
        ),
    }


def _expected_raw_trace_bindings(
    frozen: qualification.FrozenCandidate,
    *,
    receipt: FaultExtensionReceipt,
) -> dict[str, str]:
    return {
        **receipt.bindings,
        "source_revision": frozen.source["git_revision"],
        "source_archive_sha256": frozen.source["archive_sha256"],
        "release_binary_sha256": frozen.release["binary_sha256"],
        "release_bundle_sha256": frozen.release["bundle_sha256"],
        "release_image_id": frozen.release["image_id"],
        "cuda_build_image_id": frozen.images["cuda"],
        "gate_e_cuda_fault_raw_sha256": receipt.cuda_fault_raw.sha256,
    }


def _validate_trace_events(value: Any, specification: ExtendedFaultCase, label: str) -> None:
    if not isinstance(value, list) or len(value) != len(specification.events):
        _fail("expanded-fault-trace-mismatch", f"{label}.events has the wrong event inventory")
    observed: list[str] = []
    for index, event in enumerate(value, start=1):
        row = _exact(event, {"ordinal", "event"}, f"{label}.events[{index - 1}]")
        if _positive_int(row["ordinal"], f"{label}.events[{index - 1}].ordinal") != index:
            _fail("expanded-fault-trace-mismatch", f"{label}.events ordinals must start at one")
        observed.append(
            qualification._string(row["event"], f"{label}.events[{index - 1}].event")
        )
    if tuple(observed) != specification.events:
        _fail("expanded-fault-trace-mismatch", f"{label}.events do not prove the required transition order")


def _validate_extended_fault_trace(
    document: dict[str, Any],
    *,
    frozen: qualification.FrozenCandidate,
    receipt: FaultExtensionReceipt,
    cuda_environment: dict[str, str],
    cuda_files: dict[str, bytes],
) -> None:
    """Parse every C02 §8 raw trace; never accept a synthetic ``passed`` bit.

    Device/sanitizer and engine-state injection are intentionally separate:
    the former is re-read from Gate E's raw CUDA archive, while all four
    engine-state cases must be subprocess-isolated injectable traces.
    """

    row = _exact(
        document,
        {
            "schema_version",
            "candidate_id",
            "bindings",
            "real_gpu_sanitizer",
            "injectable_backend",
            "cases",
        },
        "expanded fault raw trace",
    )
    if row["schema_version"] != RAW_TRACE_VERSION:
        _fail("unsupported-expanded-fault-trace-version", "expanded fault raw trace version is unsupported")
    if _candidate_id(row["candidate_id"], "expanded fault raw trace.candidate_id") != frozen.candidate_id:
        _incomparable("expanded fault raw trace belongs to another candidate")
    if _raw_trace_bindings(row["bindings"], "expanded fault raw trace.bindings") != _expected_raw_trace_bindings(
        frozen,
        receipt=receipt,
    ):
        _incomparable("expanded fault raw trace immutable candidate bindings drifted")

    sanitizer = _exact(
        row["real_gpu_sanitizer"],
        {"execution_class", "raw_evidence_sha256", "sanitizer_logs"},
        "expanded fault raw trace.real_gpu_sanitizer",
    )
    if sanitizer["execution_class"] != "real-gpu-sanitizer":
        _fail("invalid-real-gpu-sanitizer-class", "expanded trace must label Gate E evidence real-gpu-sanitizer")
    if _sha256(
        sanitizer["raw_evidence_sha256"],
        "expanded fault raw trace.real_gpu_sanitizer.raw_evidence_sha256",
    ) != receipt.cuda_fault_raw.sha256:
        _incomparable("expanded fault raw trace points at different Gate E raw CUDA evidence")
    if sanitizer["sanitizer_logs"] != list(SANITIZER_LOGS):
        _fail("invalid-real-gpu-sanitizer-logs", "expanded trace must identify the exact Gate E sanitizer logs")
    if cuda_environment.get("compute_sanitizer") != "1" or any(
        name not in cuda_files for name in SANITIZER_LOGS
    ):
        _fail(
            "missing-real-gpu-sanitizer-evidence",
            "Gate E raw CUDA evidence must contain real GPU compute-sanitizer logs",
        )

    backend = _exact(
        row["injectable_backend"],
        {"execution_class", "backend_id", "subprocess_isolation"},
        "expanded fault raw trace.injectable_backend",
    )
    if backend["execution_class"] != "injectable-synthetic":
        _fail("invalid-injectable-backend-class", "engine-state faults must be labelled injectable-synthetic")
    if backend["backend_id"] != INJECTABLE_BACKEND_ID:
        _fail("invalid-injectable-backend", "expanded trace uses an unreviewed injectable backend")
    if backend["subprocess_isolation"] is not True:
        _fail("missing-subprocess-isolation", "injectable engine-state faults must be subprocess isolated")

    cases = row["cases"]
    if not isinstance(cases, list) or len(cases) != len(EXTENDED_FAULT_CASES):
        _fail("expanded-fault-case-inventory", "expanded trace must contain exactly four C02 engine fault cases")
    child_pids: set[int] = set()
    for index, (case, specification) in enumerate(zip(cases, EXTENDED_FAULT_CASES)):
        label = f"expanded fault raw trace.cases[{index}]"
        item = _exact(
            case,
            {"case_id", "execution_class", "injection_point", "subprocess", "events", "terminal"},
            label,
        )
        if qualification._string(item["case_id"], f"{label}.case_id") != specification.case_id:
            _fail("expanded-fault-case-inventory", f"{label}.case_id is not in the reviewed order")
        if item["execution_class"] != "injectable-synthetic":
            _fail("invalid-injectable-backend-class", f"{label} is not labelled injectable-synthetic")
        if qualification._string(item["injection_point"], f"{label}.injection_point") != specification.injection_point:
            _fail("expanded-fault-trace-mismatch", f"{label}.injection_point drifted")
        subprocess = _exact(item["subprocess"], {"parent_pid", "child_pid", "exit_code"}, f"{label}.subprocess")
        parent_pid = _positive_int(subprocess["parent_pid"], f"{label}.subprocess.parent_pid")
        child_pid = _positive_int(subprocess["child_pid"], f"{label}.subprocess.child_pid")
        if parent_pid == child_pid or child_pid in child_pids:
            _fail("missing-subprocess-isolation", f"{label} does not prove an isolated child process")
        child_pids.add(child_pid)
        if (
            isinstance(subprocess["exit_code"], bool)
            or not isinstance(subprocess["exit_code"], int)
            or subprocess["exit_code"] != 0
        ):
            _fail("expanded-fault-subprocess-failed", f"{label}.subprocess did not exit cleanly")
        _validate_trace_events(item["events"], specification, label)
        terminal = _exact(item["terminal"], set(specification.terminal), f"{label}.terminal")
        if (
            not isinstance(terminal["output_published"], bool)
            or isinstance(terminal["terminal_events"], bool)
            or not isinstance(terminal["terminal_events"], int)
        ):
            _fail("expanded-fault-trace-mismatch", f"{label}.terminal has non-canonical value types")
        if terminal != specification.terminal:
            _fail("expanded-fault-trace-mismatch", f"{label}.terminal does not prove safe terminal state")


def _empty_report() -> dict[str, Any]:
    return {
        "schema_version": CHECK_REPORT_VERSION,
        "status": "failed",
        "passed": False,
        "candidate_id": None,
        "freeze_sha256": None,
        "base_release_candidate_report": None,
        "receipt": None,
        "bindings": None,
        "replay_inputs": None,
        "cuda_fault_report": None,
        "cuda_fault_raw": None,
        "extended_faults_raw_trace": None,
        "evidence_classes": [],
        "fault_cases": [],
        "checks": [],
        "reason_codes": [],
    }


def evaluate(
    freeze_path: Path,
    evidence_root: Path,
    receipt_path: Path | str,
    *,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    """Revalidate a C02 fault-extension receipt without running CUDA."""

    report = _empty_report()
    try:
        trusted_freeze_sha256 = _sha256(expected_freeze_sha256, "--expected-freeze-sha256")
        freeze_raw = qualification._read_regular_path(freeze_path, "freeze manifest")
        freeze_sha256 = hashlib.sha256(freeze_raw).hexdigest()
        report["freeze_sha256"] = freeze_sha256
        if freeze_sha256 != trusted_freeze_sha256:
            _fail("candidate-sha-mismatch", "freeze manifest SHA-256 differs from trusted input")
        frozen = qualification._validate_freeze(
            qualification._parse_document(freeze_raw, "freeze manifest")
        )
        report["candidate_id"] = frozen.candidate_id

        base_raw, base_report_sha256 = qualification.revalidate_base_release_candidate(
            frozen, freeze_sha256, evidence_root
        )
        if hashlib.sha256(base_raw).hexdigest() != base_report_sha256:
            _fail("base-report-replay-digest-mismatch", "Gate E replay returned inconsistent bytes/digest")
        base_report = qualification._parse_document(base_raw, "final release candidate report")
        # The outer helper normally did this already.  Repeat it here so an
        # injected/mock replay result is never accepted as an opaque passed
        # report by this semantic checker.
        qualification._validate_base_report_shape(base_report, frozen)
        report["base_release_candidate_report"] = {
            "path": frozen.final_report.path,
            "sha256": base_report_sha256,
        }

        receipt_relative = qualification._relative_path(str(receipt_path), "fault receipt path")
        if receipt_relative in _frozen_output_paths(frozen):
            _fail(
                "reserved-output-path-collision",
                "raw fault receipt must not replace a freeze-declared semantic report or receipt",
            )

        with tempfile.TemporaryDirectory(prefix="riley-fault-extension-") as temporary:
            snapshots = Path(temporary)
            seen_paths: set[str] = set()
            seen_file_ids: set[tuple[int, int]] = set()
            receipt_snapshot, receipt_sha256 = _safe_snapshot(
                evidence_root,
                receipt_relative,
                expected_sha256=None,
                label="fault extension receipt",
                destination_root=snapshots,
                sequence=1,
                maximum_bytes=MAX_JSON_BYTES,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            _, receipt_document = _parse_json_snapshot(
                receipt_snapshot,
                "fault extension receipt",
                require_canonical=True,
            )
            receipt = validate_receipt(receipt_document)
            if receipt.candidate_id != frozen.candidate_id:
                _incomparable("fault receipt belongs to another candidate")
            if receipt.bindings != _expected_bindings(
                frozen,
                freeze_sha256=freeze_sha256,
                base_report_sha256=base_report_sha256,
            ):
                _incomparable("fault receipt immutable/frozen stable-default bindings drifted")
            _reject_frozen_output_aliases(receipt, frozen)
            _validate_gate_e_bindings(receipt, frozen=frozen, base_report=base_report)

            receipt_descriptor = Descriptor(receipt_relative, receipt_sha256)
            report["receipt"] = {
                "path": receipt_descriptor.path,
                "sha256": receipt_descriptor.sha256,
            }
            report["bindings"] = receipt.bindings

            source_snapshot, _ = _safe_snapshot(
                evidence_root,
                receipt.replay_inputs.source_archive.path,
                expected_sha256=receipt.replay_inputs.source_archive.sha256,
                label="fault replay source archive",
                destination_root=snapshots,
                sequence=2,
                maximum_bytes=cuda_fault.MAX_SOURCE_ARCHIVE_BYTES,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            binary_snapshot, _ = _safe_snapshot(
                evidence_root,
                receipt.replay_inputs.release_binary.path,
                expected_sha256=receipt.replay_inputs.release_binary.sha256,
                label="fault replay release binary",
                destination_root=snapshots,
                sequence=3,
                maximum_bytes=cuda_fault.MAX_RELEASE_BINARY_BYTES,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            bundle_snapshot, _ = _safe_snapshot(
                evidence_root,
                receipt.replay_inputs.release_bundle.path,
                expected_sha256=receipt.replay_inputs.release_bundle.sha256,
                label="fault replay release bundle",
                destination_root=snapshots,
                sequence=4,
                maximum_bytes=cuda_fault.MAX_SOURCE_ARCHIVE_BYTES,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            cuda_report_snapshot, _ = _safe_snapshot(
                evidence_root,
                receipt.cuda_fault_report.path,
                expected_sha256=receipt.cuda_fault_report.sha256,
                label="Gate E CUDA fault attestation",
                destination_root=snapshots,
                sequence=5,
                maximum_bytes=MAX_JSON_BYTES,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            cuda_raw_snapshot, _ = _safe_snapshot(
                evidence_root,
                receipt.cuda_fault_raw.path,
                expected_sha256=receipt.cuda_fault_raw.sha256,
                label="Gate E CUDA fault raw evidence",
                destination_root=snapshots,
                sequence=6,
                maximum_bytes=cuda_fault.MAX_RAW_EVIDENCE_ARCHIVE_BYTES,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            extended_trace_snapshot, extended_trace_sha256 = _safe_snapshot(
                evidence_root,
                receipt.extended_faults_raw_trace.path,
                expected_sha256=receipt.extended_faults_raw_trace.sha256,
                label="expanded C02 fault raw trace",
                destination_root=snapshots,
                sequence=7,
                maximum_bytes=MAX_JSON_BYTES,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            _, submitted_cuda_report = _parse_json_snapshot(
                cuda_report_snapshot,
                "Gate E CUDA fault attestation",
                # Gate E's known CUDA producer uses release_common's reviewed
                # pretty, newline-terminated JSON encoding.  Its descriptor
                # is hash-bound and its parsed value is exact-compared to a
                # fresh replay below, so accepting that production encoding
                # does not turn it into a self-authored pass claim.
                require_canonical=False,
            )
            replayed_cuda_report = cuda_fault.replay_raw_evidence(
                cuda_raw_snapshot,
                source_revision=frozen.source["git_revision"],
                source_archive=source_snapshot,
                build_image_id=frozen.images["cuda"],
                release_binary=binary_snapshot,
                release_bundle=bundle_snapshot,
                release_image_id=frozen.release["image_id"],
            )
            _validate_replayed_cuda_attestation(replayed_cuda_report)
            if submitted_cuda_report != replayed_cuda_report:
                _fail(
                    "cuda-fault-attestation-replay-mismatch",
                    "submitted Gate E CUDA attestation differs from raw semantic replay",
                )
            cuda_files, cuda_environment, raw_cuda_sha256 = cuda_fault.load_raw_evidence_archive(
                cuda_raw_snapshot
            )
            if raw_cuda_sha256 != receipt.cuda_fault_raw.sha256:
                _fail(
                    "cuda-fault-raw-replay-digest-mismatch",
                    "replayed Gate E raw CUDA archive differs from its receipt descriptor",
                )
            _, extended_trace_document = _parse_json_snapshot(
                extended_trace_snapshot,
                "expanded C02 fault raw trace",
                require_canonical=True,
            )
            _validate_extended_fault_trace(
                extended_trace_document,
                frozen=frozen,
                receipt=receipt,
                cuda_environment=cuda_environment,
                cuda_files=cuda_files,
            )

        report.update(
            {
                "status": "passed",
                "passed": True,
                "replay_inputs": {
                    "source_archive": {
                        "path": receipt.replay_inputs.source_archive.path,
                        "sha256": receipt.replay_inputs.source_archive.sha256,
                    },
                    "release_binary": {
                        "path": receipt.replay_inputs.release_binary.path,
                        "sha256": receipt.replay_inputs.release_binary.sha256,
                    },
                    "release_bundle": {
                        "path": receipt.replay_inputs.release_bundle.path,
                        "sha256": receipt.replay_inputs.release_bundle.sha256,
                    },
                },
                "cuda_fault_report": {
                    "path": receipt.cuda_fault_report.path,
                    "sha256": receipt.cuda_fault_report.sha256,
                },
                "cuda_fault_raw": {
                    "path": receipt.cuda_fault_raw.path,
                    "sha256": receipt.cuda_fault_raw.sha256,
                },
                "extended_faults_raw_trace": {
                    "path": receipt.extended_faults_raw_trace.path,
                    "sha256": extended_trace_sha256,
                },
                "evidence_classes": [
                    {
                        "execution_class": "real-gpu-sanitizer",
                        "source": "gate-e-cuda-fault-raw",
                        "sha256": receipt.cuda_fault_raw.sha256,
                        "passed": True,
                    },
                    {
                        "execution_class": "injectable-synthetic",
                        "source": "expanded-fault-raw-trace",
                        "sha256": extended_trace_sha256,
                        "passed": True,
                    },
                ],
                "fault_cases": [
                    {
                        "case_id": case_id,
                        "execution_class": execution_class,
                        "semantic_check": semantic_check,
                        "passed": True,
                    }
                    for case_id, execution_class, semantic_check in FAULT_CASES
                ],
                "checks": [{"name": name, "passed": True} for name in CHECK_NAMES],
            }
        )
    except qualification.IncomparableError as error:
        report["status"] = "incomparable"
        report["reason_codes"] = [getattr(error, "reason_code", "incomparable-binding")]
    except qualification.GateFailure as error:
        report["reason_codes"] = [getattr(error, "reason_code", "gate-failed")]
    except (OSError, qualification.QualificationError, cuda_fault.CudaFaultEvidenceError) as error:
        report["reason_codes"] = [getattr(error, "reason_code", "invalid-input")]
    return report


def validate_check_report(document: dict[str, Any]) -> FaultExtensionCheckReport:
    """Validate a passed report before a future outer checker reruns it.

    This only exposes closed descriptors; it intentionally does not make the
    submitted report authoritative.  A caller must invoke :func:`evaluate`
    again and exact-compare the result.
    """

    row = _exact(
        document,
        {
            "schema_version",
            "status",
            "passed",
            "candidate_id",
            "freeze_sha256",
            "base_release_candidate_report",
            "receipt",
            "bindings",
            "replay_inputs",
            "cuda_fault_report",
            "cuda_fault_raw",
            "extended_faults_raw_trace",
            "evidence_classes",
            "fault_cases",
            "checks",
            "reason_codes",
        },
        "fault extension check report",
    )
    if row["schema_version"] != CHECK_REPORT_VERSION:
        _fail("unsupported-fault-extension-check-report-version", "check report schema_version is unsupported")
    if row["status"] != "passed" or row["passed"] is not True:
        _fail("fault-extension-check-not-passed", "check report must be passed before outer revalidation")
    if row["reason_codes"] != []:
        _fail("invalid-fault-extension-check-report", "a passed check report must have no reason codes")
    replay_row = _exact(
        row["replay_inputs"],
        {"source_archive", "release_binary", "release_bundle"},
        "fault extension check report.replay_inputs",
    )
    replay_inputs = ReplayInputs(
        source_archive=_descriptor(
            replay_row["source_archive"], "fault extension check report.replay_inputs.source_archive"
        ),
        release_binary=_descriptor(
            replay_row["release_binary"], "fault extension check report.replay_inputs.release_binary"
        ),
        release_bundle=_descriptor(
            replay_row["release_bundle"], "fault extension check report.replay_inputs.release_bundle"
        ),
    )
    descriptors = (
        _descriptor(row["base_release_candidate_report"], "fault extension check report.base_release_candidate_report"),
        _descriptor(row["receipt"], "fault extension check report.receipt"),
        replay_inputs.source_archive,
        replay_inputs.release_binary,
        replay_inputs.release_bundle,
        _descriptor(row["cuda_fault_report"], "fault extension check report.cuda_fault_report"),
        _descriptor(row["cuda_fault_raw"], "fault extension check report.cuda_fault_raw"),
        _descriptor(
            row["extended_faults_raw_trace"],
            "fault extension check report.extended_faults_raw_trace",
        ),
    )
    _distinct_paths(tuple(descriptor.path for descriptor in descriptors), "fault extension check report")
    evidence_classes = row["evidence_classes"]
    if not isinstance(evidence_classes, list) or len(evidence_classes) != 2:
        _fail("invalid-fault-extension-check-report", "check report has an invalid evidence-class inventory")
    normalized_evidence_classes: list[dict[str, Any]] = []
    for index, evidence_class in enumerate(evidence_classes):
        item = _exact(
            evidence_class,
            {"execution_class", "source", "sha256", "passed"},
            f"fault extension check report.evidence_classes[{index}]",
        )
        if item["passed"] is not True:
            _fail("fault-extension-check-not-passed", f"evidence class {item['execution_class']!r} did not pass")
        normalized_evidence_classes.append(
            {
                "execution_class": qualification._string(
                    item["execution_class"],
                    f"fault extension check report.evidence_classes[{index}].execution_class",
                ),
                "source": qualification._string(
                    item["source"],
                    f"fault extension check report.evidence_classes[{index}].source",
                ),
                "sha256": _sha256(
                    item["sha256"],
                    f"fault extension check report.evidence_classes[{index}].sha256",
                ),
                "passed": True,
            }
        )
    expected_evidence_classes = [
        {
            "execution_class": "real-gpu-sanitizer",
            "source": "gate-e-cuda-fault-raw",
            "sha256": descriptors[6].sha256,
            "passed": True,
        },
        {
            "execution_class": "injectable-synthetic",
            "source": "expanded-fault-raw-trace",
            "sha256": descriptors[7].sha256,
            "passed": True,
        },
    ]
    if normalized_evidence_classes != expected_evidence_classes:
        _fail("invalid-fault-extension-check-report", "check report evidence classes drifted")
    checks = row["checks"]
    if not isinstance(checks, list) or len(checks) != len(CHECK_NAMES):
        _fail("invalid-fault-extension-check-report", "check report has an invalid check inventory")
    check_names: list[str] = []
    for index, check in enumerate(checks):
        item = _exact(check, {"name", "passed"}, f"fault extension check report.checks[{index}]")
        if item["passed"] is not True:
            _fail("fault-extension-check-not-passed", f"check {item['name']!r} did not pass")
        check_names.append(qualification._string(item["name"], f"fault extension check report.checks[{index}].name"))
    if tuple(check_names) != CHECK_NAMES:
        _fail("invalid-fault-extension-check-report", "check report check inventory drifted")
    cases = row["fault_cases"]
    if not isinstance(cases, list) or len(cases) != len(FAULT_CASES):
        _fail("invalid-fault-extension-check-report", "check report has an invalid fault-case inventory")
    normalized_cases: list[tuple[str, str, str]] = []
    for index, case in enumerate(cases):
        item = _exact(
            case,
            {"case_id", "execution_class", "semantic_check", "passed"},
            f"fault extension check report.fault_cases[{index}]",
        )
        if item["passed"] is not True:
            _fail("fault-extension-check-not-passed", f"fault case {item['case_id']!r} did not pass")
        normalized_cases.append(
            (
                qualification._string(item["case_id"], f"fault extension check report.fault_cases[{index}].case_id"),
                qualification._string(
                    item["execution_class"],
                    f"fault extension check report.fault_cases[{index}].execution_class",
                ),
                qualification._string(
                    item["semantic_check"],
                    f"fault extension check report.fault_cases[{index}].semantic_check",
                ),
            )
        )
    if tuple(normalized_cases) != FAULT_CASES:
        _fail("invalid-fault-extension-check-report", "check report fault-case inventory drifted")
    return FaultExtensionCheckReport(
        candidate_id=_candidate_id(row["candidate_id"], "fault extension check report.candidate_id"),
        freeze_sha256=_sha256(row["freeze_sha256"], "fault extension check report.freeze_sha256"),
        base_release_candidate_report=descriptors[0],
        receipt=descriptors[1],
        bindings=_bindings(row["bindings"], "fault extension check report.bindings"),
        replay_inputs=replay_inputs,
        cuda_fault_report=descriptors[5],
        cuda_fault_raw=descriptors[6],
        extended_faults_raw_trace=descriptors[7],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        required=True,
        help="relative fault-extension input path below --evidence-root",
    )
    parser.add_argument("--report", type=Path, help="create-only semantic check-report output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = evaluate(
        arguments.freeze,
        arguments.evidence_root,
        arguments.receipt,
        expected_freeze_sha256=arguments.expected_freeze_sha256,
    )
    if arguments.report is not None:
        try:
            qualification._write_create_only(arguments.report, report)
        except qualification.QualificationError as error:
            print(str(error), file=sys.stderr)
            return 2
    print(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
