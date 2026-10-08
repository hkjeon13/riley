#!/usr/bin/env python3
"""Replay RC3 canonical optimizer E0 through held Gate E evidence FDs.

This is one component of the closed four-gate RC3 Gate E semantic closure. It
replays the frozen/input inventory before and after work, snapshots only the
three optimizer leaves through caller-held FDs, and gives the legacy optimizer
checker only those private scratch paths. The required immutable optimizer
image ID is an explicit external trusted input; it is never learned from the
self-authored optimizer report or raw receipt.

Success establishes only optimizer canonical-E0. Native E0, Python-free,
performance, soak, aggregate Gate E, semantic receipt, qualification, and
deployment are deliberately outside this component's authority.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, NoReturn, Sequence, TypeVar


_BYTECODE_DISABLED_AT_STARTUP = bool(sys.flags.dont_write_bytecode)
_BYTECODE_DISABLED_ON_MODULE_ENTRY = sys.dont_write_bytecode
sys.dont_write_bytecode = True

import provenance_v2_common as common
import rc3_frozen_candidate_common as frozen
import rc3_frozen_candidate_topology as topology
import replay_rc3_frozen_candidate_v1 as frozen_replayer
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs
import optimizer_e0_semantic_contract as optimizer_contract


REPLAY_VERSION = "riley.rc3-gate-e-optimizer-e0-semantic-replay.v1"
SCOPE = "gate-e-optimizer-e0-semantic-component-only"
AUTHORITY = "gate-e-optimizer-e0-semantic-replay-only"
OPTIMIZER_GATE_ID = "pr15-iteration-command-batch-exact-v1"
OPTIMIZER_RECEIPT_VERSION = "riley.optimizer-execution-receipt.v3"
FIXED37_GATE_ID = "pr16-fixed37-production-batch-e0-v1"
EXPECTED_FINAL_REPORT_CONTRACT_POLICY_SHA256 = (
    "3efab239fa03631f10496109a4b04da1d6ef3caf2d9e0dfb1148138d3e0b9996"
)
MAX_OPTIMIZER_RAW_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024 + 4 * 1024 * 1024
MAX_OPTIMIZER_RAW_CONTENT_BYTES = 2 * 1024 * 1024 * 1024
MAX_OPTIMIZER_REPORT_BYTES = 8 * 1024 * 1024
MAX_PROFILE_BINARY_BYTES = 512 * 1024 * 1024
MAX_OPTIMIZER_SCRATCH_BYTES = (
    MAX_OPTIMIZER_RAW_ARCHIVE_BYTES
    + MAX_OPTIMIZER_REPORT_BYTES
    + MAX_PROFILE_BINARY_BYTES
)
EXTERNAL_SCRATCH_PARENT = Path("/var/tmp")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

CHECK_NAMES = (
    "closed-gate-e-input-inventory-replayed-before-optimizer-e0",
    "externally-supplied-optimizer-build-image-bound",
    "frozen-source-archive-bound-to-optimizer-e0",
    "optimizer-e0-raw-report-and-profile-replayed",
    "optimizer-e0-component-does-not-aggregate-gate-e",
)

NOT_ESTABLISHED = {
    "native_e0": "not-established",
    "optimizer_build_image_review": "not-established",
    "python_free": "not-established",
    "performance": "not-established",
    "soak": "not-established",
    "gate_e_pass": "not-established",
    "semantic_receipt": "not-established",
    "qualification": "not-established",
    "deployment": "not-established",
}

T = TypeVar("T")


class OptimizerE0ReplayError(ValueError):
    """The optimizer canonical-E0 component cannot be replayed safely."""


@dataclass(frozen=True)
class _OptimizerE0Artifacts:
    report: common.EvidenceDescriptor
    raw_evidence: common.EvidenceDescriptor
    profile_binary: common.EvidenceDescriptor


@dataclass(frozen=True)
class _ScratchSnapshots:
    root: Path
    root_fd: int
    descriptors: Mapping[str, common.EvidenceDescriptor]
    paths: Mapping[str, Path]
    root_identity: tuple[int, ...]
    leaf_identities: Mapping[str, tuple[int, ...]]


def _fail(code: str, message: str) -> NoReturn:
    error = OptimizerE0ReplayError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _frozen(call: Callable[[], T]) -> T:
    try:
        return call()
    except frozen.FrozenCandidateError as error:
        _fail(getattr(error, "reason_code", "invalid-frozen-candidate"), str(error))


def _topology(call: Callable[[], T]) -> T:
    try:
        return call()
    except topology.FrozenCandidateTopologyError as error:
        _fail(getattr(error, "reason_code", "unsafe-gate-e-topology"), str(error))


def _gate(call: Callable[[], T]) -> T:
    try:
        return call()
    except gate_inputs.GateEInventoryReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-gate-e-inventory"), str(error))


def _frozen_replay(call: Callable[[], T]) -> T:
    try:
        return call()
    except frozen_replayer.FrozenCandidateReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-frozen-candidate"), str(error))


def _require_bytecode_cache_disabled() -> None:
    if not (
        _BYTECODE_DISABLED_AT_STARTUP and _BYTECODE_DISABLED_ON_MODULE_ENTRY
    ):
        _fail(
            "bytecode-cache-write-not-permitted",
            "invoke this replayer with python3 -B or PYTHONDONTWRITEBYTECODE=1",
        )


def _shared_lock(directory_fd: int, label: str) -> None:
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("evidence-root-lock-unavailable", f"cannot acquire shared {label} lock: {error}")


def _unlock_quietly(directory_fd: int | None) -> None:
    if directory_fd is not None:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(directory_fd: int | None) -> None:
    if directory_fd is not None:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _stable_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _expected_build_image_id(value: str) -> str:
    if type(value) is not str or IMAGE_ID_RE.fullmatch(value) is None:
        _fail(
            "invalid-expected-optimizer-build-image-id",
            "expected optimizer build image ID must be sha256:<64 lowercase hex>",
        )
    if value == "sha256:" + "0" * 64:
        _fail(
            "invalid-expected-optimizer-build-image-id",
            "expected optimizer build image ID must not be the zero digest",
        )
    return value


def _optimizer_e0_artifacts(inventory: gate_inputs.GateEInventory) -> _OptimizerE0Artifacts:
    artifacts = dict(inventory.artifacts)
    try:
        return _OptimizerE0Artifacts(
            report=artifacts["canonical_e0.optimizer_report"],
            raw_evidence=artifacts["canonical_e0.optimizer_raw_evidence"],
            profile_binary=artifacts["release.profile_binary"],
        )
    except KeyError as error:  # pragma: no cover - parser fixes this set
        _fail("invalid-gate-e-inventory", f"missing fixed optimizer E0 role: {error}")


def _preflight_optimizer_e0_inputs(
    gate_e_evidence_root_fd: int,
) -> tuple[common.EvidenceDescriptor, _OptimizerE0Artifacts]:
    """Check component limits before the full Gate E replayer streams any leaf."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            gate_e_evidence_root_fd,
            "Gate E evidence root",
        )
    )
    _raw, inventory_descriptor, _document, inventory = _gate(
        lambda: gate_inputs._read_inventory(gate_e_evidence_root_fd)
    )
    _gate(lambda: gate_inputs._bound_inventory_closure(inventory_descriptor, inventory))
    _gate(lambda: gate_inputs._assert_exact_entries(gate_e_evidence_root_fd, inventory))
    artifacts = _optimizer_e0_artifacts(inventory)
    limits = {
        "optimizer raw evidence": (artifacts.raw_evidence, MAX_OPTIMIZER_RAW_ARCHIVE_BYTES),
        "optimizer report": (artifacts.report, MAX_OPTIMIZER_REPORT_BYTES),
        "optimizer profile binary": (artifacts.profile_binary, MAX_PROFILE_BINARY_BYTES),
    }
    for label, (descriptor, maximum) in limits.items():
        if descriptor.byte_length > maximum:
            _fail(
                "optimizer-e0-input-too-large",
                f"{label} exceeds its optimizer semantic replay byte bound",
            )
    if sum(descriptor.byte_length for descriptor, _maximum in limits.values()) > MAX_OPTIMIZER_SCRATCH_BYTES:
        _fail("optimizer-e0-input-too-large", "optimizer scratch input total exceeds its byte bound")
    return inventory_descriptor, artifacts


def _require_inventory_binding(
    structural_result: dict[str, Any],
    expected: common.EvidenceDescriptor,
    *,
    phase: str,
) -> None:
    if structural_result.get("gate_e_input_inventory") != expected.as_json():
        _fail(
            "gate-e-input-inventory-descriptor-mismatch",
            f"Gate E inventory changed between optimizer E0 {phase} and structural replay",
        )


def _frozen_source_archive(
    frozen_candidate_root_fd: int,
    structural_result: dict[str, Any],
) -> common.EvidenceDescriptor:
    _raw, manifest_descriptor, manifest = _frozen_replay(
        lambda: frozen_replayer._read_manifest(frozen_candidate_root_fd)
    )
    if manifest_descriptor.as_json() != structural_result.get("frozen_candidate_manifest"):
        _fail(
            "frozen-candidate-manifest-descriptor-mismatch",
            "held frozen manifest differs from the structural Gate E replay",
        )
    candidate_id, source_revision = _frozen(
        lambda: frozen.parse_frozen_candidate_manifest_identity(manifest)
    )
    if (
        candidate_id != structural_result.get("candidate_id")
        or source_revision != structural_result.get("source_revision")
    ):
        _fail(
            "frozen-candidate-identity-mismatch",
            "held frozen manifest identity differs from the structural Gate E replay",
        )
    bound_inputs = manifest.get("bound_inputs")
    if type(bound_inputs) is not dict or type(bound_inputs.get("source")) is not dict:
        _fail("invalid-frozen-candidate", "frozen manifest has no typed source input")
    archive = _common(
        lambda: common.parse_descriptor(
            bound_inputs["source"].get("archive"),
            "frozen candidate bound source archive",
        )
    )
    if archive.byte_length < 1:
        _fail("invalid-frozen-candidate", "frozen candidate source archive must be nonempty")
    return archive


def _snapshot_optimizer_inputs(
    gate_e_evidence_root_fd: int,
    artifacts: _OptimizerE0Artifacts,
    scratch_root: Path,
) -> _ScratchSnapshots:
    """Copy all path-based optimizer inputs beneath one pinned private root FD."""

    source_inputs = (
        ("raw_evidence", artifacts.raw_evidence, "optimizer-e0-raw-evidence.tar", MAX_OPTIMIZER_RAW_ARCHIVE_BYTES),
        ("report", artifacts.report, "optimizer-e0-report.json", MAX_OPTIMIZER_REPORT_BYTES),
        ("profile_binary", artifacts.profile_binary, "optimizer-e0-profile-binary", MAX_PROFILE_BINARY_BYTES),
    )
    scratch_fd: int | None = None
    try:
        scratch_fd = _common(
            lambda: common.open_private_evidence_directory(
                scratch_root,
                "optimizer E0 private scratch root",
            )
        )
        nofollow, _directory, cloexec, _nonblock = _common(common.require_safe_open_flags)
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | cloexec
        descriptors: dict[str, common.EvidenceDescriptor] = {}
        paths: dict[str, Path] = {}

        def write_all(output_fd: int, raw: bytes) -> None:
            offset = 0
            while offset < len(raw):
                try:
                    written = os.write(output_fd, raw[offset:])
                except OSError as error:
                    _fail("scratch-snapshot-failed", f"cannot write optimizer E0 scratch: {error}")
                if written < 1:
                    _fail("scratch-snapshot-failed", "optimizer E0 scratch write was incomplete")
                offset += written

        for role, source_descriptor, target_name, maximum in source_inputs:
            scratch_descriptor = common.EvidenceDescriptor(
                path=target_name,
                sha256=source_descriptor.sha256,
                byte_length=source_descriptor.byte_length,
            )

            def copy_from_held_file(
                source: BinaryIO,
                *,
                source_descriptor: common.EvidenceDescriptor = source_descriptor,
                target_name: str = target_name,
            ) -> None:
                output_fd = -1
                remaining = source_descriptor.byte_length
                try:
                    try:
                        output_fd = os.open(target_name, create_flags, 0o600, dir_fd=scratch_fd)
                    except FileExistsError as error:
                        _fail("scratch-snapshot-failed", f"optimizer scratch leaf already exists: {error}")
                    except (NotImplementedError, OSError, TypeError) as error:
                        _fail("scratch-snapshot-failed", f"cannot create optimizer scratch leaf: {error}")
                    os.fchmod(output_fd, 0o600)
                    while remaining:
                        chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                        if not chunk:
                            _fail("truncated-input", "optimizer evidence changed while it was snapshotted")
                        write_all(output_fd, chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        _fail("mutated-input", "optimizer evidence grew while it was snapshotted")
                    os.fsync(output_fd)
                except OSError as error:
                    _fail("scratch-snapshot-failed", f"cannot snapshot optimizer evidence: {error}")
                finally:
                    _close_quietly(output_fd)

            _common(
                lambda source_descriptor=source_descriptor, maximum=maximum, role=role: common.consume_descriptor_file(
                    gate_e_evidence_root_fd,
                    source_descriptor,
                    f"Gate E optimizer E0 {role}",
                    copy_from_held_file,
                    maximum_bytes=maximum,
                )
            )
            descriptors[role] = scratch_descriptor
            paths[role] = scratch_root / target_name

        for role, scratch_descriptor in descriptors.items():
            _common(
                lambda role=role, scratch_descriptor=scratch_descriptor: common.verify_private_snapshot_descriptor_file(
                    scratch_fd,
                    scratch_descriptor,
                    f"optimizer E0 private scratch {role}",
                    maximum_bytes=MAX_OPTIMIZER_SCRATCH_BYTES,
                )
            )
        try:
            root_identity = _stable_identity(os.fstat(scratch_fd))
            leaf_identities = {
                role: _stable_identity(os.lstat(descriptor.path, dir_fd=scratch_fd))
                for role, descriptor in descriptors.items()
            }
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot retain optimizer scratch identity: {error}")
        snapshots = _ScratchSnapshots(
            root=scratch_root,
            root_fd=scratch_fd,
            descriptors=descriptors,
            paths=paths,
            root_identity=root_identity,
            leaf_identities=leaf_identities,
        )
        scratch_fd = None
        return snapshots
    finally:
        _close_quietly(scratch_fd)


def _require_scratch_snapshots_unchanged(
    snapshots: _ScratchSnapshots,
    *,
    phase: str,
) -> None:
    """Reject any scratch root/leaf change around the pathname-only checker."""

    def verify_identities() -> None:
        try:
            held_root = _stable_identity(os.fstat(snapshots.root_fd))
            visible_root = _stable_identity(os.lstat(snapshots.root))
            visible_leaves = {
                role: _stable_identity(os.lstat(descriptor.path, dir_fd=snapshots.root_fd))
                for role, descriptor in snapshots.descriptors.items()
            }
        except OSError as error:
            _fail("scratch-snapshot-mutated", f"cannot inspect optimizer scratch after {phase}: {error}")
        if held_root != snapshots.root_identity or visible_root != snapshots.root_identity:
            _fail("scratch-snapshot-mutated", f"optimizer scratch root changed during {phase}")
        if visible_leaves != snapshots.leaf_identities:
            _fail("scratch-snapshot-mutated", f"optimizer scratch leaf changed during {phase}")

    verify_identities()
    for role, descriptor in snapshots.descriptors.items():
        _common(
            lambda role=role, descriptor=descriptor: common.verify_private_snapshot_descriptor_file(
                snapshots.root_fd,
                descriptor,
                f"optimizer E0 private scratch {role}",
                maximum_bytes=MAX_OPTIMIZER_SCRATCH_BYTES,
            )
        )
    verify_identities()


def _replay_optimizer_raw(
    snapshots: _ScratchSnapshots,
    *,
    source_revision: str,
    source_archive_sha256: str,
    expected_build_image_id: str,
) -> dict[str, Any]:
    """Run the path-based optimizer verifier only on private scratch paths."""

    try:
        import check_optimization_evidence as optimizer_evidence
    except ModuleNotFoundError as error:
        if error.name == "tomllib":
            _fail(
                "optimizer-e0-runtime-requires-tomllib",
                "optimizer E0 semantic replay requires Python 3.11+ or the reviewed tomllib compatibility wrapper",
            )
        raise
    expected_policy = {
        "GATE_ID": OPTIMIZER_GATE_ID,
        "RECEIPT_VERSION": OPTIMIZER_RECEIPT_VERSION,
        "FIXED37_PRODUCTION_BATCH_GATE_ID": FIXED37_GATE_ID,
        "MAX_ARCHIVE_BYTES": MAX_OPTIMIZER_RAW_ARCHIVE_BYTES,
        "MAX_TOTAL_BYTES": MAX_OPTIMIZER_RAW_CONTENT_BYTES,
        "MAX_JSON_BYTES": MAX_OPTIMIZER_REPORT_BYTES,
        "MAX_FILE_BYTES": MAX_PROFILE_BINARY_BYTES,
    }
    if (
        optimizer_contract.CONTRACT_VERSION
        != "riley.optimizer-e0-final-report-contract.v1"
        or optimizer_contract.POLICY_SHA256
        != EXPECTED_FINAL_REPORT_CONTRACT_POLICY_SHA256
    ):
        _fail(
            "optimizer-e0-policy-drift",
            "optimizer final-report semantic contract changed without this adapter",
        )
    for field, value in expected_policy.items():
        if getattr(optimizer_evidence, field, None) != value:
            _fail("optimizer-e0-policy-drift", f"optimizer evidence {field} changed without this adapter")
    try:
        result = optimizer_evidence.replay_raw_evidence(
            snapshots.paths["raw_evidence"],
            report=snapshots.paths["report"],
            source_revision=source_revision,
            source_archive_sha256=source_archive_sha256,
            build_image_id=expected_build_image_id,
            profile_binary=snapshots.paths["profile_binary"],
        )
    except optimizer_evidence.OptimizationEvidenceError as error:
        _fail("optimizer-e0-raw-replay-failed", str(error))
    except OSError as error:
        _fail("optimizer-e0-raw-replay-failed", str(error))
    if type(result) is not dict:
        _fail("optimizer-e0-raw-replay-failed", "optimizer raw replayer did not return an object")
    return result


def _require_optimizer_result_bindings(
    result: Mapping[str, Any],
    *,
    source_revision: str,
    source_archive: common.EvidenceDescriptor,
    artifacts: _OptimizerE0Artifacts,
    expected_build_image_id: str,
) -> None:
    expected = {
        "report_sha256": artifacts.report.sha256,
        "raw_evidence_sha256": artifacts.raw_evidence.sha256,
        "profile_binary_sha256": artifacts.profile_binary.sha256,
        "build_image_sha256": expected_build_image_id.removeprefix("sha256:"),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            _fail(
                f"optimizer-{field}-mismatch",
                f"optimizer E0 raw replay {field} does not match the fixed Gate E/trusted binding",
            )
    report = result.get("report")
    if type(report) is not dict:
        _fail("optimizer-report-mismatch", "optimizer raw replay returned no typed report")
    if (
        report.get("schema_version") != 1
        or report.get("gate_id") != OPTIMIZER_GATE_ID
        or report.get("status") != "passed"
        or report.get("semantic_class") != "E0"
    ):
        _fail("optimizer-report-mismatch", "optimizer raw replay returned the wrong E0 report contract")
    source = report.get("source")
    if (
        type(source) is not dict
        or source.get("git_commit") != source_revision
        or source.get("archive_sha256") != source_archive.sha256
        or source.get("git_dirty") is not False
    ):
        _fail("optimizer-source-archive-mismatch", "optimizer report does not bind the frozen source archive")
    try:
        report_image_sha256 = optimizer_contract.validate_final_candidate_report(
            report,
            source_revision=source_revision,
            source_archive_sha256=source_archive.sha256,
        )
    except optimizer_contract.OptimizerE0SemanticContractError as error:
        _fail("optimizer-final-report-contract-mismatch", str(error))
    if report_image_sha256 != expected_build_image_id.removeprefix("sha256:"):
        _fail(
            "optimizer-report-build-image-mismatch",
            "optimizer report image does not match the externally supplied trusted input",
        )
    fixed37 = next(
        (test for test in report["tests"] if test["id"] == "fixed37-production-batch-e0"),
        None,
    )
    if type(fixed37) is not dict:  # covered by the contract; retained as a type guard.
        _fail("optimizer-final-report-contract-mismatch", "optimizer report has no typed fixed37 test")
    replayed_logs = result.get("log_sha256")
    replayed_binaries = result.get("test_binary_sha256")
    if (
        type(replayed_logs) is not dict
        or replayed_logs.get("fixed37-production-batch-e0") != fixed37["log_sha256"]
    ):
        _fail(
            "optimizer-fixed37-log-mismatch",
            "optimizer raw replay fixed37 log digest does not match the final report",
        )
    if (
        type(replayed_binaries) is not dict
        or replayed_binaries.get("fixed37-production-batch-gpu-test")
        != fixed37["test_binary_sha256"]
    ):
        _fail(
            "optimizer-fixed37-test-binary-mismatch",
            "optimizer raw replay fixed37 binary digest does not match the final report",
        )


def _replay_rc3_gate_e_optimizer_e0_v1_on_held_fds(
    gate_e_evidence_root_fd: int,
    frozen_candidate_root_fd: int,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
    scratch_parent: Path,
    expected_optimizer_build_image_id: str,
) -> dict[str, Any]:
    """Replay optimizer canonical E0 while retaining every caller-held root FD."""

    _require_bytecode_cache_disabled()
    expected_image = _expected_build_image_id(expected_optimizer_build_image_id)
    preflight_inventory, artifacts = _preflight_optimizer_e0_inputs(gate_e_evidence_root_fd)
    structural_start = _gate(
        lambda: gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds(
            gate_e_evidence_root_fd,
            frozen_candidate_root_fd,
            input_evidence_root_fd,
            repository_root,
            repository_root_fd,
        )
    )
    _require_inventory_binding(structural_start, preflight_inventory, phase="preflight")
    source_archive = _frozen_source_archive(frozen_candidate_root_fd, structural_start)
    source_revision = structural_start.get("source_revision")
    if type(source_revision) is not str:
        _fail("invalid-frozen-candidate", "structural Gate E replay returned no source revision")

    with tempfile.TemporaryDirectory(
        prefix="riley-gate-e-optimizer-e0-",
        dir=os.fspath(scratch_parent),
    ) as temporary:
        scratch_root = Path(temporary)
        try:
            scratch_root.chmod(0o700)
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot make optimizer scratch private: {error}")
        snapshots = _snapshot_optimizer_inputs(
            gate_e_evidence_root_fd,
            artifacts,
            scratch_root,
        )
        try:
            optimizer_result = _replay_optimizer_raw(
                snapshots,
                source_revision=source_revision,
                source_archive_sha256=source_archive.sha256,
                expected_build_image_id=expected_image,
            )
            _require_scratch_snapshots_unchanged(
                snapshots,
                phase="legacy optimizer replay",
            )
        finally:
            _close_quietly(snapshots.root_fd)
    _require_optimizer_result_bindings(
        optimizer_result,
        source_revision=source_revision,
        source_archive=source_archive,
        artifacts=artifacts,
        expected_build_image_id=expected_image,
    )
    structural_end = _gate(
        lambda: gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds(
            gate_e_evidence_root_fd,
            frozen_candidate_root_fd,
            input_evidence_root_fd,
            repository_root,
            repository_root_fd,
        )
    )
    _require_inventory_binding(structural_end, preflight_inventory, phase="semantic replay")
    if common.canonical_json_bytes(structural_end) != common.canonical_json_bytes(structural_start):
        _fail("gate-e-input-replay-drift", "Gate E structural inputs changed during optimizer E0 replay")
    return {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "optimizer_e0_status": "passed",
        "candidate_id": structural_start["candidate_id"],
        "source_revision": source_revision,
        "expected_optimizer_build_image_id": expected_image,
        "gate_e_input_inventory": structural_start["gate_e_input_inventory"],
        "frozen_candidate_manifest": structural_start["frozen_candidate_manifest"],
        "optimizer_e0": {
            "report": artifacts.report.as_json(),
            "raw_evidence": artifacts.raw_evidence.as_json(),
            "profile_binary": artifacts.profile_binary.as_json(),
            "source_archive": source_archive.as_json(),
        },
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "not_established": dict(NOT_ESTABLISHED),
        "reason_codes": [],
    }


def replay_rc3_gate_e_optimizer_e0_v1(
    gate_e_evidence_root: Path,
    *,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
    expected_optimizer_build_image_id: str,
) -> dict[str, Any]:
    """Open and lock four disjoint roots for one optimizer E0 component replay."""

    _require_bytecode_cache_disabled()
    _expected_build_image_id(expected_optimizer_build_image_id)
    gate_root = _frozen(
        lambda: frozen.normalized_absolute_path(gate_e_evidence_root, "--gate-e-evidence-root")
    )
    frozen_root = _frozen(
        lambda: frozen.normalized_absolute_path(frozen_candidate_root, "--frozen-candidate-root")
    )
    input_root = _frozen(
        lambda: frozen.normalized_absolute_path(input_evidence_root, "--input-evidence-root")
    )
    source_root = _frozen(
        lambda: frozen.normalized_absolute_path(repository_root, "--repository-root")
    )
    scratch_parent = _frozen(
        lambda: frozen.normalized_absolute_path(
            EXTERNAL_SCRATCH_PARENT,
            "optimizer E0 external scratch parent",
        )
    )
    _frozen(
        lambda: frozen.require_disjoint_paths(
            {
                "Gate E evidence root": gate_root,
                "frozen candidate root": frozen_root,
                "freeze-input evidence root": input_root,
                "source checkout": source_root,
            }
        )
    )
    source_root_fd: int | None = None
    input_root_fd: int | None = None
    frozen_root_fd: int | None = None
    gate_root_fd: int | None = None
    scratch_parent_fd: int | None = None
    try:
        source_root_fd = _common(lambda: common.open_absolute_directory(source_root, "source checkout"))
        input_root_fd = _common(
            lambda: common.open_private_evidence_directory(input_root, "freeze-input evidence root")
        )
        frozen_root_fd = _common(
            lambda: common.open_private_evidence_directory(frozen_root, "frozen candidate root")
        )
        gate_root_fd = _common(
            lambda: common.open_private_evidence_directory(gate_root, "Gate E evidence root")
        )
        scratch_parent_fd = _common(
            lambda: common.open_absolute_directory(
                scratch_parent,
                "optimizer E0 external scratch parent",
            )
        )
        roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
            "frozen candidate root": (frozen_root, frozen_root_fd),
            "Gate E evidence root": (gate_root, gate_root_fd),
            "optimizer E0 external scratch parent": (
                scratch_parent,
                scratch_parent_fd,
            ),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        _shared_lock(input_root_fd, "freeze-input evidence root")
        _shared_lock(frozen_root_fd, "frozen candidate root")
        _shared_lock(gate_root_fd, "Gate E evidence root")
        result = _replay_rc3_gate_e_optimizer_e0_v1_on_held_fds(
            gate_root_fd,
            frozen_root_fd,
            input_root_fd,
            source_root,
            source_root_fd,
            scratch_parent,
            expected_optimizer_build_image_id,
        )
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        return result
    finally:
        _unlock_quietly(gate_root_fd)
        _unlock_quietly(frozen_root_fd)
        _unlock_quietly(input_root_fd)
        _close_quietly(gate_root_fd)
        _close_quietly(frozen_root_fd)
        _close_quietly(input_root_fd)
        _close_quietly(source_root_fd)
        _close_quietly(scratch_parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-e-evidence-root", required=True, type=Path)
    parser.add_argument("--frozen-candidate-root", required=True, type=Path)
    parser.add_argument("--input-evidence-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--expected-optimizer-build-image-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = replay_rc3_gate_e_optimizer_e0_v1(
            args.gate_e_evidence_root,
            frozen_candidate_root=args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
            expected_optimizer_build_image_id=args.expected_optimizer_build_image_id,
        )
    except OptimizerE0ReplayError as error:
        print(f"RC3 Gate E optimizer E0 replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
