#!/usr/bin/env python3
"""Replay the RC3 Gate E performance component through held evidence FDs.

Only the closed performance subset is consumed: its report/raw archive,
optimizer report, profile binary, native release executable, frozen source /
release / model identities, and the reviewed baseline read through the held
source-checkout descriptor. The raw archive is copied once into a private
mode-0600 scratch leaf and then replayed only through its held file descriptor.

Success establishes exactly this performance component.  It is neither an
aggregate Gate E decision, a semantic receipt, candidate qualification, nor a
deployment decision.  The caller-provided image values are equality anchors,
not independent review approvals.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, Callable, Mapping, NoReturn, Sequence, TypeVar


_BYTECODE_DISABLED_AT_STARTUP = bool(sys.flags.dont_write_bytecode)
_BYTECODE_DISABLED_ON_MODULE_ENTRY = sys.dont_write_bytecode
sys.dont_write_bytecode = True

import check_rc3_freeze_input_admission as freeze_inputs
import optimizer_e0_semantic_contract as optimizer_contract
import provenance_v2_common as common
import rc3_frozen_candidate_common as frozen
import rc3_frozen_candidate_topology as topology
import replay_rc3_frozen_candidate_v1 as frozen_replayer
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs


REPLAY_VERSION = "riley.rc3-gate-e-performance-semantic-replay.v1"
SCOPE = "gate-e-performance-semantic-component-only"
AUTHORITY = "gate-e-performance-semantic-replay-only"
PERFORMANCE_POLICY_VERSION = "riley.release-performance-bound-semantic.v1"
EXPECTED_PERFORMANCE_POLICY_SHA256 = (
    "d342fe14170203cd2c1c029eb2f159d359778fb930d5faa4083a745e2b92cb7a"
)
EXPECTED_OPTIMIZER_CONTRACT_POLICY_SHA256 = (
    "3efab239fa03631f10496109a4b04da1d6ef3caf2d9e0dfb1148138d3e0b9996"
)
MAX_PERFORMANCE_RAW_ARCHIVE_BYTES = 543_686_656
MAX_PERFORMANCE_RAW_STREAM_MEMBER_BYTES = 64 * 1024 * 1024
MAX_PERFORMANCE_RAW_SCRATCH_BYTES = MAX_PERFORMANCE_RAW_ARCHIVE_BYTES
MAX_PERFORMANCE_REPORT_BYTES = 16 * 1024 * 1024
MAX_OPTIMIZER_REPORT_BYTES = 8 * 1024 * 1024
MAX_PROFILE_BINARY_BYTES = 512 * 1024 * 1024
MAX_RELEASE_ELF_BYTES = 512 * 1024 * 1024
MAX_REVIEWED_BASELINE_BYTES = 1024 * 1024
EXTERNAL_SCRATCH_PARENT = Path("/var/tmp")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

CHECK_NAMES = (
    "closed-gate-e-input-inventory-replayed-before-performance",
    "externally-supplied-release-and-optimizer-image-anchors-bound",
    "frozen-source-release-model-and-optimizer-identities-bound",
    "private-performance-raw-and-held-baseline-replayed",
    "bounded-streaming-raw-policy-applied-to-performance-raw-replay",
    "performance-component-does-not-aggregate-gate-e",
)

NOT_ESTABLISHED = {
    "native_e0": "not-established",
    "optimizer_e0": "not-established",
    "python_free": "not-established",
    "soak": "not-established",
    "release_image_review": "not-established",
    "optimizer_build_image_review": "not-established",
    "release_container_content": "not-established",
    "model_mount_provenance": "not-established",
    "producer_sidecar_equality": "not-established",
    "source_archive_content": "not-established",
    "gate_e_pass": "not-established",
    "semantic_receipt": "not-established",
    "qualification": "not-established",
    "deployment": "not-established",
}

T = TypeVar("T")


class PerformanceReplayError(ValueError):
    """The performance Gate E component cannot be replayed safely."""


@dataclass(frozen=True)
class _PerformanceArtifacts:
    report: common.EvidenceDescriptor
    raw_evidence: common.EvidenceDescriptor
    optimizer_report: common.EvidenceDescriptor
    profile_binary: common.EvidenceDescriptor
    native_candidate_executable: common.EvidenceDescriptor


@dataclass(frozen=True)
class _FrozenBindings:
    source_archive: common.EvidenceDescriptor
    release_elf: common.EvidenceDescriptor
    release_image_digest: str
    models: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _ScratchSnapshot:
    root: Path
    root_fd: int
    descriptor: common.EvidenceDescriptor
    name: str
    path: Path
    root_identity: tuple[int, ...]
    leaf_identity: tuple[int, ...]


def _fail(code: str, message: str) -> NoReturn:
    error = PerformanceReplayError(f"{code}: {message}")
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


def _freeze_input(call: Callable[[], T]) -> T:
    try:
        return call()
    except freeze_inputs.FreezeInputAdmissionError as error:
        _fail(getattr(error, "reason_code", "invalid-freeze-input"), str(error))


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
    if not (_BYTECODE_DISABLED_AT_STARTUP and _BYTECODE_DISABLED_ON_MODULE_ENTRY):
        _fail(
            "bytecode-cache-write-not-permitted",
            "invoke this replayer with python3 -B or PYTHONDONTWRITEBYTECODE=1",
        )


def _shared_lock(directory_fd: int, label: str) -> None:
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(
            "evidence-root-lock-unavailable",
            f"cannot acquire shared {label} lock: {error}",
        )


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


def _expected_image_id(value: str, label: str) -> str:
    if type(value) is not str or IMAGE_ID_RE.fullmatch(value) is None:
        _fail(
            f"invalid-{label}",
            f"{label.replace('-', ' ')} must be sha256:<64 lowercase hex>",
        )
    if value == "sha256:" + "0" * 64:
        _fail(f"invalid-{label}", f"{label.replace('-', ' ')} must not be the zero digest")
    return value


def _performance_artifacts(inventory: gate_inputs.GateEInventory) -> _PerformanceArtifacts:
    artifacts = dict(inventory.artifacts)
    try:
        return _PerformanceArtifacts(
            report=artifacts["performance.report"],
            raw_evidence=artifacts["performance.raw_evidence"],
            optimizer_report=artifacts["canonical_e0.optimizer_report"],
            profile_binary=artifacts["release.profile_binary"],
            native_candidate_executable=artifacts["release.native_candidate_executable"],
        )
    except KeyError as error:  # pragma: no cover - inventory parser fixes the set.
        _fail("invalid-gate-e-inventory", f"missing fixed performance role: {error}")


def _preflight_performance_inputs(
    gate_e_evidence_root_fd: int,
) -> tuple[common.EvidenceDescriptor, _PerformanceArtifacts]:
    """Reject component-specific size abuse before full inventory streaming."""

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
    artifacts = _performance_artifacts(inventory)
    limits = {
        "performance raw evidence": (
            artifacts.raw_evidence,
            MAX_PERFORMANCE_RAW_ARCHIVE_BYTES,
        ),
        "performance report": (artifacts.report, MAX_PERFORMANCE_REPORT_BYTES),
        "optimizer report": (artifacts.optimizer_report, MAX_OPTIMIZER_REPORT_BYTES),
        "performance profile binary": (
            artifacts.profile_binary,
            MAX_PROFILE_BINARY_BYTES,
        ),
        "performance native candidate executable": (
            artifacts.native_candidate_executable,
            MAX_RELEASE_ELF_BYTES,
        ),
    }
    for label, (descriptor, maximum) in limits.items():
        if descriptor.byte_length > maximum:
            _fail(
                "performance-input-too-large",
                f"{label} exceeds its performance semantic replay byte bound",
            )
    return inventory_descriptor, artifacts


def _require_inventory_binding(
    structural_result: Mapping[str, Any],
    expected: common.EvidenceDescriptor,
    *,
    phase: str,
) -> None:
    if structural_result.get("gate_e_input_inventory") != expected.as_json():
        _fail(
            "gate-e-input-inventory-descriptor-mismatch",
            f"Gate E inventory changed between performance {phase} and structural replay",
        )


def _frozen_bindings(
    frozen_candidate_root_fd: int,
    structural_result: Mapping[str, Any],
) -> _FrozenBindings:
    """Recover the typed frozen source/release/model identity bindings."""

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
    if type(bound_inputs) is not dict:
        _fail("invalid-frozen-candidate", "frozen manifest has no typed bound inputs")
    request, _descriptors = _freeze_input(lambda: freeze_inputs._parse_request(bound_inputs))
    source = request.get("source")
    release = request.get("release")
    models = request.get("models")
    if type(source) is not dict or type(release) is not dict or type(models) is not list:
        _fail("invalid-frozen-candidate", "frozen manifest has malformed typed bindings")
    archive = source.get("archive")
    release_elf = release.get("elf")
    container = release.get("container")
    if (
        not isinstance(archive, common.EvidenceDescriptor)
        or not isinstance(release_elf, common.EvidenceDescriptor)
        or type(container) is not dict
    ):
        _fail("invalid-frozen-candidate", "frozen source or release binding is malformed")
    image_digest = container.get("image_digest")
    if archive.byte_length < 1 or release_elf.byte_length < 1:
        _fail("invalid-frozen-candidate", "frozen source/release descriptors must be nonempty")
    if release_elf.byte_length > MAX_RELEASE_ELF_BYTES:
        _fail("performance-input-too-large", "frozen release ELF exceeds its replay byte bound")
    return _FrozenBindings(
        source_archive=archive,
        release_elf=release_elf,
        release_image_digest=_expected_image_id(
            image_digest,
            "expected-release-image-id",
        ),
        models=tuple(models),
    )


def _read_strict_json_descriptor(
    directory_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    """Read a JSON evidence leaf only through its held direct-file descriptor."""

    def consume(source: BinaryIO) -> bytes:
        output = bytearray()
        remaining = descriptor.byte_length
        while remaining:
            chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
            if not chunk:
                _fail("truncated-input", f"{label} changed while it was read")
            output.extend(chunk)
            remaining -= len(chunk)
        if source.read(1):
            _fail("mutated-input", f"{label} grew while it was read")
        return bytes(output)

    raw = _common(
        lambda: common.consume_descriptor_file(
            directory_fd,
            descriptor,
            label,
            consume,
            maximum_bytes=maximum_bytes,
        )
    )
    document = _common(
        lambda: common.parse_strict_json(
            raw,
            label,
            maximum_bytes=maximum_bytes,
            require_object=True,
        )
    )
    assert type(document) is dict
    return document


def _verify_held_descriptor(
    directory_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
    *,
    maximum_bytes: int,
) -> None:
    _common(
        lambda: common.consume_descriptor_file(
            directory_fd,
            descriptor,
            label,
            lambda _source: None,
            maximum_bytes=maximum_bytes,
        )
    )


def _read_reviewed_baseline(repository_root_fd: int) -> bytes:
    return _common(
        lambda: common.read_bounded_regular_relative(
            repository_root_fd,
            "benchmarks/release/performance-baseline-v1.json",
            "reviewed performance baseline",
            maximum_bytes=MAX_REVIEWED_BASELINE_BYTES,
        )
    )


def _snapshot_performance_raw(
    gate_e_evidence_root_fd: int,
    descriptor: common.EvidenceDescriptor,
    scratch_root: Path,
) -> _ScratchSnapshot:
    """Copy the raw evidence once under a private scratch root."""

    scratch_fd: int | None = None
    target_name = "performance-raw-evidence.tar"
    scratch_descriptor = common.EvidenceDescriptor(
        path=target_name,
        sha256=descriptor.sha256,
        byte_length=descriptor.byte_length,
    )
    try:
        scratch_fd = _common(
            lambda: common.open_private_evidence_directory(
                scratch_root,
                "performance private scratch root",
            )
        )
        nofollow, _directory, cloexec, _nonblock = _common(common.require_safe_open_flags)
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | cloexec

        def write_all(output_fd: int, raw: bytes) -> None:
            offset = 0
            while offset < len(raw):
                try:
                    written = os.write(output_fd, raw[offset:])
                except OSError as error:
                    _fail("scratch-snapshot-failed", f"cannot write performance scratch: {error}")
                if written < 1:
                    _fail("scratch-snapshot-failed", "performance scratch write was incomplete")
                offset += written

        def copy_from_held_file(source: BinaryIO) -> None:
            output_fd = -1
            remaining = descriptor.byte_length
            try:
                try:
                    output_fd = os.open(target_name, create_flags, 0o600, dir_fd=scratch_fd)
                except FileExistsError as error:
                    _fail("scratch-snapshot-failed", f"performance scratch leaf already exists: {error}")
                except (NotImplementedError, OSError, TypeError) as error:
                    _fail("scratch-snapshot-failed", f"cannot create performance scratch leaf: {error}")
                os.fchmod(output_fd, 0o600)
                while remaining:
                    chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                    if not chunk:
                        _fail("truncated-input", "performance raw evidence changed while snapshotted")
                    write_all(output_fd, chunk)
                    remaining -= len(chunk)
                if source.read(1):
                    _fail("mutated-input", "performance raw evidence grew while snapshotted")
                os.fsync(output_fd)
            except OSError as error:
                _fail("scratch-snapshot-failed", f"cannot snapshot performance raw evidence: {error}")
            finally:
                _close_quietly(output_fd)

        _common(
            lambda: common.consume_descriptor_file(
                gate_e_evidence_root_fd,
                descriptor,
                "Gate E performance raw evidence",
                copy_from_held_file,
                maximum_bytes=MAX_PERFORMANCE_RAW_ARCHIVE_BYTES,
            )
        )
        _common(
            lambda: common.verify_private_snapshot_descriptor_file(
                scratch_fd,
                scratch_descriptor,
                "performance private scratch raw evidence",
                maximum_bytes=MAX_PERFORMANCE_RAW_ARCHIVE_BYTES,
            )
        )
        try:
            root_identity = _stable_identity(os.fstat(scratch_fd))
            leaf_identity = _stable_identity(os.lstat(target_name, dir_fd=scratch_fd))
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot retain performance scratch identity: {error}")
        snapshot = _ScratchSnapshot(
            root=scratch_root,
            root_fd=scratch_fd,
            descriptor=scratch_descriptor,
            name=target_name,
            path=scratch_root / target_name,
            root_identity=root_identity,
            leaf_identity=leaf_identity,
        )
        scratch_fd = None
        return snapshot
    finally:
        _close_quietly(scratch_fd)


def _require_scratch_snapshot_unchanged(
    snapshot: _ScratchSnapshot,
    *,
    phase: str,
) -> None:
    """Reject scratch replacement around the held-FD raw replay."""

    try:
        held_root = _stable_identity(os.fstat(snapshot.root_fd))
        visible_root = _stable_identity(os.lstat(snapshot.root))
        visible_leaf = _stable_identity(
            os.lstat(snapshot.name, dir_fd=snapshot.root_fd)
        )
    except OSError as error:
        _fail("scratch-snapshot-mutated", f"cannot inspect performance scratch after {phase}: {error}")
    if held_root != snapshot.root_identity or visible_root != snapshot.root_identity:
        _fail("scratch-snapshot-mutated", f"performance scratch root changed during {phase}")
    if visible_leaf != snapshot.leaf_identity:
        _fail("scratch-snapshot-mutated", f"performance scratch leaf changed during {phase}")
    _common(
        lambda: common.verify_private_snapshot_descriptor_file(
            snapshot.root_fd,
            snapshot.descriptor,
            "performance private scratch raw evidence",
            maximum_bytes=MAX_PERFORMANCE_RAW_ARCHIVE_BYTES,
        )
    )


def _open_scratch_snapshot_fd(snapshot: _ScratchSnapshot) -> int:
    """Open the copied raw evidence through the held scratch-root FD only."""

    nofollow, _directory, cloexec, nonblock = _common(common.require_safe_open_flags)
    flags = os.O_RDONLY | nofollow | cloexec | nonblock
    descriptor = -1
    try:
        descriptor = os.open(snapshot.name, flags, dir_fd=snapshot.root_fd)
        metadata = os.fstat(descriptor)
    except (NotImplementedError, OSError, TypeError) as error:
        _close_quietly(descriptor)
        _fail("scratch-snapshot-failed", f"cannot open held performance scratch leaf: {error}")
    if _stable_identity(metadata) != snapshot.leaf_identity:
        _close_quietly(descriptor)
        _fail("scratch-snapshot-mutated", "performance scratch leaf changed before raw replay")
    if metadata.st_size != snapshot.descriptor.byte_length:
        _close_quietly(descriptor)
        _fail("scratch-snapshot-mutated", "performance scratch leaf length changed before raw replay")
    return descriptor


def _load_performance_contract() -> ModuleType:
    """Load the path-minimal contract lazily and pin its public policy."""

    script = Path(__file__).resolve().parents[2] / "benchmarks/scripts/check_release_performance.py"
    try:
        spec = importlib.util.spec_from_file_location(
            "riley_gate_e_performance_contract",
            script,
        )
        if spec is None or spec.loader is None:  # pragma: no cover - static path.
            _fail("performance-contract-load-failed", f"cannot load performance contract: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        _fail("performance-contract-load-failed", str(error))
    expected_policy = {
        "BOUND_SEMANTIC_POLICY_VERSION": PERFORMANCE_POLICY_VERSION,
        "BOUND_SEMANTIC_POLICY_SHA256": EXPECTED_PERFORMANCE_POLICY_SHA256,
        "MAX_RAW_EVIDENCE_ARCHIVE_BYTES": MAX_PERFORMANCE_RAW_ARCHIVE_BYTES,
        "MAX_BOUND_RAW_STREAM_MEMBER_BYTES": MAX_PERFORMANCE_RAW_STREAM_MEMBER_BYTES,
        "MAX_BOUND_RAW_SCRATCH_BYTES": MAX_PERFORMANCE_RAW_SCRATCH_BYTES,
        "MAX_REVIEWED_BASELINE_BYTES": MAX_REVIEWED_BASELINE_BYTES,
    }
    for field, expected in expected_policy.items():
        if getattr(module, field, None) != expected:
            _fail("performance-policy-drift", f"performance contract {field} changed without this adapter")
    policy_digest = getattr(module, "bound_semantic_policy_sha256", None)
    validator = getattr(module, "validate_bound_performance_evidence", None)
    raw_replayer = getattr(module, "replay_bound_raw_evidence_fd", None)
    if not callable(policy_digest) or not callable(validator) or not callable(raw_replayer):
        _fail("performance-policy-drift", "performance contract lacks the public bound replay API")
    if policy_digest() != EXPECTED_PERFORMANCE_POLICY_SHA256:
        _fail("performance-policy-drift", "performance contract policy digest changed without this adapter")
    return module


def _validated_optimizer_model_tree(
    report: Mapping[str, Any],
    *,
    source_revision: str,
    source_archive_sha256: str,
    expected_optimizer_build_image_id: str,
) -> str:
    """Validate the held optimizer report and return its typed model tree hash."""

    if (
        optimizer_contract.CONTRACT_VERSION
        != "riley.optimizer-e0-final-report-contract.v1"
        or optimizer_contract.POLICY_SHA256
        != EXPECTED_OPTIMIZER_CONTRACT_POLICY_SHA256
    ):
        _fail(
            "performance-optimizer-contract-policy-drift",
            "optimizer semantic contract policy changed without this adapter",
        )
    try:
        image_sha256 = optimizer_contract.validate_final_candidate_report(
            report,
            source_revision=source_revision,
            source_archive_sha256=source_archive_sha256,
        )
    except optimizer_contract.OptimizerE0SemanticContractError as error:
        _fail("performance-optimizer-report-contract-mismatch", str(error))
    if image_sha256 != expected_optimizer_build_image_id.removeprefix("sha256:"):
        _fail(
            "performance-optimizer-image-anchor-mismatch",
            "optimizer report image does not match the externally supplied anchor",
        )
    model = report.get("model")
    if type(model) is not dict:
        _fail("performance-optimizer-report-contract-mismatch", "optimizer report has no typed model")
    tree = model.get("manifest_sha256")
    if type(tree) is not str or SHA256_RE.fullmatch(tree) is None or tree == "0" * 64:
        _fail("performance-optimizer-report-contract-mismatch", "optimizer model tree digest is malformed")
    return tree


def _require_frozen_model_binding(
    candidate: Mapping[str, Any],
    optimizer_report: Mapping[str, Any],
    optimizer_model_tree_sha256: str,
    bindings: _FrozenBindings,
) -> None:
    """Bind performance candidate and optimizer model facts to one frozen model."""

    model = candidate.get("model")
    if type(model) is not dict:
        _fail("performance-candidate-model-mismatch", "performance result has no typed candidate model")
    model_id = model.get("model_id")
    matching = [row for row in bindings.models if row.get("model_id") == model_id]
    if len(matching) != 1:
        _fail("performance-frozen-model-mismatch", "performance model ID does not select one frozen model")
    frozen_model = matching[0]
    tree = frozen_model.get("tree")
    tokenizer = frozen_model.get("tokenizer")
    weights = frozen_model.get("weights")
    if (
        type(frozen_model.get("revision")) is not str
        or not isinstance(tree, common.EvidenceDescriptor)
        or not isinstance(tokenizer, common.EvidenceDescriptor)
        or type(weights) is not list
        or not all(isinstance(weight, common.EvidenceDescriptor) for weight in weights)
    ):
        _fail("invalid-frozen-candidate", "frozen performance model binding is malformed")
    expected_model = {
        "model_revision": frozen_model["revision"],
        "tokenizer_sha256": tokenizer.sha256,
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            _fail("performance-frozen-model-mismatch", f"performance model {field} differs from frozen input")
    if len(weights) != 1 or weights[0].sha256 != model.get("weights_sha256"):
        _fail(
            "performance-frozen-model-mismatch",
            "performance model requires exactly one matching frozen weight",
        )
    if tree.sha256 != optimizer_model_tree_sha256:
        _fail("performance-frozen-model-mismatch", "optimizer model tree differs from frozen model tree")
    optimizer_model = optimizer_report.get("model")
    if type(optimizer_model) is not dict:
        _fail("performance-optimizer-report-contract-mismatch", "optimizer report has no typed model")
    expected_optimizer_model = {
        "model_id": model_id,
        "revision": frozen_model["revision"],
        "manifest_sha256": tree.sha256,
        "weights_sha256": weights[0].sha256,
        "tokenizer_sha256": tokenizer.sha256,
    }
    for field, expected in expected_optimizer_model.items():
        if optimizer_model.get(field) != expected:
            _fail("performance-optimizer-model-mismatch", f"optimizer model {field} differs from frozen performance model")


def _replay_performance_raw(
    snapshot: _ScratchSnapshot,
    report: Mapping[str, Any],
    baseline_raw: bytes,
    *,
    source_revision: str,
    bindings: _FrozenBindings,
    artifacts: _PerformanceArtifacts,
    expected_release_image_id: str,
    expected_optimizer_build_image_id: str,
    optimizer_model_tree_sha256: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Call the public held-FD contract only on private scratch raw data."""

    contract = _load_performance_contract()
    raw_fd = _open_scratch_snapshot_fd(snapshot)
    try:
        result = contract.validate_bound_performance_evidence(
            report,
            raw_fd,
            reviewed_baseline_raw=baseline_raw,
            source_revision=source_revision,
            source_archive_sha256=bindings.source_archive.sha256,
            release_binary_sha256=bindings.release_elf.sha256,
            release_image_id=expected_release_image_id,
            profile_binary_sha256=artifacts.profile_binary.sha256,
            optimizer_report_sha256=artifacts.optimizer_report.sha256,
            optimizer_image_id=expected_optimizer_build_image_id,
            optimizer_model_tree_sha256=optimizer_model_tree_sha256,
            candidate_id=candidate_id,
            raw_evidence_sha256=artifacts.raw_evidence.sha256,
            raw_evidence_byte_length=artifacts.raw_evidence.byte_length,
        )
    except (contract.InputError, contract.ComparabilityError, OSError, TypeError) as error:
        _fail("performance-raw-replay-failed", str(error))
    finally:
        _close_quietly(raw_fd)
    if type(result) is not dict or type(result.get("candidate")) is not dict:
        _fail("performance-raw-replay-failed", "performance contract returned no typed result")
    return result


def _replay_rc3_gate_e_performance_v1_on_held_fds(
    gate_e_evidence_root_fd: int,
    frozen_candidate_root_fd: int,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
    scratch_parent: Path,
    expected_release_image_id: str,
    expected_optimizer_build_image_id: str,
) -> dict[str, Any]:
    """Replay performance while retaining every root descriptor and lock."""

    _require_bytecode_cache_disabled()
    expected_release_image = _expected_image_id(
        expected_release_image_id,
        "expected-release-image-id",
    )
    expected_optimizer_image = _expected_image_id(
        expected_optimizer_build_image_id,
        "expected-optimizer-build-image-id",
    )
    preflight_inventory, artifacts = _preflight_performance_inputs(gate_e_evidence_root_fd)
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
    bindings = _frozen_bindings(frozen_candidate_root_fd, structural_start)
    source_revision = structural_start.get("source_revision")
    candidate_id = structural_start.get("candidate_id")
    if type(source_revision) is not str or GIT_REVISION_RE.fullmatch(source_revision) is None:
        _fail("invalid-frozen-candidate", "structural Gate E replay returned no valid source revision")
    if type(candidate_id) is not str:
        _fail("invalid-frozen-candidate", "structural Gate E replay returned no candidate ID")
    if bindings.release_image_digest != expected_release_image:
        _fail(
            "performance-release-image-anchor-mismatch",
            "frozen release image digest differs from the externally supplied anchor",
        )
    baseline_before = _read_reviewed_baseline(repository_root_fd)
    submitted_report = _read_strict_json_descriptor(
        gate_e_evidence_root_fd,
        artifacts.report,
        "Gate E performance report",
        maximum_bytes=MAX_PERFORMANCE_REPORT_BYTES,
    )
    optimizer_report = _read_strict_json_descriptor(
        gate_e_evidence_root_fd,
        artifacts.optimizer_report,
        "Gate E optimizer report",
        maximum_bytes=MAX_OPTIMIZER_REPORT_BYTES,
    )
    _verify_held_descriptor(
        gate_e_evidence_root_fd,
        artifacts.profile_binary,
        "Gate E performance profile binary",
        maximum_bytes=MAX_PROFILE_BINARY_BYTES,
    )
    _verify_held_descriptor(
        gate_e_evidence_root_fd,
        artifacts.native_candidate_executable,
        "Gate E native candidate executable",
        maximum_bytes=MAX_RELEASE_ELF_BYTES,
    )
    _common(
        lambda: common.verify_descriptor_file(
            input_evidence_root_fd,
            bindings.release_elf,
            "frozen release ELF",
            maximum_bytes=MAX_RELEASE_ELF_BYTES,
        )
    )
    if (
        artifacts.native_candidate_executable.sha256 != bindings.release_elf.sha256
        or artifacts.native_candidate_executable.byte_length
        != bindings.release_elf.byte_length
    ):
        _fail(
            "performance-release-executable-mismatch",
            "Gate E native candidate executable differs from frozen release ELF",
        )
    optimizer_model_tree = _validated_optimizer_model_tree(
        optimizer_report,
        source_revision=source_revision,
        source_archive_sha256=bindings.source_archive.sha256,
        expected_optimizer_build_image_id=expected_optimizer_image,
    )

    with tempfile.TemporaryDirectory(
        prefix="riley-gate-e-performance-",
        dir=os.fspath(scratch_parent),
    ) as temporary:
        scratch_root = Path(temporary)
        try:
            scratch_root.chmod(0o700)
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot make performance scratch private: {error}")
        snapshot = _snapshot_performance_raw(
            gate_e_evidence_root_fd,
            artifacts.raw_evidence,
            scratch_root,
        )
        try:
            result = _replay_performance_raw(
                snapshot,
                submitted_report,
                baseline_before,
                source_revision=source_revision,
                bindings=bindings,
                artifacts=artifacts,
                expected_release_image_id=expected_release_image,
                expected_optimizer_build_image_id=expected_optimizer_image,
                optimizer_model_tree_sha256=optimizer_model_tree,
                candidate_id=candidate_id,
            )
            _require_scratch_snapshot_unchanged(
                snapshot,
                phase="held-FD performance raw replay",
            )
        finally:
            _close_quietly(snapshot.root_fd)
    _require_frozen_model_binding(
        result["candidate"],
        optimizer_report,
        optimizer_model_tree,
        bindings,
    )
    baseline_after = _read_reviewed_baseline(repository_root_fd)
    if baseline_after != baseline_before:
        _fail(
            "reviewed-baseline-drift",
            "reviewed performance baseline changed during semantic replay",
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
        _fail("gate-e-input-replay-drift", "Gate E structural inputs changed during performance replay")
    baseline = result.get("baseline")
    if type(baseline) is not dict or type(baseline.get("sha256")) is not str:
        _fail("performance-raw-replay-failed", "performance contract returned no typed reviewed baseline")
    if (
        result.get("raw_stream_member_byte_limit")
        != MAX_PERFORMANCE_RAW_STREAM_MEMBER_BYTES
        or result.get("scratch_disk_byte_limit") != MAX_PERFORMANCE_RAW_SCRATCH_BYTES
    ):
        _fail("performance-policy-drift", "performance contract returned unexpected streaming limits")
    return {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "performance_status": "passed",
        "candidate_id": candidate_id,
        "source_revision": source_revision,
        "expected_release_image_id": expected_release_image,
        "expected_optimizer_build_image_id": expected_optimizer_image,
        "gate_e_input_inventory": structural_start["gate_e_input_inventory"],
        "frozen_candidate_manifest": structural_start["frozen_candidate_manifest"],
        "performance": {
            "report": artifacts.report.as_json(),
            "raw_evidence": artifacts.raw_evidence.as_json(),
            "optimizer_report": artifacts.optimizer_report.as_json(),
            "profile_binary": artifacts.profile_binary.as_json(),
            "native_candidate_executable": artifacts.native_candidate_executable.as_json(),
            "release_elf": bindings.release_elf.as_json(),
            "source_archive": bindings.source_archive.as_json(),
            "frozen_release_image_digest": bindings.release_image_digest,
            "reviewed_baseline_sha256": baseline["sha256"],
            "raw_stream_member_byte_limit": result["raw_stream_member_byte_limit"],
            "scratch_disk_byte_limit": result["scratch_disk_byte_limit"],
        },
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "not_established": dict(NOT_ESTABLISHED),
        "reason_codes": [],
    }


def replay_rc3_gate_e_performance_v1(
    gate_e_evidence_root: Path,
    *,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
    expected_release_image_id: str,
    expected_optimizer_build_image_id: str,
) -> dict[str, Any]:
    """Open and lock all disjoint roots for one performance component replay."""

    _require_bytecode_cache_disabled()
    _expected_image_id(expected_release_image_id, "expected-release-image-id")
    _expected_image_id(
        expected_optimizer_build_image_id,
        "expected-optimizer-build-image-id",
    )
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
            "performance external scratch parent",
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
                "performance external scratch parent",
            )
        )
        roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
            "frozen candidate root": (frozen_root, frozen_root_fd),
            "Gate E evidence root": (gate_root, gate_root_fd),
            "performance external scratch parent": (scratch_parent, scratch_parent_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        _shared_lock(input_root_fd, "freeze-input evidence root")
        _shared_lock(frozen_root_fd, "frozen candidate root")
        _shared_lock(gate_root_fd, "Gate E evidence root")
        result = _replay_rc3_gate_e_performance_v1_on_held_fds(
            gate_root_fd,
            frozen_root_fd,
            input_root_fd,
            source_root,
            source_root_fd,
            scratch_parent,
            expected_release_image_id,
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
    parser.add_argument("--expected-release-image-id", required=True)
    parser.add_argument("--expected-optimizer-build-image-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = replay_rc3_gate_e_performance_v1(
            args.gate_e_evidence_root,
            frozen_candidate_root=args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
            expected_release_image_id=args.expected_release_image_id,
            expected_optimizer_build_image_id=args.expected_optimizer_build_image_id,
        )
    except PerformanceReplayError as error:
        print(f"RC3 Gate E performance replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
