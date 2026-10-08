#!/usr/bin/env python3
"""Replay the RC3 Gate E soak component through held evidence descriptors.

This adapter closes only the soak semantic component.  It reads the Gate E
inventory before and after replay, holds the frozen/source roots open for the
entire operation, copies the raw soak archive once into a private mode-0600
scratch leaf, and calls the soak checker's FD-only public contract.  It does
not make a Gate E aggregate, semantic-receipt, qualification, deployment, or
capture claim.

The checker contract used here is deliberately path-minimal:

``validate_bound_reliability_soak_evidence(report, raw_evidence_fd, *,
correctness_golden_raw, native_correctness_report_raw, source_revision,
source_archive_sha256, release_binary_sha256, release_image_id, candidate_id,
correctness_golden_sha256, native_correctness_report_sha256,
raw_evidence_sha256, raw_evidence_byte_length, model_tree_sha256)``.

The three JSON documents are parsed only after bounded reads through held file
descriptors.  The raw archive is represented only by a held descriptor opened
below the private scratch root.  No producer-side path is passed to the
contract.
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
import provenance_v2_common as common
import rc3_frozen_candidate_common as frozen
import rc3_frozen_candidate_topology as topology
import replay_rc3_frozen_candidate_v1 as frozen_replayer
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs


REPLAY_VERSION = "riley.rc3-gate-e-soak-semantic-replay.v1"
SCOPE = "gate-e-soak-semantic-component-only"
AUTHORITY = "gate-e-soak-semantic-replay-only"
SOAK_POLICY_VERSION = "riley.reliability-soak-bound-semantic.v1"
EXPECTED_SOAK_POLICY_SHA256 = (
    "380ca5ae59da9e4945df26ea2d124652b784655799e63e264b65f313d614ba9d"
)
SOAK_REPORT_VERSION = "riley.reliability-soak-report.v2"

# The raw USTAR format contains a 512 MiB event stream, two 4 MiB documents,
# four 16 MiB inspect receipts, and the fixed small runtime receipts.
MAX_SOAK_RAW_ARCHIVE_BYTES = 613_556_224
MAX_SOAK_RAW_STREAM_MEMBER_BYTES = 512 * 1024 * 1024
MAX_SOAK_RAW_SCRATCH_BYTES = MAX_SOAK_RAW_ARCHIVE_BYTES
MAX_SOAK_REPORT_BYTES = 16 * 1024 * 1024
MAX_CORRECTNESS_GOLDEN_BYTES = 64 * 1024
MAX_NATIVE_CORRECTNESS_REPORT_BYTES = 16 * 1024 * 1024
MAX_RELEASE_ELF_BYTES = 512 * 1024 * 1024
MAX_SOURCE_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
EXTERNAL_SCRATCH_PARENT = Path("/var/tmp")

IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

CHECK_NAMES = (
    "closed-gate-e-input-inventory-replayed-before-soak",
    "externally-supplied-release-image-and-golden-anchors-bound",
    "frozen-source-release-and-model-identities-bound",
    "held-soak-report-golden-and-native-report-consumed",
    "private-held-fd-soak-raw-replayed",
    "bounded-streaming-raw-policy-applied-to-soak-raw-replay",
    "soak-component-does-not-aggregate-gate-e",
)

NOT_ESTABLISHED = {
    "native_e0": "not-established",
    "optimizer_e0": "not-established",
    "python_free": "not-established",
    "performance": "not-established",
    "release_image_review": "not-established",
    "correctness_golden_review": "not-established",
    "release_container_content": "not-established",
    "model_mount_provenance": "not-established",
    "producer_sidecar_equality": "not-established",
    "source_archive_content": "not-established",
    "actual_capture": "not-established",
    "aggregate_gate_e": "not-established",
    "gate_e_pass": "not-established",
    "semantic_receipt": "not-established",
    "qualification": "not-established",
    "deployment": "not-established",
}

T = TypeVar("T")


class SoakReplayError(ValueError):
    """The soak Gate E component cannot be replayed safely."""


@dataclass(frozen=True)
class _SoakArtifacts:
    report: common.EvidenceDescriptor
    raw_evidence: common.EvidenceDescriptor
    correctness_golden: common.EvidenceDescriptor
    native_report: common.EvidenceDescriptor


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
    error = SoakReplayError(f"{code}: {message}")
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
        _fail("evidence-root-lock-unavailable", f"cannot acquire shared {label} lock: {error}")


def _unlock_quietly(directory_fd: int | None) -> None:
    if directory_fd is not None:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
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
        _fail(f"invalid-{label}", f"{label.replace('-', ' ')} must be sha256:<64 lowercase hex>")
    if value == "sha256:" + "0" * 64:
        _fail(f"invalid-{label}", f"{label.replace('-', ' ')} must not be the zero digest")
    return value


def _expected_sha256(value: str, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"invalid-{label}", f"{label.replace('-', ' ')} must be nonzero lowercase SHA-256")
    return value


def _soak_artifacts(inventory: gate_inputs.GateEInventory) -> _SoakArtifacts:
    artifacts = dict(inventory.artifacts)
    try:
        return _SoakArtifacts(
            report=artifacts["soak.report"],
            raw_evidence=artifacts["soak.raw_evidence"],
            correctness_golden=artifacts["python_free.correctness_golden"],
            native_report=artifacts["canonical_e0.native_report"],
        )
    except KeyError as error:  # pragma: no cover - inventory parser fixes this set.
        _fail("invalid-gate-e-inventory", f"missing fixed soak role: {error}")


def _preflight_soak_inputs(
    gate_e_evidence_root_fd: int,
) -> tuple[common.EvidenceDescriptor, _SoakArtifacts]:
    """Reject component-specific size abuse before full Gate E replay."""

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
    artifacts = _soak_artifacts(inventory)
    limits = {
        "soak raw evidence": (artifacts.raw_evidence, MAX_SOAK_RAW_ARCHIVE_BYTES),
        "soak report": (artifacts.report, MAX_SOAK_REPORT_BYTES),
        "shared correctness golden": (artifacts.correctness_golden, MAX_CORRECTNESS_GOLDEN_BYTES),
        "canonical native report": (artifacts.native_report, MAX_NATIVE_CORRECTNESS_REPORT_BYTES),
    }
    for label, (descriptor, maximum) in limits.items():
        if descriptor.byte_length > maximum:
            _fail("soak-input-too-large", f"{label} exceeds its soak semantic replay byte bound")
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
            f"Gate E inventory changed between soak {phase} and structural replay",
        )


def _frozen_bindings(
    frozen_candidate_root_fd: int,
    structural_result: Mapping[str, Any],
) -> _FrozenBindings:
    """Recover typed frozen source/release/model identity bindings."""

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
    if archive.byte_length < 1 or release_elf.byte_length < 1:
        _fail("invalid-frozen-candidate", "frozen source/release descriptors must be nonempty")
    if archive.byte_length > MAX_SOURCE_ARCHIVE_BYTES or release_elf.byte_length > MAX_RELEASE_ELF_BYTES:
        _fail("soak-input-too-large", "frozen source/release descriptor exceeds soak replay bounds")
    return _FrozenBindings(
        source_archive=archive,
        release_elf=release_elf,
        release_image_digest=_expected_image_id(container.get("image_digest"), "expected-release-image-id"),
        models=tuple(models),
    )


def _read_strict_json_descriptor(
    directory_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
    *,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    """Read a strict JSON evidence leaf only through its held descriptor."""

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
    return document, raw


def _snapshot_soak_raw(
    gate_e_evidence_root_fd: int,
    descriptor: common.EvidenceDescriptor,
    scratch_root: Path,
) -> _ScratchSnapshot:
    """Copy raw soak evidence once to a private mode-0600 scratch leaf."""

    scratch_fd: int | None = None
    target_name = "soak-raw-evidence.tar"
    scratch_descriptor = common.EvidenceDescriptor(
        path=target_name,
        sha256=descriptor.sha256,
        byte_length=descriptor.byte_length,
    )
    try:
        scratch_fd = _common(
            lambda: common.open_private_evidence_directory(
                scratch_root,
                "soak private scratch root",
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
                    _fail("scratch-snapshot-failed", f"cannot write soak scratch: {error}")
                if written < 1:
                    _fail("scratch-snapshot-failed", "soak scratch write was incomplete")
                offset += written

        def copy_from_held_file(source: BinaryIO) -> None:
            output_fd = -1
            remaining = descriptor.byte_length
            try:
                try:
                    output_fd = os.open(target_name, create_flags, 0o600, dir_fd=scratch_fd)
                except FileExistsError as error:
                    _fail("scratch-snapshot-failed", f"soak scratch leaf already exists: {error}")
                except (NotImplementedError, OSError, TypeError) as error:
                    _fail("scratch-snapshot-failed", f"cannot create soak scratch leaf: {error}")
                os.fchmod(output_fd, 0o600)
                while remaining:
                    chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                    if not chunk:
                        _fail("truncated-input", "soak raw evidence changed while snapshotted")
                    write_all(output_fd, chunk)
                    remaining -= len(chunk)
                if source.read(1):
                    _fail("mutated-input", "soak raw evidence grew while snapshotted")
                os.fsync(output_fd)
            except OSError as error:
                _fail("scratch-snapshot-failed", f"cannot snapshot soak raw evidence: {error}")
            finally:
                _close_quietly(output_fd)

        _common(
            lambda: common.consume_descriptor_file(
                gate_e_evidence_root_fd,
                descriptor,
                "Gate E soak raw evidence",
                copy_from_held_file,
                maximum_bytes=MAX_SOAK_RAW_ARCHIVE_BYTES,
            )
        )
        _common(
            lambda: common.verify_private_snapshot_descriptor_file(
                scratch_fd,
                scratch_descriptor,
                "soak private scratch raw evidence",
                maximum_bytes=MAX_SOAK_RAW_ARCHIVE_BYTES,
            )
        )
        try:
            root_identity = _stable_identity(os.fstat(scratch_fd))
            leaf_identity = _stable_identity(os.lstat(target_name, dir_fd=scratch_fd))
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot retain soak scratch identity: {error}")
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


def _require_scratch_snapshot_unchanged(snapshot: _ScratchSnapshot, *, phase: str) -> None:
    """Reject scratch root/leaf replacement around held-FD replay."""

    try:
        held_root = _stable_identity(os.fstat(snapshot.root_fd))
        visible_root = _stable_identity(os.lstat(snapshot.root))
        visible_leaf = _stable_identity(os.lstat(snapshot.name, dir_fd=snapshot.root_fd))
    except OSError as error:
        _fail("scratch-snapshot-mutated", f"cannot inspect soak scratch after {phase}: {error}")
    if held_root != snapshot.root_identity or visible_root != snapshot.root_identity:
        _fail("scratch-snapshot-mutated", f"soak scratch root changed during {phase}")
    if visible_leaf != snapshot.leaf_identity:
        _fail("scratch-snapshot-mutated", f"soak scratch leaf changed during {phase}")
    _common(
        lambda: common.verify_private_snapshot_descriptor_file(
            snapshot.root_fd,
            snapshot.descriptor,
            "soak private scratch raw evidence",
            maximum_bytes=MAX_SOAK_RAW_ARCHIVE_BYTES,
        )
    )


def _open_scratch_snapshot_fd(snapshot: _ScratchSnapshot) -> int:
    """Open raw evidence through the held scratch-root FD only."""

    nofollow, _directory, cloexec, nonblock = _common(common.require_safe_open_flags)
    flags = os.O_RDONLY | nofollow | cloexec | nonblock
    descriptor = -1
    try:
        descriptor = os.open(snapshot.name, flags, dir_fd=snapshot.root_fd)
        metadata = os.fstat(descriptor)
    except (NotImplementedError, OSError, TypeError) as error:
        _close_quietly(descriptor)
        _fail("scratch-snapshot-failed", f"cannot open held soak scratch leaf: {error}")
    if _stable_identity(metadata) != snapshot.leaf_identity:
        _close_quietly(descriptor)
        _fail("scratch-snapshot-mutated", "soak scratch leaf changed before raw replay")
    if metadata.st_size != snapshot.descriptor.byte_length:
        _close_quietly(descriptor)
        _fail("scratch-snapshot-mutated", "soak scratch leaf length changed before raw replay")
    return descriptor


def _load_soak_contract() -> ModuleType:
    """Load and pin the public FD-only soak semantic contract."""

    script = Path(__file__).resolve().parents[2] / "benchmarks/scripts/check_reliability_soak.py"
    try:
        spec = importlib.util.spec_from_file_location("riley_gate_e_soak_contract", script)
        if spec is None or spec.loader is None:  # pragma: no cover - static path.
            _fail("soak-contract-load-failed", f"cannot load soak contract: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        _fail("soak-contract-load-failed", str(error))
    expected_policy = {
        "BOUND_SEMANTIC_POLICY_VERSION": SOAK_POLICY_VERSION,
        "MAX_RAW_ARCHIVE_BYTES": MAX_SOAK_RAW_ARCHIVE_BYTES,
        "MAX_BOUND_RAW_STREAM_MEMBER_BYTES": MAX_SOAK_RAW_STREAM_MEMBER_BYTES,
        "MAX_BOUND_RAW_SCRATCH_BYTES": MAX_SOAK_RAW_SCRATCH_BYTES,
        "MAX_CORRECTNESS_GOLDEN_BYTES": MAX_CORRECTNESS_GOLDEN_BYTES,
        "MAX_NATIVE_CORRECTNESS_REPORT_BYTES": MAX_NATIVE_CORRECTNESS_REPORT_BYTES,
    }
    for field, expected in expected_policy.items():
        if getattr(module, field, None) != expected:
            _fail("soak-policy-drift", f"soak contract {field} changed without this adapter")
    validator = getattr(module, "validate_bound_reliability_soak_evidence", None)
    raw_replayer = getattr(module, "replay_bound_raw_evidence_fd", None)
    policy_digest = getattr(module, "bound_semantic_policy_sha256", None)
    if not callable(validator) or not callable(raw_replayer) or not callable(policy_digest):
        _fail("soak-policy-drift", "soak contract lacks the public bound replay API")
    if policy_digest() != EXPECTED_SOAK_POLICY_SHA256:
        _fail("soak-policy-drift", "soak contract policy digest changed without this adapter")
    return module


def _require_frozen_model_binding(
    golden: Mapping[str, Any],
    native_report: Mapping[str, Any],
    bindings: _FrozenBindings,
    *,
    source_revision: str,
    native_report_sha256: str,
    release_binary_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    """Bind golden/native model fields to exactly one frozen candidate model."""

    expected_golden_keys = {
        "schema_version",
        "correctness_gate_id",
        "correctness_report_sha256",
        "source_revision",
        "model_id",
        "model_revision",
        "config_sha256",
        "weights_sha256",
        "tokenizer_aggregate_sha256",
        "tokenizer_json_sha256",
        "prompt",
        "max_tokens",
        "expected_greedy_text_sha256",
    }
    if type(golden) is not dict or set(golden) != expected_golden_keys:
        _fail("soak-correctness-golden-contract-mismatch", "shared correctness golden has the wrong shape")
    if golden.get("source_revision") != source_revision:
        _fail("soak-frozen-model-mismatch", "shared golden does not bind frozen source revision")
    model_id = golden.get("model_id")
    model_revision = golden.get("model_revision")
    if type(model_id) is not str or not model_id or type(model_revision) is not str or not model_revision:
        _fail("soak-correctness-golden-contract-mismatch", "shared golden model identity is malformed")
    for field in (
        "correctness_report_sha256",
        "config_sha256",
        "weights_sha256",
        "tokenizer_aggregate_sha256",
        "tokenizer_json_sha256",
        "expected_greedy_text_sha256",
    ):
        _expected_sha256(golden.get(field), f"correctness-golden-{field}")
    if golden["correctness_report_sha256"] != native_report_sha256:
        _fail(
            "soak-native-report-binding-mismatch",
            "shared golden does not hash the held canonical native report",
        )
    matching = [row for row in bindings.models if row.get("model_id") == model_id]
    if len(matching) != 1:
        _fail("soak-frozen-model-mismatch", "golden model ID does not select exactly one frozen model")
    model = matching[0]
    tree = model.get("tree")
    config = model.get("config")
    tokenizer = model.get("tokenizer")
    weights = model.get("weights")
    if (
        model.get("revision") != model_revision
        or not isinstance(tree, common.EvidenceDescriptor)
        or not isinstance(config, common.EvidenceDescriptor)
        or not isinstance(tokenizer, common.EvidenceDescriptor)
        or type(weights) is not list
        or not all(isinstance(weight, common.EvidenceDescriptor) for weight in weights)
    ):
        _fail("soak-frozen-model-mismatch", "frozen model does not match shared golden identity")
    if config.sha256 != golden["config_sha256"] or tokenizer.sha256 != golden["tokenizer_json_sha256"]:
        _fail("soak-frozen-model-mismatch", "shared golden config/tokenizer differs from frozen model")
    if len([weight for weight in weights if weight.sha256 == golden["weights_sha256"]]) != 1:
        _fail("soak-frozen-model-mismatch", "shared golden weight does not select one frozen model weight")
    if type(native_report) is not dict or native_report.get("status") != "pass":
        _fail("soak-native-report-contract-mismatch", "canonical native report must be passing")
    native_bindings = native_report.get("bindings")
    if type(native_bindings) is not dict:
        _fail("soak-native-report-contract-mismatch", "canonical native report has no bindings")
    expected_native = {
        "candidate_git_revision": source_revision,
        "model_id": model_id,
        "model_revision": model_revision,
        "config_sha256": config.sha256,
        "weights_sha256": golden["weights_sha256"],
        "tokenizer_sha256": golden["tokenizer_aggregate_sha256"],
    }
    for field, expected in expected_native.items():
        if native_bindings.get(field) != expected:
            _fail("soak-native-report-binding-mismatch", f"canonical native report {field} differs from frozen/golden input")
    if native_bindings.get("candidate_executable_sha256") != release_binary_sha256:
        _fail(
            "soak-native-report-binding-mismatch",
            "canonical native report does not bind the frozen release ELF",
        )
    return model, tree.sha256


def _require_soak_report_bindings(
    report: Mapping[str, Any],
    *,
    source_revision: str,
    source_archive_sha256: str,
    release_binary_sha256: str,
    release_image_id: str,
    model: Mapping[str, Any],
    model_tree_sha256: str,
) -> None:
    """Require source/release/model bindings carried by a passed soak report."""

    if type(report) is not dict or report.get("schema_version") != SOAK_REPORT_VERSION:
        _fail("soak-report-contract-mismatch", "soak report has the wrong schema")
    if report.get("status") != "passed" or report.get("passed") is not True:
        _fail("soak-report-contract-mismatch", "soak report is not passing")
    report_bindings = report.get("bindings")
    if type(report_bindings) is not dict:
        _fail("soak-report-contract-mismatch", "soak report has no bindings")
    source = report_bindings.get("source")
    if type(source) is not dict:
        _fail("soak-report-contract-mismatch", "soak report has no bound source")
    expected = {
        "git_commit": source_revision,
        "git_dirty": False,
        "source_archive_sha256": source_archive_sha256,
        "binary_sha256": release_binary_sha256,
        "image_sha256": release_image_id.removeprefix("sha256:"),
        "model_sha256": model_tree_sha256,
        "model_id": model.get("model_id"),
        "model_revision": model.get("revision"),
    }
    if source != expected:
        _fail("soak-report-binding-mismatch", "soak report does not bind frozen source/release/model inputs")


def _replay_soak_raw(
    snapshot: _ScratchSnapshot,
    submitted_report: Mapping[str, Any],
    correctness_golden_raw: bytes,
    native_correctness_report_raw: bytes,
    *,
    source_revision: str,
    bindings: _FrozenBindings,
    artifacts: _SoakArtifacts,
    expected_release_image_id: str,
    candidate_id: str,
    model_tree_sha256: str,
) -> dict[str, Any]:
    """Call only the checker's FD-only public contract on private scratch."""

    contract = _load_soak_contract()
    raw_fd = _open_scratch_snapshot_fd(snapshot)
    try:
        result = contract.validate_bound_reliability_soak_evidence(
            submitted_report,
            raw_fd,
            correctness_golden_raw=correctness_golden_raw,
            native_correctness_report_raw=native_correctness_report_raw,
            source_revision=source_revision,
            source_archive_sha256=bindings.source_archive.sha256,
            release_binary_sha256=bindings.release_elf.sha256,
            release_image_id=expected_release_image_id,
            candidate_id=candidate_id,
            correctness_golden_sha256=artifacts.correctness_golden.sha256,
            native_correctness_report_sha256=artifacts.native_report.sha256,
            raw_evidence_sha256=artifacts.raw_evidence.sha256,
            raw_evidence_byte_length=artifacts.raw_evidence.byte_length,
            model_tree_sha256=model_tree_sha256,
        )
    except (ValueError, OSError, TypeError) as error:
        _fail("soak-raw-replay-failed", str(error))
    finally:
        _close_quietly(raw_fd)
    if type(result) is not dict:
        _fail("soak-raw-replay-failed", "soak contract returned no typed result")
    return result


def _require_replayed_result(
    result: Mapping[str, Any],
    submitted_report: Mapping[str, Any],
    *,
    artifacts: _SoakArtifacts,
) -> None:
    """Bind direct-contract output to every held Gate E leaf."""

    replayed = result.get("report")
    if type(replayed) is not dict:
        _fail("soak-raw-replay-failed", "soak contract returned no replayed report")
    if common.canonical_json_bytes(replayed) != common.canonical_json_bytes(submitted_report):
        _fail("soak-report-mismatch", "submitted soak report differs from held-FD raw replay")
    expected = {
        "raw_evidence_sha256": artifacts.raw_evidence.sha256,
        "raw_evidence_byte_length": artifacts.raw_evidence.byte_length,
        "correctness_golden_sha256": artifacts.correctness_golden.sha256,
        "native_correctness_report_sha256": artifacts.native_report.sha256,
        "raw_stream_member_byte_limit": MAX_SOAK_RAW_STREAM_MEMBER_BYTES,
        "scratch_disk_byte_limit": MAX_SOAK_RAW_SCRATCH_BYTES,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            _fail("soak-raw-replay-binding-mismatch", f"soak contract result {field} differs from held input")


def _replay_rc3_gate_e_soak_v1_on_held_fds(
    gate_e_evidence_root_fd: int,
    frozen_candidate_root_fd: int,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
    scratch_parent: Path,
    expected_release_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    """Replay one soak component while retaining all root descriptors."""

    _require_bytecode_cache_disabled()
    expected_image = _expected_image_id(expected_release_image_id, "expected-release-image-id")
    expected_golden = _expected_sha256(
        expected_correctness_golden_sha256,
        "expected-correctness-golden-sha256",
    )
    preflight_inventory, artifacts = _preflight_soak_inputs(gate_e_evidence_root_fd)
    if artifacts.correctness_golden.sha256 != expected_golden:
        _fail("soak-golden-anchor-mismatch", "Gate E golden differs from the externally supplied anchor")
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
    if type(candidate_id) is not str or not candidate_id:
        _fail("invalid-frozen-candidate", "structural Gate E replay returned no candidate ID")
    if bindings.release_image_digest != expected_image:
        _fail("soak-release-image-anchor-mismatch", "frozen release image differs from the externally supplied anchor")
    submitted_report, _report_raw = _read_strict_json_descriptor(
        gate_e_evidence_root_fd,
        artifacts.report,
        "Gate E soak report",
        maximum_bytes=MAX_SOAK_REPORT_BYTES,
    )
    correctness_golden, correctness_golden_raw = _read_strict_json_descriptor(
        gate_e_evidence_root_fd,
        artifacts.correctness_golden,
        "Gate E shared correctness golden",
        maximum_bytes=MAX_CORRECTNESS_GOLDEN_BYTES,
    )
    native_report, native_correctness_report_raw = _read_strict_json_descriptor(
        gate_e_evidence_root_fd,
        artifacts.native_report,
        "Gate E canonical native report",
        maximum_bytes=MAX_NATIVE_CORRECTNESS_REPORT_BYTES,
    )
    _common(
        lambda: common.verify_descriptor_file(
            input_evidence_root_fd,
            bindings.source_archive,
            "frozen source archive",
            maximum_bytes=MAX_SOURCE_ARCHIVE_BYTES,
        )
    )
    _common(
        lambda: common.verify_descriptor_file(
            input_evidence_root_fd,
            bindings.release_elf,
            "frozen release ELF",
            maximum_bytes=MAX_RELEASE_ELF_BYTES,
        )
    )
    model, model_tree_sha256 = _require_frozen_model_binding(
        correctness_golden,
        native_report,
        bindings,
        source_revision=source_revision,
        native_report_sha256=artifacts.native_report.sha256,
        release_binary_sha256=bindings.release_elf.sha256,
    )
    _require_soak_report_bindings(
        submitted_report,
        source_revision=source_revision,
        source_archive_sha256=bindings.source_archive.sha256,
        release_binary_sha256=bindings.release_elf.sha256,
        release_image_id=expected_image,
        model=model,
        model_tree_sha256=model_tree_sha256,
    )
    with tempfile.TemporaryDirectory(
        prefix="riley-gate-e-soak-",
        dir=os.fspath(scratch_parent),
    ) as temporary:
        scratch_root = Path(temporary)
        try:
            scratch_root.chmod(0o700)
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot make soak scratch private: {error}")
        snapshot = _snapshot_soak_raw(gate_e_evidence_root_fd, artifacts.raw_evidence, scratch_root)
        try:
            result = _replay_soak_raw(
                snapshot,
                submitted_report,
                correctness_golden_raw,
                native_correctness_report_raw,
                source_revision=source_revision,
                bindings=bindings,
                artifacts=artifacts,
                expected_release_image_id=expected_image,
                candidate_id=candidate_id,
                model_tree_sha256=model_tree_sha256,
            )
            _require_scratch_snapshot_unchanged(snapshot, phase="held-FD soak raw replay")
        finally:
            _close_quietly(snapshot.root_fd)
    _require_replayed_result(result, submitted_report, artifacts=artifacts)
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
        _fail("gate-e-input-replay-drift", "Gate E structural inputs changed during soak replay")
    return {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "soak_status": "passed",
        "candidate_id": candidate_id,
        "source_revision": source_revision,
        "expected_release_image_id": expected_image,
        "expected_correctness_golden_sha256": expected_golden,
        "gate_e_input_inventory": structural_start["gate_e_input_inventory"],
        "frozen_candidate_manifest": structural_start["frozen_candidate_manifest"],
        "soak": {
            "report": artifacts.report.as_json(),
            "raw_evidence": artifacts.raw_evidence.as_json(),
            "correctness_golden": artifacts.correctness_golden.as_json(),
            "native_report": artifacts.native_report.as_json(),
            "release_elf": bindings.release_elf.as_json(),
            "source_archive": bindings.source_archive.as_json(),
            "frozen_release_image_digest": bindings.release_image_digest,
            "model_tree_sha256": model_tree_sha256,
            "raw_stream_member_byte_limit": MAX_SOAK_RAW_STREAM_MEMBER_BYTES,
            "scratch_disk_byte_limit": MAX_SOAK_RAW_SCRATCH_BYTES,
        },
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "not_established": dict(NOT_ESTABLISHED),
        "reason_codes": [],
    }


def replay_rc3_gate_e_soak_v1(
    gate_e_evidence_root: Path,
    *,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
    expected_release_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    """Open, lock, and retain all roots for a single soak component replay."""

    _require_bytecode_cache_disabled()
    _expected_image_id(expected_release_image_id, "expected-release-image-id")
    _expected_sha256(expected_correctness_golden_sha256, "expected-correctness-golden-sha256")
    gate_root = _frozen(lambda: frozen.normalized_absolute_path(gate_e_evidence_root, "--gate-e-evidence-root"))
    frozen_root = _frozen(lambda: frozen.normalized_absolute_path(frozen_candidate_root, "--frozen-candidate-root"))
    input_root = _frozen(lambda: frozen.normalized_absolute_path(input_evidence_root, "--input-evidence-root"))
    source_root = _frozen(lambda: frozen.normalized_absolute_path(repository_root, "--repository-root"))
    scratch_parent = _frozen(
        lambda: frozen.normalized_absolute_path(EXTERNAL_SCRATCH_PARENT, "soak external scratch parent")
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
            lambda: common.open_absolute_directory(scratch_parent, "soak external scratch parent")
        )
        roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
            "frozen candidate root": (frozen_root, frozen_root_fd),
            "Gate E evidence root": (gate_root, gate_root_fd),
            "soak external scratch parent": (scratch_parent, scratch_parent_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        _shared_lock(input_root_fd, "freeze-input evidence root")
        _shared_lock(frozen_root_fd, "frozen candidate root")
        _shared_lock(gate_root_fd, "Gate E evidence root")
        result = _replay_rc3_gate_e_soak_v1_on_held_fds(
            gate_root_fd,
            frozen_root_fd,
            input_root_fd,
            source_root,
            source_root_fd,
            scratch_parent,
            expected_release_image_id,
            expected_correctness_golden_sha256,
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
    parser.add_argument("--expected-correctness-golden-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = replay_rc3_gate_e_soak_v1(
            args.gate_e_evidence_root,
            frozen_candidate_root=args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
            expected_release_image_id=args.expected_release_image_id,
            expected_correctness_golden_sha256=args.expected_correctness_golden_sha256,
        )
    except SoakReplayError as error:
        print(f"RC3 Gate E soak replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
