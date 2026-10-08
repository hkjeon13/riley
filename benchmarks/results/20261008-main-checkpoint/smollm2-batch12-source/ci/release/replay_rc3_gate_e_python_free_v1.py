#!/usr/bin/env python3
"""Replay the Python-free RC3 Gate E component through held evidence FDs.

This adapter deliberately consumes only the semantic subset that the frozen
Gate E v1 inventory can close: the Python-free raw/report/golden, canonical
native report, release bundle, and frozen source/release/model identities. It
does *not* invoke the legacy full E2E ``evaluate`` path, because that path
reopens producer-sidecars, model directories, weights, and source paths that
are not members of the v1 inventory.

The two path-based legacy operations receive only private, mode-0600 scratch
copies of the raw tar and release bundle.  All other Gate E inputs are read
through caller-held descriptors.  A successful result establishes only this
Python-free semantic component; it is not an aggregate Gate E, upstream E0,
qualification, or deployment decision.
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


REPLAY_VERSION = "riley.rc3-gate-e-python-free-semantic-replay.v1"
SCOPE = "gate-e-python-free-semantic-component-only"
AUTHORITY = "gate-e-python-free-semantic-replay-only"
PYTHON_FREE_RAW_SCHEMA = "riley.python-free-release-e2e-raw.v2"
PYTHON_FREE_REPORT_SCHEMA = "riley.release-gate-attestation.v1"
PYTHON_FREE_GATE = "python-free-clean-runtime-e2e"
MAX_PYTHON_FREE_RAW_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024 + 64 * 1024 * 1024
MAX_PYTHON_FREE_JSON_BYTES = 16 * 1024 * 1024
MAX_RELEASE_BUNDLE_BYTES = 1024 * 1024 * 1024 + 16 * 1024 * 1024
MAX_RELEASE_ELF_BYTES = 512 * 1024 * 1024
MAX_PYTHON_FREE_RAW_RETAINED_BYTES = 768 * 1024 * 1024
MAX_PYTHON_FREE_RELEASE_BUNDLE_RETAINED_BYTES = 640 * 1024 * 1024
MAX_PYTHON_FREE_SCRATCH_BYTES = (
    MAX_PYTHON_FREE_RAW_ARCHIVE_BYTES + MAX_RELEASE_BUNDLE_BYTES
)
EXTERNAL_SCRATCH_PARENT = Path("/var/tmp")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

PYTHON_FREE_CHECK_IDS = (
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
)

CHECK_NAMES = (
    "closed-gate-e-input-inventory-replayed-before-python-free",
    "externally-supplied-release-image-and-golden-anchors-bound",
    "frozen-source-release-and-model-identities-bound",
    "private-release-bundle-and-python-free-raw-replayed",
    "bounded-retained-memory-policy-applied-to-legacy-replay",
    "python-free-component-does-not-aggregate-gate-e",
)

NOT_ESTABLISHED = {
    "native_e0": "not-established",
    "optimizer_e0": "not-established",
    "release_image_review": "not-established",
    "correctness_golden_review": "not-established",
    "release_container_content": "not-established",
    "model_mount_provenance": "not-established",
    "producer_sidecar_equality": "not-established",
    "source_archive_content": "not-established",
    "performance": "not-established",
    "soak": "not-established",
    "gate_e_pass": "not-established",
    "semantic_receipt": "not-established",
    "qualification": "not-established",
    "deployment": "not-established",
}

T = TypeVar("T")


class PythonFreeReplayError(ValueError):
    """The Python-free Gate E component cannot be replayed safely."""


@dataclass(frozen=True)
class _PythonFreeArtifacts:
    report: common.EvidenceDescriptor
    raw_evidence: common.EvidenceDescriptor
    correctness_golden: common.EvidenceDescriptor
    release_bundle: common.EvidenceDescriptor
    native_report: common.EvidenceDescriptor


@dataclass(frozen=True)
class _FrozenBindings:
    source_archive: common.EvidenceDescriptor
    release_elf: common.EvidenceDescriptor
    release_image_digest: str
    models: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _ScratchSnapshots:
    root: Path
    root_fd: int
    descriptors: Mapping[str, common.EvidenceDescriptor]
    paths: Mapping[str, Path]
    root_identity: tuple[int, ...]
    leaf_identities: Mapping[str, tuple[int, ...]]


def _fail(code: str, message: str) -> NoReturn:
    error = PythonFreeReplayError(f"{code}: {message}")
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


def _expected_release_image_id(value: str) -> str:
    if type(value) is not str or IMAGE_ID_RE.fullmatch(value) is None:
        _fail(
            "invalid-expected-release-image-id",
            "expected release image ID must be sha256:<64 lowercase hex>",
        )
    if value == "sha256:" + "0" * 64:
        _fail(
            "invalid-expected-release-image-id",
            "expected release image ID must not be the zero digest",
        )
    return value


def _expected_golden_sha256(value: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        _fail(
            "invalid-expected-correctness-golden-sha256",
            "expected correctness golden SHA-256 must be 64 lowercase hex characters",
        )
    if value == "0" * 64:
        _fail(
            "invalid-expected-correctness-golden-sha256",
            "expected correctness golden SHA-256 must not be the zero digest",
        )
    return value


def _python_free_artifacts(inventory: gate_inputs.GateEInventory) -> _PythonFreeArtifacts:
    artifacts = dict(inventory.artifacts)
    try:
        return _PythonFreeArtifacts(
            report=artifacts["python_free.report"],
            raw_evidence=artifacts["python_free.raw_evidence"],
            correctness_golden=artifacts["python_free.correctness_golden"],
            release_bundle=artifacts["release.bundle"],
            native_report=artifacts["canonical_e0.native_report"],
        )
    except KeyError as error:  # pragma: no cover - parser fixes this set
        _fail("invalid-gate-e-inventory", f"missing fixed Python-free role: {error}")


def _preflight_python_free_inputs(
    gate_e_evidence_root_fd: int,
) -> tuple[common.EvidenceDescriptor, _PythonFreeArtifacts]:
    """Reject bounded component inputs before the full Gate E stream replay."""

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
    artifacts = _python_free_artifacts(inventory)
    limits = {
        "Python-free raw evidence": (
            artifacts.raw_evidence,
            MAX_PYTHON_FREE_RAW_ARCHIVE_BYTES,
        ),
        "Python-free report": (artifacts.report, MAX_PYTHON_FREE_JSON_BYTES),
        "Python-free correctness golden": (
            artifacts.correctness_golden,
            MAX_PYTHON_FREE_JSON_BYTES,
        ),
        "Python-free release bundle": (
            artifacts.release_bundle,
            MAX_RELEASE_BUNDLE_BYTES,
        ),
        "Python-free native report": (
            artifacts.native_report,
            MAX_PYTHON_FREE_JSON_BYTES,
        ),
    }
    for label, (descriptor, maximum) in limits.items():
        if descriptor.byte_length > maximum:
            _fail(
                "python-free-input-too-large",
                f"{label} exceeds its Python-free semantic replay byte bound",
            )
    if (
        artifacts.raw_evidence.byte_length + artifacts.release_bundle.byte_length
        > MAX_PYTHON_FREE_SCRATCH_BYTES
    ):
        _fail(
            "python-free-input-too-large",
            "Python-free private scratch input total exceeds its byte bound",
        )
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
            f"Gate E inventory changed between Python-free {phase} and structural replay",
        )


def _frozen_bindings(
    frozen_candidate_root_fd: int,
    structural_result: Mapping[str, Any],
) -> _FrozenBindings:
    """Recover typed frozen source/release/model identities on held FDs."""

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
    request, _descriptors = _freeze_input(
        lambda: freeze_inputs._parse_request(bound_inputs)
    )
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
        _fail("python-free-input-too-large", "frozen release ELF exceeds its replay byte bound")
    image = _expected_release_image_id(image_digest)
    return _FrozenBindings(
        source_archive=archive,
        release_elf=release_elf,
        release_image_digest=image,
        models=tuple(models),
    )


def _snapshot_python_free_inputs(
    gate_e_evidence_root_fd: int,
    artifacts: _PythonFreeArtifacts,
    scratch_root: Path,
) -> _ScratchSnapshots:
    """Copy the two pathname-only inputs beneath one pinned private root FD."""

    source_inputs = (
        (
            "raw_evidence",
            artifacts.raw_evidence,
            "python-free-raw-evidence.tar",
            MAX_PYTHON_FREE_RAW_ARCHIVE_BYTES,
        ),
        (
            "release_bundle",
            artifacts.release_bundle,
            "python-free-release-bundle.tar.gz",
            MAX_RELEASE_BUNDLE_BYTES,
        ),
    )
    scratch_fd: int | None = None
    try:
        scratch_fd = _common(
            lambda: common.open_private_evidence_directory(
                scratch_root,
                "Python-free private scratch root",
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
                    _fail("scratch-snapshot-failed", f"cannot write Python-free scratch: {error}")
                if written < 1:
                    _fail("scratch-snapshot-failed", "Python-free scratch write was incomplete")
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
                        _fail("scratch-snapshot-failed", f"Python-free scratch leaf already exists: {error}")
                    except (NotImplementedError, OSError, TypeError) as error:
                        _fail("scratch-snapshot-failed", f"cannot create Python-free scratch leaf: {error}")
                    os.fchmod(output_fd, 0o600)
                    while remaining:
                        chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                        if not chunk:
                            _fail("truncated-input", "Python-free evidence changed while it was snapshotted")
                        write_all(output_fd, chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        _fail("mutated-input", "Python-free evidence grew while it was snapshotted")
                    os.fsync(output_fd)
                except OSError as error:
                    _fail("scratch-snapshot-failed", f"cannot snapshot Python-free evidence: {error}")
                finally:
                    _close_quietly(output_fd)

            _common(
                lambda source_descriptor=source_descriptor, maximum=maximum, role=role: common.consume_descriptor_file(
                    gate_e_evidence_root_fd,
                    source_descriptor,
                    f"Gate E Python-free {role}",
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
                    f"Python-free private scratch {role}",
                    maximum_bytes=MAX_PYTHON_FREE_SCRATCH_BYTES,
                )
            )
        try:
            root_identity = _stable_identity(os.fstat(scratch_fd))
            leaf_identities = {
                role: _stable_identity(os.lstat(descriptor.path, dir_fd=scratch_fd))
                for role, descriptor in descriptors.items()
            }
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot retain Python-free scratch identity: {error}")
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
    """Reject scratch root/leaf replacement around pathname-only legacy work."""

    def verify_identities() -> None:
        try:
            held_root = _stable_identity(os.fstat(snapshots.root_fd))
            visible_root = _stable_identity(os.lstat(snapshots.root))
            visible_leaves = {
                role: _stable_identity(os.lstat(descriptor.path, dir_fd=snapshots.root_fd))
                for role, descriptor in snapshots.descriptors.items()
            }
        except OSError as error:
            _fail("scratch-snapshot-mutated", f"cannot inspect Python-free scratch after {phase}: {error}")
        if held_root != snapshots.root_identity or visible_root != snapshots.root_identity:
            _fail("scratch-snapshot-mutated", f"Python-free scratch root changed during {phase}")
        if visible_leaves != snapshots.leaf_identities:
            _fail("scratch-snapshot-mutated", f"Python-free scratch leaf changed during {phase}")

    verify_identities()
    for role, descriptor in snapshots.descriptors.items():
        _common(
            lambda role=role, descriptor=descriptor: common.verify_private_snapshot_descriptor_file(
                snapshots.root_fd,
                descriptor,
                f"Python-free private scratch {role}",
                maximum_bytes=MAX_PYTHON_FREE_SCRATCH_BYTES,
            )
        )
    verify_identities()


def _read_strict_json_descriptor(
    directory_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
) -> dict[str, Any]:
    """Read a noncanonical legacy JSON report only through its held file FD."""

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
            maximum_bytes=MAX_PYTHON_FREE_JSON_BYTES,
        )
    )
    document = _common(
        lambda: common.parse_strict_json(
            raw,
            label,
            maximum_bytes=MAX_PYTHON_FREE_JSON_BYTES,
            require_object=True,
        )
    )
    assert type(document) is dict
    return document


def _verify_held_descriptor(
    directory_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
) -> None:
    _common(
        lambda: common.consume_descriptor_file(
            directory_fd,
            descriptor,
            label,
            lambda _source: None,
            maximum_bytes=MAX_PYTHON_FREE_JSON_BYTES,
        )
    )


def _load_python_free_e2e_contract() -> ModuleType:
    """Load the legacy contract lazily so Python 3.10 fails closed on tomllib."""

    script = Path(__file__).resolve().parents[2] / "benchmarks/scripts/check_python_free_release_e2e.py"
    try:
        spec = importlib.util.spec_from_file_location(
            "riley_gate_e_python_free_e2e_contract",
            script,
        )
        if spec is None or spec.loader is None:  # pragma: no cover - static path
            _fail("python-free-contract-load-failed", f"cannot load Python-free contract: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        if error.name == "tomllib":
            _fail(
                "python-free-runtime-requires-tomllib",
                "Python-free semantic replay requires Python 3.11+ or the reviewed tomllib compatibility wrapper",
            )
        raise
    expected_policy = {
        "RAW_SCHEMA": PYTHON_FREE_RAW_SCHEMA,
        "REPORT_SCHEMA": PYTHON_FREE_REPORT_SCHEMA,
        "GATE": PYTHON_FREE_GATE,
        "CHECK_IDS": PYTHON_FREE_CHECK_IDS,
        "MAX_RAW_ARCHIVE_BYTES": MAX_PYTHON_FREE_RAW_ARCHIVE_BYTES,
        "MAX_JSON_BYTES": MAX_PYTHON_FREE_JSON_BYTES,
    }
    for field, value in expected_policy.items():
        if getattr(module, field, None) != value:
            _fail("python-free-policy-drift", f"Python-free contract {field} changed without this adapter")
    if not callable(getattr(module, "verify_bound_release_bundle", None)):
        _fail("python-free-policy-drift", "Python-free contract has no public release bundle verifier")
    return module


def _replay_python_free_raw(
    snapshots: _ScratchSnapshots,
    *,
    source_revision: str,
    source_archive_sha256: str,
    release_binary_sha256: str,
    release_bundle_sha256: str,
    expected_release_image_id: str,
    native_report: Mapping[str, Any],
    native_report_sha256: str,
    correctness_golden_sha256: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Run only the closed raw replay against private scratch paths."""

    contract = _load_python_free_e2e_contract()
    try:
        contract.verify_bound_release_bundle(
            snapshots.paths["release_bundle"],
            release_binary_sha256=release_binary_sha256,
            source_revision=source_revision,
            max_uncompressed_bytes=MAX_PYTHON_FREE_RELEASE_BUNDLE_RETAINED_BYTES,
        )
        archive = contract.load_raw_evidence_archive(
            snapshots.paths["raw_evidence"],
            max_retained_bytes=MAX_PYTHON_FREE_RAW_RETAINED_BYTES,
        )
        replayed, diagnostic = contract.validate_bound_raw_archive(
            archive,
            source_revision=source_revision,
            source_archive_sha256=source_archive_sha256,
            release_binary_sha256=release_binary_sha256,
            release_bundle_sha256=release_bundle_sha256,
            image_id=expected_release_image_id,
            correctness_report=native_report,
            correctness_report_sha256=native_report_sha256,
            correctness_golden_sha256=correctness_golden_sha256,
        )
    except getattr(contract, "EvidenceError") as error:
        _fail("python-free-raw-replay-failed", str(error))
    except (OSError, TypeError) as error:
        _fail("python-free-raw-replay-failed", str(error))
    if type(replayed) is not dict or diagnostic is not None or replayed.get("status") != "passed":
        _fail("python-free-raw-replay-failed", diagnostic or "Python-free raw replay did not pass")
    if type(archive) is not dict or type(archive.get("raw")) is not dict:
        _fail("python-free-raw-replay-failed", "Python-free raw loader returned no typed raw document")
    return replayed, archive


def _require_python_free_result_bindings(
    replayed: Mapping[str, Any],
    submitted_report: Mapping[str, Any],
    *,
    source_revision: str,
    bindings: _FrozenBindings,
    artifacts: _PythonFreeArtifacts,
    expected_release_image_id: str,
    expected_correctness_golden_sha256: str,
) -> None:
    expected_source = {
        "git_revision": source_revision,
        "git_dirty": False,
        "source_archive_sha256": bindings.source_archive.sha256,
        "release_binary_sha256": bindings.release_elf.sha256,
        "release_bundle_sha256": artifacts.release_bundle.sha256,
        "release_image_sha256": expected_release_image_id.removeprefix("sha256:"),
    }
    _require_python_free_attestation(
        replayed,
        "replayed Python-free report",
        expected_source=expected_source,
        raw_evidence_sha256=artifacts.raw_evidence.sha256,
    )
    _require_python_free_attestation(
        submitted_report,
        "submitted Python-free report",
        expected_source=expected_source,
        raw_evidence_sha256=artifacts.raw_evidence.sha256,
    )
    if common.canonical_json_bytes(replayed) != common.canonical_json_bytes(submitted_report):
        _fail("python-free-report-mismatch", "submitted Python-free report differs from raw replay")
    if artifacts.correctness_golden.sha256 != expected_correctness_golden_sha256:
        _fail("python-free-golden-anchor-mismatch", "Gate E golden differs from the externally supplied anchor")
    if bindings.release_image_digest != expected_release_image_id:
        _fail("python-free-image-anchor-mismatch", "frozen release image digest differs from the externally supplied anchor")


def _require_python_free_attestation(
    report: Mapping[str, Any],
    label: str,
    *,
    expected_source: Mapping[str, Any],
    raw_evidence_sha256: str,
) -> None:
    """Require the exact, type-strict Python-free attestation contract.

    Strict JSON preserves integer and boolean tokens, but ordinary Python
    mapping equality treats ``0 == False`` and ``1 == True``.  Validate every
    typed field before canonical-byte equality so an attacker cannot publish a
    numerically substituted report descriptor as the passed raw attestation.
    """

    if type(report) is not dict:
        _fail("python-free-report-contract-mismatch", f"{label} must be an object")
    expected_keys = {
        "schema_version",
        "gate",
        "status",
        "source",
        "raw_evidence_sha256",
        "checks",
    }
    if set(report) != expected_keys:
        _fail("python-free-report-contract-mismatch", f"{label} has the wrong key set")
    if report["schema_version"] != PYTHON_FREE_REPORT_SCHEMA:
        _fail("python-free-report-contract-mismatch", f"{label} has the wrong schema")
    if report["gate"] != PYTHON_FREE_GATE or report["status"] != "passed":
        _fail("python-free-report-contract-mismatch", f"{label} has the wrong gate/status")
    source = report["source"]
    if type(source) is not dict:
        _fail("python-free-report-contract-mismatch", f"{label}.source must be an object")
    if set(source) != set(expected_source):
        _fail("python-free-report-contract-mismatch", f"{label}.source has the wrong key set")
    if type(source.get("git_revision")) is not str or GIT_REVISION_RE.fullmatch(source["git_revision"]) is None:
        _fail("python-free-report-contract-mismatch", f"{label}.source.git_revision is malformed")
    if source.get("git_dirty") is not False:
        _fail("python-free-report-contract-mismatch", f"{label}.source.git_dirty must be false")
    for field in (
        "source_archive_sha256",
        "release_binary_sha256",
        "release_bundle_sha256",
        "release_image_sha256",
    ):
        value = source.get(field)
        if type(value) is not str or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
            _fail("python-free-report-contract-mismatch", f"{label}.source.{field} is malformed")
    if source != expected_source:
        _fail("python-free-report-binding-mismatch", f"{label} does not bind frozen release/source inputs")
    actual_raw_sha256 = report["raw_evidence_sha256"]
    if (
        type(actual_raw_sha256) is not str
        or SHA256_RE.fullmatch(actual_raw_sha256) is None
        or actual_raw_sha256 == "0" * 64
    ):
        _fail("python-free-report-contract-mismatch", f"{label}.raw_evidence_sha256 is malformed")
    if actual_raw_sha256 != raw_evidence_sha256:
        _fail("python-free-report-binding-mismatch", f"{label} does not bind its Gate E raw evidence")
    checks = report["checks"]
    if type(checks) is not list or len(checks) != len(PYTHON_FREE_CHECK_IDS):
        _fail("python-free-report-contract-mismatch", f"{label}.checks has the wrong length")
    observed: set[str] = set()
    for index, check in enumerate(checks):
        if type(check) is not dict or set(check) != {"id", "passed"}:
            _fail("python-free-report-contract-mismatch", f"{label}.checks[{index}] has the wrong shape")
        check_id = check["id"]
        if type(check_id) is not str or not check_id or check_id in observed:
            _fail("python-free-report-contract-mismatch", f"{label}.checks[{index}].id is malformed")
        if check["passed"] is not True:
            _fail("python-free-report-contract-mismatch", f"{label}.checks[{index}].passed must be true")
        observed.add(check_id)
    if observed != set(PYTHON_FREE_CHECK_IDS):
        _fail("python-free-report-contract-mismatch", f"{label}.checks has the wrong ID set")


def _require_frozen_model_binding(
    archive: Mapping[str, Any],
    bindings: _FrozenBindings,
) -> None:
    """Bind raw model observations to one exact frozen model input row."""

    raw = archive.get("raw")
    if type(raw) is not dict:
        _fail("python-free-raw-model-mismatch", "Python-free raw archive has no raw object")
    model = raw.get("model")
    if type(model) is not dict:
        _fail("python-free-raw-model-mismatch", "Python-free raw document has no model object")
    fields = (
        "model_id",
        "model_revision",
        "model_tree_sha256",
        "config_sha256",
        "weights_sha256",
        "tokenizer_json_sha256",
    )
    if any(type(model.get(field)) is not str for field in fields):
        _fail("python-free-raw-model-mismatch", "Python-free raw model binding is malformed")
    matching = [row for row in bindings.models if row.get("model_id") == model["model_id"]]
    if len(matching) != 1:
        _fail("python-free-frozen-model-mismatch", "raw model ID does not select exactly one frozen model")
    frozen_model = matching[0]
    tree = frozen_model.get("tree")
    config = frozen_model.get("config")
    tokenizer = frozen_model.get("tokenizer")
    weights = frozen_model.get("weights")
    if (
        type(frozen_model.get("revision")) is not str
        or not isinstance(tree, common.EvidenceDescriptor)
        or not isinstance(config, common.EvidenceDescriptor)
        or not isinstance(tokenizer, common.EvidenceDescriptor)
        or type(weights) is not list
        or not all(isinstance(weight, common.EvidenceDescriptor) for weight in weights)
    ):
        _fail("invalid-frozen-candidate", "frozen model binding is malformed")
    expected = {
        "model_revision": frozen_model["revision"],
        "model_tree_sha256": tree.sha256,
        "config_sha256": config.sha256,
        "tokenizer_json_sha256": tokenizer.sha256,
    }
    for field, value in expected.items():
        if model[field] != value:
            _fail("python-free-frozen-model-mismatch", f"raw model {field} differs from frozen input")
    if len(weights) != 1:
        _fail(
            "python-free-frozen-model-mismatch",
            "the fixed single-safetensors Python-free archive requires exactly one frozen weight",
        )
    matching_weights = [weight for weight in weights if weight.sha256 == model["weights_sha256"]]
    if len(matching_weights) != 1:
        _fail("python-free-frozen-model-mismatch", "raw model weights do not select exactly one frozen weight")
    model_sizes = archive.get("model_sizes")
    model_manifest = archive.get("model_manifest")
    if type(model_sizes) is not dict or type(model_manifest) is not bytes:
        _fail("python-free-raw-model-mismatch", "Python-free raw archive has no model-size bindings")
    expected_sizes = {
        tree: len(model_manifest),
        config: model_sizes.get("config.json"),
        tokenizer: model_sizes.get("tokenizer.json"),
        matching_weights[0]: model_sizes.get("model.safetensors"),
    }
    for descriptor, expected_size in expected_sizes.items():
        if type(expected_size) is not int or descriptor.byte_length != expected_size:
            _fail(
                "python-free-frozen-model-mismatch",
                "raw model descriptor length differs from frozen input",
            )


def _replay_rc3_gate_e_python_free_v1_on_held_fds(
    gate_e_evidence_root_fd: int,
    frozen_candidate_root_fd: int,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
    scratch_parent: Path,
    expected_release_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    """Replay Python-free E2E while retaining every caller-held root FD."""

    _require_bytecode_cache_disabled()
    expected_image = _expected_release_image_id(expected_release_image_id)
    expected_golden = _expected_golden_sha256(expected_correctness_golden_sha256)
    preflight_inventory, artifacts = _preflight_python_free_inputs(gate_e_evidence_root_fd)
    if artifacts.correctness_golden.sha256 != expected_golden:
        _fail(
            "python-free-golden-anchor-mismatch",
            "Gate E golden differs from the externally supplied anchor",
        )
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
    if type(source_revision) is not str:
        _fail("invalid-frozen-candidate", "structural Gate E replay returned no source revision")
    if bindings.release_image_digest != expected_image:
        _fail(
            "python-free-image-anchor-mismatch",
            "frozen release image digest differs from the externally supplied anchor",
        )
    _verify_held_descriptor(
        gate_e_evidence_root_fd,
        artifacts.correctness_golden,
        "Gate E Python-free correctness golden",
    )
    submitted_report = _read_strict_json_descriptor(
        gate_e_evidence_root_fd,
        artifacts.report,
        "Gate E Python-free report",
    )
    native_report = _read_strict_json_descriptor(
        gate_e_evidence_root_fd,
        artifacts.native_report,
        "Gate E canonical native report",
    )
    _common(
        lambda: common.verify_descriptor_file(
            input_evidence_root_fd,
            bindings.release_elf,
            "frozen release ELF",
            maximum_bytes=MAX_RELEASE_ELF_BYTES,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="riley-gate-e-python-free-",
        dir=os.fspath(scratch_parent),
    ) as temporary:
        scratch_root = Path(temporary)
        try:
            scratch_root.chmod(0o700)
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot make Python-free scratch private: {error}")
        snapshots = _snapshot_python_free_inputs(
            gate_e_evidence_root_fd,
            artifacts,
            scratch_root,
        )
        try:
            replayed, archive = _replay_python_free_raw(
                snapshots,
                source_revision=source_revision,
                source_archive_sha256=bindings.source_archive.sha256,
                release_binary_sha256=bindings.release_elf.sha256,
                release_bundle_sha256=artifacts.release_bundle.sha256,
                expected_release_image_id=expected_image,
                native_report=native_report,
                native_report_sha256=artifacts.native_report.sha256,
                correctness_golden_sha256=expected_golden,
            )
            _require_scratch_snapshots_unchanged(
                snapshots,
                phase="legacy Python-free replay",
            )
        finally:
            _close_quietly(snapshots.root_fd)
    _require_python_free_result_bindings(
        replayed,
        submitted_report,
        source_revision=source_revision,
        bindings=bindings,
        artifacts=artifacts,
        expected_release_image_id=expected_image,
        expected_correctness_golden_sha256=expected_golden,
    )
    _require_frozen_model_binding(archive, bindings)
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
        _fail("gate-e-input-replay-drift", "Gate E structural inputs changed during Python-free replay")
    return {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "python_free_status": "passed",
        "candidate_id": structural_start["candidate_id"],
        "source_revision": source_revision,
        "expected_release_image_id": expected_image,
        "expected_correctness_golden_sha256": expected_golden,
        "gate_e_input_inventory": structural_start["gate_e_input_inventory"],
        "frozen_candidate_manifest": structural_start["frozen_candidate_manifest"],
        "python_free": {
            "report": artifacts.report.as_json(),
            "raw_evidence": artifacts.raw_evidence.as_json(),
            "correctness_golden": artifacts.correctness_golden.as_json(),
            "release_bundle": artifacts.release_bundle.as_json(),
            "native_report": artifacts.native_report.as_json(),
            "release_elf": bindings.release_elf.as_json(),
            "source_archive": bindings.source_archive.as_json(),
            "frozen_release_image_digest": bindings.release_image_digest,
            "legacy_replay_retained_byte_limits": {
                "raw_evidence": MAX_PYTHON_FREE_RAW_RETAINED_BYTES,
                "release_bundle_uncompressed": MAX_PYTHON_FREE_RELEASE_BUNDLE_RETAINED_BYTES,
            },
        },
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "not_established": dict(NOT_ESTABLISHED),
        "reason_codes": [],
    }


def replay_rc3_gate_e_python_free_v1(
    gate_e_evidence_root: Path,
    *,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
    expected_release_image_id: str,
    expected_correctness_golden_sha256: str,
) -> dict[str, Any]:
    """Open and lock roots for one Python-free semantic component replay."""

    _require_bytecode_cache_disabled()
    _expected_release_image_id(expected_release_image_id)
    _expected_golden_sha256(expected_correctness_golden_sha256)
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
            "Python-free external scratch parent",
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
                "Python-free external scratch parent",
            )
        )
        roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
            "frozen candidate root": (frozen_root, frozen_root_fd),
            "Gate E evidence root": (gate_root, gate_root_fd),
            "Python-free external scratch parent": (scratch_parent, scratch_parent_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        _shared_lock(input_root_fd, "freeze-input evidence root")
        _shared_lock(frozen_root_fd, "frozen candidate root")
        _shared_lock(gate_root_fd, "Gate E evidence root")
        result = _replay_rc3_gate_e_python_free_v1_on_held_fds(
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
        result = replay_rc3_gate_e_python_free_v1(
            args.gate_e_evidence_root,
            frozen_candidate_root=args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
            expected_release_image_id=args.expected_release_image_id,
            expected_correctness_golden_sha256=args.expected_correctness_golden_sha256,
        )
    except PythonFreeReplayError as error:
        print(f"RC3 Gate E Python-free replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
