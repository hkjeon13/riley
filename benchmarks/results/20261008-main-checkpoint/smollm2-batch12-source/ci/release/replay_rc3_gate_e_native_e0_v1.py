#!/usr/bin/env python3
"""Replay the native half of RC3 canonical E0 through held evidence FDs.

This is deliberately one component of the four-gate RC3 Gate E semantic
closure.  It begins and ends with the fixed Gate E input-inventory replay,
copies only the held native raw archive into a new private temporary directory,
and invokes the existing native tensor replayer only on that scratch copy.
The input evidence is never reopened by pathname.  A successful result proves
only the native canonical-E0 component; optimizer E0, Python-free,
performance, soak, aggregate Gate E, and qualification remain out of scope.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, NoReturn, Sequence, TypeVar

_BYTECODE_DISABLED_AT_STARTUP = bool(sys.flags.dont_write_bytecode)
_BYTECODE_DISABLED_ON_MODULE_ENTRY = sys.dont_write_bytecode
sys.dont_write_bytecode = True

import provenance_v2_common as common
import rc3_frozen_candidate_common as frozen
import rc3_frozen_candidate_topology as topology
import replay_rc3_frozen_candidate_v1 as frozen_replayer
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs


REPLAY_VERSION = "riley.rc3-gate-e-native-e0-semantic-replay.v1"
SCOPE = "gate-e-native-e0-semantic-component-only"
AUTHORITY = "gate-e-native-e0-semantic-replay-only"
NATIVE_RAW_SCHEMA_VERSION = "riley.native-correctness-raw-evidence.v1"
MAX_NATIVE_RAW_BYTES = 16 * 1024 * 1024 * 1024

CHECK_NAMES = (
    "closed-gate-e-input-inventory-replayed-before-native-e0",
    "frozen-source-archive-bound-to-native-e0",
    "native-e0-raw-tensor-evidence-replayed",
    "native-e0-report-and-executable-cross-bound",
    "native-e0-component-does-not-aggregate-gate-e",
)

NOT_ESTABLISHED = {
    "optimizer_e0": "not-established",
    "python_free": "not-established",
    "performance": "not-established",
    "soak": "not-established",
    "gate_e_pass": "not-established",
    "semantic_receipt": "not-established",
    "qualification": "not-established",
    "deployment": "not-established",
}

T = TypeVar("T")


class NativeE0ReplayError(ValueError):
    """The native canonical-E0 component cannot be replayed safely."""


@dataclass(frozen=True)
class _NativeE0Artifacts:
    report: common.EvidenceDescriptor
    raw_evidence: common.EvidenceDescriptor
    candidate_executable: common.EvidenceDescriptor


@dataclass(frozen=True)
class _ScratchSnapshot:
    path: Path
    root_fd: int
    descriptor: common.EvidenceDescriptor
    root_identity: tuple[int, ...]
    leaf_identity: tuple[int, ...]


def _fail(code: str, message: str) -> NoReturn:
    error = NativeE0ReplayError(f"{code}: {message}")
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


def _native_e0_artifacts(inventory: gate_inputs.GateEInventory) -> _NativeE0Artifacts:
    artifacts = dict(inventory.artifacts)
    try:
        return _NativeE0Artifacts(
            report=artifacts["canonical_e0.native_report"],
            raw_evidence=artifacts["canonical_e0.native_raw_evidence"],
            candidate_executable=artifacts["release.native_candidate_executable"],
        )
    except KeyError as error:  # pragma: no cover - parser fixes this set
        _fail("invalid-gate-e-inventory", f"missing fixed native E0 role: {error}")


def _preflight_native_e0_raw(
    gate_e_evidence_root_fd: int,
) -> tuple[common.EvidenceDescriptor, _NativeE0Artifacts]:
    """Reject an oversized native raw leaf before any full Gate E stream replay."""

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
    artifacts = _native_e0_artifacts(inventory)
    if artifacts.raw_evidence.byte_length > MAX_NATIVE_RAW_BYTES:
        _fail(
            "native-e0-raw-evidence-too-large",
            "native E0 raw evidence exceeds its semantic replay byte bound",
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
            f"Gate E inventory changed between native E0 {phase} and structural replay",
        )


def _frozen_source_archive(
    frozen_candidate_root_fd: int,
    structural_result: dict[str, Any],
) -> common.EvidenceDescriptor:
    _raw, manifest_descriptor, manifest = _frozen_replay(
        lambda: frozen_replayer._read_manifest(frozen_candidate_root_fd)
    )
    expected_descriptor = structural_result.get("frozen_candidate_manifest")
    if manifest_descriptor.as_json() != expected_descriptor:
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


def _snapshot_direct_descriptor(
    gate_e_evidence_root_fd: int,
    descriptor: common.EvidenceDescriptor,
    scratch_root: Path,
) -> _ScratchSnapshot:
    """Copy one Gate E direct leaf solely through the verified held FD."""

    target_name = "native-e0-raw-evidence.tar"
    target = scratch_root / target_name
    scratch_descriptor = common.EvidenceDescriptor(
        path=target_name,
        sha256=descriptor.sha256,
        byte_length=descriptor.byte_length,
    )
    scratch_fd: int | None = None
    try:
        scratch_fd = _common(
            lambda: common.open_private_evidence_directory(
                scratch_root,
                "native E0 private scratch root",
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
                    _fail("scratch-snapshot-failed", f"cannot write native E0 scratch: {error}")
                if written < 1:
                    _fail("scratch-snapshot-failed", "native E0 scratch write was incomplete")
                offset += written

        def copy_from_held_file(source: BinaryIO) -> None:
            output_fd = -1
            remaining = descriptor.byte_length
            try:
                try:
                    output_fd = os.open(target_name, create_flags, 0o600, dir_fd=scratch_fd)
                except FileExistsError as error:
                    _fail("scratch-snapshot-failed", f"native E0 scratch leaf already exists: {error}")
                except (NotImplementedError, OSError, TypeError) as error:
                    _fail("scratch-snapshot-failed", f"cannot create native E0 scratch leaf: {error}")
                os.fchmod(output_fd, 0o600)
                while remaining:
                    chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                    if not chunk:
                        _fail("truncated-input", "native E0 raw evidence changed while it was snapshotted")
                    write_all(output_fd, chunk)
                    remaining -= len(chunk)
                if source.read(1):
                    _fail("mutated-input", "native E0 raw evidence grew while it was snapshotted")
                os.fsync(output_fd)
            except OSError as error:
                _fail("scratch-snapshot-failed", f"cannot snapshot native E0 raw evidence: {error}")
            finally:
                _close_quietly(output_fd)

        _common(
            lambda: common.consume_descriptor_file(
                gate_e_evidence_root_fd,
                descriptor,
                "Gate E canonical E0 native raw evidence",
                copy_from_held_file,
                maximum_bytes=MAX_NATIVE_RAW_BYTES,
            )
        )
        _common(
            lambda: common.verify_private_snapshot_descriptor_file(
                scratch_fd,
                scratch_descriptor,
                "native E0 private scratch snapshot",
                maximum_bytes=MAX_NATIVE_RAW_BYTES,
            )
        )
        try:
            root_identity = _stable_identity(os.fstat(scratch_fd))
            leaf_identity = _stable_identity(
                os.lstat(scratch_descriptor.path, dir_fd=scratch_fd)
            )
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot retain native E0 scratch identity: {error}")
        snapshot = _ScratchSnapshot(
            path=target,
            root_fd=scratch_fd,
            descriptor=scratch_descriptor,
            root_identity=root_identity,
            leaf_identity=leaf_identity,
        )
        scratch_fd = None
        return snapshot
    finally:
        _close_quietly(scratch_fd)


def _require_scratch_snapshot_unchanged(snapshot: _ScratchSnapshot, *, phase: str) -> None:
    """Reject a scratch-root or leaf replacement around pathname-only legacy replay."""

    def verify_identity() -> None:
        try:
            held_root = _stable_identity(os.fstat(snapshot.root_fd))
            visible_root = _stable_identity(os.lstat(snapshot.path.parent))
            visible_leaf = _stable_identity(
                os.lstat(snapshot.descriptor.path, dir_fd=snapshot.root_fd)
            )
        except OSError as error:
            _fail("scratch-snapshot-mutated", f"cannot inspect native E0 scratch after {phase}: {error}")
        if held_root != snapshot.root_identity or visible_root != snapshot.root_identity:
            _fail("scratch-snapshot-mutated", f"native E0 scratch root changed during {phase}")
        if visible_leaf != snapshot.leaf_identity:
            _fail("scratch-snapshot-mutated", f"native E0 scratch leaf changed during {phase}")

    verify_identity()
    _common(
        lambda: common.verify_private_snapshot_descriptor_file(
            snapshot.root_fd,
            snapshot.descriptor,
            "native E0 private scratch snapshot",
            maximum_bytes=MAX_NATIVE_RAW_BYTES,
        )
    )
    verify_identity()


def _replay_native_raw(snapshot: Path, *, source_revision: str) -> Any:
    """Run the legacy tensor verifier only against our private scratch copy."""

    try:
        import check_native_correctness_evidence as native_evidence
    except ModuleNotFoundError as error:
        if error.name == "tomllib":
            _fail(
                "native-e0-runtime-requires-tomllib",
                "native E0 semantic replay requires Python 3.11+ or the reviewed tomllib compatibility wrapper",
            )
        raise
    if native_evidence.MAX_RAW_ARCHIVE_BYTES != MAX_NATIVE_RAW_BYTES:
        _fail("native-e0-policy-drift", "native raw archive byte bound changed without this adapter")
    try:
        return native_evidence.replay_raw_evidence(
            snapshot,
            source_revision=source_revision,
            require_passing_report=True,
        )
    except native_evidence.NativeCorrectnessEvidenceError as error:
        _fail("native-e0-raw-replay-failed", str(error))
    except OSError as error:
        _fail("native-e0-raw-replay-failed", str(error))


def _require_native_result_bindings(
    result: Any,
    *,
    source_revision: str,
    source_archive: common.EvidenceDescriptor,
    artifacts: _NativeE0Artifacts,
) -> None:
    expected = {
        "schema_version": NATIVE_RAW_SCHEMA_VERSION,
        "source_revision": source_revision,
        "source_archive_sha256": source_archive.sha256,
        "source_archive_byte_length": source_archive.byte_length,
        "correctness_report_sha256": artifacts.report.sha256,
        "correctness_report_byte_length": artifacts.report.byte_length,
        "candidate_executable_sha256": artifacts.candidate_executable.sha256,
        "candidate_executable_byte_length": artifacts.candidate_executable.byte_length,
        "case_count": 31,
        "failure_count": 0,
    }
    for field, value in expected.items():
        if getattr(result, field, None) != value:
            _fail(
                f"native-{field}-mismatch",
                f"native E0 raw replay {field} does not match the fixed Gate E/frozen binding",
            )


def _replay_rc3_gate_e_native_e0_v1_on_held_fds(
    gate_e_evidence_root_fd: int,
    frozen_candidate_root_fd: int,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
) -> dict[str, Any]:
    """Replay native canonical E0 while retaining all caller-held root FDs."""

    _require_bytecode_cache_disabled()
    preflight_inventory, artifacts = _preflight_native_e0_raw(gate_e_evidence_root_fd)
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
    source_revision = structural_start["source_revision"]
    if type(source_revision) is not str:
        _fail("invalid-frozen-candidate", "structural Gate E replay returned no source revision")

    with tempfile.TemporaryDirectory(prefix="riley-gate-e-native-e0-") as temporary:
        scratch_root = Path(temporary)
        try:
            scratch_root.chmod(0o700)
        except OSError as error:
            _fail("scratch-snapshot-failed", f"cannot make native E0 scratch private: {error}")
        snapshot = _snapshot_direct_descriptor(
            gate_e_evidence_root_fd,
            artifacts.raw_evidence,
            scratch_root,
        )
        try:
            native_result = _replay_native_raw(snapshot.path, source_revision=source_revision)
            _require_scratch_snapshot_unchanged(snapshot, phase="legacy native replay")
        finally:
            _close_quietly(snapshot.root_fd)
    _require_native_result_bindings(
        native_result,
        source_revision=source_revision,
        source_archive=source_archive,
        artifacts=artifacts,
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
        _fail("gate-e-input-replay-drift", "Gate E structural inputs changed during native E0 replay")
    return {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "native_e0_status": "passed",
        "candidate_id": structural_start["candidate_id"],
        "source_revision": source_revision,
        "gate_e_input_inventory": structural_start["gate_e_input_inventory"],
        "frozen_candidate_manifest": structural_start["frozen_candidate_manifest"],
        "native_e0": {
            "report": artifacts.report.as_json(),
            "raw_evidence": artifacts.raw_evidence.as_json(),
            "candidate_executable": artifacts.candidate_executable.as_json(),
            "source_archive": source_archive.as_json(),
        },
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "not_established": dict(NOT_ESTABLISHED),
        "reason_codes": [],
    }


def replay_rc3_gate_e_native_e0_v1(
    gate_e_evidence_root: Path,
    *,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Open and lock four disjoint roots for one native E0 component replay."""

    _require_bytecode_cache_disabled()
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
        roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
            "frozen candidate root": (frozen_root, frozen_root_fd),
            "Gate E evidence root": (gate_root, gate_root_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        _shared_lock(input_root_fd, "freeze-input evidence root")
        _shared_lock(frozen_root_fd, "frozen candidate root")
        _shared_lock(gate_root_fd, "Gate E evidence root")
        result = _replay_rc3_gate_e_native_e0_v1_on_held_fds(
            gate_root_fd,
            frozen_root_fd,
            input_root_fd,
            source_root,
            source_root_fd,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-e-evidence-root", required=True, type=Path)
    parser.add_argument("--frozen-candidate-root", required=True, type=Path)
    parser.add_argument("--input-evidence-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = replay_rc3_gate_e_native_e0_v1(
            args.gate_e_evidence_root,
            frozen_candidate_root=args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
        )
    except NativeE0ReplayError as error:
        print(f"RC3 Gate E native E0 replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
