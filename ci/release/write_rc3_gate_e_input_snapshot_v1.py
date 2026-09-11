#!/usr/bin/env python3
"""Create one opaque, closed RC3 Gate E input snapshot.

This preparer copies fourteen caller-selected, already-produced regular files
into a fresh private root under fixed direct names, writes the canonical
``gate-e-inputs.json`` last, and self-replays that structural inventory while
the source-checkout, freeze-input, frozen-candidate, and Gate E root
descriptors remain held.  The fourteen opaque source-file descriptors are
held only while each individual copy is made.  It is deliberately only an
immutable local input snapshot: it does not
run a producer, GPU, model, container, network action, semantic replay,
receipt, or qualification decision.

In particular, the fresh snapshot cannot establish the origin or atomic
cross-leaf coherence of the caller's source files, writer normal return,
actual Gate E producer normal return, evidence-root immutability, Gate E
semantic pass, durable receipt, deployment, or qualification.  A future
authenticated producer must capture its raw leaves in the same invocation and
must rehash this exact root again before it can make any stronger claim.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence, TypeVar


_BYTECODE_DISABLED_AT_STARTUP = bool(sys.flags.dont_write_bytecode)
_BYTECODE_DISABLED_ON_MODULE_ENTRY = sys.dont_write_bytecode
sys.dont_write_bytecode = True

import provenance_v2_common as common
import rc3_frozen_candidate_common as frozen
import rc3_frozen_candidate_topology as topology
import replay_rc3_frozen_candidate_v1 as frozen_replayer
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs


# Leave room for the canonical inventory itself: the structural replayer's
# 1-TiB closure limit counts both the fourteen snapshots and the inventory.
MAX_SNAPSHOT_TOTAL_BYTES = gate_inputs.MAX_GATE_E_INPUT_BYTES - gate_inputs.MAX_INVENTORY_BYTES


@dataclass(frozen=True)
class GateEInputSnapshotSources:
    """The fixed fourteen opaque source roles for one fresh snapshot.

    These paths are copied by value and are intentionally not recorded in the
    resulting inventory.  The inventory describes only immutable leaves below
    its own private root.
    """

    release_bundle: Path
    release_profile_binary: Path
    release_native_candidate_executable: Path
    canonical_e0_native_report: Path
    canonical_e0_native_raw_evidence: Path
    canonical_e0_optimizer_report: Path
    canonical_e0_optimizer_raw_evidence: Path
    python_free_report: Path
    python_free_raw_evidence: Path
    python_free_correctness_golden: Path
    performance_report: Path
    performance_raw_evidence: Path
    soak_report: Path
    soak_raw_evidence: Path


@dataclass(frozen=True)
class _SourceSpec:
    group: str
    field: str
    attribute: str
    snapshot_name: str
    option: str
    require_owner_executable: bool = False


SOURCE_SPECS = (
    _SourceSpec("release", "bundle", "release_bundle", "release-bundle.tar", "--release-bundle"),
    _SourceSpec(
        "release",
        "profile_binary",
        "release_profile_binary",
        "release-profile-binary",
        "--release-profile-binary",
        require_owner_executable=True,
    ),
    _SourceSpec(
        "release",
        "native_candidate_executable",
        "release_native_candidate_executable",
        "release-native-candidate-executable",
        "--release-native-candidate-executable",
        require_owner_executable=True,
    ),
    _SourceSpec(
        "canonical_e0",
        "native_report",
        "canonical_e0_native_report",
        "canonical-e0-native-report.json",
        "--canonical-e0-native-report",
    ),
    _SourceSpec(
        "canonical_e0",
        "native_raw_evidence",
        "canonical_e0_native_raw_evidence",
        "canonical-e0-native-raw.tar",
        "--canonical-e0-native-raw-evidence",
    ),
    _SourceSpec(
        "canonical_e0",
        "optimizer_report",
        "canonical_e0_optimizer_report",
        "canonical-e0-optimizer-report.json",
        "--canonical-e0-optimizer-report",
    ),
    _SourceSpec(
        "canonical_e0",
        "optimizer_raw_evidence",
        "canonical_e0_optimizer_raw_evidence",
        "canonical-e0-optimizer-raw.tar",
        "--canonical-e0-optimizer-raw-evidence",
    ),
    _SourceSpec(
        "python_free",
        "report",
        "python_free_report",
        "python-free-report.json",
        "--python-free-report",
    ),
    _SourceSpec(
        "python_free",
        "raw_evidence",
        "python_free_raw_evidence",
        "python-free-raw.tar",
        "--python-free-raw-evidence",
    ),
    _SourceSpec(
        "python_free",
        "correctness_golden",
        "python_free_correctness_golden",
        "python-free-correctness-golden.json",
        "--python-free-correctness-golden",
    ),
    _SourceSpec(
        "performance",
        "report",
        "performance_report",
        "performance-report.json",
        "--performance-report",
    ),
    _SourceSpec(
        "performance",
        "raw_evidence",
        "performance_raw_evidence",
        "performance-raw.tar",
        "--performance-raw-evidence",
    ),
    _SourceSpec("soak", "report", "soak_report", "soak-report.json", "--soak-report"),
    _SourceSpec(
        "soak",
        "raw_evidence",
        "soak_raw_evidence",
        "soak-raw.tar",
        "--soak-raw-evidence",
    ),
)


class GateEInputSnapshotError(ValueError):
    """The requested Gate E input snapshot is unsafe or incomplete."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = GateEInputSnapshotError(f"{code}: {message}")
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


def _frozen_replay(call: Callable[[], T]) -> T:
    try:
        return call()
    except frozen_replayer.FrozenCandidateReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-frozen-candidate"), str(error))


def _gate(call: Callable[[], T]) -> T:
    try:
        return call()
    except gate_inputs.GateEInventoryReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-gate-e-inventory"), str(error))


def _topology(call: Callable[[], T]) -> T:
    try:
        return call()
    except topology.FrozenCandidateTopologyError as error:
        _fail(getattr(error, "reason_code", "unsafe-gate-e-topology"), str(error))


def _require_bytecode_cache_disabled() -> None:
    if not (
        _BYTECODE_DISABLED_AT_STARTUP and _BYTECODE_DISABLED_ON_MODULE_ENTRY
    ):
        _fail(
            "bytecode-cache-write-not-permitted",
            "invoke this writer with python3 -B or PYTHONDONTWRITEBYTECODE=1",
        )


def _lock(directory_fd: int, operation: int, label: str) -> None:
    try:
        fcntl.flock(directory_fd, operation | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("evidence-root-lock-unavailable", f"cannot acquire {label} lock: {error}")


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


def _normalized_sources(sources: GateEInputSnapshotSources) -> dict[str, Path]:
    if not isinstance(sources, GateEInputSnapshotSources):
        _fail("invalid-gate-e-input-sources", "sources must be a GateEInputSnapshotSources")
    normalized: dict[str, Path] = {}
    seen: dict[Path, str] = {}
    for spec in SOURCE_SPECS:
        source = _frozen(
            lambda spec=spec: frozen.normalized_absolute_path(
                getattr(sources, spec.attribute),
                spec.option,
            )
        )
        previous = seen.get(source)
        if previous is not None:
            _fail(
                "duplicate-gate-e-input-source",
                f"{spec.option} duplicates source path supplied for {previous}",
            )
        seen[source] = spec.option
        normalized[spec.attribute] = source
    return normalized


def _snapshot_inventory_artifacts(
    gate_root_fd: int,
    sources: dict[str, Path],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Create every fixed leaf before the inventory control plane exists."""

    remaining = MAX_SNAPSHOT_TOTAL_BYTES
    groups: dict[str, dict[str, dict[str, Any]]] = {
        group_name: {} for group_name, _fields in gate_inputs.GATE_E_INPUT_GROUP_FIELDS
    }
    for spec in SOURCE_SPECS:
        if remaining < 1:
            _fail(
                "external-evidence-byte-budget-exceeded",
                "Gate E snapshots exhausted their total byte budget before all roles were copied",
            )
        created = _common(
            lambda spec=spec: common.snapshot_absolute_regular_create_only(
                sources[spec.attribute],
                gate_root_fd,
                spec.snapshot_name,
                f"Gate E input {spec.group}.{spec.field}",
                maximum_bytes=remaining,
                minimum_bytes=1,
                require_owner_executable=spec.require_owner_executable,
            )
        )
        descriptor = created.descriptor(
            spec.snapshot_name,
            f"Gate E input {spec.group}.{spec.field}",
        )
        _common(
            lambda descriptor=descriptor, spec=spec: common.verify_private_snapshot_descriptor_file(
                gate_root_fd,
                descriptor,
                f"Gate E input {spec.group}.{spec.field}",
                maximum_bytes=descriptor.byte_length,
            )
        )
        groups[spec.group][spec.field] = descriptor.as_json()
        remaining -= descriptor.byte_length
    if sum(len(values) for values in groups.values()) != gate_inputs.MAX_GATE_E_INPUT_DESCRIPTORS:
        raise AssertionError("fixed Gate E snapshot role count drifted")
    return groups


def _inventory_from_frozen_replay(
    frozen_result: dict[str, Any],
    artifacts: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    candidate_id = frozen_result.get("candidate_id")
    source_revision = frozen_result.get("source_revision")
    manifest = frozen_result.get("frozen_candidate_manifest")
    if type(candidate_id) is not str or type(source_revision) is not str or type(manifest) is not dict:
        _fail("invalid-frozen-candidate", "held frozen-candidate replay returned an incomplete identity")
    # Parsing here ensures the inventory never writes an arbitrary mapping as
    # its cross-root frozen manifest descriptor.
    _common(lambda: common.parse_descriptor(manifest, "held frozen-candidate manifest"))
    inventory: dict[str, Any] = {
        "schema_version": gate_inputs.INVENTORY_VERSION,
        "candidate_id": candidate_id,
        "source_revision": source_revision,
        "frozen_candidate_manifest": manifest,
    }
    for group_name, fields in gate_inputs.GATE_E_INPUT_GROUP_FIELDS:
        group = artifacts.get(group_name)
        if type(group) is not dict or set(group) != set(fields):
            _fail("invalid-gate-e-input-snapshot", f"fixed {group_name} snapshot role set drifted")
        inventory[group_name] = group
    raw = _common(lambda: common.canonical_json_bytes(inventory))
    if len(raw) > gate_inputs.MAX_INVENTORY_BYTES:
        _fail("gate-e-inventory-too-large", "canonical Gate E input inventory exceeds its byte bound")
    return inventory


def _require_created_inventory_binding(
    replay: dict[str, Any],
    inventory_descriptor: dict[str, Any],
) -> None:
    if replay.get("gate_e_input_inventory") != inventory_descriptor:
        _fail(
            "gate-e-inventory-self-replay-mismatch",
            "self-replay did not return the durable created Gate E inventory descriptor",
        )
    if replay.get("status") != "bound" or replay.get("qualification_status") != "not-run":
        _fail("gate-e-inventory-self-replay-mismatch", "self-replay returned an unexpected structural status")


def _reverify_private_snapshot_output(
    gate_root_fd: int,
    artifacts: dict[str, dict[str, dict[str, Any]]],
    inventory: dict[str, Any],
    inventory_descriptor: dict[str, Any],
) -> None:
    """Recheck private output modes after the generic structural self-replay.

    The inventory replayer intentionally accepts generic regular evidence
    leaves, because it is also a read-only consumer of independently prepared
    roots.  This writer promises stricter freshly-created ``0600`` snapshots,
    so it must restore that private-output check after the structural replay
    and before returning from its own producer branch.
    """

    for spec in SOURCE_SPECS:
        descriptor_value = artifacts[spec.group][spec.field]
        descriptor = _common(
            lambda descriptor_value=descriptor_value, spec=spec: common.parse_descriptor(
                descriptor_value,
                f"created Gate E input {spec.group}.{spec.field}",
            )
        )
        _common(
            lambda descriptor=descriptor, spec=spec: common.verify_private_snapshot_descriptor_file(
                gate_root_fd,
                descriptor,
                f"created Gate E input {spec.group}.{spec.field}",
                maximum_bytes=descriptor.byte_length,
            )
        )
    parsed_inventory_descriptor = _common(
        lambda: common.parse_descriptor(inventory_descriptor, "created Gate E input inventory")
    )
    _common(
        lambda: common.verify_private_snapshot_descriptor_file(
            gate_root_fd,
            parsed_inventory_descriptor,
            "created Gate E input inventory",
            maximum_bytes=parsed_inventory_descriptor.byte_length,
        )
    )
    parsed_inventory = _gate(lambda: gate_inputs._parse_gate_e_inventory(inventory))  # noqa: SLF001
    _gate(
        lambda: gate_inputs._bound_inventory_closure(  # noqa: SLF001
            parsed_inventory_descriptor,
            parsed_inventory,
        )
    )
    _gate(lambda: gate_inputs._assert_exact_entries(gate_root_fd, parsed_inventory))  # noqa: SLF001


def write_rc3_gate_e_input_snapshot_v1(
    gate_e_evidence_root: Path,
    *,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
    sources: GateEInputSnapshotSources,
) -> dict[str, Any]:
    """Copy opaque sources into one fresh structural Gate E input root.

    The fresh output root must not exist.  Every source is independently
    snapshot through no-follow descriptors.  The canonical inventory is the
    last leaf and is self-replayed before this function returns; a failure
    retains any partial root for forensic inspection and cannot be resumed.
    """

    _require_bytecode_cache_disabled()
    _common(common.require_safe_open_flags)
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
    normalized_sources = _normalized_sources(sources)

    source_root_fd: int | None = None
    input_root_fd: int | None = None
    frozen_root_fd: int | None = None
    gate_root_fd: int | None = None
    try:
        source_root_fd = _common(
            lambda: common.open_absolute_directory(source_root, "source checkout")
        )
        input_root_fd = _common(
            lambda: common.open_private_evidence_directory(
                input_root,
                "freeze-input evidence root",
            )
        )
        frozen_root_fd = _common(
            lambda: common.open_private_evidence_directory(
                frozen_root,
                "frozen candidate root",
            )
        )
        input_roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
            "frozen candidate root": (frozen_root, frozen_root_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(input_roots))
        _topology(lambda: topology.assert_new_root_parent_external(gate_root, input_roots))
        _lock(input_root_fd, fcntl.LOCK_SH, "freeze-input evidence root shared")
        _lock(frozen_root_fd, fcntl.LOCK_SH, "frozen candidate root shared")
        frozen_result = _frozen_replay(
            lambda: frozen_replayer._replay_rc3_frozen_candidate_v1_on_held_fds(  # noqa: SLF001
                frozen_root_fd,
                input_root_fd,
                source_root,
                source_root_fd,
            )
        )
        if type(frozen_result) is not dict:
            _fail("invalid-frozen-candidate", "held frozen-candidate replay returned no object")
        gate_root_fd = _common(
            lambda: common.create_private_evidence_directory(gate_root, "Gate E evidence root")
        )
        _topology(lambda: topology.require_visible_root(gate_root, gate_root_fd, "Gate E evidence root"))
        roots = {
            **input_roots,
            "Gate E evidence root": (gate_root, gate_root_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        _lock(gate_root_fd, fcntl.LOCK_EX, "Gate E evidence root exclusive")
        _frozen(
            lambda: frozen.require_distinct_root_fds(
                {
                    "Gate E evidence root": gate_root_fd,
                    "frozen candidate root": frozen_root_fd,
                    "freeze-input evidence root": input_root_fd,
                    "source checkout": source_root_fd,
                }
            )
        )
        artifacts = _snapshot_inventory_artifacts(gate_root_fd, normalized_sources)
        inventory = _inventory_from_frozen_replay(frozen_result, artifacts)
        inventory_created = _common(
            lambda: common.write_create_only_json(
                gate_root_fd,
                gate_inputs.INVENTORY_NAME,
                inventory,
                "Gate E input inventory",
            )
        )
        inventory_descriptor = inventory_created.descriptor(
            gate_inputs.INVENTORY_NAME,
            "created Gate E input inventory",
        ).as_json()
        replay = _gate(
            lambda: gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds(  # noqa: SLF001
                gate_root_fd,
                frozen_root_fd,
                input_root_fd,
                source_root,
                source_root_fd,
            )
        )
        _require_created_inventory_binding(replay, inventory_descriptor)
        _reverify_private_snapshot_output(
            gate_root_fd,
            artifacts,
            inventory,
            inventory_descriptor,
        )
        _topology(lambda: topology.require_visible_root(gate_root, gate_root_fd, "Gate E evidence root"))
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        return {
            "gate_e_input_inventory": inventory_descriptor,
            "replay": replay,
        }
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
    for spec in SOURCE_SPECS:
        parser.add_argument(spec.option, required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = write_rc3_gate_e_input_snapshot_v1(
            args.gate_e_evidence_root,
            frozen_candidate_root=args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
            sources=GateEInputSnapshotSources(
                **{spec.attribute: getattr(args, spec.option.removeprefix("--").replace("-", "_")) for spec in SOURCE_SPECS}
            ),
        )
    except (OSError, GateEInputSnapshotError) as error:
        print(f"RC3 Gate E input snapshot failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
