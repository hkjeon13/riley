#!/usr/bin/env python3
"""Replay a closed RC3 Gate E input inventory through caller-held FDs.

This is deliberately an input-identity boundary, not a Gate E decision.  It
replays the frozen candidate's original input closure and rehashes the exact
four-gate evidence inventory (canonical E0 correctness, Python-free E2E,
performance regression, and soak).  It never accepts a legacy final-candidate
manifest, runs a path-based release checker, or treats a report's self-authored
``passed`` field as authority.

The result is only ``bound/frozen/not-run``.  Dedicated FD-native semantic
adapters must later consume this exact inventory before any Gate E pass or
qualification receipt can exist.
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


INVENTORY_VERSION = "riley.rc3-gate-e-input-inventory.v1"
REPLAY_VERSION = "riley.rc3-gate-e-input-inventory-replay.v1"
INVENTORY_NAME = "gate-e-inputs.json"
SCOPE = "gate-e-input-inventory-only"
AUTHORITY = "gate-e-input-inventory-replay-only"
MAX_INVENTORY_BYTES = 512 * 1024
MAX_GATE_E_INPUT_DESCRIPTORS = 14
MAX_GATE_E_INPUT_BYTES = 1 << 40
MAX_DIRECT_LEAF_NAME_LENGTH = 128

GATE_E_INPUT_GROUP_FIELDS = (
    (
        "release",
        frozenset({"bundle", "profile_binary", "native_candidate_executable"}),
    ),
    (
        "canonical_e0",
        frozenset(
            {
                "native_report",
                "native_raw_evidence",
                "optimizer_report",
                "optimizer_raw_evidence",
            }
        ),
    ),
    ("python_free", frozenset({"report", "raw_evidence", "correctness_golden"})),
    ("performance", frozenset({"report", "raw_evidence"})),
    ("soak", frozenset({"report", "raw_evidence"})),
)

CHECK_NAMES = (
    "closed-four-gate-inventory-rehashed",
    "gate-e-inventory-closure-within-fixed-byte-budget",
    "frozen-candidate-input-identity-replayed",
    "candidate-and-source-identity-match-frozen-candidate",
    "inventory-is-structural-only",
)

NOT_ESTABLISHED = {
    "gate_e_semantic_replay": "not-established",
    "gate_e_pass": "not-established",
    "qualification": "not-established",
    "evidence_root_immutability": "not-established",
}

T = TypeVar("T")


class GateEInventoryReplayError(ValueError):
    """The closed RC3 Gate E inventory cannot be safely replayed."""


@dataclass(frozen=True)
class GateEInventory:
    """One exact structural inventory of the four common Gate E inputs."""

    candidate_id: str
    source_revision: str
    frozen_candidate_manifest: common.EvidenceDescriptor
    artifacts: tuple[tuple[str, common.EvidenceDescriptor], ...]


def _fail(code: str, message: str) -> NoReturn:
    error = GateEInventoryReplayError(f"{code}: {message}")
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
            "invoke this replayer with python3 -B or PYTHONDONTWRITEBYTECODE=1",
        )


def _exact_object(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        _fail("invalid-gate-e-inventory", f"{label} has an unexpected field set")
    return value


def _direct_descriptor(value: Any, label: str) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if descriptor.byte_length < 1:
        _fail("invalid-gate-e-input-descriptor", f"{label} must have a positive byte_length")
    if "/" in descriptor.path or len(descriptor.path) > MAX_DIRECT_LEAF_NAME_LENGTH:
        _fail(
            "gate-e-input-must-be-direct-root-leaf",
            f"{label} must name one direct evidence-root leaf no longer than {MAX_DIRECT_LEAF_NAME_LENGTH} characters",
        )
    if descriptor.path == INVENTORY_NAME:
        _fail(
            "gate-e-request-descriptor-path-reused",
            f"{label} must not reuse the inventory request leaf",
        )
    return descriptor


def _parse_gate_e_inventory(document: Any) -> GateEInventory:
    row = _exact_object(
        document,
        {
            "schema_version",
            "candidate_id",
            "source_revision",
            "release",
            "canonical_e0",
            "python_free",
            "performance",
            "soak",
            "frozen_candidate_manifest",
        },
        "Gate E input inventory",
    )
    if row["schema_version"] != INVENTORY_VERSION:
        _fail(
            "unsupported-gate-e-inventory-version",
            "Gate E input inventory.schema_version is unsupported",
        )
    candidate_id = row["candidate_id"]
    source_revision = row["source_revision"]
    if type(candidate_id) is not str or frozen.CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        _fail("invalid-gate-e-candidate-id", "Gate E input inventory candidate_id is malformed")
    if (
        type(source_revision) is not str
        or frozen.REVISION_RE.fullmatch(source_revision) is None
        or source_revision == "0" * 40
    ):
        _fail("invalid-gate-e-source-revision", "Gate E input inventory source_revision is malformed")

    frozen_candidate_manifest = _common(
        lambda: common.parse_descriptor(
            row["frozen_candidate_manifest"],
            "Gate E input inventory.frozen_candidate_manifest",
        )
    )
    if frozen_candidate_manifest.path != frozen.MANIFEST_NAME:
        _fail(
            "invalid-frozen-candidate-manifest-descriptor",
            "Gate E input inventory must name the fixed frozen-candidate manifest leaf",
        )
    if frozen_candidate_manifest.byte_length < 1:
        _fail(
            "invalid-frozen-candidate-manifest-descriptor",
            "Gate E input inventory frozen-candidate descriptor must have a positive byte_length",
        )

    artifacts: list[tuple[str, common.EvidenceDescriptor]] = []
    for group_name, fields in GATE_E_INPUT_GROUP_FIELDS:
        group = _exact_object(row[group_name], fields, f"Gate E input inventory.{group_name}")
        for field_name in sorted(fields):
            artifacts.append(
                (
                    f"{group_name}.{field_name}",
                    _direct_descriptor(group[field_name], f"Gate E input inventory.{group_name}.{field_name}"),
                )
            )
    if len(artifacts) != MAX_GATE_E_INPUT_DESCRIPTORS:
        raise AssertionError("closed Gate E inventory descriptor count drifted")
    return GateEInventory(
        candidate_id=candidate_id,
        source_revision=source_revision,
        frozen_candidate_manifest=frozen_candidate_manifest,
        artifacts=tuple(artifacts),
    )


def _read_inventory(
    evidence_root_fd: int,
) -> tuple[bytes, common.EvidenceDescriptor, dict[str, Any], GateEInventory]:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            evidence_root_fd,
            INVENTORY_NAME,
            "Gate E input inventory",
            maximum_bytes=MAX_INVENTORY_BYTES,
        )
    )
    descriptor = _common(
        lambda: common.descriptor_for_bytes(
            INVENTORY_NAME,
            raw,
            "Gate E input inventory",
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            raw,
            "Gate E input inventory",
            maximum_bytes=MAX_INVENTORY_BYTES,
        )
    )
    assert isinstance(document, dict)
    return raw, descriptor, document, _parse_gate_e_inventory(document)


def _assert_exact_entries(
    evidence_root_fd: int,
    inventory: GateEInventory,
) -> None:
    expected = {INVENTORY_NAME, *(descriptor.path for _label, descriptor in inventory.artifacts)}
    actual: set[str] = set()
    try:
        with os.scandir(evidence_root_fd) as entries:
            for entry in entries:
                actual.add(entry.name)
                if entry.name not in expected or len(actual) > len(expected):
                    _fail(
                        "unexpected-evidence-entry",
                        "Gate E evidence root must contain exactly the closed inventory leaves; "
                        f"expected={sorted(expected)}, actual_prefix={sorted(actual)}",
                    )
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot list Gate E evidence root: {error}")
    if actual != expected:
        _fail(
            "unexpected-evidence-entry",
            "Gate E evidence root must contain exactly the closed inventory leaves; "
            f"expected={sorted(expected)}, actual={sorted(actual)}",
        )


def _bound_inventory_closure(
    inventory_descriptor: common.EvidenceDescriptor,
    inventory: GateEInventory,
) -> None:
    closure = _common(
        lambda: common.require_unique_descriptors(
            (inventory_descriptor, *(descriptor for _label, descriptor in inventory.artifacts)),
            "Gate E input inventory closure",
        )
    )
    if len(closure) != MAX_GATE_E_INPUT_DESCRIPTORS + 1:
        _fail("invalid-gate-e-inventory", "Gate E input inventory closure has an unexpected size")
    total_bytes = sum(descriptor.byte_length for descriptor in closure)
    if total_bytes > MAX_GATE_E_INPUT_BYTES:
        _fail(
            "external-evidence-byte-budget-exceeded",
            "Gate E input inventory exceeds its total external evidence byte budget",
        )


def _verify_artifacts(evidence_root_fd: int, inventory: GateEInventory) -> None:
    for label, descriptor in inventory.artifacts:
        _common(
            lambda label=label, descriptor=descriptor: common.verify_descriptor_file(
                evidence_root_fd,
                descriptor,
                f"Gate E input {label}",
            )
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


def _replay_rc3_gate_e_input_inventory_v1_on_held_fds(
    gate_e_evidence_root_fd: int,
    frozen_candidate_root_fd: int,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
) -> dict[str, Any]:
    """Rehash one closed inventory without mutating caller-owned descriptors.

    A same-stack caller retains the source, freeze-input, frozen-candidate, and
    Gate E evidence-root FDs and locks throughout this operation.  The core
    rejects physical ancestry between those held descriptors, but its public
    wrapper remains responsible for visible-path and mount-alias topology.
    It reads only the inventory and original raw leaves, calls the existing
    trusted read-only Git source oracle through the frozen-candidate replay,
    and neither creates an output nor interprets a gate report as a semantic
    pass.
    """

    _require_bytecode_cache_disabled()
    _common(
        lambda: common.require_private_evidence_directory_fd(
            gate_e_evidence_root_fd,
            "Gate E evidence root",
        )
    )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            frozen_candidate_root_fd,
            "frozen candidate root",
        )
    )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            input_evidence_root_fd,
            "freeze-input evidence root",
        )
    )
    _frozen(
        lambda: frozen.require_distinct_root_fds(
            {
                "Gate E evidence root": gate_e_evidence_root_fd,
                "frozen candidate root": frozen_candidate_root_fd,
                "freeze-input evidence root": input_evidence_root_fd,
                "source checkout": repository_root_fd,
            }
        )
    )
    _topology(
        lambda: topology.assert_held_root_fds_disjoint(
            {
                "Gate E evidence root": gate_e_evidence_root_fd,
                "frozen candidate root": frozen_candidate_root_fd,
                "freeze-input evidence root": input_evidence_root_fd,
                "source checkout": repository_root_fd,
            }
        )
    )
    inventory_raw, inventory_descriptor, inventory_document, inventory = _read_inventory(
        gate_e_evidence_root_fd
    )
    _bound_inventory_closure(inventory_descriptor, inventory)
    _assert_exact_entries(gate_e_evidence_root_fd, inventory)
    frozen_result = _frozen_replay(
        lambda: frozen_replayer._replay_rc3_frozen_candidate_v1_on_held_fds(
            frozen_candidate_root_fd,
            input_evidence_root_fd,
            repository_root,
            repository_root_fd,
        )
    )
    if (
        frozen_result.get("candidate_id") != inventory.candidate_id
        or frozen_result.get("source_revision") != inventory.source_revision
    ):
        _fail(
            "frozen-candidate-identity-mismatch",
            "Gate E input inventory candidate/source identity differs from the frozen candidate",
        )
    if frozen_result.get("frozen_candidate_manifest") != inventory.frozen_candidate_manifest.as_json():
        _fail(
            "frozen-candidate-manifest-descriptor-mismatch",
            "Gate E input inventory frozen-candidate descriptor differs from the held-FD replay",
        )
    _verify_artifacts(gate_e_evidence_root_fd, inventory)
    frozen_result_end = _frozen_replay(
        lambda: frozen_replayer._replay_rc3_frozen_candidate_v1_on_held_fds(
            frozen_candidate_root_fd,
            input_evidence_root_fd,
            repository_root,
            repository_root_fd,
        )
    )
    if common.canonical_json_bytes(frozen_result_end) != common.canonical_json_bytes(
        frozen_result
    ):
        _fail(
            "frozen-input-replay-drift",
            "frozen candidate replay changed while Gate E inputs were rehashed",
        )
    _verify_artifacts(gate_e_evidence_root_fd, inventory)
    inventory_raw_end, inventory_descriptor_end, inventory_document_end, inventory_end = _read_inventory(
        gate_e_evidence_root_fd
    )
    if (
        inventory_raw_end != inventory_raw
        or inventory_descriptor_end != inventory_descriptor
        or inventory_document_end != inventory_document
        or inventory_end != inventory
    ):
        _fail("raced-input", "Gate E input inventory changed during held-FD replay")
    _assert_exact_entries(gate_e_evidence_root_fd, inventory)
    frozen_manifest = frozen_result.get("frozen_candidate_manifest")
    if type(frozen_manifest) is not dict:
        _fail("invalid-frozen-candidate", "frozen candidate replay returned no manifest descriptor")
    return {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "candidate_id": inventory.candidate_id,
        "source_revision": inventory.source_revision,
        "gate_e_input_inventory": inventory_descriptor.as_json(),
        "frozen_candidate_manifest": frozen_manifest,
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "not_established": dict(NOT_ESTABLISHED),
        "reason_codes": [],
    }


def replay_rc3_gate_e_input_inventory_v1(
    gate_e_evidence_root: Path,
    *,
    frozen_candidate_root: Path,
    input_evidence_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Open four disjoint roots and replay one closed Gate E input inventory."""

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
        gate_root_fd = _common(
            lambda: common.open_private_evidence_directory(
                gate_root,
                "Gate E evidence root",
            )
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
        result = _replay_rc3_gate_e_input_inventory_v1_on_held_fds(
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
        report = replay_rc3_gate_e_input_inventory_v1(
            args.gate_e_evidence_root,
            frozen_candidate_root=args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
        )
    except GateEInventoryReplayError as error:
        print(f"RC3 Gate E input-inventory replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
