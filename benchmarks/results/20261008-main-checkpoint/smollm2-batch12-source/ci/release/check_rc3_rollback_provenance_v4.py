#!/usr/bin/env python3
"""Replay terminal-shaped rollback provenance v4 without operational authority.

v4 joins the nonterminal reconstructed-baseline v3 manifest to the fixed
artifact preparation, atomic exchange, and held-FD transaction closure.  It
is a raw structural replayer only: a visible transaction completion pair can
remain after a producer reported ``ambiguous-terminal-publication``.  Thus a
fresh v4 replay never proves a prior producer succeeded, a host rolled back,
or a candidate qualified.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import capture_rc3_rollback_atomic_transaction_v1 as transaction
import check_rc3_rollback_provenance_v3 as v3
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common


sys.dont_write_bytecode = True

ROLLBACK_V4_MANIFEST_VERSION = "riley.rc3-rollback-terminal-provenance.v4"
ROLLBACK_V4_COMPLETION_VERSION = "riley.rc3-rollback-terminal-provenance-complete.v4"
ROLLBACK_V4_REPORT_VERSION = "riley.rc3-rollback-terminal-provenance-check.v4"
FIXED_TRANSACTION_SESSION_PATH = f"{transaction.TRANSACTION_DIRECTORY_NAME}/session.json"
MAX_MANIFEST_BYTES = 1 << 20
_MANIFEST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,122}\.json$")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "capture_status",
        "qualification_status",
        "candidate_id",
        "bindings",
        "rollback_v3_manifest",
        "atomic_transaction_session",
    }
)
_COMPLETION_FIELDS = frozenset({"schema_version", "artifact_filename", "artifact_sha256"})


class RollbackV4ProvenanceError(ValueError):
    """The v4 raw provenance closure cannot be safely replayed."""


def _fail(code: str, message: str) -> NoReturn:
    error = RollbackV4ProvenanceError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _v3(call: Callable[[], T]) -> T:
    try:
        return call()
    except v3.RollbackV3ProvenanceError as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), f"v3 replay failed: {error}")


def _transaction(call: Callable[[], T]) -> T:
    try:
        return call()
    except transaction.RollbackAtomicTransactionError as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), f"transaction replay failed: {error}")


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_external_to_source(evidence_root: Path) -> None:
    try:
        evidence_root.relative_to(_source_root())
    except ValueError:
        return
    _fail("evidence-root-inside-source-checkout", "--evidence-root must be outside the source checkout")


def _lock(descriptor: int, mode: int, label: str) -> None:
    try:
        fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("lock-unavailable", f"cannot acquire {label} lock: {error}")


def _unlock_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _exact(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail("invalid-schema", f"{label} fields differ; expected={sorted(fields)}, actual={actual}")
    return value


def _manifest_name(value: Any, label: str) -> str:
    if type(value) is not str or _MANIFEST_NAME.fullmatch(value) is None:
        _fail("invalid-manifest-name", f"{label} must be one direct nonhidden .json root leaf")
    return value


def _root_descriptor(value: Any, label: str) -> common.EvidenceDescriptor:
    candidate = value.as_json() if isinstance(value, common.EvidenceDescriptor) else value
    descriptor = _common(lambda: common.parse_descriptor(candidate, label))
    name = _manifest_name(descriptor.path, f"{label}.path")
    if descriptor.path != name:
        _fail("invalid-descriptor", f"{label} must name one direct evidence-root leaf")
    if descriptor.byte_length > MAX_MANIFEST_BYTES:
        _fail("input-too-large", f"{label} exceeds the v4 manifest byte limit")
    return descriptor


def _fixed_transaction_descriptor(value: Any, label: str) -> common.EvidenceDescriptor:
    candidate = value.as_json() if isinstance(value, common.EvidenceDescriptor) else value
    descriptor = _common(lambda: common.parse_descriptor(candidate, label))
    if descriptor.path != FIXED_TRANSACTION_SESSION_PATH:
        _fail("invalid-transaction-session-path", f"{label} must be {FIXED_TRANSACTION_SESSION_PATH!r}")
    return descriptor


def _read_private_root_json(
    root_fd: int,
    name: str,
    label: str,
) -> tuple[common.EvidenceDescriptor, dict[str, Any]]:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            name,
            label,
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
    )
    descriptor = _common(lambda: common.descriptor_for_bytes(name, raw, label))
    _raw, document = _common(
        lambda: common.read_private_descriptor_json_leaf(
            root_fd,
            descriptor,
            label,
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
    )
    return descriptor, document


def _assert_v3_has_no_terminal_siblings(root_fd: int, name: str) -> None:
    for sibling in (f"{name}.intent", f"{name}.complete"):
        try:
            os.lstat(sibling, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail("unsafe-evidence-path", f"cannot inspect v3 terminal sibling {sibling!r}: {error}")
        _fail(
            "nonterminal-v3-required",
            "v3 provenance must not have terminal-looking sidecars before the v4 closure",
        )


def _descriptor_map(value: Any, fields: frozenset[str] | set[str], label: str) -> dict[str, common.EvidenceDescriptor]:
    expected = frozenset(fields)
    row = _exact(value, expected, label)
    return {name: _common(lambda item=row[name], key=name: common.parse_descriptor(item, f"{label}.{key}")) for name in expected}


def _replay_inputs_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    v3_manifest_name: str,
) -> tuple[common.EvidenceDescriptor, Mapping[str, Any], transaction.AtomicTransactionReplay]:
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    name = _manifest_name(v3_manifest_name, "rollback v3 manifest name")
    _assert_v3_has_no_terminal_siblings(root_fd, name)
    v3_descriptor, _document = _read_private_root_json(root_fd, name, "rollback v3 raw manifest")
    v3_report = _v3(lambda: v3.verify_rollback_provenance_v3_fd(root_fd, name))
    reported_v3_descriptor = _root_descriptor(
        v3_report.get("raw_manifest"),
        "rollback v3 report raw_manifest",
    )
    if reported_v3_descriptor != v3_descriptor:
        _fail(
            "v3-manifest-replay-drift",
            "v3 manifest changed between its held private read and v3 replay",
        )
    replay = _transaction(lambda: transaction.replay_atomic_transaction_on_held_switch_fd(root_fd, switch_fd))

    v3_evidence = _exact(
        v3_report.get("raw_evidence"),
        frozenset({"candidate", "rollback", "candidate_artifacts", "rollback_artifacts", "atomic_switch"}),
        "rollback v3 report.raw_evidence",
    )
    candidate_artifacts = _descriptor_map(
        v3_evidence["candidate_artifacts"],
        v3.ARTIFACT_FIELDS,
        "rollback v3 candidate artifact map",
    )
    rollback_artifacts = _descriptor_map(
        v3_evidence["rollback_artifacts"],
        v3.ARTIFACT_FIELDS,
        "rollback v3 rollback artifact map",
    )
    atomic_switch = _descriptor_map(
        v3_evidence["atomic_switch"],
        v3.ATOMIC_SWITCH_FIELDS,
        "rollback v3 atomic switch map",
    )
    preparation = _exact(
        replay.preparation_session,
        frozenset(
            {
                "schema_version",
                "capture_status",
                "qualification_status",
                "artifact_snapshots",
                "snapshot_identities",
                "runtime_materializations",
            }
        ),
        "transaction preparation session",
    )
    preparation_snapshots = _exact(
        preparation["artifact_snapshots"],
        frozenset({"candidate", "rollback"}),
        "transaction preparation artifact snapshots",
    )
    prepared_candidate = _descriptor_map(
        preparation_snapshots["candidate"],
        v3.ARTIFACT_FIELDS,
        "transaction candidate artifact snapshots",
    )
    prepared_rollback = _descriptor_map(
        preparation_snapshots["rollback"],
        v3.ARTIFACT_FIELDS,
        "transaction rollback artifact snapshots",
    )
    atomic_session = _exact(
        replay.atomic_switch_session,
        frozenset(
            {
                "schema_version",
                "capture_status",
                "qualification_status",
                "switch_directory",
                "atomic_switch",
            }
        ),
        "transaction atomic switch session",
    )
    prepared_atomic = _descriptor_map(
        atomic_session["atomic_switch"],
        v3.ATOMIC_SWITCH_FIELDS,
        "transaction atomic switch map",
    )
    if candidate_artifacts != prepared_candidate:
        _fail(
            "candidate-artifact-transaction-mismatch",
            "v3 candidate artifact descriptors do not exactly match the preparation session",
        )
    if rollback_artifacts != prepared_rollback:
        _fail(
            "rollback-artifact-transaction-mismatch",
            "v3 rollback artifact descriptors do not exactly match the preparation session",
        )
    if atomic_switch != prepared_atomic:
        _fail(
            "atomic-switch-transaction-mismatch",
            "v3 atomic-switch descriptors do not exactly match the transaction child session",
        )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    return v3_descriptor, v3_report, replay


def _report(
    manifest_descriptor: common.EvidenceDescriptor,
    v3_descriptor: common.EvidenceDescriptor,
    v3_report: Mapping[str, Any],
    replay: transaction.AtomicTransactionReplay,
) -> dict[str, Any]:
    candidate_id = v3_report.get("candidate_id")
    bindings = v3_report.get("bindings")
    if type(candidate_id) is not str or not isinstance(bindings, Mapping):
        _fail("invalid-v3-report", "v3 replay report lacks candidate identity bindings")
    return {
        "schema_version": ROLLBACK_V4_REPORT_VERSION,
        "status": "bound",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "bindings": dict(bindings),
        "raw_manifest": manifest_descriptor.as_json(),
        "rollback_v3_manifest": v3_descriptor.as_json(),
        "atomic_transaction_session": replay.session_descriptor.as_json(),
        "checks": [
            {"name": "v4-version-only", "bound": True},
            {"name": "v3-nonterminal-held-fd-replay", "bound": True},
            {"name": "transaction-paired-terminal-replay", "bound": True},
            {"name": "v3-artifact-preparation-exact-join", "bound": True},
            {"name": "v3-atomic-switch-transaction-exact-join", "bound": True},
            {"name": "held-switch-pre-post-runtime-transaction-join", "bound": True},
        ],
        "reason_codes": [],
    }


def verify_rollback_provenance_v4_bytes_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    manifest_descriptor: common.EvidenceDescriptor | Mapping[str, Any],
    raw_document: bytes,
) -> dict[str, Any]:
    """Replay exact canonical v4 bytes through the caller-held root/switch FDs."""

    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    descriptor = _root_descriptor(manifest_descriptor, "rollback v4 manifest descriptor")
    document = _common(
        lambda: common.parse_canonical_json(
            raw_document,
            "rollback v4 manifest",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
    )
    if not isinstance(document, Mapping):
        _fail("invalid-schema", "rollback v4 manifest must be an object")
    document_descriptor = _common(
        lambda: common.descriptor_for_bytes(descriptor.path, raw_document, "rollback v4 manifest")
    )
    if document_descriptor != descriptor:
        _fail("manifest-document-descriptor-mismatch", "v4 descriptor does not bind the supplied document")
    row = _exact(document, _MANIFEST_FIELDS, "rollback v4 raw manifest")
    if (
        row["schema_version"] != ROLLBACK_V4_MANIFEST_VERSION
        or row["capture_status"] != "captured"
        or row["qualification_status"] != "not-run"
    ):
        _fail("invalid-capture-status", "rollback v4 manifest must be exact captured/not-run raw evidence")
    v3_descriptor = _root_descriptor(row["rollback_v3_manifest"], "rollback v4 rollback_v3_manifest")
    transaction_descriptor = _fixed_transaction_descriptor(
        row["atomic_transaction_session"],
        "rollback v4 atomic_transaction_session",
    )
    actual_v3_descriptor, v3_report, replay = _replay_inputs_on_held_switch_fd(
        root_fd,
        switch_fd,
        v3_descriptor.path,
    )
    if actual_v3_descriptor != v3_descriptor:
        _fail("rollback-v3-descriptor-mismatch", "v4 does not bind the held v3 manifest bytes")
    if replay.session_descriptor != transaction_descriptor:
        _fail(
            "transaction-session-descriptor-mismatch",
            "v4 does not bind the held fixed transaction session bytes",
        )
    if row["candidate_id"] != v3_report.get("candidate_id"):
        _fail("candidate-id-mismatch", "v4 candidate_id does not derive from v3")
    if row["bindings"] != v3_report.get("bindings"):
        _fail("bindings-mismatch", "v4 bindings do not derive exactly from v3")
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    return _report(descriptor, actual_v3_descriptor, v3_report, replay)


def verify_rollback_provenance_v4_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    manifest_name: str,
) -> dict[str, Any]:
    """Replay one v4 raw manifest through caller-held root and switch FDs."""

    name = _manifest_name(manifest_name, "rollback v4 manifest name")
    descriptor, document = _read_private_root_json(root_fd, name, "rollback v4 raw manifest")
    report = verify_rollback_provenance_v4_bytes_on_held_switch_fd(
        root_fd,
        switch_fd,
        descriptor,
        _common(lambda: common.canonical_json_bytes(document)),
    )
    terminal_descriptor, terminal_document = _read_private_root_json(
        root_fd,
        name,
        "rollback v4 raw manifest",
    )
    if terminal_descriptor != descriptor or terminal_document != document:
        _fail("raced-input", "rollback v4 manifest changed during held-FD replay")
    return report


def verify_completed_rollback_provenance_v4_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    manifest_name: str,
) -> dict[str, Any]:
    """Structurally replay a v4 manifest and its paired completion marker.

    Completion remains a raw publication fact only.  It must not be used as a
    producer-success substitute after a previous ambiguous final-link fsync.
    """

    name = _manifest_name(manifest_name, "rollback v4 manifest name")
    report = verify_rollback_provenance_v4_on_held_switch_fd(root_fd, switch_fd, name)
    raw_marker = _common(
        lambda: common.read_bounded_paired_hardlink(
            root_fd,
            f"{name}.complete",
            f"{name}.intent",
            "rollback v4 completion marker",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
    )
    marker = _common(
        lambda: common.parse_canonical_json(
            raw_marker,
            "rollback v4 completion marker",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
    )
    row = _exact(marker, _COMPLETION_FIELDS, "rollback v4 completion marker")
    if row["schema_version"] != ROLLBACK_V4_COMPLETION_VERSION:
        _fail("unsupported-completion-version", "rollback v4 completion marker version is unsupported")
    if row["artifact_filename"] != name:
        _fail("completion-artifact-mismatch", "rollback v4 completion marker names another manifest")
    raw_manifest = _root_descriptor(report["raw_manifest"], "rollback v4 report raw_manifest")
    if row["artifact_sha256"] != raw_manifest.sha256:
        _fail("completion-artifact-mismatch", "rollback v4 completion marker does not bind manifest bytes")
    replayed = verify_rollback_provenance_v4_on_held_switch_fd(root_fd, switch_fd, name)
    if replayed != report:
        _fail("post-completion-replay-drift", "v4 raw replay changed while its completion marker was checked")
    return report


def verify_rollback_provenance_v4(
    evidence_root: Path,
    manifest_name: str,
    *,
    require_completion: bool = False,
) -> dict[str, Any]:
    """Open a private evidence root for read-only structural v4 replay."""

    _assert_external_to_source(evidence_root)
    root_fd = _common(lambda: common.open_private_evidence_directory(evidence_root, "--evidence-root"))
    switch_fd: int | None = None
    try:
        _lock(root_fd, fcntl.LOCK_SH, "shared evidence-root")
        switch_fd = _common(
            lambda: common.open_private_child_directory(
                root_fd,
                prepare.SWITCH_DIRECTORY_NAME,
                "isolated rollback switch directory",
            )
        )
        _lock(switch_fd, fcntl.LOCK_SH, "shared rollback switch")
        verifier = (
            verify_completed_rollback_provenance_v4_on_held_switch_fd
            if require_completion
            else verify_rollback_provenance_v4_on_held_switch_fd
        )
        return verifier(root_fd, switch_fd, manifest_name)
    finally:
        _unlock_quietly(switch_fd)
        _close_quietly(switch_fd)
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--require-completion", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_rollback_provenance_v4(
            args.evidence_root,
            args.manifest_name,
            require_completion=args.require_completion,
        )
    except RollbackV4ProvenanceError as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
