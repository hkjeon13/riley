#!/usr/bin/env python3
"""Capture one fixed-layout rollback artifact exchange as a held-FD transaction.

This is deliberately a raw composition layer, not an operational rollback
runner.  With one exclusive evidence-root lock and one exclusive held switch
directory lock, it replays the prepared artifact layout, performs the already
isolated ``renameat2(RENAME_EXCHANGE)`` capture, replays its terminal inode
record, and replays the prepared layout under the exchanged names.  It writes
only fixed create-only evidence children below an existing private evidence
root and keeps every result ``captured/not-run``.

It does not launch or stop Riley, select a process, open a network connection,
run a container command, use a GPU, alter a deployment path, or decide that a
rollback/qualification succeeded.  The fixed switch directory is evidence
workspace only.  A later authenticated runner may consume this transaction's
terminal raw evidence after the remaining phase/source/config bindings exist.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import capture_rc3_rollback_atomic_switch_v1 as atomic
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common


sys.dont_write_bytecode = True

SESSION_VERSION = "riley.rc3-rollback-atomic-transaction.v1"
INCOMPLETE_MARKER_VERSION = "riley.rc3-rollback-atomic-transaction-incomplete.v1"
INCOMPLETE_MARKER_NAME = "capture-incomplete.json"
COMPLETE_MARKER_VERSION = "riley.rc3-rollback-atomic-transaction-complete.v1"
COMPLETE_INTENT_NAME = "capture-complete.intent"
COMPLETE_MARKER_NAME = "capture-complete.json"
TRANSACTION_DIRECTORY_NAME = "rollback-v3-atomic-transaction"
ATOMIC_CAPTURE_DIRECTORY_NAME = "rollback-v3-atomic-switch"
SESSION_FIELDS = frozenset(
    {
        "schema_version",
        "capture_status",
        "qualification_status",
        "preparation_session",
        "atomic_switch_session",
        "preparation_pre_switch",
        "preparation_post_switch",
    }
)
RUNTIME_FIELDS = frozenset(
    {
        "switch_directory",
        "entry_name",
        "immutable_binary_sha256",
        "immutable_binary_byte_length",
        "sha256",
        "byte_length",
        "device",
        "inode",
        "mode",
        "nlink",
    }
)
RUNTIME_REPLAY_FIELDS = frozenset(
    {
        "sha256",
        "byte_length",
        "device",
        "inode",
        "mode",
        "nlink",
        "mtime_ns",
        "ctime_ns",
    }
)
PREPARATION_LAYOUT_FIELDS = frozenset({"candidate", "rollback"})


class RollbackAtomicTransactionError(ValueError):
    """The raw held-FD rollback transaction cannot safely become terminal."""


@dataclass(frozen=True)
class AtomicTransactionReplay:
    """Structurally replayed fixed child sessions through held FDs.

    This is deliberately raw evidence only.  It establishes no producer
    success edge: a fresh replay may legitimately observe a completion pair
    left visible by a prior ``ambiguous-terminal-publication`` failure.
    """

    session: Mapping[str, Any]
    session_descriptor: common.EvidenceDescriptor
    preparation_session: Mapping[str, Any]
    preparation_descriptor: common.EvidenceDescriptor
    atomic_switch_session: Mapping[str, Any]
    atomic_switch_descriptor: common.EvidenceDescriptor
    atomic_switch_replay: atomic.AtomicSwitchReplay


def _fail(message: str, *, code: str = "unsafe-evidence") -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = RollbackAtomicTransactionError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(str(error), code=getattr(error, "reason_code", "unsafe-evidence"))


def _prepare(call: Callable[[], T]) -> T:
    try:
        return call()
    except prepare.RollbackArtifactPreparationError as error:
        _fail(
            f"artifact preparation replay failed: {error}",
            code=getattr(error, "reason_code", "unsafe-evidence"),
        )


def _atomic(call: Callable[[], T]) -> T:
    try:
        return call()
    except atomic.AtomicSwitchCaptureError as error:
        _fail(
            f"atomic switch replay failed: {error}",
            code=getattr(error, "reason_code", "unsafe-evidence"),
        )


def _source_root() -> Path:
    """Factored so CPU-only tests can place evidence outside a fake checkout."""

    return Path(__file__).resolve().parents[2]


def _assert_external_to_source(evidence_root: Path) -> None:
    try:
        evidence_root.relative_to(_source_root())
    except ValueError:
        return
    _fail("--evidence-root must be outside the source checkout")


def _lock(descriptor: int, mode: int, label: str) -> None:
    try:
        fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(f"cannot acquire {label} lock: {error}")


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


def _require_held_switch_fd(root_fd: int, switch_fd: int) -> None:
    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "rollback transaction evidence root"))
    _common(
        lambda: common.require_private_child_directory_fd(
            root_fd,
            switch_fd,
            prepare.SWITCH_DIRECTORY_NAME,
            "held rollback transaction switch directory",
        )
    )


def _assert_absent(root_fd: int, name: str, label: str) -> None:
    try:
        os.lstat(name, dir_fd=root_fd)
    except FileNotFoundError:
        return
    except OSError as error:
        _fail(f"cannot inspect reserved {label}: {error}")
    _fail(f"reserved {label} already exists")


def _check_incomplete_marker(transaction_fd: int, *, require_terminal: bool) -> None:
    try:
        os.lstat(INCOMPLETE_MARKER_NAME, dir_fd=transaction_fd)
    except FileNotFoundError:
        if require_terminal:
            return
        _fail("preterminal atomic transaction must retain its incomplete marker")
    except OSError as error:
        _fail(f"cannot inspect atomic transaction incomplete marker: {error}")
    if require_terminal:
        _fail("atomic transaction incomplete marker is still present")
    document = _common(
        lambda: common.read_private_canonical_json_leaf(
            transaction_fd,
            INCOMPLETE_MARKER_NAME,
            "atomic transaction incomplete marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    if set(document) != {"schema_version", "capture_status", "qualification_status"} or (
        document["schema_version"] != INCOMPLETE_MARKER_VERSION
        or document["capture_status"] != "incomplete"
        or document["qualification_status"] != "not-run"
    ):
        _fail("atomic transaction incomplete marker is not the exact v1 marker")


def _check_completion_marker(
    transaction_fd: int,
    session: common.EvidenceDescriptor,
    *,
    require_terminal: bool,
) -> None:
    """Require a paired terminal receipt bound to the transaction session."""

    if not require_terminal:
        for name in (COMPLETE_INTENT_NAME, COMPLETE_MARKER_NAME):
            try:
                os.lstat(name, dir_fd=transaction_fd)
            except FileNotFoundError:
                continue
            except OSError as error:
                _fail(f"cannot inspect atomic transaction completion marker: {error}")
            _fail("preterminal atomic transaction has a completion marker")
        return
    raw = _common(
        lambda: common.read_bounded_paired_hardlink(
            transaction_fd,
            COMPLETE_MARKER_NAME,
            COMPLETE_INTENT_NAME,
            "atomic transaction completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            raw,
            "atomic transaction completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    if not isinstance(document, Mapping) or set(document) != {
        "schema_version",
        "capture_status",
        "qualification_status",
        "session_sha256",
        "session_byte_length",
    } or (
        document["schema_version"] != COMPLETE_MARKER_VERSION
        or document["capture_status"] != "captured"
        or document["qualification_status"] != "not-run"
        or document["session_sha256"] != session.sha256
        or document["session_byte_length"] != session.byte_length
    ):
        _fail("atomic transaction completion marker does not bind session.json")


def _restore_marker(transaction_fd: int, marker: Mapping[str, Any]) -> None:
    try:
        _common(
            lambda: common.write_create_only_json(
                transaction_fd,
                INCOMPLETE_MARKER_NAME,
                dict(marker),
                "restored atomic transaction incomplete marker",
            )
        )
    except RollbackAtomicTransactionError:
        pass


def _publish_completion_marker(
    transaction_fd: int,
    marker: Mapping[str, Any],
    session: common.CreatedEvidence,
) -> None:
    """Publish the terminal receipt only after durable marker removal.

    A missing incomplete marker is never a terminal state by itself. If the
    directory fsync or best-effort restoration fails, no completion pair is
    present and terminal verification fails closed. A post-link directory-sync
    error is ``ambiguous-terminal-publication``: a visible pair remains raw
    structural evidence, never this invocation's producer-success result.
    """

    try:
        os.unlink(INCOMPLETE_MARKER_NAME, dir_fd=transaction_fd)
    except OSError as error:
        _fail(f"cannot remove atomic transaction incomplete marker: {error}")
    try:
        os.fsync(transaction_fd)
    except OSError as error:
        _restore_marker(transaction_fd, marker)
        _fail(f"cannot synchronize atomic transaction marker removal: {error}")
    completion = {
        "schema_version": COMPLETE_MARKER_VERSION,
        "capture_status": "captured",
        "qualification_status": "not-run",
        "session_sha256": session.sha256,
        "session_byte_length": session.byte_length,
    }
    _common(
        lambda: common.write_create_only_json(
            transaction_fd,
            COMPLETE_INTENT_NAME,
            completion,
            "atomic transaction completion intent",
        )
    )
    _common(
        lambda: common.publish_create_only_hardlink(
            transaction_fd,
            COMPLETE_INTENT_NAME,
            COMPLETE_MARKER_NAME,
            "atomic transaction completion marker",
        )
    )


def _exact(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(f"{label} fields differ; expected={sorted(fields)}, actual={actual}")
    return value


def _session_descriptor(value: Any, expected_path: str, label: str) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if descriptor.path != expected_path or descriptor.byte_length < 1:
        _fail(f"{label} must bind the fixed nonempty leaf {expected_path!r}")
    return descriptor


def _transaction_session_leaf_descriptor(transaction_fd: int) -> common.EvidenceDescriptor:
    return _common(
        lambda: common.describe_regular_relative(
            transaction_fd,
            "session.json",
            "atomic transaction session",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )


def _held_session_descriptor(
    root_fd: int,
    child_fd: int,
    child_name: str,
    label: str,
) -> tuple[common.EvidenceDescriptor, Mapping[str, Any]]:
    """Describe one private ``session.json`` through a pre-held child FD."""

    _common(
        lambda: common.require_private_child_directory_fd(
            root_fd,
            child_fd,
            child_name,
            f"held {label} directory",
        )
    )
    document = _common(
        lambda: common.read_private_canonical_json_leaf(
            child_fd,
            "session.json",
            f"{label} session",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    leaf = _common(
        lambda: common.describe_regular_relative(
            child_fd,
            "session.json",
            f"{label} session",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    _common(
        lambda: common.require_private_child_directory_fd(
            root_fd,
            child_fd,
            child_name,
            f"held {label} directory",
        )
    )
    return (
        common.EvidenceDescriptor(
            path=f"{child_name}/session.json",
            sha256=leaf.sha256,
            byte_length=leaf.byte_length,
        ),
        document,
    )


def _open_terminal_child(root_fd: int, name: str, label: str) -> int:
    return _common(lambda: common.open_private_child_directory(root_fd, name, label))


def _read_terminal_session_descriptor(
    root_fd: int,
    child_fd: int,
    child_name: str,
    descriptor: common.EvidenceDescriptor,
    label: str,
) -> Mapping[str, Any]:
    held = _common(
        lambda: common.rebase_descriptor_to_held_leaf(
            descriptor,
            expected_root_relative_path=f"{child_name}/session.json",
            leaf_name="session.json",
            label=label,
        )
    )
    _raw, document = _common(
        lambda: common.read_private_descriptor_json_leaf(
            child_fd,
            held,
            label,
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    _common(
        lambda: common.require_private_child_directory_fd(
            root_fd,
            child_fd,
            child_name,
            f"held {label} directory",
        )
    )
    return document


def _runtime_row(session: Mapping[str, Any], arm: str, expected_entry_name: str) -> Mapping[str, Any]:
    materializations = session.get("runtime_materializations")
    if not isinstance(materializations, Mapping) or set(materializations) != {"candidate", "rollback"}:
        _fail("preparation session has no exact runtime materializations")
    row = _exact(materializations.get(arm), RUNTIME_FIELDS, f"preparation {arm} runtime mapping")
    integer_fields = {
        "immutable_binary_byte_length",
        "byte_length",
        "device",
        "inode",
        "mode",
        "nlink",
    }
    if (
        row["switch_directory"] != prepare.SWITCH_DIRECTORY_NAME
        or row["entry_name"] != expected_entry_name
        or type(row["immutable_binary_sha256"]) is not str
        or type(row["sha256"]) is not str
        or row["immutable_binary_sha256"] != row["sha256"]
        or row["immutable_binary_byte_length"] != row["byte_length"]
        or any(type(row[field]) is not int or row[field] < 0 for field in integer_fields)
        or row["mode"] != 0o700
        or row["nlink"] != 1
    ):
        _fail(f"preparation {arm} runtime mapping is not an exact private staged identity")
    return row


def _same_runtime(staged: atomic.StagedIdentity, row: prepare.RuntimeReplay) -> bool:
    return (
        staged.sha256 == row.sha256
        and staged.device == row.device
        and staged.inode == row.inode
        and staged.mode == row.mode
        and staged.nlink == row.nlink
        and staged.byte_length == row.byte_length
        and staged.mtime_ns == row.mtime_ns
        and staged.ctime_ns == row.ctime_ns
    )


def _runtime_replay(value: Any, label: str) -> prepare.RuntimeReplay:
    row = _exact(value, RUNTIME_REPLAY_FIELDS, label)
    if (
        type(row["sha256"]) is not str
        or len(row["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in row["sha256"])
        or any(type(row[field]) is not int or row[field] < 0 for field in RUNTIME_REPLAY_FIELDS - {"sha256"})
        or row["mode"] != 0o700
        or row["nlink"] != 1
    ):
        _fail(f"{label} is not an exact private runtime replay")
    return prepare.RuntimeReplay(
        sha256=row["sha256"],
        byte_length=row["byte_length"],
        device=row["device"],
        inode=row["inode"],
        mode=row["mode"],
        nlink=row["nlink"],
        mtime_ns=row["mtime_ns"],
        ctime_ns=row["ctime_ns"],
    )


def _preparation_layout(value: Any, label: str) -> tuple[prepare.RuntimeReplay, prepare.RuntimeReplay]:
    row = _exact(value, PREPARATION_LAYOUT_FIELDS, label)
    return (
        _runtime_replay(row["candidate"], f"{label}.candidate"),
        _runtime_replay(row["rollback"], f"{label}.rollback"),
    )


def _runtime_matches_static(runtime: prepare.RuntimeReplay, static: Mapping[str, Any]) -> bool:
    return (
        runtime.sha256 == static["sha256"]
        and runtime.byte_length == static["byte_length"]
        and runtime.device == static["device"]
        and runtime.inode == static["inode"]
        and runtime.mode == static["mode"]
        and runtime.nlink == static["nlink"]
    )


def _cross_bind_preparation_and_exchange(
    preparation_session: Mapping[str, Any],
    pre_preparation: tuple[prepare.RuntimeReplay, prepare.RuntimeReplay],
    replay: atomic.AtomicSwitchReplay,
    post_preparation: tuple[prepare.RuntimeReplay, prepare.RuntimeReplay],
) -> None:
    """Join the raw preparation runtime map to both sides of the exchange."""

    candidate = _runtime_row(preparation_session, "candidate", prepare.ACTIVE_NAME)
    rollback = _runtime_row(preparation_session, "rollback", prepare.ROLLBACK_STAGED_NAME)
    pre_candidate, pre_rollback = pre_preparation
    post_candidate, post_rollback = post_preparation
    if not _runtime_matches_static(pre_candidate, candidate):
        _fail("pre-switch candidate runtime does not bind the prepared immutable binary")
    if not _runtime_matches_static(pre_rollback, rollback):
        _fail("pre-switch rollback runtime does not bind the prepared immutable binary")
    if not _runtime_matches_static(post_candidate, candidate):
        _fail("post-switch candidate runtime does not bind the prepared immutable binary")
    if not _runtime_matches_static(post_rollback, rollback):
        _fail("post-switch rollback runtime does not bind the prepared immutable binary")
    if not _same_runtime(replay.pre_active, pre_candidate):
        _fail("atomic pre-active inode does not bind the prepared candidate runtime")
    if not _same_runtime(replay.pre_rollback_staged, pre_rollback):
        _fail("atomic pre-rollback inode does not bind the prepared rollback runtime")
    if not _same_runtime(replay.post_active, post_rollback):
        _fail("atomic post-active inode does not bind the prepared rollback runtime")
    if not _same_runtime(replay.post_candidate_staged, post_candidate):
        _fail("atomic post-staged inode does not bind the prepared candidate runtime")


def _replay_linked_sessions(
    root_fd: int,
    switch_fd: int,
    session: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    common.EvidenceDescriptor,
    Mapping[str, Any],
    common.EvidenceDescriptor,
    atomic.AtomicSwitchReplay,
]:
    """Replay fixed child sessions and cross-bind their immutable identities."""

    row = _exact(session, SESSION_FIELDS, "atomic transaction session")
    if (
        row["schema_version"] != SESSION_VERSION
        or row["capture_status"] != "captured"
        or row["qualification_status"] != "not-run"
    ):
        _fail("atomic transaction session is not an exact captured/not-run session")
    preparation_descriptor = _session_descriptor(
        row["preparation_session"],
        f"{prepare.SNAPSHOT_DIRECTORY_NAME}/session.json",
        "atomic transaction preparation session descriptor",
    )
    atomic_descriptor = _session_descriptor(
        row["atomic_switch_session"],
        f"{ATOMIC_CAPTURE_DIRECTORY_NAME}/session.json",
        "atomic transaction atomic-switch session descriptor",
    )
    _common(
        lambda: common.require_unique_descriptors(
            (preparation_descriptor, atomic_descriptor),
            "atomic transaction session descriptors",
        )
    )
    pre_layout = _preparation_layout(
        row["preparation_pre_switch"],
        "atomic transaction pre-switch preparation replay",
    )
    post_layout = _preparation_layout(
        row["preparation_post_switch"],
        "atomic transaction post-switch preparation replay",
    )
    preparation_fd: int | None = None
    atomic_fd: int | None = None
    try:
        preparation_fd = _open_terminal_child(
            root_fd,
            prepare.SNAPSHOT_DIRECTORY_NAME,
            "artifact preparation session directory",
        )
        atomic_fd = _open_terminal_child(
            root_fd,
            ATOMIC_CAPTURE_DIRECTORY_NAME,
            "atomic switch capture directory",
        )
        preparation_document = _read_terminal_session_descriptor(
            root_fd,
            preparation_fd,
            prepare.SNAPSHOT_DIRECTORY_NAME,
            preparation_descriptor,
            "atomic transaction preparation session descriptor",
        )
        atomic_document = _read_terminal_session_descriptor(
            root_fd,
            atomic_fd,
            ATOMIC_CAPTURE_DIRECTORY_NAME,
            atomic_descriptor,
            "atomic transaction atomic-switch session descriptor",
        )
        post_preparation = _prepare(
            lambda: prepare.replay_artifact_preparation_on_held_switch_fd(
                root_fd,
                switch_fd,
                require_terminal=True,
                runtime_layout="post-switch",
            )
        )
        exchange = _atomic(
            lambda: atomic.replay_atomic_switch_capture_on_held_switch_fd(
                root_fd,
                switch_fd,
                prepare.SWITCH_DIRECTORY_NAME,
                ATOMIC_CAPTURE_DIRECTORY_NAME,
            )
        )
        if dict(post_preparation.session) != dict(preparation_document):
            _fail("post-switch preparation replay disagrees with its transaction-bound session")
        if dict(exchange.session) != dict(atomic_document):
            _fail("atomic switch replay disagrees with its transaction-bound session")
        _cross_bind_preparation_and_exchange(
            preparation_document,
            pre_layout,
            exchange,
            post_layout,
        )
        if (
            post_preparation.candidate_runtime != post_layout[0]
            or post_preparation.rollback_runtime != post_layout[1]
        ):
            _fail("post-switch held runtime replay disagrees with its transaction record")
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                preparation_fd,
                prepare.SNAPSHOT_DIRECTORY_NAME,
                "held artifact preparation session directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                atomic_fd,
                ATOMIC_CAPTURE_DIRECTORY_NAME,
                "held atomic switch capture directory",
            )
        )
        return (
            preparation_document,
            preparation_descriptor,
            atomic_document,
            atomic_descriptor,
            exchange,
        )
    finally:
        _close_quietly(atomic_fd)
        _close_quietly(preparation_fd)


def replay_atomic_transaction_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    *,
    require_terminal: bool = True,
) -> AtomicTransactionReplay:
    """Replay fixed transaction evidence while the caller retains its switch FD.

    The verifier only establishes structurally replayable raw artifact/inode
    exchange evidence. It does not report a service, deployment, qualification,
    or producer-success outcome after an earlier
    ``ambiguous-terminal-publication``.
    """

    _require_held_switch_fd(root_fd, switch_fd)
    transaction_fd = _open_terminal_child(
        root_fd,
        TRANSACTION_DIRECTORY_NAME,
        "atomic transaction capture directory",
    )
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                transaction_fd,
                TRANSACTION_DIRECTORY_NAME,
                "held atomic transaction capture directory",
            )
        )
        _check_incomplete_marker(transaction_fd, require_terminal=require_terminal)
        initial_session_descriptor = _transaction_session_leaf_descriptor(transaction_fd)
        _raw, session = _common(
            lambda: common.read_private_descriptor_json_leaf(
                transaction_fd,
                initial_session_descriptor,
                "atomic transaction session",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        _check_completion_marker(
            transaction_fd,
            initial_session_descriptor,
            require_terminal=require_terminal,
        )
        (
            preparation_session,
            preparation_descriptor,
            atomic_switch_session,
            atomic_switch_descriptor,
            atomic_switch_replay,
        ) = _replay_linked_sessions(root_fd, switch_fd, session)
        _check_incomplete_marker(transaction_fd, require_terminal=require_terminal)
        _raw, terminal_session = _common(
            lambda: common.read_private_descriptor_json_leaf(
                transaction_fd,
                initial_session_descriptor,
                "atomic transaction session",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        if terminal_session != session:
            _fail("atomic transaction session changed during replay")
        _check_completion_marker(
            transaction_fd,
            initial_session_descriptor,
            require_terminal=require_terminal,
        )
        _require_held_switch_fd(root_fd, switch_fd)
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                transaction_fd,
                TRANSACTION_DIRECTORY_NAME,
                "held atomic transaction capture directory",
            )
        )
        return AtomicTransactionReplay(
            session=dict(session),
            session_descriptor=common.EvidenceDescriptor(
                path=f"{TRANSACTION_DIRECTORY_NAME}/session.json",
                sha256=initial_session_descriptor.sha256,
                byte_length=initial_session_descriptor.byte_length,
            ),
            preparation_session=dict(preparation_session),
            preparation_descriptor=preparation_descriptor,
            atomic_switch_session=dict(atomic_switch_session),
            atomic_switch_descriptor=atomic_switch_descriptor,
            atomic_switch_replay=atomic_switch_replay,
        )
    finally:
        _close_quietly(transaction_fd)


def verify_atomic_transaction_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    *,
    require_terminal: bool = True,
) -> dict[str, Any]:
    """Compatibility wrapper returning only the replayed raw session."""

    return dict(
        replay_atomic_transaction_on_held_switch_fd(
            root_fd,
            switch_fd,
            require_terminal=require_terminal,
        ).session
    )


def verify_atomic_transaction(evidence_root: Path) -> dict[str, Any]:
    """Standalone shared-lock replay for the fixed transaction evidence."""

    _assert_external_to_source(evidence_root)
    root_fd = _common(lambda: common.open_private_evidence_directory(evidence_root, "--evidence-root"))
    switch_fd: int | None = None
    try:
        _lock(root_fd, fcntl.LOCK_SH, "shared evidence-root")
        switch_fd = _open_terminal_child(
            root_fd,
            prepare.SWITCH_DIRECTORY_NAME,
            "isolated rollback switch directory",
        )
        _lock(switch_fd, fcntl.LOCK_SH, "shared rollback switch")
        return verify_atomic_transaction_on_held_switch_fd(root_fd, switch_fd)
    finally:
        _unlock_quietly(switch_fd)
        _close_quietly(switch_fd)
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)


def _capture_atomic_transaction_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
) -> AtomicTransactionReplay:
    """Capture one transaction through caller-held exclusive root/switch FDs.

    The caller must retain one exclusive lock on both descriptors for this
    entire call. This private primitive returns only after a normal producer
    return; its private continuation helper invokes the downstream callback
    before this call stack can be resumed from a fresh replay.
    """

    transaction_fd: int | None = None
    preparation_fd: int | None = None
    atomic_fd: int | None = None
    try:
        _require_held_switch_fd(root_fd, switch_fd)
        _assert_absent(root_fd, TRANSACTION_DIRECTORY_NAME, "atomic transaction capture directory")
        _assert_absent(root_fd, ATOMIC_CAPTURE_DIRECTORY_NAME, "atomic switch capture directory")
        preparation_fd = _open_terminal_child(
            root_fd,
            prepare.SNAPSHOT_DIRECTORY_NAME,
            "artifact preparation session directory",
        )
        transaction_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                TRANSACTION_DIRECTORY_NAME,
                "atomic transaction capture directory",
            )
        )
        marker = {
            "schema_version": INCOMPLETE_MARKER_VERSION,
            "capture_status": "incomplete",
            "qualification_status": "not-run",
        }
        _common(
            lambda: common.write_create_only_json(
                transaction_fd,
                INCOMPLETE_MARKER_NAME,
                marker,
                "atomic transaction incomplete marker",
            )
        )
        pre_preparation = _prepare(
            lambda: prepare.replay_artifact_preparation_on_held_switch_fd(
                root_fd,
                switch_fd,
                require_terminal=True,
                runtime_layout="pre-switch",
            )
        )
        _atomic(
            lambda: atomic.capture_atomic_switch_on_held_switch_fd(
                root_fd,
                switch_fd,
                prepare.SWITCH_DIRECTORY_NAME,
                ATOMIC_CAPTURE_DIRECTORY_NAME,
            )
        )
        atomic_fd = _open_terminal_child(
            root_fd,
            ATOMIC_CAPTURE_DIRECTORY_NAME,
            "atomic switch capture directory",
        )
        exchange = _atomic(
            lambda: atomic.replay_atomic_switch_capture_on_held_switch_fd(
                root_fd,
                switch_fd,
                prepare.SWITCH_DIRECTORY_NAME,
                ATOMIC_CAPTURE_DIRECTORY_NAME,
            )
        )
        post_preparation = _prepare(
            lambda: prepare.replay_artifact_preparation_on_held_switch_fd(
                root_fd,
                switch_fd,
                require_terminal=True,
                runtime_layout="post-switch",
            )
        )
        preparation_descriptor, preparation_document = _held_session_descriptor(
            root_fd,
            preparation_fd,
            prepare.SNAPSHOT_DIRECTORY_NAME,
            "artifact preparation session",
        )
        atomic_descriptor, atomic_document = _held_session_descriptor(
            root_fd,
            atomic_fd,
            ATOMIC_CAPTURE_DIRECTORY_NAME,
            "atomic switch capture",
        )
        if dict(pre_preparation.session) != dict(preparation_document):
            _fail("pre-switch preparation replay disagrees with its held session")
        if dict(post_preparation.session) != dict(preparation_document):
            _fail("post-switch preparation replay disagrees with its held session")
        if dict(exchange.session) != dict(atomic_document):
            _fail("atomic switch replay disagrees with its held session")
        _cross_bind_preparation_and_exchange(
            preparation_document,
            (pre_preparation.candidate_runtime, pre_preparation.rollback_runtime),
            exchange,
            (post_preparation.candidate_runtime, post_preparation.rollback_runtime),
        )
        session = {
            "schema_version": SESSION_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "preparation_session": preparation_descriptor.as_json(),
            "atomic_switch_session": atomic_descriptor.as_json(),
            "preparation_pre_switch": {
                "candidate": pre_preparation.candidate_runtime.as_json(),
                "rollback": pre_preparation.rollback_runtime.as_json(),
            },
            "preparation_post_switch": {
                "candidate": post_preparation.candidate_runtime.as_json(),
                "rollback": post_preparation.rollback_runtime.as_json(),
            },
        }
        session_created = _common(
            lambda: common.write_create_only_json(
                transaction_fd,
                "session.json",
                session,
                "atomic transaction session",
            )
        )
        verify_atomic_transaction_on_held_switch_fd(
            root_fd,
            switch_fd,
            require_terminal=False,
        )
        _require_held_switch_fd(root_fd, switch_fd)
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                preparation_fd,
                prepare.SNAPSHOT_DIRECTORY_NAME,
                "held artifact preparation session directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                atomic_fd,
                ATOMIC_CAPTURE_DIRECTORY_NAME,
                "held atomic switch capture directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                transaction_fd,
                TRANSACTION_DIRECTORY_NAME,
                "held atomic transaction capture directory",
            )
        )
        _publish_completion_marker(transaction_fd, marker, session_created)
        replay = replay_atomic_transaction_on_held_switch_fd(root_fd, switch_fd)
        _require_held_switch_fd(root_fd, switch_fd)
        return replay
    finally:
        _close_quietly(atomic_fd)
        _close_quietly(preparation_fd)
        _close_quietly(transaction_fd)


def _capture_atomic_transaction_then_on_success_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    continuation: Callable[[AtomicTransactionReplay], T],
) -> T:
    """Invoke one trusted internal continuation after a normal held-FD capture.

    The transaction completion pair itself is never a public handoff or a
    serializable capability.  If the producer raises
    ``ambiguous-terminal-publication``, this continuation is not called.
    The narrow v4 compositor owns the callback and retains its exclusive root
    and switch locks for its full duration; a later standalone replay is not
    equivalent.
    """

    if not callable(continuation):
        _fail("transaction success continuation must be callable", code="invalid-continuation")
    replay = _capture_atomic_transaction_on_held_switch_fd(root_fd, switch_fd)
    _require_held_switch_fd(root_fd, switch_fd)
    result = continuation(replay)
    _require_held_switch_fd(root_fd, switch_fd)
    return result


def _capture_atomic_transaction_then_terminal_success_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    continuation: Callable[[AtomicTransactionReplay], T],
) -> T:
    """Invoke one terminal same-stack continuation after a normal capture.

    A successful terminal continuation may publish a hard-linked receipt whose
    success must remain the enclosing invocation's final fallible action.
    Unlike the ordinary continuation helper, this variant deliberately makes
    no post-continuation held-FD replay: an error observed after that receipt
    was published could otherwise turn a visible terminal pair into a failed
    producer return.  The callback is private and lexical; it receives no
    resumable path capability.
    """

    if not callable(continuation):
        _fail("transaction terminal continuation must be callable", code="invalid-continuation")
    replay = _capture_atomic_transaction_on_held_switch_fd(root_fd, switch_fd)
    _require_held_switch_fd(root_fd, switch_fd)
    return continuation(replay)


def capture_atomic_transaction(evidence_root: Path) -> dict[str, Any]:
    """Create the fixed raw pre-replay → exchange → post-replay transaction.

    A successful return is the only producer-success signal. A post-link
    ``ambiguous-terminal-publication`` may leave structurally replayable raw
    evidence, but callers must stop rather than retry or consume it as this
    invocation's completed transaction.
    """

    _assert_external_to_source(evidence_root)
    root_fd = _common(lambda: common.open_private_evidence_directory(evidence_root, "--evidence-root"))
    switch_fd: int | None = None
    try:
        _lock(root_fd, fcntl.LOCK_EX, "exclusive evidence-root")
        switch_fd = _open_terminal_child(
            root_fd,
            prepare.SWITCH_DIRECTORY_NAME,
            "isolated rollback switch directory",
        )
        _lock(switch_fd, fcntl.LOCK_EX, "exclusive rollback switch")
        replay = _capture_atomic_transaction_on_held_switch_fd(root_fd, switch_fd)
        return dict(replay.session)
    finally:
        _unlock_quietly(switch_fd)
        _close_quietly(switch_fd)
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        session = capture_atomic_transaction(args.evidence_root)
    except RollbackAtomicTransactionError as error:
        print(f"atomic rollback transaction: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(session) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
