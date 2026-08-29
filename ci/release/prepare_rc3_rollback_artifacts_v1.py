#!/usr/bin/env python3
"""Prepare immutable rollback artifacts and isolated executable runtime copies.

This raw preparation producer appends only fixed create-only children to an
already-private reconstructed-baseline evidence root.  It snapshots six
absolute host inputs through no-follow directory FDs into immutable mode-0600
artifact leaves, then materializes the two binary snapshots into distinct
mode-0700 ``active`` and ``rollback-staged`` files below an isolated switch
directory.  The latter shape is intentionally compatible with the separate
``renameat2(RENAME_EXCHANGE)`` raw switch producer.

It does not launch Riley, invoke Docker/NVIDIA tools, make network requests,
rename a deployment artifact, perform the atomic switch, or decide that a
rollback or qualification succeeded.  Its session is raw ``captured/not-run``
mechanism evidence only.  A later authenticated runner must replay this
session in the explicit pre-switch layout, its terminal marker closure, phase
sessions, and switch session before it may construct a v3 bind request.  A
post-switch preparation replay is available only as a content/inode mapping
check and must be paired with independently replayed atomic-switch evidence;
a later semantic layer owns all operational interpretation.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import provenance_v2_common as common


sys.dont_write_bytecode = True

SESSION_VERSION = "riley.rc3-rollback-artifact-preparation.v1"
INCOMPLETE_MARKER_VERSION = "riley.rc3-rollback-artifact-preparation-incomplete.v1"
INCOMPLETE_MARKER_NAME = "capture-incomplete.json"
COMPLETE_MARKER_VERSION = "riley.rc3-rollback-artifact-preparation-complete.v1"
COMPLETE_INTENT_NAME = "capture-complete.intent"
COMPLETE_MARKER_NAME = "capture-complete.json"
SNAPSHOT_DIRECTORY_NAME = "rollback-v3-artifact-snapshot"
ARTIFACT_DIRECTORY_NAME = "rollback-v3-artifacts"
SWITCH_DIRECTORY_NAME = "rollback-v3-switch"
ACTIVE_NAME = "active"
ROLLBACK_STAGED_NAME = "rollback-staged"
MAX_IMAGE_INSPECT_BYTES = 16 * 1024 * 1024


class RollbackArtifactPreparationError(ValueError):
    """Artifact snapshots or runtime staging cannot safely be prepared."""


def _fail(code: str, message: str) -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = RollbackArtifactPreparationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


@dataclass(frozen=True)
class PreparationRequest:
    evidence_root: Path
    candidate_binary: Path
    candidate_bundle: Path
    candidate_image_inspect: Path
    rollback_binary: Path
    rollback_bundle: Path
    rollback_image_inspect: Path


@dataclass(frozen=True)
class RuntimeReplay:
    """One hash-verified runtime state observed through a held switch FD."""

    sha256: str
    byte_length: int
    device: int
    inode: int
    mode: int
    nlink: int
    mtime_ns: int
    ctime_ns: int

    def as_json(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "nlink": self.nlink,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class ArtifactPreparationReplay:
    """Terminal replay state available to an outer held-FD transaction."""

    session: Mapping[str, Any]
    candidate_runtime: RuntimeReplay
    rollback_runtime: RuntimeReplay


def _source_root() -> Path:
    """Factored so CPU-only tests can keep evidence outside a fake checkout."""

    return Path(__file__).resolve().parents[2]


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    source_root = _source_root()
    try:
        evidence_root.relative_to(source_root)
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be outside the source checkout",
    )


def _lock_root(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("output-lock-unavailable", f"cannot acquire exclusive evidence-root lock: {error}")


def _unlock_quietly(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_UN)
    except OSError:
        pass


def _assert_root_children_absent(root_fd: int) -> None:
    """Reserve every fixed direct child before any append-only mutation."""

    for name in (SNAPSHOT_DIRECTORY_NAME, ARTIFACT_DIRECTORY_NAME, SWITCH_DIRECTORY_NAME):
        try:
            os.lstat(name, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail("output-preflight-failed", f"cannot inspect reserved root child {name!r}: {error}")
        _fail("create-only-collision", f"reserved root child {name!r} already exists")


def _descriptor(created: common.CreatedEvidence, relative_path: str, label: str) -> dict[str, Any]:
    return created.descriptor(relative_path, label).as_json()


def _snapshot_identity(created: common.CreatedEvidence) -> dict[str, int]:
    return {
        "device": created.device,
        "inode": created.inode,
        "mode": 0o600,
        "nlink": 1,
    }


def _snapshot_artifact(
    artifacts_fd: int,
    *,
    source: Path,
    output_name: str,
    label: str,
    require_executable: bool,
    maximum_bytes: int,
) -> common.CreatedEvidence:
    return _common(
        lambda: common.snapshot_absolute_regular_create_only(
            source,
            artifacts_fd,
            output_name,
            label,
            maximum_bytes=maximum_bytes,
            minimum_bytes=1,
            require_owner_executable=require_executable,
        )
    )


def _runtime_mapping(
    switch_fd: int,
    *,
    runtime: common.RuntimeMaterialization,
    snapshot: common.CreatedEvidence,
    entry_name: str,
    label: str,
) -> dict[str, Any]:
    """Bind one runtime inode to the just-created immutable binary snapshot."""

    try:
        visible = os.lstat(entry_name, dir_fd=switch_fd)
    except OSError as error:
        _fail("raced-output", f"cannot inspect {label} runtime copy: {error}")
    if (
        not stat.S_ISREG(visible.st_mode)
        or visible.st_uid != os.geteuid()
        or stat.S_IMODE(visible.st_mode) != 0o700
        or visible.st_nlink != 1
        or visible.st_size != runtime.byte_length
        or (visible.st_dev, visible.st_ino) != (runtime.device, runtime.inode)
    ):
        _fail("runtime-copy-mismatch", f"{label} runtime copy lost its private mode-0700 identity")
    if (
        runtime.sha256 != snapshot.sha256
        or runtime.byte_length != snapshot.byte_length
        or (runtime.device, runtime.inode) == (snapshot.device, snapshot.inode)
    ):
        _fail("runtime-copy-mismatch", f"{label} runtime copy is not a distinct byte-identical snapshot copy")
    return {
        "switch_directory": SWITCH_DIRECTORY_NAME,
        "entry_name": entry_name,
        "immutable_binary_sha256": snapshot.sha256,
        "immutable_binary_byte_length": snapshot.byte_length,
        "sha256": runtime.sha256,
        "byte_length": runtime.byte_length,
        "device": runtime.device,
        "inode": runtime.inode,
        "mode": 0o700,
        "nlink": 1,
    }


def _lock_shared_switch(switch_fd: int) -> None:
    """Keep the raw atomic helper's exclusive exchange out of this replay."""

    try:
        fcntl.flock(switch_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("switch-lock-unavailable", f"cannot acquire shared rollback switch lock: {error}")


def _unlock_switch_quietly(switch_fd: int) -> None:
    try:
        fcntl.flock(switch_fd, fcntl.LOCK_UN)
    except OSError:
        pass


def _require_held_switch_fd(root_fd: int, switch_fd: int) -> None:
    """Bind one caller-held switch FD to the fixed root direct child."""

    _common(
        lambda: common.require_private_child_directory_fd(
            root_fd,
            switch_fd,
            SWITCH_DIRECTORY_NAME,
            "held rollback switch directory",
        )
    )


def _parse_exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail("unexpected-field-set", f"{label} fields differ; expected={sorted(fields)}, actual={actual}")
    return value


def _read_canonical_leaf(directory_fd: int, name: str, label: str) -> Mapping[str, Any]:
    document = _common(
        lambda: common.read_private_canonical_json_leaf(
            directory_fd,
            name,
            label,
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    assert isinstance(document, Mapping)
    return document


def _check_incomplete_marker(snapshot_fd: int, *, require_terminal: bool) -> None:
    try:
        os.lstat(INCOMPLETE_MARKER_NAME, dir_fd=snapshot_fd)
    except FileNotFoundError:
        if require_terminal:
            return
        _fail("missing-incomplete-marker", "preterminal artifact preparation must retain its incomplete marker")
    except OSError as error:
        _fail("unsafe-evidence-path", f"cannot inspect artifact preparation marker: {error}")
    if require_terminal:
        _fail("incomplete-capture", "artifact preparation incomplete marker is still present")
    marker = _read_canonical_leaf(snapshot_fd, INCOMPLETE_MARKER_NAME, "artifact preparation incomplete marker")
    _parse_exact(
        marker,
        {"schema_version", "capture_status", "qualification_status"},
        "artifact preparation incomplete marker",
    )
    if (
        marker["schema_version"] != INCOMPLETE_MARKER_VERSION
        or marker["capture_status"] != "incomplete"
        or marker["qualification_status"] != "not-run"
    ):
        _fail("invalid-incomplete-marker", "artifact preparation incomplete marker is not the exact v1 marker")


def _check_completion_marker(
    snapshot_fd: int,
    session: common.EvidenceDescriptor,
    *,
    require_terminal: bool,
) -> None:
    """Require a durable paired terminal marker only for terminal replay."""

    if not require_terminal:
        for name in (COMPLETE_INTENT_NAME, COMPLETE_MARKER_NAME):
            try:
                os.lstat(name, dir_fd=snapshot_fd)
            except FileNotFoundError:
                continue
            except OSError as error:
                _fail("unsafe-evidence-path", f"cannot inspect artifact preparation terminal marker: {error}")
            _fail("unexpected-terminal-marker", "preterminal artifact preparation has a completion marker")
        return
    raw = _common(
        lambda: common.read_bounded_paired_hardlink(
            snapshot_fd,
            COMPLETE_MARKER_NAME,
            COMPLETE_INTENT_NAME,
            "artifact preparation completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    marker = _common(
        lambda: common.parse_canonical_json(
            raw,
            "artifact preparation completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    assert isinstance(marker, Mapping)
    _parse_exact(
        marker,
        {"schema_version", "capture_status", "qualification_status", "session_sha256", "session_byte_length"},
        "artifact preparation completion marker",
    )
    if (
        marker["schema_version"] != COMPLETE_MARKER_VERSION
        or marker["capture_status"] != "captured"
        or marker["qualification_status"] != "not-run"
        or marker["session_sha256"] != session.sha256
        or marker["session_byte_length"] != session.byte_length
    ):
        _fail("invalid-completion-marker", "artifact preparation completion marker does not bind session.json")


def _session_descriptor(snapshot_fd: int) -> common.EvidenceDescriptor:
    return _common(
        lambda: common.describe_regular_relative(
            snapshot_fd,
            "session.json",
            "artifact preparation session",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )


def _artifact_map(
    artifacts_fd: int,
    value: Any,
    arm: str,
) -> dict[str, common.EvidenceDescriptor]:
    row = _parse_exact(value, {"binary", "bundle", "image_inspect"}, f"{arm} artifact snapshots")
    expected = {
        "binary": f"{ARTIFACT_DIRECTORY_NAME}/{arm}-binary",
        "bundle": f"{ARTIFACT_DIRECTORY_NAME}/{arm}-bundle",
        "image_inspect": f"{ARTIFACT_DIRECTORY_NAME}/{arm}-image-inspect.json",
    }
    parsed: dict[str, common.EvidenceDescriptor] = {}
    for name in ("binary", "bundle", "image_inspect"):
        descriptor = _common(lambda value=row[name], name=name: common.parse_descriptor(value, f"{arm}.{name}"))
        if descriptor.path != expected[name] or descriptor.byte_length < 1:
            _fail("invalid-artifact-snapshot", f"{arm}.{name} must use its fixed nonempty snapshot path")
        maximum = MAX_IMAGE_INSPECT_BYTES if name == "image_inspect" else common.DEFAULT_MAX_ARTIFACT_BYTES
        held_descriptor = _common(
            lambda descriptor=descriptor, name=name: common.rebase_descriptor_to_held_leaf(
                descriptor,
                expected_root_relative_path=expected[name],
                leaf_name=f"{arm}-{'image-inspect.json' if name == 'image_inspect' else name}",
                label=f"{arm}.{name}",
            )
        )
        _common(
            lambda descriptor=held_descriptor, name=name, maximum=maximum: common.verify_private_snapshot_descriptor_file(
                artifacts_fd,
                descriptor,
                f"{arm}.{name}",
                maximum_bytes=maximum,
            )
        )
        parsed[name] = descriptor
    return parsed


def _snapshot_identity_map(
    artifacts_fd: int,
    value: Any,
    arm: str,
    descriptors: Mapping[str, common.EvidenceDescriptor],
) -> dict[str, tuple[int, int]]:
    """Require every immutable snapshot to retain its recorded private inode."""

    rows = _parse_exact(value, {"binary", "bundle", "image_inspect"}, f"{arm} snapshot identities")
    result: dict[str, tuple[int, int]] = {}
    output_names = {
        "binary": f"{arm}-binary",
        "bundle": f"{arm}-bundle",
        "image_inspect": f"{arm}-image-inspect.json",
    }
    for name in ("binary", "bundle", "image_inspect"):
        row = _parse_exact(rows[name], {"device", "inode", "mode", "nlink"}, f"{arm}.{name} snapshot identity")
        if (
            any(type(row[field]) is not int or row[field] < 0 for field in ("device", "inode", "mode", "nlink"))
            or row["mode"] != 0o600
            or row["nlink"] != 1
        ):
            _fail("invalid-snapshot-identity", f"{arm}.{name} snapshot identity is malformed")
        try:
            visible = os.lstat(output_names[name], dir_fd=artifacts_fd)
        except OSError as error:
            _fail("missing-input", f"cannot inspect {arm}.{name} immutable snapshot: {error}")
        if (
            not stat.S_ISREG(visible.st_mode)
            or visible.st_uid != os.geteuid()
            or stat.S_IMODE(visible.st_mode) != 0o600
            or visible.st_nlink != 1
            or visible.st_size != descriptors[name].byte_length
            or (visible.st_dev, visible.st_ino) != (row["device"], row["inode"])
        ):
            _fail("immutable-snapshot-mismatch", f"{arm}.{name} snapshot lost its recorded private inode")
        result[name] = (visible.st_dev, visible.st_ino)
    return result


def _runtime_row(
    switch_fd: int,
    artifacts_fd: int,
    value: Any,
    *,
    arm: str,
    recorded_entry_name: str,
    visible_entry_name: str,
    binary: common.EvidenceDescriptor,
    snapshot_identity: tuple[int, int],
) -> RuntimeReplay:
    row = _parse_exact(
        value,
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
        },
        f"{arm} runtime materialization",
    )
    integer_fields = {"immutable_binary_byte_length", "byte_length", "device", "inode", "mode", "nlink"}
    if (
        row["switch_directory"] != SWITCH_DIRECTORY_NAME
        or row["entry_name"] != recorded_entry_name
        or row["immutable_binary_sha256"] != binary.sha256
        or row["immutable_binary_byte_length"] != binary.byte_length
        or row["sha256"] != binary.sha256
        or row["byte_length"] != binary.byte_length
        or any(type(row[name]) is not int or row[name] < 0 for name in integer_fields)
        or row["mode"] != 0o700
        or row["nlink"] != 1
    ):
        _fail("invalid-runtime-materialization", f"{arm} runtime mapping does not bind its immutable binary")
    try:
        runtime_stat = os.lstat(visible_entry_name, dir_fd=switch_fd)
        snapshot_stat = os.lstat(f"{arm}-binary", dir_fd=artifacts_fd)
    except OSError as error:
        _fail("missing-input", f"cannot inspect {arm} runtime/snapshot pair: {error}")
    if (
        not stat.S_ISREG(runtime_stat.st_mode)
        or runtime_stat.st_uid != os.geteuid()
        or stat.S_IMODE(runtime_stat.st_mode) != 0o700
        or runtime_stat.st_nlink != 1
        or runtime_stat.st_size != binary.byte_length
        or (runtime_stat.st_dev, runtime_stat.st_ino) != (row["device"], row["inode"])
        or (snapshot_stat.st_dev, snapshot_stat.st_ino) != snapshot_identity
        or (runtime_stat.st_dev, runtime_stat.st_ino) == snapshot_identity
    ):
        _fail("runtime-copy-mismatch", f"{arm} runtime file does not match its recorded distinct inode")
    runtime_before = RuntimeReplay(
        sha256=binary.sha256,
        byte_length=runtime_stat.st_size,
        device=runtime_stat.st_dev,
        inode=runtime_stat.st_ino,
        mode=stat.S_IMODE(runtime_stat.st_mode),
        nlink=runtime_stat.st_nlink,
        mtime_ns=runtime_stat.st_mtime_ns,
        ctime_ns=runtime_stat.st_ctime_ns,
    )
    runtime_descriptor = common.EvidenceDescriptor(
        path=f"{SWITCH_DIRECTORY_NAME}/{visible_entry_name}",
        sha256=binary.sha256,
        byte_length=binary.byte_length,
    )
    held_runtime_descriptor = _common(
        lambda: common.rebase_descriptor_to_held_leaf(
            runtime_descriptor,
            expected_root_relative_path=f"{SWITCH_DIRECTORY_NAME}/{visible_entry_name}",
            leaf_name=visible_entry_name,
            label=f"{arm} runtime copy",
        )
    )
    _common(
        lambda: common.verify_private_runtime_descriptor_file(
            switch_fd,
            held_runtime_descriptor,
            f"{arm} runtime copy",
            maximum_bytes=common.DEFAULT_MAX_ARTIFACT_BYTES,
        )
    )
    try:
        runtime_after = os.lstat(visible_entry_name, dir_fd=switch_fd)
        snapshot_after = os.lstat(f"{arm}-binary", dir_fd=artifacts_fd)
    except OSError as error:
        _fail("raced-input", f"cannot re-inspect {arm} runtime/snapshot pair: {error}")
    if (
        not stat.S_ISREG(runtime_after.st_mode)
        or runtime_after.st_uid != os.geteuid()
        or stat.S_IMODE(runtime_after.st_mode) != 0o700
        or runtime_after.st_nlink != 1
        or runtime_after.st_size != binary.byte_length
        or (runtime_after.st_dev, runtime_after.st_ino) != (row["device"], row["inode"])
        or (snapshot_after.st_dev, snapshot_after.st_ino) != snapshot_identity
        or (runtime_after.st_dev, runtime_after.st_ino) == snapshot_identity
    ):
        _fail("runtime-copy-mismatch", f"{arm} runtime/snapshot identity changed during replay")
    runtime_after_replay = RuntimeReplay(
        sha256=binary.sha256,
        byte_length=runtime_after.st_size,
        device=runtime_after.st_dev,
        inode=runtime_after.st_ino,
        mode=stat.S_IMODE(runtime_after.st_mode),
        nlink=runtime_after.st_nlink,
        mtime_ns=runtime_after.st_mtime_ns,
        ctime_ns=runtime_after.st_ctime_ns,
    )
    if runtime_before != runtime_after_replay:
        _fail("runtime-copy-mismatch", f"{arm} runtime identity changed during hash replay")
    return runtime_after_replay


def replay_artifact_preparation_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    *,
    require_terminal: bool = True,
    runtime_layout: str = "pre-switch",
) -> ArtifactPreparationReplay:
    """Replay preparation using one caller-held switch FD without relocking it.

    ``require_terminal=False`` exists only for the producer's pre-removal
    self-check.  ``runtime_layout`` is ``pre-switch`` by default, matching the
    session's recorded entry names.  ``post-switch`` verifies the same two
    immutable binary bytes/inodes under their exchanged names, but does not
    establish that an exchange happened; callers must independently replay
    the atomic switch's terminal session for that claim.  This entry point
    intentionally neither opens nor locks the switch FD, so an authenticated
    runner can hold one exclusive lock across pre-switch replay, exchange, and
    post-switch replay.
    """

    if runtime_layout not in {"pre-switch", "post-switch"}:
        _fail("invalid-runtime-layout", "runtime layout must be pre-switch or post-switch")
    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "artifact preparation evidence root"))
    _require_held_switch_fd(root_fd, switch_fd)
    snapshot_fd = _common(
        lambda: common.open_private_child_directory(
            root_fd,
            SNAPSHOT_DIRECTORY_NAME,
            "artifact preparation session directory",
        )
    )
    artifacts_fd: int | None = None
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                snapshot_fd,
                SNAPSHOT_DIRECTORY_NAME,
                "held artifact preparation session directory",
            )
        )
        _check_incomplete_marker(snapshot_fd, require_terminal=require_terminal)
        initial_session_descriptor = _session_descriptor(snapshot_fd)
        _raw, session = _common(
            lambda: common.read_private_descriptor_json_leaf(
                snapshot_fd,
                initial_session_descriptor,
                "artifact preparation session",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        _check_completion_marker(
            snapshot_fd,
            initial_session_descriptor,
            require_terminal=require_terminal,
        )
        row = _parse_exact(
            session,
            {
                "schema_version",
                "capture_status",
                "qualification_status",
                "artifact_snapshots",
                "snapshot_identities",
                "runtime_materializations",
            },
            "artifact preparation session",
        )
        if (
            row["schema_version"] != SESSION_VERSION
            or row["capture_status"] != "captured"
            or row["qualification_status"] != "not-run"
        ):
            _fail("invalid-session", "artifact preparation session is not the exact raw v1 session")
        artifacts_fd = _common(
            lambda: common.open_private_child_directory(
                root_fd,
                ARTIFACT_DIRECTORY_NAME,
                "artifact snapshot directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                artifacts_fd,
                ARTIFACT_DIRECTORY_NAME,
                "held immutable artifact snapshot directory",
            )
        )
        snapshots = _parse_exact(row["artifact_snapshots"], {"candidate", "rollback"}, "artifact snapshots")
        candidate = _artifact_map(artifacts_fd, snapshots["candidate"], "candidate")
        rollback = _artifact_map(artifacts_fd, snapshots["rollback"], "rollback")
        identities = _parse_exact(
            row["snapshot_identities"],
            {"candidate", "rollback"},
            "snapshot identities",
        )
        candidate_identities = _snapshot_identity_map(
            artifacts_fd,
            identities["candidate"],
            "candidate",
            candidate,
        )
        rollback_identities = _snapshot_identity_map(
            artifacts_fd,
            identities["rollback"],
            "rollback",
            rollback,
        )
        runtimes = _parse_exact(
            row["runtime_materializations"],
            {"candidate", "rollback"},
            "runtime materializations",
        )
        candidate_runtime_replay = _runtime_row(
            switch_fd,
            artifacts_fd,
            runtimes["candidate"],
            arm="candidate",
            recorded_entry_name=ACTIVE_NAME,
            visible_entry_name=(ACTIVE_NAME if runtime_layout == "pre-switch" else ROLLBACK_STAGED_NAME),
            binary=candidate["binary"],
            snapshot_identity=candidate_identities["binary"],
        )
        rollback_runtime_replay = _runtime_row(
            switch_fd,
            artifacts_fd,
            runtimes["rollback"],
            arm="rollback",
            recorded_entry_name=ROLLBACK_STAGED_NAME,
            visible_entry_name=(ROLLBACK_STAGED_NAME if runtime_layout == "pre-switch" else ACTIVE_NAME),
            binary=rollback["binary"],
            snapshot_identity=rollback_identities["binary"],
        )
        candidate_runtime = os.lstat(ACTIVE_NAME, dir_fd=switch_fd)
        rollback_runtime = os.lstat(ROLLBACK_STAGED_NAME, dir_fd=switch_fd)
        if (
            candidate_runtime.st_dev != rollback_runtime.st_dev
            or candidate_runtime.st_ino == rollback_runtime.st_ino
        ):
            _fail("runtime-copy-mismatch", "candidate and rollback runtime files must be distinct same-filesystem leaves")
        _check_incomplete_marker(snapshot_fd, require_terminal=require_terminal)
        _raw, terminal_session = _common(
            lambda: common.read_private_descriptor_json_leaf(
                snapshot_fd,
                initial_session_descriptor,
                "artifact preparation session",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        if terminal_session != session:
            _fail("raced-input", "artifact preparation session changed during replay")
        _check_completion_marker(
            snapshot_fd,
            initial_session_descriptor,
            require_terminal=require_terminal,
        )
        _require_held_switch_fd(root_fd, switch_fd)
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                artifacts_fd,
                ARTIFACT_DIRECTORY_NAME,
                "held immutable artifact snapshot directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                snapshot_fd,
                SNAPSHOT_DIRECTORY_NAME,
                "held artifact preparation session directory",
            )
        )
        return ArtifactPreparationReplay(
            session=dict(session),
            candidate_runtime=candidate_runtime_replay,
            rollback_runtime=rollback_runtime_replay,
        )
    finally:
        if artifacts_fd is not None:
            os.close(artifacts_fd)
        os.close(snapshot_fd)


def verify_artifact_preparation_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    *,
    require_terminal: bool = True,
    runtime_layout: str = "pre-switch",
) -> dict[str, Any]:
    """Replay preparation and return only its raw terminal session."""

    return dict(
        replay_artifact_preparation_on_held_switch_fd(
            root_fd,
            switch_fd,
            require_terminal=require_terminal,
            runtime_layout=runtime_layout,
        ).session
    )


def verify_artifact_preparation_fd(
    root_fd: int,
    *,
    require_terminal: bool = True,
    runtime_layout: str = "pre-switch",
) -> dict[str, Any]:
    """Replay one layout while holding a short shared lock on a fresh switch FD."""

    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "artifact preparation evidence root"))
    switch_fd = _common(
        lambda: common.open_private_child_directory(
            root_fd,
            SWITCH_DIRECTORY_NAME,
            "isolated rollback switch directory",
        )
    )
    try:
        _lock_shared_switch(switch_fd)
        return verify_artifact_preparation_on_held_switch_fd(
            root_fd,
            switch_fd,
            require_terminal=require_terminal,
            runtime_layout=runtime_layout,
        )
    finally:
        _unlock_switch_quietly(switch_fd)
        os.close(switch_fd)


def verify_artifact_preparation(
    evidence_root: Path,
    *,
    runtime_layout: str = "pre-switch",
) -> dict[str, Any]:
    """Open one private external evidence root and replay terminal preparation."""

    _assert_external_to_source_checkout(evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(evidence_root, "--evidence-root")
    )
    try:
        return verify_artifact_preparation_fd(root_fd, runtime_layout=runtime_layout)
    finally:
        os.close(root_fd)


def _restore_marker(snapshot_fd: int, marker: Mapping[str, Any]) -> None:
    try:
        _common(
            lambda: common.write_create_only_json(
                snapshot_fd,
                INCOMPLETE_MARKER_NAME,
                dict(marker),
                "restored artifact preparation incomplete marker",
            )
        )
    except RollbackArtifactPreparationError:
        pass


def _publish_completion_marker(
    snapshot_fd: int,
    marker: Mapping[str, Any],
    session: common.CreatedEvidence,
) -> None:
    """Durably replace the incomplete marker with a paired terminal receipt.

    The successful terminal state is the exact two-name hard-link pair, not
    the absence of ``capture-incomplete.json``. Therefore an unlink/fsync or
    restore failure leaves no terminal receipt and every terminal verifier
    fails closed instead of mistaking a partially published capture for one
    that completed. A post-link directory-sync error is deliberately raised
    as ``ambiguous-terminal-publication``: its pair remains raw structurally
    replayable evidence, but the producer-success branch must stop.
    """

    try:
        os.unlink(INCOMPLETE_MARKER_NAME, dir_fd=snapshot_fd)
    except OSError as error:
        _fail("marker-removal-failure", f"cannot remove artifact preparation incomplete marker: {error}")
    try:
        os.fsync(snapshot_fd)
    except OSError as error:
        _restore_marker(snapshot_fd, marker)
        _fail("durability-failure", f"cannot synchronize artifact preparation marker removal: {error}")
    completion = {
        "schema_version": COMPLETE_MARKER_VERSION,
        "capture_status": "captured",
        "qualification_status": "not-run",
        "session_sha256": session.sha256,
        "session_byte_length": session.byte_length,
    }
    _common(
        lambda: common.write_create_only_json(
            snapshot_fd,
            COMPLETE_INTENT_NAME,
            completion,
            "artifact preparation completion intent",
        )
    )
    _common(
        lambda: common.publish_create_only_hardlink(
            snapshot_fd,
            COMPLETE_INTENT_NAME,
            COMPLETE_MARKER_NAME,
            "artifact preparation completion marker",
        )
    )


def prepare_artifacts(request: PreparationRequest) -> dict[str, Any]:
    """Snapshot six host artifacts then stage two linked private runtime copies.

    A successful return is the only producer-success signal. In particular,
    ``ambiguous-terminal-publication`` after completion-pair linking leaves
    raw on-disk evidence that a later structural verifier may inspect, but it
    must never be retried or consumed as a successful producer invocation.
    """

    _assert_external_to_source_checkout(request.evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(request.evidence_root, "--evidence-root")
    )
    snapshot_fd: int | None = None
    artifacts_fd: int | None = None
    switch_fd: int | None = None
    try:
        _lock_root(root_fd)
        _assert_root_children_absent(root_fd)
        snapshot_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                SNAPSHOT_DIRECTORY_NAME,
                "artifact preparation session directory",
            )
        )
        marker = {
            "schema_version": INCOMPLETE_MARKER_VERSION,
            "capture_status": "incomplete",
            "qualification_status": "not-run",
        }
        _common(
            lambda: common.write_create_only_json(
                snapshot_fd,
                INCOMPLETE_MARKER_NAME,
                marker,
                "artifact preparation incomplete marker",
            )
        )
        artifacts_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                ARTIFACT_DIRECTORY_NAME,
                "immutable artifact snapshot directory",
            )
        )
        switch_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                SWITCH_DIRECTORY_NAME,
                "isolated rollback switch directory",
            )
        )
        candidate_binary = _snapshot_artifact(
            artifacts_fd,
            source=request.candidate_binary,
            output_name="candidate-binary",
            label="candidate binary",
            require_executable=True,
            maximum_bytes=common.DEFAULT_MAX_ARTIFACT_BYTES,
        )
        candidate_bundle = _snapshot_artifact(
            artifacts_fd,
            source=request.candidate_bundle,
            output_name="candidate-bundle",
            label="candidate bundle",
            require_executable=False,
            maximum_bytes=common.DEFAULT_MAX_ARTIFACT_BYTES,
        )
        candidate_image = _snapshot_artifact(
            artifacts_fd,
            source=request.candidate_image_inspect,
            output_name="candidate-image-inspect.json",
            label="candidate raw image inspect",
            require_executable=False,
            maximum_bytes=MAX_IMAGE_INSPECT_BYTES,
        )
        rollback_binary = _snapshot_artifact(
            artifacts_fd,
            source=request.rollback_binary,
            output_name="rollback-binary",
            label="reconstructed rollback binary",
            require_executable=True,
            maximum_bytes=common.DEFAULT_MAX_ARTIFACT_BYTES,
        )
        rollback_bundle = _snapshot_artifact(
            artifacts_fd,
            source=request.rollback_bundle,
            output_name="rollback-bundle",
            label="reconstructed rollback bundle",
            require_executable=False,
            maximum_bytes=common.DEFAULT_MAX_ARTIFACT_BYTES,
        )
        rollback_image = _snapshot_artifact(
            artifacts_fd,
            source=request.rollback_image_inspect,
            output_name="rollback-image-inspect.json",
            label="reconstructed rollback raw image inspect",
            require_executable=False,
            maximum_bytes=MAX_IMAGE_INSPECT_BYTES,
        )
        candidate_binary_descriptor = candidate_binary.descriptor(
            f"{ARTIFACT_DIRECTORY_NAME}/candidate-binary",
            "candidate binary snapshot",
        )
        rollback_binary_descriptor = rollback_binary.descriptor(
            f"{ARTIFACT_DIRECTORY_NAME}/rollback-binary",
            "rollback binary snapshot",
        )
        candidate_runtime = _common(
            lambda: common.materialize_descriptor_runtime_copy(
                root_fd,
                candidate_binary_descriptor,
                switch_fd,
                ACTIVE_NAME,
                "candidate runtime materialization",
                expected_source_snapshot=candidate_binary,
            )
        )
        rollback_runtime = _common(
            lambda: common.materialize_descriptor_runtime_copy(
                root_fd,
                rollback_binary_descriptor,
                switch_fd,
                ROLLBACK_STAGED_NAME,
                "rollback runtime materialization",
                expected_source_snapshot=rollback_binary,
            )
        )
        if (
            candidate_runtime.device != rollback_runtime.device
            or candidate_runtime.inode == rollback_runtime.inode
        ):
            _fail("runtime-copy-mismatch", "runtime copies must be distinct leaves on one switch filesystem")
        artifact_snapshots = {
            "candidate": {
                "binary": _descriptor(candidate_binary, f"{ARTIFACT_DIRECTORY_NAME}/candidate-binary", "candidate binary"),
                "bundle": _descriptor(candidate_bundle, f"{ARTIFACT_DIRECTORY_NAME}/candidate-bundle", "candidate bundle"),
                "image_inspect": _descriptor(
                    candidate_image,
                    f"{ARTIFACT_DIRECTORY_NAME}/candidate-image-inspect.json",
                    "candidate image inspect",
                ),
            },
            "rollback": {
                "binary": _descriptor(rollback_binary, f"{ARTIFACT_DIRECTORY_NAME}/rollback-binary", "rollback binary"),
                "bundle": _descriptor(rollback_bundle, f"{ARTIFACT_DIRECTORY_NAME}/rollback-bundle", "rollback bundle"),
                "image_inspect": _descriptor(
                    rollback_image,
                    f"{ARTIFACT_DIRECTORY_NAME}/rollback-image-inspect.json",
                    "rollback image inspect",
                ),
            },
        }
        snapshot_identities = {
            "candidate": {
                "binary": _snapshot_identity(candidate_binary),
                "bundle": _snapshot_identity(candidate_bundle),
                "image_inspect": _snapshot_identity(candidate_image),
            },
            "rollback": {
                "binary": _snapshot_identity(rollback_binary),
                "bundle": _snapshot_identity(rollback_bundle),
                "image_inspect": _snapshot_identity(rollback_image),
            },
        }
        session = {
            "schema_version": SESSION_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "artifact_snapshots": artifact_snapshots,
            "snapshot_identities": snapshot_identities,
            "runtime_materializations": {
                "candidate": _runtime_mapping(
                    switch_fd,
                    runtime=candidate_runtime,
                    snapshot=candidate_binary,
                    entry_name=ACTIVE_NAME,
                    label="candidate",
                ),
                "rollback": _runtime_mapping(
                    switch_fd,
                    runtime=rollback_runtime,
                    snapshot=rollback_binary,
                    entry_name=ROLLBACK_STAGED_NAME,
                    label="rollback",
                ),
            },
        }
        session_created = _common(
            lambda: common.write_create_only_json(
                snapshot_fd,
                "session.json",
                session,
                "artifact preparation session",
            )
        )
        verify_artifact_preparation_fd(root_fd, require_terminal=False)
        _publish_completion_marker(snapshot_fd, marker, session_created)
        return verify_artifact_preparation_fd(root_fd)
    finally:
        if switch_fd is not None:
            os.close(switch_fd)
        if artifacts_fd is not None:
            os.close(artifacts_fd)
        if snapshot_fd is not None:
            os.close(snapshot_fd)
        _unlock_quietly(root_fd)
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--candidate-binary", required=True, type=Path)
    parser.add_argument("--candidate-bundle", required=True, type=Path)
    parser.add_argument("--candidate-image-inspect", required=True, type=Path)
    parser.add_argument("--rollback-binary", required=True, type=Path)
    parser.add_argument("--rollback-bundle", required=True, type=Path)
    parser.add_argument("--rollback-image-inspect", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        session = prepare_artifacts(
            PreparationRequest(
                evidence_root=args.evidence_root,
                candidate_binary=args.candidate_binary,
                candidate_bundle=args.candidate_bundle,
                candidate_image_inspect=args.candidate_image_inspect,
                rollback_binary=args.rollback_binary,
                rollback_bundle=args.rollback_bundle,
                rollback_image_inspect=args.rollback_image_inspect,
            )
        )
    except RollbackArtifactPreparationError as error:
        print(f"rollback artifact preparation: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(session) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
