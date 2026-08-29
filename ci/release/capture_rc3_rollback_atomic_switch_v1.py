#!/usr/bin/env python3
"""Capture one isolated FD-based rollback switch using ``renameat2`` exchange.

This raw producer is deliberately limited to two already-staged private
regular files below one private evidence-root child.  It never copies an
external artifact, launches/stops Riley, opens a network connection, uses a
GPU, or decides whether a rollback succeeded.  Its switch directory is an
isolated evidence workspace, not a deployment target.

It takes a nonblocking lock, snapshots the candidate active and reconstructed
rollback staged file identities, performs exactly Linux
``renameat2(RENAME_EXCHANGE)`` with the *same held directory FD*, snapshots the
two resulting names, and writes five create-only raw leaves.  There is no
``mv``/``rename`` fallback: lack of the Linux syscall or an exchange error is a
nonterminal failure.  A later authenticated runner may bind these raw leaves;
this helper itself emits only ``captured/not-run`` evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import provenance_v2_common as common


sys.dont_write_bytecode = True

SWITCH_VERSION = "riley.rc3-rollback-atomic-switch.v1"
STAT_VERSION = "riley.rc3-rollback-switch-stat.v1"
SESSION_VERSION = "riley.rc3-rollback-atomic-switch-capture.v1"
INCOMPLETE_MARKER_VERSION = "riley.rc3-rollback-atomic-switch-incomplete.v1"
INCOMPLETE_MARKER_NAME = "capture-incomplete.json"
ACTIVE_NAME = "active"
ROLLBACK_STAGED_NAME = "rollback-staged"
RENAME_EXCHANGE = 0x2

STAT_FIELDS = frozenset(
    {
        "schema_version",
        "entry_name",
        "device",
        "inode",
        "mode",
        "nlink",
        "byte_length",
        "mtime_ns",
        "ctime_ns",
    }
)
TRANSCRIPT_FIELDS = frozenset(
    {
        "schema_version",
        "capture_status",
        "qualification_status",
        "operation",
        "switch_directory",
        "active_name",
        "rollback_staged_name",
        "pre_active",
        "pre_rollback_staged",
        "post_active",
        "post_candidate_staged",
    }
)
SESSION_FIELDS = frozenset(
    {
        "schema_version",
        "capture_status",
        "qualification_status",
        "switch_directory",
        "atomic_switch",
    }
)
ATOMIC_SWITCH_FIELDS = frozenset(
    {
        "pre_active_stat",
        "post_active_stat",
        "candidate_staged_stat",
        "rollback_staged_stat",
        "rename_transcript",
    }
)

SAFE_LEAF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class AtomicSwitchCaptureError(ValueError):
    """The isolated atomic switch cannot safely publish raw evidence."""


def _fail(message: str) -> NoReturn:
    raise AtomicSwitchCaptureError(message)


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(str(error))


def _leaf(value: str, label: str) -> str:
    if type(value) is not str or SAFE_LEAF_RE.fullmatch(value) is None or value in {".", ".."}:
        _fail(f"{label} must be a normalized nonhidden direct-child name")
    return value


def _required_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        _fail(f"host does not expose required {name} safety flag")
    return value


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | _required_flag("O_DIRECTORY")
        | _required_flag("O_NOFOLLOW")
        | _required_flag("O_CLOEXEC")
        | _required_flag("O_NONBLOCK")
    )


def _file_flags() -> int:
    return os.O_RDONLY | _required_flag("O_NOFOLLOW") | _required_flag("O_CLOEXEC") | _required_flag("O_NONBLOCK")


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _fsync(descriptor: int, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        _fail(f"cannot fsync {label}: {error}")


def _assert_external_to_source(evidence_root: Path) -> None:
    source_root = _source_root()
    try:
        evidence_root.relative_to(source_root)
    except ValueError:
        return
    _fail("--evidence-root must be outside the source checkout")


def _source_root() -> Path:
    """Factored so CPU-only tests can place their temporary root elsewhere."""

    return Path(__file__).resolve().parents[2]


def _open_private_child(parent_fd: int, name: str, label: str) -> int:
    name = _leaf(name, f"{label} name")
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as error:
        _fail(f"cannot open {label} without following links: {error}")
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            _fail(f"{label} must be an effective-UID-owned mode 0700 directory")
        return descriptor
    except BaseException:
        _close_quietly(descriptor)
        raise


def _require_held_switch_fd(root_fd: int, switch_fd: int, switch_name: str) -> None:
    """Bind a caller-held switch FD to one visible private root child.

    The FD-native transaction entry point intentionally does not acquire or
    release a switch lock.  It does, however, prove that the supplied FD still
    resolves from the held private root under the declared direct-child name.
    """

    switch_name = _leaf(switch_name, "held switch directory name")
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "atomic switch evidence root",
        )
    )
    _common(
        lambda: common.require_private_child_directory_fd(
            root_fd,
            switch_fd,
            switch_name,
            "held atomic switch directory",
        )
    )


def _new_private_child(parent_fd: int, name: str, label: str) -> int:
    name = _leaf(name, f"{label} name")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        _fail(f"{label} already exists; capture output is create-only")
    except OSError as error:
        _fail(f"cannot create {label}: {error}")
    try:
        visible = os.lstat(name, dir_fd=parent_fd)
    except OSError as error:
        _fail(f"cannot inspect newly created {label}: {error}")
    descriptor = _open_private_child(parent_fd, name, label)
    try:
        metadata = os.fstat(descriptor)
        if (
            metadata.st_nlink != 2
            or (visible.st_dev, visible.st_ino, visible.st_mode, visible.st_nlink)
            != (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink)
        ):
            _fail(f"{label} was not created as a new private directory")
        _fsync(descriptor, label)
        _fsync(parent_fd, f"parent directory after {label}")
        try:
            after = os.lstat(name, dir_fd=parent_fd)
        except OSError as error:
            _fail(f"cannot re-inspect newly created {label}: {error}")
        if (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
        ):
            _fail(f"{label} changed before it became durable")
        return descriptor
    except BaseException:
        _close_quietly(descriptor)
        raise


def _lock(descriptor: int, label: str) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(f"cannot acquire exclusive {label} lock: {error}")


@dataclass(frozen=True)
class StagedIdentity:
    device: int
    inode: int
    mode: int
    nlink: int
    byte_length: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> "StagedIdentity":
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=stat.S_IMODE(metadata.st_mode),
            nlink=metadata.st_nlink,
            byte_length=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        )

    def as_json(self, entry_name: str) -> dict[str, Any]:
        return {
            "schema_version": STAT_VERSION,
            "entry_name": entry_name,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "nlink": self.nlink,
            "byte_length": self.byte_length,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


def _read_staged_identity(directory_fd: int, name: str, label: str) -> StagedIdentity:
    name = _leaf(name, f"{label} name")
    try:
        before = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot inspect {label}: {error}")
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o700
        or before.st_size < 1
    ):
        _fail(f"{label} must be a nonempty effective-UID-owned single-link mode 0700 regular file")
    try:
        descriptor = os.open(name, _file_flags(), dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot open {label} without following links: {error}")
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_size < 1
            or (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        ):
            _fail(f"{label} changed while it was opened")
        result = StagedIdentity.from_stat(opened)
    finally:
        _close_quietly(descriptor)
    try:
        after = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot re-inspect {label}: {error}")
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail(f"{label} changed while it was sampled")
    return result


def _rename_exchange(directory_fd: int) -> None:
    """Perform exactly one same-directory Linux exchange without fallback."""

    if sys.platform != "linux":
        _fail("renameat2 exchange is supported only on Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _fail("host libc does not expose renameat2")
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        ctypes.c_int(directory_fd),
        ctypes.c_char_p(ACTIVE_NAME.encode("ascii")),
        ctypes.c_int(directory_fd),
        ctypes.c_char_p(ROLLBACK_STAGED_NAME.encode("ascii")),
        ctypes.c_uint(RENAME_EXCHANGE),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        error = OSError(error_number, os.strerror(error_number))
        _fail(f"renameat2(RENAME_EXCHANGE) failed without fallback: {error}")


def _remove_marker(capture_fd: int, marker: dict[str, Any]) -> None:
    try:
        os.unlink(INCOMPLETE_MARKER_NAME, dir_fd=capture_fd)
    except OSError as error:
        _fail(f"cannot remove incomplete marker: {error}")
    try:
        _fsync(capture_fd, "capture directory after incomplete marker removal")
    except AtomicSwitchCaptureError:
        try:
            _common(
                lambda: common.write_create_only_json(
                    capture_fd, INCOMPLETE_MARKER_NAME, marker, "restored incomplete marker"
                )
            )
        except AtomicSwitchCaptureError:
            pass
        raise


@dataclass(frozen=True)
class SwitchRequest:
    evidence_root: Path
    switch_dir_name: str
    capture_name: str


def capture_atomic_switch_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    switch_name: str,
    capture_name: str,
) -> dict[str, Any]:
    """Capture one exchange through caller-held root/switch FDs.

    This FD-native entry point owns only its new capture-child FD.  It does
    not reopen or lock ``switch_fd``: a runner may therefore retain one
    exclusive switch lock across preparation replay, this exchange, and its
    post-switch replay.  Callers must hold that lock before entry.
    """

    switch_name = _leaf(switch_name, "held switch directory name")
    capture_name = _leaf(capture_name, "atomic switch capture name")
    if switch_name == capture_name:
        _fail("held switch directory and atomic switch capture names must differ")
    _require_held_switch_fd(root_fd, switch_fd, switch_name)
    capture_fd: int | None = None
    try:
        capture_fd = _new_private_child(root_fd, capture_name, "atomic switch capture directory")
        _lock(capture_fd, "atomic switch capture directory")
        marker = {
            "schema_version": INCOMPLETE_MARKER_VERSION,
            "capture_status": "incomplete",
            "qualification_status": "not-run",
        }
        _common(
            lambda: common.write_create_only_json(
                capture_fd, INCOMPLETE_MARKER_NAME, marker, "atomic switch incomplete marker"
            )
        )
        pre_active = _read_staged_identity(switch_fd, ACTIVE_NAME, "candidate active staged artifact")
        pre_rollback = _read_staged_identity(
            switch_fd, ROLLBACK_STAGED_NAME, "reconstructed rollback staged artifact"
        )
        if pre_active.device != pre_rollback.device:
            _fail("active and rollback staged artifacts must be on one filesystem")
        if pre_active.inode == pre_rollback.inode:
            _fail("active and rollback staged artifacts must be distinct single-link files")
        pre_active_created = _common(
            lambda: common.write_create_only_json(
                capture_fd,
                "pre-active-stat.json",
                pre_active.as_json(ACTIVE_NAME),
                "pre-switch active stat",
            )
        )
        rollback_staged_created = _common(
            lambda: common.write_create_only_json(
                capture_fd,
                "rollback-staged-stat.json",
                pre_rollback.as_json(ROLLBACK_STAGED_NAME),
                "pre-switch rollback staged stat",
            )
        )
        _rename_exchange(switch_fd)
        _fsync(switch_fd, "isolated rollback switch directory after exchange")
        post_active = _read_staged_identity(switch_fd, ACTIVE_NAME, "post-switch rollback active artifact")
        candidate_staged = _read_staged_identity(
            switch_fd, ROLLBACK_STAGED_NAME, "post-switch candidate staged artifact"
        )
        if post_active.device != pre_rollback.device or post_active.inode != pre_rollback.inode:
            _fail("renameat2 exchange did not place rollback staged inode at active")
        if candidate_staged.device != pre_active.device or candidate_staged.inode != pre_active.inode:
            _fail("renameat2 exchange did not place candidate active inode at rollback staged name")
        post_active_created = _common(
            lambda: common.write_create_only_json(
                capture_fd,
                "post-active-stat.json",
                post_active.as_json(ACTIVE_NAME),
                "post-switch active stat",
            )
        )
        candidate_staged_created = _common(
            lambda: common.write_create_only_json(
                capture_fd,
                "candidate-staged-stat.json",
                candidate_staged.as_json(ROLLBACK_STAGED_NAME),
                "post-switch candidate staged stat",
            )
        )
        transcript = {
            "schema_version": SWITCH_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "operation": "renameat2-rename-exchange",
            "switch_directory": switch_name,
            "active_name": ACTIVE_NAME,
            "rollback_staged_name": ROLLBACK_STAGED_NAME,
            "pre_active": pre_active.as_json(ACTIVE_NAME),
            "pre_rollback_staged": pre_rollback.as_json(ROLLBACK_STAGED_NAME),
            "post_active": post_active.as_json(ACTIVE_NAME),
            "post_candidate_staged": candidate_staged.as_json(ROLLBACK_STAGED_NAME),
        }
        transcript_created = _common(
            lambda: common.write_create_only_json(
                capture_fd, "rename-transcript.json", transcript, "renameat2 exchange transcript"
            )
        )

        def descriptor(created: common.CreatedEvidence, name: str) -> dict[str, Any]:
            return created.descriptor(f"{capture_name}/{name}", name).as_json()

        atomic_switch = {
            "pre_active_stat": descriptor(pre_active_created, "pre-active-stat.json"),
            "post_active_stat": descriptor(post_active_created, "post-active-stat.json"),
            "candidate_staged_stat": descriptor(candidate_staged_created, "candidate-staged-stat.json"),
            "rollback_staged_stat": descriptor(rollback_staged_created, "rollback-staged-stat.json"),
            "rename_transcript": descriptor(transcript_created, "rename-transcript.json"),
        }
        session = {
            "schema_version": SESSION_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "switch_directory": switch_name,
            "atomic_switch": atomic_switch,
        }
        _common(lambda: common.write_create_only_json(capture_fd, "session.json", session, "atomic switch session"))
        _require_held_switch_fd(root_fd, switch_fd, switch_name)
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                capture_fd,
                capture_name,
                "held atomic switch capture directory",
            )
        )
        _remove_marker(capture_fd, marker)
        return session
    finally:
        _close_quietly(capture_fd)


def capture_atomic_switch(request: SwitchRequest) -> dict[str, Any]:
    """Open/lock one isolated switch then invoke the FD-native raw producer."""

    switch_name = _leaf(request.switch_dir_name, "--switch-dir-name")
    capture_name = _leaf(request.capture_name, "--capture-name")
    if switch_name == capture_name:
        _fail("--switch-dir-name and --capture-name must differ")
    _assert_external_to_source(request.evidence_root)
    root_fd = _common(lambda: common.open_private_evidence_directory(request.evidence_root, "--evidence-root"))
    switch_fd: int | None = None
    try:
        switch_fd = _open_private_child(root_fd, switch_name, "isolated rollback switch directory")
        _lock(switch_fd, "isolated rollback switch directory")
        return capture_atomic_switch_on_held_switch_fd(
            root_fd,
            switch_fd,
            switch_name,
            capture_name,
        )
    finally:
        _close_quietly(switch_fd)
        _close_quietly(root_fd)


def _exact(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(f"{label} fields differ; expected={sorted(fields)}, actual={actual}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _parse_stat_document(value: Any, label: str, entry_name: str) -> StagedIdentity:
    row = _exact(value, STAT_FIELDS, label)
    if row["schema_version"] != STAT_VERSION or row["entry_name"] != entry_name:
        _fail(f"{label} does not identify the expected switch entry")
    identity = StagedIdentity(
        device=_nonnegative_int(row["device"], f"{label}.device"),
        inode=_nonnegative_int(row["inode"], f"{label}.inode"),
        mode=_nonnegative_int(row["mode"], f"{label}.mode"),
        nlink=_nonnegative_int(row["nlink"], f"{label}.nlink"),
        byte_length=_nonnegative_int(row["byte_length"], f"{label}.byte_length"),
        mtime_ns=_nonnegative_int(row["mtime_ns"], f"{label}.mtime_ns"),
        ctime_ns=_nonnegative_int(row["ctime_ns"], f"{label}.ctime_ns"),
    )
    if identity.mode != 0o700 or identity.nlink != 1 or identity.byte_length < 1:
        _fail(f"{label} is not a nonempty single-link mode-0700 staged artifact")
    return identity


def _marker_absent(capture_fd: int) -> None:
    try:
        os.lstat(INCOMPLETE_MARKER_NAME, dir_fd=capture_fd)
    except FileNotFoundError:
        return
    except OSError as error:
        _fail(f"cannot inspect atomic switch incomplete marker: {error}")
    _fail("atomic switch incomplete marker is still present")


def _atomic_descriptors(
    capture_fd: int,
    capture_name: str,
    value: Any,
) -> dict[str, tuple[common.EvidenceDescriptor, Mapping[str, Any]]]:
    row = _exact(value, ATOMIC_SWITCH_FIELDS, "atomic switch session.atomic_switch")
    expected_names = {
        "pre_active_stat": "pre-active-stat.json",
        "post_active_stat": "post-active-stat.json",
        "candidate_staged_stat": "candidate-staged-stat.json",
        "rollback_staged_stat": "rollback-staged-stat.json",
        "rename_transcript": "rename-transcript.json",
    }
    descriptors: dict[str, common.EvidenceDescriptor] = {}
    for field, filename in expected_names.items():
        descriptor = _common(
            lambda field=field: common.parse_descriptor(
                row[field],
                f"atomic switch session.atomic_switch.{field}",
            )
        )
        if descriptor.path != f"{capture_name}/{filename}" or descriptor.byte_length < 1:
            _fail(f"atomic switch descriptor {field!r} is not its fixed nonempty capture leaf")
        descriptors[field] = descriptor
    _common(lambda: common.require_unique_descriptors(tuple(descriptors.values()), "atomic switch descriptors"))
    result: dict[str, tuple[common.EvidenceDescriptor, Mapping[str, Any]]] = {}
    for field, descriptor in descriptors.items():
        filename = expected_names[field]
        held_descriptor = _common(
            lambda field=field, descriptor=descriptor, filename=filename: common.rebase_descriptor_to_held_leaf(
                descriptor,
                expected_root_relative_path=f"{capture_name}/{filename}",
                leaf_name=filename,
                label=f"atomic switch session.atomic_switch.{field}",
            )
        )
        _raw, document = _common(
            lambda field=field, descriptor=held_descriptor: common.read_private_descriptor_json_leaf(
                capture_fd,
                descriptor,
                f"atomic switch raw {field}",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        result[field] = (descriptor, document)
    return result


def verify_atomic_switch_capture_on_held_switch_fd(
    root_fd: int,
    switch_fd: int,
    switch_name: str,
    capture_name: str,
) -> dict[str, Any]:
    """Replay one terminal atomic-switch capture through caller-held FDs.

    This is raw mechanism replay, not a rollback-success verdict.  It checks
    fixed descriptor paths, exact canonical session/stat/transcript grammar,
    inode exchange relationships, the current held switch names, and the
    absence of the capture marker.  It never opens or locks ``switch_fd``;
    transaction callers retain their exclusive lock across this replay.
    """

    switch_name = _leaf(switch_name, "held switch directory name")
    capture_name = _leaf(capture_name, "atomic switch capture name")
    if switch_name == capture_name:
        _fail("held switch directory and atomic switch capture names must differ")
    _require_held_switch_fd(root_fd, switch_fd, switch_name)
    capture_fd = _common(
        lambda: common.open_private_child_directory(
            root_fd,
            capture_name,
            "atomic switch capture directory",
        )
    )
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                capture_fd,
                capture_name,
                "held atomic switch capture directory",
            )
        )
        _marker_absent(capture_fd)
        session = _common(
            lambda: common.read_private_canonical_json_leaf(
                capture_fd,
                "session.json",
                "atomic switch session",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        assert isinstance(session, Mapping)
        row = _exact(session, SESSION_FIELDS, "atomic switch session")
        if (
            row["schema_version"] != SESSION_VERSION
            or row["capture_status"] != "captured"
            or row["qualification_status"] != "not-run"
            or row["switch_directory"] != switch_name
        ):
            _fail("atomic switch session is not an exact captured/not-run session")
        documents = _atomic_descriptors(capture_fd, capture_name, row["atomic_switch"])
        pre_active = _parse_stat_document(
            documents["pre_active_stat"][1],
            "atomic switch pre-active stat",
            ACTIVE_NAME,
        )
        pre_rollback = _parse_stat_document(
            documents["rollback_staged_stat"][1],
            "atomic switch rollback-staged stat",
            ROLLBACK_STAGED_NAME,
        )
        post_active = _parse_stat_document(
            documents["post_active_stat"][1],
            "atomic switch post-active stat",
            ACTIVE_NAME,
        )
        candidate_staged = _parse_stat_document(
            documents["candidate_staged_stat"][1],
            "atomic switch candidate-staged stat",
            ROLLBACK_STAGED_NAME,
        )
        transcript = _exact(documents["rename_transcript"][1], TRANSCRIPT_FIELDS, "atomic switch transcript")
        if (
            transcript["schema_version"] != SWITCH_VERSION
            or transcript["capture_status"] != "captured"
            or transcript["qualification_status"] != "not-run"
            or transcript["operation"] != "renameat2-rename-exchange"
            or transcript["switch_directory"] != switch_name
            or transcript["active_name"] != ACTIVE_NAME
            or transcript["rollback_staged_name"] != ROLLBACK_STAGED_NAME
        ):
            _fail("atomic switch transcript is not an exact renameat2 exchange record")
        if (
            transcript["pre_active"] != pre_active.as_json(ACTIVE_NAME)
            or transcript["pre_rollback_staged"] != pre_rollback.as_json(ROLLBACK_STAGED_NAME)
            or transcript["post_active"] != post_active.as_json(ACTIVE_NAME)
            or transcript["post_candidate_staged"] != candidate_staged.as_json(ROLLBACK_STAGED_NAME)
        ):
            _fail("atomic switch transcript disagrees with its captured stat leaves")
        if (
            pre_active.device != pre_rollback.device
            or pre_active.inode == pre_rollback.inode
            or (post_active.device, post_active.inode) != (pre_rollback.device, pre_rollback.inode)
            or (candidate_staged.device, candidate_staged.inode) != (pre_active.device, pre_active.inode)
        ):
            _fail("atomic switch stat leaves do not prove the declared inode exchange")
        current_active = _read_staged_identity(switch_fd, ACTIVE_NAME, "current post-switch active artifact")
        current_candidate_staged = _read_staged_identity(
            switch_fd,
            ROLLBACK_STAGED_NAME,
            "current post-switch candidate staged artifact",
        )
        if current_active != post_active or current_candidate_staged != candidate_staged:
            _fail("current held switch entries disagree with the terminal atomic switch record")
        _marker_absent(capture_fd)
        _require_held_switch_fd(root_fd, switch_fd, switch_name)
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                capture_fd,
                capture_name,
                "held atomic switch capture directory",
            )
        )
        return dict(session)
    finally:
        _close_quietly(capture_fd)


def verify_atomic_switch_capture(
    evidence_root: Path,
    switch_name: str,
    capture_name: str,
) -> dict[str, Any]:
    """Standalone shared-lock wrapper for terminal atomic-switch replay."""

    switch_name = _leaf(switch_name, "--switch-dir-name")
    capture_name = _leaf(capture_name, "--capture-name")
    if switch_name == capture_name:
        _fail("--switch-dir-name and --capture-name must differ")
    _assert_external_to_source(evidence_root)
    root_fd = _common(lambda: common.open_private_evidence_directory(evidence_root, "--evidence-root"))
    switch_fd: int | None = None
    try:
        switch_fd = _open_private_child(root_fd, switch_name, "isolated rollback switch directory")
        try:
            fcntl.flock(switch_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except (AttributeError, OSError) as error:
            _fail(f"cannot acquire shared isolated rollback switch lock: {error}")
        return verify_atomic_switch_capture_on_held_switch_fd(
            root_fd,
            switch_fd,
            switch_name,
            capture_name,
        )
    finally:
        if switch_fd is not None:
            try:
                fcntl.flock(switch_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        _close_quietly(switch_fd)
        _close_quietly(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--switch-dir-name", required=True)
    parser.add_argument("--capture-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        session = capture_atomic_switch(
            SwitchRequest(
                evidence_root=args.evidence_root,
                switch_dir_name=args.switch_dir_name,
                capture_name=args.capture_name,
            )
        )
    except AtomicSwitchCaptureError as error:
        print(f"atomic switch capture: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(session) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
