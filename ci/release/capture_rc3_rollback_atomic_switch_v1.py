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
from typing import Any, Callable, NoReturn, Sequence, TypeVar

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


def capture_atomic_switch(request: SwitchRequest) -> dict[str, Any]:
    """Exchange two isolated staged files and publish five raw evidence leaves."""

    switch_name = _leaf(request.switch_dir_name, "--switch-dir-name")
    capture_name = _leaf(request.capture_name, "--capture-name")
    if switch_name == capture_name:
        _fail("--switch-dir-name and --capture-name must differ")
    _assert_external_to_source(request.evidence_root)
    root_fd = _common(lambda: common.open_private_evidence_directory(request.evidence_root, "--evidence-root"))
    switch_fd: int | None = None
    capture_fd: int | None = None
    try:
        switch_fd = _open_private_child(root_fd, switch_name, "isolated rollback switch directory")
        _lock(switch_fd, "isolated rollback switch directory")
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
        _remove_marker(capture_fd, marker)
        return session
    finally:
        _close_quietly(capture_fd)
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
