#!/usr/bin/python3.10
"""Sealed no-action private core template for the future RC3 Gate E v3 bundle.

This checkout copy is deliberately *not* a launcher.  It rejects every direct
checkout invocation before opening a control socket, inspecting a lock, or
creating a child.  A future administrator-installed bootstrap may execute an
identical, manifest-bound copy only from a sealed memfd at ``/proc/self/fd/8``.

The current core is a deliberately small protocol endpoint.  It verifies the
sealed source/config descriptors and an unnamed authenticated Unix
``SOCK_SEQPACKET`` parent channel, exchanges a nonce-bound no-action handshake,
and exits.  It has no GPU, Docker, evidence, replay, receipt, or qualification
capability.  In particular it never acquires a lock: the future bootstrap owns
the lock and must keep its descriptor out of this child.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import stat
import struct
import sys
from typing import Final, NoReturn


PYTHON_PATH: Final = "/usr/bin/python3.10"
PYTHON_SHA256: Final = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
CORE_EXECUTION_PATH: Final = "/proc/self/fd/8"
CORE_FD: Final = 8
CONFIG_FD: Final = 9
CONTROL_FD: Final = 10
PARENT_LOCK_FD: Final = 7

CONFIG_SCHEMA_VERSION: Final = "riley.rc3-gate-e-sealed-no-action-core-config.v1"
PROTOCOL_SCHEMA_VERSION: Final = "riley.rc3-gate-e-sealed-no-action-protocol.v1"
CORE_SCOPE: Final = "sealed-no-action-core"
CORE_AUTHORITY: Final = "sealed-no-action-protocol-only"

MAX_CORE_BYTES: Final = 2 * 1024 * 1024
MAX_CONFIG_BYTES: Final = 8 * 1024
MAX_CONTROL_BYTES: Final = 4 * 1024
CONTROL_TIMEOUT_SECONDS: Final = 10.0
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SEALS: Final = (
    fcntl.F_SEAL_SEAL
    | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_GROW
    | fcntl.F_SEAL_WRITE
)
PR_SET_PDEATHSIG: Final = 1


class CoreError(ValueError):
    """The no-action core's sealed parent contract is not safe."""


def _fail(message: str) -> NoReturn:
    raise CoreError(message)


def _die(message: str) -> NoReturn:
    print(f"RC3 Gate E private core v1: {message}", file=sys.stderr)
    raise SystemExit(2)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail(f"cannot encode canonical JSON: {error}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("JSON object contains a duplicate key")
        result[key] = value
    return result


def _parse_canonical_object(
    raw: bytes,
    label: str,
    *,
    newline_terminated: bool,
) -> dict[str, object]:
    expected = raw[:-1] if newline_terminated and raw.endswith(b"\n") else raw
    if newline_terminated and not raw.endswith(b"\n"):
        _fail(f"{label} must end in exactly one newline")
    if expected.endswith(b"\n"):
        _fail(f"{label} has an unexpected extra newline")
    try:
        value = json.loads(
            expected.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"cannot parse {label}: {error}")
    if type(value) is not dict or _canonical_json_bytes(value) != expected:
        _fail(f"{label} is not a canonical JSON object")
    return value


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb", buffering=0) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        _fail(f"cannot hash reviewed Python: {error}")
    return digest.hexdigest()


def _require_isolated_reviewed_python() -> None:
    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.ignore_environment
        and sys.dont_write_bytecode
        and sys.flags.optimize == 0
    ):
        _fail("must run under /usr/bin/python3.10 -I -S -E -B")
    expected_argv = [
        PYTHON_PATH,
        "-I",
        "-S",
        "-E",
        "-B",
        CORE_EXECUTION_PATH,
        "--sealed-no-action-core",
    ]
    if getattr(sys, "orig_argv", None) != expected_argv:
        _fail("reviewed Python invocation arguments differ from the fixed contract")
    try:
        same_executable = os.path.samefile("/proc/self/exe", PYTHON_PATH)
        metadata = os.stat(PYTHON_PATH, follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot authenticate reviewed Python: {error}")
    if (
        not same_executable
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or _sha256_path(PYTHON_PATH) != PYTHON_SHA256
    ):
        _fail("reviewed Python executable differs from the pinned contract")


def _require_sealed_descriptor_invocation(arguments: list[str]) -> None:
    if arguments != ["--sealed-no-action-core"]:
        _fail("only --sealed-no-action-core is accepted")
    if sys.argv[0] != CORE_EXECUTION_PATH or __file__ != CORE_EXECUTION_PATH:
        _fail("must execute only from the sealed /proc/self/fd/8 descriptor")


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
    )


def _read_sealed_memfd(
    descriptor: int,
    label: str,
    *,
    maximum_bytes: int,
) -> tuple[bytes, str]:
    try:
        before = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot inspect {label} descriptor: {error}")
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 0
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        _fail(f"{label} is not a bounded anonymous regular descriptor")
    try:
        seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
    except OSError as error:
        _fail(f"cannot read {label} seals: {error}")
    if seals != REQUIRED_SEALS:
        _fail(f"{label} does not retain every required seal")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        _fail(f"cannot rewind {label}: {error}")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    seen = 0
    while True:
        try:
            block = os.read(descriptor, 64 * 1024)
        except OSError as error:
            _fail(f"cannot read {label}: {error}")
        if not block:
            break
        seen += len(block)
        if seen > maximum_bytes:
            _fail(f"{label} exceeded its byte bound while reading")
        digest.update(block)
        chunks.append(block)
    try:
        after = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot re-inspect {label}: {error}")
    if _stable_identity(after) != _stable_identity(before) or seen != before.st_size:
        _fail(f"{label} changed while it was read")
    return b"".join(chunks), digest.hexdigest()


def _require_descriptor_allowlist() -> None:
    try:
        scanner_fd = os.open(
            "/proc/self/fd",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
    except (OSError, ValueError) as error:
        _fail(f"cannot open inherited-descriptor view: {error}")
    try:
        names = os.listdir(scanner_fd)
    except OSError as error:
        os.close(scanner_fd)
        _fail(f"cannot enumerate inherited descriptors: {error}")
    finally:
        try:
            os.close(scanner_fd)
        except OSError:
            pass
    descriptors: set[int] = set()
    for name in names:
        try:
            descriptor = int(name)
        except ValueError:
            _fail("inherited-descriptor view contains a non-numeric name")
        if descriptor == scanner_fd:
            continue
        try:
            os.fstat(descriptor)
        except OSError as error:
            # procfs can report the directory-scanner FD after it has closed;
            # reject any other live descriptor but do not mistake that transient
            # listing entry for inherited authority.
            if error.errno == errno.EBADF:
                continue
            _fail(f"cannot inspect inherited descriptor {descriptor}: {error}")
        descriptors.add(descriptor)
    required = {0, 1, 2, CORE_FD, CONFIG_FD, CONTROL_FD}
    if descriptors != required:
        _fail(
            "inherited descriptor set must be exactly "
            f"{sorted(required)!r}, got {sorted(descriptors)!r}"
        )
    if PARENT_LOCK_FD in descriptors:
        _fail("parent lock descriptor must never be inherited by the private core")


def _require_distinct_core_and_config_descriptors() -> None:
    try:
        core_metadata = os.fstat(CORE_FD)
        config_metadata = os.fstat(CONFIG_FD)
    except OSError as error:
        _fail(f"cannot compare sealed core/config descriptors: {error}")
    if (core_metadata.st_dev, core_metadata.st_ino) == (
        config_metadata.st_dev,
        config_metadata.st_ino,
    ):
        _fail("sealed core and sealed config must occupy distinct anonymous descriptors")


def _mark_internal_descriptors_close_on_exec() -> None:
    """Prevent the verified source/config/control FDs from leaking on any exit.

    This core never forks or execs.  Setting CLOEXEC after the interpreter has
    loaded the FD-8 source therefore preserves this invocation while making an
    accidental future exec fail closed instead of widening its FD authority.
    """

    for descriptor, label in (
        (CORE_FD, "sealed core"),
        (CONFIG_FD, "sealed config"),
        (CONTROL_FD, "control"),
    ):
        try:
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
            fcntl.fcntl(descriptor, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
        except OSError as error:
            _fail(f"cannot set close-on-exec on {label} descriptor: {error}")


def _require_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail(f"{label} must be a non-zero lowercase SHA-256")
    return value


def _proc_starttime_ticks(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="ascii") as handle:
            raw = handle.read()
    except OSError as error:
        _fail(f"cannot read parent process start time: {error}")
    delimiter = raw.rfind(") ")
    if delimiter < 0:
        _fail("parent process stat has an unexpected shape")
    fields = raw[delimiter + 2 :].split()
    # Field 3 is fields[0]; Linux proc stat field 22 is therefore fields[19].
    if len(fields) <= 19 or not fields[19].isdigit():
        _fail("parent process start time is invalid")
    value = int(fields[19])
    if value <= 0:
        _fail("parent process start time is invalid")
    return value


def _require_parent_identity(
    parent_pid: int,
    parent_starttime_ticks: int,
    parent_uid: int,
    parent_gid: int,
) -> None:
    if os.getppid() != parent_pid:
        _fail("immediate parent PID differs from the sealed config")
    if os.getuid() != parent_uid or os.getgid() != parent_gid:
        _fail("current identity differs from the sealed parent identity")
    if _proc_starttime_ticks(parent_pid) != parent_starttime_ticks:
        _fail("parent process start time differs from the sealed config")
    parent_executable = f"/proc/{parent_pid}/exe"
    try:
        same_executable = os.path.samefile(parent_executable, PYTHON_PATH)
    except OSError as error:
        _fail(f"cannot authenticate parent executable: {error}")
    if not same_executable:
        _fail("parent executable differs from the reviewed Python")


def _parse_config(config_raw: bytes) -> dict[str, object]:
    config = _parse_canonical_object(
        config_raw,
        "sealed config",
        newline_terminated=True,
    )
    expected_fields = {
        "schema_version",
        "scope",
        "authority",
        "nonce",
        "parent_pid",
        "parent_starttime_ticks",
        "parent_uid",
        "parent_gid",
        "parent_executable",
        "core_fd",
        "core_sha256",
        "core_byte_length",
        "config_fd",
        "control_fd",
    }
    if set(config) != expected_fields:
        _fail("sealed config has an unexpected field set")
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        _fail("sealed config schema_version differs")
    if config["scope"] != CORE_SCOPE or config["authority"] != CORE_AUTHORITY:
        _fail("sealed config scope or authority differs")
    nonce = config["nonce"]
    if type(nonce) is not str or SHA256_RE.fullmatch(nonce) is None or nonce == "0" * 64:
        _fail("sealed config nonce must be a non-zero lowercase 64-hex value")
    if config["parent_executable"] != PYTHON_PATH:
        _fail("sealed config parent executable differs")
    if config["core_fd"] != CORE_FD:
        _fail("sealed config core descriptor differs")
    if config["config_fd"] != CONFIG_FD:
        _fail("sealed config descriptor differs")
    if config["control_fd"] != CONTROL_FD:
        _fail("sealed config control descriptor differs")
    _require_positive_int(config["parent_pid"], "sealed config parent_pid")
    _require_positive_int(
        config["parent_starttime_ticks"],
        "sealed config parent_starttime_ticks",
    )
    _require_nonnegative_int(config["parent_uid"], "sealed config parent_uid")
    _require_nonnegative_int(config["parent_gid"], "sealed config parent_gid")
    _require_sha256(config["core_sha256"], "sealed config core_sha256")
    core_length = _require_positive_int(
        config["core_byte_length"],
        "sealed config core_byte_length",
    )
    if core_length > MAX_CORE_BYTES:
        _fail("sealed config core_byte_length exceeds the core byte bound")
    return config


def _require_control_socket(
    parent_pid: int,
    parent_uid: int,
    parent_gid: int,
    ) -> socket.socket:
    try:
        metadata = os.fstat(CONTROL_FD)
    except OSError as error:
        _fail(f"cannot inspect control descriptor: {error}")
    if not stat.S_ISSOCK(metadata.st_mode):
        _fail("control descriptor is not a socket")
    try:
        control = socket.socket(fileno=CONTROL_FD)
        socket_type = control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
        peer = control.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
    except OSError as error:
        _fail(f"cannot authenticate control socket: {error}")
    try:
        local_name = control.getsockname()
        peer_name = control.getpeername()
        if (
            control.family != socket.AF_UNIX
            or socket_type != socket.SOCK_SEQPACKET
            or not _is_socketpair_name(local_name)
            or not _is_socketpair_name(peer_name)
        ):
            _fail(
                "control descriptor is not an unnamed AF_UNIX SOCK_SEQPACKET "
                f"pair (family={control.family!r}, type={socket_type!r}, "
                f"local={local_name!r}, peer={peer_name!r})"
            )
        credential_size = struct.calcsize("3i")
        if len(peer) != credential_size:
            _fail("control peer credential shape differs")
        peer_pid, peer_uid, peer_gid = struct.unpack("3i", peer)
        if (peer_pid, peer_uid, peer_gid) != (parent_pid, parent_uid, parent_gid):
            _fail("control peer credentials differ from the sealed parent identity")
        control.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        control.settimeout(CONTROL_TIMEOUT_SECONDS)
        return control
    except BaseException:
        control.close()
        raise


def _is_socketpair_name(value: object) -> bool:
    """Accept only unnamed or Linux SO_PASSCRED-autobound socketpair names.

    Enabling SO_PASSCRED on an unbound AF_UNIX endpoint makes Linux autobind
    that endpoint to ``NUL + five lowercase hex characters``.  It is not a
    filesystem pathname and is required to receive SCM_CREDENTIALS.  Any
    ordinary pathname, arbitrary abstract name, or malformed address remains
    outside this private socketpair contract.
    """

    return (
        value in {"", b"", None}
        or (
            type(value) is bytes
            and re.fullmatch(br"\x00[0-9a-f]{5}", value) is not None
        )
    )


def _receive_parent_message(
    control: socket.socket,
    parent_pid: int,
    parent_uid: int,
    parent_gid: int,
) -> dict[str, object]:
    credential_size = struct.calcsize("3i")
    try:
        payload, ancillary, flags, address = control.recvmsg(
            MAX_CONTROL_BYTES,
            socket.CMSG_SPACE(credential_size),
        )
    except (OSError, socket.timeout) as error:
        _fail(f"cannot receive parent control packet: {error}")
    _close_received_rights(ancillary)
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        _fail("parent control packet was truncated")
    if not _is_socketpair_name(address):
        _fail("parent control packet has a non-socketpair sender name")
    if not payload:
        _fail("parent control packet is empty")
    if len(ancillary) != 1:
        _fail("parent control packet must carry exactly one credential record")
    level, kind, credentials = ancillary[0]
    if (
        level != socket.SOL_SOCKET
        or kind != socket.SCM_CREDENTIALS
        or len(credentials) != credential_size
    ):
        _fail("parent control packet carries an unexpected ancillary record")
    credential = struct.unpack("3i", credentials)
    if credential != (parent_pid, parent_uid, parent_gid):
        _fail("parent control packet credentials differ from the sealed parent")
    return _parse_canonical_object(
        payload,
        "parent control packet",
        newline_terminated=False,
    )


def _close_received_rights(
    ancillary: list[tuple[int, int, bytes]],
) -> None:
    """Close forbidden SCM_RIGHTS descriptors before rejecting their packet."""

    descriptor_size = struct.calcsize("i")
    for level, kind, payload in ancillary:
        if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
            continue
        if len(payload) % descriptor_size:
            continue
        for descriptor in struct.unpack(f"{len(payload) // descriptor_size}i", payload):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _require_parent_packet(
    packet: dict[str, object],
    expected_kind: str,
    nonce: str,
    config_sha256: str,
) -> None:
    if set(packet) != {"schema_version", "kind", "nonce", "config_sha256"}:
        _fail("parent control packet has an unexpected field set")
    if (
        packet["schema_version"] != PROTOCOL_SCHEMA_VERSION
        or packet["kind"] != expected_kind
        or packet["nonce"] != nonce
        or packet["config_sha256"] != config_sha256
    ):
        _fail("parent control packet does not match the nonce-bound protocol")


def _send_packet(control: socket.socket, packet: dict[str, object]) -> None:
    payload = _canonical_json_bytes(packet)
    if len(payload) <= 0 or len(payload) > MAX_CONTROL_BYTES:
        _fail("child control packet exceeds the fixed byte bound")
    try:
        sent = control.send(payload)
    except OSError as error:
        _fail(f"cannot send child control packet: {error}")
    if sent != len(payload):
        _fail("child control packet was not sent as one complete record")


def _set_parent_death_signal(parent_pid: int) -> None:
    try:
        # SIG_IGN and a blocked signal mask survive execve. Restore an
        # unblocked SIG_DFL before asking the kernel to send this child SIGTERM
        # when its immediate parent disappears.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (AttributeError, OSError, ValueError) as error:
        _fail(f"cannot configure parent-death signal: {error}")
    if result != 0:
        _fail(f"cannot configure parent-death signal: errno={ctypes.get_errno()}")
    if os.getppid() != parent_pid:
        _fail("parent exited while the child configured its death signal")


def _guarantees() -> dict[str, bool]:
    return {
        "gpu_lock_acquired": False,
        "gpu_queried": False,
        "docker_invoked": False,
        "evidence_created": False,
        "semantic_replay_run": False,
        "receipt_published": False,
        "qualification_decided": False,
    }


def main(arguments: list[str]) -> int:
    try:
        # This must remain before every descriptor/control operation so a mutable
        # checkout path is never a meaningful core invocation.
        _require_sealed_descriptor_invocation(arguments)
        _require_isolated_reviewed_python()
        _require_descriptor_allowlist()
        _require_distinct_core_and_config_descriptors()
        _mark_internal_descriptors_close_on_exec()
        core_raw, core_sha256 = _read_sealed_memfd(
            CORE_FD,
            "sealed core",
            maximum_bytes=MAX_CORE_BYTES,
        )
        config_raw, config_sha256 = _read_sealed_memfd(
            CONFIG_FD,
            "sealed config",
            maximum_bytes=MAX_CONFIG_BYTES,
        )
        config = _parse_config(config_raw)
        _require_sha256(config_sha256, "sealed config digest")
        if config["core_sha256"] != core_sha256:
            _fail("sealed core digest differs from the sealed config")
        if config["core_byte_length"] != len(core_raw):
            _fail("sealed core byte length differs from the sealed config")
        parent_pid = _require_positive_int(config["parent_pid"], "parent_pid")
        parent_starttime_ticks = _require_positive_int(
            config["parent_starttime_ticks"],
            "parent_starttime_ticks",
        )
        parent_uid = _require_nonnegative_int(config["parent_uid"], "parent_uid")
        parent_gid = _require_nonnegative_int(config["parent_gid"], "parent_gid")
        nonce = config["nonce"]
        assert type(nonce) is str
        _require_parent_identity(
            parent_pid,
            parent_starttime_ticks,
            parent_uid,
            parent_gid,
        )
        _set_parent_death_signal(parent_pid)
        control = _require_control_socket(parent_pid, parent_uid, parent_gid)
        try:
            init_packet = _receive_parent_message(
                control,
                parent_pid,
                parent_uid,
                parent_gid,
            )
            _require_parent_packet(init_packet, "init", nonce, config_sha256)
            _send_packet(
                control,
                {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "kind": "ready",
                    "nonce": nonce,
                    "core_sha256": core_sha256,
                    "config_sha256": config_sha256,
                },
            )
            run_packet = _receive_parent_message(
                control,
                parent_pid,
                parent_uid,
                parent_gid,
            )
            _require_parent_packet(
                run_packet,
                "run_no_action",
                nonce,
                config_sha256,
            )
            _require_parent_identity(
                parent_pid,
                parent_starttime_ticks,
                parent_uid,
                parent_gid,
            )
            _send_packet(
                control,
                {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "kind": "complete",
                    "nonce": nonce,
                    "core_sha256": core_sha256,
                    "config_sha256": config_sha256,
                    "guarantees": _guarantees(),
                },
            )
        finally:
            control.close()
        return 0
    except CoreError as error:
        _die(str(error))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
