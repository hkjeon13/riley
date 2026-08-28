#!/usr/bin/env python3
"""Capture serial, non-stream C02 scenarios from an already-running server.

This is the raw-scenario producer used by the future C02-P1 host lifecycle
runner.  It deliberately does *not* launch or stop Riley, choose a GPU, call
SSH/container tooling, or issue a qualification decision.  Given a literal
loopback completion endpoint, one server PID, a source-owned C02 audit
directory, and a canonical serial scenario contract, it writes exact public
HTTP request/response bytes and binds each response ID to the corresponding
source-written generation-audit-v2 record and completion marker.

The retained v1 contract intentionally rejects ``exact-backend-fallback``.
The separate v2 contract admits exactly one max-performance-exact,
non-stream, explicit ``temperature: 1`` request and replays the source-owned
generation-audit and native fallback-event marker pairs.  Neither version
launches or stops a process, operates a GPU, or decides qualification.  The
module is self-contained because its remote wrapper invokes it with
``python -I -S``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence
from urllib.parse import urlsplit


sys.dont_write_bytecode = True

CONTRACT_VERSION = "riley.c02-raw-soak-runner-contract.v1"
CAPTURE_VERSION = "riley.c02-raw-scenario-capture.v1"
LEDGER_VERSION = "riley.c02-raw-request-ledger.v1"
AUDIT_INDEX_VERSION = "riley.c02-generation-audit-index.v1"
INCOMPLETE_MARKER_VERSION = "riley.c02-raw-scenario-capture-incomplete.v1"
FALLBACK_CONTRACT_VERSION = "riley.c02-raw-soak-runner-contract.v2"
FALLBACK_CAPTURE_VERSION = "riley.c02-raw-scenario-capture.v2"
FALLBACK_AUDIT_INDEX_VERSION = "riley.c02-generation-audit-index.v2"
FALLBACK_INCOMPLETE_MARKER_VERSION = "riley.c02-raw-scenario-capture-incomplete.v2"
AUDIT_VERSION = "riley.c02-generation-audit.v2"
AUDIT_COMPLETION_VERSION = "riley.c02-generation-audit-completion.v2"
FALLBACK_EVENT_VERSION = "riley.c02-native-fallback-event.v1"
FALLBACK_EVENT_COMPLETION_VERSION = "riley.c02-native-fallback-event-completion.v1"
FALLBACK_SCENARIO_ID = "exact-backend-fallback"
INCOMPLETE_MARKER_NAME = "capture-incomplete.json"

MAX_CONTRACT_BYTES = 1024 * 1024
MAX_HTTP_BODY_BYTES = 16 * 1024 * 1024
MAX_HTTP_HEAD_BYTES = 64 * 1024
MAX_AUDIT_BYTES = 16 * 1024 * 1024
MAX_NATIVE_FALLBACK_EVENTS = 65_536
MAX_PROC_BYTES = 1024 * 1024
MAX_SERVER_FDS = 16384
MAX_SCENARIOS = 32
HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_AUDIT_WAIT_SECONDS = 30.0

CANDIDATE_RE = re.compile(r"^riley-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)-rc[1-9][0-9]*$")
PID_RE = re.compile(r"^[1-9][0-9]*$")
REQUEST_ID_RE = re.compile(r"^cmpl-[A-Za-z0-9_-]{1,123}$")
SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
SAFE_LEAF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOCKET_LINK_RE = re.compile(r"^socket:\[([1-9][0-9]*)\]$")


class RawScenarioCaptureError(ValueError):
    """The requested raw C02 scenario capture cannot safely be published."""


def _fail(message: str) -> NoReturn:
    raise RawScenarioCaptureError(message)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite JSON number {value!r} is forbidden")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        _fail(f"cannot encode canonical evidence JSON: {error}")


def _parse_json(raw: bytes, label: str, *, maximum: int, canonical: bool) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _fail(f"{label} has an invalid byte length")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        _fail(f"{label} root must be a JSON object")
    if canonical and raw != _canonical(value):
        _fail(f"{label} must use exact canonical JSON bytes")
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{label} must contain exactly {sorted(fields)}")
    return value


def _nonempty_string(value: Any, label: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        _fail(f"{label} must be a bounded nonempty string")
    return value


def _nonnegative_integer(value: Any, label: str, *, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        _fail(f"{label} must be an integer in range")
    return value


def _finite_number(value: Any, label: str, *, lower: float, upper: float, strict_lower: bool = False) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        _fail(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number > upper or (number <= lower if strict_lower else number < lower):
        _fail(f"{label} is outside its supported range")
    return number


def _candidate_id(value: Any, label: str) -> str:
    if type(value) is not str or CANDIDATE_RE.fullmatch(value) is None:
        _fail(f"{label} must be a canonical riley release-candidate ID")
    return value


def _configuration_profile(value: Any, label: str) -> str:
    if value not in {"stable-default", "max-performance-exact"}:
        _fail(f"{label} must be stable-default or max-performance-exact")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail(f"{label} must be a non-zero lowercase SHA-256")
    return value


def _leaf_name(value: str, label: str) -> str:
    if type(value) is not str or SAFE_LEAF_RE.fullmatch(value) is None or value in {".", ".."}:
        _fail(f"{label} must be a normalized safe direct-child name")
    return value


def _relative_path(value: str, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\\" in value
        or "//" in value
        or SAFE_RELATIVE_RE.fullmatch(value) is None
    ):
        _fail(f"{label} must be normalized POSIX text")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"{label} contains a path alias")
    return value


def _descriptor(path: str, raw: bytes) -> dict[str, Any]:
    _relative_path(path, "evidence descriptor path")
    if type(raw) is not bytes or not raw:
        _fail("evidence descriptor bytes must be nonempty bytes")
    return {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(raw),
    }


def _required_open_flag(name: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        _fail(f"host does not expose required {name} safety flag")
    return value


def _open_flags(*, directory: bool, writable: bool = False) -> int:
    access = os.O_WRONLY | os.O_CREAT | os.O_EXCL if writable else os.O_RDONLY
    return (
        access
        | _required_open_flag("O_NOFOLLOW")
        | _required_open_flag("O_CLOEXEC")
        | _required_open_flag("O_NONBLOCK")
        | (_required_open_flag("O_DIRECTORY") if directory else 0)
    )


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


def _absolute_components(path: Path, label: str) -> tuple[str, ...]:
    raw = os.fspath(path)
    if not os.path.isabs(raw) or "\x00" in raw or "\\" in raw or raw.startswith("//") or raw != os.path.normpath(raw):
        _fail(f"{label} must be a normalized absolute path")
    parts = Path(raw).parts
    if not parts or parts[0] != os.path.sep or any(part in {"", ".", ".."} for part in parts[1:]):
        _fail(f"{label} contains an unsafe path component")
    return tuple(parts[1:])


def _require_directory(descriptor: int, label: str) -> os.stat_result:
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot inspect {label}: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} must be a directory")
    return metadata


def _safe_ancestor(metadata: os.stat_result, label: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} must be a directory")
    writable = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable and (not metadata.st_mode & stat.S_ISVTX or metadata.st_uid not in {0, os.geteuid()}):
        _fail(f"{label} is an unsafe writable ancestor")


def _open_absolute_directory(path: Path, label: str, *, private_root: bool) -> int:
    components = _absolute_components(path, label)
    flags = _open_flags(directory=True)
    try:
        current = os.open(os.path.sep, flags)
    except OSError as error:
        _fail(f"cannot open filesystem root for {label}: {error}")
    try:
        if private_root:
            _safe_ancestor(os.fstat(current), f"{label} ancestor /")
        for component in components:
            try:
                child = os.open(component, flags, dir_fd=current)
            except OSError as error:
                _fail(f"cannot open {label} without following links: {error}")
            os.close(current)
            current = child
            if private_root:
                _safe_ancestor(os.fstat(current), f"{label} ancestor {component!r}")
        metadata = _require_directory(current, label)
        if private_root and (metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700):
            _fail(f"{label} must be effective-UID-owned with exact mode 0700")
        return current
    except BaseException:
        _close_quietly(current)
        raise


def _read_regular_at(directory_fd: int, name: str, label: str, *, maximum: int) -> bytes:
    _leaf_name(name, f"{label} name")
    _require_directory(directory_fd, f"{label} parent")
    try:
        before = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot inspect {label}: {error}")
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        _fail(f"{label} must be a single-link regular non-link file")
    try:
        descriptor = os.open(name, _open_flags(directory=False), dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot open {label} without following links: {error}")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            _fail(f"{label} changed while opened")
        chunks: list[bytes] = []
        size = 0
        while True:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            except OSError as error:
                _fail(f"cannot read {label}: {error}")
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                _fail(f"{label} exceeds its byte bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
            _fail(f"{label} changed while read")
    finally:
        _close_quietly(descriptor)
    try:
        final = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot re-inspect {label}: {error}")
    if (before.st_dev, before.st_ino) != (final.st_dev, final.st_ino):
        _fail(f"{label} changed while read")
    return b"".join(chunks)


def _read_absolute_regular(path: Path, label: str, *, maximum: int) -> bytes:
    parts = _absolute_components(path, label)
    if not parts:
        _fail(f"{label} must name a regular file")
    parent = Path(os.path.sep).joinpath(*parts[:-1])
    descriptor = _open_absolute_directory(parent, f"{label} parent", private_root=False)
    try:
        return _read_regular_at(descriptor, parts[-1], label, maximum=maximum)
    finally:
        _close_quietly(descriptor)


def _new_private_directory(parent_fd: int, name: str, label: str) -> int:
    _leaf_name(name, f"{label} name")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        _fail(f"{label} already exists; captures are create-only")
    except OSError as error:
        _fail(f"cannot create {label}: {error}")
    try:
        descriptor = os.open(name, _open_flags(directory=True), dir_fd=parent_fd)
    except OSError as error:
        _fail(f"cannot reopen {label} without following links: {error}")
    try:
        os.fchmod(descriptor, 0o700)
        metadata = _require_directory(descriptor, label)
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_nlink != 2:
            _fail(f"{label} is not a private directory")
        _fsync(descriptor, label)
        _fsync(parent_fd, f"parent after {label}")
        return descriptor
    except BaseException:
        _close_quietly(descriptor)
        raise


def _open_private_child(parent_fd: int, name: str, label: str) -> int:
    _leaf_name(name, f"{label} name")
    try:
        descriptor = os.open(name, _open_flags(directory=True), dir_fd=parent_fd)
    except OSError as error:
        _fail(f"cannot open {label} without following links: {error}")
    try:
        metadata = _require_directory(descriptor, label)
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            _fail(f"{label} must be effective-UID-owned with exact mode 0700")
        return descriptor
    except BaseException:
        _close_quietly(descriptor)
        raise


def _write_new(directory_fd: int, name: str, raw: bytes) -> None:
    _leaf_name(name, "evidence output name")
    if type(raw) is not bytes:
        _fail(f"evidence file {name} must be bytes")
    try:
        descriptor = os.open(name, _open_flags(directory=False, writable=True), 0o600, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot create evidence file {name}: {error}")
    try:
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1:
            _fail(f"new evidence file {name} is not private and single-link")
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail(f"cannot write evidence file {name}")
            offset += written
        _fsync(descriptor, f"evidence file {name}")
    except OSError as error:
        _fail(f"cannot write evidence file {name}: {error}")
    finally:
        _close_quietly(descriptor)
    _fsync(directory_fd, f"evidence directory after {name}")


def _write_json(directory_fd: int, name: str, value: dict[str, Any]) -> bytes:
    raw = _canonical(value)
    _write_new(directory_fd, name, raw)
    return raw


def _lock(directory_fd: int) -> None:
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(f"cannot acquire exclusive capture lock: {error}")


def _remove_incomplete_marker(directory_fd: int, marker: bytes) -> None:
    try:
        os.unlink(INCOMPLETE_MARKER_NAME, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot remove incomplete capture marker: {error}")
    try:
        _fsync(directory_fd, "capture directory after completion marker removal")
    except RawScenarioCaptureError:
        # A failed directory sync after unlink leaves terminality uncertain.
        # Recreate the nonterminal marker through the held FD if at all
        # possible; a restoration failure still propagates the original
        # failure and never converts the capture into success.
        try:
            _write_new(directory_fd, INCOMPLETE_MARKER_NAME, marker)
        except RawScenarioCaptureError:
            pass
        raise


def _outside_repository(evidence_root: Path, repository_root: Path) -> None:
    try:
        root = os.path.realpath(evidence_root)
        repository = os.path.realpath(repository_root)
        if os.path.commonpath([root, repository]) == repository:
            _fail("--evidence-root must be outside the source checkout")
    except ValueError as error:
        _fail(f"cannot compare evidence and source roots: {error}")


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    url: str


def parse_endpoint(value: str) -> Endpoint:
    if type(value) is not str or len(value) > 512 or "\x00" in value:
        _fail("--endpoint must be bounded text")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/completions"
    ):
        _fail("--endpoint must be literal http://127.0.0.1:PORT/v1/completions")
    try:
        port = parsed.port
    except ValueError as error:
        _fail(f"--endpoint port is invalid: {error}")
    if port is None or not 1024 <= port <= 65535:
        _fail("--endpoint must use an unprivileged TCP port")
    canonical = f"http://127.0.0.1:{port}/v1/completions"
    if value != canonical:
        _fail("--endpoint must use canonical literal loopback form")
    return Endpoint(host="127.0.0.1", port=port, url=canonical)


def _server_stat(pid: int) -> tuple[bytes, int]:
    if type(pid) is not int or pid <= 0:
        _fail("--server-pid must be a positive integer")
    raw = _read_absolute_regular(Path(f"/proc/{pid}/stat"), "server /proc stat", maximum=MAX_PROC_BYTES)
    try:
        text = raw.decode("ascii")
        first_space = text.find(" ")
        close = text.rfind(")")
        fields = text[close + 2 :].split()
        # /proc/<pid>/stat field 22 is zero-based tail item 19 after comm.
        start_ticks = int(fields[19])
    except (UnicodeDecodeError, ValueError, IndexError) as error:
        _fail(f"server /proc stat is malformed: {error}")
    if first_space < 1 or close < first_space + 2 or text[:first_space] != str(pid) or start_ticks <= 0:
        _fail("server /proc stat has an invalid start tick")
    return raw, start_ticks


@dataclass(frozen=True)
class Listener:
    tcp: bytes
    inode: int
    sockets: tuple[int, ...]


def _capture_listener(port: int) -> tuple[bytes, int]:
    raw = _read_absolute_regular(Path("/proc/net/tcp"), "/proc/net/tcp", maximum=MAX_PROC_BYTES)
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail(f"/proc/net/tcp is not ASCII: {error}")
    if not lines or not lines[0].lstrip().startswith("sl"):
        _fail("/proc/net/tcp has no valid header")
    expected = f"0100007F:{port:04X}"
    inodes: set[int] = set()
    for line in lines[1:]:
        cells = line.split()
        if not cells:
            continue
        if len(cells) < 10:
            _fail("/proc/net/tcp has a malformed socket row")
        local, remote, state, inode = cells[1], cells[2], cells[3], cells[9]
        if local.upper() == f"00000000:{port:04X}" and remote.upper() == "00000000:0000" and state.upper() == "0A":
            _fail("completion endpoint must not have an IPv4 wildcard listener")
        if local.upper() == expected and remote.upper() == "00000000:0000" and state.upper() == "0A":
            if not re.fullmatch(r"[1-9][0-9]*", inode) or int(inode) in inodes:
                _fail("/proc/net/tcp has an invalid loopback listener inode")
            inodes.add(int(inode))
    if len(inodes) != 1:
        _fail("completion endpoint must have exactly one IPv4 loopback listener")
    return raw, next(iter(inodes))


def _server_socket_inodes(pid: int) -> tuple[int, ...]:
    directory = _open_absolute_directory(Path(f"/proc/{pid}/fd"), "server /proc fd directory", private_root=False)
    try:
        names = os.listdir(directory)
        if len(names) > MAX_SERVER_FDS:
            _fail("server /proc fd directory exceeds its reviewed entry bound")
        inodes: set[int] = set()
        for name in names:
            if re.fullmatch(r"[0-9]+", name) is None:
                _fail("server /proc fd directory has a non-numeric entry")
            try:
                target = os.readlink(name, dir_fd=directory)
            except OSError as error:
                _fail(f"server /proc fd directory changed while sampled: {error}")
            matched = SOCKET_LINK_RE.fullmatch(target)
            if matched is not None:
                inodes.add(int(matched.group(1)))
        return tuple(sorted(inodes))
    finally:
        _close_quietly(directory)


def _bound_listener(endpoint: Endpoint, pid: int) -> Listener:
    tcp, inode = _capture_listener(endpoint.port)
    sockets = _server_socket_inodes(pid)
    if inode not in sockets:
        _fail("completion endpoint listener is not held by the requested server PID")
    return Listener(tcp=tcp, inode=inode, sockets=sockets)


def _socket_snapshot(pid: int, sockets: tuple[int, ...]) -> bytes:
    if not sockets:
        _fail("server PID has no observed socket inodes")
    return _canonical({
        "schema_version": "riley.c02-proc-fd-socket-snapshot.v2",
        "server_pid": pid,
        "socket_inodes": list(sockets),
    })


def _completion_request(value: Any, label: str) -> dict[str, Any]:
    row = _exact(
        value,
        {"model", "prompt", "max_tokens", "temperature", "top_p", "seed", "stream"},
        label,
    )
    _nonempty_string(row["model"], f"{label}.model", maximum=256)
    _nonempty_string(row["prompt"], f"{label}.prompt", maximum=1024 * 1024)
    _nonnegative_integer(row["max_tokens"], f"{label}.max_tokens", maximum=65536)
    if row["max_tokens"] == 0:
        _fail(f"{label}.max_tokens must be positive")
    _finite_number(row["temperature"], f"{label}.temperature", lower=0.0, upper=2.0)
    _finite_number(row["top_p"], f"{label}.top_p", lower=0.0, upper=1.0, strict_lower=True)
    _nonnegative_integer(row["seed"], f"{label}.seed", maximum=(1 << 64) - 1)
    if row["stream"] is not False:
        _fail(f"{label}.stream must be false in the serial raw-scenario v1 contract")
    raw = _canonical(row)
    if len(raw) > MAX_HTTP_BODY_BYTES:
        _fail(f"{label} exceeds the HTTP byte bound")
    return row


def _fallback_completion_request(value: Any, label: str) -> dict[str, Any]:
    """Close the first public fallback arm to one f32-stable cause.

    The current public completion grammar cannot set a repetition penalty,
    minimum output tokens, or token-ID stop mask.  Pinning temperature to one
    is the only public byte sequence this version admits: it both establishes
    the reviewed native fallback reason and cannot round to zero in Rust's
    f32 request decoder.
    """

    row = _completion_request(value, label)
    if row["max_tokens"] != 1:
        _fail(f"{label}.max_tokens must be exactly 1 in the native fallback v2 contract")
    if row["temperature"] != 1.0:
        _fail(f"{label}.temperature must be exactly 1.0 in the native fallback v2 contract")
    if row["top_p"] != 1.0:
        _fail(f"{label}.top_p must be exactly 1.0 in the native fallback v2 contract")
    return row


def validate_contract(raw: bytes, *, candidate_id: str, configuration_profile: str) -> dict[str, Any]:
    row = _exact(
        _parse_json(raw, "scenario contract", maximum=MAX_CONTRACT_BYTES, canonical=True),
        {"schema_version", "candidate_id", "configuration_profile", "scenarios"},
        "scenario contract",
    )
    version = row["schema_version"]
    if version not in {CONTRACT_VERSION, FALLBACK_CONTRACT_VERSION}:
        _fail(
            f"scenario contract must use {CONTRACT_VERSION} or {FALLBACK_CONTRACT_VERSION}"
        )
    if _candidate_id(row["candidate_id"], "scenario contract.candidate_id") != candidate_id:
        _fail("scenario contract candidate_id differs from --candidate-id")
    if _configuration_profile(row["configuration_profile"], "scenario contract.configuration_profile") != configuration_profile:
        _fail("scenario contract configuration_profile differs from --configuration-profile")
    scenarios = row["scenarios"]
    if not isinstance(scenarios, list):
        _fail("scenario contract.scenarios must be an array")
    if version == FALLBACK_CONTRACT_VERSION:
        if configuration_profile != "max-performance-exact":
            _fail("native fallback v2 capture requires max-performance-exact")
        if len(scenarios) != 1:
            _fail("native fallback v2 contract must contain exactly one scenario")
    elif not 1 <= len(scenarios) <= MAX_SCENARIOS:
        _fail("scenario contract.scenarios must be a bounded nonempty array")
    seen: set[str] = set()
    for index, item in enumerate(scenarios):
        scenario = _exact(item, {"scenario_id", "completion_request"}, f"scenario contract.scenarios[{index}]")
        scenario_id = scenario["scenario_id"]
        if type(scenario_id) is not str or SCENARIO_ID_RE.fullmatch(scenario_id) is None or scenario_id in seen:
            _fail(f"scenario contract.scenarios[{index}].scenario_id must be a unique normalized identifier")
        if version == CONTRACT_VERSION and scenario_id == FALLBACK_SCENARIO_ID:
            _fail("exact-backend-fallback requires versioned native fallback capture/binder replay and is deferred")
        if version == FALLBACK_CONTRACT_VERSION and scenario_id != FALLBACK_SCENARIO_ID:
            _fail("native fallback v2 contract permits only exact-backend-fallback")
        seen.add(scenario_id)
        request_label = f"scenario contract.scenarios[{index}].completion_request"
        if version == FALLBACK_CONTRACT_VERSION:
            _fallback_completion_request(scenario["completion_request"], request_label)
        else:
            _completion_request(scenario["completion_request"], request_label)
    return row


def _request_bytes(endpoint: Endpoint, body: bytes) -> bytes:
    return (
        f"POST /v1/completions HTTP/1.1\r\nHost: {endpoint.host}:{endpoint.port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body


def _split_head(buffer: bytes) -> tuple[bytes, bytes] | None:
    offset = buffer.find(b"\r\n\r\n")
    if offset < 0:
        return None
    return buffer[: offset + 4], buffer[offset + 4 :]


def _response_length(head: bytes) -> int:
    if not head.endswith(b"\r\n\r\n") or len(head) > MAX_HTTP_HEAD_BYTES:
        _fail("completion response head is malformed or oversized")
    try:
        lines = head[:-4].decode("ascii").split("\r\n")
    except UnicodeDecodeError as error:
        _fail(f"completion response head is not ASCII: {error}")
    if not lines or lines[0] != "HTTP/1.1 200 OK":
        _fail("completion endpoint did not return HTTP 200")
    values: dict[str, list[str]] = {}
    for line in lines[1:]:
        if not line or ":" not in line or line[:1] in {" ", "\t"}:
            _fail("completion response head contains an invalid header")
        name, value = line.split(":", 1)
        normalized = name.lower()
        if re.fullmatch(r"[A-Za-z0-9-]+", name) is None or normalized in values:
            _fail("completion response head repeats or has an invalid header")
        if any(ord(character) < 32 and character != "\t" or ord(character) == 127 for character in value):
            _fail("completion response head has a control character in a header value")
        # HTTP OWS is SP / HTAB only.  Python's generic ``strip`` would also
        # normalize VT, FF, and other invalid control bytes into a seemingly
        # valid Content-Length or media type.
        values[normalized] = [value.strip(" \t")]
    if "transfer-encoding" in values:
        _fail("completion response must not use Transfer-Encoding")
    length_values = values.get("content-length")
    if length_values is None or len(length_values) != 1 or not re.fullmatch(r"[0-9]+", length_values[0]):
        _fail("completion response must have one numeric Content-Length")
    length = int(length_values[0])
    if not 1 <= length <= MAX_HTTP_BODY_BYTES:
        _fail("completion response body is outside its byte bound")
    content_type = values.get("content-type")
    if content_type is None or len(content_type) != 1 or content_type[0].split(";", 1)[0].strip().lower() != "application/json":
        _fail("completion response must have an application/json Content-Type")
    return length


def _capture_completion(endpoint: Endpoint, body: bytes) -> tuple[bytes, bytes, bytes]:
    request = _request_bytes(endpoint, body)
    try:
        connection = socket.create_connection((endpoint.host, endpoint.port), timeout=HTTP_TIMEOUT_SECONDS)
    except OSError as error:
        _fail(f"cannot connect to completion endpoint: {error}")
    with connection:
        try:
            connection.settimeout(HTTP_TIMEOUT_SECONDS)
            connection.sendall(request)
            buffered = bytearray()
            while len(buffered) <= MAX_HTTP_HEAD_BYTES:
                part = connection.recv(min(65536, MAX_HTTP_HEAD_BYTES + 1 - len(buffered)))
                if not part:
                    _fail("completion response ended before its header")
                buffered.extend(part)
                split = _split_head(bytes(buffered))
                if split is not None:
                    head, remainder = split
                    break
            else:
                _fail("completion response head exceeds its byte bound")
            expected = _response_length(head)
            response = bytearray(remainder)
            if len(response) > expected:
                _fail("completion response contains trailing body bytes")
            while len(response) < expected:
                part = connection.recv(min(65536, expected - len(response)))
                if not part:
                    _fail("completion response body is truncated")
                response.extend(part)
            # An independent read detects a body byte that arrived separately
            # from the final length-bounded recv.
            extra = connection.recv(1)
            if extra:
                _fail("completion response contains trailing body bytes")
        except (OSError, RawScenarioCaptureError) as error:
            if isinstance(error, RawScenarioCaptureError):
                raise
            _fail(f"cannot capture completion response: {error}")
    return request, head, bytes(response)


def _response_id(raw: bytes) -> str:
    row = _parse_json(raw, "completion response body", maximum=MAX_HTTP_BODY_BYTES, canonical=False)
    request_id = row.get("id")
    if type(request_id) is not str or REQUEST_ID_RE.fullmatch(request_id) is None:
        _fail("completion response has no valid source-issued ID")
    return request_id


def _audit_record(
    raw: bytes,
    *,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    pid: int,
    start_ticks: int,
    request_id: str,
) -> dict[str, Any]:
    row = _parse_json(raw, "source generation audit", maximum=MAX_AUDIT_BYTES, canonical=True)
    required = {
        "schema_version", "candidate_id", "runtime_identity", "process_identity", "server_request_id",
        "delivery_mode", "prompt_token_ids", "committed_output_tokens", "sampling_selections", "finish_reason", "usage",
    }
    if set(row) != required or row["schema_version"] != AUDIT_VERSION:
        _fail("source generation audit does not use the exact v2 shape")
    if row["candidate_id"] != candidate_id or row["server_request_id"] != request_id or row["delivery_mode"] != "non-stream":
        _fail("source generation audit does not match the captured completion")
    identity = _exact(row["runtime_identity"], {"configuration_profile", "configuration_sha256"}, "source generation audit.runtime_identity")
    if (
        identity["configuration_profile"] != configuration_profile
        or identity["configuration_sha256"] != configuration_sha256
    ):
        _fail("source generation audit runtime identity differs from the bound configuration")
    audit_pid, audit_start_ticks = _source_process_identity(
        row["process_identity"], "source generation audit.process_identity"
    )
    if audit_pid != pid or audit_start_ticks != start_ticks:
        _fail("source generation audit process identity differs from the live server")
    return row


def _audit_marker(raw: bytes, *, record_name: str, record_raw: bytes) -> None:
    row = _exact(
        _parse_json(raw, "source generation audit completion marker", maximum=MAX_AUDIT_BYTES, canonical=True),
        {"schema_version", "artifact_filename", "artifact_sha256"},
        "source generation audit completion marker",
    )
    if row["schema_version"] != AUDIT_COMPLETION_VERSION or row["artifact_filename"] != record_name:
        _fail("source generation audit completion marker does not bind the record name")
    if row["artifact_sha256"] != hashlib.sha256(record_raw).hexdigest():
        _fail("source generation audit completion marker hash does not bind the record bytes")


def _source_process_identity(value: Any, label: str) -> tuple[int, int]:
    process = _exact(value, {"pid", "start_ticks"}, label)
    pid = process["pid"]
    start_ticks = process["start_ticks"]
    if type(pid) is not int or not 1 <= pid <= (1 << 32) - 1:
        _fail(f"{label}.pid must be a positive u32 integer")
    if type(start_ticks) is not int or start_ticks < 1:
        _fail(f"{label}.start_ticks must be a positive integer")
    return pid, start_ticks


def _fallback_event(
    raw: bytes,
    *,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    pid: int,
    start_ticks: int,
    request_id: str,
    audit_name: str,
    audit_raw: bytes,
    audit: dict[str, Any],
) -> None:
    """Replay the closed public nonzero-temperature fallback source leaf."""

    if configuration_profile != "max-performance-exact":
        _fail("source native fallback event requires max-performance-exact")
    row = _exact(
        _parse_json(raw, "source native fallback event", maximum=MAX_AUDIT_BYTES, canonical=True),
        {
            "schema_version",
            "candidate_id",
            "runtime_identity",
            "process_identity",
            "server_request_id",
            "generation_audit",
            "fallback_selections",
        },
        "source native fallback event",
    )
    if row["schema_version"] != FALLBACK_EVENT_VERSION:
        _fail("source native fallback event has an unsupported schema version")
    if row["candidate_id"] != candidate_id or row["server_request_id"] != request_id:
        _fail("source native fallback event does not match the captured completion")
    identity = _exact(
        row["runtime_identity"],
        {"configuration_profile", "configuration_sha256"},
        "source native fallback event.runtime_identity",
    )
    if (
        identity["configuration_profile"] != configuration_profile
        or identity["configuration_sha256"] != configuration_sha256
    ):
        _fail("source native fallback event runtime identity differs from the bound configuration")
    event_pid, event_start_ticks = _source_process_identity(
        row["process_identity"], "source native fallback event.process_identity"
    )
    if event_pid != pid or event_start_ticks != start_ticks:
        _fail("source native fallback event process identity differs from the live server")
    generation_audit = _exact(
        row["generation_audit"],
        {"artifact_filename", "artifact_sha256"},
        "source native fallback event.generation_audit",
    )
    if (
        generation_audit["artifact_filename"] != audit_name
        or generation_audit["artifact_sha256"] != hashlib.sha256(audit_raw).hexdigest()
    ):
        _fail("source native fallback event does not bind the exact generation audit")
    events = row["fallback_selections"]
    selections = audit["sampling_selections"]
    if not isinstance(events, list) or not 1 <= len(events) <= MAX_NATIVE_FALLBACK_EVENTS:
        _fail("source native fallback selections are outside the published event bound")
    if events != selections:
        _fail("source native fallback selections do not exactly replay the generation audit")
    for index, selection in enumerate(events):
        item = _exact(
            selection,
            {
                "iteration_id",
                "configured_backend",
                "selected_backend",
                "ineligibility_reason",
                "committed",
            },
            f"source native fallback event.fallback_selections[{index}]",
        )
        if (
            type(item["iteration_id"]) is not int
            or item["iteration_id"] < 1
            or item["configured_backend"] != "gpu-greedy"
            or item["selected_backend"] != "cpu-normative"
            or item["ineligibility_reason"] != "nonzero-temperature"
            or item["committed"] is not True
        ):
            _fail("source native fallback event is not the reviewed nonzero-temperature transition")


def _fallback_marker(raw: bytes, *, event_name: str, event_raw: bytes) -> None:
    row = _exact(
        _parse_json(
            raw,
            "source native fallback completion marker",
            maximum=MAX_AUDIT_BYTES,
            canonical=True,
        ),
        {"schema_version", "artifact_filename", "artifact_sha256"},
        "source native fallback completion marker",
    )
    if row["schema_version"] != FALLBACK_EVENT_COMPLETION_VERSION or row["artifact_filename"] != event_name:
        _fail("source native fallback completion marker does not bind the event name")
    if row["artifact_sha256"] != hashlib.sha256(event_raw).hexdigest():
        _fail("source native fallback completion marker hash does not bind the event bytes")


def _wait_for_audit(
    audit_fd: int,
    *,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    pid: int,
    start_ticks: int,
    request_id: str,
    wait_seconds: float,
) -> tuple[str, bytes, str, bytes]:
    record_name = f"{request_id}.json"
    marker_name = f"{record_name}.complete"
    deadline = time.monotonic() + wait_seconds
    last_error: RawScenarioCaptureError | None = None
    while True:
        try:
            record = _read_regular_at(audit_fd, record_name, "source generation audit", maximum=MAX_AUDIT_BYTES)
            marker = _read_regular_at(audit_fd, marker_name, "source generation audit completion marker", maximum=MAX_AUDIT_BYTES)
            _audit_record(
                record,
                candidate_id=candidate_id,
                configuration_profile=configuration_profile,
                configuration_sha256=configuration_sha256,
                pid=pid,
                start_ticks=start_ticks,
                request_id=request_id,
            )
            _audit_marker(marker, record_name=record_name, record_raw=record)
            return record_name, record, marker_name, marker
        except RawScenarioCaptureError as error:
            last_error = error
        if time.monotonic() >= deadline:
            detail = str(last_error) if last_error is not None else "source audit did not appear"
            _fail(f"timed out waiting for completed source generation audit: {detail}")
        time.sleep(0.05)


def _wait_for_native_fallback(
    audit_fd: int,
    *,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    pid: int,
    start_ticks: int,
    request_id: str,
    audit_name: str,
    audit_raw: bytes,
    wait_seconds: float,
) -> tuple[str, bytes, str, bytes]:
    """Wait for the source-owned fallback pair derived from one response ID."""

    audit = _audit_record(
        audit_raw,
        candidate_id=candidate_id,
        configuration_profile=configuration_profile,
        configuration_sha256=configuration_sha256,
        pid=pid,
        start_ticks=start_ticks,
        request_id=request_id,
    )
    event_name = f"{request_id}.fallback.json"
    marker_name = f"{event_name}.complete"
    deadline = time.monotonic() + wait_seconds
    last_error: RawScenarioCaptureError | None = None
    while True:
        try:
            event = _read_regular_at(
                audit_fd,
                event_name,
                "source native fallback event",
                maximum=MAX_AUDIT_BYTES,
            )
            marker = _read_regular_at(
                audit_fd,
                marker_name,
                "source native fallback completion marker",
                maximum=MAX_AUDIT_BYTES,
            )
            _fallback_event(
                event,
                candidate_id=candidate_id,
                configuration_profile=configuration_profile,
                configuration_sha256=configuration_sha256,
                pid=pid,
                start_ticks=start_ticks,
                request_id=request_id,
                audit_name=audit_name,
                audit_raw=audit_raw,
                audit=audit,
            )
            _fallback_marker(marker, event_name=event_name, event_raw=event)
            return event_name, event, marker_name, marker
        except RawScenarioCaptureError as error:
            last_error = error
        if time.monotonic() >= deadline:
            detail = str(last_error) if last_error is not None else "source fallback event did not appear"
            _fail(f"timed out waiting for completed source native fallback event: {detail}")
        time.sleep(0.05)


@dataclass(frozen=True)
class CaptureRequest:
    endpoint: Endpoint
    server_pid: int
    candidate_id: str
    configuration_profile: str
    configuration_sha256: str
    evidence_root: Path
    capture_name: str
    audit_dir_name: str
    scenario_contract: Path
    audit_wait_seconds: float


def capture_raw_scenarios(request: CaptureRequest, *, repository_root: Path) -> dict[str, Any]:
    _outside_repository(request.evidence_root, repository_root)
    capture_name = _leaf_name(request.capture_name, "--capture-name")
    audit_dir_name = _leaf_name(request.audit_dir_name, "--audit-dir-name")
    candidate_id = _candidate_id(request.candidate_id, "--candidate-id")
    profile = _configuration_profile(request.configuration_profile, "--configuration-profile")
    configuration_sha256 = _sha256(request.configuration_sha256, "--configuration-sha256")
    if type(request.audit_wait_seconds) not in {int, float} or not 0.1 <= float(request.audit_wait_seconds) <= 300.0:
        _fail("--audit-wait-seconds must be between 0.1 and 300")
    contract_raw = _read_absolute_regular(request.scenario_contract, "--scenario-contract", maximum=MAX_CONTRACT_BYTES)
    contract = validate_contract(contract_raw, candidate_id=candidate_id, configuration_profile=profile)
    native_fallback = contract["schema_version"] == FALLBACK_CONTRACT_VERSION
    capture_version = FALLBACK_CAPTURE_VERSION if native_fallback else CAPTURE_VERSION
    index_version = FALLBACK_AUDIT_INDEX_VERSION if native_fallback else AUDIT_INDEX_VERSION
    incomplete_marker_version = (
        FALLBACK_INCOMPLETE_MARKER_VERSION if native_fallback else INCOMPLETE_MARKER_VERSION
    )
    root_fd = _open_absolute_directory(request.evidence_root, "--evidence-root", private_root=True)
    audit_fd: int | None = None
    capture_fd: int | None = None
    raw_fd: int | None = None
    try:
        audit_fd = _open_private_child(root_fd, audit_dir_name, "source audit directory")
        capture_fd = _new_private_directory(root_fd, capture_name, "scenario capture directory")
        _lock(capture_fd)
        marker = _canonical({
            "schema_version": incomplete_marker_version,
            "capture_name": capture_name,
        })
        _write_new(capture_fd, INCOMPLETE_MARKER_NAME, marker)
        raw_fd = _new_private_directory(capture_fd, "raw", "scenario raw directory")
        contract_path = f"{capture_name}/raw/scenario-contract.json"
        _write_new(raw_fd, "scenario-contract.json", contract_raw)
        contract_descriptor = _descriptor(contract_path, contract_raw)
        scenario_rows: list[dict[str, Any]] = []
        seen_request_ids: set[str] = set()
        expected_target: dict[str, Any] | None = None
        for sequence, scenario in enumerate(contract["scenarios"]):
            scenario_id = scenario["scenario_id"]
            body = _canonical(scenario["completion_request"])
            prefix = f"{sequence:06d}"
            pre_stat, start_ticks = _server_stat(request.server_pid)
            pre_listener = _bound_listener(request.endpoint, request.server_pid)
            pre_sockets = _socket_snapshot(request.server_pid, pre_listener.sockets)
            target = {
                "server_pid": request.server_pid,
                "server_start_ticks": start_ticks,
                "listener_port": request.endpoint.port,
                "listener_inode": pre_listener.inode,
            }
            if expected_target is None:
                expected_target = target
            elif target != expected_target:
                _fail("scenario target tuple drifted before the public completion request")
            pre_stat_name = f"{prefix}.pre.proc-stat"
            pre_tcp_name = f"{prefix}.pre.proc-net-tcp"
            pre_sockets_name = f"{prefix}.pre.proc-fd-sockets.json"
            _write_new(raw_fd, pre_stat_name, pre_stat)
            _write_new(raw_fd, pre_tcp_name, pre_listener.tcp)
            _write_new(raw_fd, pre_sockets_name, pre_sockets)
            sent, head, response = _capture_completion(request.endpoint, body)
            request_name = f"{prefix}.request.http"
            head_name = f"{prefix}.response-head.http"
            response_name = f"{prefix}.response-body.json"
            # Persist the exact, framing-validated transport bytes before
            # interpreting even the public ID.  A malformed or duplicate
            # response body therefore remains behind the incomplete marker;
            # malformed transport framing itself never becomes a captured
            # scenario because no bounded response exists to publish.
            _write_new(raw_fd, request_name, sent)
            _write_new(raw_fd, head_name, head)
            _write_new(raw_fd, response_name, response)
            post_stat, post_ticks = _server_stat(request.server_pid)
            post_listener = _bound_listener(request.endpoint, request.server_pid)
            post_sockets = _socket_snapshot(request.server_pid, post_listener.sockets)
            if post_ticks != start_ticks or post_listener.inode != pre_listener.inode:
                _fail("server/listener target tuple drifted during the public completion request")
            post_stat_name = f"{prefix}.post.proc-stat"
            post_tcp_name = f"{prefix}.post.proc-net-tcp"
            post_sockets_name = f"{prefix}.post.proc-fd-sockets.json"
            _write_new(raw_fd, post_stat_name, post_stat)
            _write_new(raw_fd, post_tcp_name, post_listener.tcp)
            _write_new(raw_fd, post_sockets_name, post_sockets)
            request_id = _response_id(response)
            if request_id in seen_request_ids:
                _fail("completion endpoint reused a source-issued request ID across scenarios")
            seen_request_ids.add(request_id)
            record_name, record_raw, marker_name, marker_raw = _wait_for_audit(
                audit_fd,
                candidate_id=candidate_id,
                configuration_profile=profile,
                configuration_sha256=configuration_sha256,
                pid=request.server_pid,
                start_ticks=start_ticks,
                request_id=request_id,
                wait_seconds=float(request.audit_wait_seconds),
            )
            fallback_name: str | None = None
            fallback_raw: bytes | None = None
            fallback_marker_name: str | None = None
            fallback_marker_raw: bytes | None = None
            if native_fallback:
                (
                    fallback_name,
                    fallback_raw,
                    fallback_marker_name,
                    fallback_marker_raw,
                ) = _wait_for_native_fallback(
                    audit_fd,
                    candidate_id=candidate_id,
                    configuration_profile=profile,
                    configuration_sha256=configuration_sha256,
                    pid=request.server_pid,
                    start_ticks=start_ticks,
                    request_id=request_id,
                    audit_name=record_name,
                    audit_raw=record_raw,
                    wait_seconds=float(request.audit_wait_seconds),
                )
            # The source audit can arrive after the HTTP completion.  Recheck
            # the held PID/listener tuple after its marker is visible so a
            # restart or PID reuse during that wait leaves this capture
            # nonterminal rather than binding old audit bytes to a new server.
            final_stat, final_ticks = _server_stat(request.server_pid)
            final_listener = _bound_listener(request.endpoint, request.server_pid)
            final_sockets = _socket_snapshot(request.server_pid, final_listener.sockets)
            if final_ticks != start_ticks or final_listener.inode != pre_listener.inode:
                _fail("server/listener target tuple drifted while waiting for the source audit")
            final_stat_name = f"{prefix}.final.proc-stat"
            final_tcp_name = f"{prefix}.final.proc-net-tcp"
            final_sockets_name = f"{prefix}.final.proc-fd-sockets.json"
            _write_new(raw_fd, final_stat_name, final_stat)
            _write_new(raw_fd, final_tcp_name, final_listener.tcp)
            _write_new(raw_fd, final_sockets_name, final_sockets)
            request_path = f"{capture_name}/raw/{request_name}"
            head_path = f"{capture_name}/raw/{head_name}"
            response_path = f"{capture_name}/raw/{response_name}"
            audit_path = f"{audit_dir_name}/{record_name}"
            audit_marker_path = f"{audit_dir_name}/{marker_name}"
            ledger_name = f"{prefix}-{scenario_id}.request-ledger.json"
            index_name = f"{prefix}-{scenario_id}.generation-audit-index.json"
            ledger = {
                "schema_version": LEDGER_VERSION,
                "scenario_id": scenario_id,
                "delivery_mode": "non-stream",
                "server_request_id": request_id,
                "request": _descriptor(request_path, sent),
                "response_head": _descriptor(head_path, head),
                "response_body": _descriptor(response_path, response),
            }
            ledger_raw = _write_json(capture_fd, ledger_name, ledger)
            index = {
                "schema_version": index_version,
                "scenario_id": scenario_id,
                "server_request_id": request_id,
                "audit_record": _descriptor(audit_path, record_raw),
                "audit_completion_marker": _descriptor(audit_marker_path, marker_raw),
            }
            if native_fallback:
                if (
                    fallback_name is None
                    or fallback_raw is None
                    or fallback_marker_name is None
                    or fallback_marker_raw is None
                ):
                    _fail("native fallback source pair was not captured")
                index["fallback_event"] = _descriptor(
                    f"{audit_dir_name}/{fallback_name}", fallback_raw
                )
                index["fallback_completion_marker"] = _descriptor(
                    f"{audit_dir_name}/{fallback_marker_name}", fallback_marker_raw
                )
            index_raw = _write_json(capture_fd, index_name, index)
            scenario_row = {
                "scenario_id": scenario_id,
                "target": target,
                "process": {
                    "pre_stat": _descriptor(f"{capture_name}/raw/{pre_stat_name}", pre_stat),
                    "post_stat": _descriptor(f"{capture_name}/raw/{post_stat_name}", post_stat),
                    "final_stat": _descriptor(f"{capture_name}/raw/{final_stat_name}", final_stat),
                },
                "listener": {
                    "address": "127.0.0.1",
                    "port": request.endpoint.port,
                    "socket_inode": pre_listener.inode,
                    "pre_proc_net_tcp": _descriptor(f"{capture_name}/raw/{pre_tcp_name}", pre_listener.tcp),
                    "post_proc_net_tcp": _descriptor(f"{capture_name}/raw/{post_tcp_name}", post_listener.tcp),
                    "pre_server_fd_sockets": _descriptor(f"{capture_name}/raw/{pre_sockets_name}", pre_sockets),
                    "post_server_fd_sockets": _descriptor(f"{capture_name}/raw/{post_sockets_name}", post_sockets),
                    "final_proc_net_tcp": _descriptor(f"{capture_name}/raw/{final_tcp_name}", final_listener.tcp),
                    "final_server_fd_sockets": _descriptor(f"{capture_name}/raw/{final_sockets_name}", final_sockets),
                },
                "request_ledger": _descriptor(f"{capture_name}/{ledger_name}", ledger_raw),
                # The runtime-event leaf is the source-written typed audit
                # record itself, never a wrapper-derived event summary.
                "runtime_event_log": _descriptor(audit_path, record_raw),
                "generation_audit_index": _descriptor(f"{capture_name}/{index_name}", index_raw),
            }
            if native_fallback:
                if fallback_name is None or fallback_raw is None:
                    _fail("native fallback event was not captured")
                scenario_row["fallback_event_log"] = _descriptor(
                    f"{audit_dir_name}/{fallback_name}", fallback_raw
                )
            scenario_rows.append(scenario_row)
        if expected_target is None:
            _fail("scenario contract unexpectedly contained no scenarios")
        session = {
            "schema_version": capture_version,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "endpoint": request.endpoint.url,
            "contract": contract_descriptor,
            "runtime_identity": {
                "configuration_profile": profile,
                "configuration_sha256": configuration_sha256,
            },
            "target": expected_target,
            "scenarios": scenario_rows,
        }
        _write_json(capture_fd, "session.json", session)
        _remove_incomplete_marker(capture_fd, marker)
        return session
    finally:
        _close_quietly(raw_fd)
        _close_quietly(capture_fd)
        _close_quietly(audit_fd)
        _close_quietly(root_fd)


def _positive_pid(value: str) -> int:
    if PID_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a positive decimal PID")
    return int(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, type=parse_endpoint)
    parser.add_argument("--server-pid", required=True, type=_positive_pid)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--configuration-profile", required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--capture-name", required=True)
    parser.add_argument("--audit-dir-name", required=True)
    parser.add_argument("--scenario-contract", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--audit-wait-seconds", type=float, default=DEFAULT_AUDIT_WAIT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = capture_raw_scenarios(
            CaptureRequest(
                endpoint=args.endpoint,
                server_pid=args.server_pid,
                candidate_id=args.candidate_id,
                configuration_profile=args.configuration_profile,
                configuration_sha256=args.configuration_sha256,
                evidence_root=args.evidence_root,
                capture_name=args.capture_name,
                audit_dir_name=args.audit_dir_name,
                scenario_contract=args.scenario_contract,
                audit_wait_seconds=args.audit_wait_seconds,
            ),
            repository_root=args.repository_root,
        )
    except RawScenarioCaptureError as error:
        print(f"raw C02 scenario capture refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
