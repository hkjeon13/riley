#!/usr/bin/env python3
"""Capture one raw /v1/config response with its live process bridge.

This is a narrow raw producer for an *already running* local Riley server.  It
does not launch or stop a service, choose a model, evaluate a workload, or
make a qualification decision.  It persists the exact HTTP request/head/body
and the pre/post PID, listener, and GPU facts needed to prove that the config
response came from the same process identity later observed by soak samples.

The module intentionally uses only the standard library: its wrapper invokes
it with ``python -I -S`` on the remote host.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, NoReturn, Sequence
from urllib.parse import urlsplit


BRIDGE_VERSION = "riley.c02-config-endpoint-observation.v1"
INCOMPLETE_MARKER_VERSION = "riley.c02-config-endpoint-observation-incomplete.v1"
SOCKET_SNAPSHOT_VERSION = "riley.c02-proc-fd-socket-snapshot.v2"
# Keep this byte bound identical to the P0 runtime-config byte contract.
MAX_HTTP_BYTES = 8 * 1024 * 1024
# The HTTP envelope is independently bounded.  It must not consume part of a
# valid endpoint body at ``MAX_HTTP_BYTES`` merely because one recv coalesces
# the head and body.
MAX_HTTP_HEAD_BYTES = 64 * 1024
MAX_PROC_BYTES = 4 * 1024 * 1024
MAX_GPU_BYTES = 1024 * 1024
MAX_SERVER_FDS = 16384
HTTP_TIMEOUT_SECONDS = 15.0
NVIDIA_SMI_TIMEOUT_SECONDS = 15.0
UINT_RE = re.compile(r"^[0-9]+$")
PID_RE = re.compile(r"^[1-9][0-9]*$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
SAFE_LEAF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
SOCKET_LINK_RE = re.compile(r"^socket:\[([1-9][0-9]*)\]$")
INCOMPLETE_MARKER_NAME = "capture-incomplete.json"


class ConfigEndpointObservationError(ValueError):
    """A config endpoint observation cannot safely be published."""


def _fail(message: str) -> NoReturn:
    raise ConfigEndpointObservationError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            # Must agree with the P0 endpoint/startup byte contract.  A
            # config payload may legitimately contain non-ASCII literal text.
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail(f"cannot encode canonical evidence JSON: {error}")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("configuration endpoint body repeats a JSON key")
        result[key] = value
    return result


def _validate_canonical_object(raw: bytes, label: str) -> None:
    if not raw or len(raw) > MAX_HTTP_BYTES:
        _fail(f"{label} has an invalid byte length")
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is not canonical JSON: {error}")
    if not isinstance(value, dict) or _canonical(value) != raw:
        _fail(f"{label} is not a canonical JSON object")


def _descriptor(path: str, raw: bytes) -> dict[str, Any]:
    _validate_relative_path(path, "evidence descriptor path")
    if type(raw) is not bytes or not raw:
        _fail("evidence descriptor bytes must be nonempty bytes")
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "byte_length": len(raw)}


def _validate_leaf_name(value: str, label: str) -> str:
    if type(value) is not str or value in {".", ".."} or SAFE_LEAF_RE.fullmatch(value) is None:
        _fail(f"{label} must be a normalized safe direct-child name")
    return value


def _validate_relative_path(value: str, label: str) -> str:
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


def _open_absolute_directory(path: Path, label: str, *, private_root: bool = False) -> int:
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
    _validate_leaf_name(name, f"{label} name")
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
    descriptor = _open_absolute_directory(parent, f"{label} parent")
    try:
        return _read_regular_at(descriptor, parts[-1], label, maximum=maximum)
    finally:
        _close_quietly(descriptor)


def _new_private_directory(parent_fd: int, name: str, label: str) -> int:
    _validate_leaf_name(name, f"{label} name")
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
            _fail(f"{label} is not a private single-link directory")
        _fsync(descriptor, label)
        _fsync(parent_fd, f"parent after {label}")
        return descriptor
    except BaseException:
        _close_quietly(descriptor)
        raise


def _write_new(directory_fd: int, name: str, raw: bytes) -> None:
    _validate_leaf_name(name, "evidence output name")
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


def _capture_lock(capture_fd: int) -> None:
    try:
        fcntl.flock(capture_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(f"cannot acquire exclusive capture lock: {error}")


def _remove_incomplete_marker(capture_fd: int, marker: bytes) -> None:
    """Remove the nonterminal marker last, restoring it if durability fails."""

    try:
        os.unlink(INCOMPLETE_MARKER_NAME, dir_fd=capture_fd)
    except OSError as error:
        _fail(f"cannot remove incomplete capture marker: {error}")
    try:
        _fsync(capture_fd, "capture directory after completion marker removal")
    except ConfigEndpointObservationError:
        # An uncertain unlink must remain visibly incomplete.  A later
        # create-only restoration failure is still fail-closed via the
        # original synchronization error.
        try:
            _write_new(capture_fd, INCOMPLETE_MARKER_NAME, marker)
        except ConfigEndpointObservationError:
            pass
        raise


@dataclass(frozen=True)
class Endpoint:
    url: str
    port: int


@dataclass(frozen=True)
class CaptureRequest:
    endpoint: Endpoint
    server_pid: int
    gpu_index: int
    evidence_root: Path
    capture_name: str


@dataclass(frozen=True)
class Target:
    server_pid: int
    server_start_ticks: int
    listener_port: int
    listener_inode: int
    gpu_index: int
    gpu_uuid: str

    def as_json(self) -> dict[str, Any]:
        return {
            "server_pid": self.server_pid,
            "server_start_ticks": self.server_start_ticks,
            "listener_port": self.listener_port,
            "listener_inode": self.listener_inode,
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
        }


@dataclass(frozen=True)
class Listener:
    tcp: bytes
    inode: int
    sockets: tuple[int, ...]


def parse_endpoint(value: str) -> Endpoint:
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        _fail(f"--endpoint is invalid: {error}")
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname != "127.0.0.1"
        or parsed.path != "/v1/config"
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"127\.0\.0\.1:[0-9]+", parsed.netloc)
    ):
        _fail("--endpoint must be literal http://127.0.0.1:PORT/v1/config")
    try:
        port = parsed.port
    except ValueError as error:
        _fail(f"--endpoint has an invalid port: {error}")
    if port is None or not 1024 <= port <= 65535:
        _fail("--endpoint port must be from 1024 through 65535")
    return Endpoint(url=f"http://127.0.0.1:{port}/v1/config", port=port)


def _parse_option(value: str, label: str, *, positive: bool) -> int:
    if type(value) is not str or UINT_RE.fullmatch(value) is None:
        _fail(f"{label} must be a decimal integer")
    result = int(value)
    if result > 2**31 - 1 or (positive and result < 1):
        _fail(f"{label} is outside its reviewed range")
    return result


def _parse_proc_stat(raw: bytes, expected_pid: int) -> int:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"/proc stat is not ASCII: {error}")
    first_space, closing = text.find(" "), text.rfind(")")
    fields = text[closing + 2 :].split() if closing >= 0 else []
    if first_space < 1 or closing < first_space + 2 or len(fields) <= 19 or PID_RE.fullmatch(text[:first_space]) is None or UINT_RE.fullmatch(fields[19]) is None:
        _fail("/proc stat lacks a valid PID/start-tick tuple")
    if int(text[:first_space]) != expected_pid or int(fields[19]) < 1:
        _fail("/proc stat does not match the requested server PID")
    return int(fields[19])


def _capture_server_stat(pid: int, expected_ticks: int | None) -> tuple[bytes, int]:
    raw = _read_absolute_regular(Path(f"/proc/{pid}/stat"), "server /proc stat", maximum=MAX_PROC_BYTES)
    ticks = _parse_proc_stat(raw, pid)
    if expected_ticks is not None and ticks != expected_ticks:
        _fail("server PID start ticks changed; refusing a reused PID")
    return raw, ticks


def _capture_status(pid: int) -> bytes:
    raw = _read_absolute_regular(Path(f"/proc/{pid}/status"), "server /proc status", maximum=MAX_PROC_BYTES)
    try:
        values = [line[4:].strip() for line in raw.decode("ascii").splitlines() if line.startswith("Pid:")]
    except UnicodeDecodeError as error:
        _fail(f"/proc status is not ASCII: {error}")
    if len(values) != 1 or PID_RE.fullmatch(values[0]) is None or int(values[0]) != pid:
        _fail("/proc status does not bind the requested server PID")
    return raw


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
            _fail("configuration endpoint must not have an IPv4 wildcard listener")
        if local.upper() == expected and remote.upper() == "00000000:0000" and state.upper() == "0A":
            if UINT_RE.fullmatch(inode) is None or int(inode) < 1 or int(inode) in inodes:
                _fail("/proc/net/tcp has an invalid loopback listener inode")
            inodes.add(int(inode))
    if len(inodes) != 1:
        _fail("configuration endpoint must have exactly one IPv4 loopback listener")
    return raw, next(iter(inodes))


def _server_socket_inodes(pid: int) -> tuple[int, ...]:
    descriptor = _open_absolute_directory(Path(f"/proc/{pid}/fd"), "server /proc fd directory")
    try:
        names = os.listdir(descriptor)
        if len(names) > MAX_SERVER_FDS:
            _fail("server /proc fd directory exceeds its reviewed entry bound")
        inodes: set[int] = set()
        for name in names:
            if UINT_RE.fullmatch(name) is None:
                _fail("server /proc fd directory has a non-numeric entry")
            try:
                target = os.readlink(name, dir_fd=descriptor)
            except OSError as error:
                _fail(f"server /proc fd directory changed while sampled: {error}")
            matched = SOCKET_LINK_RE.fullmatch(target)
            if matched:
                inodes.add(int(matched.group(1)))
        return tuple(sorted(inodes))
    finally:
        _close_quietly(descriptor)


def _bound_listener(endpoint: Endpoint, pid: int) -> Listener:
    tcp, inode = _capture_listener(endpoint.port)
    sockets = _server_socket_inodes(pid)
    if inode not in sockets:
        _fail("configuration endpoint listener is not held by the requested PID")
    return Listener(tcp=tcp, inode=inode, sockets=sockets)


def _socket_snapshot(pid: int, sockets: tuple[int, ...]) -> bytes:
    if not sockets:
        _fail("server PID has no observed socket inodes")
    return _canonical({"schema_version": SOCKET_SNAPSHOT_VERSION, "server_pid": pid, "socket_inodes": list(sockets)})


def _run_nvidia_smi(arguments: list[str]) -> bytes:
    try:
        result = subprocess.run(
            ["/usr/bin/nvidia-smi", *arguments], check=True, capture_output=True,
            stdin=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        _fail(f"nvidia-smi query failed: {error}")
    if not 1 <= len(result.stdout) <= MAX_GPU_BYTES:
        _fail("nvidia-smi query output has an invalid byte length")
    return result.stdout


def _parse_selection(raw: bytes, index: int) -> str:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail(f"nvidia-smi selection is not ASCII: {error}")
    if len(lines) != 1:
        _fail("nvidia-smi selection must have exactly one row")
    values = [item.strip() for item in lines[0].split(",")]
    if len(values) != 2 or UINT_RE.fullmatch(values[0]) is None or GPU_UUID_RE.fullmatch(values[1]) is None or int(values[0]) != index:
        _fail("nvidia-smi selection does not bind --gpu-index")
    return values[1]


def _validate_compute_apps(raw: bytes, pid: int) -> None:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail(f"nvidia-smi compute-apps is not ASCII: {error}")
    found, seen = 0, set()
    for line in lines:
        if not line:
            continue
        values = [item.strip() for item in line.split(",")]
        if len(values) != 2 or PID_RE.fullmatch(values[0]) is None or UINT_RE.fullmatch(values[1]) is None or int(values[0]) in seen:
            _fail("nvidia-smi compute-apps has a malformed or duplicate PID")
        seen.add(int(values[0]))
        if int(values[0]) == pid:
            found += 1
    if found != 1:
        _fail("nvidia-smi compute-apps must contain exactly one server PID row")


def _capture_gpu(index: int, pid: int, expected_uuid: str | None) -> tuple[bytes, bytes, str]:
    selection = _run_nvidia_smi(["-i", str(index), "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    uuid = _parse_selection(selection, index)
    if expected_uuid is not None and uuid != expected_uuid:
        _fail("nvidia-smi GPU UUID changed while configuration endpoint was captured")
    apps = _run_nvidia_smi(["-i", str(index), "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    _validate_compute_apps(apps, pid)
    return selection, apps, uuid


def _config_request(port: int) -> bytes:
    return (
        f"GET /v1/config HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        "Accept: application/json\r\nConnection: close\r\n\r\n"
    ).encode("ascii")


def _response_length(head: bytes) -> int:
    try:
        text = head.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"configuration response head is not ASCII: {error}")
    if not text.endswith("\r\n\r\n"):
        _fail("configuration response head lacks a complete terminator")
    lines = text[:-4].split("\r\n")
    if not lines or lines[0] != "HTTP/1.1 200 OK":
        _fail("configuration endpoint must return HTTP/1.1 200 OK")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line or line[:1] in {" ", "\t"}:
            _fail("configuration response head has a malformed header")
        name, value = line.split(":", 1)
        key = name.lower()
        if key in headers or not re.fullmatch(r"[A-Za-z0-9-]+", name):
            _fail("configuration response head repeats or has an invalid header")
        headers[key] = value.strip()
    length = headers.get("content-length")
    content_type = headers.get("content-type")
    if "transfer-encoding" in headers:
        _fail("configuration response must not use Transfer-Encoding")
    if length is None or UINT_RE.fullmatch(length) is None or not 1 <= int(length) <= MAX_HTTP_BYTES:
        _fail("configuration response has an invalid Content-Length")
    if content_type is None or content_type.split(";", 1)[0].strip().lower() != "application/json":
        _fail("configuration response Content-Type is not application/json")
    return int(length)


def _capture_endpoint(endpoint: Endpoint) -> tuple[bytes, bytes, bytes]:
    request = _config_request(endpoint.port)
    try:
        connection = socket.create_connection(("127.0.0.1", endpoint.port), timeout=HTTP_TIMEOUT_SECONDS)
        with connection:
            connection.settimeout(HTTP_TIMEOUT_SECONDS)
            connection.sendall(request)
            data = bytearray()
            marker = -1
            while marker < 0:
                remaining_head = MAX_HTTP_HEAD_BYTES + 1 - len(data)
                if remaining_head < 1:
                    _fail("configuration response headers exceed their byte bound")
                chunk = connection.recv(min(65536, remaining_head))
                if not chunk:
                    _fail("configuration endpoint closed before response headers")
                data.extend(chunk)
                marker = data.find(b"\r\n\r\n")
                if marker < 0 and len(data) > MAX_HTTP_HEAD_BYTES:
                    _fail("configuration response headers exceed their byte bound")
            marker += 4
            if marker > MAX_HTTP_HEAD_BYTES:
                _fail("configuration response headers exceed their byte bound")
            head, body = bytes(data[:marker]), bytearray(data[marker:])
            expected = _response_length(head)
            if len(body) > expected:
                _fail("configuration response has bytes beyond Content-Length")
            # The producer asks for ``Connection: close`` and consumes to EOF,
            # not just to Content-Length.  That makes a hidden second response
            # or trailing bytes a fail-closed capture error rather than an
            # unbound transport artifact.
            while True:
                chunk = connection.recv(min(65536, expected + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > expected:
                    _fail("configuration response has bytes beyond Content-Length")
            if len(body) != expected:
                _fail("configuration endpoint closed before Content-Length bytes")
    except OSError as error:
        _fail(f"could not capture configuration endpoint: {error}")
    raw_body = bytes(body)
    _validate_canonical_object(raw_body, "configuration endpoint body")
    return request, head, raw_body


def _external_root(root: Path, repository_root: Path) -> None:
    try:
        if os.path.commonpath((os.fspath(root), os.fspath(repository_root))) == os.fspath(repository_root):
            _fail("--evidence-root must be outside the source checkout")
    except ValueError as error:
        _fail(f"cannot compare evidence root and source checkout: {error}")


def _preflight(request: CaptureRequest) -> Target:
    _before, ticks = _capture_server_stat(request.server_pid, None)
    listener = _bound_listener(request.endpoint, request.server_pid)
    _selection, _apps, uuid = _capture_gpu(request.gpu_index, request.server_pid, None)
    return Target(request.server_pid, ticks, request.endpoint.port, listener.inode, request.gpu_index, uuid)


def capture_config_endpoint_observation(request: CaptureRequest, *, repository_root: Path) -> dict[str, Any]:
    """Capture a durable raw bridge under one held external evidence-root FD."""

    _validate_leaf_name(request.capture_name, "--capture-name")
    _external_root(request.evidence_root, repository_root)
    target = _preflight(request)
    root_fd = _open_absolute_directory(request.evidence_root, "--evidence-root", private_root=True)
    capture_fd: int | None = None
    raw_fd: int | None = None
    marker = _canonical({"schema_version": INCOMPLETE_MARKER_VERSION, "capture_status": "incomplete", "qualification_status": "not-run"})
    try:
        capture_fd = _new_private_directory(root_fd, request.capture_name, "configuration endpoint capture directory")
        _capture_lock(capture_fd)
        _write_new(capture_fd, INCOMPLETE_MARKER_NAME, marker)
        raw_fd = _new_private_directory(capture_fd, "raw", "configuration endpoint raw directory")
        before_stat, ticks = _capture_server_stat(target.server_pid, target.server_start_ticks)
        if ticks != target.server_start_ticks:
            _fail("server PID start ticks drifted before configuration request")
        listener_before = _bound_listener(request.endpoint, target.server_pid)
        before_sockets = _socket_snapshot(target.server_pid, listener_before.sockets)
        request_raw, response_head, endpoint_body = _capture_endpoint(request.endpoint)
        listener_after = _bound_listener(request.endpoint, target.server_pid)
        after_sockets = _socket_snapshot(target.server_pid, listener_after.sockets)
        if listener_before.inode != listener_after.inode or listener_before.inode != target.listener_inode:
            _fail("configuration endpoint listener inode changed while sampled")
        selection, apps, uuid = _capture_gpu(target.gpu_index, target.server_pid, target.gpu_uuid)
        if uuid != target.gpu_uuid:
            _fail("configuration endpoint GPU UUID drifted while sampled")
        status = _capture_status(target.server_pid)
        after_stat, after_ticks = _capture_server_stat(target.server_pid, target.server_start_ticks)
        if after_ticks != target.server_start_ticks:
            _fail("server PID start ticks drifted after configuration request")
        raw = {
            "request": request_raw,
            "response_head": response_head,
            "endpoint": endpoint_body,
            "tcp_before": listener_before.tcp,
            "tcp_after": listener_after.tcp,
            "sockets_before": before_sockets,
            "sockets_after": after_sockets,
            "stat_before": before_stat,
            "stat_after": after_stat,
            "status": status,
            "selection": selection,
            "apps": apps,
        }
        filenames = {
            "request": "config-request.http",
            "response_head": "config-response-head.http",
            "endpoint": "config-endpoint.json",
            "tcp_before": "proc-net-tcp-before",
            "tcp_after": "proc-net-tcp-after",
            "sockets_before": "proc-fd-sockets-before.json",
            "sockets_after": "proc-fd-sockets-after.json",
            "stat_before": "proc-stat-before",
            "stat_after": "proc-stat-after",
            "status": "proc-status",
            "selection": "gpu-selection.csv",
            "apps": "gpu-compute-apps.csv",
        }
        for key, filename in filenames.items():
            _write_new(raw_fd, filename, raw[key])
        prefix = f"{request.capture_name}/raw"
        describe = lambda key: _descriptor(f"{prefix}/{filenames[key]}", raw[key])
        session = {
            "schema_version": BRIDGE_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "target": target.as_json(),
            "endpoint": {
                "method": "GET",
                "request_target": "/v1/config",
                "http_status": 200,
                "request": describe("request"),
                "response_head": describe("response_head"),
                "body_sha256": hashlib.sha256(endpoint_body).hexdigest(),
                "body_byte_length": len(endpoint_body),
                "listener": {
                    "address": "127.0.0.1",
                    "port": target.listener_port,
                    "socket_inode": target.listener_inode,
                    "before_proc_net_tcp": describe("tcp_before"),
                    "after_proc_net_tcp": describe("tcp_after"),
                    "before_server_fd_sockets": describe("sockets_before"),
                    "after_server_fd_sockets": describe("sockets_after"),
                },
            },
            "process": {
                "server_pid": target.server_pid,
                "server_start_ticks": target.server_start_ticks,
                "pre_endpoint_stat": describe("stat_before"),
                "post_endpoint_stat": describe("stat_after"),
                "status": describe("status"),
            },
            "gpu": {
                "index": target.gpu_index,
                "uuid": target.gpu_uuid,
                "selection_query": describe("selection"),
                "compute_apps": describe("apps"),
            },
        }
        _write_new(capture_fd, "session.json", _canonical(session))
        _remove_incomplete_marker(capture_fd, marker)
        return session
    finally:
        _close_quietly(raw_fd)
        _close_quietly(capture_fd)
        _close_quietly(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--server-pid", required=True)
    parser.add_argument("--gpu-index", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--capture-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = CaptureRequest(
            endpoint=parse_endpoint(args.endpoint),
            server_pid=_parse_option(args.server_pid, "--server-pid", positive=True),
            gpu_index=_parse_option(args.gpu_index, "--gpu-index", positive=False),
            evidence_root=args.evidence_root,
            capture_name=args.capture_name,
        )
        session = capture_config_endpoint_observation(
            request, repository_root=Path(__file__).resolve().parents[2]
        )
    except ConfigEndpointObservationError as error:
        print(f"C02 config endpoint observation failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical(session) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
