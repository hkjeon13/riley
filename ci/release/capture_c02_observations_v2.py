#!/usr/bin/env python3
"""Capture raw C02 v2 observations from an already-running local Riley server.

This producer is deliberately narrower than a soak, rollback, or candidate
qualification runner.  It never starts or stops Riley, CUDA processes,
containers, or SSH sessions.  Given an explicit loopback C02 endpoint, server
PID, GPU index, private evidence root, and fresh capture name, it preserves
the exact host and HTTP bytes needed by the v2 raw-provenance binder.

The script is self-contained because its remote wrapper uses ``python -I -S``.
It therefore uses only the Python standard library and fails closed when the
Linux no-follow/openat primitives required for evidence capture are missing.
Its output always says ``qualification_status: "not-run"``; later reviewed
binders and semantic checkers own all pass/fail decisions.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, NoReturn, Sequence
from urllib.parse import urlsplit


# Running a source-tree helper must never create __pycache__ in the checkout.
sys.dont_write_bytecode = True


SESSION_VERSION = "riley.c02-raw-observation-session.v2"
SAMPLE_VERSION = "riley.c02-raw-observation-sample.v2"
SOCKET_SNAPSHOT_VERSION = "riley.c02-proc-fd-socket-snapshot.v2"
METRICS_VERSION = "riley.c02-capture-metrics.v2"
INCOMPLETE_MARKER_VERSION = "riley.c02-raw-observation-incomplete.v2"
INCOMPLETE_MARKER_NAME = "capture-incomplete.json"

MAX_HTTP_BYTES = 16 * 1024 * 1024
MAX_PROC_BYTES = 1024 * 1024
MAX_GPU_BYTES = 1024 * 1024
MAX_SAMPLE_COUNT = 1024
MAX_INTERVAL_SECONDS = 300
MAX_SERVER_FDS = 65536
HTTP_TIMEOUT_SECONDS = 15
NVIDIA_SMI_TIMEOUT_SECONDS = 15

SAFE_LEAF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
UINT_RE = re.compile(r"^[0-9]+$")
PID_RE = re.compile(r"^[1-9][0-9]*$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
SOCKET_LINK_RE = re.compile(r"^socket:\[([1-9][0-9]*)\]$")


class ObservationCaptureError(ValueError):
    """The requested raw observation cannot be captured safely."""


def _fail(message: str) -> NoReturn:
    raise ObservationCaptureError(message)


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
        _fail(f"cannot encode canonical JSON: {error}")


def _parse_canonical_object(raw: bytes, label: str, *, maximum: int) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _fail(f"{label} has an invalid byte length")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        _fail(f"{label} is not strict UTF-8 JSON: {error}")
    except json.JSONDecodeError as error:
        _fail(f"{label} is not JSON: {error}")
    if not isinstance(value, dict):
        _fail(f"{label} root must be a JSON object")
    if raw != _canonical(value):
        _fail(f"{label} must use exact canonical JSON bytes")
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{label} must contain exactly {sorted(fields)}")
    return value


def _nonnegative(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a non-negative integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be a boolean")
    return value


def validate_metrics_raw(raw: bytes) -> None:
    """Require only the v2 raw metric *shape*, never a qualification result.

    This deliberately does not reject failures, require a quiescent state,
    compare counters, or apply a KV/allocation threshold.  It merely ensures
    the preserved HTTP bytes are the schema that the v2 binder can later bind.
    """

    row = _exact(
        _parse_canonical_object(raw, "C02 metrics response", maximum=MAX_HTTP_BYTES),
        {"schema_version", "request_states", "kv_blocks", "allocation", "quiescence"},
        "C02 metrics response",
    )
    if row["schema_version"] != METRICS_VERSION:
        _fail(f"C02 metrics response must use {METRICS_VERSION}")
    states = _exact(
        row["request_states"],
        {
            "active",
            "pending_requests",
            "completed",
            "failed",
            "cancelled",
            "capacity_rejections",
        },
        "C02 metrics response.request_states",
    )
    for name in states:
        _nonnegative(states[name], f"C02 metrics response.request_states.{name}")
    kv_blocks = _exact(
        row["kv_blocks"],
        {"free", "reserved", "active"},
        "C02 metrics response.kv_blocks",
    )
    for name in kv_blocks:
        _nonnegative(kv_blocks[name], f"C02 metrics response.kv_blocks.{name}")
    allocation = _exact(
        row["allocation"],
        {
            "device_live_count",
            "device_live_bytes",
            "pinned_live_count",
            "pinned_live_bytes",
        },
        "C02 metrics response.allocation",
    )
    for name in allocation:
        _nonnegative(allocation[name], f"C02 metrics response.allocation.{name}")
    quiescence = _exact(
        row["quiescence"],
        {
            "completion_outbox",
            "outstanding_iterations",
            "riley_owned_live_allocations",
            "worker_accepting",
            "scheduler_accepting",
        },
        "C02 metrics response.quiescence",
    )
    for name in (
        "completion_outbox",
        "outstanding_iterations",
        "riley_owned_live_allocations",
    ):
        _nonnegative(quiescence[name], f"C02 metrics response.quiescence.{name}")
    for name in ("worker_accepting", "scheduler_accepting"):
        _boolean(quiescence[name], f"C02 metrics response.quiescence.{name}")


def _validate_leaf_name(name: str, label: str) -> str:
    if type(name) is not str or SAFE_LEAF_RE.fullmatch(name) is None or name in {".", ".."}:
        _fail(f"{label} must be a normalized safe leaf name")
    return name


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
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"{label} must not contain a path alias")
    return value


def _descriptor(path: str, raw: bytes) -> dict[str, Any]:
    _validate_relative_path(path, "evidence descriptor path")
    if type(raw) is not bytes:
        _fail("evidence descriptor bytes must be bytes")
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
    nofollow = _required_open_flag("O_NOFOLLOW")
    directory_flag = _required_open_flag("O_DIRECTORY")
    cloexec = _required_open_flag("O_CLOEXEC")
    nonblock = _required_open_flag("O_NONBLOCK")
    access = os.O_WRONLY | os.O_CREAT | os.O_EXCL if writable else os.O_RDONLY
    return access | nofollow | cloexec | nonblock | (directory_flag if directory else 0)


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _fsync_checked(descriptor: int, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        _fail(f"cannot durably synchronize {label}: {error}")


def _absolute_components(path: Path, label: str) -> tuple[str, ...]:
    raw = os.fspath(path)
    if (
        not os.path.isabs(raw)
        or "\x00" in raw
        or "\\" in raw
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        _fail(f"{label} must be a normalized absolute path")
    parts = Path(raw).parts
    if not parts or parts[0] != os.path.sep:
        _fail(f"{label} must be a normalized absolute path")
    components = tuple(parts[1:])
    if any(component in {"", ".", ".."} for component in components):
        _fail(f"{label} contains an unsafe path component")
    return components


def _require_directory_fd(descriptor: int, label: str) -> os.stat_result:
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot inspect {label}: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} must be a directory")
    return metadata


def _open_absolute_directory(path: Path, label: str) -> int:
    """Open an absolute directory one component at a time without symlinks."""

    components = _absolute_components(path, label)
    flags = _open_flags(directory=True)
    try:
        current = os.open(os.path.sep, flags)
    except OSError as error:
        _fail(f"cannot open filesystem root for {label}: {error}")
    try:
        for component in components:
            try:
                child = os.open(component, flags, dir_fd=current)
            except OSError as error:
                _fail(f"cannot open {label} without following links: {error}")
            os.close(current)
            current = child
        _require_directory_fd(current, label)
        return current
    except BaseException:
        _close_quietly(current)
        raise


def _require_safe_evidence_ancestor(metadata: os.stat_result, label: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} must be a directory")
    writable = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if not writable:
        return
    if not metadata.st_mode & stat.S_ISVTX:
        _fail(f"{label} is group/world writable without a sticky boundary")
    if metadata.st_uid not in {0, os.geteuid()}:
        _fail(f"{label} is writable and not owned by root or the effective UID")


def _open_private_evidence_root(path: Path, label: str) -> int:
    """Pin an existing euid-owned exact-0700 root through no-follow FDs."""

    components = _absolute_components(path, label)
    flags = _open_flags(directory=True)
    try:
        current = os.open(os.path.sep, flags)
    except OSError as error:
        _fail(f"cannot open filesystem root for {label}: {error}")
    try:
        _require_safe_evidence_ancestor(os.fstat(current), f"{label} ancestor /")
        for component in components:
            try:
                child = os.open(component, flags, dir_fd=current)
            except OSError as error:
                _fail(f"cannot open {label} without following links: {error}")
            os.close(current)
            current = child
            _require_safe_evidence_ancestor(
                os.fstat(current),
                f"{label} ancestor {component!r}",
            )
        metadata = _require_directory_fd(current, label)
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            _fail(f"{label} must be effective-UID-owned with exact mode 0700")
        return current
    except BaseException:
        _close_quietly(current)
        raise


def _read_regular_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    maximum: int,
) -> bytes:
    """Read a bounded host leaf without following a raced link replacement."""

    _validate_leaf_name(name, f"{label} name")
    _require_directory_fd(directory_fd, f"{label} parent")
    try:
        before = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot inspect {label}: {error}")
    if not stat.S_ISREG(before.st_mode):
        _fail(f"{label} must be a regular non-link file")
    try:
        descriptor = os.open(name, _open_flags(directory=False), dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot open {label} without following links: {error}")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            _fail(f"{label} changed to a non-regular file while opened")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            _fail(f"{label} changed while it was opened")
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            except OSError as error:
                _fail(f"cannot read {label}: {error}")
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                _fail(f"{label} exceeds its byte bound")
        after = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
            _fail(f"{label} changed while it was read")
    finally:
        _close_quietly(descriptor)
    try:
        path_after = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot re-inspect {label}: {error}")
    if (before.st_dev, before.st_ino) != (path_after.st_dev, path_after.st_ino):
        _fail(f"{label} changed while it was read")
    return b"".join(chunks)


def _read_absolute_regular(path: Path, label: str, *, maximum: int) -> bytes:
    components = _absolute_components(path, label)
    if not components:
        _fail(f"{label} must name a regular file")
    parent = Path(os.path.sep).joinpath(*components[:-1])
    parent_fd = _open_absolute_directory(parent, f"{label} parent")
    try:
        return _read_regular_at(parent_fd, components[-1], label, maximum=maximum)
    finally:
        _close_quietly(parent_fd)


def _require_new_private_directory(
    parent_fd: int,
    name: str,
    label: str,
) -> int:
    """Create one fresh 0700 direct child and return its held directory FD."""

    _validate_leaf_name(name, f"{label} name")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        _fail(f"{label} already exists; capture names are create-only")
    except OSError as error:
        _fail(f"cannot create {label}: {error}")
    try:
        descriptor = os.open(name, _open_flags(directory=True), dir_fd=parent_fd)
    except OSError as error:
        _fail(f"cannot reopen newly created {label} without following links: {error}")
    try:
        os.fchmod(descriptor, 0o700)
        metadata = _require_directory_fd(descriptor, label)
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_nlink != 2
        ):
            _fail(f"{label} was not created as a private single-link directory")
        _fsync_checked(descriptor, label)
        _fsync_checked(parent_fd, f"parent directory after {label}")
        return descriptor
    except BaseException:
        _close_quietly(descriptor)
        raise


def _acquire_capture_lock(capture_fd: int) -> None:
    """Take an advisory exclusive lock for the lifetime of this fresh capture."""

    try:
        fcntl.flock(capture_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(f"cannot acquire exclusive capture lock: {error}")


def _open_new_output_file(directory_fd: int, name: str) -> int:
    _validate_leaf_name(name, "evidence output name")
    try:
        descriptor = os.open(name, _open_flags(directory=False, writable=True), 0o600, dir_fd=directory_fd)
    except OSError as error:
        _fail(f"cannot create evidence file {name}: {error}")
    try:
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            _fail(f"new evidence file {name} is not a private single-link regular file")
        return descriptor
    except BaseException:
        _close_quietly(descriptor)
        raise


def _write_open_output_file(descriptor: int, name: str, raw: bytes) -> None:
    if type(raw) is not bytes:
        _fail(f"evidence file {name} cannot be written from non-bytes")
    try:
        offset = 0
        while offset < len(raw):
            try:
                written = os.write(descriptor, raw[offset:])
            except OSError as error:
                _fail(f"cannot write evidence file {name}: {error}")
            if written <= 0:
                _fail(f"cannot write evidence file {name}: short write")
            offset += written
        _fsync_checked(descriptor, f"evidence file {name}")
    finally:
        _close_quietly(descriptor)


def _write_new(directory_fd: int, name: str, raw: bytes) -> None:
    descriptor = _open_new_output_file(directory_fd, name)
    _write_open_output_file(descriptor, name, raw)
    _fsync_checked(directory_fd, f"evidence directory after {name}")


def _write_terminal_session(capture_fd: int, raw: bytes) -> None:
    """Create and sync the terminal session while the incomplete marker remains."""

    created = False
    try:
        descriptor = _open_new_output_file(capture_fd, "session.json")
        created = True
        _write_open_output_file(descriptor, "session.json", raw)
        _fsync_checked(capture_fd, "capture directory after terminal session")
    except ObservationCaptureError:
        if created:
            try:
                os.unlink("session.json", dir_fd=capture_fd)
                _fsync_checked(capture_fd, "capture directory after incomplete session removal")
            except ObservationCaptureError:
                pass
            except OSError:
                pass
        raise


def _remove_incomplete_marker(capture_fd: int, raw: bytes) -> None:
    """Remove the nonterminal marker last; restore it if its sync fails."""

    try:
        os.unlink(INCOMPLETE_MARKER_NAME, dir_fd=capture_fd)
    except OSError as error:
        _fail(f"cannot remove incomplete capture marker: {error}")
    try:
        _fsync_checked(capture_fd, "capture directory after completion marker removal")
    except ObservationCaptureError:
        # If the unlink's durability is uncertain, make the capture visibly
        # incomplete again.  A failed restoration remains an explicit error.
        try:
            _write_new(capture_fd, INCOMPLETE_MARKER_NAME, raw)
        except ObservationCaptureError:
            pass
        raise


def _assert_external_to_repository(evidence_root: Path, repository_root: Path) -> None:
    root_raw = os.fspath(evidence_root)
    repository_raw = os.fspath(repository_root)
    try:
        shared = os.path.commonpath((root_raw, repository_raw))
    except ValueError as error:
        _fail(f"cannot compare evidence root and repository: {error}")
    if shared == repository_raw:
        _fail("--evidence-root must be outside the source checkout")


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
    interval_seconds: int
    sample_count: int


@dataclass(frozen=True)
class TargetIdentity:
    server_pid: int
    server_start_ticks: int
    gpu_index: int
    gpu_uuid: str

    def as_json(self) -> dict[str, Any]:
        return {
            "server_pid": self.server_pid,
            "server_start_ticks": self.server_start_ticks,
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
        }


@dataclass(frozen=True)
class BoundListener:
    proc_net_tcp: bytes
    socket_inode: int
    server_socket_inodes: tuple[int, ...]


@dataclass(frozen=True)
class CapturedSample:
    raw_files: tuple[tuple[str, bytes], ...]
    document: dict[str, Any]


@dataclass(frozen=True)
class CaptureDirectories:
    evidence_root_fd: int
    capture_fd: int
    raw_fd: int
    samples_fd: int


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
        or parsed.path != "/v1/c02/metrics"
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"127\.0\.0\.1:[0-9]+", parsed.netloc)
    ):
        _fail("--endpoint must be literal http://127.0.0.1:PORT/v1/c02/metrics")
    try:
        port = parsed.port
    except ValueError as error:
        _fail(f"--endpoint has an invalid port: {error}")
    if port is None or not 1024 <= port <= 65535:
        _fail("--endpoint port must be from 1024 through 65535")
    return Endpoint(url=f"http://127.0.0.1:{port}/v1/c02/metrics", port=port)


def _parse_unsigned_option(value: str, label: str, *, maximum: int) -> int:
    if type(value) is not str or UINT_RE.fullmatch(value) is None:
        _fail(f"{label} must be a decimal integer")
    parsed = int(value)
    if parsed > maximum:
        _fail(f"{label} exceeds its reviewed bound")
    return parsed


def _parse_positive_option(value: str, label: str, *, maximum: int) -> int:
    parsed = _parse_unsigned_option(value, label, maximum=maximum)
    if parsed < 1:
        _fail(f"{label} must be positive")
    return parsed


def _capture_endpoint(endpoint: Endpoint) -> bytes:
    connection = http.client.HTTPConnection("127.0.0.1", endpoint.port, timeout=HTTP_TIMEOUT_SECONDS)
    try:
        connection.request(
            "GET",
            "/v1/c02/metrics",
            headers={"Accept": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
        if response.version != 11:
            _fail("C02 endpoint did not respond with HTTP/1.1")
        if response.status != 200:
            _fail(f"C02 endpoint returned HTTP {response.status}, expected 200")
        content_length = response.getheader("Content-Length")
        if content_length is None or UINT_RE.fullmatch(content_length) is None:
            _fail("C02 endpoint response has no exact Content-Length")
        expected_length = int(content_length)
        if expected_length < 1 or expected_length > MAX_HTTP_BYTES:
            _fail("C02 endpoint response Content-Length is out of bounds")
        content_type = response.getheader("Content-Type")
        if content_type is None or content_type.split(";", 1)[0].strip().lower() != "application/json":
            _fail("C02 endpoint response Content-Type is not application/json")
        raw = response.read(MAX_HTTP_BYTES + 1)
        if len(raw) != expected_length:
            _fail("C02 endpoint response body length differs from Content-Length")
    except (OSError, http.client.HTTPException) as error:
        _fail(f"could not capture C02 endpoint: {error}")
    finally:
        connection.close()
    validate_metrics_raw(raw)
    return raw


def _parse_proc_stat(raw: bytes, expected_pid: int) -> int:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"/proc stat is not ASCII: {error}")
    first_space = text.find(" ")
    closing = text.rfind(")")
    if first_space < 1 or closing < first_space + 2 or closing + 2 >= len(text):
        _fail("/proc stat has an invalid comm field")
    pid_text = text[:first_space]
    fields = text[closing + 2 :].split()
    if (
        PID_RE.fullmatch(pid_text) is None
        or len(fields) <= 19
        or UINT_RE.fullmatch(fields[19]) is None
    ):
        _fail("/proc stat lacks a valid PID/start-tick tuple")
    pid = int(pid_text)
    start_ticks = int(fields[19])
    if pid != expected_pid or start_ticks < 1:
        _fail("/proc stat does not match the requested live server PID")
    return start_ticks


def _capture_server_stat(server_pid: int, expected_start_ticks: int | None) -> tuple[bytes, int]:
    raw = _read_absolute_regular(
        Path(f"/proc/{server_pid}/stat"),
        "server /proc stat",
        maximum=MAX_PROC_BYTES,
    )
    start_ticks = _parse_proc_stat(raw, server_pid)
    if expected_start_ticks is not None and start_ticks != expected_start_ticks:
        _fail("server PID start ticks changed; refusing a reused process ID")
    return raw, start_ticks


def _capture_server_status(server_pid: int) -> bytes:
    raw = _read_absolute_regular(
        Path(f"/proc/{server_pid}/status"),
        "server /proc status",
        maximum=MAX_PROC_BYTES,
    )
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail(f"/proc status is not ASCII: {error}")
    found: int | None = None
    for line in lines:
        if not line.startswith("Pid:"):
            continue
        if found is not None:
            _fail("/proc status repeats Pid")
        value = line[4:].strip()
        if PID_RE.fullmatch(value) is None:
            _fail("/proc status has an invalid Pid")
        found = int(value)
    if found != server_pid:
        _fail("/proc status does not bind the requested server PID")
    return raw


def _capture_loopback_listener(port: int) -> tuple[bytes, int]:
    raw = _read_absolute_regular(Path("/proc/net/tcp"), "/proc/net/tcp", maximum=MAX_PROC_BYTES)
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail(f"/proc/net/tcp is not ASCII: {error}")
    if not lines or not lines[0].lstrip().startswith("sl"):
        _fail("/proc/net/tcp has no valid header")
    expected = f"0100007F:{port:04X}"
    listeners: set[int] = set()
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 10:
            _fail("/proc/net/tcp has a malformed socket row")
        local, remote, state, inode = fields[1], fields[2], fields[3], fields[9]
        if (
            local.upper() == f"00000000:{port:04X}"
            and remote.upper() == "00000000:0000"
            and state.upper() == "0A"
        ):
            _fail("C02 endpoint port must not have an IPv4 wildcard listener")
        if local.upper() != expected:
            continue
        if remote.upper() != "00000000:0000" or state.upper() != "0A":
            continue
        if UINT_RE.fullmatch(inode) is None or int(inode) < 1:
            _fail("/proc/net/tcp has an invalid listener inode")
        parsed = int(inode)
        if parsed in listeners:
            _fail("/proc/net/tcp repeats a loopback listener inode")
        listeners.add(parsed)
    if len(listeners) != 1:
        _fail("C02 endpoint port must have exactly one IPv4 loopback listener")
    return raw, next(iter(listeners))


def _server_socket_inodes(server_pid: int) -> tuple[int, ...]:
    descriptor = _open_absolute_directory(
        Path(f"/proc/{server_pid}/fd"),
        "server /proc fd directory",
    )
    try:
        try:
            names = os.listdir(descriptor)
        except OSError as error:
            _fail(f"cannot list server /proc fd directory: {error}")
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
            match = SOCKET_LINK_RE.fullmatch(target)
            if match is not None:
                inodes.add(int(match.group(1)))
        return tuple(sorted(inodes))
    finally:
        _close_quietly(descriptor)


def _capture_bound_listener(endpoint: Endpoint, server_pid: int) -> BoundListener:
    proc_net_tcp, socket_inode = _capture_loopback_listener(endpoint.port)
    socket_inodes = _server_socket_inodes(server_pid)
    if socket_inode not in socket_inodes:
        _fail("C02 endpoint listener is not held by the requested server PID")
    return BoundListener(
        proc_net_tcp=proc_net_tcp,
        socket_inode=socket_inode,
        server_socket_inodes=socket_inodes,
    )


def _socket_snapshot_raw(server_pid: int, socket_inodes: tuple[int, ...]) -> bytes:
    if not socket_inodes:
        _fail("server PID has no observed socket inodes")
    return _canonical(
        {
            "schema_version": SOCKET_SNAPSHOT_VERSION,
            "server_pid": server_pid,
            "socket_inodes": list(socket_inodes),
        }
    )


def _run_nvidia_smi(arguments: list[str]) -> bytes:
    command = ["/usr/bin/nvidia-smi", *arguments]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        _fail(f"nvidia-smi query failed: {error}")
    if len(completed.stdout) < 1 or len(completed.stdout) > MAX_GPU_BYTES:
        _fail("nvidia-smi query output has an invalid byte length")
    return completed.stdout


def _parse_gpu_selection(raw: bytes, requested_index: int) -> str:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail(f"nvidia-smi GPU selection output is not ASCII: {error}")
    if len(lines) != 1:
        _fail("nvidia-smi GPU selection output must have exactly one row")
    cells = [cell.strip() for cell in lines[0].split(",")]
    if len(cells) != 2 or UINT_RE.fullmatch(cells[0]) is None or GPU_UUID_RE.fullmatch(cells[1]) is None:
        _fail("nvidia-smi GPU selection output is malformed")
    if int(cells[0]) != requested_index:
        _fail("nvidia-smi GPU selection index differs from --gpu-index")
    return cells[1]


def _validate_compute_apps(raw: bytes, server_pid: int) -> None:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail(f"nvidia-smi compute-apps output is not ASCII: {error}")
    seen: set[int] = set()
    target_rows = 0
    for line in lines:
        if not line:
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 2 or PID_RE.fullmatch(cells[0]) is None or UINT_RE.fullmatch(cells[1]) is None:
            _fail("nvidia-smi compute-apps output is malformed")
        pid = int(cells[0])
        if pid in seen:
            _fail("nvidia-smi compute-apps output repeats a PID")
        seen.add(pid)
        if pid == server_pid:
            target_rows += 1
    if target_rows != 1:
        _fail("nvidia-smi compute-apps output must contain exactly one server PID row")


def _capture_gpu(
    gpu_index: int,
    server_pid: int,
    expected_uuid: str | None,
) -> tuple[bytes, bytes, str]:
    index = str(gpu_index)
    selection = _run_nvidia_smi(
        ["-i", index, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    uuid = _parse_gpu_selection(selection, gpu_index)
    if expected_uuid is not None and uuid != expected_uuid:
        _fail("nvidia-smi GPU UUID changed while observations were captured")
    compute_apps = _run_nvidia_smi(
        [
            "-i",
            index,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    _validate_compute_apps(compute_apps, server_pid)
    return selection, compute_apps, uuid


def _preflight_target(request: CaptureRequest) -> TargetIdentity:
    """Verify a live target before making a new capture directory.

    These preflight bytes are intentionally not evidence.  Every persisted
    sample repeats the checks and preserves its own raw leaves.  Preflight
    avoids leaving a fresh capture directory for a plainly unavailable PID,
    listener, or selected GPU.
    """

    _stat_raw, start_ticks = _capture_server_stat(request.server_pid, None)
    _capture_bound_listener(request.endpoint, request.server_pid)
    _selection, _compute_apps, gpu_uuid = _capture_gpu(
        request.gpu_index,
        request.server_pid,
        None,
    )
    return TargetIdentity(
        server_pid=request.server_pid,
        server_start_ticks=start_ticks,
        gpu_index=request.gpu_index,
        gpu_uuid=gpu_uuid,
    )


def _capture_one(
    request: CaptureRequest,
    target: TargetIdentity,
    *,
    sequence: int,
    elapsed_monotonic_millis: int,
) -> CapturedSample:
    """Capture and bind all raw leaves for one observed endpoint response."""

    before_stat, before_start_ticks = _capture_server_stat(
        target.server_pid,
        target.server_start_ticks,
    )
    if before_start_ticks != target.server_start_ticks:
        _fail("server start ticks drifted before endpoint capture")
    listener_before = _capture_bound_listener(request.endpoint, target.server_pid)
    before_sockets = _socket_snapshot_raw(target.server_pid, listener_before.server_socket_inodes)
    metrics = _capture_endpoint(request.endpoint)
    listener_after = _capture_bound_listener(request.endpoint, target.server_pid)
    after_sockets = _socket_snapshot_raw(target.server_pid, listener_after.server_socket_inodes)
    if listener_before.socket_inode != listener_after.socket_inode:
        _fail("C02 endpoint listener socket inode changed while sampled")
    selection, compute_apps, gpu_uuid = _capture_gpu(
        target.gpu_index,
        target.server_pid,
        target.gpu_uuid,
    )
    if gpu_uuid != target.gpu_uuid:
        _fail("GPU UUID drifted after endpoint capture")
    status = _capture_server_status(target.server_pid)
    # This final stat is deliberately captured after the GPU process query,
    # so a PID reused during the rest of the sample cannot be paired with an
    # earlier process identity merely because it reused the same numeric PID.
    after_stat, after_start_ticks = _capture_server_stat(
        target.server_pid,
        target.server_start_ticks,
    )
    if after_start_ticks != target.server_start_ticks:
        _fail("server start ticks drifted after endpoint capture")

    raw_prefix = f"{request.capture_name}/raw/{sequence:06d}"
    document = {
        "schema_version": SAMPLE_VERSION,
        "sequence": sequence,
        "elapsed_monotonic_millis": elapsed_monotonic_millis,
        "endpoint": {
            "http_status": 200,
            "body": _descriptor(f"{raw_prefix}.metrics.json", metrics),
            "listener": {
                "address": "127.0.0.1",
                "port": request.endpoint.port,
                "socket_inode": listener_before.socket_inode,
                "before_proc_net_tcp": _descriptor(
                    f"{raw_prefix}.proc-net-tcp-before",
                    listener_before.proc_net_tcp,
                ),
                "after_proc_net_tcp": _descriptor(
                    f"{raw_prefix}.proc-net-tcp-after",
                    listener_after.proc_net_tcp,
                ),
                "before_server_fd_sockets": _descriptor(
                    f"{raw_prefix}.proc-fd-sockets-before.json",
                    before_sockets,
                ),
                "after_server_fd_sockets": _descriptor(
                    f"{raw_prefix}.proc-fd-sockets-after.json",
                    after_sockets,
                ),
            },
        },
        "process": {
            "pid": target.server_pid,
            "start_ticks": target.server_start_ticks,
            "present": True,
            "pre_endpoint_stat": _descriptor(f"{raw_prefix}.proc-stat-before", before_stat),
            "stat": _descriptor(f"{raw_prefix}.proc-stat", after_stat),
            "status": _descriptor(f"{raw_prefix}.proc-status", status),
        },
        "gpu": {
            "index": target.gpu_index,
            "uuid": target.gpu_uuid,
            "selection_query": _descriptor(f"{raw_prefix}.gpu-selection.csv", selection),
            "compute_apps": _descriptor(f"{raw_prefix}.gpu-compute-apps.csv", compute_apps),
        },
    }
    return CapturedSample(
        raw_files=(
            (f"{sequence:06d}.metrics.json", metrics),
            (f"{sequence:06d}.proc-stat-before", before_stat),
            (f"{sequence:06d}.proc-net-tcp-before", listener_before.proc_net_tcp),
            (f"{sequence:06d}.proc-fd-sockets-before.json", before_sockets),
            (f"{sequence:06d}.proc-net-tcp-after", listener_after.proc_net_tcp),
            (f"{sequence:06d}.proc-fd-sockets-after.json", after_sockets),
            (f"{sequence:06d}.proc-stat", after_stat),
            (f"{sequence:06d}.proc-status", status),
            (f"{sequence:06d}.gpu-selection.csv", selection),
            (f"{sequence:06d}.gpu-compute-apps.csv", compute_apps),
        ),
        document=document,
    )


def _open_capture_directories(
    request: CaptureRequest,
    *,
    repository_root: Path,
    incomplete_marker: bytes,
) -> CaptureDirectories:
    _validate_leaf_name(request.capture_name, "--capture-name")
    _assert_external_to_repository(request.evidence_root, repository_root)
    root_fd = _open_private_evidence_root(request.evidence_root, "--evidence-root")
    capture_fd: int | None = None
    raw_fd: int | None = None
    samples_fd: int | None = None
    ready = False
    try:
        capture_fd = _require_new_private_directory(root_fd, request.capture_name, "capture directory")
        _acquire_capture_lock(capture_fd)
        # Establish the incomplete state before this fresh capture gains any
        # other durable child.  A later initialization or host-capture error
        # is then visibly nonterminal to the v2 binder.
        _write_new(capture_fd, INCOMPLETE_MARKER_NAME, incomplete_marker)
        raw_fd = _require_new_private_directory(capture_fd, "raw", "raw evidence directory")
        samples_fd = _require_new_private_directory(capture_fd, "samples", "sample evidence directory")
        ready = True
        return CaptureDirectories(root_fd, capture_fd, raw_fd, samples_fd)
    finally:
        if not ready:
            _close_quietly(samples_fd)
            _close_quietly(raw_fd)
            _close_quietly(capture_fd)
            _close_quietly(root_fd)


def _incomplete_marker_raw() -> bytes:
    return _canonical(
        {
            "schema_version": INCOMPLETE_MARKER_VERSION,
            "capture_status": "incomplete",
            "qualification_status": "not-run",
        }
    )


def capture_observations(
    request: CaptureRequest,
    *,
    repository_root: Path,
    sleep: Callable[[float], None] = time.sleep,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, Any]:
    """Write a fresh raw v2 capture and remove its incomplete marker last."""

    marker = _incomplete_marker_raw()
    target = _preflight_target(request)
    directories = _open_capture_directories(
        request,
        repository_root=repository_root,
        incomplete_marker=marker,
    )
    previous_elapsed: int | None = None
    sample_descriptors: list[dict[str, Any]] = []
    started = monotonic_ns()
    try:
        for sequence in range(request.sample_count):
            elapsed = (monotonic_ns() - started) // 1_000_000
            if previous_elapsed is not None and elapsed <= previous_elapsed:
                _fail("monotonic clock did not advance between observation samples")
            captured = _capture_one(
                request,
                target,
                sequence=sequence,
                elapsed_monotonic_millis=elapsed,
            )
            for name, raw in captured.raw_files:
                _write_new(directories.raw_fd, name, raw)
            sample_name = f"{sequence:06d}.json"
            sample_raw = _canonical(captured.document)
            _write_new(directories.samples_fd, sample_name, sample_raw)
            sample_descriptors.append(
                _descriptor(f"{request.capture_name}/samples/{sample_name}", sample_raw)
            )
            previous_elapsed = elapsed
            if sequence + 1 < request.sample_count:
                sleep(request.interval_seconds)
        session = {
            "schema_version": SESSION_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "endpoint": {
                "url": request.endpoint.url,
                "expected_schema_version": METRICS_VERSION,
            },
            "target": target.as_json(),
            "samples": sample_descriptors,
        }
        _write_terminal_session(directories.capture_fd, _canonical(session))
        _remove_incomplete_marker(directories.capture_fd, marker)
        return session
    finally:
        _close_quietly(directories.samples_fd)
        _close_quietly(directories.raw_fd)
        _close_quietly(directories.capture_fd)
        _close_quietly(directories.evidence_root_fd)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--server-pid", required=True)
    parser.add_argument("--gpu-index", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--capture-name", required=True)
    parser.add_argument("--interval-seconds", required=True)
    parser.add_argument("--sample-count", required=True)
    return parser


def _request_from_args(args: argparse.Namespace) -> CaptureRequest:
    endpoint = parse_endpoint(args.endpoint)
    server_pid = _parse_positive_option(args.server_pid, "--server-pid", maximum=0x7FFFFFFF)
    gpu_index = _parse_unsigned_option(args.gpu_index, "--gpu-index", maximum=1024)
    interval_seconds = _parse_positive_option(
        args.interval_seconds,
        "--interval-seconds",
        maximum=MAX_INTERVAL_SECONDS,
    )
    sample_count = _parse_positive_option(
        args.sample_count,
        "--sample-count",
        maximum=MAX_SAMPLE_COUNT,
    )
    _validate_leaf_name(args.capture_name, "--capture-name")
    return CaptureRequest(
        endpoint=endpoint,
        server_pid=server_pid,
        gpu_index=gpu_index,
        evidence_root=args.evidence_root,
        capture_name=args.capture_name,
        interval_seconds=interval_seconds,
        sample_count=sample_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        session = capture_observations(
            _request_from_args(args),
            repository_root=_repository_root(),
        )
    except ObservationCaptureError as error:
        print(f"C02 v2 raw observation capture failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(_canonical(session) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
