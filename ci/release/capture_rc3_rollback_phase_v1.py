#!/usr/bin/env python3
"""Capture a raw RC3/RC2 rollback phase without deciding rollback success.

The reconstructed RC2 target has no C02 metrics, source-audit, or shutdown
surface.  This producer captures only the common legacy-compatible facts:
literal loopback ``/readyz`` and optional canonical non-stream
``/v1/completions`` exchanges plus the process/TCP/FD-socket/GPU raw leaves
from which the v3 binder derives a target.  It does not create an evidence
root, launch or stop a service, acquire a GPU lock, rename an artifact, or
make a qualification/rollback decision.

It appends a new create-only capture directory to the already-private root
which contains the reconstructed baseline closure.  Candidate source-audit
generation capture remains the separate source-owned producer; a later runner
uses this helper for its candidate health/host snapshot and that producer's
exact request/response/audit index for candidate generation.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar
from urllib.parse import urlsplit

import check_rc3_rollback_provenance_v3 as rollback_v3
import capture_c02_observations_v2 as c02
import capture_c02_raw_soak_scenarios_v1 as scenarios
import provenance_v2_common as common


sys.dont_write_bytecode = True

SESSION_VERSION = "riley.rc3-rollback-raw-phase-capture.v1"
INCOMPLETE_MARKER_VERSION = "riley.rc3-rollback-raw-phase-capture-incomplete.v1"
INCOMPLETE_MARKER_NAME = "capture-incomplete.json"
MAX_HTTP_HEAD_BYTES = 64 * 1024
MAX_HTTP_BODY_BYTES = 16 * 1024 * 1024

PID_RE = re.compile(r"^[1-9][0-9]*$")
UINT_RE = re.compile(r"^[0-9]+$")

_SESSION_FIELDS = frozenset(
    {
        "schema_version",
        "capture_status",
        "qualification_status",
        "endpoint",
        "target",
        "process_evidence",
        "health",
        "generation",
    }
)
_PROCESS_RAW_FILENAMES = {
    "pre_stat": "pre-stat",
    "post_stat": "post-stat",
    "pre_tcp": "pre-tcp",
    "post_tcp": "post-tcp",
    "pre_fd_sockets": "pre-fd-sockets.json",
    "post_fd_sockets": "post-fd-sockets.json",
    "status": "status",
    "gpu_selection": "gpu-selection.csv",
    "gpu_compute_apps": "gpu-compute-apps.csv",
}
_HEALTH_RAW_FILENAMES = {
    "request": "health-request.http",
    "response_head": "health-response-head.http",
    "response_body": "health-response-body.bin",
}
_GENERATION_RAW_FILENAMES = {
    "request": "generation-request.http",
    "response_head": "generation-response-head.http",
    "response_body": "generation-response-body.bin",
}


class RollbackPhaseCaptureError(ValueError):
    """One rollback phase cannot safely publish raw evidence."""


def _fail(message: str, *, code: str = "invalid-rollback-phase") -> NoReturn:
    error = RollbackPhaseCaptureError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _c02(call: Callable[[], T]) -> T:
    try:
        return call()
    except c02.ObservationCaptureError as error:
        _fail(str(error))


def _scenario(call: Callable[[], T]) -> T:
    try:
        return call()
    except scenarios.RawScenarioCaptureError as error:
        _fail(str(error), code=getattr(error, "reason_code", "invalid-raw-scenario"))


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(str(error), code=getattr(error, "reason_code", "unsafe-evidence"))


def _rollback_v3(call: Callable[[], T]) -> T:
    try:
        return call()
    except rollback_v3.RollbackV3ProvenanceError as error:
        _fail(str(error), code=getattr(error, "reason_code", "invalid-raw-process-evidence"))


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    url: str


@dataclass(frozen=True)
class TargetIdentity:
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
class CaptureRequest:
    endpoint: Endpoint
    server_pid: int
    gpu_index: int
    evidence_root: Path
    capture_name: str
    generation_body: bytes | None


@dataclass(frozen=True)
class CaptureDirectories:
    root_fd: int
    capture_fd: int
    raw_fd: int


@dataclass(frozen=True)
class ReplayedPhaseCapture:
    """One terminal rollback phase replayed through caller-held FDs only.

    This is intentionally a structural/raw replay.  It proves that the
    capture's exact raw inventory, descriptors, HTTP framing, and declared
    target agree; it does not interpret either HTTP response as a rollback
    verdict or operate a service.
    """

    capture_name: str
    endpoint: Endpoint
    target: TargetIdentity
    process_evidence: Mapping[str, common.EvidenceDescriptor]
    health: Mapping[str, common.EvidenceDescriptor]
    generation: Mapping[str, common.EvidenceDescriptor] | None


def parse_endpoint(value: str) -> Endpoint:
    """Accept exactly one literal IPv4 loopback base endpoint."""

    try:
        parsed = urlsplit(value)
    except ValueError as error:
        _fail(f"--endpoint is invalid: {error}")
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname != "127.0.0.1"
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or re.fullmatch(r"127\.0\.0\.1:[0-9]+", parsed.netloc) is None
    ):
        _fail("--endpoint must be literal http://127.0.0.1:PORT")
    try:
        port = parsed.port
    except ValueError as error:
        _fail(f"--endpoint has an invalid port: {error}")
    if port is None or not 1024 <= port <= 65535:
        _fail("--endpoint port must be from 1024 through 65535")
    return Endpoint("127.0.0.1", port, f"http://127.0.0.1:{port}")


def _positive_pid(value: str) -> int:
    if PID_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a positive decimal PID")
    return int(value)


def _nonnegative_index(value: str) -> int:
    if UINT_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a non-negative decimal GPU index")
    return int(value)


def _read_generation_request(path: Path) -> bytes:
    """Read one canonical public completion body through reviewed FD code."""

    raw = _c02(
        lambda: c02._read_absolute_regular(  # noqa: SLF001
            path, "--generation-request", maximum=MAX_HTTP_BODY_BYTES
        )
    )
    return _validate_generation_body(raw, "--generation-request")


def _validate_generation_body(raw: bytes, label: str) -> bytes:
    """Close both CLI and direct-API generation inputs to one public grammar."""

    document = _scenario(
        lambda: scenarios._parse_json(  # noqa: SLF001
            raw, label, maximum=MAX_HTTP_BODY_BYTES, canonical=True
        )
    )
    checked = _scenario(
        lambda: scenarios._completion_request(document, label)  # noqa: SLF001
    )
    if raw != _scenario(lambda: scenarios._canonical(checked)):  # noqa: SLF001
        _fail(f"{label} changed while it was checked")
    return raw


def _request_bytes(method: str, endpoint: Endpoint, target: str, body: bytes) -> bytes:
    if method == "GET":
        return (
            f"GET {target} HTTP/1.1\r\nHost: {endpoint.host}:{endpoint.port}\r\n"
            "Accept: */*\r\nConnection: close\r\n\r\n"
        ).encode("ascii")
    if method == "POST":
        return (
            f"POST {target} HTTP/1.1\r\nHost: {endpoint.host}:{endpoint.port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + body
    _fail("internal unsupported HTTP method")


def _split_head(buffer: bytes) -> tuple[bytes, bytes] | None:
    offset = buffer.find(b"\r\n\r\n")
    if offset < 0:
        return None
    return buffer[: offset + 4], buffer[offset + 4 :]


def _response_length(head: bytes, label: str, *, require_json: bool) -> int:
    """Require one bounded, unchunked HTTP/1.1 success response."""

    if not head.endswith(b"\r\n\r\n") or len(head) > MAX_HTTP_HEAD_BYTES:
        _fail(f"{label} response head is malformed or oversized")
    try:
        lines = head[:-4].decode("ascii").split("\r\n")
    except UnicodeDecodeError as error:
        _fail(f"{label} response head is not ASCII: {error}")
    if not lines or lines[0] != "HTTP/1.1 200 OK":
        _fail(f"{label} endpoint did not return HTTP/1.1 200 OK")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line or line[:1] in {" ", "\t"}:
            _fail(f"{label} response head contains an invalid header")
        name, value = line.split(":", 1)
        normalized = name.lower()
        if re.fullmatch(r"[A-Za-z0-9-]+", name) is None or normalized in headers:
            _fail(f"{label} response head repeats or has an invalid header")
        if any(ord(character) < 32 and character != "\t" or ord(character) == 127 for character in value):
            _fail(f"{label} response head has a control character in a header value")
        headers[normalized] = value.strip(" \t")
    if "transfer-encoding" in headers:
        _fail(f"{label} response must not use Transfer-Encoding")
    content_length = headers.get("content-length")
    if content_length is None or UINT_RE.fullmatch(content_length) is None:
        _fail(f"{label} response must have one numeric Content-Length")
    length = int(content_length)
    if not 1 <= length <= MAX_HTTP_BODY_BYTES:
        _fail(f"{label} response body is outside its byte bound")
    if require_json:
        content_type = headers.get("content-type")
        if content_type is None or content_type.split(";", 1)[0].strip().lower() != "application/json":
            _fail(f"{label} response must have an application/json Content-Type")
    return length


def _capture_exchange(
    endpoint: Endpoint,
    *,
    method: str,
    target: str,
    body: bytes,
    label: str,
    require_json: bool,
) -> tuple[bytes, bytes, bytes]:
    """Capture exact emitted request, raw head, and entity body once."""

    request = _request_bytes(method, endpoint, target, body)
    try:
        connection = socket.create_connection((endpoint.host, endpoint.port), timeout=c02.HTTP_TIMEOUT_SECONDS)
    except OSError as error:
        _fail(f"cannot connect to {label} endpoint: {error}")
    with connection:
        try:
            connection.settimeout(c02.HTTP_TIMEOUT_SECONDS)
            connection.sendall(request)
            buffered = bytearray()
            while len(buffered) <= MAX_HTTP_HEAD_BYTES:
                chunk = connection.recv(min(65536, MAX_HTTP_HEAD_BYTES + 1 - len(buffered)))
                if not chunk:
                    _fail(f"{label} response ended before its header")
                buffered.extend(chunk)
                split = _split_head(bytes(buffered))
                if split is not None:
                    head, remainder = split
                    break
            else:
                _fail(f"{label} response head exceeds its byte bound")
            expected = _response_length(head, label, require_json=require_json)
            response = bytearray(remainder)
            if len(response) > expected:
                _fail(f"{label} response contains trailing body bytes")
            while len(response) < expected:
                chunk = connection.recv(min(65536, expected - len(response)))
                if not chunk:
                    _fail(f"{label} response body is truncated")
                response.extend(chunk)
            if connection.recv(1):
                _fail(f"{label} response contains trailing body bytes")
        except OSError as error:
            _fail(f"cannot capture {label} response: {error}")
    return request, head, bytes(response)


def _preflight_target(request: CaptureRequest) -> TargetIdentity:
    _stat, ticks = _c02(lambda: c02._capture_server_stat(request.server_pid, None))  # noqa: SLF001
    listener = _c02(lambda: c02._capture_bound_listener(request.endpoint, request.server_pid))  # noqa: SLF001
    _selection, _apps, gpu_uuid = _c02(
        lambda: c02._capture_gpu(request.gpu_index, request.server_pid, None)  # noqa: SLF001
    )
    return TargetIdentity(
        server_pid=request.server_pid,
        server_start_ticks=ticks,
        listener_port=request.endpoint.port,
        listener_inode=listener.socket_inode,
        gpu_index=request.gpu_index,
        gpu_uuid=gpu_uuid,
    )


def _marker() -> bytes:
    return _c02(
        lambda: c02._canonical(  # noqa: SLF001
            {
                "schema_version": INCOMPLETE_MARKER_VERSION,
                "capture_status": "incomplete",
                "qualification_status": "not-run",
            }
        )
    )


def _repository_root() -> Path:
    """Return the checked source root; factored for CPU-only hostile tests."""

    return Path(__file__).resolve().parents[2]


def _open_capture_directories(request: CaptureRequest, marker: bytes) -> CaptureDirectories:
    _c02(lambda: c02._validate_leaf_name(request.capture_name, "--capture-name"))  # noqa: SLF001
    repository_root = _repository_root()
    _c02(lambda: c02._assert_external_to_repository(request.evidence_root, repository_root))  # noqa: SLF001
    root_fd = _c02(lambda: c02._open_private_evidence_root(request.evidence_root, "--evidence-root"))  # noqa: SLF001
    capture_fd: int | None = None
    raw_fd: int | None = None
    ready = False
    try:
        capture_fd = _c02(
            lambda: c02._require_new_private_directory(  # noqa: SLF001
                root_fd, request.capture_name, "rollback phase capture directory"
            )
        )
        _c02(lambda: c02._acquire_capture_lock(capture_fd))  # noqa: SLF001
        _c02(lambda: c02._write_new(capture_fd, INCOMPLETE_MARKER_NAME, marker))  # noqa: SLF001
        raw_fd = _c02(
            lambda: c02._require_new_private_directory(  # noqa: SLF001
                capture_fd, "raw", "rollback phase raw directory"
            )
        )
        ready = True
        return CaptureDirectories(root_fd, capture_fd, raw_fd)
    finally:
        if not ready:
            c02._close_quietly(raw_fd)  # noqa: SLF001
            c02._close_quietly(capture_fd)  # noqa: SLF001
            c02._close_quietly(root_fd)  # noqa: SLF001


def _descriptor(path: str, raw: bytes) -> dict[str, Any]:
    return _c02(lambda: c02._descriptor(path, raw))  # noqa: SLF001


def _capture_raw_files(
    request: CaptureRequest, target: TargetIdentity
) -> tuple[tuple[tuple[str, bytes], ...], dict[str, Any]]:
    """Capture one identity interval around the fixed public exchanges."""

    pre_stat, pre_ticks = _c02(
        lambda: c02._capture_server_stat(target.server_pid, target.server_start_ticks)  # noqa: SLF001
    )
    if pre_ticks != target.server_start_ticks:
        _fail("server start ticks drifted before rollback phase capture")
    before = _c02(lambda: c02._capture_bound_listener(request.endpoint, target.server_pid))  # noqa: SLF001
    pre_sockets = _c02(
        lambda: c02._socket_snapshot_raw(target.server_pid, before.server_socket_inodes)  # noqa: SLF001
    )
    health_request, health_head, health_body = _capture_exchange(
        request.endpoint,
        method="GET",
        target="/readyz",
        body=b"",
        label="health",
        require_json=False,
    )
    generation: tuple[bytes, bytes, bytes] | None = None
    if request.generation_body is not None:
        generation = _capture_exchange(
            request.endpoint,
            method="POST",
            target="/v1/completions",
            body=request.generation_body,
            label="generation",
            require_json=True,
        )
    after = _c02(lambda: c02._capture_bound_listener(request.endpoint, target.server_pid))  # noqa: SLF001
    post_sockets = _c02(
        lambda: c02._socket_snapshot_raw(target.server_pid, after.server_socket_inodes)  # noqa: SLF001
    )
    if before.socket_inode != target.listener_inode or after.socket_inode != target.listener_inode:
        _fail("loopback listener inode drifted during rollback phase capture")
    selection, apps, uuid = _c02(
        lambda: c02._capture_gpu(target.gpu_index, target.server_pid, target.gpu_uuid)  # noqa: SLF001
    )
    if uuid != target.gpu_uuid:
        _fail("GPU UUID drifted during rollback phase capture")
    status = _c02(lambda: c02._capture_server_status(target.server_pid))  # noqa: SLF001
    post_stat, post_ticks = _c02(
        lambda: c02._capture_server_stat(target.server_pid, target.server_start_ticks)  # noqa: SLF001
    )
    if post_ticks != target.server_start_ticks:
        _fail("server start ticks drifted after rollback phase capture")

    prefix = f"{request.capture_name}/raw"
    raw_files: list[tuple[str, bytes]] = [
        ("pre-stat", pre_stat),
        ("pre-tcp", before.proc_net_tcp),
        ("pre-fd-sockets.json", pre_sockets),
        ("health-request.http", health_request),
        ("health-response-head.http", health_head),
        ("health-response-body.bin", health_body),
        ("post-tcp", after.proc_net_tcp),
        ("post-fd-sockets.json", post_sockets),
        ("status", status),
        ("post-stat", post_stat),
        ("gpu-selection.csv", selection),
        ("gpu-compute-apps.csv", apps),
    ]
    document: dict[str, Any] = {
        "process_evidence": {
            "pre_stat": _descriptor(f"{prefix}/pre-stat", pre_stat),
            "post_stat": _descriptor(f"{prefix}/post-stat", post_stat),
            "pre_tcp": _descriptor(f"{prefix}/pre-tcp", before.proc_net_tcp),
            "post_tcp": _descriptor(f"{prefix}/post-tcp", after.proc_net_tcp),
            "pre_fd_sockets": _descriptor(f"{prefix}/pre-fd-sockets.json", pre_sockets),
            "post_fd_sockets": _descriptor(f"{prefix}/post-fd-sockets.json", post_sockets),
            "status": _descriptor(f"{prefix}/status", status),
            "gpu_selection": _descriptor(f"{prefix}/gpu-selection.csv", selection),
            "gpu_compute_apps": _descriptor(f"{prefix}/gpu-compute-apps.csv", apps),
        },
        "health": {
            "request": _descriptor(f"{prefix}/health-request.http", health_request),
            "response_head": _descriptor(f"{prefix}/health-response-head.http", health_head),
            "response_body": _descriptor(f"{prefix}/health-response-body.bin", health_body),
        },
        "generation": None,
    }
    if generation is not None:
        generation_request, generation_head, generation_body = generation
        raw_files.extend(
            (
                ("generation-request.http", generation_request),
                ("generation-response-head.http", generation_head),
                ("generation-response-body.bin", generation_body),
            )
        )
        document["generation"] = {
            "request": _descriptor(f"{prefix}/generation-request.http", generation_request),
            "response_head": _descriptor(f"{prefix}/generation-response-head.http", generation_head),
            "response_body": _descriptor(f"{prefix}/generation-response-body.bin", generation_body),
        }
    return tuple(raw_files), document


def _exact_object(value: Any, fields: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
            code="unknown-or-missing-field",
        )
    return value


def _positive(value: Any, label: str, *, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        _fail(f"{label} must be a positive integer in range", code="invalid-integer")
    return value


def _nonnegative(value: Any, label: str, *, maximum: int = 2**31 - 1) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        _fail(f"{label} must be a non-negative integer in range", code="invalid-integer")
    return value


def _replay_endpoint(value: Any, label: str) -> Endpoint:
    if type(value) is not str:
        _fail(f"{label} must be a canonical endpoint URL", code="invalid-endpoint")
    endpoint = parse_endpoint(value)
    if endpoint.url != value:
        _fail(f"{label} must be canonical literal loopback text", code="invalid-endpoint")
    return endpoint


def _replay_target(value: Any, label: str) -> TargetIdentity:
    row = _exact_object(
        value,
        {
            "server_pid",
            "server_start_ticks",
            "listener_port",
            "listener_inode",
            "gpu_index",
            "gpu_uuid",
        },
        label,
    )
    gpu_uuid = row["gpu_uuid"]
    if (
        type(gpu_uuid) is not str
        or len(gpu_uuid) > 128
        or c02.GPU_UUID_RE.fullmatch(gpu_uuid) is None
    ):
        _fail(f"{label}.gpu_uuid must be a canonical GPU UUID", code="invalid-gpu-uuid")
    return TargetIdentity(
        server_pid=_positive(row["server_pid"], f"{label}.server_pid"),
        server_start_ticks=_positive(row["server_start_ticks"], f"{label}.server_start_ticks"),
        listener_port=_positive(
            row["listener_port"], f"{label}.listener_port", maximum=65535
        ),
        listener_inode=_positive(row["listener_inode"], f"{label}.listener_inode"),
        gpu_index=_nonnegative(row["gpu_index"], f"{label}.gpu_index"),
        gpu_uuid=gpu_uuid,
    )


def _replay_descriptor_map(
    value: Any,
    filenames: Mapping[str, str],
    *,
    capture_name: str,
    label: str,
) -> dict[str, common.EvidenceDescriptor]:
    row = _exact_object(value, set(filenames), label)
    descriptors: dict[str, common.EvidenceDescriptor] = {}
    for field, filename in filenames.items():
        descriptor = _common(
            lambda field=field: common.parse_descriptor(row[field], f"{label}.{field}")
        )
        _common(
            lambda descriptor=descriptor, field=field, filename=filename: common.rebase_descriptor_to_held_leaf(
                descriptor,
                expected_root_relative_path=f"{capture_name}/raw/{filename}",
                leaf_name=filename,
                label=f"{label}.{field}",
            )
        )
        descriptors[field] = descriptor
    return descriptors


def _consume_replayed_raw_leaf(
    raw_fd: int,
    descriptor: common.EvidenceDescriptor,
    *,
    capture_name: str,
    filename: str,
    label: str,
    maximum_bytes: int,
) -> bytes:
    rebased = _common(
        lambda: common.rebase_descriptor_to_held_leaf(
            descriptor,
            expected_root_relative_path=f"{capture_name}/raw/{filename}",
            leaf_name=filename,
            label=label,
        )
    )
    return _common(
        lambda: common.consume_private_snapshot_descriptor_file(
            raw_fd,
            rebased,
            label,
            lambda raw_file: raw_file.read(),
            maximum_bytes=maximum_bytes,
        )
    )


def _consume_replayed_raw_group(
    raw_fd: int,
    descriptors: Mapping[str, common.EvidenceDescriptor],
    filenames: Mapping[str, str],
    *,
    capture_name: str,
    label: str,
) -> dict[str, bytes]:
    raw: dict[str, bytes] = {}
    for field, filename in filenames.items():
        if filenames is _PROCESS_RAW_FILENAMES:
            maximum = c02.MAX_GPU_BYTES if field.startswith("gpu_") else c02.MAX_PROC_BYTES
        elif field == "response_head":
            maximum = MAX_HTTP_HEAD_BYTES
        elif field == "request":
            maximum = MAX_HTTP_BODY_BYTES + MAX_HTTP_HEAD_BYTES
        else:
            maximum = MAX_HTTP_BODY_BYTES
        raw[field] = _consume_replayed_raw_leaf(
            raw_fd,
            descriptors[field],
            capture_name=capture_name,
            filename=filename,
            label=f"{label}.{field}",
            maximum_bytes=maximum,
        )
    return raw


def _replay_http_exchange(
    raw: Mapping[str, bytes],
    endpoint: Endpoint,
    *,
    method: str,
    target: str,
    label: str,
    require_json: bool,
) -> None:
    request = raw["request"]
    if method == "GET":
        expected_request = _request_bytes("GET", endpoint, target, b"")
    elif method == "POST":
        split = _split_head(request)
        if split is None:
            _fail(f"{label}.request lacks one HTTP header terminator", code="invalid-http-request")
        _head, body = split
        checked_body = _validate_generation_body(body, f"{label}.request body")
        expected_request = _request_bytes("POST", endpoint, target, checked_body)
    else:
        _fail("internal unsupported replay HTTP method")
    if request != expected_request:
        _fail(f"{label}.request does not match the fixed canonical exchange", code="invalid-http-request")
    expected_length = _response_length(raw["response_head"], label, require_json=require_json)
    if len(raw["response_body"]) != expected_length:
        _fail(f"{label}.response_body does not match Content-Length", code="invalid-http-response")


def _assert_exact_directory_inventory(directory_fd: int, expected: set[str], label: str) -> None:
    try:
        actual = set(os.listdir(directory_fd))
    except OSError as error:
        _fail(f"cannot list {label}: {error}", code="unreadable-input")
    if actual != expected:
        _fail(
            f"{label} inventory differs; expected={sorted(expected)}, actual={sorted(actual)}",
            code="unknown-or-missing-leaf",
        )


def replay_rc3_rollback_phase_v1_fd(root_fd: int, capture_name: str) -> ReplayedPhaseCapture:
    """Strictly replay one terminal raw phase below an already-held root FD.

    This helper deliberately has no path-opening wrapper or CLI mode.  A later
    authenticated compositor must call it while retaining its private root FD,
    so no arbitrary raw-path list can be supplied to the bind-request writer.
    """

    _c02(lambda: c02._validate_leaf_name(capture_name, "rollback phase capture name"))  # noqa: SLF001
    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "rollback phase root"))
    capture_fd: int | None = None
    raw_fd: int | None = None
    try:
        capture_fd = _common(
            lambda: common.open_private_child_directory(
                root_fd, capture_name, "rollback phase capture directory"
            )
        )
        raw_fd = _common(
            lambda: common.open_private_child_directory(
                capture_fd, "raw", "rollback phase raw directory"
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd, capture_fd, capture_name, "rollback phase capture directory"
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                capture_fd, raw_fd, "raw", "rollback phase raw directory"
            )
        )
        _assert_exact_directory_inventory(
            capture_fd,
            {"raw", "session.json"},
            "rollback phase capture directory",
        )
        session = _common(
            lambda: common.read_private_canonical_json_leaf(
                capture_fd,
                "session.json",
                "rollback phase session",
            )
        )
        row = _exact_object(session, _SESSION_FIELDS, "rollback phase session")
        if row["schema_version"] != SESSION_VERSION:
            _fail("rollback phase session has an unsupported schema version", code="invalid-schema-version")
        if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
            _fail("rollback phase session has an invalid terminal state", code="invalid-terminal-state")
        endpoint = _replay_endpoint(row["endpoint"], "rollback phase session.endpoint")
        target = _replay_target(row["target"], "rollback phase session.target")
        if target.listener_port != endpoint.port:
            _fail(
                "rollback phase session target port differs from its endpoint",
                code="phase-target-endpoint-mismatch",
            )
        process_evidence = _replay_descriptor_map(
            row["process_evidence"],
            _PROCESS_RAW_FILENAMES,
            capture_name=capture_name,
            label="rollback phase session.process_evidence",
        )
        health = _replay_descriptor_map(
            row["health"],
            _HEALTH_RAW_FILENAMES,
            capture_name=capture_name,
            label="rollback phase session.health",
        )
        generation_value = row["generation"]
        generation: dict[str, common.EvidenceDescriptor] | None
        if generation_value is None:
            generation = None
        else:
            generation = _replay_descriptor_map(
                generation_value,
                _GENERATION_RAW_FILENAMES,
                capture_name=capture_name,
                label="rollback phase session.generation",
            )
        expected_raw = set(_PROCESS_RAW_FILENAMES.values()) | set(_HEALTH_RAW_FILENAMES.values())
        if generation is not None:
            expected_raw.update(_GENERATION_RAW_FILENAMES.values())
        _assert_exact_directory_inventory(raw_fd, expected_raw, "rollback phase raw directory")
        raw_process = _consume_replayed_raw_group(
            raw_fd,
            process_evidence,
            _PROCESS_RAW_FILENAMES,
            capture_name=capture_name,
            label="rollback phase process evidence",
        )
        raw_health = _consume_replayed_raw_group(
            raw_fd,
            health,
            _HEALTH_RAW_FILENAMES,
            capture_name=capture_name,
            label="rollback phase health",
        )
        _replay_http_exchange(
            raw_health,
            endpoint,
            method="GET",
            target="/readyz",
            label="rollback phase health",
            require_json=False,
        )
        if generation is not None:
            raw_generation = _consume_replayed_raw_group(
                raw_fd,
                generation,
                _GENERATION_RAW_FILENAMES,
                capture_name=capture_name,
                label="rollback phase generation",
            )
            _replay_http_exchange(
                raw_generation,
                endpoint,
                method="POST",
                target="/v1/completions",
                label="rollback phase generation",
                require_json=True,
            )
        observed = _rollback_v3(
            lambda: rollback_v3.derive_phase_target_from_raw_bytes(
                raw_process,
                "rollback phase raw target",
            )
        )
        if observed.as_json() != target.as_json():
            _fail(
                "rollback phase session target differs from held raw process evidence",
                code="phase-target-raw-mismatch",
            )
        _common(lambda: common.require_private_evidence_directory_fd(root_fd, "rollback phase root"))
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd, capture_fd, capture_name, "rollback phase capture directory"
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                capture_fd, raw_fd, "raw", "rollback phase raw directory"
            )
        )
        _assert_exact_directory_inventory(
            capture_fd,
            {"raw", "session.json"},
            "rollback phase capture directory",
        )
        _assert_exact_directory_inventory(raw_fd, expected_raw, "rollback phase raw directory")
        return ReplayedPhaseCapture(
            capture_name=capture_name,
            endpoint=endpoint,
            target=target,
            process_evidence=process_evidence,
            health=health,
            generation=generation,
        )
    finally:
        c02._close_quietly(raw_fd)  # noqa: SLF001
        c02._close_quietly(capture_fd)  # noqa: SLF001


def capture_phase(request: CaptureRequest) -> dict[str, Any]:
    """Publish one create-only phase session and remove its marker last."""

    if request.generation_body is not None:
        _validate_generation_body(request.generation_body, "generation body")
    marker = _marker()
    target = _preflight_target(request)
    directories = _open_capture_directories(request, marker)
    try:
        raw_files, captured = _capture_raw_files(request, target)
        for name, raw in raw_files:
            _c02(lambda name=name, raw=raw: c02._write_new(directories.raw_fd, name, raw))  # noqa: SLF001
        session = {
            "schema_version": SESSION_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "endpoint": request.endpoint.url,
            "target": target.as_json(),
            **captured,
        }
        raw_session = _c02(lambda: c02._canonical(session))  # noqa: SLF001
        _c02(lambda: c02._write_terminal_session(directories.capture_fd, raw_session))  # noqa: SLF001
        _c02(lambda: c02._remove_incomplete_marker(directories.capture_fd, marker))  # noqa: SLF001
        return session
    finally:
        c02._close_quietly(directories.raw_fd)  # noqa: SLF001
        c02._close_quietly(directories.capture_fd)  # noqa: SLF001
        c02._close_quietly(directories.root_fd)  # noqa: SLF001


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, type=parse_endpoint)
    parser.add_argument("--server-pid", required=True, type=_positive_pid)
    parser.add_argument("--gpu-index", required=True, type=_nonnegative_index)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--capture-name", required=True)
    parser.add_argument(
        "--generation-request",
        type=Path,
        help="optional absolute canonical non-stream /v1/completions JSON body",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        generation_body = _read_generation_request(args.generation_request) if args.generation_request else None
        session = capture_phase(
            CaptureRequest(
                endpoint=args.endpoint,
                server_pid=args.server_pid,
                gpu_index=args.gpu_index,
                evidence_root=args.evidence_root,
                capture_name=args.capture_name,
                generation_body=generation_body,
            )
        )
    except RollbackPhaseCaptureError as error:
        print(f"rollback phase capture: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(_c02(lambda: c02._canonical(session)) + b"\n")  # noqa: SLF001
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
