#!/usr/bin/env python3
"""Raw-provenance binding layer for the proposed C02 soak/rollback gates.

This is intentionally narrower than the eventual semantic soak and rollback
checkers.  Its soak path accepts provenance v3 only: it proves that a manifest
is made from bounded, canonical, create-only raw leaves; reconstructs every
sampled PID/start-tick/listener/GPU tuple and joins it to the separately
observed /v1/config process tuple.  It does not turn a raw capture into a C02
qualification decision, replay Gate E, or interpret workload-specific runtime
event fields.

The schema names and event-log descriptors are deliberately generic so a
native C02 audit producer can evolve its event payload without a Python
wrapper inventing an alternate source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence

import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common


# ``soak-v2`` identifies the reviewed workload contract.  The provenance
# document itself is v3 because v2 did not bind the /v1/config response to
# the observed scenario process/listener tuple.  Do not silently widen that
# closed v2 grammar: qualification code must treat it as historical input.
SOAK_MANIFEST_VERSION = "riley.soak-v2-raw-provenance.v3"
SOAK_COMPLETION_MARKER_VERSION = "riley.soak-v2-raw-provenance-complete.v3"
CONFIG_ENDPOINT_OBSERVATION_VERSION = "riley.c02-config-endpoint-observation.v1"
ROLLBACK_MANIFEST_VERSION = "riley.rc3-rollback-raw-provenance.v2"
OBSERVATION_SESSION_VERSION = "riley.c02-raw-observation-session.v2"
OBSERVATION_SAMPLE_VERSION = "riley.c02-raw-observation-sample.v2"
SOCKET_SNAPSHOT_VERSION = "riley.c02-proc-fd-socket-snapshot.v2"
METRICS_VERSION = "riley.c02-capture-metrics.v2"
SHUTDOWN_VERSION = "riley.c02-shutdown-quiescence.v2"
SHUTDOWN_MARKER_VERSION = "riley.c02-shutdown-quiescence-complete.v2"
SOAK_REPORT_VERSION = "riley.soak-v2-provenance-check.v3"
ROLLBACK_REPORT_VERSION = "riley.rc3-rollback-provenance-check.v2"
STABLE_DEFAULT_PROFILE = "stable-default"
MAX_PERFORMANCE_EXACT_PROFILE = "max-performance-exact"
SOAK_CONFIGURATION_PROFILES = frozenset(
    (STABLE_DEFAULT_PROFILE, MAX_PERFORMANCE_EXACT_PROFILE)
)
ROLLBACK_CONFIGURATION_PROFILES = frozenset((STABLE_DEFAULT_PROFILE,))
MAX_RAW_BYTES = 16 * 1024 * 1024
MAX_RELATIVE_PATH_BYTES = 512
MAX_SAMPLES = 1024
MAX_SCENARIOS = 32
GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
CANDIDATE_RE = re.compile(
    r"^riley-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)-rc[1-9][0-9]*$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UINT_RE = re.compile(r"^[0-9]+$")
ENDPOINT_RE = re.compile(r"^http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1/c02/metrics$")
CONFIG_ENDPOINT_RE = re.compile(r"^http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1/config$")
SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SOAK_TERMINAL_MANIFEST_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")
MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES = 246


class C02ProvenanceError(ValueError):
    """Raw C02 evidence cannot establish a safe provenance binding."""


def _fail(code: str, message: str) -> NoReturn:
    error = C02ProvenanceError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _runtime_config(call: Any) -> Any:
    """Translate the pure P0 contract's errors into this raw-binder domain."""

    try:
        return call()
    except runtime_config.EffectiveRuntimeConfigError as error:
        _fail(getattr(error, "reason_code", "invalid-runtime-config"), str(error))


def _open_private_evidence_root(evidence_root: Path, label: str) -> int:
    """Open the external root through the mandatory v2 private-root guard.

    There is deliberately no compatibility fallback to ``open_absolute_directory``:
    an otherwise safe no-follow tree is still unsuitable when the terminal
    evidence directory is replaceable/readable by another principal.  The
    common primitive pins the 0700, effective-UID-owned root for this entire
    verifier invocation.
    """

    try:
        opener = common.open_private_evidence_directory
    except AttributeError:
        _fail(
            "missing-private-evidence-root-helper",
            "v2 verifier requires common.open_private_evidence_directory",
        )
    return _common(lambda: opener(evidence_root, label))


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail("unexpected-field-set", f"{label} must contain exactly {sorted(fields)}")
    return value


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail("invalid-integer", f"{label} must be a positive integer")
    return value


def _unprivileged_listener_port(value: Any, label: str) -> int:
    """Reject ports the closed bridge producer/schema could never capture."""

    port = _positive(value, label)
    if not 1024 <= port <= 65535:
        _fail("invalid-listener-port", f"{label} must be from 1024 through 65535")
    return port


def _nonnegative(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail("invalid-integer", f"{label} must be a non-negative integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail("invalid-sha256", f"{label} must be a non-zero lowercase SHA-256")
    return value


def _candidate_id(value: Any, label: str) -> str:
    if type(value) is not str or CANDIDATE_RE.fullmatch(value) is None:
        _fail("invalid-candidate-id", f"{label} is not a canonical RC candidate ID")
    return value


def _bindings(
    value: Any,
    label: str,
    *,
    allowed_profiles: frozenset[str],
) -> dict[str, str]:
    row = _exact(
        value,
        {
            "freeze_sha256",
            "base_release_candidate_report_sha256",
            "configuration_profile",
            "configuration_sha256",
        },
        label,
    )
    profile = row["configuration_profile"]
    if type(profile) is not str or profile not in allowed_profiles:
        _fail(
            "invalid-configuration-profile",
            f"{label}.configuration_profile must be one of {sorted(allowed_profiles)}",
        )
    return {
        "freeze_sha256": _sha256(row["freeze_sha256"], f"{label}.freeze_sha256"),
        "base_release_candidate_report_sha256": _sha256(
            row["base_release_candidate_report_sha256"],
            f"{label}.base_release_candidate_report_sha256",
        ),
        "configuration_profile": profile,
        "configuration_sha256": _sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
    }


def _descriptor(value: Any, label: str) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    # Published C02 raw-manifest schemas cap every descriptor path at 512
    # ASCII bytes.  common validates traversal safety, while this layer owns
    # the receipt contract's explicit DoS bound.
    if len(descriptor.path) > MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label}.path exceeds {MAX_RELATIVE_PATH_BYTES} bytes",
        )
    if descriptor.byte_length < 1:
        _fail("empty-evidence-leaf", f"{label} must bind nonempty raw evidence")
    return descriptor


def _reserve(
    descriptor: common.EvidenceDescriptor,
    *,
    label: str,
    used_paths: set[str],
) -> None:
    if descriptor.path in used_paths:
        _fail("duplicate-evidence-path", f"{label} reuses evidence path {descriptor.path!r}")
    used_paths.add(descriptor.path)


def _read_bytes(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
    *,
    maximum_bytes: int = MAX_RAW_BYTES,
) -> bytes:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            descriptor.path,
            label,
            maximum_bytes=maximum_bytes,
        )
    )
    if len(raw) != descriptor.byte_length:
        _fail("evidence-length-mismatch", f"{label} length differs from descriptor")
    if hashlib.sha256(raw).hexdigest() != descriptor.sha256:
        _fail("evidence-hash-mismatch", f"{label} SHA-256 differs from descriptor")
    return raw


def _read_json(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
    *,
    maximum_bytes: int = MAX_RAW_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    raw = _read_bytes(root_fd, descriptor, label, maximum_bytes=maximum_bytes)
    document = _common(
        lambda: common.parse_canonical_json(raw, label, maximum_bytes=maximum_bytes)
    )
    assert isinstance(document, dict)
    return raw, document


def _read_manifest(root_fd: int, relative_path: str, label: str) -> tuple[common.EvidenceDescriptor, dict[str, Any]]:
    relative = _common(lambda: common.validate_relative_path(relative_path, f"{label}.path"))
    if len(relative) > MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label}.path exceeds {MAX_RELATIVE_PATH_BYTES} bytes",
        )
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd, relative, label, maximum_bytes=MAX_RAW_BYTES
        )
    )
    document = _common(lambda: common.parse_canonical_json(raw, label, maximum_bytes=MAX_RAW_BYTES))
    assert isinstance(document, dict)
    return common.descriptor_for_bytes(relative, raw, label), document


@dataclass(frozen=True)
class TargetTuple:
    pid: int
    start_ticks: int
    gpu_index: int
    gpu_uuid: str

    def as_json(self) -> dict[str, Any]:
        return {
            "server_pid": self.pid,
            "server_start_ticks": self.start_ticks,
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
        }


@dataclass(frozen=True)
class ObservedTarget:
    """A target reconstructed from raw leaves, including listener identity.

    ``TargetTuple`` deliberately remains the public scenario-request shape so
    existing v2 observation sessions do not need to claim a listener inode at
    authoring time.  The verifier derives that fifth identity component from
    every raw observation sample instead.
    """

    target: TargetTuple
    listener_port: int
    listener_inode: int

    def as_json(self) -> dict[str, Any]:
        return {
            **self.target.as_json(),
            "listener_port": self.listener_port,
            "listener_inode": self.listener_inode,
        }


def _target(value: Any, label: str) -> TargetTuple:
    row = _exact(value, {"server_pid", "server_start_ticks", "gpu_index", "gpu_uuid"}, label)
    gpu_uuid = row["gpu_uuid"]
    if type(gpu_uuid) is not str or GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        _fail("invalid-gpu-uuid", f"{label}.gpu_uuid is invalid")
    return TargetTuple(
        pid=_positive(row["server_pid"], f"{label}.server_pid"),
        start_ticks=_positive(row["server_start_ticks"], f"{label}.server_start_ticks"),
        gpu_index=_nonnegative(row["gpu_index"], f"{label}.gpu_index"),
        gpu_uuid=gpu_uuid,
    )


def _parse_proc_stat(raw: bytes, label: str) -> tuple[int, int]:
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("invalid-proc-stat", f"{label} is not ASCII: {error}")
    closing = value.rfind(")")
    if closing < 2 or closing + 2 >= len(value):
        _fail("invalid-proc-stat", f"{label} has an invalid comm field")
    pid_text = value[: value.find(" ")]
    fields = value[closing + 2 :].split()
    if UINT_RE.fullmatch(pid_text) is None or len(fields) <= 19 or UINT_RE.fullmatch(fields[19]) is None:
        _fail("invalid-proc-stat", f"{label} lacks a valid PID/start tick tuple")
    pid = int(pid_text)
    start_ticks = int(fields[19])
    if pid < 1 or start_ticks < 1:
        _fail("invalid-proc-stat", f"{label} has a zero PID/start tick")
    return pid, start_ticks


def _parse_listener_inodes(raw: bytes, port: int, label: str) -> set[int]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("invalid-proc-net-tcp", f"{label} is not ASCII: {error}")
    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("sl"):
        _fail("invalid-proc-net-tcp", f"{label} has no valid header")
    expected = f"0100007F:{port:04X}"
    result: set[int] = set()
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 10:
            _fail("invalid-proc-net-tcp", f"{label} has a malformed socket row")
        local, remote, state, inode = fields[1], fields[2], fields[3], fields[9]
        if local.upper() == f"00000000:{port:04X}" and remote.upper() == "00000000:0000" and state.upper() == "0A":
            _fail("wildcard-listener", f"{label} contains a wildcard listener for the C02 port")
        if local.upper() != expected:
            continue
        if remote.upper() != "00000000:0000" or state.upper() != "0A" or UINT_RE.fullmatch(inode) is None:
            continue
        parsed = int(inode)
        if parsed < 1 or parsed in result:
            _fail("invalid-proc-net-tcp", f"{label} has an invalid duplicate listener inode")
        result.add(parsed)
    if len(result) != 1:
        _fail("listener-proof-missing", f"{label} must prove exactly one loopback listener")
    return result


def _parse_socket_snapshot(raw: bytes, expected: TargetTuple, label: str) -> set[int]:
    document = _common(lambda: common.parse_canonical_json(raw, label, maximum_bytes=MAX_RAW_BYTES))
    assert isinstance(document, dict)
    row = _exact(document, {"schema_version", "server_pid", "socket_inodes"}, label)
    if row["schema_version"] != SOCKET_SNAPSHOT_VERSION or _positive(row["server_pid"], f"{label}.server_pid") != expected.pid:
        _fail("socket-proof-mismatch", f"{label} has another schema or PID")
    values = row["socket_inodes"]
    if not isinstance(values, list) or not values:
        _fail("socket-proof-mismatch", f"{label}.socket_inodes must be a nonempty array")
    parsed = [_positive(item, f"{label}.socket_inodes") for item in values]
    if parsed != sorted(set(parsed)):
        _fail("socket-proof-mismatch", f"{label}.socket_inodes must be sorted and unique")
    return set(parsed)


def _parse_gpu_selection(raw: bytes, label: str) -> tuple[int, str]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail("invalid-gpu-selection", f"{label} is not ASCII: {error}")
    if len(lines) != 1:
        _fail("invalid-gpu-selection", f"{label} must have exactly one index/UUID row")
    cells = [cell.strip() for cell in lines[0].split(",")]
    if len(cells) != 2 or UINT_RE.fullmatch(cells[0]) is None or GPU_UUID_RE.fullmatch(cells[1]) is None:
        _fail("invalid-gpu-selection", f"{label} is malformed")
    index = int(cells[0])
    if index < 0:
        _fail("invalid-gpu-selection", f"{label} has a negative GPU index")
    return index, cells[1]


def _parse_compute_apps(raw: bytes, expected_pid: int, label: str) -> None:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail("invalid-gpu-compute-apps", f"{label} is not ASCII: {error}")
    rows = 0
    seen: set[int] = set()
    for line in lines:
        if not line:
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 2 or UINT_RE.fullmatch(cells[0]) is None or UINT_RE.fullmatch(cells[1]) is None:
            _fail("invalid-gpu-compute-apps", f"{label} has a malformed row")
        pid = int(cells[0])
        if pid < 1 or pid in seen:
            _fail("invalid-gpu-compute-apps", f"{label} repeats a PID")
        seen.add(pid)
        if pid == expected_pid:
            rows += 1
    if rows != 1:
        _fail("gpu-process-binding-mismatch", f"{label} must contain exactly one target PID row")


def _parse_proc_status_pid(raw: bytes, expected_pid: int, label: str) -> None:
    """Bind the raw `/proc/<pid>/status` leaf to the already-derived PID."""

    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail("invalid-proc-status", f"{label} is not ASCII: {error}")
    parsed_pid: int | None = None
    for line in lines:
        if not line.startswith("Pid:"):
            continue
        if parsed_pid is not None:
            _fail("invalid-proc-status", f"{label} repeats Pid")
        value = line[4:].strip()
        if UINT_RE.fullmatch(value) is None or int(value) < 1:
            _fail("invalid-proc-status", f"{label} has an invalid Pid")
        parsed_pid = int(value)
    if parsed_pid != expected_pid:
        _fail("pid-start-tick-mismatch", f"{label} does not bind the target PID")


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail("invalid-boolean", f"{label} must be a boolean")
    return value


def _metrics(raw: bytes, label: str) -> None:
    """Validate only the raw C02 metric shape, never a pass/fail threshold."""

    document = _common(lambda: common.parse_canonical_json(raw, label, maximum_bytes=MAX_RAW_BYTES))
    assert isinstance(document, dict)
    row = _exact(document, {"schema_version", "request_states", "kv_blocks", "allocation", "quiescence"}, label)
    if row["schema_version"] != METRICS_VERSION:
        _fail("unsupported-metrics-version", f"{label} must use {METRICS_VERSION}")
    states = _exact(
        row["request_states"],
        {"active", "pending_requests", "completed", "failed", "cancelled", "capacity_rejections"},
        f"{label}.request_states",
    )
    for name in states:
        _nonnegative(states[name], f"{label}.request_states.{name}")
    kv = _exact(row["kv_blocks"], {"free", "reserved", "active"}, f"{label}.kv_blocks")
    for name in kv:
        _nonnegative(kv[name], f"{label}.kv_blocks.{name}")
    allocation = _exact(
        row["allocation"],
        {"device_live_count", "device_live_bytes", "pinned_live_count", "pinned_live_bytes"},
        f"{label}.allocation",
    )
    for name in allocation:
        _nonnegative(allocation[name], f"{label}.allocation.{name}")
    quiescence = _exact(
        row["quiescence"],
        {"completion_outbox", "outstanding_iterations", "riley_owned_live_allocations", "worker_accepting", "scheduler_accepting"},
        f"{label}.quiescence",
    )
    for name in ("completion_outbox", "outstanding_iterations", "riley_owned_live_allocations"):
        _nonnegative(quiescence[name], f"{label}.quiescence.{name}")
    for name in ("worker_accepting", "scheduler_accepting"):
        _boolean(quiescence[name], f"{label}.quiescence.{name}")


def _assert_capture_marker_absent(root_fd: int, session_path: str, label: str) -> None:
    parent = PurePosixPath(session_path).parent
    if parent == PurePosixPath("."):
        _fail("invalid-session-path", f"{label} must live below one capture directory")
    marker = f"{parent.as_posix()}/capture-incomplete.json"
    try:
        common.read_bounded_regular_relative(root_fd, marker, label, maximum_bytes=MAX_RAW_BYTES)
    except common.ProvenanceV2Error as error:
        if getattr(error, "reason_code", None) == "missing-input":
            return
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))
    _fail("incomplete-capture", f"{label} retains its incomplete marker")


def _load_session(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
    used_paths: set[str],
) -> ObservedTarget:
    _assert_capture_marker_absent(root_fd, descriptor.path, label)
    _raw, document = _read_json(root_fd, descriptor, label)
    row = _exact(
        document,
        {"schema_version", "capture_status", "qualification_status", "endpoint", "target", "samples"},
        label,
    )
    if row["schema_version"] != OBSERVATION_SESSION_VERSION:
        _fail("historical-observation-version-rejected", f"{label} must use {OBSERVATION_SESSION_VERSION}")
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
        _fail("invalid-capture-status", f"{label} must be raw captured/not-run evidence")
    endpoint = _exact(row["endpoint"], {"url", "expected_schema_version"}, f"{label}.endpoint")
    if type(endpoint["url"]) is not str or ENDPOINT_RE.fullmatch(endpoint["url"]) is None:
        _fail("invalid-endpoint", f"{label}.endpoint.url must be literal loopback C02 metrics")
    if endpoint["expected_schema_version"] != METRICS_VERSION:
        _fail("unsupported-metrics-version", f"{label}.endpoint expected metrics schema drifted")
    target = _target(row["target"], f"{label}.target")
    samples = row["samples"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        _fail("invalid-sample-inventory", f"{label}.samples must be a bounded nonempty array")
    sample_descriptors = [_descriptor(item, f"{label}.samples[{index}]") for index, item in enumerate(samples)]
    _common(lambda: common.require_unique_descriptors(sample_descriptors, f"{label}.samples"))
    prefix = PurePosixPath(descriptor.path).parent.as_posix()
    previous_elapsed_millis: int | None = None
    observed_listener: tuple[int, int] | None = None
    for index, sample_descriptor in enumerate(sample_descriptors):
        if not sample_descriptor.path.startswith(f"{prefix}/samples/"):
            _fail("session-layout-mismatch", f"{label} sample is outside its capture samples directory")
        _reserve(sample_descriptor, label=f"{label}.samples[{index}]", used_paths=used_paths)
        elapsed_millis, listener_port, listener_inode = _load_sample(
            root_fd,
            sample_descriptor,
            target,
            endpoint["url"],
            label,
            used_paths,
            index,
        )
        if previous_elapsed_millis is not None and elapsed_millis <= previous_elapsed_millis:
            _fail(
                "sample-elapsed-not-increasing",
                f"{label}.samples must have strictly increasing elapsed_monotonic_millis",
            )
        previous_elapsed_millis = elapsed_millis
        if observed_listener is None:
            observed_listener = (listener_port, listener_inode)
        elif observed_listener != (listener_port, listener_inode):
            _fail(
                "session-listener-drift",
                f"{label}.samples do not preserve one listener port/inode tuple",
            )
    assert observed_listener is not None
    return ObservedTarget(
        target=target,
        listener_port=observed_listener[0],
        listener_inode=observed_listener[1],
    )


def _load_sample(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    expected_target: TargetTuple,
    endpoint_url: str,
    session_label: str,
    used_paths: set[str],
    expected_sequence: int,
) -> tuple[int, int, int]:
    label = f"{session_label} sample[{expected_sequence}]"
    _raw, document = _read_json(root_fd, descriptor, label)
    row = _exact(
        document,
        {"schema_version", "sequence", "elapsed_monotonic_millis", "endpoint", "process", "gpu"},
        label,
    )
    if row["schema_version"] != OBSERVATION_SAMPLE_VERSION:
        _fail("historical-observation-version-rejected", f"{label} must use {OBSERVATION_SAMPLE_VERSION}")
    if _nonnegative(row["sequence"], f"{label}.sequence") != expected_sequence:
        _fail("sample-sequence-mismatch", f"{label} sequence must be contiguous from zero")
    elapsed_millis = _nonnegative(row["elapsed_monotonic_millis"], f"{label}.elapsed_monotonic_millis")
    endpoint = _exact(row["endpoint"], {"http_status", "body", "listener"}, f"{label}.endpoint")
    if endpoint["http_status"] != 200:
        _fail("endpoint-status-mismatch", f"{label} must preserve HTTP 200 raw evidence")
    listener = _exact(
        endpoint["listener"],
        {"address", "port", "socket_inode", "before_proc_net_tcp", "after_proc_net_tcp", "before_server_fd_sockets", "after_server_fd_sockets"},
        f"{label}.endpoint.listener",
    )
    endpoint_match = ENDPOINT_RE.fullmatch(endpoint_url)
    assert endpoint_match is not None
    expected_port = int(endpoint_match.group(1))
    if listener["address"] != "127.0.0.1" or _positive(listener["port"], f"{label}.endpoint.listener.port") != expected_port:
        _fail("listener-proof-mismatch", f"{label} listener does not bind the session endpoint")
    socket_inode = _positive(listener["socket_inode"], f"{label}.endpoint.listener.socket_inode")

    process = _exact(
        row["process"],
        {"pid", "start_ticks", "present", "pre_endpoint_stat", "stat", "status"},
        f"{label}.process",
    )
    if process["present"] is not True or _positive(process["pid"], f"{label}.process.pid") != expected_target.pid or _positive(process["start_ticks"], f"{label}.process.start_ticks") != expected_target.start_ticks:
        _fail("pid-start-tick-mismatch", f"{label} does not bind the expected process tuple")
    gpu = _exact(
        row["gpu"],
        {"index", "uuid", "selection_query", "compute_apps"},
        f"{label}.gpu",
    )
    if _nonnegative(gpu["index"], f"{label}.gpu.index") != expected_target.gpu_index or gpu["uuid"] != expected_target.gpu_uuid:
        _fail("gpu-tuple-mismatch", f"{label} does not bind the expected GPU tuple")

    descriptors = [
        _descriptor(endpoint["body"], f"{label}.endpoint.body"),
        _descriptor(listener["before_proc_net_tcp"], f"{label}.listener.before_proc_net_tcp"),
        _descriptor(listener["after_proc_net_tcp"], f"{label}.listener.after_proc_net_tcp"),
        _descriptor(listener["before_server_fd_sockets"], f"{label}.listener.before_server_fd_sockets"),
        _descriptor(listener["after_server_fd_sockets"], f"{label}.listener.after_server_fd_sockets"),
        _descriptor(process["pre_endpoint_stat"], f"{label}.process.pre_endpoint_stat"),
        _descriptor(process["stat"], f"{label}.process.stat"),
        _descriptor(process["status"], f"{label}.process.status"),
        _descriptor(gpu["selection_query"], f"{label}.gpu.selection_query"),
        _descriptor(gpu["compute_apps"], f"{label}.gpu.compute_apps"),
    ]
    _common(lambda: common.require_unique_descriptors(descriptors, f"{label} raw leaves"))
    prefix = PurePosixPath(descriptor.path).parent.parent.as_posix()
    for raw_descriptor in descriptors:
        if not raw_descriptor.path.startswith(f"{prefix}/raw/"):
            _fail("session-layout-mismatch", f"{label} raw leaf is outside its capture raw directory")
        _reserve(raw_descriptor, label=label, used_paths=used_paths)

    metric_raw = _read_bytes(root_fd, descriptors[0], f"{label} metrics")
    _metrics(metric_raw, f"{label} metrics")
    before_stat = _read_bytes(root_fd, descriptors[5], f"{label} pre-endpoint stat")
    after_stat = _read_bytes(root_fd, descriptors[6], f"{label} process stat")
    if _parse_proc_stat(before_stat, f"{label} pre-endpoint stat") != (expected_target.pid, expected_target.start_ticks) or _parse_proc_stat(after_stat, f"{label} process stat") != (expected_target.pid, expected_target.start_ticks):
        _fail("pid-start-tick-mismatch", f"{label} raw /proc stat differs from the target tuple")
    _parse_proc_status_pid(
        _read_bytes(root_fd, descriptors[7], f"{label} process status"),
        expected_target.pid,
        f"{label} process status",
    )
    before_tcp = _parse_listener_inodes(_read_bytes(root_fd, descriptors[1], f"{label} proc tcp before"), expected_port, f"{label} proc tcp before")
    after_tcp = _parse_listener_inodes(_read_bytes(root_fd, descriptors[2], f"{label} proc tcp after"), expected_port, f"{label} proc tcp after")
    before_sockets = _parse_socket_snapshot(_read_bytes(root_fd, descriptors[3], f"{label} fd sockets before"), expected_target, f"{label} fd sockets before")
    after_sockets = _parse_socket_snapshot(_read_bytes(root_fd, descriptors[4], f"{label} fd sockets after"), expected_target, f"{label} fd sockets after")
    if before_tcp != {socket_inode} or after_tcp != {socket_inode} or socket_inode not in before_sockets or socket_inode not in after_sockets:
        _fail("listener-proof-mismatch", f"{label} cannot bind endpoint listener to target PID before and after request")
    selected_index, selected_uuid = _parse_gpu_selection(
        _read_bytes(root_fd, descriptors[8], f"{label} GPU selection"),
        f"{label} GPU selection",
    )
    if (selected_index, selected_uuid) != (expected_target.gpu_index, expected_target.gpu_uuid):
        _fail("gpu-tuple-mismatch", f"{label} GPU selection raw leaf drifted")
    _parse_compute_apps(_read_bytes(root_fd, descriptors[9], f"{label} GPU compute apps"), expected_target.pid, f"{label} GPU compute apps")
    return elapsed_millis, expected_port, socket_inode


def _read_opaque_leaf(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    label: str,
    used_paths: set[str],
) -> None:
    _reserve(descriptor, label=label, used_paths=used_paths)
    _read_bytes(root_fd, descriptor, label)


def _parse_config_request(raw: bytes, port: int, label: str) -> None:
    """Require the exact loopback request bytes emitted by the bridge producer."""

    expected = (
        f"GET /v1/config HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Accept: application/json\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    if raw != expected:
        _fail(
            "config-request-mismatch",
            f"{label} is not the exact canonical loopback GET /v1/config request",
        )


def _parse_config_response_head(raw: bytes, label: str) -> int:
    """Return the exact declared body length from a captured HTTP/1.1 head."""

    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("invalid-config-response-head", f"{label} is not ASCII: {error}")
    if not text.endswith("\r\n\r\n"):
        _fail("invalid-config-response-head", f"{label} lacks a complete HTTP header terminator")
    lines = text[:-4].split("\r\n")
    if not lines or lines[0] != "HTTP/1.1 200 OK":
        _fail("config-response-status-mismatch", f"{label} must be HTTP/1.1 200 OK")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line or line[:1] in {" ", "\t"}:
            _fail("invalid-config-response-head", f"{label} has a malformed header")
        name, value = line.split(":", 1)
        lowered = name.lower()
        if lowered in headers or not re.fullmatch(r"[A-Za-z0-9-]+", name):
            _fail("invalid-config-response-head", f"{label} repeats or has an invalid header name")
        headers[lowered] = value.strip()
    length = headers.get("content-length")
    content_type = headers.get("content-type")
    if "transfer-encoding" in headers:
        _fail("invalid-config-response-head", f"{label} must not use Transfer-Encoding")
    if length is None or UINT_RE.fullmatch(length) is None:
        _fail("invalid-config-response-head", f"{label} lacks one exact Content-Length")
    parsed_length = int(length)
    if parsed_length < 1 or parsed_length > runtime_config.MAX_JSON_BYTES:
        _fail("invalid-config-response-head", f"{label} Content-Length is out of bounds")
    if content_type is None or content_type.split(";", 1)[0].strip().lower() != "application/json":
        _fail("invalid-config-response-head", f"{label} Content-Type is not application/json")
    return parsed_length


def _config_observed_target(value: Any, label: str) -> ObservedTarget:
    row = _exact(
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
    target = _target(
        {
            "server_pid": row["server_pid"],
            "server_start_ticks": row["server_start_ticks"],
            "gpu_index": row["gpu_index"],
            "gpu_uuid": row["gpu_uuid"],
        },
        label,
    )
    port = _unprivileged_listener_port(row["listener_port"], f"{label}.listener_port")
    return ObservedTarget(
        target=target,
        listener_port=port,
        listener_inode=_positive(row["listener_inode"], f"{label}.listener_inode"),
    )


def _load_config_endpoint_observation(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    endpoint_descriptor: common.EvidenceDescriptor,
    *,
    used_paths: set[str],
) -> ObservedTarget:
    """Replay a raw /v1/config response and its process/listener proof.

    The endpoint body descriptor belongs only to ``configuration_evidence``.
    The bridge records its hash and length instead of another descriptor so a
    leaf can never be aliased under two independent evidence meanings.
    """

    bridge_path = PurePosixPath(descriptor.path)
    if bridge_path.name != "session.json" or bridge_path.parent == PurePosixPath("."):
        _fail(
            "config-observation-layout-mismatch",
            "configuration endpoint observation must be a capture/session.json leaf",
        )
    _assert_capture_marker_absent(
        root_fd,
        descriptor.path,
        "soak raw configuration endpoint observation",
    )
    _reserve(descriptor, label="soak raw configuration endpoint observation", used_paths=used_paths)
    _raw, document = _read_json(
        root_fd,
        descriptor,
        "soak raw configuration endpoint observation",
    )
    row = _exact(
        document,
        {"schema_version", "capture_status", "qualification_status", "target", "endpoint", "process", "gpu"},
        "soak raw configuration endpoint observation",
    )
    if row["schema_version"] != CONFIG_ENDPOINT_OBSERVATION_VERSION:
        _fail(
            "historical-config-endpoint-observation-rejected",
            "soak raw configuration endpoint observation has an unsupported schema version",
        )
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
        _fail(
            "invalid-capture-status",
            "soak raw configuration endpoint observation must be captured/not-run",
        )
    target = _config_observed_target(row["target"], "soak raw configuration endpoint observation.target")
    endpoint = _exact(
        row["endpoint"],
        {"method", "request_target", "http_status", "request", "response_head", "body_sha256", "body_byte_length", "listener"},
        "soak raw configuration endpoint observation.endpoint",
    )
    if endpoint["method"] != "GET" or endpoint["request_target"] != "/v1/config" or endpoint["http_status"] != 200:
        _fail("config-endpoint-shape-mismatch", "configuration bridge must capture GET /v1/config HTTP 200")
    if _sha256(endpoint["body_sha256"], "soak raw configuration endpoint observation.endpoint.body_sha256") != endpoint_descriptor.sha256:
        _fail("config-endpoint-body-hash-mismatch", "configuration bridge body hash differs from the bound endpoint bytes")
    if _positive(endpoint["body_byte_length"], "soak raw configuration endpoint observation.endpoint.body_byte_length") != endpoint_descriptor.byte_length:
        _fail("config-endpoint-body-length-mismatch", "configuration bridge body length differs from the bound endpoint bytes")
    listener = _exact(
        endpoint["listener"],
        {"address", "port", "socket_inode", "before_proc_net_tcp", "after_proc_net_tcp", "before_server_fd_sockets", "after_server_fd_sockets"},
        "soak raw configuration endpoint observation.endpoint.listener",
    )
    if (
        listener["address"] != "127.0.0.1"
        or _unprivileged_listener_port(listener["port"], "soak raw configuration endpoint observation.endpoint.listener.port") != target.listener_port
        or _positive(listener["socket_inode"], "soak raw configuration endpoint observation.endpoint.listener.socket_inode") != target.listener_inode
    ):
        _fail("config-listener-target-mismatch", "configuration bridge listener differs from its target tuple")
    process = _exact(
        row["process"],
        {"server_pid", "server_start_ticks", "pre_endpoint_stat", "post_endpoint_stat", "status"},
        "soak raw configuration endpoint observation.process",
    )
    if (
        _positive(process["server_pid"], "soak raw configuration endpoint observation.process.server_pid") != target.target.pid
        or _positive(process["server_start_ticks"], "soak raw configuration endpoint observation.process.server_start_ticks") != target.target.start_ticks
    ):
        _fail("config-process-target-mismatch", "configuration bridge process differs from its target tuple")
    gpu = _exact(
        row["gpu"],
        {"index", "uuid", "selection_query", "compute_apps"},
        "soak raw configuration endpoint observation.gpu",
    )
    if _nonnegative(gpu["index"], "soak raw configuration endpoint observation.gpu.index") != target.target.gpu_index or gpu["uuid"] != target.target.gpu_uuid:
        _fail("config-gpu-target-mismatch", "configuration bridge GPU differs from its target tuple")
    leaves = [
        _descriptor(endpoint["request"], "soak raw configuration endpoint observation.endpoint.request"),
        _descriptor(endpoint["response_head"], "soak raw configuration endpoint observation.endpoint.response_head"),
        _descriptor(listener["before_proc_net_tcp"], "soak raw configuration endpoint observation.listener.before_proc_net_tcp"),
        _descriptor(listener["after_proc_net_tcp"], "soak raw configuration endpoint observation.listener.after_proc_net_tcp"),
        _descriptor(listener["before_server_fd_sockets"], "soak raw configuration endpoint observation.listener.before_server_fd_sockets"),
        _descriptor(listener["after_server_fd_sockets"], "soak raw configuration endpoint observation.listener.after_server_fd_sockets"),
        _descriptor(process["pre_endpoint_stat"], "soak raw configuration endpoint observation.process.pre_endpoint_stat"),
        _descriptor(process["post_endpoint_stat"], "soak raw configuration endpoint observation.process.post_endpoint_stat"),
        _descriptor(process["status"], "soak raw configuration endpoint observation.process.status"),
        _descriptor(gpu["selection_query"], "soak raw configuration endpoint observation.gpu.selection_query"),
        _descriptor(gpu["compute_apps"], "soak raw configuration endpoint observation.gpu.compute_apps"),
    ]
    _common(lambda: common.require_unique_descriptors(leaves, "configuration endpoint bridge raw leaves"))
    raw_prefix = f"{bridge_path.parent.as_posix()}/raw/"
    for leaf in leaves:
        if not leaf.path.startswith(raw_prefix):
            _fail(
                "config-observation-layout-mismatch",
                "configuration endpoint observation raw leaf is outside its own capture/raw directory",
            )
        _reserve(leaf, label="configuration endpoint bridge raw leaf", used_paths=used_paths)
    request_raw = _read_bytes(root_fd, leaves[0], "configuration bridge request")
    _parse_config_request(request_raw, target.listener_port, "configuration bridge request")
    response_length = _parse_config_response_head(
        _read_bytes(root_fd, leaves[1], "configuration bridge response head"),
        "configuration bridge response head",
    )
    if response_length != endpoint_descriptor.byte_length:
        _fail("config-endpoint-body-length-mismatch", "configuration response Content-Length differs from endpoint bytes")
    before_stat = _read_bytes(root_fd, leaves[6], "configuration bridge pre-endpoint stat")
    after_stat = _read_bytes(root_fd, leaves[7], "configuration bridge post-endpoint stat")
    expected_process = (target.target.pid, target.target.start_ticks)
    if _parse_proc_stat(before_stat, "configuration bridge pre-endpoint stat") != expected_process or _parse_proc_stat(after_stat, "configuration bridge post-endpoint stat") != expected_process:
        _fail("config-pid-start-tick-mismatch", "configuration bridge raw stat differs from its target tuple")
    _parse_proc_status_pid(
        _read_bytes(root_fd, leaves[8], "configuration bridge process status"),
        target.target.pid,
        "configuration bridge process status",
    )
    before_tcp = _parse_listener_inodes(
        _read_bytes(root_fd, leaves[2], "configuration bridge proc tcp before"),
        target.listener_port,
        "configuration bridge proc tcp before",
    )
    after_tcp = _parse_listener_inodes(
        _read_bytes(root_fd, leaves[3], "configuration bridge proc tcp after"),
        target.listener_port,
        "configuration bridge proc tcp after",
    )
    before_sockets = _parse_socket_snapshot(
        _read_bytes(root_fd, leaves[4], "configuration bridge fd sockets before"),
        target.target,
        "configuration bridge fd sockets before",
    )
    after_sockets = _parse_socket_snapshot(
        _read_bytes(root_fd, leaves[5], "configuration bridge fd sockets after"),
        target.target,
        "configuration bridge fd sockets after",
    )
    if (
        before_tcp != {target.listener_inode}
        or after_tcp != {target.listener_inode}
        or target.listener_inode not in before_sockets
        or target.listener_inode not in after_sockets
    ):
        _fail("config-listener-proof-mismatch", "configuration bridge cannot bind listener to PID before and after response")
    selected_index, selected_uuid = _parse_gpu_selection(
        _read_bytes(root_fd, leaves[9], "configuration bridge GPU selection"),
        "configuration bridge GPU selection",
    )
    if (selected_index, selected_uuid) != (target.target.gpu_index, target.target.gpu_uuid):
        _fail("config-gpu-tuple-mismatch", "configuration bridge GPU selection raw leaf drifted")
    _parse_compute_apps(
        _read_bytes(root_fd, leaves[10], "configuration bridge GPU compute apps"),
        target.target.pid,
        "configuration bridge GPU compute apps",
    )
    return target


def _load_soak_configuration_evidence(
    root_fd: int,
    value: Any,
    *,
    candidate_id: str,
    bindings: Mapping[str, str],
    used_paths: set[str],
) -> ObservedTarget:
    """Bind raw config facts and return their observed process/listener tuple.

    The P0 module owns the JSON/configuration grammar.  This raw provenance
    layer only supplies held-FD bytes, verifies that the startup artifact
    embeds *those* endpoint bytes, and cross-binds the candidate/profile/
    configuration identity before any scenario leaf is considered.
    """

    row = _exact(
        value,
        {"endpoint", "startup_artifact", "endpoint_observation"},
        "soak raw manifest.configuration_evidence",
    )
    endpoint_descriptor = _descriptor(
        row["endpoint"], "soak raw manifest.configuration_evidence.endpoint"
    )
    startup_descriptor = _descriptor(
        row["startup_artifact"],
        "soak raw manifest.configuration_evidence.startup_artifact",
    )
    observation_descriptor = _descriptor(
        row["endpoint_observation"],
        "soak raw manifest.configuration_evidence.endpoint_observation",
    )
    _reserve(
        endpoint_descriptor,
        label="soak raw manifest.configuration_evidence.endpoint",
        used_paths=used_paths,
    )
    _reserve(
        startup_descriptor,
        label="soak raw manifest.configuration_evidence.startup_artifact",
        used_paths=used_paths,
    )
    endpoint_raw = _read_bytes(
        root_fd,
        endpoint_descriptor,
        "soak raw configuration endpoint",
        maximum_bytes=runtime_config.MAX_JSON_BYTES,
    )
    startup_raw = _read_bytes(
        root_fd,
        startup_descriptor,
        "soak raw configuration startup artifact",
        maximum_bytes=runtime_config.MAX_JSON_BYTES,
    )
    endpoint_document, endpoint = _runtime_config(
        lambda: runtime_config.validate_endpoint_bytes(
            endpoint_raw, "soak raw configuration endpoint"
        )
    )
    startup_document, startup = _runtime_config(
        lambda: runtime_config.validate_startup_artifact_bytes(
            startup_raw, "soak raw configuration startup artifact"
        )
    )
    endpoint_digest = hashlib.sha256(endpoint_raw).hexdigest()
    if startup.endpoint_payload_sha256 != endpoint_digest:
        _fail(
            "startup-endpoint-hash-mismatch",
            "soak startup artifact does not hash the bound endpoint bytes",
        )
    if startup_document["endpoint_payload"] != endpoint_document:
        _fail(
            "startup-endpoint-payload-mismatch",
            "soak startup artifact does not embed the bound endpoint payload",
        )
    if endpoint.candidate_id != candidate_id or startup.candidate_id != candidate_id:
        _fail(
            "runtime-config-candidate-mismatch",
            "soak configuration evidence candidate differs from the manifest candidate",
        )
    if endpoint.runtime_identity != startup.runtime_identity:
        _fail(
            "runtime-config-identity-mismatch",
            "soak endpoint and startup artifact runtime identities differ",
        )
    identity = endpoint.runtime_identity
    if identity["configuration_profile"] != bindings["configuration_profile"]:
        _fail(
            "runtime-config-profile-mismatch",
            "soak configuration profile differs from the bound runtime identity",
        )
    if identity["configuration_sha256"] != bindings["configuration_sha256"]:
        _fail(
            "runtime-config-sha256-mismatch",
            "soak configuration SHA-256 differs from the bound runtime identity",
        )
    return _load_config_endpoint_observation(
        root_fd,
        observation_descriptor,
        endpoint_descriptor,
        used_paths=used_paths,
    )


def _soak_terminal_manifest_name(manifest_path: str, label: str) -> str:
    """Require a terminal raw soak manifest to be one root child JSON leaf."""

    relative = _common(lambda: common.validate_relative_path(manifest_path, label))
    if (
        "/" in relative
        or len(relative) > MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES
        or SOAK_TERMINAL_MANIFEST_NAME_RE.fullmatch(relative) is None
    ):
        _fail(
            "invalid-terminal-manifest-name",
            f"{label} must be a nonhidden root direct-child .json name of at most "
            f"{MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES} bytes",
        )
    return relative


def _read_soak_completion_marker(
    root_fd: int,
    manifest: common.EvidenceDescriptor,
) -> None:
    """Require the exact durable sibling marker for a terminal soak manifest."""

    manifest_name = _soak_terminal_manifest_name(
        manifest.path, "soak completed manifest path"
    )
    marker_name = f"{manifest_name}.complete"
    try:
        marker_raw = _common(
            lambda: common.read_bounded_regular_relative(
                root_fd,
                marker_name,
                "soak raw manifest completion marker",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
    except C02ProvenanceError as error:
        if getattr(error, "reason_code", None) == "missing-input":
            _fail(
                "missing-soak-completion-marker",
                f"soak raw manifest requires exact sibling marker {marker_name!r}",
            )
        raise
    marker = _common(
        lambda: common.parse_canonical_json(
            marker_raw,
            "soak raw manifest completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    assert isinstance(marker, dict)
    row = _exact(
        marker,
        {"schema_version", "artifact_filename", "artifact_sha256"},
        "soak raw manifest completion marker",
    )
    if row["schema_version"] != SOAK_COMPLETION_MARKER_VERSION:
        _fail(
            "historical-soak-completion-version-rejected",
            "soak raw manifest completion marker has an unsupported schema version",
        )
    if row["artifact_filename"] != manifest_name:
        _fail(
            "soak-completion-marker-mismatch",
            "soak raw manifest completion marker does not bind its artifact leaf",
        )
    if _sha256(
        row["artifact_sha256"], "soak raw manifest completion marker.artifact_sha256"
    ) != manifest.sha256:
        _fail(
            "soak-completion-marker-mismatch",
            "soak raw manifest completion marker does not bind exact manifest bytes",
        )


def _shutdown_completion_marker_basename(
    artifact: common.EvidenceDescriptor,
    marker: common.EvidenceDescriptor,
    label: str,
) -> str:
    """Require the v2 marker to be the artifact's nonhidden sibling leaf."""

    artifact_path = PurePosixPath(artifact.path)
    marker_path = PurePosixPath(marker.path)
    artifact_basename = artifact_path.name
    marker_basename = marker_path.name
    expected_marker_path = f"{artifact.path}.complete"
    if (
        marker.path != expected_marker_path
        or marker_path.parent != artifact_path.parent
        or marker_basename != f"{artifact_basename}.complete"
        or marker_basename.startswith(".")
    ):
        _fail(
            "shutdown-marker-path-mismatch",
            f"{label} marker must be nonhidden direct-child {expected_marker_path!r}",
        )
    return artifact_basename


def _parse_shutdown(
    root_fd: int,
    artifact: common.EvidenceDescriptor,
    marker: common.EvidenceDescriptor,
    expected_target: TargetTuple,
    label: str,
    used_paths: set[str],
) -> None:
    artifact_basename = _shutdown_completion_marker_basename(artifact, marker, label)
    _reserve(artifact, label=f"{label} artifact", used_paths=used_paths)
    _reserve(marker, label=f"{label} marker", used_paths=used_paths)
    raw, document = _read_json(root_fd, artifact, f"{label} artifact")
    row = _exact(
        document,
        {"schema_version", "capture_status", "qualification_status", "server_pid", "server_start_ticks", "worker_ready", "final_metrics"},
        f"{label} artifact",
    )
    if row["schema_version"] != SHUTDOWN_VERSION:
        _fail("historical-shutdown-version-rejected", f"{label} artifact must use {SHUTDOWN_VERSION}")
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run" or row["worker_ready"] is not False:
        _fail("invalid-shutdown-status", f"{label} is not a raw captured shutdown artifact")
    if _positive(row["server_pid"], f"{label}.server_pid") != expected_target.pid or _positive(row["server_start_ticks"], f"{label}.server_start_ticks") != expected_target.start_ticks:
        _fail("shutdown-target-mismatch", f"{label} does not bind the candidate PID/start-tick tuple")
    _metrics(common.canonical_json_bytes(row["final_metrics"]), f"{label}.final_metrics")
    marker_raw, marker_document = _read_json(root_fd, marker, f"{label} marker")
    marker_row = _exact(marker_document, {"schema_version", "artifact_filename", "artifact_sha256"}, f"{label} marker")
    if marker_row["schema_version"] != SHUTDOWN_MARKER_VERSION:
        _fail("historical-shutdown-version-rejected", f"{label} marker must use {SHUTDOWN_MARKER_VERSION}")
    if marker_row["artifact_filename"] != artifact_basename:
        _fail("shutdown-marker-mismatch", f"{label} marker filename does not bind its artifact leaf")
    if _sha256(marker_row["artifact_sha256"], f"{label}.marker.artifact_sha256") != hashlib.sha256(raw).hexdigest():
        _fail("shutdown-marker-mismatch", f"{label} marker does not bind exact shutdown artifact bytes")
    if not marker_raw:
        _fail("shutdown-marker-mismatch", f"{label} marker is empty")


def _report(
    *,
    schema_version: str,
    manifest: common.EvidenceDescriptor,
    candidate_id: str,
    bindings: dict[str, str],
    targets: list[dict[str, Any]],
    check_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "status": "bound",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "bindings": bindings,
        "raw_manifest": manifest.as_json(),
        "targets": targets,
        "checks": [{"name": name, "bound": True} for name in check_names],
        "reason_codes": [],
    }


def verify_soak_provenance_fd(root_fd: int, manifest_path: str) -> dict[str, Any]:
    """Bind raw v3 soak leaves through one caller-held private-root FD.

    This deliberately remains a *raw* verifier.  It does not require a
    terminal manifest marker so focused fixtures can isolate raw-tree
    validation.  Any production/terminal caller must use
    :func:`verify_completed_soak_provenance_fd` instead.
    """

    manifest_descriptor, document = _read_manifest(root_fd, manifest_path, "soak raw manifest")
    row = _exact(
        document,
        {
            "schema_version",
            "capture_status",
            "qualification_status",
            "candidate_id",
            "bindings",
            "configuration_evidence",
            "scenario_contract",
            "scenarios",
        },
        "soak raw manifest",
    )
    if row["schema_version"] != SOAK_MANIFEST_VERSION:
        historical = "historical-soak-v2-rejected" if row["schema_version"] == "riley.soak-v2-raw-provenance.v2" else "historical-soak-v1-rejected"
        _fail(historical, f"soak raw manifest must use {SOAK_MANIFEST_VERSION}")
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
        _fail("invalid-capture-status", "soak raw manifest must be captured/not-run")
    candidate = _candidate_id(row["candidate_id"], "soak raw manifest.candidate_id")
    bindings = _bindings(
        row["bindings"],
        "soak raw manifest.bindings",
        allowed_profiles=SOAK_CONFIGURATION_PROFILES,
    )
    used = {manifest_descriptor.path}
    configuration_target = _load_soak_configuration_evidence(
        root_fd,
        row["configuration_evidence"],
        candidate_id=candidate,
        bindings=bindings,
        used_paths=used,
    )
    contract = _descriptor(row["scenario_contract"], "soak raw manifest.scenario_contract")
    _read_opaque_leaf(root_fd, contract, "soak scenario contract", used)
    scenarios = row["scenarios"]
    if not isinstance(scenarios, list) or not 1 <= len(scenarios) <= MAX_SCENARIOS:
        _fail("invalid-scenario-inventory", "soak raw manifest must contain a bounded nonempty scenario list")
    scenario_ids: set[str] = set()
    targets: list[dict[str, Any]] = []
    for index, item in enumerate(scenarios):
        label = f"soak raw manifest.scenarios[{index}]"
        scenario = _exact(
            item,
            {"scenario_id", "target", "observation_session", "request_ledger", "runtime_event_log", "generation_audit_index", "fallback_event_log"},
            label,
        )
        scenario_id = scenario["scenario_id"]
        if (
            type(scenario_id) is not str
            or len(scenario_id) > 128
            or SCENARIO_ID_RE.fullmatch(scenario_id) is None
            or scenario_id in scenario_ids
        ):
            _fail(
                "invalid-scenario-inventory",
                f"{label}.scenario_id must be a unique canonical scenario identifier",
            )
        scenario_ids.add(scenario_id)
        fallback = scenario["fallback_event_log"]
        if scenario_id == "exact-backend-fallback":
            if bindings["configuration_profile"] != MAX_PERFORMANCE_EXACT_PROFILE:
                _fail(
                    "fallback-profile-mismatch",
                    f"{label} requires {MAX_PERFORMANCE_EXACT_PROFILE}",
                )
            if fallback is None:
                _fail("fallback-raw-leaf-missing", f"{label} lacks raw fallback event evidence")
        declared_target = _target(scenario["target"], f"{label}.target")
        session = _descriptor(scenario["observation_session"], f"{label}.observation_session")
        _reserve(session, label=f"{label}.observation_session", used_paths=used)
        observed_target = _load_session(root_fd, session, label, used)
        if observed_target.target != declared_target:
            _fail("session-target-mismatch", f"{label} declared target differs from raw observation session")
        if observed_target != configuration_target:
            _fail(
                "configuration-scenario-target-mismatch",
                f"{label} does not share the configuration endpoint PID/start-tick/listener/GPU tuple",
            )
        for key in ("request_ledger", "runtime_event_log", "generation_audit_index"):
            _read_opaque_leaf(root_fd, _descriptor(scenario[key], f"{label}.{key}"), f"{label}.{key}", used)
        if fallback is not None:
            _read_opaque_leaf(root_fd, _descriptor(fallback, f"{label}.fallback_event_log"), f"{label}.fallback_event_log", used)
        targets.append({"scenario_id": scenario_id, "target": declared_target.as_json()})
    return _report(
        schema_version=SOAK_REPORT_VERSION,
        manifest=manifest_descriptor,
        candidate_id=candidate,
        bindings=bindings,
        targets=targets,
        check_names=(
            "v3-version-only",
            "canonical-descriptor-binding",
            "capture-marker-closure",
            "pid-start-tick-listener-gpu-binding",
            "raw-workload-and-audit-leaves",
            "runtime-config-candidate-profile-binding",
            "configuration-profile-arm-binding",
            "configuration-endpoint-scenario-process-bridge",
        ),
    )


def verify_soak_provenance(evidence_root: Path, manifest_path: str) -> dict[str, Any]:
    """Path wrapper for the nonterminal raw verifier used by focused tests."""

    root_fd = _open_private_evidence_root(evidence_root, "evidence root")
    try:
        return verify_soak_provenance_fd(root_fd, manifest_path)
    finally:
        os.close(root_fd)


def verify_completed_soak_provenance_fd(root_fd: int, manifest_path: str) -> dict[str, Any]:
    """Verify a terminal raw soak manifest and its exact sibling marker."""

    manifest_name = _soak_terminal_manifest_name(
        manifest_path, "soak completed manifest path"
    )
    report = verify_soak_provenance_fd(root_fd, manifest_name)
    manifest = _descriptor(report["raw_manifest"], "soak completed manifest descriptor")
    _read_soak_completion_marker(root_fd, manifest)
    # The marker is intentionally read after the raw verifier's first manifest
    # read.  Re-read the manifest through the same held root before returning
    # so a replacement between those two operations cannot splice a valid old
    # marker onto a different current manifest.
    final_manifest, _final_document = _read_manifest(
        root_fd, manifest_name, "soak completed manifest final revalidation"
    )
    if final_manifest != manifest:
        _fail(
            "soak-manifest-changed-during-completion-verification",
            "soak manifest changed while its completion marker was verified",
        )
    # Check the marker one final time against the final manifest descriptor as
    # well.  This catches a marker replacement during the manifest
    # revalidation window; the last read still defines the verifier's exact
    # observed state if another writer races after it returns.
    _read_soak_completion_marker(root_fd, final_manifest)
    return report


def verify_completed_soak_provenance(
    evidence_root: Path, manifest_path: str
) -> dict[str, Any]:
    """Path wrapper for the completed-only terminal raw soak verifier."""

    root_fd = _open_private_evidence_root(evidence_root, "evidence root")
    try:
        return verify_completed_soak_provenance_fd(root_fd, manifest_path)
    finally:
        os.close(root_fd)


def verify_rollback_provenance(evidence_root: Path, manifest_path: str) -> dict[str, Any]:
    """Bind v2 rollback raw leaves without trusting a self-authored timeline."""

    root_fd = _open_private_evidence_root(evidence_root, "evidence root")
    try:
        manifest_descriptor, document = _read_manifest(root_fd, manifest_path, "rollback raw manifest")
        row = _exact(
            document,
            {"schema_version", "capture_status", "qualification_status", "candidate_id", "bindings", "candidate", "rollback", "candidate_artifacts", "rollback_artifacts", "atomic_switch"},
            "rollback raw manifest",
        )
        if row["schema_version"] != ROLLBACK_MANIFEST_VERSION:
            _fail("historical-rollback-v1-rejected", f"rollback raw manifest must use {ROLLBACK_MANIFEST_VERSION}")
        if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
            _fail("invalid-capture-status", "rollback raw manifest must be captured/not-run")
        candidate_id = _candidate_id(row["candidate_id"], "rollback raw manifest.candidate_id")
        bindings = _bindings(
            row["bindings"],
            "rollback raw manifest.bindings",
            allowed_profiles=ROLLBACK_CONFIGURATION_PROFILES,
        )
        used = {manifest_descriptor.path}

        def server_phase(value: Any, label: str, *, candidate_phase: bool) -> TargetTuple:
            fields = {"target", "observation_session", "request_ledger", "runtime_event_log", "generation_audit_index"}
            if candidate_phase:
                fields |= {"shutdown_artifact", "shutdown_marker"}
            phase = _exact(value, fields, label)
            target = _target(phase["target"], f"{label}.target")
            session = _descriptor(phase["observation_session"], f"{label}.observation_session")
            _reserve(session, label=f"{label}.observation_session", used_paths=used)
            if _load_session(root_fd, session, label, used).target != target:
                _fail("session-target-mismatch", f"{label} target differs from raw observation session")
            for key in ("request_ledger", "runtime_event_log", "generation_audit_index"):
                _read_opaque_leaf(root_fd, _descriptor(phase[key], f"{label}.{key}"), f"{label}.{key}", used)
            if candidate_phase:
                _parse_shutdown(
                    root_fd,
                    _descriptor(phase["shutdown_artifact"], f"{label}.shutdown_artifact"),
                    _descriptor(phase["shutdown_marker"], f"{label}.shutdown_marker"),
                    target,
                    label,
                    used,
                )
            return target

        candidate_target = server_phase(row["candidate"], "rollback candidate", candidate_phase=True)
        rollback_target = server_phase(row["rollback"], "rollback prior artifact", candidate_phase=False)
        if (candidate_target.pid, candidate_target.start_ticks) == (rollback_target.pid, rollback_target.start_ticks):
            _fail("reused-candidate-process", "rollback must use a distinct PID/start-tick process identity")
        for phase_name in ("candidate_artifacts", "rollback_artifacts"):
            artifacts = _exact(row[phase_name], {"binary", "bundle", "image_inspect"}, phase_name)
            for key in ("binary", "bundle", "image_inspect"):
                _read_opaque_leaf(root_fd, _descriptor(artifacts[key], f"{phase_name}.{key}"), f"{phase_name}.{key}", used)
        switch = _exact(
            row["atomic_switch"],
            {"pre_active_stat", "post_active_stat", "candidate_staged_stat", "rollback_staged_stat", "rename_transcript"},
            "rollback atomic_switch",
        )
        for key in ("pre_active_stat", "post_active_stat", "candidate_staged_stat", "rollback_staged_stat", "rename_transcript"):
            _read_opaque_leaf(root_fd, _descriptor(switch[key], f"rollback atomic_switch.{key}"), f"rollback atomic_switch.{key}", used)
        return _report(
            schema_version=ROLLBACK_REPORT_VERSION,
            manifest=manifest_descriptor,
            candidate_id=candidate_id,
            bindings=bindings,
            targets=[
                {"phase": "candidate", "target": candidate_target.as_json()},
                {"phase": "rollback", "target": rollback_target.as_json()},
            ],
            check_names=(
                "v2-version-only",
                "canonical-descriptor-binding",
                "capture-marker-closure",
                "candidate-and-rollback-process-tuples",
                "shutdown-marker-binding",
                "raw-atomic-switch-material",
                "configuration-profile-arm-binding",
            ),
        )
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("soak", "rollback"))
    parser.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = (
            verify_completed_soak_provenance(args.evidence_root, args.manifest)
            if args.kind == "soak"
            else verify_rollback_provenance(args.evidence_root, args.manifest)
        )
    except (C02ProvenanceError, common.ProvenanceV2Error) as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
