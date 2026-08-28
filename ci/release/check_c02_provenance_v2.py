#!/usr/bin/env python3
"""Raw-provenance binding layer for the proposed C02 soak/rollback gates.

This is intentionally narrower than the eventual semantic soak and rollback
checkers.  Its retained v3 soak path proves bounded, canonical, create-only
raw leaves and joins sampled PID/start-tick/listener/GPU tuples to a separately
observed /v1/config process tuple.  Its separate v4 serial path additionally
replays one source-owned completion-capture session rather than accepting
opaque workload/audit leaves.  Neither path turns a raw capture into a C02
qualification decision, replays Gate E, or interprets workload-specific
runtime-event semantics.

The schema names and event-log descriptors are deliberately generic so a
native C02 audit producer can evolve its event payload without a Python
wrapper inventing an alternate source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
SOAK_V4_MANIFEST_VERSION = "riley.soak-v2-raw-provenance.v4"
SOAK_V4_COMPLETION_MARKER_VERSION = "riley.soak-v2-raw-provenance-complete.v4"
SOAK_V5_MANIFEST_VERSION = "riley.soak-v2-raw-provenance.v5"
SOAK_V5_COMPLETION_MARKER_VERSION = "riley.soak-v2-raw-provenance-complete.v5"
CONFIG_ENDPOINT_OBSERVATION_VERSION = "riley.c02-config-endpoint-observation.v1"
ROLLBACK_MANIFEST_VERSION = "riley.rc3-rollback-raw-provenance.v2"
OBSERVATION_SESSION_VERSION = "riley.c02-raw-observation-session.v2"
OBSERVATION_SAMPLE_VERSION = "riley.c02-raw-observation-sample.v2"
SOCKET_SNAPSHOT_VERSION = "riley.c02-proc-fd-socket-snapshot.v2"
METRICS_VERSION = "riley.c02-capture-metrics.v2"
SHUTDOWN_VERSION = "riley.c02-shutdown-quiescence.v2"
SHUTDOWN_MARKER_VERSION = "riley.c02-shutdown-quiescence-complete.v2"
SOAK_REPORT_VERSION = "riley.soak-v2-provenance-check.v3"
SOAK_V4_REPORT_VERSION = "riley.soak-v2-provenance-check.v4"
SOAK_V5_REPORT_VERSION = "riley.soak-v2-provenance-check.v5"
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
COMPLETION_ENDPOINT_RE = re.compile(r"^http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1/completions$")
SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
COMPLETION_REQUEST_ID_RE = re.compile(r"^cmpl-[A-Za-z0-9_-]{1,123}$")
SOAK_TERMINAL_MANIFEST_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")
MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES = 246
SHUTDOWN_ARTIFACT_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,240}\.json$")
SCENARIO_CAPTURE_SESSION_VERSION = "riley.c02-raw-scenario-capture.v1"
SCENARIO_CAPTURE_CONTRACT_VERSION = "riley.c02-raw-soak-runner-contract.v1"
SCENARIO_CAPTURE_LEDGER_VERSION = "riley.c02-raw-request-ledger.v1"
SCENARIO_CAPTURE_AUDIT_INDEX_VERSION = "riley.c02-generation-audit-index.v1"
SCENARIO_CAPTURE_AUDIT_VERSION = "riley.c02-generation-audit.v2"
SCENARIO_CAPTURE_AUDIT_MARKER_VERSION = "riley.c02-generation-audit-completion.v2"
SCENARIO_FALLBACK_CAPTURE_SESSION_VERSION = "riley.c02-raw-scenario-capture.v2"
SCENARIO_FALLBACK_CAPTURE_CONTRACT_VERSION = "riley.c02-raw-soak-runner-contract.v2"
SCENARIO_FALLBACK_CAPTURE_AUDIT_INDEX_VERSION = "riley.c02-generation-audit-index.v2"
SCENARIO_FALLBACK_EVENT_VERSION = "riley.c02-native-fallback-event.v1"
SCENARIO_FALLBACK_EVENT_MARKER_VERSION = "riley.c02-native-fallback-event-completion.v1"
FALLBACK_SCENARIO_ID = "exact-backend-fallback"
MAX_NATIVE_FALLBACK_SELECTIONS = 65_536
SERIAL_CAPTURE_V1_OPAQUE_SCHEMA_VERSIONS = frozenset(
    (
        SCENARIO_CAPTURE_CONTRACT_VERSION,
        SCENARIO_CAPTURE_LEDGER_VERSION,
        SCENARIO_CAPTURE_AUDIT_INDEX_VERSION,
    )
)


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
) -> bytes:
    _reserve(descriptor, label=label, used_paths=used_paths)
    return _read_bytes(root_fd, descriptor, label)


def reject_v1_serial_capture_opaque_leaf(raw: bytes, label: str) -> None:
    """Keep the historical v3 raw path from consuming the closed v1 producer.

    v3 deliberately treats its workload leaves as opaque, so arbitrary legacy
    text remains valid there.  The newer serial producer has uniquely versioned
    canonical contract, ledger, and audit-index objects; accepting any of
    those in v3 would bypass the v1 session's incomplete-marker and source
    audit replay that only v4 performs.
    """

    try:
        document = common.parse_strict_json(
            raw,
            label,
            maximum_bytes=MAX_RAW_BYTES,
        )
    except common.ProvenanceV2Error:
        return
    assert isinstance(document, dict)
    if document.get("schema_version") in SERIAL_CAPTURE_V1_OPAQUE_SCHEMA_VERSIONS:
        _fail(
            "v3-serial-capture-v1-rejected",
            f"{label} is a v1 serial-capture leaf and must be terminally replayed only by v4",
        )


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
    require_direct_observation_session: bool = False,
    require_gpu_greedy: bool = False,
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
    if require_direct_observation_session:
        _require_direct_v4_session_path(
            observation_descriptor,
            "v4 soak raw manifest.configuration_evidence.endpoint_observation",
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
    if require_gpu_greedy and endpoint.effective_config["sampling_backend"] != "gpu-greedy":
        _fail(
            "effective-sampling-backend-mismatch",
            "native fallback provenance requires effective_config.sampling_backend gpu-greedy",
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
    if SHUTDOWN_ARTIFACT_FILENAME_RE.fullmatch(artifact_basename) is None:
        _fail(
            "invalid-shutdown-artifact-filename",
            f"{label} artifact must be a nonhidden direct-child .json leaf of at most 246 bytes",
        )
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


@dataclass(frozen=True)
class VerifiedC02ShutdownV2:
    """One source-owned shutdown-v2 artifact and its bound completion leaf.

    This is deliberately a raw replay result rather than a qualification
    report.  The descriptors are derived from the held evidence-root FD, so a
    lifecycle supervisor can retain their exact bytes without accepting a
    caller-authored descriptor or target tuple.
    """

    artifact: common.EvidenceDescriptor
    marker: common.EvidenceDescriptor
    target: TargetTuple


def _verify_c02_shutdown_v2_descriptors_fd(
    root_fd: int,
    artifact: common.EvidenceDescriptor,
    marker: common.EvidenceDescriptor,
    expected_target: TargetTuple,
    label: str,
    used_paths: set[str],
) -> VerifiedC02ShutdownV2:
    """Replay declared shutdown descriptors without weakening manifest binding.

    Manifest consumers already own their declared descriptors and collision
    set, so they use this shared core instead of deriving replacement
    descriptors from paths.  The public path API below derives those
    descriptors only after it has pinned and validated a private root.
    """

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
    return VerifiedC02ShutdownV2(
        artifact=artifact,
        marker=marker,
        target=expected_target,
    )


def _shutdown_descriptor_from_path(
    root_fd: int,
    path: str,
    label: str,
) -> common.EvidenceDescriptor:
    """Derive one bounded shutdown descriptor through the caller-held root."""

    relative = _common(lambda: common.validate_relative_path(path, f"{label}.path"))
    if len(relative) > MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label}.path exceeds {MAX_RELATIVE_PATH_BYTES} bytes",
        )
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            relative,
            label,
            maximum_bytes=MAX_RAW_BYTES,
        )
    )
    if not raw:
        _fail("empty-evidence-leaf", f"{label} must bind nonempty raw evidence")
    return common.descriptor_for_bytes(relative, raw, label)


def verify_c02_shutdown_v2_fd(
    root_fd: int,
    artifact_path: str,
    marker_path: str,
    expected_target: TargetTuple,
) -> VerifiedC02ShutdownV2:
    """Verify one completed source-owned shutdown-v2 pair through held FDs.

    ``expected_target`` must come from a separately replayed process bridge;
    this helper never accepts a candidate, freeze, semantic verdict, or
    caller-authored evidence descriptor.  It is intentionally suitable for a
    lifecycle supervisor before it decides whether a later success receipt may
    be emitted.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "C02 shutdown evidence root",
        )
    )
    if not isinstance(expected_target, TargetTuple):
        _fail(
            "invalid-shutdown-target",
            "C02 shutdown expected target must be a derived TargetTuple",
        )
    artifact = _shutdown_descriptor_from_path(
        root_fd,
        artifact_path,
        "C02 shutdown artifact",
    )
    marker = _shutdown_descriptor_from_path(
        root_fd,
        marker_path,
        "C02 shutdown completion marker",
    )
    return _verify_c02_shutdown_v2_descriptors_fd(
        root_fd,
        artifact,
        marker,
        expected_target,
        "C02 shutdown",
        set(),
    )


def verify_c02_shutdown_v2(
    evidence_root: Path,
    artifact_path: str,
    marker_path: str,
    expected_target: TargetTuple,
) -> VerifiedC02ShutdownV2:
    """Path wrapper for :func:`verify_c02_shutdown_v2_fd`."""

    root_fd = _open_private_evidence_root(
        evidence_root,
        "C02 shutdown evidence root",
    )
    try:
        return verify_c02_shutdown_v2_fd(
            root_fd,
            artifact_path,
            marker_path,
            expected_target,
        )
    finally:
        os.close(root_fd)


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
    reject_v1_serial_capture_opaque_leaf(
        _read_opaque_leaf(root_fd, contract, "soak scenario contract", used),
        "soak scenario contract",
    )
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
            raw = _read_opaque_leaf(
                root_fd,
                _descriptor(scenario[key], f"{label}.{key}"),
                f"{label}.{key}",
                used,
            )
            if key in {"request_ledger", "generation_audit_index"}:
                reject_v1_serial_capture_opaque_leaf(raw, f"{label}.{key}")
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


@dataclass(frozen=True)
class ScenarioCaptureTarget:
    """The process/listener tuple produced by the serial completion capture.

    GPU selection is intentionally absent here.  It is proven independently
    by the C02 observation and configuration-bridge producers, then joined by
    the v4 manifest verifier.
    """

    pid: int
    start_ticks: int
    listener_port: int
    listener_inode: int


@dataclass(frozen=True)
class ReplayedScenario:
    scenario_id: str
    target: ScenarioCaptureTarget
    request_ledger: common.EvidenceDescriptor
    runtime_event_log: common.EvidenceDescriptor
    generation_audit_index: common.EvidenceDescriptor
    request_id: str
    audit_directory: str


@dataclass(frozen=True)
class ReplayedScenarioCapture:
    session: common.EvidenceDescriptor
    contract: common.EvidenceDescriptor
    target: ScenarioCaptureTarget
    scenarios: tuple[ReplayedScenario, ...]
    audit_directory: str


@dataclass(frozen=True)
class ReplayedFallbackScenario:
    """One replayed native fallback scenario from the closed capture-v2 arm."""

    scenario_id: str
    target: ScenarioCaptureTarget
    request_ledger: common.EvidenceDescriptor
    runtime_event_log: common.EvidenceDescriptor
    generation_audit_index: common.EvidenceDescriptor
    fallback_event_log: common.EvidenceDescriptor
    request_id: str
    audit_directory: str


@dataclass(frozen=True)
class ReplayedFallbackScenarioCapture:
    """The separate, closed v2 native-fallback source capture."""

    session: common.EvidenceDescriptor
    contract: common.EvidenceDescriptor
    target: ScenarioCaptureTarget
    scenarios: tuple[ReplayedFallbackScenario, ...]
    audit_directory: str


def _capture_target(value: Any, label: str) -> ScenarioCaptureTarget:
    row = _exact(
        value,
        {"server_pid", "server_start_ticks", "listener_port", "listener_inode"},
        label,
    )
    return ScenarioCaptureTarget(
        pid=_positive(row["server_pid"], f"{label}.server_pid"),
        start_ticks=_positive(row["server_start_ticks"], f"{label}.server_start_ticks"),
        listener_port=_unprivileged_listener_port(
            row["listener_port"], f"{label}.listener_port"
        ),
        listener_inode=_positive(row["listener_inode"], f"{label}.listener_inode"),
    )


def _capture_matches_observed(
    capture: ScenarioCaptureTarget,
    observed: ObservedTarget,
) -> bool:
    return (
        capture.pid == observed.target.pid
        and capture.start_ticks == observed.target.start_ticks
        and capture.listener_port == observed.listener_port
        and capture.listener_inode == observed.listener_inode
    )


def _capture_parent(descriptor: common.EvidenceDescriptor, label: str) -> str:
    path = PurePosixPath(descriptor.path)
    # The v1 producer creates the capture as one direct child of the trusted
    # root.  Requiring that exact layout prevents a session from borrowing a
    # parent marker or raw leaf namespace from another capture.
    if (
        path.name != "session.json"
        or path.parent == PurePosixPath(".")
        or len(path.parent.parts) != 1
    ):
        _fail(
            "scenario-capture-layout-mismatch",
            f"{label} must be a direct capture/session.json leaf",
        )
    return path.parent.as_posix()


def _require_direct_v4_session_path(
    descriptor: common.EvidenceDescriptor,
    label: str,
) -> None:
    """Require a v4 bridge/observation session to use one root child.

    The serial-capture producer, the config bridge, and the C02 observation
    producer each publish a private direct-child capture directory.  v4's
    published request schema fixes this namespace, so terminal replay must
    enforce it too instead of inheriting v3's more permissive historical
    observation layout.
    """

    path = PurePosixPath(descriptor.path)
    if (
        path.name != "session.json"
        or path.parent == PurePosixPath(".")
        or len(path.parent.parts) != 1
    ):
        _fail(
            "invalid-session-path",
            f"{label} must be a direct capture/session.json leaf",
        )


def _capture_raw_path(
    descriptor: common.EvidenceDescriptor,
    capture_name: str,
    expected_name: str,
    label: str,
) -> None:
    if descriptor.path != f"{capture_name}/raw/{expected_name}":
        _fail(
            "scenario-capture-layout-mismatch",
            f"{label} must be {capture_name}/raw/{expected_name}",
        )


def _capture_child_path(
    descriptor: common.EvidenceDescriptor,
    capture_name: str,
    expected_name: str,
    label: str,
) -> None:
    if descriptor.path != f"{capture_name}/{expected_name}":
        _fail(
            "scenario-capture-layout-mismatch",
            f"{label} must be {capture_name}/{expected_name}",
        )


def _capture_socket_target(target: ScenarioCaptureTarget) -> TargetTuple:
    """Adapt the PID-only socket record parser without manufacturing GPU proof."""

    return TargetTuple(
        pid=target.pid,
        start_ticks=target.start_ticks,
        gpu_index=0,
        gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
    )


def _completion_request_contract(value: Any, label: str) -> dict[str, Any]:
    row = _exact(
        value,
        {"model", "prompt", "max_tokens", "temperature", "top_p", "seed", "stream"},
        label,
    )
    for key, maximum in (("model", 256), ("prompt", 1024 * 1024)):
        item = row[key]
        if type(item) is not str or not item or len(item) > maximum or "\x00" in item:
            _fail("invalid-completion-contract", f"{label}.{key} must be a bounded nonempty string")
    max_tokens = row["max_tokens"]
    if type(max_tokens) is not int or not 1 <= max_tokens <= 65536:
        _fail("invalid-completion-contract", f"{label}.max_tokens must be from 1 through 65536")
    for key, lower, upper, strict_lower in (
        ("temperature", 0.0, 2.0, False),
        ("top_p", 0.0, 1.0, True),
    ):
        item = row[key]
        if type(item) not in {int, float} or isinstance(item, bool):
            _fail("invalid-completion-contract", f"{label}.{key} must be finite")
        number = float(item)
        if not math.isfinite(number) or number > upper or (number <= lower if strict_lower else number < lower):
            _fail("invalid-completion-contract", f"{label}.{key} is outside its supported range")
    seed = row["seed"]
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        _fail("invalid-completion-contract", f"{label}.seed is outside its supported range")
    if row["stream"] is not False:
        _fail("invalid-completion-contract", f"{label}.stream must be false")
    if len(common.canonical_json_bytes(row)) > MAX_RAW_BYTES:
        _fail("invalid-completion-contract", f"{label} exceeds the bounded completion request size")
    return row


def _fallback_completion_request_contract(value: Any, label: str) -> dict[str, Any]:
    """Close the v2 fallback arm to the one reviewed public request shape.

    In particular, the literal nonzero temperature is pinned to one so a
    decimal value cannot silently become zero when Riley decodes it as f32.
    This is evidence replay only; it does not infer a fallback from policy.
    """

    row = _completion_request_contract(value, label)
    if row["max_tokens"] != 1:
        _fail(
            "invalid-fallback-completion-contract",
            f"{label}.max_tokens must be exactly 1",
        )
    if float(row["temperature"]) != 1.0:
        _fail(
            "invalid-fallback-completion-contract",
            f"{label}.temperature must be exactly 1",
        )
    if float(row["top_p"]) != 1.0:
        _fail(
            "invalid-fallback-completion-contract",
            f"{label}.top_p must be exactly 1",
        )
    return row


def _replay_fallback_capture_contract(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    *,
    capture_name: str,
    candidate_id: str,
    configuration_profile: str,
    used_paths: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Replay only the closed single-scenario capture-v2 contract."""

    _capture_raw_path(
        descriptor,
        capture_name,
        "scenario-contract.json",
        "native fallback scenario capture contract",
    )
    _reserve(
        descriptor,
        label="native fallback scenario capture contract",
        used_paths=used_paths,
    )
    _raw, document = _read_json(
        root_fd,
        descriptor,
        "native fallback scenario capture contract",
    )
    row = _exact(
        document,
        {"schema_version", "candidate_id", "configuration_profile", "scenarios"},
        "native fallback scenario capture contract",
    )
    if row["schema_version"] != SCENARIO_FALLBACK_CAPTURE_CONTRACT_VERSION:
        _fail(
            "historical-fallback-scenario-contract-version-rejected",
            "native fallback scenario contract has an unsupported version",
        )
    if _candidate_id(
        row["candidate_id"], "native fallback scenario capture contract.candidate_id"
    ) != candidate_id:
        _fail(
            "scenario-contract-candidate-mismatch",
            "native fallback scenario contract candidate drifted",
        )
    if (
        row["configuration_profile"] != MAX_PERFORMANCE_EXACT_PROFILE
        or configuration_profile != MAX_PERFORMANCE_EXACT_PROFILE
    ):
        _fail(
            "fallback-profile-mismatch",
            "native fallback scenario contract requires max-performance-exact",
        )
    scenarios = row["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        _fail(
            "invalid-scenario-inventory",
            "native fallback scenario contract must contain exactly one scenario",
        )
    scenario = _exact(
        scenarios[0],
        {"scenario_id", "completion_request"},
        "native fallback scenario capture contract.scenarios[0]",
    )
    if scenario["scenario_id"] != FALLBACK_SCENARIO_ID:
        _fail(
            "invalid-scenario-inventory",
            "native fallback scenario contract must contain exact-backend-fallback",
        )
    return [
        (
            FALLBACK_SCENARIO_ID,
            _fallback_completion_request_contract(
                scenario["completion_request"],
                "native fallback scenario capture contract.scenarios[0].completion_request",
            ),
        )
    ]


def _replay_capture_contract(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    *,
    capture_name: str,
    candidate_id: str,
    configuration_profile: str,
    used_paths: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    _capture_raw_path(
        descriptor,
        capture_name,
        "scenario-contract.json",
        "scenario capture contract",
    )
    _reserve(descriptor, label="scenario capture contract", used_paths=used_paths)
    _raw, document = _read_json(root_fd, descriptor, "scenario capture contract")
    row = _exact(
        document,
        {"schema_version", "candidate_id", "configuration_profile", "scenarios"},
        "scenario capture contract",
    )
    if row["schema_version"] != SCENARIO_CAPTURE_CONTRACT_VERSION:
        _fail(
            "historical-scenario-contract-version-rejected",
            "scenario capture contract has an unsupported schema version",
        )
    if _candidate_id(row["candidate_id"], "scenario capture contract.candidate_id") != candidate_id:
        _fail("scenario-contract-candidate-mismatch", "scenario capture contract candidate drifted")
    if row["configuration_profile"] != configuration_profile:
        _fail("scenario-contract-profile-mismatch", "scenario capture contract profile drifted")
    values = row["scenarios"]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_SCENARIOS:
        _fail("invalid-scenario-inventory", "scenario capture contract must have a bounded nonempty scenario list")
    result: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        label = f"scenario capture contract.scenarios[{index}]"
        scenario = _exact(item, {"scenario_id", "completion_request"}, label)
        scenario_id = scenario["scenario_id"]
        if (
            type(scenario_id) is not str
            or len(scenario_id) > 128
            or SCENARIO_ID_RE.fullmatch(scenario_id) is None
            or scenario_id == "exact-backend-fallback"
            or scenario_id in seen
        ):
            _fail("invalid-scenario-inventory", f"{label}.scenario_id is invalid for serial v4")
        seen.add(scenario_id)
        result.append((scenario_id, _completion_request_contract(scenario["completion_request"], f"{label}.completion_request")))
    return result


def _completion_request_bytes(port: int, body: bytes) -> bytes:
    return (
        f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body


def _completion_response_length(raw: bytes, label: str) -> int:
    if not raw.endswith(b"\r\n\r\n") or not raw or len(raw) > 64 * 1024:
        _fail("invalid-completion-response-head", f"{label} is malformed or oversized")
    try:
        lines = raw[:-4].decode("ascii").split("\r\n")
    except UnicodeDecodeError as error:
        _fail("invalid-completion-response-head", f"{label} is not ASCII: {error}")
    if not lines or lines[0] != "HTTP/1.1 200 OK":
        _fail("completion-response-status-mismatch", f"{label} must be HTTP/1.1 200 OK")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line or line[:1] in {" ", "\t"}:
            _fail("invalid-completion-response-head", f"{label} has a malformed header")
        name, value = line.split(":", 1)
        lowered = name.lower()
        if lowered in headers or re.fullmatch(r"[A-Za-z0-9-]+", name) is None:
            _fail("invalid-completion-response-head", f"{label} repeats or has an invalid header name")
        if any((ord(character) < 32 and character != "\t") or ord(character) == 127 for character in value):
            _fail("invalid-completion-response-head", f"{label} has a control character in a header value")
        headers[lowered] = value.strip(" \t")
    if "transfer-encoding" in headers:
        _fail("invalid-completion-response-head", f"{label} must not use Transfer-Encoding")
    length = headers.get("content-length")
    content_type = headers.get("content-type")
    if length is None or UINT_RE.fullmatch(length) is None:
        _fail("invalid-completion-response-head", f"{label} lacks one numeric Content-Length")
    parsed = int(length)
    if not 1 <= parsed <= MAX_RAW_BYTES:
        _fail("invalid-completion-response-head", f"{label} Content-Length is out of bounds")
    if content_type is None or content_type.split(";", 1)[0].strip().lower() != "application/json":
        _fail("invalid-completion-response-head", f"{label} Content-Type is not application/json")
    return parsed


def _source_request_id(raw: bytes, label: str) -> str:
    document = _common(
        lambda: common.parse_strict_json(
            raw,
            label,
            maximum_bytes=MAX_RAW_BYTES,
            require_object=True,
        )
    )
    assert isinstance(document, dict)
    request_id = document.get("id")
    if type(request_id) is not str or COMPLETION_REQUEST_ID_RE.fullmatch(request_id) is None:
        _fail("invalid-completion-response-id", f"{label} has no valid source-issued completion ID")
    return request_id


def _replay_source_audit(
    root_fd: int,
    record: common.EvidenceDescriptor,
    marker: common.EvidenceDescriptor,
    *,
    capture_name: str,
    expected_audit_directory: str | None,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    target: ScenarioCaptureTarget,
    request_id: str,
    used_paths: set[str],
    label: str,
) -> str:
    record_path = PurePosixPath(record.path)
    marker_path = PurePosixPath(marker.path)
    audit_directory = record_path.parent.as_posix()
    if (
        record_path.parent == PurePosixPath(".")
        or len(record_path.parent.parts) != 1
        or audit_directory == capture_name
        or (
            expected_audit_directory is not None
            and audit_directory != expected_audit_directory
        )
        or record_path.name != f"{request_id}.json"
        or marker.path != f"{record.path}.complete"
        or marker_path.parent != record_path.parent
        or marker_path.name != f"{record_path.name}.complete"
    ):
        _fail("source-audit-layout-mismatch", f"{label} audit record/marker layout drifted")
    _reserve(record, label=f"{label} source audit", used_paths=used_paths)
    _reserve(marker, label=f"{label} source audit marker", used_paths=used_paths)
    record_raw, document = _read_json(root_fd, record, f"{label} source audit")
    row = _exact(
        document,
        {
            "schema_version", "candidate_id", "runtime_identity", "process_identity",
            "server_request_id", "delivery_mode", "prompt_token_ids",
            "committed_output_tokens", "sampling_selections", "finish_reason", "usage",
        },
        f"{label} source audit",
    )
    if row["schema_version"] != SCENARIO_CAPTURE_AUDIT_VERSION:
        _fail("historical-source-audit-version-rejected", f"{label} source audit has an unsupported version")
    if _candidate_id(row["candidate_id"], f"{label} source audit.candidate_id") != candidate_id:
        _fail("source-audit-candidate-mismatch", f"{label} source audit candidate drifted")
    identity = _exact(
        row["runtime_identity"],
        {"configuration_profile", "configuration_sha256"},
        f"{label} source audit.runtime_identity",
    )
    if identity["configuration_profile"] != configuration_profile or _sha256(
        identity["configuration_sha256"],
        f"{label} source audit.runtime_identity.configuration_sha256",
    ) != configuration_sha256:
        _fail("source-audit-runtime-identity-mismatch", f"{label} source audit runtime identity drifted")
    process = _exact(
        row["process_identity"],
        {"pid", "start_ticks"},
        f"{label} source audit.process_identity",
    )
    if (
        _positive(process["pid"], f"{label} source audit.process_identity.pid") != target.pid
        or _positive(process["start_ticks"], f"{label} source audit.process_identity.start_ticks") != target.start_ticks
    ):
        _fail("source-audit-process-mismatch", f"{label} source audit process tuple drifted")
    if row["server_request_id"] != request_id or row["delivery_mode"] != "non-stream":
        _fail("source-audit-request-mismatch", f"{label} source audit request identity drifted")
    marker_raw, marker_document = _read_json(root_fd, marker, f"{label} source audit marker")
    marker_row = _exact(
        marker_document,
        {"schema_version", "artifact_filename", "artifact_sha256"},
        f"{label} source audit marker",
    )
    if marker_row["schema_version"] != SCENARIO_CAPTURE_AUDIT_MARKER_VERSION:
        _fail("historical-source-audit-marker-version-rejected", f"{label} source audit marker has an unsupported version")
    if marker_row["artifact_filename"] != record_path.name or _sha256(
        marker_row["artifact_sha256"], f"{label} source audit marker.artifact_sha256"
    ) != hashlib.sha256(record_raw).hexdigest() or not marker_raw:
        _fail("source-audit-marker-mismatch", f"{label} source audit marker does not bind exact record bytes")
    return audit_directory


def _fallback_source_process_identity(value: Any, label: str) -> tuple[int, int]:
    """Parse the source's stricter u32 PID/start-tick representation."""

    process = _exact(value, {"pid", "start_ticks"}, label)
    pid = process["pid"]
    start_ticks = process["start_ticks"]
    if type(pid) is not int or not 1 <= pid <= (1 << 32) - 1:
        _fail("invalid-source-process-identity", f"{label}.pid must be a positive u32 integer")
    if type(start_ticks) is not int or start_ticks < 1:
        _fail(
            "invalid-source-process-identity",
            f"{label}.start_ticks must be a positive integer",
        )
    return pid, start_ticks


def _replay_fallback_source_pair(
    root_fd: int,
    *,
    audit_record: common.EvidenceDescriptor,
    audit_directory: str,
    fallback_event: common.EvidenceDescriptor,
    fallback_marker: common.EvidenceDescriptor,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    target: ScenarioCaptureTarget,
    request_id: str,
    used_paths: set[str],
    label: str,
) -> None:
    """Replay source-written fallback evidence derived from one public ID.

    The source audit and fallback markers are ordinary one-link leaves.  The
    separate paired-hardlink publication protocol is reserved for the v5
    terminal manifest below the evidence root.
    """

    if configuration_profile != MAX_PERFORMANCE_EXACT_PROFILE:
        _fail(
            "fallback-profile-mismatch",
            f"{label} native fallback source pair requires max-performance-exact",
        )
    audit_path = PurePosixPath(audit_record.path)
    expected_event_path = f"{audit_directory}/{request_id}.fallback.json"
    expected_marker_path = f"{expected_event_path}.complete"
    if fallback_event.path != expected_event_path or fallback_marker.path != expected_marker_path:
        _fail(
            "source-fallback-layout-mismatch",
            f"{label} fallback source pair is not derived from its response ID",
        )
    _reserve(fallback_event, label=f"{label} source fallback event", used_paths=used_paths)
    _reserve(
        fallback_marker,
        label=f"{label} source fallback completion marker",
        used_paths=used_paths,
    )
    audit_raw, audit_document = _read_json(root_fd, audit_record, f"{label} source audit")
    audit_row = _exact(
        audit_document,
        {
            "schema_version", "candidate_id", "runtime_identity", "process_identity",
            "server_request_id", "delivery_mode", "prompt_token_ids",
            "committed_output_tokens", "sampling_selections", "finish_reason", "usage",
        },
        f"{label} source audit",
    )
    if (
        audit_row["schema_version"] != SCENARIO_CAPTURE_AUDIT_VERSION
        or audit_row["candidate_id"] != candidate_id
        or audit_row["server_request_id"] != request_id
        or audit_row["delivery_mode"] != "non-stream"
    ):
        _fail(
            "source-audit-request-mismatch",
            f"{label} source audit identity drifted",
        )
    audit_identity = _exact(
        audit_row["runtime_identity"],
        {"configuration_profile", "configuration_sha256"},
        f"{label} source audit.runtime_identity",
    )
    if (
        audit_identity["configuration_profile"] != configuration_profile
        or _sha256(
            audit_identity["configuration_sha256"],
            f"{label} source audit.runtime_identity.configuration_sha256",
        )
        != configuration_sha256
    ):
        _fail("source-audit-runtime-identity-mismatch", f"{label} source audit identity drifted")
    if _fallback_source_process_identity(
        audit_row["process_identity"], f"{label} source audit.process_identity"
    ) != (target.pid, target.start_ticks):
        _fail("source-audit-process-mismatch", f"{label} source audit process tuple drifted")

    event_raw, event_document = _read_json(
        root_fd,
        fallback_event,
        f"{label} source native fallback event",
    )
    event = _exact(
        event_document,
        {
            "schema_version", "candidate_id", "runtime_identity", "process_identity",
            "server_request_id", "generation_audit", "fallback_selections",
        },
        f"{label} source native fallback event",
    )
    if (
        event["schema_version"] != SCENARIO_FALLBACK_EVENT_VERSION
        or event["candidate_id"] != candidate_id
        or event["server_request_id"] != request_id
    ):
        _fail(
            "source-fallback-event-identity-mismatch",
            f"{label} source native fallback event identity drifted",
        )
    event_identity = _exact(
        event["runtime_identity"],
        {"configuration_profile", "configuration_sha256"},
        f"{label} source native fallback event.runtime_identity",
    )
    if (
        event_identity["configuration_profile"] != configuration_profile
        or _sha256(
            event_identity["configuration_sha256"],
            f"{label} source native fallback event.runtime_identity.configuration_sha256",
        )
        != configuration_sha256
    ):
        _fail(
            "source-fallback-runtime-identity-mismatch",
            f"{label} source native fallback event runtime identity drifted",
        )
    if _fallback_source_process_identity(
        event["process_identity"],
        f"{label} source native fallback event.process_identity",
    ) != (target.pid, target.start_ticks):
        _fail(
            "source-fallback-process-mismatch",
            f"{label} source native fallback event process tuple drifted",
        )
    generation_audit = _exact(
        event["generation_audit"],
        {"artifact_filename", "artifact_sha256"},
        f"{label} source native fallback event.generation_audit",
    )
    if (
        generation_audit["artifact_filename"] != audit_path.name
        or _sha256(
            generation_audit["artifact_sha256"],
            f"{label} source native fallback event.generation_audit.artifact_sha256",
        )
        != hashlib.sha256(audit_raw).hexdigest()
    ):
        _fail(
            "source-fallback-audit-binding-mismatch",
            f"{label} source native fallback event does not bind exact audit bytes",
        )
    selections = event["fallback_selections"]
    if (
        not isinstance(selections, list)
        or not 1 <= len(selections) <= MAX_NATIVE_FALLBACK_SELECTIONS
        or selections != audit_row["sampling_selections"]
    ):
        _fail(
            "source-fallback-selection-mismatch",
            f"{label} source native fallback selections do not exactly replay the audit",
        )
    for index, value in enumerate(selections):
        selection = _exact(
            value,
            {
                "iteration_id", "configured_backend", "selected_backend",
                "ineligibility_reason", "committed",
            },
            f"{label} source native fallback event.fallback_selections[{index}]",
        )
        if (
            type(selection["iteration_id"]) is not int
            or selection["iteration_id"] < 1
            or selection["configured_backend"] != "gpu-greedy"
            or selection["selected_backend"] != "cpu-normative"
            or selection["ineligibility_reason"] != "nonzero-temperature"
            or selection["committed"] is not True
        ):
            _fail(
                "source-fallback-transition-mismatch",
                f"{label} source native fallback event is not the reviewed nonzero-temperature transition",
            )
    marker_raw, marker_document = _read_json(
        root_fd,
        fallback_marker,
        f"{label} source native fallback completion marker",
    )
    marker = _exact(
        marker_document,
        {"schema_version", "artifact_filename", "artifact_sha256"},
        f"{label} source native fallback completion marker",
    )
    if (
        marker["schema_version"] != SCENARIO_FALLBACK_EVENT_MARKER_VERSION
        or marker["artifact_filename"] != PurePosixPath(fallback_event.path).name
        or _sha256(
            marker["artifact_sha256"],
            f"{label} source native fallback completion marker.artifact_sha256",
        )
        != hashlib.sha256(event_raw).hexdigest()
        or not marker_raw
    ):
        _fail(
            "source-fallback-marker-mismatch",
            f"{label} source native fallback completion marker does not bind exact event bytes",
        )


def _replay_capture_scenario(
    root_fd: int,
    value: Any,
    *,
    capture_name: str,
    sequence: int,
    expected_scenario_id: str,
    completion_request: dict[str, Any],
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    expected_target: ScenarioCaptureTarget,
    expected_audit_directory: str | None,
    used_paths: set[str],
) -> ReplayedScenario:
    label = f"scenario capture.scenarios[{sequence}]"
    row = _exact(
        value,
        {
            "scenario_id", "target", "process", "listener", "request_ledger",
            "runtime_event_log", "generation_audit_index",
        },
        label,
    )
    if row["scenario_id"] != expected_scenario_id:
        _fail("scenario-capture-inventory-mismatch", f"{label} does not preserve contract scenario order/ID")
    target = _capture_target(row["target"], f"{label}.target")
    if target != expected_target:
        _fail("scenario-capture-target-drift", f"{label} target differs from the capture session target")
    prefix = f"{sequence:06d}"
    process = _exact(row["process"], {"pre_stat", "post_stat", "final_stat"}, f"{label}.process")
    listener = _exact(
        row["listener"],
        {
            "address", "port", "socket_inode", "pre_proc_net_tcp", "post_proc_net_tcp",
            "pre_server_fd_sockets", "post_server_fd_sockets", "final_proc_net_tcp",
            "final_server_fd_sockets",
        },
        f"{label}.listener",
    )
    if (
        listener["address"] != "127.0.0.1"
        or _unprivileged_listener_port(listener["port"], f"{label}.listener.port") != target.listener_port
        or _positive(listener["socket_inode"], f"{label}.listener.socket_inode") != target.listener_inode
    ):
        _fail("scenario-capture-listener-mismatch", f"{label} listener differs from its target tuple")
    process_descriptors = {
        key: _descriptor(process[key], f"{label}.process.{key}")
        for key in ("pre_stat", "post_stat", "final_stat")
    }
    listener_descriptors = {
        key: _descriptor(listener[key], f"{label}.listener.{key}")
        for key in (
            "pre_proc_net_tcp", "post_proc_net_tcp", "pre_server_fd_sockets",
            "post_server_fd_sockets", "final_proc_net_tcp", "final_server_fd_sockets",
        )
    }
    expected_names = {
        "pre_stat": f"{prefix}.pre.proc-stat",
        "post_stat": f"{prefix}.post.proc-stat",
        "final_stat": f"{prefix}.final.proc-stat",
        "pre_proc_net_tcp": f"{prefix}.pre.proc-net-tcp",
        "post_proc_net_tcp": f"{prefix}.post.proc-net-tcp",
        "pre_server_fd_sockets": f"{prefix}.pre.proc-fd-sockets.json",
        "post_server_fd_sockets": f"{prefix}.post.proc-fd-sockets.json",
        "final_proc_net_tcp": f"{prefix}.final.proc-net-tcp",
        "final_server_fd_sockets": f"{prefix}.final.proc-fd-sockets.json",
    }
    all_raw = list(process_descriptors.values()) + list(listener_descriptors.values())
    _common(lambda: common.require_unique_descriptors(all_raw, f"{label} process/listener leaves"))
    for key, descriptor in {**process_descriptors, **listener_descriptors}.items():
        _capture_raw_path(descriptor, capture_name, expected_names[key], f"{label}.{key}")
        _reserve(descriptor, label=f"{label}.{key}", used_paths=used_paths)
    expected_process = (target.pid, target.start_ticks)
    for key, descriptor in process_descriptors.items():
        if _parse_proc_stat(_read_bytes(root_fd, descriptor, f"{label}.{key}"), f"{label}.{key}") != expected_process:
            _fail("scenario-capture-pid-start-tick-mismatch", f"{label}.{key} differs from its target tuple")
    socket_target = _capture_socket_target(target)
    for key in ("pre_proc_net_tcp", "post_proc_net_tcp", "final_proc_net_tcp"):
        if _parse_listener_inodes(
            _read_bytes(root_fd, listener_descriptors[key], f"{label}.{key}"),
            target.listener_port,
            f"{label}.{key}",
        ) != {target.listener_inode}:
            _fail("scenario-capture-listener-proof-mismatch", f"{label}.{key} does not bind one expected listener inode")
    for key in ("pre_server_fd_sockets", "post_server_fd_sockets", "final_server_fd_sockets"):
        sockets = _parse_socket_snapshot(
            _read_bytes(root_fd, listener_descriptors[key], f"{label}.{key}"),
            socket_target,
            f"{label}.{key}",
        )
        if target.listener_inode not in sockets:
            _fail("scenario-capture-listener-proof-mismatch", f"{label}.{key} does not bind the listener to the target PID")

    ledger = _descriptor(row["request_ledger"], f"{label}.request_ledger")
    index = _descriptor(row["generation_audit_index"], f"{label}.generation_audit_index")
    _capture_child_path(ledger, capture_name, f"{prefix}-{expected_scenario_id}.request-ledger.json", f"{label}.request_ledger")
    _capture_child_path(index, capture_name, f"{prefix}-{expected_scenario_id}.generation-audit-index.json", f"{label}.generation_audit_index")
    _reserve(ledger, label=f"{label}.request_ledger", used_paths=used_paths)
    _reserve(index, label=f"{label}.generation_audit_index", used_paths=used_paths)
    _ledger_raw, ledger_document = _read_json(root_fd, ledger, f"{label}.request_ledger")
    ledger_row = _exact(
        ledger_document,
        {"schema_version", "scenario_id", "delivery_mode", "server_request_id", "request", "response_head", "response_body"},
        f"{label}.request_ledger",
    )
    if (
        ledger_row["schema_version"] != SCENARIO_CAPTURE_LEDGER_VERSION
        or ledger_row["scenario_id"] != expected_scenario_id
        or ledger_row["delivery_mode"] != "non-stream"
        or type(ledger_row["server_request_id"]) is not str
        or COMPLETION_REQUEST_ID_RE.fullmatch(ledger_row["server_request_id"]) is None
    ):
        _fail("scenario-capture-ledger-mismatch", f"{label} ledger identity is invalid")
    request_descriptor = _descriptor(ledger_row["request"], f"{label}.request_ledger.request")
    head_descriptor = _descriptor(ledger_row["response_head"], f"{label}.request_ledger.response_head")
    body_descriptor = _descriptor(ledger_row["response_body"], f"{label}.request_ledger.response_body")
    _common(lambda: common.require_unique_descriptors([request_descriptor, head_descriptor, body_descriptor], f"{label} ledger leaves"))
    for descriptor, name in (
        (request_descriptor, f"{prefix}.request.http"),
        (head_descriptor, f"{prefix}.response-head.http"),
        (body_descriptor, f"{prefix}.response-body.json"),
    ):
        _capture_raw_path(descriptor, capture_name, name, f"{label} ledger raw leaf")
        _reserve(descriptor, label=f"{label} ledger raw leaf", used_paths=used_paths)
    expected_body = common.canonical_json_bytes(completion_request)
    if _read_bytes(root_fd, request_descriptor, f"{label} completion request") != _completion_request_bytes(target.listener_port, expected_body):
        _fail("scenario-capture-request-mismatch", f"{label} raw completion request differs from the contract")
    response_head = _read_bytes(root_fd, head_descriptor, f"{label} completion response head")
    response_body = _read_bytes(root_fd, body_descriptor, f"{label} completion response body")
    if _completion_response_length(response_head, f"{label} completion response head") != len(response_body):
        _fail("scenario-capture-response-length-mismatch", f"{label} response body differs from Content-Length")
    request_id = _source_request_id(response_body, f"{label} completion response body")
    if request_id != ledger_row["server_request_id"]:
        _fail("scenario-capture-response-id-mismatch", f"{label} response ID differs from its ledger")

    _index_raw, index_document = _read_json(root_fd, index, f"{label}.generation_audit_index")
    index_row = _exact(
        index_document,
        {"schema_version", "scenario_id", "server_request_id", "audit_record", "audit_completion_marker"},
        f"{label}.generation_audit_index",
    )
    if (
        index_row["schema_version"] != SCENARIO_CAPTURE_AUDIT_INDEX_VERSION
        or index_row["scenario_id"] != expected_scenario_id
        or index_row["server_request_id"] != request_id
    ):
        _fail("scenario-capture-audit-index-mismatch", f"{label} audit index identity drifted")
    audit_record = _descriptor(index_row["audit_record"], f"{label}.generation_audit_index.audit_record")
    audit_marker = _descriptor(index_row["audit_completion_marker"], f"{label}.generation_audit_index.audit_completion_marker")
    runtime_event = _descriptor(row["runtime_event_log"], f"{label}.runtime_event_log")
    if runtime_event != audit_record:
        _fail("scenario-capture-runtime-event-mismatch", f"{label} runtime event must be the indexed source audit record")
    audit_directory = _replay_source_audit(
        root_fd,
        audit_record,
        audit_marker,
        capture_name=capture_name,
        expected_audit_directory=expected_audit_directory,
        candidate_id=candidate_id,
        configuration_profile=configuration_profile,
        configuration_sha256=configuration_sha256,
        target=target,
        request_id=request_id,
        used_paths=used_paths,
        label=label,
    )
    return ReplayedScenario(
        scenario_id=expected_scenario_id,
        target=target,
        request_ledger=ledger,
        runtime_event_log=runtime_event,
        generation_audit_index=index,
        request_id=request_id,
        audit_directory=audit_directory,
    )


def _replay_fallback_capture_scenario(
    root_fd: int,
    value: Any,
    *,
    capture_name: str,
    expected_target: ScenarioCaptureTarget,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    completion_request: dict[str, Any],
    used_paths: set[str],
) -> ReplayedFallbackScenario:
    """Replay the one capture-v2 scenario and its four source-owned leaves."""

    label = "native fallback scenario capture.scenarios[0]"
    row = _exact(
        value,
        {
            "scenario_id", "target", "process", "listener", "request_ledger",
            "runtime_event_log", "generation_audit_index", "fallback_event_log",
        },
        label,
    )
    if row["scenario_id"] != FALLBACK_SCENARIO_ID:
        _fail(
            "scenario-capture-inventory-mismatch",
            f"{label} must preserve exact-backend-fallback",
        )
    target = _capture_target(row["target"], f"{label}.target")
    if target != expected_target:
        _fail(
            "scenario-capture-target-drift",
            f"{label} target differs from the capture session target",
        )
    prefix = "000000"
    process = _exact(row["process"], {"pre_stat", "post_stat", "final_stat"}, f"{label}.process")
    listener = _exact(
        row["listener"],
        {
            "address", "port", "socket_inode", "pre_proc_net_tcp", "post_proc_net_tcp",
            "pre_server_fd_sockets", "post_server_fd_sockets", "final_proc_net_tcp",
            "final_server_fd_sockets",
        },
        f"{label}.listener",
    )
    if (
        listener["address"] != "127.0.0.1"
        or _unprivileged_listener_port(listener["port"], f"{label}.listener.port")
        != target.listener_port
        or _positive(listener["socket_inode"], f"{label}.listener.socket_inode")
        != target.listener_inode
    ):
        _fail(
            "scenario-capture-listener-mismatch",
            f"{label} listener differs from its target tuple",
        )
    process_descriptors = {
        key: _descriptor(process[key], f"{label}.process.{key}")
        for key in ("pre_stat", "post_stat", "final_stat")
    }
    listener_descriptors = {
        key: _descriptor(listener[key], f"{label}.listener.{key}")
        for key in (
            "pre_proc_net_tcp", "post_proc_net_tcp", "pre_server_fd_sockets",
            "post_server_fd_sockets", "final_proc_net_tcp", "final_server_fd_sockets",
        )
    }
    expected_names = {
        "pre_stat": f"{prefix}.pre.proc-stat",
        "post_stat": f"{prefix}.post.proc-stat",
        "final_stat": f"{prefix}.final.proc-stat",
        "pre_proc_net_tcp": f"{prefix}.pre.proc-net-tcp",
        "post_proc_net_tcp": f"{prefix}.post.proc-net-tcp",
        "pre_server_fd_sockets": f"{prefix}.pre.proc-fd-sockets.json",
        "post_server_fd_sockets": f"{prefix}.post.proc-fd-sockets.json",
        "final_proc_net_tcp": f"{prefix}.final.proc-net-tcp",
        "final_server_fd_sockets": f"{prefix}.final.proc-fd-sockets.json",
    }
    all_raw = list(process_descriptors.values()) + list(listener_descriptors.values())
    _common(lambda: common.require_unique_descriptors(all_raw, f"{label} process/listener leaves"))
    for key, descriptor in {**process_descriptors, **listener_descriptors}.items():
        _capture_raw_path(descriptor, capture_name, expected_names[key], f"{label}.{key}")
        _reserve(descriptor, label=f"{label}.{key}", used_paths=used_paths)
    expected_process = (target.pid, target.start_ticks)
    for key, descriptor in process_descriptors.items():
        if _parse_proc_stat(
            _read_bytes(root_fd, descriptor, f"{label}.{key}"), f"{label}.{key}"
        ) != expected_process:
            _fail(
                "scenario-capture-pid-start-tick-mismatch",
                f"{label}.{key} differs from its target tuple",
            )
    socket_target = _capture_socket_target(target)
    for key in ("pre_proc_net_tcp", "post_proc_net_tcp", "final_proc_net_tcp"):
        if _parse_listener_inodes(
            _read_bytes(root_fd, listener_descriptors[key], f"{label}.{key}"),
            target.listener_port,
            f"{label}.{key}",
        ) != {target.listener_inode}:
            _fail(
                "scenario-capture-listener-proof-mismatch",
                f"{label}.{key} does not bind one expected listener inode",
            )
    for key in ("pre_server_fd_sockets", "post_server_fd_sockets", "final_server_fd_sockets"):
        sockets = _parse_socket_snapshot(
            _read_bytes(root_fd, listener_descriptors[key], f"{label}.{key}"),
            socket_target,
            f"{label}.{key}",
        )
        if target.listener_inode not in sockets:
            _fail(
                "scenario-capture-listener-proof-mismatch",
                f"{label}.{key} does not bind the listener to the target PID",
            )

    ledger = _descriptor(row["request_ledger"], f"{label}.request_ledger")
    index = _descriptor(row["generation_audit_index"], f"{label}.generation_audit_index")
    _capture_child_path(
        ledger,
        capture_name,
        f"{prefix}-{FALLBACK_SCENARIO_ID}.request-ledger.json",
        f"{label}.request_ledger",
    )
    _capture_child_path(
        index,
        capture_name,
        f"{prefix}-{FALLBACK_SCENARIO_ID}.generation-audit-index.json",
        f"{label}.generation_audit_index",
    )
    _reserve(ledger, label=f"{label}.request_ledger", used_paths=used_paths)
    _reserve(index, label=f"{label}.generation_audit_index", used_paths=used_paths)
    _ledger_raw, ledger_document = _read_json(root_fd, ledger, f"{label}.request_ledger")
    ledger_row = _exact(
        ledger_document,
        {
            "schema_version", "scenario_id", "delivery_mode", "server_request_id",
            "request", "response_head", "response_body",
        },
        f"{label}.request_ledger",
    )
    if (
        ledger_row["schema_version"] != SCENARIO_CAPTURE_LEDGER_VERSION
        or ledger_row["scenario_id"] != FALLBACK_SCENARIO_ID
        or ledger_row["delivery_mode"] != "non-stream"
        or type(ledger_row["server_request_id"]) is not str
        or COMPLETION_REQUEST_ID_RE.fullmatch(ledger_row["server_request_id"]) is None
    ):
        _fail("scenario-capture-ledger-mismatch", f"{label} ledger identity is invalid")
    request_descriptor = _descriptor(ledger_row["request"], f"{label}.request_ledger.request")
    head_descriptor = _descriptor(
        ledger_row["response_head"], f"{label}.request_ledger.response_head"
    )
    body_descriptor = _descriptor(
        ledger_row["response_body"], f"{label}.request_ledger.response_body"
    )
    _common(
        lambda: common.require_unique_descriptors(
            [request_descriptor, head_descriptor, body_descriptor], f"{label} ledger leaves"
        )
    )
    for descriptor, name in (
        (request_descriptor, f"{prefix}.request.http"),
        (head_descriptor, f"{prefix}.response-head.http"),
        (body_descriptor, f"{prefix}.response-body.json"),
    ):
        _capture_raw_path(descriptor, capture_name, name, f"{label} ledger raw leaf")
        _reserve(descriptor, label=f"{label} ledger raw leaf", used_paths=used_paths)
    expected_body = common.canonical_json_bytes(completion_request)
    if _read_bytes(root_fd, request_descriptor, f"{label} completion request") != _completion_request_bytes(
        target.listener_port, expected_body
    ):
        _fail(
            "scenario-capture-request-mismatch",
            f"{label} raw completion request differs from the contract",
        )
    response_head = _read_bytes(root_fd, head_descriptor, f"{label} completion response head")
    response_body = _read_bytes(root_fd, body_descriptor, f"{label} completion response body")
    if _completion_response_length(response_head, f"{label} completion response head") != len(response_body):
        _fail(
            "scenario-capture-response-length-mismatch",
            f"{label} response body differs from Content-Length",
        )
    request_id = _source_request_id(response_body, f"{label} completion response body")
    if request_id != ledger_row["server_request_id"]:
        _fail(
            "scenario-capture-response-id-mismatch",
            f"{label} response ID differs from its ledger",
        )

    _index_raw, index_document = _read_json(root_fd, index, f"{label}.generation_audit_index")
    index_row = _exact(
        index_document,
        {
            "schema_version", "scenario_id", "server_request_id", "audit_record",
            "audit_completion_marker", "fallback_event", "fallback_completion_marker",
        },
        f"{label}.generation_audit_index",
    )
    if (
        index_row["schema_version"] != SCENARIO_FALLBACK_CAPTURE_AUDIT_INDEX_VERSION
        or index_row["scenario_id"] != FALLBACK_SCENARIO_ID
        or index_row["server_request_id"] != request_id
    ):
        _fail(
            "scenario-capture-audit-index-mismatch",
            f"{label} fallback audit index identity drifted",
        )
    audit_record = _descriptor(
        index_row["audit_record"], f"{label}.generation_audit_index.audit_record"
    )
    audit_marker = _descriptor(
        index_row["audit_completion_marker"],
        f"{label}.generation_audit_index.audit_completion_marker",
    )
    fallback_event = _descriptor(
        index_row["fallback_event"], f"{label}.generation_audit_index.fallback_event"
    )
    fallback_marker = _descriptor(
        index_row["fallback_completion_marker"],
        f"{label}.generation_audit_index.fallback_completion_marker",
    )
    runtime_event = _descriptor(row["runtime_event_log"], f"{label}.runtime_event_log")
    session_fallback_event = _descriptor(
        row["fallback_event_log"], f"{label}.fallback_event_log"
    )
    if runtime_event != audit_record:
        _fail(
            "scenario-capture-runtime-event-mismatch",
            f"{label} runtime event must be the indexed source audit record",
        )
    if session_fallback_event != fallback_event:
        _fail(
            "scenario-capture-fallback-event-mismatch",
            f"{label} fallback event must be the indexed source fallback event",
        )
    audit_directory = _replay_source_audit(
        root_fd,
        audit_record,
        audit_marker,
        capture_name=capture_name,
        expected_audit_directory=None,
        candidate_id=candidate_id,
        configuration_profile=configuration_profile,
        configuration_sha256=configuration_sha256,
        target=target,
        request_id=request_id,
        used_paths=used_paths,
        label=label,
    )
    _replay_fallback_source_pair(
        root_fd,
        audit_record=audit_record,
        audit_directory=audit_directory,
        fallback_event=fallback_event,
        fallback_marker=fallback_marker,
        candidate_id=candidate_id,
        configuration_profile=configuration_profile,
        configuration_sha256=configuration_sha256,
        target=target,
        request_id=request_id,
        used_paths=used_paths,
        label=label,
    )
    return ReplayedFallbackScenario(
        scenario_id=FALLBACK_SCENARIO_ID,
        target=target,
        request_ledger=ledger,
        runtime_event_log=runtime_event,
        generation_audit_index=index,
        fallback_event_log=session_fallback_event,
        request_id=request_id,
        audit_directory=audit_directory,
    )


def replay_raw_scenario_capture_v1_fd(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    *,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    used_paths: set[str],
) -> ReplayedScenarioCapture:
    """Replay a closed v1 serial capture with no current-process operations.

    All reads happen through ``root_fd``.  This function does not import the
    capture producer because that program deliberately owns socket and timing
    capabilities that a terminal raw binder must not gain.
    """

    capture_name = _capture_parent(descriptor, "scenario capture session")
    _assert_capture_marker_absent(root_fd, descriptor.path, "scenario capture session")
    _reserve(descriptor, label="scenario capture session", used_paths=used_paths)
    _raw, document = _read_json(root_fd, descriptor, "scenario capture session")
    row = _exact(
        document,
        {"schema_version", "capture_status", "qualification_status", "endpoint", "contract", "runtime_identity", "target", "scenarios"},
        "scenario capture session",
    )
    if row["schema_version"] != SCENARIO_CAPTURE_SESSION_VERSION:
        _fail("historical-scenario-capture-version-rejected", "scenario capture session has an unsupported version")
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
        _fail("invalid-capture-status", "scenario capture session must be captured/not-run")
    if type(row["endpoint"]) is not str:
        _fail("invalid-completion-endpoint", "scenario capture endpoint must be literal loopback completions")
    endpoint_match = COMPLETION_ENDPOINT_RE.fullmatch(row["endpoint"])
    if endpoint_match is None:
        _fail("invalid-completion-endpoint", "scenario capture endpoint must be literal loopback completions")
    target = _capture_target(row["target"], "scenario capture session.target")
    if int(endpoint_match.group(1)) != target.listener_port:
        _fail("scenario-capture-endpoint-target-mismatch", "scenario capture endpoint port differs from its target tuple")
    identity = _exact(
        row["runtime_identity"],
        {"configuration_profile", "configuration_sha256"},
        "scenario capture session.runtime_identity",
    )
    if identity["configuration_profile"] != configuration_profile or _sha256(
        identity["configuration_sha256"], "scenario capture session.runtime_identity.configuration_sha256"
    ) != configuration_sha256:
        _fail("scenario-capture-runtime-identity-mismatch", "scenario capture runtime identity drifted")
    contract = _descriptor(row["contract"], "scenario capture session.contract")
    contract_rows = _replay_capture_contract(
        root_fd,
        contract,
        capture_name=capture_name,
        candidate_id=candidate_id,
        configuration_profile=configuration_profile,
        used_paths=used_paths,
    )
    scenarios = row["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != len(contract_rows):
        _fail("scenario-capture-inventory-mismatch", "scenario capture does not preserve the contract inventory")
    replayed: list[ReplayedScenario] = []
    request_ids: set[str] = set()
    audit_directory: str | None = None
    for sequence, (scenario_id, completion_request) in enumerate(contract_rows):
        scenario = _replay_capture_scenario(
            root_fd,
            scenarios[sequence],
            capture_name=capture_name,
            sequence=sequence,
            expected_scenario_id=scenario_id,
            completion_request=completion_request,
            candidate_id=candidate_id,
            configuration_profile=configuration_profile,
            configuration_sha256=configuration_sha256,
            expected_target=target,
            expected_audit_directory=audit_directory,
            used_paths=used_paths,
        )
        if scenario.request_id in request_ids:
            _fail("scenario-capture-request-id-reuse", "scenario capture reuses a source request ID")
        request_ids.add(scenario.request_id)
        audit_directory = scenario.audit_directory
        replayed.append(scenario)
    if audit_directory is None:
        _fail("invalid-scenario-inventory", "scenario capture must bind one source audit directory")
    return ReplayedScenarioCapture(
        session=descriptor,
        contract=contract,
        target=target,
        scenarios=tuple(replayed),
        audit_directory=audit_directory,
    )


def replay_raw_scenario_capture_v2_fd(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    *,
    candidate_id: str,
    configuration_profile: str,
    configuration_sha256: str,
    used_paths: set[str],
) -> ReplayedFallbackScenarioCapture:
    """Replay the closed v2 native-fallback capture through one held root FD.

    This is deliberately separate from the retained v1 replay.  It accepts
    exactly one source-issued fallback scenario and derives every audit/event
    descriptor from the capture instead of treating source leaves as opaque.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd, "native fallback capture evidence root"
        )
    )
    capture_name = _capture_parent(descriptor, "native fallback scenario capture session")
    _assert_capture_marker_absent(
        root_fd,
        descriptor.path,
        "native fallback scenario capture session",
    )
    _reserve(descriptor, label="native fallback scenario capture session", used_paths=used_paths)
    _raw, document = _read_json(root_fd, descriptor, "native fallback scenario capture session")
    row = _exact(
        document,
        {
            "schema_version", "capture_status", "qualification_status", "endpoint",
            "contract", "runtime_identity", "target", "scenarios",
        },
        "native fallback scenario capture session",
    )
    if row["schema_version"] != SCENARIO_FALLBACK_CAPTURE_SESSION_VERSION:
        _fail(
            "historical-fallback-scenario-capture-version-rejected",
            "native fallback scenario capture has an unsupported version",
        )
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
        _fail(
            "invalid-capture-status",
            "native fallback scenario capture must be captured/not-run",
        )
    if configuration_profile != MAX_PERFORMANCE_EXACT_PROFILE:
        _fail(
            "fallback-profile-mismatch",
            "native fallback scenario capture requires max-performance-exact",
        )
    if type(row["endpoint"]) is not str:
        _fail(
            "invalid-completion-endpoint",
            "native fallback scenario endpoint must be literal loopback completions",
        )
    endpoint_match = COMPLETION_ENDPOINT_RE.fullmatch(row["endpoint"])
    if endpoint_match is None:
        _fail(
            "invalid-completion-endpoint",
            "native fallback scenario endpoint must be literal loopback completions",
        )
    target = _capture_target(row["target"], "native fallback scenario capture session.target")
    if int(endpoint_match.group(1)) != target.listener_port:
        _fail(
            "scenario-capture-endpoint-target-mismatch",
            "native fallback endpoint port differs from its target tuple",
        )
    identity = _exact(
        row["runtime_identity"],
        {"configuration_profile", "configuration_sha256"},
        "native fallback scenario capture session.runtime_identity",
    )
    if (
        identity["configuration_profile"] != MAX_PERFORMANCE_EXACT_PROFILE
        or _sha256(
            identity["configuration_sha256"],
            "native fallback scenario capture session.runtime_identity.configuration_sha256",
        )
        != configuration_sha256
    ):
        _fail(
            "scenario-capture-runtime-identity-mismatch",
            "native fallback scenario capture runtime identity drifted",
        )
    contract = _descriptor(
        row["contract"], "native fallback scenario capture session.contract"
    )
    contract_rows = _replay_fallback_capture_contract(
        root_fd,
        contract,
        capture_name=capture_name,
        candidate_id=candidate_id,
        configuration_profile=configuration_profile,
        used_paths=used_paths,
    )
    scenarios = row["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 1 or len(contract_rows) != 1:
        _fail(
            "scenario-capture-inventory-mismatch",
            "native fallback scenario capture must preserve one contract scenario",
        )
    scenario_id, completion_request = contract_rows[0]
    if scenario_id != FALLBACK_SCENARIO_ID:
        _fail("invalid-scenario-inventory", "native fallback capture contract drifted")
    scenario = _replay_fallback_capture_scenario(
        root_fd,
        scenarios[0],
        capture_name=capture_name,
        expected_target=target,
        candidate_id=candidate_id,
        configuration_profile=configuration_profile,
        configuration_sha256=configuration_sha256,
        completion_request=completion_request,
        used_paths=used_paths,
    )
    return ReplayedFallbackScenarioCapture(
        session=descriptor,
        contract=contract,
        target=target,
        scenarios=(scenario,),
        audit_directory=scenario.audit_directory,
    )


def _read_soak_v4_completion_marker(
    root_fd: int,
    manifest: common.EvidenceDescriptor,
) -> None:
    manifest_name = _soak_terminal_manifest_name(
        manifest.path, "completed v4 soak manifest path"
    )
    marker_name = f"{manifest_name}.complete"
    intent_name = f"{manifest_name}.intent"
    try:
        marker_raw = _common(
            lambda: common.read_bounded_paired_hardlink(
                root_fd,
                marker_name,
                intent_name,
                "v4 soak raw manifest completion marker",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
    except C02ProvenanceError as error:
        if getattr(error, "reason_code", None) == "missing-input":
            _fail(
                "missing-soak-v4-completion-marker",
                f"v4 soak raw manifest requires exact sibling marker pair "
                f"{marker_name!r} and {intent_name!r}",
            )
        raise
    marker = _common(
        lambda: common.parse_canonical_json(
            marker_raw,
            "v4 soak raw manifest completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    assert isinstance(marker, dict)
    row = _exact(
        marker,
        {"schema_version", "artifact_filename", "artifact_sha256"},
        "v4 soak raw manifest completion marker",
    )
    if row["schema_version"] != SOAK_V4_COMPLETION_MARKER_VERSION:
        _fail(
            "historical-soak-v4-completion-version-rejected",
            "v4 soak raw manifest completion marker has an unsupported schema version",
        )
    if row["artifact_filename"] != manifest_name or _sha256(
        row["artifact_sha256"], "v4 soak raw manifest completion marker.artifact_sha256"
    ) != manifest.sha256:
        _fail(
            "soak-v4-completion-marker-mismatch",
            "v4 soak raw manifest completion marker does not bind exact manifest bytes",
        )


def verify_soak_provenance_v4_fd(root_fd: int, manifest_path: str) -> dict[str, Any]:
    """Verify a raw v4 serial scenario manifest through one held root FD.

    Unlike the v3 verifier, v4 does not accept opaque workload/audit leaves:
    all such descriptors are re-derived and replayed from the one source-owned
    serial capture session before they can appear in a manifest.
    """

    manifest_descriptor, document = _read_manifest(root_fd, manifest_path, "v4 soak raw manifest")
    row = _exact(
        document,
        {
            "schema_version", "capture_status", "qualification_status", "candidate_id",
            "bindings", "configuration_evidence", "scenario_capture_session",
            "scenario_contract", "scenarios",
        },
        "v4 soak raw manifest",
    )
    if row["schema_version"] != SOAK_V4_MANIFEST_VERSION:
        if row["schema_version"] == SOAK_MANIFEST_VERSION:
            _fail("historical-soak-v3-rejected", f"v4 soak raw manifest must use {SOAK_V4_MANIFEST_VERSION}")
        if row["schema_version"] == "riley.soak-v2-raw-provenance.v2":
            _fail("historical-soak-v2-rejected", f"v4 soak raw manifest must use {SOAK_V4_MANIFEST_VERSION}")
        _fail("historical-soak-v1-rejected", f"v4 soak raw manifest must use {SOAK_V4_MANIFEST_VERSION}")
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
        _fail("invalid-capture-status", "v4 soak raw manifest must be captured/not-run")
    candidate = _candidate_id(row["candidate_id"], "v4 soak raw manifest.candidate_id")
    bindings = _bindings(
        row["bindings"],
        "v4 soak raw manifest.bindings",
        allowed_profiles=SOAK_CONFIGURATION_PROFILES,
    )
    used = {manifest_descriptor.path}
    configuration_target = _load_soak_configuration_evidence(
        root_fd,
        row["configuration_evidence"],
        candidate_id=candidate,
        bindings=bindings,
        used_paths=used,
        require_direct_observation_session=True,
    )
    capture_descriptor = _descriptor(
        row["scenario_capture_session"],
        "v4 soak raw manifest.scenario_capture_session",
    )
    capture = replay_raw_scenario_capture_v1_fd(
        root_fd,
        capture_descriptor,
        candidate_id=candidate,
        configuration_profile=bindings["configuration_profile"],
        configuration_sha256=bindings["configuration_sha256"],
        used_paths=used,
    )
    if not _capture_matches_observed(capture.target, configuration_target):
        _fail(
            "configuration-scenario-capture-target-mismatch",
            "serial capture does not share the configuration bridge PID/start-tick/listener tuple",
        )
    contract = _descriptor(row["scenario_contract"], "v4 soak raw manifest.scenario_contract")
    if contract != capture.contract:
        _fail(
            "scenario-capture-contract-descriptor-mismatch",
            "v4 manifest contract must be the descriptor derived from its serial capture",
        )
    scenarios = row["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != len(capture.scenarios):
        _fail("scenario-capture-inventory-mismatch", "v4 manifest must preserve the serial capture inventory")
    targets: list[dict[str, Any]] = []
    for index, captured in enumerate(capture.scenarios):
        label = f"v4 soak raw manifest.scenarios[{index}]"
        scenario = _exact(
            scenarios[index],
            {
                "scenario_id", "target", "observation_session", "request_ledger",
                "runtime_event_log", "generation_audit_index",
            },
            label,
        )
        scenario_id = scenario["scenario_id"]
        if (
            type(scenario_id) is not str
            or scenario_id == "exact-backend-fallback"
            or scenario_id != captured.scenario_id
        ):
            _fail("scenario-capture-inventory-mismatch", f"{label} does not preserve serial capture scenario ID/order")
        declared_target = _target(scenario["target"], f"{label}.target")
        if _descriptor(scenario["request_ledger"], f"{label}.request_ledger") != captured.request_ledger:
            _fail("scenario-capture-derived-leaf-mismatch", f"{label} request ledger was not derived from the capture")
        if _descriptor(scenario["runtime_event_log"], f"{label}.runtime_event_log") != captured.runtime_event_log:
            _fail("scenario-capture-derived-leaf-mismatch", f"{label} runtime event was not derived from the capture")
        if _descriptor(scenario["generation_audit_index"], f"{label}.generation_audit_index") != captured.generation_audit_index:
            _fail("scenario-capture-derived-leaf-mismatch", f"{label} audit index was not derived from the capture")
        observation = _descriptor(scenario["observation_session"], f"{label}.observation_session")
        _require_direct_v4_session_path(observation, f"{label}.observation_session")
        _reserve(observation, label=f"{label}.observation_session", used_paths=used)
        observed_target = _load_session(root_fd, observation, label, used)
        if declared_target != observed_target.target:
            _fail("session-target-mismatch", f"{label} target differs from its C02 observation session")
        if not _capture_matches_observed(captured.target, observed_target):
            _fail(
                "scenario-capture-observation-target-mismatch",
                f"{label} serial capture does not share the observation PID/start-tick/listener tuple",
            )
        if observed_target != configuration_target:
            _fail(
                "configuration-scenario-target-mismatch",
                f"{label} does not share the configuration bridge PID/start-tick/listener/GPU tuple",
            )
        targets.append({"scenario_id": scenario_id, "target": declared_target.as_json()})
    return _report(
        schema_version=SOAK_V4_REPORT_VERSION,
        manifest=manifest_descriptor,
        candidate_id=candidate,
        bindings=bindings,
        targets=targets,
        check_names=(
            "v4-version-only",
            "canonical-descriptor-binding",
            "capture-marker-closure",
            "serial-contract-request-response-audit-replay",
            "source-audit-marker-binding",
            "pid-start-tick-listener-gpu-binding",
            "runtime-config-candidate-profile-binding",
            "configuration-serial-observation-process-bridge",
        ),
    )


def verify_soak_provenance_v4(evidence_root: Path, manifest_path: str) -> dict[str, Any]:
    root_fd = _open_private_evidence_root(evidence_root, "evidence root")
    try:
        return verify_soak_provenance_v4_fd(root_fd, manifest_path)
    finally:
        os.close(root_fd)


def verify_completed_soak_provenance_v4_fd(
    root_fd: int,
    manifest_path: str,
) -> dict[str, Any]:
    manifest_name = _soak_terminal_manifest_name(
        manifest_path, "completed v4 soak manifest path"
    )
    report = verify_soak_provenance_v4_fd(root_fd, manifest_name)
    manifest = _descriptor(report["raw_manifest"], "completed v4 soak manifest descriptor")
    _read_soak_v4_completion_marker(root_fd, manifest)
    final_manifest, _final_document = _read_manifest(
        root_fd, manifest_name, "completed v4 soak manifest final revalidation"
    )
    if final_manifest != manifest:
        _fail(
            "soak-v4-manifest-changed-during-completion-verification",
            "v4 soak manifest changed while its completion marker was verified",
        )
    _read_soak_v4_completion_marker(root_fd, final_manifest)
    return report


def verify_completed_soak_provenance_v4(
    evidence_root: Path,
    manifest_path: str,
) -> dict[str, Any]:
    root_fd = _open_private_evidence_root(evidence_root, "evidence root")
    try:
        return verify_completed_soak_provenance_v4_fd(root_fd, manifest_path)
    finally:
        os.close(root_fd)


def _read_soak_v5_completion_marker(
    root_fd: int,
    manifest: common.EvidenceDescriptor,
) -> None:
    manifest_name = _soak_terminal_manifest_name(
        manifest.path, "completed v5 soak manifest path"
    )
    marker_name = f"{manifest_name}.complete"
    intent_name = f"{manifest_name}.intent"
    try:
        marker_raw = _common(
            lambda: common.read_bounded_paired_hardlink(
                root_fd,
                marker_name,
                intent_name,
                "v5 soak raw manifest completion marker",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
    except C02ProvenanceError as error:
        if getattr(error, "reason_code", None) == "missing-input":
            _fail(
                "missing-soak-v5-completion-marker",
                f"v5 soak raw manifest requires exact sibling marker pair "
                f"{marker_name!r} and {intent_name!r}",
            )
        raise
    marker = _common(
        lambda: common.parse_canonical_json(
            marker_raw,
            "v5 soak raw manifest completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    assert isinstance(marker, dict)
    row = _exact(
        marker,
        {"schema_version", "artifact_filename", "artifact_sha256"},
        "v5 soak raw manifest completion marker",
    )
    if row["schema_version"] != SOAK_V5_COMPLETION_MARKER_VERSION:
        _fail(
            "historical-soak-v5-completion-version-rejected",
            "v5 soak raw manifest completion marker has an unsupported schema version",
        )
    if row["artifact_filename"] != manifest_name or _sha256(
        row["artifact_sha256"], "v5 soak raw manifest completion marker.artifact_sha256"
    ) != manifest.sha256:
        _fail(
            "soak-v5-completion-marker-mismatch",
            "v5 soak raw manifest completion marker does not bind exact manifest bytes",
        )


def verify_soak_provenance_v5_fd(root_fd: int, manifest_path: str) -> dict[str, Any]:
    """Verify a raw v5 native-fallback manifest through one private root FD."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd, "v5 soak provenance evidence root"
        )
    )
    manifest_descriptor, document = _read_manifest(root_fd, manifest_path, "v5 soak raw manifest")
    row = _exact(
        document,
        {
            "schema_version", "capture_status", "qualification_status", "candidate_id",
            "bindings", "configuration_evidence", "scenario_capture_session",
            "scenario_contract", "scenarios",
        },
        "v5 soak raw manifest",
    )
    if row["schema_version"] != SOAK_V5_MANIFEST_VERSION:
        _fail(
            "historical-soak-v5-manifest-version-rejected",
            f"v5 soak raw manifest must use {SOAK_V5_MANIFEST_VERSION}",
        )
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
        _fail("invalid-capture-status", "v5 soak raw manifest must be captured/not-run")
    candidate = _candidate_id(row["candidate_id"], "v5 soak raw manifest.candidate_id")
    bindings = _bindings(
        row["bindings"],
        "v5 soak raw manifest.bindings",
        allowed_profiles=frozenset((MAX_PERFORMANCE_EXACT_PROFILE,)),
    )
    if bindings["configuration_profile"] != MAX_PERFORMANCE_EXACT_PROFILE:
        _fail(
            "fallback-profile-mismatch",
            "v5 native fallback manifest requires max-performance-exact",
        )
    used = {manifest_descriptor.path}
    configuration_target = _load_soak_configuration_evidence(
        root_fd,
        row["configuration_evidence"],
        candidate_id=candidate,
        bindings=bindings,
        used_paths=used,
        require_direct_observation_session=True,
        require_gpu_greedy=True,
    )
    capture_descriptor = _descriptor(
        row["scenario_capture_session"], "v5 soak raw manifest.scenario_capture_session"
    )
    capture = replay_raw_scenario_capture_v2_fd(
        root_fd,
        capture_descriptor,
        candidate_id=candidate,
        configuration_profile=bindings["configuration_profile"],
        configuration_sha256=bindings["configuration_sha256"],
        used_paths=used,
    )
    if not _capture_matches_observed(capture.target, configuration_target):
        _fail(
            "configuration-scenario-capture-target-mismatch",
            "native fallback capture does not share the configuration bridge PID/start-tick/listener tuple",
        )
    contract = _descriptor(row["scenario_contract"], "v5 soak raw manifest.scenario_contract")
    if contract != capture.contract:
        _fail(
            "scenario-capture-contract-descriptor-mismatch",
            "v5 manifest contract must be derived from its native fallback capture",
        )
    scenarios = row["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 1 or len(capture.scenarios) != 1:
        _fail(
            "scenario-capture-inventory-mismatch",
            "v5 manifest must preserve exactly one native fallback scenario",
        )
    captured = capture.scenarios[0]
    scenario = _exact(
        scenarios[0],
        {
            "scenario_id", "target", "observation_session", "request_ledger",
            "runtime_event_log", "generation_audit_index", "fallback_event_log",
        },
        "v5 soak raw manifest.scenarios[0]",
    )
    if scenario["scenario_id"] != FALLBACK_SCENARIO_ID or captured.scenario_id != FALLBACK_SCENARIO_ID:
        _fail(
            "scenario-capture-inventory-mismatch",
            "v5 manifest must preserve exact-backend-fallback",
        )
    declared_target = _target(scenario["target"], "v5 soak raw manifest.scenarios[0].target")
    for key, expected in (
        ("request_ledger", captured.request_ledger),
        ("runtime_event_log", captured.runtime_event_log),
        ("generation_audit_index", captured.generation_audit_index),
        ("fallback_event_log", captured.fallback_event_log),
    ):
        if _descriptor(
            scenario[key], f"v5 soak raw manifest.scenarios[0].{key}"
        ) != expected:
            _fail(
                "scenario-capture-derived-leaf-mismatch",
                f"v5 manifest {key} was not derived from the native fallback capture",
            )
    observation = _descriptor(
        scenario["observation_session"], "v5 soak raw manifest.scenarios[0].observation_session"
    )
    _require_direct_v4_session_path(observation, "v5 soak raw manifest.scenarios[0].observation_session")
    _reserve(observation, label="v5 soak raw manifest.scenarios[0].observation_session", used_paths=used)
    observed_target = _load_session(
        root_fd,
        observation,
        "v5 soak raw manifest.scenarios[0].observation_session",
        used,
    )
    if declared_target != observed_target.target:
        _fail(
            "session-target-mismatch",
            "v5 native fallback target differs from its C02 observation session",
        )
    if not _capture_matches_observed(captured.target, observed_target):
        _fail(
            "scenario-capture-observation-target-mismatch",
            "v5 native fallback capture does not share the observation PID/start-tick/listener tuple",
        )
    if observed_target != configuration_target:
        _fail(
            "configuration-scenario-target-mismatch",
            "v5 native fallback scenario does not share the configuration bridge PID/start-tick/listener/GPU tuple",
        )
    return _report(
        schema_version=SOAK_V5_REPORT_VERSION,
        manifest=manifest_descriptor,
        candidate_id=candidate,
        bindings=bindings,
        targets=[{"scenario_id": FALLBACK_SCENARIO_ID, "target": declared_target.as_json()}],
        check_names=(
            "v5-version-only",
            "canonical-descriptor-binding",
            "capture-v2-incomplete-marker-closure",
            "single-fallback-contract-request-response-replay",
            "source-audit-and-fallback-marker-binding",
            "ordered-native-fallback-transition-binding",
            "effective-config-gpu-greedy-binding",
            "configuration-serial-observation-process-gpu-bridge",
        ),
    )


def verify_soak_provenance_v5(
    evidence_root: Path,
    manifest_path: str,
) -> dict[str, Any]:
    root_fd = _open_private_evidence_root(evidence_root, "evidence root")
    try:
        return verify_soak_provenance_v5_fd(root_fd, manifest_path)
    finally:
        os.close(root_fd)


def verify_completed_soak_provenance_v5_fd(
    root_fd: int,
    manifest_path: str,
) -> dict[str, Any]:
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd, "completed v5 soak provenance evidence root"
        )
    )
    manifest_name = _soak_terminal_manifest_name(
        manifest_path, "completed v5 soak manifest path"
    )
    report = verify_soak_provenance_v5_fd(root_fd, manifest_name)
    manifest = _descriptor(report["raw_manifest"], "completed v5 soak manifest descriptor")
    _read_soak_v5_completion_marker(root_fd, manifest)
    final_manifest, _final_document = _read_manifest(
        root_fd, manifest_name, "completed v5 soak manifest final revalidation"
    )
    if final_manifest != manifest:
        _fail(
            "soak-v5-manifest-changed-during-completion-verification",
            "v5 soak manifest changed while its completion marker was verified",
        )
    _read_soak_v5_completion_marker(root_fd, final_manifest)
    return report


def verify_completed_soak_provenance_v5(
    evidence_root: Path,
    manifest_path: str,
) -> dict[str, Any]:
    root_fd = _open_private_evidence_root(evidence_root, "evidence root")
    try:
        return verify_completed_soak_provenance_v5_fd(root_fd, manifest_path)
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
                _verify_c02_shutdown_v2_descriptors_fd(
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
    parser.add_argument(
        "--kind",
        required=True,
        choices=("soak", "soak-v4", "soak-v5", "rollback"),
    )
    parser.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.kind == "soak":
            report = verify_completed_soak_provenance(args.evidence_root, args.manifest)
        elif args.kind == "soak-v4":
            report = verify_completed_soak_provenance_v4(args.evidence_root, args.manifest)
        elif args.kind == "soak-v5":
            report = verify_completed_soak_provenance_v5(args.evidence_root, args.manifest)
        else:
            report = verify_rollback_provenance(args.evidence_root, args.manifest)
    except (C02ProvenanceError, common.ProvenanceV2Error) as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
