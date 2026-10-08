#!/usr/bin/env python3
"""Raw-only reconstructed-baseline rollback provenance v3 replay.

This checker is deliberately below the future rollback semantic checker.  It
binds exact, checked raw leaves for one candidate phase and one reconstructed
baseline phase through a caller-held private evidence-root FD.  In particular,
it validates the raw process/socket/GPU identity carried by those leaves but
does not interpret HTTP health or generation responses, attest an atomic
switch, accept a candidate, or emit a rollback verdict.

The baseline is represented by its binary-bound reconstructed-prior-baseline
v2 manifest.  This raw layer replays that manifest through the same held root
FD before it binds the rollback phases.  It still does not interpret a
rollback as successful or claim that a reconstructed tag was a historical
shipped release.  The candidate audit index is likewise an exact raw
descriptor here; its source-audit content is replayed only by a later
semantic/provenance layer.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import check_c02_provenance_v2 as c02
import check_reconstructed_prior_baseline_v2 as baseline
import provenance_v2_common as common


ROLLBACK_V3_MANIFEST_VERSION = "riley.rc3-rollback-raw-provenance.v3"
ROLLBACK_V3_REPORT_VERSION = "riley.rc3-rollback-provenance-check.v3"
STABLE_DEFAULT_PROFILE = "stable-default"
RECONSTRUCTED_ROLLBACK_TAG = "riley-0.1.0-rc2"
RECONSTRUCTED_ROLLBACK_TAG_OBJECT = "a3f5203c3a72122e9da818c1e441c2a789f7aa8c"
RECONSTRUCTED_ROLLBACK_TARGET = "6093006ec2b01b784b01ba278296b676f2dfd03a"
RECONSTRUCTED_ROLLBACK_BASELINE_ID = f"reconstructed-{RECONSTRUCTED_ROLLBACK_TAG}"
MAX_MANIFEST_BYTES = common.DEFAULT_MAX_JSON_BYTES
MAX_RAW_LEAF_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = common.DEFAULT_MAX_ARTIFACT_BYTES
MAX_RELATIVE_PATH_BYTES = 512

CANDIDATE_RE = re.compile(
    r"^riley-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)-rc[1-9][0-9]*$"
)
GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
UINT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")

RAW_PROCESS_FIELDS = frozenset(
    {
        "pre_stat",
        "post_stat",
        "pre_tcp",
        "post_tcp",
        "pre_fd_sockets",
        "post_fd_sockets",
        "status",
        "gpu_selection",
        "gpu_compute_apps",
    }
)
HTTP_EXCHANGE_FIELDS = frozenset({"request", "response_head", "response_body"})
ARTIFACT_FIELDS = frozenset({"binary", "bundle", "image_inspect"})
ATOMIC_SWITCH_FIELDS = frozenset(
    {
        "pre_active_stat",
        "post_active_stat",
        "candidate_staged_stat",
        "rollback_staged_stat",
        "rename_transcript",
    }
)


class RollbackV3ProvenanceError(ValueError):
    """Raw rollback v3 evidence cannot establish a safe provenance binding."""


def _fail(code: str, message: str) -> NoReturn:
    error = RollbackV3ProvenanceError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _baseline(call: Callable[[], T]) -> T:
    try:
        return call()
    except baseline.BaselineError as error:
        _fail(getattr(error, "reason_code", "invalid-reconstructed-baseline"), str(error))
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "invalid-reconstructed-baseline"), str(error))


def _c02(call: Callable[[], T]) -> T:
    try:
        return call()
    except c02.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-candidate-shutdown"), str(error))
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "invalid-candidate-shutdown"), str(error))


def _c02_raw(call: Callable[[], T]) -> T:
    """Translate the shared raw-leaf grammar errors without widening C02."""

    try:
        return call()
    except c02.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-raw-process-evidence"), str(error))
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "invalid-raw-process-evidence"), str(error))


def _exact(value: Any, fields: frozenset[str] | set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
        )
    return value


def _string(value: Any, label: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > maximum:
        _fail("invalid-string", f"{label} must be a bounded non-empty string")
    return value


def _candidate_id(value: Any, label: str) -> str:
    value = _string(value, label, maximum=128)
    if CANDIDATE_RE.fullmatch(value) is None:
        _fail("invalid-candidate-id", f"{label} must be a canonical Riley RC ID")
    return value


def _positive(value: Any, label: str, *, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        _fail("invalid-integer", f"{label} must be a positive integer in range")
    return value


def _nonnegative(value: Any, label: str, *, maximum: int = 2**31 - 1) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        _fail("invalid-integer", f"{label} must be a non-negative integer in range")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or common.SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail("invalid-sha256", f"{label} must be a non-zero lowercase SHA-256")
    return value


def _relative_path(value: Any, label: str) -> str:
    path = _common(lambda: common.validate_relative_path(value, label))
    if len(path) > MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label} exceeds {MAX_RELATIVE_PATH_BYTES} bytes",
        )
    return path


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    """Reject a source-tree child before opening a provenance evidence root.

    The held-FD common primitive remains authoritative for no-follow path
    traversal.  This lexical preflight only prevents this checker from
    treating its own checkout as a private evidence publication location.
    """

    source_root = Path(__file__).resolve().parents[2]
    try:
        evidence_root.relative_to(source_root)
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be external to the source checkout",
    )


@dataclass(frozen=True)
class Target:
    pid: int
    start_ticks: int
    listener_port: int
    listener_inode: int
    gpu_index: int
    gpu_uuid: str

    def as_json(self) -> dict[str, Any]:
        return {
            "server_pid": self.pid,
            "server_start_ticks": self.start_ticks,
            "listener_port": self.listener_port,
            "listener_inode": self.listener_inode,
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
        }


def _target(value: Any, label: str) -> Target:
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
    gpu_uuid = _string(row["gpu_uuid"], f"{label}.gpu_uuid", maximum=128)
    if GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        _fail("invalid-gpu-uuid", f"{label}.gpu_uuid must be a canonical GPU UUID")
    return Target(
        pid=_positive(row["server_pid"], f"{label}.server_pid"),
        start_ticks=_positive(
            row["server_start_ticks"], f"{label}.server_start_ticks"
        ),
        listener_port=_positive(
            row["listener_port"], f"{label}.listener_port", maximum=65535
        ),
        listener_inode=_positive(
            row["listener_inode"], f"{label}.listener_inode"
        ),
        gpu_index=_nonnegative(row["gpu_index"], f"{label}.gpu_index"),
        gpu_uuid=gpu_uuid,
    )


def _reserve_descriptor(
    root_fd: int,
    value: Any,
    label: str,
    *,
    used_paths: set[str],
    maximum_bytes: int,
    minimum_bytes: int = 1,
) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if descriptor.path in used_paths:
        _fail(
            "duplicate-evidence-path",
            f"{label} reuses evidence path {descriptor.path!r}",
        )
    if descriptor.byte_length < minimum_bytes:
        _fail("empty-evidence-leaf", f"{label} must not be empty")
    if descriptor.byte_length > maximum_bytes:
        _fail("input-too-large", f"{label} exceeds its byte bound")
    used_paths.add(descriptor.path)
    _common(
        lambda: common.verify_descriptor_file(
            root_fd,
            descriptor,
            label,
            maximum_bytes=maximum_bytes,
        )
    )
    return descriptor


def _raw_descriptor_map(
    root_fd: int,
    value: Any,
    fields: frozenset[str],
    label: str,
    *,
    used_paths: set[str],
) -> dict[str, common.EvidenceDescriptor]:
    row = _exact(value, fields, label)
    result: dict[str, common.EvidenceDescriptor] = {}
    for name in sorted(fields):
        result[name] = _reserve_descriptor(
            root_fd,
            row[name],
            f"{label}.{name}",
            used_paths=used_paths,
            maximum_bytes=MAX_RAW_LEAF_BYTES,
        )
    return result


def _artifact_map(
    root_fd: int,
    value: Any,
    label: str,
    *,
    used_paths: set[str],
) -> dict[str, common.EvidenceDescriptor]:
    row = _exact(value, ARTIFACT_FIELDS, label)
    return {
        name: _reserve_descriptor(
            root_fd,
            row[name],
            f"{label}.{name}",
            used_paths=used_paths,
            maximum_bytes=MAX_ARTIFACT_BYTES,
        )
        for name in sorted(ARTIFACT_FIELDS)
    }


def _bind_reconstructed_baseline_artifacts(
    root_fd: int,
    rollback_artifacts: Mapping[str, common.EvidenceDescriptor],
    baseline_report: Mapping[str, Any],
) -> None:
    """Bind active rollback artifacts to the A/B binary-bound reconstruction.

    The v2 baseline establishes equal A/B server-binary and bundle descriptor
    bytes plus one Docker image ID. This raw layer joins those exact facts to
    the active rollback leaves without making a historical-release or
    rollback-success claim.
    """

    equality = baseline_report.get("equality")
    if not isinstance(equality, Mapping):
        _fail(
            "invalid-reconstructed-baseline-report",
            "reconstructed baseline report lacks its equality section",
        )
    binary = equality.get("binary")
    bundle = equality.get("bundle")
    oci_image = equality.get("oci_image")
    if (
        not isinstance(binary, Mapping)
        or not isinstance(bundle, Mapping)
        or not isinstance(oci_image, Mapping)
    ):
        _fail(
            "invalid-reconstructed-baseline-report",
            "reconstructed baseline report lacks binary, bundle, or OCI image equality",
        )
    expected_binary = _common(
        lambda: common.parse_descriptor(
            binary.get("a"),
            "reconstructed baseline equality binary A",
        )
    )
    if (
        rollback_artifacts["binary"].sha256 != expected_binary.sha256
        or rollback_artifacts["binary"].byte_length != expected_binary.byte_length
    ):
        _fail(
            "baseline-binary-binding-mismatch",
            "rollback active binary does not match the reconstructed A/B server binary",
        )
    expected_bundle = _common(
        lambda: common.parse_descriptor(
            bundle.get("a"),
            "reconstructed baseline equality bundle A",
        )
    )
    if (
        rollback_artifacts["bundle"].sha256 != expected_bundle.sha256
        or rollback_artifacts["bundle"].byte_length != expected_bundle.byte_length
    ):
        _fail(
            "baseline-bundle-binding-mismatch",
            "rollback active bundle does not match the reconstructed A/B bundle",
        )
    expected_image_id = oci_image.get("image_id")
    if not isinstance(expected_image_id, str):
        _fail(
            "invalid-reconstructed-baseline-report",
            "reconstructed baseline report lacks its OCI image ID",
        )
    active_image_id = _baseline(
        lambda: baseline._read_runtime_image_inspect_id(  # noqa: SLF001
            root_fd,
            rollback_artifacts["image_inspect"],
            "rollback active raw image inspect",
        )
    )
    if active_image_id != expected_image_id:
        _fail(
            "baseline-image-binding-mismatch",
            "rollback active image inspect does not match the reconstructed A/B image",
        )


def _audit(
    root_fd: int,
    value: Any,
    label: str,
    *,
    source_owned: bool,
    used_paths: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("unknown-or-missing-field", f"{label} must be an object")
    if source_owned:
        if value.get("availability") != "source-owned":
            _fail(
                "invalid-audit-availability",
                f"{label}.availability must be source-owned for the candidate phase",
            )
        row = _exact(value, {"availability", "generation_audit_index"}, label)
        descriptor = _reserve_descriptor(
            root_fd,
            row["generation_audit_index"],
            f"{label}.generation_audit_index",
            used_paths=used_paths,
            maximum_bytes=MAX_RAW_LEAF_BYTES,
        )
        return {
            "availability": "source-owned",
            "generation_audit_index": descriptor,
        }
    if value.get("availability") != "not-supported":
        _fail(
            "invalid-audit-availability",
            f"{label}.availability must be not-supported for the reconstructed baseline",
        )
    row = _exact(value, {"availability"}, label)
    return {"availability": "not-supported"}


def _bound_loopback_listener_from_tcp(
    raw: bytes,
    socket_inodes: set[int],
    label: str,
) -> tuple[int, int]:
    """Derive the one loopback listener owned by the captured server FD set."""

    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("invalid-proc-net-tcp", f"{label} is not ASCII: {error}")
    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("sl"):
        _fail("invalid-proc-net-tcp", f"{label} has no valid header")
    candidates: set[tuple[int, int]] = set()
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 10:
            _fail("invalid-proc-net-tcp", f"{label} has a malformed socket row")
        local, remote, state, inode_text = fields[1], fields[2], fields[3], fields[9]
        if remote.upper() != "00000000:0000" or state.upper() != "0A":
            continue
        if ":" not in local or UINT_RE.fullmatch(inode_text) is None:
            _fail("invalid-proc-net-tcp", f"{label} has a malformed listener row")
        address, port_text = local.upper().rsplit(":", 1)
        if re.fullmatch(r"[0-9A-F]{4}", port_text) is None:
            _fail("invalid-proc-net-tcp", f"{label} listener has an invalid port")
        inode = int(inode_text)
        port = int(port_text, 16)
        if inode < 1 or port < 1:
            _fail("invalid-proc-net-tcp", f"{label} listener has a zero port or inode")
        if inode not in socket_inodes:
            continue
        if address == "00000000":
            _fail("wildcard-listener", f"{label} contains a wildcard server listener")
        if address != "0100007F":
            continue
        candidate = (port, inode)
        if candidate in candidates:
            _fail("invalid-proc-net-tcp", f"{label} repeats a listener tuple")
        candidates.add(candidate)
    if len(candidates) != 1:
        _fail(
            "listener-proof-missing",
            f"{label} must prove exactly one loopback listener owned by the server",
        )
    return next(iter(candidates))


def derive_phase_target_from_raw_bytes(
    process_evidence: Mapping[str, bytes],
    label: str,
) -> Target:
    """Derive one phase target from one already-pinned raw-byte snapshot.

    The caller owns acquisition and descriptor verification.  This pure parser
    exists so an FD-only replayer can derive the target from the same leaf bytes
    it just consumed, rather than verifying a held child and then reopening its
    root-relative names.  It is provenance identity parsing, not HTTP/audit or
    rollback semantics.
    """

    if set(process_evidence) != set(RAW_PROCESS_FIELDS) or any(
        type(raw) is not bytes for raw in process_evidence.values()
    ):
        _fail(
            "unknown-or-missing-field",
            f"{label}.process_evidence must contain the closed raw byte field set",
        )

    def raw(name: str) -> bytes:
        return process_evidence[name]

    def derive() -> Target:
        before_process = c02._parse_proc_stat(  # noqa: SLF001 - shared closed raw grammar
            raw("pre_stat"),
            f"{label}.process_evidence.pre_stat",
        )
        after_process = c02._parse_proc_stat(  # noqa: SLF001
            raw("post_stat"),
            f"{label}.process_evidence.post_stat",
        )
        if before_process != after_process:
            _fail(
                "pid-start-tick-mismatch",
                f"{label} pre/post raw /proc stat tuples differ",
            )
        pid, start_ticks = before_process
        c02._parse_proc_status_pid(  # noqa: SLF001 - shared closed raw grammar
            raw("status"),
            pid,
            f"{label}.process_evidence.status",
        )
        selected_index, selected_uuid = c02._parse_gpu_selection(  # noqa: SLF001
            raw("gpu_selection"),
            f"{label}.process_evidence.gpu_selection",
        )
        expected = c02.TargetTuple(
            pid=pid,
            start_ticks=start_ticks,
            gpu_index=selected_index,
            gpu_uuid=selected_uuid,
        )
        before_sockets = c02._parse_socket_snapshot(  # noqa: SLF001
            raw("pre_fd_sockets"),
            expected,
            f"{label}.process_evidence.pre_fd_sockets",
        )
        after_sockets = c02._parse_socket_snapshot(  # noqa: SLF001
            raw("post_fd_sockets"),
            expected,
            f"{label}.process_evidence.post_fd_sockets",
        )
        before_listener = _bound_loopback_listener_from_tcp(
            raw("pre_tcp"),
            before_sockets,
            f"{label}.process_evidence.pre_tcp",
        )
        after_listener = _bound_loopback_listener_from_tcp(
            raw("post_tcp"),
            after_sockets,
            f"{label}.process_evidence.post_tcp",
        )
        if before_listener != after_listener:
            _fail(
                "listener-proof-mismatch",
                f"{label} pre/post raw TCP/FD listener tuples differ",
            )
        c02._parse_compute_apps(  # noqa: SLF001 - shared closed raw grammar
            raw("gpu_compute_apps"),
            pid,
            f"{label}.process_evidence.gpu_compute_apps",
        )
        return Target(
            pid=pid,
            start_ticks=start_ticks,
            listener_port=before_listener[0],
            listener_inode=before_listener[1],
            gpu_index=selected_index,
            gpu_uuid=selected_uuid,
        )

    return _c02_raw(derive)


def derive_phase_target_from_raw_evidence_fd(
    root_fd: int,
    process_evidence: Mapping[str, common.EvidenceDescriptor],
    label: str,
) -> Target:
    """Derive one phase target only from held-FD process/socket/GPU leaves.

    This is the path-descriptor adapter for the path-only v3 binder.  It first
    admits the root and replays every descriptor, then passes the resulting
    immutable byte snapshot to the shared parser.  A caller which already owns
    a more specific held child FD should use :func:`derive_phase_target_from_raw_bytes`
    after consuming that child directly.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback v3 raw phase evidence root",
        )
    )
    if set(process_evidence) != set(RAW_PROCESS_FIELDS) or any(
        not isinstance(descriptor, common.EvidenceDescriptor)
        for descriptor in process_evidence.values()
    ):
        _fail(
            "unknown-or-missing-field",
            f"{label}.process_evidence must contain the closed raw field set",
        )
    raw = {
        name: c02._read_bytes(  # noqa: SLF001 - shared closed raw grammar
            root_fd,
            process_evidence[name],
            f"{label}.process_evidence.{name}",
            maximum_bytes=MAX_RAW_LEAF_BYTES,
        )
        for name in RAW_PROCESS_FIELDS
    }
    return derive_phase_target_from_raw_bytes(raw, label)


def _verify_phase_target_raw_evidence(
    root_fd: int,
    target: Target,
    process_evidence: Mapping[str, common.EvidenceDescriptor],
    label: str,
) -> None:
    observed = derive_phase_target_from_raw_evidence_fd(
        root_fd,
        process_evidence,
        label,
    )
    if observed != target:
        _fail(
            "phase-target-raw-mismatch",
            f"{label} declared target differs from held-FD raw process evidence",
        )


def _phase(
    root_fd: int,
    value: Any,
    label: str,
    *,
    candidate_phase: bool,
    used_paths: set[str],
) -> tuple[Target, dict[str, Any]]:
    fields = {
        "target",
        "process_evidence",
        "health",
        "generation",
        "audit",
    }
    if candidate_phase:
        fields |= {"shutdown_artifact", "shutdown_marker"}
    row = _exact(value, fields, label)
    target = _target(row["target"], f"{label}.target")
    process_evidence = _raw_descriptor_map(
        root_fd,
        row["process_evidence"],
        RAW_PROCESS_FIELDS,
        f"{label}.process_evidence",
        used_paths=used_paths,
    )
    _verify_phase_target_raw_evidence(
        root_fd,
        target,
        process_evidence,
        label,
    )
    health = _raw_descriptor_map(
        root_fd,
        row["health"],
        HTTP_EXCHANGE_FIELDS,
        f"{label}.health",
        used_paths=used_paths,
    )
    generation = _raw_descriptor_map(
        root_fd,
        row["generation"],
        HTTP_EXCHANGE_FIELDS,
        f"{label}.generation",
        used_paths=used_paths,
    )
    audit = _audit(
        root_fd,
        row["audit"],
        f"{label}.audit",
        source_owned=candidate_phase,
        used_paths=used_paths,
    )
    result: dict[str, Any] = {
        "target": target,
        "process_evidence": process_evidence,
        "health": health,
        "generation": generation,
        "audit": audit,
    }
    if candidate_phase:
        shutdown_artifact = _reserve_descriptor(
            root_fd,
            row["shutdown_artifact"],
            f"{label}.shutdown_artifact",
            used_paths=used_paths,
            maximum_bytes=MAX_RAW_LEAF_BYTES,
        )
        shutdown_marker = _reserve_descriptor(
            root_fd,
            row["shutdown_marker"],
            f"{label}.shutdown_marker",
            used_paths=used_paths,
            maximum_bytes=MAX_RAW_LEAF_BYTES,
        )
        replayed = _c02(
            lambda: c02.verify_c02_shutdown_v2_fd(
                root_fd,
                shutdown_artifact.path,
                shutdown_marker.path,
                c02.TargetTuple(
                    pid=target.pid,
                    start_ticks=target.start_ticks,
                    gpu_index=target.gpu_index,
                    gpu_uuid=target.gpu_uuid,
                ),
            )
        )
        if replayed.artifact != shutdown_artifact or replayed.marker != shutdown_marker:
            _fail(
                "candidate-shutdown-descriptor-changed",
                f"{label} shutdown descriptor changed during replay",
            )
        result["shutdown_artifact"] = shutdown_artifact
        result["shutdown_marker"] = shutdown_marker
    return target, result


def _manifest(
    root_fd: int,
    manifest_path: str,
) -> tuple[common.EvidenceDescriptor, bytes]:
    relative = _relative_path(manifest_path, "rollback v3 manifest path")
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            relative,
            "rollback v3 raw manifest",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
    )
    return (
        _common(
            lambda: common.descriptor_for_bytes(
                relative,
                raw,
                "rollback v3 raw manifest",
            )
        ),
        raw,
    )


def _baseline_report_descriptors(
    value: Any,
    *,
    label: str,
    result: dict[str, common.EvidenceDescriptor],
) -> None:
    """Collect unique leaf descriptors returned by the baseline replay.

    The baseline report intentionally repeats some descriptor values in its
    equality section.  Equal repeated references describe the same baseline
    fact, while a repeated path with a different descriptor would be an
    ambiguous binding and is rejected here before phase evidence is accepted.
    """

    if isinstance(value, Mapping):
        if set(value) == {"path", "sha256", "byte_length"}:
            descriptor = _common(lambda: common.parse_descriptor(value, label))
            existing = result.get(descriptor.path)
            if existing is not None and existing != descriptor:
                _fail(
                    "baseline-report-descriptor-drift",
                    f"{label} repeats path {descriptor.path!r} with a different descriptor",
                )
            result[descriptor.path] = descriptor
            return
        for name, child in value.items():
            _baseline_report_descriptors(
                child,
                label=f"{label}.{name}",
                result=result,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _baseline_report_descriptors(
                child,
                label=f"{label}[{index}]",
                result=result,
            )


def _baseline_manifest(
    root_fd: int,
    value: Any,
    *,
    used_paths: set[str],
) -> tuple[common.EvidenceDescriptor, Mapping[str, Any]]:
    row = _exact(value, {"manifest"}, "rollback v3 reconstructed baseline")
    descriptor = _reserve_descriptor(
        root_fd,
        row["manifest"],
        "rollback v3 reconstructed baseline.manifest",
        used_paths=used_paths,
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    _raw, document = _common(
        lambda: common.read_descriptor_json(
            root_fd,
            descriptor,
            "rollback v3 reconstructed baseline manifest",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
    )
    if document.get("schema_version") == baseline.LEGACY_MANIFEST_VERSION:
        _fail(
            "rollback-binary-provenance-required",
            "rollback v3 requires a reconstructed baseline v2 with A/B server-binary binding",
        )
    report = _baseline(lambda: baseline.evaluate(root_fd, document))
    if (
        not isinstance(report, Mapping)
        or report.get("schema_version") != baseline.CHECK_REPORT_VERSION
        or report.get("status") != "passed"
        or report.get("passed") is not True
    ):
        _fail(
            "invalid-reconstructed-baseline-report",
            "reconstructed baseline replay did not return its closed passed report",
        )
    git_identity = report.get("git_identity")
    if not isinstance(git_identity, Mapping):
        _fail(
            "invalid-reconstructed-baseline-report",
            "reconstructed baseline replay did not return a Git identity",
        )
    if (
        report.get("baseline_id") != RECONSTRUCTED_ROLLBACK_BASELINE_ID
        or git_identity.get("tag_name") != RECONSTRUCTED_ROLLBACK_TAG
        or git_identity.get("target_commit_sha1") != RECONSTRUCTED_ROLLBACK_TARGET
    ):
        _fail(
            "unsupported-reconstructed-baseline",
            "rollback v3 is pinned to the reviewed reconstructed RC2 tag target",
        )
    if git_identity.get("tag_object_sha1") != RECONSTRUCTED_ROLLBACK_TAG_OBJECT:
        _fail(
            "reviewed-reconstructed-tag-object-mismatch",
            "rollback v3 is pinned to the reviewed reconstructed RC2 annotated tag object",
        )
    baseline_paths: dict[str, common.EvidenceDescriptor] = {}
    _baseline_report_descriptors(
        report,
        label="rollback v3 reconstructed baseline report",
        result=baseline_paths,
    )
    for path in sorted(baseline_paths):
        if path in used_paths:
            _fail(
                "duplicate-evidence-path",
                f"reconstructed baseline reuses evidence path {path!r}",
            )
        used_paths.add(path)
    return descriptor, report


def _bindings(value: Any) -> dict[str, str]:
    row = _exact(
        value,
        {
            "freeze_sha256",
            "base_release_candidate_report_sha256",
            "configuration_profile",
            "configuration_sha256",
        },
        "rollback v3 raw manifest.bindings",
    )
    if row["configuration_profile"] != STABLE_DEFAULT_PROFILE:
        _fail(
            "invalid-configuration-profile",
            "rollback v3 raw manifest must use stable-default",
        )
    return {
        "freeze_sha256": _sha256(
            row["freeze_sha256"],
            "rollback v3 raw manifest.bindings.freeze_sha256",
        ),
        "base_release_candidate_report_sha256": _sha256(
            row["base_release_candidate_report_sha256"],
            "rollback v3 raw manifest.bindings.base_release_candidate_report_sha256",
        ),
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": _sha256(
            row["configuration_sha256"],
            "rollback v3 raw manifest.bindings.configuration_sha256",
        ),
    }


def _as_json(value: Any) -> Any:
    if isinstance(value, common.EvidenceDescriptor):
        return value.as_json()
    if isinstance(value, Target):
        return value.as_json()
    if isinstance(value, dict):
        return {key: _as_json(item) for key, item in value.items()}
    return value


def verify_rollback_provenance_v3_bytes_fd(
    root_fd: int,
    manifest_descriptor: common.EvidenceDescriptor,
    raw_document: bytes,
) -> dict[str, Any]:
    """Replay canonical v3 manifest bytes through the exact caller-held FD.

    The path-only binder uses this bounded-bytes core before publication with
    a descriptor derived from the exact canonical output bytes. It
    intentionally does not read ``manifest_descriptor.path``; the file
    wrapper below owns that read, and the binder self-verifies the on-disk
    leaf after create-only publication.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback v3 provenance evidence root",
        )
    )
    if not isinstance(manifest_descriptor, common.EvidenceDescriptor):
        _fail("invalid-descriptor", "rollback v3 manifest descriptor has an invalid type")
    manifest_descriptor = _common(
        lambda: common.parse_descriptor(
            manifest_descriptor.as_json(),
            "rollback v3 manifest descriptor",
        )
    )
    manifest_path = _relative_path(
        manifest_descriptor.path,
        "rollback v3 manifest descriptor.path",
    )
    document = _common(
        lambda: common.parse_canonical_json(
            raw_document,
            "rollback v3 parsed manifest",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
    )
    assert isinstance(document, Mapping)
    document_descriptor = _common(
        lambda: common.descriptor_for_bytes(
            manifest_path,
            raw_document,
            "rollback v3 parsed manifest",
        )
    )
    if document_descriptor != manifest_descriptor:
        _fail(
            "manifest-document-descriptor-mismatch",
            "rollback v3 manifest descriptor does not bind the supplied canonical document",
        )
    row = _exact(
        document,
        {
            "schema_version",
            "capture_status",
            "qualification_status",
            "candidate_id",
            "bindings",
            "reconstructed_baseline",
            "candidate",
            "rollback",
            "candidate_artifacts",
            "rollback_artifacts",
            "atomic_switch",
        },
        "rollback v3 raw manifest",
    )
    if row["schema_version"] != ROLLBACK_V3_MANIFEST_VERSION:
        _fail(
            "unsupported-rollback-v3-version",
            f"rollback v3 raw manifest must use {ROLLBACK_V3_MANIFEST_VERSION}",
        )
    if row["capture_status"] != "captured" or row["qualification_status"] != "not-run":
        _fail(
            "invalid-capture-status",
            "rollback v3 raw manifest must be captured/not-run",
        )
    candidate_id = _candidate_id(row["candidate_id"], "rollback v3 raw manifest.candidate_id")
    bindings = _bindings(row["bindings"])
    used_paths = {manifest_descriptor.path}
    baseline_descriptor, baseline_report = _baseline_manifest(
        root_fd,
        row["reconstructed_baseline"],
        used_paths=used_paths,
    )
    candidate_target, candidate = _phase(
        root_fd,
        row["candidate"],
        "rollback v3 candidate",
        candidate_phase=True,
        used_paths=used_paths,
    )
    rollback_target, rollback = _phase(
        root_fd,
        row["rollback"],
        "rollback v3 reconstructed baseline phase",
        candidate_phase=False,
        used_paths=used_paths,
    )
    if (candidate_target.pid, candidate_target.start_ticks) == (
        rollback_target.pid,
        rollback_target.start_ticks,
    ):
        _fail(
            "reused-candidate-process",
            "candidate and reconstructed baseline must use distinct PID/start-tick identities",
        )
    candidate_artifacts = _artifact_map(
        root_fd,
        row["candidate_artifacts"],
        "rollback v3 candidate_artifacts",
        used_paths=used_paths,
    )
    rollback_artifacts = _artifact_map(
        root_fd,
        row["rollback_artifacts"],
        "rollback v3 rollback_artifacts",
        used_paths=used_paths,
    )
    _bind_reconstructed_baseline_artifacts(
        root_fd,
        rollback_artifacts,
        baseline_report,
    )
    atomic_switch = _raw_descriptor_map(
        root_fd,
        row["atomic_switch"],
        ATOMIC_SWITCH_FIELDS,
        "rollback v3 atomic_switch",
        used_paths=used_paths,
    )
    return {
        "schema_version": ROLLBACK_V3_REPORT_VERSION,
        "status": "bound",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "bindings": bindings,
        "raw_manifest": manifest_descriptor.as_json(),
        "reconstructed_baseline": {
            "manifest": baseline_descriptor.as_json(),
            "baseline_id": baseline_report["baseline_id"],
            "baseline_kind": baseline_report["baseline_kind"],
            "provenance_class": baseline_report["provenance_class"],
            "historical_distribution": baseline_report["historical_distribution"],
            "historical_stable_artifact_status": baseline_report[
                "historical_stable_artifact_status"
            ],
            "was_previously_shipped": baseline_report["was_previously_shipped"],
        },
        "targets": [
            {"phase": "candidate", "target": candidate_target.as_json()},
            {"phase": "rollback", "target": rollback_target.as_json()},
        ],
        "raw_evidence": {
            "candidate": _as_json(candidate),
            "rollback": _as_json(rollback),
            "candidate_artifacts": _as_json(candidate_artifacts),
            "rollback_artifacts": _as_json(rollback_artifacts),
            "atomic_switch": _as_json(atomic_switch),
        },
        "checks": [
            {"name": "v3-version-only", "bound": True},
            {"name": "canonical-descriptor-binding", "bound": True},
            {"name": "reconstructed-baseline-a-b-replay-binding", "bound": True},
            {"name": "active-baseline-bundle-and-image-binding", "bound": True},
            {"name": "active-baseline-binary-binding", "bound": True},
            {"name": "distinct-candidate-and-baseline-process-tuples", "bound": True},
            {"name": "candidate-shutdown-v2-marker-binding", "bound": True},
            {"name": "declared-candidate-audit-availability-and-index-inventory", "bound": True},
            {"name": "baseline-audit-not-supported-boundary", "bound": True},
            {"name": "raw-health-generation-proc-socket-gpu-inventory", "bound": True},
            {"name": "raw-atomic-switch-material", "bound": True},
            {"name": "stable-default-arm-binding", "bound": True},
        ],
        "reason_codes": [],
    }


def verify_rollback_provenance_v3_fd(
    root_fd: int,
    manifest_path: str,
) -> dict[str, Any]:
    """Replay one canonical v3 rollback manifest through one held root FD."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback v3 provenance evidence root",
        )
    )
    manifest_descriptor, raw_document = _manifest(root_fd, manifest_path)
    return verify_rollback_provenance_v3_bytes_fd(
        root_fd,
        manifest_descriptor,
        raw_document,
    )


def verify_rollback_provenance_v3(
    evidence_root: Path,
    manifest_path: str,
) -> dict[str, Any]:
    """Open one private evidence root and replay a v3 raw rollback manifest."""

    _assert_external_to_source_checkout(evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(
            evidence_root,
            "rollback v3 evidence root",
        )
    )
    try:
        return verify_rollback_provenance_v3_fd(root_fd, manifest_path)
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_rollback_provenance_v3(
            args.evidence_root,
            args.manifest,
        )
    except RollbackV3ProvenanceError as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
