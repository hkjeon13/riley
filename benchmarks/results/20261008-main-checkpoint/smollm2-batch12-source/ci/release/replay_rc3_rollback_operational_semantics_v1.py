#!/usr/bin/env python3
"""Replay narrow RC3 rollback operational facts through caller-held FDs.

This private helper is intentionally neither a path-based checker nor a
producer.  An authenticated caller should nest it while retaining the same
private evidence-root and isolated-switch ``LOCK_EX`` normal-return stack that
produced the raw closure.  It rereads the original candidate/source, rollback,
and isolated transaction leaves and returns one canonical in-memory diagnostic.

``passed`` means only that these raw operational facts agreed during this
held-FD replay.  It does not establish a host deployment rollback, a frozen
candidate, Gate E, promotion, semantic qualification, or any durable success
inference from a later path replay.  Supplied FDs bind the current raw leaves
but cannot by themselves prove prior invocation lineage or finalizer success.
This module writes nothing and deliberately has no CLI or path-opening wrapper.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, NoReturn, TypeVar

import capture_rc3_rollback_atomic_transaction_v1 as transaction
import check_c02_provenance_v2 as c02
import check_rc3_rollback_provenance_v4 as v4
import finalize_rc3_rollback_candidate_source_v4 as fixed_finalizer
import provenance_v2_common as common
import write_rc3_rollback_candidate_source_bind_request_v1 as writer


SEMANTICS_VERSION = "riley.rc3-rollback-operational-semantics.v1"
SEMANTICS_AUTHORITY = "raw-operational-semantics-only"

_CHECK_NAMES = (
    "held-fd-raw-topology-replay",
    "candidate-rollback-process-identity-distinct",
    "candidate-rollback-gpu-identity-equal",
    "candidate-rollback-listener-port-distinct",
    "candidate-source-response-audit-identity",
    "rollback-generation-response-id-distinct",
    "candidate-shutdown-drained",
    "isolated-artifact-inode-exchange",
    "v4-raw-closure-cross-binding",
)


class RollbackOperationalSemanticsError(ValueError):
    """The raw rollback closure cannot establish its narrow operational facts."""


def _fail(code: str, message: str) -> NoReturn:
    error = RollbackOperationalSemanticsError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _transaction(call: Callable[[], T]) -> T:
    try:
        return call()
    except transaction.RollbackAtomicTransactionError as error:
        _fail(getattr(error, "reason_code", "invalid-atomic-transaction"), str(error))


def _writer(call: Callable[[], T]) -> T:
    try:
        return call()
    except writer.RollbackCandidateSourceBindRequestError as error:
        _fail(getattr(error, "reason_code", "invalid-raw-topology"), str(error))


def _v4(call: Callable[[], T]) -> T:
    try:
        return call()
    except v4.RollbackV4ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-v4-raw-closure"), str(error))


def _c02(call: Callable[[], T]) -> T:
    try:
        return call()
    except c02.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-c02-raw-evidence"), str(error))


def _exact_mapping(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(
            "invalid-operational-schema",
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
        )
    return value


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail("invalid-operational-metric", f"{label} must be a positive integer")
    return value


def _nonnegative(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail("invalid-operational-metric", f"{label} must be a non-negative integer")
    return value


def _zero(value: Any, label: str) -> None:
    if _nonnegative(value, label) != 0:
        _fail("candidate-shutdown-not-drained", f"{label} must be zero")


def _source_completion_id(
    root_fd: int,
    descriptor: common.EvidenceDescriptor | Mapping[str, Any],
    label: str,
) -> str:
    raw = _common(
        lambda: common.read_descriptor_bytes(
            root_fd,
            descriptor,
            label,
            maximum_bytes=c02.MAX_RAW_BYTES,
        )
    )
    # Source response formatting is strict JSON, not necessarily canonical JSON.
    return _c02(lambda: c02._source_request_id(raw, label))  # noqa: SLF001


def _verify_candidate_shutdown_drained(root_fd: int, inputs: Any) -> None:
    joined = inputs.candidate_source
    _raw, document = _common(
        lambda: common.read_descriptor_json(
            root_fd,
            joined.shutdown.artifact,
            "candidate source shutdown artifact",
            maximum_bytes=c02.MAX_RAW_BYTES,
        )
    )
    row = _exact_mapping(
        document,
        frozenset(
            {
                "schema_version",
                "capture_status",
                "qualification_status",
                "server_pid",
                "server_start_ticks",
                "worker_ready",
                "final_metrics",
            }
        ),
        "candidate source shutdown artifact",
    )
    if (
        row["schema_version"] != c02.SHUTDOWN_VERSION
        or row["capture_status"] != "captured"
        or row["qualification_status"] != "not-run"
    ):
        _fail(
            "invalid-shutdown-status",
            "candidate source shutdown artifact is not captured/not-run raw evidence",
        )
    target = joined.candidate_phase.target
    if (
        _positive(row["server_pid"], "candidate source shutdown artifact.server_pid")
        != target.server_pid
        or _positive(
            row["server_start_ticks"],
            "candidate source shutdown artifact.server_start_ticks",
        )
        != target.server_start_ticks
    ):
        _fail(
            "candidate-shutdown-target-mismatch",
            "candidate source shutdown does not bind the candidate process identity",
        )
    if row["worker_ready"] is not False:
        _fail("candidate-shutdown-not-drained", "candidate worker_ready must be false")

    metrics = _exact_mapping(
        row["final_metrics"],
        frozenset({"schema_version", "request_states", "kv_blocks", "allocation", "quiescence"}),
        "candidate source shutdown artifact.final_metrics",
    )
    if metrics["schema_version"] != c02.METRICS_VERSION:
        _fail(
            "unsupported-metrics-version",
            "candidate source shutdown metrics use an unsupported version",
        )
    states = _exact_mapping(
        metrics["request_states"],
        frozenset(
            {
                "active",
                "pending_requests",
                "completed",
                "failed",
                "cancelled",
                "capacity_rejections",
            }
        ),
        "candidate source shutdown artifact.final_metrics.request_states",
    )
    kv_blocks = _exact_mapping(
        metrics["kv_blocks"],
        frozenset({"free", "reserved", "active"}),
        "candidate source shutdown artifact.final_metrics.kv_blocks",
    )
    allocation = _exact_mapping(
        metrics["allocation"],
        frozenset(
            {
                "device_live_count",
                "device_live_bytes",
                "pinned_live_count",
                "pinned_live_bytes",
            }
        ),
        "candidate source shutdown artifact.final_metrics.allocation",
    )
    quiescence = _exact_mapping(
        metrics["quiescence"],
        frozenset(
            {
                "completion_outbox",
                "outstanding_iterations",
                "riley_owned_live_allocations",
                "worker_accepting",
                "scheduler_accepting",
            }
        ),
        "candidate source shutdown artifact.final_metrics.quiescence",
    )
    for group, group_label in (
        (states, "request_states"),
        (kv_blocks, "kv_blocks"),
        (allocation, "allocation"),
    ):
        for field, value in group.items():
            _nonnegative(value, f"candidate shutdown {group_label}.{field}")
    for field in ("completion_outbox", "outstanding_iterations", "riley_owned_live_allocations"):
        _nonnegative(quiescence[field], f"candidate shutdown quiescence.{field}")
    for field in ("worker_accepting", "scheduler_accepting"):
        if type(quiescence[field]) is not bool:
            _fail(
                "invalid-operational-metric",
                f"candidate shutdown quiescence.{field} must be boolean",
            )

    _zero(states["active"], "candidate shutdown request_states.active")
    _zero(states["pending_requests"], "candidate shutdown request_states.pending_requests")
    _zero(kv_blocks["reserved"], "candidate shutdown kv_blocks.reserved")
    _zero(kv_blocks["active"], "candidate shutdown kv_blocks.active")
    for field in (
        "device_live_count",
        "device_live_bytes",
        "pinned_live_count",
        "pinned_live_bytes",
    ):
        _zero(allocation[field], f"candidate shutdown allocation.{field}")
    for field in ("completion_outbox", "outstanding_iterations", "riley_owned_live_allocations"):
        _zero(quiescence[field], f"candidate shutdown quiescence.{field}")
    for field in ("worker_accepting", "scheduler_accepting"):
        if quiescence[field] is not False:
            _fail(
                "candidate-shutdown-not-drained",
                f"candidate shutdown quiescence.{field} must be false",
            )


def _verify_targets(inputs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = inputs.candidate_source.candidate_phase.target
    rollback = inputs.rollback_phase.target
    for label, target in (("candidate", candidate), ("rollback", rollback)):
        if not 1024 <= target.listener_port <= 65535:
            _fail(
                "invalid-listener-port",
                f"{label} listener port must be from 1024 through 65535",
            )
    if (
        candidate.server_pid,
        candidate.server_start_ticks,
    ) == (
        rollback.server_pid,
        rollback.server_start_ticks,
    ):
        _fail(
            "reused-candidate-process",
            "candidate and rollback phases reuse one PID/start-tick identity",
        )
    if (
        candidate.gpu_index != rollback.gpu_index
        or candidate.gpu_uuid != rollback.gpu_uuid
    ):
        _fail(
            "candidate-rollback-gpu-mismatch",
            "candidate and rollback phases must bind the same GPU identity",
        )
    if candidate.listener_port == rollback.listener_port:
        _fail(
            "candidate-rollback-listener-port-reused",
            "candidate and rollback phases must use distinct listener ports",
        )
    return candidate.as_json(), rollback.as_json()


def _verify_generation_ids(inputs: Any, root_fd: int) -> tuple[str, str]:
    joined = inputs.candidate_source
    source_id = _source_completion_id(
        root_fd,
        joined.generation.response_body,
        "candidate source completion response",
    )
    if source_id != joined.source_scenario.request_id:
        _fail(
            "candidate-source-response-audit-mismatch",
            "candidate source completion ID differs from its replayed audit identity",
        )
    generation = inputs.rollback_phase.generation
    if not isinstance(generation, Mapping) or set(generation) != {
        "request",
        "response_head",
        "response_body",
    }:
        _fail(
            "rollback-generation-required",
            "rollback phase must retain exactly one generation exchange",
        )
    rollback_id = _source_completion_id(
        root_fd,
        generation["response_body"],
        "rollback generation response",
    )
    if rollback_id == source_id:
        _fail(
            "rollback-generation-id-reused",
            "rollback generation response must not reuse the candidate source ID",
        )
    return source_id, rollback_id


def _verify_isolated_artifact_exchange(inputs: Any) -> None:
    exchange = inputs.atomic_transaction.atomic_switch_replay
    pre_active = exchange.pre_active
    pre_rollback = exchange.pre_rollback_staged
    post_active = exchange.post_active
    post_candidate = exchange.post_candidate_staged
    if pre_active.device != pre_rollback.device or pre_active.inode == pre_rollback.inode:
        _fail(
            "isolated-artifact-exchange-mismatch",
            "isolated switch did not start with two distinct same-device inodes",
        )
    if (
        post_active.device,
        post_active.inode,
        post_active.sha256,
    ) != (
        pre_rollback.device,
        pre_rollback.inode,
        pre_rollback.sha256,
    ):
        _fail(
            "isolated-artifact-exchange-mismatch",
            "isolated switch active entry does not retain the rollback staged inode",
        )
    if (
        post_candidate.device,
        post_candidate.inode,
        post_candidate.sha256,
    ) != (
        pre_active.device,
        pre_active.inode,
        pre_active.sha256,
    ):
        _fail(
            "isolated-artifact-exchange-mismatch",
            "isolated switch candidate-staged entry does not retain the candidate inode",
        )


def _expected_bindings(inputs: Any) -> dict[str, str]:
    static = inputs.candidate_source.static_effective
    bindings = static.static_bindings
    if static.configuration_profile != "stable-default":
        _fail(
            "invalid-configuration-profile",
            "rollback operational semantics require stable-default",
        )
    return {
        "freeze_sha256": bindings.freeze.sha256,
        "base_release_candidate_report_sha256": bindings.base_release_candidate_report.sha256,
        "configuration_profile": "stable-default",
        "configuration_sha256": bindings.configuration.sha256,
    }


def _verify_v4_raw_closure(inputs: Any, root_fd: int, switch_fd: int) -> dict[str, Any]:
    # This replays the raw v4 manifest only.  It intentionally does not read a
    # v4 completion pair or a finalizer receipt as semantic input.
    report = _v4(
        lambda: v4.verify_rollback_provenance_v4_on_held_switch_fd(
            root_fd,
            switch_fd,
            fixed_finalizer.ROLLBACK_V4_MANIFEST_NAME,
        )
    )
    if (
        report.get("schema_version") != v4.ROLLBACK_V4_REPORT_VERSION
        or report.get("status") != "bound"
        or report.get("qualification_status") != "not-run"
    ):
        _fail("invalid-v4-raw-closure", "v4 raw replay did not return bound/not-run")
    if report.get("candidate_id") != inputs.candidate_source.static_effective.candidate_id:
        _fail("v4-candidate-id-mismatch", "v4 raw closure does not bind the candidate ID")
    if report.get("bindings") != _expected_bindings(inputs):
        _fail("v4-bindings-mismatch", "v4 raw closure does not bind static inputs")

    raw_manifest = _common(
        lambda: common.parse_descriptor(report.get("raw_manifest"), "v4 raw closure.raw_manifest")
    )
    rollback_v3_manifest = _common(
        lambda: common.parse_descriptor(
            report.get("rollback_v3_manifest"),
            "v4 raw closure.rollback_v3_manifest",
        )
    )
    atomic_session = _common(
        lambda: common.parse_descriptor(
            report.get("atomic_transaction_session"),
            "v4 raw closure.atomic_transaction_session",
        )
    )
    if raw_manifest.path != fixed_finalizer.ROLLBACK_V4_MANIFEST_NAME:
        _fail("v4-manifest-name-mismatch", "v4 raw closure has an unexpected manifest name")
    if rollback_v3_manifest.path != fixed_finalizer.ROLLBACK_V3_MANIFEST_NAME:
        _fail("v3-manifest-name-mismatch", "v4 raw closure has an unexpected v3 manifest name")
    if atomic_session != inputs.atomic_transaction.session_descriptor:
        _fail(
            "v4-transaction-session-mismatch",
            "v4 raw closure does not bind the held atomic transaction session",
        )
    return {
        "raw_manifest": raw_manifest.as_json(),
        "rollback_v3_manifest": rollback_v3_manifest.as_json(),
        "atomic_transaction_session": atomic_session.as_json(),
    }


def _replay_operational_semantics_once(root_fd: int, switch_fd: int) -> dict[str, Any]:
    inputs = _writer(lambda: writer._replay_inputs(root_fd, switch_fd))  # noqa: SLF001
    candidate_target, rollback_target = _verify_targets(inputs)
    source_id, rollback_id = _verify_generation_ids(inputs, root_fd)
    _verify_candidate_shutdown_drained(root_fd, inputs)
    _verify_isolated_artifact_exchange(inputs)
    closure = _verify_v4_raw_closure(inputs, root_fd, switch_fd)
    return {
        "schema_version": SEMANTICS_VERSION,
        "status": "passed",
        "qualification_status": "not-run",
        "authority": SEMANTICS_AUTHORITY,
        "candidate_id": inputs.candidate_source.static_effective.candidate_id,
        "bindings": _expected_bindings(inputs),
        **closure,
        "derived_facts": {
            "candidate_target": candidate_target,
            "rollback_target": rollback_target,
            "candidate_source_response_id": source_id,
            "rollback_generation_response_id": rollback_id,
            "candidate_shutdown_drained": True,
            "isolated_artifact_exchange": True,
        },
        "checks": [{"name": name, "passed": True} for name in _CHECK_NAMES],
        "reason_codes": [],
    }


def _replay_rc3_rollback_operational_semantics_on_held_root_switch_fds(
    root_fd: int,
    switch_fd: int,
) -> dict[str, Any]:
    """Return one canonical raw-only semantic diagnostic on caller-held FDs.

    An authenticated caller should retain both exclusive locks for its entire
    normal-return stack.  This function does not acquire, release, reopen, or
    close either supplied descriptor; the descriptors alone do not prove
    invocation lineage, so this raw-only result can never replace a finalizer
    normal-return capability or semantic receipt.  It writes no leaf.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback operational semantics evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    initial = _replay_operational_semantics_once(root_fd, switch_fd)
    terminal = _replay_operational_semantics_once(root_fd, switch_fd)
    if terminal != initial:
        _fail(
            "operational-semantics-replay-drift",
            "rollback raw operational facts changed during held-FD replay",
        )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback operational semantics evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    return _common(
        lambda: common.parse_canonical_json(
            common.canonical_json_bytes(initial),
            "rollback operational semantics diagnostic",
        )
    )
