#!/usr/bin/env python3
"""Read-only C02 soak v2 semantic replay over completed raw v4/v5 evidence.

This checker deliberately does not upgrade ``check_soak_v2_receipt.py``.  It
opens one external private evidence root, holds a shared lock, and asks the
raw verifier for FD-replayed source audit and observation-metric leaves.  It
then reconstructs only identity, per-observation-session interval order,
monotonic cumulative metrics, request/audit binding, and typed sampling
selection semantics from those original leaves.

``passed`` means this narrow held-FD semantic replay agreed.  It is not a
durable semantic receipt and does not establish producer success, lifecycle
success, actual GPU capture, candidate freeze, Gate E, qualification,
deployment, rollback, campaign thresholds, or a cross-session timeline.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys

sys.dont_write_bytecode = True

from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import check_c02_provenance_v2 as raw
import provenance_v2_common as common


SEMANTIC_REPORT_VERSION = "riley.soak-v2-semantic-replay.v2"
SEMANTIC_SCOPE = "c02-soak-v2-semantic-replay-only"
SEMANTIC_AUTHORITY = "soak-v2-semantic-replay-only"
SEMANTIC_POLICY_VERSION = "riley.c02-soak-v2-semantic-policy.v1"
_DIRECT_RAW_MANIFEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,240}\.json$")
_HISTORICAL_MANIFEST_REASONS = {
    "riley.soak-v2-receipt.v1": "historical-soak-v1-rejected",
    "riley.soak-v2-raw-provenance.v1": "historical-soak-v1-rejected",
    "riley.soak-v2-raw-provenance.v2": "historical-soak-v2-rejected",
    "riley.soak-v2-raw-provenance.v3": "historical-soak-v3-rejected",
}
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_PROMPT_TOKEN_IDS = 131_072
_MAX_COMMITTED_OUTPUT_TOKENS = 65_536
_MAX_SAMPLING_SELECTIONS = 65_536
_MAX_EMITTED_TEXT_DELTA_CHARS = 1_048_576
_BACKENDS = frozenset(("cpu-normative", "gpu-greedy"))
_INELIGIBILITY_REASONS = frozenset(
    (
        "gpu-greedy-not-configured",
        "addressable-vocabulary-mismatch",
        "nonzero-temperature",
        "repetition-penalty",
        "finish-token-mask",
    )
)
_CHECK_NAMES = (
    "held-fd-completed-raw-manifest-replayed",
    "dispatch-header-to-manifest-descriptor-binding",
    "identity-and-scenario-order-reconstructed-from-leaves",
    "request-response-audit-binding-reconstructed",
    "per-observation-session-interval-order-reconstructed",
    "per-observation-session-cumulative-metrics-monotonic",
    "typed-sampling-selections-reconstructed",
    "semantic-replay-does-not-establish-lifecycle-or-qualification",
)
_NOT_ESTABLISHED = {
    "producer_success": "not-established",
    "lifecycle_success": "not-established",
    "same_stack_normal_return": "not-established",
    "actual_gpu_capture": "not-established",
    "candidate_freeze": "not-established",
    "gate_e": "not-established",
    "semantic_receipt": "not-established",
    "qualification": "not-established",
    "deployment": "not-established",
    "rollback": "not-established",
    "campaign_thresholds": "not-established",
    "cross_session_interval_order": "not-established",
}


class SoakV2SemanticReplayError(ValueError):
    """Completed raw soak evidence cannot satisfy the narrow v2 semantic replay."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = SoakV2SemanticReplayError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _raw(call: Callable[[], T]) -> T:
    try:
        return call()
    except raw.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-raw-provenance"), str(error))


def _evidence_root(value: Path) -> Path:
    try:
        raw_path = os.fspath(value)
    except TypeError as error:
        _fail("invalid-evidence-root", f"--evidence-root is not a path: {error}")
    if (
        type(raw_path) is not str
        or not raw_path
        or "\x00" in raw_path
        or not os.path.isabs(raw_path)
        or raw_path.startswith("//")
        or raw_path != os.path.normpath(raw_path)
    ):
        _fail("invalid-evidence-root", "--evidence-root must be a normalized absolute path")
    root = Path(raw_path)
    source_root = Path(__file__).resolve().parents[2]
    try:
        root.relative_to(source_root)
    except ValueError:
        return root
    _fail("evidence-root-inside-source-checkout", "--evidence-root must be outside the source checkout")


def _raw_manifest_name(value: str) -> str:
    if type(value) is not str or _DIRECT_RAW_MANIFEST_RE.fullmatch(value) is None:
        _fail(
            "raw-manifest-must-be-direct-root-leaf",
            "--raw-manifest must be one direct nonhidden root JSON leaf",
        )
    return value


def _shared_lock(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("evidence-root-lock-unavailable", f"cannot acquire shared evidence-root lock: {error}")


def _unlock_quietly(root_fd: int | None) -> None:
    if root_fd is not None:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(root_fd: int | None) -> None:
    if root_fd is not None:
        try:
            os.close(root_fd)
        except OSError:
            pass


def _manifest_header(
    root_fd: int,
    manifest_name: str,
) -> tuple[common.EvidenceDescriptor, str]:
    manifest_raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            manifest_name,
            "soak semantic raw manifest dispatch header",
            maximum_bytes=raw.MAX_RAW_BYTES,
        )
    )
    descriptor = _common(
        lambda: common.descriptor_for_bytes(
            manifest_name,
            manifest_raw,
            "soak semantic raw manifest dispatch header",
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            manifest_raw,
            "soak semantic raw manifest dispatch header",
            maximum_bytes=raw.MAX_RAW_BYTES,
        )
    )
    if type(document) is not dict:
        _fail("unsupported-soak-raw-manifest-version", "raw manifest header must be a JSON object")
    schema_version = document.get("schema_version")
    if type(schema_version) is not str:
        _fail("unsupported-soak-raw-manifest-version", "raw manifest must contain text schema_version")
    return descriptor, schema_version


def _select_semantic_input_replay(schema_version: str) -> Callable[[int, str], raw.ReplayedSoakSemanticInputs]:
    if schema_version == raw.SOAK_V4_MANIFEST_VERSION:
        return raw.replay_completed_soak_v4_semantic_inputs_fd
    if schema_version == raw.SOAK_V5_MANIFEST_VERSION:
        return raw.replay_completed_soak_v5_semantic_inputs_fd
    historical_reason = _HISTORICAL_MANIFEST_REASONS.get(schema_version)
    if historical_reason is not None:
        _fail(
            historical_reason,
            f"historical soak raw manifest version {schema_version!r} is not a v2 semantic input",
        )
    _fail(
        "unsupported-soak-raw-manifest-version",
        f"unsupported soak raw manifest version {schema_version!r}",
    )


def _exact_mapping(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(
            "invalid-semantic-leaf-schema",
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
        )
    return value


def _u32(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_U32:
        _fail("invalid-semantic-integer", f"{label} must be a u32 integer")
    return value


def _positive_u64(value: Any, label: str) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_U64:
        _fail("invalid-semantic-integer", f"{label} must be a positive u64 integer")
    return value


def _nonnegative_u64(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_U64:
        _fail("invalid-semantic-integer", f"{label} must be a non-negative u64 integer")
    return value


def _descriptor(value: Any, label: str) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if descriptor.byte_length < 1:
        _fail("empty-evidence-leaf", f"{label} must bind nonempty raw evidence")
    return descriptor


def _target_fact(target: raw.ObservedTarget) -> dict[str, Any]:
    return {
        "server_pid": target.target.pid,
        "server_start_ticks": target.target.start_ticks,
        "listener_port": target.listener_port,
        "listener_inode": target.listener_inode,
        "gpu_index": target.target.gpu_index,
        "gpu_uuid": target.target.gpu_uuid,
    }


def _reconstruct_generation_audit(
    audit_bytes: bytes,
    *,
    candidate_id: str,
    bindings: raw.ReplayedSoakSemanticBindings,
    target: raw.ObservedTarget,
    request_id: str,
    label: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Reconstruct audit accounting and typed selections from source bytes."""

    document = _common(
        lambda: common.parse_canonical_json(
            audit_bytes,
            label,
            maximum_bytes=raw.MAX_RAW_BYTES,
        )
    )
    row = _exact_mapping(
        document,
        frozenset(
            {
                "schema_version", "candidate_id", "runtime_identity", "process_identity",
                "server_request_id", "delivery_mode", "prompt_token_ids",
                "committed_output_tokens", "sampling_selections", "finish_reason", "usage",
            }
        ),
        label,
    )
    if row["schema_version"] != raw.SCENARIO_CAPTURE_AUDIT_VERSION:
        _fail("historical-source-audit-version-rejected", f"{label} has an unsupported schema version")
    if row["candidate_id"] != candidate_id:
        _fail("semantic-audit-candidate-mismatch", f"{label} candidate differs from raw manifest")
    identity = _exact_mapping(
        row["runtime_identity"],
        frozenset(("configuration_profile", "configuration_sha256")),
        f"{label}.runtime_identity",
    )
    if (
        identity["configuration_profile"] != bindings.configuration_profile
        or identity["configuration_sha256"] != bindings.configuration_sha256
    ):
        _fail("semantic-audit-runtime-identity-mismatch", f"{label} runtime identity drifted")
    process = _exact_mapping(
        row["process_identity"],
        frozenset(("pid", "start_ticks")),
        f"{label}.process_identity",
    )
    if (
        _positive_u64(process["pid"], f"{label}.process_identity.pid") != target.target.pid
        or _positive_u64(process["start_ticks"], f"{label}.process_identity.start_ticks")
        != target.target.start_ticks
    ):
        _fail("semantic-audit-process-identity-mismatch", f"{label} process identity drifted")
    if row["server_request_id"] != request_id or row["delivery_mode"] != "non-stream":
        _fail("semantic-request-audit-binding-mismatch", f"{label} request binding drifted")

    prompt = row["prompt_token_ids"]
    if not isinstance(prompt, list) or not 1 <= len(prompt) <= _MAX_PROMPT_TOKEN_IDS:
        _fail("semantic-audit-prompt-token-count", f"{label}.prompt_token_ids is outside source bounds")
    for index, token_id in enumerate(prompt):
        _u32(token_id, f"{label}.prompt_token_ids[{index}]")

    outputs = row["committed_output_tokens"]
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= _MAX_COMMITTED_OUTPUT_TOKENS:
        _fail("semantic-audit-output-token-count", f"{label}.committed_output_tokens is outside source bounds")
    for index, output in enumerate(outputs):
        output_row = _exact_mapping(
            output,
            frozenset(("emitted_text_delta", "token_id")),
            f"{label}.committed_output_tokens[{index}]",
        )
        _u32(output_row["token_id"], f"{label}.committed_output_tokens[{index}].token_id")
        if (
            type(output_row["emitted_text_delta"]) is not str
            or len(output_row["emitted_text_delta"]) > _MAX_EMITTED_TEXT_DELTA_CHARS
        ):
            _fail(
                "semantic-audit-output-delta",
                f"{label}.committed_output_tokens[{index}].emitted_text_delta is outside source bounds",
            )

    selections_value = row["sampling_selections"]
    if (
        not isinstance(selections_value, list)
        or not 1 <= len(selections_value) <= _MAX_SAMPLING_SELECTIONS
        or len(selections_value) != len(outputs)
    ):
        _fail(
            "semantic-audit-selection-cardinality",
            f"{label} does not preserve one typed selection per committed output token",
        )
    selected_counts = {backend: 0 for backend in sorted(_BACKENDS)}
    reason_counts = {reason: 0 for reason in sorted(_INELIGIBILITY_REASONS)}
    previous_iteration: int | None = None
    selections: list[dict[str, Any]] = []
    for index, selection_value in enumerate(selections_value):
        selection = _exact_mapping(
            selection_value,
            frozenset(
                {
                    "iteration_id", "configured_backend", "selected_backend",
                    "ineligibility_reason", "committed",
                }
            ),
            f"{label}.sampling_selections[{index}]",
        )
        iteration_id = _positive_u64(
            selection["iteration_id"], f"{label}.sampling_selections[{index}].iteration_id"
        )
        if previous_iteration is not None and iteration_id <= previous_iteration:
            _fail(
                "semantic-audit-iteration-order",
                f"{label}.sampling_selections must have strictly increasing iteration_id",
            )
        previous_iteration = iteration_id
        configured = selection["configured_backend"]
        selected = selection["selected_backend"]
        reason = selection["ineligibility_reason"]
        if (
            type(configured) is not str
            or type(selected) is not str
            or configured not in _BACKENDS
            or selected not in _BACKENDS
        ):
            _fail("semantic-audit-unknown-backend", f"{label} contains an unknown typed backend")
        if reason is not None and (
            type(reason) is not str or reason not in _INELIGIBILITY_REASONS
        ):
            _fail("semantic-audit-unknown-ineligibility", f"{label} contains an unknown typed reason")
        if selection["committed"] is not True:
            _fail("semantic-audit-uncommitted-selection", f"{label} contains an uncommitted selection")
        if selected == "gpu-greedy":
            if configured != "gpu-greedy" or reason is not None:
                _fail(
                    "semantic-audit-selection-pairing",
                    f"{label} gpu-greedy selection has an invalid typed pairing",
                )
        elif reason is None:
            _fail(
                "semantic-audit-selection-pairing",
                f"{label} cpu-normative selection lacks an ineligibility reason",
            )
        elif configured == "cpu-normative" and reason != "gpu-greedy-not-configured":
            _fail(
                "semantic-audit-selection-pairing",
                f"{label} cpu-configured selection has an invalid typed reason",
            )
        elif configured == "gpu-greedy" and reason == "gpu-greedy-not-configured":
            _fail(
                "semantic-audit-selection-pairing",
                f"{label} gpu-configured selection has an invalid typed reason",
            )
        selected_counts[selected] += 1
        if reason is not None:
            reason_counts[reason] += 1
        selections.append(dict(selection))

    if type(row["finish_reason"]) is not str or row["finish_reason"] not in {"length", "stop"}:
        _fail("semantic-audit-terminal-reason", f"{label} is not a successful terminal audit")
    usage = _exact_mapping(
        row["usage"],
        frozenset(("prompt_tokens", "completion_tokens", "total_tokens")),
        f"{label}.usage",
    )
    prompt_tokens = _nonnegative_u64(usage["prompt_tokens"], f"{label}.usage.prompt_tokens")
    completion_tokens = _nonnegative_u64(
        usage["completion_tokens"], f"{label}.usage.completion_tokens"
    )
    total_tokens = _nonnegative_u64(usage["total_tokens"], f"{label}.usage.total_tokens")
    if (
        prompt_tokens != len(prompt)
        or completion_tokens != len(outputs)
        or total_tokens != prompt_tokens + completion_tokens
    ):
        _fail("semantic-audit-token-accounting", f"{label} token accounting does not match source leaves")
    assert previous_iteration is not None
    return (
        {
            "output_token_count": len(outputs),
            "selection_count": len(selections),
            "all_committed": True,
            "derived_selected_backend_counts": selected_counts,
            "derived_ineligibility_reason_counts": reason_counts,
            "fallback_projection_replayed": False,
            "first_iteration_id": selections[0]["iteration_id"],
            "last_iteration_id": previous_iteration,
        },
        tuple(selections),
    )


def _reconstruct_v5_fallback_projection(
    event_bytes: bytes,
    *,
    audit: common.EvidenceDescriptor,
    audit_selections: tuple[dict[str, Any], ...],
    candidate_id: str,
    bindings: raw.ReplayedSoakSemanticBindings,
    target: raw.ObservedTarget,
    request_id: str,
    label: str,
) -> None:
    """Require the native fallback event to be an exact typed audit projection."""

    document = _common(
        lambda: common.parse_canonical_json(
            event_bytes,
            label,
            maximum_bytes=raw.MAX_RAW_BYTES,
        )
    )
    row = _exact_mapping(
        document,
        frozenset(
            {
                "schema_version", "candidate_id", "runtime_identity", "process_identity",
                "server_request_id", "generation_audit", "fallback_selections",
            }
        ),
        label,
    )
    if row["schema_version"] != raw.SCENARIO_FALLBACK_EVENT_VERSION:
        _fail("historical-source-fallback-event-version-rejected", f"{label} has an unsupported version")
    if row["candidate_id"] != candidate_id or row["server_request_id"] != request_id:
        _fail("semantic-fallback-event-identity-mismatch", f"{label} identity differs from source audit")
    identity = _exact_mapping(
        row["runtime_identity"],
        frozenset(("configuration_profile", "configuration_sha256")),
        f"{label}.runtime_identity",
    )
    if (
        identity["configuration_profile"] != bindings.configuration_profile
        or identity["configuration_sha256"] != bindings.configuration_sha256
    ):
        _fail("semantic-fallback-event-runtime-identity-mismatch", f"{label} runtime identity drifted")
    process = _exact_mapping(
        row["process_identity"],
        frozenset(("pid", "start_ticks")),
        f"{label}.process_identity",
    )
    if (
        _positive_u64(process["pid"], f"{label}.process_identity.pid") != target.target.pid
        or _positive_u64(process["start_ticks"], f"{label}.process_identity.start_ticks")
        != target.target.start_ticks
    ):
        _fail("semantic-fallback-event-process-identity-mismatch", f"{label} process identity drifted")
    audit_binding = _exact_mapping(
        row["generation_audit"],
        frozenset(("artifact_filename", "artifact_sha256")),
        f"{label}.generation_audit",
    )
    if (
        audit_binding["artifact_filename"] != Path(audit.path).name
        or audit_binding["artifact_sha256"] != audit.sha256
    ):
        _fail("semantic-fallback-event-audit-binding-mismatch", f"{label} does not bind audit bytes")
    selections = row["fallback_selections"]
    if not isinstance(selections, list) or tuple(selections) != audit_selections:
        _fail("semantic-fallback-event-selection-projection", f"{label} is not the exact audit projection")
    for index, selection in enumerate(audit_selections):
        if (
            selection["configured_backend"] != "gpu-greedy"
            or selection["selected_backend"] != "cpu-normative"
            or selection["ineligibility_reason"] != "nonzero-temperature"
            or selection["committed"] is not True
        ):
            _fail(
                "semantic-fallback-event-transition",
                f"{label}.fallback_selections[{index}] is not the typed native fallback transition",
            )


def _reconstruct_observation_semantics(
    observation: raw.ReplayedObservationSessionSemanticInputs,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive only per-session timing and cumulative request-counter facts."""

    if not observation.samples:
        _fail("semantic-observation-sample-inventory", f"{label} has no samples")
    previous_elapsed: int | None = None
    previous_counters: dict[str, int] | None = None
    terminal_counters: dict[str, int] | None = None
    for index, sample in enumerate(observation.samples):
        if sample.sequence != index:
            _fail("semantic-observation-sequence", f"{label} sample sequence is not contiguous")
        elapsed = _nonnegative_u64(
            sample.elapsed_monotonic_millis,
            f"{label}.samples[{index}].elapsed_monotonic_millis",
        )
        if previous_elapsed is not None and elapsed <= previous_elapsed:
            _fail("semantic-observation-interval-order", f"{label} elapsed time is not strictly increasing")
        previous_elapsed = elapsed
        document = _common(
            lambda sample=sample, index=index: common.parse_canonical_json(
                sample.metrics_bytes,
                f"{label}.samples[{index}].metrics",
                maximum_bytes=raw.MAX_RAW_BYTES,
            )
        )
        metrics = _exact_mapping(
            document,
            frozenset(("schema_version", "request_states", "kv_blocks", "allocation", "quiescence")),
            f"{label}.samples[{index}].metrics",
        )
        if metrics["schema_version"] != raw.METRICS_VERSION:
            _fail("unsupported-metrics-version", f"{label} metrics use an unsupported version")
        states = _exact_mapping(
            metrics["request_states"],
            frozenset(
                {
                    "active", "pending_requests", "completed", "failed", "cancelled",
                    "capacity_rejections",
                }
            ),
            f"{label}.samples[{index}].metrics.request_states",
        )
        kv_blocks = _exact_mapping(
            metrics["kv_blocks"],
            frozenset(("free", "reserved", "active")),
            f"{label}.samples[{index}].metrics.kv_blocks",
        )
        allocation = _exact_mapping(
            metrics["allocation"],
            frozenset(
                {"device_live_count", "device_live_bytes", "pinned_live_count", "pinned_live_bytes"}
            ),
            f"{label}.samples[{index}].metrics.allocation",
        )
        quiescence = _exact_mapping(
            metrics["quiescence"],
            frozenset(
                {
                    "completion_outbox", "outstanding_iterations", "riley_owned_live_allocations",
                    "worker_accepting", "scheduler_accepting",
                }
            ),
            f"{label}.samples[{index}].metrics.quiescence",
        )
        for group, group_label in (
            (states, "request_states"),
            (kv_blocks, "kv_blocks"),
            (allocation, "allocation"),
        ):
            for field, value in group.items():
                _nonnegative_u64(value, f"{label}.samples[{index}].metrics.{group_label}.{field}")
        for field in ("completion_outbox", "outstanding_iterations", "riley_owned_live_allocations"):
            _nonnegative_u64(
                quiescence[field], f"{label}.samples[{index}].metrics.quiescence.{field}"
            )
        for field in ("worker_accepting", "scheduler_accepting"):
            if type(quiescence[field]) is not bool:
                _fail("invalid-semantic-boolean", f"{label} quiescence.{field} must be boolean")
        counters = {
            field: _nonnegative_u64(states[field], f"{label}.samples[{index}].metrics.request_states.{field}")
            for field in ("completed", "failed", "cancelled", "capacity_rejections")
        }
        if previous_counters is not None:
            for field, value in counters.items():
                if value < previous_counters[field]:
                    _fail(
                        "semantic-observation-cumulative-counter-regression",
                        f"{label} cumulative request counter {field} regressed",
                    )
        previous_counters = counters
        terminal_counters = counters
    assert previous_elapsed is not None
    assert terminal_counters is not None
    return (
        {
            "scope": "per-observation-session",
            "sample_count": len(observation.samples),
            "first_elapsed_monotonic_millis": observation.samples[0].elapsed_monotonic_millis,
            "last_elapsed_monotonic_millis": previous_elapsed,
            "strictly_increasing": True,
        },
        {
            "cumulative_request_counters_monotonic": True,
            "terminal_cumulative_request_counters": terminal_counters,
        },
    )


def _reconstruct_scenario(
    scenario: raw.ReplayedSoakSemanticScenarioInputs,
    *,
    candidate_id: str,
    bindings: raw.ReplayedSoakSemanticBindings,
    raw_manifest_version: str,
) -> dict[str, Any]:
    label = f"semantic scenario {scenario.scenario_id}"
    interval, metrics = _reconstruct_observation_semantics(scenario.observation, label=label)
    typed_sampling, selections = _reconstruct_generation_audit(
        scenario.generation_audit_bytes,
        candidate_id=candidate_id,
        bindings=bindings,
        target=scenario.target,
        request_id=scenario.request_id,
        label=f"{label}.generation_audit",
    )
    result: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "target": _target_fact(scenario.target),
        "request_id": scenario.request_id,
        "request_ledger": scenario.request_ledger.as_json(),
        "generation_audit": scenario.generation_audit.as_json(),
        "observation_session": scenario.observation.session.as_json(),
        "interval": interval,
        "metrics": metrics,
        "typed_sampling": typed_sampling,
    }
    if raw_manifest_version == raw.SOAK_V5_MANIFEST_VERSION:
        if (
            scenario.scenario_id != raw.FALLBACK_SCENARIO_ID
            or scenario.fallback_event is None
            or scenario.fallback_event_bytes is None
            or bindings.configuration_profile != raw.MAX_PERFORMANCE_EXACT_PROFILE
        ):
            _fail("semantic-v5-fallback-input-mismatch", f"{label} does not preserve the v5 fallback arm")
        _reconstruct_v5_fallback_projection(
            scenario.fallback_event_bytes,
            audit=scenario.generation_audit,
            audit_selections=selections,
            candidate_id=candidate_id,
            bindings=bindings,
            target=scenario.target,
            request_id=scenario.request_id,
            label=f"{label}.fallback_event",
        )
        typed_sampling["fallback_projection_replayed"] = True
        result["fallback_event"] = scenario.fallback_event.as_json()
    elif scenario.fallback_event is not None or scenario.fallback_event_bytes is not None:
        _fail("semantic-v4-fallback-input-forbidden", f"{label} unexpectedly carries a fallback event")
    return result


def _replay_once(
    root_fd: int,
    manifest_name: str,
    manifest_version: str,
    header_descriptor: common.EvidenceDescriptor,
) -> dict[str, Any]:
    semantic_input_replay = _select_semantic_input_replay(manifest_version)
    inputs = _raw(lambda: semantic_input_replay(root_fd, manifest_name))
    if inputs.raw_manifest != header_descriptor:
        _fail(
            "raw-manifest-changed-during-version-dispatch",
            "raw manifest differs between version dispatch and held-FD semantic replay",
        )
    if inputs.raw_manifest_version != manifest_version:
        _fail("semantic-input-version-drift", "raw semantic input version changed during replay")
    scenarios = tuple(
        _reconstruct_scenario(
            scenario,
            candidate_id=inputs.candidate_id,
            bindings=inputs.bindings,
            raw_manifest_version=inputs.raw_manifest_version,
        )
        for scenario in inputs.scenarios
    )
    if not scenarios:
        _fail("semantic-scenario-inventory", "semantic replay has no source scenarios")
    if manifest_version == raw.SOAK_V5_MANIFEST_VERSION and len(scenarios) != 1:
        _fail("semantic-v5-scenario-inventory", "v5 semantic replay must have exactly one scenario")
    return {
        "candidate_id": inputs.candidate_id,
        "bindings": inputs.bindings.as_json(),
        "raw_manifest": inputs.raw_manifest.as_json(),
        "raw_manifest_version": inputs.raw_manifest_version,
        "derived_facts": {"scenarios": list(scenarios)},
    }


def _semantic_report(replayed: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SEMANTIC_REPORT_VERSION,
        "scope": SEMANTIC_SCOPE,
        "status": "passed",
        "qualification_status": "not-run",
        "authority": SEMANTIC_AUTHORITY,
        "semantic_policy_version": SEMANTIC_POLICY_VERSION,
        "candidate_id": replayed["candidate_id"],
        "bindings": replayed["bindings"],
        "raw_manifest": replayed["raw_manifest"],
        "raw_manifest_version": replayed["raw_manifest_version"],
        "derived_facts": replayed["derived_facts"],
        "checks": [{"name": name, "passed": True} for name in _CHECK_NAMES],
        "not_established": dict(_NOT_ESTABLISHED),
        "reason_codes": [],
    }


def check_soak_v2_receipt_v2(evidence_root: Path, raw_manifest: str) -> dict[str, Any]:
    """Replay original completed v4/v5 raw leaves into a narrow semantic diagnostic."""

    root = _evidence_root(evidence_root)
    manifest_name = _raw_manifest_name(raw_manifest)
    root_fd = _common(lambda: common.open_private_evidence_directory(root, "--evidence-root"))
    try:
        _shared_lock(root_fd)
        header_descriptor, manifest_version = _manifest_header(root_fd, manifest_name)
        _select_semantic_input_replay(manifest_version)
        first = _replay_once(root_fd, manifest_name, manifest_version, header_descriptor)
        second = _replay_once(root_fd, manifest_name, manifest_version, header_descriptor)
        if common.canonical_json_bytes(first) != common.canonical_json_bytes(second):
            _fail("semantic-replay-drift", "semantic inputs changed between held-FD replays")
        return _semantic_report(first)
    finally:
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--raw-manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = check_soak_v2_receipt_v2(args.evidence_root, args.raw_manifest)
    except SoakV2SemanticReplayError as error:
        print(f"C02 soak v2 semantic replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
