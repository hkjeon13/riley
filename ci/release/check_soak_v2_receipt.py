#!/usr/bin/env python3
"""Fail-closed C02 semantic replay for the extended full-duration soak gate.

The existing Gate E soak archive remains authoritative for its reviewed PR16
scenario set.  C02 adds a source-controlled second full-duration trace with
explicit cancellation-rate, KV-utilisation, exact-backend fallback, and model
lifecycle arms.  Neither input contains a generic ``passed`` assertion: this
checker snapshots the declared raw inputs, replays the Gate E archive, checks
the static C02 contract, and validates every raw scenario result before it
emits a separate semantic check report.

It is deliberately CPU-only.  It does not start Riley, CUDA, a container,
SSH, or a network request.

Outer C02 integration API::

    parsed = validate_check_report(submitted_report)
    replayed = evaluate(
        freeze_path,
        evidence_root,
        parsed.receipt.path,
        expected_freeze_sha256=trusted_freeze_sha256,
    )
    assert submitted_report == replayed

The freeze-declared ``outputs.receipts.soak_v2.path`` is this check report,
never the raw receipt passed to :func:`evaluate`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, Sequence

import check_rc3_qualification as qualification


RECEIPT_VERSION = "riley.soak-v2-receipt.v1"
TRACE_VERSION = "riley.soak-v2-scenario-trace.v1"
CHECK_REPORT_VERSION = "riley.soak-v2-check.v1"
CONTRACT_VERSION = "riley.c02-soak-v2-scenario-contract.v1"
CONTRACT_RELATIVE_PATH = "benchmarks/release/candidates/soak-v2-scenarios-v1.json"
CONTRACT_SHA256 = "791a8ebc6880ad7f2227aa365e2de8ac7a99405adb39ce48ab94e300b3c21d3d"
CONTRACT_ID = "c02-soak-v2-full-duration-v1"
STABLE_DEFAULT_PROFILE = "stable-default"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_SCENARIO_RESULTS = 32
MAX_INTERVAL_OBSERVATIONS = 1024
EXPECTED_MAXIMUM_INTERVAL_SECONDS = 300

# Keep this ordered list in the C02 checker rather than inferring an inventory
# from a potentially altered source file.  The static contract hash is one
# anchor; these semantic assertions are a second, independently readable one.
V1_SCENARIOS = (
    ("steady", "steady", 14_400),
    ("burst-idle", "burst-idle", 3_600),
    ("mixed-short-long", "mixed", 3_600),
    ("invalid", "invalid", 300),
    ("overload", "overload", 600),
    ("cancellation-disconnect", "cancellation-disconnect", 900),
    ("near-kv", "near-kv", 1_800),
    ("graceful-restart", "graceful-restart", 300),
    ("rollback-iteration-batch", "rollback", 300),
    ("rollback-per-operation", "rollback", 300),
)
EXTENDED_SCENARIOS = (
    ("cancellation-0", "cancellation-rate", 3_600, 0),
    ("cancellation-10", "cancellation-rate", 3_600, 10),
    ("cancellation-50", "cancellation-rate", 3_600, 50),
    ("kv-70", "kv-utilization", 3_600, None),
    ("kv-90", "kv-utilization", 3_600, None),
    ("kv-capacity", "kv-capacity-boundary", 1_800, None),
    ("exact-backend-fallback", "exact-backend-fallback", 3_600, None),
    ("repeated-model-load-unload", "model-load-unload", 2_700, None),
)
EXPECTED_SCENARIO_IDS = tuple(item[0] for item in V1_SCENARIOS + EXTENDED_SCENARIOS)
EXPECTED_TOTAL_DURATION_SECONDS = sum(item[2] for item in V1_SCENARIOS + EXTENDED_SCENARIOS)
MODEL_LIFECYCLE_EVENTS = (
    "model-load-requested",
    "model-ready",
    "model-unload-requested",
    "model-unloaded",
)
CHECK_NAMES = (
    "freeze-binding",
    "gate-e-replay",
    "stable-default-arm-binding",
    "source-contract-binding",
    "gate-e-soak-binding",
    "gate-e-soak-raw-archive-replay",
    "soak-v2-trace-binding",
    "full-duration-scenario-inventory",
    "cancellation-rate-semantics",
    "kv-capacity-semantics",
    "exact-backend-fallback-semantics",
    "repeated-model-lifecycle-semantics",
)

_OPEN_COMMON = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
_OPEN_DIRECTORY = _OPEN_COMMON | getattr(os, "O_DIRECTORY", 0)


class SoakV2ReceiptError(qualification.QualificationError):
    """A C02 extended soak input cannot establish its semantic gate."""


class SoakV2ReceiptIncomparable(qualification.IncomparableError):
    """An otherwise valid extended soak input binds another immutable RC."""


@dataclass(frozen=True)
class Descriptor:
    path: str
    sha256: str


@dataclass(frozen=True)
class ScenarioContract:
    descriptor: Descriptor
    scenarios: tuple[dict[str, Any], ...]
    total_duration_seconds: int
    maximum_interval_seconds: int


@dataclass(frozen=True)
class GateESoakInputs:
    report: Descriptor
    raw_archive: Descriptor
    correctness_golden: Descriptor
    native_correctness_report: Descriptor


@dataclass(frozen=True)
class SoakV2Receipt:
    candidate_id: str
    bindings: dict[str, str]
    scenario_contract: Descriptor
    gate_e_soak: GateESoakInputs
    scenario_trace: Descriptor


@dataclass(frozen=True)
class SoakV2Trace:
    candidate_id: str
    bindings: dict[str, str]
    scenario_contract: Descriptor
    gate_e_soak_raw_sha256: str
    scenario_results: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SoakV2CheckReport:
    """Closed descriptors exposed to a future outer semantic gate replay."""

    candidate_id: str
    freeze_sha256: str
    base_release_candidate_report: Descriptor
    scenario_contract: Descriptor
    receipt: Descriptor
    bindings: dict[str, str]
    gate_e_soak: GateESoakInputs
    scenario_trace: Descriptor
    scenario_results: tuple[dict[str, Any], ...]


def _raise(error_type: type[qualification.QualificationError], code: str, message: str) -> NoReturn:
    error = error_type(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _fail(code: str, message: str) -> NoReturn:
    _raise(SoakV2ReceiptError, code, message)


def _incomparable(message: str) -> NoReturn:
    _raise(SoakV2ReceiptIncomparable, "incomparable-binding", message)


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    return qualification._exact(value, fields, label)


def _sha256(value: Any, label: str) -> str:
    return qualification._sha256(value, label)


def _candidate_id(value: Any, label: str) -> str:
    candidate_id = qualification._string(value, label)
    if not qualification.release_candidate.CANDIDATE_ID_RE.fullmatch(candidate_id):
        _fail("invalid-candidate-id", f"{label} is not a valid RC candidate")
    return candidate_id


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail("invalid-soak-v2-trace", f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail("invalid-soak-v2-trace", f"{label} must be a non-negative integer")
    return value


def _descriptor(value: Any, label: str) -> Descriptor:
    row = _exact(value, {"path", "sha256"}, label)
    return Descriptor(
        path=qualification._relative_path(row["path"], f"{label}.path"),
        sha256=_sha256(row["sha256"], f"{label}.sha256"),
    )


def _bindings(value: Any, label: str) -> dict[str, str]:
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
    if row["configuration_profile"] != STABLE_DEFAULT_PROFILE:
        _incomparable(f"{label}.configuration_profile is not {STABLE_DEFAULT_PROFILE}")
    return {
        "freeze_sha256": _sha256(row["freeze_sha256"], f"{label}.freeze_sha256"),
        "base_release_candidate_report_sha256": _sha256(
            row["base_release_candidate_report_sha256"],
            f"{label}.base_release_candidate_report_sha256",
        ),
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": _sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
    }


def _scenario_contract_path() -> Path:
    return Path(__file__).resolve().parents[2] / CONTRACT_RELATIVE_PATH


def _validate_contract(document: dict[str, Any]) -> ScenarioContract:
    row = _exact(
        document,
        {
            "schema_version",
            "contract_id",
            "base_gate_contract_id",
            "total_duration_seconds",
            "maximum_interval_seconds",
            "scenarios",
        },
        "soak v2 source contract",
    )
    if row["schema_version"] != CONTRACT_VERSION:
        _fail("unsupported-soak-v2-contract-version", "source scenario contract version is unsupported")
    if row["contract_id"] != CONTRACT_ID:
        _fail("invalid-soak-v2-contract", "source scenario contract ID drifted")
    if row["base_gate_contract_id"] != qualification.release_candidate.SOAK_CONTRACT_ID:
        _fail("invalid-soak-v2-contract", "source contract no longer names Gate E's reviewed soak")
    inherited_gate_e = tuple(
        (scenario_id, kind, duration)
        for scenario_id, (kind, duration) in qualification.release_candidate.SOAK_SCENARIOS.items()
    )
    if inherited_gate_e != V1_SCENARIOS:
        _fail(
            "invalid-soak-v2-contract",
            "checker inherited Gate E inventory no longer matches the reviewed base gate",
        )
    total_duration_seconds = _positive_int(
        row["total_duration_seconds"], "soak v2 source contract.total_duration_seconds"
    )
    if total_duration_seconds != EXPECTED_TOTAL_DURATION_SECONDS:
        _fail("invalid-soak-v2-contract", "source contract must retain the reviewed full duration")
    maximum_interval_seconds = _positive_int(
        row["maximum_interval_seconds"], "soak v2 source contract.maximum_interval_seconds"
    )
    if maximum_interval_seconds != EXPECTED_MAXIMUM_INTERVAL_SECONDS:
        _fail("invalid-soak-v2-contract", "source contract must retain its reviewed sampling interval")
    scenarios = row["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != len(EXPECTED_SCENARIO_IDS):
        _fail("invalid-soak-v2-contract", "source contract has an invalid scenario count")
    normalized: list[dict[str, Any]] = []
    scenario_ids: list[str] = []
    for index, value in enumerate(scenarios):
        label = f"soak v2 source contract.scenarios[{index}]"
        item = _exact(
            value,
            {
                "id",
                "kind",
                "duration_seconds",
                "minimum_requests",
                "cancellation_percent",
                "minimum_kv_utilization_percent",
                "maximum_kv_utilization_percent",
                "minimum_capacity_rejections",
                "minimum_exact_backend_fallbacks",
                "required_backend_events",
                "minimum_model_load_unload_cycles",
            },
            label,
        )
        scenario_id = qualification._string(item["id"], f"{label}.id")
        kind = qualification._string(item["kind"], f"{label}.kind")
        duration_seconds = _positive_int(item["duration_seconds"], f"{label}.duration_seconds")
        minimum_requests = _positive_int(item["minimum_requests"], f"{label}.minimum_requests")
        cancellation_percent = _nonnegative_int(
            item["cancellation_percent"], f"{label}.cancellation_percent"
        )
        if cancellation_percent not in {0, 10, 50}:
            _fail("invalid-soak-v2-contract", f"{label}.cancellation_percent is unreviewed")
        minimum_kv = _nonnegative_int(
            item["minimum_kv_utilization_percent"],
            f"{label}.minimum_kv_utilization_percent",
        )
        maximum_kv = _nonnegative_int(
            item["maximum_kv_utilization_percent"],
            f"{label}.maximum_kv_utilization_percent",
        )
        if minimum_kv > maximum_kv or maximum_kv > 100:
            _fail("invalid-soak-v2-contract", f"{label} has an invalid KV utilisation range")
        capacity_rejections = _nonnegative_int(
            item["minimum_capacity_rejections"], f"{label}.minimum_capacity_rejections"
        )
        fallback_count = _nonnegative_int(
            item["minimum_exact_backend_fallbacks"],
            f"{label}.minimum_exact_backend_fallbacks",
        )
        lifecycle_cycles = _nonnegative_int(
            item["minimum_model_load_unload_cycles"],
            f"{label}.minimum_model_load_unload_cycles",
        )
        backend_events = item["required_backend_events"]
        if not isinstance(backend_events, list) or any(
            not isinstance(event, str) or not event for event in backend_events
        ):
            _fail("invalid-soak-v2-contract", f"{label}.required_backend_events must be text")
        scenario_ids.append(scenario_id)
        normalized.append(
            {
                "id": scenario_id,
                "kind": kind,
                "duration_seconds": duration_seconds,
                "minimum_requests": minimum_requests,
                "cancellation_percent": cancellation_percent,
                "minimum_kv_utilization_percent": minimum_kv,
                "maximum_kv_utilization_percent": maximum_kv,
                "minimum_capacity_rejections": capacity_rejections,
                "minimum_exact_backend_fallbacks": fallback_count,
                "required_backend_events": list(backend_events),
                "minimum_model_load_unload_cycles": lifecycle_cycles,
            }
        )
    if tuple(scenario_ids) != EXPECTED_SCENARIO_IDS:
        _fail("invalid-soak-v2-contract", "source contract scenario IDs or order drifted")
    if sum(item["duration_seconds"] for item in normalized) != total_duration_seconds:
        _fail("invalid-soak-v2-contract", "source contract duration does not equal its inventory")
    for index, (scenario_id, kind, duration) in enumerate(V1_SCENARIOS):
        item = normalized[index]
        if (item["id"], item["kind"], item["duration_seconds"]) != (scenario_id, kind, duration):
            _fail("invalid-soak-v2-contract", "inherited Gate E v1 scenario contract drifted")
    for offset, (scenario_id, kind, duration, cancellation_percent) in enumerate(EXTENDED_SCENARIOS):
        item = normalized[len(V1_SCENARIOS) + offset]
        if (item["id"], item["kind"], item["duration_seconds"]) != (scenario_id, kind, duration):
            _fail("invalid-soak-v2-contract", "C02 extended scenario contract drifted")
        if cancellation_percent is not None and item["cancellation_percent"] != cancellation_percent:
            _fail("invalid-soak-v2-contract", "C02 cancellation-rate arm drifted")
    expected_fallback_events = [
        "fast-backend-unavailable",
        "exact-backend-fallback-selected",
        "exact-backend-generation-complete",
    ]
    fallback = normalized[EXPECTED_SCENARIO_IDS.index("exact-backend-fallback")]
    if (
        fallback["minimum_exact_backend_fallbacks"] < 1
        or fallback["required_backend_events"] != expected_fallback_events
    ):
        _fail("invalid-soak-v2-contract", "exact-backend fallback contract drifted")
    lifecycle = normalized[EXPECTED_SCENARIO_IDS.index("repeated-model-load-unload")]
    if lifecycle["minimum_model_load_unload_cycles"] < 8:
        _fail("invalid-soak-v2-contract", "model lifecycle contract is not repeated enough")
    capacity = normalized[EXPECTED_SCENARIO_IDS.index("kv-capacity")]
    if (
        capacity["minimum_kv_utilization_percent"] != 100
        or capacity["maximum_kv_utilization_percent"] != 100
        or capacity["minimum_capacity_rejections"] < 1
    ):
        _fail("invalid-soak-v2-contract", "KV capacity-boundary contract drifted")
    return ScenarioContract(
        descriptor=Descriptor(CONTRACT_RELATIVE_PATH, CONTRACT_SHA256),
        scenarios=tuple(normalized),
        total_duration_seconds=total_duration_seconds,
        maximum_interval_seconds=maximum_interval_seconds,
    )


def _load_contract() -> ScenarioContract:
    path = _scenario_contract_path()
    try:
        raw = qualification._read_regular_path(path, "soak v2 source contract")
    except qualification.QualificationError:
        raise
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != CONTRACT_SHA256:
        _fail("source-contract-hash-mismatch", "source scenario contract bytes drifted")
    return _validate_contract(qualification._parse_document(raw, "soak v2 source contract"))


def _distinct_paths(paths: Sequence[str], label: str) -> None:
    if len(paths) != len(set(paths)):
        _fail("duplicate-evidence-path", f"{label} must not reuse a path")


def validate_receipt(document: dict[str, Any]) -> SoakV2Receipt:
    """Parse the non-authoritative raw descriptor without accepting a status."""

    row = _exact(
        document,
        {"schema_version", "candidate_id", "bindings", "scenario_contract", "gate_e_soak", "scenario_trace"},
        "soak v2 receipt",
    )
    if row["schema_version"] != RECEIPT_VERSION:
        _fail("unsupported-soak-v2-receipt-version", "soak v2 receipt schema_version is unsupported")
    gate_row = _exact(
        row["gate_e_soak"],
        {"report", "raw_archive", "correctness_golden", "native_correctness_report"},
        "soak v2 receipt.gate_e_soak",
    )
    gate_e_soak = GateESoakInputs(
        report=_descriptor(gate_row["report"], "soak v2 receipt.gate_e_soak.report"),
        raw_archive=_descriptor(
            gate_row["raw_archive"], "soak v2 receipt.gate_e_soak.raw_archive"
        ),
        correctness_golden=_descriptor(
            gate_row["correctness_golden"], "soak v2 receipt.gate_e_soak.correctness_golden"
        ),
        native_correctness_report=_descriptor(
            gate_row["native_correctness_report"],
            "soak v2 receipt.gate_e_soak.native_correctness_report",
        ),
    )
    trace = _descriptor(row["scenario_trace"], "soak v2 receipt.scenario_trace")
    _distinct_paths(
        (
            gate_e_soak.report.path,
            gate_e_soak.raw_archive.path,
            gate_e_soak.correctness_golden.path,
            gate_e_soak.native_correctness_report.path,
            trace.path,
        ),
        "soak v2 receipt evidence",
    )
    return SoakV2Receipt(
        candidate_id=_candidate_id(row["candidate_id"], "soak v2 receipt.candidate_id"),
        bindings=_bindings(row["bindings"], "soak v2 receipt.bindings"),
        scenario_contract=_descriptor(
            row["scenario_contract"], "soak v2 receipt.scenario_contract"
        ),
        gate_e_soak=gate_e_soak,
        scenario_trace=trace,
    )


def _validate_lifecycle_cycles(value: Any, minimum: int, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        _fail("invalid-soak-v2-trace", f"{label} must be an array")
    if minimum == 0:
        if value != []:
            _fail("unexpected-model-lifecycle", f"{label} is not allowed for this scenario")
        return []
    if len(value) < minimum or len(value) > 1024:
        _fail("model-lifecycle-coverage-missing", f"{label} does not prove enough load/unload cycles")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        item = _exact(raw, {"cycle", "events"}, f"{label}[{index - 1}]")
        if _positive_int(item["cycle"], f"{label}[{index - 1}].cycle") != index:
            _fail("invalid-soak-v2-trace", f"{label} cycles must be contiguous from one")
        if item["events"] != list(MODEL_LIFECYCLE_EVENTS):
            _fail("model-lifecycle-order-mismatch", f"{label}[{index - 1}] event order drifted")
        normalized.append({"cycle": index, "events": list(MODEL_LIFECYCLE_EVENTS)})
    return normalized


def _validate_interval_observations(
    value: Any,
    *,
    attempted: int,
    completed: int,
    cancelled: int,
    capacity_rejections: int,
    terminal_events: int,
    observed_duration_seconds: int,
    maximum_interval_seconds: int,
    label: str,
) -> list[dict[str, Any]]:
    """Validate the C02 interval telemetry rather than accept a final summary.

    C02 requires the producer to expose operational counters during the entire
    scenario, not merely at its end.  The fields are intentionally raw totals:
    this local verifier can prove their coverage, ordering, and reconciliation
    with the scenario outcomes without pretending to observe the remote GPU.
    """

    if not isinstance(value, list) or len(value) < 2 or len(value) > MAX_INTERVAL_OBSERVATIONS:
        _fail("invalid-interval-observations", f"{label} must contain a bounded start/end interval inventory")
    minimum_samples = (observed_duration_seconds + maximum_interval_seconds - 1) // maximum_interval_seconds + 1
    if len(value) < minimum_samples:
        _fail("interval-observation-coverage-missing", f"{label} is missing reviewed soak intervals")

    normalized: list[dict[str, Any]] = []
    previous_elapsed = -1
    previous_completed = previous_cancelled = previous_rejected = previous_terminal = -1
    for index, raw in enumerate(value):
        item = _exact(
            raw,
            {
                "elapsed_seconds",
                "rss_bytes",
                "pinned_bytes",
                "vram_bytes",
                "kv_blocks",
                "request_states",
                "terminal_events_total",
            },
            f"{label}[{index}]",
        )
        elapsed = _nonnegative_int(item["elapsed_seconds"], f"{label}[{index}].elapsed_seconds")
        if index == 0 and elapsed != 0:
            _fail("interval-observation-coverage-missing", f"{label} must begin at elapsed second zero")
        if elapsed <= previous_elapsed:
            _fail("interval-observation-order-mismatch", f"{label} elapsed seconds must be strictly increasing")
        if previous_elapsed >= 0 and elapsed - previous_elapsed > maximum_interval_seconds:
            _fail("interval-observation-coverage-missing", f"{label} has a gap beyond the reviewed interval")
        if elapsed > observed_duration_seconds:
            _fail("invalid-interval-observations", f"{label} records telemetry after its scenario ended")

        rss_bytes = _positive_int(item["rss_bytes"], f"{label}[{index}].rss_bytes")
        pinned_bytes = _nonnegative_int(item["pinned_bytes"], f"{label}[{index}].pinned_bytes")
        vram_bytes = _positive_int(item["vram_bytes"], f"{label}[{index}].vram_bytes")
        kv_blocks = _exact(
            item["kv_blocks"], {"free", "reserved", "active"}, f"{label}[{index}].kv_blocks"
        )
        normalized_kv = {
            name: _nonnegative_int(kv_blocks[name], f"{label}[{index}].kv_blocks.{name}")
            for name in ("free", "reserved", "active")
        }
        if sum(normalized_kv.values()) == 0:
            _fail("invalid-interval-observations", f"{label} must expose a nonempty KV block inventory")

        request_states = _exact(
            item["request_states"],
            {"active", "completed", "cancelled", "capacity_rejections"},
            f"{label}[{index}].request_states",
        )
        normalized_states = {
            name: _nonnegative_int(request_states[name], f"{label}[{index}].request_states.{name}")
            for name in ("active", "completed", "cancelled", "capacity_rejections")
        }
        if (
            normalized_states["active"]
            + normalized_states["completed"]
            + normalized_states["cancelled"]
            + normalized_states["capacity_rejections"]
            > attempted
        ):
            _fail("request-state-accounting-mismatch", f"{label} records more requests than its scenario attempted")
        terminal_total = _nonnegative_int(
            item["terminal_events_total"], f"{label}[{index}].terminal_events_total"
        )
        if terminal_total != (
            normalized_states["completed"]
            + normalized_states["cancelled"]
            + normalized_states["capacity_rejections"]
        ):
            _fail("request-state-accounting-mismatch", f"{label} terminal counter does not match request states")
        if terminal_total > attempted:
            _fail("request-state-accounting-mismatch", f"{label} emits too many terminal events")
        if (
            normalized_states["completed"] < previous_completed
            or normalized_states["cancelled"] < previous_cancelled
            or normalized_states["capacity_rejections"] < previous_rejected
            or terminal_total < previous_terminal
        ):
            _fail("interval-observation-order-mismatch", f"{label} cumulative request counters regressed")
        previous_elapsed = elapsed
        previous_completed = normalized_states["completed"]
        previous_cancelled = normalized_states["cancelled"]
        previous_rejected = normalized_states["capacity_rejections"]
        previous_terminal = terminal_total
        normalized.append(
            {
                "elapsed_seconds": elapsed,
                "rss_bytes": rss_bytes,
                "pinned_bytes": pinned_bytes,
                "vram_bytes": vram_bytes,
                "kv_blocks": normalized_kv,
                "request_states": normalized_states,
                "terminal_events_total": terminal_total,
            }
        )

    last = normalized[-1]
    if last["elapsed_seconds"] != observed_duration_seconds:
        _fail("interval-observation-coverage-missing", f"{label} must cover the scenario's final observed second")
    last_states = last["request_states"]
    if (
        last_states["active"] != 0
        or last_states["completed"] != completed
        or last_states["cancelled"] != cancelled
        or last_states["capacity_rejections"] != capacity_rejections
        or last["terminal_events_total"] != terminal_events
    ):
        _fail("request-state-accounting-mismatch", f"{label} final counters do not equal the scenario outcome")
    return normalized


def _kv_utilization_percent(kv_blocks: dict[str, int]) -> int:
    """Use the raw free/reserved/active inventory for integer KV utilisation."""

    total = kv_blocks["free"] + kv_blocks["reserved"] + kv_blocks["active"]
    # `_validate_interval_observations` closes this inventory and rejects zero
    # totals before this helper is reached.
    assert total > 0
    return kv_blocks["active"] * 100 // total


def _validate_scenario_results(value: Any, contract: ScenarioContract) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or len(value) != len(contract.scenarios) or len(value) > MAX_SCENARIO_RESULTS:
        _fail("invalid-soak-v2-trace", "scenario trace requires the exact full-duration inventory")
    normalized: list[dict[str, Any]] = []
    previous_end = -1
    for index, (raw, expected) in enumerate(zip(value, contract.scenarios)):
        label = f"soak v2 scenario trace.results[{index}]"
        item = _exact(
            raw,
            {
                "scenario_id",
                "started_monotonic_ns",
                "ended_monotonic_ns",
                "observed_duration_seconds",
                "attempted_requests",
                "completed_requests",
                "cancelled_requests",
                "capacity_rejections",
                "terminal_events",
                "max_kv_utilization_percent",
                "exact_backend_fallbacks",
                "backend_events",
                "model_lifecycle_cycles",
                "interval_observations",
            },
            label,
        )
        scenario_id = qualification._string(item["scenario_id"], f"{label}.scenario_id")
        if scenario_id != expected["id"]:
            _fail("scenario-inventory-mismatch", f"{label} does not follow the source-controlled order")
        started = _positive_int(item["started_monotonic_ns"], f"{label}.started_monotonic_ns")
        ended = _positive_int(item["ended_monotonic_ns"], f"{label}.ended_monotonic_ns")
        observed_duration = _positive_int(
            item["observed_duration_seconds"], f"{label}.observed_duration_seconds"
        )
        if started <= previous_end or ended <= started:
            _fail("scenario-time-order-mismatch", f"{label} overlaps or reverses the previous scenario")
        if observed_duration < expected["duration_seconds"] or (
            ended - started < observed_duration * 1_000_000_000
        ):
            _fail("truncated-soak-v2-scenario", f"{label} does not span its reviewed full duration")
        previous_end = ended
        attempted = _positive_int(item["attempted_requests"], f"{label}.attempted_requests")
        completed = _nonnegative_int(item["completed_requests"], f"{label}.completed_requests")
        cancelled = _nonnegative_int(item["cancelled_requests"], f"{label}.cancelled_requests")
        capacity_rejections = _nonnegative_int(
            item["capacity_rejections"], f"{label}.capacity_rejections"
        )
        terminal_events = _positive_int(item["terminal_events"], f"{label}.terminal_events")
        if attempted < expected["minimum_requests"]:
            _fail("insufficient-soak-v2-requests", f"{label} did not reach its reviewed request minimum")
        if completed + cancelled + capacity_rejections != attempted:
            _fail("request-terminal-accounting-mismatch", f"{label} outcomes do not account for every request")
        if terminal_events != attempted:
            _fail("request-terminal-accounting-mismatch", f"{label} must emit exactly one terminal event per request")
        if attempted * expected["cancellation_percent"] % 100 or (
            cancelled * 100 != attempted * expected["cancellation_percent"]
        ):
            _fail("cancellation-rate-mismatch", f"{label} does not prove its exact cancellation arm")
        if capacity_rejections < expected["minimum_capacity_rejections"]:
            _fail("kv-capacity-boundary-missing", f"{label} lacks required capacity rejections")
        if expected["minimum_capacity_rejections"] == 0 and capacity_rejections != 0:
            _fail("unexpected-capacity-rejection", f"{label} has unreviewed capacity rejections")
        utilization = _nonnegative_int(
            item["max_kv_utilization_percent"], f"{label}.max_kv_utilization_percent"
        )
        fallback_count = _nonnegative_int(
            item["exact_backend_fallbacks"], f"{label}.exact_backend_fallbacks"
        )
        backend_events = item["backend_events"]
        if not isinstance(backend_events, list) or any(
            not isinstance(event, str) or not event for event in backend_events
        ):
            _fail("invalid-soak-v2-trace", f"{label}.backend_events must be text")
        if expected["minimum_exact_backend_fallbacks"] == 0:
            if fallback_count != 0 or backend_events != []:
                _fail("unexpected-exact-backend-fallback", f"{label} has unreviewed fallback evidence")
        elif (
            fallback_count != expected["minimum_exact_backend_fallbacks"]
            or backend_events != expected["required_backend_events"]
        ):
            _fail("exact-backend-fallback-missing", f"{label} does not prove the required fallback sequence")
        lifecycle = _validate_lifecycle_cycles(
            item["model_lifecycle_cycles"],
            expected["minimum_model_load_unload_cycles"],
            f"{label}.model_lifecycle_cycles",
        )
        interval_observations = _validate_interval_observations(
            item["interval_observations"],
            attempted=attempted,
            completed=completed,
            cancelled=cancelled,
            capacity_rejections=capacity_rejections,
            terminal_events=terminal_events,
            observed_duration_seconds=observed_duration,
            maximum_interval_seconds=contract.maximum_interval_seconds,
            label=f"{label}.interval_observations",
        )
        observed_peak_utilization = max(
            _kv_utilization_percent(observation["kv_blocks"])
            for observation in interval_observations
        )
        if utilization != observed_peak_utilization:
            _fail(
                "kv-utilization-evidence-mismatch",
                f"{label} max KV utilisation does not equal its raw block inventory",
            )
        if not expected["minimum_kv_utilization_percent"] <= utilization <= expected[
            "maximum_kv_utilization_percent"
        ]:
            _fail("kv-utilization-mismatch", f"{label} did not reach its reviewed KV arm")
        normalized.append(
            {
                "scenario_id": scenario_id,
                "started_monotonic_ns": started,
                "ended_monotonic_ns": ended,
                "observed_duration_seconds": observed_duration,
                "attempted_requests": attempted,
                "completed_requests": completed,
                "cancelled_requests": cancelled,
                "capacity_rejections": capacity_rejections,
                "terminal_events": terminal_events,
                "max_kv_utilization_percent": utilization,
                "exact_backend_fallbacks": fallback_count,
                "backend_events": list(backend_events),
                "model_lifecycle_cycles": lifecycle,
                "interval_observations": interval_observations,
            }
        )
    if sum(item["observed_duration_seconds"] for item in normalized) < contract.total_duration_seconds:
        _fail("truncated-soak-v2-run", "scenario trace does not cover the full reviewed duration")
    return tuple(normalized)


def validate_trace(document: dict[str, Any], contract: ScenarioContract) -> SoakV2Trace:
    row = _exact(
        document,
        {
            "schema_version",
            "candidate_id",
            "bindings",
            "scenario_contract",
            "gate_e_soak_raw_sha256",
            "scenario_results",
        },
        "soak v2 scenario trace",
    )
    if row["schema_version"] != TRACE_VERSION:
        _fail("unsupported-soak-v2-trace-version", "scenario trace schema_version is unsupported")
    contract_descriptor = _descriptor(
        row["scenario_contract"], "soak v2 scenario trace.scenario_contract"
    )
    if contract_descriptor != contract.descriptor:
        _fail("source-contract-binding-mismatch", "scenario trace does not bind the reviewed source contract")
    return SoakV2Trace(
        candidate_id=_candidate_id(row["candidate_id"], "soak v2 scenario trace.candidate_id"),
        bindings=_bindings(row["bindings"], "soak v2 scenario trace.bindings"),
        scenario_contract=contract_descriptor,
        gate_e_soak_raw_sha256=_sha256(
            row["gate_e_soak_raw_sha256"], "soak v2 scenario trace.gate_e_soak_raw_sha256"
        ),
        scenario_results=_validate_scenario_results(row["scenario_results"], contract),
    )


def _write_all(descriptor: int, contents: bytes, label: str) -> None:
    offset = 0
    while offset < len(contents):
        try:
            written = os.write(descriptor, contents[offset:])
        except OSError as error:
            _fail("snapshot-write-failed", f"{label} could not be snapshotted: {error}")
        if written <= 0:
            _fail("snapshot-write-failed", f"{label} could not be snapshotted")
        offset += written


def _snapshot_evidence(
    evidence_root: Path,
    relative: str,
    *,
    expected_sha256: str | None,
    label: str,
    destination_root: Path,
    sequence: int,
    maximum_bytes: int,
    reserved_paths: set[str],
    seen_paths: set[str],
    seen_file_ids: set[tuple[int, int]],
) -> tuple[Path, str]:
    """No-follow snapshot one evidence input and reject textual/hard-link aliases."""

    relative = qualification._relative_path(relative, f"{label}.path")
    if relative in reserved_paths:
        _fail("reserved-output-path-collision", f"{label} reuses a freeze-declared output")
    if relative in seen_paths:
        _fail("duplicate-evidence-path", f"{label} reuses another evidence path")
    seen_paths.add(relative)
    pure = PurePosixPath(relative)
    root_fd = current_fd = file_fd = output_fd = -1
    try:
        try:
            root_before = evidence_root.lstat()
        except OSError as error:
            _fail("missing-evidence-root", f"cannot inspect evidence root: {error}")
        if not stat.S_ISDIR(root_before.st_mode):
            _fail("unsafe-evidence-root", "evidence root must be a real directory")
        root_fd = os.open(evidence_root, _OPEN_DIRECTORY)
        root_after = os.fstat(root_fd)
        if (root_before.st_dev, root_before.st_ino) != (root_after.st_dev, root_after.st_ino):
            _fail("raced-evidence-root", "evidence root changed while it was opened")
        current_fd = root_fd
        for component in pure.parts[:-1]:
            next_fd = os.open(component, _OPEN_DIRECTORY, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(pure.parts[-1], _OPEN_COMMON, dir_fd=current_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            _fail("unsafe-evidence-path", f"{label} must be a regular file")
        if before.st_size > maximum_bytes:
            _fail("input-too-large", f"{label} exceeds its {maximum_bytes}-byte bound")
        file_id = (before.st_dev, before.st_ino)
        if file_id in seen_file_ids:
            _fail("hard-link-evidence-alias", f"{label} aliases another evidence file")
        seen_file_ids.add(file_id)
        snapshot = destination_root / f"{sequence:02d}-{pure.name}"
        output_fd = os.open(
            snapshot,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        remaining = before.st_size
        digest = hashlib.sha256()
        while remaining:
            chunk = os.read(file_fd, min(1024 * 1024, remaining))
            if not chunk:
                _fail("truncated-evidence", f"{label} changed while it was snapshotted")
            digest.update(chunk)
            _write_all(output_fd, chunk, label)
            remaining -= len(chunk)
        if os.read(file_fd, 1):
            _fail("mutated-evidence", f"{label} grew while it was snapshotted")
        os.fsync(output_fd)
        after = os.fstat(file_fd)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            _fail("mutated-evidence", f"{label} changed while it was snapshotted")
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            _fail("evidence-sha256-mismatch", f"{label} does not match its declared SHA-256")
        return snapshot, actual_sha256
    except SoakV2ReceiptError:
        raise
    except OSError as error:
        _fail("unsafe-evidence-path", f"{label} cannot be opened safely: {error}")
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        if file_fd >= 0:
            os.close(file_fd)
        if current_fd >= 0 and current_fd != root_fd:
            os.close(current_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _parse_snapshot_json(snapshot: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = snapshot.read_bytes()
    except OSError as error:
        _fail("snapshot-read-failed", f"cannot read {label} snapshot: {error}")
    document = qualification._parse_document(raw, label)
    if raw != qualification.canonical_json_bytes(document):
        _fail("noncanonical-evidence-json", f"{label} must use exact canonical JSON bytes")
    return raw, document


def _expected_bindings(
    frozen: qualification.FrozenCandidate,
    *,
    freeze_sha256: str,
    base_report_sha256: str,
) -> dict[str, str]:
    return {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }


def _validate_gate_e_descriptors(
    receipt: SoakV2Receipt,
    base_report: dict[str, Any],
) -> None:
    evidence_hashes = base_report["bindings"]["evidence_sha256"]
    expected = {
        "report": evidence_hashes["reliability_soak"],
        "raw_archive": evidence_hashes["reliability_soak_raw"],
        "correctness_golden": evidence_hashes["python_free_e2e_correctness_golden_raw"],
        "native_correctness_report": evidence_hashes["native_correctness_report"],
    }
    for name, expected_sha256 in expected.items():
        if getattr(receipt.gate_e_soak, name).sha256 != expected_sha256:
            _incomparable(f"soak v2 receipt {name} differs from replayed Gate E evidence")


def _validate_replayed_gate_e_soak(report: Any) -> dict[str, Any]:
    row = _exact(
        report,
        {"schema_version", "status", "passed", "bindings", "scenario_summaries", "observations", "checks", "errors"},
        "replayed Gate E soak report",
    )
    reliability_soak = qualification.release_candidate.reliability_soak
    if (
        row["schema_version"] != reliability_soak.REPORT_VERSION
        or row["status"] != "passed"
        or row["passed"] is not True
        or row["errors"] != []
    ):
        _fail("gate-e-soak-replay-failed", "replayed Gate E soak did not pass")
    bindings = _exact(
        row["bindings"],
        {
            "contract_id",
            "reviewed_manifest_template_canonical_sha256",
            "manifest_sha256",
            "binding_sha256",
            "trusted_correctness",
            "runtime_provenance",
            "source",
        },
        "replayed Gate E soak report.bindings",
    )
    if bindings["contract_id"] != qualification.release_candidate.SOAK_CONTRACT_ID:
        _fail("gate-e-soak-replay-failed", "replayed Gate E soak contract drifted")
    summaries = row["scenario_summaries"]
    if not isinstance(summaries, list) or [item.get("scenario_id") for item in summaries] != [
        item[0] for item in V1_SCENARIOS
    ]:
        _fail("gate-e-soak-replay-failed", "replayed Gate E soak lacks its fixed v1 scenario inventory")
    return row


def _replay_gate_e_archive(
    raw_archive: Path,
    correctness_golden: Path,
    native_correctness_report: Path,
    *,
    expected_archive_sha256: str,
    expected_report: dict[str, Any],
) -> None:
    reliability_soak = qualification.release_candidate.reliability_soak
    try:
        replay = reliability_soak.replay_raw_evidence_archive(
            raw_archive,
            correctness_golden=correctness_golden,
            native_correctness_report=native_correctness_report,
        )
    except (reliability_soak.InputError, OSError) as error:
        _fail("gate-e-soak-replay-failed", f"cannot replay Gate E raw archive: {error}")
    expected_keys = {
        "report",
        "archive_sha256",
        *{f"{name}_sha256" for name in reliability_soak.RAW_ARCHIVE_PAYLOADS},
    }
    if not isinstance(replay, dict) or set(replay) != expected_keys:
        _fail("gate-e-soak-replay-failed", "raw archive replay returned an unclosed binding inventory")
    for name in expected_keys - {"report"}:
        _sha256(replay[name], f"Gate E raw archive replay.{name}")
    if replay["archive_sha256"] != expected_archive_sha256:
        _fail("gate-e-soak-archive-hash-mismatch", "raw archive replay hash drifted")
    replayed_report = _validate_replayed_gate_e_soak(replay["report"])
    if replayed_report != expected_report:
        _fail("gate-e-soak-report-replay-mismatch", "submitted Gate E soak report differs from raw replay")


def _empty_report(contract: ScenarioContract) -> dict[str, Any]:
    return {
        "schema_version": CHECK_REPORT_VERSION,
        "status": "failed",
        "passed": False,
        "candidate_id": None,
        "freeze_sha256": None,
        "base_release_candidate_report": None,
        "scenario_contract": {
            "path": contract.descriptor.path,
            "sha256": contract.descriptor.sha256,
        },
        "receipt": None,
        "bindings": None,
        "gate_e_soak_report": None,
        "gate_e_soak_raw_archive": None,
        "correctness_golden": None,
        "native_correctness_report": None,
        "scenario_trace": None,
        "scenario_results": [],
        "checks": [],
        "reason_codes": [],
    }


def evaluate(
    freeze_path: Path,
    evidence_root: Path,
    receipt_path: Path | str,
    *,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    """Replay the C02 soak-v2 receipt without running a remote soak."""

    try:
        contract = _load_contract()
    except (OSError, qualification.QualificationError) as error:
        fallback = {
            "schema_version": CHECK_REPORT_VERSION,
            "status": "failed",
            "passed": False,
            "candidate_id": None,
            "freeze_sha256": None,
            "base_release_candidate_report": None,
            "scenario_contract": None,
            "receipt": None,
            "bindings": None,
            "gate_e_soak_report": None,
            "gate_e_soak_raw_archive": None,
            "correctness_golden": None,
            "native_correctness_report": None,
            "scenario_trace": None,
            "scenario_results": [],
            "checks": [],
            "reason_codes": [getattr(error, "reason_code", "invalid-input")],
        }
        return fallback
    report = _empty_report(contract)
    try:
        trusted_freeze_sha256 = _sha256(expected_freeze_sha256, "--expected-freeze-sha256")
        freeze_raw = qualification._read_regular_path(freeze_path, "freeze manifest")
        freeze_sha256 = hashlib.sha256(freeze_raw).hexdigest()
        report["freeze_sha256"] = freeze_sha256
        if freeze_sha256 != trusted_freeze_sha256:
            _fail("candidate-sha-mismatch", "freeze manifest SHA-256 differs from trusted input")
        frozen = qualification._validate_freeze(
            qualification._parse_document(freeze_raw, "freeze manifest")
        )
        report["candidate_id"] = frozen.candidate_id
        reserved_paths = {
            frozen.final_manifest.path,
            frozen.final_report.path,
            *(descriptor.path for descriptor in frozen.receipts.values()),
        }

        base_raw, base_report_sha256 = qualification.revalidate_base_release_candidate(
            frozen, freeze_sha256, evidence_root
        )
        if hashlib.sha256(base_raw).hexdigest() != base_report_sha256:
            _fail("base-report-replay-digest-mismatch", "Gate E replay returned inconsistent bytes/digest")
        base_report = qualification._parse_document(base_raw, "final release candidate report")
        qualification._validate_base_report_shape(base_report, frozen)
        report["base_release_candidate_report"] = {
            "path": frozen.final_report.path,
            "sha256": base_report_sha256,
        }

        receipt_relative = qualification._relative_path(str(receipt_path), "soak v2 receipt path")
        if receipt_relative in reserved_paths:
            _fail(
                "reserved-output-path-collision",
                "raw soak-v2 receipt must not replace a frozen semantic output",
            )
        with tempfile.TemporaryDirectory(prefix="riley-soak-v2-replay-") as temporary:
            snapshots = Path(temporary)
            seen_paths: set[str] = set()
            seen_file_ids: set[tuple[int, int]] = set()
            receipt_snapshot, receipt_sha256 = _snapshot_evidence(
                evidence_root,
                receipt_relative,
                expected_sha256=None,
                label="soak v2 receipt",
                destination_root=snapshots,
                sequence=1,
                maximum_bytes=MAX_JSON_BYTES,
                reserved_paths=reserved_paths,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            _, receipt_document = _parse_snapshot_json(receipt_snapshot, "soak v2 receipt")
            receipt = validate_receipt(receipt_document)
            if receipt.candidate_id != frozen.candidate_id:
                _incomparable("soak v2 receipt belongs to another candidate")
            expected_bindings = _expected_bindings(
                frozen, freeze_sha256=freeze_sha256, base_report_sha256=base_report_sha256
            )
            if receipt.bindings != expected_bindings:
                _incomparable("soak v2 receipt immutable stable-default bindings drifted")
            if receipt.scenario_contract != contract.descriptor:
                _fail("source-contract-binding-mismatch", "soak v2 receipt does not bind the reviewed source contract")
            _validate_gate_e_descriptors(receipt, base_report)
            report["receipt"] = {"path": receipt_relative, "sha256": receipt_sha256}
            report["bindings"] = receipt.bindings

            gate_report_snapshot, _ = _snapshot_evidence(
                evidence_root,
                receipt.gate_e_soak.report.path,
                expected_sha256=receipt.gate_e_soak.report.sha256,
                label="Gate E soak report",
                destination_root=snapshots,
                sequence=2,
                maximum_bytes=MAX_JSON_BYTES,
                reserved_paths=reserved_paths,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            raw_archive_snapshot, _ = _snapshot_evidence(
                evidence_root,
                receipt.gate_e_soak.raw_archive.path,
                expected_sha256=receipt.gate_e_soak.raw_archive.sha256,
                label="Gate E soak raw archive",
                destination_root=snapshots,
                sequence=3,
                maximum_bytes=qualification.release_candidate.reliability_soak.MAX_RAW_ARCHIVE_BYTES,
                reserved_paths=reserved_paths,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            golden_snapshot, _ = _snapshot_evidence(
                evidence_root,
                receipt.gate_e_soak.correctness_golden.path,
                expected_sha256=receipt.gate_e_soak.correctness_golden.sha256,
                label="Gate E soak correctness golden",
                destination_root=snapshots,
                sequence=4,
                maximum_bytes=qualification.release_candidate.reliability_soak.MAX_CORRECTNESS_GOLDEN_BYTES,
                reserved_paths=reserved_paths,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            native_snapshot, _ = _snapshot_evidence(
                evidence_root,
                receipt.gate_e_soak.native_correctness_report.path,
                expected_sha256=receipt.gate_e_soak.native_correctness_report.sha256,
                label="Gate E native correctness report",
                destination_root=snapshots,
                sequence=5,
                maximum_bytes=qualification.release_candidate.reliability_soak.MAX_NATIVE_CORRECTNESS_REPORT_BYTES,
                reserved_paths=reserved_paths,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            trace_snapshot, trace_sha256 = _snapshot_evidence(
                evidence_root,
                receipt.scenario_trace.path,
                expected_sha256=receipt.scenario_trace.sha256,
                label="soak v2 scenario trace",
                destination_root=snapshots,
                sequence=6,
                maximum_bytes=MAX_JSON_BYTES,
                reserved_paths=reserved_paths,
                seen_paths=seen_paths,
                seen_file_ids=seen_file_ids,
            )
            _, gate_report_document = _parse_snapshot_json(gate_report_snapshot, "Gate E soak report")
            _, trace_document = _parse_snapshot_json(trace_snapshot, "soak v2 scenario trace")
            trace = validate_trace(trace_document, contract)
            if trace.candidate_id != frozen.candidate_id or trace.bindings != expected_bindings:
                _incomparable("soak v2 scenario trace immutable bindings drifted")
            if trace.gate_e_soak_raw_sha256 != receipt.gate_e_soak.raw_archive.sha256:
                _incomparable("soak v2 scenario trace binds another Gate E raw archive")
            _replay_gate_e_archive(
                raw_archive_snapshot,
                golden_snapshot,
                native_snapshot,
                expected_archive_sha256=receipt.gate_e_soak.raw_archive.sha256,
                expected_report=gate_report_document,
            )
            report.update(
                {
                    "gate_e_soak_report": {
                        "path": receipt.gate_e_soak.report.path,
                        "sha256": receipt.gate_e_soak.report.sha256,
                    },
                    "gate_e_soak_raw_archive": {
                        "path": receipt.gate_e_soak.raw_archive.path,
                        "sha256": receipt.gate_e_soak.raw_archive.sha256,
                    },
                    "correctness_golden": {
                        "path": receipt.gate_e_soak.correctness_golden.path,
                        "sha256": receipt.gate_e_soak.correctness_golden.sha256,
                    },
                    "native_correctness_report": {
                        "path": receipt.gate_e_soak.native_correctness_report.path,
                        "sha256": receipt.gate_e_soak.native_correctness_report.sha256,
                    },
                    "scenario_trace": {
                        "path": receipt.scenario_trace.path,
                        "sha256": trace_sha256,
                    },
                    "scenario_results": list(trace.scenario_results),
                    "status": "passed",
                    "passed": True,
                    "checks": [{"name": name, "passed": True} for name in CHECK_NAMES],
                }
            )
    except qualification.IncomparableError as error:
        report["status"] = "incomparable"
        report["reason_codes"] = [getattr(error, "reason_code", "incomparable-binding")]
    except qualification.GateFailure as error:
        report["reason_codes"] = [getattr(error, "reason_code", "gate-failed")]
    except (OSError, qualification.QualificationError) as error:
        report["reason_codes"] = [getattr(error, "reason_code", "invalid-input")]
    return report


def validate_check_report(document: dict[str, Any]) -> SoakV2CheckReport:
    """Parse a passed semantic report before an outer exact replay."""

    contract = _load_contract()
    row = _exact(
        document,
        {
            "schema_version",
            "status",
            "passed",
            "candidate_id",
            "freeze_sha256",
            "base_release_candidate_report",
            "scenario_contract",
            "receipt",
            "bindings",
            "gate_e_soak_report",
            "gate_e_soak_raw_archive",
            "correctness_golden",
            "native_correctness_report",
            "scenario_trace",
            "scenario_results",
            "checks",
            "reason_codes",
        },
        "soak v2 check report",
    )
    if row["schema_version"] != CHECK_REPORT_VERSION:
        _fail("unsupported-soak-v2-check-report-version", "soak v2 check report schema_version is unsupported")
    if row["status"] != "passed" or row["passed"] is not True or row["reason_codes"] != []:
        _fail("soak-v2-check-not-passed", "soak v2 check report must be a clean passed report")
    contract_descriptor = _descriptor(row["scenario_contract"], "soak v2 check report.scenario_contract")
    if contract_descriptor != contract.descriptor:
        _fail("source-contract-binding-mismatch", "soak v2 check report does not bind the reviewed source contract")
    checks = row["checks"]
    if not isinstance(checks, list) or len(checks) != len(CHECK_NAMES):
        _fail("invalid-soak-v2-check-report", "soak v2 check report has an invalid check inventory")
    names: list[str] = []
    for index, value in enumerate(checks):
        item = _exact(value, {"name", "passed"}, f"soak v2 check report.checks[{index}]")
        if item["passed"] is not True:
            _fail("soak-v2-check-not-passed", f"soak v2 check {item['name']!r} did not pass")
        names.append(qualification._string(item["name"], f"soak v2 check report.checks[{index}].name"))
    if tuple(names) != CHECK_NAMES:
        _fail("invalid-soak-v2-check-report", "soak v2 check report check inventory drifted")
    base = _descriptor(
        row["base_release_candidate_report"], "soak v2 check report.base_release_candidate_report"
    )
    receipt = _descriptor(row["receipt"], "soak v2 check report.receipt")
    gate_e_soak = GateESoakInputs(
        report=_descriptor(row["gate_e_soak_report"], "soak v2 check report.gate_e_soak_report"),
        raw_archive=_descriptor(
            row["gate_e_soak_raw_archive"], "soak v2 check report.gate_e_soak_raw_archive"
        ),
        correctness_golden=_descriptor(
            row["correctness_golden"], "soak v2 check report.correctness_golden"
        ),
        native_correctness_report=_descriptor(
            row["native_correctness_report"], "soak v2 check report.native_correctness_report"
        ),
    )
    trace = _descriptor(row["scenario_trace"], "soak v2 check report.scenario_trace")
    _distinct_paths(
        (
            base.path,
            receipt.path,
            gate_e_soak.report.path,
            gate_e_soak.raw_archive.path,
            gate_e_soak.correctness_golden.path,
            gate_e_soak.native_correctness_report.path,
            trace.path,
        ),
        "soak v2 check report evidence",
    )
    return SoakV2CheckReport(
        candidate_id=_candidate_id(row["candidate_id"], "soak v2 check report.candidate_id"),
        freeze_sha256=_sha256(row["freeze_sha256"], "soak v2 check report.freeze_sha256"),
        base_release_candidate_report=base,
        scenario_contract=contract_descriptor,
        receipt=receipt,
        bindings=_bindings(row["bindings"], "soak v2 check report.bindings"),
        gate_e_soak=gate_e_soak,
        scenario_trace=trace,
        scenario_results=_validate_scenario_results(row["scenario_results"], contract),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, help="raw soak-v2 receipt relative to --evidence-root")
    parser.add_argument("--report", type=Path, help="create-only semantic check report output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.freeze,
        args.evidence_root,
        args.receipt,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            qualification._write_create_only(args.report, report)
        except qualification.QualificationError as error:
            print(str(error), file=sys.stderr)
            return 2
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
