#!/usr/bin/env python3
"""Fail-closed M4/M5 checker for a C01 competitive campaign.

The checker consumes an immutable plan plus append-only JSONL observations.  A
malformed, incomplete, environment-mixed, or development-only campaign is
``incomparable``; it never becomes a favorable latency statistic.  A completed
comparable campaign with a request failure or token mismatch is ``failed``.
Only ``passed``, ``partial-win``, ``failed``, and ``incomparable`` can be
emitted.
"""

from __future__ import annotations

import argparse
import stat
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from competitive_common import (
    PLAN_SCHEMA_VERSION,
    RAW_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    ComparabilityError,
    ContractError,
    REQUEST_WORKLOAD_FIELDS,
    WORKLOAD_EXECUTION_FIELDS,
    build_workload_identity,
    canonical_json_bytes,
    create_only_write,
    expect_keys,
    finite_number,
    geometric_mean,
    load_json,
    load_jsonl,
    load_preflight_receipt,
    matrix_cells,
    nearest_rank,
    require_campaign_artifact_path,
    r7,
    request_sets_by_cell,
    request_workload_identity,
    sha256_bytes,
    sha256_file,
    validate_contract,
    validate_identifier,
    validate_lane,
    validate_matrix,
    validate_request_manifest,
    validate_sha256,
    require_canonical_contract,
    verify_campaign_lane_binding,
    verify_campaign_matrix_binding,
    verify_campaign_request_binding,
    verify_canonical_preflight_script_receipt,
    workload_execution_receipt,
)
from materialize_lane import verify_campaign_lane_binding_value
from raw_journal import JOURNAL_FIELDS, validate_append_only_chain


SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parents[2]
FINAL_STATUSES = {"passed", "partial-win", "failed", "incomparable"}

RAW_REQUIRED_FIELDS = (
    "schema_version",
    "campaign_id",
    "campaign_plan_sha256",
    "invocation_id",
    "lane_id",
    "role",
    "execution_id",
    "cell_id",
    "run_index",
    "order",
    "position",
    "measurement_mode",
    "request_manifest_sha256",
    "workload_sha256",
    "workload",
    "recorded_at_utc",
    "source",
    "environment",
    "phase",
    "status",
    "failure_reason",
    "metrics",
    "requests",
    *JOURNAL_FIELDS,
)
RAW_OPTIONAL_FIELDS = ("preflight_receipt_sha256",)
SUCCESS_METRIC_REQUIRED_FIELDS = (
    "output_tokens_per_second",
    "slo_goodput_tokens_per_second",
    "peak_vram_bytes",
    "usable_kv_bytes",
)
SUCCESS_METRIC_OPTIONAL_FIELDS = (
    "cpu_utilization_percent",
    "gpu_utilization_percent",
    "scheduler_cpu_ns",
    "kernel_launches_per_token",
)
REQUEST_REQUIRED_FIELDS = (
    "request_id",
    "prompt_token_ids_sha256",
    "prompt_tokens",
    "generated_token_ids_sha256",
    "status",
    "failure_reason",
    "requested_output_tokens",
    "sampling",
    "seed",
    "eos_policy",
    "cache_policy",
    "arrival_schedule_id",
    "generated_tokens",
    "ttft_ms",
    "tpot_ms",
    "end_to_end_ms",
    "terminal_event_count",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a JSON array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _environment_scalar(value: Any, label: str) -> str | int | float:
    if isinstance(value, str):
        return _nonempty_string(value, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a non-empty string or finite number")
    finite_number(value, label)
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _relative_path(root: Path, value: Any, label: str) -> Path:
    raw = _nonempty_string(value, label)
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{label} must stay inside repository root") from error
    return path


def current_source_receipt(root: Path) -> Mapping[str, Any]:
    """Read the checkout state at claim time, not just plan creation time."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ComparabilityError(f"cannot inspect current Git source state: {error}") from error
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ComparabilityError("current Git HEAD is not a full lowercase 40-hex revision")
    return {"git_revision": revision, "git_dirty": dirty}


def _available_pinned_lane(lane: Mapping[str, Any], label: str) -> None:
    """Check the current lane file, not a self-declared plan receipt."""

    engine = _mapping(lane.get("engine"), f"{label}.engine")
    command = _mapping(lane.get("command"), f"{label}.command")
    if lane.get("availability") != "available":
        raise ComparabilityError(f"{label} is not actually available")
    if command.get("status") != "available":
        raise ComparabilityError(f"{label} command is not actually available")
    if engine.get("pin_status") != "pinned":
        raise ComparabilityError(f"{label} is not pinned")
    for field in ("version", "revision", "dependency_lock_sha256"):
        value = engine.get(field)
        if not isinstance(value, str) or not value:
            raise ComparabilityError(f"{label} lacks an immutable engine {field}")


def _materialized_lane_claim_reasons(
    lane: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    root: Path,
    role: str,
) -> list[str]:
    """Return fail-closed reasons when a lane is not executable claim evidence."""

    reasons: list[str] = []
    label = f"{role} lane {lane.get('lane_id')!r}"
    command = _mapping(lane.get("command"), f"{label}.command")
    if command.get("required_placeholders") != []:
        reasons.append(f"{label} command is not fully materialized")
    argv = command.get("argv")
    if not isinstance(argv, list) or any("{" in str(token) or "}" in str(token) for token in argv):
        reasons.append(f"{label} command contains an unresolved placeholder")

    materialization = lane.get("materialization")
    if not isinstance(materialization, Mapping):
        reasons.append(f"{label} lacks a materialization receipt")
        return reasons
    if materialization.get("campaign_id") != plan["campaign_id"]:
        reasons.append(f"{label} materialization campaign differs from plan")
    source = _mapping(plan["source"], "plan.source")
    if materialization.get("source_git_revision") != source["git_revision"]:
        reasons.append(f"{label} materialization source differs from plan")
    try:
        verify_campaign_lane_binding_value(root=root, lane=lane)
    except ContractError as error:
        reasons.append(f"{label} materialization receipt is not reproducible: {error}")
    return reasons


def _load_claim_lanes(
    plan: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Mapping[str, Any]]:
    """Load both immutable materialized lanes for raw environment binding."""

    lanes = _mapping(plan["lanes"], "plan.lanes")
    loaded: dict[str, Mapping[str, Any]] = {}
    for role in ("riley", "competitor"):
        receipt = _mapping(lanes[role], f"plan.lanes.{role}")
        path = require_campaign_artifact_path(
            root,
            _relative_path(root, receipt["path"], f"plan.lanes.{role}.path"),
            f"plan.lanes.{role}.path",
        )
        if sha256_file(path) != receipt["sha256"]:
            raise ComparabilityError(f"{role} lane hash drifted after plan creation")
        lane = validate_lane(load_json(path), str(path))
        verify_campaign_lane_binding(root, path, lane)
        try:
            _available_pinned_lane(lane, f"{role} lane {lane['lane_id']!r}")
        except ComparabilityError as error:
            raise ComparabilityError(str(error)) from error
        reasons = _materialized_lane_claim_reasons(lane, plan=plan, root=root, role=role)
        if reasons:
            raise ComparabilityError("; ".join(reasons))
        loaded[role] = lane
    return loaded


def _validate_source(value: Any, label: str) -> Mapping[str, Any]:
    source = _mapping(value, label)
    expect_keys(source, label, required=("git_revision", "git_dirty", "development_only"))
    revision = _nonempty_string(source["git_revision"], f"{label}.git_revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ContractError(f"{label}.git_revision must be full lowercase 40-hex")
    if not isinstance(source["git_dirty"], bool):
        raise ContractError(f"{label}.git_dirty must be boolean")
    if not isinstance(source["development_only"], bool):
        raise ContractError(f"{label}.development_only must be boolean")
    if source["git_dirty"] != source["development_only"]:
        raise ContractError(f"{label}.development_only must equal git_dirty")
    return source


def _validate_plan_workloads(
    plan: Mapping[str, Any],
    *,
    cells_by_id: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Re-derive every closed workload receipt carried by a plan.

    A plan hash alone cannot prove that an adapter was given the intended
    prompt shape, sampling, arrival behavior, or serving policy.  The plan
    therefore carries the complete per-cell workload, and the checker derives
    the same value again from its independently hashed matrix and request
    manifest before accepting it.
    """

    raw_workloads = _array(plan["workloads"], "plan.workloads")
    if not raw_workloads:
        raise ContractError("plan.workloads must not be empty")
    requests_by_cell = request_sets_by_cell(manifest)
    if set(requests_by_cell) != set(cells_by_id):
        raise ComparabilityError("plan workload/request-manifest cell coverage drift")

    receipts: dict[str, Mapping[str, Any]] = {}
    for index, raw_receipt in enumerate(raw_workloads):
        label = f"plan.workloads[{index}]"
        receipt = _mapping(raw_receipt, label)
        expect_keys(receipt, label, required=("cell_id", "sha256", "value"))
        cell_id = validate_identifier(receipt["cell_id"], f"{label}.cell_id")
        if cell_id in receipts:
            raise ContractError(f"plan.workloads repeats cell {cell_id!r}")
        cell_entry = cells_by_id.get(cell_id)
        if cell_entry is None:
            raise ContractError(f"{label} references an unknown campaign cell")
        workload = _mapping(receipt["value"], f"{label}.value")
        expect_keys(
            workload,
            f"{label}.value",
            required=(*WORKLOAD_EXECUTION_FIELDS, "requests"),
        )
        requests = _array(workload["requests"], f"{label}.value.requests")
        if not requests:
            raise ContractError(f"{label}.value.requests must not be empty")
        for request_index, request in enumerate(requests):
            request_workload_identity(
                _mapping(request, f"{label}.value.requests[{request_index}]"),
                f"{label}.value.requests[{request_index}]",
            )
        validate_sha256(receipt["sha256"], f"{label}.sha256")
        if sha256_bytes(canonical_json_bytes(workload)) != receipt["sha256"]:
            raise ComparabilityError(f"{label} SHA-256 does not match its workload value")
        expected = build_workload_identity(
            _mapping(cell_entry["cell"], f"{label}.cell"),
            requests_by_cell[cell_id],
            manifest,
            label=f"plan workload {cell_id!r}",
        )
        if canonical_json_bytes(workload) != canonical_json_bytes(expected):
            raise ComparabilityError(f"{label} drifts from its immutable matrix/request workload")
        receipts[cell_id] = receipt
    if set(receipts) != set(cells_by_id):
        raise ComparabilityError("plan.workloads do not exactly cover campaign cells")
    return receipts


def validate_plan(value: Any, *, root: Path) -> Mapping[str, Any]:
    plan = _mapping(value, "plan")
    expect_keys(
        plan,
        "plan",
        required=(
            "schema_version",
            "campaign_id",
            "created_at_utc",
            "source",
            "contract",
            "matrices",
            "lanes",
            "request_manifest",
            "workloads",
            "preflight",
            "cells",
            "execution",
            "invocations",
            "readiness",
        ),
    )
    if plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ContractError(f"plan.schema_version must be {PLAN_SCHEMA_VERSION}")
    validate_identifier(plan["campaign_id"], "plan.campaign_id")
    _nonempty_string(plan["created_at_utc"], "plan.created_at_utc")
    _validate_source(plan["source"], "plan.source")

    contract_receipt = _mapping(plan["contract"], "plan.contract")
    expect_keys(
        contract_receipt,
        "plan.contract",
        required=("path", "sha256", "schema_version", "contract_id"),
    )
    contract_path = _relative_path(root, contract_receipt["path"], "plan.contract.path")
    validate_sha256(contract_receipt["sha256"], "plan.contract.sha256")
    if sha256_file(contract_path) != contract_receipt["sha256"]:
        raise ComparabilityError("plan.contract SHA-256 no longer matches its recorded file")
    contract = validate_contract(load_json(contract_path), str(contract_path))
    require_canonical_contract(root, contract_path, contract)
    if contract["schema_version"] != contract_receipt["schema_version"]:
        raise ComparabilityError("plan.contract schema version drift")
    if contract["contract_id"] != contract_receipt["contract_id"]:
        raise ComparabilityError("plan.contract ID drift")

    matrices = _array(plan["matrices"], "plan.matrices")
    if not matrices:
        raise ContractError("plan.matrices must not be empty")
    known_matrix_ids: set[str] = set()
    expected_cell_entries: dict[str, Mapping[str, Any]] = {}
    for index, raw_receipt in enumerate(matrices):
        receipt = _mapping(raw_receipt, f"plan.matrices[{index}]")
        expect_keys(receipt, f"plan.matrices[{index}]", required=("path", "sha256", "matrix_id"))
        matrix_path = _relative_path(root, receipt["path"], f"plan.matrices[{index}].path")
        validate_sha256(receipt["sha256"], f"plan.matrices[{index}].sha256")
        if sha256_file(matrix_path) != receipt["sha256"]:
            raise ComparabilityError(f"plan.matrices[{index}] SHA-256 no longer matches its recorded file")
        matrix = validate_matrix(load_json(matrix_path), str(matrix_path))
        verify_campaign_matrix_binding(root, matrix_path, matrix)
        if matrix["matrix_id"] != receipt["matrix_id"]:
            raise ComparabilityError(f"plan.matrices[{index}] ID drift")
        if matrix["matrix_id"] in known_matrix_ids:
            raise ContractError(f"plan.matrices has duplicate ID {matrix['matrix_id']!r}")
        known_matrix_ids.add(str(matrix["matrix_id"]))
        for cell in matrix_cells(matrix):
            cell_id = str(cell["cell_id"])
            if cell_id in expected_cell_entries:
                raise ContractError(f"selected matrices repeat cell {cell_id!r}")
            expected_cell_entries[cell_id] = {
                "matrix_id": matrix["matrix_id"],
                "matrix_tier": matrix["tier"],
                "matrix_path": receipt["path"],
                "matrix_sha256": receipt["sha256"],
                "cell": cell,
            }

    lanes = _mapping(plan["lanes"], "plan.lanes")
    expect_keys(lanes, "plan.lanes", required=("riley", "competitor"))
    lane_ids: dict[str, str] = {}
    for role in ("riley", "competitor"):
        receipt = _mapping(lanes[role], f"plan.lanes.{role}")
        expect_keys(
            receipt,
            f"plan.lanes.{role}",
            required=("path", "sha256", "lane_id", "availability", "command_status", "pin_status"),
        )
        lane_path = _relative_path(root, receipt["path"], f"plan.lanes.{role}.path")
        validate_sha256(receipt["sha256"], f"plan.lanes.{role}.sha256")
        if sha256_file(lane_path) != receipt["sha256"]:
            raise ComparabilityError(f"plan.lanes.{role} SHA-256 no longer matches its recorded file")
        lane = validate_lane(load_json(lane_path), str(lane_path))
        verify_campaign_lane_binding(root, lane_path, lane)
        lane_ids[role] = validate_identifier(receipt["lane_id"], f"plan.lanes.{role}.lane_id")
        if (
            lane["lane_id"] != lane_ids[role]
            or lane["availability"] != receipt["availability"]
            or lane["command"].get("status") != receipt["command_status"]
            or lane["engine"].get("pin_status") != receipt["pin_status"]
        ):
            raise ComparabilityError(f"plan.lanes.{role} identity/availability drift")
    if lane_ids["riley"] == lane_ids["competitor"]:
        raise ContractError("plan lanes must use distinct lane IDs")

    request_manifest = _mapping(plan["request_manifest"], "plan.request_manifest")
    expect_keys(
        request_manifest,
        "plan.request_manifest",
        required=("path", "sha256", "manifest_id", "model_identity"),
    )
    request_path = _relative_path(root, request_manifest["path"], "plan.request_manifest.path")
    validate_sha256(request_manifest["sha256"], "plan.request_manifest.sha256")
    validate_identifier(request_manifest["manifest_id"], "plan.request_manifest.manifest_id")
    if sha256_file(request_path) != request_manifest["sha256"]:
        raise ComparabilityError("plan.request_manifest SHA-256 no longer matches its recorded file")
    manifest = validate_request_manifest(load_json(request_path), str(request_path))
    verify_campaign_request_binding(root, manifest)
    if (
        manifest["manifest_id"] != request_manifest["manifest_id"]
        or canonical_json_bytes(manifest["model_identity"])
        != canonical_json_bytes(request_manifest["model_identity"])
    ):
        raise ComparabilityError("plan.request_manifest identity drift")

    preflight = _mapping(plan["preflight"], "plan.preflight")
    expect_keys(preflight, "plan.preflight", required=("status", "path", "sha256", "values", "script"))
    if preflight["status"] not in {"passed", "missing"}:
        raise ContractError("plan.preflight.status must be passed or missing")
    if preflight["status"] == "passed":
        if not isinstance(preflight["path"], str) or not preflight["path"]:
            raise ContractError("plan.preflight.path must identify a passed receipt")
        preflight_path = _relative_path(root, preflight["path"], "plan.preflight.path")
        validate_sha256(preflight["sha256"], "plan.preflight.sha256")
        values = _mapping(preflight["values"], "plan.preflight.values")
        if sha256_file(preflight_path) != preflight["sha256"]:
            raise ComparabilityError("plan.preflight receipt SHA-256 no longer matches its recorded file")
        actual_values = load_preflight_receipt(preflight_path, plan["source"])
        if canonical_json_bytes(actual_values) != canonical_json_bytes(values):
            raise ComparabilityError("plan.preflight values drift from its hashed receipt")
        verify_canonical_preflight_script_receipt(root, preflight["script"], "plan.preflight.script")
    elif any(preflight[field] is not None for field in ("path", "sha256", "values")):
        raise ContractError("missing plan preflight must not contain a partial receipt")
    else:
        verify_canonical_preflight_script_receipt(root, preflight["script"], "plan.preflight.script")

    cells = _array(plan["cells"], "plan.cells")
    if not cells:
        raise ContractError("plan.cells must not be empty")
    cell_ids: set[str] = set()
    cells_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_cell in enumerate(cells):
        cell_entry = _mapping(raw_cell, f"plan.cells[{index}]")
        expect_keys(
            cell_entry,
            f"plan.cells[{index}]",
            required=("matrix_id", "matrix_tier", "matrix_path", "matrix_sha256", "cell"),
        )
        cell = _mapping(cell_entry["cell"], f"plan.cells[{index}].cell")
        cell_id = validate_identifier(cell.get("cell_id"), f"plan.cells[{index}].cell.cell_id")
        if cell.get("executable") is not True:
            raise ContractError(
                f"plan.cells[{index}] is a template/non-concrete workload and cannot support a claim"
            )
        if cell_id in cell_ids:
            raise ContractError(f"plan.cells has duplicate cell {cell_id!r}")
        cell_ids.add(cell_id)
        cells_by_id[cell_id] = cell_entry
        expected = expected_cell_entries.get(cell_id)
        if expected is None or canonical_json_bytes(cell_entry) != canonical_json_bytes(expected):
            raise ComparabilityError(f"plan.cells[{index}] drifts from its pinned matrix")
    if set(cells_by_id) != set(expected_cell_entries):
        raise ComparabilityError("plan cells do not exactly cover selected matrix cells")

    _validate_plan_workloads(plan, cells_by_id=cells_by_id, manifest=manifest)

    execution = _array(plan["execution"], "plan.execution")
    invocations = _array(plan["invocations"], "plan.invocations")
    if not execution or not invocations:
        raise ContractError("plan execution and invocations must not be empty")
    execution_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_execution in enumerate(execution):
        item = _mapping(raw_execution, f"plan.execution[{index}]")
        expect_keys(
            item,
            f"plan.execution[{index}]",
            required=("execution_id", "sequence", "cell_id", "run_index", "order", "lane_order"),
        )
        execution_id = validate_identifier(item["execution_id"], f"plan.execution[{index}].execution_id")
        if execution_id in execution_by_id:
            raise ContractError(f"plan.execution repeats execution_id {execution_id!r}")
        if item["cell_id"] not in cell_ids:
            raise ContractError(f"plan.execution[{index}] references an unknown cell")
        _integer(item["sequence"], f"plan.execution[{index}].sequence", minimum=1)
        _integer(item["run_index"], f"plan.execution[{index}].run_index", minimum=1)
        if item["order"] not in {"AB", "BA"}:
            raise ContractError(f"plan.execution[{index}].order must be AB or BA")
        lane_order = _array(item["lane_order"], f"plan.execution[{index}].lane_order")
        required_order = ["riley", "competitor"] if item["order"] == "AB" else ["competitor", "riley"]
        if lane_order != required_order:
            raise ContractError(f"plan.execution[{index}] has AB/BA lane-order drift")
        execution_by_id[execution_id] = item

    invocation_ids: set[str] = set()
    per_execution_positions: dict[str, set[str]] = defaultdict(set)
    for index, raw_invocation in enumerate(invocations):
        item = _mapping(raw_invocation, f"plan.invocations[{index}]")
        expect_keys(
            item,
            f"plan.invocations[{index}]",
            required=(
                "invocation_id",
                "sequence",
                "execution_id",
                "cell_id",
                "run_index",
                "order",
                "position",
                "role",
                "lane_id",
            ),
        )
        invocation_id = validate_identifier(item["invocation_id"], f"plan.invocations[{index}].invocation_id")
        if invocation_id in invocation_ids:
            raise ContractError(f"plan.invocations repeats invocation_id {invocation_id!r}")
        invocation_ids.add(invocation_id)
        execution_id = item["execution_id"]
        if execution_id not in execution_by_id:
            raise ContractError(f"plan.invocations[{index}] references unknown execution")
        execution_item = execution_by_id[execution_id]
        for key in ("cell_id", "run_index", "order"):
            if item[key] != execution_item[key]:
                raise ContractError(f"plan.invocations[{index}] {key} drifts from execution")
        if item["position"] not in {"A", "B"}:
            raise ContractError(f"plan.invocations[{index}].position must be A or B")
        expected_role = execution_item["lane_order"][0 if item["position"] == "A" else 1]
        if item["role"] != expected_role:
            raise ContractError(f"plan.invocations[{index}] role/order inversion")
        if item["lane_id"] != lane_ids[item["role"]]:
            raise ContractError(f"plan.invocations[{index}] lane ID/role inversion")
        per_execution_positions[execution_id].add(str(item["position"]))
    if any(positions != {"A", "B"} for positions in per_execution_positions.values()):
        raise ContractError("each planned execution must contain one A and one B invocation")
    if set(per_execution_positions) != set(execution_by_id):
        raise ContractError("one or more planned executions have no invocations")
    required_runs = int(contract["required_independent_runs"])
    per_cell_runs: dict[str, dict[int, str]] = defaultdict(dict)
    for execution_item in execution_by_id.values():
        cell_id = str(execution_item["cell_id"])
        run_index = int(execution_item["run_index"])
        if run_index in per_cell_runs[cell_id]:
            raise ContractError(f"plan repeats run index {run_index} for cell {cell_id!r}")
        per_cell_runs[cell_id][run_index] = str(execution_item["order"])
    for cell_id in cell_ids:
        expected_runs = set(range(1, required_runs + 1))
        actual_runs = set(per_cell_runs.get(cell_id, {}))
        if actual_runs != expected_runs:
            raise ContractError(
                f"cell {cell_id!r} lacks the contract's {required_runs} independent runs"
            )
        expected_orders = ["AB", "BA"]
        for run_index, order in per_cell_runs[cell_id].items():
            if order != expected_orders[(run_index - 1) % len(expected_orders)]:
                raise ContractError(f"cell {cell_id!r} has AB/BA schedule drift at run {run_index}")

    # Sequence is part of the immutable thermal/AB-BA schedule, not an
    # advisory label.  Re-derive the exact list ordering from the already
    # pinned cell catalog so a hand-edited plan cannot shuffle cells, arms, or
    # invocation sequence while retaining otherwise valid per-cell coverage.
    expected_execution: list[dict[str, Any]] = []
    expected_invocations: list[dict[str, Any]] = []
    execution_sequence = 0
    invocation_sequence = 0
    for cell_entry in cells:
        cell_id = str(cell_entry["cell"]["cell_id"])
        for run_index in range(1, required_runs + 1):
            order = expected_orders[(run_index - 1) % len(expected_orders)]
            roles = ["riley", "competitor"] if order == "AB" else ["competitor", "riley"]
            execution_sequence += 1
            execution_id = f"{cell_id}:run-{run_index:02d}"
            expected_execution.append(
                {
                    "execution_id": execution_id,
                    "sequence": execution_sequence,
                    "cell_id": cell_id,
                    "run_index": run_index,
                    "order": order,
                    "lane_order": roles,
                }
            )
            for position, role in zip(("A", "B"), roles, strict=True):
                invocation_sequence += 1
                expected_invocations.append(
                    {
                        "invocation_id": f"{execution_id}:{position}",
                        "sequence": invocation_sequence,
                        "execution_id": execution_id,
                        "cell_id": cell_id,
                        "run_index": run_index,
                        "order": order,
                        "position": position,
                        "role": role,
                        "lane_id": lane_ids[role],
                    }
                )
    if canonical_json_bytes(execution) != canonical_json_bytes(expected_execution):
        raise ComparabilityError("plan execution order/sequence drifts from immutable C01 schedule")
    if canonical_json_bytes(invocations) != canonical_json_bytes(expected_invocations):
        raise ComparabilityError("plan invocation order/sequence drifts from immutable C01 schedule")

    readiness = _mapping(plan["readiness"], "plan.readiness")
    expect_keys(readiness, "plan.readiness", required=("state", "blocked_reasons"))
    if readiness["state"] not in {"ready", "blocked"}:
        raise ContractError("plan.readiness.state must be ready or blocked")
    blocked_reasons = _array(readiness["blocked_reasons"], "plan.readiness.blocked_reasons")
    if any(not isinstance(reason, str) or not reason for reason in blocked_reasons):
        raise ContractError("plan.readiness.blocked_reasons must be a string array")
    if (readiness["state"] == "ready") != (not blocked_reasons):
        raise ContractError("plan.readiness state/reason mismatch")

    return plan


def rederive_readiness(plan: Mapping[str, Any], *, root: Path) -> list[str]:
    """Recompute every claim-time admission gate from live, pinned evidence.

    `plan.readiness` is retained as a planning aid, but it is never an
    authority for a favorable result.  This makes a plan edited from
    ``blocked`` to ``ready`` harmless.
    """

    reasons: list[str] = []
    source = _mapping(plan["source"], "plan.source")
    if source["git_dirty"] or source["development_only"]:
        reasons.append("plan was created from a dirty/development-only source tree")
    current = current_source_receipt(root)
    if current["git_dirty"]:
        reasons.append("current source tree is dirty at claim time")
    if current["git_revision"] != source["git_revision"]:
        reasons.append("current Git HEAD differs from the campaign source revision")

    preflight = _mapping(plan["preflight"], "plan.preflight")
    if preflight["status"] != "passed":
        reasons.append("no passed reviewed thermal/clock/GPU preflight receipt is bound")
    else:
        # Re-read both the receipt and the reviewed script even though
        # validate_plan already checked them.  A passed claim must not depend
        # on a cached plan value or a stale hash assertion.
        path = _relative_path(root, preflight["path"], "plan.preflight.path")
        if sha256_file(path) != preflight["sha256"]:
            reasons.append("preflight receipt hash drifted after plan creation")
        else:
            values = load_preflight_receipt(path, source)
            if canonical_json_bytes(values) != canonical_json_bytes(preflight["values"]):
                reasons.append("preflight receipt values drifted after plan creation")
        verify_canonical_preflight_script_receipt(root, preflight["script"], "plan.preflight.script")

    lanes = _mapping(plan["lanes"], "plan.lanes")
    for role in ("riley", "competitor"):
        receipt = _mapping(lanes[role], f"plan.lanes.{role}")
        try:
            path = require_campaign_artifact_path(
                root,
                _relative_path(root, receipt["path"], f"plan.lanes.{role}.path"),
                f"plan.lanes.{role}.path",
            )
        except ContractError as error:
            reasons.append(f"{role} lane is outside the declared campaign artifact workspace: {error}")
            continue
        if sha256_file(path) != receipt["sha256"]:
            reasons.append(f"{role} lane hash drifted after plan creation")
            continue
        lane = validate_lane(load_json(path), str(path))
        verify_campaign_lane_binding(root, path, lane)
        try:
            _available_pinned_lane(lane, f"{role} lane {lane['lane_id']!r}")
        except ComparabilityError as error:
            reasons.append(str(error))
        reasons.extend(_materialized_lane_claim_reasons(lane, plan=plan, root=root, role=role))
    return reasons


def _validate_metrics(value: Any, label: str) -> Mapping[str, Any]:
    metrics = _mapping(value, label)
    expect_keys(metrics, label, required=SUCCESS_METRIC_REQUIRED_FIELDS, optional=SUCCESS_METRIC_OPTIONAL_FIELDS)
    for field in SUCCESS_METRIC_REQUIRED_FIELDS:
        finite_number(metrics[field], f"{label}.{field}", minimum=0.0)
    for field in SUCCESS_METRIC_OPTIONAL_FIELDS:
        if field in metrics and metrics[field] is not None:
            finite_number(metrics[field], f"{label}.{field}", minimum=0.0)
    return metrics


def _validate_request(value: Any, label: str) -> Mapping[str, Any]:
    request = _mapping(value, label)
    expect_keys(request, label, required=REQUEST_REQUIRED_FIELDS)
    validate_identifier(request["request_id"], f"{label}.request_id")
    validate_sha256(request["prompt_token_ids_sha256"], f"{label}.prompt_token_ids_sha256")
    validate_sha256(request["generated_token_ids_sha256"], f"{label}.generated_token_ids_sha256")
    if request["status"] not in {"success", "failure"}:
        raise ContractError(f"{label}.status must be success or failure")
    if request["status"] == "success" and request["failure_reason"] is not None:
        raise ContractError(f"{label}.failure_reason must be null for success")
    if request["status"] == "failure" and not isinstance(request["failure_reason"], str):
        raise ContractError(f"{label}.failure_reason must be a string for failure")
    _integer(request["requested_output_tokens"], f"{label}.requested_output_tokens", minimum=1)
    _integer(request["generated_tokens"], f"{label}.generated_tokens", minimum=0)
    _integer(request["terminal_event_count"], f"{label}.terminal_event_count", minimum=0)
    for field in ("ttft_ms", "tpot_ms", "end_to_end_ms"):
        if request["status"] == "success":
            finite_number(request[field], f"{label}.{field}", minimum=0.0)
        elif request[field] is not None:
            finite_number(request[field], f"{label}.{field}", minimum=0.0)
    # Validate the complete input-side identity here as well as against the
    # manifest below.  In particular, a row cannot omit a seed or claim an
    # unparameterized sampling profile while keeping the prompt hash intact.
    request_workload_identity(request, label)
    return request


def validate_raw_row(
    value: Mapping[str, Any],
    *,
    label: str,
    plan: Mapping[str, Any],
    plan_sha256: str,
    contract: Mapping[str, Any],
    invocation_by_id: Mapping[str, Mapping[str, Any]],
    cells_by_id: Mapping[str, Mapping[str, Any]],
    workloads_by_cell: Mapping[str, Mapping[str, Any]],
    lanes_by_role: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    row = _mapping(value, label)
    expect_keys(row, label, required=RAW_REQUIRED_FIELDS, optional=RAW_OPTIONAL_FIELDS)
    if row["schema_version"] != RAW_SCHEMA_VERSION:
        raise ContractError(f"{label}.schema_version must be {RAW_SCHEMA_VERSION}")
    if row["campaign_id"] != plan["campaign_id"]:
        raise ComparabilityError(f"{label}.campaign_id differs from plan")
    if row["campaign_plan_sha256"] != plan_sha256:
        raise ComparabilityError(f"{label}.campaign_plan_sha256 differs from plan")
    validate_sha256(row["campaign_plan_sha256"], f"{label}.campaign_plan_sha256")
    invocation_id = validate_identifier(row["invocation_id"], f"{label}.invocation_id")
    if invocation_id not in invocation_by_id:
        raise ComparabilityError(f"{label}.invocation_id is not in the plan")
    invocation = invocation_by_id[invocation_id]
    for field in ("lane_id", "role", "execution_id", "cell_id", "run_index", "order", "position"):
        if row[field] != invocation[field]:
            raise ComparabilityError(f"{label}.{field} drifts from planned invocation")
    cell_entry = cells_by_id[row["cell_id"]]
    cell = cell_entry["cell"]
    if row["measurement_mode"] != cell["measurement_mode"]:
        raise ComparabilityError(f"{label}.measurement_mode drifts from planned cell")
    if row["request_manifest_sha256"] != plan["request_manifest"]["sha256"]:
        raise ComparabilityError(f"{label}.request_manifest_sha256 differs from plan")
    validate_sha256(row["request_manifest_sha256"], f"{label}.request_manifest_sha256")
    workload_receipt = workloads_by_cell.get(str(row["cell_id"]))
    if workload_receipt is None:
        raise ComparabilityError(f"{label}.cell_id has no immutable workload receipt")
    validate_sha256(row["workload_sha256"], f"{label}.workload_sha256")
    if row["workload_sha256"] != workload_receipt["sha256"]:
        raise ComparabilityError(f"{label}.workload_sha256 differs from the planned workload")
    workload = _mapping(row["workload"], f"{label}.workload")
    expect_keys(workload, f"{label}.workload", required=WORKLOAD_EXECUTION_FIELDS)
    expected_execution = workload_execution_receipt(
        _mapping(workload_receipt["value"], f"{label}.planned_workload")
    )
    if canonical_json_bytes(workload) != canonical_json_bytes(expected_execution):
        raise ComparabilityError(
            f"{label}.workload differs from the planned model/warm/arrival/client/SLO behavior"
        )
    _nonempty_string(row["recorded_at_utc"], f"{label}.recorded_at_utc")
    source = _validate_source(row["source"], f"{label}.source")
    if source != plan["source"]:
        raise ComparabilityError(f"{label}.source drifts from plan")
    environment = _mapping(row["environment"], f"{label}.environment")
    required_environment_keys = list(contract["required_environment_keys"])
    expect_keys(environment, f"{label}.environment", required=required_environment_keys)
    for key in required_environment_keys:
        _environment_scalar(environment[key], f"{label}.environment.{key}")
    role = str(row["role"])
    lane = lanes_by_role.get(role)
    if lane is None:  # defensive: invocation validation above already closes this set.
        raise ComparabilityError(f"{label}.role has no materialized lane receipt")
    receipts = _mapping(lane["artifact_receipts"], f"{label}.planned_lane.artifact_receipts")
    for field in ("executable_sha256", "dependency_lock_sha256"):
        validate_sha256(environment[field], f"{label}.environment.{field}")
        if environment[field] != receipts[field]:
            raise ComparabilityError(
                f"{label}.environment.{field} differs from the materialized {role} lane receipt"
            )
    if row["phase"] != "measured":
        raise ComparabilityError(f"{label}.phase must be measured; warmups are never statistics")
    if row["status"] not in {"success", "failure"}:
        raise ContractError(f"{label}.status must be success or failure")
    if row["status"] == "success":
        if row["failure_reason"] is not None:
            raise ContractError(f"{label}.failure_reason must be null for success")
        _validate_metrics(row["metrics"], f"{label}.metrics")
    else:
        if not isinstance(row["failure_reason"], str) or not row["failure_reason"]:
            raise ContractError(f"{label}.failure_reason must be non-empty for failure")
        if row["metrics"] is not None:
            raise ContractError(f"{label}.metrics must be null for failure")
    requests = _array(row["requests"], f"{label}.requests")
    if row["status"] == "success" and not requests:
        raise ContractError(f"{label}.requests must not be empty for success")
    for request_index, request in enumerate(requests):
        _validate_request(request, f"{label}.requests[{request_index}]")
    _integer(row["adapter_sequence"], f"{label}.adapter_sequence", minimum=1)
    previous_receipt = row["adapter_previous_receipt_sha256"]
    if previous_receipt is not None:
        validate_sha256(previous_receipt, f"{label}.adapter_previous_receipt_sha256")
    validate_sha256(row["adapter_receipt_sha256"], f"{label}.adapter_receipt_sha256")
    if "preflight_receipt_sha256" in row:
        validate_sha256(row["preflight_receipt_sha256"], f"{label}.preflight_receipt_sha256")
        preflight = _mapping(plan["preflight"], "plan.preflight")
        if preflight["status"] != "passed" or row["preflight_receipt_sha256"] != preflight["sha256"]:
            raise ComparabilityError(f"{label}.preflight_receipt_sha256 drifts from planned preflight")
    return row


def _incomparable_report(
    *,
    campaign_id: str | None,
    plan_sha256: str | None,
    contract_sha256: str | None,
    reasons: Iterable[str],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_plan_sha256": plan_sha256,
        "contract_sha256": contract_sha256,
        "status": "incomparable",
        "comparability": {"comparable": False, "reasons": sorted(set(reasons))},
        "evidence": {"expected_invocations": None, "received_invocations": None},
        "cells": [],
        "m4": {"evaluated": False, "passed": None, "failures": []},
        "m5": {"evaluated": False, "passed": None, "failures": []},
    }


def _plan_context(
    plan: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[
    Mapping[str, Any],
    str,
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    contract_path = _relative_path(root, plan["contract"]["path"], "plan.contract.path")
    contract = validate_contract(load_json(contract_path), str(contract_path))
    invocations = {str(item["invocation_id"]): item for item in plan["invocations"]}
    cells = {str(item["cell"]["cell_id"]): item for item in plan["cells"]}
    workloads = {str(item["cell_id"]): item for item in plan["workloads"]}
    return contract, sha256_file(contract_path), invocations, cells, workloads


def _request_identity_failures(
    row: Mapping[str, Any],
    expected_requests: Mapping[str, Mapping[str, Any]],
    label: str,
) -> list[str]:
    failures: list[str] = []
    observed: dict[str, Mapping[str, Any]] = {}
    for request in row["requests"]:
        request_id = str(request["request_id"])
        if request_id in observed:
            failures.append(f"{label}: duplicate request observation {request_id!r}")
        observed[request_id] = request
    if set(observed) != set(expected_requests):
        failures.append(f"{label}: request set differs from immutable request manifest")
        return failures
    for request_id, expected in expected_requests.items():
        actual = observed[request_id]
        expected_identity = request_workload_identity(expected, f"manifest request {request_id!r}")
        actual_identity = request_workload_identity(actual, f"{label} request {request_id!r}")
        for field in REQUEST_WORKLOAD_FIELDS:
            if canonical_json_bytes(actual_identity[field]) != canonical_json_bytes(expected_identity[field]):
                failures.append(f"{label}: request {request_id!r} {field} differs from manifest")
        if actual["status"] != "success":
            failures.append(f"{label}: request {request_id!r} failed")
        if actual["generated_tokens"] != expected["requested_output_tokens"]:
            failures.append(f"{label}: request {request_id!r} generated an unexpected token count")
        if actual["terminal_event_count"] != 1:
            failures.append(f"{label}: request {request_id!r} has duplicate/missing terminal event")
    return failures


def _aggregate_lane_cell(
    rows: Sequence[Mapping[str, Any]],
    *,
    percentile_method: str,
    expected_request_ids: set[str],
    expected_run_indices: set[int],
) -> dict[str, Any]:
    """Aggregate only equal, complete independent-run summaries.

    A pooled request percentile lets a noisy or unusually short run change its
    weight merely by changing its request count.  M4/M5 instead take a p95
    inside each independent run, then use the median of those per-run p95
    summaries.  The explicit request-ID check keeps every run on the exact
    frozen workload manifest before any statistic is calculated.
    """

    percentile = nearest_rank if percentile_method == "nearest-rank" else r7
    per_run_summaries: list[dict[str, Any]] = []
    seen_run_indices: set[int] = set()
    for row in sorted(rows, key=lambda item: int(item["run_index"])):
        run_index = _integer(row["run_index"], "raw run_index", minimum=1)
        if run_index in seen_run_indices:
            raise ComparabilityError(f"lane/cell observations repeat independent run {run_index}")
        seen_run_indices.add(run_index)
        requests = _array(row["requests"], "raw row requests")
        request_ids = [str(request["request_id"]) for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ComparabilityError(f"independent run {run_index} repeats a request observation")
        if set(request_ids) != expected_request_ids:
            raise ComparabilityError(
                f"independent run {run_index} does not exactly cover the frozen workload manifest"
            )
        metrics = _mapping(row["metrics"], "raw row metrics")
        per_run_summaries.append(
            {
                "run_index": run_index,
                "request_ids": sorted(request_ids),
                "ttft_p95_ms": percentile(
                    (float(request["ttft_ms"]) for request in requests), 0.95
                ),
                "tpot_p95_ms": percentile(
                    (float(request["tpot_ms"]) for request in requests), 0.95
                ),
                "slo_goodput_tokens_per_second": float(
                    metrics["slo_goodput_tokens_per_second"]
                ),
                "peak_vram_bytes": float(metrics["peak_vram_bytes"]),
                "usable_kv_bytes": float(metrics["usable_kv_bytes"]),
            }
        )
    if seen_run_indices != expected_run_indices:
        raise ComparabilityError("lane/cell observations do not exactly cover planned independent runs")

    return {
        "ttft_p95_ms": r7(
            (float(summary["ttft_p95_ms"]) for summary in per_run_summaries), 0.5
        ),
        "tpot_p95_ms": r7(
            (float(summary["tpot_p95_ms"]) for summary in per_run_summaries), 0.5
        ),
        "slo_goodput_tokens_per_second_median": r7(
            (
                float(summary["slo_goodput_tokens_per_second"])
                for summary in per_run_summaries
            ),
            0.5,
        ),
        "peak_vram_bytes": max(
            float(summary["peak_vram_bytes"]) for summary in per_run_summaries
        ),
        "usable_kv_bytes_median": r7(
            (float(summary["usable_kv_bytes"]) for summary in per_run_summaries), 0.5
        ),
        "per_run_summaries": per_run_summaries,
    }


def _ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0.0:
        raise ComparabilityError(f"{label} denominator must be positive")
    return numerator / denominator


def _cell_required_for(cell_entry: Mapping[str, Any]) -> set[str]:
    source: Mapping[str, Any]
    if "required_for" in cell_entry:
        source = cell_entry
    else:
        source = _mapping(cell_entry["cell"], "planned cell")
    value = source.get("required_for", [])
    if not isinstance(value, list):
        raise ContractError("planned cell.required_for must be a JSON array")
    return set(value)


def _assess_m4(
    contract: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> tuple[bool | None, list[str]]:
    required = [cell for cell in cells if "m4" in _cell_required_for(cell)]
    if not required:
        return None, ["no campaign cell is marked required_for m4"]
    rule = contract["m4"]
    failures: list[str] = []
    for cell in required:
        ratios = cell["ratios"]
        cell_id = cell["cell_id"]
        if ratios["ttft_p95"] > float(rule["ttft_p95_ratio_max"]):
            failures.append(f"{cell_id}: TTFT p95 ratio exceeds M4 limit")
        if ratios["tpot_p95"] > float(rule["tpot_p95_ratio_max"]):
            failures.append(f"{cell_id}: TPOT p95 ratio exceeds M4 limit")
        if cell["matrix_tier"] in {"serving-slo", "S"} and "slo_goodput_ratio_min" in rule:
            if ratios["slo_goodput"] < float(rule["slo_goodput_ratio_min"]):
                failures.append(f"{cell_id}: SLO goodput ratio misses M4 limit")
        if "peak_vram_ratio_max" in rule and ratios["peak_vram"] > float(rule["peak_vram_ratio_max"]):
            failures.append(f"{cell_id}: peak VRAM ratio exceeds M4 limit")
    return not failures, failures


def _assess_m5(
    contract: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> tuple[bool | None, list[str], dict[str, float]]:
    required = [cell for cell in cells if "m5" in _cell_required_for(cell)]
    if not required:
        return None, ["no campaign cell is marked required_for m5"], {}
    rule = contract["m5"]
    ratios_by_metric = {
        metric: [float(cell["ratios"][metric]) for cell in required]
        for metric in ("ttft_p95", "tpot_p95", "slo_goodput", "peak_vram")
    }
    geometric = {
        "ttft_p95": geometric_mean(ratios_by_metric["ttft_p95"], "M5 TTFT ratio"),
        "tpot_p95": geometric_mean(ratios_by_metric["tpot_p95"], "M5 TPOT ratio"),
        "slo_goodput": geometric_mean(ratios_by_metric["slo_goodput"], "M5 goodput ratio"),
        "peak_vram": geometric_mean(ratios_by_metric["peak_vram"], "M5 VRAM ratio"),
    }
    failures: list[str] = []
    if geometric["ttft_p95"] > float(rule["ttft_p95_ratio_max"]):
        failures.append("M5 geometric-mean TTFT p95 ratio exceeds limit")
    if geometric["tpot_p95"] > float(rule["tpot_p95_ratio_max"]):
        failures.append("M5 geometric-mean TPOT p95 ratio exceeds limit")
    if "slo_goodput_ratio_min" in rule and geometric["slo_goodput"] < float(rule["slo_goodput_ratio_min"]):
        failures.append("M5 geometric-mean SLO goodput ratio misses limit")
    if "peak_vram_ratio_max" in rule and geometric["peak_vram"] > float(rule["peak_vram_ratio_max"]):
        failures.append("M5 geometric-mean peak VRAM ratio exceeds limit")

    primary_ids = set(rule.get("primary_cell_ids", []))
    if not primary_ids:
        primary_ids = {str(cell["cell_id"]) for cell in required if cell.get("primary") is True}
    if not primary_ids:
        failures.append("M5 requires at least one predeclared primary cell")
    if primary_ids:
        present_ids = {str(cell["cell_id"]) for cell in required}
        missing = sorted(primary_ids - present_ids)
        if missing:
            failures.append("M5 primary cell(s) absent: " + ", ".join(missing))
        for cell in required:
            if cell["cell_id"] not in primary_ids:
                continue
            ratios = cell["ratios"]
            if ratios["ttft_p95"] > float(rule["ttft_p95_ratio_max"]):
                failures.append(f"{cell['cell_id']}: primary TTFT p95 ratio exceeds M5 limit")
            if ratios["tpot_p95"] > float(rule["tpot_p95_ratio_max"]):
                failures.append(f"{cell['cell_id']}: primary TPOT p95 ratio exceeds M5 limit")
            if "slo_goodput_ratio_min" in rule and ratios["slo_goodput"] < float(rule["slo_goodput_ratio_min"]):
                failures.append(f"{cell['cell_id']}: primary SLO goodput ratio misses M5 limit")
            if "peak_vram_ratio_max" in rule and ratios["peak_vram"] > float(rule["peak_vram_ratio_max"]):
                failures.append(f"{cell['cell_id']}: primary peak VRAM ratio exceeds M5 limit")
    return not failures, failures, geometric


def _partial_win(cells: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> bool:
    """True only when a comparable campaign has at least one M5-level win."""

    required = [cell for cell in cells if "m5" in _cell_required_for(cell)]
    if not required:
        return False
    rule = contract["m5"]
    return any(
        cell["ratios"]["ttft_p95"] <= float(rule["ttft_p95_ratio_max"])
        or cell["ratios"]["tpot_p95"] <= float(rule["tpot_p95_ratio_max"])
        or (
            "slo_goodput_ratio_min" in rule
            and cell["ratios"]["slo_goodput"] >= float(rule["slo_goodput_ratio_min"])
        )
        for cell in required
    )


def check_campaign(
    *,
    plan_path: Path,
    raw_paths: Sequence[Path],
    root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Produce a deterministic closed report without writing it."""

    plan_sha256: str | None = None
    contract_sha256: str | None = None
    campaign_id: str | None = None
    try:
        root = root.resolve()
        plan_path = require_campaign_artifact_path(root, plan_path, "claim execution plan path")
        if len(raw_paths) != 1:
            raise ComparabilityError("claim raw evidence must come from exactly one JSONL journal")
        raw_path = require_campaign_artifact_path(root, raw_paths[0], "claim raw evidence path")
        try:
            raw_metadata = raw_path.lstat()
        except OSError as error:
            raise ContractError(f"cannot inspect claim raw evidence path: {error}") from error
        if raw_path.suffix != ".jsonl" or not stat.S_ISREG(raw_metadata.st_mode):
            raise ComparabilityError("claim raw evidence must be exactly one regular JSONL journal")
        plan_sha256 = sha256_file(plan_path)
        unchecked_plan = load_json(plan_path)
        campaign_id = unchecked_plan.get("campaign_id") if isinstance(unchecked_plan, dict) else None
        plan = validate_plan(unchecked_plan, root=root)
        campaign_id = str(plan["campaign_id"])
        (
            contract,
            contract_sha256,
            invocation_by_id,
            cells_by_id,
            workloads_by_cell,
        ) = _plan_context(plan, root=root)
        readiness_reasons = rederive_readiness(plan, root=root)
        if readiness_reasons:
            raise ComparabilityError("campaign is not claim-ready: " + "; ".join(sorted(readiness_reasons)))
        lanes_by_role = _load_claim_lanes(plan, root=root)

        request_path = _relative_path(root, plan["request_manifest"]["path"], "plan.request_manifest.path")
        if sha256_file(request_path) != plan["request_manifest"]["sha256"]:
            raise ComparabilityError("request manifest SHA-256 no longer matches plan")
        manifest = validate_request_manifest(load_json(request_path), str(request_path))
        request_sets = manifest["request_sets"]
        expected_requests_by_cell: dict[str, dict[str, Mapping[str, Any]]] = {}
        for request_set in request_sets:
            request_set_object = _mapping(request_set, "request manifest.request_sets entry")
            cell_id = str(request_set_object.get("cell_id"))
            requests = _array(request_set_object.get("requests"), f"request set {cell_id}.requests")
            expected_requests_by_cell[cell_id] = {
                str(request["request_id"]): _mapping(request, f"request set {cell_id} request")
                for request in requests
            }
        if set(expected_requests_by_cell) != set(cells_by_id):
            raise ComparabilityError("request manifest/campaign cell coverage drift")

        identity_requirements = contract.get("identity_requirements", {})
        same_campaign_fields = (
            identity_requirements.get("same_campaign_fields", [])
            if isinstance(identity_requirements, dict)
            else []
        )
        common_environment_keys = [
            key for key in contract["required_environment_keys"] if key in same_campaign_fields
        ] or list(contract["required_environment_keys"])
        rows_by_invocation: dict[str, Mapping[str, Any]] = {}
        raw_rows_in_append_order: list[tuple[Path, Mapping[str, Any]]] = []
        environment_fingerprints: set[tuple[Any, ...]] = set()
        semantic_failures: list[str] = []
        preflight_values = plan["preflight"]["values"] if plan["preflight"]["status"] == "passed" else None
        preflight_environment_fields = {
            "gpu_model": "gpu_name",
            "compute_capability": "compute_capability",
            "driver_version": "driver_version",
        }
        for path, line_number, raw_row in load_jsonl([raw_path]):
            label = f"{path}:{line_number}"
            row = validate_raw_row(
                raw_row,
                label=label,
                plan=plan,
                plan_sha256=plan_sha256,
                contract=contract,
                invocation_by_id=invocation_by_id,
                cells_by_id=cells_by_id,
                workloads_by_cell=workloads_by_cell,
                lanes_by_role=lanes_by_role,
            )
            invocation_id = str(row["invocation_id"])
            if invocation_id in rows_by_invocation:
                raise ComparabilityError(f"duplicate raw observation for invocation {invocation_id!r}")
            rows_by_invocation[invocation_id] = row
            raw_rows_in_append_order.append((path, row))
            environment_fingerprints.add(tuple(row["environment"][key] for key in common_environment_keys))
            if preflight_values is not None:
                for environment_key, receipt_key in preflight_environment_fields.items():
                    if environment_key in row["environment"] and row["environment"][environment_key] != preflight_values[receipt_key]:
                        raise ComparabilityError(
                            f"{label}: {environment_key} differs from the frozen preflight receipt"
                        )
                if "git_commit" in row["environment"] and row["environment"]["git_commit"] != plan["source"]["git_revision"]:
                    raise ComparabilityError(f"{label}: environment git_commit differs from campaign source")
            if row["status"] == "failure":
                semantic_failures.append(f"{label}: lane invocation failed: {row['failure_reason']}")
            else:
                semantic_failures.extend(
                    _request_identity_failures(
                        row,
                        expected_requests_by_cell[str(row["cell_id"])],
                        label,
                    )
                )

        ordered_invocations = [
            str(item["invocation_id"])
            for item in sorted(plan["invocations"], key=lambda item: int(item["sequence"]))
        ]
        # Journal fields are part of RAW_REQUIRED_FIELDS for every claim.  A
        # legacy/imported JSONL without the one physical ordered chain is
        # intentionally incomparable rather than eligible for passed/M5.
        validate_append_only_chain(
            [row for _path, row in raw_rows_in_append_order],
            expected_invocation_ids=ordered_invocations,
        )

        expected_invocations = set(invocation_by_id)
        received_invocations = set(rows_by_invocation)
        if expected_invocations != received_invocations:
            missing = sorted(expected_invocations - received_invocations)
            unexpected = sorted(received_invocations - expected_invocations)
            reasons: list[str] = []
            if missing:
                reasons.append("missing planned invocation(s): " + ", ".join(missing))
            if unexpected:
                reasons.append("unexpected invocation(s): " + ", ".join(unexpected))
            raise ComparabilityError("; ".join(reasons))
        if len(environment_fingerprints) != 1:
            raise ComparabilityError("GPU UUID/driver/CUDA environment differs within campaign")

        token_pairs: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for row in rows_by_invocation.values():
            if row["status"] != "success":
                continue
            for request in row["requests"]:
                token_pairs[(str(row["execution_id"]), str(request["request_id"]))][str(row["role"])] = str(
                    request["generated_token_ids_sha256"]
                )
        for (execution_id, request_id), tokens in token_pairs.items():
            if set(tokens) != {"riley", "competitor"}:
                semantic_failures.append(
                    f"{execution_id}: request {request_id!r} lacks a two-lane token comparison"
                )
            elif tokens["riley"] != tokens["competitor"]:
                semantic_failures.append(
                    f"{execution_id}: request {request_id!r} generated token hash mismatch"
                )

        # Failed invocations and token-routing errors are never allowed into
        # percentile calculations.  Their presence is a valid, comparable
        # campaign outcome, but it is categorically a failed one.
        if semantic_failures:
            return {
                "schema_version": REPORT_SCHEMA_VERSION,
                "campaign_id": campaign_id,
                "campaign_plan_sha256": plan_sha256,
                "contract_sha256": contract_sha256,
                "status": "failed",
                "comparability": {"comparable": True, "reasons": []},
                "evidence": {
                    "expected_invocations": len(expected_invocations),
                    "received_invocations": len(received_invocations),
                    "environment": {
                        key: next(iter(environment_fingerprints))[index]
                        for index, key in enumerate(common_environment_keys)
                    },
                    "semantic_failures": sorted(semantic_failures),
                },
                "cells": [],
                "m4": {"evaluated": False, "passed": False, "failures": ["semantic correctness failure"]},
                "m5": {"evaluated": False, "passed": False, "failures": ["semantic correctness failure"], "geometric_mean_ratios": {}},
            }

        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows_by_invocation.values():
            if row["status"] == "success":
                grouped[(str(row["cell_id"]), str(row["role"]))].append(row)
        cell_reports: list[dict[str, Any]] = []
        for cell_id in sorted(cells_by_id):
            riley_rows = grouped[(cell_id, "riley")]
            competitor_rows = grouped[(cell_id, "competitor")]
            if not riley_rows or not competitor_rows:
                raise ComparabilityError(f"cell {cell_id!r} lacks successful observations for one lane")
            measurement_requirements = contract.get("measurement_requirements", {})
            percentile_method = (
                measurement_requirements.get("percentile_method", "r7")
                if isinstance(measurement_requirements, dict)
                else "r7"
            )
            if percentile_method not in {"r7", "nearest-rank"}:
                raise ContractError("contract measurement percentile_method is unsupported")
            expected_request_ids = {
                str(request["request_id"])
                for request in _array(
                    workloads_by_cell[cell_id]["value"]["requests"],
                    f"plan workload {cell_id!r}.requests",
                )
            }
            expected_run_indices = set(range(1, int(contract["required_independent_runs"]) + 1))
            riley = _aggregate_lane_cell(
                riley_rows,
                percentile_method=percentile_method,
                expected_request_ids=expected_request_ids,
                expected_run_indices=expected_run_indices,
            )
            competitor = _aggregate_lane_cell(
                competitor_rows,
                percentile_method=percentile_method,
                expected_request_ids=expected_request_ids,
                expected_run_indices=expected_run_indices,
            )
            cell_entry = cells_by_id[cell_id]
            cell_reports.append(
                {
                    "cell_id": cell_id,
                    "matrix_id": cell_entry["matrix_id"],
                    "matrix_tier": cell_entry["matrix_tier"],
                    "required_for": sorted(_cell_required_for(cell_entry)),
                    "primary": cell_entry["cell"].get("primary") is True,
                    "riley": riley,
                    "competitor": competitor,
                    "ratios": {
                        "ttft_p95": _ratio(riley["ttft_p95_ms"], competitor["ttft_p95_ms"], f"{cell_id} TTFT"),
                        "tpot_p95": _ratio(riley["tpot_p95_ms"], competitor["tpot_p95_ms"], f"{cell_id} TPOT"),
                        "slo_goodput": _ratio(
                            riley["slo_goodput_tokens_per_second_median"],
                            competitor["slo_goodput_tokens_per_second_median"],
                            f"{cell_id} SLO goodput",
                        ),
                        "peak_vram": _ratio(riley["peak_vram_bytes"], competitor["peak_vram_bytes"], f"{cell_id} peak VRAM"),
                    },
                }
            )

        m4_passed, m4_failures = _assess_m4(contract, cell_reports)
        m5_passed, m5_failures, m5_geometric = _assess_m5(contract, cell_reports)
        no_required_cells = m4_passed is None or m5_passed is None
        if no_required_cells:
            status = "incomparable"
            comparability = {
                "comparable": False,
                "reasons": sorted(m4_failures + m5_failures),
            }
        elif semantic_failures:
            status = "failed"
            comparability = {"comparable": True, "reasons": []}
        elif m4_passed and m5_passed:
            status = "passed"
            comparability = {"comparable": True, "reasons": []}
        elif _partial_win(cell_reports, contract):
            status = "partial-win"
            comparability = {"comparable": True, "reasons": []}
        else:
            status = "failed"
            comparability = {"comparable": True, "reasons": []}
        assert status in FINAL_STATUSES
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "campaign_plan_sha256": plan_sha256,
            "contract_sha256": contract_sha256,
            "status": status,
            "comparability": comparability,
            "evidence": {
                "expected_invocations": len(expected_invocations),
                "received_invocations": len(received_invocations),
                "environment": {
                    key: next(iter(environment_fingerprints))[index]
                    for index, key in enumerate(common_environment_keys)
                },
                "semantic_failures": sorted(semantic_failures),
            },
            "cells": cell_reports,
            "m4": {
                "evaluated": m4_passed is not None,
                "passed": m4_passed,
                "failures": sorted(m4_failures),
            },
            "m5": {
                "evaluated": m5_passed is not None,
                "passed": m5_passed,
                "failures": sorted(m5_failures),
                "geometric_mean_ratios": m5_geometric,
            },
        }
    except (ContractError, ComparabilityError) as error:
        return _incomparable_report(
            campaign_id=campaign_id,
            plan_sha256=plan_sha256,
            contract_sha256=contract_sha256,
            reasons=(str(error),),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = check_campaign(
        plan_path=arguments.plan.absolute(),
        raw_paths=[path.absolute() for path in arguments.raw],
        root=arguments.repo_root.resolve(),
    )
    encoded = canonical_json_bytes(report)
    if arguments.output is not None:
        try:
            output_path = require_campaign_artifact_path(
                arguments.repo_root.resolve(),
                arguments.output.absolute(),
                "claim report output",
            )
            create_only_write(output_path, encoded)
        except ContractError as error:
            print(f"check_campaign: {error}", file=sys.stderr)
            return 2
    print(encoded.decode("utf-8"), end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
