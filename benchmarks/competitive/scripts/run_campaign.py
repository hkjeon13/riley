#!/usr/bin/env python3
"""Create a closed, append-only C01 competitive campaign execution plan.

This program intentionally does *not* start either inference engine.  It
freezes the exact inputs and AB/BA schedule that an external lane adapter must
later execute.  Separating plan creation from execution prevents a benchmark
runner from silently changing model, prompt, environment, or lane options
mid-campaign.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from competitive_common import (
    CONTRACT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    ContractError,
    build_workload_identity,
    canonical_preflight_script_receipt,
    canonical_json_bytes,
    create_only_write,
    load_json,
    load_preflight_receipt,
    matrix_cells,
    normalise_sampling_identity,
    path_for_plan,
    request_sets_by_cell,
    require_canonical_contract,
    sha256_bytes,
    sha256_file,
    validate_contract,
    validate_identifier,
    validate_lane,
    validate_matrix,
    validate_request_manifest,
    verify_campaign_lane_binding,
    verify_campaign_matrix_binding,
    verify_campaign_request_binding,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parents[2]


def _git_output(root: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError(f"cannot inspect Git source state: {error}") from error
    return completed.stdout


def source_receipt(root: Path, *, allow_dirty_source: bool) -> dict[str, Any]:
    revision = _git_output(root, ("rev-parse", "HEAD")).strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ContractError("Git HEAD must be a full lowercase 40-hex revision")
    dirty = bool(_git_output(root, ("status", "--porcelain=v1")))
    if dirty and not allow_dirty_source:
        raise ContractError(
            "refusing to create a canonical campaign plan from a dirty source tree"
        )
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "development_only": dirty,
    }


def _load_contract(path: Path) -> Mapping[str, Any]:
    return validate_contract(load_json(path), str(path))


def _load_lane(path: Path) -> Mapping[str, Any]:
    return validate_lane(load_json(path), str(path))


def _load_matrix(path: Path) -> Mapping[str, Any]:
    return validate_matrix(load_json(path), str(path))


def _cell_catalog(root: Path, matrix_paths: Sequence[Path]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for matrix_path in matrix_paths:
        matrix = _load_matrix(matrix_path)
        for cell in matrix_cells(matrix):
            cell_id = str(cell["cell_id"])
            if cell_id in seen:
                raise ContractError(f"campaign matrices repeat cell_id {cell_id!r}")
            seen.add(cell_id)
            if not cell["executable"]:
                raise ContractError(
                    f"cell {cell_id!r} is a model-class template; supply a concrete "
                    "campaign matrix with a pinned model identity instead"
                )
            catalog.append(
                {
                    "matrix_id": matrix["matrix_id"],
                    "matrix_tier": matrix["tier"],
                    "matrix_path": path_for_plan(root, matrix_path),
                    "matrix_sha256": sha256_file(matrix_path),
                    "cell": cell,
                }
            )
    return catalog


def _validate_request_sets(
    cells: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    by_cell = request_sets_by_cell(manifest)
    expected = {str(entry["cell"]["cell_id"]) for entry in cells}
    actual = set(by_cell)
    if expected != actual:
        missing = sorted(expected - actual)
        surplus = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing request set(s): {', '.join(missing)}")
        if surplus:
            details.append(f"unexpected request set(s): {', '.join(surplus)}")
        raise ContractError("request manifest does not exactly cover campaign cells: " + "; ".join(details))

    workloads: dict[str, dict[str, Any]] = {}
    for entry in cells:
        cell = entry["cell"]
        cell_id = str(cell["cell_id"])
        requests = by_cell[cell_id]
        if "concurrency" in cell and len(requests) != cell["concurrency"]:
            raise ContractError(
                f"cell {cell_id!r} requires concurrency {cell['concurrency']}, "
                f"but request manifest has {len(requests)} requests"
            )
        for request in requests:
            for field in ("cache_policy", "eos_policy", "arrival_schedule_id"):
                if field in cell and request[field] != cell[field]:
                    raise ContractError(
                        f"cell {cell_id!r} {field} does not match request "
                        f"{request['request_id']!r}"
                    )
            if request["requested_output_tokens"] != cell["requested_output_tokens"]:
                raise ContractError(
                    f"cell {cell_id!r} requested_output_tokens does not match request "
                    f"{request['request_id']!r}"
                )
            if "prompt_tokens" in cell and request["prompt_tokens"] != cell["prompt_tokens"]:
                raise ContractError(
                    f"cell {cell_id!r} prompt_tokens does not match request "
                    f"{request['request_id']!r}"
                )
            cell_sampling = normalise_sampling_identity(
                cell["sampling"],
                f"cell {cell_id!r}.sampling",
                allow_unparameterized_seeded=True,
            )
            request_sampling = normalise_sampling_identity(
                request["sampling"],
                f"request {request['request_id']!r}.sampling",
            )
            if cell_sampling["id"] != request_sampling["id"]:
                raise ContractError(
                    f"cell {cell_id!r} sampling does not match request "
                    f"{request['request_id']!r}"
                )
        workloads[cell_id] = build_workload_identity(
            cell,
            requests,
            manifest,
            label=f"cell {cell_id!r} workload",
        )
    return workloads


def _normalise_created_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError("--created-at-utc must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ContractError("--created-at-utc must include a UTC offset")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_plan(
    *,
    root: Path,
    contract_path: Path,
    matrix_paths: Sequence[Path],
    riley_lane_path: Path,
    competitor_lane_path: Path,
    request_manifest_path: Path,
    preflight_receipt_path: Path | None,
    campaign_id: str,
    created_at_utc: str | None,
    allow_dirty_source: bool,
    require_executable_lanes: bool,
) -> dict[str, Any]:
    """Build, but do not write, a complete C01 campaign plan."""

    validate_identifier(campaign_id, "campaign_id")
    source = source_receipt(root, allow_dirty_source=allow_dirty_source)
    contract = _load_contract(contract_path)
    require_canonical_contract(root, contract_path, contract)
    riley_lane = _load_lane(riley_lane_path)
    competitor_lane = _load_lane(competitor_lane_path)
    verify_campaign_lane_binding(root, riley_lane_path, riley_lane)
    verify_campaign_lane_binding(root, competitor_lane_path, competitor_lane)
    if riley_lane["lane_id"] == competitor_lane["lane_id"]:
        raise ContractError("Riley and competitor lane IDs must differ")
    manifest = validate_request_manifest(load_json(request_manifest_path), str(request_manifest_path))
    verify_campaign_request_binding(root, manifest)
    for matrix_path in matrix_paths:
        matrix = _load_matrix(matrix_path)
        verify_campaign_matrix_binding(root, matrix_path, matrix)
    cells = _cell_catalog(root, matrix_paths)
    workloads = _validate_request_sets(cells, manifest)

    blocked_reasons: list[str] = []
    if preflight_receipt_path is None:
        preflight: dict[str, Any] = {
            "status": "missing",
            "path": None,
            "sha256": None,
            "values": None,
            "script": canonical_preflight_script_receipt(root),
        }
        blocked_reasons.append("no reviewed thermal/clock/GPU preflight receipt was supplied")
    else:
        preflight_values = load_preflight_receipt(preflight_receipt_path, source)
        preflight = {
            "status": "passed",
            "path": path_for_plan(root, preflight_receipt_path),
            "sha256": sha256_file(preflight_receipt_path),
            "values": preflight_values,
            "script": canonical_preflight_script_receipt(root),
        }
    for role, lane in (("riley", riley_lane), ("competitor", competitor_lane)):
        if lane["availability"] != "available" or lane["command"].get("status") != "available":
            blocked_reasons.append(
                f"{role} lane {lane['lane_id']!r} is {lane['availability']}/"
                f"{lane['command'].get('status')}, not executable"
            )
    if source["git_dirty"]:
        blocked_reasons.append("source tree was dirty when the development-only plan was created")
    if blocked_reasons and require_executable_lanes:
        raise ContractError("campaign cannot execute: " + "; ".join(blocked_reasons))

    invocations: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    sequence = 0
    invocation_sequence = 0
    required_runs = int(contract["required_independent_runs"])
    orders = list(contract["orders"])
    for cell_entry in cells:
        cell = cell_entry["cell"]
        cell_id = str(cell["cell_id"])
        for run_index in range(1, required_runs + 1):
            order = orders[(run_index - 1) % len(orders)]
            sequence += 1
            execution_id = f"{cell_id}:run-{run_index:02d}"
            roles = ("riley", "competitor") if order == "AB" else ("competitor", "riley")
            executions.append(
                {
                    "execution_id": execution_id,
                    "sequence": sequence,
                    "cell_id": cell_id,
                    "run_index": run_index,
                    "order": order,
                    "lane_order": list(roles),
                }
            )
            for position, role in zip(("A", "B"), roles, strict=True):
                invocation_sequence += 1
                lane = riley_lane if role == "riley" else competitor_lane
                invocations.append(
                    {
                        "invocation_id": f"{execution_id}:{position}",
                        "sequence": invocation_sequence,
                        "execution_id": execution_id,
                        "cell_id": cell_id,
                        "run_index": run_index,
                        "order": order,
                        "position": position,
                        "role": role,
                        "lane_id": lane["lane_id"],
                    }
                )

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "created_at_utc": _normalise_created_at(created_at_utc),
        "source": source,
        "contract": {
            "path": path_for_plan(root, contract_path),
            "sha256": sha256_file(contract_path),
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "contract_id": contract["contract_id"],
        },
        "matrices": [
            {
                "path": path_for_plan(root, matrix_path),
                "sha256": sha256_file(matrix_path),
                "matrix_id": _load_matrix(matrix_path)["matrix_id"],
            }
            for matrix_path in matrix_paths
        ],
        "lanes": {
            "riley": {
                "path": path_for_plan(root, riley_lane_path),
                "sha256": sha256_file(riley_lane_path),
                "lane_id": riley_lane["lane_id"],
                "availability": riley_lane["availability"],
                "command_status": riley_lane["command"]["status"],
                "pin_status": riley_lane["engine"].get("pin_status"),
            },
            "competitor": {
                "path": path_for_plan(root, competitor_lane_path),
                "sha256": sha256_file(competitor_lane_path),
                "lane_id": competitor_lane["lane_id"],
                "availability": competitor_lane["availability"],
                "command_status": competitor_lane["command"]["status"],
                "pin_status": competitor_lane["engine"].get("pin_status"),
            },
        },
        "request_manifest": {
            "path": path_for_plan(root, request_manifest_path),
            "sha256": sha256_file(request_manifest_path),
            "manifest_id": manifest["manifest_id"],
            "model_identity": manifest["model_identity"],
        },
        "workloads": [
            {
                "cell_id": cell_id,
                "sha256": sha256_bytes(canonical_json_bytes(workloads[cell_id])),
                "value": workloads[cell_id],
            }
            for cell_id in sorted(workloads)
        ],
        "preflight": preflight,
        "cells": cells,
        "execution": executions,
        "invocations": invocations,
        "readiness": {
            "state": "ready" if not blocked_reasons else "blocked",
            "blocked_reasons": blocked_reasons,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, action="append", required=True)
    parser.add_argument("--riley-lane", type=Path, required=True)
    parser.add_argument("--competitor-lane", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument(
        "--preflight-receipt",
        type=Path,
        help="successful key=value output from benchmarks/scripts/preflight.sh",
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at-utc")
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="create a development-only plan; never valid competitive evidence",
    )
    parser.add_argument(
        "--require-executable-lanes",
        action="store_true",
        help="fail instead of emitting a blocked plan when a lane is contract-only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        root = arguments.repo_root.resolve()
        plan = build_plan(
            root=root,
            contract_path=arguments.contract.resolve(),
            matrix_paths=[path.resolve() for path in arguments.matrix],
            riley_lane_path=arguments.riley_lane.resolve(),
            competitor_lane_path=arguments.competitor_lane.resolve(),
            request_manifest_path=arguments.request_manifest.resolve(),
            preflight_receipt_path=(
                arguments.preflight_receipt.resolve()
                if arguments.preflight_receipt is not None
                else None
            ),
            campaign_id=arguments.campaign_id,
            created_at_utc=arguments.created_at_utc,
            allow_dirty_source=arguments.allow_dirty_source,
            require_executable_lanes=arguments.require_executable_lanes,
        )
        create_only_write(arguments.output.resolve(), canonical_json_bytes(plan))
    except ContractError as error:
        print(f"run_campaign: {error}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(plan).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
