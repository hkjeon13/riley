"""Validate paired schedule, output fingerprints, GPU ownership and performance."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

from fixed_w3 import BINARY_SHA256, ORDER, ORIGINS, SEEDS, baseline, cell_specs, checked_read, outputs, require
from load import percentile


METRICS = ("ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms", "output_tok_s")
HOLDERS = {6: (2431228, "6339570869"), 7: (2431237, "6339570875")}


def status(folder, index, phase):
    row, _ = checked_read(folder / f"gpu{index}-{phase}.json")
    pid, start = HOLDERS[index]
    require((row["pid"], row["start_ticks"], row["physical_index"], row["phase"]) == (pid, start, index, "idle")
            and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 1024**3 and not row["other_processes"],
            "reservation missing")
    return {key: row[key] for key in ("pid", "start_ticks", "physical_index", "phase", "allocation_bytes")}


def verify(stage, source):
    all_runs = {}
    arms = {}
    for arm in ORDER:
        folder = stage / f"qwen14-{arm}"
        original = source / f"qwen14-{arm}"
        receipt, receipt_sha = checked_read(folder / "receipt.json")
        prev, _ = checked_read(original / "receipt.json")
        require(receipt["status"] == "passed" and receipt["mode"] == arm and receipt["holder_pid"] == HOLDERS[7][0]
                and receipt["holder_start_ticks"] == HOLDERS[7][1] and receipt["common_kv_blocks"] == 2576
                and receipt["nvml_samples"] > 0 and receipt["model"] == prev["model"] and receipt["gpu"] == prev["gpu"],
                "arm does not match its original runner and lease")
        binary = str(source / "target/release/riley")
        require(receipt["sha256"][binary] == prev["sha256"][binary] == BINARY_SHA256, "different serving binary")
        plan, plan_hash = checked_read(folder / "plan.json")
        orig_plan, _ = checked_read(original / "plan.json")
        require(receipt["sha256"][str(folder / "plan.json")] == plan_hash, "new plan hash differs")
        original_args = [orig_plan["address"] if arg == plan["address"] else arg for arg in plan["server"]]
        require(original_args == orig_plan["server"] and plan["env"] == orig_plan["env"]
                and plan["address"].startswith("127.0.0.1:") and plan["check_token_ids"] and not plan["profile"],
                "serving configuration differs from original")
        server_exit, _ = checked_read(folder / "server-exit.json")
        require(server_exit["exit"] == 0, "server did not stop normally")
        probe, probe_sha = checked_read(folder / "token-id-probe.json")
        previous_probe, previous_probe_sha = checked_read(original / "token-id-probe.json")
        require(receipt["token_id_probe_sha256"] == probe_sha and prev["token_id_probe_sha256"] == previous_probe_sha
                and probe == previous_probe, "generated ID probe differs")
        holders = {str(index): status(folder, index, "after") for index in HOLDERS}
        require(all(holders[str(index)] == status(folder, index, "before") for index in HOLDERS), "holder not restored")
        expected_cells = cell_specs(source, arm)
        require(list(plan["cells"]) == list(expected_cells) and plan["cells"] == expected_cells
                and set(receipt["results"]) == set(expected_cells), "planned cohorts or order changed")
        arms[arm] = {"receipt_sha256": receipt_sha, "token_id_probe_sha256": probe_sha, "holders": holders}
        for cell, spec in expected_cells.items():
            result, result_sha = checked_read(folder / f"{cell}.json")
            require(result_sha == receipt["results"][cell], "result hash differs")
            reference, offsets, source_hash = baseline(Path(spec["schedule_origin"]), spec["seed"], spec)
            require(spec["source_receipt_sha256"] == source_hash, "source receipt differs")
            stats = result["statistics"]
            rows = result["measured"]
            require(stats["config"] == spec and stats["requests"] == stats["ok"] == len(reference)
                    and stats["warmup_ok"] == spec["warmup"] and stats["errors"] == stats["rejected_429"] == 0
                    and outputs(rows) == outputs(reference), "replay lacks matching inputs, outputs or successful requests")
            actual_first = min(row["sent_at"] for row in rows)
            errors = []
            for row in rows:
                require(row["scheduled_write"] == offsets[row["request_index"]], "replay's scheduled write differs")
                error = 1000 * (row["sent_at"] - actual_first - row["scheduled_write"])
                require(isfinite(error) and error == row["relative_write_error_ms"], "actual write timing differs")
                errors.append(error)
            actual_p95 = percentile([abs(error) for error in errors], .95)
            require(actual_p95 == stats["write_error_abs_p95_ms"] and
                    percentile([row["ttft"] * 1000 for row in rows], .95) == stats["ttft_p95_ms"], "replay p95 differs")
            metrics = {field: stats[field] for field in METRICS}
            require(all(isfinite(value) and value > 0 for value in metrics.values()), "invalid replay metric")
            all_runs[cell, arm] = {
                "metrics": {**metrics, "relative_write_error_absolute_p95_ms": actual_p95,
                            "requests": len(rows), "sha256": result_sha},
                "errors": errors,
            }

    cells = {}
    for seed in SEEDS:
        for origin in ORIGINS:
            cell = f"rag-c8-s{seed}-{origin}"
            arms_here = {arm: all_runs[cell, arm]["metrics"] for arm in ORDER}
            pairs = {}
            for order in ("first", "second"):
                riley, qkv = (all_runs[cell, f"{kind}-{order}"] for kind in ("riley", "qkv"))
                pairs[order] = {k: (qkv["metrics"][k] / riley["metrics"][k] - 1) * 100 for k in METRICS}
                p95 = percentile([abs(a - b) for a, b in zip(riley["errors"], qkv["errors"], strict=True)], .95)
                pairs[order]["paired_write_error_abs_p95_ms"] = p95
            cells[cell] = {"arms": arms_here, "pairs_candidate_percent_change": pairs,
                           "timing_valid": all(row["relative_write_error_absolute_p95_ms"] <= 5 for row in arms_here.values())
                           and all(row["paired_write_error_abs_p95_ms"] <= 5 for row in pairs.values())}
    return {"binary_sha256": BINARY_SHA256, "matched_measured_requests": sum(cell["requests"]
             for cell in (run["metrics"] for run in all_runs.values())), "arms": arms, "cells": cells}


if __name__ == "__main__":
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
