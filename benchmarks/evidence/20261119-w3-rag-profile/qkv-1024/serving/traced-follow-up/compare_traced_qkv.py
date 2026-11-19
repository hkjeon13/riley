"""Compare matched request stages in the QKV follow-up ABBA trace."""

from argparse import ArgumentParser
import json
from pathlib import Path

from compare_profile_arms import arm, digest, read, require
from summarize_profile_arm import stable_server
from summarize_request_stages import prefill_union, read_phases, request_phases


ORDER = ("riley-first", "qkv-first", "qkv-second", "riley-second")
CELLS = ("rag-c8-s2", "wave-50ms")
FIELDS = ("request_index", "input_sha256", "prompt_tokens", "output_sha256", "completion_tokens", "finish")
QKV = "RILEY_W3_QKV1024"


def read_arm(root, previous, mode):
    directory = root / f"qwen14-{mode}"
    receipt, plan, workloads = arm(root, "qwen14", mode, expected_cells=CELLS)
    env = {"RILEY_W3_REQUEST_TRACE": "1"}
    enabled = mode.startswith("qkv-")
    if enabled:
        env[QKV] = "1"
    require(plan["env"] == env and not plan["check_token_ids"], f"{mode}: wrong trace switches")
    before = read(root / f"idle-before-{mode}.json")
    after = read(directory / "idle-after.json")
    require(all(before[key] == after[key] for key in before if key != "updated_ns"), f"{mode}: lease changed")
    require(read(directory / "server-exit.json")["exit"] == 0, f"{mode}: shutdown failed")
    log = directory / "server.log"
    markers = [line for line in log.read_text().splitlines() if line.startswith(QKV)]
    require(bool(markers) == enabled and all(line == f"{QKV} recorded=1 cuda=120405" for line in markers),
            f"{mode}: candidate selection did not match the control")
    phases = read_phases(log)
    all_prefill = prefill_union(phases)
    binding = {
        "sha256": {Path(key).name: value for key, value in receipt["sha256"].items()
                   if Path(key).name != "plan.json"},
        "server": stable_server(plan["server"]),
        "cells": plan["cells"],
        "identity": {key: receipt[key] for key in ("model", "common_kv_blocks", "kv_cache_bytes",
                                                    "gpu", "holder_pid", "holder_start_ticks")},
    }
    timings = {}
    seen = set()
    for name, (stats, rows) in workloads.items():
        original_file = previous / f"qwen14-{mode}" / f"{name}.json"
        original = read(original_file)
        require(original["statistics"]["config"] == stats["config"], f"{mode}/{name}: workload changed")
        require(all(all(row[key] == old[key] for key in FIELDS)
                    for row, old in zip(rows, original["measured"], strict=True)), f"{mode}/{name}: streams changed")
        by_index = {}
        for row in rows:
            index, request = row["request_index"], row["request_id"]
            require(request in phases and request not in seen, f"{mode}/{name}: trace missing or reused")
            seen.add(request)
            by_index[index] = {"client_ttft_ms": row["ttft"] * 1000, "prompt_tokens": row["prompt_tokens"],
                               **request_phases(phases[request], row["prompt_tokens"], all_prefill)}
        slow = [index for index, row in by_index.items() if row["client_ttft_ms"] >= stats["ttft_p95_ms"]]
        require(bool(slow), f"{mode}/{name}: p95 does not cover any measured request")
        timings[name] = {"rows": by_index, "ttft_p95_ms": stats["ttft_p95_ms"], "tail_request_indices": slow,
                         "complete_streams_equal_to_uninstrumented": len(rows),
                         "input_capture_sha256": digest(original_file), "traced_data_sha256": digest(directory / f"{name}.json")}
    return binding, timings, digest(log)


def compare(root, previous):
    require((root / "campaign.exit").read_text().strip() == "0", "campaign did not finish")
    starts = [line.split()[1] for line in (root / "campaign.log").read_text().splitlines() if line.startswith("START ")]
    require(starts == list(ORDER), "unexpected campaign order")
    runs = {mode: read_arm(root, previous, mode) for mode in ORDER}
    binding = runs[ORDER[0]][0]
    require(all(run[0] == binding for run in runs.values()), "the paired run inputs differ")
    require((root / "binary.sha256").read_text().split() == [binding["sha256"]["riley"], "target/release/riley"],
            "unexpected binary checksum")
    cells = {}
    for name in CELLS:
        target = {41, 42} if name == "wave-50ms" else set()
        for _, timings, _ in runs.values():
            target.update(timings[name]["tail_request_indices"])
        results = {}
        for mode, (_, timings, _) in runs.items():
            fields = dict(timings[name])
            fields["rows"] = {key: fields["rows"][key] for key in sorted(target)}
            results[mode] = fields
        cells[name] = {"focused_requests": sorted(target), "arms": results}
    return {"scope": "Four instrumented arms for stage diagnosis. Serving guardrail results are from the separate uninstrumented ABBA.",
            "order": ORDER, "physical_gpu": 7, "model": binding["identity"]["model"],
            "kv_tokens": binding["identity"]["common_kv_blocks"] * 16, "inputs_sha256": binding["sha256"],
            "same_20gib_holder_restored_each_arm": True, "server_logs_sha256": {mode: run[2] for mode, run in runs.items()},
            "cells": cells}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.root, args.previous)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
