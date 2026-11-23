"""Recompute the paired Coder-14B performance ratios from request receipts."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re

GPUS = {
    6: "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7",
    7: "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0",
}
ORDER = {6: ["riley", "vllm"], 7: ["vllm", "riley"]}
IDENTITY = ("request_index", "input_sha256", "prompt_tokens", "completion_tokens", "finish")
TARGETS = {
    "closed-c32": ("tpot_p50_ms", 1.03, 1.03),
    "rag-c1": ("ttft_p50_ms", 1.15, 1.05),
    "rag-c8-s1": ("ttft_p95_ms", 1.4, 1.15),
    "rag-c8-s2": ("ttft_p95_ms", 1.4, 1.15),
}
CHECKPOINT = "82a56cdab6e16e31478b7449be768dce15992b448d543fea4b2f2ed163237d00"
IMAGE = "sha256:3278fe2fced2c3edadb2a9348c46a135fdcfd81da5f72ebfc9a55dede0ccfa42"
REVISION = "c5ec7b3918e18428b0c5704b010aa4ee51972a86"
BINARY = "c5edc27544706cdb5baf988920e3cb838ff98a3c1a6520285539628cbdc88f0c"
CPU = "753ae255fb829f54979b95f5f734a96113a76152bc505e2819ca2460e584eb9b"
METRICS = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "output_tok_s")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def reservations(stage, index):
    rows = [read(stage / f"gpu{index}/{phase}.json") for phase in ("before", "between", "after")]
    before = rows[0]
    for row in rows:
        require(row["status"] == "running" and row["phase"] == "idle" and row["gpu"] == GPUS[index]
                and row["physical_index"] == index and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
                and row["other_processes"] == []
                and (row["pid"], row["start_ticks"]) == (before["pid"], before["start_ticks"]),
                f"invalid GPU {index} reservation")
    return before


def confirm_order(stage):
    events = re.findall(r"^(START|END) gpu([67]) (riley|vllm) (\S+?)(?: exit=(\d+))?$",
                        (stage / "campaign.log").read_text(), re.MULTILINE)
    require(len(events) == 8, "expected four starts and four endings")
    timestamps = {}
    for phase, index, engine, timestamp, exit_code in events:
        require(exit_code == ("0" if phase == "END" else ""), "one arm failed")
        key = (int(index), engine, phase)
        require(key not in timestamps, f"duplicate event {key}")
        timestamps[key] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    for index, (first, second) in ORDER.items():
        require(timestamps[index, first, "START"] <= timestamps[index, first, "END"]
                <= timestamps[index, second, "START"] <= timestamps[index, second, "END"],
                f"GPU {index} run order changed")


def records(out, name, spec, receipt):
    path = out / f"{name}.json"
    require(digest(path) == receipt["results"][name], f"request artifact changed in {out.name}/{name}")
    cell = read(path)
    rows, stats = cell["measured"], cell["statistics"]
    n = spec["requests"]
    require(len(rows) == stats["ok"] == stats["requests"] == n and stats["warmup_ok"] == spec["warmup"]
            and stats["config"] == spec and stats["errors"] == stats["rejected_429"] == 0,
            f"incomplete measurements in {out.name}/{name}")
    require([row["request_index"] for row in rows] == list(range(spec["warmup"], spec["warmup"] + n)),
            f"invalid request indexes in {out.name}/{name}")

    def quantile(values, q):
        return sorted(values)[min(int(q * n), n - 1)]

    values = {"ttft": [], "tpot": []}
    for row in rows:
        require(all(re.fullmatch(r"[0-9a-f]{64}", row[key]) for key in ("input_sha256", "output_sha256"))
                and 0 < row["prompt_tokens"] < 8192 and 0 < row["completion_tokens"] <= spec["max_tokens"]
                and row["finish"] in {"stop", "length"} and 0 < row["ttft"] <= row["e2e"],
                f"invalid request {row['request_index']} in {name}")
        values["ttft"].append(row["ttft"] * 1000)
        values["tpot"].append((row["e2e"] - row["ttft"]) * 1000 / max(row["completion_tokens"] - 1, 1))
    actual = {f"{key}_{label}_ms": quantile(latencies, percentile)
              for key, latencies in values.items() for label, percentile in (("p50", .50), ("p95", .95))}
    actual["output_tok_s"] = sum(row["completion_tokens"] for row in rows) / stats["wall_s"]
    for key in METRICS:
        require(math.isclose(actual[key], stats[key], abs_tol=1e-9), f"unreproducible {key} in {out.name}/{name}")
    return rows, actual


def check_arm(stage, index, engine, cells, holder):
    out = stage / f"gpu{index}/coder14-{engine}"
    receipt = read(out / "receipt.json")
    require(receipt["status"] == "passed" and receipt["gpu"] == GPUS[index]
            and receipt["holder_pid"] == holder["pid"] and receipt["common_kv_blocks"] == 2576
            and receipt["kv_cache_bytes"] == 8103395328 and read(out / "plan.json")["cells"] == cells,
            f"invalid GPU {index} {engine} receipt")
    if engine == "riley":
        require((receipt["image_id"], receipt["source_revision"], receipt["riley_binary_sha256"],
                 receipt["cpu_report_sha256"], receipt["checkpoint_sha256"]) ==
                (IMAGE, REVISION, BINARY, CPU, CHECKPOINT), "Riley package provenance changed")
        require(receipt["shutdown_exit"] == 0, "Riley did not exit cleanly")
        shutdown = read(out / "shutdown.json")
        require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                and shutdown["allocation"] and all(value == 0 for value in shutdown["allocation"].values()),
                "Riley retained resources after shutdown")
    else:
        require(receipt["holder_start_ticks"] == holder["start_ticks"] and receipt["kv_capacity_matches"]
                and receipt["reported_kv_tokens"] == [41216], "vLLM cache budget was different")
        require(read(out / "server-exit.json")["exit"] in (0, -15), "vLLM did not exit cleanly")
    return {name: records(out, name, spec, receipt) for name, spec in cells.items()}


def pair(left, right, name):
    left_rows, left_stats = left
    right_rows, right_stats = right
    differences = {key: sum(a[key] != b[key] for a, b in zip(left_rows, right_rows)) for key in IDENTITY}
    require(not differences["request_index"] and not differences["input_sha256"],
            f"engine inputs differ in {name}")
    ratio = {key: left_stats[key] / right_stats[key] for key in METRICS}
    field, stage_one, final = TARGETS[name]
    matched = not any(differences.values())
    return {"requests_per_engine": len(left_rows), "request_differences": differences,
            "matched_workload": matched,
            "token_ranges": {engine: {field: [min(row[field] for row in rows), max(row[field] for row in rows)]
                                     for field in ("prompt_tokens", "completion_tokens")}
                             for engine, rows in (("riley", left_rows), ("vllm", right_rows))},
            "identical_output_text": sum(a["output_sha256"] == b["output_sha256"]
                                         for a, b in zip(left_rows, right_rows)),
            "riley": left_stats, "vllm": right_stats, "riley_over_vllm": ratio,
            "goal": {"metric": field, "stage": stage_one, "final": final,
                     "stage_met": ratio[field] <= stage_one if matched else None,
                     "final_met": ratio[field] <= final if matched else None}}


def verify(stage):
    require((stage / "campaign.exit").read_text().strip() == "0", "campaign failed")
    require((stage / "vllm-version.txt").read_text().strip() == "0.27.1+cu129", "vLLM version changed")
    cells = read(stage / "cells.json")
    require(set(cells) == set(TARGETS), "missing comparison cells")
    confirm_order(stage)
    holders = {index: reservations(stage, index) for index in ORDER}
    arms = {index: {engine: check_arm(stage, index, engine, cells, holders[index]) for engine in order}
            for index, order in ORDER.items()}
    result = {"image": IMAGE, "source_revision": REVISION, "checkpoint_sha256": CHECKPOINT,
              "vllm_version": "0.27.1+cu129", "run_order": ORDER, "by_gpu": {}, "within_engine_output_parity": {}}
    for index, engines in arms.items():
        result["by_gpu"][index] = {name: pair(engines["riley"][name], engines["vllm"][name], name) for name in cells}
    for engine in ("riley", "vllm"):
        counts = {}
        for name in cells:
            first, _ = arms[6][engine][name]
            second, _ = arms[7][engine][name]
            counts[name] = sum(all(a[key] == b[key] for key in (*IDENTITY, "output_sha256"))
                               for a, b in zip(first, second))
        result["within_engine_output_parity"][engine] = counts
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage), indent=2, allow_nan=False))
