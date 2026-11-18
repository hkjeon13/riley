"""Compare uninstrumented 14B arms, retaining the capacity and workload checks."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


CELLS = ("rag-c1", "rag-c8-s1", "rag-c8-s2", "wave-sync", "wave-50ms")
METRICS = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms", "output_tok_s")
HOLDER_BYTES = 20 * 1024**3


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def arm(root, label, engine):
    directory = root / f"{label}-{engine}"
    plan_file = directory / "plan.json"
    plan = read(plan_file)
    receipt = read(directory / "receipt.json")
    reservation = read(directory / "idle-after.json")
    require(receipt["mode"] == engine and receipt["status"] == "passed", f"{engine}: incomplete arm")
    require(not plan["profile"] and tuple(plan["cells"]) == CELLS, f"{engine}: unexpected timing cells")
    hashes = {Path(path).name: checksum for path, checksum in receipt["sha256"].items()}
    require(hashes["plan.json"] == digest(plan_file), f"{engine}: plan digest mismatch")
    require(reservation["phase"] == "idle" and reservation["physical_index"] == 7
            and reservation["allocation_bytes"] == reservation["torch_reserved_bytes"] == HOLDER_BYTES
            and not reservation["other_processes"]
            and reservation["gpu"] == receipt["gpu"]
            and (reservation["pid"], reservation["start_ticks"])
            == (receipt["holder_pid"], receipt["holder_start_ticks"]), f"{engine}: reservation not restored")
    results = {}
    for name, workload in plan["cells"].items():
        path = directory / f"{name}.json"
        raw = read(path)
        stats, rows = raw["statistics"], raw["measured"]
        require(digest(path) == receipt["results"][name], f"{engine}/{name}: result digest mismatch")
        require(stats["config"] == workload and len(rows) == stats["ok"] == workload["requests"]
                and stats["warmup_ok"] == workload["warmup"] and stats["errors"] == stats["rejected_429"] == 0,
                f"{engine}/{name}: request coverage is incomplete")
        require([row["request_index"] for row in rows]
                == list(range(workload["warmup"], workload["warmup"] + workload["requests"])),
                f"{engine}/{name}: request order differs")
        results[name] = (stats, rows)
    return receipt, plan, results


def ratio_percent(new, old):
    return (new / old - 1) * 100 if old else None


def pair(root, label):
    riley, riley_plan, riley_results = arm(root, label, "riley")
    vllm, vllm_plan, vllm_results = arm(root, label, "vllm")
    require(riley["model"] == vllm["model"] and riley["gpu"] == vllm["gpu"]
            and riley_plan["cells"] == vllm_plan["cells"], "engine model, device or workload differs")
    shared = ("riley-checkpoint.json", "config.json", "tokenizer.json", "load.py", "profile_workload.py")
    inputs = [{Path(path).name: value for path, value in receipt["sha256"].items()} for receipt in (riley, vllm)]
    require(all(inputs[0][name] == inputs[1][name] for name in shared), "engine inputs differ")
    require(riley["common_kv_blocks"] == vllm["common_kv_blocks"]
            and riley["kv_cache_bytes"] == vllm["kv_cache_bytes"], "requested KV budgets differ")
    expected_tokens = riley["common_kv_blocks"] * 16
    capacity_matches = vllm["kv_capacity_matches"] and vllm["reported_kv_tokens"] == [expected_tokens]
    comparisons = {}
    for name in CELLS:
        rstats, rrows = riley_results[name]
        vstats, vrows = vllm_results[name]
        require([row["input_sha256"] for row in rrows] == [row["input_sha256"] for row in vrows],
                f"{name}: request bodies differ")
        comparisons[name] = {
            "requests": len(rrows),
            "warmup": rstats["warmup_ok"],
            "matching_prompt_token_counts": sum(a["prompt_tokens"] == b["prompt_tokens"]
                                                for a, b in zip(rrows, vrows)),
            "matching_output_text_and_finish": sum(
                (a["output_sha256"], a["finish"]) == (b["output_sha256"], b["finish"])
                for a, b in zip(rrows, vrows)),
            "riley": {key: rstats[key] for key in (*METRICS, "completion_tokens_mean")},
            "vllm": {key: vstats[key] for key in (*METRICS, "completion_tokens_mean")},
            "riley_relative_to_vllm_percent": {key: ratio_percent(rstats[key], vstats[key]) for key in METRICS},
        }
    waves, arrival_windows = {}, {}
    for name, results in (("riley", riley_results), ("vllm", vllm_results)):
        sync, sync_rows = results["wave-sync"]
        spaced, spaced_rows = results["wave-50ms"]
        require([row["input_sha256"] for row in sync_rows] == [row["input_sha256"] for row in spaced_rows],
                f"{name}: wave request bodies differ")
        waves[name] = {key: ratio_percent(spaced[key], sync[key]) for key in METRICS}
        arrival_windows[name] = {}
        for stats, rows in ((sync, sync_rows), (spaced, spaced_rows)):
            config = stats["config"]
            width = config["concurrency"]
            require(len(rows) % width == config["warmup"] % width == 0, f"{name}: incomplete waves")
            spans = []
            for start in range(0, len(rows), width):
                sent = [row["sent_at"] for row in rows[start:start + width]]
                spans.append((max(sent) - min(sent)) * 1_000)
            arrival_windows[name][str(config["stagger_ms"])] = {
                "waves": len(spans), "p50_send_span_ms": median(spans), "max_send_span_ms": max(spans),
            }
    return {"model": riley["model"], "physical_gpu": 7,
            "both_arms_requested_kv_tokens": expected_tokens, "vllm_reported_kv_tokens": vllm["reported_kv_tokens"],
            "kv_capacity_matches": capacity_matches,
            "cells": comparisons, "staggered_relative_to_sync_percent": waves,
            "socket_send_spans": arrival_windows}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("model", choices=("qwen14", "coder14"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(pair(args.root, args.model), file, indent=2, allow_nan=False)
        file.write("\n")
