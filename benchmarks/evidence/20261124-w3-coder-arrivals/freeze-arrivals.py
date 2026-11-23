"""Freeze both engines' observed Coder-14B RAG write schedules from matching receipts."""
import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

GPUS = (6, 7)
ORIGINS = ("riley", "vllm")
SEEDS = (1, 2)
FIELDS = ("request_index", "input_sha256", "prompt_tokens", "completion_tokens", "finish")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path):
    return json.loads(path.read_bytes())


def percentile(values, q):
    return sorted(values)[min(int(len(values) * q), len(values) - 1)]


def freeze_arm(arm, spec, seed):
    name = f"rag-c8-s{seed}"
    receipt = read(arm / "receipt.json")
    data = (arm / f"{name}.json").read_bytes()
    digest = sha256(data).hexdigest()
    require(receipt["status"] == "passed" and receipt["results"][name] == digest,
            f"{arm.name} {name} does not match the original receipt")
    result = json.loads(data)
    stats, rows = (result[key] for key in ("statistics", "measured"))
    require(stats["config"] == spec and len(rows) == stats["ok"] == spec["requests"] == 64
            and stats["warmup_ok"] == spec["warmup"] == 8 and stats["errors"] == stats["rejected_429"] == 0,
            f"{arm.name} {name} does not contain the full cohort")
    require([row["request_index"] for row in rows] == list(range(8, 72)),
            f"{arm.name} {name} is missing request indexes")
    for row in rows:
        require(all(isfinite(row[key]) for key in ("start", "sent_at", "ttft", "e2e"))
                and row["start"] <= row["sent_at"] <= row["start"] + row["ttft"] <= row["start"] + row["e2e"],
                "invalid write or completion timestamp")
    epoch = min(row["sent_at"] for row in rows)
    records = [{**{key: row[key] for key in FIELDS}, "offset_ns": round((row["sent_at"] - epoch) * 1e9)}
               for row in rows]
    pending = [sum(other["sent_at"] < row["sent_at"] < other["start"] + other["e2e"]
                   for other in rows) for row in rows]
    observed = {
        "span_s": max(row["offset_ns"] for row in records) / 1e9,
        "pending_measured_at_write_p50": median(pending),
        "pending_measured_at_write_peak": max(pending),
        "ttft_p95_ms": stats["ttft_p95_ms"],
    }
    return {"source_sha256": digest, "requests": records, "observed": observed}


def freeze(source):
    cells = read(source / "cells.json")
    schedules, comparisons = {}, {}
    for gpu in GPUS:
        by_origin = {}
        metrics = {}
        for seed in SEEDS:
            name = f"rag-c8-s{seed}"
            spec = cells[name]
            require(spec["seed"] == seed and spec["concurrency"] == 8 and spec["max_tokens"] == 128
                    and spec["prompt_words"] == [1500, 3000], "the 14B reference workload changed")
            riley, vllm = [freeze_arm(source / f"gpu{gpu}/coder14-{engine}", spec, seed) for engine in ORIGINS]
            require(all(all(a[key] == b[key] for key in FIELDS)
                        for a, b in zip(riley["requests"], vllm["requests"], strict=True)),
                    f"engines did not serve the same inputs and token totals on GPU {gpu} {name}")
            deltas_ms = [(a["offset_ns"] - b["offset_ns"]) / 1e6
                         for a, b in zip(riley["requests"], vllm["requests"], strict=True)]
            metrics[name] = {"origin_statistics": {"riley": riley["observed"], "vllm": vllm["observed"]},
                             "riley_minus_vllm_write_ms_p50": median(deltas_ms),
                             "absolute_write_ms_p95": percentile([abs(value) for value in deltas_ms], .95)}
            for origin, arm in zip(ORIGINS, (riley, vllm), strict=True):
                by_origin[f"{name}-{origin}-arrivals"] = {
                    "cell": spec, "requests": spec["requests"], "origin": origin,
                    "source_sha256": arm["source_sha256"], "schedule": arm["requests"],
                }
        schedules[str(gpu)] = by_origin
        comparisons[str(gpu)] = metrics
    return {"schedules": schedules, "paired_origin_differences": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source), indent=2, allow_nan=False))
