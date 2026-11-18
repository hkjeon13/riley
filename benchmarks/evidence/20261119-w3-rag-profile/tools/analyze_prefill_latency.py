"""Pair RAG and wave latency observations by their receipt-bound request identities."""

from argparse import ArgumentParser
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import median

from compare_profile_arms import CELLS, arm, require


def tpot(row):
    return (row["e2e"] - row["ttft"]) * 1000 / max(row["completion_tokens"] - 1, 1)


def percentile(rows, quantile, metric):
    values = sorted(map(metric, rows))
    return values[min(int(len(values) * quantile), len(values) - 1)]


def paired_shifts(before, after, metric):
    changes = [metric(b) - metric(a) for a, b in zip(before, after, strict=True)]
    return {"median_change_ms": median(changes), "slower_requests": sum(delta > 0 for delta in changes)}


def ttft(row):
    return row["ttft"] * 1000


def send_span(rows):
    sent = [row["sent_at"] for row in rows]
    return (max(sent) - min(sent)) * 1000


def shifts(root, candidate_mode, physical_gpu):
    expected = (*CELLS, "short-c32")
    control, plan, before = arm(root, "qwen14", "riley", expected_cells=expected, physical_gpu=physical_gpu)
    candidate, candidate_plan, after = arm(root, "qwen14", candidate_mode, expected_cells=expected,
                                         physical_gpu=physical_gpu)
    require(plan["cells"] == candidate_plan["cells"], "serving workload changed")
    fields = ("request_index", "input_sha256", "prompt_tokens", "output_sha256", "finish", "completion_tokens")
    result = {}
    for name in ("rag-c8-s1", "rag-c8-s2", "wave-sync", "wave-50ms"):
        old_stat, old = before[name]
        new_stat, new = after[name]
        require(all(all(a[field] == b[field] for field in fields) for a, b in zip(old, new, strict=True)),
                f"{name}: request or output changed")
        for stats, rows in ((old_stat, old), (new_stat, new)):
            require(all(isfinite(row["start"]) and isfinite(row["sent_at"]) and isfinite(row["e2e"])
                        and row["start"] <= row["sent_at"] and 0 <= row["ttft"] <= row["e2e"]
                        and row["completion_tokens"] > 0 for row in rows), f"{name}: invalid request timings")
            require(isclose(stats["ttft_p95_ms"], percentile(rows, .95, ttft))
                    and isclose(stats["tpot_p50_ms"], percentile(rows, .5, tpot)),
                    f"{name}: request-derived latency differs from aggregate")
        result[name] = {
            "control_source_sha256": control["results"][name], "candidate_source_sha256": candidate["results"][name],
            "requests": len(old), "ttft": paired_shifts(old, new, ttft), "tpot": paired_shifts(old, new, tpot),
        }
        if name.startswith("rag-"):
            old_tail = {row["request_index"] for row in old if ttft(row) >= old_stat["ttft_p95_ms"]}
            new_tail = {row["request_index"] for row in new if ttft(row) >= new_stat["ttft_p95_ms"]}
            new_rows = [row for row in new if row["request_index"] in new_tail]
            old_rows = [row for row in old if row["request_index"] in new_tail]
            result[name]["candidate_ttft_tail"] = {
                "requests": len(new_tail), "also_in_control_tail": len(new_tail & old_tail),
                "paired_ttft": paired_shifts(old_rows, new_rows, ttft),
            }
            continue
        width = plan["cells"][name]["concurrency"]
        require(len(old) % width == plan["cells"][name]["warmup"] % width == 0, f"{name}: partial waves")
        waves = []
        for start in range(0, len(old), width):
            old_wave = old[start:start+width]
            new_wave = new[start:start+width]
            waves.append({"wave": start // width + 1, "control_send_span_ms": send_span(old_wave),
                          "candidate_send_span_ms": send_span(new_wave),
                          "tpot": paired_shifts(old_wave, new_wave, tpot)})
        result[name]["waves"] = waves
    return {"model": control["model"], "physical_gpu": physical_gpu, "candidate_mode": candidate_mode,
            "scope": "Request TTFT and TPOT derived from client SSE timestamps; token gaps were not retained per wave",
            "cells": result}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--candidate-mode", choices=("continuation", "finishing"), default="finishing")
    parser.add_argument("--physical-gpu", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(shifts(args.root, args.candidate_mode, args.physical_gpu), indent=2, allow_nan=False))
