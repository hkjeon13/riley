#!/usr/bin/env python3
"""Average compare.py repetitions into a Riley / vLLM table. usage: summarize.py DIR"""
import glob
import json
import os
import sys

root = sys.argv[1]
rows = {}
for path in glob.glob(f"{root}/*.json"):
    name = os.path.basename(path)[:-5]
    tag, _, scenario = name.partition("-")
    if tag[0] not in "rv" or not tag[1:].isdigit():
        continue
    rows.setdefault(scenario, {}).setdefault(tag[0], []).append(json.load(open(path)))


def mean(entries, key):
    values = [entry.get(key) for entry in entries if entry.get(key) is not None]
    return sum(values) / len(values) if values else float("nan")


keys = [("output_tok_s", "tok/s"), ("ttft_p50_ms", "TTFT50"), ("ttft_p95_ms", "TTFT95"), ("ttft_p99_ms", "TTFT99"),
        ("tpot_p50_ms", "TPOT50"), ("itl_p99_ms", "ITL99")]
print("| scenario | " + " | ".join(label for _, label in keys) + " | errors | reps |")
print("|---|" + "---:|" * (len(keys) + 2))
for scenario in sorted(rows):
    riley, vllm = rows[scenario].get("r", []), rows[scenario].get("v", [])
    cells = []
    for key, _ in keys:
        cells.append(f"{mean(riley, key):.1f} / {mean(vllm, key):.1f}")
    errors = f"{sum(e['errors'] for e in riley)} / {sum(e['errors'] for e in vllm)}"
    print(f"| {scenario} | " + " | ".join(cells) + f" | {errors} | {len(riley)}/{len(vllm)} |")
