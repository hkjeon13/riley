#!/usr/bin/env python3
"""Per-model Riley / vLLM ratios of the goal's long-prompt and logprobs rows.

usage: gap_table.py BASE_DIR   (BASE_DIR/<model>/{r,v}<rep>-<scenario>.json from compare.py)
Latency ratios are Riley / vLLM (lower is better); throughput ratios are
vLLM / Riley, so for every column a value above 1 means Riley is behind.
Each engine's repetitions are averaged first.
"""
import glob
import json
import os
import sys

COLUMNS = [
    ("closed-c1", "tpot_p50_ms", "c1 TPOT"),
    ("closed-c8", "tpot_p50_ms", "c8 TPOT"),
    ("closed-c16", "tpot_p50_ms", "c16 TPOT"),
    ("closed-c32", "tpot_p50_ms", "c32 TPOT"),
    ("closed-c32", "output_tok_s", "c32 tok/s"),
    ("rag-c1", "ttft_p50_ms", "rag-c1 TTFT50"),
    ("rag6k-c1", "ttft_p50_ms", "rag6k TTFT50"),
    ("rag-c8", "ttft_p95_ms", "rag-c8 TTFT95"),
    ("rag-r3", "ttft_p95_ms", "rag-r3 TTFT95"),
    ("rag-c8", "tpot_p50_ms", "rag-c8 TPOT50"),
    ("rag-c8", "output_tok_s", "rag-c8 tok/s"),
    ("logprobs-c8", "output_tok_s", "logprobs-c8 tok/s"),
    ("logprobs-c32", "output_tok_s", "logprobs-c32 tok/s"),
]


def mean(paths, key):
    values = [json.load(open(path)).get(key) for path in paths]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


root = sys.argv[1]
print("| model | " + " | ".join(label for _, _, label in COLUMNS) + " | errors |")
print("|---|" + "---:|" * (len(COLUMNS) + 1))
for model in sorted(os.listdir(root)):
    cells, errors = [], 0
    for scenario, key, _ in COLUMNS:
        riley = glob.glob(f"{root}/{model}/r[0-9]-{scenario}.json")
        vllm = glob.glob(f"{root}/{model}/v[0-9]-{scenario}.json")
        r, v = mean(riley, key), mean(vllm, key)
        if r is None or v is None:
            cells.append("–")
            continue
        ratio = v / r if key == "output_tok_s" else r / v
        cells.append(f"{ratio:.2f}")
    for path in glob.glob(f"{root}/{model}/[rv][0-9]-*.json"):
        errors += json.load(open(path)).get("errors", 0)
    print(f"| {model} | " + " | ".join(cells) + f" | {errors} |")
