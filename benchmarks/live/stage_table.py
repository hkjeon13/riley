#!/usr/bin/env python3
"""Riley / vLLM ratios with Riley and vLLM from different runs of compare.py.

usage: stage_table.py RILEY_DIR VLLM_DIR [--before BEFORE_RILEY_DIR]
Each DIR holds <model>/{r,v}<rep>-<scenario>.json. Latency ratios are Riley /
vLLM, throughput ratios vLLM / Riley, so above 1 means Riley is behind. With
--before, each cell shows the earlier Riley run's ratio first.
"""
import argparse
import glob
import json
import os

COLUMNS = [
    ("rag-c1", "ttft_p50_ms", "rag-c1 TTFT50"),
    ("rag6k-c1", "ttft_p50_ms", "rag6k TTFT50"),
    ("rag-c8", "ttft_p95_ms", "rag-c8 TTFT95"),
    ("rag-r3", "ttft_p95_ms", "rag-r3 TTFT95"),
    ("rag-c8", "tpot_p50_ms", "rag-c8 TPOT50"),
    ("rag-c8", "output_tok_s", "rag-c8 tok/s"),
    ("logprobs-c8", "output_tok_s", "logprobs-c8 tok/s"),
    ("logprobs-c32", "output_tok_s", "logprobs-c32 tok/s"),
]
parser = argparse.ArgumentParser()
parser.add_argument("riley")
parser.add_argument("vllm")
parser.add_argument("--before")
args = parser.parse_args()


def mean(pattern, key):
    values = [json.load(open(path)).get(key) for path in glob.glob(pattern)]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def ratio(riley_dir, model, scenario, key):
    r = mean(f"{riley_dir}/{model}/r[0-9]-{scenario}.json", key)
    v = mean(f"{args.vllm}/{model}/v[0-9]-{scenario}.json", key)
    if r is None or v is None:
        return None
    return v / r if key == "output_tok_s" else r / v


print("| model | " + " | ".join(label for _, _, label in COLUMNS) + " |")
print("|---|" + "---:|" * len(COLUMNS))
for model in sorted(os.listdir(args.riley)):
    cells = []
    for scenario, key, _ in COLUMNS:
        now = ratio(args.riley, model, scenario, key)
        text = "–" if now is None else f"{now:.2f}"
        if args.before and now is not None:
            then = ratio(args.before, model, scenario, key)
            text = f"{then:.2f} → {now:.2f}" if then is not None else text
        cells.append(text)
    print(f"| {model} | " + " | ".join(cells) + " |")
