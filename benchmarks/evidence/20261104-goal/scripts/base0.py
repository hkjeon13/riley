#!/usr/bin/env python3
"""Stage-0 baseline: long-prompt cells and logprobs on every model, one queue per GPU.

usage: base0.py ROOT BIN GPU MODEL[,MODEL...]
Each model waits until its download finished, then runs compare.py (order rvvr).
Riley gets --kv-blocks sized from config.json when 32 x 8192 tokens would not fit.
"""
import json
import os
import subprocess
import sys
import time

root, binary, gpu, models = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
SCEN = "rag-c1,rag-c8,rag-r3,rag6k-c1,logprobs-c8,logprobs-c32"
GIB = 1 << 30
for model in models:
    path = f"{root}/models/{model}"
    while not os.path.exists(f"{path}/riley-checkpoint.json"):
        time.sleep(30)
    label = f"base0/{model}"
    if os.path.exists(f"{root}/evidence/{label}/r2-rag6k-c1.json"):
        continue
    config = json.load(open(f"{path}/config.json"))
    heads = config["num_attention_heads"]
    head_dim = config.get("head_dim") or config["hidden_size"] // heads
    per_token = config["num_hidden_layers"] * config.get("num_key_value_heads", heads) * head_dim * 4
    weights = sum(os.path.getsize(f"{path}/{f}") for f in os.listdir(path) if f.endswith(".safetensors"))
    budget = 0.85 * 39.5 * GIB - weights - 4 * GIB
    blocks = min(32 * 8192 // 16, int(budget // (16 * per_token)))
    long_extra = "" if blocks >= 32 * 8192 // 16 else f"--kv-blocks {blocks}"
    vllm_memory = "0.9" if weights > 20 * GIB else "0.8"
    command = ["python3", f"{root}/src/benchmarks/live/compare.py", root, label, model, gpu, binary,
               "--order", "rvvr", "--scenarios", SCEN, "--port", str(18400 + int(gpu)),
               "--riley-long", long_extra, "--vllm-memory", vllm_memory]
    print(time.strftime("%H:%M:%S"), model, "kv", long_extra or "default", "vllm", vllm_memory, flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    print(time.strftime("%H:%M:%S"), model, "rc", result.returncode, (result.stdout + result.stderr)[-300:].replace("\n", " | "), flush=True)
print("BASE0_DONE", flush=True)
