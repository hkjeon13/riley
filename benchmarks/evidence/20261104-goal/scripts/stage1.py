#!/usr/bin/env python3
"""Stage 1: Riley BIN with the 1,024-row burst on every model (order rr), vLLM numbers from base0.

usage: stage1.py ROOT BIN GPU LABEL MODEL[,MODEL...] [--order rr]
"""
import json, os, subprocess, sys, time
root, binary, gpu, label, models = sys.argv[1:6]
order = sys.argv[7] if len(sys.argv) > 7 and sys.argv[6] == "--order" else "rr"
SCEN = "rag-c1,rag-c8,rag-r3,rag6k-c1,logprobs-c8,logprobs-c32"
BURST = ("--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets "
         "1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 "
         "--prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512")
GIB = 1 << 30
for model in models.split(","):
    path = f"{root}/models/{model}"
    out = f"{label}/{model}"
    if os.path.exists(f"{root}/evidence/{out}/{order[-1]}{order.count(order[-1])}-rag6k-c1.json"):
        continue
    c = json.load(open(f"{path}/config.json")); h = c["num_attention_heads"]
    d = c.get("head_dim") or c["hidden_size"] // h
    per = c["num_hidden_layers"] * c.get("num_key_value_heads", h) * d * 4
    w = sum(os.path.getsize(f"{path}/{f}") for f in os.listdir(path) if f.endswith(".safetensors"))
    budget = (0.9 * 39.4 * GIB - w - 2.5 * GIB) if w > 20 * GIB else (0.85 * 39.5 * GIB - w - 4 * GIB)
    blocks = min(16384, int(budget // (16 * per)))
    kv = "" if blocks >= 16384 else f" --kv-blocks {blocks}"
    vmem = "0.9" if w > 20 * GIB else "0.8"
    cmd = ["python3", f"{root}/src/benchmarks/live/compare.py", root, out, model, gpu, binary, "--order", order,
           "--scenarios", SCEN, "--port", str(18400 + int(gpu)), "--riley-long", BURST + kv, "--vllm-memory", vmem]
    print(time.strftime("%H:%M:%S"), model, kv or "kv default", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(time.strftime("%H:%M:%S"), model, "rc", r.returncode, (r.stdout + r.stderr)[-200:].replace("\n", " | "), flush=True)
print("STAGE1_DONE", flush=True)
