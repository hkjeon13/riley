#!/usr/bin/env python3
"""kvfree.py MODEL_DIR GPU: --kv-blocks from the GPU memory free now (other tenants present), at most 16,384."""
import json, os, subprocess, sys
p, g = sys.argv[1], sys.argv[2]; c = json.load(open(f"{p}/config.json")); h = c["num_attention_heads"]
d = c.get("head_dim") or c["hidden_size"] // h
per = c["num_hidden_layers"] * c.get("num_key_value_heads", h) * d * 4
w = sum(os.path.getsize(f"{p}/{f}") for f in os.listdir(p) if f.endswith(".safetensors"))
free = int(subprocess.check_output(["nvidia-smi", "-i", g, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0]) * 2**20
b = int((0.85 * free - w - 3 * 2**30) // (16 * per))
print(f"--kv-blocks {min(b, 16384)}")
