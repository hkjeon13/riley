#!/usr/bin/env python3
"""kvb.py MODEL_DIR: --kv-blocks for an 8,192-token server (vLLM-matched share above 20 GiB of weights), or nothing."""
import json, os, sys
p = sys.argv[1]; c = json.load(open(f"{p}/config.json")); h = c["num_attention_heads"]
d = c.get("head_dim") or c["hidden_size"] // h
per = c["num_hidden_layers"] * c.get("num_key_value_heads", h) * d * 4
w = sum(os.path.getsize(f"{p}/{f}") for f in os.listdir(p) if f.endswith(".safetensors"))
if w > 40 * 2**30: print("SKIP"); sys.exit()
b = int(((0.9 * 39.4 * 2**30 - w - 2.5 * 2**30) if w > 20 * 2**30 else (0.85 * 39.5 * 2**30 - w - 4 * 2**30)) // (16 * per))
print("" if b >= 16384 else f"--kv-blocks {b}")
