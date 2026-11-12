#!/usr/bin/env python3
"""stage1t256_table.py: rag cells Riley/vLLM (vLLM from base0) for the two-KV-head models, lp13 with 512 (stage1t) -> lp15 with 256 (stage1t256). Run in the evidence dir."""
import glob, json
cols = [("rag-c1", "ttft_p50_ms"), ("rag6k-c1", "ttft_p50_ms"), ("rag-c8", "ttft_p95_ms"), ("rag-r3", "ttft_p95_ms"), ("rag-c8", "tpot_p50_ms"), ("rag-c8", "output_tok_s")]
def m(p, k):
    v = [json.load(open(x))[k] for x in glob.glob(p)]; return sum(v) / len(v)
print("| model | rag-c1 TTFT50 | rag6k TTFT50 | rag-c8 TTFT95 | rag-r3 TTFT95 | rag-c8 TPOT50 | rag-c8 tok/s |\n|---|---:|---:|---:|---:|---:|---:|")
for mo in sorted(x.split("/")[1] for x in glob.glob("stage1t256/*/")):
    c = []
    for s, k in cols:
        v = m(f"base0/{mo}/v[0-9]-{s}.json", k)
        c.append(f"{m(f'stage1t/{mo}/r[0-9]-{s}.json', k)/v:.2f} → {m(f'stage1t256/{mo}/r[0-9]-{s}.json', k)/v:.2f}")
    print(f"| {mo} | " + " | ".join(c) + " |")
