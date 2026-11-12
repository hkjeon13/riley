#!/usr/bin/env python3
"""val512_table.py: reason-c8/c32 Riley/vLLM table from val512/speed, 14B reruns from kv2600 in brackets (run in the evidence dir)."""
import glob, json, os
root = "val512/speed"
cols = [("reason-c8", "output_tok_s"), ("reason-c32", "output_tok_s"), ("reason-c8", "tpot_p50_ms"), ("reason-c32", "tpot_p50_ms")]
def m(p, k):
    v = [json.load(open(x)).get(k) for x in p]; v = [a for a in v if a is not None]; return sum(v) / len(v) if v else None
print("KV chunk 512 (lp12) against vLLM 0.27.1, GPU 6, order rvvr (2026-11-05). Throughput Riley/vLLM (higher is better); TPOT p50 Riley/vLLM (lower is better).")
print("14B rows in brackets: re-run on GPU 7 with the vLLM-matched KV share (`kv2600/`: 2,600 or 2,576 blocks instead of 1,859).\n")
print("| model | reason-c8 tok/s | reason-c32 tok/s | reason-c8 TPOT | reason-c32 TPOT | errors |")
print("|---|---:|---:|---:|---:|---:|")
for mo in sorted(os.listdir(root)):
    c = []; e = 0
    for s, k in cols:
        r = m(glob.glob(f"{root}/{mo}/r[0-9]-{s}.json"), k); v = m(glob.glob(f"{root}/{mo}/v[0-9]-{s}.json"), k)
        cell = f"{r/v:.3f}"
        alt = f"kv2600/{mo}"
        if glob.glob(f"{alt}/r[0-9]-{s}.json"):
            r2 = m(glob.glob(f"{alt}/r[0-9]-{s}.json"), k); v2 = m(glob.glob(f"{alt}/v[0-9]-{s}.json"), k)
            cell += f" [{r2/v2:.3f}]"
        c.append(cell)
    for p in glob.glob(f"{root}/{mo}/[rv][0-9]-*.json"): e += json.load(open(p)).get("errors", 0)
    print(f"| {mo} | " + " | ".join(c) + f" | {e} |")
