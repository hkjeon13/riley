#!/usr/bin/env python3
"""val256_table.py: Riley/vLLM for the two-KV-head models with the 256 default (val16/speed), 512 values from val512/speed in brackets (run in the evidence dir)."""
import glob, json, os
cols = [("reason-c8", "output_tok_s"), ("reason-c32", "output_tok_s"), ("reason-c8", "tpot_p50_ms"), ("reason-c32", "tpot_p50_ms"), ("rag-c1", "ttft_p50_ms")]
def m(p, k):
    v = [json.load(open(x)).get(k) for x in p]; v = [a for a in v if a is not None]; return sum(v) / len(v) if v else None
def ratio(root, mo, s, k):
    r = m(glob.glob(f"{root}/{mo}/r[0-9]-{s}.json"), k); v = m(glob.glob(f"{root}/{mo}/v[0-9]-{s}.json"), k)
    return f"{r/v:.3f}" if r and v else "–"
print("Prompt rows 512, generated rows 256 on two-KV-head models (lp16) against vLLM 0.27.1, order rvvr (2026-11-08; GPU 6 for the Coder models, GPU 7 for 0.5B, earlier runs on either). 512 (lp12, `val512/speed`) in brackets.")
print("Throughput Riley/vLLM (higher is better); TPOT and TTFT p50 Riley/vLLM (lower is better).\n")
print("| model | reason-c8 tok/s | reason-c32 tok/s | reason-c8 TPOT | reason-c32 TPOT | rag-c1 TTFT | errors |")
print("|---|---:|---:|---:|---:|---:|---:|")
for mo in sorted(os.listdir("val16/speed")):
    c = [f"{ratio('val16/speed', mo, s, k)} [{ratio('val512/speed', mo, s, k)}]" for s, k in cols]
    e = sum(json.load(open(p)).get("errors", 0) for p in glob.glob(f"val16/speed/{mo}/[rv][0-9]-*.json"))
    print(f"| {mo} | " + " | ".join(c) + f" | {e} |")
