#!/usr/bin/env python3
"""full16_table.py: Riley/vLLM (order rvvr, same session) for the 18 models re-measured on 0.1.0 (lp16) in full16/. Run in the evidence dir."""
import glob, json, os
cols = [("closed-c8", "tpot_p50_ms"), ("closed-c16", "tpot_p50_ms"), ("closed-c32", "tpot_p50_ms"),
        ("closed-c32", "output_tok_s"), ("closed-c32", "ttft_p50_ms"),
        ("reason-c8", "output_tok_s"), ("reason-c32", "output_tok_s"), ("reason-c8", "tpot_p50_ms"), ("reason-c32", "tpot_p50_ms")]
def m(p, k):
    v = [json.load(open(x))[k] for x in glob.glob(p)]; return sum(v) / len(v)
print("| model | c8 TPOT | c16 TPOT | c32 TPOT | c32 tok/s | c32 TTFT | reason-c8 tok/s | reason-c32 tok/s | reason-c8 TPOT | reason-c32 TPOT | errors |")
print("|---|" + "---:|" * 10)
for mo in sorted(os.listdir("full16")):
    c = [f"{m(f'full16/{mo}/r[0-9]-{s}.json', k)/m(f'full16/{mo}/v[0-9]-{s}.json', k):.3f}" for s, k in cols]
    e = sum(json.load(open(p)).get("errors", 0) for p in glob.glob(f"full16/{mo}/r[0-9]-*.json"))
    print(f"| {mo} | " + " | ".join(c) + f" | {e} |")
