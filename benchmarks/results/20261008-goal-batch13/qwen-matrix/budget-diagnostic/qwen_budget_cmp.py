import json, statistics as st, os, sys
b = json.load(open("qwen-matrix-steady-rileyonly-lookahead-attempt01/result.json")); v = json.load(open("qwen-matrix-steady-attempt01/result.json")); vr = [r for r in v["records"] if r["engine"] == "vllm"]
base = {r["case"]: r for r in b["records"]}
for tag in sys.argv[1:]:
    p = "qwen-matrix-steady-%s/result.json" % tag
    if not os.path.exists(p): print(tag, "missing"); continue
    d = json.load(open(p)); print("==", tag, d["complete"], d["failure"])
    cells = []
    for r in d["records"]:
        if r["cell"] not in cells: cells.append(r["cell"])
    for c in cells:
        f = lambda rs, k: st.mean(r[k] for r in rs if r["cell"] == c)
        same = all(r["outputs_by_prompt"] == base[r["case"]]["outputs_by_prompt"] for r in d["records"] if r["cell"] == c)
        out = "  %-12s" % c
        for k, fmt in (("output_tok_per_s", "%5.0f"), ("tpot_p50", "%5.2f"), ("tpot_p99", "%5.1f"), ("ttft_p50", "%3.0f"), ("ttft_p95", "%3.0f"), ("ttft_p99", "%3.0f"), ("e2e_p99", "%4.0f")):
            out += " %s %s -> %s (v %s)" % (k.replace("output_tok_per_s", "tok/s"), fmt % f(b["records"], k), fmt % f(d["records"], k), fmt % f(vr, k))
        print(out, "err", sum(r["errors"] for r in d["records"] if r["cell"] == c), "same", same)
