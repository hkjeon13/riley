import json, statistics as st, os, sys
b = json.load(open("qwen-matrix-rileyonly-sharedkv-attempt01/result.json"))
cells = sorted({r["cell"] for r in b["records"]}, key=lambda c: (int(c.split("-")[0][1:]), c))
def m(d, c, k):
    r = [x[k] for x in d["records"] if x["cell"] == c and k in x]; return st.mean(r) if r else float("nan")
for v in sys.argv[1:]:
    p = f"qwen-matrix-flags-diagnostic-{v}-attempt01/result.json"; d = json.load(open(p)); print("==", v, "complete", d["complete"], "failure", d["failure"])
    base = {y["case"]: y for y in b["records"]}
    for c in cells:
        rs = [x for x in d["records"] if x["cell"] == c]
        if not rs: continue
        same = all(x["outputs_by_prompt"] == base[x["case"]]["outputs_by_prompt"] for x in rs)
        print("  %-12s tok/s %7.0f -> %7.0f  tpot %6.2f -> %6.2f  err %d same_out %s" % (c, m(b, c, "output_tok_per_s"), m(d, c, "output_tok_per_s"), m(b, c, "tpot_p50"), m(d, c, "tpot_p50"), sum(x["errors"] for x in rs), same))
