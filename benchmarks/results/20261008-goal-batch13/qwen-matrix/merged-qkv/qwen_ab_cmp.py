"""Compare two steady result files cell by cell: python3 qwen_ab_cmp.py BASE_DIR NEW_DIR"""
import json, statistics as st, sys
b = json.load(open(sys.argv[1] + "/result.json")); d = json.load(open(sys.argv[2] + "/result.json")); v = json.load(open("qwen-matrix-steady-attempt01/result.json")); vr = [r for r in v["records"] if r["engine"] == "vllm"]
br = [r for r in b["records"] if r["engine"] == "riley-scheduler"]; dr = [r for r in d["records"] if r["engine"] == "riley-scheduler"]
print("base", b["complete"], b["failure"], "new", d["complete"], d["failure"])
base = {r["case"]: r for r in br}; cells = []
for r in dr:
    if r["cell"] not in cells: cells.append(r["cell"])
def distinct(rs):
    by = {}
    for r in rs:
        for k, o in r["outputs_by_prompt"].items(): by.setdefault(k, set()).update(o)
    return by
db, dd = distinct(br), distinct(dr)
print("prompts with one output across all cells: base", sum(len(x) == 1 for x in db.values()), "/", len(db), "new", sum(len(x) == 1 for x in dd.values()), "/", len(dd), "| prompts whose output is unchanged:", sum(db[k] == dd.get(k) for k in db), "/", len(db))
for c in cells:
    f = lambda rs, k: st.mean(r[k] for r in rs if r["cell"] == c)
    out = "  %-12s" % c
    for k, fmt in (("output_tok_per_s", "%5.0f"), ("tpot_p50", "%5.2f"), ("tpot_p99", "%5.1f"), ("ttft_p50", "%3.0f"), ("ttft_p95", "%3.0f"), ("ttft_p99", "%3.0f"), ("e2e_p50", "%4.0f")):
        out += " %s %s -> %s (v %s)" % (k.replace("output_tok_per_s", "tok/s"), fmt % f(br, k), fmt % f(dr, k), fmt % f(vr, k))
    print(out, "err", sum(r["errors"] for r in dr if r["cell"] == c), "gain %+.1f%%" % ((f(dr, "output_tok_per_s") / f(br, "output_tok_per_s") - 1) * 100))
