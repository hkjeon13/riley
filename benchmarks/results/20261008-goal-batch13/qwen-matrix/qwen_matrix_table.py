import json, statistics as st, collections, sys
r = json.load(open(sys.argv[1])); print("complete", r["complete"], r["failure"], len(r["records"]))
d = collections.defaultdict(list)
for x in r["records"]: d[(x["cell"], x["engine"])].append(x)
m = lambda v, k: st.mean(x[k] for x in v)
det = lambda v: all(a.get("outputs_by_prompt") == v[0].get("outputs_by_prompt") and all(len(s) == 1 for s in a["outputs_by_prompt"].values()) for a in v)
print("| 조건 | tok/s Riley / vLLM | 비율 | TTFT p50 ms | TPOT p50 ms | E2E p50 ms | GPU MiB | 오류 | 반복 간 출력 동일 |\n|---|---:|---:|---:|---:|---:|---:|---:|:-:|")
for cell in ["c1-fixed", "c1-natural", "c8-fixed", "c8-natural", "c16-fixed", "c16-natural", "c32-fixed", "c32-natural"]:
    a = d.get((cell, "riley-scheduler"), []); b = d.get((cell, "vllm"), [])
    if len(a) < 2 or len(b) < 2: print("|", cell, "| 미완료", len(a), len(b), "|"); continue
    ta, tb = m(a, "output_tok_per_s"), m(b, "output_tok_per_s")
    print("| %s | %.0f / %.0f | %.2f | %.0f / %.0f | %.1f / %.1f | %.0f / %.0f | %.0f / %.0f | %d / %d | %s / %s |" % (cell, ta, tb, ta / tb, m(a, "ttft_p50"), m(b, "ttft_p50"), m(a, "tpot_p50"), m(b, "tpot_p50"), m(a, "e2e_p50"), m(b, "e2e_p50"),
          max(x["gpu_peak_MiB"] for x in a), max(x["gpu_peak_MiB"] for x in b), sum(x["errors"] for x in a), sum(x["errors"] for x in b), det(a), det(b)))
