"""Per-step kernel time grouped by dense row bucket: python3 qwen_step_rows.py trace.sqlite [skip_fraction]"""
import sqlite3, statistics as st, collections, sys
db = sqlite3.connect(sys.argv[1]); names = dict(db.execute("select id,value from StringIds"))
skip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
ks = db.execute("select start,end,shortName,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
emb = [k for k, v in names.items() if v == "embedding_kernel"][0]
def short(n):
    n = names[n]
    for a, b in (("native_bf16_ragged_paged_split_gqa_d128_shared_kv_partial_state_kernel", "attn_producer"), ("native_bf16_ragged_paged_split_gqa_d128_global_merge_kernel", "attn_merge"), ("native_bf16_prefill_d128_mma_partial_state_kernel", "attn_prompt_mma"), ("wide_bf16_residual_rms_norm_kernel", "resid_norm"), ("silu_gated_multiply_bf16_kernel", "silu_mul"), ("splitKreduce_kernel", "splitk_reduce")):
        if n == a: return b
    if "gemm" in n or n == "Kernel2": return "gemm"
    if "qkv_split" in n: return "qkv_split(+kv)"
    return n[:28]
steps = []; cur = None
for s, e, n, gx, gy, gz in ks:
    if n == emb:
        if cur: steps.append(cur)
        cur = {"t0": s, "k": collections.defaultdict(lambda: [0, 0]), "rows": None, "busy_end": s, "gap": 0, "last": s}
    if cur is None: continue
    nm = short(n); cur["k"][nm][0] += 1; cur["k"][nm][1] += e - s
    if nm == "resid_norm" and cur["rows"] is None: cur["rows"] = gx * gy * gz
    if s > cur["last"]: cur["gap"] += s - cur["last"]
    cur["last"] = max(cur["last"], e)
for i in range(len(steps) - 1): steps[i]["period"] = steps[i + 1]["t0"] - steps[i]["t0"]
steps = [x for x in steps[:-1] if x["period"] < 2e8]
steps = steps[int(len(steps) * skip):]
tot = sum(x["period"] for x in steps)
print("steps", len(steps), "total s", round(tot / 1e9, 2), "mean period ms", round(tot / len(steps) / 1e6, 2), "median", round(st.median(x["period"] for x in steps) / 1e6, 2))
groups = collections.defaultdict(list)
for x in steps: groups[(x["rows"], "attn_prompt_mma" in x["k"])].append(x)
allnames = sorted({n for x in steps for n in x["k"]}, key=lambda n: -sum(x["k"][n][1] for x in steps if n in x["k"]))[:12]
print("%-12s %5s %6s %7s %7s %6s | " % ("rows,mma", "n", "time%", "period", "kernel", "gap") + " ".join("%14s" % n[:14] for n in allnames))
for key in sorted(groups, key=lambda k: (k[0] or 0, k[1])):
    g = groups[key]; p = sum(x["period"] for x in g)
    ksum = st.mean(sum(t for c, t in x["k"].values()) for x in g)
    print("%-12s %5d %5.1f%% %7.2f %7.2f %6.2f | " % ("%s%s" % (key[0], "+m" if key[1] else ""), len(g), 100 * p / tot, p / len(g) / 1e6, ksum / 1e6, st.mean(x["gap"] for x in g) / 1e6) + " ".join("%14.3f" % (st.mean(x["k"][n][1] if n in x["k"] else 0 for x in g) / 1e6) for n in allnames))
