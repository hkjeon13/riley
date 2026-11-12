#!/usr/bin/env python3
"""step_an.py DB: split kernels into steps at each fixed-tile plan_kernel; group steps by duration with kernel-time breakdown."""
import sqlite3, sys, re, collections
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
first = max((i for i in range(1, len(rows)) if rows[i][0] - rows[i-1][1] > 5e9), default=0); rows = rows[first:]
steps, cur = [], []
for r in rows:
    if "plan_kernel" in r[2] and cur:
        steps.append(cur); cur = []
    cur.append(r)
steps.append(cur)
def cat(n):
    for c, p in [("attn", r"strided_paged|merge_parts|flash|paged|attention"), ("gemm", r"gemm|Gemm|cutlass|splitK|ampere_"), ("norm", r"norm"), ("act", r"silu")]:
        if re.search(p, n): return c
    return "other"
buckets = collections.defaultdict(list)
for s in steps:
    span = (s[-1][1] - s[0][0]) / 1e6
    busy = sum(e - b for b, e, _ in s) / 1e6
    c = collections.Counter()
    for b, e, n in s: c[cat(n)] += (e - b) / 1e6
    key = "long>40ms" if span > 40 else ("mid 10-40" if span > 10 else "short<10")
    buckets[key].append((span, busy, c))
for k, v in sorted(buckets.items()):
    n = len(v)
    agg = collections.Counter()
    for _, _, c in v: agg.update(c)
    print(f"{k}: {n} steps, span {sum(x[0] for x in v)/n:.2f} busy {sum(x[1] for x in v)/n:.2f} ms; " + " ".join(f"{a} {t/n:.2f}" for a, t in agg.most_common()))
long = sorted((x for x in buckets.get("long>40ms", [])), key=lambda x: x[0])
print("long spans:", [round(x[0], 1) for x in long][:40])
