#!/usr/bin/env python3
"""step14.py DB: steps split at each plan_kernel; histogram of step spans and per-cluster kernel breakdown (top kernels)."""
import sqlite3, sys, re, collections
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
steps, cur = [], []
for r in rows:
    if "plan_kernel" in r[2] and cur:
        steps.append(cur); cur = []
    cur.append(r)
steps.append(cur)
steps = steps[len(steps)//5:]  # skip warm-up part
hist = collections.Counter(round((s[-1][1]-s[0][0])/1e6, 1) for s in steps)
print("steps", len(steps))
for k, v in sorted(hist.items()):
    if v >= 3: print(f"  {k:6.1f} ms x{v}")
def sig(s):
    return len(s)
clusters = collections.defaultdict(list)
for s in steps:
    span = (s[-1][1]-s[0][0])/1e6
    clusters[(len(s), round(span))].append(s)
for key, ss in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:6]:
    agg = collections.Counter(); grids = {}
    for s in ss:
        for b, e, n, gx, gy, gz in s:
            agg[n] += (e-b)/1e6; grids[n] = (gx, gy, gz)
    n = len(ss); span = sum((s[-1][1]-s[0][0]) for s in ss)/n/1e6; busy = sum(sum(e-b for b,e,*_ in s) for s in ss)/n/1e6
    print(f"cluster kernels={key[0]} ~{key[1]}ms: {n} steps span {span:.2f} busy {busy:.2f}")
    for name, t in agg.most_common(10):
        print(f"    {t/n:7.3f} ms  {name[:70]} grid={grids[name]}")
