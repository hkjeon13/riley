#!/usr/bin/env python3
"""prof_steps.py DB: per step of one request, mean duration of selected kernels."""
import collections, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select k.start,k.end,s.value,k.gridX*k.gridY*k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
bursts, cur = [], [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > 100_000_000:
        bursts.append(cur); cur = []
    cur.append(r)
bursts.append(cur)
b = [x for x in bursts if len(x) > 500][-2]
starts = [i for i, r in enumerate(b) if r[2].startswith("embedding_kernel")] + [len(b)]
keys = ["BatchPrefill", "MergeStates", "Kernel2", "ampere", "rms_norm", "silu"]
print("step  ms     " + "  ".join(f"{k[:12]:>12s}" for k in keys) + "   attn grid")
for j in range(len(starts) - 1):
    step = b[starts[j]:starts[j+1]]
    t = collections.defaultdict(list); grid = 0
    for s, e, n, g in step:
        for k in keys:
            if k in n:
                t[k].append(e - s)
                if k == "BatchPrefill": grid = g
    print(f"{j:4d} {(step[-1][1]-step[0][0])/1e6:6.2f} " + "  ".join(f"{sum(t[k])/max(len(t[k]),1)/1e3:12.1f}" for k in keys) + f"   {grid}")
