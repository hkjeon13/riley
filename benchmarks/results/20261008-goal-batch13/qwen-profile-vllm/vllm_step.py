"""vLLM decode-step composition from an nsys trace with node-level graph tracing: group kernels by graph node execution."""
import sqlite3, collections, statistics as st, sys
db = sqlite3.connect(sys.argv[1]); names = dict(db.execute("select id,value from StringIds"))
cols = [r[1] for r in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]; print("kernel cols has graphNodeId:", "graphNodeId" in cols)
ks = db.execute("select start,end,shortName,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
print("kernels", len(ks), "with graph node", sum(k[3] is not None and k[3] != 0 for k in ks))
gk = [k for k in ks if k[3]]
# split into graph executions: a gap in time > 200us or node id wrap
runs = []; cur = []
for k in gk:
    if cur and (k[3] <= cur[-1][3] or k[0] - cur[-1][1] > 300000): runs.append(cur); cur = []
    cur.append(k)
if cur: runs.append(cur)
sizes = collections.Counter(len(r) for r in runs); print("graph executions", len(runs), "kernel counts", sizes.most_common(6))
n0 = sizes.most_common(1)[0][0]; sel = [r for r in runs if len(r) == n0]
dur = [(r[-1][1] - r[0][0]) / 1e6 for r in sel]; busy = [sum(e - s for s, e, n, g in r) / 1e6 for r in sel]
print("selected", len(sel), "span ms median", round(st.median(dur), 3), "busy ms median", round(st.median(busy), 3), "idle", round(st.median(dur) - st.median(busy), 3))
c = collections.defaultdict(lambda: [0, 0])
for r in sel:
    for s, e, n, g in r: c[names[n]][0] += 1; c[names[n]][1] += e - s
for k, (n, t) in sorted(c.items(), key=lambda x: -x[1][1])[:22]: print(f"  {t / len(sel) / 1e6:6.3f} ms n={n / len(sel):6.1f} avg={t / n / 1e3:7.1f}us {k[:80]}")
print("first 30 kernels of a step:", [names[k[2]][:26] for k in sel[len(sel) // 2][:30]])
starts = [r[0][0] for r in sel]; per = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
if per: print("step-to-step ms p10/p50/p90", [round(sorted(per)[int(len(per) * q)], 2) for q in (.1, .5, .9)])
