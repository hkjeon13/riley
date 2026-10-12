"""Riley step composition by period class from an nsys trace (kernel list per pure-decode and mixed steps)."""
import sqlite3, collections, statistics as st, sys
db = sqlite3.connect(sys.argv[1]); names = dict(db.execute("select id,value from StringIds"))
ks = db.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
emb = [k for k, v in names.items() if v == "embedding_kernel"][0]
idx = [i for i, k in enumerate(ks) if k[2] == emb]
steps = [(ks[a][0], ks[b][0], ks[a:b]) for a, b in zip(idx, idx[1:])]; steps = [s for s in steps if s[1] - s[0] < 2e8]; steps = steps[len(steps) // 4:]
per = [(s[1] - s[0]) / 1e6 for s in steps]; med = st.median(per)
print("steps", len(steps), "mean period", round(st.mean(per), 2), "median", round(med, 2))
for label, lo, hi in (("pure decode", 0, med * 1.12), ("mixed", med * 1.12, 1e9)):
    sel = [s for s, p in zip(steps, per) if lo <= p < hi]
    if not sel: continue
    c = collections.defaultdict(lambda: [0, 0])
    for s in sel:
        for a, b, n in s[2]: c[names[n]][0] += 1; c[names[n]][1] += b - a
    mean = st.mean((s[1] - s[0]) / 1e6 for s in sel); busy = sum(t for n, t in c.values()) / len(sel) / 1e6
    print(f"-- {label}: n={len(sel)} share_of_time={sum(s[1] - s[0] for s in sel) / sum(s[1] - s[0] for s in steps):.2f} mean={mean:.2f} ms kernels/step={sum(n for n, t in c.values()) / len(sel):.0f} busy={busy:.2f} idle={mean - busy:.2f}")
    for k, (n, t) in sorted(c.items(), key=lambda x: -x[1][1])[:16]: print(f"     {t / len(sel) / 1e6:6.3f} ms n={n / len(sel):6.1f} avg={t / n / 1e3:7.1f}us {k[:66]}")
