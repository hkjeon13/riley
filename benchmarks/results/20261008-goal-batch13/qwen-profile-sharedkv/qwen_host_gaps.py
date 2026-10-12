"""Host-side view of a decode step from an nsys trace: time inside CUDA runtime calls by name, and host gaps between calls."""
import sqlite3, statistics as st, collections, sys, bisect
db = sqlite3.connect(sys.argv[1]); names = dict(db.execute("select id,value from StringIds"))
emb = [k for k, v in names.items() if v == "embedding_kernel"][0]
ks = db.execute("select start,end,shortName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
rt = db.execute("select start,end,nameId,correlationId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
corr = {c: (s, e) for s, e, n, c, t in rt}
# step boundaries by the host-side launch time of the embedding kernel
t0s = sorted(corr[c][0] for s, e, n, c in ks if n == emb and c in corr)
per = [b - a for a, b in zip(t0s, t0s[1:])]; med = st.median(per)
sel = [(a, b) for a, b in zip(t0s, t0s[1:]) if abs((b - a) - med) < 0.1 * med]
sel = sel[len(sel) // 3: 2 * len(sel) // 3]
print("steps", len(sel), "median period ms", round(med / 1e6, 2))
tid = collections.Counter(t for s, e, n, c, t in rt).most_common(1)[0][0]
calls = [(s, e, n) for s, e, n, c, t in rt if t == tid]; starts = [c[0] for c in calls]
api = collections.defaultdict(lambda: [0, 0]); gaps = collections.defaultdict(lambda: [0, 0]); gap_total = 0; biggest = collections.defaultdict(lambda: [0, 0])
for a, b in sel:
    i = bisect.bisect_left(starts, a); j = bisect.bisect_left(starts, b); prev_end = None; prev_name = None
    for s, e, n in calls[i:j]:
        nm = names[n]; api[nm][0] += 1; api[nm][1] += e - s
        if prev_end is not None:
            g = s - prev_end; gap_total += g
            bucket = "<2us" if g < 2000 else "2-10us" if g < 10000 else "10-100us" if g < 100000 else ">=100us"; gaps[bucket][0] += 1; gaps[bucket][1] += g
            if g >= 10000: biggest[prev_name + " -> " + nm][0] += 1; biggest[prev_name + " -> " + nm][1] += g
        prev_end = e; prev_name = nm
    # gap from last call of this step to the next step's first call
    if j < len(calls): g = calls[j][0] - prev_end; gap_total += g; biggest[prev_name + " -> [next step] " + names[calls[j][2]]][0] += 1; biggest[prev_name + " -> [next step] " + names[calls[j][2]]][1] += g
n = len(sel)
print("in CUDA runtime calls ms/step", round(sum(v[1] for v in api.values()) / n / 1e6, 3), "| host gaps ms/step", round(gap_total / n / 1e6, 3))
for k, (c, t) in sorted(api.items(), key=lambda x: -x[1][1])[:8]: print(f"  api {t / n / 1e6:6.3f} ms/step n={c / n:7.1f} avg={t / max(c, 1) / 1e3:7.2f}us {k}")
for k, (c, t) in gaps.items(): print(f"  gaps {k:9s} {t / n / 1e6:6.3f} ms/step n={c / n:7.1f}")
for k, (c, t) in sorted(biggest.items(), key=lambda x: -x[1][1])[:12]: print(f"  big {t / n / 1e6:6.3f} ms/step n={c / n:5.2f} avg={t / max(c, 1) / 1e3:7.1f}us {k}")
