#!/usr/bin/env python3
"""dec_an.py A.sqlite B.sqlite: per-kernel GPU time of the two runs (decode-heavy), largest differences first."""
import collections, re, sqlite3, sys
tot = []
for path in sys.argv[1:3]:
    db = sqlite3.connect(path)
    names = collections.defaultdict(lambda: [0, 0])
    rows = db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
    # Requests start after an idle gap of at least 5 s that follows start-up warm-up.
    first = max((i for i in range(1, len(rows)) if rows[i][0] - rows[i - 1][1] > 5e9), default=0)
    print(path, "kernels after warm-up", len(rows) - first)
    for s, e, name in rows[first:]:
        short = re.sub(r"\(.*", "", name)[:100]
        names[short][0] += e - s; names[short][1] += 1
    tot.append(names)
keys = set(tot[0]) | set(tot[1])
print("total ms", [round(sum(v[0] for v in t.values()) / 1e6, 1) for t in tot])
for k in sorted(keys, key=lambda k: -abs(tot[1].get(k, [0, 0])[0] - tot[0].get(k, [0, 0])[0]))[:25]:
    a, b = tot[0].get(k, [0, 0]), tot[1].get(k, [0, 0])
    print(f"{a[0]/1e6:9.2f} {b[0]/1e6:9.2f} ms  {a[1]:7d} {b[1]:7d}x  {k}")
