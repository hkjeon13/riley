import sqlite3, sys, itertools
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
first = max((i for i in range(1, len(rows)) if rows[i][0] - rows[i-1][1] > 5e9), default=0); rows = rows[first:]
steps, cur = [], []
for r in rows:
    if "plan_kernel" in r[2] and cur:
        steps.append(cur); cur = []
    cur.append(r)
steps.append(cur)
kinds = "".join("L" if (s[-1][1]-s[0][0]) > 40e6 else "d" for s in steps)
print(len(steps), "steps")
print(kinds[:600])
runs = [(k, len(list(g))) for k, g in itertools.groupby(kinds)]
print("L-run lengths:", sorted(n for k, n in runs if k == "L"))
print("d-run lengths between L runs:", [n for k, n in runs if k == "d"][:60])
