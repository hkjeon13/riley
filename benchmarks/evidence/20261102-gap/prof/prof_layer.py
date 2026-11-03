#!/usr/bin/env python3
"""prof_layer.py DB: kernels of one layer in the first prefill step of a late request, with grids."""
import re, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
bursts, cur = [], [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > 100_000_000:
        bursts.append(cur); cur = []
    cur.append(r)
bursts.append(cur)
b = [x for x in bursts if len(x) > 500][-2]
# Steps start at an embedding kernel.
starts = [i for i, r in enumerate(b) if "embedding" in r[2] and "error" not in r[2] and "validate" not in r[2]] or [0]
print("steps", len(starts), "step ms", [round((b[(starts+[len(b)])[j+1]-1][1]-b[s][0])/1e6, 2) for j, s in enumerate(starts)])
step = b[starts[1]:starts[2]] if len(starts) > 2 else b
# Print kernels from the 6th to the 2nd layer's worth: find a repeating window after the first 40 kernels.
for s, e, name, gx, gy, gz, bx in step[20:60]:
    print(f"{(e-s)/1e3:8.1f} us grid {gx}x{gy}x{gz} blk {bx}  {name[:80]}")
