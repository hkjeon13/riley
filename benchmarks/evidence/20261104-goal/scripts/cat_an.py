#!/usr/bin/env python3
"""cat_an.py DB...: GPU kernel time by category over the decode-heavy part of each profile (after the last 5 s idle gap)."""
import collections, re, sqlite3, sys
CATS = [("attention", r"flashinfer|BatchPrefill|BatchDecode|flash|fmha|attention|Attention|paged|merge_parts|merge_state|reshape_and_cache"),
        ("gemm", r"gemm|Gemm|GEMM|cutlass|sm80_xmma|ampere_|s16816|s1688|splitK|cublas|_tn$|_nn$|_nt$"),
        ("norm", r"norm|Norm"), ("rope", r"rope|rotary|Rotary"),
        ("act", r"silu|swiglu|act_and_mul|Silu|gelu"), ("sample", r"argmax|sampl|topk|softmax|Softmax")]
def cat(name):
    for c, p in CATS:
        if re.search(p, name):
            return c
    return "other"
for path in sys.argv[1:]:
    db = sqlite3.connect(path)
    rows = db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
    first = max((i for i in range(1, len(rows)) if rows[i][0] - rows[i - 1][1] > 5e9), default=0)
    rows = rows[first:]
    span = (rows[-1][1] - rows[0][0]) / 1e6
    totals, names = collections.Counter(), collections.defaultdict(lambda: [0, 0])
    for s, e, n in rows:
        totals[cat(n)] += e - s
        k = re.sub(r"\(.*", "", n)[:90]; names[k][0] += e - s; names[k][1] += 1
    print(f"== {path}: {len(rows)} kernels, span {span:.0f} ms, busy {sum(totals.values())/1e6:.0f} ms")
    for c, t in totals.most_common():
        print(f"   {c:10s} {t/1e6:9.1f} ms")
    for k, (t, c) in sorted(names.items(), key=lambda x: -x[1][0])[:12]:
        print(f"   {t/1e6:9.1f} ms {c:7d}x {t/c/1e3:7.1f} us  {k}")
