#!/usr/bin/env python3
"""prof_an.py DB [DB...]: kernel time per prefill request, by category and by kernel."""
import collections, re, sqlite3, sys

CATS = [("attention", r"flashinfer|BatchPrefill|flash|fmha|attention|Attention|paged|merge"),
        ("gemm", r"gemm|Gemm|GEMM|cutlass|sm80_xmma|ampere_|s16816|s1688|splitK|cublas|wgmma|_tn$|_nn$|_nt$"),
        ("norm", r"norm|Norm"), ("rope", r"rope|rotary|Rotary"),
        ("act", r"silu|swiglu|act_and_mul|Silu|gelu"), ("copy", r"copy|Copy|cat|reshape|cache|Cache|scatter|gather|index")]

def category(name):
    for cat, pattern in CATS:
        if re.search(pattern, name):
            return cat
    return "other"

for path in sys.argv[1:]:
    db = sqlite3.connect(path)
    rows = db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
    bursts, cur = [], [rows[0]]
    for r in rows[1:]:
        if r[0] - cur[-1][1] > 100_000_000:
            bursts.append(cur); cur = []
        cur.append(r)
    bursts.append(cur)
    big = [b for b in bursts if len(b) > 500][-6:]
    print(f"== {path}: {len(rows)} kernels, {len(bursts)} bursts, using {len(big)} with",
          [len(b) for b in big])
    n = len(big)
    cats, names = collections.Counter(), collections.defaultdict(lambda: [0, 0])
    span = busy = 0
    for b in big:
        span += b[-1][1] - b[0][0]
        end = b[0][0]
        for s, e, name in b:
            busy += max(0, e - max(s, end)); end = max(end, e)
            cats[category(name)] += e - s
            short = re.sub(r"\(.*", "", name)[:110]
            names[short][0] += e - s; names[short][1] += 1
    print(f"per request: span {span/n/1e6:.1f} ms, GPU busy {busy/n/1e6:.1f} ms, idle {(span-busy)/n/1e6:.1f} ms")
    for cat, t in cats.most_common():
        print(f"  {cat:10s} {t/n/1e6:8.2f} ms")
    for name, (t, c) in sorted(names.items(), key=lambda x: -x[1][0])[:25]:
        print(f"  {t/n/1e6:8.2f} ms {c/n:7.0f}x {t/c/1e3:8.1f} us  {name}")
