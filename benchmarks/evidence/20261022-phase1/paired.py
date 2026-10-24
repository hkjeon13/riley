"""Median paired percent changes (native-after vs native) per concurrency."""
import json, statistics as st, sys

def summary(path):
    d = json.load(open(path))["records"]
    out = {}
    for c in sorted({r["concurrency"] for r in d}):
        b = {r["repeat"]: r for r in d if r["concurrency"] == c and r["mode"] == "native"}
        a = {r["repeat"]: r for r in d if r["concurrency"] == c and r["mode"] == "native-after"}
        ch = lambda k: st.median((a[i][k] - b[i][k]) / b[i][k] * 100 for i in b)
        med = lambda k, s: st.median(s[i][k] for i in s)
        out[c] = dict(pairs=len(b), tput=ch("output_tok_s"), tpot50=ch("tpot_ms_p50"), tpot99=ch("tpot_ms_p99"),
                      ttft95=ch("ttft_ms_p95"), ttft99=ch("ttft_ms_p99"), itl99=ch("itl_ms_p99"),
                      ttft95_abs=(med("ttft_ms_p95", b), med("ttft_ms_p95", a)), itl99_abs=(med("itl_ms_p99", b), med("itl_ms_p99", a)),
                      tput_abs=(med("output_tok_s", b), med("output_tok_s", a)),
                      tput_pairs=[round((a[i]["output_tok_s"] - b[i]["output_tok_s"]) / b[i]["output_tok_s"] * 100, 2) for i in sorted(b)],
                      canon_min=min(x["measured_native_canonical_matches"] for x in list(a.values()) + list(b.values())))
    return out

if __name__ == "__main__":
    for path in sys.argv[1:]:
        for c, s in summary(path).items():
            print(path.split("/")[-2], f"C{c}", " ".join(f"{k}={v:+.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
