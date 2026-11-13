"""Ratio table for the 0.1.0 re-measure of closed-c1, logprobs, shared and 32B TP=2 (rest16/)."""
import json, statistics as st, sys
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "rest16")
def m(d, s, k):
    v = [json.loads((d / f"{p}-{s}.json").read_text()) for p in ("r1", "r2") if (d / f"{p}-{s}.json").exists()]
    w = [json.loads((d / f"{p}-{s}.json").read_text()) for p in ("v1", "v2") if (d / f"{p}-{s}.json").exists()]
    errs = sum(x["errors"] for x in v + w)
    return st.mean(x[k] for x in v), st.mean(x[k] for x in w), errs
rows = []
print("| 모델 | c1 TPOT | lp-c8 느림(vLLM/Riley 처리량) | lp-c32 느림 | 오류 |")
print("|---|---|---|---|---|")
for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("shared", "tp32")):
    if not (d / "v1-closed-c1.json").exists():
        print(f"| {d.name} | (미측정) | | | |"); continue
    r, v, e1 = m(d, "closed-c1", "tpot_p50_ms")
    a8, b8, e2 = m(d, "logprobs-c8", "output_tok_s")
    a32, b32, e3 = m(d, "logprobs-c32", "output_tok_s")
    print(f"| {d.name} | {r/v:.3f} | {b8/a8:.3f} | {b32/a32:.3f} | {e1+e2+e3} |")
s = root / "shared" / "Qwen2.5-7B-Instruct"
for c in ("shared-c8", "shared-c32"):
    r, v, e = m(s, c, "ttft_p95_ms"); tr, tv, _ = m(s, c, "output_tok_s")
    print(f"\n7B {c}: TTFT p95 {r:.1f} vs {v:.1f} ms = {r/v:.3f}x, 처리량 {tr:.0f} vs {tv:.0f}, 오류 {e}")
t = root / "tp32" / "Qwen2.5-32B-Instruct"
for c in ("closed-c1", "closed-c8", "closed-c16", "closed-c32"):
    r, v, e = m(t, c, "tpot_p50_ms"); tr, tv, _ = m(t, c, "output_tok_s"); fr, fv, _ = m(t, c, "ttft_p50_ms")
    print(f"32B TP2 {c}: TPOT {r/v:.3f}x, 처리량 {tr/tv:.3f}x, TTFT {fr/fv:.3f}x, 오류 {e}")
