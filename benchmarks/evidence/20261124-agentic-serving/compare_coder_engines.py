"""Compare same-budget responses on paired mixed Coder-14B serving waves."""

import argparse
import hashlib
import json
from pathlib import Path

from verify_coder_fleet import percentile, read


def analyze(riley, vllm):
    if (riley / "plan.json").read_bytes() != (vllm / "plan.json").read_bytes():
        raise ValueError("mixed arrival manifests differ")
    rargs = (riley / "gpu6/args").read_text().splitlines()
    vargs = read(vllm / "gpu6/args.json")
    capacity = int(rargs[rargs.index("--kv-blocks") + 1]) * 16 * 48 * 8 * 128 * 2 * 2
    if capacity != int(vargs[vargs.index("--kv-cache-memory-bytes") + 1]):
        raise ValueError("KV capacity differs")
    for gpu in (6, 7):
        d = f"gpu{gpu}"
        if (
            read(riley / d / "lease-after.json")["gpu"]
            != read(vllm / d / "lease-after.json")["gpu"]
        ):
            raise ValueError("different physical device")
    comparisons = []
    for i, j in ((0, 0), (5, 1)):
        rr = read(riley / f"arm-{i}-round-robin.json")
        vr = read(vllm / f"arm-{j}-round-robin.json")
        by_key = lambda rows: {(r["short"], r["index"]): r for r in rows}
        a, b = by_key(rr), by_key(vr)
        if len(rr) != len(vr) or len(a) != 256 or a.keys() != b.keys():
            raise ValueError("missing corpus")
        summary = {}
        for name, flag in (("short", True), ("rag", False)):
            pairs = [(a[k], b[k]) for k in a if k[0] == flag]
            for left, right in pairs:
                if (
                    left["input_sha256"] != right["input_sha256"]
                    or left["offset_s"] != right["offset_s"]
                ):
                    raise ValueError("unequal transport inputs")
            complete = [(l, r) for l, r in pairs if not l["error"] and not r["error"]]
            lengths = [
                (l, r)
                for l, r in complete
                if all(
                    l[x] == r[x]
                    for x in ("prompt_tokens", "completion_tokens", "finish")
                )
            ]
            same = [
                (l, r) for l, r in lengths if l["output_sha256"] == r["output_sha256"]
            ]
            times = {}
            for metric, q in (("ttft", 0.95), ("tpot", 0.5), ("e2e", 0.95)):
                v = [
                    1000 * percentile([pair[k][metric] for pair in lengths], q)
                    for k in (0, 1)
                ]
                times[metric] = {
                    "riley_ms": v[0],
                    "vllm_ms": v[1],
                    "ratio": v[0] / v[1],
                }
            summary[name] = {
                "total": len(pairs),
                "complete_both": len(complete),
                "matched_lengths": len(lengths),
                "identical_text": len(same),
                "metrics_for_equal_length_responses": times,
                "identical_text_paired_tpot_ratio_median": percentile(
                    [l["tpot"] / r["tpot"] for l, r in same], 0.5
                ),
            }
        comparisons.append(summary)
    return {
        "plan_sha256": hashlib.sha256((riley / "plan.json").read_bytes()).hexdigest(),
        "allocated_kv_bytes_per_gpu": capacity,
        "round_robin_comparisons": comparisons,
        "scope": "Order Riley then vLLM, two warm waves per engine on one pair of physical GPUs. Per-wave summaries, no independent cold-launch confidence bound.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("riley", type=Path)
    p.add_argument("vllm", type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.riley, a.vllm), indent=2))
