"""Localize the mixed-row cap latency changes within matched GPU workload pairs."""

from argparse import ArgumentParser
import json
from math import isclose, isfinite
from pathlib import Path

from analyze_prefill_latency import paired_shifts, percentile, send_span, tpot, ttft
from compare_mixed_step_budget import EXPECTED_CELLS
from compare_profile_arms import arm, require


def rag_cohorts(pairs, baseline_stats, candidate_stats):
    lengths = sorted({old["prompt_tokens"] for old, _ in pairs})
    require(len(lengths) >= 2, "the load has only one recorded input length")
    lower, upper = max(zip(lengths, lengths[1:]), key=lambda gap: gap[1] - gap[0])
    groups = {}
    for label, cohort in (("shorter", [(old, new) for old, new in pairs if old["prompt_tokens"] <= lower]),
                          ("longer", [(old, new) for old, new in pairs if old["prompt_tokens"] >= upper])):
        before, after = zip(*cohort, strict=True)
        groups[label] = {
            "requests": len(cohort),
            "prompt_tokens": [min(row["prompt_tokens"] for row in before), max(row["prompt_tokens"] for row in before)],
            "ttft": paired_shifts(before, after, ttft),
            "decode_tpot": paired_shifts(before, after, tpot),
            "requests_in_control_ttft_tail": sum(ttft(row) >= baseline_stats["ttft_p95_ms"] for row in before),
            "requests_in_candidate_ttft_tail": sum(ttft(row) >= candidate_stats["ttft_p95_ms"] for row in after),
        }
    return groups


def wave_cohorts(pairs, spec):
    width = spec["concurrency"]
    require(len(pairs) % width == spec["warmup"] % width == 0, "partial wave")
    waves = []
    for begin in range(0, len(pairs), width):
        before, after = zip(*pairs[begin:begin + width], strict=True)
        waves.append({
            "wave": begin // width + 1,
            "control_send_span_ms": send_span(before),
            "candidate_send_span_ms": send_span(after),
            "decode_tpot": paired_shifts(before, after, tpot),
        })
    return waves


def analyze_pair(root, point):
    baseline, base_plan, before = arm(root, "qwen14", f"riley-{point}", expected_cells=EXPECTED_CELLS)
    candidate, cap_plan, after = arm(root, "qwen14", f"cap-{point}", expected_cells=EXPECTED_CELLS)
    require(base_plan["cells"] == cap_plan["cells"], "the compared workloads differ")
    cells = {}
    for name in ("rag-c8-s1", "rag-c8-s2", "wave-sync", "wave-50ms"):
        base_stats, old = before[name]
        cap_stats, new = after[name]
        pairs = list(zip(old, new, strict=True))
        match = ("request_index", "input_sha256", "prompt_tokens", "output_sha256", "finish", "completion_tokens")
        require(all(all(a[key] == b[key] for key in match) for a, b in pairs), f"{name}: input or output changed")
        for stats, rows in ((base_stats, old), (cap_stats, new)):
            require(all(0 < row["ttft"] <= row["e2e"] and row["completion_tokens"] > 0
                        and all(isfinite(row[key]) for key in ("ttft", "e2e", "sent_at")) for row in rows)
                    and isclose(stats["ttft_p95_ms"], percentile(rows, .95, ttft))
                    and isclose(stats["tpot_p50_ms"], percentile(rows, .5, tpot)),
                    f"{name}: raw request timings do not reproduce the aggregates")
        detail = {"input_length_groups": rag_cohorts(pairs, base_stats, cap_stats)} if name.startswith("rag") else {
            "waves": wave_cohorts(pairs, base_plan["cells"][name]),
        }
        cells[name] = {
            "requests": len(pairs),
            "control_source_sha256": baseline["results"][name],
            "candidate_source_sha256": candidate["results"][name],
            "ttft": paired_shifts(old, new, ttft),
            "decode_tpot": paired_shifts(old, new, tpot),
            **detail,
        }
    return cells


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = {"pairs": {point: analyze_pair(args.root, point) for point in ("first", "second")}}
    print(json.dumps(report, indent=2, allow_nan=False))
