"""Attribute the observed 50 ms wave p95 to requests in the matched Coder-14B runs."""

from argparse import ArgumentParser
from collections import Counter
import json
import math
from pathlib import Path

from compare_coder import EXPECTED_CELLS
from compare_profile_arms import arm, ratio_percent, require


CELL = "wave-50ms"
FIELDS = ("request_index", "input_sha256", "prompt_tokens", "output_sha256", "completion_tokens", "finish")
PAIRS = (("riley-first", "qkv-first"), ("riley-second", "qkv-second"))


def p95(rows):
    samples = sorted(row["ttft"] for row in rows)
    return 1000 * samples[min(int(0.95 * len(samples)), len(samples) - 1)]


def summarize(root, first_pair=False):
    root = root.resolve()
    pairs = {}
    for control, diagnostic in PAIRS[:1] if first_pair else PAIRS:
        before, baseline, baseline_cells = arm(root, "coder14", control, expected_cells=EXPECTED_CELLS)
        after, changed, diagnostic_cells = arm(root, "coder14", diagnostic, expected_cells=EXPECTED_CELLS)
        require(baseline["env"] == {} and changed["env"] == {"RILEY_W3_QKV1024": "1"}
                and baseline["cells"] == changed["cells"], "wave plans differ")
        require(before["model"] == after["model"] == "Qwen2.5-Coder-14B-Instruct", "wrong model for tail comparison")
        config = baseline["cells"][CELL]
        width = config["concurrency"]
        require(config["stagger_ms"] == 50 and width == 8 and config["requests"] == 40
                and config["warmup"] == 8, "50 ms wave layout changed")
        base_stats, base_rows = baseline_cells[CELL]
        trial_stats, trial_rows = diagnostic_cells[CELL]
        matched = list(zip(base_rows, trial_rows, strict=True))
        require(all(all(old[key] == new[key] for key in FIELDS) for old, new in matched),
                "50 ms wave inputs or outputs differ")
        stats = (base_stats, trial_stats)
        variants = (base_rows, trial_rows)
        require(all(math.isfinite(row["ttft"]) and math.isfinite(row["sent_at"]) and row["ttft"] > 0
                    for row in (*base_rows, *trial_rows)), "invalid measurement")
        values = tuple(p95(rows) for rows in variants)
        require(all(math.isclose(value, report["ttft_p95_ms"], rel_tol=1e-12)
                    for value, report in zip(values, stats, strict=True)), "wave p95 differs from request timings")
        tail_waves = {}
        for mode, rows, threshold in zip((control, diagnostic), variants, values, strict=True):
            counts = Counter((index // width) + 1 for index, row in enumerate(rows)
                             if row["ttft"] * 1000 >= threshold)
            tail_waves[mode] = dict(counts)
        request = 41
        start = (request - config["warmup"]) // width * width
        focal = {}
        for mode, rows in zip((control, diagnostic), variants, strict=True):
            row = rows[request - config["warmup"]]
            require(row["request_index"] == request, "focal request differs")
            focal[mode] = {
                "ttft_ms": row["ttft"] * 1000,
                "prompt_tokens": row["prompt_tokens"],
                "sent_ms_after_wave_start": (row["sent_at"] - rows[start]["sent_at"]) * 1000,
            }
        omissions = []
        for start in range(0, len(base_rows), width):
            old, new = (p95(rows[:start] + rows[start + width:]) for rows in variants)
            omissions.append({"omitted_wave": start // width + 1, "ttft_p95_change_percent": ratio_percent(new, old)})
        biggest = max(matched, key=lambda rows: rows[1]["ttft"] - rows[0]["ttft"])
        pairs[control] = {
            "baseline": control, "candidate": diagnostic,
            "matched_exact_outputs": len(matched),
            "faster_requests": sum(new["ttft"] < old["ttft"] for old, new in matched),
            "ttft_p95_change_percent": ratio_percent(values[1], values[0]),
            "p95_and_slower_count_by_wave": tail_waves,
            "request_41": focal,
            "largest_increase": {"request_index": biggest[0]["request_index"],
                                 "ttft_increase_ms": 1000 * (biggest[1]["ttft"] - biggest[0]["ttft"])},
            "leave_one_wave_out": omissions,
            "verified_result_sha256": {control: before["results"][CELL], diagnostic: after["results"][CELL]},
        }
    reference = root.parent / "w3-qkv-serving-20261119.MhydzNsX"
    prior, qwen_plan, qwen_cells = arm(reference, "qwen14", "riley-first", expected_cells=EXPECTED_CELLS)
    coder, coder_plan, coder_cells = arm(root, "coder14", "riley-first", expected_cells=EXPECTED_CELLS)
    original_rows = qwen_cells[CELL][1]
    current_rows = coder_cells[CELL][1]
    matched_inputs = sum(all(old[field] == new[field] for field in FIELDS[:3])
                         for old, new in zip(original_rows, current_rows, strict=True))
    require(prior["model"] == "Qwen2.5-14B-Instruct" and qwen_plan["cells"][CELL] == coder_plan["cells"][CELL],
            "earlier Qwen2.5-14B wave configuration differs")
    return {
        "cell": CELL,
        "summary_scope": "Request timings and leave-one-wave-out sensitivity only.",
        "cross_model_inputs": {
            "models": [prior["model"], coder["model"]],
            "matched_request_index_body_hash_and_prompt_tokens": matched_inputs,
            "paired_requests": len(original_rows),
            "previous_wave_sha256": prior["results"][CELL],
            "coder_wave_sha256": coder["results"][CELL],
        },
        "pairs": pairs,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--first-pair", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.first_pair), indent=2, allow_nan=False))
