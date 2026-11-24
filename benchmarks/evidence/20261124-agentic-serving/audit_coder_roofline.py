"""Quantify 1024-row projection headroom on the archived GPU timeline."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median


def audit(root, targets):
    packed = root / "step-times.json.gz"
    with gzip.open(packed, "rt") as f:
        src = json.load(f)
    ref = json.loads((root / "verified.json").read_text())
    if (
        hashlib.sha256(packed.read_bytes()).hexdigest() != ref["step_times_sha256"]
        or src["trace_sha256"] != ref["trace_sha256"]
    ):
        raise ValueError("mismatched profiler capture")
    rows = src["steps"]
    if len(rows) != 492 or src["layers_per_step"] != 48:
        raise ValueError("unexpected capture")
    floor = ref["ideal_projection_ms_per_step"]
    savings = [sum(max(0, r[k] / 1e6 - v) for k, v in floor.items()) for r in rows]
    shares = [v / (r["span_ns"] / 1e6) for r, v in zip(rows, savings, strict=True)]
    if (
        abs(median(savings) - ref["median_all_four_optimistic_savings_ms_per_step"])
        > 1e-7
    ):
        raise ValueError("optimistic floor differs from archived verification")
    by_seed = {}
    for label, cell in json.loads(targets.read_text())["cells"].items():
        needed = cell["scenarios"]["all_prefill_iterations"][
            "fraction_removed_to_reach_target"
        ]
        by_seed[label] = {
            "fraction_needed_in_separate_fixed_timeline": needed,
            "wide_steps_whose_four_projection_headroom_reaches_fraction": sum(
                x >= needed for x in shares
            ),
        }
    return {
        "source_trace": src["trace_sha256"],
        "steps": len(rows),
        "upper_fraction_p50": median(shares),
        "upper_fraction_min": min(shares),
        "upper_fraction_max": max(shares),
        "median_other_observed_ms": {
            k: median(r[k] for r in rows) / 1e6
            for k in ("attention", "attention_merge", "rms_norm", "silu", "kv_write")
        },
        "separate_trace_sensitivity": by_seed,
        "scope": "Original producer, numerics and 1024-row shapes at dense BF16 peak. Cross-trace comparison of scale only; no predicted service p95 or claim about other batch sizes.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("targets", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.root, a.targets), indent=2))
