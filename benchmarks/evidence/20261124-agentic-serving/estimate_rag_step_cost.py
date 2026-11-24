"""Estimate time to remove on frozen 14B step timelines without changing arrival order."""

import argparse
import hashlib
import json
from pathlib import Path


def p95(values):
    ordered = sorted(values)
    return ordered[int(len(ordered) * 0.95)]


def projected(times, spans, fraction):
    return p95(t - s * fraction for t, s in zip(times, spans, strict=True))


def describe(source):
    report = json.loads(source.read_text())
    result = {}
    for name, cell in report["cells"].items():
        rows = cell["all_requests"]
        if (
            len(rows) != 64
            or len({r["request_index"] for r in rows}) != 64
            or not cell["valid_against_frozen_vllm"]
        ):
            raise ValueError("invalid registered cell")
        target = 1.15 * cell["frozen_vllm_ttft_p95_ms"]
        times = [r["client_ttft_ms"] for r in rows]
        if abs(p95(times) - cell["diagnostic_ttft_p95_ms"]) > 1e-5:
            raise ValueError("p95 differs from original")
        result[name] = {"recorded_ms": p95(times), "target_ms": target, "scenarios": {}}
        scopes = {
            "own_iteration": ["own_execution_wall_ms"],
            "all_prefill_iterations": [
                "own_execution_wall_ms",
                "foreign_prefill_execution_wall_ms",
            ],
            "newer_only_foreign": ["newer_only_prefill_wall_ms"],
        }
        for label, cols in scopes.items():
            spans = [sum(r[k] for k in cols) for r in rows]
            if any(s < 0 or s > t + 0.01 for s, t in zip(spans, times, strict=True)):
                raise ValueError("overlapping or invalid span")

            lo, hi = 0.0, 1.0
            if projected(times, spans, hi) <= target:
                for _ in range(40):
                    middle = (lo + hi) / 2
                    if projected(times, spans, middle) > target:
                        lo = middle
                    else:
                        hi = middle
                threshold = hi
            else:
                threshold = None
            result[name]["scenarios"][label] = {
                "fraction_removed_to_reach_target": threshold,
                "p95_ms_if_10pct_removed": projected(times, spans, 0.10),
                "p95_ms_if_fully_removed": projected(times, spans, 1.0),
            }
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_revision": report["base_revision"],
        "assumption": "Per-observation accounting on original execution spans. Frozen arrival, queue, batch membership and nonselected durations. Mixed steps include concurrent decode. Not an executable schedule or speed prediction.",
        "cells": result,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    print(json.dumps(describe(p.parse_args().source), indent=2))
