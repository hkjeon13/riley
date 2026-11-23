"""Measure the committed batch shapes of each short C32 run from its own snapshots."""

import json
from pathlib import Path
from statistics import median
import sys

from check import arm, require
from check_distribution_gpu import current_batch_shapes


def per_run(folder, limit):
    receipt, workload = arm(folder, limit)
    before, after = (current_batch_shapes(workload[name]) for name in ("rag", "short"))
    require(before.keys() == after.keys(), "the exposed batch shapes changed during the run")
    committed = {}
    for rows, (hits, time_ns, _) in after.items():
        steps, gpu_ns = hits - before[rows][0], time_ns - before[rows][1]
        require(steps >= 0 and gpu_ns >= 0, "committed counters went backwards")
        if steps:
            committed[rows] = {"steps": steps, "gpu_ns": gpu_ns}
    require(committed, "the short workload recorded no committed CUDA batches")
    arrivals = sorted(workload["short"]["rows"], key=lambda row: row["sent_at"])
    pending = [sum(a["start"] + a["e2e"] > row["sent_at"] for a in arrivals[:i])
               for i, row in enumerate(arrivals)]
    return {
        "committed_steps": sum(value["steps"] for value in committed.values()),
        "committed_gpu_ms": sum(value["gpu_ns"] for value in committed.values()) / 1_000_000,
        "wide_buckets": {str(key): value for key, value in committed.items() if key > 256},
        "pending_at_later_write_p50": median(pending[32:]),
        "later_writes_with_at_least_32_pending": sum(count >= 32 for count in pending[32:]),
        "workload_sha256": receipt["artifacts_sha256"]["workload.json"],
    }


def report(original, reverse=None):
    results = {}
    for name, root in (("original", original), ("reverse", reverse)):
        if root is None:
            continue
        index_runs = {}
        for gpu in (6, 7):
            index_runs[str(gpu)] = {
                label: per_run(root / f"private-short-d{mode}-control-gpu{gpu}", limit)
                for label, mode, limit in (("stock", "0", None), ("prefill_only", "prefill-only", 0))
            }
        results[name] = index_runs
    return results


if __name__ == "__main__":
    print(json.dumps(report(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
