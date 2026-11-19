"""Count prefill plans and tail-wait CUDA time that one additional bucket could affect."""

from argparse import ArgumentParser
from bisect import bisect_left
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from attribute_scheduler_wait import prefill_times
from compare_profile_arms import require
from summarize_scheduler_choices import read_scheduler


def read_buckets(root):
    plan = json.loads((root / "qwen14-riley/plan.json").read_bytes())
    argv = plan["server"]
    values = list(map(int, argv[argv.index("--batch-shape-buckets") + 1].split(",")))
    require(values[0] == 1 and values == sorted(set(values)) and values[-1] == 1024 and len(values) == 23,
            "unexpected baseline buckets")
    return values


def describe(iterations, sizes, cuda_time, buckets):
    shape_counts = Counter(sizes[iteration] for iteration in iterations)
    highest = max(buckets)
    total = sum(cuda_time[iteration] for iteration in iterations)
    full_time = sum(cuda_time[iteration] for iteration in iterations if sizes[iteration] == highest)
    candidates = []
    for added in range(16, highest, 16):
        if added in buckets:
            continue
        changed = [iteration for iteration in iterations
                   if sizes[iteration] <= added < buckets[bisect_left(buckets, sizes[iteration])]]
        savings = sum(buckets[bisect_left(buckets, sizes[iteration])] - added for iteration in changed)
        candidates.append((savings, added, len(changed)))
    removable, chosen, choices = max(candidates)
    return {
        "iterations": len(iterations), "at_budget": shape_counts[highest],
        "cuda_ms": total / 1e6, "cuda_ms_at_budget": full_time / 1e6,
        "best_single_extra_aligned_bucket": (
            {"rows": chosen, "affected_iterations": choices, "configured_padding_rows_removed": removable}
            if removable else None
        ),
    }


def summarize(root, choices_path):
    log = root / "qwen14-stage-trace/server.log"
    decisions, bindings = read_scheduler(log)
    gpu_time = prefill_times(log, decisions, bindings)
    raw = choices_path.read_bytes()
    choices = json.loads(raw)
    require(choices["traced_iterations"] == len(decisions), "trace and choices differ")
    sizes = {key: plan.selected_decode + sum(row.rows for row in plan.ready) for key, plan in decisions.items()}
    buckets = read_buckets(root)
    tails = {}
    for cell, summary in choices["cells"].items():
        rows = summary["rows"]
        eligible = sorted(row["client_ttft_ms"] for row in rows)
        cutoff = eligible[min(int(.95 * len(rows)), len(rows) - 1)]
        tail = [row for row in rows if row["client_ttft_ms"] >= cutoff]
        require(len(tail) == summary["at_or_above_ttft_p95"]["requests"], "trace and tail coverage differ")
        waiting = {
            plan.iteration for row in tail for plan, item in bindings[row["request_id"]]
            if item.started and not item.rows and plan.iteration in gpu_time
        }
        tails[cell] = {"requests": len(tail), **describe(waiting, sizes, gpu_time, buckets)}
    return {
        "scheduler_choice_summary_sha256": sha256(raw).hexdigest(), "existing_buckets": buckets,
        "all_prefill": describe(set(gpu_time), sizes, gpu_time, buckets),
        "ttft_tail_wait_between_chunks": tails,
        "scope": "Plans are deduplicated within each workload. Matching a configured bucket is a static bound on "
                 "padding rows. It does not establish which optional CUDA shape prepared or predict a speedup.",
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("choices", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.choices), indent=2, allow_nan=False))
