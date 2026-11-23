"""Bound a two-request prefill split on each observed single-request wide step."""

from collections import Counter
import json
from pathlib import Path

from check_opportunities import read_workloads


HERE = Path(__file__).resolve().parent


def causal_pairs(rows, prefix):
    return rows * prefix + rows * (rows + 1) // 2


def split_bound(rows, head_prefix, other_prefix, remaining):
    # For k rows moved to the earlier request, the change in logical work is
    # k * (other_prefix - head_prefix - rows + k).
    cap = min(rows, remaining)
    bound = min(cap, max(0, rows + head_prefix - other_prefix))
    baseline = causal_pairs(rows, head_prefix)
    reordered = causal_pairs(rows - bound, head_prefix) + causal_pairs(bound, other_prefix)
    if reordered > baseline:
        raise ValueError("split bound raised logical query–key work")
    if bound < cap:
        next_rows = bound + 1
        if causal_pairs(rows - next_rows, head_prefix) + causal_pairs(next_rows, other_prefix) <= baseline:
            raise ValueError("split bound excludes a valid reassignment")
    return bound, reordered - baseline


def compare(steps, submitted, prompt_lengths, measured, activation):
    consumed = Counter()
    candidates = []
    minimum = activation["long_prompt_tokens_min"]
    for step in steps:
        if (step.rows == 1024 and len(step.prefill) == 1
                and activation["decode_rows_min"] <= step.decode <= activation["decode_rows_max"]):
            head, rows, _ = step.prefill[0]
            if (rows == step.rows - step.decode and rows >= activation["prefill_budget_min"]
                    and prompt_lengths.get(head, 0) >= minimum):
                alternatives = []
                for request in sorted(prompt_lengths):
                    length, prefix = prompt_lengths[request], consumed[request]
                    if request >= head or submitted[request] > step.start or length < minimum or not 0 < prefix < length:
                        continue
                    remaining = length - prefix
                    limit, delta = split_bound(rows, consumed[head], prefix, remaining)
                    if limit:
                        index = measured.get(request)
                        alternatives.append({
                            "scheduler_request": request, "request_index": index, "prefix_tokens": prefix,
                            "remaining_tokens": remaining, "rows_at_bound": limit,
                            "logical_pair_delta_at_bound": delta, "would_finish_prefill": limit == remaining,
                        })
                if alternatives:
                    candidates.append({
                        "step": step.id, "decode_rows": step.decode, "prefill_rows": rows,
                        "original_scheduler_request": head, "original_request_index": measured.get(head),
                        "original_prefix_tokens": consumed[head], "alternatives": alternatives,
                    })
        for request, rows, _ in step.prefill:
            consumed[request] += rows
    touched_measured = [step for step in candidates if any(row["request_index"] is not None
                        for row in step["alternatives"])]
    return {
        "steps_with_a_nonincreasing_split": len(candidates),
        "steps_that_could_advance_a_measured_request": len(touched_measured),
        "steps_that_could_finish_a_measured_prefill": sum(any(row["request_index"] is not None
            and row["would_finish_prefill"] for row in step["alternatives"]) for step in candidates),
        "measured_request_indexes": sorted({row["request_index"] for step in candidates
                                            for row in step["alternatives"] if row["request_index"] is not None}),
        "steps": candidates,
    }


def audit():
    registration = json.loads((HERE / "registration.json").read_bytes())
    steps, submitted, workloads = read_workloads()
    return {
        "gpu": 7, "source_committed_steps": len(steps),
        "scope": "independent arithmetic bound at observed baseline cursors",
        "includes_http_warmup": True, "accounts_for_aging_override": False,
        "schedules": {
            label: compare(steps, submitted, prompt_lengths, measured, registration["activation"])
            for label, (prompt_lengths, measured) in workloads.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
