"""Measure the CUDA time of prefill iterations that ran while another prompt waited."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from compare_profile_arms import require
from summarize_request_stages import read_phases
from summarize_scheduler_choices import read_scheduler


def prefill_times(path, decisions, by_request):
    phases = read_phases(path)
    gpu_time = {}
    observed = defaultdict(set)
    for identifier, samples in by_request.items():
        picks = [(plan, item) for plan, item in samples if item.rows]
        if not picks:
            continue
        require(identifier in phases, f"missing prefill phases for {identifier}")
        for (plan, item), (start, shape), (end, done) in zip(
            picks, phases[identifier]["prefill_start"], phases[identifier]["prefill_return"], strict=True
        ):
            first, last, _final, rows, decode = shape
            completed, elapsed_ns = done
            require(last - first == item.rows and last == completed
                    and decode == plan.selected_decode and rows == decode + sum(row.rows for row in plan.ready)
                    and 0 <= elapsed_ns <= end - start,
                    f"prefill duration or plan shape disagrees at iteration {plan.iteration}")
            require(gpu_time.setdefault(plan.iteration, elapsed_ns) == elapsed_ns,
                    f"CUDA timer differs within iteration {plan.iteration}")
            observed[plan.iteration].add(item.request)
    for iteration, plan in decisions.items():
        expected = {item.request for item in plan.ready if item.rows}
        require(observed.get(iteration, set()) == expected, f"incomplete GPU timing for iteration {iteration}")
    return gpu_time


def wait_time(samples, gpu_time):
    counts = Counter()
    time_ns = Counter()
    for plan, item in samples:
        if item.rows or plan.iteration not in gpu_time:
            continue
        time = gpu_time[plan.iteration]
        stage = "between_chunks" if item.started else "before_first_chunk"
        age = plan.now - item.ready_since
        earlier_ready = [other for other in plan.ready if other.rows
                         and (other.ready_since, other.request) < (item.ready_since, item.request)]
        tags = [stage]
        later = [other for other in earlier_ready if other.request > item.request]
        if later:
            tags.append(f"{stage}_later_admitted_older_ready")
            # Count shape-preserving opportunities in the observed plan only.
            # Leave the recorded decode rows and the aging winner in place.
            ordinary_rows = sum(other.rows for other in plan.ready if other.request != plan.aging_winner)
            if (item.started and item.remaining <= min(512, plan.max_chunk, ordinary_rows)
                    and any(other.request != plan.aging_winner for other in later)):
                tags.append("between_chunks_finish_with_non_aging_rows")
        if age >= plan.aging_threshold:
            tags.append(f"{stage}_past_threshold")
            if plan.ready_decode and not plan.aging_allowed:
                tags.append(f"{stage}_past_threshold_aging_alternated")
            elif plan.aging_winner not in (0, item.request):
                tags.append(f"{stage}_past_threshold_other_aging_winner")
        for tag in tags:
            time_ns[tag] += time
            counts[tag] += 1
    return counts, time_ns


def cell(rows, by_request, gpu_time):
    requests = [wait_time(by_request[row["request_id"]], gpu_time) for row in rows]
    keys = sorted(set().union(*(counts for counts, _ in requests)))
    return {
        "requests": len(rows),
        "overlapping_prefill_steps": {key: sum(counts[key] for counts, _ in requests) for key in keys},
        "requests_with_overlap": {key: sum(counts[key] > 0 for counts, _ in requests) for key in keys},
        "overlapping_prefill_cuda_time_ms_per_request_p50": {
            key: median(values[key] for _, values in requests) / 1e6 for key in keys
        },
    }


def summarize(root, choices_path):
    path = root / "qwen14-stage-trace/server.log"
    decisions, by_request = read_scheduler(path)
    gpu_time = prefill_times(path, decisions, by_request)
    raw_choices = choices_path.read_bytes()
    choices = json.loads(raw_choices)
    require(choices["traced_iterations"] == len(decisions), "scheduler summary uses a different trace")
    cells = {}
    for name, choice in choices["cells"].items():
        rows = choice["rows"]
        boundary = sorted(row["client_ttft_ms"] for row in rows)[min(int(.95 * len(rows)), len(rows) - 1)]
        tail = [row for row in rows if row["client_ttft_ms"] >= boundary]
        require(len(tail) == choice["at_or_above_ttft_p95"]["requests"], "different TTFT tail")
        for row in rows:
            expected = [plan.iteration for plan, item in by_request[row["request_id"]] if item.rows]
            require(row["prefill_iterations"] == expected, "request summary differs from scheduler decisions")
        cells[name] = {"all": cell(rows, by_request, gpu_time), "at_or_above_ttft_p95": cell(tail, by_request, gpu_time)}
    return {"scheduler_choice_summary_sha256": sha256(raw_choices).hexdigest(),
            "timed_prefill_iterations": len(gpu_time),
            "prefill_iterations_with_zero_cuda_time": sum(time == 0 for time in gpu_time.values()),
            "cells": cells,
            "scope": "CUDA timers cover the whole submitted iteration, including any decode rows. "
                     "Each waiting request counts a selected step once; classification groups can overlap. "
                     "The 512-row completion counts are static opportunities in existing plans."}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("choices", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.choices)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
