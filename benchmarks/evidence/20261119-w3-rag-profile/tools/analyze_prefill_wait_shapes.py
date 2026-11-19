"""Attribute recorded 14B prefill wait to the shapes occupying each interval."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from math import isclose
from pathlib import Path
from statistics import median

from summarize_request_stages import overlap, prefill_union, read, read_phases, require, summarize


def merge_spans(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def shape_intervals(events, steady_budget):
    spans = defaultdict(list)
    for phases in events.values():
        for (start, fields), (end, _) in zip(phases["prefill_start"], phases["prefill_return"], strict=True):
            first, stop, _, rows, decode = fields
            require(start <= end and 0 <= decode <= rows and 0 < stop - first <= rows - decode,
                    "invalid prefill step interval")
            decode_label = "prefill_only" if decode == 0 else "decode_1_to_2" if decode <= 2 else "decode_over_2"
            width = "wide" if rows > steady_budget else "steady"
            spans[f"{width}_{decode_label}"].append((start, end))
    categorized = {key: merge_spans(value) for key, value in sorted(spans.items())}
    require(all(not any(overlap(start, end, right) for start, end in left)
                for left, right in combinations(categorized.values(), 2)), "step shapes overlap")
    require(merge_spans(interval for group in categorized.values() for interval in group) == prefill_union(events),
            "shape intervals do not cover the known prefill timeline")
    return categorized


def wait_shapes(phases, intervals, original):
    first = phases["prefill_start"][0][0]
    waits = {"command_queue": [(phases["command_send"][0][0], phases["worker_admit"][0][0])],
             "before_first_step": [(phases["admitted"][0][0], first)],
             "between_steps": [(end[0], start[0]) for end, start in
                               zip(phases["prefill_return"][:-1], phases["prefill_start"][1:], strict=True)]}
    results = {}
    for name, spans in waits.items():
        observed = {shape: sum(overlap(start, end, profile) for start, end in spans) / 1e6
                    for shape, profile in intervals.items()}
        require(isclose(sum(observed.values()), original[name], abs_tol=1e-9),
                "wait shape attribution differs from the source request trace")
        results[name] = observed
    return results


def analyze(root):
    summary = summarize(root)
    directory = root / "qwen14-stage-trace"
    server = read(directory / "plan.json")["server"]
    budget = int(server[server.index("--prefill-burst-steady-budget") + 1])
    events = read_phases(directory / "server.log")
    intervals = shape_intervals(events, budget)
    cohorts = {}
    for name, cell in summary["cells"].items():
        requests = cell["at_or_above_client_ttft_p95"]["request_indices"]
        rows = [row for row in cell["rows"] if row["request_index"] in requests]
        require(rows and len(rows) == len(requests), "incomplete TTFT tail cohort")
        details = [{"request_index": row["request_index"],
                    **wait_shapes(events[row["request_id"]], intervals, row["other_prefill_during_wait_ms"])}
                   for row in rows]
        medians = {stage: {shape: median(row[stage][shape] for row in details) for shape in intervals}
                   for stage in ("command_queue", "before_first_step", "between_steps")}
        cohorts[name] = {"requests": len(rows), "median_overlap_ms": medians, "rows": details}
    log = directory / "server.log"
    if not log.exists():
        log = log.with_suffix(".log.gz")
    return {"binary_sha256": summary["build_sha256"], "source_trace_sha256": sha256(log.read_bytes()).hexdigest(),
            "physical_gpu": summary["physical_gpu"], "steady_budget_rows": budget, "cohorts": cohorts,
            "scope": "The recorded prefill interval includes host work. The analysis measures overlap, not a scheduling counterfactual."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
