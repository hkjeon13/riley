"""Match scheduler readiness and picks to executed Qwen2.5-14B prefill steps."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from statistics import median

from compare_profile_arms import require
from summarize_request_stages import summarize as summarize_stages


@dataclass(frozen=True)
class Prefill:
    request: int
    ready_since: int
    remaining: int
    started: int
    rows: int
    plateau_trimmed: int


@dataclass(frozen=True)
class Decision:
    iteration: int
    now: int
    budget: int
    max_chunk: int
    ready_decode: int
    selected_decode: int
    aging_threshold: int
    aging_allowed: int
    aging_winner: int
    hold_new_prompts: int
    atomic: int
    ready: tuple


def parse_decision(fields):
    decision = Decision(
        *map(int, fields[:11]),
        tuple(Prefill(*map(int, item.split(":"))) for item in fields[11:]),
    )
    require(decision.iteration > 0 and 0 < decision.max_chunk and 0 < decision.budget
            and decision.aging_allowed in (0, 1) and decision.atomic in (0, 1)
            and decision.hold_new_prompts in (0, 1)
            and 0 <= decision.selected_decode <= decision.ready_decode
            and (not decision.aging_allowed or decision.ready_decode > 0)
            and bool(decision.ready), f"invalid decision {decision.iteration}")
    require(all(0 <= item.ready_since <= decision.now and 0 < item.request
                and 0 < item.remaining and item.started in (0, 1)
                and item.plateau_trimmed in (0, 1)
                and 0 <= item.rows <= min(item.remaining, decision.max_chunk) for item in decision.ready),
            f"invalid candidates in decision {decision.iteration}")
    ordering = [(item.ready_since, item.request) for item in decision.ready]
    require(ordering == sorted(ordering) and len({request for _, request in ordering}) == len(ordering),
            f"invalid readiness order in decision {decision.iteration}")
    winner = next((item.request for item in decision.ready
                   if decision.now - item.ready_since >= decision.aging_threshold), 0)
    require(decision.aging_winner == (winner if decision.aging_allowed else 0),
            f"aging winner does not match readiness in decision {decision.iteration}")

    # Replay the allocations before plateau trimming. Plateau only subtracts
    # rows from the tail and cannot run in an aging-override iteration.
    budget = decision.budget
    planned = {}
    if decision.aging_winner:
        item = decision.ready[0]
        usable = budget if decision.atomic else max(1, budget - int(budget > 1))
        planned[item.request] = min(item.remaining, decision.max_chunk, usable)
        budget -= planned[item.request]
    decode = min(budget, decision.ready_decode)
    require(decision.selected_decode == decode, f"decode allocation differs in decision {decision.iteration}")
    budget -= decode
    ordinary = []
    for item in decision.ready:
        if item.request == decision.aging_winner or (decision.hold_new_prompts and not item.started):
            continue
        desired = min(item.remaining, decision.max_chunk)
        if budget and (not decision.atomic or desired <= budget):
            planned[item.request] = min(desired, budget)
            budget -= planned[item.request]
            ordinary.append(item)
    for item in decision.ready:
        initial = planned.get(item.request, 0)
        require((item.rows < initial and item in ordinary) if item.plateau_trimmed else item.rows == initial,
                f"prefill allocation differs in decision {decision.iteration}")
    trimmed = [item for item in ordinary if item.plateau_trimmed]
    if trimmed:
        require(not decision.aging_winner and trimmed == ordinary[-len(trimmed):]
                and all(item.rows == 0 for item in trimmed[1:]),
                f"plateau did not trim a suffix in decision {decision.iteration}")
    return decision


def read_scheduler(path):
    bindings, iterations = {}, {}
    executions = defaultdict(dict)
    stream = path.open() if path.exists() else gzip.open(path.with_suffix(".log.gz"), "rt")
    with stream as handle:
        for raw in handle:
            if not raw.startswith(("W3-REQUEST ", "W3-SCHED ", "W3-PREFILL ")):
                continue
            tag, *fields = raw.split()
            if tag == "W3-REQUEST":
                scheduler_id, completion_id = fields
                request = int(scheduler_id)
                require(request not in bindings, f"duplicate scheduler request {request}")
                bindings[request] = completion_id
            elif tag == "W3-SCHED":
                decision = parse_decision(fields)
                require(decision.iteration > next(reversed(iterations), 0), "iteration ordering changed")
                iterations[decision.iteration] = decision
            else:
                iteration, request, rows = map(int, fields)
                require(request not in executions[iteration], f"duplicate prefill in iteration {iteration}")
                executions[iteration][request] = rows

    require(not set(executions) - set(iterations), "executed prefill lacks a scheduler decision")
    by_request = defaultdict(list)
    for iteration, decision in iterations.items():
        planned = {item.request: item.rows for item in decision.ready if item.rows}
        require(planned == executions.get(iteration, {}), f"plan and execution differ in iteration {iteration}")
        for item in decision.ready:
            require(item.request in bindings, f"missing engine binding for request {item.request}")
            by_request[bindings[item.request]].append((decision, item))
    require(len(set(bindings.values())) == len(bindings), "engine request IDs repeat")
    return iterations, by_request


def request_choices(decisions, row):
    counts = Counter()
    picked_ages_ms = []
    picked = []
    for decision, item in decisions:
        age = decision.now - item.ready_since
        if item.rows:
            picked.append((decision, item))
            picked_ages_ms.append(age / 1e6)
            counts["picked_by_aging"] += int(decision.aging_winner == item.request)
            continue
        prefix = "between_chunks" if item.started else "before_first_chunk"
        counts[f"{prefix}_skipped"] += 1
        counts[f"{prefix}_decode_only"] += int(not any(other.rows for other in decision.ready))
        if age >= decision.aging_threshold:
            counts[f"{prefix}_waited_at_least_threshold"] += 1
            counts[f"{prefix}_aging_alternated"] += int(decision.ready_decode > 0 and not decision.aging_allowed)
            counts[f"{prefix}_other_aging_winner"] += int(decision.aging_winner not in (0, item.request))
        others = [other for other in decision.ready if other.rows]
        earlier = [other for other in others if (other.ready_since, other.request) < (item.ready_since, item.request)]
        counts[f"{prefix}_older_ready_prefill_ran"] += int(bool(earlier))
        counts[f"{prefix}_later_admitted_older_ready_prefill_ran"] += int(
            any(other.request > item.request for other in earlier)
        )
        counts[f"{prefix}_coalesced"] += int(decision.hold_new_prompts and not item.started)
        counts[f"{prefix}_plateau_trimmed_to_zero"] += item.plateau_trimmed

    for (decision, item), step in zip(picked, row["prefill_steps"], strict=True):
        require(item.rows == step["end"] - step["begin"]
                and item.started == bool(step["begin"])
                and decision.selected_decode == step["decode_rows"]
                and decision.selected_decode + sum(ready.rows for ready in decision.ready) == step["batch_rows"],
                f"request {row['request_id']} prefill phase disagrees with iteration {decision.iteration}")
    return {"request_index": row["request_index"], "request_id": row["request_id"],
            "prompt_tokens": row["prefill_steps"][-1]["end"], "client_ttft_ms": row["client_ttft_ms"],
            "prefill_iterations": [decision.iteration for decision, _ in picked],
            "picked_ready_age_ms": picked_ages_ms, "observations": dict(counts)}


def aggregate(rows):
    totals = Counter()
    for row in rows:
        totals.update(row["observations"])
    return {"requests": len(rows), "picked_prefill_steps": sum(len(row["prefill_iterations"]) for row in rows),
            "picked_ready_age_ms_p50": median(age for row in rows for age in row["picked_ready_age_ms"]),
            "observations": dict(totals)}


def summarize(root):
    stages = summarize_stages(root)
    control, control_requests = read_scheduler(root / "qwen14-riley/server.log")
    decisions, requests = read_scheduler(root / "qwen14-stage-trace/server.log")
    require(not control and not control_requests and decisions, "unexpected control or trace coverage")
    result = {}
    for name, cell in stages["cells"].items():
        rows = [request_choices(requests[row["request_id"]], row) for row in cell["rows"]]
        tail_indices = cell["at_or_above_client_ttft_p95"]["request_indices"]
        tail = [row for row in rows if row["request_index"] in tail_indices]
        result[name] = {"all": aggregate(rows), "at_or_above_ttft_p95": aggregate(tail), "rows": rows}
    return {"model": stages["model"], "physical_gpu": stages["physical_gpu"],
            "build_sha256": stages["build_sha256"], "traced_iterations": len(decisions),
            "observations_scope": "One ready request per iteration. Observation counts can overlap.",
            "cells": result}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
