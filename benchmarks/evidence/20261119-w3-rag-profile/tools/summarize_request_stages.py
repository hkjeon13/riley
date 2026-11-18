"""Join Riley request-stage timestamps to the measured SSE request IDs."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

from compare_profile_arms import arm, ratio_percent, read, require
from summarize_profile_arm import stable_server


def read_phases(path):
    events = defaultdict(lambda: defaultdict(list))
    with path.open() as handle:
        for line in handle:
            if line.startswith("W3-PHASE "):
                _, request, stage, stamp, *fields = line.split()
                events[request][stage].append((int(stamp), [int(n) for n in fields]))
    return events


def request_phases(phases, prompt_tokens):
    def stamp(name):
        (value, fields), = phases[name]
        require(not fields, f"unexpected payload in {name}")
        return value

    front, sent, worker, admitted, delta = (
        stamp(name) for name in ("front_submit", "command_send", "worker_admit", "admitted", "first_delta")
    )
    tokenizer = "front" if phases.get("front_tokenize_start") else "worker"
    tokenizer_start = stamp(f"{tokenizer}_tokenize_start")
    tokenizer_end = stamp(f"{tokenizer}_tokenize_end")
    intervals = {"front": (front, sent), "worker": (worker, admitted)}
    encode_begin, encode_end = intervals[tokenizer]
    require(encode_begin <= tokenizer_start <= tokenizer_end <= encode_end,
            "tokenization is outside the recorded thread interval")
    require(not all(phases.get(f"{p}_tokenize_start") for p in intervals), "multiple tokenization paths")
    starts, returns = phases["prefill_start"], phases["prefill_return"]
    require(starts and len(starts) == len(returns), "prefill chunks are incomplete")
    last_timestamp, logical, wall, gpu = admitted, 0, 0, 0
    steps = []
    for (start, fields), (finish, output) in zip(starts, returns, strict=True):
        first, stop, final, batch_rows, decode_rows = fields
        completed, gpu_ns = output
        require(last_timestamp <= start <= finish and first == logical < stop <= prompt_tokens
                and completed == stop and final == int(stop == prompt_tokens)
                and 0 <= decode_rows <= batch_rows and batch_rows - decode_rows >= stop - first
                and 0 <= gpu_ns <= finish - start, "prefill range, shape or timing is incoherent")
        steps.append({"begin": first, "end": stop, "batch_rows": batch_rows,
                      "decode_rows": decode_rows, "gpu_ms": gpu_ns / 1e6})
        wall += finish - start
        gpu += gpu_ns
        last_timestamp, logical = finish, stop
    require(front <= sent <= worker <= admitted <= starts[0][0] <= last_timestamp <= delta
            and logical == prompt_tokens, "request lifecycle or final prompt coverage is incomplete")
    begin_prefill = starts[0][0]
    spent_tokenizing = tokenizer_end - tokenizer_start
    timing_ns = {
        "front_other": sent - front - (spent_tokenizing if tokenizer == "front" else 0),
        "tokenize": spent_tokenizing,
        "command_queue": worker - sent,
        "admission_other": admitted - worker - (spent_tokenizing if tokenizer == "worker" else 0),
        "scheduler_wait_before_prefill": begin_prefill - admitted,
        "prefill_executing": wall,
        "prefill_between_steps": last_timestamp - begin_prefill - wall,
        "after_prefill_to_first_delta": delta - last_timestamp,
    }
    require(sum(timing_ns.values()) == delta - front and not any(v < 0 for v in timing_ns.values()),
            "request stages do not partition the server interval")
    return {"tokenizer_thread": tokenizer, "server_ms": (delta - front) / 1e6,
            "shared_prefill_gpu_ms": gpu / 1e6, "prefill_steps": steps,
            "stages_ms": {k: v / 1e6 for k, v in timing_ns.items()}}


def stage_medians(requests):
    return {name: median(row["stages_ms"][name] for row in requests)
            for name in requests[0]["stages_ms"]}


def ttft_tail(requests, p95_ms):
    ranked = sorted(requests, key=lambda row: row["client_ttft_ms"])
    # Use the same observed order statistic as the load generator, including ties.
    first = min(int(.95 * len(ranked)), len(ranked) - 1)
    require(ranked[first]["client_ttft_ms"] == p95_ms, "TTFT p95 differs from the recorded requests")
    tail = [row for row in ranked if row["client_ttft_ms"] >= p95_ms]
    return {"requests": len(tail), "request_indices": [row["request_index"] for row in tail],
            "server_ms_p50": median(row["server_ms"] for row in tail),
            "stages_ms_p50": stage_medians(tail)}


def summarize(root):
    require((root / "campaign.exit").read_text().strip() == "0", "stage campaign did not complete")
    base, base_plan, base_cells = arm(root, "qwen14", "riley")
    focus, plan, cells = arm(root, "qwen14", "stage-trace")
    grouped = read_phases(root / "qwen14-stage-trace/server.log")
    require(not read_phases(root / "qwen14-riley/server.log"), "control arm contains request tracing")
    immutable = ("model", "gpu", "common_kv_blocks", "kv_cache_bytes")
    source_hashes = [{Path(name).name: value for name, value in receipt["sha256"].items()
                      if Path(name).name != "plan.json"} for receipt in (base, focus)]
    require(all(base[k] == focus[k] for k in immutable) and source_hashes[0] == source_hashes[1]
            and base_plan["cells"] == plan["cells"] and stable_server(base_plan["server"]) == stable_server(plan["server"])
            and base_plan["env"] == {} and plan["env"] == {"RILEY_W3_REQUEST_TRACE": "1"},
            "the arms use different serving inputs")
    require(all(read(root / f"qwen14-{mode}/server-exit.json")["exit"] == 0
                for mode in ("riley", "stage-trace")), "server did not stop gracefully")
    result = {}
    identifiers = set()
    fields = ("request_index", "input_sha256", "prompt_tokens", "completion_tokens", "output_sha256", "finish")
    for name, (stats, recorded) in cells.items():
        control, expected = base_cells[name]
        requests = []
        for row, baseline in zip(recorded, expected, strict=True):
            require(all(row[key] == baseline[key] for key in fields), f"{name}: request or output changed")
            identifier = row["request_id"]
            require(identifier not in identifiers and identifier in grouped, f"{name}: invalid trace identity")
            identifiers.add(identifier)
            requests.append({"request_index": row["request_index"], "request_id": identifier,
                             "client_ttft_ms": row["ttft"] * 1000,
                             **request_phases(grouped[identifier], row["prompt_tokens"])})
        result[name] = {"requests": len(requests), "matching_outputs": len(requests),
                        "ttft_p95_instrumented_ms": stats["ttft_p95_ms"],
                        "ttft_p95_control_ms": control["ttft_p95_ms"],
                        "ttft_p95_change_percent": ratio_percent(stats["ttft_p95_ms"], control["ttft_p95_ms"]),
                        "server_ms_p50": median(row["server_ms"] for row in requests),
                        "shared_prefill_gpu_ms_p50": median(row["shared_prefill_gpu_ms"] for row in requests),
                        "stages_ms_p50": stage_medians(requests),
                        "at_or_above_client_ttft_p95": ttft_tail(requests, stats["ttft_p95_ms"]),
                        "rows": requests}
    return {"model": focus["model"], "physical_gpu": 7, "build_sha256": source_hashes[0]["riley"],
            "scope": "Server phases from engine submission to first visible delta; GPU step durations are shared",
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
