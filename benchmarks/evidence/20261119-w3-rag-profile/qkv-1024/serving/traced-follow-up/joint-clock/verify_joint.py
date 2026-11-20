"""Join request clocks to the server's CUDA kernels in the same measured runs."""

from argparse import ArgumentParser
from bisect import bisect_left
from collections import defaultdict, Counter
from hashlib import file_digest
import json
from pathlib import Path
import re
import sqlite3
from statistics import median

from classify_full_symbols import category
from compare_profile_arms import digest, read, require, ratio_percent
from profile_summary import anchors, clock_offset, union_ns
from summarize_profile_arm import stable_server
from summarize_request_stages import read_phases, prefill_union, request_phases


CELLS = ("rag-c8-s2", "wave-50ms")
MODES = ("riley-first", "qkv-first")
MATCH_FIELDS = ("request_index", "input_sha256", "prompt_tokens", "output_sha256", "completion_tokens", "finish")


def origin_bracket(results, phases):
    bounds = []
    for workload in results.values():
        for row in workload["measured"]:
            events = phases[row["request_id"]]
            (front, _), = events["front_submit"]
            (end, _), = events["first_delta"]
            # The client starts before sending any bytes and sees content after first_delta.
            begin = round(row["start"] * 1e9)
            first = round((row["start"] + row["ttft"]) * 1e9)
            require(begin <= first and front <= end, "invalid request clock bracket")
            bounds.append((begin - front, first - end))
    low = max(a for a, _ in bounds) - 2
    high = min(b for _, b in bounds) + 2
    require(low <= high and high - low < 2_000_000, f"request clock bracket is not bounded below 2 ms: {high-low}ns")
    return low, high


def nsight_bracket(marks, workload):
    clock_offset(marks, workload["anchors"])
    bounds = [(marks[stamp["name"]] - stamp["after_ns"], marks[stamp["name"]] - stamp["before_ns"])
              for stamp in workload["anchors"]]
    return min(a for a, _ in bounds), max(b for _, b in bounds)


def gpu_spans(sqlite, receipt, pid):
    with sqlite.open("rb") as handle:
        require(file_digest(handle, "sha256").hexdigest() == receipt["sqlite_sha256"], "export digest differs")
    with sqlite3.connect(f"{sqlite.resolve().as_uri()}?mode=ro", uri=True) as db:
        marks = anchors(db)
        symbols = dict(db.execute("SELECT id,value FROM StringIds"))
        rows = db.execute(
            "SELECT globalPid,start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE end > start AND (globalPid = ? OR ((globalPid >> 24) & 16777215) = ?) ORDER BY start,end",
            (pid, pid),
        )
        groups = {}
        actual = []
        owners = set()
        for owner, begin, end, symbol in rows:
            owners.add(owner)
            if symbol not in groups:
                name = symbols[symbol]
                group = category(name)
                if group == "other" and re.search(r"(?:kv(?:_cache)?_write|split.*(?:bias.*rope|rope.*bias))", name, re.I):
                    group = "qkv_bias_rope_kv_write"
                groups[symbol] = (name, group)
            actual.append((begin, end, groups[symbol][1]))
    require(len(owners) == 1 and actual, "Nsight export has no unique Riley CUDA owner")
    counts = Counter(row[2] for row in actual)
    kv = sorted(name for name, group in groups.values() if group == "qkv_bias_rope_kv_write")
    return marks, actual, dict(counts), kv


def windows(phases):
    def at(name):
        (stamp, _), = phases[name]
        return stamp

    steps = [(start, end) for (start, _), (end, _) in
             zip(phases["prefill_start"], phases["prefill_return"], strict=True)]
    front, sent, worker, admitted, first = (at(k) for k in
        ("front_submit", "command_send", "worker_admit", "admitted", "first_delta"))
    interleaved = [(old[1], new[0]) for old, new in zip(steps, steps[1:])]
    return {
        "makes_own_prefill_progress": steps,
        "request_wait": [(sent, worker), (admitted, steps[0][0]), *interleaved],
        "other_server_time": [(front, sent), (worker, admitted), (steps[-1][1], first)],
    }


def clipped_kernel_time(windows, offset, starts, longest, kernels):
    intervals = defaultdict(list)
    for start, finish in windows:
        begin, end = start + offset[0], finish + offset[1]
        if end <= begin:
            continue
        left, right = bisect_left(starts, begin - longest), bisect_left(starts, end)
        for a, b, category_ in kernels[left:right]:
            if b > begin:
                intervals[category_].append((max(a, begin), min(b, end)))
    total = union_ns(span for item in intervals.values() for span in item)
    return {"all_kernels": total / 1e6, **{name: union_ns(spans) / 1e6 for name, spans in intervals.items()}}


def medians(rows, key):
    names = {name for row in rows for name in row[key]}
    return {name: median(row[key].get(name, 0) for row in rows) for name in sorted(names)}


def host_step_windows(log):
    rows = [
        {key: float(value) for key, value in re.findall(r"([a-z+]+)=(\d+(?:\.\d+)?)", line.split("|", 1)[0])}
        for line in log.splitlines() if line.startswith("HOSTSTEP us/step over 128 steps:")
    ]
    require(rows and all(len(row) == 12 and row.keys() == rows[0].keys() for row in rows),
            "host step blocks are missing or do not agree")
    return {"steps_per_block": 128, "completed_blocks": len(rows),
            "median_us_per_step": {field: median(row[field] for row in rows) for field in rows[0]}}


def summarize(root, prior, candidate=None, baseline_only=False):
    outcomes = {}
    modes = MODES[:1] if baseline_only else MODES
    binary = (root / "binary.sha256").read_text().split()[0]
    for mode in modes:
        stage = candidate if candidate is not None and mode == "qkv-first" else root
        require((stage / "binary.sha256").read_text().split()[0] == binary, "run used a different binary")
        path = stage / f"qwen14-{mode}"
        reference = prior / path.name
        receipt = read(path / "receipt.json")
        plan = read(path / "plan.json")
        control_plan = read(reference / "plan.json")
        args = {"RILEY_W3_REQUEST_TRACE": "1", "RILEY_HOST_STEP_TRACE": "1"}
        if mode.startswith("qkv-"):
            args["RILEY_W3_QKV1024"] = "1"
        require(receipt["status"] == "passed" and receipt["common_kv_blocks"] == 2576
                and receipt["model"] == "Qwen2.5-14B-Instruct" and receipt["holder_pid"] == 2431237
                and plan["env"] == args and plan["profile"] is True and tuple(plan["cells"]) == CELLS,
                f"{mode}: unexpected serving environment")
        old_receipt = read(reference / "receipt.json")
        runtime_inputs = ("riley", "riley-checkpoint.json", "config.json", "tokenizer.json", "load.py", "profile_workload.py")
        actual_hashes = {Path(name).name: checksum for name, checksum in receipt["sha256"].items()}
        historic_hashes = {Path(name).name: checksum for name, checksum in old_receipt["sha256"].items()}
        require(stable_server(plan["server"])[1:] == stable_server(control_plan["server"])[1:]
                and plan["cells"] == control_plan["cells"]
                and all(actual_hashes[k] == historic_hashes[k] for k in runtime_inputs)
                and actual_hashes["riley"] == binary, f"{mode}: archived input changed")
        require(read(path / "server-exit.json")["exit"] == 0, f"{mode}: server shutdown incomplete")
        log = (path / "server.log").read_text()
        markers = [line for line in log.splitlines() if line.startswith("RILEY_W3_QKV1024")]
        require(bool(markers) == mode.startswith("qkv-") and
                all(line == "RILEY_W3_QKV1024 recorded=1 cuda=120405" for line in markers),
                f"{mode}: did not observe pinned GEMM selection")
        parsed = read_phases(path / "server.log")
        observed_prefill = prefill_union(parsed)
        results = {name: read(path / f"{name}.json") for name in CELLS}
        origin = origin_bracket(results, parsed)
        marks, kernels, counts, kv = gpu_spans(path / "trace.sqlite", receipt, int((path / "server.pid").read_text()))
        starts = [row[0] for row in kernels]
        longest = max(stop - start for start, stop, _ in kernels)
        derived = {}
        for name in CELLS:
            work = results[name]
            earlier = read(reference / f"{name}.json")
            require(digest(path / f"{name}.json") == receipt["results"][name], f"{mode}/{name}: trace workload changed")
            measured, old = work["measured"], earlier["measured"]
            require(all(all(a[key] == b[key] for key in MATCH_FIELDS) for a,b in zip(measured, old, strict=True)),
                    f"{mode}/{name}: request or generated bytes differ from the prior run")
            lo, hi = nsight_bracket(marks, work)
            bracket = (origin[0] + lo, origin[1] + hi)
            median_offset = sum(bracket) // 2
            require(bracket[1] - bracket[0] < 3_000_000, f"{mode}/{name}: request-to-Nsight clock uncertainty too large")
            stats = work["statistics"]
            rows = []
            for item in measured:
                request = parsed[item["request_id"]]
                phases = request_phases(request, item["prompt_tokens"], observed_prefill)
                stages = windows(request)
                result = {}
                for group, window in stages.items():
                    point = clipped_kernel_time(window, (median_offset, median_offset), starts, longest, kernels)
                    low = clipped_kernel_time(window, bracket[::-1], starts, longest, kernels)
                    high = clipped_kernel_time(window, bracket, starts, longest, kernels)
                    uncertainty = max((high[name] - low.get(name, 0) for name in high), default=0)
                    result[group] = {"gpu_ms": point, "worst_boundary_span_ms": uncertainty}
                rows.append({"index": item["request_index"], "ttft_ms": item["ttft"] * 1000,
                             "stages_ms": phases["stages_ms"], "cuda": result})
            tail = [row for row in rows if row["ttft_ms"] >= stats["ttft_p95_ms"]]
            require(tail and min(row["ttft_ms"] for row in tail) == stats["ttft_p95_ms"],
                    f"{mode}/{name}: TTFT boundary differs from measured workload")
            derived[name] = {
                "requests": len(rows), "matched_archived_requests": len(rows),
                "request_to_nsys_bracket_ms": (bracket[1] - bracket[0]) / 1e6,
                "ttft_p95_ms": stats["ttft_p95_ms"],
                "ttft_p95_change_vs_request_trace_only_percent": ratio_percent(
                    stats["ttft_p95_ms"], earlier["statistics"]["ttft_p95_ms"]),
                "tail_request_indices": [row["index"] for row in tail],
                "tail_cpu_and_wait_stages_ms_p50": medians(tail, "stages_ms"),
                "tail_gpu_category_ms_p50": {
                    group: medians([row["cuda"][group] for row in tail], "gpu_ms") for group in tail[0]["cuda"]
                },
                "tail_gpu_worst_clock_boundary_span_ms": max(
                    item["worst_boundary_span_ms"] for row in tail for item in row["cuda"].values()),
                "focal_request_41": next((row for row in rows if row["index"] == 41), None) if name == "wave-50ms" else None,
            }
        outcomes[mode] = {
            "sqlite_sha256": receipt["sqlite_sha256"], "nsys_report_sha256": receipt["raw_report_sha256"],
            "server_log_sha256": digest(path / "server.log"),
            "host_steps_whole_process": host_step_windows(log),
            "kernel_counts_by_category": counts, "fused_rope_kv_symbols": kv, "cells": derived,
        }
    return {"scope": "Shared GPU kernel times aligned to the measured server request stages; category unions may overlap.",
            "arm_order": modes, "baseline_binary_sha256": binary,
            "arms": outcomes}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("prior", type=Path)
    cli.add_argument("--candidate", type=Path)
    cli.add_argument("--baseline-only", action="store_true")
    args = cli.parse_args()
    print(json.dumps(summarize(args.root.resolve(), args.prior.resolve(), args.candidate, args.baseline_only),
                     indent=2, allow_nan=False))
