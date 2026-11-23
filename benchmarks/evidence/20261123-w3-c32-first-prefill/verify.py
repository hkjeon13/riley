"""Validate output, timing partitions, and both GPU leases for the C32 rerun."""
import json
from math import ceil
from pathlib import Path
import re
import sys

from comparison_base import compare, p
from http_reference import join_http_queue
from observation_reference import ADMISSION, AFTER_ADMISSION, bind_observations
from private_reference import check_case
from replay_verifier import check_replay
from stage_reference import join_short, trace_data
from verify_a100_candidate import digest, read, require
from warm_reference import checked_reservation
from workload import REQUEST_COUNTS, make_workloads


ORDER = ("128-w32-control", "128-w64-control", "128-w32-trace", "128-w64-trace",
         "256-w32-trace", "256-w64-trace")
ORDERS = {"6": ORDER[::-1], "7": ORDER}
CONTRASTS = {
    "workers_128_control": ("128-w32-control", "128-w64-control"),
    "workers_128_trace": ("128-w32-trace", "128-w64-trace"),
    "workers_256_trace": ("256-w32-trace", "256-w64-trace"),
    "budget_w32_trace": ("256-w32-trace", "128-w32-trace"),
    "budget_w64_trace": ("256-w64-trace", "128-w64-trace"),
    "trace_overhead_128_w32": ("128-w32-control", "128-w32-trace"),
    "trace_overhead_128_w64": ("128-w64-control", "128-w64-trace"),
}


def read_activation(log):
    activated = {}
    for line in log.splitlines():
        if line.startswith("W3-ACTIVATION "):
            _, raw_id, *times = line.split()
            request = int(raw_id)
            submitted, ready, planned = map(int, times)
            require(request not in activated and 0 <= submitted <= ready <= planned, "bad activation trace")
            activated[request] = (ready - submitted, planned - ready)
    return activated


def c32_timing(rows, log, queue, observations, traced):
    phases, prefill, dropped = trace_data(log)
    activated = read_activation(log)
    summaries = []
    if traced:
        require(dropped == [0], "trace was absent or truncated")
        summaries = join_short(rows, phases, prefill)
    else:
        require(not phases and not prefill and not dropped and not activated, "control emitted a stage trace")

    joined = []
    for index, (row, http, server) in enumerate(zip(rows, queue, observations, strict=True)):
        values = {
            "client_ttft": row["ttft"] * 1000,
            "http_connection_wait": http,
            "backend_submit_to_return": server[ADMISSION],
            "backend_return_to_first_delta": server[AFTER_ADMISSION],
        }
        if traced:
            trace = summaries[index]
            response_id = row["response_id"]
            match = re.fullmatch(r"cmpl-([a-f0-9]{16})", response_id)
            require(match is not None and response_id == trace["request_id"], "C32 request identity changed")
            request = int(match[1], 16)
            wait_ns, plan_wait_ns = activated[request]
            accepted, worker = (phases[request][field] for field in ("admitted", "worker"))
            first_start = prefill[request][0][1]
            require(worker <= accepted <= first_start and wait_ns + plan_wait_ns <= first_start - worker + 1_000,
                    "scheduler timing extends outside its enclosing engine interval")
            parts = trace["phases_ms"]
            values.update({
                "scheduler_wait_for_slot_or_kv": wait_ns / 1e6,
                "scheduler_ready_to_first_plan": plan_wait_ns / 1e6,
                "backend_accept_to_first_prefill": parts["admitted_to_first_prefill"],
                "prefill_cuda_on_batch": parts["prefill_batch_cuda"],
                "prefill_host_on_batch": parts["prefill_batch_host_remainder"],
                "between_prefill_batches": parts["between_prefill_batches"],
                "last_prefill_to_first_delta": parts["last_prefill_to_visible"],
                "first_delta_to_client": parts["visible_to_client_content"],
            })
        joined.append(values)
    require(len(joined) == 96, "fixed C32 request count changed")
    return joined


def distribution(values):
    return {"p50_ms": p(values, .50), "p95_ms": p(values, .95),
            "at_least_1ms": sum(value >= 1 for value in values)}


def activation_capacity(argv, rag_snapshot, prompt_max, block_size):
    max_active, output_limit, kv_blocks = (
        int(argv[argv.index(option) + 1]) for option in
        ("--max-active-sequences", "--max-output-tokens", "--kv-blocks")
    )
    short_block_bound = max_active * ceil((prompt_max + output_limit - 1) / block_size)
    drained = all(rag_snapshot[key] == 0 for key in
                  ("active_requests", "waiting_requests", "kv_allocated_blocks"))
    return {"active_slot_limit": max_active, "kv_limit": kv_blocks,
            "maximum_short_kv_promises": short_block_bound, "previous_phase_drained": drained,
            "kv_promise_limit_cannot_bind": drained and short_block_bound <= kv_blocks}


def pairs(left, right, indices):
    result = {}
    for field in left[0]:
        if field not in right[0]:
            continue
        before = [left[i][field] for i in indices]
        after = [right[i][field] for i in indices]
        result[field] = {"before": distribution(before), "after": distribution(after),
                         "paired_delta_p50_ms": p([b - a for a, b in zip(before, after, strict=True)], .50)}
    return result


def verify(stage):
    evidence = stage.parent
    build = Path((stage / "build-path.txt").read_text().strip())
    parent = evidence / "w5-prefix-metrics-package-20261121.nEFyze5I/cpu-evidence/report.json"
    stock = read(parent)
    original = evidence / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first/workload.json"
    reference_hash = digest(original)
    require(reference_hash == "a77aa2092bd229a40038622d1a2de6768d2a8a6fdca159fbff42c20d9d6cfb7b",
            "reference output changed")
    require(stock["status"] == "passed" and stock["source_revision"] == "3951d90c8df381daef2ab88fd18ac9e8a83c4554",
            "parent report changed")
    toolchain = (build / "toolchain.txt").read_text()
    require(toolchain.startswith("rustc 1.85.0") and "V12.4." in toolchain, "unexpected toolchain")
    kv_source = (build / "src/crates/riley-runtime/src/paged_kv.rs").read_text()
    block_size = int(re.search(r"^pub const KV_BLOCK_SIZE: usize = (\d+);$", kv_source, re.MULTILINE)[1])
    build_checks = {
        "input_sha256": digest(build / "build-inputs.sha256"),
        "result_sha256": digest(build / "build-result.sha256"),
        "artifacts": {name: value for value, name in
                      (row.split("  ", 1) for row in (build / "build-result.sha256").read_text().splitlines())},
    }
    schedules, workloads, canonical = read(stage / "schedules.json"), make_workloads(), read(original)
    timeline = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                if row.startswith(("START ", "END "))]
    require(len(timeline) == 24 and all(row[-1] == "exit=0" for row in timeline if row[0] == "END"),
            "campaign sequence incomplete")
    runs, reservations, contrasts = {}, {}, {}
    for gpu, layouts in ORDERS.items():
        events = [row[:2] for row in timeline if row[1].endswith(f"-gpu{gpu}")]
        require(events == [[action, f"private-{layout}-gpu{gpu}"]
                           for layout in layouts for action in ("START", "END")], "GPU order changed")
        owner = checked_reservation(stage / f"gpu{gpu}-before.json", gpu)
        require(checked_reservation(stage / f"gpu{gpu}-after.json", gpu) == owner, "GPU owner changed")
        reservations.update(owner)
        raw, results, timings = {}, {}, {}
        for layout in layouts:
            tokens, workers, mode = layout.split("-")
            budget, workers, traced = int(tokens), int(workers[1:]), mode == "trace"
            tag = f"private-{layout}-gpu{gpu}"
            folder = stage / tag
            receipt = read(folder / "receipt.json")
            require(receipt["http_workers"] == workers and receipt["steady_budget"] == budget
                    and receipt["stage_trace"] is traced, f"run configuration differs: {tag}")
            sample, arm = check_case(stage, "private", gpu, owner, stock, parent, build_checks,
                                     canonical, workloads, label=tag, budget=budget)
            require(arm["exact_stock_outputs"] == REQUEST_COUNTS, "shipped output parity failed")
            shape = arm["short_gpu_by_shape"]
            prompt_max = arm["cells"]["short"]["prompt_tokens_max"]
            require(shape and max(map(int, shape)) <= budget and prompt_max < 128,
                    f"row or short-prompt budget changed: {tag}")
            capacity = activation_capacity(receipt["server_arguments"], sample["rag"]["metrics"], prompt_max, block_size)
            log = (folder / "server.log").read_text()
            queue = join_http_queue(log, sample, workers)
            obs = bind_observations(log, sample)
            timings[layout] = c32_timing(sample["short"]["rows"], log, queue["short"], obs["short"], traced)
            fidelity = check_replay(sample["short"], schedules[gpu])
            require(sorted(fidelity["first_32_indices"]) == list(range(32)), "first 32 requests changed")
            raw[layout], results[layout] = sample, arm
            runs[tag] = {**arm, "fidelity": fidelity, "trace": traced, "http_workers": workers,
                         "steady_budget": budget, "activation_capacity": capacity}
        contrasts[gpu] = {}
        for name, (before, after) in CONTRASTS.items():
            changes = compare(raw[before], raw[after], results[before], results[after])
            errors = p([abs(a["relative_write_error_ms"] - b["relative_write_error_ms"])
                        for a, b in zip(raw[before]["short"]["rows"], raw[after]["short"]["rows"], strict=True)], .95)
            require(errors <= 5, "paired fixed-arrival error exceeded 5 ms")
            cohorts = {cohort: pairs(timings[before], timings[after], indices)
                       for cohort, indices in (("first_32", range(32)), ("remaining_64", range(32, 96)))}
            contrasts[gpu][name] = {**changes, "paired_write_error_p95_ms": errors, "c32_cohorts": cohorts}
    return {"status": "passed", "remote_evidence": str(stage), "remote_build": str(build), "build": build_checks,
            "result_sha256": digest(stage / "result.sha256"), "schedules_sha256": digest(stage / "schedules.json"),
            "stock_workload_sha256": reference_hash, "source_revision": stock["source_revision"],
            "image_id": stock["image_id"], "reservations": reservations, "order": timeline,
            "runs": runs, "comparisons": contrasts}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
