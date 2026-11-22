"""Match HTTP worker queue observations to fixed C32 arrivals and compare capacity."""

import json
from pathlib import Path
import sys

from budget_reference import batch_shapes, regression_checks, requests_awaiting_last_content_at_write
from comparison_base import compare, p
from observation_reference import bind_observations, compare_cohort
from private_reference import check_case
from replay_verifier import check_replay
from verify_a100_candidate import digest, read, require
from warm_reference import checked_reservation
from workload import REQUEST_COUNTS, make_workloads

ORDER = ("off-warm-256-w32", "private-256-w32", "off-warm-256-w64", "private-256-w64",
         "private-128-w32", "private-128-w64")
ORDERS = {"6": ORDER[::-1], "7": ORDER}
COMPARISONS = {
    "workers_off_256": ("off-warm-256-w32", "off-warm-256-w64"),
    "workers_private_256": ("private-256-w32", "private-256-w64"),
    "workers_private_128": ("private-128-w32", "private-128-w64"),
    "cache_256_w32": ("off-warm-256-w32", "private-256-w32"),
    "cache_256_w64": ("off-warm-256-w64", "private-256-w64"),
    "row_budget_w32": ("private-256-w32", "private-128-w32"),
    "row_budget_w64": ("private-256-w64", "private-128-w64"),
}


def join_http_queue(log, sample, workers):
    records = {}
    require([line for line in log.splitlines() if line.startswith("W3-HTTP-WORKERS ")]
            == [f"W3-HTTP-WORKERS {workers}"], "actual HTTP worker count changed")
    for line in log.splitlines():
        if line.startswith("W3-HTTP-QUEUE "):
            _, request, raw_ns = line.split()
            ns = int(raw_ns)
            require(request not in records and ns >= 0, f"invalid HTTP queue duration: {request}")
            records[request] = ns / 1e6
    require(len(records) == sum(REQUEST_COUNTS.values()), "HTTP queue observation count changed")
    result = {name: [records.pop(row["response_id"]) for row in sample[name]["rows"]]
              for name in REQUEST_COUNTS}
    require(not records, "HTTP queue observation missing a client request")
    return result


def queue_summary(values):
    return {
        "p50_ms": p(values, .50), "p95_ms": p(values, .95),
        "at_least_1ms": sum(value >= 1 for value in values),
    }


def compare_http_queue(before, after, indices):
    base, candidate = ([rows[i] for i in indices] for rows in (before, after))
    deltas = [right - left for left, right in zip(base, candidate, strict=True)]
    return {"before": queue_summary(base), "after": queue_summary(candidate),
            "paired_delta_p50_ms": p(deltas, .50), "paired_delta_mean_ms": sum(deltas) / len(deltas)}


def verify(stage):
    evidence = stage.parent
    build = Path((stage / "build-path.txt").read_text().strip())
    parent = evidence / "w5-prefix-metrics-package-20261121.nEFyze5I/cpu-evidence/report.json"
    stock = read(parent)
    original = evidence / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first/workload.json"
    reference_hash = digest(original)
    require(reference_hash == "a77aa2092bd229a40038622d1a2de6768d2a8a6fdca159fbff42c20d9d6cfb7b", "reference output changed")
    require(stock["status"] == "passed" and stock["source_revision"] == "3951d90c8df381daef2ab88fd18ac9e8a83c4554",
            "parent report changed")
    toolchain = (build / "toolchain.txt").read_text()
    require(toolchain.startswith("rustc 1.85.0") and "V12.4." in toolchain, "unexpected toolchain")
    build_checks = {"input_sha256": digest(build / "build-inputs.sha256"),
                    "result_sha256": digest(build / "build-result.sha256"),
                    "artifacts": {name: sha for sha, name in
                                  (line.split("  ", 1) for line in (build / "build-result.sha256").read_text().splitlines())}}
    data, schedules, workloads = read(original), read(stage / "schedules.json"), make_workloads()
    timeline = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                if row.startswith(("START ", "END "))]
    require(len(timeline) == 24 and all(row[-1] == "exit=0" for row in timeline if row[0] == "END"),
            "campaign sequence incomplete")
    runs, reservations, comparisons, cache_checks = {}, {}, {}, {}
    for index, order in ORDERS.items():
        events = [row for row in timeline if row[1].endswith(f"-gpu{index}")]
        require([row[:2] for row in events] == [[phase, f"{label}-gpu{index}"]
                                               for label in order for phase in ("START", "END")]
                and [row[2] for row in events] == sorted(row[2] for row in events),
                "GPU-local sequence changed")
        owner = checked_reservation(stage / f"gpu{index}-before.json", index)
        require(checked_reservation(stage / f"gpu{index}-after.json", index) == owner, "GPU owner changed")
        reservations.update(owner)
        raw, arms, timings, queues = {}, {}, {}, {}
        for label in order:
            condition, budget, worker_label = label.rsplit("-", 2)
            budget, workers = int(budget), int(worker_label[1:])
            tag = f"{label}-gpu{index}"
            folder = stage / tag
            receipt = read(folder / "receipt.json")
            require(receipt["steady_budget"] == budget and receipt["http_workers"] == workers,
                    f"HTTP worker or row budget changed for {tag}")
            sample, arm = check_case(stage, condition, index, owner, stock, parent, build_checks, data, workloads,
                                     label=tag, budget=budget, require_stock=label == "off-warm-256-w32")
            shapes = arm["short_gpu_by_shape"]
            require(shapes and max(map(int, shapes)) == budget and arm["cells"]["short"]["prompt_tokens_max"] < 128,
                    f"C32 did not reach the steady row cap for {tag}")
            server_log = (folder / "server.log").read_text()
            queue = join_http_queue(server_log, sample, workers)
            traced = bind_observations(server_log, sample)
            arm["observations"] = {cell: len(items) for cell, items in traced.items()}
            arm["http_queue_by_cell"] = {name: queue_summary(values) for name, values in queue.items()}
            arm["http_workers"] = workers
            arm["steady_budget"] = budget
            arm["rag_gpu_by_shape"] = batch_shapes(sample["prime"], sample["rag"])
            arm["fidelity"] = check_replay(sample["short"], schedules[index])
            arm["requests_awaiting_last_content_at_write"] = requests_awaiting_last_content_at_write(sample["short"]["rows"])
            raw[label], arms[label], timings[label], queues[label] = sample, arm, traced["short"], queue["short"]
            runs[tag] = arm

        comparisons[index] = {}
        for name, (before, after) in COMPARISONS.items():
            result = compare(raw[before], raw[after], arms[before], arms[after])
            errors = [abs(left["relative_write_error_ms"] - right["relative_write_error_ms"])
                      for left, right in zip(raw[before]["short"]["rows"], raw[after]["short"]["rows"], strict=True)]
            paired_error = p(errors, .95)
            require(paired_error <= 5, "paired write fidelity failed")
            first = arms[before]["fidelity"]["first_32_indices"]
            later = sorted(set(range(96)) - set(first))
            result["paired_write_error_p95_ms"] = paired_error
            result["short_cohorts"] = {label: compare_cohort(timings[before], timings[after], indices)
                                       for label, indices in (("first_32", first), ("remaining_64", later))}
            result["http_queue"] = {label: compare_http_queue(queues[before], queues[after], indices)
                                    for label, indices in (("all", range(96)), ("first_32", first), ("remaining_64", later))}
            comparisons[index][name] = result
        cache_checks[index] = {key: regression_checks(result) for key, result in comparisons[index].items()
                               if key.startswith("cache_")}
    return {"status": "passed", "remote_evidence": str(stage), "remote_build": str(build),
            "source_revision": stock["source_revision"], "image_id": stock["image_id"], "build": build_checks,
            "stock_workload_sha256": reference_hash, "schedules_sha256": digest(stage / "schedules.json"),
            "reservations": reservations, "order": timeline, "runs": runs, "comparisons": comparisons,
            "cache_checks": cache_checks}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
