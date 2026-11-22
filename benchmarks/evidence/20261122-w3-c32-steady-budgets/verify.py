"""Recompute performance and exact-output gates for the two steady row budgets."""

import json
from pathlib import Path
import sys

from check_distribution_gpu import current_batch_shapes
from comparison_base import compare, p
from observation_reference import bind_observations, compare_cohort
from private_reference import check_case
from replay_verifier import check_replay
from verify_a100_candidate import digest, read, require
from warm_reference import checked_reservation
from workload import REQUEST_COUNTS, make_workloads

ORDER = ("off-warm-256", "private-256", "private-192", "private-128")
ORDERS = {"6": ORDER[::-1], "7": ORDER}


def batch_shapes(before, after):
    start, end = map(current_batch_shapes, (before, after))
    require(start.keys() == end.keys(), "batch shape buckets changed")
    shapes = {}
    for rows, (steps, gpu_ns, _) in end.items():
        count = steps - start[rows][0]
        elapsed = gpu_ns - start[rows][1]
        require(count >= 0 and elapsed >= 0, "batch shape totals regressed")
        if count:
            shapes[str(rows)] = {"steps": count, "gpu_ns": elapsed}
    return shapes


def requests_awaiting_last_content_at_write(rows):
    ordered = sorted(rows, key=lambda row: row["sent_at"])
    pending = [sum(previous["start"] + previous["e2e"] > row["sent_at"] for previous in ordered[:index])
               for index, row in enumerate(ordered)]
    later = pending[32:]
    return {
        "max": max(pending),
        "remaining_64_p50": p(later, .50),
        "remaining_64_with_at_least_32": sum(count >= 32 for count in later),
    }


def regression_checks(comparison):
    short = comparison["short"]["change_percent"]
    rag = comparison["rag"]["change_percent"]
    return {
        "exact_outputs": all(comparison[cell]["matched"] == expected for cell, expected in REQUEST_COUNTS.items()),
        "short_ttft_p50": short["ttft_p50_ms"] < 2,
        "short_ttft_p95": short["ttft_p95_ms"] < 2,
        "short_decode_tpot": short["tpot_p50_ms"] < 1,
        "short_throughput": short["output_tok_s"] > -2,
        "rag_ttft_p95_improves": rag["ttft_p95_ms"] < 0,
        "rag_decode_tpot": rag["tpot_p50_ms"] < 1,
    }


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
    build_checks = {"input_sha256": digest(build / "build-inputs.sha256"),
                    "result_sha256": digest(build / "build-result.sha256"),
                    "artifacts": {name: sha for sha, name in
                                  (line.split("  ", 1) for line in (build / "build-result.sha256").read_text().splitlines())}}
    require(build_checks["artifacts"]["target/release/riley"] ==
            "717c0891c1854869a6433595bddb3754b2bd242b8515a703a3af74abfa7aad18", "a different binary was used")
    data, schedules, workloads = read(original), read(stage / "schedules.json"), make_workloads()
    timeline = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                if row.startswith(("START ", "END "))]
    require(len(timeline) == 16 and all(row[-1] == "exit=0" for row in timeline if row[0] == "END"),
            "campaign sequence incomplete")
    runs, reservations, comparisons, checks = {}, {}, {}, {}
    for index, order in ORDERS.items():
        events = [row for row in timeline if row[1].endswith(f"-gpu{index}")]
        require([row[:2] for row in events] == [[phase, f"{label}-gpu{index}"]
                                               for label in order for phase in ("START", "END")]
                and [row[2] for row in events] == sorted(row[2] for row in events), "GPU-local sequence changed")
        owner = checked_reservation(stage / f"gpu{index}-before.json", index)
        require(checked_reservation(stage / f"gpu{index}-after.json", index) == owner, "GPU owner changed")
        reservations.update(owner)
        raw, arms, timings = {}, {}, {}
        for label in order:
            condition, budget = label.rsplit("-", 1)
            budget = int(budget)
            tag = f"{label}-gpu{index}"
            folder = stage / tag
            receipt = read(folder / "receipt.json")
            require(receipt["steady_budget"] == budget, f"steady budget changed for {tag}")
            sample, arm = check_case(stage, condition, index, owner, stock, parent, build_checks,
                                     data, workloads, label=tag, budget=budget, require_stock=budget == 256)
            shapes = arm["short_gpu_by_shape"]
            require(shapes and max(map(int, shapes)) == budget and arm["cells"]["short"]["prompt_tokens_max"] < 128,
                    f"C32 did not reach the configured steady row cap for {tag}")
            arm["rag_gpu_by_shape"] = batch_shapes(sample["prime"], sample["rag"])
            arm["fidelity"] = check_replay(sample["short"], schedules[index])
            arm["requests_awaiting_last_content_at_write"] = requests_awaiting_last_content_at_write(sample["short"]["rows"])
            arm["steady_budget"] = budget
            traced = bind_observations((folder / "server.log").read_text(), sample)
            arm["observations"] = {cell: len(items) for cell, items in traced.items()}
            raw[label], arms[label], timings[label] = sample, arm, traced["short"]
            runs[tag] = arm

        comparisons[index], checks[index] = {}, {}
        for candidate in ORDER[1:]:
            comparisons[index][candidate] = {}
            baselines = ("off-warm-256",) if candidate == "private-256" else ("off-warm-256", "private-256")
            for baseline in baselines:
                result = compare(raw[baseline], raw[candidate], arms[baseline], arms[candidate])
                errors = [abs(left["relative_write_error_ms"] - right["relative_write_error_ms"])
                          for left, right in zip(raw[baseline]["short"]["rows"], raw[candidate]["short"]["rows"], strict=True)]
                paired_error = p(errors, .95)
                require(paired_error <= 5, "paired write fidelity failed")
                first = arms[baseline]["fidelity"]["first_32_indices"]
                later = sorted(set(range(96)) - set(first))
                result["paired_write_error_p95_ms"] = paired_error
                result["short_cohorts"] = {name: compare_cohort(timings[baseline], timings[candidate], indices)
                                           for name, indices in (("first_32", first), ("remaining_64", later))}
                comparisons[index][candidate][baseline] = result
            checks[index][candidate] = regression_checks(comparisons[index][candidate]["off-warm-256"])

    return {
        "status": "passed", "remote_evidence": str(stage), "remote_build": str(build), "source_revision": stock["source_revision"],
        "image_id": stock["image_id"], "build": build_checks, "stock_workload_sha256": reference_hash,
        "schedules_sha256": digest(stage / "schedules.json"), "reservations": reservations, "order": timeline,
        "runs": runs, "comparisons": comparisons, "regression_checks_vs_off_warm": checks,
        "passes_both_gpus": {arm: all(all(board[arm].values()) for board in checks.values()) for arm in ORDER[1:]},
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
