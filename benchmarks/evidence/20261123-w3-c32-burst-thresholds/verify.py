"""Verify the bounded burst comparisons against the shipped output and 256-row control."""
import json
from pathlib import Path
import sys

from budget_reference import batch_shapes, regression_checks
from comparison_base import OUTPUT, compare, p
from http_reference import join_http_queue
from observation_reference import bind_observations
from private_reference import check_case
from replay_verifier import check_replay
from verify_a100_candidate import digest, read, require
from warm_reference import checked_reservation
from workload import REQUEST_COUNTS, make_workloads


BASE = "s256-d32-p512"
LOW_STEADY = "s128-d32-p512"
CANDIDATES = ("s128-d8-p64", "s128-d16-p64")
ORDER = (BASE, LOW_STEADY, *CANDIDATES)
ORDERS = {"6": ORDER[::-1], "7": ORDER}
STOCK_HASH = "a77aa2092bd229a40038622d1a2de6768d2a8a6fdca159fbff42c20d9d6cfb7b"
BINARY = "d9f7e375c4d2434e95e0a3b8ac9412919f777f5f4bd295548590620c2c8e809c"


def distribution(values, first):
    return {name: {"p50_ms": p([values[i] for i in indexes], .5),
                   "p95_ms": p([values[i] for i in indexes], .95)}
            for name, indexes in (
                ("first_32", first), ("remaining_64", sorted(set(range(96)) - set(first))))}


def output_diff(reference, measured):
    return {cell: [
        {"index": i, "fields": [name for name in OUTPUT if before[name] != after[name]]}
        for i, (before, after) in enumerate(zip(reference[cell]["rows"], measured[cell]["rows"], strict=True))
        if any(before[name] != after[name] for name in OUTPUT)]
        for cell in REQUEST_COUNTS}


def verify(stage):
    evidence = stage.parent
    build = Path((stage / "build-path.txt").read_text().strip())
    parent = evidence / "w5-prefix-metrics-package-20261121.nEFyze5I/cpu-evidence/report.json"
    stock = read(parent)
    original = evidence / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first/workload.json"
    require(digest(original) == STOCK_HASH and stock["status"] == "passed"
            and stock["source_revision"] == "3951d90c8df381daef2ab88fd18ac9e8a83c4554", "historical inputs changed")
    build_checks = {"input_sha256": digest(build / "build-inputs.sha256"),
                    "result_sha256": digest(build / "build-result.sha256"),
                    "artifacts": {name: check for check, name in
                                  (line.split("  ", 1) for line in (build / "build-result.sha256").read_text().splitlines())}}
    require(build_checks["artifacts"]["target/release/riley"] == BINARY, "binary changed")
    schedules, workloads, canonical = read(stage / "schedules.json"), make_workloads(), read(original)
    timeline = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                if row.startswith(("START ", "END "))]
    require(len(timeline) == 16 and all(row[-1] == "exit=0" for row in timeline if row[0] == "END"), "incomplete campaign")
    runs, reservations, differences = {}, {}, {}
    for gpu, layouts in ORDERS.items():
        events = [row[:2] for row in timeline if row[1].endswith(f"-gpu{gpu}")]
        require(events == [[action, f"private-{layout}-gpu{gpu}"]
                           for layout in layouts for action in ("START", "END")], "GPU order changed")
        owner = checked_reservation(stage / f"gpu{gpu}-before.json", gpu)
        require(checked_reservation(stage / f"gpu{gpu}-after.json", gpu) == owner, "GPU holder changed")
        reservations.update(owner)
        raw, arms = {}, {}
        for layout in layouts:
            steady, decode, prompt = (int(part[1:]) for part in layout.split("-"))
            tag = f"private-{layout}-gpu{gpu}"
            folder = stage / tag
            receipt = read(folder / "receipt.json")
            require(receipt["http_workers"] == 64 and receipt["steady_budget"] == steady
                    and receipt["burst_decode_cap"] == decode and receipt["burst_prompt_floor"] == prompt
                    and receipt["stage_trace"] is False, f"run configuration differs: {tag}")
            canonical_control = layout in (BASE, LOW_STEADY)
            sample, arm = check_case(stage, "private", gpu, owner, stock, parent, build_checks, canonical, workloads,
                                     label=tag, budget=steady, decode_cap=decode, prompt_floor=prompt,
                                     require_stock=canonical_control)
            log = (folder / "server.log").read_text()
            require(sum(line == "RILEY_NATIVE_ATTENTION mode=fixed-tile-unsplit-v1" for line in log.splitlines()) == 1
                    and not any(line.startswith(("W3-ACTIVATION ", "W3-PREFILL-PHASE ", "W3-DROPPED "))
                                for line in log.splitlines()), "attention or trace path changed")
            timings = bind_observations(log, sample)["short"]
            queues = join_http_queue(log, sample, 64)["short"]
            fidelity = check_replay(sample["short"], schedules[gpu])
            first = fidelity["first_32_indices"]
            require(sorted(first) == list(range(32)), "first 32 requests changed")
            shapes = arm["short_gpu_by_shape"]
            require(shapes and max(map(int, shapes)) <= (steady if canonical_control else 1024),
                    "C32 shape out of configured range")
            arm.update({
                "fidelity": fidelity,
                "steady_budget": steady, "burst_decode_cap": decode, "burst_prompt_floor": prompt,
                "short_gpu_steps_over_steady": sum(s["steps"] for rows, s in shapes.items() if int(rows) > steady),
                "short_gpu_steps": sum(s["steps"] for s in shapes.values()),
                "short_queue_ms": distribution(queues, first),
                "short_post_backend_ms": distribution([t["admitted_to_first_visible_ms"] for t in timings], first),
                "short_ttft_ms": distribution([row["ttft"] * 1000 for row in sample["short"]["rows"]], first),
                "rag_gpu_by_shape": batch_shapes(sample["prime"], sample["rag"]),
                "stock_output_mismatches": output_diff(canonical, sample),
            })
            raw[layout], arms[layout] = sample, arm
            runs[tag] = arm
        differences[gpu] = {}
        for before, others in ((BASE, (LOW_STEADY, *CANDIDATES)), (LOW_STEADY, CANDIDATES),
                               (CANDIDATES[0], CANDIDATES[1:])):
            differences[gpu][before] = {}
            for after in others:
                changes = compare(raw[before], raw[after], arms[before], arms[after])
                errors = p([abs(a["relative_write_error_ms"] - b["relative_write_error_ms"])
                            for a, b in zip(raw[before]["short"]["rows"], raw[after]["short"]["rows"], strict=True)], .95)
                require(errors <= 5, f"paired C32 write fidelity exceeded 5 ms: {before} vs {after}")
                differences[gpu][before][after] = {**changes, "paired_write_error_p95_ms": errors,
                                                    "recorded_regression_checks": regression_checks(changes)}
    return {"status": "passed", "remote_evidence": str(stage), "remote_build": str(build), "build": build_checks,
            "result_sha256": digest(stage / "result.sha256"), "schedules_sha256": digest(stage / "schedules.json"),
            "stock_workload_sha256": STOCK_HASH, "source_revision": stock["source_revision"], "image_id": stock["image_id"],
            "reservations": reservations, "order": timeline, "runs": runs, "comparisons": differences}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
