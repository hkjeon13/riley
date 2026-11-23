"""Verify fixed-tile outputs, resource isolation, and paired latency at 32 and 48 slots."""
import json
from pathlib import Path
import sys

from budget_reference import batch_shapes, regression_checks
from comparison_base import compare, p
from http_reference import join_http_queue
from observation_reference import bind_observations
from page_reference import STOCK_HASH, lease_facts, output_diff, peak_in_flight, read_peak
from private_reference import check_case
from replay_verifier import check_replay
from verify_a100_candidate import digest, read, require
from workload import REQUEST_COUNTS, make_workloads


MODE = "fixed-tile-unsplit-v1"
SOURCE = "3951d90c8df381daef2ab88fd18ac9e8a83c4554"
ORDER = ("s256-fixed-a32", "s256-fixed-a48", "s128-fixed-a48", "s128-fixed-a32")
CONTEXT = 8192
ROW_LIMIT = 1024
ZERO = ("active_requests", "waiting_requests", "kv_allocated_blocks")
PAIRS = (
    ("s256-fixed-a32", "s256-fixed-a48"),
    ("s128-fixed-a32", "s128-fixed-a48"),
    ("s256-fixed-a32", "s128-fixed-a32"),
    ("s256-fixed-a32", "s128-fixed-a48"),
)


def verify_cli(stage, build_checks):
    observed = read(stage / "cli.json")
    exits = {"baseline-32": 1, "baseline-48": 2, "diagnostic-32": 1,
             "diagnostic-48": 1, "diagnostic-49": 2, "diagnostic-8193": 2}
    require(set(observed) == set(exits), "CLI receipts missing")
    for key, code in exits.items():
        case = observed[key]
        expected_hash = (build_checks["artifacts"]["target/release/riley"] if key.startswith("diagnostic")
                         else "cf223fe96c74ac822e4d20eccbf811c871286dbe0269521a585da184a4834402")
        bound = 8193 if key == "diagnostic-8193" else CONTEXT
        slots = 48 if key == "diagnostic-8193" else int(key.rsplit("-", 1)[1])
        detail = ("cannot canonicalize checkpoint root" if code == 1
                  else "at most 32 active sequences" if key == "baseline-48" else "at most 48 active sequences")
        require(case["exit"] == code and case["binary_sha256"] == expected_hash
                and case["slots"] == slots and case["max_sequence_tokens"] == bound
                and detail in case["diagnostic"], f"CLI receipt changed: {key}")
    return observed


def verify_failure(folder, gpu, owner):
    before, before_peers = lease_facts(folder / "reservation-before.json", gpu)
    after, after_peers = lease_facts(folder / "reservation-after.json", gpu)
    require(before == after == owner, "a failed run changed the holder")
    driver = (folder / "driver.log").read_text()
    server = folder / "server.log"
    failure = {"driver_sha256": digest(folder / "driver.log"),
               "other_processes_before": before_peers, "other_processes_after": after_peers}
    if not server.exists() and (before_peers or after_peers) and "unadmitted process" in driver:
        return {**failure, "kind": "gpu_busy_at_lease", "stage": "before_launch"}
    if server.is_file() and after_peers and "runtime or checkpoint changed during measurement" in driver:
        return {**failure, "kind": "gpu_ownership_drift", "stage": "after_requests", "server_sha256": digest(server)}
    raise ValueError(f"unclassified failure: {folder.name}")


def verify_arm(stage, tag, gpu, owner, budget, slots, stock, parent, build_checks, canonical, workloads, schedule):
    folder = stage / tag
    receipt = read(folder / "receipt.json")
    require(receipt["http_workers"] == 64 and receipt["steady_budget"] == budget
            and receipt["active_slots"] == slots and receipt["attention_mode"] == MODE
            and receipt["stage_trace"] is False, f"configuration changed: {tag}")
    sample, run = check_case(stage, "private", gpu, owner, stock, parent, build_checks, canonical, workloads,
                             label=tag, budget=budget, attention_mode=MODE, active_slots=slots,
                             context=CONTEXT, max_batch_rows=ROW_LIMIT, plateau_override=False,
                             require_stock=slots == 32)
    log = (folder / "server.log").read_text()
    require(log.splitlines().count(f"RILEY_NATIVE_ATTENTION mode={MODE}") == 1
            and not any(line.startswith(("W3-ACTIVATION ", "W3-PREFILL-PHASE ", "W3-DROPPED "))
                        for line in log.splitlines()), f"attention or trace mode changed: {tag}")
    peak = read_peak(log, slots)
    priming, rag = (peak_in_flight(sample[phase]["rows"]) for phase in ("prime", "rag"))
    require(priming == 1 and rag <= 8, "priming or RAG overlap changed")
    require(all(sample[phase]["metrics"][name] == 0 for phase in REQUEST_COUNTS for name in ZERO),
            f"unfinished requests at phase boundary: {tag}")
    observations = bind_observations(log, sample)["short"]
    queues = join_http_queue(log, sample, 64)["short"]
    fidelity = check_replay(sample["short"], schedule)
    require(sorted(fidelity["first_32_indices"]) == list(range(32)), "initial 32 requests changed")
    shapes = run["short_gpu_by_shape"]
    require(shapes and max(map(int, shapes)) <= budget, "short shape exceeded steady budget")
    max_context = max(row["prompt_tokens"] + body["max_tokens"] for phase, bodies in workloads.items()
                      for row, body in zip(sample[phase]["rows"], bodies, strict=True))
    require(max_context <= CONTEXT, "request exceeded configured context")
    run.update({
        "fidelity": fidelity,
        "execution": {"steady_rows": budget, "active_slots": slots, "attention_mode": MODE,
                      "max_batch_rows": ROW_LIMIT, "max_sequence_tokens": CONTEXT},
        "live_scheduler_peak": peak,
        "device_live_bytes_at_ready": sample["startup"]["metrics"]["allocation"]["device_live_bytes"],
        "maximum_requested_context": max_context,
        "prime_peak_client_in_flight": priming,
        "rag_peak_client_in_flight": rag,
        "short_http_queue_p95_ms": p(queues[32:], .95),
        "short_after_registration_p95_ms": p([row["admitted_to_first_visible_ms"] for row in observations[32:]], .95),
        "rag_gpu_by_shape": batch_shapes(sample["prime"], sample["rag"]),
        "stock_output_mismatches": output_diff(canonical, sample),
    })
    return sample, run


def compare_arms(raw, runs):
    reports = {}
    for before, after in PAIRS:
        if before not in raw or after not in raw:
            continue
        comparison = compare(raw[before], raw[after], runs[before], runs[after])
        jitter = p([abs(a["relative_write_error_ms"] - b["relative_write_error_ms"])
                    for a, b in zip(raw[before]["short"]["rows"], raw[after]["short"]["rows"], strict=True)], .95)
        require(jitter <= 5, f"matched dispatch drifted: {before} to {after}")
        peak = runs[after]["live_scheduler_peak"]
        reports[f"{before} -> {after}"] = {**comparison, "paired_write_error_p95_ms": jitter,
                                          "target_exercised_above_32": peak["max_active"] > 32,
                                          "recorded_regression_checks": regression_checks(comparison)}
    return reports


def verify(stage):
    evidence = stage.parent
    build = Path((stage / "build-path.txt").read_text().strip())
    parent = evidence / "w5-prefix-metrics-package-20261121.nEFyze5I/cpu-evidence/report.json"
    stock = read(parent)
    stock_path = evidence / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first/workload.json"
    require(digest(stock_path) == STOCK_HASH and stock["status"] == "passed"
            and stock["source_revision"] == SOURCE, "archived image or output reference changed")
    canonical = read(stock_path)
    workloads, schedules = make_workloads(), read(stage / "schedules.json")
    build_checks = {"input_sha256": digest(build / "build-inputs.sha256"),
                    "result_sha256": digest(build / "build-result.sha256"),
                    "artifacts": {name: check for check, name in
                                  (line.split("  ", 1) for line in (build / "build-result.sha256").read_text().splitlines())}}
    cli = verify_cli(stage, build_checks)
    timeline = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                if row.startswith(("START ", "END "))]
    require(len(timeline) == 16, "incomplete measurement")
    runs, failures, comparisons, reservations, peers = {}, {}, {}, {}, {}
    for gpu, order in (("6", ORDER[::-1]), ("7", ORDER)):
        events = [row for row in timeline if row[1].endswith(f"-gpu{gpu}")]
        require([row[:2] for row in events] == [[action, f"private-{case}-gpu{gpu}"]
                for case in order for action in ("START", "END")], f"wrong run order: GPU {gpu}")
        owner, before_peers = lease_facts(stage / f"gpu{gpu}-before.json", gpu)
        after_owner, after_peers = lease_facts(stage / f"gpu{gpu}-after.json", gpu)
        require(after_owner == owner, "the GPU holder changed")
        reservations.update(owner)
        peers[gpu] = {"before": before_peers, "after": after_peers}
        raw, arms = {}, {}
        for layout, end in zip(order, events[1::2], strict=True):
            label = f"private-{layout}-gpu{gpu}"
            folder = stage / label
            code = (folder / "exit").read_text().strip()
            require(code in ("0", "1") and end[-1] == f"exit={code}", f"exit proof changed: {label}")
            if code != "0":
                failures[label] = verify_failure(folder, gpu, owner)
                continue
            rows, _, active = layout.split("-")
            sample, arm = verify_arm(stage, label, gpu, owner, int(rows[1:]), int(active[1:]), stock, parent,
                                     build_checks, canonical, workloads, schedules[gpu])
            raw[layout], arms[layout], runs[label] = sample, arm, arm
        comparisons[gpu] = compare_arms(raw, arms)
    require(runs and bool(failures) == ((stage / "campaign.exit").read_text().strip() != "0"), "campaign exit mismatch")
    return {"status": "completed_with_exclusions" if failures else "passed",
            "remote_evidence": str(stage), "remote_build": str(build), "build": build_checks, "cli": cli,
            "source_revision": SOURCE, "image_id": stock["image_id"], "stock_workload_sha256": STOCK_HASH,
            "schedules_sha256": digest(stage / "schedules.json"), "result_sha256": digest(stage / "result.sha256"),
            "order": timeline, "reservations": reservations, "reservation_peer_counts": peers,
            "runs": runs, "exclusions": failures, "comparisons": comparisons}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
