"""Bind the native attention comparison to output parity, live slots, and GPU leases."""
import json
from pathlib import Path
import re
import sys

from budget_reference import batch_shapes, regression_checks
from comparison_base import OUTPUT, compare, p
from http_reference import join_http_queue
from observation_reference import bind_observations
from private_reference import check_case
from replay_verifier import check_replay
from verify_a100_candidate import digest, read, require
from workload import REQUEST_COUNTS, make_workloads


BASE = "s256-fixed-a32"
MATCHED = "s128-fixed-a32"
PAGE32 = "s128-page-a32"
PAGE48 = "s128-page-a48"
ORDER = (BASE, MATCHED, PAGE32, PAGE48)
ATTENTION_MODES = {"fixed": "fixed-tile-unsplit-v1", "page": "page-v1"}
STOCK_HASH = "a77aa2092bd229a40038622d1a2de6768d2a8a6fdca159fbff42c20d9d6cfb7b"


def peak_in_flight(rows):
    events = []
    for row in rows:
        opened, closed = row["sent_at"], row["start"] + row["e2e"]
        require(closed > opened, "empty response interval")
        events.extend(((opened, 1), (closed, -1)))
    count = peak = 0
    for _, delta in sorted(events):
        count += delta
        peak = max(peak, count)
    return peak


def read_peak(log, limit):
    line, = re.findall(r"^W3-LIVE-PEAK (\d+) (\d+) (\d+)$", log, re.MULTILINE)
    active, decode, steps = map(int, line)
    require(0 < decode <= active <= limit and (active > 32) == (steps > 0), "invalid live scheduler peak")
    return {"max_active": active, "max_decode_rows": decode, "steps_above_32_active": steps}


def output_diff(reference, measured):
    return {cell: [
        {"index": i, "fields": [name for name in OUTPUT if before[name] != after[name]]}
        for i, (before, after) in enumerate(zip(reference[cell]["rows"], measured[cell]["rows"], strict=True))
        if any(before[name] != after[name] for name in OUTPUT)]
        for cell in REQUEST_COUNTS}


def lease_facts(path, gpu):
    state = read(path)
    require(state["phase"] == "idle" and state["physical_index"] == int(gpu)
            and state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 2**30,
            "standing GPU holder was not restored")
    return {gpu: {"pid": state["pid"], "start_ticks": state["start_ticks"], "gib": 20}}, len(state["other_processes"])


def read_failure(folder, gpu, owner, attention_mode):
    before, prior_peers = lease_facts(folder / "reservation-before.json", gpu)
    after, following_peers = lease_facts(folder / "reservation-after.json", gpu)
    require(before == after == owner and not (folder / "receipt.json").exists(), "failed case changed its proof or lease")
    driver = (folder / "driver.log").read_text()
    server_path = folder / "server.log"
    failure = {"exit": 1, "driver_log_sha256": digest(folder / "driver.log"),
               "other_processes_before": prior_peers, "other_processes_after": following_peers}
    if not server_path.exists() and (prior_peers or following_peers) and "unadmitted process" in driver:
        return {**failure, "kind": "gpu_busy_at_lease", "stage": "before_launch"}
    server = server_path.read_text()
    failure["server_log_sha256"] = digest(server_path)
    bad_plateau = [line for line in server.splitlines()
                   if "prefill_plateau.boundaries: must stay below iteration_token_budget" in line]
    if bad_plateau:
        require(len(bad_plateau) == 1 and "container exited before model readiness" in driver
                and not (prior_peers or following_peers), "unexpected CLI preflight failure")
        return {**failure, "kind": "invalid_plateau_boundary", "stage": "cli", "server_error": bad_plateau[0]}
    require(sum(line == f"RILEY_NATIVE_ATTENTION mode={attention_mode}" for line in server.splitlines()) == 1,
            "failed case did not log its actual attention mode")
    if following_peers and "runtime or checkpoint changed during measurement" in driver:
        require((folder / "workload.json").is_file(), "runtime identity guard fired before the workload")
        return {**failure, "kind": "gpu_ownership_drift", "stage": "after_requests"}
    matching = [line for line in server.splitlines() if
                "op=ragged_paged_attention: CUDA OutOfMemory error during allocate CUDA device buffer" in line]
    require(attention_mode == "page-v1" and not prior_peers and not following_peers and len(matching) == 1
            and "container exited before model readiness" in driver, f"unexpected case failure: {folder.name}")
    return {**failure, "kind": "workspace_oom", "stage": "before_ready", "server_error": matching[0]}


def verify(stage):
    initial = stage.parent / "w3-c32-page-slots-gpu-20261123.wUKuuOzr"
    context_trial = stage.parent / "w3-c32-bounded-page-gpu-20261123.sYTicdlw"
    row_trial = stage.parent / "w3-c32-row-bounded-page-gpu-20261123.u1rcSV0z"
    if stage == initial:
        return verify_campaign(stage, 8192, 1024)
    initial_input = ("initial_campaign", initial, 8192, 1024)
    context_input = ("bounded_context_campaign", context_trial, 2304, 1024)
    if stage == context_trial:
        result = verify_campaign(stage, 2304, 1024)
        predecessors = (initial_input,)
    elif stage == row_trial:
        result = verify_campaign(stage, 8192, 192)
        predecessors = (initial_input, context_input)
    else:
        require(stage.name.startswith("w3-c32-row-page-gpu-"), "unknown execution directory")
        result = verify_campaign(stage, 8192, 192, plateau_override=True)
        predecessors = (initial_input, context_input, ("rejected_row_campaign", row_trial, 8192, 192))
    for name, preceding_stage, context, batch_rows in predecessors:
        preceding = verify_campaign(preceding_stage, context, batch_rows)
        require(preceding["build"] == result["build"] and preceding["reservations"] == result["reservations"],
                "CUDA runtime or GPU leases changed between campaigns")
        result[name] = preceding
    return result


def verify_campaign(stage, context, max_batch_rows, plateau_override=False):
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
    schedules, workloads, canonical = read(stage / "schedules.json"), make_workloads(), read(original)
    log = (stage / "campaign.log").read_text()
    timeline = [row.split() for row in log.splitlines() if row.startswith(("START ", "END "))]
    order = ORDER if max_batch_rows == 1024 else ORDER[1:]
    orders = {"6": order[::-1], "7": order}
    require(len(timeline) == 4 * len(order) and all(row[-1] in ("exit=0", "exit=1") for row in timeline if row[0] == "END"),
            "incomplete campaign")
    runs, failures, reservations, snapshot_peers, differences = {}, {}, {}, {}, {}
    for gpu, layouts in orders.items():
        events = [row[:2] for row in timeline if row[1].endswith(f"-gpu{gpu}")]
        require(events == [[action, f"private-{layout}-gpu{gpu}"]
                           for layout in layouts for action in ("START", "END")], "GPU order changed")
        owner, before_peers = lease_facts(stage / f"gpu{gpu}-before.json", gpu)
        final_owner, after_peers = lease_facts(stage / f"gpu{gpu}-after.json", gpu)
        require(final_owner == owner, "GPU holder changed")
        snapshot_peers[gpu] = {"before": before_peers, "after": after_peers}
        reservations.update(owner)
        raw, arms = {}, {}
        for layout in layouts:
            row_budget, attention, active = layout.split("-")
            budget, slots, mode = int(row_budget[1:]), int(active[1:]), ATTENTION_MODES[attention]
            tag = f"private-{layout}-gpu{gpu}"
            folder = stage / tag
            exit_code = (folder / "exit").read_text().strip()
            require(next(row[-1] for row in timeline if row[:2] == ["END", tag]) == f"exit={exit_code}",
                    "case exit differs from the campaign log")
            if exit_code == "1":
                failures[tag] = {**read_failure(folder, gpu, owner, mode), "steady_rows": budget,
                                 "active_slots": slots, "attention_mode": mode}
                continue
            receipt = read(folder / "receipt.json")
            require(receipt["http_workers"] == 64 and receipt["steady_budget"] == budget
                    and receipt["active_slots"] == slots and receipt["attention_mode"] == mode
                    and receipt["stage_trace"] is False, f"run configuration differs: {tag}")
            sample, arm = check_case(stage, "private", gpu, owner, stock, parent, build_checks, canonical, workloads,
                                     label=tag, budget=budget, attention_mode=mode, active_slots=slots,
                                     context=context, max_batch_rows=max_batch_rows, plateau_override=plateau_override,
                                     require_stock=attention == "fixed")
            server = (folder / "server.log").read_text()
            require(sum(line == f"RILEY_NATIVE_ATTENTION mode={mode}" for line in server.splitlines()) == 1,
                    "actual attention mode changed")
            require(not any(line.startswith(("W3-ACTIVATION ", "W3-PREFILL-PHASE ", "W3-DROPPED "))
                            for line in server.splitlines()), "request timing trace was enabled")
            peak = read_peak(server, slots)
            primes, rag = (peak_in_flight(sample[phase]["rows"]) for phase in ("prime", "rag"))
            require(primes == 1 and rag <= 8, "prime or RAG exceeded its declared overlap")
            for phase in ("prime", "rag", "short"):
                require(all(sample[phase]["metrics"][field] == 0
                            for field in ("active_requests", "waiting_requests", "kv_allocated_blocks")),
                        f"unfinished requests at phase boundary: {tag} {phase}")
            observations = bind_observations(server, sample)["short"]
            queues = join_http_queue(server, sample, 64)["short"]
            fidelity = check_replay(sample["short"], schedules[gpu])
            first = fidelity["first_32_indices"]
            require(sorted(first) == list(range(32)), "first 32 requests changed")
            shapes = arm["short_gpu_by_shape"]
            require(shapes and max(map(int, shapes)) <= budget, "short shape exceeded steady budget")
            maximum_requested_context = max(row["prompt_tokens"] + body["max_tokens"]
                                            for phase, bodies in workloads.items()
                                            for row, body in zip(sample[phase]["rows"], bodies, strict=True))
            require(maximum_requested_context <= context, "requested tokens exceed measured context bound")
            arm.update({
                "fidelity": fidelity,
                "execution": {"steady_rows": budget, "active_slots": slots, "attention_mode": mode,
                              "max_batch_rows": max_batch_rows, "max_sequence_tokens": context},
                "live_scheduler_peak": peak,
                "device_live_bytes_at_ready": sample["startup"]["metrics"]["allocation"]["device_live_bytes"],
                "maximum_requested_context": maximum_requested_context,
                "prime_peak_client_in_flight": primes,
                "rag_peak_client_in_flight": rag,
                "short_http_queue_p95_ms": p(queues[32:], .95),
                "short_after_registration_p95_ms": p([x["admitted_to_first_visible_ms"] for x in observations[32:]], .95),
                "rag_gpu_by_shape": batch_shapes(sample["prime"], sample["rag"]),
                "stock_output_mismatches": output_diff(canonical, sample),
            })
            raw[layout], arms[layout] = sample, arm
            runs[tag] = arm
        differences[gpu] = {}
        for before, others in ((BASE, (MATCHED, PAGE32, PAGE48)), (MATCHED, (PAGE32, PAGE48)), (PAGE32, (PAGE48,))):
            if before not in raw:
                continue
            differences[gpu][before] = {}
            for after in others:
                if after not in raw:
                    continue
                changes = compare(raw[before], raw[after], arms[before], arms[after])
                errors = p([abs(a["relative_write_error_ms"] - b["relative_write_error_ms"])
                            for a, b in zip(raw[before]["short"]["rows"], raw[after]["short"]["rows"], strict=True)], .95)
                require(errors <= 5, f"paired write fidelity exceeded 5 ms: {before} vs {after}")
                differences[gpu][before][after] = {**changes, "paired_write_error_p95_ms": errors,
                                                    "recorded_regression_checks": regression_checks(changes)}
    kinds = {failure["kind"] for failure in failures.values()}
    status = "passed"
    if kinds:
        status = "completed_with_oom" if kinds == {"workspace_oom"} else "completed_with_exclusions"
    return {"status": status, "max_sequence_tokens": context,
            "max_batch_rows": max_batch_rows, "prefill_plateau_rows": [128] if plateau_override else [128, 192],
            "remote_evidence": str(stage), "remote_build": str(build), "build": build_checks,
            "result_sha256": digest(stage / "result.sha256"), "schedules_sha256": digest(stage / "schedules.json"),
            "stock_workload_sha256": STOCK_HASH, "source_revision": stock["source_revision"], "image_id": stock["image_id"],
            "reservations": reservations, "reservation_peer_counts": snapshot_peers,
            "order": timeline, "runs": runs, "failures": failures, "comparisons": differences}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
