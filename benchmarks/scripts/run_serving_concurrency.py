#!/usr/bin/env python3
"""Fresh paired HTTP closed-loop concurrency measurements; preparation is default.

Invocation: --plan PLAN --request REQUEST --binding UNCHANGED_C1_BINDING
            --output NEW_DIRECTORY [--measure]

PLAN schema: riley.serving-concurrency-plan.v1. Required fields:
  parent_c1_plan: {path, sha256}; immutable_files: {absolute_path: sha256}
  workload: {id, offered_concurrency, arrival_policy: "closed-loop-refill",
             retained_requests_per_process, warmups_per_worker_per_transport,
             pairs, purpose: "screening" | "tail-study"}
  base_environment: explicit subprocess environment (including PATH and HOME)
  http_lanes: parent HTTP lanes with only ports/capacity/budget changes:
    riley: active_capacity=1, waiting_capacity=64, http_workers=8,
           token_budget=128, argv/env/port/expected_output_text
    vllm: active_capacity>=C, token_budget>=128,
          argv/env/port/expected_output_text
  vllm_runtime: {engine_version, enforce_eager:false,
                 enable_chunked_prefill:true, compilation_mode:"VLLM_COMPILE",
                 cudagraph_mode:"FULL_AND_PIECEWISE"}
  startup_timeout_seconds: positive number
  request_timeout_seconds: total per-request deadline, positive and <=120

The parent remains c1 correctness evidence, never a concurrent result. New
immutable_files must pin this runner, its shared runner and parent plan; the
unchanged parent's recursive artifact pins are rechecked too. Lane environment
overrides must equal the parent; only explicit base_environment is inherited by
servers. No GPU query, server launch or preflight occurs during preparation.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import run_serving_optimization as shared

PLAN_SCHEMA = "riley.serving-concurrency-plan.v1"
RESULT_SCHEMA = "riley.serving-concurrency-result.v1"
CONDITION = {**shared.CONDITION, "scope": "closed-loop offered HTTP concurrency; Riley active capacity one"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)


def evidence(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": shared.digest(path)}


def positive(value, label, maximum):
    require(type(value) is int and 0 < value <= maximum, label + " is outside its integer range")
    return value


def option(argv, flag):
    require(argv.count(flag) == 1, "exactly one option required: " + flag)
    index = argv.index(flag)
    require(index + 1 < len(argv), "option value missing: " + flag)
    return argv[index + 1]


def without_options(argv, flags):
    result, index = [], 0
    while index < len(argv):
        if argv[index] in flags:
            option(argv, argv[index])
            index += 2
        else:
            result.append(argv[index])
            index += 1
    return result


def validate_manifest(plan, request_path, binding_path, request, binding):
    require(plan["schema_version"] == PLAN_SCHEMA, "not a concurrency manifest; c1 plans cannot be relabelled")
    pinned = {}
    for name, digest in plan["immutable_files"].items():
        path = Path(name)
        require(path.is_absolute(), "immutable path must be absolute")
        resolved = str(path.resolve(strict=True))
        require(resolved not in pinned, "duplicate resolved artifact path")
        require(shared.digest(path) == digest, "immutable artifact changed: " + name)
        pinned[resolved] = digest
    parent_ref = plan["parent_c1_plan"]
    require(set(parent_ref) == {"path", "sha256"} and evidence(parent_ref["path"]) == parent_ref,
            "parent plan evidence differs")
    for path in (Path(__file__), Path(shared.__file__), Path(parent_ref["path"])):
        require(pinned.get(str(path.resolve())) == shared.digest(path), "runner or parent plan not pinned: " + str(path))
    parent = read(parent_ref["path"])
    require(parent["measurement_mode"] == "http", "reference parent must be an unchanged HTTP c1 plan")
    require(Path(parent["request_path"]).resolve() == Path(request_path).resolve()
            and Path(parent["binding_path"]).resolve() == Path(binding_path).resolve(),
            "explicit reference inputs differ from parent c1 plan")
    # Deliberately validate the ORIGINAL parent and ORIGINAL c1 binding. Never
    # pass the new workload or modified lane configuration to the c1 validator.
    shared.validate_artifacts(parent, request_path, binding_path, request, binding)
    qualification_path = Path(parent["qualification_path"])
    qualification = read(qualification_path)
    require(parent["immutable_files"].get(str(qualification_path)) == shared.digest(qualification_path)
            == binding["source"]["correctness_report_sha256"], "parent qualification hash differs")
    require(qualification["passed"] is True and qualification["source_commit"] == parent["source_commit"]
            and qualification["source_clean"] is True, "parent source was not qualified")
    require(binding["workload"]["concurrency"] == 1 and binding["workload"]["prompt_tokens"] == 128
            and binding["workload"]["output_tokens"] == 32
            and binding["source"]["correctness_gate_id"] == "g04-vllm-smol-p128-v1",
            "parent is not the fixed c1/P128/O32 numerical reference")
    workload = plan["workload"]
    require(set(workload) == {"id", "offered_concurrency", "arrival_policy", "retained_requests_per_process",
                             "warmups_per_worker_per_transport", "pairs", "purpose"}, "workload fields differ")
    concurrency = positive(workload["offered_concurrency"], "offered concurrency", 8)
    count = positive(workload["retained_requests_per_process"], "retained request count", 100000)
    require(count >= concurrency, "each initial client requires a retained request")
    positive(workload["warmups_per_worker_per_transport"], "warmups per worker", 1000)
    pairs = positive(workload["pairs"], "process pairs", 5)
    require(workload["arrival_policy"] == "closed-loop-refill", "unsupported arrival policy")
    require(isinstance(workload["id"], str) and re.fullmatch(r"[A-Za-z0-9_.-]+", workload["id"])
            and workload["id"] != binding["workload"]["workload_id"], "new workload identity required")
    require(workload["purpose"] in ("screening", "tail-study"), "measurement purpose missing")
    if workload["purpose"] == "tail-study":
        require(pairs == 5 and count >= 10000, "tail-study requires five pairs and at least 10000 requests/process")
    base = plan["base_environment"]
    require(isinstance(base, dict) and base.get("PATH") and base.get("HOME"), "explicit PATH/HOME environment required")
    timeout = plan["startup_timeout_seconds"]
    require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 1800, "invalid startup timeout")
    request_timeout = plan["request_timeout_seconds"]
    require(type(request_timeout) in (int, float) and math.isfinite(request_timeout) and 0 < request_timeout <= 120,
            "invalid total request deadline")
    runtime = plan["vllm_runtime"]
    require(set(runtime) == {"engine_version", "enforce_eager", "enable_chunked_prefill", "compilation_mode", "cudagraph_mode"},
            "vLLM runtime contract fields differ")
    require(runtime["engine_version"] == "0.27.1" and runtime["enforce_eager"] is False
            and runtime["enable_chunked_prefill"] is True and runtime["compilation_mode"] == "VLLM_COMPILE"
            and runtime["cudagraph_mode"] == "FULL_AND_PIECEWISE", "unsupported vLLM runtime contract")
    require(set(plan["http_lanes"]) == {"riley", "vllm"}, "both HTTP lanes required")
    for role, lane in plan["http_lanes"].items():
        original = parent["http_lanes"][role]
        require(lane["env"] == original["env"], "lane environment differs from pinned parent")
        environment = {**base, **lane["env"]}
        require(all(isinstance(k, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
                    and isinstance(v, str) and "\0" not in v for k, v in environment.items()), "invalid explicit environment")
        require(not any(k == "LD_PRELOAD" or k.startswith("RILEY_") for k in environment), "instrumentation/preload environment is forbidden")
        require(lane["expected_output_text"] == original["expected_output_text"]
                and lane["expected_output_text"], "reference text changed")
        require(lane.get("model_id", parent.get("http_model_id", "g04-smol"))
                == original.get("model_id", parent.get("http_model_id", "g04-smol")), "served model identity changed")
        positive(lane["port"], "port", 65535)
        argv = lane["argv"]
        require(isinstance(argv, list) and all(isinstance(v, str) and v for v in argv), "invalid server argv")
        require(lane["argv"][0] == original["argv"][0], "server binary differs from qualified parent")
        if role == "riley":
            require(all(type(lane[key]) is int for key in ("active_capacity", "waiting_capacity", "http_workers", "token_budget"))
                    and lane["active_capacity"] == 1 and lane["waiting_capacity"] == 64
                    and lane["http_workers"] == 8 and lane["token_budget"] == 128, "Riley queued-c1 capacities differ")
            allowed = {"--bind", "--max-waiting-requests"}
            require(option(argv, "--bind") == "127.0.0.1:" + str(lane["port"])
                    and option(argv, "--max-waiting-requests") == "64"
                    and option(argv, "--max-active-sequences") == "1"
                    and option(argv, "--batch-token-budget") == option(argv, "--prefill-chunk-tokens") == "128",
                    "Riley command capacities differ")
            service = Path(parent["source_root"]) / "crates/riley-server/src/service.rs"
            require(parent["immutable_files"].get(str(service)) == shared.digest(service)
                    and re.search(r"impl Default for ServerConfig\s*\{.*?worker_threads:\s*8,", service.read_text(), re.S),
                    "qualified eight-worker HTTP source is not pinned")
        else:
            capacity = positive(lane["active_capacity"], "vLLM active capacity", 1024)
            budget = positive(lane["token_budget"], "vLLM token budget", 1048576)
            require(capacity >= concurrency and budget >= max(128, capacity), "vLLM capacity/budget cannot serve offered concurrency")
            allowed = {"--port", "--max-num-seqs", "--max-num-batched-tokens"}
            require(option(argv, "--port") == str(lane["port"])
                    and option(argv, "--max-num-seqs") == str(capacity)
                    and option(argv, "--max-num-batched-tokens") == str(budget), "vLLM command capacities differ")
        require(without_options(argv, allowed) == without_options(original["argv"], allowed),
                "unqualified HTTP command change: " + role)
    require(plan["http_lanes"]["riley"]["port"] != plan["http_lanes"]["vllm"]["port"], "lane ports must differ")
    return parent


def validate_vllm_startup(path, lane, expected):
    text = Path(path).read_text(errors="replace")
    configurations = []
    for line in text.splitlines():
        if "non-default args:" in line:
            value = ast.literal_eval(line.split("non-default args:", 1)[1].strip())
            require(isinstance(value, dict), "invalid vLLM startup arguments")
            configurations.append(value)
    require(len(configurations) == 1, "exactly one actual vLLM startup configuration required")
    args = configurations[0]
    require(args.get("max_num_seqs") == lane["active_capacity"]
            and args.get("max_num_batched_tokens") == lane["token_budget"]
            and args.get("max_model_len") == 160 and args.get("dtype") == "bfloat16"
            and args.get("gpu_memory_utilization") == 0.3 and args.get("enable_prefix_caching") is False,
            "actual vLLM scheduler/model settings differ")
    lines = [line for line in text.splitlines() if "Initializing a V1 LLM engine" in line]
    require(len(lines) == 1, "one vLLM engine startup record required")
    line = lines[0]
    for fragment in ("(v" + expected["engine_version"] + ")", "enforce_eager=False",
                     "enable_chunked_prefill=True", "enable_prefix_caching=False",
                     "CompilationMode." + expected["compilation_mode"], "CUDAGraphMode." + expected["cudagraph_mode"]):
        require(fragment in line, "actual vLLM runtime setting missing: " + fragment)
    return {"validated": True, "non_default_arguments": args, "runtime": expected,
            "startup_record": line, "log": evidence(path)}


def overlap_peak(rows, start_key="started_ns", end_key="finished_ns"):
    events = [(row[start_key], 1) for row in rows] + [(row[end_key], -1) for row in rows]
    active = peak = 0
    for _, delta in sorted(events):  # Ends precede starts at an equal timestamp.
        active += delta
        require(active >= 0, "request timestamp intervals are invalid")
        peak = max(peak, active)
    require(active == 0, "unbalanced request intervals")
    return peak


def run_phase(request_one, concurrency, count, phase, *, per_worker=False,
              request_timeout=120, abort_owned=None):
    """Refill C clients; a per-call watchdog aborts the owned server on timeout.

    The shared socket timeout bounds each read, not an entire SSE response.
    Production callers supply abort_owned so a trickling response cannot keep
    workers alive forever. There is no phase-duration cap on a long tail study.
    """
    lock, stopped, watchdog_stop = threading.Lock(), threading.Event(), threading.Event()
    created = time.perf_counter_ns()
    state = {"next": 0, "started_ns": created, "live": {}, "errors": [], "timeouts": []}
    barrier = threading.Barrier(concurrency + 1, action=lambda: state.update(started_ns=time.perf_counter_ns()))
    rows = []

    def abort():
        stopped.set()
        if abort_owned is not None:
            try:
                abort_owned()
            except Exception as error:
                with lock:
                    state["errors"].append({"error_type": type(error).__name__, "error": str(error), "operation": "abort_owned"})

    def watchdog():
        while not watchdog_stop.wait(min(.1, request_timeout / 4)):
            now = time.perf_counter_ns()
            with lock:
                late = [index for index, started in state["live"].values()
                        if now - started > request_timeout * 1e9]
                if late:
                    state["timeouts"].extend(late)
            if late:
                abort()
                return

    def worker(worker_id):
        completed = 0
        barrier.wait(timeout=30)
        while not stopped.is_set():
            with lock:
                if stopped.is_set() or state["next"] >= count or (per_worker and completed >= count // concurrency):
                    break
                index = state["next"]
                state["next"] += 1
                call_started = time.perf_counter_ns()
                state["live"][worker_id] = (index, call_started)
            try:
                row = request_one()
                require(row["text_exact"] is True, "request helper did not validate output")
                require(call_started <= row["started_ns"] <= row["finished_ns"], "request clock ordering differs")
                require(all(row["started_ns"] <= event <= row["finished_ns"] for event in row.get("event_ns", [])),
                        "SSE event lies outside request boundaries")
                if time.perf_counter_ns() - call_started > request_timeout * 1e9:
                    raise TimeoutError("request exceeded total wall deadline")
                row.update(status="success")
            except Exception as error:
                row = {"status": "failed", "started_ns": call_started, "finished_ns": time.perf_counter_ns(),
                       "error": str(error), "error_type": type(error).__name__}
                stopped.set()
            row.update(request_id=f"{phase}-{index}", index=index, worker_id=worker_id, phase=phase,
                       warmup=phase != "retained", call_started_ns=call_started, call_finished_ns=time.perf_counter_ns())
            require(row["call_finished_ns"] >= row["finished_ns"], "call completion precedes response completion")
            with lock:
                rows.append(row)
                state["live"].pop(worker_id, None)
            completed += 1

    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="riley-http-load")
    monitor = threading.Thread(target=watchdog, name="riley-request-watchdog", daemon=True)
    futures = []
    interrupted = None
    monitor.start()
    try:
        for worker_id in range(concurrency):
            futures.append(pool.submit(worker, worker_id))
        barrier.wait(timeout=30)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                with lock:
                    state["errors"].append({"error_type": type(error).__name__, "error": str(error), "operation": "worker"})
                abort()
    except BaseException as error:
        state["errors"].append({"error_type": type(error).__name__, "error": str(error), "operation": "phase"})
        abort()
        if not isinstance(error, Exception):
            interrupted = error
    finally:
        stopped.set()
        barrier.abort()
        pool.shutdown(wait=True, cancel_futures=True)
        watchdog_stop.set()
        monitor.join()
    finished = time.perf_counter_ns()
    rows.sort(key=lambda row: row["index"])
    successes = [row for row in rows if row["status"] == "success"]
    failures = len(rows) - len(successes)
    observed = overlap_peak(successes) if successes else 0
    result = {"phase": phase, "warmup": phase != "retained", "requested": count,
              "attempted": state["next"], "succeeded": len(successes), "failed": failures,
              "unresolved_attempts": state["next"] - len(rows), "not_started": count - state["next"],
              "offered_concurrency": concurrency, "observed_max_request_in_flight": observed,
              "observed_max_call_in_flight": overlap_peak(rows, "call_started_ns", "call_finished_ns") if rows else 0,
              "phase_started_ns": state["started_ns"], "phase_finished_ns": finished,
              "completed": len(rows) == count and failures == 0 and not state["errors"] and not state["timeouts"],
              "worker_errors": state["errors"], "watchdog_timed_out_request_indices": state["timeouts"],
              "failure_policy": "stop new requests; drain in-flight calls; total deadline aborts owned server; no retries",
              "client_socket_timeout_seconds": 120, "total_request_deadline_seconds": request_timeout}
    if rows:
        require(state["started_ns"] <= min(row["started_ns"] for row in rows)
                and max(row["finished_ns"] for row in rows) <= finished, "phase does not enclose requests")
    if interrupted is not None:
        result["interrupted"] = True
    return rows, result


def summarize(rows, phase, output_tokens):
    require(phase["completed"] and phase["observed_max_request_in_flight"] == phase["offered_concurrency"],
            "phase incomplete or offered concurrency was never observed")
    require(rows and all(row["status"] == "success" and row["phase"] == "retained" for row in rows), "non-retained result")
    start, finish = min(row["started_ns"] for row in rows), max(row["finished_ns"] for row in rows)
    require(start < finish, "nonpositive retained wall interval")
    result = {**phase, "retained_first_start_ns": start, "retained_last_finish_ns": finish,
              "retained_wall_ns": finish - start, "output_tokens": output_tokens * len(rows),
              "output_tokens_per_wall_second": output_tokens * len(rows) * 1e9 / (finish - start),
              "request_tokens_verified": "reference text and available usage; raw IDs unavailable over HTTP",
              "token_ttft": "unmeasured", "http_tpot": "unmeasured; SSE events are not token boundaries"}
    for label, values in (("e2e_ms", [(r["finished_ns"] - r["started_ns"]) / 1e6 for r in rows]),
                          ("first_text_event_ms", [(r["event_ns"][0] - r["started_ns"]) / 1e6 for r in rows])):
        require(all(math.isfinite(value) and value >= 0 for value in values), "invalid latency")
        result[label] = {"median": statistics.median(values), "p95": shared.percentile(values, .95),
                         "p99": shared.percentile(values, .99), "max": max(values)}
    return result


def save_phase(directory, rows, phase):
    with (directory / (phase["phase"] + ".jsonl")).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    shared.write_json(directory / (phase["phase"] + "-accounting.json"), phase)


def preflight(parent, binding, directory):
    common = directory / "common-preflight"
    common.mkdir()
    shared.preflight(parent, binding, common)
    checked = read(common / "live-condition.json")
    shared.write_json(directory / "live-condition.json", {
        **checked, **CONDITION, "shared_condition_evidence": evidence(common / "live-condition.json"),
        "shared_scope_note": "shared c1-labelled preflight proves GPU conditions only; workload is the new manifest"})


def run_lane(plan, parent, role, request, directory):
    lane, workload = plan["http_lanes"][role], plan["workload"]
    concurrency = workload["offered_concurrency"]
    shared.check_port(lane["port"])
    env = {**plan["base_environment"], **lane["env"]}
    log_path = directory / "server.log"
    with log_path.open("x") as log:
        process = subprocess.Popen(lane["argv"], env=env, cwd=parent["source_root"],
                                   stdout=log, stderr=log, start_new_session=True)
        cleanup_lock, cleaned = threading.Lock(), False
        def cleanup():
            nonlocal cleaned
            with cleanup_lock:
                if not cleaned:
                    shared.stop_owned_process(process)
                    cleaned = True
        try:
            shared.write_json(directory / "launch.json", {"argv": lane["argv"], "environment": env,
                              "cwd": parent["source_root"], "pid": process.pid, "fresh_process": True,
                              "active_capacity": lane["active_capacity"], "offered_concurrency": concurrency})
            shared.wait_ready(process, lane["port"], plan["startup_timeout_seconds"])
            if role == "vllm":
                # The live log continues to grow during traffic and shutdown.
                # Bind early configuration validation to an immutable snapshot.
                startup_log = directory / "vllm-startup.log"
                with startup_log.open("x") as snapshot:
                    snapshot.write(log_path.read_text(errors="replace"))
                shared.write_json(directory / "vllm-startup.json", validate_vllm_startup(startup_log, lane, plan["vllm_runtime"]))
            model = lane.get("model_id", parent.get("http_model_id", "g04-smol"))
            for phase, streaming in (("warmup-nonstream", False), ("warmup-stream", True), ("retained", True)):
                warmup = phase != "retained"
                count = concurrency * workload["warmups_per_worker_per_transport"] if warmup else workload["retained_requests_per_process"]
                rows, accounting = run_phase(
                    lambda: shared.http_request(lane["port"], model, request, lane["expected_output_text"], streaming),
                    concurrency, count, phase, per_worker=warmup,
                    request_timeout=plan["request_timeout_seconds"], abort_owned=cleanup)
                save_phase(directory, rows, accounting)
                require(accounting["completed"], phase + " failed or incomplete; see accounting and request records")
                require(accounting["observed_max_request_in_flight"] == concurrency,
                        phase + " did not exercise the offered concurrency; see accounting")
            result = summarize(rows, accounting, request["requested_output_tokens"])
            require(process.poll() is None, "server exited during measurement")
        finally:
            cleanup()
            shared.write_json(directory / "process-exit.json", {"pid": process.pid, "returncode": process.returncode,
                              "owned_session_cleanup_finished": True, "log": evidence(log_path)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("plan", "request", "binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--measure", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    args.plan, args.request, args.binding, args.output = (p.resolve() for p in (args.plan, args.request, args.binding, args.output))
    plan, request, binding = (read(p) for p in (args.plan, args.request, args.binding))
    parent = validate_manifest(plan, args.request, args.binding, request, binding)
    args.output.mkdir(exist_ok=False)
    prepared = {"schema_version": RESULT_SCHEMA, "measurement_started": args.measure,
                "workload": plan["workload"], "source_commit": parent["source_commit"],
                "condition": {**CONDITION, "host_environment_id": parent["preflight_environment_id"]},
                "inputs": {str(p): shared.digest(p) for p in (args.plan, args.request, args.binding)},
                "runner": evidence(__file__), "shared_runner": evidence(shared.__file__),
                "parent_c1_plan": plan["parent_c1_plan"], "parent_qualification_use": "c1 source/token reference only; no concurrency evidence",
                "reference_input_token_ids_sha256": shared.token_digest(binding["input_token_ids"]),
                "reference_generated_token_ids_sha256": shared.token_digest(binding["generated_token_ids"]),
                "http_generated_token_ids_verified": False, "high_concurrency_stability_claim": False,
                "tail_estimator": "nearest rank per process; screening or tail-study counts do not alone establish stability",
                "load_boundary": "closed-loop clients; no open-loop arrival-rate or overload claim",
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    shared.write_json(args.output / "preparation.json", prepared)
    if not args.measure:
        print(json.dumps({"prepared": True, "measurement_started": False, "output": str(args.output)}))
        return 0
    pairs = []
    try:
        for index in range(1, plan["workload"]["pairs"] + 1):
            pair = {"index": index, "order": ["riley", "vllm"] if index % 2 else ["vllm", "riley"]}
            for role in pair["order"]:
                directory = args.output / f"pair-{index:02}-{role}"
                directory.mkdir()
                try:
                    require(shared.digest(args.plan) == prepared["inputs"][str(args.plan)]
                            and read(args.plan) == plan, "concurrency plan changed during campaign")
                    validate_manifest(plan, args.request, args.binding, request, binding)
                    preflight(parent, binding, directory)
                    pair[role] = run_lane(plan, parent, role, request, directory)
                    require(shared.digest(args.plan) == prepared["inputs"][str(args.plan)]
                            and read(args.plan) == plan, "concurrency plan changed during lane execution")
                    validate_manifest(plan, args.request, args.binding, request, binding)
                    shared.write_json(directory / "execution-complete.json", {"completed": True, "summary": pair[role]})
                except BaseException as error:
                    shared.write_json(directory / "failure.json", {"completed": False, "error": str(error), "error_type": type(error).__name__})
                    raise
            pair["riley_over_vllm_wall_throughput"] = pair["riley"]["output_tokens_per_wall_second"] / pair["vllm"]["output_tokens_per_wall_second"]
            pairs.append(pair)
        shared.write_json(args.output / "summary.json", {**prepared, "completed": True, "pair_results": pairs,
                          "paired_throughput_ratio_median": statistics.median(p["riley_over_vllm_wall_throughput"] for p in pairs)})
        shared.write_json(args.output / "completion.json", {"completed": True, "pairs": len(pairs), "processes": 2 * len(pairs)})
    except BaseException as error:
        shared.write_json(args.output / "completion.json", {"completed": False, "completed_pairs": len(pairs), "error": str(error)})
        raise
    print(json.dumps({"completed": True, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
