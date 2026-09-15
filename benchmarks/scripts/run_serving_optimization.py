#!/usr/bin/env python3
"""Paired, immutable HTTP serving diagnostic. Preparation is the default.

Use --plan, --request, --binding and a new --output directory. --measure opts
into fresh server processes. The plan uses the G04 measurement-plan fields
source_root, source_commit, immutable_files, preflight_environment_id and
http_lanes (each with argv, env, port and expected_output_text). Optional
preflight_argv selects a reviewed host check instead of source_root's preflight.
This runner never represents GUI-retained measurements as canonical headless
qualification. HTTP SSE text events are not token timing or token-ID evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import struct
import subprocess
import time


CONDITION = {
    "condition_id": "serving-gui-retained-512mib-cool48-v1",
    "gui_retained": True,
    "idle_memory_limit_mib": 512,
    "start_temperature_limit_c": 48,
    "host_profile_id": "rtx4090-ubuntu22-driver580-host-v3",
    "host_profile_version": 3,
    "canonical_qualification": False,
    "scope": "single-concurrency HTTP serving diagnostic",
}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
        out.write("\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def token_digest(ids):
    return hashlib.sha256(b"".join(struct.pack("<I", token) for token in ids)).hexdigest()


def validate_artifacts(plan, request_path, binding_path, request, binding):
    pinned = {str(Path(path).resolve()): expected for path, expected in plan["immutable_files"].items()}
    for path, expected in pinned.items():
        if digest(path) != expected:
            raise ValueError("immutable artifact changed: " + path)
    for path in (request_path, binding_path):
        if pinned.get(str(Path(path).resolve())) != digest(path):
            raise ValueError("explicit input is not immutable-bound: " + str(path))
    source = plan["source_root"]
    revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", source, "status", "--porcelain"], text=True)
    if revision != plan["source_commit"] or dirty:
        raise ValueError("candidate source is not the pinned clean snapshot")
    if binding["source"]["git_commit"] != revision or binding["source"]["git_dirty"]:
        raise ValueError("correctness binding source differs from candidate")
    host_id = binding["environment"]["host"]["environment_id"]
    if host_id != plan["preflight_environment_id"]:
        raise ValueError("plan environment differs from correctness binding")
    workload = binding["workload"]
    if workload["concurrency"] != 1 or workload["sampling_id"] != "greedy":
        raise ValueError("runner requires concurrency one and greedy sampling")
    if (request["prompt_token_ids"] != binding["input_token_ids"]
            or request["prompt_tokens"] != len(binding["input_token_ids"])
            or request["requested_output_tokens"] != len(binding["generated_token_ids"])
            or request["prompt_tokens"] != workload["prompt_tokens"]
            or request["requested_output_tokens"] != workload["output_tokens"]):
        raise ValueError("request token contract differs from binding")
    if (request["temperature"] != 0 or request["top_p"] != 1
            or request.get("repetition_penalty", 1) != 1
            or request.get("cache_policy") != "prefix-cache-off"
            or request.get("http_eos_policy") not in ("natural-eos", "natural-eos with fixed-length receipt")):
        raise ValueError("unsupported sampling, cache or HTTP EOS policy")
    if set(plan["http_lanes"]) != {"riley", "vllm"}:
        raise ValueError("plan must supply Riley and vLLM HTTP lanes")
    texts = []
    for lane in plan["http_lanes"].values():
        if str(Path(lane["argv"][0]).resolve()) not in pinned:
            raise ValueError("server executable is not immutable-bound")
        if not isinstance(lane["port"], int) or not 0 < lane["port"] < 65536:
            raise ValueError("invalid HTTP port")
        texts.append(lane["expected_output_text"])
    if not texts[0] or texts[0] != texts[1]:
        raise ValueError("lane reference texts differ or are empty")
    if plan.get("qualification_blockers"):
        raise ValueError("qualification blockers: " + "; ".join(plan["qualification_blockers"]))


def gpu_snapshot(binding):
    index = str(binding["environment"]["gpu"]["device_index"])
    row = subprocess.check_output([
        "nvidia-smi", "-i", index,
        "--query-gpu=uuid,temperature.gpu,memory.used", "--format=csv,noheader,nounits",
    ], text=True).strip()
    parts = [part.strip() for part in row.split(",")]
    if len(parts) != 3 or parts[0] != binding["environment"]["gpu"]["uuid"]:
        raise ValueError("GPU identity changed or query returned multiple GPUs")
    processes = subprocess.check_output([
        "nvidia-smi", "-i", index, "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True).strip()
    return {"at_utc": datetime.now(timezone.utc).isoformat(), "gpu_uuid": parts[0],
            "temperature_c": int(parts[1]), "memory_used_mib": int(parts[2]),
            "compute_pids": processes.splitlines() if processes else []}


def check_idle(snapshot):
    if snapshot["compute_pids"]:
        raise ValueError("foreign CUDA compute processes present: " + ", ".join(snapshot["compute_pids"]))
    if snapshot["memory_used_mib"] > CONDITION["idle_memory_limit_mib"]:
        raise ValueError("GUI diagnostic idle GPU memory exceeds 512 MiB")


def preflight(plan, binding, directory, timeout=300):
    samples = []
    deadline = time.monotonic() + timeout
    try:
        while True:
            sample = gpu_snapshot(binding)
            samples.append(sample)
            check_idle(sample)
            if sample["temperature_c"] <= CONDITION["start_temperature_limit_c"]:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU cooldown to 48 C timed out")
            time.sleep(2)
    finally:
        write_json(directory / "cooldown.json", samples)
    env = os.environ.copy()
    env.update(RILEY_PREFLIGHT_OUTPUT_ROOT=str(directory),
               RILEY_PREFLIGHT_ENVIRONMENT_ID=plan["preflight_environment_id"])
    argv = plan.get("preflight_argv", ["bash", str(Path(plan["source_root"]) / "benchmarks/scripts/preflight.sh")])
    with (directory / "preflight.stdout").open("x") as out, (directory / "preflight.stderr").open("x") as err:
        subprocess.run(argv, cwd=plan["source_root"], env=env, stdout=out, stderr=err, check=True)
    latest = gpu_snapshot(binding)
    check_idle(latest)
    if latest["temperature_c"] > 48:
        raise ValueError("GPU warmed above 48 C during preflight")
    write_json(directory / "live-condition.json", {**CONDITION, "host_environment_id": plan["preflight_environment_id"], "gpu": latest})


def check_port(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def http_request(port, model, request, expected, streaming):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    body = {"model": model, "prompt": request["prompt"],
            "max_tokens": request["requested_output_tokens"], "temperature": 0,
            "top_p": 1, "stream": streaming}
    started = time.perf_counter_ns()
    events, text, finish, usage = [], "", None, None
    try:
        connection.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}: {response.read()[:300]!r}")
        if streaming:
            done = False
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                value = json.loads(payload)
                if value.get("error"):
                    raise ValueError("stream returned an error")
                if value.get("usage") is not None:
                    usage = value["usage"]
                for choice in value.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("unexpected additional completion")
                    delta = choice.get("text", "")
                    if delta:
                        events.append(time.perf_counter_ns())
                        text += delta
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
            if not done or not events:
                raise ValueError("stream missing DONE or text events")
        else:
            value = json.loads(response.read())
            if len(value["choices"]) != 1:
                raise ValueError("unexpected completion count")
            text = value["choices"][0]["text"]
            finish = value["choices"][0]["finish_reason"]
            usage = value["usage"]
        ended = time.perf_counter_ns()
        if text != expected or finish != "length":
            raise ValueError("output text or length finish differs from correctness reference")
        if usage is not None and (usage["prompt_tokens"] != request["prompt_tokens"]
                                 or usage["completion_tokens"] != request["requested_output_tokens"]):
            raise ValueError("HTTP token counts differ from correctness reference")
        return {"started_ns": started, "finished_ns": ended, "event_ns": events,
                "text": text, "finish_reason": finish, "usage": usage,
                "text_exact": True, "token_ids_validation": "unavailable-in-http-response",
                "token_counts_validation": "response-usage-exact" if usage else "reference-bound; no streaming usage",
                "event_unit": "sse-text-event" if streaming else None}
    finally:
        connection.close()


def wait_ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError("server exited during startup")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
        if time.monotonic() >= deadline:
            raise TimeoutError("server startup timed out")
        time.sleep(0.2)


def stop_owned_process(process):
    # start_new_session gives this invocation exclusive ownership of this group.
    # Never identify or terminate existing servers by port, name or GPU PID.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    # A worker may outlive its server leader; all members belong to our session.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_http(plan, role, request, directory, warmups, iterations, startup_timeout):
    lane = plan["http_lanes"][role]
    check_port(lane["port"])
    env = os.environ.copy()
    env.update(lane.get("env", {}))
    write_json(directory / "launch.json", {"argv": lane["argv"], "env": lane.get("env", {}),
                                           "cwd": plan["source_root"], "fresh_process": True})
    with (directory / "server.log").open("x") as log:
        process = subprocess.Popen(lane["argv"], env=env, cwd=plan["source_root"],
                                   stdout=log, stderr=log, start_new_session=True)
        try:
            wait_ready(process, lane["port"], startup_timeout)
            model = lane.get("model_id", plan.get("http_model_id", "g04-smol"))
            def request_one(streaming):
                return http_request(lane["port"], model, request, lane["expected_output_text"], streaming)
            write_json(directory / "calibration.json", request_one(False))
            with (directory / "warmups.jsonl").open("x") as out:
                for index in range(warmups):
                    # Both transports are checked even during discarded warmups.
                    row = {"index": index, "nonstream": request_one(False), "stream": request_one(True)}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            rows = []
            with (directory / "http-streaming.jsonl").open("x") as out:
                for index in range(iterations):
                    row = request_one(True)
                    row.update(index=index, mode="http-streaming", timestamp_unit="ns")
                    rows.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
            return rows
        finally:
            stop_owned_process(process)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(rows, output_tokens):
    elapsed = [(row["finished_ns"] - row["started_ns"]) / 1e6 for row in rows]
    first = [(row["event_ns"][0] - row["started_ns"]) / 1e6 for row in rows]
    return {"requests": len(rows), "e2e_ms": {"median": statistics.median(elapsed),
             "p95": percentile(elapsed, .95), "p99": percentile(elapsed, .99)},
            "first_text_event_ms": {"median": statistics.median(first),
             "p95": percentile(first, .95), "p99": percentile(first, .99)},
            "output_tokens_per_request_service_second": output_tokens * len(rows) / (sum(elapsed) / 1000),
            "output_tokens_per_wall_second": output_tokens * len(rows) / ((rows[-1]["finished_ns"] - rows[0]["started_ns"]) / 1e9),
            "http_tpot": "unmeasured; SSE events are not token boundaries"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "request", "binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--measure", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args(argv)
    if min(args.pairs, args.iterations) < 1 or args.warmups < 0 or args.startup_timeout <= 0:
        parser.error("pairs/iterations/timeout must be positive; warmups must be nonnegative")
    for name in ("plan", "request", "binding", "output"):
        setattr(args, name, getattr(args, name).resolve())
    plan, request, binding = (json.loads(path.read_text()) for path in (args.plan, args.request, args.binding))
    validate_artifacts(plan, args.request, args.binding, request, binding)
    args.output.mkdir(parents=False, exist_ok=False)
    receipt = {"schema_version": "riley.serving-optimization.v1", "measurement_started": args.measure,
               "condition": {**CONDITION, "host_environment_id": plan["preflight_environment_id"]},
               "inputs": {str(path): digest(path) for path in (args.plan, args.request, args.binding)},
               "runner_sha256": digest(__file__), "source_commit": plan["source_commit"],
               "pairs": args.pairs, "warmups_per_transport": args.warmups, "iterations": args.iterations,
               "reference_input_token_ids_sha256": token_digest(binding["input_token_ids"]),
               "reference_generated_token_ids_sha256": token_digest(binding["generated_token_ids"]),
               "http_generated_token_ids_verified": False,
               "legacy_vllm_raw": "not used; HTTP envelope binds actual environment; old raw remains uncanonical",
               "warmup_note": "one nonstream calibration then paired nonstream and stream warmups; all validated",
               "tail_estimator": "nearest rank within each process; small samples do not establish tail stability"}
    write_json(args.output / "preparation.json", receipt)
    if not args.measure:
        print(json.dumps({"prepared": True, "measurement_started": False, "output": str(args.output)}))
        return 0
    pairs = []
    try:
        for index in range(1, args.pairs + 1):
            pair = {"index": index, "order": ["riley", "vllm"] if index % 2 else ["vllm", "riley"]}
            for role in pair["order"]:
                directory = args.output / f"pair-{index:02}-{role}"
                directory.mkdir()
                try:
                    validate_artifacts(plan, args.request, args.binding, request, binding)
                    preflight(plan, binding, directory)
                    rows = run_http(plan, role, request, directory, args.warmups, args.iterations, args.startup_timeout)
                    validate_artifacts(plan, args.request, args.binding, request, binding)
                    pair[role] = summarize(rows, request["requested_output_tokens"])
                    write_json(directory / "execution-complete.json", {"completed": True, "summary": pair[role]})
                except BaseException as error:
                    write_json(directory / "failure.json", {"completed": False, "error": str(error), "type": type(error).__name__})
                    raise
            pair["riley_over_vllm_e2e"] = pair["riley"]["e2e_ms"]["median"] / pair["vllm"]["e2e_ms"]["median"]
            pairs.append(pair)
        write_json(args.output / "summary.json", {**receipt, "completed": True, "pair_results": pairs,
                   "paired_e2e_ratio_median": statistics.median(pair["riley_over_vllm_e2e"] for pair in pairs)})
        write_json(args.output / "completion.json", {"completed": True, "processes": 2 * args.pairs})
    except BaseException as error:
        write_json(args.output / "completion.json", {"completed": False, "error": str(error), "completed_pairs": len(pairs)})
        raise
    print(json.dumps({"completed": True, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
