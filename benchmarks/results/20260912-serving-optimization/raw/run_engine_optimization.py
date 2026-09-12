#!/usr/bin/env python3
"""Run immutable engine-only Riley/vLLM pairs under the shared GUI condition.

Preparation is the default. Inputs match run_serving_optimization.py, with
engine_lanes argv/env and reference_checker_python in the plan. The source and
closed raw schemas are never rewritten; an envelope binds the actual condition
and explicitly identifies a legacy vLLM environment label as noncanonical.
The existing lane contract is five pairs, five warmups, thirty measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import os
import statistics
import subprocess
import time

import run_serving_optimization as shared


PAIRS = 5
WARMUPS = 5
ITERATIONS = 30


def option(argv, name):
    if argv.count(name) != 1:
        raise ValueError("engine argv must contain exactly one " + name)
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ValueError("engine argv missing value for " + name)
    return argv[index + 1]


def validate_plan(plan, request_path, binding_path, request, binding):
    shared.validate_artifacts(plan, request_path, binding_path, request, binding)
    workload = binding["workload"]
    if workload["warmups"] != WARMUPS or workload["measured_iterations"] != ITERATIONS:
        raise ValueError("engine binding must use five warmups and thirty measurements")
    if plan.get("warmups_per_process", WARMUPS) != WARMUPS or plan.get("measured_requests_per_process", ITERATIONS) != ITERATIONS:
        raise ValueError("engine plan repetition contract differs")
    if set(plan["engine_lanes"]) != {"riley", "vllm"}:
        raise ValueError("both engine lanes required")
    pinned = {str(Path(path).resolve()): sha for path, sha in plan["immutable_files"].items()}
    for lane in plan["engine_lanes"].values():
        if str(Path(lane["argv"][0]).resolve()) not in pinned:
            raise ValueError("engine executable is not immutable-bound: " + lane["argv"][0])
        for argument in lane["argv"][1:]:
            if argument.endswith(".py") and str(Path(argument).resolve()) not in pinned:
                raise ValueError("engine Python entrypoint is not immutable-bound: " + argument)
    if str(Path(plan["reference_checker_python"]).resolve()) not in pinned:
        raise ValueError("reference checker interpreter is not immutable-bound")
    riley = plan["engine_lanes"]["riley"]["argv"]
    if option(riley, "--warmups") != str(WARMUPS) or option(riley, "--measured-iterations") != str(ITERATIONS):
        raise ValueError("Riley argv repetition contract differs")
    if option(riley, "--git-commit") != plan["source_commit"] or option(riley, "--git-dirty") != "false":
        raise ValueError("Riley argv source identity differs")
    if option(riley, "--executable-sha256") != shared.digest(riley[0]):
        raise ValueError("Riley argv executable identity differs")
    if option(riley, "--environment-id") != plan["preflight_environment_id"]:
        raise ValueError("Riley argv environment identity differs")
    vllm = plan["engine_lanes"]["vllm"]["argv"]
    if option(vllm, "--warm-state") != "warm":
        raise ValueError("vLLM engine lane must use warm-state warm")
    for argv in (riley, vllm):
        for flag, expected in (("--concurrency", "1"), ("--prompt-tokens", str(workload["prompt_tokens"])),
                               ("--output-tokens", str(workload["output_tokens"]))):
            if option(argv, flag) != expected:
                raise ValueError("engine argv workload differs: " + flag)


def timing(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid engine timing: " + name)
    return float(value)


def validate_request_times(ttft, tpot, e2e, output_tokens):
    values = {"ttft_ms": timing(ttft, "TTFT"), "tpot_ms": timing(tpot, "TPOT"), "e2e_ms": timing(e2e, "E2E")}
    if values["e2e_ms"] <= 0 or values["ttft_ms"] > values["e2e_ms"]:
        raise ValueError("engine request timing order invalid")
    if values["ttft_ms"] + (output_tokens - 1) * values["tpot_ms"] > values["e2e_ms"] + 1e-6:
        raise ValueError("engine TPOT extends beyond request duration")
    return values


def validate_vllm_rows(rows, binding, expected_count=ITERATIONS):
    if len(rows) != expected_count or [row["trial_index"] for row in rows] != list(range(1, expected_count + 1)):
        raise ValueError("vLLM request set missing, duplicated or reordered")
    prompt_hash = shared.token_digest(binding["input_token_ids"])
    output_hash = shared.token_digest(binding["generated_token_ids"])
    w = binding["workload"]
    normalized, batch_ms, labels = [], [], set()
    for row in rows:
        if row["status"] != "success" or row["failure_count"] != 0 or len(row["requests"]) != 1:
            raise ValueError("vLLM failed or concurrency changed")
        if row["model_id"] != w["model_id"] or row["model_revision"] != w["model_revision"] or row["dtype"] != w["dtype"]:
            raise ValueError("vLLM model identity changed")
        workload = row["workload"]
        for field in ("concurrency", "prompt_tokens", "output_tokens", "sampling_id"):
            if workload[field] != w[field]:
                raise ValueError("vLLM workload changed: " + field)
        if workload["warm_state"] != "warm" or row["warm_state"] != "warm":
            raise ValueError("vLLM warm state changed")
        req = row["requests"][0]
        if (req["status"] != "success" or req["prompt_tokens"] != w["prompt_tokens"]
                or req["requested_output_tokens"] != w["output_tokens"] or req["generated_tokens"] != w["output_tokens"]):
            raise ValueError("vLLM request failed or token count changed")
        if req["prompt_token_ids_sha256"] != prompt_hash or req["generated_token_ids_sha256"] != output_hash:
            raise ValueError("vLLM output differs from exact token reference")
        normalized.append(validate_request_times(req["ttft_ms"], req["mean_tpot_ms"], req["end_to_end_ms"], w["output_tokens"]))
        wall = timing(row["metrics"]["batch_wall_ms"], "batch wall")
        if wall <= 0:
            raise ValueError("vLLM batch wall must be positive")
        batch_ms.append(wall)
        labels.add(row["environment_id"])
    return normalized, {"valid": True, "requests": expected_count, "tokens_exact": True,
                        "raw_environment_ids": sorted(labels), "raw_canonical_eligible": False,
                        "aggregate_output_tokens_per_second": w["output_tokens"] * len(rows) / (sum(batch_ms) / 1000),
                        "throughput_boundary": "sum of measured vLLM batch_wall_ms"}


def validate_riley_rows(run, binding):
    w = binding["workload"]
    if run["status"] != "success" or run["failure_count"] != 0 or len(run["requests"]) != ITERATIONS:
        raise ValueError("Riley engine run incomplete or failed")
    normalized = [validate_request_times(row["ttft_ms"], row["tpot_ms"], row["e2e_ms"], w["output_tokens"])
                  for row in run["requests"]]
    aggregate = run["aggregate"]
    throughput = timing(aggregate["throughput_output_tokens_per_second"], "Riley throughput")
    if throughput <= 0:
        raise ValueError("Riley throughput must be positive")
    return normalized, {"aggregate_output_tokens_per_second": throughput,
                        "throughput_boundary": "native-profile aggregate throughput field",
                        "aggregate_host": aggregate["host"], "aggregate_cuda": aggregate["cuda"],
                        "counters": aggregate["counters"], "tokens_exact": True}


def summarize(rows, output_tokens, extra):
    result = {"requests": len(rows), "tokens_exact": True, **extra}
    for field in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[field] for row in rows]
        result[field] = {"median": statistics.median(values), "p95": shared.percentile(values, .95),
                         "p99": shared.percentile(values, .99)}
    result["output_tokens_per_request_service_second"] = output_tokens * len(rows) / (sum(row["e2e_ms"] for row in rows) / 1000)
    return result


def run_lane(plan, role, index, binding, binding_path, directory, timeout):
    lane = plan["engine_lanes"][role]
    values = {"index": str(index), "output": str(directory),
              "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    argv = [argument.format(**values) for argument in lane["argv"]]
    env = os.environ.copy()
    env.update(lane.get("env", {}))
    with (directory / "stdout.log").open("x") as out, (directory / "stderr.log").open("x") as err:
        process = subprocess.Popen(argv, cwd=plan["source_root"], env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            shared.write_json(directory / "launch.json", {"argv": argv, "env": lane.get("env", {}),
                              "cwd": plan["source_root"], "pid": process.pid, "fresh_process": True})
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, argv)
        finally:
            shared.stop_owned_process(process)
    if role == "riley":
        result_path = directory / "native-profile.json"
        checker = Path(plan["source_root"]) / "benchmarks/scripts/check_vllm_profile_run.py"
        with (directory / "reference-check.json").open("x") as out, (directory / "reference-check.stderr").open("x") as err:
            subprocess.run([plan["reference_checker_python"], str(checker), str(result_path), "--binding", str(binding_path)],
                           stdout=out, stderr=err, check=True, cwd=plan["source_root"], timeout=120)
        checked = json.loads((directory / "reference-check.json").read_text())
        if checked.get("valid") is not True or checked.get("tokens_exact") is not True:
            raise ValueError("Riley checker did not certify exact output tokens")
        normalized, extra = validate_riley_rows(json.loads(result_path.read_text()), binding)
    else:
        rows = [json.loads(line) for line in (directory / "vllm/raw.jsonl").read_text().splitlines() if line.strip()]
        normalized, extra = validate_vllm_rows(rows, binding)
        shared.write_json(directory / "reference-check.json", extra)
    return summarize(normalized, binding["workload"]["output_tokens"], extra)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "request", "binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--measure", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--process-timeout", type=float, default=1200)
    args = parser.parse_args(argv)
    if (args.pairs, args.warmups, args.iterations) != (PAIRS, WARMUPS, ITERATIONS):
        parser.error("engine contract requires exactly 5 pairs, 5 warmups and 30 measurements")
    if args.process_timeout <= 0:
        parser.error("process timeout must be positive")
    for name in ("plan", "request", "binding", "output"):
        setattr(args, name, getattr(args, name).resolve())
    plan, request, binding = (json.loads(path.read_text()) for path in (args.plan, args.request, args.binding))
    validate_plan(plan, args.request, args.binding, request, binding)
    args.output.mkdir(parents=False, exist_ok=False)
    receipt = {"schema_version": "riley.engine-optimization.v1", "measurement_started": args.measure,
               "condition": {**shared.CONDITION, "scope": "single-concurrency engine-only diagnostic",
                             "host_environment_id": plan["preflight_environment_id"]},
               "inputs": {str(path): shared.digest(path) for path in (args.plan, args.request, args.binding)},
               "runner_sha256": shared.digest(__file__), "shared_runner_sha256": shared.digest(shared.__file__),
               "source_commit": plan["source_commit"], "pairs": PAIRS, "warmups": WARMUPS, "iterations": ITERATIONS,
               "warmup_token_validation": "delegated to pinned engine adapters; retained requests independently checked",
               "legacy_vllm_raw": "preserved without rewriting; not canonical-eligible; actual environment is envelope plus live preflight",
               "timing_boundary": "engine request TTFT/TPOT/E2E; not CUDA kernel timing",
               "tail_estimator": "nearest rank per process; thirty samples do not establish high-concurrency tail stability"}
    shared.write_json(args.output / "preparation.json", receipt)
    if not args.measure:
        print(json.dumps({"prepared": True, "measurement_started": False, "output": str(args.output)}))
        return 0
    pairs = []
    try:
        for index in range(1, PAIRS + 1):
            pair = {"index": index, "order": ["riley", "vllm"] if index % 2 else ["vllm", "riley"]}
            for role in pair["order"]:
                directory = args.output / f"pair-{index:02}-{role}"
                directory.mkdir()
                try:
                    validate_plan(plan, args.request, args.binding, request, binding)
                    shared.preflight(plan, binding, directory)
                    pair[role] = run_lane(plan, role, index, binding, args.binding, directory, args.process_timeout)
                    validate_plan(plan, args.request, args.binding, request, binding)
                    shared.write_json(directory / "execution-complete.json", {"completed": True, "summary": pair[role]})
                except BaseException as error:
                    shared.write_json(directory / "failure.json", {"completed": False, "error": str(error), "type": type(error).__name__})
                    raise
            pair["riley_over_vllm"] = {field: pair["riley"][field]["median"] / pair["vllm"][field]["median"]
                                       for field in ("ttft_ms", "tpot_ms", "e2e_ms")}
            pair["riley_over_vllm"]["service_throughput"] = (pair["riley"]["output_tokens_per_request_service_second"]
                                                                          / pair["vllm"]["output_tokens_per_request_service_second"])
            pairs.append(pair)
        summary = {**receipt, "completed": True, "pair_results": pairs,
                   "paired_ratio_medians": {field: statistics.median(pair["riley_over_vllm"][field] for pair in pairs)
                                             for field in pairs[0]["riley_over_vllm"]}}
        shared.write_json(args.output / "summary.json", summary)
        shared.write_json(args.output / "completion.json", {"completed": True, "processes": 2 * PAIRS})
    except BaseException as error:
        shared.write_json(args.output / "completion.json", {"completed": False, "error": str(error), "completed_pairs": len(pairs)})
        raise
    shared.write_json(args.output / "raw-sha256.json", {str(path.relative_to(args.output)): shared.digest(path)
                                                       for path in sorted(args.output.rglob("*")) if path.is_file()})
    print(json.dumps({"completed": True, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
