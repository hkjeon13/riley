"""CPU-testable orchestration and fail-closed results for compare.py."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


class ComparisonError(ValueError):
    """A campaign did not produce its declared valid cells."""


def parser(scenarios):
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "label", "model", "gpu", "riley_bin"):
        result.add_argument(name)
    result.add_argument("--order", default="rvvr")
    result.add_argument("--scenarios", default=",".join(scenarios))
    result.add_argument("--riley-short", default="")
    result.add_argument("--riley-long", default="")
    result.add_argument("--port", type=int, default=18300)
    result.add_argument("--vllm-memory", type=float, default=0.8)
    result.add_argument("--tp", type=int, default=1)
    result.add_argument("--prefix-cache", type=int, default=0)
    result.add_argument("--riley-executable", type=Path)
    result.add_argument("--vllm-executable", type=Path)
    result.add_argument("--models-dir", type=Path)
    result.add_argument("--load-script", type=Path)
    result.add_argument("--cuda-lib-dir", default="/usr/local/cuda-12.4/lib64")
    result.add_argument("--startup-timeout", type=float, default=600)
    result.add_argument("--scenario-timeout", type=float, default=1800)
    result.add_argument("--shutdown-timeout", type=float, default=60)
    result.add_argument("--cooldown-seconds", type=float, default=5)
    return result


def validate_arguments(args, scenarios):
    names = args.scenarios.split(",")
    if not names or len(set(names)) != len(names) or set(names) - set(scenarios):
        raise ComparisonError("scenarios must be a unique, nonempty list of known cells")
    if not args.order or set(args.order) - {"r", "v"}:
        raise ComparisonError("order must contain only r and v")
    label = Path(args.label)
    if label.is_absolute() or not label.parts or ".." in label.parts:
        raise ComparisonError("label must be a relative evidence path without '..'")
    if not 1 <= args.port <= 65535 or args.tp < 1 or args.prefix_cache < 0:
        raise ComparisonError("invalid port, tensor parallel size or prefix-cache capacity")
    if args.tp > len(args.gpu.split(",")):
        raise ComparisonError("tensor parallel size exceeds selected devices")
    if not math.isfinite(args.vllm_memory) or not 0 < args.vllm_memory <= 1:
        raise ComparisonError("vLLM memory fraction must be finite and in (0, 1]")
    for name in ("startup_timeout", "scenario_timeout", "shutdown_timeout", "cooldown_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != "cooldown_seconds" and value == 0):
            raise ComparisonError(f"{name} must be finite and positive (cooldown may be zero)")
    return names


def paths(args):
    root = Path(args.root).resolve()
    legacy_load = root / "src/benchmarks/live/load.py"
    return {
        "root": root,
        "model": (args.models_dir or root / "models") / args.model,
        "riley": args.riley_executable or root / "bin" / f"riley-{args.riley_bin}",
        "vllm": args.vllm_executable or root / "tools/vllm-venv/bin/vllm",
        "load": args.load_script or (legacy_load if legacy_load.is_file() else Path(__file__).with_name("load.py")),
        "out": root / "evidence" / args.label,
    }


def environment(args, locations):
    result = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.gpu}
    result["PATH"] = str(locations["vllm"].parent) + os.pathsep + result.get("PATH", "")
    if args.cuda_lib_dir:
        result["LD_LIBRARY_PATH"] = args.cuda_lib_dir + os.pathsep + result.get("LD_LIBRARY_PATH", "")
    return result


def server_command(args, locations, engine, config):
    if engine == "r":
        command = [str(locations["riley"]), "serve", "--model", str(locations["model"]),
                   "--model-id", "m", "--bind", f"127.0.0.1:{args.port}",
                   "--max-active-sequences", "32", "--max-weight-bytes", "68719476736"]
        if args.prefix_cache:
            command += ["--prefix-cache-blocks", str(args.prefix_cache)]
        if args.tp > 1:
            command += ["--tensor-parallel-devices", ",".join(map(str, range(args.tp)))]
        command += ["--inference-profile", "native-bf16-varlen-v1"]
        if config == "short":
            command += ["--max-output-tokens", "128"] + shlex.split(args.riley_short)
        else:
            command += ["--max-sequence-tokens", "8192", "--max-output-tokens", "2048"] + shlex.split(args.riley_long)
        return command
    return [str(locations["vllm"]), "serve", str(locations["model"]), "--served-model-name", "m",
            "--port", str(args.port), "--host", "127.0.0.1", "--max-model-len", "1024" if config == "short" else "8192",
            "--gpu-memory-utilization", str(args.vllm_memory), "--generation-config", "vllm",
            "--max-num-seqs", "32", "--disable-log-stats", "--tensor-parallel-size", str(args.tp),
            "--enable-prefix-caching" if args.prefix_cache else "--no-enable-prefix-caching"]


def stop_process(process, timeout):
    """Reap this process and its own new session, including timeout descendants."""
    def send(sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                process.send_signal(sig)
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        process.wait()
    finally:
        # A leader may exit while a descendant still holds GPU memory or logs.
        if os.name == "posix":
            send(signal.SIGKILL)


def start_server(command, env, log, ready_url, timeout, shutdown_timeout):
    with log.open("wb") as handle:
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ComparisonError(f"server exited {process.returncode}; see {log}")
            try:
                with urllib.request.urlopen(ready_url, timeout=min(2, max(deadline - time.monotonic(), 0.01))) as response:
                    if response.status == 200:
                        return process
            except OSError:
                pass
            time.sleep(min(0.1, max(deadline - time.monotonic(), 0)))
        raise ComparisonError(f"server readiness timeout; see {log}")
    except BaseException:
        stop_process(process, shutdown_timeout)
        raise


def workload_arguments(text):
    """Mirror only load.py's declared inputs; unknown flags cannot pass silently."""
    p = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    for name, default, kind in (
        ("prompt-words", "30", str), ("max-tokens", 64, int), ("requests", 192, int),
        ("warmup", 32, int), ("seed", 1, int), ("concurrency", None, int),
        ("rate", None, float), ("burst", None, str), ("shared-prefix-words", 0, int), ("extra", "{}", str),
    ):
        p.add_argument("--" + name, default=default, type=kind)
    return vars(p.parse_args(shlex.split(text)))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"duplicate result key: {key}")
        result[key] = value
    return result


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ComparisonError("nonfinite JSON number")
    return value


def _reject_constant(text):
    raise ComparisonError(f"nonfinite JSON constant: {text}")


def validate_result(path, expected, address, model="m"):
    with path.open("rb") as handle:
        raw = handle.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ComparisonError(f"result exceeds size bound: {path}")
    value = json.loads(raw, object_pairs_hook=_unique_object,
                       parse_float=_finite_float, parse_constant=_reject_constant)
    if not isinstance(value, dict) or not isinstance(value.get("config"), dict):
        raise ComparisonError(f"missing result/config object: {path}")
    for key in ("requests", "ok", "errors", "rejected_429", "warmup_requests", "warmup_ok", "warmup_errors"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ComparisonError(f"invalid {key}: {path}")
    if value["requests"] != expected["requests"] or value["ok"] != value["requests"]:
        raise ComparisonError(f"incomplete measured requests: {path}")
    if value["errors"] or value["rejected_429"] or value.get("other_failures") != []:
        raise ComparisonError(f"failed/rejected requests cannot qualify: {path}")
    if value["warmup_requests"] != expected["warmup"] or value["warmup_ok"] != expected["warmup"] or value["warmup_errors"]:
        raise ComparisonError(f"failed/incomplete warmup: {path}")
    if value["config"].get("address") != address or value["config"].get("model") != model:
        raise ComparisonError(f"result endpoint/model mismatch: {path}")
    for key, wanted in expected.items():
        if key not in value["config"]:
            raise ComparisonError(f"missing workload config {key}: {path}")
        actual = value["config"].get(key)
        if key == "extra":
            if not isinstance(actual, str):
                raise ComparisonError(f"invalid workload extra: {path}")
            actual, wanted = json.loads(actual), json.loads(wanted)
        elif type(actual) is not type(wanted):
            raise ComparisonError(f"invalid workload config type for {key}: {path}")
        if actual != wanted:
            raise ComparisonError(f"workload config mismatch for {key}: {path}")
    metrics = ["wall_s", "output_tok_s", "requests_s", "prompt_tokens_mean", "completion_tokens_mean", "itl_p99_ms"]
    metrics += [f"{name}_{q}_ms" for name in ("ttft", "tpot", "e2e") for q in ("p50", "p95", "p99")]
    for key in metrics:
        number = value.get(key)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ComparisonError(f"missing/nonfinite/negative metric {key}: {path}")
    if any(value[key] <= 0 for key in metrics[:5]) or value["completion_tokens_mean"] > expected["max_tokens"]:
        raise ComparisonError(f"invalid wall time, rate or token counts: {path}")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_load(command, env, log, timeout, shutdown_timeout):
    with log.open("wb") as handle:
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise ComparisonError(f"load process exited {code}; see {log}")
    except subprocess.TimeoutExpired as error:
        raise ComparisonError(f"load timeout; see {log}") from error
    finally:
        stop_process(process, shutdown_timeout)


def run(args, scenarios):
    names = validate_arguments(args, scenarios)
    locations = paths(args)
    # A fresh label prevents old JSON from being mistaken for a new result.
    locations["out"].mkdir(parents=True, exist_ok=False)
    env = environment(args, locations)
    manifest = {"schema_version": "riley.serving-comparison-run.v1", "status": "running",
                "scope": "head-to-head" if set(args.order) == {"r", "v"} else "single-engine-diagnostic",
                "order": args.order, "scenarios": names, "devices": args.gpu, "cells": [],
                "commands": [], "python": sys.version, "source_sha256": {}}
    seen = {}
    try:
        for path in (Path(__file__), Path(__file__).with_name("compare.py"), locations["load"],
                     locations["riley"], locations["vllm"], locations["model"] / "riley-checkpoint.json"):
            if path.is_file():
                manifest["source_sha256"][str(path)] = file_sha256(path)
        for engine in args.order:
            seen[engine] = seen.get(engine, 0) + 1
            tag = f"{engine}{seen[engine]}"
            for config in ("short", "long"):
                selected = [name for name in names if scenarios[name][0] == config]
                if not selected:
                    continue
                command = server_command(args, locations, engine, config)
                manifest["commands"].append({"kind": "server", "tag": tag, "config": config, "argv": command})
                ready = f"http://127.0.0.1:{args.port}" + ("/readyz" if engine == "r" else "/health")
                process = start_server(command, env, locations["out"] / f"{tag}-{config}.server.log",
                                       ready, args.startup_timeout, args.shutdown_timeout)
                try:
                    for name in selected:
                        output = locations["out"] / f"{tag}-{name}.json"
                        load_args = scenarios[name][1]
                        command = [sys.executable, str(locations["load"]), f"127.0.0.1:{args.port}", "m",
                                   *shlex.split(load_args), "--out", str(output)]
                        manifest["commands"].append({"kind": "load", "tag": tag, "scenario": name,
                                                     "argv": command})
                        run_load(command, env, locations["out"] / f"{tag}-{name}.load.log",
                                 args.scenario_timeout, args.shutdown_timeout)
                        validate_result(output, workload_arguments(load_args), f"127.0.0.1:{args.port}")
                        if process.poll() is not None:
                            raise ComparisonError(f"server exited during {tag}-{name}")
                        manifest["cells"].append({"tag": tag, "scenario": name, "status": "passed",
                                                  "result": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
                        print(tag, name, "validated", flush=True)
                finally:
                    stop_process(process, args.shutdown_timeout)
                if process.returncode not in (0, -signal.SIGTERM):
                    raise ComparisonError(f"server shutdown failed: {tag}-{config}: {process.returncode}")
                time.sleep(args.cooldown_seconds)
        if len(manifest["cells"]) != len(args.order) * len(names):
            raise ComparisonError("declared cell inventory is incomplete")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (locations["out"] / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print("COMPARE_DONE", flush=True)
    return manifest


def main(scenarios, argv=None):
    args = parser(scenarios).parse_args(argv)
    try:
        run(args, scenarios)
    except (ComparisonError, OSError, ValueError) as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
