"""Run a held A100 model/engine arm and optionally export Nsight evidence."""
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time

from compare_runner import file_sha256
from profile_workload import save
from profile_summary import summarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holder import gpu_processes, identity

MODELS = {
    "qwen14": ("Qwen2.5-14B-Instruct", 2576),
    "coder14": ("Qwen2.5-Coder-14B-Instruct", 2576),
    "qwen4": ("Qwen3-4B", 10050),
}
MODES = ("riley", "vllm", "profile")
RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
NATIVE = ROOT / "evidence/w4-schema-build-20261119.QWnmhslP/target/release/riley"
VLLM = ROOT / "tools/vllm-venv/bin/vllm"
NSYS = ROOT / "tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys"
BUCKETS = "1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024"


def case(words, token_limit, concurrency, requests, warmup, seed=1, **settings):
    return {"prompt_words": words, "max_tokens": token_limit, "concurrency": concurrency,
            "requests": requests, "warmup": warmup, "seed": seed, **settings}


def cells(model, mode):
    if model == "qwen4":
        return {f"words-{words}": case([words], 32, 1, 12, 2)
                for words in (30, 1700, 3500, 5250, 6800)}
    tests = {"rag-c1": case([1500, 3000], 128, 1, 16, 4),
             "rag-c8-s1": case([1500, 3000], 128, 8, 64, 8)}
    if mode != "profile":
        tests["rag-c8-s2"] = case([1500, 3000], 128, 8, 64, 8, seed=2)
    tests.update({
        "wave-sync": case([1500, 3000], 128, 8, 40, 8, stagger_ms=0),
        "wave-50ms": case([1500, 3000], 128, 8, 40, 8, stagger_ms=50),
    })
    return tests


def pid_parent(pid):
    with Path(f"/proc/{pid}/status").open() as status:
        return next(int(line.split()[1]) for line in status if line.startswith("PPid:"))


def descends_from(pid, parent):
    for _ in range(32):
        if pid == parent:
            return True
        if pid <= 1:
            return False
        pid = pid_parent(pid)
    return False


def run(model_key, mode):
    model_name, expected_blocks = MODELS[model_key]
    out = RUN / f"{model_key}-{mode}"
    out.mkdir()
    model = ROOT / "models" / model_name
    model_config = json.loads((model / "config.json").read_bytes())
    heads = model_config["num_attention_heads"]
    dimension = model_config.get("head_dim") or model_config["hidden_size"] // heads
    block_bytes = model_config["num_hidden_layers"] * model_config.get("num_key_value_heads", heads) * dimension * 4 * 16
    blocks = shlex.split(subprocess.check_output(
        [sys.executable, "-B", str(ROOT / "kvb.py"), str(model)], text=True, timeout=30,
    ))
    if blocks != ["--kv-blocks", str(expected_blocks)]:
        raise ValueError(f"checkpoint KV budget changed: {blocks}")
    total_kv = expected_blocks * block_bytes
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    endpoint = f"127.0.0.1:{port}"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    env.pop("RILEY_HOST_STEP_TRACE", None)
    env.pop("RILEY_TIMING", None)
    overrides = {}
    if mode == "vllm":
        server = [
            str(VLLM), "serve", str(model), "--served-model-name", "m", "--host", "127.0.0.1", "--port", str(port),
            "--max-model-len", "8192", "--max-num-seqs", "32",
            "--block-size", "16", "--kv-cache-memory-bytes", str(total_kv), "--gpu-memory-utilization", "0.9",
            "--generation-config", "vllm", "--disable-log-stats", "--no-enable-prefix-caching",
        ]
        env.pop("LD_LIBRARY_PATH", None)
        env.update(PATH=f"{VLLM.parent}:{env['PATH']}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   VLLM_CACHE_ROOT=str(out / "vllm-cache"), TORCHINDUCTOR_CACHE_DIR=str(out / "torch-cache"))
        ready = "/health"
    else:
        server = [
            str(NATIVE), "serve", "--model", str(model), "--model-id", "m", "--bind", endpoint,
            "--max-active-sequences", "32", "--max-weight-bytes", "68719476736",
            "--inference-profile", "native-bf16-varlen-v1", "--max-sequence-tokens", "8192",
            "--max-output-tokens", "256", *blocks, "--batch-token-budget", "1024",
            "--prefill-chunk-tokens", "1024", "--batch-shape-buckets", BUCKETS,
            "--prefill-burst-steady-budget", "256", "--prefill-burst-max-decode-rows", "32",
            "--prefill-burst-min-prompt-tokens", "512",
        ]
        env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64"
        ready = "/readyz"
        if mode == "profile":
            overrides["RILEY_HOST_STEP_TRACE"] = "1"
    plan = {"server": server, "address": endpoint, "ready": f"http://{endpoint}{ready}", "env": overrides,
            "profile": mode == "profile", "cells": cells(model_key, mode)}
    plan_path = out / "plan.json"
    save(plan_path, plan)
    workload = Path(__file__).with_name("profile_workload.py")
    argv = [sys.executable, "-B", str(workload), str(plan_path)]
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    holder_pid = int(env["RILEY_GPU_HOLDER_PID"])
    holder_ticks = env["RILEY_GPU_HOLDER_START_TICKS"]
    lease = Path(env["RILEY_GPU_HOLDER_LEASE_DIR"]) / "gpu7.status.json"
    start_ticks = int(identity(os.getpid()))
    server_pid_file = out / "server.pid"
    receipt = {"model": model_name, "mode": mode, "gpu": gpu, "holder_pid": holder_pid,
               "holder_start_ticks": holder_ticks, "common_kv_blocks": expected_blocks,
               "kv_block_bytes": block_bytes, "kv_cache_bytes": total_kv,
               "sha256": {str(path): file_sha256(path) for path in (
                   NATIVE if mode != "vllm" else VLLM,
                   model / "riley-checkpoint.json", model / "config.json", model / "tokenizer.json",
                   plan_path, Path(__file__), workload, workload.with_name("load.py"),
                   workload.with_name("profile_summary.py"),
               )}, "status": "running"}
    samples = 0
    maximum = 0

    def own_server():
        if not server_pid_file.is_file():
            return None
        try:
            pid = int(server_pid_file.read_text())
            proc = Path(f"/proc/{pid}")
            command = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            expected = [os.fsencode(arg) for arg in server]
            if (proc.stat().st_uid == os.getuid() and int(identity(pid)) >= start_ticks
                    and command[-len(expected):] == expected):
                return pid
        except (OSError, ValueError):
            pass
        return None

    def sample(process=None):
        nonlocal samples, maximum
        state = json.loads(lease.read_bytes())
        if (identity(holder_pid) != holder_ticks or state["pid"] != holder_pid
                or state["phase"] != "active" or state["allocation_bytes"] != 64 * 1024**2
                or time.time_ns() - state["updated_ns"] >= 10**10):
            raise RuntimeError("GPU holder did not maintain the active lease")
        rows = gpu_processes(gpu)
        target = own_server()
        for row in rows:
            pid = row["pid"]
            if pid == holder_pid:
                continue
            try:
                owned = process is not None and (os.getpgid(pid) == process.pid
                                                 or (target is not None and descends_from(pid, target)))
            except ProcessLookupError:
                continue
            except FileNotFoundError:
                continue
            if not owned:
                raise RuntimeError(f"unexpected process on selected GPU pid={pid}")
        if not any(row["pid"] == holder_pid for row in rows):
            raise RuntimeError("missing holder GPU context")
        samples += 1
        maximum = max(maximum, sum(row["memory_mib"] for row in rows))

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    proc = None
    try:
        sample()
        with tempfile.TemporaryDirectory(prefix="riley-w3-nsys-", dir="/var/tmp") as scratch:
            report = str(out / "trace")
            if mode == "profile":
                env["NSYS_TMPDIR"] = scratch
                receipt["sha256"][str(NSYS)] = file_sha256(NSYS)
                argv = [str(NSYS), "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                        "--sample=none", "--cpuctxsw=none", "-o", report, *argv]
            receipt["command"] = argv
            with (out / "client.log").open("xb") as log:
                proc = subprocess.Popen(argv, start_new_session=True, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1250
            while proc.poll() is None:
                sample(proc)
                if time.monotonic() > deadline:
                    raise TimeoutError("W3 arm")
                time.sleep(0.85)
            if proc.returncode or own_server() is not None:
                raise RuntimeError(f"workload/profile exit={proc.returncode} or server is still live")
        results = {}
        for name, spec in plan["cells"].items():
            path = out / f"{name}.json"
            result = json.loads(path.read_bytes())
            if result["statistics"]["ok"] != spec["requests"] or len(result["measured"]) != spec["requests"]:
                raise ValueError(f"missing measured records in {name}")
            results[name] = file_sha256(path)
        receipt["results"] = results
        if mode == "vllm":
            startup = (out / "server.log").read_text(errors="replace")
            matches = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", startup)
            receipt["reported_kv_tokens"] = [int(value.replace(",", "")) for value in matches]
            receipt["kv_capacity_matches"] = receipt["reported_kv_tokens"] == [expected_blocks * 16]
        if mode == "profile":
            report_file = out / "trace.nsys-rep"
            sqlite = out / "trace.sqlite"
            with (out / "export.log").open("xb") as log:
                subprocess.run([str(NSYS), "export", "--type", "sqlite", "--output", str(sqlite), str(report_file)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=160)
            pid = int(server_pid_file.read_text())
            for name in plan["cells"]:
                data = json.loads((out / f"{name}.json").read_bytes())
                save(out / f"{name}.profile.json", summarize(sqlite, data, pid))
            receipt["raw_report_sha256"] = file_sha256(report_file)
            receipt["sqlite_sha256"] = file_sha256(sqlite)
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        target = own_server()
        if target is not None:
            try:
                os.kill(target, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        receipt.update(nvml_samples=samples, max_process_memory_mib=maximum)
        save(out / "receipt.json", receipt)


if __name__ == "__main__":
    model, mode = sys.argv[1:]
    if model not in MODELS or mode not in MODES or model == "qwen4" and mode != "profile":
        raise SystemExit("unknown model/mode")
    run(model, mode)
