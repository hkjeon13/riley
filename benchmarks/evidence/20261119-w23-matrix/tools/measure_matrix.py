"""Measure the two event modes crossed with the two mixed prefill caps."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from compare_runner import file_sha256, validate_result, workload_arguments
from load import document
from long_accuracy import complete
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holder import gpu_processes, identity

event, policy, seed = sys.argv[1:]
if event not in ("immediate", "after-submit") or policy not in ("32", "2") or seed not in ("1", "2"):
    raise SystemExit("unknown event mode, prefill cap or seed")
MODE = f"{event}-{policy}-s{seed}"
ROOT = Path(__file__).resolve().parents[2].parent
RUN = Path(__file__).resolve().parents[1]
OUT = RUN / MODE
MODEL = ROOT / "models/Qwen2.5-0.5B-Instruct"
RILEY = ROOT / "evidence/w4-schema-build-20261119.QWnmhslP/target/release/riley"
SOURCE = "f7c956d3cc7e6e59c98698cce891df42bc84ef03"
BINARY_SHA256 = "3e070d05c93b6784e24eefd423a77ed260162c7ddd9fcde67bec135c3dc53e70"

gpu = os.environ["CUDA_VISIBLE_DEVICES"]
holder_pid = int(os.environ["RILEY_GPU_HOLDER_PID"])
holder_ticks = os.environ["RILEY_GPU_HOLDER_START_TICKS"]
env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1"}
env.pop("RILEY_HOST_STEP_TRACE", None)
env.pop("RILEY_TIMING", None)
env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64"
env["RILEY_SHUTDOWN_METRICS_PATH"] = str(OUT / "shutdown-metrics.json")

OUT.mkdir()
(OUT / "tmp").mkdir()
env["TMPDIR"] = str(OUT / "tmp")
with socket.socket() as address:
    address.bind(("127.0.0.1", 0))
    port = address.getsockname()[1]
endpoint = f"127.0.0.1:{port}"

kv = shlex.split(subprocess.check_output([sys.executable, str(ROOT / "kvb.py"), str(MODEL)], text=True, timeout=20))
if kv and (len(kv) != 2 or kv[0] != "--kv-blocks" or not kv[1].isdigit() or int(kv[1]) < 1):
    raise RuntimeError("unexpected KV budget")
buckets = "1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024"
server = [str(RILEY), "serve", "--model", str(MODEL), "--model-id", "m", "--bind", endpoint,
          "--max-active-sequences", "32", "--max-weight-bytes", "68719476736",
          "--inference-profile", "native-bf16-varlen-v1", "--max-sequence-tokens", "8192",
          "--max-output-tokens", "2048", *kv, "--batch-token-budget", "1024",
          "--prefill-chunk-tokens", "1024", "--batch-shape-buckets", buckets,
          "--prefill-burst-steady-budget", "256", "--prefill-burst-max-decode-rows", policy,
          "--prefill-burst-min-prompt-tokens", "512", "--event-publication", f"{event}-v1"]
cells = {
    "closed-c32": "--prompt-words 30 --max-tokens 64 --concurrency 32 --requests 192 --warmup 32",
    "rag-c1": "--prompt-words 1500,3000 --max-tokens 128 --concurrency 1 --requests 16 --warmup 4",
    "rag-c8": "--prompt-words 1500,3000 --max-tokens 128 --concurrency 8 --requests 64 --warmup 8",
    "rag-r3": "--prompt-words 1500,3000 --max-tokens 128 --rate 3 --requests 64 --warmup 8",
}

processes = []
max_memory = 0.0
samples = 0
seen_holder_active = False
receipt = {"mode": MODE, "gpu_uuid": gpu, "holder_pid": holder_pid, "holder_start_ticks": holder_ticks,
           "server_argv": server, "seed": int(seed), "policy": int(policy), "event_publication": f"{event}-v1",
           "riley_source": SOURCE, "cells": {},
           "sha256": {str(p): file_sha256(p) for p in (RILEY, MODEL / "riley-checkpoint.json",
                                                       MODEL / "tokenizer.json", Path(__file__), *[Path(__file__).with_name(name)
                                                       for name in ("load.py", "long_accuracy.py", "compare_runner.py")])}}
if receipt["sha256"][str(RILEY)] != BINARY_SHA256:
    raise RuntimeError("Riley binary differs from the W4 candidate")


def sample(allow_server=True):
    global max_memory, samples, seen_holder_active
    if identity(holder_pid) != holder_ticks:
        raise RuntimeError("holder identity changed")
    status = json.loads((Path(os.environ["RILEY_GPU_HOLDER_LEASE_DIR"]) / f"gpu{os.environ['W3_GPU_INDEX']}.status.json").read_text())
    if (status["phase"] != "active" or status["allocation_bytes"] != 64 * 1024**2
            or status["pid"] != holder_pid or time.time_ns() - status["updated_ns"] > 10**10):
        raise RuntimeError("holder did not maintain active reservation")
    seen_holder_active = True
    rows = gpu_processes(gpu)
    if not any(r["pid"] == holder_pid for r in rows):
        raise RuntimeError("holder CUDA context no longer visible")
    for row in rows:
        pid = row["pid"]
        if pid == holder_pid:
            continue
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            # NVML may briefly retain a row for a process already gone.
            continue
        if not allow_server or not any(group == p.pid for p in processes):
            raise RuntimeError(f"unexpected process on selected GPU, pid={pid}")
    max_memory = max(max_memory, sum(row["memory_mib"] for row in rows))
    samples += 1


def spawn(argv, log):
    with (OUT / log).open("wb") as handle:
        p = subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    processes.append(p)
    return p


def group_signal(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def stop_group(process):
    group_signal(process, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        group_signal(process, signal.SIGKILL)
        process.wait(timeout=10)
    group_signal(process, signal.SIGKILL)


def interrupt(signum, frame):
    raise InterruptedError(f"received signal {signum}")


signal.signal(signal.SIGTERM, interrupt)
signal.signal(signal.SIGINT, interrupt)
receipt["status"] = "running"
try:
    sample(allow_server=False)
    service = spawn(server, "server.log")
    deadline = time.monotonic() + 420
    while True:
        sample()
        if service.poll() is not None:
            raise RuntimeError(f"startup exit={service.returncode}")
        try:
            suffix = "/readyz"
            with urllib.request.urlopen(f"http://{endpoint}{suffix}", timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("startup")
        time.sleep(1)
    time.sleep(2)
    for cell, spec in cells.items():
        args = spec + f" --seed {seed}"
        summary_path = OUT / f"{cell}.json"
        records_path = OUT / f"{cell}.records.json"
        load = [sys.executable, str(Path(__file__).with_name("load.py")), endpoint, "m", *shlex.split(args),
                "--out", str(summary_path), "--records", str(records_path)]
        load_process = spawn(load, f"{cell}.load.log")
        deadline = time.monotonic() + 180
        while load_process.poll() is None:
            sample()
            if service.poll() is not None:
                raise RuntimeError(f"server exited mid-load: {service.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("load")
            time.sleep(0.65)
        if load_process.returncode or service.poll() is not None:
            raise RuntimeError(f"load exit={load_process.returncode}, server exit={service.returncode}")
        sample()
        expected = workload_arguments(args)
        result = validate_result(summary_path, expected, endpoint)
        records = json.loads(records_path.read_text())
        if [row["request_index"] for row in records] != list(range(expected["warmup"], expected["warmup"] + expected["requests"])):
            raise RuntimeError(f"{cell} incomplete per-request records")
        receipt["cells"][cell] = {"load_argv": load, "summary_sha256": file_sha256(summary_path),
                                  "records_sha256": file_sha256(records_path)}
        print(MODE, cell, result["ok"], round(result["output_tok_s"], 2),
              round(result["ttft_p95_ms"], 2), round(result["tpot_p50_ms"], 2), flush=True)
    rng = random.Random(int(seed) + 100)
    prompts = ["Summarize the following document: " + document(rng, words)
               for words in (32, 250, 450, 1000, 1500, 2300, 3400, 5100)]
    def infer(prompt):
        return complete(endpoint, "m", prompt, limit=32)
    sequential = [infer(prompt) for prompt in prompts]
    with ThreadPoolExecutor(max_workers=8) as workers:
        concurrent = list(workers.map(infer, prompts))
    exact_file = OUT / "exact.json"
    with exact_file.open("x") as handle:
        json.dump({"prompts": prompts, "sequential": sequential, "concurrent": concurrent}, handle)
    receipt["exact_sha256"] = file_sha256(exact_file)
    if sequential != concurrent:
        raise RuntimeError("C1 and C8 exact completion results differ")
    group_signal(service, signal.SIGTERM)
    code = service.wait(timeout=40)
    if code != 0:
        raise RuntimeError(f"server shutdown exit={code}")
    receipt["server_exit"] = code
    shutdown = OUT / "shutdown-metrics.json"
    receipt["shutdown_metrics_sha256"] = file_sha256(shutdown)
    final = json.loads(shutdown.read_bytes())
    if any(final[key] != 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks")):
        raise RuntimeError("shutdown left request or KV state")
    if len(final["allocation"]) != 4 or any(final["allocation"].values()):
        raise RuntimeError("shutdown left native allocation state")
    receipt["status"] = "passed"
except BaseException as error:
    receipt["status"] = "failed"
    receipt["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    for process in reversed(processes):
        stop_group(process)
    receipt.update(nvml_samples=samples, max_process_memory_mib=max_memory, observed_active=seen_holder_active)
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
