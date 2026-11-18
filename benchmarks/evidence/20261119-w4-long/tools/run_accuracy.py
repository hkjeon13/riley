"""Run one accuracy phase under the established A100 reservation controller."""
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from compare_runner import file_sha256
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holder import gpu_processes, identity

mode = sys.argv[1]
if mode not in ("riley", "hf", "vllm"):
    raise SystemExit("unknown phase")
RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
OUT = RUN / mode
OUT.mkdir()
MODEL = ROOT / "models/Qwen2.5-7B-Instruct"
BINARY = ROOT / "evidence/w4-schema-build-20261119.QWnmhslP/target/release/riley"
VPY = ROOT / "tools/vllm-venv/bin/python"
VLLM = ROOT / "tools/vllm-venv/bin/vllm"
TOOL = str(Path(__file__).with_name("long_accuracy.py"))
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
holder_pid = int(os.environ["RILEY_GPU_HOLDER_PID"])
holder_ticks = os.environ["RILEY_GPU_HOLDER_START_TICKS"]
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
env.pop("RILEY_HOST_STEP_TRACE", None)
env.pop("RILEY_TIMING", None)

with socket.socket() as address:
    address.bind(("127.0.0.1", 0))
    port = address.getsockname()[1]
endpoint = f"127.0.0.1:{port}"
kv = shlex.split(subprocess.check_output([sys.executable, str(ROOT / "kvb.py"), str(MODEL)], text=True, timeout=20))
if kv and (len(kv) != 2 or kv[0] != "--kv-blocks" or not kv[1].isdigit()):
    raise RuntimeError("unexpected KV recommendation")
if mode == "riley":
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64:" + env.get("LD_LIBRARY_PATH", "")
    server = [str(BINARY), "serve", "--model", str(MODEL), "--model-id", "m", "--bind", endpoint,
              "--max-active-sequences", "32", "--max-weight-bytes", "68719476736",
              "--inference-profile", "native-bf16-varlen-v1", "--max-sequence-tokens", "8192",
              "--max-output-tokens", "256", *kv]
    client = [sys.executable, "-B", TOOL, "capture", endpoint, "m", str(OUT / "capture.json")]
elif mode == "vllm":
    server = [str(VLLM), "serve", str(MODEL), "--served-model-name", "m", "--host", "127.0.0.1",
              "--port", str(port), "--max-model-len", "8192", "--max-num-seqs", "32",
              "--gpu-memory-utilization", "0.8", "--generation-config", "vllm",
              "--disable-log-stats", "--no-enable-prefix-caching"]
    client = [sys.executable, "-B", TOOL, "vllm", endpoint, "m", str(RUN / "riley/capture.json"), str(OUT / "results.json")]
else:
    server = None
    client = [str(VPY), "-B", TOOL, "hf", str(MODEL), str(RUN / "riley/capture.json"), str(OUT / "reference.json")]

processes = []
max_memory = 0.0
samples = 0
seen_holder_active = False
receipt = {"mode": mode, "gpu_uuid": gpu, "holder_pid": holder_pid, "holder_start_ticks": holder_ticks,
           "server_argv": server, "client_argv": client,
           "sha256": {str(p): file_sha256(p) for p in (BINARY, MODEL / "riley-checkpoint.json",
                                                       MODEL / "generation_config.json", MODEL / "tokenizer.json",
                                                       Path(__file__), Path(TOOL))}}
if mode != "riley":
    receipt["source_capture_sha256"] = file_sha256(RUN / "riley/capture.json")

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
    service = None
    if server is not None:
        service = spawn(server, "server.log")
        deadline = time.monotonic() + 420
        suffix = "/readyz" if mode == "riley" else "/health"
        while True:
            sample()
            if service.poll() is not None:
                raise RuntimeError(f"startup exit={service.returncode}")
            try:
                with urllib.request.urlopen(f"http://{endpoint}{suffix}", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("startup")
            time.sleep(1)
    job = spawn(client, "client.log")
    deadline = time.monotonic() + 1200
    while job.poll() is None:
        sample()
        if service is not None and service.poll() is not None:
            raise RuntimeError(f"server exited during measurement: {service.returncode}")
        if time.monotonic() > deadline:
            raise TimeoutError("accuracy probe")
        time.sleep(0.75)
    if job.returncode != 0:
        raise RuntimeError(f"{mode} client exited {job.returncode}")
    if service is not None:
        group_signal(service, signal.SIGTERM)
        code = service.wait(timeout=60)
        if code not in (0, -signal.SIGTERM):
            raise RuntimeError(f"server shutdown exit={code}")
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
