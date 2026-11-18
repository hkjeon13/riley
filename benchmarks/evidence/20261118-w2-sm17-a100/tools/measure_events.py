"""Run a bounded SmolLM2 reason-c32 arm under the existing GPU controller."""
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

from compare_runner import file_sha256, validate_result, workload_arguments
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holder import gpu_processes, identity

MODE = sys.argv[1]
if MODE not in ("paired-off1", "event1", "event2", "paired-off2"):
    raise SystemExit("unknown arm")
ROOT = Path(__file__).resolve().parents[2].parent
RUN = Path(__file__).resolve().parents[1]
OUT = RUN / MODE
MODEL = ROOT / "models/SmolLM2-1.7B-Instruct"
RILEY = ROOT / "bin/riley-lp16"
VLLM = ROOT / "tools/vllm-venv/bin/vllm"
NSYS = ROOT / "tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys"

gpu = os.environ["CUDA_VISIBLE_DEVICES"]
holder_pid = int(os.environ["RILEY_GPU_HOLDER_PID"])
holder_ticks = os.environ["RILEY_GPU_HOLDER_START_TICKS"]
env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1"}
env.pop("RILEY_HOST_STEP_TRACE", None)
if MODE == "host":
    env["RILEY_HOST_STEP_TRACE"] = "1"

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
reference = MODE in ("v1", "v2")
if reference:
    server = [str(VLLM), "serve", str(MODEL), "--served-model-name", "m", "--port", str(port),
              "--host", "127.0.0.1", "--max-model-len", "8192", "--gpu-memory-utilization", "0.8",
              "--generation-config", "vllm", "--max-num-seqs", "32", "--disable-log-stats",
              "--tensor-parallel-size", "1", "--no-enable-prefix-caching"]
else:
    server = [str(RILEY), "serve", "--model", str(MODEL), "--model-id", "m", "--bind", endpoint,
              "--max-active-sequences", "32", "--max-weight-bytes", "68719476736",
              "--inference-profile", "native-bf16-varlen-v1", "--max-sequence-tokens", "8192",
              "--max-output-tokens", "2048", *kv]
if MODE.startswith("event"):
    server.extend(["--event-publication", "after-submit-v1"])
if MODE == "nsys":
    server = [str(NSYS), "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none",
              "--cpuctxsw=none", "-o", str(OUT / "trace"), *server]

args = "--prompt-words 30 --max-tokens 1024 --concurrency 32"
args += " --requests 32 --warmup 0" if MODE == "nsys" else " --requests 64 --warmup 32"
load = [sys.executable, str(Path(__file__).with_name("load.py")), endpoint, "m", *shlex.split(args),
        "--out", str(OUT / "load.json")]

processes = []
max_memory = 0.0
samples = 0
seen_holder_active = False
receipt = {"mode": MODE, "gpu_uuid": gpu, "holder_pid": holder_pid, "holder_start_ticks": holder_ticks,
           "server_argv": server, "load_argv": load, "host_trace": env.get("RILEY_HOST_STEP_TRACE"),
           "sha256": {str(p): file_sha256(p) for p in (RILEY, VLLM, MODEL / "riley-checkpoint.json",
                                                       Path(__file__), Path(__file__).with_name("load.py"))}}


def sample(allow_server=True):
    global max_memory, samples, seen_holder_active
    if identity(holder_pid) != holder_ticks:
        raise RuntimeError("holder identity changed")
    status = json.loads((Path(os.environ["RILEY_GPU_HOLDER_LEASE_DIR"]) / "gpu7.status.json").read_text())
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


def graceful_stop(service):
    if MODE != "nsys":
        group_signal(service, signal.SIGTERM)
    else:
        owned = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            pid = int(proc.name)
            try:
                if os.getpgid(pid) == service.pid and (proc / "exe").resolve(strict=True) == RILEY.resolve():
                    owned.append(pid)
            except (ProcessLookupError, PermissionError, FileNotFoundError):
                continue
        if len(owned) != 1:
            raise RuntimeError(f"expected one profiled Riley child, got {owned}")
        os.kill(owned[0], signal.SIGINT)
    code = service.wait(timeout=100 if MODE == "nsys" else 40)
    if code not in ((0,) if MODE == "nsys" else (0, -signal.SIGTERM)):
        raise RuntimeError(f"server shutdown exit={code}")
    if MODE == "nsys" and not (OUT / "trace.nsys-rep").stat().st_size:
        raise RuntimeError("empty Nsight report")
    return code


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
            suffix = "/health" if reference else "/readyz"
            with urllib.request.urlopen(f"http://{endpoint}{suffix}", timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("startup")
        time.sleep(1)
    time.sleep(2)
    load_process = spawn(load, "load.log")
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
    result = validate_result(OUT / "load.json", workload_arguments(args), endpoint)
    receipt["load_sha256"] = file_sha256(OUT / "load.json")
    receipt["server_exit"] = graceful_stop(service)
    receipt["status"] = "passed"
    print(MODE, result["ok"], result["output_tok_s"], result["tpot_p50_ms"], flush=True)
except BaseException as error:
    receipt["status"] = "failed"
    receipt["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    for process in reversed(processes):
        stop_group(process)
    receipt.update(nvml_samples=samples, max_process_memory_mib=max_memory, observed_active=seen_holder_active)
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
