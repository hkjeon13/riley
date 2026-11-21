"""Run one dedicated mixed soak arm under the A100 reservation controller."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from compare_runner import file_sha256
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holder import gpu_processes, identity

MODES = {"immediate-v1": (45, 128), "after-submit-v1": (45, 128)}
if len(sys.argv) != 2 or sys.argv[1] not in MODES:
    raise SystemExit("specify one of " + ", ".join(MODES))
mode = sys.argv[1]
seconds, prefix_blocks = MODES[mode]
RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
OUT = RUN / mode
OUT.mkdir()
MODEL = ROOT / "models/Qwen2.5-7B-Instruct"
BINARY = RUN / "bin/riley"
TOOL = Path(__file__).with_name("soak.py")
SHUTDOWN = OUT / "shutdown-metrics.json"
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
holder_pid = int(os.environ["RILEY_GPU_HOLDER_PID"])
holder_ticks = os.environ["RILEY_GPU_HOLDER_START_TICKS"]
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
       "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64",
       "RILEY_SHUTDOWN_METRICS_PATH": str(SHUTDOWN)}
env.pop("RILEY_HOST_STEP_TRACE", None)
env.pop("RILEY_TIMING", None)

with socket.socket() as address:
    address.bind(("127.0.0.1", 0))
    port = address.getsockname()[1]
endpoint = f"127.0.0.1:{port}"
server = [
    str(BINARY), "serve", "--model", str(MODEL), "--model-id", "m", "--bind", endpoint,
    "--max-active-sequences", "32", "--max-weight-bytes", "68719476736",
    "--inference-profile", "native-bf16-varlen-v1", "--max-sequence-tokens", "8192",
    "--max-output-tokens", "512", "--kv-blocks", "16384", "--event-publication", mode,
]
if prefix_blocks:
    server.extend(("--prefix-cache-blocks", str(prefix_blocks)))

processes = []
max_memory = 0.0
samples = 0
seen_holder_active = False
receipt = {
    "mode": mode, "gpu_uuid": gpu, "holder_pid": holder_pid, "holder_start_ticks": holder_ticks,
    "server_argv": server, "duration_seconds": seconds, "prefix_cache_blocks": prefix_blocks,
    "riley_source": "d3d8fe6e5b1c528bff51c0008cdc24544c99f862",
    "sha256": {str(path): file_sha256(path) for path in (
        BINARY, MODEL / "riley-checkpoint.json", MODEL / "tokenizer.json",
        Path(__file__), TOOL, TOOL.with_name("load.py"),
    )},
}
if receipt["sha256"][str(BINARY)] != (RUN / "candidate.sha256").read_text().split()[0]:
    raise RuntimeError("Riley binary differs from the W2 candidate")


def sample(allow_server=True):
    global max_memory, samples, seen_holder_active
    if identity(holder_pid) != holder_ticks:
        raise RuntimeError("holder identity changed")
    status = json.loads((Path(os.environ["RILEY_GPU_HOLDER_LEASE_DIR"]) / f"gpu{os.environ['W5_GPU_INDEX']}.status.json").read_text())
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
    deadline = time.monotonic() + 180
    while True:
        sample()
        if service.poll() is not None:
            raise RuntimeError(f"startup failed ({service.returncode})")
        try:
            with urllib.request.urlopen(f"http://{endpoint}/readyz", timeout=1) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("server readiness")
        time.sleep(1)
    api_report = OUT / "api.json"
    with (OUT / "api.log").open("wb") as log:
        subprocess.run([sys.executable, "-B", str(TOOL.with_name("api_check.py")), endpoint, "m",
                        "--tools", "--out", str(api_report)], env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=120)
    records = json.loads(api_report.read_bytes())["results"]
    if len(records) != 40 or any(record["passed"] is not True for record in records):
        raise RuntimeError("API report did not pass all 40 checks")
    receipt["api_sha256"] = file_sha256(api_report)
    client = [
        sys.executable, "-B", str(TOOL), endpoint, "m", "--pid", str(service.pid),
        "--seconds", str(seconds), "--rate", "8", "--shared-prefix",
        "--out", str(OUT / "soak.json"),
    ]
    client.append("--smoke")
    receipt["client_argv"] = client
    job = spawn(client, "client.log")
    deadline = time.monotonic() + seconds + 150
    while job.poll() is None:
        sample()
        if service.poll() is not None:
            raise RuntimeError(f"server exited during soak ({service.returncode})")
        if time.monotonic() > deadline:
            raise TimeoutError("soak client")
        time.sleep(2)
    receipt["client_exit"] = job.returncode
    result_file = OUT / "soak.json"
    if result_file.is_file():
        receipt["soak_sha256"] = file_sha256(result_file)
    group_signal(service, signal.SIGTERM)
    receipt["server_exit"] = service.wait(timeout=60)
    if receipt["server_exit"] != 0:
        raise RuntimeError("server shutdown failed")
    receipt["shutdown_metrics_sha256"] = file_sha256(SHUTDOWN)
    shutdown_metrics = json.loads(SHUTDOWN.read_bytes())
    if any(shutdown_metrics[field] != 0 for field in ("active_requests", "waiting_requests", "kv_allocated_blocks")):
        raise RuntimeError("shutdown left request or KV state")
    if not shutdown_metrics["allocation"] or any(value != 0 for value in shutdown_metrics["allocation"].values()):
        raise RuntimeError("shutdown left native allocations")
    if job.returncode != 0 or json.loads(result_file.read_bytes())["status"] != "passed":
        raise RuntimeError("soak workload checks failed")
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
