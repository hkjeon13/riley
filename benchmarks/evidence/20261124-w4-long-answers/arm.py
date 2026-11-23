"""Capture short answers with the packaged Riley or pinned vLLM on GPU 7."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from uuid import uuid4

from check_distribution_gpu import (
    RILEY, SHUTDOWN, check_config, digest, docker, find_interrupted_container, gpu_pids,
    holder_pids, inspect_container, log_command, require, wait_for_gpu_release, wait_ready,
)
from long_accuracy import save

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent.parent
MODEL = ROOT / "evidence/w1-dist-d3d8fe6e5.YnZWdh5l/model"
CPU = ROOT / "evidence/w5-shard-package-20261124.nhBcrYgP/cpu-evidence/report.json"
SOURCE = "0511b9280f62c3ee87c7a1344444423d8b4623c4"
IMAGE = "sha256:1729f15a677422d552b9a3fd2352dfeb08dd4cccd3558680a016058d27191098"
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
VLLM = ROOT / "tools/vllm-venv/bin/vllm"


def interrupt(_signal, _frame):
    raise KeyboardInterrupt


def run_client(mode, address, fixture_hash, out, verify_active):
    command = [sys.executable, "-B", str(STAGE / "long_answer_check.py"), "capture", mode, address,
               "m", str(out / "capture.json"), str(STAGE / "facts.json.gz"), fixture_hash]
    deadline = time.monotonic() + 900
    with (out / "client.log").open("xb") as log, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as client:
        try:
            while client.poll() is None:
                verify_active()
                if time.monotonic() >= deadline:
                    raise TimeoutError("long-answer capture")
                time.sleep(1)
            require(client.returncode == 0, f"{mode} answer capture failed")
        finally:
            if client.poll() is None:
                client.kill()
                client.wait(timeout=10)


def run_riley(gpu, holders, out, fixture_hash):
    cpu = json.loads(CPU.read_bytes())
    require(cpu["status"] == "passed" and cpu["image_id"] == IMAGE and cpu["source_revision"] == SOURCE,
            "package identity changed")
    check_config(json.loads(docker("image", "inspect", IMAGE)), IMAGE, SOURCE, cpu["source_archive_sha256"])
    name = f"riley-w4-answers-{uuid4().hex}"
    args = ["serve", "--model", "/models", "--model-id", "m", "--inference-profile", "native-bf16-varlen-v1",
            "--max-active-sequences", "8", "--max-sequence-tokens", "8192", "--max-output-tokens", "32",
            "--kv-blocks", "4096"]
    command = ["create", "--pull=never", "--name", name, "--label", f"io.riley.gpu-check={name}",
               "--gpus", f"device={gpu}", "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
               "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}", "--publish", "127.0.0.1::8080",
               "--mount", f"type=bind,source={MODEL},target=/models,readonly", IMAGE, *args]
    container = None
    try:
        container = docker(*command).strip()
        docker("start", container)
        runtime = inspect_container(container)
        (binding,) = runtime["NetworkSettings"]["Ports"]["8080/tcp"]
        port, pid = int(binding["HostPort"]), runtime["State"]["Pid"]
        require(runtime["Image"] == IMAGE and runtime["Config"]["User"] == "65532:65532"
                and binding["HostIp"] == "127.0.0.1" and pid > 0, "unexpected runtime identity")
        save(out / "ready.json", wait_ready(container, port))
        observed, _ = docker("exec", container, "/bin/sh", "-ec",
                             f'test "$(readlink /proc/1/exe)" = {RILEY}; sha256sum /proc/1/exe').split()
        require(observed == cpu["binaries"][RILEY], "Riley binary differs from the package")

        def verify_active():
            require(gpu_pids(gpu) == holders | {pid}, "Riley compute context or holder changed")

        verify_active()
        log_command(out, "checkpoint.log", ["docker", "exec", container, "/opt/riley/bin/riley-checkpoint",
                                            "verify", "--checkpoint", "/models"], 600)
        run_client("riley", f"127.0.0.1:{port}", fixture_hash, out, verify_active)
        verify_active()
        docker("stop", "--time", "20", container)
        state = inspect_container(container)["State"]
        require(not state["Running"] and state["ExitCode"] == 0, "Riley did not shut down normally")
        docker("cp", f"{container}:{SHUTDOWN}", str(out / "shutdown.json"))
        metrics = json.loads((out / "shutdown.json").read_bytes())
        require(all(metrics[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                and metrics["allocation"] and not any(metrics["allocation"].values()), "Riley retained allocations")
        wait_for_gpu_release(gpu, holders, pid)
        return {"image_id": IMAGE, "source_revision": SOURCE, "binary_sha256": observed, "cpu_report_sha256": digest(CPU),
                "server": command, "shutdown_exit": state["ExitCode"]}
    finally:
        container = container or find_interrupted_container(name)
        if container is not None:
            try:
                log_command(out, "server.log", ["docker", "logs", container], 25, check=False)
            finally:
                docker("rm", "--force", "--volumes", container)


def run_vllm(gpu, holders, out, fixture_hash):
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    command = [str(VLLM), "serve", str(MODEL), "--served-model-name", "m", "--host", "127.0.0.1",
               "--port", str(port), "--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "8",
               "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.80", "--generation-config", "vllm",
               "--disable-log-stats", "--no-enable-prefix-caching"]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "VLLM_CACHE_ROOT": str(out / "vllm-cache"), "TORCHINDUCTOR_CACHE_DIR": str(out / "torch-cache")}
    with (out / "server.log").open("xb") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

        def verify_active(require_compute=True):
            current = gpu_pids(gpu)
            extra = current - holders
            require(process.poll() is None and current >= holders
                    and (extra or not require_compute)
                    and all(os.getpgid(pid) == process.pid for pid in extra),
                    "vLLM exited or an unexpected compute process appeared")

        try:
            deadline = time.monotonic() + 480
            endpoint = f"127.0.0.1:{port}"
            while time.monotonic() < deadline:
                verify_active(require_compute=False)
                try:
                    with urlopen(f"http://{endpoint}/health", timeout=2) as health:
                        if health.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("vLLM readiness")
            verify_active()
            run_client("vllm", endpoint, fixture_hash, out, verify_active)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    deadline = time.monotonic() + 10
    while gpu_pids(gpu) != holders and time.monotonic() < deadline:
        time.sleep(0.25)
    require(gpu_pids(gpu) == holders and process.returncode in (0, -signal.SIGTERM), "vLLM shutdown was incomplete")
    return {"vllm": "0.27.1+cu129", "launcher_sha256": digest(VLLM), "server": command, "shutdown_exit": process.returncode}


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    mode, fixture_hash = sys.argv[1:]
    out = STAGE / mode
    out.mkdir()
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    require(gpu == "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0", "expected leased GPU 7")
    holders = holder_pids(gpu)
    require(len(holders) == 1 and gpu_pids(gpu) == holders, "GPU did not start with its holder")
    manifest = MODEL / "riley-checkpoint.json"
    require(digest(manifest) == CHECKPOINT, "checkpoint manifest changed")
    receipt = {"riley": run_riley, "vllm": run_vllm}[mode](gpu, holders, out, fixture_hash)
    require(digest(manifest) == CHECKPOINT, "checkpoint manifest changed during capture")
    save(out / "receipt.json", {**receipt, "gpu": gpu, "holder_pid": next(iter(holders)),
                               "checkpoint_sha256": CHECKPOINT, "fixture_sha256": fixture_hash,
                               "capture_sha256": digest(out / "capture.json")})
