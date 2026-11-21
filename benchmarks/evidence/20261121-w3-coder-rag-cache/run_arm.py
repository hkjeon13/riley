"""Run an image-bound Coder-14B arm with identical device KV capacity."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

from check_distribution_gpu import (
    CHECKPOINT, RILEY, SHUTDOWN, check_config, digest, docker, find_interrupted_container,
    gpu_pids, holder_pids, inspect_container, log_command, require, wait_for_gpu_release, wait_ready,
)
from workload import WARM_BLOCKS


GPU = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
MODEL_SHA = "82a56cdab6e16e31478b7449be768dce15992b448d543fea4b2f2ed163237d00"
MODEL = "Qwen2.5-Coder-14B-Instruct"
STOCK_IMAGE = "sha256:b731f48cc6ec0820197ec484e98c673e2de592f12ab8448e3fa9f38ca7217c9b"
HOST_USER = f"{os.getuid()}:{os.getgid()}"
SERVE = ["serve", "--model", "/models", "--model-id", "m", "--max-weight-bytes", "68719476736",
         "--inference-profile", "native-bf16-varlen-v1", "--max-active-sequences", "32",
         "--max-sequence-tokens", "8192", "--max-output-tokens", "256", "--kv-blocks", "2576",
         "--batch-token-budget", "1024", "--prefill-chunk-tokens", "1024",
         "--batch-shape-buckets", "1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024",
         "--prefill-burst-steady-budget", "256", "--prefill-burst-max-decode-rows", "32",
         "--prefill-burst-min-prompt-tokens", "512"]


def verify_model(stage, model):
    require(digest(model / CHECKPOINT) == MODEL_SHA, "Coder-14B checkpoint changed")
    log_command(stage, "checkpoint-verify.log",
                ["docker", "run", "--rm", "--pull=never", "--network", "none",
                 "--entrypoint", "/opt/riley/bin/riley-checkpoint", "--user", HOST_USER,
                 "--mount", f"type=bind,source={model},target=/models,readonly",
                 STOCK_IMAGE, "verify", "--checkpoint", "/models"], 600)


def run(stage, arm, cpu_report, model):
    spec = json.loads(cpu_report.read_bytes())
    revision, archive = spec["source_revision"], spec["source_archive_sha256"]
    require(spec["status"] == "passed" and spec["image_id"] == STOCK_IMAGE, "stock CPU report changed")
    check_config(json.loads(docker("image", "inspect", STOCK_IMAGE)), STOCK_IMAGE, revision, archive)
    bin_hash = spec["binaries"][RILEY]
    holders = holder_pids(GPU)
    require(holders and gpu_pids(GPU) == holders, "expected exclusive CUDA reservation in active phase")
    require(digest(model / CHECKPOINT) == MODEL_SHA, "model manifest changed")
    cache = arm.startswith("on-")
    argv = [*SERVE, *(["--prefix-cache-blocks", str(WARM_BLOCKS)] if cache else [])]
    out = stage / arm
    name = f"riley-gpu-check-coder-rag-{uuid4().hex}"
    container = None
    try:
        container = docker("create", "--pull=never", "--name", name, "--label", f"io.riley.gpu-check={name}",
                           "--user", HOST_USER,
                           "--gpus", f"device={GPU}", "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                           "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}", "--publish", "127.0.0.1::8080",
                           "--mount", f"type=bind,source={model},target=/models,readonly", STOCK_IMAGE, *argv).strip()
        docker("start", container)
        details = inspect_container(container)
        require(details["Image"] == STOCK_IMAGE and details["Args"] == argv and details["Config"]["User"] == HOST_USER,
                "container identity, user or argv changed")
        host_pid = details["State"]["Pid"]
        mapping, = details["NetworkSettings"]["Ports"]["8080/tcp"]
        require(mapping["HostIp"] == "127.0.0.1", "expected isolated loopback")
        port = int(mapping["HostPort"])
        ready = wait_ready(container, port, timeout=600)
        require(gpu_pids(GPU) == holders | {host_pid}
                and docker("exec", container, "/bin/sh", "-ec",
                           f'test "$(readlink /proc/1/exe)" = {RILEY}; sha256sum /proc/1/exe').split()[0] == bin_hash,
                "unexpected GPU process or running binary")
        log_command(out, "client.log", [sys.executable, "-B", str(stage / "workload.py"),
                                       f"127.0.0.1:{port}", str(out / "workload.json")], 420)
        require(inspect_container(container)["State"]["Pid"] == host_pid and gpu_pids(GPU) == holders | {host_pid}
                and digest(model / CHECKPOINT) == MODEL_SHA, "runtime or model changed mid-run")
        docker("stop", "--time", "20", container)
        state = inspect_container(container)["State"]
        require(not state["Running"] and state["ExitCode"] == 0, "Riley did not stop gracefully")
        subprocess.run(["docker", "cp", f"{container}:{SHUTDOWN}", str(out / "shutdown.json")], check=True, timeout=30)
        wait_for_gpu_release(GPU, holders, host_pid)
    finally:
        container = container or find_interrupted_container(name)
        if container is not None:
            try:
                log_command(out, "server.log", ["docker", "logs", container], 15, check=False)
            finally:
                docker("rm", "--force", "--volumes", container)
    receipt = {"status": "passed", "source_revision": revision, "source_archive_sha256": archive,
               "image_id": STOCK_IMAGE, "binary_sha256": bin_hash, "cpu_report_sha256": digest(cpu_report),
               "model_sha256": MODEL_SHA, "gpu_uuid": GPU, "gpu_holder": os.environ["RILEY_GPU_HOLDER_PID"],
               "server_arguments": argv, "container_user": HOST_USER, "readiness": ready, "shutdown_exit": state["ExitCode"],
               "artifacts_sha256": {file: digest(out / file) for file in (
                   "client.log", "workload.json", "shutdown.json", "server.log",
               )}}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    stage = Path(sys.argv[1]).resolve()
    model = stage.parent.parent / "models" / MODEL
    if sys.argv[2] == "verify":
        verify_model(stage, model)
    else:
        cpu = stage.parent / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
        run(stage, sys.argv[2], cpu, model)
