"""Serve the inspected distribution and the same held-GPU vLLM reference."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

import run_profile
from profile_workload import load_cell, save
from check_distribution_gpu import (
    RILEY,
    SHUTDOWN,
    check_config,
    digest,
    docker,
    find_interrupted_container,
    gpu_pids,
    holder_pids,
    inspect_container,
    log_command,
    require,
    wait_for_gpu_release,
    wait_ready,
)

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent.parent
MODEL = ROOT / "models/Qwen2.5-Coder-14B-Instruct"
CELLS = json.loads((STAGE / "cells.json").read_bytes())
OUT = STAGE / f"gpu{os.environ['W3_GPU_INDEX']}"
GPU = os.environ["CUDA_VISIBLE_DEVICES"]
IMAGE = "sha256:3278fe2fced2c3edadb2a9348c46a135fdcfd81da5f72ebfc9a55dede0ccfa42"
CPU = ROOT / "evidence/w4-schema-ids-20261123.ZsUYnqkm/cpu-evidence/report.json"
LEASE = Path(os.environ["RILEY_GPU_HOLDER_LEASE_DIR"]) / f"{OUT.name}.status.json"


def interrupt(_signal, _frame):
    raise KeyboardInterrupt


def check_lease(holders):
    status = json.loads(LEASE.read_bytes())
    require(holder_pids(GPU) == holders == {status["pid"]}
            and status["gpu"] == GPU
            and status["phase"] == "active" and status["allocation_bytes"] == 64 * 2**20
            and time.time_ns() - status["updated_ns"] < 10**10, "GPU lease was lost")


def riley():
    out = OUT / "coder14-riley"
    out.mkdir()
    cpu = json.loads(CPU.read_bytes())
    binary = cpu["binaries"][RILEY]
    require(cpu["status"] == "passed" and cpu["image_id"] == IMAGE
            and cpu["source_revision"] == "c5ec7b3918e18428b0c5704b010aa4ee51972a86", "package receipt changed")
    check_config(json.loads(docker("image", "inspect", IMAGE)), IMAGE,
                 cpu["source_revision"], cpu["source_archive_sha256"])
    holders = holder_pids(GPU)
    check_lease(holders)
    require(gpu_pids(GPU) == holders, "GPU has a foreign process")
    server = [
        "serve", "--model", "/models", "--model-id", "m", "--max-active-sequences", "32",
        "--max-weight-bytes", "68719476736", "--inference-profile", "native-bf16-varlen-v1",
        "--max-sequence-tokens", "8192", "--max-output-tokens", "256", "--kv-blocks", "2576",
        "--batch-token-budget", "1024", "--prefill-chunk-tokens", "1024",
        "--batch-shape-buckets", run_profile.BUCKETS, "--prefill-burst-steady-budget", "256",
        "--prefill-burst-max-decode-rows", "32", "--prefill-burst-min-prompt-tokens", "512",
    ]
    name = f"riley-w3-coder14-{uuid4().hex}"
    user = f"{os.getuid()}:{os.getgid()}"
    command = [
        "create", "--pull=never", "--name", name, "--label", f"io.riley.gpu-check={name}",
        "--user", user, "--gpus", f"device={GPU}", "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}", "--publish", "127.0.0.1::8080",
        "--mount", f"type=bind,source={MODEL},target=/models,readonly", IMAGE, *server,
    ]
    save(out / "plan.json", {"server": server, "docker": command, "cells": CELLS})
    container = None
    try:
        container = docker(*command).strip()
        runtime = inspect_container(container)
        require(runtime["Image"] == IMAGE and runtime["Config"]["User"] == user, "container did not use reviewed image")
        docker("start", container)
        runtime = inspect_container(container)
        (binding,) = runtime["NetworkSettings"]["Ports"]["8080/tcp"]
        port = int(binding["HostPort"])
        require(binding["HostIp"] == "127.0.0.1", "not a loopback test binding")
        save(out / "ready.json", wait_ready(container, port))
        device, = docker("exec", container, "nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader").splitlines()
        require(device.split(",")[0] == GPU, "unexpected container GPU")
        pid = runtime["State"]["Pid"]
        observed, _ = docker("exec", container, "/bin/sh", "-ec",
                             f'test "$(readlink /proc/1/exe)" = {RILEY}; sha256sum /proc/1/exe').split()
        require(observed == binary and pid > 0 and gpu_pids(GPU) == holders | {pid}, "binary or GPU context changed")

        async def measured(spec):
            job = asyncio.create_task(load_cell(f"127.0.0.1:{port}", spec))
            while not job.done():
                check_lease(holders)
                require(gpu_pids(GPU) == holders | {pid}, "GPU process changed during measurement")
                await asyncio.wait({job}, timeout=2)
            return await job

        results = {}
        for label, spec in CELLS.items():
            data = asyncio.run(measured(spec))
            save(out / f"{label}.json", data)
            require(len(data["measured"]) == spec["requests"], f"missing request records in {label}")
            results[label] = digest(out / f"{label}.json")
            print(label, data["statistics"]["ttft_p95_ms"], flush=True)
        check_lease(holders)
        docker("stop", "--time", "20", container)
        state = inspect_container(container)["State"]
        require(not state["Running"] and state["ExitCode"] == 0, "Riley exit was not graceful")
        docker("cp", f"{container}:{SHUTDOWN}", str(out / "shutdown.json"))
        shutdown = json.loads((out / "shutdown.json").read_bytes())
        require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                and shutdown["allocation"] and all(value == 0 for value in shutdown["allocation"].values()),
                "Riley retained allocation or KV state")
        wait_for_gpu_release(GPU, holders, pid)
        save(out / "receipt.json", {"status": "passed", "gpu": GPU, "device": device,
                                   "holder_pid": next(iter(holders)), "image_id": IMAGE, "riley_binary_sha256": observed,
                                   "source_revision": cpu["source_revision"], "cpu_report_sha256": digest(CPU),
                                   "checkpoint_sha256": digest(MODEL / "riley-checkpoint.json"),
                                   "common_kv_blocks": 2576, "kv_cache_bytes": 8103395328, "shutdown_exit": state["ExitCode"],
                                   "results": results})
    finally:
        container = container or find_interrupted_container(name)
        if container is not None:
            try:
                log_command(out, "server.log", ["docker", "logs", container], 25, check=False)
            finally:
                docker("rm", "--force", "--volumes", container)


def vllm():
    run_profile.RUN = OUT
    run_profile.ROOT = ROOT
    run_profile.VLLM = ROOT / "tools/vllm-venv/bin/vllm"
    run_profile.cells = lambda *_: CELLS
    run_profile.run("coder14", "vllm")
    receipt = json.loads((OUT / "coder14-vllm/receipt.json").read_bytes())
    require(receipt["kv_capacity_matches"], "vLLM cache budget did not match")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    {"riley": riley, "vllm": vllm}[sys.argv[1]]()
