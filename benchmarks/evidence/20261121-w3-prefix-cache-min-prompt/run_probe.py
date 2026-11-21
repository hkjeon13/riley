"""Measure one prefix-cache condition with the same A100 diagnostic executable."""

import json
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

from check_distribution_gpu import (
    CHECKPOINT, RILEY, SHUTDOWN, check_config, digest, docker, find_interrupted_container,
    gpu_pids, holder_pids, inspect_container, log_command, require, wait_for_gpu_release, wait_ready,
)
from run_arm import GPU, HOST_USER, MODEL, MODEL_SHA, SERVE, STOCK_IMAGE
from workload import WARM_BLOCKS


ARMS = ("off-first", "plain-first", "filtered-first", "filtered-second", "plain-second", "off-second")
SWITCH = "RILEY_W3_PREFIX_CACHE_MIN_128"
MODE_LOG = "RILEY_W3_PREFIX_CACHE_MIN_PROMPT_TOKENS="


def run(stage, build, workload, arm):
    require(arm in ARMS, "unknown cache condition")
    root = stage.parent.parent
    parent = stage.parent / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
    model = root / "models" / MODEL
    riley = build / "target/release/riley"
    source_archive = build / "src/Cargo.toml"
    require(source_archive.is_file() and (build / "build.exit").read_text().strip() == "0", "diagnostic build is incomplete")
    expected = (build / "build-result.sha256").read_text().splitlines()[0].split()[0]
    require(digest(riley) == expected and digest(model / CHECKPOINT) == MODEL_SHA, "binary or checkpoint differs")
    spec = json.loads(parent.read_bytes())
    require(spec["status"] == "passed" and spec["image_id"] == STOCK_IMAGE, "parent image report differs")
    check_config(json.loads(docker("image", "inspect", STOCK_IMAGE)), STOCK_IMAGE,
                 spec["source_revision"], spec["source_archive_sha256"])
    holders = holder_pids(GPU)
    require(holders and gpu_pids(GPU) == holders, "GPU has an unexpected process")
    mode = arm.split("-")[0]
    switch = "1" if mode == "filtered" else "0"
    argv = [*SERVE, *(["--prefix-cache-blocks", str(WARM_BLOCKS)] if mode != "off" else [])]
    out = stage / arm
    name = f"riley-gpu-check-prefix-min-{uuid4().hex}"
    container = None
    try:
        container = docker("create", "--pull=never", "--name", name, "--label", f"io.riley.gpu-check={name}",
                           "--user", HOST_USER, "--gpus", f"device={GPU}",
                           "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "--env", f"{SWITCH}={switch}",
                           "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}",
                           "--publish", "127.0.0.1::8080",
                           "--mount", f"type=bind,source={riley},target={RILEY},readonly",
                           "--mount", f"type=bind,source={model},target=/models,readonly", STOCK_IMAGE, *argv).strip()
        docker("start", container)
        details = inspect_container(container)
        require(details["Image"] == STOCK_IMAGE and details["Args"] == argv and details["Config"]["User"] == HOST_USER
                and f"{SWITCH}={switch}" in details["Config"]["Env"], "experiment container identity changed")
        require({(item["Source"], item["Destination"], item["RW"]) for item in details["Mounts"]}
                == {(str(riley), RILEY, False), (str(model), "/models", False)}, "unexpected container mounts")
        host_pid = details["State"]["Pid"]
        binding, = details["NetworkSettings"]["Ports"]["8080/tcp"]
        require(binding["HostIp"] == "127.0.0.1", "test address is not isolated")
        port = int(binding["HostPort"])
        ready = wait_ready(container, port, timeout=600)
        observed, _ = docker("exec", container, "/bin/sh", "-ec",
                             f'test "$(readlink /proc/1/exe)" = {RILEY}; sha256sum /proc/1/exe').split()
        require(gpu_pids(GPU) == holders | {host_pid} and observed == expected, "running CUDA executable differs")
        client = log_command(out, "client.log", [sys.executable, "-B", str(workload),
                                                 f"127.0.0.1:{port}", str(out / "workload.json")], 420, check=False)
        require(inspect_container(container)["State"]["Pid"] == host_pid and gpu_pids(GPU) == holders | {host_pid}
                and digest(model / CHECKPOINT) == MODEL_SHA, "runtime or checkpoint changed during measurement")
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
    lines = [line for line in (out / "server.log").read_text().splitlines() if line.startswith(MODE_LOG)]
    limit = 128 if mode == "filtered" else 0
    require(lines == ([] if mode == "off" else [f"{MODE_LOG}{limit}"]), "running cache policy differs")
    (out / "receipt.json").write_text(json.dumps({
        "mode": mode, "min_prompt_tokens": limit, "image_id": STOCK_IMAGE, "binary_sha256": observed,
        "source_revision": spec["source_revision"], "source_archive_sha256": spec["source_archive_sha256"],
        "parent_report_sha256": digest(parent), "build_result_sha256": digest(build / "build-result.sha256"),
        "model_sha256": MODEL_SHA, "gpu_uuid": GPU, "container_user": HOST_USER,
        "server_arguments": argv, "readiness": ready, "client_exit": client.returncode,
        "shutdown_exit": state["ExitCode"], "artifacts_sha256": {
            file: digest(out / file) for file in ("client.log", "workload.json", "shutdown.json", "server.log")
            if (out / file).is_file()
        },
    }, indent=2) + "\n")
    return client.returncode


if __name__ == "__main__":
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run(*(Path(arg).resolve() for arg in sys.argv[1:4]), sys.argv[4]))
