"""Exercise the cache metrics in one isolated distribution container."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

from check_distribution_gpu import (
    CHECKPOINT, RILEY, SHUTDOWN, check_config, digest, docker, find_interrupted_container, gpu_pids,
    holder_pids, inspect_container, log_command, require, wait_for_gpu_release, wait_ready,
)


GPU = "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7"
MODEL_SHA = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
IMAGE = "sha256:b731f48cc6ec0820197ec484e98c673e2de592f12ab8448e3fa9f38ca7217c9b"
MODES = {"stock": 128, "cache-off": 0, "cache-on": 128}
SERVER = ["serve", "--model", "/models", "--model-id", "m",
          "--inference-profile", "native-bf16-varlen-v1", "--max-active-sequences", "2",
          "--max-waiting-requests", "1", "--max-sequence-tokens", "8192",
          "--max-output-tokens", "512", "--kv-blocks", "4096"]


def run(stage, build, mode):
    capacity = MODES[mode]
    evidence = stage.parent
    model = evidence / "w1-dist-d3d8fe6e5.YnZWdh5l/model"
    cpu = evidence / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
    spec = json.loads(cpu.read_bytes())
    require(spec["status"] == "passed" and spec["image_id"] == IMAGE, "stock image provenance changed")
    check_config(json.loads(docker("image", "inspect", IMAGE)), IMAGE,
                 spec["source_revision"], spec["source_archive_sha256"])
    require(digest(model / CHECKPOINT) == MODEL_SHA, "checkpoint manifest changed")
    mnt = [f"type=bind,source={model},target=/models,readonly"]
    expected = spec["binaries"][RILEY]
    if mode != "stock":
        binary = build / "target/release/riley"
        expected = (build / "build-result.sha256").read_text().splitlines()[0].split()[0]
        require(digest(binary) == expected, "built service binary changed")
        mnt.append(f"type=bind,source={binary},target={RILEY},readonly")

    # The rebuilt binary is private to the host account and mounted read-only.
    # Both arms use the same non-root UID and GID.
    user = f"{os.getuid()}:{os.getgid()}"
    if user.startswith("0:"):
        raise ValueError("expected the dedicated non-root host account")
    holders = holder_pids(GPU)
    require(holders and gpu_pids(GPU) == holders, "target GPU contains an unexpected process")
    argv = [*SERVER, *(["--prefix-cache-blocks", str(capacity)] if capacity else [])]
    out = stage / mode
    name = f"riley-gpu-check-prefix-metrics-{uuid4().hex}"
    container = None
    try:
        mount_args = tuple(arg for mount in mnt for arg in ("--mount", mount))
        container = docker("create", "--pull=never", "--name", name, "--label", f"io.riley.gpu-check={name}",
                           "--user", user, "--gpus", f"device={GPU}", "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                           "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}", "--publish", "127.0.0.1::8080",
                           *mount_args, IMAGE, *argv).strip()
        docker("start", container)
        detail = inspect_container(container)
        require(detail["Image"] == IMAGE and detail["Args"] == argv and detail["Config"]["User"] == user
                and all(item["RW"] is False for item in detail["Mounts"]), "unexpected container configuration")
        host_pid = detail["State"]["Pid"]
        binding, = detail["NetworkSettings"]["Ports"]["8080/tcp"]
        require(binding["HostIp"] == "127.0.0.1", "cache probe address is not isolated")
        port = int(binding["HostPort"])
        ready = wait_ready(container, port)
        actual, _ = docker("exec", container, "/bin/sh", "-ec",
                           f'test "$(readlink /proc/1/exe)" = {RILEY}; sha256sum /proc/1/exe').split()
        require(gpu_pids(GPU) == holders | {host_pid} and actual == expected, "running CUDA binary differs")
        command = [sys.executable, "-B", str(stage / "operational_check.py"), f"127.0.0.1:{port}", "m",
                   "--prefix-cache-blocks", str(capacity), "--out", str(out / "operational.json")]
        if mode != "stock":
            command.append("--require-prefix-cache-metrics")
        client = log_command(out, "client.log", command, 180, check=False)
        require(inspect_container(container)["State"]["Pid"] == host_pid and gpu_pids(GPU) == holders | {host_pid}
                and digest(model / CHECKPOINT) == MODEL_SHA, "runtime or checkpoint changed during cache probe")
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
    (out / "receipt.json").write_text(json.dumps({
        "mode": mode, "image_id": IMAGE, "source_revision": spec["source_revision"],
        "source_archive_sha256": spec["source_archive_sha256"], "binary_sha256": actual,
        "cpu_report_sha256": digest(cpu), "checkpoint_sha256": MODEL_SHA, "gpu_uuid": GPU,
        "container_user": user, "server_arguments": argv, "ready": ready,
        "client_exit": client.returncode, "shutdown_exit": state["ExitCode"],
        "artifacts_sha256": {file: digest(out / file) for file in
                            ("client.log", "operational.json", "shutdown.json", "server.log") if (out / file).is_file()},
    }, indent=2) + "\n")
    return client.returncode


if __name__ == "__main__":
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]))
