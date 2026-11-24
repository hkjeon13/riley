"""Launch one leased, same-host vLLM replica with the registered 14B settings."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def run(index):
    if not os.environ.get("RILEY_GPU_HOLDER_LEASE_DIR") or not os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        raise RuntimeError("requires the reserved physical GPU")
    folder = Path(f"gpu{index}")
    folder.mkdir()
    base = Path("/mnt/efs/nlp/psyche/riley-gap-20261102")
    tool = base / "tools/vllm-venv/bin"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = str(probe.getsockname()[1])
    address = "127.0.0.1:" + port
    cmd = json.loads(Path("baseline-cli.json").read_text())
    if cmd[0] != str(tool / "vllm") or cmd[2] != str(
        base / "models/Qwen2.5-Coder-14B-Instruct"
    ):
        raise ValueError("unexpected executable or model")
    cmd[cmd.index("--port") + 1] = port
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        PATH=str(tool) + ":" + env["PATH"],
    )
    ver = subprocess.check_output(
        [
            tool / "python",
            "-c",
            "from importlib.metadata import version; print(version('vllm'))",
        ],
        env=env,
        text=True,
    ).strip()
    if ver != "0.27.1+cu129":
        raise ValueError("reference version changed")
    (folder / "version.txt").write_text(ver + "\n")
    (folder / "args.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with (folder / "server.log").open("wb") as log:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 720
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    with urllib.request.urlopen(
                        "http://" + address + "/v1/models", timeout=2
                    ) as rsp:
                        ready = json.load(rsp)
                    if any(m["id"] == "m" for m in ready["data"]):
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(f"startup failed, exit code {proc.poll()}")
            (folder / "ready.json").write_text(json.dumps(ready) + "\n")
            (folder / "address.tmp").write_text(address + "\n")
            (folder / "address.tmp").rename(folder / "address")
            deadline = time.monotonic() + 600
            while not Path("finished").exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("reference server or client did not finish")
                time.sleep(1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            (folder / "exit-code").write_text(str(proc.returncode) + "\n")


if __name__ == "__main__":
    run(int(sys.argv[1]))
