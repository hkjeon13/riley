"""Run the frozen synthetic workflow against the installed vLLM on a held A100."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--command-only", action="store_true", help="write the launch arguments without starting a model")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir()
    root = Path("/mnt/efs/nlp/psyche/riley-gap-20261102")
    executable = root / "tools/vllm-venv/bin/vllm"
    model = (
        root / "models" / os.environ.get("AGENT_DECODE_MODEL", "Qwen2.5-7B-Instruct")
    )
    config = json.loads((model / "config.json").read_text())
    head_dim = (
        config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    )
    block_bytes = (
        2
        * config["num_key_value_heads"]
        * head_dim
        * config["num_hidden_layers"]
        * 2
        * 16
    )
    if not args.command_only and not (
        os.environ.get("RILEY_GPU_HOLDER_LEASE_DIR") and os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        raise RuntimeError("expected a leased Riley GPU")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        str(executable),
        "serve",
        str(model),
        "--served-model-name",
        "m",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "8",
        "--dtype",
        "bfloat16",
        "--block-size",
        "16",
        "--kv-cache-memory-bytes",
        str(4096 * block_bytes),
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--exclude-tools-when-tool-choice-none",
    ]
    if reservation := os.environ.get("RILEY_TRIAL_VLLM_RESERVATION"):
        command.extend(["--gpu-memory-utilization", reservation])
    if os.environ.get("AGENT_DECODE_MODEL"):
        command.append("--enable-prompt-tokens-details")
    if os.environ.get("FINAL_LAYOUT") == "keep":
        command.remove("--exclude-tools-when-tool-choice-none")
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if args.command_only:
        return
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        PATH=str(executable.parent) + ":" + env["PATH"],
    )
    version = subprocess.check_output(
        [
            str(executable.parent / "python"),
            "-c",
            "from importlib.metadata import version; print(version('vllm'))",
        ],
        env=env,
    )
    (out / "version.txt").write_bytes(version)
    with (out / "server.log").open("xb") as log:
        server = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            address = f"127.0.0.1:{port}"
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline and server.poll() is None:
                try:
                    with urllib.request.urlopen(
                        f"http://{address}/v1/models", timeout=2
                    ) as response:
                        ready = json.load(response)
                    if any(m["id"] == "m" for m in ready["data"]):
                        (out / "ready.json").write_text(json.dumps(ready) + "\n")
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"reference unavailable, process status {server.poll()}"
                )
            if os.environ.get("CODER_CATALOG_COMPARE"):
                subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("coder_catalog_compare.py")), address, str(out)], check=True)
                return
            if os.environ.get("AGENT_DECODE_MODEL"):
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("model_decode_workload.py")),
                        address,
                        str(out),
                        os.environ["FROZEN_INPUT_DIR"],
                    ],
                    check=True,
                )
                return
            if os.environ.get("FINAL_LAYOUT"):
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("final_layout_trial.py")),
                        address,
                        str(out / "workflows.json"),
                        "vllm",
                    ],
                    check=True,
                )
                return
            for history in (0, 200):
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("agent_workflow.py")),
                        address,
                        "m",
                        "--out",
                        str(out / f"workflows-{history}.json"),
                        "--seed",
                        str(args.seed),
                        "--history-entries",
                        str(history),
                    ],
                    check=True,
                )
            if os.environ.get("FROZEN_INPUT_DIR"):
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("fixed_agent_decode.py")),
                        address,
                        os.environ["FROZEN_INPUT_DIR"],
                        str(out / "fixed.json"),
                    ],
                    check=True,
                )
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    main()
