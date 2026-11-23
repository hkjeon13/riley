"""Measure real-model readiness and compare API and streamed token IDs."""
import asyncio
from hashlib import file_digest
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time

from check_distribution_gpu import gpu_pids, holder_pids, require, wait_for_gpu_release
from load import one, prompt

STAGE = Path(__file__).resolve().parent
MODEL = STAGE.parent.parent / "models/Qwen2.5-Coder-14B-Instruct"
GPU = os.environ["CUDA_VISIBLE_DEVICES"]


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def wait_ready(port, process, start):
    while time.monotonic() - start < 480:
        require(process.poll() is None, "Riley exited during startup")
        connection = HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", "/readyz")
            response = connection.getresponse()
            if response.status == 200:
                body = json.loads(response.read())
                require(body["ready"] and body["accepting"], "readiness was not accepting")
                return body, time.monotonic() - start
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(0.25)
    raise TimeoutError("Riley did not become ready")


async def holdout(port):
    rng = random.Random(703)
    prompts = (30, 41, 1400, 1600, 2400, 2600, 2900, 3200)
    bodies = [{
        "model": "m", "temperature": 0, "max_tokens": 64, "stream": True,
        "stream_options": {"include_usage": True}, "return_token_ids": True,
        "messages": [{"role": "user", "content": prompt(rng, length)}],
    } for length in prompts]
    rows = [{} for _ in bodies]
    await asyncio.gather(*(asyncio.wait_for(
        one("127.0.0.1", port, body, time.perf_counter(), row,
            capture_hashes=True, capture_token_ids=True), timeout=90,
    ) for body, row in zip(bodies, rows, strict=True)))
    require(all(row["error"] is False for row in rows), "one streamed request failed")
    return [{key: row[key] for key in (
        "input_sha256", "output_sha256", "prompt_ids_sha256", "token_ids_sha256", "prompt_tokens",
        "completion_tokens", "finish",
    )} for row in rows]


def run(index, engine):
    binary = STAGE / f"riley-{engine}"
    out = STAGE / f"gpu{index}" / engine
    out.mkdir()
    holders = holder_pids(GPU)
    require(len(holders) == 1 and gpu_pids(GPU) == holders, "unexpected process on target GPU")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    shutdown_path = out / "shutdown.json"
    env = {**os.environ, "RILEY_SHUTDOWN_METRICS_PATH": str(shutdown_path)}
    argv = [
        str(binary), "serve", "--model", str(MODEL), "--model-id", "m", "--bind", f"127.0.0.1:{port}",
        "--inference-profile", "native-bf16-varlen-v1", "--max-sequence-tokens", "8192",
        "--max-output-tokens", "256", "--max-active-sequences", "32", "--kv-blocks", "2576",
        "--max-weight-bytes", "68719476736",
    ]
    dump(out / "plan.json", {"args": argv, "gpu": GPU, "affinity": sorted(os.sched_getaffinity(0)),
                              "binary_sha256": digest(binary)})
    with (out / "riley.log").open("xb") as log:
        started = time.monotonic()
        process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            readiness, elapsed = wait_ready(port, process, started)
            dump(out / "ready.json", {"readiness": readiness, "elapsed_s": elapsed})
            require(digest(Path(f"/proc/{process.pid}/exe")) == digest(binary)
                    and gpu_pids(GPU) == holders | {process.pid}, "binary or GPU process changed")
            with (out / "api.log").open("xb") as api_log:
                subprocess.run([sys.executable, "-B", str(STAGE / "api_check.py"),
                                f"127.0.0.1:{port}", "m", "--out", str(out / "api.json")],
                               stdout=api_log, stderr=subprocess.STDOUT, check=True, timeout=120)
            dump(out / "holdout.json", asyncio.run(holdout(port)))
            require(gpu_pids(GPU) == holders | {process.pid}, "GPU process changed during serving checks")
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                exit_code = process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
    require(exit_code == 0, f"Riley did not exit gracefully: {exit_code}")
    shutdown = json.loads(shutdown_path.read_bytes())
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and not any(shutdown["allocation"].values()), "Riley retained resources")
    wait_for_gpu_release(GPU, holders, process.pid)
    dump(out / "receipt.json", {"status": "passed", "readiness_seconds": elapsed, "server_exit": exit_code,
                                "holdout_sha256": digest(out / "holdout.json"), "shutdown_sha256": digest(shutdown_path),
                                "api_sha256": digest(out / "api.json"), "plan_sha256": digest(out / "plan.json")})
    print(f"{index} {engine} {elapsed:.3f} seconds", flush=True)


def interrupt(_signal, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    run(int(sys.argv[1]), sys.argv[2])
