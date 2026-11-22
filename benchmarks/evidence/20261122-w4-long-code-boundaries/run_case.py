"""Capture two BF16 row traces while replaying the exact archived long-code response."""

from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from holder import gpu_processes, identity
from long_accuracy import complete, read_capture, read_workload


ATTENTION = "w4-o31-serving-20261120.7W3nr5JH"
DOWN = "w4-fp32-down-serving-20261120.Saj3XXSY"
MODES = {"baseline": (DOWN, "baseline", None, 262), "attention31": (ATTENTION, "candidate", "RILEY_W4_ATTENTION_O31", 16019),
         "fp32_down21": (DOWN, "candidate", "RILEY_W4_DOWN_FP32", 16019)}
GPU = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
PREFIX = [576, 2701, 2038, 1447, 73594, 12669, 198]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as data:
        return file_digest(data, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def run(stage, build, mode):
    evidence = stage.parent
    archived, arm, switch, expected_id = MODES[mode]
    archive = evidence / archived
    fixture = build / "fixtures/20261119-w4-holdout/holdout.json.gz"
    workload = read_workload(fixture)
    old = read_capture(archive / arm / "rows/holdout.json", workload)
    index, = [i for i, row in enumerate(workload.cases) if str(row.label) == "long-code-7836"]
    case, expected = workload.cases[index], old["baseline"][index]
    ids = expected["prompt_tokens"]
    require(len(ids) == 7836 and ids[-1] == 25 and expected["tokens"][:8] == PREFIX + [expected_id],
            "expected long-code token row differs")

    holder_pid = int(os.environ["RILEY_GPU_HOLDER_PID"])
    holder_ticks = os.environ["RILEY_GPU_HOLDER_START_TICKS"]
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    require(gpu == GPU and identity(holder_pid) == holder_ticks, "wrong active GPU holder")
    require({row["pid"] for row in gpu_processes(gpu)} == {holder_pid}, "GPU already has another process")
    binary = build / "target/release/riley"
    argv = read(archive / arm / "receipt.json")["server_argv"]
    argv[0] = str(binary)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{probe.getsockname()[1]}"
    argv[argv.index("--bind") + 1] = address
    model = Path(argv[argv.index("--model") + 1])
    hash_binary = digest(binary)
    require(hash_binary == (build / "build-result.sha256").read_text().split()[0]
            and digest(model / "riley-checkpoint.json") == CHECKPOINT, "binary or model drift")
    out = stage / mode
    trace = out / "trace"
    for label in ("prompt-final", "decode-7"):
        (trace / label).mkdir(parents=True)
    env = dict(os.environ)
    for name in ("RILEY_W4_ATTENTION_O31", "RILEY_W4_DOWN_FP32", "RILEY_HOST_STEP_TRACE", "RILEY_TIMING"):
        env.pop(name, None)
    env["RILEY_W4_TRACE_DIR"] = str(trace)
    env["RILEY_SHUTDOWN_METRICS_PATH"] = str(out / "shutdown.json")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/lib64:" + env.get("LD_LIBRARY_PATH", "")
    if switch:
        env[switch] = "1"

    with (out / "server.log").open("xb") as log:
        server = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                require(server.poll() is None, "server exited during startup")
                try:
                    with urlopen(f"http://{address}/readyz", timeout=1) as response:
                        ready = json.load(response)
                    if ready["ready"] and ready["accepting"]:
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("startup")
            require({row["pid"] for row in gpu_processes(gpu)} == {holder_pid, server.pid}
                    and digest(Path(f"/proc/{server.pid}/exe")) == hash_binary, "active runtime differs")
            response = complete(address, "m", case.prompt, diagnostic=True, expected=ids, limit=case.limit)
            require(all(response[key] == expected[key] for key in ("tokens", "text", "finish", "usage", "prompt_tokens")),
                    "trace did not preserve the archived serving response")
            server.send_signal(signal.SIGTERM)
            require(server.wait(timeout=60) == 0, "graceful shutdown failed")
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
    require(identity(holder_pid) == holder_ticks, "holder changed during trace")
    receipt = {"mode": mode, "gpu": gpu, "binary_sha256": hash_binary, "model_sha256": CHECKPOINT,
               "historical_output_sha256": digest(archive / arm / "rows/holdout.json"),
               "server_argv": argv, "server_exit": server.returncode, "switch": switch, "ready": ready,
               "prompt_tokens": len(ids), "first_eight": response["tokens"][:8], "exact_archived_output": True,
               "response_sha256": sha256(json.dumps({key: response[key] for key in ("tokens","text","finish","usage")},
                                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
