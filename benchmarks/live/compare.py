#!/usr/bin/env python3
"""Run the serving scenarios on Riley and vLLM, alternating engines, on one GPU.

usage: compare.py ROOT LABEL MODEL_DIR GPU RILEY_BIN [--order rvvr] [--scenarios a,b] [--riley-extra "..."]
Writes ROOT/evidence/LABEL/<engine><rep>-<scenario>.json. Every scenario is
declared here before the run; none is dropped afterwards.
"""
import argparse
import json
import os
import shlex
import subprocess
import time
import urllib.request

SHORT, LONG = "short", "long"
# name: (server config, load.py arguments)
SCENARIOS = {
    "closed-c8": (SHORT, "--prompt-words 30 --max-tokens 64 --concurrency 8 --requests 192"),
    "closed-c32": (SHORT, "--prompt-words 30 --max-tokens 64 --concurrency 32 --requests 192"),
    "open-r6": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 6 --requests 192"),
    "open-r12": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 12 --requests 192"),
    "open-r16": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 16 --requests 192"),
    "open-r24": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 24 --requests 192"),
    "open-r32": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 32 --requests 192"),
    "burst-r16": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 16 --burst 1,3 --requests 192"),
    "burst-r6": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 6 --burst 1,3 --requests 192"),
    "mid-c32": (SHORT, "--prompt-words 300 --max-tokens 64 --concurrency 32 --requests 192"),
    "mid-r8": (SHORT, "--prompt-words 300 --max-tokens 64 --rate 8 --requests 192"),
    "mid-r16": (SHORT, "--prompt-words 300 --max-tokens 64 --rate 16 --requests 192"),
    "rag-c1": (LONG, "--prompt-words 1500,3000 --max-tokens 128 --concurrency 1 --requests 16 --warmup 4"),
    "rag-c8": (LONG, "--prompt-words 1500,3000 --max-tokens 128 --concurrency 8 --requests 64 --warmup 8"),
    "rag-r1": (LONG, "--prompt-words 1500,3000 --max-tokens 128 --rate 1 --requests 64 --warmup 8"),
    "rag-r3": (LONG, "--prompt-words 1500,3000 --max-tokens 128 --rate 3 --requests 64 --warmup 8"),
    "reason-c8": (LONG, "--prompt-words 30 --max-tokens 1024 --concurrency 8 --requests 32 --warmup 8"),
    "reason-c32": (LONG, "--prompt-words 30 --max-tokens 1024 --concurrency 32 --requests 64 --warmup 32"),
}
PORT = 18300

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("label")
parser.add_argument("model")
parser.add_argument("gpu")
parser.add_argument("riley_bin")
parser.add_argument("--order", default="rvvr")
parser.add_argument("--scenarios", default=",".join(SCENARIOS))
parser.add_argument("--riley-short", default="")
parser.add_argument("--riley-long", default="")
args = parser.parse_args()
OUT = f"{args.root}/evidence/{args.label}"
os.makedirs(OUT, exist_ok=True)
MODEL = f"{args.root}/models/{args.model}"
ENV = {**os.environ, "PATH": f"{args.root}/tools/vllm-venv/bin:" + os.environ["PATH"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.gpu,
       "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")}


def riley_command(config):
    common = (f"{args.root}/bin/riley-{args.riley_bin} serve --model {MODEL} --model-id m "
              f"--bind 127.0.0.1:{PORT} --max-active-sequences 32 --max-weight-bytes 68719476736 ")
    if config == SHORT:
        return common + "--inference-profile native-bf16-varlen-v1 --max-output-tokens 128 " + args.riley_short
    return common + ("--inference-profile native-bf16-varlen-v1 --max-sequence-tokens 8192 "
                     "--max-output-tokens 2048 " + args.riley_long)


def vllm_command(config):
    length = 1024 if config == SHORT else 8192
    return (f"{args.root}/tools/vllm-venv/bin/vllm serve {MODEL} --served-model-name m --port {PORT} "
            f"--host 127.0.0.1 --max-model-len {length} --gpu-memory-utilization 0.8 "
            f"--generation-config vllm --no-enable-prefix-caching --max-num-seqs 32 --disable-log-stats")


def start(engine, config, log):
    command = riley_command(config) if engine == "r" else vllm_command(config)
    handle = open(log, "w")
    process = subprocess.Popen(shlex.split(command), env=ENV, stdout=handle, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
    path = "/readyz" if engine == "r" else "/health"
    for _ in range(600):
        if process.poll() is not None:
            raise SystemExit(f"{engine} died; see {log}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=2)
            return process
        except OSError:
            time.sleep(1)
    process.kill()
    raise SystemExit(f"{engine} never became ready")


def stop(process):
    process.terminate()
    try:
        process.wait(60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    time.sleep(5)


names = args.scenarios.split(",")
seen = {}
for engine in args.order:
    seen[engine] = seen.get(engine, 0) + 1
    tag = f"{engine}{seen[engine]}"
    for config in (SHORT, LONG):
        selected = [name for name in names if SCENARIOS[name][0] == config]
        if not selected:
            continue
        process = start(engine, config, f"{OUT}/{tag}-{config}.server.log")
        try:
            for name in selected:
                output = f"{OUT}/{tag}-{name}.json"
                subprocess.run(
                    ["python3", f"{args.root}/src/benchmarks/live/load.py", f"127.0.0.1:{PORT}", "m",
                     *SCENARIOS[name][1].split(), "--out", output],
                    check=False, timeout=1800, stdout=subprocess.DEVNULL)
                print(tag, name, "done", flush=True)
        finally:
            stop(process)
print("COMPARE_DONE", flush=True)
