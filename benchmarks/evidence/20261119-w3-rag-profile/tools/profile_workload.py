"""Run fixed W3 request cohorts and anchor optional Nsight windows."""
import argparse
import asyncio
import ctypes
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import urllib.request

from load import one, prompt, summarize


def save(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def nvtx_clock():
    library = ctypes.CDLL("libnvToolsExt.so.1")
    mark = library.nvtxMarkA
    mark.argtypes = (ctypes.c_char_p,)
    mark.restype = None
    mark(b"W3_NVTX_INITIALIZE")

    def anchor(name):
        tag = f"RILEY_W3_CLOCK_{name}"
        before = time.perf_counter_ns()
        mark(tag.encode("ascii"))
        after = time.perf_counter_ns()
        return {"name": tag, "before_ns": before, "after_ns": after}

    return anchor


async def load_cell(address, cell):
    host, port = address.split(":")
    rng = random.Random(cell["seed"])
    warmup = cell["warmup"]
    total = warmup + cell["requests"]
    width = cell["concurrency"]
    intervals = cell.get("stagger_ms")
    requests = [{
        "model": "m", "temperature": 0, "max_tokens": cell["max_tokens"], "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt(rng, rng.choice(cell["prompt_words"]))}],
    } for _ in range(total)]
    rows = [{} for _ in range(total)]

    async def request(index):
        await asyncio.wait_for(one(host, int(port), requests[index], time.perf_counter(), rows[index],
                                   capture_hashes=True), timeout=90)

    start = time.perf_counter()
    if intervals is None:
        queue = iter(range(total))

        async def worker():
            for index in queue:
                await request(index)

        await asyncio.gather(*(worker() for _ in range(width)))
    else:
        async def scheduled(index, at):
            await asyncio.sleep(max(0, at - time.perf_counter()))
            await request(index)

        for first in range(0, total, width):
            base = time.perf_counter()
            await asyncio.gather(*(scheduled(index, base + (index - first) * intervals / 1000)
                                   for index in range(first, min(first + width, total))))
            await asyncio.sleep(0.08)
    wall = time.perf_counter() - start
    measured = rows[warmup:]
    summary = summarize(measured, wall * len(measured) / total, cell)
    summary["warmup_ok"] = sum(row.get("error") is False for row in rows[:warmup])
    if (summary["ok"] != cell["requests"] or summary["warmup_ok"] != warmup
            or summary["errors"] or summary["rejected_429"]):
        return {"statistics": summary, "measured": []}
    records = []
    for index, row in enumerate(measured, start=warmup):
        before_first = {key: row[key] for key in (
            "start", "sent_at", "ttft", "e2e", "prompt_tokens", "completion_tokens",
            "finish", "input_sha256", "output_sha256",
        )}
        records.append({"request_index": index, **before_first})
    return {"statistics": summary, "measured": records}


def run(plan_path):
    out = plan_path.absolute().parent
    plan = json.loads(plan_path.read_bytes())
    anchor = nvtx_clock() if plan["profile"] else None

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with (out / "server.log").open("xb") as logfile:
        server = subprocess.Popen(plan["server"], env={**os.environ, **plan["env"]}, stdin=subprocess.DEVNULL,
                                  stdout=logfile, stderr=subprocess.STDOUT)
        try:
            (out / "server.pid").write_text(str(server.pid) + "\n")
            deadline = time.monotonic() + 550
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server startup exit={server.returncode}")
                try:
                    with urllib.request.urlopen(plan["ready"], timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server readiness")
                time.sleep(1)

            for name, cell in plan["cells"].items():
                anchors = [anchor(name + "_begin")] if anchor else []
                result = asyncio.run(load_cell(plan["address"], cell))
                if anchor:
                    anchors.append(anchor(name + "_end"))
                result["anchors"] = anchors
                save(out / (name + ".json"), result)
                if not result["measured"]:
                    raise RuntimeError(f"invalid completion or warmup in {name}")
                print(name, result["statistics"]["ttft_p95_ms"], flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=70)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    save(out / "server-exit.json", {"exit": server.returncode})
    if server.returncode not in (0, -signal.SIGTERM):
        raise RuntimeError(f"graceful server shutdown exit={server.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    run(parser.parse_args().plan)
