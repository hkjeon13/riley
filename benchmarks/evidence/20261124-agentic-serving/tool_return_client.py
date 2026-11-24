"""Measure full tool returns arriving while another session streams its final answer."""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_workflow import stream
from long_tool_records import extract, fixture

HERE = Path(__file__).resolve().parent


def schedule(reg):
    cells = {}
    for spacing in reg["spacing_ms"]:
        for size in reg["sizes"]:
            key = f"returns-{size}-spacing-{spacing}"
            cells[key] = {
                "requests": size,
                "spacing_ms": spacing,
                "seed": reg["seed_base"] + 31 * len(cells),
                "counts": reg["records_per_return"],
            }
    return {"schedules": {gpu: cells for gpu in reg["order"]}}


def reader(address, cell, seen, anchor):
    req = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"For worksheet {cell['seed']}, list the integers 1 through 250 in order, "
                    "one per line, without commentary."
                ),
            }
        ],
        "max_tokens": 160,
        "temperature": 0,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    row = {}

    def first(n):
        if n == 1:
            anchor.append(row["content_ns"][0])
            seen.set()

    answer, calls = stream(address, req, row, on_content=first)
    if calls or not answer:
        raise ValueError("reader had no final answer")
    return row


def wave(address, cell):
    size = cell["requests"]
    ready = threading.Barrier(size + 1, timeout=30)
    seen = threading.Event()
    anchor = []

    def send(i):
        seed = cell["seed"] + i
        req, expected = fixture(seed, cell["counts"][i % len(cell["counts"])])
        req["return_token_ids"] = True
        ready.wait()
        if not seen.wait(60):
            raise TimeoutError("reader did not stream a first token")
        scheduled = anchor[0] + cell["spacing_ms"] * i * 1_000_000
        time.sleep(max(0, scheduled - time.monotonic_ns()) / 1e9)
        row = {"seed": seed, "expected": expected, "scheduled_ns": scheduled}
        text, calls = stream(address, req, row)
        row.update(answer=text, correct=not calls and extract(text) == expected)
        return row

    with ThreadPoolExecutor(max_workers=size + 1) as pool:
        returns = [pool.submit(send, i) for i in range(size)]
        ready.wait()
        ongoing = pool.submit(reader, address, cell, seen, anchor)
        return {"reader": ongoing.result(), "measured": [r.result() for r in returns]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("address", nargs="?")
    a = p.parse_args()
    if a.address:
        os.sched_setaffinity(0, {int(os.environ["W3_CLIENT_CPU"])})
        plan = json.loads(a.plan.read_text())
        for name, spec in plan["cells"].items():
            data = wave(a.address, spec)
            (a.plan.parent / f"{name}.json").write_text(json.dumps(data) + "\n")
            print(name, len(data["measured"]), flush=True)
    else:
        reg = json.loads((HERE / "tool-return-registration.json").read_text())
        a.plan.write_text(json.dumps(schedule(reg), indent=2) + "\n")
