"""Admit a bounded FIFO of long workflows while preserving original offered-arrival timestamps."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from agent_workflow import replay


def run(address, mode, base, size):
    condition = threading.Condition()
    next_rank = 0
    active = 0

    def job(i):
        nonlocal next_rank, active
        reader = i % 4 == 0
        scheduled = origin + i * 30_000_000
        time.sleep(max(0, (scheduled - time.monotonic_ns()) / 1e9))
        arrived = time.monotonic_ns()
        if not reader and mode == "window4":
            rank = i - (i // 4 + 1)
            with condition:
                condition.wait_for(lambda: rank == next_rank and active < 4)
                admitted = time.monotonic_ns()
                active += 1
                next_rank += 1
                condition.notify_all()
        else:
            admitted = time.monotonic_ns()
        try:
            w = replay(
                address,
                "m",
                base + i,
                2 if reader else 96,
                0 if reader else 0.5,
                f"job-{base+i}",
                history=0 if reader else 200,
            )
        finally:
            if mode == "window4" and not reader:
                with condition:
                    released = time.monotonic_ns()
                    active -= 1
                    condition.notify_all()
            else:
                released = time.monotonic_ns()
        return {
            "index": i,
            "role": "reader" if reader else "target",
            "scheduled": scheduled,
            "arrived": arrived,
            "admitted": admitted,
            "released": released,
            "work": w,
        }

    with ThreadPoolExecutor(size) as executor:
        origin = time.monotonic_ns() + 100_000_000
        rows = list(executor.map(job, range(size)))
    return {
        "mode": mode,
        "base": base,
        "size": size,
        "origin": origin,
        "drained": time.monotonic_ns(),
        "rows": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("folder", type=Path)
    p.add_argument("gpu")
    args = p.parse_args()
    c = json.loads(
        (Path(__file__).parent / "coder-window-registration.json").read_text()
    )
    warm = replay(args.address, "m", 7091, 2, 0, "warm")
    (args.folder / "warm.json").write_text(json.dumps(warm) + "\n")
    results = []
    for mode, group in zip(c["order"][args.gpu], c["groups"], strict=True):
        for size, base in group:
            results.append(run(args.address, mode, base, size))
            (args.folder / "waves.json").write_text(json.dumps(results) + "\n")
