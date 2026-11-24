"""Assign the GPU once using only data available at the first submitted chat request."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import threading
import time

from agent_workflow import replay, stream
from coder_pressure import snapshot


def workload(size, base, layout):
    readers = list(range(0, size, 4))
    if layout != "regular":
        readers = random.Random(base).sample(range(size), size // 4)
    long_jobs = [i for i in range(size) if i not in readers]
    wrong = set(readers[:2] + long_jobs[:2]) if layout == "imperfect" else set()
    return [
        (i in readers, 200 if ((i not in readers) != (i in wrong)) else 0)
        for i in range(size)
    ]


def wave(endpoints, dirs, mode, size, base, layout):
    totals = [0, 0]
    classes = [[0, 0], [0, 0]]
    queue_lock = threading.Lock()
    specs = workload(size, base, layout)

    def snapshots():
        return [
            snapshot(address, d / "server.log")
            for address, d in zip(endpoints, dirs, strict=True)
        ]

    before = snapshots()

    def job(i):
        reader, history = specs[i]
        scheduled = origin + 30_000_000 * i
        time.sleep(max(0, (scheduled - time.monotonic_ns()) / 1e9))
        route = {}

        def send(_, request, row, visible):
            if not route:
                length = len(
                    json.dumps(request["messages"], ensure_ascii=False).encode()
                )
                kind = int(length >= 4096)
                with queue_lock:
                    rank = sum(totals)
                    device = (
                        rank % 2
                        if mode == "rr"
                        else min(
                            range(2), key=lambda d: (classes[kind][d], totals[d], d)
                        )
                    )
                    route.update(rank=rank, device=device, bytes=length)
                    classes[kind][device] += 1
                    totals[device] += 1
            return stream(endpoints[route["device"]], request, row, visible)

        w = replay(
            "",
            "m",
            base + i,
            2 if reader else 96,
            0 if reader else 0.5,
            f"job-{base+i}",
            history=history,
            send=send,
        )
        return {
            "index": i,
            "reader": reader,
            "scheduled": scheduled,
            "route": route,
            "work": w,
        }

    with ThreadPoolExecutor(size) as pool:
        origin = time.monotonic_ns() + 100_000_000
        rows = list(pool.map(job, range(size)))
    drained = time.monotonic_ns()
    return {
        "size": size,
        "base": base,
        "layout": layout,
        "origin": origin,
        "drained": drained,
        "before": before,
        "after": snapshots(),
        "rows": rows,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    conf = json.loads((root / "coder-placement-registration.json").read_text())
    mode = os.environ["CODER_PLACEMENT"]
    assert mode in ("rr", "history")
    dirs = [root / "gpu6", root / "gpu7"]
    addresses = [(d / "address").read_text().strip() for d in dirs]
    assert len(set(addresses)) == 2
    for d, addr in zip(dirs, addresses, strict=True):
        warm = replay(addr, "m", 5091, 2, 0, "warm")
        (d / "warm.json").write_text(json.dumps(warm) + "\n")
    results = []
    for spec in conf["waves"]:
        results.append(wave(addresses, dirs, mode, *spec))
        (root / "waves.json").write_text(json.dumps(results) + "\n")
