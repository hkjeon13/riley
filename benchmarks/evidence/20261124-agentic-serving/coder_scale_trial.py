"""Replay a fixed offered workload on two independent same-host replicas with workflow affinity."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from agent_workflow import replay
from coder_pressure import snapshot


def run(root):
    conf = json.loads((root / "coder-pressure-registration.json").read_text())
    dirs = [root / "gpu6", root / "gpu7"]
    endpoints = [(d / "address").read_text().strip() for d in dirs]
    assert len(set(endpoints)) == 2
    for addr, d in zip(endpoints, dirs, strict=True):
        warm = replay(addr, "m", 5091, 2, 0, "warm")
        (d / "warm.json").write_text(json.dumps(warm) + "\n")

    def snapshots():
        return [
            snapshot(addr, d / "server.log")
            for addr, d in zip(endpoints, dirs, strict=True)
        ]

    out = []
    for size, base in conf["waves"]:
        before = snapshots()

        def job(i):
            scheduled = origin + 30_000_000 * i
            time.sleep(max(0, (scheduled - time.monotonic_ns()) / 1e9))
            reader = i % 4 == 0
            w = replay(
                endpoints[i % 2],
                "m",
                base + i,
                2 if reader else 96,
                0 if reader else 0.5,
                f"job-{base+i}",
                history=0 if reader else 200,
            )
            return {"index": i, "gpu": 6 + i % 2, "scheduled": scheduled, "work": w}

        with ThreadPoolExecutor(size) as pool:
            origin = time.monotonic_ns() + 100_000_000
            rows = list(pool.map(job, range(size)))
        drained = time.monotonic_ns()
        out.append(
            {
                "base": base,
                "size": size,
                "origin": origin,
                "drained": drained,
                "before": before,
                "after": snapshots(),
                "rows": rows,
            }
        )
        (root / "waves.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    run(Path(__file__).parent)
