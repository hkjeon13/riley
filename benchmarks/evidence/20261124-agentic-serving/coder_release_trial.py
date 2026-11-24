"""Replay two exact versions of the public client against shared per-replica preview state."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import legacy_tool_wait_trial
from agent_workflow import replay
from tool_wait_policy import ToolWaitPolicy
import tool_wait_trial


def wave(address, mode, base, size, spacing):
    measure = (legacy_tool_wait_trial if mode == "legacy" else tool_wait_trial).measure
    policy = ToolWaitPolicy()
    with ThreadPoolExecutor(size) as pool:
        origin = time.monotonic_ns() + 100_000_000

        def run(index):
            scheduled = origin + index * spacing * 1_000_000
            time.sleep(max(0, (scheduled - time.monotonic_ns()) / 1e9))
            return {
                "scheduled": scheduled,
                "record": measure(
                    address,
                    True,
                    "target",
                    base + index,
                    0.5,
                    policy=policy,
                    estimated_wait_ms=500,
                ),
            }

        rows = list(pool.map(run, range(size)))
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
    p.add_argument("directory", type=Path)
    p.add_argument("gpu")
    args = p.parse_args()
    conf = json.loads(
        (Path(__file__).parent / "coder-release-registration.json").read_text()
    )
    warm = replay(args.address, "m", 6091, 2, 0, "warm")
    (args.directory / "warm.json").write_text(json.dumps(warm) + "\n")
    results = []
    for mode, bases in zip(conf["order"][args.gpu], conf["groups"], strict=True):
        for size, base in zip(conf["sizes"], bases, strict=True):
            results.append(wave(args.address, mode, base, size, conf["arrival_ms"]))
            (args.directory / "waves.json").write_text(json.dumps(results) + "\n")
