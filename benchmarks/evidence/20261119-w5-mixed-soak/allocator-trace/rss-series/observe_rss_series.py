"""Observe additional container RSS samples around the unchanged strict W5 check."""

import asyncio
import json
from pathlib import Path
import re
import sys
import time

import soak


DURING_LOAD = (300, 600, 1200, 1500, 1740)
AFTER_DRAIN = (10, 30, 60)


def read_sample(pid, open_proc, target_seconds, started):
    before = time.perf_counter()
    result = {"target_seconds": target_seconds, "read_start_seconds": before - started}
    try:
        result.update(soak.read_smaps_rollup(pid, open_proc=open_proc))
    except (OSError, ValueError) as error:
        result["read_error"] = type(error).__name__
    result["read_end_seconds"] = time.perf_counter() - started
    return result


def observe_snapshot(snapshot, captured):
    def observe(pid, address, started, **kwargs):
        sample = snapshot(pid, address, started, **kwargs)
        if "smaps_after_seconds" in kwargs:
            elapsed = sample["at_seconds"]
            for seconds in DURING_LOAD:
                if seconds <= elapsed < seconds + 5 and seconds not in captured:
                    diagnostic = read_sample(pid, kwargs["open_proc"], seconds, started)
                    diagnostic["metrics_at_seconds"] = elapsed
                    diagnostic["gauges"] = {name: sample["metrics"][name] for name in soak.IDLE_GAUGES}
                    captured[seconds] = diagnostic
                    break
        return sample

    return observe


async def observe_after(checked_soak, args, captured, open_proc):
    verdict = await checked_soak(args, open_proc=open_proc)
    drain = verdict["smaps_rollup_second_half"]["drain"]
    after = []
    last_idle = None
    if drain is not None and verdict["checks"]["idle_after_drain"] and verdict["problem_count"] == 0:
        started = time.perf_counter()
        for seconds in AFTER_DRAIN:
            await asyncio.sleep(max(0, started + seconds - time.perf_counter()))
            sample = await asyncio.to_thread(read_sample, args.pid, open_proc, seconds, started)
            after.append(sample)
        try:
            terminal = await asyncio.to_thread(soak.snapshot, args.pid, args.address, started, open_proc=open_proc)
            last_idle = soak.idle(terminal["metrics"])
        except (OSError, ValueError):
            last_idle = False
    summary = {
        "schema_version": "riley.w5-rss-series.v1",
        "host_pid": args.pid,
        "original_drain": drain,
        "original_checks": verdict["checks"],
        "during_load": list(captured.values()),
        "after_drain": after,
        "idle_after_last_sample": last_idle,
    }
    path = Path(args.out).with_name("rss-series.json")
    with path.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")
    return verdict


def main():
    args = sys.argv[1:]
    container = args[args.index("--docker-container") + 1]
    if not re.fullmatch(r"[0-9a-f]{64}", container) or "--require-zero-rss-growth" not in args:
        raise ValueError("expected a container-bound strict W5 soak")
    if float(args[args.index("--seconds") + 1]) != 1800 or "--out" not in args:
        raise ValueError("expected a 30-minute W5 soak with an output path")

    captured = {}
    checked_soak = soak.soak
    soak.snapshot = observe_snapshot(soak.snapshot, captured)

    async def run(args, *, open_proc):
        return await observe_after(checked_soak, args, captured, open_proc)

    soak.soak = run
    return soak.main()


if __name__ == "__main__":
    raise SystemExit(main())
