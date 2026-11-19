"""Check the supplementary observer without GPU or an HTTP service."""

import asyncio
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import observe_rss_series as probe
import soak

calls = []


def open_proc(pid, name):
    if pid != 1 or name != "smaps_rollup":
        raise ValueError("the probe used the wrong process or file")
    calls.append(name)
    return StringIO("Rss: 8000 kB\nAnonymous: 6000 kB\n")


def snapshot(pid, address, started, **kwargs):
    return {
        "at_seconds": time.perf_counter() - started,
        "metrics": dict.fromkeys(soak.IDLE_GAUGES, 0),
    }


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


async def main():
    soak.snapshot = snapshot
    captured = {}
    observe = probe.observe_snapshot(snapshot, captured)
    started = time.perf_counter() - probe.DURING_LOAD[0]
    sample = observe(1, "unused", started, open_proc=open_proc, smaps_after_seconds=None)
    observe(1, "unused", started, open_proc=open_proc, smaps_after_seconds=None)
    observe(1, "unused", started, open_proc=open_proc)
    require(len(calls) == len(captured) == 1 and "smaps_rollup" not in sample,
            "supplementary reads changed the strict sampling input")
    require(captured[probe.DURING_LOAD[0]]["anonymous_kib"] == 6000, "the supplementary read is wrong")

    precise = {"rss_kib": 8000, "anonymous_kib": 6000}
    score = {"smaps_rollup_second_half": {"drain": precise},
             "checks": {"idle_after_drain": True, "rss_growth_second_half_at_most_zero": False}, "problem_count": 0}

    async def strict_soak(args, *, open_proc):
        return score

    probe.AFTER_DRAIN = (0.0, 0.01, 0.02)
    with TemporaryDirectory() as folder:
        args = SimpleNamespace(pid=1, address="unused", out=str(Path(folder) / "soak.json"))
        verdict = await probe.observe_after(strict_soak, args, captured, open_proc)
        record = json.loads(Path(folder, "rss-series.json").read_bytes())
        require(verdict is score and record["original_checks"] == score["checks"],
                "post-drain diagnostics changed the failure verdict")
        require(len(record["after_drain"]) == 3 and record["idle_after_last_sample"] is True,
                "post-drain observations are incomplete")
        require(all(row["target_seconds"] <= row["read_start_seconds"] <= row["read_end_seconds"]
                    and row["anonymous_kib"] == 6000 for row in record["after_drain"]),
                "post-drain observations have the wrong data or clock")
    missing = probe.read_sample(2, open_proc, 0, time.perf_counter())
    require(missing.get("read_error") == "ValueError" and "rss_kib" not in missing,
            "an invalid probe was treated as an RSS measurement")
    print("supplementary read, unmodified failure verdict, idle coverage and probe error verified")


if __name__ == "__main__":
    asyncio.run(main())
