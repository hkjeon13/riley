"""Bind and verify the independent periodic allocator experiment."""

import json
import os
from pathlib import Path
import subprocess


LOG = "/tmp/riley-w5-periodic-trim.jsonl"
PERIOD_MS = 31_000


def mount_options():
    library = Path(os.environ["RILEY_W5_PERIODIC_TRIM_LIBRARY"]).resolve(strict=True)
    if not library.is_file() or any(char in str(library) for char in (",", "\r", "\n")):
        raise ValueError("invalid periodic allocator library")
    return ("--mount", f"type=bind,source={library},target=/tmp/w5-periodic-trim.so,readonly",
            "--env", "LD_PRELOAD=/tmp/w5-periodic-trim.so",
            "--env", "RILEY_W5_PERIODIC_TRIM_EXE=/opt/riley/bin/riley",
            "--env", f"RILEY_W5_PERIODIC_TRIM_MS={PERIOD_MS}")


def read_telemetry(path, pid, period_ms, duration_seconds):
    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    if len(records) < 2:
        raise ValueError("periodic trim was not observed across the run")
    period_ns = period_ms * 1_000_000
    durations = []
    last = None
    for index, row in enumerate(records, 1):
        start, end = row["started_ns"], row["completed_ns"]
        if (row["pid"] != pid or row["period_ms"] != period_ms or row["index"] != index
            or type(start) is not int or type(end) is not int or not 0 < start <= end
            or (last is not None and start - last < period_ns)
            or type(row["returned"]) is not int or row["returned"] not in (0, 1)):
            raise ValueError("invalid process, timing, or counter in trim telemetry")
        durations.append((end - start) / 1_000_000)
        last = end
    span_seconds = (records[-1]["completed_ns"] - records[0]["started_ns"]) / 1_000_000_000
    if span_seconds < duration_seconds - 2 * period_ms / 1000:
        raise ValueError("trim telemetry ended before covering the soak")
    ordered = sorted(durations)
    return {"period_ms": period_ms, "calls": len(records), "returned_memory": sum(row["returned"] for row in records),
            "span_seconds": span_seconds, "call_ms_p95": ordered[min(int(.95 * len(ordered)), len(ordered) - 1)],
            "call_ms_max": ordered[-1]}


def collect_after_shutdown(container, out):
    path = out / "periodic-trim.jsonl"
    subprocess.run(["docker", "cp", f"{container}:{LOG}", str(path)], check=True, timeout=30)
    raw = json.loads((out / "soak.json").read_bytes())
    return read_telemetry(path, pid=1, period_ms=PERIOD_MS, duration_seconds=raw["seconds"])
