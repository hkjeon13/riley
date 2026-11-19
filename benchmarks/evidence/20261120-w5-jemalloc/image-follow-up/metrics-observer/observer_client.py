"""Measure extra metrics polling during the pinned mixed load."""

from argparse import ArgumentParser
from http.client import HTTPConnection
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from threading import Event, Thread
import time

from soak import open_container_proc, read_smaps_rollup


PHASES = (("baseline-first", False), ("scrape-first", True), ("scrape-second", True), ("baseline-second", False))
SECONDS = 90
RATE = 8


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def require(condition, error):
    if not condition:
        raise ValueError(error)


def process_cpu(pid):
    fields = (Path("/proc") / str(pid) / "stat").read_text().rpartition(")")[2].split()
    return {"start_ticks": int(fields[19]), "user_ticks": int(fields[11]), "system_ticks": int(fields[12])}


def poll_metrics(host, port, done, samples):
    scheduled = time.monotonic()
    while not done.is_set():
        wire = HTTPConnection(host, port, timeout=3)
        started = time.monotonic()
        try:
            wire.request("GET", "/metrics")
            reply = wire.getresponse()
            closed = json.load(reply)
            require(reply.status == 200 and {"counters", "batch_shapes", "allocation"} <= closed.keys(),
                    "extra metrics probe failed")
            samples.append({"elapsed_ms": (time.monotonic()-started) * 1000, "lag_ms": (started-scheduled) * 1000})
        except (OSError, ValueError):
            samples.append({"failed": True})
        finally:
            wire.close()
        scheduled += 1
        done.wait(max(0, scheduled - time.monotonic()))


def capture(address, model, directory):
    host, port = address.split(":")
    require(host == "127.0.0.1", "expected loopback")
    harness = Path(os.environ["RILEY_W5_PERF_HARNESS"])
    process = read(directory.parent / "process-before.json")
    pid, container = process["host_pid"], process["container_id"]

    def open_proc(_pid, filename):
        return open_container_proc(container, pid, filename)

    for name, scrape in PHASES:
        samples = []
        stop = Event()
        worker = Thread(target=poll_metrics, args=(host, int(port), stop, samples)) if scrape else None
        start_rss = read_smaps_rollup(pid, open_proc=open_proc)
        start_cpu = process_cpu(pid)
        began = time.monotonic_ns()
        if worker is not None:
            worker.start()
        try:
            subprocess.run([sys.executable, "-B", str(harness / "soak.py"), address, model,
                            "--pid", str(pid), "--docker-container", container, "--smoke",
                            "--seconds", str(SECONDS), "--rate", str(RATE), "--out", str(directory / f"{name}.json")],
                           check=True, timeout=SECONDS + 100)
        finally:
            stop.set()
            if worker is not None:
                worker.join(timeout=10)
        ended = time.monotonic_ns()
        end_cpu = process_cpu(pid)
        end_rss = read_smaps_rollup(pid, open_proc=open_proc)
        require(start_cpu["start_ticks"] == end_cpu["start_ticks"], "measured process changed")
        require(worker is None or (not worker.is_alive() and len(samples) >= SECONDS - 1
                                  and not any("failed" in item for item in samples)), "extra metrics probe failed")
        write(directory / f"{name}.observer.json",
              {"extra_metrics": scrape, "samples": samples, "process": process, "begin_ns": began, "end_ns": ended,
               "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"), "cpu_start": start_cpu, "cpu_end": end_cpu,
               "rss_start": start_rss, "rss_end": end_rss})


def summary(directory, artifacts):
    roles = {}
    counts = None
    previous_end = 0
    for name, scrape in PHASES:
        workload = read(directory / f"{name}.json")
        sample = read(directory / f"{name}.observer.json")
        require(sample["extra_metrics"] is scrape and previous_end <= sample["begin_ns"] < sample["end_ns"],
                "observer phases out of order")
        previous_end = sample["end_ns"]
        require(workload["status"] == "passed" and workload["problem_count"] == 0
                and workload["seconds"] == SECONDS and workload["rate"] == RATE
                and (counts is None or counts == workload["counts"]), "workload changed or failed")
        counts = workload["counts"]
        start, end = sample["cpu_start"], sample["cpu_end"]
        require(start["start_ticks"] == end["start_ticks"], "CPU snapshot identity changed")
        seconds = (sample["end_ns"] - sample["begin_ns"]) / 10**9
        cpu = sum(end[key] - start[key] for key in ("user_ticks", "system_ticks"))
        require(cpu > 0, "no process CPU samples")
        before, after = (workload[period]["metrics"]["batch_shapes"]["buckets"] for period in ("initial","final"))
        gpu_ns = sum(right["gpu_execution_ns_total"] - left["gpu_execution_ns_total"]
                     for left, right in zip(before, after, strict=True))
        require(gpu_ns > 0, "no GPU workload")
        samples = sample["samples"]
        latency = workload["latency_windows"]["late"]
        roles[name] = {
            "image_id": sample["process"]["image_id"], "elapsed_seconds": seconds,
            "extra_metrics_requests": len(samples),
            "extra_metrics_latency_p50_ms": median(value["elapsed_ms"] for value in samples) if samples else None,
            "cpu_seconds_per_wall_second": cpu / sample["clock_ticks_per_second"] / seconds,
            "gpu_execution_ms": gpu_ns / 10**6,
            "rss_growth_kib": sample["rss_end"]["rss_kib"] - sample["rss_start"]["rss_kib"],
            "late_completed": latency["completed"], "late_ttft_p95_ms": latency["ttft_ms_p95"],
            "late_e2e_p95_ms": latency["e2e_ms_p95"], "client_dispatch_p95_ms": workload["client_dispatch_p95_ms"],
            "workload_sha256": artifacts[f"{name}.json"], "measurement_sha256": artifacts[f"{name}.observer.json"],
        }
    pairs = {}
    for order in ("first", "second"):
        base, watched = (roles[f"{kind}-{order}"] for kind in ("baseline", "scrape"))
        pairs[order] = {key: 100 * (watched[key] / base[key] - 1) for key in
                        ("cpu_seconds_per_wall_second", "gpu_execution_ms", "late_ttft_p95_ms", "late_e2e_p95_ms")}
    return {"phases": roles, "paired_extra_scrape_percent": pairs, "same_counts_per_phase": counts}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--tools", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    api = Path(os.environ["RILEY_W5_PERF_HARNESS"]) / "api_check.py"
    subprocess.run([sys.executable, "-B", str(api), args.address, args.model, "--tools", "--out", str(args.out)],
                   check=True, timeout=600)
    capture(args.address, args.model, args.out.parent)
