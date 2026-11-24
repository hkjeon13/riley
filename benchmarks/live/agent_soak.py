"""Open-loop multi-turn traffic with tool waits, slow readers, cancellation and KV snapshots."""

import argparse
import collections
import concurrent.futures
import json
import threading
import time
from pathlib import Path
from urllib.request import urlopen

from agent_workflow import percentile, replay, stream


def sample(address, start):
    with urlopen(f"http://{address}/metrics", timeout=5) as response:
        m = json.load(response)
    return {
        "elapsed_s": time.monotonic() - start,
        **{
            k: m[k]
            for k in (
                "active_requests",
                "waiting_requests",
                "kv_allocated_blocks",
                "prefix_cache",
                "allocation",
                "counters",
            )
        },
    }


def one(address, idx, offset, start):
    kind = "cancel" if idx % 19 == 11 else "slow" if idx % 17 == 12 else "normal"
    long = idx % 3 == 1

    def send(host, body, row, first):
        def receive(n):
            if kind == "cancel" and n == 4:
                row["client_cancel_ns"] = time.monotonic_ns()
                raise ValueError("scheduled client cancellation")
            if kind == "slow":
                time.sleep(0.03)

        return stream(
            host,
            body,
            row,
            first,
            on_content=receive if row["turn"] == 2 and kind != "normal" else None,
        )

    entered = time.monotonic()
    r = replay(
        address,
        "m",
        5300 + idx,
        96 if long else 2,
        0.2 if idx % 4 == 0 else 0.01,
        f"job-{idx}",
        history=(0, 50, 200)[idx % 3],
        send=send,
    )
    return {
        "index": idx,
        "kind": kind,
        "scheduled_s": offset,
        "dispatch_lag_ms": (entered - start - offset) * 1000,
        "correct": r["ok"],
        "reason": r.get("error"),
        "cancelled_at_client": any("client_cancel_ns" in t for t in r["turns"]),
        "workflow_first_ms": r.get("first_final_ms"),
        "workflow_last_ms": r.get("last_final_ms"),
        "gap_p95_ms": r.get("final_gap_p95_ms"),
        "requests": [
            {
                k: t[k]
                for k in ("status", "response_id", "sent_ns", "done_ns", "usage")
                if k in t
            }
            for t in r["turns"]
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("directory", type=Path)
    p.add_argument("--seconds", type=int, default=1800)
    args = p.parse_args()
    start = time.monotonic()
    stop = threading.Event()
    samples = []

    def observe():
        while not stop.is_set():
            samples.append(sample(args.address, start))
            if stop.wait(10):
                break

    futures = []
    rows = []
    with concurrent.futures.ThreadPoolExecutor(48) as pool:
        meter = pool.submit(observe)
        try:
            n = 0
            offset = 0.0
            with (args.directory / "requests.jsonl").open("x") as sink:
                while offset < args.seconds:
                    time.sleep(max(0, start + offset - time.monotonic()))
                    futures.append(pool.submit(one, args.address, n, offset, start))
                    pending = []
                    for f in futures:
                        if f.done():
                            r = f.result()
                            sink.write(json.dumps(r) + "\n")
                            rows.append(r)
                        else:
                            pending.append(f)
                    sink.flush()
                    futures = pending
                    # Fourfold synchronized resumption in a 20-second window every 3min.
                    offset += 0.5 if 120 <= offset % 180 < 140 else 2
                    n += 1
                for f in concurrent.futures.as_completed(futures):
                    r = f.result()
                    sink.write(json.dumps(r) + "\n")
                    rows.append(r)
        finally:
            stop.set()
            meter.result()
    samples.append(sample(args.address, start))
    (args.directory / "time-series.json").write_text(json.dumps(samples) + "\n")
    good = [
        r
        for r in rows
        if r["correct"]
        and r["workflow_first_ms"] <= 10000
        and r["workflow_last_ms"] <= 18000
        and r["gap_p95_ms"] <= 150
    ]
    status = collections.Counter(
        r.get("status") for row in rows for r in row["requests"]
    )
    windows = {}
    for label, lo, hi in [("start", 0, 1 / 3), ("end", 2 / 3, 1)]:
        current = [
            r
            for r in rows
            if r["correct"]
            and lo * args.seconds <= r["scheduled_s"] < hi * args.seconds
        ]
        windows[label] = {
            k: percentile([r[k] for r in current], 0.95)
            for k in ("workflow_first_ms", "workflow_last_ms", "gap_p95_ms")
        }
    summary = {
        "submitted": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "slo_completed": len(good),
        "scheduled_cancellations": sum(r["kind"] == "cancel" for r in rows),
        "client_drops": sum(r["cancelled_at_client"] for r in rows),
        "status_counts": dict(status),
        "measurement_s": time.monotonic() - start,
        "windows": windows,
        "final_metrics": samples[-1],
        "dispatch_lag_p95_ms": percentile([r["dispatch_lag_ms"] for r in rows], 0.95),
    }
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
