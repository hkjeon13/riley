"""Compare the previous and current HTTP clients under real, bounded Riley overload."""

import argparse
import concurrent.futures
import http.client
import json
import threading
import time
from pathlib import Path

import agent_workflow
import original_agent_workflow
from agent_fleet_replay import inventory
from probe_capacity_route import body


def run(endpoint, gpu, arm, limit):
    dest, key = endpoint
    send = (
        agent_workflow.stream if arm == "candidate" else original_agent_workflow.stream
    )
    ready = [threading.Event() for _ in range(8)]
    peers = [{} for _ in ready]
    barrier = threading.Barrier(161, timeout=30)

    def arrive(i):
        row = {"index": i}
        req = body(
            f"Entry {i}. Write a detailed technical explanation of solar energy.", limit
        )
        barrier.wait()
        try:
            send(dest, req, row, bearer=key)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            row["error"] = type(exc).__name__
        return row

    with concurrent.futures.ThreadPoolExecutor(168) as pool:
        jobs = [pool.submit(arrive, i) for i in range(160)]
        incumbents = [
            pool.submit(
                send,
                dest,
                body(
                    f"Task {i}. Write a detailed and extensive story about a town.", 192
                ),
                row,
                event,
                bearer=key,
            )
            for i, (row, event) in enumerate(zip(peers, ready))
        ]
        if not all(ev.wait(30) for ev in ready):
            raise TimeoutError("incumbents did not stream")
        release = time.monotonic_ns()
        barrier.wait()
        rows = [j.result() for j in jobs]
        for j in incumbents:
            j.result()
    return {
        "gpu": gpu,
        "arm": arm,
        "limit": limit,
        "release_ns": release,
        "peers": peers,
        "arrivals": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    endpoints, _ = inventory(json.loads(args.inventory.read_text()))
    rows = []
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for idx, limit in enumerate((16, 16, 64, 64)):
            jobs = [
                pool.submit(
                    run, endpoint, 6 + i, ["control", "candidate"][(idx + i) % 2], limit
                )
                for i, endpoint in enumerate(endpoints)
            ]
            rows.extend(j.result() for j in jobs)
            args.output.write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    main()
