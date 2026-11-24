"""Make one live placement using real outstanding streams known to the benchmark router."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
from pathlib import Path

from agent_fleet_replay import inventory
from agent_workflow import stream
from replica_workflow import Router


def body(text, limit):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": text}],
        "tool_choice": "none",
        "tools": [],
        "max_tokens": limit,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def probe(endpoints, trial, active, policy):
    router = Router(endpoints, policy, [8, 8])
    prompt = body(
        f"Record {trial}. Read the following report and answer briefly. "
        + "Routine latency measurements stayed stable. " * 950,
        1,
    )
    initial = {}
    router.sender(0)("", prompt, initial, None)
    if (
        initial["route"]["replica"] != 0
        or not 4500 <= initial["usage"]["prompt_tokens"] <= 7900
    ):
        raise ValueError("ineligible warm prefix")
    events = [threading.Event() for _ in range(active)]
    peers = [{} for _ in events]

    def busy(i):
        dest, key = endpoints[0]
        with router.lock:
            router.active[0] += 1
        try:
            stream(
                dest,
                body(
                    f"Task {i}. Write a detailed and extensive story about a town.", 192
                ),
                peers[i],
                events[i],
                bearer=key,
            )
        finally:
            with router.lock:
                router.active[0] -= 1

    measurement = {}
    with concurrent.futures.ThreadPoolExecutor(active) as pool:
        jobs = [pool.submit(busy, i) for i in range(active)]
        if not all(event.wait(30) for event in events):
            raise TimeoutError("peer not streaming")
        router.sender(0)("", prompt, measurement, None)
        for job in jobs:
            job.result()
    overlap = sum(
        r["content_ns"][0] <= measurement["sent_ns"] < r["done_ns"] for r in peers
    )
    return {
        "trial": trial,
        "policy": policy,
        "active": active,
        "peers_active_at_send": overlap,
        "warm_receipt": initial,
        "decision_receipt": measurement,
        "peer_ids": [r["response_id"] for r in peers],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    endpoints, _ = inventory(json.loads(a.inventory.read_text()))
    results = []
    for k, (active, policy) in enumerate(
        (
            (4, "cost"),
            (4, "capacity"),
            (8, "capacity"),
            (8, "cost"),
            (7, "capacity"),
            (7, "cost"),
            (8, "cost"),
            (8, "capacity"),
        )
    ):
        results.append(probe(endpoints, 6500 + k, active, policy))
        a.out.write_text(json.dumps(results) + "\n")

    directory = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(directory / "natural_lookup_trial.py"),
            str(a.inventory),
            str(directory / "lookup-decode-registration.json"),
            str(a.out.parent / "natural-lookup.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
