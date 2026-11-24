"""Measure saturation telemetry and actual agent streams during a bounded admission burst."""

import argparse
import concurrent.futures
import http.client
import json
import threading
import time
from pathlib import Path

from agent_fleet_replay import inventory
from agent_workflow import replay, stream
from probe_capacity_route import body
from replica_load import ReplicaLoad
from replica_workflow import Router


def direct(endpoint, request, row, visible=None):
    return stream(endpoint[0], request, row, visible, bearer=endpoint[1])


def wave(endpoints, count, policy, tag, gpu, continuation=None):
    polls = []
    poll_lock = threading.Lock()

    def observe(index, start, end, status, outcome):
        with poll_lock:
            polls.append([index, start, end, status, outcome])

    telemetry = ReplicaLoad(endpoints, on_poll=observe)
    router = Router(endpoints, policy, [8, 8], telemetry)
    send = router.sender(0)
    warm, target = {}, {}
    req = body(
        f"Record {tag}. Read the following report and answer briefly. "
        + "Routine latency measurements stayed stable. " * 950,
        1,
    )
    if continuation is None:
        send("", req, warm, None)
    else:
        continuation.prepare(endpoints[0], send, warm)
    if warm["route"]["replica"] != 0:
        raise RuntimeError("wrong initial placement")
    incumbents = [{} for _ in range(8)]
    started = [threading.Event() for _ in incumbents]
    final = threading.Event()
    barrier = threading.Barrier(count + 1, timeout=30)

    def flood(i):
        request = body(
            f"Entry {i}. Write a detailed technical explanation of solar energy.", 64
        )
        row = {"index": i}
        barrier.wait()
        try:
            direct(endpoints[0], request, row)
        except (OSError, ValueError, http.client.HTTPException) as e:
            row["error"] = str(e)
        row["end_ns"] = time.monotonic_ns()
        return row

    with telemetry, concurrent.futures.ThreadPoolExecutor(count + 9) as pool:
        flood_jobs = [pool.submit(flood, i) for i in range(count)]
        peers = [
            pool.submit(
                direct,
                endpoints[0],
                body(
                    f"Task {i}. Write a detailed and extensive story about a town.", 384
                ),
                row,
                ev,
            )
            for i, (row, ev) in enumerate(zip(incumbents, started))
        ]
        if not all(e.wait(30) for e in started):
            raise TimeoutError("incumbent did not stream")
        reader = pool.submit(
            replay,
            endpoints[1],
            "m",
            761,
            2,
            0,
            f"protected-{tag}",
            final,
            history=20,
            send=direct,
        )
        if not final.wait(30):
            raise ValueError("protected final did not stream")
        release = time.monotonic_ns()
        barrier.wait()
        time.sleep(max(0, (release + 650_000_000 - time.monotonic_ns()) / 1e9))
        before = time.monotonic_ns()
        try:
            if continuation is None:
                send("", req, target, None)
            else:
                continuation.resume(target)
        except (OSError, ValueError, http.client.HTTPException) as e:
            target["error"] = str(e)
        target["end_ns"] = time.monotonic_ns()
        for peer in peers:
            peer.result()
        protected = reader.result()
        burst = [j.result() for j in flood_jobs]
        recovered = {}
        direct(
            endpoints[0], body("Name one common source of solar energy.", 8), recovered
        )
    return {
        "busy_gpu": gpu,
        "count": count,
        "policy": policy,
        "tag": tag,
        "warm": warm,
        "incumbents": incumbents,
        "protected": protected,
        "burst": burst,
        "release_ns": release,
        "before_route_ns": before,
        "target": target,
        "polls": polls,
        "recovery": recovered,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--protect-reader", action="store_true")
    args = p.parse_args()
    endpoints, _ = inventory(json.loads(args.inventory.read_text()))
    waves = []
    for block in (0, 1):
        counts = [0, 16, 64, 160]
        policies = [
            "observed-capacity",
            "defer-cold-prefill" if args.protect_reader else "fresh-capacity",
        ]
        if block:
            counts.reverse()
            policies.reverse()
        for count in counts:
            for policy in policies:
                waves.append(
                    wave(
                        endpoints[::-1] if block else endpoints,
                        count,
                        policy,
                        9650 + len(waves),
                        6 + block,
                    )
                )
                args.output.write_text(json.dumps(waves) + "\n")


if __name__ == "__main__":
    main()
