"""Compare two placement policies under generation traffic from a separate process."""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path

from agent_fleet_replay import inventory
from agent_workflow import stream
from probe_capacity_route import body
from replica_load import ReplicaLoad
from replica_workflow import Router


def load_replica(endpoint, count, opened):
    dest, key = endpoint
    signals = [threading.Event() for _ in range(count)]
    rows = [{} for _ in signals]
    with concurrent.futures.ThreadPoolExecutor(count) as pool:
        tasks = [
            pool.submit(
                stream,
                dest,
                body(
                    f"Task {i}. Write a detailed and extensive story about a town.", 192
                ),
                rows[i],
                signals[i],
                bearer=key,
            )
            for i in range(count)
        ]
        if not all(signal.wait(30) for signal in signals):
            raise TimeoutError("peer stream missing")
        opened.set()
        for task in tasks:
            task.result()
    return os.getpid(), rows


def probe(endpoints, active, policy, tag, warm_gpu):
    ctx = multiprocessing.get_context("spawn")
    req = body(
        f"Record {tag}. Read the following report and answer briefly. "
        + "Routine latency measurements stayed stable. " * 950,
        1,
    )
    sampler = ReplicaLoad(endpoints) if policy == "observed-capacity" else nullcontext()
    warm, target = {}, {}
    with sampler as telemetry, ctx.Manager() as manager:
        router = Router(endpoints, policy, [8, 8], telemetry)
        send = router.sender(0)
        send("", req, warm, None)
        if warm["route"]["replica"] != 0:
            raise RuntimeError("test did not establish the designated warm replica")
        opened = manager.Event()
        with concurrent.futures.ProcessPoolExecutor(1, mp_context=ctx) as pool:
            peer = pool.submit(load_replica, endpoints[0], active, opened)
            if not opened.wait(35):
                raise TimeoutError("separate client did not establish the wave")
            time.sleep(0.18)
            send("", req, target, None)
            pid, rows = peer.result()
    return {
        "policy": policy,
        "warm_gpu": warm_gpu,
        "active": active,
        "tag": tag,
        "router_pid": os.getpid(),
        "peer_pid": pid,
        "initial": warm,
        "target": target,
        "peers": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    endpoints, _ = inventory(json.loads(args.inventory.read_text()))
    rows = []
    for block in range(2):
        order = ["capacity", "observed-capacity"]
        if block:
            order.reverse()
        for active in [4, 7, 8] if not block else [8, 7, 4]:
            for policy in order:
                rows.append(
                    probe(
                        endpoints if not block else endpoints[::-1],
                        active,
                        policy,
                        8701 + len(rows),
                        6 + block,
                    )
                )
                args.output.write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    main()
