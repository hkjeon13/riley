"""Drive fixed arrivals and observe speculative prompt work with other agents using the replica."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import json
from pathlib import Path
import threading
import time
from urllib.request import urlopen

from agent_workflow import replay, stream
from tool_wait_policy import ToolWaitPolicy
from tool_wait_trial import prepare_prefix


def snapshot(address, log):
    with urlopen(f"http://{address}/metrics", timeout=10) as response:
        metrics = json.load(response)
    assert metrics["active_requests"] == metrics["waiting_requests"] == 0
    return {"metrics": metrics, "log_offset": log.stat().st_size}


def wave(address, size, base, mode, shared, folder):
    lock = threading.Lock()
    foreground = 0

    def send(endpoint, request, row, first_visible):
        nonlocal foreground
        with lock:
            foreground += 1
        try:
            return stream(endpoint, request, row, first_visible)
        finally:
            with lock:
                foreground -= 1

    def run_job(index, origin):
        is_reader = index % 4 == 0
        seed = base + index
        scheduled = origin + 30_000_000 * index
        time.sleep(max(0, (scheduled - time.monotonic_ns()) / 1e9))
        preview = {}
        decision = {}

        def start(turn, request, row):
            if turn != 1 or is_reader:
                return
            with lock:
                busy = foreground
            now = time.monotonic_ns()
            if mode == "off":
                choice = "off"
            elif mode == "headroom" and busy >= 6:
                choice = "foreground_busy"
            else:
                choice = shared.reserve(seed, 500, row["usage"]["prompt_tokens"])
            decision.update(choice=choice, foreground=busy, at_ns=now)
            if choice == "submit":
                prepared = prepare_prefix(request, row["call"])

                def launch():
                    try:
                        stream(address, prepared, preview)
                    except (OSError, ValueError) as exc:
                        preview["error"] = str(exc)
                    finally:
                        shared.finish(
                            seed, "done_ns" in preview and not preview.get("error")
                        )

                pending.append(previews.submit(launch))

        pending = []
        w = replay(
            address,
            "m",
            seed,
            2 if is_reader else 96,
            0 if is_reader else 0.5,
            f"job-{seed}",
            history=0 if is_reader else 200,
            send=send,
            on_tool_start=start,
        )
        for future in pending:
            future.result()
        return {
            "index": index,
            "role": "reader" if is_reader else "target",
            "scheduled_ns": scheduled,
            "work": w,
            "decision": decision,
            "preview": preview,
        }

    before = snapshot(address, folder / "server.log")
    with ThreadPoolExecutor(1) as previews, ThreadPoolExecutor(size) as clients:
        origin = time.monotonic_ns() + 100_000_000
        rows = list(clients.map(functools.partial(run_job, origin=origin), range(size)))
    drained = time.monotonic_ns()
    after = snapshot(address, folder / "server.log")
    return {
        "size": size,
        "base": base,
        "origin_ns": origin,
        "drained_ns": drained,
        "before": before,
        "after": after,
        "jobs": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("folder", type=Path)
    p.add_argument("mode", choices=["off", "bounded", "headroom"])
    args = p.parse_args()
    config = json.loads(
        (Path(__file__).parent / "coder-pressure-registration.json").read_text()
    )
    warm = replay(args.address, "m", 5091, 2, 0, "warm")
    (args.folder / "warm.json").write_text(json.dumps(warm) + "\n")
    policy = ToolWaitPolicy()
    results = []
    for size, base in config["waves"]:
        results.append(wave(args.address, size, base, args.mode, policy, args.folder))
        (args.folder / "waves.json").write_text(json.dumps(results) + "\n")
