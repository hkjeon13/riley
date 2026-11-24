"""Route real tool-return turns with a cached history while another user's answer streams."""

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path

from agent_fleet_replay import inventory
from agent_workflow import replay
from readiness_burst import direct, wave


class Continuation:
    def __init__(self, pool, seed):
        self.pool = pool
        self.seed = seed
        self.preview_done = threading.Event()
        self.tool_return = threading.Event()
        self.final = {}

    def prepare(self, busy, route, warm):
        def send(_, req, row, visible):
            if req["tool_choice"] != "none":
                return direct(busy, req, row, visible)
            try:
                return route("", req, row, visible)
            finally:
                self.final = row

        def tool_start(index, req, row):
            if index == 1:
                route("", {**req, "tool_choice": "none", "max_tokens": 1}, warm, None)
                self.preview_done.set()
                if not self.tool_return.wait(60):
                    raise TimeoutError("timed out waiting for the external tool")

        self.task = self.pool.submit(
            replay,
            "",
            "m",
            self.seed,
            96,
            0,
            f"resumed-{self.seed}",
            history=200,
            send=send,
            on_tool_start=tool_start,
        )
        if not self.preview_done.wait(35):
            if self.task.done():
                raise RuntimeError(
                    f"could not prepare final turn: {self.task.result().get('error')}"
                )
            raise TimeoutError("no prefix preview for the pending tool")

    def resume(self, target):
        self.tool_return.set()
        self.result = self.task.result(timeout=90)
        target.update(self.final)
        if not self.result["ok"]:
            target["validation_error"] = self.result.get("error")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("block", type=int, choices=[0, 1])
    args = p.parse_args()
    endpoints, _ = inventory(json.loads(args.inventory.read_text()))
    rows = []
    for idx, (gpu, count, seed) in enumerate(
        [(6, 16, 761), (7, 160, 18081), (7, 16, 19959), (6, 160, 762)]
    ):
        policy = ["observed-capacity", "defer-cold-prefill"][(idx + args.block) % 2]
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            natural = Continuation(pool, seed)
            try:
                record = wave(
                    endpoints if gpu == 6 else endpoints[::-1],
                    count,
                    policy,
                    seed,
                    gpu,
                    natural,
                )
            finally:
                natural.tool_return.set()
        record["workflow"] = natural.result
        rows.append(record)
        args.output.write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    main()
