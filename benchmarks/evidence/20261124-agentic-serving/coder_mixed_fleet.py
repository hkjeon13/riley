"""Replay independently recorded RAG and C32 arrivals concurrently on two replicas."""

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from urllib.request import urlopen

from load import one, prompt


def corpus(plan):
    events = []
    for label, cell in plan["cells"].items():
        short = label == "c32-vllm-arrivals"
        spec = cell["cell"]
        rng = random.Random(spec["seed"])
        bodies = [
            {
                "model": "m",
                "temperature": 0,
                "max_tokens": spec["max_tokens"],
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [
                    {
                        "role": "user",
                        "content": prompt(rng, rng.choice(spec["prompt_words"])),
                    }
                ],
            }
            for _ in range(spec["warmup"] + spec["requests"])
        ]
        for row in cell["schedule"]:
            body = bodies[row["request_index"]]
            if (
                hashlib.sha256(json.dumps(body).encode()).hexdigest()
                != row["input_sha256"]
            ):
                raise ValueError("changed input")
            size = len(body["messages"][0]["content"].encode())
            if (size < 2048) != short:
                raise ValueError("request shape does not match the known class")
            events.append(
                (
                    row["offset_ns"] / 1e9 + (4 if short else 0),
                    row["request_index"],
                    body,
                    short,
                )
            )
    return sorted(events, key=lambda e: (e[0], e[1]))


class Picker:
    """Single asynchronous dispatcher. Only arrivals and completed calls update state."""

    def __init__(self, policy, short_home):
        self.policy = policy
        self.short_home = short_home
        self.loads = [0, 0]
        self.shorts = 0
        self.last_empty = float("-inf")
        self.sequence = 0

    def begin(self, short, now):
        tie = self.sequence % 2
        self.sequence += 1
        if self.policy == "round-robin":
            pick = tie
        elif (
            self.policy == "dedicated"
            or short
            or self.shorts
            or now - self.last_empty < 0.25
        ):
            pick = self.short_home if short else 1 - self.short_home
        else:
            pick = min((tie, 1 - tie), key=self.loads.__getitem__)
        prior = self.loads.copy()
        self.loads[pick] += 1
        self.shorts += int(short)
        return pick, prior

    def end(self, pick, short, now):
        self.loads[pick] -= 1
        self.shorts -= int(short)
        if short and not self.shorts:
            self.last_empty = now


async def arm(events, ports, policy, home):
    picker = Picker(policy, home)
    origin = time.perf_counter() + 0.3

    async def send(event):
        offset, idx, body, short = event
        when = origin + offset
        await asyncio.sleep(max(0, when - time.perf_counter()))
        started = time.perf_counter()
        pick, before = picker.begin(short, started)
        row = {
            "short": short,
            "index": idx,
            "offset_s": offset,
            "replica": pick,
            "outstanding": before,
            "target_s": when,
        }
        try:
            await one(*ports[pick], body, started, row, capture_hashes=True)
        finally:
            picker.end(pick, short, time.perf_counter())
        return row

    rows = await asyncio.gather(*(send(e) for e in events))
    if any(picker.loads) or picker.shorts:
        raise ValueError("unreleased routing counts")
    return rows


async def run(root):
    deadline = time.monotonic() + 360
    dirs = [root / "gpu6", root / "gpu7"]
    while not all((d / "address").is_file() for d in dirs):
        if time.monotonic() >= deadline:
            raise RuntimeError("replica startup deadline")
        await asyncio.sleep(1)
    ports = []
    for d in dirs:
        host, port = (d / "address").read_text().strip().split(":")
        ports.append((host, int(port)))
    events = corpus(json.loads((root / "plan.json").read_bytes()))
    warm = []
    for host, port in ports:
        for pick in (
            next(e for e in events if e[3]),
            next(e for e in events if not e[3]),
        ):
            row = {}
            await one(
                host, port, pick[2], time.perf_counter(), row, capture_hashes=True
            )
            warm.append(row)
    (root / "warmup.json").write_text(json.dumps(warm) + "\n")
    if any(r.get("error") or r.get("status") != 200 for r in warm):
        raise ValueError("warmup failed")
    order = json.loads((root / "coder-fleet-registration.json").read_bytes())["arms"]

    def record_shapes(label):
        if not os.environ.get("CODER_GAP_BUDGET"):
            return
        for idx, (host, port) in enumerate(ports):
            with urlopen(f"http://{host}:{port}/metrics", timeout=5) as response:
                shape = json.load(response)["batch_shapes"]
            (root / f"shapes-{label}-{idx}.json").write_text(json.dumps(shape) + "\n")

    record_shapes("warm")
    for i, policy in enumerate(order):
        rows = await arm(events, ports, policy, int(i >= len(order) // 2))
        (root / f"arm-{i}-{policy}.json").write_text(json.dumps(rows) + "\n")
        record_shapes(str(i))
        print(i, policy, len(rows), flush=True)


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    try:
        asyncio.run(run(folder))
    finally:
        (folder / "finished").touch()
