"""Replay the original C32 bodies at the pinned closed-loop write offsets."""

import asyncio
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
import time
from urllib.request import urlopen

from load import one, percentile, summarize
from workload import make_workloads, require, run_cell

REFERENCES = {"6": ("off-warm-gpu6", "296af425364e1586ec6a1e46ec6ff3c3629b473124628c55863e72926f25f19a"),
              "7": ("private-gpu7", "318c4144a04522a422a3693ac20321e303cadecc5ef6cd9f219395d0e670970a")}


def make_schedules(prior):
    fixtures = make_workloads()["short"]
    plans = {}
    for index, (case, expected) in REFERENCES.items():
        raw = (prior / case / "workload.json").read_bytes()
        require(sha256(raw).hexdigest() == expected, "reference changed")
        rows = json.loads(raw)["short"]["rows"]
        require(len(rows) == len(fixtures) == 96, "incomplete C32 reference")
        for row, body in zip(rows, fixtures, strict=True):
            require(row["input_sha256"] == sha256(json.dumps(body).encode()).hexdigest()
                    and row["error"] is False and isfinite(row["sent_at"]) and row["sent_at"] >= row["start"],
                    "invalid reference request identity or timestamps")
        first = min(row["sent_at"] for row in rows)
        offsets = [round((row["sent_at"] - first) * 1e9) for row in rows]
        require(min(offsets) == 0 and max(offsets) > 0, "invalid replay duration")
        plans[index] = {"reference_case": case, "reference_workload_sha256": expected, "offsets_ns": offsets}
    return plans


async def sleep_until(deadline):
    await asyncio.sleep(max(0, deadline - time.perf_counter() - .004))
    while time.perf_counter() < deadline:
        await asyncio.sleep(0)


async def replay(address, bodies, plan):
    host, port = address.split(":")
    port = int(port)
    offsets = [value / 1e9 for value in plan["offsets_ns"]]
    order = sorted(range(len(offsets)), key=offsets.__getitem__)
    require(len(bodies) == len(order) == 96, "unexpected fixed C32 input size")
    rows = [{} for _ in offsets]
    connections = {}

    async def measure(index, connection, started):
        async with asyncio.timeout(75):
            await one(host, port, bodies[index], started, rows[index], capture_hashes=True,
                      capture_token_ids=True, connection=connection)

    try:
        # Establish the first burst in send order before the measurement clock starts.
        # Later sockets are opened near their own deadline so they cannot occupy
        # workers ahead of earlier requests.
        for index in order[:32]:
            connections[index] = await asyncio.wait_for(asyncio.open_connection(host, port, limit=1 << 22), 5)
        origin = time.perf_counter() + .030
        async with asyncio.TaskGroup() as tasks:
            for index in order:
                deadline = origin + offsets[index]
                if index not in connections:
                    await sleep_until(deadline - .001)
                    connections[index] = await asyncio.wait_for(asyncio.open_connection(host, port, limit=1 << 22), 5)
                await sleep_until(deadline)
                tasks.create_task(measure(index, connections[index], time.perf_counter()))
                await asyncio.sleep(0)
        wall = time.perf_counter() - origin
    finally:
        for _reader, writer in connections.values():
            writer.close()

    stats = summarize(rows, wall, {"cell": "short", "arrival": "fixed", "reference_case": plan["reference_case"]})
    require(all(row.get("error") is False for row in rows), f"fixed replay request failed: {stats['other_failures']}")
    first = min(row["sent_at"] for row in rows)
    errors = [1000 * (row["sent_at"] - first - offset) for row, offset in zip(rows, offsets, strict=True)]
    for row, error in zip(rows, errors, strict=True):
        row["relative_write_error_ms"] = error
        del row["gaps"]
    stats.update(write_error_p95_ms=percentile([abs(num) for num in errors], .95),
                 first_32_write_error_p95_ms=percentile([abs(errors[index]) for index in order[:32]], .95),
                 write_order_matches=(sorted(order, key=lambda i: rows[i]["sent_at"]) == order))
    return {"rows": rows, "statistics": stats}


def emit(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run(address, output):
    index = output.parent.name.rsplit("-gpu", 1)[1]
    plans = json.loads((output.parent.parent / "schedules.json").read_bytes())
    bodies = make_workloads()
    evidence = {}
    for name in ("startup", "prime", "rag", "short"):
        if name == "short":
            cell = asyncio.run(replay(address, bodies[name], plans[index]))
        else:
            cell = asyncio.run(run_cell(address, bodies[name], name)) if name in bodies else {}
        with urlopen(f"http://{address}/metrics", timeout=5) as response:
            cell["metrics"] = json.load(response)
        evidence[name] = cell
    emit(output, evidence)
    rag = evidence["rag"]["statistics"]
    short = evidence["short"]["statistics"]
    require(rag["send_error_p95_ms"] <= 5 and short["write_error_p95_ms"] <= 5
            and short["first_32_write_error_p95_ms"] <= 5 and short["write_order_matches"], "arrival fidelity failed")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "plan":
        emit(Path(sys.argv[3]), make_schedules(Path(sys.argv[2])))
    else:
        run(sys.argv[1], Path(sys.argv[2]))
