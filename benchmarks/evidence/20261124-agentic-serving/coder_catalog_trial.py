"""Time dependent tool calls when an authorized report API returns a catalog."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from agent_tool_projection import project_exact_records
from agent_workflow import fingerprint, fixture_tool_result, replay
from coder_pressure import snapshot


def catalog(seed, docs):
    rows = [
        {
            "id": f"aux-{seed}-{i:02d}",
            "p95_ms": 100 + (seed + i) % 71,
            "statistic": "service response latency",
            "notes": ["daily archive"],
        }
        for i in range(62)
    ]
    rows.insert(3 + seed % 19, docs[0])
    rows.insert(38 + seed % 23, docs[1])
    return rows


def tool_output(mode, reader, seed, receipts, index, call, docs):
    if index == 0 or reader:
        return fixture_tool_result(index, docs)
    args = json.loads(call["function"]["arguments"])
    ids = [args["first"], args["second"]]
    rows = catalog(seed, docs)
    retained = (
        rows
        if mode == "full"
        else project_exact_records(rows, key="id", requested_ids=ids)
    )
    receipts.append(
        {
            "source_sha256": fingerprint(rows),
            "result_sha256": fingerprint(retained),
            "source_count": len(rows),
            "result_count": len(retained),
            "requested_ids": ids,
        }
    )
    return retained


def wave(
    address, folder, mode, size, base, *, sample=snapshot, application_targets=False
):
    def run(index, origin):
        reader = index % 4 == 0
        seed = base + index
        scheduled = origin + 30_000_000 * index
        time.sleep(max(0, (scheduled - time.monotonic_ns()) / 1e9))
        receipts = []
        work = replay(
            address,
            "m",
            seed,
            2,
            0 if reader else 0.5,
            f"job-{seed}",
            history=0 if reader else 200,
            **(
                {"dispatch_fixed_tools": True}
                if application_targets and not reader
                else {}
            ),
            execute_tool=partial(tool_output, mode, reader, seed, receipts),
        )
        return {
            "index": index,
            "role": "reader" if reader else "target",
            "scheduled_ns": scheduled,
            "work": work,
            "preview": {},
            "decision": {},
            "tool_result_receipts": receipts,
        }

    before = sample(address, folder / "server.log")
    with ThreadPoolExecutor(size) as clients:
        origin = time.monotonic_ns() + 100_000_000
        jobs = list(clients.map(partial(run, origin=origin), range(size)))
    drained = time.monotonic_ns()
    return {
        "size": size,
        "base": base,
        "origin_ns": origin,
        "drained_ns": drained,
        "before": before,
        "after": sample(address, folder / "server.log"),
        "jobs": jobs,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("folder", type=Path)
    p.add_argument("mode", choices=["full", "selected"])
    args = p.parse_args()
    reg = json.loads(
        Path(__file__).with_name("coder-catalog-registration.json").read_text()
    )
    warm = replay(args.address, "m", 5091, 2, 0, "warm")
    (args.folder / "warm.json").write_text(json.dumps(warm) + "\n")
    waves = [wave(args.address, args.folder, args.mode, *w) for w in reg["waves"]]
    (args.folder / "waves.json").write_text(json.dumps(waves) + "\n")
