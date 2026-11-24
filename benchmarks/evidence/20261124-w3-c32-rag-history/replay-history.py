"""Replay C32 at the same server age after RAG or idle."""
import asyncio
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from profile_workload import save
from replay import load_cell

SHORT = "c32-riley-arrivals"
START_AFTER_SECONDS = 115.0
RAG_LEAD_SECONDS = 4.0


async def replay(path, address):
    cells = json.loads(path.read_bytes())["cells"]
    os.sched_setaffinity(0, {int(os.environ["W3_CLIENT_CPU"])})
    start = perf_counter()
    finish = {}
    launch = {}
    for label, cell in cells.items():
        if label == SHORT:
            remaining = start + START_AFTER_SECONDS - perf_counter()
            if remaining < 0:
                raise RuntimeError("RAG prefix exceeded the fixed C32 release offset")
            await asyncio.sleep(remaining)
        elif not launch:
            await asyncio.sleep(RAG_LEAD_SECONDS)
        launch[label] = perf_counter() - start
        result = await load_cell(address, cell)
        finish[label] = perf_counter() - start
        save(path.parent / f"{label}.json", result)
        print(label, result["statistics"]["ttft_p95_ms"], flush=True)
    save(path.parent / "history.json", {
        "short_start_target_s": START_AFTER_SECONDS, "rag_start_target_s": RAG_LEAD_SECONDS,
        "launch_s": launch, "finish_s": finish,
    })


if __name__ == "__main__":
    asyncio.run(replay(Path(sys.argv[1]), sys.argv[2]))
