"""Replay repeated RAG document requests against one isolated Coder-14B server."""

import asyncio
import json
from pathlib import Path
import random
import sys
import time
from urllib.request import urlopen

from load import document, one, percentile, prompt, summarize


MODEL = "m"
WARM_BLOCKS = 640
REQUEST_COUNTS = {"prime": 4, "rag": 64, "short": 96}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_request(messages, limit):
    return {"model": MODEL, "temperature": 0, "max_tokens": limit, "messages": messages,
            "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True}


def make_workloads():
    docs = [f"Reference collection {i}:\n\n" + document(random.Random(900 + i), 1760) for i in range(4)]
    rng = random.Random(914)
    def rag(index):
        system = {"role": "system", "content": docs[index % len(docs)]}
        user = {"role": "user", "content": document(rng, 96) + "\nWrite a detailed explanation using the reference."}
        return make_request([system, user], 64)

    return {"prime": [rag(index) for index in range(REQUEST_COUNTS["prime"])],
            "rag": [rag(index) for index in range(REQUEST_COUNTS["rag"])],
            "short": [make_request([{"role": "user", "content": prompt(rng, 30)}], 64)
                      for _ in range(REQUEST_COUNTS["short"])]}


async def run_cell(address, bodies, name):
    host, port = address.split(":")
    rows = [{} for _ in bodies]

    async def submit(index, planned=None):
        await asyncio.wait_for(one(host, int(port), bodies[index], time.perf_counter(), rows[index],
                                   capture_hashes=True, capture_token_ids=True), timeout=75)
        require(rows[index].get("error") is False, f"{name}/{index} request failed: {rows[index]}")
        if planned is not None:
            rows[index]["send_lateness_ms"] = (rows[index]["sent_at"] - planned) * 1000

    start = time.perf_counter()
    if name == "rag":
        async def replay(index):
            planned = start + (index // 8) * 6 + (index % 8) * 0.050
            await asyncio.sleep(max(0, planned - time.perf_counter() - 0.006))
            while time.perf_counter() < planned:
                await asyncio.sleep(0)
            await submit(index, planned)
        await asyncio.gather(*(replay(index) for index in range(len(bodies))))
    else:
        queue = iter(range(len(bodies)))
        async def worker():
            for index in queue:
                await submit(index)
        await asyncio.gather(*(worker() for _ in range(1 if name == "prime" else 32)))
    wall = time.perf_counter() - start
    stats = summarize(rows, wall, {"cell": name})
    if name == "rag":
        stats["send_error_p95_ms"] = percentile([abs(row["send_lateness_ms"]) for row in rows], .95)
    for row in rows:
        del row["gaps"]
    return {"rows": rows, "statistics": stats}


def run(address, output):
    evidence = {}
    for name, bodies in make_workloads().items():
        evidence[name] = asyncio.run(run_cell(address, bodies, name))
        with urlopen(f"http://{address}/metrics", timeout=5) as response:
            evidence[name]["metrics"] = json.load(response)
        print(name, evidence[name]["statistics"]["ttft_p95_ms"], flush=True)
    with output.open("x") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    lateness = evidence["rag"]["statistics"]["send_error_p95_ms"]
    require(lateness <= 5, f"fixed arrival replay send p95 {lateness:.3f}ms exceeded 5ms")


if __name__ == "__main__":
    run(sys.argv[1], Path(sys.argv[2]))
