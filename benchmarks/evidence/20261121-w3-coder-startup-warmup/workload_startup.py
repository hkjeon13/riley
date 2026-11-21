"""Sample ready-state shape counters before sending the original RAG and C32 traffic."""

import asyncio
import json
from pathlib import Path
import sys
from urllib.request import urlopen

from workload import make_workloads, require, run_cell


def metrics(address):
    with urlopen(f"http://{address}/metrics", timeout=5) as response:
        return {"metrics": json.load(response)}


def run(address, output):
    evidence = {"startup": metrics(address)}
    for name, bodies in make_workloads().items():
        cell = asyncio.run(run_cell(address, bodies, name))
        cell.update(metrics(address))
        evidence[name] = cell
        print(name, cell["statistics"]["ttft_p50_ms"], flush=True)
    with output.open("x") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    lateness = evidence["rag"]["statistics"]["send_error_p95_ms"]
    require(lateness <= 5, f"fixed arrival replay send p95 {lateness:.3f}ms exceeded 5ms")


if __name__ == "__main__":
    run(sys.argv[1], Path(sys.argv[2]))
