"""Run the original RAG and short workloads, then repeat the same short cell."""

import asyncio
import json
from pathlib import Path
import sys
from urllib.request import urlopen

from workload import make_workloads, require, run_cell


def run(address, output):
    bodies = make_workloads()
    evidence = {}
    for name, original in (("prime", "prime"), ("rag", "rag"), ("short", "short"), ("warm_short", "short")):
        cell = asyncio.run(run_cell(address, bodies[original], original))
        with urlopen(f"http://{address}/metrics", timeout=5) as response:
            cell["metrics"] = json.load(response)
        evidence[name] = cell
        print(name, cell["statistics"]["ttft_p50_ms"], flush=True)
    with output.open("x") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    lateness = evidence["rag"]["statistics"]["send_error_p95_ms"]
    require(lateness <= 5, f"fixed arrival replay send p95 {lateness:.3f}ms exceeded 5ms")


if __name__ == "__main__":
    run(sys.argv[1], Path(sys.argv[2]))
