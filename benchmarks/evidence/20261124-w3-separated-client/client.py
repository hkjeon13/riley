"""Replay the Riley plan without running GPU ownership probes in the client."""

import asyncio
import json
from pathlib import Path
import sys

from profile_workload import save
from replay import load_cell


if __name__ == "__main__":
    path = Path(sys.argv[1])
    plan = json.loads(path.read_bytes())
    for label, cell in plan["cells"].items():
        result = asyncio.run(load_cell(sys.argv[2], cell))
        save(path.parent / f"{label}.json", result)
        print(label, result["statistics"]["ttft_p95_ms"], flush=True)
