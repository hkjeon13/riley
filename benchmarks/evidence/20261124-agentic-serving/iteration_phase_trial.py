"""Bracket the measured HTTP work with idle snapshots and exact trace byte offsets."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from coder_prefill_trace import trial


def snapshot(address, folder, name):
    with urlopen(f"http://{address}/metrics", timeout=10) as response:
        metrics = json.load(response)
    if metrics["active_requests"] or metrics["waiting_requests"]:
        raise ValueError("workload boundary had active requests")
    (folder / f"{name}.json").write_text(
        json.dumps(
            {"log_offset": (folder / "server.log").stat().st_size, "metrics": metrics}
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("directory", type=Path)
    parser.add_argument("frozen", type=Path)
    args = parser.parse_args()
    snapshot(args.address, args.directory, "before")
    trial(args.address, args.directory, args.frozen)
    snapshot(args.address, args.directory, "after")
