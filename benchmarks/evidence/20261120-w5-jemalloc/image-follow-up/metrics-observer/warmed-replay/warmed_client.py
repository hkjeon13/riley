"""Run one warmup before repeating the metrics comparison in reverse order."""

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
import sys

import observer_client


WARMUP = ("warmup", False)
PHASES = (("scrape-first", True), ("baseline-first", False), ("baseline-second", False), ("scrape-second", True))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--tools", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    api = Path(os.environ["RILEY_W5_PERF_HARNESS"]) / "api_check.py"
    subprocess.run([sys.executable, "-B", str(api), args.address, args.model, "--tools", "--out", str(args.out)],
                   check=True, timeout=600)
    observer_client.PHASES = (WARMUP, *PHASES)
    observer_client.capture(args.address, args.model, args.out.parent)
