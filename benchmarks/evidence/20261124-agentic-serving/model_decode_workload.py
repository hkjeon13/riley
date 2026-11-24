"""Measure the frozen paired decode first, then two uncensored dependent workflows."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agent_workflow import replay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("directory", type=Path)
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    with (args.directory / "fixed-diagnostic.log").open("w") as log:
        fixed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("fixed_agent_decode.py")),
                args.address,
                args.fixtures,
                args.directory / "fixed.json",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    (args.directory / "fixed-exit.json").write_text(json.dumps(fixed.returncode))
    records = [
        replay(args.address, "m", seed, notes, 0.1, f"chain-{seed}", history=history)
        for seed, notes, history in ((761, 2, 0), (762, 96, 100))
    ]
    (args.directory / "natural.json").write_text(json.dumps(records) + "\n")
