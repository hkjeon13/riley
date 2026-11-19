"""Use the released API checks and load generator on one benchmark container."""

from argparse import ArgumentParser
from pathlib import Path
import os
import shlex
import subprocess
import sys


CELLS = (
    ("closed-c1", "closed-c1", 1),
    ("closed-c32", "closed-c32", 1),
    ("rag-c8-seed1", "rag-c8", 1),
    ("rag-c8-seed2", "rag-c8", 2),
    ("reason-c32", "reason-c32", 1),
    ("shared-c8", "shared-c8", 1),
)


def scenarios():
    source = Path(os.environ["RILEY_W5_PERF_SOURCE"]).resolve(strict=True)
    sys.path.insert(0, str(source))
    from compare import SCENARIOS
    return {label: (shlex.split(SCENARIOS[key][1]), seed) for label, key, seed in CELLS}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tools", action="store_true", required=True)
    args = parser.parse_args()
    harness = Path(os.environ["RILEY_W5_PERF_HARNESS"]).resolve(strict=True)
    client = [sys.executable, "-B"]
    subprocess.run([*client, str(harness / "api_check.py"), args.address, args.model,
                    "--out", str(args.out), "--tools"], check=True, timeout=600)
    results = args.out.parent
    for label, (options, seed) in scenarios().items():
        subprocess.run([*client, str(harness / "load.py"), args.address, args.model, *options,
                        "--seed", str(seed), "--out", str(results / f"{label}.summary.json"),
                        "--records", str(results / f"{label}.records.json")], check=True, timeout=1200)


if __name__ == "__main__":
    main()
