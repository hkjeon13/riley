"""Run API checks and three expanded closed-loop RAG cells against one image."""

from argparse import ArgumentParser
import os
from pathlib import Path
import shlex
import subprocess
import sys


REQUESTS = 256
SEEDS = (1, 2, 3)


def scenarios():
    sys.path.insert(0, os.environ["RILEY_W5_PERF_SOURCE"])
    from compare import SCENARIOS

    options = shlex.split(SCENARIOS["rag-c8"][1])
    options[options.index("--requests") + 1] = str(REQUESTS)
    seeds = reversed(SEEDS) if os.getenv("RILEY_RAG_PAIR") == "second" else SEEDS
    return {f"rag-c8-seed{seed}": (options, seed) for seed in seeds}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tools", action="store_true", required=True)
    args = parser.parse_args()
    harness = Path(os.environ["RILEY_W5_PERF_HARNESS"])
    client = [sys.executable, "-B"]
    subprocess.run([*client, str(harness / "api_check.py"), args.address, args.model,
                    "--out", str(args.out), "--tools"], check=True, timeout=600)
    for label, (options, seed) in scenarios().items():
        root = args.out.parent
        subprocess.run([*client, str(harness / "load.py"), args.address, args.model, *options,
                        "--seed", str(seed), "--out", str(root / f"{label}.summary.json"),
                        "--records", str(root / f"{label}.records.json")], check=True, timeout=1200)
        print(f"RAG_CELL_COMPLETED {label}", flush=True)


if __name__ == "__main__":
    main()
