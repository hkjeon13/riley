"""Capture both historical long-context corpora through the normal serving endpoint."""
from argparse import ArgumentParser
from pathlib import Path

from long_accuracy import capture, read_workload


p = ArgumentParser(description=__doc__)
p.add_argument("fixtures", type=Path)
p.add_argument("address")
p.add_argument("model")
p.add_argument("out", type=Path)
p.add_argument("server_log", type=Path)
args = p.parse_args()
holdout = args.fixtures / "20261119-w4-holdout/holdout.json.gz"
for label, workload in (("long", None), ("holdout", holdout)):
    print(label, flush=True)
    capture(args.address, args.model, args.out / f"{label}.json", read_workload(workload))
