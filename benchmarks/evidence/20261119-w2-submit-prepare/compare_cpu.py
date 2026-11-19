"""Repeat each local CPU control in alternating order and retain every sample."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import subprocess


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compare(base, candidate, rounds):
    order = ("base", "candidate", "candidate", "base") * rounds
    binaries = {"base": base, "candidate": candidate}
    runs = []
    for mode in order:
        output = subprocess.check_output([binaries[mode].resolve()], text=True)
        pairs = [line.split() for line in output.splitlines()]
        values = {case: float(value) for case, value in pairs}
        if not values or len(values) != len(pairs) or (runs and values.keys() != runs[0]["ns_per_call"].keys()):
            raise ValueError(f"missing or duplicate benchmark case: {mode}")
        runs.append({"mode": mode, "ns_per_call": values})

    medians = {}
    for name in runs[0]["ns_per_call"]:
        base, candidate = (median(r["ns_per_call"][name] for r in runs if r["mode"] == mode)
                           for mode in binaries)
        effects = []
        for i in range(0, len(runs), 4):
            first_base, first_candidate, second_candidate, second_base = (
                row["ns_per_call"][name] for row in runs[i:i+4]
            )
            effects.extend([first_candidate - first_base, second_candidate - second_base])
        medians[name] = {"baseline_median_ns": base, "candidate_median_ns": candidate,
                         "difference_ns": candidate - base,
                         "candidate_faster_pairs": sum(effect < 0 for effect in effects),
                         "paired_delta_ns": effects}
    return {"binary_sha256": {mode: digest(path) for mode, path in binaries.items()}, "runs": runs, "medians": medians}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    summary = compare(args.baseline, args.candidate, args.rounds)
    print(json.dumps(summary, indent=2, allow_nan=False))
