"""Confirm two orders on each GPU and compare both rounds at the original thresholds."""

import json
from pathlib import Path
import sys

from check import IDENTITY, COUNTS, arm, compare, require


def launch_order(folder, index):
    order = [row.split()[1] for row in (folder / "campaign.log").read_text().splitlines()
             if row.startswith("START ") and row.split()[2] == str(index)]
    require(len(order) == 2 and set(order) == {"0", "prefill-only"}, "missing same-GPU launch order")
    return order


def report(original, reverse):
    results = {}
    rounds = (compare(original), compare(reverse))
    for index in (6, 7):
        old, new = (row["gpus"][str(index)] for row in rounds)
        require(old["scope"] == new["scope"], "execution scope changed across rounds")
        a, b = (launch_order(folder, index) for folder in (original, reverse))
        require(a == b[::-1], "order did not reverse on the same hardware")
        for mode, selected in (("0", None), ("prefill-only", 0)):
            name = f"private-short-d{mode}-control-gpu{index}"
            old_raw, new_raw = (arm(folder / name, selected)[1] for folder in (original, reverse))
            require(all(all(all(x[key] == y[key] for key in IDENTITY)
                            for x, y in zip(old_raw[cell]["rows"], new_raw[cell]["rows"], strict=True))
                        for cell in COUNTS), "outputs changed across rounds")
        results[str(index)] = {"original_order": a, "reverse_order": b,
                               "original": old, "reverse": new}
    return {"status": "verified", "original_evidence": str(original),
            "reverse_evidence": str(reverse), "gpus": results}


if __name__ == "__main__":
    print(json.dumps(report(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
