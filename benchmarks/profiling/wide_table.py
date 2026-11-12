#!/usr/bin/env python3
"""Turn `gemm_anchor_search K N wide` outputs into A100 equivalence-table entries.

usage: wide_table.py w-K-N.txt...
Prints `{80, K, N, 257, 512, ...}` and `{80, K, N, 513, 1024, ...}` with the
fastest anchor-equivalent configuration at 512 and 1,024 rows, skipping a range
whose fastest member is the anchor itself (the anchor runs there anyway).
"""
import re
import sys

for path in sys.argv[1:]:
    text = open(path).read()
    shape = re.search(r"shape (\d+) -> (\d+): \d+ configurations, anchor \{([^}]*)\}", text)
    if not shape:
        print(f"    // {path}: no result", file=sys.stderr)
        continue
    k, n, anchor = shape.group(1), shape.group(2), shape.group(3)
    best = dict(re.findall(r"best m=(\d+) \{([^}]*)\}", text))
    for low, high, rows in ((257, 512, "512"), (513, 1024, "1024")):
        config = best.get(rows)
        if config is None or config == anchor:
            continue
        print(f"    {{80, {k}, {n}, {low}, {high}, {config}}},")
