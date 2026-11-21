"""Decompose matched RAG tails by the observed prompt-token cohort."""

import json
from pathlib import Path
import sys

from audit_allocator_ttft import percentile, split_timing
from verify_a100_candidate import digest, read, require


BOUNDARY = 2500
SIDES = ("parent", "packaged")


def audit(root, receipt):
    cells = {}
    for label, cell in read(receipt)["cells"].items():
        pairs = {}
        for order in ("first", "second"):
            data = {}
            for side in SIDES:
                arm = f"{side}-{order}"
                path = root / arm / "gpu-evidence" / f"{label}.records.json"
                require(digest(path) == cell["request_timing"][arm]["records_sha256"], "RAG record changed")
                raw = read(path)
                require(len(raw) == 256, "incomplete RAG record")
                data[side] = {row["request_index"]: row for row in raw}
            left, right = (data[side] for side in SIDES)
            fields = ("input_sha256", "prompt_tokens")
            require(left.keys() == right.keys() and all(
                tuple(left[index][key] for key in fields) == tuple(right[index][key] for key in fields)
                for index in left), "cohort input alignment differs")
            timing = {side: {index: split_timing(row) for index, row in rows.items()}
                      for side, rows in data.items()}
            short = [index for index, row in left.items() if row["prompt_tokens"] < BOUNDARY]
            long = [index for index, row in left.items() if row["prompt_tokens"] >= BOUNDARY]
            groups = {}
            for name, indexes in (("shorter", short), ("longer", long)):
                require(indexes, "empty prompt-length cohort")
                group = {}
                for side in SIDES:
                    group[side] = {key: percentile([timing[side][index][key] for index in indexes], .95)
                                   for key in timing[side][indexes[0]]}
                deltas = [timing["packaged"][index]["ttft_ms"] - timing["parent"][index]["ttft_ms"]
                          for index in indexes]
                groups[name] = {"requests": len(indexes),
                                "prompt_tokens": [min(left[i]["prompt_tokens"] for i in indexes),
                                                  max(left[i]["prompt_tokens"] for i in indexes)],
                                "p95_ms": group, "paired_ttft_delta_p50_ms": percentile(deltas, .50),
                                "paired_ttft_delta_p95_ms": percentile(deltas, .95)}
            top = {}
            for side in SIDES:
                ranked = sorted(timing[side], key=lambda index: (timing[side][index]["ttft_ms"], index))
                tail = ranked[int(.95 * len(ranked)):]
                top[side] = {"requests": len(tail), "longer_prompts": len(set(tail) & set(long))}
            pairs[order] = {"cohorts": groups, "ttft_tail": top}
        cells[label] = pairs
    return {"scope": "Paired prompt-length cohorts from the same closed-loop RAG captures.",
            "source_receipt_sha256": digest(receipt), "longer_prompt_threshold_tokens": BOUNDARY,
            "cells": cells}


if __name__ == "__main__":
    stage, receipt = map(Path, sys.argv[1:])
    print(json.dumps(audit(stage, receipt), indent=2, allow_nan=False))
