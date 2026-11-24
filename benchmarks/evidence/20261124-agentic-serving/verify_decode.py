"""Verify exact prompt IDs and stream lengths before comparing paired server starts."""

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import median


def q95(values):
    vals = sorted(values)
    return vals[math.ceil(0.95 * len(vals)) - 1]


def summarize(directory):
    raw = json.loads((directory / "fixed.json").read_text())
    if len(raw) != 10:
        raise ValueError("expected four identity requests and six paired waves")
    identity = {}
    cells = {}
    for trial in raw:
        history = trial["history"]
        if trial["phase"] == "identity":
            r = trial["result"]
            if not r.get("prompt_ids_sha256") or len(r.get("generated_ids", [])) != 64:
                raise ValueError("identity probe missing token IDs")
            identity[(history, r["name"])] = r
            continue
        reader, writer = trial["results"]
        if not reader["content_ns"][0] < writer["sent_ns"] < reader["content_ns"][-1]:
            raise ValueError("streams did not actually overlap")
        for r in trial["results"]:
            key = history, r["name"]
            t = identity[key]
            if (
                r["usage"]["prompt_tokens"] != t["usage"]["prompt_tokens"]
                or r["finish"] != "length"
                or r["usage"]["completion_tokens"] != 64
                or r["usage"]["completion_tokens"] != t["usage"]["completion_tokens"]
            ):
                raise ValueError("mismatched lengths or incomplete stream")
            gaps = [(b - a) / 1e6 for a, b in itertools.pairwise(r["content_ns"])]
            cells.setdefault(key, []).append(
                (r["tpot_ms"], gaps, r["schedule_error_ms"], r["input_sha256"])
            )
    if len(cells) != 4 or any(len(v) != 3 for v in cells.values()):
        raise ValueError("missing measured cells")
    return identity, cells


def compare(a, b):
    ident_a, rows_a = summarize(a)
    ident_b, rows_b = summarize(b)
    cells = []
    for key, rows in rows_a.items():
        other = rows_b[key]
        orig, ref = ident_a[key], ident_b[key]
        if (
            orig["prompt_ids_sha256"] != ref["prompt_ids_sha256"]
            or len({row[3] for row in rows + other}) != 1
        ):
            raise ValueError("submitted request or rendered token IDs differ")
        times = [median(row[0] for row in group) for group in [rows, other]]
        gaps = [q95(g for row in group for g in row[1]) for group in [rows, other]]
        cells.append(
            {
                "history": key[0],
                "stream": key[1],
                "prompt_tokens": orig["usage"]["prompt_tokens"],
                "identity_output_ids_equal": orig["generated_ids"]
                == ref["generated_ids"],
                "tpot_p50_ms_riley_vllm": times,
                "tpot_ratio": times[0] / times[1],
                "content_gap_p95_ms_riley_vllm": gaps,
                "gap_ratio": gaps[0] / gaps[1],
                "largest_arrival_drift_ms": max(abs(row[2]) for row in rows + other),
            }
        )
    return {"riley": a.name, "vllm": b.name, "cells": cells}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--second-riley", default="riley-d")
    p.add_argument("--first-riley", default="riley-a")
    args = p.parse_args()
    pairings = [(args.first_riley, "vllm-b"), (args.second_riley, "vllm-c")]
    pairs = [compare(args.directory / a, args.directory / b) for a, b in pairings]
    print(
        json.dumps(
            {
                "pairs": pairs,
                "independent_launch_pairs": len(pairs),
                "scope": "descriptive paired screen, 3 waves per launch; independent-launch population upper bound unestablished",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
