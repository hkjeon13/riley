"""Decompose recorded short and RAG TTFT at the client write boundary."""

from argparse import ArgumentParser
import json
from math import isclose, isfinite
from pathlib import Path

from compare_allocator_performance import ORDER, arm
from verify_a100_candidate import digest, read, require


CELL_COUNTS = {"closed-c32": 192, "rag-c8-seed1": 64, "rag-c8-seed2": 64}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def split_timing(row):
    start, sent, latency = (row[name] for name in ("start", "sent_at", "ttft"))
    require(all(type(value) in (int, float) and isfinite(value) for value in (start, sent, latency)),
            "invalid request timing")
    to_write = sent - start
    require(0 <= to_write <= latency, "client write is outside the TTFT interval")
    return {"ttft_ms": latency * 1000, "start_to_write_ms": to_write * 1000,
            "write_to_first_visible_text_ms": (latency - to_write) * 1000}


def summarize(packaged, evidence, stage, phase, cell):
    cases, _ = arm(packaged, evidence, stage, phase)
    metrics, selected = cases[cell]
    source = stage / phase / "gpu-evidence" / f"{cell}.records.json"
    rows = read(source)
    require(len(rows) == CELL_COUNTS[cell] and len(rows) == len(selected), "need the complete measured cell")
    timing = {row["request_index"]: split_timing(row) for row in rows}
    ttft = {index: row["ttft_ms"] for index, row in timing.items()}
    ranked = sorted(ttft, key=ttft.get)
    rank = int(.95 * len(ranked))
    require(isclose(ttft[ranked[rank]], metrics["ttft_p95_ms"], abs_tol=1e-12),
            "stored TTFT p95 differs from the measured requests")
    p95 = {key: percentile([row[key] for row in timing.values()], .95) for key in timing[ranked[rank]]}
    return timing, {
        "requests": len(timing),
        "records_sha256": digest(source),
        "p95_ms": p95,
        "ttft_p95_rank_request": {"index": ranked[rank], **timing[ranked[rank]]},
        "top_10_ttft_request_indices": ranked[-10:],
    }


def compare(parent, candidate, left, right):
    before, after = parent["p95_ms"], candidate["p95_ms"]
    require(left.keys() == right.keys(), "measured request indices differ")
    differences = {index: right[index]["ttft_ms"] - left[index]["ttft_ms"] for index in left}
    with_parent_write = [left[index]["start_to_write_ms"] + right[index]["write_to_first_visible_text_ms"]
                         for index in left]
    effective_p95 = percentile(with_parent_write, .95)
    delta = after["ttft_ms"] - before["ttft_ms"]
    return {
        "delta_p95_ms": {key: after[key] - before[key] for key in before},
        "p95_delta_after_substituting_same_request_parent_start_to_write_ms": effective_p95 - before["ttft_ms"],
        "candidate_slower_requests": sum(change > 0 for change in differences.values()),
        "paired_ttft_delta_p50_ms": percentile(list(differences.values()), .5),
        "paired_ttft_delta_p95_ms": percentile(list(differences.values()), .95),
        "ttft_p95_delta_ms": delta,
        "top_10_ttft_index_overlap": len(set(parent["top_10_ttft_request_indices"]) &
                                         set(candidate["top_10_ttft_request_indices"])),
    }


def audit(packaged, evidence, stage, phases, cell):
    runs = {name: summarize(packaged, evidence, stage, name, cell) for name in phases}
    comparisons = {}
    for order in ("first", "second"):
        parent, candidate = f"parent-{order}", f"packaged-{order}"
        if {parent, candidate} <= runs.keys():
            left, before = runs[parent]
            right, after = runs[candidate]
            comparisons[order] = compare(before, after, left, right)
    return {"scope": "Exploratory audit of captured TTFT from the matched packaged image campaign.",
            "cell": cell,
            "interpretation": "Write-to-first-visible-text includes socket and server processing, GPU work, and SSE delivery.",
            "arms": {name: run[1] for name, run in runs.items()}, "pairs": comparisons}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("packaged", "evidence", "stage"):
        parser.add_argument(name, type=Path)
    parser.add_argument("phases", nargs="+", choices=ORDER)
    parser.add_argument("--cell", choices=CELL_COUNTS, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.packaged, args.evidence, args.stage, args.phases, args.cell), indent=2, allow_nan=False))
