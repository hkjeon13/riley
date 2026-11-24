"""Verify reversed 32K serving comparison from paired HTTP and process receipts."""

import argparse
import itertools
import json
import math
import re
from pathlib import Path

from verify_long_tile import read, receipt


def describe(row, expected):
    events = row["content_ns"]
    gaps = [(b - a) / 1e6 for a, b in itertools.pairwise(events)]
    if (
        not gaps
        or events[0] < row["sent_ns"]
        or row["done_ns"] < events[-1]
        or min(gaps) < 0
    ):
        raise ValueError("invalid content event timings")
    usage = row["usage"]
    ok = row["status"] == 200 and row["valid"] and bool(row["answer"].strip())
    if expected:
        ok &= re.findall(r"\d+", row["answer"]) == expected
    return {
        "valid": ok,
        "input": usage["prompt_tokens"],
        "output": usage["completion_tokens"],
        "cached": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "first_ms": (events[0] - row["sent_ns"]) / 1e6,
        "last_ms": (events[-1] - row["sent_ns"]) / 1e6,
        "p95_content_gap_ms": sorted(gaps)[math.ceil(0.95 * len(gaps)) - 1],
        "max_content_gap_ms": max(gaps),
    }


def compare(root):
    left = receipt(root / "page", "page-v1")
    right = receipt(root / "tile", "fixed-tile-unsplit-v1")
    if left != right:
        raise ValueError("different GPU, model launch settings or binary")
    arms = {k: read(root / k / "holdout.json") for k in ("page", "tile")}
    overlaps = {}
    for k, arm in arms.items():
        events = arm["final_mixed"]["content_ns"]
        overlaps[k] = events[0] <= arm["mixed_prefill"]["sent_ns"] < events[-1]
    answers = {
        "final_solo": [],
        "cold": ["486273", "705148"],
        "continuation": ["486273", "705148"],
        "mixed_prefill": ["230961", "874215"],
        "final_mixed": [],
    }
    cases = {}
    for name, expected in answers.items():
        if arms["page"][name]["input_sha256"] != arms["tile"][name]["input_sha256"]:
            raise ValueError("different request")
        pair = {k: describe(arm[name], expected) for k, arm in arms.items()}
        pair["same_output_text"] = (
            arms["page"][name]["answer"] == arms["tile"][name]["answer"]
        )
        cases[name] = pair
    matched = all(
        c["page"]["input"] == c["tile"]["input"]
        and c["page"]["cached"] == c["tile"]["cached"]
        for c in cases.values()
    )
    reader = cases["final_mixed"]
    ratio = {
        k: reader["tile"][k] / reader["page"][k]
        for k in ("p95_content_gap_ms", "max_content_gap_ms", "last_ms")
    }
    faster = all(
        cases[k]["tile"]["first_ms"] < cases[k]["page"]["first_ms"]
        for k in ("cold", "mixed_prefill")
    )
    return {
        "physical_gpu": left[1],
        "binary_sha256": left[2],
        "cases": cases,
        "overlaps": overlaps,
        "both_released": True,
        "matched_inputs_and_cache": matched,
        "mixed_reader_tile_over_page": ratio,
        "passed_registered_gates": all(overlaps.values())
        and matched
        and all(c[k]["valid"] for c in cases.values() for k in ("page", "tile"))
        and faster
        and max(ratio.values()) <= 1.05,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(compare(p.parse_args().root), indent=2))
