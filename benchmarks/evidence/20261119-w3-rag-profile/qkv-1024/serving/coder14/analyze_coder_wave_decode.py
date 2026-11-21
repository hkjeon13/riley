"""Recalculate the Coder-14B wave decode guard from the original matched requests."""

from argparse import ArgumentParser
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import median

from compare_coder import EXPECTED_CELLS, compare
from compare_profile_arms import arm, digest, ratio_percent, read, require


FIELDS = ("request_index", "input_sha256", "prompt_tokens", "output_sha256", "completion_tokens", "finish")
REFERENCE = "712390666d906f9df399f95870037cb4b5112eb6b8607bd3834f661e579c442f"


def ranked_tpot(rows):
    return sorted(((row["e2e"] - row["ttft"]) * 1000 / max(row["completion_tokens"] - 1, 1),
                   row["request_index"]) for row in rows)


def decode_pair(before, after):
    old, new = (ranked_tpot(rows) for rows in (before, after))
    base, trial = (rank[len(rank) // 2] for rank in (old, new))
    old_by_request = {index: tpot for tpot, index in old}
    differences = [tpot - old_by_request[index] for tpot, index in new]
    return {"p50_ms": {"control": base[0], "candidate": trial[0]},
            "p50_request": {"control": base[1], "candidate": trial[1]},
            "p50_change_percent": ratio_percent(trial[0], base[0]),
            "paired_change_median_ms": median(differences),
            "slower_requests": sum(value > 0 for value in differences)}


def send_span(rows):
    sent = [row["sent_at"] for row in rows]
    return (max(sent) - min(sent)) * 1000


def wave_pair(order, name, runs):
    control, candidate = f"riley-{order}", f"qkv-{order}"
    base_receipt, base_plan, base_cells = runs[control]
    trial_receipt, trial_plan, trial_cells = runs[candidate]
    spec = base_plan["cells"][name]
    width = spec["concurrency"]
    require(trial_plan["cells"][name] == spec and spec["requests"] == 40 and width == spec["warmup"] == 8,
            "wave configuration changed")
    (base_stats, before), (trial_stats, after) = base_cells[name], trial_cells[name]
    require(all(all(a[field] == b[field] for field in FIELDS) for a, b in zip(before, after, strict=True)),
            "request inputs or outputs differ")
    for rows in (before, after):
        require(all(all(isfinite(row[key]) for key in ("start", "sent_at", "ttft", "e2e"))
                    and row["start"] <= row["sent_at"] <= row["start"] + row["ttft"]
                    and 0 < row["ttft"] < row["e2e"] and row["completion_tokens"] > 1 for row in rows),
                "invalid request timing or completion count")
    overall = decode_pair(before, after)
    require(all(isclose(overall["p50_ms"][label], recorded["tpot_p50_ms"], rel_tol=1e-12)
                for label, recorded in (("control", base_stats), ("candidate", trial_stats))),
            "request timings do not reproduce the decode gate")
    waves = []
    for begin in range(0, len(before), width):
        stop = begin + width
        base, trial = before[begin:stop], after[begin:stop]
        omitted = decode_pair(before[:begin] + before[stop:], after[:begin] + after[stop:])
        waves.append({"wave": begin // width + 1,
                      "first_request": base[0]["request_index"],
                      "control_send_span_ms": send_span(base), "candidate_send_span_ms": send_span(trial),
                      "decode": decode_pair(base, trial),
                      "overall_p50_change_without_wave_percent": omitted["p50_change_percent"]})
    return {"matched_requests": len(before), "overall": overall, "waves": waves,
            "raw_sha256": {control: base_receipt["results"][name], candidate: trial_receipt["results"][name]}}


def analyze(root):
    report = root / "verified.json"
    require((root / "campaign.exit").read_text().strip() == "0" and digest(report) == REFERENCE
            and json.loads(json.dumps(compare(root))) == read(report), "verified Coder-14B campaign differs")
    runs = {mode: arm(root, "coder14", mode, expected_cells=EXPECTED_CELLS)
            for mode in ("riley-first", "qkv-first", "qkv-second", "riley-second")}
    return {"source_receipt_sha256": REFERENCE, "remote_stage": root.name,
            "cells": {name: {order: wave_pair(order, name, runs) for order in ("first", "second")}
                      for name in ("wave-sync", "wave-50ms")}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(analyze(parser.parse_args().root), indent=2, allow_nan=False))
