"""Recompute the paired prefill-chunk results and enforce output and latency checks."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import runpy

BASE = Path(__file__).resolve().parent.parent / "20261124-w3-package-coder14/verify.py"
CHECKS = runpy.run_path(str(BASE))
digest = CHECKS["digest"]
read = CHECKS["read"]
records = CHECKS["records"]
require = CHECKS["require"]
reservations = CHECKS["reservations"]
IDENTITY = (*CHECKS["IDENTITY"], "output_sha256")
EXPECTED = {6: (512, 1024), 7: (1024, 512)}
RAG = ("rag-c8-s1", "rag-c8-s2")


def check_order(stage):
    events = re.findall(r"^(START|END) gpu([67]) chunk-(1024|512) (\S+?)(?: exit=(\d+))?$",
                        (stage / "campaign.log").read_text(), re.MULTILINE)
    require(len(events) == 8, "expected four starts and four endings")
    timestamps = {}
    for phase, index, chunk, timestamp, exit_code in events:
        require(exit_code == ("0" if phase == "END" else ""), "an arm failed")
        key = (int(index), int(chunk), phase)
        require(key not in timestamps, f"duplicate event {key}")
        timestamps[key] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    for index, (first, second) in EXPECTED.items():
        require(timestamps[index, first, "START"] <= timestamps[index, first, "END"]
                <= timestamps[index, second, "START"] <= timestamps[index, second, "END"],
                f"GPU {index} did not follow the prescribed run order")


def committed_shapes(snapshots, cells):
    before = snapshots["before"]
    output = {}
    for name in cells:
        after = snapshots[name]
        require(before.keys() == after.keys(), f"configured buckets changed in {name}")
        diff = {}
        for rows, observed in after.items():
            count, elapsed = (observed[i] - before[rows][i] for i in (0, 1))
            require(count >= 0 and elapsed >= 0 and (count > 0) == (elapsed > 0),
                    f"invalid committed counters for {name} bucket {rows}")
            if count:
                diff[rows] = (count, elapsed)
        require(diff, f"no GPU work for {name}")
        output[name] = {
            "steps": sum(count for count, _ in diff.values()),
            "gpu_ms": sum(ns for _, ns in diff.values()) / 1_000_000,
            "steps_by_bucket": {row: count for row, (count, _) in diff.items()},
        }
        before = after
    return output


def arm(stage, index, size, cells, holder):
    out = stage / f"gpu{index}/chunk-{size}"
    receipt = read(out / "receipt.json")
    plan = read(out / "plan.json")
    require(receipt["status"] == "passed" and receipt["chunk"] == size
            and receipt["gpu"] == CHECKS["GPUS"][index] and receipt["holder_pid"] == holder["pid"]
            and receipt["common_kv_blocks"] == 2576 and receipt["kv_cache_bytes"] == 8103395328
            and plan["cells"] == cells and receipt["shutdown_exit"] == 0,
            f"invalid GPU {index} chunk {size} receipt")
    for field, key in {
        "image_id": "IMAGE", "source_revision": "REVISION", "riley_binary_sha256": "BINARY",
        "cpu_report_sha256": "CPU", "checkpoint_sha256": "CHECKPOINT",
    }.items():
        require(receipt[field] == CHECKS[key], f"invalid image or model provenance in {out}")
    shutdown = read(out / "shutdown.json")
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and all(value == 0 for value in shutdown["allocation"].values()),
            "Riley retained resources after shutdown")
    argv = plan["server"]
    chunk_pos = argv.index("--prefill-chunk-tokens") + 1
    require(argv[chunk_pos] == str(size) and argv[argv.index("--batch-token-budget") + 1] == "1024",
            f"requested chunk {size} was not the only intended limit change")
    settings = [*argv[:chunk_pos], *argv[chunk_pos + 1:]]
    shapes = out / "batch-shapes.json"
    holdout = out / "token-id-holdout.json"
    require(digest(shapes) == receipt["batch_shapes_sha256"] and digest(holdout) == receipt["holdout_sha256"],
            "shape or token ID artifact changed")
    holdout_rows = read(holdout)
    require(len(holdout_rows) == 8 and all(
        all(re.fullmatch(r"[0-9a-f]{64}", row[field]) for field in (
            "input_sha256", "output_sha256", "prompt_ids_sha256", "token_ids_sha256"))
        and 0 < row["completion_tokens"] <= 64 and 0 < row["prompt_tokens"] < 8192
        for row in holdout_rows
    ), "token ID holdout is incomplete")
    return {
        "settings": settings, "cells": {name: records(out, name, spec, receipt) for name, spec in cells.items()},
        "shapes": committed_shapes(read(shapes), cells), "holdout": holdout_rows,
    }


def prompt_groups(rows):
    groups = {}
    for name, group in (
        ("under_2500_tokens", [row for row in rows if row["prompt_tokens"] < 2500]),
        ("2500_tokens_and_over", [row for row in rows if row["prompt_tokens"] >= 2500]),
    ):
        if group:
            values = sorted(row["ttft"] * 1000 for row in group)
            groups[name] = {"requests": len(values), "ttft_p50_ms": values[len(values) // 2],
                            "ttft_p95_ms": values[min(int(.95 * len(values)), len(values) - 1)]}
    return groups


def pair(control, candidate):
    a, baseline = control
    b, trial = candidate
    require(all(left["input_sha256"] == right["input_sha256"] for left, right in zip(a, b, strict=True)),
            "benchmark requests differ")
    exact = sum(all(left[key] == right[key] for key in IDENTITY) for left, right in zip(a, b, strict=True))
    ratios = {metric: trial[metric] / baseline[metric] for metric in CHECKS["METRICS"]}
    return {"requests": len(a), "exact_responses": exact,
            "reference": baseline, "chunk_512": trial, "chunk_512_over_reference": ratios,
            "ttft_by_prompt_length": {"reference": prompt_groups(a), "chunk_512": prompt_groups(b)}}


def verify(stage):
    require((stage / "campaign.exit").read_text().strip() == "0", "campaign failed")
    check_order(stage)
    cells = read(stage / "cells.json")
    require(tuple(cells) == ("closed-c32", "rag-c1", *RAG, "wave-sync", "wave-50ms"),
            "did not run the prescribed short, RAG and arrival workloads")
    reference = None
    by_gpu = {}
    for index, order in EXPECTED.items():
        holder = reservations(stage, index)
        arms = {size: arm(stage, index, size, cells, holder) for size in order}
        baseline, trial = arms[1024], arms[512]
        require(baseline["settings"] == trial["settings"], "server arguments besides chunk differ")
        if reference is None:
            reference = baseline["holdout"]
        require(reference == baseline["holdout"], "baseline token ID holdout changed between GPUs")
        summary = {name: pair(baseline["cells"][name], trial["cells"][name]) for name in cells}
        output_equal = all(row["requests"] == row["exact_responses"] for row in summary.values())
        latency = {name: row["chunk_512_over_reference"] for name, row in summary.items()}
        short = latency["closed-c32"]
        by_gpu[index] = {
            "cells": summary,
            "checks": {
                "performance_output_parity": output_equal,
                "token_id_holdout_parity": baseline["holdout"] == trial["holdout"],
                "short_p50_p95_and_throughput_within_2_percent": (
                    max(short["ttft_p50_ms"], short["ttft_p95_ms"]) <= 1.02 and short["output_tok_s"] >= 0.98),
                "decode_median_within_1_percent": all(row["tpot_p50_ms"] <= 1.01 for row in latency.values()),
                "rag_tail_improved_on_both_seeds": all(latency[name]["ttft_p95_ms"] < 1.0 for name in RAG),
            },
            "observed_batch_shapes": {size: trial["shapes"] if size == 512 else baseline["shapes"] for size in order},
        }
    return {"source_revision": CHECKS["REVISION"], "image_id": CHECKS["IMAGE"],
            "run_order": EXPECTED, "by_gpu": by_gpu}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage), indent=2, allow_nan=False))
