"""Validate the completed W2/W3 matrix and measure the policy interaction."""

from argparse import ArgumentParser
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys


EVENTS = ("immediate", "after-submit")
CAPS = (32, 2)
CELLS = ("closed-c32", "rag-c1", "rag-c8", "rag-r3")
METRICS = ("output_tok_s", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms")
PHYSICAL_GPU = 6


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_arm(root, event, cap, seed):
    label = f"{event}-{cap}-s{seed}"
    folder = root / label
    receipt = read(folder / "receipt.json")
    require((receipt["mode"], receipt["event_publication"], receipt["policy"], receipt["seed"])
            == (label, f"{event}-v1", cap, seed), f"{label}: wrong receipt identity")
    require(receipt["status"] == "passed" and receipt["server_exit"] == 0 and receipt["observed_active"],
            f"{label}: unsuccessful arm")
    gpu = read(folder / "idle-after.json")
    require(gpu["status"] == "running" and gpu["phase"] == "idle" and gpu["physical_index"] == PHYSICAL_GPU
            and gpu["allocation_bytes"] == gpu["torch_reserved_bytes"] == 20 * 1024**3
            and not gpu["other_processes"] and gpu["gpu"] == receipt["gpu_uuid"]
            and (gpu["pid"], gpu["start_ticks"]) == (receipt["holder_pid"], receipt["holder_start_ticks"]),
            f"{label}: GPU reservation recovery failed")
    flags = dict(zip(receipt["server_argv"][2::2], receipt["server_argv"][3::2], strict=True))
    for key, expected in (("--event-publication", f"{event}-v1"), ("--prefill-burst-max-decode-rows", str(cap))):
        require(flags.pop(key) == expected, f"{label}: {key} disagrees with selection")
    flags.pop("--bind")
    exact_path = folder / "exact.json"
    final_path = folder / "shutdown-metrics.json"
    require(digest(exact_path) == receipt["exact_sha256"], f"{label}: exact capture changed")
    require(digest(final_path) == receipt["shutdown_metrics_sha256"], f"{label}: shutdown capture changed")
    exact = read(exact_path)
    require(len(exact["prompts"]) == len(set(exact["prompts"])) == len(exact["sequential"]) == 8,
            f"{label}: exact completion coverage is incomplete")
    final = read(final_path)
    require(all(final[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and len(final["allocation"]) == 4 and not any(final["allocation"].values()),
            f"{label}: shutdown resource state")
    cells = {}
    require(set(receipt["cells"]) == set(CELLS), f"{label}: incomplete cell list")
    for cell in CELLS:
        record = receipt["cells"][cell]
        metric_path = folder / f"{cell}.json"
        row_path = folder / f"{cell}.records.json"
        require((digest(metric_path), digest(row_path)) == (record["summary_sha256"], record["records_sha256"]),
                f"{label}/{cell}: input digests changed")
        metric, rows = read(metric_path), read(row_path)
        config = dict(metric["config"])
        for key in ("address", "out", "records"):
            config.pop(key)
        require(config["seed"] == seed and metric["warmup_ok"] == config["warmup"]
                and metric["ok"] == len(rows) == config["requests"]
                and metric["errors"] == metric["rejected_429"] == metric["warmup_errors"] == 0,
                f"{label}/{cell}: incomplete request coverage")
        require([row["request_index"] for row in rows] == list(range(config["warmup"], config["warmup"] + len(rows))),
                f"{label}/{cell}: measurement order changed")
        cells[cell] = (config, {key: metric[key] for key in METRICS}, rows)
    return receipt, flags, exact, cells


def ratio(new, old):
    return new / old if old else None


def compare(root):
    require((root / "campaign.exit").read_text().strip() == "0", "campaign has a failed or missing arm")
    report = {}
    build = None
    for seed in (1, 2):
        arms = {(event, cap): load_arm(root, event, cap, seed) for event, cap in product(EVENTS, CAPS)}
        reference, flags, exact, cells = arms["after-submit", 32]
        seed_build = (reference["riley_source"], reference["sha256"], flags, reference["gpu_uuid"])
        if build is None:
            build = seed_build
        require(seed_build == build, "build, server flags or device changed between seeds")
        comparisons = {}
        for (event, cap), (receipt, options, samples, results) in arms.items():
            require(receipt["sha256"] == reference["sha256"] and flags == options
                    and receipt["riley_source"] == reference["riley_source"], "base execution inputs differ")
            compare_cells = {}
            for cell, (config, baseline, expected) in cells.items():
                test_config, observed, rows = results[cell]
                require(test_config == config, f"{cell}: workload flags differ")
                request_fields = ("input_sha256", "request_index", "scheduled_start")
                output_fields = ("output_sha256", "finish", "prompt_tokens", "completion_tokens")
                require(all(all(a[key] == b[key] for key in request_fields)
                            for a, b in zip(expected, rows, strict=True)), f"{cell}: requests differ")
                compare_cells[cell] = {
                    "requests": len(rows),
                    "matching_output": sum(all(a[k] == b[k] for k in output_fields)
                                           for a, b in zip(expected, rows, strict=True)),
                    "metrics": observed,
                    "ratio_to_after_submit_cap32": {key: ratio(observed[key], baseline[key]) for key in METRICS},
                }
            require(samples["prompts"] == exact["prompts"], "exact probe prompts differ")
            exact_matches = sum(a == b == c for a, b, c in
                                zip(exact["sequential"], samples["sequential"], samples["concurrent"], strict=True))
            comparisons[f"{event}-{cap}"] = {"exact_c1_c8_matches": exact_matches, "cells": compare_cells}
        interaction = {}
        for cell in CELLS:
            a, b, c, d = (arms[policy][3][cell][1] for policy in product(EVENTS, CAPS))
            interaction[cell] = {key: ratio(d[key] * a[key], c[key] * b[key]) for key in METRICS}
        report[str(seed)] = {"policies": comparisons, "interaction_ratio": interaction}
    exact_pass = all(arm["exact_c1_c8_matches"] == 8
                     and all(cell["matching_output"] == cell["requests"] for cell in arm["cells"].values())
                     for seed in report.values() for arm in seed["policies"].values())
    return {"status": "passed" if exact_pass else "output_mismatch",
            "scope": "Input binding and output parity of completed arms", "physical_gpu": PHYSICAL_GPU, "seeds": report}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    document = compare(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
    sys.exit(0 if document["status"] == "passed" else 1)
