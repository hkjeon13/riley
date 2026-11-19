"""Derive a paired performance result for the parent and packaged allocator image."""

from argparse import ArgumentParser
import json
from math import isfinite
from pathlib import Path
from statistics import fmean

from performance_client import scenarios
from qualify_allocator import check_reservations
from qualify_packaged_gpu import PROFILE, binding
from verify_allocator_image import LIB, LIB_SHA
from verify_a100_candidate import CHECKPOINT, GPU, ROOTS, RILEY, digest, gpu_evidence, read, require


ORDER = ("parent-first", "packaged-first", "packaged-second", "parent-second")
SERVER_ARGS = ("serve", "--model", "/models", "--model-id", "m", "--max-active-sequences", "32",
               "--max-sequence-tokens", "8192", "--max-output-tokens", "1024", "--kv-blocks", "16384",
               "--prefix-cache-blocks", "128")
METRICS = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms", "output_tok_s")
FIELDS = ("request_index", "input_sha256", "output_sha256", "prompt_tokens", "completion_tokens", "finish")


def cell(root, name, options, seed):
    summary = read(root / f"{name}.summary.json")
    records = read(root / f"{name}.records.json")
    table = dict(zip(options[::2], options[1::2], strict=True))
    count, warmup = int(table["--requests"]), int(table.get("--warmup", 32))
    config = summary["config"]
    require(config["seed"] == seed and all(str(config[key[2:].replace("-", "_")]) == value for key, value in table.items())
            and [row["request_index"] for row in records] == list(range(warmup, warmup + count))
            and summary["requests"] == summary["ok"] == count and summary["warmup_ok"] == warmup
            and summary["warmup_errors"] == summary["errors"] == summary["rejected_429"] == 0,
            f"{name}: a workload or terminal response differs")
    metrics = {metric: summary[metric] for metric in METRICS}
    require(all(type(value) in (int, float) and isfinite(value) and value > 0 for value in metrics.values()),
            f"{name}: missing or invalid latency")
    return metrics, [tuple(row[key] for key in FIELDS) for row in records]


def expected_image(package, evidence, stage, name):
    parent = evidence / ROOTS["base"] / "cpu-evidence/report.json"
    selected = parent if name.startswith("parent-") else stage / "gpu-checker-input.json"
    cpu = read(selected)
    if selected != parent:
        require(cpu == binding(package, evidence), "packaged performance CPU input changed")
    fields = {key: cpu[key] for key in ("image_id", "source_revision", "source_archive_sha256")}
    fields.update({"riley_binary_sha256": cpu["binaries"][RILEY], "cpu_report_sha256": digest(selected),
                   "checkpoint_sha256": CHECKPOINT, "gpu_uuid": GPU})
    return selected, fields


def arm(package, evidence, stage, name):
    directory = stage / name
    _, expected = expected_image(package, evidence, stage, name)
    receipt = gpu_evidence(directory, expected, int((directory / "exit").read_text()))
    require(receipt["status"] == "passed", f"{name}: GPU driver or API failed")
    process = read(directory / "process-before.json")
    require(process == read(directory / "process-after.json") and process["image_id"] == expected["image_id"]
            and process["server_arguments"] == list(SERVER_ARGS), f"{name}: server process or arguments changed")
    if name.startswith("parent-"):
        require(process["allocator"] == "glibc", "the parent has a different allocator")
    else:
        evidence_fields = process["allocator"]
        require(evidence_fields["library_sha256"] == LIB_SHA and evidence_fields["malloc_conf"] == PROFILE
                and evidence_fields["mounts"] == ["/models"], "packaged allocator proof differs")
    holders = check_reservations(directory, "after")
    require(holders == check_reservations(directory, "before"), "GPU reservation did not recover")
    raw = directory / "gpu-evidence"
    cases = {label: cell(raw, label, *definition) for label, definition in scenarios().items()}
    return cases, {"image_id": expected["image_id"], "server": process, "api_checks": receipt["api_checks"],
                   "report_sha256": receipt["gpu_receipt_sha256"], "artifacts_sha256": receipt["artifacts_sha256"],
                   "reservations_after": holders}


def change(candidate, baseline):
    return {key: (candidate[key] / baseline[key] - 1) * 100 for key in METRICS}


def report_cell(label, values):
    arms = {name: values[name][0] for name in ORDER}
    expected = values[ORDER[0]][1]
    matching = all(samples == expected for _, samples in values.values())
    pairs = {order: change(arms[f"packaged-{order}"], arms[f"parent-{order}"])
             for order in ("first", "second")}
    means = {kind: {metric: fmean(arms[f"{kind}-{order}"][metric] for order in ("first", "second"))
                    for metric in METRICS} for kind in ("parent", "packaged")}
    average = change(means["packaged"], means["parent"])
    guard = {"decode_tpot_below_one_percent_regression": average["tpot_p50_ms"] < 1}
    if label.startswith("closed-"):
        guard["short_ttft_p95_below_two_percent_regression"] = average["ttft_p95_ms"] < 2
        guard["short_throughput_below_two_percent_regression"] = average["output_tok_s"] > -2
    return {"aligned_samples_per_arm": len(expected), "same_complete_streams": matching,
            "metrics": arms, "change_percent_by_order": pairs, "change_percent_by_arm_means": average,
            "measured_regression_guards": guard}


def verify(package, evidence, stage):
    chronology = [line.split() for line in (stage / "campaign.log").read_text().splitlines()
                  if line.startswith(("START ", "END "))]
    require(len(chronology) == 8, "partial performance chronology")
    previous_end = ""
    for name, start, end in zip(ORDER, chronology[::2], chronology[1::2], strict=True):
        require(start[:2] == ["START", name] and end[:2] == ["END", name]
                and len(start) == 3 and len(end) == 4 and end[3] == "exit=0"
                and previous_end <= start[2] < end[2], f"{name}: incomplete or overlapping GPU arm")
        previous_end = end[2]
    arms = {name: arm(package, evidence, stage, name) for name in ORDER}
    cases = {label: report_cell(label, {name: arms[name][0][label] for name in ORDER}) for label in scenarios()}
    return {"scope": "Qwen2.5-7B-Instruct, A100 6, one matched runtime configuration.",
            "order": ORDER, "arms": {name: value[1] for name, value in arms.items()}, "cells": cases,
            "all_tested_streams_match": all(case["same_complete_streams"] for case in cases.values()),
            "all_measured_regression_guards_pass": all(
                all(case["measured_regression_guards"].values()) for case in cases.values())}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("package", "evidence", "stage"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.evidence, args.stage), indent=2, allow_nan=False))
