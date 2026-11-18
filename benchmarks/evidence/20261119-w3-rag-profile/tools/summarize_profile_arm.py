"""Bind a Riley Nsight arm to its measured requests and NVML reservation."""

from argparse import ArgumentParser
import gzip
import json
from pathlib import Path
import re
from statistics import median

from compare_profile_arms import HOLDER_BYTES, digest, read, require


def host_steps(path):
    with gzip.open(path, "rt") as log:
        rows = [
            {key: float(value) for key, value in re.findall(r"([a-z+]+)=(\d+(?:\.\d+)?)", line.split("|", 1)[0])}
            for line in log if line.startswith("HOSTSTEP us/step over 128 steps:")
        ]
    require(rows and all(len(row) == 12 and row.keys() == rows[0].keys() for row in rows),
            "host timing windows are missing or not comparable")
    return {
        "block_steps": 128,
        "blocks": len(rows),
        "median_us_per_step": {field: median(row[field] for row in rows) for field in rows[0]},
        "other_host_phases_median_us_per_step": median(
            sum(value for field, value in row.items() if field not in ("cancel+commands", "submit", "wait"))
            for row in rows),
    }


def stable_server(argv):
    address = argv.index("--bind")
    return argv[:address + 1] + argv[address + 2:]


def summarize(profile, categories_path, trace_files_path, baseline=None):
    plan = read(profile / "plan.json")
    receipt = read(profile / "receipt.json")
    idle = read(profile / "idle-after.json")
    categories = read(categories_path)
    server_pid = int((profile / "server.pid").read_text())
    require(receipt["status"] == "passed" and receipt["mode"] == "profile" and plan["profile"],
            "arm is not a completed GPU profile")
    require(receipt["nvml_samples"] > 0 and read(profile / "server-exit.json")["exit"] == 0,
            "arm lacks NVML coverage or graceful server exit")
    require(idle["phase"] == "idle" and idle["physical_index"] == 7 and not idle["other_processes"]
            and idle["gpu"] == receipt["gpu"]
            and idle["allocation_bytes"] == idle["torch_reserved_bytes"] == HOLDER_BYTES
            and (idle["pid"], idle["start_ticks"]) == (receipt["holder_pid"], receipt["holder_start_ticks"]),
            "GPU reservation did not return to idle")
    inputs = {Path(name).name: value for name, value in receipt["sha256"].items()}
    require(inputs["plan.json"] == digest(profile / "plan.json"), "measured plan digest differs")
    require(categories["sqlite_sha256"] == receipt["sqlite_sha256"] and categories["server_pid"] == server_pid,
            "kernel classification uses a different SQLite database or process")
    require(plan["cells"].keys() == receipt["results"].keys() == categories["cells"].keys(),
            "workload or classification cells differ")
    if baseline is not None:
        base_receipt = read(baseline / "receipt.json")
        base_plan = read(baseline / "plan.json")
        base_inputs = {Path(name).name: value for name, value in base_receipt["sha256"].items()}
        shared = ("riley", "riley-checkpoint.json", "tokenizer.json", "load.py", "profile_workload.py")
        require(base_receipt["status"] == "passed" and not base_plan["profile"]
                and base_receipt["gpu"] == receipt["gpu"] and base_receipt["model"] == receipt["model"]
                and all(inputs[k] == base_inputs[k] for k in shared)
                and base_inputs["plan.json"] == digest(baseline / "plan.json")
                and stable_server(plan["server"]) == stable_server(base_plan["server"]),
                "unprofiled arm has different build, device, model or server flags")
    cells = {}
    for key, expected in plan["cells"].items():
        data_path = profile / f"{key}.json"
        require(digest(data_path) == receipt["results"][key], f"{key}: measured request digest differs")
        load = read(data_path)
        timing = read(profile / f"{key}.profile.json")
        rows = timing["measurements"]
        load_stats = load["statistics"]
        category = categories["cells"][key]
        require(load_stats["config"] == expected
                and load_stats["ok"] == len(load["measured"]) == len(rows) == category["requests"]
                == timing["requests"] == expected["requests"]
                and load_stats["warmup_ok"] == expected["warmup"]
                and load_stats["errors"] == load_stats["rejected_429"] == 0,
                f"{key}: incomplete workload or mismatched profile coverage")
        require(timing["server_pid"] == server_pid
                and all(0 <= n <= 1_000_000 for n in (*timing["clock_bracket_ns"], timing["clock_offset_change_ns"]))
                and [row["request_index"] for row in rows] == [row["request_index"] for row in load["measured"]],
                f"{key}: invalid GPU owner, request identity or clock alignment")
        require(all(0 < row["gpu_kernel_union_ms"] <= row["sent_to_first_ms"] for row in rows),
                f"{key}: invalid kernel window")
        summary = {"requests": len(rows),
                   "prompt_tokens_mean": load_stats["prompt_tokens_mean"],
                   "ttft_p95_with_profile_ms": load_stats["ttft_p95_ms"],
                   "sent_to_first_ms_p50": median(row["sent_to_first_ms"] for row in rows),
                   "gpu_kernel_union_ms_p50": median(row["gpu_kernel_union_ms"] for row in rows),
                   "outside_kernels_ms_p50": median(row["time_outside_gpu_kernels_ms"] for row in rows),
                   "gpu_busy_share_p50": timing["gpu_busy_share_median"],
                   "clock_offset_change_ns": timing["clock_offset_change_ns"],
                   "kernel_category_ms_p50": category["kernel_category_ms_p50"]}
        if baseline is not None:
            original_path = baseline / f"{key}.json"
            require(digest(original_path) == base_receipt["results"][key] and expected == base_plan["cells"][key],
                    f"{key}: original workload digest or plan changed")
            original = read(original_path)
            paired = list(zip(load["measured"], original["measured"], strict=True))
            require(all(new["input_sha256"] == old["input_sha256"] for new, old in paired),
                    f"{key}: input changed")
            without = original["statistics"]["ttft_p95_ms"]
            summary.update(ttft_p95_without_profile_ms=without,
                           profile_ttft_p95_relative_percent=100 * (load_stats["ttft_p95_ms"] / without - 1),
                           matching_output_text_and_finish=sum(
                               (new["output_sha256"], new["finish"]) == (old["output_sha256"], old["finish"])
                               for new, old in paired))
        cells[key] = summary
    trace_files = read(trace_files_path)
    for name, field in (("trace.nsys-rep", "raw_report_sha256"), ("trace.sqlite", "sqlite_sha256")):
        expected = receipt[field]
        require(trace_files[name]["bytes"] > 0 and trace_files[name]["sha256"] == expected,
                f"{name}: trace inventory disagrees with the binding receipt")
    return {"model": receipt["model"], "physical_gpu": 7, "binary_sha256": inputs["riley"],
            "scope": "Request windows share kernel time for concurrent clients",
            "trace_files": trace_files, "cells": cells, "host_step_windows": host_steps(profile / "server.log.gz")}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("categories", type=Path)
    parser.add_argument("trace_files", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    data = summarize(args.profile, args.categories, args.trace_files, args.baseline)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
