"""Extract receipt-bound pre-decode plans for the priority replay."""
from argparse import ArgumentParser
import json
from pathlib import Path

from analyze_prefill_turns import reconstruct
from compare_profile_arms import digest, read, require
from summarize_request_stages import read_phases


def source_cells(report, mode, kind):
    if kind == "qwen-trace":
        logged = report["server_logs_sha256"][mode]
        content = report["cells"]["wave-50ms"]["arms"][mode]["traced_data_sha256"]
        return logged, {"wave-nominal": ("wave-50ms", content)}
    arm = report["arms"][mode]
    if kind == "coder-trace":
        content = arm["cells"]["wave-50ms"]["trace_source_sha256"]
        return arm["server_log_sha256"], {"wave-nominal": ("wave-50ms", content)}
    cells = {cell: (cell, metrics["result_sha256"]) for cell, metrics in arm["cells"].items()}
    return arm["server_log_sha256"], cells


def extract(root, receipt, kind):
    report = read(receipt)
    modes = ("riley-first", "qkv-first", "qkv-second", "riley-second")
    require(root.name == report.get("remote_stage", "w3-qkv-tracing-20261119.s9zZEnqW"), "wrong phase campaign")
    require((root / "campaign.exit").read_text().strip() == "0", "source campaign did not finish")
    model = "qwen14" if kind == "qwen-trace" else "coder14"
    arms = {}
    for mode in modes:
        folder = root / f"{model}-{mode}"
        log_sha, captures = source_cells(report, mode, kind)
        log = folder / "server.log"
        require(digest(log) == log_sha, "server trace changed")
        phases = read_phases(log)
        steps, turns, admitted, lengths = reconstruct(phases)
        cells = {}
        for cell, (name, checksum) in captures.items():
            source = folder / f"{name}.json"
            require(digest(source) == checksum, "request trace changed")
            waves = {row["request_index"]: row["request_id"] for row in read(source)["measured"]
                     if 40 <= row["request_index"] <= 47}
            require(set(waves) == set(range(40, 48)), "the measured wave is incomplete")
            indices = {request: index for index, request in waves.items()}
            prefix = turns[waves[40]][0][0]
            windows = steps[prefix:prefix + 9]
            origin = phases[waves[40]]["command_send"][0][0]
            plans = []
            for window in windows:
                chunk, = window.chunks
                require(chunk.request in indices and chunk.batch_rows == 1024 and chunk.final == chunk.decode_rows == 0,
                        "prefix is not one unfinished prefill per step")
                plans.append([indices[chunk.request], chunk.first_token, chunk.next_token])
            require(len(plans) == 9, "prefix has fewer than nine steps")
            cells[cell] = {
                "source_sha256": digest(source),
                "requests_index_send_admit_length": [[index, phases[key]["command_send"][0][0] - origin,
                                                       admitted[key] - origin, lengths[key]]
                                                      for index, key in sorted(waves.items())],
                "step_windows_ns": [[window.start - origin, window.end - origin] for window in windows],
                "observed_prefill": plans,
            }
        arms[mode] = {"server_log_sha256": digest(log), "cells": cells}
    return {"source_receipt_sha256": digest(receipt), "remote_stage": root.name, "arms": arms}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("kind", choices=("coder-phase", "coder-trace", "qwen-trace"))
    args = parser.parse_args()
    print(json.dumps(extract(args.root, args.receipt, args.kind), indent=2))
