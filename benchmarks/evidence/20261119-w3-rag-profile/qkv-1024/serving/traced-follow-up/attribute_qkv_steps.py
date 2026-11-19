"""Explain per-request first-token changes using the traced prefill turns."""

from argparse import ArgumentParser
import json
from pathlib import Path

from analyze_prefill_turns import reconstruct, request_wait
from compare_profile_arms import read, require
from compare_traced_qkv import CELLS, ORDER, check_campaign, read_arm
from summarize_request_stages import read_phases


def scheduled_steps(request_id, steps, own, admitted):
    own_steps = [
        {
            "request_prompt_rows": chunk.next_token - chunk.first_token,
            "batch_rows": chunk.batch_rows,
            "decode_rows": chunk.decode_rows,
            "prefill_requests": len(steps[index].chunks),
            "step_gpu_ms": chunk.gpu_ns / 1e6,
        }
        for index, chunk in own
    ]
    other = []
    for (left, _), (right, _) in zip(own, own[1:]):
        other.extend(steps[left + 1:right])
    request_admission = admitted[request_id]
    later_only = sum(
        all(admitted[chunk.request] > request_admission for chunk in step.chunks)
        for step in other
    )
    return {"own_steps": own_steps, "other_prefill_steps": len(other), "later_only_prefill_steps": later_only}


def explain(root, previous, modes):
    times = check_campaign(root) if modes == ORDER else {}
    reports = {mode: read_arm(root, previous, mode) for mode in modes}
    binding = reports[modes[0]][0]
    require(all(value[0] == binding for value in reports.values()), "different trace inputs")
    require((root / "binary.sha256").read_text().split()
            == [binding["sha256"]["riley"], "target/release/riley"], "different trace build")
    targets = {}
    for cell in CELLS:
        indices = {41, 42} if cell == "wave-50ms" else set()
        for _, report, _ in reports.values():
            indices.update(report[cell]["tail_request_indices"])
        targets[cell] = sorted(indices)
    result = {name: {"focused_request_indices": targets[name], "arms": {}} for name in CELLS}
    for mode, (_, report, _) in reports.items():
        directory = root / f"qwen14-{mode}"
        steps, turns, admitted, lengths = reconstruct(read_phases(directory / "server.log"))
        for cell in CELLS:
            mapping = {line["request_index"]: line["request_id"]
                       for line in read(directory / f"{cell}.json")["measured"]}
            records = {}
            for index in targets[cell]:
                measure = report[cell]["rows"][index]
                request_id = mapping[index]
                route = request_wait({**measure, "request_id": request_id, "request_index": index},
                                     steps, turns, admitted, lengths, 512)
                routing = scheduled_steps(request_id, steps, turns[request_id], admitted)
                require(sum(step["request_prompt_rows"] for step in routing["own_steps"]) == measure["prompt_tokens"],
                        "scheduled rows do not account for the prompt")
                records[index] = {**measure, "prefill_wait_routing": route["span_ms"], **routing}
            result[cell]["arms"][mode] = {
                "ttft_p95_ms": report[cell]["ttft_p95_ms"],
                "tail_request_indices": report[cell]["tail_request_indices"],
                "verified_outputs": report[cell]["complete_streams_equal_to_uninstrumented"],
                "input_capture_sha256": report[cell]["input_capture_sha256"],
                "traced_data_sha256": report[cell]["traced_data_sha256"],
                "focused_requests": records,
            }
    for cell in result.values():
        cell["paired_deltas_ms"] = {}
        for pair in ("first", "second"):
            before = cell["arms"].get(f"riley-{pair}")
            after = cell["arms"].get(f"qkv-{pair}")
            if before is None or after is None:
                continue
            deltas = {}
            for index in cell["focused_request_indices"]:
                old, new = (point["focused_requests"][index] for point in (before, after))
                deltas[index] = {
                    "client_ttft": new["client_ttft_ms"] - old["client_ttft_ms"],
                    "server_observed": new["server_ms"] - old["server_ms"],
                    "shared_prefill_gpu": new["shared_prefill_gpu_ms"] - old["shared_prefill_gpu_ms"],
                    "own_prefill_steps": len(new["own_steps"]) - len(old["own_steps"]),
                    "other_prefill_steps": new["other_prefill_steps"] - old["other_prefill_steps"],
                    "later_only_prefill_steps": new["later_only_prefill_steps"] - old["later_only_prefill_steps"],
                    "stages": {
                        key: new["stages_ms"][key] - value for key, value in old["stages_ms"].items()
                    },
                    "prefill_wait_routing": {
                        key: new["prefill_wait_routing"][key] - value
                        for key, value in old["prefill_wait_routing"].items()
                    },
                }
            cell["paired_deltas_ms"][pair] = deltas
    return {
        "scope": "Prefill timing is shared within each scheduled step. Request IDs bind to measured streams within each arm.",
        "order": modes,
        "campaign_times_utc": times,
        "physical_gpu": 7,
        "model": binding["identity"]["model"],
        "kv_tokens": binding["identity"]["common_kv_blocks"] * 16,
        "inputs_sha256": binding["sha256"],
        "same_20gib_holder_restored_each_arm": True,
        "server_logs_sha256": {mode: run[2] for mode, run in reports.items()},
        "cells": result,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("--first-pair", action="store_true")
    args = parser.parse_args()
    modes = ORDER[:2] if args.first_pair else ORDER
    result = explain(args.root, args.previous, modes)
    print(json.dumps(result, indent=2, allow_nan=False))
