"""Compare the explicit vLLM batch token budgets on the same held GPU."""

from argparse import ArgumentParser
import json
from pathlib import Path
import re

from compare_profile_arms import CELLS, METRICS, arm, ratio_percent, read, require


def budget_arm(root, cap):
    receipt, plan, results = arm(root, f"qwen14-{cap}", "vllm")
    directory = root / f"qwen14-{cap}-vllm"
    require(receipt["max_num_batched_tokens"] == cap, "unexpected budget in the receipt")
    startup = (directory / "server.log").read_text()
    api_budgets = re.findall(r"non-default args: .*'max_num_batched_tokens': (\d+)", startup)
    scheduler_budgets = re.findall(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)\.", startup)
    require(api_budgets == [str(cap)] and all(budget == str(cap) for budget in scheduler_budgets),
            "vLLM startup reported a different scheduling budget")
    require(read(directory / "server-exit.json")["exit"] == 0, "vLLM did not stop gracefully")
    server = list(plan["server"])
    token_limit = server.index("--max-num-batched-tokens")
    require(server[token_limit + 1] == str(cap), "unexpected budget in the command")
    del server[token_limit:token_limit + 2]
    port = server.index("--port")
    del server[port:port + 2]
    require(receipt["kv_capacity_matches"] and receipt["reported_kv_tokens"] == [receipt["common_kv_blocks"] * 16],
            "vLLM reported a different KV budget")
    inputs = {Path(path).name: checksum for path, checksum in receipt["sha256"].items()
              if Path(path).name != "plan.json"}
    return receipt, server, inputs, plan["cells"], results, {
        "api_logged_batched_tokens": int(api_budgets[0]),
        "scheduler_logged_batched_tokens": [int(budget) for budget in scheduler_budgets],
    }


def compare(root):
    require((root / "campaign.exit").read_text().strip() == "0", "budget campaign did not complete")
    low, low_server, low_inputs, low_config, low_rows, low_startup = budget_arm(root, 1024)
    high, high_server, high_inputs, high_config, high_rows, high_startup = budget_arm(root, 2048)
    require(low_server == high_server and low_inputs == high_inputs and low_config == high_config,
            "the arms differ in more than the explicit token budget and port")
    require(all(low[field] == high[field] for field in
                ("model", "gpu", "kv_cache_bytes", "common_kv_blocks", "reported_kv_tokens")),
            "model, device or KV allocation changed")
    cells = {}
    for name in CELLS:
        low_stats, low_records = low_rows[name]
        high_stats, high_records = high_rows[name]
        paired = list(zip(low_records, high_records, strict=True))
        require(all(a["input_sha256"] == b["input_sha256"] and a["prompt_tokens"] == b["prompt_tokens"]
                    for a, b in paired), f"{name}: prompt workload changed")
        report_fields = (*METRICS, "completion_tokens_mean")
        cells[name] = {
            "requests": len(paired),
            "matching_output_text_and_finish": sum(
                (a["output_sha256"], a["finish"]) == (b["output_sha256"], b["finish"]) for a, b in paired),
            "matching_output_token_counts": sum(a["completion_tokens"] == b["completion_tokens"] for a, b in paired),
            "1024": {key: low_stats[key] for key in report_fields},
            "2048": {key: high_stats[key] for key in report_fields},
            "1024_relative_to_2048_percent": {key: ratio_percent(low_stats[key], high_stats[key]) for key in METRICS},
        }
    return {"model": low["model"], "physical_gpu": 7, "kv_tokens": low["reported_kv_tokens"][0],
            "scope": "Explicit vLLM token ceiling with matching requests and KV within a pair",
            "startup": {"1024": low_startup, "2048": high_startup},
            "cells": cells}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.root)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
