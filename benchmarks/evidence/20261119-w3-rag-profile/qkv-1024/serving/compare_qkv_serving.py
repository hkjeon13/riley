"""Verify the same-binary A100 ABBA comparison of the 1,024-row QKV algorithm."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from compare_profile_arms import CELLS, METRICS, arm, ratio_percent, read, require
from summarize_profile_arm import stable_server


ORDER = ("riley-first", "qkv-first", "qkv-second", "riley-second")
EXPECTED_CELLS = (*CELLS, "short-c32")
ENVIRONMENT = "RILEY_W3_QKV1024"
ENABLED = "1"
DECODE_GUARD_CELLS = ("rag-c8-s1", "rag-c8-s2", "wave-sync", "wave-50ms", "short-c32")


def compare(root):
    require((root / "campaign.exit").read_text().strip() == "0", "campaign did not finish")
    starts = [line.split()[1] for line in (root / "campaign.log").read_text().splitlines()
              if line.startswith("START ")]
    require(starts == list(ORDER), "campaign order differs")
    runs = {}
    for mode in ORDER:
        directory = root / f"qwen14-{mode}"
        receipt, plan, results = arm(root, "qwen14", mode, expected_cells=EXPECTED_CELLS)
        active = mode.startswith("qkv-")
        require(plan["env"] == ({ENVIRONMENT: ENABLED} if active else {}),
                f"{mode}: unexpected diagnostic environment")
        log = (directory / "server.log").read_text()
        markers = [line for line in log.splitlines() if line.startswith(ENVIRONMENT)]
        require(bool(markers) == active
                and all(line == f"{ENVIRONMENT} recorded=1 cuda=120405" for line in markers),
                f"{mode}: exact-M candidate did not pass runtime equivalence")
        require(read(directory / "server-exit.json")["exit"] == 0,
                f"{mode}: server did not stop gracefully")
        before = read(root / f"idle-before-{mode}.json")
        after = read(directory / "idle-after.json")
        require(all(before[key] == after[key] for key in before if key != "updated_ns")
                and before["status"] == "running", f"{mode}: GPU holder changed")
        raw = (directory / "token-id-probe.json").read_bytes()
        require(sha256(raw).hexdigest() == receipt["token_id_probe_sha256"],
                f"{mode}: token probe digest differs")
        token_probe = json.loads(raw)
        require(len(token_probe["c1"]) == len(token_probe["c8"]) == 8
                and token_probe["invariant"] == (token_probe["c1"] == token_probe["c8"]),
                f"{mode}: incomplete token ID probe")
        binding = {
            "receipt": {key: receipt[key] for key in ("model", "gpu", "common_kv_blocks",
                        "kv_cache_bytes", "holder_pid", "holder_start_ticks")},
            "sha256": {Path(key).name: value for key, value in receipt["sha256"].items()
                       if Path(key).name != "plan.json"},
            "cells": plan["cells"],
            "server": stable_server(plan["server"]),
        }
        runs[mode] = {"binding": binding, "token_probe": token_probe, "results": results}
    base = runs["riley-first"]["binding"]
    require(all(run["binding"] == base for run in runs.values()), "serving inputs differed across arms")
    require((root / "binary.sha256").read_text().split() ==
            [base["sha256"]["riley"], "target/release/riley"], "binary digest differs")
    input_fields = ("request_index", "input_sha256", "prompt_tokens")
    output_fields = ("completion_tokens", "output_sha256", "finish")
    cells = {}
    for name in EXPECTED_CELLS:
        control_rows = runs["riley-first"]["results"][name][1]
        requests = len(control_rows)
        metrics, matched = {}, {}
        for mode, run in runs.items():
            stats, rows = run["results"][name]
            joined = list(zip(control_rows, rows, strict=True))
            require(all(all(old[key] == new[key] for key in input_fields) for old, new in joined),
                    f"{mode}/{name}: request inputs differ")
            matched[mode] = sum(all(old[key] == new[key] for key in output_fields) for old, new in joined)
            metrics[mode] = {key: stats[key] for key in METRICS}
        changes = {point: {key: ratio_percent(metrics[f"qkv-{point}"][key],
                                                metrics[f"riley-{point}"][key]) for key in METRICS}
                   for point in ("first", "second")}
        cells[name] = {"requests_per_arm": requests, "exact_outputs_vs_first_control": matched,
                       "metrics": metrics, "qkv_change_percent": changes,
                       "control_change_percent": {key: ratio_percent(metrics["riley-second"][key],
                                                                     metrics["riley-first"][key])
                                                  for key in METRICS}}
    probe = runs["riley-first"]["token_probe"]
    guardrails = {
        "exact_stream_results": all(all(count == cell["requests_per_arm"]
                                        for count in cell["exact_outputs_vs_first_control"].values())
                                    for cell in cells.values()),
        "exact_c1_c8_token_ids": probe["invariant"] and all(run["token_probe"] == probe for run in runs.values()),
        "decode_tpot_regression_below_1_percent": all(cells[name]["qkv_change_percent"][point]["tpot_p50_ms"] < 1
                                                     for name in DECODE_GUARD_CELLS
                                                     for point in ("first", "second")),
        "short_ttft_p95_regression_below_2_percent": all(change["ttft_p95_ms"] < 2
                                                         for change in cells["short-c32"]["qkv_change_percent"].values()),
        "short_throughput_regression_below_2_percent": all(change["output_tok_s"] > -2
                                                           for change in cells["short-c32"]["qkv_change_percent"].values()),
    }
    return {"order": ORDER, "physical_gpu": 7, "qkv_gemm_rows": 1024,
            "identity": base["receipt"], "sha256": base["sha256"], "guardrails": guardrails,
            "cells": cells}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
