"""Screen a matched 14B prefill-order experiment for parity and regressions."""

from argparse import ArgumentParser
import json
from pathlib import Path

from compare_profile_arms import CELLS, METRICS, arm, digest, ratio_percent, read, require
from summarize_profile_arm import stable_server


def compare(root):
    require((root / "campaign.exit").read_text().strip() == "0", "continuation campaign did not finish")
    expected = (*CELLS, "short-c32")
    modes = ("riley", "continuation")
    (base, base_plan, control), (candidate, candidate_plan, measured) = (
        arm(root, "qwen14", mode, expected_cells=expected) for mode in modes
    )
    immutable = ("model", "gpu", "common_kv_blocks", "kv_cache_bytes", "holder_pid", "holder_start_ticks")
    hashes = [{Path(k).name: v for k, v in record["sha256"].items() if Path(k).name != "plan.json"}
              for record in (base, candidate)]
    require(all(base[name] == candidate[name] for name in immutable) and hashes[0] == hashes[1],
            "experiment changed serving inputs")
    require(base_plan["cells"] == candidate_plan["cells"] and base_plan["env"] == {}
            and candidate_plan["env"] == {"RILEY_W3_PREFILL_CONTINUE": "1"}
            and stable_server(base_plan["server"]) == stable_server(candidate_plan["server"]),
            "experiment changed workload or server flags")

    probes = []
    for receipt, mode in zip((base, candidate), modes, strict=True):
        directory = root / f"qwen14-{mode}"
        require(read(directory / "server-exit.json")["exit"] == 0, "server did not stop gracefully")
        probe_path = directory / "token-id-probe.json"
        require(digest(probe_path) == receipt["token_id_probe_sha256"], "token probe digest changed")
        probe = read(probe_path)
        require(len(probe["c1"]) == len(probe["c8"]) == 8
                and probe["invariant"] == (probe["c1"] == probe["c8"]), "invalid token ID parity record")
        probes.append(probe)
    token_probe_matches = probes[0] == probes[1] and probes[0]["invariant"]

    cells = {}
    input_fields = ("request_index", "input_sha256", "prompt_tokens")
    output_fields = ("completion_tokens", "output_sha256", "finish")
    for name, (before, base_rows) in control.items():
        after, candidate_rows = measured[name]
        rows = list(zip(base_rows, candidate_rows, strict=True))
        require(all(all(old[field] == new[field] for field in input_fields) for old, new in rows),
                f"{name}: serving inputs differ")
        cells[name] = {"requests": len(rows), "exact_outputs": sum(
                           all(old[field] == new[field] for field in output_fields) for old, new in rows),
                       "control": {key: before[key] for key in METRICS},
                       "continuation": {key: after[key] for key in METRICS},
                       "change_percent": {key: ratio_percent(after[key], before[key]) for key in METRICS}}
    short = cells["short-c32"]["change_percent"]
    guards = {"exact_stream_results": all(cell["exact_outputs"] == cell["requests"] for cell in cells.values()),
              "exact_c1_c8_token_ids": token_probe_matches,
              "short_ttft_p95_regression_below_2_percent": short["ttft_p95_ms"] < 2,
              "short_throughput_regression_below_2_percent": short["output_tok_s"] > -2,
              "decode_tpot_regression_below_1_percent": all(cells[name]["change_percent"]["tpot_p50_ms"] < 1
                                                            for name in ("short-c32", "rag-c8-s1", "rag-c8-s2"))}
    return {"model": base["model"], "physical_gpu": 7, "binary_sha256": hashes[0]["riley"],
            "guardrails": guards, "cells": cells}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
