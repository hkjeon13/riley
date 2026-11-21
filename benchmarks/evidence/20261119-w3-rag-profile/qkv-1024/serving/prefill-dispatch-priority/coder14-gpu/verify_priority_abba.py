"""Verify the output, scheduling and latency of the same-binary priority ABBA."""
from argparse import ArgumentParser
import json
from math import isclose, isfinite
from pathlib import Path

from analyze_coder_wave_decode import decode_pair
from analyze_prefill_turns import reconstruct
from compare_coder import DECODE_GUARD_CELLS, EXPECTED_CELLS
from compare_profile_arms import HOLDER_BYTES, METRICS, arm, digest, ratio_percent, read, require
from summarize_profile_arm import stable_server
from summarize_request_stages import prefill_union, read_phases, request_phases
from verify_trace import client_bounds, window_overlap


ORDER = ("control-first", "priority-first", "priority-second", "control-second")
SWITCH = "RILEY_W3_EARLY_PREFILL_PRIORITY"
BINARY = "4337bd35053b5bb8c6e9f2ee0997888ac055760906e45bd7185940a671af5043"
REFERENCE = "712390666d906f9df399f95870037cb4b5112eb6b8607bd3834f661e579c442f"
INPUTS = ("request_index", "input_sha256", "prompt_tokens")
OUTPUTS = ("output_sha256", "completion_tokens", "finish")


def reservations(root, mode):
    result = {}
    for gpu in (6, 7):
        before, after = (read(root / f"gpu{gpu}-{side}-{mode}.json") for side in ("before", "after"))
        for state in before, after:
            require(state["physical_index"] == gpu and state["phase"] == "idle" and not state["other_processes"]
                    and state["allocation_bytes"] == state["torch_reserved_bytes"] == HOLDER_BYTES,
                    f"{mode}: GPU {gpu} reservation differs")
        fields = ("pid", "start_ticks", "gpu")
        require(all(before[key] == after[key] for key in fields), f"{mode}: GPU {gpu} owner changed")
        result[str(gpu)] = {key: after[key] for key in ("pid", "start_ticks")}
    return result


def focal_wave(log, rows):
    phases = read_phases(log)
    steps, turns, admissions, lengths = reconstruct(phases)
    wave = {row["request_id"]: row for row in rows if 40 <= row["request_index"] < 48}
    indices = {row["request_index"]: key for key, row in wave.items()}
    require(set(indices) == set(range(40, 48)) and len(wave) == 8, "fifth wave is incomplete")
    require(all(row["prompt_tokens"] == lengths[key] for key, row in wave.items()), "wave lengths differ")
    first = turns[indices[40]][0][0]
    prefix = steps[first:first + 9]
    require(len(prefix) == 9 and all(chunk.request in wave for step in prefix for chunk in step.chunks),
            "fifth-wave prefix includes a different workload")
    target = indices[41]
    row = wave[target]
    spans = prefill_union(phases)
    traced = request_phases(phases[target], lengths[target], spans)
    bounds = [client_bounds(item, phases[item["request_id"]]) for item in rows]
    clock = max(start for start, _ in bounds) - 2, min(end for _, end in bounds) + 2
    require(0 <= clock[1] - clock[0] <= 2_000_000, "wave time alignment is too wide")
    return {
        "first_nine_prefill_turns": [[wave[chunk.request]["request_index"] for chunk in step.chunks] for step in prefix],
        "request_45_admit_relative_to_first_41_prefill_ms":
            (admissions[indices[45]] - turns[target][0][1].start) / 1e6,
        "request_41_ttft_ms": row["ttft"] * 1000,
        "request_41_last_visible_content_ms": row["e2e"] * 1000,
        "request_41_decode_overlap_with_prefill_ms": window_overlap(row, spans, clock),
        "request_41_between_step_wait_ms": traced["stages_ms"]["prefill_between_steps"],
        "clock_alignment_interval_ns": clock[1] - clock[0],
    }


def checked_arm(root, archive, mode):
    directory = root / f"coder14-{mode}"
    receipt, plan, cells = arm(root, "coder14", mode, expected_cells=EXPECTED_CELLS)
    past_mode = "qkv-" + mode.split("-")[1]
    previous, previous_plan, past_cells = arm(archive, "coder14", past_mode, expected_cells=EXPECTED_CELLS)
    settings = {"RILEY_W3_QKV1024": "1", "RILEY_W3_REQUEST_TRACE": "1"}
    if mode.startswith("priority-"):
        settings[SWITCH] = "1"
    require(receipt["model"] == previous["model"] == "Qwen2.5-Coder-14B-Instruct"
            and receipt["common_kv_blocks"] == previous["common_kv_blocks"] == 2576
            and receipt["gpu"] == previous["gpu"] and receipt["kv_cache_bytes"] == previous["kv_cache_bytes"]
            and plan["cells"] == previous_plan["cells"] and plan["check_token_ids"] is True
            and stable_server(plan["server"])[1:] == stable_server(previous_plan["server"])[1:],
            f"{mode}: workload, model or server arguments differ")
    require(plan["env"] == receipt["live_diagnostic_environment"] == settings, "live switches differ")
    current_hashes = {Path(key).name: value for key, value in receipt["sha256"].items() if Path(key).name != "plan.json"}
    previous_hashes = {Path(key).name: value for key, value in previous["sha256"].items()}
    immutable = ("config.json", "riley-checkpoint.json", "tokenizer.json", "load.py", "profile_workload.py")
    require(current_hashes["riley"] == BINARY and
            all(current_hashes[key] == previous_hashes[key] for key in immutable), "immutable inputs differ")
    log = directory / "server.log"
    markers = [line for line in log.read_text().splitlines() if line.startswith("RILEY_W3_QKV1024")]
    require(markers and set(markers) == {"RILEY_W3_QKV1024 recorded=1 cuda=120405"}, "1,024-row plan did not bind")
    require(read(directory / "server-exit.json")["exit"] == 0, "server shutdown was unsuccessful")
    token_path = directory / "token-id-probe.json"
    token_probe = read(token_path)
    require(digest(token_path) == receipt["token_id_probe_sha256"] and
            len(token_probe["c1"]) == len(token_probe["c8"]) == 8 and
            token_probe["invariant"] == (token_probe["c1"] == token_probe["c8"]), "token probe differs")
    exact = {}
    for name, (stats, rows) in cells.items():
        original = past_cells[name][1]
        paired = list(zip(rows, original, strict=True))
        require(all(all(left[key] == right[key] for key in INPUTS) for left, right in paired),
                f"{mode}/{name}: request inputs differ")
        require(all(isfinite(stats[key]) and stats[key] > 0 for key in METRICS), f"{mode}/{name}: invalid performance")
        exact[name] = sum(all(left[key] == right[key] for key in OUTPUTS) for left, right in paired)
    reported = {
        "environment": settings, "reservations": reservations(root, mode),
        "matches_archived_by_cell": exact, "results_sha256": receipt["results"],
        "token_probe_sha256": digest(token_path), "server_log_sha256": digest(log),
        "wave_50ms": focal_wave(log, cells["wave-50ms"][1]),
    }
    return reported, cells, token_probe, current_hashes


def verify(root, reference):
    require((root / "campaign.exit").read_text().strip() == "0" and digest(reference / "verified.json") == REFERENCE,
            "campaign or archived baseline did not verify")
    boundaries = [line.split() for line in (root / "campaign.log").read_text().splitlines()
                  if line.startswith(("START ", "END "))]
    require([tuple(row[:2]) for row in boundaries] == [(event, tag) for tag in ORDER for event in ("START", "END")]
            and all(row[3:] == ["exit=0"] for row in boundaries[1::2])
            and [row[2] for row in boundaries] == sorted(row[2] for row in boundaries), "run sequence differs")
    runs = {mode: checked_arm(root, reference, mode) for mode in ORDER}
    arms = {name: parts[0] for name, parts in runs.items()}
    binding = runs[ORDER[0]][3]
    require(all(parts[3] == binding and parts[0]["reservations"] == arms[ORDER[0]]["reservations"]
                for parts in runs.values()), "serving lineage or reservation changed between arms")
    pairs = {}
    for order in ("first", "second"):
        baseline, candidate = (runs[f"{mode}-{order}"] for mode in ("control", "priority"))
        cells = {}
        for name in EXPECTED_CELLS:
            before, left = baseline[1][name]
            after, right = candidate[1][name]
            exact = sum(all(a[key] == b[key] for key in OUTPUTS) for a, b in zip(left, right, strict=True))
            decode = decode_pair(left, right)
            require(isclose(decode["p50_ms"]["control"], before["tpot_p50_ms"])
                    and isclose(decode["p50_ms"]["candidate"], after["tpot_p50_ms"]),
                    f"{name}: reported decode percentile differs from request times")
            cells[name] = {"requests": len(left), "exact_outputs": exact, "paired_decode": decode,
                           "control": {k: before[k] for k in METRICS},
                           "candidate": {k: after[k] for k in METRICS},
                           "change_percent": {k: ratio_percent(after[k], before[k]) for k in METRICS}}
        a, b = baseline[0]["wave_50ms"], candidate[0]["wave_50ms"]
        pairs[order] = {"cells": cells, "anticipates_request_45": a["first_nine_prefill_turns"][-2:] == [[41], [45]]
                        and b["first_nine_prefill_turns"][-2:] == [[45], [41]],
                        "request_41_ttft_change_ms": b["request_41_ttft_ms"] - a["request_41_ttft_ms"],
                        "request_41_last_visible_content_change_ms":
                            b["request_41_last_visible_content_ms"] - a["request_41_last_visible_content_ms"],
                        "request_41_between_step_wait_change_ms":
                            b["request_41_between_step_wait_ms"] - a["request_41_between_step_wait_ms"]}
    checks = {
        "exact_measured_outputs": all(cell["requests"] == cell["exact_outputs"]
                                      for pair in pairs.values() for cell in pair["cells"].values()),
        "exact_c1_c8_token_ids": all(parts[2]["invariant"] and parts[2] == runs[ORDER[0]][2]
                                    for parts in runs.values()),
        "decode_tpot_regression_below_1_percent": all(pair["cells"][name]["change_percent"]["tpot_p50_ms"] < 1
                                                     for pair in pairs.values() for name in DECODE_GUARD_CELLS),
        "short_ttft_p95_regression_below_2_percent": all(pair["cells"]["short-c32"]["change_percent"]["ttft_p95_ms"] < 2
                                                         for pair in pairs.values()),
        "short_throughput_regression_below_2_percent": all(pair["cells"]["short-c32"]["change_percent"]["output_tok_s"] > -2
                                                            for pair in pairs.values()),
        "wave_50ms_ttft_p95_improves_in_both_orders": all(
            pair["cells"]["wave-50ms"]["change_percent"]["ttft_p95_ms"] < 0 for pair in pairs.values()),
        "fifth_wave_order_changes_as_expected_in_both_orders": all(pair["anticipates_request_45"] for pair in pairs.values()),
    }
    return {"stage": root.name, "scope": "Coder-14B, the 1,024-row QKV algorithm enabled in every arm.",
            "physical_gpu": 7, "source_receipt_sha256": REFERENCE, "binary_sha256": BINARY,
            "run_order": ORDER, "boundaries": boundaries, "arms": arms, "pairs": pairs, "checks": checks}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.reference), indent=2, allow_nan=False))
