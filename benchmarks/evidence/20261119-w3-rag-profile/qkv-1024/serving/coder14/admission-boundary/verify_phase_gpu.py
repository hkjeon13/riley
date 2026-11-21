"""Verify the measured W3 phase groups from the preserved A100 traces."""

from argparse import ArgumentParser
import json
from pathlib import Path

from analyze_prefill_turns import reconstruct
from compare_profile_arms import METRICS, arm, digest, ratio_percent, read, require
from compare_traced_qkv import FIELDS, check_campaign
from phase_inputs import OFFSETS_MS, ORDER, phase_cells
from summarize_profile_arm import stable_server
from summarize_request_stages import prefill_union, read_phases, request_phases


MODEL = "Qwen2.5-Coder-14B-Instruct"
BINARY = "917caae4ff7467c58650ff5cf67004c17893aa862fd38371d2a940952c2ca398"
WAVE = set(range(40, 48))


def focus_timing(rows, phases, trajectory, union):
    steps, turns, admitted, lengths = trajectory
    wave = {row["request_index"]: row for row in rows if row["request_index"] in WAVE}
    require(set(wave) == WAVE, "the fifth measured wave is incomplete")
    ids = {row["request_id"]: index for index, row in wave.items()}
    require(len(ids) == len(WAVE) and all(lengths[key] == wave[index]["prompt_tokens"] for key, index in ids.items()),
            "the prefill trace does not match the fifth-wave request bodies")
    initial, focal, moved = (wave[index]["request_id"] for index in (40, 41, 45))
    first = turns[initial][0][0]
    focal_index = turns[focal][0][0]
    focal_start = steps[focal_index].start
    sent, = phases[moved]["command_send"]
    require(not sent[1] and admitted[initial] <= sent[0] <= admitted[moved], "invalid engine send timestamp")
    timing = request_phases(phases[focal], wave[41]["prompt_tokens"], union)
    early = steps[first:first + 9]
    require(len(early) == 9 and all(chunk.request in ids for step in early for chunk in step.chunks),
            "first nine prefill windows do not belong to this wave")
    order = [[ids[chunk.request] for chunk in step.chunks] for step in early]
    return {
        "client_send_45_relative_to_40_ms": (wave[45]["sent_at"] - wave[40]["sent_at"]) * 1000,
        "engine_send_45_after_first_admission_ms": (sent[0] - admitted[initial]) / 1e6,
        "engine_send_45_relative_to_first_41_prefill_ms": (sent[0] - focal_start) / 1e6,
        "admitted_45_relative_to_first_41_prefill_ms": (admitted[moved] - focal_start) / 1e6,
        "45_admits_before_first_41_prefill": admitted[moved] <= focal_start,
        "first_41_prefill_step": focal_index - first + 1,
        "prefill_turns_8_and_9": order[-2:],
        "focal_41_client_ttft_ms": wave[41]["ttft"] * 1000,
        "focal_41_server_stages_ms": timing["stages_ms"],
        "focal_41_prefill_chunks": len(timing["prefill_steps"]),
    }


def read_arm(root, original, label):
    cells = phase_cells("coder14", label)
    receipt, plan, workloads = arm(root, "coder14", label, expected_cells=tuple(cells))
    directory = root / f"coder14-{label}"
    hashes = {Path(path).name: value for path, value in receipt["sha256"].items() if Path(path).name != "plan.json"}
    require(receipt["model"] == MODEL and hashes["riley"] == BINARY and plan["cells"] == cells
            and receipt["common_kv_blocks"] == 2576 and not plan["check_token_ids"],
            f"{label}: pinned W3 inputs changed")
    inputs = {Path(path).name: digest(Path(path)) for path in receipt["sha256"]
              if Path(path).name != "plan.json"}
    require(inputs == hashes, f"{label}: pinned inputs did not survive the run")
    baseline = original / f"coder14-{label}"
    old_plan = read(baseline / "plan.json")
    old_receipt = read(baseline / "receipt.json")
    old_hashes = {Path(path).name: value for path, value in old_receipt["sha256"].items()}
    for name in ("riley", "load.py", "profile_summary.py", "riley-checkpoint.json", "config.json", "tokenizer.json"):
        require(old_hashes[name] == hashes[name], f"{label}: original digest changed for {name}")
    old_path = baseline / "wave-50ms.json"
    require(digest(old_path) == old_receipt["results"]["wave-50ms"], "original waveform checksum differs")
    old_rows = read(old_path)["measured"]
    old_flags = stable_server(old_plan["server"])
    server_flags = stable_server(plan["server"])
    # The immutable executable is reached through this arm's isolated path.
    require(server_flags[1:] == old_flags[1:] and plan["env"] == old_plan["env"],
            f"{label}: non-timing server options changed")
    before = read(root / f"idle-before-{label}.json")
    after = read(directory / "idle-after.json")
    require(all(before[key] == after[key] for key in before if key != "updated_ns")
            and after["pid"] == 2431237 and read(directory / "server-exit.json")["exit"] == 0,
            f"{label}: the device reservation or shutdown changed")
    before6, after6 = (read(root / f"gpu6-{side}-{label}.json") for side in ("before", "after"))
    require(before6.keys() == after6.keys()
            and all(before6[key] == after6[key] for key in before6 if key != "updated_ns")
            and after6["phase"] == "idle" and after6["pid"] == 2680326 and after6["physical_index"] == 6
            and after6["allocation_bytes"] == after6["torch_reserved_bytes"] == 20 * 1024**3
            and not after6["other_processes"], f"{label}: GPU 6's reservation changed")
    logfile = directory / "server.log"
    marker = "RILEY_W3_QKV1024 recorded=1 cuda=120405"
    markers = [line for line in logfile.read_text().splitlines() if line.startswith("RILEY_W3_QKV1024")]
    require(bool(markers) == label.startswith("qkv-") and all(line == marker for line in markers),
            f"{label}: invalid QKV selection trace")
    phases = read_phases(logfile)
    trajectory = reconstruct(phases)
    union = prefill_union(phases)
    result = {}
    for name, (stats, rows) in workloads.items():
        require(all(all(before[key] == after[key] for key in FIELDS)
                    for before, after in zip(old_rows, rows, strict=True)), f"{label}/{name}: output or input changed")
        result[name] = {
            "same_complete_streams": len(rows),
            "metrics": {metric: stats[metric] for metric in METRICS},
            "fifth_wave": focus_timing(rows, phases, trajectory, union),
            "result_sha256": receipt["results"][name],
        }
    return {"cells": result, "inputs_sha256": hashes, "server_log_sha256": digest(logfile),
            "baseline_wave_sha256": digest(old_path), "gpu7_reservation_restored": True, "same_gpu6_reservation": True}


def verify(root, original):
    chronology = check_campaign(root)
    arms = {label: read_arm(root, original, label) for label in ORDER}
    binding = arms[ORDER[0]]["inputs_sha256"]
    require(all(value["inputs_sha256"] == binding for value in arms.values()), "W3 phase arms used different inputs")
    phases = {
        name: {
            "riley_first_before": arms["riley-first"]["cells"][name]["fifth_wave"]["45_admits_before_first_41_prefill"],
            "qkv_first_before": arms["qkv-first"]["cells"][name]["fifth_wave"]["45_admits_before_first_41_prefill"],
            "qkv_second_before": arms["qkv-second"]["cells"][name]["fifth_wave"]["45_admits_before_first_41_prefill"],
            "riley_second_before": arms["riley-second"]["cells"][name]["fifth_wave"]["45_admits_before_first_41_prefill"],
        }
        for name in OFFSETS_MS
    }
    pairs = {
        point: {
            name: {metric: ratio_percent(arms[f"qkv-{point}"]["cells"][name]["metrics"][metric],
                                         arms[f"riley-{point}"]["cells"][name]["metrics"][metric])
                   for metric in METRICS}
            for name in OFFSETS_MS
        }
        for point in ("first", "second")
    }
    return {
        "scope": "Four A100 trace arms, offsetting only request 45 around the first prefill of request 41.",
        "chronology": chronology, "physical_gpu": 7, "model": MODEL,
        "same_480_measured_streams": sum(cell["same_complete_streams"] for run in arms.values() for cell in run["cells"].values()),
        "phase_by_cell": phases, "matched_early_and_late_phases": all(phases["wave-early"].values()) and not any(phases["wave-late"].values()),
        "qkv_vs_control_change_percent": pairs, "arms": arms,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("original", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.campaign, args.original), indent=2, allow_nan=False))
