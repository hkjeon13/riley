"""Bind the SmolLM2 A100 comparison to its binaries, inputs and traces."""

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import re
from statistics import median

from compare_profile_arms import HOLDER_BYTES, METRICS, arm, digest, read, require
from summarize_profile_arm import stable_server


ORDER = ("base-first", "ordered-first", "ordered-second", "base-second", "base-trace", "ordered-trace")
CELLS = ("reason-c32", "long-c32")
INPUTS = ("request_index", "input_sha256", "prompt_tokens")
OUTPUTS = ("output_sha256", "completion_tokens", "finish")
IDENTITY = ("model", "gpu", "common_kv_blocks", "kv_cache_bytes", "holder_pid", "holder_start_ticks")


def checked_order(root):
    require((root / "campaign.exit").read_text().strip() == "0", "campaign did not finish successfully")
    events = [line.split() for line in (root / "campaign.log").read_text().splitlines()
              if line.startswith(("START ", "END "))]
    require(len(events) == len(ORDER) * 2, "campaign chronology is incomplete")
    times = {}
    prior = ""
    for mode, start, end in zip(ORDER, events[::2], events[1::2], strict=True):
        require(len(start) == 3 and len(end) == 4 and start[:2] == ["START", mode]
                and end[:2] == ["END", mode] and end[3] == "exit=0" and prior <= start[2] < end[2],
                f"unexpected order or exit: {mode}")
        times[mode] = {"start_utc": start[2], "end_utc": end[2]}
        prior = end[2]
    return times


def binary_hashes(root):
    records = (root / "binaries.sha256").read_text().splitlines()
    expected = ["bin/base/riley", "bin/ordered/riley"]
    hashes = {}
    for record, path in zip(records, expected, strict=True):
        checksum, actual = record.split()
        require(actual == path and re.fullmatch(r"[0-9a-f]{64}", checksum) is not None,
                "malformed build checksum")
        hashes[Path(path).parent.name] = checksum
    require(len(set(hashes.values())) == 2, "baseline and candidate are not distinct builds")
    return hashes


def check_common(root, mode, receipt, plan, hashes):
    variant = mode.split("-", 1)[0]
    traced = mode.endswith("-trace")
    directory = root / f"smol-{mode}"
    expected_env = {"RILEY_HOST_STEP_TRACE": "1"} if traced else {}
    require(plan["env"] == expected_env and plan["profile"] is traced
            and plan["check_token_ids"] is not traced, f"{mode}: diagnostic knobs changed")
    require(plan["server"][0].endswith(f"/bin/{variant}/riley") and receipt["nvml_samples"] > 0,
            f"{mode}: binary or GPU samples are missing")
    before = read(root / f"idle-before-{mode}.json")
    after = read(directory / "idle-after.json")
    require(all(before[key] == after[key] for key in before if key != "updated_ns")
            and after["phase"] == "idle" and after["physical_index"] == 7
            and after["allocation_bytes"] == after["torch_reserved_bytes"] == HOLDER_BYTES
            and after["pid"] == receipt["holder_pid"] == 2431237 and not after["other_processes"]
            and after["gpu"] == receipt["gpu"] and after["start_ticks"] == receipt["holder_start_ticks"],
            f"{mode}: original GPU reservation was not restored")
    require(read(directory / "server-exit.json")["exit"] == 0, f"{mode}: server exit was not clean")
    inputs = {Path(path).name: checksum for path, checksum in receipt["sha256"].items()}
    require(inputs.pop("riley") == hashes[variant] and inputs.pop("plan.json") == digest(directory / "plan.json"),
            f"{mode}: running binary or plan hash changed")
    inputs.pop("nsys", None)
    require(receipt["model"] == "SmolLM2-1.7B-Instruct" and receipt["common_kv_blocks"] == 9006,
            f"{mode}: model or KV budget changed")
    return {"identity": {key: receipt[key] for key in IDENTITY}, "inputs_sha256": inputs,
            "server_flags": stable_server(plan["server"])[1:]}


def run_cell(specs, reference, candidate):
    a_stats, a_rows = reference
    b_stats, b_rows = candidate
    require(a_stats["config"] == b_stats["config"] == specs, "workload differs")
    pairs = list(zip(a_rows, b_rows, strict=True))
    require(all(all(a[key] == b[key] for key in INPUTS) for a, b in pairs), "request inputs differ")
    mismatches = [a["request_index"] for a, b in pairs if any(a[key] != b[key] for key in OUTPUTS)]
    metrics = {}
    for metric in METRICS:
        before, after = a_stats[metric], b_stats[metric]
        require(all(math.isfinite(value) and value > 0 for value in (before, after)), f"invalid {metric}")
        metrics[metric] = {"control": before, "candidate": after, "change_percent": (after / before - 1) * 100}
    guards = {"exact_outputs": not mismatches, "decode_tpot_under_1_percent_regression":
              metrics["tpot_p50_ms"]["change_percent"] < 1}
    if specs["prompt_words"] == [30]:
        guards.update(short_ttft_under_2_percent_regression=metrics["ttft_p95_ms"]["change_percent"] < 2,
                      short_throughput_under_2_percent_regression=metrics["output_tok_s"]["change_percent"] > -2)
    return {"complete_requests": len(pairs), "warmup_requests": specs["warmup"],
            "output_mismatch_request_indices": mismatches, "metrics": metrics, "guards": guards}


def probe(directory, receipt):
    path = directory / "token-id-probe.json"
    record = read(path)
    require(digest(path) == receipt["token_id_probe_sha256"] and len(record["c1"]) == len(record["c8"]) == 8
            and record["invariant"] == (record["c1"] == record["c8"]), "token ID probe is incomplete")
    return record, digest(path)


def read_trace(root, mode, hashes, shared):
    directory = root / f"smol-{mode}"
    plan = read(directory / "plan.json")
    receipt = read(directory / "receipt.json")
    require(receipt["mode"] == mode and receipt["status"] == "passed" and tuple(plan["cells"]) == CELLS[:1],
            f"{mode}: incomplete trace")
    require(check_common(root, mode, receipt, plan, hashes) == shared, f"{mode}: trace inputs differ")
    sample = read(directory / "reason-c32.json")
    statistics = sample["statistics"]
    require(receipt["results"]["reason-c32"] == digest(directory / "reason-c32.json")
            and statistics["config"] == plan["cells"]["reason-c32"]
            and statistics["ok"] == len(sample["measured"]) == 32
            and not (statistics["warmup_ok"] or statistics["errors"] or statistics["rejected_429"]),
            f"{mode}: short or mismatched Nsight workload")
    timeline = read(directory / "sync-gaps.json")
    require(timeline["sqlite_sha256"] == receipt["sqlite_sha256"] and timeline["adjacent_matched_gaps"] >= 2,
            f"{mode}: decode graph timeline was not bound to this GPU capture")
    log = directory / "server.log"
    steps = [{key: float(value) for key, value in re.findall(r"([a-z+]+)=(\d+(?:\.\d+)?)", line.split("|", 1)[0])}
             for line in log.read_text().splitlines() if line.startswith("HOSTSTEP us/step over 128 steps:")]
    require(steps and all(len(row) == 12 and row.keys() == steps[0].keys() for row in steps),
            f"{mode}: missing host timing windows")
    return {"workload": (statistics, sample["measured"]), "sync_gaps": timeline,
            "nsys_sha256": receipt["sha256"][next(path for path in receipt["sha256"] if Path(path).name == "nsys")],
            "report_sha256": receipt["raw_report_sha256"], "log_sha256": digest(log),
            "host_trace": {"windows_128_steps": len(steps),
                           "median_us_per_step": {key: median(row[key] for row in steps) for key in steps[0]}}}


def verify(root):
    times = checked_order(root)
    hashes = binary_hashes(root)
    measured = {}
    probes = {}
    bindings = {}
    for mode in ORDER[:4]:
        receipt, plan, measured[mode] = arm(root, "smol", mode, expected_cells=CELLS)
        bindings[mode] = check_common(root, mode, receipt, plan, hashes)
        probes[mode] = probe(root / f"smol-{mode}", receipt)
    shared = bindings["base-first"]
    require(all(binding == shared for binding in bindings.values()), "matched serving inputs differ")
    metrics = {}
    for cell in CELLS:
        metrics[cell] = {}
        for order in ("first", "second"):
            control = measured[f"base-{order}"][cell]
            candidate = measured[f"ordered-{order}"][cell]
            metrics[cell][order] = run_cell(control[0]["config"], control, candidate)
    traces = {mode: read_trace(root, mode, hashes, shared) for mode in ORDER[4:]}
    before, after = (traces[mode].pop("workload") for mode in ORDER[4:])
    require(traces[ORDER[4]]["nsys_sha256"] == traces[ORDER[5]]["nsys_sha256"], "Nsight build differs")
    return {"scope": "SmolLM2-1.7B on one A100. Serving deltas come from the uninstrumented comparison.",
            "gpu_index": 7, "campaign_times_utc": times, "binaries_sha256": hashes, "inputs": shared,
            "same_20gib_holder_restored": True, "serving": metrics,
            "token_id_probe": {"cases": 8, "sha256": {mode: checksum for mode, (_, checksum) in probes.items()},
                               "all_c1_c8_and_both_builds_exact": all(value[0]["invariant"]
                                   and value[0]["c1"] == probes["base-first"][0]["c1"] for value in probes.values())},
            "nsight": {"arms": traces, "trace_workload_output_check": run_cell(before[0]["config"], before, after)}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, allow_nan=False))
