"""Retain the failed wrapper exit while validating recovered W2 trace intervals."""

from argparse import ArgumentParser
import json
from pathlib import Path
import re
from statistics import median

from compare_profile_arms import HOLDER_BYTES, arm, digest, read, require
from summarize_profile_arm import stable_server
from verify_ordered_tables import CELLS, IDENTITY, binary_hashes, check_common, probe, run_cell


SERVING = ("base-first", "ordered-first", "ordered-second", "base-second")
TRACE = ("base-trace", "ordered-trace")


def uninstrumented(root, hashes):
    measured = {}
    bindings = {}
    probes = {}
    for mode in SERVING:
        receipt, plan, measured[mode] = arm(root, "smol", mode, expected_cells=CELLS)
        bindings[mode] = check_common(root, mode, receipt, plan, hashes)
        probes[mode] = probe(root / f"smol-{mode}", receipt)
    shared = bindings[SERVING[0]]
    require(all(binding == shared for binding in bindings.values()), "different serving inputs")
    require(all(data["invariant"] and data["c1"] == probes[SERVING[0]][0]["c1"] for data, _ in probes.values()),
            "C1/C8 token IDs did not match within and across the four runs")
    cells = {}
    for cell in CELLS:
        cells[cell] = {}
        for order in ("first", "second"):
            first = measured[f"base-{order}"][cell]
            second = measured[f"ordered-{order}"][cell]
            cells[cell][order] = run_cell(first[0]["config"], first, second)
    return shared, cells, {mode: value[1] for mode, value in probes.items()}


def read_trace(root, mode, shared, binaries, source_hashes):
    directory = root / f"smol-{mode}"
    report = read(directory / "receipt.json")
    plan = read(directory / "plan.json")
    recovered = mode == "ordered-trace"
    variant = mode.split("-")[0]
    pid = int((directory / "server.pid").read_text())
    log_file = directory / "server.log"
    hashes = {Path(path).name: checksum for path, checksum in report["sha256"].items()}
    require(hashes.pop("riley") == binaries[variant] and hashes.pop("plan.json") == digest(directory / "plan.json"),
            f"{mode}: trace loaded a different binary or plan")
    nsys = hashes.pop("nsys")
    require(hashes == shared["inputs_sha256"] and stable_server(plan["server"])[1:] == shared["server_flags"]
            and {key: report[key] for key in IDENTITY} == shared["identity"], f"{mode}: workload inputs differ")
    require(tuple(plan["cells"]) == CELLS[:1] and plan["profile"] and not plan["check_token_ids"]
            and plan["env"] == {"RILEY_HOST_STEP_TRACE": "1"} and report["nvml_samples"] > 0,
            f"{mode}: trace mode, token check, or NVML coverage changed")
    before = read(root / f"idle-before-{mode}.json")
    after = read(directory / "idle-after.json")
    require(all(before[key] == after[key] for key in before if key != "updated_ns") and after["phase"] == "idle"
            and after["pid"] == report["holder_pid"] == 2431237 and after["start_ticks"] == report["holder_start_ticks"]
            and after["gpu"] == report["gpu"] and after["physical_index"] == 7 and not after["other_processes"]
            and after["allocation_bytes"] == after["torch_reserved_bytes"] == HOLDER_BYTES,
            f"{mode}: 20 GiB holder was not restored")
    if recovered:
        require(report["status"] == "failed" and report["error"] == f"RuntimeError: unexpected process on selected GPU pid={pid}"
                and not (directory / "server-exit.json").exists(), "unexpected candidate trace failure")
        clock_path = root / "ordered-clock.json"
    else:
        require(report["status"] == "passed" and read(directory / "server-exit.json")["exit"] == 0,
                "reference trace did not stop cleanly")
        clock_path = directory / "reason-c32.profile.json"
    stats_path = directory / "reason-c32.json"
    sample = read(stats_path)
    stats, rows = sample["statistics"], sample["measured"]
    require(stats["config"] == plan["cells"]["reason-c32"] and stats["ok"] == len(rows) == 32
            and not (stats["warmup_ok"] or stats["errors"] or stats["rejected_429"]), f"{mode}: incomplete trace requests")
    clock = read(clock_path)
    require(clock["server_pid"] == pid and clock["requests"] == 32
            and [row["request_index"] for row in rows] == [row["request_index"] for row in clock["measurements"]]
            and max(clock["clock_bracket_ns"]) <= 1_000_000 and clock["clock_offset_change_ns"] <= 1_000_000,
            f"{mode}: requests do not bind to this process on the captured clock")
    timeline = read(root / f"{variant}-sync-gaps.json")
    if not recovered:
        require(report["results"]["reason-c32"] == digest(stats_path) and timeline["sqlite_sha256"] == report["sqlite_sha256"]
                == source_hashes["smol-base-trace/trace.sqlite"], "reference trace changed")
    windows = [{name: float(value) for name, value in re.findall(r"([a-z+]+)=(\d+(?:\.\d+)?)", line.split("|", 1)[0])}
               for line in log_file.read_text().splitlines() if line.startswith("HOSTSTEP us/step over 128 steps:")]
    require(windows and all(len(row) == 12 and row.keys() == windows[0].keys() for row in windows), "missing host timings")
    observed = {"original_wrapper_status": report["status"], "server_exit_observed": not recovered,
                "server_pid": pid, "clock_bracket_ns": clock["clock_bracket_ns"],
                "clock_offset_change_ns": clock["clock_offset_change_ns"], "matched_requests": 32,
                "host_windows_128_steps": len(windows), "host_median_us_per_step": {key: median(row[key] for row in windows)
                                                                                     for key in windows[0]},
                "file_sha256": {"requests": digest(stats_path), "clock": digest(clock_path), "server_log": digest(log_file)},
                "graph_gaps": timeline}
    return observed, nsys, (stats, rows)


def verify(serving_root, traces_root):
    require((traces_root / "campaign.exit").read_text().strip() == "1", "expected the retained failed wrapper")
    binaries = binary_hashes(serving_root)
    require(binary_hashes(traces_root) == binaries, "trace and uninstrumented binary digests differ")
    chronology = [line.split() for line in (traces_root / "campaign.log").read_text().splitlines()
                  if line.startswith(("START ", "END "))]
    require(len(chronology) == 2 * (len(SERVING) + len(TRACE)), "incomplete chronology")
    times = {}
    for mode, start, end in zip((*SERVING, *TRACE), chronology[::2], chronology[1::2], strict=True):
        exit_code = 1 if mode == "ordered-trace" else 0
        require(len(start) == 3 and len(end) == 4 and start[:2] == ["START", mode] and end[:2] == ["END", mode]
                and end[3] == f"exit={exit_code}" and start[2] < end[2], "unexpected execution record")
        times[mode] = {"start_utc": start[2], "end_utc": end[2], "exit": exit_code}
    shared, serving, probes = uninstrumented(serving_root, binaries)
    source_hashes = {file: digest for digest, file in (line.split() for line in (traces_root / "input.sha256").read_text().splitlines())}
    base, candidate = (read_trace(traces_root, mode, shared, binaries, source_hashes) for mode in TRACE)
    require(base[1] == candidate[1], "different Nsight versions")
    trace_cohort = run_cell(base[2][0]["config"], base[2], candidate[2])
    require(trace_cohort["guards"]["exact_outputs"], "trace output differs")
    require(base[0]["graph_gaps"]["matched_kernel_counts"] == candidate[0]["graph_gaps"]["matched_kernel_counts"],
            "different matched graph shapes")
    return {"scope": "Uninstrumented serving comparison and recovered pre-shutdown Nsight intervals on A100 7.",
            "campaign_wrapper_exit": 1, "campaign_times_utc": times, "binaries_sha256": binaries,
            "shared_inputs": shared, "serving": serving, "all_four_c1_c8_token_probes_exact": True,
            "token_probes_sha256": probes, "same_20gib_holder_restored_after_each_arm": True,
            "trace_cohort_exact_requests": trace_cohort["complete_requests"],
            "profile_source_sha256": source_hashes, "nsys_sha256": base[1],
            "trace": {"base": base[0], "candidate": candidate[0]}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("serving", type=Path)
    parser.add_argument("traces", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.serving, args.traces), indent=2, allow_nan=False))
