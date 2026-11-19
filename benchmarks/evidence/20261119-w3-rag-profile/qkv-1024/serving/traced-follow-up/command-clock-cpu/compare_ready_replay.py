"""Compare observed prefixes against a CPU scheduler using command send for initial readiness."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess


CELLS = ("wave-early", "wave-nominal", "wave-late")
ARMS = ("riley-first", "qkv-first", "qkv-second", "riley-second")
CAPS_NS = (0, *(1_000 * 2**power for power in range(18)))


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def run(binary, arguments, rows, windows):
    payload = "".join(" ".join(map(str, row)) + "\n" for row in (*rows, *windows))
    result = subprocess.run([binary, *arguments], input=payload, text=True, check=True,
                            capture_output=True, timeout=10)
    return [line if line.startswith("non-") else list(map(int, line.split()))
            for line in result.stdout.splitlines()]


def prefix_bound(original, inputs):
    thresholds = {}
    for arm, cases in inputs.items():
        thresholds[arm] = {}
        for name, (points, windows) in cases.items():
            first_start, first_end = windows[0]
            second_start, _ = windows[1]
            first_two = original[arm][name][:2]
            if [row[0] for row in points if row[2] <= first_start] != [40] or [row[0] for row in first_two] != [40, 40]:
                raise ValueError("the prefill prefix cannot prove a cap bound")
            # Request 40 is ready again at the first return. At equal readiness
            # it wins the tie because it was admitted first.
            caps = [admit - first_end + 1 for index, send, admit, _ in points
                    if index != 40 and send < first_end <= admit <= second_start]
            if not caps:
                raise ValueError("missing contender at the second turn")
            thresholds[arm][name] = min(caps)
    return thresholds, min(cap for values in thresholds.values() for cap in values.values())


def relevant_caps(inputs, first_unsafe):
    probes = {0, first_unsafe - 1}
    for cases in inputs.values():
        for points, windows in cases.values():
            fixed = {send for _, send, _, _ in points} | {end for _, end in windows}
            for _, _, admit, _ in points:
                for timestamp in fixed:
                    for offset in (-1, 0, 1):
                        cap = admit - timestamp + offset
                        if 0 <= cap < first_unsafe:
                            probes.add(cap)
    return probes


def compare(archive, published_file, source, patch_file, stock, probe):
    witness, published = read(archive), read(published_file)
    if witness["published_comparison_sha256"] != digest(published_file):
        raise ValueError("replay fixtures are from a different GPU comparison")
    rows_by_arm, original, inputs = {}, {}, {}
    for arm in ARMS:
        expected, output = published["arms"][arm], witness["arms"][arm]
        if expected["server_log_sha256"] != output["server_log_sha256"] or set(output["cells"]) != set(CELLS):
            raise ValueError(f"{arm}: GPU trace provenance differs")
        arm_cases, original[arm], inputs[arm] = {}, {}, {}
        for name in CELLS:
            fixture = output["cells"][name]
            baseline = fixture["observed_plans"]
            points = fixture["requests_index_send_admit_length"]
            windows = fixture["step_windows_ns"]
            if (expected["cells"][name]["result_sha256"] != fixture["cell_sha256"]
                    or any(len(row) != 3 for row in baseline)
                    or [[row[0]] for row in baseline[-2:]] != expected["cells"][name]["fifth_wave"]["prefill_turns_8_and_9"]
                    or len(baseline) != len(windows) or len(baseline) != 9
                    or [row[0] for row in points] != list(range(40, 48))):
                raise ValueError(f"{arm}/{name}: fixture identity differs")
            stock_inputs = [(admit, index, length) for index, _, admit, length in points]
            if run(stock, (), stock_inputs, windows) != baseline or run(probe, ("admit",), points, windows) != baseline:
                raise ValueError(f"{arm}/{name}: reference schedules do not replay exactly")
            original[arm][name], inputs[arm][name] = baseline, (points, windows)
            trial = run(probe, ("send",), points, windows)
            change = next((i + 1 for i, (left, right) in enumerate(zip(baseline, trial)) if left != right), None)
            if change is None and len(trial) != len(baseline):
                raise ValueError(f"{arm}/{name}: inconsistent simulation length")
            arm_cases[name] = {
                "observed_turns": [row[0] for row in baseline],
                "first_different_turn": change,
                "first_different_plan": None if change is None else {"observed": baseline[change - 1],
                                                                    "candidate": trial[change - 1]},
                "fixed_clock_candidate_plans": trial,
            }
        rows_by_arm[arm] = arm_cases
    paired = {}
    for order in ("first", "second"):
        pairs = {}
        for name in CELLS:
            a, b = [rows_by_arm[f"{mode}-{order}"][name] for mode in ("riley", "qkv")]
            pairs[name] = {
                "previous_identical_plans_through_turn_9": a["observed_turns"] == b["observed_turns"],
                "fixed_clock_candidate_identical_through_turn_9": (
                    len(a["fixed_clock_candidate_plans"]) == len(b["fixed_clock_candidate_plans"]) == 9
                    and a["fixed_clock_candidate_plans"] == b["fixed_clock_candidate_plans"]),
            }
        paired[order] = pairs
    thresholds, first_unsafe = prefix_bound(original, inputs)
    breakpoint_probes = relevant_caps(inputs, first_unsafe)
    cap_results = []
    for cap in sorted(breakpoint_probes | set(CAPS_NS) | {first_unsafe}):
        results = {arm: {name: run(probe, (str(cap),), *inputs[arm][name]) for name in CELLS} for arm in ARMS}
        early_prefix_count = sum(results[arm][name][:7] == original[arm][name][:7] for arm in ARMS for name in CELLS)
        shifted_control_count = sum(results[arm][name] == original[arm][name] for arm in ARMS
                                    for name in ("wave-early", "wave-late"))
        stable_nominal_count = sum(len(results[f"riley-{order}"]["wave-nominal"]) == 9
            and results[f"riley-{order}"]["wave-nominal"] == results[f"qkv-{order}"]["wave-nominal"]
            for order in ("first", "second"))
        cap_results.append({"max_adjustment_ns": cap, "unchanged_first_seven_cells": early_prefix_count,
                            "unchanged_shifted_control_cells": shifted_control_count,
                            "nominal_pairs_with_same_first_nine_plans": stable_nominal_count})
    if cap_results[0] != {"max_adjustment_ns": 0, "unchanged_first_seven_cells": 12,
                          "unchanged_shifted_control_cells": 8, "nominal_pairs_with_same_first_nine_plans": 0}:
        raise ValueError("zero adjustment did not match the baseline")
    if next(row for row in cap_results if row["max_adjustment_ns"] == first_unsafe)["unchanged_first_seven_cells"] == 12:
        raise ValueError("priority conflict at the analytical bound did not replay")
    eligible = [row["max_adjustment_ns"] for row in cap_results if row["unchanged_first_seven_cells"] == 12
                and row["unchanged_shifted_control_cells"] == 8 and row["nominal_pairs_with_same_first_nine_plans"] == 2]
    return {
        "scope": "CPU counterfactual using engine command send only for initial readiness of immediately admitted requests.",
        "limits": "Observed execution windows and actual admission boundaries are fixed. After a changed plan, subsequent order is advisory. No GPU timing or output is predicted.",
        "verified_records": {"cells": 12, "unchanged_scheduler_plans": 108, "candidate_admission_clock_plans": 108},
        "bindings_sha256": {"timing_fixture": digest(archive), "published_gpu_comparison": digest(published_file),
                           "stock_scheduler_source": digest(source), "candidate_patch": digest(patch_file),
                           "stock_replay_executable": digest(stock), "candidate_replay_executable": digest(probe)},
        "arms": rows_by_arm, "pairs": paired,
        "first_cap_changing_turn_2_in_any_cell_ns": first_unsafe,
        "turn_2_cap_thresholds_ns": thresholds,
        "priority_breakpoint_probes_below_threshold_ns": sorted(breakpoint_probes),
        "bounded_adjustment_sweep": cap_results, "caps_preserving_prefixes_and_stabilizing_nominal_ns": eligible,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("archive", "published", "stock_source", "patch", "stock", "probe"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.archive, args.published, args.stock_source, args.patch, args.stock, args.probe),
                     indent=2, allow_nan=False))
