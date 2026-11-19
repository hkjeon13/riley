"""Sweep one engine command across the measured early wave windows."""

from argparse import ArgumentParser
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess


ORDER = ("riley-first", "qkv-first", "qkv-second", "riley-second")
EXPECTED_PAIR_PREFIX = ((45, 41), (41, 45))
STEP_INDEX = 2


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def replay(binary, inputs, admissions):
    rows = [(stamp, index, inputs["prompt_tokens"][index]) for index, stamp in sorted(admissions.items())]
    rows.extend(inputs["step_windows_ns"])
    data = "".join(" ".join(map(str, row)) + "\n" for row in rows)
    process = subprocess.run([binary], input=data, check=True, capture_output=True, text=True, timeout=10)
    return [list(map(int, row.split())) for row in process.stdout.splitlines()]


def possible_admission(send, admission_44, admission_46, windows):
    # Hold the neighboring requests and the measured plan windows fixed.
    # Admission at a plan start participates in that plan.
    eligible = max(send, admission_44 + 1)
    for start, end in windows:
        if start < eligible <= end:
            eligible = end + 1
    if eligible >= admission_46:
        raise ValueError("sweep changed the input ordering around request 45")
    return eligible


def sweep(source, binary, scheduler_directory):
    witness = json.loads(source.read_bytes())
    arms = witness["arms"]
    if set(arms) != set(ORDER) or witness["admission_changed_in_counterfactual"] != 45:
        raise ValueError("unexpected W3 wave witness")
    source_hashes = {name: digest(scheduler_directory / name) for name in ("scheduler.rs", "config.rs")}
    if source_hashes != witness["scheduler_inputs_sha256"]:
        raise ValueError("the local scheduler source differs from the measured build")

    boundaries = {}
    observed_sends = {}
    for label in ORDER:
        arm = arms[label]
        data = arm["cpu_replay_inputs"]
        reached = replay(binary, data, data["admissions_ns"])
        baseline = arm["observed_early_plans"]
        if reached != baseline or len(baseline) != 9:
            raise ValueError(f"{label}: baseline schedule did not replay exactly")
        starts = data["step_windows_ns"][STEP_INDEX][0]
        observed = starts + arm["request_45_send_minus_wave_step_3_start_ns"]
        boundaries[label] = starts
        observed_sends[label] = observed

    sends = set(range(244_000_000, 255_000_001, 250_000))
    sends.update(observed_sends.values())
    sends.update(stamp + shift for stamp in boundaries.values() for shift in (-1, 0, 1))
    outcomes = {}
    for stamp in sorted(sends):
        by_arm = {}
        for label in ORDER:
            arm = arms[label]
            data = arm["cpu_replay_inputs"]
            arrivals = data["admissions_ns"]
            admission = possible_admission(stamp, arrivals["44"], arrivals["46"], data["step_windows_ns"])
            played = replay(binary, data, {**arrivals, "45": admission})
            if played[:7] != arm["observed_early_plans"][:7]:
                raise ValueError(f"{label} at {stamp}: another early plan changed")
            requests = tuple(row[0] for row in played[7:9])
            expected = EXPECTED_PAIR_PREFIX[int(stamp > boundaries[label])]
            if requests != expected:
                raise ValueError(f"{label} at {stamp}: first affected turns differ")
            if stamp == observed_sends[label] and played != arm["observed_early_plans"]:
                raise ValueError(f"{label}: observed command time did not reproduce the recorded sequence")
            by_arm[label] = {"modeled_admission_ns": admission, "prefill_steps_8_and_9": requests}
        outcomes[str(stamp)] = by_arm

    bands = {}
    for order in ("first", "second"):
        lo, hi = boundaries[f"qkv-{order}"], boundaries[f"riley-{order}"]
        if lo >= hi:
            raise ValueError("the candidate step did not start first")
        bands[order] = {"lower_exclusive_ns": lo, "upper_inclusive_ns": hi, "span_ns": hi - lo}
    return {
        "scope": "One engine command time relative to the first admission of the same wave. The other seven admissions and observed prefill windows are fixed independently in each arm.",
        "limits": "Idealized admission at the first available plan boundary after command send and request 44. Result covers the first nine CPU prefill plans.",
        "wave_receipt_sha256": digest(source),
        "scheduler_inputs_sha256": source_hashes,
        "cpu_replay_binary_sha256": digest(binary),
        "verified_original_plans": len(ORDER) * 9,
        "common_command_times": len(sends),
        "scheduler_replays_in_sweep": len(sends) * len(ORDER),
        "first_and_last_command_ns": [min(sends), max(sends)],
        "observed_command_ns": observed_sends,
        "before_after_split_by_execution_order_ns": bands,
        "example_outcomes_by_command_time_ns": {key: outcomes[key] for key in ("245000000", "249000000", "253000000")},
        "full_sweep_outcomes_sha256": sha256(json.dumps(outcomes, separators=(",", ":")).encode()).hexdigest(),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("wave_receipt", type=Path)
    parser.add_argument("local_replay_binary", type=Path)
    parser.add_argument("scheduler_source_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(sweep(args.wave_receipt, args.local_replay_binary, args.scheduler_source_directory),
                     indent=2, allow_nan=False))
