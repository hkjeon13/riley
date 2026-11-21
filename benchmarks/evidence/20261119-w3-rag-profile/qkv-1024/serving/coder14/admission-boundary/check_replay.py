"""Replay both sides of request 45's observed admission boundary on the CPU."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ORDER = ("riley-first", "qkv-first", "qkv-second", "riley-second")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def replay(binary, admissions, lengths, windows):
    records = [(at, index, lengths[index]) for index, at in sorted(admissions.items())] + windows
    fixture = "".join(" ".join(map(str, fields)) + "\n" for fields in records)
    lines = subprocess.check_output([binary], text=True, input=fixture).splitlines()
    return [list(map(int, line.split())) for line in lines]


def check(inputs_path, binary, scheduler_dir):
    inputs = json.loads(inputs_path.read_bytes())
    results = {}
    for mode in ORDER:
        arm = inputs["arms"][mode]
        admissions = {int(index): at for index, at in arm["admissions_ns"].items()}
        lengths = {int(index): count for index, count in arm["prompt_tokens"].items()}
        windows = arm["step_windows_ns"]
        expected = arm["observed_prefill"]
        actual = replay(binary, admissions, lengths, windows)
        if actual != expected or actual[2] != [41, 0, 1024]:
            raise ValueError(f"{mode}: archived plans did not replay unchanged")
        was_early = admissions[45] < windows[2][0]
        if was_early:
            moved = windows[2][1] + 1
            if moved >= min(admissions[46], windows[3][0]):
                raise ValueError(f"{mode}: no window for a single admission change")
        else:
            moved = max(admissions[44], windows[1][1]) + 1
            if moved >= windows[2][0]:
                raise ValueError(f"{mode}: no window for a single admission change")
        alternative = replay(binary, {**admissions, 45: moved}, lengths, windows)
        other = mode.replace("qkv", "riley") if mode.startswith("qkv") else mode.replace("riley", "qkv")
        if alternative != inputs["arms"][other]["observed_prefill"] or alternative[:7] != expected[:7]:
            raise ValueError(f"{mode}: counterfactual did not reproduce the paired plan sequence")
        results[mode] = {"admission_45_relative_to_step_3_ns": admissions[45] - windows[2][0],
                         "changed_45_admission_ns": moved,
                         "original_plans": actual, "plans_with_only_45_admission_moved": alternative}
    return {"input_sha256": digest(inputs_path),
            "source_receipt_sha256": inputs["source_receipt_sha256"],
            "scheduler_source_sha256": {name: digest(scheduler_dir / name) for name in ("scheduler.rs", "config.rs")},
            "replay_binary_sha256": digest(binary),
            "scope": "Fixed recorded windows before decode; changes affect admission order only.",
            "arms": results}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("inputs", "binary", "scheduler_dir"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.inputs, args.binary, args.scheduler_dir), indent=2))
