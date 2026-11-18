"""Verify matched Riley attention/KV controls and summarize the two pinned cases."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from long_accuracy import same_completion, save


ARMS = {
    "baseline": ("dense-v1", "fused-split-v1"),
    "shared-kv": ("shared-kv-v1", "fused-split-v1"),
    "separate-write": ("dense-v1", "separate-v1"),
}
HF_FIRST_IDS = {"3": 5257, "9": 264}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def arm(root, mode, legacy_sha):
    folder = root / mode
    receipt = read(folder / "receipt.json")
    idle = read(folder / "idle-after.json")
    require(receipt["mode"] == mode and receipt["status"] == "passed"
            and receipt["server_exit"] == 0 and receipt["observed_active"], f"{mode}: arm incomplete")
    require(idle["phase"] == "idle" and idle["physical_index"] == 6 and not idle["other_processes"]
            and idle["allocation_bytes"] == idle["torch_reserved_bytes"] == 20 * 1024**3
            and (idle["gpu"], idle["pid"], idle["start_ticks"])
            == (receipt["gpu_uuid"], receipt["holder_pid"], receipt["holder_start_ticks"]),
            f"{mode}: GPU reservation did not return to idle")
    inputs = {Path(name).name: value for name, value in receipt["sha256"].items()}
    require(inputs["legacy.json"] == legacy_sha, f"{mode}: historical input digest changed")
    command = receipt["server_argv"]
    flags = dict(zip(command[2::2], command[3::2], strict=True))
    flags.pop("--bind")
    require((flags.pop("--prompt-attention"), flags.pop("--kv-cache-write")) == ARMS[mode],
            f"{mode}: unexpected runtime selection")
    for name in ("response.json", "shutdown-metrics.json"):
        require(digest(folder / name) == receipt["result_sha256"][name], f"{mode}: {name} changed")
    metrics = read(folder / "shutdown-metrics.json")
    require(not any(metrics[key] for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and len(metrics["allocation"]) == 4 and not any(metrics["allocation"].values()),
            f"{mode}: requests or native allocations remain after shutdown")
    output = read(folder / "response.json")
    require(output["mode"] == mode and set(output["cases"]) == HF_FIRST_IDS.keys(),
            f"{mode}: cases or arm label do not match")
    return (command[:2], flags, inputs, idle["gpu"]), output["cases"]


def compare(root, legacy):
    require((root / "campaign.exit").read_text().strip() == "0", "attention campaign did not complete")
    historical = read(legacy)["baseline"]
    controls = {mode: arm(root, mode, digest(legacy)) for mode in ARMS}
    start, reference = controls["baseline"]
    results = {}
    for mode, (setup, output) in controls.items():
        require(setup == start, f"{mode}: fixed inputs or server flags changed")
        arm_result = {}
        for case, row in output.items():
            original = historical[int(case)]
            trial = row["response"]
            initial = reference[case]["response"]
            require(row["prompt_sha256"] == original["prompt_sha256"]
                    and trial["prompt_tokens"] == original["prompt_tokens"] and trial["tokens"],
                    f"{mode}/{case}: prompt or output is invalid")
            same = same_completion(trial, original)
            require(row["matches_original_ids_text_finish"] == same and (same or mode != "baseline"),
                    f"{mode}/{case}: original output check disagrees")
            first = trial["tokens"][0]
            options = trial["logprobs"]["top_logprobs"][0]
            reference_options = initial["logprobs"]["top_logprobs"][0]
            arm_result[case] = {"first_id": first, "matches_historical_completion": same,
                                "matches_hf_first_id": first == HF_FIRST_IDS[case],
                                "first_top_logprobs_match_baseline": options == reference_options,
                                "first_top_logprobs": options}
        results[mode] = arm_result
    return {"physical_gpu": 6, "binary_sha256": start[2]["riley"], "arms": results}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("root", "legacy", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    save(args.output, compare(args.root, args.legacy))
