"""Verify paired native CUDA startup, API, output parity and reservation evidence."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re

ORDER = {6: ("baseline", "candidate"), 7: ("candidate", "baseline")}
GPUS = {
    6: "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7",
    7: "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0",
}


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_events(directory):
    pattern = r"^(START|END) gpu([67]) (baseline|candidate) (\S+?)(?: exit=(\d+))?$"
    events = re.findall(pattern, (directory / "campaign.log").read_text(), re.MULTILINE)
    require(len(events) == 8, "not all four native runs completed")
    timestamps = {}
    for phase, index, engine, timestamp, exit_code in events:
        key = (int(index), engine, phase)
        require(exit_code == ("0" if phase == "END" else "") and key not in timestamps, "failed or duplicate run")
        timestamps[key] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    for index, (first, second) in ORDER.items():
        require(timestamps[index, first, "START"] <= timestamps[index, first, "END"]
                <= timestamps[index, second, "START"] <= timestamps[index, second, "END"],
                f"GPU {index} run order changed")


def check_reservations(directory, index):
    out = directory / f"gpu{index}"
    rows = [read(out / f"{phase}.json") for phase in ("before", "between", "after")]
    before = rows[0]
    for row in rows:
        require(row["status"] == "running" and row["phase"] == "idle" and row["gpu"] == GPUS[index]
                and row["physical_index"] == index and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
                and not row["other_processes"] and (row["pid"], row["start_ticks"]) == (before["pid"], before["start_ticks"]),
                f"the same 20GiB holder did not recover on GPU {index}")


def run(directory, index, name, binary, affinity):
    out = directory / f"gpu{index}" / name
    receipt = read(out / "receipt.json")
    require(receipt["status"] == "passed" and receipt["server_exit"] == 0, "native serving arm failed")
    for member in ("api", "plan", "holdout", "shutdown"):
        actual = sha256((out / f"{member}.json").read_bytes()).hexdigest()
        require(actual == receipt[f"{member}_sha256"], f"{name} {member} changed on GPU {index}")
    plan = read(out / "plan.json")
    require(plan["binary_sha256"] == binary and plan["gpu"] == GPUS[index]
            and plan["affinity"] == affinity, f"runtime identity or CPU affinity changed on GPU {index}")
    bind = plan["args"].index("--bind") + 1
    args = plan["args"]
    require(args[bind].startswith("127.0.0.1:"), "serving did not bind to loopback")
    comparable = [*args[1:bind], *args[bind + 1:]]
    ready = read(out / "ready.json")
    seconds = ready["elapsed_s"]
    require(ready["readiness"]["ready"] is True and ready["readiness"]["accepting"] is True
            and seconds == receipt["readiness_seconds"] and math.isfinite(seconds) and 0 < seconds < 480,
            "readiness receipt is incomplete")
    shutdown = read(out / "shutdown.json")
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and not any(shutdown["allocation"].values()), "native shutdown retained resources")
    api = read(out / "api.json")
    require(api["model"] == "m" and len(api["results"]) == 33
            and all(row["passed"] is True for row in api["results"]),
            "API contract failed")
    holdout = read(out / "holdout.json")
    require(len(holdout) == 8 and all(
        all(re.fullmatch(r"[0-9a-f]{64}", row[key]) for key in (
            "input_sha256", "output_sha256", "prompt_ids_sha256", "token_ids_sha256"
        )) and row["prompt_tokens"] > 0 and 0 < row["completion_tokens"] <= 64 for row in holdout
    ), "token ID holdout is incomplete")
    return comparable, seconds, api["results"], holdout


def verify(directory):
    require((directory / "campaign.exit").read_text().strip() == "0", "A100 comparisons did not complete")
    check_events(directory)
    binaries = dict((name.removeprefix("riley-"), digest) for digest, name in
                    (line.split() for line in (directory / "server-binaries.sha256").read_text().splitlines()))
    require(set(binaries) == {"baseline", "candidate"} and len(set(binaries.values())) == 2
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in binaries.values()), "missing distinct native binary identities")
    cpus = [sorted(map(int, group.split(","))) for group in (directory / "affinity.txt").read_text().split()]
    require(len(cpus) == 2 and all(len(group) == 2 for group in cpus)
            and len(set(cpus[0] + cpus[1])) == 4, "expected separate two-core CPU affinities")
    comparison = {}
    reference_outputs = None
    for index, affinity in zip(ORDER, cpus, strict=True):
        check_reservations(directory, index)
        baseline, candidate = [run(directory, index, variant, binaries[variant], affinity)
                               for variant in ("baseline", "candidate")]
        require(baseline[0] == candidate[0] and baseline[3] == candidate[3], f"config or output changed on GPU {index}")
        require([row["name"] for row in baseline[2]] == [row["name"] for row in candidate[2]],
                "API check names changed")
        if reference_outputs is None:
            reference_outputs = baseline[3]
        require(reference_outputs == baseline[3], "outputs changed between GPUs")
        comparison[index] = {
            "cpu_affinity": affinity, "baseline_seconds": baseline[1], "candidate_seconds": candidate[1],
            "elapsed_reduction_pct": (1 - candidate[1] / baseline[1]) * 100,
            "api_checks_per_arm": len(baseline[2]), "matching_token_id_responses": len(baseline[3]),
        }
    return {"native_binary_sha256": binaries, "run_order": ORDER, "gpu_results": comparison}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2, allow_nan=False))
