"""Verify the timed sharded-checkpoint validation and recompute the comparison."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
from statistics import median

ORDER = ("01-baseline", "02-candidate", "03-candidate", "04-baseline")
CHECKPOINT_SHA256 = "82a56cdab6e16e31478b7449be768dce15992b448d543fea4b2f2ed163237d00"
SOURCE_SHA256 = "7dc88437ba28ea18ca1c0c11b9b3cd85324110da913ce1b10c521da60ce54dcb"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def trial(directory, name):
    raw = dict(line.split("=", 1) for line in (directory / f"{name}.time").read_text().splitlines())
    require(set(raw) == {"wall", "user", "system", "rss_kib", "major_faults", "minor_faults", "exit"},
            f"unexpected timing fields for {name}")
    require(raw["exit"] == "0" and (directory / f"{name}.stderr").stat().st_size == 0
            and (directory / f"{name}.stdout").read_text().strip() == "checkpoint verify: verified",
            f"{name} did not verify the checkpoint")
    times = {key: float(raw[key]) for key in ("wall", "user", "system")}
    require(all(math.isfinite(value) and value >= 0 for value in times.values()) and times["wall"] > 0,
            f"invalid timing for {name}")
    memory = {key: int(raw[key]) for key in ("rss_kib", "major_faults", "minor_faults")}
    require(memory["rss_kib"] > 0 and all(value >= 0 for value in memory.values()), f"invalid memory data for {name}")
    return {"elapsed_s": times["wall"], "user_s": times["user"], "system_s": times["system"],
            "average_cores": (times["user"] + times["system"]) / times["wall"], **memory}


def verify(directory):
    require((directory / "campaign.exit").read_text().strip() == "0", "CPU comparison did not complete")
    lines = (directory / "campaign.log").read_text().splitlines()
    events = [line.split() for line in lines if line.startswith(("START ", "END "))]
    require(len(events) == 8, "expected four starts and four completed trials")
    previous = None
    for index, (event, name, engine, timestamp) in enumerate(events):
        require(event == ("START" if index % 2 == 0 else "END") and name == ORDER[index // 2]
                and engine == name.split("-", 1)[1], "trial order differs from the reviewed ABBA plan")
        current = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        require(previous is None or previous <= current, "trial timestamps run backwards")
        previous = current
    digests = dict((name, digest) for digest, name in
                   (line.split() for line in (directory / "binary.sha256").read_text().splitlines()))
    require(set(digests) == {"baseline", "candidate"} and len(set(digests.values())) == 2
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in digests.values()), "invalid binary identities")
    affinity = [int(value) for value in (directory / "affinity.txt").read_text().strip().split(",")]
    require(len(affinity) == len(set(affinity)) == 2, "expected two distinct CPU indices")
    require((directory / "rustc.txt").read_text().startswith("rustc 1.85.0 "), "unexpected Rust version")
    tests = (directory / "linux-tests.log").read_text()
    require("14 passed; 0 failed; 0 ignored;" in tests and all(f"test {name} ... ok" in tests for name in (
        "both_shards_share_the_same_weight_byte_limit", "paired_shards_report_failed_checksums_in_path_order"
    )), "Linux shard guard tests did not pass")
    model = json.loads((directory / "model-layout.json").read_bytes())
    require(model["checkpoint_sha256"] == CHECKPOINT_SHA256 and model["weight_shards"] == 6,
            "model layout differs from the timed checkpoint")
    runs = {name: trial(directory, name) for name in ORDER}

    def metric(name, key):
        return median(value[key] for trial_name, value in runs.items() if trial_name.endswith(name))

    baseline, candidate = (metric(name, "elapsed_s") for name in ("baseline", "candidate"))
    return {
        "source_archive_sha256": SOURCE_SHA256, "model": model, "binary_sha256": digests,
        "affinity": affinity, "linux_checkpoint_tests_passed": 14, "runs": runs,
        "median_elapsed_s": {"baseline": baseline, "candidate": candidate},
        "speedup": baseline / candidate,
        "elapsed_reduction_pct": (1 - candidate / baseline) * 100,
        "median_rss_kib": {name: metric(name, "rss_kib") for name in ("baseline", "candidate")},
        "paired_wall_ratios": [runs[right]["elapsed_s"] / runs[left]["elapsed_s"] for left, right in (
            ("01-baseline", "02-candidate"), ("04-baseline", "03-candidate")
        )],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2, allow_nan=False))
