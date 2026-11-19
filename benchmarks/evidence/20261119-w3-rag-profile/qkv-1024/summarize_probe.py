"""Validate a Qwen2.5-14B exact-M alignment probe and report repeatable timings."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import median


ALGORITHM = r"\{\d+, \d+, \d+, \d+, \d+\}"
HEADER = re.compile(rf"qkv1024 k=5120 n=7168 rows=1024 cublaslt=(\d+) anchor=({ALGORITHM}) table=({ALGORITHM})")
CHECK = re.compile(r"check seed=([01]) table_valid=([01]) table_equal=([01]) candidates=(\d+) enumerated=(\d+)")
TIMING = re.compile(rf"time algo=({ALGORITHM}) round=([123]) baseline_us=(\d+\.\d+) candidate_us=(\d+\.\d+)")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def summarize(path):
    raw = path.read_bytes()
    header, *lines = raw.decode().splitlines()
    match = HEADER.fullmatch(header)
    require(match is not None, "probe target is missing")
    version, anchor, table = match.groups()
    require((anchor, table) == ("{21, 18, 18, 1, 0}", "{21, 20, 11, 3, 1}"), "wrong anchor or table entry")
    checks = []
    for line in lines[:2]:
        result = CHECK.fullmatch(line)
        require(result is not None, f"invalid seed record: {line}")
        seed, supported, same, matching, enumerated = map(int, result.groups())
        require((not same or supported) and 0 < matching <= enumerated, f"incomplete byte check: {line}")
        checks.append({"seed": seed, "table_valid": bool(supported), "table_equal": bool(same),
                       "matching_configurations": matching, "enumerated_configurations": enumerated})
    require(len(checks) == 2 and [row["seed"] for row in checks] == [0, 1]
            and checks[0]["table_equal"] == checks[1]["table_equal"]
            and checks[0]["enumerated_configurations"] == checks[1]["enumerated_configurations"]
            and checks[0]["matching_configurations"] >= checks[1]["matching_configurations"],
            "the two synthetic byte checks disagree or are missing")
    samples = defaultdict(dict)
    for line in lines[2:]:
        trial = TIMING.fullmatch(line)
        require(trial is not None, f"unexpected output: {line}")
        algorithm, round_no, control, candidate = trial.groups()
        control, candidate = float(control), float(candidate)
        require(round_no not in samples[algorithm] and all(math.isfinite(x) and x > 0 for x in (control, candidate)),
                "duplicate or invalid timing sample")
        samples[algorithm][round_no] = (control, candidate)
    require(len(samples) == checks[1]["matching_configurations"], "missing algorithms after the byte check")
    timings = []
    for algorithm, measured in samples.items():
        require(set(measured) == {"1", "2", "3"}, f"missing timing rounds for {algorithm}")
        ratios = [candidate / control for control, candidate in measured.values()]
        timings.append({"algorithm": algorithm, "baseline_us_p50": median(control for control, _ in measured.values()),
                        "candidate_us_p50": median(candidate for _, candidate in measured.values()),
                        "paired_time_ratio_p50": median(ratios), "paired_time_ratio_max": max(ratios)})
    timings.sort(key=lambda row: (row["paired_time_ratio_max"], row["algorithm"]))
    return {"cuda_library": int(version), "m": 1024, "k": 5120, "n": 7168, "anchor": anchor, "current_table_entry": table,
            "timed_baseline": table if checks[0]["table_equal"] else anchor, "synthetic_checks": checks,
            "ranked_timings": timings, "log_sha256": sha256(raw).hexdigest(),
            "scope": "Two synthetic BF16 inputs; interleaved paired exact-M timings. Server validation remains separate."}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log), indent=2, allow_nan=False))
