"""Recompute the completed two-mode tcache-only comparison from original AWS receipts."""

from argparse import ArgumentParser
import json
from pathlib import Path
import re
import sys

MODES = ("cache-on", "cache-off")
FILES = ("exit", "idle-before.json", "idle-after.json", "comparison.json")
RSS_GATE = "rss_growth_second_half_at_most_zero"


def verify(root, combined):
    sys.path.insert(0, str(root))
    from compare_tcache import cpu_preflight
    from compare_tcache_only import compare_mode
    from verify_periodic_trim import digest, read, require

    require((root / "campaign.exit").read_text().strip() == "0", "paired comparison did not finish")
    expected = {"probe-default.json", "probe-disabled.json", "cpu-probe-proof.json"}
    expected.update(f"{mode}/{name}" for mode in MODES for name in FILES)
    checksums = {}
    for line in (root / "result.sha256").read_text().splitlines():
        sha, name = line.split(None, 1)
        require(name in expected and name not in checksums, f"unexpected paired artifact {name}")
        require(digest(root / name) == sha, f"paired artifact differs: {name}")
        checksums[name] = sha
    require(checksums.keys() == expected, "paired comparison checksums are incomplete")
    visits = [line.split() for line in (root / "campaign.log").read_text().splitlines()
              if re.match(r"^(START|END) cache-(on|off) ", line)]
    require(len(visits) == 4, "paired comparison chronology is incomplete")

    probe = cpu_preflight(root)
    require(read(root / "cpu-probe-proof.json") == {"cpu_probe": probe}, "CPU preflight differs")
    summaries = {}
    runs = {}
    for mode, begin, end in zip(MODES, visits[::2], visits[1::2], strict=True):
        arm = root / mode
        rc = int((arm / "exit").read_text())
        require(begin[:2] == ["START", mode] and end[:2] == ["END", mode] and begin[2] < end[2]
                and end[3] == f"exit={rc}", f"{mode} exit or timeline differs")
        summaries[mode] = compare_mode(root, combined, mode)
        require(read(arm / "comparison.json") == {"cpu_probe": probe, "modes": summaries},
                f"{mode} saved comparison differs from the original receipts")
        soak = read(arm / "gpu-evidence/soak.json")
        measurement = soak["smaps_rollup_second_half"]
        require(measurement["rss_growth_kib"] == measurement["drain"]["rss_kib"] - measurement["baseline"]["rss_kib"]
                and measurement["anonymous_growth_kib"] ==
                measurement["drain"]["anonymous_kib"] - measurement["baseline"]["anonymous_kib"],
                f"{mode} precise RSS does not follow the original samples")
        runs[mode] = {"start_utc": begin[2], "end_utc": end[2], "exit": rc, "workload_counts": soak["counts"],
                      "recorded_checks": soak["checks"], "p95_change_percent": soak["p95_change_percent"],
                      "gpu_artifact_sha256": {name: digest(arm / "gpu-evidence" / name)
                                            for name in ("report.json", "soak.json", "allocator-ready.txt",
                                                         "allocator-after-soak.txt", "api.json", "shutdown.json")}}
    require(visits[1][2] <= visits[2][2], "GPU test arms overlapped")
    for mode in MODES:
        require((RSS_GATE in summaries[mode]["configurations"]["tcache_only"]["failed_checks"]) ==
                (runs[mode]["recorded_checks"][RSS_GATE] is False), f"{mode} strict verdict differs")
    return {"scope": "Completed A100 6 pair, with the original image and all three allocator settings.",
            "cpu_probe": probe, "paired_result_sha256": digest(root / "result.sha256"),
            "campaign_log_sha256": digest(root / "campaign.log"), "artifact_sha256": checksums,
            "modes": summaries, "run_receipts": runs,
            "tcache_only_passed_both_modes": all(value["exit"] == 0 for value in runs.values())}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("combined", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.combined), indent=2, allow_nan=False))
