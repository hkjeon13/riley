"""Recompute a completed framing soak mode against the unmodified image."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

PAIR = ("cache-on", "cache-off")



def verify_samples(soak, require):
    baseline = next(sample for sample in soak["rss_samples"] if sample["at_seconds"] >= soak["seconds"] / 2)
    drain = soak["final"]
    observation = soak["smaps_rollup_second_half"]
    require(soak["rss_source"]["container_pid"] == 1 and baseline["at_seconds"] < soak["seconds"]
            and drain["at_seconds"] >= soak["seconds"], "RSS samples have the wrong process or timing")
    require(baseline["smaps_rollup"] == observation["baseline"]
            and drain["smaps_rollup"] == observation["drain"], "RSS samples differ from the reported comparison")
    for key in ("rss", "anonymous"):
        growth = drain["smaps_rollup"][f"{key}_kib"] - baseline["smaps_rollup"][f"{key}_kib"]
        require(growth == observation[f"{key}_growth_kib"], f"wrong {key} delta")
    return {"baseline_seconds": baseline["at_seconds"], "drain_seconds": drain["at_seconds"]}


def verify(stage, original, modes):
    sys.path.insert(0, str(stage))
    sys.path.insert(0, str(stage.parent / "w5-tcache-only-20261119.G3IuT0Wh"))
    from compare_candidate import compare_image, compare_mode
    from verify_periodic_trim import CONTROLS, digest, read, read_soak, require

    package = compare_image(stage, original, digest, read, require)
    require(read(stage / "cpu-comparison.json")["image"] == package, "CPU package verification changed")
    chronology = [line.split() for line in (stage / "campaign.log").read_text().splitlines()
                  if line.startswith(("START cache-", "END cache-"))]
    if modes == PAIR:
        require((stage / "campaign.exit").read_text().strip() == "0" and len(chronology) == 4,
                "the full framing workload did not finish")
        for line in (stage / "result.sha256").read_text().splitlines():
            expected, name = line.split(None, 1)
            path = Path(name)
            require(not path.is_absolute() and ".." not in path.parts and digest(stage / path) == expected,
                    f"changed campaign evidence: {name}")
    results = {}
    receipts = {}
    for index, mode in enumerate(modes):
        arm = stage / mode
        output = arm / "gpu-evidence"
        current = compare_mode(stage, mode, package, CONTROLS, read_soak, digest, read, require)
        results[mode] = current
        require(read(arm / "comparison.json")["modes"] == results, f"saved comparison differs for {mode}")
        before, after = chronology[index * 2:index * 2 + 2]
        exit_code = int((arm / "exit").read_text())
        require(before[:2] == ["START", mode] and after[:2] == ["END", mode]
                and before[2] < after[2] and after[3] == f"exit={exit_code}", f"wrong timing for {mode}")
        soak = read(output / "soak.json")
        candidate_samples = verify_samples(soak, require)
        control_samples = verify_samples(read(stage.parent / CONTROLS[mode] / "soak.json"), require)
        receipts[mode] = {"status": current["candidate_status"], "exit": exit_code,
                          "start_utc": before[2], "end_utc": after[2],
                          "p95_change_percent": soak["p95_change_percent"], "strict_checks": soak["checks"],
                          "candidate_rss_samples": candidate_samples, "control_rss_samples": control_samples,
                          "sha256": {name: digest(output / name) for name in ("api.json", "shutdown.json",
                                                                               "report.json", "soak.json")}}
    return {"scope": "A100 6 framing diagnostic, paired against the unmodified image for each mode.",
            "completed_modes": list(modes), "image": package, "comparisons": results, "receipts": receipts}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("--first", action="store_true")
    args = parser.parse_args()
    modes = PAIR[:1] if args.first else PAIR
    print(json.dumps(verify(args.stage, args.original, modes), indent=2, allow_nan=False))
