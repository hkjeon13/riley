"""Compare saved W5 RSS samples against the service gauges captured in the same snapshots."""

from argparse import ArgumentParser
import json
from pathlib import Path
from statistics import median
import sys


MODES = ("cache-on", "cache-off")
INTERVALS = {
    "minutes_10_to_15": (600, 900),
    "minutes_20_to_25": (1200, 1500),
}
TRIALS = {
    "trim_31s": "w5-periodic-trim-20261119.mI9aaaEp",
    "trim_7s": "w5-trim-7sec-retry-20261119.fKlbclsw",
    "trim_7s_tcache_off": "w5-tcache-control-20261119.liqVpVJ7",
    "tcache_off": "w5-tcache-only-20261119.G3IuT0Wh",
    "http_framing": "w5-http-framing-20261119.IyUaTevw",
}
GAUGES = ("active_requests", "waiting_requests", "kv_allocated_blocks")


def sample_gauges(sample):
    return {key: sample["metrics"][key] for key in GAUGES}


def summarize(record, require):
    duration = record["seconds"]
    snapshots = record["rss_samples"]
    times = [sample["at_seconds"] for sample in snapshots]
    require(times == sorted(set(times)) and len(times) >= 30
            and all(b - a < 62 for a, b in zip(times, times[1:])), "incomplete or reordered periodic samples")
    baseline = next(sample for sample in snapshots if sample["at_seconds"] >= duration / 2)
    drain = record["final"]
    precise = record["smaps_rollup_second_half"]
    require(baseline["smaps_rollup"] == precise["baseline"] and drain["smaps_rollup"] == precise["drain"],
            "the original RSS decision did not use these samples")

    allocations = [sample["metrics"]["allocation"] for sample in (record["initial"], *snapshots, drain)]
    native_bytes = {key: sorted({sample[key] for sample in allocations}) for key in allocations[0]}
    bands = {}
    for label, (start, stop) in INTERVALS.items():
        rss = [round(sample["rss_mib"] * 1024) for sample in snapshots if start <= sample["at_seconds"] < stop]
        require(len(rss) >= 4, f"missing samples for {label}")
        bands[label] = {"n": len(rss), "min_kib": min(rss), "median_kib": median(rss), "max_kib": max(rss)}
    return {
        "periodic_samples": len(times),
        "native_values_across_samples": native_bytes,
        "gauges_at_baseline": sample_gauges(baseline),
        "gauges_at_drain": sample_gauges(drain),
        "baseline_seconds": baseline["at_seconds"],
        "drain_seconds": drain["at_seconds"],
        "baseline_to_drain_anonymous_growth_kib": precise["anonymous_growth_kib"],
        "baseline_to_drain_total_growth_kib": precise["rss_growth_kib"],
        "coarse_rss_kib": bands,
    }


def compare(root, include_framing_off):
    sys.path.insert(0, str(root / TRIALS["tcache_off"]))
    from verify_periodic_trim import CONTROLS, digest, read, read_soak, require

    framing_image = read(root / TRIALS["http_framing"] / "cpu-comparison.json")["image"]

    results = {}
    for mode in MODES:
        control = root / CONTROLS[mode]
        reference, baseline = read_soak(control)
        case = results[mode] = {}
        trials = {"unmodified": control}
        trials.update({label: root / path / mode / "gpu-evidence" for label, path in TRIALS.items()
                       if label != "http_framing" or mode == "cache-on" or include_framing_off})
        for label, path in trials.items():
            receipt, record = read_soak(path)
            same_identity = all(receipt[key] == reference[key] for key in
                                ("source_revision", "checkpoint_sha256", "gpu_uuid", "soak_server_arguments"))
            same_load = record["counts"] == baseline["counts"] and receipt["mixed_soak"]["mode"] == mode
            require(same_identity and same_load, f"different benchmark settings for {mode}/{label}")
            expected_image = reference["image_id"]
            expected_binary = reference["riley_binary_sha256"]
            if label == "http_framing":
                expected_image = framing_image["candidate_image_id"]
                expected_binary = framing_image["candidate_binary_sha256"]
            require(receipt["image_id"] == expected_image and receipt["riley_binary_sha256"] == expected_binary,
                    f"different image or executable for {mode}/{label}")
            case[label] = {
                "report_sha256": digest(path / "report.json"),
                "soak_sha256": digest(path / "soak.json"),
                **summarize(record, require),
            }
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--include-framing-off", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compare(args.root, args.include_framing_off), indent=2, allow_nan=False))
