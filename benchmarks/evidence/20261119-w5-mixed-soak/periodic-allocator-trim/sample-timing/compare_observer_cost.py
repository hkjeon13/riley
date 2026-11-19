"""Compare matched cache-off soaks with and without the extra mapping reads."""

from argparse import ArgumentParser
import json
from pathlib import Path

from compare_sample_timing import read_run
from periodic_trim import read_telemetry
from verify_periodic_trim import IDENTITY, digest, read_soak, require


def step_counts(soak):
    metrics = [soak[point]["metrics"] for point in ("initial", "final")]
    shapes = [view["batch_shapes"] for view in metrics]
    require(all(not view["metrics_degraded"] for view in shapes), "degraded step timing metrics")
    totals = dict.fromkeys(("hit_count", "latency_sample_count", "gpu_execution_ns_total"), 0)
    for start, end in zip(*(view["buckets"] for view in shapes), strict=True):
        require(start["dense_rows"] == end["dense_rows"], "bucket rows changed")
        for key in totals:
            difference = end[key] - start[key]
            require(difference >= 0, f"decreasing step metric: {key}")
            totals[key] += difference
    require(totals["hit_count"] == totals["latency_sample_count"] > 0, "incomplete step timing")
    return {
        "step_count": totals["hit_count"],
        "execution_time_total_ms": totals["gpu_execution_ns_total"] / 1e6,
        "initial_allocation": metrics[0]["allocation"],
        "final_allocation": metrics[1]["allocation"],
    }


def compare(control, monitored):
    base, active = (read_soak(path) for path in (control, monitored))
    require(all(base[0][key] == active[0][key] for key in IDENTITY), "soak identities differ")
    require(all(report["mixed_soak"]["mode"] == "cache-off" for report, _ in (base, active))
            and base[1]["counts"] == active[1]["counts"], "soak modes or outcomes changed")
    for directory, (report, soak) in zip((control, monitored), (base, active), strict=True):
        telemetry = read_telemetry(directory / "periodic-trim.jsonl", pid=1, period_ms=31000,
                                   duration_seconds=soak["seconds"])
        require(telemetry == report["periodic_trim"], "different trim period or telemetry")
    require("W5-RSS-SAMPLE " not in (control / "soak.log").read_text(), "control had mapping reads")
    observed = read_run(monitored)
    read_cost = {}
    for phase, point in observed["sample_bounds"].items():
        span = point["map_read"]
        read_cost[phase] = span["milliseconds_since_trim_at_end"] - span["milliseconds_since_trim_at_start"]
    latency = {}
    for key in ("ttft_ms_p95", "e2e_ms_p95"):
        old, new = (soak["latency_windows"]["late"][key] for _, soak in (base, active))
        require(old > 0 and new > 0, f"invalid latency: {key}")
        latency[key] = {"control": old, "monitored": new, "change_percent": (new / old - 1) * 100}
    return {
        "scope": "One 31-second pair. The direct map-read wall intervals and the across-run metrics describe separate effects.",
        "same_30m_workload_outcomes": True,
        "report_sha256": {label: digest(path / "report.json")
                          for label, path in (("control", control), ("monitored", monitored))},
        "extra_map_read_wall_ms": read_cost,
        "late_latency_ms": latency,
        "gpu_steps": {label: step_counts(soak) for label, (_, soak) in (("control", base), ("monitored", active))},
        "scored_rollup_growth_kib": {label: soak["smaps_rollup_second_half"]["rss_growth_kib"]
                                    for label, (_, soak) in (("control", base), ("monitored", active))},
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("monitored", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.control, args.monitored), indent=2, allow_nan=False))
