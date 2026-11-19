"""Recheck the A100 candidate with the same-image uninstrumented cache controls."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from periodic_trim import PERIOD_MS, read_telemetry


CONTROLS = {
    "cache-off": "w5-dist-off-retry.ORBlgo3W/gpu-evidence",
    "cache-on": "w5-dist-d3d8fe6e5.GcPrN2e8/cache-on/gpu-evidence",
}
IDENTITY = ("image_id", "source_revision", "source_archive_sha256", "riley_binary_sha256",
            "cpu_report_sha256", "checkpoint_sha256", "gpu_uuid", "soak_server_arguments")
RSS_GATE = "rss_growth_second_half_at_most_zero"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, issue):
    if not condition:
        raise ValueError(issue)


def read_soak(directory):
    report = read(directory / "report.json")
    for file, expected in report["artifacts_sha256"].items():
        require(file == Path(file).name and digest(directory / file) == expected, f"changed artifact: {file}")
    soak = read(directory / "soak.json")
    metrics = read(directory / "shutdown.json")
    api = read(directory / "api.json")
    require(report["shutdown_exit"] == 0 and metrics["allocation"]
            and all(metrics[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and all(value == 0 for value in metrics["allocation"].values()), "shutdown retained resources")
    require(len(api["results"]) == report["api_checks"] == 40 and report["tool_checks_enabled"]
            and all(row["passed"] is True for row in api["results"]), "incomplete GPU API coverage")
    rss = soak["smaps_rollup_second_half"]
    growth = rss["rss_growth_kib"]
    failed = sorted(key for key, value in soak["checks"].items() if not value)
    require(soak["seconds"] == 1800 and soak["rate"] == 8 and soak["shared_prefixes"] == 8
            and type(growth) is int and soak["checks"][RSS_GATE] == (growth <= 0)
            and report["status"] == soak["status"] == ("failed" if failed else "passed")
            and report["mixed_soak"]["failed_checks"] == failed and report["mixed_soak"]["rss"] == rss,
            "30-minute workload or original RSS gate differs")
    return report, soak


def compare(stage, mode):
    directory = stage / mode
    candidate = directory / "gpu-evidence"
    control = stage.parent / CONTROLS[mode]
    base, base_soak = read_soak(control)
    active, active_soak = read_soak(candidate)
    require(all(base[key] == active[key] for key in IDENTITY)
            and base["mixed_soak"]["mode"] == active["mixed_soak"]["mode"] == mode,
            "periodic trial changed image, checkpoint, server arguments or GPU")
    require(int((directory / "exit").read_text()) == int(active["status"] == "failed"),
            "GPU checker exit disagrees with the unchanged soak gate")
    for sample in (base_soak, active_soak):
        require(sample["rss_source"]["container_pid"] == 1, "RSS was not read from Riley PID 1")
    telemetry = read_telemetry(candidate / "periodic-trim.jsonl", pid=1, period_ms=PERIOD_MS,
                               duration_seconds=active_soak["seconds"])
    require(telemetry == active["periodic_trim"] and telemetry["returned_memory"] > 0,
            "the periodic library did not show actual reclamation")
    before, after = (read(directory / f"idle-{point}.json") for point in ("before", "after"))
    require(all(before[key] == after[key] for key in before if key != "updated_ns")
            and after["physical_index"] == 6 and after["phase"] == "idle" and not after["other_processes"]
            and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3,
            "the existing 20 GiB reservation was not restored")
    latency = {}
    for field in ("ttft_ms_p95", "e2e_ms_p95"):
        first, second = (data["latency_windows"]["late"][field] for data in (base_soak, active_soak))
        require(first > 0 and second > 0, f"missing p95: {field}")
        latency[field] = {"control": first, "periodic": second, "difference_percent": (second / first - 1) * 100}
    return {
        "candidate_status": active["status"],
        "candidate_failed_checks": active["mixed_soak"]["failed_checks"],
        "candidate_soak_p95_drift_percent": active_soak["p95_change_percent"],
        "same_outcome_counts": active_soak["counts"] == base_soak["counts"],
        "control_rss": base_soak["smaps_rollup_second_half"],
        "candidate_rss": active_soak["smaps_rollup_second_half"],
        "late_p95_ms": latency,
        "periodic_trim": telemetry,
        "same_20gib_reservation_restored": True,
        "control_report_sha256": digest(control / "report.json"),
        "candidate_report_sha256": digest(candidate / "report.json"),
        "candidate_soak_sha256": digest(candidate / "soak.json"),
        "trim_telemetry_sha256": digest(candidate / "periodic-trim.jsonl"),
    }


def verify(stage):
    return {
        "campaign": stage.name,
        "modes": {mode: compare(stage, mode) for mode in CONTROLS if (stage / mode / "exit").is_file()},
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage), indent=2, allow_nan=False))
