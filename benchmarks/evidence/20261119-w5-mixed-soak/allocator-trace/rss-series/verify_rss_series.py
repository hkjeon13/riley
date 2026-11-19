"""Verify the RSS observer series against its unchanged W5 score and control."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from observe_rss_series import AFTER_DRAIN, DURING_LOAD


def check_points(rows, targets, require):
    require(tuple(row["target_seconds"] for row in rows) == targets, "missing or reordered RSS observations")
    for row in rows:
        require(row["target_seconds"] <= row["read_start_seconds"] <= row["read_end_seconds"]
                and row["read_start_seconds"] < row["target_seconds"] + 5
                and 0 < row["anonymous_kib"] <= row["rss_kib"], "invalid RSS observation or clock")
    return rows


def verify(stage):
    evidence = stage.parent
    sys.path.insert(0, str(evidence / "w5-tcache-only-20261119.G3IuT0Wh"))
    from verify_periodic_trim import CONTROLS, IDENTITY, digest, read, read_soak, require

    mode = "cache-off"
    folder = stage / "gpu-evidence"
    control_report, control = read_soak(evidence / CONTROLS[mode])
    report, record = read_soak(folder)
    require(all(control_report[key] == report[key] for key in IDENTITY)
            and report["mixed_soak"]["mode"] == mode and control["counts"] == record["counts"],
            "the diagnostic did not run the original packaged cache-off load")

    series = read(folder / "rss-series.json")
    precise = record["smaps_rollup_second_half"]
    require(series["schema_version"] == "riley.w5-rss-series.v1"
            and series["host_pid"] == record["rss_source"]["host_pid"]
            and record["rss_source"]["container_pid"] == 1
            and series["original_drain"] == precise["drain"]
            and series["original_checks"] == record["checks"]
            and series["idle_after_last_sample"] is True
            and record["checks"]["idle_after_drain"] and record["problem_count"] == 0,
            "series identity or original workload score changed")
    during = check_points(series["during_load"], DURING_LOAD, require)
    for row in during:
        require(row["target_seconds"] <= row["metrics_at_seconds"] <= row["read_start_seconds"],
                "RSS read preceded the corresponding service metrics")
    after = check_points(series["after_drain"], AFTER_DRAIN, require)
    reservations = {}
    for index, pid in ((6, 2431228), (7, 2431237)):
        before, restored = (read(stage / f"idle-{label}-{index}.json") for label in ("before", "after"))
        require(before.keys() == restored.keys()
                and all(before[key] == restored[key] for key in before if key != "updated_ns")
                and restored["physical_index"] == index and restored["pid"] == pid
                and restored["phase"] == "idle" and not restored["other_processes"]
                and restored["allocation_bytes"] == restored["torch_reserved_bytes"] == 20 * 1024**3,
                f"original GPU {index} reservation differs")
        reservations[str(index)] = {"same_holder_pid": pid, "reservation_bytes": restored["allocation_bytes"]}
    require(int((stage / "checker.exit").read_text()) == int(report["status"] == "failed"),
            "checker exit differs from the unchanged strict score")

    return {
        "comparison": {"cache_mode": mode, "same_image_id": report["image_id"],
                       "same_checkpoint_sha256": report["checkpoint_sha256"],
                       "same_binary_sha256": report["riley_binary_sha256"],
                       "same_outcome_counts": True},
        "control_rss": control["smaps_rollup_second_half"],
        "original_rss": precise,
        "original_status": report["status"],
        "strict_checks": record["checks"],
        "during_load": during,
        "after_drain": after,
        "api_checks": report["api_checks"],
        "reservations": reservations,
        "sha256": {name: digest(folder / name) for name in
                   ("report.json", "soak.json", "rss-series.json", "api.json", "shutdown.json")},
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage), indent=2, allow_nan=False))
