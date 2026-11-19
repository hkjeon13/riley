"""Verify the process-local allocator and compare the two strict soaks to stock glibc."""

from argparse import ArgumentParser
import json
from pathlib import Path

from run_gpu_check import IMAGE, LIBRARY, LIBRARY_SHA, MALLOC_CONF
from verify_a100_candidate import (ARCHIVE, CHECKPOINT, GPU, REVISION, ROOTS, RILEY, digest,
                                  gpu_evidence, read, require, validate_soak)


MODES = ("cache-on", "cache-off")
RECEIPT_KEYS = ("gpu", "physical_index", "phase", "pid", "start_ticks", "allocation_bytes", "torch_reserved_bytes")


def holder(path, physical, pid):
    record = read(path)
    require(record["physical_index"] == physical and record["phase"] == "idle" and record["pid"] == pid
            and record["allocation_bytes"] == record["torch_reserved_bytes"] == 20 * 1024**3
            and not record["other_processes"], f"GPU {physical} is not in its original 20 GiB idle reservation")
    return {key: record[key] for key in RECEIPT_KEYS}


def before_arm(directory):
    return holder(directory / "idle-before.json", 6, 2431228), holder(directory / "gpu7-before.json", 7, 2431237)


def identity(evidence):
    cpu = evidence / ROOTS["base"] / "cpu-evidence/report.json"
    data = read(cpu)
    require(data["status"] == "passed" and data["image_id"] == IMAGE and data["source_revision"] == REVISION
            and data["source_archive_sha256"] == ARCHIVE, "reviewed source image changed")
    return {"image_id": IMAGE, "source_revision": REVISION, "source_archive_sha256": ARCHIVE,
            "riley_binary_sha256": data["binaries"][RILEY], "cpu_report_sha256": digest(cpu),
            "checkpoint_sha256": CHECKPOINT, "gpu_uuid": GPU}


def proof(directory, candidate):
    before, after = (read(directory / f"allocator-{point}.json") for point in ("before", "after"))
    source = candidate["verified_soak"]["rss_source"]
    expected = {"image_id": IMAGE, "container_id": source["container_id"], "host_pid": source["host_pid"],
                "library": LIBRARY, "library_sha256": LIBRARY_SHA, "malloc_conf": MALLOC_CONF,
                "library_mapping_present": True, "exec_helper_uses_default_environment": True}
    require(before == after == expected, "running allocator proof differs from the RSS process")
    return expected


def stock_report(original, expected, mode):
    exit_receipt = "exit" if mode == "cache-on" else "campaign.exit"
    report = gpu_evidence(original, expected, int((original / exit_receipt).read_text()))
    require(report["status"] in ("passed", "failed"), f"{mode} original soak is incomplete")
    validate_soak(original, report, mode)
    return report


def control_dirs(evidence):
    return {"cache-on": evidence / ROOTS["soak"] / "cache-on",
            "cache-off": evidence / ROOTS["soak_retry"]}


def result(directory, original, expected, mode):
    original_report = stock_report(original, expected, mode)
    candidate = gpu_evidence(directory, expected, int((directory / "exit").read_text()))
    require(all(run["status"] in ("passed", "failed") for run in (original_report, candidate)),
            f"{mode} does not have two completed soaks")
    validate_soak(directory, candidate, mode)
    candidate_holder, other = before_arm(directory)
    require(candidate_holder == holder(directory / "idle-after.json", 6, 2431228)
            and other == holder(directory / "gpu7-after.json", 7, 2431237)
            and candidate["holder"] == original_report["holder"],
            f"{mode} did not restore the same GPU reservation")
    require(original_report["soak_server_arguments"] == candidate["soak_server_arguments"]
            and original_report["passed_api_cases"] == candidate["passed_api_cases"],
            f"{mode} server options or API cases changed")
    previous, new = (run["mixed_soak"] for run in (original_report, candidate))
    comparable = previous["counts"] == new["counts"]
    control_late, candidate_late = (read(path / "gpu-evidence/soak.json")["latency_windows"]["late"]
                                    for path in (original, directory))
    fields = ("ttft_ms_p95", "e2e_ms_p95")
    return {
        "mode": mode, "status": candidate["status"], "identical_request_outcome_counts": comparable,
        "counts": new["counts"], "failed_checks": new["failed_checks"], "api_checks": candidate["api_checks"],
        "original_rss": previous["rss"], "candidate_rss": new["rss"],
        "original_late_p95_ms": {key: control_late[key] for key in fields},
        "candidate_late_p95_ms": {key: candidate_late[key] for key in fields},
        "late_p95_change_percent": {key: (candidate_late[key] / control_late[key] - 1) * 100 for key in fields},
        "p95_drift_percent": candidate["verified_soak"]["p95_change_percent"],
        "allocator_in_rss_process": proof(directory, candidate), "same_reservations": [candidate_holder, other],
        "shutdown_metrics": candidate["shutdown_metrics"], "candidate_report_sha256": candidate["gpu_receipt_sha256"],
        "original_report_sha256": original_report["gpu_receipt_sha256"], "candidate_artifacts_sha256": candidate["artifacts_sha256"],
    }


def compare(stage, evidence, modes):
    expected = identity(evidence)
    controls = control_dirs(evidence)
    results = {mode: result(stage / mode, controls[mode], expected, mode) for mode in modes}
    return {"scope": "Isolated A100 6 allocator diagnostic on the reviewed distribution image.",
            "identity": expected, "cpu_preflight_sha256": digest(stage / "cpu-preflight.json"),
            "results": results,
            "both_modes_pass_strict_soak": tuple(modes) == MODES and all(run["status"] == "passed"
                and run["identical_request_outcome_counts"] for run in results.values())}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("evidence", type=Path)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--check-admission", choices=MODES)
    options.add_argument("--mode", choices=MODES)
    options.add_argument("--check-controls", action="store_true")
    args = parser.parse_args()
    if args.check_admission:
        before_arm(args.stage / args.check_admission)
    elif args.check_controls:
        expected = identity(args.evidence)
        for mode, control in control_dirs(args.evidence).items():
            stock = stock_report(control, expected, mode)
            print(mode, stock["api_checks"], stock["mixed_soak"]["rss"]["rss_growth_kib"])
    else:
        print(json.dumps(compare(args.stage, args.evidence, (args.mode,) if args.mode else MODES),
                         indent=2, allow_nan=False))
