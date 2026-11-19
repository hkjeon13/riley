"""Recheck identity, checksums and actual results of the staged A100 distribution campaign."""
from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
import json
import re
import sys

RILEY = "/opt/riley/bin/riley"
ROOTS = {
    "base": "w1-dist-d3d8fe6e5.YnZWdh5l",
    "accuracy": "w4-dist-d3d8fe6e5.FE5PU6Y4",
    "soak": "w5-dist-d3d8fe6e5.GcPrN2e8",
    "soak_retry": "w5-dist-off-retry.ORBlgo3W",
}
REVISION = "d3d8fe6e5b1c528bff51c0008cdc24544c99f862"
ARCHIVE = "cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5"
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
GPU = "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7"
CI_IMAGE = "sha256:8cb92e573ef1cf7e3acda80c0f9cff46e7885bcdee01b5240d7d2c09f501dca7"
CI_BINARY = "91a2de79577e54ab7fc15de78d8cacb6915f9aacfafae19b66215e015567a0ec"
HOLDOUT = "220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11"
GOLDENS = {"baseline": "b32bd641dab4e9516a8d3b45de2d77fbde91ad64acad893153f4b6c2090b72a6",
           "holdout": "12d508ed51f03509432e6302b8d10855f56125f4f6b7f31a559cff03eb10b611"}


def require(condition, issue):
    if not condition:
        raise ValueError(issue)


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def exit_code(path):
    if not path.is_file():
        return None
    value = path.read_text().strip()
    return int(value) if value else None


def reservation(directory):
    before, after = (read(directory / f"idle-{point}.json") for point in ("before", "after"))
    keys = ("pid", "start_ticks", "gpu", "physical_index", "phase", "allocation_bytes", "torch_reserved_bytes")
    require(all(before[key] == after[key] for key in keys)
            and after["physical_index"] == 6 and after["gpu"] == GPU and after["phase"] == "idle"
            and after["pid"] == 2431228
            and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3
            and not after["other_processes"], "GPU holder did not restore the reservation")
    return {key: after[key] for key in keys}


def gpu_evidence(directory, expected, rc):
    root = directory / "gpu-evidence"
    raw_report = root / "report.json"
    if not raw_report.is_file():
        require(rc != 0, "success signature has no GPU report")
        return {"exit": rc, "status": "no_report"}
    report = read(raw_report)
    require(report["schema_version"] == "riley.distribution-gpu-check.v1"
            and all(report[key] == value for key, value in expected.items())
            and report["shutdown_exit"] == 0
            and report["status"] in ("passed", "failed") and rc == int(report["status"] == "failed"),
            "incorrect GPU evidence identity, status or shutdown")
    checksums = report["artifacts_sha256"]
    require({"api.json", "ready.json", "shutdown.json", "server.log"} <= checksums.keys(), "incomplete GPU artifacts")
    for name, sha in checksums.items():
        require(Path(name).name == name and digest(root / name) == sha, f"invalid GPU artifact {name}")
    ready, api, shutdown = (read(root / file) for file in ("ready.json", "api.json", "shutdown.json"))
    require(ready["ready"] is True and ready["accepting"] is True, "runtime did not report readiness")
    require(api["model"] == "m" and report["api_checks"] == len(api["results"])
            and len(api["results"]) == 40 and len({case["name"] for case in api["results"]}) == 40
            and all(case["passed"] is True for case in api["results"])
            and report["tool_checks_enabled"] is True, "API checks do not prove the requested suite")
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and all(value == 0 for value in shutdown["allocation"].values()),
            "shutdown resource evidence is not zero")
    return {**report, "exit": rc, "gpu_receipt_sha256": digest(raw_report), "holder": reservation(directory),
            "passed_api_cases": [case["name"] for case in api["results"]],
            "shutdown_metrics": {key: shutdown[key] for key in
                                 ("active_requests", "waiting_requests", "kv_allocated_blocks", "allocation")}}


def validate_accuracy(directory, report):
    if report["status"] == "no_report":
        return
    suites = report["accuracy_suites"]
    require(set(suites) == GOLDENS.keys() and report["status"] == "passed", "missing accuracy suites")
    for label, cases in (("baseline", 18), ("holdout", 11)):
        root = directory / "gpu-evidence"
        compare = read(root / f"accuracy-{label}-comparison.json")
        modes = {"c1": cases, "diagnostic": cases, "c8": 3 * cases, "c32": 3 * cases}
        suite = suites[label]
        require(compare["status"] == "passed" and compare["golden_sha256"] == suite["golden_sha256"] == GOLDENS[label]
                and compare["workload_sha256"] == suite["workload_sha256"] == (None if label == "baseline" else HOLDOUT)
                and compare["candidate_sha256"] == digest(root / f"accuracy-{label}.json")
                and compare["modes"].keys() == modes.keys() and suite["verified_requests"] == 8 * cases,
                f"incomplete {label} comparison")
        require(all(compare["modes"][mode] == {"requests": count, "matched": count, "mismatches": []}
                    for mode, count in modes.items()), f"{label} comparison has differences")


def validate_soak(directory, report, mode):
    if report["status"] == "no_report":
        return
    raw = read(directory / "gpu-evidence/soak.json")
    measured = report["mixed_soak"]
    args = report["soak_server_arguments"]
    if mode == "cache-on":
        require(args[args.index("--prefix-cache-blocks") + 1] == "128", "cache-on did not request 128 blocks")
    else:
        require("--prefix-cache-blocks" not in args, "cache-off passed a prefix cache argument")
    delta = raw["smaps_rollup_second_half"]["rss_growth_kib"]
    rss_gate = "rss_growth_second_half_at_most_zero"
    checks = raw["checks"]
    require(raw["seconds"] == 1800 and raw["rate"] == 8 and raw["shared_prefixes"] == 8
            and checks[rss_gate] == (type(delta) is int and delta <= 0)
            and measured["mode"] == mode and measured["status"] == raw["status"] == report["status"]
            and measured["failed_checks"] == sorted(key for key, value in checks.items() if not value)
            and measured["rss"] == raw["smaps_rollup_second_half"] and measured["counts"] == raw["counts"],
            f"{mode} evidence disagrees on workload, strict RSS or status")
    baseline = next(sample for sample in raw["rss_samples"] if sample["at_seconds"] >= 900)
    first = baseline["smaps_rollup"]
    final = raw["final"]["smaps_rollup"]
    require(first == measured["rss"]["baseline"] and final == measured["rss"]["drain"]
            and raw["final"]["at_seconds"] >= 1800
            and final["rss_kib"] - first["rss_kib"] == measured["rss"]["rss_growth_kib"]
            and final["anonymous_kib"] - first["anonymous_kib"] == measured["rss"]["anonymous_growth_kib"],
            f"{mode} precise RSS snapshots differ from the comparison")
    required_checks = {"no_unexpected_http_or_stream_error", "completed_schema_and_tool_covered",
                       "disconnects_observed", "idle_after_drain", "native_allocation_plateau", "workload_coverage",
                       "p95_drift_at_most_5_percent", rss_gate}
    require(set(checks) == required_checks and all(type(value) is bool for value in checks.values())
            and report["status"] == ("failed" if measured["failed_checks"] else "passed"),
            f"{mode} strict workload contract changed")
    require(set(report["artifacts_sha256"]) == {"api.json", "api.log", "checkpoint-verify.log", "ready.json",
                                            "server.log", "shutdown.json", "soak.json", "soak.log"},
            f"{mode} artifact inventory differs from the original image execution")
    report["verified_soak"] = {
        "duration_seconds": raw["seconds"], "arrival_rate": raw["rate"],
        "shared_prefixes": raw["shared_prefixes"], "checks": checks,
        "problem_count": raw["problem_count"],
        "p95_change_percent": raw["p95_change_percent"],
        "counter_changes": raw["counter_changes"],
        "rss_baseline_time_seconds": baseline["at_seconds"],
        "rss_drain_time_seconds": raw["final"]["at_seconds"],
        "rss_proc_status_growth_mib": raw["rss_growth_second_half_mib"],
        "rss_source": raw["rss_source"],
        "rss_sample_errors": [sample["smaps_rollup_error"] for sample in (baseline, raw["final"])
                              if "smaps_rollup_error" in sample],
    }


def collect(evidence):
    roots = {name: evidence / directory for name, directory in ROOTS.items()}
    base, accuracy, soak, retry = (roots[name] for name in ROOTS)
    result = {"campaigns": {name: {"directory": root.name, "exit": exit_code(root / "campaign.exit")}
                            for name, root in roots.items()}}
    cpu_path = base / "cpu-evidence/report.json"
    if not cpu_path.is_file():
        require(result["campaigns"]["base"]["exit"] != 0, "base campaign succeeded without CPU proof")
        return result
    cpu = read(cpu_path)
    binary = cpu["binaries"][RILEY]
    image = (base / "image.id").read_text().strip()
    require(cpu["schema_version"] == "riley.distribution-image-check.v1" and cpu["status"] == "passed"
            and cpu["source_revision"] == REVISION and cpu["source_archive_sha256"] == digest(base / "source.tar") == ARCHIVE
            and cpu["image_id"] == image and re.fullmatch(r"sha256:[0-9a-f]{64}", image)
            and re.fullmatch(r"[0-9a-f]{64}", binary), "CPU evidence does not identify the approved source image")
    runtime = {name: sha for sha, name in (row.split(maxsplit=1) for row in
                (cpu_path.parent / "runtime.sha256").read_text().splitlines())}
    require(runtime[RILEY] == binary, "CPU evidence binary checksum differs")
    expected = {"image_id": image, "source_revision": REVISION, "source_archive_sha256": ARCHIVE,
                "riley_binary_sha256": binary, "cpu_report_sha256": digest(cpu_path),
                "checkpoint_sha256": CHECKPOINT, "gpu_uuid": GPU}
    result["cpu"] = {**cpu, "cpu_report_sha256": expected["cpu_report_sha256"],
                     "runtime_tree_sha256": dict(sorted(runtime.items())),
                     "build_log_sha256": digest(base / "build.log"),
                     "same_image_id_as_hosted_ci": image == CI_IMAGE,
                     "same_riley_binary_as_hosted_ci": binary == CI_BINARY}
    for name, root, code in (
        ("base", base, result["campaigns"]["base"]["exit"]),
        ("accuracy", accuracy, result["campaigns"]["accuracy"]["exit"]),
        ("cache-off", soak / "cache-off", exit_code(soak / "cache-off/exit")),
        ("cache-on", soak / "cache-on", exit_code(soak / "cache-on/exit")),
        ("cache-off-retry", retry, result["campaigns"]["soak_retry"]["exit"]),
    ):
        if code is None:
            continue
        report = gpu_evidence(root, expected, code)
        if name == "accuracy":
            validate_accuracy(root, report)
        elif name.startswith("cache-"):
            validate_soak(root, report, "cache-off" if name == "cache-off-retry" else name)
        result[name] = report
    if all(result.get(mode, {}).get("mixed_soak") for mode in ("cache-on", "cache-off-retry")):
        on, off = (result[mode] for mode in ("cache-on", "cache-off-retry"))
        on_args = on["soak_server_arguments"]
        index = on_args.index("--prefix-cache-blocks")
        require(on_args[:index] + on_args[index + 2:] == off["soak_server_arguments"]
                and on["holder"] == off["holder"]
                and on["mixed_soak"]["counts"] == off["mixed_soak"]["counts"],
                "soak pair changed more than the cache option or holder")
        result["operation_pair"] = {
            "same_image_binary_checkpoint_and_holder": True,
            "only_server_argument_difference": ["--prefix-cache-blocks", "128"],
            "same_outcome_counts": on["mixed_soak"]["counts"] == off["mixed_soak"]["counts"],
            "rss_gate_passed_in_both": all(run["verified_soak"]["checks"]["rss_growth_second_half_at_most_zero"]
                                          for run in (on, off)),
        }
    return result


def verdict(evidence, evidence_root):
    require(all(evidence["campaigns"][step]["exit"] == 0 for step in ("base", "accuracy")),
            "required candidate campaign did not complete")
    require(evidence["base"]["status"] == evidence["accuracy"]["status"] == "passed"
            and evidence["cache-on"]["status"] in ("passed", "failed")
            and evidence["cache-off-retry"]["status"] in ("passed", "failed"),
            "missing completed package, accuracy or soak evidence")
    initial_cache_off = evidence["cache-off"]
    require(initial_cache_off == {"exit": 1, "status": "no_report"},
            "initial cache-off startup failure was not retained separately")
    initial_log = evidence_root / ROOTS["soak"] / "cache-off/gpu-evidence/server.log"
    require("prefix_cache.max_blocks: must be positive" in initial_log.read_text(),
            "original cache-off startup failure differs")
    source = evidence["cpu"]
    accuracy = evidence["accuracy"]
    operation = {}
    for mode, name in (("cache-on", "cache-on"), ("cache-off", "cache-off-retry")):
        report = evidence[name]
        comparison = report["mixed_soak"]
        verification = report["verified_soak"]
        operation[mode] = {
            "status": report["status"], "exit": report["exit"], "api_checks": report["api_checks"],
            "failed_checks": comparison["failed_checks"], "request_counts": comparison["counts"],
            "rss_growth_kib": comparison["rss"]["rss_growth_kib"],
            "anonymous_growth_kib": comparison["rss"]["anonymous_growth_kib"],
            "rss_baseline_seconds": verification["rss_baseline_time_seconds"],
            "rss_drain_seconds": verification["rss_drain_time_seconds"],
            "p95_change_percent": verification["p95_change_percent"],
            "gpu_receipt_sha256": report["gpu_receipt_sha256"],
            "soak_sha256": report["artifacts_sha256"]["soak.json"],
        }
    decision = "blocked" if any(mode["status"] == "failed" for mode in operation.values()) else "passed"
    return {
        "scope": "A100 package, API, long-context and strict soak gates for the original distribution image.",
        "status": decision,
        "image_id": source["image_id"], "source_revision": source["source_revision"],
        "source_archive_sha256": source["source_archive_sha256"],
        "cpu_report_sha256": source["cpu_report_sha256"], "riley_binary_sha256": source["binaries"][RILEY],
        "checkpoint_sha256": CHECKPOINT, "gpu_uuid": GPU, "package_api_cases": evidence["base"]["api_checks"],
        "package_api_receipt_sha256": evidence["base"]["gpu_receipt_sha256"],
        "accuracy_verified_requests": sum(item["verified_requests"] for item in accuracy["accuracy_suites"].values()),
        "accuracy_receipt_sha256": accuracy["gpu_receipt_sha256"],
        "original_holder_pid": evidence["cache-on"]["holder"]["pid"],
        "initial_cache_off_startup": {**initial_cache_off, "server_log_sha256": digest(initial_log)},
        "operation_pair": evidence["operation_pair"],
        "mixed_soak": operation,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    args = parser.parse_args()
    try:
        summary = verdict(collect(args.evidence_root), args.evidence_root)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, StopIteration) as error:
        print(f"invalid or incomplete distribution evidence: {error}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(summary, indent=2, allow_nan=False))
    sys.exit(int(summary["status"] != "passed"))
