"""Recheck the packed image, both soaks and exact long-output receipts."""

import gzip
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

from check_distribution_gpu import batch_shape_evidence, prefix_cache_evidence
from current_binding import check_reservations
from verify_a100_candidate import digest, read, require


RILEY = "/opt/riley/bin/riley"
MODEL = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
HOLDOUT = "220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11"
MODES = {"cache-off": "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7",
         "cache-on": "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"}


def arm(root, label, gpu, cpu, cpu_hash):
    path = root / label
    report, api, shutdown = (read(path / name) for name in ("report.json", "api.json", "shutdown.json"))
    expected = {"image_id": cpu["image_id"], "source_revision": cpu["source_revision"],
                "source_archive_sha256": cpu["source_archive_sha256"], "riley_binary_sha256": cpu["binaries"][RILEY],
                "cpu_report_sha256": cpu_hash, "checkpoint_sha256": MODEL, "gpu_uuid": gpu}
    require(all(report[key] == value for key, value in expected.items()), f"{label} package or GPU differs")
    require(report["shutdown_exit"] == 0 and report["tool_checks_enabled"] is True and api["model"] == "m"
            and report["api_checks"] == len(api["results"]) == len({item["name"] for item in api["results"]}) == 40
            and all(item["passed"] is True for item in api["results"]), f"{label} API or shutdown failed")
    report_hashes = report["artifacts_sha256"]
    require({"api.json", "shutdown.json", "ready.json", "server.log"} <= report_hashes.keys()
            and all(Path(name).name == name and digest(path / name) == value for name, value in report_hashes.items()),
            f"{label} raw artifacts differ")
    require(shutdown["allocation"] and all(value == 0 for value in shutdown["allocation"].values())
            and all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks")),
            f"{label} resources survived shutdown")
    exit_code = int((root / f"{label}.exit").read_text())
    require(report["status"] in ("passed", "failed") and exit_code == int(report["status"] == "failed"),
            f"{label} exit code differs from verdict")
    return report, shutdown, path


def soak(root, label, gpu, cpu, cpu_hash):
    report, shutdown, path = arm(root, label, gpu, cpu, cpu_hash)
    raw = read(path / "soak.json")
    capacity = 128 if label == "cache-on" else 0
    require((raw["mode"], raw["seconds"], raw["rate"], raw["shared_prefixes"]) == ("mixed_soak", 1800, 8, 8)
            and raw["rss_source"]["container_pid"] == 1, "wrong soak workload")
    shape = batch_shape_evidence(raw)
    cache = prefix_cache_evidence(raw, shutdown, capacity)
    require(shape == report["mixed_soak"]["batch_shape_metrics"] and cache == report["prefix_cache_metrics"],
            "soak metric summaries differ from the raw samples")
    rss = raw["smaps_rollup_second_half"]
    growth = rss["rss_growth_kib"]
    require(growth == (rss["drain"]["rss_kib"] - rss["baseline"]["rss_kib"]
                       if rss["drain"] is not None and rss["baseline"] is not None else None)
            and raw["checks"]["rss_growth_second_half_at_most_zero"] == (growth is not None and growth <= 0),
            "strict RSS verdict differs from the samples")
    changes = raw["p95_change_percent"]
    require(set(changes) == {"ttft_ms", "e2e_ms"} and raw["checks"]["p95_drift_at_most_5_percent"] == all(
        value is not None and isfinite(value) and value <= 5 for value in changes.values()), "wrong latency gate")
    checks = {**raw["checks"], "current_batch_shape_metrics": shape["status"] == "passed",
              "current_prefix_cache_metrics": cache["status"] == "passed"}
    failures = sorted(name for name, passed in checks.items() if not passed)
    require(report["mixed_soak"]["failed_checks"] == failures
            and report["status"] == report["mixed_soak"]["status"] == ("failed" if failures else "passed"),
            "soak report changed its raw gate verdict")
    return {"status": report["status"], "failed_checks": failures, "rss": rss, "p95_change_percent": changes,
            "api_checks": report["api_checks"], "problem_count": raw["problem_count"],
            "completed": sum(row.get("completed", 0) for row in raw["counts"].values()),
            "client_disconnects": raw["counts"]["cancel"].get("disconnected", 0),
            "counter_changes": raw["counter_changes"], "prefix_cache_metrics": cache, "batch_shape_metrics": shape,
            "soak_sha256": digest(path / "soak.json"), "report_sha256": digest(path / "report.json"),
            "shutdown_sha256": digest(path / "shutdown.json")}


def verify_accuracy(root, cpu, cpu_hash):
    accuracy, _, _ = arm(root, "accuracy", MODES["cache-off"], cpu, cpu_hash)
    suites = accuracy["accuracy_suites"]
    structured = suites["structured_holdout"]
    counts = {"c1": 16, "c8": 16, "c32": 32}
    require(accuracy["status"] == "passed" and {"holdout", "baseline", "structured_holdout"} == suites.keys()
            and suites["baseline"]["verified_requests"] == 144 and suites["holdout"]["verified_requests"] == 88
            and suites["baseline"]["golden_sha256"] == "b32bd641dab4e9516a8d3b45de2d77fbde91ad64acad893153f4b6c2090b72a6"
            and suites["holdout"]["golden_sha256"] == "12d508ed51f03509432e6302b8d10855f56125f4f6b7f31a559cff03eb10b611"
            and suites["holdout"]["workload_sha256"] == structured["workload_sha256"] == HOLDOUT
            and structured["status"] == "passed" and structured["c1_transport_pairs"] == 8
            and {name: row["requests"] for name, row in structured["modes"].items()} == counts
            and all(structured["modes"][name]["exact_matches"] == counts[name] for name in ("c8", "c32"))
            and structured["context_rejection"]["http_status"] == 400, "package lacks exact long or structured outputs")
    return {"api_checks": accuracy["api_checks"], "shutdown_exit": accuracy["shutdown_exit"],
            "report_sha256": digest(root / "accuracy/report.json"), "suites": suites}


def verify(root):
    package = root.parent / "w5-prefix-metrics-package-20261121.nEFyze5I"
    cpu_path = package / "cpu-evidence/report.json"
    cpu, hosted = (read(path) for path in (cpu_path, package / "hosted-report.json"))
    revision = (package / "source-revision.txt").read_text().strip()
    require(cpu["status"] == hosted["status"] == "passed" and cpu["source_revision"] == hosted["source_revision"] == revision
            and cpu["binaries"] == hosted["binaries"] and cpu["source_archive_sha256"] == digest(package / "source.tar"),
            "CPU reports differ or use the wrong source")
    local_files = (package / "cpu-evidence/runtime.sha256").read_text().splitlines()
    require(len(local_files) == 21 and sorted(local_files) == sorted((package / "hosted-runtime.sha256").read_text().splitlines()),
            "runtime files do not match the hosted image")
    reservations = check_reservations(root, "before")
    require(reservations == check_reservations(root, "after-soaks") == check_reservations(root, "after"),
            "GPU reservation holders changed")
    cpu_hash = digest(cpu_path)
    modes = {label: soak(root, label, gpu, cpu, cpu_hash) for label, gpu in MODES.items()}
    accuracy = verify_accuracy(root, cpu, cpu_hash)
    objects = gzip.decompress((package / "source-objects.tsv.gz").read_bytes())
    soaks = {"source_revision": revision, "image_id": cpu["image_id"], "binary_sha256": cpu["binaries"][RILEY],
             "checkpoint_sha256": MODEL, "duration_seconds_per_mode": 1800, "requests_per_second_per_mode": 8,
             "release_gate": "passed" if all(value["status"] == "passed" for value in modes.values()) else "failed",
             "modes": modes}
    return {"status": "passed", "remote_package": str(package), "remote_evidence": str(root),
            "source_revision": revision, "image_id": cpu["image_id"], "source_archive_sha256": cpu["source_archive_sha256"],
            "riley_binary_sha256": cpu["binaries"][RILEY], "cpu_report_sha256": cpu_hash, "checkpoint_sha256": MODEL,
            "source_objects_sha256": sha256(objects).hexdigest(), "source_object_count": len(objects.splitlines()),
            "matching_runtime_files": len(local_files), "reservations": reservations, "soaks": soaks,
            "accuracy": accuracy}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
