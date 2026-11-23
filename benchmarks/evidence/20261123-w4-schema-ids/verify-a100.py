"""Verify the committed image, schema-reference HTTP checks and GPU release."""

from hashlib import sha256
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    revision = (root / "source-revision.txt").read_text().strip()
    cpu_path = root / "cpu-evidence/report.json"
    cpu = read(cpu_path)
    require(cpu["status"] == "passed" and cpu["source_revision"] == revision,
            "unexpected package validation")
    cpu_hash = sha256(cpu_path.read_bytes()).hexdigest()
    binary = cpu["binaries"]["/opt/riley/bin/riley"]
    reports = {}
    reservations = {}
    for index, mode in ((6, "structured"), (7, "smoke")):
        report = read(root / mode / "report.json")
        require(report["status"] == "passed" and report["api_checks"] == 40
                and report["tool_checks_enabled"] and report["shutdown_exit"] == 0,
                f"incomplete API checks or shutdown on GPU {index}")
        require(all(report[key] == cpu[key] for key in ("source_revision", "image_id", "source_archive_sha256"))
                and report["riley_binary_sha256"] == binary and report["cpu_report_sha256"] == cpu_hash
                and report["checkpoint_sha256"] == "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f",
                f"GPU {index} used another package or checkpoint")
        shutdown = read(root / mode / "shutdown.json")
        require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                and bool(shutdown["allocation"]) and not any(shutdown["allocation"].values()),
                f"GPU {index} left live resources")
        before, after = (read(root / f"gpu{index}-{phase}.json") for phase in ("before", "after"))
        require(all(holder["status"] == "running" and holder["phase"] == "idle"
                    and holder["physical_index"] == index and holder["gpu"] == report["gpu_uuid"]
                    and holder["allocation_bytes"] == 20 * 2**30 and not holder["other_processes"]
                    for holder in (before, after))
                and all(before[key] == after[key] for key in ("pid", "start_ticks")),
                f"GPU {index} did not restore the same reservation")
        reservations[str(index)] = {key: after[key] for key in ("pid", "start_ticks", "allocation_bytes")}
        reports[mode] = report
    holdout = reports["structured"]["accuracy_suites"]["structured_holdout"]
    modes = holdout["modes"]
    require(holdout["status"] == "passed"
            and {name: mode["requests"] for name, mode in modes.items()} == {"c1": 16, "c8": 16, "c32": 32}
            and all(modes[name]["exact_matches"] == modes[name]["requests"] for name in ("c8", "c32")),
            "the structured holdout did not preserve batch-invariant output")
    smoke = reports["smoke"]["serving_smoke"]
    require(smoke["status"] == "passed" and smoke["seconds"] == 45
            and reports["smoke"]["prefix_cache_metrics"]["status"] == "passed",
            "mixed smoke or cache metrics did not pass")
    pattern = read(root / "structured/pattern-classes.json")
    require(pattern["status"] == "passed" and len(pattern["checks"]) == 10,
            "the prior pattern-class coverage is missing")
    probes = read(root / "structured/schema-references.json")
    checks = probes["checks"]
    expected = {(case, parameter): status for case, status in (
        ("nested-id", 400), ("cross-resource-pointer", 400), ("root-id-local-pointer", 200))
        for parameter in ("response_format", "tools")}
    require(probes["status"] == "passed" and len(checks) == len(expected)
            and {(row["case"], row["parameter"]): row["status"] for row in checks} == expected
            and all(row["output"] == {"$id": 3} if row["status"] == 200 else
                    row["error"]["code"] == "invalid_schema" and row["error"]["param"] == row["parameter"]
                    for row in checks), "unexpected schema-reference responses")
    return {"status": "verified", "source_revision": revision, "image_id": cpu["image_id"],
            "binary_sha256": binary, "api_checks_per_gpu": 40, "structured_modes": modes,
            "smoke_counts": smoke["counts"], "schema_reference_checks": checks,
            "a100_image": cpu["image_id"], "riley_binary_sha256": binary,
            "pattern_class_checks": pattern["checks"], "reservations": reservations}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
