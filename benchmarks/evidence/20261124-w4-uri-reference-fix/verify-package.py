"""Match the hosted image runtime to the A100 image and its real-model checks."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy

REVISION = "8a7c618c1882effcacd49107a0b41ea3fc29d22d"
HOSTED_ARCHIVE_SHA256 = "2dcfc04d16de98305a44be08607d5b7715205cf8197081f187e50eac552ab2a4"
A100_ARCHIVE_SHA256 = "b1111fd11109c5b70d3381c977d20ea5b09daf6d8f0aa9c6f28bdda52eaaf489"
CHECKPOINT_SHA256 = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def runtime(path):
    lines = [line.split("  ", 1) for line in path.read_text().splitlines()]
    files = {name: hash_ for hash_, name in lines}
    require(len(files) == len(lines) == 21 and all(re.fullmatch(r"[0-9a-f]{64}", hash_)
                                                   for hash_ in files.values()), "invalid runtime manifest")
    return files


def arm(a100, name, index, cpu, cpu_hash):
    out = a100 / name
    report = read(out / "report.json")
    require(report["status"] == "passed" and report["api_checks"] == 40 and report["tool_checks_enabled"]
            and report["shutdown_exit"] == 0 and report["source_revision"] == REVISION
            and report["image_id"] == cpu["image_id"] and report["source_archive_sha256"] == A100_ARCHIVE_SHA256
            and report["riley_binary_sha256"] == cpu["binaries"]["/opt/riley/bin/riley"]
            and report["cpu_report_sha256"] == cpu_hash and report["checkpoint_sha256"] == CHECKPOINT_SHA256,
            f"unexpected serving package on GPU {index}")
    for member in ("api.json", "shutdown.json"):
        require(digest(out / member) == report["artifacts_sha256"][member], f"serving artifact changed: {member}")
    checks = read(out / "api.json")["results"]
    require(len(checks) == 40 and all(row["passed"] is True for row in checks), "real-model API did not pass")
    shutdown = read(out / "shutdown.json")
    require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            and shutdown["allocation"] and not any(shutdown["allocation"].values()), "server retained resources")
    before, after = (read(a100 / f"gpu{index}-{phase}.json") for phase in ("before", "after"))
    require(all(row["status"] == "running" and row["phase"] == "idle" and row["gpu"] == report["gpu_uuid"]
                and row["physical_index"] == index and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
                and not row["other_processes"] for row in (before, after))
            and (before["pid"], before["start_ticks"]) == (after["pid"], after["start_ticks"]),
            "same 20GiB holder did not recover")
    return report, {key: after[key] for key in ("pid", "start_ticks", "allocation_bytes")}


def verify(directory):
    a100 = directory / "a100"
    hosted = directory / "hosted"
    require((a100 / "campaign.exit").read_text().strip() == "0"
            and (a100 / "source-revision.txt").read_text().strip() == REVISION, "wrong A100 campaign")
    cpu_file = a100 / "cpu-evidence/report.json"
    cpu, github = read(cpu_file), read(hosted / "report.json")
    for package, expected in ((cpu, A100_ARCHIVE_SHA256), (github, HOSTED_ARCHIVE_SHA256)):
        require(package["status"] == "passed" and package["source_revision"] == REVISION
                and package["source_archive_sha256"] == expected, "missing passing CPU package receipt")
    require((hosted / "source-archive.sha256").read_text().strip() == HOSTED_ARCHIVE_SHA256
            and (a100 / "image.id").read_text().strip() == cpu["image_id"], "incorrect image or source archive")
    local, host = runtime(a100 / "cpu-evidence/runtime.sha256"), runtime(hosted / "runtime.sha256")
    require(local == host and all(local[name] == digest_ for name, digest_ in cpu["binaries"].items())
            and cpu["binaries"] == github["binaries"], "hosted and A100 runtime digests differ")
    structured, gpu6 = arm(a100, "structured", 6, cpu, digest(cpu_file))
    smoke, gpu7 = arm(a100, "smoke", 7, cpu, digest(cpu_file))
    holdout = structured["accuracy_suites"]["structured_holdout"]
    counts = {"c1": 16, "c8": 16, "c32": 32}
    require(holdout["status"] == "passed" and {key: value["requests"] for key, value in holdout["modes"].items()} == counts
            and all(holdout["modes"][key]["exact_matches"] == counts[key] for key in ("c8", "c32")), "structured outputs drifted")
    load = smoke["serving_smoke"]
    require(load["status"] == "passed" and (load["seconds"], load["rate"]) == (45, 8)
            and not load["failed_checks"] and load["checks"] and all(load["checks"].values())
            and smoke["prefix_cache_metrics"]["status"] == "passed", "mixed smoke or metrics failed")
    schema = runpy.run_path(str(Path(__file__).with_name("verify-a100.py")))["verify"](a100)
    return {"status": "verified", "source_revision": REVISION, "matching_runtime_files": len(local),
            "schema_reference_checks": schema["schema_reference_checks"],
            "pattern_class_checks": schema["pattern_class_checks"],
            "hosted_image": github["image_id"], "a100_image": cpu["image_id"],
            "riley_binary_sha256": cpu["binaries"]["/opt/riley/bin/riley"],
            "checkpoint_sha256": CHECKPOINT_SHA256, "a100_source_archive_sha256": A100_ARCHIVE_SHA256,
            "hosted_source_archive_sha256": HOSTED_ARCHIVE_SHA256, "api_checks_per_gpu": 40,
            "structured_modes": holdout["modes"], "smoke_counts": load["counts"], "reservations": {"6": gpu6, "7": gpu7}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2, allow_nan=False))
