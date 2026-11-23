"""Check the CUDA package builds, GPU receipts, and original reservations."""
from hashlib import sha256
import json
from pathlib import Path
import sys


REVISION = "9373ea412a4ce1353a309c69ccf2de62d76d3318"
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
HOSTED_ARCHIVE = "f01a9f3141fb0c3b57c6402177259db0b45f07298277e268e23adf9b712c1765"
GPU_UUIDS = {
    6: "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7",
    7: "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0",
}


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    hosted = read(root / "hosted/report.json")
    local = read(root / "cpu-evidence/report.json")
    for report in (hosted, local):
        require(report["status"] == "passed" and report["source_revision"] == REVISION,
                "unexpected image revision or package status")
    require(hosted["source_archive_sha256"] == HOSTED_ARCHIVE and hosted["binaries"] == local["binaries"],
            "the hosted archive or binaries differ")
    host_files = (root / "hosted/runtime.sha256").read_text().splitlines()
    local_files = (root / "cpu-evidence/runtime.sha256").read_text().splitlines()
    require(len(set(host_files)) == len(host_files) == 21 and sorted(host_files) == sorted(local_files),
            "runtime file manifests differ")
    cpu_hash = sha256((root / "cpu-evidence/report.json").read_bytes()).hexdigest()
    binary = local["binaries"]["/opt/riley/bin/riley"]
    checks = {}
    reservations = {}
    for gpu, name in ((6, "structured"), (7, "smoke")):
        report = read(root / name / "report.json")
        require(report["status"] == "passed" and report["api_checks"] == 40
                and report["tool_checks_enabled"] and report["shutdown_exit"] == 0
                and report["cpu_report_sha256"] == cpu_hash and report["riley_binary_sha256"] == binary
                and report["checkpoint_sha256"] == CHECKPOINT and report["gpu_uuid"] == GPU_UUIDS[gpu]
                and all(report[key] == local[key] for key in ("source_revision", "image_id", "source_archive_sha256")),
                f"unexpected GPU receipt for {name}")
        shutdown = read(root / name / "shutdown.json")
        require(all(shutdown[key] == 0 for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
                and shutdown["allocation"] and all(value == 0 for value in shutdown["allocation"].values()),
                f"resources remained after {name}")
        holders = [read(root / f"gpu{gpu}-{phase}.json") for phase in ("before", "after")]
        for holder in holders:
            require(holder["phase"] == "idle" and holder["status"] == "running"
                    and holder["physical_index"] == gpu and holder["gpu"] == GPU_UUIDS[gpu]
                    and holder["allocation_bytes"] == 20 * 2**30 and holder["other_processes"] == [],
                    f"GPU {gpu} reservation was not restored")
        require(all(holders[0][key] == holders[1][key] for key in ("pid", "start_ticks")),
                f"GPU {gpu} reservation identity changed")
        reservations[str(gpu)] = {key: holders[1][key] for key in ("pid", "start_ticks", "allocation_bytes")}
        checks[name] = report

    holdout = checks["structured"]["accuracy_suites"]["structured_holdout"]
    modes = holdout["modes"]
    require(holdout["status"] == "passed"
            and {name: mode["requests"] for name, mode in modes.items()} == {"c1": 16, "c8": 16, "c32": 32}
            and all(modes[name]["exact_matches"] == modes[name]["requests"] for name in ("c8", "c32")),
            "structured-output holdout did not preserve the C1 outputs")
    smoke = checks["smoke"]["serving_smoke"]
    require(smoke["status"] == "passed" and smoke["seconds"] == 45
            and checks["smoke"]["prefix_cache_metrics"]["status"] == "passed",
            "mixed GPU smoke or cache metrics failed")
    return {"status": "verified", "source_revision": REVISION, "hosted_image": hosted["image_id"],
            "a100_image": local["image_id"], "matching_runtime_files": len(local_files),
            "riley_binary_sha256": binary, "reservations": reservations, "structured_modes": modes,
            "smoke_counts": smoke["counts"], "api_checks_per_gpu": 40}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
