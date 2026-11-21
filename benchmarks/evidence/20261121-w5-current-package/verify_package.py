"""Verify an A100 reconstruction of the current committed distribution inputs."""

from hashlib import sha256
import json
from pathlib import Path
import sys


GPU = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
MODEL = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
RILEY = "/opt/riley/bin/riley"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    hashed = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hashed.update(block)
    return hashed.hexdigest()


def reservations(root, phase):
    holders = {}
    for index in (6, 7):
        row = read(root / f"gpu{index}-{phase}.json")
        assert row["physical_index"] == index and row["phase"] == "idle"
        assert row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
        assert row["other_processes"] == []
        holders[str(index)] = {"pid": row["pid"], "start_ticks": row["start_ticks"], "gib": 20}
    return holders


def verify(root):
    cpu_path = root / "cpu-evidence/report.json"
    cpu = read(cpu_path)
    gpu = root / "gpu-evidence"
    report = read(gpu / "report.json")
    revision = (root / "source-revision.txt").read_text().strip()
    source = digest(root / "source.tar")
    assert cpu["status"] == report["status"] == "passed"
    assert revision == cpu["source_revision"] == report["source_revision"]
    assert source == cpu["source_archive_sha256"] == report["source_archive_sha256"]
    assert cpu["image_id"] == report["image_id"] == (root / "image.id").read_text().strip()
    assert cpu["binaries"][RILEY] == report["riley_binary_sha256"] and report["cpu_report_sha256"] == digest(cpu_path)
    assert report["gpu_uuid"] == GPU and report["checkpoint_sha256"] == MODEL
    assert report["api_checks"] == 40 and report["tool_checks_enabled"] is True and report["shutdown_exit"] == 0
    for name, expected in report["artifacts_sha256"].items():
        assert Path(name).name == name and digest(gpu / name) == expected
    api = read(gpu / "api.json")["results"]
    assert len(api) == 40 and all(result["passed"] is True for result in api)
    smoke = read(gpu / "smoke.json")
    assert smoke["status"] == "passed" and smoke["seconds"] == 45 and smoke["rate"] == 8
    assert all(smoke["checks"].values()) and smoke["problem_count"] == 0 and smoke["shared_prefixes"] == 8
    completed = sum(count.get("completed", 0) for count in smoke["counts"].values())
    disconnected = smoke["counts"]["cancel"]["disconnected"]
    assert disconnected == smoke["counter_changes"]["disconnects"] == smoke["counter_changes"]["cancellations"]
    cleanup = read(gpu / "shutdown.json")
    assert not any(cleanup[key] for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
    assert cleanup["allocation"] and not any(cleanup["allocation"].values())
    holders = reservations(root, "after")
    assert holders == reservations(root, "before")
    return {"status": "passed", "remote_stage": root.name, "source_revision": revision,
            "archive_kind": "verified committed build inputs", "source_archive_sha256": source,
            "source_objects_sha256": digest(root / "source-objects.tsv"),
            "source_object_count": len((root / "source-objects.tsv").read_text().splitlines()),
            "image_id": cpu["image_id"], "riley_binary_sha256": cpu["binaries"][RILEY],
            "cpu_report_sha256": digest(cpu_path), "gpu_report_sha256": digest(gpu / "report.json"),
            "gpu_uuid": GPU, "checkpoint_sha256": MODEL, "api_checks": len(api),
            "smoke": {"seconds": smoke["seconds"], "rate": smoke["rate"], "completed": completed,
                      "client_disconnects": disconnected, "server_cancellations": smoke["counter_changes"]["cancellations"],
                      "problems": smoke["problem_count"], "effective_server_args": report["serving_smoke"]["server_arguments"]},
            "shutdown": "passed", "idle_reservations": holders}


if __name__ == "__main__":
    if sys.argv[1] == "--reservations":
        reservations(Path.cwd(), sys.argv[2])
    else:
        print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
