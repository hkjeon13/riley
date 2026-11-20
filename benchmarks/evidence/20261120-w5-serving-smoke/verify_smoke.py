"""Summarize completed GPU smoke runs without exporting raw responses or logs."""

from hashlib import file_digest
import json
from pathlib import Path
import sys


HOLDERS = {6: (2431228, "6339570869"), 7: (2431237, "6339570875")}
GPU = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
IMAGE = "sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1"
BINARY = "91a2de79577e54ab7fc15de78d8cacb6915f9aacfafae19b66215e015567a0ec"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def reservations(root, point):
    for index, (pid, ticks) in HOLDERS.items():
        j = read(root / f"gpu{index}-{point}.json")
        assert (j["physical_index"], j["pid"], j["start_ticks"]) == (index, pid, ticks)
        assert j["phase"] == "idle" and not j["other_processes"]
        assert j["allocation_bytes"] == j["torch_reserved_bytes"] == 20*1024**3


def check(root, checkpoint):
    reservations(root, "before")
    reservations(root, "after")
    arm = root / "gpu-evidence"
    terminal = int((root / "exit").read_text())
    if not (arm / "report.json").is_file():
        assert terminal != 0
        result = {"status": "execution_failed", "exit": terminal, "driver_log_sha256": digest(root / "driver.log")}
        if (arm / "api.json").is_file():
            api = read(arm / "api.json")["results"]
            result["api_checks_passed"] = sum(record["passed"] is True for record in api)
            result["api_failed_check_names"] = [record["name"] for record in api if record["passed"] is not True]
        return result
    path = arm / "report.json"
    report = read(path)
    assert (report["checkpoint_sha256"], report["image_id"], report["gpu_uuid"], report["riley_binary_sha256"]) == (
        checkpoint, IMAGE, GPU, BINARY)
    assert report["api_checks"] == 40 and report["shutdown_exit"] == 0
    assert report["tool_checks_enabled"] is True
    for name, expected in report["artifacts_sha256"].items():
        assert (arm / name).parent == arm and digest(arm / name) == expected
    raw = read(arm / "smoke.json")
    smoke = report["serving_smoke"]
    assert smoke["seconds"] == raw["seconds"] == 45 and smoke["rate"] == raw["rate"] == 8
    assert smoke["counts"] == raw["counts"] and smoke["checks"] == raw["checks"]
    assert smoke["counter_changes"] == raw["counter_changes"]
    assert report["status"] == smoke["status"] == ("passed" if terminal == 0 else "failed")
    return {"status": report["status"], "exit": terminal, "api_checks": report["api_checks"],
            "checkpoint_sha256": checkpoint, "checkpoint_source_model": report["checkpoint_source_model"],
            "checkpoint_source_revision": report["checkpoint_source_revision"],
            "report_sha256": digest(path), "smoke_sha256": digest(arm / "smoke.json"), "smoke": smoke}


def summarize(root):
    models = (row.split() for row in (root / "models.tsv").read_text().splitlines())
    results = {name: check(root / name, expected) for name, expected in models}
    return {"remote_stage": str(root), "image_id": IMAGE, "binary_sha256": BINARY, "gpu_uuid": GPU,
            "idle_reservations_gib": {"6": 20, "7": 20}, "results": results}


if __name__ == "__main__":
    result = summarize(Path(sys.argv[1]))
    print(json.dumps(result, indent=2, allow_nan=False))
