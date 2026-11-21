"""Verify both real serving modes and export a compact receipt."""

from hashlib import sha256
import json
from pathlib import Path
import sys

HOLDERS = {6: (2431228, "6339570869"), 7: (2431237, "6339570875")}
MODES = ("immediate-v1", "after-submit-v1")
GPU = "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0"
CHECKPOINT = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def reservations(folder, phase):
    for index, (pid, ticks) in HOLDERS.items():
        row = read(folder / f"gpu{index}-{phase}.json")
        assert (row["physical_index"], row["pid"], row["start_ticks"]) == (index, pid, ticks)
        assert row["phase"] == "idle" and row["other_processes"] == []
        assert row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30


def verify(root):
    binary, name = (root / "candidate.sha256").read_text().split()
    assert name == "bin/riley" and digest(root / name) == binary
    chronology = [line.split() for line in (root / "campaign.log").read_text().splitlines()
                  if line.startswith(("START ", "END "))]
    assert [tuple(row[:2]) for row in chronology] == [(event, mode) for mode in MODES for event in ("START", "END")]
    assert all(row[3:] == ["exit=0"] for row in chronology[1::2])
    results = {}
    for mode in MODES:
        folder = root / mode
        reservations(folder, "before")
        reservations(folder, "after")
        receipt = read(folder / "receipt.json")
        assert receipt["status"] == "passed" and receipt["server_exit"] == receipt["client_exit"] == 0
        assert receipt["gpu_uuid"] == GPU and receipt["observed_active"] and receipt["nvml_samples"] > 0
        assert receipt["server_argv"][0] == str(root / name)
        assert receipt["server_argv"][-4:] == ["--event-publication", mode, "--prefix-cache-blocks", "128"]
        assert receipt["sha256"][str(root / name)] == binary
        assert receipt["sha256"][str(root.parent.parent / "models/Qwen2.5-7B-Instruct/riley-checkpoint.json")] == CHECKPOINT
        for key, file in (("api", "api.json"), ("soak", "soak.json"), ("shutdown_metrics", "shutdown-metrics.json")):
            assert receipt[f"{key}_sha256"] == digest(folder / file)
        api = read(folder / "api.json")["results"]
        soak = read(folder / "soak.json")
        shutdown = read(folder / "shutdown-metrics.json")
        assert len(api) == 40 and all(row["passed"] is True for row in api)
        assert soak["status"] == "passed" and soak["problem_count"] == 0 and soak["seconds"] == 45
        assert soak["rate"] == 8 and all(soak["checks"].values())
        counts = soak["counts"]
        changes = soak["counter_changes"]
        disconnected = counts["cancel"]["disconnected"]
        assert disconnected > 0 and disconnected == changes["disconnects"] == changes["cancellations"]
        assert not any(shutdown[key] for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
        assert shutdown["allocation"] and not any(shutdown["allocation"].values())
        results[mode] = {"api_checks": len(api), "completed": sum(row.get("completed", 0) for row in counts.values()),
                         "client_disconnects": disconnected, "disconnect_counter": changes["disconnects"],
                         "cancellation_counter": changes["cancellations"], "problem_count": soak["problem_count"],
                         "shutdown": "passed", "receipt_sha256": digest(folder / "receipt.json")}
    return {"status": "passed", "source_revision": "d3d8fe6e5b1c528bff51c0008cdc24544c99f862",
            "patched_engine_sha256": (root / "source-after.sha256").read_text().split()[0],
            "binary_sha256": binary, "checkpoint_sha256": CHECKPOINT, "gpu": GPU,
            "remote_stage": root.name, "duration_seconds_per_mode": 45, "request_rate_per_second": 8,
            "idle_reservations_gib": {"6": 20, "7": 20}, "modes": results}


if __name__ == "__main__":
    if sys.argv[1] == "--reservations":
        reservations(Path(sys.argv[2]), sys.argv[3])
    else:
        print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
