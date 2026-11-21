"""Check complete release soaks and preserve failed thresholds as measured outcomes."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

IMAGE = "sha256:b731f48cc6ec0820197ec484e98c673e2de592f12ab8448e3fa9f38ca7217c9b"
BINARY = "56112a5b9bd72e77106f45497af277038308af357a5c5ee129af74ce3cc85a0f"
SOURCE = "36aa1880509941ad169be6cc327d6d89231f8b81"
CPU_REPORT = "9aba66838fb889e749e9a9235e0d633408d79917bc71a5924abb4cdc5396ddf3"
MODEL = "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
GPUS = {"cache-off": (6, "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7"),
        "cache-on": (7, "GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0")}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def reservations(root, phase):
    observed = {}
    for index in (6, 7):
        row = read(root / f"gpu{index}-{phase}.json")
        assert (row["status"], row["phase"], row["physical_index"]) == ("running", "idle", index)
        assert row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
        assert not row["other_processes"]
        observed[str(index)] = {"pid": row["pid"], "start_ticks": row["start_ticks"], "gib": 20}
    return observed


def verify_arm(root, mode, device):
    index, uuid = device
    folder = root / mode
    report = read(folder / "report.json")
    soak = read(folder / "soak.json")
    api = read(folder / "api.json")["results"]
    shutdown = read(folder / "shutdown.json")
    assert (report["source_revision"], report["image_id"], report["riley_binary_sha256"],
            report["checkpoint_sha256"], report["gpu_uuid"], report["cpu_report_sha256"]) == (
                SOURCE, IMAGE, BINARY, MODEL, uuid, CPU_REPORT)
    assert report["api_checks"] == len(api) == 40 and all(row["passed"] is True for row in api)
    assert report["tool_checks_enabled"] is True and report["shutdown_exit"] == 0
    assert all(digest(folder / name) == value for name, value in report["artifacts_sha256"].items())
    args = ["--max-active-sequences", "32", "--max-sequence-tokens", "8192", "--max-output-tokens", "512",
            "--kv-blocks", "16384"]
    if mode == "cache-on":
        args += ["--prefix-cache-blocks", "128"]
    assert report["soak_server_arguments"] == args
    assert (soak["mode"], soak["seconds"], soak["rate"], soak["shared_prefixes"]) == ("mixed_soak", 1800, 8, 8)
    assert all(type(passed) is bool for passed in soak["checks"].values())
    percents = soak["p95_change_percent"]
    assert soak["checks"]["p95_drift_at_most_5_percent"] is all(
        value is not None and isfinite(value) and value <= 5 for value in percents.values())
    rss = soak["smaps_rollup_second_half"]
    delta = rss["rss_growth_kib"]
    if delta is not None:
        assert delta == rss["drain"]["rss_kib"] - rss["baseline"]["rss_kib"]
    assert soak["checks"]["rss_growth_second_half_at_most_zero"] is (delta is not None and delta <= 0)
    failed = sorted(name for name, passed in soak["checks"].items() if not passed)
    expected = "failed" if failed else "passed"
    assert report["status"] == soak["status"] == report["mixed_soak"]["status"] == expected
    assert report["mixed_soak"]["failed_checks"] == failed
    assert int((root / f"{mode}.exit").read_text()) == int(bool(failed))
    assert not any(shutdown[key] for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
    assert shutdown["allocation"] and not any(shutdown["allocation"].values())
    counts = soak["counts"]
    return {"physical_gpu": index, "status": expected, "api_checks": len(api),
            "completed": sum(row.get("completed", 0) for row in counts.values()),
            "client_disconnects": counts["cancel"].get("disconnected", 0),
            "problem_count": soak["problem_count"], "counter_changes": soak["counter_changes"],
            "p95_change_percent": percents, "rss": rss, "failed_checks": failed,
            "server_arguments": args, "report_sha256": digest(folder / "report.json"),
            "soak_sha256": digest(folder / "soak.json"), "shutdown_sha256": digest(folder / "shutdown.json")}


def verify(root):
    holders = reservations(root, "after")
    assert holders == reservations(root, "before")
    results = {mode: verify_arm(root, mode, device) for mode, device in GPUS.items()}
    return {"remote_stage": root.name, "source_revision": SOURCE, "image_id": IMAGE, "binary_sha256": BINARY,
            "checkpoint_sha256": MODEL, "duration_seconds_per_mode": 1800, "requests_per_second_per_mode": 8,
            "parallel_arms": True, "idle_reservations": holders,
            "release_gate": "passed" if all(run["status"] == "passed" for run in results.values()) else "failed",
            "modes": results}


if __name__ == "__main__":
    if sys.argv[1] == "--reservations":
        reservations(Path(sys.argv[2]), sys.argv[3])
    else:
        print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
