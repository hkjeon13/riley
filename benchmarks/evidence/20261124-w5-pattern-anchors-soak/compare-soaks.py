"""Check the two current-image soaks and compare them to the saved allocator runs."""

import json
from pathlib import Path
import sys

from allocator import bind
from verify import MODES, check_reservations, digest, read, require, soak


def delta(before, after):
    return after - before if before is not None and after is not None else None


def compare(root, variant, cpu_path):
    stock = read(cpu_path)
    candidate, parent = bind(variant, cpu_path)
    saved = read(variant / "verified.json")
    binding = variant / "gpu-checker-input.json"
    require(read(binding) == candidate and saved["image_id"] == candidate["image_id"]
            and saved["parent_image_id"] == parent == stock["image_id"]
            and saved["source_revision"] == stock["source_revision"], "image pairing differs")
    reservations = check_reservations(root, "before")
    require(reservations == saved["reservations"] == check_reservations(root, "after-soaks"),
            "standing GPU reservations changed")
    pairings = {}
    stock_modes = {}
    for mode, gpu in MODES.items():
        previous = soak(variant, mode, gpu, candidate, digest(binding))
        current = soak(root, mode, gpu, stock, digest(cpu_path))
        require(previous == saved["soaks"]["modes"][mode], f"archived allocator verdict differs for {mode}")
        stock_run, allocator_run = (read(folder / mode / "report.json") for folder in (root, variant))
        for field in ("gpu_uuid", "gpu_name", "driver_version", "checkpoint_sha256", "soak_server_arguments"):
            require(stock_run[field] == allocator_run[field], f"{mode} changed {field} between images")
        stock_workload, allocator_workload = (read(folder / mode / "soak.json") for folder in (root, variant))
        pairings[mode] = {
            "gpu_uuid": gpu,
            "driver_version": stock_run["driver_version"],
            "server_arguments": stock_run["soak_server_arguments"],
            "stock_to_allocator_rss_change_kib": delta(current["rss"]["rss_growth_kib"],
                                                       previous["rss"]["rss_growth_kib"]),
            "stock_counts": stock_workload["counts"],
            "allocator_counts": allocator_workload["counts"],
            "stock_latency_windows": stock_workload["latency_windows"],
            "allocator_latency_windows": allocator_workload["latency_windows"],
            "stock_client_dispatch_p95_ms": stock_workload["client_dispatch_p95_ms"],
            "allocator_client_dispatch_p95_ms": allocator_workload["client_dispatch_p95_ms"],
        }
        stock_modes[mode] = current
    return {
        "status": "verified",
        "remote_evidence": str(root),
        "parent_cpu_report_sha256": digest(cpu_path),
        "allocator_report_sha256": digest(variant / "verified.json"),
        "reservations": reservations,
        "stock_soaks": {
            **{key: stock[key] for key in ("source_revision", "image_id")},
            "binary_sha256": saved["riley_binary_sha256"],
            "checkpoint_sha256": saved["checkpoint_sha256"],
            "duration_seconds_per_mode": 1800,
            "requests_per_second_per_mode": 8,
            "release_gate": "passed" if all(row["status"] == "passed" for row in stock_modes.values()) else "failed",
            "modes": stock_modes,
        },
        "comparisons": pairings,
    }


if __name__ == "__main__":
    print(json.dumps(compare(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
