"""Recompute shape evidence for the live mixed smoke and earlier 30-minute runs."""
import json
from pathlib import Path
import sys

from check_distribution_gpu import batch_shape_evidence
from current_binding import binding, check_reservations
from verify_a100_candidate import CHECKPOINT, GPU, ROOTS, RILEY, digest, gpu_evidence, read, require


SOAKS = {
    "stock": ("w5-current-soaks-20261121.OyGugR0K", "5453a6c0bc14d3d4b228373bd329c0f6eb4c69d40408153df2d2d9d9046c2279"),
    "jemalloc": ("w5-current-allocator-20261121.ZVgY3v8o", "d7c45a682ec1f0eaa3f24ad51085668c0126331d44077cc68ca23a215b87c842"),
}


def verify(stage):
    evidence = stage.parent
    stock = evidence / ROOTS["stock"] / "cpu-evidence/report.json"
    package = evidence / SOAKS["jemalloc"][0]
    adapter = stage / "gpu-checker-input.json"
    require(read(adapter) == binding(package, evidence), "allocator image binding differs")
    order = ("stock", "jemalloc")
    progression = [line.split() for line in (stage / "campaign.log").read_text().splitlines()
                   if line.startswith(("START ", "END "))]
    require([row[:2] for row in progression] == [[phase, name] for name in order for phase in ("START", "END")]
            and [row[2] for row in progression] == sorted(row[2] for row in progression), "mixed smoke order differs")
    live, retrospective = {}, {}
    for name, cpu in (("stock", stock), ("jemalloc", adapter)):
        folder = stage / name
        spec = read(cpu)
        expected = {key: spec[key] for key in ("image_id", "source_revision", "source_archive_sha256")}
        expected.update(riley_binary_sha256=spec["binaries"][RILEY], cpu_report_sha256=digest(cpu),
                        checkpoint_sha256=CHECKPOINT, gpu_uuid=GPU)
        report = gpu_evidence(folder, expected, int((folder / "exit").read_text()))
        process = read(folder / "process-before.json")
        path = folder / "gpu-evidence/smoke.json"
        raw = read(path)
        result = report["serving_smoke"]
        shape = batch_shape_evidence(raw)
        require(process == read(folder / "process-after.json") and process["image_id"] == expected["image_id"]
                and process["args"][5:] == result["server_arguments"]
                and {"container_id": process["container_id"], "host_pid": process["host_pid"], "container_pid": 1}
                == raw["rss_source"], "live process, workload or samples differ")
        require(result["seconds"] == raw["seconds"] == 45 and result["rate"] == raw["rate"] == 8
                and raw["shared_prefixes"] == 8 and result["batch_shape_metrics"] == shape
                and result["checks"]["current_batch_shape_metrics"] == (shape["status"] == "passed"),
                "current metrics summary does not match the observed samples")
        live[name] = {"image_id": report["image_id"], "status": report["status"],
                      "api_checks": report["api_checks"], "shutdown": report["shutdown_metrics"],
                      "gate": shape, "process": process, "reservations": check_reservations(folder, "after"),
                      "report_sha256": report["gpu_receipt_sha256"], "smoke_sha256": digest(path)}
        previous, summary_sha = SOAKS[name]
        root = evidence / previous
        require(digest(root / "verified.json") == summary_sha, "archived soak receipt differs")
        retrospective[name] = {}
        for mode in ("cache-off", "cache-on"):
            old = root / mode
            receipt, raw = read(old / "report.json"), read(old / "soak.json")
            digest_actual = digest(old / "soak.json")
            require(receipt["image_id"] == report["image_id"] and raw["seconds"] == 1800
                    and receipt["artifacts_sha256"]["soak.json"] == digest_actual,
                    "archived soak was not measured on the bound image")
            retrospective[name][mode] = {"original_soak_status": raw["status"], "soak_sha256": digest_actual,
                                         "original_summary_sha256": summary_sha, "gate": batch_shape_evidence(raw)}
    require(live["stock"]["reservations"] == live["jemalloc"]["reservations"], "reservation owner changed")
    return {"stage": stage.name, "checker_sha256": digest(stage / "check_distribution_gpu.py"),
            "checkpoint_sha256": CHECKPOINT, "binary_sha256": read(stock)["binaries"][RILEY],
            "run_order": progression, "measured_smoke": live, "archived_thirty_minute_soaks": retrospective}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
