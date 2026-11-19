"""Bind the mixed-load observer comparison to the exact allocator image and GPU run."""

import json
from pathlib import Path
import sys

from compare_allocator_performance import SERVER_ARGS, expected_image
from observer_client import PHASES, summary
from qualify_allocator import check_reservations
from verify_a100_candidate import digest, gpu_evidence, read, require


def verify(package, evidence, stage):
    arm = "packaged-first"
    folder = stage / arm
    _, expected = expected_image(package, evidence, stage, arm)
    rc = int((folder / "exit").read_text())
    receipt = gpu_evidence(folder, expected, rc)
    require(receipt["status"] == "passed", "GPU API or shutdown check failed")
    process = read(folder / "process-before.json")
    require(process == read(folder / "process-after.json") and tuple(process["server_arguments"]) == SERVER_ARGS
            and process["image_id"] == receipt["image_id"], "wrong allocator image or running process")
    reservations = check_reservations(folder, "after")
    require(reservations == check_reservations(folder, "before"), "GPU reservations did not return")
    raw = folder / "gpu-evidence"
    for name, _ in PHASES:
        require(read(raw / f"{name}.observer.json")["process"] == process, "CPU/RSS reported for another process")
    return {"image_id": receipt["image_id"], "candidate_binary_sha256": receipt["riley_binary_sha256"],
            "checkpoint_sha256": receipt["checkpoint_sha256"], "api_checks": receipt["api_checks"],
            "receipt_sha256": digest(raw / "report.json"), "reservations": reservations,
            **summary(raw, receipt["artifacts_sha256"])}


if __name__ == "__main__":
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
