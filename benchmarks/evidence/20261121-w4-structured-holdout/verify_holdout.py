"""Verify image provenance, independent schema assertions, and idle restoration."""

import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

from current_binding import LIB_SHA, PROFILE, binding, check_reservations
from structured_holdout_check import SCHEMA
from verify_a100_candidate import CHECKPOINT, GPU, RILEY, digest, gpu_evidence, read, require


stage = Path(sys.argv[1])
expected_images = {
    "stock": "sha256:b731f48cc6ec0820197ec484e98c673e2de592f12ab8448e3fa9f38ca7217c9b",
    "jemalloc": "sha256:c3fae872c5325cffbf02e4c41004514cc6b38bb364bb33d805f090beed1d5058",
}
summary = {"status": "passed", "remote_evidence": str(stage), "runs": {}}
stock_cpu = stage.parent / "w5-current-package-20261121.f6SEbjvA/cpu-evidence/report.json"
adapter = stage / "gpu-checker-input.json"
package = stage.parent / "w5-current-allocator-20261121.ZVgY3v8o"
require(read(adapter) == binding(package, stage.parent), "allocator image binding differs")
for name, image in expected_images.items():
    directory = stage / name
    output = directory / "gpu-evidence"
    require((directory / "exit").read_text().strip() == "0", f"{name} holdout failed")
    cpu = stock_cpu if name == "stock" else adapter
    spec = read(cpu)
    expected = {"image_id": image, "source_revision": "36aa1880509941ad169be6cc327d6d89231f8b81",
                "source_archive_sha256": spec["source_archive_sha256"], "checkpoint_sha256": CHECKPOINT,
                "gpu_uuid": GPU, "riley_binary_sha256": spec["binaries"][RILEY], "cpu_report_sha256": digest(cpu)}
    report = gpu_evidence(directory, expected, 0)
    process = read(directory / "process-before.json")
    allocation = process["allocator"]
    if name == "stock":
        require(allocation == "glibc", "stock process did not confirm glibc")
    else:
        require(allocation["library_sha256"] == LIB_SHA and allocation["malloc_conf"] == PROFILE
                and allocation["container_id"] == process["container_id"]
                and allocation["host_pid"] == process["host_pid"], "jemalloc process did not match reviewed settings")
    require(process == read(directory / "process-after.json") and process["image_id"] == image
            and process["args"] == ["serve", "--model", "/models", "--model-id", "m",
                                    *report["accuracy_server_arguments"]], f"{name} process changed")
    reservation = check_reservations(directory, "after")
    holdout = read(output / "structured-holdout.json")
    require(holdout == report["accuracy_suites"]["structured_holdout"], "holdout receipt does not match outer report")
    require(holdout["workload_sha256"] == "220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11"
            and holdout["status"] == "passed" and holdout["c1_transport_pairs"] == 8
            and len(holdout["cases"]) == 16, "incorrect holdout coverage")
    rejection = holdout["context_rejection"]
    require(rejection["sequence_tokens_requested"] == 8212 and rejection["max_sequence_tokens"] == 8192
            and rejection["http_status"] == 400 and rejection["error_code"] == "invalid_request"
            and "request requires 8211 sequence tokens, exceeding the 8192-token limit"
            in (output / "server.log").read_text(), f"{name} did not report the expected context rejection")
    for key, row in holdout["cases"].items():
        Draft202012Validator(SCHEMA).validate(json.loads(row["value"]))
        require(row["finish"] == ("tool_calls" if "/tool/" in key else "stop"), "unexpected finish reason")
    summary["runs"][name] = {"image_id": image, "source_revision": report["source_revision"],
                             "binary_sha256": report["riley_binary_sha256"], "checkpoint_sha256": CHECKPOINT,
                             "api_checks": report["api_checks"], "holdout": holdout,
                             "scheduler_context_rejection_tokens": 8211,
                             "shutdown": report["shutdown_metrics"], "reservation": reservation,
                             "raw_report_sha256": report["gpu_receipt_sha256"]}

stock, alternative = (summary["runs"][name]["holdout"]["cases"] for name in ("stock", "jemalloc"))
require(stock == alternative, "image outputs differ for the C1 holdout")
require(summary["runs"]["stock"]["reservation"] == summary["runs"]["jemalloc"]["reservation"],
        "GPU reservation owners changed between the runs")
summary["cross_image_c1_matches"] = len(stock)
print(json.dumps(summary, ensure_ascii=False, indent=2))
