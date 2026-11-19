"""Bind the chat SSE image's W5 pair to its conditional W4 accuracy run."""

from argparse import ArgumentParser
import json
from pathlib import Path

from accuracy_gate import gate, verify
from verify_a100_candidate import digest, gpu_evidence, read, require, validate_soak


EXPECTED_IMAGE = {
    "image_id": "sha256:06eed36dcc604e942901422c90d8a94d26c8144eb09d3b515a0f78dfef2d0c51",
    "riley_binary_sha256": "d62d19acf4984b7a31cef4fd10d3c20e8a3e2a4cb23ec4d1daa1a4c9e00a9a23",
    "cpu_report_sha256": "2b66df3f60168dd6cbbc42e0b1831c9a9d046c0f46f7ead32c09e540ff4eba0c",
    "source_archive_sha256": "828d81b99b228d3983c4674b07aa48eeacee2f20e4146af17fa1f091b1b819ec",
}


def verify_gpu7(stage):
    before, after = (read(stage / f"idle-{side}-7.json") for side in ("before", "after"))
    require(before.keys() == after.keys()
            and all(before[key] == after[key] for key in before if key != "updated_ns")
            and after["physical_index"] == 7 and after["pid"] == 2431237 and after["phase"] == "idle"
            and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3
            and not after["other_processes"], "GPU 7's original reservation changed")


def decision(candidate):
    result = gate(candidate)
    require(all(result["image"][field] == value for field, value in EXPECTED_IMAGE.items()),
            "the soak used a different chat SSE image or package")
    for mode, outcome in result["soak_modes"].items():
        arm = candidate / mode
        report = gpu_evidence(arm, result["image"], int(outcome["status"] == "failed"))
        validate_soak(arm, report, mode)
        require(report["gpu_receipt_sha256"] == outcome["report_sha256"], f"{mode} changed during verification")
        verify_gpu7(arm)
    result["same_gpu7_reservation_in_both_modes"] = True
    result["paired_checksums_sha256"] = digest(candidate / "result.sha256")
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--accuracy", type=Path)
    args = parser.parse_args()
    result = decision(args.candidate)
    if args.accuracy is not None:
        require(result == read(args.accuracy / "decision.json"), "the saved gate decision differs from replay")
        verify_gpu7(args.accuracy)
        result["accuracy_result"] = verify(args.accuracy, args.candidate)
        result["accuracy_gpu7_reservation_restored"] = True
    print(json.dumps(result, indent=2, allow_nan=False))
