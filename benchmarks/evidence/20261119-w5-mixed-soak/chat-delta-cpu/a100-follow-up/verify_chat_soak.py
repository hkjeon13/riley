"""Recompute the chat SSE candidate against the unmodified image for each finished mode."""

from argparse import ArgumentParser
import json
from pathlib import Path

from verify_a100_candidate import gpu_evidence, require, validate_soak
from verify_chat_accuracy import EXPECTED_IMAGE, verify_gpu7
from verify_framing_soak import PAIR, verify


def verify_chat(stage, original, modes):
    result = verify(stage, original, modes)
    result["scope"] = "A100 6 chat SSE diagnostic, paired against the unmodified image for each mode."
    image = result["image"]
    identity = {
        "image_id": image["candidate_image_id"],
        "riley_binary_sha256": image["candidate_binary_sha256"],
        "cpu_report_sha256": image["candidate_cpu_report_sha256"],
        "source_archive_sha256": image["candidate_source_archive_sha256"],
    }
    require(identity == EXPECTED_IMAGE, "the completed soak used a different chat SSE package")
    identity["source_revision"] = image["base_revision"]
    verified = {}
    for mode in modes:
        folder = stage / mode
        receipt = gpu_evidence(folder, identity, result["receipts"][mode]["exit"])
        validate_soak(folder, receipt, mode)
        verify_gpu7(folder)
        verified[mode] = receipt["verified_soak"]
    result["verified_soaks"] = verified
    result["same_gpu7_reservation_in_verified_modes"] = True
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    modes = PAIR if args.paired else PAIR[:1]
    print(json.dumps(verify_chat(args.stage, args.original, modes), indent=2, allow_nan=False))
