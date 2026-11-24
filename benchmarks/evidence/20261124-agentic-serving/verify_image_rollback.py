"""Bind drain, interrupted stream and recovery to three real container image receipts."""

import argparse
import json
from pathlib import Path

from verify_operations import summarize


def inspect(root):
    folder = root / "retry"
    workflow = summarize(folder)
    expected = json.loads((root / "image-rollback-registration.json").read_text())
    stable = (root / "base-image.id").read_text().strip()
    candidate = (root / "candidate-image.id").read_text().strip()
    digest = (root / "candidate-binary.sha256").read_text().split()[0]
    if (
        stable != expected["base"]
        or candidate == stable
        or (root / "inspected-candidate.id").read_text().strip() != candidate
    ):
        raise ValueError("ambiguous image identity")
    generations = {}
    for label in ("old", "survivor", "replacement"):
        d = folder / label
        image = (d / "image.id").read_text().strip()
        binary = (d / "binary.sha256").read_text().split()[0]
        mounts = set((d / "mount-destinations").read_text().split())
        args = (d / "arguments.txt").read_text().splitlines()
        expected_image = candidate if label == "survivor" else stable
        must_see = label == "survivor"
        if (
            image != expected_image
            or mounts != {"/models"}
            or ("--request-timeout-ms" in args) != must_see
        ):
            raise ValueError("generation or mounted runtime differs")
        if must_see and (
            args[args.index("--request-timeout-ms") + 1] != "120000" or binary != digest
        ):
            raise ValueError("incorrect candidate executable")
        generations[label] = {
            "image": image,
            "binary": binary,
            "mounts": sorted(mounts),
        }
    if (
        generations["old"]["binary"] == digest
        or generations["old"] != generations["replacement"]
    ):
        raise ValueError("rollback did not restore the previous binary")
    return {
        "scope": "single-host local Docker image rollback",
        "generations": generations,
        **workflow,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(inspect(p.parse_args().root), indent=2))
