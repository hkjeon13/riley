#!/usr/bin/env python3
"""Write the in-container completion receipt for one reproducible build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
GATE_ID = "pr16-release-build-reproducibility-v1"
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")

OUTPUT_PATHS = {
    "binary": "artifacts/rustinfer",
    "profile_binary": "artifacts/rustinfer-profile",
    "bundle": "artifacts/rustinfer.tar.gz",
    "native_manifest": "artifacts/native-dependencies.txt",
    "toolchain_log": "logs/toolchain.txt",
    "preflight_log": "logs/preflight.log",
    "cargo_build_log": "logs/cargo-build.log",
    "profile_build_log": "logs/profile-build.log",
    "bundle_build_log": "logs/bundle-build.log",
    "bundle_verify_log": "logs/bundle-verify.log",
}


class CompletionReceiptError(RuntimeError):
    """The build cannot produce a trustworthy completion receipt."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise CompletionReceiptError(f"{label} must be a regular file")
    return path


def _load_container_id(path: Path, build_image_id: str) -> str:
    _regular(path, "pre-start Docker inspect receipt")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CompletionReceiptError(
                    f"pre-start Docker inspect receipt repeats JSON field {key!r}"
                )
            result[key] = value
        return result

    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompletionReceiptError("pre-start Docker inspect receipt is invalid") from error
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        raise CompletionReceiptError("pre-start Docker inspect receipt has invalid shape")
    container_id = document[0].get("Id")
    if not isinstance(container_id, str) or CONTAINER_ID_PATTERN.fullmatch(container_id) is None:
        raise CompletionReceiptError("pre-start Docker inspect receipt has invalid container ID")
    if document[0].get("Image") != build_image_id:
        raise CompletionReceiptError("pre-start Docker inspect receipt has a different image ID")
    return container_id


def write_completion_receipt(
    *,
    build_id: str,
    source_revision: str,
    source_archive_sha256: str,
    source_date_epoch: int,
    build_image_id: str,
    container_inspect: Path,
    outputs: dict[str, Path],
    output: Path,
) -> dict[str, Any]:
    if build_id not in {"A", "B"}:
        raise CompletionReceiptError("build ID must be A or B")
    if REVISION_PATTERN.fullmatch(source_revision) is None:
        raise CompletionReceiptError("source revision must be a full lowercase Git SHA")
    if SHA256_PATTERN.fullmatch(source_archive_sha256) is None:
        raise CompletionReceiptError("source archive SHA-256 is invalid")
    if IMAGE_ID_PATTERN.fullmatch(build_image_id) is None:
        raise CompletionReceiptError("builder image ID is invalid")
    if not 0 <= source_date_epoch <= 0xFFFFFFFF:
        raise CompletionReceiptError("SOURCE_DATE_EPOCH is out of range")
    if set(outputs) != set(OUTPUT_PATHS):
        raise CompletionReceiptError("completion output inventory differs from the closed contract")

    container_id = _load_container_id(container_inspect, build_image_id)
    output_records: dict[str, dict[str, Any]] = {}
    for name, relative_path in OUTPUT_PATHS.items():
        path = _regular(outputs[name], f"completion output {name}")
        output_records[name] = {
            "path": relative_path,
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": "completed",
        "build_id": build_id,
        "container_id": container_id,
        "build_image_id": build_image_id,
        "source": {
            "revision": source_revision,
            "archive_sha256": source_archive_sha256,
            "source_date_epoch": source_date_epoch,
        },
        "cargo_commands": {
            "release": [
                "cargo",
                "build",
                "--locked",
                "--offline",
                "--release",
                "--features",
                "cuda,server",
            ],
            "profile": [
                "cargo",
                "build",
                "--locked",
                "--offline",
                "--release",
                "--features",
                "bench,cuda",
                "--bin",
                "rustinfer-profile",
            ],
        },
        "outputs": output_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(_canonical_json(receipt))
    output.chmod(0o644)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", choices=("A", "B"), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--build-image-id", required=True)
    parser.add_argument("--container-inspect", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--profile-binary", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--toolchain-log", type=Path, required=True)
    parser.add_argument("--preflight-log", type=Path, required=True)
    parser.add_argument("--cargo-build-log", type=Path, required=True)
    parser.add_argument("--profile-build-log", type=Path, required=True)
    parser.add_argument("--bundle-build-log", type=Path, required=True)
    parser.add_argument("--bundle-verify-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        write_completion_receipt(
            build_id=args.build_id,
            source_revision=args.source_revision,
            source_archive_sha256=args.source_archive_sha256,
            source_date_epoch=args.source_date_epoch,
            build_image_id=args.build_image_id,
            container_inspect=args.container_inspect,
            outputs={
                "binary": args.binary,
                "profile_binary": args.profile_binary,
                "bundle": args.bundle,
                "native_manifest": args.native_manifest,
                "toolchain_log": args.toolchain_log,
                "preflight_log": args.preflight_log,
                "cargo_build_log": args.cargo_build_log,
                "profile_build_log": args.profile_build_log,
                "bundle_build_log": args.bundle_build_log,
                "bundle_verify_log": args.bundle_verify_log,
            },
            output=args.output,
        )
    except (OSError, CompletionReceiptError) as error:
        print(f"reproducible build completion failed: {error}", file=os.sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
