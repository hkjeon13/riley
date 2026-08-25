#!/usr/bin/env python3
"""Create the deterministic, non-circular Python-free E2E raw evidence tar."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import stat
import tarfile
from pathlib import Path
from typing import Sequence


MAX_MEMBER_BYTES = 16 * 1024 * 1024
PAYLOAD_ARGUMENTS = {
    "correctness-golden.json": "correctness_golden",
    "model-SHA256SUMS": "model_manifest",
    "raw-evidence.json": "raw_evidence",
    "repeat-shutdown-metrics.json": "repeat_shutdown_metrics",
    "shutdown-metrics.json": "shutdown_metrics",
}


def _read_regular(path: Path, label: str) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label}: must be a regular file, not a link or device")
    if metadata.st_size <= 0 or metadata.st_size > MAX_MEMBER_BYTES:
        raise ValueError(f"{label}: invalid evidence size")
    return path.read_bytes()


def package(output: Path, inputs: dict[str, Path]) -> None:
    if set(inputs) != set(PAYLOAD_ARGUMENTS):
        raise ValueError("exact five-file payload inventory is required")
    payloads = {
        name: _read_regular(inputs[name], name)
        for name in sorted(PAYLOAD_ARGUMENTS)
    }
    payloads["SHA256SUMS"] = b"".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n".encode("ascii")
        for name in sorted(PAYLOAD_ARGUMENTS)
    )
    created = False
    try:
        with tarfile.open(output, "x:", format=tarfile.USTAR_FORMAT) as archive:
            created = True
            for name in sorted(payloads):
                contents = payloads[name]
                member = tarfile.TarInfo(name)
                member.size = len(contents)
                member.mode = 0o644
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.mtime = 0
                archive.addfile(member, io.BytesIO(contents))
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if created:
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-evidence", required=True, type=Path)
    parser.add_argument("--correctness-golden", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--shutdown-metrics", required=True, type=Path)
    parser.add_argument("--repeat-shutdown-metrics", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = {
        name: getattr(args, attribute)
        for name, attribute in PAYLOAD_ARGUMENTS.items()
    }
    try:
        package(args.output, inputs)
    except (FileExistsError, OSError, ValueError, tarfile.TarError) as error:
        print(f"cannot package Python-free E2E evidence: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
