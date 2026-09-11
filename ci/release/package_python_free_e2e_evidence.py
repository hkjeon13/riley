#!/usr/bin/env python3
"""Create the closed deterministic Python-free E2E raw evidence v2 tar."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import stat
import tarfile
from pathlib import Path
from typing import Sequence


MAX_FIXED_MEMBER_BYTES = 512 * 1024 * 1024
MAX_MODEL_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
SAFE_MODEL_PATH = re.compile(r"^[A-Za-z0-9._/+@=-]+$")

PAYLOAD_ARGUMENTS = {
    "cancellation-request.raw": "cancellation_request",
    "cancellation-response-prefix.raw": "cancellation_response_prefix",
    "container-first-post.json": "container_first_post",
    "container-first-pre.json": "container_first_pre",
    "container-first-runtime.json": "container_first_runtime",
    "container-second-post.json": "container_second_post",
    "container-second-pre.json": "container_second_pre",
    "container-second-runtime.json": "container_second_runtime",
    "correctness-golden.json": "correctness_golden",
    "http-greedy-stream.raw": "http_greedy_stream",
    "http-greedy.raw": "http_greedy",
    "http-metrics-after.raw": "http_metrics_after",
    "http-metrics-before.raw": "http_metrics_before",
    "http-models.raw": "http_models",
    "http-readyz.raw": "http_readyz",
    "http-sampling-first.raw": "http_sampling_first",
    "http-sampling-second.raw": "http_sampling_second",
    "image-binary": "image_binary",
    "image-inspect.json": "image_inspect",
    "image-ldd.txt": "image_ldd",
    "image-native-dependencies.txt": "image_native_dependencies",
    "image-python-scan.txt": "image_python_scan",
    "image-readelf.txt": "image_readelf",
    "model-SHA256SUMS": "model_manifest",
    "process-first-pre.txt": "process_first_pre",
    "process-first-runtime.txt": "process_first_runtime",
    "process-second-pre.txt": "process_second_pre",
    "process-second-runtime.txt": "process_second_runtime",
    "raw-evidence.json": "raw_evidence",
    "repeat-shutdown-metrics.json": "repeat_shutdown_metrics",
    "request-greedy-stream.json": "request_greedy_stream",
    "request-greedy.json": "request_greedy",
    "request-sampling.json": "request_sampling",
    "shutdown-metrics.json": "shutdown_metrics",
}


def _regular_file(path: Path, label: str, maximum: int) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label}: must be a regular file, not a link or device")
    if metadata.st_size <= 0 or metadata.st_size > maximum:
        raise ValueError(f"{label}: invalid evidence size")
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_inputs(model_dir: Path) -> dict[str, Path]:
    root = model_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model directory must be a directory")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"model/{relative}: links and special files are forbidden")
        if SAFE_MODEL_PATH.fullmatch(relative) is None or len(f"model/{relative}".encode("ascii")) > 100:
            raise ValueError(f"model/{relative}: path is not safe USTAR ASCII")
        _regular_file(path, f"model/{relative}", MAX_MODEL_MEMBER_BYTES)
        result[f"model/{relative}"] = path
    if not result:
        raise ValueError("model directory contains no regular files")
    return result


def _copy_member(archive: tarfile.TarFile, name: str, path: Path, size: int) -> None:
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o755 if name == "image-binary" else 0o644
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    with path.open("rb") as source:
        archive.addfile(member, source)


def package(output: Path, inputs: dict[str, Path], model_dir: Path) -> None:
    """Write a create-only USTAR containing every raw observation and model byte."""

    if set(inputs) != set(PAYLOAD_ARGUMENTS):
        raise ValueError("exact fixed raw-evidence v2 payload inventory is required")
    sources = dict(inputs)
    sources.update(_model_inputs(model_dir))
    sizes: dict[str, int] = {}
    total = 0
    for name, path in sources.items():
        maximum = MAX_MODEL_MEMBER_BYTES if name.startswith("model/") else MAX_FIXED_MEMBER_BYTES
        metadata = _regular_file(path, name, maximum)
        sizes[name] = metadata.st_size
        total += metadata.st_size
    if total > MAX_TOTAL_BYTES:
        raise ValueError("raw evidence payloads exceed the total size bound")

    checksums = b"".join(
        f"{_sha256(sources[name])}  {name}\n".encode("ascii") for name in sorted(sources)
    )
    created = False
    try:
        with tarfile.open(output, "x:", format=tarfile.USTAR_FORMAT) as archive:
            created = True
            for name in sorted({*sources, "SHA256SUMS"}):
                if name == "SHA256SUMS":
                    member = tarfile.TarInfo(name)
                    member.size = len(checksums)
                    member.mode = 0o644
                    member.uid = 0
                    member.gid = 0
                    member.uname = ""
                    member.gname = ""
                    member.mtime = 0
                    archive.addfile(member, io.BytesIO(checksums))
                else:
                    _copy_member(archive, name, sources[name], sizes[name])
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
    parser.add_argument("--model-dir", required=True, type=Path)
    for _name, attribute in sorted(PAYLOAD_ARGUMENTS.items()):
        option = "--" + attribute.replace("_", "-")
        parser.add_argument(option, dest=attribute, required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = {name: getattr(args, attribute) for name, attribute in PAYLOAD_ARGUMENTS.items()}
    try:
        package(args.output, inputs, args.model_dir)
    except (FileExistsError, OSError, ValueError, tarfile.TarError) as error:
        print(f"cannot package Python-free E2E evidence: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
