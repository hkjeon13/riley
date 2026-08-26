#!/usr/bin/env python3
"""Verify two independent release builds and their selected final artifacts.

The checker is intentionally standard-library-only.  It treats both evidence
archives as hostile input, validates their closed inventories without
extracting archive paths, re-verifies every release bundle, and performs byte
comparisons instead of trusting producer-reported success fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from release_common import (
    CUDA_ARCHITECTURES,
    CUDA_TOOLKIT,
    ReleaseContractError,
    canonical_json_bytes,
    load_json_object,
    parse_native_manifest,
    validate_binary,
)
from verify_release_bundle import verify_bundle

SCHEMA_VERSION = 1
GATE_ID = "pr16-release-build-reproducibility-v1"
RUST_TOOLCHAIN = "1.85.0"
RUSTUP_TOOLCHAIN = "1.85.0-x86_64-unknown-linux-gnu"
NVCC_VERSION = "12.8.93"
PLATFORM = "linux/amd64"
BUILDER_PATH = (
    "/usr/local/cargo/bin:/usr/local/cuda/bin:/usr/local/cuda/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
CONTAINER_COMMAND = (
    'test -z "$(find /workspace -mindepth 1 -print -quit)"; '
    "tar --extract --file /input/source.tar --directory /workspace; "
    "cd /workspace; "
    "exec /bin/bash ci/release/run_reproducible_build_once.sh"
)
PROXY_ENVIRONMENT = {
    "ALL_PROXY": "",
    "FTP_PROXY": "",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "NO_PROXY": "",
    "all_proxy": "",
    "ftp_proxy": "",
    "http_proxy": "",
    "https_proxy": "",
    "no_proxy": "",
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
CHECKSUM_LINE_PATTERN = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_./+-]+)")
DOCKER_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{9}Z"
)

MAX_EVIDENCE_MEMBERS = 23
MAX_EVIDENCE_MEMBER_SIZE = 2 * 1024 * 1024 * 1024
MAX_EVIDENCE_TOTAL_SIZE = 6 * 1024 * 1024 * 1024
MAX_EVIDENCE_ARCHIVE_SIZE = MAX_EVIDENCE_TOTAL_SIZE + 1024 * 1024
MAX_BINARY_SIZE = 512 * 1024 * 1024
MAX_METADATA_SIZE = 1024 * 1024
MAX_SOURCE_MEMBERS = 20_000
MAX_SOURCE_MEMBER_SIZE = 2 * 1024 * 1024 * 1024
MAX_SOURCE_TOTAL_SIZE = 8 * 1024 * 1024 * 1024
MAX_SOURCE_ARCHIVE_SIZE = MAX_SOURCE_TOTAL_SIZE + 16 * 1024 * 1024
TAR_BLOCK_SIZE = 512
TAR_RECORD_SIZE = 10_240

RELATIVE_DIRECTORIES = {
    "bin",
    "bundle",
    "logs",
    "manifest",
}
RELATIVE_FILES = {
    "SHA256SUMS",
    "bin/rustinfer",
    "bin/rustinfer-profile",
    "bundle/rustinfer.tar.gz",
    "logs/bundle-build.log",
    "logs/bundle-verify.log",
    "logs/build-completion.json",
    "logs/builder-image-inspect.json",
    "logs/cargo-build.log",
    "logs/container-inspect.json",
    "logs/container-inspect-post.json",
    "logs/container-invocation.txt",
    "logs/preflight.log",
    "logs/profile-build.log",
    "logs/toolchain.txt",
    "manifest/build.json",
    "manifest/native-dependencies.txt",
    "source.tar",
}

COMPLETION_OUTPUTS = {
    "binary": ("artifacts/rustinfer", "bin/rustinfer"),
    "profile_binary": ("artifacts/rustinfer-profile", "bin/rustinfer-profile"),
    "bundle": ("artifacts/rustinfer.tar.gz", "bundle/rustinfer.tar.gz"),
    "native_manifest": (
        "artifacts/native-dependencies.txt",
        "manifest/native-dependencies.txt",
    ),
    "toolchain_log": ("logs/toolchain.txt", "logs/toolchain.txt"),
    "preflight_log": ("logs/preflight.log", "logs/preflight.log"),
    "cargo_build_log": ("logs/cargo-build.log", "logs/cargo-build.log"),
    "profile_build_log": ("logs/profile-build.log", "logs/profile-build.log"),
    "bundle_build_log": ("logs/bundle-build.log", "logs/bundle-build.log"),
    "bundle_verify_log": ("logs/bundle-verify.log", "logs/bundle-verify.log"),
}


@dataclass(frozen=True)
class Evidence:
    build_id: str
    archive: Path
    root: str
    files: dict[str, Path]
    manifest: dict[str, Any]
    source_date_epoch: int
    container_id: str
    workspace_volume: str
    workspace_source: str
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class ContainerDetails:
    container_id: str
    workspace_volume: str
    workspace_source: str
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class BundleDetails:
    binary: bytes
    native_manifest: bytes
    source_revision: str
    source_date_epoch: int


def _fail(message: str) -> None:
    raise ReleaseContractError(message)


def _closed_object(value: Any, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    actual = set(value)
    if actual != keys:
        _fail(
            f"{label} fields differ from the closed contract; "
            f"missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} must be a string")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{label} must be an integer")
    return value


def _strict_json_value(contents: bytes, label: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        _fail(f"{label} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"{label} is not valid UTF-8 JSON") from error
    return value


def _strict_json(contents: bytes, label: str) -> dict[str, Any]:
    value = _strict_json_value(contents, label)
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _regular_path(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReleaseContractError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular file, not a link or device: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_equal(left: Path, right: Path, label: str) -> None:
    if left.stat().st_size != right.stat().st_size:
        _fail(f"{label} is not byte-exact: file sizes differ")
    with left.open("rb") as left_source, right.open("rb") as right_source:
        while True:
            left_chunk = left_source.read(1024 * 1024)
            right_chunk = right_source.read(1024 * 1024)
            if left_chunk != right_chunk:
                _fail(f"{label} is not byte-exact")
            if not left_chunk:
                return


def _tar_data_end(member: tarfile.TarInfo) -> int:
    padded_size = ((member.size + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE
    return member.offset_data + padded_size


def _validate_tar_end(path: Path, data_end: int, label: str) -> None:
    canonical_size = (
        (data_end + 2 * TAR_BLOCK_SIZE + TAR_RECORD_SIZE - 1) // TAR_RECORD_SIZE
    ) * TAR_RECORD_SIZE
    if path.stat().st_size != canonical_size:
        _fail(f"{label} has non-canonical end-of-archive padding or trailing records")
    with path.open("rb") as raw:
        raw.seek(data_end)
        padding = raw.read(canonical_size - data_end)
    if len(padding) != canonical_size - data_end or any(padding):
        _fail(f"{label} has non-canonical end-of-archive padding or trailing records")


def _safe_member_path(name: str, label: str) -> PurePosixPath:
    if not name or name.startswith("/") or "\\" in name or "//" in name:
        _fail(f"unsafe {label} member path: {name!r}")
    path = PurePosixPath(name)
    if any(part in ("", ".", "..") for part in path.parts):
        _fail(f"unsafe {label} member path: {name!r}")
    return path


def _copy_member(source: BinaryIO, destination: Path, size: int, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    remaining = size
    with destination.open("xb") as output:
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"truncated {label} member")
            output.write(chunk)
            remaining -= len(chunk)
        if source.read(1):
            _fail(f"{label} member exceeds its declared size")


def _parse_checksums(contents: bytes, expected_paths: set[str]) -> dict[str, str]:
    try:
        text = contents.decode("ascii")
    except UnicodeDecodeError as error:
        raise ReleaseContractError("evidence SHA256SUMS is not ASCII") from error
    if not text.endswith("\n"):
        _fail("evidence SHA256SUMS must end with a newline")
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = CHECKSUM_LINE_PATTERN.fullmatch(line)
        if match is None:
            _fail(f"invalid evidence SHA256SUMS line: {line!r}")
        digest, path = match.groups()
        _safe_member_path(path, "checksum")
        if path in result:
            _fail(f"duplicate evidence SHA256SUMS path: {path}")
        result[path] = digest
    if set(result) != expected_paths:
        _fail(
            "evidence SHA256SUMS path set mismatch; "
            f"missing={sorted(expected_paths - set(result))}, "
            f"extra={sorted(set(result) - expected_paths)}"
        )
    if list(result) != sorted(result):
        _fail("evidence SHA256SUMS paths must be bytewise sorted")
    return result


def expected_commands(source_revision: str, source_date_epoch: int) -> dict[str, list[str]]:
    epoch = str(source_date_epoch)
    return {
        "preflight": [
            "python3",
            "ci/release/run_release_python.py",
            "ci/release/check_release_preflight.py",
            "--source-revision",
            source_revision,
            "--source-date-epoch",
            epoch,
        ],
        "build": [
            "cargo",
            "build",
            "--locked",
            "--offline",
            "--release",
            "--features",
            "cuda,server",
        ],
        "profile_build": [
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
        "bundle": [
            "python3",
            "ci/release/run_release_python.py",
            "ci/release/build_release_bundle.py",
            "--binary",
            "target/release/rustinfer",
            "--output",
            "/workspace/release/rustinfer.tar.gz",
            "--source-revision",
            source_revision,
            "--source-date-epoch",
            epoch,
        ],
        "verify": [
            "python3",
            "ci/release/run_release_python.py",
            "ci/release/verify_release_bundle.py",
            "/workspace/release/rustinfer.tar.gz",
        ],
    }


def expected_environment(build_image_id: str) -> dict[str, Any]:
    return {
        "build_image_id": build_image_id,
        "cargo_cache_mount": "none",
        "container_rootfs": "ephemeral-overlay",
        "cuda_architectures": CUDA_ARCHITECTURES,
        "cuda_toolkit": CUDA_TOOLKIT,
        "gpu_passthrough": False,
        "network": "none",
        "nvcc_version": NVCC_VERSION,
        "platform": PLATFORM,
        "rust_toolchain": RUST_TOOLCHAIN,
        "source_mount": "read-only",
        "workspace_mount": "anonymous-volume",
    }


def invocation_bytes(build_id: str, build_image_id: str) -> bytes:
    return (
        "schema_version=1\n"
        f"build_id={build_id}\n"
        f"build_image_id={build_image_id}\n"
        "platform=linux/amd64\n"
        "network=none\n"
        "container_rootfs=ephemeral-overlay\n"
        "source_mount=read-only\n"
        "workspace_mount=anonymous-volume\n"
        "cargo_cache_mount=none\n"
        "gpu_passthrough=false\n"
    ).encode("ascii")


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_build_id: str,
    source_revision: str,
    source_archive_sha256: str,
    source_date_epoch: int,
    build_image_id: str,
    file_paths: dict[str, Path],
) -> None:
    _closed_object(
        manifest,
        "evidence build manifest",
        {"schema_version", "gate_id", "status", "build_id", "source", "environment", "commands", "artifacts"},
    )
    if _integer(manifest["schema_version"], "manifest schema_version") != SCHEMA_VERSION:
        _fail("unsupported evidence build manifest schema")
    if _string(manifest["gate_id"], "manifest gate_id") != GATE_ID:
        _fail("evidence build manifest gate_id is not the reviewed gate")
    if _string(manifest["status"], "manifest status") != "passed":
        _fail("evidence build manifest status is not passed")
    if _string(manifest["build_id"], "manifest build_id") != expected_build_id:
        _fail("evidence build_id does not match its A/B position")

    source = _closed_object(
        manifest["source"],
        "manifest source",
        {"archive_sha256", "revision", "source_date_epoch", "workspace"},
    )
    if source != {
        "archive_sha256": source_archive_sha256,
        "revision": source_revision,
        "source_date_epoch": source_date_epoch,
        "workspace": "fresh-git-archive",
    }:
        _fail("evidence source provenance differs from the canonical source contract")

    environment = _closed_object(
        manifest["environment"],
        "manifest environment",
        set(expected_environment(build_image_id)),
    )
    if environment != expected_environment(build_image_id):
        _fail("evidence build environment differs from the reviewed immutable contract")

    commands = _closed_object(
        manifest["commands"],
        "manifest commands",
        {"build", "profile_build", "bundle", "preflight", "verify"},
    )
    if commands != expected_commands(source_revision, source_date_epoch):
        _fail("evidence commands differ from the locked/offline release build contract")

    artifacts = _closed_object(
        manifest["artifacts"],
        "manifest artifacts",
        {
            "binary_sha256",
            "binary_size",
            "profile_binary_sha256",
            "profile_binary_size",
            "bundle_sha256",
            "bundle_size",
            "native_manifest_sha256",
            "native_manifest_size",
        },
    )
    expected_artifacts: dict[str, Any] = {}
    for key, relative in (
        ("binary", "bin/rustinfer"),
        ("profile_binary", "bin/rustinfer-profile"),
        ("bundle", "bundle/rustinfer.tar.gz"),
        ("native_manifest", "manifest/native-dependencies.txt"),
    ):
        path = file_paths[relative]
        expected_artifacts[f"{key}_sha256"] = _sha256_file(path)
        expected_artifacts[f"{key}_size"] = path.stat().st_size
    if artifacts != expected_artifacts:
        _fail("producer-reported artifact hashes or sizes do not match raw evidence bytes")


def _validate_toolchain_log(contents: bytes) -> None:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseContractError("toolchain log is not UTF-8") from error
    if not text.endswith("\n"):
        _fail("toolchain log must end with a newline")
    lines = text.splitlines()
    if len(lines) != 3:
        _fail("toolchain log must contain exactly rustc, cargo, and nvcc versions")
    if re.fullmatch(r"rustc_version=rustc 1\.85\.0 \(4d91de4e4 2025-02-17\)(?: \([^\n]+\))?", lines[0]) is None:
        _fail("toolchain log does not prove the reviewed rustc 1.85.0 build")
    if re.fullmatch(r"cargo_version=cargo 1\.85\.0(?: \([^\n]+\))?", lines[1]) is None:
        _fail("toolchain log does not prove the reviewed cargo 1.85.0 build")
    if lines[2] != "nvcc_version=Cuda compilation tools, release 12.8, V12.8.93":
        _fail("toolchain log does not prove the reviewed nvcc 12.8.93 build")


def _validate_logs(evidence: Evidence, build_image_id: str) -> None:
    def read(relative: str, maximum: int = 8 * 1024 * 1024) -> bytes:
        path = evidence.files[relative]
        if path.stat().st_size > maximum:
            _fail(f"{relative} exceeds its review size bound")
        return path.read_bytes()

    invocation = read("logs/container-invocation.txt", 4096)
    if invocation != invocation_bytes(evidence.build_id, build_image_id):
        _fail("container invocation evidence differs from the reviewed isolation contract")
    _validate_toolchain_log(read("logs/toolchain.txt", 4096))
    if read("logs/preflight.log", 4096) != b"release preflight passed\n":
        _fail("release preflight log is not the exact success output")
    if read("logs/bundle-build.log", 4096) != b"/workspace/release/rustinfer.tar.gz\n":
        _fail("release bundle build log is not the exact success output")
    if read("logs/bundle-verify.log", 4096) != b"verified /workspace/release/rustinfer.tar.gz\n":
        _fail("release bundle verification log is not the exact success output")
    for relative, label in (
        ("logs/cargo-build.log", "production Cargo build"),
        ("logs/profile-build.log", "profile Cargo build"),
    ):
        cargo_log = read(relative)
        try:
            cargo_text = cargo_log.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReleaseContractError(f"{label} log is not UTF-8") from error
        if "Compiling rustinfer-server" not in cargo_text:
            _fail(f"{label} log does not contain the rustinfer-server compilation")
        if re.search(r"Finished [`']release[`'] profile \[optimized\]", cargo_text) is None:
            _fail(f"{label} log does not contain a completed optimized release profile")
        lowered = cargo_text.casefold()
        if any(
            marker in lowered
            for marker in ("updating crates.io", "downloading crates", "http://", "https://")
        ):
            _fail(f"{label} log contains network access markers despite --offline")


def _parse_environment(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _fail(f"{label} is invalid")
    parsed: dict[str, str] = {}
    for item in value:
        key, separator, entry = item.partition("=")
        if not separator or not key or key in parsed:
            _fail(f"{label} contains an invalid or duplicate entry")
        parsed[key] = entry
    return parsed


def _validate_builder_image_inspect(contents: bytes, build_image_id: str) -> dict[str, str]:
    if len(contents) > MAX_METADATA_SIZE:
        _fail("Docker builder image inspect evidence exceeds its size bound")
    document = _strict_json_value(contents, "Docker builder image inspect evidence")
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        _fail("Docker builder image inspect evidence must contain exactly one image object")
    image = document[0]
    if image.get("Id") != build_image_id:
        _fail("Docker builder inspect ID differs from the pinned build image")
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        _fail("Docker builder image platform must be linux/amd64")
    config = image.get("Config")
    if not isinstance(config, dict):
        _fail("Docker builder image Config is missing")
    if config.get("WorkingDir") != "/workspace":
        _fail("Docker builder image working directory is not /workspace")
    if config.get("Volumes") not in (None, {}):
        _fail("Docker builder image declares an unreviewed persistent volume")
    environment = _parse_environment(config.get("Env"), "Docker builder image environment")
    required = {
        "PATH": BUILDER_PATH,
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs",
        "DEBIAN_FRONTEND": "noninteractive",
        "CARGO_HOME": "/usr/local/cargo",
        "RUSTUP_HOME": "/usr/local/rustup",
        "RUSTUP_TOOLCHAIN": RUSTUP_TOOLCHAIN,
        "CUDA_HOME": "/usr/local/cuda",
        "CUDAToolkit_ROOT": "/usr/local/cuda",
        "RUSTINFER_CUDA_ARCHITECTURES": "89",
        "CARGO_INCREMENTAL": "0",
        "CARGO_NET_OFFLINE": "true",
        "CUDA_VERSION": CUDA_TOOLKIT,
    }
    for key, value in required.items():
        if environment.get(key) != value:
            _fail(f"Docker builder image environment differs for {key}")
    allowed_names = set(required) | {"NVARCH", "NCCL_VERSION"}
    unreviewed = {
        key
        for key in environment
        if key not in allowed_names
        and not key.startswith("NV_")
        and not key.startswith("NVIDIA_")
    }
    if unreviewed:
        _fail(
            "Docker builder image contains an unreviewed environment: "
            + ", ".join(sorted(unreviewed))
        )
    return environment


def _validate_container_inspect(
    contents: bytes,
    *,
    build_id: str,
    build_image_id: str,
    source_revision: str,
    source_archive_sha256: str,
    source_date_epoch: int,
    builder_environment: dict[str, str],
    expected_phase: str = "created",
) -> ContainerDetails:
    if len(contents) > MAX_METADATA_SIZE:
        _fail("Docker container inspect evidence exceeds its size bound")
    document = _strict_json_value(contents, "Docker container inspect evidence")
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        _fail("Docker container inspect evidence must contain exactly one container object")
    container = document[0]
    container_id = container.get("Id")
    if not isinstance(container_id, str) or re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        _fail("Docker inspect container ID is not a full content ID")
    if container.get("Image") != build_image_id:
        _fail("Docker inspect image ID differs from the pinned build image")
    if container.get("Platform") != "linux":
        _fail("Docker inspect platform must be Linux")
    state = container.get("State")
    if not isinstance(state, dict):
        _fail("Docker inspect pre-start State is missing")
    expected_state = {
        "Running": False,
        "Paused": False,
        "Restarting": False,
        "OOMKilled": False,
        "Dead": False,
        "Pid": 0,
        "Error": "",
    }
    if expected_phase == "created":
        expected_state.update(
            {
                "Status": "created",
                "ExitCode": 0,
                "StartedAt": "0001-01-01T00:00:00Z",
                "FinishedAt": "0001-01-01T00:00:00Z",
            }
        )
    elif expected_phase == "exited":
        expected_state.update({"Status": "exited", "ExitCode": 0})
    else:
        _fail("internal Docker receipt phase is invalid")
    for key, value in expected_state.items():
        if state.get(key) != value:
            _fail(
                f"Docker inspect {expected_phase} receipt has invalid State.{key}"
            )
    started_at = state.get("StartedAt")
    finished_at = state.get("FinishedAt")
    if expected_phase == "exited":
        if (
            not isinstance(started_at, str)
            or not isinstance(finished_at, str)
            or DOCKER_TIMESTAMP_PATTERN.fullmatch(started_at) is None
            or DOCKER_TIMESTAMP_PATTERN.fullmatch(finished_at) is None
            or finished_at <= started_at
        ):
            _fail("Docker inspect exited receipt has invalid execution timestamps")
    else:
        assert isinstance(started_at, str) and isinstance(finished_at, str)
    if container.get("RestartCount") != 0:
        _fail("Docker inspect container has a nonzero restart count")
    if container.get("Path") != "/bin/bash" or container.get("Args") != [
        "-ceu",
        CONTAINER_COMMAND,
    ]:
        _fail("Docker daemon execution path/arguments differ from the reviewed build driver")

    config = container.get("Config")
    if not isinstance(config, dict):
        _fail("Docker inspect Config is missing")
    if config.get("Image") != build_image_id:
        _fail("Docker container was not created from the pinned image ID")
    if config.get("Entrypoint") != ["/bin/bash"]:
        _fail("Docker container entrypoint is not the reviewed shell")
    if config.get("User") != "0:0":
        _fail("Docker container user is not the reviewed root UID/GID")
    if config.get("WorkingDir") != "/workspace":
        _fail("Docker container working directory is not the clean workspace")
    healthcheck = config.get("Healthcheck")
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != ["NONE"]:
        _fail("Docker container must disable inherited health checks")
    command = config.get("Cmd")
    if command != ["-ceu", CONTAINER_COMMAND]:
        _fail("Docker container command is not the reviewed one-build driver")

    parsed_environment = _parse_environment(config.get("Env"), "Docker container environment")
    required_environment = {
        "RUSTINFER_REPRO_BUILD_ID": build_id,
        "RUSTINFER_SOURCE_REVISION": source_revision,
        "RUSTINFER_SOURCE_ARCHIVE_SHA256": source_archive_sha256,
        "RUSTINFER_BUILD_IMAGE_ID": build_image_id,
        "SOURCE_DATE_EPOCH": str(source_date_epoch),
    }
    expected_container_environment = dict(builder_environment)
    expected_container_environment.update(required_environment)
    expected_container_environment.update(PROXY_ENVIRONMENT)
    if parsed_environment != expected_container_environment:
        _fail("Docker container environment differs from the pinned image plus provenance contract")

    host = container.get("HostConfig")
    if not isinstance(host, dict):
        _fail("Docker inspect HostConfig is missing")
    if host.get("NetworkMode") != "none":
        _fail("Docker build container did not use network=none")
    if host.get("Runtime") != "runc":
        _fail("Docker build container must explicitly use the non-GPU runc runtime")
    if host.get("Privileged") is not False:
        _fail("Docker build container must not be privileged")
    if host.get("ReadonlyRootfs") is not False:
        _fail("Docker build container rootfs contract is not an ephemeral overlay")
    if host.get("CapDrop") != ["ALL"]:
        _fail("Docker build container must drop all capabilities")
    security_options = host.get("SecurityOpt")
    if security_options not in (["no-new-privileges"], ["no-new-privileges:true"]):
        _fail("Docker build container must enforce no-new-privileges")
    if host.get("PidsLimit") != 4096:
        _fail("Docker build container PID limit differs from the reviewed runner")
    if host.get("Devices") not in (None, []):
        _fail("Docker build container has explicit host devices")
    if host.get("DeviceRequests") not in (None, []):
        _fail("Docker build container has a GPU/device request")
    if host.get("CapAdd") not in (None, []):
        _fail("Docker build container has added Linux capabilities")
    if host.get("CgroupnsMode") != "private":
        _fail("Docker build container cgroup namespace is not private")
    for field in ("PidMode", "IpcMode", "UTSMode", "UsernsMode"):
        if host.get(field) not in (None, "", "private"):
            _fail(f"Docker build container uses an unreviewed host namespace: {field}")
    for field in ("VolumesFrom", "DeviceCgroupRules", "GroupAdd", "Links"):
        if host.get(field) not in (None, []):
            _fail(f"Docker build container has an unreviewed isolation field: {field}")
    if host.get("Sysctls") not in (None, {}):
        _fail("Docker build container has unreviewed sysctl overrides")
    restart_policy = host.get("RestartPolicy")
    if not isinstance(restart_policy, dict) or restart_policy.get("Name") != "no":
        _fail("Docker build container must disable restart policy")
    if restart_policy.get("MaximumRetryCount") != 0:
        _fail("Docker build container restart retry count must be zero")
    if host.get("Binds") not in (None, []):
        _fail("Docker build container has legacy or unreviewed bind mounts")
    if host.get("VolumeDriver") != "":
        _fail("Docker build container must use the default local volume driver")

    configured_mounts = host.get("Mounts")
    if not isinstance(configured_mounts, list) or len(configured_mounts) != 5:
        _fail("Docker HostConfig must contain exactly five reviewed mount requests")
    configured_by_target: dict[str, dict[str, Any]] = {}
    for mount in configured_mounts:
        if not isinstance(mount, dict):
            _fail("Docker HostConfig mount request is invalid")
        target = mount.get("Target")
        if not isinstance(target, str) or target in configured_by_target:
            _fail("Docker HostConfig mount target is invalid or duplicated")
        configured_by_target[target] = mount
    expected_targets = {
        "/input/source.tar",
        "/input/container-inspect.json",
        "/input/builder-image-inspect.json",
        "/evidence",
        "/workspace",
    }
    if set(configured_by_target) != expected_targets:
        _fail("Docker HostConfig mount inventory differs from the reviewed runner")
    for target in (
        "/input/source.tar",
        "/input/container-inspect.json",
        "/input/builder-image-inspect.json",
    ):
        mount = configured_by_target[target]
        if mount.get("Type") != "bind" or mount.get("ReadOnly") is not True:
            _fail(f"Docker HostConfig input is not a read-only bind: {target}")
        source = mount.get("Source")
        if not isinstance(source, str) or not source.startswith("/"):
            _fail(f"Docker HostConfig input source is not absolute: {target}")
    configured_evidence = configured_by_target["/evidence"]
    if (
        configured_evidence.get("Type") != "bind"
        or configured_evidence.get("ReadOnly", False) is not False
    ):
        _fail("Docker HostConfig evidence mount is not the sole writable host bind")
    configured_workspace = configured_by_target["/workspace"]
    if (
        configured_workspace.get("Type") != "volume"
        or configured_workspace.get("ReadOnly", False) is not False
        or configured_workspace.get("Source") not in (None, "")
    ):
        _fail("Docker HostConfig workspace request is not an anonymous volume")
    volume_options = configured_workspace.get("VolumeOptions")
    if not isinstance(volume_options, dict) or set(volume_options) != {"NoCopy", "DriverConfig"}:
        _fail("Docker anonymous workspace volume options differ from the reviewed runner")
    if volume_options.get("NoCopy") is not True:
        _fail("Docker anonymous workspace volume must disable image-content copying")
    if volume_options.get("DriverConfig") != {}:
        _fail("Docker anonymous workspace must not use a custom volume driver")

    mounts = container.get("Mounts")
    if not isinstance(mounts, list) or len(mounts) != 5:
        _fail("Docker build container must have exactly five reviewed mounts")
    by_destination: dict[str, dict[str, Any]] = {}
    for mount in mounts:
        if not isinstance(mount, dict):
            _fail("Docker inspect mount entry is invalid")
        destination = mount.get("Destination")
        if not isinstance(destination, str) or destination in by_destination:
            _fail("Docker inspect mount destination is invalid or duplicated")
        by_destination[destination] = mount
    if set(by_destination) != {
        "/input/source.tar",
        "/input/container-inspect.json",
        "/input/builder-image-inspect.json",
        "/evidence",
        "/workspace",
    }:
        _fail("Docker build container mount inventory differs from the reviewed runner")
    for destination in (
        "/input/source.tar",
        "/input/container-inspect.json",
        "/input/builder-image-inspect.json",
    ):
        mount = by_destination[destination]
        if mount.get("Type") != "bind" or mount.get("RW") is not False:
            _fail(f"Docker input mount is not a read-only bind: {destination}")
        source = mount.get("Source")
        if not isinstance(source, str) or not source.startswith("/"):
            _fail(f"Docker input mount source is not absolute: {destination}")
    evidence_mount = by_destination["/evidence"]
    if evidence_mount.get("Type") != "bind" or evidence_mount.get("RW") is not True:
        _fail("Docker evidence mount is not the sole writable host bind")
    workspace_mount = by_destination["/workspace"]
    if workspace_mount.get("Type") != "volume" or workspace_mount.get("RW") is not True:
        _fail("Docker workspace is not a fresh anonymous writable volume")
    volume_name = workspace_mount.get("Name")
    if not isinstance(volume_name, str) or not volume_name:
        _fail("Docker anonymous workspace volume has no daemon-assigned identity")
    if workspace_mount.get("Driver") != "local":
        _fail("Docker workspace did not resolve through the local volume driver")
    workspace_source = workspace_mount.get("Source")
    expected_workspace_source = f"/var/lib/docker/volumes/{volume_name}/_data"
    if workspace_source != expected_workspace_source:
        _fail("Docker workspace source does not match its daemon-assigned local volume")
    return ContainerDetails(
        container_id,
        volume_name,
        workspace_source,
        started_at,
        finished_at,
    )


def _validate_completion_receipt(
    contents: bytes,
    *,
    build_id: str,
    build_image_id: str,
    source_revision: str,
    source_archive_sha256: str,
    source_date_epoch: int,
    container_id: str,
    file_paths: dict[str, Path],
) -> None:
    if len(contents) > MAX_METADATA_SIZE:
        _fail("in-container completion receipt exceeds its size bound")
    receipt = _strict_json(contents, "in-container completion receipt")
    if contents != canonical_json_bytes(receipt):
        _fail("in-container completion receipt is not canonical JSON")
    _closed_object(
        receipt,
        "in-container completion receipt",
        {
            "schema_version",
            "gate_id",
            "status",
            "build_id",
            "container_id",
            "build_image_id",
            "source",
            "cargo_commands",
            "outputs",
        },
    )
    if _integer(receipt["schema_version"], "completion schema_version") != SCHEMA_VERSION:
        _fail("in-container completion receipt schema is unsupported")
    expected_scalars = {
        "gate_id": GATE_ID,
        "status": "completed",
        "build_id": build_id,
        "container_id": container_id,
        "build_image_id": build_image_id,
    }
    for key, value in expected_scalars.items():
        if _string(receipt[key], f"completion {key}") != value:
            _fail(f"in-container completion receipt differs for {key}")
    source = _closed_object(
        receipt["source"],
        "completion source",
        {"revision", "archive_sha256", "source_date_epoch"},
    )
    if source != {
        "revision": source_revision,
        "archive_sha256": source_archive_sha256,
        "source_date_epoch": source_date_epoch,
    }:
        _fail("in-container completion source provenance differs")
    commands = expected_commands(source_revision, source_date_epoch)
    cargo_commands = _closed_object(
        receipt["cargo_commands"],
        "completion Cargo commands",
        {"release", "profile"},
    )
    if cargo_commands != {
        "release": commands["build"],
        "profile": commands["profile_build"],
    }:
        _fail("in-container completion Cargo commands differ from the locked/offline builds")
    outputs = _closed_object(
        receipt["outputs"],
        "completion outputs",
        set(COMPLETION_OUTPUTS),
    )
    for name, (receipt_path, evidence_path) in COMPLETION_OUTPUTS.items():
        record = _closed_object(
            outputs[name],
            f"completion output {name}",
            {"path", "sha256", "size"},
        )
        if _string(record["path"], f"completion output {name}.path") != receipt_path:
            _fail(f"in-container completion output path differs for {name}")
        digest = _string(record["sha256"], f"completion output {name}.sha256")
        if SHA256_PATTERN.fullmatch(digest) is None:
            _fail(f"in-container completion output SHA-256 is invalid for {name}")
        size = _integer(record["size"], f"completion output {name}.size")
        path = file_paths[evidence_path]
        if size != path.stat().st_size or digest != _sha256_file(path):
            _fail(f"in-container completion output bytes differ for {name}")


def _validate_source_archive(path: Path, source_revision: str, source_date_epoch: int) -> None:
    _regular_path(path, "canonical source archive")
    if path.stat().st_size > MAX_SOURCE_ARCHIVE_SIZE:
        _fail("canonical source archive exceeds its size bound")
    names: list[str] = []
    seen_names: set[str] = set()
    previous_archive_name: str | None = None
    last_data_end = 0
    total_size = 0
    required = {"Cargo.lock", "Cargo.toml", "ci/release/Dockerfile"}
    try:
        with tarfile.open(path, mode="r:") as archive:
            if archive.pax_headers != {"comment": source_revision}:
                _fail("source archive does not embed the exact git archive revision")
            for member in archive:
                if len(names) >= MAX_SOURCE_MEMBERS:
                    _fail("source archive contains too many members")
                _safe_member_path(member.name, "source archive")
                if member.name in seen_names:
                    _fail(f"source archive repeats member {member.name}")
                # tarfile normalizes a directory header's trailing slash away.
                # Git archive sorts the original header names, so restore the
                # slash before comparing (for example: vllm.json precedes
                # vllm/ because '.' sorts before '/').
                archive_name = member.name + "/" if member.isdir() else member.name
                if previous_archive_name is not None and archive_name < previous_archive_name:
                    _fail("source archive members are not bytewise sorted")
                if member.pax_headers != {"comment": source_revision}:
                    _fail(f"source archive member has unreviewed PAX metadata: {member.name}")
                if not (member.isdir() or member.isreg()):
                    _fail(f"source archive contains a link or special member: {member.name}")
                if member.uid != 0 or member.gid != 0 or member.uname != "root" or member.gname != "root":
                    _fail(f"source archive ownership differs from git archive: {member.name}")
                if member.mtime != source_date_epoch:
                    _fail(f"source archive member mtime differs from SOURCE_DATE_EPOCH: {member.name}")
                expected_modes = {0o775} if member.isdir() else {0o664, 0o775}
                if member.mode not in expected_modes:
                    _fail(f"source archive member mode differs from git archive: {member.name}")
                if member.size > MAX_SOURCE_MEMBER_SIZE:
                    _fail(f"source archive member exceeds its size bound: {member.name}")
                total_size += member.size
                if total_size > MAX_SOURCE_TOTAL_SIZE:
                    _fail("source archive exceeds its total size bound")
                last_data_end = max(last_data_end, _tar_data_end(member))
                names.append(member.name)
                seen_names.add(member.name)
                previous_archive_name = archive_name
    except tarfile.TarError as error:
        raise ReleaseContractError(f"source archive is not an uncompressed git tar: {error}") from error
    _validate_tar_end(path, last_data_end, "source archive")
    if not names:
        _fail("source archive is empty")
    normalized = {name.removesuffix("/") for name in names}
    missing = required - normalized
    if missing:
        _fail("source archive is missing release build inputs: " + ", ".join(sorted(missing)))


def _read_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo, limit: int) -> bytes:
    if member.size > limit:
        _fail(f"nested release member exceeds its size bound: {member.name}")
    source = archive.extractfile(member)
    if source is None:
        _fail(f"cannot read nested release member: {member.name}")
    contents = source.read(limit + 1)
    if len(contents) != member.size or len(contents) > limit:
        _fail(f"nested release member size is invalid: {member.name}")
    return contents


def _bundle_details(path: Path) -> BundleDetails:
    _regular_path(path, "release bundle")
    verify_bundle(path)
    binary: bytes | None = None
    native: bytes | None = None
    manifest_contents: bytes | None = None
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            if member.name.endswith("/bin/rustinfer"):
                binary = _read_tar_member(archive, member, MAX_EVIDENCE_MEMBER_SIZE)
            elif member.name.endswith("/manifest/native-dependencies.txt"):
                native = _read_tar_member(archive, member, 1024 * 1024)
            elif member.name.endswith("/manifest/release.json"):
                manifest_contents = _read_tar_member(archive, member, 1024 * 1024)
    if binary is None or native is None or manifest_contents is None:
        _fail("verified release bundle is missing reproducibility comparison members")
    manifest = load_json_object(manifest_contents, "release manifest")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        _fail("release bundle artifact manifest is invalid")
    revision = artifact.get("source_revision")
    epoch = artifact.get("source_date_epoch")
    if not isinstance(revision, str) or not isinstance(epoch, int) or isinstance(epoch, bool):
        _fail("release bundle source provenance is invalid")
    return BundleDetails(binary, native, revision, epoch)


def _load_evidence(
    archive_path: Path,
    temporary_root: Path,
    *,
    expected_build_id: str,
    source_revision: str,
    source_archive: Path,
    source_archive_sha256: str,
    source_date_epoch: int,
    build_image_id: str,
) -> Evidence:
    _regular_path(archive_path, f"build {expected_build_id} evidence archive")
    if archive_path.stat().st_size > MAX_EVIDENCE_ARCHIVE_SIZE:
        _fail("reproducibility evidence archive exceeds its size bound")
    extraction_root = temporary_root / expected_build_id.lower()
    extraction_root.mkdir()
    expected_root = f"rustinfer-repro-build-{expected_build_id.lower()}"
    names: list[str] = []
    member_types: dict[str, str] = {}
    member_modes: dict[str, int] = {}
    extracted: dict[str, Path] = {}
    common_mtime: int | None = None
    total_size = 0
    expected_offset = 0
    try:
        with archive_path.open("rb") as raw_layout, tarfile.open(archive_path, mode="r:") as archive:
            if archive.pax_headers:
                _fail("reproducibility evidence archive has forbidden global PAX metadata")
            for member in archive:
                if len(names) >= MAX_EVIDENCE_MEMBERS:
                    _fail("reproducibility evidence archive contains too many members")
                path = _safe_member_path(member.name, "evidence archive")
                if member.name in member_types:
                    _fail(f"reproducibility evidence repeats member {member.name}")
                if names and member.name < names[-1]:
                    _fail("reproducibility evidence members are not bytewise sorted")
                if member.pax_headers:
                    _fail(f"reproducibility evidence has forbidden PAX metadata: {member.name}")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    _fail(f"reproducibility evidence contains a link or special file: {member.name}")
                if not (member.isdir() or member.isreg()):
                    _fail(f"reproducibility evidence member type is invalid: {member.name}")
                if member.isdir() and member.size != 0:
                    _fail(f"reproducibility evidence directory has data: {member.name}")
                if member.uid != 0 or member.gid != 0 or member.uname or member.gname:
                    _fail(f"reproducibility evidence ownership is not deterministic: {member.name}")
                if common_mtime is None:
                    common_mtime = member.mtime
                elif common_mtime != member.mtime:
                    _fail("reproducibility evidence member mtimes are inconsistent")
                if member.size > MAX_EVIDENCE_MEMBER_SIZE:
                    _fail(f"reproducibility evidence member exceeds its size bound: {member.name}")
                total_size += member.size
                if total_size > MAX_EVIDENCE_TOTAL_SIZE:
                    _fail("reproducibility evidence exceeds its total size bound")
                if member.offset != expected_offset or member.offset_data != expected_offset + TAR_BLOCK_SIZE:
                    _fail(f"reproducibility evidence has non-canonical header layout: {member.name}")
                expected_info = tarfile.TarInfo(member.name)
                expected_info.type = tarfile.DIRTYPE if member.isdir() else tarfile.REGTYPE
                expected_info.mode = member.mode
                expected_info.uid = 0
                expected_info.gid = 0
                expected_info.uname = ""
                expected_info.gname = ""
                expected_info.mtime = member.mtime
                expected_info.size = member.size
                raw_layout.seek(member.offset)
                actual_header = raw_layout.read(TAR_BLOCK_SIZE)
                if actual_header != expected_info.tobuf(format=tarfile.USTAR_FORMAT):
                    _fail(f"reproducibility evidence has non-canonical USTAR header: {member.name}")
                data_end = member.offset_data + member.size
                padded_end = _tar_data_end(member)
                raw_layout.seek(data_end)
                data_padding = raw_layout.read(padded_end - data_end)
                if len(data_padding) != padded_end - data_end or any(data_padding):
                    _fail(f"reproducibility evidence has non-zero data padding: {member.name}")
                expected_offset = padded_end
                member_types[member.name] = "directory" if member.isdir() else "file"
                member_modes[member.name] = member.mode
                if member.isreg():
                    source = archive.extractfile(member)
                    if source is None:
                        _fail(f"cannot read evidence archive member: {member.name}")
                    destination = extraction_root.joinpath(*path.parts)
                    _copy_member(source, destination, member.size, "evidence archive")
                    extracted[member.name] = destination
                names.append(member.name)
    except (tarfile.TarError, ValueError) as error:
        raise ReleaseContractError(f"evidence is not a raw uncompressed tar archive: {error}") from error

    _validate_tar_end(archive_path, expected_offset, "reproducibility evidence archive")
    if common_mtime != source_date_epoch:
        _fail("evidence archive mtime differs from SOURCE_DATE_EPOCH")
    roots = {PurePosixPath(name).parts[0] for name in names}
    if len(roots) != 1:
        _fail("evidence archive must contain exactly one root")
    root = roots.pop()
    if root != expected_root:
        _fail(f"evidence archive root must be {expected_root}")
    expected_names = {root} | {f"{root}/{name}" for name in RELATIVE_DIRECTORIES | RELATIVE_FILES}
    if set(names) != expected_names:
        _fail(
            "evidence archive inventory differs from the closed contract; "
            f"missing={sorted(expected_names - set(names))}, extra={sorted(set(names) - expected_names)}"
        )
    for name in expected_names:
        relative = name.removeprefix(f"{root}/") if name != root else ""
        expected_directory = name == root or relative in RELATIVE_DIRECTORIES
        actual_type = member_types[name]
        if (actual_type == "directory") != expected_directory:
            _fail(f"evidence archive member type is invalid: {name}")
        expected_mode = (
            0o755
            if expected_directory or relative in {"bin/rustinfer", "bin/rustinfer-profile"}
            else 0o644
        )
        if member_modes[name] != expected_mode:
            _fail(f"evidence archive member mode is invalid: {name}")

    file_paths = {
        name.removeprefix(f"{root}/"): path
        for name, path in extracted.items()
    }
    checksum_path = file_paths["SHA256SUMS"]
    if checksum_path.stat().st_size > MAX_METADATA_SIZE:
        _fail("evidence SHA256SUMS exceeds its size bound")
    checksums = _parse_checksums(checksum_path.read_bytes(), RELATIVE_FILES - {"SHA256SUMS"})
    for relative, digest in checksums.items():
        if _sha256_file(file_paths[relative]) != digest:
            _fail(f"reproducibility evidence SHA-256 mismatch: {relative}")

    if file_paths["manifest/build.json"].stat().st_size > MAX_METADATA_SIZE:
        _fail("evidence build manifest exceeds its size bound")
    manifest_contents = file_paths["manifest/build.json"].read_bytes()
    manifest = _strict_json(manifest_contents, "evidence build manifest")
    if manifest_contents != canonical_json_bytes(manifest):
        _fail("evidence build manifest is not canonical JSON")
    _validate_manifest(
        manifest,
        expected_build_id=expected_build_id,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        source_date_epoch=source_date_epoch,
        build_image_id=build_image_id,
        file_paths=file_paths,
    )
    builder_environment = _validate_builder_image_inspect(
        file_paths["logs/builder-image-inspect.json"].read_bytes(),
        build_image_id,
    )
    pre_container = _validate_container_inspect(
        file_paths["logs/container-inspect.json"].read_bytes(),
        build_id=expected_build_id,
        build_image_id=build_image_id,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        source_date_epoch=source_date_epoch,
        builder_environment=builder_environment,
    )
    post_container = _validate_container_inspect(
        file_paths["logs/container-inspect-post.json"].read_bytes(),
        build_id=expected_build_id,
        build_image_id=build_image_id,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        source_date_epoch=source_date_epoch,
        builder_environment=builder_environment,
        expected_phase="exited",
    )
    if (
        pre_container.container_id != post_container.container_id
        or pre_container.workspace_volume != post_container.workspace_volume
        or pre_container.workspace_source != post_container.workspace_source
    ):
        _fail("pre-start and post-run Docker receipts describe different containers")
    _validate_completion_receipt(
        file_paths["logs/build-completion.json"].read_bytes(),
        build_id=expected_build_id,
        build_image_id=build_image_id,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        source_date_epoch=source_date_epoch,
        container_id=pre_container.container_id,
        file_paths=file_paths,
    )
    evidence = Evidence(
        expected_build_id,
        archive_path,
        root,
        file_paths,
        manifest,
        source_date_epoch,
        pre_container.container_id,
        pre_container.workspace_volume,
        pre_container.workspace_source,
        post_container.started_at,
        post_container.finished_at,
    )
    _validate_logs(evidence, build_image_id)

    _files_equal(file_paths["source.tar"], source_archive, f"build {expected_build_id} source archive")
    if _sha256_file(file_paths["source.tar"]) != source_archive_sha256:
        _fail(f"build {expected_build_id} source archive digest differs from provenance")

    binary_path = file_paths["bin/rustinfer"]
    profile_binary_path = file_paths["bin/rustinfer-profile"]
    native_path = file_paths["manifest/native-dependencies.txt"]
    if binary_path.stat().st_size > MAX_BINARY_SIZE:
        _fail(f"build {expected_build_id} binary exceeds its size bound")
    if profile_binary_path.stat().st_size > MAX_BINARY_SIZE:
        _fail(f"build {expected_build_id} profile binary exceeds its size bound")
    if native_path.stat().st_size > MAX_METADATA_SIZE:
        _fail(f"build {expected_build_id} native manifest exceeds its size bound")
    binary = binary_path.read_bytes()
    profile_binary = profile_binary_path.read_bytes()
    native = native_path.read_bytes()
    validate_binary(binary)
    validate_binary(profile_binary)
    parse_native_manifest(native)
    details = _bundle_details(file_paths["bundle/rustinfer.tar.gz"])
    if details.source_revision != source_revision or details.source_date_epoch != source_date_epoch:
        _fail(f"build {expected_build_id} bundle provenance differs from canonical source")
    if details.binary != binary:
        _fail(f"build {expected_build_id} bundle binary differs from its raw binary")
    if details.native_manifest != native:
        _fail(f"build {expected_build_id} bundle native manifest differs from raw evidence")
    return evidence


def validate_single_evidence(
    evidence_archive: Path,
    *,
    expected_build_id: str,
    source_archive: Path,
    source_revision: str,
    source_date_epoch: int,
    build_image_id: str,
) -> None:
    if expected_build_id not in {"A", "B"}:
        _fail("expected build id must be A or B")
    if REVISION_PATTERN.fullmatch(source_revision) is None:
        _fail("source revision must be a full lowercase Git SHA")
    if IMAGE_ID_PATTERN.fullmatch(build_image_id) is None:
        _fail("build image must be an immutable sha256 OCI image ID")
    if not 0 <= source_date_epoch <= 0xFFFFFFFF:
        _fail("SOURCE_DATE_EPOCH must fit an unsigned 32-bit timestamp")
    _validate_source_archive(source_archive, source_revision, source_date_epoch)
    source_digest = _sha256_file(source_archive)
    with tempfile.TemporaryDirectory(prefix="rustinfer-repro-check-") as temporary:
        _load_evidence(
            evidence_archive,
            Path(temporary),
            expected_build_id=expected_build_id,
            source_revision=source_revision,
            source_archive=source_archive,
            source_archive_sha256=source_digest,
            source_date_epoch=source_date_epoch,
            build_image_id=build_image_id,
        )


def check_reproducible_build(
    *,
    evidence_a: Path,
    evidence_b: Path,
    source_archive: Path,
    expected_source_archive_sha256: str,
    source_revision: str,
    source_date_epoch: int,
    build_image_id: str,
    final_binary: Path,
    final_profile_binary: Path,
    final_bundle: Path,
    final_native_manifest: Path,
) -> dict[str, Any]:
    if REVISION_PATTERN.fullmatch(source_revision) is None:
        _fail("source revision must be a full lowercase Git SHA")
    if IMAGE_ID_PATTERN.fullmatch(build_image_id) is None:
        _fail("build image must be an immutable sha256 OCI image ID")
    if SHA256_PATTERN.fullmatch(expected_source_archive_sha256) is None:
        _fail("expected source archive SHA-256 must be a lowercase digest")
    if not 0 <= source_date_epoch <= 0xFFFFFFFF:
        _fail("SOURCE_DATE_EPOCH must fit an unsigned 32-bit timestamp")
    _validate_source_archive(source_archive, source_revision, source_date_epoch)
    source_digest = _sha256_file(source_archive)
    if source_digest != expected_source_archive_sha256:
        _fail("canonical source archive differs from the trusted expected SHA-256")

    _regular_path(final_binary, "final release binary")
    _regular_path(final_profile_binary, "final release profile binary")
    _regular_path(final_bundle, "final release bundle")
    _regular_path(final_native_manifest, "final native dependency manifest")
    if final_binary.stat().st_size > MAX_BINARY_SIZE:
        _fail("final release binary exceeds its size bound")
    if final_profile_binary.stat().st_size > MAX_BINARY_SIZE:
        _fail("final release profile binary exceeds its size bound")
    if final_native_manifest.stat().st_size > MAX_METADATA_SIZE:
        _fail("final native dependency manifest exceeds its size bound")
    final_binary_bytes = final_binary.read_bytes()
    final_profile_binary_bytes = final_profile_binary.read_bytes()
    final_native_bytes = final_native_manifest.read_bytes()
    validate_binary(final_binary_bytes)
    validate_binary(final_profile_binary_bytes)
    parse_native_manifest(final_native_bytes)
    final_details = _bundle_details(final_bundle)
    if final_details.source_revision != source_revision or final_details.source_date_epoch != source_date_epoch:
        _fail("final release bundle provenance differs from canonical source")
    if final_details.binary != final_binary_bytes:
        _fail("final release bundle binary differs from the explicit final binary")
    if final_details.native_manifest != final_native_bytes:
        _fail("final release bundle native manifest differs from the explicit final manifest")

    with tempfile.TemporaryDirectory(prefix="rustinfer-repro-check-") as temporary:
        temporary_root = Path(temporary)
        build_a = _load_evidence(
            evidence_a,
            temporary_root,
            expected_build_id="A",
            source_revision=source_revision,
            source_archive=source_archive,
            source_archive_sha256=source_digest,
            source_date_epoch=source_date_epoch,
            build_image_id=build_image_id,
        )
        build_b = _load_evidence(
            evidence_b,
            temporary_root,
            expected_build_id="B",
            source_revision=source_revision,
            source_archive=source_archive,
            source_archive_sha256=source_digest,
            source_date_epoch=source_date_epoch,
            build_image_id=build_image_id,
        )
        if build_a.container_id == build_b.container_id:
            _fail("A/B evidence came from the same Docker container identity")
        if build_a.workspace_volume == build_b.workspace_volume:
            _fail("A/B evidence reused the same Docker workspace volume")
        if build_a.workspace_source == build_b.workspace_source:
            _fail("A/B evidence reused the same Docker workspace source")
        if (
            build_a.files["logs/builder-image-inspect.json"].read_bytes()
            != build_b.files["logs/builder-image-inspect.json"].read_bytes()
        ):
            _fail("A/B evidence used different Docker builder image configurations")
        builder_image_inspect_sha256 = _sha256_file(
            build_a.files["logs/builder-image-inspect.json"]
        )
        for relative, final, label in (
            ("bin/rustinfer", final_binary, "release binary A/B/final"),
            (
                "bin/rustinfer-profile",
                final_profile_binary,
                "release profile binary A/B/final",
            ),
            ("bundle/rustinfer.tar.gz", final_bundle, "deterministic bundle A/B/final"),
            (
                "manifest/native-dependencies.txt",
                final_native_manifest,
                "native dependency manifest A/B/final",
            ),
        ):
            _files_equal(build_a.files[relative], build_b.files[relative], label)
            _files_equal(build_a.files[relative], final, label)
        if build_a.files["logs/toolchain.txt"].read_bytes() != build_b.files["logs/toolchain.txt"].read_bytes():
            _fail("A/B toolchain command outputs differ")

    return {
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": "passed",
        "source": {
            "revision": source_revision,
            "archive_sha256": source_digest,
            "source_date_epoch": source_date_epoch,
        },
        "build": {
            "image_id": build_image_id,
            "image_inspect_sha256": builder_image_inspect_sha256,
            "platform": PLATFORM,
            "network": "none",
            "cargo_command": expected_commands(source_revision, source_date_epoch)["build"],
            "profile_cargo_command": expected_commands(source_revision, source_date_epoch)[
                "profile_build"
            ],
            "rust_toolchain": RUST_TOOLCHAIN,
            "cuda_toolkit": CUDA_TOOLKIT,
            "nvcc_version": NVCC_VERSION,
            "cuda_architectures": CUDA_ARCHITECTURES,
            "independent_clean_containers": 2,
        },
        "evidence": {
            "a_sha256": _sha256_file(evidence_a),
            "b_sha256": _sha256_file(evidence_b),
            "a_container_id": build_a.container_id,
            "b_container_id": build_b.container_id,
            "a_workspace_volume": build_a.workspace_volume,
            "b_workspace_volume": build_b.workspace_volume,
            "a_workspace_source": build_a.workspace_source,
            "b_workspace_source": build_b.workspace_source,
            "a_started_at": build_a.started_at,
            "a_finished_at": build_a.finished_at,
            "b_started_at": build_b.started_at,
            "b_finished_at": build_b.finished_at,
        },
        "artifacts": {
            "binary_sha256": _sha256_file(final_binary),
            "profile_binary_sha256": _sha256_file(final_profile_binary),
            "bundle_sha256": _sha256_file(final_bundle),
            "native_manifest_sha256": _sha256_file(final_native_manifest),
        },
        "comparisons": {
            "binary_a_b_final_byte_exact": True,
            "profile_binary_a_b_final_byte_exact": True,
            "bundle_a_b_final_byte_exact": True,
            "native_manifest_a_b_final_byte_exact": True,
            "source_archive_a_b_final_byte_exact": True,
        },
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(canonical_json_bytes(report))
    except FileExistsError as error:
        raise ReleaseContractError(f"refusing to overwrite reproducibility report: {path}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-a", type=Path, required=True)
    parser.add_argument("--evidence-b", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--build-image-id", required=True)
    parser.add_argument("--final-binary", type=Path, required=True)
    parser.add_argument("--final-profile-binary", type=Path, required=True)
    parser.add_argument("--final-bundle", type=Path, required=True)
    parser.add_argument("--final-native-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = check_reproducible_build(
            evidence_a=args.evidence_a,
            evidence_b=args.evidence_b,
            source_archive=args.source_archive,
            expected_source_archive_sha256=args.expected_source_archive_sha256,
            source_revision=args.source_revision,
            source_date_epoch=args.source_date_epoch,
            build_image_id=args.build_image_id,
            final_binary=args.final_binary,
            final_profile_binary=args.final_profile_binary,
            final_bundle=args.final_bundle,
            final_native_manifest=args.final_native_manifest,
        )
        _write_report(args.output_report, report)
    except (OSError, ReleaseContractError, tarfile.TarError) as error:
        print(f"release reproducibility gate failed: {error}", file=os.sys.stderr)
        return 1
    print(f"release reproducibility gate passed: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
