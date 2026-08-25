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
NVCC_VERSION = "12.8.93"
PLATFORM = "linux/amd64"

IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
CHECKSUM_LINE_PATTERN = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_./+-]+)")

MAX_EVIDENCE_MEMBERS = 18
MAX_EVIDENCE_MEMBER_SIZE = 2 * 1024 * 1024 * 1024
MAX_EVIDENCE_TOTAL_SIZE = 6 * 1024 * 1024 * 1024
MAX_BINARY_SIZE = 512 * 1024 * 1024
MAX_METADATA_SIZE = 1024 * 1024
MAX_SOURCE_MEMBERS = 20_000
MAX_SOURCE_MEMBER_SIZE = 2 * 1024 * 1024 * 1024
MAX_SOURCE_TOTAL_SIZE = 8 * 1024 * 1024 * 1024

RELATIVE_DIRECTORIES = {
    "bin",
    "bundle",
    "logs",
    "manifest",
}
RELATIVE_FILES = {
    "SHA256SUMS",
    "bin/rustinfer",
    "bundle/rustinfer.tar.gz",
    "logs/bundle-build.log",
    "logs/bundle-verify.log",
    "logs/cargo-build.log",
    "logs/container-inspect.json",
    "logs/container-invocation.txt",
    "logs/preflight.log",
    "logs/toolchain.txt",
    "manifest/build.json",
    "manifest/native-dependencies.txt",
    "source.tar",
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
        "bundle": [
            "python3",
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
        {"build", "bundle", "preflight", "verify"},
    )
    if commands != expected_commands(source_revision, source_date_epoch):
        _fail("evidence commands differ from the locked/offline release build contract")

    artifacts = _closed_object(
        manifest["artifacts"],
        "manifest artifacts",
        {
            "binary_sha256",
            "binary_size",
            "bundle_sha256",
            "bundle_size",
            "native_manifest_sha256",
            "native_manifest_size",
        },
    )
    expected_artifacts: dict[str, Any] = {}
    for key, relative in (
        ("binary", "bin/rustinfer"),
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
    cargo_log = read("logs/cargo-build.log")
    try:
        cargo_text = cargo_log.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseContractError("Cargo build log is not UTF-8") from error
    if "Compiling rustinfer-server" not in cargo_text:
        _fail("Cargo build log does not contain the production server compilation")
    if re.search(r"Finished [`']release[`'] profile \[optimized\]", cargo_text) is None:
        _fail("Cargo build log does not contain a completed optimized release profile")
    lowered = cargo_text.casefold()
    if any(marker in lowered for marker in ("updating crates.io", "downloading crates", "http://", "https://")):
        _fail("Cargo build log contains network access markers despite --offline")


def _validate_container_inspect(
    contents: bytes,
    *,
    build_id: str,
    build_image_id: str,
    source_revision: str,
    source_archive_sha256: str,
    source_date_epoch: int,
) -> tuple[str, str]:
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

    config = container.get("Config")
    if not isinstance(config, dict):
        _fail("Docker inspect Config is missing")
    if config.get("Image") != build_image_id:
        _fail("Docker container was not created from the pinned image ID")
    if config.get("WorkingDir") != "/workspace":
        _fail("Docker container working directory is not the clean workspace")
    command = config.get("Cmd")
    if not isinstance(command, list) or len(command) != 3 or command[:2] != ["/bin/bash", "-ceu"]:
        _fail("Docker container command is not the reviewed one-build driver")
    if not isinstance(command[2], str):
        _fail("Docker container build command body is invalid")
    for marker in (
        "tar --extract --file /input/source.tar --directory /workspace",
        "exec /bin/bash ci/release/run_reproducible_build_once.sh",
    ):
        if marker not in command[2]:
            _fail(f"Docker container command is missing reviewed marker: {marker}")

    environment = config.get("Env")
    if not isinstance(environment, list) or not all(isinstance(item, str) for item in environment):
        _fail("Docker inspect environment is invalid")
    parsed_environment: dict[str, str] = {}
    for item in environment:
        key, separator, value = item.partition("=")
        if not separator or not key or key in parsed_environment:
            _fail("Docker inspect environment contains an invalid or duplicate entry")
        parsed_environment[key] = value
    required_environment = {
        "RUSTINFER_REPRO_BUILD_ID": build_id,
        "RUSTINFER_SOURCE_REVISION": source_revision,
        "RUSTINFER_SOURCE_ARCHIVE_SHA256": source_archive_sha256,
        "RUSTINFER_BUILD_IMAGE_ID": build_image_id,
        "SOURCE_DATE_EPOCH": str(source_date_epoch),
    }
    for key, value in required_environment.items():
        if parsed_environment.get(key) != value:
            _fail(f"Docker inspect environment provenance differs for {key}")

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
    if not isinstance(security_options, list) or not any(
        isinstance(option, str) and option.startswith("no-new-privileges")
        for option in security_options
    ):
        _fail("Docker build container must enforce no-new-privileges")
    if host.get("PidsLimit") != 4096:
        _fail("Docker build container PID limit differs from the reviewed runner")
    if host.get("Devices") not in (None, []):
        _fail("Docker build container has explicit host devices")
    if host.get("DeviceRequests") not in (None, []):
        _fail("Docker build container has a GPU/device request")

    mounts = container.get("Mounts")
    if not isinstance(mounts, list) or len(mounts) != 4:
        _fail("Docker build container must have exactly four reviewed mounts")
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
        "/evidence",
        "/workspace",
    }:
        _fail("Docker build container mount inventory differs from the reviewed runner")
    for destination in ("/input/source.tar", "/input/container-inspect.json"):
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
    return container_id, volume_name


def _validate_source_archive(path: Path, source_revision: str, source_date_epoch: int) -> None:
    _regular_path(path, "canonical source archive")
    names: list[str] = []
    seen_names: set[str] = set()
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
                if names and member.name < names[-1]:
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
                names.append(member.name)
                seen_names.add(member.name)
    except tarfile.TarError as error:
        raise ReleaseContractError(f"source archive is not an uncompressed git tar: {error}") from error
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
    extraction_root = temporary_root / expected_build_id.lower()
    extraction_root.mkdir()
    names: list[str] = []
    member_types: dict[str, str] = {}
    member_modes: dict[str, int] = {}
    extracted: dict[str, Path] = {}
    common_mtime: int | None = None
    total_size = 0
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
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
    except tarfile.TarError as error:
        raise ReleaseContractError(f"evidence is not a raw uncompressed tar archive: {error}") from error

    if common_mtime != source_date_epoch:
        _fail("evidence archive mtime differs from SOURCE_DATE_EPOCH")
    roots = {PurePosixPath(name).parts[0] for name in names}
    if len(roots) != 1:
        _fail("evidence archive must contain exactly one root")
    root = roots.pop()
    expected_root = f"rustinfer-repro-build-{expected_build_id.lower()}"
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
        expected_mode = 0o755 if expected_directory or relative == "bin/rustinfer" else 0o644
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
    container_id, workspace_volume = _validate_container_inspect(
        file_paths["logs/container-inspect.json"].read_bytes(),
        build_id=expected_build_id,
        build_image_id=build_image_id,
        source_revision=source_revision,
        source_archive_sha256=source_archive_sha256,
        source_date_epoch=source_date_epoch,
    )
    evidence = Evidence(
        expected_build_id,
        archive_path,
        root,
        file_paths,
        manifest,
        source_date_epoch,
        container_id,
        workspace_volume,
    )
    _validate_logs(evidence, build_image_id)

    _files_equal(file_paths["source.tar"], source_archive, f"build {expected_build_id} source archive")
    if _sha256_file(file_paths["source.tar"]) != source_archive_sha256:
        _fail(f"build {expected_build_id} source archive digest differs from provenance")

    binary_path = file_paths["bin/rustinfer"]
    native_path = file_paths["manifest/native-dependencies.txt"]
    if binary_path.stat().st_size > MAX_BINARY_SIZE:
        _fail(f"build {expected_build_id} binary exceeds its size bound")
    if native_path.stat().st_size > MAX_METADATA_SIZE:
        _fail(f"build {expected_build_id} native manifest exceeds its size bound")
    binary = binary_path.read_bytes()
    native = native_path.read_bytes()
    validate_binary(binary)
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
    source_revision: str,
    source_date_epoch: int,
    build_image_id: str,
    final_binary: Path,
    final_bundle: Path,
    final_native_manifest: Path,
) -> dict[str, Any]:
    if REVISION_PATTERN.fullmatch(source_revision) is None:
        _fail("source revision must be a full lowercase Git SHA")
    if IMAGE_ID_PATTERN.fullmatch(build_image_id) is None:
        _fail("build image must be an immutable sha256 OCI image ID")
    if not 0 <= source_date_epoch <= 0xFFFFFFFF:
        _fail("SOURCE_DATE_EPOCH must fit an unsigned 32-bit timestamp")
    _validate_source_archive(source_archive, source_revision, source_date_epoch)
    source_digest = _sha256_file(source_archive)

    _regular_path(final_binary, "final release binary")
    _regular_path(final_bundle, "final release bundle")
    _regular_path(final_native_manifest, "final native dependency manifest")
    if final_binary.stat().st_size > MAX_BINARY_SIZE:
        _fail("final release binary exceeds its size bound")
    if final_native_manifest.stat().st_size > MAX_METADATA_SIZE:
        _fail("final native dependency manifest exceeds its size bound")
    final_binary_bytes = final_binary.read_bytes()
    final_native_bytes = final_native_manifest.read_bytes()
    validate_binary(final_binary_bytes)
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
        for relative, final, label in (
            ("bin/rustinfer", final_binary, "release binary A/B/final"),
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
            "platform": PLATFORM,
            "network": "none",
            "cargo_command": expected_commands(source_revision, source_date_epoch)["build"],
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
        },
        "artifacts": {
            "binary_sha256": _sha256_file(final_binary),
            "bundle_sha256": _sha256_file(final_bundle),
            "native_manifest_sha256": _sha256_file(final_native_manifest),
        },
        "comparisons": {
            "binary_a_b_final_byte_exact": True,
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
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--build-image-id", required=True)
    parser.add_argument("--final-binary", type=Path, required=True)
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
            source_revision=args.source_revision,
            source_date_epoch=args.source_date_epoch,
            build_image_id=args.build_image_id,
            final_binary=args.final_binary,
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
