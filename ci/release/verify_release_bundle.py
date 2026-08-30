#!/usr/bin/env python3
"""Verify a riley release archive without trusting or extracting its paths."""

from __future__ import annotations

import argparse
import os
import re
import stat
import struct
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from release_common import (
    FORBIDDEN_ARCHIVE_SUFFIXES,
    FORBIDDEN_RUNTIME_TERMS,
    ReleaseContractError,
    canonical_json_bytes,
    load_json_object,
    parse_native_manifest,
    release_manifest,
    release_root,
    sha256_bytes,
    validate_binary,
    validate_license,
)

MAX_MEMBER_SIZE = 512 * 1024 * 1024
MAX_TOTAL_SIZE = 1024 * 1024 * 1024
MAX_MEMBERS = 9
CHECKSUM_PATTERN = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_./+-]+)")
ManifestContract = Callable[[bytes, dict[str, Any]], tuple[dict[str, Any], str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    return parser.parse_args()


def _validate_member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "//" in name or name.startswith("/"):
        raise ReleaseContractError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ReleaseContractError(f"unsafe archive member path: {name!r}")
    lowered = name.casefold()
    if any(part in lowered for part in FORBIDDEN_RUNTIME_TERMS):
        raise ReleaseContractError(f"forbidden Python runtime artifact path: {name}")
    if lowered.endswith(FORBIDDEN_ARCHIVE_SUFFIXES):
        raise ReleaseContractError(f"forbidden Python runtime artifact suffix: {name}")
    return path


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = archive.extractfile(member)
    if source is None:
        raise ReleaseContractError(f"cannot read regular archive member: {member.name}")
    contents = source.read(MAX_MEMBER_SIZE + 1)
    if len(contents) != member.size or len(contents) > MAX_MEMBER_SIZE:
        raise ReleaseContractError(f"archive member size is invalid: {member.name}")
    return contents


def _parse_checksums(contents: bytes, expected_paths: set[str]) -> dict[str, str]:
    try:
        text = contents.decode("ascii")
    except UnicodeDecodeError as error:
        raise ReleaseContractError("SHA256SUMS is not ASCII") from error
    if not text.endswith("\n"):
        raise ReleaseContractError("SHA256SUMS must end with a newline")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        match = CHECKSUM_PATTERN.fullmatch(line)
        if match is None:
            raise ReleaseContractError(f"invalid SHA256SUMS line: {line!r}")
        digest, path = match.groups()
        _validate_member_path(path)
        if path in entries:
            raise ReleaseContractError(f"duplicate SHA256SUMS path: {path}")
        entries[path] = digest
    if set(entries) != expected_paths:
        missing = sorted(expected_paths - set(entries))
        extra = sorted(set(entries) - expected_paths)
        raise ReleaseContractError(f"SHA256SUMS path set mismatch; missing={missing}, extra={extra}")
    if list(entries) != sorted(entries):
        raise ReleaseContractError("SHA256SUMS paths must be bytewise sorted")
    return entries


def _gzip_mtime(bundle: Path) -> int:
    with bundle.open("rb") as source:
        header = source.read(10)
    if len(header) != 10 or header[:3] != b"\x1f\x8b\x08":
        raise ReleaseContractError("release bundle is not a gzip stream")
    if header[3] != 0:
        raise ReleaseContractError("gzip optional headers are forbidden for deterministic output")
    return struct.unpack_from("<I", header, 4)[0]


def _active_manifest_contract(
    raw: bytes,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ReleaseContractError("release manifest artifact field is invalid")
    source_date_epoch = artifact.get("source_date_epoch")
    if not isinstance(source_date_epoch, int) or isinstance(source_date_epoch, bool):
        raise ReleaseContractError("release manifest SOURCE_DATE_EPOCH must be an integer")
    expected_manifest = release_manifest(
        str(artifact.get("version", "")),
        str(artifact.get("source_revision", "")),
        source_date_epoch,
    )
    if manifest != expected_manifest or raw != canonical_json_bytes(expected_manifest):
        raise ReleaseContractError("release manifest differs from the reviewed canonical contract")
    return expected_manifest, release_root(expected_manifest["artifact"]["version"])


def _verify_bundle(
    bundle: Path,
    *,
    max_total_bytes: int | None = None,
    manifest_contract: ManifestContract = _active_manifest_contract,
) -> None:
    """Verify one bundle with a caller-selected, internal manifest contract.

    ``manifest_contract`` is intentionally an internal hook.  The public
    verifier below always retains the active release contract; a separately
    reviewed historical wrapper may select its own closed implementation.
    """

    if (
        max_total_bytes is not None
        and (type(max_total_bytes) is not int or max_total_bytes < 1)
    ):
        raise ReleaseContractError("release bundle retained-byte cap must be a positive integer")
    try:
        metadata = bundle.lstat()
    except OSError as error:
        raise ReleaseContractError(f"cannot inspect release bundle {bundle}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ReleaseContractError("release bundle must be a regular file, not a link or device")
    gzip_mtime = _gzip_mtime(bundle)

    with tarfile.open(bundle, mode="r|gz") as archive:
        members: list[tarfile.TarInfo] = []
        names: list[str] = []
        paths: list[PurePosixPath] = []
        common_mtime = None
        total_size = 0
        file_contents: dict[str, bytes] = {}
        for member in archive:
            if len(members) >= MAX_MEMBERS:
                raise ReleaseContractError("release bundle contains too many members")
            path = _validate_member_path(member.name)
            if member.name in file_contents or member.name in names:
                raise ReleaseContractError("release bundle contains duplicate member paths")
            if names and member.name < names[-1]:
                raise ReleaseContractError("release bundle members are not bytewise sorted")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ReleaseContractError(f"links and special files are forbidden: {member.name}")
            if member.pax_headers:
                raise ReleaseContractError(f"PAX metadata is forbidden: {member.name}")
            if not member.isdir() and not member.isreg():
                raise ReleaseContractError(f"archive member must be regular: {member.name}")
            if member.uid != 0 or member.gid != 0 or member.uname or member.gname:
                raise ReleaseContractError(f"archive ownership metadata is not deterministic: {member.name}")
            if common_mtime is None:
                common_mtime = member.mtime
            elif member.mtime != common_mtime:
                raise ReleaseContractError("archive member mtimes are inconsistent")
            if member.size > MAX_MEMBER_SIZE:
                raise ReleaseContractError(f"archive member exceeds the size bound: {member.name}")
            total_size += member.size
            if total_size > MAX_TOTAL_SIZE:
                raise ReleaseContractError("release bundle exceeds the total uncompressed size bound")
            if (
                max_total_bytes is not None
                and total_size > max_total_bytes
            ):
                raise ReleaseContractError(
                    "release bundle exceeds the caller retained-byte cap"
                )
            if member.isreg():
                file_contents[member.name] = _read_member(archive, member)
            members.append(member)
            names.append(member.name)
            paths.append(path)

    if not members:
        raise ReleaseContractError("release bundle is empty")
    roots = {path.parts[0] for path in paths}
    if len(roots) != 1:
        raise ReleaseContractError("release bundle must contain exactly one root directory")
    root = roots.pop()
    required_files = {
        f"{root}/LICENSE",
        f"{root}/SHA256SUMS",
        f"{root}/bin/riley",
        f"{root}/manifest/native-dependencies.txt",
        f"{root}/manifest/release.json",
    }
    optional_files = {f"{root}/NOTICE"}
    required_directories = {root, f"{root}/bin", f"{root}/manifest"}
    allowed_names = required_files | optional_files | required_directories
    if not required_files <= set(names):
        raise ReleaseContractError(
            "release bundle is missing required files: "
            + ", ".join(sorted(required_files - set(names)))
        )
    extra_names = set(names) - allowed_names
    if extra_names:
        raise ReleaseContractError(
            "release bundle contains unreviewed extra members: "
            + ", ".join(sorted(extra_names))
        )
    for member in members:
        expected_directory = member.name in required_directories
        if expected_directory != member.isdir():
            raise ReleaseContractError(f"archive member type is invalid: {member.name}")
        expected_mode = 0o755 if expected_directory or member.name.endswith("/bin/riley") else 0o644
        if member.mode != expected_mode:
            raise ReleaseContractError(f"archive member mode is invalid: {member.name}")

    if common_mtime != gzip_mtime:
        raise ReleaseContractError("gzip and tar SOURCE_DATE_EPOCH values differ")
    manifest_path = f"{root}/manifest/release.json"
    manifest = load_json_object(file_contents[manifest_path], "release manifest")
    expected_manifest, expected_root = manifest_contract(
        file_contents[manifest_path],
        manifest,
    )
    if root != expected_root:
        raise ReleaseContractError("archive root does not match the release manifest version")
    if expected_manifest["artifact"]["source_date_epoch"] != common_mtime:
        raise ReleaseContractError("release manifest SOURCE_DATE_EPOCH differs from archive metadata")

    license_path = f"{root}/LICENSE"
    validate_license(file_contents[license_path])
    notice_path = f"{root}/NOTICE"
    if notice_path in file_contents and not file_contents[notice_path].strip():
        raise ReleaseContractError("NOTICE must not be empty when present")

    binary_path = f"{root}/bin/riley"
    dependencies = validate_binary(file_contents[binary_path])
    native_path = f"{root}/manifest/native-dependencies.txt"
    recorded_dependencies = parse_native_manifest(file_contents[native_path])
    if recorded_dependencies != dependencies:
        raise ReleaseContractError("native dependency manifest does not match CLI ELF DT_NEEDED entries")

    checksum_path = f"{root}/SHA256SUMS"
    relative_files = {
        name.removeprefix(f"{root}/") for name in file_contents if name != checksum_path
    }
    checksums = _parse_checksums(file_contents[checksum_path], relative_files)
    for relative_path, expected_digest in checksums.items():
        actual_digest = sha256_bytes(file_contents[f"{root}/{relative_path}"])
        if actual_digest != expected_digest:
            raise ReleaseContractError(f"SHA-256 mismatch: {relative_path}")


def verify_bundle(
    bundle: Path,
    *,
    max_total_bytes: int | None = None,
) -> None:
    """Verify one bundle under the active reviewed release contract."""

    _verify_bundle(bundle, max_total_bytes=max_total_bytes)


def main() -> int:
    args = parse_args()
    try:
        verify_bundle(args.bundle)
    except (OSError, ReleaseContractError, tarfile.TarError) as error:
        print(f"release bundle verification failed: {error}", file=os.sys.stderr)
        return 1
    print(f"verified {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
