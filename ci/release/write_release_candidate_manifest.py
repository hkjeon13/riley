#!/usr/bin/env python3
"""Write one deterministic, self-checked final release-candidate manifest.

The writer is intentionally only an assembler.  It never discovers promotion
anchors from submitted evidence and it never runs a producer, model, GPU,
container, or network operation.  Every referenced artifact must already live
under the read-only evidence root.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence

import check_release_candidate as candidate_checker
from release_common import ReleaseContractError, canonical_json_bytes
from verify_release_bundle import verify_bundle


MANIFEST_VERSION = candidate_checker.MANIFEST_VERSION
REPORT_VERSION = candidate_checker.REPORT_VERSION


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    option: str
    manifest_path: tuple[str, ...]


ARTIFACT_SPECS = (
    ArtifactSpec("source_archive", "--source-archive", ("source", "archive")),
    ArtifactSpec("release_binary", "--release-binary", ("release", "binary")),
    ArtifactSpec("release_bundle", "--release-bundle", ("release", "bundle")),
    ArtifactSpec(
        "python_free_e2e_report",
        "--python-free-e2e-report",
        ("evidence", "python_free_e2e", "report"),
    ),
    ArtifactSpec(
        "python_free_e2e_raw_evidence",
        "--python-free-e2e-raw-evidence",
        ("evidence", "python_free_e2e", "raw_evidence"),
    ),
    ArtifactSpec(
        "python_free_e2e_correctness_golden",
        "--python-free-e2e-correctness-golden",
        ("evidence", "python_free_e2e", "correctness_golden"),
    ),
    ArtifactSpec(
        "cuda_fault_report",
        "--cuda-fault-report",
        ("evidence", "cuda_fault", "report"),
    ),
    ArtifactSpec(
        "cuda_fault_raw_evidence",
        "--cuda-fault-raw-evidence",
        ("evidence", "cuda_fault", "raw_evidence"),
    ),
    ArtifactSpec(
        "native_correctness_report",
        "--native-correctness-report",
        ("evidence", "native_correctness", "report"),
    ),
    ArtifactSpec(
        "native_correctness_raw_replay",
        "--native-correctness-raw-replay",
        ("evidence", "native_correctness", "raw_replay"),
    ),
    ArtifactSpec(
        "native_correctness_candidate_executable",
        "--native-correctness-candidate-executable",
        ("evidence", "native_correctness", "candidate_executable"),
    ),
    ArtifactSpec(
        "reproducible_build_a",
        "--reproducible-build-a",
        ("evidence", "reproducible_build", "build_a"),
    ),
    ArtifactSpec(
        "reproducible_build_b",
        "--reproducible-build-b",
        ("evidence", "reproducible_build", "build_b"),
    ),
    ArtifactSpec(
        "reproducible_profile_binary",
        "--reproducible-profile-binary",
        ("evidence", "reproducible_build", "profile_binary"),
    ),
    ArtifactSpec(
        "reproducible_native_manifest",
        "--reproducible-native-manifest",
        ("evidence", "reproducible_build", "native_manifest"),
    ),
    ArtifactSpec(
        "optimization_correctness_report",
        "--optimization-correctness-report",
        ("evidence", "optimization_correctness", "report"),
    ),
    ArtifactSpec(
        "optimization_correctness_raw_evidence",
        "--optimization-correctness-raw-evidence",
        ("evidence", "optimization_correctness", "raw_evidence"),
    ),
    ArtifactSpec(
        "performance_report",
        "--performance-report",
        ("evidence", "performance", "report"),
    ),
    ArtifactSpec(
        "performance_raw_evidence",
        "--performance-raw-evidence",
        ("evidence", "performance", "raw_evidence"),
    ),
    ArtifactSpec(
        "reliability_soak_report",
        "--reliability-soak-report",
        ("evidence", "reliability_soak", "report"),
    ),
    ArtifactSpec(
        "reliability_soak_raw_evidence",
        "--reliability-soak-raw-evidence",
        ("evidence", "reliability_soak", "raw_evidence"),
    ),
)

ARTIFACT_KEYS = {spec.key for spec in ARTIFACT_SPECS}
EXECUTABLE_KEYS = {
    "release_binary",
    "native_correctness_candidate_executable",
    "reproducible_profile_binary",
}
FINAL_CHECK_NAMES = (
    "release_bundle",
    "reproducible_build",
    "python_free_e2e",
    "cuda_fault",
    "native_correctness",
    "optimization_correctness",
    "performance",
    "reliability_soak",
    "cross_bindings",
)
STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
LINUX_AT_EMPTY_PATH = 0x1000
LINUX_AT_FDCWD = -100
LINUX_AT_SYMLINK_FOLLOW = 0x400
LINUX_EMPTY_PATH_UNAVAILABLE = {
    errno.EINVAL,
    errno.ENOENT,
    errno.ENOSYS,
    errno.EOPNOTSUPP,
    errno.EPERM,
}
LINUX_TMPFILE_UNSUPPORTED = {
    errno.EINVAL,
    errno.EISDIR,
    errno.ENOENT,
    errno.ENOSYS,
    errno.EOPNOTSUPP,
    errno.EPERM,
}


class ManifestWriterError(ValueError):
    """The requested manifest would be ambiguous, unsafe, or rejected."""


@dataclass
class ArtifactSnapshot:
    key: str
    relative: str
    digest: str
    snapshot_path: Path
    file_fd: int
    metadata: os.stat_result


@dataclass(frozen=True)
class OutputParentBinding:
    path: Path
    output_name: str
    component_names: tuple[str, ...]
    component_identities: tuple[tuple[int, int], ...]
    parent_metadata: os.stat_result


@dataclass(frozen=True)
class StagedManifest:
    file_fd: int
    checker_path: Path
    linkable_tmpfile: bool


@dataclass(frozen=True)
class WriteResult:
    manifest: dict[str, Any]
    manifest_sha256: str
    self_check_report: dict[str, Any]
    published_path: Path


def _fail(path: str, message: str) -> NoReturn:
    raise ManifestWriterError(f"{path}: {message}")


def _normalize_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(label, "must be a non-empty POSIX relative path")
    if (
        "\x00" in value
        or "\\" in value
        or "//" in value
        or candidate_checker.PLACEHOLDER_RE.search(value)
    ):
        _fail(label, "must be a normalized non-placeholder POSIX relative path")
    pure = PurePosixPath(value)
    if (
        not pure.parts
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(label, "absolute paths, traversal, and normalization aliases are forbidden")
    return value


def _validate_anchors(
    *,
    expected_candidate_id: str,
    expected_revision: str,
    expected_source_archive_sha256: str,
    expected_release_image_id: str,
    expected_reproducible_build_image_id: str,
    expected_cuda_build_image_id: str,
    expected_optimization_build_image_id: str,
    expected_correctness_golden_sha256: str,
) -> tuple[str, str, str, str, str, str, str, str, str]:
    try:
        candidate_id, release_version = candidate_checker._candidate_id(
            expected_candidate_id, "--expected-candidate-id"
        )
        revision = candidate_checker._revision(
            expected_revision, "--expected-revision"
        )
        archive_sha256 = candidate_checker._sha256(
            expected_source_archive_sha256,
            "--expected-source-archive-sha256",
        )
        release_image_id = candidate_checker._image_id(
            expected_release_image_id, "--expected-release-image-id"
        )
        reproducible_image_id = candidate_checker._image_id(
            expected_reproducible_build_image_id,
            "--expected-reproducible-build-image-id",
        )
        cuda_image_id = candidate_checker._image_id(
            expected_cuda_build_image_id,
            "--expected-cuda-build-image-id",
        )
        optimization_image_id = candidate_checker._image_id(
            expected_optimization_build_image_id,
            "--expected-optimization-build-image-id",
        )
        correctness_golden_sha256 = candidate_checker._sha256(
            expected_correctness_golden_sha256,
            "--expected-correctness-golden-sha256",
        )
    except ValueError as error:
        raise ManifestWriterError(str(error)) from error
    return (
        candidate_id,
        release_version,
        revision,
        archive_sha256,
        release_image_id,
        reproducible_image_id,
        cuda_image_id,
        optimization_image_id,
        correctness_golden_sha256,
    )


def _open_root(path: Path) -> tuple[Path, int, os.stat_result]:
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    required_dir_fd_functions = (os.open, os.stat)
    if (
        not all(hasattr(os, name) for name in required_flags)
        or not all(
            function in getattr(os, "supports_dir_fd", set())
            for function in required_dir_fd_functions
        )
        or os.stat not in getattr(os, "supports_follow_symlinks", set())
    ):
        _fail("--evidence-root", "platform lacks required no-follow open flags")
    try:
        resolved = path.resolve(strict=True)
        before = resolved.lstat()
        if not stat.S_ISDIR(before.st_mode):
            _fail("--evidence-root", "must be a real directory")
        root_fd = os.open(
            resolved,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        after = os.fstat(root_fd)
    except ManifestWriterError:
        raise
    except OSError as error:
        _fail("--evidence-root", f"cannot open directory without following links: {error}")
    if (
        not stat.S_ISDIR(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
    ):
        os.close(root_fd)
        _fail("--evidence-root", "directory changed while it was opened")
    return resolved, root_fd, after


def _open_relative_file(root_fd: int, relative: str, label: str) -> int:
    parts = PurePosixPath(relative).parts
    directory_fd = root_fd
    file_fd = -1
    try:
        directory_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            if directory_fd != root_fd:
                os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        return file_fd
    except OSError as error:
        if file_fd >= 0:
            os.close(file_fd)
        _fail(label, f"cannot open without following links: {error}")
    finally:
        if directory_fd != root_fd:
            os.close(directory_fd)


def _metadata_equal(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in STABLE_STAT_FIELDS)


def _hash_fd(file_fd: int, label: str) -> tuple[str, os.stat_result, os.stat_result]:
    before = os.fstat(file_fd)
    if not stat.S_ISREG(before.st_mode):
        _fail(label, "must be a regular file, not a link or device")
    digest = hashlib.sha256()
    offset = 0
    while block := os.pread(file_fd, 1024 * 1024, offset):
        digest.update(block)
        offset += len(block)
    after = os.fstat(file_fd)
    if not _metadata_equal(before, after):
        _fail(label, "changed while it was hashed")
    return digest.hexdigest(), before, after


def _snapshot_artifact(
    *,
    key: str,
    relative: str,
    label: str,
    root_fd: int,
    snapshot_root: Path,
    ordinal: int,
) -> ArtifactSnapshot:
    file_fd = _open_relative_file(root_fd, relative, label)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        snapshot_path = snapshot_root / f"{ordinal:03d}-{PurePosixPath(relative).name}"
        digest = hashlib.sha256()
        duplicate = os.dup(file_fd)
        os.lseek(duplicate, 0, os.SEEK_SET)
        with os.fdopen(duplicate, "rb") as source, snapshot_path.open(
            "xb"
        ) as output:
            while block := source.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        after = os.fstat(file_fd)
        if not _metadata_equal(before, after):
            _fail(label, "changed while it was snapshotted")
        snapshot_path.chmod(stat.S_IMODE(before.st_mode) & 0o777)
        if key in EXECUTABLE_KEYS and before.st_mode & 0o111 == 0:
            _fail(label, "must have at least one executable bit")
        return ArtifactSnapshot(
            key=key,
            relative=relative,
            digest=digest.hexdigest(),
            snapshot_path=snapshot_path,
            file_fd=file_fd,
            metadata=before,
        )
    except BaseException:
        os.close(file_fd)
        raise


def _snapshot_all(
    root_fd: int,
    artifact_paths: Mapping[str, str],
    snapshot_root: Path,
) -> dict[str, ArtifactSnapshot]:
    if set(artifact_paths) != ARTIFACT_KEYS:
        missing = sorted(ARTIFACT_KEYS - set(artifact_paths))
        extra = sorted(set(artifact_paths) - ARTIFACT_KEYS)
        _fail("artifacts", f"closed input mismatch; missing={missing}, unexpected={extra}")
    relative_paths: set[str] = set()
    identities: dict[tuple[int, int], str] = {}
    snapshots: dict[str, ArtifactSnapshot] = {}
    try:
        for ordinal, spec in enumerate(ARTIFACT_SPECS, 1):
            relative = _normalize_relative(artifact_paths[spec.key], spec.option)
            if relative in relative_paths:
                _fail(spec.option, "duplicates another artifact path")
            relative_paths.add(relative)
            snapshot = _snapshot_artifact(
                key=spec.key,
                relative=relative,
                label=spec.option,
                root_fd=root_fd,
                snapshot_root=snapshot_root,
                ordinal=ordinal,
            )
            identity = (snapshot.metadata.st_dev, snapshot.metadata.st_ino)
            previous = identities.get(identity)
            if previous is not None:
                os.close(snapshot.file_fd)
                _fail(spec.option, f"hard-link aliases artifact {previous}")
            identities[identity] = spec.option
            snapshots[spec.key] = snapshot
        return snapshots
    except BaseException:
        for snapshot in snapshots.values():
            os.close(snapshot.file_fd)
        raise


def _bundle_metadata(bundle: Path) -> tuple[str, str, int]:
    try:
        verify_bundle(bundle)
        manifest_bytes: bytes | None = None
        with tarfile.open(bundle, mode="r:gz") as archive:
            for member in archive:
                if member.name.endswith("/manifest/release.json"):
                    if manifest_bytes is not None:
                        _fail("--release-bundle", "contains multiple release manifests")
                    source = archive.extractfile(member)
                    if source is None:
                        _fail("--release-bundle", "cannot read release manifest")
                    manifest_bytes = source.read(1024 * 1024 + 1)
                    if len(manifest_bytes) != member.size or len(manifest_bytes) > 1024 * 1024:
                        _fail("--release-bundle", "release manifest exceeds its size bound")
        if manifest_bytes is None:
            _fail("--release-bundle", "does not contain a release manifest")
        manifest = json.loads(manifest_bytes)
    except ManifestWriterError:
        raise
    except (OSError, tarfile.TarError, json.JSONDecodeError, ReleaseContractError) as error:
        _fail("--release-bundle", f"cannot verify release metadata: {error}")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifact"), dict):
        _fail("--release-bundle", "release manifest artifact is invalid")
    artifact = manifest["artifact"]
    version = artifact.get("version")
    revision = artifact.get("source_revision")
    source_date_epoch = artifact.get("source_date_epoch")
    if not isinstance(version, str) or not version:
        _fail("--release-bundle", "release version is invalid")
    if not isinstance(revision, str):
        _fail("--release-bundle", "source revision is invalid")
    if (
        not isinstance(source_date_epoch, int)
        or isinstance(source_date_epoch, bool)
        or not 0 <= source_date_epoch <= 0xFFFFFFFF
    ):
        _fail("--release-bundle", "SOURCE_DATE_EPOCH is invalid")
    return version, revision, source_date_epoch


def _descriptor(snapshot: ArtifactSnapshot) -> dict[str, str]:
    return {"path": snapshot.relative, "sha256": snapshot.digest}


def _build_manifest(
    snapshots: Mapping[str, ArtifactSnapshot],
    *,
    candidate_id: str,
    revision: str,
    release_image_id: str,
    reproducible_image_id: str,
    cuda_image_id: str,
    optimization_image_id: str,
    source_date_epoch: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "candidate_id": candidate_id,
        "source": {
            "git_revision": revision,
            "git_dirty": False,
        },
        "release": {
            "image_digest": release_image_id,
        },
        "evidence": {
            "python_free_e2e": {},
            "cuda_fault": {
                "build_image_id": cuda_image_id,
            },
            "native_correctness": {},
            "reproducible_build": {
                "build_image_id": reproducible_image_id,
                "source_date_epoch": source_date_epoch,
            },
            "optimization_correctness": {
                "build_image_id": optimization_image_id,
            },
            "performance": {},
            "reliability_soak": {},
        },
    }
    for spec in ARTIFACT_SPECS:
        parent = manifest
        for component in spec.manifest_path[:-1]:
            child = parent.get(component)
            if not isinstance(child, dict):  # pragma: no cover - static contract
                raise AssertionError(f"invalid artifact manifest path: {spec.key}")
            parent = child
        parent[spec.manifest_path[-1]] = _descriptor(snapshots[spec.key])
    return manifest


def _walk_output_components(
    component_names: tuple[str, ...],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = -1
    try:
        directory_fd = os.open(os.sep, directory_flags)
        metadata = os.fstat(directory_fd)
        identities = [(metadata.st_dev, metadata.st_ino)]
        for component in component_names:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
            metadata = os.fstat(directory_fd)
            if not stat.S_ISDIR(metadata.st_mode):  # pragma: no cover - O_DIRECTORY
                _fail("--output", "parent component must be a real directory")
            identities.append((metadata.st_dev, metadata.st_ino))
        return directory_fd, tuple(identities)
    except ManifestWriterError:
        if directory_fd >= 0:
            os.close(directory_fd)
        raise
    except OSError as error:
        if directory_fd >= 0:
            os.close(directory_fd)
        _fail(
            "--output",
            f"cannot traverse lexical parent without following symlinks: {error}",
        )


def _open_output_parent(
    output: Path,
    evidence_root: Path,
) -> tuple[OutputParentBinding, int]:
    if output.name in {"", ".", ".."} or "\x00" in output.name:
        _fail("--output", "must name a file")
    try:
        absolute_output = output if output.is_absolute() else Path.cwd() / output
    except OSError as error:
        _fail("--output", f"cannot anchor relative output path: {error}")
    parts = absolute_output.parts
    if (
        not parts
        or parts[0] != os.sep
        or any(part in {"", ".", ".."} for part in parts[1:])
    ):
        _fail("--output", "must be a normalized absolute or cwd-relative path")
    component_names = tuple(parts[1:-1])
    parent = Path(os.sep).joinpath(*component_names)
    target = parent / parts[-1]
    if target.is_relative_to(evidence_root):
        _fail("--output", "must be outside the read-only evidence root")

    parent_fd, identities = _walk_output_components(component_names)
    parent_metadata = os.fstat(parent_fd)
    try:
        os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as error:
        os.close(parent_fd)
        _fail("--output", f"cannot inspect target: {error}")
    else:
        os.close(parent_fd)
        _fail("--output", "refusing to replace an existing path")
    return (
        OutputParentBinding(
            path=parent,
            output_name=parts[-1],
            component_names=component_names,
            component_identities=identities,
            parent_metadata=parent_metadata,
        ),
        parent_fd,
    )


def _reopen_bound_output_parent(
    binding: OutputParentBinding,
    parent_fd: int,
) -> int:
    held = os.fstat(parent_fd)
    if (
        held.st_dev != binding.parent_metadata.st_dev
        or held.st_ino != binding.parent_metadata.st_ino
    ):
        _fail("--output", "held parent changed during the final self-check")
    fresh_fd, identities = _walk_output_components(binding.component_names)
    if identities != binding.component_identities:
        os.close(fresh_fd)
        _fail("--output", "lexical parent path changed during the final self-check")
    fresh = os.fstat(fresh_fd)
    if fresh.st_dev != held.st_dev or fresh.st_ino != held.st_ino:
        os.close(fresh_fd)
        _fail("--output", "lexical parent no longer names the held directory")
    return fresh_fd


def _revalidate_output_parent(
    binding: OutputParentBinding,
    parent_fd: int,
) -> None:
    fresh_fd = _reopen_bound_output_parent(binding, parent_fd)
    os.close(fresh_fd)


def _checker_fd_path(file_fd: int) -> Path:
    if sys.platform.startswith("linux"):
        path = Path(f"/proc/self/fd/{file_fd}")
    elif sys.platform == "darwin":
        path = Path(f"/dev/fd/{file_fd}")
    else:
        _fail("staged manifest", "anonymous FD self-check requires Linux or macOS")
    named_fd = -1
    try:
        named_fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
        named = os.fstat(named_fd)
        held = os.fstat(file_fd)
    except OSError as error:
        _fail("staged manifest", f"cannot expose held FD to the final checker: {error}")
    finally:
        if named_fd >= 0:
            os.close(named_fd)
    if named.st_dev != held.st_dev or named.st_ino != held.st_ino:
        _fail("staged manifest", "FD checker path does not name the held manifest")
    return path


def _create_staged_manifest(parent_fd: int) -> StagedManifest:
    file_fd = -1
    linkable_tmpfile = False
    if sys.platform.startswith("linux") and hasattr(os, "O_TMPFILE"):
        try:
            file_fd = os.open(
                ".",
                os.O_RDWR | os.O_CLOEXEC | os.O_TMPFILE,
                0o600,
                dir_fd=parent_fd,
            )
        except OSError as error:
            if error.errno not in LINUX_TMPFILE_UNSUPPORTED:
                _fail("--output", f"cannot create anonymous manifest: {error}")
        else:
            linkable_tmpfile = True
    if file_fd < 0:
        try:
            with tempfile.TemporaryFile(mode="w+b") as temporary:
                file_fd = os.dup(temporary.fileno())
            if os.fstat(file_fd).st_nlink != 0:
                os.close(file_fd)
                _fail("staged manifest", "fallback staging file is not anonymous")
        except ManifestWriterError:
            raise
        except OSError as error:
            if file_fd >= 0:
                os.close(file_fd)
            _fail("staged manifest", f"cannot create anonymous fallback: {error}")
    try:
        checker_path = _checker_fd_path(file_fd)
    except BaseException:
        os.close(file_fd)
        raise
    return StagedManifest(file_fd, checker_path, linkable_tmpfile)


def _write_staged_manifest(file_fd: int, contents: bytes) -> None:
    os.ftruncate(file_fd, 0)
    os.lseek(file_fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(file_fd), "wb") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())
    os.fchmod(file_fd, 0o644)
    os.fsync(file_fd)


def _revalidate_staged_manifest(
    staged: StagedManifest,
    expected_sha256: str,
) -> None:
    held_digest, held_before, _ = _hash_fd(staged.file_fd, "staged manifest")
    if stat.S_IMODE(held_before.st_mode) != 0o644 or held_digest != expected_sha256:
        _fail("staged manifest", "changed during the final self-check")
    checker_fd = -1
    try:
        checker_fd = os.open(
            staged.checker_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK,
        )
        checker_metadata = os.fstat(checker_fd)
    except OSError as error:
        _fail("staged manifest", f"cannot revalidate FD checker path: {error}")
    finally:
        if checker_fd >= 0:
            os.close(checker_fd)
    if (
        checker_metadata.st_dev != held_before.st_dev
        or checker_metadata.st_ino != held_before.st_ino
    ):
        _fail("staged manifest", "FD checker path was rebound")


def _link_tmpfile_noreplace(
    file_fd: int,
    output_name: str,
    parent_fd: int,
) -> None:
    """Atomically publish one Linux O_TMPFILE inode without replacing a name."""

    if not sys.platform.startswith("linux"):  # pragma: no cover - caller contract
        _fail("--output", "anonymous hard-link publication requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        _fail("--output", "linkat(AT_EMPTY_PATH) is unavailable")
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = linkat(
        file_fd,
        b"",
        parent_fd,
        os.fsencode(output_name),
        LINUX_AT_EMPTY_PATH,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number in LINUX_EMPTY_PATH_UNAVAILABLE:
        # Unprivileged Linux commonly denies AT_EMPTY_PATH even for a linkable
        # O_TMPFILE inode.  The procfs descriptor link remains kernel-bound to
        # this process's held FD and avoids any mutable staging pathname.
        ctypes.set_errno(0)
        result = linkat(
            LINUX_AT_FDCWD,
            os.fsencode(f"/proc/self/fd/{file_fd}"),
            parent_fd,
            os.fsencode(output_name),
            LINUX_AT_SYMLINK_FOLLOW,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno() or errno.EIO
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), output_name)
    raise OSError(error_number, os.strerror(error_number), output_name)


def _copy_fd(source_fd: int, destination_fd: int) -> None:
    os.ftruncate(destination_fd, 0)
    os.lseek(destination_fd, 0, os.SEEK_SET)
    offset = 0
    while block := os.pread(source_fd, 1024 * 1024, offset):
        view = memoryview(block)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:  # pragma: no cover - POSIX write contract
                raise OSError(errno.EIO, "zero-length manifest write")
            view = view[written:]
        offset += len(block)


def _revalidate_published_output(
    binding: OutputParentBinding,
    parent_fd: int,
    published_fd: int,
    expected_sha256: str,
) -> None:
    published_digest, held, _ = _hash_fd(published_fd, "published manifest")
    if stat.S_IMODE(held.st_mode) != 0o644 or published_digest != expected_sha256:
        _fail("--output", "published manifest differs from the self-checked bytes")
    fresh_parent_fd = _reopen_bound_output_parent(binding, parent_fd)
    opened: list[int] = []
    try:
        for directory_fd in (parent_fd, fresh_parent_fd):
            file_fd = os.open(
                binding.output_name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            opened.append(file_fd)
            digest, metadata, _ = _hash_fd(file_fd, "published manifest")
            if (
                metadata.st_dev != held.st_dev
                or metadata.st_ino != held.st_ino
                or digest != expected_sha256
            ):
                _fail("--output", "published path does not name the held manifest")
    except ManifestWriterError:
        raise
    except OSError as error:
        _fail("--output", f"cannot revalidate published path: {error}")
    finally:
        for file_fd in opened:
            os.close(file_fd)
        os.close(fresh_parent_fd)


def _publish_create_only(
    staged: StagedManifest,
    binding: OutputParentBinding,
    parent_fd: int,
    expected_sha256: str,
) -> None:
    _revalidate_staged_manifest(staged, expected_sha256)
    _revalidate_output_parent(binding, parent_fd)
    published_fd = staged.file_fd
    close_published_fd = False
    try:
        try:
            if staged.linkable_tmpfile:
                _link_tmpfile_noreplace(
                    staged.file_fd,
                    binding.output_name,
                    parent_fd,
                )
            else:
                published_fd = os.open(
                    binding.output_name,
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
                close_published_fd = True
                _copy_fd(staged.file_fd, published_fd)
                os.fsync(published_fd)
                os.fchmod(published_fd, 0o644)
                os.fsync(published_fd)
        except FileExistsError as error:
            _fail("--output", f"refusing to replace an existing path: {error}")
        except OSError as error:
            _fail("--output", f"cannot publish create-only manifest: {error}")
        os.fsync(parent_fd)
        _revalidate_published_output(
            binding,
            parent_fd,
            published_fd,
            expected_sha256,
        )
    finally:
        if close_published_fd:
            os.close(published_fd)


def _validate_self_check(
    report: object,
    *,
    manifest_sha256: str,
    candidate_id: str,
    revision: str,
    archive_sha256: str,
    release_image_id: str,
    reproducible_image_id: str,
    cuda_image_id: str,
    optimization_image_id: str,
    correctness_golden_sha256: str,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        _fail("self-check", "final checker did not return an object")
    if (
        report.get("schema_version") != REPORT_VERSION
        or report.get("status") != "passed"
        or report.get("passed") is not True
        or report.get("errors") != []
        or report.get("candidate_id") != candidate_id
        or report.get("manifest_sha256") != manifest_sha256
    ):
        _fail("self-check", f"final checker rejected the manifest: {report.get('errors')!r}")
    bindings = report.get("bindings")
    if not isinstance(bindings, dict):
        _fail("self-check.bindings", "final checker omitted passed bindings")
    expected = {
        "git_revision": revision,
        "source_archive_sha256": archive_sha256,
        "release_image_sha256": release_image_id.removeprefix("sha256:"),
        "build_image_ids": {
            "reproducible_build": reproducible_image_id,
            "cuda_fault": cuda_image_id,
            "optimization_correctness": optimization_image_id,
        },
        "correctness_golden_sha256": correctness_golden_sha256,
    }
    for field, value in expected.items():
        if bindings.get(field) != value:
            _fail(f"self-check.bindings.{field}", "does not equal the writer input")
    checks = report.get("checks")
    if (
        not isinstance(checks, list)
        or len(checks) != len(FINAL_CHECK_NAMES)
        or any(
            not isinstance(check, dict)
            or set(check) != {"name", "passed"}
            or check.get("name") != name
            or check.get("passed") is not True
            for check, name in zip(checks, FINAL_CHECK_NAMES)
        )
    ):
        _fail("self-check.checks", "final checker did not pass every closed gate")
    return report


def _revalidate_artifacts(
    root_path: Path,
    root_fd: int,
    root_metadata: os.stat_result,
    snapshots: Mapping[str, ArtifactSnapshot],
) -> None:
    current_root = os.fstat(root_fd)
    try:
        named_root = root_path.lstat()
    except OSError as error:
        _fail("--evidence-root", f"cannot revalidate directory path: {error}")
    if (
        current_root.st_dev != root_metadata.st_dev
        or current_root.st_ino != root_metadata.st_ino
        or named_root.st_dev != root_metadata.st_dev
        or named_root.st_ino != root_metadata.st_ino
    ):
        _fail("--evidence-root", "changed during the final self-check")
    for spec in ARTIFACT_SPECS:
        snapshot = snapshots[spec.key]
        reopened_fd = _open_relative_file(root_fd, snapshot.relative, spec.option)
        try:
            digest, before, _ = _hash_fd(reopened_fd, spec.option)
            if (
                before.st_dev != snapshot.metadata.st_dev
                or before.st_ino != snapshot.metadata.st_ino
                or digest != snapshot.digest
            ):
                _fail(spec.option, "path was replaced during the final self-check")
        finally:
            os.close(reopened_fd)
        held = os.fstat(snapshot.file_fd)
        if not _metadata_equal(snapshot.metadata, held):
            _fail(spec.option, "changed during the final self-check")


def write_manifest(
    evidence_root: Path,
    output: Path,
    *,
    expected_candidate_id: str,
    expected_revision: str,
    expected_source_archive_sha256: str,
    expected_release_image_id: str,
    expected_reproducible_build_image_id: str,
    expected_cuda_build_image_id: str,
    expected_optimization_build_image_id: str,
    expected_correctness_golden_sha256: str,
    artifact_paths: Mapping[str, str],
) -> WriteResult:
    (
        candidate_id,
        release_version,
        revision,
        archive_sha256,
        release_image_id,
        reproducible_image_id,
        cuda_image_id,
        optimization_image_id,
        correctness_golden_sha256,
    ) = _validate_anchors(
        expected_candidate_id=expected_candidate_id,
        expected_revision=expected_revision,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_release_image_id=expected_release_image_id,
        expected_reproducible_build_image_id=expected_reproducible_build_image_id,
        expected_cuda_build_image_id=expected_cuda_build_image_id,
        expected_optimization_build_image_id=expected_optimization_build_image_id,
        expected_correctness_golden_sha256=expected_correctness_golden_sha256,
    )
    root_path, root_fd, root_metadata = _open_root(evidence_root)
    snapshots: dict[str, ArtifactSnapshot] = {}
    parent_fd = -1
    staged: StagedManifest | None = None
    try:
        output_binding, parent_fd = _open_output_parent(output, root_path)
        with tempfile.TemporaryDirectory(
            prefix="rustinfer-manifest-writer-snapshot-"
        ) as snapshot_directory:
            snapshots = _snapshot_all(
                root_fd, artifact_paths, Path(snapshot_directory)
            )
            if snapshots["source_archive"].digest != archive_sha256:
                _fail(
                    "--source-archive",
                    "differs from --expected-source-archive-sha256",
                )
            if (
                snapshots["python_free_e2e_correctness_golden"].digest
                != correctness_golden_sha256
            ):
                _fail(
                    "--python-free-e2e-correctness-golden",
                    "differs from --expected-correctness-golden-sha256",
                )
            bundle_version, bundle_revision, source_date_epoch = _bundle_metadata(
                snapshots["release_bundle"].snapshot_path
            )
            if bundle_version != release_version:
                _fail(
                    "--expected-candidate-id",
                    "release version differs from the verified bundle",
                )
            if bundle_revision != revision:
                _fail(
                    "--expected-revision",
                    "source revision differs from the verified bundle",
                )
            manifest = _build_manifest(
                snapshots,
                candidate_id=candidate_id,
                revision=revision,
                release_image_id=release_image_id,
                reproducible_image_id=reproducible_image_id,
                cuda_image_id=cuda_image_id,
                optimization_image_id=optimization_image_id,
                source_date_epoch=source_date_epoch,
            )
            encoded = canonical_json_bytes(manifest)
            manifest_sha256 = hashlib.sha256(encoded).hexdigest()
            staged = _create_staged_manifest(parent_fd)
            _write_staged_manifest(staged.file_fd, encoded)
            _revalidate_staged_manifest(staged, manifest_sha256)
            self_check = candidate_checker.evaluate(
                staged.checker_path,
                root_path,
                manifest_fd=staged.file_fd,
                expected_candidate_id=candidate_id,
                expected_revision=revision,
                expected_source_archive_sha256=archive_sha256,
                expected_release_image_id=release_image_id,
                expected_reproducible_build_image_id=reproducible_image_id,
                expected_cuda_build_image_id=cuda_image_id,
                expected_optimization_build_image_id=optimization_image_id,
                expected_correctness_golden_sha256=correctness_golden_sha256,
            )
            checked_report = _validate_self_check(
                self_check,
                manifest_sha256=manifest_sha256,
                candidate_id=candidate_id,
                revision=revision,
                archive_sha256=archive_sha256,
                release_image_id=release_image_id,
                reproducible_image_id=reproducible_image_id,
                cuda_image_id=cuda_image_id,
                optimization_image_id=optimization_image_id,
                correctness_golden_sha256=correctness_golden_sha256,
            )
            _revalidate_artifacts(root_path, root_fd, root_metadata, snapshots)
            _revalidate_staged_manifest(staged, manifest_sha256)
            _revalidate_output_parent(output_binding, parent_fd)
            _publish_create_only(
                staged,
                output_binding,
                parent_fd,
                manifest_sha256,
            )
            return WriteResult(
                manifest,
                manifest_sha256,
                checked_report,
                output_binding.path / output_binding.output_name,
            )
    finally:
        for snapshot in snapshots.values():
            try:
                os.close(snapshot.file_fd)
            except OSError:
                pass
        if staged is not None:
            try:
                os.close(staged.file_fd)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-candidate-id", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--expected-release-image-id", required=True)
    parser.add_argument("--expected-reproducible-build-image-id", required=True)
    parser.add_argument("--expected-cuda-build-image-id", required=True)
    parser.add_argument("--expected-optimization-build-image-id", required=True)
    parser.add_argument("--expected-correctness-golden-sha256", required=True)
    for spec in ARTIFACT_SPECS:
        parser.add_argument(spec.option, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    arguments = vars(args)
    artifact_paths = {
        spec.key: arguments[spec.option.removeprefix("--").replace("-", "_")]
        for spec in ARTIFACT_SPECS
    }
    try:
        result = write_manifest(
            args.evidence_root,
            args.output,
            expected_candidate_id=args.expected_candidate_id,
            expected_revision=args.expected_revision,
            expected_source_archive_sha256=args.expected_source_archive_sha256,
            expected_release_image_id=args.expected_release_image_id,
            expected_reproducible_build_image_id=(
                args.expected_reproducible_build_image_id
            ),
            expected_cuda_build_image_id=args.expected_cuda_build_image_id,
            expected_optimization_build_image_id=(
                args.expected_optimization_build_image_id
            ),
            expected_correctness_golden_sha256=(
                args.expected_correctness_golden_sha256
            ),
            artifact_paths=artifact_paths,
        )
    except (ManifestWriterError, OSError, ReleaseContractError, tarfile.TarError) as error:
        print(f"release-candidate manifest writer failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(
        json.dumps(
            {
                "schema_version": MANIFEST_VERSION,
                "candidate_id": result.manifest["candidate_id"],
                "manifest_sha256": result.manifest_sha256,
                "path": str(result.published_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
