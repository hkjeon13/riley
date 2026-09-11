#!/usr/bin/env python3
"""Compose deterministic archives used by the raw runtime-assembly host runner.

This stdlib-only helper has three source-only subcommands:

* ``context`` makes the exact three-member source-free Docker build context;
* ``runtime-tree`` converts Docker's raw ``docker cp`` tar stream into a
  canonical USTAR runtime-tree snapshot; and
* ``capture`` assembles the fixed eleven-member raw capture consumed by
  ``prepare_reconstructed_runtime_assembly_capture_v1.py``.

It never invokes Docker, starts a container, accesses a GPU, contacts a
network endpoint, or makes a qualification decision.  The surrounding host
runner is responsible for obtaining the raw Docker leaves.  This helper
normalizes only the bytes supplied to it and fails instead of using PAX/GNU
output extensions when a path or member size cannot be represented by USTAR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Mapping, Sequence


# Do not write bytecode for sibling imports when this helper is embedded in an
# isolated host-runner invocation.  Direct execution is also covered by -B.
sys.dont_write_bytecode = True


CAPTURE_INVOCATION_VERSION = "riley.reconstructed-runtime-assembly-build-invocation.v1"
OCI_EXPORT_INVOCATION_VERSION = "riley.reconstructed-runtime-assembly-oci-export-invocation.v1"
CAPTURE_COMPLETION_VERSION = "riley.reconstructed-runtime-assembly-capture-completion.v1"
CONTEXT_ARCHIVE_FORMAT = "ustar-v1"
OCI_ARCHIVE_FORMAT = "oci-image-layout-tar.v1"
PLATFORM = {"os": "linux", "architecture": "amd64"}
CAPTURE_MEMBER_NAMES = (
    "SHA256SUMS",
    "build.iid",
    "build.log",
    "capture-completion.json",
    "capture-invocation.json",
    "container-inspect.json",
    "container-opt-riley.tar",
    "context.tar",
    "image-inspect.json",
    "oci-export-invocation.json",
    "oci-image-layout.tar",
)
CAPTURE_COMPLETION_MEMBER_NAMES = tuple(
    name for name in CAPTURE_MEMBER_NAMES if name not in {"SHA256SUMS", "capture-completion.json"}
)
CONTEXT_MEMBER_NAMES = ("Dockerfile", "input/riley", "input/riley.tar.gz")
MAX_USTAR_REGULAR_MEMBER_BYTES = 8**11 - 1
MAX_DOCKERFILE_BYTES = 1024 * 1024
MAX_BINARY_BYTES = 512 * 1024 * 1024
MAX_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
MAX_BUILD_LOG_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_RUNTIME_TREE_BYTES = 2 * 1024 * 1024 * 1024
MAX_OCI_ARCHIVE_BYTES = MAX_USTAR_REGULAR_MEMBER_BYTES
# Keep the raw Docker-save stream bounded by the exact input limit accepted by
# the OCI normalizer.  The extra allowance is for Docker-save layout metadata
# that the normalizer removes before writing the canonical OCI archive.
MAX_IMAGE_EXPORT_ARCHIVE_BYTES = MAX_OCI_ARCHIVE_BYTES + 64 * 1024 * 1024
MAX_RUNTIME_TREE_MEMBERS = 8192
TAR_BLOCK_BYTES = 512
TAR_SIZE_START = 124
TAR_SIZE_END = 136
TAR_TYPE_OFFSET = 156
MAX_TAR_ZERO_TRAILER_BYTES = 20 * TAR_BLOCK_BYTES
MAX_CONTEXT_ARCHIVE_BYTES = MAX_BINARY_BYTES + MAX_BUNDLE_BYTES + MAX_DOCKERFILE_BYTES + 4096
MAX_CAPTURE_ARCHIVE_BYTES = (
    MAX_RUNTIME_TREE_BYTES
    + MAX_OCI_ARCHIVE_BYTES
    + MAX_CONTEXT_ARCHIVE_BYTES
    + MAX_BUILD_LOG_BYTES
    + 65536
)
MAX_ID_BYTES = 80
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class AssemblyCaptureComposeError(ValueError):
    """The raw host-runner inputs cannot form a closed USTAR capture."""


def _fail(message: str) -> None:
    raise AssemblyCaptureComposeError(message)


def _required_flag(name: str) -> int:
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        _fail(f"requires os.{name}")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _absolute_path(value: Path, label: str) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\x00" in raw or "\n" in raw or "\r" in raw:
        _fail(f"{label} must be a nonempty single-line path")
    if not os.path.isabs(raw) or raw.startswith("//") or raw == os.path.sep or os.path.normpath(raw) != raw:
        _fail(f"{label} must be a normalized non-root absolute path")
    return Path(raw)


def _sha256(value: str, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be a non-zero lowercase SHA-256")
    return value


def _image_id(value: str, label: str) -> str:
    if type(value) is not str or IMAGE_ID_RE.fullmatch(value) is None or value == "sha256:" + "0" * 64:
        _fail(f"{label} must be a non-zero lowercase sha256 image ID")
    return value


def _container_id(value: str, label: str) -> str:
    if type(value) is not str or CONTAINER_ID_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be a non-zero lowercase 64-hex container ID")
    return value


def _reconstruction_id(value: str) -> str:
    if type(value) is not str or value not in {"a", "b"}:
        _fail("--reconstruction-id must be exactly a or b")
    return value


def _revision(value: str) -> str:
    if type(value) is not str or REVISION_RE.fullmatch(value) is None or value == "0" * 40:
        _fail("--source-revision must be a non-zero lowercase 40-hex Git revision")
    return value


@dataclass(frozen=True)
class FileState:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class FileDigest:
    sha256: str
    byte_length: int


@dataclass
class HeldInput:
    """One no-follow input held across digesting and USTAR serialization."""

    descriptor: int
    state: FileState
    label: str
    maximum_bytes: int
    allow_empty: bool = False

    @classmethod
    def open(
        cls,
        path: Path,
        label: str,
        maximum_bytes: int,
        *,
        allow_empty: bool = False,
    ) -> "HeldInput":
        descriptor, state = _open_input(path, label, maximum_bytes)
        return cls(descriptor, state, label, maximum_bytes, allow_empty)

    def _rewind(self) -> None:
        try:
            os.lseek(self.descriptor, 0, os.SEEK_SET)
        except OSError as error:
            _fail(f"cannot rewind {self.label}: {error}")

    def digest(self) -> FileDigest:
        self._rewind()
        digest = hashlib.sha256()
        try:
            duplicate = os.dup(self.descriptor)
        except OSError as error:
            _fail(f"cannot duplicate held {self.label}: {error}")
        try:
            with os.fdopen(duplicate, "rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        except OSError as error:
            _fail(f"cannot hash held {self.label}: {error}")
        self.verify()
        if self.state.size == 0 and not self.allow_empty:
            _fail(f"{self.label} must not be empty")
        return FileDigest(digest.hexdigest(), self.state.size)

    def read_all(self) -> bytes:
        self._rewind()
        try:
            duplicate = os.dup(self.descriptor)
        except OSError as error:
            _fail(f"cannot duplicate held {self.label}: {error}")
        try:
            with os.fdopen(duplicate, "rb") as stream:
                raw = stream.read(self.maximum_bytes + 1)
        except OSError as error:
            _fail(f"cannot read held {self.label}: {error}")
        self.verify()
        if len(raw) != self.state.size or len(raw) > self.maximum_bytes:
            _fail(f"{self.label} changed while it was read")
        return raw

    def tar_stream(self) -> BinaryIO:
        self._rewind()
        try:
            return os.fdopen(os.dup(self.descriptor), "rb")
        except OSError as error:
            _fail(f"cannot duplicate held {self.label}: {error}")

    def verify(self) -> None:
        _verify_unchanged(self.descriptor, self.state, self.label)

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError as error:
            _fail(f"cannot close held {self.label}: {error}")


def _state(metadata: os.stat_result) -> FileState:
    return FileState(
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_input(path: Path, label: str, maximum_bytes: int) -> tuple[int, FileState]:
    path = _absolute_path(path, label)
    try:
        before = os.lstat(path)
    except OSError as error:
        _fail(f"cannot inspect {label}: {error}")
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 0
        or before.st_size > maximum_bytes
    ):
        _fail(f"{label} must be a single-link regular file within its byte bound")
    flags = os.O_RDONLY | _required_flag("O_CLOEXEC") | _required_flag("O_NOFOLLOW") | _required_flag("O_NONBLOCK")
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _fail(f"cannot open {label} safely: {error}")
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        _fail(f"cannot inspect opened {label}: {error}")
    if _state(opened) != _state(before):
        os.close(descriptor)
        _fail(f"{label} changed while it was opened")
    return descriptor, _state(opened)


def _verify_unchanged(descriptor: int, expected: FileState, label: str) -> None:
    try:
        current = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot recheck {label}: {error}")
    if _state(current) != expected:
        _fail(f"{label} changed while it was read")


def _digest_input(path: Path, label: str, maximum_bytes: int, *, allow_empty: bool = False) -> FileDigest:
    descriptor, expected = _open_input(path, label, maximum_bytes)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        _verify_unchanged(descriptor, expected, label)
    finally:
        os.close(descriptor)
    if expected.size == 0 and not allow_empty:
        _fail(f"{label} must not be empty")
    return FileDigest(digest.hexdigest(), expected.size)


def _require_private_parent(path: Path, label: str) -> tuple[int, str]:
    path = _absolute_path(path, label)
    parent = path.parent
    name = path.name
    if name in {"", ".", ".."} or "/" in name:
        _fail(f"{label} has an unsafe output filename")
    try:
        before = os.lstat(parent)
    except OSError as error:
        _fail(f"cannot inspect {label} parent: {error}")
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        _fail(f"{label} parent must be an effective-UID-owned mode-0700 non-symlink directory")
    flags = os.O_RDONLY | _required_flag("O_DIRECTORY") | _required_flag("O_CLOEXEC") | _required_flag("O_NOFOLLOW")
    try:
        descriptor = os.open(parent, flags)
    except OSError as error:
        _fail(f"cannot open {label} parent safely: {error}")
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        _fail(f"cannot inspect opened {label} parent: {error}")
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
    ):
        os.close(descriptor)
        _fail(f"{label} parent changed while it was opened")
    return descriptor, name


def _open_output(path: Path, label: str) -> tuple[int, int]:
    parent_fd, name = _require_private_parent(path, label)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _required_flag("O_CLOEXEC") | _required_flag("O_NOFOLLOW")
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
        except OSError as error:
            _fail(f"cannot create {label} exactly once: {error}")
    except BaseException:
        os.close(parent_fd)
        raise
    return descriptor, parent_fd


def _write_tar(path: Path, label: str, rows: Iterable[tuple[tarfile.TarInfo, BinaryIO | None]]) -> None:
    descriptor, parent_fd = _open_output(path, label)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for member, source in rows:
                    archive.addfile(member, source)
            stream.flush()
            os.fsync(descriptor)
            os.fsync(parent_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600:
            _fail(f"{label} was not created as a mode-0600 single-link regular file")
    except (OSError, tarfile.TarError, ValueError) as error:
        _fail(f"cannot write {label} as USTAR: {error}")
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _write_stream(path: Path, label: str, maximum_bytes: int) -> FileDigest:
    """Copy stdin once into a create-only private output with a hard cap."""

    output = _absolute_path(path, label)
    if type(maximum_bytes) is not int or maximum_bytes < 1 or maximum_bytes > MAX_IMAGE_EXPORT_ARCHIVE_BYTES:
        _fail(f"{label} has an invalid stream byte bound")
    descriptor, parent_fd = _open_output(output, label)
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            while True:
                try:
                    chunk = sys.stdin.buffer.read(1024 * 1024)
                except OSError as error:
                    _fail(f"cannot read stdin for {label}: {error}")
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    _fail(f"{label} exceeds its byte bound")
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(descriptor)
            os.fsync(parent_fd)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != total
        ):
            _fail(f"{label} was not created as a bounded mode-0600 single-link regular file")
    except OSError as error:
        _fail(f"cannot write bounded {label}: {error}")
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    return FileDigest(digest.hexdigest(), total)


def _tar_member(name: str, size: int, *, mode: int, uid: int, gid: int, directory: bool = False) -> tarfile.TarInfo:
    if type(name) is not str or not name or name.startswith("/") or "\\" in name:
        _fail(f"cannot represent unsafe USTAR member {name!r}")
    try:
        encoded_name = name.encode("ascii")
    except UnicodeEncodeError:
        _fail(f"cannot represent non-ASCII USTAR member {name!r}")
    # The replayer deliberately rejects the POSIX USTAR prefix field.  Do not
    # let tarfile transparently use it for a long but otherwise legal name.
    if len(encoded_name) > 100 or size < 0 or size > MAX_USTAR_REGULAR_MEMBER_BYTES:
        _fail(f"cannot represent unsafe or oversized USTAR member {name!r}")
    member = tarfile.TarInfo(name)
    member.mode = mode
    member.uid = uid
    member.gid = gid
    member.mtime = 0
    member.uname = ""
    member.gname = ""
    if directory:
        member.type = tarfile.DIRTYPE
        member.size = 0
    else:
        member.type = tarfile.REGTYPE
        member.size = size
    return member


def _bytes_member(name: str, raw: bytes, *, mode: int = 0o644, uid: int = 0, gid: int = 0) -> tuple[tarfile.TarInfo, BinaryIO]:
    import io

    return _tar_member(name, len(raw), mode=mode, uid=uid, gid=gid), io.BytesIO(raw)


def _check_no_overlap(outputs: Sequence[Path], inputs: Sequence[Path]) -> None:
    normalized_outputs = [_absolute_path(path, "output") for path in outputs]
    normalized_inputs = [_absolute_path(path, "input") for path in inputs]
    if len(set(normalized_outputs)) != len(normalized_outputs):
        _fail("output paths must be distinct")
    for output in normalized_outputs:
        for input_path in normalized_inputs:
            if output == input_path or output in input_path.parents or input_path in output.parents:
                _fail("output paths must not overlap input paths")


def _context(args: argparse.Namespace) -> dict[str, object]:
    output = _absolute_path(args.output, "--output")
    dockerfile = _absolute_path(args.dockerfile, "--dockerfile")
    binary = _absolute_path(args.release_binary, "--release-binary")
    bundle = _absolute_path(args.release_bundle, "--release-bundle")
    _check_no_overlap([output], [dockerfile, binary, bundle])
    expected_dockerfile = _sha256(args.dockerfile_sha256, "--dockerfile-sha256")
    expected_binary = _sha256(args.release_binary_sha256, "--release-binary-sha256")
    expected_bundle = _sha256(args.release_bundle_sha256, "--release-bundle-sha256")
    held = {
        "Dockerfile": HeldInput.open(dockerfile, "Dockerfile", MAX_DOCKERFILE_BYTES),
        "input/riley": HeldInput.open(binary, "release binary", MAX_BINARY_BYTES),
        "input/riley.tar.gz": HeldInput.open(bundle, "release bundle", MAX_BUNDLE_BYTES),
    }
    streams: list[BinaryIO] = []
    try:
        dockerfile_digest = held["Dockerfile"].digest()
        binary_digest = held["input/riley"].digest()
        bundle_digest = held["input/riley.tar.gz"].digest()
        if (
            dockerfile_digest.sha256 != expected_dockerfile
            or binary_digest.sha256 != expected_binary
            or bundle_digest.sha256 != expected_bundle
        ):
            _fail("Dockerfile, release binary, or bundle differs from its supplied SHA-256")
        rows: list[tuple[tarfile.TarInfo, BinaryIO | None]] = []
        for name in CONTEXT_MEMBER_NAMES:
            source = held[name].tar_stream()
            streams.append(source)
            member = _tar_member(name, held[name].state.size, mode=0o644, uid=0, gid=0)
            rows.append((member, source))
        _write_tar(output, "source-free build context", rows)
    finally:
        for stream in streams:
            stream.close()
        for value in held.values():
            value.verify()
            value.close()
    result = _digest_input(output, "source-free build context", MAX_CONTEXT_ARCHIVE_BYTES)
    return {
        "schema_version": "riley.reconstructed-runtime-assembly-context.v1",
        "status": "composed",
        "context": {"sha256": result.sha256, "byte_length": result.byte_length},
        "members": {
            "Dockerfile": {"sha256": dockerfile_digest.sha256, "byte_length": dockerfile_digest.byte_length},
            "input/riley": {"sha256": binary_digest.sha256, "byte_length": binary_digest.byte_length},
            "input/riley.tar.gz": {"sha256": bundle_digest.sha256, "byte_length": bundle_digest.byte_length},
        },
    }


def _safe_tar_name(name: str, label: str) -> tuple[str, ...]:
    if type(name) is not str or not name or "\x00" in name or "\\" in name or name.startswith("/"):
        _fail(f"{label} has an unsafe path")
    try:
        name.encode("ascii")
    except UnicodeEncodeError:
        _fail(f"{label} must be ASCII for the no-prefix USTAR contract")
    normalized = name.rstrip("/")
    if normalized in {"", "."}:
        return ()
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        _fail(f"{label} has a traversal path")
    return tuple(parts)


def _parse_runtime_tar_size(raw: bytes) -> int:
    if len(raw) != TAR_SIZE_END - TAR_SIZE_START:
        _fail("raw docker cp tar has an invalid size field")
    # POSIX base-256 sizes and extension records are deliberately outside the
    # fixed USTAR producer boundary.  Reject them before tarfile can allocate
    # a PAX/GNU extension payload.
    if raw and raw[0] & 0x80:
        _fail("raw docker cp tar uses an unsupported non-octal size field")
    text = raw.rstrip(b"\0 ").lstrip(b" ")
    if not text:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in text):
        _fail("raw docker cp tar has an invalid size field")
    return int(text, 8)


def _padded_tar_size(size: int) -> int:
    return ((size + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES) * TAR_BLOCK_BYTES


def _preflight_runtime_tree_tar(stream: BinaryIO) -> None:
    """Reject extension payloads before :mod:`tarfile` parses raw Docker bytes."""

    try:
        stream.seek(0, os.SEEK_END)
        archive_size = stream.tell()
        stream.seek(0, os.SEEK_SET)
    except OSError as error:
        _fail(f"cannot inspect raw docker cp tar before parsing: {error}")
    if archive_size < 2 * TAR_BLOCK_BYTES or archive_size > MAX_RUNTIME_TREE_BYTES:
        _fail("raw docker cp tar has an invalid byte length")
    if archive_size % TAR_BLOCK_BYTES:
        _fail("raw docker cp tar must be tar-block aligned")
    offset = 0
    member_count = 0
    payload_total = 0
    zero_block = b"\0" * TAR_BLOCK_BYTES
    try:
        while offset < archive_size:
            header = stream.read(TAR_BLOCK_BYTES)
            if len(header) != TAR_BLOCK_BYTES:
                _fail("raw docker cp tar header is truncated")
            offset += TAR_BLOCK_BYTES
            if header == zero_block:
                second = stream.read(TAR_BLOCK_BYTES)
                if second != zero_block:
                    _fail("raw docker cp tar has an invalid end marker")
                offset += TAR_BLOCK_BYTES
                remaining = archive_size - offset
                if remaining > MAX_TAR_ZERO_TRAILER_BYTES:
                    _fail("raw docker cp tar has an oversized zero trailer")
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        _fail("raw docker cp tar trailer is truncated")
                    if any(chunk):
                        _fail("raw docker cp tar has nonzero trailing bytes")
                    remaining -= len(chunk)
                stream.seek(0, os.SEEK_SET)
                return
            member_type = header[TAR_TYPE_OFFSET : TAR_TYPE_OFFSET + 1]
            if member_type not in {b"\0", b"0", b"5"}:
                _fail("raw docker cp tar uses a PAX/GNU/link/sparse/special extension")
            member_size = _parse_runtime_tar_size(header[TAR_SIZE_START:TAR_SIZE_END])
            if member_type == b"5" and member_size != 0:
                _fail("raw docker cp tar directory carries payload bytes")
            member_count += 1
            if member_count > MAX_RUNTIME_TREE_MEMBERS:
                _fail("raw docker cp tar exceeds its runtime member inventory bound")
            if member_type != b"5":
                if member_size > MAX_USTAR_REGULAR_MEMBER_BYTES:
                    _fail("raw docker cp tar has an unrepresentable runtime member size")
                payload_total += member_size
                if payload_total > MAX_RUNTIME_TREE_BYTES:
                    _fail("raw docker cp tar exceeds the runtime-tree byte bound")
            padded = _padded_tar_size(member_size)
            if padded > archive_size - offset:
                _fail("raw docker cp tar payload is truncated")
            stream.seek(padded, os.SEEK_CUR)
            offset += padded
    except OSError as error:
        _fail(f"cannot preflight raw docker cp tar: {error}")
    _fail("raw docker cp tar has no two-block zero end marker")


@dataclass(frozen=True)
class RuntimeEntry:
    source: tarfile.TarInfo
    name: str
    directory: bool


def _runtime_tree(args: argparse.Namespace) -> dict[str, object]:
    source_path = _absolute_path(args.input, "--input")
    output = _absolute_path(args.output, "--output")
    _check_no_overlap([output], [source_path])
    descriptor, expected = _open_input(source_path, "raw docker cp tar", MAX_RUNTIME_TREE_BYTES)
    entries: dict[str, RuntimeEntry] = {}
    stream: BinaryIO | None = None
    archive: tarfile.TarFile | None = None
    try:
        stream = os.fdopen(descriptor, "rb", closefd=False)
        _preflight_runtime_tree_tar(stream)
        try:
            archive = tarfile.open(fileobj=stream, mode="r:")
        except tarfile.TarError as error:
            _fail(f"raw docker cp tar is not an uncompressed tar archive: {error}")
        # ``TarFile.getmembers()`` scans the entire archive and retains every
        # TarInfo before a caller can apply the inventory cap.  Docker's raw
        # stream is hostile input here, so stop at member 8193 instead of
        # allocating millions of empty-header records from its byte allowance.
        raw_rows: list[tarfile.TarInfo] = []
        while True:
            member = archive.next()
            if member is None:
                break
            if len(raw_rows) >= MAX_RUNTIME_TREE_MEMBERS:
                _fail("raw docker cp tar exceeds its runtime member inventory bound")
            raw_rows.append(member)
        if not raw_rows:
            _fail("raw docker cp tar must contain a bounded nonempty inventory")
        parsed = [(member, _safe_tar_name(member.name, "raw docker cp member")) for member in raw_rows]
        nonempty = [parts for _member, parts in parsed if parts]
        root = None
        if nonempty:
            candidate = nonempty[0][0]
            has_explicit_root = any(parts == (candidate,) and member.isdir() for member, parts in parsed)
            if has_explicit_root and all(parts and parts[0] == candidate for parts in nonempty):
                root = candidate
        for member, parts in parsed:
            if not parts:
                continue
            if root is not None:
                if parts == (root,):
                    continue
                parts = parts[1:]
            if not parts:
                continue
            if not (member.isdir() or member.isreg()):
                _fail("raw docker cp tar must contain only regular files and directories")
            name = "/".join(parts)
            if name in entries:
                _fail("raw docker cp tar has duplicate runtime paths")
            if member.size < 0 or member.size > MAX_USTAR_REGULAR_MEMBER_BYTES:
                _fail("raw docker cp tar has an unrepresentable runtime member size")
            entries[name] = RuntimeEntry(member, name, member.isdir())
        if not entries:
            _fail("raw docker cp tar has no runtime tree after its optional root is removed")
        total = sum(0 if entry.directory else entry.source.size for entry in entries.values())
        if total > MAX_RUNTIME_TREE_BYTES:
            _fail("raw docker cp tar exceeds the runtime-tree byte bound")
        rows: list[tuple[tarfile.TarInfo, BinaryIO | None]] = []
        for name in sorted(entries):
            entry = entries[name]
            if entry.directory:
                rows.append((_tar_member(name, 0, mode=entry.source.mode, uid=65532, gid=65532, directory=True), None))
            else:
                payload = archive.extractfile(entry.source)
                if payload is None:
                    _fail(f"cannot read raw docker cp member {name!r}")
                rows.append((_tar_member(name, entry.source.size, mode=entry.source.mode, uid=65532, gid=65532), payload))
        _write_tar(output, "canonical runtime-tree archive", rows)
        for _member, payload in rows:
            if payload is not None:
                payload.close()
        _verify_unchanged(descriptor, expected, "raw docker cp tar")
    except (OSError, tarfile.TarError) as error:
        _fail(f"cannot canonicalize raw docker cp tar: {error}")
    finally:
        if archive is not None:
            archive.close()
        if stream is not None:
            stream.close()
        os.close(descriptor)
    digest = _digest_input(output, "canonical runtime-tree archive", MAX_RUNTIME_TREE_BYTES)
    return {
        "schema_version": "riley.reconstructed-runtime-assembly-runtime-tree.v1",
        "status": "composed",
        "runtime_tree": {"sha256": digest.sha256, "byte_length": digest.byte_length, "entry_count": len(entries)},
    }


@dataclass(frozen=True)
class CaptureInput:
    name: str
    path: Path | None
    raw: bytes | None
    maximum_bytes: int
    allow_empty: bool = False


def _raw_capture_input_digest(value: CaptureInput) -> FileDigest:
    if value.raw is None:
        _fail(f"capture member {value.name!r} is missing raw bytes")
    if len(value.raw) > value.maximum_bytes or (not value.allow_empty and not value.raw):
        _fail(f"capture member {value.name!r} is empty or oversized")
    return FileDigest(hashlib.sha256(value.raw).hexdigest(), len(value.raw))


def _hold_capture_inputs(values: Mapping[str, CaptureInput]) -> dict[str, HeldInput]:
    held: dict[str, HeldInput] = {}
    try:
        for name, value in values.items():
            if value.path is None:
                continue
            held[name] = HeldInput.open(
                value.path,
                f"capture member {name}",
                value.maximum_bytes,
                allow_empty=value.allow_empty,
            )
    except BaseException:
        for input_value in held.values():
            try:
                input_value.close()
            except AssemblyCaptureComposeError:
                pass
        raise
    return held


def _read_id(args: argparse.Namespace) -> str:
    input_path = _absolute_path(args.input, "--input")
    kind = args.kind
    if kind not in {"image", "container"}:
        _fail("--kind must be image or container")
    held = HeldInput.open(input_path, f"Docker {kind} ID", MAX_ID_BYTES)
    try:
        raw = held.read_all()
        if kind == "image":
            try:
                return _image_id(raw.decode("ascii"), "Docker --iidfile")
            except UnicodeDecodeError:
                _fail("Docker --iidfile must be ASCII and contain no trailing newline")
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            _fail("Docker create output must be one lowercase container ID followed by one newline")
        try:
            return _container_id(raw[:-1].decode("ascii"), "Docker create output")
        except UnicodeDecodeError:
            _fail("Docker create output must be ASCII")
    finally:
        held.close()


def _stream(args: argparse.Namespace) -> dict[str, object]:
    maximum_bytes = args.maximum_bytes
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        _fail("--maximum-bytes must be a positive integer")
    result = _write_stream(_absolute_path(args.output, "--output"), "bounded stream output", maximum_bytes)
    return {
        "schema_version": "riley.reconstructed-runtime-assembly-bounded-stream.v1",
        "status": "captured",
        "output": {"sha256": result.sha256, "byte_length": result.byte_length},
    }


def _capture(args: argparse.Namespace) -> dict[str, object]:
    output = _absolute_path(args.output, "--output")
    context = _absolute_path(args.context, "--context")
    runtime_tree = _absolute_path(args.runtime_tree, "--runtime-tree")
    build_iid = _absolute_path(args.build_iid, "--build-iid")
    build_log = _absolute_path(args.build_log, "--build-log")
    image_inspect = _absolute_path(args.image_inspect, "--image-inspect")
    oci_archive = _absolute_path(args.oci_archive, "--oci-archive")
    container_inspect = _absolute_path(args.container_inspect, "--container-inspect")
    inputs = [context, runtime_tree, build_iid, build_log, image_inspect, oci_archive, container_inspect]
    _check_no_overlap([output], inputs)
    reconstruction_id = _reconstruction_id(args.reconstruction_id)
    source_revision = _revision(args.source_revision)
    source_archive_sha256 = _sha256(args.expected_source_archive_sha256, "--expected-source-archive-sha256")
    repro_inputs_sha256 = _sha256(args.repro_build_inputs_sha256, "--repro-build-inputs-sha256")
    binary_sha256 = _sha256(args.release_binary_sha256, "--release-binary-sha256")
    bundle_sha256 = _sha256(args.release_bundle_sha256, "--release-bundle-sha256")
    recipe_sha256 = _sha256(args.recipe_normalized_instructions_sha256, "--recipe-normalized-instructions-sha256")
    image_id = _image_id(args.image_id, "--image-id")
    container_id = _container_id(args.container_id, "--container-id")
    sources = {
        "build.iid": CaptureInput("build.iid", build_iid, None, 80),
        "build.log": CaptureInput("build.log", build_log, None, MAX_BUILD_LOG_BYTES, allow_empty=True),
        "container-inspect.json": CaptureInput("container-inspect.json", container_inspect, None, MAX_JSON_BYTES),
        "container-opt-riley.tar": CaptureInput("container-opt-riley.tar", runtime_tree, None, MAX_RUNTIME_TREE_BYTES),
        "context.tar": CaptureInput("context.tar", context, None, MAX_CONTEXT_ARCHIVE_BYTES),
        "image-inspect.json": CaptureInput("image-inspect.json", image_inspect, None, MAX_JSON_BYTES),
        "oci-image-layout.tar": CaptureInput("oci-image-layout.tar", oci_archive, None, MAX_OCI_ARCHIVE_BYTES),
    }
    held = _hold_capture_inputs(sources)
    opened: list[BinaryIO] = []
    try:
        iid_raw = held["build.iid"].read_all()
        if iid_raw != image_id.encode("ascii"):
            _fail("build iidfile must equal the exact --image-id without a trailing newline")
        context_digest = held["context.tar"].digest()
        invocation = _canonical_json(
            {
                "schema_version": CAPTURE_INVOCATION_VERSION,
                "argv": [
                    "docker", "build", "--file", "Dockerfile", "--platform", "linux/amd64", "--network", "none",
                    "--pull=false", "--no-cache", "--iidfile", "build.iid",
                    "--build-arg", f"RILEY_RECONSTRUCTION_ID={reconstruction_id}",
                    "--build-arg", f"RILEY_SOURCE_REVISION={source_revision}",
                    "--build-arg", f"RILEY_SOURCE_ARCHIVE_SHA256={source_archive_sha256}",
                    "--build-arg", f"RILEY_REPRO_BUILD_INPUTS_SHA256={repro_inputs_sha256}",
                    "--build-arg", f"RILEY_RELEASE_BINARY_SHA256={binary_sha256}",
                    "--build-arg", f"RILEY_RELEASE_BUNDLE_SHA256={bundle_sha256}",
                    "--build-arg", f"RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256={recipe_sha256}",
                    "-",
                ],
                "stdin": {
                    "member": "context.tar",
                    "format": CONTEXT_ARCHIVE_FORMAT,
                    "sha256": context_digest.sha256,
                    "byte_length": context_digest.byte_length,
                },
            }
        )
        oci_invocation = _canonical_json(
            {
                "schema_version": OCI_EXPORT_INVOCATION_VERSION,
                "source_image_id": image_id,
                "output_member": "oci-image-layout.tar",
                "format": OCI_ARCHIVE_FORMAT,
                "platform": PLATFORM,
            }
        )
        outer_sources: dict[str, CaptureInput] = {
            "SHA256SUMS": CaptureInput("SHA256SUMS", None, b"", MAX_JSON_BYTES),
            "capture-invocation.json": CaptureInput("capture-invocation.json", None, invocation, MAX_JSON_BYTES),
            "capture-completion.json": CaptureInput("capture-completion.json", None, b"", MAX_JSON_BYTES),
            "oci-export-invocation.json": CaptureInput("oci-export-invocation.json", None, oci_invocation, MAX_JSON_BYTES),
            **sources,
        }
        digests = {
            name: (
                _raw_capture_input_digest(source)
                if source.raw is not None
                else held[name].digest()
            )
            for name, source in outer_sources.items()
            if name not in {"SHA256SUMS", "capture-completion.json"}
        }
        completion = _canonical_json(
            {
                "schema_version": CAPTURE_COMPLETION_VERSION,
                "reconstruction_id": reconstruction_id,
                "image_id": image_id,
                "container_id": container_id,
                "container_state": "created",
                "container_started": False,
                "members": {
                    name: {"sha256": digests[name].sha256, "byte_length": digests[name].byte_length}
                    for name in CAPTURE_COMPLETION_MEMBER_NAMES
                },
            }
        )
        completion_digest = FileDigest(hashlib.sha256(completion).hexdigest(), len(completion))
        checksum_digests = {**digests, "capture-completion.json": completion_digest}
        checksums = b"".join(
            f"{checksum_digests[name].sha256}  {name}\n".encode("ascii") for name in CAPTURE_MEMBER_NAMES[1:]
        )
        outer_sources["SHA256SUMS"] = CaptureInput("SHA256SUMS", None, checksums, MAX_JSON_BYTES)
        outer_sources["capture-completion.json"] = CaptureInput(
            "capture-completion.json", None, completion, MAX_JSON_BYTES
        )
        outer_digests = {
            **digests,
            "SHA256SUMS": _raw_capture_input_digest(outer_sources["SHA256SUMS"]),
            "capture-completion.json": completion_digest,
        }
        rows: list[tuple[tarfile.TarInfo, BinaryIO | None]] = []
        for name in CAPTURE_MEMBER_NAMES:
            source = outer_sources[name]
            if source.raw is not None:
                rows.append(_bytes_member(name, source.raw))
                continue
            input_value = held[name]
            stream = input_value.tar_stream()
            opened.append(stream)
            member = _tar_member(name, input_value.state.size, mode=0o644, uid=0, gid=0)
            rows.append((member, stream))
        _write_tar(output, "runtime assembly capture", rows)
        for input_value in held.values():
            input_value.verify()
    finally:
        for stream in opened:
            stream.close()
        for input_value in held.values():
            input_value.close()
    result = _digest_input(output, "runtime assembly capture", MAX_CAPTURE_ARCHIVE_BYTES)
    return {
        "schema_version": "riley.reconstructed-runtime-assembly-raw-capture-composer.v1",
        "status": "composed",
        "capture": {"sha256": result.sha256, "byte_length": result.byte_length},
        "image_id": image_id,
        "container_id": container_id,
        "members": {
            name: {"sha256": outer_digests[name].sha256, "byte_length": outer_digests[name].byte_length}
            for name in CAPTURE_MEMBER_NAMES
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    context = subcommands.add_parser("context", help="write the exact three-member source-free Docker context")
    context.add_argument("--output", type=Path, required=True)
    context.add_argument("--dockerfile", type=Path, required=True)
    context.add_argument("--dockerfile-sha256", required=True)
    context.add_argument("--release-binary", type=Path, required=True)
    context.add_argument("--release-binary-sha256", required=True)
    context.add_argument("--release-bundle", type=Path, required=True)
    context.add_argument("--release-bundle-sha256", required=True)
    runtime = subcommands.add_parser("runtime-tree", help="canonicalize a Docker cp tar into USTAR")
    runtime.add_argument("--input", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)
    stream = subcommands.add_parser("stream", help="copy stdin to a private create-only file with a hard byte bound")
    stream.add_argument("--output", type=Path, required=True)
    stream.add_argument("--maximum-bytes", type=int, required=True)
    read_id = subcommands.add_parser("read-id", help="validate a raw Docker ID file and emit only the canonical ID")
    read_id.add_argument("--kind", choices=("image", "container"), required=True)
    read_id.add_argument("--input", type=Path, required=True)
    capture = subcommands.add_parser("capture", help="write the fixed eleven-member raw capture USTAR")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--context", type=Path, required=True)
    capture.add_argument("--runtime-tree", type=Path, required=True)
    capture.add_argument("--build-iid", type=Path, required=True)
    capture.add_argument("--build-log", type=Path, required=True)
    capture.add_argument("--image-inspect", type=Path, required=True)
    capture.add_argument("--oci-archive", type=Path, required=True)
    capture.add_argument("--container-inspect", type=Path, required=True)
    capture.add_argument("--reconstruction-id", required=True)
    capture.add_argument("--source-revision", required=True)
    capture.add_argument("--expected-source-archive-sha256", required=True)
    capture.add_argument("--repro-build-inputs-sha256", required=True)
    capture.add_argument("--release-binary-sha256", required=True)
    capture.add_argument("--release-bundle-sha256", required=True)
    capture.add_argument("--recipe-normalized-instructions-sha256", required=True)
    capture.add_argument("--image-id", required=True)
    capture.add_argument("--container-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "context":
            report = _context(args)
        elif args.command == "runtime-tree":
            report = _runtime_tree(args)
        elif args.command == "stream":
            report = _stream(args)
        elif args.command == "read-id":
            sys.stdout.write(_read_id(args))
            return 0
        else:
            report = _capture(args)
    except AssemblyCaptureComposeError as error:
        print(f"runtime assembly capture composition failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(_canonical_json(report) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
