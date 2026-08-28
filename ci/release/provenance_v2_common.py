#!/usr/bin/env python3
"""Strict, stdlib-only filesystem primitives for C02-P1 provenance v2.

This module deliberately owns only mechanical evidence safety.  It does not
accept a qualification decision, start a process, or infer a candidate
binding.  Callers supply those semantics after they have captured exact raw
bytes through these primitives.

The helpers are Linux/POSIX-oriented and fail closed if the kernel interfaces
needed to prevent link traversal are unavailable.  In particular, a platform
without both ``O_NOFOLLOW`` and ``O_DIRECTORY`` is not a supported evidence
producer or verifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence


DEFAULT_MAX_JSON_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_BYTES = 1 << 42
DEFAULT_READ_CHUNK_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_LEAF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class ProvenanceV2Error(ValueError):
    """An input, descriptor, or filesystem operation is unsafe for evidence."""


def _fail(code: str, message: str) -> NoReturn:
    error = ProvenanceV2Error(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _required_open_flag(name: str) -> int:
    """Return one mandatory OS open flag or reject unsupported hosts.

    ``getattr(..., 0)`` is intentionally not used: silently omitting one of
    these flags would turn a verifier into a link-following verifier.
    """

    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        _fail("missing-open-safety-flag", f"host does not expose required {name}")
    return value


def require_safe_open_flags() -> tuple[int, int, int, int]:
    """Check the mandatory flags and return no-follow read/directory flags.

    ``O_CLOEXEC`` and ``O_NONBLOCK`` are also required so a raced replacement
    with a FIFO cannot block an evidence reader and descriptors do not leak to
    a later child process.  Linux provides all four flags used here.
    """

    nofollow = _required_open_flag("O_NOFOLLOW")
    directory = _required_open_flag("O_DIRECTORY")
    cloexec = _required_open_flag("O_CLOEXEC")
    nonblock = _required_open_flag("O_NONBLOCK")
    return nofollow, directory, cloexec, nonblock


def _file_open_flags() -> int:
    nofollow, _directory, cloexec, nonblock = require_safe_open_flags()
    return os.O_RDONLY | nofollow | cloexec | nonblock


def _directory_open_flags() -> int:
    nofollow, directory, cloexec, _nonblock = require_safe_open_flags()
    return os.O_RDONLY | nofollow | directory | cloexec


def _output_open_flags() -> int:
    nofollow, _directory, cloexec, _nonblock = require_safe_open_flags()
    return os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | cloexec


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate-json-key", f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _fail("non-finite-json-number", f"non-finite JSON number {value!r} is forbidden")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode exact canonical JSON used for evidence digests and files."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        _fail("unencodable-canonical-json", f"cannot encode canonical JSON: {error}")


def parse_strict_json(
    raw: bytes,
    label: str,
    *,
    maximum_bytes: int = DEFAULT_MAX_JSON_BYTES,
    require_object: bool = True,
) -> Any:
    """Parse finite, duplicate-key-free UTF-8 JSON without byte normalization.

    This is for source-owned raw JSON captures such as ``docker image
    inspect`` output.  Such tools commonly emit formatting whitespace, so a
    verifier must bind raw bytes with a descriptor and parse them strictly
    without requiring the producer to rewrite them canonically.
    """

    _validate_maximum(maximum_bytes, f"{label} maximum byte bound")
    if type(raw) is not bytes or not raw or len(raw) > maximum_bytes:
        _fail("invalid-json-byte-length", f"{label} has an invalid byte length")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        _fail("invalid-json", f"{label} is not strict UTF-8 JSON: {error}")
    except json.JSONDecodeError as error:
        _fail("invalid-json", f"{label} is not JSON: {error}")
    if require_object and not isinstance(decoded, dict):
        _fail("invalid-json-root", f"{label} root must be a JSON object")
    return decoded


def parse_canonical_json(
    raw: bytes,
    label: str,
    *,
    maximum_bytes: int = DEFAULT_MAX_JSON_BYTES,
    require_object: bool = True,
) -> Any:
    """Parse only exact canonical, finite, duplicate-key-free UTF-8 JSON.

    Exact byte equality with ``canonical_json_bytes`` rejects whitespace,
    alternate numeric spellings, escaped Unicode aliases, and trailing
    newlines.  Receipt/schema code can therefore bind a digest to the same
    bytes that this parser interpreted.
    """

    decoded = parse_strict_json(
        raw,
        label,
        maximum_bytes=maximum_bytes,
        require_object=require_object,
    )
    if raw != canonical_json_bytes(decoded):
        _fail("noncanonical-json", f"{label} must use exact canonical JSON bytes")
    return decoded


def _validate_maximum(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        _fail("invalid-byte-bound", f"{label} must be a non-negative integer")


def _stable_stat(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_regular_single_link(metadata: os.stat_result, label: str) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        _fail("unsafe-evidence-path", f"{label} must be a regular non-link file")
    if metadata.st_nlink != 1:
        _fail("nonunique-evidence-inode", f"{label} must have exactly one hard link")


def _validate_leaf_name(name: str, label: str) -> str:
    if type(name) is not str or SAFE_LEAF_RE.fullmatch(name) is None:
        _fail("invalid-evidence-name", f"{label} must be a normalized safe file name")
    if name in {".", ".."}:
        _fail("invalid-evidence-name", f"{label} must not be a path alias")
    return name


def validate_relative_path(value: str, label: str) -> str:
    """Require a non-empty, normalized POSIX relative evidence path."""

    if type(value) is not str or not value:
        _fail("invalid-relative-path", f"{label} must be a non-empty relative path")
    if (
        "\x00" in value
        or "\\" in value
        or "//" in value
        or SAFE_RELATIVE_RE.fullmatch(value) is None
    ):
        _fail("invalid-relative-path", f"{label} must be normalized POSIX text")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail("invalid-relative-path", f"{label} must not contain traversal or aliases")
    return value


def _absolute_components(path: Path, label: str) -> tuple[str, ...]:
    raw = os.fspath(path)
    if (
        not os.path.isabs(raw)
        or "\x00" in raw
        or "\\" in raw
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        _fail("invalid-absolute-path", f"{label} must be an absolute path")
    parts = Path(raw).parts
    if not parts or parts[0] != os.path.sep:
        _fail("invalid-absolute-path", f"{label} must be a normalized absolute path")
    components = tuple(parts[1:])
    if any(component in {"", ".", ".."} for component in components):
        _fail("invalid-absolute-path", f"{label} must not contain traversal or aliases")
    return components


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _require_directory_fd(directory_fd: int, label: str) -> os.stat_result:
    if type(directory_fd) is not int or directory_fd < 0:
        _fail("invalid-directory-fd", f"{label} must be an open directory descriptor")
    try:
        metadata = os.fstat(directory_fd)
    except OSError as error:
        _fail("invalid-directory-fd", f"{label} cannot be inspected: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("invalid-directory-fd", f"{label} must refer to a directory")
    return metadata


def open_absolute_directory(path: Path, label: str) -> int:
    """Open every absolute-directory component through no-follow directory FDs.

    The returned FD pins the directory inode for later ``openat`` operations;
    callers own it and must close it.
    """

    components = _absolute_components(path, label)
    flags = _directory_open_flags()
    try:
        current_fd = os.open(os.path.sep, flags)
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot open root for {label}: {error}")
    try:
        for component in components:
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as error:
                _fail(
                    "unsafe-evidence-directory",
                    f"cannot open {label} without following links: {error}",
                )
            os.close(current_fd)
            current_fd = child_fd
        _require_directory_fd(current_fd, label)
        return current_fd
    except BaseException:
        _close_quietly(current_fd)
        raise


def _validate_private_evidence_ancestor(metadata: os.stat_result, label: str) -> None:
    """Reject an unsafe writable parent while allowing a sticky trusted boundary."""

    if not stat.S_ISDIR(metadata.st_mode):
        _fail("unsafe-evidence-directory", f"{label} must be a directory")
    writable_by_others = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if not writable_by_others:
        return
    if not metadata.st_mode & stat.S_ISVTX:
        _fail(
            "unsafe-evidence-ancestor",
            f"{label} is group/world writable without a sticky boundary",
        )
    if metadata.st_uid not in {0, os.geteuid()}:
        _fail(
            "unsafe-evidence-ancestor",
            f"{label} is writable and not owned by root or the effective UID",
        )


def _validate_private_evidence_root(metadata: os.stat_result, label: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("unsafe-evidence-root", f"{label} must be a regular directory")
    if metadata.st_uid != os.geteuid():
        _fail("unsafe-evidence-root-owner", f"{label} must be owned by the effective UID")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        _fail("unsafe-evidence-root-mode", f"{label} mode must be exactly 0700")


def open_private_evidence_directory(path: Path, label: str) -> int:
    """Pin a private evidence root without accepting unsafe writable parents.

    This is intentionally stricter than ``open_absolute_directory``.  C02-P1
    receipts may be used as qualification evidence, so the terminal root must
    be an effective-UID-owned 0700 directory.  Every ancestor is opened with
    no-follow directory FDs; an ancestor writable by group/other is accepted
    only when it is a sticky boundary owned by root or the effective UID.
    """

    components = _absolute_components(path, label)
    flags = _directory_open_flags()
    try:
        current_fd = os.open(os.path.sep, flags)
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot open root for {label}: {error}")
    try:
        _validate_private_evidence_ancestor(os.fstat(current_fd), f"{label} ancestor /")
        for component in components:
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as error:
                _fail(
                    "unsafe-evidence-directory",
                    f"cannot open {label} component {component!r} without following links: {error}",
                )
            os.close(current_fd)
            current_fd = child_fd
            _validate_private_evidence_ancestor(
                os.fstat(current_fd), f"{label} ancestor {component!r}"
            )
        _validate_private_evidence_root(os.fstat(current_fd), label)
        return current_fd
    except BaseException:
        _close_quietly(current_fd)
        raise


def _open_relative_directory_chain(
    root_fd: int,
    components: Sequence[str],
    label: str,
) -> tuple[int, tuple[int, ...]]:
    """Open a relative directory chain without taking ownership of ``root_fd``."""

    _require_directory_fd(root_fd, f"{label} root")
    flags = _directory_open_flags()
    current_fd = root_fd
    owned: list[int] = []
    try:
        for component in components:
            _validate_leaf_name(component, f"{label} path component")
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as error:
                _fail(
                    "unsafe-evidence-directory",
                    f"cannot open {label} component {component!r} without following links: {error}",
                )
            owned.append(child_fd)
            current_fd = child_fd
        return current_fd, tuple(owned)
    except BaseException:
        for descriptor in reversed(owned):
            _close_quietly(descriptor)
        raise


def _read_exact_bounded(
    descriptor: int,
    initial_size: int,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if initial_size < 0 or initial_size > maximum_bytes:
        _fail("input-too-large", f"{label} exceeds its byte bound")
    chunks: list[bytes] = []
    remaining = initial_size
    while remaining:
        try:
            chunk = os.read(descriptor, min(DEFAULT_READ_CHUNK_BYTES, remaining))
        except OSError as error:
            _fail("unreadable-input", f"cannot read {label}: {error}")
        if not chunk:
            _fail("truncated-input", f"{label} changed while it was read")
        chunks.append(chunk)
        remaining -= len(chunk)
    try:
        if os.read(descriptor, 1):
            _fail("mutated-input", f"{label} grew while it was read")
    except OSError as error:
        _fail("unreadable-input", f"cannot re-read {label}: {error}")
    return b"".join(chunks)


def _read_regular_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    maximum_bytes: int,
) -> bytes:
    _validate_maximum(maximum_bytes, f"{label} maximum byte bound")
    _require_directory_fd(directory_fd, f"{label} parent")
    name = _validate_leaf_name(name, f"{label} name")
    try:
        before = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail("missing-input", f"cannot inspect {label}: {error}")
    _require_regular_single_link(before, label)
    if before.st_size > maximum_bytes:
        _fail("input-too-large", f"{label} exceeds its byte bound")
    try:
        descriptor = os.open(name, _file_open_flags(), dir_fd=directory_fd)
    except OSError as error:
        _fail("unsafe-evidence-path", f"cannot open {label} without following links: {error}")
    try:
        opened = os.fstat(descriptor)
        _require_regular_single_link(opened, label)
        if _stable_stat(before) != _stable_stat(opened):
            _fail("raced-input", f"{label} changed while it was opened")
        raw = _read_exact_bounded(descriptor, opened.st_size, maximum_bytes, label)
        after = os.fstat(descriptor)
        _require_regular_single_link(after, label)
        if _stable_stat(opened) != _stable_stat(after):
            _fail("mutated-input", f"{label} changed while it was read")
    finally:
        _close_quietly(descriptor)
    try:
        path_after = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail("raced-input", f"cannot re-inspect {label}: {error}")
    _require_regular_single_link(path_after, label)
    if _stable_stat(before) != _stable_stat(path_after):
        _fail("raced-input", f"{label} changed while it was read")
    return raw


def read_bounded_regular_relative(
    root_fd: int,
    relative_path: str,
    label: str,
    *,
    maximum_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> bytes:
    """Read one regular, single-link file below a pinned evidence-root FD."""

    relative = validate_relative_path(relative_path, f"{label} path")
    parts = PurePosixPath(relative).parts
    parent_fd, owned = _open_relative_directory_chain(root_fd, parts[:-1], label)
    try:
        return _read_regular_at(parent_fd, parts[-1], label, maximum_bytes=maximum_bytes)
    finally:
        for descriptor in reversed(owned):
            _close_quietly(descriptor)


def read_bounded_regular_path(
    path: Path,
    label: str,
    *,
    maximum_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> bytes:
    """Read an absolute regular path through a no-follow parent directory FD."""

    raw_path = os.fspath(path)
    components = _absolute_components(Path(raw_path), label)
    if not components:
        _fail("invalid-evidence-path", f"{label} must name a regular file, not root")
    parent_path = Path(os.path.sep).joinpath(*components[:-1])
    parent_fd = open_absolute_directory(parent_path, f"{label} parent")
    try:
        return _read_regular_at(
            parent_fd,
            components[-1],
            label,
            maximum_bytes=maximum_bytes,
        )
    finally:
        _close_quietly(parent_fd)


@dataclass(frozen=True)
class EvidenceDescriptor:
    """A self-contained reference to exact raw evidence bytes."""

    path: str
    sha256: str
    byte_length: int

    def as_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }


def descriptor_for_bytes(relative_path: str, raw: bytes, label: str) -> EvidenceDescriptor:
    """Describe raw bytes under a normalized relative evidence-root path."""

    relative = validate_relative_path(relative_path, f"{label}.path")
    if type(raw) is not bytes:
        _fail("invalid-evidence-bytes", f"{label} must be bytes")
    return EvidenceDescriptor(
        path=relative,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw),
    )


def parse_descriptor(value: Any, label: str) -> EvidenceDescriptor:
    """Parse one exact v2 descriptor without accepting field aliases."""

    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "byte_length"}:
        _fail("invalid-descriptor", f"{label} must contain exactly path, sha256, byte_length")
    path = validate_relative_path(value["path"], f"{label}.path")
    digest = value["sha256"]
    if (
        type(digest) is not str
        or SHA256_RE.fullmatch(digest) is None
        or digest == "0" * 64
    ):
        _fail("invalid-descriptor", f"{label}.sha256 must be a lowercase SHA-256 digest")
    byte_length = value["byte_length"]
    if type(byte_length) is not int or byte_length < 0:
        _fail("invalid-descriptor", f"{label}.byte_length must be a non-negative integer")
    return EvidenceDescriptor(path=path, sha256=digest, byte_length=byte_length)


def require_unique_descriptors(
    values: Sequence[EvidenceDescriptor | Mapping[str, Any]],
    label: str,
) -> tuple[EvidenceDescriptor, ...]:
    """Parse descriptors and reject reused evidence locations.

    A digest may legitimately recur when two independently captured files have
    identical bytes.  Reused *paths*, however, would make a receipt's binding
    ambiguous and are always rejected.
    """

    parsed: list[EvidenceDescriptor] = []
    used_paths: set[str] = set()
    for index, value in enumerate(values):
        candidate = value.as_json() if isinstance(value, EvidenceDescriptor) else value
        descriptor = parse_descriptor(candidate, f"{label}[{index}]")
        if descriptor.path in used_paths:
            _fail("duplicate-evidence-path", f"{label} reuses evidence path {descriptor.path!r}")
        used_paths.add(descriptor.path)
        parsed.append(descriptor)
    return tuple(parsed)


def read_descriptor_json(
    root_fd: int,
    descriptor: EvidenceDescriptor | Mapping[str, Any],
    label: str,
    *,
    maximum_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    """Load one descriptor, verify digest/length, then parse exact canonical JSON."""

    raw = read_descriptor_bytes(
        root_fd,
        descriptor,
        label,
        maximum_bytes=maximum_bytes,
    )
    parsed_document = parse_canonical_json(raw, label, maximum_bytes=maximum_bytes)
    assert isinstance(parsed_document, dict)
    return raw, parsed_document


def read_descriptor_bytes(
    root_fd: int,
    descriptor: EvidenceDescriptor | Mapping[str, Any],
    label: str,
    *,
    maximum_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> bytes:
    """Read one bounded descriptor and bind its exact raw bytes.

    Use this for a raw JSON capture when a caller must parse a source tool's
    original formatting.  Large opaque artifacts should instead use
    ``verify_descriptor_file`` to stream their digest without materializing
    them in memory.
    """

    candidate = descriptor.as_json() if isinstance(descriptor, EvidenceDescriptor) else descriptor
    parsed = parse_descriptor(candidate, label)
    if parsed.byte_length > maximum_bytes:
        _fail("input-too-large", f"{label} exceeds its byte bound")
    raw = read_bounded_regular_relative(
        root_fd,
        parsed.path,
        label,
        maximum_bytes=maximum_bytes,
    )
    if len(raw) != parsed.byte_length:
        _fail("evidence-length-mismatch", f"{label} byte length differs from descriptor")
    if hashlib.sha256(raw).hexdigest() != parsed.sha256:
        _fail("evidence-hash-mismatch", f"{label} SHA-256 differs from descriptor")
    return raw


def _verify_regular_at(
    directory_fd: int,
    name: str,
    descriptor: EvidenceDescriptor,
    label: str,
    *,
    maximum_bytes: int,
) -> None:
    """Stream-hash one descriptor without loading an artifact into memory."""

    _validate_maximum(maximum_bytes, f"{label} maximum byte bound")
    _require_directory_fd(directory_fd, f"{label} parent")
    name = _validate_leaf_name(name, f"{label} name")
    if descriptor.byte_length > maximum_bytes:
        _fail("input-too-large", f"{label} exceeds its byte bound")
    try:
        before = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail("missing-input", f"cannot inspect {label}: {error}")
    _require_regular_single_link(before, label)
    if before.st_size != descriptor.byte_length:
        _fail("evidence-length-mismatch", f"{label} byte length differs from descriptor")
    try:
        opened_fd = os.open(name, _file_open_flags(), dir_fd=directory_fd)
    except OSError as error:
        _fail("unsafe-evidence-path", f"cannot open {label} without following links: {error}")
    try:
        opened = os.fstat(opened_fd)
        _require_regular_single_link(opened, label)
        if _stable_stat(before) != _stable_stat(opened):
            _fail("raced-input", f"{label} changed while it was opened")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            try:
                chunk = os.read(opened_fd, min(DEFAULT_READ_CHUNK_BYTES, remaining))
            except OSError as error:
                _fail("unreadable-input", f"cannot read {label}: {error}")
            if not chunk:
                _fail("truncated-input", f"{label} changed while it was read")
            digest.update(chunk)
            remaining -= len(chunk)
        try:
            if os.read(opened_fd, 1):
                _fail("mutated-input", f"{label} grew while it was read")
        except OSError as error:
            _fail("unreadable-input", f"cannot re-read {label}: {error}")
        after = os.fstat(opened_fd)
        _require_regular_single_link(after, label)
        if _stable_stat(opened) != _stable_stat(after):
            _fail("mutated-input", f"{label} changed while it was read")
    finally:
        _close_quietly(opened_fd)
    try:
        path_after = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail("raced-input", f"cannot re-inspect {label}: {error}")
    _require_regular_single_link(path_after, label)
    if _stable_stat(before) != _stable_stat(path_after):
        _fail("raced-input", f"{label} changed while it was read")
    if digest.hexdigest() != descriptor.sha256:
        _fail("evidence-hash-mismatch", f"{label} SHA-256 differs from descriptor")


def verify_descriptor_file(
    root_fd: int,
    descriptor: EvidenceDescriptor | Mapping[str, Any],
    label: str,
    *,
    maximum_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> None:
    """Fail-closed stream verification for any checksummed evidence leaf.

    Unlike ``read_descriptor_json``, this helper never materializes the raw
    file.  It is appropriate for source archives, bundles, and OCI artifacts
    whose configured upper bound may exceed the JSON evidence limit.
    """

    candidate = descriptor.as_json() if isinstance(descriptor, EvidenceDescriptor) else descriptor
    parsed = parse_descriptor(candidate, label)
    relative = validate_relative_path(parsed.path, f"{label}.path")
    parts = PurePosixPath(relative).parts
    parent_fd, owned = _open_relative_directory_chain(root_fd, parts[:-1], label)
    try:
        _verify_regular_at(
            parent_fd,
            parts[-1],
            parsed,
            label,
            maximum_bytes=maximum_bytes,
        )
    finally:
        for owned_fd in reversed(owned):
            _close_quietly(owned_fd)


@dataclass(frozen=True)
class CreatedEvidence:
    """Identity and digest returned only after a create-only file is durable."""

    name: str
    sha256: str
    byte_length: int
    device: int
    inode: int

    def descriptor(self, relative_path: str, label: str) -> EvidenceDescriptor:
        relative = validate_relative_path(relative_path, f"{label}.path")
        if PurePosixPath(relative).name != self.name:
            _fail(
                "invalid-descriptor",
                f"{label}.path must end with created evidence name {self.name!r}",
            )
        return EvidenceDescriptor(
            path=relative,
            sha256=self.sha256,
            byte_length=self.byte_length,
        )


def _fsync_checked(descriptor: int, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        _fail("durability-failure", f"cannot durably synchronize {label}: {error}")


def _write_all(descriptor: int, raw: bytes, label: str) -> None:
    offset = 0
    while offset < len(raw):
        try:
            count = os.write(descriptor, raw[offset:])
        except OSError as error:
            _fail("unwritable-output", f"cannot write {label}: {error}")
        if count <= 0:
            _fail("unwritable-output", f"cannot write {label}: short write")
        offset += count


def write_create_only(
    directory_fd: int,
    name: str,
    raw: bytes,
    label: str,
) -> CreatedEvidence:
    """Durably write new bytes under a pinned directory without replacement.

    Existing names, symlinks, and non-directory parent FDs are rejected.  The
    file is created private (0600), fsynced, then the containing directory is
    fsynced before its identity is returned.
    """

    _require_directory_fd(directory_fd, f"{label} parent")
    name = _validate_leaf_name(name, f"{label} name")
    if type(raw) is not bytes:
        _fail("invalid-evidence-bytes", f"{label} must be bytes")
    try:
        descriptor = os.open(name, _output_open_flags(), 0o600, dir_fd=directory_fd)
    except FileExistsError as error:
        _fail("create-only-collision", f"cannot create new {label}: {error}")
    except OSError as error:
        _fail("unwritable-output", f"cannot create new {label}: {error}")
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except OSError as error:
            _fail("unsafe-output-mode", f"cannot make {label} private: {error}")
        metadata = os.fstat(descriptor)
        _require_regular_single_link(metadata, label)
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            _fail("unsafe-output-mode", f"{label} is not mode 0600")
        _write_all(descriptor, raw, label)
        _fsync_checked(descriptor, label)
        stable = os.fstat(descriptor)
        _require_regular_single_link(stable, label)
        if (metadata.st_dev, metadata.st_ino) != (stable.st_dev, stable.st_ino):
            _fail("raced-output", f"{label} changed while it was written")
    finally:
        _close_quietly(descriptor)
    try:
        visible = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail("raced-output", f"cannot re-inspect {label}: {error}")
    _require_regular_single_link(visible, label)
    if (visible.st_dev, visible.st_ino) != (stable.st_dev, stable.st_ino):
        _fail("raced-output", f"{label} changed before it could be published")
    _fsync_checked(directory_fd, f"{label} parent directory")
    return CreatedEvidence(
        name=name,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_length=len(raw),
        device=stable.st_dev,
        inode=stable.st_ino,
    )


def write_create_only_json(
    directory_fd: int,
    name: str,
    value: Any,
    label: str,
) -> CreatedEvidence:
    """Create one durable evidence file containing exact canonical JSON."""

    return write_create_only(directory_fd, name, canonical_json_bytes(value), label)


def create_incomplete_marker(
    directory_fd: int,
    name: str,
    marker: Mapping[str, Any],
    label: str = "incomplete evidence marker",
) -> CreatedEvidence:
    """Create a canonical, create-only nonterminal marker.

    A producer should write this marker before any terminal receipt.  Its
    removal/final transition remains producer-specific because only that
    producer can define which terminal receipt makes a capture complete.
    """

    if not isinstance(marker, Mapping):
        _fail("invalid-marker", f"{label} must be a JSON object")
    return write_create_only_json(directory_fd, name, dict(marker), label)
