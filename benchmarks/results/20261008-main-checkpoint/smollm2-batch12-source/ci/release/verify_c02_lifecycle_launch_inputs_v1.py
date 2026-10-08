#!/usr/bin/env python3
"""Verify immutable-looking host launch inputs before and after a C02 run.

The lifecycle runner invokes this pure, stdlib-only verifier immediately before
launch and again after the service exits.  It never starts a process, queries a
GPU, or writes evidence.  It rejects link traversal and mutable-by-other-user
binary/model entries while calculating the same deterministic model-tree
digest used by the host C02 capture contract.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

import provenance_v2_common as common


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_MODEL_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._+@=-]+$")


class LaunchInputVerificationError(ValueError):
    """A host binary or model tree cannot safely be bound to this run."""


def _fail(code: str, message: str) -> NoReturn:
    error = LaunchInputVerificationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-input"), str(error))


def _absolute_parts(path: Path, label: str) -> tuple[Path, str]:
    raw = os.fspath(path)
    if (
        not os.path.isabs(raw)
        or "\x00" in raw
        or "\\" in raw
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        _fail("invalid-absolute-path", f"{label} must be a normalized absolute path")
    parent_text, name = os.path.split(raw)
    if not parent_text or not name or common.SAFE_LEAF_RE.fullmatch(name) is None:
        _fail("invalid-input-path", f"{label} must name one safe final path component")
    return Path(parent_text), name


def _safe_file_flags() -> int:
    nofollow, _directory, cloexec, nonblock = _common(common.require_safe_open_flags)
    return os.O_RDONLY | nofollow | cloexec | nonblock


def _stable(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_trusted_regular(metadata: os.stat_result, label: str, *, executable: bool = False) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        _fail("unsafe-input-path", f"{label} must be a single-link regular file")
    if metadata.st_uid not in {0, os.geteuid()}:
        _fail("unsafe-input-owner", f"{label} must be owned by root or the effective UID")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        _fail("unsafe-input-mode", f"{label} must not be group/world writable")
    if executable and not metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        _fail("binary-not-executable", f"{label} must have an executable mode bit")


def _require_trusted_directory(metadata: os.stat_result, label: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("unsafe-input-path", f"{label} must be a directory")
    if metadata.st_uid not in {0, os.geteuid()}:
        _fail("unsafe-input-owner", f"{label} must be owned by root or the effective UID")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        _fail("unsafe-input-mode", f"{label} must not be group/world writable")


def _hash_open_file(descriptor: int, expected: os.stat_result, label: str) -> str:
    digest = hashlib.sha256()
    while True:
        try:
            block = os.read(descriptor, 1024 * 1024)
        except OSError as error:
            _fail("unreadable-input", f"cannot read {label}: {error}")
        if not block:
            break
        digest.update(block)
    try:
        after = os.fstat(descriptor)
    except OSError as error:
        _fail("unsafe-input-path", f"cannot re-stat {label}: {error}")
    if _stable(after) != _stable(expected):
        _fail("raced-input", f"{label} changed while it was hashed")
    return digest.hexdigest()


@dataclass(frozen=True)
class FileIdentity:
    sha256: str
    device: int
    inode: int
    byte_length: int

    def as_json(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "byte_length": self.byte_length,
        }


def _hash_relative_regular(parent_fd: int, name: str, label: str, *, executable: bool = False) -> FileIdentity:
    try:
        before = os.lstat(name, dir_fd=parent_fd)
    except OSError as error:
        _fail("missing-input", f"cannot inspect {label}: {error}")
    _require_trusted_regular(before, label, executable=executable)
    try:
        descriptor = os.open(name, _safe_file_flags(), dir_fd=parent_fd)
    except OSError as error:
        _fail("unsafe-input-path", f"cannot open {label} without following links: {error}")
    try:
        opened = os.fstat(descriptor)
        _require_trusted_regular(opened, label, executable=executable)
        if _stable(opened) != _stable(before):
            _fail("raced-input", f"{label} changed while it was opened")
        digest = _hash_open_file(descriptor, opened, label)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(name, dir_fd=parent_fd)
    except OSError as error:
        _fail("raced-input", f"cannot re-inspect {label}: {error}")
    if _stable(after) != _stable(before):
        _fail("raced-input", f"{label} changed while it was hashed")
    return FileIdentity(
        sha256=digest,
        device=before.st_dev,
        inode=before.st_ino,
        byte_length=before.st_size,
    )


def verify_binary(path: Path, expected_sha256: str) -> FileIdentity:
    if SHA256_RE.fullmatch(expected_sha256) is None or expected_sha256 == "0" * 64:
        _fail("invalid-sha256", "--binary-sha256 must be a non-zero lowercase SHA-256")
    parent, name = _absolute_parts(path, "--binary")
    parent_fd = _common(lambda: common.open_absolute_directory(parent, "--binary parent"))
    try:
        identity = _hash_relative_regular(parent_fd, name, "--binary", executable=True)
    finally:
        os.close(parent_fd)
    if identity.sha256 != expected_sha256:
        _fail("binary-sha256-mismatch", "--binary SHA-256 does not match --binary-sha256")
    return identity


def _open_child_directory(parent_fd: int, name: str, label: str) -> int:
    nofollow, directory, cloexec, _nonblock = _common(common.require_safe_open_flags)
    try:
        before = os.lstat(name, dir_fd=parent_fd)
    except OSError as error:
        _fail("missing-input", f"cannot inspect {label}: {error}")
    _require_trusted_directory(before, label)
    try:
        descriptor = os.open(name, os.O_RDONLY | nofollow | directory | cloexec, dir_fd=parent_fd)
    except OSError as error:
        _fail("unsafe-input-path", f"cannot open {label} without following links: {error}")
    opened = os.fstat(descriptor)
    _require_trusted_directory(opened, label)
    if _stable(opened) != _stable(before):
        os.close(descriptor)
        _fail("raced-input", f"{label} changed while it was opened")
    return descriptor


def _tree_entries(
    directory_fd: int,
    prefix: str,
    entries: list[tuple[str, FileIdentity]],
    label: str,
) -> int:
    try:
        names = os.listdir(directory_fd)
    except OSError as error:
        _fail("unreadable-input", f"cannot list {label}: {error}")
    count = 0
    for name in sorted(names):
        if SAFE_MODEL_COMPONENT_RE.fullmatch(name) is None:
            _fail("unsafe-model-path", f"{label} contains an unsafe path component {name!r}")
        relative = name if not prefix else f"{prefix}/{name}"
        try:
            metadata = os.lstat(name, dir_fd=directory_fd)
        except OSError as error:
            _fail("missing-input", f"cannot inspect model entry {relative!r}: {error}")
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = _open_child_directory(directory_fd, name, f"model directory {relative!r}")
            try:
                count += _tree_entries(child_fd, relative, entries, label)
            finally:
                os.close(child_fd)
            try:
                after = os.lstat(name, dir_fd=directory_fd)
            except OSError as error:
                _fail("raced-input", f"cannot re-inspect model directory {relative!r}: {error}")
            if _stable(after) != _stable(metadata):
                _fail("raced-input", f"model directory {relative!r} changed while it was read")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _fail("unsafe-model-path", f"model entry {relative!r} must be a regular file or directory")
        identity = _hash_relative_regular(directory_fd, name, f"model file {relative!r}")
        entries.append((relative, identity))
        count += 1
    return count


def verify_model_tree(path: Path, expected_sha256: str) -> dict[str, Any]:
    if SHA256_RE.fullmatch(expected_sha256) is None or expected_sha256 == "0" * 64:
        _fail("invalid-sha256", "--model-tree-sha256 must be a non-zero lowercase SHA-256")
    parent, name = _absolute_parts(path, "--model-dir")
    parent_fd = _common(lambda: common.open_absolute_directory(parent, "--model-dir parent"))
    try:
        model_fd = _open_child_directory(parent_fd, name, "--model-dir")
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        root_before = os.fstat(model_fd)
        entries: list[tuple[str, FileIdentity]] = []
        file_count = _tree_entries(model_fd, "", entries, "--model-dir")
        if not file_count:
            _fail("empty-model-tree", "--model-dir contains no regular files")
        # The established C02 host contract sorts the complete POSIX relative
        # file path globally.  Directory-first traversal is not equivalent:
        # e.g. ``tokenizer/merges.txt`` must sort after ``tokenizer.json``.
        lines = [
            f"{identity.sha256}  {relative}\n".encode("ascii")
            for relative, identity in sorted(entries, key=lambda item: item[0])
        ]
        digest = hashlib.sha256(b"".join(lines)).hexdigest()
        root_metadata = os.fstat(model_fd)
        if _stable(root_metadata) != _stable(root_before):
            _fail("raced-input", "--model-dir changed while its tree was hashed")
        named_after = os.lstat(name, dir_fd=parent_fd)
        if _stable(named_after) != _stable(root_before):
            _fail("raced-input", "--model-dir path changed while its tree was hashed")
    finally:
        os.close(model_fd)
        os.close(parent_fd)
    if digest != expected_sha256:
        _fail("model-tree-sha256-mismatch", "--model-dir tree SHA-256 does not match --model-tree-sha256")
    return {
        "tree_sha256": digest,
        "file_count": file_count,
        "device": root_metadata.st_dev,
        "inode": root_metadata.st_ino,
    }


def verify_launch_inputs(
    *,
    binary: Path,
    binary_sha256: str,
    model_dir: Path,
    model_tree_sha256: str,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"pre-launch", "post-exit"}:
        _fail("invalid-phase", "--phase must be pre-launch or post-exit")
    return {
        "schema_version": "riley.c02-lifecycle-launch-input-check.v1",
        "status": "verified",
        "phase": phase,
        "binary": verify_binary(binary, binary_sha256).as_json(),
        "model": verify_model_tree(model_dir, model_tree_sha256),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-tree-sha256", required=True)
    parser.add_argument("--phase", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_launch_inputs(
            binary=args.binary,
            binary_sha256=args.binary_sha256,
            model_dir=args.model_dir,
            model_tree_sha256=args.model_tree_sha256,
            phase=args.phase,
        )
    except (LaunchInputVerificationError, OSError) as error:
        print(f"C02 lifecycle launch-input verification refused: {error}", file=sys.stderr)
        return 2
    print(common.canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
