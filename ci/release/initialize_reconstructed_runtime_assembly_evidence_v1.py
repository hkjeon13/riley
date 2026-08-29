#!/usr/bin/env python3
"""Create the private evidence root used by one raw assembly capture.

This is a deliberately small filesystem-only boundary for the host runner.
It creates one fresh external mode-0700 evidence root and its fixed ``raw``
child through held no-follow directory descriptors.  It neither invokes a
container engine nor reads build/runtime inputs; those operations remain in
the authenticated shell runner after this initializer returns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Sequence

import provenance_v2_common as common


sys.dont_write_bytecode = True

INITIALIZATION_VERSION = "riley.reconstructed-runtime-assembly-evidence-initialization.v1"
RAW_DIRECTORY_NAME = "raw"
SOURCE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _MountRecord:
    device: str
    root: str
    mount_point: str


class AssemblyEvidenceInitializationError(ValueError):
    """The fresh assembly evidence directory cannot be safely initialized."""


def _fail(reason_code: str, message: str) -> NoReturn:
    error = AssemblyEvidenceInitializationError(message)
    error.reason_code = reason_code  # type: ignore[attr-defined]
    raise error


def _common(call):  # type: ignore[no-untyped-def]
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _path(value: Path, label: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError:
        _fail("invalid-absolute-path", f"{label} must be an absolute path")
    if type(raw) is not str:
        _fail("invalid-absolute-path", f"{label} must be an absolute path")
    if (
        not os.path.isabs(raw)
        or "\x00" in raw
        or "\\" in raw
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        _fail("invalid-absolute-path", f"{label} must be a normalized absolute path")
    return Path(raw)


def _directory_flags() -> int:
    nofollow, directory, cloexec, _nonblock = _common(common.require_safe_open_flags)
    return os.O_RDONLY | nofollow | directory | cloexec


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _mountinfo_component(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _mount_records(raw: str) -> tuple[_MountRecord, ...]:
    records: list[_MountRecord] = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 7 or "-" not in fields:
            _fail("unsafe-evidence-directory", "cannot parse Linux mountinfo safely")
        separator = fields.index("-")
        if separator < 6:
            _fail("unsafe-evidence-directory", "Linux mountinfo has no filesystem separator")
        root = _mountinfo_component(fields[3])
        mount_point = _mountinfo_component(fields[4])
        # Namespace filesystems can use a non-path root (for example
        # ``mnt:[4026533116]``).  Such unrelated records cannot cover one of
        # our normalized absolute paths.  Retain a non-absolute root only
        # until a relevant mount is selected below, where it is rejected
        # fail-closed instead of making every ordinary checkout unusable.
        if not os.path.isabs(mount_point):
            continue
        records.append(
            _MountRecord(
                fields[2],
                os.path.normpath(root) if os.path.isabs(root) else root,
                os.path.normpath(mount_point),
            )
        )
    if not records:
        _fail("unsafe-evidence-directory", "Linux mountinfo is empty")
    return tuple(records)


def _mount_for_path(records: Sequence[_MountRecord], path: Path) -> _MountRecord:
    raw = os.fspath(path)
    candidates = [
        record
        for record in records
        if raw == record.mount_point or raw.startswith(record.mount_point.rstrip("/") + "/")
    ]
    if not candidates:
        _fail("unsafe-evidence-directory", "no Linux mountinfo entry covers the evidence path")
    return max(candidates, key=lambda record: len(record.mount_point))


def _mount_coordinate(record: _MountRecord, path: Path) -> str:
    if not os.path.isabs(record.root):
        _fail("unsafe-evidence-directory", "relevant Linux mountinfo root is not an absolute path")
    relative = os.path.relpath(os.fspath(path), record.mount_point)
    if relative == ".":
        return record.root
    return os.path.normpath(os.path.join(record.root, relative))


def _is_at_or_below_path(path: str, ancestor: str) -> bool:
    return path == ancestor or path.startswith(ancestor.rstrip("/") + "/")


def _assert_not_source_bind_alias(path: Path, source_root: Path) -> None:
    """Reject Linux bind mounts whose backing path lies in the checkout."""

    if not sys.platform.startswith("linux"):
        return
    try:
        raw = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot read Linux mountinfo: {error}")
    records = _mount_records(raw)
    evidence_mount = _mount_for_path(records, path.parent)
    evidence_coordinate = _mount_coordinate(evidence_mount, path.parent)
    source_mount = _mount_for_path(records, source_root)
    source_coordinates: list[tuple[str, str]] = [
        (source_mount.device, _mount_coordinate(source_mount, source_root))
    ]
    source_raw = os.fspath(source_root)
    # A checkout may itself contain nested mounts. A bind of one of those
    # descendants has a different device from the checkout's top-level mount,
    # so compare it against every mountpoint lexically rooted in the checkout.
    for record in records:
        if record.mount_point == source_raw or record.mount_point.startswith(source_raw.rstrip("/") + "/"):
            if not os.path.isabs(record.root):
                _fail(
                    "unsafe-evidence-directory",
                    "a checkout-descendant Linux mount has a non-absolute root",
                )
            source_coordinates.append((record.device, record.root))
    for source_device, source_coordinate in source_coordinates:
        if source_device == evidence_mount.device and _is_at_or_below_path(
            evidence_coordinate,
            source_coordinate,
        ):
            _fail(
                "evidence-root-inside-source-checkout",
                "--evidence-dir must be external to the source checkout, including descendant bind mounts",
            )


def _is_at_or_below(held_directory_fd: int, held_ancestor_fd: int) -> bool:
    """Check ancestry using only directory FDs, including a bind-mount alias."""

    try:
        ancestor = os.fstat(held_ancestor_fd)
        current_fd = os.dup(held_directory_fd)
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot inspect evidence/source directory identity: {error}")
    seen: set[tuple[int, int]] = set()
    try:
        while True:
            try:
                current = os.fstat(current_fd)
            except OSError as error:
                _fail("unsafe-evidence-directory", f"cannot inspect evidence ancestry: {error}")
            identity = (current.st_dev, current.st_ino)
            if identity == (ancestor.st_dev, ancestor.st_ino):
                return True
            if identity in seen:
                _fail("unsafe-evidence-directory", "evidence ancestry contains a directory loop")
            seen.add(identity)
            try:
                parent_fd = os.open("..", _directory_flags(), dir_fd=current_fd)
                parent = os.fstat(parent_fd)
            except OSError as error:
                _fail("unsafe-evidence-directory", f"cannot inspect evidence parent without following links: {error}")
            if _same_inode(parent, current):
                os.close(parent_fd)
                return False
            os.close(current_fd)
            current_fd = parent_fd
    finally:
        try:
            os.close(current_fd)
        except OSError:
            pass


def _assert_external_to_source_checkout(path: Path, source_root: Path) -> None:
    source_fd = _common(lambda: common.open_absolute_directory(source_root, "source checkout"))
    parent_fd = _common(lambda: common.open_absolute_directory(path.parent, "evidence parent"))
    try:
        if _is_at_or_below(parent_fd, source_fd):
            _fail(
                "evidence-root-inside-source-checkout",
                "--evidence-dir must be external to the source checkout, including mount aliases",
            )
    finally:
        try:
            os.close(parent_fd)
        finally:
            os.close(source_fd)
    _assert_not_source_bind_alias(path, source_root)


def _require_visible_root(path: Path, root_fd: int) -> None:
    try:
        named = os.lstat(path)
        held = os.fstat(root_fd)
    except OSError as error:
        _fail("raced-output", f"cannot bind the evidence root path to its held FD: {error}")
    if not stat.S_ISDIR(named.st_mode) or not stat.S_ISDIR(held.st_mode) or not _same_inode(named, held):
        _fail("raced-output", "evidence root path differs from the created held directory")


def initialize_reconstructed_runtime_assembly_evidence(
    evidence_dir: Path,
    *,
    source_root: Path = SOURCE_ROOT,
) -> dict[str, object]:
    """Create the new root/raw pair and reject source-tree aliases before use."""

    root_path = _path(evidence_dir, "--evidence-dir")
    source_path = _path(source_root, "source checkout")
    # Check before any create so an obvious source-tree or bind-mount alias
    # cannot leave a fresh directory in the checkout on an expected failure.
    _assert_external_to_source_checkout(root_path, source_path)
    root_fd = _common(
        lambda: common.create_private_evidence_directory(
            root_path,
            "runtime assembly evidence root",
        )
    )
    raw_fd: int | None = None
    try:
        _common(lambda: common.require_private_evidence_directory_fd(root_fd, "runtime assembly evidence root"))
        _assert_external_to_source_checkout(root_path, source_path)
        _require_visible_root(root_path, root_fd)
        raw_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                RAW_DIRECTORY_NAME,
                "runtime assembly raw evidence directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                raw_fd,
                RAW_DIRECTORY_NAME,
                "runtime assembly raw evidence directory",
            )
        )
        _require_visible_root(root_path, root_fd)
        return {
            "schema_version": INITIALIZATION_VERSION,
            "status": "initialized",
            "raw_directory": RAW_DIRECTORY_NAME,
        }
    finally:
        if raw_fd is not None:
            try:
                os.close(raw_fd)
            except OSError as error:
                _fail("unsafe-evidence-directory", f"cannot close raw evidence FD: {error}")
        try:
            os.close(root_fd)
        except OSError as error:
            _fail("unsafe-evidence-directory", f"cannot close evidence root FD: {error}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = initialize_reconstructed_runtime_assembly_evidence(args.evidence_dir)
    except AssemblyEvidenceInitializationError as error:
        print(
            f"runtime assembly evidence initialization failed: {getattr(error, 'reason_code', 'unsafe-evidence')}: {error}",
            file=sys.stderr,
        )
        return 1
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
