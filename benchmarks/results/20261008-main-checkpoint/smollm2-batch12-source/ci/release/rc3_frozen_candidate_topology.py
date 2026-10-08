#!/usr/bin/env python3
"""Path/FD topology checks around RC3 frozen-candidate public wrappers.

The public writer/replayer wrappers use this module before entering their
private held-FD cores.  Those wrappers close ordinary lexical overlap,
directory-FD ancestry, Linux bind-mount backing-coordinate overlap, and
output-path replacement gaps.  Private cores additionally use the held-FD
helper below to reject physical directory ancestry without re-opening caller
paths or attempting mount-alias discovery.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NoReturn, Sequence

import provenance_v2_common as common


@dataclass(frozen=True)
class _MountRecord:
    device: str
    root: str
    mount_point: str


class FrozenCandidateTopologyError(ValueError):
    """The public frozen-candidate root topology is unsafe or raced."""


def _fail(code: str, message: str) -> NoReturn:
    error = FrozenCandidateTopologyError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call):  # type: ignore[no-untyped-def]
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _directory_flags() -> int:
    nofollow, directory, cloexec, _nonblock = _common(common.require_safe_open_flags)
    return os.O_RDONLY | nofollow | directory | cloexec


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _is_at_or_below(held_directory_fd: int, held_ancestor_fd: int) -> bool:
    """Check physical directory ancestry from FDs without following names."""

    try:
        ancestor = os.fstat(held_ancestor_fd)
        current_fd = os.dup(held_directory_fd)
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot inspect held directory ancestry: {error}")
    seen: set[tuple[int, int]] = set()
    try:
        while True:
            try:
                current = os.fstat(current_fd)
            except OSError as error:
                _fail("unsafe-evidence-directory", f"cannot inspect held directory ancestry: {error}")
            identity = current.st_dev, current.st_ino
            if identity == (ancestor.st_dev, ancestor.st_ino):
                return True
            if identity in seen:
                _fail("unsafe-evidence-directory", "held directory ancestry contains a loop")
            seen.add(identity)
            try:
                parent_fd = os.open("..", _directory_flags(), dir_fd=current_fd)
                parent = os.fstat(parent_fd)
            except OSError as error:
                _fail("unsafe-evidence-directory", f"cannot inspect held directory parent: {error}")
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
        _fail("unsafe-evidence-directory", "no Linux mountinfo entry covers a frozen-candidate path")
    longest = max(len(record.mount_point) for record in candidates)
    selected = tuple(record for record in candidates if len(record.mount_point) == longest)
    if len(selected) != 1:
        _fail(
            "unsafe-evidence-directory",
            "Linux mountinfo has ambiguous equally specific mounts for a frozen-candidate path",
        )
    return selected[0]


def _mount_coordinate(record: _MountRecord, path: Path) -> str:
    if not os.path.isabs(record.root):
        _fail("unsafe-evidence-directory", "relevant Linux mountinfo root is not absolute")
    relative = os.path.relpath(os.fspath(path), record.mount_point)
    if relative == ".":
        return record.root
    return os.path.normpath(os.path.join(record.root, relative))


def _is_at_or_below_path(path: str, ancestor: str) -> bool:
    return path == ancestor or path.startswith(ancestor.rstrip("/") + "/")


def _mount_coordinates_for_region(
    records: Sequence[_MountRecord],
    path: Path,
) -> tuple[tuple[str, str], ...]:
    selected = _mount_for_path(records, path)
    coordinates: list[tuple[str, str]] = [
        (selected.device, _mount_coordinate(selected, path))
    ]
    raw = os.fspath(path)
    for record in records:
        if record.mount_point == raw or record.mount_point.startswith(raw.rstrip("/") + "/"):
            if not os.path.isabs(record.root):
                _fail("unsafe-evidence-directory", "nested Linux mount has a non-absolute root")
            coordinates.append((record.device, record.root))
    return tuple(coordinates)


def _assert_mount_regions_disjoint(left: Path, left_label: str, right: Path, right_label: str) -> None:
    """Reject Linux mount aliases whose backing coordinates overlap either way."""

    if not sys.platform.startswith("linux"):
        return
    try:
        raw = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot read Linux mountinfo: {error}")
    records = _mount_records(raw)
    left_coordinates = _mount_coordinates_for_region(records, left)
    right_coordinates = _mount_coordinates_for_region(records, right)
    for left_device, left_coordinate in left_coordinates:
        for right_device, right_coordinate in right_coordinates:
            if left_device != right_device:
                continue
            if _is_at_or_below_path(left_coordinate, right_coordinate) or _is_at_or_below_path(
                right_coordinate,
                left_coordinate,
            ):
                _fail(
                    "frozen-candidate-mount-alias",
                    f"{left_label} and {right_label} overlap through Linux mount backing coordinates",
                )


def _assert_mount_region_external(
    candidate: Path,
    candidate_label: str,
    existing: Path,
    existing_label: str,
) -> None:
    """Reject only a candidate region located inside an existing region.

    A fresh output's parent normally contains the source/input siblings, so
    this check is intentionally directional.  The symmetric disjointness
    assertion is applied after the output root itself is created.
    """

    if not sys.platform.startswith("linux"):
        return
    try:
        raw = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot read Linux mountinfo: {error}")
    records = _mount_records(raw)
    candidate_coordinates = _mount_coordinates_for_region(records, candidate)
    existing_coordinates = _mount_coordinates_for_region(records, existing)
    for candidate_device, candidate_coordinate in candidate_coordinates:
        for existing_device, existing_coordinate in existing_coordinates:
            if (
                candidate_device == existing_device
                and _is_at_or_below_path(candidate_coordinate, existing_coordinate)
            ):
                _fail(
                    "frozen-candidate-mount-alias",
                    f"{candidate_label} lies inside {existing_label} through Linux mount backing coordinates",
                )


def _assert_fd_regions_disjoint(
    left_fd: int,
    left_label: str,
    right_fd: int,
    right_label: str,
) -> None:
    if _is_at_or_below(left_fd, right_fd) or _is_at_or_below(right_fd, left_fd):
        _fail(
            "frozen-candidate-root-overlap",
            f"{left_label} and {right_label} overlap through held directory ancestry",
        )


def assert_held_root_fds_disjoint(roots: Mapping[str, int]) -> None:
    """Require caller-held directory FDs to have no physical ancestry overlap.

    This is the private-core counterpart to :func:`assert_existing_roots_disjoint`.
    It intentionally does not resolve visible paths or Linux mount aliases:
    callers that have paths must use the public assertion as well.
    """

    rows = tuple(roots.items())
    for index, (left_label, left_fd) in enumerate(rows):
        for right_label, right_fd in rows[index + 1 :]:
            _assert_fd_regions_disjoint(left_fd, left_label, right_fd, right_label)


def _require_visible_roots(roots: Mapping[str, tuple[Path, int]]) -> None:
    for label, (path, directory_fd) in roots.items():
        require_visible_root(path, directory_fd, label)


def assert_existing_roots_disjoint(
    roots: Mapping[str, tuple[Path, int]],
) -> None:
    """Require every existing public root to be physically disjoint."""

    _require_visible_roots(roots)
    rows = tuple(roots.items())
    for index, (left_label, (left_path, left_fd)) in enumerate(rows):
        for right_label, (right_path, right_fd) in rows[index + 1 :]:
            _assert_fd_regions_disjoint(left_fd, left_label, right_fd, right_label)
            _assert_mount_regions_disjoint(left_path, left_label, right_path, right_label)
    _require_visible_roots(roots)


def assert_new_root_parent_external(
    frozen_root: Path,
    existing_roots: Mapping[str, tuple[Path, int]],
) -> None:
    """Reject a new output root below an existing source or input root.

    A parent that merely contains the source/input as ordinary siblings is
    valid.  Once the new child exists, callers apply the symmetric check.
    """

    _require_visible_roots(existing_roots)
    parent_fd = _common(
        lambda: common.open_absolute_directory(
            frozen_root.parent,
            "frozen candidate root parent",
        )
    )
    try:
        for label, (path, directory_fd) in existing_roots.items():
            if _is_at_or_below(parent_fd, directory_fd):
                _fail(
                    "frozen-candidate-root-overlap",
                    f"frozen candidate root parent lies inside {label}",
                )
            _assert_mount_region_external(
                frozen_root.parent,
                "frozen candidate root parent",
                path,
                label,
            )
    finally:
        try:
            os.close(parent_fd)
        except OSError:
            pass
    _require_visible_roots(existing_roots)


def require_visible_root(path: Path, root_fd: int, label: str) -> None:
    """Bind a created/replayed absolute root path back to its held FD."""

    reopened_fd = -1
    try:
        reopened_fd = _common(lambda: common.open_absolute_directory(path, label))
        held = os.fstat(root_fd)
        reopened = os.fstat(reopened_fd)
    except OSError as error:
        _fail("raced-output", f"cannot bind {label} path to its held FD: {error}")
    finally:
        if reopened_fd >= 0:
            try:
                os.close(reopened_fd)
            except OSError:
                pass
    if not _same_inode(held, reopened):
        _fail("raced-output", f"{label} path differs from its held directory FD")
