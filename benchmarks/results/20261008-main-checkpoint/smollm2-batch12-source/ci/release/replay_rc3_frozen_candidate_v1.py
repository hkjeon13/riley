#!/usr/bin/env python3
"""Replay an RC3 frozen-candidate input identity through caller-held FDs.

This module establishes only that the fixed frozen-candidate manifest still
matches a fresh replay of the original request and its raw input leaves.  It
does not infer writer normal-return lineage, source/archive or image content
provenance, Gate E, semantic receipt, deployment, rollback success, or
qualification.  The private core has no direct Python write/open/close/lock
surface for caller-owned FDs.  Its admission replay invokes the existing
trusted read-only Git source oracle through a pinned descriptor, so callers
must also trust that oracle and the configured Git executable.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence, TypeVar

import provenance_v2_common as common
import rc3_frozen_candidate_common as frozen
import rc3_frozen_candidate_topology as topology


class FrozenCandidateReplayError(ValueError):
    """The held frozen-candidate topology cannot be replayed safely."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = FrozenCandidateReplayError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _frozen(call: Callable[[], T]) -> T:
    try:
        return call()
    except frozen.FrozenCandidateError as error:
        _fail(getattr(error, "reason_code", "invalid-frozen-candidate"), str(error))


def _topology(call: Callable[[], T]) -> T:
    try:
        return call()
    except topology.FrozenCandidateTopologyError as error:
        _fail(getattr(error, "reason_code", "unsafe-frozen-candidate-topology"), str(error))


def _shared_lock(directory_fd: int, label: str) -> None:
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("evidence-root-lock-unavailable", f"cannot acquire shared {label} lock: {error}")


def _unlock_quietly(directory_fd: int | None) -> None:
    if directory_fd is not None:
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(directory_fd: int | None) -> None:
    if directory_fd is not None:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _assert_exact_entries(directory_fd: int, expected: set[str], label: str) -> None:
    actual: set[str] = set()
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                actual.add(entry.name)
                if entry.name not in expected or len(actual) > len(expected):
                    _fail(
                        "unexpected-evidence-entry",
                        f"{label} entries differ; expected={sorted(expected)}, actual_prefix={sorted(actual)}",
                    )
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot list {label}: {error}")
    if actual != expected:
        _fail(
            "unexpected-evidence-entry",
            f"{label} entries differ; expected={sorted(expected)}, actual={sorted(actual)}",
        )


def _read_manifest(directory_fd: int) -> tuple[bytes, common.EvidenceDescriptor, dict[str, Any]]:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            directory_fd,
            frozen.MANIFEST_NAME,
            "frozen candidate manifest",
            maximum_bytes=frozen.MAX_MANIFEST_BYTES,
        )
    )
    descriptor = _common(
        lambda: common.descriptor_for_bytes(
            frozen.MANIFEST_NAME,
            raw,
            "frozen candidate manifest",
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            raw,
            "frozen candidate manifest",
            maximum_bytes=frozen.MAX_MANIFEST_BYTES,
        )
    )
    assert isinstance(document, dict)
    return raw, descriptor, document


def _replay_rc3_frozen_candidate_v1_on_held_fds(
    frozen_candidate_root_fd: int,
    input_evidence_root_fd: int,
    repository_root: Path,
    repository_root_fd: int,
) -> dict[str, Any]:
    """Replay the fixed manifest without direct Python FD mutation.

    The original request name is obtained only from the manifest's typed
    descriptor and is revalidated by the original request replayer.  A
    same-stack caller must retain all three root FDs and any locks throughout
    this call.  The held-FD source admission is the existing trusted
    read-only Git source oracle rather than a hermetic no-subprocess layer.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            frozen_candidate_root_fd,
            "frozen candidate root",
        )
    )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            input_evidence_root_fd,
            "freeze-input evidence root",
        )
    )
    _frozen(
        lambda: frozen.require_distinct_root_fds(
            {
                "frozen candidate root": frozen_candidate_root_fd,
                "freeze-input evidence root": input_evidence_root_fd,
                "source checkout": repository_root_fd,
            }
        )
    )
    _topology(
        lambda: topology.assert_held_root_fds_disjoint(
            {
                "frozen candidate root": frozen_candidate_root_fd,
                "freeze-input evidence root": input_evidence_root_fd,
                "source checkout": repository_root_fd,
            }
        )
    )
    _assert_exact_entries(
        frozen_candidate_root_fd,
        {frozen.MANIFEST_NAME},
        "frozen candidate root",
    )
    manifest_raw, manifest_descriptor, manifest = _read_manifest(frozen_candidate_root_fd)
    candidate_id, source_revision = _frozen(
        lambda: frozen.parse_frozen_candidate_manifest_identity(manifest)
    )
    manifest_request = _common(
        lambda: common.parse_descriptor(
            manifest["freeze_input_request"],
            "frozen candidate manifest freeze_input_request",
        )
    )
    expected_first = _frozen(
        lambda: frozen.derive_frozen_candidate_manifest_on_held_fds(
            repository_root=repository_root,
            repository_root_fd=repository_root_fd,
            input_evidence_root_fd=input_evidence_root_fd,
            expected_revision=source_revision,
            candidate_id=candidate_id,
            request_name=manifest_request.path,
        )
    )
    expected_second = _frozen(
        lambda: frozen.derive_frozen_candidate_manifest_on_held_fds(
            repository_root=repository_root,
            repository_root_fd=repository_root_fd,
            input_evidence_root_fd=input_evidence_root_fd,
            expected_revision=source_revision,
            candidate_id=candidate_id,
            request_name=manifest_request.path,
        )
    )
    if common.canonical_json_bytes(expected_first) != common.canonical_json_bytes(expected_second):
        _fail(
            "frozen-input-replay-drift",
            "fresh frozen-candidate input replays differ while held FDs remain open",
        )
    if common.canonical_json_bytes(manifest) != common.canonical_json_bytes(expected_first):
        _fail(
            "frozen-manifest-replay-mismatch",
            "frozen candidate manifest differs from the freshly replayed input identity",
        )
    manifest_raw_end, manifest_descriptor_end, manifest_end = _read_manifest(
        frozen_candidate_root_fd
    )
    if (
        manifest_raw_end != manifest_raw
        or manifest_descriptor_end != manifest_descriptor
        or manifest_end != manifest
    ):
        _fail("raced-input", "frozen candidate manifest changed during held-FD replay")
    _assert_exact_entries(
        frozen_candidate_root_fd,
        {frozen.MANIFEST_NAME},
        "frozen candidate root",
    )
    return _frozen(lambda: frozen.replay_result(manifest_descriptor, manifest))


def replay_rc3_frozen_candidate_v1(
    frozen_candidate_root: Path,
    *,
    input_evidence_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Open three disjoint roots and replay one frozen input identity safely."""

    frozen_root = _frozen(
        lambda: frozen.normalized_absolute_path(
            frozen_candidate_root,
            "--frozen-candidate-root",
        )
    )
    input_root = _frozen(
        lambda: frozen.normalized_absolute_path(input_evidence_root, "--input-evidence-root")
    )
    source_root = _frozen(
        lambda: frozen.normalized_absolute_path(repository_root, "--repository-root")
    )
    _frozen(
        lambda: frozen.require_disjoint_paths(
            {
                "frozen candidate root": frozen_root,
                "freeze-input evidence root": input_root,
                "source checkout": source_root,
            }
        )
    )
    source_root_fd: int | None = None
    input_root_fd: int | None = None
    frozen_root_fd: int | None = None
    try:
        source_root_fd = _common(
            lambda: common.open_absolute_directory(source_root, "source checkout")
        )
        input_root_fd = _common(
            lambda: common.open_private_evidence_directory(
                input_root,
                "freeze-input evidence root",
            )
        )
        frozen_root_fd = _common(
            lambda: common.open_private_evidence_directory(
                frozen_root,
                "frozen candidate root",
            )
        )
        roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
            "frozen candidate root": (frozen_root, frozen_root_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        _topology(
            lambda: topology.require_visible_root(
                frozen_root,
                frozen_root_fd,
                "frozen candidate root",
            )
        )
        _shared_lock(input_root_fd, "freeze-input evidence root")
        _shared_lock(frozen_root_fd, "frozen candidate root")
        result = _replay_rc3_frozen_candidate_v1_on_held_fds(
            frozen_root_fd,
            input_root_fd,
            source_root,
            source_root_fd,
        )
        _topology(
            lambda: topology.require_visible_root(
                frozen_root,
                frozen_root_fd,
                "frozen candidate root",
            )
        )
        _topology(lambda: topology.assert_existing_roots_disjoint(roots))
        return result
    finally:
        _unlock_quietly(frozen_root_fd)
        _unlock_quietly(input_root_fd)
        _close_quietly(frozen_root_fd)
        _close_quietly(input_root_fd)
        _close_quietly(source_root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-candidate-root", required=True, type=Path)
    parser.add_argument("--input-evidence-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = replay_rc3_frozen_candidate_v1(
            args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
        )
    except (OSError, FrozenCandidateReplayError) as error:
        print(f"RC3 frozen candidate replay failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
