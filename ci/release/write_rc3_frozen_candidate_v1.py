#!/usr/bin/env python3
"""Create one RC3 frozen-candidate input-identity manifest.

This producer creates a fresh private root containing exactly one canonical
manifest.  It replays the original RC3 freeze-input request and raw leaves
through held descriptors before publication and self-replays the manifest
before returning.  It does not run a model, GPU, container, server, Gate E,
deployment action, rollback, semantic receipt, or qualification decision.

The manifest is a recheckable static input identity, not proof that this
writer later returned normally.  A future same-stack semantic producer must
replay the manifest and original raw inputs again; it may not substitute
``freeze.raw`` or the read-only admission report.
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
import replay_rc3_frozen_candidate_v1 as replayer


class FrozenCandidateWriterError(ValueError):
    """A frozen candidate manifest cannot safely be created."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = FrozenCandidateWriterError(f"{code}: {message}")
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


def _replayer(call: Callable[[], T]) -> T:
    try:
        return call()
    except replayer.FrozenCandidateReplayError as error:
        _fail(getattr(error, "reason_code", "frozen-candidate-self-replay-failed"), str(error))


def _topology(call: Callable[[], T]) -> T:
    try:
        return call()
    except topology.FrozenCandidateTopologyError as error:
        _fail(getattr(error, "reason_code", "unsafe-frozen-candidate-topology"), str(error))


def _lock(directory_fd: int, operation: int, label: str) -> None:
    try:
        fcntl.flock(directory_fd, operation | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("evidence-root-lock-unavailable", f"cannot acquire {label} lock: {error}")


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


def write_rc3_frozen_candidate_v1(
    frozen_candidate_root: Path,
    *,
    input_evidence_root: Path,
    repository_root: Path,
    expected_revision: str,
    candidate_id: str,
    request_name: str,
) -> dict[str, Any]:
    """Create and self-replay one fresh frozen-candidate input identity.

    ``frozen_candidate_root`` must not exist.  The output is intentionally a
    root separate from the mutable input tree: the manifest pins original
    descriptor hashes rather than copying model weights that can be up to the
    admission byte budget.  Every later consumer must still hold and rehash
    the original input root.
    """

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
        input_roots = {
            "source checkout": (source_root, source_root_fd),
            "freeze-input evidence root": (input_root, input_root_fd),
        }
        _topology(lambda: topology.assert_existing_roots_disjoint(input_roots))
        _topology(lambda: topology.assert_new_root_parent_external(frozen_root, input_roots))
        _lock(input_root_fd, fcntl.LOCK_SH, "freeze-input evidence root shared")
        draft = _frozen(
            lambda: frozen.derive_frozen_candidate_manifest_on_held_fds(
                repository_root=source_root,
                repository_root_fd=source_root_fd,
                input_evidence_root_fd=input_root_fd,
                expected_revision=expected_revision,
                candidate_id=candidate_id,
                request_name=request_name,
            )
        )
        frozen_root_fd = _common(
            lambda: common.create_private_evidence_directory(
                frozen_root,
                "frozen candidate root",
            )
        )
        _topology(
            lambda: topology.require_visible_root(
                frozen_root,
                frozen_root_fd,
                "frozen candidate root",
            )
        )
        _topology(
            lambda: topology.assert_existing_roots_disjoint(
                {
                    "source checkout": (source_root, source_root_fd),
                    "freeze-input evidence root": (input_root, input_root_fd),
                    "frozen candidate root": (frozen_root, frozen_root_fd),
                }
            )
        )
        _lock(frozen_root_fd, fcntl.LOCK_EX, "frozen candidate root exclusive")
        _frozen(
            lambda: frozen.require_distinct_root_fds(
                {
                    "frozen candidate root": frozen_root_fd,
                    "freeze-input evidence root": input_root_fd,
                    "source checkout": source_root_fd,
                }
            )
        )
        created = _common(
            lambda: common.write_create_only_json(
                frozen_root_fd,
                frozen.MANIFEST_NAME,
                draft,
                "frozen candidate manifest",
            )
        )
        replay = _replayer(
            lambda: replayer._replay_rc3_frozen_candidate_v1_on_held_fds(  # noqa: SLF001
                frozen_root_fd,
                input_root_fd,
                source_root,
                source_root_fd,
            )
        )
        expected_descriptor = created.descriptor(
            frozen.MANIFEST_NAME,
            "created frozen candidate manifest",
        ).as_json()
        if replay.get("frozen_candidate_manifest") != expected_descriptor:
            _fail(
                "frozen-candidate-self-replay-mismatch",
                "self-replay did not return the durable created manifest descriptor",
            )
        _topology(
            lambda: topology.require_visible_root(
                frozen_root,
                frozen_root_fd,
                "frozen candidate root",
            )
        )
        _topology(
            lambda: topology.assert_existing_roots_disjoint(
                {
                    "source checkout": (source_root, source_root_fd),
                    "freeze-input evidence root": (input_root, input_root_fd),
                    "frozen candidate root": (frozen_root, frozen_root_fd),
                }
            )
        )
        return {
            "manifest": draft,
            "manifest_descriptor": expected_descriptor,
            "replay": replay,
        }
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
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--request", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = write_rc3_frozen_candidate_v1(
            args.frozen_candidate_root,
            input_evidence_root=args.input_evidence_root,
            repository_root=args.repository_root,
            expected_revision=args.expected_revision,
            candidate_id=args.candidate_id,
            request_name=args.request,
        )
    except (OSError, FrozenCandidateWriterError) as error:
        print(f"RC3 frozen candidate write failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
