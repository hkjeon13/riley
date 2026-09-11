#!/usr/bin/env python3
"""Private terminal raw-v5 finalizer for the authenticated C02 runner.

This module intentionally has no command-line entry point or public finalize,
resume, or receipt API.  The authenticated runner invokes the private helper
once after its source-owned shutdown check, with fixed names retained by the
underlying compositor.  It opens and exclusively locks one private evidence
root only for the writer-to-binder normal-return edge; it cannot turn an
existing marker pair into a later lifecycle-success decision.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any, Callable, NoReturn, TypeVar

import compose_c02_lifecycle_v5_raw as composer
import provenance_v2_common as common
import write_c02_lifecycle_bind_request_v5 as writer


class C02LifecycleV5RawFinalizationError(ValueError):
    """The authenticated runner cannot safely complete its raw-v5 edge."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = C02LifecycleV5RawFinalizationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _writer(call: Callable[[], T]) -> T:
    try:
        return call()
    except writer.C02LifecycleV5BindRequestError as error:
        _fail(getattr(error, "reason_code", "invalid-v5-input"), str(error))


def _composer(call: Callable[[], T]) -> T:
    try:
        return call()
    except composer.C02LifecycleV5RawComposeError as error:
        _fail(getattr(error, "reason_code", "invalid-v5-raw-chain"), str(error))


def _lock(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("output-lock-unavailable", f"cannot acquire exclusive v5 evidence-root lock: {error}")


def _unlock_quietly(root_fd: int | None) -> None:
    if root_fd is not None:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(root_fd: int | None) -> None:
    if root_fd is not None:
        try:
            os.close(root_fd)
        except OSError:
            pass


def _finalize_authenticated_v5_raw_once(
    *,
    evidence_root: Path,
    bridge_report_path: Path,
    candidate_id: str,
    freeze_sha256: str,
    base_release_candidate_report_sha256: str,
) -> dict[str, Any]:
    """Run the fixed raw v5 terminal edge once under a freshly held root lock.

    This private runner-only helper deliberately accepts neither output names
    nor a continuation.  The compositor refuses every pre-existing fixed
    request or terminal sibling, so a failure or ambiguous marker cannot be
    retried into a distinct successful terminal chain.
    """

    root_path = _writer(lambda: writer._path(evidence_root, "--evidence-root"))
    _writer(lambda: writer._assert_external_to_source_checkout(root_path))
    root_fd = _common(
        lambda: common.open_private_evidence_directory(root_path, "--evidence-root")
    )
    try:
        _lock(root_fd)
        return _composer(
            lambda: composer._write_and_bind_v5_held_locked_root_fd(  # noqa: SLF001
                root_fd,
                evidence_root=root_path,
                bridge_report_path=bridge_report_path,
                candidate_id=candidate_id,
                freeze_sha256=freeze_sha256,
                base_release_candidate_report_sha256=base_release_candidate_report_sha256,
            )
        )
    finally:
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)
