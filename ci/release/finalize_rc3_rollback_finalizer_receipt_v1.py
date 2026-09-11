#!/usr/bin/env python3
"""Private runner-only entry for one fixed RC3 rollback receipt chain.

The future authenticated remote runner calls the one private helper here after
it has captured every dynamic candidate/source/rollback/config input at its
fixed paths.  This module only opens and exclusively locks the private
evidence root, then enters the held-FD compositor once.  It cannot resume an
existing preparation, transaction, v3/v4 marker, or receipt pair.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any, Callable, NoReturn, TypeVar

import compose_rc3_rollback_finalizer_receipt_v1 as composer
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common


class AuthenticatedRollbackFinalizationError(ValueError):
    """The authenticated rollback raw chain cannot safely complete."""


def _fail(code: str, message: str) -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = AuthenticatedRollbackFinalizationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _prepare(call: Callable[[], T]) -> T:
    try:
        return call()
    except prepare.RollbackArtifactPreparationError as error:
        _fail(getattr(error, "reason_code", "invalid-preparation-request"), str(error))


def _composer(call: Callable[[], T]) -> T:
    try:
        return call()
    except composer.RollbackFinalizerReceiptComposeError as error:
        _fail(getattr(error, "reason_code", "invalid-rollback-receipt-chain"), str(error))


def _lock_root_exclusive(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("lock-unavailable", f"cannot acquire exclusive rollback evidence-root lock: {error}")


def _unlock_root_quietly(root_fd: int | None) -> None:
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


def _finalize_authenticated_rollback_raw_once(
    request: prepare.PreparationRequest,
) -> dict[str, Any]:
    """Run one fixed raw receipt chain under one fresh private root EX lock.

    The six external artifact paths are the only caller-controlled inputs and
    are snapshot by the existing preparation primitive.  Candidate identity,
    config, source, phase, bind request, v3/v4 names, receipt name, target,
    and all descriptors are fixed and re-derived from preexisting evidence.
    """

    if not isinstance(request, prepare.PreparationRequest):
        _fail("invalid-preparation-request", "request must be a PreparationRequest")
    evidence_root = request.evidence_root
    if not isinstance(evidence_root, Path):
        _fail("invalid-preparation-request", "PreparationRequest.evidence_root must be a Path")
    _prepare(lambda: prepare._assert_external_to_source_checkout(evidence_root))  # noqa: SLF001
    root_fd = _common(
        lambda: common.open_private_evidence_directory(
            evidence_root,
            "authenticated rollback evidence root",
        )
    )
    try:
        _lock_root_exclusive(root_fd)
        return _composer(
            lambda: composer._prepare_transaction_and_write_fixed_receipt_on_held_root_fd(  # noqa: SLF001
                root_fd,
                request,
            )
        )
    finally:
        # A receipt hardlink success must not be turned into a later error by
        # best-effort lock/FD cleanup in this outer runner-only wrapper.
        _unlock_root_quietly(root_fd)
        _close_quietly(root_fd)
