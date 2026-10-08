#!/usr/bin/env python3
"""Private held-root raw compositor for native-fallback C02 lifecycle v5.

This module deliberately has no CLI, public reopen wrapper, callback, or
receipt publisher.  A future authenticated lifecycle runner may call its one
private helper only while it owns the same live private evidence-root FD and
exclusive root lock from request publication through the v5 binder's normal
return.  The returned document is raw ``bound/not-run`` provenance, never a
lifecycle-success or qualification decision.

In particular, an ambiguous final marker sync from the v5 binder raises and
does not create a continuation capability.  A later path-based structural
replay is not an authority to resume this chain.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, NoReturn, TypeVar

import bind_raw_c02_soak_v4 as v4_binder
import bind_raw_c02_soak_v5 as v5_binder
import check_c02_provenance_v2 as checker
import provenance_v2_common as common
import write_c02_lifecycle_bind_request_v5 as writer


BIND_REQUEST_NAME = "c02-lifecycle-v5-bind-request.json"
MANIFEST_NAME = "c02-lifecycle-v5-raw-manifest.json"


class C02LifecycleV5RawComposeError(ValueError):
    """The private native-fallback v5 raw chain cannot safely continue."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = C02LifecycleV5RawComposeError(message)
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
        _fail(getattr(error, "reason_code", "invalid-v5-bind-request"), str(error))


def _v4_binder(call: Callable[[], T]) -> T:
    try:
        return call()
    except v4_binder.RawSoakBindError as error:
        _fail(getattr(error, "reason_code", "invalid-v5-terminal-output"), str(error))


def _v5_binder(call: Callable[[], T]) -> T:
    try:
        return call()
    except v5_binder.RawSoakBindError as error:
        _fail(getattr(error, "reason_code", "invalid-v5-terminal-publication"), str(error))


def _assert_fixed_request_absent(root_fd: int) -> None:
    """Reserve the only fixed nonterminal output before any writer call."""

    try:
        os.lstat(BIND_REQUEST_NAME, dir_fd=root_fd)
    except FileNotFoundError:
        return
    except OSError as error:
        _fail(
            "output-preflight-failed",
            f"cannot inspect fixed v5 bind request {BIND_REQUEST_NAME!r}: {error}",
        )
    _fail(
        "output-name-collision",
        f"fixed v5 bind request {BIND_REQUEST_NAME!r} already exists",
    )


def _require_secure_reopen_matches_held_root_fd(root_fd: int, root_path: Path) -> None:
    """Retain no-follow ancestor validation for a caller-supplied root FD."""

    reopened_fd = _common(
        lambda: common.open_private_evidence_directory(
            root_path,
            "--evidence-root",
        )
    )
    try:
        try:
            held = os.fstat(root_fd)
            reopened = os.fstat(reopened_fd)
        except OSError as error:
            _fail(
                "evidence-root-fd-path-unavailable",
                f"cannot compare held and securely reopened evidence roots: {error}",
            )
        if (held.st_dev, held.st_ino) != (reopened.st_dev, reopened.st_ino):
            _fail(
                "evidence-root-fd-path-mismatch",
                "held evidence root FD differs from its securely reopened path",
            )
    finally:
        try:
            os.close(reopened_fd)
        except OSError as error:
            _fail(
                "evidence-root-fd-path-unavailable",
                f"cannot close securely reopened evidence root: {error}",
            )


def _write_and_bind_v5_held_locked_root_fd(
    root_fd: int,
    *,
    evidence_root: Path,
    bridge_report_path: Path,
    candidate_id: str,
    freeze_sha256: str,
    base_release_candidate_report_sha256: str,
) -> dict[str, Any]:
    """Write then bind only on one caller-held root-FD normal-return edge.

    The caller, not this helper, owns opening, closing, and holding ``LOCK_EX``
    on ``root_fd`` for the complete lexical invocation.  This helper has no
    continuation argument: a successful raw report cannot be turned into a
    receipt by a later call or by reopening the root.
    """

    root_path = _writer(lambda: writer._path(evidence_root, "--evidence-root"))
    _writer(lambda: writer._assert_external_to_source_checkout(root_path))
    report_path = _writer(lambda: writer._path(bridge_report_path, "--bridge-report"))
    candidate = _writer(lambda: writer._candidate_id(candidate_id, "expected candidate ID"))
    freeze = _writer(lambda: writer._sha256(freeze_sha256, "freeze SHA-256"))
    base_report = _writer(
        lambda: writer._sha256(
            base_release_candidate_report_sha256,
            "base release-candidate report SHA-256",
        )
    )
    _writer(lambda: writer._output_name(BIND_REQUEST_NAME))
    _v4_binder(lambda: v4_binder._manifest_name(MANIFEST_NAME))  # noqa: SLF001

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "C02 lifecycle v5 raw evidence root",
        )
    )
    _require_secure_reopen_matches_held_root_fd(root_fd, root_path)
    _writer(lambda: writer._require_evidence_root_fd_matches_path(root_fd, root_path))

    # Both reservations happen before the request is written.  The v5 binder
    # repeats terminal preflight immediately before its own publication.
    _assert_fixed_request_absent(root_fd)
    _v4_binder(
        lambda: v4_binder._assert_terminal_output_pair_absent(  # noqa: SLF001
            root_fd,
            MANIFEST_NAME,
        )
    )

    _writer(
        lambda: writer._write_c02_lifecycle_bind_request_v5_fd(  # noqa: SLF001
            root_fd,
            evidence_root=root_path,
            bridge_report_path=report_path,
            candidate_id=candidate,
            configuration_profile=checker.MAX_PERFORMANCE_EXACT_PROFILE,
            freeze_sha256=freeze,
            base_release_candidate_report_sha256=base_report,
            output_name=BIND_REQUEST_NAME,
        )
    )
    return _v5_binder(
        lambda: v5_binder._bind_raw_soak_manifest_held_locked_fd(  # noqa: SLF001
            root_fd,
            BIND_REQUEST_NAME,
            MANIFEST_NAME,
        )
    )
