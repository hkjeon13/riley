#!/usr/bin/env python3
"""Privately compose one fixed RC3 rollback finalizer receipt chain.

This module is the held-FD core for a future authenticated rollback runner.
It has no CLI, path-resume API, operational lifecycle action, or caller-chosen
output names.  Given only an already-exclusive private evidence-root FD and
the existing six-artifact preparation request, it keeps the normal-return
chain lexical:

``preparation -> atomic transaction -> fixed v3/v4 finalizer -> receipt``.

The dynamic candidate/config/source/rollback evidence must already exist under
their fixed paths.  This compositor neither creates nor interprets that live
evidence.  In particular, a terminal receipt pair is raw same-stack provenance
only; it does not prove host rollback, service lifecycle, GPU success, freeze,
semantic qualification, or promotion.
"""

from __future__ import annotations

import fcntl
from typing import Any, Callable, NoReturn, TypeVar

import capture_rc3_rollback_atomic_transaction_v1 as transaction
import capture_rc3_rollback_phase_v1 as phase_capture
import finalize_rc3_rollback_candidate_source_v4 as fixed_finalizer
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common
import replay_rc3_rollback_candidate_source_v1 as candidate_source
import write_rc3_rollback_candidate_source_bind_request_v1 as writer
import write_rc3_rollback_finalizer_receipt_v1 as receipt


class RollbackFinalizerReceiptComposeError(ValueError):
    """The fixed rollback receipt chain cannot safely continue."""


def _fail(code: str, message: str) -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = RollbackFinalizerReceiptComposeError(message)
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
        _fail(getattr(error, "reason_code", "invalid-artifact-preparation"), str(error))


def _transaction(call: Callable[[], T]) -> T:
    try:
        return call()
    except transaction.RollbackAtomicTransactionError as error:
        _fail(getattr(error, "reason_code", "invalid-atomic-transaction"), str(error))


def _fixed_finalizer(call: Callable[[], T]) -> T:
    try:
        return call()
    except fixed_finalizer.RollbackCandidateSourceFinalizerError as error:
        _fail(getattr(error, "reason_code", "invalid-fixed-finalizer"), str(error))


def _candidate_source(call: Callable[[], T]) -> T:
    try:
        return call()
    except candidate_source.CandidateSourceJoinError as error:
        _fail(getattr(error, "reason_code", "invalid-candidate-source"), str(error))


def _phase(call: Callable[[], T]) -> T:
    try:
        return call()
    except phase_capture.RollbackPhaseCaptureError as error:
        _fail(getattr(error, "reason_code", "invalid-rollback-phase"), str(error))


def _writer(call: Callable[[], T]) -> T:
    try:
        return call()
    except writer.RollbackCandidateSourceBindRequestError as error:
        _fail(getattr(error, "reason_code", "invalid-fixed-bind-request"), str(error))


def _receipt(call: Callable[[], T]) -> T:
    try:
        return call()
    except receipt.RollbackFinalizerReceiptError as error:
        _fail(getattr(error, "reason_code", "invalid-finalizer-receipt"), str(error))


def _lock_switch_exclusive(switch_fd: int) -> None:
    try:
        fcntl.flock(switch_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("lock-unavailable", f"cannot acquire exclusive rollback switch lock: {error}")


def _unlock_switch_quietly(switch_fd: int | None) -> None:
    if switch_fd is not None:
        try:
            fcntl.flock(switch_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _preflight_fixed_terminal_outputs(root_fd: int) -> None:
    """Reject all fixed finalizer/receipt collisions before mutation."""

    _fixed_finalizer(lambda: fixed_finalizer._assert_all_fixed_outputs_absent(root_fd))  # noqa: SLF001
    _receipt(lambda: receipt._assert_receipt_names_are_distinct_from_finalizer())  # noqa: SLF001
    _receipt(lambda: receipt._assert_receipt_outputs_absent(root_fd))  # noqa: SLF001


def _replay_dynamic_raw_evidence_once(
    root_fd: int,
) -> tuple[candidate_source.ReplayedCandidateSourceJoin, phase_capture.ReplayedPhaseCapture]:
    """Replay every live input the fixed writer can validate before mutation."""

    joined = _candidate_source(
        lambda: candidate_source._replay_candidate_source_join_on_held_root_fd(  # noqa: SLF001
            root_fd
        )
    )
    rollback = _phase(
        lambda: phase_capture.replay_rc3_rollback_phase_v1_fd(
            root_fd,
            writer.ROLLBACK_PHASE_CAPTURE_NAME,
        )
    )
    if (
        joined.candidate_phase.target.server_pid,
        joined.candidate_phase.target.server_start_ticks,
    ) == (rollback.target.server_pid, rollback.target.server_start_ticks):
        _fail(
            "reused-candidate-process",
            "candidate and rollback phase must use distinct PID/start-tick identities",
        )
    _candidate_request, candidate_descriptors = _writer(
        lambda: writer._candidate_request(joined)  # noqa: SLF001
    )
    _rollback_request, rollback_descriptors = _writer(
        lambda: writer._rollback_request(rollback)  # noqa: SLF001
    )
    candidate_paths = _writer(
        lambda: writer._candidate_source_consumed_paths(  # noqa: SLF001
            joined,
            candidate_descriptors,
        )
    )
    _writer(
        lambda: writer._assert_no_cross_role_path_reuse(  # noqa: SLF001
            candidate_paths,
            (("rollback raw evidence", rollback_descriptors),),
        )
    )
    return joined, rollback


def _preflight_dynamic_raw_evidence(root_fd: int) -> None:
    """Admit the fixed live closure before preparation can mutate the root.

    The future writer/finalizer must replay these inputs again after the
    preparation and atomic transaction exist, because those later products
    are themselves part of the final closure.  This earlier, read-only pass
    intentionally covers the dynamic candidate/config/source and rollback
    phase inputs that already exist: malformed live evidence must not leave a
    fresh snapshot, artifact, or switch surface behind.
    """

    initial = _replay_dynamic_raw_evidence_once(root_fd)
    terminal = _replay_dynamic_raw_evidence_once(root_fd)
    if terminal != initial:
        _fail(
            "dynamic-evidence-replay-drift",
            "dynamic candidate/config/source/rollback evidence changed during held-FD preflight",
        )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback finalizer receipt evidence root",
        )
    )


def _prepare_transaction_and_write_fixed_receipt_on_held_root_fd(
    root_fd: int,
    request: prepare.PreparationRequest,
) -> dict[str, Any]:
    """Close the fixed receipt only from one caller-held root lock stack.

    The caller owns a no-follow private evidence-root FD and its nonblocking
    exclusive lock.  No input supplies a candidate target, config path,
    request/manifest name, receipt name, descriptor, or continuation: all are
    re-derived by the fixed finalizer from the held root.  If any preparation,
    atomic, v4, or receipt terminal publication is ambiguous, the relevant
    successor is never invoked and no fresh path-based call can resume it.
    """

    if not isinstance(request, prepare.PreparationRequest):
        _fail("invalid-preparation-request", "request must be a PreparationRequest")
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback finalizer receipt evidence root",
        )
    )
    # Do this before preparation creates any child.  An occupied fixed final
    # output must not leave a new preparation/transaction that another call
    # might try to pair with a later terminal receipt.
    _preflight_fixed_terminal_outputs(root_fd)
    _preflight_dynamic_raw_evidence(root_fd)

    def after_normal_preparation(
        _preparation_replay: prepare.ArtifactPreparationReplay,
        switch_fd: int,
    ) -> dict[str, Any]:
        """Hold switch EX through the one terminal transaction callback."""

        _lock_switch_exclusive(switch_fd)
        try:
            _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
            # The root EX remains held, but repeat the reservation after
            # preparation so no terminal callback begins on a stale surface.
            _preflight_fixed_terminal_outputs(root_fd)

            def after_normal_transaction(
                _transaction_replay: transaction.AtomicTransactionReplay,
            ) -> dict[str, Any]:
                # Receipt publication is the lexical terminal operation.  The
                # transaction's terminal-continuation primitive makes no
                # post-callback replay that could invalidate this success edge.
                return _receipt(
                    lambda: receipt._finalize_and_write_rollback_receipt_on_held_root_switch_fds(  # noqa: SLF001
                        root_fd,
                        switch_fd,
                    )
                )

            return _transaction(
                lambda: transaction._capture_atomic_transaction_then_terminal_success_held_switch_fd(  # noqa: SLF001
                    root_fd,
                    switch_fd,
                    after_normal_transaction,
                )
            )
        finally:
            _unlock_switch_quietly(switch_fd)

    # This terminal variant leaves only quiet FD cleanup after a successful
    # receipt hardlink, preserving the finalizer's normal-return authority.
    return _prepare(
        lambda: prepare._prepare_artifacts_then_terminal_success_held_root_fd(  # noqa: SLF001
            request,
            root_fd,
            after_normal_preparation,
        )
    )
