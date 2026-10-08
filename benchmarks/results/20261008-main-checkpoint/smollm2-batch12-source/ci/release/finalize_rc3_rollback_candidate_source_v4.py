#!/usr/bin/env python3
"""Publish one fixed rollback v3/v4 closure only from one held-FD stack.

This private compositor is deliberately not an operational runner and not a
path-based resume API.  Its report-returning compatibility helper and its
typed same-stack continuation helper accept only the caller-owned private
evidence-root FD and already exclusively held rollback-switch FD.  While both
locks remain held, they write the fixed candidate/source v3 request, compare a
fresh replay of every consumed raw input plus that request's descriptor before
v3 publication, then publish fixed v3 and v4 raw manifests.  The continuation
retains the successful closure only for the immediately nested receipt writer;
a path-only v3/v4 replay cannot silently consume leaves changed after the
writer returned.

It never opens an evidence-root path, takes or releases caller locks, starts a
service, contacts a GPU, performs an artifact exchange, or claims rollback or
qualification success.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NoReturn, TypeVar

import bind_raw_rc3_rollback_capture as v3_binder
import capture_rc3_rollback_atomic_transaction_v1 as transaction
import check_rc3_rollback_provenance_v4 as v4_checker
import provenance_v2_common as common
import write_rc3_rollback_candidate_source_bind_request_v1 as writer


ROLLBACK_V3_MANIFEST_NAME = "rollback-v3-candidate-source-manifest.json"
ROLLBACK_V4_MANIFEST_NAME = "rollback-v4-candidate-source-manifest.json"


class RollbackCandidateSourceFinalizerError(ValueError):
    """The fixed rollback candidate/source closure cannot become terminal."""


@dataclass(frozen=True)
class _FinalizedRollbackCandidateSourceV4:
    """One successful same-stack finalizer result retained for its caller.

    This is intentionally private typed state, not a resumable publication
    token.  A direct caller can pass it only to an immediately nested
    normal-return consumer that retains the same root and switch descriptors.
    The on-disk v4 completion pair alone cannot reconstruct this success edge
    after a post-link directory-sync ambiguity.
    """

    written: writer.WrittenCandidateSourceBindRequest
    v3_descriptor: common.EvidenceDescriptor
    v3_report: Mapping[str, Any]
    v4_descriptor: common.EvidenceDescriptor
    v4_report: Mapping[str, Any]


def _fail(code: str, message: str) -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = RollbackCandidateSourceFinalizerError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _writer(call: Callable[[], T]) -> T:
    try:
        return call()
    except writer.RollbackCandidateSourceBindRequestError as error:
        _fail(getattr(error, "reason_code", "invalid-candidate-source-writer"), str(error))


def _v3(call: Callable[[], T]) -> T:
    try:
        return call()
    except v3_binder.RollbackBindError as error:
        _fail(getattr(error, "reason_code", "invalid-rollback-v3"), str(error))


def _v4(call: Callable[[], T]) -> T:
    try:
        return call()
    except v4_checker.RollbackV4ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-rollback-v4"), str(error))


def _transaction(call: Callable[[], T]) -> T:
    try:
        return call()
    except transaction.RollbackAtomicTransactionError as error:
        _fail(getattr(error, "reason_code", "invalid-atomic-transaction"), str(error))


def _output_names(name: str) -> tuple[str, str, str]:
    return name, f"{name}.intent", f"{name}.complete"


def _all_fixed_output_names() -> tuple[str, ...]:
    names = (
        *writer._output_names(),  # noqa: SLF001 - writer's closed fixed request names
        *_output_names(ROLLBACK_V3_MANIFEST_NAME),
        *_output_names(ROLLBACK_V4_MANIFEST_NAME),
    )
    if len(set(names)) != len(names):
        _fail("fixed-output-name-alias", "fixed writer, v3, and v4 output names must differ")
    return names


def _assert_all_fixed_outputs_absent(root_fd: int) -> None:
    for name in _all_fixed_output_names():
        try:
            os.lstat(name, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail(
                "output-preflight-failed",
                f"cannot inspect fixed rollback finalizer output {name!r}: {error}",
            )
        _fail(
            "output-name-collision",
            f"fixed rollback finalizer output or reserved sibling {name!r} already exists",
        )


def _expected_v3_bindings(
    written: writer.WrittenCandidateSourceBindRequest,
) -> dict[str, str]:
    bindings = written.static_bindings
    if (
        written.candidate_source.static_effective.candidate_id != bindings.candidate_id
        or written.candidate_source.static_effective.configuration_profile
        != bindings.configuration_profile
    ):
        _fail(
            "writer-static-identity-mismatch",
            "writer result does not retain one static preparation identity",
        )
    return {
        "freeze_sha256": bindings.freeze.sha256,
        "base_release_candidate_report_sha256": bindings.base_release_candidate_report.sha256,
        "configuration_profile": bindings.configuration_profile,
        "configuration_sha256": bindings.configuration.sha256,
    }


def _recheck_written_request(
    root_fd: int,
    written: writer.WrittenCandidateSourceBindRequest,
) -> None:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            writer.BIND_REQUEST_NAME,
            "fixed rollback candidate/source bind request",
            maximum_bytes=v3_binder.MAX_BIND_REQUEST_BYTES,
        )
    )
    descriptor = _common(
        lambda: common.descriptor_for_bytes(
            writer.BIND_REQUEST_NAME,
            raw,
            "fixed rollback candidate/source bind request",
        )
    )
    if descriptor != written.request_descriptor:
        _fail(
            "bind-request-descriptor-drift",
            "fixed rollback bind request bytes changed after writer publication",
        )
    document = _common(
        lambda: common.parse_canonical_json(
            raw,
            "fixed rollback candidate/source bind request",
            maximum_bytes=v3_binder.MAX_BIND_REQUEST_BYTES,
        )
    )
    if document != written.request:
        _fail(
            "bind-request-document-drift",
            "fixed rollback bind request document differs from the writer result",
        )


def _recheck_written_state(
    root_fd: int,
    switch_fd: int,
    written: writer.WrittenCandidateSourceBindRequest,
) -> None:
    """Compare the complete writer closure and fixed request at one use point."""

    if not isinstance(written, writer.WrittenCandidateSourceBindRequest):
        _fail("invalid-writer-result", "finalizer requires the typed same-stack writer result")
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "fixed rollback finalizer evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    replayed = _writer(lambda: writer._replay_inputs(root_fd, switch_fd))  # noqa: SLF001
    if replayed.candidate_source.static_effective.static_bindings != written.static_bindings:
        _fail(
            "static-preparation-replay-drift",
            "static preparation identity or immutable bindings changed after writer publication",
        )
    if replayed.candidate_source != written.candidate_source:
        _fail(
            "candidate-source-replay-drift",
            "candidate/source closure changed after writer publication",
        )
    if replayed.rollback_phase != written.rollback_phase:
        _fail(
            "rollback-phase-replay-drift",
            "rollback phase closure changed after writer publication",
        )
    if replayed.atomic_transaction != written.atomic_transaction:
        _fail(
            "atomic-transaction-replay-drift",
            "atomic transaction closure changed after writer publication",
        )
    _recheck_written_request(root_fd, written)
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "fixed rollback finalizer evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001


def _replay_v3_transaction(
    root_fd: int,
    switch_fd: int,
    written: writer.WrittenCandidateSourceBindRequest,
    *,
    expected_v3_descriptor: common.EvidenceDescriptor | None = None,
) -> tuple[common.EvidenceDescriptor, Mapping[str, Any]]:
    descriptor, report, replayed_transaction = _v4(
        lambda: v4_checker._replay_inputs_on_held_switch_fd(  # noqa: SLF001
            root_fd,
            switch_fd,
            ROLLBACK_V3_MANIFEST_NAME,
        )
    )
    if expected_v3_descriptor is not None and descriptor != expected_v3_descriptor:
        _fail(
            "rollback-v3-descriptor-drift",
            "rollback v3 manifest changed after its first held-FD replay",
        )
    if replayed_transaction != written.atomic_transaction:
        _fail(
            "atomic-transaction-replay-drift",
            "v3 transaction replay differs from the writer closure",
        )
    candidate_id = written.static_bindings.candidate_id
    if report.get("candidate_id") != candidate_id:
        _fail(
            "rollback-v3-candidate-id-mismatch",
            "rollback v3 report candidate ID differs from the writer static identity",
        )
    if report.get("bindings") != _expected_v3_bindings(written):
        _fail(
            "rollback-v3-bindings-mismatch",
            "rollback v3 report bindings differ from the writer static bindings",
        )
    return descriptor, report


def _completion_pair_is_visible(root_fd: int, name: str) -> bool:
    try:
        final = os.lstat(f"{name}.complete", dir_fd=root_fd)
        intent = os.lstat(f"{name}.intent", dir_fd=root_fd)
    except OSError:
        return False
    return (
        stat.S_ISREG(final.st_mode)
        and stat.S_ISREG(intent.st_mode)
        and stat.S_IMODE(final.st_mode) == 0o600
        and stat.S_IMODE(intent.st_mode) == 0o600
        and final.st_uid == os.geteuid()
        and intent.st_uid == os.geteuid()
        and final.st_nlink == 2
        and intent.st_nlink == 2
        and (final.st_dev, final.st_ino) == (intent.st_dev, intent.st_ino)
    )


def _finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds(
    root_fd: int,
    switch_fd: int,
) -> _FinalizedRollbackCandidateSourceV4:
    """Write fixed request → v3 → v4 only while caller root/switch EX persist.

    This private function has no resume path.  A failure after a create-only
    leaf or completion pair leaves that evidence visible and must not be
    restarted from a fresh path/FD invocation.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "fixed rollback finalizer evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    _assert_all_fixed_outputs_absent(root_fd)

    written = _writer(
        lambda: writer._write_fixed_candidate_source_bind_request_on_held_root_switch_fds(  # noqa: SLF001
            root_fd,
            switch_fd,
        )
    )
    _recheck_written_state(root_fd, switch_fd, written)
    _v3(
        lambda: v3_binder._bind_raw_rollback_manifest_held_locked_fd(  # noqa: SLF001
            root_fd,
            writer.BIND_REQUEST_NAME,
            ROLLBACK_V3_MANIFEST_NAME,
        )
    )
    v3_descriptor, v3_report = _replay_v3_transaction(root_fd, switch_fd, written)
    bindings = v3_report.get("bindings")
    candidate_id = v3_report.get("candidate_id")
    if type(candidate_id) is not str or not isinstance(bindings, Mapping):
        _fail("invalid-rollback-v3-report", "held rollback v3 report lacks identity bindings")
    manifest = {
        "schema_version": v4_checker.ROLLBACK_V4_MANIFEST_VERSION,
        "capture_status": "captured",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "bindings": dict(bindings),
        "rollback_v3_manifest": v3_descriptor.as_json(),
        "atomic_transaction_session": written.atomic_transaction.session_descriptor.as_json(),
    }
    raw_manifest = _common(lambda: common.canonical_json_bytes(manifest))
    draft_descriptor = _common(
        lambda: common.descriptor_for_bytes(
            ROLLBACK_V4_MANIFEST_NAME,
            raw_manifest,
            "fixed rollback v4 draft manifest",
        )
    )
    preflight = _v4(
        lambda: v4_checker.verify_rollback_provenance_v4_bytes_on_held_switch_fd(
            root_fd,
            switch_fd,
            draft_descriptor,
            raw_manifest,
        )
    )
    created = _common(
        lambda: common.write_create_only_json(
            root_fd,
            ROLLBACK_V4_MANIFEST_NAME,
            manifest,
            "fixed rollback v4 raw manifest",
        )
    )
    if created.descriptor(ROLLBACK_V4_MANIFEST_NAME, "fixed rollback v4 created manifest") != draft_descriptor:
        _fail(
            "published-manifest-descriptor-mismatch",
            "create-only v4 publication bytes differ from the held-FD preflight",
        )
    replayed_v4 = _v4(
        lambda: v4_checker.verify_rollback_provenance_v4_on_held_switch_fd(
            root_fd,
            switch_fd,
            ROLLBACK_V4_MANIFEST_NAME,
        )
    )
    if replayed_v4 != preflight:
        _fail(
            "post-publication-replay-drift",
            "on-disk rollback v4 replay differs from its held-FD preflight",
        )

    # The v4 manifest is intentionally still nonterminal here.  Recheck the
    # complete writer closure immediately before publishing the paired marker.
    _recheck_written_state(root_fd, switch_fd, written)
    terminal_v3_descriptor, terminal_v3_report = _replay_v3_transaction(
        root_fd,
        switch_fd,
        written,
        expected_v3_descriptor=v3_descriptor,
    )
    if terminal_v3_report != v3_report:
        _fail(
            "rollback-v3-replay-drift",
            "rollback v3 report changed before terminal v4 publication",
        )
    terminal_v4 = _v4(
        lambda: v4_checker.verify_rollback_provenance_v4_on_held_switch_fd(
            root_fd,
            switch_fd,
            ROLLBACK_V4_MANIFEST_NAME,
        )
    )
    if terminal_v4 != preflight or terminal_v3_descriptor != v3_descriptor:
        _fail(
            "rollback-v4-replay-drift",
            "rollback v4 closure changed before terminal publication",
        )
    marker = {
        "schema_version": v4_checker.ROLLBACK_V4_COMPLETION_VERSION,
        "artifact_filename": ROLLBACK_V4_MANIFEST_NAME,
        "artifact_sha256": created.sha256,
    }
    intent_name = f"{ROLLBACK_V4_MANIFEST_NAME}.intent"
    _common(
        lambda: common.write_create_only_json(
            root_fd,
            intent_name,
            marker,
            "fixed rollback v4 completion marker intent",
        )
    )
    try:
        _common(
            lambda: common.publish_create_only_hardlink(
                root_fd,
                intent_name,
                f"{ROLLBACK_V4_MANIFEST_NAME}.complete",
                "fixed rollback v4 completion marker",
            )
        )
    except RollbackCandidateSourceFinalizerError:
        if _completion_pair_is_visible(root_fd, ROLLBACK_V4_MANIFEST_NAME):
            _fail(
                "ambiguous-terminal-publication",
                "completion marker became visible but final directory sync failed; "
                "no later invocation may treat it as producer success",
            )
        raise
    completed_v4 = _v4(
        lambda: v4_checker.verify_completed_rollback_provenance_v4_on_held_switch_fd(
            root_fd,
            switch_fd,
            ROLLBACK_V4_MANIFEST_NAME,
        )
    )
    if completed_v4 != preflight:
        _fail(
            "rollback-v4-replay-drift",
            "completed rollback v4 replay differs from its held-FD preflight",
        )
    return _FinalizedRollbackCandidateSourceV4(
        written=written,
        v3_descriptor=v3_descriptor,
        v3_report=v3_report,
        v4_descriptor=draft_descriptor,
        v4_report=completed_v4,
    )


def _finalize_rollback_candidate_source_v4_on_held_root_switch_fds(
    root_fd: int,
    switch_fd: int,
) -> dict[str, Any]:
    """Compatibility result for the fixed same-stack finalizer.

    The receipt-only continuation uses the private typed closure above.  This
    existing narrow helper continues to return the raw v4 replay report so it
    does not create an external continuation or path-resume surface.
    """

    return _finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds(
        root_fd,
        switch_fd,
    ).v4_report
