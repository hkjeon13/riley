#!/usr/bin/env python3
"""Write one fixed nonterminal RC3 rollback v3 bind request from held evidence.

This module is deliberately a private held-FD compositor primitive rather than
an operational runner.  Its only callable entry accepts an already-held
private evidence-root FD and the already-held fixed switch FD; the caller owns
both exclusive locks for the complete lexical invocation.  It replays fixed
candidate/source/config/shutdown evidence, the fixed rollback phase, and the
fixed artifact-exchange transaction, then creates one fixed path-only v3 bind
request.  It never starts a service, contacts a GPU, performs an exchange,
binds a v3 manifest, publishes a completion marker, or decides rollback or
qualification.

The request is not a continuation capability.  A future one-shot finalizer
must recheck the returned static bindings immediately before v3 publication
and again before its later terminal publication while retaining the same held
root/switch lock stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NoReturn, TypeVar

import bind_raw_rc3_rollback_capture as v3_binder
import capture_rc3_rollback_atomic_transaction_v1 as transaction
import capture_rc3_rollback_phase_v1 as phase_capture
import check_rc3_rollback_provenance_v3 as v3_checker
import check_rc3_static_effective_config_v1 as static_effective
import provenance_v2_common as common
import replay_rc3_rollback_candidate_source_v1 as candidate_source


BIND_REQUEST_NAME = "rollback-v3-candidate-source-bind-request.json"
ROLLBACK_PHASE_CAPTURE_NAME = "rollback-phase"


class RollbackCandidateSourceBindRequestError(ValueError):
    """Fixed rollback evidence cannot safely produce its v3 bind request."""


def _fail(code: str, message: str) -> NoReturn:
    error = RollbackCandidateSourceBindRequestError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


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


def _static_effective(call: Callable[[], T]) -> T:
    try:
        return call()
    except static_effective.StaticEffectiveConfigError as error:
        _fail(
            getattr(error, "reason_code", "invalid-static-effective-config"),
            str(error),
        )


def _transaction(call: Callable[[], T]) -> T:
    try:
        return call()
    except transaction.RollbackAtomicTransactionError as error:
        _fail(getattr(error, "reason_code", "invalid-atomic-transaction"), str(error))


@dataclass(frozen=True)
class WrittenCandidateSourceBindRequest:
    """Typed raw state that a future same-stack finalizer must recheck."""

    request: Mapping[str, Any]
    request_descriptor: common.EvidenceDescriptor
    candidate_source: candidate_source.ReplayedCandidateSourceJoin
    rollback_phase: phase_capture.ReplayedPhaseCapture
    atomic_transaction: transaction.AtomicTransactionReplay
    static_bindings: static_effective.StaticPreparationBindings


@dataclass(frozen=True)
class _ReplayedWriterInputs:
    candidate_source: candidate_source.ReplayedCandidateSourceJoin
    rollback_phase: phase_capture.ReplayedPhaseCapture
    atomic_transaction: transaction.AtomicTransactionReplay
    candidate_artifacts: Mapping[str, common.EvidenceDescriptor]
    rollback_artifacts: Mapping[str, common.EvidenceDescriptor]
    atomic_switch: Mapping[str, common.EvidenceDescriptor]


def _output_names() -> tuple[str, str, str]:
    """Reserve terminal-looking siblings even though this writer never creates them."""

    return (
        BIND_REQUEST_NAME,
        f"{BIND_REQUEST_NAME}.intent",
        f"{BIND_REQUEST_NAME}.complete",
    )


def _assert_fixed_output_absent(root_fd: int) -> None:
    for name in _output_names():
        try:
            os.lstat(name, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail(
                "output-preflight-failed",
                f"cannot inspect fixed rollback bind request output {name!r}: {error}",
            )
        _fail(
            "output-name-collision",
            f"fixed rollback bind request output or reserved sibling {name!r} already exists",
        )


def _descriptor_map(
    value: Mapping[str, common.EvidenceDescriptor],
    fields: frozenset[str],
    label: str,
) -> dict[str, common.EvidenceDescriptor]:
    if set(value) != set(fields):
        _fail(
            "unexpected-descriptor-map",
            f"{label} fields differ; expected={sorted(fields)}, actual={sorted(value)}",
        )
    result: dict[str, common.EvidenceDescriptor] = {}
    for field in sorted(fields):
        descriptor = value[field]
        if not isinstance(descriptor, common.EvidenceDescriptor):
            _fail(
                "invalid-evidence-descriptor",
                f"{label}.{field} is not a typed evidence descriptor",
            )
        result[field] = descriptor
    return result


def _session_descriptor_map(
    value: Any,
    fields: frozenset[str],
    label: str,
) -> dict[str, common.EvidenceDescriptor]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(
            "unexpected-field-set",
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
        )
    result: dict[str, common.EvidenceDescriptor] = {}
    for field in sorted(fields):
        result[field] = _common(
            lambda field=field: common.parse_descriptor(
                value[field],
                f"{label}.{field}",
            )
        )
    return result


def _transaction_descriptor_maps(
    replayed: transaction.AtomicTransactionReplay,
) -> tuple[
    dict[str, common.EvidenceDescriptor],
    dict[str, common.EvidenceDescriptor],
    dict[str, common.EvidenceDescriptor],
]:
    """Recover only the fixed descriptor maps the transaction already replayed."""

    preparation_session = replayed.preparation_session
    if not isinstance(preparation_session, Mapping):
        _fail(
            "invalid-atomic-transaction",
            "atomic transaction lacks its typed preparation session",
        )
    snapshots = preparation_session.get("artifact_snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != {"candidate", "rollback"}:
        _fail(
            "invalid-atomic-transaction",
            "atomic transaction preparation lacks exact artifact snapshot arms",
        )
    candidate_artifacts = _session_descriptor_map(
        snapshots["candidate"],
        v3_checker.ARTIFACT_FIELDS,
        "atomic transaction candidate artifact snapshots",
    )
    rollback_artifacts = _session_descriptor_map(
        snapshots["rollback"],
        v3_checker.ARTIFACT_FIELDS,
        "atomic transaction rollback artifact snapshots",
    )
    atomic_session = replayed.atomic_switch_session
    if not isinstance(atomic_session, Mapping):
        _fail(
            "invalid-atomic-transaction",
            "atomic transaction lacks its typed atomic-switch session",
        )
    atomic_switch = _session_descriptor_map(
        atomic_session.get("atomic_switch"),
        v3_checker.ATOMIC_SWITCH_FIELDS,
        "atomic transaction atomic-switch descriptors",
    )
    return candidate_artifacts, rollback_artifacts, atomic_switch


def _path_map(
    descriptors: Mapping[str, common.EvidenceDescriptor],
    fields: frozenset[str],
    label: str,
) -> dict[str, str]:
    checked = _descriptor_map(descriptors, fields, label)
    return {f"{field}_path": checked[field].path for field in sorted(fields)}


def _candidate_request(
    replayed: candidate_source.ReplayedCandidateSourceJoin,
) -> tuple[dict[str, Any], tuple[common.EvidenceDescriptor, ...]]:
    phase = replayed.candidate_phase
    if phase.generation is not None:
        _fail(
            "candidate-local-generation-forbidden",
            "candidate phase must not provide a local generation exchange",
        )
    process = _descriptor_map(
        phase.process_evidence,
        v3_checker.RAW_PROCESS_FIELDS,
        "candidate phase process evidence",
    )
    health = _descriptor_map(
        phase.health,
        v3_checker.HTTP_EXCHANGE_FIELDS,
        "candidate phase health",
    )
    generation = {
        "request": replayed.generation.request,
        "response_head": replayed.generation.response_head,
        "response_body": replayed.generation.response_body,
    }
    generation_checked = _descriptor_map(
        generation,
        v3_checker.HTTP_EXCHANGE_FIELDS,
        "candidate source-owned generation",
    )
    audit = replayed.generation.generation_audit_index
    shutdown_artifact = replayed.shutdown.artifact
    shutdown_marker = replayed.shutdown.marker
    for descriptor, label in (
        (audit, "candidate source generation audit index"),
        (shutdown_artifact, "candidate source shutdown artifact"),
        (shutdown_marker, "candidate source shutdown marker"),
    ):
        if not isinstance(descriptor, common.EvidenceDescriptor):
            _fail(
                "invalid-evidence-descriptor",
                f"{label} is not a typed evidence descriptor",
            )
    return (
        {
            "process_evidence": _path_map(
                process,
                v3_checker.RAW_PROCESS_FIELDS,
                "candidate phase process evidence",
            ),
            "health": _path_map(
                health,
                v3_checker.HTTP_EXCHANGE_FIELDS,
                "candidate phase health",
            ),
            "generation": _path_map(
                generation_checked,
                v3_checker.HTTP_EXCHANGE_FIELDS,
                "candidate source-owned generation",
            ),
            "generation_audit_index_path": audit.path,
            "shutdown_artifact_path": shutdown_artifact.path,
            "shutdown_marker_path": shutdown_marker.path,
        },
        tuple(process.values())
        + tuple(health.values())
        + tuple(generation_checked.values())
        + (audit, shutdown_artifact, shutdown_marker),
    )


def _rollback_request(
    replayed: phase_capture.ReplayedPhaseCapture,
) -> tuple[dict[str, Any], tuple[common.EvidenceDescriptor, ...]]:
    generation = replayed.generation
    if generation is None:
        _fail(
            "rollback-generation-required",
            "fixed rollback phase must retain one non-stream generation exchange",
        )
    process = _descriptor_map(
        replayed.process_evidence,
        v3_checker.RAW_PROCESS_FIELDS,
        "rollback phase process evidence",
    )
    health = _descriptor_map(
        replayed.health,
        v3_checker.HTTP_EXCHANGE_FIELDS,
        "rollback phase health",
    )
    generation_checked = _descriptor_map(
        generation,
        v3_checker.HTTP_EXCHANGE_FIELDS,
        "rollback phase generation",
    )
    return (
        {
            "process_evidence": _path_map(
                process,
                v3_checker.RAW_PROCESS_FIELDS,
                "rollback phase process evidence",
            ),
            "health": _path_map(
                health,
                v3_checker.HTTP_EXCHANGE_FIELDS,
                "rollback phase health",
            ),
            "generation": _path_map(
                generation_checked,
                v3_checker.HTTP_EXCHANGE_FIELDS,
                "rollback phase generation",
            ),
        },
        tuple(process.values()) + tuple(health.values()) + tuple(generation_checked.values()),
    )


def _candidate_source_consumed_paths(
    replayed: candidate_source.ReplayedCandidateSourceJoin,
    candidate_descriptors: tuple[common.EvidenceDescriptor, ...],
) -> frozenset[str]:
    """Require the candidate/source replayer's complete closed path inventory."""

    paths = replayed.consumed_paths
    if not isinstance(paths, frozenset) or not paths:
        _fail(
            "invalid-candidate-source-consumed-paths",
            "candidate/source join must retain a nonempty immutable path inventory",
        )
    for path in paths:
        normalized = _common(
            lambda path=path: common.validate_relative_path(
                path,
                "candidate/source consumed evidence path",
            )
        )
        if normalized != path:
            _fail(
                "invalid-candidate-source-consumed-paths",
                "candidate/source consumed evidence paths must be canonical",
            )
    bindings = replayed.static_effective.static_bindings
    bridge = replayed.static_effective.config_bridge
    source = replayed.source_capture
    scenario = replayed.source_scenario
    expected = (
        bindings.reconstructed_baseline,
        bindings.freeze,
        bindings.base_release_candidate_report,
        bindings.configuration,
        bridge.endpoint,
        bridge.startup_artifact,
        bridge.endpoint_observation,
        source.session,
        source.contract,
        scenario.request_ledger,
        scenario.runtime_event_log,
    ) + candidate_descriptors
    for descriptor in expected:
        if not isinstance(descriptor, common.EvidenceDescriptor):
            _fail(
                "invalid-candidate-source-consumed-paths",
                "candidate/source path inventory has a non-descriptor direct input",
            )
        if descriptor.path not in paths:
            _fail(
                "candidate-source-inventory-incomplete",
                "candidate/source path inventory omits a directly consumed descriptor",
            )
    return paths


def _assert_no_cross_role_path_reuse(
    initial_paths: frozenset[str],
    groups: tuple[tuple[str, tuple[common.EvidenceDescriptor, ...]], ...],
) -> None:
    seen: set[str] = set(initial_paths)
    for label, descriptors in groups:
        for descriptor in descriptors:
            if descriptor.path in seen:
                _fail(
                    "duplicate-evidence-path",
                    f"{label} reuses root-relative evidence path {descriptor.path!r}",
                )
            seen.add(descriptor.path)


def _replay_inputs(
    root_fd: int,
    switch_fd: int,
) -> _ReplayedWriterInputs:
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "fixed rollback candidate-source writer evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    joined = _candidate_source(
        lambda: candidate_source._replay_candidate_source_join_on_held_root_fd(  # noqa: SLF001
            root_fd
        )
    )
    rollback = _phase(
        lambda: phase_capture.replay_rc3_rollback_phase_v1_fd(
            root_fd,
            ROLLBACK_PHASE_CAPTURE_NAME,
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
    replayed_transaction = _transaction(
        lambda: transaction.replay_atomic_transaction_on_held_switch_fd(
            root_fd,
            switch_fd,
        )
    )
    candidate_artifacts, rollback_artifacts, atomic_switch = _transaction_descriptor_maps(
        replayed_transaction
    )
    # This is the publication-bound static descriptor checkpoint.  The full
    # candidate-source replay above checks semantic config intent; this exact
    # comparison retains the four descriptors from that original session.
    _static_effective(
        lambda: static_effective._recheck_static_preparation_bindings_on_held_root_fd(  # noqa: SLF001
            root_fd,
            joined.static_effective.static_bindings,
        )
    )
    candidate_request, candidate_descriptors = _candidate_request(joined)
    rollback_request, rollback_descriptors = _rollback_request(rollback)
    candidate_source_paths = _candidate_source_consumed_paths(
        joined,
        candidate_descriptors,
    )
    _assert_no_cross_role_path_reuse(
        candidate_source_paths,
        (
            ("rollback raw evidence", rollback_descriptors),
            ("candidate artifacts", tuple(candidate_artifacts.values())),
            ("rollback artifacts", tuple(rollback_artifacts.values())),
            ("atomic switch evidence", tuple(atomic_switch.values())),
        )
    )
    # Keep the request derivation below visibly tied to the same typed replay.
    _ = candidate_request, rollback_request
    return _ReplayedWriterInputs(
        candidate_source=joined,
        rollback_phase=rollback,
        atomic_transaction=replayed_transaction,
        candidate_artifacts=candidate_artifacts,
        rollback_artifacts=rollback_artifacts,
        atomic_switch=atomic_switch,
    )


def _request_from_inputs(inputs: _ReplayedWriterInputs) -> dict[str, Any]:
    joined = inputs.candidate_source
    bindings = joined.static_effective.static_bindings
    candidate, _candidate_descriptors = _candidate_request(joined)
    rollback, _rollback_descriptors = _rollback_request(inputs.rollback_phase)
    return {
        "schema_version": v3_binder.BIND_REQUEST_VERSION,
        "candidate_id": joined.static_effective.candidate_id,
        "binding_evidence": {
            "freeze_path": bindings.freeze.path,
            "base_release_candidate_report_path": bindings.base_release_candidate_report.path,
            "configuration_path": bindings.configuration.path,
        },
        "reconstructed_baseline": {
            "manifest_path": bindings.reconstructed_baseline.path,
        },
        "candidate": candidate,
        "rollback": rollback,
        "candidate_artifacts": _path_map(
            inputs.candidate_artifacts,
            v3_checker.ARTIFACT_FIELDS,
            "transaction candidate artifact snapshots",
        ),
        "rollback_artifacts": _path_map(
            inputs.rollback_artifacts,
            v3_checker.ARTIFACT_FIELDS,
            "transaction rollback artifact snapshots",
        ),
        "atomic_switch": _path_map(
            inputs.atomic_switch,
            v3_checker.ATOMIC_SWITCH_FIELDS,
            "transaction atomic-switch descriptors",
        ),
    }


def _write_fixed_candidate_source_bind_request_on_held_root_switch_fds(
    root_fd: int,
    switch_fd: int,
) -> WrittenCandidateSourceBindRequest:
    """Write the sole fixed request while caller-owned root/switch EX stay held.

    The caller must retain the two exclusive locks and invoke a future v3/v4
    finalizer only from this function's normal-return stack.  This helper never
    opens, reopens, locks, unlocks, or closes either caller-owned descriptor.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "fixed rollback candidate-source writer evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    _assert_fixed_output_absent(root_fd)

    initial = _replay_inputs(root_fd, switch_fd)
    request = _request_from_inputs(initial)
    raw_request = _common(lambda: common.canonical_json_bytes(request))
    draft_descriptor = _common(
        lambda: common.descriptor_for_bytes(
            BIND_REQUEST_NAME,
            raw_request,
            "fixed rollback candidate-source bind request",
        )
    )

    # Rebuild every typed input immediately before publication.  This catches
    # same-EUID changes between the original static/source/phase/transaction
    # replay and create-only request publication.
    terminal = _replay_inputs(root_fd, switch_fd)
    if terminal != initial:
        _fail(
            "fixed-bind-request-replay-drift",
            "fixed rollback inputs changed during held-FD bind-request construction",
        )
    _assert_fixed_output_absent(root_fd)
    created = _common(
        lambda: common.write_create_only_json(
            root_fd,
            BIND_REQUEST_NAME,
            request,
            "fixed rollback candidate-source bind request",
        )
    )
    created_descriptor = created.descriptor(
        BIND_REQUEST_NAME,
        "fixed rollback candidate-source bind request",
    )
    if created_descriptor != draft_descriptor:
        _fail(
            "published-request-descriptor-mismatch",
            "published fixed bind request differs from its held-FD draft",
        )
    published = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            BIND_REQUEST_NAME,
            "published fixed rollback candidate-source bind request",
            maximum_bytes=v3_binder.MAX_BIND_REQUEST_BYTES,
        )
    )
    if published != raw_request:
        _fail(
            "post-publication-request-drift",
            "published fixed bind request changed during its self-check",
        )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "fixed rollback candidate-source writer evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    return WrittenCandidateSourceBindRequest(
        request=request,
        request_descriptor=draft_descriptor,
        candidate_source=terminal.candidate_source,
        rollback_phase=terminal.rollback_phase,
        atomic_transaction=terminal.atomic_transaction,
        static_bindings=terminal.candidate_source.static_effective.static_bindings,
    )
