#!/usr/bin/env python3
"""Publish one private RC3 rollback finalizer normal-return receipt.

This is the immediately nested continuation of the fixed rollback v3/v4
finalizer, not a path-based receipt writer or an authenticated rollback
runner.  Its sole private entry accepts only the caller-owned private
evidence-root and rollback-switch descriptors while the caller retains both
exclusive locks.  It invokes the finalizer directly, retains that successful
normal-return closure in memory, and replays every fixed source/static/phase/
transaction input before it writes a fixed receipt and completion pair.

The receipt is raw finalizer-normal-return evidence only.  It does not prove a
host rollback, service lifecycle, GPU capture, candidate freeze, Gate E,
semantic receipt, or qualification result.  A visible receipt pair after a
post-link directory-sync failure is not a successful return and cannot be
reopened or resumed by another invocation.
"""

from __future__ import annotations

import os
import stat
from typing import Any, Callable, Mapping, NoReturn, TypeVar

import capture_rc3_rollback_atomic_transaction_v1 as transaction
import check_rc3_rollback_provenance_v3 as v3_checker
import check_rc3_rollback_provenance_v4 as v4_checker
import finalize_rc3_rollback_candidate_source_v4 as finalizer
import provenance_v2_common as common
import replay_rc3_rollback_candidate_source_v1 as candidate_source
import write_rc3_rollback_candidate_source_bind_request_v1 as writer


RECEIPT_NAME = "rollback-finalizer-receipt-v1.json"
RECEIPT_VERSION = "riley.rc3-rollback-finalizer-receipt.v1"
RECEIPT_COMPLETION_VERSION = "riley.rc3-rollback-finalizer-receipt-complete.v1"
RAW_FINALIZER_NORMAL_RETURN_AUTHORITY = "raw-finalizer-normal-return-only"
MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES


class RollbackFinalizerReceiptError(ValueError):
    """The fixed rollback finalizer cannot safely publish its raw receipt."""


def _fail(code: str, message: str) -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = RollbackFinalizerReceiptError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _finalizer(call: Callable[[], T]) -> T:
    try:
        return call()
    except finalizer.RollbackCandidateSourceFinalizerError as error:
        _fail(getattr(error, "reason_code", "invalid-rollback-finalizer"), str(error))


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


def _receipt_output_names() -> tuple[str, str, str]:
    return RECEIPT_NAME, f"{RECEIPT_NAME}.intent", f"{RECEIPT_NAME}.complete"


def _assert_receipt_names_are_distinct_from_finalizer() -> None:
    finalizer_names = set(finalizer._all_fixed_output_names())  # noqa: SLF001
    receipt_names = _receipt_output_names()
    if any(name in finalizer_names for name in receipt_names):
        _fail(
            "fixed-output-name-alias",
            "fixed rollback receipt names must differ from every finalizer output",
        )


def _assert_receipt_outputs_absent(root_fd: int) -> None:
    for name in _receipt_output_names():
        try:
            os.lstat(name, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail(
                "output-preflight-failed",
                f"cannot inspect fixed rollback finalizer receipt output {name!r}: {error}",
            )
        _fail(
            "output-name-collision",
            f"fixed rollback finalizer receipt output or reserved sibling {name!r} already exists",
        )


def _descriptor(value: Any, label: str) -> common.EvidenceDescriptor:
    encoded = value.as_json() if isinstance(value, common.EvidenceDescriptor) else value
    parsed = _common(lambda: common.parse_descriptor(encoded, label))
    if isinstance(value, common.EvidenceDescriptor) and parsed != value:
        _fail("invalid-evidence-descriptor", f"{label} cannot round-trip as one descriptor")
    return parsed


def _descriptor_json(value: Any, label: str) -> tuple[dict[str, Any], common.EvidenceDescriptor]:
    descriptor = _descriptor(value, label)
    return descriptor.as_json(), descriptor


def _descriptor_map_json(
    value: Any,
    fields: frozenset[str],
    label: str,
) -> tuple[dict[str, dict[str, Any]], tuple[common.EvidenceDescriptor, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(
            "unexpected-descriptor-map",
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
        )
    result: dict[str, dict[str, Any]] = {}
    descriptors: list[common.EvidenceDescriptor] = []
    for field in sorted(fields):
        document, descriptor = _descriptor_json(value[field], f"{label}.{field}")
        result[field] = document
        descriptors.append(descriptor)
    return result, tuple(descriptors)


def _target_json(value: Any, label: str) -> dict[str, Any]:
    try:
        document = value.as_json()
    except AttributeError as error:
        _fail("invalid-phase-target", f"{label} has no typed target encoding: {error}")
    if not isinstance(document, dict) or set(document) != {
        "server_pid",
        "server_start_ticks",
        "listener_port",
        "listener_inode",
        "gpu_index",
        "gpu_uuid",
    }:
        _fail("invalid-phase-target", f"{label} does not have the closed rollback target shape")
    return document


def _phase_document(
    value: Any,
    *,
    expected_capture_name: str,
    require_generation: bool,
    label: str,
) -> tuple[dict[str, Any], tuple[common.EvidenceDescriptor, ...]]:
    if getattr(value, "capture_name", None) != expected_capture_name:
        _fail(
            "rollback-phase-capture-name-mismatch",
            f"{label} must retain fixed capture name {expected_capture_name!r}",
        )
    process, process_descriptors = _descriptor_map_json(
        getattr(value, "process_evidence", None),
        v3_checker.RAW_PROCESS_FIELDS,
        f"{label}.process_evidence",
    )
    health, health_descriptors = _descriptor_map_json(
        getattr(value, "health", None),
        v3_checker.HTTP_EXCHANGE_FIELDS,
        f"{label}.health",
    )
    generation_value = getattr(value, "generation", None)
    if require_generation and generation_value is None:
        _fail("rollback-generation-required", f"{label} must retain one generation exchange")
    if not require_generation and generation_value is not None:
        _fail("candidate-local-generation-forbidden", f"{label} must not retain a local generation exchange")
    generation: dict[str, dict[str, Any]] | None
    generation_descriptors: tuple[common.EvidenceDescriptor, ...]
    if generation_value is None:
        generation = None
        generation_descriptors = ()
    else:
        generation, generation_descriptors = _descriptor_map_json(
            generation_value,
            v3_checker.HTTP_EXCHANGE_FIELDS,
            f"{label}.generation",
        )
    return (
        {
            "capture_name": expected_capture_name,
            "target": _target_json(getattr(value, "target", None), f"{label}.target"),
            "process_evidence": process,
            "health": health,
            "generation": generation,
        },
        process_descriptors + health_descriptors + generation_descriptors,
    )


def _source_capture_document(
    root_fd: int,
    joined: candidate_source.ReplayedCandidateSourceJoin,
) -> tuple[dict[str, Any], tuple[common.EvidenceDescriptor, ...]]:
    capture = joined.source_capture
    scenario = joined.source_scenario
    if getattr(capture, "scenarios", None) != (scenario,):
        _fail(
            "candidate-source-scenario-count",
            "candidate source closure must retain exactly its one replayed scenario",
        )
    session, session_descriptor = _descriptor_json(
        capture.session,
        "candidate source capture session",
    )
    contract, contract_descriptor = _descriptor_json(
        capture.contract,
        "candidate source capture contract",
    )
    fields = (
        ("request_ledger", scenario.request_ledger),
        ("request", scenario.request),
        ("response_head", scenario.response_head),
        ("response_body", scenario.response_body),
        ("runtime_event_log", scenario.runtime_event_log),
        ("generation_audit_index", scenario.generation_audit_index),
    )
    documents: dict[str, Any] = {"session": session, "contract": contract}
    descriptors: list[common.EvidenceDescriptor] = [session_descriptor, contract_descriptor]
    for name, value in fields:
        document, descriptor = _descriptor_json(value, f"candidate source scenario.{name}")
        documents[name] = document
        descriptors.append(descriptor)
    index_raw = _common(
        lambda: common.read_descriptor_bytes(
            root_fd,
            scenario.generation_audit_index,
            "candidate source generation audit index",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    index_document = _common(
        lambda: common.parse_canonical_json(
            index_raw,
            "candidate source generation audit index",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    if not isinstance(index_document, Mapping) or set(index_document) != {
        "schema_version",
        "scenario_id",
        "server_request_id",
        "audit_record",
        "audit_completion_marker",
    }:
        _fail(
            "invalid-source-audit-index",
            "candidate source generation audit index has an unsupported shape",
        )
    audit_record = _descriptor(
        index_document["audit_record"],
        "candidate source generation audit index.audit_record",
    )
    if audit_record != scenario.runtime_event_log:
        _fail(
            "source-audit-record-mismatch",
            "candidate source audit index does not bind its retained runtime event",
        )
    audit_marker = _descriptor(
        index_document["audit_completion_marker"],
        "candidate source generation audit index.audit_completion_marker",
    )
    _common(
        lambda: common.read_descriptor_bytes(
            root_fd,
            audit_marker,
            "candidate source audit completion marker",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    documents["audit_completion_marker"] = audit_marker.as_json()
    descriptors.append(audit_marker)
    scenario_id = getattr(scenario, "scenario_id", None)
    if type(scenario_id) is not str or not scenario_id:
        _fail("invalid-scenario-id", "candidate source scenario ID must be nonempty text")
    documents["scenario_id"] = scenario_id
    return documents, tuple(descriptors)


def _recheck_finalized_closure(
    root_fd: int,
    switch_fd: int,
    closure: finalizer._FinalizedRollbackCandidateSourceV4,  # noqa: SLF001
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Replay the exact closure retained across the finalizer return edge."""

    if not isinstance(closure, finalizer._FinalizedRollbackCandidateSourceV4):  # noqa: SLF001
        _fail("invalid-finalizer-result", "receipt requires the typed same-stack finalizer result")
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback finalizer receipt evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    _finalizer(lambda: finalizer._recheck_written_state(root_fd, switch_fd, closure.written))  # noqa: SLF001
    v3_descriptor, v3_report = _finalizer(
        lambda: finalizer._replay_v3_transaction(  # noqa: SLF001
            root_fd,
            switch_fd,
            closure.written,
            expected_v3_descriptor=closure.v3_descriptor,
        )
    )
    if v3_descriptor != closure.v3_descriptor or v3_report != closure.v3_report:
        _fail(
            "rollback-v3-replay-drift",
            "rollback v3 closure changed after the finalizer normal return",
        )
    v4_report = _v4(
        lambda: v4_checker.verify_completed_rollback_provenance_v4_on_held_switch_fd(
            root_fd,
            switch_fd,
            finalizer.ROLLBACK_V4_MANIFEST_NAME,
        )
    )
    if v4_report != closure.v4_report:
        _fail(
            "rollback-v4-replay-drift",
            "rollback v4 closure changed after the finalizer normal return",
        )
    reported_descriptor = _descriptor(
        v4_report.get("raw_manifest"),
        "completed rollback v4 report.raw_manifest",
    )
    if reported_descriptor != closure.v4_descriptor:
        _fail(
            "rollback-v4-descriptor-drift",
            "completed rollback v4 report no longer binds the finalizer output bytes",
        )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback finalizer receipt evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    return v3_report, v4_report


def _receipt_document_from_closure(
    root_fd: int,
    switch_fd: int,
    closure: finalizer._FinalizedRollbackCandidateSourceV4,  # noqa: SLF001
) -> tuple[dict[str, Any], tuple[common.EvidenceDescriptor, ...]]:
    v3_report, v4_report = _recheck_finalized_closure(root_fd, switch_fd, closure)
    written = closure.written
    joined = written.candidate_source
    static = written.static_bindings
    if (
        joined.static_effective.static_bindings != static
        or joined.static_effective.candidate_id != static.candidate_id
        or joined.static_effective.configuration_profile != static.configuration_profile
    ):
        _fail(
            "writer-static-identity-mismatch",
            "finalizer closure does not retain one static candidate identity",
        )
    expected_bindings = _finalizer(lambda: finalizer._expected_v3_bindings(written))  # noqa: SLF001
    if v3_report.get("candidate_id") != static.candidate_id or v4_report.get("candidate_id") != static.candidate_id:
        _fail("rollback-candidate-id-mismatch", "v3/v4 reports differ from the retained static candidate")
    if v3_report.get("bindings") != expected_bindings or v4_report.get("bindings") != expected_bindings:
        _fail("rollback-bindings-mismatch", "v3/v4 reports differ from retained static bindings")

    finalizer_outputs: dict[str, Any] = {}
    finalizer_descriptors: list[common.EvidenceDescriptor] = []
    for name, value in (
        ("v4_manifest", closure.v4_descriptor),
        ("v3_manifest", closure.v3_descriptor),
        ("bind_request", written.request_descriptor),
    ):
        document, descriptor = _descriptor_json(value, f"rollback finalizer output.{name}")
        finalizer_outputs[name] = document
        finalizer_descriptors.append(descriptor)

    static_document: dict[str, Any] = {}
    static_descriptors: list[common.EvidenceDescriptor] = []
    for name, value in (
        ("reconstructed_baseline", static.reconstructed_baseline),
        ("freeze", static.freeze),
        ("base_release_candidate_report", static.base_release_candidate_report),
        ("configuration", static.configuration),
    ):
        document, descriptor = _descriptor_json(value, f"static preparation.{name}")
        static_document[name] = document
        static_descriptors.append(descriptor)

    bridge = joined.static_effective.config_bridge
    bridge_document: dict[str, Any] = {}
    bridge_descriptors: list[common.EvidenceDescriptor] = []
    for name, value in (
        ("endpoint", bridge.endpoint),
        ("startup_artifact", bridge.startup_artifact),
        ("endpoint_observation", bridge.endpoint_observation),
    ):
        document, descriptor = _descriptor_json(value, f"candidate config bridge.{name}")
        bridge_document[name] = document
        bridge_descriptors.append(descriptor)
    candidate_phase, candidate_phase_descriptors = _phase_document(
        joined.candidate_phase,
        expected_capture_name=candidate_source.CANDIDATE_PHASE_CAPTURE_NAME,
        require_generation=False,
        label="candidate rollback phase",
    )
    source_capture, source_capture_descriptors = _source_capture_document(root_fd, joined)
    shutdown_artifact, shutdown_artifact_descriptor = _descriptor_json(
        joined.shutdown.artifact,
        "candidate source shutdown artifact",
    )
    shutdown_marker, shutdown_marker_descriptor = _descriptor_json(
        joined.shutdown.marker,
        "candidate source shutdown completion marker",
    )
    candidate_descriptors = tuple(
        static_descriptors
        + bridge_descriptors
        + list(candidate_phase_descriptors)
        + list(source_capture_descriptors)
        + [shutdown_artifact_descriptor, shutdown_marker_descriptor]
    )
    # The typed join retains a transitive inventory: config-bridge and serial
    # capture session descriptors themselves bind their closed raw children.
    # Keep both those parent descriptors and the complete path inventory, so a
    # later same-stack consumer cannot silently omit an indirect raw leaf.
    candidate_paths = {descriptor.path for descriptor in candidate_descriptors}
    consumed_paths = set(joined.consumed_paths)
    if not candidate_paths.issubset(consumed_paths):
        _fail(
            "candidate-source-inventory-projection-mismatch",
            "receipt projection omits one directly retained candidate/source descriptor",
        )

    rollback_phase, rollback_phase_descriptors = _phase_document(
        written.rollback_phase,
        expected_capture_name=writer.ROLLBACK_PHASE_CAPTURE_NAME,
        require_generation=True,
        label="rollback baseline phase",
    )
    atomic = written.atomic_transaction
    atomic_document: dict[str, Any] = {}
    atomic_descriptors: list[common.EvidenceDescriptor] = []
    for name, value in (
        ("session", atomic.session_descriptor),
        ("preparation_session", atomic.preparation_descriptor),
        ("atomic_switch_session", atomic.atomic_switch_descriptor),
    ):
        document, descriptor = _descriptor_json(value, f"atomic transaction.{name}")
        atomic_document[name] = document
        atomic_descriptors.append(descriptor)

    all_descriptors = tuple(finalizer_descriptors) + candidate_descriptors + rollback_phase_descriptors + tuple(atomic_descriptors)
    _common(lambda: common.require_unique_descriptors(all_descriptors, "rollback finalizer receipt closure"))
    output_names = set(_receipt_output_names())
    if any(descriptor.path in output_names for descriptor in all_descriptors):
        _fail(
            "output-name-collision",
            "fixed rollback receipt output collides with a replayed closure descriptor",
        )

    return (
        {
            "schema_version": RECEIPT_VERSION,
            "status": "completed",
            "qualification_status": "not-run",
            "authority": RAW_FINALIZER_NORMAL_RETURN_AUTHORITY,
            "candidate_id": static.candidate_id,
            "bindings": expected_bindings,
            "finalizer_outputs": finalizer_outputs,
            "static_preparation": static_document,
            "candidate_source": {
                "consumed_paths": sorted(consumed_paths),
                "config_bridge": bridge_document,
                "candidate_phase": candidate_phase,
                "source_capture": source_capture,
                "shutdown": {
                    "artifact": shutdown_artifact,
                    "completion_marker": shutdown_marker,
                },
            },
            "rollback_phase": rollback_phase,
            "atomic_transaction": atomic_document,
            "reason_codes": [],
        },
        all_descriptors,
    )


def _completion_pair_is_visible(root_fd: int) -> bool:
    try:
        final = os.lstat(f"{RECEIPT_NAME}.complete", dir_fd=root_fd)
        intent = os.lstat(f"{RECEIPT_NAME}.intent", dir_fd=root_fd)
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


def _finalize_and_write_rollback_receipt_on_held_root_switch_fds(
    root_fd: int,
    switch_fd: int,
) -> dict[str, Any]:
    """Run fixed v3/v4 finalization and write one same-stack raw receipt.

    The caller must already own both nonblocking exclusive locks.  There is no
    public path wrapper, CLI, retry, or resume mode. The final hardlink
    publication is deliberately the success path's last fallible operation:
    after it returns, this helper returns immediately. If that operation
    reports ambiguity, no new path/FD invocation may retry or treat its
    visible pair as normal-return authority.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback finalizer receipt evidence root",
        )
    )
    _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
    _assert_receipt_names_are_distinct_from_finalizer()
    # This preflight is intentionally before the finalizer: an occupied receipt
    # name must not leave a fresh v3/v4 closure that a later caller could try to
    # pair with another receipt publication.
    _assert_receipt_outputs_absent(root_fd)
    closure = _finalizer(
        lambda: finalizer._finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds(  # noqa: SLF001
            root_fd,
            switch_fd,
        )
    )
    document, _descriptors = _receipt_document_from_closure(root_fd, switch_fd, closure)
    raw_document = _common(lambda: common.canonical_json_bytes(document))
    draft_descriptor = _common(
        lambda: common.descriptor_for_bytes(
            RECEIPT_NAME,
            raw_document,
            "rollback finalizer receipt draft",
        )
    )
    _assert_receipt_outputs_absent(root_fd)
    # Rebuild the exact document once more before creating any receipt leaf.
    # This is the final closure/descriptor replay: a failure leaves no receipt
    # output at all, while the later terminal hardlink has no fallible
    # continuation that could turn a visible pair into a failed return.
    final_document, _ = _receipt_document_from_closure(root_fd, switch_fd, closure)
    if final_document != document:
        _fail(
            "receipt-closure-drift",
            "rollback finalizer receipt draft changed during its final held-FD replay",
        )
    created = _common(
        lambda: common.write_create_only_json(
            root_fd,
            RECEIPT_NAME,
            document,
            "rollback finalizer receipt",
        )
    )
    created_descriptor = created.descriptor(RECEIPT_NAME, "rollback finalizer receipt")
    if created_descriptor != draft_descriptor:
        _fail(
            "published-receipt-descriptor-mismatch",
            "create-only rollback receipt differs from its held-FD draft",
        )
    marker = {
        "schema_version": RECEIPT_COMPLETION_VERSION,
        "artifact_filename": RECEIPT_NAME,
        "artifact_sha256": created_descriptor.sha256,
    }
    _common(
        lambda: common.write_create_only_json(
            root_fd,
            f"{RECEIPT_NAME}.intent",
            marker,
            "rollback finalizer receipt completion marker intent",
        )
    )
    try:
        # `publish_create_only_hardlink` performs its own paired-link
        # validation and directory fsync. Nothing fallible may follow a
        # successful call: a post-link error is caught below as the sole
        # ambiguous terminal-publication case.
        _common(
            lambda: common.publish_create_only_hardlink(
                root_fd,
                f"{RECEIPT_NAME}.intent",
                f"{RECEIPT_NAME}.complete",
                "rollback finalizer receipt completion marker",
            )
        )
        return document
    except RollbackFinalizerReceiptError:
        if _completion_pair_is_visible(root_fd):
            _fail(
                "ambiguous-terminal-publication",
                "receipt marker became visible but final directory sync failed; "
                "no later invocation may treat it as finalizer success",
                )
        raise
