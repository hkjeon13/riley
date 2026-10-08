#!/usr/bin/env python3
"""Publish and replay one raw-only C02 lifecycle-supervisor receipt v1.

The writer is deliberately the *same-process* finalizer for a new v4 bind:
it first invokes the closed v4 binder, then replays that completed manifest
and the source-owned shutdown pair through one held private evidence-root FD.
Only that ordering can distinguish a successful v4 publication from the
binder's ``ambiguous-terminal-publication`` failure.  It never operates a
service, device, network endpoint, or qualification gate.

The resulting receipt records only raw byte descriptors and a derived target.
It says that the narrow lifecycle sequence completed, while deliberately
leaving ``qualification_status`` as ``not-run``.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

import bind_raw_c02_soak_v4 as v4_binder
import check_c02_provenance_v2 as checker
import provenance_v2_common as common


RECEIPT_VERSION = "riley.c02-lifecycle-supervisor-receipt.v1"
RECEIPT_COMPLETION_VERSION = "riley.c02-lifecycle-supervisor-receipt-complete.v1"
DEFAULT_SHUTDOWN_ARTIFACT_PATH = "source-audit/shutdown.json"
DEFAULT_SHUTDOWN_MARKER_PATH = "source-audit/shutdown.json.complete"
MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES
MAX_RECEIPT_NAME_BYTES = checker.MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES
RECEIPT_NAME_RE = checker.SOAK_TERMINAL_MANIFEST_NAME_RE


class LifecycleSupervisorReceiptError(ValueError):
    """The narrow raw lifecycle receipt cannot safely be published or replayed."""


def _fail(code: str, message: str) -> NoReturn:
    error = LifecycleSupervisorReceiptError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _checker(call: Any) -> Any:
    try:
        return call()
    except checker.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-raw-provenance"), str(error))


def _v4_binder(call: Any) -> Any:
    try:
        return call()
    except v4_binder.RawSoakBindError as error:
        # In particular, retain ambiguous-terminal-publication: the caller
        # must never continue to receipt publication after that binder result.
        _fail(getattr(error, "reason_code", "invalid-v4-publication"), str(error))


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail("unexpected-field-set", f"{label} must contain exactly {sorted(fields)}")
    return value


def _receipt_name(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or "/" in value
        or len(value) > MAX_RECEIPT_NAME_BYTES
        or RECEIPT_NAME_RE.fullmatch(value) is None
    ):
        _fail(
            "invalid-receipt-name",
            f"{label} must be a nonhidden root direct-child .json name of at most "
            f"{MAX_RECEIPT_NAME_BYTES} bytes",
        )
    return value


def _descriptor(value: Any, label: str) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if len(descriptor.path) > checker.MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label}.path exceeds {checker.MAX_RELATIVE_PATH_BYTES} bytes",
        )
    if descriptor.byte_length < 1:
        _fail("empty-evidence-leaf", f"{label} must bind nonempty raw evidence")
    return descriptor


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    """Reject the known source checkout before opening an evidence root."""

    source_root = Path(__file__).resolve().parents[2]
    try:
        evidence_root.relative_to(source_root)
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be external to the source checkout",
    )


def _read_canonical_object(
    root_fd: int,
    path: str,
    label: str,
    *,
    maximum_bytes: int = MAX_RECEIPT_BYTES,
) -> tuple[common.EvidenceDescriptor, dict[str, Any]]:
    relative = _common(lambda: common.validate_relative_path(path, f"{label}.path"))
    if len(relative) > checker.MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label}.path exceeds {checker.MAX_RELATIVE_PATH_BYTES} bytes",
        )
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            relative,
            label,
            maximum_bytes=maximum_bytes,
        )
    )
    if not raw:
        _fail("empty-evidence-leaf", f"{label} must be nonempty")
    document = _common(
        lambda: common.parse_canonical_json(raw, label, maximum_bytes=maximum_bytes)
    )
    assert isinstance(document, dict)
    return _common(lambda: common.descriptor_for_bytes(relative, raw, label)), document


def _target_from_report(report: dict[str, Any]) -> tuple[str, checker.TargetTuple]:
    """Take the one allowed target only from the replayed v4 report."""

    targets = report.get("targets")
    if not isinstance(targets, list) or len(targets) != 1:
        _fail(
            "lifecycle-scenario-count",
            "initial lifecycle receipt requires exactly one replayed v4 scenario",
        )
    row = _exact(targets[0], {"scenario_id", "target"}, "completed v4 report.targets[0]")
    scenario_id = row["scenario_id"]
    if (
        type(scenario_id) is not str
        or len(scenario_id) > 128
        or checker.SCENARIO_ID_RE.fullmatch(scenario_id) is None
        or scenario_id == "exact-backend-fallback"
    ):
        _fail("invalid-scenario-id", "completed v4 report has an unsupported scenario ID")
    target = _checker(
        lambda: checker._target(  # noqa: SLF001 - strict raw-provenance scalar parser
            row["target"],
            "completed v4 report.targets[0].target",
        )
    )
    return scenario_id, target


def _derive_receipt_document_fd(root_fd: int, v4_manifest_name: str) -> dict[str, Any]:
    """Replay all input evidence and build the sole allowed receipt document.

    The v4 verifier is intentionally called before any output exists.  Its
    report is not trusted as a caller-authored receipt field: the held-FD
    manifest bytes are reread through its derived descriptor to obtain the
    descriptor inventory recorded below.
    """

    manifest_name = _receipt_name(v4_manifest_name, "v4 manifest name")
    report = _checker(
        lambda: checker.verify_completed_soak_provenance_v4_fd(root_fd, manifest_name)
    )
    report_row = _exact(
        report,
        {
            "schema_version",
            "status",
            "qualification_status",
            "candidate_id",
            "bindings",
            "raw_manifest",
            "targets",
            "checks",
            "reason_codes",
        },
        "completed v4 report",
    )
    if (
        report_row["schema_version"] != checker.SOAK_V4_REPORT_VERSION
        or report_row["status"] != "bound"
        or report_row["qualification_status"] != "not-run"
        or report_row["reason_codes"] != []
    ):
        _fail("invalid-v4-report", "completed v4 replay did not return raw bound/not-run data")
    candidate_id = _checker(
        lambda: checker._candidate_id(  # noqa: SLF001 - strict raw-provenance scalar parser
            report_row["candidate_id"],
            "completed v4 report.candidate_id",
        )
    )
    bindings = report_row["bindings"]
    if not isinstance(bindings, dict):
        _fail("invalid-v4-report", "completed v4 report.bindings must be an object")
    raw_manifest = _descriptor(report_row["raw_manifest"], "completed v4 report.raw_manifest")
    if raw_manifest.path != manifest_name:
        _fail(
            "v4-report-manifest-mismatch",
            "completed v4 report does not bind the requested manifest name",
        )
    raw_manifest_bytes = _common(
        lambda: common.read_descriptor_bytes(
            root_fd,
            raw_manifest,
            "completed v4 manifest receipt extraction",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    manifest = _common(
        lambda: common.parse_canonical_json(
            raw_manifest_bytes,
            "completed v4 manifest receipt extraction",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    assert isinstance(manifest, dict)
    manifest_row = _exact(
        manifest,
        {
            "schema_version",
            "capture_status",
            "qualification_status",
            "candidate_id",
            "bindings",
            "configuration_evidence",
            "scenario_capture_session",
            "scenario_contract",
            "scenarios",
        },
        "completed v4 manifest receipt extraction",
    )
    if (
        manifest_row["schema_version"] != checker.SOAK_V4_MANIFEST_VERSION
        or manifest_row["capture_status"] != "captured"
        or manifest_row["qualification_status"] != "not-run"
        or manifest_row["candidate_id"] != candidate_id
        or manifest_row["bindings"] != bindings
    ):
        _fail("v4-report-manifest-mismatch", "completed v4 manifest differs from its replay")

    configuration_row = _exact(
        manifest_row["configuration_evidence"],
        {"endpoint", "startup_artifact", "endpoint_observation"},
        "completed v4 manifest.configuration_evidence",
    )
    configuration_evidence = {
        key: _descriptor(value, f"completed v4 manifest.configuration_evidence.{key}")
        for key, value in configuration_row.items()
    }
    capture_session = _descriptor(
        manifest_row["scenario_capture_session"],
        "completed v4 manifest.scenario_capture_session",
    )
    scenario_contract = _descriptor(
        manifest_row["scenario_contract"],
        "completed v4 manifest.scenario_contract",
    )
    scenarios = manifest_row["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        _fail(
            "lifecycle-scenario-count",
            "initial lifecycle receipt requires exactly one v4 manifest scenario",
        )
    scenario_row = _exact(
        scenarios[0],
        {
            "scenario_id",
            "target",
            "observation_session",
            "request_ledger",
            "runtime_event_log",
            "generation_audit_index",
        },
        "completed v4 manifest.scenarios[0]",
    )
    scenario_id, target = _target_from_report(report)
    if scenario_row["scenario_id"] != scenario_id:
        _fail(
            "v4-report-scenario-mismatch",
            "completed v4 manifest scenario differs from its replayed report",
        )
    scenario_target = _checker(
        lambda: checker._target(  # noqa: SLF001 - strict raw-provenance scalar parser
            scenario_row["target"],
            "completed v4 manifest.scenarios[0].target",
        )
    )
    if scenario_target != target:
        _fail(
            "v4-report-target-mismatch",
            "completed v4 manifest target differs from its replayed report",
        )
    observation_session = _descriptor(
        scenario_row["observation_session"],
        "completed v4 manifest.scenarios[0].observation_session",
    )
    scenario_evidence = {
        "scenario_id": scenario_id,
        "capture_session": capture_session,
        "contract": scenario_contract,
        "request_ledger": _descriptor(
            scenario_row["request_ledger"],
            "completed v4 manifest.scenarios[0].request_ledger",
        ),
        "runtime_event_log": _descriptor(
            scenario_row["runtime_event_log"],
            "completed v4 manifest.scenarios[0].runtime_event_log",
        ),
        "generation_audit_index": _descriptor(
            scenario_row["generation_audit_index"],
            "completed v4 manifest.scenarios[0].generation_audit_index",
        ),
    }

    # The shutdown checker takes precisely the v4-derived TargetTuple; neither
    # this writer nor its CLI accepts a PID, start tick, or GPU fact.
    shutdown = _checker(
        lambda: checker.verify_c02_shutdown_v2_fd(
            root_fd,
            DEFAULT_SHUTDOWN_ARTIFACT_PATH,
            DEFAULT_SHUTDOWN_MARKER_PATH,
            target,
        )
    )
    if shutdown.target != target:
        _fail("shutdown-target-mismatch", "shutdown replay returned another target tuple")

    descriptors = [
        raw_manifest,
        *configuration_evidence.values(),
        capture_session,
        scenario_contract,
        observation_session,
        scenario_evidence["request_ledger"],
        scenario_evidence["runtime_event_log"],
        scenario_evidence["generation_audit_index"],
        shutdown.artifact,
        shutdown.marker,
    ]
    _common(lambda: common.require_unique_descriptors(descriptors, "lifecycle receipt inputs"))

    return {
        "schema_version": RECEIPT_VERSION,
        "status": "completed",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "bindings": bindings,
        "target": target.as_json(),
        "raw_manifest": raw_manifest.as_json(),
        "configuration_evidence": {
            key: descriptor.as_json() for key, descriptor in configuration_evidence.items()
        },
        "scenario_evidence": {
            "scenario_id": scenario_id,
            "capture_session": capture_session.as_json(),
            "contract": scenario_contract.as_json(),
            "request_ledger": scenario_evidence["request_ledger"].as_json(),
            "runtime_event_log": scenario_evidence["runtime_event_log"].as_json(),
            "generation_audit_index": scenario_evidence[
                "generation_audit_index"
            ].as_json(),
        },
        "observation_evidence": {"session": observation_session.as_json()},
        "shutdown_evidence": {
            "artifact": shutdown.artifact.as_json(),
            "completion_marker": shutdown.marker.as_json(),
        },
        "reason_codes": [],
    }


def _receipt_input_descriptors(document: dict[str, Any]) -> tuple[common.EvidenceDescriptor, ...]:
    """Parse receipt descriptor slots before replaying the claimed manifest."""

    row = _exact(
        document,
        {
            "schema_version",
            "status",
            "qualification_status",
            "candidate_id",
            "bindings",
            "target",
            "raw_manifest",
            "configuration_evidence",
            "scenario_evidence",
            "observation_evidence",
            "shutdown_evidence",
            "reason_codes",
        },
        "lifecycle supervisor receipt",
    )
    if row["schema_version"] != RECEIPT_VERSION:
        _fail(
            "historical-lifecycle-receipt-version-rejected",
            f"lifecycle supervisor receipt must use {RECEIPT_VERSION}",
        )
    if row["status"] != "completed" or row["qualification_status"] != "not-run":
        _fail(
            "invalid-lifecycle-receipt-status",
            "lifecycle supervisor receipt must be completed/not-run",
        )
    if row["reason_codes"] != []:
        _fail("invalid-lifecycle-receipt-reasons", "lifecycle supervisor receipt must have no reasons")
    _checker(
        lambda: checker._candidate_id(  # noqa: SLF001 - strict raw-provenance scalar parser
            row["candidate_id"],
            "lifecycle supervisor receipt.candidate_id",
        )
    )
    _checker(
        lambda: checker._target(  # noqa: SLF001 - strict raw-provenance scalar parser
            row["target"],
            "lifecycle supervisor receipt.target",
        )
    )
    if not isinstance(row["bindings"], dict):
        _fail("invalid-lifecycle-receipt-bindings", "lifecycle supervisor receipt.bindings must be an object")
    configuration = _exact(
        row["configuration_evidence"],
        {"endpoint", "startup_artifact", "endpoint_observation"},
        "lifecycle supervisor receipt.configuration_evidence",
    )
    scenario = _exact(
        row["scenario_evidence"],
        {
            "scenario_id",
            "capture_session",
            "contract",
            "request_ledger",
            "runtime_event_log",
            "generation_audit_index",
        },
        "lifecycle supervisor receipt.scenario_evidence",
    )
    if (
        type(scenario["scenario_id"]) is not str
        or checker.SCENARIO_ID_RE.fullmatch(scenario["scenario_id"]) is None
        or scenario["scenario_id"] == "exact-backend-fallback"
    ):
        _fail("invalid-scenario-id", "lifecycle supervisor receipt has an unsupported scenario ID")
    observation = _exact(
        row["observation_evidence"],
        {"session"},
        "lifecycle supervisor receipt.observation_evidence",
    )
    shutdown = _exact(
        row["shutdown_evidence"],
        {"artifact", "completion_marker"},
        "lifecycle supervisor receipt.shutdown_evidence",
    )
    descriptors = (
        _descriptor(row["raw_manifest"], "lifecycle supervisor receipt.raw_manifest"),
        _descriptor(configuration["endpoint"], "lifecycle supervisor receipt.configuration_evidence.endpoint"),
        _descriptor(
            configuration["startup_artifact"],
            "lifecycle supervisor receipt.configuration_evidence.startup_artifact",
        ),
        _descriptor(
            configuration["endpoint_observation"],
            "lifecycle supervisor receipt.configuration_evidence.endpoint_observation",
        ),
        _descriptor(
            scenario["capture_session"],
            "lifecycle supervisor receipt.scenario_evidence.capture_session",
        ),
        _descriptor(scenario["contract"], "lifecycle supervisor receipt.scenario_evidence.contract"),
        _descriptor(
            scenario["request_ledger"],
            "lifecycle supervisor receipt.scenario_evidence.request_ledger",
        ),
        _descriptor(
            scenario["runtime_event_log"],
            "lifecycle supervisor receipt.scenario_evidence.runtime_event_log",
        ),
        _descriptor(
            scenario["generation_audit_index"],
            "lifecycle supervisor receipt.scenario_evidence.generation_audit_index",
        ),
        _descriptor(observation["session"], "lifecycle supervisor receipt.observation_evidence.session"),
        _descriptor(shutdown["artifact"], "lifecycle supervisor receipt.shutdown_evidence.artifact"),
        _descriptor(
            shutdown["completion_marker"],
            "lifecycle supervisor receipt.shutdown_evidence.completion_marker",
        ),
    )
    _common(lambda: common.require_unique_descriptors(descriptors, "lifecycle receipt inputs"))
    return descriptors


def _compare_receipt_to_replay(document: dict[str, Any], expected: dict[str, Any]) -> None:
    """Reject a receipt that copied only part of a valid raw replay."""

    for field, code in (
        ("candidate_id", "lifecycle-receipt-candidate-mismatch"),
        ("bindings", "lifecycle-receipt-bindings-mismatch"),
        ("target", "lifecycle-receipt-target-mismatch"),
        ("raw_manifest", "lifecycle-receipt-v4-manifest-mismatch"),
        ("configuration_evidence", "lifecycle-receipt-configuration-mismatch"),
        ("scenario_evidence", "lifecycle-receipt-scenario-mismatch"),
        ("observation_evidence", "lifecycle-receipt-observation-mismatch"),
        ("shutdown_evidence", "lifecycle-receipt-shutdown-mismatch"),
    ):
        if document[field] != expected[field]:
            _fail(code, f"lifecycle supervisor receipt {field} differs from held-FD replay")


def _verify_raw_receipt_fd(
    root_fd: int,
    receipt_name: str,
) -> tuple[common.EvidenceDescriptor, dict[str, Any]]:
    """Replay one unmarked create-only receipt before marker publication."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "lifecycle supervisor evidence root",
        )
    )
    name = _receipt_name(receipt_name, "lifecycle supervisor receipt name")
    descriptor, document = _read_canonical_object(
        root_fd,
        name,
        "lifecycle supervisor receipt",
    )
    descriptors = _receipt_input_descriptors(document)
    raw_manifest = descriptors[0]
    expected = _derive_receipt_document_fd(root_fd, raw_manifest.path)
    _compare_receipt_to_replay(document, expected)
    return descriptor, document


def _read_completion_marker(
    root_fd: int,
    receipt: common.EvidenceDescriptor,
) -> None:
    receipt_name = _receipt_name(receipt.path, "completed lifecycle receipt path")
    try:
        raw = _common(
            lambda: common.read_bounded_paired_hardlink(
                root_fd,
                f"{receipt_name}.complete",
                f"{receipt_name}.intent",
                "lifecycle supervisor receipt completion marker",
                maximum_bytes=MAX_RECEIPT_BYTES,
            )
        )
    except LifecycleSupervisorReceiptError as error:
        if getattr(error, "reason_code", None) == "missing-input":
            _fail(
                "missing-lifecycle-receipt-completion-marker",
                "lifecycle supervisor receipt requires exact sibling completion and intent markers",
            )
        raise
    marker = _common(
        lambda: common.parse_canonical_json(
            raw,
            "lifecycle supervisor receipt completion marker",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    assert isinstance(marker, dict)
    row = _exact(
        marker,
        {"schema_version", "artifact_filename", "artifact_sha256"},
        "lifecycle supervisor receipt completion marker",
    )
    if row["schema_version"] != RECEIPT_COMPLETION_VERSION:
        _fail(
            "historical-lifecycle-receipt-completion-version-rejected",
            "lifecycle supervisor receipt completion marker has an unsupported schema version",
        )
    if row["artifact_filename"] != receipt_name:
        _fail(
            "lifecycle-receipt-completion-marker-mismatch",
            "lifecycle supervisor receipt completion marker does not bind its artifact leaf",
        )
    digest = row["artifact_sha256"]
    if (
        type(digest) is not str
        or common.SHA256_RE.fullmatch(digest) is None
        or digest == "0" * 64
        or digest != receipt.sha256
    ):
        _fail(
            "lifecycle-receipt-completion-marker-mismatch",
            "lifecycle supervisor receipt completion marker does not bind exact receipt bytes",
        )


def verify_completed_lifecycle_supervisor_receipt_v1_fd(
    root_fd: int,
    receipt_name: str,
) -> dict[str, Any]:
    """Verify one terminal lifecycle receipt and replay every bound raw input."""

    name = _receipt_name(receipt_name, "completed lifecycle receipt name")
    receipt, document = _verify_raw_receipt_fd(root_fd, name)
    _read_completion_marker(root_fd, receipt)
    final_receipt, final_document = _read_canonical_object(
        root_fd,
        name,
        "completed lifecycle receipt final revalidation",
    )
    if final_receipt != receipt or final_document != document:
        _fail(
            "lifecycle-receipt-changed-during-completion-verification",
            "lifecycle supervisor receipt changed while its marker was verified",
        )
    replayed_receipt, replayed_document = _verify_raw_receipt_fd(root_fd, name)
    if replayed_receipt != final_receipt or replayed_document != final_document:
        _fail(
            "lifecycle-receipt-changed-during-completion-verification",
            "lifecycle supervisor receipt changed during final raw replay",
        )
    _read_completion_marker(root_fd, final_receipt)
    return final_document


def verify_completed_lifecycle_supervisor_receipt_v1(
    evidence_root: Path,
    receipt_name: str,
) -> dict[str, Any]:
    """Path wrapper for the completed-only lifecycle receipt verifier."""

    _assert_external_to_source_checkout(evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(
            evidence_root,
            "lifecycle supervisor evidence root",
        )
    )
    try:
        return verify_completed_lifecycle_supervisor_receipt_v1_fd(root_fd, receipt_name)
    finally:
        os.close(root_fd)


def verify_lifecycle_supervisor_receipt_v1_fd(
    root_fd: int,
    receipt_name: str,
) -> dict[str, Any]:
    """Completed-only public verifier alias for the v1 receipt contract."""

    return verify_completed_lifecycle_supervisor_receipt_v1_fd(root_fd, receipt_name)


def verify_lifecycle_supervisor_receipt_v1(
    evidence_root: Path,
    receipt_name: str,
) -> dict[str, Any]:
    """Completed-only path wrapper for the v1 receipt contract."""

    return verify_completed_lifecycle_supervisor_receipt_v1(evidence_root, receipt_name)


def _lock_terminal_output_pair(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail(
            "output-lock-unavailable",
            f"cannot acquire exclusive lifecycle receipt output lock: {error}",
        )


def _assert_terminal_output_pair_absent(root_fd: int, receipt_name: str) -> None:
    for name, label in (
        (receipt_name, "lifecycle supervisor receipt"),
        (f"{receipt_name}.complete", "lifecycle supervisor receipt completion marker"),
        (f"{receipt_name}.intent", "lifecycle supervisor receipt completion marker intent"),
    ):
        try:
            os.lstat(name, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail("output-preflight-failed", f"cannot inspect {label}: {error}")
        _fail("output-name-collision", f"{label} output {name!r} already exists")


def _assert_output_names_do_not_reuse_input_paths(
    receipt_name: str,
    document: dict[str, Any],
) -> None:
    descriptors = _receipt_input_descriptors(document)
    input_paths = {descriptor.path for descriptor in descriptors}
    for output_name in (receipt_name, f"{receipt_name}.complete", f"{receipt_name}.intent"):
        if output_name in input_paths:
            _fail(
                "output-name-collision",
                f"lifecycle receipt output {output_name!r} collides with replayed raw evidence",
            )


def _completion_marker_pair_is_visible(root_fd: int, receipt_name: str) -> bool:
    try:
        final = os.lstat(f"{receipt_name}.complete", dir_fd=root_fd)
        intent = os.lstat(f"{receipt_name}.intent", dir_fd=root_fd)
    except FileNotFoundError:
        return False
    except OSError as error:
        _fail("output-preflight-failed", f"cannot inspect receipt marker publication state: {error}")
    return (
        stat.S_ISREG(final.st_mode)
        and stat.S_ISREG(intent.st_mode)
        and stat.S_IMODE(final.st_mode) == 0o600
        and stat.S_IMODE(intent.st_mode) == 0o600
        and final.st_nlink == 2
        and intent.st_nlink == 2
        and (final.st_dev, final.st_ino) == (intent.st_dev, intent.st_ino)
    )


def _publish_receipt_after_successful_v4_fd(
    root_fd: int,
    v4_manifest_name: str,
    receipt_name: str,
) -> dict[str, Any]:
    """Publish a receipt only after this process has seen a successful v4 bind."""

    name = _receipt_name(receipt_name, "lifecycle supervisor receipt name")
    _lock_terminal_output_pair(root_fd)
    try:
        _assert_terminal_output_pair_absent(root_fd, name)
        # All v4 + shutdown replays occur before a receipt output exists.
        document = _derive_receipt_document_fd(root_fd, v4_manifest_name)
        _assert_output_names_do_not_reuse_input_paths(name, document)
        created = _common(
            lambda: common.write_create_only_json(
                root_fd,
                name,
                document,
                "lifecycle supervisor receipt",
            )
        )
        # Do not publish a terminal marker until an independent raw receipt
        # replay confirms its complete descriptor inventory.
        _verify_raw_receipt_fd(root_fd, name)
        marker = {
            "schema_version": RECEIPT_COMPLETION_VERSION,
            "artifact_filename": name,
            "artifact_sha256": created.sha256,
        }
        intent_name = f"{name}.intent"
        _common(
            lambda: common.write_create_only_json(
                root_fd,
                intent_name,
                marker,
                "lifecycle supervisor receipt completion marker intent",
            )
        )
        try:
            _common(
                lambda: common.publish_create_only_hardlink(
                    root_fd,
                    intent_name,
                    f"{name}.complete",
                    "lifecycle supervisor receipt completion marker",
                )
            )
        except LifecycleSupervisorReceiptError:
            if _completion_marker_pair_is_visible(root_fd, name):
                _fail(
                    "ambiguous-terminal-publication",
                    "receipt marker became visible but its final directory sync failed; "
                    "no lifecycle success receipt may be emitted",
                )
            raise
        return verify_completed_lifecycle_supervisor_receipt_v1_fd(root_fd, name)
    finally:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def publish_lifecycle_supervisor_receipt_v1_fd(
    root_fd: int,
    *,
    bind_request_path: str,
    v4_manifest_name: str,
    receipt_name: str,
) -> dict[str, Any]:
    """Bind v4 and atomically continue only on a successful v4 publication.

    The caller supplies paths, never descriptors, a target, a shutdown path,
    or a success flag.  If the v4 binder reports an ambiguous final marker,
    this function raises before receipt publication is attempted.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "lifecycle supervisor evidence root",
        )
    )
    manifest = _receipt_name(v4_manifest_name, "v4 manifest name")
    receipt = _receipt_name(receipt_name, "lifecycle supervisor receipt name")
    if manifest == receipt:
        _fail(
            "output-name-collision",
            "v4 manifest and lifecycle supervisor receipt names must differ",
        )
    # `bind_raw_soak_manifest_fd` is the only source of a successful-v4 edge
    # accepted here.  Its ambiguous result raises, so stale visible marker
    # pairs cannot be reinterpreted by a later standalone finalizer.
    _v4_binder(
        lambda: v4_binder.bind_raw_soak_manifest_fd(
            root_fd,
            bind_request_path,
            manifest,
        )
    )
    return _publish_receipt_after_successful_v4_fd(root_fd, manifest, receipt)


def publish_lifecycle_supervisor_receipt_v1(
    evidence_root: Path,
    *,
    bind_request_path: str,
    v4_manifest_name: str,
    receipt_name: str,
) -> dict[str, Any]:
    """Open one private root and bind+publish its narrow lifecycle receipt."""

    _assert_external_to_source_checkout(evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(
            evidence_root,
            "lifecycle supervisor evidence root",
        )
    )
    try:
        return publish_lifecycle_supervisor_receipt_v1_fd(
            root_fd,
            bind_request_path=bind_request_path,
            v4_manifest_name=v4_manifest_name,
            receipt_name=receipt_name,
        )
    finally:
        os.close(root_fd)


def write_lifecycle_supervisor_receipt_v1_fd(
    root_fd: int,
    *,
    bind_request_path: str,
    v4_manifest_name: str,
    receipt_name: str,
) -> dict[str, Any]:
    """Public writer alias; it preserves the same-process v4-success edge."""

    return publish_lifecycle_supervisor_receipt_v1_fd(
        root_fd,
        bind_request_path=bind_request_path,
        v4_manifest_name=v4_manifest_name,
        receipt_name=receipt_name,
    )


def write_lifecycle_supervisor_receipt_v1(
    evidence_root: Path,
    *,
    bind_request_path: str,
    v4_manifest_name: str,
    receipt_name: str,
) -> dict[str, Any]:
    """Path writer alias; it preserves the same-process v4-success edge."""

    return publish_lifecycle_supervisor_receipt_v1(
        evidence_root,
        bind_request_path=bind_request_path,
        v4_manifest_name=v4_manifest_name,
        receipt_name=receipt_name,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--bind-request", required=True)
    parser.add_argument("--v4-manifest-name", required=True)
    parser.add_argument("--receipt-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = publish_lifecycle_supervisor_receipt_v1(
            args.evidence_root,
            bind_request_path=args.bind_request,
            v4_manifest_name=args.v4_manifest_name,
            receipt_name=args.receipt_name,
        )
    except (LifecycleSupervisorReceiptError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(document) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
