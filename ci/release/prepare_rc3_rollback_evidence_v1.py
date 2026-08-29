#!/usr/bin/env python3
"""Prepare static RC3 rollback binding inputs without operating a server.

This producer accepts an already complete binary-bound reconstructed RC2 A/B
baseline evidence root. It replays that v2 baseline through one held private
root FD, requires the reviewed RC2 tag/target identity, and creates exactly
three immutable copies of future v3 binding inputs. It deliberately does not
materialize a baseline, start or stop a process, contact an endpoint, inspect
a GPU, execute an artifact, exchange a runtime file, create a freeze, or
qualify a candidate.  Its terminal session is only raw static preparation
evidence with ``qualification_status = not-run``.

The baseline root is not mutable input to this helper.  A caller must seed its
complete reconstructed closure by a separately reviewed mechanism before this
program runs.  Candidate/rollback phase capture, source-owned audit capture,
artifact preparation, atomic exchange, and the v3/v4 terminal join remain
separate later steps because their concrete evidence names are only known at
capture time.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import check_rc3_rollback_provenance_v3 as rollback
import check_reconstructed_prior_baseline_v2 as baseline
import provenance_v2_common as common


sys.dont_write_bytecode = True

SESSION_VERSION = "riley.rc3-rollback-evidence-preparation.v1"
INCOMPLETE_MARKER_VERSION = "riley.rc3-rollback-evidence-preparation-incomplete.v1"
INCOMPLETE_MARKER_NAME = "capture-incomplete.json"
COMPLETE_MARKER_VERSION = "riley.rc3-rollback-evidence-preparation-complete.v1"
COMPLETE_INTENT_NAME = "capture-complete.intent"
COMPLETE_MARKER_NAME = "capture-complete.json"
PREPARATION_DIRECTORY_NAME = "rollback-v3-evidence-preparation"
INPUTS_DIRECTORY_NAME = "rollback-v3-evidence-inputs"
FREEZE_NAME = "freeze.raw"
BASE_REPORT_NAME = "base-release-candidate-report.raw"
CONFIGURATION_NAME = "stable-default-configuration.raw"
STABLE_DEFAULT_PROFILE = "stable-default"
AUTHORITY = "raw-static-preparation-only"
MAX_INPUT_BYTES = common.DEFAULT_MAX_JSON_BYTES
MAX_RELATIVE_PATH_BYTES = 512

RC_IDENTIFIER_RE = re.compile(
    r"^(riley-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))-rc([1-9][0-9]*)$"
)


class RollbackEvidencePreparationError(ValueError):
    """Static rollback evidence cannot safely be prepared or replayed."""


def _fail(code: str, message: str) -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = RollbackEvidencePreparationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _baseline(call: Callable[[], T]) -> T:
    try:
        return call()
    except baseline.BaselineError as error:
        _fail(getattr(error, "reason_code", "invalid-reconstructed-baseline"), str(error))
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "invalid-reconstructed-baseline"), str(error))


@dataclass(frozen=True)
class EvidencePreparationRequest:
    """All caller-controlled paths for one create-only static preparation."""

    evidence_root: Path
    baseline_manifest_path: str
    candidate_id: str
    freeze_input: Path
    base_release_candidate_report_input: Path
    stable_default_configuration_input: Path


def _source_root() -> Path:
    """Factored so CPU-only tests can use an isolated mock checkout."""

    return Path(__file__).resolve().parents[2]


def _normalized_absolute_path(value: Path, label: str) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\x00" in raw or not os.path.isabs(raw):
        _fail("invalid-absolute-path", f"{label} must be a normalized absolute path")
    if raw.startswith("//") or os.path.normpath(raw) != raw or raw == os.path.sep:
        _fail("non-normalized-absolute-path", f"{label} must be a normalized non-root absolute path")
    path = Path(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail("non-normalized-absolute-path", f"{label} must not contain traversal components")
    return path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    source_root = _normalized_absolute_path(_source_root(), "source checkout")
    if _is_within(evidence_root, source_root):
        _fail(
            "evidence-root-inside-source-checkout",
            "--evidence-root must be outside the source checkout",
        )


def _assert_external_input(path: Path, evidence_root: Path, label: str) -> None:
    source_root = _normalized_absolute_path(_source_root(), "source checkout")
    if _is_within(path, source_root):
        _fail("input-inside-source-checkout", f"{label} must be outside the source checkout")
    if _is_within(path, evidence_root):
        _fail("input-inside-evidence-root", f"{label} must be outside --evidence-root")


def _baseline_manifest_path(value: str) -> str:
    path = _common(lambda: common.validate_relative_path(value, "--baseline-manifest-path"))
    if len(path) > MAX_RELATIVE_PATH_BYTES:
        _fail("invalid-relative-path", "--baseline-manifest-path exceeds its byte bound")
    if path.split("/", 1)[0] in {PREPARATION_DIRECTORY_NAME, INPUTS_DIRECTORY_NAME}:
        _fail("reserved-evidence-path", "--baseline-manifest-path may not be inside a reserved output child")
    return path


def _candidate_id(value: str) -> str:
    if not isinstance(value, str) or rollback.CANDIDATE_RE.fullmatch(value) is None:
        _fail("invalid-candidate-id", "--candidate-id must be a canonical Riley RC ID")
    candidate = RC_IDENTIFIER_RE.fullmatch(value)
    prior = RC_IDENTIFIER_RE.fullmatch(rollback.RECONSTRUCTED_ROLLBACK_TAG)
    if candidate is None or prior is None:  # pragma: no cover - reviewed constant invariant
        _fail("invalid-candidate-id", "candidate or reviewed baseline tag is malformed")
    if candidate.group(1) != prior.group(1) or int(candidate.group(2)) != int(prior.group(2)) + 1:
        _fail(
            "candidate-not-immediate-prior-baseline-successor",
            "--candidate-id must be the immediate same-version RC after the reviewed baseline tag",
        )
    return value


def _normalize_request(request: EvidencePreparationRequest) -> EvidencePreparationRequest:
    if not isinstance(request, EvidencePreparationRequest):
        _fail("invalid-request", "request must be an EvidencePreparationRequest")
    root = _normalized_absolute_path(request.evidence_root, "--evidence-root")
    _assert_external_to_source_checkout(root)
    baseline_manifest = _baseline_manifest_path(request.baseline_manifest_path)
    candidate = _candidate_id(request.candidate_id)
    freeze = _normalized_absolute_path(request.freeze_input, "--freeze-input")
    report = _normalized_absolute_path(
        request.base_release_candidate_report_input,
        "--base-release-candidate-report-input",
    )
    configuration = _normalized_absolute_path(
        request.stable_default_configuration_input,
        "--stable-default-configuration-input",
    )
    for path, label in (
        (freeze, "--freeze-input"),
        (report, "--base-release-candidate-report-input"),
        (configuration, "--stable-default-configuration-input"),
    ):
        _assert_external_input(path, root, label)
    values = (os.fspath(freeze), os.fspath(report), os.fspath(configuration))
    if len(set(values)) != len(values):
        _fail("duplicate-input-path", "the three static input paths must be distinct")
    return EvidencePreparationRequest(
        evidence_root=root,
        baseline_manifest_path=baseline_manifest,
        candidate_id=candidate,
        freeze_input=freeze,
        base_release_candidate_report_input=report,
        stable_default_configuration_input=configuration,
    )


def _lock_root(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("output-lock-unavailable", f"cannot acquire exclusive evidence-root lock: {error}")


def _unlock_quietly(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_UN)
    except OSError:
        pass


def _assert_root_children_absent(root_fd: int) -> None:
    for name in (PREPARATION_DIRECTORY_NAME, INPUTS_DIRECTORY_NAME):
        try:
            os.lstat(name, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail("output-preflight-failed", f"cannot inspect reserved root child {name!r}: {error}")
        _fail("create-only-collision", f"reserved root child {name!r} already exists")


def _parse_exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail("unexpected-field-set", f"{label} fields differ; expected={sorted(fields)}, actual={actual}")
    return value


def _assert_exact_entries(directory_fd: int, expected: set[str], label: str) -> None:
    try:
        actual = set(os.listdir(directory_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot enumerate {label}: {error}")
    if actual != expected:
        _fail(
            "unexpected-directory-contents",
            f"{label} entries differ; expected={sorted(expected)}, actual={sorted(actual)}",
        )


def _read_canonical_leaf(directory_fd: int, name: str, label: str) -> Mapping[str, Any]:
    document = _common(
        lambda: common.read_private_canonical_json_leaf(
            directory_fd,
            name,
            label,
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    assert isinstance(document, Mapping)
    return document


def _check_incomplete_marker(preparation_fd: int, *, require_terminal: bool) -> None:
    try:
        os.lstat(INCOMPLETE_MARKER_NAME, dir_fd=preparation_fd)
    except FileNotFoundError:
        if require_terminal:
            return
        _fail("missing-incomplete-marker", "preterminal preparation must retain its incomplete marker")
    except OSError as error:
        _fail("unsafe-evidence-path", f"cannot inspect preparation marker: {error}")
    if require_terminal:
        _fail("incomplete-capture", "static rollback evidence incomplete marker is still present")
    marker = _read_canonical_leaf(
        preparation_fd,
        INCOMPLETE_MARKER_NAME,
        "static rollback evidence incomplete marker",
    )
    _parse_exact(
        marker,
        {"schema_version", "capture_status", "qualification_status"},
        "static rollback evidence incomplete marker",
    )
    if (
        marker["schema_version"] != INCOMPLETE_MARKER_VERSION
        or marker["capture_status"] != "incomplete"
        or marker["qualification_status"] != "not-run"
    ):
        _fail("invalid-incomplete-marker", "static rollback evidence incomplete marker is not exact v1")


def _session_descriptor(preparation_fd: int) -> common.EvidenceDescriptor:
    return _common(
        lambda: common.describe_regular_relative(
            preparation_fd,
            "session.json",
            "static rollback evidence preparation session",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )


def _check_completion_marker(
    preparation_fd: int,
    session: common.EvidenceDescriptor,
    *,
    require_terminal: bool,
) -> None:
    if not require_terminal:
        for name in (COMPLETE_INTENT_NAME, COMPLETE_MARKER_NAME):
            try:
                os.lstat(name, dir_fd=preparation_fd)
            except FileNotFoundError:
                continue
            except OSError as error:
                _fail("unsafe-evidence-path", f"cannot inspect preparation completion marker: {error}")
            _fail("unexpected-terminal-marker", "preterminal preparation has a completion marker")
        return
    raw = _common(
        lambda: common.read_bounded_paired_hardlink(
            preparation_fd,
            COMPLETE_MARKER_NAME,
            COMPLETE_INTENT_NAME,
            "static rollback evidence completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    marker = _common(
        lambda: common.parse_canonical_json(
            raw,
            "static rollback evidence completion marker",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    assert isinstance(marker, Mapping)
    _parse_exact(
        marker,
        {"schema_version", "capture_status", "qualification_status", "session_sha256", "session_byte_length"},
        "static rollback evidence completion marker",
    )
    if (
        marker["schema_version"] != COMPLETE_MARKER_VERSION
        or marker["capture_status"] != "captured"
        or marker["qualification_status"] != "not-run"
        or type(marker["session_sha256"]) is not str
        or type(marker["session_byte_length"]) is not int
        or marker["session_sha256"] != session.sha256
        or marker["session_byte_length"] != session.byte_length
    ):
        _fail("invalid-completion-marker", "completion marker does not bind session.json")


def _read_and_replay_baseline(root_fd: int, manifest_path: str) -> dict[str, Any]:
    """Re-read the complete existing baseline without recording host paths."""

    before = _common(
        lambda: common.describe_regular_relative(
            root_fd,
            manifest_path,
            "reconstructed baseline manifest",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    if before.byte_length < 1:
        _fail("invalid-reconstructed-baseline", "reconstructed baseline manifest must be nonempty")
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            manifest_path,
            "reconstructed baseline manifest",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            raw,
            "reconstructed baseline manifest",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    if not isinstance(document, dict):  # pragma: no cover - common parser contract
        _fail("invalid-reconstructed-baseline", "reconstructed baseline manifest must be a JSON object")
    if document.get("schema_version") == baseline.LEGACY_MANIFEST_VERSION:
        _fail(
            "rollback-binary-provenance-required",
            "static rollback preparation requires a baseline v2 with A/B server-binary binding",
        )
    report = _baseline(lambda: baseline.evaluate(root_fd, document))
    after = _common(
        lambda: common.describe_regular_relative(
            root_fd,
            manifest_path,
            "reconstructed baseline manifest",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    if before != after:
        _fail("raced-input", "reconstructed baseline manifest changed during replay")
    if not isinstance(report, Mapping):  # pragma: no cover - checker contract
        _fail("invalid-reconstructed-baseline", "baseline checker did not return an object")
    identity = report.get("git_identity")
    if not isinstance(identity, Mapping) or (
        report.get("baseline_id") != rollback.RECONSTRUCTED_ROLLBACK_BASELINE_ID
        or identity.get("tag_name") != rollback.RECONSTRUCTED_ROLLBACK_TAG
        or identity.get("target_commit_sha1") != rollback.RECONSTRUCTED_ROLLBACK_TARGET
    ):
        _fail(
            "unsupported-reconstructed-baseline",
            "reconstructed baseline is not the reviewed RC2 tag/target identity",
        )
    return {
        "manifest": before.as_json(),
        "baseline_id": rollback.RECONSTRUCTED_ROLLBACK_BASELINE_ID,
        "tag_name": rollback.RECONSTRUCTED_ROLLBACK_TAG,
        "target_commit_sha1": rollback.RECONSTRUCTED_ROLLBACK_TARGET,
    }


def _snapshot_identity(created: common.CreatedEvidence) -> dict[str, int]:
    return {
        "device": created.device,
        "inode": created.inode,
        "mode": 0o600,
        "nlink": 1,
    }


def _snapshot_input(
    inputs_fd: int,
    *,
    source: Path,
    output_name: str,
    label: str,
) -> common.CreatedEvidence:
    return _common(
        lambda: common.snapshot_absolute_regular_create_only(
            source,
            inputs_fd,
            output_name,
            label,
            maximum_bytes=MAX_INPUT_BYTES,
            minimum_bytes=1,
            require_owner_executable=False,
        )
    )


def _verify_snapshot(
    inputs_fd: int,
    *,
    descriptor_value: Any,
    identity_value: Any,
    key: str,
    output_name: str,
    label: str,
) -> None:
    descriptor = _common(lambda: common.parse_descriptor(descriptor_value, f"{label} descriptor"))
    expected_path = f"{INPUTS_DIRECTORY_NAME}/{output_name}"
    if descriptor.path != expected_path or descriptor.byte_length < 1 or descriptor.byte_length > MAX_INPUT_BYTES:
        _fail("invalid-static-snapshot", f"{label} must use its fixed bounded nonempty snapshot path")
    held_descriptor = _common(
        lambda: common.rebase_descriptor_to_held_leaf(
            descriptor,
            expected_root_relative_path=expected_path,
            leaf_name=output_name,
            label=label,
        )
    )
    _common(
        lambda: common.verify_private_snapshot_descriptor_file(
            inputs_fd,
            held_descriptor,
            label,
            maximum_bytes=MAX_INPUT_BYTES,
        )
    )
    identity = _parse_exact(identity_value, {"device", "inode", "mode", "nlink"}, f"{key} snapshot identity")
    if (
        any(type(identity[name]) is not int or identity[name] < 0 for name in ("device", "inode", "mode", "nlink"))
        or identity["mode"] != 0o600
        or identity["nlink"] != 1
    ):
        _fail("invalid-snapshot-identity", f"{key} snapshot identity is malformed")
    try:
        visible = os.lstat(output_name, dir_fd=inputs_fd)
    except OSError as error:
        _fail("missing-input", f"cannot inspect {label}: {error}")
    if (
        not stat.S_ISREG(visible.st_mode)
        or visible.st_uid != os.geteuid()
        or stat.S_IMODE(visible.st_mode) != 0o600
        or visible.st_nlink != 1
        or visible.st_size != descriptor.byte_length
        or (visible.st_dev, visible.st_ino) != (identity["device"], identity["inode"])
    ):
        _fail("immutable-snapshot-mismatch", f"{label} lost its recorded private inode")


def _validate_session(
    root_fd: int,
    inputs_fd: int,
    session: Mapping[str, Any],
) -> None:
    row = _parse_exact(
        session,
        {
            "schema_version",
            "capture_status",
            "qualification_status",
            "authority",
            "candidate_id",
            "configuration_profile",
            "reconstructed_baseline",
            "binding_input_snapshots",
            "snapshot_identities",
        },
        "static rollback evidence preparation session",
    )
    if (
        row["schema_version"] != SESSION_VERSION
        or row["capture_status"] != "captured"
        or row["qualification_status"] != "not-run"
        or row["authority"] != AUTHORITY
        or row["configuration_profile"] != STABLE_DEFAULT_PROFILE
    ):
        _fail("invalid-session", "static rollback evidence session is not the exact raw v1 session")
    _candidate_id(row["candidate_id"])
    reconstructed = _parse_exact(
        row["reconstructed_baseline"],
        {"manifest", "baseline_id", "tag_name", "target_commit_sha1"},
        "session reconstructed baseline",
    )
    manifest = _common(lambda: common.parse_descriptor(reconstructed["manifest"], "session baseline manifest"))
    if (
        manifest.byte_length < 1
        or len(manifest.path) > MAX_RELATIVE_PATH_BYTES
        or manifest.path.split("/", 1)[0] in {PREPARATION_DIRECTORY_NAME, INPUTS_DIRECTORY_NAME}
    ):
        _fail("invalid-reconstructed-baseline", "session baseline manifest must be nonempty")
    replayed = _read_and_replay_baseline(root_fd, manifest.path)
    if dict(reconstructed) != replayed:
        _fail("baseline-session-mismatch", "session does not bind the replayed reviewed baseline")
    snapshots = _parse_exact(
        row["binding_input_snapshots"],
        {"freeze", "base_release_candidate_report", "configuration"},
        "static binding input snapshots",
    )
    identities = _parse_exact(
        row["snapshot_identities"],
        {"freeze", "base_release_candidate_report", "configuration"},
        "static binding input snapshot identities",
    )
    for key, output_name, label in (
        ("freeze", FREEZE_NAME, "freeze static snapshot"),
        ("base_release_candidate_report", BASE_REPORT_NAME, "base release candidate report static snapshot"),
        ("configuration", CONFIGURATION_NAME, "stable-default configuration static snapshot"),
    ):
        _verify_snapshot(
            inputs_fd,
            descriptor_value=snapshots[key],
            identity_value=identities[key],
            key=key,
            output_name=output_name,
            label=label,
        )


def replay_rollback_evidence_preparation_fd(
    root_fd: int,
    *,
    require_terminal: bool = True,
) -> dict[str, Any]:
    """Replay static preparation using a caller-held reconstructed root FD."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "static rollback evidence root",
        )
    )
    preparation_fd = _common(
        lambda: common.open_private_child_directory(
            root_fd,
            PREPARATION_DIRECTORY_NAME,
            "static rollback evidence preparation directory",
        )
    )
    inputs_fd: int | None = None
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                preparation_fd,
                PREPARATION_DIRECTORY_NAME,
                "held static rollback evidence preparation directory",
            )
        )
        inputs_fd = _common(
            lambda: common.open_private_child_directory(
                root_fd,
                INPUTS_DIRECTORY_NAME,
                "static rollback evidence inputs directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                inputs_fd,
                INPUTS_DIRECTORY_NAME,
                "held static rollback evidence inputs directory",
            )
        )
        _check_incomplete_marker(preparation_fd, require_terminal=require_terminal)
        _assert_exact_entries(
            preparation_fd,
            ({"session.json", COMPLETE_INTENT_NAME, COMPLETE_MARKER_NAME}
             if require_terminal else {"session.json", INCOMPLETE_MARKER_NAME}),
            "static rollback evidence preparation directory",
        )
        _assert_exact_entries(
            inputs_fd,
            {FREEZE_NAME, BASE_REPORT_NAME, CONFIGURATION_NAME},
            "static rollback evidence inputs directory",
        )
        initial_session_descriptor = _session_descriptor(preparation_fd)
        _raw, session = _common(
            lambda: common.read_private_descriptor_json_leaf(
                preparation_fd,
                initial_session_descriptor,
                "static rollback evidence preparation session",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        if not isinstance(session, Mapping):  # pragma: no cover - common parser contract
            _fail("invalid-session", "static rollback evidence session must be a JSON object")
        _check_completion_marker(preparation_fd, initial_session_descriptor, require_terminal=require_terminal)
        _validate_session(root_fd, inputs_fd, session)
        _check_incomplete_marker(preparation_fd, require_terminal=require_terminal)
        _check_completion_marker(preparation_fd, initial_session_descriptor, require_terminal=require_terminal)
        _assert_exact_entries(
            preparation_fd,
            ({"session.json", COMPLETE_INTENT_NAME, COMPLETE_MARKER_NAME}
             if require_terminal else {"session.json", INCOMPLETE_MARKER_NAME}),
            "static rollback evidence preparation directory",
        )
        _assert_exact_entries(
            inputs_fd,
            {FREEZE_NAME, BASE_REPORT_NAME, CONFIGURATION_NAME},
            "static rollback evidence inputs directory",
        )
        _raw, terminal_session = _common(
            lambda: common.read_private_descriptor_json_leaf(
                preparation_fd,
                initial_session_descriptor,
                "static rollback evidence preparation session",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        if terminal_session != session:
            _fail("raced-input", "static rollback evidence session changed during replay")
        _validate_session(root_fd, inputs_fd, terminal_session)
        _check_incomplete_marker(preparation_fd, require_terminal=require_terminal)
        _check_completion_marker(preparation_fd, initial_session_descriptor, require_terminal=require_terminal)
        _assert_exact_entries(
            preparation_fd,
            ({"session.json", COMPLETE_INTENT_NAME, COMPLETE_MARKER_NAME}
             if require_terminal else {"session.json", INCOMPLETE_MARKER_NAME}),
            "static rollback evidence preparation directory",
        )
        _assert_exact_entries(
            inputs_fd,
            {FREEZE_NAME, BASE_REPORT_NAME, CONFIGURATION_NAME},
            "static rollback evidence inputs directory",
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                inputs_fd,
                INPUTS_DIRECTORY_NAME,
                "held static rollback evidence inputs directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                preparation_fd,
                PREPARATION_DIRECTORY_NAME,
                "held static rollback evidence preparation directory",
            )
        )
        return dict(session)
    finally:
        if inputs_fd is not None:
            os.close(inputs_fd)
        os.close(preparation_fd)


def verify_rollback_evidence_preparation_fd(root_fd: int) -> dict[str, Any]:
    """Require the terminal paired marker and replay this raw preparation."""

    return replay_rollback_evidence_preparation_fd(root_fd, require_terminal=True)


def verify_rollback_evidence_preparation(evidence_root: Path) -> dict[str, Any]:
    """Open one existing external private root and replay terminal evidence."""

    root = _normalized_absolute_path(evidence_root, "--evidence-root")
    _assert_external_to_source_checkout(root)
    root_fd = _common(lambda: common.open_private_evidence_directory(root, "--evidence-root"))
    try:
        return verify_rollback_evidence_preparation_fd(root_fd)
    finally:
        os.close(root_fd)


def _restore_marker(preparation_fd: int, marker: Mapping[str, Any]) -> None:
    try:
        _common(
            lambda: common.write_create_only_json(
                preparation_fd,
                INCOMPLETE_MARKER_NAME,
                dict(marker),
                "restored static rollback evidence incomplete marker",
            )
        )
    except RollbackEvidencePreparationError:
        pass


def _publish_completion_marker(
    preparation_fd: int,
    marker: Mapping[str, Any],
    session: common.CreatedEvidence,
) -> None:
    """Publish only a durable two-name terminal marker pair."""

    try:
        os.unlink(INCOMPLETE_MARKER_NAME, dir_fd=preparation_fd)
    except OSError as error:
        _fail("marker-removal-failure", f"cannot remove static preparation incomplete marker: {error}")
    try:
        os.fsync(preparation_fd)
    except OSError as error:
        _restore_marker(preparation_fd, marker)
        _fail("durability-failure", f"cannot synchronize static preparation marker removal: {error}")
    completion = {
        "schema_version": COMPLETE_MARKER_VERSION,
        "capture_status": "captured",
        "qualification_status": "not-run",
        "session_sha256": session.sha256,
        "session_byte_length": session.byte_length,
    }
    _common(
        lambda: common.write_create_only_json(
            preparation_fd,
            COMPLETE_INTENT_NAME,
            completion,
            "static rollback evidence completion intent",
        )
    )
    _common(
        lambda: common.publish_create_only_hardlink(
            preparation_fd,
            COMPLETE_INTENT_NAME,
            COMPLETE_MARKER_NAME,
            "static rollback evidence completion marker",
        )
    )


def _prepare_on_held_root_fd(request: EvidencePreparationRequest, root_fd: int) -> dict[str, Any]:
    """Create one static preparation while retaining the root EX lock."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "held static rollback evidence root",
        )
    )
    preparation_fd: int | None = None
    inputs_fd: int | None = None
    try:
        _assert_root_children_absent(root_fd)
        baseline_record = _read_and_replay_baseline(root_fd, request.baseline_manifest_path)
        preparation_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                PREPARATION_DIRECTORY_NAME,
                "static rollback evidence preparation directory",
            )
        )
        incomplete = {
            "schema_version": INCOMPLETE_MARKER_VERSION,
            "capture_status": "incomplete",
            "qualification_status": "not-run",
        }
        _common(
            lambda: common.write_create_only_json(
                preparation_fd,
                INCOMPLETE_MARKER_NAME,
                incomplete,
                "static rollback evidence incomplete marker",
            )
        )
        inputs_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                INPUTS_DIRECTORY_NAME,
                "static rollback evidence inputs directory",
            )
        )
        freeze = _snapshot_input(
            inputs_fd,
            source=request.freeze_input,
            output_name=FREEZE_NAME,
            label="freeze static input",
        )
        base_report = _snapshot_input(
            inputs_fd,
            source=request.base_release_candidate_report_input,
            output_name=BASE_REPORT_NAME,
            label="base release candidate report static input",
        )
        configuration = _snapshot_input(
            inputs_fd,
            source=request.stable_default_configuration_input,
            output_name=CONFIGURATION_NAME,
            label="stable-default configuration static input",
        )
        snapshots = {
            "freeze": freeze.descriptor(f"{INPUTS_DIRECTORY_NAME}/{FREEZE_NAME}", "freeze static snapshot").as_json(),
            "base_release_candidate_report": base_report.descriptor(
                f"{INPUTS_DIRECTORY_NAME}/{BASE_REPORT_NAME}",
                "base release candidate report static snapshot",
            ).as_json(),
            "configuration": configuration.descriptor(
                f"{INPUTS_DIRECTORY_NAME}/{CONFIGURATION_NAME}",
                "stable-default configuration static snapshot",
            ).as_json(),
        }
        session = {
            "schema_version": SESSION_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "authority": AUTHORITY,
            "candidate_id": request.candidate_id,
            "configuration_profile": STABLE_DEFAULT_PROFILE,
            "reconstructed_baseline": baseline_record,
            "binding_input_snapshots": snapshots,
            "snapshot_identities": {
                "freeze": _snapshot_identity(freeze),
                "base_release_candidate_report": _snapshot_identity(base_report),
                "configuration": _snapshot_identity(configuration),
            },
        }
        session_created = _common(
            lambda: common.write_create_only_json(
                preparation_fd,
                "session.json",
                session,
                "static rollback evidence preparation session",
            )
        )
        preterminal = replay_rollback_evidence_preparation_fd(root_fd, require_terminal=False)
        if preterminal != session:
            _fail("prepublication-replay-drift", "held preparation replay differs from the draft session")
        _publish_completion_marker(preparation_fd, incomplete, session_created)
        terminal = verify_rollback_evidence_preparation_fd(root_fd)
        if terminal != session:
            _fail("post-publication-replay-drift", "held preparation replay differs from the published session")
        _common(
            lambda: common.require_private_evidence_directory_fd(
                root_fd,
                "held static rollback evidence root",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                inputs_fd,
                INPUTS_DIRECTORY_NAME,
                "held static rollback evidence inputs directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                preparation_fd,
                PREPARATION_DIRECTORY_NAME,
                "held static rollback evidence preparation directory",
            )
        )
        return terminal
    finally:
        if inputs_fd is not None:
            os.close(inputs_fd)
        if preparation_fd is not None:
            os.close(preparation_fd)


def prepare_rollback_evidence(request: EvidencePreparationRequest) -> dict[str, Any]:
    """Create three immutable static inputs below an admitted RC2 root."""

    normalized = _normalize_request(request)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(normalized.evidence_root, "--evidence-root")
    )
    try:
        _lock_root(root_fd)
        return _prepare_on_held_root_fd(normalized, root_fd)
    finally:
        _unlock_quietly(root_fd)
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--baseline-manifest-path", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--freeze-input", required=True, type=Path)
    parser.add_argument("--base-release-candidate-report-input", required=True, type=Path)
    parser.add_argument("--stable-default-configuration-input", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        session = prepare_rollback_evidence(
            EvidencePreparationRequest(
                evidence_root=args.evidence_root,
                baseline_manifest_path=args.baseline_manifest_path,
                candidate_id=args.candidate_id,
                freeze_input=args.freeze_input,
                base_release_candidate_report_input=args.base_release_candidate_report_input,
                stable_default_configuration_input=args.stable_default_configuration_input,
            )
        )
    except RollbackEvidencePreparationError as error:
        print(f"rollback evidence preparation: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(session) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
