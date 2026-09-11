#!/usr/bin/env python3
"""Pure held-FD RC3 frozen-candidate input-identity derivation.

The first RC3 freeze boundary deliberately pins only the identity of the
already captured input closure.  It does not assert that an archive came from
the Git revision, that an ELF or image was built from it, that a model tree is
complete, or that correctness/Gate E/qualification has passed.  Those claims
need their own reviewed raw consumers before a later same-stack semantic
receipt may rely on them.

This module never opens caller-supplied root paths, takes locks, publishes
outputs, or directly writes.  It invokes held-FD leaf readers and the existing
trusted source-pre-freeze oracle; both the create-only writer and FD-safe
replayer call the same derivation while retaining their caller-owned
descriptors.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, TypeVar

import check_rc3_freeze_input_admission as freeze_inputs
import check_reconstructed_prior_baseline_v2 as baseline
import provenance_v2_common as common


MANIFEST_VERSION = "riley.rc3-frozen-candidate.v1"
REPLAY_VERSION = "riley.rc3-frozen-candidate-replay.v1"
MANIFEST_NAME = "frozen-candidate.json"
SCOPE = "frozen-candidate-input-identity-only"
MANIFEST_AUTHORITY = "frozen-candidate-input-identity-only"
REPLAY_AUTHORITY = "frozen-candidate-input-identity-replay-only"
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_FROZEN_INPUT_CLOSURE_BYTES = freeze_inputs.MAX_TOTAL_EXTERNAL_INPUT_BYTES
MAX_FROZEN_INPUT_CLOSURE_DESCRIPTORS = freeze_inputs.MAX_EXTERNAL_DESCRIPTORS

CHECK_NAMES = (
    "original-freeze-input-request-replayed",
    "declared-external-input-descriptors-rehashed",
    "source-prefreeze-and-live-source-lock-registry-cross-bound",
    "launch-input-self-reference-rejected",
    "reconstructed-prior-baseline-v2-replayed",
    "complete-frozen-input-closure-within-fixed-byte-budget",
    "candidate-and-baseline-raw-evidence-paths-disjoint",
    "manifest-is-input-identity-only",
)

NOT_ESTABLISHED = {
    "writer_normal_return": "not-established",
    "input_root_immutability": "not-established",
    "source_archive_content": "not-established",
    "release_binary_provenance": "not-established",
    "release_container_content": "not-established",
    "toolchain_probe_semantics": "not-established",
    "model_content": "not-established",
    "correctness_gate": "not-established",
    "gate_e": "not-established",
    "semantic_receipt": "not-established",
    "deployment": "not-established",
    "rollback_result": "not-established",
}

REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
CANDIDATE_ID_RE = re.compile(
    r"^riley-((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*))-rc([1-9][0-9]*)$"
)


class FrozenCandidateError(ValueError):
    """A frozen candidate cannot establish its narrow input identity."""


T = TypeVar("T")


def _fail(code: str, message: str) -> NoReturn:
    error = FrozenCandidateError(f"{code}: {message}")
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _freeze_inputs(call: Callable[[], T]) -> T:
    try:
        return call()
    except freeze_inputs.FreezeInputAdmissionError as error:
        _fail(getattr(error, "reason_code", "invalid-freeze-input"), str(error))


def _baseline(call: Callable[[], T]) -> T:
    try:
        return call()
    except baseline.BaselineError as error:
        _fail(getattr(error, "reason_code", "invalid-reconstructed-baseline"), str(error))
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "invalid-reconstructed-baseline"), str(error))


def normalized_absolute_path(value: Path, label: str) -> Path:
    """Validate one lexical absolute path without resolving or opening it."""

    try:
        raw = os.fspath(value)
    except TypeError as error:
        _fail("invalid-absolute-path", f"{label} is not a path: {error}")
    if (
        type(raw) is not str
        or not raw
        or "\x00" in raw
        or not os.path.isabs(raw)
        or raw.startswith("//")
        or os.path.normpath(raw) != raw
        or raw == os.path.sep
        or "\n" in raw
        or "\r" in raw
    ):
        _fail("invalid-absolute-path", f"{label} must be a normalized non-root absolute path")
    path = Path(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail("invalid-absolute-path", f"{label} must not contain traversal components")
    return path


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def require_disjoint_paths(paths: Mapping[str, Path]) -> None:
    rows = tuple(paths.items())
    for index, (left_label, left) in enumerate(rows):
        for right_label, right in rows[index + 1 :]:
            if paths_overlap(left, right):
                _fail(
                    "frozen-candidate-root-overlap",
                    f"{left_label} and {right_label} must be disjoint normalized paths",
                )


def require_distinct_root_fds(roots: Mapping[str, int]) -> None:
    seen: dict[tuple[int, int], str] = {}
    for label, directory_fd in roots.items():
        try:
            metadata = os.fstat(directory_fd)
        except OSError as error:
            _fail("unsafe-evidence-directory", f"cannot inspect {label}: {error}")
        identity = metadata.st_dev, metadata.st_ino
        previous = seen.get(identity)
        if previous is not None:
            _fail("input-root-alias", f"{label} aliases the already-held {previous}")
        seen[identity] = label


def _as_json(value: Any) -> Any:
    if isinstance(value, common.EvidenceDescriptor):
        return value.as_json()
    if type(value) is list:
        return [_as_json(item) for item in value]
    if type(value) is dict:
        return {key: _as_json(item) for key, item in value.items()}
    return value


def _source_pre_freeze_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    source_inputs = report.get("source_inputs")
    if type(source_inputs) is not dict or set(source_inputs) != {
        "workspace_manifests",
        "cargo_lock",
        "extension_registry",
        "server_defaults_source",
    }:
        _fail("invalid-source-prefreeze-report", "source pre-freeze report fields are unexpected")
    schema_version = report.get("schema_version")
    workspace_version = report.get("workspace_version")
    if type(schema_version) is not str or type(workspace_version) is not str:
        _fail("invalid-source-prefreeze-report", "source pre-freeze identity is malformed")
    return {
        "schema_version": schema_version,
        "source_inputs": source_inputs,
    }


def _bound_frozen_input_closure(
    request_descriptors: tuple[common.EvidenceDescriptor, ...],
    baseline_descriptors: tuple[common.EvidenceDescriptor, ...],
) -> None:
    """Bound the exact candidate plus baseline closure before raw streaming.

    The original request already contributes its baseline-manifest descriptor;
    the baseline callback contributes the exact 23 physical leaves beneath
    that manifest.  This makes duplicate paths and the single aggregate
    resource limit fail before baseline recipes or large artifacts are read.
    """

    closure = _common(
        lambda: common.require_unique_descriptors(
            (*request_descriptors, *baseline_descriptors),
            "frozen candidate input closure",
        )
    )
    if len(closure) > MAX_FROZEN_INPUT_CLOSURE_DESCRIPTORS:
        _fail(
            "too-many-external-descriptors",
            "frozen candidate input closure exceeds its descriptor budget",
        )
    total_bytes = sum(descriptor.byte_length for descriptor in closure)
    if total_bytes > MAX_FROZEN_INPUT_CLOSURE_BYTES:
        _fail(
            "external-evidence-byte-budget-exceeded",
            "frozen candidate input closure exceeds its total byte budget",
        )


def _baseline_identity(report: Mapping[str, Any]) -> dict[str, str]:
    if (
        report.get("schema_version") != baseline.CHECK_REPORT_VERSION
        or report.get("status") != "passed"
        or report.get("passed") is not True
        or report.get("provenance_class") != baseline.PROVENANCE_CLASS
        or report.get("historical_distribution") != baseline.HISTORICAL_DISTRIBUTION
    ):
        _fail("invalid-reconstructed-baseline", "baseline full replay did not return its exact v2 contract")
    baseline_id = report.get("baseline_id")
    git_identity = report.get("git_identity")
    if type(baseline_id) is not str or not baseline_id or type(git_identity) is not dict:
        _fail("invalid-reconstructed-baseline", "baseline replay identity is malformed")
    tag_name = git_identity.get("tag_name")
    target_commit_sha1 = git_identity.get("target_commit_sha1")
    if type(tag_name) is not str or type(target_commit_sha1) is not str or REVISION_RE.fullmatch(target_commit_sha1) is None:
        _fail("invalid-reconstructed-baseline", "baseline replay Git identity is malformed")
    return {
        "baseline_id": baseline_id,
        "tag_name": tag_name,
        "target_commit_sha1": target_commit_sha1,
        "relationship": "immediately-prior-rc-same-semver",
        "provenance_class": baseline.PROVENANCE_CLASS,
        "historical_distribution": baseline.HISTORICAL_DISTRIBUTION,
    }


def derive_frozen_candidate_manifest_on_held_fds(
    *,
    repository_root: Path,
    repository_root_fd: int,
    input_evidence_root_fd: int,
    expected_revision: str,
    candidate_id: str,
    request_name: str,
) -> dict[str, Any]:
    """Derive one exact manifest body by replaying original held-FD inputs.

    The caller owns every FD and must retain any locks for the full call.  No
    field is accepted from a prior admission report or ``freeze.raw``; this
    rereads the canonical request and all raw leaves itself.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            input_evidence_root_fd,
            "freeze-input evidence root",
        )
    )
    require_distinct_root_fds(
        {
            "source checkout": repository_root_fd,
            "freeze-input evidence root": input_evidence_root_fd,
        }
    )
    preflight = _freeze_inputs(
        lambda: freeze_inputs.prepare_rc3_freeze_input_request_on_held_root_fd(
            repository_root,
            repository_root_fd,
            expected_revision,
            candidate_id,
            input_evidence_root_fd,
            request_name,
        )
    )
    baseline_descriptor = preflight.request["rollback"]["reconstructed_baseline_manifest"]
    baseline_raw = _common(
        lambda: common.read_descriptor_bytes(
            input_evidence_root_fd,
            baseline_descriptor,
            "reconstructed prior baseline manifest",
            maximum_bytes=freeze_inputs.MAX_REQUEST_BYTES,
        )
    )
    baseline_document = _common(
        lambda: common.parse_canonical_json(
            baseline_raw,
            "reconstructed prior baseline manifest",
            maximum_bytes=freeze_inputs.MAX_REQUEST_BYTES,
        )
    )
    assert isinstance(baseline_document, dict)
    baseline_report = _baseline(
        lambda: baseline.evaluate(
            input_evidence_root_fd,
            baseline_document,
            descriptor_preflight=lambda baseline_descriptors: _bound_frozen_input_closure(
                (preflight.request_descriptor, *preflight.descriptors),
                baseline_descriptors,
            ),
        )
    )
    replay = _freeze_inputs(
        lambda: freeze_inputs.complete_rc3_freeze_input_request_on_held_root_fd(
            repository_root,
            repository_root_fd,
            expected_revision,
            candidate_id,
            input_evidence_root_fd,
            request_name,
            preflight,
        )
    )
    source_pre_freeze = _source_pre_freeze_projection(replay.source_prefreeze)
    workspace_version = replay.source_prefreeze.get("workspace_version")
    if type(workspace_version) is not str or not workspace_version:
        _fail("invalid-source-prefreeze-report", "source pre-freeze workspace version is malformed")
    return {
        "schema_version": MANIFEST_VERSION,
        "scope": SCOPE,
        "status": "frozen",
        "authority": MANIFEST_AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "source_revision": expected_revision,
        "workspace_version": workspace_version,
        "freeze_input_request": replay.request_descriptor.as_json(),
        "source_pre_freeze": source_pre_freeze,
        "bound_inputs": {
            "schema_version": freeze_inputs.REQUEST_VERSION,
            **_as_json(replay.request),
        },
        "reconstructed_baseline": _baseline_identity(baseline_report),
        "not_established": dict(NOT_ESTABLISHED),
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "reason_codes": [],
    }


def parse_frozen_candidate_manifest_identity(document: Any) -> tuple[str, str]:
    """Read only the fixed identity needed to derive an expected manifest."""

    expected_fields = {
        "schema_version",
        "scope",
        "status",
        "authority",
        "candidate_status",
        "qualification_status",
        "candidate_id",
        "source_revision",
        "workspace_version",
        "freeze_input_request",
        "source_pre_freeze",
        "bound_inputs",
        "reconstructed_baseline",
        "not_established",
        "checks",
        "reason_codes",
    }
    if type(document) is not dict or set(document) != expected_fields:
        _fail("invalid-frozen-candidate-manifest", "frozen candidate manifest has an unexpected field set")
    if (
        document["schema_version"] != MANIFEST_VERSION
        or document["scope"] != SCOPE
        or document["status"] != "frozen"
        or document["authority"] != MANIFEST_AUTHORITY
        or document["candidate_status"] != "frozen"
        or document["qualification_status"] != "not-run"
    ):
        _fail("invalid-frozen-candidate-manifest", "frozen candidate manifest has an invalid fixed status")
    candidate_id = document["candidate_id"]
    source_revision = document["source_revision"]
    if type(candidate_id) is not str or CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        _fail("invalid-frozen-candidate-manifest", "candidate_id is malformed")
    if (
        type(source_revision) is not str
        or REVISION_RE.fullmatch(source_revision) is None
        or source_revision == "0" * 40
    ):
        _fail("invalid-frozen-candidate-manifest", "source_revision is malformed")
    return candidate_id, source_revision


def replay_result(
    manifest_descriptor: common.EvidenceDescriptor,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the narrow non-qualification result of one held-FD replay."""

    return {
        "schema_version": REPLAY_VERSION,
        "scope": SCOPE,
        "status": "bound",
        "authority": REPLAY_AUTHORITY,
        "candidate_status": "frozen",
        "qualification_status": "not-run",
        "candidate_id": manifest["candidate_id"],
        "source_revision": manifest["source_revision"],
        "frozen_candidate_manifest": manifest_descriptor.as_json(),
        "checks": [{"name": name, "satisfied": True} for name in CHECK_NAMES],
        "reason_codes": [],
    }
