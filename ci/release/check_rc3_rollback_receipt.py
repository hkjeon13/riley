#!/usr/bin/env python3
"""Fail-closed C02 verifier for an actual prior-artifact rollback drill.

The C02 rollback gate is deliberately distinct from Gate E's execution-mode
fallback soak scenario.  It binds a frozen candidate to its separately frozen
prior binary/bundle/image, replays Gate E, and validates a concrete ordered
drain -> atomic switch -> restart -> health/generation -> resource-zero
timeline.  The checker is CPU-only: it never starts Riley, CUDA, a container,
SSH session, or a network request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

import check_rc3_qualification as qualification


RECEIPT_VERSION = "riley.rc3-rollback-receipt.v1"
DRILL_VERSION = "riley.rc3-rollback-drill.v1"
CHECK_REPORT_VERSION = "riley.rc3-rollback-check.v1"
STABLE_DEFAULT_PROFILE = "stable-default"
MODEL_KIND = "smollm2"
MIN_OUTPUT_TOKENS = 4
MAX_TOKENS = 4096
MAX_TOKEN_ID = 2**32 - 1

INSTANCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
PROBE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")

EVENT_ORDER = (
    "candidate-ready",
    "candidate-drain-started",
    "candidate-drained",
    "atomic-switch",
    "rollback-ready",
    "rollback-generation",
    "candidate-resources-zero",
    "rollback-healthy",
)
CHECK_NAMES = (
    "freeze-binding",
    "gate-e-replay",
    "stable-default-arm-binding",
    "candidate-and-rollback-artifact-binding",
    "candidate-drain",
    "atomic-switch",
    "rollback-health-and-generation",
    "candidate-resource-zero",
    "worker-model-nonreuse",
)
ZERO_RESOURCE_FIELDS = (
    "active_requests",
    "pending_requests",
    "completion_outbox",
    "kv_promised_blocks",
    "kv_active_blocks",
    "riley_owned_live_allocations",
    "worker_processes",
)


class RollbackReceiptError(qualification.QualificationError):
    """A rollback receipt cannot establish the C02 semantic gate."""


class RollbackReceiptIncomparable(qualification.IncomparableError):
    """Rollback evidence belongs to another immutable candidate."""


@dataclass(frozen=True)
class Descriptor:
    path: str
    sha256: str


@dataclass(frozen=True)
class ArtifactSet:
    binary_sha256: str
    bundle_sha256: str
    image_id: str


@dataclass(frozen=True)
class RollbackReceipt:
    candidate_id: str
    bindings: dict[str, str]
    drill: Descriptor


@dataclass(frozen=True)
class RollbackCheckReport:
    candidate_id: str
    freeze_sha256: str
    base_release_candidate_report: Descriptor
    bindings: dict[str, str]
    receipt: Descriptor
    drill: Descriptor
    candidate_artifacts: ArtifactSet
    rollback_artifacts: ArtifactSet


def _raise(error_type: type[qualification.QualificationError], code: str, message: str) -> NoReturn:
    error = error_type(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _fail(code: str, message: str) -> NoReturn:
    _raise(RollbackReceiptError, code, message)


def _incomparable(message: str) -> NoReturn:
    _raise(RollbackReceiptIncomparable, "incomparable-binding", message)


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    return qualification._exact(value, fields, label)


def _sha256(value: Any, label: str) -> str:
    return qualification._sha256(value, label)


def _candidate_id(value: Any, label: str) -> str:
    candidate_id = qualification._string(value, label)
    if not qualification.release_candidate.CANDIDATE_ID_RE.fullmatch(candidate_id):
        _fail("invalid-candidate-id", f"{label} is not a valid RC candidate")
    return candidate_id


def _descriptor(value: Any, label: str) -> Descriptor:
    row = _exact(value, {"path", "sha256"}, label)
    return Descriptor(
        path=qualification._relative_path(row["path"], f"{label}.path"),
        sha256=_sha256(row["sha256"], f"{label}.sha256"),
    )


def _artifact_set(value: Any, label: str) -> ArtifactSet:
    row = _exact(value, {"binary_sha256", "bundle_sha256", "image_id"}, label)
    return ArtifactSet(
        binary_sha256=_sha256(row["binary_sha256"], f"{label}.binary_sha256"),
        bundle_sha256=_sha256(row["bundle_sha256"], f"{label}.bundle_sha256"),
        image_id=qualification._image(row["image_id"], f"{label}.image_id"),
    )


def _artifact_document(value: ArtifactSet) -> dict[str, str]:
    return {
        "binary_sha256": value.binary_sha256,
        "bundle_sha256": value.bundle_sha256,
        "image_id": value.image_id,
    }


def _bindings(value: Any, label: str) -> dict[str, str]:
    row = _exact(
        value,
        {
            "freeze_sha256",
            "base_release_candidate_report_sha256",
            "configuration_profile",
            "configuration_sha256",
        },
        label,
    )
    if row["configuration_profile"] != STABLE_DEFAULT_PROFILE:
        _incomparable(f"{label}.configuration_profile is not stable-default")
    return {
        "freeze_sha256": _sha256(row["freeze_sha256"], f"{label}.freeze_sha256"),
        "base_release_candidate_report_sha256": _sha256(
            row["base_release_candidate_report_sha256"],
            f"{label}.base_release_candidate_report_sha256",
        ),
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": _sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
    }


def _tokens(value: Any, label: str, *, minimum: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not (minimum <= len(value) <= MAX_TOKENS):
        _fail("invalid-token-sequence", f"{label} must contain {minimum}..{MAX_TOKENS} token IDs")
    result: list[int] = []
    for index, token in enumerate(value):
        if type(token) is not int or token < 0 or token > MAX_TOKEN_ID:
            _fail("invalid-token-id", f"{label}[{index}] must be a bounded non-negative integer")
        result.append(token)
    return tuple(result)


def _token_digest(tokens: Sequence[int]) -> str:
    return hashlib.sha256(qualification.canonical_json_bytes(list(tokens))).hexdigest()


def _instance_id(value: Any, label: str) -> str:
    value = qualification._string(value, label)
    if not INSTANCE_ID_RE.fullmatch(value):
        _fail("invalid-instance-id", f"{label} is not a normalized instance ID")
    return value


def _read_canonical_json(
    evidence_root: Path,
    descriptor: Descriptor,
    label: str,
    used_paths: set[str],
) -> tuple[bytes, dict[str, Any]]:
    if descriptor.path in used_paths:
        _fail("duplicate-evidence-path", f"{label} reuses another evidence path")
    used_paths.add(descriptor.path)
    raw = qualification._read_relative(evidence_root, descriptor.path, label)
    if hashlib.sha256(raw).hexdigest() != descriptor.sha256:
        _fail("evidence-hash-mismatch", f"{label} digest mismatch")
    document = qualification._parse_document(raw, label)
    if raw != qualification.canonical_json_bytes(document):
        _fail("noncanonical-evidence", f"{label} must be exact canonical JSON bytes")
    return raw, document


def validate_receipt(document: dict[str, Any]) -> RollbackReceipt:
    """Parse a non-authoritative descriptor for a rollback raw drill."""

    row = _exact(document, {"schema_version", "candidate_id", "bindings", "drill"}, "rollback receipt")
    if row["schema_version"] != RECEIPT_VERSION:
        _fail("unsupported-rollback-receipt-version", "rollback receipt schema_version is unsupported")
    return RollbackReceipt(
        candidate_id=_candidate_id(row["candidate_id"], "rollback receipt.candidate_id"),
        bindings=_bindings(row["bindings"], "rollback receipt.bindings"),
        drill=_descriptor(row["drill"], "rollback receipt.drill"),
    )


def _validate_bound_header(
    *,
    candidate_id: str,
    bindings: dict[str, str],
    frozen: qualification.FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
    label: str,
) -> None:
    if candidate_id != frozen.candidate_id:
        _incomparable(f"{label} belongs to another candidate")
    expected = {
        "freeze_sha256": freeze_sha256,
        "base_release_candidate_report_sha256": base_report_sha256,
        "configuration_profile": STABLE_DEFAULT_PROFILE,
        "configuration_sha256": frozen.arms["stable_default"]["configuration_sha256"],
    }
    if bindings != expected:
        _incomparable(f"{label} immutable bindings drifted from frozen stable-default")


def _zero_resources(value: dict[str, Any], label: str) -> None:
    for field in ZERO_RESOURCE_FIELDS:
        if type(value[field]) is not int or value[field] != 0:
            _fail("candidate-not-quiescent", f"{label}.{field} must be exactly zero")


def _generation_event(
    row: dict[str, Any],
    *,
    label: str,
    expected_tokens: tuple[int, ...],
    worker_id: str | None = None,
    model_instance_id: str | None = None,
    include_active_requests: bool,
) -> tuple[str, str]:
    fields = {
        "sequence",
        "event",
        "worker_id",
        "model_instance_id",
        "health_status",
        "output_token_ids",
        "output_token_ids_sha256",
        "finish_reason",
    }
    if include_active_requests:
        fields.add("active_requests")
    _exact(row, fields, label)
    actual_worker = _instance_id(row["worker_id"], f"{label}.worker_id")
    actual_model = _instance_id(row["model_instance_id"], f"{label}.model_instance_id")
    if worker_id is not None and actual_worker != worker_id:
        _fail("worker-instance-mismatch", f"{label} uses another worker instance")
    if model_instance_id is not None and actual_model != model_instance_id:
        _fail("model-instance-mismatch", f"{label} uses another model instance")
    if type(row["health_status"]) is not int or row["health_status"] != 200:
        _fail("rollback-health-failed", f"{label}.health_status must be 200")
    tokens = _tokens(row["output_token_ids"], f"{label}.output_token_ids", minimum=MIN_OUTPUT_TOKENS)
    if tokens != expected_tokens:
        _fail("rollback-token-mismatch", f"{label} tokens differ from the candidate probe")
    if _sha256(row["output_token_ids_sha256"], f"{label}.output_token_ids_sha256") != _token_digest(tokens):
        _fail("output-token-hash-mismatch", f"{label}.output_token_ids_sha256 mismatch")
    if row["finish_reason"] != "length":
        _fail("invalid-finish-reason", f"{label}.finish_reason must be length")
    if include_active_requests:
        if type(row["active_requests"]) is not int or row["active_requests"] < 1:
            _fail("missing-candidate-work", f"{label}.active_requests must prove a live candidate request")
    return actual_worker, actual_model


def _validate_drill(
    document: dict[str, Any],
    *,
    frozen: qualification.FrozenCandidate,
    freeze_sha256: str,
    base_report_sha256: str,
) -> tuple[ArtifactSet, ArtifactSet, dict[str, Any]]:
    row = _exact(
        document,
        {
            "schema_version",
            "candidate_id",
            "bindings",
            "model",
            "candidate_artifacts",
            "rollback_artifacts",
            "probe",
            "events",
        },
        "rollback drill",
    )
    if row["schema_version"] != DRILL_VERSION:
        _fail("unsupported-rollback-drill-version", "rollback drill schema_version is unsupported")
    candidate_id = _candidate_id(row["candidate_id"], "rollback drill.candidate_id")
    bindings = _bindings(row["bindings"], "rollback drill.bindings")
    _validate_bound_header(
        candidate_id=candidate_id,
        bindings=bindings,
        frozen=frozen,
        freeze_sha256=freeze_sha256,
        base_report_sha256=base_report_sha256,
        label="rollback drill",
    )
    if qualification._model(row["model"], "rollback drill.model") != frozen.models[MODEL_KIND]:
        _incomparable("rollback drill model identity drifted from frozen SmolLM2")
    candidate_artifacts = _artifact_set(row["candidate_artifacts"], "rollback drill.candidate_artifacts")
    rollback_artifacts = _artifact_set(row["rollback_artifacts"], "rollback drill.rollback_artifacts")
    expected_candidate = ArtifactSet(
        frozen.release["binary_sha256"], frozen.release["bundle_sha256"], frozen.release["image_id"]
    )
    expected_rollback = ArtifactSet(
        frozen.rollback["binary_sha256"], frozen.rollback["bundle_sha256"], frozen.rollback["image_id"]
    )
    if candidate_artifacts != expected_candidate:
        _incomparable("rollback drill candidate artifacts drifted from freeze.release")
    if rollback_artifacts != expected_rollback:
        _incomparable("rollback drill rollback artifacts drifted from freeze.rollback")
    if candidate_artifacts == rollback_artifacts:
        _fail("indistinguishable-rollback-artifacts", "rollback artifact must differ from candidate artifact")

    probe = _exact(
        row["probe"],
        {
            "probe_id",
            "prompt_token_ids",
            "prompt_token_ids_sha256",
            "expected_output_token_ids",
            "expected_output_token_ids_sha256",
            "sampling",
            "correctness_golden_sha256",
        },
        "rollback drill.probe",
    )
    probe_id = qualification._string(probe["probe_id"], "rollback drill.probe.probe_id")
    if not PROBE_ID_RE.fullmatch(probe_id):
        _fail("invalid-probe-id", "rollback drill.probe.probe_id is not normalized")
    prompt_tokens = _tokens(probe["prompt_token_ids"], "rollback drill.probe.prompt_token_ids", minimum=1)
    expected_tokens = _tokens(
        probe["expected_output_token_ids"], "rollback drill.probe.expected_output_token_ids", minimum=MIN_OUTPUT_TOKENS
    )
    if _sha256(probe["prompt_token_ids_sha256"], "rollback drill.probe.prompt_token_ids_sha256") != _token_digest(prompt_tokens):
        _fail("prompt-token-hash-mismatch", "rollback drill probe prompt token digest mismatch")
    if _sha256(probe["expected_output_token_ids_sha256"], "rollback drill.probe.expected_output_token_ids_sha256") != _token_digest(expected_tokens):
        _fail("expected-token-hash-mismatch", "rollback drill probe output token digest mismatch")
    sampling = _exact(probe["sampling"], {"mode", "temperature", "top_p"}, "rollback drill.probe.sampling")
    if sampling != {"mode": "greedy", "temperature": 0, "top_p": 1}:
        _fail("unsupported-sampling", "rollback drill probe must be exact greedy sampling")
    if _sha256(probe["correctness_golden_sha256"], "rollback drill.probe.correctness_golden_sha256") != frozen.source["correctness_golden_sha256"]:
        _incomparable("rollback drill does not bind frozen canonical correctness golden")

    events = row["events"]
    if not isinstance(events, list) or len(events) != len(EVENT_ORDER):
        _fail("invalid-rollback-event-inventory", "rollback drill must contain the closed event inventory")
    for index, item in enumerate(events):
        if not isinstance(item, dict):
            _fail("invalid-rollback-event", f"rollback drill.events[{index}] must be an object")
        if type(item.get("sequence")) is not int or item["sequence"] != index:
            _fail("rollback-event-order-mismatch", f"rollback drill.events[{index}] sequence is not contiguous")
        if item.get("event") != EVENT_ORDER[index]:
            _fail("rollback-event-order-mismatch", f"rollback drill.events[{index}] has the wrong event")

    candidate_worker, candidate_model = _generation_event(
        events[0],
        label="rollback drill.events[0]",
        expected_tokens=expected_tokens,
        include_active_requests=True,
    )
    _exact(
        events[1],
        {"sequence", "event", "worker_id", "model_instance_id", "active_requests"},
        "rollback drill.events[1]",
    )
    if _instance_id(events[1]["worker_id"], "rollback drill.events[1].worker_id") != candidate_worker or _instance_id(events[1]["model_instance_id"], "rollback drill.events[1].model_instance_id") != candidate_model:
        _fail("candidate-instance-mismatch", "candidate drain must target the live candidate instance")
    if type(events[1]["active_requests"]) is not int or events[1]["active_requests"] < 1:
        _fail("missing-candidate-work", "candidate drain must begin while a request is active")

    drained_fields = {"sequence", "event", "worker_id", "model_instance_id", *ZERO_RESOURCE_FIELDS}
    _exact(events[2], drained_fields, "rollback drill.events[2]")
    if _instance_id(events[2]["worker_id"], "rollback drill.events[2].worker_id") != candidate_worker or _instance_id(events[2]["model_instance_id"], "rollback drill.events[2].model_instance_id") != candidate_model:
        _fail("candidate-instance-mismatch", "candidate drain completion changed instance")
    _zero_resources(events[2], "rollback drill.events[2]")

    _exact(
        events[3],
        {"sequence", "event", "strategy", "from_artifacts", "to_artifacts"},
        "rollback drill.events[3]",
    )
    if events[3]["strategy"] != "atomic-rename":
        _fail("non-atomic-rollback-switch", "rollback drill must record an atomic-rename switch")
    if _artifact_set(events[3]["from_artifacts"], "rollback drill.events[3].from_artifacts") != candidate_artifacts or _artifact_set(events[3]["to_artifacts"], "rollback drill.events[3].to_artifacts") != rollback_artifacts:
        _incomparable("atomic rollback switch artifacts drifted from frozen identities")

    _exact(
        events[4],
        {"sequence", "event", "worker_id", "model_instance_id", "health_status"},
        "rollback drill.events[4]",
    )
    rollback_worker = _instance_id(events[4]["worker_id"], "rollback drill.events[4].worker_id")
    rollback_model = _instance_id(events[4]["model_instance_id"], "rollback drill.events[4].model_instance_id")
    if rollback_worker == candidate_worker or rollback_model == candidate_model:
        _fail("reused-candidate-instance", "rollback must not reuse the drained candidate worker or model")
    if type(events[4]["health_status"]) is not int or events[4]["health_status"] != 200:
        _fail("rollback-health-failed", "rollback worker did not become healthy")
    _generation_event(
        events[5],
        label="rollback drill.events[5]",
        expected_tokens=expected_tokens,
        worker_id=rollback_worker,
        model_instance_id=rollback_model,
        include_active_requests=False,
    )

    zero_fields = {
        "sequence",
        "event",
        "worker_id",
        "model_instance_id",
        "worker_present",
        "model_present",
        *ZERO_RESOURCE_FIELDS,
    }
    _exact(events[6], zero_fields, "rollback drill.events[6]")
    if _instance_id(events[6]["worker_id"], "rollback drill.events[6].worker_id") != candidate_worker or _instance_id(events[6]["model_instance_id"], "rollback drill.events[6].model_instance_id") != candidate_model:
        _fail("candidate-instance-mismatch", "resource-zero event changed candidate instance")
    if events[6]["worker_present"] is not False or events[6]["model_present"] is not False:
        _fail("candidate-resource-still-live", "candidate worker/model must be absent after rollback")
    _zero_resources(events[6], "rollback drill.events[6]")

    _exact(
        events[7],
        {"sequence", "event", "worker_id", "model_instance_id", "health_status", "active_requests"},
        "rollback drill.events[7]",
    )
    if _instance_id(events[7]["worker_id"], "rollback drill.events[7].worker_id") != rollback_worker or _instance_id(events[7]["model_instance_id"], "rollback drill.events[7].model_instance_id") != rollback_model:
        _fail("rollback-instance-mismatch", "final health belongs to another rollback instance")
    if type(events[7]["health_status"]) is not int or events[7]["health_status"] != 200 or type(events[7]["active_requests"]) is not int or events[7]["active_requests"] != 0:
        _fail("rollback-health-failed", "rollback final health is not quiescent HTTP 200")

    return candidate_artifacts, rollback_artifacts, {
        "probe_id": probe_id,
        "prompt_token_ids_sha256": _token_digest(prompt_tokens),
        "output_token_ids_sha256": _token_digest(expected_tokens),
        "completion_tokens": len(expected_tokens),
        "candidate_worker_id": candidate_worker,
        "candidate_model_instance_id": candidate_model,
        "rollback_worker_id": rollback_worker,
        "rollback_model_instance_id": rollback_model,
    }


def _empty_report() -> dict[str, Any]:
    return {
        "schema_version": CHECK_REPORT_VERSION,
        "status": "failed",
        "passed": False,
        "candidate_id": None,
        "freeze_sha256": None,
        "base_release_candidate_report": None,
        "bindings": None,
        "receipt": None,
        "drill": None,
        "candidate_artifacts": None,
        "rollback_artifacts": None,
        "probe": None,
        "checks": [],
        "reason_codes": [],
    }


def evaluate(
    freeze_path: Path,
    evidence_root: Path,
    receipt_path: Path | str,
    *,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    """Revalidate a C02 rollback drill without deploying an artifact."""

    report = _empty_report()
    try:
        expected_digest = _sha256(expected_freeze_sha256, "--expected-freeze-sha256")
        freeze_raw = qualification._read_regular_path(freeze_path, "freeze manifest")
        freeze_sha256 = hashlib.sha256(freeze_raw).hexdigest()
        report["freeze_sha256"] = freeze_sha256
        if freeze_sha256 != expected_digest:
            _fail("candidate-sha-mismatch", "freeze manifest SHA-256 differs from trusted input")
        frozen = qualification._validate_freeze(qualification._parse_document(freeze_raw, "freeze manifest"))
        report["candidate_id"] = frozen.candidate_id
        reserved_paths = {
            frozen.final_manifest.path,
            frozen.final_report.path,
            *(descriptor.path for descriptor in frozen.receipts.values()),
        }
        receipt_relative = qualification._relative_path(str(receipt_path), "rollback receipt.path")
        if receipt_relative in reserved_paths:
            _fail("reserved-output-path-collision", "raw rollback receipt must not reuse a freeze-declared output path")
        receipt_raw = qualification._read_relative(evidence_root, receipt_relative, "rollback receipt")
        receipt_document = qualification._parse_document(receipt_raw, "rollback receipt")
        if receipt_raw != qualification.canonical_json_bytes(receipt_document):
            _fail("noncanonical-evidence", "rollback receipt must be exact canonical JSON bytes")
        receipt_descriptor = Descriptor(receipt_relative, hashlib.sha256(receipt_raw).hexdigest())
        report["receipt"] = {"path": receipt_descriptor.path, "sha256": receipt_descriptor.sha256}
        receipt = validate_receipt(receipt_document)

        base_raw, base_report_sha256 = qualification.revalidate_base_release_candidate(
            frozen, freeze_sha256, evidence_root
        )
        report["base_release_candidate_report"] = {
            "path": frozen.final_report.path,
            "sha256": base_report_sha256,
        }
        _validate_bound_header(
            candidate_id=receipt.candidate_id,
            bindings=receipt.bindings,
            frozen=frozen,
            freeze_sha256=freeze_sha256,
            base_report_sha256=base_report_sha256,
            label="rollback receipt",
        )
        report["bindings"] = receipt.bindings
        if receipt.drill.path in reserved_paths:
            _fail("reserved-output-path-collision", "rollback drill must not reuse a freeze-declared output path")
        drill_raw, drill_document = _read_canonical_json(
            evidence_root,
            receipt.drill,
            "rollback drill",
            {receipt_descriptor.path},
        )
        report["drill"] = {"path": receipt.drill.path, "sha256": hashlib.sha256(drill_raw).hexdigest()}
        candidate_artifacts, rollback_artifacts, probe = _validate_drill(
            drill_document,
            frozen=frozen,
            freeze_sha256=freeze_sha256,
            base_report_sha256=base_report_sha256,
        )
        if hashlib.sha256(base_raw).hexdigest() != base_report_sha256:
            _fail("base-report-replay-digest-mismatch", "Gate E replay returned inconsistent bytes/digest")
        report.update(
            {
                "status": "passed",
                "passed": True,
                "candidate_artifacts": _artifact_document(candidate_artifacts),
                "rollback_artifacts": _artifact_document(rollback_artifacts),
                "probe": probe,
                "checks": [{"name": name, "passed": True} for name in CHECK_NAMES],
            }
        )
    except qualification.IncomparableError as error:
        report["status"] = "incomparable"
        report["reason_codes"] = [getattr(error, "reason_code", "incomparable-binding")]
    except qualification.GateFailure as error:
        report["reason_codes"] = [getattr(error, "reason_code", "gate-failed")]
    except (OSError, qualification.QualificationError) as error:
        report["reason_codes"] = [getattr(error, "reason_code", "invalid-input")]
    return report


def validate_check_report(document: dict[str, Any]) -> RollbackCheckReport:
    """Parse a passed report so the outer finalizer can replay it exactly."""

    row = _exact(
        document,
        {
            "schema_version",
            "status",
            "passed",
            "candidate_id",
            "freeze_sha256",
            "base_release_candidate_report",
            "bindings",
            "receipt",
            "drill",
            "candidate_artifacts",
            "rollback_artifacts",
            "probe",
            "checks",
            "reason_codes",
        },
        "rollback check report",
    )
    if row["schema_version"] != CHECK_REPORT_VERSION:
        _fail("unsupported-rollback-check-report-version", "rollback check report schema_version is unsupported")
    if row["status"] != "passed" or row["passed"] is not True or row["reason_codes"] != []:
        _fail("rollback-check-not-passed", "rollback check report must be a clean passed result")
    checks = row["checks"]
    if not isinstance(checks, list) or len(checks) != len(CHECK_NAMES):
        _fail("invalid-rollback-check-report", "rollback check report has an invalid check inventory")
    names: list[str] = []
    for index, check in enumerate(checks):
        item = _exact(check, {"name", "passed"}, f"rollback check report.checks[{index}]")
        if item["passed"] is not True:
            _fail("rollback-check-not-passed", f"rollback check {item['name']!r} did not pass")
        names.append(qualification._string(item["name"], f"rollback check report.checks[{index}].name"))
    if tuple(names) != CHECK_NAMES:
        _fail("invalid-rollback-check-report", "rollback check report check inventory drifted")
    candidate_artifacts = _artifact_set(row["candidate_artifacts"], "rollback check report.candidate_artifacts")
    rollback_artifacts = _artifact_set(row["rollback_artifacts"], "rollback check report.rollback_artifacts")
    probe = _exact(
        row["probe"],
        {
            "probe_id",
            "prompt_token_ids_sha256",
            "output_token_ids_sha256",
            "completion_tokens",
            "candidate_worker_id",
            "candidate_model_instance_id",
            "rollback_worker_id",
            "rollback_model_instance_id",
        },
        "rollback check report.probe",
    )
    probe_id = qualification._string(probe["probe_id"], "rollback check report.probe.probe_id")
    if not PROBE_ID_RE.fullmatch(probe_id):
        _fail("invalid-probe-id", "rollback check report probe ID is not normalized")
    _sha256(probe["prompt_token_ids_sha256"], "rollback check report.probe.prompt_token_ids_sha256")
    _sha256(probe["output_token_ids_sha256"], "rollback check report.probe.output_token_ids_sha256")
    if type(probe["completion_tokens"]) is not int or probe["completion_tokens"] < MIN_OUTPUT_TOKENS:
        _fail("invalid-rollback-check-report", "rollback check report completion token count is invalid")
    instance_ids = [
        _instance_id(probe[field], f"rollback check report.probe.{field}")
        for field in (
            "candidate_worker_id",
            "candidate_model_instance_id",
            "rollback_worker_id",
            "rollback_model_instance_id",
        )
    ]
    if instance_ids[0] == instance_ids[2] or instance_ids[1] == instance_ids[3]:
        _fail("reused-candidate-instance", "rollback report reuses a candidate worker or model")
    descriptors = (
        _descriptor(row["base_release_candidate_report"], "rollback check report.base_release_candidate_report"),
        _descriptor(row["receipt"], "rollback check report.receipt"),
        _descriptor(row["drill"], "rollback check report.drill"),
    )
    if len({descriptor.path for descriptor in descriptors}) != len(descriptors):
        _fail("duplicate-evidence-path", "rollback check report descriptors must be distinct")
    return RollbackCheckReport(
        candidate_id=_candidate_id(row["candidate_id"], "rollback check report.candidate_id"),
        freeze_sha256=_sha256(row["freeze_sha256"], "rollback check report.freeze_sha256"),
        base_release_candidate_report=descriptors[0],
        bindings=_bindings(row["bindings"], "rollback check report.bindings"),
        receipt=descriptors[1],
        drill=descriptors[2],
        candidate_artifacts=candidate_artifacts,
        rollback_artifacts=rollback_artifacts,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, help="relative raw rollback receipt path below evidence root")
    parser.add_argument("--report", type=Path, help="create-only semantic check-report output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = evaluate(
        arguments.freeze,
        arguments.evidence_root,
        arguments.receipt,
        expected_freeze_sha256=arguments.expected_freeze_sha256,
    )
    if arguments.report is not None:
        try:
            qualification._write_create_only(arguments.report, report)
        except qualification.QualificationError as error:
            print(str(error), file=sys.stderr)
            return 2
    print(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
