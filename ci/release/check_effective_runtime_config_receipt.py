#!/usr/bin/env python3
"""Fail-closed C02 verifier for an effective runtime configuration receipt.

This checker deliberately does not start Riley, CUDA, a container, SSH, or a
network request.  A remote qualification producer captures the canonical
``GET /v1/config`` response, writes one create-only startup artifact, and
passes both files here together with the frozen candidate and passed Gate E
report.  The two JSON files are required to be byte-for-byte canonical so the
artifact's payload digest proves exactly what the endpoint returned.

The C02 outer checker continues to own the final RC3 decision.  Its
``startup_configuration`` receipt is this check report, but the outer checker
must rerun this verifier and exact-compare the result; a self-authored
``passed`` report is never authoritative evidence.
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


ENDPOINT_VERSION = "riley.effective-runtime-config.v1"
STARTUP_ARTIFACT_VERSION = "riley.effective-runtime-config-startup-artifact.v1"
CHECK_REPORT_VERSION = "riley.effective-runtime-config-check.v1"
STABLE_DEFAULT_PROFILE = "stable-default"
MAX_PERFORMANCE_EXACT_PROFILE = "max-performance-exact"
ARM_PROFILES = (STABLE_DEFAULT_PROFILE, MAX_PERFORMANCE_EXACT_PROFILE)
PROFILE_TO_FREEZE_ARM = {
    STABLE_DEFAULT_PROFILE: "stable_default",
    MAX_PERFORMANCE_EXACT_PROFILE: "max_performance_exact",
}

# These names are the C02 §4 dimensions.  ``effective_config`` has precisely
# this inventory: an endpoint cannot smuggle a hidden mode through a second
# top-level configuration object or omit one of the release-critical modes.
CONFIG_DIMENSIONS = (
    "execution_completion_mode",
    "batch_shape",
    "metadata_transport",
    "sampling_backend",
    "attention_backend",
    "gemm_reduction_policy",
    "experimental_flags",
    "fallback_policy",
    "batch_token_budget",
    "kv_geometry",
)

IMPLEMENTATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,255}$")
EXPERIMENTAL_FLAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ConfigReceiptError(qualification.QualificationError):
    """A configuration receipt is malformed or cannot prove the C02 gate."""


class ConfigReceiptIncomparable(qualification.IncomparableError):
    """A receipt is valid JSON but belongs to another immutable candidate."""


@dataclass(frozen=True)
class EndpointPayload:
    candidate_id: str
    runtime_identity: dict[str, str]
    effective_config: dict[str, Any]
    effective_config_sha256: str


@dataclass(frozen=True)
class StartupArtifact:
    candidate_id: str
    runtime_identity: dict[str, str]
    endpoint_payload_sha256: str
    endpoint_payload: EndpointPayload


@dataclass(frozen=True)
class StartupArtifactWriteResult:
    candidate_id: str
    configuration_profile: str
    endpoint_payload_sha256: str
    startup_artifact_sha256: str
    path: Path


@dataclass(frozen=True)
class ResolvedEvidence:
    path: str
    sha256: str


@dataclass(frozen=True)
class ArmEvidence:
    """One independently captured effective configuration arm."""

    configuration_profile: str
    configuration_sha256: str
    endpoint_payload: ResolvedEvidence
    startup_artifact: ResolvedEvidence
    effective_config_sha256: str


@dataclass(frozen=True)
class ConfigCheckReport:
    """The only data an outer startup-gate validator needs to rerun this check."""

    candidate_id: str
    freeze_sha256: str
    base_release_candidate_report: ResolvedEvidence
    stable_promotion_profile: str
    arms: dict[str, ArmEvidence]


def _raise(error_type: type[qualification.QualificationError], code: str, message: str) -> NoReturn:
    error = error_type(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _fail(code: str, message: str) -> NoReturn:
    _raise(ConfigReceiptError, code, message)


def _incomparable(message: str) -> NoReturn:
    _raise(ConfigReceiptIncomparable, "incomparable-binding", message)


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    return qualification._exact(value, fields, label)


def _sha256(value: Any, label: str) -> str:
    return qualification._sha256(value, label)


def _candidate_id(value: Any, label: str) -> str:
    candidate_id = qualification._string(value, label)
    if not qualification.release_candidate.CANDIDATE_ID_RE.fullmatch(candidate_id):
        _fail("invalid-candidate-id", f"{label} is not a valid RC candidate")
    return candidate_id


def _implementation_id(value: Any, label: str) -> str:
    value = qualification._string(value, label)
    if not IMPLEMENTATION_ID_RE.fullmatch(value):
        _fail("invalid-effective-config", f"{label} must be a lowercase implementation ID")
    return value


def _positive_integer(value: Any, label: str) -> int:
    # ``bool`` is an ``int`` subclass, but never an acceptable runtime bound.
    if type(value) is not int or value <= 0:
        _fail("invalid-effective-config", f"{label} must be a positive integer")
    return value


def _runtime_identity(value: Any, label: str) -> dict[str, str]:
    """Parse the startup-time identity which exists before Gate E exists.

    The raw endpoint and its create-only startup artifact are emitted by the
    release binary after cold prepare.  The replayed Gate E report does not
    yet exist, and neither that digest nor the frozen-manifest digest belongs
    in a raw startup fact.  Those cross-gate bindings intentionally live only
    in the post-capture semantic check report.
    """

    row = _exact(
        value,
        {
            "configuration_profile",
            "configuration_sha256",
        },
        label,
    )
    configuration_profile = qualification._string(
        row["configuration_profile"], f"{label}.configuration_profile"
    )
    if configuration_profile not in PROFILE_TO_FREEZE_ARM:
        _incomparable(f"{label}.configuration_profile is not a frozen C02 arm")
    return {
        "configuration_profile": configuration_profile,
        "configuration_sha256": _sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
    }


def _validate_effective_config(value: Any, label: str) -> dict[str, Any]:
    row = _exact(value, set(CONFIG_DIMENSIONS), label)

    completion = row["execution_completion_mode"]
    if completion not in {"per-operation", "iteration-batch"}:
        _fail("invalid-effective-config", f"{label}.execution_completion_mode is unsupported")

    batch_token_budget = _positive_integer(row["batch_token_budget"], f"{label}.batch_token_budget")
    batch_shape = _exact(row["batch_shape"], {"policy", "buckets"}, f"{label}.batch_shape")
    policy = batch_shape["policy"]
    if policy not in {"fixed-max", "power-of-two"}:
        _fail("invalid-effective-config", f"{label}.batch_shape.policy is unsupported")
    buckets_value = batch_shape["buckets"]
    if not isinstance(buckets_value, list) or not buckets_value or len(buckets_value) > 32:
        _fail("invalid-effective-config", f"{label}.batch_shape.buckets must be a non-empty bounded list")
    buckets = [
        _positive_integer(bucket, f"{label}.batch_shape.buckets[{index}]")
        for index, bucket in enumerate(buckets_value)
    ]
    if len(set(buckets)) != len(buckets) or any(left >= right for left, right in zip(buckets, buckets[1:])):
        _fail("invalid-effective-config", f"{label}.batch_shape.buckets must be strictly increasing")
    if buckets[-1] != batch_token_budget:
        _fail(
            "invalid-effective-config",
            f"{label}.batch_shape.buckets must end at batch_token_budget",
        )
    if policy == "fixed-max" and buckets != [batch_token_budget]:
        _fail(
            "invalid-effective-config",
            f"{label}.batch_shape.fixed-max must expose only its effective maximum bucket",
        )
    if policy == "power-of-two" and buckets[0] != 1:
        _fail(
            "invalid-effective-config",
            f"{label}.batch_shape.power-of-two must expose its 1-row bucket",
        )

    metadata_transport = row["metadata_transport"]
    if metadata_transport not in {"synchronous", "packed-async"}:
        _fail("invalid-effective-config", f"{label}.metadata_transport is unsupported")
    if metadata_transport == "packed-async" and completion != "iteration-batch":
        _fail(
            "invalid-effective-config",
            f"{label}.packed-async metadata requires iteration-batch completion",
        )

    sampling_backend = row["sampling_backend"]
    if sampling_backend not in {"cpu", "gpu-greedy"}:
        _fail("invalid-effective-config", f"{label}.sampling_backend is unsupported")

    attention = _exact(row["attention_backend"], {"prefill", "decode"}, f"{label}.attention_backend")
    attention_prefill = _implementation_id(attention["prefill"], f"{label}.attention_backend.prefill")
    attention_decode = _implementation_id(attention["decode"], f"{label}.attention_backend.decode")
    gemm_reduction_policy = _implementation_id(
        row["gemm_reduction_policy"], f"{label}.gemm_reduction_policy"
    )

    flags = row["experimental_flags"]
    if not isinstance(flags, dict):
        _fail("invalid-effective-config", f"{label}.experimental_flags must be a string map")
    normalized_flags: dict[str, str] = {}
    for flag_name, flag_value in flags.items():
        if (
            not isinstance(flag_name, str)
            or not EXPERIMENTAL_FLAG_RE.fullmatch(flag_name)
            or not isinstance(flag_value, str)
            or len(flag_value) > 1024
            or "\r" in flag_value
            or "\n" in flag_value
        ):
            _fail("invalid-effective-config", f"{label}.experimental_flags has an invalid entry")
        normalized_flags[flag_name] = flag_value

    fallback = _exact(
        row["fallback_policy"],
        {"cross_profile_fallback", "runtime_selection"},
        f"{label}.fallback_policy",
    )
    if fallback["cross_profile_fallback"] != "forbidden":
        _fail(
            "invalid-effective-config",
            f"{label}.fallback_policy.cross_profile_fallback must be forbidden",
        )
    if fallback["runtime_selection"] not in {"exact-fallback-allowed", "fail-closed"}:
        _fail("invalid-effective-config", f"{label}.fallback_policy.runtime_selection is unsupported")

    kv_geometry = _exact(
        row["kv_geometry"],
        {"layout", "block_tokens", "physical_blocks"},
        f"{label}.kv_geometry",
    )
    if kv_geometry["layout"] not in {"contiguous", "paged"}:
        _fail("invalid-effective-config", f"{label}.kv_geometry.layout is unsupported")
    block_tokens = _positive_integer(kv_geometry["block_tokens"], f"{label}.kv_geometry.block_tokens")
    physical_blocks = _positive_integer(
        kv_geometry["physical_blocks"], f"{label}.kv_geometry.physical_blocks"
    )

    return {
        "execution_completion_mode": completion,
        "batch_shape": {"policy": policy, "buckets": buckets},
        "metadata_transport": metadata_transport,
        "sampling_backend": sampling_backend,
        "attention_backend": {"prefill": attention_prefill, "decode": attention_decode},
        "gemm_reduction_policy": gemm_reduction_policy,
        "experimental_flags": normalized_flags,
        "fallback_policy": {
            "cross_profile_fallback": "forbidden",
            "runtime_selection": fallback["runtime_selection"],
        },
        "batch_token_budget": batch_token_budget,
        "kv_geometry": {
            "layout": kv_geometry["layout"],
            "block_tokens": block_tokens,
            "physical_blocks": physical_blocks,
        },
    }


def validate_endpoint_payload(document: dict[str, Any], label: str = "endpoint payload") -> EndpointPayload:
    """Validate one decoded /v1/config response without a candidate checkout."""

    row = _exact(
        document,
        {
            "schema_version",
            "candidate_id",
            "runtime_identity",
            "effective_config",
            "effective_config_sha256",
        },
        label,
    )
    if row["schema_version"] != ENDPOINT_VERSION:
        _fail("unsupported-endpoint-version", f"{label}.schema_version is unsupported")
    effective_config = _validate_effective_config(row["effective_config"], f"{label}.effective_config")
    expected_effective_config_sha256 = hashlib.sha256(
        qualification.canonical_json_bytes(effective_config)
    ).hexdigest()
    effective_config_sha256 = _sha256(
        row["effective_config_sha256"], f"{label}.effective_config_sha256"
    )
    if effective_config_sha256 != expected_effective_config_sha256:
        _fail(
            "effective-config-hash-mismatch",
            f"{label}.effective_config_sha256 does not hash canonical effective_config",
        )
    return EndpointPayload(
        candidate_id=_candidate_id(row["candidate_id"], f"{label}.candidate_id"),
        runtime_identity=_runtime_identity(
            row["runtime_identity"], f"{label}.runtime_identity"
        ),
        effective_config=effective_config,
        effective_config_sha256=effective_config_sha256,
    )


def _validate_endpoint_bytes(raw: bytes, label: str) -> tuple[dict[str, Any], EndpointPayload]:
    document = qualification._parse_document(raw, label)
    canonical = qualification.canonical_json_bytes(document)
    if raw != canonical:
        _fail(
            "noncanonical-endpoint-payload",
            f"{label} must be exact canonical JSON bytes for startup digest agreement",
        )
    return document, validate_endpoint_payload(document, label)


def validate_startup_artifact(
    document: dict[str, Any], label: str = "startup artifact"
) -> StartupArtifact:
    """Validate the decoded create-only startup artifact's closed shape."""

    row = _exact(
        document,
        {
            "schema_version",
            "created_at_utc",
            "candidate_id",
            "endpoint_path",
            "runtime_identity",
            "endpoint_payload_sha256",
            "endpoint_payload",
        },
        label,
    )
    if row["schema_version"] != STARTUP_ARTIFACT_VERSION:
        _fail("unsupported-startup-artifact-version", f"{label}.schema_version is unsupported")
    if not isinstance(row["created_at_utc"], str) or not qualification.UTC_RE.fullmatch(row["created_at_utc"]):
        _fail("invalid-created-at", f"{label}.created_at_utc must be UTC second precision")
    if row["endpoint_path"] != "/v1/config":
        _fail("wrong-config-endpoint", f"{label}.endpoint_path must be /v1/config")
    endpoint = validate_endpoint_payload(row["endpoint_payload"], f"{label}.endpoint_payload")
    candidate_id = _candidate_id(row["candidate_id"], f"{label}.candidate_id")
    runtime_identity = _runtime_identity(
        row["runtime_identity"], f"{label}.runtime_identity"
    )
    if candidate_id != endpoint.candidate_id:
        _incomparable(f"{label}.candidate_id differs from embedded endpoint payload")
    if runtime_identity != endpoint.runtime_identity:
        _incomparable(f"{label}.runtime_identity differs from embedded endpoint payload")
    return StartupArtifact(
        candidate_id=candidate_id,
        runtime_identity=runtime_identity,
        endpoint_payload_sha256=_sha256(
            row["endpoint_payload_sha256"], f"{label}.endpoint_payload_sha256"
        ),
        endpoint_payload=endpoint,
    )


def _validate_startup_artifact_bytes(
    raw: bytes, label: str
) -> tuple[dict[str, Any], StartupArtifact]:
    document = qualification._parse_document(raw, label)
    if raw != qualification.canonical_json_bytes(document):
        _fail(
            "noncanonical-startup-artifact",
            f"{label} must be exact canonical JSON bytes for create-only evidence",
        )
    return document, validate_startup_artifact(document, label)


def _validate_endpoint_runtime_identity(
    endpoint: EndpointPayload,
    frozen: qualification.FrozenCandidate,
    *,
    configuration_profile: str,
) -> None:
    if endpoint.candidate_id != frozen.candidate_id:
        _incomparable("endpoint payload belongs to another candidate")
    if configuration_profile not in PROFILE_TO_FREEZE_ARM:
        _fail("unsupported-configuration-profile", "configuration profile is not a C02 receipt arm")
    if endpoint.runtime_identity["configuration_profile"] != configuration_profile:
        _incomparable("endpoint payload profile differs from its evidence arm")
    expected = {
        "configuration_profile": configuration_profile,
        "configuration_sha256": frozen.arms[PROFILE_TO_FREEZE_ARM[configuration_profile]][
            "configuration_sha256"
        ],
    }
    if endpoint.runtime_identity != expected:
        _incomparable("endpoint payload runtime identity drifted from its frozen arm")


def _empty_report() -> dict[str, Any]:
    return {
        "schema_version": CHECK_REPORT_VERSION,
        "status": "failed",
        "passed": False,
        "candidate_id": None,
        "freeze_sha256": None,
        "base_release_candidate_report": None,
        "stable_promotion_profile": None,
        "arms": {},
        "checks": [],
        "reason_codes": [],
    }


CHECK_NAMES = (
    "freeze-binding",
    "base-release-candidate-binding",
    "stable-default-endpoint-effective-config-shape",
    "stable-default-effective-config-hash",
    "stable-default-startup-artifact-binding",
    "stable-default-endpoint-artifact-byte-agreement",
    "max-performance-exact-endpoint-effective-config-shape",
    "max-performance-exact-effective-config-hash",
    "max-performance-exact-startup-artifact-binding",
    "max-performance-exact-endpoint-artifact-byte-agreement",
    "cross-arm-evidence-separation",
)


def _resolved_evidence(value: Any, label: str) -> ResolvedEvidence:
    row = _exact(value, {"path", "sha256"}, label)
    return ResolvedEvidence(
        path=qualification._relative_path(row["path"], f"{label}.path"),
        sha256=_sha256(row["sha256"], f"{label}.sha256"),
    )


def _validate_report_arm(value: Any, profile: str, label: str) -> ArmEvidence:
    row = _exact(
        value,
        {
            "configuration_profile",
            "configuration_sha256",
            "endpoint_payload",
            "startup_artifact",
            "effective_config_sha256",
        },
        label,
    )
    declared_profile = qualification._string(
        row["configuration_profile"], f"{label}.configuration_profile"
    )
    if declared_profile != profile:
        _fail("invalid-config-check-report", f"{label} profile does not match its closed arm key")
    return ArmEvidence(
        configuration_profile=profile,
        configuration_sha256=_sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
        endpoint_payload=_resolved_evidence(
            row["endpoint_payload"], f"{label}.endpoint_payload"
        ),
        startup_artifact=_resolved_evidence(
            row["startup_artifact"], f"{label}.startup_artifact"
        ),
        effective_config_sha256=_sha256(
            row["effective_config_sha256"], f"{label}.effective_config_sha256"
        ),
    )


def _validate_report_arms(value: Any, base_report: ResolvedEvidence) -> dict[str, ArmEvidence]:
    row = _exact(value, set(ARM_PROFILES), "config check report.arms")
    arms = {
        profile: _validate_report_arm(row[profile], profile, f"config check report.arms.{profile}")
        for profile in ARM_PROFILES
    }
    paths = [
        base_report.path,
        *(
            path
            for arm in arms.values()
            for path in (arm.endpoint_payload.path, arm.startup_artifact.path)
        ),
    ]
    if len(paths) != len(set(paths)):
        _fail("duplicate-config-evidence-path", "config report evidence paths must be distinct")
    # The frozen command/environment inputs are already required to differ by
    # the outer freeze.  C02 additionally records what the process actually
    # resolved at runtime, so neither arm may reuse the other's resolved
    # configuration nor pretend that different evidence paths prove the same
    # bytes.  This remains useful to the outer parser before it starts the
    # expensive exact replay below.
    comparisons = (
        (
            "configuration_sha256",
            tuple(arm.configuration_sha256 for arm in arms.values()),
            "indistinguishable-config-arms",
            "config report arm configuration hashes must differ",
        ),
        (
            "endpoint_payload.sha256",
            tuple(arm.endpoint_payload.sha256 for arm in arms.values()),
            "indistinguishable-config-evidence",
            "config report endpoint payload bytes must differ across arms",
        ),
        (
            "startup_artifact.sha256",
            tuple(arm.startup_artifact.sha256 for arm in arms.values()),
            "indistinguishable-config-evidence",
            "config report startup artifact bytes must differ across arms",
        ),
        (
            "effective_config_sha256",
            tuple(arm.effective_config_sha256 for arm in arms.values()),
            "indistinguishable-effective-config",
            "config report effective configuration hashes must differ across arms",
        ),
    )
    for _field, values, code, message in comparisons:
        if len(values) != len(set(values)):
            _fail(code, message)
    return arms


def validate_check_report(document: dict[str, Any]) -> ConfigCheckReport:
    """Validate a passed config-check report before an outer checker reruns it.

    This parser does not make a self-authored report authoritative.  It only
    exposes closed, relative evidence descriptors so a caller can invoke
    :func:`evaluate` again and exact-compare the resulting report bytes.
    """

    row = _exact(
        document,
        {
            "schema_version",
            "status",
            "passed",
            "candidate_id",
            "freeze_sha256",
            "base_release_candidate_report",
            "stable_promotion_profile",
            "arms",
            "checks",
            "reason_codes",
        },
        "effective runtime config check report",
    )
    if row["schema_version"] != CHECK_REPORT_VERSION:
        _fail("unsupported-config-check-report-version", "config check report schema_version is unsupported")
    if row["status"] != "passed" or row["passed"] is not True:
        _fail("config-check-not-passed", "config check report must be passed before outer revalidation")
    if row["reason_codes"] != []:
        _fail("invalid-config-check-report", "a passed config check report must have no reason codes")
    checks = row["checks"]
    if not isinstance(checks, list) or len(checks) != len(CHECK_NAMES):
        _fail("invalid-config-check-report", "config check report has an invalid check inventory")
    names: list[str] = []
    for index, check in enumerate(checks):
        item = _exact(check, {"name", "passed"}, f"config check report.checks[{index}]")
        if item["passed"] is not True:
            _fail("config-check-not-passed", f"config check report check {item['name']!r} did not pass")
        names.append(qualification._string(item["name"], f"config check report.checks[{index}].name"))
    if tuple(names) != CHECK_NAMES:
        _fail("invalid-config-check-report", "config check report check inventory drifted")
    if row["stable_promotion_profile"] != STABLE_DEFAULT_PROFILE:
        _fail(
            "invalid-config-check-report",
            "only stable-default may be the stable promotion configuration profile",
        )
    base_report = _resolved_evidence(
        row["base_release_candidate_report"], "config check report.base_release_candidate_report"
    )
    return ConfigCheckReport(
        candidate_id=_candidate_id(row["candidate_id"], "config check report.candidate_id"),
        freeze_sha256=_sha256(row["freeze_sha256"], "config check report.freeze_sha256"),
        base_release_candidate_report=base_report,
        stable_promotion_profile=STABLE_DEFAULT_PROFILE,
        arms=_validate_report_arms(row["arms"], base_report),
    )


def _descriptor(path: Path | str, raw: bytes) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _read_evidence_bytes(
    evidence_root: Path,
    relative_path: Path | str,
    label: str,
) -> tuple[str, bytes]:
    """Read an endpoint/artifact only from the external immutable evidence root."""

    relative = qualification._relative_path(str(relative_path), f"{label}.path")
    return relative, qualification._read_relative(evidence_root, relative, label)


def _resolve_arm_paths(
    frozen: qualification.FrozenCandidate,
    *,
    stable_endpoint_payload_path: Path | str,
    stable_startup_artifact_path: Path | str,
    max_endpoint_payload_path: Path | str,
    max_startup_artifact_path: Path | str,
) -> dict[str, tuple[str, str]]:
    """Normalize and separate every raw dual-arm input before opening it."""

    supplied = {
        STABLE_DEFAULT_PROFILE: (stable_endpoint_payload_path, stable_startup_artifact_path),
        MAX_PERFORMANCE_EXACT_PROFILE: (max_endpoint_payload_path, max_startup_artifact_path),
    }
    resolved: dict[str, tuple[str, str]] = {}
    all_paths: list[str] = []
    for profile in ARM_PROFILES:
        endpoint_path, artifact_path = supplied[profile]
        endpoint = qualification._relative_path(
            str(endpoint_path), f"{profile} endpoint payload.path"
        )
        artifact = qualification._relative_path(
            str(artifact_path), f"{profile} startup artifact.path"
        )
        resolved[profile] = (endpoint, artifact)
        all_paths.extend((endpoint, artifact))
    if len(all_paths) != len(set(all_paths)):
        _fail(
            "duplicate-config-evidence-path",
            "stable-default and max-performance-exact endpoint/artifact paths must all differ",
        )
    reserved = {
        frozen.final_manifest.path,
        frozen.final_report.path,
        *(descriptor.path for descriptor in frozen.receipts.values()),
    }
    collision = sorted(set(all_paths) & reserved)
    if collision:
        _fail(
            "reserved-output-path-collision",
            f"raw configuration evidence overlaps freeze-declared output paths: {collision}",
        )
    return resolved


def _evaluate_arm_evidence(
    *,
    profile: str,
    endpoint_relative: str,
    startup_artifact_relative: str,
    evidence_root: Path,
    frozen: qualification.FrozenCandidate,
) -> dict[str, Any]:
    """Replay one profile's endpoint/artifact byte agreement and arm binding."""

    endpoint_raw = qualification._read_relative(
        evidence_root, endpoint_relative, f"{profile} endpoint payload"
    )
    endpoint_document, endpoint = _validate_endpoint_bytes(
        endpoint_raw, f"{profile} endpoint payload"
    )
    _validate_endpoint_runtime_identity(
        endpoint,
        frozen,
        configuration_profile=profile,
    )

    artifact_raw = qualification._read_relative(
        evidence_root, startup_artifact_relative, f"{profile} startup artifact"
    )
    artifact_document, artifact = _validate_startup_artifact_bytes(
        artifact_raw, f"{profile} startup artifact"
    )
    _validate_endpoint_runtime_identity(
        artifact.endpoint_payload,
        frozen,
        configuration_profile=profile,
    )
    if (
        artifact.candidate_id != endpoint.candidate_id
        or artifact.runtime_identity != endpoint.runtime_identity
    ):
        _incomparable(f"{profile} startup artifact runtime identity differs from endpoint payload")
    if artifact.endpoint_payload.effective_config_sha256 != endpoint.effective_config_sha256:
        _incomparable(f"{profile} startup artifact effective configuration differs from endpoint payload")

    endpoint_digest = hashlib.sha256(endpoint_raw).hexdigest()
    embedded_endpoint_raw = qualification.canonical_json_bytes(artifact_document["endpoint_payload"])
    if artifact.endpoint_payload_sha256 != endpoint_digest:
        _fail(
            "endpoint-artifact-digest-mismatch",
            f"{profile} startup artifact endpoint payload digest differs",
        )
    if embedded_endpoint_raw != endpoint_raw or artifact_document["endpoint_payload"] != endpoint_document:
        _fail(
            "endpoint-artifact-byte-mismatch",
            f"{profile} startup artifact does not embed endpoint bytes exactly",
        )
    return {
        "configuration_profile": profile,
        "configuration_sha256": endpoint.runtime_identity["configuration_sha256"],
        "endpoint_payload": _descriptor(endpoint_relative, endpoint_raw),
        "startup_artifact": _descriptor(startup_artifact_relative, artifact_raw),
        "effective_config_sha256": endpoint.effective_config_sha256,
    }


def evaluate(
    freeze_path: Path,
    evidence_root: Path,
    stable_endpoint_payload_path: Path | str,
    stable_startup_artifact_path: Path | str,
    max_endpoint_payload_path: Path | str,
    max_startup_artifact_path: Path | str,
    *,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    """Verify both frozen C02 configuration arms without executing an engine.

    Passing remains a *stable-default* promotion result, but it is emitted
    only after the separately frozen max-performance-exact evidence is also
    parsed, byte-bound, and replayed.  Callers cannot silently substitute the
    stable proof for the opt-in arm or omit that required audit evidence.
    """

    report = _empty_report()
    try:
        expected_freeze_digest = _sha256(
            expected_freeze_sha256, "--expected-freeze-sha256"
        )
        freeze_raw = qualification._read_regular_path(freeze_path, "freeze manifest")
        freeze_sha256 = hashlib.sha256(freeze_raw).hexdigest()
        report["freeze_sha256"] = freeze_sha256
        if freeze_sha256 != expected_freeze_digest:
            _fail("candidate-sha-mismatch", "freeze manifest SHA-256 differs from trusted input")
        frozen = qualification._validate_freeze(
            qualification._parse_document(freeze_raw, "freeze manifest")
        )
        report["candidate_id"] = frozen.candidate_id
        arm_paths = _resolve_arm_paths(
            frozen,
            stable_endpoint_payload_path=stable_endpoint_payload_path,
            stable_startup_artifact_path=stable_startup_artifact_path,
            max_endpoint_payload_path=max_endpoint_payload_path,
            max_startup_artifact_path=max_startup_artifact_path,
        )

        # Gate E cannot be a hand-authored ``passed`` document.  The outer
        # checker snapshots the freeze-declared manifest, replays its evidence
        # through check_release_candidate, and exact-compares the submitted
        # result before handing the bytes to this C02-specific verifier.
        base_raw, base_report_sha256 = qualification.revalidate_base_release_candidate(
            frozen,
            freeze_sha256,
            evidence_root,
        )
        if hashlib.sha256(base_raw).hexdigest() != base_report_sha256:
            _fail(
                "base-report-replay-digest-mismatch",
                "Gate E revalidation returned inconsistent base report bytes/digest",
        )
        report["base_release_candidate_report"] = _descriptor(frozen.final_report.path, base_raw)
        report["arms"] = {
            profile: _evaluate_arm_evidence(
                profile=profile,
                endpoint_relative=arm_paths[profile][0],
                startup_artifact_relative=arm_paths[profile][1],
                evidence_root=evidence_root,
                frozen=frozen,
            )
            for profile in ARM_PROFILES
        }
        # Keep the emitted report subject to the same closed-inventory rules
        # that an outer semantic validator applies before it replays us.
        _validate_report_arms(
            report["arms"],
            _resolved_evidence(
                report["base_release_candidate_report"],
                "config check report.base_release_candidate_report",
            ),
        )

        report.update(
            {
                "status": "passed",
                "passed": True,
                "stable_promotion_profile": STABLE_DEFAULT_PROFILE,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument(
        "--stable-endpoint-payload",
        required=True,
        help="stable-default endpoint payload path relative to --evidence-root",
    )
    parser.add_argument(
        "--stable-startup-artifact",
        required=True,
        help="stable-default startup artifact path relative to --evidence-root",
    )
    parser.add_argument(
        "--max-endpoint-payload",
        required=True,
        help="max-performance-exact endpoint payload path relative to --evidence-root",
    )
    parser.add_argument(
        "--max-startup-artifact",
        required=True,
        help="max-performance-exact startup artifact path relative to --evidence-root",
    )
    parser.add_argument("--report", type=Path, help="create without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.freeze,
        args.evidence_root,
        args.stable_endpoint_payload,
        args.stable_startup_artifact,
        args.max_endpoint_payload,
        args.max_startup_artifact,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            qualification._write_create_only(args.report, report)
        except qualification.QualificationError as error:
            print(str(error), file=sys.stderr)
            return 2
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
