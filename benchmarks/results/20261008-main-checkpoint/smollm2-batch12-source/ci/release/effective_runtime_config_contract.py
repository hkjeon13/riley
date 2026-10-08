#!/usr/bin/env python3
"""P0 contract for a runtime's effective-configuration startup facts.

This module deliberately owns only the raw, pre-freeze contract: canonical
``GET /v1/config`` bytes, a create-only startup artifact, and the local file
operations needed to build that artifact. It never parses a candidate freeze,
replays Gate E, or decides a C02 qualification result.

``validate_raw_c02_runtime_config.py`` intentionally duplicates this raw
schema under its own Python-3.10 isolated/stdlib-only boundary. Do not make
that remote producer import this module: ``python -I -S`` does not include
the script directory on ``sys.path`` on the GPU host.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn


ENDPOINT_VERSION = "riley.effective-runtime-config.v1"
STARTUP_ARTIFACT_VERSION = "riley.effective-runtime-config-startup-artifact.v1"
STABLE_DEFAULT_PROFILE = "stable-default"
MAX_PERFORMANCE_EXACT_PROFILE = "max-performance-exact"
ARM_PROFILES = (STABLE_DEFAULT_PROFILE, MAX_PERFORMANCE_EXACT_PROFILE)

# The isolated Python-3.10 remote validator mirrors this P0 contract in a
# self-contained file. It cannot import this module under isolated mode;
# test_validate_raw_c02_runtime_config exercises the same valid and adversarial
# raw contract.
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

MAX_JSON_BYTES = 8 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CANDIDATE_ID_RE = re.compile(
    r"^riley-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)-rc(?:[1-9][0-9]*)$"
)
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
IMPLEMENTATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,255}$")
EXPERIMENTAL_FLAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class EffectiveRuntimeConfigError(ValueError):
    """A P0 effective-runtime-config fact is malformed or unsafe."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = code


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
    """Digest-level result returned after a create-only startup write."""

    candidate_id: str
    configuration_profile: str
    endpoint_payload_sha256: str
    startup_artifact_sha256: str
    path: Path


def _fail(code: str, message: str) -> NoReturn:
    raise EffectiveRuntimeConfigError(code, message)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON encoding bound by endpoint and artifact bytes."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        _fail("noncanonical-json", f"JSON value cannot be canonically encoded: {error}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate-json-key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail("non-finite-json-number", f"non-finite JSON number {value!r} is forbidden")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail("non-finite-json-number", f"non-finite JSON number {value!r} is forbidden")
    return parsed


def _parse_document(raw: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
            parse_float=_finite_float,
        )
    except UnicodeDecodeError as error:
        _fail("invalid-json", f"{label} is not UTF-8: {error}")
    except json.JSONDecodeError as error:
        _fail("invalid-json", f"{label} is not JSON: {error}")
    if not isinstance(decoded, dict):
        _fail("invalid-json", f"{label} root must be an object")
    return decoded


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("invalid-shape", f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; missing={sorted(fields - actual)}, extra={sorted(actual - fields)}",
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail("invalid-string", f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    value = _string(value, label)
    if not SHA256_RE.fullmatch(value) or value == "0" * 64:
        _fail("invalid-sha256", f"{label} must be a lowercase SHA-256")
    return value


def _candidate_id(value: Any, label: str) -> str:
    candidate_id = _string(value, label)
    if not CANDIDATE_ID_RE.fullmatch(candidate_id):
        _fail("invalid-candidate-id", f"{label} is not a valid RC candidate")
    return candidate_id


def _implementation_id(value: Any, label: str) -> str:
    value = _string(value, label)
    if not IMPLEMENTATION_ID_RE.fullmatch(value):
        _fail("invalid-effective-config", f"{label} must be a lowercase implementation ID")
    return value


def _positive_integer(value: Any, label: str) -> int:
    # bool is an int subclass, but never an acceptable runtime bound.
    if type(value) is not int or value <= 0:
        _fail("invalid-effective-config", f"{label} must be a positive integer")
    return value


def _runtime_identity(value: Any, label: str) -> dict[str, str]:
    row = _exact(value, {"configuration_profile", "configuration_sha256"}, label)
    profile = _string(row["configuration_profile"], f"{label}.configuration_profile")
    if profile not in ARM_PROFILES:
        _fail("incomparable-binding", f"{label}.configuration_profile is not a C02 arm")
    return {
        "configuration_profile": profile,
        "configuration_sha256": _sha256(
            row["configuration_sha256"], f"{label}.configuration_sha256"
        ),
    }


def _choice(value: Any, allowed: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail("invalid-effective-config", f"{label} is unsupported")
    return value


def validate_effective_config(value: Any, label: str = "effective config") -> dict[str, Any]:
    """Validate and normalize the closed effective-runtime configuration map.

    C02 consumers that bind a pre-launch configuration intent to a captured
    ``/v1/config`` fact must use this same semantic grammar rather than only a
    JSON-schema shape check.  In particular, bucket, transport, and fallback
    invariants are enforced here.
    """

    row = _exact(value, set(CONFIG_DIMENSIONS), label)

    completion = _choice(
        row["execution_completion_mode"],
        {"per-operation", "iteration-batch"},
        f"{label}.execution_completion_mode",
    )
    batch_token_budget = _positive_integer(
        row["batch_token_budget"], f"{label}.batch_token_budget"
    )
    batch_shape = _exact(row["batch_shape"], {"policy", "buckets"}, f"{label}.batch_shape")
    policy = _choice(
        batch_shape["policy"], {"fixed-max", "power-of-two"}, f"{label}.batch_shape.policy"
    )
    buckets_value = batch_shape["buckets"]
    if not isinstance(buckets_value, list) or not buckets_value or len(buckets_value) > 32:
        _fail("invalid-effective-config", f"{label}.batch_shape.buckets must be a non-empty bounded list")
    buckets = [
        _positive_integer(bucket, f"{label}.batch_shape.buckets[{index}]")
        for index, bucket in enumerate(buckets_value)
    ]
    if len(set(buckets)) != len(buckets) or any(
        left >= right for left, right in zip(buckets, buckets[1:])
    ):
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

    metadata_transport = _choice(
        row["metadata_transport"],
        {"synchronous", "packed-async"},
        f"{label}.metadata_transport",
    )
    if metadata_transport == "packed-async" and completion != "iteration-batch":
        _fail(
            "invalid-effective-config",
            f"{label}.packed-async metadata requires iteration-batch completion",
        )

    sampling_backend = _choice(
        row["sampling_backend"], {"cpu", "gpu-greedy"}, f"{label}.sampling_backend"
    )
    attention = _exact(row["attention_backend"], {"prefill", "decode"}, f"{label}.attention_backend")
    attention_prefill = _implementation_id(
        attention["prefill"], f"{label}.attention_backend.prefill"
    )
    attention_decode = _implementation_id(
        attention["decode"], f"{label}.attention_backend.decode"
    )
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
    runtime_selection = _choice(
        fallback["runtime_selection"],
        {"exact-fallback-allowed", "fail-closed"},
        f"{label}.fallback_policy.runtime_selection",
    )

    kv_geometry = _exact(
        row["kv_geometry"],
        {"layout", "block_tokens", "physical_blocks"},
        f"{label}.kv_geometry",
    )
    layout = _choice(kv_geometry["layout"], {"contiguous", "paged"}, f"{label}.kv_geometry.layout")
    block_tokens = _positive_integer(
        kv_geometry["block_tokens"], f"{label}.kv_geometry.block_tokens"
    )
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
            "runtime_selection": runtime_selection,
        },
        "batch_token_budget": batch_token_budget,
        "kv_geometry": {
            "layout": layout,
            "block_tokens": block_tokens,
            "physical_blocks": physical_blocks,
        },
    }


def validate_endpoint_payload(document: dict[str, Any], label: str = "endpoint payload") -> EndpointPayload:
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
    effective_config = validate_effective_config(
        row["effective_config"],
        f"{label}.effective_config",
    )
    expected_effective_config_sha256 = hashlib.sha256(
        canonical_json_bytes(effective_config)
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
        runtime_identity=_runtime_identity(row["runtime_identity"], f"{label}.runtime_identity"),
        effective_config=effective_config,
        effective_config_sha256=effective_config_sha256,
    )


def validate_endpoint_bytes(raw: bytes, label: str = "endpoint payload") -> tuple[dict[str, Any], EndpointPayload]:
    document = _parse_document(raw, label)
    if raw != canonical_json_bytes(document):
        _fail("noncanonical-endpoint-payload", f"{label} must be exact canonical JSON bytes")
    return document, validate_endpoint_payload(document, label)


def validate_startup_artifact(
    document: dict[str, Any], label: str = "startup artifact"
) -> StartupArtifact:
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
    if not isinstance(row["created_at_utc"], str) or not UTC_RE.fullmatch(row["created_at_utc"]):
        _fail("invalid-created-at", f"{label}.created_at_utc must be UTC second precision")
    if row["endpoint_path"] != "/v1/config":
        _fail("wrong-config-endpoint", f"{label}.endpoint_path must be /v1/config")
    endpoint = validate_endpoint_payload(row["endpoint_payload"], f"{label}.endpoint_payload")
    candidate_id = _candidate_id(row["candidate_id"], f"{label}.candidate_id")
    runtime_identity = _runtime_identity(row["runtime_identity"], f"{label}.runtime_identity")
    if candidate_id != endpoint.candidate_id:
        _fail("incomparable-binding", f"{label}.candidate_id differs from embedded endpoint payload")
    if runtime_identity != endpoint.runtime_identity:
        _fail("incomparable-binding", f"{label}.runtime_identity differs from embedded endpoint payload")
    return StartupArtifact(
        candidate_id=candidate_id,
        runtime_identity=runtime_identity,
        endpoint_payload_sha256=_sha256(
            row["endpoint_payload_sha256"], f"{label}.endpoint_payload_sha256"
        ),
        endpoint_payload=endpoint,
    )


def validate_startup_artifact_bytes(
    raw: bytes, label: str = "startup artifact"
) -> tuple[dict[str, Any], StartupArtifact]:
    document = _parse_document(raw, label)
    if raw != canonical_json_bytes(document):
        _fail("noncanonical-startup-artifact", f"{label} must be exact canonical JSON bytes")
    return document, validate_startup_artifact(document, label)


def _read_fd(fd: int, label: str) -> bytes:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        _fail("unsafe-evidence-path", f"{label} must be a regular file")
    if metadata.st_size > MAX_JSON_BYTES:
        _fail("input-too-large", f"{label} exceeds {MAX_JSON_BYTES} bytes")
    chunks: list[bytes] = []
    remaining = metadata.st_size
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            _fail("truncated-input", f"{label} changed while it was read")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        _fail("mutated-input", f"{label} grew while it was read")
    return b"".join(chunks)


def _required_open_flag(name: str) -> int:
    """Return a race-hardening open flag or reject an unsafe platform."""

    flag = getattr(os, name, None)
    if not isinstance(flag, int) or flag == 0:
        _fail("unsafe-platform", f"effective-runtime-config evidence requires os.{name}")
    return flag


def read_regular_path(path: Path, label: str) -> bytes:
    """Read a bounded regular non-symlink path without following a race."""

    try:
        before = path.lstat()
    except OSError as error:
        _fail("missing-input", f"{label} cannot be inspected: {error}")
    if not stat.S_ISREG(before.st_mode):
        _fail("unsafe-evidence-path", f"{label} must be a regular non-link file")
    flags = os.O_RDONLY | os.O_CLOEXEC | _required_open_flag("O_NOFOLLOW")
    try:
        fd = os.open(path, flags)
    except OSError as error:
        _fail("missing-input", f"{label} cannot be opened safely: {error}")
    try:
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            _fail("raced-input", f"{label} changed while it was opened")
        return _read_fd(fd, label)
    finally:
        os.close(fd)


def write_create_only_bytes(path: Path, encoded: bytes) -> None:
    """Create one regular file without replacement, then durably publish bytes.

    The runtime writes its startup artifact only after cold prepare and before
    accepting traffic. A pre-existing artifact is therefore an error, not an
    instruction to overwrite or append to another run's evidence.
    """

    if path.name in {"", ".", ".."}:
        _fail("unsafe-output-path", "startup artifact must have one regular filename")
    try:
        parent_metadata = path.parent.lstat()
    except OSError as error:
        _fail("unsafe-output-path", f"startup artifact parent cannot be inspected: {error}")
    if not stat.S_ISDIR(parent_metadata.st_mode):
        _fail("unsafe-output-path", "startup artifact parent must be a real directory")
    directory_flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | _required_open_flag("O_DIRECTORY")
        | _required_open_flag("O_NOFOLLOW")
    )
    try:
        parent_fd = os.open(path.parent, directory_flags)
    except OSError as error:
        _fail("unsafe-output-path", f"startup artifact parent cannot be opened safely: {error}")
    try:
        output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | _required_open_flag(
            "O_NOFOLLOW"
        )
        try:
            output_fd = os.open(path.name, output_flags, 0o644, dir_fd=parent_fd)
        except OSError as error:
            _fail("create-only-write-failed", f"cannot create startup artifact {path}: {error}")
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(output_fd, encoded[offset:])
                if written <= 0:
                    _fail("create-only-write-failed", f"short write for startup artifact {path}")
                offset += written
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
