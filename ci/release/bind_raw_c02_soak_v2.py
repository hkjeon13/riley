#!/usr/bin/env python3
"""Create one completed, raw-only C02 soak provenance-v3 manifest.

This binder never starts a service, invokes a GPU tool, or decides whether a
candidate qualifies.  It takes a canonical path-only bind request, reads every
declared leaf through one held private evidence-root FD, derives all
descriptors from those bytes, and publishes a create-only manifest followed by
its exact sibling completion marker.  The returned report remains strictly
``bound`` / ``not-run``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common


BIND_REQUEST_VERSION = "riley.soak-v2-bind-request.v3"
MAX_BIND_REQUEST_BYTES = common.DEFAULT_MAX_JSON_BYTES
MAX_RAW_LEAF_BYTES = checker.MAX_RAW_BYTES
MAX_RELATIVE_PATH_BYTES = checker.MAX_RELATIVE_PATH_BYTES
MAX_MANIFEST_NAME_BYTES = checker.MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES
MANIFEST_NAME_RE = checker.SOAK_TERMINAL_MANIFEST_NAME_RE


class RawSoakBindError(ValueError):
    """A raw soak bind request cannot safely be published."""


def _fail(code: str, message: str) -> NoReturn:
    error = RawSoakBindError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _runtime_config(call: Any) -> Any:
    try:
        return call()
    except runtime_config.EffectiveRuntimeConfigError as error:
        _fail(getattr(error, "reason_code", "invalid-runtime-config"), str(error))


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail("unexpected-field-set", f"{label} must contain exactly {sorted(fields)}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or common.SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail("invalid-sha256", f"{label} must be a non-zero lowercase SHA-256")
    return value


def _candidate_id(value: Any, label: str) -> str:
    if type(value) is not str or checker.CANDIDATE_RE.fullmatch(value) is None:
        _fail("invalid-candidate-id", f"{label} must be a canonical RC candidate ID")
    return value


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail("invalid-integer", f"{label} must be a positive integer")
    return value


def _nonnegative(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail("invalid-integer", f"{label} must be a non-negative integer")
    return value


def _target(value: Any, label: str) -> dict[str, Any]:
    row = _exact(
        value,
        {"server_pid", "server_start_ticks", "gpu_index", "gpu_uuid"},
        label,
    )
    gpu_uuid = row["gpu_uuid"]
    if type(gpu_uuid) is not str or checker.GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        _fail("invalid-gpu-uuid", f"{label}.gpu_uuid must be a canonical GPU UUID")
    return {
        "server_pid": _positive(row["server_pid"], f"{label}.server_pid"),
        "server_start_ticks": _positive(
            row["server_start_ticks"], f"{label}.server_start_ticks"
        ),
        "gpu_index": _nonnegative(row["gpu_index"], f"{label}.gpu_index"),
        "gpu_uuid": gpu_uuid,
    }


def _manifest_name(value: str) -> str:
    if (
        type(value) is not str
        or len(value) > MAX_MANIFEST_NAME_BYTES
        or MANIFEST_NAME_RE.fullmatch(value) is None
        or "/" in value
    ):
        _fail(
            "invalid-manifest-name",
            "--manifest-name must be a nonhidden root direct-child .json name of at most "
            f"{MAX_MANIFEST_NAME_BYTES} bytes",
        )
    return value


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    """Reject a lexical source-checkout child before opening evidence.

    The held-FD common primitive still performs the authoritative no-follow
    absolute traversal.  This preflight merely prevents the known source tree
    containing this binder from being used as a terminal evidence location;
    it never reads a requested evidence leaf through a pathname API.
    """

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
    maximum_bytes: int,
) -> tuple[str, bytes, dict[str, Any]]:
    relative = _path(path, f"{label}.path")
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            relative,
            label,
            maximum_bytes=maximum_bytes,
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(raw, label, maximum_bytes=maximum_bytes)
    )
    assert isinstance(document, dict)
    return relative, raw, document


def _descriptor_from_path(
    root_fd: int,
    path: str,
    label: str,
    *,
    maximum_bytes: int = MAX_RAW_LEAF_BYTES,
) -> tuple[common.EvidenceDescriptor, bytes]:
    relative = _path(path, f"{label}.path")
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            relative,
            label,
            maximum_bytes=maximum_bytes,
        )
    )
    if not raw:
        _fail("empty-evidence-leaf", f"{label} must bind nonempty raw evidence")
    return _common(lambda: common.descriptor_for_bytes(relative, raw, label)), raw


def _path(value: Any, label: str) -> str:
    relative = _common(lambda: common.validate_relative_path(value, label))
    if len(relative) > MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label} exceeds {MAX_RELATIVE_PATH_BYTES} bytes",
        )
    return relative


def _binding_inputs(value: Any) -> dict[str, str]:
    row = _exact(
        value,
        {
            "freeze_sha256",
            "base_release_candidate_report_sha256",
            "configuration_profile",
        },
        "soak bind request.binding_inputs",
    )
    profile = row["configuration_profile"]
    if type(profile) is not str or profile not in checker.SOAK_CONFIGURATION_PROFILES:
        _fail(
            "invalid-configuration-profile",
            "soak bind request.binding_inputs.configuration_profile must be a C02 soak arm",
        )
    return {
        "freeze_sha256": _sha256(
            row["freeze_sha256"], "soak bind request.binding_inputs.freeze_sha256"
        ),
        "base_release_candidate_report_sha256": _sha256(
            row["base_release_candidate_report_sha256"],
            "soak bind request.binding_inputs.base_release_candidate_report_sha256",
        ),
        "configuration_profile": profile,
    }


def _reserve_declared_path(path: str, label: str, paths: set[str]) -> str:
    relative = _path(path, label)
    if relative in paths:
        _fail("duplicate-evidence-path", f"{label} reuses declared evidence path {relative!r}")
    paths.add(relative)
    return relative


def _configuration_evidence(
    root_fd: int,
    value: Any,
    *,
    candidate_id: str,
    binding_inputs: Mapping[str, str],
    declared_paths: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    row = _exact(
        value,
        {"endpoint_path", "startup_artifact_path", "endpoint_observation_path"},
        "soak bind request.configuration_evidence",
    )
    endpoint_path = _reserve_declared_path(
        row["endpoint_path"],
        "soak bind request.configuration_evidence.endpoint_path",
        declared_paths,
    )
    startup_path = _reserve_declared_path(
        row["startup_artifact_path"],
        "soak bind request.configuration_evidence.startup_artifact_path",
        declared_paths,
    )
    observation_path = _reserve_declared_path(
        row["endpoint_observation_path"],
        "soak bind request.configuration_evidence.endpoint_observation_path",
        declared_paths,
    )
    endpoint_descriptor, endpoint_raw = _descriptor_from_path(
        root_fd,
        endpoint_path,
        "soak raw configuration endpoint",
        maximum_bytes=runtime_config.MAX_JSON_BYTES,
    )
    startup_descriptor, startup_raw = _descriptor_from_path(
        root_fd,
        startup_path,
        "soak raw configuration startup artifact",
        maximum_bytes=runtime_config.MAX_JSON_BYTES,
    )
    observation_descriptor, _observation_raw = _descriptor_from_path(
        root_fd,
        observation_path,
        "soak raw configuration endpoint observation",
    )
    endpoint_document, endpoint = _runtime_config(
        lambda: runtime_config.validate_endpoint_bytes(
            endpoint_raw, "soak raw configuration endpoint"
        )
    )
    startup_document, startup = _runtime_config(
        lambda: runtime_config.validate_startup_artifact_bytes(
            startup_raw, "soak raw configuration startup artifact"
        )
    )
    if startup.endpoint_payload_sha256 != hashlib.sha256(endpoint_raw).hexdigest():
        _fail(
            "startup-endpoint-hash-mismatch",
            "soak startup artifact does not hash the held-FD endpoint bytes",
        )
    if startup_document["endpoint_payload"] != endpoint_document:
        _fail(
            "startup-endpoint-payload-mismatch",
            "soak startup artifact does not embed the held-FD endpoint payload",
        )
    if endpoint.candidate_id != candidate_id or startup.candidate_id != candidate_id:
        _fail(
            "runtime-config-candidate-mismatch",
            "soak configuration evidence candidate differs from the bind request",
        )
    if endpoint.runtime_identity != startup.runtime_identity:
        _fail(
            "runtime-config-identity-mismatch",
            "soak endpoint and startup artifact runtime identities differ",
        )
    identity = endpoint.runtime_identity
    if identity["configuration_profile"] != binding_inputs["configuration_profile"]:
        _fail(
            "runtime-config-profile-mismatch",
            "soak runtime identity profile differs from the bind request",
        )
    return (
        {
            "endpoint": endpoint_descriptor.as_json(),
            "startup_artifact": startup_descriptor.as_json(),
            "endpoint_observation": observation_descriptor.as_json(),
        },
        identity,
    )


def _scenario_manifest_rows(
    root_fd: int,
    value: Any,
    *,
    configuration_profile: str,
    declared_paths: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= checker.MAX_SCENARIOS:
        _fail("invalid-scenario-inventory", "soak bind request must contain a bounded nonempty scenario list")
    seen_ids: set[str] = set()
    scenarios: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        label = f"soak bind request.scenarios[{index}]"
        row = _exact(
            item,
            {
                "scenario_id",
                "target",
                "observation_session_path",
                "request_ledger_path",
                "runtime_event_log_path",
                "generation_audit_index_path",
                "fallback_event_log_path",
            },
            label,
        )
        scenario_id = row["scenario_id"]
        if (
            type(scenario_id) is not str
            or len(scenario_id) > 128
            or checker.SCENARIO_ID_RE.fullmatch(scenario_id) is None
            or scenario_id in seen_ids
        ):
            _fail(
                "invalid-scenario-inventory",
                f"{label}.scenario_id must be a unique canonical scenario identifier",
            )
        seen_ids.add(scenario_id)
        target = _target(row["target"], f"{label}.target")
        paths = {
            "observation_session": _reserve_declared_path(
                row["observation_session_path"],
                f"{label}.observation_session_path",
                declared_paths,
            ),
            "request_ledger": _reserve_declared_path(
                row["request_ledger_path"],
                f"{label}.request_ledger_path",
                declared_paths,
            ),
            "runtime_event_log": _reserve_declared_path(
                row["runtime_event_log_path"],
                f"{label}.runtime_event_log_path",
                declared_paths,
            ),
            "generation_audit_index": _reserve_declared_path(
                row["generation_audit_index_path"],
                f"{label}.generation_audit_index_path",
                declared_paths,
            ),
        }
        fallback_path = row["fallback_event_log_path"]
        if fallback_path is not None:
            paths["fallback_event_log"] = _reserve_declared_path(
                fallback_path,
                f"{label}.fallback_event_log_path",
                declared_paths,
            )
        if scenario_id == "exact-backend-fallback":
            if configuration_profile != checker.MAX_PERFORMANCE_EXACT_PROFILE:
                _fail(
                    "fallback-profile-mismatch",
                    f"{label} requires {checker.MAX_PERFORMANCE_EXACT_PROFILE}",
                )
            if fallback_path is None:
                _fail("fallback-raw-leaf-missing", f"{label} lacks a fallback event-log path")
        descriptors: dict[str, dict[str, Any]] = {}
        for key, path in paths.items():
            descriptor, _raw = _descriptor_from_path(root_fd, path, f"{label}.{key}")
            descriptors[key] = descriptor.as_json()
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "target": target,
                "observation_session": descriptors["observation_session"],
                "request_ledger": descriptors["request_ledger"],
                "runtime_event_log": descriptors["runtime_event_log"],
                "generation_audit_index": descriptors["generation_audit_index"],
                "fallback_event_log": descriptors.get("fallback_event_log"),
            }
        )
    return scenarios


def _manifest_from_request(
    root_fd: int,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    request_path, _request_raw, request = _read_canonical_object(
        root_fd,
        bind_request_path,
        "soak bind request",
        maximum_bytes=MAX_BIND_REQUEST_BYTES,
    )
    row = _exact(
        request,
        {
            "schema_version",
            "candidate_id",
            "binding_inputs",
            "configuration_evidence",
            "scenario_contract_path",
            "scenarios",
        },
        "soak bind request",
    )
    if row["schema_version"] != BIND_REQUEST_VERSION:
        _fail(
            "unsupported-bind-request-version",
            f"soak bind request must use {BIND_REQUEST_VERSION}",
        )
    candidate_id = _candidate_id(row["candidate_id"], "soak bind request.candidate_id")
    binding_inputs = _binding_inputs(row["binding_inputs"])
    marker_name = f"{manifest_name}.complete"
    declared_paths = {request_path}
    configuration_evidence, runtime_identity = _configuration_evidence(
        root_fd,
        row["configuration_evidence"],
        candidate_id=candidate_id,
        binding_inputs=binding_inputs,
        declared_paths=declared_paths,
    )
    scenario_contract_path = _reserve_declared_path(
        row["scenario_contract_path"],
        "soak bind request.scenario_contract_path",
        declared_paths,
    )
    scenario_contract, _contract_raw = _descriptor_from_path(
        root_fd,
        scenario_contract_path,
        "soak scenario contract",
    )
    scenarios = _scenario_manifest_rows(
        root_fd,
        row["scenarios"],
        configuration_profile=binding_inputs["configuration_profile"],
        declared_paths=declared_paths,
    )
    if manifest_name in declared_paths or marker_name in declared_paths:
        _fail(
            "output-name-collision",
            "manifest or completion-marker name collides with a declared input path",
        )
    return {
        "schema_version": checker.SOAK_MANIFEST_VERSION,
        "capture_status": "captured",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "bindings": {
            "freeze_sha256": binding_inputs["freeze_sha256"],
            "base_release_candidate_report_sha256": binding_inputs[
                "base_release_candidate_report_sha256"
            ],
            "configuration_profile": binding_inputs["configuration_profile"],
            # This is the endpoint runtime identity, not a descriptor digest
            # and not effective_config_sha256.
            "configuration_sha256": runtime_identity["configuration_sha256"],
        },
        "configuration_evidence": configuration_evidence,
        "scenario_contract": scenario_contract.as_json(),
        "scenarios": scenarios,
    }


def bind_raw_soak_manifest_fd(
    root_fd: int,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    """Publish and terminally verify one manifest via a caller-held root FD.

    This function intentionally has no retry or cleanup path.  Once a manifest
    name exists, a later invocation cannot replace it; if marker publication
    fails, the unmarked manifest remains nonterminal and must be investigated.
    """

    name = _manifest_name(manifest_name)
    manifest = _manifest_from_request(root_fd, bind_request_path, name)
    created = _common(
        lambda: common.write_create_only_json(
            root_fd,
            name,
            manifest,
            "soak raw manifest",
        )
    )
    # Verify descriptors and P0 config bytes through the same held root before
    # publishing a terminal marker.
    checker.verify_soak_provenance_fd(root_fd, name)
    marker = {
        "schema_version": checker.SOAK_COMPLETION_MARKER_VERSION,
        "artifact_filename": name,
        "artifact_sha256": created.sha256,
    }
    _common(
        lambda: common.write_create_only_json(
            root_fd,
            f"{name}.complete",
            marker,
            "soak raw manifest completion marker",
        )
    )
    return checker.verify_completed_soak_provenance_fd(root_fd, name)


def bind_raw_soak_manifest(
    evidence_root: Path,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    """Open one private root and bind a terminal raw soak manifest there."""

    _assert_external_to_source_checkout(evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(evidence_root, "--evidence-root")
    )
    try:
        return bind_raw_soak_manifest_fd(root_fd, bind_request_path, manifest_name)
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--bind-request", required=True)
    parser.add_argument("--manifest-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = bind_raw_soak_manifest(
            args.evidence_root,
            args.bind_request,
            args.manifest_name,
        )
    except (
        RawSoakBindError,
        checker.C02ProvenanceError,
        common.ProvenanceV2Error,
        runtime_config.EffectiveRuntimeConfigError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
