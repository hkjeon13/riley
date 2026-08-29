#!/usr/bin/env python3
"""Cross-bind static RC3 rollback config intent to a captured C02 config fact.

This held-FD-only replayer is a raw input-derivation helper. It neither opens
an evidence-root path nor captures an endpoint, starts a process, inspects a
GPU, writes evidence, or decides a rollback or qualification result. A future
authenticated writer may consume its typed result only while retaining the
same caller-owned root lock and replaying the inputs again before publication.

The static configuration snapshot is intentionally not compared with the
runtime launch-identity SHA-256. They are distinct hash domains. Instead, the
snapshot declares the intended effective configuration and this helper compares
that canonical value and digest with the independently replayed ``/v1/config``
fact.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NoReturn, TypeVar

import check_c02_config_bridge_v1 as config_bridge
import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import prepare_rc3_rollback_evidence_v1 as preparation
import provenance_v2_common as common


STATIC_EFFECTIVE_CONFIG_VERSION = "riley.rc3-static-effective-config.v1"
MAX_STATIC_CONFIGURATION_BYTES = common.DEFAULT_MAX_JSON_BYTES


class StaticEffectiveConfigError(ValueError):
    """Static config intent cannot be safely joined to runtime config evidence."""


def _fail(code: str, message: str) -> NoReturn:
    error = StaticEffectiveConfigError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _preparation(call: Callable[[], T]) -> T:
    try:
        return call()
    except preparation.RollbackEvidencePreparationError as error:
        _fail(getattr(error, "reason_code", "invalid-static-preparation"), str(error))


def _bridge(call: Callable[[], T]) -> T:
    try:
        return call()
    except config_bridge.ConfigBridgeReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-config-bridge"), str(error))


def _runtime_config(call: Callable[[], T]) -> T:
    try:
        return call()
    except runtime_config.EffectiveRuntimeConfigError as error:
        _fail(getattr(error, "reason_code", "invalid-static-effective-config"), str(error))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(
            "unexpected-field-set",
            f"{label} must contain exactly {sorted(fields)}",
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or common.SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail("invalid-sha256", f"{label} must be a non-zero lowercase SHA-256")
    return value


@dataclass(frozen=True)
class StaticPreparationBindings:
    """Exact static preparation leaves retained for a later writer recheck."""

    reconstructed_baseline: common.EvidenceDescriptor
    freeze: common.EvidenceDescriptor
    base_release_candidate_report: common.EvidenceDescriptor
    configuration: common.EvidenceDescriptor


@dataclass(frozen=True)
class ReplayedStaticEffectiveConfig:
    """One static intent and one independently captured effective config fact."""

    candidate_id: str
    configuration_profile: str
    static_bindings: StaticPreparationBindings
    expected_effective_config: dict[str, Any]
    expected_effective_config_sha256: str
    config_bridge: config_bridge.ReplayedConfigBridge

    @property
    def runtime_configuration_sha256(self) -> str:
        """Return only the bridge-derived launch-identity digest."""

        return self.config_bridge.configuration_sha256


def _static_bindings(session: Mapping[str, Any]) -> StaticPreparationBindings:
    reconstructed = _exact(
        session.get("reconstructed_baseline"),
        {"manifest", "baseline_id", "tag_name", "target_commit_sha1"},
        "static preparation reconstructed baseline",
    )
    snapshots = _exact(
        session.get("binding_input_snapshots"),
        {"freeze", "base_release_candidate_report", "configuration"},
        "static preparation binding snapshots",
    )
    baseline_descriptor = _common(
        lambda: common.parse_descriptor(
            reconstructed["manifest"],
            "static preparation reconstructed baseline manifest",
        )
    )
    freeze = _common(
        lambda: common.parse_descriptor(
            snapshots["freeze"],
            "static preparation freeze snapshot",
        )
    )
    base_report = _common(
        lambda: common.parse_descriptor(
            snapshots["base_release_candidate_report"],
            "static preparation base report snapshot",
        )
    )
    configuration = _common(
        lambda: common.parse_descriptor(
            snapshots["configuration"],
            "static preparation configuration snapshot",
        )
    )
    expected_paths = {
        "freeze": f"{preparation.INPUTS_DIRECTORY_NAME}/{preparation.FREEZE_NAME}",
        "base_release_candidate_report": (
            f"{preparation.INPUTS_DIRECTORY_NAME}/{preparation.BASE_REPORT_NAME}"
        ),
        "configuration": (
            f"{preparation.INPUTS_DIRECTORY_NAME}/{preparation.CONFIGURATION_NAME}"
        ),
    }
    if (
        freeze.path != expected_paths["freeze"]
        or base_report.path != expected_paths["base_release_candidate_report"]
        or configuration.path != expected_paths["configuration"]
    ):
        _fail(
            "static-snapshot-path-mismatch",
            "static preparation snapshot paths do not have their fixed names",
        )
    paths = {
        baseline_descriptor.path,
        freeze.path,
        base_report.path,
        configuration.path,
    }
    if len(paths) != 4:
        _fail(
            "duplicate-evidence-path",
            "static baseline and snapshot descriptors must all have distinct paths",
        )
    return StaticPreparationBindings(
        reconstructed_baseline=baseline_descriptor,
        freeze=freeze,
        base_release_candidate_report=base_report,
        configuration=configuration,
    )


def _reserve_static_bindings(
    bindings: StaticPreparationBindings,
    used_paths: set[str],
) -> None:
    for descriptor, label in (
        (bindings.reconstructed_baseline, "static reconstructed baseline"),
        (bindings.freeze, "static freeze snapshot"),
        (bindings.base_release_candidate_report, "static base report snapshot"),
        (bindings.configuration, "static configuration snapshot"),
    ):
        if descriptor.path in used_paths:
            _fail(
                "duplicate-evidence-path",
                f"{label} reuses evidence path {descriptor.path!r}",
            )
        used_paths.add(descriptor.path)


def _read_static_configuration(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
) -> Mapping[str, Any]:
    inputs_fd = _common(
        lambda: common.open_private_child_directory(
            root_fd,
            preparation.INPUTS_DIRECTORY_NAME,
            "static effective config inputs directory",
        )
    )
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                inputs_fd,
                preparation.INPUTS_DIRECTORY_NAME,
                "held static effective config inputs directory",
            )
        )
        rebased = _common(
            lambda: common.rebase_descriptor_to_held_leaf(
                descriptor,
                expected_root_relative_path=(
                    f"{preparation.INPUTS_DIRECTORY_NAME}/{preparation.CONFIGURATION_NAME}"
                ),
                leaf_name=preparation.CONFIGURATION_NAME,
                label="static effective configuration snapshot",
            )
        )
        _raw, document = _common(
            lambda: common.read_private_descriptor_json_leaf(
                inputs_fd,
                rebased,
                "static effective configuration snapshot",
                maximum_bytes=MAX_STATIC_CONFIGURATION_BYTES,
            )
        )
        return document
    finally:
        os.close(inputs_fd)


def _parse_static_effective_config(
    document: Mapping[str, Any],
    *,
    candidate_id: str,
    configuration_profile: str,
) -> tuple[dict[str, Any], str]:
    row = _exact(
        document,
        {
            "schema_version",
            "candidate_id",
            "configuration_profile",
            "expected_effective_config",
            "expected_effective_config_sha256",
        },
        "static effective configuration",
    )
    if row["schema_version"] != STATIC_EFFECTIVE_CONFIG_VERSION:
        _fail(
            "unsupported-static-effective-config-version",
            "static configuration snapshot has an unsupported schema version",
        )
    if row["candidate_id"] != candidate_id:
        _fail(
            "static-config-candidate-mismatch",
            "static configuration candidate differs from static preparation",
        )
    if row["configuration_profile"] != configuration_profile:
        _fail(
            "static-config-profile-mismatch",
            "static configuration profile differs from static preparation",
        )
    expected = _runtime_config(
        lambda: runtime_config.validate_effective_config(
            row["expected_effective_config"],
            "static effective configuration.expected_effective_config",
        )
    )
    declared_sha256 = _sha256(
        row["expected_effective_config_sha256"],
        "static effective configuration.expected_effective_config_sha256",
    )
    actual_sha256 = hashlib.sha256(common.canonical_json_bytes(expected)).hexdigest()
    if declared_sha256 != actual_sha256:
        _fail(
            "expected-effective-config-hash-mismatch",
            "static expected_effective_config_sha256 does not hash its canonical value",
        )
    return expected, declared_sha256


def _session_identity(session: Mapping[str, Any]) -> tuple[str, str]:
    candidate_id = session.get("candidate_id")
    profile = session.get("configuration_profile")
    if type(candidate_id) is not str or checker.CANDIDATE_RE.fullmatch(candidate_id) is None:
        _fail("invalid-candidate-id", "static preparation candidate ID is not canonical")
    if profile != preparation.STABLE_DEFAULT_PROFILE:
        _fail(
            "invalid-configuration-profile",
            "static preparation must use stable-default configuration",
        )
    return candidate_id, profile


def replay_static_effective_config_v1_fd(
    root_fd: int,
    *,
    endpoint_path: str,
    startup_artifact_path: str,
    session_path: str,
    used_paths: set[str] | None = None,
) -> ReplayedStaticEffectiveConfig:
    """Join terminal static intent to one held-FD C02 config bridge.

    Candidate identity and configuration profile come only from the completed
    static preparation. The three config-bridge paths name existing evidence;
    a future fixed-topology writer must supply them only from its own constants.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "static effective config evidence root",
        )
    )
    initial_session = _preparation(
        lambda: preparation.verify_rollback_evidence_preparation_fd(root_fd)
    )
    candidate_id, profile = _session_identity(initial_session)
    bindings = _static_bindings(initial_session)
    replayed_paths = set() if used_paths is None else used_paths
    _reserve_static_bindings(bindings, replayed_paths)
    static_document = _read_static_configuration(root_fd, bindings.configuration)
    expected_config, expected_sha256 = _parse_static_effective_config(
        static_document,
        candidate_id=candidate_id,
        configuration_profile=profile,
    )
    bridge = _bridge(
        lambda: config_bridge.replay_config_bridge_v1_fd(
            root_fd,
            candidate_id=candidate_id,
            configuration_profile=profile,
            endpoint_path=endpoint_path,
            startup_artifact_path=startup_artifact_path,
            session_path=session_path,
            used_paths=replayed_paths,
        )
    )
    if bridge.effective_config != expected_config:
        _fail(
            "effective-config-mismatch",
            "captured /v1/config effective config differs from static intent",
        )
    if bridge.effective_config_sha256 != expected_sha256:
        _fail(
            "effective-config-hash-mismatch",
            "captured /v1/config effective config hash differs from static intent",
        )

    terminal_session = _preparation(
        lambda: preparation.verify_rollback_evidence_preparation_fd(root_fd)
    )
    if terminal_session != initial_session or _static_bindings(terminal_session) != bindings:
        _fail(
            "static-preparation-replay-drift",
            "static preparation changed during config bridge replay",
        )
    terminal_bridge = _bridge(
        lambda: config_bridge.replay_config_bridge_v1_fd(
            root_fd,
            candidate_id=candidate_id,
            configuration_profile=profile,
            endpoint_path=endpoint_path,
            startup_artifact_path=startup_artifact_path,
            session_path=session_path,
            used_paths=set(
                {
                    bindings.reconstructed_baseline.path,
                    bindings.freeze.path,
                    bindings.base_release_candidate_report.path,
                    bindings.configuration.path,
                }
            ),
        )
    )
    if terminal_bridge != bridge:
        _fail(
            "config-bridge-replay-drift",
            "config bridge changed during static effective config replay",
        )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "static effective config evidence root",
        )
    )
    return ReplayedStaticEffectiveConfig(
        candidate_id=candidate_id,
        configuration_profile=profile,
        static_bindings=bindings,
        expected_effective_config=expected_config,
        expected_effective_config_sha256=expected_sha256,
        config_bridge=bridge,
    )
