#!/usr/bin/env python3
"""Privately replay the fixed RC3 candidate/source evidence join.

This module is intentionally not a producer and has no path-based public
surface. Its sole callable entry is for a future authenticated compositor that
already owns one private evidence-root FD and its exclusive lock. It derives
candidate generation only from a completed source-owned serial capture, joins
that capture to a candidate host phase and source-owned shutdown pair, and
returns raw typed inputs. It never writes a bind request or terminal evidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NoReturn, TypeVar

import capture_rc3_rollback_phase_v1 as phase_capture
import check_c02_provenance_v2 as c02
import check_rc3_static_effective_config_v1 as static_effective
import provenance_v2_common as common


CANDIDATE_PHASE_CAPTURE_NAME = "candidate-phase"
SOURCE_CAPTURE_SESSION_PATH = "serial-capture/session.json"
SOURCE_AUDIT_DIRECTORY_NAME = "source-audit"
SHUTDOWN_ARTIFACT_PATH = f"{SOURCE_AUDIT_DIRECTORY_NAME}/shutdown.json"
SHUTDOWN_MARKER_PATH = f"{SHUTDOWN_ARTIFACT_PATH}.complete"
CONFIG_ENDPOINT_PATH = "config/endpoint.json"
CONFIG_STARTUP_ARTIFACT_PATH = "config/startup.json"
CONFIG_BRIDGE_SESSION_PATH = "config-bridge/session.json"


class CandidateSourceJoinError(ValueError):
    """Fixed candidate/source evidence cannot establish a raw join."""


def _fail(code: str, message: str) -> NoReturn:
    error = CandidateSourceJoinError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _c02(call: Callable[[], T]) -> T:
    try:
        return call()
    except c02.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-c02-provenance"), str(error))


def _phase(call: Callable[[], T]) -> T:
    try:
        return call()
    except phase_capture.RollbackPhaseCaptureError as error:
        _fail(getattr(error, "reason_code", "invalid-rollback-phase"), str(error))


def _static_effective(call: Callable[[], T]) -> T:
    try:
        return call()
    except static_effective.StaticEffectiveConfigError as error:
        _fail(getattr(error, "reason_code", "invalid-static-effective-config"), str(error))


@dataclass(frozen=True)
class CandidateGenerationInputs:
    """Exact source-owned completion exchange used for candidate generation."""

    request: common.EvidenceDescriptor
    response_head: common.EvidenceDescriptor
    response_body: common.EvidenceDescriptor
    generation_audit_index: common.EvidenceDescriptor


@dataclass(frozen=True)
class ReplayedCandidateSourceJoin:
    """Closed raw inputs for a later fixed-name rollback request writer."""

    static_effective: static_effective.ReplayedStaticEffectiveConfig
    candidate_phase: phase_capture.ReplayedPhaseCapture
    source_capture: c02.ReplayedScenarioCapture
    source_scenario: c02.ReplayedScenario
    generation: CandidateGenerationInputs
    shutdown: c02.VerifiedC02ShutdownV2


def _reserve(
    descriptor: common.EvidenceDescriptor,
    *,
    label: str,
    used_paths: set[str],
) -> None:
    if descriptor.path in used_paths:
        _fail(
            "duplicate-evidence-path",
            f"{label} reuses evidence path {descriptor.path!r}",
        )
    used_paths.add(descriptor.path)


def _fixed_descriptor(
    root_fd: int,
    path: str,
    label: str,
    *,
    maximum_bytes: int,
) -> common.EvidenceDescriptor:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            path,
            label,
            maximum_bytes=maximum_bytes,
        )
    )
    if not raw:
        _fail("empty-evidence-leaf", f"{label} must be nonempty")
    return _common(lambda: common.descriptor_for_bytes(path, raw, label))


def _reserve_phase(
    replayed: phase_capture.ReplayedPhaseCapture,
    used_paths: set[str],
) -> None:
    for group_name, group in (
        ("candidate phase process evidence", replayed.process_evidence),
        ("candidate phase health", replayed.health),
    ):
        for field in sorted(group):
            _reserve(
                group[field],
                label=f"{group_name}.{field}",
                used_paths=used_paths,
            )
    if replayed.generation is not None:
        _fail(
            "candidate-local-generation-forbidden",
            "candidate phase must not retain a local generation exchange",
        )


def _require_source_target(
    source: c02.ScenarioCaptureTarget,
    phase: phase_capture.TargetIdentity,
) -> None:
    if (
        source.pid != phase.server_pid
        or source.start_ticks != phase.server_start_ticks
        or source.listener_port != phase.listener_port
        or source.listener_inode != phase.listener_inode
    ):
        _fail(
            "candidate-source-target-mismatch",
            "source scenario PID/start-tick/listener tuple differs from candidate phase",
        )


def _require_bridge_target(
    bridge: c02.ObservedTarget,
    phase: phase_capture.TargetIdentity,
) -> None:
    target = bridge.target
    if (
        target.pid != phase.server_pid
        or target.start_ticks != phase.server_start_ticks
        or bridge.listener_port != phase.listener_port
        or bridge.listener_inode != phase.listener_inode
        or target.gpu_index != phase.gpu_index
        or target.gpu_uuid != phase.gpu_uuid
    ):
        _fail(
            "candidate-config-bridge-target-mismatch",
            "config bridge target differs from candidate phase target",
        )


def _replay_candidate_source_once(
    root_fd: int,
    used_paths: set[str],
) -> ReplayedCandidateSourceJoin:
    static = _static_effective(
        lambda: static_effective.replay_static_effective_config_v1_fd(
            root_fd,
            endpoint_path=CONFIG_ENDPOINT_PATH,
            startup_artifact_path=CONFIG_STARTUP_ARTIFACT_PATH,
            session_path=CONFIG_BRIDGE_SESSION_PATH,
            used_paths=used_paths,
        )
    )
    candidate_phase = _phase(
        lambda: phase_capture.replay_rc3_rollback_phase_v1_fd(
            root_fd,
            CANDIDATE_PHASE_CAPTURE_NAME,
        )
    )
    _reserve_phase(candidate_phase, used_paths)
    _require_bridge_target(static.config_bridge.target, candidate_phase.target)

    source_session = _fixed_descriptor(
        root_fd,
        SOURCE_CAPTURE_SESSION_PATH,
        "candidate source capture session",
        maximum_bytes=c02.MAX_RAW_BYTES,
    )
    source_capture = _c02(
        lambda: c02.replay_raw_scenario_capture_v1_fd(
            root_fd,
            source_session,
            candidate_id=static.candidate_id,
            configuration_profile=static.configuration_profile,
            configuration_sha256=static.runtime_configuration_sha256,
            used_paths=used_paths,
        )
    )
    if len(source_capture.scenarios) != 1:
        _fail(
            "candidate-source-scenario-count",
            "candidate source capture must contain exactly one scenario",
        )
    source_scenario = source_capture.scenarios[0]
    if (
        source_capture.audit_directory != SOURCE_AUDIT_DIRECTORY_NAME
        or source_scenario.audit_directory != SOURCE_AUDIT_DIRECTORY_NAME
    ):
        _fail(
            "candidate-source-audit-directory-mismatch",
            "candidate source capture must use the fixed source-audit directory",
        )
    _require_source_target(source_capture.target, candidate_phase.target)
    _require_source_target(source_scenario.target, candidate_phase.target)

    shutdown_artifact = _fixed_descriptor(
        root_fd,
        SHUTDOWN_ARTIFACT_PATH,
        "candidate source shutdown artifact",
        maximum_bytes=c02.MAX_RAW_BYTES,
    )
    shutdown_marker = _fixed_descriptor(
        root_fd,
        SHUTDOWN_MARKER_PATH,
        "candidate source shutdown completion marker",
        maximum_bytes=c02.MAX_RAW_BYTES,
    )
    shutdown = _c02(
        lambda: c02._verify_c02_shutdown_v2_descriptors_fd(  # noqa: SLF001 - fixed held-FD core
            root_fd,
            shutdown_artifact,
            shutdown_marker,
            c02.TargetTuple(
                pid=candidate_phase.target.server_pid,
                start_ticks=candidate_phase.target.server_start_ticks,
                gpu_index=candidate_phase.target.gpu_index,
                gpu_uuid=candidate_phase.target.gpu_uuid,
            ),
            "candidate source shutdown",
            used_paths,
        )
    )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "candidate source evidence root",
        )
    )
    return ReplayedCandidateSourceJoin(
        static_effective=static,
        candidate_phase=candidate_phase,
        source_capture=source_capture,
        source_scenario=source_scenario,
        generation=CandidateGenerationInputs(
            request=source_scenario.request,
            response_head=source_scenario.response_head,
            response_body=source_scenario.response_body,
            generation_audit_index=source_scenario.generation_audit_index,
        ),
        shutdown=shutdown,
    )


def _replay_candidate_source_join_on_held_root_fd(
    root_fd: int,
) -> ReplayedCandidateSourceJoin:
    """Replay the fixed candidate/source join while the caller retains root EX.

    There are no caller-supplied paths, candidate IDs, profiles, hashes,
    targets, scenarios, or descriptors. The final independent replay detects
    any drift across the complete join before a future writer can use it.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "candidate source evidence root",
        )
    )
    initial = _replay_candidate_source_once(root_fd, set())
    terminal = _replay_candidate_source_once(root_fd, set())
    if terminal != initial:
        _fail(
            "candidate-source-replay-drift",
            "candidate/source evidence changed during held-FD replay",
        )
    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "candidate source evidence root",
        )
    )
    return initial
