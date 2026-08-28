#!/usr/bin/env python3
"""Create one completed raw-only C02 native-fallback provenance-v5 manifest.

This closed terminal binder consumes exactly one capture-v2
``exact-backend-fallback`` session.  It never starts a service, opens a
socket, invokes GPU tooling, or decides qualification.  Every descriptor is
derived by replaying held-FD evidence, including the source-written audit and
native-fallback event pairs and the effective ``gpu-greedy`` runtime setting.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import bind_raw_c02_soak_v4 as v4
import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common


BIND_REQUEST_VERSION = "riley.soak-v2-bind-request.v5"
MAX_BIND_REQUEST_BYTES = v4.MAX_BIND_REQUEST_BYTES

# Preserve the established raw-binder error domain while deliberately keeping
# all v5 publication and replay entry points distinct from the retained v4
# binder.  The v4 module supplies only pure validation/publishing primitives.
RawSoakBindError = v4.RawSoakBindError


def _fail(code: str, message: str) -> None:
    v4._fail(code, message)  # noqa: SLF001 - shared closed raw-binder domain


def _scenario_manifest_rows_v5(
    root_fd: int,
    value: Any,
    *,
    capture: checker.ReplayedFallbackScenarioCapture,
    configuration_target: checker.ObservedTarget,
    declared_paths: set[str],
    replayed_paths: set[str],
) -> list[dict[str, Any]]:
    """Derive the sole terminal scenario from the closed capture-v2 replay."""

    if not isinstance(value, list) or len(value) != 1 or len(capture.scenarios) != 1:
        _fail(
            "scenario-capture-inventory-mismatch",
            "v5 bind request must preserve exactly one native fallback scenario",
        )
    captured = capture.scenarios[0]
    row = v4._exact(  # noqa: SLF001 - shared closed input-shape helper
        value[0],
        {"scenario_id", "observation_session_path"},
        "soak v5 bind request.scenarios[0]",
    )
    if (
        row["scenario_id"] != checker.FALLBACK_SCENARIO_ID
        or captured.scenario_id != checker.FALLBACK_SCENARIO_ID
    ):
        _fail(
            "scenario-capture-inventory-mismatch",
            "v5 bind request must preserve exact-backend-fallback",
        )
    observation_path = v4._reserve_direct_session_path(  # noqa: SLF001
        row["observation_session_path"],
        "soak v5 bind request.scenarios[0].observation_session_path",
        declared_paths,
    )
    observation_descriptor, _raw = v4._descriptor_from_path(  # noqa: SLF001
        root_fd,
        observation_path,
        "soak v5 bind request.scenarios[0].observation_session",
    )
    v4._reserve_replayed_descriptor(  # noqa: SLF001
        observation_descriptor,
        "soak v5 bind request.scenarios[0].observation_session",
        replayed_paths,
    )
    observed = v4._checker(  # noqa: SLF001
        lambda: checker._load_session(  # noqa: SLF001 - pure held-FD replay primitive
            root_fd,
            observation_descriptor,
            "soak v5 bind request.scenarios[0].observation_session",
            replayed_paths,
        )
    )
    if not checker._capture_matches_observed(captured.target, observed):  # noqa: SLF001
        _fail(
            "scenario-capture-observation-target-mismatch",
            "v5 native fallback observation does not share the capture PID/start-tick/listener tuple",
        )
    if observed != configuration_target:
        _fail(
            "configuration-scenario-target-mismatch",
            "v5 native fallback observation does not share the configuration bridge PID/start-tick/listener/GPU tuple",
        )
    return [
        {
            "scenario_id": checker.FALLBACK_SCENARIO_ID,
            "target": observed.target.as_json(),
            "observation_session": observation_descriptor.as_json(),
            "request_ledger": captured.request_ledger.as_json(),
            "runtime_event_log": captured.runtime_event_log.as_json(),
            "generation_audit_index": captured.generation_audit_index.as_json(),
            "fallback_event_log": captured.fallback_event_log.as_json(),
        }
    ]


def _manifest_from_request(
    root_fd: int,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    request_path, _request_raw, request = v4._read_canonical_object(  # noqa: SLF001
        root_fd,
        bind_request_path,
        "soak v5 bind request",
        maximum_bytes=MAX_BIND_REQUEST_BYTES,
    )
    row = v4._exact(  # noqa: SLF001
        request,
        {
            "schema_version", "candidate_id", "binding_inputs", "configuration_evidence",
            "scenario_capture_session_path", "scenarios",
        },
        "soak v5 bind request",
    )
    if row["schema_version"] != BIND_REQUEST_VERSION:
        _fail(
            "unsupported-bind-request-version",
            f"soak v5 bind request must use {BIND_REQUEST_VERSION}",
        )
    candidate_id = v4._candidate_id(  # noqa: SLF001
        row["candidate_id"], "soak v5 bind request.candidate_id"
    )
    binding_inputs = v4._binding_inputs(row["binding_inputs"])  # noqa: SLF001
    if binding_inputs["configuration_profile"] != checker.MAX_PERFORMANCE_EXACT_PROFILE:
        _fail(
            "fallback-profile-mismatch",
            "soak v5 bind request requires max-performance-exact",
        )
    marker_name = f"{manifest_name}.complete"
    declared_paths = {request_path}
    replayed_paths = {request_path}
    configuration_evidence, runtime_identity, configuration_target = v4._configuration_evidence(  # noqa: SLF001
        root_fd,
        row["configuration_evidence"],
        candidate_id=candidate_id,
        binding_inputs=binding_inputs,
        declared_paths=declared_paths,
        replayed_paths=replayed_paths,
        require_gpu_greedy=True,
    )
    capture_path = v4._reserve_direct_session_path(  # noqa: SLF001
        row["scenario_capture_session_path"],
        "soak v5 bind request.scenario_capture_session_path",
        declared_paths,
    )
    capture_descriptor, _capture_raw = v4._descriptor_from_path(  # noqa: SLF001
        root_fd,
        capture_path,
        "soak v5 native fallback scenario capture session",
    )
    capture = v4._checker(  # noqa: SLF001
        lambda: checker.replay_raw_scenario_capture_v2_fd(
            root_fd,
            capture_descriptor,
            candidate_id=candidate_id,
            configuration_profile=binding_inputs["configuration_profile"],
            configuration_sha256=runtime_identity["configuration_sha256"],
            used_paths=replayed_paths,
        )
    )
    if not checker._capture_matches_observed(capture.target, configuration_target):  # noqa: SLF001
        _fail(
            "configuration-scenario-capture-target-mismatch",
            "native fallback capture does not share the configuration bridge PID/start-tick/listener tuple",
        )
    scenarios = _scenario_manifest_rows_v5(
        root_fd,
        row["scenarios"],
        capture=capture,
        configuration_target=configuration_target,
        declared_paths=declared_paths,
        replayed_paths=replayed_paths,
    )
    if manifest_name in declared_paths or marker_name in declared_paths:
        _fail(
            "output-name-collision",
            "manifest or completion-marker name collides with a declared input path",
        )
    return {
        "schema_version": checker.SOAK_V5_MANIFEST_VERSION,
        "capture_status": "captured",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "bindings": {
            "freeze_sha256": binding_inputs["freeze_sha256"],
            "base_release_candidate_report_sha256": binding_inputs[
                "base_release_candidate_report_sha256"
            ],
            "configuration_profile": binding_inputs["configuration_profile"],
            # This remains the runtime identity SHA, not effective_config_sha256.
            "configuration_sha256": runtime_identity["configuration_sha256"],
        },
        "configuration_evidence": configuration_evidence,
        "scenario_capture_session": capture_descriptor.as_json(),
        "scenario_contract": capture.contract.as_json(),
        "scenarios": scenarios,
    }


def bind_raw_soak_manifest_fd(
    root_fd: int,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    """Publish one v5 terminal manifest with a durable paired marker.

    The manifest is only created after full config/capture/audit/fallback and
    observation replay succeeds.  A visible marker after an unsynced final
    hardlink remains ambiguous and cannot be reported as terminal success.
    """

    name = v4._manifest_name(manifest_name)  # noqa: SLF001
    v4._common(  # noqa: SLF001
        lambda: common.require_private_evidence_directory_fd(
            root_fd, "v5 raw soak evidence root"
        )
    )
    v4._lock_terminal_output_pair(root_fd)  # noqa: SLF001
    try:
        v4._assert_terminal_output_pair_absent(root_fd, name)  # noqa: SLF001
        manifest = _manifest_from_request(root_fd, bind_request_path, name)
        created = v4._common(  # noqa: SLF001
            lambda: common.write_create_only_json(
                root_fd,
                name,
                manifest,
                "v5 soak raw manifest",
            )
        )
        v4._checker(lambda: checker.verify_soak_provenance_v5_fd(root_fd, name))  # noqa: SLF001
        marker = {
            "schema_version": checker.SOAK_V5_COMPLETION_MARKER_VERSION,
            "artifact_filename": name,
            "artifact_sha256": created.sha256,
        }
        intent_name = f"{name}.intent"
        v4._common(  # noqa: SLF001
            lambda: common.write_create_only_json(
                root_fd,
                intent_name,
                marker,
                "v5 soak raw manifest completion marker intent",
            )
        )
        try:
            v4._common(  # noqa: SLF001
                lambda: common.publish_create_only_hardlink(
                    root_fd,
                    intent_name,
                    f"{name}.complete",
                    "v5 soak raw manifest completion marker",
                )
            )
        except RawSoakBindError:
            if v4._completion_marker_pair_is_visible(root_fd, name):  # noqa: SLF001
                _fail(
                    "ambiguous-terminal-publication",
                    "completion marker became visible but its final directory sync failed; "
                    "no lifecycle success receipt may be emitted",
                )
            raise
        return v4._checker(  # noqa: SLF001
            lambda: checker.verify_completed_soak_provenance_v5_fd(root_fd, name)
        )
    finally:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def bind_raw_soak_manifest(
    evidence_root: Path,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    """Open one private root and bind a v5 native-fallback raw manifest."""

    v4._assert_external_to_source_checkout(evidence_root)  # noqa: SLF001
    root_fd = v4._common(  # noqa: SLF001
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
