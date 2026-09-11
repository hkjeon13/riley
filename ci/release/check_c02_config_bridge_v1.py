#!/usr/bin/env python3
"""Replay one captured C02 `/v1/config` process bridge through held FDs.

This diagnostic helper neither captures evidence nor decides a qualification
result.  It opens one private evidence root, derives the exact endpoint,
startup-artifact, and direct config-bridge session descriptors from held-FD
bytes, then replays the existing raw proof.  The configuration SHA-256 and
observed process/listener/GPU tuple are derived from those bytes; callers may
not supply either fact.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common


REPORT_VERSION = "riley.c02-config-bridge-replay.v1"


class ConfigBridgeReplayError(ValueError):
    """A config-bridge replay request cannot establish raw provenance."""


def _fail(code: str, message: str) -> NoReturn:
    error = ConfigBridgeReplayError(message)
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
        _fail(getattr(error, "reason_code", "invalid-config-bridge"), str(error))


def _runtime_config(call: Any) -> Any:
    try:
        return call()
    except runtime_config.EffectiveRuntimeConfigError as error:
        _fail(getattr(error, "reason_code", "invalid-runtime-config"), str(error))


def _candidate_id(value: Any, label: str) -> str:
    if type(value) is not str or checker.CANDIDATE_RE.fullmatch(value) is None:
        _fail("invalid-candidate-id", f"{label} must be a canonical RC candidate ID")
    return value


def _configuration_profile(value: Any, label: str) -> str:
    if type(value) is not str or value not in checker.SOAK_CONFIGURATION_PROFILES:
        _fail(
            "invalid-configuration-profile",
            f"{label} must be one of {sorted(checker.SOAK_CONFIGURATION_PROFILES)}",
        )
    return value


def _path(value: Any, label: str) -> str:
    relative = _common(lambda: common.validate_relative_path(value, label))
    if len(relative) > checker.MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label} exceeds {checker.MAX_RELATIVE_PATH_BYTES} bytes",
        )
    return relative


def _descriptor_from_path(
    root_fd: int,
    path: str,
    label: str,
    *,
    maximum_bytes: int,
) -> tuple[common.EvidenceDescriptor, bytes]:
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
    return _common(lambda: common.descriptor_for_bytes(path, raw, label)), raw


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    """Reject the known source checkout as an evidence location before open."""

    source_root = Path(__file__).resolve().parents[2]
    try:
        evidence_root.relative_to(source_root)
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be external to the source checkout",
    )


@dataclass(frozen=True)
class ReplayedConfigBridge:
    """The raw configuration facts derived by one successful replay."""

    candidate_id: str
    configuration_profile: str
    configuration_sha256: str
    endpoint: common.EvidenceDescriptor
    startup_artifact: common.EvidenceDescriptor
    endpoint_observation: common.EvidenceDescriptor
    effective_config: dict[str, Any]
    effective_config_sha256: str
    target: checker.ObservedTarget

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_VERSION,
            "status": "bound",
            "qualification_status": "not-run",
            "candidate_id": self.candidate_id,
            "runtime_identity": {
                "configuration_profile": self.configuration_profile,
                "configuration_sha256": self.configuration_sha256,
            },
            "configuration_evidence": {
                "endpoint": self.endpoint.as_json(),
                "startup_artifact": self.startup_artifact.as_json(),
                "endpoint_observation": self.endpoint_observation.as_json(),
            },
            "target": self.target.as_json(),
            "reason_codes": [],
        }


def replay_config_bridge_v1_fd(
    root_fd: int,
    *,
    candidate_id: str,
    configuration_profile: str,
    endpoint_path: str,
    startup_artifact_path: str,
    session_path: str,
    used_paths: set[str] | None = None,
) -> ReplayedConfigBridge:
    """Derive one config bridge through a caller-held private root descriptor."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "config bridge evidence root",
        )
    )
    candidate = _candidate_id(candidate_id, "expected candidate ID")
    profile = _configuration_profile(
        configuration_profile,
        "expected configuration profile",
    )
    endpoint_relative = _path(endpoint_path, "endpoint path")
    startup_relative = _path(startup_artifact_path, "startup artifact path")
    session_relative = _path(session_path, "config bridge session path")
    if len({endpoint_relative, startup_relative, session_relative}) != 3:
        _fail(
            "duplicate-evidence-path",
            "endpoint, startup artifact, and config bridge session paths must differ",
        )
    endpoint_descriptor, endpoint_raw = _descriptor_from_path(
        root_fd,
        endpoint_relative,
        "config bridge endpoint",
        maximum_bytes=runtime_config.MAX_JSON_BYTES,
    )
    startup_descriptor, _startup_raw = _descriptor_from_path(
        root_fd,
        startup_relative,
        "config bridge startup artifact",
        maximum_bytes=runtime_config.MAX_JSON_BYTES,
    )
    session_descriptor, _session_raw = _descriptor_from_path(
        root_fd,
        session_relative,
        "config bridge observation session",
        maximum_bytes=checker.MAX_RAW_BYTES,
    )
    _checker(
        lambda: checker._require_direct_v4_session_path(  # noqa: SLF001 - v4 layout primitive
            session_descriptor,
            "config bridge session",
        )
    )
    _endpoint_document, endpoint = _runtime_config(
        lambda: runtime_config.validate_endpoint_bytes(endpoint_raw, "config bridge endpoint")
    )
    if endpoint.candidate_id != candidate:
        _fail(
            "runtime-config-candidate-mismatch",
            "config bridge endpoint candidate differs from the expected candidate",
        )
    identity = endpoint.runtime_identity
    if identity["configuration_profile"] != profile:
        _fail(
            "runtime-config-profile-mismatch",
            "config bridge endpoint profile differs from the expected profile",
        )
    configuration_sha256 = _checker(
        lambda: checker._sha256(  # noqa: SLF001 - canonical raw-provenance scalar parser
            identity["configuration_sha256"],
            "config bridge endpoint.runtime_identity.configuration_sha256",
        )
    )
    replayed_paths = set() if used_paths is None else used_paths
    target = _checker(
        lambda: checker._load_soak_configuration_evidence(  # noqa: SLF001 - pure held-FD replay
            root_fd,
            {
                "endpoint": endpoint_descriptor.as_json(),
                "startup_artifact": startup_descriptor.as_json(),
                "endpoint_observation": session_descriptor.as_json(),
            },
            candidate_id=candidate,
            bindings={
                "configuration_profile": profile,
                "configuration_sha256": configuration_sha256,
            },
            used_paths=replayed_paths,
            require_direct_observation_session=True,
        )
    )
    return ReplayedConfigBridge(
        candidate_id=candidate,
        configuration_profile=profile,
        configuration_sha256=configuration_sha256,
        endpoint=endpoint_descriptor,
        startup_artifact=startup_descriptor,
        endpoint_observation=session_descriptor,
        effective_config=endpoint.effective_config,
        effective_config_sha256=endpoint.effective_config_sha256,
        target=target,
    )


def replay_config_bridge_v1(
    evidence_root: Path,
    *,
    candidate_id: str,
    configuration_profile: str,
    endpoint_path: str,
    startup_artifact_path: str,
    session_path: str,
) -> ReplayedConfigBridge:
    """Open one private evidence root and replay its config bridge."""

    _assert_external_to_source_checkout(evidence_root)
    root_fd = _checker(
        lambda: checker._open_private_evidence_root(  # noqa: SLF001 - verifier root guard
            evidence_root,
            "config bridge evidence root",
        )
    )
    try:
        return replay_config_bridge_v1_fd(
            root_fd,
            candidate_id=candidate_id,
            configuration_profile=configuration_profile,
            endpoint_path=endpoint_path,
            startup_artifact_path=startup_artifact_path,
            session_path=session_path,
        )
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--endpoint-path", required=True)
    parser.add_argument("--startup-artifact-path", required=True)
    parser.add_argument("--session-path", required=True)
    parser.add_argument("--expected-candidate-id", required=True)
    parser.add_argument("--expected-configuration-profile", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        replayed = replay_config_bridge_v1(
            args.evidence_root,
            candidate_id=args.expected_candidate_id,
            configuration_profile=args.expected_configuration_profile,
            endpoint_path=args.endpoint_path,
            startup_artifact_path=args.startup_artifact_path,
            session_path=args.session_path,
        )
    except (ConfigBridgeReplayError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(common.canonical_json_bytes(replayed.report()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
