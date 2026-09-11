#!/usr/bin/env python3
"""Write one closed, nonoperational C02 native-fallback bind request v5.

This runner-side helper does not contact a service, inspect a GPU, or make a
qualification decision.  It pins one private evidence root, independently
replays one max-performance-exact config bridge, one source-owned native
fallback capture-v2, and one immediate fallback observation before it writes
the only v5 bind-request shape accepted by the separate raw v5 binder.

It is deliberately not a lifecycle receipt writer or a terminal binder.  In
particular, it emits no completion marker and cannot reinterpret an existing
v5 completion pair as lifecycle authority.

The config-bridge diagnostic is deliberately supplied as an external,
absolute regular file.  It must be the exact stdout byte stream emitted by
``check_c02_config_bridge_v1.py`` for the held-FD replay; parsing equivalent
JSON is not sufficient.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

import bind_raw_c02_soak_v5 as v5_binder
import check_c02_config_bridge_v1 as config_bridge
import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common


CONFIG_ENDPOINT_PATH = "config-bridge/raw/config-endpoint.json"
STARTUP_ARTIFACT_PATH = "startup-artifact.json"
CONFIG_BRIDGE_SESSION_PATH = "config-bridge/session.json"
FALLBACK_CAPTURE_SESSION_PATH = "fallback-capture/session.json"
FALLBACK_OBSERVATION_SESSION_PATH = "fallback-observation/session.json"

MAX_STDOUT_REPORT_BYTES = common.DEFAULT_MAX_JSON_BYTES + 1
MAX_OUTPUT_NAME_BYTES = checker.MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES
OUTPUT_NAME_RE = checker.SOAK_TERMINAL_MANIFEST_NAME_RE


class C02LifecycleV5BindRequestError(ValueError):
    """The fixed C02 native-fallback v5 bind request cannot be written safely."""


def _fail(code: str, message: str) -> NoReturn:
    error = C02LifecycleV5BindRequestError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _config_bridge(call: Any) -> Any:
    try:
        return call()
    except config_bridge.ConfigBridgeReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-config-bridge"), str(error))


def _checker(call: Any) -> Any:
    try:
        return call()
    except checker.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-raw-provenance"), str(error))


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
    if value != checker.MAX_PERFORMANCE_EXACT_PROFILE:
        _fail(
            "invalid-configuration-profile",
            f"{label} must be {checker.MAX_PERFORMANCE_EXACT_PROFILE}",
        )
    return checker.MAX_PERFORMANCE_EXACT_PROFILE


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or common.SHA256_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail("invalid-sha256", f"{label} must be a non-zero lowercase SHA-256")
    return value


def _output_name(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) > MAX_OUTPUT_NAME_BYTES
        or OUTPUT_NAME_RE.fullmatch(value) is None
        or "/" in value
    ):
        _fail(
            "invalid-output-name",
            "--output-name must be a nonhidden root direct-child .json name of at most "
            f"{MAX_OUTPUT_NAME_BYTES} bytes",
        )
    if value == STARTUP_ARTIFACT_PATH:
        _fail(
            "output-name-input-collision",
            "--output-name must not reuse the fixed startup-artifact input leaf",
        )
    return value


def _path(value: Any, label: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError:
        _fail("invalid-absolute-path", f"{label} must be an absolute path")
    if type(raw) is not str:
        _fail("invalid-absolute-path", f"{label} must be an absolute path")
    if (
        not os.path.isabs(raw)
        or "\x00" in raw
        or "\\" in raw
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
    ):
        _fail("invalid-absolute-path", f"{label} must be a normalized absolute path")
    return Path(raw)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    """Reject the source checkout as a terminal evidence location before open."""

    source_root = Path(__file__).resolve().parents[2]
    try:
        evidence_root.relative_to(source_root)
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be external to the source checkout",
    )


def _require_evidence_root_fd_matches_path(root_fd: int, evidence_root: Path) -> None:
    """Keep the external-report boundary tied to the actual held root FD."""

    try:
        named = os.stat(evidence_root, follow_symlinks=False)
        opened = os.fstat(root_fd)
    except OSError as error:
        _fail("evidence-root-fd-path-unavailable", f"cannot compare evidence root path and FD: {error}")
    if (
        not stat.S_ISDIR(named.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        _fail(
            "evidence-root-fd-path-mismatch",
            "held evidence root FD does not match the supplied evidence-root path",
        )


def _read_external_config_bridge_stdout(
    bridge_report_path: Path,
    evidence_root: Path,
) -> bytes:
    """Read one non-symlink external stdout file through a held parent FD."""

    source_root = Path(__file__).resolve().parents[2]
    if _is_within(bridge_report_path, evidence_root) or _is_within(
        bridge_report_path, source_root
    ):
        _fail(
            "bridge-report-not-external",
            "--bridge-report must be external to both the evidence root and source checkout",
        )
    raw = _common(
        lambda: common.read_bounded_regular_path(
            bridge_report_path,
            "config bridge stdout report",
            maximum_bytes=MAX_STDOUT_REPORT_BYTES,
        )
    )
    if not raw.endswith(b"\n"):
        _fail(
            "noncanonical-config-bridge-stdout",
            "config bridge stdout report must end in exactly one stdout newline",
        )
    _common(
        lambda: common.parse_canonical_json(
            raw[:-1],
            "config bridge stdout report",
            maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
        )
    )
    return raw


def _require_exact_config_bridge_stdout(
    raw: bytes,
    replayed: config_bridge.ReplayedConfigBridge,
) -> None:
    expected = common.canonical_json_bytes(replayed.report()) + b"\n"
    if raw != expected:
        _fail(
            "config-bridge-report-mismatch",
            "config bridge stdout report does not exactly match the held-FD replay",
        )


def _fallback_capture_descriptor(root_fd: int) -> common.EvidenceDescriptor:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            FALLBACK_CAPTURE_SESSION_PATH,
            "native fallback capture session",
            maximum_bytes=checker.MAX_RAW_BYTES,
        )
    )
    if not raw:
        _fail("empty-evidence-leaf", "native fallback capture session must be nonempty")
    return _common(
        lambda: common.descriptor_for_bytes(
            FALLBACK_CAPTURE_SESSION_PATH,
            raw,
            "native fallback capture session",
        )
    )


def _require_gpu_greedy_config(
    root_fd: int,
    replayed_bridge: config_bridge.ReplayedConfigBridge,
) -> None:
    """Bind the v5-only backend fact to the exact bridge endpoint bytes."""

    endpoint_raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            CONFIG_ENDPOINT_PATH,
            "native fallback config endpoint",
            maximum_bytes=runtime_config.MAX_JSON_BYTES,
        )
    )
    endpoint_descriptor = _common(
        lambda: common.descriptor_for_bytes(
            CONFIG_ENDPOINT_PATH,
            endpoint_raw,
            "native fallback config endpoint",
        )
    )
    if endpoint_descriptor != replayed_bridge.endpoint:
        _fail(
            "configuration-endpoint-descriptor-drift",
            "native fallback config endpoint changed after config-bridge replay",
        )
    _document, endpoint = _runtime_config(
        lambda: runtime_config.validate_endpoint_bytes(
            endpoint_raw,
            "native fallback config endpoint",
        )
    )
    if endpoint.effective_config["sampling_backend"] != "gpu-greedy":
        _fail(
            "effective-sampling-backend-mismatch",
            "native fallback v5 bind request requires effective_config.sampling_backend gpu-greedy",
        )


def _fallback_observation_descriptor(root_fd: int) -> common.EvidenceDescriptor:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            FALLBACK_OBSERVATION_SESSION_PATH,
            "native fallback observation session",
            maximum_bytes=checker.MAX_RAW_BYTES,
        )
    )
    if not raw:
        _fail("empty-evidence-leaf", "native fallback observation session must be nonempty")
    return _common(
        lambda: common.descriptor_for_bytes(
            FALLBACK_OBSERVATION_SESSION_PATH,
            raw,
            "native fallback observation session",
        )
    )


def _write_c02_lifecycle_bind_request_v5_fd(
    root_fd: int,
    *,
    evidence_root: Path,
    bridge_report_path: Path,
    candidate_id: str,
    configuration_profile: str,
    freeze_sha256: str,
    base_release_candidate_report_sha256: str,
    output_name: str,
) -> dict[str, Any]:
    """Replay fixed v5 inputs and create one nonterminal request root leaf."""

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "C02 lifecycle evidence root",
        )
    )
    _require_evidence_root_fd_matches_path(root_fd, evidence_root)
    candidate = _candidate_id(candidate_id, "expected candidate ID")
    profile = _configuration_profile(
        configuration_profile,
        "expected configuration profile",
    )
    freeze = _sha256(freeze_sha256, "freeze SHA-256")
    base_report = _sha256(
        base_release_candidate_report_sha256,
        "base release-candidate report SHA-256",
    )
    name = _output_name(output_name)

    replayed_paths: set[str] = set()
    replayed_bridge = _config_bridge(
        lambda: config_bridge.replay_config_bridge_v1_fd(
            root_fd,
            candidate_id=candidate,
            configuration_profile=profile,
            endpoint_path=CONFIG_ENDPOINT_PATH,
            startup_artifact_path=STARTUP_ARTIFACT_PATH,
            session_path=CONFIG_BRIDGE_SESSION_PATH,
            used_paths=replayed_paths,
        )
    )
    report_raw = _read_external_config_bridge_stdout(bridge_report_path, evidence_root)
    _require_exact_config_bridge_stdout(report_raw, replayed_bridge)

    _require_gpu_greedy_config(root_fd, replayed_bridge)

    capture = _checker(
        lambda: checker.replay_raw_scenario_capture_v2_fd(
            root_fd,
            _fallback_capture_descriptor(root_fd),
            candidate_id=replayed_bridge.candidate_id,
            configuration_profile=replayed_bridge.configuration_profile,
            configuration_sha256=replayed_bridge.configuration_sha256,
            used_paths=replayed_paths,
        )
    )
    if not checker._capture_matches_observed(  # noqa: SLF001 - pure replay join
        capture.target,
        replayed_bridge.target,
    ):
        _fail(
            "configuration-scenario-capture-target-mismatch",
            "native fallback capture does not share the config bridge PID/start-tick/listener tuple",
        )
    if (
        len(capture.scenarios) != 1
        or capture.scenarios[0].scenario_id != checker.FALLBACK_SCENARIO_ID
    ):
        _fail(
            "lifecycle-scenario-inventory-mismatch",
            "native fallback v5 bind requests require exactly one exact-backend-fallback scenario",
        )
    observation = _fallback_observation_descriptor(root_fd)
    _checker(
        lambda: checker._reserve(  # noqa: SLF001 - shared strict path inventory primitive
            observation,
            label="native fallback observation session",
            used_paths=replayed_paths,
        )
    )
    observed_target = _checker(
        lambda: checker._load_session(  # noqa: SLF001 - strict held-FD observation replay
            root_fd,
            observation,
            "native fallback observation session",
            replayed_paths,
        )
    )
    if observed_target != replayed_bridge.target:
        _fail(
            "configuration-scenario-target-mismatch",
            "native fallback observation does not share the config bridge PID/start-tick/listener/GPU tuple",
        )

    request = {
        "schema_version": v5_binder.BIND_REQUEST_VERSION,
        "candidate_id": replayed_bridge.candidate_id,
        "binding_inputs": {
            "freeze_sha256": freeze,
            "base_release_candidate_report_sha256": base_report,
            "configuration_profile": replayed_bridge.configuration_profile,
        },
        "configuration_evidence": {
            "endpoint_path": CONFIG_ENDPOINT_PATH,
            "startup_artifact_path": STARTUP_ARTIFACT_PATH,
            "endpoint_observation_path": CONFIG_BRIDGE_SESSION_PATH,
        },
        "scenario_capture_session_path": FALLBACK_CAPTURE_SESSION_PATH,
        "scenarios": [
            {
                "scenario_id": checker.FALLBACK_SCENARIO_ID,
                "observation_session_path": FALLBACK_OBSERVATION_SESSION_PATH,
            }
        ],
    }
    _common(
        lambda: common.write_create_only_json(
            root_fd,
            name,
            request,
            "C02 lifecycle native-fallback v5 bind request",
        )
    )
    return request


def write_c02_lifecycle_bind_request_v5(
    evidence_root: Path,
    *,
    bridge_report_path: Path,
    candidate_id: str,
    configuration_profile: str,
    freeze_sha256: str,
    base_release_candidate_report_sha256: str,
    output_name: str,
) -> dict[str, Any]:
    """Open one private evidence root and create its fixed v5 request leaf."""

    root_path = _path(evidence_root, "--evidence-root")
    _assert_external_to_source_checkout(root_path)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(
            root_path,
            "--evidence-root",
        )
    )
    try:
        return _write_c02_lifecycle_bind_request_v5_fd(
            root_fd,
            evidence_root=root_path,
            bridge_report_path=_path(bridge_report_path, "--bridge-report"),
            candidate_id=candidate_id,
            configuration_profile=configuration_profile,
            freeze_sha256=freeze_sha256,
            base_release_candidate_report_sha256=base_release_candidate_report_sha256,
            output_name=output_name,
        )
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--bridge-report", required=True, type=Path)
    parser.add_argument("--candidate-id", "--expected-candidate-id", dest="candidate_id", required=True)
    parser.add_argument(
        "--configuration-profile",
        "--expected-configuration-profile",
        dest="configuration_profile",
        required=True,
    )
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--base-release-candidate-report-sha256", required=True)
    parser.add_argument("--output-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = write_c02_lifecycle_bind_request_v5(
            args.evidence_root,
            bridge_report_path=args.bridge_report,
            candidate_id=args.candidate_id,
            configuration_profile=args.configuration_profile,
            freeze_sha256=args.freeze_sha256,
            base_release_candidate_report_sha256=args.base_release_candidate_report_sha256,
            output_name=args.output_name,
        )
    except (C02LifecycleV5BindRequestError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.write(common.canonical_json_bytes(request).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
