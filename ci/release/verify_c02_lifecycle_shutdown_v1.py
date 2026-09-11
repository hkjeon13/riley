#!/usr/bin/env python3
"""Replay the source-owned shutdown-v2 pair for one C02 lifecycle run.

This controller is intentionally pure: it derives the expected process/GPU
tuple from the already-captured config bridge through the private evidence-root
FD, then validates the source shutdown artifact and marker against that tuple.
It cannot start a service, query a GPU, publish a manifest, or issue a
qualification verdict.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

import check_c02_config_bridge_v1 as bridge
import check_c02_provenance_v2 as checker
import provenance_v2_common as common


ENDPOINT_PATH = "config-bridge/raw/config-endpoint.json"
STARTUP_ARTIFACT_PATH = "startup-artifact.json"
SESSION_PATH = "config-bridge/session.json"
SHUTDOWN_ARTIFACT_PATH = "source-audit/shutdown.json"
SHUTDOWN_MARKER_PATH = "source-audit/shutdown.json.complete"
REPORT_VERSION = "riley.c02-lifecycle-shutdown-check.v1"


class LifecycleShutdownVerificationError(ValueError):
    """A lifecycle shutdown cannot be tied to its replayed config bridge."""


def _fail(code: str, message: str) -> NoReturn:
    error = LifecycleShutdownVerificationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _bridge(call: Any) -> Any:
    try:
        return call()
    except bridge.ConfigBridgeReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-config-bridge"), str(error))


def _checker(call: Any) -> Any:
    try:
        return call()
    except checker.C02ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-shutdown"), str(error))


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    try:
        evidence_root.relative_to(source_root)
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be external to the source checkout",
    )


def verify_lifecycle_shutdown(
    evidence_root: Path,
    *,
    candidate_id: str,
    configuration_profile: str,
) -> dict[str, Any]:
    _assert_external_to_source_checkout(evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(evidence_root, "--evidence-root")
    )
    try:
        replayed = _bridge(
            lambda: bridge.replay_config_bridge_v1_fd(
                root_fd,
                candidate_id=candidate_id,
                configuration_profile=configuration_profile,
                endpoint_path=ENDPOINT_PATH,
                startup_artifact_path=STARTUP_ARTIFACT_PATH,
                session_path=SESSION_PATH,
            )
        )
        shutdown = _checker(
            lambda: checker.verify_c02_shutdown_v2_fd(
                root_fd,
                SHUTDOWN_ARTIFACT_PATH,
                SHUTDOWN_MARKER_PATH,
                replayed.target.target,
            )
        )
        return {
            "schema_version": REPORT_VERSION,
            "status": "bound",
            "qualification_status": "not-run",
            "candidate_id": replayed.candidate_id,
            "runtime_identity": {
                "configuration_profile": replayed.configuration_profile,
                "configuration_sha256": replayed.configuration_sha256,
            },
            "target": replayed.target.as_json(),
            "shutdown_artifact": shutdown.artifact.as_json(),
            "shutdown_marker": shutdown.marker.as_json(),
            "reason_codes": [],
        }
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--configuration-profile", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_lifecycle_shutdown(
            args.evidence_root,
            candidate_id=args.candidate_id,
            configuration_profile=args.configuration_profile,
        )
    except (LifecycleShutdownVerificationError, OSError) as error:
        print(f"C02 lifecycle shutdown verification refused: {error}", file=sys.stderr)
        return 2
    print(common.canonical_json_bytes(report).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
