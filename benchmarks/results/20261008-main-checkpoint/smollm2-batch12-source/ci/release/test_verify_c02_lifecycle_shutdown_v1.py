#!/usr/bin/env python3
"""CPU-only tests for the fixed C02 lifecycle shutdown replay controller."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_c02_config_bridge_v1 as bridge
import check_c02_provenance_v2 as checker
import provenance_v2_common as common
import test_c02_provenance_v2 as provenance_fixtures
import verify_c02_lifecycle_shutdown_v1 as lifecycle


class VerifyC02LifecycleShutdownV1Tests(unittest.TestCase):
    """The controller must derive, rather than accept, its shutdown tuple."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        os.chmod(self.root, 0o700)
        self.fixture = provenance_fixtures.C02ProvenanceV2Tests()
        self.tree = provenance_fixtures.EvidenceTree(self.root)
        self.configuration = self.fixture.configuration_evidence(
            self.tree,
            self.fixture.bindings,
            bridge_prefix="config-bridge",
        )

        # The lifecycle controller intentionally has no caller-controlled
        # bridge paths.  Copy otherwise-valid fixture bytes into its fixed
        # locations while retaining the source-owned config-bridge raw leaves.
        self.tree.put(
            lifecycle.ENDPOINT_PATH,
            (self.root / self.configuration["endpoint"]["path"]).read_bytes(),
        )
        self.tree.put(
            lifecycle.STARTUP_ARTIFACT_PATH,
            (self.root / self.configuration["startup_artifact"]["path"]).read_bytes(),
        )
        self.shutdown = self.tree.put(
            lifecycle.SHUTDOWN_ARTIFACT_PATH,
            {
                "schema_version": checker.SHUTDOWN_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "server_pid": 1111,
                "server_start_ticks": 2222,
                "worker_ready": False,
                "final_metrics": provenance_fixtures.metrics(),
            },
        )
        self.marker = self.tree.put(
            lifecycle.SHUTDOWN_MARKER_PATH,
            {
                "schema_version": checker.SHUTDOWN_MARKER_VERSION,
                "artifact_filename": "shutdown.json",
                "artifact_sha256": self.shutdown.sha256,
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _verify(self) -> dict:
        return lifecycle.verify_lifecycle_shutdown(
            self.root,
            candidate_id=self.fixture.candidate,
            configuration_profile=self.fixture.bindings["configuration_profile"],
        )

    def test_replays_fixed_bridge_and_binds_completed_shutdown_pair(self) -> None:
        report = self._verify()
        replayed = bridge.replay_config_bridge_v1(
            self.root,
            candidate_id=self.fixture.candidate,
            configuration_profile=self.fixture.bindings["configuration_profile"],
            endpoint_path=lifecycle.ENDPOINT_PATH,
            startup_artifact_path=lifecycle.STARTUP_ARTIFACT_PATH,
            session_path=lifecycle.SESSION_PATH,
        )
        self.assertEqual(
            report,
            {
                "schema_version": lifecycle.REPORT_VERSION,
                "status": "bound",
                "qualification_status": "not-run",
                "candidate_id": self.fixture.candidate,
                "runtime_identity": {
                    "configuration_profile": self.fixture.bindings["configuration_profile"],
                    "configuration_sha256": self.fixture.bindings["configuration_sha256"],
                },
                "target": replayed.target.as_json(),
                "shutdown_artifact": self.shutdown.as_json(),
                "shutdown_marker": self.marker.as_json(),
                "reason_codes": [],
            },
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            exit_code = lifecycle.main(
                [
                    "--evidence-root",
                    str(self.root),
                    "--candidate-id",
                    self.fixture.candidate,
                    "--configuration-profile",
                    self.fixture.bindings["configuration_profile"],
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            common.canonical_json_bytes(report).decode("utf-8") + "\n",
        )

    def test_refuses_shutdown_target_drift_before_emitting_report(self) -> None:
        self.tree.put(
            lifecycle.SHUTDOWN_ARTIFACT_PATH,
            {
                "schema_version": checker.SHUTDOWN_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "server_pid": 9999,
                "server_start_ticks": 2222,
                "worker_ready": False,
                "final_metrics": provenance_fixtures.metrics(),
            },
        )
        # The completion marker deliberately remains bound to the original
        # bytes.  The controller must derive and compare the bridge target
        # before a success diagnostic can be emitted.
        with self.assertRaises(lifecycle.LifecycleShutdownVerificationError) as raised:
            self._verify()
        self.assert_reason(raised, "shutdown-target-mismatch")

    def test_refuses_source_checkout_roots_and_has_no_operational_imports(self) -> None:
        inside_source = Path(lifecycle.__file__).resolve().parents[2]
        with self.assertRaises(lifecycle.LifecycleShutdownVerificationError) as raised:
            lifecycle.verify_lifecycle_shutdown(
                inside_source,
                candidate_id=self.fixture.candidate,
                configuration_profile=self.fixture.bindings["configuration_profile"],
            )
        self.assert_reason(raised, "evidence-root-inside-source-checkout")

        source = Path(lifecycle.__file__).read_text(encoding="utf-8")
        for forbidden in ("import socket", "import subprocess", "nvidia-smi", "docker", "podman", "ssh "):
            self.assertNotIn(forbidden, source)

    def test_help_is_available_without_an_evidence_capture(self) -> None:
        with mock.patch("sys.stdout", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                lifecycle.main(["--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
