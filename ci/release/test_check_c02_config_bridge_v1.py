#!/usr/bin/env python3
"""CPU-only hostile tests for the strict C02 config-bridge replay CLI."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_c02_config_bridge_v1 as bridge
import check_c02_provenance_v2 as checker
import provenance_v2_common as common
import test_c02_provenance_v2 as provenance_fixtures


class CheckC02ConfigBridgeV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        os.chmod(self.root, 0o700)
        self.fixture = provenance_fixtures.C02ProvenanceV2Tests()
        self.tree = provenance_fixtures.EvidenceTree(self.root)
        self.evidence = self.fixture.configuration_evidence(
            self.tree,
            self.fixture.bindings,
            bridge_prefix="config-bridge",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _paths(self) -> dict[str, str]:
        return {
            "endpoint_path": self.evidence["endpoint"]["path"],
            "startup_artifact_path": self.evidence["startup_artifact"]["path"],
            "session_path": self.evidence["endpoint_observation"]["path"],
        }

    def _replay(self, **overrides: str) -> bridge.ReplayedConfigBridge:
        paths = {**self._paths(), **overrides}
        return bridge.replay_config_bridge_v1(
            self.root,
            candidate_id=self.fixture.candidate,
            configuration_profile=self.fixture.bindings["configuration_profile"],
            **paths,
        )

    def test_replays_direct_bridge_and_cli_emits_canonical_diagnostic_report(self) -> None:
        replayed = self._replay()
        report = replayed.report()
        self.assertEqual(report["schema_version"], bridge.REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(
            report["runtime_identity"],
            {
                "configuration_profile": self.fixture.bindings["configuration_profile"],
                "configuration_sha256": self.fixture.bindings["configuration_sha256"],
            },
        )
        endpoint_document = common.parse_canonical_json(
            (self.root / self._paths()["endpoint_path"]).read_bytes(),
            "fixture endpoint",
        )
        assert isinstance(endpoint_document, dict)
        self.assertEqual(replayed.effective_config, endpoint_document["effective_config"])
        self.assertEqual(
            replayed.effective_config_sha256,
            endpoint_document["effective_config_sha256"],
        )
        self.assertEqual(report["target"]["server_pid"], 1111)
        self.assertEqual(report["target"]["listener_port"], 8080)
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/c02-config-bridge-replay-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], bridge.REPORT_VERSION)
        self.assertEqual(set(schema["required"]), set(report))
        self.assertEqual(
            schema["properties"]["configuration_evidence"]["properties"]["endpoint_observation"]
            ["allOf"][1]["properties"]["path"]["pattern"],
            "^[A-Za-z0-9][A-Za-z0-9._-]*/session\\.json$",
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            exit_code = bridge.main(
                [
                    "--evidence-root",
                    str(self.root),
                    "--endpoint-path",
                    self._paths()["endpoint_path"],
                    "--startup-artifact-path",
                    self._paths()["startup_artifact_path"],
                    "--session-path",
                    self._paths()["session_path"],
                    "--expected-candidate-id",
                    self.fixture.candidate,
                    "--expected-configuration-profile",
                    self.fixture.bindings["configuration_profile"],
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            common.canonical_json_bytes(report).decode("utf-8") + "\n",
        )

    def test_rejects_nested_session_incomplete_capture_and_identity_drift(self) -> None:
        nested = self.root / "nested" / "bridge" / "session.json"
        nested.parent.mkdir(parents=True)
        nested.write_bytes((self.root / self._paths()["session_path"]).read_bytes())
        with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
            self._replay(session_path="nested/bridge/session.json")
        self.assert_reason(raised, "invalid-session-path")

        self.tree.put(
            "config-bridge/capture-incomplete.json",
            {"capture_status": "incomplete", "schema_version": "fixture"},
        )
        with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "incomplete-capture")

        with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
            bridge.replay_config_bridge_v1(
                self.root,
                candidate_id="riley-0.1.1-rc3",
                configuration_profile=self.fixture.bindings["configuration_profile"],
                **self._paths(),
            )
        self.assert_reason(raised, "runtime-config-candidate-mismatch")

        with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
            bridge.replay_config_bridge_v1(
                self.root,
                candidate_id=self.fixture.candidate,
                configuration_profile=checker.MAX_PERFORMANCE_EXACT_PROFILE,
                **self._paths(),
            )
        self.assert_reason(raised, "runtime-config-profile-mismatch")

    def test_rejects_aliases_unsafe_root_and_has_no_operational_imports(self) -> None:
        paths = self._paths()
        with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
            self._replay(startup_artifact_path=paths["endpoint_path"])
        self.assert_reason(raised, "duplicate-evidence-path")

        os.chmod(self.root, 0o755)
        try:
            with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
                self._replay()
            self.assert_reason(raised, "unsafe-evidence-root-mode")
        finally:
            os.chmod(self.root, 0o700)

        with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
            bridge.replay_config_bridge_v1(
                Path(bridge.__file__).resolve().parents[2],
                candidate_id=self.fixture.candidate,
                configuration_profile=self.fixture.bindings["configuration_profile"],
                **paths,
            )
        self.assert_reason(raised, "evidence-root-inside-source-checkout")

        source = Path(bridge.__file__).read_text(encoding="utf-8")
        for forbidden in ("import socket", "import subprocess", "nvidia-smi", "docker", "podman", "ssh "):
            self.assertNotIn(forbidden, source)

    def test_fd_api_respects_a_caller_shared_descriptor_reservation_set(self) -> None:
        root_fd = common.open_private_evidence_directory(self.root, "test evidence root")
        try:
            with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
                bridge.replay_config_bridge_v1_fd(
                    root_fd,
                    candidate_id=self.fixture.candidate,
                    configuration_profile=self.fixture.bindings["configuration_profile"],
                    used_paths={self._paths()["endpoint_path"]},
                    **self._paths(),
                )
        finally:
            os.close(root_fd)
        self.assert_reason(raised, "duplicate-evidence-path")

    def test_fd_api_rejects_a_nonprivate_root_even_without_path_opening(self) -> None:
        os.chmod(self.root, 0o755)
        root_fd = common.open_absolute_directory(self.root, "unsafe test evidence root")
        try:
            with self.assertRaises(bridge.ConfigBridgeReplayError) as raised:
                bridge.replay_config_bridge_v1_fd(
                    root_fd,
                    candidate_id=self.fixture.candidate,
                    configuration_profile=self.fixture.bindings["configuration_profile"],
                    **self._paths(),
                )
        finally:
            os.close(root_fd)
            os.chmod(self.root, 0o700)
        self.assert_reason(raised, "unsafe-evidence-root-mode")

    def test_help_is_available_without_an_evidence_capture(self) -> None:
        with mock.patch("sys.stdout", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                bridge.main(["--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
