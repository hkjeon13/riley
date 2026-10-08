#!/usr/bin/env python3
"""CPU-only hostile tests for the closed native-fallback bind-request v5 writer."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bind_raw_c02_soak_v5 as binder
import capture_c02_raw_soak_scenarios_v1 as capture
import check_c02_config_bridge_v1 as config_bridge
import check_c02_provenance_v2 as checker
import provenance_v2_common as common
import test_bind_raw_c02_soak_v5 as v5_fixtures
import write_c02_lifecycle_bind_request_v5 as writer


class WriteC02LifecycleBindRequestV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._activate()

    def tearDown(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        external = getattr(self, "external", None)
        if external is not None:
            external.cleanup()
            del self.external
        fixture = getattr(self, "v5_fixture", None)
        if fixture is not None:
            fixture.tearDown()
            del self.v5_fixture

    def _activate(self) -> None:
        self._cleanup()
        self.v5_fixture = v5_fixtures.BindRawC02SoakV5Tests(methodName="runTest")
        self.v5_fixture.setUp()
        self.fixture = self.v5_fixture.fixture
        self.root = self.fixture.root
        self.tree = self.fixture.tree

        configuration = self.fixture._configuration_evidence()
        self.tree.put(
            writer.CONFIG_ENDPOINT_PATH,
            (self.root / configuration["endpoint_path"]).read_bytes(),
        )
        self.tree.put(
            writer.STARTUP_ARTIFACT_PATH,
            (self.root / configuration["startup_artifact_path"]).read_bytes(),
        )
        self.v5_fixture._fallback_capture()
        self.fixture._observation_session("fallback-observation")

        self.external = tempfile.TemporaryDirectory()
        self.external_path = Path(self.external.name).resolve()
        self.bridge_report = self.external_path / "config-bridge.stdout"
        self._write_exact_bridge_report()

    def _write_exact_bridge_report(self) -> None:
        replayed = config_bridge.replay_config_bridge_v1(
            self.root,
            candidate_id=self.fixture.candidate_id,
            configuration_profile=self.fixture.profile,
            endpoint_path=writer.CONFIG_ENDPOINT_PATH,
            startup_artifact_path=writer.STARTUP_ARTIFACT_PATH,
            session_path=writer.CONFIG_BRIDGE_SESSION_PATH,
        )
        self.bridge_report.write_bytes(common.canonical_json_bytes(replayed.report()) + b"\n")

    def _copy_configuration_inputs(self) -> None:
        self.tree.put(
            writer.CONFIG_ENDPOINT_PATH,
            (self.root / "config/endpoint.json").read_bytes(),
        )
        self.tree.put(
            writer.STARTUP_ARTIFACT_PATH,
            (self.root / "config/startup.json").read_bytes(),
        )

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _write(self, **overrides: object) -> dict:
        values: dict[str, object] = {
            "bridge_report_path": self.bridge_report,
            "candidate_id": self.fixture.candidate_id,
            "configuration_profile": checker.MAX_PERFORMANCE_EXACT_PROFILE,
            "freeze_sha256": "a" * 64,
            "base_release_candidate_report_sha256": "b" * 64,
            "output_name": "lifecycle-v5-bind-request.json",
        }
        values.update(overrides)
        return writer.write_c02_lifecycle_bind_request_v5(self.root, **values)  # type: ignore[arg-type]

    def test_writes_closed_fallback_request_consumed_by_v5_binder(self) -> None:
        request = self._write()
        self.assertEqual(request["schema_version"], binder.BIND_REQUEST_VERSION)
        self.assertEqual(request["candidate_id"], self.fixture.candidate_id)
        self.assertEqual(
            request["configuration_evidence"],
            {
                "endpoint_path": writer.CONFIG_ENDPOINT_PATH,
                "startup_artifact_path": writer.STARTUP_ARTIFACT_PATH,
                "endpoint_observation_path": writer.CONFIG_BRIDGE_SESSION_PATH,
            },
        )
        self.assertEqual(request["scenario_capture_session_path"], writer.FALLBACK_CAPTURE_SESSION_PATH)
        self.assertEqual(
            request["scenarios"],
            [
                {
                    "scenario_id": checker.FALLBACK_SCENARIO_ID,
                    "observation_session_path": writer.FALLBACK_OBSERVATION_SESSION_PATH,
                }
            ],
        )
        raw = (self.root / "lifecycle-v5-bind-request.json").read_bytes()
        self.assertEqual(raw, common.canonical_json_bytes(request))
        self.assertEqual(os.stat(self.root / "lifecycle-v5-bind-request.json").st_mode & 0o777, 0o600)

        report = binder.bind_raw_soak_manifest(
            self.root,
            "lifecycle-v5-bind-request.json",
            "bound-v5.json",
        )
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")

    def test_refuses_bridge_stdout_mismatch_and_non_gpu_config_before_publication(self) -> None:
        report = json.loads(self.bridge_report.read_text(encoding="utf-8"))
        report["reason_codes"] = ["caller-authored"]
        self.bridge_report.write_bytes(common.canonical_json_bytes(report) + b"\n")
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "config-bridge-report-mismatch")
        self.assertFalse((self.root / "lifecycle-v5-bind-request.json").exists())

        self._activate()
        self.v5_fixture._set_effective_sampling_backend("cpu")
        self._copy_configuration_inputs()
        self._write_exact_bridge_report()
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "effective-sampling-backend-mismatch")
        self.assertFalse((self.root / "lifecycle-v5-bind-request.json").exists())

    def test_refuses_v1_or_incomplete_fallback_capture_before_publication(self) -> None:
        session_path = self.root / writer.FALLBACK_CAPTURE_SESSION_PATH
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["schema_version"] = capture.CAPTURE_VERSION
        self.tree.put(writer.FALLBACK_CAPTURE_SESSION_PATH, session)
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "historical-fallback-scenario-capture-version-rejected")
        self.assertFalse((self.root / "lifecycle-v5-bind-request.json").exists())

        self._activate()
        self.tree.put(
            "fallback-capture/capture-incomplete.json",
            {
                "schema_version": capture.FALLBACK_INCOMPLETE_MARKER_VERSION,
                "capture_name": "fallback-capture",
            },
        )
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "incomplete-capture")
        self.assertFalse((self.root / "lifecycle-v5-bind-request.json").exists())

    def test_refuses_rebound_source_audit_or_fallback_marker_before_publication(self) -> None:
        self.v5_fixture._rewrite_fallback_event(
            lambda event: event["generation_audit"].update({"artifact_sha256": "e" * 64})
        )
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "source-fallback-audit-binding-mismatch")
        self.assertFalse((self.root / "lifecycle-v5-bind-request.json").exists())

        self._activate()
        self.v5_fixture._rewrite_fallback_marker(
            lambda marker: marker.update({"artifact_sha256": "f" * 64})
        )
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "source-fallback-marker-mismatch")
        self.assertFalse((self.root / "lifecycle-v5-bind-request.json").exists())

    def test_refuses_mismatched_fallback_observation_before_publication(self) -> None:
        self.fixture._observation_session(
            "fallback-observation",
            gpu_uuid="GPU-ffffffff-ffff-ffff-ffff-ffffffffffff",
        )
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "configuration-scenario-target-mismatch")
        self.assertFalse((self.root / "lifecycle-v5-bind-request.json").exists())

    def test_requires_external_nonlinked_stdout_and_create_only_output(self) -> None:
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write(bridge_report_path=Path("relative-config-bridge.stdout"))
        self.assert_reason(raised, "invalid-absolute-path")

        inside_root = self.root / "bridge-report.json"
        inside_root.write_bytes(self.bridge_report.read_bytes())
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write(bridge_report_path=inside_root)
        self.assert_reason(raised, "bridge-report-not-external")

        linked = self.external_path / "linked.stdout"
        linked.symlink_to(self.bridge_report)
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write(bridge_report_path=linked)
        self.assert_reason(raised, "unsafe-evidence-path")

        self._write()
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "create-only-collision")
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write(output_name=writer.STARTUP_ARTIFACT_PATH)
        self.assert_reason(raised, "output-name-input-collision")

    def test_private_fd_helper_rejects_a_path_for_another_evidence_root(self) -> None:
        other_root = self.external_path / "other-evidence-root"
        other_root.mkdir(mode=0o700)
        other_root.chmod(0o700)
        root_fd = common.open_private_evidence_directory(self.root, "test root")
        try:
            with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
                writer._write_c02_lifecycle_bind_request_v5_fd(  # noqa: SLF001
                    root_fd,
                    evidence_root=other_root,
                    bridge_report_path=self.bridge_report,
                    candidate_id=self.fixture.candidate_id,
                    configuration_profile=checker.MAX_PERFORMANCE_EXACT_PROFILE,
                    freeze_sha256="a" * 64,
                    base_release_candidate_report_sha256="b" * 64,
                    output_name="mismatched-root-request.json",
                )
        finally:
            os.close(root_fd)
        self.assert_reason(raised, "evidence-root-fd-path-mismatch")
        self.assertFalse((self.root / "mismatched-root-request.json").exists())

    def test_rejects_nonfallback_profile_without_creating_a_request(self) -> None:
        with self.assertRaises(writer.C02LifecycleV5BindRequestError) as raised:
            self._write(configuration_profile=checker.STABLE_DEFAULT_PROFILE)
        self.assert_reason(raised, "invalid-configuration-profile")
        self.assertFalse((self.root / "lifecycle-v5-bind-request.json").exists())

    def test_cli_is_closed_and_writer_never_binds_or_publishes_a_terminal_marker(self) -> None:
        source = Path(writer.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "bind_raw_soak_manifest(",
            "publish_create_only_hardlink",
            "--configuration-sha256",
            "--target",
        ):
            self.assertNotIn(forbidden, source)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            exit_code = writer.main(
                [
                    "--evidence-root",
                    str(self.root),
                    "--bridge-report",
                    str(self.bridge_report),
                    "--expected-candidate-id",
                    self.fixture.candidate_id,
                    "--expected-configuration-profile",
                    checker.MAX_PERFORMANCE_EXACT_PROFILE,
                    "--freeze-sha256",
                    "a" * 64,
                    "--base-release-candidate-report-sha256",
                    "b" * 64,
                    "--output-name",
                    "cli-lifecycle-v5-bind-request.json",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            (self.root / "cli-lifecycle-v5-bind-request.json").read_text(encoding="utf-8") + "\n",
        )


if __name__ == "__main__":
    unittest.main()
