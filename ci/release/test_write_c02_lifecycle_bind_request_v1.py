#!/usr/bin/env python3
"""Focused CPU-only tests for the fixed C02 lifecycle bind-request writer."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bind_raw_c02_soak_v4 as binder
import check_c02_config_bridge_v1 as config_bridge
import provenance_v2_common as common
import test_bind_raw_c02_soak_v4 as v4_fixtures
import write_c02_lifecycle_bind_request_v1 as writer


class WriteC02LifecycleBindRequestV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self._activate(("smoke",))

    def tearDown(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        external = getattr(self, "external", None)
        if external is not None:
            external.cleanup()
            del self.external
        fixture = getattr(self, "fixture", None)
        if fixture is not None:
            fixture.tearDown()
            del self.fixture

    def _activate(self, scenario_ids: tuple[str, ...]) -> None:
        self._cleanup()
        self.fixture = v4_fixtures.BindRawC02SoakV4Tests()
        self.fixture.setUp()
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
        self.fixture._serial_capture(scenario_ids)
        self.fixture._observation_session("observation")

        self.external = tempfile.TemporaryDirectory()
        # macOS commonly exposes this temporary directory through /var, a
        # symlink to /private/var.  The writer intentionally rejects symlink
        # ancestors, so use its physical absolute path for a valid fixture.
        self.external_path = Path(self.external.name).resolve()
        self.bridge_report = self.external_path / "config-bridge.stdout"
        replayed = config_bridge.replay_config_bridge_v1(
            self.root,
            candidate_id=self.fixture.candidate_id,
            configuration_profile=self.fixture.profile,
            endpoint_path=writer.CONFIG_ENDPOINT_PATH,
            startup_artifact_path=writer.STARTUP_ARTIFACT_PATH,
            session_path=writer.CONFIG_BRIDGE_SESSION_PATH,
        )
        self.bridge_report.write_bytes(common.canonical_json_bytes(replayed.report()) + b"\n")

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _write(self, **overrides: object) -> dict:
        values: dict[str, object] = {
            "bridge_report_path": self.bridge_report,
            "candidate_id": self.fixture.candidate_id,
            "configuration_profile": self.fixture.profile,
            "freeze_sha256": "a" * 64,
            "base_release_candidate_report_sha256": "b" * 64,
            "output_name": "lifecycle-bind-request.json",
        }
        values.update(overrides)
        return writer.write_c02_lifecycle_bind_request_v1(self.root, **values)  # type: ignore[arg-type]

    def test_writes_fixed_single_scenario_request_consumed_by_existing_v4_binder(self) -> None:
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
        self.assertEqual(
            request["scenario_capture_session_path"],
            writer.SERIAL_CAPTURE_SESSION_PATH,
        )
        self.assertEqual(
            request["scenarios"],
            [
                {
                    "scenario_id": "smoke",
                    "observation_session_path": writer.OBSERVATION_SESSION_PATH,
                }
            ],
        )
        raw = (self.root / "lifecycle-bind-request.json").read_bytes()
        self.assertEqual(raw, common.canonical_json_bytes(request))
        self.assertEqual(os.stat(self.root / "lifecycle-bind-request.json").st_mode & 0o777, 0o600)

        report = binder.bind_raw_soak_manifest(
            self.root,
            "lifecycle-bind-request.json",
            "bound-v4.json",
        )
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")

    def test_requires_exact_config_bridge_stdout_bytes_and_never_publishes_on_mismatch(self) -> None:
        report = json.loads(self.bridge_report.read_text(encoding="utf-8"))
        report["reason_codes"] = ["caller-authored"]
        self.bridge_report.write_bytes(common.canonical_json_bytes(report) + b"\n")
        with self.assertRaises(writer.C02LifecycleBindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "config-bridge-report-mismatch")
        self.assertFalse((self.root / "lifecycle-bind-request.json").exists())

    def test_requires_one_serial_scenario_and_never_publishes_on_inventory_drift(self) -> None:
        self._activate(("first", "second"))
        with self.assertRaises(writer.C02LifecycleBindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "lifecycle-scenario-inventory-mismatch")
        self.assertFalse((self.root / "lifecycle-bind-request.json").exists())

    def test_rejects_nonexternal_or_linked_stdout_and_preserves_create_only_output(self) -> None:
        with self.assertRaises(writer.C02LifecycleBindRequestError) as raised:
            self._write(bridge_report_path=Path("relative-config-bridge.stdout"))
        self.assert_reason(raised, "invalid-absolute-path")

        inside_root = self.root / "bridge-report.json"
        inside_root.write_bytes(self.bridge_report.read_bytes())
        with self.assertRaises(writer.C02LifecycleBindRequestError) as raised:
            self._write(bridge_report_path=inside_root)
        self.assert_reason(raised, "bridge-report-not-external")

        linked = self.external_path / "linked.stdout"
        linked.symlink_to(self.bridge_report)
        with self.assertRaises(writer.C02LifecycleBindRequestError) as raised:
            self._write(bridge_report_path=linked)
        self.assert_reason(raised, "unsafe-evidence-path")

        self._write()
        with self.assertRaises(writer.C02LifecycleBindRequestError) as raised:
            self._write()
        self.assert_reason(raised, "create-only-collision")
        with self.assertRaises(writer.C02LifecycleBindRequestError) as raised:
            self._write(output_name=writer.STARTUP_ARTIFACT_PATH)
        self.assert_reason(raised, "output-name-input-collision")

    def test_cli_has_only_closed_input_surface_and_no_operational_imports(self) -> None:
        source = Path(writer.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
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
                    self.fixture.profile,
                    "--freeze-sha256",
                    "a" * 64,
                    "--base-release-candidate-report-sha256",
                    "b" * 64,
                    "--output-name",
                    "cli-lifecycle-bind-request.json",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            (self.root / "cli-lifecycle-bind-request.json").read_text(encoding="utf-8") + "\n",
        )


if __name__ == "__main__":
    unittest.main()
