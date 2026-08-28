#!/usr/bin/env python3
"""CPU-only hostile tests for the narrow C02 lifecycle-supervisor receipt v1."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import check_c02_provenance_v2 as checker
import provenance_v2_common as common
import test_bind_raw_c02_soak_v4 as v4_fixtures
import write_c02_lifecycle_supervisor_receipt_v1 as receipt


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class LifecycleSupervisorReceiptV1Tests(unittest.TestCase):
    """Reuse the v4 fixture so no device, endpoint, or server is involved."""

    def setUp(self) -> None:
        self.fixture = v4_fixtures.BindRawC02SoakV4Tests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.tree = self.fixture.tree

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _shutdown(self, *, start_ticks: int | None = None, marker_sha256: str | None = None) -> None:
        ticks = self.fixture.ticks if start_ticks is None else start_ticks
        artifact = {
            "schema_version": checker.SHUTDOWN_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "server_pid": self.fixture.pid,
            "server_start_ticks": ticks,
            "worker_ready": False,
            "final_metrics": v4_fixtures._metrics(),
        }
        raw = self.tree.put(receipt.DEFAULT_SHUTDOWN_ARTIFACT_PATH, artifact)
        self.tree.put(
            receipt.DEFAULT_SHUTDOWN_MARKER_PATH,
            {
                "schema_version": checker.SHUTDOWN_MARKER_VERSION,
                "artifact_filename": "shutdown.json",
                "artifact_sha256": (
                    hashlib.sha256(raw).hexdigest()
                    if marker_sha256 is None
                    else marker_sha256
                ),
            },
        )

    def _publish(
        self,
        *,
        scenario_ids: tuple[str, ...] = ("smoke",),
        manifest_name: str = "soak-v4.json",
        receipt_name: str = "lifecycle-receipt.json",
    ) -> dict:
        request_path, _request = self.fixture._request(scenario_ids=scenario_ids)
        return receipt.write_lifecycle_supervisor_receipt_v1(
            self.root,
            bind_request_path=request_path,
            v4_manifest_name=manifest_name,
            receipt_name=receipt_name,
        )

    def test_publishes_completed_raw_receipt_bound_to_all_lifecycle_evidence(self) -> None:
        self._shutdown()
        with mock.patch.object(
            checker,
            "verify_c02_shutdown_v2_fd",
            wraps=checker.verify_c02_shutdown_v2_fd,
        ) as verify_shutdown:
            document = self._publish()

        self.assertEqual(document["schema_version"], receipt.RECEIPT_VERSION)
        self.assertEqual(document["status"], "completed")
        self.assertEqual(document["qualification_status"], "not-run")
        self.assertEqual(document["reason_codes"], [])
        self.assertEqual(document["scenario_evidence"]["scenario_id"], "smoke")
        self.assertEqual(
            document["shutdown_evidence"]["artifact"]["path"],
            receipt.DEFAULT_SHUTDOWN_ARTIFACT_PATH,
        )
        self.assertEqual(
            document["shutdown_evidence"]["completion_marker"]["path"],
            receipt.DEFAULT_SHUTDOWN_MARKER_PATH,
        )
        self.assertGreaterEqual(verify_shutdown.call_count, 1)
        for call in verify_shutdown.call_args_list:
            self.assertEqual(call.args[1:3], (
                receipt.DEFAULT_SHUTDOWN_ARTIFACT_PATH,
                receipt.DEFAULT_SHUTDOWN_MARKER_PATH,
            ))
            self.assertIsInstance(call.args[3], checker.TargetTuple)

        persisted = common.parse_canonical_json(
            (self.root / "lifecycle-receipt.json").read_bytes(),
            "lifecycle receipt",
        )
        self.assertEqual(persisted, document)
        manifest = common.parse_canonical_json(
            (self.root / "soak-v4.json").read_bytes(),
            "v4 manifest",
        )
        self.assertEqual(document["raw_manifest"], common.descriptor_for_bytes(
            "soak-v4.json", (self.root / "soak-v4.json").read_bytes(), "v4 manifest"
        ).as_json())
        self.assertEqual(
            document["configuration_evidence"], manifest["configuration_evidence"]
        )
        self.assertEqual(
            document["scenario_evidence"]["capture_session"],
            manifest["scenario_capture_session"],
        )
        self.assertEqual(
            document["scenario_evidence"]["contract"],
            manifest["scenario_contract"],
        )
        self.assertEqual(
            document["observation_evidence"]["session"],
            manifest["scenarios"][0]["observation_session"],
        )

        final = os.lstat(self.root / "lifecycle-receipt.json.complete")
        intent = os.lstat(self.root / "lifecycle-receipt.json.intent")
        self.assertEqual((final.st_dev, final.st_ino), (intent.st_dev, intent.st_ino))
        self.assertEqual(final.st_nlink, 2)
        self.assertEqual(intent.st_nlink, 2)
        self.assertEqual(
            receipt.verify_lifecycle_supervisor_receipt_v1(
                self.root,
                "lifecycle-receipt.json",
            ),
            document,
        )

    def test_cli_emits_canonical_receipt_only_after_the_same_process_v4_bind(self) -> None:
        self._shutdown()
        request_path, _request = self.fixture._request()
        stdout = _CapturedStdout()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            exit_code = receipt.main(
                [
                    "--evidence-root",
                    str(self.root),
                    "--bind-request",
                    request_path,
                    "--v4-manifest-name",
                    "cli-v4.json",
                    "--receipt-name",
                    "cli-lifecycle.json",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        document = receipt.verify_completed_lifecycle_supervisor_receipt_v1(
            self.root,
            "cli-lifecycle.json",
        )
        self.assertEqual(stdout.buffer.getvalue(), common.canonical_json_bytes(document) + b"\n")

    def test_refuses_multiple_v4_scenarios_before_receipt_publication(self) -> None:
        self._shutdown()
        with self.assertRaises(receipt.LifecycleSupervisorReceiptError) as raised:
            self._publish(scenario_ids=("first", "second"))
        self.assert_reason(raised, "lifecycle-scenario-count")
        # The raw v4 artifact is valid evidence, but it cannot become the
        # initial lifecycle authority because it contains two scenarios.
        self.assertTrue((self.root / "soak-v4.json").is_file())
        self.assertFalse((self.root / "lifecycle-receipt.json").exists())
        self.assertFalse((self.root / "lifecycle-receipt.json.intent").exists())
        self.assertFalse((self.root / "lifecycle-receipt.json.complete").exists())

    def test_invalid_shutdown_cannot_create_a_receipt(self) -> None:
        self._shutdown(start_ticks=self.fixture.ticks + 1)
        with self.assertRaises(receipt.LifecycleSupervisorReceiptError) as raised:
            self._publish()
        self.assert_reason(raised, "shutdown-target-mismatch")
        self.assertTrue((self.root / "soak-v4.json").is_file())
        self.assertFalse((self.root / "lifecycle-receipt.json").exists())

    def test_ambiguous_v4_publication_never_continues_to_a_receipt(self) -> None:
        self._shutdown()
        request_path, _request = self.fixture._request()
        original = common._fsync_checked

        def fail_v4_final_parent(descriptor: int, label: str) -> None:
            if label == "soak raw manifest completion marker parent directory":
                error = common.ProvenanceV2Error("fixture v4 final marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_v4_final_parent):
            with self.assertRaises(receipt.LifecycleSupervisorReceiptError) as raised:
                receipt.publish_lifecycle_supervisor_receipt_v1(
                    self.root,
                    bind_request_path=request_path,
                    v4_manifest_name="ambiguous-v4.json",
                    receipt_name="ambiguous-lifecycle.json",
                )
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / "ambiguous-v4.json.complete").is_file())
        self.assertTrue((self.root / "ambiguous-v4.json.intent").is_file())
        self.assertFalse((self.root / "ambiguous-lifecycle.json").exists())
        self.assertFalse((self.root / "ambiguous-lifecycle.json.complete").exists())

    def test_receipt_output_collision_is_fail_closed_before_v4_publication(self) -> None:
        self._shutdown()
        request_path, _request = self.fixture._request()
        with self.assertRaises(receipt.LifecycleSupervisorReceiptError) as raised:
            receipt.publish_lifecycle_supervisor_receipt_v1(
                self.root,
                bind_request_path=request_path,
                v4_manifest_name="same-name.json",
                receipt_name="same-name.json",
            )
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / "same-name.json").exists())
        self.assertFalse((self.root / "same-name.json.complete").exists())
        self.assertFalse((self.root / "same-name.json.intent").exists())

    def test_preexisting_receipt_output_refuses_terminal_publication(self) -> None:
        self._shutdown()
        request_path, _request = self.fixture._request()
        self.tree.put("occupied-lifecycle.json", {"stale": True})
        with self.assertRaises(receipt.LifecycleSupervisorReceiptError) as raised:
            receipt.write_lifecycle_supervisor_receipt_v1(
                self.root,
                bind_request_path=request_path,
                v4_manifest_name="occupied-v4.json",
                receipt_name="occupied-lifecycle.json",
            )
        self.assert_reason(raised, "output-name-collision")
        self.assertTrue((self.root / "occupied-v4.json").is_file())
        self.assertFalse((self.root / "occupied-lifecycle.json.complete").exists())
        self.assertFalse((self.root / "occupied-lifecycle.json.intent").exists())

    def test_receipt_verifier_replays_not_copies_shutdown_and_marker_pair(self) -> None:
        self._shutdown()
        self._publish()
        receipt_path = self.root / "lifecycle-receipt.json"
        document = json.loads(receipt_path.read_text(encoding="utf-8"))
        document["shutdown_evidence"]["artifact"]["byte_length"] += 1
        receipt_path.write_bytes(common.canonical_json_bytes(document))
        with self.assertRaises(receipt.LifecycleSupervisorReceiptError) as raised:
            receipt.verify_completed_lifecycle_supervisor_receipt_v1(
                self.root,
                "lifecycle-receipt.json",
            )
        self.assert_reason(raised, "lifecycle-receipt-shutdown-mismatch")

    def test_receipt_marker_directory_sync_failure_is_ambiguous_not_successful(self) -> None:
        self._shutdown()
        request_path, _request = self.fixture._request()
        original = common._fsync_checked

        def fail_receipt_final_parent(descriptor: int, label: str) -> None:
            if label == "lifecycle supervisor receipt completion marker parent directory":
                error = common.ProvenanceV2Error("fixture receipt marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_receipt_final_parent):
            with self.assertRaises(receipt.LifecycleSupervisorReceiptError) as raised:
                receipt.publish_lifecycle_supervisor_receipt_v1(
                    self.root,
                    bind_request_path=request_path,
                    v4_manifest_name="receipt-marker-v4.json",
                    receipt_name="receipt-marker-lifecycle.json",
                )
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / "receipt-marker-lifecycle.json").is_file())
        self.assertTrue((self.root / "receipt-marker-lifecycle.json.complete").is_file())
        self.assertTrue((self.root / "receipt-marker-lifecycle.json.intent").is_file())

    def test_schema_is_closed_and_helper_has_no_operational_imports(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/c02-lifecycle-supervisor-receipt-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        receipt_schema = schema["$defs"]["receipt"]
        self.assertEqual(
            receipt_schema["properties"]["schema_version"]["const"],
            receipt.RECEIPT_VERSION,
        )
        self.assertEqual(receipt_schema["properties"]["status"]["const"], "completed")
        self.assertEqual(
            receipt_schema["properties"]["qualification_status"]["const"],
            "not-run",
        )
        self.assertFalse(receipt_schema["additionalProperties"])
        self.assertEqual(
            schema["$defs"]["shutdownEvidence"]["properties"]["artifact"]["allOf"][1]
            ["properties"]["path"]["const"],
            receipt.DEFAULT_SHUTDOWN_ARTIFACT_PATH,
        )
        source = Path(receipt.__file__).read_text(encoding="utf-8")
        for forbidden in ("import socket", "import subprocess", "nvidia-smi", "docker", "podman", "ssh "):
            self.assertNotIn(forbidden, source)

    def test_help_is_available_without_a_lifecycle_capture(self) -> None:
        with mock.patch.object(sys, "stdout", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                receipt.main(["--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
