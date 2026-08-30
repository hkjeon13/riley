#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 Gate E soak adapter.

The shared Gate E fixture supplies the closed inventory and frozen-root
topology.  GPU-produced report semantics are mocked here; the tests exercise
the adapter's descriptor, anchor, scratch, and scope boundaries only.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent))

import provenance_v2_common as common  # noqa: E402
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs  # noqa: E402
import replay_rc3_gate_e_soak_v1 as soak  # noqa: E402
import test_rc3_gate_e_input_inventory_v1 as gate_inventory_tests  # noqa: E402


class Rc3GateESoakV1Tests(unittest.TestCase):
    """Reuse structural fixtures while isolating GPU-produced semantics."""

    def setUp(self) -> None:
        self.gate = gate_inventory_tests.Rc3GateEInputInventoryV1Tests(
            "test_closed_four_gate_inventory_replays_without_a_gate_e_decision"
        )
        self.gate.setUp()
        self._install_typed_leaves()

    def tearDown(self) -> None:
        self.gate.tearDown()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext,
        reason: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _descriptor(self, group: str, field: str) -> common.EvidenceDescriptor:
        record = self.gate.inventory[group]
        self.assertIsInstance(record, dict)
        return common.parse_descriptor(record[field], f"fixture {group}.{field}")

    def _write_leaf(self, group: str, field: str, raw: bytes) -> common.EvidenceDescriptor:
        record = self.gate.inventory[group]
        self.assertIsInstance(record, dict)
        previous = common.parse_descriptor(record[field], f"fixture {group}.{field}")
        (self.gate.gate_root / previous.path).write_bytes(raw)
        descriptor = common.descriptor_for_bytes(previous.path, raw, f"fixture {group}.{field}")
        record[field] = descriptor.as_json()
        self.gate._write_inventory()
        return descriptor

    def _release_image(self) -> str:
        request = self.gate.fixture.request
        release = request["release"]
        self.assertIsInstance(release, dict)
        container = release["container"]
        self.assertIsInstance(container, dict)
        value = container["image_digest"]
        self.assertIsInstance(value, str)
        return value

    def _golden_sha256(self) -> str:
        return self._descriptor("python_free", "correctness_golden").sha256

    def _install_typed_leaves(self) -> None:
        self._write_leaf("soak", "report", b"{}\n")
        self._write_leaf("python_free", "correctness_golden", b"{}\n")
        self._write_leaf("canonical_e0", "native_report", b"{}\n")

    def _replay(
        self,
        *,
        release_image: str | None = None,
        golden_sha256: str | None = None,
    ) -> dict[str, object]:
        return soak.replay_rc3_gate_e_soak_v1(
            self.gate.gate_root,
            frozen_candidate_root=self.gate.frozen_root,
            input_evidence_root=self.gate.fixture.evidence,
            repository_root=self.gate.fixture.root,
            expected_release_image_id=release_image or self._release_image(),
            expected_correctness_golden_sha256=golden_sha256 or self._golden_sha256(),
        )

    def _semantic_patches(self, side_effect: object | None = None):
        def replay(
            snapshot: soak._ScratchSnapshot,
            submitted_report: object,
            golden_raw: object,
            native_raw: object,
            **kwargs: object,
        ) -> dict[str, object]:
            if side_effect is not None:
                return side_effect(snapshot, submitted_report, kwargs)  # type: ignore[operator]
            self.assertTrue(snapshot.path.is_file())
            self.assertNotEqual(snapshot.path.parent, self.gate.gate_root)
            self.assertEqual(golden_raw, b"{}\n")
            self.assertEqual(native_raw, b"{}\n")
            self.assertEqual(
                snapshot.path.read_bytes(),
                (self.gate.gate_root / self._descriptor("soak", "raw_evidence").path).read_bytes(),
            )
            artifacts = kwargs["artifacts"]
            self.assertIsInstance(artifacts, soak._SoakArtifacts)
            return {
                "report": submitted_report,
                "raw_evidence_sha256": artifacts.raw_evidence.sha256,
                "raw_evidence_byte_length": artifacts.raw_evidence.byte_length,
                "correctness_golden_sha256": artifacts.correctness_golden.sha256,
                "native_correctness_report_sha256": artifacts.native_report.sha256,
                "raw_stream_member_byte_limit": soak.MAX_SOAK_RAW_STREAM_MEMBER_BYTES,
                "scratch_disk_byte_limit": soak.MAX_SOAK_RAW_SCRATCH_BYTES,
            }

        return (
            mock.patch.object(soak, "_replay_soak_raw", side_effect=replay),
            mock.patch.object(soak, "_require_frozen_model_binding", return_value=({}, "a" * 64)),
            mock.patch.object(soak, "_require_soak_report_bindings"),
        )

    def test_private_held_fd_snapshot_replays_only_soak_component(self) -> None:
        before = self.gate._snapshot()
        patches = self._semantic_patches()
        with self.gate._defaults(), patches[0], patches[1], patches[2]:
            result = self._replay()

        self.assertEqual(self.gate._snapshot(), before)
        self.assertEqual(result["schema_version"], soak.REPLAY_VERSION)
        self.assertEqual(result["scope"], soak.SCOPE)
        self.assertEqual(result["authority"], soak.AUTHORITY)
        self.assertEqual(result["status"], "bound")
        self.assertEqual(result["candidate_status"], "frozen")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["soak_status"], "passed")
        self.assertEqual(result["not_established"], soak.NOT_ESTABLISHED)
        self.assertNotIn("gate_e_status", result)
        self.assertNotIn("qualification", result)
        self.assertEqual(result["checks"], [{"name": name, "satisfied": True} for name in soak.CHECK_NAMES])
        details = result["soak"]
        self.assertIsInstance(details, dict)
        self.assertEqual(details["raw_stream_member_byte_limit"], soak.MAX_SOAK_RAW_STREAM_MEMBER_BYTES)
        self.assertEqual(details["scratch_disk_byte_limit"], soak.MAX_SOAK_RAW_SCRATCH_BYTES)
        self.assertEqual(details["correctness_golden"], self._descriptor("python_free", "correctness_golden").as_json())
        self.assertEqual(details["native_report"], self._descriptor("canonical_e0", "native_report").as_json())

    def test_component_caps_reject_before_full_gate_stream_replay(self) -> None:
        cases = (
            ("soak", "raw_evidence", soak.MAX_SOAK_RAW_ARCHIVE_BYTES),
            ("soak", "report", soak.MAX_SOAK_REPORT_BYTES),
            ("python_free", "correctness_golden", soak.MAX_CORRECTNESS_GOLDEN_BYTES),
            ("canonical_e0", "native_report", soak.MAX_NATIVE_CORRECTNESS_REPORT_BYTES),
        )
        for group, field, maximum in cases:
            with self.subTest(group=group, field=field):
                record = self.gate.inventory[group]
                self.assertIsInstance(record, dict)
                descriptor = record[field]
                self.assertIsInstance(descriptor, dict)
                original = descriptor["byte_length"]
                descriptor["byte_length"] = maximum + 1
                self.gate._write_inventory()
                try:
                    with self.gate._defaults(), mock.patch.object(
                        gate_inputs,
                        "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                        side_effect=AssertionError("full Gate E replay must not start"),
                    ) as structural, self.assertRaises(soak.SoakReplayError) as raised:
                        self._replay()
                    self.assert_reason(raised, "soak-input-too-large")
                    structural.assert_not_called()
                finally:
                    descriptor["byte_length"] = original
                    self.gate._write_inventory()

    def test_external_anchors_are_validated_before_semantic_replay(self) -> None:
        invalid = (
            ("sha256:" + "0" * 64, self._golden_sha256(), "invalid-expected-release-image-id"),
            ("not-an-image", self._golden_sha256(), "invalid-expected-release-image-id"),
            (self._release_image(), "0" * 64, "invalid-expected-correctness-golden-sha256"),
            (self._release_image(), "A" * 64, "invalid-expected-correctness-golden-sha256"),
        )
        for release_image, golden_sha256, reason in invalid:
            with self.subTest(release_image=release_image, golden_sha256=golden_sha256), mock.patch.object(
                gate_inputs,
                "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                side_effect=AssertionError("full Gate E replay must not start"),
            ) as structural, self.assertRaises(soak.SoakReplayError) as raised:
                self._replay(release_image=release_image, golden_sha256=golden_sha256)
            self.assert_reason(raised, reason)
            structural.assert_not_called()

    def test_mismatched_external_golden_anchor_is_rejected(self) -> None:
        patches = self._semantic_patches()
        with self.gate._defaults(), patches[0], patches[1], patches[2], self.assertRaises(
            soak.SoakReplayError
        ) as raised:
            self._replay(golden_sha256="a" * 64)
        self.assert_reason(raised, "soak-golden-anchor-mismatch")

    def test_private_scratch_replacement_after_raw_replay_is_rejected(self) -> None:
        def replace(
            snapshot: soak._ScratchSnapshot,
            submitted_report: object,
            kwargs: object,
        ) -> dict[str, object]:
            replacement = snapshot.path.with_name("soak-private-replacement.tmp")
            replacement.write_bytes(snapshot.path.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, snapshot.path)
            self.assertIsInstance(kwargs, dict)
            artifacts = kwargs["artifacts"]
            self.assertIsInstance(artifacts, soak._SoakArtifacts)
            return {
                "report": submitted_report,
                "raw_evidence_sha256": artifacts.raw_evidence.sha256,
                "raw_evidence_byte_length": artifacts.raw_evidence.byte_length,
                "correctness_golden_sha256": artifacts.correctness_golden.sha256,
                "native_correctness_report_sha256": artifacts.native_report.sha256,
                "raw_stream_member_byte_limit": soak.MAX_SOAK_RAW_STREAM_MEMBER_BYTES,
                "scratch_disk_byte_limit": soak.MAX_SOAK_RAW_SCRATCH_BYTES,
            }

        patches = self._semantic_patches(replace)
        with self.gate._defaults(), patches[0], patches[1], patches[2], self.assertRaises(
            soak.SoakReplayError
        ) as raised:
            self._replay()
        self.assert_reason(raised, "scratch-snapshot-mutated")

    def test_adapter_has_no_legacy_or_final_checker_dependency(self) -> None:
        source = inspect.getsource(soak)
        self.assertNotIn("check_release_candidate", source)
        self.assertNotIn("replay_raw_evidence_archive", source)
        self.assertNotIn("evaluate(", source)
        self.assertIn("validate_bound_reliability_soak_evidence", source)
        snapshot_source = inspect.getsource(soak._snapshot_soak_raw)
        self.assertIn("dir_fd=scratch_fd", snapshot_source)
        self.assertIn("os.O_EXCL", snapshot_source)
        self.assertIn("nofollow", snapshot_source)
        self.assertIn("_open_scratch_snapshot_fd", inspect.getsource(soak._replay_soak_raw))

    def test_contract_result_requires_every_held_input_binding(self) -> None:
        artifacts = soak._SoakArtifacts(
            report=common.EvidenceDescriptor("report", "1" * 64, 1),
            raw_evidence=common.EvidenceDescriptor("raw", "2" * 64, 2),
            correctness_golden=common.EvidenceDescriptor("golden", "3" * 64, 3),
            native_report=common.EvidenceDescriptor("native", "4" * 64, 4),
        )
        result = {
            "report": {},
            "raw_evidence_sha256": artifacts.raw_evidence.sha256,
            "raw_evidence_byte_length": artifacts.raw_evidence.byte_length,
            "correctness_golden_sha256": artifacts.correctness_golden.sha256,
            "native_correctness_report_sha256": artifacts.native_report.sha256,
            "raw_stream_member_byte_limit": soak.MAX_SOAK_RAW_STREAM_MEMBER_BYTES,
            "scratch_disk_byte_limit": soak.MAX_SOAK_RAW_SCRATCH_BYTES,
        }
        soak._require_replayed_result(result, {}, artifacts=artifacts)
        result["native_correctness_report_sha256"] = "0" * 64
        with self.assertRaises(soak.SoakReplayError) as raised:
            soak._require_replayed_result(result, {}, artifacts=artifacts)
        self.assert_reason(raised, "soak-raw-replay-binding-mismatch")

    def test_schema_reserves_component_only_soak_authority(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-gate-e-soak-semantic-replay-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            soak.REPLAY_VERSION,
        )
        self.assertEqual(schema["properties"]["scope"]["const"], soak.SCOPE)
        self.assertEqual(schema["properties"]["authority"]["const"], soak.AUTHORITY)
        self.assertEqual(schema["properties"]["soak_status"]["const"], "passed")
        self.assertNotIn("gate_e_status", schema["properties"])
        soak_schema = schema["properties"]["soak"]["properties"]
        self.assertEqual(
            soak_schema["raw_stream_member_byte_limit"]["const"],
            soak.MAX_SOAK_RAW_STREAM_MEMBER_BYTES,
        )
        self.assertEqual(
            soak_schema["scratch_disk_byte_limit"]["const"],
            soak.MAX_SOAK_RAW_SCRATCH_BYTES,
        )
        self.assertEqual(
            schema["$defs"]["notEstablished"]["properties"]["actual_capture"]["const"],
            "not-established",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
