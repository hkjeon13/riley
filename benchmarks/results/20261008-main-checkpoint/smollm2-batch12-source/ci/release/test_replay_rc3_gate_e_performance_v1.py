#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 Gate E performance adapter."""

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
import replay_rc3_gate_e_performance_v1 as performance  # noqa: E402
import test_rc3_gate_e_input_inventory_v1 as gate_inventory_tests  # noqa: E402


EXPECTED_OPTIMIZER_IMAGE = "sha256:" + "a" * 64


class Rc3GateEPerformanceV1Tests(unittest.TestCase):
    """Reuse the structural fixture while mocking only GPU-produced semantics."""

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

    def _write_leaf(
        self,
        group: str,
        field: str,
        raw: bytes,
    ) -> common.EvidenceDescriptor:
        record = self.gate.inventory[group]
        self.assertIsInstance(record, dict)
        previous = common.parse_descriptor(record[field], f"fixture {group}.{field}")
        (self.gate.gate_root / previous.path).write_bytes(raw)
        descriptor = common.descriptor_for_bytes(
            previous.path,
            raw,
            f"fixture {group}.{field}",
        )
        record[field] = descriptor.as_json()
        self.gate._write_inventory()
        return descriptor

    def _release_binding(self) -> common.EvidenceDescriptor:
        request = self.gate.fixture.request
        release = request["release"]
        self.assertIsInstance(release, dict)
        return common.parse_descriptor(release["elf"], "fixture frozen release ELF")

    def _release_image(self) -> str:
        request = self.gate.fixture.request
        release = request["release"]
        self.assertIsInstance(release, dict)
        container = release["container"]
        self.assertIsInstance(container, dict)
        value = container["image_digest"]
        self.assertIsInstance(value, str)
        return value

    def _model_tree(self) -> str:
        request = self.gate.fixture.request
        models = request["models"]
        self.assertIsInstance(models, list)
        model = models[0]
        self.assertIsInstance(model, dict)
        tree = common.parse_descriptor(model["tree"], "fixture frozen model tree")
        return tree.sha256

    def _install_typed_leaves(self) -> None:
        self._write_leaf("performance", "report", b"{}\n")
        self._write_leaf("canonical_e0", "optimizer_report", b"{}\n")
        release_elf = self._release_binding()
        release_bytes = (self.gate.fixture.evidence / release_elf.path).read_bytes()
        self._write_leaf("release", "native_candidate_executable", release_bytes)

    def _raw_result(self) -> dict[str, object]:
        return {
            "candidate": {"model": {}},
            "baseline": {
                "sha256": "3052b334bfb6370fc47b327566d8553cb7591ac23bbfa636e69ca99c893edf7c"
            },
            "raw_stream_member_byte_limit": performance.MAX_PERFORMANCE_RAW_STREAM_MEMBER_BYTES,
            "scratch_disk_byte_limit": performance.MAX_PERFORMANCE_RAW_SCRATCH_BYTES,
        }

    def _replay(self, *, release_image: str | None = None, optimizer_image: str | None = None) -> dict[str, object]:
        return performance.replay_rc3_gate_e_performance_v1(
            self.gate.gate_root,
            frozen_candidate_root=self.gate.frozen_root,
            input_evidence_root=self.gate.fixture.evidence,
            repository_root=self.gate.fixture.root,
            expected_release_image_id=release_image or self._release_image(),
            expected_optimizer_build_image_id=optimizer_image or EXPECTED_OPTIMIZER_IMAGE,
        )

    def _semantic_patches(self, side_effect: object | None = None):
        def replay(
            snapshot: performance._ScratchSnapshot,
            _report: object,
            _baseline: object,
            **_kwargs: object,
        ) -> dict[str, object]:
            if side_effect is not None:
                return side_effect(snapshot)  # type: ignore[operator]
            self.assertTrue(snapshot.path.is_file())
            self.assertNotEqual(snapshot.path.parent, self.gate.gate_root)
            self.assertEqual(
                snapshot.path.read_bytes(),
                (self.gate.gate_root / self._descriptor("performance", "raw_evidence").path).read_bytes(),
            )
            return self._raw_result()

        return (
            mock.patch.object(performance, "_replay_performance_raw", side_effect=replay),
            mock.patch.object(performance, "_validated_optimizer_model_tree", return_value=self._model_tree()),
            mock.patch.object(performance, "_require_frozen_model_binding"),
            mock.patch.object(performance, "_read_reviewed_baseline", return_value=b"{}\n"),
        )

    def test_private_held_fd_snapshot_replays_only_performance_component(self) -> None:
        before = self.gate._snapshot()
        patches = self._semantic_patches()
        with self.gate._defaults(), patches[0], patches[1], patches[2], patches[3]:
            result = self._replay()

        release_elf = self._release_binding()
        native = self._descriptor("release", "native_candidate_executable")
        self.assertEqual(self.gate._snapshot(), before)
        self.assertNotEqual(native.path, release_elf.path)
        self.assertEqual(native.sha256, release_elf.sha256)
        self.assertEqual(native.byte_length, release_elf.byte_length)
        self.assertEqual(result["schema_version"], performance.REPLAY_VERSION)
        self.assertEqual(result["scope"], performance.SCOPE)
        self.assertEqual(result["authority"], performance.AUTHORITY)
        self.assertEqual(result["status"], "bound")
        self.assertEqual(result["performance_status"], "passed")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["not_established"], performance.NOT_ESTABLISHED)
        self.assertNotIn("gate_e_status", result)
        self.assertNotIn("qualification", result)
        details = result["performance"]
        self.assertIsInstance(details, dict)
        self.assertEqual(
            details["reviewed_baseline_sha256"],
            "3052b334bfb6370fc47b327566d8553cb7591ac23bbfa636e69ca99c893edf7c",
        )
        self.assertEqual(
            details["raw_stream_member_byte_limit"],
            performance.MAX_PERFORMANCE_RAW_STREAM_MEMBER_BYTES,
        )
        self.assertEqual(
            details["scratch_disk_byte_limit"],
            performance.MAX_PERFORMANCE_RAW_SCRATCH_BYTES,
        )
        self.assertEqual(result["checks"], [
            {"name": name, "satisfied": True} for name in performance.CHECK_NAMES
        ])

    def test_component_caps_reject_before_full_gate_stream_replay(self) -> None:
        cases = (
            ("performance", "raw_evidence", performance.MAX_PERFORMANCE_RAW_ARCHIVE_BYTES),
            ("performance", "report", performance.MAX_PERFORMANCE_REPORT_BYTES),
            ("canonical_e0", "optimizer_report", performance.MAX_OPTIMIZER_REPORT_BYTES),
            ("release", "profile_binary", performance.MAX_PROFILE_BINARY_BYTES),
            ("release", "native_candidate_executable", performance.MAX_RELEASE_ELF_BYTES),
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
                    ) as structural, self.assertRaises(performance.PerformanceReplayError) as raised:
                        self._replay()
                    self.assert_reason(raised, "performance-input-too-large")
                    structural.assert_not_called()
                finally:
                    descriptor["byte_length"] = original
                    self.gate._write_inventory()

    def test_external_image_anchors_are_validated_before_semantic_replay(self) -> None:
        for release, optimizer in (
            ("sha256:" + "0" * 64, EXPECTED_OPTIMIZER_IMAGE),
            (self._release_image(), "sha256:" + "0" * 64),
            ("not-an-image", EXPECTED_OPTIMIZER_IMAGE),
            (self._release_image(), "sha256:" + "A" * 64),
        ):
            with self.subTest(release=release, optimizer=optimizer), mock.patch.object(
                gate_inputs,
                "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                side_effect=AssertionError("full Gate E replay must not start"),
            ) as structural, self.assertRaises(performance.PerformanceReplayError) as raised:
                self._replay(release_image=release, optimizer_image=optimizer)
            self.assert_reason(
                raised,
                (
                    "invalid-expected-release-image-id"
                    if release != self._release_image()
                    else "invalid-expected-optimizer-build-image-id"
                ),
            )
            structural.assert_not_called()

    def test_release_executable_digest_or_length_mismatch_is_rejected(self) -> None:
        release_elf = self._release_binding()
        cases = (
            b"x" * release_elf.byte_length,
            b"x" * (release_elf.byte_length + 1),
        )
        for raw in cases:
            with self.subTest(length=len(raw)):
                self._write_leaf("release", "native_candidate_executable", raw)
                patches = self._semantic_patches()
                with self.gate._defaults(), patches[0], patches[1], patches[2], patches[3], self.assertRaises(
                    performance.PerformanceReplayError
                ) as raised:
                    self._replay()
                self.assert_reason(raised, "performance-release-executable-mismatch")
                self._install_typed_leaves()

    def test_private_scratch_replacement_after_raw_replay_is_rejected(self) -> None:
        def replace(snapshot: performance._ScratchSnapshot) -> dict[str, object]:
            replacement = snapshot.path.with_name("performance-private-replacement.tmp")
            replacement.write_bytes(snapshot.path.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, snapshot.path)
            return self._raw_result()

        patches = self._semantic_patches(replace)
        with self.gate._defaults(), patches[0], patches[1], patches[2], patches[3], self.assertRaises(
            performance.PerformanceReplayError
        ) as raised:
            self._replay()
        self.assert_reason(raised, "scratch-snapshot-mutated")

    def test_optimizer_contract_policy_drift_is_rejected(self) -> None:
        with mock.patch.object(performance.optimizer_contract, "POLICY_SHA256", "0" * 64), self.assertRaises(
            performance.PerformanceReplayError
        ) as raised:
            performance._validated_optimizer_model_tree(
                {},
                source_revision=self.gate.fixture.revision,
                source_archive_sha256="a" * 64,
                expected_optimizer_build_image_id=EXPECTED_OPTIMIZER_IMAGE,
            )
        self.assert_reason(raised, "performance-optimizer-contract-policy-drift")

    def test_adapter_has_no_final_checker_or_legacy_full_buffer_dependency(self) -> None:
        source = inspect.getsource(performance)
        self.assertNotIn("check_release_candidate", source)
        self.assertNotIn("replay_raw_evidence_archive", source)
        self.assertNotIn("evaluate(", source)
        snapshot_source = inspect.getsource(performance._snapshot_performance_raw)
        self.assertIn("dir_fd=scratch_fd", snapshot_source)
        self.assertIn("os.O_EXCL", snapshot_source)
        self.assertIn("nofollow", snapshot_source)
        self.assertIn("_open_scratch_snapshot_fd", inspect.getsource(performance._replay_performance_raw))

    def test_schema_forbids_aggregate_gate_e_or_qualification_authority(self) -> None:
        schema_path = (
            Path(performance.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-gate-e-performance-semantic-replay-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        properties = schema["properties"]
        self.assertEqual(properties["qualification_status"], {"const": "not-run"})
        self.assertEqual(properties["performance_status"], {"const": "passed"})
        self.assertNotIn("gate_e_status", properties)
        self.assertNotIn("qualification", properties)
        details = properties["performance"]["properties"]
        self.assertEqual(details["raw_stream_member_byte_limit"], {"const": 67108864})
        self.assertEqual(details["scratch_disk_byte_limit"], {"const": 543686656})
        self.assertEqual(
            schema["$defs"]["notEstablished"]["required"],
            list(performance.NOT_ESTABLISHED),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
