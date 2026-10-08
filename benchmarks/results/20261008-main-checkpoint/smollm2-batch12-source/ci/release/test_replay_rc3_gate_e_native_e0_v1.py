#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 native canonical-E0 adapter."""

from __future__ import annotations

import json
import inspect
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent))

import provenance_v2_common as common  # noqa: E402
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs  # noqa: E402
import replay_rc3_gate_e_native_e0_v1 as native_e0  # noqa: E402
import test_rc3_gate_e_input_inventory_v1 as gate_inventory_tests  # noqa: E402


class Rc3GateENativeE0V1Tests(unittest.TestCase):
    """Reuse the closed, non-semantic Gate E fixture without duplicating it."""

    def setUp(self) -> None:
        self.gate = gate_inventory_tests.Rc3GateEInputInventoryV1Tests(
            "test_closed_four_gate_inventory_replays_without_a_gate_e_decision"
        )
        self.gate.setUp()

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

    def _source_archive(self) -> common.EvidenceDescriptor:
        source = self.gate.fixture.request["source"]
        self.assertIsInstance(source, dict)
        return common.parse_descriptor(source["archive"], "fixture source archive")

    def _artifacts(self) -> native_e0._NativeE0Artifacts:
        return native_e0._NativeE0Artifacts(
            report=self._descriptor("canonical_e0", "native_report"),
            raw_evidence=self._descriptor("canonical_e0", "native_raw_evidence"),
            candidate_executable=self._descriptor(
                "release",
                "native_candidate_executable",
            ),
        )

    def _native_result(self) -> types.SimpleNamespace:
        source = self._source_archive()
        artifacts = self._artifacts()
        return types.SimpleNamespace(
            schema_version=native_e0.NATIVE_RAW_SCHEMA_VERSION,
            source_revision=self.gate.fixture.revision,
            source_archive_sha256=source.sha256,
            source_archive_byte_length=source.byte_length,
            correctness_report_sha256=artifacts.report.sha256,
            correctness_report_byte_length=artifacts.report.byte_length,
            candidate_executable_sha256=artifacts.candidate_executable.sha256,
            candidate_executable_byte_length=artifacts.candidate_executable.byte_length,
            case_count=31,
            failure_count=0,
        )

    def _replay(self) -> dict[str, object]:
        return native_e0.replay_rc3_gate_e_native_e0_v1(
            self.gate.gate_root,
            frozen_candidate_root=self.gate.frozen_root,
            input_evidence_root=self.gate.fixture.evidence,
            repository_root=self.gate.fixture.root,
        )

    def test_verified_private_snapshot_replays_only_native_component(self) -> None:
        raw = self._artifacts().raw_evidence
        expected_raw = (self.gate.gate_root / raw.path).read_bytes()
        snapshots: list[Path] = []

        def replay(snapshot: Path, *, source_revision: str) -> types.SimpleNamespace:
            self.assertEqual(source_revision, self.gate.fixture.revision)
            self.assertEqual(snapshot.name, "native-e0-raw-evidence.tar")
            self.assertEqual(snapshot.read_bytes(), expected_raw)
            observed = common.descriptor_for_bytes(
                snapshot.name,
                snapshot.read_bytes(),
                "private native E0 scratch snapshot",
            )
            self.assertEqual(observed.sha256, raw.sha256)
            self.assertEqual(observed.byte_length, raw.byte_length)
            snapshots.append(snapshot)
            return self._native_result()

        before = self.gate._snapshot()
        with self.gate._defaults(), mock.patch.object(
            native_e0,
            "_replay_native_raw",
            side_effect=replay,
        ):
            result = self._replay()

        self.assertEqual(self.gate._snapshot(), before)
        self.assertEqual(result["schema_version"], native_e0.REPLAY_VERSION)
        self.assertEqual(result["scope"], native_e0.SCOPE)
        self.assertEqual(result["authority"], native_e0.AUTHORITY)
        self.assertEqual(result["status"], "bound")
        self.assertEqual(result["native_e0_status"], "passed")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["not_established"], native_e0.NOT_ESTABLISHED)
        self.assertNotIn("gate_e_status", result)
        self.assertNotIn("qualification", result)
        self.assertEqual(result["checks"], [
            {"name": name, "satisfied": True} for name in native_e0.CHECK_NAMES
        ])
        self.assertEqual(len(snapshots), 1)
        self.assertFalse(snapshots[0].exists())

    def test_oversized_native_raw_is_rejected_before_full_gate_stream_replay(self) -> None:
        canonical_e0 = self.gate.inventory["canonical_e0"]
        self.assertIsInstance(canonical_e0, dict)
        raw = canonical_e0["native_raw_evidence"]
        self.assertIsInstance(raw, dict)
        raw["byte_length"] = native_e0.MAX_NATIVE_RAW_BYTES + 1
        self.gate._write_inventory()

        with self.gate._defaults(), mock.patch.object(
            gate_inputs,
            "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
            side_effect=AssertionError("full Gate E replay must not start"),
        ) as structural, mock.patch.object(
            native_e0,
            "_replay_native_raw",
            side_effect=AssertionError("native raw replay must not start"),
        ) as raw_replay, self.assertRaises(native_e0.NativeE0ReplayError) as raised:
            self._replay()

        self.assert_reason(raised, "native-e0-raw-evidence-too-large")
        structural.assert_not_called()
        raw_replay.assert_not_called()

    def test_result_cross_binding_rejects_every_sha_and_length_mismatch(self) -> None:
        source = self._source_archive()
        artifacts = self._artifacts()
        mismatches = {
            "source_archive_sha256": "0" * 64,
            "source_archive_byte_length": source.byte_length + 1,
            "correctness_report_sha256": "0" * 64,
            "correctness_report_byte_length": artifacts.report.byte_length + 1,
            "candidate_executable_sha256": "0" * 64,
            "candidate_executable_byte_length": artifacts.candidate_executable.byte_length + 1,
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                result = self._native_result()
                setattr(result, field, value)
                with self.assertRaises(native_e0.NativeE0ReplayError) as raised:
                    native_e0._require_native_result_bindings(
                        result,
                        source_revision=self.gate.fixture.revision,
                        source_archive=source,
                        artifacts=artifacts,
                    )
                self.assert_reason(raised, f"native-{field}-mismatch")

    def test_structural_end_replay_rejects_gate_raw_replacement(self) -> None:
        raw = self._artifacts().raw_evidence
        raw_path = self.gate.gate_root / raw.path

        def replace_after_snapshot(
            _snapshot: Path,
            *,
            source_revision: str,
        ) -> types.SimpleNamespace:
            self.assertEqual(source_revision, self.gate.fixture.revision)
            replacement = raw_path.with_name("native-e0-replacement.tmp")
            replacement.write_bytes(b"x" * raw_path.stat().st_size)
            os.replace(replacement, raw_path)
            return self._native_result()

        with self.gate._defaults(), mock.patch.object(
            native_e0,
            "_replay_native_raw",
            side_effect=replace_after_snapshot,
        ), self.assertRaises(native_e0.NativeE0ReplayError) as raised:
            self._replay()

        self.assert_reason(raised, "evidence-hash-mismatch")

    def test_private_scratch_replacement_after_legacy_replay_is_rejected(self) -> None:
        def replace_private_snapshot(
            snapshot: Path,
            *,
            source_revision: str,
        ) -> types.SimpleNamespace:
            self.assertEqual(source_revision, self.gate.fixture.revision)
            replacement = snapshot.with_name("native-e0-private-replacement.tmp")
            replacement.write_bytes(snapshot.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, snapshot)
            return self._native_result()

        with self.gate._defaults(), mock.patch.object(
            native_e0,
            "_replay_native_raw",
            side_effect=replace_private_snapshot,
        ), self.assertRaises(native_e0.NativeE0ReplayError) as raised:
            self._replay()

        self.assert_reason(raised, "scratch-snapshot-mutated")

    def test_scratch_copy_is_created_only_below_the_pinned_directory_fd(self) -> None:
        source = inspect.getsource(native_e0._snapshot_direct_descriptor)
        self.assertIn("common.open_private_evidence_directory", source)
        self.assertIn("dir_fd=scratch_fd", source)
        self.assertIn("os.O_EXCL", source)
        self.assertIn("nofollow", source)
        self.assertNotIn("target.open", source)

    def test_native_adapter_requires_passing_report_from_legacy_replayer(self) -> None:
        calls: list[dict[str, object]] = []
        sentinel = object()
        fake = types.ModuleType("check_native_correctness_evidence")

        class FakeNativeEvidenceError(ValueError):
            pass

        def replay_raw_evidence(
            snapshot: Path,
            *,
            source_revision: str,
            require_passing_report: bool,
        ) -> object:
            calls.append(
                {
                    "snapshot": snapshot,
                    "source_revision": source_revision,
                    "require_passing_report": require_passing_report,
                }
            )
            return sentinel

        fake.MAX_RAW_ARCHIVE_BYTES = native_e0.MAX_NATIVE_RAW_BYTES
        fake.NativeCorrectnessEvidenceError = FakeNativeEvidenceError
        fake.replay_raw_evidence = replay_raw_evidence
        with mock.patch.dict(sys.modules, {"check_native_correctness_evidence": fake}):
            result = native_e0._replay_native_raw(
                Path("/private/native-e0-raw-evidence.tar"),
                source_revision=self.gate.fixture.revision,
            )

        self.assertIs(result, sentinel)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["source_revision"], self.gate.fixture.revision)
        self.assertTrue(calls[0]["require_passing_report"])

    def test_bytecode_cache_must_have_been_disabled_before_entry(self) -> None:
        with mock.patch.object(native_e0, "_BYTECODE_DISABLED_AT_STARTUP", False), mock.patch.object(
            native_e0,
            "_BYTECODE_DISABLED_ON_MODULE_ENTRY",
            False,
        ), self.assertRaises(native_e0.NativeE0ReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")

    def test_schema_forbids_aggregate_gate_e_or_qualification_authority(self) -> None:
        schema_path = (
            Path(native_e0.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-gate-e-native-e0-semantic-replay-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"],
            "https://riley.invalid/benchmarks/release/candidates/"
            "rc3-gate-e-native-e0-semantic-replay-v1.schema.json",
        )
        self.assertFalse(schema["additionalProperties"])
        properties = schema["properties"]
        self.assertEqual(properties["qualification_status"], {"const": "not-run"})
        self.assertEqual(properties["native_e0_status"], {"const": "passed"})
        self.assertNotIn("gate_e_status", properties)
        self.assertNotIn("qualification", properties)
        self.assertEqual(
            schema["$defs"]["notEstablished"]["required"],
            list(native_e0.NOT_ESTABLISHED),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
