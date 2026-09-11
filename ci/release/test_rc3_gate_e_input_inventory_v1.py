#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 Gate E input inventory boundary."""

from __future__ import annotations

import copy
import fcntl
import inspect
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent))

import check_rc3_prefreeze as prefreeze  # noqa: E402
import provenance_v2_common as common  # noqa: E402
import rc3_frozen_candidate_common as frozen  # noqa: E402
import replay_rc3_frozen_candidate_v1 as frozen_replayer  # noqa: E402
import replay_rc3_gate_e_input_inventory_v1 as gate_e  # noqa: E402
import write_rc3_frozen_candidate_v1 as frozen_writer  # noqa: E402
from test_check_rc3_freeze_input_admission import (  # noqa: E402
    AdmissionFixture,
    CANDIDATE_ID,
    _sha256,
)


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class Rc3GateEInputInventoryV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve(strict=True)
        self.fixture = AdmissionFixture(self.base)
        self.frozen_root = self.base / "frozen-candidate"
        self.gate_root = self.base / "gate-e-inputs"
        self._write_frozen()
        self.gate_root.mkdir(mode=0o700)
        self.gate_root.chmod(0o700)
        self.inventory: dict[str, object] = {
            "schema_version": gate_e.INVENTORY_VERSION,
            "candidate_id": CANDIDATE_ID,
            "source_revision": self.fixture.revision,
            "frozen_candidate_manifest": self._frozen_manifest_descriptor(),
            "release": {},
            "canonical_e0": {},
            "python_free": {},
            "performance": {},
            "soak": {},
        }
        self._write_artifacts()
        self._write_inventory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext,
        reason: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _defaults(self):
        return mock.patch.object(
            prefreeze,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            _sha256(self.fixture.defaults),
        )

    def _write_frozen(self) -> None:
        with self._defaults():
            frozen_writer.write_rc3_frozen_candidate_v1(
                self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
                expected_revision=self.fixture.revision,
                candidate_id=CANDIDATE_ID,
                request_name=self.fixture.request_name,
            )

    def _frozen_manifest_descriptor(self) -> dict[str, object]:
        raw = (self.frozen_root / frozen.MANIFEST_NAME).read_bytes()
        return common.descriptor_for_bytes(
            frozen.MANIFEST_NAME,
            raw,
            "fixture frozen candidate manifest",
        ).as_json()

    def _put(self, name: str, contents: bytes) -> dict[str, object]:
        path = self.gate_root / name
        path.write_bytes(contents)
        return common.descriptor_for_bytes(name, contents, f"fixture {name}").as_json()

    def _write_artifacts(self) -> None:
        groups = {
            "release": {
                "bundle": "release-bundle.tar",
                "profile_binary": "profile-binary",
                "native_candidate_executable": "native-candidate-executable",
            },
            "canonical_e0": {
                "native_report": "native-e0-report.json",
                "native_raw_evidence": "native-e0-raw.tar",
                "optimizer_report": "optimizer-e0-report.json",
                "optimizer_raw_evidence": "optimizer-e0-raw.tar",
            },
            "python_free": {
                "report": "python-free-report.json",
                "raw_evidence": "python-free-raw.tar",
                "correctness_golden": "python-free-golden.json",
            },
            "performance": {
                "report": "performance-report.json",
                "raw_evidence": "performance-raw.tar",
            },
            "soak": {
                "report": "soak-report.json",
                "raw_evidence": "soak-raw.tar",
            },
        }
        for group, fields in groups.items():
            target = self.inventory[group]
            assert isinstance(target, dict)
            for field, name in fields.items():
                target[field] = self._put(name, f"fixture {group}.{field}\n".encode())

    def _write_inventory(self) -> None:
        (self.gate_root / gate_e.INVENTORY_NAME).write_bytes(
            common.canonical_json_bytes(self.inventory)
        )

    def _snapshot(self) -> dict[str, bytes]:
        return {
            path.name: path.read_bytes()
            for path in sorted(self.gate_root.iterdir())
        }

    def _replay(self) -> dict[str, object]:
        with self._defaults():
            return gate_e.replay_rc3_gate_e_input_inventory_v1(
                self.gate_root,
                frozen_candidate_root=self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
            )

    def test_closed_four_gate_inventory_replays_without_a_gate_e_decision(self) -> None:
        before = self._snapshot()
        result = self._replay()

        self.assertEqual(self._snapshot(), before)
        self.assertEqual(result["schema_version"], gate_e.REPLAY_VERSION)
        self.assertEqual(result["status"], "bound")
        self.assertEqual(result["candidate_status"], "frozen")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["authority"], gate_e.AUTHORITY)
        self.assertEqual(result["candidate_id"], CANDIDATE_ID)
        self.assertEqual(result["source_revision"], self.fixture.revision)
        self.assertEqual(result["not_established"], gate_e.NOT_ESTABLISHED)
        self.assertNotIn("passed", result)
        self.assertNotIn(
            "qualification",
            {check["name"] for check in result["checks"]},
        )
        self.assertEqual(
            result["frozen_candidate_manifest"],
            self.inventory["frozen_candidate_manifest"],
        )

    def test_public_and_private_replays_are_read_only(self) -> None:
        before = self._snapshot()
        with self._defaults(), mock.patch.object(
            common,
            "write_create_only",
            side_effect=AssertionError("replayer must not write"),
        ), mock.patch.object(
            common,
            "write_create_only_json",
            side_effect=AssertionError("replayer must not write JSON"),
        ):
            public = gate_e.replay_rc3_gate_e_input_inventory_v1(
                self.gate_root,
                frozen_candidate_root=self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
            )

        source_fd = common.open_absolute_directory(self.fixture.root, "fixture source")
        input_fd = common.open_private_evidence_directory(self.fixture.evidence, "fixture input")
        frozen_fd = common.open_private_evidence_directory(self.frozen_root, "fixture frozen")
        gate_fd = common.open_private_evidence_directory(self.gate_root, "fixture gate")
        try:
            for descriptor in (input_fd, frozen_fd, gate_fd):
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with self._defaults():
                private = gate_e._replay_rc3_gate_e_input_inventory_v1_on_held_fds(
                    gate_fd,
                    frozen_fd,
                    input_fd,
                    self.fixture.root,
                    source_fd,
                )
        finally:
            for descriptor in (gate_fd, frozen_fd, input_fd):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(gate_fd)
            os.close(frozen_fd)
            os.close(input_fd)
            os.close(source_fd)
        self.assertEqual(public, private)
        self.assertEqual(self._snapshot(), before)

    def test_private_core_rejects_gate_root_nested_in_freeze_input(self) -> None:
        nested_gate_root = self.fixture.evidence / "nested-gate-e-inputs"
        os.rename(self.gate_root, nested_gate_root)

        source_fd = common.open_absolute_directory(self.fixture.root, "fixture source")
        input_fd = common.open_private_evidence_directory(self.fixture.evidence, "fixture input")
        frozen_fd = common.open_private_evidence_directory(self.frozen_root, "fixture frozen")
        gate_fd = common.open_private_evidence_directory(nested_gate_root, "fixture nested gate")
        try:
            for descriptor in (input_fd, frozen_fd, gate_fd):
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with self._defaults(), self.assertRaises(
                gate_e.GateEInventoryReplayError
            ) as raised:
                gate_e._replay_rc3_gate_e_input_inventory_v1_on_held_fds(
                    gate_fd,
                    frozen_fd,
                    input_fd,
                    self.fixture.root,
                    source_fd,
                )
        finally:
            for descriptor in (gate_fd, frozen_fd, input_fd):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(gate_fd)
            os.close(frozen_fd)
            os.close(input_fd)
            os.close(source_fd)
        self.assert_reason(raised, "frozen-candidate-root-overlap")

    def test_cli_emits_only_canonical_structural_result(self) -> None:
        stdout = _CapturedStdout()
        with self._defaults(), mock.patch.object(gate_e.sys, "stdout", stdout):
            result = gate_e.main(
                [
                    "--gate-e-evidence-root",
                    os.fspath(self.gate_root),
                    "--frozen-candidate-root",
                    os.fspath(self.frozen_root),
                    "--input-evidence-root",
                    os.fspath(self.fixture.evidence),
                    "--repository-root",
                    os.fspath(self.fixture.root),
                ]
            )
        self.assertEqual(result, 0)
        raw = stdout.buffer.getvalue()
        document = json.loads(raw)
        self.assertEqual(raw, common.canonical_json_bytes(document) + b"\n")
        self.assertEqual(document["status"], "bound")
        self.assertNotIn("passed", document)

    def test_duplicate_or_oversized_closure_fails_before_frozen_raw_replay(self) -> None:
        performance = self.inventory["performance"]
        soak = self.inventory["soak"]
        assert isinstance(performance, dict)
        assert isinstance(soak, dict)
        performance["raw_evidence"] = copy.deepcopy(soak["raw_evidence"])
        self._write_inventory()
        with mock.patch.object(
            frozen_replayer,
            "_replay_rc3_frozen_candidate_v1_on_held_fds",
            side_effect=AssertionError("duplicate closure must fail before frozen raw replay"),
        ), self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "duplicate-evidence-path")

        performance["raw_evidence"] = self._put("performance-replacement.tar", b"replacement\n")
        performance["raw_evidence"]["byte_length"] = gate_e.MAX_GATE_E_INPUT_BYTES + 1
        self._write_inventory()
        with mock.patch.object(
            frozen_replayer,
            "_replay_rc3_frozen_candidate_v1_on_held_fds",
            side_effect=AssertionError("budget must fail before frozen raw replay"),
        ), self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "external-evidence-byte-budget-exceeded")

    def test_rejects_extra_leaf_nested_descriptor_and_root_overlap(self) -> None:
        (self.gate_root / "unexpected").write_bytes(b"extra\n")
        with self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "unexpected-evidence-entry")
        (self.gate_root / "unexpected").unlink()

        performance = self.inventory["performance"]
        assert isinstance(performance, dict)
        performance["raw_evidence"]["path"] = "nested/performance.tar"
        self._write_inventory()
        with self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "gate-e-input-must-be-direct-root-leaf")

        performance["raw_evidence"]["path"] = "a" * (
            gate_e.MAX_DIRECT_LEAF_NAME_LENGTH + 1
        )
        self._write_inventory()
        with self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "gate-e-input-must-be-direct-root-leaf")

        nested = self.fixture.evidence / "gate-e"
        with self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            with self._defaults():
                gate_e.replay_rc3_gate_e_input_inventory_v1(
                    nested,
                    frozen_candidate_root=self.frozen_root,
                    input_evidence_root=self.fixture.evidence,
                    repository_root=self.fixture.root,
                )
        self.assert_reason(raised, "frozen-candidate-root-overlap")

    def test_rejects_frozen_identity_or_manifest_descriptor_mismatch(self) -> None:
        self.inventory["candidate_id"] = "riley-0.1.0-rc4"
        self._write_inventory()
        with self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "frozen-candidate-identity-mismatch")

        self.inventory["candidate_id"] = CANDIDATE_ID
        self.inventory["frozen_candidate_manifest"] = copy.deepcopy(
            self.inventory["frozen_candidate_manifest"]
        )
        self.inventory["frozen_candidate_manifest"]["sha256"] = "a" * 64
        self._write_inventory()
        with self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "frozen-candidate-manifest-descriptor-mismatch")

    def test_detects_inventory_or_visible_gate_root_replacement(self) -> None:
        original_verify = common.verify_descriptor_file
        changed = False

        def mutate_inventory(*args, **kwargs):
            nonlocal changed
            result = original_verify(*args, **kwargs)
            if not changed:
                changed = True
                (self.gate_root / gate_e.INVENTORY_NAME).write_bytes(b"{}")
            return result

        with mock.patch.object(common, "verify_descriptor_file", side_effect=mutate_inventory), self.assertRaises(
            gate_e.GateEInventoryReplayError
        ) as raised:
            self._replay()
        self.assertIn(getattr(raised.exception, "reason_code", ""), {"invalid-gate-e-inventory", "noncanonical-json"})

        self._write_inventory()
        displaced = self.base / "displaced-gate-e"
        original_core = gate_e._replay_rc3_gate_e_input_inventory_v1_on_held_fds

        def replace_visible_root(*args, **kwargs):
            os.rename(self.gate_root, displaced)
            self.gate_root.mkdir(mode=0o700)
            self.gate_root.chmod(0o700)
            return original_core(*args, **kwargs)

        with mock.patch.object(
            gate_e,
            "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
            side_effect=replace_visible_root,
        ), self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "raced-output")

    def test_rehashes_gate_artifacts_after_the_second_frozen_replay(self) -> None:
        original_verify = common.verify_descriptor_file
        mutated = False

        def mutate_after_first_round(*args, **kwargs):
            nonlocal mutated
            result = original_verify(*args, **kwargs)
            label = args[2] if len(args) > 2 else kwargs.get("label")
            if label == "Gate E input soak.report" and not mutated:
                mutated = True
                (self.gate_root / "release-bundle.tar").write_bytes(b"changed-after-first-round\n")
            return result

        with mock.patch.object(
            common,
            "verify_descriptor_file",
            side_effect=mutate_after_first_round,
        ), self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assertTrue(mutated)
        self.assertIn(
            getattr(raised.exception, "reason_code", ""),
            {"evidence-length-mismatch", "evidence-hash-mismatch"},
        )

    def test_rejects_bytecode_enabled_before_opening_any_evidence_root(self) -> None:
        with mock.patch.object(gate_e, "_BYTECODE_DISABLED_AT_STARTUP", False), mock.patch.object(
            common,
            "open_absolute_directory",
            side_effect=AssertionError("bytecode guard must precede evidence opens"),
        ), self.assertRaises(gate_e.GateEInventoryReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")

    def test_schemas_and_private_core_do_not_expand_to_semantic_or_writer_surface(self) -> None:
        schemas = (
            "rc3-gate-e-input-inventory-v1.schema.json",
            "rc3-gate-e-input-inventory-replay-v1.schema.json",
        )
        expected_group_fields = {
            "release": {
                "bundle",
                "profile_binary",
                "native_candidate_executable",
            },
            "canonical_e0": {
                "native_report",
                "native_raw_evidence",
                "optimizer_report",
                "optimizer_raw_evidence",
            },
            "python_free": {"report", "raw_evidence", "correctness_golden"},
            "performance": {"report", "raw_evidence"},
            "soak": {"report", "raw_evidence"},
        }
        for name in schemas:
            path = Path(__file__).resolve().parents[2] / "benchmarks/release/candidates" / name
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(document["additionalProperties"])
            self.assertNotIn("passed", json.dumps(document, sort_keys=True))
            if name == "rc3-gate-e-input-inventory-v1.schema.json":
                self.assertEqual(
                    set(document["required"]),
                    {
                        "schema_version",
                        "candidate_id",
                        "source_revision",
                        "frozen_candidate_manifest",
                        "release",
                        "canonical_e0",
                        "python_free",
                        "performance",
                        "soak",
                    },
                )
                self.assertEqual(
                    {
                        "release": set(document["$defs"]["release"]["required"]),
                        "canonical_e0": set(document["$defs"]["canonicalE0"]["required"]),
                        "python_free": set(document["$defs"]["pythonFree"]["required"]),
                        "performance": set(document["$defs"]["performance"]["required"]),
                        "soak": set(document["$defs"]["soak"]["required"]),
                    },
                    expected_group_fields,
                )
                self.assertEqual(
                    {
                        group: set(fields)
                        for group, fields in gate_e.GATE_E_INPUT_GROUP_FIELDS
                    },
                    expected_group_fields,
                )
                self.assertEqual(
                    sum(len(fields) for _group, fields in gate_e.GATE_E_INPUT_GROUP_FIELDS),
                    gate_e.MAX_GATE_E_INPUT_DESCRIPTORS,
                )
            else:
                self.assertEqual(
                    [
                        document["$defs"][item["$ref"].rsplit("/", 1)[-1]]["allOf"][1]["properties"]["name"]["const"]
                        for item in document["properties"]["checks"]["prefixItems"]
                    ],
                    list(gate_e.CHECK_NAMES),
                )
        core_source = inspect.getsource(gate_e._replay_rc3_gate_e_input_inventory_v1_on_held_fds)
        for forbidden in (
            "fcntl.",
            "os.open",
            "os.close",
            "os.mkdir",
            "write_create",
            "check_release_candidate",
            "docker",
            "nvidia-smi",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, core_source)
        self.assertIn("trusted", inspect.getdoc(gate_e._replay_rc3_gate_e_input_inventory_v1_on_held_fds) or "")


if __name__ == "__main__":
    unittest.main()
