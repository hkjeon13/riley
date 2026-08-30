#!/usr/bin/env python3
"""CPU-only boundary tests for the RC3 Gate E aggregate semantic replay.

The components themselves own the costly evidence semantics.  These tests
mock only their held-FD private cores, so they concentrate on the aggregate
boundary: one root stack, explicit anchors, cross-component descriptor
bindings, and the deliberately narrow aggregate-only conclusion.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent))

import provenance_v2_common as common  # noqa: E402
import replay_rc3_gate_e_aggregate_v1 as aggregate  # noqa: E402
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs  # noqa: E402
import replay_rc3_gate_e_native_e0_v1 as native_e0  # noqa: E402
import replay_rc3_gate_e_optimizer_e0_v1 as optimizer_e0  # noqa: E402
import replay_rc3_gate_e_performance_v1 as performance  # noqa: E402
import replay_rc3_gate_e_python_free_v1 as python_free  # noqa: E402
import replay_rc3_gate_e_soak_v1 as soak  # noqa: E402
import test_rc3_gate_e_input_inventory_v1 as gate_inventory_tests  # noqa: E402


OPTIMIZER_IMAGE = "sha256:" + "b" * 64


class Rc3GateEAggregateV1Tests(unittest.TestCase):
    """Reuse the closed-inventory fixture while mocking component semantics."""

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

    def _descriptor(self, group: str, field: str) -> dict[str, object]:
        record = self.gate.inventory[group]
        self.assertIsInstance(record, dict)
        value = record[field]
        self.assertIsInstance(value, dict)
        return copy.deepcopy(value)

    def _request_descriptor(self, *keys: str) -> dict[str, object]:
        value: object = self.gate.fixture.request
        for key in keys:
            self.assertIsInstance(value, dict)
            value = value[key]
        self.assertIsInstance(value, dict)
        return copy.deepcopy(value)

    def _release_image(self) -> str:
        value = self._request_descriptor("release", "container")
        image = value["image_digest"]
        self.assertIsInstance(image, str)
        return image

    def _golden_sha256(self) -> str:
        value = self._descriptor("python_free", "correctness_golden")["sha256"]
        self.assertIsInstance(value, str)
        return value

    def _structural(self) -> dict[str, object]:
        raw = (self.gate.gate_root / gate_inputs.INVENTORY_NAME).read_bytes()
        return {
            "schema_version": gate_inputs.REPLAY_VERSION,
            "scope": gate_inputs.SCOPE,
            "status": "bound",
            "authority": gate_inputs.AUTHORITY,
            "candidate_status": "frozen",
            "qualification_status": "not-run",
            "candidate_id": self.gate.inventory["candidate_id"],
            "source_revision": self.gate.inventory["source_revision"],
            "gate_e_input_inventory": common.descriptor_for_bytes(
                gate_inputs.INVENTORY_NAME, raw, "fixture Gate E inventory"
            ).as_json(),
            "frozen_candidate_manifest": copy.deepcopy(
                self.gate.inventory["frozen_candidate_manifest"]
            ),
            "checks": [
                {"name": name, "satisfied": True}
                for name in gate_inputs.CHECK_NAMES
            ],
            "not_established": dict(gate_inputs.NOT_ESTABLISHED),
            "reason_codes": [],
        }

    def _component_results(self) -> dict[str, dict[str, object]]:
        structural = self._structural()
        common_fields = {
            "status": "bound",
            "candidate_status": "frozen",
            "qualification_status": "not-run",
            "candidate_id": structural["candidate_id"],
            "source_revision": structural["source_revision"],
            "gate_e_input_inventory": structural["gate_e_input_inventory"],
            "frozen_candidate_manifest": structural["frozen_candidate_manifest"],
            "reason_codes": [],
        }
        source_archive = self._request_descriptor("source", "archive")
        # The calibration executable has a distinct Gate E path, but the
        # performance contract requires its digest/length identity to match the
        # frozen release ELF.  Aggregate must preserve that narrow distinction:
        # native E0 <-> performance share the descriptor; Python-free,
        # performance, and soak share the frozen release-ELF descriptor.
        release_elf = self._request_descriptor("release", "elf")
        native_executable = copy.deepcopy(release_elf)
        native_executable["path"] = self._descriptor(
            "release", "native_candidate_executable"
        )["path"]
        self.assertNotEqual(native_executable, release_elf)
        self.assertEqual(native_executable["sha256"], release_elf["sha256"])
        self.assertEqual(native_executable["byte_length"], release_elf["byte_length"])
        native_report = self._descriptor("canonical_e0", "native_report")
        optimizer_report = self._descriptor("canonical_e0", "optimizer_report")
        profile_binary = self._descriptor("release", "profile_binary")
        golden = self._descriptor("python_free", "correctness_golden")
        image = self._release_image()
        golden_sha = self._golden_sha256()
        return {
            "native_e0": {
                **common_fields,
                "schema_version": native_e0.REPLAY_VERSION,
                "scope": native_e0.SCOPE,
                "authority": native_e0.AUTHORITY,
                "native_e0_status": "passed",
                "checks": [
                    {"name": name, "satisfied": True} for name in native_e0.CHECK_NAMES
                ],
                "not_established": dict(native_e0.NOT_ESTABLISHED),
                "native_e0": {
                    "report": native_report,
                    "raw_evidence": self._descriptor("canonical_e0", "native_raw_evidence"),
                    "candidate_executable": native_executable,
                    "source_archive": source_archive,
                },
            },
            "optimizer_e0": {
                **common_fields,
                "schema_version": optimizer_e0.REPLAY_VERSION,
                "scope": optimizer_e0.SCOPE,
                "authority": optimizer_e0.AUTHORITY,
                "optimizer_e0_status": "passed",
                "expected_optimizer_build_image_id": OPTIMIZER_IMAGE,
                "checks": [
                    {"name": name, "satisfied": True} for name in optimizer_e0.CHECK_NAMES
                ],
                "not_established": dict(optimizer_e0.NOT_ESTABLISHED),
                "optimizer_e0": {
                    "report": optimizer_report,
                    "raw_evidence": self._descriptor("canonical_e0", "optimizer_raw_evidence"),
                    "profile_binary": profile_binary,
                    "source_archive": source_archive,
                },
            },
            "python_free": {
                **common_fields,
                "schema_version": python_free.REPLAY_VERSION,
                "scope": python_free.SCOPE,
                "authority": python_free.AUTHORITY,
                "python_free_status": "passed",
                "expected_release_image_id": image,
                "expected_correctness_golden_sha256": golden_sha,
                "checks": [
                    {"name": name, "satisfied": True} for name in python_free.CHECK_NAMES
                ],
                "not_established": dict(python_free.NOT_ESTABLISHED),
                "python_free": {
                    "report": self._descriptor("python_free", "report"),
                    "raw_evidence": self._descriptor("python_free", "raw_evidence"),
                    "correctness_golden": golden,
                    "release_bundle": self._descriptor("release", "bundle"),
                    "native_report": native_report,
                    "release_elf": release_elf,
                    "source_archive": source_archive,
                    "frozen_release_image_digest": image,
                },
            },
            "performance": {
                **common_fields,
                "schema_version": performance.REPLAY_VERSION,
                "scope": performance.SCOPE,
                "authority": performance.AUTHORITY,
                "performance_status": "passed",
                "expected_release_image_id": image,
                "expected_optimizer_build_image_id": OPTIMIZER_IMAGE,
                "checks": [
                    {"name": name, "satisfied": True} for name in performance.CHECK_NAMES
                ],
                "not_established": dict(performance.NOT_ESTABLISHED),
                "performance": {
                    "report": self._descriptor("performance", "report"),
                    "raw_evidence": self._descriptor("performance", "raw_evidence"),
                    "optimizer_report": optimizer_report,
                    "profile_binary": profile_binary,
                    "native_candidate_executable": native_executable,
                    "release_elf": release_elf,
                    "source_archive": source_archive,
                    "frozen_release_image_digest": image,
                    "reviewed_baseline_sha256": "c" * 64,
                },
            },
            "soak": {
                **common_fields,
                "schema_version": soak.REPLAY_VERSION,
                "scope": soak.SCOPE,
                "authority": soak.AUTHORITY,
                "soak_status": "passed",
                "expected_release_image_id": image,
                "expected_correctness_golden_sha256": golden_sha,
                "checks": [
                    {"name": name, "satisfied": True} for name in soak.CHECK_NAMES
                ],
                "not_established": dict(soak.NOT_ESTABLISHED),
                "soak": {
                    "report": self._descriptor("soak", "report"),
                    "raw_evidence": self._descriptor("soak", "raw_evidence"),
                    "correctness_golden": golden,
                    "native_report": native_report,
                    "release_elf": release_elf,
                    "source_archive": source_archive,
                    "frozen_release_image_digest": image,
                    "model_tree_sha256": "d" * 64,
                },
            },
        }

    def _snapshot_tree(self, root: Path) -> dict[str, bytes]:
        return {
            os.fspath(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        }

    def _component_patches(
        self,
        results: dict[str, dict[str, object]],
        calls: dict[str, tuple[object, ...]] | None = None,
    ):
        def call(name: str):
            def replay(*args: object) -> dict[str, object]:
                if calls is not None:
                    calls[name] = args
                return copy.deepcopy(results[name])

            return replay

        return (
            mock.patch.object(
                native_e0, "_replay_rc3_gate_e_native_e0_v1_on_held_fds", side_effect=call("native_e0")
            ),
            mock.patch.object(
                optimizer_e0,
                "_replay_rc3_gate_e_optimizer_e0_v1_on_held_fds",
                side_effect=call("optimizer_e0"),
            ),
            mock.patch.object(
                python_free, "_replay_rc3_gate_e_python_free_v1_on_held_fds", side_effect=call("python_free")
            ),
            mock.patch.object(
                performance,
                "_replay_rc3_gate_e_performance_v1_on_held_fds",
                side_effect=call("performance"),
            ),
            mock.patch.object(soak, "_replay_rc3_gate_e_soak_v1_on_held_fds", side_effect=call("soak")),
        )

    def _replay(
        self,
        results: dict[str, dict[str, object]],
        *,
        structural_side_effect: object | None = None,
        calls: dict[str, tuple[object, ...]] | None = None,
    ) -> dict[str, object]:
        patches = self._component_patches(results, calls)
        inventory_patch = (
            mock.patch.object(
                gate_inputs,
                "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                return_value=self._structural(),
            )
            if structural_side_effect is None
            else mock.patch.object(
                gate_inputs,
                "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                side_effect=structural_side_effect,
            )
        )
        with self.gate._defaults(), inventory_patch as inventory, patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = aggregate.replay_rc3_gate_e_aggregate_v1(
                self.gate.gate_root,
                frozen_candidate_root=self.gate.frozen_root,
                input_evidence_root=self.gate.fixture.evidence,
                repository_root=self.gate.fixture.root,
                expected_release_image_id=self._release_image(),
                expected_optimizer_build_image_id=OPTIMIZER_IMAGE,
                expected_correctness_golden_sha256=self._golden_sha256(),
            )
        self.assertEqual(inventory.call_count, 2)
        return result

    def test_replays_all_components_on_one_held_fd_stack_without_mutation(self) -> None:
        before = {
            "gate": self._snapshot_tree(self.gate.gate_root),
            "frozen": self._snapshot_tree(self.gate.frozen_root),
            "input": self._snapshot_tree(self.gate.fixture.evidence),
            "source": self._snapshot_tree(self.gate.fixture.root),
        }
        calls: dict[str, tuple[object, ...]] = {}
        result = self._replay(self._component_results(), calls=calls)

        self.assertEqual(
            {
                "gate": self._snapshot_tree(self.gate.gate_root),
                "frozen": self._snapshot_tree(self.gate.frozen_root),
                "input": self._snapshot_tree(self.gate.fixture.evidence),
                "source": self._snapshot_tree(self.gate.fixture.root),
            },
            before,
        )
        self.assertEqual(set(calls), {"native_e0", "optimizer_e0", "python_free", "performance", "soak"})
        native = calls["native_e0"]
        self.assertEqual(len(native), 5)
        for name in ("optimizer_e0", "python_free", "performance", "soak"):
            self.assertEqual(calls[name][:5], native)
        self.assertEqual(calls["optimizer_e0"][5:], (aggregate.EXTERNAL_SCRATCH_PARENT, OPTIMIZER_IMAGE))
        self.assertEqual(
            calls["python_free"][5:],
            (aggregate.EXTERNAL_SCRATCH_PARENT, self._release_image(), self._golden_sha256()),
        )
        self.assertEqual(
            calls["performance"][5:],
            (aggregate.EXTERNAL_SCRATCH_PARENT, self._release_image(), OPTIMIZER_IMAGE),
        )
        self.assertEqual(
            calls["soak"][5:],
            (aggregate.EXTERNAL_SCRATCH_PARENT, self._release_image(), self._golden_sha256()),
        )
        self.assertEqual(result["schema_version"], aggregate.REPLAY_VERSION)
        self.assertEqual(result["scope"], aggregate.SCOPE)
        self.assertEqual(result["authority"], aggregate.AUTHORITY)
        self.assertEqual(
            result["aggregate_policy_version"], aggregate.AGGREGATE_POLICY_VERSION
        )
        self.assertEqual(
            result["aggregate_policy_sha256"], aggregate.AGGREGATE_POLICY_SHA256
        )
        self.assertEqual(result["status"], "bound")
        self.assertEqual(result["candidate_status"], "frozen")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["gate_e_status"], "passed")
        self.assertEqual(result["not_established"], aggregate.NOT_ESTABLISHED)
        components = result["components"]
        self.assertIsInstance(components, dict)
        self.assertEqual(
            components["native_e0"]["candidate_executable"],
            components["performance"]["native_candidate_executable"],
        )
        self.assertNotEqual(
            components["native_e0"]["candidate_executable"],
            result["shared_bindings"]["release_elf"],
        )
        self.assertEqual(
            components["native_e0"]["candidate_executable"]["sha256"],
            result["shared_bindings"]["release_elf"]["sha256"],
        )

    def test_rejects_component_with_child_or_wrong_scope_contract(self) -> None:
        results = self._component_results()
        results["soak"]["scope"] = "child-contract"
        with self.assertRaises(aggregate.AggregateReplayError) as raised:
            self._replay(results)
        self.assert_reason(raised, "invalid-component-result")

    def test_rejects_cross_component_descriptor_mismatch(self) -> None:
        results = self._component_results()
        payload = results["performance"]["performance"]
        self.assertIsInstance(payload, dict)
        payload["optimizer_report"] = {
            "path": "other-optimizer-report.json",
            "sha256": "e" * 64,
            "byte_length": 1,
        }
        with self.assertRaises(aggregate.AggregateReplayError) as raised:
            self._replay(results)
        self.assert_reason(raised, "optimizer-report-binding-mismatch")

    def test_rejects_native_executable_identity_mismatch_with_frozen_release_elf(self) -> None:
        results = self._component_results()
        payload = results["native_e0"]["native_e0"]
        self.assertIsInstance(payload, dict)
        candidate = payload["candidate_executable"]
        self.assertIsInstance(candidate, dict)
        candidate["sha256"] = "e" * 64
        performance_payload = results["performance"]["performance"]
        self.assertIsInstance(performance_payload, dict)
        performance_candidate = performance_payload["native_candidate_executable"]
        self.assertIsInstance(performance_candidate, dict)
        performance_candidate["sha256"] = "e" * 64
        with self.assertRaises(aggregate.AggregateReplayError) as raised:
            self._replay(results)
        self.assert_reason(raised, "release-executable-identity-mismatch")

    def test_rejects_final_inventory_drift_after_components_finish(self) -> None:
        start = self._structural()
        end = copy.deepcopy(start)
        descriptor = end["gate_e_input_inventory"]
        self.assertIsInstance(descriptor, dict)
        descriptor["sha256"] = "f" * 64
        with self.assertRaises(aggregate.AggregateReplayError) as raised:
            self._replay(self._component_results(), structural_side_effect=[start, end])
        self.assert_reason(raised, "gate-e-input-replay-drift")

    def test_bytecode_guard_and_static_ban_on_public_component_wrappers(self) -> None:
        with mock.patch.object(aggregate, "_BYTECODE_DISABLED_AT_STARTUP", False), self.assertRaises(
            aggregate.AggregateReplayError
        ) as raised:
            aggregate._replay_rc3_gate_e_aggregate_v1_on_held_fds(
                -1,
                -1,
                -1,
                self.gate.fixture.root,
                -1,
                aggregate.EXTERNAL_SCRATCH_PARENT,
                self._release_image(),
                OPTIMIZER_IMAGE,
                self._golden_sha256(),
            )
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")
        source = inspect.getsource(aggregate)
        self.assertNotIn("check_release_candidate", source)
        self.assertNotIn("replay_rc3_gate_e_native_e0_v1(", source)
        self.assertNotIn("replay_rc3_gate_e_optimizer_e0_v1(", source)
        self.assertNotIn("replay_rc3_gate_e_python_free_v1(", source)
        self.assertNotIn("replay_rc3_gate_e_performance_v1(", source)
        self.assertNotIn("replay_rc3_gate_e_soak_v1(", source)
        self.assertIn("_replay_rc3_gate_e_native_e0_v1_on_held_fds", source)
        self.assertIn("_replay_rc3_gate_e_soak_v1_on_held_fds", source)

    def test_schema_reserves_the_pinned_aggregate_and_child_contracts(self) -> None:
        schema = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "benchmarks/release/candidates"
                / "rc3-gate-e-aggregate-semantic-replay-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        properties = schema["properties"]
        self.assertEqual(properties["schema_version"]["const"], aggregate.REPLAY_VERSION)
        self.assertEqual(properties["scope"]["const"], aggregate.SCOPE)
        self.assertEqual(properties["authority"]["const"], aggregate.AUTHORITY)
        self.assertEqual(
            properties["aggregate_policy_version"]["const"],
            aggregate.AGGREGATE_POLICY_VERSION,
        )
        self.assertEqual(
            properties["aggregate_policy_sha256"]["const"],
            aggregate.AGGREGATE_POLICY_SHA256,
        )
        not_established = schema["$defs"]["notEstablished"]
        self.assertEqual(
            set(not_established["required"]), set(aggregate.NOT_ESTABLISHED),
        )
        self.assertEqual(
            not_established["properties"]["actual_candidate_gate_e_pass"]["const"],
            "not-established",
        )
        child_cases = (
            ("native", native_e0, "native_e0_status"),
            ("optimizer", optimizer_e0, "optimizer_e0_status"),
            ("pythonFree", python_free, "python_free_status"),
            ("performance", performance, "performance_status"),
            ("soak", soak, "soak_status"),
        )
        for schema_name, module, status_field in child_cases:
            with self.subTest(component=schema_name):
                child = schema["$defs"][schema_name]["properties"]
                self.assertEqual(child["schema_version"]["const"], module.REPLAY_VERSION)
                self.assertEqual(child["scope"]["const"], module.SCOPE)
                self.assertEqual(child["authority"]["const"], module.AUTHORITY)
                self.assertEqual(child["status"]["const"], "bound")
                self.assertEqual(child[status_field]["const"], "passed")
                self.assertIn("anchors", schema["$defs"][schema_name]["required"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
