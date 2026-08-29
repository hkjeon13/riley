#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 optimizer canonical-E0 adapter."""

from __future__ import annotations

import inspect
import json
import os
import sys
import types
import unittest
import builtins
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent))

import provenance_v2_common as common  # noqa: E402
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs  # noqa: E402
import replay_rc3_gate_e_optimizer_e0_v1 as optimizer_e0  # noqa: E402
import optimizer_e0_semantic_contract as optimizer_contract  # noqa: E402
import test_rc3_gate_e_input_inventory_v1 as gate_inventory_tests  # noqa: E402


EXPECTED_IMAGE = "sha256:" + "a" * 64


class Rc3GateEOptimizerE0V1Tests(unittest.TestCase):
    """Reuse the closed structural Gate E fixture without duplicating it."""

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

    def _artifacts(self) -> optimizer_e0._OptimizerE0Artifacts:
        return optimizer_e0._OptimizerE0Artifacts(
            report=self._descriptor("canonical_e0", "optimizer_report"),
            raw_evidence=self._descriptor("canonical_e0", "optimizer_raw_evidence"),
            profile_binary=self._descriptor("release", "profile_binary"),
        )

    def _optimizer_result(self, image: str = EXPECTED_IMAGE) -> dict[str, object]:
        source = self._source_archive()
        artifacts = self._artifacts()
        digest = "b" * 64
        fixed37 = {
            "id": "fixed37-production-batch-e0",
            "result": "passed",
            "gate_id": optimizer_contract.FIXED37_PRODUCTION_BATCH_GATE_ID,
            "fixture_sha256": optimizer_contract.EXPECTED_FIXED37_FIXTURE_SHA256,
            "generated_token_ids_sha256": (
                optimizer_contract.EXPECTED_FIXED37_TOKEN_IDS_SHA256
            ),
            "cases": 31,
            "compared_steps": 481,
            "exact_window": 16,
            "fixed_profile": "fixed-contiguous-37-balanced-v1",
            "canonical_profile": "canonical-v1",
            "residual_rmsnorm": "separate",
            "execution_completion": "iteration-batch",
            "fixed_prefill_raw_logit_mismatches": 0,
            "fixed_cached_growing_token_id_mismatches": 0,
            "fixed_cached_growing_cosine_min": (
                optimizer_contract.FIXED37_CACHED_GROWING_COSINE_MIN
            ),
            "fixed_cached_growing_max_abs_max": (
                optimizer_contract.FIXED37_CACHED_GROWING_MAX_ABS_MAX
            ),
            "fixed_cached_growing_mean_abs_max": (
                optimizer_contract.FIXED37_CACHED_GROWING_MEAN_ABS_MAX
            ),
            "fixed_cached_growing_worst_cosine": 0.999,
            "fixed_cached_growing_worst_max_abs": 1.0,
            "fixed_cached_growing_worst_mean_abs": 0.25,
            "fixed_cached_growing_threshold_violations": 0,
            "fixed_golden_token_id_mismatches": 0,
            "canonical_golden_token_id_mismatches": 0,
            "cuda_live_allocation_delta": 0,
            "owner_close_live_allocation_count": 0,
            "compile_command_id": "compile-fixed37-production-batch-e0",
            "execute_command_id": "fixed37-production-batch-e0",
            "compile_log_sha256": digest,
            "test_binary_sha256": digest,
            "log_sha256": digest,
        }
        return {
            "report": {
                "schema_version": 1,
                "gate_id": optimizer_e0.OPTIMIZER_GATE_ID,
                "recorded_at_utc": "2026-08-30T00:00:00Z",
                "status": "passed",
                "semantic_class": "E0",
                "source": {
                    "git_commit": self.gate.fixture.revision,
                    "git_dirty": False,
                    "archive_sha256": source.sha256,
                },
                "build": {
                    "container_image_sha256": image.removeprefix("sha256:"),
                    "network": "none",
                    "cargo_locked": True,
                    "cargo_offline": True,
                    "rustc": "1.85.0",
                    "cuda_toolkit": "12.8.93",
                    "cuda_architecture": "89",
                },
                "gpu": {
                    "model": "NVIDIA GeForce RTX 4090",
                    "uuid": "GPU-fixture",
                    "pci_bus_id": "00000000:01:00.0",
                    "compute_capability": "8.9",
                    "vram_mib": 24564,
                    "driver_version": "580.173.02",
                },
                "model": {
                    **optimizer_contract.EXPECTED_MODEL,
                    "manifest_sha256": digest,
                },
                "implementations": dict(optimizer_contract.EXPECTED_IMPLEMENTATIONS),
                "tests": [
                    {
                        "id": "cuda-compile-only",
                        "result": "passed",
                        "log_sha256": digest,
                    },
                    {
                        "id": "workspace-all-features-all-targets",
                        "result": "passed",
                        "log_sha256": digest,
                    },
                    {
                        "id": "command-batch-lifecycle",
                        "result": "passed",
                        "one_shot_finish": True,
                        "drop_restores_stream": True,
                        "log_sha256": digest,
                    },
                    {
                        "id": "command-batch-resource-ledger",
                        "result": "passed",
                        "validation_fail_closed": True,
                        "queued_chain_raw_byte_mismatches": 0,
                        "cuda_live_allocation_delta": 0,
                        "stream_reuse_after_finish": True,
                        "owner_close_live_allocation_count": 0,
                        "log_sha256": digest,
                    },
                    {
                        "id": "smollm2-multi-step-greedy-exact",
                        "result": "passed",
                        "decode_steps": 16,
                        "committed_iterations": 16,
                        "raw_logit_mismatches": 0,
                        "generated_token_ids": list(optimizer_contract.EXPECTED_TOKENS),
                        "token_id_mismatches": 0,
                        "cuda_live_allocation_delta": 0,
                        "owner_close_live_allocation_count": 0,
                        "log_sha256": digest,
                    },
                    fixed37,
                ],
            },
            "report_sha256": artifacts.report.sha256,
            "raw_evidence_sha256": artifacts.raw_evidence.sha256,
            "profile_binary_sha256": artifacts.profile_binary.sha256,
            "build_image_sha256": image.removeprefix("sha256:"),
            "log_sha256": {"fixed37-production-batch-e0": digest},
            "test_binary_sha256": {"fixed37-production-batch-gpu-test": digest},
        }

    def _replay(self, image: str = EXPECTED_IMAGE) -> dict[str, object]:
        return optimizer_e0.replay_rc3_gate_e_optimizer_e0_v1(
            self.gate.gate_root,
            frozen_candidate_root=self.gate.frozen_root,
            input_evidence_root=self.gate.fixture.evidence,
            repository_root=self.gate.fixture.root,
            expected_optimizer_build_image_id=image,
        )

    def test_verified_private_snapshots_replay_only_optimizer_component(self) -> None:
        artifacts = self._artifacts()
        expected_inputs = {
            "raw_evidence": (self.gate.gate_root / artifacts.raw_evidence.path).read_bytes(),
            "report": (self.gate.gate_root / artifacts.report.path).read_bytes(),
            "profile_binary": (self.gate.gate_root / artifacts.profile_binary.path).read_bytes(),
        }
        snapshots: list[Path] = []

        def replay(
            raw_evidence: Path,
            *,
            report: Path,
            source_revision: str,
            source_archive_sha256: str,
            build_image_id: str,
            profile_binary: Path,
        ) -> dict[str, object]:
            self.assertEqual(source_revision, self.gate.fixture.revision)
            self.assertEqual(source_archive_sha256, self._source_archive().sha256)
            self.assertEqual(build_image_id, EXPECTED_IMAGE)
            observed = {
                "raw_evidence": raw_evidence,
                "report": report,
                "profile_binary": profile_binary,
            }
            for role, path in observed.items():
                self.assertFalse(path.is_relative_to(self.gate.gate_root))
                self.assertEqual(path.read_bytes(), expected_inputs[role])
                snapshots.append(path)
            return self._optimizer_result()

        before = self.gate._snapshot()
        with self.gate._defaults(), mock.patch.object(
            optimizer_e0,
            "_replay_optimizer_raw",
            side_effect=lambda snapshots, **kwargs: replay(
                snapshots.paths["raw_evidence"],
                report=snapshots.paths["report"],
                source_revision=kwargs["source_revision"],
                source_archive_sha256=kwargs["source_archive_sha256"],
                build_image_id=kwargs["expected_build_image_id"],
                profile_binary=snapshots.paths["profile_binary"],
            ),
        ):
            result = self._replay()

        self.assertEqual(self.gate._snapshot(), before)
        self.assertEqual(result["schema_version"], optimizer_e0.REPLAY_VERSION)
        self.assertEqual(result["scope"], optimizer_e0.SCOPE)
        self.assertEqual(result["authority"], optimizer_e0.AUTHORITY)
        self.assertEqual(result["status"], "bound")
        self.assertEqual(result["optimizer_e0_status"], "passed")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["expected_optimizer_build_image_id"], EXPECTED_IMAGE)
        self.assertEqual(result["not_established"], optimizer_e0.NOT_ESTABLISHED)
        self.assertNotIn("gate_e_status", result)
        self.assertNotIn("qualification", result)
        self.assertEqual(result["checks"], [
            {"name": name, "satisfied": True} for name in optimizer_e0.CHECK_NAMES
        ])
        self.assertEqual(len(snapshots), 3)
        self.assertTrue(all(not path.exists() for path in snapshots))

    def test_component_limits_reject_each_input_before_full_gate_stream_replay(self) -> None:
        cases = (
            ("canonical_e0", "optimizer_raw_evidence", optimizer_e0.MAX_OPTIMIZER_RAW_ARCHIVE_BYTES),
            ("canonical_e0", "optimizer_report", optimizer_e0.MAX_OPTIMIZER_REPORT_BYTES),
            ("release", "profile_binary", optimizer_e0.MAX_PROFILE_BINARY_BYTES),
        )
        for group, field, maximum in cases:
            with self.subTest(group=group, field=field):
                record = self.gate.inventory[group]
                self.assertIsInstance(record, dict)
                descriptor = record[field]
                self.assertIsInstance(descriptor, dict)
                original_length = descriptor["byte_length"]
                descriptor["byte_length"] = maximum + 1
                self.gate._write_inventory()
                try:
                    with self.gate._defaults(), mock.patch.object(
                        gate_inputs,
                        "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                        side_effect=AssertionError("full Gate E replay must not start"),
                    ) as structural, self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
                        self._replay()
                    self.assert_reason(raised, "optimizer-e0-input-too-large")
                    structural.assert_not_called()
                finally:
                    descriptor["byte_length"] = original_length
                    self.gate._write_inventory()

    def test_expected_build_image_id_is_required_and_never_learned_from_report(self) -> None:
        for image in ("sha256:" + "0" * 64, "sha256:" + "A" * 64, "not-an-image"):
            with self.subTest(image=image), mock.patch.object(
                gate_inputs,
                "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                side_effect=AssertionError("full Gate E replay must not start"),
            ) as structural, self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
                self._replay(image)
            self.assert_reason(raised, "invalid-expected-optimizer-build-image-id")
            structural.assert_not_called()

    def test_result_cross_binding_rejects_gate_source_and_anchor_mismatches(self) -> None:
        source = self._source_archive()
        artifacts = self._artifacts()
        cases: list[tuple[str, object]] = [
            ("report_sha256", "0" * 64),
            ("raw_evidence_sha256", "0" * 64),
            ("profile_binary_sha256", "0" * 64),
            ("build_image_sha256", "0" * 64),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                result = self._optimizer_result()
                result[field] = value
                with self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
                    optimizer_e0._require_optimizer_result_bindings(
                        result,
                        source_revision=self.gate.fixture.revision,
                        source_archive=source,
                        artifacts=artifacts,
                        expected_build_image_id=EXPECTED_IMAGE,
                    )
                self.assert_reason(raised, f"optimizer-{field}-mismatch")

        report = self._optimizer_result()["report"]
        self.assertIsInstance(report, dict)
        report["source"] = {"git_commit": "0" * 40, "git_dirty": False, "archive_sha256": source.sha256}
        with self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
            optimizer_e0._require_optimizer_result_bindings(
                self._optimizer_result() | {"report": report},
                source_revision=self.gate.fixture.revision,
                source_archive=source,
                artifacts=artifacts,
                expected_build_image_id=EXPECTED_IMAGE,
            )
        self.assert_reason(raised, "optimizer-source-archive-mismatch")

    def test_final_candidate_optimizer_report_contract_is_required(self) -> None:
        source = self._source_archive()
        artifacts = self._artifacts()
        cases = ("implementations", "model")
        for field in cases:
            with self.subTest(field=field):
                result = self._optimizer_result()
                report = result["report"]
                self.assertIsInstance(report, dict)
                del report[field]
                with self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
                    optimizer_e0._require_optimizer_result_bindings(
                        result,
                        source_revision=self.gate.fixture.revision,
                        source_archive=source,
                        artifacts=artifacts,
                        expected_build_image_id=EXPECTED_IMAGE,
                    )
                self.assert_reason(raised, "optimizer-final-report-contract-mismatch")

    def test_fixed37_raw_replay_maps_must_match_the_closed_report(self) -> None:
        source = self._source_archive()
        artifacts = self._artifacts()
        for field in ("log_sha256", "test_binary_sha256"):
            with self.subTest(field=field):
                result = self._optimizer_result()
                binding = result[field]
                self.assertIsInstance(binding, dict)
                binding.clear()
                with self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
                    optimizer_e0._require_optimizer_result_bindings(
                        result,
                        source_revision=self.gate.fixture.revision,
                        source_archive=source,
                        artifacts=artifacts,
                        expected_build_image_id=EXPECTED_IMAGE,
                    )
                self.assert_reason(
                    raised,
                    (
                        "optimizer-fixed37-log-mismatch"
                        if field == "log_sha256"
                        else "optimizer-fixed37-test-binary-mismatch"
                    ),
                )

    def test_private_scratch_replacement_after_legacy_replay_is_rejected(self) -> None:
        def replace_private_snapshot(
            snapshots: optimizer_e0._ScratchSnapshots,
            **_kwargs: object,
        ) -> dict[str, object]:
            target = snapshots.paths["report"]
            replacement = target.with_name("optimizer-e0-private-replacement.tmp")
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, target)
            return self._optimizer_result()

        with self.gate._defaults(), mock.patch.object(
            optimizer_e0,
            "_replay_optimizer_raw",
            side_effect=replace_private_snapshot,
        ), self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "scratch-snapshot-mutated")

    def test_structural_end_replay_rejects_gate_profile_replacement(self) -> None:
        profile = self._artifacts().profile_binary
        profile_path = self.gate.gate_root / profile.path

        def replace_gate_profile(
            _snapshots: optimizer_e0._ScratchSnapshots,
            **_kwargs: object,
        ) -> dict[str, object]:
            replacement = profile_path.with_name("optimizer-e0-gate-replacement.tmp")
            replacement.write_bytes(b"x" * profile_path.stat().st_size)
            os.replace(replacement, profile_path)
            return self._optimizer_result()

        with self.gate._defaults(), mock.patch.object(
            optimizer_e0,
            "_replay_optimizer_raw",
            side_effect=replace_gate_profile,
        ), self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "evidence-hash-mismatch")

    def test_scratch_copy_is_created_only_below_the_pinned_directory_fd(self) -> None:
        source = inspect.getsource(optimizer_e0._snapshot_optimizer_inputs)
        self.assertIn("common.open_private_evidence_directory", source)
        self.assertIn("dir_fd=scratch_fd", source)
        self.assertIn("os.O_EXCL", source)
        self.assertIn("nofollow", source)
        self.assertNotIn("Path.open", source)

    def test_scratch_uses_fixed_external_parent_not_tmpdir_environment(self) -> None:
        source = inspect.getsource(optimizer_e0._replay_rc3_gate_e_optimizer_e0_v1_on_held_fds)
        wrapper = inspect.getsource(optimizer_e0.replay_rc3_gate_e_optimizer_e0_v1)
        self.assertIn("dir=os.fspath(scratch_parent)", source)
        self.assertIn("EXTERNAL_SCRATCH_PARENT", wrapper)
        self.assertIn("optimizer E0 external scratch parent", wrapper)

    def test_bytecode_cache_must_have_been_disabled_before_entry(self) -> None:
        with mock.patch.object(optimizer_e0, "_BYTECODE_DISABLED_AT_STARTUP", False), mock.patch.object(
            optimizer_e0,
            "_BYTECODE_DISABLED_ON_MODULE_ENTRY",
            False,
        ), self.assertRaises(optimizer_e0.OptimizerE0ReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")

    def test_raw_replayer_fails_closed_without_tomllib_compatibility(self) -> None:
        original_import = builtins.__import__

        def reject_optimizer_checker(name: str, *args: object, **kwargs: object) -> object:
            if name == "check_optimization_evidence":
                error = ModuleNotFoundError("No module named 'tomllib'")
                error.name = "tomllib"
                raise error
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_optimizer_checker), self.assertRaises(
            optimizer_e0.OptimizerE0ReplayError
        ) as raised:
            optimizer_e0._replay_optimizer_raw(
                None,  # type: ignore[arg-type]
                source_revision=self.gate.fixture.revision,
                source_archive_sha256=self._source_archive().sha256,
                expected_build_image_id=EXPECTED_IMAGE,
            )
        self.assert_reason(raised, "optimizer-e0-runtime-requires-tomllib")

    def test_schema_forbids_aggregate_gate_e_or_qualification_authority(self) -> None:
        schema_path = (
            Path(optimizer_e0.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-gate-e-optimizer-e0-semantic-replay-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"],
            "https://riley.invalid/benchmarks/release/candidates/"
            "rc3-gate-e-optimizer-e0-semantic-replay-v1.schema.json",
        )
        self.assertFalse(schema["additionalProperties"])
        properties = schema["properties"]
        self.assertEqual(properties["qualification_status"], {"const": "not-run"})
        self.assertEqual(properties["optimizer_e0_status"], {"const": "passed"})
        self.assertNotIn("gate_e_status", properties)
        self.assertNotIn("qualification", properties)
        self.assertEqual(
            properties["expected_optimizer_build_image_id"],
            {"$ref": "#/$defs/imageId"},
        )
        self.assertEqual(
            schema["$defs"]["notEstablished"]["required"],
            list(optimizer_e0.NOT_ESTABLISHED),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
