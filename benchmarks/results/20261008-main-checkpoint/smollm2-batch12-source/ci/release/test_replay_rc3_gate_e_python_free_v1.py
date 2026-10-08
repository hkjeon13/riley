#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 Python-free Gate E adapter."""

from __future__ import annotations

import inspect
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent))

import check_rc3_freeze_input_admission as freeze_inputs  # noqa: E402
import provenance_v2_common as common  # noqa: E402
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs  # noqa: E402
import replay_rc3_gate_e_python_free_v1 as python_free  # noqa: E402
import test_rc3_gate_e_input_inventory_v1 as gate_inventory_tests  # noqa: E402


class Rc3GateEPythonFreeV1Tests(unittest.TestCase):
    """Reuse the closed structural fixture while mocking only legacy semantics."""

    def setUp(self) -> None:
        self.gate = gate_inventory_tests.Rc3GateEInputInventoryV1Tests(
            "test_closed_four_gate_inventory_replays_without_a_gate_e_decision"
        )
        self.gate.setUp()
        self._install_json_leaves()

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
        path = self.gate.gate_root / previous.path
        path.write_bytes(raw)
        descriptor = common.descriptor_for_bytes(previous.path, raw, f"fixture {group}.{field}")
        record[field] = descriptor.as_json()
        self.gate._write_inventory()
        return descriptor

    def _install_json_leaves(self) -> None:
        self._write_leaf("python_free", "report", b'{}\n')
        self._write_leaf("python_free", "correctness_golden", b'{"golden":true}\n')
        self._write_leaf("canonical_e0", "native_report", b'{"native":true}\n')

    def _frozen_request(self) -> dict[str, object]:
        request, _ = freeze_inputs._parse_request(self.gate.fixture.request)
        return request

    def _bindings(self) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        request = self._frozen_request()
        source = request["source"]
        release = request["release"]
        model = request["models"][0]
        self.assertIsInstance(source, dict)
        self.assertIsInstance(release, dict)
        self.assertIsInstance(model, dict)
        return source, release, model

    def _expected_image(self) -> str:
        _source, release, _model = self._bindings()
        container = release["container"]
        self.assertIsInstance(container, dict)
        value = container["image_digest"]
        self.assertIsInstance(value, str)
        return value

    def _expected_golden(self) -> str:
        return self._descriptor("python_free", "correctness_golden").sha256

    def _result_and_archive(self) -> tuple[dict[str, object], dict[str, object]]:
        source, release, frozen_model = self._bindings()
        source_archive = source["archive"]
        release_elf = release["elf"]
        tree = frozen_model["tree"]
        config = frozen_model["config"]
        tokenizer = frozen_model["tokenizer"]
        weights = frozen_model["weights"]
        self.assertIsInstance(source_archive, common.EvidenceDescriptor)
        self.assertIsInstance(release_elf, common.EvidenceDescriptor)
        self.assertIsInstance(tree, common.EvidenceDescriptor)
        self.assertIsInstance(config, common.EvidenceDescriptor)
        self.assertIsInstance(tokenizer, common.EvidenceDescriptor)
        self.assertIsInstance(weights, list)
        self.assertEqual(len(weights), 1)
        weight = weights[0]
        self.assertIsInstance(weight, common.EvidenceDescriptor)
        bundle = self._descriptor("release", "bundle")
        raw = self._descriptor("python_free", "raw_evidence")
        image = self._expected_image()
        report = {
            "schema_version": python_free.PYTHON_FREE_REPORT_SCHEMA,
            "gate": python_free.PYTHON_FREE_GATE,
            "status": "passed",
            "source": {
                "git_revision": self.gate.fixture.revision,
                "git_dirty": False,
                "source_archive_sha256": source_archive.sha256,
                "release_binary_sha256": release_elf.sha256,
                "release_bundle_sha256": bundle.sha256,
                "release_image_sha256": image.removeprefix("sha256:"),
            },
            "raw_evidence_sha256": raw.sha256,
            "checks": [
                {"id": check_id, "passed": True}
                for check_id in python_free.PYTHON_FREE_CHECK_IDS
            ],
        }
        self._write_leaf(
            "python_free",
            "report",
            (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        raw_model = {
            "model_id": frozen_model["model_id"],
            "model_revision": frozen_model["revision"],
            "model_tree_sha256": tree.sha256,
            "config_sha256": config.sha256,
            "weights_sha256": weight.sha256,
            "tokenizer_json_sha256": tokenizer.sha256,
        }
        archive = {
            "raw": {"model": raw_model},
            "model_manifest": b"x" * tree.byte_length,
            "model_sizes": {
                "config.json": config.byte_length,
                "tokenizer.json": tokenizer.byte_length,
                "model.safetensors": weight.byte_length,
            },
        }
        return report, archive

    def _replay(self) -> dict[str, object]:
        return python_free.replay_rc3_gate_e_python_free_v1(
            self.gate.gate_root,
            frozen_candidate_root=self.gate.frozen_root,
            input_evidence_root=self.gate.fixture.evidence,
            repository_root=self.gate.fixture.root,
            expected_release_image_id=self._expected_image(),
            expected_correctness_golden_sha256=self._expected_golden(),
        )

    def test_private_scratch_replays_only_python_free_component(self) -> None:
        report, archive = self._result_and_archive()
        observed: dict[str, Path] = {}

        def replay(snapshots: python_free._ScratchSnapshots, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
            observed["raw"] = snapshots.paths["raw_evidence"]
            observed["bundle"] = snapshots.paths["release_bundle"]
            self.assertEqual(kwargs["source_revision"], self.gate.fixture.revision)
            self.assertEqual(kwargs["expected_release_image_id"], self._expected_image())
            self.assertEqual(kwargs["correctness_golden_sha256"], self._expected_golden())
            self.assertNotEqual(snapshots.paths["raw_evidence"].parent, self.gate.gate_root)
            self.assertNotEqual(snapshots.paths["release_bundle"].parent, self.gate.gate_root)
            self.assertEqual(
                snapshots.paths["raw_evidence"].read_bytes(),
                (self.gate.gate_root / self._descriptor("python_free", "raw_evidence").path).read_bytes(),
            )
            self.assertEqual(
                snapshots.paths["release_bundle"].read_bytes(),
                (self.gate.gate_root / self._descriptor("release", "bundle").path).read_bytes(),
            )
            return report, archive

        before = self.gate._snapshot()
        with self.gate._defaults(), mock.patch.object(
            python_free,
            "_replay_python_free_raw",
            side_effect=replay,
        ):
            result = self._replay()

        self.assertEqual(self.gate._snapshot(), before)
        self.assertEqual(result["schema_version"], python_free.REPLAY_VERSION)
        self.assertEqual(result["scope"], python_free.SCOPE)
        self.assertEqual(result["authority"], python_free.AUTHORITY)
        self.assertEqual(result["status"], "bound")
        self.assertEqual(result["python_free_status"], "passed")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["not_established"], python_free.NOT_ESTABLISHED)
        self.assertEqual(
            result["python_free"]["legacy_replay_retained_byte_limits"],
            {
                "raw_evidence": python_free.MAX_PYTHON_FREE_RAW_RETAINED_BYTES,
                "release_bundle_uncompressed": python_free.MAX_PYTHON_FREE_RELEASE_BUNDLE_RETAINED_BYTES,
            },
        )
        self.assertNotIn("gate_e_status", result)
        self.assertNotIn("qualification", result)
        self.assertEqual(result["checks"], [
            {"name": name, "satisfied": True} for name in python_free.CHECK_NAMES
        ])
        self.assertEqual(set(observed), {"raw", "bundle"})
        self.assertFalse(observed["raw"].exists())
        self.assertFalse(observed["bundle"].exists())

    def test_component_caps_reject_before_full_gate_stream_replay(self) -> None:
        cases = (
            ("python_free", "raw_evidence", python_free.MAX_PYTHON_FREE_RAW_ARCHIVE_BYTES),
            ("python_free", "report", python_free.MAX_PYTHON_FREE_JSON_BYTES),
            ("python_free", "correctness_golden", python_free.MAX_PYTHON_FREE_JSON_BYTES),
            ("release", "bundle", python_free.MAX_RELEASE_BUNDLE_BYTES),
            ("canonical_e0", "native_report", python_free.MAX_PYTHON_FREE_JSON_BYTES),
        )
        for group, field, maximum in cases:
            with self.subTest(group=group, field=field):
                descriptor = self.gate.inventory[group][field]
                self.assertIsInstance(descriptor, dict)
                original_length = descriptor["byte_length"]
                descriptor["byte_length"] = maximum + 1
                self.gate._write_inventory()
                with self.gate._defaults(), mock.patch.object(
                    gate_inputs,
                    "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                    side_effect=AssertionError("full Gate E replay must not start"),
                ) as structural, self.assertRaises(python_free.PythonFreeReplayError) as raised:
                    self._replay()
                self.assert_reason(raised, "python-free-input-too-large")
                structural.assert_not_called()
                descriptor["byte_length"] = original_length
                self.gate._write_inventory()

    def test_external_anchors_are_validated_before_semantic_replay(self) -> None:
        report, archive = self._result_and_archive()
        cases = {
            "image": {"expected_release_image_id": "sha256:" + "0" * 64},
            "golden": {"expected_correctness_golden_sha256": "0" * 64},
        }
        for case, override in cases.items():
            with self.subTest(case=case):
                kwargs = {
                    "expected_release_image_id": self._expected_image(),
                    "expected_correctness_golden_sha256": self._expected_golden(),
                    **override,
                }
                with mock.patch.object(
                    python_free,
                    "_replay_python_free_raw",
                    return_value=(report, archive),
                ) as raw_replay, self.assertRaises(python_free.PythonFreeReplayError) as raised:
                    python_free.replay_rc3_gate_e_python_free_v1(
                        self.gate.gate_root,
                        frozen_candidate_root=self.gate.frozen_root,
                        input_evidence_root=self.gate.fixture.evidence,
                        repository_root=self.gate.fixture.root,
                        **kwargs,
                    )
                self.assert_reason(
                    raised,
                    "invalid-expected-release-image-id"
                    if case == "image"
                    else "invalid-expected-correctness-golden-sha256",
                )
                raw_replay.assert_not_called()

    def test_foreign_release_contract_error_is_fail_closed_by_adapter(self) -> None:
        """The dynamically loaded verifier has a distinct release-common class.

        ``check_python_free_release_e2e.py`` loads one copy of
        ``release_common`` for its own bindings, while ``verify_release_bundle``
        imports the normal module name.  A malformed bundle must still become
        the adapter's closed replay diagnostic rather than escape as a foreign
        exception class.
        """

        tomllib = types.ModuleType("tomllib")
        tomllib.TOMLDecodeError = ValueError
        tomllib.loads = lambda _contents: {}
        script = (
            Path(python_free.__file__).resolve().parents[2]
            / "benchmarks/scripts/check_python_free_release_e2e.py"
        )
        with mock.patch.dict(sys.modules, {"tomllib": tomllib}, clear=False):
            spec = importlib.util.spec_from_file_location(
                "python_free_malformed_bundle_contract_test",
                script,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            contract = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(contract)
            self.assertIsNot(
                contract.release_common.ReleaseContractError,
                contract.release_verify.ReleaseContractError,
            )
            snapshots = types.SimpleNamespace(
                paths={
                    "release_bundle": Path("/var/tmp/nonexistent-python-free-release-bundle.tgz"),
                    "raw_evidence": Path("/var/tmp/not-reached-python-free-raw-evidence.tgz"),
                }
            )
            with mock.patch.object(
                contract.release_verify,
                "verify_bundle",
                side_effect=contract.release_verify.ReleaseContractError("malformed release bundle"),
            ), mock.patch.object(
                python_free,
                "_load_python_free_e2e_contract",
                return_value=contract,
            ), self.assertRaises(python_free.PythonFreeReplayError) as raised:
                python_free._replay_python_free_raw(
                    snapshots,
                    source_revision="a" * 40,
                    source_archive_sha256="b" * 64,
                    release_binary_sha256="c" * 64,
                    release_bundle_sha256="d" * 64,
                    expected_release_image_id="sha256:" + "e" * 64,
                    native_report={},
                    native_report_sha256="f" * 64,
                    correctness_golden_sha256="0" * 64,
                )
        self.assert_reason(raised, "python-free-raw-replay-failed")

    def test_adapter_passes_component_retained_byte_caps_to_legacy_contract(self) -> None:
        observed: dict[str, object] = {}

        class ContractEvidenceError(ValueError):
            pass

        def verify_bundle(_path: Path, **kwargs: object) -> None:
            observed["bundle"] = kwargs

        def load_raw(_path: Path, **kwargs: object) -> dict[str, object]:
            observed["raw"] = kwargs
            return {"raw": {}}

        def validate_raw(_archive: object, **_kwargs: object) -> tuple[dict[str, object], None]:
            return {"status": "passed"}, None

        contract = types.SimpleNamespace(
            EvidenceError=ContractEvidenceError,
            verify_bound_release_bundle=verify_bundle,
            load_raw_evidence_archive=load_raw,
            validate_bound_raw_archive=validate_raw,
        )
        snapshots = types.SimpleNamespace(
            paths={
                "release_bundle": Path("/var/tmp/python-free-release-bundle.tgz"),
                "raw_evidence": Path("/var/tmp/python-free-raw-evidence.tar"),
            }
        )
        with mock.patch.object(
            python_free,
            "_load_python_free_e2e_contract",
            return_value=contract,
        ):
            replayed, archive = python_free._replay_python_free_raw(
                snapshots,
                source_revision="a" * 40,
                source_archive_sha256="b" * 64,
                release_binary_sha256="c" * 64,
                release_bundle_sha256="d" * 64,
                expected_release_image_id="sha256:" + "e" * 64,
                native_report={},
                native_report_sha256="f" * 64,
                correctness_golden_sha256="0" * 64,
            )
        self.assertEqual(replayed, {"status": "passed"})
        self.assertEqual(archive, {"raw": {}})
        self.assertEqual(
            observed["bundle"],
            {
                "release_binary_sha256": "c" * 64,
                "source_revision": "a" * 40,
                "max_uncompressed_bytes": python_free.MAX_PYTHON_FREE_RELEASE_BUNDLE_RETAINED_BYTES,
            },
        )
        self.assertEqual(
            observed["raw"],
            {"max_retained_bytes": python_free.MAX_PYTHON_FREE_RAW_RETAINED_BYTES},
        )

    def test_raw_report_mismatch_fails_closed(self) -> None:
        report, archive = self._result_and_archive()
        report = {**report, "checks": list(reversed(report["checks"]))}
        with self.gate._defaults(), mock.patch.object(
            python_free,
            "_replay_python_free_raw",
            return_value=(report, archive),
        ), self.assertRaises(python_free.PythonFreeReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "python-free-report-mismatch")

    def test_numeric_json_values_cannot_substitute_for_report_booleans(self) -> None:
        report, archive = self._result_and_archive()
        submitted = {
            **report,
            "source": {**report["source"], "git_dirty": 0},
            "checks": [
                {**check, "passed": 1}
                for check in report["checks"]
            ],
        }
        self._write_leaf(
            "python_free",
            "report",
            (json.dumps(submitted, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        with self.gate._defaults(), mock.patch.object(
            python_free,
            "_replay_python_free_raw",
            return_value=(report, archive),
        ), self.assertRaises(python_free.PythonFreeReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "python-free-report-contract-mismatch")

    def test_frozen_model_mismatch_fails_closed(self) -> None:
        _report, archive = self._result_and_archive()
        request = self._frozen_request()
        source = request["source"]
        release = request["release"]
        self.assertIsInstance(source, dict)
        self.assertIsInstance(release, dict)
        source_archive = source["archive"]
        release_elf = release["elf"]
        container = release["container"]
        self.assertIsInstance(source_archive, common.EvidenceDescriptor)
        self.assertIsInstance(release_elf, common.EvidenceDescriptor)
        self.assertIsInstance(container, dict)
        bindings = python_free._FrozenBindings(
            source_archive=source_archive,
            release_elf=release_elf,
            release_image_digest=container["image_digest"],
            models=tuple(request["models"]),
        )
        archive["raw"]["model"]["weights_sha256"] = "f" * 64
        with self.assertRaises(python_free.PythonFreeReplayError) as raised:
            python_free._require_frozen_model_binding(archive, bindings)
        self.assert_reason(raised, "python-free-frozen-model-mismatch")

    def test_multi_weight_frozen_model_cannot_authorize_single_weight_raw_archive(self) -> None:
        _report, archive = self._result_and_archive()
        request = self._frozen_request()
        source = request["source"]
        release = request["release"]
        model = request["models"][0]
        self.assertIsInstance(source, dict)
        self.assertIsInstance(release, dict)
        self.assertIsInstance(model, dict)
        source_archive = source["archive"]
        release_elf = release["elf"]
        container = release["container"]
        weights = model["weights"]
        self.assertIsInstance(source_archive, common.EvidenceDescriptor)
        self.assertIsInstance(release_elf, common.EvidenceDescriptor)
        self.assertIsInstance(container, dict)
        self.assertIsInstance(weights, list)
        self.assertEqual(len(weights), 1)
        bindings = python_free._FrozenBindings(
            source_archive=source_archive,
            release_elf=release_elf,
            release_image_digest=container["image_digest"],
            models=({**model, "weights": [*weights, weights[0]]},),
        )
        with self.assertRaises(python_free.PythonFreeReplayError) as raised:
            python_free._require_frozen_model_binding(archive, bindings)
        self.assert_reason(raised, "python-free-frozen-model-mismatch")

    def test_private_scratch_replacement_is_rejected(self) -> None:
        report, archive = self._result_and_archive()

        def replace(snapshots: python_free._ScratchSnapshots, **_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
            target = snapshots.paths["raw_evidence"]
            replacement = target.with_name("python-free-private-replacement.tmp")
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, target)
            return report, archive

        with self.gate._defaults(), mock.patch.object(
            python_free,
            "_replay_python_free_raw",
            side_effect=replace,
        ), self.assertRaises(python_free.PythonFreeReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "scratch-snapshot-mutated")

    def test_gate_leaf_replacement_during_replay_is_rejected(self) -> None:
        report, archive = self._result_and_archive()
        descriptor = self._descriptor("python_free", "raw_evidence")
        target = self.gate.gate_root / descriptor.path

        def replace(_snapshots: python_free._ScratchSnapshots, **_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
            replacement = target.with_name("python-free-gate-replacement.tmp")
            replacement.write_bytes(b"x" * target.stat().st_size)
            os.replace(replacement, target)
            return report, archive

        with self.gate._defaults(), mock.patch.object(
            python_free,
            "_replay_python_free_raw",
            side_effect=replace,
        ), self.assertRaises(python_free.PythonFreeReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "evidence-hash-mismatch")

    def test_static_scratch_and_scope_guards_are_present(self) -> None:
        snapshot = inspect.getsource(python_free._snapshot_python_free_inputs)
        module = inspect.getsource(python_free)
        self.assertIn("common.open_private_evidence_directory", snapshot)
        self.assertIn("dir_fd=scratch_fd", snapshot)
        self.assertIn("os.O_EXCL", snapshot)
        self.assertIn("nofollow", snapshot)
        self.assertEqual(python_free.EXTERNAL_SCRATCH_PARENT, Path("/var/tmp"))
        self.assertNotIn("check_release_candidate", module)

    def test_schema_forbids_aggregate_gate_e_or_qualification_authority(self) -> None:
        schema_path = (
            Path(python_free.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-gate-e-python-free-semantic-replay-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"],
            "https://riley.invalid/benchmarks/release/candidates/"
            "rc3-gate-e-python-free-semantic-replay-v1.schema.json",
        )
        self.assertFalse(schema["additionalProperties"])
        properties = schema["properties"]
        self.assertEqual(properties["qualification_status"], {"const": "not-run"})
        self.assertEqual(properties["python_free_status"], {"const": "passed"})
        descriptor_path = schema["$defs"]["descriptor"]["properties"]["path"]
        self.assertEqual(descriptor_path["allOf"][0], {"pattern": "^[A-Za-z0-9][A-Za-z0-9._/-]*$"})
        self.assertEqual(descriptor_path["allOf"][1], {"not": {"pattern": "(^|/)\\.\\.?($|/)"}})
        self.assertEqual(descriptor_path["allOf"][2], {"not": {"pattern": "//"}})
        self.assertEqual(descriptor_path["allOf"][3], {"not": {"pattern": "/$"}})
        limits = properties["python_free"]["properties"]["legacy_replay_retained_byte_limits"]
        self.assertEqual(
            limits["properties"]["raw_evidence"],
            {"const": python_free.MAX_PYTHON_FREE_RAW_RETAINED_BYTES},
        )
        self.assertEqual(
            limits["properties"]["release_bundle_uncompressed"],
            {"const": python_free.MAX_PYTHON_FREE_RELEASE_BUNDLE_RETAINED_BYTES},
        )
        self.assertNotIn("gate_e_status", properties)
        self.assertNotIn("qualification", properties)
        self.assertEqual(
            schema["$defs"]["notEstablished"]["required"],
            list(python_free.NOT_ESTABLISHED),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
