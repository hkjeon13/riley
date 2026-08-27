#!/usr/bin/env python3
"""CPU-only tests for the PR 17 extension-admission gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))

from check_extension_gates import (  # noqa: E402
    APPROVAL_ANSWER_KEYS,
    CLASS_GATE_KEYS,
    CONTRACT_KEYS,
    ENTRY_KEYS,
    EXTENSION_ID,
    EXTENSION_TRACKS,
    IMPLEMENTATION_KEYS,
    PERFORMANCE_METRIC_PATHS,
    PROPOSAL_KEYS,
    QUALITY_METRIC_PATHS,
    REGISTRY_KEYS,
    REPOSITORY_PATH,
    RESULT_METRIC_PATHS,
    RUNTIME_FLAG,
    SEMANTIC_CLASSES,
    TRACK_SEMANTIC_CLASSES,
    TRACK_REQUIRED_METRICS,
    ExtensionGateError,
    _relative_parts,
    validate_repository,
)


SCHEMA_PATHS = (
    "deploy/extensions/registry.schema.json",
    "deploy/extensions/proposal.schema.json",
    "benchmarks/extensions/benchmark-contract.schema.json",
    "deploy/extensions/implementation.schema.json",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _class_gate(semantic_class: str) -> dict[str, object]:
    if semantic_class == "reference":
        return {
            "kind": "reference",
            "behavioral_parity": True,
            "token_parity": True,
            "stable_fallback": True,
            "lifetime_resource_regression": True,
        }
    if semantic_class == "E0":
        return {
            "kind": "E0",
            "reference_parity": True,
            "dtype_tolerances": [{"dtype": "bf16", "atol": 0.01, "rtol": 0.01}],
            "extreme_value_cases": True,
            "token_level_regression": True,
        }
    if semantic_class == "E1":
        return {
            "kind": "E1",
            "distribution_contract": "Target sampling distribution is preserved.",
            "statistical_test": "Predeclared chi-squared test and family-wise alpha.",
            "rng_isolation": True,
            "rng_snapshot_restore": True,
            "greedy_exact": True,
            "fixed_seed_definition": "Seed fixes request-local streams, not branch consumption.",
        }
    if semantic_class == "A1":
        return {
            "kind": "A1",
            "error_budget": {
                "metric": "sparse_attention.omitted_mass_bound",
                "unit": "fraction",
                "maximum": 0.1,
            },
            "exact_fallback": True,
            "opt_in": True,
            "usage_disclosure": True,
            "quality_latency_curve": True,
        }
    if semantic_class == "M1":
        return {
            "kind": "M1",
            "research_track": "Offline calibrated model artifact study.",
            "calibration_or_training_provenance": "Pinned corpus, tool, seed, and source revision.",
            "production_core_isolated": True,
            "opt_in": True,
            "usage_disclosure": True,
            "quality_latency_curve": True,
        }
    raise AssertionError(semantic_class)


class ExtensionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary: tempfile.TemporaryDirectory[str] | None = None
        self._reset_fixture()

    def _reset_fixture(self) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        subprocess.run(
            ["git", "-C", str(self.root), "init", "-q"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for relative in SCHEMA_PATHS:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        self.registry: dict[str, object] = {
            "$schema": "registry.schema.json",
            "schema_version": "rustinfer.extension-registry.v1",
            "extensions": [],
        }
        self._write_registry()
        self._track_all()

    def tearDown(self) -> None:
        assert self.temporary is not None
        self.temporary.cleanup()

    def _write_registry(self) -> None:
        _write_json(self.root / "deploy/extensions/registry.json", self.registry)

    def _track_all(self) -> None:
        subprocess.run(
            ["git", "-C", str(self.root), "add", "--all"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _commit(self, message: str) -> str:
        self._track_all()
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Extension Gate Test",
                "-c",
                "user.email=extension-gate@example.invalid",
                "commit",
                "-q",
                "-m",
                message,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def _add_extension(
        self,
        semantic_class: str = "A1",
        extension_id: str = "example-extension",
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        proposal_path = f"deploy/extensions/proposals/{extension_id}.json"
        plan_path = f"deploy/extensions/plans/{extension_id}.md"
        contract_path = f"benchmarks/extensions/contracts/{extension_id}.json"
        reference_path = f"src/{extension_id}-reference.rs"
        fallback_path = f"src/{extension_id}-fallback.rs"
        workload_path = f"benchmarks/workloads/{extension_id}.json"
        for relative, contents in (
            (reference_path, "// correctness reference fixture\n"),
            (fallback_path, "// stable fallback fixture\n"),
            (workload_path, '{"workload":"end-to-end"}\n'),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        plan = self.root / plan_path
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(f"# {extension_id}\n", encoding="utf-8")

        gate = _class_gate(semantic_class)
        track = {
            "reference": "prefix-cache",
            "E0": "quantization",
            "E1": "speculative-decoding",
            "A1": "query-aware-kv-selection",
            "M1": "jacobi-lookahead",
        }[semantic_class]
        quality_metric = (
            "sparse_attention.omitted_mass_bound"
            if semantic_class == "A1"
            else "failure_count"
        )
        digest = lambda relative: hashlib.sha256(  # noqa: E731
            (self.root / relative).read_bytes()
        ).hexdigest()
        entry: dict[str, object] = {
            "extension_id": extension_id,
            "status": "approved-for-implementation",
            "track": track,
            "semantic_class": semantic_class,
            "proposal_path": proposal_path,
            "deploy_document_path": plan_path,
            "benchmark_contract_path": contract_path,
            "implementation_link_path": None,
        }
        proposal: dict[str, object] = {
            "$schema": "../proposal.schema.json",
            "schema_version": "rustinfer.extension-proposal.v1",
            "extension_id": extension_id,
            "status": entry["status"],
            "track": entry["track"],
            "title": "Example extension",
            "semantic_class": semantic_class,
            "problem_statement": "The reviewed workload has a measured bottleneck.",
            "implementation_boundary": "research" if semantic_class == "M1" else "core",
            "reference_path": reference_path,
            "reference_sha256": digest(reference_path),
            "fallback_path": fallback_path,
            "fallback_sha256": digest(fallback_path),
            "primary_metric": "metrics.output_tokens_per_second",
            "required_metrics": sorted(TRACK_REQUIRED_METRICS[track]),
            "quality_or_error_metric": quality_metric,
            "runtime_flag": f"RUSTINFER_EXPERIMENTAL_{extension_id.upper().replace('-', '_')}",
            "default_enabled": False,
            "stable_default": False,
            "result_disclosure": "Every result records whether the extension ran.",
            "rollback": "Disable the runtime flag and use the exact fallback.",
            "deploy_document_path": plan_path,
            "benchmark_contract_path": contract_path,
            "class_gate": gate,
            "approval_answers": {
                "user_workload_bottleneck": "HBM traffic dominates the reviewed user workload.",
                "semantic_class_rationale": "The declared class matches the output contract.",
                "existing_ir_expression": {
                    "disposition": "existing-ir",
                    "rationale": "The existing linear operation can express this proposal.",
                },
                "implementation_location_rationale": "The reviewed ownership boundary matches the proposal.",
                "correctness_reference": reference_path,
                "error_or_distribution_contract": "The class gate fixes the comparison contract.",
                "memory_and_operational_complexity": "One optional buffer and one rollback flag.",
                "fallback_and_rollback": "Disable the flag and use the exact fallback path.",
                "expected_resource_reduction": ["hbm-traffic"],
                "end_to_end_benefit_hypothesis": "The reviewed workload should improve at equal gates.",
            },
        }
        contract: dict[str, object] = {
            "$schema": "../benchmark-contract.schema.json",
            "schema_version": "rustinfer.extension-benchmark-contract.v1",
            "extension_id": extension_id,
            "status": entry["status"],
            "track": entry["track"],
            "semantic_class": semantic_class,
            "proposal_path": proposal_path,
            "deploy_document_path": plan_path,
            "reference_path": reference_path,
            "reference_sha256": proposal["reference_sha256"],
            "fallback_path": fallback_path,
            "fallback_sha256": proposal["fallback_sha256"],
            "runtime_flag": proposal["runtime_flag"],
            "primary_metric": proposal["primary_metric"],
            "required_metrics": proposal["required_metrics"],
            "quality_or_error_metric": proposal["quality_or_error_metric"],
            "workloads": [{"path": workload_path, "sha256": digest(workload_path)}],
            "comparison_environment": {
                "gpu_count": 1,
                "gpu": "NVIDIA GeForce RTX 4090",
                "driver": "580.65.06",
                "cuda": "13.0",
                "model_id": "example/model",
                "model_revision": "0123456789abcdef0123456789abcdef01234567",
                "dtype": "bf16",
                "concurrency": [1, 8],
                "prompt_tokens": [128, 4096],
                "output_tokens": [32, 128],
                "sampling_configs": [
                    {
                        "id": "greedy",
                        "strategy": "greedy",
                        "temperature": None,
                        "top_p": None,
                        "top_k": None,
                        "seed": None,
                        "ignore_eos": True,
                        "fixed_output_length": True,
                    }
                ],
                "warm_states": ["cold", "warm"],
            },
            "measurement": {
                "independent_process_runs": 5,
                "measured_iterations_per_run": 5,
                "required_statistics": ["median", "p95"],
                "end_to_end": True,
                "fallback_comparison": True,
                "environment_dimensions": [
                    "gpu",
                    "driver",
                    "cuda",
                    "model_id",
                    "model_revision",
                    "dtype",
                    "concurrency",
                    "prompt_output_lengths",
                    "sampling",
                    "warm_state",
                ],
            },
            "class_gate": copy.deepcopy(gate),
        }
        extensions = self.registry["extensions"]
        assert isinstance(extensions, list)
        extensions.append(entry)
        extensions.sort(key=lambda item: str(item["extension_id"]))
        self._write_registry()
        _write_json(self.root / proposal_path, proposal)
        _write_json(self.root / contract_path, contract)
        self._track_all()
        return entry, proposal, contract

    def _add_implementation_link(
        self,
        entry: dict[str, object],
        proposal: dict[str, object],
    ) -> dict[str, object]:
        extension_id = str(entry["extension_id"])
        source_path = f"crates/{extension_id}/src/lib.rs"
        test_path = f"crates/{extension_id}/tests/default_off.rs"
        test_id = "experimental_extension_defaults_off_and_falls_back"
        (self.root / "Cargo.toml").write_text(
            f'[workspace]\nresolver = "3"\nmembers = ["crates/{extension_id}"]\n',
            encoding="utf-8",
        )
        crate_manifest = self.root / f"crates/{extension_id}/Cargo.toml"
        crate_manifest.parent.mkdir(parents=True, exist_ok=True)
        crate_manifest.write_text(
            f'[package]\nname = "{extension_id}"\nversion = "0.0.0"\nedition = "2024"\n',
            encoding="utf-8",
        )
        source = self.root / source_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            f'pub const FLAG: &str = "{proposal["runtime_flag"]}";\n\n'
            "pub fn selected_backend(\n"
            "    flag_value: Option<&str>,\n"
            "    experimental_available: bool,\n"
            ") -> &'static str {\n"
            "    if matches!(flag_value, Some(\"1\") | Some(\"true\"))\n"
            "        && experimental_available\n"
            "    {\n"
            "        \"experimental\"\n"
            "    } else {\n"
            "        \"stable\"\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        test = self.root / test_path
        test.parent.mkdir(parents=True, exist_ok=True)
        crate_identifier = extension_id.replace("-", "_")
        test.write_text(
            f"use {crate_identifier}::{{selected_backend, FLAG}};\n\n"
            "#[test]\n"
            f"fn {test_id}() {{\n"
            f'    assert_eq!(FLAG, "{proposal["runtime_flag"]}");\n'
            '    assert_eq!(selected_backend(None, true), "stable");\n'
            '    assert_eq!(selected_backend(Some("1"), true), "experimental");\n'
            '    assert_eq!(selected_backend(Some("1"), false), "stable");\n'
            "}\n",
            encoding="utf-8",
        )
        implementation_path = f"deploy/extensions/implementations/{extension_id}.json"
        implementation: dict[str, object] = {
            "$schema": "../implementation.schema.json",
            "schema_version": "rustinfer.extension-implementation.v1",
            "extension_id": extension_id,
            "status": "experimental-implementation",
            "proposal_path": entry["proposal_path"],
            "deploy_document_path": entry["deploy_document_path"],
            "benchmark_contract_path": entry["benchmark_contract_path"],
            "runtime_flag": proposal["runtime_flag"],
            "runtime_flag_source_path": source_path,
            "implementation_paths": [source_path, test_path],
            "validation_tests": [
                {
                    "path": test_path,
                    "sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                    "test_id": test_id,
                }
            ],
            "default_enabled": False,
            "stable_default": False,
        }
        entry["implementation_link_path"] = implementation_path
        self._write_registry()
        _write_json(self.root / implementation_path, implementation)
        self._track_all()
        return implementation

    def test_checked_in_registry_count_is_dynamic(self) -> None:
        checked_in = json.loads(
            (ROOT / "deploy/extensions/registry.json").read_text(encoding="utf-8")
        )
        self.assertEqual(validate_repository(ROOT), len(checked_in["extensions"]))
        self.assertEqual(validate_repository(self.root), 0)

    def test_each_semantic_class_has_a_closed_valid_contract(self) -> None:
        for semantic_class in ("reference", "E0", "A1"):
            with self.subTest(semantic_class=semantic_class):
                self._reset_fixture()
                self._add_extension(semantic_class)
                self.assertEqual(validate_repository(self.root), 1)

    def test_classes_without_common_quality_metric_fail_closed(self) -> None:
        for semantic_class in ("E1", "M1"):
            with self.subTest(semantic_class=semantic_class):
                self._reset_fixture()
                self._add_extension(semantic_class)
                with self.assertRaisesRegex(
                    ExtensionGateError, "requires a future common result schema version"
                ):
                    validate_repository(self.root)

    def test_duplicate_json_key_is_rejected(self) -> None:
        path = self.root / "deploy/extensions/registry.json"
        path.write_text(
            '{"$schema":"registry.schema.json",'
            '"schema_version":"rustinfer.extension-registry.v1",'
            '"schema_version":"rustinfer.extension-registry.v1","extensions":[]}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ExtensionGateError, "duplicate JSON key"):
            validate_repository(self.root)

    def test_duplicate_extension_id_is_rejected(self) -> None:
        entry, _, _ = self._add_extension()
        extensions = self.registry["extensions"]
        assert isinstance(extensions, list)
        extensions.append(copy.deepcopy(entry))
        self._write_registry()
        with self.assertRaisesRegex(ExtensionGateError, "duplicate extension_id"):
            validate_repository(self.root)

    def test_unknown_proposal_field_is_rejected(self) -> None:
        entry, proposal, _ = self._add_extension()
        proposal["surprise"] = True
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        with self.assertRaisesRegex(ExtensionGateError, "unknown fields: surprise"):
            validate_repository(self.root)

    def test_unknown_semantic_class_is_rejected(self) -> None:
        entry, _, _ = self._add_extension()
        entry["semantic_class"] = "unknown"
        self._write_registry()
        with self.assertRaisesRegex(ExtensionGateError, "unknown value 'unknown'"):
            validate_repository(self.root)

    def test_status_and_track_are_closed(self) -> None:
        entry, _, _ = self._add_extension()
        entry["status"] = "proposed"
        self._write_registry()
        with self.assertRaisesRegex(ExtensionGateError, "approved-for-implementation"):
            validate_repository(self.root)

        self._reset_fixture()
        entry, _, _ = self._add_extension()
        entry["track"] = "unknown-track"
        self._write_registry()
        with self.assertRaisesRegex(ExtensionGateError, "unknown value 'unknown-track'"):
            validate_repository(self.root)

    def test_all_ten_approval_answers_are_closed(self) -> None:
        entry, proposal, _ = self._add_extension()
        answers = proposal["approval_answers"]
        assert isinstance(answers, dict)
        answers.pop("memory_and_operational_complexity")
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        with self.assertRaisesRegex(
            ExtensionGateError, "missing fields: memory_and_operational_complexity"
        ):
            validate_repository(self.root)

        self._reset_fixture()
        entry, proposal, _ = self._add_extension()
        answers = proposal["approval_answers"]
        assert isinstance(answers, dict)
        answers["expected_resource_reduction"] = ["none"]
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        with self.assertRaisesRegex(ExtensionGateError, "expected_resource_reduction"):
            validate_repository(self.root)

    def test_unregistered_contract_is_rejected(self) -> None:
        orphan = self.root / "benchmarks/extensions/contracts/orphan-contract.json"
        _write_json(orphan, {})
        with self.assertRaisesRegex(ExtensionGateError, "unregistered"):
            validate_repository(self.root)

    def test_path_traversal_and_symlinks_are_rejected(self) -> None:
        entry, proposal, _ = self._add_extension()
        entry["proposal_path"] = "deploy/extensions/proposals/../outside.json"
        self._write_registry()
        with self.assertRaisesRegex(ExtensionGateError, "traversal"):
            validate_repository(self.root)

        entry["proposal_path"] = "deploy/extensions/proposals/example-extension.json"
        self._write_registry()
        proposal_file = self.root / str(entry["proposal_path"])
        proposal_file.unlink()
        outside = self.root / "outside.json"
        _write_json(outside, proposal)
        proposal_file.symlink_to(outside)
        with self.assertRaisesRegex(ExtensionGateError, "symlinks are forbidden"):
            validate_repository(self.root)

    def test_reference_fallback_and_workload_are_tracked_and_hash_bound(self) -> None:
        entry, proposal, _ = self._add_extension()
        untracked = self.root / "src/untracked-reference.rs"
        untracked.write_text("// not in the Git index\n", encoding="utf-8")
        proposal["reference_path"] = "src/untracked-reference.rs"
        proposal["reference_sha256"] = hashlib.sha256(untracked.read_bytes()).hexdigest()
        answers = proposal["approval_answers"]
        assert isinstance(answers, dict)
        answers["correctness_reference"] = proposal["reference_path"]
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        with self.assertRaisesRegex(ExtensionGateError, "must be Git-tracked"):
            validate_repository(self.root)

        self._reset_fixture()
        entry, proposal, _ = self._add_extension()
        reference = self.root / str(proposal["reference_path"])
        reference.write_text("// moved baseline after admission\n", encoding="utf-8")
        with self.assertRaisesRegex(ExtensionGateError, "reference_sha256"):
            validate_repository(self.root)

        self._reset_fixture()
        entry, _, contract = self._add_extension()
        workloads = contract["workloads"]
        assert isinstance(workloads, list)
        workload = workloads[0]
        assert isinstance(workload, dict)
        (self.root / str(workload["path"])).write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ExtensionGateError, r"workloads\[0\]\.sha256"):
            validate_repository(self.root)

    def test_reference_and_fallback_must_be_different(self) -> None:
        entry, proposal, _ = self._add_extension()
        proposal["fallback_path"] = proposal["reference_path"]
        proposal["fallback_sha256"] = proposal["reference_sha256"]
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        with self.assertRaisesRegex(ExtensionGateError, "must be different files"):
            validate_repository(self.root)

    def test_track_class_metric_and_environment_contracts_fail_closed(self) -> None:
        entry, _, _ = self._add_extension()
        entry["semantic_class"] = "E0"
        self._write_registry()
        with self.assertRaisesRegex(ExtensionGateError, "track .* requires"):
            validate_repository(self.root)

        self._reset_fixture()
        entry, proposal, _ = self._add_extension()
        proposal["primary_metric"] = "throughput"
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        with self.assertRaisesRegex(ExtensionGateError, "scalar path"):
            validate_repository(self.root)

        self._reset_fixture()
        entry, _, contract = self._add_extension()
        environment = contract["comparison_environment"]
        assert isinstance(environment, dict)
        environment["gpu_count"] = 2
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(ExtensionGateError, "gpu_count"):
            validate_repository(self.root)

    def test_implementation_link_binds_flag_sources_and_workspace_tests(self) -> None:
        entry, proposal, _ = self._add_extension()
        implementation = self._add_implementation_link(entry, proposal)
        self.assertEqual(validate_repository(self.root), 1)

        implementation["default_enabled"] = True
        _write_json(
            self.root / str(entry["implementation_link_path"]), implementation
        )
        with self.assertRaisesRegex(ExtensionGateError, "default_enabled"):
            validate_repository(self.root)

    def test_implementation_test_must_be_an_unconditional_cargo_target(self) -> None:
        entry, proposal, _ = self._add_extension()
        implementation = self._add_implementation_link(entry, proposal)
        dead_test_path = "crates/example-extension/src/not_linked.rs"
        dead_test = self.root / dead_test_path
        dead_test.write_text(
            "const TEST_ID: &str = "
            '"experimental_extension_defaults_off_and_falls_back";\n',
            encoding="utf-8",
        )
        implementation_paths = implementation["implementation_paths"]
        assert isinstance(implementation_paths, list)
        implementation_paths.append(dead_test_path)
        validation_tests = implementation["validation_tests"]
        assert isinstance(validation_tests, list)
        validation_test = validation_tests[0]
        assert isinstance(validation_test, dict)
        validation_test["path"] = dead_test_path
        validation_test["sha256"] = hashlib.sha256(dead_test.read_bytes()).hexdigest()
        _write_json(
            self.root / str(entry["implementation_link_path"]), implementation
        )
        self._track_all()
        with self.assertRaisesRegex(
            ExtensionGateError, "auto-discovered Rust integration test"
        ):
            validate_repository(self.root)

    def test_validation_test_id_cannot_be_faked_by_tokens_or_literals(self) -> None:
        entry, proposal, _ = self._add_extension()
        implementation = self._add_implementation_link(entry, proposal)
        validation_tests = implementation["validation_tests"]
        assert isinstance(validation_tests, list)
        validation_test = validation_tests[0]
        assert isinstance(validation_test, dict)
        test_file = self.root / str(validation_test["path"])

        for fake_source, expected in (
            (
                'const FAKE: &str = r#"#[test]\n'
                'fn experimental_extension_defaults_off_and_falls_back() {}"#;\n',
                "expected exactly one direct",
            ),
            (
                "macro_rules! fake { () => { #[test] "
                "fn experimental_extension_defaults_off_and_falls_back() {} } }\n",
                "must be declared at target top level",
            ),
        ):
            with self.subTest(expected=expected):
                test_file.write_text(fake_source, encoding="utf-8")
                validation_test["sha256"] = hashlib.sha256(
                    test_file.read_bytes()
                ).hexdigest()
                _write_json(
                    self.root / str(entry["implementation_link_path"]),
                    implementation,
                )
                self._track_all()
                with self.assertRaisesRegex(ExtensionGateError, expected):
                    validate_repository(self.root)

        self._reset_fixture()
        entry, proposal, _ = self._add_extension()
        self._add_implementation_link(entry, proposal)
        crate_manifest = self.root / "crates/example-extension/Cargo.toml"
        crate_manifest.write_text(
            '[package]\nname = "example-extension"\nversion = "0.0.0"\n'
            'edition = "2024"\nautotests = false\n',
            encoding="utf-8",
        )
        self._track_all()
        with self.assertRaisesRegex(ExtensionGateError, "autotests must remain enabled"):
            validate_repository(self.root)

        crate_manifest.write_text(
            '[package]\nname = "example-extension"\nversion = "0.0.0"\n'
            'edition = "2024"\n\n[[test]]\nname = "default_off"\n'
            'path = "./tests/default_off.rs"\nrequired-features = ["cuda"]\n',
            encoding="utf-8",
        )
        self._track_all()
        with self.assertRaisesRegex(
            ExtensionGateError, "validation target cannot require features"
        ):
            validate_repository(self.root)

        self._reset_fixture()
        entry, proposal, _ = self._add_extension()
        implementation = self._add_implementation_link(entry, proposal)
        validation_tests = implementation["validation_tests"]
        assert isinstance(validation_tests, list)
        validation_test = validation_tests[0]
        assert isinstance(validation_test, dict)
        test_file = self.root / str(validation_test["path"])
        test_file.write_text(
            'pub const TEST_ID: &str = "experimental_extension_defaults_off_and_falls_back";\n',
            encoding="utf-8",
        )
        validation_test["sha256"] = hashlib.sha256(test_file.read_bytes()).hexdigest()
        _write_json(
            self.root / str(entry["implementation_link_path"]), implementation
        )
        self._track_all()
        with self.assertRaisesRegex(
            ExtensionGateError, "expected exactly one direct #\\[test\\] function"
        ):
            validate_repository(self.root)

        test_file.write_text(
            "#[cfg(any())]\n#[test]\n"
            "fn experimental_extension_defaults_off_and_falls_back() {}\n",
            encoding="utf-8",
        )
        validation_test["sha256"] = hashlib.sha256(test_file.read_bytes()).hexdigest()
        _write_json(
            self.root / str(entry["implementation_link_path"]), implementation
        )
        self._track_all()
        with self.assertRaisesRegex(ExtensionGateError, "cannot be cfg-gated or ignored"):
            validate_repository(self.root)

        hidden_test_path = "crates/example-extension/tests/.dead.rs"
        hidden_test = self.root / hidden_test_path
        hidden_test.write_text(
            "#[test]\nfn experimental_extension_defaults_off_and_falls_back() {}\n",
            encoding="utf-8",
        )
        implementation_paths = implementation["implementation_paths"]
        assert isinstance(implementation_paths, list)
        implementation_paths.append(hidden_test_path)
        validation_test["path"] = hidden_test_path
        validation_test["sha256"] = hashlib.sha256(hidden_test.read_bytes()).hexdigest()
        _write_json(
            self.root / str(entry["implementation_link_path"]), implementation
        )
        self._track_all()
        with self.assertRaisesRegex(
            ExtensionGateError, "auto-discovered Rust integration test"
        ):
            validate_repository(self.root)

    def test_base_revision_transition_is_append_only_and_admission_only(self) -> None:
        extension_id = "example-extension"
        fixtures = {
            f"src/{extension_id}-reference.rs": "// correctness reference fixture\n",
            f"src/{extension_id}-fallback.rs": "// stable fallback fixture\n",
            f"benchmarks/workloads/{extension_id}.json": '{"workload":"end-to-end"}\n',
        }
        for relative, contents in fixtures.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        base_revision = self._commit("empty registry with reviewed baselines")

        entry, proposal, _ = self._add_extension()
        self.assertEqual(validate_repository(self.root, base_revision), 1)
        admitted_revision = self._commit("admit one extension")

        proposal["title"] = "Moved admission baseline"
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        with self.assertRaisesRegex(ExtensionGateError, "artifact is immutable"):
            validate_repository(self.root, admitted_revision)

        _write_json(self.root / str(entry["proposal_path"]), {**proposal, "title": "Example extension"})
        implementation = self._add_implementation_link(entry, proposal)
        self.assertEqual(validate_repository(self.root, admitted_revision), 1)
        linked_revision = self._commit("link experimental implementation")

        implementation_paths = implementation["implementation_paths"]
        assert isinstance(implementation_paths, list)
        implementation["implementation_paths"] = list(reversed(implementation_paths))
        _write_json(
            self.root / str(entry["implementation_link_path"]), implementation
        )
        with self.assertRaisesRegex(
            ExtensionGateError, "linked implementation artifact is immutable"
        ):
            validate_repository(self.root, linked_revision)

    def test_admission_transition_rejects_unrelated_changes(self) -> None:
        extension_id = "example-extension"
        fixtures = {
            f"src/{extension_id}-reference.rs": "// correctness reference fixture\n",
            f"src/{extension_id}-fallback.rs": "// stable fallback fixture\n",
            f"benchmarks/workloads/{extension_id}.json": '{"workload":"end-to-end"}\n',
        }
        for relative, contents in fixtures.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        base_revision = self._commit("empty registry with reviewed baselines")
        self._add_extension()
        unrelated = self.root / "unrelated.txt"
        unrelated.write_text("implementation mixed into admission\n", encoding="utf-8")
        self._track_all()
        with self.assertRaisesRegex(ExtensionGateError, "admission-only PR"):
            validate_repository(self.root, base_revision)

    def test_admission_transition_counts_rename_sources_as_changes(self) -> None:
        extension_id = "example-extension"
        fixtures = {
            f"src/{extension_id}-reference.rs": "// correctness reference fixture\n",
            f"src/{extension_id}-fallback.rs": "// stable fallback fixture\n",
            f"benchmarks/workloads/{extension_id}.json": '{"workload":"end-to-end"}\n',
            f"preposition/{extension_id}.md": f"# {extension_id}\n",
        }
        for relative, contents in fixtures.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        base_revision = self._commit("prepare a potential rename source")
        self._add_extension()
        (self.root / f"preposition/{extension_id}.md").unlink()
        self._track_all()
        with self.assertRaisesRegex(ExtensionGateError, "admission-only PR"):
            validate_repository(self.root, base_revision)

    def test_new_runtime_flag_requires_an_approved_implementation_link(self) -> None:
        (self.root / "Cargo.toml").write_text(
            '[workspace]\nresolver = "3"\nmembers = ["crates/existing"]\n',
            encoding="utf-8",
        )
        crate_manifest = self.root / "crates/existing/Cargo.toml"
        crate_manifest.parent.mkdir(parents=True, exist_ok=True)
        crate_manifest.write_text(
            '[package]\nname = "existing"\nversion = "0.0.0"\nedition = "2024"\n',
            encoding="utf-8",
        )
        source = self.root / "crates/existing/src/lib.rs"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("pub fn stable() {}\n", encoding="utf-8")
        base_revision = self._commit("empty registry with an existing crate")

        source.write_text(
            'pub const FLAG: &str = "RUSTINFER_EXPERIMENTAL_BYPASS";\n',
            encoding="utf-8",
        )
        self._track_all()
        with self.assertRaisesRegex(
            ExtensionGateError, "new experimental runtime flags require exactly one"
        ):
            validate_repository(self.root, base_revision)

    def test_one_implementation_pr_cannot_link_multiple_extensions(self) -> None:
        first_entry, first_proposal, _ = self._add_extension(
            extension_id="first-extension"
        )
        second_entry, second_proposal, _ = self._add_extension(
            extension_id="second-extension"
        )
        admitted_revision = self._commit("admit two reviewed fixture extensions")

        self._add_implementation_link(first_entry, first_proposal)
        self._add_implementation_link(second_entry, second_proposal)
        (self.root / "Cargo.toml").write_text(
            '[workspace]\nresolver = "3"\n'
            'members = ["crates/first-extension", "crates/second-extension"]\n',
            encoding="utf-8",
        )
        self._track_all()

        with self.assertRaisesRegex(
            ExtensionGateError,
            "one implementation PR may link only one extension",
        ):
            validate_repository(self.root, admitted_revision)

    def test_cross_file_mismatch_is_rejected(self) -> None:
        entry, _, contract = self._add_extension()
        contract["runtime_flag"] = "RUSTINFER_EXPERIMENTAL_DIFFERENT"
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(ExtensionGateError, "proposal mismatch"):
            validate_repository(self.root)

    def test_contract_gate_is_validated_before_type_strict_equality(self) -> None:
        entry, _, contract = self._add_extension("E0")
        gate = contract["class_gate"]
        assert isinstance(gate, dict)
        gate["reference_parity"] = 1
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(ExtensionGateError, "reference_parity"):
            validate_repository(self.root)

    def test_a1_error_budget_metric_matches_quality_metric(self) -> None:
        entry, proposal, contract = self._add_extension("A1")
        for document in (proposal, contract):
            gate = document["class_gate"]
            assert isinstance(gate, dict)
            budget = gate["error_budget"]
            assert isinstance(budget, dict)
            budget["metric"] = "sparse_attention.exact_fallback_rate"
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(
            ExtensionGateError, "must equal quality_or_error_metric"
        ):
            validate_repository(self.root)

    def test_a1_error_budget_uses_fraction_range(self) -> None:
        entry, proposal, contract = self._add_extension("A1")
        for document in (proposal, contract):
            gate = document["class_gate"]
            assert isinstance(gate, dict)
            budget = gate["error_budget"]
            assert isinstance(budget, dict)
            budget["unit"] = "bytes"
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(ExtensionGateError, "error_budget.unit"):
            validate_repository(self.root)

        self._reset_fixture()
        entry, proposal, contract = self._add_extension("A1")
        for document in (proposal, contract):
            gate = document["class_gate"]
            assert isinstance(gate, dict)
            budget = gate["error_budget"]
            assert isinstance(budget, dict)
            budget["maximum"] = 1.0
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(ExtensionGateError, "fraction must be less than 1"):
            validate_repository(self.root)

    def test_e0_tolerance_matches_environment_dtype_and_is_bounded(self) -> None:
        entry, _, contract = self._add_extension("E0")
        environment = contract["comparison_environment"]
        assert isinstance(environment, dict)
        environment["dtype"] = "fp16"
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(
            ExtensionGateError, "must match comparison_environment.dtype exactly"
        ):
            validate_repository(self.root)

        self._reset_fixture()
        entry, proposal, contract = self._add_extension("E0")
        for document in (proposal, contract):
            gate = document["class_gate"]
            assert isinstance(gate, dict)
            tolerances = gate["dtype_tolerances"]
            assert isinstance(tolerances, list)
            tolerances[0]["rtol"] = 1.0
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(ExtensionGateError, "must each be less than 1"):
            validate_repository(self.root)

    def test_primary_metric_is_bound_to_track(self) -> None:
        entry, proposal, contract = self._add_extension("A1")
        proposal["primary_metric"] = "quantization.gemm_throughput_tflops"
        contract["primary_metric"] = proposal["primary_metric"]
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(
            ExtensionGateError, "query-aware-kv-selection requires one of"
        ):
            validate_repository(self.root)

    def test_all_track_required_metrics_are_immutable_contract_fields(self) -> None:
        entry, proposal, contract = self._add_extension("E0")
        required_metrics = proposal["required_metrics"]
        assert isinstance(required_metrics, list)
        proposal["required_metrics"] = required_metrics[:-1]
        contract["required_metrics"] = proposal["required_metrics"]
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(
            ExtensionGateError, "must equal the closed quantization metric set"
        ):
            validate_repository(self.root)

    def test_comparison_environment_requires_model_identity(self) -> None:
        entry, _, contract = self._add_extension()
        environment = contract["comparison_environment"]
        assert isinstance(environment, dict)
        del environment["model_id"]
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(ExtensionGateError, "comparison_environment"):
            validate_repository(self.root)

        self._reset_fixture()
        entry, _, contract = self._add_extension()
        environment = contract["comparison_environment"]
        assert isinstance(environment, dict)
        environment["model_revision"] = "main"
        _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
        with self.assertRaisesRegex(ExtensionGateError, "pinned 40-character"):
            validate_repository(self.root)

    def test_portable_schemas_match_the_closed_checker_shapes(self) -> None:
        registry = json.loads((ROOT / SCHEMA_PATHS[0]).read_text(encoding="utf-8"))
        proposal = json.loads((ROOT / SCHEMA_PATHS[1]).read_text(encoding="utf-8"))
        contract = json.loads((ROOT / SCHEMA_PATHS[2]).read_text(encoding="utf-8"))
        implementation = json.loads((ROOT / SCHEMA_PATHS[3]).read_text(encoding="utf-8"))

        self.assertEqual(set(registry["required"]), REGISTRY_KEYS)
        self.assertEqual(set(registry["properties"]), REGISTRY_KEYS)
        self.assertEqual(set(registry["$defs"]["entry"]["required"]), ENTRY_KEYS)
        self.assertEqual(set(registry["$defs"]["entry"]["properties"]), ENTRY_KEYS)
        self.assertEqual(
            set(registry["$defs"]["entry"]["properties"]["semantic_class"]["enum"]),
            SEMANTIC_CLASSES,
        )
        self.assertEqual(
            set(registry["$defs"]["track"]["enum"]), EXTENSION_TRACKS
        )
        self.assertEqual(
            registry["$defs"]["extensionId"]["pattern"], EXTENSION_ID.pattern
        )

        self.assertEqual(set(proposal["required"]), PROPOSAL_KEYS)
        self.assertEqual(set(proposal["properties"]), PROPOSAL_KEYS)
        self.assertEqual(set(contract["required"]), CONTRACT_KEYS)
        self.assertEqual(set(contract["properties"]), CONTRACT_KEYS)
        for schema in (proposal, contract):
            self.assertEqual(
                schema["$defs"]["extensionId"]["pattern"], EXTENSION_ID.pattern
            )
            self.assertEqual(
                set(schema["$defs"]["semanticClass"]["enum"]), SEMANTIC_CLASSES
            )
            self.assertEqual(set(schema["$defs"]["track"]["enum"]), EXTENSION_TRACKS)
            self.assertEqual(
                set(schema["$defs"]["performanceMetricPath"]["enum"]),
                PERFORMANCE_METRIC_PATHS,
            )
            self.assertEqual(
                set(schema["$defs"]["qualityMetricPath"]["enum"]),
                QUALITY_METRIC_PATHS,
            )
            self.assertEqual(
                schema["properties"]["runtime_flag"]["pattern"],
                RUNTIME_FLAG.pattern,
            )
            for semantic_class, expected_keys in CLASS_GATE_KEYS.items():
                gate = schema["$defs"][f"{semantic_class.lower()}Gate"]
                self.assertEqual(set(gate["required"]), expected_keys)
                self.assertEqual(set(gate["properties"]), expected_keys)

        answers = proposal["$defs"]["approvalAnswers"]
        self.assertEqual(set(answers["required"]), APPROVAL_ANSWER_KEYS)
        self.assertEqual(set(answers["properties"]), APPROVAL_ANSWER_KEYS)
        self.assertEqual(proposal["properties"]["default_enabled"], {"const": False})
        self.assertEqual(proposal["properties"]["stable_default"], {"const": False})
        self.assertEqual(set(implementation["required"]), IMPLEMENTATION_KEYS)
        self.assertEqual(set(implementation["properties"]), IMPLEMENTATION_KEYS)
        self.assertFalse(implementation["additionalProperties"])
        m1_rule = next(
            rule
            for rule in proposal["allOf"]
            if "if" in rule
            and rule["if"]["properties"]["semantic_class"].get("const") == "M1"
        )
        self.assertEqual(
            m1_rule["then"]["properties"]["implementation_boundary"],
            {"const": "research"},
        )

    def test_portable_schema_policy_relaxation_is_rejected(self) -> None:
        path = self.root / "deploy/extensions/proposal.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        schema["additionalProperties"] = True
        _write_json(path, schema)
        with self.assertRaisesRegex(ExtensionGateError, "schema contract digest mismatch"):
            validate_repository(self.root)

    def test_portable_repository_path_pattern_matches_checker_canonicality(self) -> None:
        proposal = json.loads((ROOT / SCHEMA_PATHS[1]).read_text(encoding="utf-8"))
        contract = json.loads((ROOT / SCHEMA_PATHS[2]).read_text(encoding="utf-8"))
        proposal_pattern = proposal["$defs"]["repositoryPath"]["pattern"]
        self.assertEqual(
            proposal_pattern, contract["$defs"]["repositoryPath"]["pattern"]
        )
        self.assertEqual(proposal_pattern, REPOSITORY_PATH.pattern)
        pattern = re.compile(proposal_pattern)
        for path in ("src/reference.rs", "a/.../b", "deploy/extensions/file.json"):
            with self.subTest(path=path, expected="accepted"):
                self.assertIsNotNone(pattern.fullmatch(path))
                self.assertEqual("/".join(_relative_parts(path, "path")), path)
        for path in (
            "/absolute",
            "a\\b",
            ".",
            "..",
            "./a",
            "a/./b",
            "a/../b",
            "a//b",
            "a/",
            "src/reference file.rs",
            "src/reference\tfile.rs",
            "src/참조.rs",
            ".git/HEAD",
            ".gitignore",
            "nested/.gitattributes",
        ):
            with self.subTest(path=path, expected="rejected"):
                self.assertIsNone(pattern.fullmatch(path))
                with self.assertRaises(ExtensionGateError):
                    _relative_parts(path, "path")

    def test_class_specific_fail_closed_rules(self) -> None:
        mutations = {
            "reference": ("stable_fallback", False, "stable_fallback"),
            "E0": ("reference_parity", False, "reference_parity"),
            "E1": ("rng_snapshot_restore", False, "rng_snapshot_restore"),
            "A1": ("exact_fallback", False, "exact_fallback"),
            "M1": ("production_core_isolated", False, "production_core_isolated"),
        }
        for semantic_class, (field, value, message) in mutations.items():
            with self.subTest(semantic_class=semantic_class):
                self._reset_fixture()
                entry, proposal, contract = self._add_extension(semantic_class)
                proposal_gate = proposal["class_gate"]
                contract_gate = contract["class_gate"]
                assert isinstance(proposal_gate, dict)
                assert isinstance(contract_gate, dict)
                proposal_gate[field] = value
                contract_gate[field] = value
                _write_json(self.root / str(entry["proposal_path"]), proposal)
                _write_json(self.root / str(entry["benchmark_contract_path"]), contract)
                with self.assertRaisesRegex(ExtensionGateError, message):
                    validate_repository(self.root)

    def test_m1_cannot_enter_the_production_core_boundary(self) -> None:
        entry, proposal, _ = self._add_extension("M1")
        proposal["implementation_boundary"] = "core"
        _write_json(self.root / str(entry["proposal_path"]), proposal)
        with self.assertRaisesRegex(ExtensionGateError, "M1 requires"):
            validate_repository(self.root)


if __name__ == "__main__":
    unittest.main()
