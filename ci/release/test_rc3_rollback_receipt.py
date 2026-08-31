#!/usr/bin/env python3
"""CPU-only adversarial tests for the C02 prior-artifact rollback receipt."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import check_rc3_qualification as qualification
import check_rc3_rollback_receipt as rollback


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class RollbackFixture:
    """One complete local raw drill; Gate E replay stays independently mocked."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.freeze_path = root / "candidate.freeze.json"
        self.candidate_id = "riley-0.1.0-rc3"
        self.base_relative = "reports/final.json"
        self.raw_receipt_relative = "raw/rollback-receipt.json"
        self.drill_relative = "raw/rollback-drill.json"
        stable = {"argv": ["serve", "--execution-completion", "iteration-batch"], "environment": {}}
        maximum = {"argv": ["serve", "--execution-completion", "per-operation"], "environment": {"RILEY_EXACT": "1"}}
        self.arms = {
            "stable_default": {
                **stable,
                "configuration_sha256": digest(qualification.canonical_json_bytes(stable)),
            },
            "max_performance_exact": {
                **maximum,
                "configuration_sha256": digest(qualification.canonical_json_bytes(maximum)),
            },
        }
        self.candidate_artifacts = {
            "binary_sha256": digest("candidate binary"),
            "bundle_sha256": digest("candidate bundle"),
            "image_id": "sha256:" + digest("candidate image"),
        }
        self.rollback_artifacts = {
            "binary_sha256": digest("prior binary"),
            "bundle_sha256": digest("prior bundle"),
            "image_id": "sha256:" + digest("prior image"),
        }
        self.models = {
            name: {
                "model_id": f"fixture/{name}",
                "model_revision": "b" * 40,
                "config_sha256": digest(f"{name} config"),
                "weights_sha256": digest(f"{name} weights"),
                "tokenizer_revision": "c" * 40,
                "tokenizer_files_sha256": digest(f"{name} tokenizer"),
            }
            for name in ("smollm2", "qwen")
        }
        self.freeze: dict[str, object] = {
            "schema_version": qualification.FREEZE_VERSION,
            "candidate_id": self.candidate_id,
            "created_at_utc": "2026-08-28T00:00:00Z",
            "status": "frozen",
            "source": {
                "git_revision": "a" * 40,
                "archive_sha256": digest("source archive"),
                "cargo_lock_sha256": digest("cargo lock"),
                "extension_registry_sha256": digest("extension registry"),
                "correctness_golden_sha256": digest("correctness golden"),
            },
            "release": {**self.candidate_artifacts, "cuda_c_abi_version": "12.8.1"},
            "images": {
                "reproducible": "sha256:" + digest("repro image"),
                "cuda": "sha256:" + digest("cuda image"),
                "optimization": "sha256:" + digest("optimization image"),
            },
            "toolchain": {
                "rustc": "rustc 1.85.0",
                "nvcc": "Cuda compilation tools, release 12.8, V12.8.93",
                "driver": "580.173.02",
                "cuda_runtime": "12.8.1",
                "cuda_toolkit": "12.8.93",
                "cublas": "12.8.4.1",
            },
            "models": self.models,
            "arms": self.arms,
            "rollback": self.rollback_artifacts,
            "outputs": {
                "final_release_candidate_manifest": {"path": "manifests/final.json"},
                "final_release_candidate": {"path": self.base_relative},
                "receipts": {
                    gate: {"path": f"receipts/{gate}.json"}
                    for gate in qualification.REQUIRED_GATES
                },
            },
            "required_gates": list(qualification.REQUIRED_GATES),
        }

    def write_json(self, relative_or_path: str | Path, document: object) -> bytes:
        path = (
            self.evidence / relative_or_path
            if isinstance(relative_or_path, str)
            else relative_or_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = qualification.canonical_json_bytes(document)
        path.write_bytes(encoded)
        return encoded

    def bindings(self) -> dict[str, str]:
        return {
            "freeze_sha256": self.freeze_sha,
            "base_release_candidate_report_sha256": self.base_sha,
            "configuration_profile": rollback.STABLE_DEFAULT_PROFILE,
            "configuration_sha256": self.arms["stable_default"]["configuration_sha256"],
        }

    def drill_document(self) -> dict[str, object]:
        prompt = [11, 12, 13]
        output = [101, 102, 103, 104]
        token_hash = digest(qualification.canonical_json_bytes(output))
        generation = {
            "health_status": 200,
            "output_token_ids": output,
            "output_token_ids_sha256": token_hash,
            "finish_reason": "length",
        }
        zeros = {
            "active_requests": 0,
            "pending_requests": 0,
            "completion_outbox": 0,
            "kv_promised_blocks": 0,
            "kv_active_blocks": 0,
            "riley_owned_live_allocations": 0,
            "worker_processes": 0,
        }
        candidate_worker = "candidate-worker-1"
        candidate_model = "candidate-model-1"
        rollback_worker = "rollback-worker-1"
        rollback_model = "rollback-model-1"
        return {
            "schema_version": rollback.DRILL_VERSION,
            "candidate_id": self.candidate_id,
            "bindings": self.bindings(),
            "model": self.models["smollm2"],
            "candidate_artifacts": self.candidate_artifacts,
            "rollback_artifacts": self.rollback_artifacts,
            "probe": {
                "probe_id": "rollback-greedy-8",
                "prompt_token_ids": prompt,
                "prompt_token_ids_sha256": digest(qualification.canonical_json_bytes(prompt)),
                "expected_output_token_ids": output,
                "expected_output_token_ids_sha256": token_hash,
                "sampling": {"mode": "greedy", "temperature": 0, "top_p": 1},
                "correctness_golden_sha256": self.freeze["source"]["correctness_golden_sha256"],
            },
            "events": [
                {
                    "sequence": 0,
                    "event": "candidate-ready",
                    "worker_id": candidate_worker,
                    "model_instance_id": candidate_model,
                    **generation,
                    "active_requests": 1,
                },
                {
                    "sequence": 1,
                    "event": "candidate-drain-started",
                    "worker_id": candidate_worker,
                    "model_instance_id": candidate_model,
                    "active_requests": 1,
                },
                {
                    "sequence": 2,
                    "event": "candidate-drained",
                    "worker_id": candidate_worker,
                    "model_instance_id": candidate_model,
                    **zeros,
                },
                {
                    "sequence": 3,
                    "event": "atomic-switch",
                    "strategy": "atomic-rename",
                    "from_artifacts": self.candidate_artifacts,
                    "to_artifacts": self.rollback_artifacts,
                },
                {
                    "sequence": 4,
                    "event": "rollback-ready",
                    "worker_id": rollback_worker,
                    "model_instance_id": rollback_model,
                    "health_status": 200,
                },
                {
                    "sequence": 5,
                    "event": "rollback-generation",
                    "worker_id": rollback_worker,
                    "model_instance_id": rollback_model,
                    **generation,
                },
                {
                    "sequence": 6,
                    "event": "candidate-resources-zero",
                    "worker_id": candidate_worker,
                    "model_instance_id": candidate_model,
                    "worker_present": False,
                    "model_present": False,
                    **zeros,
                },
                {
                    "sequence": 7,
                    "event": "rollback-healthy",
                    "worker_id": rollback_worker,
                    "model_instance_id": rollback_model,
                    "health_status": 200,
                    "active_requests": 0,
                },
            ],
        }

    def materialize(self) -> str:
        self.freeze_sha = digest(self.write_json(self.freeze_path, self.freeze))
        self.base_raw = self.write_json(self.base_relative, {"fixture": "gate-e-report"})
        self.base_sha = digest(self.base_raw)
        self.write_json("manifests/final.json", {"fixture": "gate-e-manifest"})
        self.drill = self.drill_document()
        drill_raw = self.write_json(self.drill_relative, self.drill)
        self.receipt = {
            "schema_version": rollback.RECEIPT_VERSION,
            "candidate_id": self.candidate_id,
            "bindings": self.bindings(),
            "drill": {"path": self.drill_relative, "sha256": digest(drill_raw)},
        }
        self.write_json(self.raw_receipt_relative, self.receipt)
        return self.freeze_sha

    def rewrite_drill(self, document: dict[str, object]) -> None:
        self.drill = document
        drill_raw = self.write_json(self.drill_relative, document)
        self.receipt["drill"] = {"path": self.drill_relative, "sha256": digest(drill_raw)}
        self.write_json(self.raw_receipt_relative, self.receipt)

    def evaluate(self) -> dict[str, object]:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.base_raw, self.base_sha),
        ):
            return rollback.evaluate(
                self.freeze_path,
                self.evidence,
                self.raw_receipt_relative,
                expected_freeze_sha256=self.freeze_sha,
            )


class RollbackReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = RollbackFixture(Path(self.temporary.name).resolve())
        self.fixture.materialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_prior_artifact_rollback_drill_passes_and_round_trips(self) -> None:
        report = self.fixture.evaluate()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["passed"])
        self.assertEqual(report["rollback_artifacts"], self.fixture.rollback_artifacts)
        parsed = rollback.validate_check_report(report)
        self.assertEqual(parsed.candidate_id, self.fixture.candidate_id)
        self.assertEqual(parsed.bindings, self.fixture.bindings())

    def test_generic_passed_envelope_is_not_a_rollback_receipt(self) -> None:
        self.fixture.receipt["status"] = "passed"
        self.fixture.write_json(self.fixture.raw_receipt_relative, self.fixture.receipt)
        report = self.fixture.evaluate()
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_rollback_generation_token_drift_fails_even_with_rehashed_raw_evidence(self) -> None:
        drill = copy.deepcopy(self.fixture.drill)
        event = drill["events"][5]
        event["output_token_ids"] = [201, 202, 203, 204]
        event["output_token_ids_sha256"] = digest(
            qualification.canonical_json_bytes(event["output_token_ids"])
        )
        self.fixture.rewrite_drill(drill)
        report = self.fixture.evaluate()
        self.assertEqual(report["reason_codes"], ["rollback-token-mismatch"])

    def test_nonzero_candidate_drain_resource_fails(self) -> None:
        drill = copy.deepcopy(self.fixture.drill)
        drill["events"][2]["kv_active_blocks"] = 1
        self.fixture.rewrite_drill(drill)
        report = self.fixture.evaluate()
        self.assertEqual(report["reason_codes"], ["candidate-not-quiescent"])

    def test_non_atomic_switch_fails(self) -> None:
        drill = copy.deepcopy(self.fixture.drill)
        drill["events"][3]["strategy"] = "best-effort-copy"
        self.fixture.rewrite_drill(drill)
        report = self.fixture.evaluate()
        self.assertEqual(report["reason_codes"], ["non-atomic-rollback-switch"])

    def test_rollback_must_not_reuse_candidate_worker_or_model(self) -> None:
        drill = copy.deepcopy(self.fixture.drill)
        for index in (4, 5, 7):
            drill["events"][index]["worker_id"] = "candidate-worker-1"
            drill["events"][index]["model_instance_id"] = "candidate-model-1"
        self.fixture.rewrite_drill(drill)
        report = self.fixture.evaluate()
        self.assertEqual(report["reason_codes"], ["reused-candidate-instance"])

    def test_raw_drill_cannot_alias_a_freeze_declared_output(self) -> None:
        self.fixture.receipt["drill"]["path"] = "receipts/rollback.json"
        self.fixture.receipt["drill"]["sha256"] = digest("replacement")
        self.fixture.write_json(self.fixture.raw_receipt_relative, self.fixture.receipt)
        report = self.fixture.evaluate()
        self.assertEqual(report["reason_codes"], ["reserved-output-path-collision"])

    def test_check_report_requires_closed_checks_and_nonreused_instances(self) -> None:
        report = self.fixture.evaluate()
        report["checks"] = report["checks"][:-1]
        with self.assertRaises(rollback.RollbackReceiptError):
            rollback.validate_check_report(report)
        report = self.fixture.evaluate()
        report["probe"]["rollback_worker_id"] = report["probe"]["candidate_worker_id"]
        with self.assertRaises(rollback.RollbackReceiptError):
            rollback.validate_check_report(report)

    def test_cli_report_is_create_only(self) -> None:
        output = self.fixture.root / "rollback-check.json"
        arguments = [
            "--freeze",
            str(self.fixture.freeze_path),
            "--expected-freeze-sha256",
            self.fixture.freeze_sha,
            "--evidence-root",
            str(self.fixture.evidence),
            "--receipt",
            self.fixture.raw_receipt_relative,
            "--report",
            str(output),
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(rollback.main(arguments), 0)
        original = output.read_bytes()
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(rollback.main(arguments), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_schema_declares_raw_and_semantic_rollback_documents(self) -> None:
        schema_path = (
            Path(__file__).parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "rollback-receipt-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        variants = {reference["$ref"] for reference in schema["oneOf"]}
        self.assertEqual(
            variants,
            {"#/$defs/receipt", "#/$defs/drill", "#/$defs/checkReport"},
        )
        self.assertEqual(
            schema["$defs"]["checkReport"]["properties"]["schema_version"]["const"],
            rollback.CHECK_REPORT_VERSION,
        )
        self.assertIn("bindings", schema["$defs"]["checkReport"]["required"])


if __name__ == "__main__":
    unittest.main()
