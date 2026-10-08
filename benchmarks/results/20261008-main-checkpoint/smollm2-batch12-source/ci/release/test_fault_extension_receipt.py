#!/usr/bin/env python3
"""CPU-only adversarial tests for the C02 fault-extension semantic checker."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import check_fault_extension_receipt as fault_extension  # noqa: E402
import check_rc3_qualification as qualification  # noqa: E402
import check_release_candidate as release_candidate  # noqa: E402
from test_cuda_fault_evidence import (  # noqa: E402
    BUILD_IMAGE_ID,
    RELEASE_IMAGE_ID,
    REVISION,
    Fixture as CudaFaultFixture,
)


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class FaultExtensionFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        cuda_root = root / "cuda-fixture"
        cuda_root.mkdir()
        self.cuda = CudaFaultFixture(cuda_root)
        # C02's expanded receipt cannot relabel ordinary Gate E evidence as
        # sanitizer coverage.  The fixture deliberately emits the actual
        # reviewed sanitizer files so the CPU checker can distinguish it from
        # the injectable engine-state traces below.
        self.cuda.enable_sanitizer()
        self.cuda_attestation, raw, cuda_report = self.cuda.produce()

        self.source_relative = "artifacts/source.tar"
        self.binary_relative = "artifacts/riley"
        self.bundle_relative = "artifacts/riley.tar.gz"
        self.cuda_report_relative = "gate-e/cuda-fault-report.json"
        self.cuda_raw_relative = "gate-e/cuda-fault-evidence.tar"
        self.extended_trace_relative = "fault-extension/expanded-raw-trace.json"
        self.receipt_relative = "receipts/fault-extension-input.json"
        self.semantic_report_relative = "reports/fault-extension-check.json"
        self.base_manifest_relative = "candidates/final-release-candidate.json"
        self.base_report_relative = "reports/final-release-candidate.json"
        self.freeze_path = root / "riley-0.1.0-rc3.freeze.json"

        self.source_path = self._copy(self.cuda.source_archive, self.source_relative)
        self.binary_path = self._copy(self.cuda.release_binary, self.binary_relative)
        self.bundle_path = self._copy(self.cuda.release_bundle, self.bundle_relative)
        self.cuda_report_path = self._copy(cuda_report, self.cuda_report_relative)
        self.cuda_raw_path = self._copy(raw, self.cuda_raw_relative)
        self.candidate_id = "riley-0.1.0-rc3"
        self.stable_input = {
            "argv": ["serve", "--fault-extension", "stable-default"],
            "environment": {"RILEY_C02_ARM": "stable-default"},
        }
        maximum_input = {
            "argv": ["serve", "--fault-extension", "max-performance"],
            "environment": {"RILEY_C02_ARM": "max-performance"},
        }
        self.arms = {
            "stable_default": {
                **self.stable_input,
                "configuration_sha256": digest(
                    qualification.canonical_json_bytes(self.stable_input)
                ),
            },
            "max_performance_exact": {
                **maximum_input,
                "configuration_sha256": digest(
                    qualification.canonical_json_bytes(maximum_input)
                ),
            },
        }
        self.freeze = self._freeze()
        self._write_canonical(
            self.evidence / self.base_manifest_relative,
            {"fixture": "release-candidate-manifest"},
        )
        self.base = self._base_report()
        self._write_base()
        self.freeze_raw = self._write_canonical(self.freeze_path, self.freeze)
        self.freeze_sha = digest(self.freeze_raw)
        self.extended_trace = self._extended_trace()
        self._write_extended_trace()
        self.receipt = self._receipt()
        self._write_receipt()

    def _copy(self, source: Path, relative: str) -> Path:
        target = self.evidence / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    def _write_canonical(self, path: Path, document: object) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = qualification.canonical_json_bytes(document)
        path.write_bytes(raw)
        return raw

    def _descriptor(self, relative: str) -> dict[str, str]:
        path = self.evidence / relative
        return {"path": relative, "sha256": digest(path.read_bytes())}

    def _freeze(self) -> dict[str, object]:
        return {
            "schema_version": qualification.FREEZE_VERSION,
            "candidate_id": self.candidate_id,
            "created_at_utc": "2026-08-28T00:00:00Z",
            "status": "frozen",
            "source": {
                "git_revision": REVISION,
                "archive_sha256": digest(self.source_path.read_bytes()),
                "cargo_lock_sha256": digest("cargo lock"),
                "extension_registry_sha256": digest("extension registry"),
                "correctness_golden_sha256": digest("correctness golden"),
            },
            "release": {
                "binary_sha256": digest(self.binary_path.read_bytes()),
                "bundle_sha256": digest(self.bundle_path.read_bytes()),
                "image_id": RELEASE_IMAGE_ID,
                "cuda_c_abi_version": "12.8.1",
            },
            "images": {
                "reproducible": "sha256:" + digest("reproducible image"),
                "cuda": BUILD_IMAGE_ID,
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
            "models": {
                name: {
                    "model_id": f"fixture/{name}",
                    "model_revision": "b" * 40,
                    "config_sha256": digest(f"{name} config"),
                    "weights_sha256": digest(f"{name} weights"),
                    "tokenizer_revision": "c" * 40,
                    "tokenizer_files_sha256": digest(f"{name} tokenizer"),
                }
                for name in ("smollm2", "qwen")
            },
            "arms": self.arms,
            "rollback": {
                "binary_sha256": digest("rollback binary"),
                "bundle_sha256": digest("rollback bundle"),
                "image_id": "sha256:" + digest("rollback image"),
            },
            "outputs": {
                "final_release_candidate_manifest": {"path": self.base_manifest_relative},
                "final_release_candidate": {"path": self.base_report_relative},
                "receipts": {
                    gate: {
                        "path": (
                            self.semantic_report_relative
                            if gate == "fault_extension"
                            else f"receipts/{gate}.json"
                        )
                    }
                    for gate in qualification.REQUIRED_GATES
                },
            },
            "required_gates": list(qualification.REQUIRED_GATES),
        }

    def _base_report(self) -> dict[str, object]:
        evidence_hashes = {
            key: digest(f"Gate E {key}")
            for key in qualification.BASE_EVIDENCE_SHA256_KEYS
        }
        evidence_hashes["cuda_fault"] = digest(self.cuda_report_path.read_bytes())
        evidence_hashes["cuda_fault_raw"] = digest(self.cuda_raw_path.read_bytes())
        return {
            "schema_version": release_candidate.REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "candidate_id": self.candidate_id,
            "manifest_sha256": digest((self.evidence / self.base_manifest_relative).read_bytes()),
            "bindings": {
                "git_revision": REVISION,
                "source_archive_sha256": self.freeze["source"]["archive_sha256"],  # type: ignore[index]
                "release_binary_sha256": self.freeze["release"]["binary_sha256"],  # type: ignore[index]
                "release_bundle_sha256": self.freeze["release"]["bundle_sha256"],  # type: ignore[index]
                "release_image_sha256": RELEASE_IMAGE_ID.removeprefix("sha256:"),
                "build_image_ids": {
                    "reproducible_build": self.freeze["images"]["reproducible"],  # type: ignore[index]
                    "cuda_fault": BUILD_IMAGE_ID,
                    "optimization_correctness": self.freeze["images"]["optimization"],  # type: ignore[index]
                },
                "native_correctness_executable_sha256": digest("native executable"),
                "profile_binary_sha256": digest("profile binary"),
                "reproducibility_report_sha256": digest("repro report"),
                "correctness_golden_sha256": self.freeze["source"]["correctness_golden_sha256"],  # type: ignore[index]
                "evidence_sha256": evidence_hashes,
            },
            "checks": [
                {"name": name, "passed": True} for name in qualification.BASE_CHECKS
            ],
            "errors": [],
        }

    def _write_base(self) -> None:
        self.base_raw = self._write_canonical(
            self.evidence / self.base_report_relative,
            self.base,
        )
        self.base_sha = digest(self.base_raw)

    def _receipt(self) -> dict[str, object]:
        return {
            "schema_version": fault_extension.RECEIPT_VERSION,
            "candidate_id": self.candidate_id,
            "bindings": {
                "freeze_sha256": self.freeze_sha,
                "base_release_candidate_report_sha256": self.base_sha,
                "configuration_profile": fault_extension.STABLE_DEFAULT_PROFILE,
                "configuration_sha256": self.arms["stable_default"]["configuration_sha256"],
            },
            "replay_inputs": {
                "source_archive": self._descriptor(self.source_relative),
                "release_binary": self._descriptor(self.binary_relative),
                "release_bundle": self._descriptor(self.bundle_relative),
            },
            "cuda_fault": {
                "report": self._descriptor(self.cuda_report_relative),
                "raw_evidence": self._descriptor(self.cuda_raw_relative),
            },
            "extended_faults": {
                "raw_trace": self._descriptor(self.extended_trace_relative),
            },
        }

    def _extended_trace(self) -> dict[str, object]:
        return {
            "schema_version": fault_extension.RAW_TRACE_VERSION,
            "candidate_id": self.candidate_id,
            "bindings": {
                "freeze_sha256": self.freeze_sha,
                "base_release_candidate_report_sha256": self.base_sha,
                "configuration_profile": fault_extension.STABLE_DEFAULT_PROFILE,
                "configuration_sha256": self.arms["stable_default"]["configuration_sha256"],
                "source_revision": REVISION,
                "source_archive_sha256": self.freeze["source"]["archive_sha256"],  # type: ignore[index]
                "release_binary_sha256": self.freeze["release"]["binary_sha256"],  # type: ignore[index]
                "release_bundle_sha256": self.freeze["release"]["bundle_sha256"],  # type: ignore[index]
                "release_image_id": RELEASE_IMAGE_ID,
                "cuda_build_image_id": BUILD_IMAGE_ID,
                "gate_e_cuda_fault_raw_sha256": digest(self.cuda_raw_path.read_bytes()),
            },
            "real_gpu_sanitizer": {
                "execution_class": "real-gpu-sanitizer",
                "raw_evidence_sha256": digest(self.cuda_raw_path.read_bytes()),
                "sanitizer_logs": list(fault_extension.SANITIZER_LOGS),
            },
            "injectable_backend": {
                "execution_class": "injectable-synthetic",
                "backend_id": fault_extension.INJECTABLE_BACKEND_ID,
                "subprocess_isolation": True,
            },
            "cases": [
                {
                    "case_id": specification.case_id,
                    "execution_class": "injectable-synthetic",
                    "injection_point": specification.injection_point,
                    "subprocess": {
                        "parent_pid": 1000,
                        "child_pid": 2000 + index,
                        "exit_code": 0,
                    },
                    "events": [
                        {"ordinal": ordinal, "event": event}
                        for ordinal, event in enumerate(specification.events, start=1)
                    ],
                    "terminal": dict(specification.terminal),
                }
                for index, specification in enumerate(fault_extension.EXTENDED_FAULT_CASES)
            ],
        }

    def _write_extended_trace(self) -> None:
        self._write_canonical(
            self.evidence / self.extended_trace_relative,
            self.extended_trace,
        )

    def refresh_extended_trace_and_receipt(self) -> None:
        self._write_extended_trace()
        self.receipt["extended_faults"]["raw_trace"] = self._descriptor(  # type: ignore[index]
            self.extended_trace_relative
        )
        self._write_receipt()

    def _write_receipt(self) -> None:
        self.receipt_raw = self._write_canonical(
            self.evidence / self.receipt_relative,
            self.receipt,
        )

    def refresh_base_and_receipt(self) -> None:
        self._write_base()
        self.receipt["bindings"]["base_release_candidate_report_sha256"] = self.base_sha  # type: ignore[index]
        self.receipt["cuda_fault"]["report"] = self._descriptor(self.cuda_report_relative)  # type: ignore[index]
        self.receipt["cuda_fault"]["raw_evidence"] = self._descriptor(self.cuda_raw_relative)  # type: ignore[index]
        self.extended_trace["bindings"]["base_release_candidate_report_sha256"] = self.base_sha  # type: ignore[index]
        self.extended_trace["bindings"]["gate_e_cuda_fault_raw_sha256"] = digest(  # type: ignore[index]
            self.cuda_raw_path.read_bytes()
        )
        self.extended_trace["real_gpu_sanitizer"]["raw_evidence_sha256"] = digest(  # type: ignore[index]
            self.cuda_raw_path.read_bytes()
        )
        self.refresh_extended_trace_and_receipt()

    def replace_gate_e_with_non_sanitized_raw_evidence(self) -> None:
        """Keep Gate E semantically valid while removing its sanitizer class."""

        (self.cuda.evidence / "environment.txt").write_bytes(
            self.cuda._environment(sanitizer=False)  # type: ignore[attr-defined]
        )
        for name in fault_extension.SANITIZER_LOGS:
            (self.cuda.evidence / name).unlink()
        self.cuda.refresh_checksums()
        self.cuda_attestation, raw, cuda_report = self.cuda.produce()
        shutil.copyfile(raw, self.cuda_raw_path)
        shutil.copyfile(cuda_report, self.cuda_report_path)
        self.base["bindings"]["evidence_sha256"]["cuda_fault"] = digest(  # type: ignore[index]
            self.cuda_report_path.read_bytes()
        )
        self.base["bindings"]["evidence_sha256"]["cuda_fault_raw"] = digest(  # type: ignore[index]
            self.cuda_raw_path.read_bytes()
        )
        self.refresh_base_and_receipt()


class FaultExtensionReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = FaultExtensionFixture(Path(self.temporary.name).resolve())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evaluate_path(self, receipt_path: str) -> dict[str, object]:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ):
            return fault_extension.evaluate(
                self.fixture.freeze_path,
                self.fixture.evidence,
                receipt_path,
                expected_freeze_sha256=self.fixture.freeze_sha,
            )

    def evaluate(self) -> dict[str, object]:
        return self.evaluate_path(self.fixture.receipt_relative)

    def test_known_gate_e_and_expanded_fault_traces_replay_into_a_semantic_pass(self) -> None:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ) as gate_e_replay:
            report = fault_extension.evaluate(
                self.fixture.freeze_path,
                self.fixture.evidence,
                self.fixture.receipt_relative,
                expected_freeze_sha256=self.fixture.freeze_sha,
            )
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["passed"])
        self.assertEqual(
            [
                (case["case_id"], case["execution_class"], case["semantic_check"])
                for case in report["fault_cases"]  # type: ignore[index]
            ],
            list(fault_extension.FAULT_CASES),
        )
        self.assertEqual(
            [row["execution_class"] for row in report["evidence_classes"]],  # type: ignore[index]
            ["real-gpu-sanitizer", "injectable-synthetic"],
        )
        self.assertEqual([row["name"] for row in report["checks"]], list(fault_extension.CHECK_NAMES))  # type: ignore[index]
        parsed = fault_extension.validate_check_report(report)
        self.assertEqual(parsed.candidate_id, self.fixture.candidate_id)
        self.assertEqual(parsed.receipt.path, self.fixture.receipt_relative)
        self.assertEqual(parsed.cuda_fault_raw.path, self.fixture.cuda_raw_relative)
        self.assertEqual(parsed.extended_faults_raw_trace.path, self.fixture.extended_trace_relative)
        self.assertEqual(
            self.fixture.freeze["outputs"]["receipts"]["fault_extension"]["path"],  # type: ignore[index]
            self.fixture.semantic_report_relative,
        )
        self.assertNotEqual(
            self.fixture.receipt_relative,
            self.fixture.semantic_report_relative,
        )
        gate_e_replay.assert_called_once()
        self.assertEqual(self.evaluate_path(parsed.receipt.path), report)

    def test_generic_passed_or_hash_only_envelope_is_not_a_fault_receipt(self) -> None:
        self.fixture.receipt = {
            "schema_version": fault_extension.RECEIPT_VERSION,
            "candidate_id": self.fixture.candidate_id,
            "status": "passed",
            "passed": True,
            "cuda_fault": {"raw_evidence_sha256": digest("raw")},
        }
        self.fixture._write_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_stable_default_arm_drift_is_incomparable(self) -> None:
        self.fixture.receipt["bindings"]["configuration_sha256"] = digest("other arm")  # type: ignore[index]
        self.fixture._write_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])

    def test_gate_e_cuda_hash_binding_cannot_be_replaced(self) -> None:
        self.fixture.base["bindings"]["evidence_sha256"]["cuda_fault_raw"] = digest("other raw")  # type: ignore[index]
        self.fixture.refresh_base_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])

    def test_forged_cuda_passed_attestation_cannot_replace_raw_replay(self) -> None:
        forged = dict(self.fixture.cuda_attestation)
        forged["source"] = dict(forged["source"])  # type: ignore[index]
        forged["source"]["git_revision"] = "f" * 40  # type: ignore[index]
        self.fixture._write_canonical(self.fixture.cuda_report_path, forged)
        self.fixture.base["bindings"]["evidence_sha256"]["cuda_fault"] = digest(  # type: ignore[index]
            self.fixture.cuda_report_path.read_bytes()
        )
        self.fixture.refresh_base_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["cuda-fault-attestation-replay-mismatch"])

    def test_tampered_raw_evidence_cannot_be_rehashed_into_a_pass(self) -> None:
        self.fixture.cuda_raw_path.write_bytes(
            self.fixture.cuda_raw_path.read_bytes() + b"tampered"
        )
        self.fixture.base["bindings"]["evidence_sha256"]["cuda_fault_raw"] = digest(  # type: ignore[index]
            self.fixture.cuda_raw_path.read_bytes()
        )
        self.fixture.refresh_base_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["invalid-input"])

    def test_replay_inputs_must_be_the_frozen_candidate_artifacts(self) -> None:
        other_source = self.fixture.evidence / "artifacts/other-source.tar"
        other_source.write_bytes(b"not the frozen source archive")
        self.fixture.receipt["replay_inputs"]["source_archive"] = {  # type: ignore[index]
            "path": "artifacts/other-source.tar",
            "sha256": digest(other_source.read_bytes()),
        }
        self.fixture._write_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])

    def test_expanded_raw_trace_is_not_optional_or_a_pass_claim(self) -> None:
        self.fixture.receipt.pop("extended_faults")
        self.fixture._write_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

        self.fixture.receipt["extended_faults"] = {  # type: ignore[index]
            "raw_trace": self.fixture._descriptor(self.fixture.extended_trace_relative)
        }
        self.fixture.extended_trace["status"] = "passed"  # type: ignore[index]
        self.fixture.refresh_extended_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_expanded_trace_requires_every_c02_engine_fault_case(self) -> None:
        self.fixture.extended_trace["cases"].pop()  # type: ignore[index]
        self.fixture.refresh_extended_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["expanded-fault-case-inventory"])

    def test_expanded_trace_replays_terminal_state_and_event_order(self) -> None:
        self.fixture.extended_trace["cases"][0]["terminal"]["output_published"] = True  # type: ignore[index]
        self.fixture.refresh_extended_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["expanded-fault-trace-mismatch"])

    def test_expanded_trace_rejects_json_numeric_aliases(self) -> None:
        self.fixture.extended_trace["cases"][0]["subprocess"]["exit_code"] = 0.0  # type: ignore[index]
        self.fixture.refresh_extended_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["expanded-fault-subprocess-failed"])

        third_root = Path(self.temporary.name).resolve() / "third"
        third_root.mkdir()
        self.fixture = FaultExtensionFixture(third_root)
        self.fixture.extended_trace["cases"][0]["terminal"]["terminal_events"] = 1.0  # type: ignore[index]
        self.fixture.refresh_extended_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["expanded-fault-trace-mismatch"])

        second_root = Path(self.temporary.name).resolve() / "second"
        second_root.mkdir()
        self.fixture = FaultExtensionFixture(second_root)
        self.fixture.extended_trace["cases"][2]["events"][2]["event"] = "scheduler-commit-failed"  # type: ignore[index]
        self.fixture.refresh_extended_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["expanded-fault-trace-mismatch"])

    def test_injectable_cases_cannot_be_relabelled_as_real_gpu_evidence(self) -> None:
        self.fixture.extended_trace["cases"][0]["execution_class"] = "real-gpu-sanitizer"  # type: ignore[index]
        self.fixture.refresh_extended_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["invalid-injectable-backend-class"])

    def test_gate_e_must_contain_real_compute_sanitizer_evidence(self) -> None:
        self.fixture.replace_gate_e_with_non_sanitized_raw_evidence()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["missing-real-gpu-sanitizer-evidence"])

    def test_expanded_trace_cannot_bind_a_different_candidate_artifact(self) -> None:
        self.fixture.extended_trace["bindings"]["release_binary_sha256"] = digest("other binary")  # type: ignore[index]
        self.fixture.refresh_extended_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])

    def test_raw_trace_may_not_replace_a_freeze_declared_output(self) -> None:
        self.fixture.receipt["extended_faults"]["raw_trace"] = self.fixture._descriptor(  # type: ignore[index]
            self.fixture.base_report_relative
        )
        self.fixture._write_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["reserved-output-path-collision"])

    def test_raw_receipt_may_not_replace_the_freeze_declared_semantic_report(self) -> None:
        self.fixture._write_canonical(
            self.fixture.evidence / self.fixture.semantic_report_relative,
            self.fixture.receipt,
        )
        report = self.evaluate_path(self.fixture.semantic_report_relative)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["reserved-output-path-collision"])

    def test_failed_gate_e_replay_blocks_the_receipt(self) -> None:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            side_effect=qualification.GateFailure("Gate E did not pass"),
        ):
            report = fault_extension.evaluate(
                self.fixture.freeze_path,
                self.fixture.evidence,
                self.fixture.receipt_relative,
                expected_freeze_sha256=self.fixture.freeze_sha,
            )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["gate-failed"])

    def test_outer_check_report_parser_rejects_a_self_authored_pass(self) -> None:
        with self.assertRaises(qualification.QualificationError):
            fault_extension.validate_check_report(
                {
                    "schema_version": fault_extension.CHECK_REPORT_VERSION,
                    "status": "passed",
                    "passed": True,
                }
            )

    def test_outer_check_report_parser_rejects_evidence_class_drift(self) -> None:
        report = self.evaluate()
        self.assertEqual(report["status"], "passed")
        report["evidence_classes"][1]["source"] = "gate-e-cuda-fault-raw"  # type: ignore[index]
        with self.assertRaises(qualification.QualificationError):
            fault_extension.validate_check_report(report)

    def test_cli_report_is_create_only_after_semantic_replay(self) -> None:
        output = self.fixture.root / "fault-extension-check.json"
        arguments = [
            "--freeze",
            str(self.fixture.freeze_path),
            "--expected-freeze-sha256",
            self.fixture.freeze_sha,
            "--evidence-root",
            str(self.fixture.evidence),
            "--receipt",
            self.fixture.receipt_relative,
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
                self.assertEqual(fault_extension.main(arguments), 0)
        original = output.read_bytes()
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(fault_extension.main(arguments), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_schema_declares_separate_gpu_and_injectable_fault_contracts(self) -> None:
        schema_path = (
            Path(__file__).parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "fault-extension-receipt-v2.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertIn({"$ref": "#/$defs/rawTrace"}, schema["oneOf"])
        self.assertEqual(
            schema["$defs"]["inputReceipt"]["properties"]["schema_version"]["const"],
            fault_extension.RECEIPT_VERSION,
        )
        self.assertEqual(
            schema["$defs"]["rawTrace"]["properties"]["schema_version"]["const"],
            fault_extension.RAW_TRACE_VERSION,
        )
        self.assertEqual(
            tuple(schema["$defs"]["traceCase"]["properties"]["case_id"]["enum"]),
            tuple(case.case_id for case in fault_extension.EXTENDED_FAULT_CASES),
        )
        self.assertEqual(
            schema["$defs"]["realGpuSanitizer"]["properties"]["execution_class"]["const"],
            "real-gpu-sanitizer",
        )
        self.assertEqual(
            schema["$defs"]["injectableBackend"]["properties"]["execution_class"]["const"],
            "injectable-synthetic",
        )
        self.assertEqual(
            schema["$defs"]["checkReport"]["properties"]["fault_cases"]["maxItems"],
            len(fault_extension.FAULT_CASES),
        )


if __name__ == "__main__":
    unittest.main()
