#!/usr/bin/env python3
"""CPU-only adversarial tests for the C02 RC3 qualification envelope."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import check_rc3_qualification as qualification
import check_effective_runtime_config_receipt as runtime_config
import check_fault_extension_receipt as fault_extension
import check_qwen_multistep_receipt as qwen_multistep
import check_rc3_routing_receipt as routing
import check_rc3_rollback_receipt as rollback
import check_soak_v2_receipt as soak
import check_release_candidate as release_candidate
import write_rc3_candidate_freeze as freeze_writer


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class QualificationFixture:
    """A deliberately non-runnable Gate E fixture for outer-envelope tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.freeze_path = root / "riley-0.1.0-rc3.freeze.json"
        self.candidate_id = "riley-0.1.0-rc3"
        self.revision = "a" * 40
        self.release = {
            "binary_sha256": digest("release binary"),
            "bundle_sha256": digest("release bundle"),
            "image_id": "sha256:" + digest("release image"),
            "cuda_c_abi_version": "12.8.1",
        }
        self.images = {
            "reproducible": "sha256:" + digest("repro image"),
            "cuda": "sha256:" + digest("cuda image"),
            "optimization": "sha256:" + digest("optimization image"),
        }
        stable = {
            "argv": ["serve", "--execution-completion", "iteration-batch"],
            "environment": {},
        }
        maximum = {
            "argv": ["serve", "--execution-completion", "per-operation"],
            "environment": {"RILEY_EXACT": "1"},
        }
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
        self.freeze = {
            "schema_version": qualification.FREEZE_VERSION,
            "candidate_id": self.candidate_id,
            "created_at_utc": "2026-08-28T00:00:00Z",
            "status": "frozen",
            "source": {
                "git_revision": self.revision,
                "archive_sha256": digest("source archive"),
                "cargo_lock_sha256": digest("cargo lock"),
                "extension_registry_sha256": digest("extension registry"),
                "correctness_golden_sha256": digest("correctness golden"),
            },
            "release": self.release,
            "images": self.images,
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
                "binary_sha256": digest("prior binary"),
                "bundle_sha256": digest("prior bundle"),
                "image_id": "sha256:" + digest("prior image"),
            },
            "outputs": {
                "final_release_candidate_manifest": {"path": "manifests/final.json"},
                "final_release_candidate": {"path": "reports/final.json"},
                "receipts": {
                    gate: {"path": f"receipts/{gate}.json"}
                    for gate in qualification.REQUIRED_GATES
                },
            },
            "required_gates": list(qualification.REQUIRED_GATES),
        }

    def write_json(self, relative: str, document: object) -> bytes:
        path = self.evidence / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = qualification.canonical_json_bytes(document)
        path.write_bytes(encoded)
        return encoded

    def base_report(self, manifest_sha256: str) -> dict[str, object]:
        return {
            "schema_version": release_candidate.REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "candidate_id": self.candidate_id,
            "manifest_sha256": manifest_sha256,
            "bindings": {
                "git_revision": self.revision,
                "source_archive_sha256": self.freeze["source"]["archive_sha256"],
                "release_binary_sha256": self.release["binary_sha256"],
                "release_bundle_sha256": self.release["bundle_sha256"],
                "release_image_sha256": self.release["image_id"].removeprefix("sha256:"),
                "build_image_ids": {
                    "reproducible_build": self.images["reproducible"],
                    "cuda_fault": self.images["cuda"],
                    "optimization_correctness": self.images["optimization"],
                },
                "native_correctness_executable_sha256": digest("native executable"),
                "profile_binary_sha256": digest("profile binary"),
                "reproducibility_report_sha256": digest("repro report"),
                "correctness_golden_sha256": self.freeze["source"]["correctness_golden_sha256"],
                "evidence_sha256": {
                    name: digest(f"fixture evidence {name}")
                    for name in qualification.BASE_EVIDENCE_SHA256_KEYS
                },
            },
            "checks": [{"name": name, "passed": True} for name in qualification.BASE_CHECKS],
            "errors": [],
        }

    def materialize(self) -> str:
        manifest_raw = self.write_json("manifests/final.json", {"fixture": "not-a-real-gate-e-manifest"})
        base_raw = self.write_json("reports/final.json", self.base_report(digest(manifest_raw)))
        self.freeze_path.write_bytes(qualification.canonical_json_bytes(self.freeze))
        freeze_sha = digest(self.freeze_path.read_bytes())
        base_sha = digest(base_raw)
        for gate in qualification.REQUIRED_GATES:
            payload = f"evidence for {gate}\n".encode("ascii")
            evidence_relative = f"payloads/{gate}.txt"
            path = self.evidence / evidence_relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            self.write_json(
                f"receipts/{gate}.json",
                {
                    "schema_version": qualification.RECEIPT_VERSION,
                    "gate": gate,
                    "status": "passed",
                    "passed": True,
                    "candidate_id": self.candidate_id,
                    "bindings": {
                        "freeze_sha256": freeze_sha,
                        "base_release_candidate_report_sha256": base_sha,
                        "configuration_profile": "stable-default",
                        "configuration_sha256": self.arms["stable_default"]["configuration_sha256"],
                    },
                    "evidence": {"path": evidence_relative, "sha256": digest(payload)},
                },
            )
        return freeze_sha


class Rc3QualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = QualificationFixture(Path(self.temporary.name).resolve())
        self.freeze_sha = self.fixture.materialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evaluate(self, *, repository_root: Path | None = None) -> dict[str, object]:
        return qualification.evaluate(
            self.fixture.freeze_path,
            self.fixture.evidence,
            expected_candidate_sha256=self.freeze_sha,
            repository_root=repository_root,
        )

    def _evaluate_with_replayed_gate_e(self) -> dict[str, object]:
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed = self.fixture.base_report(digest(manifest_raw))
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed),
        ):
            return self.evaluate(repository_root=self.fixture.root)

    def _startup_configuration_report(self) -> dict[str, object]:
        base_raw = (self.fixture.evidence / "reports/final.json").read_bytes()
        arm_names = {
            runtime_config.STABLE_DEFAULT_PROFILE: "stable_default",
            runtime_config.MAX_PERFORMANCE_EXACT_PROFILE: "max_performance_exact",
        }
        return {
            "schema_version": runtime_config.CHECK_REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "candidate_id": self.fixture.candidate_id,
            "freeze_sha256": self.freeze_sha,
            "base_release_candidate_report": {
                "path": "reports/final.json",
                "sha256": digest(base_raw),
            },
            "stable_promotion_profile": runtime_config.STABLE_DEFAULT_PROFILE,
            "arms": {
                profile: {
                    "configuration_profile": profile,
                    "configuration_sha256": self.fixture.arms[arm_name][
                        "configuration_sha256"
                    ],
                    "endpoint_payload": {
                        "path": f"startup/{profile}-v1-config.json",
                        "sha256": digest(f"{profile} endpoint"),
                    },
                    "startup_artifact": {
                        "path": f"startup/{profile}-startup-config.json",
                        "sha256": digest(f"{profile} startup artifact"),
                    },
                    "effective_config_sha256": digest(f"{profile} effective config"),
                }
                for profile, arm_name in arm_names.items()
            },
            "checks": [{"name": name, "passed": True} for name in runtime_config.CHECK_NAMES],
            "reason_codes": [],
        }

    def _qwen_multistep_report(self) -> dict[str, object]:
        """Minimal outer-envelope fixture; Qwen validates its full shape elsewhere."""

        return {
            "schema_version": qwen_multistep.CHECK_REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "fixture": "qwen",
        }

    def _parsed_qwen_multistep_report(self) -> qwen_multistep.QwenCheckReport:
        base_raw = (self.fixture.evidence / "reports/final.json").read_bytes()
        return qwen_multistep.QwenCheckReport(
            candidate_id=self.fixture.candidate_id,
            freeze_sha256=self.freeze_sha,
            base_release_candidate_report=qwen_multistep.Descriptor(
                "reports/final.json", digest(base_raw)
            ),
            bindings={
                "freeze_sha256": self.freeze_sha,
                "base_release_candidate_report_sha256": digest(base_raw),
                "configuration_profile": qwen_multistep.STABLE_DEFAULT_PROFILE,
                "configuration_sha256": self.fixture.arms["stable_default"][
                    "configuration_sha256"
                ],
            },
            golden=qwen_multistep.Descriptor(
                qwen_multistep.GOLDEN_RELATIVE_PATH,
                qwen_multistep.GOLDEN_SHA256,
            ),
            wire=qwen_multistep.Descriptor(
                qwen_multistep.WIRE_RELATIVE_PATH,
                qwen_multistep.WIRE_SHA256,
            ),
            receipt=qwen_multistep.Descriptor("raw/qwen-receipt.json", digest("qwen raw")),
            case_manifest=qwen_multistep.Descriptor(
                "raw/qwen-cases.json", digest("qwen cases")
            ),
            model=self.fixture.freeze["models"]["qwen"],
            cases=(),
        )

    def _routing_report(self) -> dict[str, object]:
        """Minimal outer-envelope fixture; routing validates its full shape elsewhere."""

        return {
            "schema_version": routing.CHECK_REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "fixture": "routing",
        }

    def _parsed_routing_report(self) -> routing.RoutingCheckReport:
        base_raw = (self.fixture.evidence / "reports/final.json").read_bytes()
        return routing.RoutingCheckReport(
            candidate_id=self.fixture.candidate_id,
            freeze_sha256=self.freeze_sha,
            base_release_candidate_report=routing.Descriptor("reports/final.json", digest(base_raw)),
            bindings={
                "freeze_sha256": self.freeze_sha,
                "base_release_candidate_report_sha256": digest(base_raw),
                "configuration_profile": routing.STABLE_DEFAULT_PROFILE,
                "configuration_sha256": self.fixture.arms["stable_default"][
                    "configuration_sha256"
                ],
            },
            corpus=routing.Descriptor(routing.CORPUS_RELATIVE_PATH, routing.CORPUS_SHA256),
            receipt=routing.Descriptor("raw/routing-receipt.json", digest("routing raw")),
            trace_manifest=routing.Descriptor("raw/routing-traces.json", digest("routing traces")),
            model=self.fixture.freeze["models"]["smollm2"],
            execution={
                "source_revision": self.fixture.revision,
                "source_archive_sha256": self.fixture.freeze["source"]["archive_sha256"],
                "release_binary_sha256": self.fixture.release["binary_sha256"],
                "release_bundle_sha256": self.fixture.release["bundle_sha256"],
                "release_image_id": self.fixture.release["image_id"],
                "test_image_id": self.fixture.images["cuda"],
                "test_executable": {
                    "path": "raw/riley-release",
                    "sha256": self.fixture.release["binary_sha256"],
                },
            },
            traces=(),
        )

    def _fault_extension_report(self) -> dict[str, object]:
        """Minimal outer-envelope fixture; fault validates its full shape elsewhere."""

        return {
            "schema_version": fault_extension.CHECK_REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "fixture": "fault-extension",
        }

    def _parsed_fault_extension_report(
        self,
    ) -> fault_extension.FaultExtensionCheckReport:
        base_raw = (self.fixture.evidence / "reports/final.json").read_bytes()
        return fault_extension.FaultExtensionCheckReport(
            candidate_id=self.fixture.candidate_id,
            freeze_sha256=self.freeze_sha,
            base_release_candidate_report=fault_extension.Descriptor(
                "reports/final.json", digest(base_raw)
            ),
            receipt=fault_extension.Descriptor(
                "raw/fault-extension-receipt.json", digest("fault-extension raw receipt")
            ),
            bindings={
                "freeze_sha256": self.freeze_sha,
                "base_release_candidate_report_sha256": digest(base_raw),
                "configuration_profile": fault_extension.STABLE_DEFAULT_PROFILE,
                "configuration_sha256": self.fixture.arms["stable_default"][
                    "configuration_sha256"
                ],
            },
            replay_inputs=fault_extension.ReplayInputs(
                source_archive=fault_extension.Descriptor(
                    "raw/fault-source.tar", self.fixture.freeze["source"]["archive_sha256"]
                ),
                release_binary=fault_extension.Descriptor(
                    "raw/fault-riley", self.fixture.release["binary_sha256"]
                ),
                release_bundle=fault_extension.Descriptor(
                    "raw/fault-riley.tar.gz", self.fixture.release["bundle_sha256"]
                ),
            ),
            cuda_fault_report=fault_extension.Descriptor(
                "raw/fault-gate-e-report.json", digest("fault Gate E report")
            ),
            cuda_fault_raw=fault_extension.Descriptor(
                "raw/fault-gate-e-raw.tar", digest("fault Gate E raw")
            ),
            extended_faults_raw_trace=fault_extension.Descriptor(
                "raw/fault-extension-trace.json", digest("fault extension trace")
            ),
        )

    def _fault_extension_outer_inputs(self) -> tuple[qualification.FrozenCandidate, str]:
        frozen = qualification._validate_freeze(
            qualification._parse_document(
                self.fixture.freeze_path.read_bytes(), "outer fault-extension fixture freeze"
            )
        )
        return frozen, digest((self.fixture.evidence / "reports/final.json").read_bytes())

    def _soak_v2_report(self) -> dict[str, object]:
        """Minimal outer-envelope fixture; soak-v2 validates its full shape elsewhere."""

        return {
            "schema_version": soak.CHECK_REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "fixture": "soak-v2",
        }

    def _parsed_soak_v2_report(self) -> soak.SoakV2CheckReport:
        base_raw = (self.fixture.evidence / "reports/final.json").read_bytes()
        return soak.SoakV2CheckReport(
            candidate_id=self.fixture.candidate_id,
            freeze_sha256=self.freeze_sha,
            base_release_candidate_report=soak.Descriptor("reports/final.json", digest(base_raw)),
            scenario_contract=soak.Descriptor(soak.CONTRACT_RELATIVE_PATH, soak.CONTRACT_SHA256),
            receipt=soak.Descriptor("raw/soak-v2-receipt.json", digest("soak-v2 raw receipt")),
            bindings={
                "freeze_sha256": self.freeze_sha,
                "base_release_candidate_report_sha256": digest(base_raw),
                "configuration_profile": soak.STABLE_DEFAULT_PROFILE,
                "configuration_sha256": self.fixture.arms["stable_default"][
                    "configuration_sha256"
                ],
            },
            gate_e_soak=soak.GateESoakInputs(
                report=soak.Descriptor("raw/gate-e-soak-report.json", digest("soak gate report")),
                raw_archive=soak.Descriptor("raw/gate-e-soak.tar", digest("soak raw archive")),
                correctness_golden=soak.Descriptor("raw/correctness-golden.json", digest("soak golden")),
                native_correctness_report=soak.Descriptor(
                    "raw/native-correctness.json", digest("soak native correctness")
                ),
            ),
            scenario_trace=soak.Descriptor("raw/soak-v2-trace.json", digest("soak trace")),
            scenario_results=(),
        )

    def _soak_outer_inputs(self) -> tuple[qualification.FrozenCandidate, str]:
        frozen = qualification._validate_freeze(
            qualification._parse_document(
                self.fixture.freeze_path.read_bytes(), "outer soak-v2 fixture freeze"
            )
        )
        return frozen, digest((self.fixture.evidence / "reports/final.json").read_bytes())

    def _rollback_report(self) -> dict[str, object]:
        """Minimal outer-envelope fixture; rollback validates its full drill elsewhere."""

        return {
            "schema_version": rollback.CHECK_REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "fixture": "rollback",
        }

    def _parsed_rollback_report(self) -> rollback.RollbackCheckReport:
        base_raw = (self.fixture.evidence / "reports/final.json").read_bytes()
        return rollback.RollbackCheckReport(
            candidate_id=self.fixture.candidate_id,
            freeze_sha256=self.freeze_sha,
            base_release_candidate_report=rollback.Descriptor("reports/final.json", digest(base_raw)),
            bindings={
                "freeze_sha256": self.freeze_sha,
                "base_release_candidate_report_sha256": digest(base_raw),
                "configuration_profile": rollback.STABLE_DEFAULT_PROFILE,
                "configuration_sha256": self.fixture.arms["stable_default"][
                    "configuration_sha256"
                ],
            },
            receipt=rollback.Descriptor("raw/rollback-receipt.json", digest("rollback raw")),
            drill=rollback.Descriptor("raw/rollback-drill.json", digest("rollback drill")),
            candidate_artifacts=rollback.ArtifactSet(
                binary_sha256=self.fixture.release["binary_sha256"],
                bundle_sha256=self.fixture.release["bundle_sha256"],
                image_id=self.fixture.release["image_id"],
            ),
            rollback_artifacts=rollback.ArtifactSet(
                binary_sha256=self.fixture.freeze["rollback"]["binary_sha256"],
                bundle_sha256=self.fixture.freeze["rollback"]["bundle_sha256"],
                image_id=self.fixture.freeze["rollback"]["image_id"],
            ),
        )

    def _rollback_outer_inputs(self) -> tuple[qualification.FrozenCandidate, str]:
        frozen = qualification._validate_freeze(
            qualification._parse_document(
                self.fixture.freeze_path.read_bytes(), "outer rollback fixture freeze"
            )
        )
        return frozen, digest((self.fixture.evidence / "reports/final.json").read_bytes())

    def test_missing_repository_root_fails_closed(self) -> None:
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["missing-repository-root"])

    def test_expected_freeze_digest_is_a_trusted_input(self) -> None:
        report = qualification.evaluate(
            self.fixture.freeze_path,
            self.fixture.evidence,
            expected_candidate_sha256="f" * 64,
            repository_root=self.fixture.root,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["candidate-sha-mismatch"])

    def test_generic_passed_receipt_cannot_qualify(self) -> None:
        report = self._evaluate_with_replayed_gate_e()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["qualified"])
        self.assertEqual(report["reason_codes"], ["unimplemented-gate-validator"])

    def test_rehashed_arbitrary_receipt_is_not_semantic_proof(self) -> None:
        payload = self.fixture.evidence / "payloads/routing.txt"
        payload.write_text("different arbitrary bytes\n", encoding="utf-8")
        receipt_path = self.fixture.evidence / "receipts/routing.json"
        document = json.loads(receipt_path.read_text(encoding="utf-8"))
        document["evidence"]["sha256"] = digest(payload.read_bytes())
        receipt_path.write_bytes(qualification.canonical_json_bytes(document))
        report = self._evaluate_with_replayed_gate_e()
        self.assertEqual(report["reason_codes"], ["unimplemented-gate-validator"])

    def test_startup_configuration_receipt_is_replayed_before_next_gate(self) -> None:
        startup_report = self._startup_configuration_report()
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
            mock.patch.object(runtime_config, "evaluate", return_value=startup_report) as config_replay,
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["unimplemented-gate-validator"])
        config_replay.assert_called_once_with(
            self.fixture.freeze_path,
            self.fixture.evidence,
            "startup/stable-default-v1-config.json",
            "startup/stable-default-startup-config.json",
            "startup/max-performance-exact-v1-config.json",
            "startup/max-performance-exact-startup-config.json",
            expected_freeze_sha256=self.freeze_sha,
        )

    def test_startup_configuration_report_must_equal_its_semantic_replay(self) -> None:
        startup_report = self._startup_configuration_report()
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        replayed_startup = json.loads(json.dumps(startup_report))
        replayed_startup["arms"][runtime_config.STABLE_DEFAULT_PROFILE][
            "effective_config_sha256"
        ] = digest("replayed different stable config")
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
            mock.patch.object(runtime_config, "evaluate", return_value=replayed_startup),
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["startup-configuration-replay-mismatch"])

    def test_qwen_multistep_receipt_is_replayed_before_next_gate(self) -> None:
        startup_report = self._startup_configuration_report()
        qwen_report = self._qwen_multistep_report()
        parsed_qwen_report = self._parsed_qwen_multistep_report()
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        self.fixture.write_json("receipts/qwen_multistep.json", qwen_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
            mock.patch.object(runtime_config, "evaluate", return_value=startup_report),
            mock.patch.object(
                qwen_multistep,
                "validate_check_report",
                return_value=parsed_qwen_report,
            ) as qwen_parse,
            mock.patch.object(qwen_multistep, "evaluate", return_value=qwen_report) as qwen_replay,
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["unimplemented-gate-validator"])
        qwen_parse.assert_called_once_with(qwen_report)
        qwen_replay.assert_called_once_with(
            self.fixture.freeze_path,
            self.fixture.evidence,
            "raw/qwen-receipt.json",
            expected_freeze_sha256=self.freeze_sha,
        )

    def test_qwen_multistep_report_must_equal_its_semantic_replay(self) -> None:
        startup_report = self._startup_configuration_report()
        qwen_report = self._qwen_multistep_report()
        parsed_qwen_report = self._parsed_qwen_multistep_report()
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        self.fixture.write_json("receipts/qwen_multistep.json", qwen_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        replayed_qwen_report = dict(qwen_report)
        replayed_qwen_report["fixture"] = "replayed-different"
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
            mock.patch.object(runtime_config, "evaluate", return_value=startup_report),
            mock.patch.object(
                qwen_multistep,
                "validate_check_report",
                return_value=parsed_qwen_report,
            ),
            mock.patch.object(qwen_multistep, "evaluate", return_value=replayed_qwen_report),
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["qwen-multistep-replay-mismatch"])

    def test_qwen_outer_rejects_forged_stable_default_binding_before_replay(self) -> None:
        startup_report = self._startup_configuration_report()
        qwen_report = self._qwen_multistep_report()
        forged_qwen_report = replace(
            self._parsed_qwen_multistep_report(),
            bindings={
                **self._parsed_qwen_multistep_report().bindings,
                "configuration_sha256": digest("forged qwen stable-default configuration"),
            },
        )
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        self.fixture.write_json("receipts/qwen_multistep.json", qwen_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
            mock.patch.object(runtime_config, "evaluate", return_value=startup_report),
            mock.patch.object(
                qwen_multistep,
                "validate_check_report",
                return_value=forged_qwen_report,
            ),
            mock.patch.object(qwen_multistep, "evaluate") as qwen_replay,
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])
        qwen_replay.assert_not_called()

    def test_routing_receipt_is_replayed_before_next_gate(self) -> None:
        startup_report = self._startup_configuration_report()
        qwen_report = self._qwen_multistep_report()
        routing_report = self._routing_report()
        parsed_qwen_report = self._parsed_qwen_multistep_report()
        parsed_routing_report = self._parsed_routing_report()
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        self.fixture.write_json("receipts/qwen_multistep.json", qwen_report)
        self.fixture.write_json("receipts/routing.json", routing_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
            mock.patch.object(runtime_config, "evaluate", return_value=startup_report),
            mock.patch.object(qwen_multistep, "validate_check_report", return_value=parsed_qwen_report),
            mock.patch.object(qwen_multistep, "evaluate", return_value=qwen_report),
            mock.patch.object(
                routing, "validate_check_report", return_value=parsed_routing_report
            ) as routing_parse,
            mock.patch.object(routing, "evaluate", return_value=routing_report) as routing_replay,
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["unimplemented-gate-validator"])
        routing_parse.assert_called_once_with(routing_report)
        routing_replay.assert_called_once_with(
            self.fixture.freeze_path,
            self.fixture.evidence,
            "raw/routing-receipt.json",
            expected_freeze_sha256=self.freeze_sha,
        )

    def test_routing_outer_rejects_forged_candidate_freeze_base_model_or_stable_arm(self) -> None:
        startup_report = self._startup_configuration_report()
        qwen_report = self._qwen_multistep_report()
        routing_report = self._routing_report()
        parsed_qwen_report = self._parsed_qwen_multistep_report()
        parsed_routing_report = self._parsed_routing_report()
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        self.fixture.write_json("receipts/qwen_multistep.json", qwen_report)
        self.fixture.write_json("receipts/routing.json", routing_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        forged = {
            "candidate": replace(parsed_routing_report, candidate_id="riley-9.9.9-rc1"),
            "freeze": replace(parsed_routing_report, freeze_sha256="f" * 64),
            "base": replace(
                parsed_routing_report,
                base_release_candidate_report=routing.Descriptor("reports/other.json", digest("other base")),
            ),
            "model": replace(parsed_routing_report, model={"other": "model"}),
            "stable_arm": replace(
                parsed_routing_report,
                bindings={
                    **parsed_routing_report.bindings,
                    "configuration_sha256": digest("wrong stable configuration"),
                },
            ),
        }
        for label, forged_report in forged.items():
            with self.subTest(label=label):
                with (
                    mock.patch.object(qualification, "_validate_repository"),
                    mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
                    mock.patch.object(runtime_config, "evaluate", return_value=startup_report),
                    mock.patch.object(qwen_multistep, "validate_check_report", return_value=parsed_qwen_report),
                    mock.patch.object(qwen_multistep, "evaluate", return_value=qwen_report),
                    mock.patch.object(routing, "validate_check_report", return_value=forged_report),
                    mock.patch.object(routing, "evaluate") as routing_replay,
                ):
                    report = self.evaluate(repository_root=self.fixture.root)
                self.assertEqual(report["status"], "incomparable")
                self.assertEqual(report["reason_codes"], ["incomparable-binding"])
                routing_replay.assert_not_called()

    def test_routing_raw_receipt_may_not_reuse_a_frozen_output(self) -> None:
        startup_report = self._startup_configuration_report()
        qwen_report = self._qwen_multistep_report()
        routing_report = self._routing_report()
        parsed_qwen_report = self._parsed_qwen_multistep_report()
        parsed_routing_report = replace(
            self._parsed_routing_report(),
            receipt=routing.Descriptor("receipts/routing.json", digest("aliased routing output")),
        )
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        self.fixture.write_json("receipts/qwen_multistep.json", qwen_report)
        self.fixture.write_json("receipts/routing.json", routing_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
            mock.patch.object(runtime_config, "evaluate", return_value=startup_report),
            mock.patch.object(qwen_multistep, "validate_check_report", return_value=parsed_qwen_report),
            mock.patch.object(qwen_multistep, "evaluate", return_value=qwen_report),
            mock.patch.object(routing, "validate_check_report", return_value=parsed_routing_report),
            mock.patch.object(routing, "evaluate") as routing_replay,
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["reserved-output-path-collision"])
        routing_replay.assert_not_called()

    def test_routing_report_must_equal_its_semantic_replay(self) -> None:
        startup_report = self._startup_configuration_report()
        qwen_report = self._qwen_multistep_report()
        routing_report = self._routing_report()
        parsed_qwen_report = self._parsed_qwen_multistep_report()
        parsed_routing_report = self._parsed_routing_report()
        replayed_routing_report = dict(routing_report)
        replayed_routing_report["fixture"] = "replayed-different"
        self.fixture.write_json("receipts/startup_configuration.json", startup_report)
        self.fixture.write_json("receipts/qwen_multistep.json", qwen_report)
        self.fixture.write_json("receipts/routing.json", routing_report)
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed_base = self.fixture.base_report(digest(manifest_raw))
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed_base),
            mock.patch.object(runtime_config, "evaluate", return_value=startup_report),
            mock.patch.object(qwen_multistep, "validate_check_report", return_value=parsed_qwen_report),
            mock.patch.object(qwen_multistep, "evaluate", return_value=qwen_report),
            mock.patch.object(routing, "validate_check_report", return_value=parsed_routing_report),
            mock.patch.object(routing, "evaluate", return_value=replayed_routing_report),
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["routing-replay-mismatch"])

    def test_fault_extension_receipt_is_semantically_replayed_from_its_raw_descriptor(self) -> None:
        frozen, base_sha = self._fault_extension_outer_inputs()
        fault_report = self._fault_extension_report()
        parsed_fault_report = self._parsed_fault_extension_report()
        with (
            mock.patch.object(
                fault_extension, "validate_check_report", return_value=parsed_fault_report
            ) as fault_parse,
            mock.patch.object(
                fault_extension, "evaluate", return_value=fault_report
            ) as fault_replay,
        ):
            qualification._validate_receipt(
                fault_report,
                gate="fault_extension",
                freeze_path=self.fixture.freeze_path,
                frozen=frozen,
                freeze_sha256=self.freeze_sha,
                base_report_sha256=base_sha,
                evidence_root=self.fixture.evidence,
            )
        fault_parse.assert_called_once_with(fault_report)
        fault_replay.assert_called_once_with(
            self.fixture.freeze_path,
            self.fixture.evidence,
            "raw/fault-extension-receipt.json",
            expected_freeze_sha256=self.freeze_sha,
        )

    def test_fault_extension_outer_rejects_forged_candidate_freeze_base_or_stable_binding(
        self,
    ) -> None:
        frozen, base_sha = self._fault_extension_outer_inputs()
        fault_report = self._fault_extension_report()
        parsed_fault_report = self._parsed_fault_extension_report()
        forged = {
            "candidate": replace(parsed_fault_report, candidate_id="riley-9.9.9-rc1"),
            "freeze": replace(parsed_fault_report, freeze_sha256="f" * 64),
            "base": replace(
                parsed_fault_report,
                base_release_candidate_report=fault_extension.Descriptor(
                    "reports/other.json", digest("other base")
                ),
            ),
            "stable_arm": replace(
                parsed_fault_report,
                bindings={
                    **parsed_fault_report.bindings,
                    "configuration_sha256": digest("wrong stable configuration"),
                },
            ),
        }
        for label, forged_report in forged.items():
            with self.subTest(label=label):
                with (
                    mock.patch.object(
                        fault_extension, "validate_check_report", return_value=forged_report
                    ),
                    mock.patch.object(fault_extension, "evaluate") as fault_replay,
                ):
                    with self.assertRaises(qualification.IncomparableError):
                        qualification._validate_receipt(
                            fault_report,
                            gate="fault_extension",
                            freeze_path=self.fixture.freeze_path,
                            frozen=frozen,
                            freeze_sha256=self.freeze_sha,
                            base_report_sha256=base_sha,
                            evidence_root=self.fixture.evidence,
                        )
                fault_replay.assert_not_called()

    def test_fault_extension_outer_rejects_reserved_raw_path_and_replay_mismatch(self) -> None:
        frozen, base_sha = self._fault_extension_outer_inputs()
        fault_report = self._fault_extension_report()
        for reserved_path in (
            "manifests/final.json",
            "reports/final.json",
            "receipts/fault_extension.json",
        ):
            with self.subTest(reserved_path=reserved_path):
                reserved = replace(
                    self._parsed_fault_extension_report(),
                    receipt=fault_extension.Descriptor(
                        reserved_path, digest(f"aliased {reserved_path}")
                    ),
                )
                with (
                    mock.patch.object(
                        fault_extension, "validate_check_report", return_value=reserved
                    ),
                    mock.patch.object(fault_extension, "evaluate") as fault_replay,
                ):
                    with self.assertRaises(qualification.QualificationError) as raised:
                        qualification._validate_receipt(
                            fault_report,
                            gate="fault_extension",
                            freeze_path=self.fixture.freeze_path,
                            frozen=frozen,
                            freeze_sha256=self.freeze_sha,
                            base_report_sha256=base_sha,
                            evidence_root=self.fixture.evidence,
                        )
                self.assertEqual(
                    getattr(raised.exception, "reason_code"),
                    "reserved-output-path-collision",
                )
                fault_replay.assert_not_called()

        parsed = self._parsed_fault_extension_report()
        replayed = dict(fault_report)
        replayed["fixture"] = "replayed-different"
        with (
            mock.patch.object(fault_extension, "validate_check_report", return_value=parsed),
            mock.patch.object(fault_extension, "evaluate", return_value=replayed),
        ):
            with self.assertRaises(qualification.QualificationError) as raised:
                qualification._validate_receipt(
                    fault_report,
                    gate="fault_extension",
                    freeze_path=self.fixture.freeze_path,
                    frozen=frozen,
                    freeze_sha256=self.freeze_sha,
                    base_report_sha256=base_sha,
                    evidence_root=self.fixture.evidence,
                )
        self.assertEqual(
            getattr(raised.exception, "reason_code"),
            "fault-extension-replay-mismatch",
        )

    def test_soak_v2_receipt_is_semantically_replayed_from_its_raw_descriptor(self) -> None:
        frozen, base_sha = self._soak_outer_inputs()
        soak_report = self._soak_v2_report()
        parsed_soak_report = self._parsed_soak_v2_report()
        with (
            mock.patch.object(soak, "validate_check_report", return_value=parsed_soak_report) as soak_parse,
            mock.patch.object(soak, "evaluate", return_value=soak_report) as soak_replay,
        ):
            qualification._validate_receipt(
                soak_report,
                gate="soak_v2",
                freeze_path=self.fixture.freeze_path,
                frozen=frozen,
                freeze_sha256=self.freeze_sha,
                base_report_sha256=base_sha,
                evidence_root=self.fixture.evidence,
            )
        soak_parse.assert_called_once_with(soak_report)
        soak_replay.assert_called_once_with(
            self.fixture.freeze_path,
            self.fixture.evidence,
            "raw/soak-v2-receipt.json",
            expected_freeze_sha256=self.freeze_sha,
        )

    def test_soak_v2_outer_rejects_forged_candidate_freeze_base_or_stable_binding(self) -> None:
        frozen, base_sha = self._soak_outer_inputs()
        soak_report = self._soak_v2_report()
        parsed_soak_report = self._parsed_soak_v2_report()
        forged = {
            "candidate": replace(parsed_soak_report, candidate_id="riley-9.9.9-rc1"),
            "freeze": replace(parsed_soak_report, freeze_sha256="f" * 64),
            "base": replace(
                parsed_soak_report,
                base_release_candidate_report=soak.Descriptor(
                    "reports/other.json", digest("other base")
                ),
            ),
            "stable_arm": replace(
                parsed_soak_report,
                bindings={
                    **parsed_soak_report.bindings,
                    "configuration_sha256": digest("wrong stable configuration"),
                },
            ),
        }
        for label, forged_report in forged.items():
            with self.subTest(label=label):
                with (
                    mock.patch.object(soak, "validate_check_report", return_value=forged_report),
                    mock.patch.object(soak, "evaluate") as soak_replay,
                ):
                    with self.assertRaises(qualification.IncomparableError):
                        qualification._validate_receipt(
                            soak_report,
                            gate="soak_v2",
                            freeze_path=self.fixture.freeze_path,
                            frozen=frozen,
                            freeze_sha256=self.freeze_sha,
                            base_report_sha256=base_sha,
                            evidence_root=self.fixture.evidence,
                        )
                soak_replay.assert_not_called()

    def test_soak_v2_outer_rejects_reserved_raw_path_and_replay_mismatch(self) -> None:
        frozen, base_sha = self._soak_outer_inputs()
        soak_report = self._soak_v2_report()
        reserved = replace(
            self._parsed_soak_v2_report(),
            receipt=soak.Descriptor("receipts/soak_v2.json", digest("aliased soak output")),
        )
        with (
            mock.patch.object(soak, "validate_check_report", return_value=reserved),
            mock.patch.object(soak, "evaluate") as soak_replay,
        ):
            with self.assertRaises(qualification.QualificationError) as raised:
                qualification._validate_receipt(
                    soak_report,
                    gate="soak_v2",
                    freeze_path=self.fixture.freeze_path,
                    frozen=frozen,
                    freeze_sha256=self.freeze_sha,
                    base_report_sha256=base_sha,
                    evidence_root=self.fixture.evidence,
                )
        self.assertEqual(getattr(raised.exception, "reason_code"), "reserved-output-path-collision")
        soak_replay.assert_not_called()

        parsed = self._parsed_soak_v2_report()
        replayed = dict(soak_report)
        replayed["fixture"] = "replayed-different"
        with (
            mock.patch.object(soak, "validate_check_report", return_value=parsed),
            mock.patch.object(soak, "evaluate", return_value=replayed),
        ):
            with self.assertRaises(qualification.QualificationError) as raised:
                qualification._validate_receipt(
                    soak_report,
                    gate="soak_v2",
                    freeze_path=self.fixture.freeze_path,
                    frozen=frozen,
                    freeze_sha256=self.freeze_sha,
                    base_report_sha256=base_sha,
                    evidence_root=self.fixture.evidence,
                )
        self.assertEqual(getattr(raised.exception, "reason_code"), "soak-v2-replay-mismatch")

    def test_rollback_receipt_is_semantically_replayed_from_its_raw_descriptor(self) -> None:
        frozen, base_sha = self._rollback_outer_inputs()
        rollback_report = self._rollback_report()
        parsed_rollback_report = self._parsed_rollback_report()
        with (
            mock.patch.object(
                rollback, "validate_check_report", return_value=parsed_rollback_report
            ) as rollback_parse,
            mock.patch.object(rollback, "evaluate", return_value=rollback_report) as rollback_replay,
        ):
            qualification._validate_receipt(
                rollback_report,
                gate="rollback",
                freeze_path=self.fixture.freeze_path,
                frozen=frozen,
                freeze_sha256=self.freeze_sha,
                base_report_sha256=base_sha,
                evidence_root=self.fixture.evidence,
            )
        rollback_parse.assert_called_once_with(rollback_report)
        rollback_replay.assert_called_once_with(
            self.fixture.freeze_path,
            self.fixture.evidence,
            "raw/rollback-receipt.json",
            expected_freeze_sha256=self.freeze_sha,
        )

    def test_rollback_outer_rejects_forged_bindings_or_artifact_identities(self) -> None:
        frozen, base_sha = self._rollback_outer_inputs()
        rollback_report = self._rollback_report()
        parsed_rollback_report = self._parsed_rollback_report()
        forged = {
            "candidate": replace(parsed_rollback_report, candidate_id="riley-9.9.9-rc1"),
            "freeze": replace(parsed_rollback_report, freeze_sha256="f" * 64),
            "base": replace(
                parsed_rollback_report,
                base_release_candidate_report=rollback.Descriptor("reports/other.json", digest("other base")),
            ),
            "stable_arm": replace(
                parsed_rollback_report,
                bindings={
                    **parsed_rollback_report.bindings,
                    "configuration_sha256": digest("wrong stable configuration"),
                },
            ),
            "candidate_artifacts": replace(
                parsed_rollback_report,
                candidate_artifacts=rollback.ArtifactSet(
                    digest("other candidate binary"),
                    parsed_rollback_report.candidate_artifacts.bundle_sha256,
                    parsed_rollback_report.candidate_artifacts.image_id,
                ),
            ),
            "prior_artifacts": replace(
                parsed_rollback_report,
                rollback_artifacts=rollback.ArtifactSet(
                    parsed_rollback_report.rollback_artifacts.binary_sha256,
                    digest("other prior bundle"),
                    parsed_rollback_report.rollback_artifacts.image_id,
                ),
            ),
        }
        for label, forged_report in forged.items():
            with self.subTest(label=label):
                with (
                    mock.patch.object(rollback, "validate_check_report", return_value=forged_report),
                    mock.patch.object(rollback, "evaluate") as rollback_replay,
                ):
                    with self.assertRaises(qualification.IncomparableError):
                        qualification._validate_receipt(
                            rollback_report,
                            gate="rollback",
                            freeze_path=self.fixture.freeze_path,
                            frozen=frozen,
                            freeze_sha256=self.freeze_sha,
                            base_report_sha256=base_sha,
                            evidence_root=self.fixture.evidence,
                        )
                rollback_replay.assert_not_called()

    def test_rollback_outer_rejects_reserved_raw_path_and_replay_mismatch(self) -> None:
        frozen, base_sha = self._rollback_outer_inputs()
        rollback_report = self._rollback_report()
        reserved = replace(
            self._parsed_rollback_report(),
            receipt=rollback.Descriptor("receipts/rollback.json", digest("aliased rollback output")),
        )
        with (
            mock.patch.object(rollback, "validate_check_report", return_value=reserved),
            mock.patch.object(rollback, "evaluate") as rollback_replay,
        ):
            with self.assertRaises(qualification.QualificationError) as raised:
                qualification._validate_receipt(
                    rollback_report,
                    gate="rollback",
                    freeze_path=self.fixture.freeze_path,
                    frozen=frozen,
                    freeze_sha256=self.freeze_sha,
                    base_report_sha256=base_sha,
                    evidence_root=self.fixture.evidence,
                )
        self.assertEqual(getattr(raised.exception, "reason_code"), "reserved-output-path-collision")
        rollback_replay.assert_not_called()

        parsed = self._parsed_rollback_report()
        replayed = dict(rollback_report)
        replayed["fixture"] = "replayed-different"
        with (
            mock.patch.object(rollback, "validate_check_report", return_value=parsed),
            mock.patch.object(rollback, "evaluate", return_value=replayed),
        ):
            with self.assertRaises(qualification.QualificationError) as raised:
                qualification._validate_receipt(
                    rollback_report,
                    gate="rollback",
                    freeze_path=self.fixture.freeze_path,
                    frozen=frozen,
                    freeze_sha256=self.freeze_sha,
                    base_report_sha256=base_sha,
                    evidence_root=self.fixture.evidence,
                )
        self.assertEqual(getattr(raised.exception, "reason_code"), "rollback-replay-mismatch")

    def test_structurally_passed_base_report_cannot_skip_gate_e_replay(self) -> None:
        with mock.patch.object(qualification, "_validate_repository"):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["gate-failed"])

    def test_submitted_base_report_must_equal_gate_e_replay(self) -> None:
        base_path = self.fixture.evidence / "reports/final.json"
        document = json.loads(base_path.read_text(encoding="utf-8"))
        document["manifest_sha256"] = digest("different manifest")
        base_path.write_bytes(qualification.canonical_json_bytes(document))
        manifest_raw = (self.fixture.evidence / "manifests/final.json").read_bytes()
        replayed = self.fixture.base_report(digest(manifest_raw))
        with (
            mock.patch.object(qualification, "_validate_repository"),
            mock.patch.object(qualification.release_candidate, "evaluate", return_value=replayed),
        ):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["base-report-replay-mismatch"])

    def test_symlinked_gate_e_manifest_is_rejected_before_replay(self) -> None:
        target = self.fixture.evidence / "manifests/final.json"
        target.unlink()
        target.symlink_to(self.fixture.evidence / "reports/final.json")
        with mock.patch.object(qualification, "_validate_repository"):
            report = self.evaluate(repository_root=self.fixture.root)
        self.assertEqual(report["reason_codes"], ["missing-input"])

    def test_unknown_freeze_field_fails_closed(self) -> None:
        self.fixture.freeze["unexpected"] = True
        self.fixture.freeze_path.write_bytes(qualification.canonical_json_bytes(self.fixture.freeze))
        report = qualification.evaluate(
            self.fixture.freeze_path,
            self.fixture.evidence,
            expected_candidate_sha256=digest(self.fixture.freeze_path.read_bytes()),
            repository_root=self.fixture.root,
        )
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_equal_arms_are_rejected(self) -> None:
        self.fixture.freeze["arms"]["max_performance_exact"] = self.fixture.freeze["arms"]["stable_default"]
        self.fixture.freeze_path.write_bytes(qualification.canonical_json_bytes(self.fixture.freeze))
        report = qualification.evaluate(
            self.fixture.freeze_path,
            self.fixture.evidence,
            expected_candidate_sha256=digest(self.fixture.freeze_path.read_bytes()),
            repository_root=self.fixture.root,
        )
        self.assertEqual(report["reason_codes"], ["indistinguishable-arms"])

    def test_duplicate_json_key_is_rejected(self) -> None:
        self.fixture.freeze_path.write_text(
            '{"schema_version":"x","schema_version":"x"}', encoding="utf-8"
        )
        report = qualification.evaluate(
            self.fixture.freeze_path,
            self.fixture.evidence,
            expected_candidate_sha256=digest(self.fixture.freeze_path.read_bytes()),
            repository_root=self.fixture.root,
        )
        self.assertEqual(report["reason_codes"], ["duplicate-json-key"])

    def test_cli_requires_repository_root(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            qualification.main(
                [
                    "--freeze", str(self.fixture.freeze_path),
                    "--expected-candidate-sha256", self.freeze_sha,
                    "--evidence-root", str(self.fixture.evidence),
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_cli_report_is_create_only_even_for_a_failed_candidate(self) -> None:
        output = self.fixture.root / "decision.json"
        arguments = [
            "--freeze", str(self.fixture.freeze_path),
            "--expected-candidate-sha256", self.freeze_sha,
            "--evidence-root", str(self.fixture.evidence),
            "--repository-root", str(self.fixture.root),
            "--report", str(output),
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(qualification.main(arguments), 1)
        original = output.read_bytes()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(qualification.main(arguments), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_script_mode_preloaded_child_writes_failed_create_only_report(self) -> None:
        """A preloaded script-mode child parser error must reach the outer handler."""

        checker_path = Path(__file__).with_name("check_rc3_qualification.py")
        original_main = sys.modules["__main__"]
        original_qualification = sys.modules.get("check_rc3_qualification")
        original_qwen = sys.modules.get("check_qwen_multistep_receipt")
        original_argv = sys.argv
        try:
            self.assertIsNotNone(original_qualification)
            assert original_qualification is not None
            qwen_path = Path(__file__).with_name("check_qwen_multistep_receipt.py")
            qwen_spec = importlib.util.spec_from_file_location(
                "check_qwen_multistep_receipt", qwen_path
            )
            self.assertIsNotNone(qwen_spec)
            assert qwen_spec is not None
            self.assertIsNotNone(qwen_spec.loader)
            assert qwen_spec.loader is not None
            preloaded_qwen = importlib.util.module_from_spec(qwen_spec)
            sys.modules["check_qwen_multistep_receipt"] = preloaded_qwen
            qwen_spec.loader.exec_module(preloaded_qwen)
            self.assertIs(preloaded_qwen.qualification, original_qualification)

            outer_spec = importlib.util.spec_from_file_location("__main__", checker_path)
            self.assertIsNotNone(outer_spec)
            assert outer_spec is not None
            self.assertIsNotNone(outer_spec.loader)
            assert outer_spec.loader is not None
            script_outer = importlib.util.module_from_spec(outer_spec)
            sys.modules["__main__"] = script_outer
            sys.argv = [str(checker_path), "--help"]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    outer_spec.loader.exec_module(script_outer)
            self.assertEqual(raised.exception.code, 0)
            self.assertIs(sys.modules["check_rc3_qualification"], script_outer)
            self.assertIsNot(preloaded_qwen.qualification, script_outer)
            script_outer.REQUIRED_GATES = ("qwen_multistep",)

            freeze_raw = script_outer.canonical_json_bytes({"fixture": "script-mode"})
            malformed = script_outer.canonical_json_bytes(
                {"schema_version": preloaded_qwen.CHECK_REPORT_VERSION}
            )
            base_raw = script_outer.canonical_json_bytes(
                {"manifest_sha256": "a" * 64, "bindings": {"evidence_sha256": {}}}
            )
            frozen = SimpleNamespace(
                candidate_id=self.fixture.candidate_id,
                receipts={"qwen_multistep": script_outer.Descriptor("receipts/qwen_multistep.json")},
            )

            def read_regular(path: Path, label: str) -> bytes:
                if label == "freeze manifest":
                    return freeze_raw
                self.fail(f"unexpected regular read: {label}")

            output = self.fixture.root / "script-mode-failed-report.json"
            arguments = [
                "--freeze",
                str(self.fixture.freeze_path),
                "--expected-candidate-sha256",
                digest(freeze_raw),
                "--evidence-root",
                str(self.fixture.evidence),
                "--repository-root",
                str(self.fixture.root),
                "--report",
                str(output),
            ]
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(script_outer, "_read_regular_path", side_effect=read_regular),
                mock.patch.object(script_outer, "_read_relative", return_value=malformed),
                mock.patch.object(script_outer, "_validate_freeze", return_value=frozen),
                mock.patch.object(script_outer, "_validate_repository"),
                mock.patch.object(
                    script_outer,
                    "revalidate_base_release_candidate",
                    return_value=(base_raw, digest(base_raw)),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                self.assertEqual(script_outer.main(arguments), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["qualified"])
            self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])
            self.assertNotIn("Traceback", stdout.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIs(preloaded_qwen.qualification, script_outer)
        finally:
            sys.argv = original_argv
            sys.modules["__main__"] = original_main
            if original_qualification is None:
                sys.modules.pop("check_rc3_qualification", None)
            else:
                sys.modules["check_rc3_qualification"] = original_qualification
            if original_qwen is None:
                sys.modules.pop("check_qwen_multistep_receipt", None)
            else:
                sys.modules["check_qwen_multistep_receipt"] = original_qwen

    def test_freeze_writer_canonicalizes_and_is_create_only(self) -> None:
        source = self.fixture.root / "reviewed-freeze.json"
        source.write_text(json.dumps(self.fixture.freeze, indent=4), encoding="utf-8")
        output = self.fixture.root / "external" / "rc3.freeze.json"
        output.parent.mkdir()
        result = freeze_writer.write_freeze(source, output)
        document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(output.read_bytes(), qualification.canonical_json_bytes(document))
        self.assertEqual(result.sha256, digest(output.read_bytes()))
        with self.assertRaises(qualification.QualificationError):
            freeze_writer.write_freeze(source, output)

    def test_static_schema_retires_the_generic_receipt_envelope(self) -> None:
        schema_path = Path(__file__).parents[2] / "benchmarks/release/candidates/rc3-qualification.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        top_level_refs = {entry["$ref"] for entry in schema["oneOf"]}
        self.assertNotIn("#/$defs/gateReceipt", top_level_refs)
        self.assertIn("correctness_golden_sha256", schema["$defs"]["source"]["required"])
        self.assertIn(
            "final_release_candidate_manifest",
            schema["$defs"]["outputPaths"]["required"],
        )
        self.assertEqual(
            set(schema["$defs"]["baseGateEEvidenceHashes"]["required"]),
            set(qualification.BASE_EVIDENCE_SHA256_KEYS),
        )

    def test_finalizer_shell_has_valid_bash_syntax(self) -> None:
        script = Path(__file__).with_name("run_rc3_qualification.sh")
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
