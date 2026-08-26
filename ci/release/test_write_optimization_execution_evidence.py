#!/usr/bin/env python3
"""CPU-only tests for the remote optimizer evidence writer."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import write_optimization_execution_evidence as writer  # noqa: E402


REVISION = "1a2b3c4d5e6f78901234567890abcdef12345678"
SOURCE_SHA = hashlib.sha256(b"source").hexdigest()
IMAGE_ID = "sha256:" + hashlib.sha256(b"image").hexdigest()
MODEL_TREE_SHA = hashlib.sha256(b"model tree").hexdigest()


def encoded(value: object) -> str:
    return base64.b64encode(str(value).encode()).decode()


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.commands = root / "commands.v2"
        self.subjects = root / "subjects.v2"
        self.gpu = root / "gpu.csv"
        self.profile = root / "rustinfer-profile"
        self.report = root / "report.json"
        self.receipt = self.evidence / "run-receipt.json"
        self.profile.write_bytes(b"profile executable fixture")
        self.profile.chmod(0o755)
        self.gpu.write_text(
            f"{writer.GPU_NAME}, {writer.GPU_UUID}, 00000000:01:00.0, "
            "24564, 580.173.02, 8.9\n"
        )
        self._write_logs_and_subjects()
        self._write_commands()

    def _write_logs_and_subjects(self) -> None:
        rows = [writer.SUBJECT_RECORD_VERSION]
        for filename, spec in writer.TEST_SUBJECTS.items():
            binary = self.evidence / filename
            binary.write_bytes(f"#!/bin/sh\n# {filename}\n".encode())
            binary.chmod(0o755)
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            cargo_path = f"{spec['target_dir']}/debug/deps/{spec['cargo_test_target']}-fixture"
            artifact = {
                "reason": "compiler-artifact",
                "target": {
                    "name": spec["cargo_test_target"],
                    "kind": ["test"],
                    "crate_types": ["bin"],
                },
                "profile": {"test": True},
                "features": ["cuda"],
                "fresh": False,
                "executable": cargo_path,
            }
            (self.evidence / spec["compile_log"]).write_text(
                json.dumps(artifact, separators=(",", ":")) + "\n"
            )
            rows.append(
                "\t".join(
                    [
                        filename,
                        spec["cargo_test_target"],
                        cargo_path,
                        digest,
                        f"/evidence/{filename}",
                        digest,
                        str(binary.stat().st_size),
                        spec["compile_command_id"],
                        spec["execute_command_id"],
                    ]
                )
            )
        self.subjects.write_text("\n".join(rows) + "\n")

        (self.evidence / writer.LOG_FILES["cuda-compile-only"]).write_text("compile passed\n")
        (self.evidence / writer.LOG_FILES["workspace-all-features-all-targets"]).write_text(
            "workspace passed\n"
        )
        (self.evidence / writer.LOG_FILES["command-batch-lifecycle"]).write_text(
            "running 1 test\n"
            "test command_batch_proxy_is_one_shot_and_drop_restores_stream_use ... "
            "pr16-command-batch-lifecycle schema_version=1 one_shot_finish=true "
            "drop_restores_stream=true status=passed\n"
            "ok\n\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
            "7 filtered out; finished in 0.01s\n"
        )
        (self.evidence / writer.LOG_FILES["command-batch-resource-ledger"]).write_text(
            "running 1 test\n"
            "test command_batch_releases_multi_primitive_resource_ledger_after_validation_error ... "
            "pr16-command-batch-resource-ledger schema_version=1 validation_fail_closed=true "
            "queued_chain_raw_byte_mismatches=0 cuda_live_allocation_delta=0 "
            "stream_reuse_after_finish=true owner_close_live_allocation_count=0 status=passed\n"
            "ok\n\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
            "5 filtered out; finished in 0.01s\n"
        )
        ids = ", ".join(str(value) for value in writer.EXPECTED_TOKENS)
        (self.evidence / writer.LOG_FILES["smollm2-multi-step-greedy-exact"]).write_text(
            "running 1 test\n"
            "test iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly ... "
            "pr15-execution-completion-parity schema_version=1 decode_steps=16 "
            "committed_iterations=16 raw_logit_mismatches=0 token_id_mismatches=0 "
            "cuda_live_allocation_delta=0 owner_close_live_allocation_count=0 "
            f"generated_token_ids=[{ids}] status=passed\n"
            "ok\n\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
            "9 filtered out; finished in 0.01s\n"
        )
        (self.evidence / writer.LOG_FILES["fixed37-production-batch-e0"]).write_text(
            "running 1 test\n"
            "test fixed37_production_batch_growing_prefix_matches_golden_exactly ... "
            f"{writer.FIXED37_PRODUCTION_BATCH_GATE_ID} schema_version=1 "
            f"fixture_sha256={writer.EXPECTED_FIXED37_FIXTURE_SHA256} "
            f"generated_token_ids_sha256={writer.EXPECTED_FIXED37_TOKEN_IDS_SHA256} "
            "cases=31 compared_steps=481 exact_window=16 "
            "fixed_profile=fixed-contiguous-37-balanced-v1 "
            "canonical_profile=canonical-v1 residual_rmsnorm=separate "
            "execution_completion=iteration-batch "
            "fixed_prefill_raw_logit_mismatches=0 "
            "fixed_cached_growing_token_id_mismatches=0 "
            "fixed_cached_growing_cosine_min=0.9979035305495393 "
            "fixed_cached_growing_max_abs_max=5.852936458587647 "
            "fixed_cached_growing_mean_abs_max=1.151280319263363 "
            "fixed_cached_growing_worst_cosine=0.999 "
            "fixed_cached_growing_worst_max_abs=1.0 "
            "fixed_cached_growing_worst_mean_abs=0.25 "
            "fixed_cached_growing_threshold_violations=0 "
            "fixed_golden_token_id_mismatches=0 "
            "canonical_golden_token_id_mismatches=0 "
            "cuda_live_allocation_delta=0 owner_close_live_allocation_count=0 "
            "status=passed\n"
            "ok\n\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
            "9 filtered out; finished in 0.01s\n"
        )

    def _write_commands(
        self,
        argv_override: dict[str, list[str]] | None = None,
        environment_override: dict[str, dict[str, str]] | None = None,
    ) -> None:
        rows = [writer.COMMAND_RECORD_VERSION]
        for command_id in writer.COMMAND_ORDER:
            expected = writer.EXPECTED_COMMANDS[command_id]
            environment = dict(writer.BASE_ENVIRONMENT)
            if command_id in {
                "smollm2-multi-step-greedy-exact",
                "fixed37-production-batch-e0",
            }:
                environment["RUSTINFER_REAL_CHECKPOINT"] = "/model"
            environment.update((environment_override or {}).get(command_id, {}))
            rows.extend(
                [
                    f"BEGIN {encoded(command_id)}",
                    f"LOG {encoded(expected['log'])}",
                    f"SUBJECT {encoded(expected['test_binary'] or '-')}",
                ]
            )
            rows.extend(f"ENV {encoded(key)} {encoded(value)}" for key, value in environment.items())
            argv = (argv_override or {}).get(command_id, expected["argv"])
            rows.extend(f"ARG {encoded(argument)}" for argument in argv)
            rows.extend([f"EXIT {encoded(0)}", "END"])
        self.commands.write_text("\n".join(rows) + "\n")

    def write(self) -> tuple[dict[str, object], dict[str, object]]:
        return writer.write_evidence(
            self.evidence,
            command_records=self.commands,
            subject_records=self.subjects,
            gpu_csv=self.gpu,
            report_path=self.report,
            receipt_path=self.receipt,
            source_revision=REVISION,
            source_archive_sha256=SOURCE_SHA,
            build_image_id=IMAGE_ID,
            profile_binary=self.profile,
            model_tree_sha256=MODEL_TREE_SHA,
            recorded_at_utc="2026-08-26T00:00:00Z",
        )


class EvidenceWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_writes_direct_elf_execution_and_build_provenance(self) -> None:
        report, receipt = self.fixture.write()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(receipt["schema_version"], writer.RECEIPT_VERSION)
        self.assertEqual([row["id"] for row in receipt["commands"]], writer.COMMAND_ORDER)
        direct = receipt["commands"][3]
        self.assertEqual(direct["argv"][0], "/evidence/host-runtime-gpu-test")
        subject = receipt["subjects"]["host-runtime-gpu-test"]
        self.assertEqual(subject["compile_command_id"], "compile-command-batch-lifecycle")
        self.assertEqual(subject["execute_command_id"], "command-batch-lifecycle")
        fixed37 = next(
            row for row in report["tests"] if row["id"] == "fixed37-production-batch-e0"
        )
        self.assertEqual(fixed37["gate_id"], writer.FIXED37_PRODUCTION_BATCH_GATE_ID)
        self.assertEqual(fixed37["cases"], 31)
        self.assertEqual(fixed37["compared_steps"], 481)
        self.assertEqual(fixed37["compile_command_id"], "compile-fixed37-production-batch-e0")
        self.assertEqual(
            fixed37["test_binary_sha256"],
            subject_digest := hashlib.sha256(
                (self.fixture.evidence / "fixed37-production-batch-gpu-test").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            receipt["subjects"]["fixed37-production-batch-gpu-test"]["sha256"],
            subject_digest,
        )
        self.assertEqual(self.fixture.report.read_bytes(), writer.canonical_json_bytes(report))
        self.assertEqual(self.fixture.receipt.read_bytes(), writer.canonical_json_bytes(receipt))

    def test_tampered_copied_subject_is_rejected(self) -> None:
        path = self.fixture.evidence / "host-runtime-gpu-test"
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(writer.EvidenceWriterError, "differs from copied bytes"):
            self.fixture.write()

    def test_cargo_compiler_artifact_path_is_required(self) -> None:
        log = self.fixture.evidence / writer.COMPILE_LOG_FILES["compile-command-batch-lifecycle"]
        log.write_text(log.read_text().replace("host_runtime_gpu-fixture", "other-fixture"))
        with self.assertRaisesRegex(writer.EvidenceWriterError, "matching compiler-artifact"):
            self.fixture.write()

    def test_direct_execution_argv_cannot_be_replaced_by_cargo(self) -> None:
        self.fixture._write_commands({"command-batch-lifecycle": ["cargo", "test"]})
        with self.assertRaisesRegex(writer.EvidenceWriterError, "reviewed invocation"):
            self.fixture.write()

    def test_fixed37_direct_execution_argv_is_exact(self) -> None:
        self.fixture._write_commands(
            {"fixed37-production-batch-e0": ["/evidence/fixed37-production-batch-gpu-test", "unreviewed_test"]}
        )
        with self.assertRaisesRegex(writer.EvidenceWriterError, "reviewed invocation"):
            self.fixture.write()

    def test_toolchain_environment_cannot_follow_directory_override(self) -> None:
        self.fixture._write_commands(
            environment_override={
                "workspace-all-features-all-targets": {
                    "RUSTUP_TOOLCHAIN": "stable-x86_64-unknown-linux-gnu"
                }
            }
        )
        with self.assertRaisesRegex(writer.EvidenceWriterError, "reviewed invocation"):
            self.fixture.write()

    def test_parity_marker_is_derived_not_trusted(self) -> None:
        log = self.fixture.evidence / writer.LOG_FILES["smollm2-multi-step-greedy-exact"]
        log.write_text(log.read_text().replace("raw_logit_mismatches=0", "raw_logit_mismatches=1"))
        with self.assertRaisesRegex(writer.EvidenceWriterError, "reviewed E0 result"):
            self.fixture.write()

    def test_fixed37_marker_fields_are_derived_not_trusted(self) -> None:
        path = self.fixture.evidence / writer.LOG_FILES["fixed37-production-batch-e0"]
        original = path.read_text()
        mutations = (
            ("fixture_sha256=8733", "fixture_sha256=9733"),
            ("generated_token_ids_sha256=9e38", "generated_token_ids_sha256=8e38"),
            ("cases=31", "cases=30"),
            ("compared_steps=481", "compared_steps=480"),
            ("exact_window=16", "exact_window=15"),
            ("fixed_profile=fixed-contiguous-37-balanced-v1", "fixed_profile=canonical-v1"),
            ("residual_rmsnorm=separate", "residual_rmsnorm=fused"),
            ("execution_completion=iteration-batch", "execution_completion=per-operation"),
            (
                "fixed_prefill_raw_logit_mismatches=0",
                "fixed_prefill_raw_logit_mismatches=1",
            ),
            (
                "fixed_cached_growing_worst_cosine=0.999",
                "fixed_cached_growing_worst_cosine=0.900",
            ),
            (
                "fixed_cached_growing_threshold_violations=0",
                "fixed_cached_growing_threshold_violations=1",
            ),
            ("fixed_golden_token_id_mismatches=0", "fixed_golden_token_id_mismatches=1"),
            ("cuda_live_allocation_delta=0", "cuda_live_allocation_delta=1"),
        )
        for old, new in mutations:
            with self.subTest(field=old.split("=")[0]):
                changed = original.replace(old, new, 1)
                self.assertNotEqual(changed, original)
                path.write_text(changed)
                with self.assertRaisesRegex(
                    writer.EvidenceWriterError,
                    "exact fixed37 production-batch marker|reviewed production-batch E0 result|immutable E0 bounds",
                ):
                    self.fixture.write()
                path.write_text(original)

    def test_llama_batch_exact_logs_require_nine_filtered_tests(self) -> None:
        for test_id in (
            "smollm2-multi-step-greedy-exact",
            "fixed37-production-batch-e0",
        ):
            with self.subTest(test_id=test_id):
                path = self.fixture.evidence / writer.LOG_FILES[test_id]
                original = path.read_text()
                changed = original.replace("9 filtered out", "6 filtered out", 1)
                self.assertNotEqual(changed, original)
                path.write_text(changed)
                with self.assertRaisesRegex(
                    writer.EvidenceWriterError, "exact passing libtest summary"
                ):
                    self.fixture.write()
                path.write_text(original)

    def test_fixed37_log_and_compile_artifact_are_required(self) -> None:
        log = self.fixture.evidence / writer.LOG_FILES["fixed37-production-batch-e0"]
        log.unlink()
        with self.assertRaisesRegex(writer.EvidenceWriterError, "closed pre-receipt inventory"):
            self.fixture.write()

    def test_fixed37_subject_cannot_claim_parity_command_provenance(self) -> None:
        text = self.fixture.subjects.read_text()
        text = text.replace(
            "compile-fixed37-production-batch-e0\tfixed37-production-batch-e0\n",
            "compile-smollm2-multi-step-greedy-exact\tsmollm2-multi-step-greedy-exact\n",
        )
        self.fixture.subjects.write_text(text)
        with self.assertRaisesRegex(writer.EvidenceWriterError, "provenance mismatch"):
            self.fixture.write()

    def test_pre_receipt_inventory_is_closed(self) -> None:
        (self.fixture.evidence / "unreviewed.log").write_text("extra\n")
        with self.assertRaisesRegex(writer.EvidenceWriterError, "closed pre-receipt inventory"):
            self.fixture.write()


if __name__ == "__main__":
    unittest.main()
