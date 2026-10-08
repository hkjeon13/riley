"""Hostile-input tests for the C03-A V1 diagnostic receipt checker."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CHECKER = REPOSITORY_ROOT / "benchmarks/scripts/check_routing_fuzz_receipt.py"
SOURCE_REVISION = "0123456789abcdef0123456789abcdef01234567"


def canonical_document(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"


def descriptor(
    case_id: str,
    decoder_count: int,
    final_prefill_count: int,
    prime_slot_order: list[int],
    mixed_slot_order: list[int],
    cancel_decoder_index: int | None,
    settlement: str,
) -> dict[str, object]:
    return {
        "format": "riley.scheduler.general-mixed-operation",
        "format_version": 1,
        "trace_kind": "general-mixed-operation-v1",
        "case_id": case_id,
        "source_seed": "0x75c481a23b9de6f0",
        "decoder_count": decoder_count,
        "final_prefill_count": final_prefill_count,
        "prime_slot_order": prime_slot_order,
        "mixed_slot_order": mixed_slot_order,
        "cancel_decoder_index": cancel_decoder_index,
        "settlement": settlement,
    }


def operations(trace: dict[str, object]) -> str:
    cancellation = trace["cancel_decoder_index"]
    cancellation_text = "none" if cancellation is None else f"cancel decoder[{cancellation}]"
    settlement_text = (
        "complete" if trace["settlement"] == "commit" else "abort(not-dispatched)"
    )
    return (
        f"submit decoder[0..{trace['decoder_count']}) -> plan-prime -> "
        f"complete-prime(order={trace['prime_slot_order']}) -> "
        f"submit final-prefill[0..{trace['final_prefill_count']}) -> plan-mixed -> "
        f"{cancellation_text} -> {settlement_text}(order={trace['mixed_slot_order']}) "
        "-> close"
    )


def scheduler_config(trace: dict[str, object]) -> dict[str, object]:
    width = int(trace["decoder_count"]) + int(trace["final_prefill_count"])
    return {
        "max_waiting_requests": width,
        "max_waiting_prompt_tokens": width,
        "max_active_sequences": width,
        "max_sequence_tokens": 3,
        "iteration_token_budget": width,
        "max_prefill_chunk_tokens": 1,
        "aging_threshold_ns": 2,
        "overload_policy": "wait",
        "admission_timeout_ns": None,
        "max_promised_kv_blocks": width,
        "metrics_window_samples": 8,
    }


def valid_receipt() -> dict[str, object]:
    source = descriptor(
        "receipt-source",
        3,
        2,
        [2, 0, 1],
        [4, 1, 3, 0, 2],
        2,
        "abort_not_dispatched",
    )
    minimized = descriptor(
        "failing-minimized",
        2,
        1,
        [0, 1],
        [0, 1, 2],
        None,
        "abort_not_dispatched",
    )
    return {
        "format": "riley.scheduler.routing-fuzz-receipt",
        "format_version": 1,
        "scope": "diagnostic-only",
        "trace_kind": "general-mixed-operation-v1",
        "test_target": "riley-scheduler::general_mixed_operation_routing",
        "source_revision": SOURCE_REVISION,
        "source_case_id": "receipt-source",
        "failure_predicate": "inner-replayer-panicked-only",
        "reducer_scope": "v1-selector-local",
        "source_descriptor_json": canonical_document(source),
        "minimized_descriptor_json": canonical_document(minimized),
        "source_operations": operations(source),
        "minimized_operations": operations(minimized),
        "source_scheduler_config": scheduler_config(source),
        "minimized_scheduler_config": scheduler_config(minimized),
        "symbolic_kv_layout": {
            "layer_count": 1,
            "physical_block_count": 64,
            "key_value_head_count": 1,
            "head_dimension": 8,
            "block_size_tokens": 16,
        },
        "replay_timeline_ns": {
            "decoder_submit_and_prime_ns": 0,
            "final_prefill_submit_and_mixed_ns": 1,
            "close_ns": 2,
        },
        "not_established": [
            "c02_qualification",
            "c03_b_gpu_evidence",
            "general_or_global_minimum",
            "panic_site_payload_signature_root_cause",
            "scheduler_reexecution",
        ],
    }


class RoutingFuzzReceiptCheckerTests(unittest.TestCase):
    def run_checker(
        self, receipt_path: Path, expected_source_revision: str = SOURCE_REVISION
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--receipt",
                str(receipt_path),
                "--expected-source-revision",
                expected_source_revision,
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

    def write_document(self, directory: Path, document: str) -> Path:
        receipt = json.loads(document)
        source = json.loads(str(receipt["source_descriptor_json"]))
        path = directory / (
            "general-mixed-operation-v1-"
            f"{receipt['source_case_id']}-{str(source['source_seed'])[2:]}.json"
        )
        path.write_text(document, encoding="utf-8")
        return path

    def test_accepts_exact_canonical_diagnostic_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_document(Path(temporary), canonical_document(valid_receipt()))
            result = self.run_checker(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("structurally valid", result.stdout)
        self.assertIn("remain not established", result.stdout)

    def test_rejects_filename_that_does_not_bind_source_case_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "renamed-receipt.json"
            path.write_text(canonical_document(valid_receipt()), encoding="utf-8")
            result = self.run_checker(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("filename", result.stderr)

    def test_rejects_noncanonical_and_inconsistent_receipts(self) -> None:
        valid = valid_receipt()
        malformed_cases: list[tuple[str, str, str]] = []

        whitespace = canonical_document(valid)
        malformed_cases.append(("outer-whitespace", f" {whitespace}", SOURCE_REVISION))

        duplicate = canonical_document(valid).replace(
            '{"format":"riley.scheduler.routing-fuzz-receipt",',
            '{"format":"riley.scheduler.routing-fuzz-receipt",'
            '"format":"riley.scheduler.routing-fuzz-receipt",',
            1,
        )
        malformed_cases.append(("duplicate-outer-key", duplicate, SOURCE_REVISION))

        unknown = copy.deepcopy(valid)
        unknown["unexpected"] = True
        malformed_cases.append(("unknown-field", canonical_document(unknown), SOURCE_REVISION))

        wrong_revision = copy.deepcopy(valid)
        wrong_revision["source_revision"] = "fedcba9876543210fedcba9876543210fedcba98"
        malformed_cases.append(("revision-mismatch", canonical_document(wrong_revision), SOURCE_REVISION))

        zero_revision = copy.deepcopy(valid)
        zero_revision["source_revision"] = "0" * 40
        malformed_cases.append(
            ("zero-revision", canonical_document(zero_revision), "0" * 40)
        )

        wrong_scope = copy.deepcopy(valid)
        wrong_scope["scope"] = "qualification"
        malformed_cases.append(("wrong-scope", canonical_document(wrong_scope), SOURCE_REVISION))

        exponent_float = canonical_document(valid).replace(
            '"format_version":1,', '"format_version":1e9999,', 1
        )
        malformed_cases.append(("exponent-float", exponent_float, SOURCE_REVISION))

        seed_drift = copy.deepcopy(valid)
        minimized_seed = json.loads(str(seed_drift["minimized_descriptor_json"]))
        minimized_seed["source_seed"] = "0x0000000000000000"
        seed_drift["minimized_descriptor_json"] = canonical_document(minimized_seed)
        malformed_cases.append(("seed-drift", canonical_document(seed_drift), SOURCE_REVISION))

        settlement_drift = copy.deepcopy(valid)
        minimized_settlement = json.loads(str(settlement_drift["minimized_descriptor_json"]))
        minimized_settlement["settlement"] = "commit"
        settlement_drift["minimized_descriptor_json"] = canonical_document(minimized_settlement)
        malformed_cases.append(
            ("settlement-drift", canonical_document(settlement_drift), SOURCE_REVISION)
        )

        settlement_type = copy.deepcopy(valid)
        minimized_settlement_type = json.loads(
            str(settlement_type["minimized_descriptor_json"])
        )
        minimized_settlement_type["settlement"] = []
        settlement_type["minimized_descriptor_json"] = canonical_document(minimized_settlement_type)
        malformed_cases.append(
            ("settlement-type", canonical_document(settlement_type), SOURCE_REVISION)
        )

        malformed_permutation = copy.deepcopy(valid)
        source_slots = json.loads(str(malformed_permutation["source_descriptor_json"]))
        source_slots["prime_slot_order"] = [0, 0, 1]
        malformed_permutation["source_descriptor_json"] = canonical_document(source_slots)
        malformed_cases.append(
            ("malformed-permutation", canonical_document(malformed_permutation), SOURCE_REVISION)
        )

        config_drift = copy.deepcopy(valid)
        config_drift["source_scheduler_config"]["max_active_sequences"] = 99
        malformed_cases.append(("config-drift", canonical_document(config_drift), SOURCE_REVISION))

        config_boolean = copy.deepcopy(valid)
        config_boolean["source_scheduler_config"]["max_prefill_chunk_tokens"] = True
        malformed_cases.append(
            ("config-boolean", canonical_document(config_boolean), SOURCE_REVISION)
        )

        layout_drift = copy.deepcopy(valid)
        layout_drift["symbolic_kv_layout"]["physical_block_count"] = 65
        malformed_cases.append(("layout-drift", canonical_document(layout_drift), SOURCE_REVISION))

        layout_boolean = copy.deepcopy(valid)
        layout_boolean["symbolic_kv_layout"]["layer_count"] = True
        malformed_cases.append(
            ("layout-boolean", canonical_document(layout_boolean), SOURCE_REVISION)
        )

        timeline_drift = copy.deepcopy(valid)
        timeline_drift["replay_timeline_ns"]["close_ns"] = 3
        malformed_cases.append(("timeline-drift", canonical_document(timeline_drift), SOURCE_REVISION))

        timeline_boolean = copy.deepcopy(valid)
        timeline_boolean["replay_timeline_ns"]["decoder_submit_and_prime_ns"] = False
        malformed_cases.append(
            ("timeline-boolean", canonical_document(timeline_boolean), SOURCE_REVISION)
        )

        boundary_drift = copy.deepcopy(valid)
        boundary_drift["not_established"] = []
        malformed_cases.append(("boundary-drift", canonical_document(boundary_drift), SOURCE_REVISION))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for label, document, expected_revision in malformed_cases:
                with self.subTest(label=label):
                    path = self.write_document(directory, document)
                    result = self.run_checker(path, expected_revision)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("routing fuzz receipt rejected:", result.stderr)

    def test_rejects_symlinks_and_oversized_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = self.write_document(directory, canonical_document(valid_receipt()))
            symlink = directory / "receipt-link.json"
            symlink.symlink_to(target)
            result = self.run_checker(symlink)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("regular file", result.stderr)

            oversized = directory / "oversized.json"
            oversized.write_bytes(b"{" + b" " * (64 * 1024))
            result = self.run_checker(oversized)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exceeds", result.stderr)


if __name__ == "__main__":
    unittest.main()
