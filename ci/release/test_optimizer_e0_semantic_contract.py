#!/usr/bin/env python3
"""Unit tests for the stdlib-only final optimizer E0 report contract."""

from __future__ import annotations

import copy
import unittest

import optimizer_e0_semantic_contract as contract


REVISION = "1" * 40
ARCHIVE_SHA256 = "a" * 64
IMAGE_SHA256 = "b" * 64
OTHER_SHA256 = "c" * 64


def valid_report() -> dict[str, object]:
    """Return a closed report without relying on raw-evidence I/O fixtures."""

    fixed37 = {
        "id": "fixed37-production-batch-e0",
        "result": "passed",
        "gate_id": contract.FIXED37_PRODUCTION_BATCH_GATE_ID,
        "fixture_sha256": contract.EXPECTED_FIXED37_FIXTURE_SHA256,
        "generated_token_ids_sha256": contract.EXPECTED_FIXED37_TOKEN_IDS_SHA256,
        "cases": 31,
        "compared_steps": 481,
        "exact_window": 16,
        "fixed_profile": "fixed-contiguous-37-balanced-v1",
        "canonical_profile": "canonical-v1",
        "residual_rmsnorm": "separate",
        "execution_completion": "iteration-batch",
        "fixed_prefill_raw_logit_mismatches": 0,
        "fixed_cached_growing_token_id_mismatches": 0,
        "fixed_cached_growing_cosine_min": contract.FIXED37_CACHED_GROWING_COSINE_MIN,
        "fixed_cached_growing_max_abs_max": contract.FIXED37_CACHED_GROWING_MAX_ABS_MAX,
        "fixed_cached_growing_mean_abs_max": contract.FIXED37_CACHED_GROWING_MEAN_ABS_MAX,
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
        "compile_log_sha256": OTHER_SHA256,
        "test_binary_sha256": OTHER_SHA256,
        "log_sha256": OTHER_SHA256,
    }
    return {
        "schema_version": 1,
        "gate_id": contract.GATE_ID,
        "recorded_at_utc": "2026-08-30T00:00:00Z",
        "status": "passed",
        "semantic_class": "E0",
        "source": {
            "git_commit": REVISION,
            "git_dirty": False,
            "archive_sha256": ARCHIVE_SHA256,
        },
        "build": {
            "container_image_sha256": IMAGE_SHA256,
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
        "model": {**contract.EXPECTED_MODEL, "manifest_sha256": OTHER_SHA256},
        "implementations": dict(contract.EXPECTED_IMPLEMENTATIONS),
        "tests": [
            {
                "id": "cuda-compile-only",
                "result": "passed",
                "log_sha256": OTHER_SHA256,
            },
            {
                "id": "workspace-all-features-all-targets",
                "result": "passed",
                "log_sha256": OTHER_SHA256,
            },
            {
                "id": "command-batch-lifecycle",
                "result": "passed",
                "one_shot_finish": True,
                "drop_restores_stream": True,
                "log_sha256": OTHER_SHA256,
            },
            {
                "id": "command-batch-resource-ledger",
                "result": "passed",
                "validation_fail_closed": True,
                "queued_chain_raw_byte_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "stream_reuse_after_finish": True,
                "owner_close_live_allocation_count": 0,
                "log_sha256": OTHER_SHA256,
            },
            {
                "id": "smollm2-multi-step-greedy-exact",
                "result": "passed",
                "decode_steps": 16,
                "committed_iterations": 16,
                "raw_logit_mismatches": 0,
                "generated_token_ids": list(contract.EXPECTED_TOKENS),
                "token_id_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "owner_close_live_allocation_count": 0,
                "log_sha256": OTHER_SHA256,
            },
            fixed37,
        ],
    }


class OptimizerE0SemanticContractTests(unittest.TestCase):
    def validate(self, report: dict[str, object]) -> str:
        return contract.validate_final_candidate_report(
            report,
            source_revision=REVISION,
            source_archive_sha256=ARCHIVE_SHA256,
        )

    def test_valid_closed_final_candidate_report_returns_image_digest(self) -> None:
        self.assertEqual(self.validate(valid_report()), IMAGE_SHA256)

    def test_final_contract_rejects_closed_topology_source_and_policy_drift(self) -> None:
        cases: list[tuple[str, callable]] = [
            ("top-level-extra", lambda report: report.__setitem__("extra", True)),
            (
                "source",
                lambda report: report["source"].__setitem__("git_dirty", True),  # type: ignore[index,union-attr]
            ),
            (
                "model",
                lambda report: report["model"].__setitem__("weights_sha256", OTHER_SHA256),  # type: ignore[index,union-attr]
            ),
            (
                "implementations",
                lambda report: report["implementations"].__setitem__("candidate", "per-operation"),  # type: ignore[index,union-attr]
            ),
            (
                "fixed37-threshold",
                lambda report: report["tests"][-1].__setitem__(  # type: ignore[index,union-attr]
                    "fixed_cached_growing_worst_cosine", 0.1
                ),
            ),
            (
                "gpu-placeholder",
                lambda report: report["gpu"].__setitem__("model", "placeholder GPU"),  # type: ignore[index,union-attr]
            ),
            (
                "gpu-all-zero-placeholder",
                lambda report: report["gpu"].__setitem__("uuid", "0" * 64),  # type: ignore[index,union-attr]
            ),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                report = copy.deepcopy(valid_report())
                mutate(report)
                with self.assertRaises(contract.OptimizerE0SemanticContractError):
                    self.validate(report)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
