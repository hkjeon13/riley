from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "check_repeatability.py"
REPOSITORY_ROOT = SCRIPT.parents[2]
SPEC = importlib.util.spec_from_file_location("check_repeatability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_repeatability = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_repeatability
SPEC.loader.exec_module(check_repeatability)


class RepeatabilityFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.matrix_path = root / "matrix.yaml"
        self.trials_path = root / "trials.jsonl"
        prompts_path = REPOSITORY_ROOT / "benchmarks/prompts.jsonl"
        lane_path = REPOSITORY_ROOT / "benchmarks/lanes/hf-transformers.json"
        self.prompts_sha256 = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
        self.lane_sha256 = hashlib.sha256(lane_path.read_bytes()).hexdigest()
        self.matrix = {
            "contract_version": "1.0.0",
            "matrix_id": "pr01-test-matrix-v1",
            "benchmark_scope": "end-to-end",
            "model": {
                "id": "HuggingFaceTB/SmolLM2-135M",
                "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            },
            "correctness_gate": {
                "gate_id": "smollm2-fp32-bf16-native-e0-v2",
                "path": "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json",
            },
            "lane_manifests": ["benchmarks/lanes/hf-transformers.json"],
            "allowed_semantic_classes": ["reference", "E0"],
            "axes": {
                "concurrency": [1],
                "prompt_tokens": [128],
                "output_tokens": [32],
                "warm_state": ["cold", "warm"],
            },
            "measurement": {
                "independent_runs": 5,
                "warm": {"measured_iterations_per_run": 3},
                "cold": {"measured_iterations_per_run": 1},
            },
            "repeatability_gate": {
                "cells": [
                    {
                        "concurrency": 1,
                        "prompt_tokens": 128,
                        "output_tokens": 32,
                        "warm_state": "warm",
                    },
                    {
                        "concurrency": 1,
                        "prompt_tokens": 128,
                        "output_tokens": 32,
                        "warm_state": "cold",
                    },
                ],
                "thresholds": {
                    "warm_p50_cv_max": 0.05,
                    "warm_p95_cv_max": 0.05,
                    "throughput_cv_max": 0.05,
                    "cold_model_load_p50_cv_max": 0.05,
                    "peak_vram_relative_range_max": 0.02,
                    "failure_count_max": 0,
                },
            },
        }
        encoded = (json.dumps(self.matrix, sort_keys=True, indent=2) + "\n").encode()
        self.matrix_path.write_bytes(encoded)
        self.matrix_sha256 = hashlib.sha256(encoded).hexdigest()
        self.rows = self._rows()
        self.write_rows()

    def _rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for run_number in range(5):
            for warm_state in ("warm", "cold"):
                measured_iterations = 3 if warm_state == "warm" else 1
                for trial_index in range(1, measured_iterations + 1):
                    factor = 1.0 + run_number * 0.002 + trial_index * 0.001
                    row: dict[str, object] = {
                        "contract_version": "1.0.0",
                        "trial_id": (
                            f"run-{run_number}:{warm_state}:i{trial_index}"
                        ),
                        "run_id": f"run-{run_number}",
                        "trial_index": trial_index,
                        "recorded_at_utc": "2026-08-24T00:00:00Z",
                        "scope": "end-to-end",
                        "matrix_id": self.matrix["matrix_id"],
                        "matrix_sha256": self.matrix_sha256,
                        "prompts_sha256": self.prompts_sha256,
                        "lane_manifest_sha256": self.lane_sha256,
                        "environment_id": "rtx4090-ubuntu22-driver580-v1",
                        "semantic_class": "reference",
                        "correctness_gate_id": None,
                        "correctness_report_sha256": None,
                        "implementation_id": "hf-transformers-eager",
                        "reference_implementation": "hf-transformers-eager",
                        "runtime_dependency_class": "python-reference",
                        "approximation_enabled": False,
                        "error_budget": None,
                        "seed": None,
                        "warm_state": warm_state,
                        "model_id": "HuggingFaceTB/SmolLM2-135M",
                        "model_revision": (
                            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
                        ),
                        "engine_revision": "transformers-5.15.1+torch-2.13.0",
                        "dtype": "bf16",
                        "environment": {
                            "gpu_model": "NVIDIA GeForce RTX 4090",
                            "compute_capability": "8.9",
                            "gpu_count": 1,
                            "cpu_model": "Intel Core i7-13700K (synthetic)",
                            "ram_bytes": 67_185_598_464,
                            "os": "Ubuntu 22.04 synthetic",
                            "nvidia_driver_version": "580.173.02",
                            "cuda_toolkit_version": "test toolkit",
                            "cuda_runtime_version": "test runtime",
                        },
                        "provenance": {
                            "git_revision": "d" * 40,
                            "git_dirty": False,
                        },
                        "status": "success",
                        "failure_reason": None,
                        "failure_count": 0,
                        "workload": {
                            "concurrency": 1,
                            "prompt_tokens": 128,
                            "output_tokens": 32,
                            "sampling_id": "greedy",
                            "warm_state": warm_state,
                        },
                        "microbenchmark": None,
                        "metrics": {
                            "model_load_ms": 800.0 * factor,
                            "batch_wall_ms": 150.0 * factor,
                            "output_tokens_per_second": 500.0 / factor,
                            "cpu_utilization_percent": None,
                            "gpu_utilization_percent": None,
                            "peak_vram_bytes": int(4_000_000_000 * factor),
                        },
                        "requests": [
                            {
                                "request_id": "request-0",
                                "prompt_id": "short-en",
                                "prompt_token_ids_sha256": "a" * 64,
                                "generated_token_ids_sha256": "b" * 64,
                                "status": "success",
                                "failure_reason": None,
                                "prompt_tokens": 128,
                                "requested_output_tokens": 32,
                                "generated_tokens": 32,
                                "ttft_ms": 20.0 * factor,
                                "end_to_end_ms": 100.0 * factor,
                                "mean_tpot_ms": 2.5 * factor,
                                "itl_ms": [2.5 * factor] * 31,
                            }
                        ],
                        "speculative": {
                            "draft_model": None,
                            "lookahead": None,
                            "acceptance_rate": None,
                            "accepted_tokens_per_verify": None,
                            "target_calls_per_output_token": None,
                            "draft_latency_ms": None,
                            "verification_latency_ms": None,
                            "rejected_suffix_tokens": None,
                            "rollback_count": None,
                        },
                        "sparse_attention": {
                            "selected_pages": None,
                            "total_pages": None,
                            "page_metadata_bytes": None,
                            "page_bound_time_ms": None,
                            "omitted_mass_bound": None,
                            "exact_fallback_rate": None,
                        },
                        "quantization": {
                            "weight_format": None,
                            "activation_format": None,
                            "kv_format": None,
                            "calibration_revision": None,
                            "transform_runtime_ms": None,
                            "weight_bytes": None,
                            "kv_bytes": None,
                            "gemm_throughput_tflops": None,
                        },
                    }
                    rows.append(row)
        return rows

    def write_rows(self) -> None:
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows)
        self.trials_path.write_text(text, encoding="utf-8")


class StatisticsTests(unittest.TestCase):
    def test_r7_and_across_run_statistics(self) -> None:
        self.assertEqual(check_repeatability.r7([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(check_repeatability.r7([1, 2, 3, 4], 0.95), 3.85)
        self.assertEqual(check_repeatability.sample_cv([10, 10, 10]), 0.0)
        self.assertAlmostEqual(
            check_repeatability.relative_range([90, 100, 110]), 0.2
        )


class RepeatabilityGateTests(unittest.TestCase):
    def test_noncanonical_matrix_requires_explicit_test_only_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            report = check_repeatability.evaluate(
                fixture.matrix_path, [fixture.trials_path]
            )
            self.assertEqual(report["status"], "error")
            self.assertFalse(report["passed"])
            self.assertIn("requires the canonical", report["errors"][0])

    def test_synthetic_warm_and_cold_cells_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            report = check_repeatability.evaluate(
                fixture.matrix_path,
                [fixture.trials_path],
                allow_noncanonical_matrix=True,
            )

            self.assertTrue(report["passed"], report)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                report["contract_version"],
                check_repeatability.REPORT_CONTRACT_VERSION,
            )
            self.assertEqual(len(report["cells"]), 2)
            by_state = {
                cell["workload"]["warm_state"]: cell for cell in report["cells"]
            }
            self.assertEqual(by_state["warm"]["independent_run_count"], 5)
            self.assertEqual(by_state["warm"]["required_trials_per_run"], 3)
            self.assertEqual(by_state["cold"]["required_trials_per_run"], 1)
            self.assertTrue(
                all(
                    run["trial_count"] == 1
                    for run in by_state["cold"]["run_summaries"]
                )
            )
            self.assertIn(
                "warm_end_to_end_r7_p95_sample_cv", by_state["warm"]["statistics"]
            )
            self.assertIn(
                "warm_request_mean_tpot_r7_p95_sample_cv",
                by_state["warm"]["statistics"],
            )
            self.assertIn(
                "warm_pooled_itl_r7_p95_sample_cv",
                by_state["warm"]["statistics"],
            )
            for run in by_state["warm"]["run_summaries"]:
                self.assertEqual(run["request_mean_tpot_ms"]["observation_count"], 3)
                self.assertEqual(run["pooled_itl_ms"]["observation_count"], 93)
            self.assertIn(
                "cold_model_load_r7_p50_sample_cv", by_state["cold"]["statistics"]
            )
            self.assertIn(
                "throughput_p50_sample_cv", by_state["cold"]["statistics"]
            )
            self.assertEqual(
                {check["name"] for check in by_state["warm"]["checks"]},
                {
                    "throughput_cv_max",
                    "peak_vram_relative_range_max",
                    "warm_p50_cv_max",
                    "warm_p95_cv_max",
                    "failure_count_max",
                },
            )
            self.assertEqual(
                {check["name"] for check in by_state["cold"]["checks"]},
                {
                    "peak_vram_relative_range_max",
                    "cold_model_load_p50_cv_max",
                    "failure_count_max",
                },
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = check_repeatability.main(
                    [
                        "--matrix",
                        str(fixture.matrix_path),
                        "--allow-noncanonical-matrix",
                        str(fixture.trials_path),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_cold_throughput_outlier_is_reported_but_not_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            for row in fixture.rows:
                if row["run_id"] == "run-4" and row["warm_state"] == "cold":
                    metrics = row["metrics"]
                    assert isinstance(metrics, dict)
                    metrics["output_tokens_per_second"] = float(
                        metrics["output_tokens_per_second"]
                    ) / 4.0
            fixture.write_rows()

            report = check_repeatability.evaluate(
                fixture.matrix_path,
                [fixture.trials_path],
                allow_noncanonical_matrix=True,
            )

            self.assertTrue(report["passed"], report)
            cold = next(
                cell
                for cell in report["cells"]
                if cell["workload"]["warm_state"] == "cold"
            )
            self.assertGreater(
                cold["statistics"]["throughput_p50_sample_cv"],
                fixture.matrix["repeatability_gate"]["thresholds"][
                    "throughput_cv_max"
                ],
            )
            self.assertNotIn(
                "throughput_cv_max", {check["name"] for check in cold["checks"]}
            )

    def test_cold_model_load_outlier_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            for row in fixture.rows:
                if row["run_id"] == "run-4" and row["warm_state"] == "cold":
                    metrics = row["metrics"]
                    assert isinstance(metrics, dict)
                    metrics["model_load_ms"] = float(metrics["model_load_ms"]) * 4.0
            fixture.write_rows()

            report = check_repeatability.evaluate(
                fixture.matrix_path,
                [fixture.trials_path],
                allow_noncanonical_matrix=True,
            )

            self.assertFalse(report["passed"])
            cold = next(
                cell
                for cell in report["cells"]
                if cell["workload"]["warm_state"] == "cold"
            )
            failed_checks = {
                check["name"] for check in cold["checks"] if not check["passed"]
            }
            self.assertEqual(failed_checks, {"cold_model_load_p50_cv_max"})

    def test_rows_outside_exact_gate_cells_fail_instead_of_being_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            fixture.matrix["axes"]["prompt_tokens"] = [128, 256]
            encoded = (
                json.dumps(fixture.matrix, sort_keys=True, indent=2) + "\n"
            ).encode()
            fixture.matrix_path.write_bytes(encoded)
            fixture.matrix_sha256 = hashlib.sha256(encoded).hexdigest()
            for row in fixture.rows:
                row["matrix_sha256"] = fixture.matrix_sha256

            extra = json.loads(json.dumps(fixture.rows[0]))
            extra["trial_id"] = "unexpected-gate-cell"
            extra["workload"]["prompt_tokens"] = 256
            extra["requests"][0]["prompt_tokens"] = 256
            fixture.rows.append(extra)
            fixture.write_rows()

            report = check_repeatability.evaluate(
                fixture.matrix_path,
                [fixture.trials_path],
                allow_noncanonical_matrix=True,
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["ignored_trial_count"], 1)
            self.assertTrue(
                any("outside the exact repeatability gate" in error for error in report["errors"]),
                report,
            )

    def test_synthetic_variance_fails_thresholds_and_cli_is_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            for row in fixture.rows:
                if row["run_id"] == "run-4" and row["warm_state"] == "warm":
                    metrics = row["metrics"]
                    assert isinstance(metrics, dict)
                    metrics["output_tokens_per_second"] = float(
                        metrics["output_tokens_per_second"]
                    ) / 4.0
                    metrics["peak_vram_bytes"] = int(metrics["peak_vram_bytes"] * 1.5)
                    requests = row["requests"]
                    assert isinstance(requests, list)
                    for request in requests:
                        assert isinstance(request, dict)
                        request["end_to_end_ms"] = float(request["end_to_end_ms"]) * 4.0
            fixture.write_rows()

            report = check_repeatability.evaluate(
                fixture.matrix_path,
                [fixture.trials_path],
                allow_noncanonical_matrix=True,
            )
            self.assertFalse(report["passed"])
            warm = next(
                cell
                for cell in report["cells"]
                if cell["workload"]["warm_state"] == "warm"
            )
            failed_checks = {
                check["name"] for check in warm["checks"] if not check["passed"]
            }
            self.assertIn("warm_p50_cv_max", failed_checks)
            self.assertIn("warm_p95_cv_max", failed_checks)
            self.assertIn("throughput_cv_max", failed_checks)
            self.assertIn("peak_vram_relative_range_max", failed_checks)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = check_repeatability.main(
                    [
                        "--matrix",
                        str(fixture.matrix_path),
                        "--allow-noncanonical-matrix",
                        str(fixture.trials_path),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "failed")

    def test_missing_measured_trial_fails_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            fixture.rows = [
                row
                for row in fixture.rows
                if not (
                    row["run_id"] == "run-0"
                    and row["warm_state"] == "warm"
                    and row["trial_index"] == 3
                )
            ]
            fixture.write_rows()

            report = check_repeatability.evaluate(
                fixture.matrix_path,
                [fixture.trials_path],
                allow_noncanonical_matrix=True,
            )
            self.assertFalse(report["passed"])
            warm = next(
                cell
                for cell in report["cells"]
                if cell["workload"]["warm_state"] == "warm"
            )
            self.assertIn("trial_index 1..3", "\n".join(warm["errors"]))

    def test_nonzero_failure_count_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            failed = fixture.rows[0]
            failed["failure_count"] = 1
            failed["status"] = "failure"
            failed["failure_reason"] = "synthetic failure"
            requests = failed["requests"]
            assert isinstance(requests, list) and isinstance(requests[0], dict)
            requests[0].update(
                status="failure",
                failure_reason="synthetic failure",
                generated_tokens=0,
                generated_token_ids_sha256=hashlib.sha256(b"").hexdigest(),
                ttft_ms=None,
                end_to_end_ms=None,
                mean_tpot_ms=None,
                itl_ms=None,
            )
            fixture.write_rows()

            report = check_repeatability.evaluate(
                fixture.matrix_path,
                [fixture.trials_path],
                allow_noncanonical_matrix=True,
            )
            self.assertFalse(report["passed"])
            warm = next(
                cell
                for cell in report["cells"]
                if cell["workload"]["warm_state"] == "warm"
            )
            failure_check = next(
                check
                for check in warm["checks"]
                if check["name"] == "failure_count_max"
            )
            self.assertEqual(failure_check["value"], 1.0)
            self.assertFalse(failure_check["passed"])

    def test_incomparable_or_dirty_trials_are_rejected(self) -> None:
        mutators = {
            "dirty git": (
                lambda row: row["provenance"].update(git_dirty=True),
                "incomparable",
            ),
            "prompt hash": (
                lambda row: row.update(prompts_sha256="c" * 64),
                "incomparable",
            ),
            "environment": (
                lambda row: row.update(environment_id="other-environment"),
                "incomparable",
            ),
            "scope": (lambda row: row.update(scope="microbenchmark"), "error"),
        }
        for label, (mutate, expected_status) in mutators.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = RepeatabilityFixture(Path(directory))
                mutate(fixture.rows[-1])
                fixture.write_rows()
                report = check_repeatability.evaluate(
                    fixture.matrix_path,
                    [fixture.trials_path],
                    allow_noncanonical_matrix=True,
                )
                self.assertFalse(report["passed"])
                self.assertEqual(report["status"], expected_status)
                self.assertTrue(report["errors"])

    def test_schema_and_cross_file_validation_precede_statistics(self) -> None:
        mutators = {
            "missing generated hash": lambda row: row["requests"][0].pop(
                "generated_token_ids_sha256"
            ),
            "request count": lambda row: row["requests"].append(
                dict(row["requests"][0], request_id="unexpected-request")
            ),
            "generated count": lambda row: row["requests"][0].update(
                generated_tokens=31
            ),
            "ITL length": lambda row: row["requests"][0]["itl_ms"].pop(),
            "ITL mean": lambda row: row["requests"][0].update(
                mean_tpot_ms=999.0
            ),
        }
        for label, mutate in mutators.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = RepeatabilityFixture(Path(directory))
                mutate(fixture.rows[0])
                fixture.write_rows()
                report = check_repeatability.evaluate(
                    fixture.matrix_path,
                    [fixture.trials_path],
                    allow_noncanonical_matrix=True,
                )
                self.assertEqual(report["status"], "error")
                self.assertFalse(report["passed"])
                self.assertIn("raw result contract validation", report["errors"][0])
                self.assertEqual(report["cells"], [])

    def test_generated_token_identity_must_match_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RepeatabilityFixture(Path(directory))
            changed = next(
                row
                for row in fixture.rows
                if row["run_id"] == "run-4"
                and row["warm_state"] == "warm"
                and row["trial_index"] == 1
            )
            requests = changed["requests"]
            assert isinstance(requests, list) and isinstance(requests[0], dict)
            requests[0]["generated_token_ids_sha256"] = "c" * 64
            fixture.write_rows()

            report = check_repeatability.evaluate(
                fixture.matrix_path,
                [fixture.trials_path],
                allow_noncanonical_matrix=True,
            )
            self.assertEqual(report["status"], "failed")
            warm = next(
                cell
                for cell in report["cells"]
                if cell["workload"]["warm_state"] == "warm"
            )
            self.assertIn("generated token identity", "\n".join(warm["errors"]))


if __name__ == "__main__":
    unittest.main()
