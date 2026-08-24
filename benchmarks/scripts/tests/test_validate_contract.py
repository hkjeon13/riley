from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

import validate_contract as contract  # noqa: E402


class ContractValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_schema = json.loads(
            (REPOSITORY_ROOT / "benchmarks/schemas/result.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.prompt_schema = json.loads(
            (REPOSITORY_ROOT / "benchmarks/schemas/prompt.schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_repository_contract_is_valid(self) -> None:
        counts = contract.validate_contract(REPOSITORY_ROOT)
        self.assertEqual(counts["lanes"], 3)

    def test_vllm_lane_has_an_available_single_cell_adapter(self) -> None:
        lane = json.loads(
            (REPOSITORY_ROOT / "benchmarks/lanes/vllm.json").read_text(encoding="utf-8")
        )
        self.assertEqual(lane["availability"], "available")
        command = lane["commands"]["benchmark"]
        self.assertEqual(command["status"], "available")
        self.assertIn("rustinfer-vllm-benchmark", command["argv"])
        self.assertEqual(
            command["environment"],
            {
                "DO_NOT_TRACK": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "VLLM_DO_NOT_TRACK": "1",
                "VLLM_NO_USAGE_STATS": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
        )
        for flag in (
            "--warm-state",
            "--concurrency",
            "--prompt-tokens",
            "--output-tokens",
        ):
            self.assertIn(flag, command["argv"])

    def test_native_lane_contract_is_single_cell_per_process(self) -> None:
        lane = json.loads(
            (REPOSITORY_ROOT / "benchmarks/lanes/rustinfer-native.json").read_text(
                encoding="utf-8"
            )
        )
        command = lane["commands"]["benchmark"]
        self.assertEqual(command["status"], "contract-only")
        for flag in contract.SINGLE_CELL_BENCHMARK_FLAGS:
            self.assertIn(flag, command["argv"])

    def test_dependency_lock_rejects_changed_reproducibility_pins(self) -> None:
        manifest = "benchmarks/lanes/vllm/pyproject.toml"
        lane = {"lane_id": "vllm", "dependency_manifest": manifest}
        cases = {
            "requires-python": 'requires-python = ">=3.13"',
            "exclude-newer": 'exclude-newer = "2026-08-25T00:00:00Z"',
            "vllm version": 'version = "0.27.0"',
        }
        for expected_message, replacement in cases.items():
            with self.subTest(expected_message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                project_dir = root / "benchmarks/lanes/vllm"
                project_dir.mkdir(parents=True)
                (project_dir / "pyproject.toml").write_text(
                    """\
[project]
name = "rustinfer-vllm-benchmark-lane"
requires-python = ">=3.13,<3.14"
dependencies = [
  "nvidia-ml-py==13.610.43",
  "psutil==7.2.2",
  "vllm==0.27.1",
]

[project.scripts]
rustinfer-vllm-benchmark = "rustinfer_vllm_benchmark.cli:main"
""",
                    encoding="utf-8",
                )
                (project_dir / ".python-version").write_text(
                    "3.13.15\n", encoding="utf-8"
                )
                lock = """\
requires-python = "==3.13.*"

[options]
exclude-newer = "2026-08-24T23:59:59Z"

[[package]]
name = "rustinfer-vllm-benchmark-lane"
source = { editable = "." }

[package.metadata]
requires-dist = [
  { name = "nvidia-ml-py", specifier = "==13.610.43" },
  { name = "psutil", specifier = "==7.2.2" },
  { name = "vllm", specifier = "==0.27.1" },
]

[[package]]
name = "nvidia-ml-py"
version = "13.610.43"

[[package]]
name = "psutil"
version = "7.2.2"

[[package]]
name = "vllm"
version = "0.27.1"
"""
                if expected_message == "requires-python":
                    lock = lock.replace('requires-python = "==3.13.*"', replacement)
                elif expected_message == "exclude-newer":
                    lock = lock.replace(
                        'exclude-newer = "2026-08-24T23:59:59Z"', replacement
                    )
                else:
                    lock = lock.replace('version = "0.27.1"', replacement)
                (project_dir / "uv.lock").write_text(lock, encoding="utf-8")
                with self.assertRaisesRegex(contract.ContractError, expected_message):
                    contract.validate_dependency_project(root, lane)

    def test_matrix_rejects_post_hoc_threshold_relaxation(self) -> None:
        matrix = json.loads(
            (REPOSITORY_ROOT / "benchmarks/matrix.yaml").read_text(encoding="utf-8")
        )
        matrix["repeatability_gate"]["thresholds"]["warm_p50_cv_max"] = 0.5
        with self.assertRaisesRegex(contract.ContractError, "warm_p50_cv_max"):
            contract.validate_matrix(matrix, REPOSITORY_ROOT)

    def test_matrix_rejects_cache_scope_or_prime_profile_changes(self) -> None:
        canonical = json.loads(
            (REPOSITORY_ROOT / "benchmarks/matrix.yaml").read_text(encoding="utf-8")
        )
        mutations = (
            ("cold scope", lambda value: value["cache_policy"].__setitem__("cold_scope", "filesystem-cold")),
            ("uv version", lambda value: value["cache_policy"].__setitem__("uv_version", "uv 0.12.6")),
            ("Python hash", lambda value: value["cache_policy"].__setitem__("python_linux_x86_64_sha256", "0" * 64)),
            ("CUDA cache size", lambda value: value["cache_policy"].__setitem__("cuda_cache_maxsize", "1")),
            (
                "HF prime profile",
                lambda value: value["cache_policy"]["lane_prime_cells"][
                    "hf-transformers"
                ][0].__setitem__("concurrency", 2),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label):
                matrix = json.loads(json.dumps(canonical))
                mutate(matrix)
                with self.assertRaisesRegex(contract.ContractError, "cache_policy"):
                    contract.validate_matrix(matrix, REPOSITORY_ROOT)

    def test_result_schema_accepts_nullable_observability_metrics(self) -> None:
        contract.validate_instance(self._successful_result(), self.result_schema)

    def test_result_schema_rejects_unknown_fields(self) -> None:
        result = self._successful_result()
        result["unreviewed_metric"] = 1
        with self.assertRaisesRegex(contract.ContractError, "unexpected properties"):
            contract.validate_instance(result, self.result_schema)

    def test_disabled_approximation_requires_null_error_budget(self) -> None:
        result = self._successful_result()
        result["error_budget"] = 0.01
        with self.assertRaisesRegex(contract.ContractError, "type"):
            contract.validate_instance(result, self.result_schema)

    def test_request_requires_exact_token_identity_hash(self) -> None:
        result = self._successful_result()
        del result["requests"][0]["prompt_token_ids_sha256"]
        with self.assertRaisesRegex(contract.ContractError, "prompt_token_ids_sha256"):
            contract.validate_instance(result, self.result_schema)

    def test_request_requires_generated_token_identity_hash(self) -> None:
        result = self._successful_result()
        del result["requests"][0]["generated_token_ids_sha256"]
        with self.assertRaisesRegex(
            contract.ContractError, "generated_token_ids_sha256"
        ):
            contract.validate_instance(result, self.result_schema)

    def test_seed_is_bounded_to_unsigned_64_bit(self) -> None:
        result = self._successful_result()
        result["seed"] = 18_446_744_073_709_551_615
        contract.validate_instance(result, self.result_schema)
        result["seed"] = 18_446_744_073_709_551_616
        with self.assertRaisesRegex(contract.ContractError, "must be <="):
            contract.validate_instance(result, self.result_schema)

    def test_success_request_requires_actual_itl_array(self) -> None:
        result = self._successful_result()
        result["requests"][0]["itl_ms"] = 0.03
        with self.assertRaisesRegex(contract.ContractError, "oneOf|array"):
            contract.validate_instance(result, self.result_schema)

    def test_failure_request_requires_canonical_empty_observation(self) -> None:
        result = self._successful_result()
        request = result["requests"][0]
        request.update(
            status="failure",
            failure_reason="synthetic failure",
            generated_tokens=0,
            generated_token_ids_sha256=contract.EMPTY_TOKEN_IDS_SHA256,
            ttft_ms=None,
            end_to_end_ms=None,
            mean_tpot_ms=None,
            itl_ms=None,
        )
        contract.validate_instance(result, self.result_schema)
        request["ttft_ms"] = 1.0
        with self.assertRaisesRegex(contract.ContractError, "null"):
            contract.validate_instance(result, self.result_schema)

    def test_prompt_target_length_must_be_positive(self) -> None:
        prompt = {
            "contract_version": "1.0.0",
            "prompt_id": "test",
            "category": "short",
            "language": "en",
            "text": "x",
            "target_prompt_tokens": 1,
            "boundary_kind": "none",
            "expected_behavior": "normal-generation",
            "contains_sensitive_data": False,
        }
        contract.validate_instance(prompt, self.prompt_schema)
        prompt["target_prompt_tokens"] = 0
        with self.assertRaisesRegex(contract.ContractError, "must be >= 1"):
            contract.validate_instance(prompt, self.prompt_schema)

    def test_failed_result_requires_nonzero_failure_count(self) -> None:
        result = self._successful_result()
        result["status"] = "failure"
        result["failure_reason"] = "out of memory"
        with self.assertRaisesRegex(contract.ContractError, "must be >= 1"):
            contract.validate_instance(result, self.result_schema)

    def test_successful_e0_result_requires_correctness_evidence(self) -> None:
        result = self._successful_result()
        result["semantic_class"] = "E0"
        with self.assertRaisesRegex(contract.ContractError, "correctness_gate_id"):
            contract.validate_instance(result, self.result_schema)
        result["correctness_gate_id"] = "greedy-exact-v1"
        result["correctness_report_sha256"] = "5" * 64
        contract.validate_instance(result, self.result_schema)

    def test_contract_only_e0_rejects_schema_valid_forged_pass_report(self) -> None:
        matrix_path = REPOSITORY_ROOT / "benchmarks/matrix.yaml"
        prompts_path = REPOSITORY_ROOT / "benchmarks/prompts.jsonl"
        gate_path = (
            REPOSITORY_ROOT
            / "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v1.json"
        )
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        report_schema = json.loads(
            (
                REPOSITORY_ROOT
                / "benchmarks/schemas/correctness-report.schema.json"
            ).read_text(encoding="utf-8")
        )
        forged_report = self._forged_passing_correctness_report(
            gate=gate,
            matrix_sha256=contract._sha256(matrix_path),
            prompts_sha256=contract._sha256(prompts_path),
            gate_sha256=contract._sha256(gate_path),
        )
        # This is deliberately coherent enough to pass the structural schema;
        # no raw manifest, sidecar, or executable produced it.
        contract.validate_instance(forged_report, report_schema)
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            report_path = result_dir / "correctness-report.json"
            report_path.write_text(
                json.dumps(forged_report, sort_keys=True) + "\n", encoding="utf-8"
            )
            result = self._successful_result()
            result.update(
                semantic_class="E0",
                correctness_gate_id=matrix["correctness_gate"]["gate_id"],
                correctness_report_sha256=contract._sha256(report_path),
                matrix_sha256=contract._sha256(matrix_path),
                prompts_sha256=contract._sha256(prompts_path),
            )
            raw_path = result_dir / "raw.jsonl"
            raw_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                contract.ContractError, "contract-only.*raw evidence replay"
            ):
                contract.validate_result_file(
                    raw_path,
                    self.result_schema,
                    matrix,
                    matrix_path,
                    prompts_path,
                    {},
                    {},
                )

    @staticmethod
    def _forged_passing_correctness_report(
        *,
        gate: dict[str, object],
        matrix_sha256: str,
        prompts_sha256: str,
        gate_sha256: str,
    ) -> dict[str, object]:
        metrics = {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "max_relative": 0.0,
            "mean_relative": 0.0,
            "cosine_similarity": 1.0,
        }
        numeric = {
            name: {"metrics": dict(metrics), "pass": True}
            for name in ("first_layer_hidden", "final_logits", "final_log_probs")
        }
        semantic = {
            "cache_on_exact": True,
            "cache_off_exact": True,
            "top_1_exact": True,
            "top_k_set_exact": True,
            "hf_cross_cache_first_divergence_step": None,
            "candidate_cross_cache_first_divergence_step": None,
            "cross_cache_exact_window": 16,
            "cross_cache_exact_window_match": True,
            "pass": True,
        }
        variant = {
            "numeric": numeric,
            "semantic": semantic,
            "pass": True,
        }
        variant_ids = ("canonical-v1", "fixed-contiguous-37-balanced-v1")
        cases = [
            {
                "prompt_id": f"forged-{index:02d}",
                "variants": {
                    variant_id: json.loads(json.dumps(variant))
                    for variant_id in variant_ids
                },
                "pass": True,
            }
            for index in range(31)
        ]
        variant_summary = {
            "case_count": 31,
            "failure_count": 0,
            "numeric_pass": True,
            "semantic_pass": True,
            "aggregate_numeric": numeric,
            "pass": True,
        }
        threshold_tensors = gate["numeric"]["tensors"]
        return {
            "schema_version": "1.0.0",
            "gate_id": gate["gate_id"],
            "created_at": "2026-08-24T00:00:00Z",
            "status": "pass",
            "roles": {
                "fp32": "numeric-only",
                "bf16": "semantic-only",
                "candidate_numeric_reference": "fp32",
                "candidate_semantic_reference": "hf-bf16-path-matched",
            },
            "gate_contract": {
                "thresholds": {
                    name: threshold_tensors[name]["thresholds"]
                    for name in (
                        "first_layer_hidden",
                        "final_logits",
                        "final_log_probs",
                    )
                },
                "oracle_reduction_variant": gate["reduction_variants"]["oracle"],
                "required_candidate_reduction_variants": gate["reduction_variants"][
                    "required_candidate"
                ],
                "cross_cache_exact_window": 16,
                "top_k_comparison": "set-exact",
                "top_1_comparison": "ordered-exact",
                "threshold_activation_evidence": (
                    "replayed-passing-full-31-hf-oracle-calibration-report"
                ),
            },
            "inputs": {
                key: "a" * 64
                for key in (
                    "fp32_manifest_sha256",
                    "bf16_manifest_sha256",
                    "candidate_manifest_sha256",
                    "oracle_calibration_report_sha256",
                    "fp32_sidecar_sha256",
                    "bf16_sidecar_sha256",
                    "candidate_sidecar_sha256",
                )
            },
            "bindings": {
                "model_id": contract.MODEL_ID,
                "model_revision": contract.MODEL_REVISION,
                "config_sha256": contract.CONFIG_SHA256,
                "weights_sha256": contract.WEIGHTS_SHA256,
                "tokenizer_sha256": contract.TOKENIZER_SHA256,
                "matrix_sha256": matrix_sha256,
                "prompts_sha256": prompts_sha256,
                "gate_manifest_sha256": gate_sha256,
                "environment_sha256": "b" * 64,
                "environment_id": contract.PRIMARY_ENVIRONMENT_ID,
                "oracle_git_revision": "c" * 40,
                "oracle_git_status_sha256": contract.EMPTY_TOKEN_IDS_SHA256,
                "candidate_git_revision": "d" * 40,
                "candidate_git_status_sha256": contract.EMPTY_TOKEN_IDS_SHA256,
                "candidate_executable_sha256": "e" * 64,
                "candidate_build_argv_sha256": "f" * 64,
                "candidate_capture_argv_sha256": "0" * 64,
                "dependency_locks": {
                    "fp32": "1" * 64,
                    "bf16": "1" * 64,
                    "candidate": "2" * 64,
                },
                "lane_manifests": {
                    "fp32": "3" * 64,
                    "bf16": "3" * 64,
                    "candidate": "4" * 64,
                },
            },
            "summary": {
                "case_count": 31,
                "candidate_variant_count": 2,
                "failure_count": 0,
                "numeric_pass": True,
                "semantic_pass": True,
                "variants": {
                    variant_id: json.loads(json.dumps(variant_summary))
                    for variant_id in variant_ids
                },
            },
            "cases": cases,
        }

    def _successful_result(self) -> dict[str, object]:
        return {
            "contract_version": "1.0.0",
            "trial_id": "trial-1",
            "run_id": "run-1",
            "trial_index": 1,
            "recorded_at_utc": "2026-08-24T00:00:00Z",
            "scope": "end-to-end",
            "matrix_id": "smollm2-135m-rtx4090-bf16-v1",
            "matrix_sha256": "0" * 64,
            "prompts_sha256": "1" * 64,
            "lane_manifest_sha256": "2" * 64,
            "environment_id": "rtx4090-primary",
            "semantic_class": "reference",
            "correctness_gate_id": None,
            "correctness_report_sha256": None,
            "implementation_id": "hf-transformers-eager",
            "reference_implementation": "hf-transformers-eager",
            "runtime_dependency_class": "python-reference",
            "approximation_enabled": False,
            "error_budget": None,
            "seed": None,
            "warm_state": "warm",
            "model_id": "HuggingFaceTB/SmolLM2-135M",
            "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "engine_revision": "transformers-5.15.1+torch-2.13.0",
            "dtype": "bf16",
            "environment": {
                "gpu_model": "NVIDIA GeForce RTX 4090",
                "compute_capability": "8.9",
                "gpu_count": 1,
                "cpu_model": "test cpu",
                "ram_bytes": 1,
                "os": "test os",
                "nvidia_driver_version": "test driver",
                "cuda_toolkit_version": "test toolkit",
                "cuda_runtime_version": "test runtime",
            },
            "provenance": {"git_revision": "a" * 40, "git_dirty": False},
            "status": "success",
            "failure_reason": None,
            "failure_count": 0,
            "workload": {
                "concurrency": 1,
                "prompt_tokens": 128,
                "output_tokens": 32,
                "sampling_id": "greedy",
                "warm_state": "warm",
            },
            "microbenchmark": None,
            "metrics": {
                "model_load_ms": 1.0,
                "batch_wall_ms": 2.0,
                "output_tokens_per_second": 16.0,
                "cpu_utilization_percent": None,
                "gpu_utilization_percent": None,
                "peak_vram_bytes": None,
            },
            "requests": [
                {
                    "request_id": "request-1",
                    "prompt_id": "short-en",
                    "prompt_token_ids_sha256": "3" * 64,
                    "generated_token_ids_sha256": "4" * 64,
                    "status": "success",
                    "failure_reason": None,
                    "prompt_tokens": 128,
                    "requested_output_tokens": 32,
                    "generated_tokens": 32,
                    "ttft_ms": 1.0,
                    "end_to_end_ms": 2.0,
                    "mean_tpot_ms": 1.0 / 31.0,
                    "itl_ms": [1.0 / 31.0] * 31,
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


if __name__ == "__main__":
    unittest.main()
