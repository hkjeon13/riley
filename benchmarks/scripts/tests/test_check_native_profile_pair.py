from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "check_native_profile_pair.py"
REPOSITORY_ROOT = SCRIPT.parents[2]
SPEC = importlib.util.spec_from_file_location("check_native_profile_pair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def measured(value: int | float) -> dict[str, object]:
    return {"validity": "measured", "value": value}


class ProfilePairFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.baseline_paths: list[Path] = []
        self.candidate_paths: list[Path] = []
        self.baseline: list[dict[str, object]] = []
        self.candidate: list[dict[str, object]] = []
        for pair_index in range(1, 6):
            self.baseline.append(self._run("baseline", pair_index))
            self.candidate.append(self._run("candidate", pair_index))
        self.write()

    def _run(self, role: str, pair_index: int) -> dict[str, object]:
        candidate = role == "candidate"
        run_factor = 1.0 + (pair_index - 3) * 0.002
        ttft = (90.0 if candidate else 100.0) * run_factor
        tpot = (9.0 if candidate else 10.0) * run_factor
        e2e = (405.0 if candidate else 450.0) * run_factor
        throughput = (110.0 if candidate else 100.0) / run_factor
        requests = []
        for input_index in range(30):
            request_factor = 1.0 + input_index * 0.001
            requests.append(
                {
                    "input_index": input_index,
                    "prompt_u32le_sha256": digest(f"prompt-{input_index}"),
                    "generated_u32le_sha256": digest(f"generated-{input_index}"),
                    "prompt_token_count": 128,
                    "requested_output_token_count": 32,
                    "generated_token_count": 32,
                    "ttft_ms": ttft * request_factor,
                    "tpot_ms": tpot * request_factor,
                    "e2e_ms": e2e * request_factor,
                }
            )
        return {
            "schema_version": "rustinfer.native-profile-run.v1",
            "role": role,
            "pair_index": pair_index,
            "run_id": f"{role}-run-{pair_index}",
            "recorded_at_utc": f"2026-08-26T00:00:0{pair_index}Z",
            "status": "success",
            "failure_count": 0,
            "source": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "executable_sha256": "b" * 64,
                "implementation_id": f"native-residual-rmsnorm-{'fused' if candidate else 'separate'}",
                "runtime_flag": {
                    "name": "residual_rmsnorm",
                    "value": "fused" if candidate else "separate",
                },
                "semantic_class": "E0",
                "correctness_gate_id": "pr15-fused-residual-rmsnorm-exact-v1",
                "correctness_report_sha256": "c" * 64,
            },
            "environment": {
                "gpu": {
                    "model": "NVIDIA GeForce RTX 4090",
                    "uuid": "GPU-synthetic-4090",
                    "device_index": 0,
                    "pci_bus_id": "00000000:01:00.0",
                    "compute_capability": "8.9",
                    "vram_bytes": 24_564 * 1024 * 1024,
                },
                "host": {
                    "environment_id": "server-4096-v1",
                    "cpu_model": "Intel Core i7-13700K",
                    "physical_core_count": 16,
                    "logical_core_count": 24,
                    "ram_bytes": 67_185_598_464,
                    "os_release": "Ubuntu 22.04",
                    "kernel_release": "6.8.0-138-generic",
                    "architecture": "x86_64",
                },
                "software": {
                    "nvidia_driver_version": "580.173.02",
                    "cuda_runtime_version": "13.0",
                    "cuda_toolkit_version": "13.0",
                    "cublas_version": "13.0",
                    "container_image_sha256": "d" * 64,
                },
            },
            "workload": {
                "workload_id": "warm-c1-p128-o32",
                "model_id": "HuggingFaceTB/SmolLM2-135M",
                "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "weights_sha256": "e" * 64,
                "tokenizer_sha256": "f" * 64,
                "dtype": "bf16",
                "concurrency": 1,
                "prompt_tokens": 128,
                "output_tokens": 32,
                "warmups": 5,
                "measured_iterations": 30,
                "sampling_id": "greedy",
                "seed": None,
            },
            "trace": {
                "capacity": 4096,
                "retained_records": 158,
                "dropped_records": 0,
            },
            "primary_metric": "aggregate.host.execute_ns",
            "aggregate": {
                "host": {
                    "plan_ns": measured(900 if candidate else 1000),
                    "execute_ns": measured(90_000 if candidate else 100_000),
                    "sampling_ns": measured(900 if candidate else 1000),
                    "commit_ns": measured(900 if candidate else 1000),
                },
                "cuda": {
                    "stream_span_ns": measured(80_000 if candidate else 100_000),
                    "idle_ns": {"validity": "unmeasured", "value": None},
                },
                "counters": {
                    "iterations": measured(128),
                    "kernel_launches": measured(480 if candidate else 520),
                    "copies": {
                        "h2d_calls": measured(7),
                        "h2d_bytes": measured(4096),
                        "d2h_calls": measured(1),
                        "d2h_bytes": measured(1024),
                    },
                    "allocations": {
                        "device_allocations": measured(0),
                        "device_frees": measured(0),
                        "pinned_allocations": measured(0),
                        "pinned_frees": measured(0),
                        "peak_device_bytes": measured(2_000_000_000),
                    },
                },
                "throughput_output_tokens_per_second": throughput,
            },
            "requests": requests,
        }

    def write(self) -> None:
        self.baseline_paths = self._write_role("baseline", self.baseline)
        self.candidate_paths = self._write_role("candidate", self.candidate)

    def _write_role(
        self, role: str, rows: list[dict[str, object]]
    ) -> list[Path]:
        paths = []
        for index, row in enumerate(rows, 1):
            path = self.root / f"{role}-{index}.json"
            path.write_text(
                json.dumps(row, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            paths.append(path)
        return paths


class NativeProfilePairTests(unittest.TestCase):
    def test_passing_pair_reports_bindings_paired_statistics_and_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )

            self.assertTrue(report["passed"], report)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                report["schema_version"],
                "rustinfer.native-profile-pair-report.v1",
            )
            self.assertEqual(report["run_counts"], {"required": 5, "baseline": 5, "candidate": 5})
            self.assertEqual(len(report["run_pairs"]), 5)
            self.assertEqual(
                report["bindings"]["baseline_runtime"]["runtime_flag"]["value"],
                "separate",
            )
            self.assertEqual(
                report["bindings"]["candidate_runtime"]["runtime_flag"]["value"],
                "fused",
            )
            self.assertLess(report["metrics"]["ttft_ms"]["p95_ratio"], 1.0)
            self.assertGreater(
                report["metrics"]["throughput_output_tokens_per_second"]["median_ratio"],
                1.0,
            )
            self.assertEqual(
                {check["name"] for check in report["checks"]},
                {
                    "zero_failures",
                    "zero_dropped_trace_records",
                    "ttft_p95_regression",
                    "tpot_p95_regression",
                    "throughput_regression",
                    "primary_effective_improvement",
                },
            )

    def test_cli_accepts_grouped_paths_and_emits_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = checker.main(
                    [
                        "--baseline",
                        *(str(path) for path in fixture.baseline_paths),
                        "--candidate",
                        *(str(path) for path in fixture.candidate_paths),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(stdout.getvalue())["passed"])

    def test_requires_five_independent_runs_and_unique_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            report = checker.evaluate(
                fixture.baseline_paths[:4], fixture.candidate_paths
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("exactly 5", report["errors"][0])

            fixture.candidate[0]["run_id"] = fixture.baseline[0]["run_id"]
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertEqual(report["status"], "incomparable")
            self.assertIn("run IDs", report["errors"][0])

    def test_exact_provenance_workload_and_output_identity_are_required(self) -> None:
        mutations = {
            "source": lambda row: row["source"].__setitem__("executable_sha256", "0" * 64),
            "environment": lambda row: row["environment"]["software"].__setitem__("cuda_runtime_version", "different"),
            "workload": lambda row: row["workload"].__setitem__("warmups", 6),
            "u32-LE hashes": lambda row: row["requests"][0].__setitem__("generated_u32le_sha256", "1" * 64),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected), tempfile.TemporaryDirectory() as directory:
                fixture = ProfilePairFixture(Path(directory))
                mutate(fixture.candidate[0])
                fixture.write()
                report = checker.evaluate(
                    fixture.baseline_paths, fixture.candidate_paths
                )
                self.assertEqual(report["status"], "incomparable", report)
                self.assertIn(expected, report["errors"][0])

    def test_runtime_flag_is_bound_to_supported_baseline_candidate_pair(self) -> None:
        mutations = {
            "baseline": lambda fixture: [
                row["source"]["runtime_flag"].__setitem__("value", "fused")
                for row in fixture.baseline
            ],
            "candidate": lambda fixture: [
                row["source"]["runtime_flag"].__setitem__("value", "separate")
                for row in fixture.candidate
            ],
            "name": lambda fixture: [
                row["source"]["runtime_flag"].__setitem__("name", "other_flag")
                for row in fixture.baseline + fixture.candidate
            ],
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected), tempfile.TemporaryDirectory() as directory:
                fixture = ProfilePairFixture(Path(directory))
                mutate(fixture)
                fixture.write()
                report = checker.evaluate(
                    fixture.baseline_paths, fixture.candidate_paths
                )
                self.assertEqual(report["status"], "incomparable")
                self.assertIn("runtime_flag", report["errors"][0])

    def test_iteration_completion_pair_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            for row in fixture.baseline:
                row["source"]["implementation_id"] = "native-per-operation"
                row["source"]["runtime_flag"] = {
                    "name": "execution_completion",
                    "value": "per-operation",
                }
                row["source"]["correctness_gate_id"] = (
                    "pr15-iteration-command-batch-exact-v1"
                )
            for row in fixture.candidate:
                row["source"]["implementation_id"] = "native-iteration-batch"
                row["source"]["runtime_flag"] = {
                    "name": "execution_completion",
                    "value": "iteration-batch",
                }
                row["source"]["correctness_gate_id"] = (
                    "pr15-iteration-command-batch-exact-v1"
                )
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertTrue(report["passed"], report)
            self.assertEqual(
                report["bindings"]["candidate_runtime"]["runtime_flag"],
                {"name": "execution_completion", "value": "iteration-batch"},
            )

    def test_runtime_pair_requires_its_exact_correctness_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            for row in fixture.baseline + fixture.candidate:
                row["source"]["correctness_gate_id"] = "wrong-exact-gate"
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertEqual(report["status"], "incomparable")
            self.assertIn("correctness_gate_id", report["errors"][0])

    def test_success_request_must_generate_the_exact_fixed_output_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            fixture.candidate[0]["requests"][0]["generated_token_count"] = 1
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("must equal requested count", report["errors"][0])

    def test_minimum_samples_latency_shape_and_trace_retention_fail_closed(self) -> None:
        mutations = {
            "warmups": lambda row: row["workload"].__setitem__("warmups", 4),
            "measured_iterations": lambda row: row["workload"].__setitem__(
                "measured_iterations", 29
            ),
            "inter-token interval": lambda row: row["requests"][0].__setitem__(
                "e2e_ms", row["requests"][0]["ttft_ms"]
            ),
            "request_count + iteration_count": lambda row: row["trace"].__setitem__(
                "retained_records", 0
            ),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected), tempfile.TemporaryDirectory() as directory:
                fixture = ProfilePairFixture(Path(directory))
                mutate(fixture.candidate[0])
                fixture.write()
                report = checker.evaluate(
                    fixture.baseline_paths, fixture.candidate_paths
                )
                self.assertEqual(report["status"], "error", report)
                self.assertIn(expected, report["errors"][0])

    def test_extreme_numbers_return_error_instead_of_overflowing(self) -> None:
        mutations = {
            "finite IEEE-754": lambda fixture: fixture.candidate[0]["requests"][0].__setitem__(
                "ttft_ms", 10**400
            ),
            "latency composition must remain finite": lambda fixture: (
                fixture.candidate[0]["requests"][0].__setitem__("tpot_ms", 1.0e308),
                fixture.candidate[0]["requests"][0].__setitem__("e2e_ms", 1.0e308),
            ),
            "ratio must be finite": lambda fixture: [
                (
                    baseline["aggregate"].__setitem__(
                        "throughput_output_tokens_per_second", 1.0e-308
                    ),
                    candidate["aggregate"].__setitem__(
                        "throughput_output_tokens_per_second", 1.0e308
                    ),
                )
                for baseline, candidate in zip(
                    fixture.baseline, fixture.candidate, strict=True
                )
            ],
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected), tempfile.TemporaryDirectory() as directory:
                fixture = ProfilePairFixture(Path(directory))
                mutate(fixture)
                fixture.write()
                report = checker.evaluate(
                    fixture.baseline_paths, fixture.candidate_paths
                )
                self.assertEqual(report["status"], "error", report)
                self.assertIn(expected, report["errors"][0])

    def test_unknown_or_raw_request_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            fixture.candidate[0]["requests"][0]["generated_token_ids"] = [1, 2, 3]
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("generated_token_ids", report["errors"][0])

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            path = fixture.candidate_paths[0]
            raw = path.read_text(encoding="utf-8")
            raw = raw.replace(
                '  "schema_version": "rustinfer.native-profile-run.v1",',
                '  "schema_version": "rustinfer.native-profile-run.v1",\n  "schema_version": "rustinfer.native-profile-run.v1",',
                1,
            )
            path.write_text(raw, encoding="utf-8")
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("duplicate JSON object key", report["errors"][0])

    def test_zero_failure_and_bounded_trace_drop_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            fixture.candidate[0]["status"] = "failure"
            fixture.candidate[0]["failure_count"] = 1
            fixture.candidate[1]["trace"]["dropped_records"] = 1
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertEqual(report["status"], "failed")
            checks = {check["name"]: check for check in report["checks"]}
            self.assertFalse(checks["zero_failures"]["passed"])
            self.assertFalse(checks["zero_dropped_trace_records"]["passed"])

    def test_each_regression_and_effective_improvement_threshold_fails_closed(self) -> None:
        def set_request_metric(
            fixture: ProfilePairFixture, field: str, baseline_ratio: float
        ) -> None:
            for baseline, candidate in zip(
                fixture.baseline, fixture.candidate, strict=True
            ):
                for left, right in zip(
                    baseline["requests"], candidate["requests"], strict=True
                ):
                    right[field] = left[field] * baseline_ratio
                    right["e2e_ms"] = max(
                        right["e2e_ms"],
                        right["ttft_ms"]
                        + right["tpot_ms"] * (right["generated_token_count"] - 1),
                    )

        cases = {
            "ttft_p95_regression": lambda fixture: set_request_metric(fixture, "ttft_ms", 1.051),
            "tpot_p95_regression": lambda fixture: set_request_metric(fixture, "tpot_ms", 1.051),
            "throughput_regression": lambda fixture: [
                candidate["aggregate"].__setitem__(
                    "throughput_output_tokens_per_second",
                    baseline["aggregate"]["throughput_output_tokens_per_second"] * 0.949,
                )
                for baseline, candidate in zip(fixture.baseline, fixture.candidate, strict=True)
            ],
            "primary_effective_improvement": lambda fixture: [
                candidate["aggregate"]["host"]["execute_ns"].__setitem__(
                    "value",
                    baseline["aggregate"]["host"]["execute_ns"]["value"] * 0.951,
                )
                for baseline, candidate in zip(
                    fixture.baseline, fixture.candidate, strict=True
                )
            ],
        }
        for failed_check, mutate in cases.items():
            with self.subTest(failed_check), tempfile.TemporaryDirectory() as directory:
                fixture = ProfilePairFixture(Path(directory))
                mutate(fixture)
                fixture.write()
                report = checker.evaluate(
                    fixture.baseline_paths, fixture.candidate_paths
                )
                self.assertEqual(report["status"], "failed", report)
                checks = {check["name"]: check for check in report["checks"]}
                self.assertFalse(checks[failed_check]["passed"], report)

    def test_exact_regression_and_improvement_boundaries_are_inclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            for baseline, candidate in zip(
                fixture.baseline, fixture.candidate, strict=True
            ):
                baseline_throughput = baseline["aggregate"][
                    "throughput_output_tokens_per_second"
                ]
                candidate["aggregate"][
                    "throughput_output_tokens_per_second"
                ] = baseline_throughput * 1.05
                baseline_execute = baseline["aggregate"]["host"]["execute_ns"][
                    "value"
                ]
                candidate["aggregate"]["host"]["execute_ns"]["value"] = (
                    baseline_execute * 0.95
                )
                for left, right in zip(
                    baseline["requests"], candidate["requests"], strict=True
                ):
                    right["ttft_ms"] = left["ttft_ms"] * 1.05
                    right["tpot_ms"] = left["tpot_ms"] * 1.05
                    right["e2e_ms"] = max(
                        right["e2e_ms"],
                        right["ttft_ms"]
                        + right["tpot_ms"] * (right["generated_token_count"] - 1),
                    )
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertTrue(report["passed"], report)
            checks = {check["name"]: check for check in report["checks"]}
            self.assertTrue(checks["ttft_p95_regression"]["passed"])
            self.assertTrue(checks["tpot_p95_regression"]["passed"])
            self.assertTrue(checks["primary_effective_improvement"]["passed"])

    def test_declared_aggregate_primary_must_be_measured_in_all_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            fixture.candidate[0]["aggregate"]["host"]["execute_ns"] = {
                "validity": "unmeasured",
                "value": None,
            }
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertEqual(report["status"], "error")
            self.assertIn("declared primary metric is unmeasured", report["errors"][0])

    def test_noncanonical_sampling_seed_and_primary_metric_fail_closed(self) -> None:
        cases = {
            "sampling": lambda row: row["workload"].__setitem__("sampling_id", "top-p"),
            "seed": lambda row: row["workload"].__setitem__("seed", 7),
            "primary": lambda row: row.__setitem__(
                "primary_metric", "aggregate.cuda.stream_span_ns"
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                fixture = ProfilePairFixture(Path(directory))
                mutate(fixture.candidate[0])
                fixture.write()
                report = checker.evaluate(
                    fixture.baseline_paths, fixture.candidate_paths
                )
                self.assertEqual(report["status"], "error", report)

    def test_single_output_token_cannot_claim_a_tpot_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            fixture.candidate[0]["workload"]["output_tokens"] = 1
            fixture.write()
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertEqual(report["status"], "error", report)

    def test_nonprimary_unmeasured_metric_is_explicit_and_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ProfilePairFixture(Path(directory))
            report = checker.evaluate(
                fixture.baseline_paths, fixture.candidate_paths
            )
            self.assertTrue(report["passed"], report)
            self.assertTrue(
                all(
                    row["aggregate"]["cuda"]["idle_ns"]
                    == {"validity": "unmeasured", "value": None}
                    for row in fixture.baseline + fixture.candidate
                )
            )

    def test_schemas_are_versioned_and_close_root_and_request_objects(self) -> None:
        run_schema = json.loads(
            (REPOSITORY_ROOT / "benchmarks/schemas/native-profile-run.schema.json").read_text(
                encoding="utf-8"
            )
        )
        report_schema = json.loads(
            (
                REPOSITORY_ROOT
                / "benchmarks/schemas/native-profile-pair-report.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(run_schema["additionalProperties"])
        self.assertFalse(run_schema["$defs"]["request"]["additionalProperties"])
        self.assertEqual(
            run_schema["properties"]["schema_version"]["const"],
            "rustinfer.native-profile-run.v1",
        )
        self.assertFalse(report_schema["additionalProperties"])
        self.assertEqual(
            report_schema["properties"]["schema_version"]["const"],
            "rustinfer.native-profile-pair-report.v1",
        )
        self.assertEqual(
            report_schema["properties"]["primary_metric"]["oneOf"][0]["$ref"],
            "#/$defs/primaryMetric",
        )
        self.assertEqual(
            report_schema["$defs"]["primaryComparison"]["properties"]["name"]["$ref"],
            "#/$defs/primaryMetric",
        )
        self.assertEqual(
            report_schema["$defs"]["primaryMetric"],
            run_schema["$defs"]["primaryMetric"],
        )
        self.assertEqual(
            run_schema["$defs"]["primaryMetric"],
            {"const": "aggregate.host.execute_ns"},
        )
        self.assertEqual(
            run_schema["$defs"]["workload"]["properties"]["sampling_id"],
            {"const": "greedy"},
        )
        self.assertEqual(
            run_schema["$defs"]["workload"]["properties"]["seed"],
            {"type": "null"},
        )

        def assert_local_refs_resolve(root: dict[str, object], value: object) -> None:
            if isinstance(value, dict):
                reference = value.get("$ref")
                if reference is not None:
                    self.assertIsInstance(reference, str)
                    self.assertTrue(reference.startswith("#/"), reference)
                    resolved: object = root
                    for raw_segment in reference[2:].split("/"):
                        segment = raw_segment.replace("~1", "/").replace("~0", "~")
                        self.assertIsInstance(resolved, dict)
                        self.assertIn(segment, resolved)
                        resolved = resolved[segment]
                for child in value.values():
                    assert_local_refs_resolve(root, child)
            elif isinstance(value, list):
                for child in value:
                    assert_local_refs_resolve(root, child)

        assert_local_refs_resolve(run_schema, run_schema)
        assert_local_refs_resolve(report_schema, report_schema)


if __name__ == "__main__":
    unittest.main()
