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
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_release_performance.py"
REPOSITORY_ROOT = SCRIPT.parents[2]
BASELINE = REPOSITORY_ROOT / "benchmarks/release/performance-baseline-v1.json"
SPEC = importlib.util.spec_from_file_location("check_release_performance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)

PROFILE_FIXTURE_SCRIPT = Path(__file__).with_name("test_check_native_profile_pair.py")
PROFILE_SPEC = importlib.util.spec_from_file_location(
    "release_native_profile_fixture", PROFILE_FIXTURE_SCRIPT
)
assert PROFILE_SPEC is not None and PROFILE_SPEC.loader is not None
profile_fixture_module = importlib.util.module_from_spec(PROFILE_SPEC)
sys.modules[PROFILE_SPEC.name] = profile_fixture_module
PROFILE_SPEC.loader.exec_module(profile_fixture_module)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ReleaseFixture:
    def __init__(self, root: Path) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.root = root
        self.paths = {
            "source_archive": root / "source.tar",
            "profile_binary": root / "rustinfer-profile",
            "release_binary": root / "rustinfer",
            "weights": root / "model.safetensors",
            "tokenizer": root / "tokenizer.json",
            "correctness_report": root / "correctness.json",
        }
        self.digests = {name: digest(name) for name in self.paths}
        self.profile_image_digest = digest("profile image")
        self.release_image_digest = digest("release image")

        raw_root = root / "raw"
        raw_root.mkdir()
        self.profile_fixture = profile_fixture_module.ProfilePairFixture(raw_root)
        for run_index, run in enumerate(self.profile_fixture.candidate):
            run["source"] = {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "executable_sha256": self.digests["profile_binary"],
                "implementation_id": "native-iteration-command-batch",
                "runtime_flag": {
                    "name": "execution_completion",
                    "value": "iteration-batch",
                },
                "semantic_class": "E0",
                "correctness_gate_id": checker.CORRECTNESS_GATE_ID,
                "correctness_report_sha256": self.digests["correctness_report"],
            }
            run["environment"]["gpu"]["uuid"] = baseline["environment"][
                "gpu_uuid"
            ]
            run["environment"]["gpu"]["compute_capability"] = baseline[
                "environment"
            ]["compute_capability"]
            run["environment"]["host"]["environment_id"] = baseline[
                "environment"
            ]["environment_id"]
            software = run["environment"]["software"]
            software["nvidia_driver_version"] = baseline["environment"][
                "driver_version"
            ]
            software["cuda_runtime_version"] = baseline["environment"][
                "cuda_runtime_version"
            ]
            software["cuda_toolkit_version"] = baseline["environment"][
                "cuda_toolkit_version"
            ]
            software["container_image_sha256"] = self.profile_image_digest

            workload = run["workload"]
            workload.update(
                {
                    "workload_id": baseline["workload"]["workload_id"],
                    "model_id": baseline["model"]["model_id"],
                    "model_revision": baseline["model"]["model_revision"],
                    "weights_sha256": baseline["model"]["weights_sha256"],
                    "tokenizer_sha256": baseline["model"]["tokenizer_sha256"],
                    "dtype": baseline["model"]["dtype"],
                    "concurrency": baseline["workload"]["concurrency"],
                    "prompt_tokens": baseline["workload"]["prompt_tokens"],
                    "output_tokens": baseline["workload"]["output_tokens"],
                    "warmups": baseline["workload"]["warmups_per_run"],
                    "measured_iterations": baseline["workload"][
                        "measured_iterations_per_run"
                    ],
                    "sampling_id": baseline["workload"]["sampling"],
                    "seed": None,
                }
            )
            run_factor = 1.0 + (run_index - 2) * 0.002
            for request_index, request in enumerate(run["requests"]):
                request_factor = 1.0 + request_index * 0.0005
                request["ttft_ms"] = 5.4 * run_factor * request_factor
                request["tpot_ms"] = 7.0 * run_factor * request_factor
                request["e2e_ms"] = 225.0 * run_factor * request_factor
            run["aggregate"]["throughput_output_tokens_per_second"] = (
                140.0 / run_factor
            )

        self.candidate = {
            "schema_version": "rustinfer.release-performance-candidate.v1",
            "baseline_sha256": checker.BASELINE_SHA256,
            "candidate_id": "pr16-release-candidate-fixture",
            "recorded_at_utc": "2026-08-26T12:34:56Z",
            "status": "success",
            "source": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "source_archive_sha256": self.digests["source_archive"],
                "profile_binary_sha256": self.digests["profile_binary"],
                "release_binary_sha256": self.digests["release_binary"],
                "profile_image_sha256": self.profile_image_digest,
                "release_image_sha256": self.release_image_digest,
                "semantic_class": "E0",
                "correctness_gate_id": checker.CORRECTNESS_GATE_ID,
                "correctness_report_sha256": self.digests["correctness_report"],
            },
            "model": copy.deepcopy(baseline["model"]),
            "environment": copy.deepcopy(baseline["environment"]),
            "workload": copy.deepcopy(baseline["workload"]),
            "run_summary": {},
            "metrics": {},
            "raw_runs": [],
        }
        self.digests["weights"] = self.candidate["model"]["weights_sha256"]
        self.digests["tokenizer"] = self.candidate["model"]["tokenizer_sha256"]
        self.candidate_path = root / "candidate.json"
        for path in self.paths.values():
            path.write_bytes(b"fixture artifact")
        self.write_correctness_report()
        self.refresh()

    def optimization_correctness_report(self) -> dict[str, object]:
        source = self.candidate["source"]
        model = self.candidate["model"]
        environment = self.candidate["environment"]
        return {
            "schema_version": 1,
            "gate_id": checker.CORRECTNESS_GATE_ID,
            "recorded_at_utc": "2026-08-26T12:30:00Z",
            "status": "passed",
            "semantic_class": "E0",
            "source": {
                "git_commit": source["git_commit"],
                "git_dirty": False,
                "archive_sha256": source["source_archive_sha256"],
            },
            "model": {
                "model_id": model["model_id"],
                "revision": model["model_revision"],
                "dtype": model["dtype"],
                "manifest_sha256": digest("model manifest"),
                "weights_sha256": model["weights_sha256"],
                "tokenizer_sha256": model["tokenizer_sha256"],
            },
            "gpu": {
                "model": "NVIDIA GeForce RTX 4090",
                "uuid": environment["gpu_uuid"],
                "pci_bus_id": "00000000:01:00.0",
                "compute_capability": environment["compute_capability"],
                "vram_mib": 24564,
                "driver_version": environment["driver_version"],
            },
            "build": {
                "rustc": "1.85.0",
                "cuda_toolkit": environment["cuda_toolkit_version"],
                "cuda_architecture": environment["cuda_architecture"],
                "container_image_sha256": source["profile_image_sha256"],
                "network": "none",
                "cargo_locked": True,
                "cargo_offline": True,
            },
            "implementations": {
                "baseline": "per-operation",
                "candidate": "iteration-batch",
                "residual_rmsnorm": "separate",
                "rollback": "--execution-completion per-operation",
            },
            "tests": [
                {
                    "id": "cuda-compile-only",
                    "result": "passed",
                    "log_sha256": digest("cuda compile log"),
                },
                {
                    "id": "workspace-all-features-all-targets",
                    "result": "passed",
                    "log_sha256": digest("workspace test log"),
                },
                {
                    "id": "command-batch-lifecycle",
                    "result": "passed",
                    "log_sha256": digest("lifecycle log"),
                    "one_shot_finish": True,
                    "drop_restores_stream": True,
                },
                {
                    "id": "command-batch-resource-ledger",
                    "result": "passed",
                    "log_sha256": digest("resource ledger log"),
                    "queued_chain_raw_byte_mismatches": 0,
                    "cuda_live_allocation_delta": 0,
                    "owner_close_live_allocation_count": 0,
                    "validation_fail_closed": True,
                    "stream_reuse_after_finish": True,
                },
                {
                    "id": "smollm2-multi-step-greedy-exact",
                    "result": "passed",
                    "log_sha256": digest("smollm2 exact log"),
                    "decode_steps": 16,
                    "committed_iterations": 16,
                    "generated_token_ids": list(
                        checker.OPTIMIZATION_GOLDEN_TOKEN_IDS
                    ),
                    "raw_logit_mismatches": 0,
                    "token_id_mismatches": 0,
                    "cuda_live_allocation_delta": 0,
                    "owner_close_live_allocation_count": 0,
                },
            ],
        }

    def write_correctness_report(self) -> None:
        self.paths["correctness_report"].write_text(
            json.dumps(
                self.optimization_correctness_report(), sort_keys=True, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    def refresh(self) -> None:
        self.profile_fixture.write()
        runs = self.profile_fixture.candidate
        self.raw_paths = self.profile_fixture.candidate_paths
        self.candidate["raw_runs"] = [
            {
                "pair_index": run["pair_index"],
                "run_id": run["run_id"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, run in zip(self.raw_paths, runs, strict=True)
        ]
        workload = runs[0]["workload"]
        self.candidate["run_summary"] = {
            "independent_runs": len(runs),
            "warmups_per_run": workload["warmups"],
            "measured_iterations_per_run": workload["measured_iterations"],
            "failure_count": sum(run["failure_count"] for run in runs),
            "dropped_trace_records": sum(
                run["trace"]["dropped_records"] for run in runs
            ),
        }
        requests = [request for run in runs for request in run["requests"]]
        self.candidate["metrics"] = {
            "ttft_p95_ms": checker.native_profile.r7(
                [request["ttft_ms"] for request in requests], 0.95
            ),
            "tpot_p95_ms": checker.native_profile.r7(
                [request["tpot_ms"] for request in requests], 0.95
            ),
            "e2e_median_ms": checker.native_profile.r7(
                [request["e2e_ms"] for request in requests], 0.50
            ),
            "throughput_median_output_tokens_per_second": checker.native_profile.r7(
                [
                    run["aggregate"]["throughput_output_tokens_per_second"]
                    for run in runs
                ],
                0.50,
            ),
        }
        self.write()

    def write(self) -> None:
        self.candidate_path.write_text(
            json.dumps(self.candidate, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def digest_for(self, path: Path, _label: str) -> str:
        for name, expected_path in self.paths.items():
            if path == expected_path:
                return self.digests[name]
        if path in self.raw_paths:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        raise AssertionError(f"unexpected artifact path: {path}")

    def evaluate(self, *, baseline: Path = BASELINE) -> dict[str, object]:
        with mock.patch.object(checker, "_digest_file", side_effect=self.digest_for):
            return checker.evaluate(
                baseline,
                self.candidate_path,
                source_archive=self.paths["source_archive"],
                profile_binary=self.paths["profile_binary"],
                release_binary=self.paths["release_binary"],
                weights=self.paths["weights"],
                tokenizer=self.paths["tokenizer"],
                correctness_report=self.paths["correctness_report"],
                profile_image_id=f"sha256:{self.profile_image_digest}",
                release_image_id=f"sha256:{self.release_image_digest}",
                run_paths=self.raw_paths,
            )


class ReleasePerformanceTests(unittest.TestCase):
    def test_reviewed_baseline_digest_is_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(BASELINE.read_bytes()).hexdigest(), checker.BASELINE_SHA256
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            changed = Path(directory) / "baseline.json"
            changed.write_bytes(BASELINE.read_bytes() + b"\n")
            report = fixture.evaluate(baseline=changed)
            self.assertEqual(report["status"], "error")
            self.assertIn("not the reviewed v1 baseline", report["errors"][0])

    def test_passing_candidate_binds_producer_release_and_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            report = fixture.evaluate()
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["checks"]), 4)
            source = report["candidate"]["source"]
            self.assertEqual(
                source["profile_binary_sha256"], fixture.digests["profile_binary"]
            )
            self.assertEqual(
                source["release_binary_sha256"], fixture.digests["release_binary"]
            )
            self.assertEqual(len(report["candidate"]["raw_runs"]), 5)

    def test_threshold_regression_is_derived_from_raw_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            for run in fixture.profile_fixture.candidate:
                for request in run["requests"]:
                    request["ttft_ms"] *= 1.2
            fixture.refresh()
            report = fixture.evaluate()
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "failed")
            failed = [check["name"] for check in report["checks"] if not check["passed"]]
            self.assertEqual(failed, ["ttft_p95_regression"])

    def test_self_asserted_metric_or_raw_digest_tamper_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.candidate["metrics"]["ttft_p95_ms"] *= 0.5
            fixture.write()
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("raw-derived R7 metrics", report["errors"][0])

        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.raw_paths[0].write_bytes(fixture.raw_paths[0].read_bytes() + b" ")
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("raw run binding", report["errors"][0])

    def test_model_or_environment_drift_is_incomparable(self) -> None:
        for field, value in [("model", "other/model"), ("environment", "8.0")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = ReleaseFixture(Path(directory))
                if field == "model":
                    fixture.candidate["model"]["model_id"] = value
                else:
                    fixture.candidate["environment"]["compute_capability"] = value
                fixture.write()
                report = fixture.evaluate()
                self.assertEqual(report["status"], "incomparable")
                self.assertFalse(report["passed"])

    def test_artifact_mismatch_and_unknown_fields_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.digests["release_binary"] = digest("different release binary")
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("does not match artifact", report["errors"][0])

        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.candidate["unexpected"] = True
            fixture.write()
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("unknown fields", report["errors"][0])

    def test_correctness_report_is_semantically_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            correctness = fixture.optimization_correctness_report()
            correctness["gate_id"] = "self-asserted-gate"
            fixture.paths["correctness_report"].write_text(
                json.dumps(correctness), encoding="utf-8"
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("correctness_report.gate_id", report["errors"][0])

        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            correctness = fixture.optimization_correctness_report()
            correctness["tests"][-1]["generated_token_ids"][-1] += 1
            fixture.paths["correctness_report"].write_text(
                json.dumps(correctness), encoding="utf-8"
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("generated_token_ids", report["errors"][0])

    def test_cli_refuses_to_overwrite_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            report_path = fixture.root / "report.json"
            report_path.write_text("occupied", encoding="utf-8")
            argv = [
                "--baseline", str(BASELINE),
                "--candidate", str(fixture.candidate_path),
                "--source-archive", str(fixture.paths["source_archive"]),
                "--profile-binary", str(fixture.paths["profile_binary"]),
                "--release-binary", str(fixture.paths["release_binary"]),
                "--weights", str(fixture.paths["weights"]),
                "--tokenizer", str(fixture.paths["tokenizer"]),
                "--correctness-report", str(fixture.paths["correctness_report"]),
                "--profile-image-id", f"sha256:{fixture.profile_image_digest}",
                "--release-image-id", f"sha256:{fixture.release_image_digest}",
                "--run", *(str(path) for path in fixture.raw_paths),
                "--report", str(report_path),
            ]
            stderr = io.StringIO()
            with mock.patch.object(checker, "_digest_file", side_effect=fixture.digest_for):
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    exit_code = checker.main(argv)
            self.assertEqual(exit_code, 2)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "occupied")
            self.assertIn("refusing to overwrite", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
