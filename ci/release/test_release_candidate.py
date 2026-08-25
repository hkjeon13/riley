#!/usr/bin/env python3
"""CPU-only tests for the final release-candidate evidence gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

from build_release_bundle import build_bundle  # noqa: E402
from check_release_candidate import (  # noqa: E402
    ATTESTATION_VERSION,
    CUDA_FAULT_CHECKS,
    MANIFEST_VERSION,
    NATIVE_REPLAY_VERSION,
    OPTIMIZATION_LOGS,
    PYTHON_FREE_CHECKS,
    SOAK_COMMON_SCENARIO_CHECKS,
    SOAK_CONTRACT_ID,
    SOAK_GLOBAL_CHECKS,
    SOAK_GOLDEN_SCENARIOS,
    SOAK_SCENARIOS,
    SOAK_TEMPLATE_CANONICAL_SHA256,
    cuda_fault_evidence,
    evaluate,
    reliability_soak,
    release_performance,
)
from test_cuda_fault_evidence import (  # noqa: E402
    BUILD_IMAGE_ID as CUDA_BUILD_IMAGE_ID,
    Fixture as CudaEvidenceFixture,
)
from test_release import EPOCH, fixture_elf  # noqa: E402


REVISION = "1a2b3c4d5e6f78901234567890abcdef12345678"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_BASELINE = (
    REPOSITORY_ROOT / "benchmarks/release/performance-baseline-v1.json"
)
PROFILE_FIXTURE_SCRIPT = (
    REPOSITORY_ROOT
    / "benchmarks/scripts/tests/test_check_native_profile_pair.py"
)
PROFILE_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "release_candidate_native_profile_fixture", PROFILE_FIXTURE_SCRIPT
)
assert PROFILE_FIXTURE_SPEC is not None and PROFILE_FIXTURE_SPEC.loader is not None
profile_fixture_module = importlib.util.module_from_spec(PROFILE_FIXTURE_SPEC)
sys.modules[PROFILE_FIXTURE_SPEC.name] = profile_fixture_module
PROFILE_FIXTURE_SPEC.loader.exec_module(profile_fixture_module)
E2E_FIXTURE_SCRIPT = (
    REPOSITORY_ROOT
    / "benchmarks/scripts/tests/test_check_python_free_release_e2e.py"
)
E2E_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "release_candidate_python_free_e2e_fixture", E2E_FIXTURE_SCRIPT
)
assert E2E_FIXTURE_SPEC is not None and E2E_FIXTURE_SPEC.loader is not None
e2e_fixture_module = importlib.util.module_from_spec(E2E_FIXTURE_SPEC)
sys.modules[E2E_FIXTURE_SPEC.name] = e2e_fixture_module
E2E_FIXTURE_SPEC.loader.exec_module(e2e_fixture_module)


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


class CandidateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        repository = root / "repository"
        repository.mkdir()
        (repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\n'
            'license = "LicenseRef-Test-Fixture"\n',
            encoding="utf-8",
        )
        (repository / "LICENSE").write_text(
            "Owner-approved fixture license for release contract tests.\n"
            "Permission is granted only inside this temporary unit-test fixture.\n",
            encoding="utf-8",
        )
        self.paths = {
            "source": root / "source.tar",
            "binary": root / "rustinfer",
            "bundle": root / "rustinfer.tar.gz",
            "python_raw": root / "python-free-evidence.tar",
            "cuda_raw": root / "cuda-fault-evidence.tar",
            "python_report": root / "python-free-report.json",
            "cuda_report": root / "cuda-fault-report.json",
            "native_correctness": root / "native-correctness-report.json",
            "native_replay": root / "native-correctness-replay.tar",
            "native_replay_validation": root / "native-replay-validation.json",
            "optimization_correctness": root / "optimization-correctness-report.json",
            "optimization_raw": root / "optimization-correctness-evidence.tar",
            "performance": root / "performance-report.json",
            "performance_raw": root / "performance-evidence.tar",
            "soak": root / "soak-report.json",
            "soak_raw": root / "soak-evidence.tar",
        }
        self._write_tar(
            self.paths["source"],
            {"README.md": b"exact source archive fixture"},
            pax_headers={"comment": REVISION},
        )
        self.paths["binary"].write_bytes(fixture_elf())
        self.paths["binary"].chmod(0o755)
        build_bundle(
            binary_path=self.paths["binary"],
            output=self.paths["bundle"],
            repository_root=repository,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
        )
        self.image_sha = digest(b"release image")
        self.cuda_build_image_id = CUDA_BUILD_IMAGE_ID
        cuda_template_root = root / "cuda-evidence-template"
        cuda_template_root.mkdir()
        cuda_template = CudaEvidenceFixture(cuda_template_root)
        environment_path = cuda_template.evidence / "environment.txt"
        environment = environment_path.read_text(encoding="utf-8")
        environment = environment.replace(
            f"source_archive_sha256={digest(cuda_template.source_archive.read_bytes())}",
            f"source_archive_sha256={digest(self.paths['source'].read_bytes())}",
        )
        environment_path.write_text(environment, encoding="utf-8")
        (cuda_template.evidence / "release-binary.sha256").write_text(
            f"{digest(self.paths['binary'].read_bytes())}  target/release/rustinfer\n",
            encoding="ascii",
        )
        cuda_template.refresh_checksums()
        self.cuda_attestation = cuda_fault_evidence.produce(
            cuda_template.evidence,
            source_revision=REVISION,
            source_archive=self.paths["source"],
            build_image_id=self.cuda_build_image_id,
            release_binary=self.paths["binary"],
            release_bundle=self.paths["bundle"],
            release_image_id=f"sha256:{self.image_sha}",
            raw_evidence=self.paths["cuda_raw"],
            report=self.paths["cuda_report"],
        )
        self.paths["soak_raw"].write_bytes(b"soak raw evidence fixture")
        self._write_tar(
            self.paths["native_replay"],
            {"replay-summary.json": b'{"status":"passed"}\n'},
        )
        self.optimization_logs = {
            test_id: f"raw log for {test_id}\n".encode()
            for test_id in OPTIMIZATION_LOGS
        }
        self._write_tar(
            self.paths["optimization_raw"],
            {
                OPTIMIZATION_LOGS[test_id]: contents
                for test_id, contents in self.optimization_logs.items()
            },
        )
        self.documents: dict[str, dict[str, object]] = {}
        self._build_documents()
        self.write_reports()
        self.manifest_path = root / "release-candidate.json"
        self.refresh_manifest()

    @staticmethod
    def _write_tar(
        path: Path,
        files: dict[str, bytes],
        *,
        pax_headers: dict[str, str] | None = None,
    ) -> None:
        with tarfile.open(
            path,
            "w",
            format=tarfile.PAX_FORMAT,
            pax_headers=pax_headers,
        ) as archive:
            for name, contents in sorted(files.items()):
                member = tarfile.TarInfo(name)
                member.size = len(contents)
                member.mtime = EPOCH
                member.uid = 0
                member.gid = 0
                member.uname = "root"
                member.gname = "root"
                archive.addfile(member, io.BytesIO(contents))

    def _binding(self) -> dict[str, object]:
        return {
            "git_revision": REVISION,
            "git_dirty": False,
            "source_archive_sha256": digest(self.paths["source"].read_bytes()),
            "release_binary_sha256": digest(self.paths["binary"].read_bytes()),
            "release_bundle_sha256": digest(self.paths["bundle"].read_bytes()),
            "release_image_sha256": self.image_sha,
        }

    def _attestation(self, gate: str, raw: str, checks: set[str]) -> dict[str, object]:
        return {
            "schema_version": ATTESTATION_VERSION,
            "gate": gate,
            "status": "passed",
            "source": self._binding(),
            "raw_evidence_sha256": digest(self.paths[raw].read_bytes()),
            "checks": [{"id": check, "passed": True} for check in sorted(checks)],
        }

    def _build_python_e2e_document(
        self, native_correctness_sha256: str
    ) -> dict[str, object]:
        template_root = self.root / "python-e2e-template"
        template_root.mkdir()
        template = e2e_fixture_module.E2EFixture(template_root)
        raw = copy.deepcopy(template.raw)
        binding = self._binding()
        raw["source"] = {
            "git_revision": REVISION,
            "git_dirty": False,
            "source_archive_sha256": binding["source_archive_sha256"],
        }
        raw["release"] = {
            "binary_sha256": binding["release_binary_sha256"],
            "bundle_sha256": binding["release_bundle_sha256"],
            "image_sha256": self.image_sha,
        }
        raw["runtime"]["image_id"] = f"sha256:{self.image_sha}"
        raw["runtime"]["image_binary_sha256"] = binding[
            "release_binary_sha256"
        ]

        model_files = {
            "config.json": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
            "model.safetensors": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
            **e2e_fixture_module.checker.TOKENIZER_FILES_SHA256,
        }
        model_manifest = b"".join(
            f"{sha256}  {name}\n".encode("ascii")
            for name, sha256 in sorted(model_files.items())
        )
        golden = {
            "schema_version": e2e_fixture_module.checker.GOLDEN_SCHEMA,
            "correctness_gate_id": "smollm2-fp32-bf16-native-e0-v2",
            "correctness_report_sha256": native_correctness_sha256,
            "source_revision": REVISION,
            "model_id": "HuggingFaceTB/SmolLM2-135M",
            "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "config_sha256": model_files["config.json"],
            "weights_sha256": model_files["model.safetensors"],
            "tokenizer_aggregate_sha256": "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db",
            "tokenizer_json_sha256": model_files["tokenizer.json"],
            "prompt": "A bounded release probe",
            "max_tokens": 8,
            "expected_greedy_text_sha256": template.expected_text_sha256,
        }
        golden_bytes = (
            json.dumps(golden, sort_keys=True, indent=2) + "\n"
        ).encode()
        raw["model"] = {
            "model_id": golden["model_id"],
            "model_revision": golden["model_revision"],
            "model_tree_sha256": digest(model_manifest),
            "config_sha256": golden["config_sha256"],
            "weights_sha256": golden["weights_sha256"],
            "tokenizer_aggregate_sha256": golden["tokenizer_aggregate_sha256"],
            "tokenizer_json_sha256": golden["tokenizer_json_sha256"],
            "correctness_gate_id": golden["correctness_gate_id"],
            "correctness_report_sha256": native_correctness_sha256,
            "correctness_golden_sha256": digest(golden_bytes),
        }
        self.e2e_model_tree_sha256 = raw["model"]["model_tree_sha256"]
        raw["observations"]["models"]["model_ids"] = [golden["model_id"]]
        raw_bytes = (json.dumps(raw, sort_keys=True, indent=2) + "\n").encode()
        payloads = {
            "correctness-golden.json": golden_bytes,
            "model-SHA256SUMS": model_manifest,
            "raw-evidence.json": raw_bytes,
            "repeat-shutdown-metrics.json": template.repeat_shutdown_metrics.read_bytes(),
            "shutdown-metrics.json": template.shutdown_metrics.read_bytes(),
        }
        payloads["SHA256SUMS"] = b"".join(
            f"{digest(payloads[name])}  {name}\n".encode("ascii")
            for name in e2e_fixture_module.checker.RAW_ARCHIVE_PAYLOADS
        )
        e2e_fixture_module.write_raw_tar(self.paths["python_raw"], payloads)
        archive = e2e_fixture_module.checker.load_raw_evidence_archive(
            self.paths["python_raw"]
        )
        report, diagnostic = e2e_fixture_module.checker.validate_bound_raw_archive(
            archive,
            source_revision=REVISION,
            source_archive_sha256=binding["source_archive_sha256"],
            release_binary_sha256=binding["release_binary_sha256"],
            release_bundle_sha256=binding["release_bundle_sha256"],
            image_id=f"sha256:{self.image_sha}",
            correctness_report=self.documents["native_correctness"],
            correctness_report_sha256=native_correctness_sha256,
        )
        if diagnostic is not None:
            raise AssertionError(diagnostic)
        return report

    def _build_performance_document(self, optimization_sha: str) -> dict[str, object]:
        baseline = json.loads(PERFORMANCE_BASELINE.read_text(encoding="utf-8"))
        raw_root = self.root / "performance-runs"
        raw_root.mkdir()
        fixture = profile_fixture_module.ProfilePairFixture(raw_root)
        profile_binary_sha = digest(b"profile binary")
        profile_image_sha = digest(b"profile image")
        for run_index, run in enumerate(fixture.candidate):
            run["source"] = {
                "git_commit": REVISION,
                "git_dirty": False,
                "executable_sha256": profile_binary_sha,
                "implementation_id": "native-iteration-command-batch",
                "runtime_flag": {
                    "name": "execution_completion",
                    "value": "iteration-batch",
                },
                "semantic_class": "E0",
                "correctness_gate_id": "pr15-iteration-command-batch-exact-v1",
                "correctness_report_sha256": optimization_sha,
            }
            run["environment"]["gpu"]["uuid"] = baseline["environment"]["gpu_uuid"]
            run["environment"]["gpu"]["compute_capability"] = baseline["environment"][
                "compute_capability"
            ]
            run["environment"]["host"]["environment_id"] = baseline["environment"][
                "environment_id"
            ]
            software = run["environment"]["software"]
            software["nvidia_driver_version"] = baseline["environment"]["driver_version"]
            software["cuda_runtime_version"] = baseline["environment"][
                "cuda_runtime_version"
            ]
            software["cuda_toolkit_version"] = baseline["environment"][
                "cuda_toolkit_version"
            ]
            software["container_image_sha256"] = profile_image_sha
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
        fixture.write()
        self.performance_fixture = fixture

        candidate: dict[str, object] = {
            "candidate_id": "fixture",
            "recorded_at_utc": "2026-08-26T00:00:00Z",
            "source": {
                "git_commit": REVISION,
                "git_dirty": False,
                "source_archive_sha256": self._binding()["source_archive_sha256"],
                "profile_binary_sha256": profile_binary_sha,
                "release_binary_sha256": self._binding()["release_binary_sha256"],
                "profile_image_sha256": profile_image_sha,
                "release_image_sha256": self.image_sha,
                "semantic_class": "E0",
                "correctness_gate_id": "pr15-iteration-command-batch-exact-v1",
                "correctness_report_sha256": optimization_sha,
            },
            "model": copy.deepcopy(baseline["model"]),
            "environment": copy.deepcopy(baseline["environment"]),
            "workload": copy.deepcopy(baseline["workload"]),
            "metrics": {},
            "run_summary": {},
            "raw_runs": [
                {
                    "pair_index": run["pair_index"],
                    "run_id": run["run_id"],
                    "sha256": digest(path.read_bytes()),
                }
                for path, run in zip(
                    fixture.candidate_paths, fixture.candidate, strict=True
                )
            ],
        }
        workload = fixture.candidate[0]["workload"]
        candidate["run_summary"] = {
            "independent_runs": len(fixture.candidate),
            "warmups_per_run": workload["warmups"],
            "measured_iterations_per_run": workload["measured_iterations"],
            "failure_count": 0,
            "dropped_trace_records": 0,
        }
        requests = [
            request
            for run in fixture.candidate
            for request in run["requests"]
        ]
        candidate["metrics"] = {
            "ttft_p95_ms": release_performance.native_profile.r7(
                [request["ttft_ms"] for request in requests], 0.95
            ),
            "tpot_p95_ms": release_performance.native_profile.r7(
                [request["tpot_ms"] for request in requests], 0.95
            ),
            "e2e_median_ms": release_performance.native_profile.r7(
                [request["e2e_ms"] for request in requests], 0.50
            ),
            "throughput_median_output_tokens_per_second": (
                release_performance.native_profile.r7(
                    [
                        run["aggregate"]["throughput_output_tokens_per_second"]
                        for run in fixture.candidate
                    ],
                    0.50,
                )
            ),
        }
        self._write_tar(
            self.paths["performance_raw"],
            {
                f"candidate-{index}.json": raw_path.read_bytes()
                for index, raw_path in enumerate(fixture.candidate_paths, 1)
            },
        )
        ratios = {
            metric: candidate["metrics"][metric] / baseline["metrics"][metric]
            for metric in release_performance.METRIC_FIELDS
        }
        return {
            "schema_version": "rustinfer.release-performance-report.v1",
            "status": "passed",
            "passed": True,
            "baseline": {
                "baseline_id": baseline["baseline_id"],
                "sha256": release_performance.BASELINE_SHA256,
                "metrics": baseline["metrics"],
            },
            "candidate": candidate,
            "ratios": ratios,
            "checks": [
                release_performance._check(
                    "ttft_p95_regression", ratios["ttft_p95_ms"], "<=", 1.05
                ),
                release_performance._check(
                    "tpot_p95_regression", ratios["tpot_p95_ms"], "<=", 1.05
                ),
                release_performance._check(
                    "e2e_median_regression", ratios["e2e_median_ms"], "<=", 1.05
                ),
                release_performance._check(
                    "throughput_median_regression",
                    ratios["throughput_median_output_tokens_per_second"],
                    ">=",
                    0.95,
                ),
            ],
            "errors": [],
        }

    def _build_documents(self) -> None:
        self.documents["cuda_report"] = copy.deepcopy(self.cuda_attestation)
        summary_variant = {
            "case_count": 31,
            "failure_count": 0,
            "numeric_pass": True,
            "semantic_pass": True,
            "aggregate_numeric": {},
            "pass": True,
        }
        metric = {"metrics": {}, "pass": True}
        case_variant = {
            "numeric": {
                "first_layer_hidden": copy.deepcopy(metric),
                "final_logits": copy.deepcopy(metric),
                "final_log_probs": copy.deepcopy(metric),
            },
            "semantic": {"pass": True},
            "pass": True,
        }
        self.documents["native_correctness"] = {
            "schema_version": "1.0.0",
            "gate_id": "smollm2-fp32-bf16-native-e0-v2",
            "created_at": "2026-08-26T00:00:00Z",
            "status": "pass",
            "roles": {},
            "gate_contract": {},
            "inputs": {},
            "bindings": {
                "candidate_git_revision": REVISION,
                "candidate_git_status_sha256": hashlib.sha256(b"").hexdigest(),
                "candidate_executable_sha256": digest(b"correctness executable"),
                "model_id": "HuggingFaceTB/SmolLM2-135M",
                "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "config_sha256": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
                "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
                "tokenizer_sha256": "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db",
            },
            "summary": {
                "case_count": 31,
                "candidate_variant_count": 2,
                "failure_count": 0,
                "numeric_pass": True,
                "semantic_pass": True,
                "variants": {
                    "canonical-v1": copy.deepcopy(summary_variant),
                    "fixed-contiguous-37-balanced-v1": copy.deepcopy(summary_variant),
                },
            },
            "cases": [
                {
                    "prompt_id": f"prompt-{index:02d}",
                    "variants": {
                        "canonical-v1": copy.deepcopy(case_variant),
                        "fixed-contiguous-37-balanced-v1": copy.deepcopy(case_variant),
                    },
                    "pass": True,
                }
                for index in range(31)
            ],
        }
        native_correctness_sha = digest(
            (json.dumps(self.documents["native_correctness"], sort_keys=True, indent=2) + "\n").encode()
        )
        self.documents["python_report"] = self._build_python_e2e_document(
            native_correctness_sha
        )
        self.documents["native_replay_validation"] = {
            "schema_version": NATIVE_REPLAY_VERSION,
            "status": "passed",
            "source": {
                "git_revision": REVISION,
                "git_dirty": False,
                "source_archive_sha256": self._binding()["source_archive_sha256"],
            },
            "correctness_report_sha256": native_correctness_sha,
            "raw_replay_sha256": digest(self.paths["native_replay"].read_bytes()),
            "case_count": 31,
            "failure_count": 0,
            "checks": [
                {"id": check_id, "passed": True}
                for check_id in sorted(
                    {
                        "schema-closed-validation",
                        "raw-input-hashes-replayed",
                        "all-cases-replayed",
                        "summary-recomputed",
                    }
                )
            ],
        }
        log_hashes = {
            test_id: digest(contents)
            for test_id, contents in self.optimization_logs.items()
        }
        self.documents["optimization_correctness"] = {
            "schema_version": 1,
            "gate_id": "pr15-iteration-command-batch-exact-v1",
            "recorded_at_utc": "2026-08-26T00:00:00Z",
            "status": "passed",
            "semantic_class": "E0",
            "source": {
                "git_commit": REVISION,
                "git_dirty": False,
                "archive_sha256": self._binding()["source_archive_sha256"],
            },
            "build": {
                "container_image_sha256": digest(b"profile image"),
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
            "model": {
                "model_id": "HuggingFaceTB/SmolLM2-135M",
                "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "dtype": "bf16",
                "manifest_sha256": digest(b"model manifest"),
                "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
                "tokenizer_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
            },
            "implementations": {
                "baseline": "per-operation",
                "candidate": "iteration-batch",
                "residual_rmsnorm": "separate",
                "rollback": "--execution-completion per-operation",
            },
            "tests": [
                {"id": "cuda-compile-only", "result": "passed", "log_sha256": log_hashes["cuda-compile-only"]},
                {"id": "workspace-all-features-all-targets", "result": "passed", "log_sha256": log_hashes["workspace-all-features-all-targets"]},
                {
                    "id": "command-batch-lifecycle", "result": "passed",
                    "one_shot_finish": True, "drop_restores_stream": True,
                    "log_sha256": log_hashes["command-batch-lifecycle"],
                },
                {
                    "id": "command-batch-resource-ledger", "result": "passed",
                    "validation_fail_closed": True, "queued_chain_raw_byte_mismatches": 0,
                    "hot_loop_allocation_delta": 0, "stream_reuse_after_finish": True,
                    "owner_close_allocation_count": 0,
                    "log_sha256": log_hashes["command-batch-resource-ledger"],
                },
                {
                    "id": "smollm2-multi-step-greedy-exact", "result": "passed",
                    "decode_steps": 16, "committed_iterations": 16,
                    "raw_logit_mismatches": 0,
                    "generated_token_ids": [4052, 2025, 284, 965, 6497, 288, 1492, 418, 260, 16438, 30, 198, 198, 504, 16438, 314],
                    "token_id_mismatches": 0, "hot_loop_allocation_delta": 0,
                    "owner_close_allocation_count": 0,
                    "log_sha256": log_hashes["smollm2-multi-step-greedy-exact"],
                },
            ],
        }
        optimization_sha = digest(
            (json.dumps(self.documents["optimization_correctness"], sort_keys=True, indent=2) + "\n").encode()
        )
        self.documents["performance"] = self._build_performance_document(
            optimization_sha
        )
        soak_checks = set(SOAK_GLOBAL_CHECKS)
        soak_summaries = []
        for scenario_id, (kind, duration) in SOAK_SCENARIOS.items():
            soak_checks.update(
                f"{scenario_id}.{suffix}"
                for suffix in SOAK_COMMON_SCENARIO_CHECKS
            )
            if scenario_id in SOAK_GOLDEN_SCENARIOS:
                soak_checks.add(f"{scenario_id}.golden_parity")
            soak_summaries.append(
                {
                    "scenario_id": scenario_id,
                    "kind": kind,
                    "events": duration + 2,
                    "samples": duration,
                    "requests": 1,
                    "maximum_sample_gap_ms": 1000.0,
                    "observed_duration_seconds": float(duration),
                    "sample_span_seconds": max(0.0, duration - 1.0),
                    "rss_slope_bytes_per_hour": 0.0,
                    "vram_slope_bytes_per_hour": 0.0,
                }
            )
        self.documents["soak"] = {
            "schema_version": "rustinfer.reliability-soak-report.v1",
            "status": "passed",
            "passed": True,
            "bindings": {
                "contract_id": SOAK_CONTRACT_ID,
                "reviewed_manifest_template_canonical_sha256": (
                    SOAK_TEMPLATE_CANONICAL_SHA256
                ),
                "manifest_sha256": digest(b"soak manifest"),
                "binding_sha256": digest(b"soak binding"),
                "source": {
                    "git_commit": REVISION,
                    "git_dirty": False,
                    "source_archive_sha256": self._binding()["source_archive_sha256"],
                    "binary_sha256": self._binding()["release_binary_sha256"],
                    "image_sha256": self.image_sha,
                    "model_sha256": self.e2e_model_tree_sha256,
                    "model_id": "HuggingFaceTB/SmolLM2-135M",
                    "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                },
            },
            "scenario_summaries": soak_summaries,
            "observations": {
                "event_count": sum(summary["events"] for summary in soak_summaries),
                "outcome_counts": {
                    "cancelled": 100,
                    "disconnected": 100,
                    "overload": 20,
                },
                "service_counter_maxima": {
                    "cancellations": 100,
                    "disconnects": 100,
                    "overloads": 20,
                },
                "final": {
                    "active_requests": 0,
                    "waiting_requests": 0,
                    "kv_allocated_blocks": 0,
                    "device_live_count": 0,
                    "device_live_bytes": 0,
                    "pinned_live_count": 0,
                    "pinned_live_bytes": 0,
                },
            },
            "checks": [
                {
                    "name": name,
                    "passed": True,
                    "observed": True,
                    "threshold": True,
                }
                for name in sorted(soak_checks)
            ],
            "errors": [],
        }

    def write_reports(self) -> None:
        for name in (
            "python_report", "cuda_report", "native_correctness",
            "native_replay_validation", "optimization_correctness", "performance", "soak",
        ):
            self.paths[name].write_text(
                json.dumps(self.documents[name], sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

    def refresh_manifest(self) -> None:
        self.write_reports()

        def artifact(name: str) -> dict[str, str]:
            path = self.paths[name]
            return {"path": path.relative_to(self.root).as_posix(), "sha256": digest(path.read_bytes())}

        self.manifest = {
            "schema_version": MANIFEST_VERSION,
            "candidate_id": "rustinfer-0.1.0-rc1",
            "source": {
                "git_revision": REVISION,
                "git_dirty": False,
                "archive": artifact("source"),
            },
            "release": {
                "binary": artifact("binary"),
                "bundle": artifact("bundle"),
                "image_digest": f"sha256:{self.image_sha}",
            },
            "evidence": {
                "python_free_e2e": {"report": artifact("python_report"), "raw_evidence": artifact("python_raw")},
                "cuda_fault": {
                    "build_image_id": self.cuda_build_image_id,
                    "report": artifact("cuda_report"),
                    "raw_evidence": artifact("cuda_raw"),
                },
                "native_correctness": {
                    "report": artifact("native_correctness"),
                    "raw_replay": artifact("native_replay"),
                    "replay_validation": artifact("native_replay_validation"),
                },
                "optimization_correctness": {
                    "report": artifact("optimization_correctness"),
                    "raw_evidence": artifact("optimization_raw"),
                },
                "performance": {
                    "report": artifact("performance"),
                    "raw_evidence": artifact("performance_raw"),
                },
                "reliability_soak": {
                    "report": artifact("soak"),
                    "raw_evidence": artifact("soak_raw"),
                },
            },
        }
        self.write_manifest()

    def write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def evaluate(self) -> dict[str, object]:
        replay = {"report": copy.deepcopy(self.documents["soak"])}
        with mock.patch.object(
            reliability_soak,
            "replay_raw_evidence_archive",
            return_value=replay,
        ):
            return evaluate(self.manifest_path, self.root)


class ReleaseCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = CandidateFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_source_bound_candidate_passes(self) -> None:
        report = self.fixture.evaluate()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["bindings"]["git_revision"], REVISION)
        self.assertEqual(len(report["bindings"]["evidence_sha256"]), 13)

    def test_failed_or_missing_gate_fails_closed(self) -> None:
        del self.fixture.documents["performance"]["status"]
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("closed object mismatch", report["errors"][0])

    def test_cross_binding_mismatch_fails_closed(self) -> None:
        source = self.fixture.documents["performance"]["candidate"]["source"]
        source["release_binary_sha256"] = digest(b"different binary")
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("release_binary_sha256", report["errors"][0])

    def test_performance_must_bind_optimization_not_native_report(self) -> None:
        native_bytes = (
            json.dumps(
                self.fixture.documents["native_correctness"], sort_keys=True, indent=2
            )
            + "\n"
        ).encode()
        source = self.fixture.documents["performance"]["candidate"]["source"]
        source["correctness_report_sha256"] = digest(native_bytes)
        source["correctness_gate_id"] = "smollm2-fp32-bf16-native-e0-v2"
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("correctness_gate_id", report["errors"][0])

    def test_optimizer_log_hashes_must_match_exact_raw_inventory(self) -> None:
        logs = {
            OPTIMIZATION_LOGS[test_id]: contents
            for test_id, contents in self.fixture.optimization_logs.items()
        }
        logs[OPTIMIZATION_LOGS["command-batch-resource-ledger"]] = b"different log\n"
        self.fixture._write_tar(self.fixture.paths["optimization_raw"], logs)
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("declared log hashes", report["errors"][0])

    def test_performance_metrics_are_recomputed_from_raw_runs(self) -> None:
        self.fixture.documents["performance"]["candidate"]["metrics"][
            "ttft_p95_ms"
        ] *= 0.5
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("raw-derived R7 metrics", report["errors"][0])

    def test_performance_raw_inventory_is_closed(self) -> None:
        files = {
            f"candidate-{index}.json": path.read_bytes()
            for index, path in enumerate(
                self.fixture.performance_fixture.candidate_paths, 1
            )
        }
        files["self-asserted-summary.json"] = b'{"passed":true}\n'
        self.fixture._write_tar(self.fixture.paths["performance_raw"], files)
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("must contain only", report["errors"][0])

    def test_native_replay_validation_binds_raw_bundle(self) -> None:
        self.fixture.documents["native_replay_validation"]["raw_replay_sha256"] = digest(
            b"unrelated replay"
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("raw replay bundle digest mismatch", report["errors"][0])

    def test_source_archive_requires_exact_git_pax_comment(self) -> None:
        self.fixture._write_tar(
            self.fixture.paths["source"],
            {"README.md": b"exact source archive fixture"},
            pax_headers={"comment": "f" * 40},
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("git-archive pax global comment", report["errors"][0])

    def test_tampered_hashed_artifact_fails_closed(self) -> None:
        self.fixture.paths["cuda_raw"].write_bytes(b"tampered after manifest")
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("artifact digest mismatch", report["errors"][0])

    def test_path_traversal_fails_before_file_access(self) -> None:
        self.fixture.manifest["release"]["binary"]["path"] = "../rustinfer"
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("path traversal", report["errors"][0])

    def test_symlink_artifact_is_rejected(self) -> None:
        link = self.fixture.root / "binary-link"
        link.symlink_to(self.fixture.paths["binary"])
        self.fixture.manifest["release"]["binary"] = {
            "path": link.name,
            "sha256": digest(self.fixture.paths["binary"].read_bytes()),
        }
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("symlink", report["errors"][0])

    def test_duplicate_json_key_is_rejected(self) -> None:
        raw = self.fixture.manifest_path.read_text(encoding="utf-8")
        raw = raw.replace(
            '"candidate_id": "rustinfer-0.1.0-rc1",',
            '"candidate_id": "first", "candidate_id": "second",',
            1,
        )
        self.fixture.manifest_path.write_text(raw, encoding="utf-8")
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("duplicate JSON key", report["errors"][0])

    def test_placeholder_is_rejected(self) -> None:
        self.fixture.manifest["candidate_id"] = "replace-me"
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("placeholder", report["errors"][0])

    def test_bundle_and_standalone_binary_must_match(self) -> None:
        self.fixture.paths["binary"].write_bytes(
            self.fixture.paths["binary"].read_bytes() + b"changed"
        )
        self.fixture.paths["binary"].chmod(0o755)
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("standalone binary differs", report["errors"][0])

    def test_required_attestation_check_set_is_closed(self) -> None:
        checks = self.fixture.documents["cuda_report"]["checks"]
        checks.pop()
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("required check set mismatch", report["errors"][0])

    def test_cuda_attestation_must_equal_raw_replay(self) -> None:
        self.fixture.documents["cuda_report"]["checks"].reverse()
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("submitted attestation differs from raw replay", report["errors"][0])

    def test_cuda_raw_payload_cannot_be_self_attested_after_tampering(self) -> None:
        raw_path = self.fixture.paths["cuda_raw"]
        contents = raw_path.read_bytes()
        self.assertIn(b"fixture evidence\n", contents)
        raw_path.write_bytes(
            contents.replace(b"fixture evidence\n", b"fixture evidencf\n", 1)
        )
        self.fixture.documents["cuda_report"]["raw_evidence_sha256"] = digest(
            raw_path.read_bytes()
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("digest mismatch", report["errors"][0])

    def test_cuda_build_image_is_rebound_from_raw_environment(self) -> None:
        self.fixture.manifest["evidence"]["cuda_fault"]["build_image_id"] = (
            "sha256:" + "f" * 64
        )
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("gpu_image_id", report["errors"][0])

    def test_python_free_attestation_must_equal_raw_replay(self) -> None:
        self.fixture.documents["python_report"]["checks"].reverse()
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("differs from raw replay", report["errors"][0])

    def test_python_free_raw_source_cannot_be_self_rebound(self) -> None:
        payloads = e2e_fixture_module.read_raw_tar(
            self.fixture.paths["python_raw"]
        )
        raw = json.loads(payloads["raw-evidence.json"])
        raw["source"]["source_archive_sha256"] = digest(b"other source")
        payloads["raw-evidence.json"] = (
            json.dumps(raw, sort_keys=True, indent=2) + "\n"
        ).encode()
        payloads["SHA256SUMS"] = b"".join(
            f"{digest(payloads[name])}  {name}\n".encode("ascii")
            for name in e2e_fixture_module.checker.RAW_ARCHIVE_PAYLOADS
        )
        e2e_fixture_module.write_raw_tar(
            self.fixture.paths["python_raw"], payloads
        )
        self.fixture.documents["python_report"]["raw_evidence_sha256"] = digest(
            self.fixture.paths["python_raw"].read_bytes()
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("raw.source", report["errors"][0])

    def test_soak_contract_and_duration_checks_cannot_be_self_asserted(self) -> None:
        self.fixture.documents["soak"]["bindings"][
            "reviewed_manifest_template_canonical_sha256"
        ] = digest(b"easier soak contract")
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("reviewed soak manifest digest", report["errors"][0])

        second_root = Path(self.temporary.name) / "second"
        second_root.mkdir()
        second_fixture = CandidateFixture(second_root)
        checks = second_fixture.documents["soak"]["checks"]
        checks[:] = [
            check
            for check in checks
            if check["name"] != "steady.duration_seconds"
        ]
        second_fixture.refresh_manifest()
        report = second_fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("exact reviewed soak check inventory", report["errors"][0])

    def test_soak_submitted_report_must_equal_raw_replay(self) -> None:
        replayed = copy.deepcopy(self.fixture.documents["soak"])
        self.fixture.documents["soak"]["scenario_summaries"][0]["events"] += 1
        self.fixture.refresh_manifest()
        with mock.patch.object(
            reliability_soak,
            "replay_raw_evidence_archive",
            return_value={"report": replayed},
        ):
            report = evaluate(self.fixture.manifest_path, self.fixture.root)
        self.assertFalse(report["passed"])
        self.assertIn("differs from the raw-replayed report", report["errors"][0])

    def test_soak_model_tree_must_match_python_free_e2e(self) -> None:
        self.fixture.documents["soak"]["bindings"]["source"][
            "model_sha256"
        ] = digest(b"substituted soak model")
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("model_sha256", report["errors"][0])

    def test_soak_summary_must_span_the_reviewed_duration(self) -> None:
        summary = next(
            summary
            for summary in self.fixture.documents["soak"]["scenario_summaries"]
            if summary["scenario_id"] == "steady"
        )
        summary["observed_duration_seconds"] = 60.0
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("scenario was truncated", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
