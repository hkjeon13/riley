#!/usr/bin/env python3
"""CPU-only tests for the final release-candidate evidence gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import check_release_candidate as release_candidate_module  # noqa: E402
from build_release_bundle import build_bundle  # noqa: E402
from check_release_candidate import (  # noqa: E402
    ATTESTATION_VERSION,
    CUDA_FAULT_CHECKS,
    MANIFEST_VERSION,
    OPTIMIZATION_LOGS,
    PYTHON_FREE_CHECKS,
    REPORT_VERSION,
    SOAK_COMMON_SCENARIO_CHECKS,
    SOAK_CONTRACT_ID,
    SOAK_GLOBAL_CHECKS,
    SOAK_GOLDEN_SCENARIOS,
    SOAK_SCENARIOS,
    SOAK_TEMPLATE_CANONICAL_SHA256,
    _validate_python_free_e2e_replay,
    cuda_fault_evidence,
    evaluate,
    optimization_evidence,
    python_free_e2e,
    reliability_soak,
    release_performance,
    reproducible_build_evidence,
)
from release_common import MIT_LICENSE_BYTES  # noqa: E402
from test_native_correctness_evidence import NativeFixture as NativeEvidenceFixture  # noqa: E402
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

_NATIVE_TEMPLATE_TEMP = tempfile.TemporaryDirectory()
_NATIVE_TEMPLATE_ROOT = Path(_NATIVE_TEMPLATE_TEMP.name)
_NATIVE_TEMPLATE = NativeEvidenceFixture(_NATIVE_TEMPLATE_ROOT)


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


class CandidateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.revision = _NATIVE_TEMPLATE.result.source_revision
        repository = root / "repository"
        repository.mkdir()
        (repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\n'
            'license = "MIT"\n',
            encoding="utf-8",
        )
        (repository / "LICENSE").write_bytes(MIT_LICENSE_BYTES)
        self.paths = {
            "source": root / "source.tar",
            "binary": root / "rustinfer",
            "bundle": root / "rustinfer.tar.gz",
            "python_raw": root / "python-free-evidence.tar",
            "correctness_golden": root / "correctness-golden.json",
            "cuda_raw": root / "cuda-fault-evidence.tar",
            "python_report": root / "python-free-report.json",
            "cuda_report": root / "cuda-fault-report.json",
            "native_correctness": root / "native-correctness-report.json",
            "native_replay": root / "native-correctness-replay.tar",
            "native_executable": root / "rustinfer-native",
            "repro_build_a": root / "reproducible-build-a.tar",
            "repro_build_b": root / "reproducible-build-b.tar",
            "profile_binary": root / "rustinfer-profile",
            "native_manifest": root / "native-dependencies.txt",
            "optimization_correctness": root / "optimization-correctness-report.json",
            "optimization_raw": root / "optimization-correctness-evidence.tar",
            "performance": root / "performance-report.json",
            "performance_raw": root / "performance-evidence.tar",
            "soak": root / "soak-report.json",
            "soak_raw": root / "soak-evidence.tar",
        }
        shutil.copyfile(_NATIVE_TEMPLATE.candidate_source, self.paths["source"])
        self.trusted_source_sha256 = digest(self.paths["source"].read_bytes())
        shutil.copyfile(
            _NATIVE_TEMPLATE.correctness_report, self.paths["native_correctness"]
        )
        shutil.copyfile(_NATIVE_TEMPLATE.raw, self.paths["native_replay"])
        shutil.copyfile(_NATIVE_TEMPLATE.executable, self.paths["native_executable"])
        self.paths["native_executable"].chmod(0o755)
        self.paths["profile_binary"].write_bytes(
            fixture_elf() + b"rustinfer-profile\0"
        )
        self.paths["profile_binary"].chmod(0o755)
        self.paths["binary"].write_bytes(fixture_elf())
        self.paths["binary"].chmod(0o755)
        build_bundle(
            binary_path=self.paths["binary"],
            output=self.paths["bundle"],
            repository_root=repository,
            source_revision=self.revision,
            source_date_epoch=EPOCH,
        )
        with tarfile.open(self.paths["bundle"], "r:gz") as archive:
            native_member = next(
                member
                for member in archive.getmembers()
                if member.name.endswith("/manifest/native-dependencies.txt")
            )
            native_source = archive.extractfile(native_member)
            assert native_source is not None
            self.paths["native_manifest"].write_bytes(native_source.read())
        self.paths["repro_build_a"].write_bytes(b"reproducible build A fixture\n")
        self.paths["repro_build_b"].write_bytes(b"reproducible build B fixture\n")
        self.image_sha = digest(b"release image")
        self.cuda_build_image_id = CUDA_BUILD_IMAGE_ID
        self.reproducible_build_image_id = "sha256:" + digest(
            b"reproducible build image"
        )
        self.optimization_build_image_id = "sha256:" + digest(
            b"optimization build image"
        )
        cuda_template_root = root / "cuda-evidence-template"
        cuda_template_root.mkdir()
        cuda_template = CudaEvidenceFixture(cuda_template_root)
        environment_path = cuda_template.evidence / "environment.txt"
        environment = environment_path.read_text(encoding="utf-8")
        environment = environment.replace(
            f"source_revision={REVISION}",
            f"source_revision={self.revision}",
        )
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
            source_revision=self.revision,
            source_archive=self.paths["source"],
            build_image_id=self.cuda_build_image_id,
            release_binary=self.paths["binary"],
            release_bundle=self.paths["bundle"],
            release_image_id=f"sha256:{self.image_sha}",
            raw_evidence=self.paths["cuda_raw"],
            report=self.paths["cuda_report"],
        )
        self.paths["soak_raw"].write_bytes(b"soak raw evidence fixture")
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
        self.trusted_python_report = copy.deepcopy(self.documents["python_report"])
        self.trusted_python_raw_model = copy.deepcopy(self.python_raw_model)
        self.trusted_optimization_report = copy.deepcopy(
            self.documents["optimization_correctness"]
        )
        self.trusted_optimization_report_sha = digest(
            self.paths["optimization_correctness"].read_bytes()
        )
        self.trusted_optimization_raw_sha = digest(
            self.paths["optimization_raw"].read_bytes()
        )
        self.trusted_reproducible_hashes = {
            "a": digest(self.paths["repro_build_a"].read_bytes()),
            "b": digest(self.paths["repro_build_b"].read_bytes()),
            "binary": digest(self.paths["binary"].read_bytes()),
            "profile_binary": digest(self.paths["profile_binary"].read_bytes()),
            "bundle": digest(self.paths["bundle"].read_bytes()),
            "native_manifest": digest(self.paths["native_manifest"].read_bytes()),
        }
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
            "git_revision": self.revision,
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
        shutil.copyfile(template.raw_archive, self.paths["python_raw"])
        checker = e2e_fixture_module.checker
        golden = {
            "schema_version": checker.GOLDEN_SCHEMA,
            "correctness_gate_id": checker.CORRECTNESS_GATE,
            "correctness_report_sha256": native_correctness_sha256,
            "source_revision": self.revision,
            "model_id": checker.MODEL_ID,
            "model_revision": checker.MODEL_REVISION,
            "config_sha256": checker.MODEL_CONFIG_SHA256,
            "weights_sha256": checker.MODEL_WEIGHTS_SHA256,
            "tokenizer_aggregate_sha256": checker.TOKENIZER_AGGREGATE_SHA256,
            "tokenizer_json_sha256": checker.TOKENIZER_JSON_SHA256,
            "prompt": "A bounded release probe",
            "max_tokens": 8,
            "expected_greedy_text_sha256": template.expected_text_sha256,
        }
        golden_bytes = (
            json.dumps(golden, sort_keys=True, indent=2) + "\n"
        ).encode()
        self.paths["correctness_golden"].write_bytes(golden_bytes)
        self.correctness_golden_sha256 = digest(golden_bytes)
        self.python_raw_model = {
            "model_id": golden["model_id"],
            "model_revision": golden["model_revision"],
            "model_tree_sha256": digest(b"model manifest"),
            "config_sha256": golden["config_sha256"],
            "weights_sha256": golden["weights_sha256"],
            "tokenizer_aggregate_sha256": golden["tokenizer_aggregate_sha256"],
            "tokenizer_json_sha256": golden["tokenizer_json_sha256"],
            "correctness_gate_id": golden["correctness_gate_id"],
            "correctness_report_sha256": native_correctness_sha256,
            "correctness_golden_sha256": digest(golden_bytes),
        }
        self.e2e_model_tree_sha256 = str(
            self.python_raw_model["model_tree_sha256"]
        )
        # The final-candidate tests exercise cross-gate binding and mock only the
        # already unit-tested raw replay boundary. The copied archive remains a
        # complete v2 fixture, so artifact resolution still handles the real
        # archive shape instead of the retired synthetic five-file format.
        return self._attestation(
            "python-free-clean-runtime-e2e", "python_raw", PYTHON_FREE_CHECKS
        )

    def _build_performance_document(self, optimization_sha: str) -> dict[str, object]:
        baseline = json.loads(PERFORMANCE_BASELINE.read_text(encoding="utf-8"))
        raw_root = self.root / "performance-runs"
        raw_root.mkdir()
        fixture = profile_fixture_module.ProfilePairFixture(raw_root)
        profile_binary_sha = digest(self.paths["profile_binary"].read_bytes())
        profile_image_sha = self.optimization_build_image_id.removeprefix(
            "sha256:"
        )
        for run_index, run in enumerate(fixture.candidate):
            run["source"] = {
                "git_commit": self.revision,
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
            "candidate_id": "rustinfer-0.1.0-rc1",
            "recorded_at_utc": "2026-08-26T00:00:00Z",
            "source": {
                "git_commit": self.revision,
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
        release_performance.write_raw_evidence_archive(
            self.paths["performance_raw"],
            [
                (f"candidate-{index}.json", raw_path.read_bytes())
                for index, raw_path in enumerate(fixture.candidate_paths, 1)
            ],
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
        self.documents["native_correctness"] = json.loads(
            self.paths["native_correctness"].read_text(encoding="utf-8")
        )
        native_correctness_sha = digest(
            (json.dumps(self.documents["native_correctness"], sort_keys=True, indent=2) + "\n").encode()
        )
        self.documents["python_report"] = self._build_python_e2e_document(
            native_correctness_sha
        )
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
                "git_commit": self.revision,
                "git_dirty": False,
                "archive_sha256": self._binding()["source_archive_sha256"],
            },
            "build": {
                "container_image_sha256": (
                    self.optimization_build_image_id.removeprefix("sha256:")
                ),
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
                    "cuda_live_allocation_delta": 0, "stream_reuse_after_finish": True,
                    "owner_close_live_allocation_count": 0,
                    "log_sha256": log_hashes["command-batch-resource-ledger"],
                },
                {
                    "id": "smollm2-multi-step-greedy-exact", "result": "passed",
                    "decode_steps": 16, "committed_iterations": 16,
                    "raw_logit_mismatches": 0,
                    "generated_token_ids": [4052, 2025, 284, 965, 6497, 288, 1492, 418, 260, 16438, 30, 198, 198, 504, 16438, 314],
                    "token_id_mismatches": 0, "cuda_live_allocation_delta": 0,
                    "owner_close_live_allocation_count": 0,
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
                    "git_commit": self.revision,
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
            "optimization_correctness", "performance", "soak",
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
                "git_revision": self.revision,
                "git_dirty": False,
                "archive": artifact("source"),
            },
            "release": {
                "binary": artifact("binary"),
                "bundle": artifact("bundle"),
                "image_digest": f"sha256:{self.image_sha}",
            },
            "evidence": {
                "python_free_e2e": {
                    "report": artifact("python_report"),
                    "raw_evidence": artifact("python_raw"),
                    "correctness_golden": artifact("correctness_golden"),
                },
                "cuda_fault": {
                    "build_image_id": self.cuda_build_image_id,
                    "report": artifact("cuda_report"),
                    "raw_evidence": artifact("cuda_raw"),
                },
                "native_correctness": {
                    "report": artifact("native_correctness"),
                    "raw_replay": artifact("native_replay"),
                    "candidate_executable": artifact("native_executable"),
                },
                "reproducible_build": {
                    "build_image_id": self.reproducible_build_image_id,
                    "source_date_epoch": EPOCH,
                    "build_a": artifact("repro_build_a"),
                    "build_b": artifact("repro_build_b"),
                    "profile_binary": artifact("profile_binary"),
                    "native_manifest": artifact("native_manifest"),
                },
                "optimization_correctness": {
                    "build_image_id": self.optimization_build_image_id,
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

    def reproducibility_replay(self) -> dict[str, object]:
        commands = reproducible_build_evidence.expected_commands(self.revision, EPOCH)
        return {
            "schema_version": reproducible_build_evidence.SCHEMA_VERSION,
            "gate_id": reproducible_build_evidence.GATE_ID,
            "status": "passed",
            "source": {
                "revision": self.revision,
                "archive_sha256": self.trusted_source_sha256,
                "source_date_epoch": EPOCH,
            },
            "build": {
                "image_id": self.reproducible_build_image_id,
                "image_inspect_sha256": digest(b"builder image inspect"),
                "platform": reproducible_build_evidence.PLATFORM,
                "network": "none",
                "cargo_command": commands["build"],
                "profile_cargo_command": commands["profile_build"],
                "rust_toolchain": reproducible_build_evidence.RUST_TOOLCHAIN,
                "cuda_toolkit": reproducible_build_evidence.CUDA_TOOLKIT,
                "nvcc_version": reproducible_build_evidence.NVCC_VERSION,
                "cuda_architectures": reproducible_build_evidence.CUDA_ARCHITECTURES,
                "independent_clean_containers": 2,
            },
            "evidence": {
                "a_sha256": self.trusted_reproducible_hashes["a"],
                "b_sha256": self.trusted_reproducible_hashes["b"],
                "a_container_id": "a" * 64,
                "b_container_id": "b" * 64,
                "a_workspace_volume": "fixture-a",
                "b_workspace_volume": "fixture-b",
                "a_workspace_source": "/fixture/a",
                "b_workspace_source": "/fixture/b",
                "a_started_at": "2026-08-26T00:00:00.000000000Z",
                "a_finished_at": "2026-08-26T00:01:00.000000000Z",
                "b_started_at": "2026-08-26T00:02:00.000000000Z",
                "b_finished_at": "2026-08-26T00:03:00.000000000Z",
            },
            "artifacts": {
                "binary_sha256": self.trusted_reproducible_hashes["binary"],
                "profile_binary_sha256": self.trusted_reproducible_hashes[
                    "profile_binary"
                ],
                "bundle_sha256": self.trusted_reproducible_hashes["bundle"],
                "native_manifest_sha256": self.trusted_reproducible_hashes[
                    "native_manifest"
                ],
            },
            "comparisons": {
                "binary_a_b_final_byte_exact": True,
                "profile_binary_a_b_final_byte_exact": True,
                "bundle_a_b_final_byte_exact": True,
                "native_manifest_a_b_final_byte_exact": True,
                "source_archive_a_b_final_byte_exact": True,
            },
        }

    def optimization_replay(self) -> dict[str, object]:
        return {
            "report": copy.deepcopy(self.trusted_optimization_report),
            "report_sha256": self.trusted_optimization_report_sha,
            "raw_evidence_sha256": self.trusted_optimization_raw_sha,
            "profile_binary_sha256": self.trusted_reproducible_hashes[
                "profile_binary"
            ],
            "build_image_sha256": self.optimization_build_image_id.removeprefix(
                "sha256:"
            ),
            "log_sha256": {},
            "test_binary_sha256": {},
        }

    def evaluate(
        self,
        *,
        manifest_path: Path | None = None,
        manifest_fd: int | None = None,
        soak_replay: dict[str, object] | None = None,
        reproducibility_replay: dict[str, object] | None = None,
        optimization_replay: dict[str, object] | None = None,
        **anchor_overrides: str,
    ) -> dict[str, object]:
        if soak_replay is None:
            soak_replay = {"report": copy.deepcopy(self.documents["soak"])}
        if reproducibility_replay is None:
            reproducibility_replay = self.reproducibility_replay()
        if optimization_replay is None:
            optimization_replay = self.optimization_replay()
        anchors = {
            "expected_candidate_id": "rustinfer-0.1.0-rc1",
            "expected_revision": self.revision,
            "expected_source_archive_sha256": self.trusted_source_sha256,
            "expected_release_image_id": f"sha256:{self.image_sha}",
            "expected_reproducible_build_image_id": (
                self.reproducible_build_image_id
            ),
            "expected_cuda_build_image_id": self.cuda_build_image_id,
            "expected_optimization_build_image_id": (
                self.optimization_build_image_id
            ),
            "expected_correctness_golden_sha256": (
                self.correctness_golden_sha256
            ),
        }
        anchors.update(anchor_overrides)
        python_archive = {
            "raw": {"model": copy.deepcopy(self.trusted_python_raw_model)}
        }

        def replay_python_free(
            archive: object, **arguments: object
        ) -> tuple[dict[str, object], None]:
            if archive is not python_archive:
                raise AssertionError("candidate did not replay the loaded E2E archive")
            expected = {
                "source_revision": self.revision,
                "source_archive_sha256": digest(self.paths["source"].read_bytes()),
                "release_binary_sha256": digest(self.paths["binary"].read_bytes()),
                "release_bundle_sha256": digest(self.paths["bundle"].read_bytes()),
                "image_id": f"sha256:{self.image_sha}",
                "correctness_report": self.documents["native_correctness"],
                "correctness_report_sha256": digest(
                    self.paths["native_correctness"].read_bytes()
                ),
                "correctness_golden_sha256": self.correctness_golden_sha256,
            }
            if arguments != expected:
                raise AssertionError(
                    f"candidate E2E replay arguments differ: {arguments!r}"
                )
            return copy.deepcopy(self.trusted_python_report), None

        def replay_reproducible_build(**arguments: object) -> dict[str, object]:
            expected_scalars = {
                "expected_source_archive_sha256": self.trusted_source_sha256,
                "source_revision": self.revision,
                "source_date_epoch": EPOCH,
                "build_image_id": self.reproducible_build_image_id,
            }
            for field, expected in expected_scalars.items():
                if arguments.get(field) != expected:
                    raise AssertionError(
                        f"reproducibility replay {field} differs: "
                        f"{arguments.get(field)!r}"
                    )
            expected_artifacts = {
                "evidence_a": self.paths["repro_build_a"],
                "evidence_b": self.paths["repro_build_b"],
                "source_archive": self.paths["source"],
                "final_binary": self.paths["binary"],
                "final_profile_binary": self.paths["profile_binary"],
                "final_bundle": self.paths["bundle"],
                "final_native_manifest": self.paths["native_manifest"],
            }
            for field, expected_path in expected_artifacts.items():
                actual_path = arguments.get(field)
                if not isinstance(actual_path, Path):
                    raise AssertionError(
                        f"reproducibility replay {field} is not a Path"
                    )
                if digest(actual_path.read_bytes()) != digest(expected_path.read_bytes()):
                    raise AssertionError(
                        f"reproducibility replay {field} has the wrong bytes"
                    )
            return reproducibility_replay

        def replay_optimization(
            raw_evidence: object, **arguments: object
        ) -> dict[str, object]:
            if not isinstance(raw_evidence, Path) or digest(
                raw_evidence.read_bytes()
            ) != digest(self.paths["optimization_raw"].read_bytes()):
                raise AssertionError("optimizer replay received the wrong raw evidence")
            expected_scalars = {
                "source_revision": self.revision,
                "source_archive_sha256": self.trusted_source_sha256,
                "build_image_id": self.optimization_build_image_id,
            }
            for field, expected in expected_scalars.items():
                if arguments.get(field) != expected:
                    raise AssertionError(
                        f"optimizer replay {field} differs: {arguments.get(field)!r}"
                    )
            expected_artifacts = {
                "report": self.paths["optimization_correctness"],
                "profile_binary": self.paths["profile_binary"],
            }
            for field, expected_path in expected_artifacts.items():
                actual_path = arguments.get(field)
                if not isinstance(actual_path, Path):
                    raise AssertionError(f"optimizer replay {field} is not a Path")
                if digest(actual_path.read_bytes()) != digest(expected_path.read_bytes()):
                    raise AssertionError(f"optimizer replay {field} has the wrong bytes")
            return optimization_replay

        with mock.patch.object(
            reliability_soak,
            "replay_raw_evidence_archive",
            return_value=soak_replay,
        ), mock.patch.object(
            reproducible_build_evidence,
            "check_reproducible_build",
            side_effect=replay_reproducible_build,
        ), mock.patch.object(
            optimization_evidence,
            "replay_raw_evidence",
            side_effect=replay_optimization,
        ), mock.patch.object(
            python_free_e2e,
            "load_raw_evidence_archive",
            return_value=python_archive,
        ), mock.patch.object(
            python_free_e2e,
            "validate_bound_raw_archive",
            side_effect=replay_python_free,
        ):
            return evaluate(
                manifest_path or self.manifest_path,
                self.root,
                manifest_fd=manifest_fd,
                **anchors,
            )


class ReleaseCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = CandidateFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_source_bound_candidate_passes(self) -> None:
        self.assertNotEqual(
            digest(self.fixture.paths["native_executable"].read_bytes()),
            digest(self.fixture.paths["profile_binary"].read_bytes()),
        )
        self.assertEqual(
            len(
                {
                    self.fixture.reproducible_build_image_id,
                    self.fixture.cuda_build_image_id,
                    self.fixture.optimization_build_image_id,
                }
            ),
            3,
        )
        report = self.fixture.evaluate()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["schema_version"], REPORT_VERSION)
        self.assertEqual(report["candidate_id"], "rustinfer-0.1.0-rc1")
        self.assertEqual(report["bindings"]["git_revision"], self.fixture.revision)
        self.assertEqual(
            report["bindings"]["build_image_ids"],
            {
                "reproducible_build": self.fixture.reproducible_build_image_id,
                "cuda_fault": self.fixture.cuda_build_image_id,
                "optimization_correctness": (
                    self.fixture.optimization_build_image_id
                ),
            },
        )
        self.assertEqual(
            report["bindings"]["native_correctness_executable_sha256"],
            digest(self.fixture.paths["native_executable"].read_bytes()),
        )
        self.assertEqual(
            report["bindings"]["profile_binary_sha256"],
            digest(self.fixture.paths["profile_binary"].read_bytes()),
        )
        self.assertEqual(
            set(report["bindings"]["evidence_sha256"]),
            {
                "cuda_fault",
                "cuda_fault_raw",
                "native_correctness_report",
                "native_correctness_raw",
                "optimization_correctness",
                "optimization_correctness_raw",
                "performance",
                "performance_raw",
                "python_free_e2e",
                "python_free_e2e_correctness_golden_raw",
                "python_free_e2e_raw",
                "reliability_soak",
                "reliability_soak_raw",
                "reproducible_build_a_raw",
                "reproducible_build_b_raw",
                "reproducible_build_native_manifest_raw",
            },
        )

    def test_cli_exposes_only_role_specific_build_image_anchors(self) -> None:
        destinations = {
            action.dest for action in release_candidate_module._parser()._actions
        }
        self.assertNotIn("expected_build_image_id", destinations)
        self.assertTrue(
            {
                "expected_candidate_id",
                "expected_reproducible_build_image_id",
                "expected_cuda_build_image_id",
                "expected_optimization_build_image_id",
            }.issubset(destinations)
        )

    def test_legacy_manifest_v1_is_rejected(self) -> None:
        self.fixture.manifest["schema_version"] = (
            "rustinfer.release-candidate-manifest.v1"
        )
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn(MANIFEST_VERSION, report["errors"][0])

    def test_python_free_v2_archive_replays_through_candidate_boundary(self) -> None:
        root = self.fixture.root / "candidate-e2e-integration"
        root.mkdir()
        e2e = e2e_fixture_module.E2EFixture(root)
        report, diagnostic = e2e.replay()
        self.assertIsNone(diagnostic)
        model = _validate_python_free_e2e_replay(
            report,
            e2e.raw_archive,
            revision=e2e.revision,
            archive_sha256=e2e.hashes["archive"],
            binary_sha256=e2e.hashes["binary"],
            bundle_sha256=e2e.hashes["bundle"],
            image_sha256=e2e.image_id.removeprefix("sha256:"),
            native_correctness=json.loads(
                e2e.correctness_report.read_text(encoding="utf-8")
            ),
            native_correctness_sha256=e2e.hashes["correctness"],
            optimization_correctness={
                "model": {
                    "model_id": e2e.model_id,
                    "revision": e2e.model_revision,
                    "manifest_sha256": e2e.hashes["model"],
                    "weights_sha256": e2e.hashes["weights"],
                    "tokenizer_sha256": e2e.hashes["tokenizer_json"],
                }
            },
            correctness_golden_sha256=e2e.hashes["golden"],
        )
        self.assertEqual(model["model_tree_sha256"], e2e.hashes["model"])

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

    def test_optimizer_raw_evidence_must_equal_v2_replay(self) -> None:
        logs = {
            OPTIMIZATION_LOGS[test_id]: contents
            for test_id, contents in self.fixture.optimization_logs.items()
        }
        logs[OPTIMIZATION_LOGS["command-batch-resource-ledger"]] = b"different log\n"
        self.fixture._write_tar(self.fixture.paths["optimization_raw"], logs)
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("raw_evidence_sha256", report["errors"][0])

    def test_trusted_external_anchors_are_required(self) -> None:
        cases = {
            "expected_candidate_id": "rustinfer-0.1.0-rc2",
            "expected_revision": "f" * 40,
            "expected_source_archive_sha256": "e" * 64,
            "expected_release_image_id": "sha256:" + "d" * 64,
            "expected_reproducible_build_image_id": "sha256:" + "c" * 64,
            "expected_cuda_build_image_id": "sha256:" + "b" * 64,
            "expected_optimization_build_image_id": "sha256:" + "a" * 64,
            "expected_correctness_golden_sha256": "b" * 64,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                report = self.fixture.evaluate(**{field: value})
                self.assertFalse(report["passed"])
                self.assertIn("trusted expected", report["errors"][0])

    def test_candidate_id_is_a_closed_positive_rc_identity(self) -> None:
        for candidate_id in (
            "rustinfer-0.1.0-rc0",
            "rustinfer-00.1.0-rc1",
            "rustinfer-0.1.0-rc01",
            "release-candidate",
        ):
            with self.subTest(candidate_id=candidate_id):
                report = self.fixture.evaluate(expected_candidate_id=candidate_id)
                self.assertFalse(report["passed"])
                self.assertIn("rc<positive integer>", report["errors"][0])

    def test_candidate_id_base_must_match_release_bundle_version(self) -> None:
        self.fixture.manifest["candidate_id"] = "rustinfer-0.2.0-rc1"
        self.fixture.write_manifest()
        report = self.fixture.evaluate(
            expected_candidate_id="rustinfer-0.2.0-rc1"
        )
        self.assertFalse(report["passed"])
        self.assertIn("artifact.version", report["errors"][0])

    def test_performance_candidate_id_must_match_final_candidate(self) -> None:
        self.fixture.documents["performance"]["candidate"]["candidate_id"] = (
            "rustinfer-0.1.0-rc2"
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("trusted final candidate ID", report["errors"][0])

    def test_optimizer_profile_binary_must_match_reproducible_profile(self) -> None:
        replay = self.fixture.optimization_replay()
        replay["profile_binary_sha256"] = digest(b"substituted profile")
        report = self.fixture.evaluate(optimization_replay=replay)
        self.assertFalse(report["passed"])
        self.assertIn("profile_binary_sha256", report["errors"][0])

    def test_reproducibility_profile_binary_must_match_selected_profile(self) -> None:
        replay = self.fixture.reproducibility_replay()
        replay["artifacts"]["profile_binary_sha256"] = digest(  # type: ignore[index]
            b"substituted profile"
        )
        report = self.fixture.evaluate(reproducibility_replay=replay)
        self.assertFalse(report["passed"])
        self.assertIn("final artifact binding mismatch", report["errors"][0])

    def test_native_calibration_executable_cannot_replace_profile_binary(self) -> None:
        self.fixture.paths["profile_binary"].write_bytes(
            self.fixture.paths["native_executable"].read_bytes()
        )
        self.fixture.paths["profile_binary"].chmod(0o755)
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("final artifact binding mismatch", report["errors"][0])

    def test_performance_profile_binary_must_match_selected_profile(self) -> None:
        self.fixture.documents["performance"]["candidate"]["source"][
            "profile_binary_sha256"
        ] = digest(b"substituted profile")
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("profile_binary_sha256", report["errors"][0])

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
        self.assertIn("exact ordered inventory", report["errors"][0])

    def test_performance_raw_archive_must_be_canonical_ustar(self) -> None:
        files = {
            f"candidate-{index}.json": path.read_bytes()
            for index, path in enumerate(
                self.fixture.performance_fixture.candidate_paths, 1
            )
        }
        self.fixture.paths["performance_raw"].unlink()
        self.fixture._write_tar(self.fixture.paths["performance_raw"], files)
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("non-canonical metadata", report["errors"][0])

    def test_native_replay_rejects_self_declared_report_tamper(self) -> None:
        self.fixture.documents["native_correctness"]["cases"][0]["variants"][
            "canonical-v1"
        ]["numeric"]["first_layer_hidden"]["metrics"]["max_abs"] = 0.0
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("bytes differ from raw replay bundle", report["errors"][0])

    def test_source_archive_requires_exact_git_pax_comment(self) -> None:
        self.fixture._write_tar(
            self.fixture.paths["source"],
            {"README.md": b"exact source archive fixture"},
            pax_headers={"comment": "f" * 40},
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate(
            expected_source_archive_sha256=digest(
                self.fixture.paths["source"].read_bytes()
            )
        )
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

    def test_artifact_path_must_be_exactly_normalized(self) -> None:
        for relative in (
            ".",
            "./rustinfer",
            "nested/../rustinfer",
            "rustinfer/",
            "bad\x00name",
        ):
            with self.subTest(relative=relative):
                self.fixture.manifest["release"]["binary"]["path"] = relative
                self.fixture.write_manifest()
                report = self.fixture.evaluate()
                self.assertFalse(report["passed"])
                self.assertIn("manifest.release.binary.path", report["errors"][0])

    def test_fifo_artifact_is_rejected_without_blocking(self) -> None:
        fifo = self.fixture.root / "release-fifo"
        os.mkfifo(fifo)
        self.fixture.manifest["release"]["binary"] = {
            "path": fifo.name,
            "sha256": digest(b"fifo cannot be hashed"),
        }
        self.fixture.write_manifest()
        previous = signal.signal(
            signal.SIGALRM,
            lambda _signum, _frame: (_ for _ in ()).throw(
                AssertionError("FIFO artifact open blocked")
            ),
        )
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        try:
            report = self.fixture.evaluate()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
        self.assertFalse(report["passed"])
        self.assertIn("regular file", report["errors"][0])

    def test_evidence_root_inode_swap_is_rejected(self) -> None:
        substitute = self.fixture.root / "substitute-root"
        substitute.mkdir()
        resolved_root = self.fixture.root.resolve()
        original_open = os.open
        swapped = False

        def open_with_swapped_root(
            path: object,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal swapped
            if (
                not swapped
                and dir_fd is None
                and Path(path) == resolved_root
            ):
                swapped = True
                return original_open(substitute, flags, mode)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        with mock.patch.object(
            release_candidate_module.os,
            "open",
            side_effect=open_with_swapped_root,
        ) as patched_open, mock.patch.object(
            release_candidate_module.os,
            "supports_dir_fd",
            {patched_open},
        ):
            report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("directory changed while it was opened", report["errors"][0])

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

    def test_manifest_fd_must_match_the_checker_path(self) -> None:
        with self.fixture.manifest_path.open("rb") as manifest_file:
            report = self.fixture.evaluate(
                manifest_path=self.fixture.paths["performance"],
                manifest_fd=manifest_file.fileno(),
            )
        self.assertFalse(report["passed"])
        self.assertIn("manifest path does not name the held FD", report["errors"][0])

    def test_hard_link_artifact_alias_is_rejected(self) -> None:
        self.fixture.paths["profile_binary"].unlink()
        os.link(
            self.fixture.paths["native_executable"],
            self.fixture.paths["profile_binary"],
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("hard-link alias", report["errors"][0])

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

    def test_manifest_must_be_utf8_not_utf16_json(self) -> None:
        self.fixture.manifest_path.write_bytes(
            json.dumps(self.fixture.manifest, sort_keys=True).encode("utf-16")
        )
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("strict UTF-8 JSON", report["errors"][0])

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
        self.assertIn(b"Linux fixture 6.8.0", contents)
        raw_path.write_bytes(
            contents.replace(b"Linux fixture 6.8.0", b"Linux fixturf 6.8.0", 1)
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
        self.assertIn("trusted expected CUDA-build image ID", report["errors"][0])

    def test_reproducible_build_image_has_its_own_external_anchor(self) -> None:
        self.fixture.manifest["evidence"]["reproducible_build"][
            "build_image_id"
        ] = "sha256:" + "f" * 64
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn(
            "trusted expected reproducible-build image ID", report["errors"][0]
        )

    def test_optimization_build_image_has_its_own_external_anchor(self) -> None:
        self.fixture.manifest["evidence"]["optimization_correctness"][
            "build_image_id"
        ] = "sha256:" + "f" * 64
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn(
            "trusted expected optimization-build image ID", report["errors"][0]
        )

    def test_python_free_attestation_must_equal_raw_replay(self) -> None:
        self.fixture.documents["python_report"]["checks"].reverse()
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("differs from raw replay", report["errors"][0])

    def test_python_free_raw_source_cannot_be_self_rebound(self) -> None:
        raw_path = self.fixture.paths["python_raw"]
        raw_path.write_bytes(raw_path.read_bytes() + b"self-rebound")
        self.fixture.documents["python_report"]["raw_evidence_sha256"] = digest(
            raw_path.read_bytes()
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("differs from raw replay", report["errors"][0])

    def test_python_free_golden_is_an_external_trust_anchor(self) -> None:
        self.fixture.paths["correctness_golden"].write_bytes(
            b'{"fabricated":"golden"}\n'
        )
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("trusted expected correctness golden", report["errors"][0])

    def test_python_free_model_tree_must_match_optimizer_manifest(self) -> None:
        self.fixture.trusted_python_raw_model["model_tree_sha256"] = digest(
            b"substituted model tree"
        )
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("model_tree_sha256", report["errors"][0])

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
        report = self.fixture.evaluate(soak_replay={"report": replayed})
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
