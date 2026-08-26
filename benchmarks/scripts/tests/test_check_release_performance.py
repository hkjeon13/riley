from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import sys
import tarfile
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

PACKAGE_SCRIPT = SCRIPT.with_name("package_release_performance_evidence.py")
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "package_release_performance_evidence", PACKAGE_SCRIPT
)
assert PACKAGE_SPEC is not None and PACKAGE_SPEC.loader is not None
packager = importlib.util.module_from_spec(PACKAGE_SPEC)
sys.modules[PACKAGE_SPEC.name] = packager
PACKAGE_SPEC.loader.exec_module(packager)

FIXTURE_REQUEST_IDENTITY_SHA256 = (
    "9b01fb16a80a6be223fe574f64def29a257a27f51b7d754586b86f8153391262"
)
_ORIGINAL_REQUIRE_REQUEST_IDENTITY = checker._require_request_identity_sha256


def _require_fixture_request_identity(
    derived: object, _expected: str, path: str
) -> None:
    _ORIGINAL_REQUIRE_REQUEST_IDENTITY(
        derived, FIXTURE_REQUEST_IDENTITY_SHA256, path
    )


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
        self.supervisor_token = digest("release performance supervisor token")

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
            pair_index = run_index + 1
            capture_id = checker._runner_capture_id(
                self.supervisor_token, pair_index
            )
            run["run_id"] = checker._runner_run_id(
                "a" * 40, capture_id, pair_index
            )
            run["recorded_at_utc"] = (
                f"2026-08-26T12:00:{run_index * 10 + 2:02d}.000000000Z"
            )

        self.fixture_request_identity_sha256 = checker.native_profile._sha256_json(
            checker.native_profile._request_identity(self.profile_fixture.candidate[0])
        )
        if self.fixture_request_identity_sha256 != FIXTURE_REQUEST_IDENTITY_SHA256:
            raise AssertionError("native profile fixture request identity drifted")

        self.candidate = {
            "schema_version": "rustinfer.release-performance-candidate.v1",
            "baseline_sha256": checker.BASELINE_SHA256,
            "candidate_id": "rustinfer-0.1.0-rc1",
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
                {
                    "id": "fixed37-production-batch-e0",
                    "result": "passed",
                    "gate_id": checker.FIXED37_PRODUCTION_BATCH_GATE_ID,
                    "fixture_sha256": checker.FIXED37_GOLDEN_FIXTURE_SHA256,
                    "generated_token_ids_sha256": checker.FIXED37_GOLDEN_TOKEN_IDS_SHA256,
                    "cases": 31,
                    "compared_steps": 481,
                    "exact_window": 16,
                    "fixed_profile": "fixed-contiguous-37-balanced-v1",
                    "canonical_profile": "canonical-v1",
                    "residual_rmsnorm": "separate",
                    "execution_completion": "iteration-batch",
                    "fixed_prefill_raw_logit_mismatches": 0,
                    "fixed_cached_growing_token_id_mismatches": 0,
                    "fixed_cached_growing_cosine_min": checker.FIXED37_CACHED_GROWING_COSINE_MIN,
                    "fixed_cached_growing_max_abs_max": checker.FIXED37_CACHED_GROWING_MAX_ABS_MAX,
                    "fixed_cached_growing_mean_abs_max": checker.FIXED37_CACHED_GROWING_MEAN_ABS_MAX,
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
                    "compile_log_sha256": digest("fixed37 compile log"),
                    "test_binary_sha256": digest("fixed37 test binary"),
                    "log_sha256": digest("fixed37 exact log"),
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
        self.write_runner_receipts()

    def write_runner_receipts(self) -> None:
        self.runner_receipt_root = self.root / "runner-receipts"
        self.runner_receipt_root.mkdir(exist_ok=True)
        revision = self.profile_fixture.candidate[0]["source"]["git_commit"]
        image_id = f"sha256:{self.profile_image_digest}"
        image_environment = {"CUDA_VERSION": "12.8.1"}
        image_labels = {
            "maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>",
            "org.opencontainers.image.ref.name": "ubuntu",
            "org.opencontainers.image.version": "22.04",
        }
        overrides = {
            "RUSTINFER_PERF_SOURCE_REVISION": revision,
            "RUSTINFER_PERF_SOURCE_ARCHIVE_SHA256": self.digests["source_archive"],
            "RUSTINFER_PERF_PROFILE_BINARY_SHA256": self.digests["profile_binary"],
            "RUSTINFER_PERF_OPTIMIZER_REPORT_SHA256": self.digests["correctness_report"],
            "RUSTINFER_PERF_OPTIMIZER_IMAGE_SHA256": self.profile_image_digest,
            "RUSTINFER_PERF_MODEL_TREE_SHA256": digest("model manifest"),
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            **checker.RUNNER_PROXY_ENV,
        }
        read_only_sources = {
            "/input/source.tar": str(self.paths["source_archive"].resolve()),
            "/input/rustinfer-profile": str(self.paths["profile_binary"].resolve()),
            "/input/optimizer-correctness-report.json": str(
                self.paths["correctness_report"].resolve()
            ),
            "/model": str((self.root / "model").resolve()),
        }
        evidence_sources = [f"/evidence/run-{index}" for index in range(1, 6)]
        volume_names = [f"fixture-volume-{index}" for index in range(1, 6)]
        repository = checker.Path(checker.__file__).resolve().parents[2]
        tools = copy.deepcopy(checker.RUNNER_REVIEWED_TOOLS)
        manifest_environment = {**image_environment, **overrides}
        manifest_environment["RUSTINFER_PERF_PAIR_INDEX"] = "{pair_index}"
        manifest_environment["RUSTINFER_PERF_CAPTURE_ID"] = "{capture_id}"
        manifest = {
            "schema_version": checker.RUNNER_MANIFEST_SCHEMA,
            "candidate": {
                "source_revision": revision,
                "source_archive_sha256": self.digests["source_archive"],
                "profile_binary_sha256": self.digests["profile_binary"],
                "model_tree_sha256": digest("model manifest"),
                "optimizer_correctness_report_sha256": self.digests[
                    "correctness_report"
                ],
                "optimizer_image_id": image_id,
            },
            "runner": {
                "revision": revision,
                "host_script_sha256": hashlib.sha256(
                    (repository / "ci/run_remote_release_performance.sh").read_bytes()
                ).hexdigest(),
                "inner_script_sha256": hashlib.sha256(
                    (repository / "ci/release/run_release_performance_once.sh").read_bytes()
                ).hexdigest(),
                "tools": tools,
            },
            "container": {
                "entrypoint": checker.RUNNER_CONTAINER_ENTRYPOINT,
                "cmd": checker.RUNNER_CONTAINER_CMD,
                "environment": manifest_environment,
                "read_only_mount_sources": read_only_sources,
                "evidence_mount_sources": evidence_sources,
                "workspace_volume_names": volume_names,
                "supervisor_label": {
                    "name": checker.RUNNER_SUPERVISOR_LABEL,
                    "value": self.supervisor_token,
                },
                "labels": {
                    **image_labels,
                    checker.RUNNER_SUPERVISOR_LABEL: self.supervisor_token,
                },
            },
            "executions": [],
        }
        image_inspect = [
            {
                "Id": image_id,
                "Os": "linux",
                "Architecture": "amd64",
                "Config": {
                    "Env": ["CUDA_VERSION=12.8.1"],
                    "WorkingDir": "/workspace",
                    "Labels": image_labels,
                },
            }
        ]
        documents: dict[str, bytes] = {
            "gpu.csv": (", ".join(checker.RUNNER_GPU_ROW) + "\n").encode("utf-8"),
            "optimizer-image-inspect-before.json": checker._json_document_bytes(
                {"image": image_inspect}
            ),
            "optimizer-image-inspect-after.json": checker._json_document_bytes(
                {"image": image_inspect}
            ),
        }
        # Docker inspect is an array; avoid making the general JSON helper accept arrays.
        image_raw = (json.dumps(image_inspect, sort_keys=True, indent=2) + "\n").encode()
        documents["optimizer-image-inspect-before.json"] = image_raw
        documents["optimizer-image-inspect-after.json"] = image_raw
        preflight = {
            **checker.RUNNER_FIXED_PREFLIGHT,
            "git_revision": revision,
            "memory_used_mib": "0",
            "temperature_c": "35",
            "staging_available_bytes": str(30 * 1024**3),
        }
        preflight_raw = "".join(
            f"{name}={value}\n" for name, value in preflight.items()
        ).encode()
        for pair_index, raw_path in enumerate(self.raw_paths, 1):
            prefix = f"run-{pair_index}"
            capture_id = checker._runner_capture_id(
                self.supervisor_token, pair_index
            )
            environment = {**image_environment, **overrides}
            environment["RUSTINFER_PERF_PAIR_INDEX"] = str(pair_index)
            environment["RUSTINFER_PERF_CAPTURE_ID"] = capture_id
            mounts = [
                {
                    "Type": "bind",
                    "Source": source,
                    "Destination": destination,
                    "RW": False,
                    "Mode": "",
                    "Propagation": "rprivate",
                }
                for destination, source in read_only_sources.items()
            ]
            mounts.extend(
                [
                    {
                        "Type": "bind",
                        "Source": evidence_sources[pair_index - 1],
                        "Destination": "/evidence",
                        "RW": True,
                        "Mode": "",
                        "Propagation": "rprivate",
                    },
                    {
                        "Type": "volume",
                        "Source": volume_names[pair_index - 1],
                        "Destination": "/workspace",
                        "RW": True,
                    },
                ]
            )
            base = {
                "Id": format(pair_index, "x") * 64,
                "Image": image_id,
                "Path": checker.RUNNER_CONTAINER_ENTRYPOINT[0],
                "Args": checker.RUNNER_CONTAINER_CMD,
                "RestartCount": 0,
                "Created": f"2026-08-26T12:00:{(pair_index - 1) * 10:02d}.000000000Z",
                "Config": {
                    "Image": image_id,
                    "User": "0:0",
                    "WorkingDir": "/workspace",
                    "Entrypoint": checker.RUNNER_CONTAINER_ENTRYPOINT,
                    "Cmd": checker.RUNNER_CONTAINER_CMD,
                    "Healthcheck": {"Test": ["NONE"]},
                    "Labels": {
                        **image_labels,
                        checker.RUNNER_SUPERVISOR_LABEL: self.supervisor_token,
                    },
                    "Env": [f"{name}={value}" for name, value in environment.items()],
                },
                "HostConfig": {
                    "NetworkMode": "none",
                    "ReadonlyRootfs": True,
                    "AutoRemove": False,
                    "CapDrop": ["ALL"],
                    "CapAdd": None,
                    "SecurityOpt": ["no-new-privileges:true"],
                    "PidsLimit": 512,
                    "Privileged": False,
                    "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                    "Tmpfs": {"/tmp": "rw,nosuid,nodev,noexec,size=2147483648"},
                    "PidMode": "",
                    "IpcMode": "private",
                    "UTSMode": "",
                    "UsernsMode": "",
                    "CgroupnsMode": "private",
                    "Runtime": "runc",
                    "CpuShares": 0,
                    "Memory": 0,
                    "NanoCpus": 0,
                    "CpuPeriod": 0,
                    "CpuQuota": 0,
                    "CpusetCpus": "",
                    "CpusetMems": "",
                    "MemoryReservation": 0,
                    "MemorySwap": 0,
                    "Devices": [],
                    "DeviceCgroupRules": None,
                    "DeviceRequests": [
                        {
                            "Driver": "",
                            "Count": 0,
                            "DeviceIDs": [checker.RUNNER_GPU_ROW[1]],
                            "Capabilities": [["gpu"]],
                            "Options": {},
                        }
                    ],
                },
                "NetworkSettings": {"Networks": {}},
                "Mounts": mounts,
            }
            before = copy.deepcopy(base)
            before["State"] = {
                "Status": "created",
                "Running": False,
                "Paused": False,
                "Restarting": False,
                "OOMKilled": False,
                "Dead": False,
                "Pid": 0,
                "ExitCode": 0,
                "Error": "",
                "StartedAt": checker.RUNNER_ZERO_TIME,
                "FinishedAt": checker.RUNNER_ZERO_TIME,
            }
            after = copy.deepcopy(base)
            after["State"] = {
                "Status": "exited",
                "Running": False,
                "Paused": False,
                "Restarting": False,
                "OOMKilled": False,
                "Dead": False,
                "ExitCode": 0,
                "Error": "",
                "StartedAt": f"2026-08-26T12:00:{(pair_index - 1) * 10 + 1:02d}.000000000Z",
                "FinishedAt": f"2026-08-26T12:00:{(pair_index - 1) * 10 + 3:02d}.000000000Z",
            }
            documents[f"{prefix}/preflight.txt"] = preflight_raw
            documents[f"{prefix}/gpu-monitor.csv"] = (
                ",".join(checker.RUNNER_GPU_MONITOR_HEADER)
                + f"\n{capture_id},{base['Id']},pre_start,0,450.00,[N/A],[N/A],35,0,none"
                + f"\n{capture_id},{base['Id']},running,1,450.00,[N/A],[N/A],55,1024,container:{1000 + pair_index}"
                + f"\n{capture_id},{base['Id']},post_exit,2,450.00,[N/A],[N/A],40,0,none\n"
            ).encode()
            documents[f"{prefix}/container-inspect-before.json"] = (
                json.dumps([before], sort_keys=True, indent=2) + "\n"
            ).encode()
            documents[f"{prefix}/container-inspect-after.json"] = (
                json.dumps([after], sort_keys=True, indent=2) + "\n"
            ).encode()
            documents[f"{prefix}/candidate.json"] = raw_path.read_bytes()
            run = self.profile_fixture.candidate[pair_index - 1]
            execution = {
                "schema_version": checker.RUNNER_EXECUTION_SCHEMA,
                "pair_index": pair_index,
                "capture_id": capture_id,
                "container_id": base["Id"],
                "run_id": run["run_id"],
                "candidate_recorded_at_utc": run["recorded_at_utc"],
                "docker": {
                    "created_at_utc": base["Created"],
                    "started_at_utc": after["State"]["StartedAt"],
                    "finished_at_utc": after["State"]["FinishedAt"],
                    "exit_code": 0,
                    "oom_killed": False,
                },
                "sha256": {
                    "preflight": hashlib.sha256(preflight_raw).hexdigest(),
                    "candidate": hashlib.sha256(
                        documents[f"{prefix}/candidate.json"]
                    ).hexdigest(),
                    "gpu_monitor": hashlib.sha256(
                        documents[f"{prefix}/gpu-monitor.csv"]
                    ).hexdigest(),
                    "container_inspect_before": hashlib.sha256(
                        documents[f"{prefix}/container-inspect-before.json"]
                    ).hexdigest(),
                    "container_inspect_after": hashlib.sha256(
                        documents[f"{prefix}/container-inspect-after.json"]
                    ).hexdigest(),
                },
            }
            manifest["executions"].append(execution)
            documents[f"{prefix}/execution-receipt.json"] = (
                checker._json_document_bytes(execution)
            )
        documents["runner-manifest.json"] = checker._json_document_bytes(manifest)
        documents["SHA256SUMS"] = checker._runner_sha256s(documents)
        for name in checker.RUNNER_RECEIPT_FILES:
            path = self.runner_receipt_root.joinpath(*name.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(documents[name])

    def write(self) -> None:
        self.candidate_path.write_text(
            json.dumps(self.candidate, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def replace_runner_receipt(self, name: str, raw: bytes) -> None:
        """Replace one fixture receipt and rebuild the external checksum index."""
        path = self.runner_receipt_root.joinpath(*name.split("/"))
        path.write_bytes(raw)
        match = re.fullmatch(
            r"run-([1-5])/(preflight\.txt|candidate\.json|gpu-monitor\.csv|"
            r"container-inspect-before\.json|container-inspect-after\.json|"
            r"execution-receipt\.json)",
            name,
        )
        if match is not None:
            pair_index = int(match.group(1))
            leaf = match.group(2)
            execution_path = self.runner_receipt_root / f"run-{pair_index}" / "execution-receipt.json"
            if leaf == "execution-receipt.json":
                execution = json.loads(raw)
            else:
                execution = json.loads(execution_path.read_text(encoding="utf-8"))
                digest_field = {
                    "preflight.txt": "preflight",
                    "candidate.json": "candidate",
                    "gpu-monitor.csv": "gpu_monitor",
                    "container-inspect-before.json": "container_inspect_before",
                    "container-inspect-after.json": "container_inspect_after",
                }[leaf]
                execution["sha256"][digest_field] = hashlib.sha256(raw).hexdigest()
                execution_path.write_bytes(checker._json_document_bytes(execution))
            manifest_path = self.runner_receipt_root / "runner-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["executions"][pair_index - 1] = execution
            manifest_path.write_bytes(checker._json_document_bytes(manifest))
        payloads = {
            receipt_name: self.runner_receipt_root.joinpath(
                *receipt_name.split("/")
            ).read_bytes()
            for receipt_name in checker.RUNNER_RECEIPT_FILES
            if receipt_name != "SHA256SUMS"
        }
        (self.runner_receipt_root / "SHA256SUMS").write_bytes(
            checker._runner_sha256s(payloads)
        )

    def digest_for(self, path: Path, _label: str) -> str:
        for name, expected_path in self.paths.items():
            if path == expected_path:
                return self.digests[name]
        if path in self.raw_paths:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        raise AssertionError(f"unexpected artifact path: {path}")

    def digest_for_package(self, path: Path, label: str) -> str:
        try:
            return self.digest_for(path, label)
        except AssertionError:
            return hashlib.sha256(path.read_bytes()).hexdigest()

    def evaluate(self, *, baseline: Path = BASELINE) -> dict[str, object]:
        original_identity_check = checker._require_request_identity_sha256

        def require_fixture_identity(
            derived: object, _expected: str, path: str
        ) -> None:
            original_identity_check(
                derived, self.fixture_request_identity_sha256, path
            )

        with mock.patch.object(
            checker, "_digest_file", side_effect=self.digest_for
        ), mock.patch.object(
            checker,
            "_require_request_identity_sha256",
            side_effect=require_fixture_identity,
        ):
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
                runner_receipt_root=self.runner_receipt_root,
            )


class ReleasePerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity_patch = mock.patch.object(
            checker,
            "_require_request_identity_sha256",
            side_effect=_require_fixture_request_identity,
        )
        self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)

    def test_reviewed_baseline_digest_is_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(BASELINE.read_bytes()).hexdigest(), checker.BASELINE_SHA256
        )
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(
            baseline["measurement_binding"]["request_identity_sha256"],
            checker.PR15_REQUEST_IDENTITY_SHA256,
        )
        self.assertEqual(
            checker.PR15_REQUEST_IDENTITY_SHA256,
            "e6a99a749c41a8227574c96a1d23f8b7d877d6e75b0df4d99154db1b1921a2e6",
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            derived = checker.derive_raw_run_payloads(
                [(path.name, path.read_bytes()) for path in fixture.raw_paths]
            )
            with self.assertRaisesRegex(
                checker.ComparabilityError,
                "canonical native request identity differs",
            ):
                _ORIGINAL_REQUIRE_REQUEST_IDENTITY(
                    derived,
                    checker.PR15_REQUEST_IDENTITY_SHA256,
                    "candidate runs.request_identity_sha256",
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
            self.assertIn("runner-receipt-root", report["errors"][0])

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
            parity = next(
                row
                for row in correctness["tests"]
                if row["id"] == "smollm2-multi-step-greedy-exact"
            )
            parity["generated_token_ids"][-1] += 1
            fixture.paths["correctness_report"].write_text(
                json.dumps(correctness), encoding="utf-8"
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("generated_token_ids", report["errors"][0])

    def test_fixed37_production_batch_correctness_gate_is_closed(self) -> None:
        for mutation, expected_error in (
            ("omit", "expected exactly six checks"),
            ("fixture", "fixture_sha256"),
            ("mismatch", "fixed_prefill_raw_logit_mismatches"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = ReleaseFixture(Path(directory))
                correctness = fixture.optimization_correctness_report()
                fixed37 = next(
                    row
                    for row in correctness["tests"]
                    if row["id"] == "fixed37-production-batch-e0"
                )
                if mutation == "omit":
                    correctness["tests"].remove(fixed37)
                elif mutation == "fixture":
                    fixed37["fixture_sha256"] = "0" * 64
                else:
                    fixed37["fixed_prefill_raw_logit_mismatches"] = 1
                fixture.paths["correctness_report"].write_text(
                    json.dumps(correctness), encoding="utf-8"
                )
                report = fixture.evaluate()
                self.assertEqual(report["status"], "error")
                self.assertIn(expected_error, report["errors"][0])

    def test_runner_tool_inventory_cannot_self_authorize_an_alternate_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            manifest_path = fixture.runner_receipt_root / "runner-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runner"]["tools"]["bash"] = {
                "path": "/usr/bin/true",
                "sha256": hashlib.sha256(b"self-authorized fixture").hexdigest(),
            }
            fixture.replace_runner_receipt(
                "runner-manifest.json", checker._json_document_bytes(manifest)
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("runner-manifest.runner.tools.bash.path", report["errors"][0])

    def test_runner_receipt_requires_real_docker_normalized_security_fields(self) -> None:
        for field, value in (
            ("SecurityOpt", ["no-new-privileges"]),
            ("Tmpfs", {"/tmp": "rw,nosuid,nodev,noexec,size=2g"}),
            ("CapAdd", []),
            ("CapAdd", ["NET_ADMIN"]),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = ReleaseFixture(Path(directory))
                name = "run-1/container-inspect-before.json"
                path = fixture.runner_receipt_root.joinpath(*name.split("/"))
                document = json.loads(path.read_text(encoding="utf-8"))
                document[0]["HostConfig"][field] = value
                fixture.replace_runner_receipt(
                    name,
                    (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(),
                )
                report = fixture.evaluate()
                self.assertEqual(report["status"], "error")
                self.assertIn(f"HostConfig.{field}", report["errors"][0])

        mutations = (
            (
                "HostConfig.CapAdd",
                lambda row: row[0]["HostConfig"].pop("CapAdd"),
            ),
            (
                "Config.Labels",
                lambda row: row[0]["Config"]["Labels"].__setitem__(
                    "unreviewed", "label"
                ),
            ),
            (
                "HostConfig.DeviceRequests",
                lambda row: row[0]["HostConfig"]["DeviceRequests"][0].__setitem__(
                    "Driver", "nvidia"
                ),
            ),
            (
                "HostConfig.DeviceRequests",
                lambda row: row[0]["HostConfig"]["DeviceRequests"][0].__setitem__(
                    "Count", False
                ),
            ),
            (
                "HostConfig.DeviceRequests",
                lambda row: row[0]["HostConfig"]["DeviceRequests"][0].__setitem__(
                    "Options", {"capabilities": "all"}
                ),
            ),
        )
        for expected, mutate in mutations:
            with self.subTest(field=expected), tempfile.TemporaryDirectory() as directory:
                fixture = ReleaseFixture(Path(directory))
                name = "run-1/container-inspect-before.json"
                path = fixture.runner_receipt_root.joinpath(*name.split("/"))
                document = json.loads(path.read_text(encoding="utf-8"))
                mutate(document)
                fixture.replace_runner_receipt(
                    name,
                    (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(),
                )
                report = fixture.evaluate()
                self.assertEqual(report["status"], "error")
                self.assertIn(expected, report["errors"][0])

    def test_execution_receipt_rejects_same_revision_splices_and_time_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            name = "run-1/candidate.json"
            path = fixture.runner_receipt_root.joinpath(*name.split("/"))
            candidate = json.loads(path.read_text(encoding="utf-8"))
            candidate["run_id"] = checker._runner_run_id(
                "a" * 40, "f" * 64, 1
            )
            fixture.replace_runner_receipt(
                name, checker._json_document_bytes(candidate)
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("raw candidate identity", report["errors"][0])

        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            name = "run-2/gpu-monitor.csv"
            path = fixture.runner_receipt_root.joinpath(*name.split("/"))
            old_capture = checker._runner_capture_id(fixture.supervisor_token, 2)
            fixture.replace_runner_receipt(
                name,
                path.read_bytes().replace(old_capture.encode(), b"f" * 64),
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("capture_id", report["errors"][0])

        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            inspect_name = "run-1/container-inspect-after.json"
            inspect_path = fixture.runner_receipt_root.joinpath(
                *inspect_name.split("/")
            )
            after = json.loads(inspect_path.read_text(encoding="utf-8"))
            rewritten_start = "2026-08-26T12:00:02.500000000Z"
            after[0]["State"]["StartedAt"] = rewritten_start
            fixture.replace_runner_receipt(
                inspect_name,
                (json.dumps(after, sort_keys=True, indent=2) + "\n").encode(),
            )
            execution_name = "run-1/execution-receipt.json"
            execution_path = fixture.runner_receipt_root.joinpath(
                *execution_name.split("/")
            )
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["docker"]["started_at_utc"] = rewritten_start
            fixture.replace_runner_receipt(
                execution_name, checker._json_document_bytes(execution)
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("candidate recorded_at", report["errors"][0])

    def test_runner_environment_rejects_loader_git_and_imported_bash_controls(self) -> None:
        for name in (
            "LD_AUDIT",
            "GIT_EXEC_PATH",
            "GIT_CONFIG_PARAMETERS",
            "BASH_FUNC_injected%%",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(checker.InputError, "forbidden"):
                    checker._runner_environment(
                        ["CUDA_VERSION=12.8.1", f"{name}=injected"],
                        "fixture environment",
                    )

    def test_runner_model_tree_must_match_submitted_optimizer_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            alternate = "9" * 64
            manifest_path = fixture.runner_receipt_root / "runner-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["candidate"]["model_tree_sha256"] = alternate
            manifest["container"]["environment"][
                "RUSTINFER_PERF_MODEL_TREE_SHA256"
            ] = alternate
            fixture.replace_runner_receipt(
                "runner-manifest.json", checker._json_document_bytes(manifest)
            )
            for pair_index in range(1, 6):
                for stage in ("before", "after"):
                    name = f"run-{pair_index}/container-inspect-{stage}.json"
                    path = fixture.runner_receipt_root.joinpath(*name.split("/"))
                    document = json.loads(path.read_text(encoding="utf-8"))
                    environment = document[0]["Config"]["Env"]
                    environment[:] = [
                        (
                            f"RUSTINFER_PERF_MODEL_TREE_SHA256={alternate}"
                            if value.startswith("RUSTINFER_PERF_MODEL_TREE_SHA256=")
                            else value
                        )
                        for value in environment
                    ]
                    fixture.replace_runner_receipt(
                        name,
                        (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(),
                    )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn(
                "submitted optimizer correctness model manifest", report["errors"][0]
            )

    def test_runner_monitor_self_rehash_cannot_hide_foreign_cuda_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            name = "run-2/gpu-monitor.csv"
            path = fixture.runner_receipt_root.joinpath(*name.split("/"))
            fixture.replace_runner_receipt(
                name,
                path.read_bytes().replace(b"container:1002", b"foreign:1002"),
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("foreign process receipt", report["errors"][0])

    def test_runner_bind_mode_and_propagation_are_exact_receipt_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            name = "run-1/container-inspect-before.json"
            path = fixture.runner_receipt_root.joinpath(*name.split("/"))
            document = json.loads(path.read_text(encoding="utf-8"))
            model_mount = next(
                mount
                for mount in document[0]["Mounts"]
                if mount["Destination"] == "/model"
            )
            model_mount["Mode"] = "ro"
            model_mount["Propagation"] = "shared"
            fixture.replace_runner_receipt(
                name,
                (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(),
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("exact read-only bind", report["errors"][0])

    def test_raw_run_reader_rejects_symlink_and_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "candidate-1.json"
            target.write_bytes(b"{}")
            symlink = root / "candidate-link.json"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(checker.InputError, "stable snapshot"):
                checker._read_raw_run_paths([symlink])

            fifo = root / "candidate.fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(checker.InputError, "regular file"):
                checker._read_raw_run_paths([fifo])

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
                "--runner-receipt-root", str(fixture.runner_receipt_root),
                "--report", str(report_path),
            ]
            stderr = io.StringIO()
            with mock.patch.object(checker, "_digest_file", side_effect=fixture.digest_for):
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    exit_code = checker.main(argv)
            self.assertEqual(exit_code, 2)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "occupied")
            self.assertIn("refusing to overwrite", stderr.getvalue())


class ReleasePerformancePackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity_patch = mock.patch.object(
            checker,
            "_require_request_identity_sha256",
            side_effect=_require_fixture_request_identity,
        )
        self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)

    def package(
        self,
        fixture: ReleaseFixture,
        output: Path,
        *,
        candidate_id: str = "rustinfer-0.1.0-rc1",
    ) -> dict[str, object]:
        with mock.patch.object(
            checker, "_digest_file", side_effect=fixture.digest_for_package
        ):
            return checker.package_release_performance_evidence(
                BASELINE,
                output,
                candidate_id=candidate_id,
                recorded_at_utc="2026-08-26T12:34:56Z",
                source_archive=fixture.paths["source_archive"],
                profile_binary=fixture.paths["profile_binary"],
                release_binary=fixture.paths["release_binary"],
                weights=fixture.paths["weights"],
                tokenizer=fixture.paths["tokenizer"],
                correctness_report=fixture.paths["correctness_report"],
                profile_image_id=f"sha256:{fixture.profile_image_digest}",
                release_image_id=f"sha256:{fixture.release_image_digest}",
                run_paths=fixture.raw_paths,
                runner_receipt_root=fixture.runner_receipt_root,
            )

    @staticmethod
    def cli_argv(
        fixture: ReleaseFixture,
        output: Path,
        *,
        candidate_id: str = "rustinfer-0.1.0-rc1",
    ) -> list[str]:
        return [
            "--baseline", str(BASELINE),
            "--candidate-id", candidate_id,
            "--recorded-at-utc", "2026-08-26T12:34:56Z",
            "--source-archive", str(fixture.paths["source_archive"]),
            "--profile-binary", str(fixture.paths["profile_binary"]),
            "--release-binary", str(fixture.paths["release_binary"]),
            "--weights", str(fixture.paths["weights"]),
            "--tokenizer", str(fixture.paths["tokenizer"]),
            "--correctness-report", str(fixture.paths["correctness_report"]),
            "--profile-image-id", f"sha256:{fixture.profile_image_digest}",
            "--release-image-id", f"sha256:{fixture.release_image_digest}",
            "--run", *(str(path) for path in fixture.raw_paths),
            "--runner-receipt-root", str(fixture.runner_receipt_root),
            "--output-directory", str(output),
        ]

    def test_raw_derivation_does_not_require_a_candidate_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            payloads = [
                (str(path), path.read_bytes())
                for path in reversed(fixture.raw_paths)
            ]
            derived = checker.derive_raw_run_payloads(payloads)
            self.assertEqual(
                [name for name, _ in derived["payloads"]],
                list(checker.RAW_EVIDENCE_FILES),
            )
            self.assertEqual(derived["model"], fixture.candidate["model"])
            self.assertEqual(
                derived["environment"], fixture.candidate["environment"]
            )
            self.assertEqual(derived["workload"], fixture.candidate["workload"])
            self.assertEqual(
                derived["run_summary"], fixture.candidate["run_summary"]
            )
            self.assertEqual(derived["metrics"], fixture.candidate["metrics"])
            self.assertEqual(derived["raw_runs"], fixture.candidate["raw_runs"])

    def test_raw_archive_is_deterministic_canonical_ustar_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            payloads = [
                (str(path), path.read_bytes())
                for path in reversed(fixture.raw_paths)
            ]
            first = root / "first.tar"
            second = root / "second.tar"
            first_sha = checker.write_raw_evidence_archive(
                first, payloads, runner_receipt_root=fixture.runner_receipt_root
            )
            second_sha = checker.write_raw_evidence_archive(
                second, payloads, runner_receipt_root=fixture.runner_receipt_root
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(second.stat().st_mode), 0o644)
            self.assertEqual(first_sha, second_sha)
            self.assertEqual(first_sha, hashlib.sha256(first.read_bytes()).hexdigest())
            with tarfile.open(first, "r:") as archive:
                members = archive.getmembers()
                self.assertEqual(
                    checker.RUNNER_MANIFEST_SCHEMA,
                    "rustinfer.release-performance-runner-manifest.v3",
                )
                self.assertEqual(
                    [
                        name
                        for name in checker.RUNNER_RECEIPT_FILES
                        if name.endswith("/execution-receipt.json")
                    ],
                    [f"run-{index}/execution-receipt.json" for index in range(1, 6)],
                )
                self.assertEqual(
                    [member.name for member in members],
                    list(checker.RUNNER_RECEIPT_FILES),
                )
                for member in members:
                    self.assertEqual(member.mode, 0o644)
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(member.gid, 0)
                    self.assertEqual(member.uname, "")
                    self.assertEqual(member.gname, "")
                    self.assertEqual(member.mtime, 0)
            replay = checker.replay_raw_evidence_archive(first)
            self.assertEqual(replay["archive_sha256"], first_sha)
            self.assertEqual(
                replay["derived"]["raw_runs"], fixture.candidate["raw_runs"]
            )
            self.assertEqual(
                replay["runner_manifest"]["candidate"]["model_tree_sha256"],
                digest("model manifest"),
            )
            self.assertEqual(
                replay["runner_manifest"]["runner"]["tools"],
                checker.RUNNER_REVIEWED_TOOLS,
            )

    def test_v3_receipts_reject_missing_legacy_and_self_rehashed_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            missing = fixture.runner_receipt_root / "run-1/preflight.txt"
            saved = missing.read_bytes()
            missing.unlink()
            with self.assertRaisesRegex(checker.InputError, "run-1/preflight"):
                checker.load_runner_receipt_root(fixture.runner_receipt_root)
            missing.write_bytes(saved)

            fixture.replace_runner_receipt(
                "run-1/preflight.txt",
                saved.replace(b"power_limit_w=450.00", b"power_limit_w=451.00"),
            )
            with self.assertRaisesRegex(checker.InputError, "power_limit_w"):
                checker.load_runner_receipt_root(fixture.runner_receipt_root)

            legacy = root / "legacy-five-json.tar"
            candidates = checker.derive_raw_run_payloads(
                [(str(path), path.read_bytes()) for path in fixture.raw_paths]
            )["payloads"]
            with tarfile.open(legacy, "w:", format=tarfile.USTAR_FORMAT) as archive:
                for name, raw in candidates:
                    archive.addfile(
                        checker._canonical_tar_info(name, len(raw)), io.BytesIO(raw)
                    )
            with self.assertRaisesRegex(checker.InputError, "exact ordered inventory"):
                checker.load_raw_evidence_archive(legacy)

    def test_raw_archive_rejects_noncanonical_metadata_compression_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            payloads = checker.derive_raw_run_payloads(
                [(str(path), path.read_bytes()) for path in fixture.raw_paths]
            )["payloads"]
            archive_payloads = checker.load_runner_receipt_root(
                fixture.runner_receipt_root
            )["archive_payloads"]

            wrong_mode = root / "wrong-mode.tar"
            with tarfile.open(
                wrong_mode, "w:", format=tarfile.USTAR_FORMAT
            ) as archive:
                for index, (name, raw) in enumerate(archive_payloads):
                    info = checker._canonical_tar_info(name, len(raw))
                    if index == 0:
                        info.mode = 0o600
                    archive.addfile(info, io.BytesIO(raw))
            with self.assertRaisesRegex(checker.InputError, "non-canonical metadata"):
                checker.load_raw_evidence_archive(wrong_mode)

            compressed = root / "compressed.tar.gz"
            with tarfile.open(
                compressed, "w:gz", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name, raw in archive_payloads:
                    archive.addfile(
                        checker._canonical_tar_info(name, len(raw)),
                        io.BytesIO(raw),
                    )
            with self.assertRaisesRegex(checker.InputError, "uncompressed USTAR"):
                checker.load_raw_evidence_archive(compressed)

            tailed = root / "tailed.tar"
            checker.write_raw_evidence_archive(
                tailed, payloads, runner_receipt_root=fixture.runner_receipt_root
            )
            tailed.write_bytes(tailed.read_bytes() + b"unexpected trailing bytes")
            with self.assertRaisesRegex(checker.InputError, "canonical deterministic"):
                checker.load_raw_evidence_archive(tailed)

    def test_raw_archive_writer_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "existing.tar"
            output.write_bytes(b"owner data")
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            with self.assertRaises(FileExistsError):
                checker.write_raw_evidence_archive(
                    output, payloads, runner_receipt_root=fixture.runner_receipt_root
                )
            self.assertEqual(output.read_bytes(), b"owner data")

    def test_raw_archive_writer_loses_publish_race_without_unlinking_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "raced.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]

            def competing_publish(_source: Path, target: Path) -> None:
                self.assertEqual(target, output)
                target.write_bytes(b"owner data created by the winner")
                raise FileExistsError(target)

            with mock.patch.object(
                checker, "_rename_noreplace", side_effect=competing_publish
            ):
                with self.assertRaises(FileExistsError):
                    checker.write_raw_evidence_archive(
                        output, payloads, runner_receipt_root=fixture.runner_receipt_root
                    )
            self.assertEqual(output.read_bytes(), b"owner data created by the winner")
            residues = list(root.glob(f".{output.name}.staging-*"))
            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_file())
            self.assertEqual(stat.S_IMODE(residues[0].stat().st_mode), 0o644)
            replay = checker.replay_raw_evidence_archive(residues[0])
            self.assertEqual(
                replay["derived"]["raw_runs"], fixture.candidate["raw_runs"]
            )

    def test_raw_archive_writer_does_not_unlink_replaced_published_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "published.tar"
            displaced = root / "displaced-generated.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            fsync_directory = checker._fsync_directory

            def replace_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    output.rename(displaced)
                    output.write_bytes(b"owner replacement")

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=replace_after_publish
            ):
                with self.assertRaisesRegex(checker.InputError, "held inode"):
                    checker.write_raw_evidence_archive(
                        output, payloads, runner_receipt_root=fixture.runner_receipt_root
                    )
            self.assertEqual(output.read_bytes(), b"owner replacement")
            self.assertTrue(displaced.is_file())
            self.assertEqual(list(root.glob(f".{output.name}.staging-*")), [])

    def test_raw_archive_writer_detects_same_inode_tamper_after_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "tampered-after-publish.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            rename_noreplace = checker._rename_noreplace

            def tamper_after_publish(source: Path, target: Path) -> None:
                rename_noreplace(source, target)
                if target == output:
                    target.write_bytes(b"same inode replacement bytes")

            with mock.patch.object(
                checker, "_rename_noreplace", side_effect=tamper_after_publish
            ):
                with self.assertRaisesRegex(checker.InputError, "digest changed"):
                    checker.write_raw_evidence_archive(
                        output, payloads, runner_receipt_root=fixture.runner_receipt_root
                    )
            self.assertEqual(output.read_bytes(), b"same inode replacement bytes")

    def test_raw_archive_writer_preserves_complete_publish_on_parent_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "fsync-failed.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            fsync_directory = checker._fsync_directory

            def fail_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    raise OSError("injected parent fsync failure")

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=fail_after_publish
            ):
                with self.assertRaisesRegex(OSError, "parent fsync failure"):
                    checker.write_raw_evidence_archive(
                        output, payloads, runner_receipt_root=fixture.runner_receipt_root
                    )
            replay = checker.replay_raw_evidence_archive(output)
            self.assertEqual(
                replay["derived"]["raw_runs"], fixture.candidate["raw_runs"]
            )

    def test_raw_archive_replay_digest_and_payload_share_one_open_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            archive = root / "evidence.tar"
            moved = root / "opened-snapshot.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            expected_sha = checker.write_raw_evidence_archive(
                archive, payloads, runner_receipt_root=fixture.runner_receipt_root
            )
            stable_snapshot = checker._stable_fd_snapshot
            replaced = False

            def replace_after_open(
                descriptor: int, label: str, maximum: int
            ) -> tuple[bytes, str, os.stat_result]:
                nonlocal replaced
                if not replaced:
                    archive.rename(moved)
                    archive.write_bytes(b"replacement path contents")
                    replaced = True
                return stable_snapshot(descriptor, label, maximum)

            with mock.patch.object(
                checker, "_stable_fd_snapshot", side_effect=replace_after_open
            ):
                replay = checker.replay_raw_evidence_archive(archive)
            self.assertEqual(replay["archive_sha256"], expected_sha)
            self.assertEqual(
                replay["derived"]["raw_runs"], fixture.candidate["raw_runs"]
            )
            self.assertEqual(archive.read_bytes(), b"replacement path contents")
            self.assertEqual(hashlib.sha256(moved.read_bytes()).hexdigest(), expected_sha)

    def test_raw_archive_writer_preserves_only_private_failed_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "failed.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            partial_bytes = b"partial archive bytes"

            def write_partial_then_fail(handle: object, _payloads: object) -> None:
                handle.write(partial_bytes)  # type: ignore[attr-defined]
                raise OSError("injected archive write failure")

            with mock.patch.object(
                checker,
                "_write_canonical_raw_archive",
                side_effect=write_partial_then_fail,
            ):
                with self.assertRaisesRegex(OSError, "injected archive write failure"):
                    checker.write_raw_evidence_archive(
                        output, payloads, runner_receipt_root=fixture.runner_receipt_root
                    )
            self.assertFalse(output.exists())
            residues = list(root.glob(f".{output.name}.staging-*"))
            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_file())
            self.assertEqual(residues[0].read_bytes(), partial_bytes)
            self.assertEqual(stat.S_IMODE(residues[0].stat().st_mode), 0o600)

    def test_packager_emits_only_candidate_report_and_raw_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            first = root / "package-one"
            second = root / "package-two"
            result = self.package(fixture, first)
            self.package(fixture, second)
            self.assertTrue(result["report"]["passed"], result)
            expected_names = {
                checker.PACKAGE_CANDIDATE_NAME,
                checker.PACKAGE_REPORT_NAME,
                checker.PACKAGE_RAW_EVIDENCE_NAME,
            }
            self.assertEqual({path.name for path in first.iterdir()}, expected_names)
            for name in expected_names:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertEqual(stat.S_IMODE((first / name).stat().st_mode), 0o644)
            candidate = json.loads(
                (first / checker.PACKAGE_CANDIDATE_NAME).read_text(encoding="utf-8")
            )
            report = json.loads(
                (first / checker.PACKAGE_REPORT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(candidate["raw_runs"], fixture.candidate["raw_runs"])
            self.assertEqual(report, result["report"])
            replay = checker.replay_raw_evidence_archive(
                first / checker.PACKAGE_RAW_EVIDENCE_NAME
            )
            checker.validate_raw_run_payloads(
                replay["payloads"], checker._validate_candidate(candidate)
            )

    def test_packager_is_create_only_and_preserves_structurally_failed_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            existing = root / "existing"
            existing.mkdir()
            owner = existing / "owner.txt"
            owner.write_text("owner data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.package(fixture, existing)
            self.assertEqual(owner.read_text(encoding="utf-8"), "owner data")

            correctness = fixture.optimization_correctness_report()
            correctness["gate_id"] = "invalid-gate"
            fixture.paths["correctness_report"].write_text(
                json.dumps(correctness), encoding="utf-8"
            )
            invalid = root / "invalid"
            with self.assertRaisesRegex(checker.InputError, "structurally invalid"):
                self.package(fixture, invalid)
            self.assertFalse(invalid.exists())
            residues = list(root.glob(f".{invalid.name}.staging-*"))
            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_dir())
            self.assertEqual(
                {path.name for path in residues[0].iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )

    def test_packager_preserves_comparable_threshold_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            for run in fixture.profile_fixture.candidate:
                for request in run["requests"]:
                    request["ttft_ms"] *= 1.2
            fixture.refresh()
            output = root / "regression-evidence"
            result = self.package(fixture, output)
            self.assertFalse(result["report"]["passed"])
            self.assertEqual(result["report"]["status"], "failed")
            self.assertEqual(result["report"]["errors"], [])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )

    def test_packager_loses_atomic_publish_race_without_touching_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "raced-package"
            original_rename = checker._rename_noreplace

            def competing_publish(source: Path, target: Path) -> None:
                if target == output:
                    target.mkdir()
                    (target / "owner.txt").write_text("owner data", encoding="utf-8")
                    raise FileExistsError(target)
                original_rename(source, target)

            with mock.patch.object(
                checker, "_rename_noreplace", side_effect=competing_publish
            ):
                with self.assertRaises(FileExistsError):
                    self.package(fixture, output)
            self.assertEqual(
                (output / "owner.txt").read_text(encoding="utf-8"), "owner data"
            )
            residues = list(root.glob(f".{output.name}.staging-*"))
            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_dir())
            self.assertEqual(
                {path.name for path in residues[0].iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )

    def test_packager_cleanup_skips_replaced_post_publish_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "published-package"
            displaced = root / "displaced-package"
            fsync_directory = checker._fsync_directory

            def replace_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    output.rename(displaced)
                    output.mkdir()
                    (output / "owner.txt").write_text(
                        "owner replacement", encoding="utf-8"
                    )

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=replace_after_publish
            ):
                with self.assertRaisesRegex(checker.InputError, "changed"):
                    self.package(fixture, output)
            self.assertEqual(
                (output / "owner.txt").read_text(encoding="utf-8"),
                "owner replacement",
            )
            self.assertEqual(
                {path.name for path in displaced.iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )

    def test_packager_preserves_triplet_on_parent_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "fsync-failed-package"
            fsync_directory = checker._fsync_directory

            def fail_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    raise OSError("injected parent fsync failure")

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=fail_after_publish
            ):
                with self.assertRaisesRegex(OSError, "parent fsync failure"):
                    self.package(fixture, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )
            checker.replay_raw_evidence_archive(
                output / checker.PACKAGE_RAW_EVIDENCE_NAME
            )
            candidate = json.loads(
                (output / checker.PACKAGE_CANDIDATE_NAME).read_text(
                    encoding="utf-8"
                )
            )
            report = json.loads(
                (output / checker.PACKAGE_REPORT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(candidate["raw_runs"], fixture.candidate["raw_runs"])
            self.assertTrue(report["passed"], report)

    def test_packager_never_deletes_post_publish_child_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "owner-replaced-package"
            owner_bytes = b"owner replacement after directory publish"
            fsync_directory = checker._fsync_directory

            def replace_child_and_fail(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    candidate = output / checker.PACKAGE_CANDIDATE_NAME
                    candidate.unlink()
                    candidate.write_bytes(owner_bytes)
                    raise OSError("injected post-publish failure")

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=replace_child_and_fail
            ):
                with self.assertRaisesRegex(OSError, "post-publish failure"):
                    self.package(fixture, output)
            self.assertEqual(
                (output / checker.PACKAGE_CANDIDATE_NAME).read_bytes(), owner_bytes
            )
            self.assertTrue((output / checker.PACKAGE_REPORT_NAME).is_file())
            self.assertTrue((output / checker.PACKAGE_RAW_EVIDENCE_NAME).is_file())

    def test_packager_revalidates_every_child_after_atomic_publish(self) -> None:
        for child_name in (
            checker.PACKAGE_CANDIDATE_NAME,
            checker.PACKAGE_REPORT_NAME,
            checker.PACKAGE_RAW_EVIDENCE_NAME,
        ):
            with self.subTest(child_name=child_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    fixture = ReleaseFixture(root)
                    output = root / "tampered-package"
                    rename_noreplace = checker._rename_noreplace

                    def tamper_then_publish(source: Path, target: Path) -> None:
                        if target == output:
                            (source / child_name).write_bytes(b"tampered child bytes")
                        rename_noreplace(source, target)

                    with mock.patch.object(
                        checker,
                        "_rename_noreplace",
                        side_effect=tamper_then_publish,
                    ):
                        with self.assertRaisesRegex(
                            checker.InputError, "changed"
                        ):
                            self.package(fixture, output)
                    self.assertEqual(
                        (output / child_name).read_bytes(), b"tampered child bytes"
                    )

    def test_packager_rejects_post_check_extra_inventory_without_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "extra-child-package"
            rename_noreplace = checker._rename_noreplace

            def add_extra_then_publish(source: Path, target: Path) -> None:
                if target == output:
                    (source / "unexpected-owner-file").write_bytes(b"owner data")
                rename_noreplace(source, target)

            with mock.patch.object(
                checker, "_rename_noreplace", side_effect=add_extra_then_publish
            ):
                with self.assertRaisesRegex(
                    checker.InputError, "exact three-file inventory"
                ):
                    self.package(fixture, output)
            self.assertEqual(
                (output / "unexpected-owner-file").read_bytes(), b"owner data"
            )

    def test_invalid_candidate_id_fails_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            for index, candidate_id in enumerate(
                (
                    "pr16-candidate",
                    "rustinfer-00.1.0-rc1",
                    "rustinfer-0.01.0-rc1",
                    "rustinfer-0.1.00-rc1",
                    "rustinfer-0.1.0-rc01",
                )
            ):
                with self.subTest(candidate_id=candidate_id):
                    output = root / f"invalid-id-{index}"
                    with mock.patch.object(
                        checker,
                        "_read_raw_run_paths",
                        side_effect=AssertionError("raw inputs were read"),
                    ):
                        with self.assertRaisesRegex(
                            checker.InputError, "candidate_id"
                        ):
                            self.package(
                                fixture, output, candidate_id=candidate_id
                            )
                    self.assertFalse(output.exists())


    def test_packager_cli_uses_the_same_fail_closed_producer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "cli-package"
            argv = self.cli_argv(fixture, output)
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ):
                self.assertEqual(packager.main(argv), 0)
            self.assertTrue(
                json.loads(
                    (output / checker.PACKAGE_REPORT_NAME).read_text(
                        encoding="utf-8"
                    )
                )["passed"]
            )

    def test_packager_cli_returns_one_and_publishes_threshold_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            for run in fixture.profile_fixture.candidate:
                for request in run["requests"]:
                    request["ttft_ms"] *= 1.2
            fixture.refresh()
            output = root / "cli-regression"
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ):
                self.assertEqual(packager.main(self.cli_argv(fixture, output)), 1)
            report = json.loads(
                (output / checker.PACKAGE_REPORT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["passed"])

    def test_packager_cli_returns_two_without_overwriting_or_invalid_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)

            invalid = root / "invalid-cli"
            stderr = io.StringIO()
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(
                        packager.main(
                            self.cli_argv(
                                fixture, invalid, candidate_id="not-a-release-candidate"
                            )
                        ),
                        2,
                    )
            self.assertFalse(invalid.exists())
            self.assertIn("candidate_id", stderr.getvalue())

            existing = root / "existing-cli"
            existing.mkdir()
            owner = existing / "owner.txt"
            owner.write_text("owner data", encoding="utf-8")
            stderr = io.StringIO()
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(
                        packager.main(self.cli_argv(fixture, existing)), 2
                    )
            self.assertEqual(owner.read_text(encoding="utf-8"), "owner data")
            self.assertIn("refusing to overwrite", stderr.getvalue())

    def test_packager_cli_returns_two_but_preserves_post_publish_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "cli-post-publish-failure"
            fsync_directory = checker._fsync_directory

            def fail_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    raise OSError("injected CLI parent fsync failure")

            stderr = io.StringIO()
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ), mock.patch.object(
                checker, "_fsync_directory", side_effect=fail_after_publish
            ):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(
                        packager.main(self.cli_argv(fixture, output)), 2
                    )
            self.assertIn("parent fsync failure", stderr.getvalue())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )


if __name__ == "__main__":
    unittest.main()
