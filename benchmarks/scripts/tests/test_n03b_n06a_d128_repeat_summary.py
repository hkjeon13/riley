from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import n03a_paged_d128_repeat_summary as n03a
import n03b_n06a_d128_repeat_summary as summary


VLLM_IMAGE_DIGEST = "sha256:" + "a" * 64


def psi(value: float) -> dict[str, object]:
    return {
        "status": "ok",
        "some": {
            "avg10": value,
            "avg60": value + 0.1,
            "avg300": value + 0.2,
            "total": int(value * 100),
        },
        "full": {
            "avg10": value / 2.0,
            "avg60": value / 2.0,
            "avg300": value / 2.0,
            "total": int(value * 10),
        },
        "error": None,
    }


def observation(value: float) -> dict[str, object]:
    return {
        "psi": {
            "cpu": psi(value),
            "io": psi(value + 10.0),
            "memory": psi(value + 20.0),
        }
    }


def n03a_record(case: n03a.ExpectedCase, *, offset: float) -> str:
    native_median = 0.200000 + offset + case.logical_tokens / 1_000_000.0
    reference_median = native_median * 1.6
    native_p95 = native_median * 1.1
    reference_p95 = reference_median * 1.1
    return (
        "riley-cuda-n03a-paged-gqa "
        f"schema_version=2 case={case.label} logical_tokens={case.logical_tokens} "
        "batch=1 query_heads=16 key_value_heads=2 head_size=128 page_size=16 "
        "fixture=patterned shuffled_page_ids=true partial_last_page=false "
        "timing_scope=prepared_paged_decode_execute_cuda_event "
        "internal_warmups_per_backend=8 paired_rounds=24 paired_order=ABBA "
        f"native_median_ms={native_median:.6f} native_p95_ms={native_p95:.6f} "
        f"reference_median_ms={reference_median:.6f} reference_p95_ms={reference_p95:.6f} "
        f"paired_speedup_ratio={reference_median / native_median:.6f} "
        f"paired_delta_ms={reference_median - native_median:.6f} "
        "native_workspace_bytes=270400 native_v2_state_prefix_bytes=266240 "
        "native_v2_transition_step_bytes=4096 native_v2_normalizer_bytes=64 "
        "reference_workspace_bytes=266240 "
        f"implementation_id={summary.OPERATOR_IMPLEMENTATION_ID} "
        "implementation_version=2 graph_capture_supported=false operator_parity=passed "
        "allocation_delta=0 python_free=true full_model_serving=false "
        "vllm_comparison=false status=passed"
    )


class N03bN06aD128RepeatSummaryTests(unittest.TestCase):
    def replace_marker_field(
        self, stdout_path: Path, *, lane: str, field: str, value: str
    ) -> None:
        lines: list[str] = []
        changed = False
        pattern = re.compile(rf"(?<!\S){re.escape(field)}=[^\s]+")
        for line in stdout_path.read_text(encoding="utf-8").splitlines():
            if f"lane={lane}" in line:
                line, replacements = pattern.subn(f"{field}={value}", line, count=1)
                self.assertEqual(replacements, 1)
                changed = True
            lines.append(line)
        self.assertTrue(changed)
        stdout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def make_startup_snapshot(self, directory: Path) -> Path:
        path = directory / "riley.startup.stderr.log"
        path.write_text(
            "RILEY_DECODE_ATTENTION "
            "requested_backend=native-bf16-paged-split-gqa-d128-two-stage "
            f"resolved_ragged_backend={summary.SERVING_IMPLEMENTATION_ID} "
            "fallback_reason=none query_heads=16 key_value_heads=2 head_size=128 "
            "page_size=16 graph=false\n",
            encoding="utf-8",
        )
        return path

    def make_model_identity_artifacts(self, attempt: Path, directory: Path) -> dict[str, str]:
        """Create a tiny regular-file model plus the same bounded receipts as N06."""
        model = directory / "fixture-qwen-model"
        model.mkdir(exist_ok=True)
        metadata = {
            "config.json": b'{"architectures":["Qwen2ForCausalLM"]}\n',
            "tokenizer_config.json": b'{"model_max_length":32768}\n',
        }
        shards = {
            "model-00001-of-00002.safetensors": b"fixture first shard",
            "model-00002-of-00002.safetensors": b"fixture second shard",
        }
        for relative_path, payload in {**metadata, **shards}.items():
            path = model / relative_path
            if not path.exists():
                path.write_bytes(payload)
        manifest_document = {
            "schema_version": summary.MODEL_IDENTITY_MANIFEST_SCHEMA_VERSION,
            "model_id": summary.QWEN_MODEL_ID,
            "model_revision": summary.QWEN_MODEL_REVISION,
            "model_path": str(model.resolve()),
            "metadata_files": [
                {
                    "path": relative_path,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for relative_path, payload in metadata.items()
            ],
            "shards": [
                {
                    "path": relative_path,
                    "size_bytes": len(payload),
                    "lfs_oid_sha256": hashlib.sha256(("lfs:" + relative_path).encode("utf-8")).hexdigest(),
                }
                for relative_path, payload in shards.items()
            ],
        }
        manifest_path = attempt / "model.identity.manifest.json"
        manifest_path.write_text(json.dumps(manifest_document, sort_keys=True) + "\n", encoding="utf-8")
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        git_path = "/usr/bin/git"
        git_sha256 = "b" * 64
        oids = {entry["path"]: entry["lfs_oid_sha256"] for entry in manifest_document["shards"]}

        def command(argv: list[str], stdout: str, *, started_ns: int) -> dict[str, object]:
            return {
                "argv": argv,
                "started_ns": started_ns,
                "finished_ns": started_ns + 1,
                "timeout_seconds": summary.MODEL_IDENTITY_GIT_TIMEOUT_SECONDS,
                "returncode": 0,
                "stdout_text": stdout,
                "stdout_bytes": len(stdout.encode("utf-8")),
                "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                "stderr_text": "",
                "stderr_bytes": 0,
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            }

        lfs_stdout = "".join(f"{oids[path]} * {path}\n" for path in sorted(oids))
        validation_document = {
            "schema_version": summary.MODEL_IDENTITY_VALIDATION_SCHEMA_VERSION,
            "manifest_sha256": manifest_sha256,
            "model_path": str(model.resolve()),
            "model_revision": summary.QWEN_MODEL_REVISION,
            "git": {"path": git_path, "sha256": git_sha256},
            "commands": [
                command(
                    [git_path, "-C", str(model.resolve()), "rev-parse", "HEAD"],
                    summary.QWEN_MODEL_REVISION + "\n",
                    started_ns=10,
                ),
                command(
                    [git_path, "-C", str(model.resolve()), "lfs", "ls-files", "-l"],
                    lfs_stdout,
                    started_ns=12,
                ),
            ],
            "observed_git_head": summary.QWEN_MODEL_REVISION,
            "observed_lfs_oids": {path: oids[path] for path in sorted(oids)},
            "validated": True,
            "errors": [],
        }
        validation_path = attempt / "model.identity.validation.json"
        validation_path.write_text(json.dumps(validation_document, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "model_path": str(model.resolve()),
            "model_identity_manifest_path": str(manifest_path.resolve()),
            "model_identity_manifest_sha256": manifest_sha256,
            "model_identity_validation_path": str(validation_path.resolve()),
            "model_identity_validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
            "model_identity_git_path": git_path,
            "model_identity_git_sha256": git_sha256,
        }

    def make_attempt_directory(self, directory: Path, *, index: int, pair_order: str) -> Path:
        attempt = directory / "paired-driver" / f"timed-{index:03d}"
        attempt.mkdir(parents=True)
        prompt_token_ids = list(range(2_048))
        output_token_ids = list(range(10_000, 10_128))
        workload_document = {
            "schema_version": "riley.n06a-d128-serving-workload.v1",
            "case": "general-c32-2048-plus-128",
            "model_id": summary.QWEN_MODEL_ID,
            "model_revision": summary.QWEN_MODEL_REVISION,
            "prompt": "fixture prompt",
            "prompt_token_ids": prompt_token_ids,
            "output_token_ids": output_token_ids,
            "output_text": "fixture-output",
            "finish_reason": "length",
            "sampling": {"temperature": 0.0, "top_p": 1.0},
        }
        workload_path = attempt / "workload.pinned.json"
        workload_path.write_text(
            json.dumps(workload_document, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        workload_sha256 = hashlib.sha256(workload_path.read_bytes()).hexdigest()
        reference = summary.token_client.TokenReference(
            summary.QWEN_MODEL_ID,
            tuple(prompt_token_ids),
            tuple(output_token_ids),
            "fixture-output",
            "length",
        )
        container_name = (
            f"riley-n06a-vllm-timed-{index:04d}-"
            f"{hashlib.sha256(str(attempt.resolve()).encode('utf-8')).hexdigest()[:20]}"
        )
        model_identity = self.make_model_identity_artifacts(attempt, directory)
        (attempt / "attempt.config.json").write_text(
            json.dumps(
                {
                    "schema_version": summary.ATTEMPT_ARTIFACT_SCHEMA_VERSION,
                    "phase": "timed",
                    "index": index,
                    "pair_order": pair_order,
                    "workload": {
                        "path": str(workload_path.resolve()),
                        "sha256": workload_sha256,
                        "case": "general-c32-2048-plus-128",
                        "model_id": summary.QWEN_MODEL_ID,
                        "model_revision": summary.QWEN_MODEL_REVISION,
                        "prompt_tokens": 2048,
                        "max_output_tokens": 128,
                        "reference_sha256": reference.sha256,
                    },
                    "configuration": {
                        "concurrency": 32,
                        "batch_token_budget": 32,
                        "max_model_len": 32768,
                        "warmup_requests": 100,
                        "request_timeout_seconds": 120.0,
                        "cuda_visible_devices": "0",
                        "whole_gpu_memory_sampling": {
                            "nvidia_smi": "/usr/bin/nvidia-smi",
                            "gpu_index": 0,
                            "sampling_interval_seconds": 0.25,
                            "max_sample_duration_seconds": 0.5,
                            "max_start_gap_seconds": 0.75,
                            "peak_limit_bytes": summary.WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES,
                        },
                        "lane_psi": {
                            "schema_version": summary.LANE_PSI_SCHEMA_VERSION,
                            "proc_root": "/proc",
                            "resources": ["cpu", "io", "memory"],
                            "policy": summary.LANE_PSI_POLICY,
                        },
                    },
                    "launch_provenance": {
                        "model_identity": {
                            "path": model_identity["model_identity_manifest_path"],
                            "sha256": model_identity["model_identity_manifest_sha256"],
                            "source_path": model_identity["model_identity_manifest_path"],
                            "source_sha256": model_identity["model_identity_manifest_sha256"],
                            "validation_path": model_identity["model_identity_validation_path"],
                            "validation_sha256": model_identity["model_identity_validation_sha256"],
                            "git_path": model_identity["model_identity_git_path"],
                            "git_sha256": model_identity["model_identity_git_sha256"],
                        },
                        "vllm_image_digest": VLLM_IMAGE_DIGEST,
                        "vllm_docker_container": {
                            "name": container_name,
                            "detached": False,
                            "cleanup_schema_version": "riley.n06a-docker-container-cleanup.v1",
                        },
                        "vllm_backend_attestation": {
                            "backend_requested": summary.VLLM_AUTO_BACKEND_REQUESTED,
                            "startup_receipt_regex": summary.VLLM_BACKEND_RECEIPT_REGEX.pattern,
                            "startup_required_fragments": [
                                "Using FLASH_ATTN attention backend out of potential backends:"
                            ],
                            "required_match_count": 1,
                        },
                        "vllm_fixed_prompt_fairness": {
                            "prefix_caching": summary.VLLM_PREFIX_CACHING_DISABLED,
                            "docker_gpu_binding": "device=0",
                            "gpu_memory_utilization": summary.VLLM_GPU_MEMORY_UTILIZATION,
                            "whole_gpu_sampled_peak_limit_bytes": summary.WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES,
                        }
                    },
                    "vllm_argv": [
                        "/usr/bin/docker",
                        "run",
                        "--name",
                        container_name,
                        "--rm",
                        "--gpus",
                        "device=0",
                        "--ipc",
                        "host",
                        "--network",
                        "host",
                        "-v",
                        model_identity["model_path"] + ":/model:ro",
                        "vllm/vllm-openai@" + VLLM_IMAGE_DIGEST,
                        "--model",
                        "/model",
                        "--served-model-name",
                        summary.QWEN_MODEL_ID,
                        "--dtype",
                        "bfloat16",
                        "--kv-cache-dtype",
                        "bfloat16",
                        "--enable-chunked-prefill",
                        "--no-enable-prefix-caching",
                        "--max-model-len",
                        "32768",
                        "--max-num-seqs",
                        "32",
                        "--max-num-batched-tokens",
                        "32",
                        "--gpu-memory-utilization",
                        "0.65",
                    ],
                    "riley_argv": [
                        "/usr/bin/riley",
                        "serve",
                        "--model",
                        model_identity["model_path"],
                        "--model-id",
                        summary.QWEN_MODEL_ID,
                    ],
                    "controller": {
                        "token_client_path": str(Path(summary.token_client.__file__).resolve()),
                        "token_client_sha256": hashlib.sha256(
                            Path(summary.token_client.__file__).read_bytes()
                        ).hexdigest(),
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return attempt

    def make_lane_artifacts(self, attempt: Path, *, lane: str, wall_ms: float) -> dict[str, str]:
        count, output_tokens = 100, 128
        config = json.loads((attempt / "attempt.config.json").read_text(encoding="utf-8"))
        workload_path = Path(config["workload"]["path"])
        workload = json.loads(workload_path.read_text(encoding="utf-8"))
        reference = summary.token_client.TokenReference(
            workload["model_id"],
            tuple(workload["prompt_token_ids"]),
            tuple(workload["output_token_ids"]),
            workload["output_text"],
            workload["finish_reason"],
        )
        ttft_ns = 9_000_000 if lane == "riley" else 10_000_000
        tpot_ns = 1_100_000 if lane == "riley" else 1_300_000
        e2e_ns = 180_000_000 if lane == "riley" else 200_000_000
        request = {
            "model": reference.model,
            "prompt": workload["prompt"],
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
        request_sha256 = hashlib.sha256(
            json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()

        def strict_row(index: int, *, phase: str, warmup: bool) -> dict[str, object]:
            batch = index // 32
            started_ns = 10_000_000 + batch * 200_000_000
            headers_received_ns = started_ns + 1
            finished_ns = started_ns + e2e_ns
            identity = {
                "id": f"{lane}-response-{index}",
                "model": reference.model,
                "object": "text_completion",
                "created": 1,
            }
            frames: list[dict[str, object]] = []
            for token_index, token_id in enumerate(reference.output_token_ids):
                choice: dict[str, object] = {
                    "index": 0,
                    "text": workload["output_text"] if token_index == 0 else "",
                    "token_ids": [token_id],
                    "finish_reason": None,
                }
                if token_index == 0:
                    choice["prompt_token_ids"] = list(reference.prompt_token_ids)
                frames.append(
                    {
                        "arrived_ns": started_ns + ttft_ns + token_index * tpot_ns,
                        "data": json.dumps({**identity, "choices": [choice]}, separators=(",", ":")),
                    }
                )
            frames.extend(
                [
                    {
                        "arrived_ns": finished_ns - 2,
                        "data": json.dumps(
                            {
                                **identity,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": "",
                                        "token_ids": [],
                                        "finish_reason": "length",
                                    }
                                ],
                            },
                            separators=(",", ":"),
                        ),
                    },
                    {
                        "arrived_ns": finished_ns - 1,
                        "data": json.dumps(
                            {
                                **identity,
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": 2048,
                                    "completion_tokens": 128,
                                    "total_tokens": 2176,
                                },
                            },
                            separators=(",", ":"),
                        ),
                    },
                    {"arrived_ns": finished_ns, "data": "[DONE]"},
                ]
            )
            parser = summary.token_client.TokenResponseParser(
                reference, streaming=True, started_ns=started_ns, mode="strict"
            )
            for frame in frames:
                parser.feed_sse(str(frame["data"]).encode("utf-8"), int(frame["arrived_ns"]))
            snapshot = parser.finish()
            return {
                **snapshot,
                "status": "success",
                "error": None,
                "http_status": 200,
                "headers_received_ns": headers_received_ns,
                "call_finished_ns": finished_ns + 1,
                "raw_body_bytes": 1,
                "total_deadline_seconds": 120.0,
                "owned_connection_closed": True,
                "cleanup_errors": [],
                "request": request,
                "request_body_sha256": request_sha256,
                "parser_protocol_valid": True,
                "transport_complete": True,
                "index": index,
                "worker_id": index % 32,
                "phase": phase,
                "warmup": warmup,
                "call_started_ns": started_ns - 1,
            }

        rows = [strict_row(index, phase="retained", warmup=False) for index in range(count)]
        rows_path = attempt / f"{lane}.requests.jsonl"
        rows_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        phase_path = attempt / f"{lane}.phase.json"
        phase_path.write_text(
            json.dumps(
                {
                    "schema_version": summary.RETAINED_PHASE_SCHEMA_VERSION,
                    "phase": "retained",
                    "warmup": False,
                    "offered_concurrency": 32,
                    "requested": count,
                    "attempted": count,
                    "succeeded": count,
                    "failed": 0,
                    "unresolved_attempts": 0,
                    "not_started": 0,
                    "observed_max_request_in_flight": 32,
                    "phase_started_ns": 1,
                    "phase_finished_ns": int(wall_ms * 1_000_000) + 1,
                    "phase_wall_ns": int(wall_ms * 1_000_000),
                    "duplicate_response_ids": [],
                    "worker_errors": [],
                    "completed": True,
                    "failure_policy": "stop refill, drain owned in-flight requests, preserve all started rows, no retries",
                    "timing_scope": "client-observed single-token SSE delivery; not CUDA or scheduler commit time",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        warmup_rows = [strict_row(index, phase="server-warmup", warmup=True) for index in range(count)]
        warmup_rows_path = attempt / f"{lane}.server-warmup.requests.jsonl"
        warmup_rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in warmup_rows),
            encoding="utf-8",
        )
        warmup_phase_path = attempt / f"{lane}.server-warmup.phase.json"
        warmup_phase_path.write_text(
            json.dumps(
                {
                    "schema_version": summary.RETAINED_PHASE_SCHEMA_VERSION,
                    "phase": "server-warmup",
                    "warmup": True,
                    "offered_concurrency": 32,
                    "requested": count,
                    "attempted": count,
                    "succeeded": count,
                    "failed": 0,
                    "unresolved_attempts": 0,
                    "not_started": 0,
                    "observed_max_request_in_flight": 32,
                    "phase_started_ns": 1,
                    "phase_finished_ns": int(wall_ms * 1_000_000) + 1,
                    "phase_wall_ns": int(wall_ms * 1_000_000),
                    "duplicate_response_ids": [],
                    "worker_errors": [],
                    "completed": True,
                    "failure_policy": "stop refill, drain owned in-flight requests, preserve all started rows, no retries",
                    "timing_scope": "client-observed single-token SSE delivery; not CUDA or scheduler commit time",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        config_path = attempt / "attempt.config.json"
        model_identity = config["launch_provenance"]["model_identity"]
        return {
            "model_identity_manifest_path": model_identity["path"],
            "model_identity_manifest_sha256": model_identity["sha256"],
            "model_identity_validation_path": model_identity["validation_path"],
            "model_identity_validation_sha256": model_identity["validation_sha256"],
            "request_rows_path": str(rows_path),
            "request_rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            "phase_path": str(phase_path),
            "phase_sha256": hashlib.sha256(phase_path.read_bytes()).hexdigest(),
            "server_warmup_request_rows_path": str(warmup_rows_path),
            "server_warmup_request_rows_sha256": hashlib.sha256(warmup_rows_path.read_bytes()).hexdigest(),
            "server_warmup_phase_path": str(warmup_phase_path),
            "server_warmup_phase_sha256": hashlib.sha256(warmup_phase_path.read_bytes()).hexdigest(),
            "attempt_config_path": str(config_path),
            "attempt_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }

    def make_gpu_memory_artifacts(self, attempt: Path, *, lane: str) -> dict[str, str]:
        sample_path = attempt / f"{lane}.whole-gpu-memory.samples.jsonl"
        samples = [
            {
                "started_ns": started_ns,
                "finished_ns": started_ns + 10,
                "monotonic_ns": started_ns + 10,
                "memory_used_bytes": 18_500_000_000 if started_ns == 250_000_000 else 18_000_000_000,
            }
            for started_ns in range(0, 2_000_000_001, 250_000_000)
        ]
        sample_path.write_text(
            "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in samples),
            encoding="utf-8",
        )
        sample_sha256 = hashlib.sha256(sample_path.read_bytes()).hexdigest()
        sample_resolved = sample_path.resolve()
        peak_path = attempt / f"{lane}.whole-gpu-memory.peak.json"
        peak_path.write_text(
            json.dumps(
                {
                    "schema_version": "riley.n06a-whole-gpu-memory-peak.v1",
                    "lane": lane,
                    "gpu_index": 0,
                    "sampling_interval_seconds": 0.25,
                    "max_sample_duration_seconds": 0.5,
                    "max_start_gap_seconds": 0.75,
                    "sample_count": len(samples),
                    "sample_path": str(sample_resolved),
                    "sample_sha256": sample_sha256,
                    "peak_used_bytes": 18_500_000_000,
                    "peak_limit_bytes": summary.WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES,
                    "compliant": True,
                    "errors": [],
                    "lifecycle": {
                        "server_started_ns": 0,
                        "server_cleanup_completed_ns": 2_000_000_000,
                        "sample_started_before_server_launch": True,
                        "sample_overlapped_server_lifetime": True,
                        "sample_finished_after_server_cleanup": True,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "whole_gpu_sampled_peak_bytes": "18500000000",
            "whole_gpu_sample_path": str(sample_resolved),
            "whole_gpu_sample_sha256": sample_sha256,
            "whole_gpu_peak_receipt_path": str(peak_path.resolve()),
            "whole_gpu_peak_receipt_sha256": hashlib.sha256(peak_path.read_bytes()).hexdigest(),
        }

    def make_vllm_startup_snapshots(self, attempt: Path) -> dict[str, str]:
        stdout_path = attempt / "vllm.startup.stdout.log"
        stderr_path = attempt / "vllm.startup.stderr.log"
        stdout_path.write_text(
            "INFO Using FLASH_ATTN attention backend out of potential backends: FLASH_ATTN, XFORMERS.\n",
            encoding="utf-8",
        )
        stderr_path.write_text("vLLM fixture stderr\n", encoding="utf-8")
        return {
            "startup_stdout_log_path": str(stdout_path.resolve()),
            "startup_stdout_log_sha256": hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
            "startup_stderr_log_path": str(stderr_path.resolve()),
            "startup_stderr_log_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
        }

    def make_lane_provenance(
        self,
        attempt: Path,
        *,
        lane: str,
        gpu_artifacts: dict[str, str],
        startup_log: Path | None,
        vllm_startup: dict[str, str] | None,
    ) -> dict[str, str]:
        config = json.loads((attempt / "attempt.config.json").read_text(encoding="utf-8"))
        sample_path = Path(gpu_artifacts["whole_gpu_sample_path"])
        peak_path = Path(gpu_artifacts["whole_gpu_peak_receipt_path"])
        gpu = {
            "sample_path": str(sample_path),
            "sample_sha256": gpu_artifacts["whole_gpu_sample_sha256"],
            "peak_path": str(peak_path),
            "peak_sha256": gpu_artifacts["whole_gpu_peak_receipt_sha256"],
            "sample_count": 9,
            "peak_used_bytes": 18_500_000_000,
            "peak_limit_bytes": summary.WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES,
            "compliant": True,
            "errors": [],
            "sampling_interval_seconds": 0.25,
            "max_sample_duration_seconds": 0.5,
            "max_start_gap_seconds": 0.75,
            "lifecycle": {
                "server_started_ns": 0,
                "server_cleanup_completed_ns": 2_000_000_000,
                "sample_started_before_server_launch": True,
                "sample_overlapped_server_lifetime": True,
                "sample_finished_after_server_cleanup": True,
            },
        }
        def lane_psi_snapshot(*, started_ns: int, finished_ns: int, base: int) -> dict[str, object]:
            def resource(name: str, offset: int, *, full: bool) -> dict[str, object]:
                value = float(base + offset)
                return {
                    "status": "ok",
                    "source": f"/proc/pressure/{name}",
                    "some": {
                        "avg10": value,
                        "avg60": value + 0.1,
                        "avg300": value + 0.2,
                        "total": base + offset,
                    },
                    "full": None
                    if not full
                    else {
                        "avg10": value / 2.0,
                        "avg60": value / 2.0 + 0.1,
                        "avg300": value / 2.0 + 0.2,
                        "total": base + offset,
                    },
                    "error": None,
                }

            return {
                "schema_version": summary.LANE_PSI_SCHEMA_VERSION,
                "policy": summary.LANE_PSI_POLICY,
                "snapshot_started_ns": started_ns,
                "snapshot_finished_ns": finished_ns,
                "psi": {
                    "cpu": resource("cpu", 0, full=False),
                    "io": resource("io", 100, full=True),
                    "memory": resource("memory", 200, full=True),
                },
            }

        lane_pressure_path = attempt / f"{lane}.lane-psi.json"
        lane_pressure = {
            "schema_version": summary.LANE_PSI_SCHEMA_VERSION,
            "policy": summary.LANE_PSI_POLICY,
            "lane": lane,
            "attempt": {
                "phase": config["phase"],
                "index": config["index"],
                "pair_order": config["pair_order"],
            },
            "proc_root": "/proc",
            "pre": lane_psi_snapshot(started_ns=0, finished_ns=0, base=1_000 if lane == "riley" else 2_000),
            "post": lane_psi_snapshot(
                started_ns=2_000_001_000,
                finished_ns=2_000_001_001,
                base=1_500 if lane == "riley" else 2_500,
            ),
        }
        lane_pressure_path.write_text(
            json.dumps(lane_pressure, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance: dict[str, object] = {
            "argv": config[f"{lane}_argv"],
            "pid": 10_000 if lane == "riley" else 10_001,
            "started_ns": 0,
            "ready": {
                "endpoint": "/readyz" if lane == "riley" else "/health",
                "http_status": 200,
                "ready_ns": 0,
            },
            "stdout_path": str((attempt / f"{lane}.stdout.log").resolve()),
            "stderr_path": str((attempt / f"{lane}.stderr.log").resolve()),
            "cleanup": {
                "lane": lane,
                "pid": 10_000 if lane == "riley" else 10_001,
                "returncode": 0,
                "cleanup_verified": True,
                "errors": [],
            },
            "cleanup_completed_ns": 2_000_000_000,
            "whole_gpu_memory": gpu,
            "lane_pressure": {
                "path": str(lane_pressure_path.resolve()),
                "sha256": hashlib.sha256(lane_pressure_path.read_bytes()).hexdigest(),
                "policy": summary.LANE_PSI_POLICY,
            },
            "post_lane_gpu_idle": {
                "schema_version": "riley.n06a-post-lane-gpu-idle-census.v1",
                "gpu_index": 0,
                "max_idle_memory_bytes": 512 * 1024 * 1024,
                "compute_process_pids": [],
                "memory_used_bytes": 64 * 1024 * 1024,
                "compliant": True,
                "errors": [],
                "commands": [],
            },
        }
        def idle_command(argv: list[str], stdout: str, *, started_ns: int) -> dict[str, object]:
            stderr = ""
            return {
                "argv": argv,
                "started_ns": started_ns,
                "finished_ns": started_ns + 1,
                "timeout_seconds": 5.0,
                "returncode": 0,
                "stdout_text": stdout,
                "stdout_bytes": len(stdout.encode("utf-8")),
                "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                "stderr_text": stderr,
                "stderr_bytes": 0,
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            }

        provenance["post_lane_gpu_idle"]["commands"] = [
            idle_command(
                ["/usr/bin/nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                "",
                started_ns=2_000_001_010,
            ),
            idle_command(
                ["/usr/bin/nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                "64 MiB\n",
                started_ns=2_000_001_012,
            ),
        ]
        if lane == "riley":
            assert startup_log is not None
            provenance["riley_startup_snapshot"] = {
                "path": str(startup_log.resolve()),
                "sha256": hashlib.sha256(startup_log.read_bytes()).hexdigest(),
                "receipt": {},
            }
        else:
            assert vllm_startup is not None
            container_name = config["launch_provenance"]["vllm_docker_container"]["name"]

            def cleanup_command(operation: str, arguments: list[str], returncode: int, stderr: str = "") -> dict[str, object]:
                stdout = ""
                return {
                    "operation": operation,
                    "argv": ["/usr/bin/docker", *arguments],
                    "timeout_seconds": 5.0,
                    "returncode": returncode,
                    "stdout_bytes": len(stdout.encode("utf-8")),
                    "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                    "stderr_bytes": len(stderr.encode("utf-8")),
                    "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
                    "stdout_text": stdout,
                    "stderr_text": stderr,
                }

            inventory = [
                "container",
                "ls",
                "--all",
                "--filter",
                f"name=^/{container_name}$",
                "--format",
                "{{.Names}}\t{{.ID}}",
            ]
            provenance["cleanup"] = {
                **provenance["cleanup"],
                "container": {
                    "schema_version": "riley.n06a-docker-container-cleanup.v1",
                    "launcher": "/usr/bin/docker",
                    "container_name": container_name,
                    "commands": [
                        cleanup_command("inventory-before", inventory, 0),
                        cleanup_command("inventory-after-stop", inventory, 0),
                        cleanup_command(
                            "inspect-absence",
                            ["container", "inspect", "--format", "{{.Id}}", container_name],
                            1,
                            "No such container\n",
                        ),
                    ],
                    "notes": [],
                    "errors": [],
                    "final_state": "absent",
                    "cleanup_verified": True,
                },
            }
            provenance["vllm_startup_snapshot"] = {
                "stdout_path": vllm_startup["startup_stdout_log_path"],
                "stdout_sha256": vllm_startup["startup_stdout_log_sha256"],
                "stderr_path": vllm_startup["startup_stderr_log_path"],
                "stderr_sha256": vllm_startup["startup_stderr_log_sha256"],
                "backend_requested": summary.VLLM_AUTO_BACKEND_REQUESTED,
                "backend_resolved": "FLASH_ATTN",
            }
        path = attempt / f"{lane}.provenance.json"
        path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "lane_provenance_path": str(path.resolve()),
            "lane_provenance_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def serving_marker(
        self,
        *,
        lane: str,
        startup_log: Path | None,
        pair_order: str,
        artifacts: dict[str, str],
        riley_wall_ms: float = 800.0,
        vllm_wall_ms: float = 1_000.0,
    ) -> str:
        output_tokens = 100 * 128
        wall_ms = riley_wall_ms if lane == "riley" else vllm_wall_ms
        values = {
            "schema_version": "1",
            "lane": lane,
            "case": "general-c32-2048-plus-128",
            "pair_order": pair_order,
            "model_id": summary.QWEN_MODEL_ID,
            "model_revision": summary.QWEN_MODEL_REVISION,
            "prompt_tokens": "2048",
            "max_output_tokens": "128",
            "offered_concurrency": "32",
            "batch_token_budget": "32",
            "max_model_len": "32768",
            "vllm_prefix_caching": summary.VLLM_PREFIX_CACHING_DISABLED,
            "vllm_image_digest": VLLM_IMAGE_DIGEST,
            "vllm_gpu_memory_utilization": summary.VLLM_GPU_MEMORY_UTILIZATION,
            "whole_gpu_sampled_peak_limit_bytes": str(summary.WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES),
            **artifacts,
            "attempted_requests": "100",
            "successful_requests": "100",
            "failed_requests": "0",
            "output_tokens": str(output_tokens),
            "retained_wall_ms": f"{wall_ms:.6f}",
            "output_tokens_per_second": f"{output_tokens * 1000.0 / wall_ms:.6f}",
            "token_timing_scope": "client-observed-single-token-sse",
            "token_ttft_median_ms": "9.000000" if lane == "riley" else "10.000000",
            "token_ttft_p95_ms": "9.000000" if lane == "riley" else "10.000000",
            "token_ttft_p99_ms": "9.000000" if lane == "riley" else "10.000000",
            "token_tpot_median_ms": "1.100000" if lane == "riley" else "1.300000",
            "token_tpot_p95_ms": "1.100000" if lane == "riley" else "1.300000",
            "token_tpot_p99_ms": "1.100000" if lane == "riley" else "1.300000",
            "e2e_median_ms": "180.000000" if lane == "riley" else "200.000000",
            "e2e_p95_ms": "180.000000" if lane == "riley" else "200.000000",
            "e2e_p99_ms": "180.000000" if lane == "riley" else "200.000000",
            "latency_sample_count": "100",
            "p99_status": "descriptive",
            "quality_status": "passed",
            "status": "passed",
        }
        if lane == "riley":
            assert startup_log is not None
            values.update(
                backend_requested=summary.REQUESTED_BACKEND_CLI_ID,
                backend_resolved=summary.SERVING_IMPLEMENTATION_ID,
                fallback_reason="none",
                startup_log_path=str(startup_log.resolve()),
                startup_log_sha256=hashlib.sha256(startup_log.read_bytes()).hexdigest(),
                query_heads="16",
                key_value_heads="2",
                head_size="128",
                page_size="16",
                graph_capture_enabled="false",
            )
        else:
            values.update(
                backend_requested=summary.VLLM_AUTO_BACKEND_REQUESTED,
                backend_resolved="FLASH_ATTN",
                fallback_reason="none",
            )
        return summary.SERVING_MARKER_PREFIX + " " + " ".join(
            f"{key}={value}" for key, value in values.items()
        )

    def make_serving_receipt(
        self,
        root: Path,
        *,
        statuses: tuple[str, ...] = ("succeeded", "failed", "succeeded"),
    ) -> Path:
        output = root / "serving-evidence"
        output.mkdir()
        runs: list[dict[str, object]] = []
        for index, status in enumerate(statuses, start=1):
            stdout = output / f"timed-{index:03d}.stdout.log"
            stderr = output / f"timed-{index:03d}.stderr.log"
            if status == "succeeded":
                pair_order = "riley-vllm" if index % 2 else "vllm-riley"
                attempt = self.make_attempt_directory(output, index=index, pair_order=pair_order)
                startup_log = self.make_startup_snapshot(attempt)
                riley_wall_ms = 800.0 + index * 10.0
                vllm_wall_ms = 1000.0 + index * 10.0
                vllm_startup = self.make_vllm_startup_snapshots(attempt)
                riley_gpu = self.make_gpu_memory_artifacts(attempt, lane="riley")
                vllm_gpu = self.make_gpu_memory_artifacts(attempt, lane="vllm")
                riley_artifacts = {
                    **self.make_lane_artifacts(attempt, lane="riley", wall_ms=riley_wall_ms),
                    **riley_gpu,
                }
                vllm_artifacts = {
                    **self.make_lane_artifacts(attempt, lane="vllm", wall_ms=vllm_wall_ms),
                    **vllm_gpu,
                    **vllm_startup,
                }
                riley_artifacts.update(
                    self.make_lane_provenance(
                        attempt,
                        lane="riley",
                        gpu_artifacts=riley_gpu,
                        startup_log=startup_log,
                        vllm_startup=None,
                    )
                )
                vllm_artifacts.update(
                    self.make_lane_provenance(
                        attempt,
                        lane="vllm",
                        gpu_artifacts=vllm_gpu,
                        startup_log=None,
                        vllm_startup=vllm_startup,
                    )
                )
                riley_marker = self.serving_marker(
                    lane="riley",
                    startup_log=startup_log,
                    pair_order=pair_order,
                    artifacts=riley_artifacts,
                    riley_wall_ms=riley_wall_ms,
                )
                vllm_marker = self.serving_marker(
                    lane="vllm",
                    startup_log=None,
                    pair_order=pair_order,
                    artifacts=vllm_artifacts,
                    vllm_wall_ms=vllm_wall_ms,
                )
                ordered_markers = (
                    (riley_marker, vllm_marker)
                    if pair_order == "riley-vllm"
                    else (vllm_marker, riley_marker)
                )
                stdout.write_text(
                    "\n".join(
                        [
                            "paired serving command started",
                            *ordered_markers,
                            "paired serving command completed",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                stdout.write_text("failed outer command retained\n", encoding="utf-8")
            stderr.write_text("fixture stderr\n", encoding="utf-8")
            runs.append(
                {
                    "kind": "timed",
                    "index": index,
                    "status": status,
                    "exit_code": 0 if status == "succeeded" else 17,
                    "timed_out": False,
                    "wall_time_ms": 100.0 + index,
                    "error": None if status == "succeeded" else "fixture failure retained",
                    "stdout_path": str(stdout),
                    "stderr_path": str(stderr),
                    "pre": observation(20.0 + index),
                    "post": observation(30.0 + index),
                }
            )
        receipt_path = output / "n01-repeat-control-receipt.json"
        receipt = {
            "schema_version": summary.REPEAT_RECEIPT_SCHEMA_VERSION,
            "receipt_path": str(receipt_path),
            "output_dir": str(output),
            "configuration": {
                "timed_repeats": len(statuses),
                "bootstrap_resamples": 300,
                "bootstrap_seed": 260913,
            },
            "status": "completed" if all(status == "succeeded" for status in statuses) else "completed-with-failures",
            "timed_runs": runs,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt_path

    def attach_n01_timeout_cleanup(
        self,
        receipt_path: Path,
        *,
        failed_cleanup_indices: frozenset[int] = frozenset(),
    ) -> None:
        """Attach N01's exact-name parent cleanup sidecars to a fixture receipt."""
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        output = Path(receipt["output_dir"]).resolve()
        artifact_root = (output / "paired-driver").resolve()
        configuration = receipt["configuration"]
        configuration["warmups"] = 0
        receipt["warmup_runs"] = []
        configuration["n06a_timeout_cleanup"] = {
            "artifact_root": str(artifact_root),
            "docker_launcher": "/usr/bin/docker",
            "command_timeout_seconds": 5.0,
            "scope": "one exact N06-A vLLM name per outer attempt",
        }

        def command(operation: str, argv: list[str], returncode: int, stderr: str = "") -> dict[str, object]:
            stdout = ""
            return {
                "operation": operation,
                "argv": argv,
                "timeout_seconds": 5.0,
                "returncode": returncode,
                "stdout_text": stdout,
                "stdout_bytes": len(stdout.encode("utf-8")),
                "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                "stderr_text": stderr,
                "stderr_bytes": len(stderr.encode("utf-8")),
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            }

        for run in receipt["timed_runs"]:
            index = int(run["index"])
            attempt = (artifact_root / f"timed-{index:03d}").resolve()
            container_name = (
                f"riley-n06a-vllm-timed-{index:04d}-"
                f"{hashlib.sha256(str(attempt).encode('utf-8')).hexdigest()[:20]}"
            )
            inventory = [
                "/usr/bin/docker",
                "container",
                "ls",
                "--all",
                "--filter",
                f"name=^/{container_name}$",
                "--format",
                "{{.Names}}\t{{.ID}}",
            ]
            docker_cleanup = {
                "schema_version": "riley.n01-n06a-docker-container-cleanup.v1",
                "launcher": "/usr/bin/docker",
                "container_name": container_name,
                "commands": [
                    command("inventory-before", inventory, 0),
                    command("inventory-after-stop", inventory, 0),
                    command(
                        "inspect-absence",
                        [
                            "/usr/bin/docker",
                            "container",
                            "inspect",
                            "--format",
                            "{{.Id}}",
                            container_name,
                        ],
                        1,
                        "No such container\\n",
                    ),
                ],
                "notes": [],
                "errors": [],
                "final_state": "absent",
                "cleanup_verified": True,
            }
            failed_cleanup = index in failed_cleanup_indices
            parent = {
                "schema_version": "riley.n01-n06a-parent-cleanup.v1",
                "kind": "timed",
                "index": index,
                "attempt_artifact_directory": str(attempt),
                "container_name": container_name,
                "timed_out": bool(run["timed_out"]),
                "parent_termination_requested": False,
                "timeout_descendant_cleanup": {
                    "status": "not-required",
                    "cleanup_verified": True,
                    "reason": "outer attempt did not time out",
                },
                "outer_process_cleanup": {
                    "status": "not-required",
                    "cleanup_verified": True,
                    "reason": "outer attempt did not require parent termination",
                },
                "docker_container_cleanup": docker_cleanup,
                "errors": ["fixture cleanup proof unavailable"] if failed_cleanup else [],
                "cleanup_verified": not failed_cleanup,
            }
            sidecar = output / f"timed-{index:03d}.n06a-parent-cleanup.json"
            sidecar.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            run["n06a_parent_cleanup"] = {
                **parent,
                "receipt_path": str(sidecar),
                "receipt_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            }
            if failed_cleanup:
                run["status"] = "failed-cleanup"
                run["exit_code"] = 17
                run["error"] = "fixture parent cleanup proof unavailable"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    def make_operator_receipt(self, root: Path) -> Path:
        output = root / "operator-evidence"
        output.mkdir()
        stdout = output / "timed-001.stdout.log"
        stdout.write_text(
            "\n".join(n03a_record(case, offset=0.01) for case in n03a.EXPECTED_CASES) + "\n",
            encoding="utf-8",
        )
        stderr = output / "timed-001.stderr.log"
        stderr.write_text("fixture stderr\n", encoding="utf-8")
        receipt_path = output / "n01-repeat-control-receipt.json"
        receipt = {
            "schema_version": summary.REPEAT_RECEIPT_SCHEMA_VERSION,
            "receipt_path": str(receipt_path),
            "output_dir": str(output),
            "configuration": {"timed_repeats": 1, "bootstrap_resamples": 300, "bootstrap_seed": 99},
            "status": "completed",
            "timed_runs": [
                {
                    "kind": "timed",
                    "index": 1,
                    "status": "succeeded",
                    "exit_code": 0,
                    "timed_out": False,
                    "wall_time_ms": 100.0,
                    "error": None,
                    "stdout_path": str(stdout),
                    "stderr_path": str(stderr),
                    "pre": observation(1.0),
                    "post": observation(2.0),
                }
            ],
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt_path

    def test_serving_summary_retains_all_attempts_pressure_and_paired_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt = self.make_serving_receipt(Path(temporary))
            first = summary.summarize(serving_receipt=receipt)
            second = summary.summarize(serving_receipt=receipt)

        self.assertEqual(first, second)
        serving = first["serving"]
        assert serving is not None
        self.assertEqual(first["schema_version"], summary.SUMMARY_SCHEMA_VERSION)
        self.assertIsNone(first["operator_control"])
        self.assertEqual(serving["receipt"]["successful_timed_runs"], 2)
        self.assertEqual(serving["receipt"]["non_successful_timed_runs"], 1)
        self.assertEqual(serving["promotion_status"]["status"], "incomplete")
        self.assertEqual(serving["promotion_status"]["pair_completion"]["completed_over_planned"], "2/3")
        self.assertEqual(serving["paired_tail_effect_status"]["status"], "incomplete")
        self.assertEqual(len(serving["timed_run_pressure_covariates"]), 3)
        self.assertEqual(len(serving["lane_pressure_covariates"]), 4)
        first_lane_pressure = serving["lane_pressure_covariates"][0]
        self.assertEqual(first_lane_pressure["timed_index"], 1)
        self.assertEqual(first_lane_pressure["pair_order"], "riley-vllm")
        self.assertEqual(first_lane_pressure["lane"], "riley")
        self.assertEqual(first_lane_pressure["lane_position"], "first")
        self.assertGreater(first_lane_pressure["derived"]["io"]["some"]["total_delta_us"], 0)
        pressure_view = serving["order_by_lane_pressure_sensitivity"]
        self.assertEqual(pressure_view["status"], "descriptive")
        self.assertEqual(
            pressure_view["by_pair_order"]["riley-vllm"]["lanes"]["riley"]["lane_position"],
            "first",
        )
        self.assertEqual(
            pressure_view["by_pair_order"]["vllm-riley"]["lanes"]["riley"]["lane_position"],
            "second",
        )
        failed = serving["timed_run_pressure_covariates"][1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["psi"]["pre"]["cpu"]["some"]["avg10"], 22.0)
        self.assertEqual(failed["psi"]["pre"]["io"]["some"]["avg10"], 32.0)
        self.assertEqual(failed["psi"]["pre"]["memory"]["some"]["avg10"], 42.0)
        self.assertIn("fixture failure retained", failed["error"])
        paired = serving["paired_outer_process_metrics"]
        self.assertGreater(paired["throughput_riley_over_vllm"]["median"], 1.0)
        self.assertLess(paired["token_ttft_median_riley_over_vllm"]["median"], 1.0)
        self.assertLess(paired["token_ttft_p95_riley_over_vllm"]["median"], 1.0)
        self.assertIn("token_ttft_p99_vllm_minus_riley_ms", serving["paired_tail_outer_process_metrics"])
        interval = paired["throughput_riley_over_vllm"]["deterministic_bootstrap_median_95_ci"]
        self.assertEqual(interval["resamples"], 300)
        self.assertLessEqual(interval["lower"], paired["throughput_riley_over_vllm"]["median"])
        self.assertGreaterEqual(interval["upper"], paired["throughput_riley_over_vllm"]["median"])
        self.assertEqual(
            serving["expected_riley_backend"]["resolved_implementation_id"],
            summary.SERVING_IMPLEMENTATION_ID,
        )
        self.assertEqual(
            serving["successful_outer_pair_observations"][0]["riley"]["startup_snapshot"]["marker_prefix"],
            summary.STARTUP_RECEIPT_PREFIX,
        )

    def test_lane_psi_unavailable_and_malformed_are_exposed_without_excluding_a_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(
                Path(temporary), statuses=("succeeded", "succeeded")
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            riley_line = next(
                line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=riley" in line
            )
            provenance_path = Path(
                next(token.split("=", 1)[1] for token in riley_line.split() if token.startswith("lane_provenance_path="))
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            pressure_path = Path(provenance["lane_pressure"]["path"])
            pressure = json.loads(pressure_path.read_text(encoding="utf-8"))
            for phase, status in (("pre", "unavailable"), ("post", "malformed")):
                pressure[phase]["psi"]["io"] = {
                    "status": status,
                    "source": "/proc/pressure/io",
                    "some": None,
                    "full": None,
                    "error": f"fixture {status}",
                }
            pressure_path.write_text(json.dumps(pressure, sort_keys=True) + "\n", encoding="utf-8")
            provenance["lane_pressure"]["sha256"] = hashlib.sha256(pressure_path.read_bytes()).hexdigest()
            provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="lane_provenance_sha256",
                value=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            )
            serving = summary.summarize(serving_receipt=receipt_path)["serving"]
        assert serving is not None
        self.assertEqual(serving["receipt"]["successful_timed_runs"], 2)
        riley_covariate = next(
            item
            for item in serving["lane_pressure_covariates"]
            if item["timed_index"] == 1 and item["lane"] == "riley"
        )
        self.assertEqual(riley_covariate["pre"]["psi"]["io"]["status"], "unavailable")
        self.assertEqual(riley_covariate["post"]["psi"]["io"]["status"], "malformed")
        self.assertEqual(
            serving["order_by_lane_pressure_sensitivity"]["by_pair_order"]["riley-vllm"]
            ["lanes"]["riley"]["resources"]["io"]["categories"]["some"]["available_pair_count"],
            0,
        )

    def test_rejects_lane_psi_path_and_cumulative_total_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            riley_line = next(
                line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=riley" in line
            )
            provenance_path = Path(
                next(token.split("=", 1)[1] for token in riley_line.split() if token.startswith("lane_provenance_path="))
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            vllm_pressure_path = provenance_path.parent / "vllm.lane-psi.json"
            provenance["lane_pressure"]["path"] = str(vllm_pressure_path)
            provenance["lane_pressure"]["sha256"] = hashlib.sha256(
                vllm_pressure_path.read_bytes()
            ).hexdigest()
            provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="lane_provenance_sha256",
                value=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(summary.SummaryError, "dedicated lane file"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            riley_line = next(
                line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=riley" in line
            )
            provenance_path = Path(
                next(token.split("=", 1)[1] for token in riley_line.split() if token.startswith("lane_provenance_path="))
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            pressure_path = Path(provenance["lane_pressure"]["path"])
            pressure = json.loads(pressure_path.read_text(encoding="utf-8"))
            pre_total = pressure["pre"]["psi"]["io"]["some"]["total"]
            pressure["post"]["psi"]["io"]["some"]["total"] = pre_total - 1
            pressure_path.write_text(json.dumps(pressure, sort_keys=True) + "\n", encoding="utf-8")
            provenance["lane_pressure"]["sha256"] = hashlib.sha256(pressure_path.read_bytes()).hexdigest()
            provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="lane_provenance_sha256",
                value=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(summary.SummaryError, "total moves backwards"):
                summary.summarize(serving_receipt=receipt_path)

    def test_rejects_lane_psi_sampler_and_idle_census_lifecycle_tampering(self) -> None:
        def riley_artifacts(receipt_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            riley_line = next(
                line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=riley" in line
            )
            provenance_path = Path(
                next(token.split("=", 1)[1] for token in riley_line.split() if token.startswith("lane_provenance_path="))
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            pressure_path = Path(provenance["lane_pressure"]["path"])
            pressure = json.loads(pressure_path.read_text(encoding="utf-8"))
            return stdout_path, provenance_path, provenance, pressure

        def rebind_lane_pressure(
            *, stdout_path: Path, provenance_path: Path, provenance: dict[str, object], pressure: dict[str, object]
        ) -> None:
            pressure_path = Path(provenance["lane_pressure"]["path"])
            pressure_path.write_text(json.dumps(pressure, sort_keys=True) + "\n", encoding="utf-8")
            provenance["lane_pressure"]["sha256"] = hashlib.sha256(pressure_path.read_bytes()).hexdigest()
            provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="lane_provenance_sha256",
                value=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            )

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            stdout_path, provenance_path, provenance, pressure = riley_artifacts(receipt_path)
            pressure["post"]["snapshot_started_ns"] = 2_000_000_009
            pressure["post"]["snapshot_finished_ns"] = 2_000_000_009
            rebind_lane_pressure(
                stdout_path=stdout_path,
                provenance_path=provenance_path,
                provenance=provenance,
                pressure=pressure,
            )
            with self.assertRaisesRegex(summary.SummaryError, "before final whole-GPU sampler completion"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            stdout_path, provenance_path, provenance, pressure = riley_artifacts(receipt_path)
            pressure["post"]["snapshot_started_ns"] = 2_000_001_011
            pressure["post"]["snapshot_finished_ns"] = 2_000_001_011
            rebind_lane_pressure(
                stdout_path=stdout_path,
                provenance_path=provenance_path,
                provenance=provenance,
                pressure=pressure,
            )
            with self.assertRaisesRegex(summary.SummaryError, "extends past post-lane GPU idle census"):
                summary.summarize(serving_receipt=receipt_path)

    def test_operator_and_serving_evidence_remain_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = summary.summarize(
                operator_receipt=self.make_operator_receipt(root),
                serving_receipt=self.make_serving_receipt(root),
            )

        operator = result["operator_control"]
        serving = result["serving"]
        assert operator is not None and serving is not None
        self.assertFalse(operator["full_model_serving"])
        self.assertFalse(operator["vllm_comparison"])
        self.assertEqual(operator["expected_operator_implementation_id"], summary.OPERATOR_IMPLEMENTATION_ID)
        self.assertEqual(len(operator["timed_run_pressure_covariates"]), 1)
        self.assertEqual(operator["timed_run_pressure_covariates"][0]["psi"]["post"]["memory"]["some"]["avg10"], 22.0)
        self.assertEqual(serving["classification"], "paired-full-model-serving-d128-v2-versus-vllm")

    def test_all_failed_planned_pairs_return_an_explicit_zero_over_planned_incomplete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("failed", "failed"))
            result = summary.summarize(serving_receipt=receipt_path)
        serving = result["serving"]
        assert serving is not None
        self.assertEqual(serving["promotion_status"]["status"], "incomplete")
        self.assertEqual(serving["promotion_status"]["pair_completion"]["completed_over_planned"], "0/2")
        self.assertEqual(len(serving["promotion_status"]["pair_completion"]["non_successful_reasons"]), 2)
        self.assertIsNone(serving["paired_outer_process_metrics"])
        self.assertIsNone(serving["paired_tail_outer_process_metrics"])

    def test_n01_parent_cleanup_sidecars_bind_exact_owned_commands_and_retain_failed_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(
                Path(temporary), statuses=("succeeded", "failed")
            )
            self.attach_n01_timeout_cleanup(receipt_path, failed_cleanup_indices=frozenset({2}))
            serving = summary.summarize(serving_receipt=receipt_path)["serving"]
        assert serving is not None
        self.assertEqual(serving["promotion_status"]["status"], "incomplete")
        self.assertEqual(serving["promotion_status"]["pair_completion"]["completed_over_planned"], "1/2")
        failed = serving["timed_run_pressure_covariates"][1]
        self.assertEqual(failed["status"], "failed-cleanup")
        self.assertIn("cleanup proof", failed["error"])

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            self.attach_n01_timeout_cleanup(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            run = receipt["timed_runs"][0]
            parent = run["n06a_parent_cleanup"]
            sidecar = Path(parent["receipt_path"])
            sidecar_document = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_document["docker_container_cleanup"]["commands"][0]["argv"] = [
                "/usr/bin/docker",
                "container",
                "ls",
                "--all",
            ]
            sidecar.write_text(json.dumps(sidecar_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            run["n06a_parent_cleanup"] = {
                **sidecar_document,
                "receipt_path": str(sidecar),
                "receipt_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            }
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(summary.SummaryError, "exact owned-container command"):
                summary.summarize(serving_receipt=receipt_path)

    def test_n01_cleanup_configuration_validates_warmup_sidecars_too(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            self.attach_n01_timeout_cleanup(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            output = Path(receipt["output_dir"]).resolve()
            artifact_root = Path(receipt["configuration"]["n06a_timeout_cleanup"]["artifact_root"])
            timed_parent = receipt["timed_runs"][0]["n06a_parent_cleanup"]
            old_name = timed_parent["container_name"]
            attempt = (artifact_root / "warmup-001").resolve()
            warmup_name = (
                "riley-n06a-vllm-warmup-0001-"
                + hashlib.sha256(str(attempt).encode("utf-8")).hexdigest()[:20]
            )
            warmup_parent = json.loads(
                json.dumps(
                    {
                        key: value
                        for key, value in timed_parent.items()
                        if key not in {"receipt_path", "receipt_sha256"}
                    }
                )
            )
            warmup_parent["kind"] = "warmup"
            warmup_parent["attempt_artifact_directory"] = str(attempt)
            warmup_parent["container_name"] = warmup_name
            docker = warmup_parent["docker_container_cleanup"]
            docker["container_name"] = warmup_name
            for command in docker["commands"]:
                command["argv"] = [
                    warmup_name if item == old_name else item.replace(old_name, warmup_name)
                    for item in command["argv"]
                ]
            sidecar = output / "warmup-001.n06a-parent-cleanup.json"
            sidecar.write_text(json.dumps(warmup_parent, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            stdout = output / "warmup-001.stdout.log"
            stderr = output / "warmup-001.stderr.log"
            stdout.write_text("N06 warmup retained\n", encoding="utf-8")
            stderr.write_text("fixture stderr\n", encoding="utf-8")
            timed = receipt["timed_runs"][0]
            receipt["configuration"]["warmups"] = 1
            receipt["warmup_runs"] = [
                {
                    "kind": "warmup",
                    "index": 1,
                    "status": "succeeded",
                    "exit_code": 0,
                    "timed_out": False,
                    "wall_time_ms": 10.0,
                    "error": None,
                    "stdout_path": str(stdout),
                    "stderr_path": str(stderr),
                    "pre": timed["pre"],
                    "post": timed["post"],
                    "n06a_parent_cleanup": {
                        **warmup_parent,
                        "receipt_path": str(sidecar),
                        "receipt_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                    },
                }
            ]
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            serving = summary.summarize(serving_receipt=receipt_path)["serving"]
        assert serving is not None
        self.assertEqual(serving["receipt"]["successful_timed_runs"], 1)

    def test_n01_fail_stop_placeholder_is_incomplete_without_fake_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("failed", "failed"))
            self.attach_n01_timeout_cleanup(receipt_path, failed_cleanup_indices=frozenset({1}))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            second = receipt["timed_runs"][1]
            for key in ("stdout_path", "stderr_path"):
                Path(second[key]).unlink()
            receipt["timed_runs"][1] = {
                "kind": "timed",
                "index": 2,
                "status": "not-started-after-failed-cleanup",
                "blocking_kind": "timed",
                "blocking_index": 1,
                "reason": "N06-A parent cleanup was unproven; fail-stop prevents further launches",
            }
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            serving = summary.summarize(serving_receipt=receipt_path)["serving"]
        assert serving is not None
        self.assertEqual(serving["promotion_status"]["status"], "incomplete")
        self.assertEqual(serving["promotion_status"]["pair_completion"]["completed_over_planned"], "0/2")
        placeholder = serving["timed_run_pressure_covariates"][1]
        self.assertEqual(placeholder["status"], "not-started-after-failed-cleanup")
        self.assertIsNone(placeholder["attempt_log_paths"])
        self.assertIsNone(placeholder["psi"])
        self.assertEqual(
            serving["promotion_status"]["pair_completion"]["non_successful_reasons"][1]["blocking"],
            {"kind": "timed", "index": 1},
        )

    def test_n01_unavailable_docker_cleanup_is_retained_as_failed_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("failed",))
            self.attach_n01_timeout_cleanup(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            run = receipt["timed_runs"][0]
            parent = run["n06a_parent_cleanup"]
            sidecar = Path(parent["receipt_path"])
            sidecar_document = json.loads(sidecar.read_text(encoding="utf-8"))
            docker = sidecar_document["docker_container_cleanup"]
            command = docker["commands"][0]
            command["returncode"] = None
            command["error"] = "FileNotFoundError: Docker launcher unavailable"
            command.pop("stdout_text")
            command.pop("stderr_text")
            docker["commands"] = [command]
            docker["cleanup_verified"] = False
            docker["errors"] = ["Docker launcher unavailable"]
            docker.pop("final_state")
            sidecar_document["cleanup_verified"] = False
            sidecar_document["errors"] = ["N06-A Docker daemon absence is unproven"]
            sidecar.write_text(json.dumps(sidecar_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            run["n06a_parent_cleanup"] = {
                **sidecar_document,
                "receipt_path": str(sidecar),
                "receipt_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            }
            run["status"] = "failed-cleanup"
            run["exit_code"] = 17
            run["error"] = "Docker launcher unavailable"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            serving = summary.summarize(serving_receipt=receipt_path)["serving"]
        assert serving is not None
        self.assertEqual(serving["promotion_status"]["pair_completion"]["completed_over_planned"], "0/1")
        self.assertEqual(serving["timed_run_pressure_covariates"][0]["status"], "failed-cleanup")

    def test_even_complete_receipt_reports_pooled_and_order_stratified_tail_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(
                Path(temporary), statuses=("succeeded", "succeeded", "succeeded", "succeeded")
            )
            serving = summary.summarize(serving_receipt=receipt_path)["serving"]
        assert serving is not None
        self.assertEqual(serving["promotion_status"]["status"], "descriptive")
        self.assertEqual(serving["pair_order_balance"]["status"], "balanced")
        self.assertEqual(
            serving["order_stratified_effects"]["riley-vllm"]["successful_outer_pairs"], 2
        )
        tail = serving["paired_tail_outer_process_metrics"]
        assert tail is not None
        self.assertIn("token_tpot_p95_riley_over_vllm", tail)
        self.assertIn("deterministic_bootstrap_median_95_ci", tail["e2e_p99_vllm_minus_riley_ms"])
        self.assertEqual(serving["paired_tail_effect_status"]["status"], "descriptive")

    def test_rejects_underpowered_p99_and_tampered_raw_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            stdout_path.write_text(
                stdout_path.read_text(encoding="utf-8").replace(
                    "p99_status=descriptive", "p99_status=qualified"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "1000 samples"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            riley_line = next(line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=riley" in line)
            rows_path = Path(
                next(token.split("=", 1)[1] for token in riley_line.split() if token.startswith("request_rows_path="))
            )
            rows_path.write_text(rows_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(summary.SummaryError, "retained request rows SHA-256"):
                summary.summarize(serving_receipt=receipt_path)

    def test_replays_hashed_strict_rows_and_rejects_metric_token_frame_or_marker_tampering(self) -> None:
        def riley_rows_path(stdout_path: Path) -> Path:
            line = next(line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=riley" in line)
            return Path(
                next(token.split("=", 1)[1] for token in line.split() if token.startswith("request_rows_path="))
            )

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            rows_path = riley_rows_path(stdout_path)
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["metrics"]["token_ttft_ns"] += 1
            rows_path.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="request_rows_sha256",
                value=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(summary.SummaryError, "metrics differs from replayed"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            rows_path = riley_rows_path(stdout_path)
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["token_ids"][0] += 1
            rows_path.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="request_rows_sha256",
                value=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(summary.SummaryError, "token_ids differs from replayed"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            rows_path = riley_rows_path(stdout_path)
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["frames"][0]["data"] = rows[0]["frames"][0]["data"].replace(
                '"token_ids":[10000]', '"token_ids":[10001]', 1
            )
            rows_path.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="request_rows_sha256",
                value=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(summary.SummaryError, "cannot be replayed strictly"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="token_ttft_median_ms",
                value="8.999999",
            )
            with self.assertRaisesRegex(summary.SummaryError, "differs from R7 replay"):
                summary.summarize(serving_receipt=receipt_path)

    def test_rejects_a_hash_rebound_vllm_provenance_without_daemon_container_absence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            vllm_line = next(line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=vllm" in line)
            provenance_path = Path(
                next(token.split("=", 1)[1] for token in vllm_line.split() if token.startswith("lane_provenance_path="))
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["cleanup"]["container"]["final_state"] = "present"
            provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
            self.replace_marker_field(
                stdout_path,
                lane="vllm",
                field="lane_provenance_sha256",
                value=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(summary.SummaryError, "does not prove daemon container absence"):
                summary.summarize(serving_receipt=receipt_path)

    def test_rejects_phase_lifecycle_and_post_lane_gpu_idle_tampering(self) -> None:
        def provenance_path(stdout_path: Path, lane: str) -> Path:
            line = next(line for line in stdout_path.read_text(encoding="utf-8").splitlines() if f"lane={lane}" in line)
            return Path(
                next(
                    token.split("=", 1)[1]
                    for token in line.split()
                    if token.startswith("lane_provenance_path=")
                )
            )

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            path = provenance_path(stdout_path, "riley")
            provenance = json.loads(path.read_text(encoding="utf-8"))
            provenance["ready"]["ready_ns"] = 2
            path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="lane_provenance_sha256",
                value=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(summary.SummaryError, "outside the owned server lifecycle/readiness"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            path = provenance_path(stdout_path, "vllm")
            provenance = json.loads(path.read_text(encoding="utf-8"))
            provenance["post_lane_gpu_idle"]["compute_process_pids"] = [12345]
            path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
            self.replace_marker_field(
                stdout_path,
                lane="vllm",
                field="lane_provenance_sha256",
                value=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(summary.SummaryError, "post-lane GPU idle census differs"):
                summary.summarize(serving_receipt=receipt_path)

    def test_rejects_tampered_vllm_snapshot_and_unobserved_gpu_sampling_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            vllm_line = next(line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=vllm" in line)
            snapshot_path = Path(
                next(
                    token.split("=", 1)[1]
                    for token in vllm_line.split()
                    if token.startswith("startup_stdout_log_path=")
                )
            )
            snapshot_path.write_text(snapshot_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(summary.SummaryError, "vLLM startup stdout snapshot SHA-256"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            lines = stdout_path.read_text(encoding="utf-8").splitlines()
            vllm_line_index = next(index for index, line in enumerate(lines) if "lane=vllm" in line)
            vllm_line = lines[vllm_line_index]

            def field(name: str) -> str:
                return next(
                    token.split("=", 1)[1]
                    for token in vllm_line.split()
                    if token.startswith(name + "=")
                )

            sample_path = Path(field("whole_gpu_sample_path"))
            peak_path = Path(field("whole_gpu_peak_receipt_path"))
            provenance_path = Path(field("lane_provenance_path"))
            old_sample_sha = field("whole_gpu_sample_sha256")
            old_peak_sha = field("whole_gpu_peak_receipt_sha256")
            old_provenance_sha = field("lane_provenance_sha256")
            samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
            samples[1].update(
                started_ns=1_000_000_000,
                finished_ns=1_000_000_010,
                monotonic_ns=1_000_000_010,
            )
            samples[2].update(
                started_ns=1_100_000_000,
                finished_ns=1_100_000_010,
                monotonic_ns=1_100_000_010,
            )
            sample_path.write_text(
                "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in samples),
                encoding="utf-8",
            )
            new_sample_sha = hashlib.sha256(sample_path.read_bytes()).hexdigest()
            peak = json.loads(peak_path.read_text(encoding="utf-8"))
            peak["sample_sha256"] = new_sample_sha
            peak_path.write_text(json.dumps(peak, sort_keys=True) + "\n", encoding="utf-8")
            new_peak_sha = hashlib.sha256(peak_path.read_bytes()).hexdigest()
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["whole_gpu_memory"]["sample_sha256"] = new_sample_sha
            provenance["whole_gpu_memory"]["peak_sha256"] = new_peak_sha
            provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n", encoding="utf-8")
            new_provenance_sha = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            lines[vllm_line_index] = (
                vllm_line.replace(old_sample_sha, new_sample_sha)
                .replace(old_peak_sha, new_peak_sha)
                .replace(old_provenance_sha, new_provenance_sha)
            )
            stdout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(summary.SummaryError, "cadence has an unobserved gap"):
                summary.summarize(serving_receipt=receipt_path)

    def test_rejects_fallback_and_reused_backend_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(receipt["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(
                    "backend_resolved=" + summary.SERVING_IMPLEMENTATION_ID,
                    "backend_resolved=fallback-d64",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "request the native D128 CLI backend"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded", "succeeded"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            first = Path(receipt["timed_runs"][0]["stdout_path"]).read_text(encoding="utf-8")
            second_path = Path(receipt["timed_runs"][1]["stdout_path"])
            second = second_path.read_text(encoding="utf-8")
            first_riley_line = next(line for line in first.splitlines() if "lane=riley" in line)
            second_riley_line = next(line for line in second.splitlines() if "lane=riley" in line)
            first_backend = next(
                token.split("=", 1)[1]
                for token in first_riley_line.split()
                if token.startswith("startup_log_path=")
            )
            second_backend = next(
                token.split("=", 1)[1]
                for token in second_riley_line.split()
                if token.startswith("startup_log_path=")
            )
            second_path.write_text(second.replace(second_backend, first_backend), encoding="utf-8")
            with self.assertRaisesRegex(summary.SummaryError, "Riley startup provenance differs"):
                summary.summarize(serving_receipt=receipt_path)

    def test_rejects_mismatched_or_infeasible_batch_budget_and_model_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(receipt["timed_runs"][0]["stdout_path"])
            original = log_path.read_text(encoding="utf-8")
            log_path.write_text(
                original.replace("lane=vllm", "batch_token_budget=16 lane=vllm", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "batch_token_budget"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(receipt["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace("max_model_len=32768", "max_model_len=2048"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "max_model_len must fit"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(receipt["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace("batch_token_budget=32", "batch_token_budget=33"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "native D128 bound"):
                summary.summarize(serving_receipt=receipt_path)

    def test_rejects_digest_token_that_is_not_the_docker_image_operand(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            riley_line = next(line for line in stdout_path.read_text(encoding="utf-8").splitlines() if "lane=riley" in line)
            config_path = Path(
                next(token.split("=", 1)[1] for token in riley_line.split() if token.startswith("attempt_config_path="))
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            image = "vllm/vllm-openai@" + VLLM_IMAGE_DIGEST
            image_index = config["vllm_argv"].index(image)
            config["vllm_argv"][image_index] = "untrusted-vllm:latest"
            config["vllm_argv"].append(image)
            config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.replace_marker_field(
                stdout_path,
                lane="riley",
                field="attempt_config_sha256",
                value=config_sha,
            )
            self.replace_marker_field(
                stdout_path,
                lane="vllm",
                field="attempt_config_sha256",
                value=config_sha,
            )
            with self.assertRaisesRegex(summary.SummaryError, "first non-option operand"):
                summary.summarize(serving_receipt=receipt_path)

    def test_rejects_docker_control_or_mount_after_image_operand(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            stdout_path = Path(receipt["timed_runs"][0]["stdout_path"])
            riley_line = next(
                line
                for line in stdout_path.read_text(encoding="utf-8").splitlines()
                if "lane=riley" in line
            )
            config_path = Path(
                next(
                    token.split("=", 1)[1]
                    for token in riley_line.split()
                    if token.startswith("attempt_config_path=")
                )
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            mount_index = config["vllm_argv"].index("-v")
            mount = config["vllm_argv"][mount_index : mount_index + 2]
            del config["vllm_argv"][mount_index : mount_index + 2]
            image_index = config["vllm_argv"].index("vllm/vllm-openai@" + VLLM_IMAGE_DIGEST)
            config["vllm_argv"][image_index + 1 : image_index + 1] = mount
            config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            for lane in ("riley", "vllm"):
                self.replace_marker_field(
                    stdout_path,
                    lane=lane,
                    field="attempt_config_sha256",
                    value=config_sha,
                )
            with self.assertRaisesRegex(summary.SummaryError, r"Docker control option '-v' must occur before"):
                summary.summarize(serving_receipt=receipt_path)

    def test_docker_control_prefix_rejects_each_control_after_image(self) -> None:
        image = "vllm/vllm-openai@sha256:" + "c" * 64
        prefix = [
            "/usr/bin/docker",
            "run",
            "--name",
            "riley-n06a-vllm-timed-0001-0123456789abcdef0123",
            "--rm",
            "--gpus",
            "device=0",
            "--ipc",
            "host",
            "--network",
            "host",
            "-v",
            "/models:/model:ro",
            "--volume",
            "/scratch:/scratch:ro",
            image,
            "--model",
            "/model",
        ]
        controls = (
            ("--name", ("riley-n06a-vllm-timed-0001-0123456789abcdef0123",)),
            ("--rm", ()),
            ("--gpus", ("device=0",)),
            ("--ipc", ("host",)),
            ("--network", ("host",)),
            ("-v", ("/models:/model:ro",)),
            ("--volume", ("/scratch:/scratch:ro",)),
        )
        for flag, values in controls:
            with self.subTest(flag=flag):
                argv = list(prefix)
                flag_index = argv.index(flag)
                del argv[flag_index : flag_index + 1 + len(values)]
                image_index = argv.index(image)
                argv[image_index + 1 : image_index + 1] = [flag, *values]
                resolved_image_index = summary._docker_run_image_operand_index(
                    argv, label="fixture Docker argv"
                )
                self.assertEqual(argv[resolved_image_index], image)
                with self.assertRaisesRegex(summary.SummaryError, "Docker control option"):
                    summary._require_docker_controls_before_image(
                        argv,
                        image_operand_index=resolved_image_index,
                        label="fixture Docker argv",
                    )

    def test_rejects_tampered_startup_snapshot_and_nonalternating_pair_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            log_path = Path(receipt["timed_runs"][0]["stdout_path"])
            line = next(line for line in log_path.read_text(encoding="utf-8").splitlines() if "lane=riley" in line)
            startup_path = Path(
                next(token.split("=", 1)[1] for token in line.split() if token.startswith("startup_log_path="))
            )
            startup_path.write_text(
                startup_path.read_text(encoding="utf-8").replace(
                    summary.SERVING_IMPLEMENTATION_ID,
                    "riley.cuda.ragged-paged-attention.d64",
                ),
                encoding="utf-8",
            )
            old_sha = next(token.split("=", 1)[1] for token in line.split() if token.startswith("startup_log_sha256="))
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(
                    old_sha,
                    hashlib.sha256(startup_path.read_bytes()).hexdigest(),
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "resolved_ragged_backend differs"):
                summary.summarize(serving_receipt=receipt_path)

        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded", "succeeded"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            second_path = Path(receipt["timed_runs"][1]["stdout_path"])
            second_path.write_text(
                second_path.read_text(encoding="utf-8").replace("pair_order=vllm-riley", "pair_order=riley-vllm"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "pair_order differs"):
                summary.summarize(serving_receipt=receipt_path)

    def test_invalid_marker_and_cli_do_not_modify_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            before = receipt_path.read_bytes()
            receipt = json.loads(before)
            log_path = Path(receipt["timed_runs"][0]["stdout_path"])
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(" graph_capture_enabled=false", "", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(summary.SummaryError, "missing graph_capture_enabled"):
                summary.summarize(serving_receipt=receipt_path)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = summary.main(["--serving-receipt", str(receipt_path)])
            self.assertEqual(result, 2)
            self.assertIn("n03b_n06a_d128_repeat_summary: error:", stderr.getvalue())
            self.assertEqual(receipt_path.read_bytes(), before)

    def test_rejects_unbounded_bootstrap_work_before_reading_serving_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = self.make_serving_receipt(Path(temporary), statuses=("succeeded",))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["configuration"]["bootstrap_resamples"] = summary.MAX_BOOTSTRAP_RESAMPLES + 1
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(summary.SummaryError, "bounded shared-host reader limit"):
                summary.summarize(serving_receipt=receipt_path)


if __name__ == "__main__":
    unittest.main()
