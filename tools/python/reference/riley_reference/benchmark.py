"""Correctness-first matrix runner for the Hugging Face eager reference lane."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .constants import (
    DTYPE,
    MAX_CONTEXT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    NVIDIA_ML_PY_VERSION,
    PSUTIL_VERSION,
    PRIMARY_ENVIRONMENT_ID,
    PRIMARY_GPU_COMPUTE_CAPABILITY,
    PRIMARY_GPU_NAME,
    PRIMARY_NVIDIA_DRIVER_VERSION,
    PRIMARY_RAM_BYTES,
    RUNTIME_DEPENDENCY_CLASS,
    SAFETENSORS_VERSION,
    SEMANTIC_CLASS,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
)
from .fixture import BackendMetadata, FixtureError, PromptRecord, load_prompts, utc_now

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_EMPTY_TOKEN_IDS_SHA256 = hashlib.sha256(b"").hexdigest()
_EXPECTED_CACHE_POLICY = {
    "cold_scope": "process-and-model-state-only",
    "dependency_preparation": "uv-sync-frozen-offline",
    "uv_version": "uv 0.12.5 (x86_64-unknown-linux-gnu)",
    "uv_linux_x86_64_sha256": "b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46",
    "uv_python": "3.13.15",
    "uv_python_downloads": "never",
    "python_linux_x86_64_sha256": "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866",
    "python_dont_write_bytecode": "1",
    "cuda_cache_maxsize": "4294967296",
    "python_hash_seed": "0",
    "tokenizers_parallelism": "false",
    "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "telemetry_environment": {
        "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1",
    },
    "reuse_external_disk_caches_across_independent_runs": True,
    "external_cache_environment": [
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "HF_HOME",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ],
    "offline_environment": {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    },
    "lane_prime_cells": {
        lane: [
            {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32, "warm_state": "warm"},
            {"concurrency": 1, "prompt_tokens": 4096, "output_tokens": 128, "warm_state": "warm"},
            {"concurrency": 8, "prompt_tokens": 128, "output_tokens": 32, "warm_state": "warm"},
        ]
        for lane in ("hf-transformers", "vllm")
    },
    "reject_measured_cache_mutation": True,
}
_TOP_LEVEL_KEYS = {
    "contract_version",
    "trial_id",
    "run_id",
    "trial_index",
    "recorded_at_utc",
    "scope",
    "matrix_id",
    "matrix_sha256",
    "prompts_sha256",
    "lane_manifest_sha256",
    "environment_id",
    "semantic_class",
    "implementation_id",
    "reference_implementation",
    "runtime_dependency_class",
    "approximation_enabled",
    "error_budget",
    "correctness_gate_id",
    "correctness_report_sha256",
    "seed",
    "warm_state",
    "model_id",
    "model_revision",
    "engine_revision",
    "dtype",
    "environment",
    "provenance",
    "status",
    "failure_reason",
    "failure_count",
    "workload",
    "microbenchmark",
    "metrics",
    "requests",
    "speculative",
    "sparse_attention",
    "quantization",
}


class BenchmarkMeasurement(Protocol):
    prompt_token_counts: tuple[int, ...]
    prompt_token_ids_sha256: tuple[str, ...]
    output_token_counts: tuple[int, ...]
    generated_token_ids_sha256: tuple[str, ...]
    ttft_seconds: float
    itl_seconds: tuple[float, ...]
    end_to_end_seconds: float
    output_tokens_per_second: float
    cpu_utilization_percent: float
    gpu_utilization_percent: float
    peak_gpu_memory_bytes: int


class BenchmarkBackend(Protocol):
    metadata: BackendMetadata

    def environment(self) -> dict[str, object]: ...

    def prompt_token_ids_sha256(
        self, texts: tuple[str, ...], *, prompt_tokens: int
    ) -> tuple[str, ...]: ...

    def benchmark_batch(
        self,
        texts: tuple[str, ...],
        *,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> BenchmarkMeasurement: ...


def _load_json(path: Path, label: str) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FixtureError(f"{label} {path} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FixtureError(f"{path} must be an object")
    return value


def _int_axis(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise FixtureError(f"{path} must be a non-empty array")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise FixtureError(f"{path} must contain positive integers")
    return tuple(value)


def _validate_matrix(matrix: Mapping[str, object]) -> dict[str, object]:
    if matrix.get("contract_version") != "1.0.0":
        raise FixtureError("matrix.contract_version must be '1.0.0'")
    if matrix.get("benchmark_scope") != "end-to-end":
        raise FixtureError("reference runner only supports end-to-end matrices")
    if matrix.get("tokenization") != {
        "input": "pretokenized",
        "latency_included": False,
        "add_special_tokens": True,
    }:
        raise FixtureError("matrix.tokenization differs from the canonical contract")
    hardware = _object(matrix.get("primary_hardware"), "matrix.primary_hardware")
    expected_hardware = {
        "gpu_model": PRIMARY_GPU_NAME,
        "compute_capability": PRIMARY_GPU_COMPUTE_CAPABILITY,
        "dtype": DTYPE,
    }
    if hardware != expected_hardware:
        raise FixtureError("matrix primary hardware differs from the PR 01 contract")
    model = _object(matrix.get("model"), "matrix.model")
    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise FixtureError(f"matrix.model.{key} must be {expected!r}")
    weights_sha256 = model.get("weights_sha256")
    if not isinstance(weights_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", weights_sha256):
        raise FixtureError("matrix.model.weights_sha256 must be lowercase SHA-256")
    axes = _object(matrix.get("axes"), "matrix.axes")
    concurrency = _int_axis(axes.get("concurrency"), "matrix.axes.concurrency")
    prompt_tokens = _int_axis(axes.get("prompt_tokens"), "matrix.axes.prompt_tokens")
    output_tokens = _int_axis(axes.get("output_tokens"), "matrix.axes.output_tokens")
    warm_states = axes.get("warm_state")
    if (
        not isinstance(warm_states, list)
        or not warm_states
        or any(state not in {"cold", "warm"} for state in warm_states)
    ):
        raise FixtureError("matrix.axes.warm_state must contain cold/warm")
    sampling = axes.get("sampling")
    if sampling != [
        {
            "id": "greedy",
            "strategy": "greedy",
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "seed": None,
            "ignore_eos": True,
            "fixed_output_length": True,
        }
    ]:
        raise FixtureError("reference benchmark supports only the canonical greedy axis")
    if axes.get("approximation_enabled") != [False]:
        raise FixtureError("reference benchmark requires approximation disabled")
    if matrix.get("cache_policy") != _EXPECTED_CACHE_POLICY:
        raise FixtureError("matrix.cache_policy differs from the Gate A contract")
    measurement = _object(matrix.get("measurement"), "matrix.measurement")
    independent_runs = measurement.get("independent_runs")
    if (
        isinstance(independent_runs, bool)
        or not isinstance(independent_runs, int)
        or independent_runs <= 0
    ):
        raise FixtureError("matrix.measurement.independent_runs must be positive")
    cold = _object(measurement.get("cold"), "matrix.measurement.cold")
    warm = _object(measurement.get("warm"), "matrix.measurement.warm")
    measured: dict[str, int] = {}
    for state, config in (("cold", cold), ("warm", warm)):
        expected_lifecycle = {
            "fresh_process_per_independent_run": True,
            "reset_model_state_per_independent_run": True,
            "reuse_model_within_run": state == "warm",
        }
        for key, expected in expected_lifecycle.items():
            if config.get(key) is not expected:
                raise FixtureError(
                    f"matrix.measurement.{state}.{key} must be {expected!r}"
                )
        count = config.get("warmup_iterations")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise FixtureError(
                f"matrix.measurement.{state}.warmup_iterations must be non-negative"
            )
        iterations = config.get("measured_iterations_per_run")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise FixtureError(
                f"matrix.measurement.{state}.measured_iterations_per_run must be positive"
            )
        measured[state] = iterations
    return {
        "matrix_id": matrix.get("matrix_id"),
        "model": model,
        "lane_manifests": matrix.get("lane_manifests"),
        "concurrency": concurrency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "warm_states": tuple(warm_states),
        "independent_runs": independent_runs,
        "warmups": {
            "cold": cold["warmup_iterations"],
            "warm": warm["warmup_iterations"],
        },
        "measured_iterations": measured,
    }


def _resolve_repo_relative(path_text: str, matrix_path: Path) -> Path:
    direct = Path(path_text)
    if direct.is_file():
        return direct
    for parent in matrix_path.resolve().parents:
        candidate = parent / path_text
        if candidate.is_file():
            return candidate
    raise FixtureError(f"cannot resolve matrix path {path_text!r}")


def _load_hf_lane(
    matrix_contract: Mapping[str, object], matrix_path: Path
) -> tuple[dict[str, object], str, Path]:
    paths = matrix_contract["lane_manifests"]
    if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
        raise FixtureError("matrix.lane_manifests must be an array of paths")
    for path_text in paths:
        lane_path = _resolve_repo_relative(path_text, matrix_path)
        lane, lane_sha256 = _load_json(lane_path, "lane manifest")
        if lane.get("lane_id") == "hf-transformers":
            expected = {
                "implementation_id": "hf-transformers-eager",
                "reference_implementation": "hf-transformers-eager",
                "runtime_dependency_class": RUNTIME_DEPENDENCY_CLASS,
                "semantic_class": SEMANTIC_CLASS,
                "dtype": DTYPE,
            }
            for key, value in expected.items():
                if lane.get(key) != value:
                    raise FixtureError(f"lane.{key} must be {value!r}")
            engine = _object(lane.get("engine"), "lane.engine")
            if engine.get("version") != TRANSFORMERS_VERSION:
                raise FixtureError("lane Transformers version differs from pinned backend")
            dependencies = _object(engine.get("dependencies"), "lane.engine.dependencies")
            if dependencies != {
                "nvidia-ml-py": NVIDIA_ML_PY_VERSION,
                "psutil": PSUTIL_VERSION,
                "safetensors": SAFETENSORS_VERSION,
                "torch": TORCH_VERSION,
                "transformers": TRANSFORMERS_VERSION,
            }:
                raise FixtureError("lane dependency pins differ from reference project")
            return lane, lane_sha256, lane_path
    raise FixtureError("matrix does not contain an hf-transformers lane manifest")


def _git_provenance(start: Path) -> dict[str, object]:
    try:
        repository_root = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "-C", repository_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_output = subprocess.run(
            [
                "git",
                "-C",
                repository_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
                "--",
                ".",
                ":(top,exclude)benchmarks/results",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise FixtureError(f"cannot capture Git provenance: {error}") from error
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise FixtureError("Git revision is not a 40-character SHA")
    return {"git_revision": revision, "git_dirty": bool(dirty_output)}


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise FixtureError("benchmark timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _safe_run_id(value: str | None) -> str:
    if value is None:
        raise FixtureError(
            "run_id is required and must be shared by every cell in one independent run"
        )
    if not _ID_RE.fullmatch(value):
        raise FixtureError("run_id must match ^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    return value


def _select_prompts(
    prompts: Sequence[PromptRecord], concurrency: int, prompt_tokens: int
) -> tuple[PromptRecord, ...]:
    matching = [
        prompt for prompt in prompts if prompt.target_prompt_tokens == prompt_tokens
    ]
    pool: Sequence[PromptRecord] = matching or prompts
    # File order is canonical; logical requests stay distinct when cycling is needed.
    return tuple(pool[index % len(pool)] for index in range(concurrency))


def _null_optimization_fields() -> tuple[dict[str, None], dict[str, None], dict[str, None]]:
    return (
        {
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
        {
            "selected_pages": None,
            "total_pages": None,
            "page_metadata_bytes": None,
            "page_bound_time_ms": None,
            "omitted_mass_bound": None,
            "exact_fallback_rate": None,
        },
        {
            "weight_format": None,
            "activation_format": None,
            "kv_format": None,
            "calibration_revision": None,
            "transform_runtime_ms": None,
            "weight_bytes": None,
            "kv_bytes": None,
            "gemm_throughput_tflops": None,
        },
    )


def _base_row(
    *,
    trial_id: str,
    run_id: str,
    trial_index: int,
    recorded_at: str,
    matrix_contract: Mapping[str, object],
    matrix_sha256: str,
    prompts_sha256: str,
    lane: Mapping[str, object],
    lane_sha256: str,
    environment: Mapping[str, object],
    provenance: Mapping[str, object],
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
    warm_state: str,
) -> dict[str, object]:
    speculative, sparse_attention, quantization = _null_optimization_fields()
    return {
        "contract_version": "1.0.0",
        "trial_id": trial_id,
        "run_id": run_id,
        "trial_index": trial_index,
        "recorded_at_utc": recorded_at,
        "scope": "end-to-end",
        "matrix_id": matrix_contract["matrix_id"],
        "matrix_sha256": matrix_sha256,
        "prompts_sha256": prompts_sha256,
        "lane_manifest_sha256": lane_sha256,
        "environment_id": PRIMARY_ENVIRONMENT_ID,
        "semantic_class": lane["semantic_class"],
        "implementation_id": lane["implementation_id"],
        "reference_implementation": lane["reference_implementation"],
        "runtime_dependency_class": lane["runtime_dependency_class"],
        "approximation_enabled": False,
        "error_budget": None,
        "correctness_gate_id": None,
        "correctness_report_sha256": None,
        "seed": None,
        "warm_state": warm_state,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "engine_revision": _object(lane["engine"], "lane.engine")["revision"],
        "dtype": DTYPE,
        "environment": dict(environment),
        "provenance": dict(provenance),
        "status": "success",
        "failure_reason": None,
        "failure_count": 0,
        "workload": {
            "concurrency": concurrency,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "sampling_id": "greedy",
            "warm_state": warm_state,
        },
        "microbenchmark": None,
        "metrics": {},
        "requests": [],
        "speculative": speculative,
        "sparse_attention": sparse_attention,
        "quantization": quantization,
    }


def _success_row(
    base: dict[str, object],
    measurement: BenchmarkMeasurement,
    selected: Sequence[PromptRecord],
    *,
    model_load_ms: float,
    output_tokens: int,
    prompt_hashes: Sequence[str],
) -> dict[str, object]:
    concurrency = len(selected)
    if (
        len(measurement.prompt_token_counts) != concurrency
        or len(measurement.prompt_token_ids_sha256) != concurrency
        or len(measurement.output_token_counts) != concurrency
        or len(measurement.generated_token_ids_sha256) != concurrency
    ):
        raise FixtureError("backend returned request count inconsistent with concurrency")
    if any(count != output_tokens for count in measurement.output_token_counts):
        raise FixtureError("benchmark backend did not honor fixed output length")
    expected_prompt_tokens = _object(base["workload"], "row.workload")["prompt_tokens"]
    if any(
        count != expected_prompt_tokens for count in measurement.prompt_token_counts
    ):
        raise FixtureError("benchmark backend did not honor fixed prompt length")
    if tuple(measurement.prompt_token_ids_sha256) != tuple(prompt_hashes):
        raise FixtureError("benchmark backend changed pretokenized request identity")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in measurement.generated_token_ids_sha256
    ):
        raise FixtureError("benchmark backend returned invalid generated-token hashes")
    if any(not math.isfinite(value) or value < 0 for value in (
        measurement.ttft_seconds,
        measurement.end_to_end_seconds,
        measurement.output_tokens_per_second,
    )):
        raise FixtureError("backend returned invalid benchmark timing")
    if measurement.ttft_seconds > measurement.end_to_end_seconds:
        raise FixtureError("backend TTFT exceeds end-to-end time")
    if len(measurement.itl_seconds) != max(0, output_tokens - 1) or any(
        not math.isfinite(value) or value < 0 for value in measurement.itl_seconds
    ):
        raise FixtureError("backend returned invalid inter-token intervals")
    if (
        not math.isfinite(measurement.cpu_utilization_percent)
        or measurement.cpu_utilization_percent < 0
    ):
        raise FixtureError("backend returned invalid CPU utilization")
    if (
        not math.isfinite(measurement.gpu_utilization_percent)
        or measurement.gpu_utilization_percent < 0
        or measurement.gpu_utilization_percent > 100
    ):
        raise FixtureError("backend returned invalid GPU utilization")
    if (
        isinstance(measurement.peak_gpu_memory_bytes, bool)
        or not isinstance(measurement.peak_gpu_memory_bytes, int)
        or measurement.peak_gpu_memory_bytes < 0
    ):
        raise FixtureError("backend returned invalid peak GPU memory")
    mean_itl_ms = (
        1000.0 * math.fsum(measurement.itl_seconds) / len(measurement.itl_seconds)
        if measurement.itl_seconds
        else 0.0
    )
    base["metrics"] = {
        "model_load_ms": model_load_ms,
        "batch_wall_ms": 1000.0 * measurement.end_to_end_seconds,
        "output_tokens_per_second": measurement.output_tokens_per_second,
        "cpu_utilization_percent": measurement.cpu_utilization_percent,
        "gpu_utilization_percent": measurement.gpu_utilization_percent,
        "peak_vram_bytes": measurement.peak_gpu_memory_bytes,
    }
    trial_id = str(base["trial_id"])
    base["requests"] = [
        {
            "request_id": f"{trial_id}:request-{index + 1}",
            "prompt_id": prompt.prompt_id,
            "prompt_token_ids_sha256": measurement.prompt_token_ids_sha256[index],
            "generated_token_ids_sha256": measurement.generated_token_ids_sha256[index],
            "status": "success",
            "failure_reason": None,
            "prompt_tokens": measurement.prompt_token_counts[index],
            "requested_output_tokens": output_tokens,
            "generated_tokens": measurement.output_token_counts[index],
            "ttft_ms": 1000.0 * measurement.ttft_seconds,
            "end_to_end_ms": 1000.0 * measurement.end_to_end_seconds,
            "mean_tpot_ms": mean_itl_ms,
            "itl_ms": [1000.0 * value for value in measurement.itl_seconds],
        }
        for index, prompt in enumerate(selected)
    ]
    return base


def _failure_row(
    base: dict[str, object],
    selected: Sequence[PromptRecord],
    *,
    prompt_tokens: int,
    output_tokens: int,
    model_load_ms: float,
    prompt_hashes: Sequence[str],
    error: Exception,
) -> dict[str, object]:
    reason = f"{type(error).__name__}: {error}"[:1000]
    base["status"] = "failure"
    base["failure_reason"] = reason
    base["failure_count"] = len(selected)
    base["metrics"] = {
        "model_load_ms": model_load_ms,
        "batch_wall_ms": None,
        "output_tokens_per_second": None,
        "cpu_utilization_percent": None,
        "gpu_utilization_percent": None,
        "peak_vram_bytes": None,
    }
    trial_id = str(base["trial_id"])
    base["requests"] = [
        {
            "request_id": f"{trial_id}:request-{index + 1}",
            "prompt_id": prompt.prompt_id,
            "prompt_token_ids_sha256": prompt_hashes[index],
            "generated_token_ids_sha256": _EMPTY_TOKEN_IDS_SHA256,
            "status": "failure",
            "failure_reason": reason,
            "prompt_tokens": prompt_tokens,
            "requested_output_tokens": output_tokens,
            "generated_tokens": 0,
            "ttft_ms": None,
            "end_to_end_ms": None,
            "mean_tpot_ms": None,
            "itl_ms": None,
        }
        for index, prompt in enumerate(selected)
    ]
    return base


def validate_result_row(row: Mapping[str, object]) -> None:
    """Cheap producer-side checks for schema-required keys and invariants."""

    if set(row) != _TOP_LEVEL_KEYS:
        raise FixtureError("benchmark result row top-level keys differ from schema v1")
    for digest_key in ("matrix_sha256", "prompts_sha256", "lane_manifest_sha256"):
        if not isinstance(row[digest_key], str) or not re.fullmatch(
            r"[0-9a-f]{64}", row[digest_key]
        ):
            raise FixtureError(f"benchmark row {digest_key} is invalid")
    status = row["status"]
    failure_count = row["failure_count"]
    if status == "success":
        if row["failure_reason"] is not None or failure_count != 0:
            raise FixtureError("successful row has failure metadata")
    elif status == "failure":
        if not isinstance(row["failure_reason"], str) or not failure_count:
            raise FixtureError("failed row lacks failure metadata")
    else:
        raise FixtureError("benchmark row status is invalid")
    workload = _object(row["workload"], "row.workload")
    if row["warm_state"] != workload.get("warm_state"):
        raise FixtureError("row warm_state differs from workload")
    requests = row["requests"]
    if not isinstance(requests, list) or len(requests) != workload.get("concurrency"):
        raise FixtureError("row request count differs from concurrency")
    for request in requests:
        request_object = _object(request, "row.requests[]")
        for digest_key in (
            "prompt_token_ids_sha256",
            "generated_token_ids_sha256",
        ):
            digest = request_object.get(digest_key)
            if not isinstance(digest, str) or not re.fullmatch(
                r"[0-9a-f]{64}", digest
            ):
                raise FixtureError(f"request {digest_key} is invalid")
        generated_tokens = request_object.get("generated_tokens")
        itl_ms = request_object.get("itl_ms")
        mean_tpot_ms = request_object.get("mean_tpot_ms")
        if request_object.get("status") == "success":
            if not isinstance(generated_tokens, int) or isinstance(
                generated_tokens, bool
            ):
                raise FixtureError("successful request generated_tokens is invalid")
            if not isinstance(itl_ms, list) or len(itl_ms) != max(
                0, generated_tokens - 1
            ):
                raise FixtureError("successful request ITL count is invalid")
            expected_mean = math.fsum(float(value) for value in itl_ms) / len(
                itl_ms
            ) if itl_ms else 0.0
            if not isinstance(mean_tpot_ms, (int, float)) or not math.isclose(
                float(mean_tpot_ms), expected_mean, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise FixtureError("successful request mean TPOT differs from ITLs")


def _filtered(values: Sequence[Any], selected: Any, name: str) -> tuple[Any, ...]:
    if selected is None:
        return tuple(values)
    if selected not in values:
        raise FixtureError(f"requested {name}={selected!r} is absent from matrix")
    return (selected,)


def _write_artifact(
    result_dir: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    independent_run_index: int,
    matrix_sha256: str,
    prompts_sha256: str,
    lane_sha256: str,
) -> None:
    try:
        result_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FixtureError(f"refusing to reuse result directory: {result_dir}") from error
    try:
        with (result_dir / "raw.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                json.dump(row, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
        metadata = {
            "contract_version": "1.0.0",
            "run_id": run_id,
            "independent_run_index": independent_run_index,
            "row_count": len(rows),
            "matrix_sha256": matrix_sha256,
            "prompts_sha256": prompts_sha256,
            "lane_manifest_sha256": lane_sha256,
        }
        with (result_dir / "metadata.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with (result_dir / "README.md").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                "# Hugging Face eager reference benchmark\n\n"
                f"Run `{run_id}`, independent run index {independent_run_index}; "
                f"{len(rows)} result-schema-v1 rows. Greedy decode ignores EOS "
                "and emits the matrix's fixed output length. Each request records "
                "u32-le SHA-256 identities for both prompt and generated tokens. "
                "Metrics use 10 ms device-wide NVML GPU samples and recursive "
                "process-tree CPU-time deltas. "
                "Invoke each independent trial in a fresh process and a new result "
                "directory.\n"
            )
    except BaseException:
        # Preserve a partial directory for diagnosis; append-only artifacts are never replaced.
        raise


def run_benchmark(
    *,
    matrix_path: Path,
    prompts_path: Path,
    result_dir: Path,
    backend_factory: Callable[..., BenchmarkBackend],
    device: str,
    local_files_only: bool,
    run_index: int,
    run_id: str | None,
    warm_state_filter: str | None,
    concurrency_filter: int | None,
    prompt_tokens_filter: int | None,
    output_tokens_filter: int | None,
    now: Callable[[], datetime] = utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Run one matrix cell in one process and create an append-only artifact."""

    if result_dir.exists():
        raise FixtureError(f"refusing to reuse result directory: {result_dir}")
    if any(
        value is None
        for value in (
            warm_state_filter,
            concurrency_filter,
            prompt_tokens_filter,
            output_tokens_filter,
        )
    ):
        raise FixtureError(
            "benchmark requires warm-state, concurrency, prompt-tokens, and "
            "output-tokens filters so each cell runs in an isolated process"
        )
    matrix, matrix_sha256 = _load_json(matrix_path, "matrix")
    matrix_contract = _validate_matrix(matrix)
    if run_index > matrix_contract["independent_runs"]:
        raise FixtureError(
            f"run_index {run_index} exceeds matrix independent_runs "
            f"{matrix_contract['independent_runs']}"
        )
    prompts, prompts_sha256 = load_prompts(prompts_path)
    lane, lane_sha256, _ = _load_hf_lane(matrix_contract, matrix_path)
    provenance = _git_provenance(matrix_path.resolve().parent)
    timestamp = now()
    resolved_run_id = _safe_run_id(run_id)
    recorded_at = _utc_text(timestamp)

    load_start = timer()
    backend = backend_factory(device=device, local_files_only=local_files_only)
    model_load_ms = 1000.0 * (timer() - load_start)
    if model_load_ms < 0:
        raise FixtureError("monotonic timer moved backwards during model load")
    environment = backend.environment()
    if environment.get("gpu_model") != PRIMARY_GPU_NAME or environment.get(
        "compute_capability"
    ) != PRIMARY_GPU_COMPUTE_CAPABILITY:
        raise FixtureError("runtime GPU differs from the primary matrix contract")
    if environment.get("nvidia_driver_version") != PRIMARY_NVIDIA_DRIVER_VERSION:
        raise FixtureError("runtime NVIDIA driver differs from the primary environment")
    if environment.get("ram_bytes") != PRIMARY_RAM_BYTES:
        raise FixtureError("runtime RAM differs from the primary environment")
    if "Ubuntu 22.04" not in str(environment.get("os")):
        raise FixtureError("runtime OS differs from the primary environment")
    if "i7-13700K" not in str(environment.get("cpu_model")):
        raise FixtureError("runtime CPU differs from the primary environment")
    if backend.metadata.torch_version != TORCH_VERSION or (
        backend.metadata.transformers_version != TRANSFORMERS_VERSION
    ):
        raise FixtureError("loaded backend dependency versions differ from lane manifest")
    if backend.metadata.weights_sha256 != matrix_contract["model"]["weights_sha256"]:
        raise FixtureError("loaded model weight checksum differs from matrix contract")

    concurrencies = _filtered(
        matrix_contract["concurrency"], concurrency_filter, "concurrency"
    )
    prompt_lengths = _filtered(
        matrix_contract["prompt_tokens"], prompt_tokens_filter, "prompt_tokens"
    )
    output_lengths = _filtered(
        matrix_contract["output_tokens"], output_tokens_filter, "output_tokens"
    )
    warm_states = _filtered(
        matrix_contract["warm_states"], warm_state_filter, "warm_state"
    )

    rows: list[dict[str, object]] = []
    for warm_state in warm_states:
        for prompt_tokens in prompt_lengths:
            for output_tokens in output_lengths:
                if prompt_tokens + output_tokens > MAX_CONTEXT_TOKENS:
                    raise FixtureError("matrix cell exceeds model context")
                for concurrency in concurrencies:
                    selected = _select_prompts(
                        prompts, concurrency, prompt_tokens
                    )
                    texts = tuple(prompt.text for prompt in selected)
                    prompt_hashes = backend.prompt_token_ids_sha256(
                        texts, prompt_tokens=prompt_tokens
                    )
                    if len(prompt_hashes) != concurrency or any(
                        not re.fullmatch(r"[0-9a-f]{64}", digest)
                        for digest in prompt_hashes
                    ):
                        raise FixtureError("backend returned invalid pretokenized hashes")
                    warmup_error: Exception | None = None
                    try:
                        for _ in range(matrix_contract["warmups"][warm_state]):
                            backend.benchmark_batch(
                                texts,
                                prompt_tokens=prompt_tokens,
                                max_new_tokens=output_tokens,
                            )
                    except Exception as error:
                        warmup_error = error
                    for trial_index in range(
                        1, matrix_contract["measured_iterations"][warm_state] + 1
                    ):
                        trial_id = (
                            f"{resolved_run_id}:{warm_state}:c{concurrency}:"
                            f"p{prompt_tokens}:o{output_tokens}:i{trial_index}"
                        )
                        base = _base_row(
                            trial_id=trial_id,
                            run_id=resolved_run_id,
                            trial_index=trial_index,
                            recorded_at=recorded_at,
                            matrix_contract=matrix_contract,
                            matrix_sha256=matrix_sha256,
                            prompts_sha256=prompts_sha256,
                            lane=lane,
                            lane_sha256=lane_sha256,
                            environment=environment,
                            provenance=provenance,
                            concurrency=concurrency,
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            warm_state=warm_state,
                        )
                        try:
                            if warmup_error is not None:
                                raise warmup_error
                            measurement = backend.benchmark_batch(
                                texts,
                                prompt_tokens=prompt_tokens,
                                max_new_tokens=output_tokens,
                            )
                            row = _success_row(
                                base,
                                measurement,
                                selected,
                                model_load_ms=model_load_ms,
                                output_tokens=output_tokens,
                                prompt_hashes=prompt_hashes,
                            )
                        except Exception as error:
                            row = _failure_row(
                                base,
                                selected,
                                prompt_tokens=prompt_tokens,
                                output_tokens=output_tokens,
                                model_load_ms=model_load_ms,
                                prompt_hashes=prompt_hashes,
                                error=error,
                            )
                        validate_result_row(row)
                        rows.append(row)

    _write_artifact(
        result_dir,
        rows,
        run_id=resolved_run_id,
        independent_run_index=run_index,
        matrix_sha256=matrix_sha256,
        prompts_sha256=prompts_sha256,
        lane_sha256=lane_sha256,
    )
    return len(rows)
