"""Strict single-cell vLLM 0.27.1 benchmark adapter.

The module has no import-time dependency on vLLM, PyTorch, or Hugging Face Hub.
Those packages are loaded only by :func:`load_default_backend`, after CLI and
contract validation have completed.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


CONTRACT_VERSION = "1.0.0"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_WEIGHTS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
MODEL_CONFIG_SHA256 = "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
TOKENIZER_FILES_SHA256 = {
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}
TOKENIZER_SHA256 = "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db"
MAX_CONTEXT_TOKENS = 8192
VLLM_VERSION = "0.27.1"
ENGINE_REVISION = "vllm-0.27.1"
PRIMARY_ENVIRONMENT_ID = "rtx4090-ubuntu22-driver580-v1"
PRIMARY_GPU_NAME = "NVIDIA GeForce RTX 4090"
PRIMARY_COMPUTE_CAPABILITY = "8.9"
PRIMARY_DRIVER_VERSION = "580.173.02"
PRIMARY_RAM_BYTES = 67_185_598_464
DTYPE = "bf16"
ENGINE_TIMING_SANITY_TOLERANCE_SECONDS = 0.050

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

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
    "correctness_gate_id",
    "correctness_report_sha256",
    "implementation_id",
    "reference_implementation",
    "runtime_dependency_class",
    "approximation_enabled",
    "error_budget",
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


class AdapterError(ValueError):
    """The lane cannot produce a comparable contract result."""


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str
    target_prompt_tokens: int | None


@dataclass(frozen=True)
class BackendMetadata:
    engine_version: str
    model_revision: str
    weights_sha256: str
    local_files_only: bool


@dataclass(frozen=True)
class RequestMeasurement:
    generated_tokens: int
    generated_token_ids: tuple[int, ...]
    ttft_seconds: float
    end_to_end_seconds: float
    itl_seconds: tuple[float, ...]
    mean_tpot_seconds: float


@dataclass(frozen=True)
class BatchMeasurement:
    requests: tuple[RequestMeasurement, ...]
    wall_seconds: float
    output_tokens_per_second: float
    cpu_utilization_percent: float | None
    gpu_utilization_percent: float | None
    peak_gpu_memory_bytes: int | None


@dataclass(frozen=True)
class ObservabilityMeasurement:
    cpu_utilization_percent: float
    gpu_utilization_percent: float
    peak_gpu_memory_bytes: int


class ObservabilitySampler(Protocol):
    def start(self) -> None: ...

    def stop(self, *, wall_seconds: float) -> ObservabilityMeasurement: ...


class Backend(Protocol):
    metadata: BackendMetadata

    def environment(self) -> dict[str, object]: ...

    def materialize_token_ids(
        self, texts: Sequence[str], *, prompt_tokens: int
    ) -> tuple[tuple[int, ...], ...]: ...

    def generate_batch(
        self,
        token_rows: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        timer: Callable[[], float],
    ) -> BatchMeasurement: ...


class NvmlProcessTreeSampler:
    """Sample device-wide NVML metrics and CPU time for the full process tree."""

    def __init__(
        self,
        *,
        nvml_module: object,
        psutil_module: object,
        device_index: int = 0,
        sample_interval_seconds: float = 0.01,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise AdapterError("observability sample interval must be positive")
        self._nvml = nvml_module
        self._psutil = psutil_module
        self._sample_interval_seconds = sample_interval_seconds
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_error: Exception | None = None
        self._memory_samples: list[int] = []
        self._utilization_samples: list[float] = []
        self._cpu_start: dict[tuple[int, float], float] = {}
        self._cpu_latest: dict[tuple[int, float], float] = {}
        try:
            self._nvml.nvmlInit()
            self._device_handle = self._nvml.nvmlDeviceGetHandleByIndex(
                device_index
            )
            self._root_process = self._psutil.Process(os.getpid())
        except Exception as error:
            raise AdapterError(
                f"cannot initialize NVML/process-tree sampler: {error}"
            ) from error

    @staticmethod
    def _cpu_seconds(process: object) -> tuple[tuple[int, float], float]:
        pid = getattr(process, "pid", None)
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise AdapterError("psutil returned an invalid process ID")
        created = float(process.create_time())
        times = process.cpu_times()
        seconds = float(times.user) + float(times.system)
        if not math.isfinite(created) or not math.isfinite(seconds) or seconds < 0:
            raise AdapterError("psutil returned invalid process CPU times")
        return (pid, created), seconds

    def _cpu_snapshot(self) -> dict[tuple[int, float], float]:
        try:
            processes = [
                self._root_process,
                *self._root_process.children(recursive=True),
            ]
        except Exception as error:
            raise AdapterError(f"cannot enumerate benchmark process tree: {error}") from error
        snapshot: dict[tuple[int, float], float] = {}
        for process in processes:
            try:
                identity, seconds = self._cpu_seconds(process)
            except Exception:
                # A worker may exit between process-tree enumeration and observation.
                continue
            snapshot[identity] = seconds
        if not snapshot:
            raise AdapterError("benchmark process tree has no observable CPU times")
        return snapshot

    def _sample_once(self) -> None:
        try:
            memory = int(
                self._nvml.nvmlDeviceGetMemoryInfo(self._device_handle).used
            )
            utilization = float(
                self._nvml.nvmlDeviceGetUtilizationRates(self._device_handle).gpu
            )
        except Exception as error:
            raise AdapterError(f"cannot sample primary GPU with NVML: {error}") from error
        if memory < 0:
            raise AdapterError("NVML returned negative used GPU memory")
        if (
            not math.isfinite(utilization)
            or utilization < 0
            or utilization > 100
        ):
            raise AdapterError("NVML returned invalid GPU utilization")
        cpu_snapshot = self._cpu_snapshot()
        with self._lock:
            self._memory_samples.append(memory)
            self._utilization_samples.append(utilization)
            for identity, seconds in cpu_snapshot.items():
                self._cpu_latest[identity] = max(
                    seconds, self._cpu_latest.get(identity, seconds)
                )

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self._sample_interval_seconds):
            try:
                self._sample_once()
            except Exception as error:
                with self._lock:
                    self._thread_error = error
                self._stop_event.set()
                return

    def start(self) -> None:
        if self._thread is not None:
            raise AdapterError("observability sampler is already active")
        self._stop_event.clear()
        with self._lock:
            self._thread_error = None
            self._memory_samples = []
            self._utilization_samples = []
            self._cpu_start = self._cpu_snapshot()
            self._cpu_latest = dict(self._cpu_start)
        self._sample_once()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="rustinfer-vllm-observability",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, wall_seconds: float) -> ObservabilityMeasurement:
        thread = self._thread
        if thread is None:
            raise AdapterError("observability sampler is not active")
        self._stop_event.set()
        thread.join(timeout=max(1.0, self._sample_interval_seconds * 10))
        self._thread = None
        if thread.is_alive():
            raise AdapterError("observability sampler thread did not stop")
        self._sample_once()
        with self._lock:
            thread_error = self._thread_error
            memory_samples = tuple(self._memory_samples)
            utilization_samples = tuple(self._utilization_samples)
            cpu_start = dict(self._cpu_start)
            cpu_latest = dict(self._cpu_latest)
        if thread_error is not None:
            raise AdapterError(f"observability sampler failed: {thread_error}")
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise AdapterError("observability interval must be positive and finite")
        cpu_seconds = math.fsum(
            max(0.0, latest - cpu_start.get(identity, 0.0))
            for identity, latest in cpu_latest.items()
        )
        return ObservabilityMeasurement(
            cpu_utilization_percent=100.0 * cpu_seconds / wall_seconds,
            gpu_utilization_percent=math.fsum(utilization_samples)
            / len(utilization_samples),
            peak_gpu_memory_bytes=max(memory_samples),
        )


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdapterError(f"{path} must be an object")
    return value


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterError(f"{path} must be a positive integer")
    return value


def _load_json(path: Path, label: str) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AdapterError(f"{label} {path} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _resolve_repo_relative(path_text: str, anchor: Path) -> Path:
    direct = Path(path_text)
    if direct.is_file():
        return direct
    for parent in anchor.resolve().parents:
        candidate = parent / path_text
        if candidate.is_file():
            return candidate
    raise AdapterError(f"cannot resolve repository path {path_text!r}")


def _validate_matrix(
    matrix: Mapping[str, object],
    *,
    run_index: int,
    warm_state: str,
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, object]:
    if matrix.get("contract_version") != CONTRACT_VERSION:
        raise AdapterError("matrix.contract_version differs from the adapter")
    if matrix.get("benchmark_scope") != "end-to-end":
        raise AdapterError("vLLM adapter accepts only end-to-end matrices")
    matrix_id = matrix.get("matrix_id")
    if not isinstance(matrix_id, str) or not matrix_id:
        raise AdapterError("matrix.matrix_id must be a non-empty string")
    if matrix.get("tokenization") != {
        "input": "pretokenized",
        "latency_included": False,
        "add_special_tokens": True,
    }:
        raise AdapterError("matrix tokenization contract differs from the adapter")
    if matrix.get("primary_hardware") != {
        "gpu_model": PRIMARY_GPU_NAME,
        "compute_capability": PRIMARY_COMPUTE_CAPABILITY,
        "dtype": DTYPE,
    }:
        raise AdapterError("matrix primary hardware differs from the pinned lane")
    model = _object(matrix.get("model"), "matrix.model")
    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_file": "model.safetensors",
        "weights_sha256": MODEL_WEIGHTS_SHA256,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }
    if model != expected_model:
        raise AdapterError("matrix model identity differs from the immutable lane")

    axes = _object(matrix.get("axes"), "matrix.axes")
    selections = {
        "warm_state": warm_state,
        "concurrency": concurrency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
    for name, selected in selections.items():
        values = axes.get(name)
        if not isinstance(values, list) or selected not in values:
            raise AdapterError(f"selected {name}={selected!r} is outside matrix axes")
    expected_sampling = [
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
    ]
    if axes.get("sampling") != expected_sampling:
        raise AdapterError("vLLM adapter requires the canonical greedy sampling axis")
    if axes.get("approximation_enabled") != [False]:
        raise AdapterError("vLLM adapter requires approximation disabled")
    if matrix.get("cache_policy") != _EXPECTED_CACHE_POLICY:
        raise AdapterError("matrix cache policy differs from the Gate A contract")
    if prompt_tokens + output_tokens > MAX_CONTEXT_TOKENS:
        raise AdapterError("selected cell exceeds the model context length")

    measurement = _object(matrix.get("measurement"), "matrix.measurement")
    independent_runs = _positive_int(
        measurement.get("independent_runs"), "matrix.measurement.independent_runs"
    )
    if run_index > independent_runs:
        raise AdapterError(
            f"run_index {run_index} exceeds independent_runs {independent_runs}"
        )
    state_policy = _object(
        measurement.get(warm_state), f"matrix.measurement.{warm_state}"
    )
    warmups = state_policy.get("warmup_iterations")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise AdapterError("warmup_iterations must be a non-negative integer")
    measured = _positive_int(
        state_policy.get("measured_iterations_per_run"),
        f"matrix.measurement.{warm_state}.measured_iterations_per_run",
    )
    if (
        state_policy.get("fresh_process_per_independent_run") is not True
        or state_policy.get("reset_model_state_per_independent_run") is not True
    ):
        raise AdapterError("each independent run requires a fresh model process")
    expected_reuse = warm_state == "warm"
    if state_policy.get("reuse_model_within_run") is not expected_reuse:
        raise AdapterError("matrix within-run model reuse differs from warm state")
    if warm_state == "cold" and (warmups != 0 or measured != 1):
        raise AdapterError("cold cells require one measurement in a fresh process")
    if warm_state == "warm" and (warmups != 5 or measured != 30):
        raise AdapterError("warm cells require five warmups and thirty measurements")
    lane_paths = matrix.get("lane_manifests")
    if not isinstance(lane_paths, list) or not all(
        isinstance(item, str) for item in lane_paths
    ):
        raise AdapterError("matrix.lane_manifests must be a string array")
    return {
        "matrix_id": matrix_id,
        "lane_manifests": lane_paths,
        "warmups": warmups,
        "measured_iterations": measured,
    }


def _load_vllm_lane(
    matrix_contract: Mapping[str, object], matrix_path: Path
) -> tuple[dict[str, object], str]:
    for path_text in matrix_contract["lane_manifests"]:
        lane_path = _resolve_repo_relative(str(path_text), matrix_path)
        lane, checksum = _load_json(lane_path, "lane manifest")
        if lane.get("lane_id") != "vllm":
            continue
        expected = {
            "implementation_id": "vllm",
            "reference_implementation": "hf-transformers-eager",
            "runtime_dependency_class": "python-reference",
            "semantic_class": "reference",
            "availability": "available",
            "dtype": DTYPE,
            "sampling_id": "greedy",
        }
        for key, value in expected.items():
            if lane.get(key) != value:
                raise AdapterError(f"vLLM lane {key} must be {value!r}")
        if lane.get("model") != {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        }:
            raise AdapterError("vLLM lane model identity differs from matrix")
        engine = _object(lane.get("engine"), "vLLM lane.engine")
        if engine.get("version") != VLLM_VERSION or engine.get("revision") != ENGINE_REVISION:
            raise AdapterError("vLLM lane engine revision differs from the adapter")
        benchmark = _object(
            _object(lane.get("commands"), "vLLM lane.commands").get("benchmark"),
            "vLLM lane.commands.benchmark",
        )
        if benchmark.get("status") != "available":
            raise AdapterError("vLLM benchmark command is not available")
        return lane, checksum
    raise AdapterError("matrix does not contain the vLLM lane manifest")


def load_prompts(path: Path) -> tuple[tuple[PromptRecord, ...], str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise AdapterError(f"cannot read prompts {path}: {error}") from error
    prompts: list[PromptRecord] = []
    prompt_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise AdapterError(f"{path}:{line_number}: blank JSONL line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AdapterError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise AdapterError(f"{path}:{line_number}: prompt must be an object")
        prompt_id = value.get("prompt_id")
        prompt_text = value.get("text")
        target = value.get("target_prompt_tokens")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in prompt_ids:
            raise AdapterError(f"{path}:{line_number}: invalid or duplicate prompt_id")
        if not isinstance(prompt_text, str):
            raise AdapterError(f"{path}:{line_number}: text must be a string")
        if target is not None and (
            isinstance(target, bool)
            or not isinstance(target, int)
            or not 1 <= target <= MAX_CONTEXT_TOKENS
        ):
            raise AdapterError(
                f"{path}:{line_number}: target_prompt_tokens must be null or in "
                f"[1, {MAX_CONTEXT_TOKENS}]"
            )
        prompt_ids.add(prompt_id)
        prompts.append(PromptRecord(prompt_id, prompt_text, target))
    if not prompts:
        raise AdapterError("prompt corpus must not be empty")
    return tuple(prompts), hashlib.sha256(raw).hexdigest()


def _select_prompts(
    prompts: Sequence[PromptRecord], *, concurrency: int, prompt_tokens: int
) -> tuple[PromptRecord, ...]:
    matching = [
        prompt for prompt in prompts if prompt.target_prompt_tokens == prompt_tokens
    ]
    if not matching:
        raise AdapterError(
            f"prompt corpus has no performance seeds for {prompt_tokens} tokens"
        )
    return tuple(matching[index % len(matching)] for index in range(concurrency))


def _resize_token_ids(token_ids: Sequence[int], target: int) -> tuple[int, ...]:
    if target <= 0 or not token_ids:
        raise AdapterError("cannot resize an empty token sequence")
    normalized: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise AdapterError("tokenizer returned a non-integer token ID")
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise AdapterError("token ID is outside canonical unsigned 32-bit range")
        normalized.append(token_id)
    if len(normalized) >= target:
        return tuple(normalized[:target])
    repeats = math.ceil(target / len(normalized))
    return tuple((normalized * repeats)[:target])


def prompt_token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise AdapterError("prompt token hash requires integer IDs")
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise AdapterError("prompt token ID cannot be encoded as unsigned u32")
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def _read_metric(metrics: object, names: Sequence[str]) -> float | None:
    for name in names:
        if isinstance(metrics, Mapping):
            value = metrics.get(name)
        else:
            value = getattr(metrics, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result = float(value)
            if math.isfinite(result) and result > 0:
                return result
    return None


def _engine_timing_durations(
    metrics: object, *, generated_tokens: int
) -> tuple[float, float, float]:
    """Return engine-reported durations used only as a host-timing sanity check.

    vLLM V1 exposes arrival as wall clock but its token timestamps as monotonic,
    so those values are never subtracted from each other.  V1 E2E is instead
    first_token_latency plus the monotonic first-to-last decode interval.  The
    legacy RequestMetrics timestamps all share one clock and can be subtracted.
    """

    direct_ttft = _read_metric(metrics, ("first_token_latency",))
    arrival = _read_metric(metrics, ("arrival_time",))
    legacy_first = _read_metric(metrics, ("first_token_time",))
    current_first = _read_metric(metrics, ("first_token_ts",))
    current_last = _read_metric(metrics, ("last_token_ts",))
    legacy_last = _read_metric(metrics, ("last_token_time",))
    legacy_finished = _read_metric(metrics, ("finished_time",))

    if direct_ttft is not None and current_first is not None:
        decode = 0.0
        if generated_tokens > 1:
            if current_last is None or current_last < current_first:
                raise AdapterError("vLLM RequestOutput lacks ordered V1 TPOT timestamps")
            decode = current_last - current_first
        mean_tpot = decode / (generated_tokens - 1) if generated_tokens > 1 else 0.0
        return direct_ttft, direct_ttft + decode, mean_tpot

    if (
        arrival is not None
        and legacy_first is not None
        and legacy_finished is not None
        and arrival <= legacy_first <= legacy_finished
    ):
        if generated_tokens > 1:
            last = legacy_last if legacy_last is not None else legacy_finished
            if last < legacy_first:
                raise AdapterError("vLLM RequestOutput has unordered legacy timestamps")
            mean_tpot = (last - legacy_first) / (generated_tokens - 1)
        else:
            mean_tpot = 0.0
        return legacy_first - arrival, legacy_finished - arrival, mean_tpot
    raise AdapterError("vLLM RequestOutput lacks one supported timing metric family")


def _validate_engine_timing_sanity(
    metrics: object,
    *,
    generated_tokens: int,
    host_ttft: float,
    host_end_to_end: float,
    host_itl: Sequence[float],
) -> tuple[float, float, float]:
    engine_ttft, engine_end_to_end, engine_mean_tpot = _engine_timing_durations(
        metrics, generated_tokens=generated_tokens
    )
    host_decode = math.fsum(host_itl)
    engine_decode = engine_mean_tpot * max(0, generated_tokens - 1)
    pairs = (
        ("TTFT", engine_ttft, host_ttft),
        ("E2E", engine_end_to_end, host_end_to_end),
        ("decode", engine_decode, host_decode),
    )
    for label, engine_value, host_value in pairs:
        if (
            not math.isfinite(engine_value)
            or engine_value < 0
            or engine_value
            > host_value + ENGINE_TIMING_SANITY_TOLERANCE_SECONDS
        ):
            raise AdapterError(
                f"vLLM engine {label} duration is inconsistent with host monotonic "
                "DELTA observations"
            )
    return engine_ttft, engine_end_to_end, engine_mean_tpot


def _validate_engine_api(llm: object, request_output_kind_delta: object) -> object:
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        raise AdapterError("vLLM 0.27.1 LLM lacks public llm_engine")
    required_methods = (
        "add_request",
        "step",
        "has_unfinished_requests",
        "abort_request",
    )
    for name in required_methods:
        if not callable(getattr(engine, name, None)):
            raise AdapterError(f"vLLM 0.27.1 LLMEngine lacks callable {name}")
    try:
        add_parameters = inspect.signature(engine.add_request).parameters
        abort_parameters = inspect.signature(engine.abort_request).parameters
    except (TypeError, ValueError) as error:
        raise AdapterError(f"cannot inspect vLLM 0.27.1 LLMEngine API: {error}") from error
    for name in ("request_id", "prompt", "params", "arrival_time"):
        if name not in add_parameters:
            raise AdapterError(f"LLMEngine.add_request lacks pinned parameter {name}")
    if "request_ids" not in abort_parameters or "internal" not in abort_parameters:
        raise AdapterError("LLMEngine.abort_request signature differs from v0.27.1")
    engine_core = getattr(engine, "engine_core", None)
    if not callable(getattr(engine_core, "shutdown", None)):
        raise AdapterError("vLLM 0.27.1 engine_core lacks callable shutdown")
    if (
        getattr(request_output_kind_delta, "name", None) != "DELTA"
        or getattr(request_output_kind_delta, "value", None) != 1
    ):
        raise AdapterError("vLLM RequestOutputKind.DELTA differs from v0.27.1")
    return engine


class VllmBackend:
    """Adapter around vLLM 0.27.1's public synchronous LLMEngine API."""

    def __init__(
        self,
        *,
        llm: object,
        tokenizer: object,
        sampling_params_type: Callable[..., object],
        tokens_prompt_type: Callable[..., object],
        request_output_kind_delta: object,
        metadata: BackendMetadata,
        environment: Mapping[str, object],
        observability_sampler: ObservabilitySampler,
        wall_timer: Callable[[], float] = time.time,
    ) -> None:
        self._llm = llm
        self._tokenizer = tokenizer
        self._sampling_params_type = sampling_params_type
        self._tokens_prompt_type = tokens_prompt_type
        self._request_output_kind_delta = request_output_kind_delta
        self._engine = _validate_engine_api(llm, request_output_kind_delta)
        self._environment = dict(environment)
        self._observability_sampler = observability_sampler
        self._wall_timer = wall_timer
        self._batch_counter = 0
        self._closed = False
        self.metadata = metadata

    def environment(self) -> dict[str, object]:
        return dict(self._environment)

    def materialize_token_ids(
        self, texts: Sequence[str], *, prompt_tokens: int
    ) -> tuple[tuple[int, ...], ...]:
        rows: list[tuple[int, ...]] = []
        for text in texts:
            encoded = self._tokenizer.encode(text, add_special_tokens=True)
            if not isinstance(encoded, (list, tuple)):
                raise AdapterError("tokenizer.encode must return a token ID sequence")
            token_ids = list(encoded)
            if not token_ids:
                bos_token_id = getattr(self._tokenizer, "bos_token_id", None)
                if isinstance(bos_token_id, bool) or not isinstance(bos_token_id, int):
                    raise AdapterError("empty tokenization requires an integer BOS token")
                token_ids = [bos_token_id]
            rows.append(_resize_token_ids(token_ids, prompt_tokens))
        return tuple(rows)

    def generate_batch(
        self,
        token_rows: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        timer: Callable[[], float],
    ) -> BatchMeasurement:
        if self._closed:
            raise AdapterError("vLLM engine is closed")
        if not token_rows:
            raise AdapterError("vLLM batch must contain at least one prompt")
        if self._engine.has_unfinished_requests():
            raise AdapterError("vLLM engine has requests left from an earlier batch")
        prompts = [
            self._tokens_prompt_type(prompt_token_ids=list(token_ids))
            for token_ids in token_rows
        ]
        sampling_params = self._sampling_params_type(
            temperature=0,
            max_tokens=max_new_tokens,
            min_tokens=max_new_tokens,
            ignore_eos=True,
            detokenize=False,
            output_kind=self._request_output_kind_delta,
        )
        self._observability_sampler.start()
        started = timer()
        request_ids: list[str] = []
        internal_request_ids: dict[str, str] = {}
        observed_internal_ids: set[str] = set()
        arrivals: dict[str, float] = {}
        token_ids: dict[str, list[int]] = {}
        token_times: dict[str, list[float]] = {}
        final_times: dict[str, float] = {}
        final_outputs: dict[str, object] = {}
        self._batch_counter += 1
        try:
            for index, prompt in enumerate(prompts):
                request_id = f"rustinfer-{self._batch_counter}-{index}"
                arrival = timer()
                engine_arrival = self._wall_timer()
                if not math.isfinite(engine_arrival) or engine_arrival <= 0:
                    raise AdapterError("wall-clock engine arrival must be positive and finite")
                actual_id = self._engine.add_request(
                    request_id=request_id,
                    prompt=prompt,
                    params=sampling_params,
                    arrival_time=engine_arrival,
                )
                if not isinstance(actual_id, str) or actual_id in observed_internal_ids:
                    raise AdapterError("LLMEngine.add_request returned an invalid ID")
                request_ids.append(request_id)
                internal_request_ids[request_id] = actual_id
                observed_internal_ids.add(actual_id)
                arrivals[request_id] = arrival
                token_ids[request_id] = []
                token_times[request_id] = []

            while self._engine.has_unfinished_requests():
                step_outputs = self._engine.step()
                observed = timer()
                if not isinstance(step_outputs, Sequence):
                    raise AdapterError("LLMEngine.step must return a sequence")
                for output in step_outputs:
                    request_id = getattr(output, "request_id", None)
                    if request_id not in arrivals:
                        raise AdapterError("LLMEngine.step returned an unknown request")
                    completions = getattr(output, "outputs", None)
                    if not isinstance(completions, Sequence) or len(completions) != 1:
                        raise AdapterError("greedy DELTA output must have one completion")
                    delta = getattr(completions[0], "token_ids", None)
                    if not isinstance(delta, Sequence) or isinstance(
                        delta, (str, bytes)
                    ):
                        raise AdapterError("vLLM DELTA token_ids must be a sequence")
                    if len(delta) != 1:
                        raise AdapterError(
                            "vLLM DELTA emitted multiple or zero tokens in one host "
                            "observation; per-token ITL would be ambiguous"
                        )
                    token_id = delta[0]
                    if isinstance(token_id, bool) or not isinstance(token_id, int):
                        raise AdapterError("vLLM completion contains a non-integer token ID")
                    previous = (
                        token_times[request_id][-1]
                        if token_times[request_id]
                        else arrivals[request_id]
                    )
                    if not math.isfinite(observed) or observed < previous:
                        raise AdapterError("host monotonic clock moved backwards")
                    token_ids[request_id].append(token_id)
                    token_times[request_id].append(observed)
                    returned_prompt = getattr(output, "prompt_token_ids", None)
                    request_index = request_ids.index(request_id)
                    if returned_prompt is not None and tuple(returned_prompt) != tuple(
                        token_rows[request_index]
                    ):
                        raise AdapterError("vLLM changed the supplied TokensPrompt IDs")
                    if bool(getattr(output, "finished", False)):
                        if request_id in final_outputs:
                            raise AdapterError("vLLM finished one request more than once")
                        final_times[request_id] = observed
                        final_outputs[request_id] = output
            if set(final_outputs) != set(request_ids):
                raise AdapterError("vLLM engine stopped before every request finished")
        except BaseException:
            pending = [
                internal_request_ids[request_id]
                for request_id in request_ids
                if request_id not in final_outputs
            ]
            try:
                if pending:
                    self._engine.abort_request(pending, internal=True)
            finally:
                self._engine.engine_core.shutdown()
                self._closed = True
            raise
        finally:
            finished = timer()
            observation = self._observability_sampler.stop(
                wall_seconds=finished - started
            )
        wall = finished - started
        if not math.isfinite(wall) or wall <= 0:
            raise AdapterError("batch wall time must be positive and finite")
        measurements: list[RequestMeasurement] = []
        try:
            for request_id in request_ids:
                generated = token_ids[request_id]
                if len(generated) != max_new_tokens:
                    raise AdapterError("vLLM did not honor fixed output token length")
                observations = token_times[request_id]
                ttft = observations[0] - arrivals[request_id]
                end_to_end = final_times[request_id] - arrivals[request_id]
                itl = tuple(
                    observations[index] - observations[index - 1]
                    for index in range(1, len(observations))
                )
                mean_tpot = math.fsum(itl) / len(itl) if itl else 0.0
                _validate_engine_timing_sanity(
                    getattr(final_outputs[request_id], "metrics", None),
                    generated_tokens=len(generated),
                    host_ttft=ttft,
                    host_end_to_end=end_to_end,
                    host_itl=itl,
                )
                observed_and_reported = (ttft, end_to_end, mean_tpot, *itl)
                if any(
                    not math.isfinite(value) or value < 0
                    for value in observed_and_reported
                ) or ttft > end_to_end:
                    raise AdapterError("vLLM request timing observations are invalid")
                measurements.append(
                    RequestMeasurement(
                        len(generated),
                        tuple(generated),
                        ttft,
                        end_to_end,
                        itl,
                        mean_tpot,
                    )
                )
        except BaseException:
            self.close()
            raise

        total_output_tokens = sum(item.generated_tokens for item in measurements)
        return BatchMeasurement(
            requests=tuple(measurements),
            wall_seconds=wall,
            output_tokens_per_second=total_output_tokens / wall,
            cpu_utilization_percent=observation.cpu_utilization_percent,
            gpu_utilization_percent=observation.gpu_utilization_percent,
            peak_gpu_memory_bytes=observation.peak_gpu_memory_bytes,
        )

    def close(self) -> None:
        if not self._closed:
            self._engine.engine_core.shutdown()
            self._closed = True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AdapterError(f"cannot hash immutable snapshot artifact {path}: {error}") from error
    return digest.hexdigest()


def verify_snapshot_artifacts(
    snapshot_path: Path,
    *,
    expected_weights_sha256: str = MODEL_WEIGHTS_SHA256,
    expected_config_sha256: str = MODEL_CONFIG_SHA256,
    expected_tokenizer_files_sha256: Mapping[str, str] = TOKENIZER_FILES_SHA256,
    expected_tokenizer_sha256: str = TOKENIZER_SHA256,
) -> str:
    expected_files = {
        "model.safetensors": expected_weights_sha256,
        "config.json": expected_config_sha256,
        **dict(expected_tokenizer_files_sha256),
    }
    actual: dict[str, str] = {}
    for filename, expected in expected_files.items():
        path = snapshot_path / filename
        if not path.is_file():
            raise AdapterError(f"immutable snapshot lacks {filename}")
        actual[filename] = _sha256_file(path)
        if actual[filename] != expected:
            raise AdapterError(
                f"{filename} SHA-256 {actual[filename]} differs from immutable contract"
            )
    tokenizer_hashes = {
        filename: actual[filename] for filename in expected_tokenizer_files_sha256
    }
    canonical = json.dumps(
        tokenizer_hashes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    aggregate = hashlib.sha256(canonical).hexdigest()
    if aggregate != expected_tokenizer_sha256:
        raise AdapterError(
            f"tokenizer aggregate SHA-256 {aggregate} differs from immutable contract"
        )
    return actual["model.safetensors"]


def _resolve_snapshot(*, local_files_only: bool) -> Path:
    try:
        hub = importlib.import_module("huggingface_hub")
        snapshot = hub.snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
        )
    except Exception as error:
        mode = "local cache" if local_files_only else "Hugging Face Hub/cache"
        raise AdapterError(
            f"cannot resolve immutable model revision from {mode}: {error}"
        ) from error
    snapshot_path = Path(snapshot).resolve()
    if snapshot_path.name != MODEL_REVISION:
        raise AdapterError(
            f"resolved snapshot {snapshot_path.name!r} is not revision {MODEL_REVISION}"
        )
    return snapshot_path


def _query_environment(torch_module: object) -> dict[str, object]:
    try:
        gpu_name = str(torch_module.cuda.get_device_name(0))
        major, minor = torch_module.cuda.get_device_capability(0)
        gpu_count = int(torch_module.cuda.device_count())
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError, AttributeError) as error:
        raise AdapterError(f"cannot query primary GPU environment: {error}") from error
    cpu_model = platform.processor().strip()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    cpu_model = line.split(":", maxsplit=1)[1].strip()
                    break
    except OSError:
        pass
    cpu_model = cpu_model or platform.machine() or "unknown"
    try:
        release = platform.freedesktop_os_release()
        os_text = (
            f"{release.get('NAME', 'Linux')} {release.get('VERSION_ID', '')}, "
            f"Linux {platform.release()}, {platform.machine()}"
        )
    except OSError:
        os_text = platform.platform()
    cuda_version = str(getattr(torch_module.version, "cuda", None) or "unknown")
    return {
        "gpu_model": gpu_name,
        "compute_capability": f"{major}.{minor}",
        "gpu_count": gpu_count,
        "cpu_model": cpu_model,
        "ram_bytes": int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
        "os": os_text,
        "nvidia_driver_version": driver,
        "cuda_toolkit_version": f"wheel-build-{cuda_version}",
        "cuda_runtime_version": cuda_version,
    }


def _llm_options(*, max_num_seqs: int) -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "tokenizer": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "max_model_len": MAX_CONTEXT_TOKENS,
        "max_num_seqs": max_num_seqs,
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
        "load_format": "safetensors",
        "enable_prefix_caching": False,
        "disable_log_stats": False,
    }


def _pin_vllm_environment() -> None:
    variable = "VLLM_USE_FLASHINFER_SAMPLER"
    existing = os.environ.get(variable)
    if existing not in {None, "0"}:
        raise AdapterError(f"{variable} must be 0 for the pinned non-JIT sampler")
    os.environ[variable] = "0"


def load_default_backend(
    *, local_files_only: bool, max_num_seqs: int
) -> VllmBackend:
    """Load the immutable checkpoint and public vLLM offline API lazily."""

    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    _pin_vllm_environment()
    snapshot_path = _resolve_snapshot(local_files_only=local_files_only)
    weights_sha256 = verify_snapshot_artifacts(snapshot_path)
    try:
        installed_version = importlib.metadata.version("vllm")
        vllm = importlib.import_module("vllm")
        torch_module = importlib.import_module("torch")
        nvml_module = importlib.import_module("pynvml")
        psutil_module = importlib.import_module("psutil")
        llm_type = getattr(vllm, "LLM")
        sampling_params_type = getattr(vllm, "SamplingParams")
        tokens_prompt_type = getattr(vllm, "TokensPrompt")
        sampling_module = importlib.import_module("vllm.sampling_params")
        request_output_kind = getattr(sampling_module, "RequestOutputKind")
        request_output_kind_delta = getattr(request_output_kind, "DELTA")
    except Exception as error:
        raise AdapterError(f"cannot import pinned vLLM public API: {error}") from error
    if installed_version != VLLM_VERSION:
        raise AdapterError(
            f"installed vLLM {installed_version!r} differs from {VLLM_VERSION!r}"
        )
    llm = llm_type(**_llm_options(max_num_seqs=max_num_seqs))
    tokenizer = llm.get_tokenizer()
    observability_sampler = NvmlProcessTreeSampler(
        nvml_module=nvml_module,
        psutil_module=psutil_module,
    )
    return VllmBackend(
        llm=llm,
        tokenizer=tokenizer,
        sampling_params_type=sampling_params_type,
        tokens_prompt_type=tokens_prompt_type,
        request_output_kind_delta=request_output_kind_delta,
        observability_sampler=observability_sampler,
        metadata=BackendMetadata(
            engine_version=installed_version,
            model_revision=MODEL_REVISION,
            weights_sha256=weights_sha256,
            local_files_only=local_files_only,
        ),
        environment=_query_environment(torch_module),
    )


def _validate_environment(environment: Mapping[str, object]) -> None:
    if environment.get("gpu_model") != PRIMARY_GPU_NAME:
        raise AdapterError("runtime GPU differs from primary environment")
    if environment.get("compute_capability") != PRIMARY_COMPUTE_CAPABILITY:
        raise AdapterError("runtime compute capability differs from primary environment")
    if environment.get("gpu_count") != 1:
        raise AdapterError("primary performance lane requires exactly one GPU")
    if environment.get("nvidia_driver_version") != PRIMARY_DRIVER_VERSION:
        raise AdapterError("runtime NVIDIA driver differs from primary environment")
    if environment.get("ram_bytes") != PRIMARY_RAM_BYTES:
        raise AdapterError("runtime RAM differs from primary environment")
    if "Ubuntu 22.04" not in str(environment.get("os")):
        raise AdapterError("runtime OS differs from primary environment")
    if "i7-13700K" not in str(environment.get("cpu_model")):
        raise AdapterError("runtime CPU differs from primary environment")
    required = {
        "gpu_model",
        "compute_capability",
        "gpu_count",
        "cpu_model",
        "ram_bytes",
        "os",
        "nvidia_driver_version",
        "cuda_toolkit_version",
        "cuda_runtime_version",
    }
    if set(environment) != required:
        raise AdapterError("runtime environment fields differ from result schema")


def _git_provenance(anchor: Path) -> dict[str, object]:
    try:
        repository_root = subprocess.run(
            ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
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
        dirty = subprocess.run(
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
        raise AdapterError(f"cannot capture Git provenance: {error}") from error
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise AdapterError("Git revision is not a full SHA-1")
    return {"git_revision": revision, "git_dirty": bool(dirty)}


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise AdapterError("benchmark timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _null_optimizations() -> tuple[dict[str, None], dict[str, None], dict[str, None]]:
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
    matrix_id: str,
    matrix_sha256: str,
    prompts_sha256: str,
    lane_sha256: str,
    environment: Mapping[str, object],
    provenance: Mapping[str, object],
    warm_state: str,
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, object]:
    speculative, sparse_attention, quantization = _null_optimizations()
    return {
        "contract_version": CONTRACT_VERSION,
        "trial_id": trial_id,
        "run_id": run_id,
        "trial_index": trial_index,
        "recorded_at_utc": recorded_at,
        "scope": "end-to-end",
        "matrix_id": matrix_id,
        "matrix_sha256": matrix_sha256,
        "prompts_sha256": prompts_sha256,
        "lane_manifest_sha256": lane_sha256,
        "environment_id": PRIMARY_ENVIRONMENT_ID,
        "semantic_class": "reference",
        "correctness_gate_id": None,
        "correctness_report_sha256": None,
        "implementation_id": "vllm",
        "reference_implementation": "hf-transformers-eager",
        "runtime_dependency_class": "python-reference",
        "approximation_enabled": False,
        "error_budget": None,
        "seed": None,
        "warm_state": warm_state,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "engine_revision": ENGINE_REVISION,
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
    measurement: BatchMeasurement,
    prompts: Sequence[PromptRecord],
    token_rows: Sequence[Sequence[int]],
    *,
    model_load_ms: float,
    output_tokens: int,
) -> dict[str, object]:
    if len(measurement.requests) != len(prompts):
        raise AdapterError("vLLM measurement count differs from concurrency")
    base["metrics"] = {
        "model_load_ms": model_load_ms,
        "batch_wall_ms": 1000.0 * measurement.wall_seconds,
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
            "prompt_token_ids_sha256": prompt_token_ids_sha256(token_rows[index]),
            "generated_token_ids_sha256": prompt_token_ids_sha256(
                request.generated_token_ids
            ),
            "status": "success",
            "failure_reason": None,
            "prompt_tokens": len(token_rows[index]),
            "requested_output_tokens": output_tokens,
            "generated_tokens": request.generated_tokens,
            "ttft_ms": 1000.0 * request.ttft_seconds,
            "end_to_end_ms": 1000.0 * request.end_to_end_seconds,
            "mean_tpot_ms": 1000.0 * request.mean_tpot_seconds,
            "itl_ms": [1000.0 * value for value in request.itl_seconds],
        }
        for index, (prompt, request) in enumerate(zip(prompts, measurement.requests))
    ]
    return base


def _failure_row(
    base: dict[str, object],
    prompts: Sequence[PromptRecord],
    token_rows: Sequence[Sequence[int]],
    *,
    model_load_ms: float,
    output_tokens: int,
    error: Exception,
) -> dict[str, object]:
    reason = f"{type(error).__name__}: {error}"[:1000]
    base["status"] = "failure"
    base["failure_reason"] = reason
    base["failure_count"] = len(prompts)
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
            "prompt_token_ids_sha256": prompt_token_ids_sha256(token_rows[index]),
            "generated_token_ids_sha256": prompt_token_ids_sha256(()),
            "status": "failure",
            "failure_reason": reason,
            "prompt_tokens": len(token_rows[index]),
            "requested_output_tokens": output_tokens,
            "generated_tokens": 0,
            "ttft_ms": None,
            "end_to_end_ms": None,
            "mean_tpot_ms": None,
            "itl_ms": None,
        }
        for index, prompt in enumerate(prompts)
    ]
    return base


def validate_result_row(row: Mapping[str, object]) -> None:
    if set(row) != _TOP_LEVEL_KEYS:
        raise AdapterError("result row keys differ from result schema v1")
    for name in (
        "matrix_sha256",
        "prompts_sha256",
        "lane_manifest_sha256",
    ):
        if not isinstance(row[name], str) or not _SHA256_RE.fullmatch(row[name]):
            raise AdapterError(f"result {name} is not lowercase SHA-256")
    workload = _object(row["workload"], "result.workload")
    requests = row["requests"]
    if not isinstance(requests, list) or len(requests) != workload.get("concurrency"):
        raise AdapterError("result request count differs from concurrency")
    if row["warm_state"] != workload.get("warm_state"):
        raise AdapterError("result warm state is inconsistent")
    for request in requests:
        request_object = _object(request, "result.requests[]")
        for field in (
            "prompt_token_ids_sha256",
            "generated_token_ids_sha256",
        ):
            digest = request_object.get(field)
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise AdapterError(f"request {field} is invalid")
        if request_object.get("status") == "success":
            generated_tokens = request_object.get("generated_tokens")
            itl_ms = request_object.get("itl_ms")
            mean_tpot_ms = request_object.get("mean_tpot_ms")
            if not isinstance(generated_tokens, int) or isinstance(
                generated_tokens, bool
            ):
                raise AdapterError("successful request generated_tokens is invalid")
            if not isinstance(itl_ms, list) or len(itl_ms) != max(
                0, generated_tokens - 1
            ):
                raise AdapterError("successful request ITL count is invalid")
            expected_mean = (
                math.fsum(float(value) for value in itl_ms) / len(itl_ms)
                if itl_ms
                else 0.0
            )
            if not isinstance(mean_tpot_ms, (int, float)) or not math.isclose(
                float(mean_tpot_ms), expected_mean, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise AdapterError("successful request mean TPOT differs from ITLs")


def _write_artifact(
    result_dir: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    run_index: int,
    matrix_sha256: str,
    prompts_sha256: str,
    lane_sha256: str,
    cell: Mapping[str, object],
    local_files_only: bool,
) -> None:
    try:
        result_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise AdapterError(f"refusing to reuse result directory: {result_dir}") from error
    with (result_dir / "raw.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            json.dump(
                row,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "independent_run_index": run_index,
        "row_count": len(rows),
        "cell": dict(cell),
        "local_files_only": local_files_only,
        "matrix_sha256": matrix_sha256,
        "prompts_sha256": prompts_sha256,
        "lane_manifest_sha256": lane_sha256,
    }
    with (result_dir / "metadata.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            metadata,
            handle,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    with (result_dir / "README.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "# vLLM 0.27.1 baseline benchmark\n\n"
            f"Independent run `{run_id}` (index {run_index}) produced {len(rows)} "
            "strict result-schema-v1 rows for one matrix cell. Inputs were supplied "
            "as exact `TokensPrompt` IDs; greedy generation used equal min/max output "
            "length and ignored EOS.\n"
        )


def run_benchmark(
    *,
    matrix_path: Path,
    prompts_path: Path,
    result_dir: Path,
    run_index: int,
    run_id: str,
    warm_state: str,
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
    allow_download: bool = False,
    backend_factory: Callable[..., Backend] = load_default_backend,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Run exactly one matrix cell in one process and write an append-only artifact."""

    if result_dir.exists():
        raise AdapterError(f"refusing to reuse result directory: {result_dir}")
    if not _ID_RE.fullmatch(run_id):
        raise AdapterError("run_id contains characters forbidden by the result schema")
    run_index = _positive_int(run_index, "run_index")
    concurrency = _positive_int(concurrency, "concurrency")
    prompt_tokens = _positive_int(prompt_tokens, "prompt_tokens")
    output_tokens = _positive_int(output_tokens, "output_tokens")
    if warm_state not in {"cold", "warm"}:
        raise AdapterError("warm_state must be cold or warm")

    matrix, matrix_sha256 = _load_json(matrix_path, "matrix")
    contract = _validate_matrix(
        matrix,
        run_index=run_index,
        warm_state=warm_state,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )
    _, lane_sha256 = _load_vllm_lane(contract, matrix_path)
    prompts, prompts_sha256 = load_prompts(prompts_path)
    selected_prompts = _select_prompts(
        prompts, concurrency=concurrency, prompt_tokens=prompt_tokens
    )
    timestamp = now()
    recorded_at = _utc_text(timestamp)
    provenance = _git_provenance(matrix_path.resolve().parent)

    load_started = timer()
    backend = backend_factory(
        local_files_only=not allow_download,
        max_num_seqs=concurrency,
    )
    model_load_ms = 1000.0 * (timer() - load_started)
    if not math.isfinite(model_load_ms) or model_load_ms < 0:
        raise AdapterError("model load time is invalid")
    if backend.metadata != BackendMetadata(
        engine_version=VLLM_VERSION,
        model_revision=MODEL_REVISION,
        weights_sha256=MODEL_WEIGHTS_SHA256,
        local_files_only=not allow_download,
    ):
        raise AdapterError("loaded backend metadata differs from the immutable contract")
    environment = backend.environment()
    _validate_environment(environment)
    texts = tuple(prompt.text for prompt in selected_prompts)
    token_rows = backend.materialize_token_ids(texts, prompt_tokens=prompt_tokens)
    if len(token_rows) != concurrency or any(
        len(token_ids) != prompt_tokens for token_ids in token_rows
    ):
        raise AdapterError("pretokenized request shape differs from the selected cell")

    warmup_error: Exception | None = None
    try:
        for _ in range(int(contract["warmups"])):
            backend.generate_batch(
                token_rows,
                max_new_tokens=output_tokens,
                timer=timer,
            )
    except Exception as error:
        warmup_error = error

    rows: list[dict[str, object]] = []
    for trial_index in range(1, int(contract["measured_iterations"]) + 1):
        trial_id = (
            f"{run_id}:{warm_state}:c{concurrency}:p{prompt_tokens}:"
            f"o{output_tokens}:i{trial_index}"
        )
        base = _base_row(
            trial_id=trial_id,
            run_id=run_id,
            trial_index=trial_index,
            recorded_at=recorded_at,
            matrix_id=str(contract["matrix_id"]),
            matrix_sha256=matrix_sha256,
            prompts_sha256=prompts_sha256,
            lane_sha256=lane_sha256,
            environment=environment,
            provenance=provenance,
            warm_state=warm_state,
            concurrency=concurrency,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        try:
            if warmup_error is not None:
                raise warmup_error
            measurement = backend.generate_batch(
                token_rows,
                max_new_tokens=output_tokens,
                timer=timer,
            )
            row = _success_row(
                base,
                measurement,
                selected_prompts,
                token_rows,
                model_load_ms=model_load_ms,
                output_tokens=output_tokens,
            )
        except Exception as error:
            row = _failure_row(
                base,
                selected_prompts,
                token_rows,
                model_load_ms=model_load_ms,
                output_tokens=output_tokens,
                error=error,
            )
        validate_result_row(row)
        rows.append(row)

    cell = {
        "warm_state": warm_state,
        "concurrency": concurrency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
    close = getattr(backend, "close", None)
    if callable(close):
        close()
    _write_artifact(
        result_dir,
        rows,
        run_id=run_id,
        run_index=run_index,
        matrix_sha256=matrix_sha256,
        prompts_sha256=prompts_sha256,
        lane_sha256=lane_sha256,
        cell=cell,
        local_files_only=not allow_download,
    )
    return len(rows)
