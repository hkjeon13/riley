from __future__ import annotations

import hashlib
import json
import math
import struct
import copy
from dataclasses import dataclass
from pathlib import Path

from rustinfer_reference.constants import (
    MODEL_CONFIG_SHA256,
    MODEL_WEIGHTS_SHA256,
    PRIMARY_GPU_COMPUTE_CAPABILITY,
    PRIMARY_GPU_NAME,
    PRIMARY_NVIDIA_DRIVER_VERSION,
    PYTHON_EXECUTABLE_SHA256,
    PYTHON_PLATFORM_MACHINE,
    PYTHON_PLATFORM_SYSTEM,
    PYTHON_VERSION,
    PYTHON_VERSION_FILE_SHA256,
    PRIMARY_RAM_BYTES,
    TORCH_VERSION,
    TOKENIZER_FILES_SHA256,
    TOKENIZER_SHA256,
    TRANSFORMERS_VERSION,
)
from rustinfer_reference.fixture import (
    BackendMetadata,
    CaseResult,
    FIXTURE_SOURCE_PATHS,
    summarize_values,
)
from rustinfer_reference.environment import PRIMARY_ENVIRONMENT_SNAPSHOT


def fixture_provenance() -> dict[str, object]:
    return {
        "git_revision": "1" * 40,
        "git_tree": "2" * 40,
        "git_dirty": False,
        "git_status_sha256": hashlib.sha256(b"").hexdigest(),
        "environment_id": "rtx4090-ubuntu22-driver580-v1",
        "observed_environment": copy.deepcopy(PRIMARY_ENVIRONMENT_SNAPSHOT),
        "sources": {
            name: {
                "path": path,
                "sha256": (
                    PYTHON_VERSION_FILE_SHA256
                    if name == "python_version_file"
                    else hashlib.sha256(path.encode()).hexdigest()
                ),
            }
            for name, path in FIXTURE_SOURCE_PATHS.items()
        },
    }


def prompt_rows() -> list[dict[str, object]]:
    base = {
        "contract_version": "1.0.0",
        "target_prompt_tokens": None,
        "boundary_kind": "none",
        "expected_behavior": "deterministic greedy output",
        "contains_sensitive_data": False,
    }
    cases = [
        ("short-en", "short", "en", "The sky is"),
        ("multi-ko", "multilingual", "mixed", "안녕하세요, hello"),
        ("symbols-code", "symbols-code", "code", "x = 42 ** 2; # !?"),
        ("repeat", "long-repetition", "en", "repeat repeat"),
        ("boundary", "context-boundary", "en", "boundary seed"),
        ("minimal", "minimal", "none", ""),
        ("early-eos", "early-eos", "en", "End now."),
    ]
    rows: list[dict[str, object]] = []
    for prompt_id, category, language, text in cases:
        row = dict(base)
        row.update(
            {
                "prompt_id": prompt_id,
                "category": category,
                "language": language,
                "text": text,
            }
        )
        if category == "context-boundary":
            row["target_prompt_tokens"] = 7168
            row["boundary_kind"] = "near-max-context"
        if category == "early-eos":
            row["expected_behavior"] = "greedy-eos-at-first-output-token"
        rows.append(row)
    return rows


def write_prompts(path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompt_rows()),
        encoding="utf-8",
    )


def _log_softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    log_sum = maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))
    return [value - log_sum for value in values]


@dataclass(frozen=True)
class FakeMeasurement:
    prompt_token_counts: tuple[int, ...]
    prompt_token_ids_sha256: tuple[str, ...]
    output_token_counts: tuple[int, ...]
    generated_token_ids_sha256: tuple[str, ...]
    ttft_seconds: float = 0.002
    itl_seconds: tuple[float, ...] = (0.001,)
    end_to_end_seconds: float = 0.010
    output_tokens_per_second: float = 3200.0
    cpu_utilization_percent: float = 12.5
    gpu_utilization_percent: float = 45.0
    peak_gpu_memory_bytes: int = 987_654_321


class FakeBackend:
    eos_token_ids = (99,)

    def __init__(self, *, local_files_only: bool = True) -> None:
        self.metadata = BackendMetadata(
            python_version=PYTHON_VERSION,
            python_executable_sha256=PYTHON_EXECUTABLE_SHA256,
            python_platform_system=PYTHON_PLATFORM_SYSTEM,
            python_platform_machine=PYTHON_PLATFORM_MACHINE,
            torch_version=TORCH_VERSION,
            transformers_version=TRANSFORMERS_VERSION,
            device="cuda:0",
            device_name=PRIMARY_GPU_NAME,
            compute_capability=PRIMARY_GPU_COMPUTE_CAPABILITY,
            local_files_only=local_files_only,
            weights_sha256=MODEL_WEIGHTS_SHA256,
            config_sha256=MODEL_CONFIG_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            tokenizer_files_sha256=dict(TOKENIZER_FILES_SHA256),
        )
        self.benchmark_calls = 0

    def generate_case(
        self,
        text: str,
        *,
        max_new_tokens: int,
        hidden_state_index: int,
        top_k: int,
        target_prompt_tokens: int | None,
    ) -> CaseResult:
        del hidden_state_index
        count = target_prompt_tokens if target_prompt_tokens is not None else max(1, len(text))
        if count <= 0:
            raise ValueError("fake token count must be positive")
        input_ids = tuple((index % 17) + 1 for index in range(count))
        hidden_values = [((index % 11) - 5) / 8 for index in range(count * 2)]
        logits = [0.25, 2.0, -1.5, 0.5]
        is_early_eos = text == "End now."
        generated = (99,) if is_early_eos else tuple(
            10 + index for index in range(max_new_tokens)
        )
        stop_reason = "eos" if is_early_eos else "max_new_tokens"
        return CaseResult(
            input_token_ids=input_ids,
            hidden_state=summarize_values(hidden_values, (1, count, 2)),
            final_logits=summarize_values(logits, (len(logits),), top_k=top_k),
            processed_log_probs={
                "pipeline_id": "log-softmax-fp32-v1",
                "tensor": summarize_values(
                    _log_softmax(logits), (len(logits),), top_k=top_k
                ),
            },
            cache_on_token_ids=generated,
            cache_off_token_ids=generated,
            cache_on_stop_reason=stop_reason,
            cache_off_stop_reason=stop_reason,
        )

    @staticmethod
    def _hashes(texts: tuple[str, ...], prompt_tokens: int) -> tuple[str, ...]:
        results: list[str] = []
        for request_index, text in enumerate(texts):
            digest = hashlib.sha256()
            seed = (len(text.encode("utf-8")) + request_index) % 97
            for index in range(prompt_tokens):
                digest.update(struct.pack("<I", (seed + index) % 32000))
            results.append(digest.hexdigest())
        return tuple(results)

    def prompt_token_ids_sha256(
        self, texts: tuple[str, ...], *, prompt_tokens: int
    ) -> tuple[str, ...]:
        return self._hashes(texts, prompt_tokens)

    @staticmethod
    def _generated_hashes(
        request_count: int, max_new_tokens: int
    ) -> tuple[str, ...]:
        results: list[str] = []
        for request_index in range(request_count):
            digest = hashlib.sha256()
            for step in range(max_new_tokens):
                digest.update(struct.pack("<I", 1000 + request_index * 256 + step))
            results.append(digest.hexdigest())
        return tuple(results)

    def benchmark_batch(
        self,
        texts: tuple[str, ...],
        *,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> FakeMeasurement:
        self.benchmark_calls += 1
        return FakeMeasurement(
            prompt_token_counts=(prompt_tokens,) * len(texts),
            prompt_token_ids_sha256=self._hashes(texts, prompt_tokens),
            output_token_counts=(max_new_tokens,) * len(texts),
            generated_token_ids_sha256=self._generated_hashes(
                len(texts), max_new_tokens
            ),
            itl_seconds=(0.001,) * max(0, max_new_tokens - 1),
        )

    def environment(self) -> dict[str, object]:
        return {
            "gpu_model": PRIMARY_GPU_NAME,
            "compute_capability": PRIMARY_GPU_COMPUTE_CAPABILITY,
            "gpu_count": 1,
            "cpu_model": "Intel Core i7-13700K (fake)",
            "ram_bytes": PRIMARY_RAM_BYTES,
            "os": "Ubuntu 22.04, Linux fake, x86_64",
            "nvidia_driver_version": PRIMARY_NVIDIA_DRIVER_VERSION,
            "cuda_toolkit_version": "99.0",
            "cuda_runtime_version": "99.0",
        }
