"""Lazy Hugging Face eager backend for the external Python reference lane."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    ATTENTION_BACKEND,
    GOLDEN_GREEDY_MAX_NEW_TOKENS,
    MAX_CONTEXT_TOKENS,
    MODEL_CONFIG_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    NVIDIA_ML_PY_VERSION,
    PSUTIL_VERSION,
    PYTHON_EXECUTABLE_SHA256,
    PYTHON_PLATFORM_MACHINE,
    PYTHON_PLATFORM_SYSTEM,
    PYTHON_VERSION,
    TORCH_VERSION,
    TOKENIZER_ARTIFACT_FILENAMES,
    TOKENIZER_FILES_SHA256,
    TOKENIZER_SHA256,
    TRANSFORMERS_VERSION,
)
from .fixture import BackendMetadata, CaseResult, FixtureError, summarize_values


class BackendUnavailableError(RuntimeError):
    """The pinned reference backend cannot be initialized."""


@dataclass(frozen=True)
class BatchMeasurement:
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


def _base_version(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _python_executable_sha256() -> str:
    digest = hashlib.sha256()
    try:
        with Path(sys.executable).resolve().open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BackendUnavailableError(
            f"cannot hash resolved Python executable {sys.executable!r}: {error}"
        ) from error
    return digest.hexdigest()


def _normalize_eos_ids(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (list, tuple)):
        ids = tuple(value)
        if all(isinstance(item, int) and not isinstance(item, bool) for item in ids):
            return tuple(dict.fromkeys(ids))
    raise BackendUnavailableError("model config has an invalid eos_token_id")


class HuggingFaceBackend:
    """Pinned SmolLM2 BF16 eager implementation.

    Heavy dependencies are imported only from :meth:`load`, so fixture validation
    and Philox contract tests stay usable without PyTorch or Transformers.
    """

    def __init__(
        self,
        *,
        torch: Any,
        model: Any,
        tokenizer: Any,
        device: Any,
        local_files_only: bool,
        observability_sampler: Any | None = None,
    ) -> None:
        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._observability_sampler = observability_sampler
        self.eos_token_ids = _normalize_eos_ids(model.config.eos_token_id)
        capability = torch.cuda.get_device_capability(device)
        self.metadata = BackendMetadata(
            python_version=platform.python_version(),
            python_executable_sha256=_python_executable_sha256(),
            python_platform_system=platform.system().lower(),
            python_platform_machine=platform.machine(),
            torch_version=_base_version(torch.__version__),
            transformers_version=TRANSFORMERS_VERSION,
            device=str(device),
            device_name=torch.cuda.get_device_name(device),
            compute_capability=f"{capability[0]}.{capability[1]}",
            local_files_only=local_files_only,
            weights_sha256=MODEL_WEIGHTS_SHA256,
            config_sha256=MODEL_CONFIG_SHA256,
            tokenizer_sha256=TOKENIZER_SHA256,
            tokenizer_files_sha256=dict(TOKENIZER_FILES_SHA256),
        )

    @classmethod
    def load(
        cls,
        *,
        device: str = "cuda:0",
        local_files_only: bool = True,
        enable_observability: bool = False,
    ) -> "HuggingFaceBackend":
        if platform.python_version() != PYTHON_VERSION:
            raise BackendUnavailableError(
                f"reference generation requires Python {PYTHON_VERSION}, "
                f"found {platform.python_version()}"
            )
        if (
            platform.system().lower() != PYTHON_PLATFORM_SYSTEM
            or platform.machine() != PYTHON_PLATFORM_MACHINE
        ):
            raise BackendUnavailableError(
                "reference generation requires the pinned linux/x86_64 Python runtime"
            )
        if _python_executable_sha256() != PYTHON_EXECUTABLE_SHA256:
            raise BackendUnavailableError(
                "resolved Python executable SHA-256 differs from immutable contract"
            )
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in (None, ":4096:8"):
            raise BackendUnavailableError(
                "CUBLAS_WORKSPACE_CONFIG must be unset or ':4096:8'"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import torch
            import transformers
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise BackendUnavailableError(
                "install the pinned reference project dependencies before generation"
            ) from error

        torch_version = _base_version(torch.__version__)
        transformers_version = _base_version(transformers.__version__)
        if torch_version != TORCH_VERSION:
            raise BackendUnavailableError(
                f"torch must be {TORCH_VERSION}, found {torch_version}"
            )
        if transformers_version != TRANSFORMERS_VERSION:
            raise BackendUnavailableError(
                f"transformers must be {TRANSFORMERS_VERSION}, found {transformers_version}"
            )
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise BackendUnavailableError("the canonical BF16 reference lane requires CUDA")
        if not torch.cuda.is_available():
            raise BackendUnavailableError("CUDA is unavailable")
        torch.cuda.set_device(resolved_device)
        if not torch.cuda.is_bf16_supported():
            raise BackendUnavailableError("the selected CUDA device does not support BF16")

        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            artifact_paths = {
                filename: hf_hub_download(
                    repo_id=MODEL_ID,
                    filename=filename,
                    revision=MODEL_REVISION,
                    local_files_only=local_files_only,
                )
                for filename in (
                    "model.safetensors",
                    "config.json",
                    *TOKENIZER_ARTIFACT_FILENAMES,
                )
            }
            weights_checksum = hashlib.sha256()
            with open(artifact_paths["model.safetensors"], "rb") as weights_file:
                while chunk := weights_file.read(8 * 1024 * 1024):
                    weights_checksum.update(chunk)
            if weights_checksum.hexdigest() != MODEL_WEIGHTS_SHA256:
                raise BackendUnavailableError(
                    "cached model.safetensors checksum differs from immutable contract"
                )
            artifact_hashes: dict[str, str] = {}
            for filename in ("config.json", *TOKENIZER_ARTIFACT_FILENAMES):
                checksum = hashlib.sha256()
                with open(artifact_paths[filename], "rb") as artifact_file:
                    for chunk in iter(
                        lambda: artifact_file.read(8 * 1024 * 1024), b""
                    ):
                        checksum.update(chunk)
                artifact_hashes[filename] = checksum.hexdigest()
            if artifact_hashes["config.json"] != MODEL_CONFIG_SHA256:
                raise BackendUnavailableError(
                    "cached config.json checksum differs from immutable contract"
                )
            tokenizer_hashes = {
                filename: artifact_hashes[filename]
                for filename in TOKENIZER_ARTIFACT_FILENAMES
            }
            if tokenizer_hashes != TOKENIZER_FILES_SHA256:
                raise BackendUnavailableError(
                    "cached tokenizer file checksums differ from immutable contract"
                )
            canonical_tokenizer_hashes = json.dumps(
                tokenizer_hashes,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if hashlib.sha256(canonical_tokenizer_hashes).hexdigest() != TOKENIZER_SHA256:
                raise BackendUnavailableError(
                    "cached tokenizer aggregate checksum differs from immutable contract"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=False,
                local_files_only=local_files_only,
                use_fast=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=False,
                local_files_only=local_files_only,
                use_safetensors=True,
                dtype=torch.bfloat16,
                attn_implementation=ATTENTION_BACKEND,
            )
        except Exception as error:
            mode = "local cache" if local_files_only else "Hugging Face Hub/cache"
            raise BackendUnavailableError(
                f"cannot load immutable model revision from {mode}: {error}"
            ) from error
        model.to(resolved_device)
        model.eval()
        backend = cls(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            local_files_only=local_files_only,
        )
        if enable_observability:
            backend._get_observability_sampler()
        return backend

    def _get_observability_sampler(self):
        """Initialize benchmark-only dependencies on first measured batch."""

        if self._observability_sampler is not None:
            return self._observability_sampler
        try:
            if importlib.metadata.version("nvidia-ml-py") != NVIDIA_ML_PY_VERSION:
                raise FixtureError(
                    f"nvidia-ml-py must be {NVIDIA_ML_PY_VERSION}"
                )
            if importlib.metadata.version("psutil") != PSUTIL_VERSION:
                raise FixtureError(f"psutil must be {PSUTIL_VERSION}")
            import psutil
            import pynvml

            from .observability import NvmlProcessTreeSampler

            device_index = self._device.index
            self._observability_sampler = NvmlProcessTreeSampler(
                nvml_module=pynvml,
                psutil_module=psutil,
                device_index=0 if device_index is None else int(device_index),
            )
        except FixtureError:
            raise
        except Exception as error:
            raise FixtureError(
                f"cannot load pinned benchmark observability dependencies: {error}"
            ) from error
        return self._observability_sampler

    @staticmethod
    def _resize_token_ids(token_ids: list[int], target: int) -> list[int]:
        if target <= 0:
            raise FixtureError("target_prompt_tokens must be positive for model execution")
        if not token_ids:
            raise FixtureError("cannot resize an empty token sequence")
        if len(token_ids) >= target:
            return token_ids[:target]
        return (token_ids * math.ceil(target / len(token_ids)))[:target]

    def _encode_one(
        self, text: str, target_prompt_tokens: int | None
    ) -> tuple[Any, Any]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if input_ids.shape[1] == 0:
            bos_token_id = self._tokenizer.bos_token_id
            if bos_token_id is None:
                raise FixtureError("tokenizer produced no tokens and defines no BOS token")
            input_ids = self._torch.tensor([[bos_token_id]], dtype=self._torch.long)
            attention_mask = self._torch.ones_like(input_ids)
        if target_prompt_tokens is not None:
            resized = self._resize_token_ids(
                [int(token_id) for token_id in input_ids[0].tolist()],
                target_prompt_tokens,
            )
            input_ids = self._torch.tensor([resized], dtype=self._torch.long)
            attention_mask = self._torch.ones_like(input_ids)
        return input_ids.to(self._device), attention_mask.to(self._device)

    def _tensor_values(self, tensor: Any):
        flat = (
            tensor.detach()
            .to(device="cpu", dtype=self._torch.float32)
            .contiguous()
            .reshape(-1)
        )
        for offset in range(0, flat.numel(), 65536):
            yield from flat[offset : offset + 65536].tolist()

    def _summary(self, tensor: Any, *, top_k: int | None = None) -> dict[str, object]:
        return summarize_values(
            self._tensor_values(tensor), tuple(tensor.shape), top_k=top_k
        )

    def _empty_cuda_cache(self) -> None:
        """Release dead eager-attention blocks between correctness forwards.

        Full-prefix eager attention grows its temporary score matrix at every
        decode step.  Without an explicit allocator flush, the slightly
        smaller blocks from earlier steps remain reserved and can accumulate
        until a long-context fixture hits an avoidable CUDA OOM.  This helper
        is deliberately absent from the timed benchmark path.
        """

        self._torch.cuda.empty_cache()

    def _greedy_cache_off(
        self, input_ids: Any, attention_mask: Any, max_new_tokens: int
    ) -> tuple[tuple[int, ...], str]:
        generated: list[int] = []
        current_ids = input_ids
        current_mask = attention_mask
        try:
            with self._torch.inference_mode():
                for step in range(max_new_tokens):
                    position_ids = self._torch.arange(
                        current_ids.shape[1],
                        device=self._device,
                        dtype=self._torch.long,
                    ).unsqueeze(0)
                    output = self._model(
                        input_ids=current_ids,
                        attention_mask=current_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    next_token = int(self._torch.argmax(output.logits[0, -1]).item())
                    generated.append(next_token)
                    # Drop the model output before returning allocator blocks.
                    # This is essential for monotonically growing eager score
                    # matrices near the 8192-token context boundary.
                    del output, position_ids
                    self._empty_cuda_cache()
                    if next_token in self.eos_token_ids:
                        return tuple(generated), "eos"
                    if step + 1 < max_new_tokens:
                        token_tensor = self._torch.tensor(
                            [[next_token]], device=self._device, dtype=current_ids.dtype
                        )
                        current_ids = self._torch.cat(
                            (current_ids, token_tensor), dim=1
                        )
                        current_mask = self._torch.cat(
                            (current_mask, self._torch.ones_like(token_tensor)), dim=1
                        )
            return tuple(generated), "max_new_tokens"
        finally:
            del current_ids, current_mask
            self._empty_cuda_cache()

    def _greedy_cache_on(
        self, input_ids: Any, attention_mask: Any, max_new_tokens: int
    ) -> tuple[tuple[int, ...], str]:
        generated: list[int] = []
        current_ids = input_ids
        current_mask = attention_mask
        past_key_values = None
        next_position = int(input_ids.shape[1])
        position_ids = self._torch.arange(
            next_position, device=self._device, dtype=self._torch.long
        ).unsqueeze(0)
        try:
            with self._torch.inference_mode():
                for step in range(max_new_tokens):
                    output = self._model(
                        input_ids=current_ids,
                        attention_mask=current_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    next_token = int(self._torch.argmax(output.logits[0, -1]).item())
                    next_cache = output.past_key_values
                    if next_cache is None:
                        raise FixtureError("cache-on forward returned no KV cache")
                    generated.append(next_token)
                    del output, position_ids
                    if next_token in self.eos_token_ids:
                        del next_cache
                        self._empty_cuda_cache()
                        return tuple(generated), "eos"
                    if step + 1 < max_new_tokens:
                        past_key_values = next_cache
                        current_ids = self._torch.tensor(
                            [[next_token]], device=self._device, dtype=input_ids.dtype
                        )
                        current_mask = self._torch.cat(
                            (current_mask, self._torch.ones_like(current_ids)), dim=1
                        )
                        position_ids = self._torch.tensor(
                            [[next_position]],
                            device=self._device,
                            dtype=self._torch.long,
                        )
                        next_position += 1
                    else:
                        del next_cache
                    self._empty_cuda_cache()
            return tuple(generated), "max_new_tokens"
        finally:
            del current_ids, current_mask, past_key_values
            self._empty_cuda_cache()

    def generate_case(
        self,
        text: str,
        *,
        max_new_tokens: int,
        hidden_state_index: int,
        top_k: int,
        target_prompt_tokens: int | None,
    ) -> CaseResult:
        if max_new_tokens > GOLDEN_GREEDY_MAX_NEW_TOKENS:
            raise FixtureError(
                "max_new_tokens exceeds the predeclared exact BF16 cache-parity "
                f"window of {GOLDEN_GREEDY_MAX_NEW_TOKENS}"
            )
        input_ids, attention_mask = self._encode_one(text, target_prompt_tokens)
        if input_ids.shape[1] + max_new_tokens > MAX_CONTEXT_TOKENS:
            raise FixtureError(
                f"tokenized prompt length {input_ids.shape[1]} plus max output "
                f"{max_new_tokens} exceeds {MAX_CONTEXT_TOKENS}"
            )
        with self._torch.inference_mode():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                logits_to_keep=1,
                return_dict=True,
            )
        if output.hidden_states is None or hidden_state_index >= len(output.hidden_states):
            available = 0 if output.hidden_states is None else len(output.hidden_states)
            raise FixtureError(
                f"hidden_state_index {hidden_state_index} unavailable; "
                f"model returned {available} states"
            )
        hidden = output.hidden_states[hidden_state_index]
        logits = output.logits[0, -1]
        processed_log_probs = self._torch.log_softmax(
            logits.to(dtype=self._torch.float32), dim=-1
        )
        hidden_summary = self._summary(hidden)
        logits_summary = self._summary(logits, top_k=top_k)
        processed_summary = self._summary(processed_log_probs, top_k=top_k)
        input_token_ids = tuple(int(token) for token in input_ids[0].tolist())
        del output, hidden, logits, processed_log_probs
        self._empty_cuda_cache()
        cache_on, cache_on_reason = self._greedy_cache_on(
            input_ids, attention_mask, max_new_tokens
        )
        self._empty_cuda_cache()
        cache_off, cache_off_reason = self._greedy_cache_off(
            input_ids, attention_mask, max_new_tokens
        )
        self._empty_cuda_cache()
        return CaseResult(
            input_token_ids=input_token_ids,
            hidden_state=hidden_summary,
            final_logits=logits_summary,
            processed_log_probs={
                "pipeline_id": "log-softmax-fp32-v1",
                "tensor": processed_summary,
            },
            cache_on_token_ids=cache_on,
            cache_off_token_ids=cache_off,
            cache_on_stop_reason=cache_on_reason,
            cache_off_stop_reason=cache_off_reason,
        )

    def _synchronize(self) -> None:
        self._torch.cuda.synchronize(self._device)

    def _materialize_benchmark_rows(
        self, texts: tuple[str, ...], prompt_tokens: int
    ) -> list[list[int]]:
        rows: list[list[int]] = []
        for text in texts:
            token_ids = self._tokenizer.encode(text, add_special_tokens=True)
            if not token_ids:
                fallback_token_id = (
                    self._tokenizer.bos_token_id
                    if self._tokenizer.bos_token_id is not None
                    else self._tokenizer.eos_token_id
                )
                if fallback_token_id is None:
                    raise FixtureError("tokenizer produced no benchmark tokens")
                token_ids = [fallback_token_id]
            rows.append(self._resize_token_ids(token_ids, prompt_tokens))
        return rows

    @staticmethod
    def _token_id_hashes(rows: list[list[int]]) -> tuple[str, ...]:
        hashes: list[str] = []
        for row in rows:
            digest = hashlib.sha256()
            for token_id in row:
                if not 0 <= token_id <= 0xFFFFFFFF:
                    raise FixtureError("token ID cannot be encoded as canonical u32")
                digest.update(struct.pack("<I", token_id))
            hashes.append(digest.hexdigest())
        return tuple(hashes)

    def prompt_token_ids_sha256(
        self, texts: tuple[str, ...], *, prompt_tokens: int
    ) -> tuple[str, ...]:
        return self._token_id_hashes(
            self._materialize_benchmark_rows(texts, prompt_tokens)
        )

    def environment(self) -> dict[str, object]:
        """Return the non-null environment fields required by result schema v1."""

        torch = self._torch
        try:
            device_index = self._device.index if self._device.index is not None else 0
            driver_version = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError) as error:
            raise FixtureError(f"cannot query NVIDIA driver version: {error}") from error
        cuda_version = str(torch.version.cuda or "unknown")
        cpu_model = platform.processor().strip()
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", maxsplit=1)[1].strip()
                        break
        except OSError:
            pass
        cpu_model = cpu_model or platform.machine().strip() or "unknown"
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        try:
            os_release = platform.freedesktop_os_release()
            os_text = (
                f"{os_release.get('NAME', 'Linux')} {os_release.get('VERSION_ID', '')}, "
                f"Linux {platform.release()}, {platform.machine()}"
            )
        except OSError:
            os_text = platform.platform()
        return {
            "gpu_model": self.metadata.device_name or "unknown",
            "compute_capability": self.metadata.compute_capability or "0.0",
            "gpu_count": 1,
            "cpu_model": cpu_model,
            "ram_bytes": int(page_size * page_count),
            "os": os_text,
            "nvidia_driver_version": driver_version,
            "cuda_toolkit_version": f"wheel-build-{cuda_version}",
            "cuda_runtime_version": cuda_version,
        }

    def benchmark_batch(
        self,
        texts: tuple[str, ...],
        *,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> BatchMeasurement:
        """Measure fixed-length cache-on greedy generation for one distinct-prompt batch."""

        if not texts:
            raise FixtureError("benchmark batch must contain at least one prompt")
        if prompt_tokens <= 0:
            raise FixtureError("benchmark prompt_tokens must be positive")
        if max_new_tokens <= 0:
            raise FixtureError("benchmark max_new_tokens must be positive")
        rows = self._materialize_benchmark_rows(texts, prompt_tokens)
        input_ids = self._torch.tensor(
            rows, device=self._device, dtype=self._torch.long
        )
        prompt_hashes = self._token_id_hashes(rows)
        attention_mask = self._torch.ones_like(input_ids)
        prompt_counts = (prompt_tokens,) * len(texts)
        if prompt_tokens + max_new_tokens > MAX_CONTEXT_TOKENS:
            raise FixtureError("benchmark prompt plus output exceeds model context")

        completions: list[float] = []
        generated_steps: list[Any] = []
        current_ids = input_ids
        current_mask = attention_mask
        past_key_values = None
        next_position = int(input_ids.shape[1])
        position_ids = self._torch.arange(
            input_ids.shape[1], device=self._device, dtype=self._torch.long
        ).unsqueeze(0)
        sampler = self._get_observability_sampler()
        self._synchronize()
        sampler.start()
        start = time.perf_counter()
        try:
            with self._torch.inference_mode():
                for _ in range(max_new_tokens):
                    output = self._model(
                        input_ids=current_ids,
                        attention_mask=current_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    next_tokens = self._torch.argmax(
                        output.logits[:, -1, :], dim=-1
                    )
                    generated_steps.append(next_tokens.detach())
                    past_key_values = output.past_key_values
                    current_ids = next_tokens.unsqueeze(1)
                    position_ids = self._torch.tensor(
                        [[next_position]], device=self._device, dtype=self._torch.long
                    )
                    next_position += 1
                    current_mask = self._torch.cat(
                        (
                            current_mask,
                            self._torch.ones(
                                (current_mask.shape[0], 1),
                                device=self._device,
                                dtype=current_mask.dtype,
                            ),
                        ),
                        dim=1,
                    )
                    del output
                    self._synchronize()
                    completions.append(time.perf_counter())
        finally:
            finished = time.perf_counter()
            observation = sampler.stop(wall_seconds=finished - start)
        generated_token_rows = (
            self._torch.stack(generated_steps, dim=1).to(device="cpu").tolist()
        )
        generated_rows = [
            [int(token_id) for token_id in row]
            for row in generated_token_rows
        ]
        generated_hashes = self._token_id_hashes(generated_rows)
        ttft = completions[0] - start
        itl = tuple(
            completions[index] - completions[index - 1]
            for index in range(1, len(completions))
        )
        elapsed = finished - start
        total_output_tokens = len(texts) * max_new_tokens
        return BatchMeasurement(
            prompt_token_counts=prompt_counts,
            prompt_token_ids_sha256=prompt_hashes,
            output_token_counts=(max_new_tokens,) * len(texts),
            generated_token_ids_sha256=generated_hashes,
            ttft_seconds=ttft,
            itl_seconds=itl,
            end_to_end_seconds=elapsed,
            output_tokens_per_second=total_output_tokens / elapsed,
            cpu_utilization_percent=observation.cpu_utilization_percent,
            gpu_utilization_percent=observation.gpu_utilization_percent,
            peak_gpu_memory_bytes=observation.peak_gpu_memory_bytes,
        )
