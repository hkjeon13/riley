"""Memory-bounded Hugging Face producers for calibration oracle artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .calibration import (
    BF16_ORACLE_KIND,
    CALIBRATION_SCHEMA_VERSION,
    CALIBRATION_TOP_K,
    CROSS_CACHE_EXACT_WINDOW,
    FP32_ORACLE_KIND,
    HF_ORACLE_REDUCTION_VARIANT,
    HF_SOURCE_PATHS,
    LOG_PROB_PIPELINE,
    MODEL_EOS_TOKEN_IDS,
    ORACLE_MANIFEST_GATE_ID,
    ORACLE_REQUIRED_CANDIDATE_REDUCTION_VARIANTS,
    SEMANTIC_GENERATION_STEPS,
    CalibrationError,
    aggregate_tokenizer_sha256,
    first_divergence,
    ranked_top_k,
    sha256_file,
    token_ids_sha256,
    top_k_token_set,
    utc_text,
    validate_calibration_manifest,
    write_json_exclusive,
)
from .constants import (
    ATTENTION_BACKEND,
    MAX_CONTEXT_TOKENS,
    MODEL_CONFIG_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PRIMARY_ENVIRONMENT_ID,
    PRIMARY_GPU_COMPUTE_CAPABILITY,
    PRIMARY_GPU_NAME,
    PYTHON_EXECUTABLE_SHA256,
    PYTHON_PLATFORM_MACHINE,
    PYTHON_PLATFORM_SYSTEM,
    PYTHON_VERSION,
    RUNTIME_DEPENDENCY_CLASS,
    SAFETENSORS_VERSION,
    TORCH_VERSION,
    TOKENIZER_ARTIFACT_FILENAMES,
    TOKENIZER_FILES_SHA256,
    TOKENIZER_SHA256,
    TRANSFORMERS_VERSION,
)
from .fixture import PromptRecord, load_prompts
from .environment import (
    EnvironmentContractError,
    EnvironmentProbe,
    probe_primary_environment,
    validate_environment_snapshot,
)


@dataclass(frozen=True)
class OracleArtifactMetadata:
    python_version: str
    python_executable_sha256: str
    python_platform_system: str
    python_platform_machine: str
    torch_version: str
    transformers_version: str
    safetensors_version: str
    config_sha256: str
    tokenizer_sha256: str
    tokenizer_files_sha256: Mapping[str, str]


@dataclass(frozen=True)
class CapturedOracleCase:
    input_token_ids: tuple[int, ...]
    first_layer_hidden: object
    final_logits: object
    final_log_probs: object
    semantic: Mapping[str, object] | None


class OracleBackend(Protocol):
    metadata: OracleArtifactMetadata

    def capture_case(self, prompt: PromptRecord) -> CapturedOracleCase: ...

    def close(self) -> None: ...


SidecarWriter = Callable[[Mapping[str, object], Path], None]
BackendFactory = Callable[..., OracleBackend]


def _base_version(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _normalize_eos_ids(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (list, tuple)):
        result = tuple(value)
        if all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and 0 <= item <= 0xFFFFFFFF
            for item in result
        ):
            return tuple(dict.fromkeys(result))
    raise CalibrationError("model config has invalid EOS token IDs")


def materialize_token_ids(
    tokenizer: object, text: str, target_prompt_tokens: int | None
) -> list[int]:
    token_ids = [
        int(token_id)
        for token_id in tokenizer.encode(text, add_special_tokens=True)
    ]
    if not token_ids:
        fallback = getattr(tokenizer, "bos_token_id", None)
        if fallback is None:
            fallback = getattr(tokenizer, "eos_token_id", None)
        if fallback is None:
            raise CalibrationError("tokenizer produced no tokens and has no BOS/EOS fallback")
        token_ids = [int(fallback)]
    if target_prompt_tokens is not None:
        if target_prompt_tokens <= 0:
            raise CalibrationError("target_prompt_tokens must be positive")
        if len(token_ids) >= target_prompt_tokens:
            token_ids = token_ids[:target_prompt_tokens]
        else:
            token_ids = (
                token_ids * math.ceil(target_prompt_tokens / len(token_ids))
            )[:target_prompt_tokens]
    token_ids_sha256(token_ids)
    if len(token_ids) + SEMANTIC_GENERATION_STEPS > MAX_CONTEXT_TOKENS:
        raise CalibrationError(
            f"{len(token_ids)} input + {SEMANTIC_GENERATION_STEPS} semantic tokens "
            f"exceed context {MAX_CONTEXT_TOKENS}"
        )
    return token_ids


def hidden_anchor_positions(token_count: int) -> dict[str, int]:
    if token_count <= 0:
        raise CalibrationError("hidden tensor requires at least one valid token")
    return {
        "first": 0,
        "middle": (token_count - 1) // 2,
        "last": token_count - 1,
    }


def _resolve_first_layer(model: object) -> object:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "base_model", None), "layers", None),
        getattr(
            getattr(getattr(model, "model", None), "decoder", None),
            "layers",
            None,
        ),
    )
    for layers in candidates:
        if layers is not None and len(layers) > 0:
            return layers[0]
    raise CalibrationError("cannot resolve the first transformer layer for capture")


class HuggingFaceCalibrationBackend:
    """One-process, one-dtype oracle producer; FP32 and BF16 run separately."""

    def __init__(
        self,
        *,
        artifact_kind: str,
        torch: object,
        model: object,
        tokenizer: object,
        device: object,
        metadata: OracleArtifactMetadata,
    ) -> None:
        self.artifact_kind = artifact_kind
        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._first_layer = _resolve_first_layer(model)
        self._eos_token_ids = _normalize_eos_ids(model.config.eos_token_id)
        if self._eos_token_ids != MODEL_EOS_TOKEN_IDS:
            raise CalibrationError("model EOS IDs differ from the immutable contract")
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        *,
        artifact_kind: str,
        device: str = "cuda:0",
        local_files_only: bool = True,
    ) -> "HuggingFaceCalibrationBackend":
        if artifact_kind not in {FP32_ORACLE_KIND, BF16_ORACLE_KIND}:
            raise CalibrationError("HF producer only creates FP32 or BF16 oracle artifacts")
        if platform.python_version() != PYTHON_VERSION:
            raise CalibrationError(
                f"calibration requires Python {PYTHON_VERSION}, "
                f"found {platform.python_version()}"
            )
        if (
            platform.system().lower() != PYTHON_PLATFORM_SYSTEM
            or platform.machine() != PYTHON_PLATFORM_MACHINE
        ):
            raise CalibrationError(
                "calibration requires the pinned linux/x86_64 Python runtime"
            )
        python_executable_sha256 = sha256_file(Path(sys.executable).resolve())
        if python_executable_sha256 != PYTHON_EXECUTABLE_SHA256:
            raise CalibrationError(
                "resolved Python executable SHA-256 differs from immutable contract"
            )
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in (None, ":4096:8"):
            raise CalibrationError("CUBLAS_WORKSPACE_CONFIG must be unset or ':4096:8'")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import torch
            import transformers
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise CalibrationError("install the pinned reference dependencies") from error
        versions = {
            "torch": _base_version(torch.__version__),
            "transformers": _base_version(transformers.__version__),
            "safetensors": _base_version(importlib.metadata.version("safetensors")),
        }
        expected = {
            "torch": TORCH_VERSION,
            "transformers": TRANSFORMERS_VERSION,
            "safetensors": SAFETENSORS_VERSION,
        }
        if versions != expected:
            raise CalibrationError(
                f"calibration dependency pins differ: expected={expected}, found={versions}"
            )
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" or not torch.cuda.is_available():
            raise CalibrationError("canonical calibration requires CUDA")
        torch.cuda.set_device(resolved_device)
        capability = torch.cuda.get_device_capability(resolved_device)
        device_name = torch.cuda.get_device_name(resolved_device)
        if device_name != PRIMARY_GPU_NAME or f"{capability[0]}.{capability[1]}" != PRIMARY_GPU_COMPUTE_CAPABILITY:
            raise CalibrationError("calibration device differs from the primary RTX 4090 contract")
        if artifact_kind == BF16_ORACLE_KIND and not torch.cuda.is_bf16_supported():
            raise CalibrationError("selected CUDA device does not support BF16")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        filenames = (
            "config.json",
            "model.safetensors",
            *TOKENIZER_ARTIFACT_FILENAMES,
        )
        try:
            paths = {
                filename: Path(
                    hf_hub_download(
                        repo_id=MODEL_ID,
                        filename=filename,
                        revision=MODEL_REVISION,
                        local_files_only=local_files_only,
                    )
                )
                for filename in filenames
            }
        except Exception as error:
            mode = "local cache" if local_files_only else "Hugging Face Hub/cache"
            raise CalibrationError(
                f"cannot resolve immutable calibration artifacts from {mode}: {error}"
            ) from error
        weights_sha256 = sha256_file(paths["model.safetensors"])
        if weights_sha256 != MODEL_WEIGHTS_SHA256:
            raise CalibrationError("model.safetensors SHA-256 differs from immutable contract")
        config_sha256 = sha256_file(paths["config.json"])
        if config_sha256 != MODEL_CONFIG_SHA256:
            raise CalibrationError("config.json SHA-256 differs from immutable contract")
        tokenizer_file_hashes = {
            filename: sha256_file(paths[filename])
            for filename in TOKENIZER_ARTIFACT_FILENAMES
        }
        if tokenizer_file_hashes != TOKENIZER_FILES_SHA256:
            raise CalibrationError(
                "tokenizer artifact SHA-256 values differ from immutable contract"
            )
        tokenizer_sha256 = aggregate_tokenizer_sha256(tokenizer_file_hashes)
        if tokenizer_sha256 != TOKENIZER_SHA256:
            raise CalibrationError(
                "tokenizer aggregate SHA-256 differs from immutable contract"
            )
        metadata = OracleArtifactMetadata(
            python_version=platform.python_version(),
            python_executable_sha256=python_executable_sha256,
            python_platform_system=platform.system().lower(),
            python_platform_machine=platform.machine(),
            torch_version=versions["torch"],
            transformers_version=versions["transformers"],
            safetensors_version=versions["safetensors"],
            config_sha256=config_sha256,
            tokenizer_sha256=tokenizer_sha256,
            tokenizer_files_sha256=tokenizer_file_hashes,
        )
        dtype = torch.float32 if artifact_kind == FP32_ORACLE_KIND else torch.bfloat16
        try:
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
                dtype=dtype,
                attn_implementation=ATTENTION_BACKEND,
            )
        except Exception as error:
            raise CalibrationError(f"cannot load immutable calibration model: {error}") from error
        model.to(resolved_device)
        model.eval()
        return cls(
            artifact_kind=artifact_kind,
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            metadata=metadata,
        )

    def _empty_cache(self) -> None:
        self._torch.cuda.empty_cache()

    def _input_tensors(self, prompt: PromptRecord) -> tuple[tuple[int, ...], object, object]:
        token_ids = tuple(
            materialize_token_ids(
                self._tokenizer, prompt.text, prompt.target_prompt_tokens
            )
        )
        input_ids = self._torch.tensor(
            [token_ids], device=self._device, dtype=self._torch.long
        )
        attention_mask = self._torch.ones_like(input_ids)
        return token_ids, input_ids, attention_mask

    def _capture_numeric(self, input_ids: object, attention_mask: object) -> tuple[object, object, object]:
        captured: dict[str, object] = {}

        def capture_first_layer(_module: object, _args: object, output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if len(hidden.shape) != 3 or hidden.shape[0] != 1:
                raise CalibrationError("first-layer hook returned an unexpected shape")
            # The fixed thresholds were calibrated on the full selected-layer
            # tensor. first/middle/last are manifest anchors, not a sampled metric.
            captured["hidden"] = hidden[0].detach().to(device="cpu").contiguous()

        hook = self._first_layer.register_forward_hook(capture_first_layer)
        position_ids = self._torch.arange(
            input_ids.shape[1], device=self._device, dtype=self._torch.long
        ).unsqueeze(0)
        try:
            with self._torch.inference_mode():
                output = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            if "hidden" not in captured:
                raise CalibrationError("first-layer capture hook did not execute")
            logits_gpu = output.logits[0, -1]
            logits = logits_gpu.detach().to(device="cpu").contiguous()
            log_probs = self._torch.log_softmax(
                logits_gpu.to(dtype=self._torch.float32), dim=-1
            ).detach().to(device="cpu").contiguous()
            hidden = captured["hidden"]
            del output, logits_gpu, position_ids
            return hidden, logits, log_probs
        finally:
            hook.remove()
            self._empty_cache()

    def _greedy_cache_off(
        self, input_ids: object, attention_mask: object
    ) -> tuple[tuple[int, ...], str]:
        generated: list[int] = []
        current_ids = input_ids
        current_mask = attention_mask
        try:
            with self._torch.inference_mode():
                for step in range(SEMANTIC_GENERATION_STEPS):
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
                    del output, position_ids
                    self._empty_cache()
                    if next_token in self._eos_token_ids:
                        return tuple(generated), "eos"
                    if step + 1 < SEMANTIC_GENERATION_STEPS:
                        token = self._torch.tensor(
                            [[next_token]], device=self._device, dtype=current_ids.dtype
                        )
                        current_ids = self._torch.cat((current_ids, token), dim=1)
                        current_mask = self._torch.cat(
                            (current_mask, self._torch.ones_like(token)), dim=1
                        )
            return tuple(generated), "max_new_tokens"
        finally:
            del current_ids, current_mask
            self._empty_cache()

    def _greedy_cache_on(
        self, input_ids: object, attention_mask: object
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
                for step in range(SEMANTIC_GENERATION_STEPS):
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
                        raise CalibrationError("cache-on calibration returned no KV cache")
                    generated.append(next_token)
                    del output, position_ids
                    if next_token in self._eos_token_ids:
                        del next_cache
                        self._empty_cache()
                        return tuple(generated), "eos"
                    if step + 1 < SEMANTIC_GENERATION_STEPS:
                        past_key_values = next_cache
                        current_ids = self._torch.tensor(
                            [[next_token]], device=self._device, dtype=input_ids.dtype
                        )
                        current_mask = self._torch.cat(
                            (current_mask, self._torch.ones_like(current_ids)), dim=1
                        )
                        position_ids = self._torch.tensor(
                            [[next_position]], device=self._device, dtype=self._torch.long
                        )
                        next_position += 1
                    else:
                        del next_cache
                    self._empty_cache()
            return tuple(generated), "max_new_tokens"
        finally:
            del current_ids, current_mask, past_key_values
            self._empty_cache()

    def capture_case(self, prompt: PromptRecord) -> CapturedOracleCase:
        token_ids, input_ids, attention_mask = self._input_tensors(prompt)
        hidden, logits, log_probs = self._capture_numeric(input_ids, attention_mask)
        semantic: Mapping[str, object] | None = None
        if self.artifact_kind == BF16_ORACLE_KIND:
            ranked = ranked_top_k(logits.detach().float().tolist(), CALIBRATION_TOP_K)
            cache_on, cache_on_reason = self._greedy_cache_on(input_ids, attention_mask)
            self._empty_cache()
            cache_off, cache_off_reason = self._greedy_cache_off(input_ids, attention_mask)
            divergence = first_divergence(cache_on, cache_off)
            if not cache_on or not cache_off:
                raise CalibrationError(f"{prompt.prompt_id}: greedy path generated no token")
            if cache_on[0] != ranked[0] or cache_off[0] != ranked[0]:
                raise CalibrationError(
                    f"{prompt.prompt_id}: captured top-1 differs from first greedy token"
                )
            semantic = {
                "top_1_token_id": ranked[0],
                "top_k_token_id_set": sorted(ranked),
                "cache_on": {
                    "generated_token_ids": list(cache_on),
                    "stop_reason": cache_on_reason,
                },
                "cache_off": {
                    "generated_token_ids": list(cache_off),
                    "stop_reason": cache_off_reason,
                },
                "cross_cache_first_divergence_step": divergence,
                "cross_cache_exact_window_match": (
                    divergence is None or divergence >= CROSS_CACHE_EXACT_WINDOW
                ),
            }
        del input_ids, attention_mask
        self._empty_cache()
        return CapturedOracleCase(
            input_token_ids=token_ids,
            first_layer_hidden=hidden,
            final_logits=logits,
            final_log_probs=log_probs,
            semantic=semantic,
        )

    def close(self) -> None:
        model = self._model
        self._model = None
        self._first_layer = None
        del model
        self._empty_cache()


def _canonical_tensor_dtype(tensor: object) -> str:
    value = str(tensor.dtype)
    if value in {"torch.float32", "float32", "F32"}:
        return "float32"
    if value in {"torch.bfloat16", "bfloat16", "BF16"}:
        return "bfloat16"
    raise CalibrationError(f"captured tensor has unsupported dtype {value!r}")


def _tensor_ref(tensor: object, key: str) -> dict[str, object]:
    shape = [int(dimension) for dimension in tensor.shape]
    if not shape or any(dimension <= 0 for dimension in shape):
        raise CalibrationError(f"captured tensor {key!r} has invalid shape")
    return {
        "key": key,
        "shape": shape,
        "dtype": _canonical_tensor_dtype(tensor),
        "cache_path": "off",
    }


def repository_provenance(
    repo_root: Path, *, observed_environment: Mapping[str, object]
) -> dict[str, object]:
    root = repo_root.resolve()
    producer_paths = (
        *HF_SOURCE_PATHS.values(),
        "tools/python/reference/pyproject.toml",
        "tools/python/reference/rustinfer_reference/calibration.py",
        "tools/python/reference/rustinfer_reference/hf_calibration.py",
        "tools/python/reference/rustinfer_reference/oracle_calibration.py",
        "tools/python/reference/rustinfer_reference/cli.py",
    )
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.decode("ascii").strip()
        tracked_status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
                "--",
                ".",
                ":(exclude)benchmarks/results",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        for relative in producer_paths:
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=root,
                check=True,
                capture_output=True,
            )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise CalibrationError(f"cannot resolve Git provenance: {error}") from error
    if tracked_status:
        raise CalibrationError("calibration requires a clean Git worktree")
    sources = {
        name: {
            "path": relative,
            "sha256": sha256_file(root / relative),
        }
        for name, relative in HF_SOURCE_PATHS.items()
    }
    try:
        validate_environment_snapshot(observed_environment)
    except EnvironmentContractError as error:
        raise CalibrationError(f"primary environment preflight failed: {error}") from error
    return {
        "sources": sources,
        "git_revision": revision,
        "git_dirty": False,
        "git_status_sha256": hashlib.sha256(tracked_status).hexdigest(),
        "environment_id": PRIMARY_ENVIRONMENT_ID,
        "observed_environment": dict(observed_environment),
    }


def _default_sidecar_writer(tensors: Mapping[str, object], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise CalibrationError("install pinned safetensors to write calibration") from error
    save_file(dict(tensors), str(path))


def _write_sidecar_exclusive(
    path: Path, tensors: Mapping[str, object], writer: SidecarWriter
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CalibrationError(f"refusing to overwrite existing sidecar: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        writer(tensors, temporary)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise CalibrationError(f"refusing to overwrite existing sidecar: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def produce_hf_oracle(
    *,
    artifact_kind: str,
    prompts_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    repo_root: Path,
    device: str,
    local_files_only: bool,
    created_at: datetime,
    backend_factory: BackendFactory = HuggingFaceCalibrationBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    environment_probe: EnvironmentProbe = probe_primary_environment,
) -> dict[str, object]:
    """Produce one dtype in one process, streaming prompt work off the GPU."""

    if artifact_kind not in {FP32_ORACLE_KIND, BF16_ORACLE_KIND}:
        raise CalibrationError("artifact_kind must select an HF oracle role")
    if manifest_path.exists() or sidecar_path.exists():
        raise CalibrationError("refusing to overwrite an existing calibration artifact")
    repository = repo_root.resolve()
    for output in (manifest_path.resolve(), sidecar_path.resolve()):
        if output == repository or repository in output.parents:
            raise CalibrationError(
                "canonical oracle artifacts must be written outside the repository"
            )
    if manifest_path.resolve().parent != sidecar_path.resolve().parent:
        raise CalibrationError("manifest and sidecar must be sibling files")
    if sidecar_path.suffix != ".safetensors":
        raise CalibrationError("calibration sidecar must use .safetensors")
    try:
        observed_environment = environment_probe()
        validate_environment_snapshot(observed_environment)
    except EnvironmentContractError as error:
        raise CalibrationError(f"primary environment preflight failed: {error}") from error
    prompts, corpus_sha256 = load_prompts(prompts_path)
    provenance = repository_provenance(
        repo_root, observed_environment=observed_environment
    )
    if provenance["sources"]["prompts"]["sha256"] != corpus_sha256:
        raise CalibrationError("--prompts does not match the repository-bound prompt corpus")
    backend = backend_factory(
        artifact_kind=artifact_kind,
        device=device,
        local_files_only=local_files_only,
    )
    tensors: dict[str, object] = {}
    cases: list[dict[str, object]] = []
    variant_id = str(HF_ORACLE_REDUCTION_VARIANT["variant_id"])
    try:
        for prompt in prompts:
            captured = backend.capture_case(prompt)
            count = len(captured.input_token_ids)
            prefix = f"cases/{prompt.prompt_id}/{variant_id}"
            case_tensors = {
                "first_layer_hidden": captured.first_layer_hidden,
                "final_logits": captured.final_logits,
                "final_log_probs": captured.final_log_probs,
            }
            refs: dict[str, object] = {}
            for tensor_name, tensor in case_tensors.items():
                key = f"{prefix}/{tensor_name}"
                tensors[key] = tensor
                refs[tensor_name] = _tensor_ref(tensor, key)
            cases.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "prompt_text_sha256": hashlib.sha256(
                        prompt.text.encode("utf-8")
                    ).hexdigest(),
                    "prompt_metadata": prompt.metadata,
                    "input_token_ids_sha256": token_ids_sha256(
                        captured.input_token_ids
                    ),
                    "input_first_token_id": captured.input_token_ids[0],
                    "input_token_count": count,
                    "hidden_anchor_positions": hidden_anchor_positions(count),
                    "variants": {
                        variant_id: {
                            "config": dict(HF_ORACLE_REDUCTION_VARIANT),
                            "tensors": refs,
                            "semantic": captured.semantic,
                        }
                    },
                }
            )
    finally:
        backend.close()
    metadata = backend.metadata
    manifest: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "created_at": utc_text(created_at),
        "producer": {
            "implementation_id": "hf-transformers-eager",
            "engine_revision": f"transformers-{TRANSFORMERS_VERSION}+torch-{TORCH_VERSION}",
            "runtime_dependency_class": RUNTIME_DEPENDENCY_CLASS,
            "python_version": metadata.python_version,
            "python_executable_sha256": metadata.python_executable_sha256,
            "python_platform_system": metadata.python_platform_system,
            "python_platform_machine": metadata.python_platform_machine,
            "torch_version": metadata.torch_version,
            "transformers_version": metadata.transformers_version,
            "safetensors_version": metadata.safetensors_version,
        },
        "candidate_execution": None,
        "contract": {
            "gate_id": ORACLE_MANIFEST_GATE_ID,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config_sha256": metadata.config_sha256,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "tokenizer_sha256": metadata.tokenizer_sha256,
            "tokenizer_files_sha256": dict(metadata.tokenizer_files_sha256),
            "dtype": "float32" if artifact_kind == FP32_ORACLE_KIND else "bfloat16",
            "attention_backend": ATTENTION_BACKEND,
            "tensor_capture_cache_path": "off",
            "log_prob_pipeline": LOG_PROB_PIPELINE,
            "trust_remote_code": False,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "eos_token_ids": list(MODEL_EOS_TOKEN_IDS),
            "semantic_generation_steps": SEMANTIC_GENERATION_STEPS,
            "cross_cache_exact_window": CROSS_CACHE_EXACT_WINDOW,
            "top_k": CALIBRATION_TOP_K,
            "oracle_reduction_variant": dict(HF_ORACLE_REDUCTION_VARIANT),
            "required_candidate_reduction_variants": [],
        },
        "provenance": provenance,
        "corpus": {"prompt_count": len(cases)},
        "sidecar": {
            "path": sidecar_path.name,
            "sha256": "0" * 64,
            "format": "safetensors",
            "tensor_count": len(tensors),
        },
        "cases": cases,
    }
    # The oracle declares candidate profiles in the gate contract too; no
    # duplicated oracle tensors are created for those execution profiles.
    manifest["contract"]["required_candidate_reduction_variants"] = [
        dict(variant) for variant in ORACLE_REQUIRED_CANDIDATE_REDUCTION_VARIANTS
    ]
    _write_sidecar_exclusive(sidecar_path, tensors, sidecar_writer)
    try:
        manifest["sidecar"]["sha256"] = sha256_file(sidecar_path)
        validate_calibration_manifest(manifest)
        write_json_exclusive(manifest_path, manifest)
    except BaseException:
        try:
            sidecar_path.unlink()
        except OSError:
            pass
        raise
    return manifest
