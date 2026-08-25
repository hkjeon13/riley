"""Pinned, remote-only Hugging Face producer for the PR 07 forward trace.

The calibration producers deliberately remain unchanged.  This module wraps the
same immutable BF16 loader, but captures one fixed token-ID sequence at explicit
Llama module boundaries for reference-forward bring-up.  Importing this module
does not import torch; the CUDA/model dependency stays behind ``load``.
"""

from __future__ import annotations

import inspect
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .calibration import (
    BF16_ORACLE_KIND,
    CalibrationError,
    parse_utc,
    sha256_file,
    token_ids_sha256,
    utc_text,
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
    PYTHON_EXECUTABLE_SHA256,
    PYTHON_PLATFORM_MACHINE,
    PYTHON_PLATFORM_SYSTEM,
    PYTHON_VERSION,
    RUNTIME_DEPENDENCY_CLASS,
    SAFETENSORS_VERSION,
    TOKENIZER_FILES_SHA256,
    TOKENIZER_SHA256,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
)
from .environment import (
    EnvironmentContractError,
    EnvironmentProbe,
    probe_primary_environment,
    validate_environment_snapshot,
)
from .hf_calibration import (
    HuggingFaceCalibrationBackend,
    OracleArtifactMetadata,
    SidecarWriter,
    _default_sidecar_writer,
    _write_sidecar_exclusive,
    repository_provenance,
)


PR07_TRACE_SCHEMA_VERSION = "1.0.0"
PR07_TRACE_ARTIFACT_KIND = "pr07-hf-bf16-forward-trace"
PR07_TRACE_ID = "smollm2-pr07-bf16-reference-forward-v1"
PR07_TRACE_TOKEN_IDS = (504, 2365, 6354, 16438, 11139, 253, 1890)
PR07_TRACE_TOKEN_IDS_SHA256 = token_ids_sha256(PR07_TRACE_TOKEN_IDS)

MODEL_HIDDEN_SIZE = 576
MODEL_INTERMEDIATE_SIZE = 1536
MODEL_LAYER_COUNT = 30
MODEL_QUERY_HEAD_COUNT = 9
MODEL_KV_HEAD_COUNT = 3
MODEL_HEAD_DIM = 64
MODEL_VOCAB_SIZE = 49_152
MODEL_RMS_NORM_EPS = 1e-5
MODEL_ROPE_THETA = 100_000

TRANSFORMERS_LLAMA_SOURCE_PATH = "transformers/models/llama/modeling_llama.py"
TRANSFORMERS_LLAMA_SOURCE_SHA256 = (
    "13e65b752a9c9d8a5c22b83df73009a8940c0eefdc58c101df3eb910e3efc2f9"
)

PR07_TRACE_SOURCE_PATHS = {
    "plan": "deploy/07-llama-reference-forward.md",
    "dependency_lock": "tools/python/reference/uv.lock",
    "python_version_file": "tools/python/reference/.python-version",
    "lane_project": "tools/python/reference/pyproject.toml",
    "immutable_pins": "tools/python/reference/rustinfer_reference/constants.py",
    "calibration_contract": "tools/python/reference/rustinfer_reference/calibration.py",
    "hf_immutable_loader": "tools/python/reference/rustinfer_reference/hf_calibration.py",
    "trace_producer": "tools/python/reference/rustinfer_reference/pr07_trace.py",
    "cli": "tools/python/reference/rustinfer_reference/cli.py",
}

PR07_TRACE_TENSOR_NAMES = (
    "embedding",
    "layer0.input_norm",
    "layer0.q_proj",
    "layer0.k_proj",
    "layer0.v_proj",
    "layer0.attention_probs",
    "layer0.attention_context",
    "layer0.after_attention_residual",
    "layer0.post_attention_norm",
    "layer0.gate_proj",
    "layer0.up_proj",
    "layer0.gated",
    "layer0.down_proj",
    "layer0.output",
    "layer14.output",
    "final_norm.input",
    "final_norm.output",
    "last_logits",
)

_EXPECTED_LLAMA_TYPES = {
    "causal_lm": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
    "model": "transformers.models.llama.modeling_llama.LlamaModel",
    "layer": "transformers.models.llama.modeling_llama.LlamaDecoderLayer",
    "attention": "transformers.models.llama.modeling_llama.LlamaAttention",
    "mlp": "transformers.models.llama.modeling_llama.LlamaMLP",
    "norm": "transformers.models.llama.modeling_llama.LlamaRMSNorm",
}

TraceBackendFactory = Callable[..., "Pr07TraceBackendProtocol"]
TraceProvenanceFactory = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class CapturedPr07Trace:
    tensors: Mapping[str, object]


class Pr07TraceBackendProtocol(Protocol):
    metadata: OracleArtifactMetadata
    transformers_llama_source_path: str
    transformers_llama_source_sha256: str

    def capture_trace(self) -> CapturedPr07Trace: ...

    def close(self) -> None: ...


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


def _expect_qualified_type(value: object, expected: str, label: str) -> None:
    observed = _qualified_type(value)
    if observed != expected:
        raise CalibrationError(
            f"PR07 trace requires {label} type {expected}, found {observed}"
        )


def _expect_config_value(config: object, name: str, expected: object) -> None:
    observed = getattr(config, name, None)
    if observed != expected or type(observed) is not type(expected):
        raise CalibrationError(
            f"PR07 trace config.{name} differs: expected={expected!r}, "
            f"found={observed!r}"
        )


def _validate_llama_topology(model: object) -> tuple[object, Sequence[object]]:
    """Fail closed on every Transformers module boundary used by the hooks."""

    _expect_qualified_type(model, _EXPECTED_LLAMA_TYPES["causal_lm"], "model")
    base_model = getattr(model, "model", None)
    _expect_qualified_type(base_model, _EXPECTED_LLAMA_TYPES["model"], "base model")
    config = getattr(model, "config", None)
    if config is None or config is not getattr(base_model, "config", None):
        raise CalibrationError("PR07 trace model/config identity is unexpected")

    exact_config = {
        "model_type": "llama",
        "hidden_size": MODEL_HIDDEN_SIZE,
        "intermediate_size": MODEL_INTERMEDIATE_SIZE,
        "num_hidden_layers": MODEL_LAYER_COUNT,
        "num_attention_heads": MODEL_QUERY_HEAD_COUNT,
        "num_key_value_heads": MODEL_KV_HEAD_COUNT,
        "head_dim": MODEL_HEAD_DIM,
        "vocab_size": MODEL_VOCAB_SIZE,
        "max_position_embeddings": MAX_CONTEXT_TOKENS,
        "rms_norm_eps": MODEL_RMS_NORM_EPS,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
    }
    for name, expected in exact_config.items():
        _expect_config_value(config, name, expected)
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, Mapping) or dict(rope_parameters) != {
        "rope_theta": MODEL_ROPE_THETA,
        "rope_type": "default",
    }:
        raise CalibrationError("PR07 trace requires the pinned default RoPE parameters")
    if getattr(config, "_attn_implementation", None) != ATTENTION_BACKEND:
        raise CalibrationError("PR07 trace requires the eager attention implementation")

    layers = getattr(base_model, "layers", None)
    if layers is None or len(layers) != MODEL_LAYER_COUNT:
        raise CalibrationError(
            f"PR07 trace requires exactly {MODEL_LAYER_COUNT} decoder layers"
        )
    for index, layer in enumerate(layers):
        _expect_qualified_type(
            layer, _EXPECTED_LLAMA_TYPES["layer"], f"decoder layer {index}"
        )
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        input_norm = getattr(layer, "input_layernorm", None)
        post_norm = getattr(layer, "post_attention_layernorm", None)
        _expect_qualified_type(
            attention, _EXPECTED_LLAMA_TYPES["attention"], f"layer {index} attention"
        )
        _expect_qualified_type(mlp, _EXPECTED_LLAMA_TYPES["mlp"], f"layer {index} MLP")
        _expect_qualified_type(
            input_norm, _EXPECTED_LLAMA_TYPES["norm"], f"layer {index} input norm"
        )
        _expect_qualified_type(
            post_norm, _EXPECTED_LLAMA_TYPES["norm"], f"layer {index} post norm"
        )
        if getattr(attention, "layer_idx", None) != index:
            raise CalibrationError(f"PR07 trace layer {index} has the wrong attention index")
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not hasattr(attention, projection):
                raise CalibrationError(
                    f"PR07 trace layer {index} lacks attention.{projection}"
                )
        for projection in ("gate_proj", "up_proj", "down_proj"):
            if not hasattr(mlp, projection):
                raise CalibrationError(f"PR07 trace layer {index} lacks MLP.{projection}")
    _expect_qualified_type(
        getattr(base_model, "norm", None),
        _EXPECTED_LLAMA_TYPES["norm"],
        "final norm",
    )
    if not hasattr(base_model, "embed_tokens") or not hasattr(model, "lm_head"):
        raise CalibrationError("PR07 trace model lacks embedding or LM head")
    embedding_weight = getattr(base_model.embed_tokens, "weight", None)
    lm_head_weight = getattr(model.lm_head, "weight", None)
    if embedding_weight is None or lm_head_weight is not embedding_weight:
        raise CalibrationError("PR07 trace requires the pinned tied LM-head weight")
    return base_model, layers


def _module_source_sha256(model: object) -> tuple[str, str]:
    source_path_text = inspect.getsourcefile(type(model))
    if source_path_text is None:
        raise CalibrationError("cannot resolve the pinned Llama module source")
    source_path = Path(source_path_text).resolve()
    observed = sha256_file(source_path)
    if observed != TRANSFORMERS_LLAMA_SOURCE_SHA256:
        raise CalibrationError(
            "Transformers Llama source differs from the pinned 5.15.1 hook contract"
        )
    return TRANSFORMERS_LLAMA_SOURCE_PATH, observed


def _expected_trace_shapes() -> dict[str, list[int]]:
    sequence_length = len(PR07_TRACE_TOKEN_IDS)
    hidden = [sequence_length, MODEL_HIDDEN_SIZE]
    intermediate = [sequence_length, MODEL_INTERMEDIATE_SIZE]
    return {
        "embedding": hidden,
        "layer0.input_norm": hidden,
        "layer0.q_proj": hidden,
        "layer0.k_proj": [sequence_length, MODEL_KV_HEAD_COUNT * MODEL_HEAD_DIM],
        "layer0.v_proj": [sequence_length, MODEL_KV_HEAD_COUNT * MODEL_HEAD_DIM],
        "layer0.attention_probs": [
            MODEL_QUERY_HEAD_COUNT,
            sequence_length,
            sequence_length,
        ],
        "layer0.attention_context": [
            sequence_length,
            MODEL_QUERY_HEAD_COUNT,
            MODEL_HEAD_DIM,
        ],
        "layer0.after_attention_residual": hidden,
        "layer0.post_attention_norm": hidden,
        "layer0.gate_proj": intermediate,
        "layer0.up_proj": intermediate,
        "layer0.gated": intermediate,
        "layer0.down_proj": hidden,
        "layer0.output": hidden,
        "layer14.output": hidden,
        "final_norm.input": hidden,
        "final_norm.output": hidden,
        "last_logits": [MODEL_VOCAB_SIZE],
    }


def _canonical_tensor_dtype(tensor: object) -> str:
    observed = str(getattr(tensor, "dtype", ""))
    if observed in {"torch.bfloat16", "bfloat16", "BF16"}:
        return "bfloat16"
    raise CalibrationError(f"PR07 trace tensor has unsupported dtype {observed!r}")


def _tensor_shape(tensor: object) -> list[int]:
    try:
        shape = [int(dimension) for dimension in tensor.shape]
    except (AttributeError, TypeError, ValueError) as error:
        raise CalibrationError("PR07 trace tensor has no valid shape") from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise CalibrationError("PR07 trace tensor has an empty or invalid shape")
    return shape


def _validate_trace_tensors(tensors: Mapping[str, object]) -> None:
    expected_shapes = _expected_trace_shapes()
    if tuple(tensors) != PR07_TRACE_TENSOR_NAMES:
        raise CalibrationError("PR07 trace tensor names/order differ from the contract")
    identities: set[int] = set()
    for name in PR07_TRACE_TENSOR_NAMES:
        tensor = tensors[name]
        shape = _tensor_shape(tensor)
        if shape != expected_shapes[name]:
            raise CalibrationError(
                f"PR07 trace tensor {name!r} shape differs: "
                f"expected={expected_shapes[name]}, found={shape}"
            )
        if _canonical_tensor_dtype(tensor) != "bfloat16":
            raise CalibrationError(f"PR07 trace tensor {name!r} must be BF16")
        identity = id(tensor)
        if identity in identities:
            raise CalibrationError("PR07 trace tensors must be distinct captures")
        identities.add(identity)


class HuggingFacePr07TraceBackend:
    """Version-pinned hook adapter over ``HuggingFaceCalibrationBackend``."""

    def __init__(self, calibration_backend: HuggingFaceCalibrationBackend) -> None:
        if calibration_backend.artifact_kind != BF16_ORACLE_KIND:
            raise CalibrationError("PR07 trace backend requires the BF16 HF loader")
        self._calibration_backend = calibration_backend
        self._torch = calibration_backend._torch
        self._model = calibration_backend._model
        self._device = calibration_backend._device
        self.metadata = calibration_backend.metadata
        self._base_model, self._layers = _validate_llama_topology(self._model)
        (
            self.transformers_llama_source_path,
            self.transformers_llama_source_sha256,
        ) = _module_source_sha256(self._model)

    @classmethod
    def load(
        cls, *, device: str = "cuda:0", local_files_only: bool = True
    ) -> "HuggingFacePr07TraceBackend":
        if device != "cuda:0":
            raise CalibrationError("canonical PR07 trace capture requires --device cuda:0")
        calibration_backend = HuggingFaceCalibrationBackend.load(
            artifact_kind=BF16_ORACLE_KIND,
            device=device,
            local_files_only=local_files_only,
        )
        try:
            return cls(calibration_backend)
        except BaseException:
            calibration_backend.close()
            raise

    def _capture_cpu(
        self, captured: dict[str, object], name: str, tensor: object
    ) -> None:
        if name in captured:
            raise CalibrationError(f"PR07 trace hook {name!r} executed more than once")
        try:
            cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        except (AttributeError, RuntimeError) as error:
            raise CalibrationError(
                f"PR07 trace hook {name!r} did not receive a tensor"
            ) from error
        captured[name] = cpu_tensor

    def _capture_without_batch(
        self, captured: dict[str, object], name: str, tensor: object
    ) -> None:
        try:
            if len(tensor.shape) < 2 or int(tensor.shape[0]) != 1:
                raise CalibrationError(
                    f"PR07 trace hook {name!r} requires a single batch dimension"
                )
            value = tensor[0]
        except (AttributeError, IndexError, TypeError) as error:
            raise CalibrationError(
                f"PR07 trace hook {name!r} returned an unexpected value"
            ) from error
        self._capture_cpu(captured, name, value)

    def capture_trace(self) -> CapturedPr07Trace:
        torch = self._torch
        model = self._model
        if model is None:
            raise CalibrationError("PR07 trace backend is closed")
        layer0 = self._layers[0]
        layer14 = self._layers[14]
        attention = layer0.self_attn
        mlp = layer0.mlp
        captured: dict[str, object] = {}
        handles: list[object] = []

        def capture_output(name: str):
            def hook(_module: object, _args: object, output: object) -> None:
                self._capture_without_batch(captured, name, output)

            return hook

        def capture_input(name: str):
            def hook(_module: object, args: object) -> None:
                if not isinstance(args, tuple) or not args:
                    raise CalibrationError(f"PR07 trace pre-hook {name!r} has no input")
                self._capture_without_batch(captured, name, args[0])

            return hook

        def capture_attention(
            _module: object, _args: object, output: object
        ) -> None:
            if not isinstance(output, tuple) or len(output) != 2:
                raise CalibrationError("PR07 eager attention hook contract changed")
            self._capture_without_batch(
                captured, "layer0.attention_probs", output[1]
            )

        def capture_context(_module: object, args: object) -> None:
            if not isinstance(args, tuple) or len(args) != 1:
                raise CalibrationError("PR07 attention context hook contract changed")
            context = args[0]
            try:
                expected = (1, len(PR07_TRACE_TOKEN_IDS), MODEL_HIDDEN_SIZE)
                if tuple(int(value) for value in context.shape) != expected:
                    raise CalibrationError(
                        "PR07 attention context has an unexpected flattened shape"
                    )
                context = context.reshape(
                    1,
                    len(PR07_TRACE_TOKEN_IDS),
                    MODEL_QUERY_HEAD_COUNT,
                    MODEL_HEAD_DIM,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise CalibrationError(
                    "PR07 attention context cannot be restored to [B,S,QH,D]"
                ) from error
            self._capture_without_batch(
                captured, "layer0.attention_context", context
            )

        handles.extend(
            (
                self._base_model.embed_tokens.register_forward_hook(
                    capture_output("embedding")
                ),
                layer0.input_layernorm.register_forward_hook(
                    capture_output("layer0.input_norm")
                ),
                attention.q_proj.register_forward_hook(
                    capture_output("layer0.q_proj")
                ),
                attention.k_proj.register_forward_hook(
                    capture_output("layer0.k_proj")
                ),
                attention.v_proj.register_forward_hook(
                    capture_output("layer0.v_proj")
                ),
                attention.register_forward_hook(capture_attention),
                attention.o_proj.register_forward_pre_hook(capture_context),
                layer0.post_attention_layernorm.register_forward_pre_hook(
                    capture_input("layer0.after_attention_residual")
                ),
                layer0.post_attention_layernorm.register_forward_hook(
                    capture_output("layer0.post_attention_norm")
                ),
                mlp.gate_proj.register_forward_hook(
                    capture_output("layer0.gate_proj")
                ),
                mlp.up_proj.register_forward_hook(capture_output("layer0.up_proj")),
                mlp.down_proj.register_forward_pre_hook(capture_input("layer0.gated")),
                mlp.down_proj.register_forward_hook(
                    capture_output("layer0.down_proj")
                ),
                layer0.register_forward_hook(capture_output("layer0.output")),
                layer14.register_forward_hook(capture_output("layer14.output")),
                self._base_model.norm.register_forward_pre_hook(
                    capture_input("final_norm.input")
                ),
                self._base_model.norm.register_forward_hook(
                    capture_output("final_norm.output")
                ),
            )
        )

        input_ids = torch.tensor(
            [PR07_TRACE_TOKEN_IDS], device=self._device, dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            len(PR07_TRACE_TOKEN_IDS), device=self._device, dtype=torch.long
        ).unsqueeze(0)
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            if getattr(output, "past_key_values", None) is not None:
                raise CalibrationError("PR07 cache-free trace unexpectedly returned a cache")
            logits = output.logits
            expected_logits_shape = (1, 1, MODEL_VOCAB_SIZE)
            if tuple(int(value) for value in logits.shape) != expected_logits_shape:
                raise CalibrationError("PR07 trace returned an unexpected logits shape")
            self._capture_cpu(captured, "last_logits", logits[0, -1])
            ordered = {name: captured[name] for name in PR07_TRACE_TENSOR_NAMES}
            _validate_trace_tensors(ordered)
            return CapturedPr07Trace(tensors=ordered)
        except KeyError as error:
            raise CalibrationError(f"PR07 trace hook did not capture {error.args[0]!r}") from error
        finally:
            for handle in reversed(handles):
                handle.remove()
            del input_ids, attention_mask, position_ids
            torch.cuda.empty_cache()

    def close(self) -> None:
        backend = self._calibration_backend
        if backend is None:
            return
        self._calibration_backend = None
        self._model = None
        self._base_model = None
        self._layers = ()
        backend.close()


def pr07_trace_repository_provenance(
    repo_root: Path, *, observed_environment: Mapping[str, object]
) -> dict[str, object]:
    """Extend the existing clean-tree provenance with PR07 producer sources."""

    provenance = repository_provenance(
        repo_root, observed_environment=observed_environment
    )
    root = repo_root.resolve()
    try:
        for relative in PR07_TRACE_SOURCE_PATHS.values():
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=root,
                check=True,
                capture_output=True,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CalibrationError(f"cannot bind PR07 trace source provenance: {error}") from error
    provenance["trace_sources"] = {
        name: {"path": relative, "sha256": sha256_file(root / relative)}
        for name, relative in PR07_TRACE_SOURCE_PATHS.items()
    }
    return provenance


def _validate_metadata(metadata: OracleArtifactMetadata) -> None:
    expected = {
        "python_version": PYTHON_VERSION,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
        "python_platform_system": PYTHON_PLATFORM_SYSTEM,
        "python_platform_machine": PYTHON_PLATFORM_MACHINE,
        "torch_version": TORCH_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "safetensors_version": SAFETENSORS_VERSION,
        "config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    for field, value in expected.items():
        if getattr(metadata, field) != value:
            raise CalibrationError(f"PR07 trace metadata.{field} differs from the pin")
    if dict(metadata.tokenizer_files_sha256) != TOKENIZER_FILES_SHA256:
        raise CalibrationError("PR07 trace tokenizer artifact hashes differ from the pins")


def _source_records_are_valid(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for record in value.values():
        if not isinstance(record, Mapping):
            return False
        path = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
    return True


def validate_pr07_trace_manifest(manifest: Mapping[str, object]) -> None:
    if set(manifest) != {
        "schema_version",
        "artifact_kind",
        "trace_id",
        "created_at",
        "producer",
        "contract",
        "provenance",
        "sidecar",
        "tensors",
    }:
        raise CalibrationError("PR07 trace manifest root fields differ from the contract")
    if manifest["schema_version"] != PR07_TRACE_SCHEMA_VERSION:
        raise CalibrationError("PR07 trace schema version differs")
    if manifest["artifact_kind"] != PR07_TRACE_ARTIFACT_KIND:
        raise CalibrationError("PR07 trace artifact kind differs")
    if manifest["trace_id"] != PR07_TRACE_ID:
        raise CalibrationError("PR07 trace ID differs")
    parse_utc(manifest["created_at"])
    producer = manifest["producer"]
    contract = manifest["contract"]
    provenance = manifest["provenance"]
    sidecar = manifest["sidecar"]
    tensors = manifest["tensors"]
    if not all(
        isinstance(value, Mapping)
        for value in (producer, contract, provenance, sidecar, tensors)
    ):
        raise CalibrationError("PR07 trace manifest contains a non-object section")
    if set(producer) != {
        "implementation_id",
        "runtime_dependency_class",
        "python_version",
        "python_executable_sha256",
        "python_platform_system",
        "python_platform_machine",
        "torch_version",
        "transformers_version",
        "safetensors_version",
        "selected_device",
        "transformers_llama_source",
    }:
        raise CalibrationError("PR07 trace producer fields differ")
    if producer.get("implementation_id") != "hf-transformers-llama-hooks-pr07-v1":
        raise CalibrationError("PR07 trace producer implementation differs")
    if producer.get("runtime_dependency_class") != RUNTIME_DEPENDENCY_CLASS:
        raise CalibrationError("PR07 trace runtime dependency class differs")
    pinned_runtime = {
        "python_version": PYTHON_VERSION,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
        "python_platform_system": PYTHON_PLATFORM_SYSTEM,
        "python_platform_machine": PYTHON_PLATFORM_MACHINE,
        "torch_version": TORCH_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "safetensors_version": SAFETENSORS_VERSION,
    }
    for name, expected in pinned_runtime.items():
        if producer.get(name) != expected:
            raise CalibrationError(f"PR07 trace producer {name} differs")
    module_source = producer.get("transformers_llama_source")
    if not isinstance(module_source, Mapping) or dict(module_source) != {
        "path": TRANSFORMERS_LLAMA_SOURCE_PATH,
        "sha256": TRANSFORMERS_LLAMA_SOURCE_SHA256,
    }:
        raise CalibrationError("PR07 trace Transformers source binding differs")
    if contract.get("model_id") != MODEL_ID or contract.get("model_revision") != MODEL_REVISION:
        raise CalibrationError("PR07 trace model identity differs")
    if contract.get("config_sha256") != MODEL_CONFIG_SHA256:
        raise CalibrationError("PR07 trace config hash differs")
    if contract.get("weights_sha256") != MODEL_WEIGHTS_SHA256:
        raise CalibrationError("PR07 trace weights hash differs")
    if set(contract) != {
        "model_id",
        "model_revision",
        "config_sha256",
        "weights_sha256",
        "input_token_ids",
        "input_token_ids_sha256",
        "execution",
    }:
        raise CalibrationError("PR07 trace contract fields differ")
    if contract.get("input_token_ids") != list(PR07_TRACE_TOKEN_IDS):
        raise CalibrationError("PR07 trace token IDs differ")
    if contract.get("input_token_ids_sha256") != PR07_TRACE_TOKEN_IDS_SHA256:
        raise CalibrationError("PR07 trace token hash differs")
    execution = contract.get("execution")
    if not isinstance(execution, Mapping) or dict(execution) != {
        "attention_backend": ATTENTION_BACKEND,
        "batch_size": 1,
        "dtype": "bfloat16",
        "explicit_position_ids": list(range(len(PR07_TRACE_TOKEN_IDS))),
        "inference_mode": True,
        "logits_to_keep": 1,
        "return_dict": True,
        "sequence_length": len(PR07_TRACE_TOKEN_IDS),
        "trust_remote_code": False,
        "use_cache": False,
    }:
        raise CalibrationError("PR07 trace execution contract differs")
    if provenance.get("environment_id") != PRIMARY_ENVIRONMENT_ID:
        raise CalibrationError("PR07 trace environment ID differs")
    try:
        validate_environment_snapshot(provenance["observed_environment"])
    except (KeyError, EnvironmentContractError) as error:
        raise CalibrationError("PR07 trace GPU provenance differs") from error
    observed_accelerator = provenance["observed_environment"]["accelerator"]
    observed_gpu = observed_accelerator["gpus"][0]
    selected_device = producer.get("selected_device")
    if not isinstance(selected_device, Mapping) or dict(selected_device) != {
        "requested": "cuda:0",
        "index": observed_gpu["index"],
        "name": observed_gpu["name"],
        "compute_capability": observed_gpu["compute_capability"],
        "driver_version": observed_accelerator["nvidia_driver_version"],
    }:
        raise CalibrationError("PR07 trace selected GPU provenance differs")
    if not _source_records_are_valid(provenance.get("sources")) or not _source_records_are_valid(
        provenance.get("trace_sources")
    ):
        raise CalibrationError("PR07 trace source hashes are missing or malformed")
    if sidecar.get("format") != "safetensors" or sidecar.get("tensor_count") != len(
        PR07_TRACE_TENSOR_NAMES
    ):
        raise CalibrationError("PR07 trace sidecar metadata differs")
    sidecar_name = sidecar.get("path")
    if (
        not isinstance(sidecar_name, str)
        or Path(sidecar_name).name != sidecar_name
        or not sidecar_name.endswith(".safetensors")
        or set(sidecar) != {"path", "sha256", "format", "tensor_count"}
    ):
        raise CalibrationError("PR07 trace sidecar path/fields differ")
    digest = sidecar.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CalibrationError("PR07 trace sidecar SHA-256 is invalid")
    if tuple(tensors) != PR07_TRACE_TENSOR_NAMES:
        raise CalibrationError("PR07 trace tensor manifest names/order differ")
    expected_shapes = _expected_trace_shapes()
    keys: set[str] = set()
    for name in PR07_TRACE_TENSOR_NAMES:
        reference = tensors[name]
        expected_key = f"trace/{name.replace('.', '/')}"
        if not isinstance(reference, Mapping) or dict(reference) != {
            "key": expected_key,
            "shape": expected_shapes[name],
            "dtype": "bfloat16",
        }:
            raise CalibrationError(f"PR07 trace tensor reference {name!r} differs")
        if expected_key in keys:
            raise CalibrationError("PR07 trace sidecar key is duplicated")
        keys.add(expected_key)


def produce_pr07_trace(
    *,
    manifest_path: Path,
    sidecar_path: Path,
    repo_root: Path,
    device: str,
    local_files_only: bool,
    created_at: datetime,
    backend_factory: TraceBackendFactory = HuggingFacePr07TraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    environment_probe: EnvironmentProbe = probe_primary_environment,
    provenance_factory: TraceProvenanceFactory = pr07_trace_repository_provenance,
) -> dict[str, object]:
    """Produce the fixed BF16 trace without replacing any existing artifact."""

    if manifest_path.exists() or sidecar_path.exists():
        raise CalibrationError("refusing to overwrite an existing PR07 trace artifact")
    repository = repo_root.resolve()
    manifest = manifest_path.resolve()
    sidecar = sidecar_path.resolve()
    for output in (manifest, sidecar):
        if output == repository or repository in output.parents:
            raise CalibrationError("PR07 trace artifacts must be outside the repository")
    if manifest.parent != sidecar.parent:
        raise CalibrationError("PR07 trace manifest and sidecar must be sibling files")
    if manifest.suffix != ".json" or sidecar.suffix != ".safetensors":
        raise CalibrationError("PR07 trace requires .json and .safetensors outputs")
    if device != "cuda:0":
        raise CalibrationError("canonical PR07 trace capture requires --device cuda:0")
    try:
        observed_environment = environment_probe()
        validate_environment_snapshot(observed_environment)
    except EnvironmentContractError as error:
        raise CalibrationError(f"primary environment preflight failed: {error}") from error
    provenance = provenance_factory(
        repo_root, observed_environment=observed_environment
    )
    backend = backend_factory(
        device=device,
        local_files_only=local_files_only,
    )
    try:
        captured = backend.capture_trace()
    finally:
        backend.close()
    _validate_metadata(backend.metadata)
    if backend.transformers_llama_source_path != TRANSFORMERS_LLAMA_SOURCE_PATH:
        raise CalibrationError("PR07 trace backend reported the wrong Llama source path")
    if backend.transformers_llama_source_sha256 != TRANSFORMERS_LLAMA_SOURCE_SHA256:
        raise CalibrationError("PR07 trace backend reported the wrong Llama source hash")
    tensors = dict(captured.tensors)
    _validate_trace_tensors(tensors)
    sidecar_tensors: dict[str, object] = {}
    tensor_references: dict[str, object] = {}
    for name in PR07_TRACE_TENSOR_NAMES:
        key = f"trace/{name.replace('.', '/')}"
        tensor = tensors[name]
        sidecar_tensors[key] = tensor
        tensor_references[name] = {
            "key": key,
            "shape": _tensor_shape(tensor),
            "dtype": _canonical_tensor_dtype(tensor),
        }
    gpu = observed_environment["accelerator"]["gpus"][0]
    metadata = backend.metadata
    document: dict[str, object] = {
        "schema_version": PR07_TRACE_SCHEMA_VERSION,
        "artifact_kind": PR07_TRACE_ARTIFACT_KIND,
        "trace_id": PR07_TRACE_ID,
        "created_at": utc_text(created_at),
        "producer": {
            "implementation_id": "hf-transformers-llama-hooks-pr07-v1",
            "runtime_dependency_class": RUNTIME_DEPENDENCY_CLASS,
            "python_version": metadata.python_version,
            "python_executable_sha256": metadata.python_executable_sha256,
            "python_platform_system": metadata.python_platform_system,
            "python_platform_machine": metadata.python_platform_machine,
            "torch_version": metadata.torch_version,
            "transformers_version": metadata.transformers_version,
            "safetensors_version": metadata.safetensors_version,
            "selected_device": {
                "requested": device,
                "index": gpu["index"],
                "name": gpu["name"],
                "compute_capability": gpu["compute_capability"],
                "driver_version": observed_environment["accelerator"][
                    "nvidia_driver_version"
                ],
            },
            "transformers_llama_source": {
                "path": backend.transformers_llama_source_path,
                "sha256": backend.transformers_llama_source_sha256,
            },
        },
        "contract": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config_sha256": metadata.config_sha256,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "input_token_ids": list(PR07_TRACE_TOKEN_IDS),
            "input_token_ids_sha256": PR07_TRACE_TOKEN_IDS_SHA256,
            "execution": {
                "attention_backend": ATTENTION_BACKEND,
                "batch_size": 1,
                "dtype": "bfloat16",
                "explicit_position_ids": list(range(len(PR07_TRACE_TOKEN_IDS))),
                "inference_mode": True,
                "logits_to_keep": 1,
                "return_dict": True,
                "sequence_length": len(PR07_TRACE_TOKEN_IDS),
                "trust_remote_code": False,
                "use_cache": False,
            },
        },
        "provenance": provenance,
        "sidecar": {
            "path": sidecar_path.name,
            "sha256": "0" * 64,
            "format": "safetensors",
            "tensor_count": len(sidecar_tensors),
        },
        "tensors": tensor_references,
    }
    _write_sidecar_exclusive(sidecar_path, sidecar_tensors, sidecar_writer)
    try:
        document["sidecar"]["sha256"] = sha256_file(sidecar_path)
        validate_pr07_trace_manifest(document)
        write_json_exclusive(manifest_path, document)
    except BaseException:
        try:
            sidecar_path.unlink()
        except OSError:
            pass
        raise
    return document


__all__ = [
    "CapturedPr07Trace",
    "HuggingFacePr07TraceBackend",
    "PR07_TRACE_ARTIFACT_KIND",
    "PR07_TRACE_ID",
    "PR07_TRACE_SCHEMA_VERSION",
    "PR07_TRACE_TENSOR_NAMES",
    "PR07_TRACE_TOKEN_IDS",
    "PR07_TRACE_TOKEN_IDS_SHA256",
    "produce_pr07_trace",
    "validate_pr07_trace_manifest",
]
