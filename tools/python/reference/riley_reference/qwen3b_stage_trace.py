"""Offline Qwen2.5-3B pre-attention trace producer.

The producer is intentionally separate from serving.  It records a compact,
source-bound Hugging Face eager reference artifact for layer zero through RoPE.
The sidecar contains exactly seven BF16 tensors and lets the Rust diagnostic
measure every element without storing a second activation dump.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import qwen3b_serving_oracle as oracle
from .hf_calibration import SidecarWriter, _default_sidecar_writer, _write_sidecar_exclusive

SCHEMA_VERSION = "1.0.0"
ARTIFACT_KIND = "qwen3b-hf-eager-layer0-pre-attention-trace"
TRACE_ID = "qwen3b-p2048-layer0-pre-attention-v1"
IMPLEMENTATION_ID = "hf-transformers-qwen2-hooks-stage0-v1"
TRANSFORMERS_QWEN2_SOURCE_PATH = "transformers/models/qwen2/modeling_qwen2.py"
TRANSFORMERS_QWEN2_SOURCE_SHA256 = (
    "34390fea7648b444d5727abef0b85acceba28f856aedec6c335d51253806bfee"
)
BF16_BYTES = 2
MAX_SAFETENSORS_HEADER_BYTES = 1_048_576
MODEL_HIDDEN_SIZE = 2_048
MODEL_LAYER_COUNT = 36
MODEL_QUERY_HEAD_COUNT = 16
MODEL_KEY_VALUE_HEAD_COUNT = 2
MODEL_HEAD_DIMENSION = 128

TRACE_TENSORS = (
    "embedding",
    "layer0.input_norm",
    "layer0.q_proj",
    "layer0.k_proj",
    "layer0.v_proj",
    "layer0.q_rope",
    "layer0.k_rope",
)

SOURCE_PATHS = {
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "stage_trace": "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    "rust_trace_driver": "crates/riley-runtime/tests/qwen3b_stage_trace_gpu.rs",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}

BackendFactory = Callable[..., "HuggingFaceQwen3BStageTraceBackend"]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BStageTraceError(RuntimeError):
    """Raised when the immutable stage-trace contract cannot be met."""


@dataclass(frozen=True)
class CapturedTrace:
    tensors: Mapping[str, object]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BStageTraceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Qwen3BStageTraceError(f"{label} must be a regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BStageTraceError(f"cannot resolve {label}: {path}") from error


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BStageTraceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Qwen3BStageTraceError(f"{label} must be a directory, not a symlink")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BStageTraceError(f"cannot resolve {label}: {path}") from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Qwen3BStageTraceError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise Qwen3BStageTraceError(
            f"{label} fields differ: missing={missing!r} extra={extra!r}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen3BStageTraceError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    text = _require_string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise Qwen3BStageTraceError(f"{label} must be lowercase SHA-256")
    return text


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    sequence = oracle.PROMPT_TOKEN_COUNT
    hidden = MODEL_HIDDEN_SIZE
    kv_width = MODEL_KEY_VALUE_HEAD_COUNT * MODEL_HEAD_DIMENSION
    return {
        "embedding": (sequence, hidden),
        "layer0.input_norm": (sequence, hidden),
        "layer0.q_proj": (sequence, hidden),
        "layer0.k_proj": (sequence, kv_width),
        "layer0.v_proj": (sequence, kv_width),
        "layer0.q_rope": (
            sequence,
            MODEL_QUERY_HEAD_COUNT,
            MODEL_HEAD_DIMENSION,
        ),
        "layer0.k_rope": (
            sequence,
            MODEL_KEY_VALUE_HEAD_COUNT,
            MODEL_HEAD_DIMENSION,
        ),
    }


def _shape_element_count(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise Qwen3BStageTraceError("tensor shape dimensions must be positive integers")
        count *= dimension
    return count


def _canonical_bf16_le_bytes(tensor: object, torch: Any) -> bytes:
    try:
        raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BStageTraceError("cannot obtain raw BF16 tensor bytes") from error
    if len(raw) % BF16_BYTES != 0:
        raise Qwen3BStageTraceError("BF16 tensor byte count must be even")
    if sys.byteorder == "little":
        return raw
    if sys.byteorder == "big":
        return b"".join(
            raw[index : index + BF16_BYTES][::-1]
            for index in range(0, len(raw), BF16_BYTES)
        )
    raise Qwen3BStageTraceError("unsupported host byte order")


def _tensor_shape(tensor: object) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise Qwen3BStageTraceError("trace tensor has no valid shape") from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise Qwen3BStageTraceError("trace tensor shape is empty or invalid")
    return shape


def _validate_tensors(tensors: Mapping[str, object], torch: Any) -> None:
    expected = _expected_shapes()
    if tuple(tensors) != TRACE_TENSORS:
        raise Qwen3BStageTraceError("trace tensor names/order differ")
    identities: set[int] = set()
    for name in TRACE_TENSORS:
        tensor = tensors[name]
        if _tensor_shape(tensor) != expected[name]:
            raise Qwen3BStageTraceError(f"trace tensor {name} shape differs")
        if getattr(tensor, "dtype", None) != torch.bfloat16:
            raise Qwen3BStageTraceError(f"trace tensor {name} must be BF16")
        if not bool(torch.isfinite(tensor).all().item()):
            raise Qwen3BStageTraceError(f"trace tensor {name} is non-finite")
        if id(tensor) in identities:
            raise Qwen3BStageTraceError("trace tensor captures must be distinct")
        identities.add(id(tensor))


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


def _validate_topology(model: object) -> tuple[object, Sequence[object], object]:
    expected_types = {
        "causal_lm": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "model": "transformers.models.qwen2.modeling_qwen2.Qwen2Model",
        "layer": "transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer",
        "attention": "transformers.models.qwen2.modeling_qwen2.Qwen2Attention",
        "mlp": "transformers.models.qwen2.modeling_qwen2.Qwen2MLP",
        "norm": "transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm",
    }
    if _qualified_type(model) != expected_types["causal_lm"]:
        raise Qwen3BStageTraceError("loaded model type differs")
    base_model = getattr(model, "model", None)
    if _qualified_type(base_model) != expected_types["model"]:
        raise Qwen3BStageTraceError("loaded base model type differs")
    config = getattr(model, "config", None)
    if config is None or config is not getattr(base_model, "config", None):
        raise Qwen3BStageTraceError("loaded model/config identity differs")
    oracle._validate_loaded_model_config(config)
    if getattr(config, "_attn_implementation", None) != "eager":
        raise Qwen3BStageTraceError("trace requires eager attention")
    layers = getattr(base_model, "layers", None)
    if layers is None or len(layers) != MODEL_LAYER_COUNT:
        raise Qwen3BStageTraceError("loaded layer count differs")
    for index, layer in enumerate(layers):
        if _qualified_type(layer) != expected_types["layer"]:
            raise Qwen3BStageTraceError(f"layer {index} type differs")
        attention = getattr(layer, "self_attn", None)
        if _qualified_type(attention) != expected_types["attention"]:
            raise Qwen3BStageTraceError(f"layer {index} attention type differs")
        if _qualified_type(getattr(layer, "mlp", None)) != expected_types["mlp"]:
            raise Qwen3BStageTraceError(f"layer {index} MLP type differs")
        if _qualified_type(getattr(layer, "input_layernorm", None)) != expected_types["norm"]:
            raise Qwen3BStageTraceError(f"layer {index} input norm type differs")
        if _qualified_type(getattr(layer, "post_attention_layernorm", None)) != expected_types["norm"]:
            raise Qwen3BStageTraceError(f"layer {index} post norm type differs")
        if getattr(attention, "layer_idx", None) != index:
            raise Qwen3BStageTraceError(f"layer {index} attention index differs")
    if _qualified_type(getattr(base_model, "norm", None)) != expected_types["norm"]:
        raise Qwen3BStageTraceError("final norm type differs")
    embeddings = getattr(base_model, "embed_tokens", None)
    lm_head = getattr(model, "lm_head", None)
    if embeddings is None or lm_head is None:
        raise Qwen3BStageTraceError("trace model lacks tied embeddings")
    embedding_weight = getattr(embeddings, "weight", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    if (
        embedding_weight is None
        or lm_head_weight is None
        or embedding_weight.data_ptr() != lm_head_weight.data_ptr()
    ):
        raise Qwen3BStageTraceError("trace model weights are not tied")
    module = inspect.getmodule(type(model))
    if module is None or getattr(module, "apply_rotary_pos_emb", None) is None:
        raise Qwen3BStageTraceError("cannot resolve Qwen rotary function")
    source = inspect.getsourcefile(type(model))
    if source is None:
        raise Qwen3BStageTraceError("cannot resolve Qwen source file")
    source_path = Path(source).resolve()
    if _sha256_file(source_path) != TRANSFORMERS_QWEN2_SOURCE_SHA256:
        raise Qwen3BStageTraceError("Transformers Qwen source hash differs")
    return base_model, layers, module


class HuggingFaceQwen3BStageTraceBackend:
    """Lazy HF adapter that captures seven layer-zero BF16 tensors."""

    def __init__(self, loader: oracle.HuggingFaceQwen3BBackend) -> None:
        self._loader: oracle.HuggingFaceQwen3BBackend | None = loader
        self._torch = loader._torch
        self._model = loader._model
        self._device = loader._device
        self._base_model, self._layers, self._module = _validate_topology(self._model)
        self.producer_metadata = dict(loader.producer_metadata)
        self.producer_metadata["implementation_id"] = IMPLEMENTATION_ID
        self.producer_metadata["transformers_qwen2_source"] = {
            "path": TRANSFORMERS_QWEN2_SOURCE_PATH,
            "sha256": TRANSFORMERS_QWEN2_SOURCE_SHA256,
        }

    @classmethod
    def load(
        cls, *, checkpoint: oracle.CheckpointManifest, device: str
    ) -> "HuggingFaceQwen3BStageTraceBackend":
        loader = oracle.HuggingFaceQwen3BBackend.load(
            checkpoint=checkpoint, device=device
        )
        try:
            return cls(loader)
        except BaseException:
            loader.close()
            raise

    def _capture_without_batch(
        self, captured: dict[str, object], name: str, tensor: object
    ) -> None:
        if name in captured:
            raise Qwen3BStageTraceError(f"trace hook {name} ran more than once")
        try:
            if int(tensor.shape[0]) != 1:
                raise Qwen3BStageTraceError(f"trace hook {name} requires batch size one")
            value = tensor[0].detach().to(device="cpu").contiguous()
        except (AttributeError, IndexError, RuntimeError, TypeError) as error:
            raise Qwen3BStageTraceError(f"trace hook {name} returned no tensor") from error
        captured[name] = value

    def capture(self, input_token_ids: Sequence[int]) -> CapturedTrace:
        model = self._model
        if model is None:
            raise Qwen3BStageTraceError("trace backend is closed")
        expected_input = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        if tuple(input_token_ids) != expected_input:
            raise Qwen3BStageTraceError("trace input IDs differ from pinned P2048")
        torch = self._torch
        layer0 = self._layers[0]
        attention = layer0.self_attn
        captured: dict[str, object] = {}
        handles: list[object] = []
        original_rope = self._module.apply_rotary_pos_emb
        rope_calls = 0

        def capture_output(name: str):
            def hook(_module: object, _args: object, output: object) -> None:
                self._capture_without_batch(captured, name, output)

            return hook

        def traced_rope(
            query: object,
            key: object,
            cosine: object,
            sine: object,
            unsqueeze_dim: int = 1,
        ) -> object:
            nonlocal rope_calls
            output = original_rope(query, key, cosine, sine, unsqueeze_dim)
            if rope_calls == 0:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise Qwen3BStageTraceError("Qwen rotary hook contract changed")
                rotated_query, rotated_key = output
                try:
                    query_token_major = rotated_query.transpose(1, 2)
                    key_token_major = rotated_key.transpose(1, 2)
                except (AttributeError, RuntimeError) as error:
                    raise Qwen3BStageTraceError("Qwen rotary outputs cannot be transposed") from error
                self._capture_without_batch(captured, "layer0.q_rope", query_token_major)
                self._capture_without_batch(captured, "layer0.k_rope", key_token_major)
            rope_calls += 1
            return output

        handles.extend(
            (
                self._base_model.embed_tokens.register_forward_hook(
                    capture_output("embedding")
                ),
                layer0.input_layernorm.register_forward_hook(
                    capture_output("layer0.input_norm")
                ),
                attention.q_proj.register_forward_hook(capture_output("layer0.q_proj")),
                attention.k_proj.register_forward_hook(capture_output("layer0.k_proj")),
                attention.v_proj.register_forward_hook(capture_output("layer0.v_proj")),
            )
        )
        self._module.apply_rotary_pos_emb = traced_rope
        input_ids = torch.tensor(
            [list(input_token_ids)], dtype=torch.long, device=self._device
        )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            oracle.PROMPT_TOKEN_COUNT, dtype=torch.long, device=self._device
        ).unsqueeze(0)
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            if getattr(output, "past_key_values", None) is not None:
                raise Qwen3BStageTraceError("cache-free trace returned a KV cache")
            if rope_calls != MODEL_LAYER_COUNT:
                raise Qwen3BStageTraceError("Qwen rotary invocation count differs")
            ordered = {name: captured[name] for name in TRACE_TENSORS}
            _validate_tensors(ordered, torch)
            return CapturedTrace(tensors=ordered)
        except KeyError as error:
            raise Qwen3BStageTraceError(
                f"trace hook did not capture {error.args[0]}"
            ) from error
        finally:
            self._module.apply_rotary_pos_emb = original_rope
            for handle in reversed(handles):
                handle.remove()
            del input_ids, attention_mask, position_ids
            torch.cuda.empty_cache()

    def close(self) -> None:
        loader = self._loader
        if loader is None:
            return
        self._loader = None
        self._model = None
        self._base_model = None
        self._layers = ()
        loader.close()


def _source_record(root: Path, relative: str) -> dict[str, object]:
    source = _regular_file(root / relative, f"source {relative}")
    return {"path": relative, "sha256": _sha256_file(source)}


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    root = _regular_directory(repo_root.expanduser(), "repository root")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *SOURCE_PATHS.values(),
            ],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Qwen3BStageTraceError("cannot collect Git provenance") from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BStageTraceError("Git revision has an unexpected format")
    return {
        "git_revision": revision,
        "source_dirty": bool(status),
        "source_status_sha256": _sha256_bytes(status),
        "sources": {
            name: _source_record(root, relative)
            for name, relative in SOURCE_PATHS.items()
        },
    }


def _workload_document(workload: oracle.ServingWorkload) -> dict[str, object]:
    return {
        "schema_version": oracle.WORKLOAD_SCHEMA_VERSION,
        "case": oracle.WORKLOAD_CASE,
        "source_sha256": workload.source_sha256,
        "prompt_token_count": len(workload.prompt_token_ids),
        "prompt_token_ids_le_u32_sha256": oracle._token_ids_sha256(
            workload.prompt_token_ids
        ),
    }


def _execution_document() -> dict[str, object]:
    return {
        "attention_implementation": "eager",
        "dtype": "bfloat16",
        "explicit_attention_mask": True,
        "explicit_position_ids": True,
        "logits_to_keep": 1,
        "tf32_enabled": False,
        "trace_capture": "module-hooks+scoped-rope-wrapper",
        "use_cache": False,
    }


def _tensor_manifest(tensors: Mapping[str, object], torch: Any) -> dict[str, object]:
    _validate_tensors(tensors, torch)
    document: dict[str, object] = {}
    for name in TRACE_TENSORS:
        tensor = tensors[name]
        raw = _canonical_bf16_le_bytes(tensor, torch)
        shape = _tensor_shape(tensor)
        if len(raw) != _shape_element_count(shape) * BF16_BYTES:
            raise Qwen3BStageTraceError(f"trace tensor {name} byte count differs")
        document[name] = {
            "key": f"trace/{name.replace('.', '/')}",
            "shape": list(shape),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_sha256": _sha256_bytes(raw),
            "bf16_le_bytes": len(raw),
        }
    return document


def build_manifest(
    *,
    workload: oracle.ServingWorkload,
    checkpoint: oracle.CheckpointManifest,
    tensors: Mapping[str, object],
    torch: Any,
    producer_metadata: Mapping[str, object],
    source_provenance: Mapping[str, object],
    sidecar_name: str,
    sidecar_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(".safetensors"):
        raise Qwen3BStageTraceError("sidecar name differs")
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "trace_id": TRACE_ID,
        "performance_claim_eligible": False,
        "created_at_unix_seconds": int(created_at.timestamp()),
        "producer": dict(producer_metadata),
        "trace_profile": {
            "capture_domain": "all-2048-token-positions",
            "id": TRACE_ID,
            "tensor_count": len(TRACE_TENSORS),
        },
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": _workload_document(workload),
            "execution": _execution_document(),
        },
        "model": {
            "checkpoint_path": str(checkpoint.root),
            "checkpoint_receipt_filename": checkpoint.receipt.path,
            "checkpoint_receipt_sha256": checkpoint.receipt.sha256,
        },
        "provenance": {"source_repository": dict(source_provenance)},
        "sidecar": {
            "path": sidecar_name,
            "sha256": sidecar_sha256,
            "format": "safetensors",
            "tensor_count": len(TRACE_TENSORS),
        },
        "tensors": _tensor_manifest(tensors, torch),
    }
    validate_manifest(document)
    return document


def _output_paths(
    manifest_path: Path, sidecar_path: Path, repo_root: Path
) -> tuple[Path, Path]:
    root = _regular_directory(repo_root.expanduser(), "repository root")
    if manifest_path.exists() or sidecar_path.exists():
        raise Qwen3BStageTraceError("refusing to overwrite an existing trace artifact")
    manifest = manifest_path.expanduser()
    sidecar = sidecar_path.expanduser()
    if not manifest.is_absolute() or not sidecar.is_absolute():
        raise Qwen3BStageTraceError("trace artifact paths must be absolute")
    if manifest.parent != sidecar.parent:
        raise Qwen3BStageTraceError("manifest and sidecar must be siblings")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(manifest.parent, "trace output directory")
    manifest = parent / manifest.name
    sidecar = parent / sidecar.name
    if (
        manifest.suffix != ".json"
        or sidecar.suffix != ".safetensors"
        or not manifest.name
        or not sidecar.name
    ):
        raise Qwen3BStageTraceError("trace artifact extensions differ")
    for output in (manifest, sidecar):
        if output == root or root in output.parents:
            raise Qwen3BStageTraceError("trace artifacts must be outside the repository")
    return manifest, sidecar


def produce_hf_trace(
    *,
    checkpoint_path: Path,
    workload_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    repo_root: Path,
    device: str,
    created_at: datetime | None = None,
    backend_factory: BackendFactory = HuggingFaceQwen3BStageTraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Create a source-bound, external HF trace manifest and sidecar."""

    manifest, sidecar = _output_paths(manifest_path, sidecar_path, repo_root)
    workload = oracle.load_workload(workload_path)
    checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    provenance = source_provenance_factory(repo_root)
    backend = backend_factory(checkpoint=checkpoint, device=device)
    try:
        captured = backend.capture(workload.prompt_token_ids)
        tensors = dict(captured.tensors)
        _validate_tensors(tensors, backend._torch)
        sidecar_tensors = {
            f"trace/{name.replace('.', '/')}": tensors[name] for name in TRACE_TENSORS
        }
        _write_sidecar_exclusive(sidecar, sidecar_tensors, sidecar_writer)
        try:
            return_document = build_manifest(
                workload=workload,
                checkpoint=checkpoint,
                tensors=tensors,
                torch=backend._torch,
                producer_metadata=backend.producer_metadata,
                source_provenance=provenance,
                sidecar_name=sidecar.name,
                sidecar_sha256=_sha256_file(sidecar),
                created_at=created_at or datetime.now(timezone.utc),
            )
            oracle.write_artifact_exclusive(manifest, return_document)
            return return_document
        except BaseException:
            try:
                sidecar.unlink()
            except OSError:
                pass
            raise
    finally:
        backend.close()


def _validate_workload_document(value: object) -> None:
    workload = _require_mapping(value, "trace workload")
    _require_exact_keys(
        workload,
        {
            "schema_version",
            "case",
            "source_sha256",
            "prompt_token_count",
            "prompt_token_ids_le_u32_sha256",
        },
        "trace workload",
    )
    if (
        workload["schema_version"] != oracle.WORKLOAD_SCHEMA_VERSION
        or workload["case"] != oracle.WORKLOAD_CASE
        or workload["source_sha256"] != oracle.WORKLOAD_SHA256
        or workload["prompt_token_count"] != oracle.PROMPT_TOKEN_COUNT
        or workload["prompt_token_ids_le_u32_sha256"] != oracle.PROMPT_TOKEN_IDS_SHA256
    ):
        raise Qwen3BStageTraceError("trace workload contract differs")


def validate_manifest(document: Mapping[str, object]) -> None:
    """Validate the light manifest without loading Torch or model weights."""

    _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact_kind",
            "trace_id",
            "performance_claim_eligible",
            "created_at_unix_seconds",
            "producer",
            "trace_profile",
            "contract",
            "model",
            "provenance",
            "sidecar",
            "tensors",
        },
        "trace manifest",
    )
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["artifact_kind"] != ARTIFACT_KIND
        or document["trace_id"] != TRACE_ID
        or document["performance_claim_eligible"] is not False
    ):
        raise Qwen3BStageTraceError("trace manifest identity differs")
    timestamp = document["created_at_unix_seconds"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise Qwen3BStageTraceError("trace manifest timestamp differs")
    producer = _require_mapping(document["producer"], "trace producer")
    if producer.get("implementation_id") != IMPLEMENTATION_ID:
        raise Qwen3BStageTraceError("trace producer implementation differs")
    for field in ("runtime_dependency_class", "torch_version", "transformers_version"):
        _require_string(producer.get(field), f"trace producer {field}")
    qwen_source = _require_mapping(
        producer.get("transformers_qwen2_source"), "trace producer Qwen source"
    )
    if dict(qwen_source) != {
        "path": TRANSFORMERS_QWEN2_SOURCE_PATH,
        "sha256": TRANSFORMERS_QWEN2_SOURCE_SHA256,
    }:
        raise Qwen3BStageTraceError("trace producer Qwen source differs")
    profile = _require_mapping(document["trace_profile"], "trace profile")
    if dict(profile) != {
        "capture_domain": "all-2048-token-positions",
        "id": TRACE_ID,
        "tensor_count": len(TRACE_TENSORS),
    }:
        raise Qwen3BStageTraceError("trace profile differs")
    contract = _require_mapping(document["contract"], "trace contract")
    _require_exact_keys(
        contract, {"model_id", "model_revision", "workload", "execution"}, "trace contract"
    )
    if contract["model_id"] != oracle.MODEL_ID or contract["model_revision"] != oracle.MODEL_REVISION:
        raise Qwen3BStageTraceError("trace model identity differs")
    _validate_workload_document(contract["workload"])
    if _require_mapping(contract["execution"], "trace execution") != _execution_document():
        raise Qwen3BStageTraceError("trace execution contract differs")
    model = _require_mapping(document["model"], "trace model")
    _require_exact_keys(
        model,
        {
            "checkpoint_path",
            "checkpoint_receipt_filename",
            "checkpoint_receipt_sha256",
        },
        "trace model",
    )
    _require_string(model["checkpoint_path"], "trace checkpoint path")
    if model["checkpoint_receipt_filename"] != oracle.CHECKPOINT_RECEIPT_FILENAME:
        raise Qwen3BStageTraceError("trace checkpoint receipt filename differs")
    _require_sha256(model["checkpoint_receipt_sha256"], "trace checkpoint receipt SHA-256")
    provenance = _require_mapping(document["provenance"], "trace provenance")
    source = _require_mapping(provenance.get("source_repository"), "trace source provenance")
    _require_exact_keys(
        source,
        {"git_revision", "source_dirty", "source_status_sha256", "sources"},
        "trace source provenance",
    )
    revision = _require_string(source["git_revision"], "trace Git revision")
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BStageTraceError("trace Git revision is malformed")
    if not isinstance(source["source_dirty"], bool):
        raise Qwen3BStageTraceError("trace source dirty must be a boolean")
    _require_sha256(source["source_status_sha256"], "trace source status SHA-256")
    sources = _require_mapping(source["sources"], "trace sources")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BStageTraceError("trace source records differ")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], f"trace source {name}")
        if dict(record).get("path") != relative:
            raise Qwen3BStageTraceError("trace source path differs")
        _require_sha256(record.get("sha256"), "trace source SHA-256")
    sidecar = _require_mapping(document["sidecar"], "trace sidecar")
    _require_exact_keys(sidecar, {"path", "sha256", "format", "tensor_count"}, "trace sidecar")
    name = _require_string(sidecar["path"], "trace sidecar path")
    if Path(name).name != name or not name.endswith(".safetensors"):
        raise Qwen3BStageTraceError("trace sidecar path differs")
    if sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(TRACE_TENSORS):
        raise Qwen3BStageTraceError("trace sidecar metadata differs")
    _require_sha256(sidecar["sha256"], "trace sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "trace tensors")
    if tuple(tensors) != TRACE_TENSORS:
        raise Qwen3BStageTraceError("trace tensor names/order differ")
    shapes = _expected_shapes()
    for name in TRACE_TENSORS:
        tensor = _require_mapping(tensors[name], f"trace tensor {name}")
        expected_key = f"trace/{name.replace('.', '/')}"
        expected_bytes = _shape_element_count(shapes[name]) * BF16_BYTES
        if dict(tensor).get("key") != expected_key:
            raise Qwen3BStageTraceError(f"trace tensor {name} key differs")
        if tuple(tensor.get("shape", ())) != shapes[name]:
            raise Qwen3BStageTraceError(f"trace tensor {name} shape differs")
        if (
            tensor.get("dtype") != "bfloat16"
            or tensor.get("canonical_byte_order") != "little-endian-u16"
            or tensor.get("bf16_le_bytes") != expected_bytes
        ):
            raise Qwen3BStageTraceError(f"trace tensor {name} metadata differs")
        _require_sha256(tensor.get("bf16_le_sha256"), f"trace tensor {name} SHA-256")


def _read_safetensors_header(path: Path) -> tuple[Mapping[str, object], int, int]:
    source = _regular_file(path, "trace sidecar")
    try:
        size = source.stat().st_size
        with source.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise Qwen3BStageTraceError("trace sidecar lacks an 8-byte header length")
            header_bytes = int.from_bytes(prefix, "little")
            if not 0 < header_bytes <= MAX_SAFETENSORS_HEADER_BYTES:
                raise Qwen3BStageTraceError("trace sidecar header size differs")
            raw_header = handle.read(header_bytes)
    except OSError as error:
        raise Qwen3BStageTraceError("cannot read trace sidecar") from error
    if len(raw_header) != header_bytes:
        raise Qwen3BStageTraceError("trace sidecar header is truncated")
    try:
        header = _require_mapping(json.loads(raw_header), "trace sidecar header")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BStageTraceError("trace sidecar header is invalid JSON") from error
    return header, 8 + header_bytes, size


def validate_sidecar_against_manifest(manifest: Mapping[str, object], sidecar_path: Path) -> None:
    """Replay tensor names, ranges and raw BF16 hashes without Torch."""

    validate_manifest(manifest)
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    metadata = _require_mapping(manifest["sidecar"], "trace sidecar")
    if sidecar.name != metadata["path"] or _sha256_file(sidecar) != metadata["sha256"]:
        raise Qwen3BStageTraceError("trace sidecar binding differs")
    header, data_start, size = _read_safetensors_header(sidecar)
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    if set(header) - {"__metadata__"} != {
        f"trace/{name.replace('.', '/')}" for name in TRACE_TENSORS
    }:
        raise Qwen3BStageTraceError("trace sidecar tensor set differs")
    with sidecar.open("rb") as handle:
        for name in TRACE_TENSORS:
            reference = _require_mapping(tensors[name], f"trace tensor {name}")
            entry = _require_mapping(header[reference["key"]], f"sidecar tensor {name}")
            if entry.get("dtype") != "BF16" or tuple(entry.get("shape", ())) != tuple(reference["shape"]):
                raise Qwen3BStageTraceError(f"sidecar tensor {name} metadata differs")
            offsets = entry.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise Qwen3BStageTraceError(f"sidecar tensor {name} offsets differ")
            start, end = offsets
            expected_bytes = reference["bf16_le_bytes"]
            if end - start != expected_bytes or data_start + end > size:
                raise Qwen3BStageTraceError(f"sidecar tensor {name} byte range differs")
            handle.seek(data_start + start)
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise Qwen3BStageTraceError(f"sidecar tensor {name} is truncated")
            if _sha256_bytes(raw) != reference["bf16_le_sha256"]:
                raise Qwen3BStageTraceError(f"sidecar tensor {name} raw BF16 hash differs")


def _load_manifest(path: Path) -> dict[str, object]:
    source = _regular_file(path.expanduser(), "trace manifest")
    try:
        document = dict(_require_mapping(json.loads(source.read_bytes()), "trace manifest"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BStageTraceError("trace manifest is invalid JSON") from error
    validate_manifest(document)
    return document


def validate_bindings(
    *, manifest_path: Path, sidecar_path: Path, workload_path: Path, repo_root: Path
) -> dict[str, object]:
    manifest = _load_manifest(manifest_path)
    root = _regular_directory(repo_root.expanduser(), "repository root")
    workload = oracle.load_workload(workload_path)
    contract = _require_mapping(manifest["contract"], "trace contract")
    if contract["workload"] != _workload_document(workload):
        raise Qwen3BStageTraceError("trace workload binding differs")
    observed_sources = _require_mapping(
        _require_mapping(manifest["provenance"], "trace provenance")["source_repository"],
        "trace source provenance",
    )
    expected_sources = collect_source_provenance(root)
    if observed_sources["sources"] != expected_sources["sources"]:
        raise Qwen3BStageTraceError("trace source hashes differ")
    validate_sidecar_against_manifest(manifest, sidecar_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_stage_trace",
        description="offline Qwen2.5-3B P2048 HF eager pre-attention trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser("produce", help="write a new manifest and safetensors sidecar")
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--workload", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--sidecar", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    validate = commands.add_parser("validate", help="validate trace without CUDA or Torch")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--sidecar", type=Path, required=True)
    validate.add_argument("--workload", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "produce":
            document = produce_hf_trace(
                checkpoint_path=args.checkpoint,
                workload_path=args.workload,
                manifest_path=args.manifest,
                sidecar_path=args.sidecar,
                repo_root=args.repo_root,
                device=args.device,
            )
            print(
                f"wrote {document['artifact_kind']} to {args.manifest}; "
                f"manifest_sha256={_sha256_file(args.manifest)}"
            )
            return 0
        document = validate_bindings(
            manifest_path=args.manifest,
            sidecar_path=args.sidecar,
            workload_path=args.workload,
            repo_root=args.repo_root,
        )
        print(
            f"valid {document['artifact_kind']}: {args.manifest}; "
            f"sidecar_sha256={document['sidecar']['sha256']}"
        )
        return 0
    except (OSError, RuntimeError, Qwen3BStageTraceError, oracle.Qwen3BServingOracleError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
