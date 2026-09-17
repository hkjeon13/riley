"""Offline source-bound P2051 eager-attention trace for one Qwen2.5-3B layer.

This module is an offline Hugging Face oracle only. It records the selected
layer's actual eager Q/K/V inputs, final-query QK/scale-mask/probability rows,
and full pre-output context. Riley serving never imports Python or this module.

The saved eager result is produced by the installed Transformers function. A
same-expression replay is checked byte-for-byte before the intermediate values
are published, so this artifact detects a Transformers eager-contract change
instead of silently treating a reimplementation as an oracle.
"""

from __future__ import annotations

import argparse
import hashlib
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

from . import qwen3b_p2051_layer_stage_trace as base
from . import qwen3b_serving_oracle as oracle
from .hf_calibration import SidecarWriter, _default_sidecar_writer, _write_sidecar_exclusive

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2051-cache-off-attention-window-trace.v1"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-attention-window-trace"
TRACE_ID = "qwen3b-p2051-cache-off-eager-attention-window-v1"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-p2051-attention-window-trace-v1"

TEACHER_PREFIX_TOKEN_COUNT = base.TEACHER_PREFIX_TOKEN_COUNT
CONTEXT_TOKEN_COUNT = base.CONTEXT_TOKEN_COUNT
LAST_TOKEN_ROW_INDEX = base.LAST_TOKEN_ROW_INDEX
BF16_BYTES = base.BF16_BYTES
MAX_SAFETENSORS_HEADER_BYTES = base.MAX_SAFETENSORS_HEADER_BYTES
MODEL_LAYER_COUNT = base.MODEL_LAYER_COUNT
MODEL_QUERY_HEAD_COUNT = base.MODEL_QUERY_HEAD_COUNT
MODEL_KEY_VALUE_HEAD_COUNT = base.MODEL_KEY_VALUE_HEAD_COUNT
MODEL_HEAD_DIMENSION = base.MODEL_HEAD_DIMENSION

# Ordered names are the artifact ABI. The Q/K/V and context tensors retain
# every token because later layers attend to the whole prior sequence; only
# score-like checkpoints retain the final causal query row.
ATTENTION_SUFFIXES = (
    "attention.query_bshd",
    "attention.key_bshd",
    "attention.value_bshd",
    "attention.raw_qk_last",
    "attention.scaled_masked_last",
    "attention.probabilities_last",
    "attention.context_bshd",
    "attention.mask_last",
)

SOURCE_PATHS = {
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "teacher_forced_generation": "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    "stage_trace_support": "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    "p2051_layer_stage_contract": "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
    "p2051_layer_window_trace": "tools/python/reference/riley_reference/qwen3b_p2051_layer_window_trace.py",
    "p2051_attention_window_trace": "tools/python/reference/riley_reference/qwen3b_p2051_attention_window_trace.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "native_attention_trace": "kernels/src/attention_cublaslt.cu",
    "native_attention_trace_header": "kernels/include/riley_cuda.h",
    "rust_attention_trace_ffi": "crates/riley-cuda/src/ffi.rs",
    "rust_attention_trace_api": "crates/riley-cuda/src/prefill.rs",
    "rust_attention_trace_exports": "crates/riley-cuda/src/lib.rs",
    "rust_attention_trace_consumer": "crates/riley-cuda/tests/qwen3b_p2051_attention_window_gpu.rs",
    "rust_attention_trace_package": "crates/riley-cuda/Cargo.toml",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}

BackendFactory = Callable[..., "HuggingFaceQwen3BP2051AttentionWindowTraceBackend"]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BP2051AttentionWindowTraceError(RuntimeError):
    """Raised when the selected-layer eager-attention contract is violated."""


@dataclass(frozen=True)
class CapturedTrace:
    layer_index: int
    tensors: Mapping[str, object]


TeacherPrefix = base.TeacherPrefix


def _error_from_base(error: BaseException) -> Qwen3BP2051AttentionWindowTraceError:
    return Qwen3BP2051AttentionWindowTraceError(str(error))


def _validate_layer_index(value: object, label: str = "layer index") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Qwen3BP2051AttentionWindowTraceError(f"{label} must be an integer")
    if not 0 <= value < MODEL_LAYER_COUNT:
        raise Qwen3BP2051AttentionWindowTraceError(
            f"{label} must be in 0..{MODEL_LAYER_COUNT - 1}"
        )
    return value


def trace_tensor_names(layer_index: int) -> tuple[str, ...]:
    index = _validate_layer_index(layer_index)
    return tuple(f"layer{index}.{suffix}" for suffix in ATTENTION_SUFFIXES)


def _sidecar_key(name: str) -> str:
    return f"trace/{name.replace('.', '/')}"


def _expected_shapes(layer_index: int) -> dict[str, tuple[int, ...]]:
    names = trace_tensor_names(layer_index)
    sequence = CONTEXT_TOKEN_COUNT
    query = (1, sequence, MODEL_QUERY_HEAD_COUNT, MODEL_HEAD_DIMENSION)
    key_value = (1, sequence, MODEL_KEY_VALUE_HEAD_COUNT, MODEL_HEAD_DIMENSION)
    final_rows = (MODEL_QUERY_HEAD_COUNT, sequence)
    shapes = {
        names[0]: query,
        names[1]: key_value,
        names[2]: key_value,
        names[3]: final_rows,
        names[4]: final_rows,
        names[5]: final_rows,
        names[6]: query,
        names[7]: (sequence,),
    }
    if tuple(shapes) != names:
        raise AssertionError("attention-window shape ordering differs from trace ABI")
    return shapes


def _rust_consumer_document(layer_index: int) -> dict[str, object]:
    index = _validate_layer_index(layer_index)
    return {
        "api": "riley_cuda::PreparedPrefillAttention::execute_hf_eager_qwen_p2051_last_row_traced",
        "attention_backend": "riley.cuda.hf-eager-cublaslt-qwen-p2051-probe.bf16",
        "cache": False,
        "input_context_token_count": CONTEXT_TOKEN_COUNT,
        "last_token_row_index": LAST_TOKEN_ROW_INDEX,
        "layer_index": index,
        "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
        "trace_row_layout": "full-qkv-context-plus-last-score-row",
    }


def _capture_profile_document(layer_index: int) -> dict[str, object]:
    index = _validate_layer_index(layer_index)
    names = trace_tensor_names(index)
    return {
        "capture_domain": "cache-free-p2051-selected-layer-hf-eager-attention-boundaries",
        "id": TRACE_ID,
        "layer_index": index,
        "tensor_names": list(names),
        "tensor_count": len(names),
        "rust_consumer": _rust_consumer_document(index),
    }


def _canonical_bf16_le_bytes(tensor: object, torch: Any) -> bytes:
    try:
        raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051AttentionWindowTraceError(
            "cannot obtain raw BF16 tensor bytes"
        ) from error
    if len(raw) % BF16_BYTES != 0:
        raise Qwen3BP2051AttentionWindowTraceError("BF16 tensor byte count must be even")
    if sys.byteorder == "little":
        return raw
    if sys.byteorder == "big":
        return b"".join(
            raw[offset : offset + BF16_BYTES][::-1]
            for offset in range(0, len(raw), BF16_BYTES)
        )
    raise Qwen3BP2051AttentionWindowTraceError("unsupported host byte order")


def _tensor_shape(tensor: object) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise Qwen3BP2051AttentionWindowTraceError(
            "trace tensor has no valid shape"
        ) from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise Qwen3BP2051AttentionWindowTraceError("trace tensor shape is empty or invalid")
    return shape


def _validate_tensors(
    tensors: Mapping[str, object], torch: Any, layer_index: int
) -> None:
    names = trace_tensor_names(layer_index)
    shapes = _expected_shapes(layer_index)
    if set(tensors) != set(names):
        raise Qwen3BP2051AttentionWindowTraceError("trace tensor names differ")
    identities: set[int] = set()
    for name in names:
        tensor = tensors[name]
        if _tensor_shape(tensor) != shapes[name]:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace tensor {name} shape differs"
            )
        if getattr(tensor, "dtype", None) != torch.bfloat16:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace tensor {name} must be BF16"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace tensor {name} is non-finite"
            )
        if id(tensor) in identities:
            raise Qwen3BP2051AttentionWindowTraceError(
                "trace tensor captures must be distinct"
            )
        identities.add(id(tensor))


class HuggingFaceQwen3BP2051AttentionWindowTraceBackend:
    """Lazy HF adapter for one selected layer's eager-attention boundaries."""

    def __init__(
        self, loader: oracle.HuggingFaceQwen3BBackend, *, layer_index: int
    ) -> None:
        self._loader: oracle.HuggingFaceQwen3BBackend | None = loader
        self._torch = loader._torch
        self._model = loader._model
        self._device = loader._device
        self._base_model, self._layers, self._module = base.stage._validate_topology(
            self._model
        )
        self._layer_index = _validate_layer_index(layer_index)
        self.producer_metadata = dict(loader.producer_metadata)
        self.producer_metadata["implementation_id"] = IMPLEMENTATION_ID
        self.producer_metadata["transformers_qwen2_source"] = {
            "path": base.stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
            "sha256": base.stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
        }

    @classmethod
    def load(
        cls, *, checkpoint: oracle.CheckpointManifest, device: str, layer_index: int
    ) -> "HuggingFaceQwen3BP2051AttentionWindowTraceBackend":
        loader = oracle.HuggingFaceQwen3BBackend.load(
            checkpoint=checkpoint, device=device
        )
        try:
            return cls(loader, layer_index=layer_index)
        except BaseException:
            loader.close()
            raise

    @staticmethod
    def _capture(
        captured: dict[str, object], name: str, tensor: object
    ) -> None:
        if name in captured:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace checkpoint {name} ran more than once"
            )
        try:
            value = tensor.detach().to(device="cpu").contiguous()
        except (AttributeError, RuntimeError, TypeError) as error:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace checkpoint {name} returned no tensor"
            ) from error
        captured[name] = value

    @staticmethod
    def _expect_equal(name: str, actual: object, replay: object, torch: Any) -> None:
        try:
            equal = bool(torch.equal(actual, replay))
        except (AttributeError, RuntimeError, TypeError) as error:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace replay {name} cannot be compared"
            ) from error
        if not equal:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"actual HF eager {name} differs from its source-expression replay"
            )

    def capture(self, input_token_ids: Sequence[int]) -> CapturedTrace:
        model = self._model
        if model is None:
            raise Qwen3BP2051AttentionWindowTraceError("trace backend is closed")
        if len(input_token_ids) != CONTEXT_TOKEN_COUNT:
            raise Qwen3BP2051AttentionWindowTraceError("trace input token count differs")
        prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        if tuple(input_token_ids[: oracle.PROMPT_TOKEN_COUNT]) != prompt:
            raise Qwen3BP2051AttentionWindowTraceError(
                "trace input prompt differs from pinned P2048"
            )
        for ordinal, token_id in enumerate(input_token_ids[oracle.PROMPT_TOKEN_COUNT :]):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
            ):
                raise Qwen3BP2051AttentionWindowTraceError(
                    f"trace teacher token {ordinal} is invalid"
                )

        torch = self._torch
        index = self._layer_index
        names = trace_tensor_names(index)
        layer = self._layers[index]
        attention = layer.self_attn
        original_eager = getattr(self._module, "eager_attention_forward", None)
        repeat_kv = getattr(self._module, "repeat_kv", None)
        if original_eager is None or repeat_kv is None:
            raise Qwen3BP2051AttentionWindowTraceError(
                "Qwen eager attention source symbols differ"
            )
        captured: dict[str, object] = {}
        eager_calls = 0

        def traced_eager_attention(
            module: object,
            query: object,
            key: object,
            value: object,
            attention_mask: object,
            scaling: float,
            dropout: float = 0.0,
            **kwargs: object,
        ) -> object:
            nonlocal eager_calls
            result = original_eager(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )
            if module is not attention:
                return result
            eager_calls += 1
            if eager_calls != 1:
                raise Qwen3BP2051AttentionWindowTraceError(
                    "selected eager attention ran more than once"
                )
            if dropout != 0.0 or bool(getattr(module, "training", True)):
                raise Qwen3BP2051AttentionWindowTraceError(
                    "attention trace requires inference-mode zero dropout"
                )
            if not isinstance(result, tuple) or len(result) != 2:
                raise Qwen3BP2051AttentionWindowTraceError(
                    "Qwen eager attention result contract changed"
                )
            actual_context, actual_probabilities = result
            if attention_mask is None:
                raise Qwen3BP2051AttentionWindowTraceError(
                    "Qwen eager attention omitted its causal mask"
                )
            try:
                query_bshd = query.transpose(1, 2)
                key_bshd = key.transpose(1, 2)
                value_bshd = value.transpose(1, 2)
                key_states = repeat_kv(key, module.num_key_value_groups)
                value_states = repeat_kv(value, module.num_key_value_groups)
                raw_qk = torch.matmul(query, key_states.transpose(2, 3))
                scaled = raw_qk * scaling
                scaled_masked = scaled + attention_mask
                replay_probabilities = torch.nn.functional.softmax(
                    scaled_masked, dim=-1, dtype=torch.float32
                ).to(query.dtype)
                replay_context = torch.matmul(
                    replay_probabilities, value_states
                ).transpose(1, 2).contiguous()
                raw_qk_last = raw_qk[0, :, -1, :]
                scaled_masked_last = scaled_masked[0, :, -1, :]
                probabilities_last = actual_probabilities[0, :, -1, :]
                mask_last = attention_mask[0, 0, -1, :]
            except (AttributeError, IndexError, RuntimeError, TypeError) as error:
                raise Qwen3BP2051AttentionWindowTraceError(
                    "Qwen eager attention intermediate contract changed"
                ) from error
            self._expect_equal("probabilities", actual_probabilities, replay_probabilities, torch)
            self._expect_equal("context", actual_context, replay_context, torch)
            for name, tensor in (
                (names[0], query_bshd),
                (names[1], key_bshd),
                (names[2], value_bshd),
                (names[3], raw_qk_last),
                (names[4], scaled_masked_last),
                (names[5], probabilities_last),
                (names[6], actual_context),
                (names[7], mask_last),
            ):
                self._capture(captured, name, tensor)
            return result

        self._module.eager_attention_forward = traced_eager_attention
        input_ids = torch.tensor(
            [list(input_token_ids)], dtype=torch.long, device=self._device
        )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            CONTEXT_TOKEN_COUNT, dtype=torch.long, device=self._device
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
                raise Qwen3BP2051AttentionWindowTraceError(
                    "cache-free trace returned a KV cache"
                )
            if eager_calls != 1:
                raise Qwen3BP2051AttentionWindowTraceError(
                    "selected eager attention did not run exactly once"
                )
            ordered = {name: captured[name] for name in names}
            _validate_tensors(ordered, torch, index)
            return CapturedTrace(layer_index=index, tensors=ordered)
        except KeyError as error:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace checkpoint did not capture {error.args[0]}"
            ) from error
        finally:
            self._module.eager_attention_forward = original_eager
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return base._sha256_file(path)


def _regular_file(path: Path, label: str) -> Path:
    try:
        return base._regular_file(path, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def _regular_directory(path: Path, label: str) -> Path:
    try:
        return base._regular_directory(path, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    try:
        return base._require_mapping(value, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    try:
        base._require_exact_keys(value, expected, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def _require_string(value: object, label: str) -> str:
    try:
        return base._require_string(value, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def _require_sha256(value: object, label: str) -> str:
    try:
        return base._require_sha256(value, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def _shape_element_count(shape: Sequence[int]) -> int:
    try:
        return base._shape_element_count(shape)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def _shape_from_document(value: object, label: str) -> tuple[int, ...]:
    try:
        return base._shape_from_document(value, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


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
        raise Qwen3BP2051AttentionWindowTraceError(
            "cannot collect Git provenance"
        ) from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BP2051AttentionWindowTraceError("Git revision has an unexpected format")
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
    return base._workload_document(workload)


def _execution_document() -> dict[str, object]:
    return base._execution_document()


def load_teacher_prefix(
    *, teacher_manifest_path: Path, teacher_cache_off_sidecar_path: Path
) -> TeacherPrefix:
    try:
        return base.load_teacher_prefix(
            teacher_manifest_path=teacher_manifest_path,
            teacher_cache_off_sidecar_path=teacher_cache_off_sidecar_path,
        )
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def build_input_token_ids(
    workload: oracle.ServingWorkload, teacher_prefix: TeacherPrefix
) -> tuple[int, ...]:
    try:
        return base.build_input_token_ids(workload, teacher_prefix)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def _teacher_input_document(
    workload: oracle.ServingWorkload, teacher_prefix: TeacherPrefix
) -> dict[str, object]:
    return base._teacher_input_document(workload, teacher_prefix)


def _tensor_manifest(
    tensors: Mapping[str, object], torch: Any, layer_index: int
) -> dict[str, object]:
    _validate_tensors(tensors, torch, layer_index)
    document: dict[str, object] = {}
    for name in trace_tensor_names(layer_index):
        tensor = tensors[name]
        raw = _canonical_bf16_le_bytes(tensor, torch)
        shape = _tensor_shape(tensor)
        if len(raw) != _shape_element_count(shape) * BF16_BYTES:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace tensor {name} byte count differs"
            )
        document[name] = {
            "key": _sidecar_key(name),
            "shape": list(shape),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_sha256": _sha256_bytes(raw),
            "bf16_le_bytes": len(raw),
        }
    return document


def build_manifest(
    *,
    layer_index: int,
    workload: oracle.ServingWorkload,
    checkpoint: oracle.CheckpointManifest,
    teacher_prefix: TeacherPrefix,
    tensors: Mapping[str, object],
    torch: Any,
    producer_metadata: Mapping[str, object],
    source_provenance: Mapping[str, object],
    sidecar_name: str,
    sidecar_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    index = _validate_layer_index(layer_index)
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(
        ".safetensors"
    ):
        raise Qwen3BP2051AttentionWindowTraceError("sidecar name differs")
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "trace_id": TRACE_ID,
        "performance_claim_eligible": False,
        "created_at": oracle._utc_text(created_at),
        "producer": dict(producer_metadata),
        "trace_profile": _capture_profile_document(index),
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": _workload_document(workload),
            "execution": _execution_document(),
            "input": _teacher_input_document(workload, teacher_prefix),
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
            "tensor_count": len(trace_tensor_names(index)),
        },
        "tensors": _tensor_manifest(tensors, torch, index),
    }
    validate_manifest(document)
    return document


def _output_paths(
    manifest_path: Path, sidecar_path: Path, repo_root: Path
) -> tuple[Path, Path]:
    try:
        return base._output_paths(manifest_path, sidecar_path, repo_root)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error


def produce_hf_trace(
    *,
    layer_index: int,
    checkpoint_path: Path,
    workload_path: Path,
    teacher_manifest_path: Path,
    teacher_cache_off_sidecar_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    repo_root: Path,
    device: str,
    created_at: datetime | None = None,
    backend_factory: BackendFactory = HuggingFaceQwen3BP2051AttentionWindowTraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Create a source-bound, cache-free selected-layer trace outside the repo."""

    index = _validate_layer_index(layer_index)
    manifest, sidecar = _output_paths(manifest_path, sidecar_path, repo_root)
    try:
        workload = oracle.load_workload(workload_path)
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051AttentionWindowTraceError(str(error)) from error
    teacher_prefix = load_teacher_prefix(
        teacher_manifest_path=teacher_manifest_path,
        teacher_cache_off_sidecar_path=teacher_cache_off_sidecar_path,
    )
    input_token_ids = build_input_token_ids(workload, teacher_prefix)
    provenance = source_provenance_factory(repo_root)
    backend = backend_factory(checkpoint=checkpoint, device=device, layer_index=index)
    sidecar_written = False
    try:
        captured = backend.capture(input_token_ids)
        if captured.layer_index != index:
            raise Qwen3BP2051AttentionWindowTraceError(
                "trace backend captured a different layer"
            )
        tensors = dict(captured.tensors)
        _validate_tensors(tensors, backend._torch, index)
        _write_sidecar_exclusive(
            sidecar,
            {_sidecar_key(name): tensors[name] for name in trace_tensor_names(index)},
            sidecar_writer,
        )
        sidecar_written = True
        document = build_manifest(
            layer_index=index,
            workload=workload,
            checkpoint=checkpoint,
            teacher_prefix=teacher_prefix,
            tensors=tensors,
            torch=backend._torch,
            producer_metadata=backend.producer_metadata,
            source_provenance=provenance,
            sidecar_name=sidecar.name,
            sidecar_sha256=_sha256_file(sidecar),
            created_at=created_at or datetime.now(timezone.utc),
        )
        validate_sidecar_against_manifest(document, sidecar)
        try:
            oracle.write_artifact_exclusive(manifest, document)
        except oracle.Qwen3BServingOracleError as error:
            raise Qwen3BP2051AttentionWindowTraceError(str(error)) from error
        return document
    except BaseException:
        if sidecar_written:
            try:
                sidecar.unlink()
            except OSError:
                pass
        raise
    finally:
        backend.close()


def _validate_producer(value: object) -> None:
    producer = _require_mapping(value, "trace producer")
    if producer.get("implementation_id") != IMPLEMENTATION_ID:
        raise Qwen3BP2051AttentionWindowTraceError("trace producer implementation differs")
    for field in ("runtime_dependency_class", "torch_version", "transformers_version"):
        _require_string(producer.get(field), f"trace producer {field}")
    qwen_source = _require_mapping(
        producer.get("transformers_qwen2_source"), "trace producer Qwen source"
    )
    if dict(qwen_source) != {
        "path": base.stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
        "sha256": base.stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
    }:
        raise Qwen3BP2051AttentionWindowTraceError("trace producer Qwen source differs")


def _validate_source_provenance(value: object) -> None:
    source = _require_mapping(value, "trace source provenance")
    _require_exact_keys(
        source,
        {"git_revision", "source_dirty", "source_status_sha256", "sources"},
        "trace source provenance",
    )
    revision = _require_string(source["git_revision"], "trace Git revision")
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BP2051AttentionWindowTraceError("trace Git revision is malformed")
    if not isinstance(source["source_dirty"], bool):
        raise Qwen3BP2051AttentionWindowTraceError("trace source dirty must be a boolean")
    _require_sha256(source["source_status_sha256"], "trace source status SHA-256")
    sources = _require_mapping(source["sources"], "trace source records")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BP2051AttentionWindowTraceError("trace source records differ")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], f"trace source {name}")
        _require_exact_keys(record, {"path", "sha256"}, f"trace source {name}")
        if record["path"] != relative:
            raise Qwen3BP2051AttentionWindowTraceError("trace source path differs")
        _require_sha256(record["sha256"], f"trace source {name} SHA-256")


def _layer_index_from_profile(value: object) -> int:
    profile = _require_mapping(value, "trace profile")
    _require_exact_keys(
        profile,
        {
            "capture_domain",
            "id",
            "layer_index",
            "tensor_names",
            "tensor_count",
            "rust_consumer",
        },
        "trace profile",
    )
    index = _validate_layer_index(profile["layer_index"], "trace profile layer index")
    if dict(profile) != _capture_profile_document(index):
        raise Qwen3BP2051AttentionWindowTraceError("trace profile differs")
    return index


def validate_manifest(document: Mapping[str, object]) -> int:
    """Validate a fixed selected-layer contract without Torch or a model."""

    _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact_kind",
            "trace_id",
            "performance_claim_eligible",
            "created_at",
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
        raise Qwen3BP2051AttentionWindowTraceError("trace manifest identity differs")
    try:
        oracle._validate_utc_text(document["created_at"], "trace created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051AttentionWindowTraceError(str(error)) from error
    index = _layer_index_from_profile(document["trace_profile"])
    _validate_producer(document["producer"])
    contract = _require_mapping(document["contract"], "trace contract")
    _require_exact_keys(
        contract,
        {"model_id", "model_revision", "workload", "execution", "input"},
        "trace contract",
    )
    if (
        contract["model_id"] != oracle.MODEL_ID
        or contract["model_revision"] != oracle.MODEL_REVISION
    ):
        raise Qwen3BP2051AttentionWindowTraceError("trace model identity differs")
    try:
        base._validate_workload_document(contract["workload"])
        if _require_mapping(contract["execution"], "trace execution") != _execution_document():
            raise Qwen3BP2051AttentionWindowTraceError("trace execution contract differs")
        base._validate_teacher_input(contract["input"])
        base._validate_model(document["model"])
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise _error_from_base(error) from error
    provenance = _require_mapping(document["provenance"], "trace provenance")
    _require_exact_keys(provenance, {"source_repository"}, "trace provenance")
    _validate_source_provenance(provenance["source_repository"])
    sidecar = _require_mapping(document["sidecar"], "trace sidecar")
    _require_exact_keys(
        sidecar, {"path", "sha256", "format", "tensor_count"}, "trace sidecar"
    )
    name = _require_string(sidecar["path"], "trace sidecar path")
    names = trace_tensor_names(index)
    if Path(name).name != name or not name.endswith(".safetensors"):
        raise Qwen3BP2051AttentionWindowTraceError("trace sidecar path differs")
    if sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(names):
        raise Qwen3BP2051AttentionWindowTraceError("trace sidecar metadata differs")
    _require_sha256(sidecar["sha256"], "trace sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "trace tensors")
    if set(tensors) != set(names):
        raise Qwen3BP2051AttentionWindowTraceError("trace tensor names differ")
    shapes = _expected_shapes(index)
    for name in names:
        tensor = _require_mapping(tensors[name], f"trace tensor {name}")
        _require_exact_keys(
            tensor,
            {
                "key",
                "shape",
                "dtype",
                "canonical_byte_order",
                "bf16_le_sha256",
                "bf16_le_bytes",
            },
            f"trace tensor {name}",
        )
        expected_bytes = _shape_element_count(shapes[name]) * BF16_BYTES
        if (
            tensor["key"] != _sidecar_key(name)
            or _shape_from_document(tensor["shape"], f"trace tensor {name} shape")
            != shapes[name]
            or tensor["dtype"] != "bfloat16"
            or tensor["canonical_byte_order"] != "little-endian-u16"
            or tensor["bf16_le_bytes"] != expected_bytes
        ):
            raise Qwen3BP2051AttentionWindowTraceError(
                f"trace tensor {name} metadata differs"
            )
        _require_sha256(tensor["bf16_le_sha256"], f"trace tensor {name} SHA-256")
    return index


def _duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise Qwen3BP2051AttentionWindowTraceError(f"JSON object repeats key {key!r}")
        document[key] = value
    return document


def _nonfinite(value: str) -> None:
    raise Qwen3BP2051AttentionWindowTraceError(
        f"non-finite JSON constant {value!r} is forbidden"
    )


def _read_safetensors_header(path: Path) -> tuple[Mapping[str, object], int, int]:
    source = _regular_file(path, "trace sidecar")
    try:
        size = source.stat().st_size
        with source.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise Qwen3BP2051AttentionWindowTraceError(
                    "trace sidecar lacks an 8-byte header length"
                )
            header_bytes = int.from_bytes(prefix, "little")
            if not 0 < header_bytes <= MAX_SAFETENSORS_HEADER_BYTES:
                raise Qwen3BP2051AttentionWindowTraceError(
                    "trace sidecar header size differs"
                )
            raw_header = handle.read(header_bytes)
    except OSError as error:
        raise Qwen3BP2051AttentionWindowTraceError("cannot read trace sidecar") from error
    if len(raw_header) != header_bytes:
        raise Qwen3BP2051AttentionWindowTraceError("trace sidecar header is truncated")
    try:
        header = _require_mapping(
            json.loads(
                raw_header,
                object_pairs_hook=_duplicate_key,
                parse_constant=_nonfinite,
            ),
            "trace sidecar header",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051AttentionWindowTraceError(
            "trace sidecar header is invalid JSON"
        ) from error
    return header, 8 + header_bytes, size


def validate_sidecar_against_manifest(
    manifest: Mapping[str, object], sidecar_path: Path
) -> None:
    """Replay each source-bound BF16 sidecar binding using stdlib only."""

    index = validate_manifest(manifest)
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    metadata = _require_mapping(manifest["sidecar"], "trace sidecar")
    if sidecar.name != metadata["path"] or _sha256_file(sidecar) != metadata["sha256"]:
        raise Qwen3BP2051AttentionWindowTraceError("trace sidecar binding differs")
    header, data_start, size = _read_safetensors_header(sidecar)
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    names = trace_tensor_names(index)
    expected_keys = {_sidecar_key(name) for name in names}
    if set(header) - {"__metadata__"} != expected_keys:
        raise Qwen3BP2051AttentionWindowTraceError("trace sidecar tensor set differs")
    ranges: list[tuple[int, int, str]] = []
    with sidecar.open("rb") as handle:
        for name in names:
            reference = _require_mapping(tensors[name], f"trace tensor {name}")
            entry = _require_mapping(header[reference["key"]], f"sidecar tensor {name}")
            _require_exact_keys(
                entry, {"dtype", "shape", "data_offsets"}, f"sidecar tensor {name}"
            )
            if entry["dtype"] != "BF16" or _shape_from_document(
                entry["shape"], f"sidecar tensor {name} shape"
            ) != _shape_from_document(reference["shape"], f"trace tensor {name} shape"):
                raise Qwen3BP2051AttentionWindowTraceError(
                    f"sidecar tensor {name} metadata differs"
                )
            offsets = entry["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise Qwen3BP2051AttentionWindowTraceError(
                    f"sidecar tensor {name} offsets differ"
                )
            start, end = offsets
            expected_bytes = reference["bf16_le_bytes"]
            if end - start != expected_bytes or data_start + end > size:
                raise Qwen3BP2051AttentionWindowTraceError(
                    f"sidecar tensor {name} byte range differs"
                )
            handle.seek(data_start + start)
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise Qwen3BP2051AttentionWindowTraceError(
                    f"sidecar tensor {name} is truncated"
                )
            if _sha256_bytes(raw) != reference["bf16_le_sha256"]:
                raise Qwen3BP2051AttentionWindowTraceError(
                    f"sidecar tensor {name} raw BF16 hash differs"
                )
            try:
                base._validate_finite_bf16(raw, f"sidecar tensor {name}")
            except base.Qwen3BP2051LayerStageTraceError as error:
                raise _error_from_base(error) from error
            ranges.append((start, end, name))
    expected_start = 0
    for start, end, name in sorted(ranges):
        if start != expected_start:
            raise Qwen3BP2051AttentionWindowTraceError(
                f"sidecar tensor {name} offsets are non-contiguous"
            )
        expected_start = end
    if data_start + expected_start != size:
        raise Qwen3BP2051AttentionWindowTraceError("trace sidecar trailing bytes differ")


def _load_manifest(path: Path) -> dict[str, object]:
    source = _regular_file(path.expanduser(), "trace manifest")
    try:
        raw = source.read_bytes()
        document = dict(
            _require_mapping(
                json.loads(
                    raw,
                    object_pairs_hook=_duplicate_key,
                    parse_constant=_nonfinite,
                ),
                "trace manifest",
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051AttentionWindowTraceError("trace manifest is invalid JSON") from error
    validate_manifest(document)
    if raw != oracle._canonical_json_bytes(document):
        raise Qwen3BP2051AttentionWindowTraceError("trace manifest JSON is not canonical")
    return document


def validate_bindings(
    *,
    manifest_path: Path,
    sidecar_path: Path,
    workload_path: Path,
    teacher_manifest_path: Path,
    teacher_cache_off_sidecar_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Validate source, workload, teacher, and sidecar bindings without CUDA."""

    manifest = _load_manifest(manifest_path)
    root = _regular_directory(repo_root.expanduser(), "repository root")
    try:
        workload = oracle.load_workload(workload_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051AttentionWindowTraceError(str(error)) from error
    teacher_prefix = load_teacher_prefix(
        teacher_manifest_path=teacher_manifest_path,
        teacher_cache_off_sidecar_path=teacher_cache_off_sidecar_path,
    )
    contract = _require_mapping(manifest["contract"], "trace contract")
    if contract["workload"] != _workload_document(workload):
        raise Qwen3BP2051AttentionWindowTraceError("trace workload binding differs")
    if contract["input"] != _teacher_input_document(workload, teacher_prefix):
        raise Qwen3BP2051AttentionWindowTraceError("trace teacher input binding differs")
    observed_sources = _require_mapping(
        _require_mapping(manifest["provenance"], "trace provenance")[
            "source_repository"
        ],
        "trace source provenance",
    )
    if observed_sources["source_dirty"] is not False:
        raise Qwen3BP2051AttentionWindowTraceError("trace source provenance is dirty")
    expected_sources = collect_source_provenance(root)
    if observed_sources["sources"] != expected_sources["sources"]:
        raise Qwen3BP2051AttentionWindowTraceError("trace source hashes differ")
    validate_sidecar_against_manifest(manifest, sidecar_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_p2051_attention_window_trace",
        description="offline Qwen2.5-3B P2051 selected-layer eager-attention trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser(
        "produce", help="write a create-only manifest and BF16 safetensors sidecar"
    )
    produce.add_argument("--layer-index", type=int, required=True)
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--workload", type=Path, required=True)
    produce.add_argument("--teacher-manifest", type=Path, required=True)
    produce.add_argument("--teacher-cache-off-sidecar", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--sidecar", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    validate = commands.add_parser(
        "validate",
        help="validate output, workload, source, and teacher bindings without CUDA",
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--sidecar", type=Path, required=True)
    validate.add_argument("--workload", type=Path, required=True)
    validate.add_argument("--teacher-manifest", type=Path, required=True)
    validate.add_argument("--teacher-cache-off-sidecar", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "produce":
            document = produce_hf_trace(
                layer_index=args.layer_index,
                checkpoint_path=args.checkpoint,
                workload_path=args.workload,
                teacher_manifest_path=args.teacher_manifest,
                teacher_cache_off_sidecar_path=args.teacher_cache_off_sidecar,
                manifest_path=args.manifest,
                sidecar_path=args.sidecar,
                repo_root=args.repo_root,
                device=args.device,
            )
        else:
            document = validate_bindings(
                manifest_path=args.manifest,
                sidecar_path=args.sidecar,
                workload_path=args.workload,
                teacher_manifest_path=args.teacher_manifest,
                teacher_cache_off_sidecar_path=args.teacher_cache_off_sidecar,
                repo_root=args.repo_root,
            )
        profile = _require_mapping(document["trace_profile"], "trace profile")
        print(
            f"validated {document['artifact_kind']}: "
            f"layer={profile['layer_index']} {document['sidecar']['path']} "
            f"sha256={document['sidecar']['sha256']}"
        )
        return 0
    except (
        OSError,
        RuntimeError,
        Qwen3BP2051AttentionWindowTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
