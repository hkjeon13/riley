"""Offline Qwen2.5-3B P2048/M=1 selected-layer boundary trace.

This module is an offline Hugging Face diagnostic.  It replays the immutable
P2048 cache-building prefill and the first teacher-forced M=1 decode, then
captures all internal last-token boundaries in decoder layer three.  Riley
serving never imports this module or its Python dependencies.

The prefill and M1 logits are each compared with the immutable cache-on
teacher before an artifact is written.  The artifact is therefore a narrow
quality discriminator for a future native candidate, never a benchmark,
fallback, or serving selector input.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import qwen3b_cache_on_layer_stage_trace as cache_on
from . import qwen3b_generation_trace as generation
from . import qwen3b_p2051_layer_stage_trace as cache_free
from . import qwen3b_serving_oracle as oracle
from . import qwen3b_stage_trace as stage
from .hf_calibration import SidecarWriter, _default_sidecar_writer, _write_sidecar_exclusive

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2048-cache-on-layer-detail-trace.v2"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2048-cache-on-layer-detail-trace"
TRACE_ID = "qwen3b-p2048-cache-on-m1-layer3-attention-detail-v2"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-cache-on-layer-detail-v2"
DETAILED_LAYER_INDEX = 3
BF16_BYTES = cache_free.BF16_BYTES
TEACHER_DECODE_TOKEN_COUNT = 2

_DETAIL_STAGE_SUFFIXES = (
    "input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_rope",
    "k_rope",
    "attention_scores",
    "attention_probabilities",
    "attention_context",
    "after_attention_residual",
    "post_attention_norm",
    "gate_proj",
    "up_proj",
    "gated",
    "down_proj",
    "output",
)
DETAIL_TENSORS = tuple(
    f"layer{DETAILED_LAYER_INDEX}.{suffix}.last"
    for suffix in _DETAIL_STAGE_SUFFIXES
)
TRACE_TENSORS = (*DETAIL_TENSORS, "last_logits")

SOURCE_PATHS = {
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "teacher_forced_generation": "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    "stage_trace_support": "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    "cache_free_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
    "cache_on_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_cache_on_layer_stage_trace.py",
    "cache_on_layer_detail_trace": "tools/python/reference/riley_reference/qwen3b_cache_on_layer_detail_trace.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "rust_forward": "crates/riley-runtime/src/llama/forward.rs",
    "rust_decode": "crates/riley-runtime/src/llama/decode.rs",
    "rust_p2051_quality_gate": "crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}
SOURCE_PROVENANCE_ENV = "RILEY_QWEN3B_CACHE_ON_LAYER_DETAIL_SOURCE_PROVENANCE"

BackendFactory = Callable[..., "HuggingFaceQwen3BCacheOnLayerDetailTraceBackend"]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BCacheOnLayerDetailTraceError(RuntimeError):
    """Raised when the selected-layer cache-on trace contract is violated."""


@dataclass(frozen=True)
class CapturedTrace:
    """CPU BF16 rows captured by the offline selected-layer diagnostic."""

    tensors: Mapping[str, object]


def _rethrow(error: BaseException) -> Qwen3BCacheOnLayerDetailTraceError:
    return Qwen3BCacheOnLayerDetailTraceError(str(error))


def _regular_file(path: Path, label: str) -> Path:
    try:
        return cache_on._regular_file(path, label)
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error


def _regular_directory(path: Path, label: str) -> Path:
    try:
        return cache_on._regular_directory(path, label)
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    try:
        return cache_on._require_mapping(value, label)
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    try:
        cache_on._require_exact_keys(value, expected, label)
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error


def _require_string(value: object, label: str) -> str:
    try:
        return cache_on._require_string(value, label)
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error


def _require_sha256(value: object, label: str) -> str:
    try:
        return cache_on._require_sha256(value, label)
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error


def _sha256_bytes(value: bytes) -> str:
    return cache_on._sha256_bytes(value)


def _sha256_file(path: Path) -> str:
    return cache_on._sha256_file(path)


def _shape_element_count(shape: Sequence[int]) -> int:
    try:
        return cache_on._shape_element_count(shape)
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error


def _shape_from_document(value: object, label: str) -> tuple[int, ...]:
    try:
        return cache_on._shape_from_document(value, label)
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error


def _sidecar_key(name: str) -> str:
    return f"trace/{name.replace('.', '/')}"


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    base_shapes = cache_free._expected_shapes()
    result: dict[str, tuple[int, ...]] = {}
    attention_shape = (
        cache_free.MODEL_QUERY_HEAD_COUNT,
        oracle.PROMPT_TOKEN_COUNT + 1,
    )
    for suffix in _DETAIL_STAGE_SUFFIXES:
        name = f"layer{DETAILED_LAYER_INDEX}.{suffix}.last"
        result[name] = (
            attention_shape
            if suffix in {"attention_scores", "attention_probabilities"}
            else base_shapes[f"layer0.{suffix}.last"]
        )
    result["last_logits"] = base_shapes["last_logits"]
    if set(result) != set(TRACE_TENSORS):
        raise AssertionError("selected-layer trace shape table differs")
    return result


def _tensor_shape(tensor: object) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace tensor shape is invalid"
        ) from error


def _canonical_bf16_le_bytes(tensor: object, torch: Any) -> bytes:
    try:
        return cache_free._canonical_bf16_le_bytes(tensor, torch)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _validate_tensors(tensors: Mapping[str, object], torch: Any) -> None:
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BCacheOnLayerDetailTraceError("trace tensor names differ")
    expected_shapes = _expected_shapes()
    identities: set[int] = set()
    for name in TRACE_TENSORS:
        tensor = tensors[name]
        if _tensor_shape(tensor) != expected_shapes[name]:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace tensor {name} shape differs"
            )
        if getattr(tensor, "dtype", None) != torch.bfloat16:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace tensor {name} dtype differs"
            )
        identity = id(tensor)
        if identity in identities:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace tensor {name} aliases another capture"
            )
        identities.add(identity)
        _canonical_bf16_le_bytes(tensor, torch)


def _capture_profile_document() -> dict[str, object]:
    return {
        "capture_domain": "cache-on-p2048-m1-decode-selected-layer-last-token-and-attention-rows",
        "id": TRACE_ID,
        "detailed_layer_index": DETAILED_LAYER_INDEX,
        "prefill_source_logit_row": 0,
        "m1_source_logit_row": 1,
        "tensor_count": len(TRACE_TENSORS),
        "rust_consumer": {
            "api": (
                "riley_runtime::llama::PreparedLlamaDecode::"
                "prepare_hf_eager_qwen_p2048_cache_on_m1_attention_detail_trace_for_layer+"
                "decode_hf_eager_qwen_p2048_cache_on_m1_traced"
            ),
            "cache_layout": "contiguous-kv-only",
            "execution": "P2048 prefill then teacher-forced M=1 decode",
            "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
            "serving_eligibility": "none-until-every-stage-is-bf16-exact",
        },
    }


def _execution_document() -> dict[str, object]:
    return dict(cache_on._execution_document())


def _source_logit_bindings(source_rows: Mapping[int, bytes]) -> dict[str, object]:
    if set(source_rows) != {0, 1}:
        raise Qwen3BCacheOnLayerDetailTraceError("source logit row set differs")
    if any(len(raw) != oracle.RAW_LOGIT_BYTES for raw in source_rows.values()):
        raise Qwen3BCacheOnLayerDetailTraceError("source logit row byte count differs")
    return {
        "prefill": {
            "source_logit_row": 0,
            "bf16_le_sha256": _sha256_bytes(source_rows[0]),
        },
        "decode_step_1": {
            "source_logit_row": 1,
            "bf16_le_sha256": _sha256_bytes(source_rows[1]),
        },
    }


class HuggingFaceQwen3BCacheOnLayerDetailTraceBackend:
    """Lazy CUDA-only HF adapter for one P2048/M=1 selected-layer trace."""

    def __init__(self, loader: oracle.HuggingFaceQwen3BBackend) -> None:
        self._loader: oracle.HuggingFaceQwen3BBackend | None = loader
        self._torch = loader._torch
        self._model = loader._model
        self._device = loader._device
        self._base_model, self._layers, self._module = stage._validate_topology(
            self._model
        )
        if DETAILED_LAYER_INDEX >= len(self._layers):
            raise Qwen3BCacheOnLayerDetailTraceError(
                "selected layer is unavailable in the loaded model"
            )
        self.producer_metadata = dict(loader.producer_metadata)
        self.producer_metadata["implementation_id"] = IMPLEMENTATION_ID
        self.producer_metadata["transformers_qwen2_source"] = {
            "path": stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
            "sha256": stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
        }

    @classmethod
    def load(
        cls, *, checkpoint: oracle.CheckpointManifest, device: str
    ) -> "HuggingFaceQwen3BCacheOnLayerDetailTraceBackend":
        loader = oracle.HuggingFaceQwen3BBackend.load(
            checkpoint=checkpoint, device=device
        )
        try:
            return cls(loader)
        except BaseException:
            loader.close()
            raise

    def _capture_last_row(
        self, captured: dict[str, object], name: str, tensor: object
    ) -> None:
        if name in captured:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace hook {name} ran more than once"
            )
        try:
            if int(tensor.shape[0]) != 1:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"trace hook {name} requires batch size one"
                )
            value = tensor[0, -1].detach().to(device="cpu").contiguous()
        except (AttributeError, IndexError, RuntimeError, TypeError) as error:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace hook {name} returned no last-token row"
            ) from error
        captured[name] = value

    def _capture_attention_row(
        self, captured: dict[str, object], name: str, tensor: object
    ) -> None:
        """Capture the selected M1 `[QH,T]` eager-attention row unchanged."""

        if name in captured:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace hook {name} ran more than once"
            )
        expected = (
            1,
            cache_free.MODEL_QUERY_HEAD_COUNT,
            1,
            oracle.PROMPT_TOKEN_COUNT + 1,
        )
        try:
            if _tensor_shape(tensor) != expected:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"trace hook {name} attention shape differs"
                )
            value = tensor[0, :, 0, :].detach().to(device="cpu").contiguous()
        except (AttributeError, IndexError, RuntimeError, TypeError) as error:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace hook {name} returned no attention row"
            ) from error
        captured[name] = value

    @staticmethod
    def _first_tensor(output: object, name: str) -> object:
        if isinstance(output, tuple):
            if not output:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"trace hook {name} returned an empty tuple"
                )
            return output[0]
        return output

    def _cache_sequence_length(self, cache: object) -> int:
        try:
            return generation.HuggingFaceQwen3BGenerationTraceBackend._cache_sequence_length(
                cache
            )
        except generation.Qwen3BGenerationTraceError as error:
            raise _rethrow(error) from error

    def _call(
        self,
        *,
        token_ids: Sequence[int],
        attention_mask_token_count: int,
        position_start: int,
        position_end: int,
        past: object | None,
    ) -> object:
        model = self._model
        if model is None:
            raise Qwen3BCacheOnLayerDetailTraceError("trace backend is closed")
        torch = self._torch
        input_ids = attention_mask = position_ids = None
        try:
            input_ids = torch.tensor(
                [list(token_ids)], dtype=torch.long, device=self._device
            )
            attention_mask = torch.ones(
                (1, attention_mask_token_count),
                dtype=torch.long,
                device=self._device,
            )
            position_ids = torch.arange(
                position_start,
                position_end + 1,
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)
            kwargs: dict[str, object] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": True,
                "logits_to_keep": 1,
                "return_dict": True,
            }
            if past is not None:
                kwargs["past_key_values"] = past
            with torch.inference_mode():
                return model(**kwargs)
        finally:
            del input_ids, attention_mask, position_ids

    def capture(
        self,
        *,
        prompt_token_ids: Sequence[int],
        teacher_decode_token_ids: Sequence[int],
        expected_source_logits: Mapping[int, bytes],
    ) -> CapturedTrace:
        expected_prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        if tuple(prompt_token_ids) != expected_prompt:
            raise Qwen3BCacheOnLayerDetailTraceError(
                "trace input differs from pinned P2048 prompt"
            )
        if len(teacher_decode_token_ids) != TEACHER_DECODE_TOKEN_COUNT:
            raise Qwen3BCacheOnLayerDetailTraceError(
                "trace teacher decode token count differs"
            )
        for index, token_id in enumerate(teacher_decode_token_ids):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
            ):
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"trace teacher decode token {index} is invalid"
                )
        if set(expected_source_logits) != {0, 1}:
            raise Qwen3BCacheOnLayerDetailTraceError("source logit row set differs")
        if any(
            len(raw) != oracle.RAW_LOGIT_BYTES
            for raw in expected_source_logits.values()
        ):
            raise Qwen3BCacheOnLayerDetailTraceError(
                "source logit row byte count differs"
            )

        past: object | None = None
        prefill: object | None = None
        prefill_logits: object | None = None
        try:
            prefill = self._call(
                token_ids=prompt_token_ids,
                attention_mask_token_count=oracle.PROMPT_TOKEN_COUNT,
                position_start=0,
                position_end=oracle.PROMPT_TOKEN_COUNT - 1,
                past=None,
            )
            past = getattr(prefill, "past_key_values", None)
            if past is None or self._cache_sequence_length(past) != oracle.PROMPT_TOKEN_COUNT:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    "prefill trace did not return the P2048 DynamicCache"
                )
            prefill_logits = self._first_tensor(
                getattr(prefill, "logits", None), "prefill logits"
            )
            if prefill_logits is None:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    "prefill trace logits are unavailable"
                )
            prefill_last = prefill_logits[0, -1].detach().to(
                device="cpu"
            ).contiguous()
            if (
                _canonical_bf16_le_bytes(prefill_last, self._torch)
                != expected_source_logits[0]
            ):
                raise Qwen3BCacheOnLayerDetailTraceError(
                    "captured prefill logits differ from source cache-on row"
                )
        except (AttributeError, TypeError):
            raise Qwen3BCacheOnLayerDetailTraceError(
                "prefill trace logits are unavailable"
            ) from None
        finally:
            if prefill_logits is not None:
                del prefill_logits
            if prefill is not None:
                del prefill

        layer = self._layers[DETAILED_LAYER_INDEX]
        attention = layer.self_attn
        active: dict[str, object] | None = None
        rope_calls = 0
        handles: list[object] = []
        original_rope = self._module.apply_rotary_pos_emb
        try:
            original_eager = self._module.eager_attention_forward
            repeat_kv = original_eager.__globals__["repeat_kv"]
        except (AttributeError, KeyError, TypeError) as error:
            raise Qwen3BCacheOnLayerDetailTraceError(
                "Qwen eager attention hook contract changed"
            ) from error

        def capture_output(name: str):
            def hook(_module: object, _args: object, output: object) -> None:
                if active is None:
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        "trace hook ran outside M1"
                    )
                self._capture_last_row(active, name, self._first_tensor(output, name))

            return hook

        def capture_input(name: str):
            def hook(_module: object, args: object) -> None:
                if active is None:
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        "trace hook ran outside M1"
                    )
                if not isinstance(args, tuple) or len(args) != 1:
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        f"trace hook {name} input contract changed"
                    )
                self._capture_last_row(active, name, args[0])

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
            if rope_calls == DETAILED_LAYER_INDEX:
                if active is None or not isinstance(output, tuple) or len(output) != 2:
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        "Qwen rotary hook contract changed"
                    )
                rotated_query, rotated_key = output
                try:
                    query_token_major = rotated_query.transpose(1, 2)
                    key_token_major = rotated_key.transpose(1, 2)
                except (AttributeError, RuntimeError) as error:
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        "Qwen rotary outputs cannot be transposed"
                    ) from error
                prefix = f"layer{DETAILED_LAYER_INDEX}"
                self._capture_last_row(active, f"{prefix}.q_rope.last", query_token_major)
                self._capture_last_row(active, f"{prefix}.k_rope.last", key_token_major)
            rope_calls += 1
            return output

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
            if active is None or module is not attention:
                return original_eager(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling,
                    dropout=dropout,
                    **kwargs,
                )
            try:
                key_states = repeat_kv(key, module.num_key_value_groups)
                value_states = repeat_kv(value, module.num_key_value_groups)
                attention_scores = self._torch.matmul(
                    query, key_states.transpose(2, 3)
                ) * scaling
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                prefix = f"layer{DETAILED_LAYER_INDEX}"
                self._capture_attention_row(
                    active, f"{prefix}.attention_scores.last", attention_scores
                )
                attention_probabilities = self._torch.nn.functional.softmax(
                    attention_scores, dim=-1, dtype=self._torch.float32
                ).to(query.dtype)
                self._capture_attention_row(
                    active,
                    f"{prefix}.attention_probabilities.last",
                    attention_probabilities,
                )
                attention_probabilities = self._torch.nn.functional.dropout(
                    attention_probabilities,
                    p=dropout,
                    training=module.training,
                )
                attention_output = self._torch.matmul(
                    attention_probabilities, value_states
                )
                attention_output = attention_output.transpose(1, 2).contiguous()
                return attention_output, attention_probabilities
            except (AttributeError, RuntimeError, TypeError) as error:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    "Qwen eager attention trace contract changed"
                ) from error

        prefix = f"layer{DETAILED_LAYER_INDEX}"
        handles.extend(
            (
                layer.input_layernorm.register_forward_hook(
                    capture_output(f"{prefix}.input_norm.last")
                ),
                attention.q_proj.register_forward_hook(
                    capture_output(f"{prefix}.q_proj.last")
                ),
                attention.k_proj.register_forward_hook(
                    capture_output(f"{prefix}.k_proj.last")
                ),
                attention.v_proj.register_forward_hook(
                    capture_output(f"{prefix}.v_proj.last")
                ),
                attention.o_proj.register_forward_pre_hook(
                    capture_input(f"{prefix}.attention_context.last")
                ),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    capture_input(f"{prefix}.after_attention_residual.last")
                ),
                layer.post_attention_layernorm.register_forward_hook(
                    capture_output(f"{prefix}.post_attention_norm.last")
                ),
                layer.mlp.gate_proj.register_forward_hook(
                    capture_output(f"{prefix}.gate_proj.last")
                ),
                layer.mlp.up_proj.register_forward_hook(
                    capture_output(f"{prefix}.up_proj.last")
                ),
                layer.mlp.down_proj.register_forward_hook(
                    capture_output(f"{prefix}.down_proj.last")
                ),
                layer.mlp.down_proj.register_forward_pre_hook(
                    capture_input(f"{prefix}.gated.last")
                ),
                layer.register_forward_hook(capture_output(f"{prefix}.output.last")),
            )
        )
        self._module.apply_rotary_pos_emb = traced_rope
        self._module.eager_attention_forward = traced_eager_attention
        try:
            active = {}
            output: object | None = self._call(
                token_ids=(teacher_decode_token_ids[0],),
                attention_mask_token_count=oracle.PROMPT_TOKEN_COUNT + 1,
                position_start=oracle.PROMPT_TOKEN_COUNT,
                position_end=oracle.PROMPT_TOKEN_COUNT,
                past=past,
            )
            try:
                if output is None:
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        "M1 trace did not return an output"
                    )
                next_past = getattr(output, "past_key_values", None)
                if (
                    next_past is None
                    or self._cache_sequence_length(next_past)
                    != oracle.PROMPT_TOKEN_COUNT + 1
                ):
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        "M1 trace DynamicCache length differs"
                    )
                self._capture_last_row(active, "last_logits", output.logits)
                if (
                    _canonical_bf16_le_bytes(active["last_logits"], self._torch)
                    != expected_source_logits[1]
                ):
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        "captured M1 logits differ from source cache-on row"
                    )
                if rope_calls != len(self._layers):
                    raise Qwen3BCacheOnLayerDetailTraceError(
                        "Qwen rotary invocation count differs"
                    )
                ordered = {name: active[name] for name in TRACE_TENSORS}
                _validate_tensors(ordered, self._torch)
                return CapturedTrace(tensors=ordered)
            finally:
                if output is not None:
                    del output
        except KeyError as error:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace hook did not capture {error.args[0]}"
            ) from error
        finally:
            active = None
            self._module.apply_rotary_pos_emb = original_rope
            self._module.eager_attention_forward = original_eager
            for handle in reversed(handles):
                handle.remove()
            del past
            self._torch.cuda.empty_cache()

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


def _validate_source_provenance(value: object) -> None:
    source = _require_mapping(value, "trace source provenance")
    _require_exact_keys(
        source,
        {"git_revision", "source_dirty", "source_status_sha256", "sources"},
        "trace source provenance",
    )
    revision = _require_string(source["git_revision"], "trace Git revision")
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BCacheOnLayerDetailTraceError("trace Git revision is malformed")
    if not isinstance(source["source_dirty"], bool):
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace source dirty must be a boolean"
        )
    _require_sha256(source["source_status_sha256"], "trace source status SHA-256")
    sources = _require_mapping(source["sources"], "trace source records")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BCacheOnLayerDetailTraceError("trace source records differ")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], f"trace source {name}")
        _require_exact_keys(record, {"path", "sha256"}, f"trace source {name}")
        if record["path"] != relative:
            raise Qwen3BCacheOnLayerDetailTraceError("trace source path differs")
        _require_sha256(record["sha256"], f"trace source {name} SHA-256")


def _collect_git_source_provenance(root: Path) -> dict[str, object]:
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
        raise Qwen3BCacheOnLayerDetailTraceError(
            "cannot collect Git provenance"
        ) from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "Git revision has an unexpected format"
        )
    return {
        "git_revision": revision,
        "source_dirty": bool(status),
        "source_status_sha256": _sha256_bytes(status),
        "sources": {
            name: _source_record(root, relative)
            for name, relative in SOURCE_PATHS.items()
        },
    }


def _load_external_source_provenance(root: Path) -> dict[str, object] | None:
    configured = os.environ.get(SOURCE_PROVENANCE_ENV)
    if configured is None:
        return None
    if not configured:
        raise Qwen3BCacheOnLayerDetailTraceError(
            f"{SOURCE_PROVENANCE_ENV} must not be empty"
        )
    source = _regular_file(Path(configured).expanduser(), "external source provenance")
    try:
        document = _require_mapping(
            json.loads(
                source.read_bytes(),
                object_pairs_hook=cache_on._duplicate_key,
                parse_constant=cache_on._nonfinite,
            ),
            "external source provenance",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "external source provenance is invalid"
        ) from error
    _validate_source_provenance(document)
    if document["source_dirty"] is not False:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "external source provenance is dirty"
        )
    if document["source_status_sha256"] != _sha256_bytes(b""):
        raise Qwen3BCacheOnLayerDetailTraceError(
            "external source provenance does not prove a clean pathspec"
        )
    expected_sources = {
        name: _source_record(root, relative) for name, relative in SOURCE_PATHS.items()
    }
    if document["sources"] != expected_sources:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "external source provenance hashes differ"
        )
    return dict(document)


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Collect or validate named source hashes without ML imports."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    external = _load_external_source_provenance(root)
    if external is not None:
        return external
    return _collect_git_source_provenance(root)


def write_source_provenance_exclusive(
    *, repo_root: Path, output_path: Path
) -> dict[str, object]:
    """Write a clean host-Git provenance document for a pinned remote source."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    output = output_path.expanduser()
    if not output.is_absolute() or output.suffix != ".json":
        raise Qwen3BCacheOnLayerDetailTraceError(
            "source provenance output must be an absolute .json path"
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        parent = _regular_directory(output.parent, "source provenance output parent")
    except OSError as error:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "cannot create source provenance output parent"
        ) from error
    output = parent / output.name
    if output == root or output.is_relative_to(root):
        raise Qwen3BCacheOnLayerDetailTraceError(
            "source provenance output must remain outside the repository"
        )
    if output.exists() or output.is_symlink():
        raise Qwen3BCacheOnLayerDetailTraceError(
            "refusing to overwrite source provenance output"
        )
    document = _collect_git_source_provenance(root)
    if document["source_dirty"]:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "cannot write source provenance from a dirty source pathspec"
        )
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        with output.open("xb") as handle:
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "cannot write source provenance output"
        ) from error
    return document


def _tensor_manifest(tensors: Mapping[str, object], torch: Any) -> dict[str, object]:
    _validate_tensors(tensors, torch)
    expected_shapes = _expected_shapes()
    document: dict[str, object] = {}
    for name in TRACE_TENSORS:
        tensor = tensors[name]
        raw = _canonical_bf16_le_bytes(tensor, torch)
        shape = _tensor_shape(tensor)
        expected_bytes = _shape_element_count(expected_shapes[name]) * BF16_BYTES
        if shape != expected_shapes[name] or len(raw) != expected_bytes:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace tensor {name} metadata differs"
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


def _validate_producer(value: object) -> None:
    producer = _require_mapping(value, "trace producer")
    if producer.get("implementation_id") != IMPLEMENTATION_ID:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace producer implementation differs"
        )
    for field in ("runtime_dependency_class", "torch_version", "transformers_version"):
        _require_string(producer.get(field), f"trace producer {field}")
    qwen_source = _require_mapping(
        producer.get("transformers_qwen2_source"), "trace producer Qwen source"
    )
    if dict(qwen_source) != {
        "path": stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
        "sha256": stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
    }:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace producer Qwen source differs"
        )


def _validate_source_logit_bindings(value: object) -> None:
    bindings = _require_mapping(value, "trace source logit bindings")
    _require_exact_keys(
        bindings, {"prefill", "decode_step_1"}, "trace source logit bindings"
    )
    for name, row in (("prefill", 0), ("decode_step_1", 1)):
        record = _require_mapping(bindings[name], f"trace source logit {name}")
        _require_exact_keys(
            record,
            {"source_logit_row", "bf16_le_sha256"},
            f"trace source logit {name}",
        )
        if record["source_logit_row"] != row:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace source logit {name} row differs"
            )
        _require_sha256(
            record["bf16_le_sha256"], f"trace source logit {name} SHA-256"
        )


def validate_manifest(document: Mapping[str, object]) -> None:
    """Validate the immutable selected-layer diagnostic contract without Torch."""

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
        raise Qwen3BCacheOnLayerDetailTraceError("trace manifest identity differs")
    try:
        oracle._validate_utc_text(document["created_at"], "trace created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise _rethrow(error) from error
    _validate_producer(document["producer"])
    if dict(_require_mapping(document["trace_profile"], "trace profile")) != _capture_profile_document():
        raise Qwen3BCacheOnLayerDetailTraceError("trace profile differs")
    contract = _require_mapping(document["contract"], "trace contract")
    _require_exact_keys(
        contract,
        {
            "model_id",
            "model_revision",
            "workload",
            "execution",
            "input",
            "source_logit_bindings",
        },
        "trace contract",
    )
    if (
        contract["model_id"] != oracle.MODEL_ID
        or contract["model_revision"] != oracle.MODEL_REVISION
    ):
        raise Qwen3BCacheOnLayerDetailTraceError("trace model identity differs")
    try:
        cache_free._validate_workload_document(contract["workload"])
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error
    if dict(_require_mapping(contract["execution"], "trace execution")) != _execution_document():
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace execution contract differs"
        )
    try:
        cache_on._validate_teacher_input(contract["input"])
        cache_on._validate_model(document["model"])
    except cache_on.Qwen3BCacheOnLayerStageTraceError as error:
        raise _rethrow(error) from error
    _validate_source_logit_bindings(contract["source_logit_bindings"])
    provenance = _require_mapping(document["provenance"], "trace provenance")
    _require_exact_keys(provenance, {"source_repository"}, "trace provenance")
    _validate_source_provenance(provenance["source_repository"])
    sidecar = _require_mapping(document["sidecar"], "trace sidecar")
    _require_exact_keys(
        sidecar, {"path", "sha256", "format", "tensor_count"}, "trace sidecar"
    )
    name = _require_string(sidecar["path"], "trace sidecar path")
    if Path(name).name != name or not name.endswith(".safetensors"):
        raise Qwen3BCacheOnLayerDetailTraceError("trace sidecar path differs")
    if sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(TRACE_TENSORS):
        raise Qwen3BCacheOnLayerDetailTraceError("trace sidecar metadata differs")
    _require_sha256(sidecar["sha256"], "trace sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "trace tensors")
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BCacheOnLayerDetailTraceError("trace tensor names differ")
    shapes = _expected_shapes()
    for name in TRACE_TENSORS:
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
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"trace tensor {name} metadata differs"
            )
        _require_sha256(tensor["bf16_le_sha256"], f"trace tensor {name} SHA-256")


def validate_sidecar_against_manifest(
    manifest: Mapping[str, object], sidecar_path: Path
) -> None:
    """Replay every BF16 tensor hash and contiguous Safetensors range."""

    validate_manifest(manifest)
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    metadata = _require_mapping(manifest["sidecar"], "trace sidecar")
    if sidecar.name != metadata["path"] or _sha256_file(sidecar) != metadata["sha256"]:
        raise Qwen3BCacheOnLayerDetailTraceError("trace sidecar binding differs")
    try:
        header, data_start, size = cache_free._read_safetensors_header(sidecar)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    expected_keys = {_sidecar_key(name) for name in TRACE_TENSORS}
    if set(header) - {"__metadata__"} != expected_keys:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace sidecar tensor set differs"
        )
    ranges: list[tuple[int, int, str]] = []
    with sidecar.open("rb") as handle:
        for name in TRACE_TENSORS:
            reference = _require_mapping(tensors[name], f"trace tensor {name}")
            entry = _require_mapping(
                header[reference["key"]], f"sidecar tensor {name}"
            )
            _require_exact_keys(
                entry, {"dtype", "shape", "data_offsets"}, f"sidecar tensor {name}"
            )
            if (
                entry["dtype"] != "BF16"
                or _shape_from_document(entry["shape"], f"sidecar tensor {name} shape")
                != _shape_from_document(
                    reference["shape"], f"trace tensor {name} shape"
                )
            ):
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"sidecar tensor {name} metadata differs"
                )
            offsets = entry["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"sidecar tensor {name} offsets differ"
                )
            start, end = offsets
            expected_bytes = reference["bf16_le_bytes"]
            if end - start != expected_bytes or data_start + end > size:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"sidecar tensor {name} byte range differs"
                )
            handle.seek(data_start + start)
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"sidecar tensor {name} is truncated"
                )
            if _sha256_bytes(raw) != reference["bf16_le_sha256"]:
                raise Qwen3BCacheOnLayerDetailTraceError(
                    f"sidecar tensor {name} raw BF16 hash differs"
                )
            try:
                cache_free._validate_finite_bf16(raw, f"sidecar tensor {name}")
            except cache_free.Qwen3BP2051LayerStageTraceError as error:
                raise _rethrow(error) from error
            ranges.append((start, end, name))
    expected_start = 0
    for start, end, name in sorted(ranges):
        if start != expected_start:
            raise Qwen3BCacheOnLayerDetailTraceError(
                f"sidecar tensor {name} offsets are non-contiguous"
            )
        expected_start = end
    if data_start + expected_start != size:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace sidecar trailing bytes differ"
        )


def _load_manifest(path: Path) -> dict[str, object]:
    source = _regular_file(path.expanduser(), "trace manifest")
    try:
        raw = source.read_bytes()
        document = dict(
            _require_mapping(
                json.loads(
                    raw,
                    object_pairs_hook=cache_on._duplicate_key,
                    parse_constant=cache_on._nonfinite,
                ),
                "trace manifest",
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace manifest is invalid JSON"
        ) from error
    validate_manifest(document)
    if raw != oracle._canonical_json_bytes(document):
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace manifest JSON is not canonical"
        )
    return document


def _read_trace_tensor_raw(
    *, manifest: Mapping[str, object], sidecar_path: Path, name: str
) -> bytes:
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    reference = _require_mapping(tensors[name], f"trace tensor {name}")
    try:
        header, data_start, _size = cache_free._read_safetensors_header(sidecar_path)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error
    entry = _require_mapping(header[reference["key"]], f"sidecar tensor {name}")
    offsets = entry["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise Qwen3BCacheOnLayerDetailTraceError("trace sidecar offsets differ")
    start, end = offsets
    with sidecar_path.open("rb") as handle:
        handle.seek(data_start + start)
        raw = handle.read(end - start)
    if len(raw) != end - start:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace sidecar tensor is truncated"
        )
    return raw


def build_manifest(
    *,
    workload: oracle.ServingWorkload,
    checkpoint: oracle.CheckpointManifest,
    teacher: cache_on.TeacherStream,
    tensors: Mapping[str, object],
    torch: Any,
    producer_metadata: Mapping[str, object],
    source_provenance: Mapping[str, object],
    source_rows: Mapping[int, bytes],
    sidecar_name: str,
    sidecar_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(".safetensors"):
        raise Qwen3BCacheOnLayerDetailTraceError("sidecar name differs")
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "trace_id": TRACE_ID,
        "performance_claim_eligible": False,
        "created_at": oracle._utc_text(created_at),
        "producer": dict(producer_metadata),
        "trace_profile": _capture_profile_document(),
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": cache_free._workload_document(workload),
            "execution": _execution_document(),
            "input": cache_on._teacher_input_document(workload, teacher),
            "source_logit_bindings": _source_logit_bindings(source_rows),
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


def produce_hf_trace(
    *,
    checkpoint_path: Path,
    workload_path: Path,
    teacher_manifest_path: Path,
    teacher_cache_on_sidecar_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    repo_root: Path,
    device: str,
    created_at: datetime | None = None,
    backend_factory: BackendFactory = HuggingFaceQwen3BCacheOnLayerDetailTraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Write a create-only selected-layer artifact; Riley is never started."""

    try:
        manifest, sidecar = cache_free._output_paths(
            manifest_path, sidecar_path, repo_root
        )
        workload = oracle.load_workload(workload_path)
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
        teacher = cache_on._load_teacher_stream(
            teacher_manifest_path=teacher_manifest_path,
            teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path,
        )
        all_source_rows = cache_on._load_source_logit_rows(
            teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path
        )
    except (
        OSError,
        cache_free.Qwen3BP2051LayerStageTraceError,
        cache_on.Qwen3BCacheOnLayerStageTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        raise _rethrow(error) from error
    source_rows = {row: all_source_rows[row] for row in (0, 1)}
    provenance = source_provenance_factory(repo_root)
    if provenance.get("source_dirty") is not False:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "cannot produce a source-bound trace from dirty source"
        )
    backend = backend_factory(checkpoint=checkpoint, device=device)
    sidecar_written = False
    try:
        captured = backend.capture(
            prompt_token_ids=workload.prompt_token_ids,
            teacher_decode_token_ids=teacher.token_ids[:TEACHER_DECODE_TOKEN_COUNT],
            expected_source_logits=source_rows,
        )
        tensors = dict(captured.tensors)
        _validate_tensors(tensors, backend._torch)
        _write_sidecar_exclusive(
            sidecar,
            {_sidecar_key(name): tensors[name] for name in TRACE_TENSORS},
            sidecar_writer,
        )
        sidecar_written = True
        try:
            cache_free._ensure_sidecar_consumer_readable(sidecar)
        except cache_free.Qwen3BP2051LayerStageTraceError as error:
            raise _rethrow(error) from error
        document = build_manifest(
            workload=workload,
            checkpoint=checkpoint,
            teacher=teacher,
            tensors=tensors,
            torch=backend._torch,
            producer_metadata=backend.producer_metadata,
            source_provenance=provenance,
            source_rows=source_rows,
            sidecar_name=sidecar.name,
            sidecar_sha256=_sha256_file(sidecar),
            created_at=created_at or datetime.now(timezone.utc),
        )
        validate_sidecar_against_manifest(document, sidecar)
        if (
            _read_trace_tensor_raw(
                manifest=document, sidecar_path=sidecar, name="last_logits"
            )
            != source_rows[1]
        ):
            raise Qwen3BCacheOnLayerDetailTraceError(
                "detail trace M1 logits differ from source cache-on row"
            )
        try:
            oracle.write_artifact_exclusive(manifest, document)
        except oracle.Qwen3BServingOracleError as error:
            raise _rethrow(error) from error
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


def validate_bindings(
    *,
    manifest_path: Path,
    sidecar_path: Path,
    workload_path: Path,
    teacher_manifest_path: Path,
    teacher_cache_on_sidecar_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Replay all source, teacher, sidecar, and logit bindings without CUDA."""

    manifest = _load_manifest(manifest_path)
    root = _regular_directory(repo_root.expanduser(), "repository root")
    try:
        workload = oracle.load_workload(workload_path)
        teacher = cache_on._load_teacher_stream(
            teacher_manifest_path=teacher_manifest_path,
            teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path,
        )
        source_rows = {
            row: cache_on._load_source_logit_rows(
                teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path
            )[row]
            for row in (0, 1)
        }
    except (
        OSError,
        cache_on.Qwen3BCacheOnLayerStageTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        raise _rethrow(error) from error
    contract = _require_mapping(manifest["contract"], "trace contract")
    if contract["workload"] != cache_free._workload_document(workload):
        raise Qwen3BCacheOnLayerDetailTraceError("trace workload binding differs")
    if contract["input"] != cache_on._teacher_input_document(workload, teacher):
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace teacher input binding differs"
        )
    if contract["source_logit_bindings"] != _source_logit_bindings(source_rows):
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace source logit binding differs"
        )
    observed_sources = _require_mapping(
        _require_mapping(manifest["provenance"], "trace provenance")[
            "source_repository"
        ],
        "trace source provenance",
    )
    expected_sources = collect_source_provenance(root)
    if observed_sources["sources"] != expected_sources["sources"]:
        raise Qwen3BCacheOnLayerDetailTraceError(
            "trace source hashes differ"
        )
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    validate_sidecar_against_manifest(manifest, sidecar)
    if (
        _read_trace_tensor_raw(
            manifest=manifest, sidecar_path=sidecar, name="last_logits"
        )
        != source_rows[1]
    ):
        raise Qwen3BCacheOnLayerDetailTraceError(
            "detail trace M1 logits differ from source cache-on row"
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_cache_on_layer_detail_trace",
        description="offline Qwen2.5-3B P2048/M1 layer-three cache-on trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser(
        "produce", help="write a create-only manifest and BF16 Safetensors sidecar"
    )
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--workload", type=Path, required=True)
    produce.add_argument("--teacher-manifest", type=Path, required=True)
    produce.add_argument("--teacher-cache-on-sidecar", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--sidecar", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    provenance = commands.add_parser(
        "provenance",
        help="write clean host-Git provenance for a pinned remote source",
    )
    provenance.add_argument("--repo-root", type=Path, required=True)
    provenance.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser(
        "validate", help="validate output, workload, source, and teacher bindings"
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--sidecar", type=Path, required=True)
    validate.add_argument("--workload", type=Path, required=True)
    validate.add_argument("--teacher-manifest", type=Path, required=True)
    validate.add_argument("--teacher-cache-on-sidecar", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "provenance":
            document = write_source_provenance_exclusive(
                repo_root=args.repo_root, output_path=args.output
            )
            print(
                "wrote source provenance: "
                f"revision={document['git_revision']} "
                f"status_sha256={document['source_status_sha256']}"
            )
            return 0
        if args.command == "produce":
            document = produce_hf_trace(
                checkpoint_path=args.checkpoint,
                workload_path=args.workload,
                teacher_manifest_path=args.teacher_manifest,
                teacher_cache_on_sidecar_path=args.teacher_cache_on_sidecar,
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
                teacher_cache_on_sidecar_path=args.teacher_cache_on_sidecar,
                repo_root=args.repo_root,
            )
        print(
            f"validated {document['artifact_kind']}: "
            f"{document['sidecar']['path']} "
            f"sha256={document['sidecar']['sha256']}"
        )
        return 0
    except (
        OSError,
        RuntimeError,
        Qwen3BCacheOnLayerDetailTraceError,
        cache_on.Qwen3BCacheOnLayerStageTraceError,
        generation.Qwen3BGenerationTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
