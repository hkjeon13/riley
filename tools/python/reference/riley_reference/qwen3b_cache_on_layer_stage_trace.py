"""Offline cache-on Qwen2.5-3B P2048/M=1 layer-boundary trace.

This producer is deliberately an offline Hugging Face diagnostic.  It captures
the same narrow, last-token checkpoints as the cache-free P2051 trace, but
replays the serving-shaped contract: a P2048 cache-building prefill followed
by two teacher-forced M=1 DynamicCache decodes.  The output is a create-only
artifact; importing this module never imports Torch, Transformers, or
Safetensors, and Riley serving never imports this Python code.

The source cache-on logits are validated before capture and the three captured
``last_logits`` rows are byte-compared to that source while producing and while
validating bindings.  That makes this a quality gate for a future contiguous
cache-on Rust candidate, not a benchmark or a serving fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import qwen3b_generation_trace as generation
from . import qwen3b_p2051_layer_stage_trace as cache_free
from . import qwen3b_serving_oracle as oracle
from . import qwen3b_stage_trace as stage
from .hf_calibration import (
    SidecarWriter,
    _default_sidecar_writer,
    _write_sidecar_exclusive,
)

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2048-cache-on-layer-stage-trace.v1"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2048-cache-on-layer-stage-trace"
TRACE_ID = "qwen3b-p2048-cache-on-prefill-m1-layer-stage-v1"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-cache-on-layer-stage-v1"

BF16_BYTES = cache_free.BF16_BYTES
MODEL_LAYER_COUNT = cache_free.MODEL_LAYER_COUNT
TEACHER_DECODE_TOKEN_COUNT = 2


@dataclass(frozen=True)
class TraceStep:
    """One exact HF cache-on invocation represented by this artifact."""

    name: str
    source_logit_row: int
    input_token_count: int
    attention_mask_token_count: int
    position_start: int
    position_end: int
    cache_length_before: int
    cache_length_after: int
    teacher_decode_token_index: int | None


TRACE_STEPS = (
    TraceStep(
        name="prefill",
        source_logit_row=0,
        input_token_count=oracle.PROMPT_TOKEN_COUNT,
        attention_mask_token_count=oracle.PROMPT_TOKEN_COUNT,
        position_start=0,
        position_end=oracle.PROMPT_TOKEN_COUNT - 1,
        cache_length_before=0,
        cache_length_after=oracle.PROMPT_TOKEN_COUNT,
        teacher_decode_token_index=None,
    ),
    TraceStep(
        name="decode_step_1",
        source_logit_row=1,
        input_token_count=1,
        attention_mask_token_count=oracle.PROMPT_TOKEN_COUNT + 1,
        position_start=oracle.PROMPT_TOKEN_COUNT,
        position_end=oracle.PROMPT_TOKEN_COUNT,
        cache_length_before=oracle.PROMPT_TOKEN_COUNT,
        cache_length_after=oracle.PROMPT_TOKEN_COUNT + 1,
        teacher_decode_token_index=0,
    ),
    TraceStep(
        name="decode_step_2",
        source_logit_row=2,
        input_token_count=1,
        attention_mask_token_count=oracle.PROMPT_TOKEN_COUNT + 2,
        position_start=oracle.PROMPT_TOKEN_COUNT + 1,
        position_end=oracle.PROMPT_TOKEN_COUNT + 1,
        cache_length_before=oracle.PROMPT_TOKEN_COUNT + 1,
        cache_length_after=oracle.PROMPT_TOKEN_COUNT + 2,
        teacher_decode_token_index=1,
    ),
)

_BASE_TRACE_TENSORS = cache_free.TRACE_TENSORS
TRACE_TENSORS = tuple(
    f"{step.name}.{name}" for step in TRACE_STEPS for name in _BASE_TRACE_TENSORS
)

SOURCE_PATHS = {
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "teacher_forced_generation": "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    "stage_trace_support": "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    "cache_free_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
    "cache_on_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_cache_on_layer_stage_trace.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "rust_forward": "crates/riley-runtime/src/llama/forward.rs",
    "rust_decode": "crates/riley-runtime/src/llama/decode.rs",
    "rust_cache_on_trace": "crates/riley-server/tests/qwen3b_native_d128_trace_gpu.rs",
    "rust_p2051_quality_gate": "crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}

SOURCE_PROVENANCE_ENV = "RILEY_QWEN3B_CACHE_ON_LAYER_STAGE_SOURCE_PROVENANCE"

BackendFactory = Callable[..., "HuggingFaceQwen3BCacheOnLayerStageTraceBackend"]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BCacheOnLayerStageTraceError(RuntimeError):
    """Raised when the cache-on layer-trace contract is violated."""


@dataclass(frozen=True)
class TeacherStream:
    """Verified cache-on teacher source required for the two M=1 calls."""

    token_ids: tuple[int, ...]
    full_teacher_token_ids_sha256: str
    artifact_filename: str
    artifact_sha256: str
    artifact_schema_version: str
    artifact_kind: str
    cache_on_sidecar_filename: str
    cache_on_sidecar_sha256: str
    cache_on_sidecar_tensor_key: str


@dataclass(frozen=True)
class CapturedTrace:
    tensors: Mapping[str, object]


def _rethrow(error: BaseException) -> Qwen3BCacheOnLayerStageTraceError:
    return Qwen3BCacheOnLayerStageTraceError(str(error))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return oracle._sha256_file(path)


def _regular_file(path: Path, label: str) -> Path:
    try:
        return cache_free._regular_file(path, label)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _regular_directory(path: Path, label: str) -> Path:
    try:
        return cache_free._regular_directory(path, label)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    try:
        return cache_free._require_mapping(value, label)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    try:
        cache_free._require_exact_keys(value, expected, label)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _require_string(value: object, label: str) -> str:
    try:
        return cache_free._require_string(value, label)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _require_sha256(value: object, label: str) -> str:
    try:
        return cache_free._require_sha256(value, label)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _shape_element_count(shape: Sequence[int]) -> int:
    try:
        return cache_free._shape_element_count(shape)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _shape_from_document(value: object, label: str) -> tuple[int, ...]:
    try:
        return cache_free._shape_from_document(value, label)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _sidecar_key(name: str) -> str:
    return f"trace/{name.replace('.', '/')}"


def _split_trace_name(name: str) -> tuple[TraceStep, str]:
    for step in TRACE_STEPS:
        prefix = f"{step.name}."
        if name.startswith(prefix):
            base_name = name[len(prefix) :]
            if base_name in _BASE_TRACE_TENSORS:
                return step, base_name
    raise Qwen3BCacheOnLayerStageTraceError("trace tensor name differs")


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    base_shapes = cache_free._expected_shapes()
    return {
        name: base_shapes[_split_trace_name(name)[1]] for name in TRACE_TENSORS
    }


def _step_document(step: TraceStep) -> dict[str, object]:
    return {
        "name": step.name,
        "source_logit_row": step.source_logit_row,
        "input_token_count": step.input_token_count,
        "attention_mask_token_count": step.attention_mask_token_count,
        "position_start": step.position_start,
        "position_end": step.position_end,
        "cache_length_before": step.cache_length_before,
        "cache_length_after": step.cache_length_after,
        "teacher_decode_token_index": step.teacher_decode_token_index,
    }


def _capture_profile_document() -> dict[str, object]:
    return {
        "capture_domain": "cache-on-p2048-prefill-and-m1-decode-last-token-rows",
        "id": TRACE_ID,
        "captured_step_count": len(TRACE_STEPS),
        "tensor_count_per_step": len(_BASE_TRACE_TENSORS),
        "tensor_count": len(TRACE_TENSORS),
        "selected_steps": [_step_document(step) for step in TRACE_STEPS],
        "future_rust_quality_consumer": {
            "cache_layout": "contiguous-kv-only",
            "execution": "P2048 prefill then teacher-forced M=1 decode",
            "serving_eligibility": "none-until-every-stage-is-bf16-exact",
        },
    }


def _execution_document() -> dict[str, object]:
    document = dict(generation._execution_document())
    document["use_cache"] = True
    document["cache_position_argument"] = "omitted-transformers-5.15.1"
    return document


def _validate_step_tensors(tensors: Mapping[str, object], torch: Any) -> None:
    try:
        cache_free._validate_tensors(tensors, torch)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _validate_tensors(tensors: Mapping[str, object], torch: Any) -> None:
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BCacheOnLayerStageTraceError("trace tensor names differ")
    for step in TRACE_STEPS:
        prefix = f"{step.name}."
        selected = {
            name[len(prefix) :]: tensors[name]
            for name in TRACE_TENSORS
            if name.startswith(prefix)
        }
        _validate_step_tensors(selected, torch)


def _canonical_bf16_le_bytes(tensor: object, torch: Any) -> bytes:
    try:
        return cache_free._canonical_bf16_le_bytes(tensor, torch)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


class HuggingFaceQwen3BCacheOnLayerStageTraceBackend:
    """Lazy CUDA-only HF adapter for P2048 prefill plus two M=1 decodes."""

    def __init__(self, loader: oracle.HuggingFaceQwen3BBackend) -> None:
        self._loader: oracle.HuggingFaceQwen3BBackend | None = loader
        self._torch = loader._torch
        self._model = loader._model
        self._device = loader._device
        self._base_model, self._layers, self._module = stage._validate_topology(
            self._model
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
    ) -> "HuggingFaceQwen3BCacheOnLayerStageTraceBackend":
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
            raise Qwen3BCacheOnLayerStageTraceError(
                f"trace hook {name} ran more than once"
            )
        try:
            if int(tensor.shape[0]) != 1:
                raise Qwen3BCacheOnLayerStageTraceError(
                    f"trace hook {name} requires batch size one"
                )
            value = tensor[0, -1].detach().to(device="cpu").contiguous()
        except (AttributeError, IndexError, RuntimeError, TypeError) as error:
            raise Qwen3BCacheOnLayerStageTraceError(
                f"trace hook {name} returned no last-token row"
            ) from error
        captured[name] = value

    @staticmethod
    def _first_tensor(output: object, name: str) -> object:
        if isinstance(output, tuple):
            if not output:
                raise Qwen3BCacheOnLayerStageTraceError(
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

    def capture(
        self,
        *,
        prompt_token_ids: Sequence[int],
        teacher_decode_token_ids: Sequence[int],
        expected_source_logits: Mapping[int, bytes],
    ) -> CapturedTrace:
        model = self._model
        if model is None:
            raise Qwen3BCacheOnLayerStageTraceError("trace backend is closed")
        expected_prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        if tuple(prompt_token_ids) != expected_prompt:
            raise Qwen3BCacheOnLayerStageTraceError(
                "trace input differs from pinned P2048 prompt"
            )
        if len(teacher_decode_token_ids) != TEACHER_DECODE_TOKEN_COUNT:
            raise Qwen3BCacheOnLayerStageTraceError(
                "trace teacher decode token count differs"
            )
        for index, token_id in enumerate(teacher_decode_token_ids):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
            ):
                raise Qwen3BCacheOnLayerStageTraceError(
                    f"trace teacher decode token {index} is invalid"
                )
        if set(expected_source_logits) != {step.source_logit_row for step in TRACE_STEPS}:
            raise Qwen3BCacheOnLayerStageTraceError("source logit row set differs")
        if any(len(raw) != oracle.RAW_LOGIT_BYTES for raw in expected_source_logits.values()):
            raise Qwen3BCacheOnLayerStageTraceError("source logit row byte count differs")

        torch = self._torch
        layer0 = self._layers[0]
        attention = layer0.self_attn
        active: dict[str, object] | None = None
        rope_calls = 0
        handles: list[object] = []
        original_rope = self._module.apply_rotary_pos_emb

        def capture_output(name: str):
            def hook(_module: object, _args: object, output: object) -> None:
                if active is None:
                    raise Qwen3BCacheOnLayerStageTraceError("trace hook ran outside a step")
                self._capture_last_row(active, name, self._first_tensor(output, name))

            return hook

        def capture_input(name: str):
            def hook(_module: object, args: object) -> None:
                if active is None:
                    raise Qwen3BCacheOnLayerStageTraceError("trace hook ran outside a step")
                if not isinstance(args, tuple) or len(args) != 1:
                    raise Qwen3BCacheOnLayerStageTraceError(
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
            if rope_calls == 0:
                if active is None or not isinstance(output, tuple) or len(output) != 2:
                    raise Qwen3BCacheOnLayerStageTraceError(
                        "Qwen rotary hook contract changed"
                    )
                rotated_query, rotated_key = output
                try:
                    query_token_major = rotated_query.transpose(1, 2)
                    key_token_major = rotated_key.transpose(1, 2)
                except (AttributeError, RuntimeError) as error:
                    raise Qwen3BCacheOnLayerStageTraceError(
                        "Qwen rotary outputs cannot be transposed"
                    ) from error
                self._capture_last_row(active, "layer0.q_rope.last", query_token_major)
                self._capture_last_row(active, "layer0.k_rope.last", key_token_major)
            rope_calls += 1
            return output

        handles.extend(
            (
                self._base_model.embed_tokens.register_forward_hook(
                    capture_output("embedding.last")
                ),
                layer0.input_layernorm.register_forward_hook(
                    capture_output("layer0.input_norm.last")
                ),
                attention.q_proj.register_forward_hook(
                    capture_output("layer0.q_proj.last")
                ),
                attention.k_proj.register_forward_hook(
                    capture_output("layer0.k_proj.last")
                ),
                attention.v_proj.register_forward_hook(
                    capture_output("layer0.v_proj.last")
                ),
                attention.o_proj.register_forward_pre_hook(
                    capture_input("layer0.attention_context.last")
                ),
                layer0.post_attention_layernorm.register_forward_pre_hook(
                    capture_input("layer0.after_attention_residual.last")
                ),
                layer0.post_attention_layernorm.register_forward_hook(
                    capture_output("layer0.post_attention_norm.last")
                ),
                layer0.mlp.gate_proj.register_forward_hook(
                    capture_output("layer0.gate_proj.last")
                ),
                layer0.mlp.up_proj.register_forward_hook(
                    capture_output("layer0.up_proj.last")
                ),
                layer0.mlp.down_proj.register_forward_hook(
                    capture_output("layer0.down_proj.last")
                ),
                layer0.mlp.down_proj.register_forward_pre_hook(
                    capture_input("layer0.gated.last")
                ),
                layer0.register_forward_hook(capture_output("layer0.output.last")),
                self._base_model.norm.register_forward_hook(
                    capture_output("final_norm.output.last")
                ),
            )
        )
        for index in range(1, MODEL_LAYER_COUNT):
            handles.append(
                self._layers[index].register_forward_hook(
                    capture_output(f"layer{index}.output.last")
                )
            )

        self._module.apply_rotary_pos_emb = traced_rope
        past: object | None = None
        combined: dict[str, object] = {}
        try:
            for step in TRACE_STEPS:
                if step.teacher_decode_token_index is None:
                    call_ids = tuple(prompt_token_ids)
                    if past is not None:
                        raise Qwen3BCacheOnLayerStageTraceError(
                            "prefill trace cache state differs"
                        )
                else:
                    if past is None:
                        raise Qwen3BCacheOnLayerStageTraceError(
                            "decode trace lacks DynamicCache"
                        )
                    if self._cache_sequence_length(past) != step.cache_length_before:
                        raise Qwen3BCacheOnLayerStageTraceError(
                            "DynamicCache length differs before decode"
                        )
                    call_ids = (
                        teacher_decode_token_ids[step.teacher_decode_token_index],
                    )
                if len(call_ids) != step.input_token_count:
                    raise Qwen3BCacheOnLayerStageTraceError("trace call input count differs")

                active = {}
                rope_calls = 0
                input_ids = torch.tensor(
                    [list(call_ids)], dtype=torch.long, device=self._device
                )
                attention_mask = torch.ones(
                    (1, step.attention_mask_token_count),
                    dtype=torch.long,
                    device=self._device,
                )
                position_ids = torch.arange(
                    step.position_start,
                    step.position_end + 1,
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
                try:
                    with torch.inference_mode():
                        output = model(**kwargs)
                    next_past = getattr(output, "past_key_values", None)
                    if next_past is None:
                        raise Qwen3BCacheOnLayerStageTraceError(
                            "cache-on trace did not return DynamicCache"
                        )
                    if self._cache_sequence_length(next_past) != step.cache_length_after:
                        raise Qwen3BCacheOnLayerStageTraceError(
                            "DynamicCache length differs after model call"
                        )
                    self._capture_last_row(active, "last_logits", output.logits)
                    observed_raw = _canonical_bf16_le_bytes(active["last_logits"], torch)
                    if observed_raw != expected_source_logits[step.source_logit_row]:
                        raise Qwen3BCacheOnLayerStageTraceError(
                            f"captured {step.name} logits differ from source cache-on row"
                        )
                    if rope_calls != MODEL_LAYER_COUNT:
                        raise Qwen3BCacheOnLayerStageTraceError(
                            "Qwen rotary invocation count differs"
                        )
                    _validate_step_tensors(active, torch)
                    combined.update(
                        {f"{step.name}.{name}": active[name] for name in _BASE_TRACE_TENSORS}
                    )
                    past = next_past
                finally:
                    active = None
                    del input_ids, attention_mask, position_ids
                    try:
                        del output
                    except UnboundLocalError:
                        pass
            _validate_tensors(combined, torch)
            return CapturedTrace(tensors=combined)
        except KeyError as error:
            raise Qwen3BCacheOnLayerStageTraceError(
                f"trace hook did not capture {error.args[0]}"
            ) from error
        finally:
            self._module.apply_rotary_pos_emb = original_rope
            for handle in reversed(handles):
                handle.remove()
            del past
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
        raise Qwen3BCacheOnLayerStageTraceError("cannot collect Git provenance") from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BCacheOnLayerStageTraceError("Git revision has an unexpected format")
    return {
        "git_revision": revision,
        "source_dirty": bool(status),
        "source_status_sha256": _sha256_bytes(status),
        "sources": {
            name: _source_record(root, relative)
            for name, relative in SOURCE_PATHS.items()
        },
    }


def _duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise Qwen3BCacheOnLayerStageTraceError(f"JSON object repeats key {key!r}")
        document[key] = value
    return document


def _nonfinite(value: str) -> None:
    raise Qwen3BCacheOnLayerStageTraceError(
        f"non-finite JSON constant {value!r} is forbidden"
    )


def _validate_source_provenance(value: object) -> None:
    source = _require_mapping(value, "trace source provenance")
    _require_exact_keys(
        source,
        {"git_revision", "source_dirty", "source_status_sha256", "sources"},
        "trace source provenance",
    )
    revision = _require_string(source["git_revision"], "trace Git revision")
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BCacheOnLayerStageTraceError("trace Git revision is malformed")
    if not isinstance(source["source_dirty"], bool):
        raise Qwen3BCacheOnLayerStageTraceError("trace source dirty must be a boolean")
    _require_sha256(source["source_status_sha256"], "trace source status SHA-256")
    sources = _require_mapping(source["sources"], "trace source records")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BCacheOnLayerStageTraceError("trace source records differ")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], f"trace source {name}")
        _require_exact_keys(record, {"path", "sha256"}, f"trace source {name}")
        if record["path"] != relative:
            raise Qwen3BCacheOnLayerStageTraceError("trace source path differs")
        _require_sha256(record["sha256"], f"trace source {name} SHA-256")


def _load_external_source_provenance(root: Path) -> dict[str, object] | None:
    configured = os.environ.get(SOURCE_PROVENANCE_ENV)
    if configured is None:
        return None
    if not configured:
        raise Qwen3BCacheOnLayerStageTraceError(
            f"{SOURCE_PROVENANCE_ENV} must not be empty"
        )
    source = _regular_file(Path(configured).expanduser(), "external source provenance")
    try:
        document = _require_mapping(
            json.loads(
                source.read_bytes(),
                object_pairs_hook=_duplicate_key,
                parse_constant=_nonfinite,
            ),
            "external source provenance",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BCacheOnLayerStageTraceError(
            "external source provenance is invalid"
        ) from error
    _validate_source_provenance(document)
    if document["source_dirty"] is not False:
        raise Qwen3BCacheOnLayerStageTraceError("external source provenance is dirty")
    if document["source_status_sha256"] != _sha256_bytes(b""):
        raise Qwen3BCacheOnLayerStageTraceError(
            "external source provenance does not prove a clean pathspec"
        )
    expected_sources = {
        name: _source_record(root, relative) for name, relative in SOURCE_PATHS.items()
    }
    if document["sources"] != expected_sources:
        raise Qwen3BCacheOnLayerStageTraceError(
            "external source provenance hashes differ"
        )
    return dict(document)


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Collect or validate the named source hashes without ML imports."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    external = _load_external_source_provenance(root)
    if external is not None:
        return external
    return _collect_git_source_provenance(root)


def write_source_provenance_exclusive(*, repo_root: Path, output_path: Path) -> dict[str, object]:
    root = _regular_directory(repo_root.expanduser(), "repository root")
    output = output_path.expanduser()
    if not output.is_absolute() or output.suffix != ".json":
        raise Qwen3BCacheOnLayerStageTraceError(
            "source provenance output must be an absolute .json path"
        )
    parent = output.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent = _regular_directory(parent, "source provenance output parent")
    except OSError as error:
        raise Qwen3BCacheOnLayerStageTraceError(
            "cannot create source provenance output parent"
        ) from error
    output = parent / output.name
    if output == root or output.is_relative_to(root):
        raise Qwen3BCacheOnLayerStageTraceError(
            "source provenance output must remain outside the repository"
        )
    if output.exists() or output.is_symlink():
        raise Qwen3BCacheOnLayerStageTraceError(
            "refusing to overwrite source provenance output"
        )
    document = _collect_git_source_provenance(root)
    if document["source_dirty"]:
        raise Qwen3BCacheOnLayerStageTraceError(
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
        raise Qwen3BCacheOnLayerStageTraceError(
            "cannot write source provenance output"
        ) from error
    return document


def _load_teacher_stream(
    *, teacher_manifest_path: Path, teacher_cache_on_sidecar_path: Path
) -> TeacherStream:
    """Load teacher IDs only after replaying the immutable cache-on sidecar."""

    try:
        manifest = _regular_file(teacher_manifest_path.expanduser(), "teacher manifest")
        sidecar = _regular_file(
            teacher_cache_on_sidecar_path.expanduser(), "teacher cache-on sidecar"
        )
        document = generation._load_manifest(manifest)
        generation.validate_sidecar_against_artifact(
            document=document,
            mode=generation.CACHE_ON,
            sidecar_path=sidecar,
        )
    except (
        OSError,
        generation.Qwen3BGenerationTraceError,
        oracle.Qwen3BServingOracleError,
        Qwen3BCacheOnLayerStageTraceError,
    ) as error:
        raise Qwen3BCacheOnLayerStageTraceError(
            "verified cache-on teacher artifact is unavailable"
        ) from error
    generation_document = _require_mapping(document["generation"], "teacher generation")
    token_ids_value = generation_document["teacher_token_ids"]
    if not isinstance(token_ids_value, list):
        raise Qwen3BCacheOnLayerStageTraceError("teacher token IDs are invalid")
    token_ids = tuple(token_ids_value)
    if len(token_ids) != oracle.OUTPUT_TOKEN_COUNT or any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
        for token_id in token_ids
    ):
        raise Qwen3BCacheOnLayerStageTraceError("teacher token IDs differ")
    cache_on = _require_mapping(generation_document["cache_on"], "teacher cache-on")
    sidecar_document = _require_mapping(cache_on["sidecar"], "teacher cache-on sidecar")
    if sidecar.name != sidecar_document["path"]:
        raise Qwen3BCacheOnLayerStageTraceError("teacher cache-on sidecar name differs")
    return TeacherStream(
        token_ids=token_ids,
        full_teacher_token_ids_sha256=_require_sha256(
            generation_document["teacher_token_ids_le_u32_sha256"],
            "teacher full token SHA-256",
        ),
        artifact_filename=manifest.name,
        artifact_sha256=_sha256_file(manifest),
        artifact_schema_version=_require_string(
            document["schema_version"], "teacher artifact schema version"
        ),
        artifact_kind=_require_string(document["artifact_kind"], "teacher artifact kind"),
        cache_on_sidecar_filename=sidecar.name,
        cache_on_sidecar_sha256=_sha256_file(sidecar),
        cache_on_sidecar_tensor_key=_require_string(
            sidecar_document["tensor_key"], "teacher cache-on tensor key"
        ),
    )


def _load_source_logit_rows(
    *, teacher_cache_on_sidecar_path: Path
) -> dict[int, bytes]:
    """Read the three already-validated source rows with stdlib I/O only."""

    sidecar = _regular_file(
        teacher_cache_on_sidecar_path.expanduser(), "teacher cache-on sidecar"
    )
    try:
        header, data_start, size = generation._read_safetensors_header(sidecar)
        entry = _require_mapping(
            header[generation.SIDECAR_TENSOR_KEY], "teacher cache-on logits tensor"
        )
        offsets = entry["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise Qwen3BCacheOnLayerStageTraceError("teacher cache-on offsets differ")
        start, end = offsets
        if start != 0 or end != oracle.OUTPUT_TOKEN_COUNT * oracle.RAW_LOGIT_BYTES:
            raise Qwen3BCacheOnLayerStageTraceError("teacher cache-on byte range differs")
        if data_start + end != size:
            raise Qwen3BCacheOnLayerStageTraceError("teacher cache-on trailing bytes differ")
        rows: dict[int, bytes] = {}
        with sidecar.open("rb") as handle:
            for step in TRACE_STEPS:
                handle.seek(data_start + step.source_logit_row * oracle.RAW_LOGIT_BYTES)
                raw = handle.read(oracle.RAW_LOGIT_BYTES)
                if len(raw) != oracle.RAW_LOGIT_BYTES:
                    raise Qwen3BCacheOnLayerStageTraceError(
                        "teacher cache-on logit row is truncated"
                    )
                rows[step.source_logit_row] = raw
        return rows
    except (OSError, generation.Qwen3BGenerationTraceError) as error:
        raise Qwen3BCacheOnLayerStageTraceError(
            "cannot read teacher cache-on logits"
        ) from error


def _teacher_input_document(
    workload: oracle.ServingWorkload, teacher: TeacherStream
) -> dict[str, object]:
    prompt = tuple(workload.prompt_token_ids)
    decode_ids = teacher.token_ids[:TEACHER_DECODE_TOKEN_COUNT]
    if len(prompt) != oracle.PROMPT_TOKEN_COUNT or len(decode_ids) != TEACHER_DECODE_TOKEN_COUNT:
        raise Qwen3BCacheOnLayerStageTraceError("trace input token count differs")
    return {
        "construction": "verified_hf_cache_on.P2048_prefill_then_teacher_token_ids[:2]_M1_decodes",
        "prefill_prompt_token_count": len(prompt),
        "prefill_prompt_token_ids_le_u32_sha256": oracle._token_ids_sha256(prompt),
        "teacher_decode_token_count": len(decode_ids),
        "teacher_decode_token_ids": list(decode_ids),
        "teacher_decode_token_ids_le_u32_sha256": oracle._token_ids_sha256(decode_ids),
        "teacher_source": {
            "artifact_kind": teacher.artifact_kind,
            "artifact_path": teacher.artifact_filename,
            "artifact_schema_version": teacher.artifact_schema_version,
            "artifact_sha256": teacher.artifact_sha256,
            "cache_mode": generation.CACHE_ON,
            "cache_on_sidecar_path": teacher.cache_on_sidecar_filename,
            "cache_on_sidecar_sha256": teacher.cache_on_sidecar_sha256,
            "cache_on_sidecar_tensor_key": teacher.cache_on_sidecar_tensor_key,
            "full_teacher_token_count": oracle.OUTPUT_TOKEN_COUNT,
            "full_teacher_token_ids_le_u32_sha256": teacher.full_teacher_token_ids_sha256,
        },
    }


def _tensor_shape(tensor: object) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise Qwen3BCacheOnLayerStageTraceError("trace tensor shape is invalid") from error


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
            raise Qwen3BCacheOnLayerStageTraceError(
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
        raise Qwen3BCacheOnLayerStageTraceError("trace producer implementation differs")
    for field in ("runtime_dependency_class", "torch_version", "transformers_version"):
        _require_string(producer.get(field), f"trace producer {field}")
    qwen_source = _require_mapping(
        producer.get("transformers_qwen2_source"), "trace producer Qwen source"
    )
    if dict(qwen_source) != {
        "path": stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
        "sha256": stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
    }:
        raise Qwen3BCacheOnLayerStageTraceError("trace producer Qwen source differs")


def _validate_teacher_input(value: object) -> None:
    document = _require_mapping(value, "trace input contract")
    _require_exact_keys(
        document,
        {
            "construction",
            "prefill_prompt_token_count",
            "prefill_prompt_token_ids_le_u32_sha256",
            "teacher_decode_token_count",
            "teacher_decode_token_ids",
            "teacher_decode_token_ids_le_u32_sha256",
            "teacher_source",
        },
        "trace input contract",
    )
    if (
        document["construction"]
        != "verified_hf_cache_on.P2048_prefill_then_teacher_token_ids[:2]_M1_decodes"
        or document["prefill_prompt_token_count"] != oracle.PROMPT_TOKEN_COUNT
        or document["prefill_prompt_token_ids_le_u32_sha256"]
        != oracle.PROMPT_TOKEN_IDS_SHA256
        or document["teacher_decode_token_count"] != TEACHER_DECODE_TOKEN_COUNT
    ):
        raise Qwen3BCacheOnLayerStageTraceError("trace input construction differs")
    ids_value = document["teacher_decode_token_ids"]
    if not isinstance(ids_value, list) or len(ids_value) != TEACHER_DECODE_TOKEN_COUNT:
        raise Qwen3BCacheOnLayerStageTraceError("trace teacher decode IDs differ")
    ids: list[int] = []
    for index, token_id in enumerate(ids_value):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
        ):
            raise Qwen3BCacheOnLayerStageTraceError(
                f"trace teacher decode ID {index} differs"
            )
        ids.append(token_id)
    if document["teacher_decode_token_ids_le_u32_sha256"] != oracle._token_ids_sha256(ids):
        raise Qwen3BCacheOnLayerStageTraceError("trace teacher decode SHA-256 differs")
    source = _require_mapping(document["teacher_source"], "trace teacher source")
    _require_exact_keys(
        source,
        {
            "artifact_kind",
            "artifact_path",
            "artifact_schema_version",
            "artifact_sha256",
            "cache_mode",
            "cache_on_sidecar_path",
            "cache_on_sidecar_sha256",
            "cache_on_sidecar_tensor_key",
            "full_teacher_token_count",
            "full_teacher_token_ids_le_u32_sha256",
        },
        "trace teacher source",
    )
    if (
        source["artifact_schema_version"] != generation.SCHEMA_VERSION
        or source["artifact_kind"] != generation.ARTIFACT_KIND
        or source["cache_mode"] != generation.CACHE_ON
        or source["cache_on_sidecar_tensor_key"] != generation.SIDECAR_TENSOR_KEY
        or source["full_teacher_token_count"] != oracle.OUTPUT_TOKEN_COUNT
    ):
        raise Qwen3BCacheOnLayerStageTraceError("trace teacher source contract differs")
    for field, suffix in (("artifact_path", ".json"), ("cache_on_sidecar_path", ".safetensors")):
        name = _require_string(source[field], f"trace teacher source {field}")
        if Path(name).name != name or not name.endswith(suffix):
            raise Qwen3BCacheOnLayerStageTraceError(
                f"trace teacher source {field} differs"
            )
    for field in (
        "artifact_sha256",
        "cache_on_sidecar_sha256",
        "full_teacher_token_ids_le_u32_sha256",
    ):
        _require_sha256(source[field], f"trace teacher source {field}")


def _validate_model(value: object) -> None:
    model = _require_mapping(value, "trace model")
    _require_exact_keys(
        model,
        {"checkpoint_path", "checkpoint_receipt_filename", "checkpoint_receipt_sha256"},
        "trace model",
    )
    _require_string(model["checkpoint_path"], "trace checkpoint path")
    if model["checkpoint_receipt_filename"] != oracle.CHECKPOINT_RECEIPT_FILENAME:
        raise Qwen3BCacheOnLayerStageTraceError(
            "trace checkpoint receipt filename differs"
        )
    _require_sha256(model["checkpoint_receipt_sha256"], "trace checkpoint receipt SHA-256")


def validate_manifest(document: Mapping[str, object]) -> None:
    """Validate the immutable P2048/M=1 cache-on contract without ML packages."""

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
        raise Qwen3BCacheOnLayerStageTraceError("trace manifest identity differs")
    try:
        oracle._validate_utc_text(document["created_at"], "trace created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise _rethrow(error) from error
    _validate_producer(document["producer"])
    if dict(_require_mapping(document["trace_profile"], "trace profile")) != _capture_profile_document():
        raise Qwen3BCacheOnLayerStageTraceError("trace profile differs")
    contract = _require_mapping(document["contract"], "trace contract")
    _require_exact_keys(
        contract, {"model_id", "model_revision", "workload", "execution", "input"}, "trace contract"
    )
    if contract["model_id"] != oracle.MODEL_ID or contract["model_revision"] != oracle.MODEL_REVISION:
        raise Qwen3BCacheOnLayerStageTraceError("trace model identity differs")
    try:
        cache_free._validate_workload_document(contract["workload"])
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error
    if dict(_require_mapping(contract["execution"], "trace execution")) != _execution_document():
        raise Qwen3BCacheOnLayerStageTraceError("trace execution contract differs")
    _validate_teacher_input(contract["input"])
    _validate_model(document["model"])
    provenance = _require_mapping(document["provenance"], "trace provenance")
    _require_exact_keys(provenance, {"source_repository"}, "trace provenance")
    _validate_source_provenance(provenance["source_repository"])
    sidecar = _require_mapping(document["sidecar"], "trace sidecar")
    _require_exact_keys(sidecar, {"path", "sha256", "format", "tensor_count"}, "trace sidecar")
    name = _require_string(sidecar["path"], "trace sidecar path")
    if Path(name).name != name or not name.endswith(".safetensors"):
        raise Qwen3BCacheOnLayerStageTraceError("trace sidecar path differs")
    if sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(TRACE_TENSORS):
        raise Qwen3BCacheOnLayerStageTraceError("trace sidecar metadata differs")
    _require_sha256(sidecar["sha256"], "trace sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "trace tensors")
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BCacheOnLayerStageTraceError("trace tensor names differ")
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
            or _shape_from_document(tensor["shape"], f"trace tensor {name} shape") != shapes[name]
            or tensor["dtype"] != "bfloat16"
            or tensor["canonical_byte_order"] != "little-endian-u16"
            or tensor["bf16_le_bytes"] != expected_bytes
        ):
            raise Qwen3BCacheOnLayerStageTraceError(
                f"trace tensor {name} metadata differs"
            )
        _require_sha256(tensor["bf16_le_sha256"], f"trace tensor {name} SHA-256")


def _read_safetensors_header(path: Path) -> tuple[Mapping[str, object], int, int]:
    try:
        return cache_free._read_safetensors_header(path)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def validate_sidecar_against_manifest(
    manifest: Mapping[str, object], sidecar_path: Path
) -> None:
    """Replay all captured tensor hashes and layout using stdlib byte operations."""

    validate_manifest(manifest)
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    metadata = _require_mapping(manifest["sidecar"], "trace sidecar")
    if sidecar.name != metadata["path"] or _sha256_file(sidecar) != metadata["sha256"]:
        raise Qwen3BCacheOnLayerStageTraceError("trace sidecar binding differs")
    header, data_start, size = _read_safetensors_header(sidecar)
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    expected_keys = {_sidecar_key(name) for name in TRACE_TENSORS}
    if set(header) - {"__metadata__"} != expected_keys:
        raise Qwen3BCacheOnLayerStageTraceError("trace sidecar tensor set differs")
    ranges: list[tuple[int, int, str]] = []
    with sidecar.open("rb") as handle:
        for name in TRACE_TENSORS:
            reference = _require_mapping(tensors[name], f"trace tensor {name}")
            entry = _require_mapping(header[reference["key"]], f"sidecar tensor {name}")
            _require_exact_keys(entry, {"dtype", "shape", "data_offsets"}, f"sidecar tensor {name}")
            if (
                entry["dtype"] != "BF16"
                or _shape_from_document(entry["shape"], f"sidecar tensor {name} shape")
                != _shape_from_document(reference["shape"], f"trace tensor {name} shape")
            ):
                raise Qwen3BCacheOnLayerStageTraceError(
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
                raise Qwen3BCacheOnLayerStageTraceError(
                    f"sidecar tensor {name} offsets differ"
                )
            start, end = offsets
            expected_bytes = reference["bf16_le_bytes"]
            if end - start != expected_bytes or data_start + end > size:
                raise Qwen3BCacheOnLayerStageTraceError(
                    f"sidecar tensor {name} byte range differs"
                )
            handle.seek(data_start + start)
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise Qwen3BCacheOnLayerStageTraceError(
                    f"sidecar tensor {name} is truncated"
                )
            if _sha256_bytes(raw) != reference["bf16_le_sha256"]:
                raise Qwen3BCacheOnLayerStageTraceError(
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
            raise Qwen3BCacheOnLayerStageTraceError(
                f"sidecar tensor {name} offsets are non-contiguous"
            )
        expected_start = end
    if data_start + expected_start != size:
        raise Qwen3BCacheOnLayerStageTraceError("trace sidecar trailing bytes differ")


def _load_manifest(path: Path) -> dict[str, object]:
    source = _regular_file(path.expanduser(), "trace manifest")
    try:
        raw = source.read_bytes()
        document = dict(
            _require_mapping(
                json.loads(raw, object_pairs_hook=_duplicate_key, parse_constant=_nonfinite),
                "trace manifest",
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BCacheOnLayerStageTraceError("trace manifest is invalid JSON") from error
    validate_manifest(document)
    if raw != oracle._canonical_json_bytes(document):
        raise Qwen3BCacheOnLayerStageTraceError("trace manifest JSON is not canonical")
    return document


def _read_trace_tensor_raw(
    *, manifest: Mapping[str, object], sidecar_path: Path, name: str
) -> bytes:
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    reference = _require_mapping(tensors[name], f"trace tensor {name}")
    header, data_start, _size = _read_safetensors_header(sidecar_path)
    entry = _require_mapping(header[reference["key"]], f"sidecar tensor {name}")
    offsets = entry["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise Qwen3BCacheOnLayerStageTraceError("trace sidecar offsets differ")
    start, end = offsets
    with sidecar_path.open("rb") as handle:
        handle.seek(data_start + start)
        raw = handle.read(end - start)
    if len(raw) != end - start:
        raise Qwen3BCacheOnLayerStageTraceError("trace sidecar tensor is truncated")
    return raw


def _validate_trace_logits_against_source(
    *,
    manifest: Mapping[str, object],
    sidecar_path: Path,
    source_rows: Mapping[int, bytes],
) -> None:
    for step in TRACE_STEPS:
        observed = _read_trace_tensor_raw(
            manifest=manifest,
            sidecar_path=sidecar_path,
            name=f"{step.name}.last_logits",
        )
        if observed != source_rows[step.source_logit_row]:
            raise Qwen3BCacheOnLayerStageTraceError(
                f"trace {step.name} logits differ from source cache-on row"
            )


def _output_paths(
    manifest_path: Path, sidecar_path: Path, repo_root: Path
) -> tuple[Path, Path]:
    try:
        return cache_free._output_paths(manifest_path, sidecar_path, repo_root)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def build_manifest(
    *,
    workload: oracle.ServingWorkload,
    checkpoint: oracle.CheckpointManifest,
    teacher: TeacherStream,
    tensors: Mapping[str, object],
    torch: Any,
    producer_metadata: Mapping[str, object],
    source_provenance: Mapping[str, object],
    sidecar_name: str,
    sidecar_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(".safetensors"):
        raise Qwen3BCacheOnLayerStageTraceError("sidecar name differs")
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
            "input": _teacher_input_document(workload, teacher),
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
    backend_factory: BackendFactory = HuggingFaceQwen3BCacheOnLayerStageTraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Write a create-only cache-on stage artifact; this never starts Riley."""

    manifest, sidecar = _output_paths(manifest_path, sidecar_path, repo_root)
    try:
        workload = oracle.load_workload(workload_path)
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise _rethrow(error) from error
    teacher = _load_teacher_stream(
        teacher_manifest_path=teacher_manifest_path,
        teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path,
    )
    source_rows = _load_source_logit_rows(
        teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path
    )
    provenance = source_provenance_factory(repo_root)
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
            sidecar_name=sidecar.name,
            sidecar_sha256=_sha256_file(sidecar),
            created_at=created_at or datetime.now(timezone.utc),
        )
        validate_sidecar_against_manifest(document, sidecar)
        _validate_trace_logits_against_source(
            manifest=document, sidecar_path=sidecar, source_rows=source_rows
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
    """Replay all source, logits, and output bindings without CUDA or Torch."""

    manifest = _load_manifest(manifest_path)
    root = _regular_directory(repo_root.expanduser(), "repository root")
    try:
        workload = oracle.load_workload(workload_path)
    except oracle.Qwen3BServingOracleError as error:
        raise _rethrow(error) from error
    teacher = _load_teacher_stream(
        teacher_manifest_path=teacher_manifest_path,
        teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path,
    )
    contract = _require_mapping(manifest["contract"], "trace contract")
    if contract["workload"] != cache_free._workload_document(workload):
        raise Qwen3BCacheOnLayerStageTraceError("trace workload binding differs")
    if contract["input"] != _teacher_input_document(workload, teacher):
        raise Qwen3BCacheOnLayerStageTraceError("trace teacher input binding differs")
    observed_sources = _require_mapping(
        _require_mapping(manifest["provenance"], "trace provenance")["source_repository"],
        "trace source provenance",
    )
    expected_sources = collect_source_provenance(root)
    if observed_sources["sources"] != expected_sources["sources"]:
        raise Qwen3BCacheOnLayerStageTraceError("trace source hashes differ")
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    validate_sidecar_against_manifest(manifest, sidecar)
    _validate_trace_logits_against_source(
        manifest=manifest,
        sidecar_path=sidecar,
        source_rows=_load_source_logit_rows(
            teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path
        ),
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_cache_on_layer_stage_trace",
        description="offline Qwen2.5-3B P2048 cache-on prefill/M=1 layer trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser(
        "produce", help="write a create-only manifest and BF16 safetensors sidecar"
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
        help="write host-Git provenance for a pinned container without Git",
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
        Qwen3BCacheOnLayerStageTraceError,
        generation.Qwen3BGenerationTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
