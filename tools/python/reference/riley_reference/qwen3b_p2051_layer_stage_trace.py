"""Offline P2051 cache-free Qwen2.5-3B last-token layer trace.

This module is an offline Hugging Face diagnostic producer.  It never starts or
participates in Riley serving.  Its input is constructed from the immutable
P2048 workload plus the first three tokens of a *verified* cache-off
teacher-forced artifact; no teacher token is embedded in this module.

The sidecar keeps only the last token row at each checkpoint.  It is therefore
small enough to locate the first divergent layer without dumping every token
position.  The immutable manifest describes the sidecar key convention and the
cache-free materialized-reference contract a Rust consumer must replay.
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

from . import qwen3b_generation_trace as generation
from . import qwen3b_serving_oracle as oracle
from . import qwen3b_stage_trace as stage
from .hf_calibration import (
    SidecarWriter,
    _default_sidecar_writer,
    _write_sidecar_exclusive,
)

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2051-cache-off-layer-stage-trace.v1"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-layer-stage-trace"
TRACE_ID = "qwen3b-p2051-cache-off-last-token-layer-stage-v1"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-p2051-layer-stage-v1"

TEACHER_PREFIX_TOKEN_COUNT = 3
CONTEXT_TOKEN_COUNT = oracle.PROMPT_TOKEN_COUNT + TEACHER_PREFIX_TOKEN_COUNT
LAST_TOKEN_ROW_INDEX = CONTEXT_TOKEN_COUNT - 1
BF16_BYTES = 2
BF16_EXPONENT_MASK = 0x7F80
MAX_SAFETENSORS_HEADER_BYTES = 1_048_576
MODEL_HIDDEN_SIZE = 2_048
MODEL_LAYER_COUNT = 36
MODEL_QUERY_HEAD_COUNT = 16
MODEL_KEY_VALUE_HEAD_COUNT = 2
MODEL_HEAD_DIMENSION = 128
MODEL_INTERMEDIATE_SIZE = 11_008

# Names are intentionally stable external API.  ``trace/<name with dots
# replaced by slashes>`` is the safetensors key consumed by Rust.  Layer output
# names cover every decoder layer, while layer-zero records include the
# narrower checkpoints necessary to distinguish QKV, attention, residual, and
# MLP boundaries.
_LAYER_OUTPUT_TENSORS = tuple(
    f"layer{index}.output.last" for index in range(1, MODEL_LAYER_COUNT)
)
TRACE_TENSORS = (
    "embedding.last",
    "layer0.input_norm.last",
    "layer0.q_proj.last",
    "layer0.k_proj.last",
    "layer0.v_proj.last",
    "layer0.q_rope.last",
    "layer0.k_rope.last",
    "layer0.attention_context.last",
    "layer0.after_attention_residual.last",
    "layer0.post_attention_norm.last",
    "layer0.gate_proj.last",
    "layer0.up_proj.last",
    "layer0.gated.last",
    "layer0.down_proj.last",
    "layer0.output.last",
    *_LAYER_OUTPUT_TENSORS,
    "final_norm.output.last",
    "last_logits",
)

SOURCE_PATHS = {
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "teacher_forced_generation": "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    "stage_trace_support": "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    "p2051_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "rust_trace_point_api": "crates/riley-runtime/src/llama/forward.rs",
    "rust_trace_point_module": "crates/riley-runtime/src/llama/mod.rs",
    "rust_stage_discriminator": "crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs",
    "rust_stage_discriminator_package": "crates/riley-runtime/Cargo.toml",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}

BackendFactory = Callable[..., "HuggingFaceQwen3BP2051LayerStageTraceBackend"]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BP2051LayerStageTraceError(RuntimeError):
    """Raised when the immutable P2051 layer-trace contract is violated."""


@dataclass(frozen=True)
class CapturedTrace:
    tensors: Mapping[str, object]


@dataclass(frozen=True)
class TeacherPrefix:
    """Verified cache-off teacher prefix and immutable source bindings."""

    token_ids: tuple[int, ...]
    full_teacher_token_ids_sha256: str
    artifact_filename: str
    artifact_sha256: str
    artifact_schema_version: str
    artifact_kind: str
    cache_off_sidecar_filename: str
    cache_off_sidecar_sha256: str
    cache_off_sidecar_tensor_key: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return oracle._sha256_file(path)


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BP2051LayerStageTraceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Qwen3BP2051LayerStageTraceError(
            f"{label} must be a regular non-symlink file"
        )
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BP2051LayerStageTraceError(
            f"cannot resolve {label}: {path}"
        ) from error


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BP2051LayerStageTraceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Qwen3BP2051LayerStageTraceError(
            f"{label} must be a directory, not a symlink"
        )
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BP2051LayerStageTraceError(
            f"cannot resolve {label}: {path}"
        ) from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise Qwen3BP2051LayerStageTraceError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise Qwen3BP2051LayerStageTraceError(
            f"{label} fields differ: missing={missing!r} extra={extra!r}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen3BP2051LayerStageTraceError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    try:
        return oracle._require_sha256(value, label)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051LayerStageTraceError(str(error)) from error


def _shape_element_count(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise Qwen3BP2051LayerStageTraceError(
                "tensor shape dimensions must be positive integers"
            )
        count *= dimension
    return count


def _shape_from_document(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise Qwen3BP2051LayerStageTraceError(f"{label} must be an array")
    try:
        shape = tuple(value)
        _shape_element_count(shape)
    except Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051LayerStageTraceError(f"{label} differs") from error
    return shape


def _validate_finite_bf16(raw: bytes, label: str) -> None:
    if len(raw) % BF16_BYTES != 0:
        raise Qwen3BP2051LayerStageTraceError(f"{label} BF16 byte count differs")
    for offset in range(0, len(raw), BF16_BYTES):
        bits = int.from_bytes(raw[offset : offset + BF16_BYTES], "little")
        if bits & BF16_EXPONENT_MASK == BF16_EXPONENT_MASK:
            raise Qwen3BP2051LayerStageTraceError(f"{label} contains non-finite BF16")


def _sidecar_key(name: str) -> str:
    return f"trace/{name.replace('.', '/')}"


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    hidden = (MODEL_HIDDEN_SIZE,)
    intermediate = (MODEL_INTERMEDIATE_SIZE,)
    shapes: dict[str, tuple[int, ...]] = {
        "embedding.last": hidden,
        "layer0.input_norm.last": hidden,
        "layer0.q_proj.last": hidden,
        "layer0.k_proj.last": (MODEL_KEY_VALUE_HEAD_COUNT * MODEL_HEAD_DIMENSION,),
        "layer0.v_proj.last": (MODEL_KEY_VALUE_HEAD_COUNT * MODEL_HEAD_DIMENSION,),
        "layer0.q_rope.last": (MODEL_QUERY_HEAD_COUNT, MODEL_HEAD_DIMENSION),
        "layer0.k_rope.last": (MODEL_KEY_VALUE_HEAD_COUNT, MODEL_HEAD_DIMENSION),
        "layer0.attention_context.last": hidden,
        "layer0.after_attention_residual.last": hidden,
        "layer0.post_attention_norm.last": hidden,
        "layer0.gate_proj.last": intermediate,
        "layer0.up_proj.last": intermediate,
        "layer0.gated.last": intermediate,
        "layer0.down_proj.last": hidden,
        "layer0.output.last": hidden,
        "final_norm.output.last": hidden,
        "last_logits": (oracle.MODEL_VOCABULARY_SIZE,),
    }
    shapes.update({name: hidden for name in _LAYER_OUTPUT_TENSORS})
    if set(shapes) != set(TRACE_TENSORS):
        raise AssertionError("P2051 trace shape table differs from tensor ordering")
    return shapes


def _rust_consumer_document() -> dict[str, object]:
    """Immutable parser contract for the later Rust materialized-forward test."""

    return {
        "api": "riley_runtime::llama::PreparedLlamaForward::prepare_last_token_layer_trace+execute_last_token_layer_traced",
        "attention_backend": "materialized-reference",
        "cache": False,
        "input_context_token_count": CONTEXT_TOKEN_COUNT,
        "last_token_row_index": LAST_TOKEN_ROW_INDEX,
        "layer_output_indices": list(range(MODEL_LAYER_COUNT)),
        "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
        "trace_row_layout": "last-token-row-only",
    }


def _capture_profile_document() -> dict[str, object]:
    return {
        "capture_domain": "cache-free-p2051-last-token-rows",
        "id": TRACE_ID,
        "tensor_count": len(TRACE_TENSORS),
        "rust_consumer": _rust_consumer_document(),
    }


def _canonical_bf16_le_bytes(tensor: object, torch: Any) -> bytes:
    try:
        raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051LayerStageTraceError(
            "cannot obtain raw BF16 tensor bytes"
        ) from error
    if len(raw) % BF16_BYTES != 0:
        raise Qwen3BP2051LayerStageTraceError("BF16 tensor byte count must be even")
    if sys.byteorder == "little":
        return raw
    if sys.byteorder == "big":
        return b"".join(
            raw[index : index + BF16_BYTES][::-1]
            for index in range(0, len(raw), BF16_BYTES)
        )
    raise Qwen3BP2051LayerStageTraceError("unsupported host byte order")


def _tensor_shape(tensor: object) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise Qwen3BP2051LayerStageTraceError(
            "trace tensor has no valid shape"
        ) from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise Qwen3BP2051LayerStageTraceError("trace tensor shape is empty or invalid")
    return shape


def _validate_tensors(tensors: Mapping[str, object], torch: Any) -> None:
    expected = _expected_shapes()
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BP2051LayerStageTraceError("trace tensor names differ")
    identities: set[int] = set()
    for name in TRACE_TENSORS:
        tensor = tensors[name]
        if _tensor_shape(tensor) != expected[name]:
            raise Qwen3BP2051LayerStageTraceError(f"trace tensor {name} shape differs")
        if getattr(tensor, "dtype", None) != torch.bfloat16:
            raise Qwen3BP2051LayerStageTraceError(f"trace tensor {name} must be BF16")
        if not bool(torch.isfinite(tensor).all().item()):
            raise Qwen3BP2051LayerStageTraceError(f"trace tensor {name} is non-finite")
        if id(tensor) in identities:
            raise Qwen3BP2051LayerStageTraceError(
                "trace tensor captures must be distinct"
            )
        identities.add(id(tensor))


class HuggingFaceQwen3BP2051LayerStageTraceBackend:
    """Lazy HF adapter that captures P2051 last-token rows only."""

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
    ) -> "HuggingFaceQwen3BP2051LayerStageTraceBackend":
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
            raise Qwen3BP2051LayerStageTraceError(
                f"trace hook {name} ran more than once"
            )
        try:
            if int(tensor.shape[0]) != 1:
                raise Qwen3BP2051LayerStageTraceError(
                    f"trace hook {name} requires batch size one"
                )
            value = tensor[0, -1].detach().to(device="cpu").contiguous()
        except (AttributeError, IndexError, RuntimeError, TypeError) as error:
            raise Qwen3BP2051LayerStageTraceError(
                f"trace hook {name} returned no last-token row"
            ) from error
        captured[name] = value

    @staticmethod
    def _first_tensor(output: object, name: str) -> object:
        if isinstance(output, tuple):
            if not output:
                raise Qwen3BP2051LayerStageTraceError(
                    f"trace hook {name} returned an empty tuple"
                )
            return output[0]
        return output

    def capture(self, input_token_ids: Sequence[int]) -> CapturedTrace:
        model = self._model
        if model is None:
            raise Qwen3BP2051LayerStageTraceError("trace backend is closed")
        if len(input_token_ids) != CONTEXT_TOKEN_COUNT:
            raise Qwen3BP2051LayerStageTraceError("trace input token count differs")
        prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        if tuple(input_token_ids[: oracle.PROMPT_TOKEN_COUNT]) != prompt:
            raise Qwen3BP2051LayerStageTraceError(
                "trace input prompt differs from pinned P2048"
            )
        for index, token_id in enumerate(input_token_ids[oracle.PROMPT_TOKEN_COUNT :]):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
            ):
                raise Qwen3BP2051LayerStageTraceError(
                    f"trace teacher token {index} is invalid"
                )

        torch = self._torch
        layer0 = self._layers[0]
        attention = layer0.self_attn
        captured: dict[str, object] = {}
        handles: list[object] = []
        original_rope = self._module.apply_rotary_pos_emb
        rope_calls = 0

        def capture_output(name: str):
            def hook(_module: object, _args: object, output: object) -> None:
                self._capture_last_row(captured, name, self._first_tensor(output, name))

            return hook

        def capture_input(name: str):
            def hook(_module: object, args: object) -> None:
                if not isinstance(args, tuple) or len(args) != 1:
                    raise Qwen3BP2051LayerStageTraceError(
                        f"trace hook {name} input contract changed"
                    )
                self._capture_last_row(captured, name, args[0])

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
                    raise Qwen3BP2051LayerStageTraceError(
                        "Qwen rotary hook contract changed"
                    )
                rotated_query, rotated_key = output
                try:
                    query_token_major = rotated_query.transpose(1, 2)
                    key_token_major = rotated_key.transpose(1, 2)
                except (AttributeError, RuntimeError) as error:
                    raise Qwen3BP2051LayerStageTraceError(
                        "Qwen rotary outputs cannot be transposed"
                    ) from error
                self._capture_last_row(
                    captured, "layer0.q_rope.last", query_token_major
                )
                self._capture_last_row(captured, "layer0.k_rope.last", key_token_major)
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
                raise Qwen3BP2051LayerStageTraceError(
                    "cache-free trace returned a KV cache"
                )
            self._capture_last_row(captured, "last_logits", output.logits)
            if rope_calls != MODEL_LAYER_COUNT:
                raise Qwen3BP2051LayerStageTraceError(
                    "Qwen rotary invocation count differs"
                )
            ordered = {name: captured[name] for name in TRACE_TENSORS}
            _validate_tensors(ordered, torch)
            return CapturedTrace(tensors=ordered)
        except KeyError as error:
            raise Qwen3BP2051LayerStageTraceError(
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
        raise Qwen3BP2051LayerStageTraceError(
            "cannot collect Git provenance"
        ) from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BP2051LayerStageTraceError("Git revision has an unexpected format")
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
        "cache_free": True,
        "dtype": "bfloat16",
        "explicit_attention_mask": True,
        "explicit_input_ids": True,
        "explicit_position_ids": True,
        "inference_mode": True,
        "logits_to_keep": 1,
        "return_dict": True,
        "tf32_enabled": False,
        "use_cache": False,
    }


def load_teacher_prefix(
    *, teacher_manifest_path: Path, teacher_cache_off_sidecar_path: Path
) -> TeacherPrefix:
    """Load teacher[:3] only after replaying the bound cache-off sidecar.

    This is deliberately a stdlib-only verification path.  The returned token
    IDs come from the validated cache-off teacher-forced artifact rather than a
    fixed list in the P2051 trace producer.
    """

    try:
        manifest = _regular_file(
            teacher_manifest_path.expanduser(), "teacher-forced manifest"
        )
        sidecar = _regular_file(
            teacher_cache_off_sidecar_path.expanduser(),
            "teacher-forced cache-off sidecar",
        )
        document = generation._load_manifest(manifest)
        generation.validate_sidecar_against_artifact(
            document=document,
            mode=generation.CACHE_OFF,
            sidecar_path=sidecar,
        )
    except (
        OSError,
        generation.Qwen3BGenerationTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        raise Qwen3BP2051LayerStageTraceError(
            "verified cache-off teacher artifact is unavailable"
        ) from error

    generation_document = _require_mapping(document["generation"], "teacher generation")
    teacher_ids_value = generation_document["teacher_token_ids"]
    if not isinstance(teacher_ids_value, list):
        raise Qwen3BP2051LayerStageTraceError("teacher token IDs are invalid")
    teacher_ids = tuple(teacher_ids_value)
    if len(teacher_ids) != oracle.OUTPUT_TOKEN_COUNT:
        raise Qwen3BP2051LayerStageTraceError("teacher token count differs")
    prefix = teacher_ids[:TEACHER_PREFIX_TOKEN_COUNT]
    if len(prefix) != TEACHER_PREFIX_TOKEN_COUNT or any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
        for token_id in prefix
    ):
        raise Qwen3BP2051LayerStageTraceError("teacher prefix differs")
    cache_off = _require_mapping(
        generation_document["cache_off"], "teacher cache-off document"
    )
    sidecar_document = _require_mapping(
        cache_off["sidecar"], "teacher cache-off sidecar document"
    )
    if sidecar.name != sidecar_document["path"]:
        raise Qwen3BP2051LayerStageTraceError("teacher cache-off sidecar name differs")
    return TeacherPrefix(
        token_ids=prefix,
        full_teacher_token_ids_sha256=_require_sha256(
            generation_document["teacher_token_ids_le_u32_sha256"],
            "teacher full token SHA-256",
        ),
        artifact_filename=manifest.name,
        artifact_sha256=_sha256_file(manifest),
        artifact_schema_version=_require_string(
            document["schema_version"], "teacher artifact schema version"
        ),
        artifact_kind=_require_string(
            document["artifact_kind"], "teacher artifact kind"
        ),
        cache_off_sidecar_filename=sidecar.name,
        cache_off_sidecar_sha256=_sha256_file(sidecar),
        cache_off_sidecar_tensor_key=_require_string(
            sidecar_document["tensor_key"], "teacher cache-off sidecar tensor key"
        ),
    )


def build_input_token_ids(
    workload: oracle.ServingWorkload, teacher_prefix: TeacherPrefix
) -> tuple[int, ...]:
    """Return the exact P2048 + verified cache-off teacher[:3] input."""

    prompt = tuple(workload.prompt_token_ids)
    if len(prompt) != oracle.PROMPT_TOKEN_COUNT:
        raise Qwen3BP2051LayerStageTraceError("workload prompt token count differs")
    if len(teacher_prefix.token_ids) != TEACHER_PREFIX_TOKEN_COUNT:
        raise Qwen3BP2051LayerStageTraceError("teacher prefix token count differs")
    result = prompt + teacher_prefix.token_ids
    if len(result) != CONTEXT_TOKEN_COUNT:
        raise Qwen3BP2051LayerStageTraceError("P2051 input token count differs")
    return result


def _teacher_input_document(
    workload: oracle.ServingWorkload, teacher_prefix: TeacherPrefix
) -> dict[str, object]:
    input_ids = build_input_token_ids(workload, teacher_prefix)
    return {
        "construction": "workload.prompt_token_ids+verified_hf_cache_off.teacher_token_ids[:3]",
        "context_token_count": len(input_ids),
        "input_token_ids_le_u32_sha256": oracle._token_ids_sha256(input_ids),
        "prompt_token_count": oracle.PROMPT_TOKEN_COUNT,
        "teacher_prefix_token_count": TEACHER_PREFIX_TOKEN_COUNT,
        "teacher_prefix_token_ids": list(teacher_prefix.token_ids),
        "teacher_prefix_token_ids_le_u32_sha256": oracle._token_ids_sha256(
            teacher_prefix.token_ids
        ),
        "teacher_source": {
            "artifact_kind": teacher_prefix.artifact_kind,
            "artifact_path": teacher_prefix.artifact_filename,
            "artifact_schema_version": teacher_prefix.artifact_schema_version,
            "artifact_sha256": teacher_prefix.artifact_sha256,
            "cache_mode": generation.CACHE_OFF,
            "cache_off_sidecar_path": teacher_prefix.cache_off_sidecar_filename,
            "cache_off_sidecar_sha256": teacher_prefix.cache_off_sidecar_sha256,
            "cache_off_sidecar_tensor_key": teacher_prefix.cache_off_sidecar_tensor_key,
            "full_teacher_token_count": oracle.OUTPUT_TOKEN_COUNT,
            "full_teacher_token_ids_le_u32_sha256": (
                teacher_prefix.full_teacher_token_ids_sha256
            ),
        },
    }


def _tensor_manifest(tensors: Mapping[str, object], torch: Any) -> dict[str, object]:
    _validate_tensors(tensors, torch)
    document: dict[str, object] = {}
    for name in TRACE_TENSORS:
        tensor = tensors[name]
        raw = _canonical_bf16_le_bytes(tensor, torch)
        shape = _tensor_shape(tensor)
        if len(raw) != _shape_element_count(shape) * BF16_BYTES:
            raise Qwen3BP2051LayerStageTraceError(
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
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(
        ".safetensors"
    ):
        raise Qwen3BP2051LayerStageTraceError("sidecar name differs")
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
    manifest = manifest_path.expanduser()
    sidecar = sidecar_path.expanduser()
    if not manifest.is_absolute() or not sidecar.is_absolute():
        raise Qwen3BP2051LayerStageTraceError("trace artifact paths must be absolute")
    if manifest.parent != sidecar.parent:
        raise Qwen3BP2051LayerStageTraceError("manifest and sidecar must be siblings")
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
        raise Qwen3BP2051LayerStageTraceError("trace artifact extensions differ")
    if manifest == sidecar:
        raise Qwen3BP2051LayerStageTraceError("trace output paths must differ")
    for output in (manifest, sidecar):
        if output == root or root in output.parents:
            raise Qwen3BP2051LayerStageTraceError(
                "trace artifacts must be outside the repository"
            )
        if output.exists() or output.is_symlink():
            raise Qwen3BP2051LayerStageTraceError(
                f"refusing to overwrite existing trace artifact: {output}"
            )
    return manifest, sidecar


def produce_hf_trace(
    *,
    checkpoint_path: Path,
    workload_path: Path,
    teacher_manifest_path: Path,
    teacher_cache_off_sidecar_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    repo_root: Path,
    device: str,
    created_at: datetime | None = None,
    backend_factory: BackendFactory = HuggingFaceQwen3BP2051LayerStageTraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Create a source-bound, cache-free P2051 layer trace outside the repo."""

    manifest, sidecar = _output_paths(manifest_path, sidecar_path, repo_root)
    try:
        workload = oracle.load_workload(workload_path)
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051LayerStageTraceError(str(error)) from error
    teacher_prefix = load_teacher_prefix(
        teacher_manifest_path=teacher_manifest_path,
        teacher_cache_off_sidecar_path=teacher_cache_off_sidecar_path,
    )
    input_token_ids = build_input_token_ids(workload, teacher_prefix)
    provenance = source_provenance_factory(repo_root)
    backend = backend_factory(checkpoint=checkpoint, device=device)
    sidecar_written = False
    try:
        captured = backend.capture(input_token_ids)
        tensors = dict(captured.tensors)
        _validate_tensors(tensors, backend._torch)
        _write_sidecar_exclusive(
            sidecar,
            {_sidecar_key(name): tensors[name] for name in TRACE_TENSORS},
            sidecar_writer,
        )
        sidecar_written = True
        document = build_manifest(
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
            raise Qwen3BP2051LayerStageTraceError(str(error)) from error
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
        raise Qwen3BP2051LayerStageTraceError("trace workload contract differs")


def _validate_teacher_input(value: object) -> None:
    input_document = _require_mapping(value, "trace input contract")
    _require_exact_keys(
        input_document,
        {
            "construction",
            "context_token_count",
            "input_token_ids_le_u32_sha256",
            "prompt_token_count",
            "teacher_prefix_token_count",
            "teacher_prefix_token_ids",
            "teacher_prefix_token_ids_le_u32_sha256",
            "teacher_source",
        },
        "trace input contract",
    )
    if (
        input_document["construction"]
        != "workload.prompt_token_ids+verified_hf_cache_off.teacher_token_ids[:3]"
        or input_document["context_token_count"] != CONTEXT_TOKEN_COUNT
        or input_document["prompt_token_count"] != oracle.PROMPT_TOKEN_COUNT
        or input_document["teacher_prefix_token_count"] != TEACHER_PREFIX_TOKEN_COUNT
    ):
        raise Qwen3BP2051LayerStageTraceError("trace input construction differs")
    prefix_value = input_document["teacher_prefix_token_ids"]
    if (
        not isinstance(prefix_value, list)
        or len(prefix_value) != TEACHER_PREFIX_TOKEN_COUNT
    ):
        raise Qwen3BP2051LayerStageTraceError("trace teacher prefix IDs differ")
    prefix: list[int] = []
    for index, token_id in enumerate(prefix_value):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
        ):
            raise Qwen3BP2051LayerStageTraceError(
                f"trace teacher prefix ID {index} differs"
            )
        prefix.append(token_id)
    if input_document[
        "teacher_prefix_token_ids_le_u32_sha256"
    ] != oracle._token_ids_sha256(prefix):
        raise Qwen3BP2051LayerStageTraceError("trace teacher prefix SHA-256 differs")
    expected_input = (
        (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
    ) + tuple(prefix)
    if input_document["input_token_ids_le_u32_sha256"] != oracle._token_ids_sha256(
        expected_input
    ):
        raise Qwen3BP2051LayerStageTraceError("trace P2051 input SHA-256 differs")
    _require_sha256(
        input_document["input_token_ids_le_u32_sha256"], "trace P2051 input SHA-256"
    )
    source = _require_mapping(input_document["teacher_source"], "trace teacher source")
    _require_exact_keys(
        source,
        {
            "artifact_kind",
            "artifact_path",
            "artifact_schema_version",
            "artifact_sha256",
            "cache_mode",
            "cache_off_sidecar_path",
            "cache_off_sidecar_sha256",
            "cache_off_sidecar_tensor_key",
            "full_teacher_token_count",
            "full_teacher_token_ids_le_u32_sha256",
        },
        "trace teacher source",
    )
    if (
        source["artifact_schema_version"] != generation.SCHEMA_VERSION
        or source["artifact_kind"] != generation.ARTIFACT_KIND
        or source["cache_mode"] != generation.CACHE_OFF
        or source["cache_off_sidecar_tensor_key"] != generation.SIDECAR_TENSOR_KEY
        or source["full_teacher_token_count"] != oracle.OUTPUT_TOKEN_COUNT
    ):
        raise Qwen3BP2051LayerStageTraceError("trace teacher source contract differs")
    for field, suffix in (
        ("artifact_path", ".json"),
        ("cache_off_sidecar_path", ".safetensors"),
    ):
        name = _require_string(source[field], f"trace teacher source {field}")
        if Path(name).name != name or not name.endswith(suffix):
            raise Qwen3BP2051LayerStageTraceError(
                f"trace teacher source {field} differs"
            )
    for field in (
        "artifact_sha256",
        "cache_off_sidecar_sha256",
        "full_teacher_token_ids_le_u32_sha256",
    ):
        _require_sha256(source[field], f"trace teacher source {field}")


def _validate_producer(value: object) -> None:
    producer = _require_mapping(value, "trace producer")
    if producer.get("implementation_id") != IMPLEMENTATION_ID:
        raise Qwen3BP2051LayerStageTraceError("trace producer implementation differs")
    for field in ("runtime_dependency_class", "torch_version", "transformers_version"):
        _require_string(producer.get(field), f"trace producer {field}")
    qwen_source = _require_mapping(
        producer.get("transformers_qwen2_source"), "trace producer Qwen source"
    )
    if dict(qwen_source) != {
        "path": stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
        "sha256": stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
    }:
        raise Qwen3BP2051LayerStageTraceError("trace producer Qwen source differs")


def _validate_model(value: object) -> None:
    model = _require_mapping(value, "trace model")
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
        raise Qwen3BP2051LayerStageTraceError(
            "trace checkpoint receipt filename differs"
        )
    _require_sha256(
        model["checkpoint_receipt_sha256"], "trace checkpoint receipt SHA-256"
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
        raise Qwen3BP2051LayerStageTraceError("trace Git revision is malformed")
    if not isinstance(source["source_dirty"], bool):
        raise Qwen3BP2051LayerStageTraceError("trace source dirty must be a boolean")
    _require_sha256(source["source_status_sha256"], "trace source status SHA-256")
    sources = _require_mapping(source["sources"], "trace source records")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BP2051LayerStageTraceError("trace source records differ")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], f"trace source {name}")
        _require_exact_keys(record, {"path", "sha256"}, f"trace source {name}")
        if record["path"] != relative:
            raise Qwen3BP2051LayerStageTraceError("trace source path differs")
        _require_sha256(record["sha256"], f"trace source {name} SHA-256")


def validate_manifest(document: Mapping[str, object]) -> None:
    """Validate the fixed P2051 contract without loading Torch or a model."""

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
        raise Qwen3BP2051LayerStageTraceError("trace manifest identity differs")
    try:
        oracle._validate_utc_text(document["created_at"], "trace created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051LayerStageTraceError(str(error)) from error
    _validate_producer(document["producer"])
    profile = _require_mapping(document["trace_profile"], "trace profile")
    if dict(profile) != _capture_profile_document():
        raise Qwen3BP2051LayerStageTraceError("trace profile differs")
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
        raise Qwen3BP2051LayerStageTraceError("trace model identity differs")
    _validate_workload_document(contract["workload"])
    if (
        _require_mapping(contract["execution"], "trace execution")
        != _execution_document()
    ):
        raise Qwen3BP2051LayerStageTraceError("trace execution contract differs")
    _validate_teacher_input(contract["input"])
    _validate_model(document["model"])
    provenance = _require_mapping(document["provenance"], "trace provenance")
    _require_exact_keys(provenance, {"source_repository"}, "trace provenance")
    _validate_source_provenance(provenance["source_repository"])
    sidecar = _require_mapping(document["sidecar"], "trace sidecar")
    _require_exact_keys(
        sidecar, {"path", "sha256", "format", "tensor_count"}, "trace sidecar"
    )
    name = _require_string(sidecar["path"], "trace sidecar path")
    if Path(name).name != name or not name.endswith(".safetensors"):
        raise Qwen3BP2051LayerStageTraceError("trace sidecar path differs")
    if sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(
        TRACE_TENSORS
    ):
        raise Qwen3BP2051LayerStageTraceError("trace sidecar metadata differs")
    _require_sha256(sidecar["sha256"], "trace sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "trace tensors")
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BP2051LayerStageTraceError("trace tensor names differ")
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
            raise Qwen3BP2051LayerStageTraceError(
                f"trace tensor {name} metadata differs"
            )
        _require_sha256(tensor["bf16_le_sha256"], f"trace tensor {name} SHA-256")


def _duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise Qwen3BP2051LayerStageTraceError(f"JSON object repeats key {key!r}")
        document[key] = value
    return document


def _nonfinite(value: str) -> None:
    raise Qwen3BP2051LayerStageTraceError(
        f"non-finite JSON constant {value!r} is forbidden"
    )


def _read_safetensors_header(path: Path) -> tuple[Mapping[str, object], int, int]:
    source = _regular_file(path, "trace sidecar")
    try:
        size = source.stat().st_size
        with source.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise Qwen3BP2051LayerStageTraceError(
                    "trace sidecar lacks an 8-byte header length"
                )
            header_bytes = int.from_bytes(prefix, "little")
            if not 0 < header_bytes <= MAX_SAFETENSORS_HEADER_BYTES:
                raise Qwen3BP2051LayerStageTraceError(
                    "trace sidecar header size differs"
                )
            raw_header = handle.read(header_bytes)
    except OSError as error:
        raise Qwen3BP2051LayerStageTraceError("cannot read trace sidecar") from error
    if len(raw_header) != header_bytes:
        raise Qwen3BP2051LayerStageTraceError("trace sidecar header is truncated")
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
        raise Qwen3BP2051LayerStageTraceError(
            "trace sidecar header is invalid JSON"
        ) from error
    return header, 8 + header_bytes, size


def validate_sidecar_against_manifest(
    manifest: Mapping[str, object], sidecar_path: Path
) -> None:
    """Replay every output tensor binding using only stdlib byte operations."""

    validate_manifest(manifest)
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    metadata = _require_mapping(manifest["sidecar"], "trace sidecar")
    if sidecar.name != metadata["path"] or _sha256_file(sidecar) != metadata["sha256"]:
        raise Qwen3BP2051LayerStageTraceError("trace sidecar binding differs")
    header, data_start, size = _read_safetensors_header(sidecar)
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    expected_keys = {_sidecar_key(name) for name in TRACE_TENSORS}
    if set(header) - {"__metadata__"} != expected_keys:
        raise Qwen3BP2051LayerStageTraceError("trace sidecar tensor set differs")
    ranges: list[tuple[int, int, str]] = []
    with sidecar.open("rb") as handle:
        for name in TRACE_TENSORS:
            reference = _require_mapping(tensors[name], f"trace tensor {name}")
            entry = _require_mapping(header[reference["key"]], f"sidecar tensor {name}")
            _require_exact_keys(
                entry, {"dtype", "shape", "data_offsets"}, f"sidecar tensor {name}"
            )
            if entry["dtype"] != "BF16" or _shape_from_document(
                entry["shape"], f"sidecar tensor {name} shape"
            ) != _shape_from_document(reference["shape"], f"trace tensor {name} shape"):
                raise Qwen3BP2051LayerStageTraceError(
                    f"sidecar tensor {name} metadata differs"
                )
            offsets = entry["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets
                )
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise Qwen3BP2051LayerStageTraceError(
                    f"sidecar tensor {name} offsets differ"
                )
            start, end = offsets
            expected_bytes = reference["bf16_le_bytes"]
            if end - start != expected_bytes or data_start + end > size:
                raise Qwen3BP2051LayerStageTraceError(
                    f"sidecar tensor {name} byte range differs"
                )
            handle.seek(data_start + start)
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise Qwen3BP2051LayerStageTraceError(
                    f"sidecar tensor {name} is truncated"
                )
            if _sha256_bytes(raw) != reference["bf16_le_sha256"]:
                raise Qwen3BP2051LayerStageTraceError(
                    f"sidecar tensor {name} raw BF16 hash differs"
                )
            _validate_finite_bf16(raw, f"sidecar tensor {name}")
            ranges.append((start, end, name))
    expected_start = 0
    for start, end, name in sorted(ranges):
        if start != expected_start:
            raise Qwen3BP2051LayerStageTraceError(
                f"sidecar tensor {name} offsets are non-contiguous"
            )
        expected_start = end
    if data_start + expected_start != size:
        raise Qwen3BP2051LayerStageTraceError("trace sidecar trailing bytes differ")


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
        raise Qwen3BP2051LayerStageTraceError(
            "trace manifest is invalid JSON"
        ) from error
    validate_manifest(document)
    if raw != oracle._canonical_json_bytes(document):
        raise Qwen3BP2051LayerStageTraceError("trace manifest JSON is not canonical")
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
    """Validate sidecar, workload, source, and teacher provenance without CUDA."""

    manifest = _load_manifest(manifest_path)
    root = _regular_directory(repo_root.expanduser(), "repository root")
    try:
        workload = oracle.load_workload(workload_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051LayerStageTraceError(str(error)) from error
    teacher_prefix = load_teacher_prefix(
        teacher_manifest_path=teacher_manifest_path,
        teacher_cache_off_sidecar_path=teacher_cache_off_sidecar_path,
    )
    contract = _require_mapping(manifest["contract"], "trace contract")
    if contract["workload"] != _workload_document(workload):
        raise Qwen3BP2051LayerStageTraceError("trace workload binding differs")
    if contract["input"] != _teacher_input_document(workload, teacher_prefix):
        raise Qwen3BP2051LayerStageTraceError("trace teacher input binding differs")
    observed_sources = _require_mapping(
        _require_mapping(manifest["provenance"], "trace provenance")[
            "source_repository"
        ],
        "trace source provenance",
    )
    expected_sources = collect_source_provenance(root)
    if observed_sources["sources"] != expected_sources["sources"]:
        raise Qwen3BP2051LayerStageTraceError("trace source hashes differ")
    validate_sidecar_against_manifest(manifest, sidecar_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_p2051_layer_stage_trace",
        description="offline Qwen2.5-3B P2051 cache-free last-token layer trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser(
        "produce", help="write a create-only manifest and BF16 safetensors sidecar"
    )
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
        print(
            f"validated {document['artifact_kind']}: "
            f"{document['sidecar']['path']} "
            f"sha256={document['sidecar']['sha256']}"
        )
        return 0
    except (
        OSError,
        RuntimeError,
        Qwen3BP2051LayerStageTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
