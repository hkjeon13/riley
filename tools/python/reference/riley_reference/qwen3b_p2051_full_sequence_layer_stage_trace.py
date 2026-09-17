"""Offline full-sequence P2051 Qwen2.5-3B layer-stage trace.

The existing last-token trace proved that a candidate can reproduce the final
row while an earlier causal row still changes the next decoder layer.  This
module captures eleven whole BF16 tensors from a selected decoder layer of the
actual Hugging Face eager forward, including the BF16 cosine/sine tables handed
to rotary embedding. It is an offline oracle only: it never starts Riley, does
not participate in serving, and has no Python dependency at the Rust runtime
boundary.
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
from . import qwen3b_stage_trace as stage
from .hf_calibration import SidecarWriter, _default_sidecar_writer, _write_sidecar_exclusive

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2051-cache-off-full-sequence-layer-stage-trace.v3"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-full-sequence-selected-layer-stage-rope-table-trace"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-p2051-full-sequence-selected-layer-stage-rope-table-v3"
SOURCE_PROVENANCE_ENV = "RILEY_QWEN3B_P2051_FULL_SEQUENCE_STAGE_SOURCE_PROVENANCE"

# The import-time default keeps the ordinary unit-test helpers convenient.
# Produced artifacts bind their selected layer immutably in `trace_profile`.
LAYER_INDEX = 1
BF16_BYTES = 2
MODEL_HIDDEN_SIZE = base.MODEL_HIDDEN_SIZE
MODEL_KEY_VALUE_WIDTH = base.MODEL_KEY_VALUE_HEAD_COUNT * base.MODEL_HEAD_DIMENSION
CONTEXT_TOKEN_COUNT = base.CONTEXT_TOKEN_COUNT

def _validate_layer_index(layer_index: object) -> int:
    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace layer index must be an integer"
        )
    if not 0 <= layer_index < base.MODEL_LAYER_COUNT:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace layer index is outside the model"
        )
    return layer_index


def _trace_id(layer_index: int) -> str:
    _validate_layer_index(layer_index)
    return f"qwen3b-p2051-cache-off-full-sequence-layer{layer_index}-stage-rope-table-v3"


def _trace_tensors(layer_index: int) -> tuple[str, ...]:
    _validate_layer_index(layer_index)
    prefix = f"layer{layer_index}"
    return (
        f"{prefix}.input_norm.full",
        f"{prefix}.q_proj.full",
        f"{prefix}.k_proj.full",
        f"{prefix}.v_proj.full",
        f"{prefix}.rope_cos.full",
        f"{prefix}.rope_sin.full",
        f"{prefix}.q_rope.full",
        f"{prefix}.k_rope.full",
        f"{prefix}.attention_context.full",
        f"{prefix}.after_attention_residual.full",
        f"{prefix}.output.full",
    )


TRACE_ID = _trace_id(LAYER_INDEX)
TRACE_TENSORS = _trace_tensors(LAYER_INDEX)

SOURCE_PATHS = {
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "teacher_forced_generation": "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    "stage_trace_support": "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    "p2051_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
    "p2051_full_sequence_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_p2051_full_sequence_layer_stage_trace.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "rust_trace_point_api": "crates/riley-runtime/src/llama/forward.rs",
    "rust_trace_point_module": "crates/riley-runtime/src/llama/mod.rs",
    "rust_full_sequence_discriminator": "crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs",
    "rust_stage_discriminator_package": "crates/riley-runtime/Cargo.toml",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}

BackendFactory = Callable[..., "HuggingFaceQwen3BP2051FullSequenceLayerStageTraceBackend"]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BP2051FullSequenceLayerStageTraceError(RuntimeError):
    """Raised when the immutable full-sequence P2051 contract is violated."""


@dataclass(frozen=True)
class CapturedTrace:
    tensors: Mapping[str, object]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return oracle._sha256_file(path)


def _regular_file(path: Path, label: str) -> Path:
    try:
        return base._regular_file(path, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error


def _regular_directory(path: Path, label: str) -> Path:
    try:
        return base._regular_directory(path, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    try:
        return base._require_mapping(value, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    try:
        base._require_exact_keys(value, expected, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error


def _require_string(value: object, label: str) -> str:
    try:
        return base._require_string(value, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error


def _require_sha256(value: object, label: str) -> str:
    try:
        return base._require_sha256(value, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error


def _shape_element_count(shape: Sequence[int]) -> int:
    try:
        return base._shape_element_count(shape)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error


def _shape_from_document(value: object, label: str) -> tuple[int, ...]:
    try:
        return base._shape_from_document(value, label)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error


def _sidecar_key(name: str) -> str:
    return f"trace/{name.replace('.', '/')}"


def _expected_shapes(layer_index: int = LAYER_INDEX) -> dict[str, tuple[int, ...]]:
    layer_index = _validate_layer_index(layer_index)
    prefix = f"layer{layer_index}"
    hidden = (CONTEXT_TOKEN_COUNT, MODEL_HIDDEN_SIZE)
    key_value = (CONTEXT_TOKEN_COUNT, MODEL_KEY_VALUE_WIDTH)
    rope = (CONTEXT_TOKEN_COUNT, base.MODEL_HEAD_DIMENSION)
    query_rope = (
        CONTEXT_TOKEN_COUNT,
        base.MODEL_QUERY_HEAD_COUNT,
        base.MODEL_HEAD_DIMENSION,
    )
    key_rope = (
        CONTEXT_TOKEN_COUNT,
        base.MODEL_KEY_VALUE_HEAD_COUNT,
        base.MODEL_HEAD_DIMENSION,
    )
    return {
        f"{prefix}.input_norm.full": hidden,
        f"{prefix}.q_proj.full": hidden,
        f"{prefix}.k_proj.full": key_value,
        f"{prefix}.v_proj.full": key_value,
        f"{prefix}.rope_cos.full": rope,
        f"{prefix}.rope_sin.full": rope,
        f"{prefix}.q_rope.full": query_rope,
        f"{prefix}.k_rope.full": key_rope,
        f"{prefix}.attention_context.full": hidden,
        f"{prefix}.after_attention_residual.full": hidden,
        f"{prefix}.output.full": hidden,
    }


def _capture_profile_document(layer_index: int = LAYER_INDEX) -> dict[str, object]:
    layer_index = _validate_layer_index(layer_index)
    return {
        "capture_domain": "cache-free-p2051-full-sequence-layer-boundaries",
        "id": _trace_id(layer_index),
        "layer_index": layer_index,
        "tensor_count": len(_trace_tensors(layer_index)),
        "rust_consumer": {
            "api": "riley_runtime::llama::PreparedLlamaForward::prepare_full_sequence_layer_stage_trace+execute_full_sequence_layer_stage_traced",
            "attention_backend": "hf-eager-cublaslt-qwen-p2051-probe",
            "cache": False,
            "input_context_token_count": CONTEXT_TOKEN_COUNT,
            "rope_table_capture": "PreparedLlamaForward::download_hugging_face_bf16_rope_table_trace",
            "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
            "trace_row_layout": "full-sequence-token-major",
        },
    }


def _validate_tensors(
    tensors: Mapping[str, object], torch: Any, layer_index: int = LAYER_INDEX
) -> None:
    layer_index = _validate_layer_index(layer_index)
    expected = _expected_shapes(layer_index)
    trace_tensors = _trace_tensors(layer_index)
    if set(tensors) != set(trace_tensors):
        raise Qwen3BP2051FullSequenceLayerStageTraceError("trace tensor names differ")
    identities: set[int] = set()
    for name in trace_tensors:
        tensor = tensors[name]
        try:
            shape = tuple(int(dimension) for dimension in tensor.shape)
        except (AttributeError, TypeError, ValueError) as error:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                f"trace tensor {name} has no valid shape"
            ) from error
        if shape != expected[name]:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                f"trace tensor {name} shape differs"
            )
        if getattr(tensor, "dtype", None) != torch.bfloat16:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                f"trace tensor {name} must be BF16"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                f"trace tensor {name} is non-finite"
            )
        if id(tensor) in identities:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                "trace tensor captures must be distinct"
            )
        identities.add(id(tensor))


class HuggingFaceQwen3BP2051FullSequenceLayerStageTraceBackend:
    """Lazy actual-HF eager adapter that captures one layer's whole sequence."""

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
    ) -> "HuggingFaceQwen3BP2051FullSequenceLayerStageTraceBackend":
        loader = oracle.HuggingFaceQwen3BBackend.load(
            checkpoint=checkpoint, device=device
        )
        try:
            return cls(loader)
        except BaseException:
            loader.close()
            raise

    @staticmethod
    def _first_tensor(output: object, name: str) -> object:
        if isinstance(output, tuple):
            if not output:
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    f"trace hook {name} returned an empty tuple"
                )
            return output[0]
        return output

    def _capture_full(
        self, captured: dict[str, object], name: str, tensor: object
    ) -> None:
        if name in captured:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                f"trace hook {name} ran more than once"
            )
        try:
            if int(tensor.shape[0]) != 1:
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    f"trace hook {name} requires batch size one"
                )
            value = tensor[0].detach().to(device="cpu").contiguous()
        except (AttributeError, IndexError, RuntimeError, TypeError) as error:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                f"trace hook {name} returned no full sequence"
            ) from error
        captured[name] = value

    def capture(
        self, input_token_ids: Sequence[int], *, layer_index: int = LAYER_INDEX
    ) -> CapturedTrace:
        layer_index = _validate_layer_index(layer_index)
        model = self._model
        if model is None:
            raise Qwen3BP2051FullSequenceLayerStageTraceError("trace backend is closed")
        if len(input_token_ids) != CONTEXT_TOKEN_COUNT:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                "trace input token count differs"
            )
        prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        if tuple(input_token_ids[: oracle.PROMPT_TOKEN_COUNT]) != prompt:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                "trace input prompt differs from pinned P2048"
            )
        for index, token_id in enumerate(input_token_ids[oracle.PROMPT_TOKEN_COUNT :]):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < oracle.ADDRESSABLE_TOKEN_COUNT
            ):
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    f"trace teacher token {index} is invalid"
                )

        torch = self._torch
        layer = self._layers[layer_index]
        attention = layer.self_attn
        prefix = f"layer{layer_index}"
        captured: dict[str, object] = {}
        handles: list[object] = []
        original_rope = self._module.apply_rotary_pos_emb
        rope_calls = 0

        def capture_output(name: str):
            def hook(_module: object, _args: object, output: object) -> None:
                self._capture_full(captured, name, self._first_tensor(output, name))

            return hook

        def capture_input(name: str):
            def hook(_module: object, args: object) -> None:
                if not isinstance(args, tuple) or len(args) != 1:
                    raise Qwen3BP2051FullSequenceLayerStageTraceError(
                        f"trace hook {name} input contract changed"
                    )
                self._capture_full(captured, name, args[0])

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
            if rope_calls == layer_index:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise Qwen3BP2051FullSequenceLayerStageTraceError(
                        "Qwen rotary hook contract changed"
                    )
                rotated_query, rotated_key = output
                try:
                    query_token_major = rotated_query.transpose(1, 2)
                    key_token_major = rotated_key.transpose(1, 2)
                except (AttributeError, RuntimeError) as error:
                    raise Qwen3BP2051FullSequenceLayerStageTraceError(
                        "Qwen rotary outputs cannot be transposed"
                    ) from error
                self._capture_full(captured, f"{prefix}.rope_cos.full", cosine)
                self._capture_full(captured, f"{prefix}.rope_sin.full", sine)
                self._capture_full(captured, f"{prefix}.q_rope.full", query_token_major)
                self._capture_full(captured, f"{prefix}.k_rope.full", key_token_major)
            rope_calls += 1
            return output

        handles.extend(
            (
                layer.input_layernorm.register_forward_hook(
                    capture_output(f"{prefix}.input_norm.full")
                ),
                attention.q_proj.register_forward_hook(capture_output(f"{prefix}.q_proj.full")),
                attention.k_proj.register_forward_hook(capture_output(f"{prefix}.k_proj.full")),
                attention.v_proj.register_forward_hook(capture_output(f"{prefix}.v_proj.full")),
                attention.o_proj.register_forward_pre_hook(
                    capture_input(f"{prefix}.attention_context.full")
                ),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    capture_input(f"{prefix}.after_attention_residual.full")
                ),
                layer.register_forward_hook(capture_output(f"{prefix}.output.full")),
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
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    "cache-free trace returned a KV cache"
                )
            if rope_calls != base.MODEL_LAYER_COUNT:
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    "Qwen rotary invocation count differs"
                )
            trace_tensors = _trace_tensors(layer_index)
            ordered = {name: captured[name] for name in trace_tensors}
            _validate_tensors(ordered, torch, layer_index)
            return CapturedTrace(tensors=ordered)
        except KeyError as error:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
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
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "cannot collect Git provenance"
        ) from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
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


def _validate_source_provenance(value: object) -> None:
    source = _require_mapping(value, "trace source provenance")
    _require_exact_keys(
        source,
        {"git_revision", "source_dirty", "source_status_sha256", "sources"},
        "trace source provenance",
    )
    revision = _require_string(source["git_revision"], "trace Git revision")
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace Git revision is malformed"
        )
    if not isinstance(source["source_dirty"], bool):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace source dirty must be a boolean"
        )
    _require_sha256(source["source_status_sha256"], "trace source status SHA-256")
    sources = _require_mapping(source["sources"], "trace source records")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BP2051FullSequenceLayerStageTraceError("trace source records differ")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], f"trace source {name}")
        _require_exact_keys(record, {"path", "sha256"}, f"trace source {name}")
        if record["path"] != relative:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                "trace source path differs"
            )
        _require_sha256(record["sha256"], f"trace source {name} SHA-256")


def _load_external_source_provenance(root: Path) -> dict[str, object] | None:
    configured = os.environ.get(SOURCE_PROVENANCE_ENV)
    if configured is None:
        return None
    if not configured:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            f"{SOURCE_PROVENANCE_ENV} must not be empty"
        )
    source = _regular_file(Path(configured).expanduser(), "external source provenance")
    try:
        document = _require_mapping(
            json.loads(
                source.read_bytes(),
                object_pairs_hook=base._duplicate_key,
                parse_constant=base._nonfinite,
            ),
            "external source provenance",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "external source provenance is invalid"
        ) from error
    _validate_source_provenance(document)
    if document["source_dirty"] is not False:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "external source provenance is dirty"
        )
    if document["source_status_sha256"] != _sha256_bytes(b""):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "external source provenance does not prove a clean pathspec"
        )
    expected_sources = {
        name: _source_record(root, relative) for name, relative in SOURCE_PATHS.items()
    }
    if document["sources"] != expected_sources:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "external source provenance hashes differ"
        )
    return dict(document)


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Collect or revalidate source provenance without importing ML packages."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    external = _load_external_source_provenance(root)
    if external is not None:
        return external
    return _collect_git_source_provenance(root)


def write_source_provenance_exclusive(*, repo_root: Path, output_path: Path) -> dict[str, object]:
    """Write clean host-Git provenance for a pinned container without Git."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    output = output_path.expanduser()
    if not output.is_absolute() or output.suffix != ".json":
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "source provenance output must be an absolute .json path"
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        parent = _regular_directory(output.parent, "source provenance output parent")
    except OSError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "cannot create source provenance output parent"
        ) from error
    output = parent / output.name
    if output == root or output.is_relative_to(root):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "source provenance output must remain outside the repository"
        )
    if output.exists() or output.is_symlink():
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "refusing to overwrite source provenance output"
        )
    document = _collect_git_source_provenance(root)
    if document["source_dirty"]:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
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
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "cannot write source provenance output"
        ) from error
    return document


def _tensor_manifest(
    tensors: Mapping[str, object], torch: Any, layer_index: int = LAYER_INDEX
) -> dict[str, object]:
    layer_index = _validate_layer_index(layer_index)
    _validate_tensors(tensors, torch, layer_index)
    document: dict[str, object] = {}
    for name in _trace_tensors(layer_index):
        tensor = tensors[name]
        try:
            raw = base._canonical_bf16_le_bytes(tensor, torch)
            shape = tuple(int(dimension) for dimension in tensor.shape)
        except base.Qwen3BP2051LayerStageTraceError as error:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
        if len(raw) != _shape_element_count(shape) * BF16_BYTES:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
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


def _validate_producer(value: object) -> None:
    producer = _require_mapping(value, "trace producer")
    if producer.get("implementation_id") != IMPLEMENTATION_ID:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
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
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace producer Qwen source differs"
        )


def build_manifest(
    *,
    workload: oracle.ServingWorkload,
    checkpoint: oracle.CheckpointManifest,
    teacher_prefix: base.TeacherPrefix,
    tensors: Mapping[str, object],
    torch: Any,
    producer_metadata: Mapping[str, object],
    source_provenance: Mapping[str, object],
    sidecar_name: str,
    sidecar_sha256: str,
    created_at: datetime,
    layer_index: int = LAYER_INDEX,
) -> dict[str, object]:
    layer_index = _validate_layer_index(layer_index)
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(
        ".safetensors"
    ):
        raise Qwen3BP2051FullSequenceLayerStageTraceError("sidecar name differs")
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "trace_id": _trace_id(layer_index),
        "performance_claim_eligible": False,
        "created_at": oracle._utc_text(created_at),
        "producer": dict(producer_metadata),
        "trace_profile": _capture_profile_document(layer_index),
        "contract": {
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "workload": base._workload_document(workload),
            "execution": base._execution_document(),
            "input": base._teacher_input_document(workload, teacher_prefix),
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
            "tensor_count": len(_trace_tensors(layer_index)),
        },
        "tensors": _tensor_manifest(tensors, torch, layer_index),
    }
    validate_manifest(document)
    return document


def validate_manifest(document: Mapping[str, object]) -> None:
    """Validate the fixed trace contract without loading Torch or a model."""

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
        or document["performance_claim_eligible"] is not False
    ):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace manifest identity differs"
        )
    try:
        oracle._validate_utc_text(document["created_at"], "trace created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
    _validate_producer(document["producer"])
    profile = _require_mapping(document["trace_profile"], "trace profile")
    layer_index = _validate_layer_index(profile.get("layer_index"))
    if (
        document["trace_id"] != _trace_id(layer_index)
        or dict(profile) != _capture_profile_document(layer_index)
    ):
        raise Qwen3BP2051FullSequenceLayerStageTraceError("trace profile differs")
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
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace model identity differs"
        )
    try:
        base._validate_workload_document(contract["workload"])
        base._validate_teacher_input(contract["input"])
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
    if dict(_require_mapping(contract["execution"], "trace execution")) != base._execution_document():
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace execution contract differs"
        )
    model = _require_mapping(document["model"], "trace model")
    _require_exact_keys(
        model,
        {"checkpoint_path", "checkpoint_receipt_filename", "checkpoint_receipt_sha256"},
        "trace model",
    )
    _require_string(model["checkpoint_path"], "trace checkpoint path")
    if model["checkpoint_receipt_filename"] != oracle.CHECKPOINT_RECEIPT_FILENAME:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace checkpoint receipt filename differs"
        )
    _require_sha256(model["checkpoint_receipt_sha256"], "trace checkpoint receipt SHA-256")
    provenance = _require_mapping(document["provenance"], "trace provenance")
    _require_exact_keys(provenance, {"source_repository"}, "trace provenance")
    _validate_source_provenance(provenance["source_repository"])
    sidecar = _require_mapping(document["sidecar"], "trace sidecar")
    _require_exact_keys(sidecar, {"path", "sha256", "format", "tensor_count"}, "trace sidecar")
    sidecar_name = _require_string(sidecar["path"], "trace sidecar path")
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(".safetensors"):
        raise Qwen3BP2051FullSequenceLayerStageTraceError("trace sidecar path differs")
    trace_tensors = _trace_tensors(layer_index)
    if sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(trace_tensors):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace sidecar metadata differs"
        )
    _require_sha256(sidecar["sha256"], "trace sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "trace tensors")
    if set(tensors) != set(trace_tensors):
        raise Qwen3BP2051FullSequenceLayerStageTraceError("trace tensor names differ")
    shapes = _expected_shapes(layer_index)
    for name in trace_tensors:
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
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                f"trace tensor {name} metadata differs"
            )
        _require_sha256(tensor["bf16_le_sha256"], f"trace tensor {name} SHA-256")


def validate_sidecar_against_manifest(
    manifest: Mapping[str, object], sidecar_path: Path
) -> None:
    """Replay every full-sequence tensor binding with stdlib byte operations."""

    validate_manifest(manifest)
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    metadata = _require_mapping(manifest["sidecar"], "trace sidecar")
    if sidecar.name != metadata["path"] or _sha256_file(sidecar) != metadata["sha256"]:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace sidecar binding differs"
        )
    try:
        header, data_start, size = base._read_safetensors_header(sidecar)
    except base.Qwen3BP2051LayerStageTraceError as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    profile = _require_mapping(manifest["trace_profile"], "trace profile")
    layer_index = _validate_layer_index(profile.get("layer_index"))
    trace_tensors = _trace_tensors(layer_index)
    expected_keys = {_sidecar_key(name) for name in trace_tensors}
    if set(header) - {"__metadata__"} != expected_keys:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace sidecar tensor set differs"
        )
    ranges: list[tuple[int, int, str]] = []
    with sidecar.open("rb") as handle:
        for name in trace_tensors:
            reference = _require_mapping(tensors[name], f"trace tensor {name}")
            entry = _require_mapping(header[reference["key"]], f"sidecar tensor {name}")
            _require_exact_keys(
                entry, {"dtype", "shape", "data_offsets"}, f"sidecar tensor {name}"
            )
            if entry["dtype"] != "BF16" or _shape_from_document(
                entry["shape"], f"sidecar tensor {name} shape"
            ) != _shape_from_document(reference["shape"], f"trace tensor {name} shape"):
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
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
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    f"sidecar tensor {name} offsets differ"
                )
            start, end = offsets
            expected_bytes = reference["bf16_le_bytes"]
            if end - start != expected_bytes or data_start + end > size:
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    f"sidecar tensor {name} byte range differs"
                )
            handle.seek(data_start + start)
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    f"sidecar tensor {name} is truncated"
                )
            if _sha256_bytes(raw) != reference["bf16_le_sha256"]:
                raise Qwen3BP2051FullSequenceLayerStageTraceError(
                    f"sidecar tensor {name} raw BF16 hash differs"
                )
            try:
                base._validate_finite_bf16(raw, f"sidecar tensor {name}")
            except base.Qwen3BP2051LayerStageTraceError as error:
                raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
            ranges.append((start, end, name))
    expected_start = 0
    for start, end, name in sorted(ranges):
        if start != expected_start:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(
                f"sidecar tensor {name} offsets are non-contiguous"
            )
        expected_start = end
    if data_start + expected_start != size:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
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
                    object_pairs_hook=base._duplicate_key,
                    parse_constant=base._nonfinite,
                ),
                "trace manifest",
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace manifest is invalid JSON"
        ) from error
    validate_manifest(document)
    if raw != oracle._canonical_json_bytes(document):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace manifest JSON is not canonical"
        )
    return document


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
    layer_index: int = LAYER_INDEX,
    created_at: datetime | None = None,
    backend_factory: BackendFactory = HuggingFaceQwen3BP2051FullSequenceLayerStageTraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Create the source-bound actual-HF full-sequence layer trace."""

    try:
        layer_index = _validate_layer_index(layer_index)
        manifest, sidecar = base._output_paths(manifest_path, sidecar_path, repo_root)
        workload = oracle.load_workload(workload_path)
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
        teacher_prefix = base.load_teacher_prefix(
            teacher_manifest_path=teacher_manifest_path,
            teacher_cache_off_sidecar_path=teacher_cache_off_sidecar_path,
        )
    except (base.Qwen3BP2051LayerStageTraceError, oracle.Qwen3BServingOracleError) as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
    input_token_ids = base.build_input_token_ids(workload, teacher_prefix)
    provenance = source_provenance_factory(repo_root)
    backend = backend_factory(checkpoint=checkpoint, device=device)
    sidecar_written = False
    try:
        captured = backend.capture(input_token_ids, layer_index=layer_index)
        tensors = dict(captured.tensors)
        _validate_tensors(tensors, backend._torch, layer_index)
        _write_sidecar_exclusive(
            sidecar,
            {
                _sidecar_key(name): tensors[name]
                for name in _trace_tensors(layer_index)
            },
            sidecar_writer,
        )
        sidecar_written = True
        try:
            base._ensure_sidecar_consumer_readable(sidecar)
        except base.Qwen3BP2051LayerStageTraceError as error:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
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
            layer_index=layer_index,
        )
        validate_sidecar_against_manifest(document, sidecar)
        try:
            oracle.write_artifact_exclusive(manifest, document)
        except oracle.Qwen3BServingOracleError as error:
            raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
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
    teacher_cache_off_sidecar_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Validate sidecar, workload, source and teacher bindings without CUDA."""

    manifest = _load_manifest(manifest_path)
    root = _regular_directory(repo_root.expanduser(), "repository root")
    try:
        workload = oracle.load_workload(workload_path)
        teacher_prefix = base.load_teacher_prefix(
            teacher_manifest_path=teacher_manifest_path,
            teacher_cache_off_sidecar_path=teacher_cache_off_sidecar_path,
        )
    except (base.Qwen3BP2051LayerStageTraceError, oracle.Qwen3BServingOracleError) as error:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(str(error)) from error
    contract = _require_mapping(manifest["contract"], "trace contract")
    if contract["workload"] != base._workload_document(workload):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace workload binding differs"
        )
    if contract["input"] != base._teacher_input_document(workload, teacher_prefix):
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace teacher input binding differs"
        )
    observed_sources = _require_mapping(
        _require_mapping(manifest["provenance"], "trace provenance")[
            "source_repository"
        ],
        "trace source provenance",
    )
    expected_sources = collect_source_provenance(root)
    if observed_sources["sources"] != expected_sources["sources"]:
        raise Qwen3BP2051FullSequenceLayerStageTraceError(
            "trace source hashes differ"
        )
    validate_sidecar_against_manifest(manifest, sidecar_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_p2051_full_sequence_layer_stage_trace",
        description="offline Qwen2.5-3B P2051 cache-free full-sequence layer trace",
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
    produce.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    provenance = commands.add_parser(
        "provenance",
        help="write host-Git source provenance for a pinned container without Git",
    )
    provenance.add_argument("--repo-root", type=Path, required=True)
    provenance.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser(
        "validate",
        help="validate output, workload, source and teacher bindings without CUDA",
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
                teacher_cache_off_sidecar_path=args.teacher_cache_off_sidecar,
                manifest_path=args.manifest,
                sidecar_path=args.sidecar,
                repo_root=args.repo_root,
                device=args.device,
                layer_index=args.layer_index,
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
        base.Qwen3BP2051LayerStageTraceError,
        oracle.Qwen3BServingOracleError,
        Qwen3BP2051FullSequenceLayerStageTraceError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
