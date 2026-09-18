"""Offline Qwen2.5-3B P2048 cache-on prefill KV-cache trace.

This module captures the Hugging Face eager ``DynamicCache`` after the pinned
P2048 prefill, after first proving that the prefill logits match the immutable
teacher row.  The sidecar contains a bounded prefix of the cache, not a serving
fallback or a timing result.  Riley serving never imports this module or its
Python dependencies.

The selected layer prefix ends at layer 13 because an M1 discrepancy visible
at layer 14 input must originate in an earlier prefill cache value.  The Rust
consumer compares the logical ``[kv_head, token, head_dim]`` tensors directly
and repeats the prefill with the same owner before reporting a quality result.
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

from . import qwen3b_cache_on_layer_detail_trace as detail
from . import qwen3b_cache_on_layer_stage_trace as cache_on
from . import qwen3b_generation_trace as generation
from . import qwen3b_p2051_layer_stage_trace as cache_free
from . import qwen3b_serving_oracle as oracle
from . import qwen3b_stage_trace as stage
from .hf_calibration import SidecarWriter, _default_sidecar_writer, _write_sidecar_exclusive

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2048-cache-on-prefill-kv-trace.v1"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2048-cache-on-prefill-kv-trace"
TRACE_ID = "qwen3b-p2048-cache-on-prefill-kv-prefix-v1"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-cache-on-prefill-kv-v1"
PREFIX_LAYER_COUNT = 14
BF16_BYTES = cache_free.BF16_BYTES

SOURCE_PATHS = {
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "teacher_forced_generation": "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
    "stage_trace_support": "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    "cache_free_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
    "cache_on_layer_stage_trace": "tools/python/reference/riley_reference/qwen3b_cache_on_layer_stage_trace.py",
    "cache_on_layer_detail_trace": "tools/python/reference/riley_reference/qwen3b_cache_on_layer_detail_trace.py",
    "cache_on_prefill_kv_trace": "tools/python/reference/riley_reference/qwen3b_cache_on_prefill_kv_trace.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "rust_forward": "crates/riley-runtime/src/llama/forward.rs",
    "rust_decode": "crates/riley-runtime/src/llama/decode.rs",
    "rust_p2051_quality_gate": "crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}
SOURCE_PROVENANCE_ENV = "RILEY_QWEN3B_CACHE_ON_PREFILL_KV_SOURCE_PROVENANCE"

BackendFactory = Callable[..., "HuggingFaceQwen3BCacheOnPrefillKvTraceBackend"]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BCacheOnPrefillKvTraceError(RuntimeError):
    """Raised when the fixed prefill KV-cache trace contract is violated."""


@dataclass(frozen=True)
class CapturedTrace:
    """CPU BF16 cache tensors captured by the offline HF diagnostic."""

    tensors: Mapping[str, object]


def _rethrow(error: BaseException) -> Qwen3BCacheOnPrefillKvTraceError:
    return Qwen3BCacheOnPrefillKvTraceError(str(error))


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


def _sha256_bytes(value: bytes) -> str:
    return cache_on._sha256_bytes(value)


def _sha256_file(path: Path) -> str:
    return cache_on._sha256_file(path)


def _tensor_name(layer_index: int, kind: str) -> str:
    if not 0 <= layer_index < PREFIX_LAYER_COUNT:
        raise Qwen3BCacheOnPrefillKvTraceError("cache prefix layer index differs")
    if kind not in {"key", "value"}:
        raise Qwen3BCacheOnPrefillKvTraceError("cache tensor kind differs")
    return f"prefill.layer{layer_index}.{kind}"


def _trace_tensors() -> tuple[str, ...]:
    return tuple(
        _tensor_name(layer_index, kind)
        for layer_index in range(PREFIX_LAYER_COUNT)
        for kind in ("key", "value")
    )


TRACE_TENSORS = _trace_tensors()


def _sidecar_key(name: str) -> str:
    return f"trace/{name.replace('.', '/')}"


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    shape = (
        cache_free.MODEL_KEY_VALUE_HEAD_COUNT,
        oracle.PROMPT_TOKEN_COUNT,
        cache_free.MODEL_HEAD_DIMENSION,
    )
    return {name: shape for name in TRACE_TENSORS}


def _tensor_shape(tensor: object) -> tuple[int, ...]:
    try:
        return tuple(int(dimension) for dimension in tensor.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise Qwen3BCacheOnPrefillKvTraceError("cache tensor shape is invalid") from error


def _canonical_bf16_le_bytes(tensor: object, torch: Any) -> bytes:
    try:
        return cache_free._canonical_bf16_le_bytes(tensor, torch)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error


def _validate_tensors(tensors: Mapping[str, object], torch: Any) -> None:
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BCacheOnPrefillKvTraceError("cache tensor names differ")
    shapes = _expected_shapes()
    identities: set[int] = set()
    for name in TRACE_TENSORS:
        tensor = tensors[name]
        if _tensor_shape(tensor) != shapes[name]:
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"cache tensor {name} shape differs"
            )
        if getattr(tensor, "dtype", None) != torch.bfloat16:
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"cache tensor {name} dtype differs"
            )
        identity = id(tensor)
        if identity in identities:
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"cache tensor {name} aliases another capture"
            )
        identities.add(identity)
        _canonical_bf16_le_bytes(tensor, torch)


def _capture_profile_document() -> dict[str, object]:
    return {
        "capture_domain": "cache-on-p2048-prefill-dynamic-cache-kv-prefix",
        "id": TRACE_ID,
        "prefill_source_logit_row": 0,
        "prefill_token_count": oracle.PROMPT_TOKEN_COUNT,
        "selected_layer_indices": list(range(PREFIX_LAYER_COUNT)),
        "key_value_head_count": cache_free.MODEL_KEY_VALUE_HEAD_COUNT,
        "head_dimension": cache_free.MODEL_HEAD_DIMENSION,
        "tensor_count": len(TRACE_TENSORS),
        "hf_dynamic_cache_layout": "[batch,kv_head,token,head_dim]",
        "rust_consumer": {
            "api": (
                "riley_runtime::llama::PreparedLlamaDecode::"
                "download_hf_eager_qwen_p2048_cache_on_prefill_layer_prefix"
            ),
            "cache_layout": "contiguous-head-major-logical-[kv_head,token,head_dim]",
            "execution": "P2048 prefill then layer-prefix K/V download",
            "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
            "serving_eligibility": "none-until-cache-on-full-forward-is-bf16-exact",
        },
    }


def _execution_document() -> dict[str, object]:
    return dict(cache_on._execution_document())


def _source_logit_binding(source_row: bytes) -> dict[str, object]:
    if len(source_row) != oracle.RAW_LOGIT_BYTES:
        raise Qwen3BCacheOnPrefillKvTraceError("prefill source logit byte count differs")
    return {"source_logit_row": 0, "bf16_le_sha256": _sha256_bytes(source_row)}


class HuggingFaceQwen3BCacheOnPrefillKvTraceBackend:
    """Lazy CUDA-only adapter that copies one verified HF DynamicCache prefix."""

    def __init__(self, loader: oracle.HuggingFaceQwen3BBackend) -> None:
        self._loader: oracle.HuggingFaceQwen3BBackend | None = loader
        self._torch = loader._torch
        self._model = loader._model
        self._device = loader._device
        _base_model, layers, _module = stage._validate_topology(self._model)
        if len(layers) < PREFIX_LAYER_COUNT:
            raise Qwen3BCacheOnPrefillKvTraceError(
                "loaded model has fewer layers than the cache prefix"
            )
        self._layers = layers
        self.producer_metadata = dict(loader.producer_metadata)
        self.producer_metadata["implementation_id"] = IMPLEMENTATION_ID
        self.producer_metadata["transformers_qwen2_source"] = {
            "path": stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
            "sha256": stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
        }

    @classmethod
    def load(
        cls, *, checkpoint: oracle.CheckpointManifest, device: str
    ) -> "HuggingFaceQwen3BCacheOnPrefillKvTraceBackend":
        loader = oracle.HuggingFaceQwen3BBackend.load(
            checkpoint=checkpoint, device=device
        )
        try:
            return cls(loader)
        except BaseException:
            loader.close()
            raise

    def _cache_sequence_length(self, cache: object) -> int:
        try:
            return generation.HuggingFaceQwen3BGenerationTraceBackend._cache_sequence_length(
                cache
            )
        except generation.Qwen3BGenerationTraceError as error:
            raise _rethrow(error) from error

    @staticmethod
    def _first_tensor(output: object, label: str) -> object:
        if isinstance(output, tuple):
            if not output:
                raise Qwen3BCacheOnPrefillKvTraceError(
                    f"{label} returned an empty tuple"
                )
            return output[0]
        return output

    def _call_prefill(self, prompt_token_ids: Sequence[int]) -> object:
        model = self._model
        if model is None:
            raise Qwen3BCacheOnPrefillKvTraceError("trace backend is closed")
        torch = self._torch
        input_ids = attention_mask = position_ids = None
        try:
            input_ids = torch.tensor(
                [list(prompt_token_ids)], dtype=torch.long, device=self._device
            )
            attention_mask = torch.ones(
                (1, oracle.PROMPT_TOKEN_COUNT),
                dtype=torch.long,
                device=self._device,
            )
            position_ids = torch.arange(
                oracle.PROMPT_TOKEN_COUNT,
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)
            with torch.inference_mode():
                return model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
        finally:
            del input_ids, attention_mask, position_ids

    def _cache_layer_tensor(self, cache: object, layer_index: int, kind: str) -> object:
        try:
            layers = cache.layers
            layer = layers[layer_index]
            tensor = layer.keys if kind == "key" else layer.values
        except (AttributeError, IndexError, TypeError) as error:
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"DynamicCache layer {layer_index} {kind} is unavailable"
            ) from error
        expected = (
            1,
            cache_free.MODEL_KEY_VALUE_HEAD_COUNT,
            oracle.PROMPT_TOKEN_COUNT,
            cache_free.MODEL_HEAD_DIMENSION,
        )
        if _tensor_shape(tensor) != expected:
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"DynamicCache layer {layer_index} {kind} shape differs"
            )
        try:
            result = tensor[0].detach().to(device="cpu").contiguous()
        except (AttributeError, IndexError, RuntimeError, TypeError) as error:
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"DynamicCache layer {layer_index} {kind} cannot be copied"
            ) from error
        return result

    def capture(
        self, *, prompt_token_ids: Sequence[int], expected_source_logit: bytes
    ) -> CapturedTrace:
        expected_prompt = (oracle.EXPECTED_PROMPT_TOKEN_ID,) * oracle.PROMPT_TOKEN_COUNT
        if tuple(prompt_token_ids) != expected_prompt:
            raise Qwen3BCacheOnPrefillKvTraceError(
                "trace input differs from pinned P2048 prompt"
            )
        if len(expected_source_logit) != oracle.RAW_LOGIT_BYTES:
            raise Qwen3BCacheOnPrefillKvTraceError("prefill source logit byte count differs")
        output: object | None = None
        logits: object | None = None
        try:
            output = self._call_prefill(prompt_token_ids)
            cache = getattr(output, "past_key_values", None)
            if cache is None or self._cache_sequence_length(cache) != oracle.PROMPT_TOKEN_COUNT:
                raise Qwen3BCacheOnPrefillKvTraceError(
                    "prefill did not return the P2048 DynamicCache"
                )
            logits = self._first_tensor(getattr(output, "logits", None), "prefill logits")
            if logits is None:
                raise Qwen3BCacheOnPrefillKvTraceError("prefill logits are unavailable")
            try:
                prefill_last = logits[0, -1].detach().to(device="cpu").contiguous()
            except (AttributeError, IndexError, RuntimeError, TypeError) as error:
                raise Qwen3BCacheOnPrefillKvTraceError(
                    "prefill logits have no last-token row"
                ) from error
            if _canonical_bf16_le_bytes(prefill_last, self._torch) != expected_source_logit:
                raise Qwen3BCacheOnPrefillKvTraceError(
                    "captured prefill logits differ from source cache-on row"
                )
            captured = {
                _tensor_name(layer_index, kind): self._cache_layer_tensor(
                    cache, layer_index, kind
                )
                for layer_index in range(PREFIX_LAYER_COUNT)
                for kind in ("key", "value")
            }
            _validate_tensors(captured, self._torch)
            return CapturedTrace(tensors=captured)
        finally:
            if logits is not None:
                del logits
            if output is not None:
                del output
            self._torch.cuda.empty_cache()

    def close(self) -> None:
        loader = self._loader
        if loader is None:
            return
        self._loader = None
        self._model = None
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
        raise Qwen3BCacheOnPrefillKvTraceError("trace Git revision is malformed")
    if not isinstance(source["source_dirty"], bool):
        raise Qwen3BCacheOnPrefillKvTraceError("trace source dirty must be a boolean")
    _require_sha256(source["source_status_sha256"], "trace source status SHA-256")
    records = _require_mapping(source["sources"], "trace source records")
    if set(records) != set(SOURCE_PATHS):
        raise Qwen3BCacheOnPrefillKvTraceError("trace source records differ")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(records[name], f"trace source {name}")
        _require_exact_keys(record, {"path", "sha256"}, f"trace source {name}")
        if record["path"] != relative:
            raise Qwen3BCacheOnPrefillKvTraceError("trace source path differs")
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
        raise Qwen3BCacheOnPrefillKvTraceError(
            "cannot collect Git provenance"
        ) from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BCacheOnPrefillKvTraceError(
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
        raise Qwen3BCacheOnPrefillKvTraceError(
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
        raise Qwen3BCacheOnPrefillKvTraceError(
            "external source provenance is invalid"
        ) from error
    _validate_source_provenance(document)
    if document["source_dirty"] is not False or document["source_status_sha256"] != _sha256_bytes(b""):
        raise Qwen3BCacheOnPrefillKvTraceError(
            "external source provenance does not prove a clean pathspec"
        )
    expected = {name: _source_record(root, path) for name, path in SOURCE_PATHS.items()}
    if document["sources"] != expected:
        raise Qwen3BCacheOnPrefillKvTraceError(
            "external source provenance hashes differ"
        )
    return dict(document)


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Collect or validate named source hashes without importing ML dependencies."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    external = _load_external_source_provenance(root)
    if external is not None:
        return external
    return _collect_git_source_provenance(root)


def write_source_provenance_exclusive(
    *, repo_root: Path, output_path: Path
) -> dict[str, object]:
    """Write a clean host-Git provenance document for a pinned container."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    output = output_path.expanduser()
    if not output.is_absolute() or output.suffix != ".json":
        raise Qwen3BCacheOnPrefillKvTraceError(
            "source provenance output must be an absolute .json path"
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        parent = _regular_directory(output.parent, "source provenance output parent")
    except OSError as error:
        raise Qwen3BCacheOnPrefillKvTraceError(
            "cannot create source provenance output parent"
        ) from error
    output = parent / output.name
    if output == root or output.is_relative_to(root) or output.exists() or output.is_symlink():
        raise Qwen3BCacheOnPrefillKvTraceError(
            "refusing unsafe source provenance output"
        )
    document = _collect_git_source_provenance(root)
    if document["source_dirty"]:
        raise Qwen3BCacheOnPrefillKvTraceError(
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
        raise Qwen3BCacheOnPrefillKvTraceError(
            "cannot write source provenance output"
        ) from error
    return document


def _tensor_manifest(tensors: Mapping[str, object], torch: Any) -> dict[str, object]:
    _validate_tensors(tensors, torch)
    shapes = _expected_shapes()
    document: dict[str, object] = {}
    for name in TRACE_TENSORS:
        raw = _canonical_bf16_le_bytes(tensors[name], torch)
        shape = _tensor_shape(tensors[name])
        expected_bytes = _shape_element_count(shapes[name]) * BF16_BYTES
        if shape != shapes[name] or len(raw) != expected_bytes:
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"cache tensor {name} metadata differs"
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
        raise Qwen3BCacheOnPrefillKvTraceError("trace producer implementation differs")
    for field in ("runtime_dependency_class", "torch_version", "transformers_version"):
        _require_string(producer.get(field), f"trace producer {field}")
    source = _require_mapping(
        producer.get("transformers_qwen2_source"), "trace producer Qwen source"
    )
    if dict(source) != {
        "path": stage.TRANSFORMERS_QWEN2_SOURCE_PATH,
        "sha256": stage.TRANSFORMERS_QWEN2_SOURCE_SHA256,
    }:
        raise Qwen3BCacheOnPrefillKvTraceError("trace producer Qwen source differs")


def validate_manifest(document: Mapping[str, object]) -> None:
    """Validate the immutable cache-prefix contract without Torch."""

    _require_exact_keys(
        document,
        {
            "schema_version", "artifact_kind", "trace_id", "performance_claim_eligible",
            "created_at", "producer", "trace_profile", "contract", "model",
            "provenance", "sidecar", "tensors",
        },
        "trace manifest",
    )
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["artifact_kind"] != ARTIFACT_KIND
        or document["trace_id"] != TRACE_ID
        or document["performance_claim_eligible"] is not False
    ):
        raise Qwen3BCacheOnPrefillKvTraceError("trace manifest identity differs")
    try:
        oracle._validate_utc_text(document["created_at"], "trace created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise _rethrow(error) from error
    _validate_producer(document["producer"])
    if dict(_require_mapping(document["trace_profile"], "trace profile")) != _capture_profile_document():
        raise Qwen3BCacheOnPrefillKvTraceError("trace profile differs")
    contract = _require_mapping(document["contract"], "trace contract")
    _require_exact_keys(
        contract,
        {"model_id", "model_revision", "workload", "execution", "input", "source_logit_binding"},
        "trace contract",
    )
    if contract["model_id"] != oracle.MODEL_ID or contract["model_revision"] != oracle.MODEL_REVISION:
        raise Qwen3BCacheOnPrefillKvTraceError("trace model identity differs")
    try:
        cache_free._validate_workload_document(contract["workload"])
        cache_on._validate_teacher_input(contract["input"])
        cache_on._validate_model(document["model"])
    except (
        cache_free.Qwen3BP2051LayerStageTraceError,
        cache_on.Qwen3BCacheOnLayerStageTraceError,
    ) as error:
        raise _rethrow(error) from error
    if dict(_require_mapping(contract["execution"], "trace execution")) != _execution_document():
        raise Qwen3BCacheOnPrefillKvTraceError("trace execution differs")
    binding = _require_mapping(contract["source_logit_binding"], "trace source logit")
    _require_exact_keys(binding, {"source_logit_row", "bf16_le_sha256"}, "trace source logit")
    if binding["source_logit_row"] != 0:
        raise Qwen3BCacheOnPrefillKvTraceError("trace source logit row differs")
    _require_sha256(binding["bf16_le_sha256"], "trace source logit SHA-256")
    provenance = _require_mapping(document["provenance"], "trace provenance")
    _require_exact_keys(provenance, {"source_repository"}, "trace provenance")
    _validate_source_provenance(provenance["source_repository"])
    sidecar = _require_mapping(document["sidecar"], "trace sidecar")
    _require_exact_keys(sidecar, {"path", "sha256", "format", "tensor_count"}, "trace sidecar")
    sidecar_name = _require_string(sidecar["path"], "trace sidecar path")
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(".safetensors"):
        raise Qwen3BCacheOnPrefillKvTraceError("trace sidecar path differs")
    if sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(TRACE_TENSORS):
        raise Qwen3BCacheOnPrefillKvTraceError("trace sidecar metadata differs")
    _require_sha256(sidecar["sha256"], "trace sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "trace tensors")
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BCacheOnPrefillKvTraceError("trace tensor names differ")
    shapes = _expected_shapes()
    for name in TRACE_TENSORS:
        tensor = _require_mapping(tensors[name], f"trace tensor {name}")
        _require_exact_keys(
            tensor,
            {"key", "shape", "dtype", "canonical_byte_order", "bf16_le_sha256", "bf16_le_bytes"},
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
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"trace tensor {name} metadata differs"
            )
        _require_sha256(tensor["bf16_le_sha256"], f"trace tensor {name} SHA-256")


def validate_sidecar_against_manifest(
    manifest: Mapping[str, object], sidecar_path: Path
) -> None:
    """Replay all tensor hashes and contiguous Safetensors ranges."""

    validate_manifest(manifest)
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    metadata = _require_mapping(manifest["sidecar"], "trace sidecar")
    if sidecar.name != metadata["path"] or _sha256_file(sidecar) != metadata["sha256"]:
        raise Qwen3BCacheOnPrefillKvTraceError("trace sidecar binding differs")
    try:
        header, data_start, size = cache_free._read_safetensors_header(sidecar)
    except cache_free.Qwen3BP2051LayerStageTraceError as error:
        raise _rethrow(error) from error
    tensors = _require_mapping(manifest["tensors"], "trace tensors")
    expected_keys = {_sidecar_key(name) for name in TRACE_TENSORS}
    if set(header) - {"__metadata__"} != expected_keys:
        raise Qwen3BCacheOnPrefillKvTraceError("trace sidecar tensor set differs")
    ranges: list[tuple[int, int, str]] = []
    with sidecar.open("rb") as handle:
        for name in TRACE_TENSORS:
            reference = _require_mapping(tensors[name], f"trace tensor {name}")
            entry = _require_mapping(header[reference["key"]], f"sidecar tensor {name}")
            _require_exact_keys(entry, {"dtype", "shape", "data_offsets"}, f"sidecar tensor {name}")
            if entry["dtype"] != "BF16" or _shape_from_document(
                entry["shape"], f"sidecar tensor {name} shape"
            ) != _shape_from_document(reference["shape"], f"trace tensor {name} shape"):
                raise Qwen3BCacheOnPrefillKvTraceError(
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
                raise Qwen3BCacheOnPrefillKvTraceError(
                    f"sidecar tensor {name} offsets differ"
                )
            start, end = offsets
            expected_bytes = reference["bf16_le_bytes"]
            if end - start != expected_bytes or data_start + end > size:
                raise Qwen3BCacheOnPrefillKvTraceError(
                    f"sidecar tensor {name} byte range differs"
                )
            handle.seek(data_start + start)
            raw = handle.read(expected_bytes)
            if len(raw) != expected_bytes or _sha256_bytes(raw) != reference["bf16_le_sha256"]:
                raise Qwen3BCacheOnPrefillKvTraceError(
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
            raise Qwen3BCacheOnPrefillKvTraceError(
                f"sidecar tensor {name} offsets are non-contiguous"
            )
        expected_start = end
    if data_start + expected_start != size:
        raise Qwen3BCacheOnPrefillKvTraceError("trace sidecar trailing bytes differ")


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
        raise Qwen3BCacheOnPrefillKvTraceError("trace manifest is invalid JSON") from error
    validate_manifest(document)
    if raw != oracle._canonical_json_bytes(document):
        raise Qwen3BCacheOnPrefillKvTraceError("trace manifest JSON is not canonical")
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
        raise Qwen3BCacheOnPrefillKvTraceError("trace sidecar offsets differ")
    start, end = offsets
    with sidecar_path.open("rb") as handle:
        handle.seek(data_start + start)
        raw = handle.read(end - start)
    if len(raw) != end - start:
        raise Qwen3BCacheOnPrefillKvTraceError("trace sidecar tensor is truncated")
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
    source_logit: bytes,
    sidecar_name: str,
    sidecar_sha256: str,
    created_at: datetime,
) -> dict[str, object]:
    if Path(sidecar_name).name != sidecar_name or not sidecar_name.endswith(".safetensors"):
        raise Qwen3BCacheOnPrefillKvTraceError("sidecar name differs")
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
            "source_logit_binding": _source_logit_binding(source_logit),
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
    backend_factory: BackendFactory = HuggingFaceQwen3BCacheOnPrefillKvTraceBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Write a create-only trace after validating the pinned HF prefill row."""

    try:
        manifest, sidecar = cache_free._output_paths(manifest_path, sidecar_path, repo_root)
        workload = oracle.load_workload(workload_path)
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except (
        cache_free.Qwen3BP2051LayerStageTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        raise _rethrow(error) from error
    teacher = cache_on._load_teacher_stream(
        teacher_manifest_path=teacher_manifest_path,
        teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path,
    )
    source_rows = cache_on._load_source_logit_rows(
        teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path
    )
    source_logit = source_rows.get(0)
    if source_logit is None:
        raise Qwen3BCacheOnPrefillKvTraceError("teacher prefill source row is missing")
    provenance = source_provenance_factory(repo_root)
    backend = backend_factory(checkpoint=checkpoint, device=device)
    sidecar_written = False
    try:
        captured = backend.capture(
            prompt_token_ids=workload.prompt_token_ids,
            expected_source_logit=source_logit,
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
            source_logit=source_logit,
            sidecar_name=sidecar.name,
            sidecar_sha256=_sha256_file(sidecar),
            created_at=created_at or datetime.now(timezone.utc),
        )
        validate_sidecar_against_manifest(document, sidecar)
        if _read_trace_tensor_raw(
            manifest=document, sidecar_path=sidecar, name=TRACE_TENSORS[0]
        ) == b"":
            raise Qwen3BCacheOnPrefillKvTraceError("trace sidecar first tensor is empty")
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
    """Replay source, teacher, sidecar, and prefill-logit bindings without CUDA."""

    manifest = _load_manifest(manifest_path)
    root = _regular_directory(repo_root.expanduser(), "repository root")
    try:
        workload = oracle.load_workload(workload_path)
    except oracle.Qwen3BServingOracleError as error:
        raise _rethrow(error) from error
    teacher = cache_on._load_teacher_stream(
        teacher_manifest_path=teacher_manifest_path,
        teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path,
    )
    contract = _require_mapping(manifest["contract"], "trace contract")
    if contract["workload"] != cache_free._workload_document(workload):
        raise Qwen3BCacheOnPrefillKvTraceError("trace workload binding differs")
    if contract["input"] != cache_on._teacher_input_document(workload, teacher):
        raise Qwen3BCacheOnPrefillKvTraceError("trace teacher input binding differs")
    source_rows = cache_on._load_source_logit_rows(
        teacher_cache_on_sidecar_path=teacher_cache_on_sidecar_path
    )
    if contract["source_logit_binding"] != _source_logit_binding(source_rows[0]):
        raise Qwen3BCacheOnPrefillKvTraceError("trace prefill source logit binding differs")
    observed_sources = _require_mapping(
        _require_mapping(manifest["provenance"], "trace provenance")["source_repository"],
        "trace source provenance",
    )
    expected_sources = collect_source_provenance(root)
    if observed_sources["sources"] != expected_sources["sources"]:
        raise Qwen3BCacheOnPrefillKvTraceError("trace source hashes differ")
    sidecar = _regular_file(sidecar_path.expanduser(), "trace sidecar")
    validate_sidecar_against_manifest(manifest, sidecar)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_cache_on_prefill_kv_trace",
        description="offline Qwen2.5-3B P2048 HF DynamicCache K/V prefix trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser("produce", help="write a create-only BF16 cache trace")
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--workload", type=Path, required=True)
    produce.add_argument("--teacher-manifest", type=Path, required=True)
    produce.add_argument("--teacher-cache-on-sidecar", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--sidecar", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    provenance = commands.add_parser("provenance", help="write host-Git source provenance")
    provenance.add_argument("--repo-root", type=Path, required=True)
    provenance.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate source and artifact bindings")
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
            f"{document['sidecar']['path']} sha256={document['sidecar']['sha256']}"
        )
        return 0
    except (
        OSError,
        Qwen3BCacheOnPrefillKvTraceError,
        cache_on.Qwen3BCacheOnLayerStageTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
