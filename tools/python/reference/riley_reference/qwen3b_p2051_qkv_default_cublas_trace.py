"""Offline P2051 raw-Q/K/V default-cuBLAS arithmetic trace.

This producer creates an immutable, create-only sidecar for the layer-zero
Q/K/V *unbiased* projections at P2051.  It exists to qualify the arithmetic
contract that a native direct-cuBLAS candidate must reproduce before any
serving selector can use it.  Python and Hugging Face run only while making or
checking the offline artifact; Riley's serving hot path never imports this
module or calls Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import qwen3b_p2051_projection_trace as projection
from . import qwen3b_serving_oracle as oracle
from . import qwen3b_stage_trace as stage
from .hf_calibration import SidecarWriter, _default_sidecar_writer, _write_sidecar_exclusive

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2051-qkv-default-cublas-trace.v1"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2051-raw-qkv-default-cublas-trace"
TRACE_ID = "qwen3b-p2051-layer0-raw-qkv-default-cublas-v1"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-p2051-qkv-default-cublas-trace-v1"
BF16_BYTES = 2
M = projection.CONTEXT_TOKEN_COUNT
K = projection.MODEL_HIDDEN_SIZE
Q_WIDTH = projection.MODEL_HIDDEN_SIZE
KV_WIDTH = projection.MODEL_KEY_VALUE_HEAD_COUNT * projection.MODEL_HEAD_DIMENSION

P7_INPUT_KEY = "trace/layer0/input_norm"
P7_RAW_Q_KEY = "trace/layer0/q_proj/unbiased_linear"
P7_INPUT_NAME = "p7_input_norm"
P7_RAW_Q_NAME = "p7_shadow_raw_q"


@dataclass(frozen=True)
class Projection:
    """One fixed layer-zero no-bias projection in the P11 contract."""

    identifier: str
    module_name: str
    width: int
    weight_name: str
    output_name: str
    checkpoint_weight_key: str


PROJECTIONS = (
    Projection(
        identifier="q",
        module_name="q_proj",
        width=Q_WIDTH,
        weight_name="layer0_q_proj_weight",
        output_name="raw_q/default_cublas_reduced_splitk",
        checkpoint_weight_key="model.layers.0.self_attn.q_proj.weight",
    ),
    Projection(
        identifier="k",
        module_name="k_proj",
        width=KV_WIDTH,
        weight_name="layer0_k_proj_weight",
        output_name="raw_k/default_cublas_reduced_splitk",
        checkpoint_weight_key="model.layers.0.self_attn.k_proj.weight",
    ),
    Projection(
        identifier="v",
        module_name="v_proj",
        width=KV_WIDTH,
        weight_name="layer0_v_proj_weight",
        output_name="raw_v/default_cublas_reduced_splitk",
        checkpoint_weight_key="model.layers.0.self_attn.v_proj.weight",
    ),
)
TRACE_TENSORS = (
    P7_INPUT_NAME,
    P7_RAW_Q_NAME,
    *(name for projection_item in PROJECTIONS for name in (projection_item.weight_name, projection_item.output_name)),
)
SOURCE_PATHS = {
    "p11_qkv_default_cublas_trace": "tools/python/reference/riley_reference/qwen3b_p2051_qkv_default_cublas_trace.py",
    "p7_projection_trace": "tools/python/reference/riley_reference/qwen3b_p2051_projection_trace.py",
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "stage_trace_support": "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}

BackendFactory = Callable[..., oracle.HuggingFaceQwen3BBackend]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


class Qwen3BP2051QkvDefaultCublasTraceError(RuntimeError):
    """Raised when the fixed P11 raw-Q/K/V trace contract is violated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return oracle._sha256_file(path)


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} must be a directory")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"cannot resolve {label}: {path}") from error


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} must be a regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"cannot resolve {label}: {path}") from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise Qwen3BP2051QkvDefaultCublasTraceError(
            f"{label} fields differ: missing={sorted(expected - set(value))!r} "
            f"extra={sorted(set(value) - expected)!r}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    try:
        return oracle._require_sha256(value, label)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(str(error)) from error


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise Qwen3BP2051QkvDefaultCublasTraceError("created_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sidecar_key(name: str) -> str:
    return f"trace/{name}"


def _expected_shapes() -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {
        P7_INPUT_NAME: (M, K),
        P7_RAW_Q_NAME: (M, Q_WIDTH),
    }
    for projection_item in PROJECTIONS:
        result[projection_item.weight_name] = (projection_item.width, K)
        result[projection_item.output_name] = (M, projection_item.width)
    if tuple(result) != TRACE_TENSORS:
        raise AssertionError("P11 tensor ordering differs")
    return result


def _element_count(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise Qwen3BP2051QkvDefaultCublasTraceError("tensor shape dimensions must be positive integers")
        count *= dimension
    return count


def _default_policy_document() -> dict[str, object]:
    return {
        "id": "default_cublas_reduced_splitk",
        "preferred_blas_requested": "cublas",
        "allow_bf16_reduced_precision_reduction_requested": True,
        "allow_bf16_reduced_precision_reduction_split_k_requested": True,
        "matmul_tf32_requested": False,
        "cudnn_tf32_requested": False,
        "operator": "torch.nn.functional.linear",
        "bias": False,
    }


def _source_record(root: Path, relative: str) -> dict[str, object]:
    path = _regular_file(root / relative, f"source file {relative}")
    return {"path": relative, "sha256": _sha256_file(path)}


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Pin tracked producer inputs without rejecting unrelated checkout state."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_PATHS.values()],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        for relative in SOURCE_PATHS.values():
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=root,
                check=True,
                capture_output=True,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("cannot record P11 Git provenance") from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 Git revision is malformed")
    if status:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 producer source must be clean")
    return {
        "git_revision": revision,
        "source_dirty": False,
        "source_status_sha256": _sha256_bytes(status),
        "sources": {name: _source_record(root, path) for name, path in SOURCE_PATHS.items()},
    }


def _output_paths(manifest_path: Path, sidecar_path: Path, repo_root: Path) -> tuple[Path, Path]:
    root = _regular_directory(repo_root.expanduser(), "repository root")
    manifest = manifest_path.expanduser()
    sidecar = sidecar_path.expanduser()
    if not manifest.is_absolute() or not sidecar.is_absolute():
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 output paths must be absolute")
    if manifest.parent != sidecar.parent:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 manifest and sidecar must be siblings")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(manifest.parent, "P11 output directory")
    manifest = parent / manifest.name
    sidecar = parent / sidecar.name
    if manifest.suffix != ".json" or sidecar.suffix != ".safetensors" or manifest == sidecar:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 output extensions differ")
    for output in (manifest, sidecar):
        if output == root or root in output.parents:
            raise Qwen3BP2051QkvDefaultCublasTraceError("P11 artifacts must be outside the repository")
        if output.exists() or output.is_symlink():
            raise Qwen3BP2051QkvDefaultCublasTraceError(f"refusing to overwrite P11 artifact: {output}")
    return manifest, sidecar


def _canonical_bf16_le_bytes(tensor: object, torch: Any, label: str) -> bytes:
    try:
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"cannot read {label} as BF16") from error
    if sys.byteorder == "little":
        return raw
    if sys.byteorder == "big":
        return b"".join(raw[offset : offset + 2][::-1] for offset in range(0, len(raw), 2))
    raise Qwen3BP2051QkvDefaultCublasTraceError("unsupported host byte order for BF16")


def _validate_tensor(tensor: object, shape: tuple[int, int], torch: Any, label: str) -> bytes:
    try:
        actual_shape = tuple(tensor.shape)
        dtype = tensor.dtype
        contiguous = tensor.is_contiguous()
    except AttributeError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} is not a tensor") from error
    if actual_shape != shape or dtype != torch.bfloat16 or not contiguous:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} tensor contract differs")
    raw = _canonical_bf16_le_bytes(tensor, torch, label)
    expected_bytes = _element_count(shape) * BF16_BYTES
    if len(raw) != expected_bytes:
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} BF16 byte count differs")
    try:
        projection._validate_finite_bf16(raw, label)
    except projection.Qwen3BP2051ProjectionTraceError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(str(error)) from error
    return raw


def _metrics(expected: bytes, actual: bytes, torch: Any) -> dict[str, object]:
    if len(expected) != len(actual) or len(expected) % BF16_BYTES:
        raise Qwen3BP2051QkvDefaultCublasTraceError("comparison BF16 byte lengths differ")
    expected_u16 = torch.frombuffer(bytearray(expected), dtype=torch.uint16)
    actual_u16 = torch.frombuffer(bytearray(actual), dtype=torch.uint16)
    unequal = int((expected_u16 != actual_u16).sum().item())
    expected_f32 = expected_u16.view(torch.bfloat16).to(torch.float32)
    actual_f32 = actual_u16.view(torch.bfloat16).to(torch.float32)
    return {
        "bf16_exact": expected == actual,
        "unequal_elements": unequal,
        "total_elements": int(expected_u16.numel()),
        "max_abs": float((expected_f32 - actual_f32).abs().max().item()),
    }


def _backend_name(torch: Any) -> str:
    try:
        observed = str(torch.backends.cuda.preferred_blas_library())
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("PyTorch preferred BLAS API is unavailable") from error
    if observed.endswith("Cublas"):
        return "cublas"
    if observed.endswith("Cublaslt"):
        return "cublaslt"
    raise Qwen3BP2051QkvDefaultCublasTraceError(f"unsupported preferred BLAS backend: {observed}")


def _policy_readback(torch: Any) -> tuple[bool, bool]:
    try:
        return (
            bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k),
        )
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("PyTorch BF16 policy API is unavailable") from error


def _tf32_readback(torch: Any) -> tuple[bool, bool]:
    try:
        return bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32)
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("PyTorch TF32 policy API is unavailable") from error


def _apply_default_policy(torch: Any) -> dict[str, object]:
    try:
        torch.backends.cuda.preferred_blas_library("cublas")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (True, True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("cannot configure P11 default Cublas policy") from error
    backend = _backend_name(torch)
    reduced, split_k = _policy_readback(torch)
    matmul_tf32, cudnn_tf32 = _tf32_readback(torch)
    if backend != "cublas" or (reduced, split_k) != (True, True) or matmul_tf32 or cudnn_tf32:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 default Cublas policy readback differs")
    return {
        "preferred_blas_actual": backend,
        "allow_bf16_reduced_precision_reduction_actual": reduced,
        "allow_bf16_reduced_precision_reduction_split_k_actual": split_k,
        "matmul_tf32_actual": matmul_tf32,
        "cudnn_tf32_actual": cudnn_tf32,
    }


def _restore_policy(
    torch: Any, backend: str, flags: tuple[bool, bool], tf32_flags: tuple[bool, bool]
) -> None:
    try:
        torch.backends.cuda.preferred_blas_library(backend)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            bool(flags[0]),
            bool(flags[1]),
        )
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32_flags[0])
        torch.backends.cudnn.allow_tf32 = bool(tf32_flags[1])
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("cannot restore PyTorch BF16 policy") from error
    if _backend_name(torch) != backend or _policy_readback(torch) != flags or _tf32_readback(torch) != tf32_flags:
        raise Qwen3BP2051QkvDefaultCublasTraceError("restored PyTorch BF16 policy differs")


def _load_p7_artifact(p7_manifest_path: Path, p7_sidecar_path: Path) -> tuple[dict[str, object], Path, Path]:
    manifest_path = _regular_file(p7_manifest_path.expanduser(), "P7 manifest")
    sidecar_path = _regular_file(p7_sidecar_path.expanduser(), "P7 sidecar")
    try:
        document = projection._load_manifest(manifest_path)
        projection.validate_manifest(document)
        projection.validate_sidecar_against_manifest(document, sidecar_path)
    except projection.Qwen3BP2051ProjectionTraceError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(str(error)) from error
    expected_sidecar = _require_mapping(document["sidecar"], "P7 sidecar")
    if _sha256_file(sidecar_path) != expected_sidecar["sha256"]:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P7 sidecar SHA-256 differs")
    return document, manifest_path, sidecar_path


def _checkpoint_weight_raw(
    checkpoint: oracle.CheckpointManifest, projection_item: Projection
) -> bytes:
    """Read one BF16 Q/K/V weight directly from the indexed checkpoint."""

    index_path = _regular_file(checkpoint.root / "model.safetensors.index.json", "checkpoint safetensors index")
    try:
        index = _require_mapping(
            json.loads(index_path.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_key),
            "checkpoint safetensors index",
        )
        weight_map = _require_mapping(index["weight_map"], "checkpoint weight map")
        shard_name = _require_string(
            weight_map[projection_item.checkpoint_weight_key],
            f"checkpoint {projection_item.identifier.upper()} weight shard",
        )
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(
            f"cannot resolve checkpoint {projection_item.identifier.upper()} weight shard"
        ) from error
    if Path(shard_name).name != shard_name or not shard_name.endswith(".safetensors"):
        raise Qwen3BP2051QkvDefaultCublasTraceError("checkpoint weight shard name differs")
    shard = _regular_file(checkpoint.root / shard_name, "checkpoint weight shard")
    expected_bytes = projection_item.width * K * BF16_BYTES
    try:
        header, payload_offset, file_size = projection._read_safetensors_header(shard)
        metadata = _require_mapping(
            header[projection_item.checkpoint_weight_key],
            f"checkpoint {projection_item.identifier.upper()} weight",
        )
        _require_exact_keys(metadata, {"dtype", "shape", "data_offsets"}, "checkpoint weight")
        offsets = metadata["data_offsets"]
        if (
            metadata["dtype"] != "BF16"
            or metadata["shape"] != [projection_item.width, K]
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[1] != offsets[0] + expected_bytes
            or payload_offset + offsets[1] > file_size
        ):
            raise Qwen3BP2051QkvDefaultCublasTraceError("checkpoint weight layout differs")
        with shard.open("rb") as stream:
            stream.seek(payload_offset + offsets[0])
            raw = stream.read(expected_bytes)
    except (OSError, KeyError, projection.Qwen3BP2051ProjectionTraceError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("cannot read checkpoint weight") from error
    if len(raw) != expected_bytes:
        raise Qwen3BP2051QkvDefaultCublasTraceError("checkpoint weight byte length differs")
    try:
        projection._validate_finite_bf16(raw, "checkpoint weight")
    except projection.Qwen3BP2051ProjectionTraceError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(str(error)) from error
    return raw


def _load_p7_tensors(sidecar: Path, torch: Any) -> tuple[object, object]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("pinned safetensors is required for P11 produce") from error
    try:
        with safe_open(str(sidecar), framework="pt", device="cpu") as source:
            input_norm = source.get_tensor(P7_INPUT_KEY)
            raw_q = source.get_tensor(P7_RAW_Q_KEY)
    except (OSError, RuntimeError, ValueError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("cannot load P7 raw-Q tensors") from error
    _validate_tensor(input_norm, (M, K), torch, "P7 input norm")
    _validate_tensor(raw_q, (M, Q_WIDTH), torch, "P7 shadow raw Q")
    return input_norm, raw_q


def _tensor_manifest(tensors: Mapping[str, object], torch: Any) -> dict[str, object]:
    records: dict[str, object] = {}
    for name, shape in _expected_shapes().items():
        raw = _validate_tensor(tensors[name], shape, torch, name)
        records[name] = {
            "key": _sidecar_key(name),
            "shape": list(shape),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_bytes": len(raw),
            "bf16_le_sha256": _sha256_bytes(raw),
        }
    return records


def _producer_document(backend: oracle.HuggingFaceQwen3BBackend, torch: Any) -> dict[str, object]:
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "runtime_dependency_class": "offline-python-reference",
        "torch_version": str(torch.__version__),
        "runtime_cuda_version": str(torch.version.cuda or "unknown"),
        "model_loader": dict(backend.producer_metadata),
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32_enabled": bool(torch.backends.cudnn.allow_tf32),
    }


def _p7_binding_document(
    p7_document: Mapping[str, object], p7_manifest: Path, p7_sidecar: Path
) -> dict[str, object]:
    p7_model = _require_mapping(p7_document["model"], "P7 model")
    p7_provenance = _require_mapping(p7_document["provenance"], "P7 provenance")
    source = _require_mapping(p7_provenance["source_repository"], "P7 source provenance")
    p7_tensors = _require_mapping(p7_document["tensors"], "P7 tensors")
    p7_input = _require_mapping(p7_tensors["layer0.input_norm"], "P7 input tensor")
    p7_raw_q = _require_mapping(p7_tensors["layer0.q_proj.unbiased_linear"], "P7 raw-Q tensor")
    return {
        "manifest_filename": p7_manifest.name,
        "manifest_sha256": _sha256_file(p7_manifest),
        "sidecar_filename": p7_sidecar.name,
        "sidecar_sha256": _sha256_file(p7_sidecar),
        "checkpoint_receipt_sha256": p7_model["checkpoint_receipt_sha256"],
        "source_revision": source["git_revision"],
        "input_tensor_key": P7_INPUT_KEY,
        "raw_q_tensor_key": P7_RAW_Q_KEY,
        "input_bf16_le_sha256": p7_input["bf16_le_sha256"],
        "raw_q_bf16_le_sha256": p7_raw_q["bf16_le_sha256"],
    }


def _projection_result_document(
    projection_item: Projection,
    output_raw: bytes,
    repeated_raw: bytes,
    p7_raw_q: bytes,
    torch: Any,
) -> dict[str, object]:
    p7_comparison = _metrics(p7_raw_q, output_raw, torch) if projection_item.identifier == "q" else None
    return {
        "identifier": projection_item.identifier,
        "module_name": projection_item.module_name,
        "weight_tensor": projection_item.weight_name,
        "output_tensor": projection_item.output_name,
        "checkpoint_weight_key": projection_item.checkpoint_weight_key,
        "shape": {"m": M, "n": projection_item.width, "k": K},
        "output_bf16_le_sha256": _sha256_bytes(output_raw),
        "repeated_bf16_exact": output_raw == repeated_raw,
        "p7_default_raw_q_comparison": p7_comparison,
    }


def build_manifest(
    *,
    checkpoint: oracle.CheckpointManifest,
    p7_document: Mapping[str, object],
    p7_manifest: Path,
    p7_sidecar: Path,
    sidecar: Path,
    tensors: Mapping[str, object],
    torch: Any,
    producer: Mapping[str, object],
    policy_actual: Mapping[str, object],
    projection_results: Mapping[str, Mapping[str, object]],
    source_provenance: Mapping[str, object],
    created_at: datetime,
) -> dict[str, object]:
    tensor_records = _tensor_manifest(tensors, torch)
    q_result = _require_mapping(projection_results["q"], "P11 Q projection result")
    q_comparison = _require_mapping(q_result["p7_default_raw_q_comparison"], "P11 Q/P7 comparison")
    all_repeated = all(
        _require_mapping(projection_results[item.identifier], "P11 projection result")["repeated_bf16_exact"] is True
        for item in PROJECTIONS
    )
    quality_pass = bool(all_repeated and q_comparison["bf16_exact"] is True)
    policy = _default_policy_document()
    policy.update(policy_actual)
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "trace_id": TRACE_ID,
        "performance_claim_eligible": False,
        "vllm_comparison_eligible": False,
        "serving_selector_changed": False,
        "capture_status": "captured",
        "quality_pass": quality_pass,
        "created_at": _utc_text(created_at),
        "scope": {
            "endpoints": ["layer0.q_proj.raw_no_bias", "layer0.k_proj.raw_no_bias", "layer0.v_proj.raw_no_bias"],
            "operator": "torch.nn.functional.linear",
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "shape": {"m": M, "k": K, "q_n": Q_WIDTH, "kv_n": KV_WIDTH},
            "offline_only": True,
            "serving_path": False,
        },
        "producer": dict(producer),
        "default_policy": policy,
        "p7_binding": _p7_binding_document(p7_document, p7_manifest, p7_sidecar),
        "model": {
            "checkpoint_path": str(checkpoint.root),
            "checkpoint_receipt_filename": checkpoint.receipt.path,
            "checkpoint_receipt_sha256": checkpoint.receipt.sha256,
        },
        "projection_results": {item.identifier: dict(projection_results[item.identifier]) for item in PROJECTIONS},
        "comparisons": {
            "q_default_vs_p7_shadow_raw_q": dict(q_comparison),
            "outputs_are_not_riley_results": True,
        },
        "provenance": {"source_repository": dict(source_provenance)},
        "sidecar": {
            "path": sidecar.name,
            "sha256": _sha256_file(sidecar),
            "format": "safetensors",
            "tensor_count": len(TRACE_TENSORS),
        },
        "tensors": tensor_records,
    }
    validate_manifest(document)
    return document


def _duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise Qwen3BP2051QkvDefaultCublasTraceError(f"JSON object repeats key {key!r}")
        value[key] = item
    return value


def _nonfinite(value: str) -> None:
    raise Qwen3BP2051QkvDefaultCublasTraceError(f"non-finite JSON constant {value!r} is forbidden")


def _load_manifest(path: Path) -> dict[str, object]:
    source = _regular_file(path, "P11 manifest")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_key, parse_constant=_nonfinite
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("cannot parse P11 manifest") from error
    return dict(_require_mapping(value, "P11 manifest"))


def _validate_metric(value: object, label: str, total_elements: int) -> None:
    metric = _require_mapping(value, label)
    _require_exact_keys(metric, {"bf16_exact", "unequal_elements", "total_elements", "max_abs"}, label)
    if (
        not isinstance(metric["bf16_exact"], bool)
        or isinstance(metric["unequal_elements"], bool)
        or not isinstance(metric["unequal_elements"], int)
        or metric["unequal_elements"] < 0
        or metric["unequal_elements"] > total_elements
        or metric["total_elements"] != total_elements
        or not isinstance(metric["max_abs"], (int, float))
        or isinstance(metric["max_abs"], bool)
        or metric["max_abs"] < 0
    ):
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} differs")
    if metric["bf16_exact"] is True and (metric["unequal_elements"] != 0 or metric["max_abs"] != 0):
        raise Qwen3BP2051QkvDefaultCublasTraceError(f"{label} exactness differs")


def _validate_source_provenance(value: object) -> None:
    source = _require_mapping(value, "P11 source provenance")
    _require_exact_keys(source, {"git_revision", "source_dirty", "source_status_sha256", "sources"}, "P11 source provenance")
    if source["source_dirty"] is not False or oracle.GIT_REVISION_RE.fullmatch(_require_string(source["git_revision"], "P11 Git revision")) is None:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 source provenance differs")
    _require_sha256(source["source_status_sha256"], "P11 source status SHA-256")
    sources = _require_mapping(source["sources"], "P11 source records")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 source files differ")
    for name, path in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], "P11 source record")
        _require_exact_keys(record, {"path", "sha256"}, "P11 source record")
        if record["path"] != path:
            raise Qwen3BP2051QkvDefaultCublasTraceError("P11 source path differs")
        _require_sha256(record["sha256"], "P11 source SHA-256")


def validate_manifest(document: Mapping[str, object]) -> None:
    _require_exact_keys(
        document,
        {
            "schema_version", "artifact_kind", "trace_id", "performance_claim_eligible", "vllm_comparison_eligible",
            "serving_selector_changed", "capture_status", "quality_pass", "created_at", "scope", "producer",
            "default_policy", "p7_binding", "model", "projection_results", "comparisons", "provenance", "sidecar", "tensors",
        },
        "P11 manifest",
    )
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["artifact_kind"] != ARTIFACT_KIND
        or document["trace_id"] != TRACE_ID
        or document["performance_claim_eligible"] is not False
        or document["vllm_comparison_eligible"] is not False
        or document["serving_selector_changed"] is not False
        or document["capture_status"] != "captured"
        or not isinstance(document["quality_pass"], bool)
    ):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 manifest identity differs")
    try:
        oracle._validate_utc_text(document["created_at"], "P11 created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(str(error)) from error
    scope = _require_mapping(document["scope"], "P11 scope")
    _require_exact_keys(scope, {"endpoints", "operator", "model_id", "model_revision", "shape", "offline_only", "serving_path"}, "P11 scope")
    if (
        scope["endpoints"] != ["layer0.q_proj.raw_no_bias", "layer0.k_proj.raw_no_bias", "layer0.v_proj.raw_no_bias"]
        or scope["operator"] != "torch.nn.functional.linear"
        or scope["model_id"] != oracle.MODEL_ID
        or scope["model_revision"] != oracle.MODEL_REVISION
        or scope["shape"] != {"m": M, "k": K, "q_n": Q_WIDTH, "kv_n": KV_WIDTH}
        or scope["offline_only"] is not True
        or scope["serving_path"] is not False
    ):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 scope differs")
    producer = _require_mapping(document["producer"], "P11 producer")
    _require_exact_keys(producer, {"implementation_id", "runtime_dependency_class", "torch_version", "runtime_cuda_version", "model_loader", "tf32_enabled", "cudnn_tf32_enabled"}, "P11 producer")
    if (
        producer["implementation_id"] != IMPLEMENTATION_ID
        or producer["runtime_dependency_class"] != "offline-python-reference"
        or producer["tf32_enabled"] is not False
        or producer["cudnn_tf32_enabled"] is not False
    ):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 producer differs")
    _require_string(producer["torch_version"], "P11 torch version")
    _require_string(producer["runtime_cuda_version"], "P11 CUDA runtime version")
    _require_mapping(producer["model_loader"], "P11 model loader")
    policy = _require_mapping(document["default_policy"], "P11 default policy")
    expected_policy = _default_policy_document()
    expected_policy.update(
        {
            "preferred_blas_actual": "cublas",
            "allow_bf16_reduced_precision_reduction_actual": True,
            "allow_bf16_reduced_precision_reduction_split_k_actual": True,
            "matmul_tf32_actual": False,
            "cudnn_tf32_actual": False,
        }
    )
    _require_exact_keys(policy, set(expected_policy), "P11 default policy")
    if dict(policy) != expected_policy:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 default policy differs")
    binding = _require_mapping(document["p7_binding"], "P11 P7 binding")
    _require_exact_keys(binding, {"manifest_filename", "manifest_sha256", "sidecar_filename", "sidecar_sha256", "checkpoint_receipt_sha256", "source_revision", "input_tensor_key", "raw_q_tensor_key", "input_bf16_le_sha256", "raw_q_bf16_le_sha256"}, "P11 P7 binding")
    if binding["input_tensor_key"] != P7_INPUT_KEY or binding["raw_q_tensor_key"] != P7_RAW_Q_KEY:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 P7 binding differs")
    for name in ("manifest_sha256", "sidecar_sha256", "checkpoint_receipt_sha256", "input_bf16_le_sha256", "raw_q_bf16_le_sha256"):
        _require_sha256(binding[name], f"P11 P7 {name}")
    _require_string(binding["manifest_filename"], "P11 P7 manifest filename")
    _require_string(binding["sidecar_filename"], "P11 P7 sidecar filename")
    if oracle.GIT_REVISION_RE.fullmatch(_require_string(binding["source_revision"], "P11 P7 source revision")) is None:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 P7 source revision differs")
    model = _require_mapping(document["model"], "P11 model")
    _require_exact_keys(model, {"checkpoint_path", "checkpoint_receipt_filename", "checkpoint_receipt_sha256"}, "P11 model")
    _require_string(model["checkpoint_path"], "P11 checkpoint path")
    if model["checkpoint_receipt_filename"] != oracle.CHECKPOINT_RECEIPT_FILENAME:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 checkpoint receipt filename differs")
    _require_sha256(model["checkpoint_receipt_sha256"], "P11 checkpoint receipt SHA-256")
    results = _require_mapping(document["projection_results"], "P11 projection results")
    if set(results) != {item.identifier for item in PROJECTIONS}:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 projection result set differs")
    q_comparison: Mapping[str, object] | None = None
    all_repeated = True
    for item in PROJECTIONS:
        result = _require_mapping(results[item.identifier], "P11 projection result")
        _require_exact_keys(result, {"identifier", "module_name", "weight_tensor", "output_tensor", "checkpoint_weight_key", "shape", "output_bf16_le_sha256", "repeated_bf16_exact", "p7_default_raw_q_comparison"}, "P11 projection result")
        if (
            result["identifier"] != item.identifier
            or result["module_name"] != item.module_name
            or result["weight_tensor"] != item.weight_name
            or result["output_tensor"] != item.output_name
            or result["checkpoint_weight_key"] != item.checkpoint_weight_key
            or result["shape"] != {"m": M, "n": item.width, "k": K}
            or not isinstance(result["repeated_bf16_exact"], bool)
        ):
            raise Qwen3BP2051QkvDefaultCublasTraceError("P11 projection result differs")
        _require_sha256(result["output_bf16_le_sha256"], "P11 output SHA-256")
        all_repeated = all_repeated and result["repeated_bf16_exact"] is True
        comparison = result["p7_default_raw_q_comparison"]
        if item.identifier == "q":
            _validate_metric(comparison, "P11 Q/P7 comparison", M * Q_WIDTH)
            q_comparison = _require_mapping(comparison, "P11 Q/P7 comparison")
        elif comparison is not None:
            raise Qwen3BP2051QkvDefaultCublasTraceError("P11 K/V must not claim a P7 raw-Q comparison")
    comparisons = _require_mapping(document["comparisons"], "P11 comparisons")
    _require_exact_keys(comparisons, {"q_default_vs_p7_shadow_raw_q", "outputs_are_not_riley_results"}, "P11 comparisons")
    if comparisons["outputs_are_not_riley_results"] is not True or q_comparison is None or comparisons["q_default_vs_p7_shadow_raw_q"] != q_comparison:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 comparison contract differs")
    if document["quality_pass"] != bool(all_repeated and q_comparison["bf16_exact"] is True):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 quality result differs")
    provenance = _require_mapping(document["provenance"], "P11 provenance")
    _require_exact_keys(provenance, {"source_repository"}, "P11 provenance")
    _validate_source_provenance(provenance["source_repository"])
    sidecar = _require_mapping(document["sidecar"], "P11 sidecar")
    _require_exact_keys(sidecar, {"path", "sha256", "format", "tensor_count"}, "P11 sidecar")
    if Path(_require_string(sidecar["path"], "P11 sidecar path")).name != sidecar["path"] or sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(TRACE_TENSORS):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 sidecar metadata differs")
    _require_sha256(sidecar["sha256"], "P11 sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "P11 tensors")
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 tensor set differs")
    for name, shape in _expected_shapes().items():
        record = _require_mapping(tensors[name], f"P11 tensor {name}")
        _require_exact_keys(record, {"key", "shape", "dtype", "canonical_byte_order", "bf16_le_bytes", "bf16_le_sha256"}, f"P11 tensor {name}")
        if (
            record["key"] != _sidecar_key(name)
            or record["shape"] != list(shape)
            or record["dtype"] != "bfloat16"
            or record["canonical_byte_order"] != "little-endian-u16"
            or record["bf16_le_bytes"] != _element_count(shape) * BF16_BYTES
        ):
            raise Qwen3BP2051QkvDefaultCublasTraceError(f"P11 tensor {name} metadata differs")
        _require_sha256(record["bf16_le_sha256"], f"P11 tensor {name} SHA-256")
    for item in PROJECTIONS:
        result = _require_mapping(results[item.identifier], "P11 projection result")
        if result["output_bf16_le_sha256"] != _require_mapping(tensors[item.output_name], "P11 output tensor")["bf16_le_sha256"]:
            raise Qwen3BP2051QkvDefaultCublasTraceError("P11 result/output hash binding differs")


def validate_sidecar_against_manifest(document: Mapping[str, object], sidecar_path: Path) -> None:
    validate_manifest(document)
    sidecar = _regular_file(sidecar_path.expanduser(), "P11 sidecar")
    if _sha256_file(sidecar) != _require_mapping(document["sidecar"], "P11 sidecar")["sha256"]:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 sidecar SHA-256 differs")
    try:
        header, payload_offset, file_size = projection._read_safetensors_header(sidecar)
        expected_keys = {_sidecar_key(name) for name in TRACE_TENSORS}
        actual_keys = {key for key in header if key != "__metadata__"}
        if actual_keys != expected_keys:
            raise Qwen3BP2051QkvDefaultCublasTraceError("P11 sidecar keys differ")
        tensors = _require_mapping(document["tensors"], "P11 tensors")
        ranges: list[tuple[int, int]] = []
        with sidecar.open("rb") as stream:
            for name, shape in _expected_shapes().items():
                metadata = _require_mapping(header[_sidecar_key(name)], f"P11 sidecar tensor {name}")
                _require_exact_keys(metadata, {"dtype", "shape", "data_offsets"}, f"P11 sidecar tensor {name}")
                record = _require_mapping(tensors[name], f"P11 tensor {name}")
                offsets = metadata["data_offsets"]
                if (
                    metadata["dtype"] != "BF16"
                    or metadata["shape"] != list(shape)
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
                    or offsets[0] < 0
                    or offsets[1] != offsets[0] + _element_count(shape) * BF16_BYTES
                    or payload_offset + offsets[1] > file_size
                ):
                    raise Qwen3BP2051QkvDefaultCublasTraceError(f"P11 sidecar tensor {name} layout differs")
                stream.seek(payload_offset + offsets[0])
                raw = stream.read(offsets[1] - offsets[0])
                if len(raw) != offsets[1] - offsets[0] or _sha256_bytes(raw) != record["bf16_le_sha256"]:
                    raise Qwen3BP2051QkvDefaultCublasTraceError(f"P11 sidecar tensor {name} raw BF16 hash differs")
                projection._validate_finite_bf16(raw, f"P11 sidecar tensor {name}")
                ranges.append((offsets[0], offsets[1]))
    except (OSError, KeyError, projection.Qwen3BP2051ProjectionTraceError) as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError("cannot read P11 sidecar") from error
    expected_offset = 0
    for start, end in sorted(ranges):
        if start != expected_offset:
            raise Qwen3BP2051QkvDefaultCublasTraceError("P11 sidecar tensor ranges are not contiguous")
        expected_offset = end
    if payload_offset + expected_offset != file_size:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 sidecar payload size differs")


def _validate_p7_checkpoint_binding(p7_document: Mapping[str, object], checkpoint: oracle.CheckpointManifest) -> None:
    p7_model = _require_mapping(p7_document["model"], "P7 model")
    if p7_model["checkpoint_receipt_sha256"] != checkpoint.receipt.sha256:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P7 checkpoint receipt binding differs")


def produce_hf_trace(
    *,
    checkpoint_path: Path,
    p7_manifest_path: Path,
    p7_sidecar_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    repo_root: Path,
    device: str,
    created_at: datetime | None = None,
    backend_factory: BackendFactory = oracle.HuggingFaceQwen3BBackend.load,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Create the P11 Q/K/V default-Cublas artifact without serving integration."""

    manifest, sidecar = _output_paths(manifest_path, sidecar_path, repo_root)
    p7_document, p7_manifest, p7_sidecar = _load_p7_artifact(p7_manifest_path, p7_sidecar_path)
    try:
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(str(error)) from error
    _validate_p7_checkpoint_binding(p7_document, checkpoint)
    checkpoint_weights = {item.identifier: _checkpoint_weight_raw(checkpoint, item) for item in PROJECTIONS}
    provenance = source_provenance_factory(repo_root)
    backend = backend_factory(checkpoint=checkpoint, device=device)
    sidecar_written = False
    torch = backend._torch
    original_backend = _backend_name(torch)
    original_flags = _policy_readback(torch)
    original_tf32_flags = _tf32_readback(torch)
    try:
        _base, layers, _module = stage._validate_topology(backend._model)
        p7_input, p7_raw_q = _load_p7_tensors(p7_sidecar, torch)
        input_norm = p7_input.to(device=backend._device, dtype=torch.bfloat16).contiguous()
        policy_actual = _apply_default_policy(torch)
        tensors: dict[str, object] = {
            P7_INPUT_NAME: p7_input.contiguous(),
            P7_RAW_Q_NAME: p7_raw_q.contiguous(),
        }
        p7_raw_q_bytes = _validate_tensor(tensors[P7_RAW_Q_NAME], (M, Q_WIDTH), torch, P7_RAW_Q_NAME)
        results: dict[str, dict[str, object]] = {}
        for item in PROJECTIONS:
            module = getattr(layers[0].self_attn, item.module_name)
            weight = module.weight.detach().contiguous()
            weight_raw = _validate_tensor(weight, (item.width, K), torch, item.weight_name)
            if weight_raw != checkpoint_weights[item.identifier]:
                raise Qwen3BP2051QkvDefaultCublasTraceError(
                    f"loaded {item.identifier.upper()} weight differs from checkpoint bytes"
                )
            with torch.inference_mode():
                output = torch.nn.functional.linear(input_norm, weight, bias=None).contiguous()
                repeated = torch.nn.functional.linear(input_norm, weight, bias=None).contiguous()
            torch.cuda.synchronize(backend._device)
            output_raw = _validate_tensor(output, (M, item.width), torch, item.output_name)
            repeated_raw = _validate_tensor(repeated, (M, item.width), torch, f"repeat {item.output_name}")
            tensors[item.weight_name] = weight.cpu().contiguous()
            tensors[item.output_name] = output.cpu().contiguous()
            results[item.identifier] = _projection_result_document(
                item, output_raw, repeated_raw, p7_raw_q_bytes, torch
            )
        _write_sidecar_exclusive(
            sidecar,
            {_sidecar_key(name): tensors[name] for name in TRACE_TENSORS},
            sidecar_writer,
        )
        sidecar_written = True
        document = build_manifest(
            checkpoint=checkpoint,
            p7_document=p7_document,
            p7_manifest=p7_manifest,
            p7_sidecar=p7_sidecar,
            sidecar=sidecar,
            tensors=tensors,
            torch=torch,
            producer=_producer_document(backend, torch),
            policy_actual=policy_actual,
            projection_results=results,
            source_provenance=provenance,
            created_at=created_at or datetime.now(timezone.utc),
        )
        validate_sidecar_against_manifest(document, sidecar)
        try:
            oracle.write_artifact_exclusive(manifest, document)
        except oracle.Qwen3BServingOracleError as error:
            raise Qwen3BP2051QkvDefaultCublasTraceError(str(error)) from error
        return document
    except BaseException:
        if sidecar_written:
            try:
                sidecar.unlink()
            except OSError:
                pass
        raise
    finally:
        try:
            _restore_policy(torch, original_backend, original_flags, original_tf32_flags)
        finally:
            backend.close()


def validate_bindings(
    *,
    manifest_path: Path,
    sidecar_path: Path,
    p7_manifest_path: Path,
    p7_sidecar_path: Path,
    checkpoint_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Validate P11/P7 sidecars, all three checkpoint weights, and source hashes."""

    document = _load_manifest(manifest_path)
    validate_manifest(document)
    validate_sidecar_against_manifest(document, sidecar_path)
    p7_document, p7_manifest, p7_sidecar = _load_p7_artifact(p7_manifest_path, p7_sidecar_path)
    binding = _require_mapping(document["p7_binding"], "P11 P7 binding")
    if (
        binding["manifest_sha256"] != _sha256_file(p7_manifest)
        or binding["sidecar_sha256"] != _sha256_file(p7_sidecar)
        or binding["manifest_filename"] != p7_manifest.name
        or binding["sidecar_filename"] != p7_sidecar.name
    ):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 P7 artifact binding differs")
    p7_model = _require_mapping(p7_document["model"], "P7 model")
    p7_source = _require_mapping(_require_mapping(p7_document["provenance"], "P7 provenance")["source_repository"], "P7 source provenance")
    p7_tensors = _require_mapping(p7_document["tensors"], "P7 tensors")
    p7_input = _require_mapping(p7_tensors["layer0.input_norm"], "P7 input tensor")
    p7_raw_q = _require_mapping(p7_tensors["layer0.q_proj.unbiased_linear"], "P7 raw-Q tensor")
    if (
        binding["checkpoint_receipt_sha256"] != p7_model["checkpoint_receipt_sha256"]
        or binding["source_revision"] != p7_source["git_revision"]
        or binding["input_bf16_le_sha256"] != p7_input["bf16_le_sha256"]
        or binding["raw_q_bf16_le_sha256"] != p7_raw_q["bf16_le_sha256"]
    ):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 P7 manifest tensor binding differs")
    try:
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051QkvDefaultCublasTraceError(str(error)) from error
    _validate_p7_checkpoint_binding(p7_document, checkpoint)
    model = _require_mapping(document["model"], "P11 model")
    if model["checkpoint_receipt_sha256"] != checkpoint.receipt.sha256:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 checkpoint receipt differs")
    tensors = _require_mapping(document["tensors"], "P11 tensors")
    if (
        _require_mapping(tensors[P7_INPUT_NAME], "P11 input tensor")["bf16_le_sha256"] != binding["input_bf16_le_sha256"]
        or _require_mapping(tensors[P7_RAW_Q_NAME], "P11 raw-Q tensor")["bf16_le_sha256"] != binding["raw_q_bf16_le_sha256"]
    ):
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 P7 operand byte binding differs")
    for item in PROJECTIONS:
        expected_raw = _checkpoint_weight_raw(checkpoint, item)
        if _require_mapping(tensors[item.weight_name], "P11 weight tensor")["bf16_le_sha256"] != _sha256_bytes(expected_raw):
            raise Qwen3BP2051QkvDefaultCublasTraceError("P11 checkpoint weight binding differs")
    expected_sources = collect_source_provenance(_regular_directory(repo_root.expanduser(), "repository root"))
    observed_sources = _require_mapping(_require_mapping(document["provenance"], "P11 provenance")["source_repository"], "P11 source provenance")
    if observed_sources != expected_sources:
        raise Qwen3BP2051QkvDefaultCublasTraceError("P11 source provenance differs")
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_p2051_qkv_default_cublas_trace",
        description="offline Qwen2.5-3B P2051 raw-Q/K/V default-Cublas trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser("produce", help="write a create-only P11 BF16 sidecar and manifest")
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--p7-manifest", type=Path, required=True)
    produce.add_argument("--p7-sidecar", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--sidecar", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    validate = commands.add_parser("validate", help="validate P11/P7 bindings without CUDA")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--sidecar", type=Path, required=True)
    validate.add_argument("--p7-manifest", type=Path, required=True)
    validate.add_argument("--p7-sidecar", type=Path, required=True)
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "produce":
            document = produce_hf_trace(
                checkpoint_path=args.checkpoint,
                p7_manifest_path=args.p7_manifest,
                p7_sidecar_path=args.p7_sidecar,
                manifest_path=args.manifest,
                sidecar_path=args.sidecar,
                repo_root=args.repo_root,
                device=args.device,
            )
            print(
                f"created {document['artifact_kind']}: {document['sidecar']['path']} "
                f"quality_pass={document['quality_pass']}"
            )
            return 0 if document["quality_pass"] else 3
        document = validate_bindings(
            manifest_path=args.manifest,
            sidecar_path=args.sidecar,
            p7_manifest_path=args.p7_manifest,
            p7_sidecar_path=args.p7_sidecar,
            checkpoint_path=args.checkpoint,
            repo_root=args.repo_root,
        )
        print(f"validated {document['artifact_kind']}: {document['sidecar']['path']}")
        return 0
    except (OSError, RuntimeError, Qwen3BP2051QkvDefaultCublasTraceError, oracle.Qwen3BServingOracleError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
