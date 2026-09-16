"""Offline P2051 raw-Q BF16 arithmetic trace.

This diagnostic consumes the immutable P7 projection sidecar and checkpoint, then
captures three direct Hugging Face ``F.linear`` raw-Q outputs under explicit
PyTorch BLAS/reduction settings.  It is offline-only: it is neither imported by
Riley serving nor a Rust-to-Python execution path.

The outputs deliberately distinguish the original P7 default from two controls:
``Cublas + (reduced=True, split_k=True)``, ``Cublas + (False, True)``, and
``CublasLt + (False, False)``.  The latter changes the preferred PyTorch BLAS
backend, so it is recorded as an arithmetic control, never as P7/vLLM serving
behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2051-bf16-arithmetic-trace.v1"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2051-raw-q-arithmetic-trace"
TRACE_ID = "qwen3b-p2051-layer0-raw-q-bf16-arithmetic-v1"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-p2051-bf16-arithmetic-trace-v1"
BF16_BYTES = 2
M = projection.CONTEXT_TOKEN_COUNT
K = projection.MODEL_HIDDEN_SIZE
N = projection.MODEL_HIDDEN_SIZE

P7_INPUT_KEY = "trace/layer0/input_norm"
P7_RAW_Q_KEY = "trace/layer0/q_proj/unbiased_linear"
P7_INPUT_NAME = "p7_input_norm"
P7_RAW_Q_NAME = "p7_shadow_raw_q"
Q_WEIGHT_NAME = "layer0_q_proj_weight"


@dataclass(frozen=True)
class ArithmeticPolicy:
    """One explicit PyTorch arithmetic observation, not a serving selector."""

    identifier: str
    preferred_blas: str
    allow_reduced_precision_reduction: bool
    allow_split_k: bool
    role: str


POLICIES = (
    ArithmeticPolicy(
        identifier="p7_default_cublas_reduced_splitk",
        preferred_blas="cublas",
        allow_reduced_precision_reduction=True,
        allow_split_k=True,
        role="p7_default_reproduction",
    ),
    ArithmeticPolicy(
        identifier="cublas_reduced_off_splitk_on",
        preferred_blas="cublas",
        allow_reduced_precision_reduction=False,
        allow_split_k=True,
        role="reduced_precision_control",
    ),
    ArithmeticPolicy(
        identifier="cublaslt_reduced_off_splitk_off",
        preferred_blas="cublaslt",
        allow_reduced_precision_reduction=False,
        allow_split_k=False,
        role="explicit_cublaslt_full_reduction_control",
    ),
)

TRACE_TENSORS = (
    P7_INPUT_NAME,
    Q_WEIGHT_NAME,
    P7_RAW_Q_NAME,
    *(f"raw_q/{policy.identifier}" for policy in POLICIES),
)

SOURCE_PATHS = {
    "p9_bf16_arithmetic_trace": "tools/python/reference/riley_reference/qwen3b_p2051_bf16_arithmetic_trace.py",
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


class Qwen3BP2051Bf16ArithmeticTraceError(RuntimeError):
    """Raised when the fixed P9 raw-Q arithmetic contract is violated."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return oracle._sha256_file(path)


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"{label} must be a directory")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"cannot resolve {label}: {path}") from error


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"{label} must be a regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"cannot resolve {label}: {path}") from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise Qwen3BP2051Bf16ArithmeticTraceError(
            f"{label} fields differ: missing={sorted(expected - set(value))!r} "
            f"extra={sorted(set(value) - expected)!r}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    try:
        return oracle._require_sha256(value, label)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise Qwen3BP2051Bf16ArithmeticTraceError("created_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sidecar_key(name: str) -> str:
    return f"trace/{name}"


def _expected_shapes() -> dict[str, tuple[int, int]]:
    result = {
        P7_INPUT_NAME: (M, K),
        Q_WEIGHT_NAME: (N, K),
        P7_RAW_Q_NAME: (M, N),
    }
    result.update({f"raw_q/{policy.identifier}": (M, N) for policy in POLICIES})
    if tuple(result) != TRACE_TENSORS:
        raise AssertionError("P9 tensor ordering differs")
    return result


def _element_count(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise Qwen3BP2051Bf16ArithmeticTraceError("tensor shape dimensions must be positive integers")
        count *= dimension
    return count


def _policy_document(policy: ArithmeticPolicy) -> dict[str, object]:
    return {
        "id": policy.identifier,
        "role": policy.role,
        "preferred_blas_requested": policy.preferred_blas,
        "allow_bf16_reduced_precision_reduction_requested": policy.allow_reduced_precision_reduction,
        "allow_bf16_reduced_precision_reduction_split_k_requested": policy.allow_split_k,
        "operator": "torch.nn.functional.linear",
        "bias": False,
        "operand_shape": {"m": M, "n": N, "k": K},
        "runtime_status": "captured",
    }


def _policy_documents() -> list[dict[str, object]]:
    return [_policy_document(policy) for policy in POLICIES]


def _source_record(root: Path, relative: str) -> dict[str, object]:
    path = _regular_file(root / relative, f"source file {relative}")
    return {"path": relative, "sha256": _sha256_file(path)}


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Pin the P9 producer files; the artifact may be created only from tracked sources."""

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
        raise Qwen3BP2051Bf16ArithmeticTraceError("cannot record P9 Git provenance") from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 Git revision is malformed")
    if status:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 producer source must be clean")
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
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 output paths must be absolute")
    if manifest.parent != sidecar.parent:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 manifest and sidecar must be siblings")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(manifest.parent, "P9 output directory")
    manifest = parent / manifest.name
    sidecar = parent / sidecar.name
    if manifest.suffix != ".json" or sidecar.suffix != ".safetensors" or manifest == sidecar:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 output extensions differ")
    for output in (manifest, sidecar):
        if output == root or root in output.parents:
            raise Qwen3BP2051Bf16ArithmeticTraceError("P9 artifacts must be outside the repository")
        if output.exists() or output.is_symlink():
            raise Qwen3BP2051Bf16ArithmeticTraceError(f"refusing to overwrite P9 artifact: {output}")
    return manifest, sidecar


def _canonical_bf16_le_bytes(tensor: object, torch: Any, label: str) -> bytes:
    try:
        raw = bytes(tensor.detach().contiguous().view(torch.uint8).cpu().tolist())
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"cannot read {label} as BF16") from error
    if sys.byteorder == "little":
        return raw
    if sys.byteorder == "big":
        return b"".join(raw[offset : offset + 2][::-1] for offset in range(0, len(raw), 2))
    raise Qwen3BP2051Bf16ArithmeticTraceError("unsupported host byte order for BF16")


def _validate_tensor(tensor: object, shape: tuple[int, int], torch: Any, label: str) -> bytes:
    try:
        actual_shape = tuple(tensor.shape)
        dtype = tensor.dtype
        contiguous = tensor.is_contiguous()
    except AttributeError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"{label} is not a tensor") from error
    if actual_shape != shape or dtype != torch.bfloat16 or not contiguous:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"{label} tensor contract differs")
    raw = _canonical_bf16_le_bytes(tensor, torch, label)
    expected_bytes = _element_count(shape) * BF16_BYTES
    if len(raw) != expected_bytes:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"{label} BF16 byte count differs")
    try:
        projection._validate_finite_bf16(raw, label)
    except projection.Qwen3BP2051ProjectionTraceError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error
    return raw


def _metrics(expected: bytes, actual: bytes, torch: Any) -> dict[str, object]:
    if len(expected) != len(actual) or len(expected) % BF16_BYTES:
        raise Qwen3BP2051Bf16ArithmeticTraceError("comparison BF16 byte lengths differ")
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
        raise Qwen3BP2051Bf16ArithmeticTraceError("PyTorch preferred BLAS API is unavailable") from error
    if observed.endswith("Cublaslt"):
        return "cublaslt"
    if observed.endswith("Cublas"):
        return "cublas"
    raise Qwen3BP2051Bf16ArithmeticTraceError(f"unsupported preferred BLAS backend: {observed}")


def _policy_readback(torch: Any) -> tuple[bool, bool]:
    try:
        reduced = bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        split_k = bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError("PyTorch BF16 policy API is unavailable") from error
    return reduced, split_k


def _apply_policy(torch: Any, policy: ArithmeticPolicy) -> dict[str, object]:
    try:
        torch.backends.cuda.preferred_blas_library(policy.preferred_blas)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            policy.allow_reduced_precision_reduction,
            policy.allow_split_k,
        )
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"cannot configure P9 policy {policy.identifier}") from error
    backend = _backend_name(torch)
    reduced, split_k = _policy_readback(torch)
    if (
        backend != policy.preferred_blas
        or reduced != policy.allow_reduced_precision_reduction
        or split_k != policy.allow_split_k
    ):
        raise Qwen3BP2051Bf16ArithmeticTraceError(f"P9 policy readback differs: {policy.identifier}")
    return {
        "preferred_blas_actual": backend,
        "allow_bf16_reduced_precision_reduction_actual": reduced,
        "allow_bf16_reduced_precision_reduction_split_k_actual": split_k,
    }


def _restore_policy(torch: Any, backend: str, flags: tuple[bool, bool]) -> None:
    try:
        torch.backends.cuda.preferred_blas_library(backend)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flags
    except (AttributeError, RuntimeError, TypeError) as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError("cannot restore PyTorch BF16 policy") from error
    if _backend_name(torch) != backend or _policy_readback(torch) != flags:
        raise Qwen3BP2051Bf16ArithmeticTraceError("restored PyTorch BF16 policy differs")


def _load_p7_artifact(p7_manifest_path: Path, p7_sidecar_path: Path) -> tuple[dict[str, object], Path, Path]:
    manifest_path = _regular_file(p7_manifest_path.expanduser(), "P7 manifest")
    sidecar_path = _regular_file(p7_sidecar_path.expanduser(), "P7 sidecar")
    try:
        document = projection._load_manifest(manifest_path)
        projection.validate_manifest(document)
        projection.validate_sidecar_against_manifest(document, sidecar_path)
    except projection.Qwen3BP2051ProjectionTraceError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error
    expected_sidecar = _require_mapping(document["sidecar"], "P7 sidecar")
    if _sha256_file(sidecar_path) != expected_sidecar["sha256"]:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P7 sidecar SHA-256 differs")
    return document, manifest_path, sidecar_path


def _load_p7_tensors(sidecar: Path, torch: Any) -> tuple[object, object]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError("pinned safetensors is required for P9 produce") from error
    try:
        with safe_open(str(sidecar), framework="pt", device="cpu") as source:
            input_norm = source.get_tensor(P7_INPUT_KEY)
            raw_q = source.get_tensor(P7_RAW_Q_KEY)
    except (OSError, RuntimeError, ValueError) as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError("cannot load P7 raw-Q tensors") from error
    _validate_tensor(input_norm, (M, K), torch, "P7 input norm")
    _validate_tensor(raw_q, (M, N), torch, "P7 shadow raw Q")
    return input_norm, raw_q


def _tensor_manifest(tensors: Mapping[str, object], torch: Any) -> dict[str, object]:
    records: dict[str, object] = {}
    for name in TRACE_TENSORS:
        shape = _expected_shapes()[name]
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
    metadata = dict(backend.producer_metadata)
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "runtime_dependency_class": "offline-python-reference",
        "torch_version": str(torch.__version__),
        "runtime_cuda_version": str(torch.version.cuda or "unknown"),
        "model_loader": metadata,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def _p7_binding_document(p7_document: Mapping[str, object], p7_manifest: Path, p7_sidecar: Path) -> dict[str, object]:
    p7_model = _require_mapping(p7_document["model"], "P7 model")
    p7_provenance = _require_mapping(p7_document["provenance"], "P7 provenance")
    source = _require_mapping(p7_provenance["source_repository"], "P7 source provenance")
    return {
        "manifest_filename": p7_manifest.name,
        "manifest_sha256": _sha256_file(p7_manifest),
        "sidecar_filename": p7_sidecar.name,
        "sidecar_sha256": _sha256_file(p7_sidecar),
        "checkpoint_receipt_sha256": p7_model["checkpoint_receipt_sha256"],
        "source_revision": source["git_revision"],
        "input_tensor_key": P7_INPUT_KEY,
        "raw_q_tensor_key": P7_RAW_Q_KEY,
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
    policy_observations: list[dict[str, object]],
    source_provenance: Mapping[str, object],
    created_at: datetime,
) -> dict[str, object]:
    p7_raw = _canonical_bf16_le_bytes(tensors[P7_RAW_Q_NAME], torch, P7_RAW_Q_NAME)
    default_raw = _canonical_bf16_le_bytes(
        tensors[f"raw_q/{POLICIES[0].identifier}"], torch, POLICIES[0].identifier
    )
    repeats_pass = all(bool(observation["repeated_bf16_exact"]) for observation in policy_observations)
    p7_match = _metrics(p7_raw, default_raw, torch)
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "trace_id": TRACE_ID,
        "performance_claim_eligible": False,
        "vllm_comparison_eligible": False,
        "serving_selector_changed": False,
        "quality_pass": bool(repeats_pass and p7_match["bf16_exact"]),
        "created_at": _utc_text(created_at),
        "scope": {
            "endpoint": "layer0.q_proj.raw_no_bias",
            "operator": "torch.nn.functional.linear",
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "shape": {"m": M, "n": N, "k": K},
            "offline_only": True,
            "serving_path": False,
        },
        "producer": dict(producer),
        "p7_binding": _p7_binding_document(p7_document, p7_manifest, p7_sidecar),
        "model": {
            "checkpoint_path": str(checkpoint.root),
            "checkpoint_receipt_filename": checkpoint.receipt.path,
            "checkpoint_receipt_sha256": checkpoint.receipt.sha256,
        },
        "policies": policy_observations,
        "comparisons": {
            "p7_default_vs_p7_shadow_raw_q": p7_match,
            "policy_outputs_are_not_riley_results": True,
        },
        "provenance": {"source_repository": dict(source_provenance)},
        "sidecar": {
            "path": sidecar.name,
            "sha256": _sha256_file(sidecar),
            "format": "safetensors",
            "tensor_count": len(TRACE_TENSORS),
        },
        "tensors": _tensor_manifest(tensors, torch),
    }
    validate_manifest(document)
    return document


def _duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise Qwen3BP2051Bf16ArithmeticTraceError(f"JSON object repeats key {key!r}")
        value[key] = item
    return value


def _nonfinite(value: str) -> None:
    raise Qwen3BP2051Bf16ArithmeticTraceError(f"non-finite JSON constant {value!r} is forbidden")


def _load_manifest(path: Path) -> dict[str, object]:
    source = _regular_file(path, "P9 manifest")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_key, parse_constant=_nonfinite
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError("cannot parse P9 manifest") from error
    return dict(_require_mapping(value, "P9 manifest"))


def _validate_policies(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(POLICIES):
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 policies differ")
    for observed, policy in zip(value, POLICIES, strict=True):
        record = _require_mapping(observed, "P9 policy")
        _require_exact_keys(
            record,
            {
                "id", "role", "preferred_blas_requested", "allow_bf16_reduced_precision_reduction_requested",
                "allow_bf16_reduced_precision_reduction_split_k_requested", "operator", "bias", "operand_shape",
                "runtime_status", "preferred_blas_actual", "allow_bf16_reduced_precision_reduction_actual",
                "allow_bf16_reduced_precision_reduction_split_k_actual", "repeated_bf16_exact",
                "output_tensor", "output_bf16_le_sha256", "p7_default_comparison",
            },
            "P9 policy",
        )
        expected = _policy_document(policy)
        for key, expected_value in expected.items():
            if record.get(key) != expected_value:
                raise Qwen3BP2051Bf16ArithmeticTraceError("P9 policy request differs")
        if (
            record["preferred_blas_actual"] != policy.preferred_blas
            or record["allow_bf16_reduced_precision_reduction_actual"] != policy.allow_reduced_precision_reduction
            or record["allow_bf16_reduced_precision_reduction_split_k_actual"] != policy.allow_split_k
            or not isinstance(record["repeated_bf16_exact"], bool)
            or record["output_tensor"] != f"raw_q/{policy.identifier}"
        ):
            raise Qwen3BP2051Bf16ArithmeticTraceError("P9 policy observation differs")
        _require_sha256(record["output_bf16_le_sha256"], "P9 policy output SHA-256")
        comparison = _require_mapping(record["p7_default_comparison"], "P9 policy comparison")
        _require_exact_keys(comparison, {"bf16_exact", "unequal_elements", "total_elements", "max_abs"}, "P9 policy comparison")
        if not isinstance(comparison["bf16_exact"], bool) or comparison["total_elements"] != M * N:
            raise Qwen3BP2051Bf16ArithmeticTraceError("P9 policy comparison differs")


def validate_manifest(document: Mapping[str, object]) -> None:
    _require_exact_keys(
        document,
        {
            "schema_version", "artifact_kind", "trace_id", "performance_claim_eligible", "vllm_comparison_eligible",
            "serving_selector_changed", "quality_pass", "created_at", "scope", "producer", "p7_binding", "model",
            "policies", "comparisons", "provenance", "sidecar", "tensors",
        },
        "P9 manifest",
    )
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["artifact_kind"] != ARTIFACT_KIND
        or document["trace_id"] != TRACE_ID
        or document["performance_claim_eligible"] is not False
        or document["vllm_comparison_eligible"] is not False
        or document["serving_selector_changed"] is not False
        or not isinstance(document["quality_pass"], bool)
    ):
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 manifest identity differs")
    try:
        oracle._validate_utc_text(document["created_at"], "P9 created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error
    scope = _require_mapping(document["scope"], "P9 scope")
    if dict(scope) != {
        "endpoint": "layer0.q_proj.raw_no_bias", "operator": "torch.nn.functional.linear",
        "model_id": oracle.MODEL_ID, "model_revision": oracle.MODEL_REVISION,
        "shape": {"m": M, "n": N, "k": K}, "offline_only": True, "serving_path": False,
    }:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 scope differs")
    producer = _require_mapping(document["producer"], "P9 producer")
    _require_exact_keys(producer, {"implementation_id", "runtime_dependency_class", "torch_version", "runtime_cuda_version", "model_loader", "tf32_enabled"}, "P9 producer")
    if producer["implementation_id"] != IMPLEMENTATION_ID or producer["runtime_dependency_class"] != "offline-python-reference" or producer["tf32_enabled"] is not False:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 producer differs")
    _require_string(producer["torch_version"], "P9 torch version")
    _require_string(producer["runtime_cuda_version"], "P9 CUDA runtime version")
    _require_mapping(producer["model_loader"], "P9 model loader")
    binding = _require_mapping(document["p7_binding"], "P9 P7 binding")
    _require_exact_keys(binding, {"manifest_filename", "manifest_sha256", "sidecar_filename", "sidecar_sha256", "checkpoint_receipt_sha256", "source_revision", "input_tensor_key", "raw_q_tensor_key"}, "P9 P7 binding")
    if binding["input_tensor_key"] != P7_INPUT_KEY or binding["raw_q_tensor_key"] != P7_RAW_Q_KEY:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 P7 tensor binding differs")
    for key in ("manifest_sha256", "sidecar_sha256", "checkpoint_receipt_sha256"):
        _require_sha256(binding[key], f"P9 P7 {key}")
    _require_string(binding["manifest_filename"], "P9 P7 manifest filename")
    _require_string(binding["sidecar_filename"], "P9 P7 sidecar filename")
    _require_string(binding["source_revision"], "P9 P7 source revision")
    model = _require_mapping(document["model"], "P9 model")
    _require_exact_keys(model, {"checkpoint_path", "checkpoint_receipt_filename", "checkpoint_receipt_sha256"}, "P9 model")
    _require_string(model["checkpoint_path"], "P9 checkpoint path")
    if model["checkpoint_receipt_filename"] != oracle.CHECKPOINT_RECEIPT_FILENAME:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 checkpoint receipt filename differs")
    _require_sha256(model["checkpoint_receipt_sha256"], "P9 checkpoint receipt SHA-256")
    _validate_policies(document["policies"])
    comparisons = _require_mapping(document["comparisons"], "P9 comparisons")
    _require_exact_keys(comparisons, {"p7_default_vs_p7_shadow_raw_q", "policy_outputs_are_not_riley_results"}, "P9 comparisons")
    comparison = _require_mapping(comparisons["p7_default_vs_p7_shadow_raw_q"], "P9 P7 comparison")
    _require_exact_keys(comparison, {"bf16_exact", "unequal_elements", "total_elements", "max_abs"}, "P9 P7 comparison")
    if not isinstance(comparison["bf16_exact"], bool) or comparison["total_elements"] != M * N or comparisons["policy_outputs_are_not_riley_results"] is not True:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 comparison contract differs")
    provenance = _require_mapping(document["provenance"], "P9 provenance")
    _require_exact_keys(provenance, {"source_repository"}, "P9 provenance")
    source = _require_mapping(provenance["source_repository"], "P9 source provenance")
    _require_exact_keys(source, {"git_revision", "source_dirty", "source_status_sha256", "sources"}, "P9 source provenance")
    if source["source_dirty"] is not False or oracle.GIT_REVISION_RE.fullmatch(_require_string(source["git_revision"], "P9 Git revision")) is None:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 source provenance differs")
    _require_sha256(source["source_status_sha256"], "P9 source status SHA-256")
    sources = _require_mapping(source["sources"], "P9 source records")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 source files differ")
    for name, path in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], "P9 source record")
        if dict(record).get("path") != path:
            raise Qwen3BP2051Bf16ArithmeticTraceError("P9 source path differs")
        _require_sha256(record.get("sha256"), "P9 source SHA-256")
    sidecar = _require_mapping(document["sidecar"], "P9 sidecar")
    _require_exact_keys(sidecar, {"path", "sha256", "format", "tensor_count"}, "P9 sidecar")
    if Path(_require_string(sidecar["path"], "P9 sidecar path")).name != sidecar["path"] or sidecar["format"] != "safetensors" or sidecar["tensor_count"] != len(TRACE_TENSORS):
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 sidecar metadata differs")
    _require_sha256(sidecar["sha256"], "P9 sidecar SHA-256")
    tensors = _require_mapping(document["tensors"], "P9 tensors")
    if tuple(tensors) != TRACE_TENSORS:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 tensor ordering differs")
    for name, shape in _expected_shapes().items():
        record = _require_mapping(tensors[name], f"P9 tensor {name}")
        _require_exact_keys(record, {"key", "shape", "dtype", "canonical_byte_order", "bf16_le_bytes", "bf16_le_sha256"}, f"P9 tensor {name}")
        if record["key"] != _sidecar_key(name) or record["shape"] != list(shape) or record["dtype"] != "bfloat16" or record["canonical_byte_order"] != "little-endian-u16" or record["bf16_le_bytes"] != _element_count(shape) * BF16_BYTES:
            raise Qwen3BP2051Bf16ArithmeticTraceError(f"P9 tensor {name} metadata differs")
        _require_sha256(record["bf16_le_sha256"], f"P9 tensor {name} SHA-256")


def validate_sidecar_against_manifest(document: Mapping[str, object], sidecar_path: Path) -> None:
    validate_manifest(document)
    sidecar = _regular_file(sidecar_path.expanduser(), "P9 sidecar")
    if _sha256_file(sidecar) != _require_mapping(document["sidecar"], "P9 sidecar")["sha256"]:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 sidecar SHA-256 differs")
    try:
        header, payload_offset, file_size = projection._read_safetensors_header(sidecar)
    except projection.Qwen3BP2051ProjectionTraceError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error
    tensors = _require_mapping(document["tensors"], "P9 tensors")
    if set(header) != {_sidecar_key(name) for name in TRACE_TENSORS}:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 sidecar keys differ")
    expected_offset = 0
    try:
        with sidecar.open("rb") as stream:
            for name in TRACE_TENSORS:
                meta = _require_mapping(header[_sidecar_key(name)], f"P9 sidecar tensor {name}")
                _require_exact_keys(meta, {"dtype", "shape", "data_offsets"}, f"P9 sidecar tensor {name}")
                record = _require_mapping(tensors[name], f"P9 tensor {name}")
                offsets = meta["data_offsets"]
                if meta["dtype"] != "BF16" or meta["shape"] != record["shape"] or not isinstance(offsets, list) or len(offsets) != 2 or offsets[0] != expected_offset or offsets[1] != expected_offset + record["bf16_le_bytes"]:
                    raise Qwen3BP2051Bf16ArithmeticTraceError(f"P9 sidecar tensor {name} layout differs")
                stream.seek(payload_offset + offsets[0])
                raw = stream.read(record["bf16_le_bytes"])
                if len(raw) != record["bf16_le_bytes"] or _sha256_bytes(raw) != record["bf16_le_sha256"]:
                    raise Qwen3BP2051Bf16ArithmeticTraceError(f"P9 sidecar tensor {name} raw BF16 hash differs")
                try:
                    projection._validate_finite_bf16(raw, f"P9 sidecar tensor {name}")
                except projection.Qwen3BP2051ProjectionTraceError as error:
                    raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error
                expected_offset = offsets[1]
    except OSError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError("cannot read P9 sidecar") from error
    if payload_offset + expected_offset != file_size:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 sidecar payload size differs")


def _validate_p7_checkpoint_binding(p7_document: Mapping[str, object], checkpoint: oracle.CheckpointManifest) -> None:
    p7_model = _require_mapping(p7_document["model"], "P7 model")
    if p7_model["checkpoint_receipt_sha256"] != checkpoint.receipt.sha256:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P7 checkpoint receipt binding differs")


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
    """Create the P9 sidecar and manifest; quality false is written, not erased."""

    manifest, sidecar = _output_paths(manifest_path, sidecar_path, repo_root)
    p7_document, p7_manifest, p7_sidecar = _load_p7_artifact(p7_manifest_path, p7_sidecar_path)
    try:
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error
    _validate_p7_checkpoint_binding(p7_document, checkpoint)
    provenance = source_provenance_factory(repo_root)
    backend = backend_factory(checkpoint=checkpoint, device=device)
    sidecar_written = False
    torch = backend._torch
    original_backend = _backend_name(torch)
    original_flags = _policy_readback(torch)
    try:
        if original_backend != "cublas" or original_flags != (True, True):
            raise Qwen3BP2051Bf16ArithmeticTraceError("P9 must start from P7 Cublas/(true,true) policy")
        _base, layers, _module = stage._validate_topology(backend._model)
        weight = layers[0].self_attn.q_proj.weight.detach()
        p7_input, p7_raw_q = _load_p7_tensors(p7_sidecar, torch)
        input_norm = p7_input.to(device=backend._device, dtype=torch.bfloat16).contiguous()
        tensors: dict[str, object] = {
            P7_INPUT_NAME: p7_input.contiguous(),
            Q_WEIGHT_NAME: weight.contiguous(),
            P7_RAW_Q_NAME: p7_raw_q.contiguous(),
        }
        _validate_tensor(tensors[P7_INPUT_NAME], (M, K), torch, P7_INPUT_NAME)
        _validate_tensor(tensors[Q_WEIGHT_NAME], (N, K), torch, Q_WEIGHT_NAME)
        p7_raw_bytes = _validate_tensor(tensors[P7_RAW_Q_NAME], (M, N), torch, P7_RAW_Q_NAME)
        observations: list[dict[str, object]] = []
        for policy in POLICIES:
            actual = _apply_policy(torch, policy)
            with torch.inference_mode():
                output = torch.nn.functional.linear(input_norm, weight, bias=None).contiguous()
                repeated = torch.nn.functional.linear(input_norm, weight, bias=None).contiguous()
            torch.cuda.synchronize(backend._device)
            output_raw = _validate_tensor(output, (M, N), torch, f"P9 {policy.identifier}")
            repeated_raw = _validate_tensor(repeated, (M, N), torch, f"P9 repeat {policy.identifier}")
            tensor_name = f"raw_q/{policy.identifier}"
            tensors[tensor_name] = output.cpu().contiguous()
            observation = _policy_document(policy)
            observation.update(actual)
            observation.update(
                {
                    "repeated_bf16_exact": output_raw == repeated_raw,
                    "output_tensor": tensor_name,
                    "output_bf16_le_sha256": _sha256_bytes(output_raw),
                    "p7_default_comparison": _metrics(p7_raw_bytes, output_raw, torch),
                }
            )
            observations.append(observation)
        _write_sidecar_exclusive(sidecar, {_sidecar_key(name): tensors[name] for name in TRACE_TENSORS}, sidecar_writer)
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
            policy_observations=observations,
            source_provenance=provenance,
            created_at=created_at or datetime.now(timezone.utc),
        )
        validate_sidecar_against_manifest(document, sidecar)
        try:
            oracle.write_artifact_exclusive(manifest, document)
        except oracle.Qwen3BServingOracleError as error:
            raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error
        return document
    except BaseException:
        if sidecar_written:
            # A published sidecar without its create-only manifest has no valid artifact identity.
            try:
                sidecar.unlink()
            except OSError:
                pass
        raise
    finally:
        try:
            _restore_policy(torch, original_backend, original_flags)
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
    """Validate P9/P7 sidecars, checkpoint receipt, and P9 source hashes without CUDA."""

    document = _load_manifest(manifest_path)
    validate_manifest(document)
    validate_sidecar_against_manifest(document, sidecar_path)
    p7_document, p7_manifest, p7_sidecar = _load_p7_artifact(p7_manifest_path, p7_sidecar_path)
    binding = _require_mapping(document["p7_binding"], "P9 P7 binding")
    if (
        binding["manifest_sha256"] != _sha256_file(p7_manifest)
        or binding["sidecar_sha256"] != _sha256_file(p7_sidecar)
        or binding["manifest_filename"] != p7_manifest.name
        or binding["sidecar_filename"] != p7_sidecar.name
    ):
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 P7 artifact binding differs")
    try:
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051Bf16ArithmeticTraceError(str(error)) from error
    _validate_p7_checkpoint_binding(p7_document, checkpoint)
    model = _require_mapping(document["model"], "P9 model")
    if model["checkpoint_receipt_sha256"] != checkpoint.receipt.sha256:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 checkpoint receipt differs")
    expected_sources = collect_source_provenance(_regular_directory(repo_root.expanduser(), "repository root"))
    observed_sources = _require_mapping(_require_mapping(document["provenance"], "P9 provenance")["source_repository"], "P9 source provenance")
    if observed_sources != expected_sources:
        raise Qwen3BP2051Bf16ArithmeticTraceError("P9 source provenance differs")
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_p2051_bf16_arithmetic_trace",
        description="offline Qwen2.5-3B P2051 raw-Q BF16 arithmetic trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser("produce", help="write a create-only P9 BF16 sidecar and manifest")
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--p7-manifest", type=Path, required=True)
    produce.add_argument("--p7-sidecar", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--sidecar", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    validate = commands.add_parser("validate", help="validate P9/P7 bindings without CUDA")
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
    except (OSError, RuntimeError, Qwen3BP2051Bf16ArithmeticTraceError, oracle.Qwen3BServingOracleError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
