"""Offline P2051 Q/K/V staged-bias reference producer.

P11 fixes default-cuBLAS no-bias arithmetic for the complete P2051 input.  This
module binds that immutable P11 sidecar, the P7 HF module-output sidecar, and
the checkpoint Q/K/V BF16 biases, then publishes the predeclared profile:

BF16(P11_raw_default_cublas.to(FP32) + checkpoint_bias.to(FP32)).

The retained P7 HF module output is observational.  It is never silently used
as the staged target when its BF16 bytes differ.  This remains offline-only:
there is no Python path in Riley serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import qwen3b_p2051_projection_trace as p7
from . import qwen3b_p2051_qkv_default_cublas_trace as p11
from . import qwen3b_serving_oracle as oracle
from .hf_calibration import (
    SidecarWriter,
    _default_sidecar_writer,
    _write_sidecar_exclusive,
)

SCHEMA_VERSION = "riley.qwen3b-hf-eager-p2051-qkv-staged-bias-trace.v1"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2051-qkv-staged-bias-trace"
TRACE_ID = "qwen3b-p2051-layer0-qkv-staged-bias-v1"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-p2051-qkv-staged-bias-trace-v1"
BF16_BYTES = 2
M = p11.M
K = p11.K
Q_WIDTH = p11.Q_WIDTH
KV_WIDTH = p11.KV_WIDTH

P7_MANIFEST_SHA256 = "d469e6fc0695e5fc8ec21c0c94bc7665d0d79c447f4d60e58c38ddf72fcd60f7"
P7_SIDECAR_SHA256 = "fffdebe4123a434ce572a6201b1d81c0bdb146aaad355db56342fc299acd6f96"
P11_MANIFEST_SHA256 = "bff87ad504a406f5b8be98414bc4397f03a15efddb1b6b2cf11d625908bd0255"
P11_SIDECAR_SHA256 = "f92424889ee044678de3b316f97a731577537ecfc1ea8d2b11200165c9a53b8c"


@dataclass(frozen=True)
class Projection:
    identifier: str
    module_name: str
    width: int
    p11_raw_name: str
    bias_name: str
    staged_name: str
    p7_actual_name: str
    checkpoint_bias_key: str


PROJECTIONS = (
    Projection(
        "q",
        "q_proj",
        Q_WIDTH,
        "raw_q/default_cublas_reduced_splitk",
        "layer0_q_proj_bias",
        "staged_q/explicit_fp32_bias_add_bf16",
        "p7_actual_q",
        "model.layers.0.self_attn.q_proj.bias",
    ),
    Projection(
        "k",
        "k_proj",
        KV_WIDTH,
        "raw_k/default_cublas_reduced_splitk",
        "layer0_k_proj_bias",
        "staged_k/explicit_fp32_bias_add_bf16",
        "p7_actual_k",
        "model.layers.0.self_attn.k_proj.bias",
    ),
    Projection(
        "v",
        "v_proj",
        KV_WIDTH,
        "raw_v/default_cublas_reduced_splitk",
        "layer0_v_proj_bias",
        "staged_v/explicit_fp32_bias_add_bf16",
        "p7_actual_v",
        "model.layers.0.self_attn.v_proj.bias",
    ),
)
P11_INPUT_NAME = "p11_input_norm"
TRACE_TENSORS = (
    P11_INPUT_NAME,
    *(
        name
        for item in PROJECTIONS
        for name in (
            item.p11_raw_name,
            item.bias_name,
            item.staged_name,
            item.p7_actual_name,
        )
    ),
)
SOURCE_PATHS = {
    "p12_qkv_staged_bias_trace": "tools/python/reference/riley_reference/qwen3b_p2051_qkv_staged_bias_trace.py",
    "p11_qkv_default_cublas_trace": "tools/python/reference/riley_reference/qwen3b_p2051_qkv_default_cublas_trace.py",
    "p7_projection_trace": "tools/python/reference/riley_reference/qwen3b_p2051_projection_trace.py",
    "qwen_serving_oracle": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "hf_calibration": "tools/python/reference/riley_reference/hf_calibration.py",
    "reference_project": "tools/python/reference/pyproject.toml",
    "reference_lock": "tools/python/reference/uv.lock",
    "reference_python": "tools/python/reference/.python-version",
}


class Qwen3BP2051QkvStagedBiasTraceError(RuntimeError):
    """Raised when the P12 artifact or its fixed lineage is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            f"cannot stat {label}: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Qwen3BP2051QkvStagedBiasTraceError(
            f"{label} must be a regular non-symlink file"
        )
    return path.resolve(strict=True)


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            f"cannot stat {label}: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Qwen3BP2051QkvStagedBiasTraceError(
            f"{label} must be a directory, not a symlink"
        )
    return path.resolve(strict=True)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise Qwen3BP2051QkvStagedBiasTraceError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            f"{label} fields differ: missing={sorted(expected - set(value))!r} "
            f"extra={sorted(set(value) - expected)!r}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen3BP2051QkvStagedBiasTraceError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise Qwen3BP2051QkvStagedBiasTraceError(f"{label} must be lowercase SHA-256")
    return text


def _element_count(shape: Sequence[int]) -> int:
    count = 1
    for value in shape:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Qwen3BP2051QkvStagedBiasTraceError("tensor shape dimensions differ")
        count *= value
    return count


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {P11_INPUT_NAME: (M, K)}
    for item in PROJECTIONS:
        shapes[item.p11_raw_name] = (M, item.width)
        shapes[item.bias_name] = (item.width,)
        shapes[item.staged_name] = (M, item.width)
        shapes[item.p7_actual_name] = (M, item.width)
    return shapes


def _sidecar_key(name: str) -> str:
    return f"trace/{name}"


def _metric(expected: bytes, actual: bytes, torch: Any) -> dict[str, object]:
    try:
        return p11._metrics(expected, actual, torch)
    except p11.Qwen3BP2051QkvDefaultCublasTraceError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error


def _validate_metric(value: object, label: str, total: int) -> Mapping[str, object]:
    metric = _mapping(value, label)
    _exact_keys(
        metric, {"bf16_exact", "unequal_elements", "total_elements", "max_abs"}, label
    )
    if (
        not isinstance(metric["bf16_exact"], bool)
        or isinstance(metric["unequal_elements"], bool)
        or not isinstance(metric["unequal_elements"], int)
        or not 0 <= metric["unequal_elements"] <= total
        or metric["total_elements"] != total
        or isinstance(metric["max_abs"], bool)
        or not isinstance(metric["max_abs"], (int, float))
        or metric["max_abs"] < 0
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError(f"{label} differs")
    if metric["bf16_exact"] and (
        metric["unequal_elements"] != 0 or metric["max_abs"] != 0
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError(f"{label} exactness differs")
    return metric


def _source_record(root: Path, relative: str) -> dict[str, object]:
    source = _regular_file(root / relative, f"source {relative}")
    return {"path": relative, "sha256": _sha256_file(source)}


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Pin P12 sources while preserving all unrelated checkout state."""

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
        for relative in SOURCE_PATHS.values():
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=root,
                check=True,
                capture_output=True,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "cannot collect P12 Git provenance"
        ) from error
    if oracle.GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 Git revision is malformed")
    if status:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 producer source must be clean")
    return {
        "git_revision": revision,
        "source_dirty": False,
        "source_status_sha256": _sha256_bytes(status),
        "sources": {
            name: _source_record(root, relative)
            for name, relative in SOURCE_PATHS.items()
        },
    }


def _output_paths(
    manifest_path: Path, sidecar_path: Path, repo_root: Path
) -> tuple[Path, Path]:
    root = _regular_directory(repo_root.expanduser(), "repository root")
    manifest = manifest_path.expanduser()
    sidecar = sidecar_path.expanduser()
    if not manifest.is_absolute() or not sidecar.is_absolute():
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 output paths must be absolute")
    if manifest.parent != sidecar.parent:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "P12 manifest and sidecar must be siblings"
        )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    parent = _regular_directory(manifest.parent, "P12 output directory")
    manifest = parent / manifest.name
    sidecar = parent / sidecar.name
    if (
        manifest.suffix != ".json"
        or sidecar.suffix != ".safetensors"
        or manifest == sidecar
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 output extensions differ")
    for output in (manifest, sidecar):
        if output == root or root in output.parents:
            raise Qwen3BP2051QkvStagedBiasTraceError(
                "P12 artifacts must be outside the repository"
            )
        if output.exists() or output.is_symlink():
            raise Qwen3BP2051QkvStagedBiasTraceError(
                f"refusing to overwrite P12 artifact: {output}"
            )
    return manifest, sidecar


def _load_pinned_artifacts(
    *,
    p7_manifest_path: Path,
    p7_sidecar_path: Path,
    p11_manifest_path: Path,
    p11_sidecar_path: Path,
) -> tuple[Mapping[str, object], Path, Path, Mapping[str, object], Path, Path]:
    try:
        p7_document, p7_manifest, p7_sidecar = p11._load_p7_artifact(
            p7_manifest_path, p7_sidecar_path
        )
        p11_manifest = _regular_file(p11_manifest_path.expanduser(), "P11 manifest")
        p11_sidecar = _regular_file(p11_sidecar_path.expanduser(), "P11 sidecar")
        p11_document = p11._load_manifest(p11_manifest)
        p11.validate_manifest(p11_document)
        p11.validate_sidecar_against_manifest(p11_document, p11_sidecar)
    except (
        p11.Qwen3BP2051QkvDefaultCublasTraceError,
        p7.Qwen3BP2051ProjectionTraceError,
    ) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error
    if (
        _sha256_file(p7_manifest) != P7_MANIFEST_SHA256
        or _sha256_file(p7_sidecar) != P7_SIDECAR_SHA256
        or _sha256_file(p11_manifest) != P11_MANIFEST_SHA256
        or _sha256_file(p11_sidecar) != P11_SIDECAR_SHA256
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "P12 requires the pinned P7 and P11 artifact hashes"
        )
    if (
        p11_document["quality_pass"] is not True
        or p11_document["performance_claim_eligible"] is not False
        or p11_document["vllm_comparison_eligible"] is not False
        or p11_document["serving_selector_changed"] is not False
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P11 eligibility differs")
    binding = _mapping(p11_document["p7_binding"], "P11 P7 binding")
    if (
        binding.get("manifest_sha256") != P7_MANIFEST_SHA256
        or binding.get("sidecar_sha256") != P7_SIDECAR_SHA256
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P11/P7 lineage differs")
    return (
        p7_document,
        p7_manifest,
        p7_sidecar,
        p11_document,
        p11_manifest,
        p11_sidecar,
    )


def _load_sidecar_tensors(
    *,
    sidecar: Path,
    requested: Mapping[str, tuple[str, tuple[int, ...]]],
    torch: Any,
    label: str,
) -> dict[str, object]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "pinned safetensors is required for P12 produce"
        ) from error
    loaded: dict[str, object] = {}
    try:
        with safe_open(str(sidecar), framework="pt", device="cpu") as source:
            for name, (key, shape) in requested.items():
                tensor = source.get_tensor(key).contiguous()
                p11._validate_tensor(tensor, shape, torch, f"{label} {name}")
                loaded[name] = tensor
    except (
        OSError,
        RuntimeError,
        ValueError,
        p11.Qwen3BP2051QkvDefaultCublasTraceError,
    ) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            f"cannot load {label} tensors"
        ) from error
    return loaded


def _checkpoint_bias_raw(
    checkpoint: oracle.CheckpointManifest, item: Projection
) -> bytes:
    index_path = _regular_file(
        checkpoint.root / "model.safetensors.index.json",
        "checkpoint safetensors index",
    )
    try:
        index = _mapping(
            json.loads(index_path.read_text(encoding="utf-8")),
            "checkpoint safetensors index",
        )
        weight_map = _mapping(index["weight_map"], "checkpoint weight map")
        shard_name = _string(
            weight_map[item.checkpoint_bias_key],
            f"checkpoint {item.identifier.upper()} bias shard",
        )
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            f"cannot resolve checkpoint {item.identifier.upper()} bias shard"
        ) from error
    if Path(shard_name).name != shard_name or not shard_name.endswith(".safetensors"):
        raise Qwen3BP2051QkvStagedBiasTraceError("checkpoint bias shard name differs")
    shard = _regular_file(checkpoint.root / shard_name, "checkpoint bias shard")
    expected_bytes = item.width * BF16_BYTES
    try:
        header, payload_offset, file_size = p7._read_safetensors_header(shard)
        record = _mapping(
            header[item.checkpoint_bias_key],
            f"checkpoint {item.identifier.upper()} bias",
        )
        _exact_keys(record, {"dtype", "shape", "data_offsets"}, "checkpoint bias")
        offsets = record["data_offsets"]
        if (
            record["dtype"] != "BF16"
            or record["shape"] != [item.width]
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] != offsets[0] + expected_bytes
            or payload_offset + offsets[1] > file_size
        ):
            raise Qwen3BP2051QkvStagedBiasTraceError("checkpoint bias layout differs")
        with shard.open("rb") as source:
            source.seek(payload_offset + offsets[0])
            raw = source.read(expected_bytes)
    except (
        OSError,
        KeyError,
        p7.Qwen3BP2051ProjectionTraceError,
    ) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "cannot read checkpoint bias"
        ) from error
    if len(raw) != expected_bytes:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "checkpoint bias byte length differs"
        )
    try:
        p7._validate_finite_bf16(raw, "checkpoint bias")
    except p7.Qwen3BP2051ProjectionTraceError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error
    return raw


def _bf16_tensor_from_le(raw: bytes, shape: tuple[int, ...], torch: Any) -> object:
    if len(raw) != _element_count(shape) * BF16_BYTES:
        raise Qwen3BP2051QkvStagedBiasTraceError("BF16 byte count differs")
    if sys.byteorder == "big":
        raw = b"".join(raw[index : index + 2][::-1] for index in range(0, len(raw), 2))
    elif sys.byteorder != "little":
        raise Qwen3BP2051QkvStagedBiasTraceError("unsupported host byte order")
    try:
        tensor = (
            torch.frombuffer(bytearray(raw), dtype=torch.uint16)
            .view(torch.bfloat16)
            .clone()
            .reshape(shape)
            .contiguous()
        )
        p11._validate_tensor(tensor, shape, torch, "checkpoint BF16 tensor")
        return tensor
    except (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        p11.Qwen3BP2051QkvDefaultCublasTraceError,
    ) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "cannot construct BF16 tensor from checkpoint bytes"
        ) from error


def _explicit_staged_bias(
    *, raw: object, bias: object, device: str, torch: Any
) -> tuple[object, object]:
    if not str(device).startswith("cuda") or not bool(torch.cuda.is_available()):
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "P12 staged-bias artifact requires an available CUDA device"
        )
    try:
        raw_gpu = raw.to(device=device, dtype=torch.bfloat16).contiguous()
        bias_gpu = bias.to(device=device, dtype=torch.bfloat16).contiguous()
        first = (
            raw_gpu.to(dtype=torch.float32)
            .add(bias_gpu.to(dtype=torch.float32))
            .to(dtype=torch.bfloat16)
            .contiguous()
        )
        repeated = (
            raw_gpu.to(dtype=torch.float32)
            .add(bias_gpu.to(dtype=torch.float32))
            .to(dtype=torch.bfloat16)
            .contiguous()
        )
        torch.cuda.synchronize(device)
        return first.cpu().contiguous(), repeated.cpu().contiguous()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "cannot calculate explicit staged BF16 bias reference"
        ) from error


def _tensor_manifest(tensors: Mapping[str, object], torch: Any) -> dict[str, object]:
    expected = _expected_shapes()
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 tensor names differ")
    records: dict[str, object] = {}
    for name in TRACE_TENSORS:
        try:
            raw = p11._validate_tensor(tensors[name], expected[name], torch, name)
        except p11.Qwen3BP2051QkvDefaultCublasTraceError as error:
            raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error
        records[name] = {
            "key": _sidecar_key(name),
            "shape": list(expected[name]),
            "dtype": "bfloat16",
            "canonical_byte_order": "little-endian-u16",
            "bf16_le_bytes": len(raw),
            "bf16_le_sha256": _sha256_bytes(raw),
        }
    return records


def _p7_binding_document(
    document: Mapping[str, object], manifest: Path, sidecar: Path
) -> dict[str, object]:
    model = _mapping(document["model"], "P7 model")
    provenance = _mapping(document["provenance"], "P7 provenance")
    source = _mapping(provenance["source_repository"], "P7 source provenance")
    tensors = _mapping(document["tensors"], "P7 tensors")
    return {
        "manifest_filename": manifest.name,
        "manifest_sha256": _sha256_file(manifest),
        "sidecar_filename": sidecar.name,
        "sidecar_sha256": _sha256_file(sidecar),
        "checkpoint_receipt_sha256": model["checkpoint_receipt_sha256"],
        "source_revision": source["git_revision"],
        "input_tensor_key": p11.P7_INPUT_KEY,
        "actual_tensor_keys": {
            item.identifier: f"trace/layer0/{item.module_name}" for item in PROJECTIONS
        },
        "actual_bf16_le_sha256": {
            item.identifier: _mapping(
                tensors[f"layer0.{item.module_name}"], f"P7 {item.identifier} tensor"
            )["bf16_le_sha256"]
            for item in PROJECTIONS
        },
    }


def _p11_binding_document(
    document: Mapping[str, object], manifest: Path, sidecar: Path
) -> dict[str, object]:
    model = _mapping(document["model"], "P11 model")
    provenance = _mapping(document["provenance"], "P11 provenance")
    source = _mapping(provenance["source_repository"], "P11 source provenance")
    tensors = _mapping(document["tensors"], "P11 tensors")
    return {
        "manifest_filename": manifest.name,
        "manifest_sha256": _sha256_file(manifest),
        "sidecar_filename": sidecar.name,
        "sidecar_sha256": _sha256_file(sidecar),
        "checkpoint_receipt_sha256": model["checkpoint_receipt_sha256"],
        "source_revision": source["git_revision"],
        "input_tensor_name": p11.P7_INPUT_NAME,
        "raw_tensor_names": {
            item.identifier: item.p11_raw_name for item in PROJECTIONS
        },
        "input_bf16_le_sha256": _mapping(
            tensors[p11.P7_INPUT_NAME], "P11 input tensor"
        )["bf16_le_sha256"],
        "raw_bf16_le_sha256": {
            item.identifier: _mapping(
                tensors[item.p11_raw_name], f"P11 {item.identifier} raw tensor"
            )["bf16_le_sha256"]
            for item in PROJECTIONS
        },
    }


def _projection_result_document(
    *,
    item: Projection,
    raw_bytes: bytes,
    bias_bytes: bytes,
    staged_bytes: bytes,
    repeated_staged_bytes: bytes,
    p7_actual_bytes: bytes,
    torch: Any,
) -> dict[str, object]:
    return {
        "identifier": item.identifier,
        "module_name": item.module_name,
        "raw_tensor": item.p11_raw_name,
        "bias_tensor": item.bias_name,
        "staged_tensor": item.staged_name,
        "actual_tensor": item.p7_actual_name,
        "checkpoint_bias_key": item.checkpoint_bias_key,
        "shape": {"m": M, "n": item.width, "k": K},
        "raw_bf16_le_sha256": _sha256_bytes(raw_bytes),
        "bias_bf16_le_sha256": _sha256_bytes(bias_bytes),
        "staged_bf16_le_sha256": _sha256_bytes(staged_bytes),
        "actual_bf16_le_sha256": _sha256_bytes(p7_actual_bytes),
        "raw_vs_p11_default_cublas": _metric(raw_bytes, raw_bytes, torch),
        "staged_repeated_bf16_exact": staged_bytes == repeated_staged_bytes,
        "staged_vs_p7_actual_module": _metric(
            p7_actual_bytes, staged_bytes, torch
        ),
    }


def build_manifest(
    *,
    checkpoint: oracle.CheckpointManifest,
    p7_document: Mapping[str, object],
    p7_manifest: Path,
    p7_sidecar: Path,
    p11_document: Mapping[str, object],
    p11_manifest: Path,
    p11_sidecar: Path,
    sidecar: Path,
    tensors: Mapping[str, object],
    torch: Any,
    source_provenance: Mapping[str, object],
    projection_results: Mapping[str, Mapping[str, object]],
    device: str,
    created_at: datetime,
) -> dict[str, object]:
    records = _tensor_manifest(tensors, torch)
    all_raw_exact = all(
        _mapping(projection_results[item.identifier], f"P12 {item.identifier} result")[
            "raw_vs_p11_default_cublas"
        ]["bf16_exact"]
        is True
        for item in PROJECTIONS
    )
    all_repeated = all(
        _mapping(projection_results[item.identifier], f"P12 {item.identifier} result")[
            "staged_repeated_bf16_exact"
        ]
        is True
        for item in PROJECTIONS
    )
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "trace_id": TRACE_ID,
        "performance_claim_eligible": False,
        "vllm_comparison_eligible": False,
        "serving_selector_changed": False,
        "capture_status": "captured",
        "quality_pass": bool(all_raw_exact and all_repeated),
        "created_at": created_at.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "scope": {
            "endpoints": [
                "layer0.q_proj.raw_default_cublas_then_staged_bias",
                "layer0.k_proj.raw_default_cublas_then_staged_bias",
                "layer0.v_proj.raw_default_cublas_then_staged_bias",
            ],
            "primary_profile": "explicit_fp32_bias_add_bf16",
            "actual_module_output": "observational-not-primary-profile",
            "model_id": oracle.MODEL_ID,
            "model_revision": oracle.MODEL_REVISION,
            "shape": {"m": M, "k": K, "q_n": Q_WIDTH, "kv_n": KV_WIDTH},
            "offline_only": True,
            "serving_path": False,
        },
        "producer": {
            "implementation_id": IMPLEMENTATION_ID,
            "runtime_dependency_class": "offline-python-reference",
            "torch_version": str(torch.__version__),
            "runtime_cuda_version": str(torch.version.cuda or "unknown"),
            "device": str(device),
            "staged_bias_operator": (
                "bf16(raw_default_cublas.to(float32)+"
                "checkpoint_bf16_bias.to(float32))"
            ),
        },
        "p7_binding": _p7_binding_document(p7_document, p7_manifest, p7_sidecar),
        "p11_binding": _p11_binding_document(
            p11_document, p11_manifest, p11_sidecar
        ),
        "model": {
            "checkpoint_path": str(checkpoint.root),
            "checkpoint_receipt_filename": checkpoint.receipt.path,
            "checkpoint_receipt_sha256": checkpoint.receipt.sha256,
        },
        "projection_results": {
            item.identifier: dict(projection_results[item.identifier])
            for item in PROJECTIONS
        },
        "comparisons": {
            "primary_profile": "explicit_fp32_bias_add_bf16",
            "raw_qkv_vs_p11_default_cublas_exact": all_raw_exact,
            "staged_qkv_repeated_bf16_exact": all_repeated,
            "actual_module_output_is_observational": True,
            "staged_vs_p7_actual_module": {
                item.identifier: dict(
                    _mapping(
                        projection_results[item.identifier],
                        f"P12 {item.identifier} result",
                    )["staged_vs_p7_actual_module"]
                )
                for item in PROJECTIONS
            },
        },
        "provenance": {"source_repository": dict(source_provenance)},
        "sidecar": {
            "path": sidecar.name,
            "sha256": _sha256_file(sidecar),
            "format": "safetensors",
            "tensor_count": len(TRACE_TENSORS),
        },
        "tensors": records,
    }
    validate_manifest(document)
    return document


def _validate_source_provenance(value: object) -> None:
    source = _mapping(value, "P12 source provenance")
    _exact_keys(
        source, {"git_revision", "source_dirty", "source_status_sha256", "sources"}, "P12 source provenance"
    )
    if (
        source["source_dirty"] is not False
        or oracle.GIT_REVISION_RE.fullmatch(_string(source["git_revision"], "P12 Git revision"))
        is None
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 source provenance differs")
    _sha256(source["source_status_sha256"], "P12 source status SHA-256")
    sources = _mapping(source["sources"], "P12 source records")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 source records differ")
    for name, path in SOURCE_PATHS.items():
        record = _mapping(sources[name], f"P12 source {name}")
        _exact_keys(record, {"path", "sha256"}, f"P12 source {name}")
        if record["path"] != path:
            raise Qwen3BP2051QkvStagedBiasTraceError("P12 source path differs")
        _sha256(record["sha256"], f"P12 source {name} SHA-256")


def _validate_p7_binding(value: object) -> Mapping[str, object]:
    binding = _mapping(value, "P12 P7 binding")
    _exact_keys(
        binding,
        {
            "manifest_filename",
            "manifest_sha256",
            "sidecar_filename",
            "sidecar_sha256",
            "checkpoint_receipt_sha256",
            "source_revision",
            "input_tensor_key",
            "actual_tensor_keys",
            "actual_bf16_le_sha256",
        },
        "P12 P7 binding",
    )
    if (
        binding["manifest_sha256"] != P7_MANIFEST_SHA256
        or binding["sidecar_sha256"] != P7_SIDECAR_SHA256
        or binding["input_tensor_key"] != p11.P7_INPUT_KEY
        or binding["actual_tensor_keys"]
        != {item.identifier: f"trace/layer0/{item.module_name}" for item in PROJECTIONS}
        or oracle.GIT_REVISION_RE.fullmatch(
            _string(binding["source_revision"], "P12 P7 source revision")
        )
        is None
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 P7 binding differs")
    for field in (
        "manifest_filename",
        "sidecar_filename",
        "checkpoint_receipt_sha256",
    ):
        _string(binding[field], f"P12 P7 {field}")
    _sha256(binding["checkpoint_receipt_sha256"], "P12 P7 checkpoint receipt")
    hashes = _mapping(binding["actual_bf16_le_sha256"], "P12 P7 actual hashes")
    if set(hashes) != {item.identifier for item in PROJECTIONS}:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 P7 hashes differ")
    for item in PROJECTIONS:
        _sha256(hashes[item.identifier], f"P12 P7 {item.identifier} hash")
    return binding


def _validate_p11_binding(value: object) -> Mapping[str, object]:
    binding = _mapping(value, "P12 P11 binding")
    _exact_keys(
        binding,
        {
            "manifest_filename",
            "manifest_sha256",
            "sidecar_filename",
            "sidecar_sha256",
            "checkpoint_receipt_sha256",
            "source_revision",
            "input_tensor_name",
            "raw_tensor_names",
            "input_bf16_le_sha256",
            "raw_bf16_le_sha256",
        },
        "P12 P11 binding",
    )
    if (
        binding["manifest_sha256"] != P11_MANIFEST_SHA256
        or binding["sidecar_sha256"] != P11_SIDECAR_SHA256
        or binding["input_tensor_name"] != p11.P7_INPUT_NAME
        or binding["raw_tensor_names"]
        != {item.identifier: item.p11_raw_name for item in PROJECTIONS}
        or oracle.GIT_REVISION_RE.fullmatch(
            _string(binding["source_revision"], "P12 P11 source revision")
        )
        is None
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 P11 binding differs")
    for field in (
        "manifest_filename",
        "sidecar_filename",
        "checkpoint_receipt_sha256",
        "input_bf16_le_sha256",
    ):
        _string(binding[field], f"P12 P11 {field}")
    _sha256(binding["checkpoint_receipt_sha256"], "P12 P11 checkpoint receipt")
    _sha256(binding["input_bf16_le_sha256"], "P12 P11 input hash")
    hashes = _mapping(binding["raw_bf16_le_sha256"], "P12 P11 raw hashes")
    if set(hashes) != {item.identifier for item in PROJECTIONS}:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 P11 hashes differ")
    for item in PROJECTIONS:
        _sha256(hashes[item.identifier], f"P12 P11 {item.identifier} hash")
    return binding


def validate_manifest(document: Mapping[str, object]) -> None:
    _exact_keys(
        document,
        {
            "schema_version",
            "artifact_kind",
            "trace_id",
            "performance_claim_eligible",
            "vllm_comparison_eligible",
            "serving_selector_changed",
            "capture_status",
            "quality_pass",
            "created_at",
            "scope",
            "producer",
            "p7_binding",
            "p11_binding",
            "model",
            "projection_results",
            "comparisons",
            "provenance",
            "sidecar",
            "tensors",
        },
        "P12 manifest",
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
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 manifest identity differs")
    try:
        oracle._validate_utc_text(document["created_at"], "P12 created_at")
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error
    scope = _mapping(document["scope"], "P12 scope")
    _exact_keys(
        scope,
        {
            "endpoints",
            "primary_profile",
            "actual_module_output",
            "model_id",
            "model_revision",
            "shape",
            "offline_only",
            "serving_path",
        },
        "P12 scope",
    )
    if (
        scope["endpoints"]
        != [
            "layer0.q_proj.raw_default_cublas_then_staged_bias",
            "layer0.k_proj.raw_default_cublas_then_staged_bias",
            "layer0.v_proj.raw_default_cublas_then_staged_bias",
        ]
        or scope["primary_profile"] != "explicit_fp32_bias_add_bf16"
        or scope["actual_module_output"] != "observational-not-primary-profile"
        or scope["model_id"] != oracle.MODEL_ID
        or scope["model_revision"] != oracle.MODEL_REVISION
        or scope["shape"] != {"m": M, "k": K, "q_n": Q_WIDTH, "kv_n": KV_WIDTH}
        or scope["offline_only"] is not True
        or scope["serving_path"] is not False
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 scope differs")
    producer = _mapping(document["producer"], "P12 producer")
    _exact_keys(
        producer,
        {
            "implementation_id",
            "runtime_dependency_class",
            "torch_version",
            "runtime_cuda_version",
            "device",
            "staged_bias_operator",
        },
        "P12 producer",
    )
    if (
        producer["implementation_id"] != IMPLEMENTATION_ID
        or producer["runtime_dependency_class"] != "offline-python-reference"
        or producer["staged_bias_operator"]
        != "bf16(raw_default_cublas.to(float32)+checkpoint_bf16_bias.to(float32))"
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 producer differs")
    for field in ("torch_version", "runtime_cuda_version", "device"):
        _string(producer[field], f"P12 producer {field}")
    p7_binding = _validate_p7_binding(document["p7_binding"])
    p11_binding = _validate_p11_binding(document["p11_binding"])
    model = _mapping(document["model"], "P12 model")
    _exact_keys(
        model,
        {"checkpoint_path", "checkpoint_receipt_filename", "checkpoint_receipt_sha256"},
        "P12 model",
    )
    _string(model["checkpoint_path"], "P12 checkpoint path")
    if model["checkpoint_receipt_filename"] != oracle.CHECKPOINT_RECEIPT_FILENAME:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "P12 checkpoint receipt filename differs"
        )
    _sha256(model["checkpoint_receipt_sha256"], "P12 checkpoint receipt")
    if (
        p7_binding["checkpoint_receipt_sha256"]
        != model["checkpoint_receipt_sha256"]
        or p11_binding["checkpoint_receipt_sha256"]
        != model["checkpoint_receipt_sha256"]
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 checkpoint lineage differs")
    results = _mapping(document["projection_results"], "P12 projection results")
    if set(results) != {item.identifier for item in PROJECTIONS}:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 projection result set differs")
    comparisons = _mapping(document["comparisons"], "P12 comparisons")
    _exact_keys(
        comparisons,
        {
            "primary_profile",
            "raw_qkv_vs_p11_default_cublas_exact",
            "staged_qkv_repeated_bf16_exact",
            "actual_module_output_is_observational",
            "staged_vs_p7_actual_module",
        },
        "P12 comparisons",
    )
    actual_metrics = _mapping(
        comparisons["staged_vs_p7_actual_module"], "P12 actual comparisons"
    )
    all_raw_exact = True
    all_repeated = True
    for item in PROJECTIONS:
        result = _mapping(results[item.identifier], f"P12 {item.identifier} result")
        _exact_keys(
            result,
            {
                "identifier",
                "module_name",
                "raw_tensor",
                "bias_tensor",
                "staged_tensor",
                "actual_tensor",
                "checkpoint_bias_key",
                "shape",
                "raw_bf16_le_sha256",
                "bias_bf16_le_sha256",
                "staged_bf16_le_sha256",
                "actual_bf16_le_sha256",
                "raw_vs_p11_default_cublas",
                "staged_repeated_bf16_exact",
                "staged_vs_p7_actual_module",
            },
            f"P12 {item.identifier} result",
        )
        if (
            result["identifier"] != item.identifier
            or result["module_name"] != item.module_name
            or result["raw_tensor"] != item.p11_raw_name
            or result["bias_tensor"] != item.bias_name
            or result["staged_tensor"] != item.staged_name
            or result["actual_tensor"] != item.p7_actual_name
            or result["checkpoint_bias_key"] != item.checkpoint_bias_key
            or result["shape"] != {"m": M, "n": item.width, "k": K}
            or not isinstance(result["staged_repeated_bf16_exact"], bool)
        ):
            raise Qwen3BP2051QkvStagedBiasTraceError(
                f"P12 {item.identifier} result differs"
            )
        for field in (
            "raw_bf16_le_sha256",
            "bias_bf16_le_sha256",
            "staged_bf16_le_sha256",
            "actual_bf16_le_sha256",
        ):
            _sha256(result[field], f"P12 {item.identifier} {field}")
        raw_metric = _validate_metric(
            result["raw_vs_p11_default_cublas"],
            f"P12 {item.identifier} raw/P11 comparison",
            M * item.width,
        )
        observed = _validate_metric(
            result["staged_vs_p7_actual_module"],
            f"P12 {item.identifier} staged/P7 comparison",
            M * item.width,
        )
        if actual_metrics.get(item.identifier) != observed:
            raise Qwen3BP2051QkvStagedBiasTraceError(
                "P12 staged/actual comparison binding differs"
            )
        all_raw_exact = all_raw_exact and raw_metric["bf16_exact"] is True
        all_repeated = all_repeated and result["staged_repeated_bf16_exact"] is True
    if (
        comparisons["primary_profile"] != "explicit_fp32_bias_add_bf16"
        or comparisons["raw_qkv_vs_p11_default_cublas_exact"] is not all_raw_exact
        or comparisons["staged_qkv_repeated_bf16_exact"] is not all_repeated
        or comparisons["actual_module_output_is_observational"] is not True
        or document["quality_pass"] is not bool(all_raw_exact and all_repeated)
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 quality result differs")
    provenance = _mapping(document["provenance"], "P12 provenance")
    _exact_keys(provenance, {"source_repository"}, "P12 provenance")
    _validate_source_provenance(provenance["source_repository"])
    sidecar = _mapping(document["sidecar"], "P12 sidecar")
    _exact_keys(sidecar, {"path", "sha256", "format", "tensor_count"}, "P12 sidecar")
    if (
        Path(_string(sidecar["path"], "P12 sidecar path")).name != sidecar["path"]
        or sidecar["format"] != "safetensors"
        or sidecar["tensor_count"] != len(TRACE_TENSORS)
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 sidecar metadata differs")
    _sha256(sidecar["sha256"], "P12 sidecar SHA-256")
    tensors = _mapping(document["tensors"], "P12 tensors")
    if set(tensors) != set(TRACE_TENSORS):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 tensor set differs")
    for name, shape in _expected_shapes().items():
        record = _mapping(tensors[name], f"P12 tensor {name}")
        _exact_keys(
            record,
            {
                "key",
                "shape",
                "dtype",
                "canonical_byte_order",
                "bf16_le_bytes",
                "bf16_le_sha256",
            },
            f"P12 tensor {name}",
        )
        if (
            record["key"] != _sidecar_key(name)
            or record["shape"] != list(shape)
            or record["dtype"] != "bfloat16"
            or record["canonical_byte_order"] != "little-endian-u16"
            or record["bf16_le_bytes"] != _element_count(shape) * BF16_BYTES
        ):
            raise Qwen3BP2051QkvStagedBiasTraceError(
                f"P12 tensor {name} metadata differs"
            )
        _sha256(record["bf16_le_sha256"], f"P12 tensor {name} SHA-256")
    if tensors[P11_INPUT_NAME]["bf16_le_sha256"] != p11_binding["input_bf16_le_sha256"]:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 input tensor lineage differs")
    for item in PROJECTIONS:
        result = _mapping(results[item.identifier], f"P12 {item.identifier} result")
        if (
            result["raw_bf16_le_sha256"]
            != tensors[item.p11_raw_name]["bf16_le_sha256"]
            or result["bias_bf16_le_sha256"]
            != tensors[item.bias_name]["bf16_le_sha256"]
            or result["staged_bf16_le_sha256"]
            != tensors[item.staged_name]["bf16_le_sha256"]
            or result["actual_bf16_le_sha256"]
            != tensors[item.p7_actual_name]["bf16_le_sha256"]
            or result["raw_bf16_le_sha256"]
            != p11_binding["raw_bf16_le_sha256"][item.identifier]
            or result["actual_bf16_le_sha256"]
            != p7_binding["actual_bf16_le_sha256"][item.identifier]
        ):
            raise Qwen3BP2051QkvStagedBiasTraceError(
                f"P12 {item.identifier} tensor lineage differs"
            )


def validate_sidecar_against_manifest(
    document: Mapping[str, object], sidecar_path: Path
) -> None:
    validate_manifest(document)
    sidecar = _regular_file(sidecar_path.expanduser(), "P12 sidecar")
    metadata = _mapping(document["sidecar"], "P12 sidecar")
    if sidecar.name != metadata["path"] or _sha256_file(sidecar) != metadata["sha256"]:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 sidecar binding differs")
    try:
        header, data_start, size = p7._read_safetensors_header(sidecar)
    except p7.Qwen3BP2051ProjectionTraceError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error
    if set(header) - {"__metadata__"} != {_sidecar_key(name) for name in TRACE_TENSORS}:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 sidecar tensor set differs")
    tensors = _mapping(document["tensors"], "P12 tensors")
    ranges: list[tuple[int, int]] = []
    try:
        with sidecar.open("rb") as source:
            for name in TRACE_TENSORS:
                record = _mapping(tensors[name], f"P12 tensor {name}")
                entry = _mapping(header[record["key"]], f"P12 sidecar {name}")
                _exact_keys(
                    entry, {"dtype", "shape", "data_offsets"}, f"P12 sidecar {name}"
                )
                offsets = entry["data_offsets"]
                if (
                    entry["dtype"] != "BF16"
                    or entry["shape"] != record["shape"]
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
                    or offsets[0] < 0
                    or offsets[1] < offsets[0]
                    or offsets[1] - offsets[0] != record["bf16_le_bytes"]
                    or data_start + offsets[1] > size
                ):
                    raise Qwen3BP2051QkvStagedBiasTraceError(
                        f"P12 sidecar {name} layout differs"
                    )
                source.seek(data_start + offsets[0])
                raw = source.read(record["bf16_le_bytes"])
                if (
                    len(raw) != record["bf16_le_bytes"]
                    or _sha256_bytes(raw) != record["bf16_le_sha256"]
                ):
                    raise Qwen3BP2051QkvStagedBiasTraceError(
                        f"P12 sidecar {name} raw BF16 hash differs"
                    )
                p7._validate_finite_bf16(raw, f"P12 sidecar {name}")
                ranges.append((offsets[0], offsets[1]))
    except (OSError, KeyError, p7.Qwen3BP2051ProjectionTraceError) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError("cannot read P12 sidecar") from error
    expected_offset = 0
    for start, end in sorted(ranges):
        if start != expected_offset:
            raise Qwen3BP2051QkvStagedBiasTraceError(
                "P12 sidecar tensor ranges are not contiguous"
            )
        expected_offset = end
    if data_start + expected_offset != size:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 sidecar payload size differs")


def _load_manifest(path: Path) -> dict[str, object]:
    source = _regular_file(path.expanduser(), "P12 manifest")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BP2051QkvStagedBiasTraceError("cannot parse P12 manifest") from error
    result = dict(_mapping(document, "P12 manifest"))
    validate_manifest(result)
    return result


def produce_hf_trace(
    *,
    checkpoint_path: Path,
    p7_manifest_path: Path,
    p7_sidecar_path: Path,
    p11_manifest_path: Path,
    p11_sidecar_path: Path,
    manifest_path: Path,
    sidecar_path: Path,
    repo_root: Path,
    device: str,
    created_at: datetime | None = None,
    sidecar_writer: SidecarWriter = _default_sidecar_writer,
) -> dict[str, object]:
    manifest, sidecar = _output_paths(manifest_path, sidecar_path, repo_root)
    (
        p7_document,
        p7_manifest,
        p7_sidecar,
        p11_document,
        p11_manifest,
        p11_sidecar,
    ) = _load_pinned_artifacts(
        p7_manifest_path=p7_manifest_path,
        p7_sidecar_path=p7_sidecar_path,
        p11_manifest_path=p11_manifest_path,
        p11_sidecar_path=p11_sidecar_path,
    )
    try:
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error
    if (
        checkpoint.receipt.sha256 != _mapping(p7_document["model"], "P7 model")["checkpoint_receipt_sha256"]
        or checkpoint.receipt.sha256 != _mapping(p11_document["model"], "P11 model")["checkpoint_receipt_sha256"]
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "P12 checkpoint receipt lineage differs"
        )
    provenance = collect_source_provenance(repo_root)
    try:
        import torch
    except ImportError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(
            "pinned torch is required for P12 produce"
        ) from error
    p11_requested: dict[str, tuple[str, tuple[int, ...]]] = {
        P11_INPUT_NAME: (_sidecar_key(p11.P7_INPUT_NAME), (M, K))
    }
    p7_requested: dict[str, tuple[str, tuple[int, ...]]] = {}
    for item in PROJECTIONS:
        p11_requested[item.p11_raw_name] = (
            _sidecar_key(item.p11_raw_name),
            (M, item.width),
        )
        p7_requested[item.p7_actual_name] = (
            f"trace/layer0/{item.module_name}",
            (M, item.width),
        )
    p11_tensors = _load_sidecar_tensors(
        sidecar=p11_sidecar, requested=p11_requested, torch=torch, label="P11"
    )
    p7_tensors = _load_sidecar_tensors(
        sidecar=p7_sidecar, requested=p7_requested, torch=torch, label="P7"
    )
    tensors: dict[str, object] = {P11_INPUT_NAME: p11_tensors[P11_INPUT_NAME]}
    results: dict[str, Mapping[str, object]] = {}
    sidecar_written = False
    try:
        for item in PROJECTIONS:
            raw = p11_tensors[item.p11_raw_name]
            raw_bytes = p11._validate_tensor(raw, (M, item.width), torch, item.p11_raw_name)
            bias = _bf16_tensor_from_le(
                _checkpoint_bias_raw(checkpoint, item), (item.width,), torch
            )
            bias_bytes = p11._validate_tensor(bias, (item.width,), torch, item.bias_name)
            staged, repeated = _explicit_staged_bias(
                raw=raw, bias=bias, device=device, torch=torch
            )
            staged_bytes = p11._validate_tensor(
                staged, (M, item.width), torch, item.staged_name
            )
            repeated_bytes = p11._validate_tensor(
                repeated, (M, item.width), torch, f"repeat {item.staged_name}"
            )
            actual = p7_tensors[item.p7_actual_name]
            actual_bytes = p11._validate_tensor(
                actual, (M, item.width), torch, item.p7_actual_name
            )
            tensors[item.p11_raw_name] = raw
            tensors[item.bias_name] = bias
            tensors[item.staged_name] = staged
            tensors[item.p7_actual_name] = actual
            results[item.identifier] = _projection_result_document(
                item=item,
                raw_bytes=raw_bytes,
                bias_bytes=bias_bytes,
                staged_bytes=staged_bytes,
                repeated_staged_bytes=repeated_bytes,
                p7_actual_bytes=actual_bytes,
                torch=torch,
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
            p11_document=p11_document,
            p11_manifest=p11_manifest,
            p11_sidecar=p11_sidecar,
            sidecar=sidecar,
            tensors=tensors,
            torch=torch,
            source_provenance=provenance,
            projection_results=results,
            device=device,
            created_at=created_at or datetime.now(timezone.utc),
        )
        validate_sidecar_against_manifest(document, sidecar)
        try:
            oracle.write_artifact_exclusive(manifest, document)
        except oracle.Qwen3BServingOracleError as error:
            raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error
        return document
    except BaseException:
        if sidecar_written:
            try:
                sidecar.unlink()
            except OSError:
                pass
        raise
    finally:
        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()


def validate_bindings(
    *,
    manifest_path: Path,
    sidecar_path: Path,
    p7_manifest_path: Path,
    p7_sidecar_path: Path,
    p11_manifest_path: Path,
    p11_sidecar_path: Path,
    checkpoint_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    document = _load_manifest(manifest_path)
    validate_sidecar_against_manifest(document, sidecar_path)
    (
        p7_document,
        p7_manifest,
        p7_sidecar,
        p11_document,
        p11_manifest,
        p11_sidecar,
    ) = _load_pinned_artifacts(
        p7_manifest_path=p7_manifest_path,
        p7_sidecar_path=p7_sidecar_path,
        p11_manifest_path=p11_manifest_path,
        p11_sidecar_path=p11_sidecar_path,
    )
    if (
        document["p7_binding"]
        != _p7_binding_document(p7_document, p7_manifest, p7_sidecar)
        or document["p11_binding"]
        != _p11_binding_document(p11_document, p11_manifest, p11_sidecar)
    ):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 input artifact binding differs")
    try:
        checkpoint = oracle.inspect_checkpoint(checkpoint_path)
    except oracle.Qwen3BServingOracleError as error:
        raise Qwen3BP2051QkvStagedBiasTraceError(str(error)) from error
    if _mapping(document["model"], "P12 model")["checkpoint_receipt_sha256"] != checkpoint.receipt.sha256:
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 checkpoint receipt differs")
    observed_sources = _mapping(
        _mapping(document["provenance"], "P12 provenance")["source_repository"],
        "P12 source provenance",
    )
    if observed_sources != collect_source_provenance(repo_root):
        raise Qwen3BP2051QkvStagedBiasTraceError("P12 source provenance differs")
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_p2051_qkv_staged_bias_trace",
        description="offline Qwen2.5-3B P2051 Q/K/V staged-bias trace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser(
        "produce", help="write a create-only P12 BF16 sidecar and manifest"
    )
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--p7-manifest", type=Path, required=True)
    produce.add_argument("--p7-sidecar", type=Path, required=True)
    produce.add_argument("--p11-manifest", type=Path, required=True)
    produce.add_argument("--p11-sidecar", type=Path, required=True)
    produce.add_argument("--manifest", type=Path, required=True)
    produce.add_argument("--sidecar", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    validate = commands.add_parser(
        "validate", help="validate P12/P11/P7 bindings without CUDA"
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--sidecar", type=Path, required=True)
    validate.add_argument("--p7-manifest", type=Path, required=True)
    validate.add_argument("--p7-sidecar", type=Path, required=True)
    validate.add_argument("--p11-manifest", type=Path, required=True)
    validate.add_argument("--p11-sidecar", type=Path, required=True)
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
                p11_manifest_path=args.p11_manifest,
                p11_sidecar_path=args.p11_sidecar,
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
            p11_manifest_path=args.p11_manifest,
            p11_sidecar_path=args.p11_sidecar,
            checkpoint_path=args.checkpoint,
            repo_root=args.repo_root,
        )
        print(f"validated {document['artifact_kind']}: {document['sidecar']['path']}")
        return 0
    except (
        OSError,
        RuntimeError,
        Qwen3BP2051QkvStagedBiasTraceError,
        oracle.Qwen3BServingOracleError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
