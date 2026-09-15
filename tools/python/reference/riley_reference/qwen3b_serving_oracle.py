"""Offline-only HF eager raw-logit fingerprint for the N06-A Qwen2.5-3B input.

This diagnostic is deliberately separate from the 135M calibration lane and
from the Rust serving process.  It accepts the immutable P2048 N06-A workload,
uses its supplied token IDs directly (without a tokenizer), and produces a
create-only JSON fingerprint of one cache-free eager BF16 forward pass.

``produce`` lazily imports PyTorch and Transformers only after the workload,
checkpoint, output, and source-provenance contracts have been checked.
``validate`` is standard-library-only and never loads a model or CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

SCHEMA_VERSION = "riley.qwen3b-hf-eager-raw-logits.v2"
ARTIFACT_KIND = "qwen2.5-3b-hf-eager-bf16-p2048-raw-logits-v2"
IMPLEMENTATION_ID = "riley-python-qwen3b-hf-eager-raw-logits-v2"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
WORKLOAD_SCHEMA_VERSION = "riley.n06a-d128-serving-workload.v1"
WORKLOAD_SHA256 = "7a0a8fec31d45e397e1ec57335fa1c9de2d3da7daa9a32e9002a62763c05261e"
WORKLOAD_CASE = "qwen3b-c8-p2048-o128"
PROMPT_TOKEN_COUNT = 2_048
PROMPT_TOKEN_IDS_SHA256 = (
    "56619bc156fb385345c12e523c71604fa9ef8ad3c52d9c913d0f6ff09f1c1fd9"
)
OUTPUT_TOKEN_COUNT = 128
OUTPUT_TOKEN_IDS_SHA256 = (
    "83ffd904307abe16d04e2ad022d5113498337532245c170d3a729ffbb5a82df1"
)
EXPECTED_OUTPUT_PREFIX = (374, 264, 198, 750, 198, 16, 17, 17)
EXPECTED_PROMPT_TOKEN_ID = 3_409
MODEL_VOCABULARY_SIZE = 151_936
ADDRESSABLE_TOKEN_COUNT = 151_665
RAW_LOGIT_BYTES = MODEL_VOCABULARY_SIZE * 2
ADDRESSABLE_LOGIT_BYTES = ADDRESSABLE_TOKEN_COUNT * 2
NON_ADDRESSABLE_LOGIT_BYTES = RAW_LOGIT_BYTES - ADDRESSABLE_LOGIT_BYTES
TOP_K = 32
PROBE_IDS = tuple(
    sorted(
        {
            0,
            304,
            374,
            3_409,
            *EXPECTED_OUTPUT_PREFIX,
            151_643,
            151_645,
            151_664,
            151_665,
            151_935,
        }
    )
)
MAX_WORKLOAD_BYTES = 4 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")

SOURCE_PATHS = {
    "oracle_module": "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
    "python_project": "tools/python/reference/pyproject.toml",
    "dependency_lock": "tools/python/reference/uv.lock",
}

EXPECTED_CHECKPOINT_METADATA = {
    "config.json": {
        "size_bytes": 661,
        "sha256": "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b",
    },
    "tokenizer.json": {
        "size_bytes": 7_031_645,
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    },
    "tokenizer_config.json": {
        "size_bytes": 7_305,
        "sha256": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    },
    "model.safetensors.index.json": {
        "size_bytes": 35_581,
        "sha256": "bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56",
    },
}
CHECKPOINT_RECEIPT_FILENAME = "riley-checkpoint.json"
CHECKPOINT_RECEIPT_BYTES = 1_054
CHECKPOINT_RECEIPT_SHA256 = (
    "f85648630f4aef16ddc62b04848ce082b577d08d6e14915c2ccd402fdbd377f8"
)
EXPECTED_WEIGHT_FILES = frozenset(
    {
        (
            3_968_658_944,
            "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
        ),
        (
            2_203_268_048,
            "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
        ),
    }
)


class Qwen3BServingOracleError(ValueError):
    """The diagnostic's immutable input or output contract was violated."""


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str

    def document(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ServingWorkload:
    source_path: Path
    source_bytes: int
    source_sha256: str
    case: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_text: str


@dataclass(frozen=True)
class CheckpointManifest:
    root: Path
    receipt: FileRecord
    files: tuple[FileRecord, ...]


@dataclass(frozen=True)
class RawLogitCapture:
    raw_bf16_le_sha256: str
    addressable_bf16_le_sha256: str
    non_addressable_bf16_le_sha256: str
    argmax_token_id: int
    argmax_value_bf16_as_f32: float
    top_token_ids: tuple[int, ...]
    top_values_bf16_as_f32: tuple[float, ...]
    probe_values_bf16_as_f32: Mapping[int, float]


class RawLogitBackend(Protocol):
    producer_metadata: Mapping[str, object]

    def capture(self, input_token_ids: Sequence[int]) -> RawLogitCapture: ...

    def close(self) -> None: ...


BackendFactory = Callable[..., RawLogitBackend]
SourceProvenanceFactory = Callable[[Path], dict[str, object]]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise Qwen3BServingOracleError("token IDs must be canonical U32 values")
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise Qwen3BServingOracleError("token ID is outside canonical U32 range")
        digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise Qwen3BServingOracleError("artifact timestamp must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _validate_utc_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Qwen3BServingOracleError(f"{label} must be canonical UTC text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise Qwen3BServingOracleError(f"{label} is invalid") from error
    if _utc_text(parsed) != value:
        raise Qwen3BServingOracleError(f"{label} is not canonical to whole seconds")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise Qwen3BServingOracleError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise Qwen3BServingOracleError(
            f"{label} fields differ: missing={missing!r} extra={extra!r}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Qwen3BServingOracleError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Qwen3BServingOracleError(f"{label} must be lowercase SHA-256 hex")
    return value


def _require_u32_ids(
    value: object,
    *,
    label: str,
    count: int,
    upper_bound: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise Qwen3BServingOracleError(f"{label} must contain exactly {count} U32 IDs")
    output: list[int] = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise Qwen3BServingOracleError(f"{label}[{index}] must be an integer")
        if not 0 <= token_id < upper_bound:
            raise Qwen3BServingOracleError(
                f"{label}[{index}] is outside [0, {upper_bound})"
            )
        output.append(token_id)
    return tuple(output)


def _require_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen3BServingOracleError(f"{label} must be a finite JSON number")
    output = float(value)
    if not math.isfinite(output):
        raise Qwen3BServingOracleError(f"{label} must be finite")
    return output


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BServingOracleError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Qwen3BServingOracleError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BServingOracleError(f"cannot resolve {label}: {path}") from error


def _regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Qwen3BServingOracleError(f"cannot stat {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise Qwen3BServingOracleError(
            f"{label} must be a directory, not a symlink: {path}"
        )
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise Qwen3BServingOracleError(f"cannot resolve {label}: {path}") from error


def _parse_json(raw: bytes, label: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen3BServingOracleError(f"{label} is not valid UTF-8 JSON") from error
    return _require_mapping(decoded, label)


def _validate_workload_document(
    document: Mapping[str, object],
    *,
    source_sha256: str,
    source_path: Path,
    source_bytes: int,
) -> ServingWorkload:
    _require_exact_keys(
        document,
        {
            "schema_version",
            "case",
            "model_id",
            "model_revision",
            "prompt",
            "prompt_token_ids",
            "output_token_ids",
            "output_text",
            "finish_reason",
            "sampling",
        },
        "workload",
    )
    if document["schema_version"] != WORKLOAD_SCHEMA_VERSION:
        raise Qwen3BServingOracleError("workload schema version differs")
    if document["case"] != WORKLOAD_CASE:
        raise Qwen3BServingOracleError(
            "workload case differs from the pinned C8 P2048 case"
        )
    if document["model_id"] != MODEL_ID or document["model_revision"] != MODEL_REVISION:
        raise Qwen3BServingOracleError("workload model identity differs")
    prompt = _require_string(document["prompt"], "workload.prompt")
    prompt_token_ids = _require_u32_ids(
        document["prompt_token_ids"],
        label="workload.prompt_token_ids",
        count=PROMPT_TOKEN_COUNT,
        upper_bound=ADDRESSABLE_TOKEN_COUNT,
    )
    if _token_ids_sha256(prompt_token_ids) != PROMPT_TOKEN_IDS_SHA256:
        raise Qwen3BServingOracleError("workload P2048 token-ID bytes differ")
    if any(token_id != EXPECTED_PROMPT_TOKEN_ID for token_id in prompt_token_ids):
        raise Qwen3BServingOracleError(
            "workload P2048 token IDs differ from the pinned repeat"
        )
    output_token_ids = _require_u32_ids(
        document["output_token_ids"],
        label="workload.output_token_ids",
        count=OUTPUT_TOKEN_COUNT,
        upper_bound=ADDRESSABLE_TOKEN_COUNT,
    )
    if _token_ids_sha256(output_token_ids) != OUTPUT_TOKEN_IDS_SHA256:
        raise Qwen3BServingOracleError("workload output token-ID bytes differ")
    if tuple(output_token_ids[: len(EXPECTED_OUTPUT_PREFIX)]) != EXPECTED_OUTPUT_PREFIX:
        raise Qwen3BServingOracleError("workload output prefix differs")
    output_text = _require_string(document["output_text"], "workload.output_text")
    if document["finish_reason"] != "length":
        raise Qwen3BServingOracleError("workload finish reason must be length")
    sampling = _require_mapping(document["sampling"], "workload.sampling")
    _require_exact_keys(sampling, {"temperature", "top_p"}, "workload.sampling")
    if (
        _require_finite_float(sampling["temperature"], "workload.sampling.temperature")
        != 0.0
    ):
        raise Qwen3BServingOracleError("workload temperature must be 0.0")
    if _require_finite_float(sampling["top_p"], "workload.sampling.top_p") != 1.0:
        raise Qwen3BServingOracleError("workload top_p must be 1.0")
    return ServingWorkload(
        source_path=source_path,
        source_bytes=source_bytes,
        source_sha256=source_sha256,
        case=WORKLOAD_CASE,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        output_text=output_text,
    )


def load_workload(path: Path) -> ServingWorkload:
    """Read the exact immutable C8/P2048 serving workload before CUDA starts."""

    source_path = _regular_file(path.expanduser(), "workload")
    try:
        source_bytes = source_path.stat().st_size
    except OSError as error:
        raise Qwen3BServingOracleError(
            f"cannot size workload: {source_path}"
        ) from error
    if not 0 < source_bytes <= MAX_WORKLOAD_BYTES:
        raise Qwen3BServingOracleError(
            "workload byte size is outside the bounded contract"
        )
    raw = source_path.read_bytes()
    source_sha256 = _sha256_bytes(raw)
    if source_sha256 != WORKLOAD_SHA256:
        raise Qwen3BServingOracleError(
            "workload bytes differ from the immutable N06-A C8 P2048 artifact"
        )
    return _validate_workload_document(
        _parse_json(raw, "workload"),
        source_sha256=source_sha256,
        source_path=source_path,
        source_bytes=source_bytes,
    )


def _validate_qwen3b_geometry(config: Mapping[str, object], *, label: str) -> None:
    expected = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 2_048,
        "intermediate_size": 11_008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32_768,
        "vocab_size": MODEL_VOCABULARY_SIZE,
        "tie_word_embeddings": True,
    }
    for field, required in expected.items():
        if config.get(field) != required:
            raise Qwen3BServingOracleError(
                f"{label}.{field} differs from the Qwen2.5-3B contract"
            )


def _validate_checkpoint_config(config: Mapping[str, object]) -> None:
    """Validate execution-relevant fields in the pinned on-disk config.json."""

    _validate_qwen3b_geometry(config, label="checkpoint config")
    expected = {
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "sliding_window": 32_768,
        "use_sliding_window": False,
        "bos_token_id": 151_643,
        "eos_token_id": 151_645,
        "torch_dtype": "bfloat16",
    }
    for field, required in expected.items():
        if config.get(field) != required:
            raise Qwen3BServingOracleError(
                f"checkpoint config.{field} differs from the Qwen2.5-3B contract"
            )


def _loaded_model_config_mapping(config: object) -> Mapping[str, object]:
    exporter = getattr(config, "to_dict", None)
    if not callable(exporter):
        raise Qwen3BServingOracleError("loaded model config does not expose to_dict")
    return _require_mapping(exporter(), "loaded model config")


def _validate_loaded_model_config(config: object) -> None:
    """Validate the HF-normalized config after `from_pretrained` has run."""

    mapping = _loaded_model_config_mapping(config)
    _validate_qwen3b_geometry(mapping, label="loaded model config")
    for field, required in {
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "use_sliding_window": False,
        "bos_token_id": 151_643,
        "eos_token_id": 151_645,
    }.items():
        if mapping.get(field) != required:
            raise Qwen3BServingOracleError(
                f"loaded model config.{field} differs from the Qwen2.5-3B contract"
            )
    rope_parameters = _require_mapping(
        mapping.get("rope_parameters"), "loaded model config.rope_parameters"
    )
    if (
        rope_parameters.get("rope_type") != "default"
        or rope_parameters.get("rope_theta") != 1_000_000.0
    ):
        raise Qwen3BServingOracleError(
            "loaded model config rope parameters differ from the Qwen2.5-3B contract"
        )


def _checkpoint_file_size(root: Path, filename: str) -> int:
    candidate = root / filename
    path = _regular_file(candidate, f"checkpoint file {filename}")
    try:
        return path.stat().st_size
    except OSError as error:
        raise Qwen3BServingOracleError(
            f"cannot size checkpoint file: {filename}"
        ) from error


def _weight_filenames(index: Mapping[str, object]) -> tuple[str, ...]:
    _require_exact_keys(index, {"metadata", "weight_map"}, "safetensors index")
    weights = _require_mapping(index["weight_map"], "safetensors index.weight_map")
    if not weights:
        raise Qwen3BServingOracleError("safetensors index.weight_map is empty")
    filenames: set[str] = set()
    for tensor_name, filename in weights.items():
        _require_string(tensor_name, "safetensors index tensor name")
        if not isinstance(filename, str):
            raise Qwen3BServingOracleError(
                "safetensors index weight filename must be a string"
            )
        pure = PurePosixPath(filename)
        if (
            pure.name != filename
            or filename in {"", ".", ".."}
            or not filename.endswith(".safetensors")
        ):
            raise Qwen3BServingOracleError(
                "safetensors index contains an unsafe shard path"
            )
        filenames.add(filename)
    return tuple(sorted(filenames))


def _receipt_files(root: Path) -> tuple[FileRecord, ...]:
    receipt_path = _regular_file(
        root / CHECKPOINT_RECEIPT_FILENAME, "checkpoint receipt"
    )
    try:
        receipt_bytes = receipt_path.stat().st_size
    except OSError as error:
        raise Qwen3BServingOracleError("cannot size checkpoint receipt") from error
    raw = receipt_path.read_bytes()
    if (
        receipt_bytes != CHECKPOINT_RECEIPT_BYTES
        or _sha256_bytes(raw) != CHECKPOINT_RECEIPT_SHA256
    ):
        raise Qwen3BServingOracleError(
            "checkpoint receipt differs from the pinned source"
        )
    receipt = _parse_json(raw, "checkpoint receipt")
    _require_exact_keys(
        receipt,
        {
            "format",
            "source_model",
            "source_revision",
            "converter_revision",
            "transforms",
            "dtype",
            "files",
        },
        "checkpoint receipt",
    )
    if (
        receipt["format"] != "riley-checkpoint-v1"
        or receipt["source_model"] != MODEL_ID
        or receipt["source_revision"] != MODEL_REVISION
        or receipt["dtype"] != "bf16"
        or receipt["transforms"] != []
    ):
        raise Qwen3BServingOracleError("checkpoint receipt contract differs")
    converter_revision = receipt["converter_revision"]
    if converter_revision is not None:
        revision = _require_string(
            converter_revision, "checkpoint receipt converter revision"
        )
        if not (7 <= len(revision) <= 64) or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise Qwen3BServingOracleError(
                "checkpoint receipt converter revision is malformed"
            )
    files_value = receipt["files"]
    if not isinstance(files_value, list) or len(files_value) != 6:
        raise Qwen3BServingOracleError("checkpoint receipt must list six source files")
    files: list[FileRecord] = []
    paths: set[str] = set()
    for item in files_value:
        mapping = _require_mapping(item, "checkpoint receipt file")
        _require_exact_keys(
            mapping, {"path", "bytes", "sha256"}, "checkpoint receipt file"
        )
        filename = _require_string(mapping["path"], "checkpoint receipt file.path")
        if PurePosixPath(filename).name != filename or filename in paths:
            raise Qwen3BServingOracleError(
                "checkpoint receipt file path is unsafe or duplicated"
            )
        paths.add(filename)
        size_bytes = mapping["bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise Qwen3BServingOracleError(
                "checkpoint receipt file bytes must be positive"
            )
        files.append(
            FileRecord(
                path=filename,
                size_bytes=size_bytes,
                sha256=_require_sha256(
                    mapping["sha256"], "checkpoint receipt file.sha256"
                ),
            )
        )
    return tuple(sorted(files, key=lambda record: record.path))


def inspect_checkpoint(path: Path) -> CheckpointManifest:
    """Verify small metadata hashes and shard sizes without rereading 6GB weights."""

    root = _regular_directory(path.expanduser(), "checkpoint")
    receipt = FileRecord(
        path=CHECKPOINT_RECEIPT_FILENAME,
        size_bytes=CHECKPOINT_RECEIPT_BYTES,
        sha256=CHECKPOINT_RECEIPT_SHA256,
    )
    records = _receipt_files(root)
    by_name = {record.path: record for record in records}
    for filename, expected in EXPECTED_CHECKPOINT_METADATA.items():
        record = by_name.get(filename)
        if (
            record is None
            or record.size_bytes != expected["size_bytes"]
            or record.sha256 != expected["sha256"]
        ):
            raise Qwen3BServingOracleError(
                f"checkpoint {filename} differs from the pinned Qwen2.5-3B source"
            )
        if _checkpoint_file_size(root, filename) != record.size_bytes:
            raise Qwen3BServingOracleError(
                f"checkpoint {filename} regular-file size differs from its receipt"
            )
        metadata_path = _regular_file(root / filename, f"checkpoint file {filename}")
        if _sha256_file(metadata_path) != record.sha256:
            raise Qwen3BServingOracleError(
                f"checkpoint {filename} SHA-256 differs from its receipt"
            )
    config_raw = (root / "config.json").read_bytes()
    _validate_checkpoint_config(_parse_json(config_raw, "checkpoint config"))
    index_raw = (root / "model.safetensors.index.json").read_bytes()
    shard_names = _weight_filenames(_parse_json(index_raw, "safetensors index"))
    shard_records = tuple(by_name.get(filename) for filename in shard_names)
    if any(record is None for record in shard_records):
        raise Qwen3BServingOracleError(
            "safetensors index references a shard absent from receipt"
        )
    exact_shard_records = tuple(
        record for record in shard_records if record is not None
    )
    observed_weight_files = frozenset(
        (record.size_bytes, record.sha256) for record in exact_shard_records
    )
    if observed_weight_files != EXPECTED_WEIGHT_FILES or len(exact_shard_records) != 2:
        raise Qwen3BServingOracleError(
            "checkpoint weight shards differ from the pinned Qwen2.5-3B source"
        )
    for record in exact_shard_records:
        if _checkpoint_file_size(root, record.path) != record.size_bytes:
            raise Qwen3BServingOracleError(
                f"checkpoint shard {record.path} regular-file size differs from its receipt"
            )
    expected_file_paths = set(EXPECTED_CHECKPOINT_METADATA) | set(shard_names)
    if set(by_name) != expected_file_paths:
        raise Qwen3BServingOracleError(
            "checkpoint receipt file set differs from the HF source"
        )
    return CheckpointManifest(root=root, receipt=receipt, files=records)


def _query_driver_version(device_index: int) -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Qwen3BServingOracleError(
            "cannot record the selected GPU NVIDIA driver version"
        ) from error
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise Qwen3BServingOracleError(
            "NVIDIA driver probe returned an unexpected row count"
        )
    return rows[0]


def _canonical_bf16_le_bytes(tensor: Any, torch: Any) -> bytes:
    raw = bytes(tensor.detach().contiguous().view(torch.uint8).cpu().tolist())
    if len(raw) != RAW_LOGIT_BYTES:
        raise Qwen3BServingOracleError("HF raw BF16 logit row byte length differs")
    if sys.byteorder == "little":
        return raw
    if sys.byteorder == "big":
        return b"".join(
            raw[offset : offset + 2][::-1] for offset in range(0, len(raw), 2)
        )
    raise Qwen3BServingOracleError("unsupported host byte order for BF16 fingerprint")


class HuggingFaceQwen3BBackend:
    """Lazy, CUDA-only model adapter used solely by this diagnostic CLI."""

    def __init__(
        self,
        *,
        torch: Any,
        model: Any,
        device: Any,
        producer_metadata: Mapping[str, object],
    ) -> None:
        self._torch = torch
        self._model = model
        self._device = device
        self.producer_metadata = dict(producer_metadata)

    @classmethod
    def load(
        cls, *, checkpoint: CheckpointManifest, device: str
    ) -> HuggingFaceQwen3BBackend:
        # Disable Hub and Transformers network fallback before their imports.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in (None, ":4096:8"):
            raise Qwen3BServingOracleError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        try:
            import safetensors
            import torch
            import transformers
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise Qwen3BServingOracleError(
                "the pinned torch, transformers, and safetensors dependencies are required"
            ) from error
        if not torch.cuda.is_available():
            raise Qwen3BServingOracleError("Qwen3B raw-logit oracle requires CUDA")
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise Qwen3BServingOracleError("--device must select a CUDA device")
        device_index = resolved_device.index if resolved_device.index is not None else 0
        torch.cuda.set_device(resolved_device)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint.root,
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(resolved_device)
        try:
            model.eval()
            config = getattr(model, "config", None)
            _validate_loaded_model_config(config)
            if getattr(config, "_attn_implementation", None) != "eager":
                raise Qwen3BServingOracleError(
                    "loaded model did not retain eager attention"
                )
            embeddings = model.get_input_embeddings()
            lm_head = model.get_output_embeddings()
            if embeddings is None or lm_head is None:
                raise Qwen3BServingOracleError(
                    "loaded model lacks tied embedding and LM head"
                )
            if (
                embeddings.weight.dtype != torch.bfloat16
                or lm_head.weight.dtype != torch.bfloat16
            ):
                raise Qwen3BServingOracleError("loaded model weights must be BF16")
            if embeddings.weight.data_ptr() != lm_head.weight.data_ptr():
                raise Qwen3BServingOracleError(
                    "loaded model embedding and LM head are not tied"
                )
            source_path_text = inspect.getsourcefile(type(model))
            if source_path_text is None:
                raise Qwen3BServingOracleError(
                    "cannot resolve Transformers Qwen model source"
                )
            source_path = Path(source_path_text).resolve()
            properties = torch.cuda.get_device_properties(resolved_device)
            producer_metadata = {
                "implementation_id": IMPLEMENTATION_ID,
                "runtime_dependency_class": "offline-python-reference",
                "python_version": platform.python_version(),
                "python_executable_sha256": _sha256_file(
                    Path(sys.executable).resolve()
                ),
                "python_platform_system": platform.system().lower(),
                "python_platform_machine": platform.machine().lower(),
                "torch_version": str(torch.__version__),
                "transformers_version": str(transformers.__version__),
                "safetensors_version": str(safetensors.__version__),
                "selected_device": {
                    "requested": device,
                    "resolved": str(resolved_device),
                    "index": device_index,
                    "name": str(torch.cuda.get_device_name(resolved_device)),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "driver_version": _query_driver_version(device_index),
                    "runtime_cuda_version": str(torch.version.cuda or "unknown"),
                },
                "transformers_model_source": {
                    "module": type(model).__module__,
                    "filename": source_path.name,
                    "sha256": _sha256_file(source_path),
                },
            }
            return cls(
                torch=torch,
                model=model,
                device=resolved_device,
                producer_metadata=producer_metadata,
            )
        except BaseException:
            del model
            torch.cuda.empty_cache()
            raise

    def capture(self, input_token_ids: Sequence[int]) -> RawLogitCapture:
        torch = self._torch
        model = self._model
        if model is None:
            raise Qwen3BServingOracleError("HF Qwen3B backend is closed")
        if tuple(input_token_ids) != (EXPECTED_PROMPT_TOKEN_ID,) * PROMPT_TOKEN_COUNT:
            raise Qwen3BServingOracleError(
                "HF input IDs differ from the pinned P2048 input"
            )
        input_ids = torch.tensor(
            [list(input_token_ids)], dtype=torch.long, device=self._device
        )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            PROMPT_TOKEN_COUNT, device=self._device, dtype=torch.long
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
                raise Qwen3BServingOracleError(
                    "cache-free forward unexpectedly returned KV cache"
                )
            logits = output.logits
            expected_shape = (1, 1, MODEL_VOCABULARY_SIZE)
            if tuple(int(value) for value in logits.shape) != expected_shape:
                raise Qwen3BServingOracleError(
                    "HF logits shape differs from [1, 1, 151936]"
                )
            last_logits = logits[0, -1]
            if last_logits.dtype != torch.bfloat16:
                raise Qwen3BServingOracleError("HF raw last logits are not BF16")
            if not bool(torch.isfinite(last_logits).all().item()):
                raise Qwen3BServingOracleError(
                    "HF raw last logits contain non-finite values"
                )
            raw_bf16_le_bytes = _canonical_bf16_le_bytes(last_logits, torch)
            raw_bf16_le_sha256 = _sha256_bytes(raw_bf16_le_bytes)
            values, token_ids = torch.topk(
                last_logits.float(), k=TOP_K, largest=True, sorted=True
            )
            top_token_ids = tuple(int(value) for value in token_ids.cpu().tolist())
            top_values = tuple(float(value) for value in values.cpu().tolist())
            probes = {
                token_id: float(last_logits[token_id].float().item())
                for token_id in PROBE_IDS
            }
            return RawLogitCapture(
                raw_bf16_le_sha256=raw_bf16_le_sha256,
                addressable_bf16_le_sha256=_sha256_bytes(
                    raw_bf16_le_bytes[:ADDRESSABLE_LOGIT_BYTES]
                ),
                non_addressable_bf16_le_sha256=_sha256_bytes(
                    raw_bf16_le_bytes[ADDRESSABLE_LOGIT_BYTES:]
                ),
                argmax_token_id=top_token_ids[0],
                argmax_value_bf16_as_f32=top_values[0],
                top_token_ids=top_token_ids,
                top_values_bf16_as_f32=top_values,
                probe_values_bf16_as_f32=probes,
            )
        finally:
            del input_ids, attention_mask, position_ids

    def close(self) -> None:
        model = self._model
        if model is None:
            return
        self._model = None
        del model
        self._torch.cuda.empty_cache()


def _source_record(root: Path, relative: str) -> dict[str, object]:
    path = _regular_file(root / relative, f"source file {relative}")
    return {"path": relative, "sha256": _sha256_file(path)}


def collect_source_provenance(repo_root: Path) -> dict[str, object]:
    """Record the exact standalone diagnostic sources without demanding a clean tree."""

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
        raise Qwen3BServingOracleError(
            "cannot record standalone oracle Git provenance"
        ) from error
    if GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BServingOracleError("Git revision has an unexpected format")
    return {
        "git_revision": revision,
        "source_dirty": bool(status),
        "source_status_sha256": _sha256_bytes(status),
        "sources": {
            name: _source_record(root, relative)
            for name, relative in SOURCE_PATHS.items()
        },
    }


def _workload_document(workload: ServingWorkload) -> dict[str, object]:
    return {
        "path": str(workload.source_path),
        "bytes": workload.source_bytes,
        "sha256": workload.source_sha256,
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "case": workload.case,
        "prompt_sha256": _sha256_bytes(workload.prompt.encode("utf-8")),
        "prompt_token_count": len(workload.prompt_token_ids),
        "prompt_token_ids": list(workload.prompt_token_ids),
        "prompt_token_ids_le_u32_sha256": _token_ids_sha256(workload.prompt_token_ids),
        "output_token_count": len(workload.output_token_ids),
        "output_token_ids_le_u32_sha256": _token_ids_sha256(workload.output_token_ids),
        "expected_output_prefix": list(EXPECTED_OUTPUT_PREFIX),
        "sampling": {"temperature": 0.0, "top_p": 1.0},
    }


def _execution_document() -> dict[str, object]:
    return {
        "attention_implementation": "eager",
        "batch_size": 1,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "dtype": "bfloat16",
        "explicit_attention_mask": True,
        "explicit_input_ids": True,
        "explicit_position_ids": list(range(PROMPT_TOKEN_COUNT)),
        "hf_hub_offline": True,
        "inference_mode": True,
        "local_files_only": True,
        "logits_to_keep": 1,
        "return_dict": True,
        "sampling_applied": False,
        "tf32_enabled": False,
        "transformers_offline": True,
        "trust_remote_code": False,
        "use_cache": False,
    }


def build_oracle_artifact(
    *,
    workload: ServingWorkload,
    checkpoint: CheckpointManifest,
    capture: RawLogitCapture,
    producer_metadata: Mapping[str, object],
    source_provenance: Mapping[str, object],
    created_at: datetime,
) -> dict[str, object]:
    """Assemble the JSON-only raw-logit fingerprint before exclusive publication."""

    if (
        len(capture.top_token_ids) != TOP_K
        or len(capture.top_values_bf16_as_f32) != TOP_K
    ):
        raise Qwen3BServingOracleError("HF capture top-k cardinality differs")
    if capture.argmax_token_id != capture.top_token_ids[0]:
        raise Qwen3BServingOracleError(
            "HF capture argmax differs from top-k index zero"
        )
    if capture.argmax_value_bf16_as_f32 != capture.top_values_bf16_as_f32[0]:
        raise Qwen3BServingOracleError(
            "HF capture argmax value differs from top-k index zero"
        )
    if set(capture.probe_values_bf16_as_f32) != set(PROBE_IDS):
        raise Qwen3BServingOracleError("HF capture probe IDs differ")
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "performance_claim_eligible": False,
        "created_at": _utc_text(created_at),
        "producer": dict(producer_metadata),
        "contract": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "workload": _workload_document(workload),
            "execution": _execution_document(),
        },
        "model": {
            "checkpoint_path": str(checkpoint.root),
            "checkpoint_verification": "riley-checkpoint-v1-receipt-sha256+regular-file-size",
            "checkpoint_receipt": checkpoint.receipt.document(),
            "files": [record.document() for record in checkpoint.files],
        },
        "provenance": {"source_repository": dict(source_provenance)},
        "last_logits": {
            "dtype": "bfloat16",
            "element_count": MODEL_VOCABULARY_SIZE,
            "canonical_byte_order": "little-endian-u16",
            "raw_bf16_le_sha256": capture.raw_bf16_le_sha256,
            "raw_bf16_le_bytes": RAW_LOGIT_BYTES,
            "addressable_bf16_le_sha256": capture.addressable_bf16_le_sha256,
            "addressable_bf16_le_bytes": ADDRESSABLE_LOGIT_BYTES,
            "non_addressable_bf16_le_sha256": capture.non_addressable_bf16_le_sha256,
            "non_addressable_bf16_le_bytes": NON_ADDRESSABLE_LOGIT_BYTES,
            "argmax_token_id": capture.argmax_token_id,
            "argmax_value_bf16_as_f32": capture.argmax_value_bf16_as_f32,
            "top_k": TOP_K,
            "top_token_ids": list(capture.top_token_ids),
            "top_values_bf16_as_f32": list(capture.top_values_bf16_as_f32),
            "probe_values_bf16_as_f32": {
                str(token_id): capture.probe_values_bf16_as_f32[token_id]
                for token_id in PROBE_IDS
            },
        },
    }
    validate_oracle_artifact(artifact)
    return artifact


def _validate_file_records(value: object) -> None:
    if not isinstance(value, list) or len(value) != 6:
        raise Qwen3BServingOracleError(
            "artifact model.files must contain six pinned files"
        )
    paths: set[str] = set()
    observed_weight_files: set[tuple[int, str]] = set()
    metadata_by_name = {
        record["path"]: record for record in value if isinstance(record, Mapping)
    }
    if len(metadata_by_name) != len(value):
        raise Qwen3BServingOracleError(
            "artifact model.files contains an invalid record"
        )
    for record in value:
        mapping = _require_mapping(record, "artifact model file")
        _require_exact_keys(
            mapping, {"path", "size_bytes", "sha256"}, "artifact model file"
        )
        path = _require_string(mapping["path"], "artifact model file.path")
        if path in paths or PurePosixPath(path).name != path:
            raise Qwen3BServingOracleError(
                "artifact model file paths must be unique leaf names"
            )
        paths.add(path)
        size_bytes = mapping["size_bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise Qwen3BServingOracleError("artifact model file size must be positive")
        digest = _require_sha256(mapping["sha256"], "artifact model file.sha256")
        if path in EXPECTED_CHECKPOINT_METADATA:
            expected = EXPECTED_CHECKPOINT_METADATA[path]
            if size_bytes != expected["size_bytes"] or digest != expected["sha256"]:
                raise Qwen3BServingOracleError("artifact model metadata pin differs")
        else:
            observed_weight_files.add((size_bytes, digest))
    if not set(EXPECTED_CHECKPOINT_METADATA).issubset(paths):
        raise Qwen3BServingOracleError("artifact model metadata files differ")
    if observed_weight_files != set(EXPECTED_WEIGHT_FILES):
        raise Qwen3BServingOracleError("artifact model weight file pins differ")


def _validate_source_provenance(value: object) -> None:
    mapping = _require_mapping(value, "artifact source provenance")
    _require_exact_keys(
        mapping,
        {"git_revision", "source_dirty", "source_status_sha256", "sources"},
        "artifact source provenance",
    )
    revision = _require_string(mapping["git_revision"], "artifact Git revision")
    if GIT_REVISION_RE.fullmatch(revision) is None:
        raise Qwen3BServingOracleError("artifact Git revision is malformed")
    if not isinstance(mapping["source_dirty"], bool):
        raise Qwen3BServingOracleError("artifact source_dirty must be a boolean")
    _require_sha256(mapping["source_status_sha256"], "artifact source status SHA-256")
    sources = _require_mapping(mapping["sources"], "artifact sources")
    if set(sources) != set(SOURCE_PATHS):
        raise Qwen3BServingOracleError("artifact source provenance file set differs")
    for name, relative in SOURCE_PATHS.items():
        record = _require_mapping(sources[name], f"artifact source {name}")
        _require_exact_keys(record, {"path", "sha256"}, f"artifact source {name}")
        if record["path"] != relative:
            raise Qwen3BServingOracleError("artifact source path differs")
        _require_sha256(record["sha256"], f"artifact source {name} SHA-256")


def _validate_producer(value: object) -> None:
    producer = _require_mapping(value, "artifact producer")
    _require_exact_keys(
        producer,
        {
            "implementation_id",
            "runtime_dependency_class",
            "python_version",
            "python_executable_sha256",
            "python_platform_system",
            "python_platform_machine",
            "torch_version",
            "transformers_version",
            "safetensors_version",
            "selected_device",
            "transformers_model_source",
        },
        "artifact producer",
    )
    if producer["implementation_id"] != IMPLEMENTATION_ID:
        raise Qwen3BServingOracleError("artifact producer implementation differs")
    if producer["runtime_dependency_class"] != "offline-python-reference":
        raise Qwen3BServingOracleError("artifact producer dependency class differs")
    for field in (
        "python_version",
        "python_platform_system",
        "python_platform_machine",
        "torch_version",
        "transformers_version",
        "safetensors_version",
    ):
        _require_string(producer[field], f"artifact producer.{field}")
    _require_sha256(
        producer["python_executable_sha256"], "artifact Python executable SHA-256"
    )
    selected = _require_mapping(producer["selected_device"], "artifact selected device")
    _require_exact_keys(
        selected,
        {
            "requested",
            "resolved",
            "index",
            "name",
            "compute_capability",
            "driver_version",
            "runtime_cuda_version",
        },
        "artifact selected device",
    )
    for field in (
        "requested",
        "resolved",
        "name",
        "compute_capability",
        "driver_version",
        "runtime_cuda_version",
    ):
        _require_string(selected[field], f"artifact selected device.{field}")
    if isinstance(selected["index"], bool) or not isinstance(selected["index"], int):
        raise Qwen3BServingOracleError(
            "artifact selected device.index must be an integer"
        )
    source = _require_mapping(
        producer["transformers_model_source"], "artifact model source"
    )
    _require_exact_keys(
        source, {"module", "filename", "sha256"}, "artifact model source"
    )
    _require_string(source["module"], "artifact model source.module")
    _require_string(source["filename"], "artifact model source.filename")
    _require_sha256(source["sha256"], "artifact model source SHA-256")


def validate_oracle_artifact(document: Mapping[str, object]) -> None:
    """Validate the JSON schema and immutable Qwen3B/P2048 bindings without Torch."""

    _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact_kind",
            "performance_claim_eligible",
            "created_at",
            "producer",
            "contract",
            "model",
            "provenance",
            "last_logits",
        },
        "artifact",
    )
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["artifact_kind"] != ARTIFACT_KIND
    ):
        raise Qwen3BServingOracleError("artifact identity differs")
    if document["performance_claim_eligible"] is not False:
        raise Qwen3BServingOracleError(
            "raw-logit artifact must not be performance eligible"
        )
    _validate_utc_text(document["created_at"], "artifact created_at")
    _validate_producer(document["producer"])
    contract = _require_mapping(document["contract"], "artifact contract")
    _require_exact_keys(
        contract,
        {"model_id", "model_revision", "workload", "execution"},
        "artifact contract",
    )
    if contract["model_id"] != MODEL_ID or contract["model_revision"] != MODEL_REVISION:
        raise Qwen3BServingOracleError("artifact model identity differs")
    workload = _require_mapping(contract["workload"], "artifact workload")
    _require_exact_keys(
        workload,
        {
            "path",
            "bytes",
            "sha256",
            "schema_version",
            "case",
            "prompt_sha256",
            "prompt_token_count",
            "prompt_token_ids",
            "prompt_token_ids_le_u32_sha256",
            "output_token_count",
            "output_token_ids_le_u32_sha256",
            "expected_output_prefix",
            "sampling",
        },
        "artifact workload",
    )
    _require_string(workload["path"], "artifact workload.path")
    workload_bytes = workload["bytes"]
    if (
        isinstance(workload_bytes, bool)
        or not isinstance(workload_bytes, int)
        or workload_bytes <= 0
    ):
        raise Qwen3BServingOracleError("artifact workload.bytes must be positive")
    if workload["sha256"] != WORKLOAD_SHA256:
        raise Qwen3BServingOracleError("artifact workload SHA-256 differs")
    if (
        workload["schema_version"] != WORKLOAD_SCHEMA_VERSION
        or workload["case"] != WORKLOAD_CASE
    ):
        raise Qwen3BServingOracleError("artifact workload identity differs")
    _require_sha256(workload["prompt_sha256"], "artifact workload prompt SHA-256")
    prompt_ids = _require_u32_ids(
        workload["prompt_token_ids"],
        label="artifact workload prompt token IDs",
        count=PROMPT_TOKEN_COUNT,
        upper_bound=ADDRESSABLE_TOKEN_COUNT,
    )
    if (
        workload["prompt_token_count"] != PROMPT_TOKEN_COUNT
        or _token_ids_sha256(prompt_ids) != PROMPT_TOKEN_IDS_SHA256
    ):
        raise Qwen3BServingOracleError("artifact prompt token contract differs")
    if workload["prompt_token_ids_le_u32_sha256"] != PROMPT_TOKEN_IDS_SHA256:
        raise Qwen3BServingOracleError("artifact prompt token hash differs")
    if workload["output_token_count"] != OUTPUT_TOKEN_COUNT:
        raise Qwen3BServingOracleError("artifact output token count differs")
    if workload["output_token_ids_le_u32_sha256"] != OUTPUT_TOKEN_IDS_SHA256:
        raise Qwen3BServingOracleError("artifact output token hash differs")
    if workload["expected_output_prefix"] != list(EXPECTED_OUTPUT_PREFIX):
        raise Qwen3BServingOracleError("artifact expected output prefix differs")
    sampling = _require_mapping(workload["sampling"], "artifact workload sampling")
    if dict(sampling) != {"temperature": 0.0, "top_p": 1.0}:
        raise Qwen3BServingOracleError("artifact workload sampling differs")
    if contract["execution"] != _execution_document():
        raise Qwen3BServingOracleError("artifact execution contract differs")
    model = _require_mapping(document["model"], "artifact model")
    _require_exact_keys(
        model,
        {
            "checkpoint_path",
            "checkpoint_verification",
            "checkpoint_receipt",
            "files",
        },
        "artifact model",
    )
    _require_string(model["checkpoint_path"], "artifact checkpoint path")
    if (
        model["checkpoint_verification"]
        != "riley-checkpoint-v1-receipt-sha256+regular-file-size"
    ):
        raise Qwen3BServingOracleError(
            "artifact checkpoint verification method differs"
        )
    receipt = _require_mapping(
        model["checkpoint_receipt"], "artifact checkpoint receipt"
    )
    _require_exact_keys(
        receipt, {"path", "size_bytes", "sha256"}, "artifact checkpoint receipt"
    )
    if (
        receipt["path"] != CHECKPOINT_RECEIPT_FILENAME
        or receipt["size_bytes"] != CHECKPOINT_RECEIPT_BYTES
        or receipt["sha256"] != CHECKPOINT_RECEIPT_SHA256
    ):
        raise Qwen3BServingOracleError("artifact checkpoint receipt differs")
    _validate_file_records(model["files"])
    provenance = _require_mapping(document["provenance"], "artifact provenance")
    _require_exact_keys(provenance, {"source_repository"}, "artifact provenance")
    _validate_source_provenance(provenance["source_repository"])
    last_logits = _require_mapping(document["last_logits"], "artifact last logits")
    _require_exact_keys(
        last_logits,
        {
            "dtype",
            "element_count",
            "canonical_byte_order",
            "raw_bf16_le_sha256",
            "raw_bf16_le_bytes",
            "addressable_bf16_le_sha256",
            "addressable_bf16_le_bytes",
            "non_addressable_bf16_le_sha256",
            "non_addressable_bf16_le_bytes",
            "argmax_token_id",
            "argmax_value_bf16_as_f32",
            "top_k",
            "top_token_ids",
            "top_values_bf16_as_f32",
            "probe_values_bf16_as_f32",
        },
        "artifact last logits",
    )
    if (
        last_logits["dtype"] != "bfloat16"
        or last_logits["element_count"] != MODEL_VOCABULARY_SIZE
        or last_logits["canonical_byte_order"] != "little-endian-u16"
        or last_logits["raw_bf16_le_bytes"] != RAW_LOGIT_BYTES
        or last_logits["addressable_bf16_le_bytes"] != ADDRESSABLE_LOGIT_BYTES
        or last_logits["non_addressable_bf16_le_bytes"] != NON_ADDRESSABLE_LOGIT_BYTES
        or last_logits["top_k"] != TOP_K
    ):
        raise Qwen3BServingOracleError("artifact raw-logit metadata differs")
    _require_sha256(last_logits["raw_bf16_le_sha256"], "artifact raw-logit SHA-256")
    _require_sha256(
        last_logits["addressable_bf16_le_sha256"],
        "artifact addressable raw-logit SHA-256",
    )
    _require_sha256(
        last_logits["non_addressable_bf16_le_sha256"],
        "artifact non-addressable raw-logit SHA-256",
    )
    top_ids = _require_u32_ids(
        last_logits["top_token_ids"],
        label="artifact top token IDs",
        count=TOP_K,
        upper_bound=MODEL_VOCABULARY_SIZE,
    )
    if len(set(top_ids)) != TOP_K:
        raise Qwen3BServingOracleError("artifact top token IDs must be unique")
    top_values_value = last_logits["top_values_bf16_as_f32"]
    if not isinstance(top_values_value, list) or len(top_values_value) != TOP_K:
        raise Qwen3BServingOracleError("artifact top values cardinality differs")
    top_values = tuple(
        _require_finite_float(value, f"artifact top value {index}")
        for index, value in enumerate(top_values_value)
    )
    if any(left < right for left, right in pairwise(top_values)):
        raise Qwen3BServingOracleError("artifact top values must be descending")
    argmax_token_id = last_logits["argmax_token_id"]
    if isinstance(argmax_token_id, bool) or not isinstance(argmax_token_id, int):
        raise Qwen3BServingOracleError("artifact argmax token ID must be an integer")
    if argmax_token_id != top_ids[0]:
        raise Qwen3BServingOracleError(
            "artifact argmax token differs from top-k index zero"
        )
    if (
        _require_finite_float(
            last_logits["argmax_value_bf16_as_f32"], "artifact argmax value"
        )
        != top_values[0]
    ):
        raise Qwen3BServingOracleError(
            "artifact argmax value differs from top-k index zero"
        )
    probes = _require_mapping(
        last_logits["probe_values_bf16_as_f32"], "artifact probes"
    )
    if set(probes) != {str(token_id) for token_id in PROBE_IDS}:
        raise Qwen3BServingOracleError("artifact probe IDs differ")
    for token_id in PROBE_IDS:
        _require_finite_float(probes[str(token_id)], f"artifact probe {token_id}")


def write_artifact_exclusive(path: Path, artifact: Mapping[str, object]) -> None:
    """Publish one canonical JSON artifact without overwrite or replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(artifact)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise Qwen3BServingOracleError(
            f"refusing to overwrite existing artifact: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _output_outside_repository(output: Path, repo_root: Path) -> Path:
    resolved = output.expanduser().resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        raise Qwen3BServingOracleError(
            "oracle artifact output must be outside the repository"
        )
    return resolved


def produce_oracle(
    *,
    checkpoint_path: Path,
    workload_path: Path,
    output_path: Path,
    repo_root: Path,
    device: str,
    created_at: datetime | None = None,
    backend_factory: BackendFactory = HuggingFaceQwen3BBackend.load,
    source_provenance_factory: SourceProvenanceFactory = collect_source_provenance,
) -> dict[str, object]:
    """Create one source-bound HF raw-logit fingerprint, without serving integration."""

    root = _regular_directory(repo_root.expanduser(), "repository root")
    output = _output_outside_repository(output_path, root)
    if output.exists():
        raise Qwen3BServingOracleError(
            f"refusing to overwrite existing artifact: {output}"
        )
    workload = load_workload(workload_path)
    checkpoint = inspect_checkpoint(checkpoint_path)
    source_provenance = source_provenance_factory(root)
    backend = backend_factory(checkpoint=checkpoint, device=device)
    try:
        capture = backend.capture(workload.prompt_token_ids)
        artifact = build_oracle_artifact(
            workload=workload,
            checkpoint=checkpoint,
            capture=capture,
            producer_metadata=backend.producer_metadata,
            source_provenance=source_provenance,
            created_at=created_at or datetime.now(timezone.utc),
        )
    finally:
        backend.close()
    write_artifact_exclusive(output, artifact)
    return artifact


def _load_artifact(path: Path) -> dict[str, object]:
    source_path = _regular_file(path.expanduser(), "oracle artifact")
    raw = source_path.read_bytes()
    document = dict(_parse_json(raw, "oracle artifact"))
    validate_oracle_artifact(document)
    if raw != _canonical_json_bytes(document):
        raise Qwen3BServingOracleError("oracle artifact JSON is not canonical")
    return document


def validate_artifact_bindings(
    *, artifact_path: Path, workload_path: Path, repo_root: Path
) -> dict[str, object]:
    """Replay immutable workload and current standalone source bindings without Torch."""

    artifact = _load_artifact(artifact_path)
    workload = load_workload(workload_path)
    root = _regular_directory(repo_root.expanduser(), "repository root")
    expected_sources = collect_source_provenance(root)
    artifact_workload = _require_mapping(
        _require_mapping(artifact["contract"], "artifact contract")["workload"],
        "artifact workload",
    )
    if artifact_workload != _workload_document(workload):
        raise Qwen3BServingOracleError(
            "artifact workload binding differs from the supplied input"
        )
    observed_sources = _require_mapping(
        _require_mapping(artifact["provenance"], "artifact provenance")[
            "source_repository"
        ],
        "artifact source provenance",
    )
    if observed_sources["sources"] != expected_sources["sources"]:
        raise Qwen3BServingOracleError(
            "artifact source file hashes differ from the repository"
        )
    return artifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m riley_reference.qwen3b_serving_oracle",
        description="offline-only Qwen2.5-3B P2048 HF eager raw-logit fingerprint",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser(
        "produce", help="write one create-only BF16 eager raw-logit JSON artifact"
    )
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--workload", type=Path, required=True)
    produce.add_argument("--output", type=Path, required=True)
    produce.add_argument("--repo-root", type=Path, required=True)
    produce.add_argument("--device", default="cuda:0")
    validate = subparsers.add_parser(
        "validate", help="validate artifact schema and bindings without Torch or CUDA"
    )
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--workload", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "produce":
            artifact = produce_oracle(
                checkpoint_path=args.checkpoint,
                workload_path=args.workload,
                output_path=args.output,
                repo_root=args.repo_root,
                device=args.device,
            )
            print(
                f"wrote {artifact['artifact_kind']} to {args.output}; "
                f"artifact_sha256={_sha256_file(args.output.resolve())}"
            )
            return 0
        artifact = validate_artifact_bindings(
            artifact_path=args.artifact,
            workload_path=args.workload,
            repo_root=args.repo_root,
        )
        print(
            f"valid {artifact['artifact_kind']}: {args.artifact}; "
            f"raw_bf16_le_sha256={artifact['last_logits']['raw_bf16_le_sha256']}"
        )
        return 0
    except (OSError, RuntimeError, Qwen3BServingOracleError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
