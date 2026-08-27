#!/usr/bin/env python3
"""Approve exactly two independent PR16 HF golden-oracle receipts.

The checker uses only the Python standard library.  It never loads a model or
CUDA and writes one create-only approval that byte-binds both reviewed inputs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence


RECEIPT_SCHEMA = "riley.hf-golden-oracle-receipt.v1"
APPROVAL_SCHEMA = "riley.hf-golden-oracle-approval.v1"
GATE_ID = "pr16-hf-bf16-eager-golden-oracle-v1"
PROMPT = "Explain why deterministic benchmarks need immutable inputs."
PROMPT_SHA256 = "4bc5a3f851d466e92f931bcd16540019311b6930fdad3a9ccb4aa6d11fc3d9f4"
MAX_NEW_TOKENS = 8

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_CONFIG_SHA256 = "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
MODEL_WEIGHTS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
TOKENIZER_FILES_SHA256 = {
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}
TOKENIZER_AGGREGATE_SHA256 = "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db"

PYTHON_VERSION = "3.13.15"
PYTHON_EXECUTABLE_SHA256 = "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866"
DEPENDENCY_LOCK_SHA256 = "101d21486780e57492b3053149c0a594fcf2859d1955854250bd644b6fdaff30"
TORCH_VERSION = "2.13.0"
TRANSFORMERS_VERSION = "5.15.1"
TOKENIZERS_VERSION = "0.22.2"
CUDA_RUNTIME_VERSION = "13.0"
DRIVER_VERSION = "580.173.02"
GPU_UUID = "GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0"
GPU_NAME = "NVIDIA GeForce RTX 4090"
GPU_COMPUTE_CAPABILITY = "8.9"

EXPECTED_TOKEN_IDS = [198, 198, 504, 44771, 9577, 359, 260, 9577]
EXPECTED_TOKEN_IDS_U32LE_SHA256 = "d9b9a665ea62ae4e21235b347973ee811267bcf205f090e376c6ed71be2c8ba4"
EXPECTED_TEXT_UTF8_SHA256 = "e79401a64f79f3a3bf47c04cb0d0d0c0116eb97ee10e7caef4c60dc716831d47"

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
BOOT_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
SAFE_ABSOLUTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/+@=-]+$")
MAX_RECEIPT_BYTES = 1024 * 1024


class OracleApprovalError(ValueError):
    """A receipt, reviewer anchor, or output violates the closed contract."""


def _fail(path: str, message: str) -> NoReturn:
    raise OracleApprovalError(f"{path}: {message}")


def _canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise OracleApprovalError(f"JSON: cannot encode canonical bytes: {error}") from error


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate object key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> NoReturn:
    _fail("JSON", f"non-finite numeric literal {value!r} is forbidden")


def _load_receipt(path: Path, label: str) -> tuple[dict[str, Any], bytes, os.stat_result]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise OracleApprovalError(f"{label}: cannot stat {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        _fail(label, "must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_RECEIPT_BYTES:
        _fail(label, f"must contain 1..{MAX_RECEIPT_BYTES} bytes")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise OracleApprovalError(f"{label}: cannot read {path}: {error}") from error
    try:
        document = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleApprovalError(f"{label}: invalid strict UTF-8 JSON: {error}") from error
    if not isinstance(document, dict):
        _fail(label, "root must be an object")
    if raw != _canonical_json_bytes(document):
        _fail(label, "must use exact canonical JSON bytes with one trailing newline")
    return document, raw, metadata


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _exact(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    row = _object(value, path)
    if set(row) != keys:
        _fail(path, f"keys must be exactly {sorted(keys)!r}")
    return row


def _string(value: Any, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, "has invalid syntax")
    return value


def _sha(value: Any, path: str) -> str:
    return _string(value, path, SHA_RE)


def _integer(value: Any, path: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        _fail(path, f"must be an integer in [{minimum}, {maximum}]")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a boolean")
    return value


def _timestamp(value: Any, path: str) -> datetime:
    text = _string(value, path, UTC_RE)
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise OracleApprovalError(f"{path}: invalid UTC timestamp: {error}") from error
    if parsed.tzinfo != timezone.utc:
        _fail(path, "must be UTC")
    return parsed


def _safe_absolute_path(value: Any, path: str) -> str:
    text = _string(value, path, SAFE_ABSOLUTE_PATH_RE)
    pure = PurePosixPath(text)
    if not pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(path, "must be a normalized safe absolute POSIX path")
    return text


def _expected_normalized_argv(model_tree: str, dependency_lock: str) -> list[str]:
    return [
        "$PINNED_PYTHON",
        "-I",
        "ci/release/write_hf_golden_oracle_receipt.py",
        "--output",
        "$OUTPUT_RECEIPT",
        "--model-dir",
        "$LOCAL_MODEL_DIR",
        "--dependency-lock",
        "$DEPENDENCY_LOCK",
        "--expected-model-tree-sha256",
        model_tree,
        "--expected-dependency-lock-sha256",
        dependency_lock,
    ]


def _u32le_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<I", value) for value in values)).hexdigest()


def _validate_observation(value: Any, path: str, expected_mode: str) -> tuple[tuple[int, ...], str]:
    row = _exact(
        value,
        {
            "cache_mode",
            "generated_text",
            "generated_text_utf8_sha256",
            "generated_token_ids",
            "generated_token_ids_u32le_sha256",
        },
        path,
    )
    if row["cache_mode"] != expected_mode:
        _fail(f"{path}.cache_mode", f"must be {expected_mode!r}")
    token_ids_value = row["generated_token_ids"]
    if not isinstance(token_ids_value, list) or len(token_ids_value) != MAX_NEW_TOKENS:
        _fail(f"{path}.generated_token_ids", "must contain exactly eight IDs")
    token_ids = tuple(
        _integer(value, f"{path}.generated_token_ids[{index}]", 0, 0xFFFFFFFF)
        for index, value in enumerate(token_ids_value)
    )
    if list(token_ids) != EXPECTED_TOKEN_IDS:
        _fail(f"{path}.generated_token_ids", "differs from the independently reviewed IDs")
    token_sha = _sha(
        row["generated_token_ids_u32le_sha256"],
        f"{path}.generated_token_ids_u32le_sha256",
    )
    if token_sha != _u32le_sha256(token_ids) or token_sha != EXPECTED_TOKEN_IDS_U32LE_SHA256:
        _fail(f"{path}.generated_token_ids_u32le_sha256", "does not bind the reviewed u32le IDs")
    text = _string(row["generated_text"], f"{path}.generated_text")
    if len(text.encode("utf-8")) > 4096:
        _fail(f"{path}.generated_text", "is unreasonably large")
    text_sha = _sha(row["generated_text_utf8_sha256"], f"{path}.generated_text_utf8_sha256")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != text_sha:
        _fail(f"{path}.generated_text_utf8_sha256", "does not hash the exact UTF-8 text")
    if text_sha != EXPECTED_TEXT_UTF8_SHA256:
        _fail(f"{path}.generated_text_utf8_sha256", "differs from the reviewed completion")
    return token_ids, text


def validate_receipt(
    document: Mapping[str, Any],
    *,
    path: str,
    expected_model_tree_sha256: str,
    expected_dependency_lock_sha256: str,
    expected_python_executable_sha256: str,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    root = _exact(
        document,
        {
            "gate_id",
            "invocation",
            "model",
            "observations",
            "oracle",
            "process",
            "prompt",
            "run_id",
            "run_nonce",
            "runtime",
            "schema_version",
            "settings",
            "status",
            "timing",
        },
        path,
    )
    if root["schema_version"] != RECEIPT_SCHEMA or root["gate_id"] != GATE_ID:
        _fail(path, "schema/gate identity differs from the PR16 oracle contract")
    if root["status"] != "passed":
        _fail(f"{path}.status", "must be 'passed'")

    nonce = _string(root["run_nonce"], f"{path}.run_nonce", NONCE_RE)
    run_id = _string(root["run_id"], f"{path}.run_id")
    if run_id != f"hf-golden-oracle-{nonce}":
        _fail(f"{path}.run_id", "must be derived from run_nonce")
    process = _exact(root["process"], {"boot_id", "pid", "start_time_ticks"}, f"{path}.process")
    process_identity = (
        _string(process["boot_id"], f"{path}.process.boot_id", BOOT_ID_RE),
        _integer(process["pid"], f"{path}.process.pid", 1, 1 << 30),
        _integer(process["start_time_ticks"], f"{path}.process.start_time_ticks", 1, 1 << 63),
    )

    timing = _exact(root["timing"], {"ended_at_utc", "started_at_utc"}, f"{path}.timing")
    started = _timestamp(timing["started_at_utc"], f"{path}.timing.started_at_utc")
    ended = _timestamp(timing["ended_at_utc"], f"{path}.timing.ended_at_utc")
    elapsed = (ended - started).total_seconds()
    if not math.isfinite(elapsed) or elapsed < 0 or elapsed > 3600:
        _fail(f"{path}.timing", "must be ordered and bounded to one hour")

    invocation = _exact(root["invocation"], {"normalized_argv"}, f"{path}.invocation")
    if invocation["normalized_argv"] != _expected_normalized_argv(
        expected_model_tree_sha256, expected_dependency_lock_sha256
    ):
        _fail(f"{path}.invocation.normalized_argv", "differs from the reviewed isolated command")

    oracle = _exact(
        root["oracle"],
        {"dependency_lock", "implementation_id", "provenance_kind"},
        f"{path}.oracle",
    )
    if oracle["implementation_id"] != "hf-transformers-eager" or oracle["provenance_kind"] != "dependency-lock":
        _fail(f"{path}.oracle", "must identify the independent pinned HF eager lane")
    dependency_lock = _exact(
        oracle["dependency_lock"], {"name", "sha256"}, f"{path}.oracle.dependency_lock"
    )
    if dependency_lock["name"] != "uv.lock":
        _fail(f"{path}.oracle.dependency_lock.name", "must be uv.lock")
    if _sha(dependency_lock["sha256"], f"{path}.oracle.dependency_lock.sha256") != expected_dependency_lock_sha256:
        _fail(f"{path}.oracle.dependency_lock.sha256", "differs from reviewer anchor")

    model = _exact(
        root["model"],
        {
            "config_sha256",
            "file_count",
            "id",
            "revision",
            "tokenizer_aggregate_sha256",
            "tokenizer_files_sha256",
            "tree_sha256",
            "weights_sha256",
        },
        f"{path}.model",
    )
    expected_scalars = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_sha256": MODEL_CONFIG_SHA256,
        "weights_sha256": MODEL_WEIGHTS_SHA256,
        "tokenizer_aggregate_sha256": TOKENIZER_AGGREGATE_SHA256,
        "tree_sha256": expected_model_tree_sha256,
    }
    for key, expected in expected_scalars.items():
        if model[key] != expected:
            _fail(f"{path}.model.{key}", "differs from the immutable/reviewer anchor")
    _integer(model["file_count"], f"{path}.model.file_count", 8, 100000)
    tokenizer_files = _exact(
        model["tokenizer_files_sha256"],
        set(TOKENIZER_FILES_SHA256),
        f"{path}.model.tokenizer_files_sha256",
    )
    if tokenizer_files != TOKENIZER_FILES_SHA256:
        _fail(f"{path}.model.tokenizer_files_sha256", "differs from immutable tokenizer files")

    prompt = _exact(root["prompt"], {"text", "utf8_base64", "utf8_sha256"}, f"{path}.prompt")
    if prompt["text"] != PROMPT or prompt["utf8_sha256"] != PROMPT_SHA256:
        _fail(f"{path}.prompt", "differs from the exact reviewed prompt")
    try:
        decoded_prompt = base64.b64decode(prompt["utf8_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise OracleApprovalError(f"{path}.prompt.utf8_base64: invalid canonical base64: {error}") from error
    if decoded_prompt != PROMPT.encode("utf-8") or base64.b64encode(decoded_prompt).decode("ascii") != prompt["utf8_base64"]:
        _fail(f"{path}.prompt.utf8_base64", "does not encode the exact prompt bytes")

    settings = _exact(
        root["settings"],
        {
            "add_special_tokens",
            "attention_implementation",
            "cache_modes",
            "clean_up_tokenization_spaces",
            "device",
            "do_sample",
            "dtype",
            "local_files_only",
            "max_new_tokens",
            "num_beams",
            "skip_special_tokens",
            "trust_remote_code",
        },
        f"{path}.settings",
    )
    expected_settings = {
        "add_special_tokens": True,
        "attention_implementation": "eager",
        "cache_modes": ["on", "off"],
        "clean_up_tokenization_spaces": False,
        "device": "cuda:0",
        "do_sample": False,
        "dtype": "bfloat16",
        "local_files_only": True,
        "max_new_tokens": 8,
        "num_beams": 1,
        "skip_special_tokens": True,
        "trust_remote_code": False,
    }
    if settings != expected_settings:
        _fail(f"{path}.settings", "differs from exact BF16 eager greedy settings")

    runtime = _exact(
        root["runtime"],
        {"cuda_runtime_version", "dependencies", "driver_version", "environment", "gpu", "python"},
        f"{path}.runtime",
    )
    dependencies = _exact(
        runtime["dependencies"], {"tokenizers", "torch", "transformers"}, f"{path}.runtime.dependencies"
    )
    if dependencies != {
        "tokenizers": TOKENIZERS_VERSION,
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
    }:
        _fail(f"{path}.runtime.dependencies", "differs from the pinned dependency lock")
    if runtime["cuda_runtime_version"] != CUDA_RUNTIME_VERSION or runtime["driver_version"] != DRIVER_VERSION:
        _fail(f"{path}.runtime", "CUDA runtime/driver differs from the primary contract")
    environment = _exact(
        runtime["environment"],
        {
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES",
            "HF_HUB_OFFLINE",
            "TOKENIZERS_PARALLELISM",
            "TRANSFORMERS_OFFLINE",
        },
        f"{path}.runtime.environment",
    )
    if environment != {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": expected_gpu_uuid,
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }:
        _fail(f"{path}.runtime.environment", "differs from the isolated offline environment")
    gpu = _exact(runtime["gpu"], {"compute_capability", "name", "uuid"}, f"{path}.runtime.gpu")
    if gpu != {"compute_capability": GPU_COMPUTE_CAPABILITY, "name": GPU_NAME, "uuid": expected_gpu_uuid}:
        _fail(f"{path}.runtime.gpu", "differs from the reviewer-selected primary GPU")
    python = _exact(
        runtime["python"],
        {
            "executable",
            "executable_sha256",
            "ignore_environment",
            "isolated",
            "no_user_site",
            "platform_machine",
            "platform_system",
            "version",
        },
        f"{path}.runtime.python",
    )
    _safe_absolute_path(python["executable"], f"{path}.runtime.python.executable")
    if (
        _sha(python["executable_sha256"], f"{path}.runtime.python.executable_sha256")
        != expected_python_executable_sha256
        or python["version"] != PYTHON_VERSION
        or python["platform_system"] != "linux"
        or python["platform_machine"] != "x86_64"
        or _boolean(python["isolated"], f"{path}.runtime.python.isolated") is not True
        or _boolean(python["no_user_site"], f"{path}.runtime.python.no_user_site") is not True
        or _boolean(python["ignore_environment"], f"{path}.runtime.python.ignore_environment") is not True
    ):
        _fail(f"{path}.runtime.python", "differs from the pinned isolated CPython runtime")

    observations = root["observations"]
    if not isinstance(observations, list) or len(observations) != 2:
        _fail(f"{path}.observations", "must contain exactly cache-on then cache-off")
    on = _validate_observation(observations[0], f"{path}.observations[0]", "on")
    off = _validate_observation(observations[1], f"{path}.observations[1]", "off")
    if on != off:
        _fail(f"{path}.observations", "cache-on and cache-off outputs differ")
    return {
        "ended": ended,
        "process_identity": process_identity,
        "run_id": run_id,
        "run_nonce": nonce,
        "started": started,
    }


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise OracleApprovalError(f"--output: create-only open failed for {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def approve(
    *,
    receipts: Sequence[Path],
    output: Path,
    expected_model_tree_sha256: str,
    expected_dependency_lock_sha256: str,
    expected_python_executable_sha256: str,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    if len(receipts) != 2:
        _fail("--receipt", "must be supplied exactly twice")
    anchors = {
        "--expected-model-tree-sha256": expected_model_tree_sha256,
        "--expected-dependency-lock-sha256": expected_dependency_lock_sha256,
        "--expected-python-executable-sha256": expected_python_executable_sha256,
    }
    for label, value in anchors.items():
        if SHA_RE.fullmatch(value) is None:
            _fail(label, "must be lowercase SHA-256")
    if expected_dependency_lock_sha256 != DEPENDENCY_LOCK_SHA256:
        _fail("--expected-dependency-lock-sha256", "differs from the reviewed reference lock")
    if expected_python_executable_sha256 != PYTHON_EXECUTABLE_SHA256:
        _fail("--expected-python-executable-sha256", "differs from the pinned Linux CPython binary")
    if expected_gpu_uuid != GPU_UUID:
        _fail("--expected-gpu-uuid", "differs from the designated primary GPU")
    if output.exists() or output.is_symlink():
        _fail("--output", "already exists; approvals are create-only")

    loaded = [_load_receipt(path, f"receipt[{index}]") for index, path in enumerate(receipts)]
    if (loaded[0][2].st_dev, loaded[0][2].st_ino) == (loaded[1][2].st_dev, loaded[1][2].st_ino):
        _fail("--receipt", "both arguments resolve to the same file identity")
    validated = [
        validate_receipt(
            document,
            path=f"receipt[{index}]",
            expected_model_tree_sha256=expected_model_tree_sha256,
            expected_dependency_lock_sha256=expected_dependency_lock_sha256,
            expected_python_executable_sha256=expected_python_executable_sha256,
            expected_gpu_uuid=expected_gpu_uuid,
        )
        for index, (document, _raw, _metadata) in enumerate(loaded)
    ]
    if validated[0]["run_nonce"] == validated[1]["run_nonce"] or validated[0]["run_id"] == validated[1]["run_id"]:
        _fail("receipts", "must have distinct fresh run IDs and nonces")
    if validated[0]["process_identity"] == validated[1]["process_identity"]:
        _fail("receipts", "must come from distinct fresh process identities")
    receipt_hashes = [hashlib.sha256(raw).hexdigest() for _document, raw, _metadata in loaded]
    if receipt_hashes[0] == receipt_hashes[1]:
        _fail("receipts", "must have distinct canonical bytes")

    bindings = []
    for index, ((_, raw, _), row) in enumerate(zip(loaded, validated, strict=True), 1):
        process_payload = _canonical_json_bytes(
            {
                "boot_id": row["process_identity"][0],
                "pid": row["process_identity"][1],
                "start_time_ticks": row["process_identity"][2],
            }
        )
        bindings.append(
            {
                "process_identity_sha256": hashlib.sha256(process_payload).hexdigest(),
                "receipt_sha256": hashlib.sha256(raw).hexdigest(),
                "run_id": row["run_id"],
                "run_nonce": row["run_nonce"],
                "slot": index,
            }
        )
    approval_nonce = hashlib.sha256("".join(receipt_hashes).encode("ascii")).hexdigest()
    approval: dict[str, Any] = {
        "approval_id": f"hf-golden-oracle-approval-{approval_nonce}",
        "cache_paths_verified": [
            f"{validated[0]['run_id']}:on",
            f"{validated[0]['run_id']}:off",
            f"{validated[1]['run_id']}:on",
            f"{validated[1]['run_id']}:off",
        ],
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "expected_generation": {
            "generated_text_utf8_sha256": EXPECTED_TEXT_UTF8_SHA256,
            "generated_token_ids": EXPECTED_TOKEN_IDS,
            "generated_token_ids_u32le_sha256": EXPECTED_TOKEN_IDS_U32LE_SHA256,
        },
        "gate_id": GATE_ID,
        "receipt_bindings": bindings,
        "reviewed_anchors": {
            "dependency_lock_sha256": expected_dependency_lock_sha256,
            "gpu_uuid": expected_gpu_uuid,
            "model_tree_sha256": expected_model_tree_sha256,
            "python_executable_sha256": expected_python_executable_sha256,
        },
        "schema_version": APPROVAL_SCHEMA,
        "status": "approved",
    }
    _write_create_only(output, _canonical_json_bytes(approval))
    return approval


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-model-tree-sha256", required=True)
    parser.add_argument("--expected-dependency-lock-sha256", required=True)
    parser.add_argument("--expected-python-executable-sha256", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        approval = approve(
            receipts=args.receipt,
            output=args.output,
            expected_model_tree_sha256=args.expected_model_tree_sha256,
            expected_dependency_lock_sha256=args.expected_dependency_lock_sha256,
            expected_python_executable_sha256=args.expected_python_executable_sha256,
            expected_gpu_uuid=args.expected_gpu_uuid,
        )
    except OracleApprovalError as error:
        print(f"HF golden oracle approval failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"approval_id": approval["approval_id"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
