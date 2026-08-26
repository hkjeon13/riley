#!/usr/bin/env python3
"""Create one remote-only PR16 Hugging Face golden-oracle receipt.

This development tool is intentionally separate from the release candidate.
It loads only a caller-supplied local checkpoint and never imports rustinfer.
Run it twice in two fresh pinned Python processes; each process proves exact
greedy parity between the Transformers cache-on and cache-off paths.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import stat
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, NoReturn, Protocol, Sequence


SCHEMA_VERSION = "rustinfer.hf-golden-oracle-receipt.v1"
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
SAFE_RELATIVE_PATH_RE = re.compile(r"^[A-Za-z0-9._/+@=-]+$")
MAX_LOCK_BYTES = 16 * 1024 * 1024


class OracleReceiptError(ValueError):
    """The remote oracle environment or output violates the closed contract."""


def _fail(path: str, message: str) -> NoReturn:
    raise OracleReceiptError(f"{path}: {message}")


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
        raise OracleReceiptError(f"receipt: cannot encode canonical JSON: {error}") from error


def _sha256_file(path: Path, label: str, *, max_bytes: int | None = None) -> str:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise OracleReceiptError(f"{label}: cannot stat {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        _fail(label, "must be a regular non-symlink file")
    if max_bytes is not None and metadata.st_size > max_bytes:
        _fail(label, f"exceeds {max_bytes} bytes")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise OracleReceiptError(f"{label}: cannot read {path}: {error}") from error
    return digest.hexdigest()


def _safe_model_relative_path(relative: str) -> None:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or SAFE_RELATIVE_PATH_RE.fullmatch(relative) is None
    ):
        _fail("--model-dir", f"unsafe model path {relative!r}")


def canonical_model_tree(model_dir: Path) -> tuple[str, dict[str, str]]:
    """Return the E2E canonical full-tree digest and its closed file map."""

    try:
        root_metadata = model_dir.lstat()
    except OSError as error:
        raise OracleReceiptError(f"--model-dir: cannot stat {model_dir}: {error}") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        _fail("--model-dir", "must be a real directory, not a symlink")

    files: dict[str, str] = {}
    try:
        for directory, dirnames, filenames in os.walk(model_dir, followlinks=False):
            directory_path = Path(directory)
            for name in sorted(dirnames):
                path = directory_path / name
                relative = path.relative_to(model_dir).as_posix()
                _safe_model_relative_path(relative)
                if not stat.S_ISDIR(path.lstat().st_mode):
                    _fail("--model-dir", f"contains non-directory entry {relative!r}")
            for name in sorted(filenames):
                path = directory_path / name
                relative = path.relative_to(model_dir).as_posix()
                _safe_model_relative_path(relative)
                if relative in files:
                    _fail("--model-dir", f"contains duplicate path {relative!r}")
                files[relative] = _sha256_file(path, f"model file {relative}")
    except OSError as error:
        raise OracleReceiptError(f"--model-dir: cannot enumerate tree: {error}") from error
    if not files:
        _fail("--model-dir", "must contain regular model files")
    ordered = b"".join(
        f"{files[name]}  {name}\n".encode("ascii") for name in sorted(files)
    )
    return hashlib.sha256(ordered).hexdigest(), {name: files[name] for name in sorted(files)}


def _verify_model_files(files: Mapping[str, str]) -> None:
    expected = {
        "config.json": MODEL_CONFIG_SHA256,
        "model.safetensors": MODEL_WEIGHTS_SHA256,
        **TOKENIZER_FILES_SHA256,
    }
    for name, digest in expected.items():
        if files.get(name) != digest:
            _fail("--model-dir", f"{name} does not match the immutable model contract")
    canonical_tokenizer = json.dumps(
        {name: files[name] for name in sorted(TOKENIZER_FILES_SHA256)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical_tokenizer).hexdigest() != TOKENIZER_AGGREGATE_SHA256:
        _fail("--model-dir", "tokenizer aggregate does not match the immutable contract")


def _u32le_sha256(token_ids: Sequence[int]) -> str:
    try:
        payload = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
    except (struct.error, TypeError) as error:
        raise OracleReceiptError(f"generated token IDs are not unsigned 32-bit integers: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def _decode_nvml_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "strict")
    if isinstance(value, str):
        return value
    _fail("runtime.gpu", "NVML returned a non-text identity")


class OracleBackend(Protocol):
    metadata: Mapping[str, Any]

    def generate(self, *, use_cache: bool) -> tuple[list[int], str]: ...


class HuggingFaceBackend:
    """Pinned Transformers eager BF16 backend, imported only on the remote lane."""

    def __init__(self, model_dir: Path) -> None:
        try:
            import pynvml
            import tokenizers
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise OracleReceiptError(
                "oracle dependencies: install the exact tools/python/reference/uv.lock environment"
            ) from error

        versions = {
            "torch": _base_version(torch.__version__),
            "transformers": _base_version(transformers.__version__),
            "tokenizers": _base_version(tokenizers.__version__),
        }
        expected_versions = {
            "torch": TORCH_VERSION,
            "transformers": TRANSFORMERS_VERSION,
            "tokenizers": TOKENIZERS_VERSION,
        }
        if versions != expected_versions:
            _fail("runtime.dependencies", f"expected {expected_versions}, found {versions}")
        if _base_version(str(torch.version.cuda)) != CUDA_RUNTIME_VERSION:
            _fail("runtime.cuda_runtime_version", "does not match the pinned CUDA wheel runtime")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            _fail("runtime.gpu", "requires exactly one visible CUDA device")
        torch.cuda.set_device(0)
        if not torch.cuda.is_bf16_supported():
            _fail("runtime.gpu", "selected CUDA device lacks BF16 support")
        capability_tuple = torch.cuda.get_device_capability(0)
        capability = f"{capability_tuple[0]}.{capability_tuple[1]}"
        device_name = torch.cuda.get_device_name(0)
        if device_name != GPU_NAME or capability != GPU_COMPUTE_CAPABILITY:
            _fail("runtime.gpu", "does not match the primary RTX 4090 contract")

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByUUID(GPU_UUID)
            nvml_uuid = _decode_nvml_text(pynvml.nvmlDeviceGetUUID(handle))
            nvml_name = _decode_nvml_text(pynvml.nvmlDeviceGetName(handle))
            driver_version = _decode_nvml_text(pynvml.nvmlSystemGetDriverVersion())
        except Exception as error:
            raise OracleReceiptError(f"runtime.gpu: cannot query designated NVML device: {error}") from error
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        if nvml_uuid != GPU_UUID or nvml_name != GPU_NAME or driver_version != DRIVER_VERSION:
            _fail("runtime.gpu", "NVML identity differs from the reviewed primary host")

        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            model.eval().to(torch.device("cuda:0"))
        except Exception as error:
            raise OracleReceiptError(f"oracle model load: {error}") from error
        floating_dtypes = {
            parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
        }
        if floating_dtypes != {torch.bfloat16}:
            _fail("settings.dtype", f"model floating parameter dtypes are {floating_dtypes!r}")

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self.metadata = {
            "cuda_runtime_version": CUDA_RUNTIME_VERSION,
            "dependencies": versions,
            "driver_version": driver_version,
            "gpu": {
                "compute_capability": capability,
                "name": device_name,
                "uuid": nvml_uuid,
            },
        }

    def generate(self, *, use_cache: bool) -> tuple[list[int], str]:
        torch = self._torch
        encoded = self._tokenizer(
            PROMPT,
            add_special_tokens=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        try:
            with torch.inference_mode():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    use_cache=use_cache,
                )
            torch.cuda.synchronize(0)
        except Exception as error:
            raise OracleReceiptError(f"cache-{'on' if use_cache else 'off'} generation: {error}") from error
        token_ids = [int(value) for value in output[0, input_length:].tolist()]
        text = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(text, str):
            _fail("generated_text", "tokenizer decode did not return text")
        return token_ids, text


def _run_cache_pair(backend: OracleBackend) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for label, enabled in (("on", True), ("off", False)):
        token_ids, text = backend.generate(use_cache=enabled)
        if (
            len(token_ids) != MAX_NEW_TOKENS
            or any(type(value) is not int or value < 0 or value > 0xFFFFFFFF for value in token_ids)
        ):
            _fail(f"observations.cache_{label}.generated_token_ids", "must contain eight u32 IDs")
        try:
            text_bytes = text.encode("utf-8", "strict")
        except UnicodeEncodeError as error:
            raise OracleReceiptError(f"observations.cache_{label}.generated_text: {error}") from error
        observations.append(
            {
                "cache_mode": label,
                "generated_text": text,
                "generated_text_utf8_sha256": hashlib.sha256(text_bytes).hexdigest(),
                "generated_token_ids": token_ids,
                "generated_token_ids_u32le_sha256": _u32le_sha256(token_ids),
            }
        )
    comparable = [
        (
            row["generated_token_ids"],
            row["generated_token_ids_u32le_sha256"],
            row["generated_text"],
            row["generated_text_utf8_sha256"],
        )
        for row in observations
    ]
    if comparable[0] != comparable[1]:
        _fail("observations", "cache-on and cache-off generation differ")
    if observations[0]["generated_token_ids"] != EXPECTED_TOKEN_IDS:
        _fail("observations", "generated token IDs differ from the reviewed PR16 oracle")
    if observations[0]["generated_text_utf8_sha256"] != EXPECTED_TEXT_UTF8_SHA256:
        _fail("observations", "generated text differs from the reviewed PR16 oracle")
    return observations


def _process_identity() -> dict[str, Any]:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        fields = Path("/proc/self/stat").read_text(encoding="ascii").split()
        start_time_ticks = int(fields[21])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise OracleReceiptError(f"process identity: cannot read Linux procfs: {error}") from error
    return {
        "boot_id": boot_id,
        "pid": os.getpid(),
        "start_time_ticks": start_time_ticks,
    }


def _normalized_argv(model_tree_sha256: str, dependency_lock_sha256: str) -> list[str]:
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
        model_tree_sha256,
        "--expected-dependency-lock-sha256",
        dependency_lock_sha256,
    ]


def _write_create_only(path: Path, payload: bytes) -> None:
    if not path.name or path.name in {".", ".."}:
        _fail("--output", "must name a receipt file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise OracleReceiptError(f"--output: create-only open failed for {path}: {error}") from error
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


def create_receipt(
    *,
    output: Path,
    model_dir: Path,
    dependency_lock: Path,
    expected_model_tree_sha256: str,
    expected_dependency_lock_sha256: str,
    backend_factory: Callable[[Path], OracleBackend] = HuggingFaceBackend,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        _fail("--output", "already exists; receipts are create-only")
    if SHA_RE.fullmatch(expected_model_tree_sha256) is None:
        _fail("--expected-model-tree-sha256", "must be lowercase SHA-256")
    if expected_dependency_lock_sha256 != DEPENDENCY_LOCK_SHA256:
        _fail("--expected-dependency-lock-sha256", "differs from the pinned reference lock")
    if dependency_lock.name != "uv.lock":
        _fail("--dependency-lock", "must be the pinned uv.lock")
    lock_sha256 = _sha256_file(dependency_lock, "--dependency-lock", max_bytes=MAX_LOCK_BYTES)
    if lock_sha256 != expected_dependency_lock_sha256:
        _fail("--dependency-lock", "bytes differ from the independently reviewed digest")
    tree_sha256, model_files = canonical_model_tree(model_dir)
    if tree_sha256 != expected_model_tree_sha256:
        _fail("--model-dir", "canonical full-tree SHA-256 differs from the reviewed digest")
    _verify_model_files(model_files)

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        _fail("runtime.python", "oracle receipts are remote Linux x86_64 only")
    if platform.python_version() != PYTHON_VERSION:
        _fail("runtime.python.version", f"must be {PYTHON_VERSION}")
    if not sys.flags.isolated or not sys.flags.no_user_site or not sys.flags.ignore_environment:
        _fail("runtime.python.flags", "invoke the pinned interpreter with -I")
    python_path = Path(sys.executable).resolve(strict=True)
    python_sha256 = _sha256_file(python_path, "runtime.python.executable")
    if python_sha256 != PYTHON_EXECUTABLE_SHA256:
        _fail("runtime.python.executable", "SHA-256 differs from the pinned CPython binary")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        _fail("runtime.environment.CUDA_VISIBLE_DEVICES", f"must equal {GPU_UUID}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    started_at_utc = _utc_now()
    process = _process_identity()
    run_nonce = secrets.token_hex(32)
    backend = backend_factory(model_dir)
    observations = _run_cache_pair(backend)
    ended_at_utc = _utc_now()
    runtime = dict(backend.metadata)
    receipt: dict[str, Any] = {
        "gate_id": GATE_ID,
        "invocation": {
            "normalized_argv": _normalized_argv(tree_sha256, lock_sha256),
        },
        "model": {
            "config_sha256": model_files["config.json"],
            "file_count": len(model_files),
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_aggregate_sha256": TOKENIZER_AGGREGATE_SHA256,
            "tokenizer_files_sha256": {
                name: model_files[name] for name in sorted(TOKENIZER_FILES_SHA256)
            },
            "tree_sha256": tree_sha256,
            "weights_sha256": model_files["model.safetensors"],
        },
        "observations": observations,
        "oracle": {
            "dependency_lock": {
                "name": "uv.lock",
                "sha256": lock_sha256,
            },
            "implementation_id": "hf-transformers-eager",
            "provenance_kind": "dependency-lock",
        },
        "process": process,
        "prompt": {
            "text": PROMPT,
            "utf8_base64": base64.b64encode(PROMPT.encode("utf-8")).decode("ascii"),
            "utf8_sha256": PROMPT_SHA256,
        },
        "run_id": f"hf-golden-oracle-{run_nonce}",
        "run_nonce": run_nonce,
        "runtime": {
            **runtime,
            "environment": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_VISIBLE_DEVICES": GPU_UUID,
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            },
            "python": {
                "executable": python_path.as_posix(),
                "executable_sha256": python_sha256,
                "ignore_environment": True,
                "isolated": True,
                "no_user_site": True,
                "platform_machine": "x86_64",
                "platform_system": "linux",
                "version": PYTHON_VERSION,
            },
        },
        "schema_version": SCHEMA_VERSION,
        "settings": {
            "add_special_tokens": True,
            "attention_implementation": "eager",
            "cache_modes": ["on", "off"],
            "clean_up_tokenization_spaces": False,
            "device": "cuda:0",
            "do_sample": False,
            "dtype": "bfloat16",
            "local_files_only": True,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_beams": 1,
            "skip_special_tokens": True,
            "trust_remote_code": False,
        },
        "status": "passed",
        "timing": {
            "ended_at_utc": ended_at_utc,
            "started_at_utc": started_at_utc,
        },
    }
    _write_create_only(output, _canonical_json_bytes(receipt))
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--expected-model-tree-sha256", required=True)
    parser.add_argument("--expected-dependency-lock-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = create_receipt(
            output=args.output,
            model_dir=args.model_dir,
            dependency_lock=args.dependency_lock,
            expected_model_tree_sha256=args.expected_model_tree_sha256,
            expected_dependency_lock_sha256=args.expected_dependency_lock_sha256,
        )
    except OracleReceiptError as error:
        print(f"HF golden oracle receipt failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "run_id": receipt["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
