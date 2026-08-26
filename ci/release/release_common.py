#!/usr/bin/env python3
"""Shared, standard-library-only release bundle contract helpers."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import struct
import tomllib
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TARGET = "x86_64-unknown-linux-gnu"
CUDA_TOOLKIT = "12.8.1"
CUDA_ARCHITECTURES = ["89"]
ARCHIVE_SUFFIX = "linux-x86_64-cuda12.8"
MIT_LICENSE_EXPRESSION = "MIT"
SERVER_DEFAULTS_SOURCE_PATH = Path("crates/rustinfer-server/src/main.rs")
SERVER_DEFAULTS_SOURCE_SHA256 = (
    "32389b697e360da6b7b7c21ff2b5b4bd8b4064370812f73287cc284b3c436c1b"
)
STABLE_OPTIMIZATION_DEFAULTS = {
    "execution_completion": "iteration-batch",
    "residual_rmsnorm": "separate",
    "reduction_profile": "canonical-v1",
}
MIT_LICENSE_BYTES = b"""MIT License

Copyright (c) 2026 rustinfer contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

REQUIRED_CUDA_DEPENDENCIES = {
    "libcublasLt.so.12",
    "libcuda.so.1",
    "libcudart.so.12",
}
ALLOWED_NATIVE_DEPENDENCIES = REQUIRED_CUDA_DEPENDENCIES | {
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
}
CALIBRATION_NVML_DEPENDENCY = "libnvidia-ml.so.1"
REQUIRED_CALIBRATION_DEPENDENCIES = REQUIRED_CUDA_DEPENDENCIES | {
    CALIBRATION_NVML_DEPENDENCY,
}
ALLOWED_CALIBRATION_DEPENDENCIES = ALLOWED_NATIVE_DEPENDENCIES | {
    CALIBRATION_NVML_DEPENDENCY,
}
FORBIDDEN_RUNTIME_TERMS = (
    "libpython",
    "pytorch",
    "python",
    "torch",
    "transformers",
    "triton",
    "pickle",
)
FORBIDDEN_ARCHIVE_SUFFIXES = (
    ".py",
    ".pyc",
    ".pyo",
    ".pyd",
    ".whl",
    ".pkl",
    ".pickle",
)
NVCC_TEMPORARY_SYMBOL_RE = re.compile(
    rb"(?:^|\x00)(?:[A-Za-z0-9_.+/-]*/)?"
    rb"tmpxft_[0-9A-Fa-f]+_[0-9A-Fa-f]+-[0-9]+_"
    rb"[A-Za-z0-9_.+-]+\.cudafe[0-9A-Za-z_.+-]*(?:\x00|$)"
)


class ReleaseContractError(RuntimeError):
    """A fail-closed release contract violation."""


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def validate_server_defaults_source(repository_root: Path) -> None:
    """Bind release metadata to the exact Rust CLI default resolver source."""
    path = repository_root / SERVER_DEFAULTS_SOURCE_PATH
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReleaseContractError(
            f"release default source is missing: {SERVER_DEFAULTS_SOURCE_PATH}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ReleaseContractError(
            f"release default source must be a regular file: {SERVER_DEFAULTS_SOURCE_PATH}"
        )
    actual = sha256_bytes(path.read_bytes())
    if actual != SERVER_DEFAULTS_SOURCE_SHA256:
        raise ReleaseContractError(
            "Rust serve defaults changed without a reviewed release-contract update: "
            f"{actual} != {SERVER_DEFAULTS_SOURCE_SHA256}"
        )


def release_root(version: str) -> str:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", version):
        raise ReleaseContractError(f"invalid semantic release version: {version!r}")
    return f"rustinfer-{version}-{ARCHIVE_SUFFIX}"


def release_manifest(version: str, source_revision: str, source_date_epoch: int) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ReleaseContractError("source revision must be a full lowercase 40-character Git SHA")
    if not 0 <= source_date_epoch <= 0xFFFFFFFF:
        raise ReleaseContractError("SOURCE_DATE_EPOCH must fit an unsigned 32-bit timestamp")
    release_root(version)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": {
            "name": "rustinfer",
            "version": version,
            "license": MIT_LICENSE_EXPRESSION,
            "target": TARGET,
            "cuda_toolkit": CUDA_TOOLKIT,
            "cuda_architectures": CUDA_ARCHITECTURES,
            "source_revision": source_revision,
            "source_date_epoch": source_date_epoch,
        },
        "features": {
            "enabled": ["cuda", "server"],
            "disabled": ["bench", "experimental"],
            "production_binary": "bin/rustinfer",
            "semantic_paths": [
                {
                    "feature_id": "iteration-command-batch",
                    "semantic_class": "E0",
                    "selector": {
                        "flag": "--execution-completion",
                        "value": "iteration-batch",
                    },
                    "default_enabled": True,
                    "release_qualified": True,
                    "availability": "supported",
                    "exact_fallback": {
                        "flag": "--execution-completion",
                        "value": "per-operation",
                    },
                    "approval_gates": ["pr15-iteration-command-batch-exact-v1"],
                    "prior_evidence_gates": [],
                    "release_evidence": ["optimization-correctness"],
                },
                {
                    "feature_id": "fused-residual-rmsnorm",
                    "semantic_class": "E0",
                    "selector": {
                        "flag": "--residual-rmsnorm",
                        "value": "fused",
                    },
                    "default_enabled": False,
                    "release_qualified": False,
                    "availability": "unsupported in the first release candidate",
                    "exact_fallback": {
                        "flag": "--residual-rmsnorm",
                        "value": "separate",
                    },
                    "approval_gates": [],
                    "prior_evidence_gates": [
                        "pr15-fused-residual-rmsnorm-exact-v1"
                    ],
                    "release_evidence": [],
                },
                {
                    "feature_id": "fixed-contiguous-37-balanced-reductions",
                    "semantic_class": "E0",
                    "selector": {
                        "flag": "--reduction-profile",
                        "value": "fixed-contiguous-37-balanced-v1",
                    },
                    "default_enabled": False,
                    "release_qualified": False,
                    "availability": "unsupported in the first release candidate",
                    "exact_fallback": {
                        "flag": "--reduction-profile",
                        "value": "canonical-v1",
                    },
                    "approval_gates": [],
                    "prior_evidence_gates": [],
                    "release_evidence": [],
                },
            ],
            "approximation_policy": {
                "included_semantic_classes": ["reference", "E0"],
                "excluded_semantic_classes": ["E1", "A1", "M1"],
                "approximation_enabled_by_default": False,
                "error_budget": None,
                "quality_budget": None,
                "exact_fallback_required": True,
            },
        },
        "defaults": {
            "bind": "127.0.0.1:8080",
            "device": 0,
            "max_active_sequences": 8,
            "max_waiting_requests": 64,
            "batch_token_budget": 512,
            "prefill_chunk_tokens": 512,
            **STABLE_OPTIMIZATION_DEFAULTS,
            "source_contract": {
                "path": str(SERVER_DEFAULTS_SOURCE_PATH),
                "sha256": SERVER_DEFAULTS_SOURCE_SHA256,
            },
            "max_weight_bytes": 2_147_483_648,
        },
        "support": {
            "host_os": "linux",
            "architecture": "x86_64",
            "cuda_toolkit": CUDA_TOOLKIT,
            "cuda_architectures": CUDA_ARCHITECTURES,
            "gpu_topology": "single CUDA device",
            "source_families": [
                {
                    "model_type": "llama",
                    "architecture": "LlamaForCausalLM",
                    "scope": "dense causal text decoder",
                    "artifact_profile": "SmolLM2-compatible ByteLevel BPE tokenizer",
                },
                {
                    "model_type": "qwen2",
                    "architecture": "Qwen2ForCausalLM",
                    "scope": "dense Qwen2 causal text decoder",
                    "artifact_profile": (
                        "Qwen2.5-compatible only for the pinned NFC/Split/ByteLevel "
                        "BPE and no-tools tokenizer_config profile"
                    ),
                },
            ],
            "checkpoint_format": "safetensors",
            "checkpoint_layouts": [
                "model.safetensors",
                "model.safetensors.index.json with declared shards",
            ],
            "checkpoint_parser_dtypes": ["BF16", "FP16"],
            "cuda_execution_dtypes": ["BF16"],
            "cuda_execution_head_dimension": 64,
            "checkpoint_provenance": (
                "required rustinfer-checkpoint.json with immutable revision, "
                "exact file inventory, byte lengths, and SHA-256 digests"
            ),
            "model_config_constraints": [
                "strict config.json with duplicate and unknown fields rejected",
                "model_type must match one declared source family; architectures may be absent, empty, or exactly that family's declared architecture",
                "hidden_act must be silu; execution requires a dense gated MLP without bias",
                "num_attention_heads * head_dim must equal hidden_size and num_key_value_heads must divide num_attention_heads; the parser requires an even head_dim and production CUDA serving requires head_dim exactly 64",
                "standard non-interleaved full RoPE only; rope_scaling must be absent or null and partial_rotary_factor must be absent or 1.0",
                "rms_norm_eps and rope_theta must be finite and positive",
                "Llama sliding_window must be absent; Qwen use_sliding_window must be absent or false",
                "torch_dtype may be bfloat16/bf16 or float16/fp16 at the parser boundary; CUDA execution requires BF16",
            ],
            "python_runtime": False,
            "network_model_download": False,
            "model_delivery": "operator-mounted local checkpoint",
        },
        "unsupported": {
            "model_architectures": [
                "model_type values other than llama and qwen2",
                "mixture-of-experts",
                "multimodal and vision-language",
                "Qwen3 and Qwen-VL",
                "encoder-only and encoder-decoder",
            ],
            "checkpoint_and_loading": [
                "quantized weights",
                "PyTorch pickle/bin and GGUF weights",
                "checkpoint transforms",
                "remote model code",
                "network model download",
            ],
            "execution": [
                "FP16 CUDA execution",
                "CUDA serving with head_dim values other than 64",
                "fused residual RMSNorm (the selector remains for development compatibility but is not candidate-qualified)",
                "fixed-contiguous-37-balanced reductions (the selector remains for development compatibility but is not candidate-qualified)",
                "CPU inference",
                "multi-GPU, tensor-parallel, pipeline-parallel, and distributed execution",
            ],
            "serving": [
                "OpenAI chat-completions, embeddings, and responses endpoints",
                "HTTP/2",
                "TLS termination",
                "built-in authentication",
            ],
            "runtime_fallbacks": [
                "Python",
                "PyTorch",
                "Transformers",
                "Triton",
            ],
        },
        "validation": {
            "pr16_release_qualification_lane": {
                "model_id": "HuggingFaceTB/SmolLM2-135M",
                "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "model_type": "llama",
                "architecture": "LlamaForCausalLM",
                "dtype": "BF16",
                "gpu_count": 1,
                "cuda_architecture": "89",
                "evidence_role": (
                    "required PR16 release qualification; broader source-family "
                    "support is not release-lane qualification"
                ),
            },
            "prior_pr12_qwen_compatibility_evidence": {
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "model_type": "qwen2",
                "architecture": "Qwen2ForCausalLM",
                "dtype": "BF16",
                "gpu_count": 1,
                "cuda_architecture": "89",
                "evidence_role": (
                    "prior PR12 compatibility evidence only; not PR16 release "
                    "qualification"
                ),
            },
        },
        "known_limitations": [
            "The release build and PR16 qualification matrix cover Linux x86_64, CUDA 12.8.1, one GPU, and sm_89 only.",
            "PR16 release qualification is pinned to SmolLM2-135M BF16; source-family support does not claim that every conforming Llama or Qwen2 checkpoint was release-qualified.",
            "Dense Qwen2.5 evidence is the prior pinned PR12 Qwen2.5-0.5B-Instruct run, not the PR16 release lane.",
            "FP16 is accepted by the strict config and safetensors parsers, but production CUDA execution is BF16-only.",
            "The config parser accepts bounded even head dimensions, but the production continuous-batch CUDA serving executor supports head_dim 64 only.",
            "Fused residual RMSNorm retains prior PR15 E0 evidence but is disabled by default and unsupported in this candidate because no current-revision fused parity report is bound to the final release gate.",
            "Fixed-contiguous-37 balanced reductions remain an opt-in development compatibility selector; optimizer regression diagnostics do not qualify that arithmetic profile for the first release candidate.",
            "Models must be local checksummed safetensors and are read into resident memory within the configured maximum weight-byte bound.",
            "Serving exposes a strict, close-delimited HTTP/1.1 completions surface; chat-completions and other OpenAI endpoints are not implemented.",
            "The first stable release candidate has no preceding stable rustinfer bundle; PR16 evidence validates conservative E0 flag restart within the current checksummed bundle, while binary rollback requires a preceding stable bundle to exist.",
        ],
        "configuration": {
            "required": ["serve", "--model PATH"],
            "optional": [
                "--model-id ID",
                "--bind ADDRESS",
                "--device ORDINAL",
                "--max-active-sequences N",
                "--max-waiting-requests N",
                "--max-sequence-tokens N",
                "--max-output-tokens N",
                "--batch-token-budget N",
                "--prefill-chunk-tokens N",
                "--kv-blocks N",
                "--residual-rmsnorm {fused,separate}",
                "--execution-completion {per-operation,iteration-batch}",
                "--reduction-profile {canonical-v1,fixed-contiguous-37-balanced-v1}",
                "--max-weight-bytes N",
                "--shutdown-on-stdin",
            ],
            "incompatible": [
                "--residual-rmsnorm fused with --execution-completion iteration-batch",
                "--reduction-profile fixed-contiguous-37-balanced-v1 with an effective --max-sequence-tokens greater than 8192",
            ],
        },
        "rollback": {
            "safe_flags": [
                "--residual-rmsnorm separate",
                "--execution-completion per-operation",
                "--reduction-profile canonical-v1",
            ],
            "validated_scope": (
                "restart the current checksummed bundle with all conservative E0 "
                "safe flags to isolate an optimization regression"
            ),
            "previous_release_scope": (
                "restart a preceding checksummed stable rustinfer bundle only when "
                "one exists; unavailable for the first stable release candidate"
            ),
            "procedure": [
                "drain or cancel active requests and stop the current server",
                "restart the current checksummed bundle with every safe flag when isolating an optimization regression",
                "when a preceding stable checksummed rustinfer bundle exists and binary rollback is required, restart that bundle with the same model and configuration",
                "verify /v1/models before restoring traffic",
            ],
        },
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_native_manifest(contents: bytes) -> list[str]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseContractError("native dependency manifest is not UTF-8") from error
    if not text.endswith("\n"):
        raise ReleaseContractError("native dependency manifest must end with a newline")
    lines = text.splitlines()
    expected_headers = [
        "schema_version=1",
        "binary=bin/rustinfer",
        "inspection=elf-dt-needed",
    ]
    if lines[:3] != expected_headers:
        raise ReleaseContractError("native dependency manifest header is invalid")
    if any(not line.startswith("dependency=") for line in lines[3:]):
        raise ReleaseContractError("native dependency manifest contains an unknown field")
    dependencies = [line.removeprefix("dependency=") for line in lines[3:]]
    if not dependencies:
        raise ReleaseContractError("native dependency manifest is empty")
    if dependencies != sorted(set(dependencies)):
        raise ReleaseContractError("native dependencies must be unique and bytewise sorted")
    unknown = set(dependencies) - ALLOWED_NATIVE_DEPENDENCIES
    if unknown:
        raise ReleaseContractError(
            "native dependency manifest contains unreviewed libraries: "
            + ", ".join(sorted(unknown))
        )
    missing = REQUIRED_CUDA_DEPENDENCIES - set(dependencies)
    if missing:
        raise ReleaseContractError(
            "native dependency manifest is missing CUDA libraries: "
            + ", ".join(sorted(missing))
        )
    lowered = text.casefold()
    if any(term in lowered for term in FORBIDDEN_RUNTIME_TERMS):
        raise ReleaseContractError("native dependency manifest contains a forbidden runtime term")
    return dependencies


def native_manifest_bytes(dependencies: list[str]) -> bytes:
    normalized = sorted(set(dependencies))
    contents = "\n".join(
        [
            "schema_version=1",
            "binary=bin/rustinfer",
            "inspection=elf-dt-needed",
            *(f"dependency={dependency}" for dependency in normalized),
            "",
        ]
    ).encode("utf-8")
    parse_native_manifest(contents)
    return contents


def inspect_elf_dynamic(binary: bytes) -> tuple[list[str], list[str]]:
    """Return DT_NEEDED and RPATH/RUNPATH strings from a Linux ELF binary."""
    if len(binary) < 64 or binary[:4] != b"\x7fELF":
        raise ReleaseContractError("CLI binary is not an ELF file")
    elf_class = binary[4]
    if binary[5] != 1:
        raise ReleaseContractError("CLI binary must be little-endian ELF")
    if elf_class == 2:
        header = struct.unpack_from("<HHIQQQIHHHHHH", binary, 16)
        elf_type, machine = header[0], header[1]
        phoff, phentsize, phnum = header[4], header[8], header[9]
        ph_format, ph_size = "<IIQQQQQQ", 56
        dyn_format, dyn_size = "<qQ", 16
    elif elf_class == 1:
        header = struct.unpack_from("<HHIIIIIHHHHHH", binary, 16)
        elf_type, machine = header[0], header[1]
        phoff, phentsize, phnum = header[4], header[8], header[9]
        ph_format, ph_size = "<IIIIIIII", 32
        dyn_format, dyn_size = "<iI", 8
    else:
        raise ReleaseContractError("CLI binary has an unsupported ELF class")
    if elf_type not in (2, 3) or machine != 62:
        raise ReleaseContractError("CLI binary must be Linux x86_64 ET_EXEC or ET_DYN")
    if phentsize != ph_size or phnum == 0 or phoff + phentsize * phnum > len(binary):
        raise ReleaseContractError("CLI binary has an invalid program-header table")

    load_segments: list[tuple[int, int, int]] = []
    dynamic_segment: tuple[int, int] | None = None
    for index in range(phnum):
        values = struct.unpack_from(ph_format, binary, phoff + index * phentsize)
        if elf_class == 2:
            segment_type, offset, virtual_address, file_size = (
                values[0], values[2], values[3], values[5]
            )
        else:
            segment_type, offset, virtual_address, file_size = (
                values[0], values[1], values[2], values[4]
            )
        if offset + file_size > len(binary):
            raise ReleaseContractError("CLI binary contains an out-of-range segment")
        if segment_type == 1:
            load_segments.append((virtual_address, offset, file_size))
        elif segment_type == 2:
            if dynamic_segment is not None:
                raise ReleaseContractError("CLI binary contains multiple PT_DYNAMIC segments")
            dynamic_segment = (offset, file_size)
    if dynamic_segment is None:
        raise ReleaseContractError("CLI binary has no PT_DYNAMIC segment")

    dynamic_offset, dynamic_size = dynamic_segment
    if dynamic_size % dyn_size != 0:
        raise ReleaseContractError("CLI binary has a truncated PT_DYNAMIC segment")
    needed_offsets: list[int] = []
    path_offsets: list[int] = []
    string_table_address = None
    string_table_size = None
    found_null = False
    for offset in range(dynamic_offset, dynamic_offset + dynamic_size, dyn_size):
        tag, value = struct.unpack_from(dyn_format, binary, offset)
        if tag == 0:
            found_null = True
            break
        if tag == 1:
            needed_offsets.append(value)
        elif tag == 5:
            string_table_address = value
        elif tag == 10:
            string_table_size = value
        elif tag in (15, 29):
            path_offsets.append(value)
    if not found_null or string_table_address is None or string_table_size is None:
        raise ReleaseContractError("CLI binary has incomplete dynamic string metadata")

    string_table_offset = None
    for virtual_address, file_offset, file_size in load_segments:
        delta = string_table_address - virtual_address
        if 0 <= delta < file_size and delta + string_table_size <= file_size:
            string_table_offset = file_offset + delta
            break
    if string_table_offset is None or string_table_offset + string_table_size > len(binary):
        raise ReleaseContractError("CLI binary dynamic string table is out of range")
    strings = binary[string_table_offset : string_table_offset + string_table_size]

    def read_string(offset: int) -> str:
        if not 0 <= offset < len(strings):
            raise ReleaseContractError("CLI binary dynamic string offset is out of range")
        end = strings.find(b"\0", offset)
        if end < 0:
            raise ReleaseContractError("CLI binary dynamic string is unterminated")
        try:
            return strings[offset:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise ReleaseContractError("CLI binary dynamic string is not ASCII") from error

    dependencies = [read_string(offset) for offset in needed_offsets]
    paths = [read_string(offset) for offset in path_offsets]
    if not dependencies:
        raise ReleaseContractError("CLI binary has no DT_NEEDED entries")
    if len(dependencies) != len(set(dependencies)):
        raise ReleaseContractError("CLI binary repeats a DT_NEEDED entry")
    return sorted(dependencies), paths


def validate_binary(binary: bytes) -> list[str]:
    if NVCC_TEMPORARY_SYMBOL_RE.search(binary) is not None:
        raise ReleaseContractError(
            "CLI binary contains a nondeterministic nvcc temporary symbol name"
        )
    dependencies, dynamic_paths = inspect_elf_dynamic(binary)
    if any(dynamic_paths):
        raise ReleaseContractError("CLI binary must not contain DT_RPATH or DT_RUNPATH")
    native_manifest_bytes(dependencies)
    return dependencies


def validate_calibration_binary(binary: bytes) -> list[str]:
    """Validate the development-only native calibration executable.

    The production server ABI deliberately excludes NVML.  The calibration
    producer is a distinct role and must link the reviewed NVML soname so it
    can bind the captured hardware state into its evidence manifest.
    """

    if NVCC_TEMPORARY_SYMBOL_RE.search(binary) is not None:
        raise ReleaseContractError(
            "calibration binary contains a nondeterministic nvcc temporary symbol name"
        )
    dependencies, dynamic_paths = inspect_elf_dynamic(binary)
    if any(dynamic_paths):
        raise ReleaseContractError(
            "calibration binary must not contain DT_RPATH or DT_RUNPATH"
        )
    unknown = set(dependencies) - ALLOWED_CALIBRATION_DEPENDENCIES
    if unknown:
        raise ReleaseContractError(
            "calibration binary contains unreviewed libraries: "
            + ", ".join(sorted(unknown))
        )
    missing = REQUIRED_CALIBRATION_DEPENDENCIES - set(dependencies)
    if missing:
        raise ReleaseContractError(
            "calibration binary is missing reviewed CUDA/NVML libraries: "
            + ", ".join(sorted(missing))
        )
    lowered = "\n".join(dependencies).casefold()
    if any(term in lowered for term in FORBIDDEN_RUNTIME_TERMS):
        raise ReleaseContractError(
            "calibration binary dependencies contain a forbidden runtime term"
        )
    return dependencies


def validate_license(contents: bytes) -> None:
    if contents != MIT_LICENSE_BYTES:
        raise ReleaseContractError(
            "root LICENSE must exactly match the reviewed MIT license bytes for "
            "Copyright (c) 2026 rustinfer contributors"
        )


def load_json_object(contents: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ReleaseContractError(f"{label} must be a JSON object")
    return value


def read_workspace_version(repository_root: Path) -> str:
    cargo_toml = repository_root / "Cargo.toml"
    try:
        contents = cargo_toml.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseContractError(f"cannot read {cargo_toml}: {error}") from error
    workspace_package = contents.partition("[workspace.package]")[2].partition("[")[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', workspace_package, re.MULTILINE)
    if match is None:
        raise ReleaseContractError("workspace package version is missing")
    version = match.group(1)
    release_root(version)
    return version


def validate_license_metadata(repository_root: Path) -> str:
    cargo_toml = repository_root / "Cargo.toml"
    try:
        root_manifest = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseContractError(f"cannot parse {cargo_toml}: {error}") from error
    workspace = root_manifest.get("workspace")
    if not isinstance(workspace, dict):
        raise ReleaseContractError("root Cargo.toml has no workspace table")
    package = workspace.get("package")
    if (
        not isinstance(package, dict)
        or package.get("license") != MIT_LICENSE_EXPRESSION
    ):
        raise ReleaseContractError(
            'workspace.package.license must exactly equal the reviewed SPDX expression "MIT"'
        )
    if "license-file" in package:
        raise ReleaseContractError(
            "workspace.package.license-file is forbidden; release metadata must use "
            "only the reviewed MIT SPDX expression"
        )
    members = workspace.get("members")
    if not isinstance(members, list) or not all(
        isinstance(member, str) for member in members
    ):
        raise ReleaseContractError("workspace members must be an explicit string list")
    for member in members:
        member_manifest_path = repository_root / member / "Cargo.toml"
        try:
            member_manifest = tomllib.loads(
                member_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ReleaseContractError(
                f"cannot parse {member_manifest_path}: {error}"
            ) from error
        member_package = member_manifest.get("package")
        if (
            not isinstance(member_package, dict)
            or member_package.get("license") != {"workspace": True}
        ):
            raise ReleaseContractError(
                f"{member_manifest_path}: package.license must be license.workspace = true"
            )
        if "license-file" in member_package:
            raise ReleaseContractError(
                f"{member_manifest_path}: package.license-file is forbidden; "
                "inherit only the reviewed MIT SPDX expression"
            )
    return MIT_LICENSE_EXPRESSION
