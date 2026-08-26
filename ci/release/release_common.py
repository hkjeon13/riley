#!/usr/bin/env python3
"""Shared, standard-library-only release bundle contract helpers."""

from __future__ import annotations

import hashlib
import json
import re
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
        },
        "defaults": {
            "bind": "127.0.0.1:8080",
            "device": 0,
            "max_active_sequences": 8,
            "max_waiting_requests": 64,
            "batch_token_budget": 512,
            "prefill_chunk_tokens": 512,
            "execution_completion": "iteration-batch",
            "residual_rmsnorm": "separate",
            "max_weight_bytes": 2_147_483_648,
        },
        "support": {
            "host_os": "linux",
            "architecture": "x86_64",
            "cuda_toolkit": CUDA_TOOLKIT,
            "cuda_architectures": CUDA_ARCHITECTURES,
            "checkpoint_format": "safetensors",
            "python_runtime": False,
            "network_model_download": False,
            "model_delivery": "operator-mounted local checkpoint",
        },
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
                "--max-weight-bytes N",
                "--shutdown-on-stdin",
            ],
            "incompatible": [
                "--residual-rmsnorm fused with --execution-completion iteration-batch"
            ],
        },
        "rollback": {
            "safe_flags": [
                "--residual-rmsnorm separate",
                "--execution-completion per-operation",
            ],
            "procedure": [
                "drain or cancel active requests and stop the current server",
                "restart the preceding checksummed release bundle",
                "use the safe flags when isolating an optimization regression",
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
