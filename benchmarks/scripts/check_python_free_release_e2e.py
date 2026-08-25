#!/usr/bin/env python3
"""Replay Python-free real-model release E2E evidence without running CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import struct
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence


RAW_SCHEMA = "rustinfer.python-free-release-e2e-raw.v2"
GOLDEN_SCHEMA = "rustinfer.python-free-release-e2e-golden.v1"
REPORT_SCHEMA = "rustinfer.release-gate-attestation.v1"
GATE = "python-free-clean-runtime-e2e"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
CONTAINER_RE = re.compile(r"^[0-9a-f]{12,64}$")
MODEL_PATH_RE = re.compile(r"^[A-Za-z0-9._/+@=-]+$")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_FIXED_MEMBER_BYTES = 512 * 1024 * 1024
MAX_MODEL_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_RAW_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024 + 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 256

RAW_ARCHIVE_PAYLOADS = (
    "cancellation-request.raw",
    "cancellation-response-prefix.raw",
    "container-first-post.json",
    "container-first-pre.json",
    "container-first-runtime.json",
    "container-second-post.json",
    "container-second-pre.json",
    "container-second-runtime.json",
    "correctness-golden.json",
    "http-greedy-stream.raw",
    "http-greedy.raw",
    "http-metrics-after.raw",
    "http-metrics-before.raw",
    "http-models.raw",
    "http-readyz.raw",
    "http-sampling-first.raw",
    "http-sampling-second.raw",
    "image-binary",
    "image-inspect.json",
    "image-ldd.txt",
    "image-native-dependencies.txt",
    "image-python-scan.txt",
    "image-readelf.txt",
    "model-SHA256SUMS",
    "process-first-pre.txt",
    "process-first-runtime.txt",
    "process-second-pre.txt",
    "process-second-runtime.txt",
    "raw-evidence.json",
    "repeat-shutdown-metrics.json",
    "request-greedy-stream.json",
    "request-greedy.json",
    "request-sampling.json",
    "shutdown-metrics.json",
)
RAW_ARCHIVE_MEMBERS = {*RAW_ARCHIVE_PAYLOADS, "SHA256SUMS"}
TOKENIZER_ARTIFACT_FILENAMES = (
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
REQUIRED_DEPENDENCIES = {"libcublasLt.so.12", "libcuda.so.1", "libcudart.so.12"}
ALLOWED_DEPENDENCIES = REQUIRED_DEPENDENCIES | {
    "ld-linux-x86-64.so.2", "libc.so.6", "libdl.so.2", "libgcc_s.so.1",
    "libm.so.6", "libpthread.so.0", "librt.so.1",
}
FORBIDDEN_RE = re.compile(r"python|pip|pytorch|torch|transformers|triton|pickle", re.I)
CHECK_IDS = (
    "release_bundle_verified", "no_python_executable", "no_python_child",
    "no_forbidden_runtime_artifact", "native_dependencies_verified", "model_load",
    "prefill", "decode", "greedy_golden", "sampling", "streaming", "cancellation",
    "graceful_shutdown",
)
CORRECTNESS_GATE = "smollm2-fp32-bf16-native-e0-v2"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_CONFIG_SHA256 = "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
MODEL_WEIGHTS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
TOKENIZER_AGGREGATE_SHA256 = "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db"
TOKENIZER_JSON_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
TOKENIZER_FILES_SHA256 = {
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
    "tokenizer.json": TOKENIZER_JSON_SHA256,
    "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}

_RELEASE_DIR = Path(__file__).resolve().parents[2] / "ci/release"
_COMMON_SPEC = importlib.util.spec_from_file_location("python_free_e2e_release_common", _RELEASE_DIR / "release_common.py")
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot load release bundle contract")
release_common = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules[_COMMON_SPEC.name] = release_common
_COMMON_SPEC.loader.exec_module(release_common)
_VERIFY_SPEC = importlib.util.spec_from_file_location("python_free_e2e_release_verify", _RELEASE_DIR / "verify_release_bundle.py")
if _VERIFY_SPEC is None or _VERIFY_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot load release bundle verifier")
release_verify = importlib.util.module_from_spec(_VERIFY_SPEC)
_VERIFY_SPEC.loader.exec_module(release_verify)


class EvidenceError(ValueError):
    """Raw evidence or one of its immutable bindings is invalid."""


def _fail(path: str, message: str) -> NoReturn:
    raise EvidenceError(f"{path}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail("JSON", f"non-finite number {value!r} is forbidden")


def _parse_json_value(raw: bytes, label: str) -> Any:
    if len(raw) > MAX_JSON_BYTES:
        _fail(label, f"exceeds {MAX_JSON_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"cannot read strict UTF-8 JSON: {error}")


def _parse_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    value = _parse_json_value(raw, label)
    if not isinstance(value, dict):
        _fail(label, "root must be an object")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        raw = path.read_bytes()
    except OSError as error:
        _fail(label, f"cannot read file: {error}")
    return _parse_json_bytes(raw, label), raw


def _closed(value: Any, keys: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing:
        _fail(path, f"missing fields: {', '.join(missing)}")
    if extra:
        _fail(path, f"unknown fields: {', '.join(extra)}")
    return value


def _string(value: Any, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, "has an invalid format")
    return value


def _sha(value: Any, path: str) -> str:
    return _string(value, path, SHA_RE)


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        _fail(path, "must be finite and positive" if positive else "must be finite and nonnegative")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _file_sha256(path: Path, label: str) -> str:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        _fail(label, f"cannot hash file: {error}")


def model_tree_manifest_bytes(model_dir: Path) -> bytes:
    try:
        root = model_dir.resolve(strict=True)
        if not root.is_dir():
            _fail("--model-dir", "must be a directory")
        paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    except OSError as error:
        _fail("--model-dir", f"cannot enumerate model tree: {error}")
    lines: list[bytes] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _fail("--model-dir", f"contains non-regular entry {relative!r}")
        if MODEL_PATH_RE.fullmatch(relative) is None:
            _fail("--model-dir", "model paths must use the safe ASCII path alphabet")
        lines.append(f"{_file_sha256(path, 'model file')}  {relative}\n".encode("ascii"))
    if not lines:
        _fail("--model-dir", "contains no regular files")
    return b"".join(lines)


def model_tree_sha256(model_dir: Path) -> str:
    return hashlib.sha256(model_tree_manifest_bytes(model_dir)).hexdigest()


def _parse_model_manifest(raw: bytes) -> dict[str, str]:
    if not raw or len(raw) > MAX_JSON_BYTES or not raw.endswith(b"\n"):
        _fail("raw archive.model-SHA256SUMS", "must be a bounded newline-terminated manifest")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail("raw archive.model-SHA256SUMS", f"must be ASCII: {error}")
    result: dict[str, str] = {}
    previous: str | None = None
    for index, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/+@=-]+)", line)
        if match is None:
            _fail("raw archive.model-SHA256SUMS", f"invalid line {index}")
        digest, name = match.groups()
        pure = PurePosixPath(name)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            _fail("raw archive.model-SHA256SUMS", f"unsafe path {name!r}")
        if name in result or (previous is not None and name <= previous):
            _fail("raw archive.model-SHA256SUMS", "paths must be unique and bytewise sorted")
        result[name] = digest
        previous = name
    if not result:
        _fail("raw archive.model-SHA256SUMS", "must contain model files")
    return result


def _tokenizer_aggregate_sha256(model_files: Mapping[str, str]) -> str:
    missing = sorted(set(TOKENIZER_ARTIFACT_FILENAMES) - set(model_files))
    if missing:
        _fail("raw archive.model-SHA256SUMS", f"missing tokenizer files: {missing}")
    canonical = json.dumps(
        {name: model_files[name] for name in TOKENIZER_ARTIFACT_FILENAMES},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _parse_archive_checksums(raw: bytes) -> dict[str, str]:
    if not raw or not raw.endswith(b"\n") or len(raw) > MAX_JSON_BYTES:
        _fail("raw archive.SHA256SUMS", "must be bounded and newline terminated")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        _fail("raw archive.SHA256SUMS", f"must be ASCII: {error}")
    result: dict[str, str] = {}
    previous: str | None = None
    for index, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/+@=-]+)", line)
        if match is None:
            _fail("raw archive.SHA256SUMS", f"invalid line {index}")
        digest, name = match.groups()
        if name == "SHA256SUMS" or (name not in RAW_ARCHIVE_MEMBERS and not name.startswith("model/")):
            _fail("raw archive.SHA256SUMS", f"unreviewed path {name!r}")
        if name in result or (previous is not None and name <= previous):
            _fail("raw archive.SHA256SUMS", "paths must be unique and bytewise sorted")
        result[name] = digest
        previous = name
    return result


def _validate_tar_layout(path: Path, members: Sequence[tarfile.TarInfo]) -> None:
    data_end = 0
    try:
        with path.open("rb") as source:
            for member in members:
                if member.offset != data_end or member.offset_data != member.offset + 512:
                    _fail("raw evidence archive", "contains non-canonical headers or hidden records")
                source.seek(member.offset + 257)
                if source.read(8) != b"ustar\x0000":
                    _fail("raw evidence archive", "must use canonical POSIX USTAR headers")
                data_end = member.offset_data + ((member.size + 511) // 512) * 512
            expected_size = ((data_end + 1024 + 10239) // 10240) * 10240
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                _fail("raw evidence archive", "has hidden, missing, or non-canonical trailing records")
            source.seek(data_end)
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                if any(chunk):
                    _fail("raw evidence archive", "trailing records must be all zero")
    except OSError as error:
        _fail("raw evidence archive", f"cannot verify raw tar layout: {error}")


def _validate_safetensors(prefix: bytes, total_size: int) -> None:
    if len(prefix) < 9:
        _fail("raw archive.model/model.safetensors", "truncated safetensors header")
    header_size = struct.unpack_from("<Q", prefix, 0)[0]
    if header_size <= 1 or header_size > MAX_JSON_BYTES or 8 + header_size > total_size:
        _fail("raw archive.model/model.safetensors", "invalid safetensors header length")
    if len(prefix) < 8 + header_size:
        _fail("raw archive.model/model.safetensors", "safetensors header exceeds replay prefix")
    header = _parse_json_bytes(prefix[8 : 8 + header_size], "model.safetensors header")
    data_size = total_size - 8 - header_size
    intervals: list[tuple[int, int]] = []
    for name, value in header.items():
        if name == "__metadata__":
            if not isinstance(value, dict):
                _fail("model.safetensors header.__metadata__", "must be an object")
            continue
        tensor = _closed(value, {"dtype", "shape", "data_offsets"}, f"model.safetensors header.{name}")
        _string(tensor["dtype"], f"model.safetensors header.{name}.dtype")
        if not isinstance(tensor["shape"], list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tensor["shape"]
        ):
            _fail(f"model.safetensors header.{name}.shape", "must be nonnegative integer dimensions")
        offsets = tensor["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            _fail(f"model.safetensors header.{name}.data_offsets", "must contain two offsets")
        start = _integer(offsets[0], f"model.safetensors header.{name}.data_offsets[0]")
        end = _integer(offsets[1], f"model.safetensors header.{name}.data_offsets[1]")
        if end <= start or end > data_size:
            _fail(f"model.safetensors header.{name}.data_offsets", "is empty or out of range")
        intervals.append((start, end))
    if not intervals or sorted(intervals)[0][0] != 0 or sorted(intervals)[-1][1] != data_size:
        _fail("model.safetensors header", "tensor ranges do not cover the data section")
    ordered = sorted(intervals)
    if any(left[1] != right[0] for left, right in zip(ordered, ordered[1:])):
        _fail("model.safetensors header", "tensor ranges overlap or contain gaps")


def load_raw_evidence_archive(path: Path) -> dict[str, Any]:
    """Load a canonical v2 tar and hash every archived observation/model byte."""

    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail("raw evidence archive", "must be a regular file, not a link or device")
        if metadata.st_size <= 0 or metadata.st_size > MAX_RAW_ARCHIVE_BYTES:
            _fail("raw evidence archive", "is empty or exceeds the reviewed bound")
        with tarfile.open(path, "r:") as archive:
            members = archive.getmembers()
            if not members or len(members) > MAX_ARCHIVE_MEMBERS:
                _fail("raw evidence archive", "invalid member count")
            names = [member.name for member in members]
            if names != sorted(names) or len(names) != len(set(names)):
                _fail("raw evidence archive", "members must be unique and bytewise sorted")
            payloads: dict[str, bytes] = {}
            digests: dict[str, str] = {}
            sizes: dict[str, int] = {}
            model_prefixes: dict[str, bytes] = {}
            total = 0
            for member in members:
                name = member.name
                if name not in RAW_ARCHIVE_MEMBERS and not name.startswith("model/"):
                    _fail("raw evidence archive", f"unexpected member {name!r}")
                if not member.isreg() or member.pax_headers or member.linkname:
                    _fail("raw evidence archive", f"member must be a plain regular USTAR file: {name}")
                expected_mode = 0o755 if name == "image-binary" else 0o644
                if (member.uid, member.gid, member.uname, member.gname, member.mode, member.mtime) != (0, 0, "", "", expected_mode, 0):
                    _fail("raw evidence archive", f"non-canonical metadata for {name}")
                maximum = MAX_MODEL_MEMBER_BYTES if name.startswith("model/") else MAX_FIXED_MEMBER_BYTES
                if member.size <= 0 or member.size > maximum:
                    _fail("raw evidence archive", f"invalid size for {name}")
                total += member.size
                if total > MAX_RAW_ARCHIVE_BYTES:
                    _fail("raw evidence archive", "uncompressed payload exceeds the reviewed bound")
                source = archive.extractfile(member)
                if source is None:
                    _fail("raw evidence archive", f"cannot read {name}")
                digest = hashlib.sha256()
                saved = bytearray()
                save_limit = member.size if not name.startswith("model/") else min(member.size, MAX_JSON_BYTES + 8)
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        _fail("raw evidence archive", f"truncated member {name}")
                    digest.update(chunk)
                    if len(saved) < save_limit:
                        saved.extend(chunk[: save_limit - len(saved)])
                    remaining -= len(chunk)
                if source.read(1):
                    _fail("raw evidence archive", f"oversized member {name}")
                digests[name] = digest.hexdigest()
                sizes[name] = member.size
                if name.startswith("model/"):
                    model_prefixes[name.removeprefix("model/")] = bytes(saved)
                else:
                    payloads[name] = bytes(saved)
    except EvidenceError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail("raw evidence archive", f"cannot read uncompressed tar: {error}")

    _validate_tar_layout(path, members)
    missing_fixed = sorted(RAW_ARCHIVE_MEMBERS - set(digests))
    if missing_fixed:
        _fail("raw evidence archive", f"missing fixed v2 members: {missing_fixed}")
    checksums = _parse_archive_checksums(payloads["SHA256SUMS"])
    expected_names = set(digests) - {"SHA256SUMS"}
    if set(checksums) != expected_names:
        _fail("raw archive.SHA256SUMS", "does not close over the exact archive inventory")
    for name, expected in checksums.items():
        if digests[name] != expected:
            _fail("raw archive.SHA256SUMS", f"digest mismatch for {name}")
    model_files = _parse_model_manifest(payloads["model-SHA256SUMS"])
    if {f"model/{name}" for name in model_files} != {name for name in digests if name.startswith("model/")}:
        _fail("raw archive.model-SHA256SUMS", "does not close over the exact archived model files")
    for name, expected in model_files.items():
        if digests[f"model/{name}"] != expected:
            _fail("raw archive.model-SHA256SUMS", f"actual model bytes mismatch for {name}")
    for required in ("config.json", "model.safetensors", "tokenizer.json", *TOKENIZER_ARTIFACT_FILENAMES):
        if required not in model_files:
            _fail("raw archive.model-SHA256SUMS", f"missing required model subject {required}")
    _parse_json_bytes(model_prefixes["config.json"], "archived model config.json")
    _parse_json_bytes(model_prefixes["tokenizer.json"], "archived model tokenizer.json")
    _validate_safetensors(model_prefixes["model.safetensors"], sizes["model/model.safetensors"])
    return {
        "archive_sha256": _file_sha256(path, "raw evidence archive"),
        "payloads": payloads,
        "payload_digests": digests,
        "raw": _parse_json_bytes(payloads["raw-evidence.json"], "raw evidence"),
        "golden": _parse_json_bytes(payloads["correctness-golden.json"], "correctness golden"),
        "shutdown": _parse_json_bytes(payloads["shutdown-metrics.json"], "shutdown metrics"),
        "repeat_shutdown": _parse_json_bytes(payloads["repeat-shutdown-metrics.json"], "repeat shutdown metrics"),
        "model_manifest": payloads["model-SHA256SUMS"],
        "model_files": model_files,
        "model_sizes": {name.removeprefix("model/"): size for name, size in sizes.items() if name.startswith("model/")},
    }


def _verify_bundle(bundle: Path, binary_sha256: str, revision: str) -> None:
    try:
        release_verify.verify_bundle(bundle)
        with tarfile.open(bundle, "r:gz") as archive:
            binaries = [member for member in archive.getmembers() if member.name.endswith("/bin/rustinfer")]
            manifests = [member for member in archive.getmembers() if member.name.endswith("/manifest/release.json")]
            if len(binaries) != 1 or len(manifests) != 1:
                _fail("--release-bundle", "must contain one binary and one release manifest")
            binary_file = archive.extractfile(binaries[0])
            manifest_file = archive.extractfile(manifests[0])
            if binary_file is None or manifest_file is None:
                _fail("--release-bundle", "cannot read embedded release files")
            embedded_binary = hashlib.sha256(binary_file.read()).hexdigest()
            manifest = _parse_json_bytes(manifest_file.read(), "release bundle manifest")
    except EvidenceError:
        raise
    except (OSError, tarfile.TarError, release_common.ReleaseContractError) as error:
        _fail("--release-bundle", f"cannot inspect release bundle: {error}")
    if embedded_binary != binary_sha256:
        _fail("--release-bundle", "embedded binary differs from --release-binary")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("source_revision") != revision:
        _fail("--release-bundle", "embedded source revision differs from candidate")


def _validate_golden(document: Mapping[str, Any]) -> dict[str, Any]:
    row = _closed(
        document,
        {
            "schema_version", "correctness_gate_id", "correctness_report_sha256",
            "source_revision", "model_id", "model_revision", "config_sha256", "weights_sha256",
            "tokenizer_aggregate_sha256", "tokenizer_json_sha256", "prompt", "max_tokens",
            "expected_greedy_text_sha256",
        },
        "correctness golden",
    )
    if row["schema_version"] != GOLDEN_SCHEMA:
        _fail("correctness golden.schema_version", f"must be {GOLDEN_SCHEMA}")
    prompt = _string(row["prompt"], "correctness golden.prompt")
    if len(prompt.encode("utf-8")) > 16 * 1024 or "\n" in prompt or "\r" in prompt:
        _fail("correctness golden.prompt", "must be a bounded single line")
    return {
        "correctness_gate_id": _string(row["correctness_gate_id"], "correctness golden.correctness_gate_id"),
        "correctness_report_sha256": _sha(row["correctness_report_sha256"], "correctness golden.correctness_report_sha256"),
        "source_revision": _string(row["source_revision"], "correctness golden.source_revision", GIT_RE),
        "model_id": _string(row["model_id"], "correctness golden.model_id", MODEL_ID_RE),
        "model_revision": _string(row["model_revision"], "correctness golden.model_revision"),
        "config_sha256": _sha(row["config_sha256"], "correctness golden.config_sha256"),
        "weights_sha256": _sha(row["weights_sha256"], "correctness golden.weights_sha256"),
        "tokenizer_aggregate_sha256": _sha(row["tokenizer_aggregate_sha256"], "correctness golden.tokenizer_aggregate_sha256"),
        "tokenizer_json_sha256": _sha(row["tokenizer_json_sha256"], "correctness golden.tokenizer_json_sha256"),
        "prompt": prompt,
        "max_tokens": _integer(row["max_tokens"], "correctness golden.max_tokens", 2),
        "expected_greedy_text_sha256": _sha(row["expected_greedy_text_sha256"], "correctness golden.expected_greedy_text_sha256"),
    }


def _validate_correctness_report(document: Mapping[str, Any]) -> dict[str, str]:
    if document.get("gate_id") != CORRECTNESS_GATE or document.get("status") != "pass":
        _fail("correctness report", "must be the passing native E0 v2 gate")
    bindings = document.get("bindings")
    if not isinstance(bindings, dict):
        _fail("correctness report.bindings", "must be an object")
    required = {
        "candidate_git_revision", "candidate_git_status_sha256", "model_id",
        "model_revision", "config_sha256", "weights_sha256", "tokenizer_sha256",
    }
    if not required <= set(bindings):
        _fail("correctness report.bindings", f"missing fields: {sorted(required - set(bindings))}")
    if bindings["candidate_git_status_sha256"] != hashlib.sha256(b"").hexdigest():
        _fail("correctness report.bindings.candidate_git_status_sha256", "candidate tree was dirty")
    return {
        "correctness_gate_id": CORRECTNESS_GATE,
        "source_revision": _string(bindings["candidate_git_revision"], "correctness report.bindings.candidate_git_revision", GIT_RE),
        "model_id": _string(bindings["model_id"], "correctness report.bindings.model_id", MODEL_ID_RE),
        "model_revision": _string(bindings["model_revision"], "correctness report.bindings.model_revision"),
        "config_sha256": _sha(bindings["config_sha256"], "correctness report.bindings.config_sha256"),
        "weights_sha256": _sha(bindings["weights_sha256"], "correctness report.bindings.weights_sha256"),
        "tokenizer_aggregate_sha256": _sha(bindings["tokenizer_sha256"], "correctness report.bindings.tokenizer_sha256"),
    }


def _validate_metrics(value: Any, path: str, *, final: bool) -> dict[str, Any]:
    row = _closed(value, {"active_requests", "waiting_requests", "kv_allocated_blocks", "allocation", "counters"}, path)
    allocation = _closed(
        row["allocation"],
        {"device_live_count", "device_live_bytes", "pinned_live_count", "pinned_live_bytes"},
        f"{path}.allocation",
    )
    counters = _closed(row["counters"], {"cancellations", "disconnects", "overloads", "dropped_observations"}, f"{path}.counters")
    result = {
        "active_requests": _integer(row["active_requests"], f"{path}.active_requests"),
        "waiting_requests": _integer(row["waiting_requests"], f"{path}.waiting_requests"),
        "kv_allocated_blocks": _integer(row["kv_allocated_blocks"], f"{path}.kv_allocated_blocks"),
        "allocation": {key: _integer(item, f"{path}.allocation.{key}") for key, item in allocation.items()},
        "counters": {key: _integer(item, f"{path}.counters.{key}") for key, item in counters.items()},
    }
    if final and any((result["active_requests"], result["waiting_requests"], result["kv_allocated_blocks"], *result["allocation"].values())):
        _fail(path, "final live-resource gauges must all be zero")
    return result


def _validate_process_claims(value: Any, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail(path, "must record at least the server process")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        row = _closed(item, {"pid", "ppid", "comm", "args"}, f"{path}[{index}]")
        process = {
            "pid": _integer(row["pid"], f"{path}[{index}].pid", 1),
            "ppid": _integer(row["ppid"], f"{path}[{index}].ppid"),
            "comm": _string(row["comm"], f"{path}[{index}].comm"),
            "args": _string(row["args"], f"{path}[{index}].args"),
        }
        if FORBIDDEN_RE.search(process["comm"] + " " + process["args"]):
            _fail(f"{path}[{index}]", "contains a forbidden Python-family process")
        result.append(process)
    if not any(row["comm"] == "rustinfer" for row in result):
        _fail(path, "does not contain the rustinfer server")
    return result


def _validate_raw(document: Mapping[str, Any]) -> dict[str, Any]:
    row = _closed(document, {"schema_version", "run_id", "recorded_at_utc", "status", "source", "release", "model", "runtime", "observations"}, "raw")
    if row["schema_version"] != RAW_SCHEMA or row["status"] != "success":
        _fail("raw", "schema must be v2 and status must be success")
    _string(row["run_id"], "raw.run_id", ID_RE)
    _string(row["recorded_at_utc"], "raw.recorded_at_utc", UTC_RE)
    source = _closed(row["source"], {"git_revision", "git_dirty", "source_archive_sha256"}, "raw.source")
    if source["git_dirty"] is not False:
        _fail("raw.source.git_dirty", "must be false")
    source_result = {
        "git_revision": _string(source["git_revision"], "raw.source.git_revision", GIT_RE),
        "git_dirty": False,
        "source_archive_sha256": _sha(source["source_archive_sha256"], "raw.source.source_archive_sha256"),
    }
    release = _closed(row["release"], {"binary_sha256", "bundle_sha256", "image_sha256"}, "raw.release")
    release_result = {key: _sha(value, f"raw.release.{key}") for key, value in release.items()}
    model = _closed(
        row["model"],
        {"model_id", "model_revision", "model_tree_sha256", "config_sha256", "weights_sha256", "tokenizer_aggregate_sha256", "tokenizer_json_sha256", "correctness_gate_id", "correctness_report_sha256", "correctness_golden_sha256"},
        "raw.model",
    )
    model_result = {
        "model_id": _string(model["model_id"], "raw.model.model_id", MODEL_ID_RE),
        "model_revision": _string(model["model_revision"], "raw.model.model_revision"),
        "model_tree_sha256": _sha(model["model_tree_sha256"], "raw.model.model_tree_sha256"),
        "config_sha256": _sha(model["config_sha256"], "raw.model.config_sha256"),
        "weights_sha256": _sha(model["weights_sha256"], "raw.model.weights_sha256"),
        "tokenizer_aggregate_sha256": _sha(model["tokenizer_aggregate_sha256"], "raw.model.tokenizer_aggregate_sha256"),
        "tokenizer_json_sha256": _sha(model["tokenizer_json_sha256"], "raw.model.tokenizer_json_sha256"),
        "correctness_gate_id": _string(model["correctness_gate_id"], "raw.model.correctness_gate_id"),
        "correctness_report_sha256": _sha(model["correctness_report_sha256"], "raw.model.correctness_report_sha256"),
        "correctness_golden_sha256": _sha(model["correctness_golden_sha256"], "raw.model.correctness_golden_sha256"),
    }
    runtime = _closed(row["runtime"], {"container_ids", "network_mode", "image_id", "image_binary_sha256"}, "raw.runtime")
    if runtime["network_mode"] != "none":
        _fail("raw.runtime.network_mode", "must be none")
    image_id = _string(runtime["image_id"], "raw.runtime.image_id")
    if not image_id.startswith("sha256:"):
        _fail("raw.runtime.image_id", "must be sha256:<digest>")
    _sha(image_id.removeprefix("sha256:"), "raw.runtime.image_id")
    container_ids = runtime["container_ids"]
    if not isinstance(container_ids, list) or len(container_ids) != 2:
        _fail("raw.runtime.container_ids", "must record exactly two clean-start containers")
    ids = [_string(value, f"raw.runtime.container_ids[{index}]", CONTAINER_RE) for index, value in enumerate(container_ids)]
    if len(set(ids)) != 2:
        _fail("raw.runtime.container_ids", "clean-start container IDs must be distinct")
    runtime_result = {"container_ids": ids, "network_mode": "none", "image_id": image_id, "image_binary_sha256": _sha(runtime["image_binary_sha256"], "raw.runtime.image_binary_sha256")}

    observations = _closed(row["observations"], {"readyz", "models", "greedy", "sampling", "cancellation", "shutdown", "python_free"}, "raw.observations")
    ready = _closed(observations["readyz"], {"http_status", "ready", "accepting"}, "raw.observations.readyz")
    ready_result = {
        "http_status": _integer(ready["http_status"], "readyz.http_status"),
        "ready": _boolean(ready["ready"], "readyz.ready"),
        "accepting": _boolean(ready["accepting"], "readyz.accepting"),
    }
    if ready_result != {"http_status": 200, "ready": True, "accepting": True}:
        _fail("raw.observations.readyz", "must record accepting HTTP 200 readiness")
    models = _closed(observations["models"], {"http_status", "model_ids"}, "raw.observations.models")
    models_result = {"http_status": _integer(models["http_status"], "models.http_status"), "model_ids": models["model_ids"]}
    if models_result != {"http_status": 200, "model_ids": [model_result["model_id"]]}:
        _fail("raw.observations.models", "must contain exactly the bound loaded model")

    greedy = _closed(observations["greedy"], {"non_stream_http_status", "stream_http_status", "non_stream_text_sha256", "stream_text_sha256", "approved_text_sha256", "completion_tokens", "stream_token_events", "finish_reason", "stream_done", "prompt_sha256", "max_tokens"}, "raw.observations.greedy")
    greedy_result = {
        "non_stream_http_status": _integer(greedy["non_stream_http_status"], "greedy.non_stream_http_status"),
        "stream_http_status": _integer(greedy["stream_http_status"], "greedy.stream_http_status"),
        "non_stream_text_sha256": _sha(greedy["non_stream_text_sha256"], "greedy.non_stream_text_sha256"),
        "stream_text_sha256": _sha(greedy["stream_text_sha256"], "greedy.stream_text_sha256"),
        "approved_text_sha256": _sha(greedy["approved_text_sha256"], "greedy.approved_text_sha256"),
        "completion_tokens": _integer(greedy["completion_tokens"], "greedy.completion_tokens", 2),
        "stream_token_events": _integer(greedy["stream_token_events"], "greedy.stream_token_events", 2),
        "finish_reason": greedy["finish_reason"], "stream_done": greedy["stream_done"],
        "prompt_sha256": _sha(greedy["prompt_sha256"], "greedy.prompt_sha256"),
        "max_tokens": _integer(greedy["max_tokens"], "greedy.max_tokens", 2),
    }
    if greedy_result["non_stream_http_status"] != 200 or greedy_result["stream_http_status"] != 200:
        _fail("raw.observations.greedy", "both requests must return HTTP 200")
    if len({greedy_result[key] for key in ("non_stream_text_sha256", "stream_text_sha256", "approved_text_sha256")}) != 1:
        _fail("raw.observations.greedy", "non-stream, SSE, and approved golden hashes must match")
    if greedy_result["completion_tokens"] != greedy_result["stream_token_events"] or greedy_result["finish_reason"] not in {"length", "stop"} or greedy_result["stream_done"] is not True:
        _fail("raw.observations.greedy", "invalid completion/SSE terminal claims")

    sampling = _closed(observations["sampling"], {"seed", "temperature", "top_p", "first_http_status", "second_http_status", "first_completion_tokens", "second_completion_tokens", "first_finish_reason", "second_finish_reason", "first_text_sha256", "second_text_sha256"}, "raw.observations.sampling")
    sampling_result = {
        "seed": _integer(sampling["seed"], "sampling.seed"),
        "temperature": _number(sampling["temperature"], "sampling.temperature", positive=True),
        "top_p": _number(sampling["top_p"], "sampling.top_p", positive=True),
        "first_http_status": _integer(sampling["first_http_status"], "sampling.first_http_status"),
        "second_http_status": _integer(sampling["second_http_status"], "sampling.second_http_status"),
        "first_completion_tokens": _integer(sampling["first_completion_tokens"], "sampling.first_completion_tokens", 1),
        "second_completion_tokens": _integer(sampling["second_completion_tokens"], "sampling.second_completion_tokens", 1),
        "first_finish_reason": sampling["first_finish_reason"], "second_finish_reason": sampling["second_finish_reason"],
        "first_text_sha256": _sha(sampling["first_text_sha256"], "sampling.first_text_sha256"),
        "second_text_sha256": _sha(sampling["second_text_sha256"], "sampling.second_text_sha256"),
    }
    if sampling_result["top_p"] > 1 or sampling_result["first_http_status"] != 200 or sampling_result["second_http_status"] != 200:
        _fail("raw.observations.sampling", "invalid sampling request or response parameters")
    if sampling_result["first_completion_tokens"] != sampling_result["second_completion_tokens"] or sampling_result["first_finish_reason"] != sampling_result["second_finish_reason"] or sampling_result["first_finish_reason"] not in {"length", "stop"} or sampling_result["first_text_sha256"] != sampling_result["second_text_sha256"] or sampling_result["first_text_sha256"] == hashlib.sha256(b"").hexdigest():
        _fail("raw.observations.sampling", "clean-start fixed-seed responses must be identical and nonempty")

    cancellation = _closed(observations["cancellation"], {"disconnect_probe_sent", "cancellations_before", "cancellations_after", "disconnects_before", "disconnects_after", "active_requests_after", "waiting_requests_after"}, "raw.observations.cancellation")
    cancellation_result = {
        "disconnect_probe_sent": _boolean(
            cancellation["disconnect_probe_sent"],
            "cancellation.disconnect_probe_sent",
        ),
        "cancellations_before": _integer(cancellation["cancellations_before"], "cancellation.cancellations_before"),
        "cancellations_after": _integer(cancellation["cancellations_after"], "cancellation.cancellations_after"),
        "disconnects_before": _integer(cancellation["disconnects_before"], "cancellation.disconnects_before"),
        "disconnects_after": _integer(cancellation["disconnects_after"], "cancellation.disconnects_after"),
        "active_requests_after": _integer(cancellation["active_requests_after"], "cancellation.active_requests_after"),
        "waiting_requests_after": _integer(cancellation["waiting_requests_after"], "cancellation.waiting_requests_after"),
    }
    if cancellation_result["disconnect_probe_sent"] is not True or cancellation_result["cancellations_after"] <= cancellation_result["cancellations_before"] or cancellation_result["disconnects_after"] <= cancellation_result["disconnects_before"] or cancellation_result["active_requests_after"] or cancellation_result["waiting_requests_after"]:
        _fail("raw.observations.cancellation", "must prove disconnect cancellation and reclamation")

    shutdown = _closed(observations["shutdown"], {"signal", "exit_code", "metrics", "metrics_sha256", "repeat_exit_code", "repeat_metrics", "repeat_metrics_sha256"}, "raw.observations.shutdown")
    if shutdown["signal"] != "SIGTERM" or _integer(shutdown["exit_code"], "shutdown.exit_code") != 0 or _integer(shutdown["repeat_exit_code"], "shutdown.repeat_exit_code") != 0:
        _fail("raw.observations.shutdown", "both clean starts must record SIGTERM exit zero")
    shutdown_result = {
        "signal": "SIGTERM", "exit_code": 0, "repeat_exit_code": 0,
        "metrics": _validate_metrics(shutdown["metrics"], "raw.observations.shutdown.metrics", final=True),
        "repeat_metrics": _validate_metrics(shutdown["repeat_metrics"], "raw.observations.shutdown.repeat_metrics", final=True),
        "metrics_sha256": _sha(shutdown["metrics_sha256"], "shutdown.metrics_sha256"),
        "repeat_metrics_sha256": _sha(shutdown["repeat_metrics_sha256"], "shutdown.repeat_metrics_sha256"),
    }

    python_free = _closed(observations["python_free"], {"forbidden_executables", "forbidden_artifact_count", "processes", "manifest_dependencies", "loader_dependencies", "unresolved_dependencies", "forbidden_dependency_matches"}, "raw.observations.python_free")
    for field in ("forbidden_executables", "unresolved_dependencies", "forbidden_dependency_matches"):
        if python_free[field] != []:
            _fail(f"raw.observations.python_free.{field}", "must be an empty array")
    if _integer(python_free["forbidden_artifact_count"], "python_free.forbidden_artifact_count") != 0:
        _fail("python_free.forbidden_artifact_count", "must be zero")
    dependencies = python_free["manifest_dependencies"]
    loaders = python_free["loader_dependencies"]
    if not isinstance(dependencies, list) or dependencies != sorted(set(dependencies)) or not REQUIRED_DEPENDENCIES <= set(dependencies) or set(dependencies) - ALLOWED_DEPENDENCIES:
        _fail("python_free.manifest_dependencies", "missing or unreviewed native dependency")
    if not isinstance(loaders, list) or not loaders or any(not isinstance(item, str) or FORBIDDEN_RE.search(item) for item in loaders):
        _fail("python_free.loader_dependencies", "contains invalid dependency text")
    python_result = {
        "forbidden_executables": [], "forbidden_artifact_count": 0,
        "processes": _validate_process_claims(python_free["processes"], "python_free.processes"),
        "manifest_dependencies": dependencies, "loader_dependencies": loaders,
        "unresolved_dependencies": [], "forbidden_dependency_matches": [],
    }
    return {
        "source": source_result, "release": release_result, "model": model_result,
        "runtime": runtime_result,
        "observations": {"readyz": ready_result, "models": models_result, "greedy": greedy_result, "sampling": sampling_result, "cancellation": cancellation_result, "shutdown": shutdown_result, "python_free": python_result},
    }


def _decode_chunked(body: bytes, label: str) -> bytes:
    result = bytearray()
    cursor = 0
    while True:
        line_end = body.find(b"\r\n", cursor)
        if line_end < 0:
            _fail(label, "truncated HTTP chunk header")
        size_text = body[cursor:line_end]
        if not re.fullmatch(rb"[0-9A-Fa-f]+", size_text):
            _fail(label, "invalid or extended HTTP chunk size")
        size = int(size_text, 16)
        cursor = line_end + 2
        if size == 0:
            if body[cursor:] != b"\r\n":
                _fail(label, "trailers and bytes after the terminal chunk are forbidden")
            return bytes(result)
        end = cursor + size
        if end + 2 > len(body) or body[end : end + 2] != b"\r\n":
            _fail(label, "truncated HTTP chunk")
        result.extend(body[cursor:end])
        cursor = end + 2


def _http_response(raw: bytes, label: str) -> tuple[int, dict[str, str], bytes]:
    head, separator, body = raw.partition(b"\r\n\r\n")
    if not separator or len(head) > 64 * 1024:
        _fail(label, "must contain a bounded CRLF HTTP header")
    try:
        lines = head.decode("iso-8859-1").split("\r\n")
    except UnicodeDecodeError as error:  # pragma: no cover - ISO-8859-1 is total
        _fail(label, f"cannot decode HTTP header: {error}")
    match = re.fullmatch(r"HTTP/1\.[01] ([0-9]{3})(?: [\x20-\x7e]*)?", lines[0])
    if match is None:
        _fail(label, "invalid HTTP status line")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or line[0].isspace() or ":" not in line:
            _fail(label, "invalid or folded HTTP header")
        name, value = line.split(":", 1)
        key = name.lower()
        if not re.fullmatch(r"[a-z0-9-]+", key) or key in headers:
            _fail(label, "duplicate or invalid HTTP header")
        headers[key] = value.strip()
    transfer = headers.get("transfer-encoding", "").lower()
    if transfer:
        if transfer != "chunked" or "content-length" in headers:
            _fail(label, "unsupported HTTP transfer encoding")
        body = _decode_chunked(body, label)
    elif "content-length" in headers:
        if not headers["content-length"].isdigit() or int(headers["content-length"]) != len(body):
            _fail(label, "Content-Length does not equal the preserved body")
    if len(body) > MAX_JSON_BYTES:
        _fail(label, "HTTP body exceeds the replay bound")
    return int(match.group(1)), headers, body


def _http_json(raw: bytes, label: str) -> tuple[int, dict[str, Any]]:
    status_code, headers, body = _http_response(raw, label)
    if "application/json" not in headers.get("content-type", "").lower():
        _fail(label, "must preserve an application/json Content-Type")
    return status_code, _parse_json_bytes(body, f"{label} body")


def _completion_response(raw: bytes, label: str) -> dict[str, Any]:
    status_code, document = _http_json(raw, label)
    choices = document.get("choices")
    usage = document.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict) or not isinstance(usage, dict):
        _fail(label, "must contain exactly one completion choice and usage")
    text = choices[0].get("text")
    finish = choices[0].get("finish_reason")
    if not isinstance(text, str) or not text or finish not in {"length", "stop"}:
        _fail(label, "completion text must be nonempty with a reviewed finish reason")
    tokens = _integer(usage.get("completion_tokens"), f"{label}.usage.completion_tokens", 1)
    return {"http_status": status_code, "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "completion_tokens": tokens, "finish_reason": finish}


def _sse_completion(raw: bytes, label: str) -> dict[str, Any]:
    status_code, headers, body = _http_response(raw, label)
    if "text/event-stream" not in headers.get("content-type", "").lower():
        _fail(label, "must preserve a text/event-stream Content-Type")
    try:
        text = body.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as error:
        _fail(label, f"SSE body is not UTF-8: {error}")
    if "\r" in text:
        _fail(label, "SSE body contains a bare carriage return")
    blocks = [block for block in text.split("\n\n") if block]
    if len(blocks) < 3 or blocks[-1] != "data: [DONE]" or blocks.count("data: [DONE]") != 1:
        _fail(label, "SSE must contain JSON events followed by one terminal [DONE]")
    output: list[str] = []
    token_events = 0
    terminal_reason: str | None = None
    for index, block in enumerate(blocks[:-1]):
        lines = block.splitlines()
        if len(lines) != 1 or not lines[0].startswith("data: "):
            _fail(label, "SSE events must be one preserved data line each")
        event = _parse_json_bytes(lines[0].removeprefix("data: ").encode("utf-8"), f"{label}.event[{index}]")
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            _fail(label, "each SSE event must contain exactly one choice")
        fragment = choices[0].get("text")
        reason = choices[0].get("finish_reason")
        if not isinstance(fragment, str) or reason not in {None, "length", "stop"}:
            _fail(label, "invalid SSE text or finish reason")
        output.append(fragment)
        if reason is None:
            token_events += 1
        else:
            if terminal_reason is not None or index != len(blocks) - 2:
                _fail(label, "finish reason must appear once in the last JSON event")
            terminal_reason = reason
    if terminal_reason is None or token_events < 2:
        _fail(label, "SSE must contain token events and a terminal finish event")
    completion = "".join(output)
    return {"http_status": status_code, "text_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(), "stream_token_events": token_events, "finish_reason": terminal_reason, "stream_done": True}


def _request_json(payloads: Mapping[str, bytes], name: str, keys: set[str]) -> dict[str, Any]:
    return dict(_closed(_parse_json_bytes(payloads[name], name), keys, name))


def _validated_completion_request(
    document: Mapping[str, Any], label: str, *, with_seed: bool
) -> dict[str, Any]:
    result = {
        "model": _string(document["model"], f"{label}.model", MODEL_ID_RE),
        "prompt": _string(document["prompt"], f"{label}.prompt"),
        "max_tokens": _integer(document["max_tokens"], f"{label}.max_tokens", 1),
        "temperature": _number(document["temperature"], f"{label}.temperature"),
        "top_p": _number(document["top_p"], f"{label}.top_p", positive=True),
        "stream": _boolean(document["stream"], f"{label}.stream"),
    }
    if result["top_p"] > 1:
        _fail(f"{label}.top_p", "must be no larger than 1")
    if with_seed:
        result["seed"] = _integer(document["seed"], f"{label}.seed")
    return result


def _http_request(raw: bytes, label: str) -> tuple[str, str, dict[str, str], bytes]:
    head, separator, body = raw.partition(b"\r\n\r\n")
    if not separator:
        _fail(label, "missing HTTP request header terminator")
    try:
        lines = head.decode("ascii").split("\r\n")
    except UnicodeDecodeError as error:
        _fail(label, f"request header is not ASCII: {error}")
    match = re.fullmatch(r"(GET|POST) ([A-Za-z0-9_./-]+) HTTP/1\.1", lines[0])
    if match is None:
        _fail(label, "invalid request line")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            _fail(label, "invalid request header")
        name, value = line.split(":", 1)
        key = name.lower()
        if key in headers:
            _fail(label, "duplicate request header")
        headers[key] = value.strip()
    if headers.get("host") != "localhost" or headers.get("connection", "").lower() != "close":
        _fail(label, "request must bind localhost and close the connection")
    if headers.get("content-length") != str(len(body)):
        _fail(label, "request Content-Length does not bind the preserved body")
    return match.group(1), match.group(2), headers, body


def _docker_record(raw: bytes, label: str) -> dict[str, Any]:
    value = _parse_json_value(raw, label)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        _fail(label, "must be one raw Docker inspect record")
    return value[0]


def _validate_image_inspect(raw: bytes, image_id: str) -> None:
    row = _docker_record(raw, "image-inspect.json")
    if row.get("Id") != image_id or row.get("Architecture") != "amd64" or row.get("Os") != "linux":
        _fail("image-inspect.json", "immutable ID or linux/amd64 platform mismatch")
    config = row.get("Config")
    rootfs = row.get("RootFS")
    if not isinstance(config, dict) or config.get("Entrypoint") != ["/opt/rustinfer/bin/rustinfer"] or config.get("Cmd") != ["--help"] or config.get("User") != "65532:65532":
        _fail("image-inspect.json.Config", "release entrypoint, command, or non-root user mismatch")
    if not isinstance(rootfs, dict) or rootfs.get("Type") != "layers" or not isinstance(rootfs.get("Layers"), list) or not rootfs["Layers"]:
        _fail("image-inspect.json.RootFS", "must preserve nonempty immutable layers")
    for index, layer in enumerate(rootfs["Layers"]):
        if not isinstance(layer, str) or not layer.startswith("sha256:"):
            _fail(f"image-inspect.json.RootFS.Layers[{index}]", "must be sha256:<digest>")
        _sha(layer.removeprefix("sha256:"), f"image-inspect.json.RootFS.Layers[{index}]")
    env = config.get("Env")
    if not isinstance(env, list) or not any(isinstance(item, str) and item.startswith("PATH=/opt/rustinfer/bin:") for item in env):
        _fail("image-inspect.json.Config.Env", "release PATH binding is missing")


def _container_snapshot(raw: bytes, label: str, *, container_id: str, image_id: str, model_id: str, running: bool) -> dict[str, Any]:
    row = _docker_record(raw, label)
    expected_args = [
        "serve", "--model", "/models/checkpoint", "--model-id", model_id,
        "--bind", "127.0.0.1:8080", "--max-output-tokens", "1024",
        "--execution-completion", "iteration-batch", "--residual-rmsnorm", "separate",
    ]
    if row.get("Id") != container_id or row.get("Image") != image_id or row.get("Path") != "/opt/rustinfer/bin/rustinfer" or row.get("Args") != expected_args:
        _fail(label, "container identity/image/executable arguments mismatch")
    config = row.get("Config")
    host = row.get("HostConfig")
    state = row.get("State")
    mounts = row.get("Mounts")
    if not isinstance(config, dict) or config.get("Image") != image_id:
        _fail(label, "container Config.Image is not immutable")
    if not isinstance(host, dict) or host.get("NetworkMode") != "none" or host.get("ReadonlyRootfs") is not True:
        _fail(label, "network-none/read-only runtime contract mismatch")
    requests = host.get("DeviceRequests")
    if not isinstance(requests, list) or not any(
        isinstance(item, dict) and item.get("Driver") in {"", "nvidia"} and any(
            isinstance(group, list) and "gpu" in group for group in item.get("Capabilities", [])
        ) for item in requests
    ):
        _fail(label, "NVIDIA GPU device request is missing")
    if not isinstance(mounts, list):
        _fail(label, "mount inventory is missing")
    destinations = {item.get("Destination"): item.get("RW") for item in mounts if isinstance(item, dict)}
    if destinations.get("/models/checkpoint") is not False or destinations.get("/evidence") is not True:
        _fail(label, "model/evidence mount access contract mismatch")
    if not isinstance(state, dict) or state.get("Running") is not running:
        _fail(label, "container running state mismatch")
    pid = state.get("Pid")
    if running:
        _integer(pid, f"{label}.State.Pid", 1)
    elif (
        _integer(pid, f"{label}.State.Pid") != 0
        or _integer(state.get("ExitCode"), f"{label}.State.ExitCode") != 0
        or state.get("OOMKilled") is not False
        or state.get("Error") not in {"", None}
    ):
        _fail(label, "post-SIGTERM container must be stopped cleanly with exit zero")
    return row


def _process_snapshot(raw: bytes, label: str, expected_pid: int) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        _fail(label, f"docker top output is not UTF-8: {error}")
    if len(lines) < 2 or not re.match(r"^\s*PID\s+PPID\s+", lines[0], re.I):
        _fail(label, "must preserve the docker top header and processes")
    result: list[dict[str, Any]] = []
    for index, line in enumerate(lines[1:], 1):
        parts = line.strip().split(None, 3)
        if len(parts) != 4 or not parts[0].isdigit() or not parts[1].isdigit():
            _fail(label, f"invalid docker top row {index}")
        row = {"pid": int(parts[0]), "ppid": int(parts[1]), "comm": parts[2], "args": parts[3]}
        if FORBIDDEN_RE.search(row["comm"] + " " + row["args"]):
            _fail(label, "contains a forbidden Python-family process")
        result.append(row)
    if not any(row["pid"] == expected_pid and row["comm"] == "rustinfer" for row in result):
        _fail(label, "does not bind the inspected rustinfer server PID")
    return result


def _replay_runtime(payloads: Mapping[str, bytes], validated: Mapping[str, Any], golden: Mapping[str, Any], image_id: str, binary_sha256: str) -> None:
    runtime = validated["runtime"]
    observations = validated["observations"]
    if hashlib.sha256(payloads["image-binary"]).hexdigest() != binary_sha256:
        _fail("image-binary", "actual image filesystem executable differs from the release binary")
    try:
        binary_dependencies = release_common.validate_binary(payloads["image-binary"])
        manifest_dependencies = release_common.parse_native_manifest(payloads["image-native-dependencies.txt"])
    except release_common.ReleaseContractError as error:
        _fail("image-binary", f"ELF/native manifest validation failed: {error}")
    if manifest_dependencies != binary_dependencies:
        _fail("image-native-dependencies.txt", "does not equal actual ELF DT_NEEDED entries")

    try:
        readelf = payloads["image-readelf.txt"].decode("utf-8")
        ldd = payloads["image-ldd.txt"].decode("utf-8")
    except UnicodeDecodeError as error:
        _fail("image static evidence", f"must be UTF-8: {error}")
    if "Class:                             ELF64" not in readelf or "Machine:                           Advanced Micro Devices X86-64" not in readelf or "Type:                              DYN" not in readelf:
        _fail("image-readelf.txt", "does not describe an x86_64 ELF64 PIE executable")
    readelf_needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", readelf)
    if len(readelf_needed) != len(set(readelf_needed)) or sorted(readelf_needed) != binary_dependencies:
        _fail("image-readelf.txt", "NEEDED entries differ from the actual ELF parser")
    if "not found" in ldd or FORBIDDEN_RE.search(ldd + "\n" + readelf):
        _fail("image-ldd.txt", "contains unresolved or forbidden runtime dependencies")
    for dependency in binary_dependencies:
        if re.search(rf"(?m)^\s*{re.escape(dependency)}(?:\s|$)", ldd) is None:
            _fail("image-ldd.txt", f"does not resolve {dependency}")
    loader_lines = [line for line in ldd.splitlines() if line]
    if observations["python_free"]["manifest_dependencies"] != manifest_dependencies or observations["python_free"]["loader_dependencies"] != loader_lines:
        _fail("raw.observations.python_free", "dependency summary differs from preserved ELF/manifest/ldd evidence")
    if payloads["image-python-scan.txt"] != b"[forbidden-executables]\n[forbidden-artifacts]\n":
        _fail("image-python-scan.txt", "contains a forbidden runtime executable or artifact")
    _validate_image_inspect(payloads["image-inspect.json"], image_id)

    ids = runtime["container_ids"]
    snapshots: dict[tuple[str, str], dict[str, Any]] = {}
    for ordinal, container_id in zip(("first", "second"), ids):
        pre = _container_snapshot(payloads[f"container-{ordinal}-pre.json"], f"container-{ordinal}-pre.json", container_id=container_id, image_id=image_id, model_id=golden["model_id"], running=True)
        active = _container_snapshot(payloads[f"container-{ordinal}-runtime.json"], f"container-{ordinal}-runtime.json", container_id=container_id, image_id=image_id, model_id=golden["model_id"], running=True)
        post = _container_snapshot(payloads[f"container-{ordinal}-post.json"], f"container-{ordinal}-post.json", container_id=container_id, image_id=image_id, model_id=golden["model_id"], running=False)
        for field in ("Created", "Image", "Path", "Args"):
            if pre.get(field) != active.get(field) or active.get(field) != post.get(field):
                _fail(f"container-{ordinal}", f"{field} changed across pre/runtime/post snapshots")
        pre_state, active_state, post_state = pre["State"], active["State"], post["State"]
        if pre_state.get("Pid") != active_state.get("Pid") or pre_state.get("StartedAt") != active_state.get("StartedAt") or active_state.get("StartedAt") != post_state.get("StartedAt"):
            _fail(f"container-{ordinal}", "PID/start identity changed across runtime snapshots")
        finished = post_state.get("FinishedAt")
        if not isinstance(finished, str) or not finished or finished.startswith("0001-"):
            _fail(f"container-{ordinal}-post.json", "missing post-SIGTERM finish timestamp")
        pre_processes = _process_snapshot(payloads[f"process-{ordinal}-pre.txt"], f"process-{ordinal}-pre.txt", pre_state["Pid"])
        runtime_processes = _process_snapshot(payloads[f"process-{ordinal}-runtime.txt"], f"process-{ordinal}-runtime.txt", active_state["Pid"])
        if not any(row["comm"] == "rustinfer" for row in pre_processes) or not any(row["comm"] == "rustinfer" for row in runtime_processes):
            _fail(f"process-{ordinal}", "rustinfer disappeared during the observed runtime")
        snapshots[(ordinal, "runtime_processes")] = {"rows": runtime_processes}
    if observations["python_free"]["processes"] != snapshots[("first", "runtime_processes")]["rows"]:
        _fail("raw.observations.python_free.processes", "differs from process-first-runtime.txt")

    greedy_request = _validated_completion_request(
        _request_json(payloads, "request-greedy.json", {"model", "prompt", "max_tokens", "temperature", "top_p", "stream"}),
        "request-greedy.json",
        with_seed=False,
    )
    greedy_stream_request = _validated_completion_request(
        _request_json(payloads, "request-greedy-stream.json", {"model", "prompt", "max_tokens", "temperature", "top_p", "stream"}),
        "request-greedy-stream.json",
        with_seed=False,
    )
    sampling_request = _validated_completion_request(
        _request_json(payloads, "request-sampling.json", {"model", "prompt", "max_tokens", "temperature", "top_p", "seed", "stream"}),
        "request-sampling.json",
        with_seed=True,
    )
    expected_greedy_request = {"model": golden["model_id"], "prompt": golden["prompt"], "max_tokens": golden["max_tokens"], "temperature": 0.0, "top_p": 1.0, "stream": False}
    if greedy_request != expected_greedy_request or greedy_stream_request != {**expected_greedy_request, "stream": True}:
        _fail("request-greedy.json", "greedy request bytes differ from the reviewed golden probe")
    expected_sampling_request = {"model": golden["model_id"], "prompt": golden["prompt"], "max_tokens": 16, "temperature": 0.8, "top_p": 0.95, "seed": 424242, "stream": False}
    if sampling_request != expected_sampling_request:
        _fail("request-sampling.json", "fixed-seed sampling request bytes differ from the reviewed probe")

    ready_status, ready = _http_json(payloads["http-readyz.raw"], "http-readyz.raw")
    derived_ready = {
        "http_status": ready_status,
        "ready": _boolean(ready.get("ready"), "http-readyz.raw.ready"),
        "accepting": _boolean(
            ready.get("accepting"), "http-readyz.raw.accepting"
        ),
    }
    if derived_ready != observations["readyz"]:
        _fail("raw.observations.readyz", "differs from http-readyz.raw")
    models_status, models = _http_json(payloads["http-models.raw"], "http-models.raw")
    data = models.get("data")
    if not isinstance(data, list) or any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in data):
        _fail("http-models.raw", "invalid OpenAI models response")
    derived_models = {"http_status": models_status, "model_ids": [item["id"] for item in data]}
    if derived_models != observations["models"]:
        _fail("raw.observations.models", "differs from http-models.raw")

    greedy = _completion_response(payloads["http-greedy.raw"], "http-greedy.raw")
    stream = _sse_completion(payloads["http-greedy-stream.raw"], "http-greedy-stream.raw")
    derived_greedy = {
        "non_stream_http_status": greedy["http_status"], "stream_http_status": stream["http_status"],
        "non_stream_text_sha256": greedy["text_sha256"], "stream_text_sha256": stream["text_sha256"],
        "approved_text_sha256": golden["expected_greedy_text_sha256"],
        "completion_tokens": greedy["completion_tokens"], "stream_token_events": stream["stream_token_events"],
        "finish_reason": greedy["finish_reason"], "stream_done": stream["stream_done"],
        "prompt_sha256": hashlib.sha256(golden["prompt"].encode("utf-8")).hexdigest(), "max_tokens": golden["max_tokens"],
    }
    if stream["finish_reason"] != greedy["finish_reason"] or derived_greedy != observations["greedy"]:
        _fail("raw.observations.greedy", "differs from preserved request/JSON/SSE transcripts")

    sample_first = _completion_response(payloads["http-sampling-first.raw"], "http-sampling-first.raw")
    sample_second = _completion_response(payloads["http-sampling-second.raw"], "http-sampling-second.raw")
    derived_sampling = {
        "seed": sampling_request["seed"], "temperature": float(sampling_request["temperature"]), "top_p": float(sampling_request["top_p"]),
        "first_http_status": sample_first["http_status"], "second_http_status": sample_second["http_status"],
        "first_completion_tokens": sample_first["completion_tokens"], "second_completion_tokens": sample_second["completion_tokens"],
        "first_finish_reason": sample_first["finish_reason"], "second_finish_reason": sample_second["finish_reason"],
        "first_text_sha256": sample_first["text_sha256"], "second_text_sha256": sample_second["text_sha256"],
    }
    if derived_sampling != observations["sampling"]:
        _fail("raw.observations.sampling", "differs from clean-start sampling transcripts")

    before_status, before_document = _http_json(payloads["http-metrics-before.raw"], "http-metrics-before.raw")
    after_status, after_document = _http_json(payloads["http-metrics-after.raw"], "http-metrics-after.raw")
    if before_status != 200 or after_status != 200:
        _fail("HTTP metrics", "both cancellation metric probes must return 200")
    before = _validate_metrics(before_document, "http-metrics-before.raw body", final=False)
    after = _validate_metrics(after_document, "http-metrics-after.raw body", final=False)
    method, target, headers, cancel_body = _http_request(payloads["cancellation-request.raw"], "cancellation-request.raw")
    if method != "POST" or target != "/v1/completions" or headers.get("content-type") != "application/json":
        _fail("cancellation-request.raw", "must preserve a JSON completion POST")
    cancel = _validated_completion_request(
        _closed(_parse_json_bytes(cancel_body, "cancellation request body"), {"model", "prompt", "max_tokens", "temperature", "top_p", "stream"}, "cancellation request body"),
        "cancellation request body",
        with_seed=False,
    )
    if cancel["model"] != golden["model_id"] or cancel["prompt"] != golden["prompt"] or cancel["temperature"] != 0.0 or cancel["top_p"] != 1.0 or cancel["stream"] is not True or cancel["max_tokens"] < 32 or cancel["max_tokens"] > 1024:
        _fail("cancellation-request.raw", "does not match the bounded disconnect probe")
    if payloads["cancellation-response-prefix.raw"] != b"HTTP/1.1 200":
        _fail("cancellation-response-prefix.raw", "does not prove an admitted HTTP 200 stream before disconnect")
    derived_cancellation = {
        "disconnect_probe_sent": True,
        "cancellations_before": before["counters"]["cancellations"], "cancellations_after": after["counters"]["cancellations"],
        "disconnects_before": before["counters"]["disconnects"], "disconnects_after": after["counters"]["disconnects"],
        "active_requests_after": after["active_requests"], "waiting_requests_after": after["waiting_requests"],
    }
    if derived_cancellation != observations["cancellation"]:
        _fail("raw.observations.cancellation", "differs from cancellation request/response/metric transcripts")


def _error_report() -> dict[str, Any]:
    return {"schema_version": REPORT_SCHEMA, "gate": GATE, "status": "error", "source": None, "raw_evidence_sha256": None, "checks": [{"id": "input_contract", "passed": False}]}


def _closed_archive_document(archive: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = archive.get(key)
    if not isinstance(value, dict):
        _fail("raw archive", f"parsed {key} document is missing")
    return value


def validate_bound_raw_archive(
    archive: Mapping[str, Any], *, source_revision: str, source_archive_sha256: str,
    release_binary_sha256: str, release_bundle_sha256: str, image_id: str,
    correctness_report: Mapping[str, Any], correctness_report_sha256: str,
    correctness_golden_sha256: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Semantically replay v2 raw evidence against immutable candidate bindings."""

    report = _error_report()
    try:
        _string(source_revision, "source revision", GIT_RE)
        source_sha = _sha(source_archive_sha256, "source archive SHA-256")
        binary_sha = _sha(release_binary_sha256, "release binary SHA-256")
        bundle_sha = _sha(release_bundle_sha256, "release bundle SHA-256")
        if not image_id.startswith("sha256:"):
            _fail("image ID", "must be sha256:<digest>")
        image_sha = _sha(image_id.removeprefix("sha256:"), "image ID")
        correctness_sha = _sha(correctness_report_sha256, "correctness report SHA-256")
        if correctness_golden_sha256 is None:
            _fail("correctness golden SHA-256", "an independently reviewed binding is required")
        reviewed_golden_sha = _sha(correctness_golden_sha256, "correctness golden SHA-256")
        archive_digest = _sha(archive.get("archive_sha256"), "raw archive SHA-256")
        payloads = archive.get("payloads")
        if not isinstance(payloads, dict) or not RAW_ARCHIVE_MEMBERS <= set(payloads):
            _fail("raw archive", "parsed fixed v2 payload inventory mismatch")
        validated = _validate_raw(_closed_archive_document(archive, "raw"))
        golden = _validate_golden(_closed_archive_document(archive, "golden"))
        correctness = _validate_correctness_report(correctness_report)
        model_manifest = archive.get("model_manifest")
        model_files = archive.get("model_files")
        if not isinstance(model_manifest, bytes) or not isinstance(model_files, dict):
            _fail("raw archive", "model subjects are missing")

        expected_source = {"git_revision": source_revision, "git_dirty": False, "source_archive_sha256": source_sha}
        expected_release = {"binary_sha256": binary_sha, "bundle_sha256": bundle_sha, "image_sha256": image_sha}
        if validated["source"] != expected_source:
            _fail("raw.source", "does not match the candidate source")
        if validated["release"] != expected_release:
            _fail("raw.release", "does not match the candidate release")
        if validated["runtime"]["image_id"] != image_id or validated["runtime"]["image_binary_sha256"] != binary_sha:
            _fail("raw.runtime", "does not match the immutable release image/binary")

        model = validated["model"]
        if model["model_tree_sha256"] != hashlib.sha256(model_manifest).hexdigest():
            _fail("raw.model.model_tree_sha256", "does not hash model-SHA256SUMS")
        for name, field in (("config.json", "config_sha256"), ("model.safetensors", "weights_sha256"), ("tokenizer.json", "tokenizer_json_sha256")):
            if model_files.get(name) != model[field]:
                _fail("raw archive.model-SHA256SUMS", f"{name} binding mismatch")
        if model["tokenizer_aggregate_sha256"] != _tokenizer_aggregate_sha256(model_files):
            _fail("raw.model.tokenizer_aggregate_sha256", "does not equal the archived tokenizer subjects")
        golden_sha = hashlib.sha256(payloads["correctness-golden.json"]).hexdigest()
        if golden_sha != reviewed_golden_sha:
            _fail("correctness golden", "differs from the independently reviewed SHA-256")
        if model["correctness_golden_sha256"] != golden_sha or model["correctness_report_sha256"] != correctness_sha:
            _fail("raw.model", "correctness golden/report binding mismatch")
        expected_provenance = {
            "correctness_gate_id": CORRECTNESS_GATE, "source_revision": source_revision,
            "model_id": model["model_id"], "model_revision": model["model_revision"],
            "config_sha256": model["config_sha256"], "weights_sha256": model["weights_sha256"],
            "tokenizer_aggregate_sha256": model["tokenizer_aggregate_sha256"],
        }
        if {key: correctness[key] for key in expected_provenance} != expected_provenance:
            _fail("correctness report.bindings", "does not match candidate/model provenance")
        golden_expected = {**expected_provenance, "correctness_report_sha256": correctness_sha}
        if {key: golden[key] for key in golden_expected} != golden_expected or golden["tokenizer_json_sha256"] != model["tokenizer_json_sha256"]:
            _fail("correctness golden", "does not match candidate/model provenance")
        if model["correctness_gate_id"] != CORRECTNESS_GATE:
            _fail("raw.model.correctness_gate_id", f"must be {CORRECTNESS_GATE}")

        shutdown = _closed_archive_document(archive, "shutdown")
        repeat_shutdown = _closed_archive_document(archive, "repeat_shutdown")
        claimed_shutdown = validated["observations"]["shutdown"]
        if shutdown != claimed_shutdown["metrics"] or repeat_shutdown != claimed_shutdown["repeat_metrics"]:
            _fail("raw.observations.shutdown", "differs from archived shutdown metrics")
        if hashlib.sha256(payloads["shutdown-metrics.json"]).hexdigest() != claimed_shutdown["metrics_sha256"] or hashlib.sha256(payloads["repeat-shutdown-metrics.json"]).hexdigest() != claimed_shutdown["repeat_metrics_sha256"]:
            _fail("raw.observations.shutdown", "metrics digest mismatch")

        _replay_runtime(payloads, validated, golden, image_id, binary_sha)
        report = {
            "schema_version": REPORT_SCHEMA, "gate": GATE, "status": "passed",
            "source": {"git_revision": source_revision, "git_dirty": False, "source_archive_sha256": source_sha, "release_binary_sha256": binary_sha, "release_bundle_sha256": bundle_sha, "release_image_sha256": image_sha},
            "raw_evidence_sha256": archive_digest,
            "checks": [{"id": check_id, "passed": True} for check_id in CHECK_IDS],
        }
        return report, None
    except (EvidenceError, OSError) as error:
        return report, str(error)


def evaluate(
    evidence: Path, *, raw_archive: Path, source_revision: str, source_archive: Path,
    release_binary: Path, release_bundle: Path, image_id: str, model_dir: Path,
    expected_model_tree_sha256: str, weights: Path, expected_weights_sha256: str,
    tokenizer: Path, expected_tokenizer_json_sha256: str,
    expected_tokenizer_aggregate_sha256: str, correctness_golden: Path,
    expected_correctness_golden_sha256: str, correctness_report: Path,
    expected_correctness_report_sha256: str, shutdown_metrics: Path,
    repeat_shutdown_metrics: Path,
) -> tuple[dict[str, Any], str | None]:
    """Verify external producer artifacts and replay the closed v2 archive."""

    report = _error_report()
    try:
        archive = load_raw_evidence_archive(raw_archive)
        payloads = archive["payloads"]
        external_payloads = {
            "raw-evidence.json": evidence, "correctness-golden.json": correctness_golden,
            "shutdown-metrics.json": shutdown_metrics,
            "repeat-shutdown-metrics.json": repeat_shutdown_metrics,
        }
        for name, path in external_payloads.items():
            try:
                actual = path.read_bytes()
            except OSError as error:
                _fail(name, f"cannot read producer output: {error}")
            if actual != payloads[name]:
                _fail(name, "external producer output differs from raw archive")

        actual_model_manifest = model_tree_manifest_bytes(model_dir)
        if actual_model_manifest != payloads["model-SHA256SUMS"]:
            _fail("model-SHA256SUMS", "does not equal the actual model directory")
        expected_model = _sha(expected_model_tree_sha256, "--model-tree-sha256")
        if hashlib.sha256(actual_model_manifest).hexdigest() != expected_model:
            _fail("--model-tree-sha256", "actual model tree binding mismatch")
        expected_weights = _sha(expected_weights_sha256, "--weights-sha256")
        if _file_sha256(weights, "--weights") != expected_weights:
            _fail("--weights-sha256", "actual weights binding mismatch")
        expected_tokenizer_json = _sha(expected_tokenizer_json_sha256, "--tokenizer-json-sha256")
        if _file_sha256(tokenizer, "--tokenizer") != expected_tokenizer_json:
            _fail("--tokenizer-json-sha256", "actual tokenizer.json binding mismatch")
        expected_tokenizer_aggregate = _sha(expected_tokenizer_aggregate_sha256, "--tokenizer-aggregate-sha256")
        if _tokenizer_aggregate_sha256(archive["model_files"]) != expected_tokenizer_aggregate:
            _fail("--tokenizer-aggregate-sha256", "archived tokenizer subjects mismatch")
        expected_golden = _sha(expected_correctness_golden_sha256, "--correctness-golden-sha256")
        if hashlib.sha256(payloads["correctness-golden.json"]).hexdigest() != expected_golden:
            _fail("--correctness-golden-sha256", "golden binding mismatch")
        correctness_document, correctness_bytes = _read_json(correctness_report, "correctness report")
        expected_correctness = _sha(expected_correctness_report_sha256, "--correctness-report-sha256")
        if hashlib.sha256(correctness_bytes).hexdigest() != expected_correctness:
            _fail("--correctness-report-sha256", "native report binding mismatch")
        source_sha = _file_sha256(source_archive, "--source-archive")
        binary_sha = _file_sha256(release_binary, "--release-binary")
        bundle_sha = _file_sha256(release_bundle, "--release-bundle")
        _verify_bundle(release_bundle, binary_sha, source_revision)
        return validate_bound_raw_archive(
            archive, source_revision=source_revision, source_archive_sha256=source_sha,
            release_binary_sha256=binary_sha, release_bundle_sha256=bundle_sha,
            image_id=image_id, correctness_report=correctness_document,
            correctness_report_sha256=expected_correctness,
            correctness_golden_sha256=expected_golden,
        )
    except (EvidenceError, OSError) as error:
        return report, str(error)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--raw-archive", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--release-binary", required=True, type=Path)
    parser.add_argument("--release-bundle", required=True, type=Path)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-tree-sha256", required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--tokenizer-json-sha256", required=True)
    parser.add_argument("--tokenizer-aggregate-sha256", required=True)
    parser.add_argument("--correctness-golden", required=True, type=Path)
    parser.add_argument("--correctness-golden-sha256", required=True)
    parser.add_argument("--correctness-report", required=True, type=Path)
    parser.add_argument("--correctness-report-sha256", required=True)
    parser.add_argument("--shutdown-metrics", required=True, type=Path)
    parser.add_argument("--repeat-shutdown-metrics", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="create without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report, diagnostic = evaluate(
        args.evidence, raw_archive=args.raw_archive, source_revision=args.source_revision,
        source_archive=args.source_archive, release_binary=args.release_binary,
        release_bundle=args.release_bundle, image_id=args.image_id, model_dir=args.model_dir,
        expected_model_tree_sha256=args.model_tree_sha256, weights=args.weights,
        expected_weights_sha256=args.weights_sha256, tokenizer=args.tokenizer,
        expected_tokenizer_json_sha256=args.tokenizer_json_sha256,
        expected_tokenizer_aggregate_sha256=args.tokenizer_aggregate_sha256,
        correctness_golden=args.correctness_golden,
        expected_correctness_golden_sha256=args.correctness_golden_sha256,
        correctness_report=args.correctness_report,
        expected_correctness_report_sha256=args.correctness_report_sha256,
        shutdown_metrics=args.shutdown_metrics,
        repeat_shutdown_metrics=args.repeat_shutdown_metrics,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            with args.report.open("x", encoding="utf-8", newline="") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            print(f"refusing to overwrite existing report: {args.report}", file=sys.stderr)
            return 2
        except OSError as error:
            print(f"cannot create report {args.report}: {error}", file=sys.stderr)
            return 2
    sys.stdout.write(encoded)
    if diagnostic is not None:
        print(f"python-free release E2E evidence failed: {diagnostic}", file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
