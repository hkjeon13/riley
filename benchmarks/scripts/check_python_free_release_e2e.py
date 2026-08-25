#!/usr/bin/env python3
"""Verify Python-free real-model release E2E evidence without running CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence


RAW_SCHEMA = "rustinfer.python-free-release-e2e-raw.v1"
GOLDEN_SCHEMA = "rustinfer.python-free-release-e2e-golden.v1"
REPORT_SCHEMA = "rustinfer.release-gate-attestation.v1"
GATE = "python-free-clean-runtime-e2e"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
CONTAINER_RE = re.compile(r"^[0-9a-f]{12,64}$")
MODEL_PATH_RE = re.compile(r"^[A-Za-z0-9._/+@=-]+$")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_RAW_ARCHIVE_BYTES = 64 * 1024 * 1024
RAW_ARCHIVE_PAYLOADS = (
    "correctness-golden.json",
    "model-SHA256SUMS",
    "raw-evidence.json",
    "repeat-shutdown-metrics.json",
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
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
}
FORBIDDEN_RE = re.compile(r"python|pip|pytorch|torch|transformers|triton|pickle", re.I)
CHECK_IDS = (
    "release_bundle_verified",
    "no_python_executable",
    "no_python_child",
    "no_forbidden_runtime_artifact",
    "native_dependencies_verified",
    "model_load",
    "prefill",
    "decode",
    "greedy_golden",
    "sampling",
    "streaming",
    "cancellation",
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
_RELEASE_COMMON_SPEC = importlib.util.spec_from_file_location(
    "python_free_e2e_release_common", _RELEASE_DIR / "release_common.py"
)
if _RELEASE_COMMON_SPEC is None or _RELEASE_COMMON_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot load release bundle contract")
release_common = importlib.util.module_from_spec(_RELEASE_COMMON_SPEC)
sys.modules["release_common"] = release_common
_RELEASE_COMMON_SPEC.loader.exec_module(release_common)
_RELEASE_VERIFY_SPEC = importlib.util.spec_from_file_location(
    "python_free_e2e_release_verify", _RELEASE_DIR / "verify_release_bundle.py"
)
if _RELEASE_VERIFY_SPEC is None or _RELEASE_VERIFY_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot load release bundle verifier")
release_verify = importlib.util.module_from_spec(_RELEASE_VERIFY_SPEC)
_RELEASE_VERIFY_SPEC.loader.exec_module(release_verify)


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


def _parse_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    if len(raw) > MAX_JSON_BYTES:
        _fail(label, f"exceeds {MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"cannot read strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        _fail(label, "root must be an object")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        if metadata.st_size > MAX_JSON_BYTES:
            _fail(label, f"exceeds {MAX_JSON_BYTES} bytes")
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
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        _fail(path, "must be finite and positive" if positive else "must be finite and nonnegative")
    return number


def _true(value: Any, path: str) -> None:
    if value is not True:
        _fail(path, "must be true")


def _file_sha256(path: Path, label: str) -> str:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        _fail(label, f"cannot hash {path}: {error}")


def model_tree_manifest_bytes(model_dir: Path) -> bytes:
    """Return the canonical model-tree checksum manifest bytes."""
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
        if "\n" in relative or "\r" in relative:
            _fail("--model-dir", "model paths must not contain line breaks")
        if MODEL_PATH_RE.fullmatch(relative) is None:
            _fail("--model-dir", "model paths must use the safe ASCII path alphabet")
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _fail("--model-dir", f"contains non-regular entry {relative!r}")
        lines.append(f"{_file_sha256(path, 'model file')}  {relative}\n".encode("utf-8"))
    if not lines:
        _fail("--model-dir", "contains no regular files")
    return b"".join(lines)


def model_tree_sha256(model_dir: Path) -> str:
    """Hash sorted ``sha256 + two spaces + relative POSIX path`` manifest lines."""

    return hashlib.sha256(model_tree_manifest_bytes(model_dir)).hexdigest()


def _parse_model_manifest(raw: bytes) -> dict[str, str]:
    if not raw or len(raw) > MAX_JSON_BYTES or not raw.endswith(b"\n"):
        _fail("raw archive.model-SHA256SUMS", "must be a bounded newline-terminated manifest")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail("raw archive.model-SHA256SUMS", f"must be UTF-8: {error}")
    result: dict[str, str] = {}
    previous: str | None = None
    for index, line in enumerate(text.splitlines(), 1):
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
    tokenizer = {
        name: model_files[name] for name in TOKENIZER_ARTIFACT_FILENAMES
    }
    canonical = json.dumps(
        tokenizer,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_raw_evidence_archive(path: Path) -> dict[str, Any]:
    """Load and checksum the exact non-circular E2E raw evidence bundle."""

    label = "raw evidence archive"
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        if metadata.st_size <= 0 or metadata.st_size > MAX_RAW_ARCHIVE_BYTES:
            _fail(label, f"must be between 1 and {MAX_RAW_ARCHIVE_BYTES} bytes")
        archive_bytes = path.read_bytes()
        with tarfile.open(path, "r:") as archive:
            members = archive.getmembers()
            if len(members) != len(RAW_ARCHIVE_MEMBERS):
                _fail(label, "exact six-file inventory is required")
            by_name: dict[str, tarfile.TarInfo] = {}
            for member in members:
                if member.name in by_name:
                    _fail(label, f"duplicate member {member.name!r}")
                if member.name not in RAW_ARCHIVE_MEMBERS or "/" in member.name:
                    _fail(label, f"unexpected member {member.name!r}")
                if not member.isreg():
                    _fail(label, f"member must be a regular file: {member.name}")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mode != 0o644
                    or member.mtime != 0
                ):
                    _fail(label, f"non-canonical metadata for {member.name}")
                if member.size <= 0 or member.size > MAX_JSON_BYTES:
                    _fail(label, f"invalid size for {member.name}")
                by_name[member.name] = member
            if set(by_name) != RAW_ARCHIVE_MEMBERS:
                _fail(label, f"inventory mismatch: {sorted(by_name)}")
            payloads: dict[str, bytes] = {}
            for name, member in by_name.items():
                source = archive.extractfile(member)
                if source is None:
                    _fail(label, f"cannot read {name}")
                raw = source.read(MAX_JSON_BYTES + 1)
                if len(raw) != member.size:
                    _fail(label, f"truncated or oversized member {name}")
                payloads[name] = raw
    except EvidenceError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail(label, f"cannot read uncompressed tar: {error}")

    expected_checksums = b"".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n".encode("ascii")
        for name in RAW_ARCHIVE_PAYLOADS
    )
    if payloads["SHA256SUMS"] != expected_checksums:
        _fail(f"{label}.SHA256SUMS", "does not exactly checksum the five payload files")
    model_manifest = payloads["model-SHA256SUMS"]
    return {
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "payloads": payloads,
        "raw": _parse_json_bytes(payloads["raw-evidence.json"], "raw evidence"),
        "golden": _parse_json_bytes(
            payloads["correctness-golden.json"], "correctness golden"
        ),
        "shutdown": _parse_json_bytes(
            payloads["shutdown-metrics.json"], "shutdown metrics"
        ),
        "repeat_shutdown": _parse_json_bytes(
            payloads["repeat-shutdown-metrics.json"], "repeat shutdown metrics"
        ),
        "model_manifest": model_manifest,
        "model_files": _parse_model_manifest(model_manifest),
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
            manifest = json.loads(manifest_file.read(), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (
        OSError,
        tarfile.TarError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        release_common.ReleaseContractError,
    ) as error:
        _fail("--release-bundle", f"cannot inspect release bundle: {error}")
    if embedded_binary != binary_sha256:
        _fail("--release-bundle", "embedded binary differs from --release-binary")
    if not isinstance(manifest, dict) or manifest.get("artifact", {}).get("source_revision") != revision:
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
    if len(prompt.encode("utf-8")) > 16 * 1024:
        _fail("correctness golden.prompt", "exceeds 16384 UTF-8 bytes")
    if "\n" in prompt or "\r" in prompt:
        _fail("correctness golden.prompt", "must be one line for the shell-free probe contract")
    return {
        "correctness_gate_id": _string(
            row["correctness_gate_id"], "correctness golden.correctness_gate_id"
        ),
        "correctness_report_sha256": _sha(
            row["correctness_report_sha256"], "correctness golden.correctness_report_sha256"
        ),
        "source_revision": _string(
            row["source_revision"], "correctness golden.source_revision", GIT_RE
        ),
        "model_id": _string(
            row["model_id"], "correctness golden.model_id", MODEL_ID_RE
        ),
        "model_revision": _string(
            row["model_revision"], "correctness golden.model_revision"
        ),
        "config_sha256": _sha(
            row["config_sha256"], "correctness golden.config_sha256"
        ),
        "weights_sha256": _sha(row["weights_sha256"], "correctness golden.weights_sha256"),
        "tokenizer_aggregate_sha256": _sha(
            row["tokenizer_aggregate_sha256"],
            "correctness golden.tokenizer_aggregate_sha256",
        ),
        "tokenizer_json_sha256": _sha(
            row["tokenizer_json_sha256"],
            "correctness golden.tokenizer_json_sha256",
        ),
        "prompt": prompt,
        "max_tokens": _integer(row["max_tokens"], "correctness golden.max_tokens", 2),
        "expected_greedy_text_sha256": _sha(
            row["expected_greedy_text_sha256"],
            "correctness golden.expected_greedy_text_sha256",
        ),
    }


def _validate_correctness_report(document: Mapping[str, Any]) -> dict[str, str]:
    if document.get("gate_id") != CORRECTNESS_GATE or document.get("status") != "pass":
        _fail("correctness report", "must be the passing native E0 v2 gate")
    bindings = document.get("bindings")
    if not isinstance(bindings, dict):
        _fail("correctness report.bindings", "must be an object")
    required = {
        "candidate_git_revision",
        "candidate_git_status_sha256",
        "model_id",
        "model_revision",
        "config_sha256",
        "weights_sha256",
        "tokenizer_sha256",
    }
    missing = sorted(required - set(bindings))
    if missing:
        _fail("correctness report.bindings", f"missing fields: {', '.join(missing)}")
    if bindings["candidate_git_status_sha256"] != hashlib.sha256(b"").hexdigest():
        _fail("correctness report.bindings.candidate_git_status_sha256", "candidate tree was dirty")
    return {
        "correctness_gate_id": CORRECTNESS_GATE,
        "source_revision": _string(
            bindings["candidate_git_revision"],
            "correctness report.bindings.candidate_git_revision",
            GIT_RE,
        ),
        "model_id": _string(
            bindings["model_id"],
            "correctness report.bindings.model_id",
            MODEL_ID_RE,
        ),
        "model_revision": _string(
            bindings["model_revision"], "correctness report.bindings.model_revision"
        ),
        "config_sha256": _sha(
            bindings["config_sha256"], "correctness report.bindings.config_sha256"
        ),
        "weights_sha256": _sha(
            bindings["weights_sha256"], "correctness report.bindings.weights_sha256"
        ),
        "tokenizer_aggregate_sha256": _sha(
            bindings["tokenizer_sha256"], "correctness report.bindings.tokenizer_sha256"
        ),
    }


def _validate_metrics(value: Any) -> dict[str, Any]:
    row = _closed(
        value,
        {"active_requests", "waiting_requests", "kv_allocated_blocks", "allocation", "counters"},
        "raw.observations.shutdown.metrics",
    )
    allocation = _closed(
        row["allocation"],
        {"device_live_count", "device_live_bytes", "pinned_live_count", "pinned_live_bytes"},
        "raw.observations.shutdown.metrics.allocation",
    )
    counters = _closed(
        row["counters"],
        {"cancellations", "disconnects", "overloads", "dropped_observations"},
        "raw.observations.shutdown.metrics.counters",
    )
    result = {
        "active_requests": _integer(row["active_requests"], "shutdown.active_requests"),
        "waiting_requests": _integer(row["waiting_requests"], "shutdown.waiting_requests"),
        "kv_allocated_blocks": _integer(row["kv_allocated_blocks"], "shutdown.kv_allocated_blocks"),
        "allocation": {key: _integer(value, f"shutdown.allocation.{key}") for key, value in allocation.items()},
        "counters": {key: _integer(value, f"shutdown.counters.{key}") for key, value in counters.items()},
    }
    zero_values = [
        result["active_requests"],
        result["waiting_requests"],
        result["kv_allocated_blocks"],
        *result["allocation"].values(),
    ]
    if any(zero_values):
        _fail("raw.observations.shutdown.metrics", "final live-resource gauges must all be zero")
    return result


def _validate_raw(document: Mapping[str, Any]) -> dict[str, Any]:
    row = _closed(
        document,
        {"schema_version", "run_id", "recorded_at_utc", "status", "source", "release", "model", "runtime", "observations"},
        "raw",
    )
    if row["schema_version"] != RAW_SCHEMA or row["status"] != "success":
        _fail("raw", "schema must be v1 and status must be success")
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
        {
            "model_id", "model_revision", "model_tree_sha256", "config_sha256", "weights_sha256",
            "tokenizer_aggregate_sha256", "tokenizer_json_sha256", "correctness_gate_id",
            "correctness_report_sha256", "correctness_golden_sha256",
        },
        "raw.model",
    )
    model_result = {
        "model_id": _string(model["model_id"], "raw.model.model_id", MODEL_ID_RE),
        "model_revision": _string(model["model_revision"], "raw.model.model_revision"),
        "model_tree_sha256": _sha(model["model_tree_sha256"], "raw.model.model_tree_sha256"),
        "config_sha256": _sha(model["config_sha256"], "raw.model.config_sha256"),
        "weights_sha256": _sha(model["weights_sha256"], "raw.model.weights_sha256"),
        "tokenizer_aggregate_sha256": _sha(
            model["tokenizer_aggregate_sha256"],
            "raw.model.tokenizer_aggregate_sha256",
        ),
        "tokenizer_json_sha256": _sha(
            model["tokenizer_json_sha256"], "raw.model.tokenizer_json_sha256"
        ),
        "correctness_gate_id": _string(
            model["correctness_gate_id"], "raw.model.correctness_gate_id"
        ),
        "correctness_report_sha256": _sha(
            model["correctness_report_sha256"], "raw.model.correctness_report_sha256"
        ),
        "correctness_golden_sha256": _sha(model["correctness_golden_sha256"], "raw.model.correctness_golden_sha256"),
    }
    runtime = _closed(
        row["runtime"],
        {"container_ids", "network_mode", "image_id", "image_binary_sha256"},
        "raw.runtime",
    )
    if runtime["network_mode"] != "none":
        _fail("raw.runtime.network_mode", "must be none")
    image_id = _string(runtime["image_id"], "raw.runtime.image_id")
    if not image_id.startswith("sha256:"):
        _fail("raw.runtime.image_id", "must be sha256:<digest>")
    _sha(image_id.removeprefix("sha256:"), "raw.runtime.image_id")
    container_ids = runtime["container_ids"]
    if not isinstance(container_ids, list) or len(container_ids) != 2:
        _fail("raw.runtime.container_ids", "must record exactly two clean-start containers")
    container_ids = [
        _string(value, f"raw.runtime.container_ids[{index}]", CONTAINER_RE)
        for index, value in enumerate(container_ids)
    ]
    if len(set(container_ids)) != 2:
        _fail("raw.runtime.container_ids", "clean-start container IDs must be distinct")
    runtime_result = {
        "container_ids": container_ids,
        "network_mode": "none",
        "image_id": image_id,
        "image_binary_sha256": _sha(runtime["image_binary_sha256"], "raw.runtime.image_binary_sha256"),
    }

    observations = _closed(
        row["observations"],
        {"readyz", "models", "greedy", "sampling", "cancellation", "shutdown", "python_free"},
        "raw.observations",
    )
    ready = _closed(observations["readyz"], {"http_status", "ready", "accepting"}, "raw.observations.readyz")
    if _integer(ready["http_status"], "readyz.http_status") != 200:
        _fail("readyz.http_status", "must be 200")
    _true(ready["ready"], "readyz.ready")
    _true(ready["accepting"], "readyz.accepting")
    models = _closed(observations["models"], {"http_status", "model_ids"}, "raw.observations.models")
    if _integer(models["http_status"], "models.http_status") != 200:
        _fail("models.http_status", "must be 200")
    if models["model_ids"] != [model_result["model_id"]]:
        _fail("models.model_ids", "must contain exactly the bound loaded model")

    greedy = _closed(
        observations["greedy"],
        {
            "non_stream_http_status", "stream_http_status", "non_stream_text_sha256",
            "stream_text_sha256", "approved_text_sha256", "completion_tokens",
            "stream_token_events", "finish_reason", "stream_done", "prompt_sha256",
            "max_tokens",
        },
        "raw.observations.greedy",
    )
    for key in ("non_stream_http_status", "stream_http_status"):
        if _integer(greedy[key], f"greedy.{key}") != 200:
            _fail(f"greedy.{key}", "must be 200")
    greedy_hashes = [
        _sha(greedy[key], f"greedy.{key}")
        for key in ("non_stream_text_sha256", "stream_text_sha256", "approved_text_sha256")
    ]
    prompt_sha256 = _sha(greedy["prompt_sha256"], "greedy.prompt_sha256")
    max_tokens = _integer(greedy["max_tokens"], "greedy.max_tokens", 2)
    if len(set(greedy_hashes)) != 1:
        _fail("raw.observations.greedy", "non-stream, stream, and approved golden hashes must match")
    completion_tokens = _integer(greedy["completion_tokens"], "greedy.completion_tokens", 2)
    if _integer(greedy["stream_token_events"], "greedy.stream_token_events", 2) != completion_tokens:
        _fail("greedy.stream_token_events", "must equal non-stream completion tokens")
    if greedy["finish_reason"] not in {"length", "stop"}:
        _fail("greedy.finish_reason", "must be length or stop")
    _true(greedy["stream_done"], "greedy.stream_done")

    sampling = _closed(
        observations["sampling"],
        {
            "seed", "temperature", "top_p", "first_http_status", "second_http_status",
            "first_completion_tokens", "second_completion_tokens", "first_finish_reason",
            "second_finish_reason", "first_text_sha256", "second_text_sha256",
        },
        "raw.observations.sampling",
    )
    _integer(sampling["seed"], "sampling.seed")
    _number(sampling["temperature"], "sampling.temperature", positive=True)
    top_p = _number(sampling["top_p"], "sampling.top_p", positive=True)
    if top_p > 1:
        _fail("sampling.top_p", "must be <= 1")
    for key in ("first_http_status", "second_http_status"):
        if _integer(sampling[key], f"sampling.{key}") != 200:
            _fail(f"sampling.{key}", "must be 200")
    first_sampling_tokens = _integer(
        sampling["first_completion_tokens"], "sampling.first_completion_tokens", 1
    )
    second_sampling_tokens = _integer(
        sampling["second_completion_tokens"], "sampling.second_completion_tokens", 1
    )
    if first_sampling_tokens != second_sampling_tokens:
        _fail("raw.observations.sampling", "fixed-seed completion token counts must match")
    for key in ("first_finish_reason", "second_finish_reason"):
        if sampling[key] not in {"length", "stop"}:
            _fail(f"sampling.{key}", "must be length or stop")
    if sampling["first_finish_reason"] != sampling["second_finish_reason"]:
        _fail("raw.observations.sampling", "fixed-seed finish reasons must match")
    first_sampling_sha = _sha(sampling["first_text_sha256"], "sampling.first_text_sha256")
    second_sampling_sha = _sha(sampling["second_text_sha256"], "sampling.second_text_sha256")
    if first_sampling_sha == hashlib.sha256(b"").hexdigest():
        _fail("sampling.first_text_sha256", "empty output is forbidden")
    if first_sampling_sha != second_sampling_sha:
        _fail("raw.observations.sampling", "fixed-seed responses must match")

    cancellation = _closed(
        observations["cancellation"],
        {
            "disconnect_probe_sent", "cancellations_before", "cancellations_after",
            "disconnects_before", "disconnects_after", "active_requests_after",
            "waiting_requests_after",
        },
        "raw.observations.cancellation",
    )
    _true(cancellation["disconnect_probe_sent"], "cancellation.disconnect_probe_sent")
    cancels_before = _integer(cancellation["cancellations_before"], "cancellation.cancellations_before")
    cancels_after = _integer(cancellation["cancellations_after"], "cancellation.cancellations_after")
    disconnects_before = _integer(cancellation["disconnects_before"], "cancellation.disconnects_before")
    disconnects_after = _integer(cancellation["disconnects_after"], "cancellation.disconnects_after")
    if cancels_after <= cancels_before or disconnects_after <= disconnects_before:
        _fail("raw.observations.cancellation", "must increment cancellation and disconnect counters")
    if _integer(cancellation["active_requests_after"], "cancellation.active_requests_after") != 0:
        _fail("cancellation.active_requests_after", "must return to zero")
    if _integer(cancellation["waiting_requests_after"], "cancellation.waiting_requests_after") != 0:
        _fail("cancellation.waiting_requests_after", "must return to zero")

    shutdown = _closed(
        observations["shutdown"],
        {
            "signal", "exit_code", "metrics", "metrics_sha256", "repeat_exit_code",
            "repeat_metrics", "repeat_metrics_sha256",
        },
        "raw.observations.shutdown",
    )
    if (
        shutdown["signal"] != "SIGTERM"
        or _integer(shutdown["exit_code"], "shutdown.exit_code") != 0
        or _integer(shutdown["repeat_exit_code"], "shutdown.repeat_exit_code") != 0
    ):
        _fail("raw.observations.shutdown", "both clean starts must record SIGTERM exit zero")
    shutdown_metrics = _validate_metrics(shutdown["metrics"])
    repeat_shutdown_metrics = _validate_metrics(shutdown["repeat_metrics"])
    shutdown_metrics_sha256 = _sha(shutdown["metrics_sha256"], "shutdown.metrics_sha256")
    repeat_shutdown_metrics_sha256 = _sha(
        shutdown["repeat_metrics_sha256"], "shutdown.repeat_metrics_sha256"
    )

    python_free = _closed(
        observations["python_free"],
        {
            "forbidden_executables", "forbidden_artifact_count", "processes",
            "manifest_dependencies", "loader_dependencies", "unresolved_dependencies",
            "forbidden_dependency_matches",
        },
        "raw.observations.python_free",
    )
    for field in ("forbidden_executables", "unresolved_dependencies", "forbidden_dependency_matches"):
        if python_free[field] != []:
            _fail(f"raw.observations.python_free.{field}", "must be an empty array")
    if _integer(python_free["forbidden_artifact_count"], "python_free.forbidden_artifact_count") != 0:
        _fail("python_free.forbidden_artifact_count", "must be zero")
    dependencies = python_free["manifest_dependencies"]
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        _fail("python_free.manifest_dependencies", "must be a string array")
    if dependencies != sorted(set(dependencies)):
        _fail("python_free.manifest_dependencies", "must be sorted and unique")
    dependency_set = set(dependencies)
    if not REQUIRED_DEPENDENCIES <= dependency_set or dependency_set - ALLOWED_DEPENDENCIES:
        _fail("python_free.manifest_dependencies", "missing or unreviewed native dependency")
    loader_dependencies = python_free["loader_dependencies"]
    if not isinstance(loader_dependencies, list) or not loader_dependencies:
        _fail("python_free.loader_dependencies", "must be a non-empty string array")
    if any(not isinstance(item, str) or FORBIDDEN_RE.search(item) for item in loader_dependencies):
        _fail("python_free.loader_dependencies", "contains invalid or forbidden dependency text")
    processes = python_free["processes"]
    if not isinstance(processes, list) or not processes:
        _fail("python_free.processes", "must record at least the server process")
    saw_server = False
    for index, value in enumerate(processes):
        process = _closed(value, {"pid", "ppid", "comm", "args"}, f"python_free.processes[{index}]")
        _integer(process["pid"], f"python_free.processes[{index}].pid", 1)
        _integer(process["ppid"], f"python_free.processes[{index}].ppid")
        comm = _string(process["comm"], f"python_free.processes[{index}].comm")
        arguments = _string(process["args"], f"python_free.processes[{index}].args")
        if FORBIDDEN_RE.search(comm) or FORBIDDEN_RE.search(arguments):
            _fail(f"python_free.processes[{index}]", "contains a forbidden Python-family process")
        saw_server |= comm == "rustinfer"
    if not saw_server:
        _fail("python_free.processes", "does not contain the rustinfer server")

    return {
        "source": source_result,
        "release": release_result,
        "model": model_result,
        "runtime": runtime_result,
        "approved_text_sha256": greedy_hashes[0],
        "prompt_sha256": prompt_sha256,
        "max_tokens": max_tokens,
        "shutdown_metrics": shutdown_metrics,
        "repeat_shutdown_metrics": repeat_shutdown_metrics,
        "shutdown_metrics_sha256": shutdown_metrics_sha256,
        "repeat_shutdown_metrics_sha256": repeat_shutdown_metrics_sha256,
    }


def _error_report() -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "gate": GATE,
        "status": "error",
        "source": None,
        "raw_evidence_sha256": None,
        "checks": [{"id": "input_contract", "passed": False}],
    }


def validate_bound_raw_archive(
    archive: Mapping[str, Any],
    *,
    source_revision: str,
    source_archive_sha256: str,
    release_binary_sha256: str,
    release_bundle_sha256: str,
    image_id: str,
    correctness_report: Mapping[str, Any],
    correctness_report_sha256: str,
) -> tuple[dict[str, Any], str | None]:
    """Replay a parsed raw archive against immutable candidate bindings."""

    report = _error_report()
    try:
        _string(source_revision, "source revision", GIT_RE)
        archive_sha256 = _sha(source_archive_sha256, "source archive SHA-256")
        binary_sha256 = _sha(release_binary_sha256, "release binary SHA-256")
        bundle_sha256 = _sha(release_bundle_sha256, "release bundle SHA-256")
        if not image_id.startswith("sha256:"):
            _fail("image ID", "must be sha256:<digest>")
        image_sha256 = _sha(image_id.removeprefix("sha256:"), "image ID")
        correctness_sha256 = _sha(
            correctness_report_sha256, "correctness report SHA-256"
        )
        archive_digest = _sha(archive.get("archive_sha256"), "raw archive SHA-256")
        payloads = archive.get("payloads")
        if not isinstance(payloads, dict) or set(payloads) != RAW_ARCHIVE_MEMBERS:
            _fail("raw archive", "parsed payload inventory mismatch")

        validated = _validate_raw(_closed_archive_document(archive, "raw"))
        golden = _validate_golden(_closed_archive_document(archive, "golden"))
        correctness = _validate_correctness_report(correctness_report)
        shutdown_document = _closed_archive_document(archive, "shutdown")
        repeat_shutdown_document = _closed_archive_document(
            archive, "repeat_shutdown"
        )
        model_manifest = archive.get("model_manifest")
        model_files = archive.get("model_files")
        if not isinstance(model_manifest, bytes) or not isinstance(model_files, dict):
            _fail("raw archive", "parsed model manifest is missing")

        expected_source = {
            "git_revision": source_revision,
            "git_dirty": False,
            "source_archive_sha256": archive_sha256,
        }
        expected_release = {
            "binary_sha256": binary_sha256,
            "bundle_sha256": bundle_sha256,
            "image_sha256": image_sha256,
        }
        if validated["source"] != expected_source:
            _fail("raw.source", "does not match the candidate source")
        if validated["release"] != expected_release:
            _fail("raw.release", "does not match the candidate release")
        if validated["runtime"]["image_id"] != image_id:
            _fail("raw.runtime.image_id", "does not match the candidate image")
        if validated["runtime"]["image_binary_sha256"] != binary_sha256:
            _fail("raw.runtime.image_binary_sha256", "image binary differs from release binary")

        model = validated["model"]
        manifest_sha256 = hashlib.sha256(model_manifest).hexdigest()
        if model["model_tree_sha256"] != manifest_sha256:
            _fail("raw.model.model_tree_sha256", "does not hash model-SHA256SUMS")
        expected_model_files = {
            "config.json": model["config_sha256"],
            "model.safetensors": model["weights_sha256"],
            "tokenizer.json": model["tokenizer_json_sha256"],
        }
        for name, digest in expected_model_files.items():
            if model_files.get(name) != digest:
                _fail("raw archive.model-SHA256SUMS", f"{name} binding mismatch")
        aggregate = _tokenizer_aggregate_sha256(model_files)
        if model["tokenizer_aggregate_sha256"] != aggregate:
            _fail(
                "raw.model.tokenizer_aggregate_sha256",
                "does not equal the five-file tokenizer aggregate",
            )

        golden_sha256 = hashlib.sha256(
            payloads["correctness-golden.json"]
        ).hexdigest()
        if model["correctness_golden_sha256"] != golden_sha256:
            _fail("raw.model.correctness_golden_sha256", "golden archive binding mismatch")
        if model["correctness_report_sha256"] != correctness_sha256:
            _fail("raw.model.correctness_report_sha256", "native report binding mismatch")

        expected_provenance = {
            "correctness_gate_id": CORRECTNESS_GATE,
            "correctness_report_sha256": correctness_sha256,
            "source_revision": source_revision,
            "model_id": model["model_id"],
            "model_revision": model["model_revision"],
            "config_sha256": model["config_sha256"],
            "weights_sha256": model["weights_sha256"],
            "tokenizer_aggregate_sha256": model["tokenizer_aggregate_sha256"],
        }
        golden_provenance = {
            key: golden[key] for key in expected_provenance
        }
        if golden_provenance != expected_provenance:
            _fail("correctness golden", "does not match candidate/model provenance")
        if golden["tokenizer_json_sha256"] != model["tokenizer_json_sha256"]:
            _fail("correctness golden.tokenizer_json_sha256", "model binding mismatch")
        correctness_expected = {
            key: value
            for key, value in expected_provenance.items()
            if key != "correctness_report_sha256"
        }
        correctness_provenance = {
            key: correctness[key] for key in correctness_expected
        }
        if correctness_provenance != correctness_expected:
            _fail("correctness report.bindings", "does not match candidate/model provenance")
        if model["correctness_gate_id"] != CORRECTNESS_GATE:
            _fail("raw.model.correctness_gate_id", f"must be {CORRECTNESS_GATE}")
        if model["model_id"] != golden["model_id"]:
            _fail("raw.model.model_id", "differs from correctness golden")
        if validated["approved_text_sha256"] != golden["expected_greedy_text_sha256"]:
            _fail("raw.observations.greedy.approved_text_sha256", "differs from correctness golden")
        if validated["prompt_sha256"] != hashlib.sha256(golden["prompt"].encode("utf-8")).hexdigest():
            _fail("raw.observations.greedy.prompt_sha256", "differs from correctness golden")
        if validated["max_tokens"] != golden["max_tokens"]:
            _fail("raw.observations.greedy.max_tokens", "differs from correctness golden")

        if shutdown_document != validated["shutdown_metrics"]:
            _fail("raw.observations.shutdown.metrics", "differs from archived shutdown metrics")
        if repeat_shutdown_document != validated["repeat_shutdown_metrics"]:
            _fail(
                "raw.observations.shutdown.repeat_metrics",
                "differs from archived repeat shutdown metrics",
            )
        shutdown_sha256 = hashlib.sha256(payloads["shutdown-metrics.json"]).hexdigest()
        repeat_shutdown_sha256 = hashlib.sha256(
            payloads["repeat-shutdown-metrics.json"]
        ).hexdigest()
        if shutdown_sha256 != validated["shutdown_metrics_sha256"]:
            _fail("raw.observations.shutdown.metrics_sha256", "shutdown digest mismatch")
        if repeat_shutdown_sha256 != validated["repeat_shutdown_metrics_sha256"]:
            _fail("raw.observations.shutdown.repeat_metrics_sha256", "repeat shutdown digest mismatch")

        report = {
            "schema_version": REPORT_SCHEMA,
            "gate": GATE,
            "status": "passed",
            "source": {
                "git_revision": source_revision,
                "git_dirty": False,
                "source_archive_sha256": archive_sha256,
                "release_binary_sha256": binary_sha256,
                "release_bundle_sha256": bundle_sha256,
                "release_image_sha256": image_sha256,
            },
            "raw_evidence_sha256": archive_digest,
            "checks": [{"id": check_id, "passed": True} for check_id in CHECK_IDS],
        }
        return report, None
    except (EvidenceError, OSError) as error:
        return report, str(error)


def _closed_archive_document(archive: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = archive.get(key)
    if not isinstance(value, dict):
        _fail("raw archive", f"parsed {key} document is missing")
    return value


def _evaluate_unbundled_legacy(
    evidence: Path,
    *,
    source_revision: str,
    source_archive: Path,
    release_binary: Path,
    release_bundle: Path,
    image_id: str,
    model_dir: Path,
    expected_model_tree_sha256: str,
    weights: Path,
    expected_weights_sha256: str,
    tokenizer: Path,
    expected_tokenizer_json_sha256: str,
    expected_tokenizer_aggregate_sha256: str,
    correctness_golden: Path,
    expected_correctness_golden_sha256: str,
    correctness_report: Path,
    expected_correctness_report_sha256: str,
    shutdown_metrics: Path,
    repeat_shutdown_metrics: Path,
) -> tuple[dict[str, Any], str | None]:
    """Return a closed release attestation and an optional diagnostic."""

    report = _error_report()
    try:
        _string(source_revision, "--source-revision", GIT_RE)
        if not image_id.startswith("sha256:"):
            _fail("--image-id", "must be sha256:<digest>")
        image_sha256 = _sha(image_id.removeprefix("sha256:"), "--image-id")
        expected_model = _sha(expected_model_tree_sha256, "--model-tree-sha256")
        expected_weights = _sha(expected_weights_sha256, "--weights-sha256")
        expected_tokenizer_json = _sha(
            expected_tokenizer_json_sha256, "--tokenizer-json-sha256"
        )
        expected_tokenizer_aggregate = _sha(
            expected_tokenizer_aggregate_sha256, "--tokenizer-aggregate-sha256"
        )
        expected_golden = _sha(expected_correctness_golden_sha256, "--correctness-golden-sha256")
        expected_correctness = _sha(
            expected_correctness_report_sha256, "--correctness-report-sha256"
        )
        raw, raw_bytes = _read_json(evidence, "raw evidence")
        validated = _validate_raw(raw)
        archive_sha256 = _file_sha256(source_archive, "--source-archive")
        binary_sha256 = _file_sha256(release_binary, "--release-binary")
        bundle_sha256 = _file_sha256(release_bundle, "--release-bundle")
        actual_model = model_tree_sha256(model_dir)
        actual_weights = _file_sha256(weights, "--weights")
        actual_tokenizer = _file_sha256(tokenizer, "--tokenizer")
        golden_document, _ = _read_json(correctness_golden, "correctness golden")
        golden = _validate_golden(golden_document)
        actual_golden = _file_sha256(correctness_golden, "--correctness-golden")
        correctness_document, _ = _read_json(correctness_report, "correctness report")
        correctness = _validate_correctness_report(correctness_document)
        actual_correctness = _file_sha256(correctness_report, "--correctness-report")
        shutdown_document, _ = _read_json(shutdown_metrics, "shutdown metrics")
        repeat_shutdown_document, _ = _read_json(
            repeat_shutdown_metrics, "repeat shutdown metrics"
        )
        actual_shutdown = _file_sha256(shutdown_metrics, "--shutdown-metrics")
        actual_repeat_shutdown = _file_sha256(
            repeat_shutdown_metrics, "--repeat-shutdown-metrics"
        )
        _verify_bundle(release_bundle, binary_sha256, source_revision)

        expected_source = {
            "git_revision": source_revision,
            "git_dirty": False,
            "source_archive_sha256": archive_sha256,
        }
        expected_release = {
            "binary_sha256": binary_sha256,
            "bundle_sha256": bundle_sha256,
            "image_sha256": image_sha256,
        }
        if validated["source"] != expected_source:
            _fail("raw.source", "does not match the supplied source archive and revision")
        if validated["release"] != expected_release:
            _fail("raw.release", "does not match the supplied release artifacts and image")
        if validated["runtime"]["image_id"] != image_id:
            _fail("raw.runtime.image_id", "does not match --image-id")
        if validated["runtime"]["image_binary_sha256"] != binary_sha256:
            _fail("raw.runtime.image_binary_sha256", "image binary differs from release binary")
        if actual_model != expected_model or validated["model"]["model_tree_sha256"] != expected_model:
            _fail("raw.model.model_tree_sha256", "model tree binding mismatch")
        if actual_weights != expected_weights or validated["model"]["weights_sha256"] != expected_weights:
            _fail("raw.model.weights_sha256", "weights binding mismatch")
        if (
            actual_tokenizer != expected_tokenizer_json
            or validated["model"]["tokenizer_json_sha256"]
            != expected_tokenizer_json
            or golden["tokenizer_json_sha256"] != expected_tokenizer_json
        ):
            _fail("raw.model.tokenizer_json_sha256", "tokenizer.json binding mismatch")
        if (
            validated["model"]["tokenizer_aggregate_sha256"]
            != expected_tokenizer_aggregate
        ):
            _fail(
                "raw.model.tokenizer_aggregate_sha256",
                "tokenizer aggregate binding mismatch",
            )
        if actual_golden != expected_golden or validated["model"]["correctness_golden_sha256"] != expected_golden:
            _fail("raw.model.correctness_golden_sha256", "correctness golden binding mismatch")
        if (
            actual_correctness != expected_correctness
            or validated["model"]["correctness_report_sha256"] != expected_correctness
        ):
            _fail("raw.model.correctness_report_sha256", "correctness report binding mismatch")
        expected_provenance = {
            "correctness_gate_id": CORRECTNESS_GATE,
            "correctness_report_sha256": expected_correctness,
            "source_revision": source_revision,
            "model_id": validated["model"]["model_id"],
            "model_revision": validated["model"]["model_revision"],
            "config_sha256": validated["model"]["config_sha256"],
            "weights_sha256": expected_weights,
            "tokenizer_aggregate_sha256": expected_tokenizer_aggregate,
        }
        golden_provenance = {
            key: golden[key] for key in expected_provenance
        }
        correctness_provenance = {
            key: correctness[key]
            for key in expected_provenance
            if key not in {"correctness_report_sha256"}
        }
        if golden_provenance != expected_provenance:
            _fail("correctness golden", "does not match candidate/model correctness provenance")
        if correctness_provenance != {
            key: value
            for key, value in expected_provenance.items()
            if key != "correctness_report_sha256"
        }:
            _fail("correctness report.bindings", "does not match candidate/model provenance")
        if validated["model"]["correctness_gate_id"] != CORRECTNESS_GATE:
            _fail("raw.model.correctness_gate_id", f"must be {CORRECTNESS_GATE}")
        if validated["model"]["model_id"] != golden["model_id"]:
            _fail("raw.model.model_id", "differs from correctness golden")
        if validated["approved_text_sha256"] != golden["expected_greedy_text_sha256"]:
            _fail("raw.observations.greedy.approved_text_sha256", "differs from correctness golden")
        if validated["prompt_sha256"] != hashlib.sha256(golden["prompt"].encode("utf-8")).hexdigest():
            _fail("raw.observations.greedy.prompt_sha256", "differs from correctness golden")
        if validated["max_tokens"] != golden["max_tokens"]:
            _fail("raw.observations.greedy.max_tokens", "differs from correctness golden")
        if shutdown_document != validated["shutdown_metrics"]:
            _fail("raw.observations.shutdown.metrics", "differs from --shutdown-metrics")
        if repeat_shutdown_document != validated["repeat_shutdown_metrics"]:
            _fail(
                "raw.observations.shutdown.repeat_metrics",
                "differs from --repeat-shutdown-metrics",
            )
        if actual_shutdown != validated["shutdown_metrics_sha256"]:
            _fail("raw.observations.shutdown.metrics_sha256", "shutdown file digest mismatch")
        if actual_repeat_shutdown != validated["repeat_shutdown_metrics_sha256"]:
            _fail(
                "raw.observations.shutdown.repeat_metrics_sha256",
                "repeat shutdown file digest mismatch",
            )

        report = {
            "schema_version": REPORT_SCHEMA,
            "gate": GATE,
            "status": "passed",
            "source": {
                "git_revision": source_revision,
                "git_dirty": False,
                "source_archive_sha256": archive_sha256,
                "release_binary_sha256": binary_sha256,
                "release_bundle_sha256": bundle_sha256,
                "release_image_sha256": image_sha256,
            },
            "raw_evidence_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "checks": [{"id": check_id, "passed": True} for check_id in CHECK_IDS],
        }
        return report, None
    except (EvidenceError, OSError) as error:
        return report, str(error)


def evaluate(
    evidence: Path,
    *,
    raw_archive: Path,
    source_revision: str,
    source_archive: Path,
    release_binary: Path,
    release_bundle: Path,
    image_id: str,
    model_dir: Path,
    expected_model_tree_sha256: str,
    weights: Path,
    expected_weights_sha256: str,
    tokenizer: Path,
    expected_tokenizer_json_sha256: str,
    expected_tokenizer_aggregate_sha256: str,
    correctness_golden: Path,
    expected_correctness_golden_sha256: str,
    correctness_report: Path,
    expected_correctness_report_sha256: str,
    shutdown_metrics: Path,
    repeat_shutdown_metrics: Path,
) -> tuple[dict[str, Any], str | None]:
    """Verify producer artifacts and replay their non-circular raw archive."""

    report = _error_report()
    try:
        archive = load_raw_evidence_archive(raw_archive)
        payloads = archive["payloads"]

        external_payloads = {
            "raw-evidence.json": evidence,
            "correctness-golden.json": correctness_golden,
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
        expected_tokenizer_json = _sha(
            expected_tokenizer_json_sha256, "--tokenizer-json-sha256"
        )
        if _file_sha256(tokenizer, "--tokenizer") != expected_tokenizer_json:
            _fail("--tokenizer-json-sha256", "actual tokenizer.json binding mismatch")
        expected_tokenizer_aggregate = _sha(
            expected_tokenizer_aggregate_sha256, "--tokenizer-aggregate-sha256"
        )
        if _tokenizer_aggregate_sha256(archive["model_files"]) != expected_tokenizer_aggregate:
            _fail("--tokenizer-aggregate-sha256", "model tokenizer aggregate mismatch")

        expected_golden = _sha(
            expected_correctness_golden_sha256,
            "--correctness-golden-sha256",
        )
        if hashlib.sha256(payloads["correctness-golden.json"]).hexdigest() != expected_golden:
            _fail("--correctness-golden-sha256", "golden binding mismatch")
        correctness_document, correctness_bytes = _read_json(
            correctness_report, "correctness report"
        )
        expected_correctness = _sha(
            expected_correctness_report_sha256,
            "--correctness-report-sha256",
        )
        if hashlib.sha256(correctness_bytes).hexdigest() != expected_correctness:
            _fail("--correctness-report-sha256", "native report binding mismatch")

        archive_sha256 = _file_sha256(source_archive, "--source-archive")
        binary_sha256 = _file_sha256(release_binary, "--release-binary")
        bundle_sha256 = _file_sha256(release_bundle, "--release-bundle")
        _verify_bundle(release_bundle, binary_sha256, source_revision)
        return validate_bound_raw_archive(
            archive,
            source_revision=source_revision,
            source_archive_sha256=archive_sha256,
            release_binary_sha256=binary_sha256,
            release_bundle_sha256=bundle_sha256,
            image_id=image_id,
            correctness_report=correctness_document,
            correctness_report_sha256=expected_correctness,
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
        args.evidence,
        raw_archive=args.raw_archive,
        source_revision=args.source_revision,
        source_archive=args.source_archive,
        release_binary=args.release_binary,
        release_bundle=args.release_bundle,
        image_id=args.image_id,
        model_dir=args.model_dir,
        expected_model_tree_sha256=args.model_tree_sha256,
        weights=args.weights,
        expected_weights_sha256=args.weights_sha256,
        tokenizer=args.tokenizer,
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
