#!/usr/bin/env python3
"""Fail-closed checker for source-bound PR-16 reliability soak evidence."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import math
import os
import posixpath
import re
import stat
import sys
import tarfile
import tempfile
from contextlib import ExitStack
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence


MANIFEST_VERSION = "riley.reliability-soak-manifest.v1"
RUN_VERSION = "riley.reliability-soak-run.v1"
EVENT_VERSION = "riley.reliability-soak-event.v1"
REPORT_VERSION = "riley.reliability-soak-report.v2"
E2E_GOLDEN_VERSION = "riley.python-free-release-e2e-golden.v1"
NATIVE_CORRECTNESS_VERSION = "1.0.0"
NATIVE_CORRECTNESS_GATE = "smollm2-fp32-bf16-native-e0-v3"
REQUIRED_KINDS = {
    "steady",
    "burst-idle",
    "mixed",
    "invalid",
    "overload",
    "cancellation-disconnect",
    "near-kv",
    "graceful-restart",
    "rollback",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
PYTHON_RE = re.compile(r"(^|/)(python|python[23](?:\.[0-9]+)?)(?:$|\s)", re.IGNORECASE)
MAX_INPUT_BYTES = 512 * 1024 * 1024
RUNTIME_RECEIPT_FILENAMES = (
    "host-gpu.csv",
    "launcher-receipt.json",
    "release-runtime-closure.tsv",
    "release-image-inspect.json",
    "test-layer-image-inspect.json",
    "container-inspect-pre.json",
    "container-inspect-post.json",
)
RAW_ARCHIVE_PAYLOADS = tuple(
    sorted(("events.jsonl", "manifest.json", "run.json", *RUNTIME_RECEIPT_FILENAMES))
)
RAW_ARCHIVE_MEMBERS = (*RAW_ARCHIVE_PAYLOADS, "SHA256SUMS")
RAW_MEMBER_MAX_BYTES = {
    "events.jsonl": MAX_INPUT_BYTES,
    "manifest.json": 4 * 1024 * 1024,
    "run.json": 4 * 1024 * 1024,
    "host-gpu.csv": 4 * 1024,
    "launcher-receipt.json": 64 * 1024,
    "release-runtime-closure.tsv": 1024 * 1024,
    "release-image-inspect.json": 16 * 1024 * 1024,
    "test-layer-image-inspect.json": 16 * 1024 * 1024,
    "container-inspect-pre.json": 16 * 1024 * 1024,
    "container-inspect-post.json": 16 * 1024 * 1024,
    "SHA256SUMS": 4 * 1024,
}
MAX_RAW_ARCHIVE_BYTES = sum(RAW_MEMBER_MAX_BYTES.values()) + 64 * 1024
MAX_CORRECTNESS_GOLDEN_BYTES = 64 * 1024
MAX_NATIVE_CORRECTNESS_REPORT_BYTES = 16 * 1024 * 1024
CURL_TIMEOUT_EXIT_CODE = 28
CURL_WRITE_ERROR_EXIT_CODE = 23
DISCONNECT_RESPONSE_BYTES = 1024
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256 = (
    "45166193094802629b1f2d1c57fa4d6d71094802b0f28d6f2f5304531b4c5775"
)
EXPECTED_LAUNCH_ARGUMENTS = [
    "serve",
    "--model",
    "{model_path}",
    "--bind",
    "{bind}",
    "--reduction-profile",
    "canonical-v1",
    "--execution-completion",
    "{execution_completion}",
    "--residual-rmsnorm",
    "separate",
]
LAUNCHER_RECEIPT_VERSION = "riley.reliability-soak-launcher-receipt.v3"
DESIGNATED_HOSTNAME = "psyche-MS-7D91"
DESIGNATED_GPU = {
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_uuid": "GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0",
    "compute_capability": "8.9",
    "memory_total_mib": 24564,
    "driver_version": "580.173.02",
}
SOAK_USER = "65532:65532"
SOAK_ENTRYPOINT = ["/opt/riley-soak/ci/run_release_soak.sh"]
SOAK_CMD: list[str] = []
SOAK_MANIFEST_DESTINATION = "/run-input/reliability-soak-v1.json"
SOAK_MODEL_DESTINATION = "/model"
SOAK_EVIDENCE_DESTINATION = "/evidence"
SOAK_TMPFS_OPTIONS = {
    "rw",
    "nosuid",
    "nodev",
    "noexec",
    "size=67108864",
}
SOAK_IMAGE_LABELS = {
    "release_image_id": "org.riley.reliability-soak.release-image-id",
    "source_revision": "org.riley.reliability-soak.source-revision",
    "source_archive_sha256": (
        "org.riley.reliability-soak.source-archive-sha256"
    ),
    "release_binary_sha256": (
        "org.riley.reliability-soak.release-binary-sha256"
    ),
}
SOAK_IMAGE_ENVIRONMENT_OVERRIDES = {
    "DEBIAN_FRONTEND": "noninteractive",
    "LC_ALL": "C",
    "TZ": "UTC",
}
SOAK_RELEASE_ENVIRONMENT = {
    "PATH": (
        "/opt/riley/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
}
FORBIDDEN_RUNTIME_ENVIRONMENT = {
    "BASH_ENV",
    "BASHOPTS",
    "CDPATH",
    "CURL_HOME",
    "CUDA_VISIBLE_DEVICES",
    "ENV",
    "GCONV_PATH",
    "GLOBIGNORE",
    "GLIBC_TUNABLES",
    "HOME",
    "IFS",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LOCPATH",
    "MALLOC_TRACE",
    "NVIDIA_VISIBLE_DEVICES",
    "NLSPATH",
    "POSIXLY_CORRECT",
    "SHELLOPTS",
    "XDG_CONFIG_HOME",
}
FORBIDDEN_RUNTIME_ENVIRONMENT_PREFIXES = ("BASH_FUNC_", "LD_")
SOAK_HEALTHCHECK = {"Test": ["NONE"]}
SOAK_CONTAINER_PATH = "/opt/riley-soak/ci/run_release_soak.sh"
SOAK_CONTAINER_ARGS: list[str] = []
MINIMUM_SOAK_RUNTIME_SECONDS = 26_100
MAXIMUM_CONTAINER_NAME_LEAD_SECONDS = 300
DOCKER_TIMESTAMP_RE = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,9}))?Z$"
)
DOCKER_ZERO_TIMESTAMP = "0001-01-01T00:00:00Z"
RUN_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
COMPACT_UTC_TIMESTAMP_RE = re.compile(
    r"^(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})Z$"
)


class InputError(ValueError):
    """Evidence is malformed, incomplete, or not source-bound."""


def _fail(path: str, message: str) -> NoReturn:
    raise InputError(f"{path}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise InputError(f"non-finite JSON number {value!r} is forbidden")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            _fail(str(path), "exceeds evidence size bound")
        with path.open(encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_pairs,
                parse_constant=_nonfinite,
            )
    except FileNotFoundError:
        _fail(str(path), "file does not exist")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(str(path), f"cannot read strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        _fail(str(path), "root must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            _fail(str(path), "exceeds evidence size bound")
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _fail(str(path), "file does not exist")
    except (OSError, UnicodeDecodeError) as error:
        _fail(str(path), f"cannot read UTF-8 JSONL: {error}")
    if not lines:
        _fail(str(path), "must contain events")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            _fail(f"{path}:{line_number}", "blank JSONL lines are forbidden")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_pairs,
                parse_constant=_nonfinite,
            )
        except (json.JSONDecodeError, InputError) as error:
            _fail(f"{path}:{line_number}", f"invalid JSON: {error}")
        if not isinstance(value, dict):
            _fail(f"{path}:{line_number}", "event must be an object")
        rows.append(value)
    return rows


def _parse_jsonl_bytes(raw: bytes, path: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        _fail(path, f"cannot read UTF-8 JSONL: {error}")
    if not lines:
        _fail(path, "must contain events")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            _fail(f"{path}:{line_number}", "blank JSONL lines are forbidden")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_pairs,
                parse_constant=_nonfinite,
            )
        except (json.JSONDecodeError, InputError) as error:
            _fail(f"{path}:{line_number}", f"invalid JSON: {error}")
        if not isinstance(value, dict):
            _fail(f"{path}:{line_number}", "event must be an object")
        rows.append(value)
    return rows


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _exact(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    result = _object(value, path)
    missing = sorted(keys - set(result))
    extra = sorted(set(result) - keys)
    if missing or extra:
        _fail(path, f"closed object mismatch; missing={missing}, unexpected={extra}")
    return result


def _string(value: Any, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, "has invalid format")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(path, "must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(path, "must be representable as a finite number")
    if not math.isfinite(result) or result < minimum:
        _fail(path, f"must be finite and >= {minimum}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        _fail(str(path), f"cannot hash manifest: {error}")
    return digest.hexdigest()


def _stable_stat_fields() -> tuple[str, ...]:
    return ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in _stable_stat_fields()
    )


def _regular_file(
    path: Path,
    label: str,
    maximum_bytes: int,
    *,
    directory_fd: int | None = None,
    entry_name: str | None = None,
) -> tuple[Any, os.stat_result]:
    target: str | Path = entry_name if entry_name is not None else path
    try:
        if directory_fd is None:
            link_metadata = path.lstat()
        else:
            link_metadata = os.stat(
                target,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        if not stat.S_ISREG(link_metadata.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        descriptor = os.open(
            target,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        try:
            handle = os.fdopen(descriptor, "rb")
        except Exception:
            os.close(descriptor)
            raise
        metadata = os.fstat(handle.fileno())
    except (FileNotFoundError, OSError) as error:
        _fail(label, f"cannot open evidence file: {error}")
    if not stat.S_ISREG(metadata.st_mode) or not _same_stat(link_metadata, metadata):
        handle.close()
        _fail(label, "path changed while it was opened or is not a regular file")
    if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
        handle.close()
        _fail(label, f"must be between 1 and {maximum_bytes} bytes")
    return handle, metadata


def _load_regular_bytes(
    path: Path,
    label: str,
    maximum_bytes: int,
    *,
    directory_fd: int | None = None,
    entry_name: str | None = None,
) -> tuple[bytes, str]:
    handle, metadata = _regular_file(
        path,
        label,
        maximum_bytes,
        directory_fd=directory_fd,
        entry_name=entry_name,
    )
    try:
        raw = handle.read(metadata.st_size + 1)
        after = os.fstat(handle.fileno())
    except OSError as error:
        _fail(label, f"cannot read strict UTF-8 JSON: {error}")
    finally:
        handle.close()
    if len(raw) != metadata.st_size:
        _fail(label, "file changed or was truncated while being read")
    if not _same_stat(metadata, after):
        _fail(label, "file changed while being read")
    return raw, hashlib.sha256(raw).hexdigest()


def _load_regular_json_value(
    path: Path, label: str, maximum_bytes: int
) -> tuple[Any, str]:
    raw, digest = _load_regular_bytes(path, label, maximum_bytes)
    return _parse_json_value(raw, label), digest


def _parse_json_value(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InputError) as error:
        _fail(label, f"cannot read strict UTF-8 JSON: {error}")


def _load_regular_json(
    path: Path, label: str, maximum_bytes: int
) -> tuple[dict[str, Any], str]:
    value, digest = _load_regular_json_value(path, label, maximum_bytes)
    if not isinstance(value, dict):
        _fail(label, "root must be an object")
    return value, digest


def _load_runtime_receipt_payloads(
    directory: Path,
) -> dict[str, tuple[bytes, str]]:
    descriptor, before = _open_runtime_receipt_directory(directory)
    try:
        payloads = {
            name: _load_regular_bytes(
                directory / name,
                name,
                RAW_MEMBER_MAX_BYTES[name],
                directory_fd=descriptor,
                entry_name=name,
            )
            for name in sorted(RUNTIME_RECEIPT_FILENAMES)
        }
        _assert_runtime_receipt_directory_stable(descriptor, before)
        return payloads
    finally:
        os.close(descriptor)


def _validate_runtime_closure_receipt(raw: bytes) -> str:
    label = "release-runtime-closure.tsv"
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail(label, f"must be canonical ASCII: {error}")
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        _fail(label, "must be newline-terminated canonical TSV")
    lines = text.splitlines()
    if not lines or len(lines) > 1024 or lines != sorted(set(lines)):
        _fail(label, "must contain 1..1024 unique bytewise-sorted closure rows")
    loader_rows = 0
    unresolved_rows = 0
    for index, line in enumerate(lines, 1):
        row_label = f"{label}:{index}"
        fields = line.split("\t")
        if len(fields) != 4:
            _fail(row_label, "must contain dependency, resolved path, target, and SHA-256")
        dependency, resolved_path, target_path, target_sha256 = fields
        if re.fullmatch(r"[A-Za-z0-9_+./-]+", dependency) is None:
            _fail(row_label, "dependency name contains noncanonical characters")
        if resolved_path == "NOT_FOUND":
            if (
                dependency != "libcuda.so.1"
                or target_path != "-"
                or target_sha256 != "-"
            ):
                _fail(
                    row_label,
                    "only libcuda.so.1 may be unresolved as NOT_FOUND, -, -",
                )
            unresolved_rows += 1
            continue
        for field_name, path_value in (
            ("resolved path", resolved_path),
            ("target path", target_path),
        ):
            if (
                not path_value.startswith("/")
                or path_value.startswith("//")
                or posixpath.normpath(path_value) != path_value
            ):
                _fail(row_label, f"{field_name} must be a normalized absolute path")
        _string(target_sha256, f"{row_label}.sha256", SHA256_RE)
        if dependency.startswith("/"):
            if dependency != resolved_path or "ld-linux" not in dependency:
                _fail(row_label, "absolute dependency row must identify the resolved loader")
            loader_rows += 1
    if loader_rows != 1:
        _fail(label, "must contain exactly one resolved dynamic-loader row")
    if unresolved_rows != 1:
        _fail(label, "must contain exactly one unresolved libcuda.so.1 row")
    return hashlib.sha256(raw).hexdigest()


def _open_runtime_receipt_directory(
    directory: Path,
) -> tuple[int, os.stat_result]:
    label = "runtime receipt directory"
    descriptor = -1
    try:
        link_metadata = directory.lstat()
        if not stat.S_ISDIR(link_metadata.st_mode):
            _fail(label, "must be a directory, not a link or special file")
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or not _same_stat(
            link_metadata, before
        ):
            _fail(label, "path changed while it was opened or is not a directory")
        _assert_runtime_receipt_directory_stable(descriptor, before)
        return descriptor, before
    except InputError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except (FileNotFoundError, OSError) as error:
        if descriptor >= 0:
            os.close(descriptor)
        _fail(label, f"cannot open exact receipt inventory: {error}")


def _assert_runtime_receipt_directory_stable(
    descriptor: int, metadata: os.stat_result
) -> None:
    label = "runtime receipt directory"
    try:
        names = os.listdir(descriptor)
        after = os.fstat(descriptor)
    except OSError as error:
        _fail(label, f"cannot revalidate exact receipt inventory: {error}")
    expected_names = sorted(RUNTIME_RECEIPT_FILENAMES)
    if sorted(names) != expected_names:
        _fail(label, f"exact receipt inventory required: {expected_names}")
    if not _same_stat(metadata, after):
        _fail(label, "directory changed while receipts were consumed")


def _stream_sha256(handle: Any) -> str:
    digest = hashlib.sha256()
    while block := handle.read(1024 * 1024):
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest()


def _assert_held_file_stable(
    handle: Any, metadata: os.stat_result, label: str
) -> None:
    try:
        after = os.fstat(handle.fileno())
    except OSError as error:
        _fail(label, f"cannot revalidate held evidence file: {error}")
    if not _same_stat(metadata, after):
        _fail(label, "held evidence file changed while it was consumed")


def _assert_path_still_identifies_held_file(
    path: Path, metadata: os.stat_result, label: str
) -> None:
    try:
        current = path.lstat()
    except OSError as error:
        _fail(label, f"held evidence file changed while it was consumed: {error}")
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != metadata.st_dev
        or current.st_ino != metadata.st_ino
    ):
        _fail(label, "held evidence file changed while it was consumed")


def _canonical_tar_info(name: str, size: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o644
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    return member


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = _canonical_json_bytes(value)
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _jq_1_6_request_json_bytes(value: Any) -> bytes:
    """Mirror the pinned remote jq 1.6 shape for integral JSON numbers.

    jq 1.6 serializes an input such as ``0.0`` as ``0``. The soak driver hashes
    those exact jq-produced request bytes, so the offline checker must not use
    Python's distinct ``0.0`` spelling for the same reviewed manifest value.
    """

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    return _canonical_json_bytes(normalize(value))


def _normalized_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the reviewed contract while allowing only materialized golden hashes."""

    normalized = dict(manifest)
    golden = dict(_object(manifest.get("golden"), "manifest.golden"))
    golden["generated_sha256"] = "0" * 64
    golden["provenance_sha256"] = "0" * 64
    normalized["golden"] = golden
    return _canonical_sha256(normalized)


def _validate_manifest(value: dict[str, Any], path: str) -> dict[str, Any]:
    manifest = _exact(
        value,
        {"schema_version", "contract_id", "target", "thresholds", "requests", "golden", "scenarios"},
        path,
    )
    if manifest["schema_version"] != MANIFEST_VERSION:
        _fail(f"{path}.schema_version", f"must be {MANIFEST_VERSION}")
    _string(manifest["contract_id"], f"{path}.contract_id")
    target = _exact(
        manifest["target"],
        {"kind", "binary", "model_path", "bind", "completion_path", "health_path", "metrics_path", "shutdown_signal", "launch_arguments"},
        f"{path}.target",
    )
    if target["kind"] not in {"process", "container"}:
        _fail(f"{path}.target.kind", "must be process or container")
    if target["shutdown_signal"] != "TERM":
        _fail(f"{path}.target.shutdown_signal", "release soak requires SIGTERM")
    for key in ("binary", "model_path", "bind", "completion_path", "health_path", "metrics_path"):
        _string(target[key], f"{path}.target.{key}")
    if target["launch_arguments"] != EXPECTED_LAUNCH_ARGUMENTS:
        _fail(
            f"{path}.target.launch_arguments",
            "must be the exact canonical-v1/completion-mode/separate E0 command",
        )
    requests = _object(manifest["requests"], f"{path}.requests")
    golden = _exact(
        manifest["golden"],
        {"request_profile", "digest_domain", "generated_sha256", "provenance_sha256"},
        f"{path}.golden",
    )
    if golden["request_profile"] not in requests:
        _fail(f"{path}.golden.request_profile", "references an absent request")
    if golden["digest_domain"] != "completion-text-utf8":
        _fail(f"{path}.golden.digest_domain", "must be completion-text-utf8")
    for key in ("generated_sha256", "provenance_sha256"):
        digest = _string(golden[key], f"{path}.golden.{key}", SHA256_RE)
        if digest == "0" * 64:
            _fail(f"{path}.golden.{key}", "template placeholder must be materialized")
    thresholds = _exact(
        manifest["thresholds"],
        {
            "sample_interval_ms", "maximum_sample_gap_ms", "minimum_samples_per_scenario",
            "plateau_tail_fraction", "maximum_rss_plateau_growth_bytes",
            "maximum_rss_slope_bytes_per_hour", "maximum_vram_plateau_growth_bytes",
            "maximum_vram_slope_bytes_per_hour", "minimum_cancellations",
            "minimum_disconnects", "minimum_overloads", "graceful_shutdown_deadline_ms",
        },
        f"{path}.thresholds",
    )
    for key in (
        "sample_interval_ms", "maximum_sample_gap_ms", "minimum_samples_per_scenario",
        "maximum_rss_plateau_growth_bytes", "minimum_cancellations", "minimum_disconnects",
        "minimum_overloads", "graceful_shutdown_deadline_ms",
    ):
        _integer(thresholds[key], f"{path}.thresholds.{key}", 1 if key != "maximum_rss_plateau_growth_bytes" else 0)
    for key in (
        "maximum_rss_slope_bytes_per_hour", "maximum_vram_plateau_growth_bytes",
        "maximum_vram_slope_bytes_per_hour",
    ):
        _number(thresholds[key], f"{path}.thresholds.{key}")
    fraction = _number(thresholds["plateau_tail_fraction"], f"{path}.thresholds.plateau_tail_fraction")
    if fraction <= 0 or fraction > 1:
        _fail(f"{path}.thresholds.plateau_tail_fraction", "must be in (0, 1]")
    scenarios = manifest["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        _fail(f"{path}.scenarios", "must be a non-empty array")
    seen: set[str] = set()
    kind_counts: Counter[str] = Counter()
    rollback_modes: set[str] = set()
    for index, raw in enumerate(scenarios):
        scenario_path = f"{path}.scenarios[{index}]"
        scenario = _object(raw, scenario_path)
        required = {"id", "kind", "required", "duration_seconds", "concurrency", "cycle_interval_ms", "request_profile", "execution_completion"}
        if scenario.get("kind") == "mixed":
            required.add("secondary_request_profile")
        _exact(scenario, required, scenario_path)
        scenario_id = _string(scenario["id"], f"{scenario_path}.id")
        if scenario_id in seen:
            _fail(f"{scenario_path}.id", "duplicate scenario id")
        seen.add(scenario_id)
        kind = _string(scenario["kind"], f"{scenario_path}.kind")
        if kind not in REQUIRED_KINDS:
            _fail(f"{scenario_path}.kind", "is not a closed v1 scenario kind")
        if scenario["required"] is not True:
            _fail(f"{scenario_path}.required", "all release scenarios must be required")
        kind_counts[kind] += 1
        _integer(scenario["duration_seconds"], f"{scenario_path}.duration_seconds", 1)
        _integer(scenario["concurrency"], f"{scenario_path}.concurrency", 1)
        cycle_interval_ms = _integer(scenario["cycle_interval_ms"], f"{scenario_path}.cycle_interval_ms")
        if kind == "burst-idle" and cycle_interval_ms < 1000:
            _fail(f"{scenario_path}.cycle_interval_ms", "burst-idle requires a visible idle interval")
        profile = _string(scenario["request_profile"], f"{scenario_path}.request_profile")
        if profile not in requests:
            _fail(f"{scenario_path}.request_profile", "references an absent request")
        mode = scenario["execution_completion"]
        if mode not in {"iteration-batch", "per-operation"}:
            _fail(f"{scenario_path}.execution_completion", "must be a stable exact mode")
        if kind == "rollback":
            rollback_modes.add(mode)
    expected_counts = Counter({kind: 1 for kind in REQUIRED_KINDS})
    expected_counts["rollback"] = 2
    if kind_counts != expected_counts:
        _fail(f"{path}.scenarios", f"required kind counts mismatch: {dict(kind_counts)}")
    if rollback_modes != {"iteration-batch", "per-operation"}:
        _fail(f"{path}.scenarios", "rollback must cover both exact completion modes")
    normalized_digest = _normalized_manifest_sha256(manifest)
    if normalized_digest != REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256:
        _fail(
            path,
            "does not match the reviewed PR16 soak contract after golden normalization",
        )
    return manifest


def _validate_run(value: dict[str, Any], path: str, manifest_sha: str) -> dict[str, Any]:
    run = _exact(
        value,
        {"schema_version", "run_id", "manifest_sha256", "binding_sha256", "source", "target", "started_at_utc"},
        path,
    )
    if run["schema_version"] != RUN_VERSION:
        _fail(f"{path}.schema_version", f"must be {RUN_VERSION}")
    run_id = _string(run["run_id"], f"{path}.run_id")
    if run["manifest_sha256"] != manifest_sha:
        _fail(f"{path}.manifest_sha256", "does not bind the exact manifest bytes")
    source = _exact(
        run["source"],
        {"git_commit", "git_dirty", "source_archive_sha256", "binary_sha256", "image_sha256", "model_sha256", "model_id", "model_revision"},
        f"{path}.source",
    )
    _string(source["git_commit"], f"{path}.source.git_commit", GIT_RE)
    if source["git_dirty"] is not False:
        _fail(f"{path}.source.git_dirty", "release evidence requires a clean source tree")
    for key in ("source_archive_sha256", "binary_sha256", "image_sha256", "model_sha256"):
        _string(source[key], f"{path}.source.{key}", SHA256_RE)
    _string(source["model_id"], f"{path}.source.model_id")
    _string(source["model_revision"], f"{path}.source.model_revision")
    binding_sha = _canonical_sha256(source)
    if run["binding_sha256"] != binding_sha:
        _fail(f"{path}.binding_sha256", "does not bind the canonical source object")
    target = _exact(run["target"], {"kind", "pid", "image_id", "command_sha256"}, f"{path}.target")
    if target["kind"] not in {"process", "container"}:
        _fail(f"{path}.target.kind", "must be process or container")
    _integer(target["pid"], f"{path}.target.pid", 1)
    _string(target["image_id"], f"{path}.target.image_id")
    _string(target["command_sha256"], f"{path}.target.command_sha256", SHA256_RE)
    started_at_utc = _string(
        run["started_at_utc"], f"{path}.started_at_utc", RUN_TIMESTAMP_RE
    )
    _docker_timestamp_ns(started_at_utc, f"{path}.started_at_utc")
    expected_run_id = (
        f"soak-{started_at_utc.replace('-', '').replace(':', '')}-"
        f"{source['git_commit'][:12]}"
    )
    if run_id != expected_run_id:
        _fail(
            f"{path}.run_id",
            "must embed the exact started_at_utc stamp and source revision prefix",
        )
    return run


def _validate_trusted_correctness(
    manifest: dict[str, Any],
    run: dict[str, Any],
    correctness_golden_path: Path | str | None,
    native_correctness_report_path: Path | str | None,
) -> dict[str, str]:
    if correctness_golden_path is None:
        _fail(
            "--correctness-golden",
            "the independently reviewed E2E correctness golden is required",
        )
    if native_correctness_report_path is None:
        _fail(
            "--native-correctness-report",
            "the passing native E0 correctness report is required",
        )

    correctness_golden, correctness_golden_sha256 = _load_regular_json(
        Path(correctness_golden_path),
        "correctness golden",
        MAX_CORRECTNESS_GOLDEN_BYTES,
    )
    native_correctness, native_correctness_report_sha256 = _load_regular_json(
        Path(native_correctness_report_path),
        "native correctness report",
        MAX_NATIVE_CORRECTNESS_REPORT_BYTES,
    )

    golden = _exact(
        correctness_golden,
        {
            "schema_version",
            "correctness_gate_id",
            "correctness_report_sha256",
            "source_revision",
            "model_id",
            "model_revision",
            "config_sha256",
            "weights_sha256",
            "tokenizer_aggregate_sha256",
            "tokenizer_json_sha256",
            "prompt",
            "max_tokens",
            "expected_greedy_text_sha256",
        },
        "correctness golden",
    )
    if golden["schema_version"] != E2E_GOLDEN_VERSION:
        _fail(
            "correctness golden.schema_version",
            f"must be {E2E_GOLDEN_VERSION}",
        )
    if golden["correctness_gate_id"] != NATIVE_CORRECTNESS_GATE:
        _fail(
            "correctness golden.correctness_gate_id",
            f"must be {NATIVE_CORRECTNESS_GATE}",
        )
    for key in (
        "correctness_report_sha256",
        "config_sha256",
        "weights_sha256",
        "tokenizer_aggregate_sha256",
        "tokenizer_json_sha256",
        "expected_greedy_text_sha256",
    ):
        _string(golden[key], f"correctness golden.{key}", SHA256_RE)
    _string(golden["source_revision"], "correctness golden.source_revision", GIT_RE)
    _string(golden["model_id"], "correctness golden.model_id")
    _string(golden["model_revision"], "correctness golden.model_revision")
    prompt = _string(golden["prompt"], "correctness golden.prompt")
    if len(prompt.encode("utf-8")) > 16 * 1024 or "\n" in prompt or "\r" in prompt:
        _fail("correctness golden.prompt", "must be a bounded single line")
    max_tokens = _integer(golden["max_tokens"], "correctness golden.max_tokens", 2)
    if max_tokens > 1024:
        _fail("correctness golden.max_tokens", "must be <= 1024")

    if native_correctness.get("schema_version") != NATIVE_CORRECTNESS_VERSION:
        _fail(
            "native correctness report.schema_version",
            f"must be {NATIVE_CORRECTNESS_VERSION}",
        )
    if native_correctness.get("gate_id") != NATIVE_CORRECTNESS_GATE:
        _fail(
            "native correctness report.gate_id",
            f"must be {NATIVE_CORRECTNESS_GATE}",
        )
    if native_correctness.get("status") != "pass":
        _fail("native correctness report.status", "must be pass")
    native_bindings = _object(
        native_correctness.get("bindings"), "native correctness report.bindings"
    )
    required_native_bindings = {
        "candidate_git_revision",
        "candidate_git_status_sha256",
        "candidate_executable_sha256",
        "model_id",
        "model_revision",
        "config_sha256",
        "weights_sha256",
        "tokenizer_sha256",
    }
    missing_native_bindings = required_native_bindings - set(native_bindings)
    if missing_native_bindings:
        _fail(
            "native correctness report.bindings",
            f"missing fields: {sorted(missing_native_bindings)}",
        )
    if native_bindings["candidate_git_status_sha256"] != hashlib.sha256(b"").hexdigest():
        _fail(
            "native correctness report.bindings.candidate_git_status_sha256",
            "candidate tree was not clean",
        )
    _string(
        native_bindings["candidate_executable_sha256"],
        "native correctness report.bindings.candidate_executable_sha256",
        SHA256_RE,
    )

    manifest_golden = _object(manifest["golden"], "manifest.golden")
    if manifest_golden["generated_sha256"] != golden["expected_greedy_text_sha256"]:
        _fail(
            "manifest.golden.generated_sha256",
            "does not match the trusted E2E correctness golden",
        )
    if manifest_golden["provenance_sha256"] != native_correctness_report_sha256:
        _fail(
            "manifest.golden.provenance_sha256",
            "does not hash the submitted native correctness report",
        )
    if golden["correctness_report_sha256"] != native_correctness_report_sha256:
        _fail(
            "correctness golden.correctness_report_sha256",
            "does not hash the submitted native correctness report",
        )

    run_source = _object(run["source"], "run.json.source")
    source_cross_bindings = {
        "source_revision": (
            golden["source_revision"],
            run_source["git_commit"],
            "git_commit",
        ),
        "model_id": (golden["model_id"], run_source["model_id"], "model_id"),
        "model_revision": (
            golden["model_revision"],
            run_source["model_revision"],
            "model_revision",
        ),
    }
    for field, (trusted_value, run_value, run_field) in source_cross_bindings.items():
        if trusted_value != run_value:
            _fail(
                f"correctness golden.{field}",
                f"does not match run.json.source.{run_field}",
            )

    native_cross_bindings = {
        "candidate_git_revision": golden["source_revision"],
        "model_id": golden["model_id"],
        "model_revision": golden["model_revision"],
        "config_sha256": golden["config_sha256"],
        "weights_sha256": golden["weights_sha256"],
        "tokenizer_sha256": golden["tokenizer_aggregate_sha256"],
    }
    for field, expected in native_cross_bindings.items():
        if native_bindings[field] != expected:
            _fail(
                f"native correctness report.bindings.{field}",
                "does not match the trusted E2E correctness golden",
            )

    golden_profile = manifest_golden["request_profile"]
    request = _object(
        _object(manifest["requests"], "manifest.requests").get(golden_profile),
        f"manifest.requests.{golden_profile}",
    )
    request_cross_bindings = {
        "model": golden["model_id"],
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    for field, expected in request_cross_bindings.items():
        if request.get(field) != expected:
            _fail(
                f"manifest.requests.{golden_profile}.{field}",
                "does not match the trusted E2E correctness golden",
            )
    temperature = request.get("temperature")
    if isinstance(temperature, bool) or temperature not in {0, 0.0}:
        _fail(
            f"manifest.requests.{golden_profile}.temperature",
            "must select greedy generation for the trusted golden",
        )

    return {
        "correctness_gate_id": NATIVE_CORRECTNESS_GATE,
        "e2e_correctness_golden_sha256": correctness_golden_sha256,
        "generated_text_sha256": golden["expected_greedy_text_sha256"],
        "native_correctness_report_sha256": native_correctness_report_sha256,
    }


def _required(value: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in value:
        _fail(path, f"missing required field {key!r}")
    return value[key]


def _single_inspect_receipt(
    payload: tuple[bytes, str], label: str
) -> tuple[dict[str, Any], str]:
    raw, digest = payload
    value = _parse_json_value(raw, label)
    if not isinstance(value, list) or len(value) != 1:
        _fail(label, "must be a one-element Docker inspect array")
    return _object(value[0], f"{label}[0]"), digest


def _environment_map(value: Any, path: str) -> dict[str, str]:
    if not isinstance(value, list):
        _fail(path, "must be an array of NAME=value strings")
    result: dict[str, str] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or "=" not in raw:
            _fail(f"{path}[{index}]", "must be a NAME=value string")
        name, setting = raw.split("=", 1)
        if not name or "\x00" in name or name in result:
            _fail(f"{path}[{index}]", "has an empty, duplicate, or invalid name")
        result[name] = setting
    return result


def _validate_safe_environment(environment: Mapping[str, str], path: str) -> None:
    for name, expected in SOAK_RELEASE_ENVIRONMENT.items():
        if environment.get(name) != expected:
            _fail(
                path,
                f"{name} must equal the exact reviewed release-image value",
            )
    forbidden = sorted(
        name
        for name in environment
        if name not in SOAK_RELEASE_ENVIRONMENT
        and (
            name in FORBIDDEN_RUNTIME_ENVIRONMENT
            or name.startswith(FORBIDDEN_RUNTIME_ENVIRONMENT_PREFIXES)
        )
    )
    if forbidden:
        _fail(
            path,
            f"forbidden shell/loader/runtime environment overrides: {forbidden}",
        )


def _string_map(value: Any, path: str) -> dict[str, str]:
    result = _object(value, path)
    for key, setting in result.items():
        if not isinstance(key, str) or not key or not isinstance(setting, str):
            _fail(path, "must be an object with non-empty string keys and string values")
    return result


def _docker_timestamp_ns(value: Any, path: str) -> int:
    timestamp = _string(value, path)
    match = DOCKER_TIMESTAMP_RE.fullmatch(timestamp)
    if match is None:
        _fail(path, "must be a strict UTC RFC3339 timestamp with <= 9 fractional digits")
    parts = {key: match.group(key) for key in match.groupdict()}
    try:
        parsed = datetime.datetime(
            int(parts["year"]),
            int(parts["month"]),
            int(parts["day"]),
            int(parts["hour"]),
            int(parts["minute"]),
            int(parts["second"]),
            tzinfo=datetime.timezone.utc,
        )
    except ValueError as error:
        _fail(path, f"is not a valid Gregorian UTC timestamp: {error}")
    fraction = parts["fraction"] or ""
    nanoseconds = int(fraction.ljust(9, "0")) if fraction else 0
    days = parsed.date().toordinal() - 1
    seconds = (
        days * 86_400
        + parsed.hour * 3_600
        + parsed.minute * 60
        + parsed.second
    )
    return seconds * 1_000_000_000 + nanoseconds


def _compact_utc_timestamp_ns(value: str, path: str) -> int:
    match = COMPACT_UTC_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        _fail(path, "must be a strict compact UTC timestamp")
    parts = match.groupdict()
    expanded = (
        f"{parts['year']}-{parts['month']}-{parts['day']}T"
        f"{parts['hour']}:{parts['minute']}:{parts['second']}Z"
    )
    return _docker_timestamp_ns(expanded, path)


def _rootfs_layers(value: Mapping[str, Any], path: str) -> list[str]:
    rootfs = _object(_required(value, "RootFS", path), f"{path}.RootFS")
    if _required(rootfs, "Type", f"{path}.RootFS") != "layers":
        _fail(f"{path}.RootFS.Type", "must be layers")
    layers = _required(rootfs, "Layers", f"{path}.RootFS")
    if not isinstance(layers, list) or not layers:
        _fail(f"{path}.RootFS.Layers", "must be a non-empty array")
    for index, layer in enumerate(layers):
        _string(layer, f"{path}.RootFS.Layers[{index}]", IMAGE_ID_RE)
    return layers


def _validate_no_network_addresses(value: Mapping[str, Any], path: str) -> None:
    network = _object(
        _required(value, "NetworkSettings", path), f"{path}.NetworkSettings"
    )
    for field in (
        "Gateway",
        "IPAddress",
        "IPv6Gateway",
        "GlobalIPv6Address",
        "MacAddress",
    ):
        if _required(network, field, f"{path}.NetworkSettings") != "":
            _fail(f"{path}.NetworkSettings.{field}", "network address must be empty")
    for field in ("IPPrefixLen", "GlobalIPv6PrefixLen"):
        prefix = _required(network, field, f"{path}.NetworkSettings")
        if _integer(prefix, f"{path}.NetworkSettings.{field}") != 0:
            _fail(f"{path}.NetworkSettings.{field}", "network prefix must be zero")
    ports = _required(network, "Ports", f"{path}.NetworkSettings")
    if ports not in (None, {}):
        _fail(f"{path}.NetworkSettings.Ports", "network-none container has ports")
    networks = _object(
        _required(network, "Networks", f"{path}.NetworkSettings"),
        f"{path}.NetworkSettings.Networks",
    )
    if set(networks) != {"none"}:
        _fail(f"{path}.NetworkSettings.Networks", "must contain only network none")
    none = _object(networks["none"], f"{path}.NetworkSettings.Networks.none")
    for field in (
        "Gateway",
        "IPAddress",
        "IPv6Gateway",
        "GlobalIPv6Address",
        "MacAddress",
    ):
        if _required(none, field, f"{path}.NetworkSettings.Networks.none") != "":
            _fail(
                f"{path}.NetworkSettings.Networks.none.{field}",
                "network address must be empty",
            )
    for field in ("IPPrefixLen", "GlobalIPv6PrefixLen"):
        prefix = _required(none, field, f"{path}.NetworkSettings.Networks.none")
        if _integer(
            prefix, f"{path}.NetworkSettings.Networks.none.{field}"
        ) != 0:
            _fail(
                f"{path}.NetworkSettings.Networks.none.{field}",
                "network prefix must be zero",
            )


def _validate_container_receipt(
    value: dict[str, Any],
    path: str,
    *,
    container_id: str,
    container_name: str,
    test_image_id: str,
    expected_environment: Mapping[str, str],
    expected_labels: Mapping[str, str],
    expected_working_directory: str,
    gpu_uuid: str,
    post_run: bool,
) -> dict[str, int]:
    if _required(value, "Id", path) != container_id:
        _fail(f"{path}.Id", "does not match the launcher container ID")
    if _required(value, "Name", path) != f"/{container_name}":
        _fail(f"{path}.Name", "does not match the launcher container name")
    if _required(value, "Image", path) != test_image_id:
        _fail(f"{path}.Image", "does not match the inspected test-layer image")
    if _required(value, "Path", path) != SOAK_CONTAINER_PATH:
        _fail(f"{path}.Path", "does not equal the reviewed soak entrypoint path")
    if _required(value, "Args", path) != SOAK_CONTAINER_ARGS:
        _fail(f"{path}.Args", "does not equal the empty reviewed argument vector")
    created_ns = _docker_timestamp_ns(
        _required(value, "Created", path), f"{path}.Created"
    )
    if created_ns <= _docker_timestamp_ns(DOCKER_ZERO_TIMESTAMP, "Docker zero time"):
        _fail(f"{path}.Created", "must identify a real container creation time")

    config = _object(_required(value, "Config", path), f"{path}.Config")
    if _required(config, "Image", f"{path}.Config") != test_image_id:
        _fail(f"{path}.Config.Image", "must be the immutable test-layer image ID")
    if _required(config, "User", f"{path}.Config") != SOAK_USER:
        _fail(f"{path}.Config.User", f"must be {SOAK_USER}")
    if _required(config, "Entrypoint", f"{path}.Config") != SOAK_ENTRYPOINT:
        _fail(f"{path}.Config.Entrypoint", "does not equal the soak entrypoint")
    if _required(config, "Cmd", f"{path}.Config") != SOAK_CMD:
        _fail(f"{path}.Config.Cmd", "does not equal the empty soak command")
    if (
        _required(config, "WorkingDir", f"{path}.Config")
        != expected_working_directory
    ):
        _fail(
            f"{path}.Config.WorkingDir",
            "does not preserve the inspected test-layer working directory",
        )
    healthcheck = _exact(
        _required(config, "Healthcheck", f"{path}.Config"),
        {"Test"},
        f"{path}.Config.Healthcheck",
    )
    if healthcheck != SOAK_HEALTHCHECK:
        _fail(
            f"{path}.Config.Healthcheck",
            "must be the exact --no-healthcheck result",
        )
    labels = _string_map(
        _required(config, "Labels", f"{path}.Config"),
        f"{path}.Config.Labels",
    )
    if labels != dict(expected_labels):
        _fail(f"{path}.Config.Labels", "does not equal the inspected test-layer labels")
    environment = _environment_map(
        _required(config, "Env", f"{path}.Config"), f"{path}.Config.Env"
    )
    _validate_safe_environment(environment, f"{path}.Config.Env")
    if environment != dict(expected_environment):
        _fail(f"{path}.Config.Env", "does not equal the image plus bound soak environment")

    host = _object(_required(value, "HostConfig", path), f"{path}.HostConfig")
    expected_host_scalars = {
        "NetworkMode": "none",
        "PidMode": "host",
        "IpcMode": "private",
        "UTSMode": "",
        "UsernsMode": "",
        "CgroupnsMode": "private",
        "Runtime": "runc",
        "ReadonlyRootfs": True,
        "PidsLimit": 8192,
        "Privileged": False,
        "AutoRemove": False,
        "PublishAllPorts": False,
    }
    for field, expected in expected_host_scalars.items():
        actual = _required(host, field, f"{path}.HostConfig")
        if (
            (isinstance(expected, bool) and actual is not expected)
            or (
                isinstance(expected, int)
                and not isinstance(expected, bool)
                and (
                    not isinstance(actual, int)
                    or isinstance(actual, bool)
                    or actual != expected
                )
            )
            or (
                not isinstance(expected, (bool, int))
                and actual != expected
            )
        ):
            _fail(f"{path}.HostConfig.{field}", f"must be {expected!r}")
    expected_absent_host_fields = {
        "Binds": None,
        "DeviceCgroupRules": None,
        "Devices": [],
        "ExtraHosts": None,
        "GroupAdd": None,
        "Links": None,
        "Sysctls": None,
        "VolumesFrom": None,
    }
    for field, expected in expected_absent_host_fields.items():
        if _required(host, field, f"{path}.HostConfig") != expected:
            _fail(
                f"{path}.HostConfig.{field}",
                "must equal the exact safe empty/default value",
            )
    restart = _exact(
        _required(host, "RestartPolicy", f"{path}.HostConfig"),
        {"Name", "MaximumRetryCount"},
        f"{path}.HostConfig.RestartPolicy",
    )
    if restart["Name"] != "no" or _integer(
        restart["MaximumRetryCount"],
        f"{path}.HostConfig.RestartPolicy.MaximumRetryCount",
    ) != 0:
        _fail(f"{path}.HostConfig.RestartPolicy", "must disable restart")
    if _required(host, "CapDrop", f"{path}.HostConfig") != ["ALL"]:
        _fail(f"{path}.HostConfig.CapDrop", "must drop ALL capabilities")
    if _required(host, "CapAdd", f"{path}.HostConfig") not in (None, []):
        _fail(f"{path}.HostConfig.CapAdd", "must not restore capabilities")
    if _required(host, "PortBindings", f"{path}.HostConfig") not in (None, {}):
        _fail(f"{path}.HostConfig.PortBindings", "must not publish ports")
    security = _required(host, "SecurityOpt", f"{path}.HostConfig")
    if security not in (["no-new-privileges"], ["no-new-privileges:true"]):
        _fail(
            f"{path}.HostConfig.SecurityOpt",
            "must contain only enabled no-new-privileges",
        )
    tmpfs = _object(_required(host, "Tmpfs", f"{path}.HostConfig"), f"{path}.HostConfig.Tmpfs")
    if set(tmpfs) != {"/tmp"} or not isinstance(tmpfs["/tmp"], str):
        _fail(f"{path}.HostConfig.Tmpfs", "must contain only the bounded /tmp tmpfs")
    tmpfs_options = tmpfs["/tmp"].split(",")
    if (
        len(tmpfs_options) != len(SOAK_TMPFS_OPTIONS)
        or set(tmpfs_options) != SOAK_TMPFS_OPTIONS
    ):
        _fail(f"{path}.HostConfig.Tmpfs./tmp", "tmpfs options differ from the soak contract")

    requests = _required(host, "DeviceRequests", f"{path}.HostConfig")
    if not isinstance(requests, list) or len(requests) != 1:
        _fail(f"{path}.HostConfig.DeviceRequests", "exactly one GPU request is required")
    request = _exact(
        requests[0],
        {"Driver", "Count", "DeviceIDs", "Capabilities", "Options"},
        f"{path}.HostConfig.DeviceRequests[0]",
    )
    if request["Driver"] != "":
        _fail(f"{path}.HostConfig.DeviceRequests[0].Driver", "must use Docker's exact GPU request")
    if _integer(
        request["Count"], f"{path}.HostConfig.DeviceRequests[0].Count"
    ) != 0:
        _fail(f"{path}.HostConfig.DeviceRequests[0].Count", "must be exact-ID selected")
    if request["DeviceIDs"] != [gpu_uuid]:
        _fail(f"{path}.HostConfig.DeviceRequests[0].DeviceIDs", "wrong GPU UUID")
    if request["Capabilities"] != [["gpu"]] or request["Options"] != {}:
        _fail(f"{path}.HostConfig.DeviceRequests[0]", "GPU request contract mismatch")

    mounts = _required(value, "Mounts", path)
    if not isinstance(mounts, list) or len(mounts) != 3:
        _fail(f"{path}.Mounts", "exactly three bind mounts are required")
    expected_mounts = {
        SOAK_MODEL_DESTINATION: {
            "rw": False,
            "mode": "",
            "propagation": "rprivate",
        },
        SOAK_MANIFEST_DESTINATION: {
            "rw": False,
            "mode": "",
            "propagation": "rprivate",
        },
        SOAK_EVIDENCE_DESTINATION: {
            "rw": True,
            "mode": "",
            "propagation": "rprivate",
        },
    }
    observed_mounts: dict[str, dict[str, Any]] = {}
    for index, raw_mount in enumerate(mounts):
        mount_path = f"{path}.Mounts[{index}]"
        mount = _object(raw_mount, mount_path)
        if _required(mount, "Type", mount_path) != "bind":
            _fail(f"{mount_path}.Type", "must be bind")
        source = _string(_required(mount, "Source", mount_path), f"{mount_path}.Source")
        if not source.startswith("/") or "\x00" in source:
            _fail(f"{mount_path}.Source", "must be an absolute host path")
        destination = _string(
            _required(mount, "Destination", mount_path), f"{mount_path}.Destination"
        )
        if destination in observed_mounts:
            _fail(f"{mount_path}.Destination", "duplicate mount destination")
        writable = _required(mount, "RW", mount_path)
        if not isinstance(writable, bool):
            _fail(f"{mount_path}.RW", "must be boolean")
        mode = _required(mount, "Mode", mount_path)
        if mode != "":
            _fail(f"{mount_path}.Mode", "must be the exact empty bind mode")
        propagation = _required(mount, "Propagation", mount_path)
        if propagation != "rprivate":
            _fail(
                f"{mount_path}.Propagation",
                "must be the exact non-propagating rprivate bind mode",
            )
        observed_mounts[destination] = {
            "rw": writable,
            "mode": mode,
            "propagation": propagation,
        }
    if observed_mounts != expected_mounts:
        _fail(f"{path}.Mounts", "model/manifest/evidence mount policy mismatch")

    _validate_no_network_addresses(value, path)
    restart_count = _required(value, "RestartCount", path)
    if restart_count != 0 or isinstance(restart_count, bool):
        _fail(f"{path}.RestartCount", "must be zero")
    state = _object(_required(value, "State", path), f"{path}.State")
    expected_state = {
        "Status": "exited" if post_run else "created",
        "Running": False,
        "Paused": False,
        "Restarting": False,
        "OOMKilled": False,
        "Dead": False,
        "Error": "",
        "Pid": 0,
    }
    for field, expected in expected_state.items():
        actual = _required(state, field, f"{path}.State")
        if (
            (isinstance(expected, bool) and actual is not expected)
            or (not isinstance(expected, bool) and actual != expected)
        ):
            _fail(f"{path}.State.{field}", f"must be {expected!r}")
    if _integer(
        _required(state, "ExitCode", f"{path}.State"),
        f"{path}.State.ExitCode",
    ) != 0:
        _fail(f"{path}.State.ExitCode", "must be zero")
    started_at = _required(state, "StartedAt", f"{path}.State")
    finished_at = _required(state, "FinishedAt", f"{path}.State")
    started_ns = _docker_timestamp_ns(started_at, f"{path}.State.StartedAt")
    finished_ns = _docker_timestamp_ns(finished_at, f"{path}.State.FinishedAt")
    if not post_run:
        if started_at != DOCKER_ZERO_TIMESTAMP or finished_at != DOCKER_ZERO_TIMESTAMP:
            _fail(
                f"{path}.State",
                "created receipt must retain exact zero start/finish timestamps",
            )
        return {
            "created_ns": created_ns,
            "started_ns": started_ns,
            "finished_ns": finished_ns,
            "elapsed_ns": 0,
        }
    if started_ns < created_ns:
        _fail(f"{path}.State.StartedAt", "must not precede container creation")
    if finished_ns <= started_ns:
        _fail(f"{path}.State.FinishedAt", "must be later than StartedAt")
    elapsed_ns = finished_ns - started_ns
    minimum_ns = MINIMUM_SOAK_RUNTIME_SECONDS * 1_000_000_000
    if elapsed_ns < minimum_ns:
        _fail(
            f"{path}.State",
            f"container runtime must be at least {MINIMUM_SOAK_RUNTIME_SECONDS} seconds",
        )
    return {
        "created_ns": created_ns,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "elapsed_ns": elapsed_ns,
    }


def _validate_runtime_receipts(
    runtime_receipts_directory: Path | str | None,
    run: Mapping[str, Any],
    trusted_correctness: Mapping[str, str],
    *,
    run_json_sha256: str,
    events_jsonl_sha256: str,
) -> tuple[dict[str, str], dict[str, int]]:
    if runtime_receipts_directory is None:
        _fail(
            "--runtime-receipts-directory",
            "the seven remote launcher runtime receipts are required",
        )
    directory = Path(runtime_receipts_directory)
    receipt_payloads = _load_runtime_receipt_payloads(directory)
    launcher_bytes, launcher_sha256 = receipt_payloads["launcher-receipt.json"]
    launcher_value = _parse_json_value(launcher_bytes, "launcher-receipt.json")
    launcher = _object(launcher_value, "launcher-receipt.json")
    launcher = _exact(
        launcher,
        {"schema_version", "host", "source", "evidence", "images", "container"},
        "launcher-receipt.json",
    )
    if launcher["schema_version"] != LAUNCHER_RECEIPT_VERSION:
        _fail(
            "launcher-receipt.json.schema_version",
            f"must be {LAUNCHER_RECEIPT_VERSION}",
        )
    host = _exact(
        launcher["host"],
        {
            "hostname",
            "gpu_name",
            "gpu_uuid",
            "compute_capability",
            "memory_total_mib",
            "driver_version",
        },
        "launcher-receipt.json.host",
    )
    expected_host: dict[str, Any] = {
        "hostname": DESIGNATED_HOSTNAME,
        **DESIGNATED_GPU,
    }
    if host != expected_host:
        _fail("launcher-receipt.json.host", "is not the designated server-4096 GPU")
    _integer(host["memory_total_mib"], "launcher-receipt.json.host.memory_total_mib", 1)
    for key in (
        "hostname",
        "gpu_name",
        "gpu_uuid",
        "compute_capability",
        "driver_version",
    ):
        _string(host[key], f"launcher-receipt.json.host.{key}")

    host_gpu_bytes, host_gpu_sha256 = receipt_payloads["host-gpu.csv"]
    expected_host_gpu = (
        f"{host['gpu_name']}, {host['gpu_uuid']}, {host['compute_capability']}, "
        f"{host['memory_total_mib']}, {host['driver_version']}\n"
    ).encode("ascii")
    if host_gpu_bytes != expected_host_gpu:
        _fail("host-gpu.csv", "must be the exact designated single GPU row")
    closure_bytes, closure_sha256 = receipt_payloads[
        "release-runtime-closure.tsv"
    ]
    if _validate_runtime_closure_receipt(closure_bytes) != closure_sha256:
        _fail("release-runtime-closure.tsv", "closure receipt digest changed while read")

    source = _exact(
        launcher["source"],
        {
            "git_revision",
            "source_archive_sha256",
            "release_binary_sha256",
            "model_tree_sha256",
            "manifest_sha256",
            "correctness_golden_sha256",
            "native_correctness_report_sha256",
        },
        "launcher-receipt.json.source",
    )
    run_source = _object(run["source"], "run.json.source")
    expected_source = {
        "git_revision": run_source["git_commit"],
        "source_archive_sha256": run_source["source_archive_sha256"],
        "release_binary_sha256": run_source["binary_sha256"],
        "model_tree_sha256": run_source["model_sha256"],
        "manifest_sha256": run["manifest_sha256"],
        "correctness_golden_sha256": trusted_correctness[
            "e2e_correctness_golden_sha256"
        ],
        "native_correctness_report_sha256": trusted_correctness[
            "native_correctness_report_sha256"
        ],
    }
    if source != expected_source:
        _fail("launcher-receipt.json.source", "does not bind the checked soak inputs")
    for key, value in source.items():
        _string(
            value,
            f"launcher-receipt.json.source.{key}",
            GIT_RE if key == "git_revision" else SHA256_RE,
        )

    evidence = _exact(
        launcher["evidence"],
        {
            "run_json_sha256",
            "events_jsonl_sha256",
            "release_runtime_closure_sha256",
        },
        "launcher-receipt.json.evidence",
    )
    expected_evidence = {
        "run_json_sha256": run_json_sha256,
        "events_jsonl_sha256": events_jsonl_sha256,
        "release_runtime_closure_sha256": closure_sha256,
    }
    if evidence != expected_evidence:
        _fail(
            "launcher-receipt.json.evidence",
            "does not hash the exact exported run.json/events.jsonl bytes and runtime closure",
        )
    for key, value in evidence.items():
        _string(value, f"launcher-receipt.json.evidence.{key}", SHA256_RE)

    images = _exact(
        launcher["images"],
        {"release_image_id", "test_layer_image_id"},
        "launcher-receipt.json.images",
    )
    release_image_id = _string(
        images["release_image_id"],
        "launcher-receipt.json.images.release_image_id",
        IMAGE_ID_RE,
    )
    test_image_id = _string(
        images["test_layer_image_id"],
        "launcher-receipt.json.images.test_layer_image_id",
        IMAGE_ID_RE,
    )
    if release_image_id != f"sha256:{run_source['image_sha256']}":
        _fail(
            "launcher-receipt.json.images.release_image_id",
            "does not match run.json.source.image_sha256",
        )
    if test_image_id == release_image_id:
        _fail("launcher-receipt.json.images.test_layer_image_id", "must be a derivative image")

    container = _exact(
        launcher["container"],
        {"id", "name", "exit_code"},
        "launcher-receipt.json.container",
    )
    container_id = _string(
        container["id"], "launcher-receipt.json.container.id", CONTAINER_ID_RE
    )
    container_name = _string(container["name"], "launcher-receipt.json.container.name")
    expected_name = re.compile(
        rf"riley-soak-{re.escape(run_source['git_commit'][:12])}-"
        rf"(?P<stamp>[0-9]{{8}}T[0-9]{{6}}Z)"
    )
    container_name_match = expected_name.fullmatch(container_name)
    if container_name_match is None:
        _fail("launcher-receipt.json.container.name", "does not bind the source prefix and UTC run stamp")
    container_name_ns = _compact_utc_timestamp_ns(
        container_name_match.group("stamp"),
        "launcher-receipt.json.container.name",
    )
    if _integer(container["exit_code"], "launcher-receipt.json.container.exit_code") != 0:
        _fail("launcher-receipt.json.container.exit_code", "must be zero")

    release_image, release_image_sha256 = _single_inspect_receipt(
        receipt_payloads["release-image-inspect.json"],
        "release-image-inspect.json",
    )
    if _required(release_image, "Id", "release-image-inspect.json[0]") != release_image_id:
        _fail("release-image-inspect.json[0].Id", "does not match the release image ID")
    if _required(release_image, "Os", "release-image-inspect.json[0]") != "linux":
        _fail("release-image-inspect.json[0].Os", "must be linux")
    if _required(release_image, "Architecture", "release-image-inspect.json[0]") != "amd64":
        _fail("release-image-inspect.json[0].Architecture", "must be amd64")
    release_config = _object(
        _required(release_image, "Config", "release-image-inspect.json[0]"),
        "release-image-inspect.json[0].Config",
    )
    if _required(release_config, "User", "release-image-inspect.json[0].Config") != SOAK_USER:
        _fail("release-image-inspect.json[0].Config.User", f"must be {SOAK_USER}")
    release_environment = _environment_map(
        _required(release_config, "Env", "release-image-inspect.json[0].Config"),
        "release-image-inspect.json[0].Config.Env",
    )
    _validate_safe_environment(
        release_environment, "release-image-inspect.json[0].Config.Env"
    )
    release_labels_value = release_config.get("Labels")
    release_labels = (
        {}
        if release_labels_value is None
        else _string_map(
            release_labels_value, "release-image-inspect.json[0].Config.Labels"
        )
    )
    release_working_directory = _required(
        release_config, "WorkingDir", "release-image-inspect.json[0].Config"
    )
    if not isinstance(release_working_directory, str):
        _fail("release-image-inspect.json[0].Config.WorkingDir", "must be a string")
    release_layers = _rootfs_layers(release_image, "release-image-inspect.json[0]")

    test_image, test_image_sha256 = _single_inspect_receipt(
        receipt_payloads["test-layer-image-inspect.json"],
        "test-layer-image-inspect.json",
    )
    if _required(test_image, "Id", "test-layer-image-inspect.json[0]") != test_image_id:
        _fail("test-layer-image-inspect.json[0].Id", "does not match the test image ID")
    if _required(test_image, "Os", "test-layer-image-inspect.json[0]") != "linux":
        _fail("test-layer-image-inspect.json[0].Os", "must be linux")
    if _required(test_image, "Architecture", "test-layer-image-inspect.json[0]") != "amd64":
        _fail("test-layer-image-inspect.json[0].Architecture", "must be amd64")
    test_config = _object(
        _required(test_image, "Config", "test-layer-image-inspect.json[0]"),
        "test-layer-image-inspect.json[0].Config",
    )
    if _required(test_config, "User", "test-layer-image-inspect.json[0].Config") != SOAK_USER:
        _fail("test-layer-image-inspect.json[0].Config.User", f"must be {SOAK_USER}")
    if _required(test_config, "Entrypoint", "test-layer-image-inspect.json[0].Config") != SOAK_ENTRYPOINT:
        _fail("test-layer-image-inspect.json[0].Config.Entrypoint", "wrong soak entrypoint")
    if _required(test_config, "Cmd", "test-layer-image-inspect.json[0].Config") != SOAK_CMD:
        _fail("test-layer-image-inspect.json[0].Config.Cmd", "wrong soak command")
    if (
        _required(test_config, "WorkingDir", "test-layer-image-inspect.json[0].Config")
        != release_working_directory
    ):
        _fail(
            "test-layer-image-inspect.json[0].Config.WorkingDir",
            "must preserve the release image working directory",
        )
    labels = _string_map(
        _required(test_config, "Labels", "test-layer-image-inspect.json[0].Config"),
        "test-layer-image-inspect.json[0].Config.Labels",
    )
    expected_labels = {
        **release_labels,
        SOAK_IMAGE_LABELS["release_image_id"]: release_image_id,
        SOAK_IMAGE_LABELS["source_revision"]: run_source["git_commit"],
        SOAK_IMAGE_LABELS["source_archive_sha256"]: run_source[
            "source_archive_sha256"
        ],
        SOAK_IMAGE_LABELS["release_binary_sha256"]: run_source["binary_sha256"],
    }
    if labels != expected_labels:
        _fail(
            "test-layer-image-inspect.json[0].Config.Labels",
            "must equal inherited labels plus the exact soak provenance labels",
        )
    test_layers = _rootfs_layers(test_image, "test-layer-image-inspect.json[0]")
    if len(test_layers) <= len(release_layers) or test_layers[: len(release_layers)] != release_layers:
        _fail(
            "test-layer-image-inspect.json[0].RootFS.Layers",
            "must strictly extend the exact release image RootFS",
        )

    image_environment = _environment_map(
        _required(test_config, "Env", "test-layer-image-inspect.json[0].Config"),
        "test-layer-image-inspect.json[0].Config.Env",
    )
    _validate_safe_environment(
        image_environment, "test-layer-image-inspect.json[0].Config.Env"
    )
    expected_image_environment = dict(release_environment)
    expected_image_environment.update(SOAK_IMAGE_ENVIRONMENT_OVERRIDES)
    if image_environment != expected_image_environment:
        _fail(
            "test-layer-image-inspect.json[0].Config.Env",
            "must equal the release environment plus exact soak-layer overrides",
        )
    expected_environment = dict(image_environment)
    expected_environment.update(
        {
            "RILEY_SOAK_MANIFEST": SOAK_MANIFEST_DESTINATION,
            "RILEY_SOAK_OUTPUT": "/evidence/run",
            "RILEY_SOURCE_REVISION": run_source["git_commit"],
            "RILEY_SOURCE_ARCHIVE_SHA256": run_source[
                "source_archive_sha256"
            ],
            "RILEY_BINARY_SHA256": run_source["binary_sha256"],
            "RILEY_IMAGE_SHA256": run_source["image_sha256"],
            "RILEY_MODEL_SHA256": run_source["model_sha256"],
            "RILEY_MODEL_ID": run_source["model_id"],
            "RILEY_MODEL_REVISION": run_source["model_revision"],
            "RILEY_SOAK_FINAL_METRICS_JSON": "/evidence/final-metrics.json",
            "RILEY_SOAK_BINARY": "/opt/riley/bin/riley",
            "RILEY_SOAK_MODEL_PATH": SOAK_MODEL_DESTINATION,
            "RILEY_SOAK_BIND": "127.0.0.1:18080",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "ALL_PROXY": "",
            "FTP_PROXY": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "NO_PROXY": "",
            "all_proxy": "",
            "ftp_proxy": "",
            "http_proxy": "",
            "https_proxy": "",
            "no_proxy": "",
        }
    )

    pre, pre_sha256 = _single_inspect_receipt(
        receipt_payloads["container-inspect-pre.json"],
        "container-inspect-pre.json",
    )
    post, post_sha256 = _single_inspect_receipt(
        receipt_payloads["container-inspect-post.json"],
        "container-inspect-post.json",
    )
    pre_timing = _validate_container_receipt(
        pre,
        "container-inspect-pre.json[0]",
        container_id=container_id,
        container_name=container_name,
        test_image_id=test_image_id,
        expected_environment=expected_environment,
        expected_labels=expected_labels,
        expected_working_directory=release_working_directory,
        gpu_uuid=host["gpu_uuid"],
        post_run=False,
    )
    post_timing = _validate_container_receipt(
        post,
        "container-inspect-post.json[0]",
        container_id=container_id,
        container_name=container_name,
        test_image_id=test_image_id,
        expected_environment=expected_environment,
        expected_labels=expected_labels,
        expected_working_directory=release_working_directory,
        gpu_uuid=host["gpu_uuid"],
        post_run=True,
    )
    for field in (
        "Id",
        "Name",
        "Image",
        "Path",
        "Args",
        "Created",
        "Config",
        "HostConfig",
        "Mounts",
    ):
        if _required(pre, field, "container-inspect-pre.json[0]") != _required(
            post, field, "container-inspect-post.json[0]"
        ):
            _fail(
                f"container-inspect-post.json[0].{field}",
                "immutable container configuration differs from the pre-run receipt",
            )

    runtime_provenance = {
        "host_gpu_sha256": host_gpu_sha256,
        "launcher_receipt_sha256": launcher_sha256,
        "release_image_inspect_sha256": release_image_sha256,
        "test_layer_image_inspect_sha256": test_image_sha256,
        "container_inspect_pre_sha256": pre_sha256,
        "container_inspect_post_sha256": post_sha256,
        "release_runtime_closure_sha256": closure_sha256,
        "run_json_sha256": evidence["run_json_sha256"],
        "events_jsonl_sha256": evidence["events_jsonl_sha256"],
        "hostname": host["hostname"],
        "gpu_uuid": host["gpu_uuid"],
        "release_image_id": release_image_id,
        "test_layer_image_id": test_image_id,
        "container_id": container_id,
        "container_name": container_name,
    }
    if pre_timing["created_ns"] != post_timing["created_ns"]:
        _fail(
            "container-inspect-post.json[0].Created",
            "does not equal the pre-run container creation timestamp",
        )
    container_name_lead_ns = pre_timing["created_ns"] - container_name_ns
    maximum_name_lead_ns = MAXIMUM_CONTAINER_NAME_LEAD_SECONDS * 1_000_000_000
    if not 0 <= container_name_lead_ns <= maximum_name_lead_ns:
        _fail(
            "launcher-receipt.json.container.name",
            "timestamp must precede Docker Created by at most "
            f"{MAXIMUM_CONTAINER_NAME_LEAD_SECONDS} seconds",
        )
    run_started_ns = _docker_timestamp_ns(
        run["started_at_utc"], "run.json.started_at_utc"
    )
    if not post_timing["started_ns"] <= run_started_ns <= post_timing["finished_ns"]:
        _fail(
            "run.json.started_at_utc",
            "must fall within the validated Docker StartedAt..FinishedAt lifecycle",
        )
    return runtime_provenance, post_timing


def _validate_sample(event: dict[str, Any], path: str) -> None:
    process = _exact(event["process"], {"pid", "rss_bytes", "hwm_bytes", "fd_count", "thread_count", "children"}, f"{path}.process")
    _integer(process["pid"], f"{path}.process.pid", 0 if event["scenario_id"] is None else 1)
    for key in ("rss_bytes", "hwm_bytes", "fd_count", "thread_count"):
        _integer(process[key], f"{path}.process.{key}")
    if not isinstance(process["children"], list):
        _fail(f"{path}.process.children", "must be an array")
    for index, raw in enumerate(process["children"]):
        child = _exact(raw, {"pid", "comm", "executable"}, f"{path}.process.children[{index}]")
        _integer(child["pid"], f"{path}.process.children[{index}].pid", 1)
        _string(child["comm"], f"{path}.process.children[{index}].comm")
        _string(child["executable"], f"{path}.process.children[{index}].executable")
    gpu = _exact(event["gpu"], {"vram_bytes"}, f"{path}.gpu")
    _integer(gpu["vram_bytes"], f"{path}.gpu.vram_bytes")
    metrics = _exact(
        event["metrics"],
        {"active_requests", "waiting_requests", "kv_allocated_blocks", "allocation", "batch_shapes", "counters"},
        f"{path}.metrics",
    )
    for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"):
        _integer(metrics[key], f"{path}.metrics.{key}")
    allocation = _exact(metrics["allocation"], {"device_live_count", "device_live_bytes", "pinned_live_count", "pinned_live_bytes"}, f"{path}.metrics.allocation")
    for key, value in allocation.items():
        _integer(value, f"{path}.metrics.allocation.{key}")
    _validate_batch_shapes(metrics["batch_shapes"], f"{path}.metrics.batch_shapes")
    counters = _exact(metrics["counters"], {"cancellations", "disconnects", "overloads", "dropped_observations"}, f"{path}.metrics.counters")
    for key, value in counters.items():
        _integer(value, f"{path}.metrics.counters.{key}")
    if not isinstance(event["sample_dropped"], bool):
        _fail(f"{path}.sample_dropped", "must be boolean")


def _validate_batch_shapes(value: Any, path: str) -> None:
    shapes = _exact(value, {"metrics_degraded", "last", "bucket_count", "buckets"}, path)
    if not isinstance(shapes["metrics_degraded"], bool):
        _fail(f"{path}.metrics_degraded", "must be boolean")
    bucket_count = _integer(shapes["bucket_count"], f"{path}.bucket_count")
    if bucket_count > 10:
        _fail(f"{path}.bucket_count", "must not exceed 10")
    buckets = shapes["buckets"]
    if not isinstance(buckets, list) or len(buckets) != 10:
        _fail(f"{path}.buckets", "must contain exactly 10 fixed-capacity entries")
    previous_dense_rows = 0
    for index, raw in enumerate(buckets):
        bucket_path = f"{path}.buckets[{index}]"
        bucket = _exact(raw, {
            "dense_rows", "hit_count", "latency_sample_count", "gpu_execution_ns_total",
            "gpu_execution_ns_average", "gpu_execution_ns_maximum", "gpu_execution_ns_last",
        }, bucket_path)
        parsed = {name: _integer(item, f"{bucket_path}.{name}") for name, item in bucket.items()}
        if index < bucket_count:
            if parsed["dense_rows"] <= previous_dense_rows:
                _fail(f"{bucket_path}.dense_rows", "prepared buckets must be positive and strictly increasing")
            previous_dense_rows = parsed["dense_rows"]
            samples = parsed["latency_sample_count"]
            if parsed["hit_count"] != samples:
                _fail(bucket_path, "committed hit and latency sample counts must match")
            expected_average = parsed["gpu_execution_ns_total"] // samples if samples else 0
            if parsed["gpu_execution_ns_average"] != expected_average:
                _fail(f"{bucket_path}.gpu_execution_ns_average", "must equal total divided by sample count")
        elif any(parsed.values()):
            _fail(bucket_path, "unused fixed-capacity entries must be zeroed")
    last = shapes["last"]
    if last is None:
        if any(buckets[index]["hit_count"] for index in range(bucket_count)):
            _fail(f"{path}.last", "must exist after the first committed bucket hit")
        return
    observation = _exact(last, {"active_rows", "selected_dense_rows", "padding_rows"}, f"{path}.last")
    active = _integer(observation["active_rows"], f"{path}.last.active_rows", 1)
    selected = _integer(observation["selected_dense_rows"], f"{path}.last.selected_dense_rows", 1)
    padding = _integer(observation["padding_rows"], f"{path}.last.padding_rows")
    if active + padding != selected:
        _fail(f"{path}.last", "active_rows + padding_rows must equal selected_dense_rows")
    if not any(buckets[index]["dense_rows"] == selected and buckets[index]["hit_count"] > 0 for index in range(bucket_count)):
        _fail(f"{path}.last.selected_dense_rows", "must reference a committed prepared bucket")


def _validate_events(
    rows: list[dict[str, Any]],
    binding_sha: str,
    manifest_scenarios: Sequence[Mapping[str, Any]],
    manifest_requests: Mapping[str, Any],
) -> None:
    common = {"schema_version", "sequence", "monotonic_ns", "kind", "scenario_id", "binding_sha256"}
    extras = {
        "run_start": set(), "scenario_start": {"execution_completion"},
        "sample": {"process", "gpu", "metrics", "sample_dropped"},
        "request": {
            "request_id", "request_profile", "client_action", "request_stream",
            "curl_exit_code", "request_body_sha256", "response_body_sha256",
            "response_bytes", "outcome", "http_status", "latency_ms",
            "generated_sha256",
        },
        "restart": {"graceful", "exit_code", "elapsed_ms", "before_generated_sha256", "after_generated_sha256"},
        "scenario_end": {"status"}, "failure": {"stage", "message"}, "run_end": {"status"},
    }
    scenario_order = [scenario["id"] for scenario in manifest_scenarios]
    scenarios = {scenario["id"]: scenario for scenario in manifest_scenarios}
    active_scenario: str | None = None
    completed_scenarios = 0
    previous_time = -1
    for index, event in enumerate(rows, 1):
        path = f"events[{index}]"
        kind = event.get("kind")
        if kind not in extras:
            _fail(f"{path}.kind", "is not a closed v1 event kind")
        _exact(event, common | extras[kind], path)
        if event["schema_version"] != EVENT_VERSION:
            _fail(f"{path}.schema_version", f"must be {EVENT_VERSION}")
        if event["sequence"] != index:
            _fail(f"{path}.sequence", f"must be contiguous value {index}")
        monotonic_ns = _integer(event["monotonic_ns"], f"{path}.monotonic_ns")
        if monotonic_ns <= previous_time:
            _fail(f"{path}.monotonic_ns", "must be strictly increasing")
        previous_time = monotonic_ns
        if event["binding_sha256"] != binding_sha:
            _fail(f"{path}.binding_sha256", "does not match run binding")
        scenario_id = event["scenario_id"]
        if scenario_id is not None:
            _string(scenario_id, f"{path}.scenario_id")
            if scenario_id not in scenarios:
                _fail(f"{path}.scenario_id", "is absent from manifest")
        if kind not in {"run_start", "run_end", "sample", "failure"} and scenario_id is None:
            _fail(f"{path}.scenario_id", "must identify a scenario")
        if kind in {"run_start", "run_end"} and scenario_id is not None:
            _fail(f"{path}.scenario_id", "run boundary events must use null")
        if kind == "run_start":
            if index != 1 or active_scenario is not None or completed_scenarios != 0:
                _fail(path, "run_start must be the unique first event")
        elif kind == "scenario_start":
            if active_scenario is not None:
                _fail(path, f"overlaps active scenario {active_scenario}")
            if completed_scenarios >= len(scenario_order):
                _fail(path, "starts after the manifest scenario inventory completed")
            expected_scenario = scenario_order[completed_scenarios]
            if scenario_id != expected_scenario:
                _fail(
                    f"{path}.scenario_id",
                    f"must follow manifest order; expected {expected_scenario}",
                )
            active_scenario = scenario_id
        elif kind == "scenario_end":
            if active_scenario is None or scenario_id != active_scenario:
                _fail(path, "does not close the active manifest scenario")
            active_scenario = None
            completed_scenarios += 1
        elif scenario_id is not None and scenario_id != active_scenario:
            _fail(path, "must occur inside its non-overlapping scenario interval")
        elif scenario_id is None and kind == "sample":
            if active_scenario is not None or completed_scenarios != len(scenario_order):
                _fail(path, "global sample must follow all manifest scenarios")
        elif kind == "run_end":
            if active_scenario is not None or completed_scenarios != len(scenario_order):
                _fail(path, "run_end requires every manifest scenario to finish in order")
        if kind == "sample":
            _validate_sample(event, path)
        elif kind == "request":
            _string(event["request_id"], f"{path}.request_id")
            profile = _string(event["request_profile"], f"{path}.request_profile")
            scenario = scenarios[scenario_id]
            allowed_profiles = {scenario["request_profile"]}
            if "secondary_request_profile" in scenario:
                allowed_profiles.add(scenario["secondary_request_profile"])
            if profile not in allowed_profiles:
                _fail(f"{path}.request_profile", "does not belong to the manifest scenario")
            action = _string(event["client_action"], f"{path}.client_action")
            expected_actions = {"normal"}
            if scenario["kind"] == "invalid":
                expected_actions = {"invalid"}
            elif scenario["kind"] == "overload":
                expected_actions = {"overload"}
            elif scenario["kind"] == "cancellation-disconnect":
                expected_actions = {"cancel", "disconnect"}
            if action not in expected_actions:
                _fail(
                    f"{path}.client_action",
                    f"does not match scenario kind {scenario['kind']}",
                )
            if not isinstance(event["request_stream"], bool):
                _fail(f"{path}.request_stream", "must be boolean")
            request_stream = event["request_stream"]
            if request_stream != (action == "disconnect"):
                _fail(
                    f"{path}.request_stream",
                    "must be true exactly for the disconnect client action",
                )
            curl_exit_code = _integer(
                event["curl_exit_code"], f"{path}.curl_exit_code"
            )
            if curl_exit_code > 255:
                _fail(f"{path}.curl_exit_code", "must be <= 255")
            request_body_sha256 = _string(
                event["request_body_sha256"],
                f"{path}.request_body_sha256",
                SHA256_RE,
            )
            expected_request = dict(
                _object(manifest_requests[profile], f"manifest.requests.{profile}")
            )
            if "prompt_repeat" in expected_request:
                repeat = _integer(
                    expected_request.pop("prompt_repeat"),
                    f"manifest.requests.{profile}.prompt_repeat",
                    1,
                )
                prompt = _string(
                    expected_request.get("prompt"),
                    f"manifest.requests.{profile}.prompt",
                )
                expected_request["prompt"] = prompt * repeat
            expected_request["stream"] = request_stream
            expected_request_sha256 = hashlib.sha256(
                _jq_1_6_request_json_bytes(expected_request)
            ).hexdigest()
            if request_body_sha256 != expected_request_sha256:
                _fail(
                    f"{path}.request_body_sha256",
                    "does not bind the manifest profile and exact stream action bytes",
                )
            response_sha256 = _string(
                event["response_body_sha256"],
                f"{path}.response_body_sha256",
                SHA256_RE,
            )
            response_bytes = _integer(
                event["response_bytes"], f"{path}.response_bytes"
            )
            if response_bytes == 0 and response_sha256 != EMPTY_SHA256:
                _fail(
                    f"{path}.response_body_sha256",
                    "zero response bytes require the empty-body SHA-256",
                )
            outcome = _string(event["outcome"], f"{path}.outcome")
            if outcome not in {"success", "invalid", "overload", "cancelled", "disconnected", "timeout", "failure"}:
                _fail(f"{path}.outcome", "is not a closed outcome")
            status = _integer(event["http_status"], f"{path}.http_status")
            if status > 599:
                _fail(f"{path}.http_status", "must be <= 599")
            _number(event["latency_ms"], f"{path}.latency_ms")
            generated = event["generated_sha256"]
            if generated is not None:
                _string(generated, f"{path}.generated_sha256", SHA256_RE)
            if (outcome == "success") != (generated is not None):
                _fail(f"{path}.generated_sha256", "must be present exactly for success")
            if outcome == "success" and not 200 <= status < 300:
                _fail(f"{path}.http_status", "success requires 2xx")
            if outcome == "invalid" and not (400 <= status < 500 and status != 429):
                _fail(f"{path}.http_status", "invalid requires non-429 4xx")
            if outcome == "overload" and status != 429:
                _fail(f"{path}.http_status", "overload requires 429")
            if outcome != "failure":
                transport_contracts = {
                    "normal": (False, 0, {"success"}, lambda: 200 <= status < 300 and response_bytes > 0),
                    "invalid": (False, 0, {"invalid"}, lambda: 400 <= status < 500 and status != 429 and response_bytes > 0),
                    "overload": (False, 0, {"success", "overload"}, lambda: (200 <= status < 300 or status == 429) and response_bytes > 0),
                    "cancel": (False, CURL_TIMEOUT_EXIT_CODE, {"cancelled"}, lambda: status == 0 and response_bytes == 0 and response_sha256 == EMPTY_SHA256),
                    "disconnect": (True, CURL_WRITE_ERROR_EXIT_CODE, {"disconnected"}, lambda: status == 200 and response_bytes == DISCONNECT_RESPONSE_BYTES),
                }
                expected_stream, expected_exit, expected_outcomes, proof_matches = (
                    transport_contracts[action]
                )
                if (
                    request_stream != expected_stream
                    or curl_exit_code != expected_exit
                    or outcome not in expected_outcomes
                    or not proof_matches()
                ):
                    _fail(path, f"does not satisfy the exact {action} transport contract")
        elif kind == "restart":
            if not isinstance(event["graceful"], bool):
                _fail(f"{path}.graceful", "must be boolean")
            _integer(event["exit_code"], f"{path}.exit_code")
            _number(event["elapsed_ms"], f"{path}.elapsed_ms")
            _string(event["before_generated_sha256"], f"{path}.before_generated_sha256", SHA256_RE)
            _string(event["after_generated_sha256"], f"{path}.after_generated_sha256", SHA256_RE)
        elif kind in {"scenario_end", "run_end"} and event["status"] not in {"success", "failure"}:
            _fail(f"{path}.status", "must be success or failure")
        elif kind == "failure":
            _string(event["stage"], f"{path}.stage")
            _string(event["message"], f"{path}.message")
    if rows[0]["kind"] != "run_start" or rows[-1]["kind"] != "run_end":
        _fail("events", "must be bracketed by run_start and run_end")


def _slope_per_hour(samples: list[dict[str, Any]], field: str) -> float:
    if len(samples) < 2:
        return math.inf
    x = [(sample["monotonic_ns"] - samples[0]["monotonic_ns"]) / 1_000_000_000 for sample in samples]
    y = [float(sample[field.split(".")[0]][field.split(".")[1]]) for sample in samples]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        return math.inf
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator * 3600


def _check(name: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "observed": observed, "threshold": threshold}


def evaluate(
    manifest_path: Path | str,
    run_directory: Path | str,
    *,
    runtime_receipts_directory: Path | str | None = None,
    correctness_golden: Path | str | None = None,
    native_correctness_report: Path | str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION, "status": "error", "passed": False,
        "bindings": None, "scenario_summaries": [], "observations": {}, "checks": [], "errors": [],
    }
    try:
        manifest_file = Path(manifest_path)
        directory = Path(run_directory)
        manifest = _validate_manifest(_load_json(manifest_file), str(manifest_file))
        run_path = directory / "run.json"
        run_value, run_json_sha256 = _load_regular_json(
            run_path, str(run_path), RAW_MEMBER_MAX_BYTES["run.json"]
        )
        run = _validate_run(
            run_value,
            str(run_path),
            _sha256(manifest_file),
        )
        events_path = directory / "events.jsonl"
        events_raw, events_jsonl_sha256 = _load_regular_bytes(
            events_path,
            str(events_path),
            RAW_MEMBER_MAX_BYTES["events.jsonl"],
        )
        rows = _parse_jsonl_bytes(events_raw, str(events_path))
        if run["target"]["kind"] != manifest["target"]["kind"]:
            _fail("run.json.target.kind", "does not match manifest target kind")
        if run["target"]["image_id"] != f"sha256:{run['source']['image_sha256']}":
            _fail("run.json.target.image_id", "does not match bound image SHA-256")
        trusted_correctness = _validate_trusted_correctness(
            manifest,
            run,
            correctness_golden,
            native_correctness_report,
        )
        runtime_provenance, runtime_timing = _validate_runtime_receipts(
            runtime_receipts_directory,
            run,
            trusted_correctness,
            run_json_sha256=run_json_sha256,
            events_jsonl_sha256=events_jsonl_sha256,
        )
        scenarios = {scenario["id"]: scenario for scenario in manifest["scenarios"]}
        _validate_events(
            rows,
            run["binding_sha256"],
            manifest["scenarios"],
            manifest["requests"],
        )
        event_span_ns = rows[-1]["monotonic_ns"] - rows[0]["monotonic_ns"]
        if runtime_timing["elapsed_ns"] < event_span_ns:
            _fail(
                "container-inspect-post.json[0].State",
                "Docker runtime is shorter than the preserved monotonic event span",
            )
        by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in rows:
            if event["scenario_id"] is not None:
                by_scenario[event["scenario_id"]].append(event)
        checks: list[dict[str, Any]] = []
        boundary_counts = Counter(event["kind"] for event in rows)
        checks.append(_check("run_boundaries", boundary_counts["run_start"] == 1 and boundary_counts["run_end"] == 1 and rows[-1]["status"] == "success", {"run_start": boundary_counts["run_start"], "run_end": boundary_counts["run_end"], "status": rows[-1].get("status")}, "one successful pair"))
        thresholds = manifest["thresholds"]
        outcome_counts: Counter[str] = Counter()
        metric_counter_maxima: Counter[str] = Counter()
        rollback_hashes: dict[str, set[str]] = {}
        golden_profile = manifest["golden"]["request_profile"]
        expected_golden = manifest["golden"]["generated_sha256"]
        for scenario_id, scenario in scenarios.items():
            events = by_scenario[scenario_id]
            kinds = Counter(event["kind"] for event in events)
            samples = [event for event in events if event["kind"] == "sample"]
            requests = [event for event in events if event["kind"] == "request"]
            starts = [event for event in events if event["kind"] == "scenario_start"]
            ends = [event for event in events if event["kind"] == "scenario_end"]
            for request in requests:
                outcome_counts[request["outcome"]] += 1
            for sample in samples:
                for name, value in sample["metrics"]["counters"].items():
                    metric_counter_maxima[name] = max(metric_counter_maxima[name], value)
            counters_monotonic = all(
                right["process"]["pid"] != left["process"]["pid"]
                or all(right["metrics"]["counters"][name] >= left["metrics"]["counters"][name] for name in left["metrics"]["counters"])
                for left, right in zip(samples, samples[1:])
            )
            complete = kinds["scenario_start"] == 1 and kinds["scenario_end"] == 1 and events and events[0]["kind"] == "scenario_start" and events[-1].get("status") == "success"
            checks.append(_check(f"{scenario_id}.complete", complete, dict(kinds), "one successful start/end"))
            mode_matches = len(starts) == 1 and starts[0]["execution_completion"] == scenario["execution_completion"]
            checks.append(_check(f"{scenario_id}.execution_completion", mode_matches, None if not starts else starts[0]["execution_completion"], scenario["execution_completion"]))
            checks.append(_check(f"{scenario_id}.service_counters_monotonic", counters_monotonic, counters_monotonic, True))
            checks.append(_check(f"{scenario_id}.samples", len(samples) >= thresholds["minimum_samples_per_scenario"], len(samples), thresholds["minimum_samples_per_scenario"]))
            checks.append(_check(f"{scenario_id}.requests", bool(requests), len(requests), ">= 1"))
            observed_duration_seconds = (
                (ends[0]["monotonic_ns"] - starts[0]["monotonic_ns"])
                / 1_000_000_000
                if len(starts) == 1 and len(ends) == 1
                else 0.0
            )
            required_duration_seconds = scenario["duration_seconds"]
            checks.append(
                _check(
                    f"{scenario_id}.duration_seconds",
                    observed_duration_seconds >= required_duration_seconds,
                    observed_duration_seconds,
                    required_duration_seconds,
                )
            )
            sample_span_seconds = (
                (samples[-1]["monotonic_ns"] - samples[0]["monotonic_ns"])
                / 1_000_000_000
                if len(samples) >= 2
                else 0.0
            )
            sample_coverage_tolerance_seconds = max(
                thresholds["maximum_sample_gap_ms"] / 1000,
                thresholds["sample_interval_ms"] * 2 / 1000,
            )
            required_sample_span_seconds = max(
                0.0,
                required_duration_seconds - sample_coverage_tolerance_seconds,
            )
            checks.append(
                _check(
                    f"{scenario_id}.sample_coverage_seconds",
                    sample_span_seconds >= required_sample_span_seconds,
                    sample_span_seconds,
                    required_sample_span_seconds,
                )
            )
            restart_times = [event["monotonic_ns"] for event in events if event["kind"] == "restart"]
            gaps = [
                (right["monotonic_ns"] - left["monotonic_ns"]) / 1_000_000
                for left, right in zip(samples, samples[1:])
                if not any(left["monotonic_ns"] < restart < right["monotonic_ns"] for restart in restart_times)
            ]
            maximum_gap = max(gaps, default=0.0)
            checks.append(_check(f"{scenario_id}.sample_gap_ms", maximum_gap <= thresholds["maximum_sample_gap_ms"], maximum_gap, thresholds["maximum_sample_gap_ms"]))
            tail_count = max(2, math.ceil(len(samples) * thresholds["plateau_tail_fraction"]))
            tail = samples[-tail_count:]
            if len(tail) >= 2:
                rss_growth = max(sample["process"]["rss_bytes"] for sample in tail) - min(sample["process"]["rss_bytes"] for sample in tail)
                vram_growth = max(sample["gpu"]["vram_bytes"] for sample in tail) - min(sample["gpu"]["vram_bytes"] for sample in tail)
                rss_slope = _slope_per_hour(tail, "process.rss_bytes")
                vram_slope = _slope_per_hour(tail, "gpu.vram_bytes")
            else:
                rss_growth = vram_growth = 0
                rss_slope = vram_slope = None
            checks.extend([
                _check(f"{scenario_id}.rss_plateau_growth", rss_growth <= thresholds["maximum_rss_plateau_growth_bytes"], rss_growth, thresholds["maximum_rss_plateau_growth_bytes"]),
                _check(f"{scenario_id}.rss_slope_per_hour", rss_slope is not None and rss_slope <= thresholds["maximum_rss_slope_bytes_per_hour"], rss_slope, thresholds["maximum_rss_slope_bytes_per_hour"]),
                _check(f"{scenario_id}.vram_plateau_growth", vram_growth <= thresholds["maximum_vram_plateau_growth_bytes"], vram_growth, thresholds["maximum_vram_plateau_growth_bytes"]),
                _check(f"{scenario_id}.vram_slope_per_hour", vram_slope is not None and vram_slope <= thresholds["maximum_vram_slope_bytes_per_hour"], vram_slope, thresholds["maximum_vram_slope_bytes_per_hour"]),
            ])
            allowed = {"success"}
            if scenario["kind"] == "invalid":
                allowed = {"invalid"}
            elif scenario["kind"] == "overload":
                allowed = {"success", "overload"}
            elif scenario["kind"] == "cancellation-disconnect":
                allowed = {"success", "cancelled", "disconnected"}
            unexpected = Counter(request["outcome"] for request in requests if request["outcome"] not in allowed)
            checks.append(_check(f"{scenario_id}.request_outcomes", not unexpected, dict(unexpected), sorted(allowed)))
            golden_only = scenario["request_profile"] == golden_profile and scenario.get(
                "secondary_request_profile", golden_profile
            ) == golden_profile
            if golden_only:
                successful_hashes = {
                    request["generated_sha256"]
                    for request in requests
                    if request["outcome"] == "success"
                }
                checks.append(
                    _check(
                        f"{scenario_id}.golden_parity",
                        successful_hashes == {expected_golden},
                        sorted(successful_hashes),
                        [expected_golden],
                    )
                )
            if scenario["kind"] == "rollback":
                hashes = {request["generated_sha256"] for request in requests if request["outcome"] == "success"}
                rollback_hashes[scenario["execution_completion"]] = hashes
            report["scenario_summaries"].append({
                "scenario_id": scenario_id, "kind": scenario["kind"], "events": len(events),
                "samples": len(samples), "requests": len(requests), "maximum_sample_gap_ms": maximum_gap,
                "observed_duration_seconds": observed_duration_seconds,
                "sample_span_seconds": sample_span_seconds,
                "rss_slope_bytes_per_hour": rss_slope, "vram_slope_bytes_per_hour": vram_slope,
            })
        final_samples = [event for event in rows if event["kind"] == "sample" and event["scenario_id"] is None]
        final_shape = len(final_samples) == 1 and len(rows) >= 2 and rows[-2] is final_samples[0]
        checks.append(_check("final_sample_position", final_shape, len(final_samples), "exactly one penultimate global sample"))
        first_process_sample = next((event for event in rows if event["kind"] == "sample" and event["scenario_id"] is not None), None)
        checks.append(_check("initial_target_pid_binding", first_process_sample is not None and first_process_sample["process"]["pid"] == run["target"]["pid"], None if first_process_sample is None else first_process_sample["process"]["pid"], run["target"]["pid"]))
        final = final_samples[-1] if final_samples else None
        final_values = None if final is None else {
            "process_pid": final["process"]["pid"],
            "process_rss_bytes": final["process"]["rss_bytes"],
            "process_hwm_bytes": final["process"]["hwm_bytes"],
            "process_fd_count": final["process"]["fd_count"],
            "process_thread_count": final["process"]["thread_count"],
            "process_children": final["process"]["children"],
            "gpu_vram_bytes": final["gpu"]["vram_bytes"],
            "active_requests": final["metrics"]["active_requests"],
            "waiting_requests": final["metrics"]["waiting_requests"],
            "kv_allocated_blocks": final["metrics"]["kv_allocated_blocks"],
            **final["metrics"]["allocation"],
        }
        final_quiescent = final_values is not None and all(
            value == ([] if key == "process_children" else 0)
            for key, value in final_values.items()
        )
        checks.append(
            _check(
                "final_quiescence",
                final_quiescent,
                final_values,
                "zero process/GPU/service/allocation state and no children",
            )
        )
        python_children = []
        for event in rows:
            if event["kind"] == "sample":
                for child in event["process"]["children"]:
                    if PYTHON_RE.search(child["comm"]) or PYTHON_RE.search(child["executable"]):
                        python_children.append(child)
        checks.append(_check("no_python_children", not python_children, python_children, []))
        dropped = any(
            event["kind"] == "sample" and event["sample_dropped"]
            for event in rows
        )
        checks.append(_check("no_dropped_samples", not dropped, dropped, False))
        failures = [event for event in rows if event["kind"] == "failure"]
        checks.append(_check("no_failure_events", not failures, len(failures), 0))
        checks.extend([
            _check("cancellations_observed", outcome_counts["cancelled"] >= thresholds["minimum_cancellations"], outcome_counts["cancelled"], thresholds["minimum_cancellations"]),
            _check("disconnects_observed", outcome_counts["disconnected"] >= thresholds["minimum_disconnects"], outcome_counts["disconnected"], thresholds["minimum_disconnects"]),
            _check("overloads_observed", outcome_counts["overload"] >= thresholds["minimum_overloads"], outcome_counts["overload"], thresholds["minimum_overloads"]),
            _check("service_cancellations_observed", metric_counter_maxima["cancellations"] >= thresholds["minimum_cancellations"], metric_counter_maxima["cancellations"], thresholds["minimum_cancellations"]),
            _check("service_disconnects_observed", metric_counter_maxima["disconnects"] >= thresholds["minimum_disconnects"], metric_counter_maxima["disconnects"], thresholds["minimum_disconnects"]),
            _check("service_overloads_observed", metric_counter_maxima["overloads"] >= thresholds["minimum_overloads"], metric_counter_maxima["overloads"], thresholds["minimum_overloads"]),
        ])
        restarts = [event for event in rows if event["kind"] == "restart"]
        restart_ok = len(restarts) == 1 and restarts[0]["graceful"] and restarts[0]["exit_code"] == 0 and restarts[0]["elapsed_ms"] <= thresholds["graceful_shutdown_deadline_ms"] and restarts[0]["before_generated_sha256"] == expected_golden and restarts[0]["after_generated_sha256"] == expected_golden
        checks.append(_check("graceful_restart_golden_parity", restart_ok, restarts, "one bounded graceful exact-parity restart"))
        left = rollback_hashes.get("iteration-batch", set())
        right = rollback_hashes.get("per-operation", set())
        rollback_ok = left == {expected_golden} and right == {expected_golden}
        checks.append(_check("rollback_golden_parity", rollback_ok, {"iteration-batch": sorted(left), "per-operation": sorted(right)}, "one identical non-null hash"))
        passed = all(check["passed"] for check in checks)
        report.update({
            "status": "passed" if passed else "failed", "passed": passed,
            "bindings": {
                "contract_id": manifest["contract_id"],
                "reviewed_manifest_template_canonical_sha256": (
                    REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256
                ),
                "manifest_sha256": run["manifest_sha256"],
                "binding_sha256": run["binding_sha256"],
                "trusted_correctness": trusted_correctness,
                "runtime_provenance": runtime_provenance,
                "source": run["source"],
            },
            "observations": {"event_count": len(rows), "outcome_counts": dict(sorted(outcome_counts.items())), "service_counter_maxima": dict(sorted(metric_counter_maxima.items())), "final": final_values},
            "checks": checks,
        })
    except InputError as error:
        report["errors"] = [str(error)]
    return report


def _raw_payload_paths(
    manifest_path: Path | str,
    run_directory: Path | str,
    runtime_receipts_directory: Path | str,
) -> dict[str, Path]:
    directory = Path(run_directory)
    receipts = Path(runtime_receipts_directory)
    result = {
        "manifest.json": Path(manifest_path),
        "run.json": directory / "run.json",
        "events.jsonl": directory / "events.jsonl",
    }
    result.update({name: receipts / name for name in RUNTIME_RECEIPT_FILENAMES})
    return result


def _write_raw_archive(
    output: Path,
    inputs: Mapping[str, Path],
    *,
    runtime_receipts_directory: Path,
) -> None:
    if set(inputs) != set(RAW_ARCHIVE_PAYLOADS):
        _fail("raw evidence inputs", "exact canonical payload inventory is required")
    created = False
    try:
        with ExitStack() as stack:
            receipts_descriptor, receipts_metadata = (
                _open_runtime_receipt_directory(runtime_receipts_directory)
            )
            stack.callback(os.close, receipts_descriptor)
            opened: dict[str, tuple[Any, os.stat_result]] = {}
            digests: dict[str, str] = {}
            for name in RAW_ARCHIVE_PAYLOADS:
                handle, metadata = _regular_file(
                    inputs[name],
                    name,
                    RAW_MEMBER_MAX_BYTES[name],
                    directory_fd=(
                        receipts_descriptor
                        if name in RUNTIME_RECEIPT_FILENAMES
                        else None
                    ),
                    entry_name=(
                        name if name in RUNTIME_RECEIPT_FILENAMES else None
                    ),
                )
                stack.callback(handle.close)
                opened[name] = (handle, metadata)
                digests[name] = _stream_sha256(handle)
                _assert_held_file_stable(handle, metadata, name)
            checksums = b"".join(
                f"{digests[name]}  {name}\n".encode("ascii")
                for name in RAW_ARCHIVE_PAYLOADS
            )
            payloads: dict[str, tuple[int, Any]] = {
                name: (metadata.st_size, handle)
                for name, (handle, metadata) in opened.items()
            }
            payloads["SHA256SUMS"] = (len(checksums), io.BytesIO(checksums))
            with tarfile.open(output, "x:", format=tarfile.USTAR_FORMAT) as archive:
                created = True
                for name in sorted(payloads):
                    size, handle = payloads[name]
                    archive.addfile(_canonical_tar_info(name, size), handle)
            for name, (handle, metadata) in opened.items():
                _assert_held_file_stable(handle, metadata, name)
            _assert_runtime_receipt_directory_stable(
                receipts_descriptor, receipts_metadata
            )
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if created:
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        raise


def _held_file_equals_path(
    left_handle: Any,
    left_metadata: os.stat_result,
    right: Path,
) -> bool:
    right_handle: Any | None = None
    try:
        right_handle, right_metadata = _regular_file(
            right, "canonical raw evidence archive", MAX_RAW_ARCHIVE_BYTES
        )
        if left_metadata.st_size != right_metadata.st_size:
            return False
        left_handle.seek(0)
        while True:
            left_block = left_handle.read(1024 * 1024)
            right_block = right_handle.read(1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                _assert_held_file_stable(
                    left_handle, left_metadata, "raw evidence archive"
                )
                _assert_held_file_stable(
                    right_handle, right_metadata, "canonical raw evidence archive"
                )
                return True
    except OSError as error:
        _fail("raw evidence archive", f"cannot compare canonical archive: {error}")
    finally:
        left_handle.seek(0)
        if right_handle is not None:
            right_handle.close()


def _materialize_raw_evidence_archive(path: Path, destination: Path) -> dict[str, str]:
    label = "raw evidence archive"
    handle: Any | None = None
    try:
        handle, metadata = _regular_file(path, label, MAX_RAW_ARCHIVE_BYTES)
        archive_sha256 = _stream_sha256(handle)
        _assert_held_file_stable(handle, metadata, label)
        _assert_path_still_identifies_held_file(path, metadata, label)
        receipts_destination = destination / "runtime-receipts"
        receipts_destination.mkdir(mode=0o700)

        def materialized_path(name: str) -> Path:
            parent = (
                receipts_destination
                if name in RUNTIME_RECEIPT_FILENAMES
                else destination
            )
            return parent / name

        with tarfile.open(fileobj=handle, mode="r:") as archive:
            members = archive.getmembers()
            expected_names = sorted(RAW_ARCHIVE_MEMBERS)
            if [member.name for member in members] != expected_names:
                _fail(label, f"exact ordered inventory required: {expected_names}")
            digests: dict[str, str] = {}
            checksum_bytes: bytes | None = None
            for member in members:
                name = member.name
                if not member.isreg():
                    _fail(label, f"member must be a regular file: {name}")
                if member.pax_headers:
                    _fail(label, f"PAX extensions are forbidden: {name}")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mode != 0o644
                    or member.mtime != 0
                ):
                    _fail(label, f"non-canonical metadata for {name}")
                maximum = RAW_MEMBER_MAX_BYTES[name]
                if member.size <= 0 or member.size > maximum:
                    _fail(label, f"invalid size for {name}")
                source = archive.extractfile(member)
                if source is None:
                    _fail(label, f"cannot read {name}")
                target_path = materialized_path(name)
                digest = hashlib.sha256()
                total = 0
                with target_path.open("xb") as target:
                    while block := source.read(1024 * 1024):
                        total += len(block)
                        if total > maximum:
                            _fail(label, f"oversized member {name}")
                        digest.update(block)
                        target.write(block)
                if total != member.size:
                    _fail(label, f"truncated member {name}")
                digests[name] = digest.hexdigest()
                if name == "SHA256SUMS":
                    checksum_bytes = target_path.read_bytes()
        _assert_held_file_stable(handle, metadata, label)

        expected_checksums = b"".join(
            f"{digests[name]}  {name}\n".encode("ascii")
            for name in RAW_ARCHIVE_PAYLOADS
        )
        if checksum_bytes != expected_checksums:
            _fail(
                f"{label}.SHA256SUMS",
                "does not exactly checksum the canonical payload files",
            )
        canonical = destination / "canonical.tar"
        _write_raw_archive(
            canonical,
            {name: materialized_path(name) for name in RAW_ARCHIVE_PAYLOADS},
            runtime_receipts_directory=receipts_destination,
        )
        if not _held_file_equals_path(handle, metadata, canonical):
            _fail(label, "bytes are not the canonical deterministic USTAR encoding")
        _assert_path_still_identifies_held_file(path, metadata, label)
        return {
            "archive_sha256": archive_sha256,
            **{f"{name}_sha256": digests[name] for name in RAW_ARCHIVE_PAYLOADS},
        }
    except InputError:
        raise
    except (FileExistsError, OSError, tarfile.TarError) as error:
        _fail(label, f"cannot materialize deterministic uncompressed tar: {error}")
    finally:
        if handle is not None:
            handle.close()


def replay_raw_evidence_archive(
    path: Path | str,
    *,
    correctness_golden: Path | str | None = None,
    native_correctness_report: Path | str | None = None,
) -> dict[str, Any]:
    """Rebuild a soak report only from a canonical raw evidence archive."""

    with tempfile.TemporaryDirectory(prefix="riley-soak-replay-") as temporary:
        directory = Path(temporary)
        bindings = _materialize_raw_evidence_archive(Path(path), directory)
        report = evaluate(
            directory / "manifest.json",
            directory,
            runtime_receipts_directory=directory / "runtime-receipts",
            correctness_golden=correctness_golden,
            native_correctness_report=native_correctness_report,
        )
        return {"report": report, **bindings}


def package_raw_evidence(
    manifest_path: Path | str,
    run_directory: Path | str,
    output: Path | str,
    *,
    runtime_receipts_directory: Path | str | None = None,
    correctness_golden: Path | str | None = None,
    native_correctness_report: Path | str | None = None,
) -> dict[str, Any]:
    """Create and self-replay the canonical raw soak evidence archive."""

    report = evaluate(
        manifest_path,
        run_directory,
        runtime_receipts_directory=runtime_receipts_directory,
        correctness_golden=correctness_golden,
        native_correctness_report=native_correctness_report,
    )
    if report["passed"] is not True:
        detail = "; ".join(report["errors"]) or report["status"]
        _fail("soak run", f"cannot package non-passing evidence: {detail}")
    output_path = Path(output)
    if runtime_receipts_directory is None:
        _fail(
            "--runtime-receipts-directory",
            "the seven remote launcher runtime receipts are required",
        )
    _write_raw_archive(
        output_path,
        _raw_payload_paths(
            manifest_path,
            run_directory,
            runtime_receipts_directory,
        ),
        runtime_receipts_directory=Path(runtime_receipts_directory),
    )
    try:
        replay = replay_raw_evidence_archive(
            output_path,
            correctness_golden=correctness_golden,
            native_correctness_report=native_correctness_report,
        )
        if _canonical_json_bytes(replay["report"]) != _canonical_json_bytes(report):
            _fail("raw evidence archive", "self-replayed report differs from source run")
        return replay
    except Exception:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--runtime-receipts-directory", required=True, type=Path)
    parser.add_argument("--correctness-golden", required=True, type=Path)
    parser.add_argument("--native-correctness-report", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="create without overwriting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.manifest,
        args.run_directory,
        runtime_receipts_directory=args.runtime_receipts_directory,
        correctness_golden=args.correctness_golden,
        native_correctness_report=args.native_correctness_report,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        try:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        except (FileExistsError, OSError) as error:
            print(f"cannot create report {args.report}: {error}", file=sys.stderr)
            return 2
    sys.stdout.write(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
