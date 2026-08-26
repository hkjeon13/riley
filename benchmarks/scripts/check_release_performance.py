#!/usr/bin/env python3
"""Check a release candidate against the immutable PR15 performance baseline."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import stat
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


_NATIVE_CHECKER_PATH = Path(__file__).with_name("check_native_profile_pair.py")
_NATIVE_SPEC = importlib.util.spec_from_file_location(
    "rustinfer_release_native_profile_contract", _NATIVE_CHECKER_PATH
)
if _NATIVE_SPEC is None or _NATIVE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load native profile contract: {_NATIVE_CHECKER_PATH}")
native_profile = importlib.util.module_from_spec(_NATIVE_SPEC)
sys.modules[_NATIVE_SPEC.name] = native_profile
_NATIVE_SPEC.loader.exec_module(native_profile)


BASELINE_SCHEMA = "rustinfer.release-performance-baseline.v1"
CANDIDATE_SCHEMA = "rustinfer.release-performance-candidate.v1"
REPORT_SCHEMA = "rustinfer.release-performance-report.v1"
BASELINE_SHA256 = "fb80fd829ac3a7ff9097d046413c0b2dc3c12053597ec562c8d964737eae9bf5"
PR15_REQUEST_IDENTITY_SHA256 = (
    "e6a99a749c41a8227574c96a1d23f8b7d877d6e75b0df4d99154db1b1921a2e6"
)
CORRECTNESS_GATE_ID = "pr15-iteration-command-batch-exact-v1"
FIXED37_PRODUCTION_BATCH_GATE_ID = "pr16-fixed37-production-batch-e0-v1"
FIXED37_GOLDEN_FIXTURE_SHA256 = (
    "87333a1859be45a2f8e7563d898dde5e64256ccc03ca4da3cab90def07dd3c95"
)
FIXED37_GOLDEN_TOKEN_IDS_SHA256 = (
    "9e38488c0d41dae4a28e7e262baf772f2c643e9f8a9c57941a9e47aaec77ac5c"
)
FIXED37_CACHED_GROWING_COSINE_MIN = 0.997_903_530_549_539_3
FIXED37_CACHED_GROWING_MAX_ABS_MAX = 5.852_936_458_587_647
FIXED37_CACHED_GROWING_MEAN_ABS_MAX = 1.151_280_319_263_363
OPTIMIZATION_GOLDEN_TOKEN_IDS = [
    4052,
    2025,
    284,
    965,
    6497,
    288,
    1492,
    418,
    260,
    16438,
    30,
    198,
    198,
    504,
    16438,
    314,
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
CANDIDATE_ID_RE = re.compile(
    r"^rustinfer-((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*))-rc([1-9][0-9]*)$"
)
RUNNER_MANIFEST_SCHEMA = "rustinfer.release-performance-runner-manifest.v3"
RUNNER_EXECUTION_SCHEMA = "rustinfer.release-performance-execution-receipt.v1"
RUNNER_SUPERVISOR_LABEL = "org.rustinfer.release-performance-supervisor"
RUNNER_ZERO_TIME = "0001-01-01T00:00:00Z"
RUNNER_CONTAINER_COMMAND = (
    "test -z \"$(/usr/bin/find /workspace -mindepth 1 -print -quit)\"; "
    "/usr/bin/tar --extract --file /input/source.tar --directory /workspace; "
    "cd /workspace; exec /bin/bash "
    "ci/release/run_release_performance_once.sh"
)
RUNNER_CONTAINER_ENTRYPOINT = ["/bin/bash"]
RUNNER_CONTAINER_CMD = ["-ceu", RUNNER_CONTAINER_COMMAND]
RUNNER_FIXED_PREFLIGHT = {
    "environment_id": "rtx4090-ubuntu22-driver580-v1",
    "os_id": "ubuntu",
    "os_version_id": "22.04",
    "kernel_release": "6.8.0-138-generic",
    "machine": "x86_64",
    "cpu_model": "Intel Core i7-13700K",
    "physical_cpu_cores": "16",
    "logical_cpu_threads": "24",
    "ram_bytes": "67185598464",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "compute_capability": "8.9",
    "memory_total_mib": "24564",
    "driver_version": "580.173.02",
    "persistence_mode": "Disabled",
    "power_limit_w": "450.00",
    "graphics_clock_mhz": "[N/A]",
    "memory_clock_mhz": "[N/A]",
    "cpu_governor": "powersave",
    "cpu_governor_policy_count": "24",
    "clock_synchronized": "yes",
    "staging_minimum_bytes": "21474836480",
}
RUNNER_PREFLIGHT_FIELDS = frozenset(
    {
        *RUNNER_FIXED_PREFLIGHT,
        "git_revision",
        "memory_used_mib",
        "temperature_c",
        "staging_available_bytes",
    }
)
RUNNER_GPU_ROW = (
    "NVIDIA GeForce RTX 4090",
    "GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0",
    "00000000:01:00.0",
    "24564",
    "580.173.02",
    "8.9",
)
RUNNER_PROXY_ENV = {
    name: ""
    for name in (
        "ALL_PROXY",
        "FTP_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "ftp_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    )
}
RUNNER_REVIEWED_TOOLS = {
    "mawk": {
        "path": "/usr/bin/mawk",
        "sha256": "dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e",
    },
    "basename": {
        "path": "/usr/bin/basename",
        "sha256": "3c19cca8e2630f570580104778cc1e3398811c4c57e3252f0727ce411ab0ad22",
    },
    "bash": {
        "path": "/usr/bin/bash",
        "sha256": "59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4",
    },
    "cat": {
        "path": "/bin/cat",
        "sha256": "210ffa7daedb3ef6e9230d391e9a10043699ba81080ebf40c6de70ed77e278ba",
    },
    "chmod": {
        "path": "/usr/bin/chmod",
        "sha256": "e624a2e918718e570f989dd05b219278c9fa7ae3b3ab8830302b2d98e0c7dca8",
    },
    "cmp": {
        "path": "/usr/bin/cmp",
        "sha256": "b355472d3c90ea94d11ebb8b750e6946ccd348edc6fca4aefc1235c3994ef791",
    },
    "cp": {
        "path": "/usr/bin/cp",
        "sha256": "8da5881bb59f65673bc22b3a09b0d663b19bc0e785cf986b05d41b8222449ec2",
    },
    "dirname": {
        "path": "/usr/bin/dirname",
        "sha256": "674a6c35e9ece6a6ac62e6442e3c65f391f8a1a8d1537bdd4b2203423ec16e94",
    },
    "df": {
        "path": "/usr/bin/df",
        "sha256": "b06fe81669b9383abed94bb5cae1cb7a63c6e02801b1b7dd1c08d7d2c8987e86",
    },
    "docker": {
        "path": "/usr/bin/docker",
        "sha256": "29be5f37ee7fcb32bed170244a7d94f2eb94d272912e0bbe9328374e2eb4b7f6",
    },
    "env": {
        "path": "/usr/bin/env",
        "sha256": "85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0",
    },
    "find": {
        "path": "/usr/bin/find",
        "sha256": "791b89c8bffb8101fd7d4d212b80af66a2332834b05a42721104eb47e8fa2eb1",
    },
    "git": {
        "path": "/usr/bin/git",
        "sha256": "587ef21868c948b883993e23209b86a72a6ddc06aab1545c697ffc31075acd4a",
    },
    "grep": {
        "path": "/usr/bin/grep",
        "sha256": "73abb4280520053564fd4917286909ba3b054598b32c9cdfaf1d733e0202cc96",
    },
    "head": {
        "path": "/usr/bin/head",
        "sha256": "9e457645cdcfd74ee0a9688b25b7b017d8d393233a0c0bdf3bef3c57a1238ce2",
    },
    "hostname": {
        "path": "/usr/bin/hostname",
        "sha256": "d254481d352a5a2b55848a4aeac6002ad594d4ab605e7f1fd49a25683b33559e",
    },
    "install": {
        "path": "/usr/bin/install",
        "sha256": "519a00d199d07da6028ec5a9800d92c562934582a2ea1793b2cbc378a85c1439",
    },
    "mkdir": {
        "path": "/usr/bin/mkdir",
        "sha256": "bd2f081ac37d653181332bd27f35a6041dbf215a7957f65838a9cbec9e64928b",
    },
    "nvidia-smi": {
        "path": "/usr/bin/nvidia-smi",
        "sha256": "22964713c1701fb62b4dd10b26b0dd25d174e100af5bda20c65e0b0fcc32b3be",
    },
    "python3": {
        "path": "/usr/bin/python3.10",
        "sha256": "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86",
    },
    "sed": {
        "path": "/usr/bin/sed",
        "sha256": "42e2ce00721556ff9d371778fc36adcbb7c1697f65c3f996c6c9b28206dba565",
    },
    "sha256sum": {
        "path": "/usr/bin/sha256sum",
        "sha256": "7645c8e76d75515ccb75c9086bdcf0d4071f2985f380f249253ead7d7c6810b3",
    },
    "sleep": {
        "path": "/usr/bin/sleep",
        "sha256": "b9aec374a2b2a175a182f615291ad408820b7fb8c663a184e37fa3492d3f8eff",
    },
    "sort": {
        "path": "/usr/bin/sort",
        "sha256": "0fc26ce295e8e549635da2129e389f63685745b3be7c1737db6251a296f1cd78",
    },
    "stat": {
        "path": "/usr/bin/stat",
        "sha256": "9b571b54bd2f17f5fbb841e1886c2d364f5138a02533f4ac3dbfbdaf4dddbea3",
    },
    "tail": {
        "path": "/usr/bin/tail",
        "sha256": "d686c3513b6ecbcc6ac826383bd4b8b0f00aa6500d8d3d5e593687a3dee8fce0",
    },
    "timedatectl": {
        "path": "/usr/bin/timedatectl",
        "sha256": "a1d1298afc514e7143d1a7a4c0039ce1256871faf33fe356fd9063dd283df5d9",
    },
    "tr": {
        "path": "/usr/bin/tr",
        "sha256": "24f53bbf7e48b1be3b71f20cf29963a44dbf084aafe5301f0ed1425b91d1c60c",
    },
    "uname": {
        "path": "/usr/bin/uname",
        "sha256": "37df0311d0e24169abfd166bc6018d40b87306f7ff64d9eec256c8331ac26347",
    },
    "wc": {
        "path": "/usr/bin/wc",
        "sha256": "504463c7a12780b7439321be6e67f43ab61a3ff429cbf916c0722d19f98692a8",
    },
    "tar": {
        "path": "/usr/bin/tar",
        "sha256": "fd0d62eed19efd3e115aa1be44160f89d777cd1e6d6d8eb0ce7c8bdc879f59e2",
    },
}
RUNNER_REQUIRED_TOOLS = frozenset(RUNNER_REVIEWED_TOOLS)
RUNNER_FORBIDDEN_ENV_NAMES = frozenset(
    {
        "BASH_ENV",
        "BASHOPTS",
        "CDPATH",
        "DOCKER_CERT_PATH",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "DOCKER_TLS_VERIFY",
        "ENV",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_WORK_TREE",
        "LD_AUDIT",
        "LD_PRELOAD",
        "POSIXLY_CORRECT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONWARNINGS",
        "SHELLOPTS",
        "TAR_OPTIONS",
    }
)
RUNNER_GPU_MONITOR_HEADER = (
    "capture_id",
    "container_id",
    "stage",
    "sample_index",
    "power_limit_w",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "temperature_c",
    "memory_used_mib",
    "compute_processes",
)
RUNNER_RECEIPT_FILES = tuple(
    sorted(
        {
            "runner-manifest.json",
            "gpu.csv",
            "optimizer-image-inspect-before.json",
            "optimizer-image-inspect-after.json",
            "SHA256SUMS",
            *{
                f"run-{index}/{name}"
                for index in range(1, 6)
                for name in (
                    "preflight.txt",
                    "gpu-monitor.csv",
                    "container-inspect-before.json",
                    "container-inspect-after.json",
                    "candidate.json",
                    "execution-receipt.json",
                )
            },
        }
    )
)
RAW_EVIDENCE_FILES = tuple(f"candidate-{index}.json" for index in range(1, 6))
_RUNNER_RECEIPT_MAXIMUMS = {
    name: (
        native_profile.MAX_EVIDENCE_BYTES
        if name.endswith("/candidate.json")
        else 16 * 1024 * 1024
        if "inspect" in name
        else 256 * 1024
    )
    for name in RUNNER_RECEIPT_FILES
}
MAX_RAW_EVIDENCE_ARCHIVE_BYTES = (
    sum(_RUNNER_RECEIPT_MAXIMUMS.values()) + 2 * 1024 * 1024
)
PACKAGE_CANDIDATE_NAME = "release-performance-candidate.json"
PACKAGE_REPORT_NAME = "release-performance-report.json"
PACKAGE_RAW_EVIDENCE_NAME = "release-performance-evidence.tar"
_PACKAGE_STAGING_FILES = frozenset(
    (PACKAGE_CANDIDATE_NAME, PACKAGE_REPORT_NAME, PACKAGE_RAW_EVIDENCE_NAME)
)
_AT_FDCWD = -100
_LINUX_RENAME_NOREPLACE = 1
_DARWIN_RENAME_EXCL = 0x00000004


@dataclass(frozen=True)
class _HeldFileBinding:
    name: str
    digest: str
    metadata: os.stat_result
    maximum: int
    mode: int


class InputError(ValueError):
    """Malformed or integrity-invalid evidence."""


class ComparabilityError(ValueError):
    """Well-formed evidence from a different release lane."""


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise InputError(f"non-finite JSON number {value!r} is forbidden")


def _load_json_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = _read_bounded_regular(path, label, native_profile.MAX_EVIDENCE_BYTES)
    except OSError as error:
        raise InputError(f"cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InputError) as error:
        raise InputError(f"invalid {label} JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise InputError(f"{label}: root must be an object")
    return value, raw


def _closed_object(
    value: Any, path: str, required: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{path}: must be an object")
    keys = set(value)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        raise InputError(f"{path}: missing fields: {', '.join(missing)}")
    if extra:
        raise InputError(f"{path}: unknown fields: {', '.join(extra)}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{path}: must be a non-empty string")
    return value


def _candidate_id(value: Any, path: str) -> str:
    candidate_id = _string(value, path)
    if CANDIDATE_ID_RE.fullmatch(candidate_id) is None:
        raise InputError(
            f"{path}: expected "
            "rustinfer-<major>.<minor>.<patch>-rc<positive integer>"
        )
    return candidate_id


def _sha256(value: Any, path: str) -> str:
    text = _string(value, path)
    if SHA256_RE.fullmatch(text) is None:
        raise InputError(f"{path}: must be a lowercase SHA-256")
    return text


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InputError(f"{path}: must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{path}: must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "finite and > 0" if positive else "finite and >= 0"
        raise InputError(f"{path}: must be {qualifier}")
    return result


def _exact_json_value(value: Any, expected: Any) -> bool:
    """Compare decoded JSON without Python's bool/int equality aliasing."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _exact_json_value(value[name], expected[name]) for name in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact_json_value(actual, wanted)
            for actual, wanted in zip(value, expected, strict=True)
        )
    return value == expected


def _literal(value: Any, expected: Any, path: str) -> None:
    if not _exact_json_value(value, expected):
        raise InputError(f"{path}: expected {expected!r}, got {value!r}")


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest_file(path: Path, label: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        path_metadata = path.lstat()
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != path_metadata.st_dev
            or before.st_ino != path_metadata.st_ino
            or before.st_size <= 0
            or before.st_size > 4 * 1024**3
        ):
            raise InputError(f"{label}: must be one stable bounded regular file")
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if byte_count != before.st_size or any(
            getattr(before, field) != getattr(after, field) for field in stable
        ):
            raise InputError(f"{label}: changed while it was hashed")
        return digest.hexdigest()
    except InputError:
        raise
    except OSError as error:
        raise InputError(f"cannot hash {label} {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically rename without replacing an existing path."""

    source_bytes = os.fsencode(os.path.abspath(source))
    target_bytes = os.fsencode(os.path.abspath(target))
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise InputError(
                "atomic no-replace publish requires libc renameat2 on Linux"
            )
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        arguments = (
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            target_bytes,
            _LINUX_RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise InputError(
                "atomic no-replace publish requires renamex_np on macOS"
            )
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        arguments = (source_bytes, target_bytes, _DARWIN_RENAME_EXCL)
    else:
        raise InputError(
            "atomic no-replace evidence publish is supported only on Linux and macOS"
        )

    ctypes.set_errno(0)
    if rename(*arguments) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(target),
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive kernel contract check
            raise OSError("short write while creating release evidence")
        view = view[written:]


def _write_new_file(
    directory_descriptor: int, name: str, raw: bytes, mode: int = 0o644
) -> int:
    """Create, sync, and return a held read/write descriptor for a child."""

    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, mode, dir_fd=directory_descriptor)
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _stable_fd_snapshot(
    descriptor: int, label: str, maximum: int
) -> tuple[bytes, str, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise InputError(f"{label}: must be a regular file, not a link or device")
    if before.st_size <= 0 or before.st_size > maximum:
        raise InputError(f"{label}: must be between 1 and {maximum} bytes")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    after = os.fstat(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise InputError(f"{label}: changed while it was snapshotted")
    if len(raw) != before.st_size:
        raise InputError(f"{label}: changed or was truncated while it was read")
    return raw, digest.hexdigest(), before


def _same_inode(path: Path, expected: os.stat_result, *, directory: bool) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    return (
        expected_type(current.st_mode)
        and current.st_dev == expected.st_dev
        and current.st_ino == expected.st_ino
    )


def _record_held_file(
    descriptor: int,
    name: str,
    *,
    maximum: int,
    mode: int = 0o644,
) -> _HeldFileBinding:
    _raw, digest, metadata = _stable_fd_snapshot(
        descriptor, f"package child {name}", maximum
    )
    actual_mode = stat.S_IMODE(metadata.st_mode)
    if actual_mode != mode:
        raise InputError(
            f"package child {name}: expected mode {mode:#o}, got {actual_mode:#o}"
        )
    return _HeldFileBinding(
        name=name,
        digest=digest,
        metadata=metadata,
        maximum=maximum,
        mode=mode,
    )


def _verify_held_file(
    descriptor: int, binding: _HeldFileBinding, label: str
) -> None:
    _raw, digest, metadata = _stable_fd_snapshot(
        descriptor, label, binding.maximum
    )
    if (
        metadata.st_dev != binding.metadata.st_dev
        or metadata.st_ino != binding.metadata.st_ino
        or metadata.st_size != binding.metadata.st_size
        or stat.S_IMODE(metadata.st_mode) != binding.mode
        or digest != binding.digest
    ):
        raise InputError(f"{label}: inode, mode, size, or digest changed")


def _path_metadata_matches_binding(
    metadata: os.stat_result, binding: _HeldFileBinding
) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_dev == binding.metadata.st_dev
        and metadata.st_ino == binding.metadata.st_ino
        and metadata.st_size == binding.metadata.st_size
        and stat.S_IMODE(metadata.st_mode) == binding.mode
    )


def _verify_bound_file_path(
    descriptor: int,
    path: Path,
    binding: _HeldFileBinding,
    label: str,
) -> None:
    """Cross-check a visible pathname with an immutable held-FD binding."""

    if not _same_inode(path, binding.metadata, directory=False):
        raise InputError(f"{label}: path no longer names the held inode")
    _verify_held_file(descriptor, binding, label)
    if not _same_inode(path, binding.metadata, directory=False):
        raise InputError(f"{label}: path changed during verification")


def _verify_package_children(
    directory_descriptor: int,
    directory_metadata: os.stat_result,
    descriptors: Mapping[str, int],
    bindings: Mapping[str, _HeldFileBinding],
    label: str,
) -> None:
    current_directory = os.fstat(directory_descriptor)
    if (
        not stat.S_ISDIR(current_directory.st_mode)
        or current_directory.st_dev != directory_metadata.st_dev
        or current_directory.st_ino != directory_metadata.st_ino
    ):
        raise InputError(f"{label}: held directory inode changed")
    names_before = os.listdir(directory_descriptor)
    if len(names_before) != len(_PACKAGE_STAGING_FILES) or set(
        names_before
    ) != _PACKAGE_STAGING_FILES:
        raise InputError(
            f"{label}: exact three-file inventory required, got {sorted(names_before)}"
        )
    if set(descriptors) != _PACKAGE_STAGING_FILES or set(bindings) != _PACKAGE_STAGING_FILES:
        raise InputError(f"{label}: internal held-file inventory is incomplete")

    for name in sorted(_PACKAGE_STAGING_FILES):
        binding = bindings[name]
        descriptor = descriptors[name]
        if binding.name != name:
            raise InputError(f"{label}:{name}: held binding name mismatch")
        metadata_before = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if not _path_metadata_matches_binding(metadata_before, binding):
            raise InputError(f"{label}:{name}: path binding changed")
        _verify_held_file(descriptor, binding, f"{label}:{name}")
        metadata_after = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if not _path_metadata_matches_binding(metadata_after, binding):
            raise InputError(f"{label}:{name}: path changed during verification")

    names_after = os.listdir(directory_descriptor)
    if len(names_after) != len(_PACKAGE_STAGING_FILES) or set(
        names_after
    ) != _PACKAGE_STAGING_FILES:
        raise InputError(f"{label}: inventory changed during verification")


MODEL_FIELDS = {
    "model_id",
    "model_revision",
    "dtype",
    "weights_sha256",
    "tokenizer_sha256",
}
ENVIRONMENT_FIELDS = {
    "environment_id",
    "gpu_uuid",
    "compute_capability",
    "driver_version",
    "cuda_runtime_version",
    "cuda_toolkit_version",
    "cuda_architecture",
}
WORKLOAD_FIELDS = {
    "workload_id",
    "concurrency",
    "prompt_tokens",
    "output_tokens",
    "warmups_per_run",
    "measured_iterations_per_run",
    "independent_runs",
    "sampling",
    "execution_completion",
    "residual_rmsnorm",
}
METRIC_FIELDS = {
    "ttft_p95_ms",
    "tpot_p95_ms",
    "e2e_median_ms",
    "throughput_median_output_tokens_per_second",
}


def _validate_model(value: Any, path: str) -> dict[str, Any]:
    row = _closed_object(value, path, MODEL_FIELDS)
    result = {
        "model_id": _string(row["model_id"], f"{path}.model_id"),
        "model_revision": _string(
            row["model_revision"], f"{path}.model_revision"
        ),
        "dtype": _string(row["dtype"], f"{path}.dtype"),
        "weights_sha256": _sha256(
            row["weights_sha256"], f"{path}.weights_sha256"
        ),
        "tokenizer_sha256": _sha256(
            row["tokenizer_sha256"], f"{path}.tokenizer_sha256"
        ),
    }
    _literal(result["dtype"], "bf16", f"{path}.dtype")
    return result


def _validate_environment(value: Any, path: str) -> dict[str, str]:
    row = _closed_object(value, path, ENVIRONMENT_FIELDS)
    return {field: _string(row[field], f"{path}.{field}") for field in sorted(row)}


def _validate_workload(value: Any, path: str) -> dict[str, Any]:
    row = _closed_object(value, path, WORKLOAD_FIELDS)
    result: dict[str, Any] = {}
    for field in [
        "concurrency",
        "prompt_tokens",
        "output_tokens",
        "warmups_per_run",
        "measured_iterations_per_run",
        "independent_runs",
    ]:
        result[field] = _integer(row[field], f"{path}.{field}", 1)
    for field in [
        "workload_id",
        "sampling",
        "execution_completion",
        "residual_rmsnorm",
    ]:
        result[field] = _string(row[field], f"{path}.{field}")
    _literal(result["sampling"], "greedy", f"{path}.sampling")
    _literal(
        result["execution_completion"],
        "iteration-batch",
        f"{path}.execution_completion",
    )
    _literal(result["residual_rmsnorm"], "separate", f"{path}.residual_rmsnorm")
    return result


def _validate_metrics(value: Any, path: str) -> dict[str, float]:
    row = _closed_object(value, path, METRIC_FIELDS)
    return {
        field: _number(row[field], f"{path}.{field}", positive=True)
        for field in sorted(row)
    }


def _validate_baseline(document: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    actual_digest = _digest_bytes(raw)
    if actual_digest != BASELINE_SHA256:
        raise InputError(
            "baseline bytes are not the reviewed v1 baseline: "
            f"{actual_digest} != {BASELINE_SHA256}"
        )
    row = _closed_object(
        document,
        "baseline",
        {
            "schema_version",
            "baseline_id",
            "accepted",
            "measurement_binding",
            "promotion_binding",
            "model",
            "environment",
            "workload",
            "metrics",
            "thresholds",
            "evidence",
        },
    )
    _literal(row["schema_version"], BASELINE_SCHEMA, "baseline.schema_version")
    _literal(row["accepted"], True, "baseline.accepted")
    binding = _closed_object(
        row["measurement_binding"],
        "baseline.measurement_binding",
        {
            "git_commit",
            "source_archive_sha256",
            "profile_binary_sha256",
            "profile_image_sha256",
            "request_identity_sha256",
            "correctness_gate_id",
            "correctness_report_sha256",
            "semantic_class",
        },
    )
    if GIT_RE.fullmatch(_string(binding["git_commit"], "baseline.git_commit")) is None:
        raise InputError("baseline.git_commit: invalid commit")
    for field in [
        "source_archive_sha256",
        "profile_binary_sha256",
        "profile_image_sha256",
        "request_identity_sha256",
        "correctness_report_sha256",
    ]:
        _sha256(binding[field], f"baseline.measurement_binding.{field}")
    _literal(binding["semantic_class"], "E0", "baseline.semantic_class")
    _literal(
        binding["request_identity_sha256"],
        PR15_REQUEST_IDENTITY_SHA256,
        "baseline.measurement_binding.request_identity_sha256",
    )
    thresholds = _closed_object(
        row["thresholds"],
        "baseline.thresholds",
        {
            "ttft_p95_ratio_max",
            "tpot_p95_ratio_max",
            "e2e_median_ratio_max",
            "throughput_median_ratio_min",
        },
    )
    expected_thresholds = {
        "ttft_p95_ratio_max": 1.05,
        "tpot_p95_ratio_max": 1.05,
        "e2e_median_ratio_max": 1.05,
        "throughput_median_ratio_min": 0.95,
    }
    for field, expected in expected_thresholds.items():
        _literal(_number(thresholds[field], f"baseline.thresholds.{field}"), expected, f"baseline.thresholds.{field}")
    return {
        "sha256": actual_digest,
        "baseline_id": _string(row["baseline_id"], "baseline.baseline_id"),
        "request_identity_sha256": binding["request_identity_sha256"],
        "model": _validate_model(row["model"], "baseline.model"),
        "environment": _validate_environment(
            row["environment"], "baseline.environment"
        ),
        "workload": _validate_workload(row["workload"], "baseline.workload"),
        "metrics": _validate_metrics(row["metrics"], "baseline.metrics"),
        "thresholds": expected_thresholds,
    }


def _validate_candidate(document: Mapping[str, Any]) -> dict[str, Any]:
    row = _closed_object(
        document,
        "candidate",
        {
            "schema_version",
            "baseline_sha256",
            "candidate_id",
            "recorded_at_utc",
            "status",
            "source",
            "model",
            "environment",
            "workload",
            "run_summary",
            "metrics",
            "raw_runs",
        },
    )
    _literal(row["schema_version"], CANDIDATE_SCHEMA, "candidate.schema_version")
    _literal(row["status"], "success", "candidate.status")
    candidate_id = _candidate_id(row["candidate_id"], "candidate.candidate_id")
    recorded = _string(row["recorded_at_utc"], "candidate.recorded_at_utc")
    if UTC_RE.fullmatch(recorded) is None:
        raise InputError("candidate.recorded_at_utc: expected YYYY-MM-DDTHH:MM:SSZ")
    source = _closed_object(
        row["source"],
        "candidate.source",
        {
            "git_commit",
            "git_dirty",
            "source_archive_sha256",
            "profile_binary_sha256",
            "release_binary_sha256",
            "profile_image_sha256",
            "release_image_sha256",
            "semantic_class",
            "correctness_gate_id",
            "correctness_report_sha256",
        },
    )
    commit = _string(source["git_commit"], "candidate.source.git_commit")
    if GIT_RE.fullmatch(commit) is None:
        raise InputError("candidate.source.git_commit: invalid commit")
    _literal(source["git_dirty"], False, "candidate.source.git_dirty")
    _literal(source["semantic_class"], "E0", "candidate.source.semantic_class")
    _literal(
        source["correctness_gate_id"],
        CORRECTNESS_GATE_ID,
        "candidate.source.correctness_gate_id",
    )
    source_result = {
        "git_commit": commit,
        "git_dirty": False,
        "semantic_class": "E0",
        "correctness_gate_id": _string(
            source["correctness_gate_id"], "candidate.source.correctness_gate_id"
        ),
    }
    for field in [
        "source_archive_sha256",
        "profile_binary_sha256",
        "release_binary_sha256",
        "profile_image_sha256",
        "release_image_sha256",
        "correctness_report_sha256",
    ]:
        source_result[field] = _sha256(source[field], f"candidate.source.{field}")
    summary = _closed_object(
        row["run_summary"],
        "candidate.run_summary",
        {
            "independent_runs",
            "warmups_per_run",
            "measured_iterations_per_run",
            "failure_count",
            "dropped_trace_records",
        },
    )
    summary_result = {
        field: _integer(summary[field], f"candidate.run_summary.{field}")
        for field in summary
    }
    if summary_result["independent_runs"] < 5:
        raise InputError("candidate.run_summary.independent_runs: must be >= 5")
    if summary_result["warmups_per_run"] < 5:
        raise InputError("candidate.run_summary.warmups_per_run: must be >= 5")
    if summary_result["measured_iterations_per_run"] < 30:
        raise InputError(
            "candidate.run_summary.measured_iterations_per_run: must be >= 30"
        )
    if summary_result["failure_count"] != 0 or summary_result["dropped_trace_records"] != 0:
        raise InputError("candidate run must have zero failures and dropped records")
    raw_runs = row["raw_runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != 5:
        raise InputError("candidate.raw_runs: must contain exactly five bindings")
    raw_result = []
    for index, value in enumerate(raw_runs):
        binding = _closed_object(
            value,
            f"candidate.raw_runs[{index}]",
            {"pair_index", "run_id", "sha256"},
        )
        raw_result.append(
            {
                "pair_index": _integer(
                    binding["pair_index"],
                    f"candidate.raw_runs[{index}].pair_index",
                    1,
                ),
                "run_id": _string(
                    binding["run_id"], f"candidate.raw_runs[{index}].run_id"
                ),
                "sha256": _sha256(
                    binding["sha256"], f"candidate.raw_runs[{index}].sha256"
                ),
            }
        )
    if sorted(binding["pair_index"] for binding in raw_result) != list(range(1, 6)):
        raise InputError("candidate.raw_runs: pair_index values must be exactly 1..5")
    if len({binding["run_id"] for binding in raw_result}) != 5:
        raise InputError("candidate.raw_runs: run_id values must be unique")
    return {
        "baseline_sha256": _sha256(
            row["baseline_sha256"], "candidate.baseline_sha256"
        ),
        "candidate_id": candidate_id,
        "recorded_at_utc": recorded,
        "source": source_result,
        "model": _validate_model(row["model"], "candidate.model"),
        "environment": _validate_environment(
            row["environment"], "candidate.environment"
        ),
        "workload": _validate_workload(row["workload"], "candidate.workload"),
        "run_summary": summary_result,
        "metrics": _validate_metrics(row["metrics"], "candidate.metrics"),
        "raw_runs": sorted(raw_result, key=lambda binding: binding["pair_index"]),
    }


def _validate_optimization_correctness(
    document: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    row = _closed_object(
        document,
        "correctness_report",
        {
            "schema_version",
            "gate_id",
            "recorded_at_utc",
            "status",
            "semantic_class",
            "source",
            "model",
            "gpu",
            "build",
            "implementations",
            "tests",
        },
    )
    _literal(row["schema_version"], 1, "correctness_report.schema_version")
    _literal(row["gate_id"], CORRECTNESS_GATE_ID, "correctness_report.gate_id")
    _literal(row["status"], "passed", "correctness_report.status")
    _literal(row["semantic_class"], "E0", "correctness_report.semantic_class")
    recorded = _string(row["recorded_at_utc"], "correctness_report.recorded_at_utc")
    if UTC_RE.fullmatch(recorded) is None:
        raise InputError(
            "correctness_report.recorded_at_utc: expected YYYY-MM-DDTHH:MM:SSZ"
        )

    source = _closed_object(
        row["source"],
        "correctness_report.source",
        {"git_commit", "git_dirty", "archive_sha256"},
    )
    expected_source = {
        "git_commit": candidate["source"]["git_commit"],
        "git_dirty": False,
        "archive_sha256": candidate["source"]["source_archive_sha256"],
    }
    if source != expected_source:
        raise InputError("correctness_report.source: candidate source binding mismatch")

    model = _closed_object(
        row["model"],
        "correctness_report.model",
        {
            "model_id",
            "revision",
            "dtype",
            "manifest_sha256",
            "weights_sha256",
            "tokenizer_sha256",
        },
    )
    expected_model = {
        "model_id": candidate["model"]["model_id"],
        "revision": candidate["model"]["model_revision"],
        "dtype": candidate["model"]["dtype"],
        "weights_sha256": candidate["model"]["weights_sha256"],
        "tokenizer_sha256": candidate["model"]["tokenizer_sha256"],
    }
    for field, expected in expected_model.items():
        _literal(model[field], expected, f"correctness_report.model.{field}")
    _sha256(model["manifest_sha256"], "correctness_report.model.manifest_sha256")

    environment = candidate["environment"]
    gpu = _closed_object(
        row["gpu"],
        "correctness_report.gpu",
        {
            "model",
            "uuid",
            "pci_bus_id",
            "compute_capability",
            "vram_mib",
            "driver_version",
        },
    )
    for field in ("model", "pci_bus_id"):
        _string(gpu[field], f"correctness_report.gpu.{field}")
    _integer(gpu["vram_mib"], "correctness_report.gpu.vram_mib", 1)
    for report_field, environment_field in (
        ("uuid", "gpu_uuid"),
        ("compute_capability", "compute_capability"),
        ("driver_version", "driver_version"),
    ):
        _literal(
            gpu[report_field],
            environment[environment_field],
            f"correctness_report.gpu.{report_field}",
        )

    build = _closed_object(
        row["build"],
        "correctness_report.build",
        {
            "rustc",
            "cuda_toolkit",
            "cuda_architecture",
            "container_image_sha256",
            "network",
            "cargo_locked",
            "cargo_offline",
        },
    )
    expected_build = {
        "rustc": "1.85.0",
        "cuda_toolkit": environment["cuda_toolkit_version"],
        "cuda_architecture": environment["cuda_architecture"],
        "container_image_sha256": candidate["source"]["profile_image_sha256"],
        "network": "none",
        "cargo_locked": True,
        "cargo_offline": True,
    }
    if build != expected_build:
        raise InputError(
            "correctness_report.build: reviewed offline build binding mismatch"
        )

    implementations = _closed_object(
        row["implementations"],
        "correctness_report.implementations",
        {"baseline", "candidate", "residual_rmsnorm", "rollback"},
    )
    expected_implementations = {
        "baseline": "per-operation",
        "candidate": "iteration-batch",
        "residual_rmsnorm": "separate",
        "rollback": "--execution-completion per-operation",
    }
    if implementations != expected_implementations:
        raise InputError("correctness_report.implementations: exact E0 pair mismatch")

    tests = row["tests"]
    if not isinstance(tests, list) or len(tests) != 6:
        raise InputError("correctness_report.tests: expected exactly six checks")
    tests_by_id: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(tests):
        test = _closed_object(
            value,
            f"correctness_report.tests[{index}]",
            set(value) if isinstance(value, dict) else set(),
        )
        test_id = _string(test.get("id"), f"correctness_report.tests[{index}].id")
        if test_id in tests_by_id:
            raise InputError(f"correctness_report.tests[{index}].id: duplicate check")
        tests_by_id[test_id] = test
    expected_ids = {
        "cuda-compile-only",
        "workspace-all-features-all-targets",
        "command-batch-lifecycle",
        "command-batch-resource-ledger",
        "smollm2-multi-step-greedy-exact",
        "fixed37-production-batch-e0",
    }
    if set(tests_by_id) != expected_ids:
        raise InputError("correctness_report.tests: exact check inventory mismatch")

    for test_id in ("cuda-compile-only", "workspace-all-features-all-targets"):
        test = _closed_object(
            tests_by_id[test_id],
            f"correctness_report.tests.{test_id}",
            {"id", "log_sha256", "result"},
        )
        _literal(
            test["result"],
            "passed",
            f"correctness_report.tests.{test_id}.result",
        )
        _sha256(
            test["log_sha256"],
            f"correctness_report.tests.{test_id}.log_sha256",
        )

    lifecycle = _closed_object(
        tests_by_id["command-batch-lifecycle"],
        "correctness_report.tests.command-batch-lifecycle",
        {"id", "log_sha256", "result", "one_shot_finish", "drop_restores_stream"},
    )
    for field in ("one_shot_finish", "drop_restores_stream"):
        _literal(
            lifecycle[field],
            True,
            f"correctness_report.tests.command-batch-lifecycle.{field}",
        )
    _literal(
        lifecycle["result"],
        "passed",
        "correctness_report.tests.command-batch-lifecycle.result",
    )
    _sha256(
        lifecycle["log_sha256"],
        "correctness_report.tests.command-batch-lifecycle.log_sha256",
    )

    ledger = _closed_object(
        tests_by_id["command-batch-resource-ledger"],
        "correctness_report.tests.command-batch-resource-ledger",
        {
            "id",
            "log_sha256",
            "result",
            "queued_chain_raw_byte_mismatches",
            "cuda_live_allocation_delta",
            "owner_close_live_allocation_count",
            "validation_fail_closed",
            "stream_reuse_after_finish",
        },
    )
    _literal(
        ledger["result"],
        "passed",
        "correctness_report.tests.command-batch-resource-ledger.result",
    )
    _sha256(
        ledger["log_sha256"],
        "correctness_report.tests.command-batch-resource-ledger.log_sha256",
    )
    for field in (
        "queued_chain_raw_byte_mismatches",
        "cuda_live_allocation_delta",
        "owner_close_live_allocation_count",
    ):
        _literal(
            ledger[field],
            0,
            f"correctness_report.tests.command-batch-resource-ledger.{field}",
        )
    for field in ("validation_fail_closed", "stream_reuse_after_finish"):
        _literal(
            ledger[field],
            True,
            f"correctness_report.tests.command-batch-resource-ledger.{field}",
        )

    parity = _closed_object(
        tests_by_id["smollm2-multi-step-greedy-exact"],
        "correctness_report.tests.smollm2-multi-step-greedy-exact",
        {
            "id",
            "log_sha256",
            "result",
            "decode_steps",
            "committed_iterations",
            "generated_token_ids",
            "raw_logit_mismatches",
            "token_id_mismatches",
            "cuda_live_allocation_delta",
            "owner_close_live_allocation_count",
        },
    )
    _literal(
        parity["result"],
        "passed",
        "correctness_report.tests.smollm2-multi-step-greedy-exact.result",
    )
    _sha256(
        parity["log_sha256"],
        "correctness_report.tests.smollm2-multi-step-greedy-exact.log_sha256",
    )
    for field in ("decode_steps", "committed_iterations"):
        _literal(
            parity[field],
            16,
            f"correctness_report.tests.smollm2-multi-step-greedy-exact.{field}",
        )

    fixed37 = _closed_object(
        tests_by_id["fixed37-production-batch-e0"],
        "correctness_report.tests.fixed37-production-batch-e0",
        {
            "id",
            "result",
            "gate_id",
            "fixture_sha256",
            "generated_token_ids_sha256",
            "cases",
            "compared_steps",
            "exact_window",
            "fixed_profile",
            "canonical_profile",
            "residual_rmsnorm",
            "execution_completion",
            "fixed_prefill_raw_logit_mismatches",
            "fixed_cached_growing_token_id_mismatches",
            "fixed_cached_growing_cosine_min",
            "fixed_cached_growing_max_abs_max",
            "fixed_cached_growing_mean_abs_max",
            "fixed_cached_growing_worst_cosine",
            "fixed_cached_growing_worst_max_abs",
            "fixed_cached_growing_worst_mean_abs",
            "fixed_cached_growing_threshold_violations",
            "fixed_golden_token_id_mismatches",
            "canonical_golden_token_id_mismatches",
            "cuda_live_allocation_delta",
            "owner_close_live_allocation_count",
            "compile_command_id",
            "execute_command_id",
            "compile_log_sha256",
            "test_binary_sha256",
            "log_sha256",
        },
    )
    expected_fixed37 = {
        "id": "fixed37-production-batch-e0",
        "result": "passed",
        "gate_id": FIXED37_PRODUCTION_BATCH_GATE_ID,
        "fixture_sha256": FIXED37_GOLDEN_FIXTURE_SHA256,
        "generated_token_ids_sha256": FIXED37_GOLDEN_TOKEN_IDS_SHA256,
        "cases": 31,
        "compared_steps": 481,
        "exact_window": 16,
        "fixed_profile": "fixed-contiguous-37-balanced-v1",
        "canonical_profile": "canonical-v1",
        "residual_rmsnorm": "separate",
        "execution_completion": "iteration-batch",
        "fixed_prefill_raw_logit_mismatches": 0,
        "fixed_cached_growing_token_id_mismatches": 0,
        "fixed_cached_growing_cosine_min": FIXED37_CACHED_GROWING_COSINE_MIN,
        "fixed_cached_growing_max_abs_max": FIXED37_CACHED_GROWING_MAX_ABS_MAX,
        "fixed_cached_growing_mean_abs_max": FIXED37_CACHED_GROWING_MEAN_ABS_MAX,
        "fixed_cached_growing_threshold_violations": 0,
        "fixed_golden_token_id_mismatches": 0,
        "canonical_golden_token_id_mismatches": 0,
        "cuda_live_allocation_delta": 0,
        "owner_close_live_allocation_count": 0,
        "compile_command_id": "compile-fixed37-production-batch-e0",
        "execute_command_id": "fixed37-production-batch-e0",
    }
    for field, expected in expected_fixed37.items():
        _literal(
            fixed37[field],
            expected,
            f"correctness_report.tests.fixed37-production-batch-e0.{field}",
        )
    for field in (
        "compile_log_sha256",
        "test_binary_sha256",
        "log_sha256",
    ):
        _sha256(
            fixed37[field],
            f"correctness_report.tests.fixed37-production-batch-e0.{field}",
        )
    fixed37_worst_cosine = _number(
        fixed37["fixed_cached_growing_worst_cosine"],
        "correctness_report.tests.fixed37-production-batch-e0.fixed_cached_growing_worst_cosine",
    )
    fixed37_worst_max_abs = _number(
        fixed37["fixed_cached_growing_worst_max_abs"],
        "correctness_report.tests.fixed37-production-batch-e0.fixed_cached_growing_worst_max_abs",
    )
    fixed37_worst_mean_abs = _number(
        fixed37["fixed_cached_growing_worst_mean_abs"],
        "correctness_report.tests.fixed37-production-batch-e0.fixed_cached_growing_worst_mean_abs",
    )
    if (
        fixed37_worst_cosine < FIXED37_CACHED_GROWING_COSINE_MIN
        or fixed37_worst_max_abs > FIXED37_CACHED_GROWING_MAX_ABS_MAX
        or fixed37_worst_mean_abs > FIXED37_CACHED_GROWING_MEAN_ABS_MAX
    ):
        raise InputError(
            "correctness_report.tests.fixed37-production-batch-e0: "
            "cached/growing metrics exceed the immutable E0 bounds"
        )
    _literal(
        parity["generated_token_ids"],
        OPTIMIZATION_GOLDEN_TOKEN_IDS,
        "correctness_report.tests.smollm2-multi-step-greedy-exact.generated_token_ids",
    )
    for field in (
        "raw_logit_mismatches",
        "token_id_mismatches",
        "cuda_live_allocation_delta",
        "owner_close_live_allocation_count",
    ):
        _literal(
            parity[field],
            0,
            f"correctness_report.tests.smollm2-multi-step-greedy-exact.{field}",
        )


def _check(name: str, observed: float, operator: str, limit: float) -> dict[str, Any]:
    passed = observed <= limit if operator == "<=" else observed >= limit
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "operator": operator,
        "limit": limit,
    }


def _empty_report() -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "error",
        "passed": False,
        "baseline": None,
        "candidate": None,
        "ratios": None,
        "checks": [],
        "errors": [],
    }


def derive_raw_run_payloads(
    payloads: Sequence[tuple[str, bytes]],
) -> dict[str, Any]:
    """Derive candidate fields from five immutable native-profile payloads.

    This is deliberately independent of a self-asserted candidate document so
    the release evidence producer can construct that document from the raw
    measurements and the checker can subsequently replay the same derivation.
    """

    if len(payloads) != 5:
        raise InputError(
            f"candidate: expected exactly 5 independent run files, got {len(payloads)}"
        )
    loaded: list[tuple[str, bytes, dict[str, Any], str]] = []
    try:
        for label, raw in payloads:
            if not isinstance(label, str) or not label:
                raise InputError("candidate: raw run label must be a non-empty string")
            if not isinstance(raw, bytes):
                raise InputError(f"{label}: raw native profile payload must be bytes")
            if len(raw) > native_profile.MAX_EVIDENCE_BYTES:
                raise InputError(
                    f"{label}: exceeds the raw native profile evidence bound"
                )
            try:
                run = json.loads(
                    raw,
                    object_pairs_hook=_pairs_no_duplicates,
                    parse_constant=_reject_nonfinite,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, InputError) as error:
                raise InputError(f"{label}: invalid raw native profile JSON: {error}") from error
            if not isinstance(run, dict):
                raise InputError(f"{label}: raw native profile root must be an object")
            native_profile._validate_run(run, label)
            if run["role"] != "candidate":
                raise InputError(
                    f"{label}.role: expected 'candidate', got {run['role']!r}"
                )
            loaded.append((label, raw, run, _digest_bytes(raw)))
        if sorted(run["pair_index"] for _, _, run, _ in loaded) != list(range(1, 6)):
            raise InputError("candidate: pair_index values must be exactly 1..5")
        if len({run["run_id"] for _, _, run, _ in loaded}) != 5:
            raise InputError("candidate: raw run_id values must be unique")
        loaded.sort(key=lambda row: row[2]["pair_index"])
        runs = [run for _, _, run, _ in loaded]
        source = native_profile._require_equal(
            [run["source"] for run in runs], "release candidate raw source"
        )
        environment = native_profile._require_equal(
            [run["environment"] for run in runs],
            "release candidate raw environment",
        )
        workload = native_profile._require_equal(
            [run["workload"] for run in runs], "release candidate raw workload"
        )
        request_identity = native_profile._require_equal(
            [native_profile._request_identity(run) for run in runs],
            "release candidate raw request identities",
        )
    except native_profile.ComparabilityError as error:
        raise ComparabilityError(str(error)) from error
    except native_profile.InputError as error:
        raise InputError(str(error)) from error

    raw_model = {
        "model_id": workload["model_id"],
        "model_revision": workload["model_revision"],
        "dtype": workload["dtype"],
        "weights_sha256": workload["weights_sha256"],
        "tokenizer_sha256": workload["tokenizer_sha256"],
    }
    raw_environment = {
        "environment_id": environment["host"]["environment_id"],
        "gpu_uuid": environment["gpu"]["uuid"],
        "compute_capability": environment["gpu"]["compute_capability"],
        "driver_version": environment["software"]["nvidia_driver_version"],
        "cuda_runtime_version": environment["software"]["cuda_runtime_version"],
        "cuda_toolkit_version": environment["software"]["cuda_toolkit_version"],
        "cuda_architecture": environment["gpu"]["compute_capability"].replace(
            ".", ""
        ),
    }
    raw_workload = {
        "workload_id": workload["workload_id"],
        "concurrency": workload["concurrency"],
        "prompt_tokens": workload["prompt_tokens"],
        "output_tokens": workload["output_tokens"],
        "warmups_per_run": workload["warmups"],
        "measured_iterations_per_run": workload["measured_iterations"],
        "independent_runs": len(runs),
        "sampling": workload["sampling_id"],
        "execution_completion": "iteration-batch",
        "residual_rmsnorm": "separate",
    }
    derived_summary = {
        "independent_runs": len(runs),
        "warmups_per_run": workload["warmups"],
        "measured_iterations_per_run": workload["measured_iterations"],
        "failure_count": sum(run["failure_count"] for run in runs),
        "dropped_trace_records": sum(
            run["trace"]["dropped_records"] for run in runs
        ),
    }
    request_rows = [request for run in runs for request in run["requests"]]
    derived_metrics = {
        "ttft_p95_ms": native_profile.r7(
            [request["ttft_ms"] for request in request_rows], 0.95
        ),
        "tpot_p95_ms": native_profile.r7(
            [request["tpot_ms"] for request in request_rows], 0.95
        ),
        "e2e_median_ms": native_profile.r7(
            [request["e2e_ms"] for request in request_rows], 0.50
        ),
        "throughput_median_output_tokens_per_second": native_profile.r7(
            [native_profile._throughput(run) for run in runs], 0.50
        ),
    }
    return {
        "runs": runs,
        "payloads": [
            (f"candidate-{run['pair_index']}.json", raw)
            for _, raw, run, _ in loaded
        ],
        "source": source,
        "model": raw_model,
        "environment": raw_environment,
        "workload": raw_workload,
        "run_summary": derived_summary,
        "metrics": derived_metrics,
        "raw_runs": [
            {
                "pair_index": run["pair_index"],
                "run_id": run["run_id"],
                "sha256": actual_digest,
            }
            for _, _, run, actual_digest in loaded
        ],
        "request_identity_sha256": native_profile._sha256_json(request_identity),
    }


def _require_request_identity_sha256(
    derived: Mapping[str, Any], expected: str, path: str
) -> None:
    actual = derived.get("request_identity_sha256")
    if actual != expected:
        raise ComparabilityError(
            f"{path}: canonical native request identity differs from the reviewed PR15 baseline: "
            f"{actual} != {expected}"
        )


def validate_raw_run_payloads(
    payloads: Sequence[tuple[str, bytes]], candidate: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    """Validate five raw candidate runs supplied as immutable byte payloads."""

    derived = derive_raw_run_payloads(payloads)
    loaded = derived["payloads"]
    runs = derived["runs"]
    declared_by_pair = {
        binding["pair_index"]: binding for binding in candidate["raw_runs"]
    }
    for (path, _), run, binding in zip(
        loaded, runs, derived["raw_runs"], strict=True
    ):
        pair_index = run["pair_index"]
        if declared_by_pair.get(pair_index) != binding:
            raise InputError(f"{path}: raw run binding does not match file contents")

    candidate_source = candidate["source"]
    expected_source = {
        "git_commit": candidate_source["git_commit"],
        "git_dirty": False,
        "executable_sha256": candidate_source["profile_binary_sha256"],
        "semantic_class": "E0",
        "correctness_gate_id": candidate_source["correctness_gate_id"],
        "correctness_report_sha256": candidate_source[
            "correctness_report_sha256"
        ],
    }
    for field, expected in expected_source.items():
        if derived["source"][field] != expected:
            raise InputError(
                f"raw source.{field} does not match candidate source binding"
            )
    if derived["source"]["runtime_flag"] != {
        "name": "execution_completion",
        "value": "iteration-batch",
    }:
        raise ComparabilityError(
            "raw source.runtime_flag must select execution_completion=iteration-batch"
        )

    for name in ("model", "environment", "workload"):
        raw_value = derived[name]
        if candidate[name] != raw_value:
            raise ComparabilityError(
                f"candidate {name} does not match its raw native profile runs"
            )
    if derived["runs"][0]["environment"]["software"][
        "container_image_sha256"
    ] != candidate_source["profile_image_sha256"]:
        raise InputError(
            "raw environment producer image does not match profile_image_sha256"
        )

    derived_summary = derived["run_summary"]
    derived_metrics = derived["metrics"]
    if candidate["run_summary"] != derived_summary:
        raise InputError("candidate.run_summary does not equal raw-derived summary")
    if candidate["metrics"] != derived_metrics:
        raise InputError("candidate.metrics do not equal raw-derived R7 metrics")
    return runs, derived_summary, derived_metrics


def _load_raw_runs(
    paths: Sequence[Path | str], candidate: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    return validate_raw_run_payloads(_read_raw_run_paths(paths), candidate)


def _read_raw_run_paths(
    paths: Sequence[Path | str],
) -> list[tuple[str, bytes]]:
    payloads: list[tuple[str, bytes]] = []
    for value in paths:
        path = Path(value)
        raw = _read_bounded_regular(
            path,
            f"raw native profile run {path}",
            native_profile.MAX_EVIDENCE_BYTES,
        )
        payloads.append((str(path), raw))
    return payloads


def _strict_json_payload(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InputError) as error:
        raise InputError(f"{label}: invalid strict UTF-8 JSON: {error}") from error


def _read_bounded_regular(path: Path, label: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        raw, _digest, _metadata = _stable_fd_snapshot(descriptor, label, maximum)
        return raw
    except InputError:
        raise
    except OSError as error:
        raise InputError(f"{label}: cannot open stable snapshot: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _runner_sha256s(payloads: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{_digest_bytes(payloads[name])}  {name}\n"
        for name in RUNNER_RECEIPT_FILES
        if name != "SHA256SUMS"
    ).encode("ascii")


def _runner_forbidden_environment(names: Sequence[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name in RUNNER_FORBIDDEN_ENV_NAMES or name.startswith("BASH_FUNC_")
    )


def _runner_environment(value: Any, path: str) -> dict[str, str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InputError(f"{path}: must be an array of NAME=value strings")
    result: dict[str, str] = {}
    for item in value:
        name, separator, setting = item.partition("=")
        if not separator or not name or name in result:
            raise InputError(f"{path}: malformed or duplicate environment variable")
        result[name] = setting
    forbidden = _runner_forbidden_environment(list(result))
    if forbidden:
        raise InputError(f"{path}: forbidden environment variables: {forbidden}")
    return result


def _runner_image(raw: bytes, label: str, expected_id: str) -> dict[str, Any]:
    document = _strict_json_payload(raw, label)
    if not isinstance(document, list) or len(document) != 1:
        raise InputError(f"{label}: must contain exactly one image object")
    image = document[0]
    if not isinstance(image, dict):
        raise InputError(f"{label}[0]: must be an object")
    if image.get("Id") != expected_id:
        raise InputError(f"{label}.Id: does not equal the trusted image ID")
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        raise InputError(f"{label}: image platform must be linux/amd64")
    config = image.get("Config")
    if not isinstance(config, dict) or config.get("WorkingDir") != "/workspace":
        raise InputError(f"{label}.Config: reviewed /workspace image config required")
    environment = _runner_environment(config.get("Env"), f"{label}.Config.Env")
    if environment.get("CUDA_VERSION") != "12.8.1":
        raise InputError(f"{label}.Config.Env: CUDA_VERSION must equal 12.8.1")
    labels = config.get("Labels")
    if not isinstance(labels, dict) or not all(
        isinstance(name, str) and name and isinstance(value, str)
        for name, value in labels.items()
    ):
        raise InputError(f"{label}.Config.Labels: exact string map required")
    return {"environment": environment, "labels": dict(labels)}


def _runner_preflight(raw: bytes, label: str, revision: str) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise InputError(f"{label}: must be strict UTF-8") from error
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise InputError(f"{label}:{line_number}: malformed or duplicate key")
        values[key] = value
    if set(values) != RUNNER_PREFLIGHT_FIELDS:
        raise InputError(f"{label}: exact reviewed field inventory required")
    for name, expected in RUNNER_FIXED_PREFLIGHT.items():
        if values[name] != expected:
            raise InputError(f"{label}.{name}: expected {expected!r}")
    if values["git_revision"] != revision:
        raise InputError(f"{label}.git_revision: candidate revision mismatch")
    for name, maximum in (("memory_used_mib", 256), ("temperature_c", 50)):
        value = values[name]
        if re.fullmatch(r"0|[1-9][0-9]*", value) is None or int(value) > maximum:
            raise InputError(f"{label}.{name}: outside the reviewed bound")
    available = values["staging_available_bytes"]
    if re.fullmatch(r"0|[1-9][0-9]*", available) is None:
        raise InputError(f"{label}.staging_available_bytes: invalid integer")
    if int(available) < int(RUNNER_FIXED_PREFLIGHT["staging_minimum_bytes"]):
        raise InputError(f"{label}.staging_available_bytes: below reviewed minimum")
    return values


_RUNNER_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.([0-9]{1,9}))?Z$"
)


def _runner_capture_id(supervisor_token: str, pair_index: int) -> str:
    return hashlib.sha256(
        f"{supervisor_token}:{pair_index}\n".encode("ascii")
    ).hexdigest()


def _runner_run_id(revision: str, capture_id: str, pair_index: int) -> str:
    return (
        f"pr16-iteration-batch-{revision[:12]}-{capture_id}-pair{pair_index}"
    )


def _runner_timestamp_ns(value: Any, path: str) -> int:
    text = _string(value, path)
    match = _RUNNER_TIMESTAMP_RE.fullmatch(text)
    if match is None or text == RUNNER_ZERO_TIME:
        raise InputError(f"{path}: non-zero RFC3339 UTC timestamp required")
    try:
        whole = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise InputError(f"{path}: invalid RFC3339 UTC timestamp") from error
    fraction = (match.group(2) or "").ljust(9, "0")
    return int(whole.timestamp()) * 1_000_000_000 + int(fraction or "0")


def _runner_execution(value: Any, label: str) -> dict[str, Any]:
    row = _closed_object(
        value,
        label,
        {
            "schema_version",
            "pair_index",
            "capture_id",
            "container_id",
            "run_id",
            "candidate_recorded_at_utc",
            "docker",
            "sha256",
        },
    )
    _literal(row["schema_version"], RUNNER_EXECUTION_SCHEMA, f"{label}.schema_version")
    pair_index = _integer(row["pair_index"], f"{label}.pair_index", 1)
    if pair_index > 5:
        raise InputError(f"{label}.pair_index: must be in 1..5")
    capture_id = _sha256(row["capture_id"], f"{label}.capture_id")
    container_id = _string(row["container_id"], f"{label}.container_id")
    if SHA256_RE.fullmatch(container_id) is None:
        raise InputError(f"{label}.container_id: lowercase 64-character ID required")
    _string(row["run_id"], f"{label}.run_id")
    _runner_timestamp_ns(
        row["candidate_recorded_at_utc"], f"{label}.candidate_recorded_at_utc"
    )
    docker = _closed_object(
        row["docker"],
        f"{label}.docker",
        {
            "created_at_utc",
            "started_at_utc",
            "finished_at_utc",
            "exit_code",
            "oom_killed",
        },
    )
    for name in ("created_at_utc", "started_at_utc", "finished_at_utc"):
        _runner_timestamp_ns(docker[name], f"{label}.docker.{name}")
    _literal(docker["exit_code"], 0, f"{label}.docker.exit_code")
    _literal(docker["oom_killed"], False, f"{label}.docker.oom_killed")
    digests = _closed_object(
        row["sha256"],
        f"{label}.sha256",
        {
            "preflight",
            "candidate",
            "gpu_monitor",
            "container_inspect_before",
            "container_inspect_after",
        },
    )
    for name, digest in digests.items():
        _sha256(digest, f"{label}.sha256.{name}")
    return dict(row)


def _runner_execution_payload(raw: bytes, label: str) -> dict[str, Any]:
    return _runner_execution(_strict_json_payload(raw, label), label)


def _validate_runner_execution_timeline(
    execution: Mapping[str, Any], *, label: str
) -> None:
    docker = execution["docker"]
    created = _runner_timestamp_ns(docker["created_at_utc"], f"{label}.docker.created_at_utc")
    started = _runner_timestamp_ns(docker["started_at_utc"], f"{label}.docker.started_at_utc")
    recorded = _runner_timestamp_ns(
        execution["candidate_recorded_at_utc"],
        f"{label}.candidate_recorded_at_utc",
    )
    finished = _runner_timestamp_ns(docker["finished_at_utc"], f"{label}.docker.finished_at_utc")
    if not created <= started <= recorded <= finished:
        raise InputError(
            f"{label}: require Created <= StartedAt <= candidate recorded_at <= FinishedAt"
        )


def _runner_manifest(
    raw: bytes,
    *,
    image_environment: Mapping[str, str],
    image_labels: Mapping[str, str],
) -> dict[str, Any]:
    document = _strict_json_payload(raw, "runner-manifest.json")
    root = _closed_object(
        document,
        "runner-manifest",
        {"schema_version", "candidate", "runner", "container", "executions"},
    )
    _literal(
        root["schema_version"], RUNNER_MANIFEST_SCHEMA, "runner-manifest.schema_version"
    )
    candidate = _closed_object(
        root["candidate"],
        "runner-manifest.candidate",
        {
            "source_revision",
            "source_archive_sha256",
            "profile_binary_sha256",
            "model_tree_sha256",
            "optimizer_correctness_report_sha256",
            "optimizer_image_id",
        },
    )
    revision = _string(candidate["source_revision"], "runner-manifest.candidate.source_revision")
    if GIT_RE.fullmatch(revision) is None or len(revision) != 40:
        raise InputError("runner-manifest.candidate.source_revision: expected 40-character Git SHA")
    for name in (
        "source_archive_sha256",
        "profile_binary_sha256",
        "model_tree_sha256",
        "optimizer_correctness_report_sha256",
    ):
        _sha256(candidate[name], f"runner-manifest.candidate.{name}")
    image_id = _string(candidate["optimizer_image_id"], "runner-manifest.candidate.optimizer_image_id")
    if not image_id.startswith("sha256:"):
        raise InputError("runner-manifest.candidate.optimizer_image_id: expected sha256 digest")
    _sha256(image_id.removeprefix("sha256:"), "runner-manifest.candidate.optimizer_image_id")

    runner = _closed_object(
        root["runner"],
        "runner-manifest.runner",
        {"revision", "host_script_sha256", "inner_script_sha256", "tools"},
    )
    _literal(runner["revision"], revision, "runner-manifest.runner.revision")
    repository = Path(__file__).resolve().parents[2]
    script_bindings = {
        "host_script_sha256": repository / "ci" / "run_remote_release_performance.sh",
        "inner_script_sha256": repository / "ci" / "release" / "run_release_performance_once.sh",
    }
    for name, path in script_bindings.items():
        declared = _sha256(runner[name], f"runner-manifest.runner.{name}")
        actual = _digest_bytes(
            _read_bounded_regular(path, f"reviewed {name}", 2 * 1024 * 1024)
        )
        if declared != actual:
            raise InputError(f"runner-manifest.runner.{name}: does not match reviewed source")
    tools = runner["tools"]
    if not isinstance(tools, dict) or set(tools) != RUNNER_REQUIRED_TOOLS:
        raise InputError("runner-manifest.runner.tools: exact trusted tool inventory required")
    for name in sorted(tools):
        tool = _closed_object(
            tools[name], f"runner-manifest.runner.tools.{name}", {"path", "sha256"}
        )
        expected_tool = RUNNER_REVIEWED_TOOLS[name]
        _literal(
            tool["path"],
            expected_tool["path"],
            f"runner-manifest.runner.tools.{name}.path",
        )
        _literal(
            tool["sha256"],
            expected_tool["sha256"],
            f"runner-manifest.runner.tools.{name}.sha256",
        )

    container = _closed_object(
        root["container"],
        "runner-manifest.container",
        {
            "entrypoint",
            "cmd",
            "environment",
            "read_only_mount_sources",
            "evidence_mount_sources",
            "workspace_volume_names",
            "supervisor_label",
            "labels",
        },
    )
    _literal(container["entrypoint"], RUNNER_CONTAINER_ENTRYPOINT, "runner-manifest.container.entrypoint")
    _literal(container["cmd"], RUNNER_CONTAINER_CMD, "runner-manifest.container.cmd")
    supervisor_label = _closed_object(
        container["supervisor_label"],
        "runner-manifest.container.supervisor_label",
        {"name", "value"},
    )
    _literal(
        supervisor_label["name"],
        RUNNER_SUPERVISOR_LABEL,
        "runner-manifest.container.supervisor_label.name",
    )
    supervisor_token = _sha256(
        supervisor_label["value"],
        "runner-manifest.container.supervisor_label.value",
    )
    labels = container["labels"]
    if not isinstance(labels, dict) or not all(
        isinstance(name, str) and name and isinstance(value, str)
        for name, value in labels.items()
    ):
        raise InputError("runner-manifest.container.labels: exact string map required")
    expected_labels = {
        **image_labels,
        RUNNER_SUPERVISOR_LABEL: supervisor_token,
    }
    if not _exact_json_value(labels, expected_labels):
        raise InputError(
            "runner-manifest.container.labels: must equal image labels plus the exact supervisor label"
        )
    environment = container["environment"]
    if not isinstance(environment, dict) or not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in environment.items()
    ):
        raise InputError("runner-manifest.container.environment: must be a string map")
    fixed_overrides = {
        "RUSTINFER_PERF_SOURCE_REVISION": revision,
        "RUSTINFER_PERF_SOURCE_ARCHIVE_SHA256": candidate["source_archive_sha256"],
        "RUSTINFER_PERF_PROFILE_BINARY_SHA256": candidate["profile_binary_sha256"],
        "RUSTINFER_PERF_OPTIMIZER_REPORT_SHA256": candidate["optimizer_correctness_report_sha256"],
        "RUSTINFER_PERF_OPTIMIZER_IMAGE_SHA256": image_id.removeprefix("sha256:"),
        "RUSTINFER_PERF_MODEL_TREE_SHA256": candidate["model_tree_sha256"],
        "RUSTINFER_PERF_PAIR_INDEX": "{pair_index}",
        "RUSTINFER_PERF_CAPTURE_ID": "{capture_id}",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        **RUNNER_PROXY_ENV,
    }
    expected_environment = {**image_environment, **fixed_overrides}
    if not _exact_json_value(environment, expected_environment):
        raise InputError("runner-manifest.container.environment: not image base env plus exact overrides")
    forbidden = _runner_forbidden_environment(list(environment))
    if forbidden:
        raise InputError(
            "runner-manifest.container.environment: forbidden control-plane "
            f"overrides: {forbidden}"
        )
    read_only_sources = container["read_only_mount_sources"]
    expected_destinations = {
        "/input/source.tar",
        "/input/rustinfer-profile",
        "/input/optimizer-correctness-report.json",
        "/model",
    }
    if not isinstance(read_only_sources, dict) or set(read_only_sources) != expected_destinations:
        raise InputError("runner-manifest.container.read_only_mount_sources: exact inventory required")
    if not all(isinstance(value, str) and value.startswith("/") for value in read_only_sources.values()):
        raise InputError("runner-manifest.container.read_only_mount_sources: absolute sources required")
    evidence_sources = container["evidence_mount_sources"]
    volume_names = container["workspace_volume_names"]
    for value, label in (
        (evidence_sources, "evidence_mount_sources"),
        (volume_names, "workspace_volume_names"),
    ):
        if not isinstance(value, list) or len(value) != 5 or not all(
            isinstance(item, str) and item for item in value
        ) or len(set(value)) != 5:
            raise InputError(f"runner-manifest.container.{label}: five distinct values required")
    if not all(value.startswith("/") for value in evidence_sources):
        raise InputError("runner-manifest.container.evidence_mount_sources: absolute sources required")
    executions = root["executions"]
    if not isinstance(executions, list) or len(executions) != 5:
        raise InputError("runner-manifest.executions: exactly five receipts required")
    parsed_executions = [
        _runner_execution(value, f"runner-manifest.executions[{index - 1}]")
        for index, value in enumerate(executions, 1)
    ]
    if [value["pair_index"] for value in parsed_executions] != list(range(1, 6)):
        raise InputError("runner-manifest.executions: canonical pair order 1..5 required")
    for pair_index, execution in enumerate(parsed_executions, 1):
        expected_capture_id = _runner_capture_id(supervisor_token, pair_index)
        _literal(
            execution["capture_id"],
            expected_capture_id,
            f"runner-manifest.executions[{pair_index - 1}].capture_id",
        )
        _literal(
            execution["run_id"],
            _runner_run_id(revision, expected_capture_id, pair_index),
            f"runner-manifest.executions[{pair_index - 1}].run_id",
        )
    if len({value["container_id"] for value in parsed_executions}) != 5:
        raise InputError("runner-manifest.executions: five distinct container IDs required")
    return dict(document)


def _runner_container_document(raw: bytes, label: str) -> Mapping[str, Any]:
    document = _strict_json_payload(raw, label)
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        raise InputError(f"{label}: must contain exactly one container object")
    return document[0]


def _runner_gpu_monitor(
    raw: bytes,
    label: str,
    *,
    expected_capture_id: str | None = None,
    expected_container_id: str | None = None,
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise InputError(f"{label}: invalid strict UTF-8 CSV: {error}") from error
    if not rows or tuple(rows[0]) != RUNNER_GPU_MONITOR_HEADER:
        raise InputError(f"{label}: exact monitor header required")
    samples = rows[1:]
    if len(samples) < 3:
        raise InputError(f"{label}: pre-start, running, and post-exit samples required")
    stages = [row[2] if len(row) > 2 else "" for row in samples]
    if stages[0] != "pre_start" or stages[-1] != "post_exit" or any(
        stage != "running" for stage in stages[1:-1]
    ):
        raise InputError(f"{label}: exact pre_start/running+/post_exit order required")
    observed_container_process = False
    for index, row in enumerate(samples):
        path = f"{label}:sample[{index}]"
        if len(row) != len(RUNNER_GPU_MONITOR_HEADER):
            raise InputError(f"{path}: exact column count required")
        (
            capture_id,
            container_id,
            _stage,
            sample_index,
            power_limit,
            graphics_clock,
            memory_clock,
            temperature,
            memory_used,
            processes,
        ) = row
        if SHA256_RE.fullmatch(capture_id) is None:
            raise InputError(f"{path}.capture_id: lowercase SHA-256 required")
        if SHA256_RE.fullmatch(container_id) is None:
            raise InputError(f"{path}.container_id: lowercase container ID required")
        if expected_capture_id is not None and capture_id != expected_capture_id:
            raise InputError(f"{path}.capture_id: execution receipt mismatch")
        if expected_container_id is not None and container_id != expected_container_id:
            raise InputError(f"{path}.container_id: execution receipt mismatch")
        if sample_index != str(index):
            raise InputError(f"{path}.sample_index: must be the canonical sequence")
        if (
            power_limit != RUNNER_FIXED_PREFLIGHT["power_limit_w"]
            or graphics_clock != RUNNER_FIXED_PREFLIGHT["graphics_clock_mhz"]
            or memory_clock != RUNNER_FIXED_PREFLIGHT["memory_clock_mhz"]
        ):
            raise InputError(f"{path}: power/application-clock lane drifted")
        for name, value, maximum in (
            ("temperature_c", temperature, 95),
            ("memory_used_mib", memory_used, int(RUNNER_FIXED_PREFLIGHT["memory_total_mib"])),
        ):
            if re.fullmatch(r"0|[1-9][0-9]*", value) is None or int(value) > maximum:
                raise InputError(f"{path}.{name}: invalid or outside reviewed bound")
        if index in (0, len(samples) - 1):
            if processes != "none" or int(memory_used) > 256:
                raise InputError(f"{path}: boundary sample must prove an idle GPU")
        elif processes != "none":
            process_rows = processes.split(";")
            if not all(re.fullmatch(r"container:[1-9][0-9]*", value) for value in process_rows):
                raise InputError(f"{path}.compute_processes: foreign process receipt")
            observed_container_process = True
    if not observed_container_process:
        raise InputError(f"{label}: no designated-container CUDA process was observed")
    return {
        "sample_count": len(samples),
        "observed_container_process": True,
        "capture_id": samples[0][0],
        "container_id": samples[0][1],
    }


def _validate_runner_container(
    container: Mapping[str, Any],
    *,
    label: str,
    pair_index: int,
    manifest: Mapping[str, Any],
    after: bool,
) -> dict[str, Any]:
    candidate = manifest["candidate"]
    contract = manifest["container"]
    image_id = candidate["optimizer_image_id"]
    container_id = container.get("Id")
    if not isinstance(container_id, str) or re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise InputError(f"{label}.Id: expected lowercase 64-character ID")
    if container.get("Image") != image_id:
        raise InputError(f"{label}.Image: optimizer image mismatch")
    if container.get("Path") != RUNNER_CONTAINER_ENTRYPOINT[0] or container.get(
        "Args"
    ) != RUNNER_CONTAINER_CMD:
        raise InputError(f"{label}.Path/Args: exact resolved process required")
    config = container.get("Config")
    if not isinstance(config, dict):
        raise InputError(f"{label}.Config: must be an object")
    required_config = {
        "Image": image_id,
        "User": "0:0",
        "WorkingDir": "/workspace",
        "Entrypoint": RUNNER_CONTAINER_ENTRYPOINT,
        "Cmd": RUNNER_CONTAINER_CMD,
        "Healthcheck": {"Test": ["NONE"]},
    }
    for name, expected in required_config.items():
        if name not in config or not _exact_json_value(config[name], expected):
            raise InputError(f"{label}.Config.{name}: exact runner value required")
    expected_labels = contract["labels"]
    if not _exact_json_value(config.get("Labels"), expected_labels):
        raise InputError(
            f"{label}.Config.Labels: exact image-plus-supervisor labels required"
        )
    expected_environment = dict(contract["environment"])
    expected_environment["RUSTINFER_PERF_PAIR_INDEX"] = str(pair_index)
    expected_environment["RUSTINFER_PERF_CAPTURE_ID"] = manifest["executions"][
        pair_index - 1
    ]["capture_id"]
    if _runner_environment(config.get("Env"), f"{label}.Config.Env") != expected_environment:
        raise InputError(f"{label}.Config.Env: exact full environment required")

    host = container.get("HostConfig")
    if not isinstance(host, dict):
        raise InputError(f"{label}.HostConfig: must be an object")
    exact_host = {
        "NetworkMode": "none",
        "ReadonlyRootfs": True,
        "AutoRemove": False,
        "CapDrop": ["ALL"],
        "CapAdd": None,
        "SecurityOpt": ["no-new-privileges:true"],
        "PidsLimit": 512,
        "Privileged": False,
        "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
        "Tmpfs": {"/tmp": "rw,nosuid,nodev,noexec,size=2147483648"},
        "PidMode": "",
        "IpcMode": "private",
        "UTSMode": "",
        "UsernsMode": "",
        "CgroupnsMode": "private",
        "Runtime": "runc",
        "CpuShares": 0,
        "Memory": 0,
        "NanoCpus": 0,
        "CpuPeriod": 0,
        "CpuQuota": 0,
        "CpusetCpus": "",
        "CpusetMems": "",
        "MemoryReservation": 0,
        "MemorySwap": 0,
        "Devices": [],
        "DeviceCgroupRules": None,
    }
    for name, expected in exact_host.items():
        if name not in host or not _exact_json_value(host[name], expected):
            raise InputError(f"{label}.HostConfig.{name}: exact isolated runner value required")
    requests = host.get("DeviceRequests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], dict):
        raise InputError(f"{label}.HostConfig.DeviceRequests: one GPU request required")
    request = requests[0]
    if (
        not _exact_json_value(request.get("Driver"), "")
        or not _exact_json_value(request.get("Count"), 0)
        or not _exact_json_value(request.get("DeviceIDs"), [RUNNER_GPU_ROW[1]])
        or not _exact_json_value(request.get("Capabilities"), [["gpu"]])
        or not _exact_json_value(request.get("Options"), {})
    ):
        raise InputError(f"{label}.HostConfig.DeviceRequests: designated GPU UUID only")
    networks = container.get("NetworkSettings")
    attached = networks.get("Networks") if isinstance(networks, dict) else None
    if not isinstance(attached, dict) or not set(attached) <= {"none"}:
        raise InputError(f"{label}.NetworkSettings.Networks: isolated none network only")
    isolated = attached.get("none")
    if isolated is not None:
        if not isinstance(isolated, dict) or any(
            isolated.get(name) not in ("", None)
            for name in ("Gateway", "IPAddress", "GlobalIPv6Address", "MacAddress")
        ):
            raise InputError(
                f"{label}.NetworkSettings.Networks.none: addressless receipt required"
            )

    mounts = container.get("Mounts")
    if not isinstance(mounts, list):
        raise InputError(f"{label}.Mounts: must be an array")
    by_destination: dict[str, Mapping[str, Any]] = {}
    for mount in mounts:
        if not isinstance(mount, dict) or not isinstance(mount.get("Destination"), str):
            raise InputError(f"{label}.Mounts: malformed mount")
        destination = mount["Destination"]
        if destination in by_destination:
            raise InputError(f"{label}.Mounts: duplicate destination")
        by_destination[destination] = mount
    allowed = {*contract["read_only_mount_sources"], "/evidence", "/workspace", "/tmp"}
    if set(by_destination) not in (allowed - {"/tmp"}, allowed):
        raise InputError(f"{label}.Mounts: exact reviewed destination inventory required")
    for destination, source in contract["read_only_mount_sources"].items():
        mount = by_destination[destination]
        if (
            mount.get("Type") != "bind"
            or mount.get("Source") != source
            or mount.get("RW") is not False
            or mount.get("Mode") != ""
            or mount.get("Propagation") != "rprivate"
        ):
            raise InputError(f"{label}.Mounts[{destination}]: exact read-only bind required")
    evidence = by_destination["/evidence"]
    if (
        evidence.get("Type") != "bind"
        or evidence.get("Source") != contract["evidence_mount_sources"][pair_index - 1]
        or evidence.get("RW") is not True
        or evidence.get("Mode") != ""
        or evidence.get("Propagation") != "rprivate"
    ):
        raise InputError(f"{label}.Mounts[/evidence]: run-specific writable bind required")
    workspace = by_destination["/workspace"]
    if (
        workspace.get("Type") != "volume"
        or workspace.get("Source") != contract["workspace_volume_names"][pair_index - 1]
        or workspace.get("RW") is not True
    ):
        raise InputError(f"{label}.Mounts[/workspace]: named fresh writable volume required")
    if "/tmp" in by_destination:
        temporary = by_destination["/tmp"]
        if temporary.get("Type") != "tmpfs" or temporary.get("RW") is not True:
            raise InputError(f"{label}.Mounts[/tmp]: reviewed tmpfs required")

    state = container.get("State")
    if not isinstance(state, dict):
        raise InputError(f"{label}.State: must be an object")
    created_at = container.get("Created")
    _runner_timestamp_ns(created_at, f"{label}.Created")
    if not after:
        expected_state = {
            "Status": "created",
            "Running": False,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "Pid": 0,
            "ExitCode": 0,
            "Error": "",
            "StartedAt": RUNNER_ZERO_TIME,
            "FinishedAt": RUNNER_ZERO_TIME,
        }
        for name, expected in expected_state.items():
            if name not in state or not _exact_json_value(state[name], expected):
                raise InputError(f"{label}.State.{name}: pristine created receipt required")
        started_at = RUNNER_ZERO_TIME
        finished_at = RUNNER_ZERO_TIME
    else:
        expected_state = {
            "Status": "exited",
            "Running": False,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "ExitCode": 0,
            "Error": "",
        }
        for name, expected in expected_state.items():
            if name not in state or not _exact_json_value(state[name], expected):
                raise InputError(f"{label}.State.{name}: clean exit-zero receipt required")
        started_at = state.get("StartedAt")
        finished_at = state.get("FinishedAt")
        _runner_timestamp_ns(started_at, f"{label}.State.StartedAt")
        _runner_timestamp_ns(finished_at, f"{label}.State.FinishedAt")
        if not (
            _runner_timestamp_ns(created_at, f"{label}.Created")
            <= _runner_timestamp_ns(started_at, f"{label}.State.StartedAt")
            <= _runner_timestamp_ns(finished_at, f"{label}.State.FinishedAt")
        ):
            raise InputError(f"{label}.State: Created/StartedAt/FinishedAt order invalid")
    if not _exact_json_value(container.get("RestartCount"), 0):
        raise InputError(f"{label}.RestartCount: must remain zero")
    return {
        "container_id": container_id,
        "workspace_volume_name": workspace["Source"],
        "created_at_utc": created_at,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "exit_code": state.get("ExitCode"),
        "oom_killed": state.get("OOMKilled"),
    }


def validate_runner_receipt_payloads(
    payloads: Sequence[tuple[str, bytes]],
) -> dict[str, Any]:
    """Replay a complete closed v3 runner receipt inventory."""

    if [name for name, _raw in payloads] != list(RUNNER_RECEIPT_FILES):
        raise InputError(f"runner receipts: exact ordered inventory required: {list(RUNNER_RECEIPT_FILES)}")
    by_name = dict(payloads)
    if by_name["SHA256SUMS"] != _runner_sha256s(by_name):
        raise InputError("runner receipts: SHA256SUMS does not exactly bind every receipt")
    try:
        gpu_line = by_name["gpu.csv"].decode("utf-8", errors="strict").strip("\n")
    except UnicodeDecodeError as error:
        raise InputError("gpu.csv: must be strict UTF-8") from error
    gpu_values = tuple(value.strip() for value in gpu_line.split(","))
    if gpu_values != RUNNER_GPU_ROW or by_name["gpu.csv"].count(b"\n") != 1:
        raise InputError("gpu.csv: exact designated server-4096 GPU row required")

    manifest_document = _strict_json_payload(by_name["runner-manifest.json"], "runner-manifest.json")
    if not isinstance(manifest_document, dict) or not isinstance(manifest_document.get("candidate"), dict):
        raise InputError("runner-manifest.json: malformed candidate binding")
    image_id = manifest_document["candidate"].get("optimizer_image_id")
    if not isinstance(image_id, str):
        raise InputError("runner-manifest.json: missing optimizer image ID")
    before_image = _runner_image(
        by_name["optimizer-image-inspect-before.json"],
        "optimizer-image-inspect-before.json",
        image_id,
    )
    after_image = _runner_image(
        by_name["optimizer-image-inspect-after.json"],
        "optimizer-image-inspect-after.json",
        image_id,
    )
    if (
        by_name["optimizer-image-inspect-before.json"]
        != by_name["optimizer-image-inspect-after.json"]
    ):
        raise InputError("optimizer image inspect: before/after receipts differ")
    if before_image != after_image:
        raise InputError("optimizer image inspect: image environment or labels changed across run")
    manifest = _runner_manifest(
        by_name["runner-manifest.json"],
        image_environment=before_image["environment"],
        image_labels=before_image["labels"],
    )
    revision = manifest["candidate"]["source_revision"]
    candidate_payloads: list[tuple[str, bytes]] = []
    container_ids: list[str] = []
    volume_names: list[str] = []
    gpu_monitors: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for pair_index in range(1, 6):
        prefix = f"run-{pair_index}"
        preflight_raw = by_name[f"{prefix}/preflight.txt"]
        monitor_raw = by_name[f"{prefix}/gpu-monitor.csv"]
        before_raw = by_name[f"{prefix}/container-inspect-before.json"]
        after_raw = by_name[f"{prefix}/container-inspect-after.json"]
        candidate_raw = by_name[f"{prefix}/candidate.json"]
        execution = _runner_execution_payload(
            by_name[f"{prefix}/execution-receipt.json"],
            f"{prefix}/execution-receipt.json",
        )
        if not _exact_json_value(
            execution, manifest["executions"][pair_index - 1]
        ):
            raise InputError(f"{prefix}: execution receipt differs from runner manifest")
        expected_hashes = {
            "preflight": _digest_bytes(preflight_raw),
            "candidate": _digest_bytes(candidate_raw),
            "gpu_monitor": _digest_bytes(monitor_raw),
            "container_inspect_before": _digest_bytes(before_raw),
            "container_inspect_after": _digest_bytes(after_raw),
        }
        if execution["sha256"] != expected_hashes:
            raise InputError(f"{prefix}: execution receipt SHA-256 cross-binding mismatch")
        _runner_preflight(preflight_raw, f"{prefix}/preflight.txt", revision)
        before = _runner_container_document(
            before_raw,
            f"{prefix}/container-inspect-before.json",
        )
        after = _runner_container_document(
            after_raw,
            f"{prefix}/container-inspect-after.json",
        )
        before_facts = _validate_runner_container(
            before,
            label=f"{prefix}/container-inspect-before.json",
            pair_index=pair_index,
            manifest=manifest,
            after=False,
        )
        after_facts = _validate_runner_container(
            after,
            label=f"{prefix}/container-inspect-after.json",
            pair_index=pair_index,
            manifest=manifest,
            after=True,
        )
        if (
            before_facts["container_id"] != after_facts["container_id"]
            or before_facts["workspace_volume_name"]
            != after_facts["workspace_volume_name"]
            or before_facts["created_at_utc"] != after_facts["created_at_utc"]
        ):
            raise InputError(f"{prefix}: before/after container identity changed")
        container_id = before_facts["container_id"]
        if execution["container_id"] != container_id:
            raise InputError(f"{prefix}: execution receipt container ID mismatch")
        monitor = _runner_gpu_monitor(
            monitor_raw,
            f"{prefix}/gpu-monitor.csv",
            expected_capture_id=execution["capture_id"],
            expected_container_id=container_id,
        )
        gpu_monitors.append(monitor)
        try:
            candidate_document = json.loads(
                candidate_raw,
                object_pairs_hook=_pairs_no_duplicates,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, InputError) as error:
            raise InputError(f"{prefix}/candidate.json: invalid strict JSON: {error}") from error
        if not isinstance(candidate_document, dict):
            raise InputError(f"{prefix}/candidate.json: object required")
        expected_run_id = _runner_run_id(
            revision, execution["capture_id"], pair_index
        )
        if (
            candidate_document.get("pair_index") != pair_index
            or candidate_document.get("run_id") != expected_run_id
            or candidate_document.get("run_id") != execution["run_id"]
            or candidate_document.get("recorded_at_utc")
            != execution["candidate_recorded_at_utc"]
        ):
            raise InputError(f"{prefix}: raw candidate identity differs from execution receipt")
        expected_docker = {
            "created_at_utc": after_facts["created_at_utc"],
            "started_at_utc": after_facts["started_at_utc"],
            "finished_at_utc": after_facts["finished_at_utc"],
            "exit_code": after_facts["exit_code"],
            "oom_killed": after_facts["oom_killed"],
        }
        if not _exact_json_value(execution["docker"], expected_docker):
            raise InputError(f"{prefix}: Docker timeline differs from execution receipt")
        _validate_runner_execution_timeline(
            execution, label=f"{prefix}/execution-receipt.json"
        )
        container_ids.append(container_id)
        volume_names.append(before_facts["workspace_volume_name"])
        candidate_payloads.append((f"candidate-{pair_index}.json", candidate_raw))
        executions.append(execution)
    if len(set(container_ids)) != 5:
        raise InputError("runner receipts: exactly five distinct container IDs required")
    if len(set(volume_names)) != 5:
        raise InputError("runner receipts: exactly five distinct workspace volumes required")
    for previous, current in zip(executions, executions[1:], strict=False):
        if _runner_timestamp_ns(
            previous["docker"]["finished_at_utc"], "previous FinishedAt"
        ) > _runner_timestamp_ns(
            current["docker"]["created_at_utc"], "next Created"
        ):
            raise InputError("runner receipts: sequential pair timelines overlap")
    derived = derive_raw_run_payloads(candidate_payloads)
    source = derived["source"]
    candidate_binding = manifest["candidate"]
    expected_source = {
        "git_commit": candidate_binding["source_revision"],
        "git_dirty": False,
        "executable_sha256": candidate_binding["profile_binary_sha256"],
        "implementation_id": "native-iteration-command-batch",
        "runtime_flag": {"name": "execution_completion", "value": "iteration-batch"},
        "semantic_class": "E0",
        "correctness_gate_id": CORRECTNESS_GATE_ID,
        "correctness_report_sha256": candidate_binding["optimizer_correctness_report_sha256"],
    }
    if source != expected_source:
        raise InputError("runner receipts: raw source does not match runner manifest")
    if derived["runs"][0]["environment"]["software"]["container_image_sha256"] != image_id.removeprefix("sha256:"):
        raise InputError("runner receipts: raw profile image does not match inspected image")
    if derived["run_summary"] != {
        "independent_runs": 5,
        "warmups_per_run": 5,
        "measured_iterations_per_run": 30,
        "failure_count": 0,
        "dropped_trace_records": 0,
    }:
        raise InputError("runner receipts: exact 5 x (5 warmups + 30 measured) derivation required")
    return {
        "manifest": manifest,
        "payloads": derived["payloads"],
        "derived": derived,
        "container_ids": container_ids,
        "workspace_volume_names": volume_names,
        "gpu_monitors": gpu_monitors,
        "executions": executions,
    }


def load_runner_receipt_root(path: Path | str) -> dict[str, Any]:
    """Open every required runner receipt once, without following links."""

    root = Path(path)
    root_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    root_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_descriptor = -1
    try:
        path_metadata = root.lstat()
        root_descriptor = os.open(root, root_flags)
        metadata = os.fstat(root_descriptor)
    except OSError as error:
        raise InputError(f"runner receipt root: cannot inspect {root}: {error}") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path_metadata.st_dev != metadata.st_dev
        or path_metadata.st_ino != metadata.st_ino
    ):
        os.close(root_descriptor)
        root_descriptor = -1
        raise InputError("runner receipt root: must be a real directory")
    try:
        payloads: list[tuple[str, bytes]] = []
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        for name in RUNNER_RECEIPT_FILES:
            parts = name.split("/")
            parent_descriptor = os.dup(root_descriptor)
            descriptor = -1
            try:
                for component in parts[:-1]:
                    child = os.open(component, directory_flags, dir_fd=parent_descriptor)
                    os.close(parent_descriptor)
                    parent_descriptor = child
                descriptor = os.open(parts[-1], file_flags, dir_fd=parent_descriptor)
                raw, _digest, _file_metadata = _stable_fd_snapshot(
                    descriptor,
                    f"runner receipt {name}",
                    _RUNNER_RECEIPT_MAXIMUMS[name],
                )
                payloads.append((name, raw))
            except InputError:
                raise
            except OSError as error:
                raise InputError(f"runner receipt {name}: cannot open: {error}") from error
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                os.close(parent_descriptor)
        after = os.fstat(root_descriptor)
        stable = ("st_dev", "st_ino", "st_mode", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(metadata, field) != getattr(after, field) for field in stable):
            raise InputError("runner receipt root: changed while receipts were opened")
        result = validate_runner_receipt_payloads(payloads)
        result["archive_payloads"] = payloads
        return result
    finally:
        if root_descriptor >= 0:
            os.close(root_descriptor)


def _canonical_tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_canonical_raw_archive(
    handle: Any, payloads: Sequence[tuple[str, bytes]]
) -> None:
    with tarfile.open(
        fileobj=handle, mode="w:", format=tarfile.USTAR_FORMAT
    ) as archive:
        for name, raw in payloads:
            archive.addfile(_canonical_tar_info(name, len(raw)), io.BytesIO(raw))


def _canonical_raw_archive_bytes(
    payloads: Sequence[tuple[str, bytes]],
) -> bytes:
    buffer = io.BytesIO()
    _write_canonical_raw_archive(buffer, payloads)
    return buffer.getvalue()


def write_raw_evidence_archive(
    output: Path | str,
    payloads: Sequence[tuple[str, bytes]],
    *,
    runner_receipt_root: Path | str,
) -> str:
    """Create the canonical v3 USTAR after replaying all runner receipts."""

    canonical_candidates = derive_raw_run_payloads(payloads)["payloads"]
    receipt = load_runner_receipt_root(runner_receipt_root)
    receipt_candidates = receipt["payloads"]
    if canonical_candidates != receipt_candidates:
        raise InputError("raw evidence: --run payloads differ from runner receipts")
    canonical_payloads = receipt["archive_payloads"]
    output_path = Path(output)
    if not output_path.name:
        raise InputError("raw performance evidence output must name a new file")
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    parent = output_path.parent
    try:
        parent_metadata = parent.stat()
    except OSError as error:
        raise InputError(f"cannot inspect raw evidence output parent {parent}: {error}") from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise InputError(f"raw evidence output parent is not a directory: {parent}")

    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.staging-", dir=parent
    )
    staged_archive = Path(staging_name)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            _write_canonical_raw_archive(handle, canonical_payloads)
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        binding = _record_held_file(
            descriptor,
            output_path.name,
            maximum=MAX_RAW_EVIDENCE_ARCHIVE_BYTES,
        )
        _verify_bound_file_path(
            descriptor,
            staged_archive,
            binding,
            "staged raw performance evidence archive before publish",
        )
        _rename_noreplace(staged_archive, output_path)
        _fsync_directory(parent)
        _verify_bound_file_path(
            descriptor,
            output_path,
            binding,
            "published raw performance evidence archive",
        )
        return binding.digest
    finally:
        os.close(descriptor)


def _snapshot_raw_evidence_archive(path: Path) -> tuple[bytes, str]:
    label = "raw performance evidence archive"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        raw, digest, _metadata = _stable_fd_snapshot(
            descriptor, label, MAX_RAW_EVIDENCE_ARCHIVE_BYTES
        )
        return raw, digest
    except InputError:
        raise
    except OSError as error:
        raise InputError(f"{label}: cannot open stable snapshot: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_raw_evidence_archive_snapshot(
    path: Path,
) -> tuple[list[tuple[str, bytes]], str, dict[str, Any]]:
    archive_raw, archive_digest = _snapshot_raw_evidence_archive(path)
    label = "raw performance evidence archive"
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != list(RUNNER_RECEIPT_FILES):
                raise InputError(
                    f"{label}: exact ordered inventory required: {list(RUNNER_RECEIPT_FILES)}"
                )
            payloads: list[tuple[str, bytes]] = []
            for member in members:
                name = member.name
                if not member.isreg():
                    raise InputError(f"{label}: member must be a regular file: {name}")
                if member.pax_headers:
                    raise InputError(f"{label}: PAX extensions are forbidden: {name}")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mode != 0o644
                    or member.mtime != 0
                ):
                    raise InputError(f"{label}: non-canonical metadata for {name}")
                if member.size <= 0 or member.size > _RUNNER_RECEIPT_MAXIMUMS[name]:
                    raise InputError(f"{label}: invalid size for {name}")
                source = archive.extractfile(member)
                if source is None:
                    raise InputError(f"{label}: cannot read {name}")
                raw = source.read(_RUNNER_RECEIPT_MAXIMUMS[name] + 1)
                if len(raw) != member.size:
                    raise InputError(f"{label}: truncated or oversized member {name}")
                payloads.append((name, raw))
    except InputError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise InputError(
            f"{label}: cannot read deterministic uncompressed USTAR: {error}"
        ) from error

    validation = validate_runner_receipt_payloads(payloads)
    if archive_raw != _canonical_raw_archive_bytes(payloads):
        raise InputError(
            f"{label}: bytes are not the canonical deterministic USTAR encoding"
        )
    return validation["payloads"], archive_digest, validation["manifest"]


def load_raw_evidence_archive(
    path: Path | str,
) -> list[tuple[str, bytes]]:
    """Replay v3 receipts and return only the five candidate payloads."""

    payloads, _digest, _manifest = _load_raw_evidence_archive_snapshot(Path(path))
    return payloads


def replay_raw_evidence_archive(path: Path | str) -> dict[str, Any]:
    """Replay raw field derivation from a canonical performance archive."""

    payloads, archive_digest, manifest = _load_raw_evidence_archive_snapshot(Path(path))
    return {
        "archive_sha256": archive_digest,
        "derived": derive_raw_run_payloads(payloads),
        "payloads": payloads,
        "runner_manifest": manifest,
    }


def evaluate(
    baseline_path: Path | str,
    candidate_path: Path | str,
    *,
    source_archive: Path | str,
    profile_binary: Path | str,
    release_binary: Path | str,
    weights: Path | str,
    tokenizer: Path | str,
    correctness_report: Path | str,
    profile_image_id: str,
    release_image_id: str,
    run_paths: Sequence[Path | str],
    runner_receipt_root: Path | str,
) -> dict[str, Any]:
    """Evaluate already-produced CPU-readable release evidence."""

    report = _empty_report()
    try:
        baseline_doc, baseline_raw = _load_json_bytes(Path(baseline_path), "baseline")
        candidate_doc, _ = _load_json_bytes(Path(candidate_path), "candidate")
        baseline = _validate_baseline(baseline_doc, baseline_raw)
        candidate = _validate_candidate(candidate_doc)
        if candidate["baseline_sha256"] != baseline["sha256"]:
            raise InputError("candidate does not bind the reviewed baseline bytes")
        for field in ["model", "environment", "workload"]:
            if candidate[field] != baseline[field]:
                raise ComparabilityError(
                    f"candidate {field} differs from baseline lane"
                )

        if not profile_image_id.startswith("sha256:"):
            raise InputError("--profile-image-id: expected sha256:<lowercase digest>")
        if not release_image_id.startswith("sha256:"):
            raise InputError("--release-image-id: expected sha256:<lowercase digest>")
        profile_image_digest = profile_image_id.removeprefix("sha256:")
        release_image_digest = release_image_id.removeprefix("sha256:")
        _sha256(profile_image_digest, "--profile-image-id")
        _sha256(release_image_digest, "--release-image-id")
        receipt = load_runner_receipt_root(runner_receipt_root)
        _require_request_identity_sha256(
            receipt["derived"],
            baseline["request_identity_sha256"],
            "runner receipts.request_identity_sha256",
        )
        supplied_payloads = derive_raw_run_payloads(_read_raw_run_paths(run_paths))["payloads"]
        if supplied_payloads != receipt["payloads"]:
            raise InputError("--run payloads differ from --runner-receipt-root")
        actual = {
            "source_archive_sha256": _digest_file(
                Path(source_archive), "source archive"
            ),
            "profile_binary_sha256": _digest_file(
                Path(profile_binary), "profile binary"
            ),
            "release_binary_sha256": _digest_file(
                Path(release_binary), "release binary"
            ),
            "profile_image_sha256": profile_image_digest,
            "release_image_sha256": release_image_digest,
            "correctness_report_sha256": _digest_file(
                Path(correctness_report), "correctness report"
            ),
        }
        for field, digest in actual.items():
            if candidate["source"][field] != digest:
                raise InputError(
                    f"candidate.source.{field}: bound digest does not match artifact"
                )
        receipt_candidate = receipt["manifest"]["candidate"]
        receipt_bindings = {
            "source_revision": candidate["source"]["git_commit"],
            "source_archive_sha256": actual["source_archive_sha256"],
            "profile_binary_sha256": actual["profile_binary_sha256"],
            "optimizer_correctness_report_sha256": actual[
                "correctness_report_sha256"
            ],
            "optimizer_image_id": profile_image_id,
        }
        for field, expected in receipt_bindings.items():
            if receipt_candidate[field] != expected:
                raise InputError(
                    f"runner-manifest.candidate.{field}: does not match submitted artifact"
                )
        weights_digest = _digest_file(Path(weights), "model weights")
        tokenizer_digest = _digest_file(Path(tokenizer), "tokenizer")
        if candidate["model"]["weights_sha256"] != weights_digest:
            raise InputError("candidate.model.weights_sha256 does not match --weights")
        if candidate["model"]["tokenizer_sha256"] != tokenizer_digest:
            raise InputError(
                "candidate.model.tokenizer_sha256 does not match --tokenizer"
            )

        correctness_doc, _ = _load_json_bytes(
            Path(correctness_report), "optimization correctness report"
        )
        _validate_optimization_correctness(correctness_doc, candidate)
        optimizer_model_tree = correctness_doc["model"]["manifest_sha256"]
        if receipt_candidate["model_tree_sha256"] != optimizer_model_tree:
            raise InputError(
                "runner-manifest.candidate.model_tree_sha256: does not match "
                "the submitted optimizer correctness model manifest"
            )

        _runs, raw_summary, raw_metrics = validate_raw_run_payloads(
            receipt["payloads"], candidate
        )

        summary = raw_summary
        workload = baseline["workload"]
        for field in [
            "independent_runs",
            "warmups_per_run",
            "measured_iterations_per_run",
        ]:
            if summary[field] != workload[field]:
                raise ComparabilityError(
                    f"candidate run_summary.{field} differs from baseline workload"
                )

        metrics = baseline["metrics"]
        candidate_metrics = raw_metrics
        ratios = {
            field: candidate_metrics[field] / metrics[field] for field in METRIC_FIELDS
        }
        thresholds = baseline["thresholds"]
        checks = [
            _check("ttft_p95_regression", ratios["ttft_p95_ms"], "<=", thresholds["ttft_p95_ratio_max"]),
            _check("tpot_p95_regression", ratios["tpot_p95_ms"], "<=", thresholds["tpot_p95_ratio_max"]),
            _check("e2e_median_regression", ratios["e2e_median_ms"], "<=", thresholds["e2e_median_ratio_max"]),
            _check(
                "throughput_median_regression",
                ratios["throughput_median_output_tokens_per_second"],
                ">=",
                thresholds["throughput_median_ratio_min"],
            ),
        ]
        passed = all(check["passed"] for check in checks)
        report.update(
            {
                "status": "passed" if passed else "failed",
                "passed": passed,
                "baseline": {
                    "baseline_id": baseline["baseline_id"],
                    "sha256": baseline["sha256"],
                    "metrics": metrics,
                },
                "candidate": {
                    "candidate_id": candidate["candidate_id"],
                    "recorded_at_utc": candidate["recorded_at_utc"],
                    "source": candidate["source"],
                    "model": candidate["model"],
                    "environment": candidate["environment"],
                    "workload": candidate["workload"],
                    "metrics": candidate_metrics,
                    "run_summary": summary,
                    "raw_runs": candidate["raw_runs"],
                },
                "ratios": ratios,
                "checks": checks,
            }
        )
    except ComparabilityError as error:
        report["status"] = "incomparable"
        report["errors"] = [str(error)]
    except InputError as error:
        report["errors"] = [str(error)]
    return report


def _image_digest(value: str, path: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise InputError(f"{path}: expected sha256:<lowercase digest>")
    digest = value.removeprefix("sha256:")
    _sha256(digest, path)
    return digest


def _json_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _build_candidate_from_payloads(
    baseline_path: Path | str,
    *,
    candidate_id: str,
    recorded_at_utc: str,
    source_archive: Path | str,
    profile_binary: Path | str,
    release_binary: Path | str,
    correctness_report: Path | str,
    profile_image_id: str,
    release_image_id: str,
    payloads: Sequence[tuple[str, bytes]],
) -> dict[str, Any]:
    baseline_document, baseline_raw = _load_json_bytes(
        Path(baseline_path), "baseline"
    )
    baseline = _validate_baseline(baseline_document, baseline_raw)
    derived = derive_raw_run_payloads(payloads)
    _require_request_identity_sha256(
        derived,
        baseline["request_identity_sha256"],
        "candidate runs.request_identity_sha256",
    )
    raw_source = derived["source"]
    candidate = {
        "schema_version": CANDIDATE_SCHEMA,
        "baseline_sha256": baseline["sha256"],
        "candidate_id": candidate_id,
        "recorded_at_utc": recorded_at_utc,
        "status": "success",
        "source": {
            "git_commit": raw_source["git_commit"],
            "git_dirty": raw_source["git_dirty"],
            "source_archive_sha256": _digest_file(
                Path(source_archive), "source archive"
            ),
            "profile_binary_sha256": _digest_file(
                Path(profile_binary), "profile binary"
            ),
            "release_binary_sha256": _digest_file(
                Path(release_binary), "release binary"
            ),
            "profile_image_sha256": _image_digest(
                profile_image_id, "--profile-image-id"
            ),
            "release_image_sha256": _image_digest(
                release_image_id, "--release-image-id"
            ),
            "semantic_class": raw_source["semantic_class"],
            "correctness_gate_id": raw_source["correctness_gate_id"],
            "correctness_report_sha256": _digest_file(
                Path(correctness_report), "correctness report"
            ),
        },
        "model": derived["model"],
        "environment": derived["environment"],
        "workload": derived["workload"],
        "run_summary": derived["run_summary"],
        "metrics": derived["metrics"],
        "raw_runs": derived["raw_runs"],
    }
    validated = _validate_candidate(candidate)
    for field in ("model", "environment", "workload"):
        if validated[field] != baseline[field]:
            raise ComparabilityError(
                f"candidate {field} differs from baseline lane"
            )
    validate_raw_run_payloads(payloads, validated)
    return candidate


def build_candidate_document(
    baseline_path: Path | str,
    *,
    candidate_id: str,
    recorded_at_utc: str,
    source_archive: Path | str,
    profile_binary: Path | str,
    release_binary: Path | str,
    correctness_report: Path | str,
    profile_image_id: str,
    release_image_id: str,
    run_paths: Sequence[Path | str],
) -> dict[str, Any]:
    """Build a closed candidate document from raw run files and artifacts."""

    return _build_candidate_from_payloads(
        baseline_path,
        candidate_id=candidate_id,
        recorded_at_utc=recorded_at_utc,
        source_archive=source_archive,
        profile_binary=profile_binary,
        release_binary=release_binary,
        correctness_report=correctness_report,
        profile_image_id=profile_image_id,
        release_image_id=release_image_id,
        payloads=_read_raw_run_paths(run_paths),
    )


def _evaluate_payload_snapshot(
    baseline_path: Path | str,
    candidate_path: Path,
    payloads: Sequence[tuple[str, bytes]],
    *,
    source_archive: Path | str,
    profile_binary: Path | str,
    release_binary: Path | str,
    weights: Path | str,
    tokenizer: Path | str,
    correctness_report: Path | str,
    profile_image_id: str,
    release_image_id: str,
    runner_receipt_root: Path | str,
) -> dict[str, Any]:
    canonical_payloads = derive_raw_run_payloads(payloads)["payloads"]
    with tempfile.TemporaryDirectory(
        prefix="rustinfer-performance-evaluate-"
    ) as temporary:
        directory = Path(temporary)
        run_paths: list[Path] = []
        for name, raw in canonical_payloads:
            path = directory / name
            with path.open("xb") as handle:
                handle.write(raw)
            run_paths.append(path)
        return evaluate(
            baseline_path,
            candidate_path,
            source_archive=source_archive,
            profile_binary=profile_binary,
            release_binary=release_binary,
            weights=weights,
            tokenizer=tokenizer,
            correctness_report=correctness_report,
            profile_image_id=profile_image_id,
            release_image_id=release_image_id,
            run_paths=run_paths,
            runner_receipt_root=runner_receipt_root,
        )


def package_release_performance_evidence(
    baseline_path: Path | str,
    output_directory: Path | str,
    *,
    candidate_id: str,
    recorded_at_utc: str,
    source_archive: Path | str,
    profile_binary: Path | str,
    release_binary: Path | str,
    weights: Path | str,
    tokenizer: Path | str,
    correctness_report: Path | str,
    profile_image_id: str,
    release_image_id: str,
    run_paths: Sequence[Path | str],
    runner_receipt_root: Path | str,
) -> dict[str, Any]:
    """Create a checked three-file performance evidence directory.

    All validation and archive self-replay occur in a staging directory.  The
    requested output directory is then created exclusively and is never reused
    or overwritten.
    """

    candidate_id = _candidate_id(candidate_id, "candidate.candidate_id")
    output = Path(output_directory)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    parent = output.parent
    if not output.name:
        raise InputError("--output-directory: must name a new directory")
    try:
        parent_metadata = parent.stat()
    except OSError as error:
        raise InputError(f"cannot inspect output parent {parent}: {error}") from error
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise InputError(f"output parent is not a directory: {parent}")

    receipt = load_runner_receipt_root(runner_receipt_root)
    input_payloads = _read_raw_run_paths(run_paths)
    derived = derive_raw_run_payloads(input_payloads)
    canonical_payloads = derived["payloads"]
    if canonical_payloads != receipt["payloads"]:
        raise InputError("--run payloads differ from --runner-receipt-root")
    candidate = _build_candidate_from_payloads(
        baseline_path,
        candidate_id=candidate_id,
        recorded_at_utc=recorded_at_utc,
        source_archive=source_archive,
        profile_binary=profile_binary,
        release_binary=release_binary,
        correctness_report=correctness_report,
        profile_image_id=profile_image_id,
        release_image_id=release_image_id,
        payloads=canonical_payloads,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    staging_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    staging_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    staging_descriptor = os.open(staging, staging_flags)
    staging_metadata = os.fstat(staging_descriptor)
    held_descriptors: dict[str, int] = {}
    try:
        candidate_path = staging / PACKAGE_CANDIDATE_NAME
        candidate_bytes = _json_document_bytes(candidate)
        held_descriptors[PACKAGE_CANDIDATE_NAME] = _write_new_file(
            staging_descriptor,
            PACKAGE_CANDIDATE_NAME,
            candidate_bytes,
        )
        raw_evidence_path = staging / PACKAGE_RAW_EVIDENCE_NAME
        raw_evidence_sha256 = write_raw_evidence_archive(
            raw_evidence_path,
            canonical_payloads,
            runner_receipt_root=runner_receipt_root,
        )
        raw_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        raw_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        held_descriptors[PACKAGE_RAW_EVIDENCE_NAME] = os.open(
            PACKAGE_RAW_EVIDENCE_NAME,
            raw_flags,
            dir_fd=staging_descriptor,
        )

        source_report = _evaluate_payload_snapshot(
            baseline_path,
            candidate_path,
            canonical_payloads,
            source_archive=source_archive,
            profile_binary=profile_binary,
            release_binary=release_binary,
            weights=weights,
            tokenizer=tokenizer,
            correctness_report=correctness_report,
            profile_image_id=profile_image_id,
            release_image_id=release_image_id,
            runner_receipt_root=runner_receipt_root,
        )
        if source_report["status"] not in {"passed", "failed"}:
            detail = "; ".join(source_report["errors"]) or source_report["status"]
            if source_report["status"] == "incomparable":
                raise ComparabilityError(
                    "cannot package incomparable release performance evidence: "
                    f"{detail}"
                )
            raise InputError(
                "cannot package structurally invalid release performance evidence: "
                f"{detail}"
            )
        if source_report["passed"] is not (source_report["status"] == "passed"):
            raise InputError("performance report status/pass fields are inconsistent")
        if source_report["errors"] != []:
            raise InputError("comparable performance report must not contain errors")

        replay = replay_raw_evidence_archive(raw_evidence_path)
        validate_raw_run_payloads(replay["payloads"], _validate_candidate(candidate))
        replayed_report = _evaluate_payload_snapshot(
            baseline_path,
            candidate_path,
            replay["payloads"],
            source_archive=source_archive,
            profile_binary=profile_binary,
            release_binary=release_binary,
            weights=weights,
            tokenizer=tokenizer,
            correctness_report=correctness_report,
            profile_image_id=profile_image_id,
            release_image_id=release_image_id,
            runner_receipt_root=runner_receipt_root,
        )
        if _json_document_bytes(replayed_report) != _json_document_bytes(
            source_report
        ):
            raise InputError(
                "raw performance evidence self-replay changed the checked report"
            )

        report_bytes = _json_document_bytes(replayed_report)
        held_descriptors[PACKAGE_REPORT_NAME] = _write_new_file(
            staging_descriptor,
            PACKAGE_REPORT_NAME,
            report_bytes,
        )
        bindings = {
            PACKAGE_CANDIDATE_NAME: _record_held_file(
                held_descriptors[PACKAGE_CANDIDATE_NAME],
                PACKAGE_CANDIDATE_NAME,
                maximum=native_profile.MAX_EVIDENCE_BYTES,
            ),
            PACKAGE_REPORT_NAME: _record_held_file(
                held_descriptors[PACKAGE_REPORT_NAME],
                PACKAGE_REPORT_NAME,
                maximum=native_profile.MAX_EVIDENCE_BYTES,
            ),
            PACKAGE_RAW_EVIDENCE_NAME: _record_held_file(
                held_descriptors[PACKAGE_RAW_EVIDENCE_NAME],
                PACKAGE_RAW_EVIDENCE_NAME,
                maximum=MAX_RAW_EVIDENCE_ARCHIVE_BYTES,
            ),
        }
        expected_digests = {
            PACKAGE_CANDIDATE_NAME: _digest_bytes(candidate_bytes),
            PACKAGE_REPORT_NAME: _digest_bytes(report_bytes),
            PACKAGE_RAW_EVIDENCE_NAME: raw_evidence_sha256,
        }
        for name, expected_digest in expected_digests.items():
            if bindings[name].digest != expected_digest:
                raise InputError(
                    f"package child {name}: held bytes differ from generated bytes"
                )
        os.fsync(staging_descriptor)
        if not _same_inode(staging, staging_metadata, directory=True):
            raise InputError(
                "private performance staging directory changed before publish"
            )
        _verify_package_children(
            staging_descriptor,
            staging_metadata,
            held_descriptors,
            bindings,
            "private performance staging directory at publish",
        )
        _rename_noreplace(staging, output)
        _fsync_directory(parent)
        if not _same_inode(output, staging_metadata, directory=True):
            raise InputError(
                "published performance evidence directory changed before completion"
            )
        _verify_package_children(
            staging_descriptor,
            staging_metadata,
            held_descriptors,
            bindings,
            "published performance evidence directory after path check",
        )
        if not _same_inode(output, staging_metadata, directory=True):
            raise InputError(
                "published performance evidence path changed during verification"
            )
        return {
            "candidate": candidate,
            "report": replayed_report,
            "candidate_sha256": bindings[PACKAGE_CANDIDATE_NAME].digest,
            "report_sha256": bindings[PACKAGE_REPORT_NAME].digest,
            "raw_evidence_sha256": bindings[PACKAGE_RAW_EVIDENCE_NAME].digest,
        }
    finally:
        for descriptor in held_descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        os.close(staging_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--profile-binary", required=True, type=Path)
    parser.add_argument("--release-binary", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--correctness-report", required=True, type=Path)
    parser.add_argument("--profile-image-id", required=True)
    parser.add_argument("--release-image-id", required=True)
    parser.add_argument("--run", required=True, nargs=5, type=Path)
    parser.add_argument("--runner-receipt-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.baseline,
        args.candidate,
        source_archive=args.source_archive,
        profile_binary=args.profile_binary,
        release_binary=args.release_binary,
        weights=args.weights,
        tokenizer=args.tokenizer,
        correctness_report=args.correctness_report,
        profile_image_id=args.profile_image_id,
        release_image_id=args.release_image_id,
        run_paths=args.run,
        runner_receipt_root=args.runner_receipt_root,
    )
    encoded = json.dumps(
        report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if args.report is not None:
        try:
            with args.report.open("x", encoding="utf-8", newline="") as handle:
                handle.write(encoded)
        except FileExistsError:
            print(f"refusing to overwrite existing report: {args.report}", file=sys.stderr)
            return 2
        except OSError as error:
            print(f"cannot create report {args.report}: {error}", file=sys.stderr)
            return 2
    sys.stdout.write(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
