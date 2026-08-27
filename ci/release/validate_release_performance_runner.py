#!/usr/bin/env python3
"""Validate inputs and raw outputs of the remote release-performance runner.

This helper is intentionally CPU-only.  The shell runner captures facts from
the designated host and Docker daemon; this program rejects incomplete or
self-inconsistent captures before a GPU process starts and revalidates the
five native profile documents after the containers have exited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPTS = ROOT / "benchmarks" / "scripts"
sys.path.insert(0, str(BENCHMARK_SCRIPTS))

import check_release_performance as performance  # noqa: E402


SCHEMA_VERSION = "riley.release-performance-runner-validation.v2"
GATE_ID = "pr15-iteration-command-batch-exact-v1"
BASELINE_PATH = ROOT / "benchmarks" / "release" / "performance-baseline-v1.json"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
WEIGHTS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
GPU_NAME = "NVIDIA GeForce RTX 4090"
GPU_UUID = "GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0"
GPU_PCI_BUS_ID = "00000000:01:00.0"
GPU_MEMORY_MIB = 24_564
GPU_COMPUTE_CAPABILITY = "8.9"
DRIVER_VERSION = "580.173.02"
CUDA_RUNTIME_VERSION = "12.8.1"
CUDA_TOOLKIT_VERSION = "12.8.93"
CUDA_ARCHITECTURE = "89"
ENVIRONMENT_ID = "server-4096-rtx4090-pr15-v1"
IMPLEMENTATION_ID = "native-iteration-command-batch"
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_MODEL_PATH_RE = re.compile(r"^[A-Za-z0-9._/+@=-]+$")

PREFLIGHT_FIELDS = {
    "environment_id",
    "os_id",
    "os_version_id",
    "kernel_release",
    "machine",
    "cpu_model",
    "physical_cpu_cores",
    "logical_cpu_threads",
    "ram_bytes",
    "git_revision",
    "gpu_name",
    "compute_capability",
    "memory_total_mib",
    "memory_used_mib",
    "driver_version",
    "persistence_mode",
    "temperature_c",
    "power_limit_w",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "cpu_governor",
    "cpu_governor_policy_count",
    "clock_synchronized",
    "staging_available_bytes",
    "staging_minimum_bytes",
}
FIXED_PREFLIGHT = {
    "environment_id": "rtx4090-ubuntu22-driver580-v1",
    "os_id": "ubuntu",
    "os_version_id": "22.04",
    "kernel_release": "6.8.0-138-generic",
    "machine": "x86_64",
    "cpu_model": "Intel Core i7-13700K",
    "physical_cpu_cores": "16",
    "logical_cpu_threads": "24",
    "ram_bytes": "67185598464",
    "gpu_name": GPU_NAME,
    "compute_capability": GPU_COMPUTE_CAPABILITY,
    "memory_total_mib": str(GPU_MEMORY_MIB),
    "driver_version": DRIVER_VERSION,
    "persistence_mode": "Disabled",
    "power_limit_w": "450.00",
    "graphics_clock_mhz": "[N/A]",
    "memory_clock_mhz": "[N/A]",
    "cpu_governor": "powersave",
    "cpu_governor_policy_count": "24",
    "clock_synchronized": "yes",
    "staging_minimum_bytes": "21474836480",
}


class ContractError(ValueError):
    """The captured runner input is malformed or outside the reviewed lane."""


def _fail(path: str, message: str) -> NoReturn:
    raise ContractError(f"{path}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail("JSON", f"non-finite number {value!r} is forbidden")


def _load_json(path: Path, label: str) -> Any:
    try:
        raw = _snapshot(path, label, maximum=16 * 1024 * 1024)
        return json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(label, f"cannot read strict UTF-8 JSON: {error}")


def _regular(path: Path, label: str, *, executable: bool = False) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        _fail(label, f"cannot inspect file: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(label, "must be a regular file, not a link or device")
    if metadata.st_size <= 0:
        _fail(label, "must not be empty")
    if executable and metadata.st_mode & 0o111 == 0:
        _fail(label, "must have an executable mode bit")
    return metadata


def _snapshot(
    path: Path,
    label: str,
    *,
    maximum: int,
    executable: bool = False,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = -1
    try:
        before_path = path.lstat()
        if not stat.S_ISREG(before_path.st_mode):
            _fail(label, "must be a regular file, not a link or device")
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            before.st_dev != before_path.st_dev
            or before.st_ino != before_path.st_ino
            or not stat.S_ISREG(before.st_mode)
        ):
            _fail(label, "changed while it was opened")
        if before.st_size <= 0 or before.st_size > maximum:
            _fail(label, f"must be between 1 and {maximum} bytes")
        if executable and before.st_mode & 0o111 == 0:
            _fail(label, "must have an executable mode bit")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable):
            _fail(label, "changed while it was read")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            _fail(label, "was truncated or enlarged while it was read")
        return raw
    except ContractError:
        raise
    except OSError as error:
        _fail(label, f"cannot read stable regular file: {error}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha256(
    path: Path,
    label: str,
    *,
    maximum: int = 4 * 1024**3,
    executable: bool = False,
) -> str:
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
            or before.st_size > maximum
        ):
            _fail(label, "is not one stable bounded regular file")
        if executable and before.st_mode & 0o111 == 0:
            _fail(label, "must have an executable mode bit")
        digest = hashlib.sha256()
        byte_count = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if byte_count != before.st_size or any(
            getattr(before, name) != getattr(after, name) for name in stable
        ):
            _fail(label, "changed while it was hashed")
        return digest.hexdigest()
    except ContractError:
        raise
    except OSError as error:
        _fail(label, f"cannot hash stable file: {error}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _expected_sha(value: str, label: str) -> str:
    if SHA_RE.fullmatch(value) is None:
        _fail(label, "must be a lowercase SHA-256")
    return value


def _expected_revision(value: str) -> str:
    if GIT_RE.fullmatch(value) is None:
        _fail("--source-revision", "must be a lowercase 40-character Git SHA")
    return value


def _expected_image(value: str) -> tuple[str, str]:
    match = IMAGE_RE.fullmatch(value)
    if match is None:
        _fail("--optimizer-image-id", "must be sha256:<lowercase digest>")
    return value, match.group(1)


def _strict_decimal(value: str, path: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        _fail(path, "must be a canonical nonnegative decimal integer")
    return int(value)


def _strict_finite(value: str, path: str) -> float:
    try:
        number = float(value)
    except ValueError:
        _fail(path, "must be a finite number")
    if not math.isfinite(number):
        _fail(path, "must be a finite number")
    return number


def _parse_preflight(path: Path, revision: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = _snapshot(path, "preflight snapshot", maximum=256 * 1024).decode(
            "utf-8"
        ).splitlines()
    except UnicodeDecodeError as error:
        _fail("preflight snapshot", f"cannot read UTF-8 text: {error}")
    for line_number, line in enumerate(lines, 1):
        if not line or "=" not in line:
            _fail(f"preflight snapshot:{line_number}", "expected key=value")
        key, value = line.split("=", 1)
        if not key or key in values:
            _fail(f"preflight snapshot:{line_number}", "duplicate or empty key")
        values[key] = value
    if set(values) != PREFLIGHT_FIELDS:
        missing = sorted(PREFLIGHT_FIELDS - set(values))
        extra = sorted(set(values) - PREFLIGHT_FIELDS)
        _fail("preflight snapshot", f"closed fields changed; missing={missing}, extra={extra}")
    for key, expected in FIXED_PREFLIGHT.items():
        if values[key] != expected:
            _fail(f"preflight snapshot.{key}", f"expected {expected!r}, got {values[key]!r}")
    if values["git_revision"] != revision:
        _fail("preflight snapshot.git_revision", "does not match selected source revision")
    if _strict_decimal(values["memory_used_mib"], "preflight.memory_used_mib") > 256:
        _fail("preflight.memory_used_mib", "idle VRAM exceeds 256 MiB")
    if _strict_decimal(values["temperature_c"], "preflight.temperature_c") > 50:
        _fail("preflight.temperature_c", "start temperature exceeds 50 C")
    available = _strict_decimal(
        values["staging_available_bytes"], "preflight.staging_available_bytes"
    )
    minimum = _strict_decimal(
        values["staging_minimum_bytes"], "preflight.staging_minimum_bytes"
    )
    if available < minimum:
        _fail("preflight.staging_available_bytes", "is below the reviewed minimum")
    return values


def validate_preflights(paths: Sequence[Path], revision: str) -> list[dict[str, str]]:
    if not 1 <= len(paths) <= 5:
        _fail("--preflight", "requires between one and five accepted snapshots")
    if len({str(path.resolve()) for path in paths}) != len(paths):
        _fail("--preflight", "each run must supply a distinct receipt path")
    snapshots = [_parse_preflight(path, revision) for path in paths]
    stable_fields = ("power_limit_w", "graphics_clock_mhz", "memory_clock_mhz")
    reference = {field: snapshots[0][field] for field in stable_fields}
    for index, snapshot in enumerate(snapshots[1:], 2):
        for field, expected in reference.items():
            if snapshot[field] != expected:
                _fail(
                    f"preflight[{index}].{field}",
                    f"changed across independent runs: {snapshot[field]!r} != {expected!r}",
                )
    return snapshots


def validate_gpu_csv(path: Path) -> dict[str, Any]:
    try:
        raw = _snapshot(path, "GPU snapshot", maximum=256 * 1024)
        rows = list(
            csv.reader(io.StringIO(raw.decode("utf-8", errors="strict"), newline=""))
        )
    except (UnicodeDecodeError, csv.Error) as error:
        _fail("GPU snapshot", f"cannot parse CSV: {error}")
    if len(rows) != 1 or len(rows[0]) != 6:
        _fail("GPU snapshot", "must contain exactly one six-column GPU row")
    name, uuid, pci_bus_id, memory_mib, driver, capability = (
        value.strip() for value in rows[0]
    )
    expected = {
        "name": GPU_NAME,
        "uuid": GPU_UUID,
        "pci_bus_id": GPU_PCI_BUS_ID,
        "driver_version": DRIVER_VERSION,
        "compute_capability": GPU_COMPUTE_CAPABILITY,
    }
    actual = {
        "name": name,
        "uuid": uuid,
        "pci_bus_id": pci_bus_id,
        "driver_version": driver,
        "compute_capability": capability,
    }
    for key, value in expected.items():
        if actual[key] != value:
            _fail(f"GPU snapshot.{key}", f"expected {value!r}, got {actual[key]!r}")
    if _strict_decimal(memory_mib, "GPU snapshot.memory_total_mib") != GPU_MEMORY_MIB:
        _fail("GPU snapshot.memory_total_mib", f"must equal {GPU_MEMORY_MIB}")
    return {**actual, "memory_total_mib": GPU_MEMORY_MIB}


def validate_image_inspect(path: Path, image_id: str) -> dict[str, Any]:
    document = _load_json(path, "optimizer image inspect")
    if not isinstance(document, list) or len(document) != 1:
        _fail("optimizer image inspect", "must contain exactly one image object")
    image = document[0]
    if not isinstance(image, dict):
        _fail("optimizer image inspect[0]", "must be an object")
    expected = {"Id": image_id, "Os": "linux", "Architecture": "amd64"}
    for key, value in expected.items():
        if image.get(key) != value:
            _fail(f"optimizer image inspect.{key}", f"expected {value!r}")
    config = image.get("Config")
    if not isinstance(config, dict):
        _fail("optimizer image inspect.Config", "must be an object")
    environment = config.get("Env")
    if not isinstance(environment, list) or not all(
        isinstance(value, str) for value in environment
    ):
        _fail("optimizer image inspect.Config.Env", "must be a string array")
    if f"CUDA_VERSION={CUDA_RUNTIME_VERSION}" not in environment:
        _fail(
            "optimizer image inspect.Config.Env",
            f"must contain CUDA_VERSION={CUDA_RUNTIME_VERSION}",
        )
    if config.get("WorkingDir") != "/workspace":
        _fail("optimizer image inspect.Config.WorkingDir", "must equal /workspace")
    environment_map = _container_environment(
        environment, "optimizer image inspect.Config.Env"
    )
    labels = config.get("Labels")
    if not isinstance(labels, dict) or not all(
        isinstance(name, str) and name and isinstance(value, str)
        for name, value in labels.items()
    ):
        _fail("optimizer image inspect.Config.Labels", "must be an exact string map")
    return {
        "id": image_id,
        "platform": "linux/amd64",
        "cuda_runtime_version": CUDA_RUNTIME_VERSION,
        "environment": environment_map,
        "labels": dict(labels),
    }


def _container_document(path: Path, label: str) -> dict[str, Any]:
    document = _load_json(path, label)
    if not isinstance(document, list) or len(document) != 1:
        _fail(label, "must contain exactly one container object")
    container = document[0]
    if not isinstance(container, dict):
        _fail(f"{label}[0]", "must be an object")
    return container


def _container_environment(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _fail(label, "must be a string array")
    result: dict[str, str] = {}
    for item in value:
        name, separator, setting = item.partition("=")
        if not separator or not name or name in result:
            _fail(label, "contains an empty, malformed, or duplicate variable")
        result[name] = setting
    forbidden = performance._runner_forbidden_environment(list(result))
    if forbidden:
        _fail(label, f"contains forbidden control-plane overrides: {forbidden}")
    return result


def _validate_container_common(
    container: Mapping[str, Any],
    *,
    label: str,
    pair_index: int,
    image_id: str,
    expected_environment: Mapping[str, str],
    expected_mount_sources: Mapping[str, str],
    capture_id: str,
    expected_labels: Mapping[str, str],
) -> dict[str, Any]:
    container_id = container.get("Id")
    if not isinstance(container_id, str) or CONTAINER_ID_RE.fullmatch(container_id) is None:
        _fail(f"{label}.Id", "must be a lowercase 64-character container ID")
    if container.get("Image") != image_id:
        _fail(f"{label}.Image", "does not equal the immutable optimizer image ID")
    if container.get("Path") != performance.RUNNER_CONTAINER_ENTRYPOINT[0]:
        _fail(f"{label}.Path", "must equal the exact resolved executable")
    if not performance._exact_json_value(
        container.get("Args"), performance.RUNNER_CONTAINER_CMD
    ):
        _fail(f"{label}.Args", "must equal the exact resolved arguments")

    config = container.get("Config")
    if not isinstance(config, dict):
        _fail(f"{label}.Config", "must be an object")
    if config.get("Image") != image_id:
        _fail(f"{label}.Config.Image", "does not equal the immutable optimizer image ID")
    if config.get("User") != "0:0" or config.get("WorkingDir") != "/workspace":
        _fail(f"{label}.Config", "user or working directory differs from the runner")
    if not performance._exact_json_value(config.get("Entrypoint"), ["/bin/bash"]):
        _fail(f"{label}.Config.Entrypoint", "must equal ['/bin/bash']")
    if not performance._exact_json_value(
        config.get("Cmd"), performance.RUNNER_CONTAINER_CMD
    ):
        _fail(f"{label}.Config.Cmd", "must equal the exact reviewed runner command")
    if not performance._exact_json_value(
        config.get("Healthcheck"), {"Test": ["NONE"]}
    ):
        _fail(f"{label}.Config.Healthcheck", "must disable image health checks")
    if not performance._exact_json_value(config.get("Labels"), expected_labels):
        _fail(
            f"{label}.Config.Labels",
            "must equal the exact image labels plus supervisor label",
        )
    environment = _container_environment(config.get("Env"), f"{label}.Config.Env")
    required_environment = dict(expected_environment)
    required_environment["RILEY_PERF_PAIR_INDEX"] = str(pair_index)
    required_environment["RILEY_PERF_CAPTURE_ID"] = capture_id
    if not performance._exact_json_value(environment, required_environment):
        _fail(f"{label}.Config.Env", "must equal the exact image env plus runner overrides")

    host = container.get("HostConfig")
    if not isinstance(host, dict):
        _fail(f"{label}.HostConfig", "must be an object")
    if host.get("NetworkMode") != "none":
        _fail(f"{label}.HostConfig.NetworkMode", "must equal 'none'")
    if host.get("ReadonlyRootfs") is not True:
        _fail(f"{label}.HostConfig.ReadonlyRootfs", "must be true")
    if host.get("AutoRemove") is not False:
        _fail(f"{label}.HostConfig.AutoRemove", "must be false so receipts are capturable")
    exact_host = {
        "CapDrop": ["ALL"],
        "CapAdd": None,
        "SecurityOpt": ["no-new-privileges:true"],
        "PidsLimit": 512,
        "Privileged": False,
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
        if name not in host or not performance._exact_json_value(host[name], expected):
            _fail(f"{label}.HostConfig.{name}", f"expected {expected!r}")
    restart = host.get("RestartPolicy")
    if not performance._exact_json_value(
        restart, {"Name": "no", "MaximumRetryCount": 0}
    ):
        _fail(f"{label}.HostConfig.RestartPolicy", "must disable all restarts")
    requests = host.get("DeviceRequests")
    if not isinstance(requests, list) or len(requests) != 1:
        _fail(f"{label}.HostConfig.DeviceRequests", "must contain exactly one GPU request")
    request = requests[0]
    if not isinstance(request, dict):
        _fail(f"{label}.HostConfig.DeviceRequests[0]", "must be an object")
    if not performance._exact_json_value(request.get("Driver"), ""):
        _fail(f"{label}.HostConfig.DeviceRequests[0].Driver", "must equal Docker's probed empty driver")
    if not performance._exact_json_value(request.get("DeviceIDs"), [GPU_UUID]):
        _fail(
            f"{label}.HostConfig.DeviceRequests[0].DeviceIDs",
            "must select only the designated GPU UUID",
        )
    capabilities = request.get("Capabilities")
    if not performance._exact_json_value(capabilities, [["gpu"]]):
        _fail(
            f"{label}.HostConfig.DeviceRequests[0].Capabilities",
            "must contain only the GPU capability",
        )
    if not performance._exact_json_value(
        request.get("Count"), 0
    ) or not performance._exact_json_value(request.get("Options"), {}):
        _fail(
            f"{label}.HostConfig.DeviceRequests[0]",
            "must select the UUID without count-based or option-based widening",
        )

    networks = container.get("NetworkSettings")
    attached_networks = networks.get("Networks") if isinstance(networks, dict) else None
    if not isinstance(attached_networks, dict) or not set(attached_networks) <= {"none"}:
        _fail(
            f"{label}.NetworkSettings.Networks",
            "may contain only Docker's isolated 'none' network receipt",
        )
    isolated_network = attached_networks.get("none")
    if isolated_network is not None:
        if not isinstance(isolated_network, dict) or any(
            isolated_network.get(field) not in {"", None}
            for field in ("Gateway", "IPAddress", "GlobalIPv6Address", "MacAddress")
        ):
            _fail(
                f"{label}.NetworkSettings.Networks.none",
                "must not assign an address, gateway, or MAC",
            )

    mounts = container.get("Mounts")
    if not isinstance(mounts, list) or len(mounts) not in {6, 7}:
        _fail(f"{label}.Mounts", "must contain only the reviewed mounts")
    by_destination: dict[str, Mapping[str, Any]] = {}
    for mount in mounts:
        if not isinstance(mount, dict):
            _fail(f"{label}.Mounts", "entries must be objects")
        destination = mount.get("Destination")
        if not isinstance(destination, str) or destination in by_destination:
            _fail(f"{label}.Mounts", "destinations must be unique strings")
        by_destination[destination] = mount
    expected_destinations = {*expected_mount_sources, "/workspace"}
    actual_destinations = set(by_destination)
    if actual_destinations not in (
        expected_destinations,
        expected_destinations | {"/tmp"},
    ):
        _fail(f"{label}.Mounts", "destinations differ from the reviewed inventory")
    for destination, source in expected_mount_sources.items():
        mount = by_destination[destination]
        expected_rw = destination == "/evidence"
        if (
            mount.get("Type") != "bind"
            or mount.get("Source") != source
            or mount.get("RW") is not expected_rw
            or mount.get("Mode") != ""
            or mount.get("Propagation") != "rprivate"
        ):
            _fail(
                f"{label}.Mounts[{destination}]",
                "source, type, access mode, or propagation mismatch",
            )
    workspace = by_destination["/workspace"]
    if (
        workspace.get("Type") != "volume"
        or workspace.get("RW") is not True
        or not isinstance(workspace.get("Source"), str)
        or not workspace["Source"]
    ):
        _fail(f"{label}.Mounts[/workspace]", "must be a writable fresh volume")
    if "/tmp" in by_destination:
        temporary = by_destination["/tmp"]
        if temporary.get("Type") != "tmpfs" or temporary.get("RW") is not True:
            _fail(f"{label}.Mounts[/tmp]", "must be the reviewed writable tmpfs")
    created_at = container.get("Created")
    try:
        performance._runner_timestamp_ns(created_at, f"{label}.Created")
    except performance.InputError as error:
        _fail(f"{label}.Created", str(error))
    return {
        "container_id": container_id,
        "workspace_volume_name": workspace["Source"],
        "created_at_utc": created_at,
    }


def validate_container_receipts(
    before_paths: Sequence[Path],
    after_paths: Sequence[Path],
    *,
    image_id: str,
    expected_environment: Mapping[str, str],
    expected_mount_sources: Mapping[str, str],
    expected_evidence_sources: Sequence[str] | None = None,
    supervisor_token: str,
    capture_ids: Sequence[str],
    image_labels: Mapping[str, str],
) -> dict[str, Any]:
    if len(before_paths) != 5 or len(after_paths) != 5:
        _fail("container receipts", "requires exactly five before/after pairs")
    if expected_evidence_sources is not None and (
        len(expected_evidence_sources) != 5
        or len(set(expected_evidence_sources)) != 5
    ):
        _fail("container receipts", "requires five distinct evidence mount sources")
    if len(capture_ids) != 5 or len(set(capture_ids)) != 5:
        _fail("container receipts", "requires five distinct capture IDs")
    container_ids: list[str] = []
    workspace_volumes: list[str] = []
    execution_facts: list[dict[str, Any]] = []
    expected_labels = {
        **image_labels,
        performance.RUNNER_SUPERVISOR_LABEL: supervisor_token,
    }
    for pair_index, (before_path, after_path) in enumerate(
        zip(before_paths, after_paths, strict=True), 1
    ):
        before_label = f"container receipt[{pair_index}].before"
        after_label = f"container receipt[{pair_index}].after"
        before = _container_document(before_path, before_label)
        after = _container_document(after_path, after_label)
        mount_sources = dict(expected_mount_sources)
        if expected_evidence_sources is not None:
            mount_sources["/evidence"] = expected_evidence_sources[pair_index - 1]
        before_facts = _validate_container_common(
            before,
            label=before_label,
            pair_index=pair_index,
            image_id=image_id,
            expected_environment=expected_environment,
            expected_mount_sources=mount_sources,
            capture_id=capture_ids[pair_index - 1],
            expected_labels=expected_labels,
        )
        after_facts = _validate_container_common(
            after,
            label=after_label,
            pair_index=pair_index,
            image_id=image_id,
            expected_environment=expected_environment,
            expected_mount_sources=mount_sources,
            capture_id=capture_ids[pair_index - 1],
            expected_labels=expected_labels,
        )
        if (
            before_facts["container_id"] != after_facts["container_id"]
            or before_facts["workspace_volume_name"]
            != after_facts["workspace_volume_name"]
            or before_facts["created_at_utc"] != after_facts["created_at_utc"]
        ):
            _fail(f"container receipt[{pair_index}]", "before/after identities differ")
        before_state = before.get("State")
        expected_before_state = {
            "Status": "created",
            "Running": False,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "Pid": 0,
            "ExitCode": 0,
            "Error": "",
            "StartedAt": performance.RUNNER_ZERO_TIME,
            "FinishedAt": performance.RUNNER_ZERO_TIME,
        }
        if not isinstance(before_state, dict) or any(
            name not in before_state
            or not performance._exact_json_value(before_state[name], expected)
            for name, expected in expected_before_state.items()
        ):
            _fail(before_label, "must capture a pristine not-yet-started container")
        after_state = after.get("State")
        expected_after_state = {
            "Status": "exited",
            "Running": False,
            "Paused": False,
            "Restarting": False,
            "OOMKilled": False,
            "Dead": False,
            "ExitCode": 0,
            "Error": "",
        }
        if not isinstance(after_state, dict) or any(
            name not in after_state
            or not performance._exact_json_value(after_state[name], expected)
            for name, expected in expected_after_state.items()
        ):
            _fail(after_label, "must capture a clean exit-zero container")
        try:
            created_ns = performance._runner_timestamp_ns(
                after_facts["created_at_utc"], f"{after_label}.Created"
            )
            started_at = after_state.get("StartedAt")
            finished_at = after_state.get("FinishedAt")
            started_ns = performance._runner_timestamp_ns(
                started_at, f"{after_label}.State.StartedAt"
            )
            finished_ns = performance._runner_timestamp_ns(
                finished_at, f"{after_label}.State.FinishedAt"
            )
        except performance.InputError as error:
            _fail(after_label, str(error))
        if not created_ns <= started_ns <= finished_ns:
            _fail(after_label, "Created/StartedAt/FinishedAt order is invalid")
        if not performance._exact_json_value(
            before.get("RestartCount"), 0
        ) or not performance._exact_json_value(after.get("RestartCount"), 0):
            _fail(f"container receipt[{pair_index}].RestartCount", "must remain zero")
        container_ids.append(before_facts["container_id"])
        workspace_volumes.append(before_facts["workspace_volume_name"])
        execution_facts.append(
            {
                "container_id": before_facts["container_id"],
                "created_at_utc": after_facts["created_at_utc"],
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "exit_code": after_state["ExitCode"],
                "oom_killed": after_state["OOMKilled"],
            }
        )
    if len(set(container_ids)) != 5:
        _fail("container receipts", "all five runs must use distinct fresh containers")
    if len(set(workspace_volumes)) != 5:
        _fail("container receipts", "all five runs must use distinct fresh workspace volumes")
    return {
        "count": 5,
        "distinct_container_ids": container_ids,
        "distinct_workspace_volumes": workspace_volumes,
        "execution_facts": execution_facts,
    }


def validate_gpu_monitors(
    paths: Sequence[Path],
    *,
    capture_ids: Sequence[str],
    container_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if len(paths) != 5:
        _fail("--gpu-monitor", "requires exactly five per-run monitor receipts")
    if len({str(path.resolve()) for path in paths}) != 5:
        _fail("--gpu-monitor", "each run must supply a distinct receipt path")
    results: list[dict[str, Any]] = []
    for index, path in enumerate(paths, 1):
        raw = _snapshot(path, f"GPU monitor receipt {index}", maximum=256 * 1024)
        try:
            results.append(
                {
                    **performance._runner_gpu_monitor(
                        raw,
                        f"run-{index}/gpu-monitor.csv",
                        expected_capture_id=capture_ids[index - 1],
                        expected_container_id=container_ids[index - 1],
                    ),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        except performance.InputError as error:
            _fail(f"--gpu-monitor[{index}]", str(error))
    return results


def canonical_model_tree_sha256(model_dir: Path) -> tuple[str, int]:
    try:
        root_metadata = model_dir.lstat()
    except OSError as error:
        _fail("model directory", f"cannot inspect: {error}")
    if not stat.S_ISDIR(root_metadata.st_mode):
        _fail("model directory", "must be a real directory, not a link")
    entries: list[tuple[str, Path]] = []
    try:
        for root, directories, files in os.walk(model_dir, followlinks=False):
            root_path = Path(root)
            for name in directories:
                path = root_path / name
                if not stat.S_ISDIR(path.lstat().st_mode):
                    _fail("model directory", f"non-directory entry: {path}")
            for name in files:
                path = root_path / name
                metadata = path.lstat()
                if not stat.S_ISREG(metadata.st_mode):
                    _fail("model directory", f"non-regular entry: {path}")
                relative = path.relative_to(model_dir).as_posix()
                if SAFE_MODEL_PATH_RE.fullmatch(relative) is None:
                    _fail("model directory", f"unsafe relative path: {relative!r}")
                entries.append((relative, path))
    except OSError as error:
        _fail("model directory", f"cannot walk tree: {error}")
    if not entries:
        _fail("model directory", "must contain regular files")
    manifest = bytearray()
    for relative, path in sorted(entries):
        manifest.extend(f"{_sha256(path, f'model/{relative}')}  {relative}\n".encode("ascii"))
    return hashlib.sha256(manifest).hexdigest(), len(entries)


def validate_optimizer_report(
    path: Path,
    *,
    expected_sha256: str,
    revision: str,
    source_archive_sha256: str,
    image_digest: str,
    model_tree_sha256: str,
    gpu: Mapping[str, Any],
) -> dict[str, Any]:
    actual_sha256 = _sha256(path, "optimizer correctness report")
    if actual_sha256 != expected_sha256:
        _fail("optimizer correctness report", "does not match its external SHA-256")
    document = _load_json(path, "optimizer correctness report")
    if not isinstance(document, dict):
        _fail("optimizer correctness report", "root must be an object")
    baseline_document, baseline_raw = performance._load_json_bytes(
        BASELINE_PATH, "performance baseline"
    )
    baseline = performance._validate_baseline(baseline_document, baseline_raw)
    candidate = {
        "source": {
            "git_commit": revision,
            "source_archive_sha256": source_archive_sha256,
            "profile_image_sha256": image_digest,
        },
        "model": baseline["model"],
        "environment": baseline["environment"],
    }
    try:
        performance._validate_optimization_correctness(document, candidate)
    except performance.InputError as error:
        _fail("optimizer correctness report", str(error))
    if document["model"]["manifest_sha256"] != model_tree_sha256:
        _fail("optimizer correctness report.model.manifest_sha256", "model tree mismatch")
    expected_gpu = {
        "model": gpu["name"],
        "uuid": gpu["uuid"],
        "pci_bus_id": gpu["pci_bus_id"],
        "compute_capability": gpu["compute_capability"],
        "vram_mib": gpu["memory_total_mib"],
        "driver_version": gpu["driver_version"],
    }
    if document["gpu"] != expected_gpu:
        _fail("optimizer correctness report.gpu", "does not equal actual designated GPU facts")
    return {"sha256": actual_sha256, "gate_id": GATE_ID, "status": "passed"}


def validate_runs(
    paths: Sequence[Path],
    *,
    revision: str,
    profile_sha256: str,
    optimizer_report_sha256: str,
    image_digest: str,
) -> dict[str, Any]:
    if len(paths) != 5:
        _fail("--run", "requires exactly five raw native profile documents")
    payloads: list[tuple[str, bytes]] = []
    for index, path in enumerate(paths, 1):
        if path.name != f"candidate-{index}.json":
            _fail(f"--run[{index}]", f"must be named candidate-{index}.json")
        payloads.append(
            (
                path.name,
                _snapshot(
                    path,
                    f"candidate run {index}",
                    maximum=performance.native_profile.MAX_EVIDENCE_BYTES,
                ),
            )
        )
    try:
        derived = performance.derive_raw_run_payloads(payloads)
    except (performance.InputError, performance.ComparabilityError) as error:
        _fail("candidate runs", str(error))
    baseline_document, baseline_raw = performance._load_json_bytes(
        BASELINE_PATH, "performance baseline"
    )
    baseline = performance._validate_baseline(baseline_document, baseline_raw)
    try:
        performance._require_request_identity_sha256(
            derived,
            baseline["request_identity_sha256"],
            "candidate runs.request_identity_sha256",
        )
    except performance.ComparabilityError as error:
        _fail("candidate runs.request_identity_sha256", str(error))
    source = derived["source"]
    expected_source = {
        "git_commit": revision,
        "git_dirty": False,
        "executable_sha256": profile_sha256,
        "implementation_id": IMPLEMENTATION_ID,
        "runtime_flag": {"name": "execution_completion", "value": "iteration-batch"},
        "semantic_class": "E0",
        "correctness_gate_id": GATE_ID,
        "correctness_report_sha256": optimizer_report_sha256,
    }
    if source != expected_source:
        _fail("candidate runs.source", "does not equal the reviewed runner binding")
    for field in ("model", "environment", "workload"):
        if derived[field] != baseline[field]:
            _fail(f"candidate runs.{field}", "differs from the immutable baseline lane")
    if derived["runs"][0]["environment"]["software"]["container_image_sha256"] != image_digest:
        _fail("candidate runs.environment", "optimizer image digest mismatch")
    summary = derived["run_summary"]
    if summary != {
        "independent_runs": 5,
        "warmups_per_run": 5,
        "measured_iterations_per_run": 30,
        "failure_count": 0,
        "dropped_trace_records": 0,
    }:
        _fail("candidate runs.run_summary", "must be exact 5x(5 warmup + 30 measured) success")
    return {
        "count": 5,
        "raw_runs": derived["raw_runs"],
        "execution_runs": [
            {
                "pair_index": run["pair_index"],
                "run_id": run["run_id"],
                "recorded_at_utc": run["recorded_at_utc"],
                "sha256": binding["sha256"],
            }
            for run, binding in zip(
                derived["runs"], derived["raw_runs"], strict=True
            )
        ],
        "request_identity_sha256": derived["request_identity_sha256"],
        "metrics": derived["metrics"],
        "run_summary": summary,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    revision = _expected_revision(args.source_revision)
    image_id, image_digest = _expected_image(args.optimizer_image_id)
    source_sha = _expected_sha(
        args.expected_source_archive_sha256, "--expected-source-archive-sha256"
    )
    profile_sha = _expected_sha(
        args.expected_profile_binary_sha256, "--expected-profile-binary-sha256"
    )
    report_sha = _expected_sha(
        args.expected_optimizer_correctness_report_sha256,
        "--expected-optimizer-correctness-report-sha256",
    )
    model_tree_sha = _expected_sha(
        args.expected_model_tree_sha256, "--expected-model-tree-sha256"
    )
    supervisor_token: str | None = None
    capture_ids: list[str] = []
    if args.mode == "final":
        if args.supervisor_token is None:
            _fail("--supervisor-token", "is required in final mode")
        supervisor_token = _expected_sha(args.supervisor_token, "--supervisor-token")
        if args.capture_id is None or len(args.capture_id) != 5:
            _fail("--capture-id", "requires exactly five values")
        capture_ids = [
            _expected_sha(value, f"--capture-id[{index}]")
            for index, value in enumerate(args.capture_id, 1)
        ]
        expected_capture_ids = [
            performance._runner_capture_id(supervisor_token, pair_index)
            for pair_index in range(1, 6)
        ]
        if capture_ids != expected_capture_ids:
            _fail("--capture-id", "must be the canonical supervisor-token derivation")
    if _sha256(args.source_archive, "source archive") != source_sha:
        _fail("source archive", "does not match its external SHA-256")
    if _sha256(
        args.profile_binary,
        "reproducible profile binary",
        executable=True,
    ) != profile_sha:
        _fail("reproducible profile binary", "does not match its external SHA-256")
    actual_model_tree_sha, model_file_count = canonical_model_tree_sha256(args.model_dir)
    if actual_model_tree_sha != model_tree_sha:
        _fail("model directory", "canonical tree SHA-256 mismatch")
    if _sha256(args.model_dir / "model.safetensors", "model weights") != WEIGHTS_SHA256:
        _fail("model weights", "reviewed SmolLM2 weights mismatch")
    if _sha256(args.model_dir / "tokenizer.json", "tokenizer") != TOKENIZER_SHA256:
        _fail("tokenizer", "reviewed SmolLM2 tokenizer mismatch")

    preflights = validate_preflights(args.preflight, revision)
    if args.mode == "preflight":
        if len(preflights) != 1:
            _fail("--preflight", "preflight mode requires exactly one snapshot")
        if any(
            value
            for value in (
                args.run,
                args.container_inspect_before,
                args.container_inspect_after,
                args.gpu_monitor,
                args.image_inspect_after,
                args.runner_manifest_output,
                args.execution_receipt_output,
                args.supervisor_token,
                args.capture_id,
                args.tool,
            )
        ):
            _fail("--mode preflight", "final-run receipts are forbidden")
    elif len(preflights) != 5:
        _fail("--preflight", "final mode requires exactly five distinct snapshots")
    gpu = validate_gpu_csv(args.gpu_csv)
    image = validate_image_inspect(args.image_inspect, image_id)
    image_after = None
    if args.mode == "final":
        if args.image_inspect_after is None:
            _fail("--image-inspect-after", "is required in final mode")
        image_after = validate_image_inspect(args.image_inspect_after, image_id)
        if image_after != image:
            _fail("optimizer image inspect", "before/after image config changed")
    optimizer = validate_optimizer_report(
        args.optimizer_correctness_report,
        expected_sha256=report_sha,
        revision=revision,
        source_archive_sha256=source_sha,
        image_digest=image_digest,
        model_tree_sha256=model_tree_sha,
        gpu=gpu,
    )
    runs = None
    containers = None
    gpu_monitors = None
    if bool(args.container_inspect_before) != bool(args.container_inspect_after):
        _fail(
            "container receipts",
            "before and after receipts must be supplied together",
        )
    if args.container_inspect_before:
        if not args.run:
            _fail("container receipts", "may be supplied only with the five raw runs")
        evidence_sources = [str(path.parent.resolve()) for path in args.run]
        if len(set(evidence_sources)) != 5:
            _fail("--run", "each raw run must use a distinct evidence directory")
        full_environment = {
            **image["environment"],
            "RILEY_PERF_SOURCE_REVISION": revision,
            "RILEY_PERF_SOURCE_ARCHIVE_SHA256": source_sha,
            "RILEY_PERF_PROFILE_BINARY_SHA256": profile_sha,
            "RILEY_PERF_OPTIMIZER_REPORT_SHA256": report_sha,
            "RILEY_PERF_OPTIMIZER_IMAGE_SHA256": image_digest,
            "RILEY_PERF_MODEL_TREE_SHA256": model_tree_sha,
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
        read_only_mount_sources = {
            "/input/source.tar": str(args.source_archive.resolve()),
            "/input/riley-profile": str(args.profile_binary.resolve()),
            "/input/optimizer-correctness-report.json": str(
                args.optimizer_correctness_report.resolve()
            ),
            "/model": str(args.model_dir.resolve()),
        }
        containers = validate_container_receipts(
            args.container_inspect_before,
            args.container_inspect_after,
            image_id=image_id,
            expected_environment=full_environment,
            expected_mount_sources=read_only_mount_sources,
            expected_evidence_sources=evidence_sources,
            supervisor_token=supervisor_token,
            capture_ids=capture_ids,
            image_labels=image["labels"],
        )
    if args.run:
        if containers is None:
            _fail("--run", "requires all five before/after container inspect receipts")
        runs = validate_runs(
            args.run,
            revision=revision,
            profile_sha256=profile_sha,
            optimizer_report_sha256=report_sha,
            image_digest=image_digest,
        )
    if args.gpu_monitor:
        if args.mode != "final":
            _fail("--gpu-monitor", "is accepted only in final mode")
        if containers is None:
            _fail(
                "--gpu-monitor",
                "requires all five before/after container inspect receipts",
            )
        gpu_monitors = validate_gpu_monitors(
            args.gpu_monitor,
            capture_ids=capture_ids,
            container_ids=containers["distinct_container_ids"],
        )
    if args.mode == "final" and (
        runs is None
        or containers is None
        or gpu_monitors is None
        or args.runner_manifest_output is None
        or args.execution_receipt_output is None
        or len(args.execution_receipt_output) != 5
        or supervisor_token is None
        or len(capture_ids) != 5
        or len(args.tool) != len(performance.RUNNER_REQUIRED_TOOLS)
    ):
        _fail(
            "--mode final",
            "requires all five runs/receipts, manifest output, and trusted tool paths",
        )

    runner_manifest = None
    execution_receipts: list[dict[str, Any]] | None = None
    if args.mode == "final":
        execution_receipts = []
        for pair_index in range(1, 6):
            execution_run = runs["execution_runs"][pair_index - 1]
            container_facts = containers["execution_facts"][pair_index - 1]
            expected_run_id = performance._runner_run_id(
                revision, capture_ids[pair_index - 1], pair_index
            )
            if execution_run["run_id"] != expected_run_id:
                _fail(
                    f"candidate run {pair_index}.run_id",
                    "does not bind the canonical per-pair capture ID",
                )
            execution = {
                "schema_version": performance.RUNNER_EXECUTION_SCHEMA,
                "pair_index": pair_index,
                "capture_id": capture_ids[pair_index - 1],
                "container_id": container_facts["container_id"],
                "run_id": execution_run["run_id"],
                "candidate_recorded_at_utc": execution_run["recorded_at_utc"],
                "docker": {
                    "created_at_utc": container_facts["created_at_utc"],
                    "started_at_utc": container_facts["started_at_utc"],
                    "finished_at_utc": container_facts["finished_at_utc"],
                    "exit_code": container_facts["exit_code"],
                    "oom_killed": container_facts["oom_killed"],
                },
                "sha256": {
                    "preflight": _sha256(
                        args.preflight[pair_index - 1],
                        f"preflight receipt {pair_index}",
                    ),
                    "candidate": execution_run["sha256"],
                    "gpu_monitor": gpu_monitors[pair_index - 1]["sha256"],
                    "container_inspect_before": _sha256(
                        args.container_inspect_before[pair_index - 1],
                        f"container receipt {pair_index} before",
                    ),
                    "container_inspect_after": _sha256(
                        args.container_inspect_after[pair_index - 1],
                        f"container receipt {pair_index} after",
                    ),
                },
            }
            try:
                execution = performance._runner_execution(
                    execution, f"execution receipt {pair_index}"
                )
                performance._validate_runner_execution_timeline(
                    execution, label=f"execution receipt {pair_index}"
                )
            except performance.InputError as error:
                _fail(f"execution receipt {pair_index}", str(error))
            execution_receipts.append(execution)
        for previous, current in zip(
            execution_receipts, execution_receipts[1:], strict=False
        ):
            if performance._runner_timestamp_ns(
                previous["docker"]["finished_at_utc"], "previous FinishedAt"
            ) > performance._runner_timestamp_ns(
                current["docker"]["created_at_utc"], "next Created"
            ):
                _fail("execution receipts", "sequential pair timelines overlap")
        tools: dict[str, dict[str, str]] = {}
        for receipt in args.tool:
            name, separator, path_text = receipt.partition("=")
            if not separator or name in tools or name not in performance.RUNNER_REQUIRED_TOOLS:
                _fail("--tool", "expected one unique reviewed NAME=/absolute/path")
            tool_path = Path(path_text)
            expected_tool = performance.RUNNER_REVIEWED_TOOLS[name]
            if path_text != expected_tool["path"]:
                _fail(f"--tool {name}", "does not equal the reviewed exact path")
            try:
                tool_metadata = tool_path.lstat()
            except OSError as error:
                _fail(f"--tool {name}", f"cannot inspect trusted tool: {error}")
            if (
                not stat.S_ISREG(tool_metadata.st_mode)
                or tool_metadata.st_uid != 0
                or tool_metadata.st_mode & 0o022
            ):
                _fail(
                    f"--tool {name}",
                    "must be a non-link root-owned regular file and not group/world writable",
                )
            actual_tool_sha256 = _sha256(tool_path, f"tool {name}")
            if actual_tool_sha256 != expected_tool["sha256"]:
                _fail(f"--tool {name}", "does not equal the reviewed exact SHA-256")
            tools[name] = dict(expected_tool)
        if set(tools) != performance.RUNNER_REQUIRED_TOOLS:
            _fail("--tool", "exact trusted tool inventory required")
        manifest_environment = dict(full_environment)
        manifest_environment["RILEY_PERF_PAIR_INDEX"] = "{pair_index}"
        manifest_environment["RILEY_PERF_CAPTURE_ID"] = "{capture_id}"
        runner_manifest = {
            "schema_version": performance.RUNNER_MANIFEST_SCHEMA,
            "candidate": {
                "source_revision": revision,
                "source_archive_sha256": source_sha,
                "profile_binary_sha256": profile_sha,
                "model_tree_sha256": model_tree_sha,
                "optimizer_correctness_report_sha256": report_sha,
                "optimizer_image_id": image_id,
            },
            "runner": {
                "revision": revision,
                "host_script_sha256": _sha256(
                    ROOT / "ci" / "run_remote_release_performance.sh",
                    "host runner script",
                ),
                "inner_script_sha256": _sha256(
                    ROOT / "ci" / "release" / "run_release_performance_once.sh",
                    "inner runner script",
                ),
                "tools": tools,
            },
            "container": {
                "entrypoint": performance.RUNNER_CONTAINER_ENTRYPOINT,
                "cmd": performance.RUNNER_CONTAINER_CMD,
                "environment": manifest_environment,
                "read_only_mount_sources": read_only_mount_sources,
                "evidence_mount_sources": evidence_sources,
                "workspace_volume_names": containers["distinct_workspace_volumes"],
                "supervisor_label": {
                    "name": performance.RUNNER_SUPERVISOR_LABEL,
                    "value": supervisor_token,
                },
                "labels": {
                    **image["labels"],
                    performance.RUNNER_SUPERVISOR_LABEL: supervisor_token,
                },
            },
            "executions": execution_receipts,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "bindings": {
            "source_revision": revision,
            "source_archive_sha256": source_sha,
            "profile_binary_sha256": profile_sha,
            "optimizer_image_id": image_id,
            "optimizer_correctness_report_sha256": report_sha,
            "model_tree_sha256": model_tree_sha,
        },
        "host": {
            "accepted_preflight_snapshots": len(preflights),
            "kernel_release": preflights[0]["kernel_release"],
            "cpu_model": preflights[0]["cpu_model"],
            "physical_cpu_cores": int(preflights[0]["physical_cpu_cores"]),
            "logical_cpu_threads": int(preflights[0]["logical_cpu_threads"]),
            "ram_bytes": int(preflights[0]["ram_bytes"]),
            "power_limit_w": preflights[0]["power_limit_w"],
            "graphics_clock_mhz": preflights[0]["graphics_clock_mhz"],
            "memory_clock_mhz": preflights[0]["memory_clock_mhz"],
        },
        "gpu": gpu,
        "image": image,
        "image_after": image_after,
        "model_file_count": model_file_count,
        "optimizer_correctness": optimizer,
        "containers": containers,
        "gpu_monitors": gpu_monitors,
        "raw_runs": runs,
        "runner_manifest": runner_manifest,
        "execution_receipts": execution_receipts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "final"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--profile-binary", required=True, type=Path)
    parser.add_argument("--expected-profile-binary-sha256", required=True)
    parser.add_argument("--optimizer-image-id", required=True)
    parser.add_argument("--image-inspect", required=True, type=Path)
    parser.add_argument("--image-inspect-after", type=Path)
    parser.add_argument("--gpu-csv", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--expected-model-tree-sha256", required=True)
    parser.add_argument("--optimizer-correctness-report", required=True, type=Path)
    parser.add_argument(
        "--expected-optimizer-correctness-report-sha256", required=True
    )
    parser.add_argument("--preflight", required=True, nargs="+", type=Path)
    parser.add_argument("--run", nargs=5, type=Path)
    parser.add_argument("--container-inspect-before", nargs=5, type=Path)
    parser.add_argument("--container-inspect-after", nargs=5, type=Path)
    parser.add_argument("--gpu-monitor", nargs=5, type=Path)
    parser.add_argument("--supervisor-token")
    parser.add_argument("--capture-id", nargs=5)
    parser.add_argument("--execution-receipt-output", nargs=5, type=Path)
    parser.add_argument("--runner-manifest-output", type=Path)
    parser.add_argument("--tool", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args)
    except (ContractError, performance.InputError, performance.ComparabilityError) as error:
        print(f"release performance runner validation failed: {error}", file=sys.stderr)
        return 2
    outputs: list[tuple[Path, Mapping[str, Any], str]] = []
    if args.execution_receipt_output is not None:
        outputs.extend(
            (path, receipt, f"execution receipt {index}")
            for index, (path, receipt) in enumerate(
                zip(
                    args.execution_receipt_output,
                    result["execution_receipts"],
                    strict=True,
                ),
                1,
            )
        )
    if args.runner_manifest_output is not None:
        outputs.append(
            (args.runner_manifest_output, result["runner_manifest"], "runner manifest")
        )
    if len({str(path.resolve()) for path, _document, _label in outputs}) != len(outputs):
        print(
            "release performance runner validation failed: output paths must be distinct",
            file=sys.stderr,
        )
        return 2
    for path, _document, _label in outputs:
        if os.path.lexists(path):
            print(
                f"release performance runner validation failed: refusing to overwrite {path}",
                file=sys.stderr,
            )
            return 2
    for path, document, label in outputs:
        encoded = (
            json.dumps(
                document,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        try:
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(encoded)
            path.chmod(0o444)
        except FileExistsError:
            print(
                f"release performance runner validation failed: refusing to overwrite {path}",
                file=sys.stderr,
            )
            return 2
        except OSError as error:
            print(
                f"release performance runner validation failed: cannot publish {label}: {error}",
                file=sys.stderr,
            )
            return 2
    printable = dict(result)
    printable.pop("runner_manifest", None)
    printable.pop("execution_receipts", None)
    print(
        json.dumps(
            printable,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
