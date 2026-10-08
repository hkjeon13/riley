"""Fail-closed, standard-library-only primary-host environment binding.

The probe intentionally runs before importing or loading a model backend.  It
uses only immutable host facts exposed by ``nvidia-smi``, ``/proc`` and
``/etc/os-release``.  The BF16 fact is derived from the observed CUDA compute
capability; each BF16 model backend additionally checks the selected CUDA
device through its pinned runtime before executing a model.
"""

from __future__ import annotations

import math
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from .constants import (
    PRIMARY_CPU_LOGICAL_THREADS,
    PRIMARY_CPU_MODEL,
    PRIMARY_CPU_PHYSICAL_CORES,
    PRIMARY_CPU_GOVERNOR,
    PRIMARY_CPU_GOVERNOR_POLICY_COUNT,
    PRIMARY_DRIVER_CUDA_API_VERSION,
    PRIMARY_ENVIRONMENT_ID,
    PRIMARY_GPU_COMPUTE_CAPABILITY,
    PRIMARY_GPU_COUNT,
    PRIMARY_GPU_MEMORY_MIB,
    PRIMARY_GPU_NAME,
    PRIMARY_KERNEL_RELEASE,
    PRIMARY_MACHINE,
    PRIMARY_NVIDIA_DRIVER_VERSION,
    PRIMARY_OS_ID,
    PRIMARY_OS_VERSION_ID,
    PRIMARY_RAM_BYTES,
)

ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION = "1.0.0"


class EnvironmentContractError(ValueError):
    """The observed host cannot claim the primary environment ID."""


PRIMARY_ENVIRONMENT_SNAPSHOT: dict[str, object] = {
    "schema_version": ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
    "environment_id": PRIMARY_ENVIRONMENT_ID,
    "host": {
        "os_id": PRIMARY_OS_ID,
        "os_version_id": PRIMARY_OS_VERSION_ID,
        "kernel_release": PRIMARY_KERNEL_RELEASE,
        "machine": PRIMARY_MACHINE,
        "cpu_model": PRIMARY_CPU_MODEL,
        "physical_cpu_cores": PRIMARY_CPU_PHYSICAL_CORES,
        "logical_cpu_threads": PRIMARY_CPU_LOGICAL_THREADS,
        "ram_bytes": PRIMARY_RAM_BYTES,
        "cpu_governor": PRIMARY_CPU_GOVERNOR,
        "cpu_governor_policy_count": PRIMARY_CPU_GOVERNOR_POLICY_COUNT,
        "clock_synchronized": True,
    },
    "accelerator": {
        "gpu_count": PRIMARY_GPU_COUNT,
        "gpus": [
            {
                "index": 0,
                "name": PRIMARY_GPU_NAME,
                "compute_capability": PRIMARY_GPU_COMPUTE_CAPABILITY,
                "memory_total_mib": PRIMARY_GPU_MEMORY_MIB,
                "bf16_compute_supported": True,
            }
        ],
        "nvidia_driver_version": PRIMARY_NVIDIA_DRIVER_VERSION,
        "driver_cuda_api_version": PRIMARY_DRIVER_CUDA_API_VERSION,
        "cuda_driver_available": True,
        "host_cuda_toolkit_present": False,
        "primary_compute_dtype": "bf16",
        "persistence_mode": "Disabled",
        "compute_process_count": 0,
        "memory_used_mib": 0,
        "temperature_c": 40,
        "power_limit_w": 450.0,
        "application_graphics_clock_mhz": "[N/A]",
        "application_memory_clock_mhz": "[N/A]",
    },
}

CommandRunner = Callable[[Sequence[str]], str]
TextReader = Callable[[Path], str]
UnameProbe = Callable[[], platform.uname_result]
WhichProbe = Callable[[str], str | None]
GovernorProbe = Callable[[], Sequence[str]]


def _run_text(arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(
            list(arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise EnvironmentContractError(
            f"cannot probe primary host with {' '.join(arguments)!r}: {error}"
        ) from error


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise EnvironmentContractError(f"cannot read host fact {path}: {error}") from error


def _probe_governors() -> list[str]:
    paths = sorted(
        Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"),
        key=lambda path: int(path.parent.name.removeprefix("policy")),
    )
    if not paths:
        raise EnvironmentContractError("host exposes no CPU frequency policies")
    return [_read_text(path).strip() for path in paths]


def _os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", maxsplit=1)
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _cpu_facts(text: str) -> tuple[str, int, int]:
    processors: list[dict[str, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", maxsplit=1)
                fields[key.strip()] = value.strip()
        if "processor" in fields:
            processors.append(fields)
    if not processors:
        raise EnvironmentContractError("/proc/cpuinfo contains no processor records")
    raw_models = {record.get("model name", "") for record in processors}
    if len(raw_models) != 1 or not next(iter(raw_models)):
        raise EnvironmentContractError("CPU model is missing or inconsistent across threads")
    raw_model = next(iter(raw_models))
    if "i7-13700K" in raw_model:
        cpu_model = "Intel Core i7-13700K"
    else:
        cpu_model = raw_model
    core_ids = {
        (record.get("physical id", "0"), record["core id"])
        for record in processors
        if "core id" in record
    }
    if not core_ids:
        raise EnvironmentContractError("/proc/cpuinfo lacks physical core identity")
    return cpu_model, len(core_ids), len(processors)


def _memory_bytes(text: str) -> int:
    match = re.search(r"(?m)^MemTotal:\s*([0-9]+)\s+kB\s*$", text)
    if match is None:
        raise EnvironmentContractError("/proc/meminfo lacks MemTotal in kB")
    return int(match.group(1)) * 1024


def _gpu_facts(query_text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(query_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 11:
            raise EnvironmentContractError(
                f"nvidia-smi GPU row {line_number} has {len(fields)} fields, expected 11"
            )
        (
            index_text,
            name,
            driver,
            capability,
            memory_total_text,
            memory_used_text,
            temperature_text,
            persistence_mode,
            power_limit_text,
            graphics_clock_text,
            memory_clock_text,
        ) = fields
        try:
            index = int(index_text)
            memory_total_mib = int(memory_total_text)
            memory_used_mib = int(memory_used_text)
            temperature_c = int(temperature_text)
            power_limit_w = float(power_limit_text)
        except ValueError as error:
            raise EnvironmentContractError(
                f"nvidia-smi GPU row {line_number} has a non-numeric hardware fact"
            ) from error
        if not re.fullmatch(r"[0-9]+\.[0-9]+", capability):
            raise EnvironmentContractError(
                f"nvidia-smi GPU row {line_number} has invalid compute capability"
            )
        major = int(capability.split(".", maxsplit=1)[0])
        rows.append(
            {
                "index": index,
                "name": name,
                "driver_version": driver,
                "compute_capability": capability,
                "memory_total_mib": memory_total_mib,
                "bf16_compute_supported": major >= 8,
                "memory_used_mib": memory_used_mib,
                "temperature_c": temperature_c,
                "persistence_mode": persistence_mode,
                "power_limit_w": power_limit_w,
                "application_graphics_clock_mhz": graphics_clock_text,
                "application_memory_clock_mhz": memory_clock_text,
            }
        )
    if not rows:
        raise EnvironmentContractError("nvidia-smi reported no GPUs")
    if [row["index"] for row in rows] != list(range(len(rows))):
        raise EnvironmentContractError("nvidia-smi GPU indexes must be contiguous from zero")
    return rows


def _exact_equal(observed: object, expected: object, path: str) -> None:
    if type(observed) is not type(expected):
        raise EnvironmentContractError(
            f"{path}: expected type {type(expected).__name__}, got {type(observed).__name__}"
        )
    if isinstance(expected, dict):
        if set(observed) != set(expected):  # type: ignore[arg-type]
            observed_keys = set(observed)  # type: ignore[arg-type]
            raise EnvironmentContractError(
                f"{path}: keys differ; missing={sorted(set(expected) - observed_keys)}, "
                f"extra={sorted(observed_keys - set(expected))}"
            )
        for key, value in expected.items():
            _exact_equal(observed[key], value, f"{path}.{key}")  # type: ignore[index]
        return
    if isinstance(expected, list):
        if len(observed) != len(expected):  # type: ignore[arg-type]
            raise EnvironmentContractError(
                f"{path}: expected {len(expected)} items, got {len(observed)}"  # type: ignore[arg-type]
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(observed, expected, strict=True)  # type: ignore[arg-type]
        ):
            _exact_equal(actual_item, expected_item, f"{path}[{index}]")
        return
    if observed != expected:
        raise EnvironmentContractError(
            f"{path}: expected {expected!r}, got {observed!r}"
        )


def _exact_keys(value: object, expected: set[str], path: str) -> dict[str, object]:
    if type(value) is not dict:
        raise EnvironmentContractError(f"{path}: expected object")
    result = value
    if set(result) != expected:
        raise EnvironmentContractError(
            f"{path}: keys differ; missing={sorted(expected - set(result))}, "
            f"extra={sorted(set(result) - expected)}"
        )
    return result


def _expect_int_range(value: object, path: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise EnvironmentContractError(
            f"{path}: expected integer in [{minimum}, {maximum}], got {value!r}"
        )
    return value


def _expect_positive_number(value: object, path: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)) or value <= 0:
        raise EnvironmentContractError(f"{path}: expected positive finite number")
    return float(value)


def validate_environment_snapshot(value: object, path: str = "observed_environment") -> None:
    """Validate exact host identity plus bounded, fail-closed start conditions."""

    root = _exact_keys(
        value, {"schema_version", "environment_id", "host", "accelerator"}, path
    )
    _exact_equal(root["schema_version"], ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION, f"{path}.schema_version")
    _exact_equal(root["environment_id"], PRIMARY_ENVIRONMENT_ID, f"{path}.environment_id")
    host = _exact_keys(
        root["host"],
        {
            "os_id",
            "os_version_id",
            "kernel_release",
            "machine",
            "cpu_model",
            "physical_cpu_cores",
            "logical_cpu_threads",
            "ram_bytes",
            "cpu_governor",
            "cpu_governor_policy_count",
            "clock_synchronized",
        },
        f"{path}.host",
    )
    expected_host = PRIMARY_ENVIRONMENT_SNAPSHOT["host"]
    if not isinstance(expected_host, dict):
        raise AssertionError("primary host contract must be an object")
    for key, expected in expected_host.items():
        _exact_equal(host[key], expected, f"{path}.host.{key}")

    accelerator = _exact_keys(
        root["accelerator"],
        {
            "gpu_count",
            "gpus",
            "nvidia_driver_version",
            "driver_cuda_api_version",
            "cuda_driver_available",
            "host_cuda_toolkit_present",
            "primary_compute_dtype",
            "persistence_mode",
            "compute_process_count",
            "memory_used_mib",
            "temperature_c",
            "power_limit_w",
            "application_graphics_clock_mhz",
            "application_memory_clock_mhz",
        },
        f"{path}.accelerator",
    )
    expected_accelerator = PRIMARY_ENVIRONMENT_SNAPSHOT["accelerator"]
    if not isinstance(expected_accelerator, dict):
        raise AssertionError("primary accelerator contract must be an object")
    for key in (
        "gpu_count",
        "gpus",
        "nvidia_driver_version",
        "driver_cuda_api_version",
        "cuda_driver_available",
        "host_cuda_toolkit_present",
        "primary_compute_dtype",
        "persistence_mode",
        "compute_process_count",
        "application_graphics_clock_mhz",
        "application_memory_clock_mhz",
    ):
        _exact_equal(
            accelerator[key], expected_accelerator[key], f"{path}.accelerator.{key}"
        )
    _expect_int_range(
        accelerator["memory_used_mib"], f"{path}.accelerator.memory_used_mib", 0, 256
    )
    _expect_int_range(
        accelerator["temperature_c"], f"{path}.accelerator.temperature_c", 0, 50
    )
    _expect_positive_number(
        accelerator["power_limit_w"], f"{path}.accelerator.power_limit_w"
    )


def environment_comparability_signature(value: object) -> dict[str, object]:
    """Return start conditions that must match between paired oracle producers."""

    validate_environment_snapshot(value)
    root = value
    accelerator = root["accelerator"]  # type: ignore[index]
    host = root["host"]  # type: ignore[index]
    return {
        "environment_id": root["environment_id"],  # type: ignore[index]
        "cpu_governor": host["cpu_governor"],  # type: ignore[index]
        "cpu_governor_policy_count": host["cpu_governor_policy_count"],  # type: ignore[index]
        "persistence_mode": accelerator["persistence_mode"],  # type: ignore[index]
        "power_limit_w": accelerator["power_limit_w"],  # type: ignore[index]
        "application_graphics_clock_mhz": accelerator["application_graphics_clock_mhz"],  # type: ignore[index]
        "application_memory_clock_mhz": accelerator["application_memory_clock_mhz"],  # type: ignore[index]
    }


def probe_primary_environment(
    *,
    command_runner: CommandRunner = _run_text,
    text_reader: TextReader = _read_text,
    uname_probe: UnameProbe = platform.uname,
    which_probe: WhichProbe = shutil.which,
    governor_probe: GovernorProbe = _probe_governors,
) -> dict[str, object]:
    """Collect and approve the actual primary host before canonical generation."""

    gpu_rows = _gpu_facts(
        command_runner(
            (
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,compute_cap,memory.total,memory.used,temperature.gpu,persistence_mode,power.limit,clocks.applications.graphics,clocks.applications.memory",
                "--format=csv,noheader,nounits",
            )
        )
    )
    driver_versions = {str(row.pop("driver_version")) for row in gpu_rows}
    if len(driver_versions) != 1:
        raise EnvironmentContractError("GPUs report inconsistent NVIDIA drivers")
    smi_overview = command_runner(("nvidia-smi",))
    cuda_match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", smi_overview)
    if cuda_match is None:
        raise EnvironmentContractError("nvidia-smi overview lacks driver CUDA API version")
    compute_processes = [
        line
        for line in command_runner(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        ).splitlines()
        if line.strip()
    ]
    governors = [value.strip() for value in governor_probe()]
    if not governors:
        raise EnvironmentContractError("host exposes no CPU frequency policies")
    clock_synchronized = command_runner(
        ("timedatectl", "show", "-p", "NTPSynchronized", "--value")
    ).strip()
    os_release = _os_release(text_reader(Path("/etc/os-release")))
    cpu_model, physical_cores, logical_threads = _cpu_facts(
        text_reader(Path("/proc/cpuinfo"))
    )
    ram_bytes = _memory_bytes(text_reader(Path("/proc/meminfo")))
    uname = uname_probe()
    snapshot: dict[str, object] = {
        "schema_version": ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
        "environment_id": PRIMARY_ENVIRONMENT_ID,
        "host": {
            "os_id": os_release.get("ID", ""),
            "os_version_id": os_release.get("VERSION_ID", ""),
            "kernel_release": uname.release,
            "machine": uname.machine,
            "cpu_model": cpu_model,
            "physical_cpu_cores": physical_cores,
            "logical_cpu_threads": logical_threads,
            "ram_bytes": ram_bytes,
            "cpu_governor": (
                governors[0] if len(set(governors)) == 1 else "mixed"
            ),
            "cpu_governor_policy_count": len(governors),
            "clock_synchronized": clock_synchronized == "yes",
        },
        "accelerator": {
            "gpu_count": len(gpu_rows),
            "gpus": [
                {
                    key: row[key]
                    for key in (
                        "index",
                        "name",
                        "compute_capability",
                        "memory_total_mib",
                        "bf16_compute_supported",
                    )
                }
                for row in gpu_rows
            ],
            "nvidia_driver_version": next(iter(driver_versions)),
            "driver_cuda_api_version": cuda_match.group(1),
            "cuda_driver_available": True,
            "host_cuda_toolkit_present": which_probe("nvcc") is not None,
            "primary_compute_dtype": "bf16",
            "persistence_mode": gpu_rows[0]["persistence_mode"],
            "compute_process_count": len(compute_processes),
            "memory_used_mib": gpu_rows[0]["memory_used_mib"],
            "temperature_c": gpu_rows[0]["temperature_c"],
            "power_limit_w": gpu_rows[0]["power_limit_w"],
            "application_graphics_clock_mhz": gpu_rows[0][
                "application_graphics_clock_mhz"
            ],
            "application_memory_clock_mhz": gpu_rows[0][
                "application_memory_clock_mhz"
            ],
        },
    }
    validate_environment_snapshot(snapshot)
    return snapshot


EnvironmentProbe = Callable[[], dict[str, object]]
