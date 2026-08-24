#!/usr/bin/env python3
"""Run the canonical PR-01 repeatability cells as isolated subprocesses.

The runner intentionally uses only the Python standard library.  It creates a
new, repository-external artifact tree, records the complete plan before any
benchmark starts, runs preflight immediately before every single-cell lane
invocation, and finally delegates statistical evaluation to
``check_repeatability.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "benchmarks/results"
CANONICAL_MATRIX = REPOSITORY_ROOT / "benchmarks/matrix.yaml"
CANONICAL_PROMPTS = REPOSITORY_ROOT / "benchmarks/prompts.jsonl"
CANONICAL_PREFLIGHT = REPOSITORY_ROOT / "benchmarks/scripts/preflight.sh"
CANONICAL_CHECKER = REPOSITORY_ROOT / "benchmarks/scripts/check_repeatability.py"
SUPPORTED_LANES = ("hf-transformers", "vllm")
PLACEHOLDERS = (
    "result_dir",
    "independent_run_index",
    "run_id",
    "warm_state",
    "concurrency",
    "prompt_tokens",
    "output_tokens",
)
PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")
EXPECTED_CELLS = (
    {
        "concurrency": 1,
        "prompt_tokens": 128,
        "output_tokens": 32,
        "warm_state": "warm",
    },
    {
        "concurrency": 1,
        "prompt_tokens": 4096,
        "output_tokens": 128,
        "warm_state": "warm",
    },
    {
        "concurrency": 8,
        "prompt_tokens": 128,
        "output_tokens": 32,
        "warm_state": "warm",
    },
    {
        "concurrency": 1,
        "prompt_tokens": 128,
        "output_tokens": 32,
        "warm_state": "cold",
    },
)
EXPECTED_INDEPENDENT_RUNS = 5
PRIMARY_DRIVER_VERSION = "580.173.02"
PRIMARY_PERSISTENCE_MODE = "Disabled"
PRIMARY_CPU_GOVERNOR = "powersave"
PRIMARY_ENVIRONMENT_ID = "rtx4090-ubuntu22-driver580-v1"
PRIMARY_OS_ID = "ubuntu"
PRIMARY_OS_VERSION_ID = "22.04"
PRIMARY_KERNEL_RELEASE = "6.8.0-138-generic"
PRIMARY_MACHINE = "x86_64"
PRIMARY_CPU_MODEL = "Intel Core i7-13700K"
PRIMARY_CPU_PHYSICAL_CORES = 16
PRIMARY_CPU_LOGICAL_THREADS = 24
PRIMARY_CPU_GOVERNOR_POLICY_COUNT = 24
PRIMARY_RAM_BYTES = 67_185_598_464
PRIMARY_GPU_MEMORY_MIB = 24_564
MINIMUM_STAGING_AVAILABLE_BYTES = 20 * 1024 * 1024 * 1024
REPRODUCIBILITY_ENVIRONMENT_KEYS = (
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
)
CACHE_PATH_ENVIRONMENT_KEYS = (
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "HF_HOME",
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
)
REQUIRED_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
RUNNER_CANONICAL_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_CACHE_MAXSIZE": "4294967296",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "UV_OFFLINE": "1",
    "UV_PYTHON": "3.13.15",
    "UV_PYTHON_DOWNLOADS": "never",
    "VLLM_DO_NOT_TRACK": "1",
    "VLLM_NO_USAGE_STATS": "1",
}
PROJECT_ENVIRONMENT_VARIABLE = "UV_PROJECT_ENVIRONMENT"
CANONICAL_PYTHON_VERSION = "3.13.15"
CANONICAL_PYTHON_LINUX_X86_64_SHA256 = (
    "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866"
)
CANONICAL_PYTHON_VERSION_FILE_SHA256 = (
    "861b3dd8083d28f336ef70f6755bc399538ddad627b1d095820ca34cb953cf14"
)
CANONICAL_UV_VERSION = "uv 0.12.5 (x86_64-unknown-linux-gnu)"
CANONICAL_UV_LINUX_X86_64_SHA256 = (
    "b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46"
)
SAFE_INHERITED_ENVIRONMENT_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "TZ",
)
SENSITIVE_RUNTIME_ENVIRONMENT_PREFIXES = (
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "HF_",
    "MKL_",
    "NCCL_",
    "NVIDIA_",
    "OMP_",
    "PYTORCH_",
    "RUSTINFER_",
    "TOKENIZERS_",
    "TORCH_",
    "TRANSFORMERS_",
    "TRITON_",
    "UV_",
    "VLLM_",
)
SENSITIVE_RUNTIME_ENVIRONMENT_NAMES = {
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "PYTHONHOME",
    "PYTHONPATH",
}
EXPECTED_THRESHOLDS = {
    "warm_p50_cv_max": 0.05,
    "warm_p95_cv_max": 0.10,
    "throughput_cv_max": 0.05,
    "cold_model_load_p50_cv_max": 0.10,
    "peak_vram_relative_range_max": 0.01,
    "failure_count_max": 0,
}
EXPECTED_REPEATABILITY_REPORT_CONTRACT = "rustinfer.repeatability.v2"
PRIME_CELLS = EXPECTED_CELLS[:3]
LANE_PRIME_CELLS = {
    lane_id: PRIME_CELLS for lane_id in SUPPORTED_LANES
}
EXPECTED_THERMAL_STABILIZATION = {
    "temperature_limit_c": 50,
    "retry_interval_seconds": 30,
    "maximum_wait_seconds": 1200,
    "retry_only_on_temperature_limit": True,
    "final_full_preflight_required": True,
}
EXPECTED_CACHE_POLICY = {
    "cold_scope": "process-and-model-state-only",
    "dependency_preparation": "uv-sync-frozen-offline",
    "uv_version": "uv 0.12.5 (x86_64-unknown-linux-gnu)",
    "uv_linux_x86_64_sha256": "b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46",
    "uv_python": "3.13.15",
    "uv_python_downloads": "never",
    "python_linux_x86_64_sha256": "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866",
    "python_dont_write_bytecode": "1",
    "cuda_cache_maxsize": "4294967296",
    "python_hash_seed": "0",
    "tokenizers_parallelism": "false",
    "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "telemetry_environment": {
        "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_NO_USAGE_STATS": "1",
    },
    "reuse_external_disk_caches_across_independent_runs": True,
    "external_cache_environment": list(CACHE_PATH_ENVIRONMENT_KEYS),
    "offline_environment": dict(REQUIRED_OFFLINE_ENVIRONMENT),
    "lane_prime_cells": {
        lane_id: [dict(cell) for cell in cells]
        for lane_id, cells in LANE_PRIME_CELLS.items()
    },
    "reject_measured_cache_mutation": True,
}
PREFLIGHT_COMPARABILITY_KEYS = (
    "environment_id",
    "os_id",
    "os_version_id",
    "kernel_release",
    "machine",
    "cpu_model",
    "physical_cpu_cores",
    "logical_cpu_threads",
    "ram_bytes",
    "persistence_mode",
    "power_limit_w",
    "graphics_clock_mhz",
    "memory_clock_mhz",
    "cpu_governor",
    "cpu_governor_policy_count",
    "driver_version",
    "memory_total_mib",
)
PREFLIGHT_REQUIRED_KEYS = (
    *PREFLIGHT_COMPARABILITY_KEYS,
    "clock_synchronized",
    "staging_available_bytes",
    "staging_minimum_bytes",
)


class RunnerError(ValueError):
    """The requested run cannot safely or faithfully execute the contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RunnerError(f"cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError(f"{label} must be a UTF-8 JSON object: {error}") from error
    if not isinstance(value, dict):
        raise RunnerError(f"{label} must be a JSON object")
    return value


def _require_passing_repeatability_report(
    report: Mapping[str, Any], label: str
) -> None:
    if report.get("contract_version") != EXPECTED_REPEATABILITY_REPORT_CONTRACT:
        raise RunnerError(
            f"{label}.contract_version must be "
            f"{EXPECTED_REPEATABILITY_REPORT_CONTRACT!r}"
        )
    if report.get("passed") is not True or report.get("status") != "passed":
        raise RunnerError(f"{label} is not a pass")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RunnerError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RunnerError(f"required input is not a file: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        located = shutil.which(value)
        if located is None:
            raise RunnerError(f"cannot find executable on PATH: {value}")
        resolved = Path(located).resolve()
    if not resolved.is_file():
        raise RunnerError(f"executable is not a file: {resolved}")
    return resolved


def _executable_version(path: Path) -> str:
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise RunnerError(f"cannot execute {path} --version: {error}") from error
    output = completed.stdout.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0 or not output:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RunnerError(
            f"cannot determine executable version for {path}: {stderr or output}"
        )
    return output


def _python_runtime_version() -> str:
    return platform.python_version()


def _toolchain_evidence(uv_path: Path, *, canonical: bool) -> dict[str, Any]:
    uv = {**_artifact(uv_path), "version": _executable_version(uv_path)}
    python_path = Path(sys.executable).resolve()
    python = {
        **_artifact(python_path),
        "implementation": sys.implementation.name,
        "version": _python_runtime_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
    }
    if canonical:
        if uv["version"] != CANONICAL_UV_VERSION:
            raise RunnerError(
                f"canonical uv version must be {CANONICAL_UV_VERSION!r}, "
                f"found {uv['version']!r}"
            )
        if uv["sha256"] != CANONICAL_UV_LINUX_X86_64_SHA256:
            raise RunnerError("canonical uv Linux x86_64 binary SHA-256 mismatch")
        if python["version"] != CANONICAL_PYTHON_VERSION:
            raise RunnerError(
                f"canonical runner requires Python {CANONICAL_PYTHON_VERSION}, "
                f"found {python['version']}"
            )
        if python["sha256"] != CANONICAL_PYTHON_LINUX_X86_64_SHA256:
            raise RunnerError("canonical Python Linux x86_64 binary SHA-256 mismatch")
        if (
            python["implementation"] != "cpython"
            or python["platform"] != "linux"
            or python["machine"] != "x86_64"
        ):
            raise RunnerError(
                "canonical runner requires CPython on Linux x86_64"
            )
    return {"uv": uv, "python": python}


def _read_single_absolute_path(path: Path, label: str) -> Path:
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise RunnerError(f"cannot read {label} output {path}: {error}") from error
    if len(lines) != 1:
        raise RunnerError(f"{label} must print exactly one nonblank path")
    value = Path(lines[0]).expanduser()
    if not value.is_absolute():
        raise RunnerError(f"{label} returned a non-absolute path: {lines[0]!r}")
    resolved = value.resolve()
    if not resolved.is_file():
        raise RunnerError(f"{label} did not resolve to a file: {resolved}")
    return resolved


def _inspect_python_interpreter(
    path: Path,
    *,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    canonical: bool,
) -> dict[str, Any]:
    resolved = path.resolve()
    script = (
        "import json,platform,sys;"
        "print(json.dumps({"
        "'implementation':sys.implementation.name,"
        "'version':platform.python_version(),"
        "'platform':sys.platform,"
        "'machine':platform.machine(),"
        "'executable':sys.executable},sort_keys=True))"
    )
    returncode, launch_error = _run_captured(
        [str(resolved), "-c", script],
        environment=environment,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if returncode != 0:
        raise RunnerError(
            f"cannot inspect Python interpreter {resolved}: exit {returncode}"
            + (f" ({launch_error})" if launch_error else "")
        )
    evidence = _load_json(stdout_path, "Python interpreter evidence")
    evidence.update(_artifact(resolved))
    evidence["launcher_path"] = str(path)
    evidence["reported_executable"] = evidence.pop("executable", None)
    if canonical:
        expected = {
            "implementation": "cpython",
            "version": CANONICAL_PYTHON_VERSION,
            "platform": "linux",
            "machine": "x86_64",
        }
        for key, expected_value in expected.items():
            if evidence.get(key) != expected_value:
                raise RunnerError(
                    f"canonical Python interpreter {key} must be "
                    f"{expected_value!r}, found {evidence.get(key)!r}"
                )
        if evidence["sha256"] != CANONICAL_PYTHON_LINUX_X86_64_SHA256:
            raise RunnerError("canonical managed Python binary SHA-256 mismatch")
    return evidence


def _repo_path(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RunnerError(f"{label} must be a non-empty repository-relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise RunnerError(f"{label} must be repository-relative: {value}")
    resolved = (REPOSITORY_ROOT / raw).resolve()
    if resolved != REPOSITORY_ROOT and REPOSITORY_ROOT not in resolved.parents:
        raise RunnerError(f"{label} escapes the repository: {value}")
    return resolved


def _safe_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise RunnerError(
            f"output root must be outside the repository ({REPOSITORY_ROOT}): {resolved}"
        )
    if resolved.exists():
        raise RunnerError(f"output root already exists: {resolved}")
    return resolved


def _safe_finalize_destination(path: Path) -> Path:
    if RESULTS_ROOT.is_symlink():
        raise RunnerError(f"benchmark results root cannot be a symlink: {RESULTS_ROOT}")
    results_root = RESULTS_ROOT.resolve(strict=False)
    if not results_root.is_dir():
        raise RunnerError(f"benchmark results root is not a directory: {results_root}")
    raw = path.expanduser()
    if not raw.is_absolute():
        raw = REPOSITORY_ROOT / raw
    destination = raw.resolve(strict=False)
    if destination.parent != results_root:
        raise RunnerError(
            "--finalize-to must be one direct child of "
            f"{results_root}: {destination}"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", destination.name):
        raise RunnerError("--finalize-to result id contains unsafe characters")
    if destination.exists() or destination.is_symlink():
        raise RunnerError(f"finalize destination already exists: {destination}")
    return destination


def _validate_finalize_name(destination: Path, implementation_id: str) -> None:
    match = re.fullmatch(
        rf"(?P<timestamp>\d{{8}}T\d{{6}}Z)-{re.escape(implementation_id)}-"
        r"repeatability-(?P<run_id>[A-Za-z0-9][A-Za-z0-9._-]*)",
        destination.name,
    )
    if match is None:
        raise RunnerError(
            "--finalize-to result id must be "
            f"<YYYYMMDDTHHMMSSZ>-{implementation_id}-repeatability-<run-id>"
        )
    try:
        datetime.strptime(match.group("timestamp"), "%Y%m%dT%H%M%SZ")
    except ValueError as error:
        raise RunnerError("--finalize-to timestamp is not a real UTC date/time") from error


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RunnerError(f"{label} must be a JSON object")
    return value


def _canonical_cells(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    if matrix.get("cache_policy") != EXPECTED_CACHE_POLICY:
        raise RunnerError("matrix.cache_policy differs from the canonical Gate A policy")
    measurement = _mapping(matrix.get("measurement"), "matrix.measurement")
    if measurement.get("independent_runs") != EXPECTED_INDEPENDENT_RUNS:
        raise RunnerError(
            "matrix.measurement.independent_runs must be exactly "
            f"{EXPECTED_INDEPENDENT_RUNS}"
        )
    if measurement.get("run_index_origin") != 1:
        raise RunnerError("matrix.measurement.run_index_origin must be 1")
    if measurement.get("thermal_stabilization") != EXPECTED_THERMAL_STABILIZATION:
        raise RunnerError(
            "matrix.measurement.thermal_stabilization differs from the canonical "
            "bounded retry policy"
        )

    lifecycle = {
        "cold": (0, 1, False),
        "warm": (5, 30, True),
    }
    for state, (warmups, iterations, reuse) in lifecycle.items():
        policy = _mapping(
            measurement.get(state), f"matrix.measurement.{state}"
        )
        expected = {
            "warmup_iterations": warmups,
            "measured_iterations_per_run": iterations,
            "fresh_process_per_independent_run": True,
            "reset_model_state_per_independent_run": True,
            "reuse_model_within_run": reuse,
        }
        for key, expected_value in expected.items():
            if policy.get(key) != expected_value:
                raise RunnerError(
                    f"matrix.measurement.{state}.{key} must be "
                    f"{expected_value!r}"
                )

    gate = _mapping(matrix.get("repeatability_gate"), "matrix.repeatability_gate")
    if gate.get("thresholds") != EXPECTED_THRESHOLDS:
        raise RunnerError(
            "matrix.repeatability_gate.thresholds differs from the canonical Gate A "
            "thresholds"
        )
    raw_cells = gate.get("cells")
    if not isinstance(raw_cells, list):
        raise RunnerError("matrix.repeatability_gate.cells must be a JSON array")
    cells: list[dict[str, Any]] = []
    expected_keys = {"concurrency", "prompt_tokens", "output_tokens", "warm_state"}
    for index, value in enumerate(raw_cells, start=1):
        cell = _mapping(value, f"matrix.repeatability_gate.cells[{index - 1}]")
        if set(cell) != expected_keys:
            raise RunnerError(
                f"matrix.repeatability_gate.cells[{index - 1}] must contain exactly "
                f"{sorted(expected_keys)}"
            )
        cells.append(dict(cell))
    if cells != list(EXPECTED_CELLS):
        raise RunnerError(
            "matrix.repeatability_gate.cells must be the canonical ordered four-cell set"
        )
    return cells


def _lane_manifest(matrix: Mapping[str, Any], lane_id: str) -> tuple[Path, dict[str, Any]]:
    if lane_id not in SUPPORTED_LANES:
        raise RunnerError(
            f"lane must be one of {', '.join(SUPPORTED_LANES)}: {lane_id}"
        )
    raw_manifests = matrix.get("lane_manifests")
    if not isinstance(raw_manifests, list) or not all(
        isinstance(item, str) for item in raw_manifests
    ):
        raise RunnerError("matrix.lane_manifests must be an array of paths")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for item in raw_manifests:
        path = _repo_path(item, "matrix lane manifest")
        manifest = _load_json(path, "lane manifest")
        if manifest.get("lane_id") == lane_id:
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RunnerError(
            f"matrix must reference exactly one lane manifest for {lane_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _manifest_command(
    manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, str]]:
    if manifest.get("availability") != "available":
        raise RunnerError("selected lane availability must be 'available'")
    commands = _mapping(manifest.get("commands"), "lane.commands")
    benchmark = _mapping(commands.get("benchmark"), "lane.commands.benchmark")
    if benchmark.get("status") != "available":
        raise RunnerError("lane benchmark command status must be 'available'")
    raw_argv = benchmark.get("argv")
    if (
        not isinstance(raw_argv, list)
        or not raw_argv
        or not all(isinstance(item, str) and item for item in raw_argv)
    ):
        raise RunnerError("lane benchmark argv must be a non-empty string array")
    argv = list(raw_argv)
    expected_prefix = ["uv", "run", "--frozen", "--offline", "--no-sync", "--project"]
    if argv[: len(expected_prefix)] != expected_prefix:
        raise RunnerError(
            "lane benchmark argv must begin with canonical offline/no-sync uv run"
        )

    counts = {name: 0 for name in PLACEHOLDERS}
    for token in argv:
        matches = PLACEHOLDER_PATTERN.findall(token)
        if matches and token != "{" + matches[0] + "}":
            raise RunnerError(f"benchmark placeholder must occupy one argv token: {token}")
        if ("{" in token or "}" in token) and not matches:
            raise RunnerError(f"malformed benchmark placeholder: {token}")
        for name in matches:
            if name not in counts:
                raise RunnerError(f"unknown benchmark placeholder: {{{name}}}")
            counts[name] += 1
    invalid_counts = {name: count for name, count in counts.items() if count != 1}
    if invalid_counts:
        raise RunnerError(
            "benchmark argv must contain every canonical placeholder exactly once: "
            + json.dumps(invalid_counts, sort_keys=True)
        )

    raw_environment = benchmark.get("environment", {})
    if not isinstance(raw_environment, dict):
        raise RunnerError("lane benchmark environment must be a JSON object")
    environment: dict[str, str] = {}
    for key, value in raw_environment.items():
        if (
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\0" in key
            or not isinstance(value, str)
            or "\0" in value
        ):
            raise RunnerError("lane benchmark environment must contain safe strings")
        if "{" in value or "}" in value:
            raise RunnerError("lane benchmark environment cannot contain placeholders")
        if key in REPRODUCIBILITY_ENVIRONMENT_KEYS:
            raise RunnerError(
                f"lane benchmark environment cannot override canonical inherited {key}"
            )
        if key in SAFE_INHERITED_ENVIRONMENT_KEYS:
            raise RunnerError(
                f"lane benchmark environment cannot override sanitized runtime {key}"
            )
        if (
            key in RUNNER_CANONICAL_ENVIRONMENT
            and value != RUNNER_CANONICAL_ENVIRONMENT[key]
        ):
            raise RunnerError(
                f"lane benchmark environment conflicts with canonical runtime {key}"
            )
        environment[key] = value
    return argv, environment


def _replace_flag_value(argv: list[str], flag: str, value: str) -> None:
    positions = [index for index, item in enumerate(argv) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RunnerError(f"benchmark argv must contain exactly one {flag} value")
    argv[positions[0] + 1] = value


def _render_argv(
    template: Sequence[str],
    values: Mapping[str, str],
    *,
    matrix_path: Path,
    prompts_path: Path,
    uv: str,
) -> list[str]:
    if not uv:
        raise RunnerError("--uv must be a non-empty executable path or name")
    rendered = [
        values.get(token[1:-1], token) if token.startswith("{") else token
        for token in template
    ]
    if any("{" in token or "}" in token for token in rendered):
        raise RunnerError("benchmark argv contains an unresolved placeholder")
    rendered[0] = uv
    _replace_flag_value(rendered, "--matrix", str(matrix_path))
    _replace_flag_value(rendered, "--prompts", str(prompts_path))
    return rendered


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise RunnerError(f"cannot invoke git for revision: {error}") from error
    revision = completed.stdout.decode("ascii", errors="replace").strip()
    if completed.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", revision):
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RunnerError(f"cannot determine exact Git revision: {stderr or revision}")
    return revision


def _load_contract_validator() -> Any:
    path = REPOSITORY_ROOT / "benchmarks/scripts/validate_contract.py"
    spec = importlib.util.spec_from_file_location(
        "rustinfer_repeatability_runner_contract_validator", path
    )
    if spec is None or spec.loader is None:
        raise RunnerError(f"cannot load benchmark contract validator {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (OSError, ImportError) as error:
        raise RunnerError(f"cannot load benchmark contract validator: {error}") from error
    return module


def _validate_canonical_contract() -> dict[str, int]:
    validator = _load_contract_validator()
    try:
        return validator.validate_contract(REPOSITORY_ROOT)
    except (validator.ContractError, OSError) as error:
        raise RunnerError(f"canonical benchmark contract validation failed: {error}") from error


def _sensitive_runtime_environment_keys() -> list[str]:
    allowed = set(REPRODUCIBILITY_ENVIRONMENT_KEYS) | set(
        RUNNER_CANONICAL_ENVIRONMENT
    )
    return sorted(
        key
        for key in os.environ
        if key not in allowed
        and (
            key in SENSITIVE_RUNTIME_ENVIRONMENT_NAMES
            or key.startswith(SENSITIVE_RUNTIME_ENVIRONMENT_PREFIXES)
        )
    )


def _sanitized_child_environment(*, allow_test_environment: bool) -> dict[str, str]:
    forbidden = _sensitive_runtime_environment_keys()
    conflicting_pins = [
        key
        for key, expected in RUNNER_CANONICAL_ENVIRONMENT.items()
        if key in os.environ and os.environ[key] != expected
    ]
    rejected = sorted(set(forbidden) | set(conflicting_pins))
    if rejected:
        raise RunnerError(
            "canonical runner refuses inherited runtime/preflight overrides or "
            "conflicting owned pins: " + ", ".join(rejected)
        )
    environment = {
        key: os.environ[key]
        for key in SAFE_INHERITED_ENVIRONMENT_KEYS
        if key in os.environ and os.environ[key]
    }
    environment.update(
        {
            key: os.environ[key]
            for key in REPRODUCIBILITY_ENVIRONMENT_KEYS
            if key in os.environ
        }
    )
    environment.update(RUNNER_CANONICAL_ENVIRONMENT)
    if allow_test_environment:
        environment.update(
            {
                key: value
                for key, value in os.environ.items()
                if key.startswith("FAKE_")
            }
        )
    return dict(sorted(environment.items()))


def _write_new_bytes(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except OSError as error:
        raise RunnerError(f"cannot create artifact {path}: {error}") from error


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_new_bytes(path, encoded)


def _append_event(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with path.open("ab") as stream:
            stream.write(encoded)
    except OSError as error:
        raise RunnerError(f"cannot append execution event {path}: {error}") from error


def _run_captured(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, str | None]:
    child_environment = dict(environment)
    if any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\0" in key
        or not isinstance(value, str)
        or "\0" in value
        for key, value in child_environment.items()
    ):
        raise RunnerError("planned child environment contains an unsafe key or value")
    launch_error: str | None = None
    try:
        completed = subprocess.run(
            list(argv),
            cwd=REPOSITORY_ROOT,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except OSError as error:
        returncode = 127
        launch_error = str(error)
        stdout = b""
        stderr = (f"failed to launch {argv[0]!r}: {error}\n").encode(
            "utf-8", errors="replace"
        )
    _write_new_bytes(stdout_path, stdout)
    _write_new_bytes(stderr_path, stderr)
    return returncode, launch_error


def _cell_slug(cell: Mapping[str, Any]) -> str:
    return (
        f"{cell['warm_state']}-c{cell['concurrency']}-"
        f"p{cell['prompt_tokens']}-o{cell['output_tokens']}"
    )


def _environment_sha256(overrides: Mapping[str, str]) -> str:
    encoded = json.dumps(
        dict(overrides), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reproducibility_environment() -> tuple[dict[str, str], dict[str, str]]:
    values: dict[str, str] = {}
    resolved_paths: dict[str, str] = {}
    for key in REPRODUCIBILITY_ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if not isinstance(value, str) or not value:
            raise RunnerError(
                f"canonical gate requires non-empty inherited environment variable {key}"
            )
        values[key] = value
    for key, expected in REQUIRED_OFFLINE_ENVIRONMENT.items():
        if values[key] != expected:
            raise RunnerError(
                f"canonical gate requires {key}={expected}, found {values[key]!r}"
            )
    for key in CACHE_PATH_ENVIRONMENT_KEYS:
        raw = Path(values[key]).expanduser()
        if not raw.is_absolute():
            raise RunnerError(f"{key} must be an absolute external cache path")
        if raw.is_symlink():
            raise RunnerError(f"{key} cache root cannot be a symlink: {raw}")
        resolved = raw.resolve(strict=False)
        if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
            raise RunnerError(f"{key} cache root must be outside the repository: {raw}")
        if not resolved.is_dir():
            raise RunnerError(f"{key} cache root must already exist as a directory: {raw}")
        resolved_paths[key] = str(resolved)
    return dict(sorted(values.items())), dict(sorted(resolved_paths.items()))


def _derive_project_environment(
    *,
    uv_python_install_root: str,
    lane_id: str,
    dependency_lock_sha256: str,
    execution_nonce: str,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", lane_id):
        raise RunnerError(f"lane id is unsafe for project environment path: {lane_id!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", dependency_lock_sha256):
        raise RunnerError("dependency lock SHA-256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{12}", execution_nonce):
        raise RunnerError("execution nonce is invalid for project environment path")
    install_root = Path(uv_python_install_root).resolve()
    parent = install_root / "project-environments"
    destination = (
        parent / f"{lane_id}-{dependency_lock_sha256[:16]}-{execution_nonce}"
    )
    for candidate in (parent, destination):
        if candidate.is_symlink():
            raise RunnerError(
                f"derived UV project environment cannot traverse a symlink: {candidate}"
            )
    if destination.exists() or destination.is_symlink():
        raise RunnerError(
            f"fresh derived UV project environment already exists: {destination}"
        )
    resolved = destination.resolve(strict=False)
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise RunnerError("derived UV project environment must be repository-external")
    if resolved != install_root and install_root not in resolved.parents:
        raise RunnerError(
            "derived UV project environment escapes UV_PYTHON_INSTALL_DIR"
        )
    return resolved


def _cache_inventory(
    reproducibility_environment: Mapping[str, str],
) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    for key in CACHE_PATH_ENVIRONMENT_KEYS:
        root = Path(reproducibility_environment[key]).expanduser().resolve()
        entries: list[dict[str, Any]] = []

        def raise_walk_error(error: OSError) -> None:
            raise error

        try:
            for current, directory_names, file_names in os.walk(
                root, topdown=True, followlinks=False, onerror=raise_walk_error
            ):
                directory_names.sort()
                file_names.sort()
                current_path = Path(current)
                for name in directory_names:
                    path = current_path / name
                    relative = path.relative_to(root).as_posix()
                    stat_result = path.lstat()
                    if path.is_symlink():
                        resolved_target = path.resolve(strict=False)
                        if resolved_target != root and root not in resolved_target.parents:
                            raise RunnerError(
                                f"cache symlink escapes its declared root: {path}"
                            )
                        entries.append(
                            {
                                "path": relative,
                                "type": "symlink",
                                "target": os.readlink(path),
                                "bytes": stat_result.st_size,
                                "mtime_ns": stat_result.st_mtime_ns,
                            }
                        )
                    else:
                        entries.append(
                            {
                                "path": relative,
                                "type": "directory",
                                "bytes": stat_result.st_size,
                                "mtime_ns": stat_result.st_mtime_ns,
                            }
                        )
                for name in file_names:
                    path = current_path / name
                    relative = path.relative_to(root).as_posix()
                    stat_result = path.lstat()
                    if path.is_symlink():
                        kind = "symlink"
                        resolved_target = path.resolve(strict=False)
                        if resolved_target != root and root not in resolved_target.parents:
                            raise RunnerError(
                                f"cache symlink escapes its declared root: {path}"
                            )
                    elif path.is_file():
                        kind = "file"
                    else:
                        raise RunnerError(
                            f"cache inventory encountered non-regular entry: {path}"
                        )
                    entry: dict[str, Any] = {
                        "path": relative,
                        "type": kind,
                        "bytes": stat_result.st_size,
                        "mtime_ns": stat_result.st_mtime_ns,
                    }
                    if kind == "symlink":
                        entry["target"] = os.readlink(path)
                    entries.append(entry)
        except OSError as error:
            raise RunnerError(f"cannot inventory cache root {root}: {error}") from error
        encoded = json.dumps(
            entries,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        roots.append(
            {
                "environment_key": key,
                "path": str(root),
                "entry_count": len(entries),
                "total_regular_file_bytes": sum(
                    item["bytes"] for item in entries if item["type"] == "file"
                ),
                "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
                "entries": entries,
            }
        )
    aggregate_encoded = json.dumps(
        [
            {
                key: value
                for key, value in root.items()
                if key != "entries"
            }
            for root in roots
        ],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "contract_version": "rustinfer.cache-inventory.v1",
        "captured_at_utc": _utc_now(),
        "aggregate_sha256": hashlib.sha256(aggregate_encoded).hexdigest(),
        "roots": roots,
    }


def _cache_inventory_summary(inventory: Mapping[str, Any]) -> dict[str, Any]:
    roots = inventory.get("roots")
    if not isinstance(roots, list):
        raise RunnerError("cache inventory roots must be an array")
    return {
        "contract_version": inventory["contract_version"],
        "captured_at_utc": inventory["captured_at_utc"],
        "aggregate_sha256": inventory["aggregate_sha256"],
        "roots": [
            {key: value for key, value in root.items() if key != "entries"}
            for root in roots
        ],
    }


def _shared_result_validation_context(
    matrix_path: Path, prompts_path: Path
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Path], dict[str, dict[str, Any]]]:
    validator = _load_contract_validator()
    result_schema_path = REPOSITORY_ROOT / "benchmarks/schemas/result.schema.json"
    prompt_schema_path = REPOSITORY_ROOT / "benchmarks/schemas/prompt.schema.json"
    try:
        matrix = validator.validate_matrix(
            _load_json(matrix_path, "matrix"), REPOSITORY_ROOT
        )
        result_schema = validator.validate_schema_document(
            _load_json(result_schema_path, "result schema"),
            "result",
            str(result_schema_path),
        )
        prompt_schema = validator.validate_schema_document(
            _load_json(prompt_schema_path, "prompt schema"),
            "prompt",
            str(prompt_schema_path),
        )
        validator.validate_prompts(prompts_path, prompt_schema)
        lane_paths: dict[str, Path] = {}
        lanes_by_implementation: dict[str, dict[str, Any]] = {}
        for relative in matrix["lane_manifests"]:
            lane_path = REPOSITORY_ROOT / relative
            lane = validator.validate_lane_manifest(
                _load_json(lane_path, "lane manifest"), matrix, str(lane_path)
            )
            validator.validate_dependency_project(REPOSITORY_ROOT, lane)
            lane_paths[lane["lane_id"]] = lane_path
            lanes_by_implementation[lane["implementation_id"]] = lane
    except (validator.ContractError, OSError) as error:
        raise RunnerError(
            f"cannot construct shared result validation context: {error}"
        ) from error
    return (
        validator,
        result_schema,
        matrix,
        lane_paths,
        lanes_by_implementation,
    )


def _validate_prime_raw(
    path: Path,
    cell: Mapping[str, Any],
    run_id: str,
    validation_context: tuple[
        Any,
        dict[str, Any],
        dict[str, Any],
        dict[str, Path],
        dict[str, dict[str, Any]],
    ]
    | None,
    *,
    matrix_path: Path,
    prompts_path: Path,
) -> int:
    if validation_context is not None:
        validator, schema, matrix, lane_paths, lanes_by_implementation = (
            validation_context
        )
        try:
            shared_count = validator.validate_result_file(
                path,
                schema,
                matrix,
                matrix_path,
                prompts_path,
                lane_paths,
                lanes_by_implementation,
            )
        except (validator.ContractError, OSError) as error:
            raise RunnerError(
                f"cache-prime raw failed shared result validation: {error}"
            ) from error
        if shared_count != 30:
            raise RunnerError(
                "shared result validator did not observe exactly 30 cache-prime rows"
            )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RunnerError(f"cannot read cache-prime raw {path}: {error}") from error
    if len(lines) != 30 or any(not line.strip() for line in lines):
        raise RunnerError("warm cache-prime raw must contain exactly 30 nonblank rows")
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunnerError(
                f"cache-prime raw {path}:{line_number} is invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise RunnerError(f"cache-prime raw {path}:{line_number} is not an object")
        if row.get("status") != "success" or row.get("failure_count") != 0:
            raise RunnerError(
                f"cache-prime raw {path}:{line_number} did not complete successfully"
            )
        if row.get("run_id") != run_id or row.get("trial_index") != line_number:
            raise RunnerError(
                f"cache-prime raw {path}:{line_number} has unexpected run/trial identity"
            )
        workload = row.get("workload")
        if not isinstance(workload, dict) or any(
            workload.get(key) != cell[key]
            for key in ("concurrency", "prompt_tokens", "output_tokens", "warm_state")
        ):
            raise RunnerError(
                f"cache-prime raw {path}:{line_number} workload differs from its plan"
            )
    return len(lines)


def _build_plan(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    execution_nonce = uuid.uuid4().hex[:12]
    matrix_path = args.matrix.expanduser().resolve()
    prompts_path = args.prompts.expanduser().resolve()
    preflight_path = args.preflight.expanduser().resolve()
    checker_path = args.checker.expanduser().resolve()
    uv_path = _resolve_executable(args.uv)
    matrix = _load_json(matrix_path, "matrix")
    cells = _canonical_cells(matrix)
    lane_path, lane = _lane_manifest(matrix, args.lane)
    template, environment = _manifest_command(lane)
    dependency_manifest = _repo_path(
        lane.get("dependency_manifest"), "lane.dependency_manifest"
    )
    dependency_lock = dependency_manifest.with_name("uv.lock")
    dependency_python_version = dependency_manifest.with_name(".python-version")
    dependency_lock_sha256 = _sha256(dependency_lock)
    dependency_python_version_sha256 = _sha256(dependency_python_version)
    reproducibility_environment, resolved_cache_paths = (
        _reproducibility_environment()
    )
    child_base_environment = _sanitized_child_environment(
        allow_test_environment=args.allow_noncanonical_tools
    )
    project_environment = _derive_project_environment(
        uv_python_install_root=resolved_cache_paths["UV_PYTHON_INSTALL_DIR"],
        lane_id=args.lane,
        dependency_lock_sha256=dependency_lock_sha256,
        execution_nonce=execution_nonce,
    )
    child_base_environment[PROJECT_ENVIRONMENT_VARIABLE] = str(project_environment)
    child_base_environment = dict(sorted(child_base_environment.items()))
    execution_environment = {**child_base_environment, **environment}

    for path, label in (
        (prompts_path, "prompts"),
        (preflight_path, "preflight"),
        (checker_path, "repeatability checker"),
    ):
        if not path.is_file():
            raise RunnerError(f"{label} is not a file: {path}")
    canonical_tools = (
        preflight_path == CANONICAL_PREFLIGHT.resolve()
        and checker_path == CANONICAL_CHECKER.resolve()
    )
    canonical_inputs = (
        matrix_path == CANONICAL_MATRIX.resolve()
        and prompts_path == CANONICAL_PROMPTS.resolve()
    )
    if not canonical_inputs and not args.allow_noncanonical_tools:
        raise RunnerError(
            "noncanonical --matrix/--prompts paths require "
            "--allow-noncanonical-tools (offline tests only)"
        )
    if not canonical_tools and not args.allow_noncanonical_tools:
        raise RunnerError(
            "noncanonical --preflight/--checker paths require "
            "--allow-noncanonical-tools (offline tests only)"
        )
    canonical_execution = (
        canonical_inputs and canonical_tools and not args.allow_noncanonical_tools
    )
    if (
        canonical_execution
        and dependency_python_version_sha256
        != CANONICAL_PYTHON_VERSION_FILE_SHA256
    ):
        raise RunnerError(
            "selected dependency .python-version SHA-256 differs from the canonical "
            "3.13.15 contract"
        )
    toolchain_evidence = _toolchain_evidence(
        uv_path, canonical=canonical_execution
    )
    contract_validation_counts: dict[str, int] | None = None
    if canonical_execution:
        contract_validation_counts = _validate_canonical_contract()

    finalize_destination = (
        _safe_finalize_destination(args.finalize_to)
        if args.finalize_to is not None
        else None
    )
    if finalize_destination is not None:
        _validate_finalize_name(finalize_destination, str(lane["implementation_id"]))
    if finalize_destination is not None and not canonical_execution:
        raise RunnerError(
            "--finalize-to requires canonical matrix, prompts, preflight, checker, "
            "and full validate_contract validation"
        )
    created_at = _utc_now()
    preparation_root = output_root / "preparation"
    sync_argv = [
        str(uv_path),
        "sync",
        "--frozen",
        "--offline",
        "--project",
        str(dependency_manifest.parent),
    ]
    prime_invocations: list[dict[str, Any]] = []
    for prime_index, cell in enumerate(LANE_PRIME_CELLS[args.lane], start=1):
        artifact_dir = (
            preparation_root / f"prime-{prime_index:02d}-{_cell_slug(cell)}"
        )
        result_dir = artifact_dir / "result"
        prime_run_id = (
            f"{args.lane}-cache-prime-{execution_nonce}-{prime_index:02d}"
        )
        values = {
            "result_dir": str(result_dir),
            "independent_run_index": "1",
            "run_id": prime_run_id,
            "warm_state": str(cell["warm_state"]),
            "concurrency": str(cell["concurrency"]),
            "prompt_tokens": str(cell["prompt_tokens"]),
            "output_tokens": str(cell["output_tokens"]),
        }
        prime_invocations.append(
            {
                "prime_index": prime_index,
                "cell": dict(cell),
                "run_id": prime_run_id,
                "artifact_dir": str(artifact_dir),
                "result_dir": str(result_dir),
                "raw_result": str(result_dir / "raw.jsonl"),
                "argv": _render_argv(
                    template,
                    values,
                    matrix_path=matrix_path,
                    prompts_path=prompts_path,
                    uv=str(uv_path),
                ),
                "environment": dict(sorted(execution_environment.items())),
                "stdout": str(artifact_dir / "benchmark.stdout.txt"),
                "stderr": str(artifact_dir / "benchmark.stderr.txt"),
            }
        )
    invocations: list[dict[str, Any]] = []
    raw_results: list[str] = []
    run_ids: list[str] = []
    ordinal = 0
    for run_index in range(1, EXPECTED_INDEPENDENT_RUNS + 1):
        run_id = (
            f"{args.lane}-{created_at.replace(':', '').replace('-', '')}-"
            f"{execution_nonce}-run-{run_index:02d}"
        )
        run_ids.append(run_id)
        for cell_index, cell in enumerate(cells, start=1):
            ordinal += 1
            invocation_dir = (
                output_root
                / "runs"
                / f"run-{run_index:02d}"
                / f"cell-{cell_index:02d}-{_cell_slug(cell)}"
            )
            result_dir = invocation_dir / "result"
            values = {
                "result_dir": str(result_dir),
                "independent_run_index": str(run_index),
                "run_id": run_id,
                "warm_state": str(cell["warm_state"]),
                "concurrency": str(cell["concurrency"]),
                "prompt_tokens": str(cell["prompt_tokens"]),
                "output_tokens": str(cell["output_tokens"]),
            }
            argv = _render_argv(
                template,
                values,
                matrix_path=matrix_path,
                prompts_path=prompts_path,
                uv=str(uv_path),
            )
            raw_path = result_dir / "raw.jsonl"
            raw_results.append(str(raw_path))
            preflight_environment = dict(child_base_environment)
            preflight_environment["RUSTINFER_PREFLIGHT_OUTPUT_ROOT"] = str(output_root)
            invocations.append(
                {
                    "ordinal": ordinal,
                    "independent_run_index": run_index,
                    "run_id": run_id,
                    "cell_index": cell_index,
                    "cell": cell,
                    "cwd": str(REPOSITORY_ROOT),
                    "artifact_dir": str(invocation_dir),
                    "result_dir": str(result_dir),
                    "raw_result": str(raw_path),
                    "preflight_argv": [str(preflight_path)],
                    "preflight_environment": dict(
                        sorted(preflight_environment.items())
                    ),
                    "benchmark_argv": argv,
                    "environment": dict(sorted(execution_environment.items())),
                    "reproducibility_environment": reproducibility_environment,
                    "effective_environment_sha256": _environment_sha256(
                        execution_environment
                    ),
                    "environment_inheritance": (
                        "exact sanitized environment; ambient values are not inherited"
                    ),
                    "artifacts": {
                        "preflight_attempts": str(
                            invocation_dir / "preflight-attempts"
                        ),
                        "preflight_stdout": str(
                            invocation_dir / "preflight.stdout.txt"
                        ),
                        "preflight_stderr": str(
                            invocation_dir / "preflight.stderr.txt"
                        ),
                        "preflight_snapshot": str(
                            invocation_dir / "preflight.snapshot.json"
                        ),
                        "benchmark_stdout": str(
                            invocation_dir / "benchmark.stdout.txt"
                        ),
                        "benchmark_stderr": str(
                            invocation_dir / "benchmark.stderr.txt"
                        ),
                        "cache_snapshot": str(
                            invocation_dir / "cache.snapshot.json"
                        ),
                        "cache_drift_inventory": str(
                            invocation_dir / "cache.drift.inventory.json"
                        ),
                    },
                }
            )

    report_path = output_root / "repeatability-report.json"
    checker_argv = [
        sys.executable,
        str(checker_path),
        "--matrix",
        str(matrix_path),
        "--output",
        str(report_path),
        *raw_results,
    ]
    return {
        "contract_version": "rustinfer.repeatability-runner.v1",
        "created_at_utc": created_at,
        "repository_root": str(REPOSITORY_ROOT),
        "output_root": str(output_root),
        "runner_argv": list(args.runner_argv),
        "canonical_gate_tools": canonical_tools,
        "canonical_gate_inputs": canonical_inputs,
        "canonical_execution": canonical_execution,
        "contract_validation": {
            "validator": str(
                REPOSITORY_ROOT / "benchmarks/scripts/validate_contract.py"
            ),
            "status": "passed" if canonical_execution else "noncanonical-test-bypass",
            "counts": contract_validation_counts,
        },
        "git_revision": _git_revision(),
        "lane_id": args.lane,
        "implementation_id": lane["implementation_id"],
        "independent_runs": EXPECTED_INDEPENDENT_RUNS,
        "cells_per_run": len(cells),
        "invocation_count": len(invocations),
        "run_ids": run_ids,
        "inputs": {
            "matrix": _artifact(matrix_path),
            "prompts": _artifact(prompts_path),
            "lane_manifest": _artifact(lane_path),
            "dependency_manifest": _artifact(dependency_manifest),
            "dependency_lock": _artifact(dependency_lock),
            "dependency_python_version": _artifact(dependency_python_version),
            "preflight": _artifact(preflight_path),
            "repeatability_checker": _artifact(checker_path),
            "runner": _artifact(Path(__file__).resolve()),
            "python": toolchain_evidence["python"],
            "uv": toolchain_evidence["uv"],
        },
        "reproducibility_environment": {
            "allowlisted_values": reproducibility_environment,
            "resolved_cache_paths": resolved_cache_paths,
            "required_offline_values": REQUIRED_OFFLINE_ENVIRONMENT,
            "inheritance": (
                "runtime subprocesses receive one exact sanitized environment; "
                "unrelated or secret ambient values are not inherited"
            ),
        },
        "runtime_environment_policy": {
            "exact_child_base": child_base_environment,
            "safe_inherited_keys": list(SAFE_INHERITED_ENVIRONMENT_KEYS),
            "reproducibility_keys": list(REPRODUCIBILITY_ENVIRONMENT_KEYS),
            "runner_pins": RUNNER_CANONICAL_ENVIRONMENT,
            "sensitive_prefixes_rejected": list(
                SENSITIVE_RUNTIME_ENVIRONMENT_PREFIXES
            ),
            "sensitive_names_rejected": sorted(
                SENSITIVE_RUNTIME_ENVIRONMENT_NAMES
            ),
            "ambient_inheritance": False,
            "project_environment": {
                "variable": PROJECT_ENVIRONMENT_VARIABLE,
                "path": str(project_environment),
                "derivation": "UV_PYTHON_INSTALL_DIR/project-environments/<lane>-<lock-prefix16>-<nonce>",
                "lane_id": args.lane,
                "dependency_lock_sha256": dependency_lock_sha256,
                "repository_external": True,
                "covered_by_cache_inventory": "UV_PYTHON_INSTALL_DIR",
            },
        },
        "preparation": {
            "root": str(preparation_root),
            "policy": (
                "the pinned managed Python is resolved and verified; uv lock "
                "synchronization is offline into one fresh external project "
                "environment; the selected lane primes every distinct "
                "repeatability compile profile in fresh unmeasured subprocesses"
            ),
            "python_find": {
                "argv": [
                    str(uv_path),
                    "python",
                    "find",
                    CANONICAL_PYTHON_VERSION,
                ],
                "environment": child_base_environment,
                "stdout": str(preparation_root / "uv-python-find.stdout.txt"),
                "stderr": str(preparation_root / "uv-python-find.stderr.txt"),
                "runtime_stdout": str(
                    preparation_root / "managed-python-runtime.stdout.json"
                ),
                "runtime_stderr": str(
                    preparation_root / "managed-python-runtime.stderr.txt"
                ),
            },
            "sync": {
                "argv": sync_argv,
                "cwd": str(REPOSITORY_ROOT),
                "environment": child_base_environment,
                "stdout": str(preparation_root / "uv-sync.stdout.txt"),
                "stderr": str(preparation_root / "uv-sync.stderr.txt"),
            },
            "project_python": {
                "path": str(project_environment / "bin/python"),
                "runtime_stdout": str(
                    preparation_root / "project-python-runtime.stdout.json"
                ),
                "runtime_stderr": str(
                    preparation_root / "project-python-runtime.stderr.txt"
                ),
            },
            "prime_invocations": prime_invocations,
            "cache_inventory_before": str(
                preparation_root / "cache.inventory.before.json"
            ),
            "cache_inventory_after": str(
                preparation_root / "cache.inventory.after.json"
            ),
            "summary": str(preparation_root / "summary.json"),
            "python_evidence": str(
                preparation_root / "python-evidence.json"
            ),
            "immutable_evidence": {
                "model_id": matrix["model"]["id"],
                "model_revision": matrix["model"]["revision"],
                "weights_sha256": matrix["model"]["weights_sha256"],
                "dependency_lock_sha256": _sha256(dependency_lock),
                "project_environment": str(project_environment),
                "lane_manifest_sha256": _sha256(lane_path),
                "uv_sha256": _sha256(uv_path),
            },
        },
        "preflight": {
            "path": str(preflight_path),
            "required_comparability_keys": list(PREFLIGHT_COMPARABILITY_KEYS),
            "required_keys": list(PREFLIGHT_REQUIRED_KEYS),
            "required_driver_version": PRIMARY_DRIVER_VERSION,
            "required_persistence_mode": PRIMARY_PERSISTENCE_MODE,
            "required_cpu_governor": PRIMARY_CPU_GOVERNOR,
            "required_host": {
                "environment_id": PRIMARY_ENVIRONMENT_ID,
                "os_id": PRIMARY_OS_ID,
                "os_version_id": PRIMARY_OS_VERSION_ID,
                "kernel_release": PRIMARY_KERNEL_RELEASE,
                "machine": PRIMARY_MACHINE,
                "cpu_model": PRIMARY_CPU_MODEL,
                "physical_cpu_cores": PRIMARY_CPU_PHYSICAL_CORES,
                "logical_cpu_threads": PRIMARY_CPU_LOGICAL_THREADS,
                "cpu_governor_policy_count": PRIMARY_CPU_GOVERNOR_POLICY_COUNT,
                "ram_bytes": PRIMARY_RAM_BYTES,
            },
            "required_gpu_memory_mib": PRIMARY_GPU_MEMORY_MIB,
            "minimum_staging_available_bytes": MINIMUM_STAGING_AVAILABLE_BYTES,
            "thermal_stabilization": EXPECTED_THERMAL_STABILIZATION,
            "baseline": str(output_root / "preflight-baseline.json"),
        },
        "finalize_to": (
            str(finalize_destination) if finalize_destination is not None else None
        ),
        "finalization": (
            {
                "destination": str(finalize_destination),
                "combined_raw": str(output_root / "raw.jsonl"),
                "metadata": str(output_root / "metadata.json"),
                "readme": str(output_root / "README.md"),
                "validation_report": str(
                    output_root / "finalize-repeatability-report.json"
                ),
                "validation_stdout": str(output_root / "finalize-checker.stdout.txt"),
                "validation_stderr": str(output_root / "finalize-checker.stderr.txt"),
                "validation_argv": [
                    sys.executable,
                    str(checker_path),
                    "--matrix",
                    str(matrix_path),
                    "--output",
                    str(output_root / "finalize-repeatability-report.json"),
                    str(output_root / "raw.jsonl"),
                ],
            }
            if finalize_destination is not None
            else None
        ),
        "invocations": invocations,
        "checker": {
            "argv": checker_argv,
            "cwd": str(REPOSITORY_ROOT),
            "environment": child_base_environment,
            "report": str(report_path),
            "stdout": str(output_root / "checker.stdout.txt"),
            "stderr": str(output_root / "checker.stderr.txt"),
        },
    }


def _record_failure(
    output_root: Path,
    *,
    stage: str,
    message: str,
    invocation: Mapping[str, Any] | None = None,
    returncode: int | None = None,
) -> None:
    failure: dict[str, Any] = {
        "contract_version": "rustinfer.repeatability-runner.v1",
        "recorded_at_utc": _utc_now(),
        "status": "failed",
        "stage": stage,
        "message": message,
        "returncode": returncode,
    }
    if invocation is not None:
        failure["ordinal"] = invocation["ordinal"]
        failure["independent_run_index"] = invocation["independent_run_index"]
        failure["run_id"] = invocation["run_id"]
        failure["cell"] = invocation["cell"]
    path = output_root / "failure.json"
    if not path.exists():
        _write_new_json(path, failure)


def _parse_preflight(path: Path, git_revision: str) -> tuple[dict[str, str], dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RunnerError(f"cannot parse preflight stdout {path}: {error}") from error
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line:
            continue
        if "=" not in raw_line:
            raise RunnerError(
                f"preflight stdout line {line_number} is not key=value: {raw_line!r}"
            )
        key, value = raw_line.split("=", 1)
        if not key or not value:
            raise RunnerError(
                f"preflight stdout line {line_number} has an empty key or value"
            )
        if key in values:
            raise RunnerError(f"preflight stdout contains duplicate key {key!r}")
        values[key] = value
    missing = [key for key in PREFLIGHT_REQUIRED_KEYS if key not in values]
    if missing:
        raise RunnerError(
            "preflight stdout is missing required keys: " + ", ".join(missing)
        )
    if values["driver_version"] != PRIMARY_DRIVER_VERSION:
        raise RunnerError(
            "preflight driver_version must be "
            f"{PRIMARY_DRIVER_VERSION}, found {values['driver_version']}"
        )
    if values["persistence_mode"] != PRIMARY_PERSISTENCE_MODE:
        raise RunnerError(
            "preflight persistence_mode must be "
            f"{PRIMARY_PERSISTENCE_MODE}, found {values['persistence_mode']}"
        )
    if values["cpu_governor"] != PRIMARY_CPU_GOVERNOR:
        raise RunnerError(
            "preflight cpu_governor must be "
            f"{PRIMARY_CPU_GOVERNOR}, found {values['cpu_governor']}"
        )
    try:
        governor_policy_count = int(values["cpu_governor_policy_count"])
    except ValueError as error:
        raise RunnerError(
            "preflight cpu_governor_policy_count must be a base-10 integer"
        ) from error
    if governor_policy_count != PRIMARY_CPU_GOVERNOR_POLICY_COUNT:
        raise RunnerError(
            "preflight cpu_governor_policy_count must be "
            f"{PRIMARY_CPU_GOVERNOR_POLICY_COUNT}, found {governor_policy_count}"
        )
    exact_host_values = {
        "environment_id": PRIMARY_ENVIRONMENT_ID,
        "os_id": PRIMARY_OS_ID,
        "os_version_id": PRIMARY_OS_VERSION_ID,
        "kernel_release": PRIMARY_KERNEL_RELEASE,
        "machine": PRIMARY_MACHINE,
        "cpu_model": PRIMARY_CPU_MODEL,
    }
    for key, expected in exact_host_values.items():
        if values[key] != expected:
            raise RunnerError(
                f"preflight {key} must be {expected!r}, found {values[key]!r}"
            )
    integer_host_values = {
        "physical_cpu_cores": PRIMARY_CPU_PHYSICAL_CORES,
        "logical_cpu_threads": PRIMARY_CPU_LOGICAL_THREADS,
        "ram_bytes": PRIMARY_RAM_BYTES,
    }
    for key, expected in integer_host_values.items():
        try:
            observed = int(values[key])
        except ValueError as error:
            raise RunnerError(f"preflight {key} must be a base-10 integer") from error
        if observed != expected:
            raise RunnerError(
                f"preflight {key} must be {expected}, found {observed}"
            )
    try:
        memory_total_mib = int(values["memory_total_mib"])
    except ValueError as error:
        raise RunnerError("preflight memory_total_mib must be a base-10 integer") from error
    if memory_total_mib != PRIMARY_GPU_MEMORY_MIB:
        raise RunnerError(
            f"preflight memory_total_mib must be {PRIMARY_GPU_MEMORY_MIB}, "
            f"found {memory_total_mib}"
        )
    if values["clock_synchronized"] != "yes":
        raise RunnerError(
            "preflight clock_synchronized must be yes, found "
            f"{values['clock_synchronized']}"
        )
    try:
        staging_available = int(values["staging_available_bytes"])
        staging_minimum = int(values["staging_minimum_bytes"])
    except ValueError as error:
        raise RunnerError(
            "preflight staging available/minimum bytes must be base-10 integers"
        ) from error
    if staging_minimum != MINIMUM_STAGING_AVAILABLE_BYTES:
        raise RunnerError(
            "preflight staging_minimum_bytes must be "
            f"{MINIMUM_STAGING_AVAILABLE_BYTES}, found {staging_minimum}"
        )
    if staging_available < staging_minimum:
        raise RunnerError(
            f"preflight staging_available_bytes {staging_available} is below "
            f"{staging_minimum}"
        )
    if "git_revision" in values and values["git_revision"] != git_revision:
        raise RunnerError(
            "preflight git_revision does not match the execution plan: "
            f"{values['git_revision']} != {git_revision}"
        )
    snapshot = {key: values[key] for key in PREFLIGHT_COMPARABILITY_KEYS}
    return values, snapshot


def _preflight_drift(
    baseline: Mapping[str, str], snapshot: Mapping[str, str]
) -> list[str]:
    return [
        key
        for key in PREFLIGHT_COMPARABILITY_KEYS
        if baseline.get(key) != snapshot.get(key)
    ]


_TEMPERATURE_LIMIT_FAILURE = re.compile(
    r"preflight: start temperature (?P<temperature>[0-9]+) C exceeds 50 C"
)


def _retryable_temperature_failure(
    returncode: int, stdout_path: Path, stderr_path: Path
) -> tuple[bool, int | None]:
    if returncode != 2:
        return False, None
    try:
        stdout = stdout_path.read_text(encoding="utf-8")
        stderr_lines = [
            line.strip()
            for line in stderr_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise RunnerError(f"cannot inspect failed preflight artifacts: {error}") from error
    if stdout.strip() or len(stderr_lines) != 1:
        return False, None
    match = _TEMPERATURE_LIMIT_FAILURE.fullmatch(stderr_lines[0])
    if match is None:
        return False, None
    return True, int(match.group("temperature"))


def _execute_preparation(plan: Mapping[str, Any]) -> dict[str, Any]:
    output_root = Path(str(plan["output_root"]))
    events_path = output_root / "execution-events.jsonl"
    preparation = _mapping(plan["preparation"], "planned preparation")
    preparation_root = Path(str(preparation["root"]))
    preparation_root.mkdir(parents=True, exist_ok=False)
    reproducibility = _mapping(
        _mapping(
            plan["reproducibility_environment"],
            "planned reproducibility environment",
        )["allowlisted_values"],
        "planned reproducibility allowlist",
    )
    inputs = _mapping(plan["inputs"], "planned inputs")
    matrix_path = Path(
        str(_mapping(inputs["matrix"], "planned matrix input")["path"])
    )
    prompts_path = Path(
        str(_mapping(inputs["prompts"], "planned prompts input")["path"])
    )
    validation_context = (
        _shared_result_validation_context(matrix_path, prompts_path)
        if plan.get("canonical_execution") is True
        else None
    )

    project_policy = _mapping(
        _mapping(
            plan["runtime_environment_policy"], "runtime environment policy"
        )["project_environment"],
        "planned project environment",
    )
    project_environment = Path(str(project_policy["path"]))
    if project_environment.exists() or project_environment.is_symlink():
        raise RunnerError(
            "fresh derived UV project environment appeared before preparation: "
            f"{project_environment}"
        )

    inventory_before = _cache_inventory(reproducibility)
    _write_new_json(
        Path(str(preparation["cache_inventory_before"])), inventory_before
    )

    python_find = _mapping(preparation["python_find"], "planned Python lookup")
    python_find_started = _utc_now()
    returncode, launch_error = _run_captured(
        python_find["argv"],
        environment=_mapping(
            python_find["environment"], "planned Python lookup environment"
        ),
        stdout_path=Path(str(python_find["stdout"])),
        stderr_path=Path(str(python_find["stderr"])),
    )
    python_find_finished = _utc_now()
    _append_event(
        events_path,
        {
            "stage": "preparation-python-find",
            "started_at_utc": python_find_started,
            "finished_at_utc": python_find_finished,
            "argv": list(python_find["argv"]),
            "returncode": returncode,
            "launch_error": launch_error,
        },
    )
    if returncode != 0:
        raise RunnerError(
            f"offline uv python find failed with exit code {returncode}"
        )
    managed_python = _read_single_absolute_path(
        Path(str(python_find["stdout"])), "uv python find"
    )
    install_root = Path(str(reproducibility["UV_PYTHON_INSTALL_DIR"])).resolve()
    if managed_python != install_root and install_root not in managed_python.parents:
        raise RunnerError(
            "uv python find resolved outside UV_PYTHON_INSTALL_DIR: "
            f"{managed_python}"
        )
    managed_started = _utc_now()
    managed_evidence = _inspect_python_interpreter(
        managed_python,
        environment=_mapping(
            python_find["environment"], "planned Python lookup environment"
        ),
        stdout_path=Path(str(python_find["runtime_stdout"])),
        stderr_path=Path(str(python_find["runtime_stderr"])),
        canonical=plan.get("canonical_execution") is True,
    )
    managed_finished = _utc_now()
    _append_event(
        events_path,
        {
            "stage": "preparation-managed-python-inspect",
            "started_at_utc": managed_started,
            "finished_at_utc": managed_finished,
            "argv": [str(managed_python), "-c", "<runtime-evidence>"],
            "returncode": 0,
            "launch_error": None,
        },
    )

    sync = _mapping(preparation["sync"], "planned dependency sync")
    sync_started = _utc_now()
    returncode, launch_error = _run_captured(
        sync["argv"],
        environment=_mapping(sync["environment"], "planned sync environment"),
        stdout_path=Path(str(sync["stdout"])),
        stderr_path=Path(str(sync["stderr"])),
    )
    sync_finished = _utc_now()
    _append_event(
        events_path,
        {
            "stage": "preparation-sync",
            "started_at_utc": sync_started,
            "finished_at_utc": sync_finished,
            "argv": list(sync["argv"]),
            "returncode": returncode,
            "launch_error": launch_error,
        },
    )
    if returncode != 0:
        raise RunnerError(f"offline uv sync failed with exit code {returncode}")

    if not project_environment.is_dir() or project_environment.is_symlink():
        raise RunnerError(
            "offline uv sync did not create the planned regular project environment: "
            f"{project_environment}"
        )
    project_python_plan = _mapping(
        preparation["project_python"], "planned project Python"
    )
    project_python = Path(str(project_python_plan["path"]))
    if not project_python.exists():
        raise RunnerError(
            f"offline uv sync did not create project interpreter {project_python}"
        )
    resolved_project_python = project_python.resolve()
    if (
        resolved_project_python != install_root
        and install_root not in resolved_project_python.parents
    ):
        raise RunnerError(
            "project interpreter resolves outside UV_PYTHON_INSTALL_DIR: "
            f"{resolved_project_python}"
        )
    project_python_started = _utc_now()
    project_python_evidence = _inspect_python_interpreter(
        project_python,
        environment=_mapping(sync["environment"], "planned sync environment"),
        stdout_path=Path(str(project_python_plan["runtime_stdout"])),
        stderr_path=Path(str(project_python_plan["runtime_stderr"])),
        canonical=plan.get("canonical_execution") is True,
    )
    project_python_finished = _utc_now()
    _append_event(
        events_path,
        {
            "stage": "preparation-project-python-inspect",
            "started_at_utc": project_python_started,
            "finished_at_utc": project_python_finished,
            "argv": [str(project_python), "-c", "<runtime-evidence>"],
            "returncode": 0,
            "launch_error": None,
        },
    )
    if project_python_evidence["sha256"] != managed_evidence["sha256"]:
        raise RunnerError(
            "project interpreter binary differs from the uv-resolved managed Python"
        )
    python_evidence = {
        "contract_version": "rustinfer.python-runtime-evidence.v1",
        "uv_python_find": {
            "argv": python_find["argv"],
            "started_at_utc": python_find_started,
            "finished_at_utc": python_find_finished,
            "stdout": python_find["stdout"],
            "stderr": python_find["stderr"],
        },
        "managed_python": managed_evidence,
        "project_python": project_python_evidence,
        "same_binary_sha256": True,
    }
    _write_new_json(Path(str(preparation["python_evidence"])), python_evidence)

    prime_evidence: list[dict[str, Any]] = []
    for prime_value in preparation["prime_invocations"]:
        prime = _mapping(prime_value, "planned cache-prime invocation")
        artifact_dir = Path(str(prime["artifact_dir"]))
        artifact_dir.mkdir(parents=True, exist_ok=False)
        started = _utc_now()
        returncode, launch_error = _run_captured(
            prime["argv"],
            environment=_mapping(prime["environment"], "planned prime environment"),
            stdout_path=Path(str(prime["stdout"])),
            stderr_path=Path(str(prime["stderr"])),
        )
        finished = _utc_now()
        _append_event(
            events_path,
            {
                "stage": "preparation-prime",
                "prime_index": prime["prime_index"],
                "cell": prime["cell"],
                "started_at_utc": started,
                "finished_at_utc": finished,
                "argv": list(prime["argv"]),
                "returncode": returncode,
                "launch_error": launch_error,
            },
        )
        raw_path = Path(str(prime["raw_result"]))
        if returncode != 0:
            raise RunnerError(
                f"cache-prime invocation {prime['prime_index']} failed with "
                f"exit code {returncode}"
            )
        if not raw_path.is_file():
            raise RunnerError(
                f"cache-prime invocation {prime['prime_index']} did not create {raw_path}"
            )
        row_count = _validate_prime_raw(
            raw_path,
            _mapping(prime["cell"], "prime cell"),
            str(prime["run_id"]),
            validation_context,
            matrix_path=matrix_path,
            prompts_path=prompts_path,
        )
        prime_evidence.append(
            {
                "prime_index": prime["prime_index"],
                "cell": prime["cell"],
                "argv": prime["argv"],
                "environment": prime["environment"],
                "started_at_utc": started,
                "finished_at_utc": finished,
                "returncode": returncode,
                "stdout": prime["stdout"],
                "stderr": prime["stderr"],
                "raw_result": prime["raw_result"],
                "raw_result_sha256": _sha256(raw_path),
                "raw_result_row_count": row_count,
            }
        )

    inventory_after = _cache_inventory(reproducibility)
    _write_new_json(
        Path(str(preparation["cache_inventory_after"])), inventory_after
    )
    summary = {
        "contract_version": "rustinfer.repeatability-preparation.v1",
        "status": "passed",
        "completed_at_utc": _utc_now(),
        "policy": preparation["policy"],
        "reproducibility_environment": plan["reproducibility_environment"],
        "project_environment": _mapping(
            plan["runtime_environment_policy"], "runtime environment policy"
        )["project_environment"],
        "immutable_evidence": preparation["immutable_evidence"],
        "python_evidence": python_evidence,
        "uv_sync": {
            "argv": sync["argv"],
            "environment": sync["environment"],
            "started_at_utc": sync_started,
            "finished_at_utc": sync_finished,
            "returncode": 0,
            "stdout": sync["stdout"],
            "stderr": sync["stderr"],
        },
        "prime_invocations": prime_evidence,
        "prime_results_excluded_from_checker": True,
        "cache_inventory_before": _cache_inventory_summary(inventory_before),
        "cache_inventory_after": _cache_inventory_summary(inventory_after),
        "measured_cache_baseline_sha256": inventory_after["aggregate_sha256"],
    }
    _write_new_json(Path(str(preparation["summary"])), summary)
    return inventory_after


def _execute_plan(plan: Mapping[str, Any]) -> int:
    output_root = Path(str(plan["output_root"]))
    events_path = output_root / "execution-events.jsonl"
    try:
        measured_cache_baseline = _execute_preparation(plan)
    except RunnerError as error:
        _record_failure(
            output_root,
            stage="preparation",
            message=str(error),
            returncode=None,
        )
        print(f"repeatability runner: {error}", file=sys.stderr)
        return 2
    preflight_baseline: dict[str, str] | None = None
    for invocation_value in plan["invocations"]:
        invocation = _mapping(invocation_value, "planned invocation")
        artifact_dir = Path(str(invocation["artifact_dir"]))
        artifact_dir.mkdir(parents=True, exist_ok=False)
        artifacts = _mapping(invocation["artifacts"], "planned invocation artifacts")
        environment = _mapping(invocation["environment"], "planned environment")
        preflight_environment = _mapping(
            invocation["preflight_environment"], "planned preflight environment"
        )
        planned_reproducibility = _mapping(
            invocation["reproducibility_environment"],
            "planned invocation reproducibility environment",
        )
        current_reproducibility = {
            key: os.environ.get(key) for key in REPRODUCIBILITY_ENVIRONMENT_KEYS
        }
        if current_reproducibility != dict(planned_reproducibility):
            message = "allowlisted reproducibility environment changed after planning"
            _record_failure(
                output_root,
                stage="environment-drift",
                message=message,
                invocation=invocation,
            )
            print(f"repeatability runner: {message}", file=sys.stderr)
            return 2

        attempts_dir = Path(str(artifacts["preflight_attempts"]))
        attempts_dir.mkdir(parents=True, exist_ok=False)
        thermal_policy = _mapping(
            _mapping(plan["preflight"], "planned preflight")[
                "thermal_stabilization"
            ],
            "planned thermal stabilization",
        )
        retry_interval = int(thermal_policy["retry_interval_seconds"])
        maximum_wait = int(thermal_policy["maximum_wait_seconds"])
        maximum_attempts = 1 + maximum_wait // retry_interval
        returncode = 127
        launch_error: str | None = None
        final_attempt_stdout: Path | None = None
        final_attempt_stderr: Path | None = None
        for attempt in range(1, maximum_attempts + 1):
            attempt_stdout = attempts_dir / f"attempt-{attempt:03d}.stdout.txt"
            attempt_stderr = attempts_dir / f"attempt-{attempt:03d}.stderr.txt"
            started = _utc_now()
            returncode, launch_error = _run_captured(
                invocation["preflight_argv"],
                environment=preflight_environment,
                stdout_path=attempt_stdout,
                stderr_path=attempt_stderr,
            )
            finished = _utc_now()
            retryable, observed_temperature = _retryable_temperature_failure(
                returncode, attempt_stdout, attempt_stderr
            )
            _write_new_json(
                attempts_dir / f"attempt-{attempt:03d}.snapshot.json",
                {
                    "attempt": attempt,
                    "started_at_utc": started,
                    "finished_at_utc": finished,
                    "returncode": returncode,
                    "launch_error": launch_error,
                    "retryable_temperature_limit": retryable,
                    "observed_temperature_c": observed_temperature,
                },
            )
            _append_event(
                events_path,
                {
                    "ordinal": invocation["ordinal"],
                    "stage": "preflight",
                    "attempt": attempt,
                    "started_at_utc": started,
                    "finished_at_utc": finished,
                    "returncode": returncode,
                    "launch_error": launch_error,
                    "retryable_temperature_limit": retryable,
                },
            )
            if returncode == 0 or not retryable or attempt == maximum_attempts:
                final_attempt_stdout = attempt_stdout
                final_attempt_stderr = attempt_stderr
                break
            time.sleep(retry_interval)

        if final_attempt_stdout is None or final_attempt_stderr is None:
            raise RunnerError("preflight stabilization did not select a final attempt")
        _write_new_bytes(
            Path(str(artifacts["preflight_stdout"])),
            final_attempt_stdout.read_bytes(),
        )
        _write_new_bytes(
            Path(str(artifacts["preflight_stderr"])),
            final_attempt_stderr.read_bytes(),
        )
        if returncode != 0:
            message = f"preflight failed with exit code {returncode}"
            _record_failure(
                output_root,
                stage="preflight",
                message=message,
                invocation=invocation,
                returncode=returncode,
            )
            print(f"repeatability runner: {message}", file=sys.stderr)
            return 2

        try:
            parsed, snapshot = _parse_preflight(
                Path(str(artifacts["preflight_stdout"])), str(plan["git_revision"])
            )
        except RunnerError as error:
            _record_failure(
                output_root,
                stage="preflight-contract",
                message=str(error),
                invocation=invocation,
                returncode=0,
            )
            print(f"repeatability runner: {error}", file=sys.stderr)
            return 2
        _write_new_json(
            Path(str(artifacts["preflight_snapshot"])),
            {"all_values": parsed, "comparability": snapshot},
        )
        if preflight_baseline is None:
            preflight_baseline = snapshot
            _write_new_json(
                output_root / "preflight-baseline.json",
                {
                    "contract_version": "rustinfer.preflight-baseline.v1",
                    "source_ordinal": invocation["ordinal"],
                    "comparability": snapshot,
                },
            )
        else:
            drift = _preflight_drift(preflight_baseline, snapshot)
            if drift:
                message = (
                    "preflight settings differ from the first invocation for keys: "
                    + ", ".join(drift)
                )
                _record_failure(
                    output_root,
                    stage="preflight-comparability",
                    message=message,
                    invocation=invocation,
                    returncode=0,
                )
                print(f"repeatability runner: {message}", file=sys.stderr)
                return 2

        started = _utc_now()
        returncode, launch_error = _run_captured(
            invocation["benchmark_argv"],
            environment=environment,
            stdout_path=Path(str(artifacts["benchmark_stdout"])),
            stderr_path=Path(str(artifacts["benchmark_stderr"])),
        )
        _append_event(
            events_path,
            {
                "ordinal": invocation["ordinal"],
                "stage": "benchmark",
                "started_at_utc": started,
                "finished_at_utc": _utc_now(),
                "returncode": returncode,
                "launch_error": launch_error,
            },
        )
        if returncode != 0:
            message = f"benchmark failed with exit code {returncode}"
            _record_failure(
                output_root,
                stage="benchmark",
                message=message,
                invocation=invocation,
                returncode=returncode,
            )
            print(f"repeatability runner: {message}", file=sys.stderr)
            return 2
        raw_path = Path(str(invocation["raw_result"]))
        if not raw_path.is_file():
            message = f"benchmark did not create {raw_path}"
            _record_failure(
                output_root,
                stage="benchmark-output",
                message=message,
                invocation=invocation,
            )
            print(f"repeatability runner: {message}", file=sys.stderr)
            return 2
        cache_inventory = _cache_inventory(planned_reproducibility)
        _write_new_json(
            Path(str(artifacts["cache_snapshot"])),
            _cache_inventory_summary(cache_inventory),
        )
        if (
            cache_inventory["aggregate_sha256"]
            != measured_cache_baseline["aggregate_sha256"]
        ):
            _write_new_json(
                Path(str(artifacts["cache_drift_inventory"])), cache_inventory
            )
            message = (
                "external cache inventory changed after preparation; measured "
                "invocation may have fetched or compiled new artifacts"
            )
            _record_failure(
                output_root,
                stage="cache-drift",
                message=message,
                invocation=invocation,
            )
            print(f"repeatability runner: {message}", file=sys.stderr)
            return 2

    checker = _mapping(plan["checker"], "planned checker")
    started = _utc_now()
    returncode, launch_error = _run_captured(
        checker["argv"],
        environment=_mapping(checker["environment"], "planned checker environment"),
        stdout_path=Path(str(checker["stdout"])),
        stderr_path=Path(str(checker["stderr"])),
    )
    _append_event(
        events_path,
        {
            "stage": "checker",
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "returncode": returncode,
            "launch_error": launch_error,
        },
    )
    report_path = Path(str(checker["report"]))
    if returncode != 0:
        message = f"repeatability checker failed with exit code {returncode}"
        _record_failure(
            output_root,
            stage="checker",
            message=message,
            returncode=returncode,
        )
        print(f"repeatability runner: {message}", file=sys.stderr)
        return 1
    try:
        report = _load_json(report_path, "repeatability report")
        _require_passing_repeatability_report(report, "repeatability report")
    except RunnerError as error:
        _record_failure(
            output_root, stage="checker-report", message=str(error), returncode=0
        )
        print(f"repeatability runner: {error}", file=sys.stderr)
        return 1

    _write_new_json(
        output_root / "completion.json",
        {
            "contract_version": "rustinfer.repeatability-runner.v1",
            "completed_at_utc": _utc_now(),
            "status": "passed",
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
        },
    )
    return 0


def _combine_raw_results(plan: Mapping[str, Any], destination: Path) -> int:
    combined = bytearray()
    nonempty_lines = 0
    for invocation_value in plan["invocations"]:
        invocation = _mapping(invocation_value, "planned invocation")
        raw_path = Path(str(invocation["raw_result"]))
        if raw_path.is_symlink() or not raw_path.is_file():
            raise RunnerError(f"cannot combine non-regular raw artifact: {raw_path}")
        try:
            content = raw_path.read_bytes()
        except OSError as error:
            raise RunnerError(f"cannot read raw artifact {raw_path}: {error}") from error
        if content:
            combined.extend(content)
            if not content.endswith(b"\n"):
                combined.extend(b"\n")
            nonempty_lines += sum(1 for line in content.splitlines() if line.strip())
    _write_new_bytes(destination, bytes(combined))
    return nonempty_lines


def _render_result_readme(
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    raw_sha256: str,
    raw_lines: int,
) -> str:
    runner_command = shlex.join([str(item) for item in plan["runner_argv"]])
    benchmark_commands = "\n".join(
        shlex.join([str(item) for item in invocation["benchmark_argv"]])
        for invocation in plan["invocations"]
    )
    summary = {
        "status": report.get("status"),
        "passed": report.get("passed"),
        "thresholds": report.get("thresholds"),
    }
    variance = report.get("cells", [])
    comparability = report.get("comparability", {})
    preparation = _load_json(
        Path(str(_mapping(plan["preparation"], "planned preparation")["summary"])),
        "preparation summary",
    )
    return f"""# PR-01 repeatability evidence

This directory is an append-only import of one passing, externally staged
repeatability gate for lane `{plan['lane_id']}` at Git revision
`{plan['git_revision']}`.

## Exact runner invocation

```shell
{runner_command}
```

## Summary

- Gate status: `passed`
- Independent runs: {plan['independent_runs']}
- Predeclared cells per run: {plan['cells_per_run']}
- Fresh single-cell benchmark subprocesses: {plan['invocation_count']}
- Combined raw observations: {raw_lines} JSONL rows
- Combined raw SHA-256: `{raw_sha256}`

```json
{json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)}
```

## Variance and threshold evidence

The values below are copied from the passing repeatability report. Statistical
definitions and the complete report remain in `repeatability-report.json`.

```json
{json.dumps(variance, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)}
```

## Comparability

```json
{json.dumps(comparability, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)}
```

All 20 preflight stdout/stderr snapshots, the canonical baseline, lane logs,
per-cell raw files, dependency hashes, and exact argv/environment evidence are
preserved below this directory. `raw.jsonl` is the canonical deterministic
concatenation of the 20 per-cell raw files and was checked again before import.

## Cache preparation and cold scope

`cold` means a fresh benchmark subprocess and freshly loaded model state. It is
not an OS page-cache, immutable model-file cache, uv package cache, or compiled
kernel disk-cache cold start. Before measurement the runner completed offline
`uv sync --frozen --offline`; for the selected lane it also ran one unmeasured
fresh process for every distinct repeatability compile/model profile. Those prime raws are preserved under
`preparation/` and excluded from the checker. The external cache roots below
were reused unchanged by all 20 measured invocations; every invocation stored
an inventory fingerprint equal to the post-prime baseline.

```json
{json.dumps(preparation, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)}
```

## Exact lane commands

```shell
{benchmark_commands}
```

## Known limitations

- This gate covers only the four cells predeclared by PR-01, not all 48 matrix cells.
- It establishes repeatability for one lane and one primary environment; it does
  not by itself establish cross-lane performance superiority.
- External caches are inventory-fingerprinted rather than copied into Git;
  model weights and profiler traces are not copied into this artifact.
"""


def _prepare_finalize_artifacts(
    staging_root: Path, plan: Mapping[str, Any]
) -> None:
    finalization = _mapping(plan.get("finalization"), "planned finalization")
    combined_raw = Path(str(finalization["combined_raw"]))
    raw_lines = _combine_raw_results(plan, combined_raw)
    checker_argv = [str(item) for item in finalization["validation_argv"]]
    returncode, launch_error = _run_captured(
        checker_argv,
        environment=_mapping(
            _mapping(plan["checker"], "planned checker")["environment"],
            "planned checker environment",
        ),
        stdout_path=Path(str(finalization["validation_stdout"])),
        stderr_path=Path(str(finalization["validation_stderr"])),
    )
    _append_event(
        staging_root / "execution-events.jsonl",
        {
            "stage": "finalize-checker",
            "started_at_utc": None,
            "finished_at_utc": _utc_now(),
            "returncode": returncode,
            "launch_error": launch_error,
        },
    )
    if returncode != 0:
        raise RunnerError(
            f"combined raw repeatability validation failed with exit code {returncode}"
        )
    validation_report = _load_json(
        Path(str(finalization["validation_report"])),
        "combined raw repeatability report",
    )
    _require_passing_repeatability_report(
        validation_report, "combined raw repeatability report"
    )

    report = _load_json(
        Path(str(_mapping(plan["checker"], "planned checker")["report"])),
        "repeatability report",
    )
    _require_passing_repeatability_report(report, "repeatability report")
    baseline = _load_json(
        staging_root / "preflight-baseline.json", "preflight baseline"
    )
    preparation_summary = _load_json(
        Path(str(_mapping(plan["preparation"], "planned preparation")["summary"])),
        "preparation summary",
    )
    raw_sha256 = _sha256(combined_raw)
    raw_bytes = combined_raw.stat().st_size
    metadata = {
        "contract_version": "rustinfer.benchmark-result.v1",
        "status": "passed",
        "lane_id": plan["lane_id"],
        "git_revision": plan["git_revision"],
        "created_at_utc": plan["created_at_utc"],
        "finalize_destination": finalization["destination"],
        "runner_argv": plan["runner_argv"],
        "benchmark_argv": [
            invocation["benchmark_argv"] for invocation in plan["invocations"]
        ],
        "checker_argv": _mapping(plan["checker"], "planned checker")["argv"],
        "combined_raw_checker_argv": checker_argv,
        "inputs": plan["inputs"],
        "reproducibility_environment": plan["reproducibility_environment"],
        "preparation": preparation_summary,
        "preflight_baseline": baseline,
        "combined_raw": {
            "path": "raw.jsonl",
            "bytes": raw_bytes,
            "nonempty_lines": raw_lines,
            "sha256": raw_sha256,
        },
        "repeatability_summary": report,
        "combined_raw_validation": validation_report,
        "known_limitations": [
            "only the four PR-01 repeatability cells are included",
            "one lane and one primary environment are evaluated",
            "large external model and profiler artifacts are not imported",
        ],
    }
    _write_new_json(Path(str(finalization["metadata"])), metadata)
    _write_new_bytes(
        Path(str(finalization["readme"])),
        _render_result_readme(plan, report, raw_sha256, raw_lines).encode("utf-8"),
    )


def _tree_manifest(source: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    try:
        entries = sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix())
    except OSError as error:
        raise RunnerError(f"cannot enumerate staging tree {source}: {error}") from error
    for path in entries:
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise RunnerError(f"staging tree cannot contain a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RunnerError(f"staging tree contains a non-regular file: {relative}")
        try:
            size = path.stat().st_size
        except OSError as error:
            raise RunnerError(f"cannot stat staging artifact {path}: {error}") from error
        files.append({"path": relative, "bytes": size, "sha256": _sha256(path)})
    return files


def _verify_tree_manifest(root: Path, files: Sequence[Mapping[str, Any]]) -> None:
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RunnerError("finalize manifest contains an invalid relative path")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RunnerError(f"finalized artifact is not a regular file: {relative}")
        if path.stat().st_size != item.get("bytes") or _sha256(path) != item.get(
            "sha256"
        ):
            raise RunnerError(f"finalized artifact hash mismatch: {relative}")


def _finalize_staging(
    staging_root: Path, destination: Path, plan: Mapping[str, Any]
) -> None:
    destination = _safe_finalize_destination(destination)
    _validate_finalize_name(destination, str(plan["implementation_id"]))
    _tree_manifest(staging_root)
    _prepare_finalize_artifacts(staging_root, plan)
    files = _tree_manifest(staging_root)
    manifest_path = staging_root / "finalize-manifest.json"
    _write_new_json(
        manifest_path,
        {
            "contract_version": "rustinfer.benchmark-finalize.v1",
            "created_at_utc": _utc_now(),
            "source_staging_root": str(staging_root),
            "destination": str(destination),
            "git_revision": plan["git_revision"],
            "lane_id": plan["lane_id"],
            "file_count_excluding_this_manifest": len(files),
            "files_excluding_this_manifest": files,
        },
    )
    temporary = destination.parent / (
        f".{destination.name}.importing-{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(staging_root, temporary, symlinks=False)
        _verify_tree_manifest(temporary, files)
        copied_manifest = temporary / manifest_path.name
        if _sha256(copied_manifest) != _sha256(manifest_path):
            raise RunnerError("copied finalize-manifest.json hash mismatch")
        if destination.exists() or destination.is_symlink():
            raise RunnerError(f"finalize destination appeared during import: {destination}")
        os.rename(temporary, destination)
    except (OSError, RunnerError) as error:
        if temporary.exists() and temporary.parent == destination.parent:
            shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(error, RunnerError):
            raise
        raise RunnerError(f"cannot finalize staging artifacts: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=SUPPORTED_LANES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--matrix", type=Path, default=CANONICAL_MATRIX
    )
    parser.add_argument(
        "--prompts", type=Path, default=CANONICAL_PROMPTS
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/scripts/preflight.sh",
    )
    parser.add_argument(
        "--checker",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/scripts/check_repeatability.py",
    )
    parser.add_argument(
        "--uv",
        default="uv",
        help="replace only the leading literal uv in the lane manifest argv",
    )
    parser.add_argument(
        "--finalize-to",
        type=Path,
        help=(
            "after a passing gate, atomically import staging into one new direct "
            "child of benchmarks/results"
        ),
    )
    parser.add_argument(
        "--allow-noncanonical-tools",
        action="store_true",
        help="allow preflight/checker overrides for offline tests; disables finalize",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argument_vector = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argument_vector)
    args.runner_argv = [sys.executable, str(Path(__file__).resolve()), *argument_vector]
    try:
        output_root = _safe_output_root(args.output_root)
        plan = _build_plan(args, output_root)
        output_root.mkdir(parents=True, exist_ok=False)
        _write_new_json(output_root / "execution-plan.json", plan)
        returncode = _execute_plan(plan)
        if returncode != 0:
            return returncode
        if plan["finalize_to"] is not None:
            try:
                destination = Path(str(plan["finalize_to"]))
                _finalize_staging(output_root, destination, plan)
            except RunnerError as error:
                _record_failure(
                    output_root,
                    stage="finalize",
                    message=str(error),
                    returncode=None,
                )
                print(f"repeatability runner: {error}", file=sys.stderr)
                return 2
            print(
                "repeatability gate passed; staging preserved at "
                f"{output_root}; finalized artifacts: {destination}"
            )
        else:
            print(f"repeatability gate passed; staging artifacts: {output_root}")
        return 0
    except (RunnerError, OSError) as error:
        print(f"repeatability runner: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
