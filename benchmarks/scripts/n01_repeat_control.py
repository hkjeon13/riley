#!/usr/bin/env python3
"""Run a bounded N01 command repeatedly and retain host-pressure evidence.

This is an offline benchmark controller.  It launches the exact argv supplied
after ``--`` with ``shell=False`` and never becomes part of Riley's serving
runtime.  The controller deliberately retains every configured timed attempt,
including non-zero exits and timeouts.  Linux PSI, host-memory, and GPU-0
``nvidia-smi`` observations are recorded before and after each attempt so an
operating server's I/O pressure is evidence to analyse, not a reason to erase
an inconvenient sample.

GPU observations are discrete pre/post samples only.  They do not establish a
continuous GPU high-water mark, and this receipt alone is neither a serving
performance qualification nor a correctness qualification.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Sequence


RECEIPT_SCHEMA_VERSION = "riley.n01-repeat-control-receipt.v1"
DEFAULT_RECEIPT_NAME = "n01-repeat-control-receipt.json"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 260_913
GPU_INDEX = 0
MIB_BYTES = 1024 * 1024
PSI_RESOURCES = ("cpu", "io", "memory")
FORBIDDEN_BLENDER_NAMES = {
    "blender",
    "blender.exe",
    "restore-blender",
    "blender-restore",
    "restore_blender",
    "blender_restore",
}


class ControlError(ValueError):
    """A command-line or output-contract error for the repeat controller."""


@dataclass(frozen=True)
class RepeatControlConfig:
    """Validated inputs for a repeat-control receipt."""

    output_dir: Path
    cwd: Path
    command: tuple[str, ...]
    warmups: int
    repeats: int
    timeout_seconds: float
    nvidia_smi: str
    proc_root: Path
    bootstrap_resamples: int
    bootstrap_seed: int
    receipt_name: str


def utc_now() -> str:
    return (
        datetime_module.datetime.now(datetime_module.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_argv(argv: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ControlError(f"{label} must be a positive finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ControlError(f"{label} must be a positive finite number") from error
    if not math.isfinite(converted) or converted <= 0:
        raise ControlError(f"{label} must be a positive finite number")
    return converted


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ControlError(f"{label} must be an integer >= 1")
    return value


def _resolve_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ControlError(f"{label} cannot be resolved: {path}") from error
    if not resolved.is_dir():
        raise ControlError(f"{label} must be a directory: {path}")
    return resolved


def _is_forbidden_blender_argv(command: Sequence[str]) -> bool:
    return any(Path(token).name.casefold() in FORBIDDEN_BLENDER_NAMES for token in command)


def validate_config(config: RepeatControlConfig) -> RepeatControlConfig:
    """Reject malformed inputs before any benchmark command is launched."""
    if not config.command or any(not isinstance(token, str) or not token for token in config.command):
        raise ControlError("command argv after -- must be a nonempty string array")
    if _is_forbidden_blender_argv(config.command):
        raise ControlError("command argv may not invoke Blender or a Blender restoration helper")
    _positive_integer(config.warmups, "warmups")
    _positive_integer(config.repeats, "repeats")
    _positive_integer(config.bootstrap_resamples, "bootstrap-resamples")
    _finite_positive(config.timeout_seconds, "timeout-seconds")
    if not isinstance(config.bootstrap_seed, int) or isinstance(config.bootstrap_seed, bool):
        raise ControlError("bootstrap-seed must be an integer")
    if not isinstance(config.nvidia_smi, str) or not config.nvidia_smi:
        raise ControlError("nvidia-smi must be a nonempty executable path or command name")
    if not isinstance(config.receipt_name, str) or not config.receipt_name:
        raise ControlError("receipt-name must be a nonempty filename")
    receipt_component = Path(config.receipt_name)
    if receipt_component.name != config.receipt_name or config.receipt_name in {".", ".."}:
        raise ControlError("receipt-name must be a filename inside output-dir")
    if not config.receipt_name.endswith(".json"):
        raise ControlError("receipt-name must end in .json")
    cwd = _resolve_directory(config.cwd, "working-directory")
    try:
        output_dir = config.output_dir.expanduser().resolve()
    except OSError as error:
        raise ControlError(f"output-dir cannot be resolved: {config.output_dir}") from error
    if output_dir.exists() and not output_dir.is_dir():
        raise ControlError(f"output-dir must be a directory: {config.output_dir}")
    return RepeatControlConfig(
        output_dir=output_dir,
        cwd=cwd,
        command=tuple(config.command),
        warmups=config.warmups,
        repeats=config.repeats,
        timeout_seconds=float(config.timeout_seconds),
        nvidia_smi=config.nvidia_smi,
        proc_root=config.proc_root.expanduser(),
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
        receipt_name=config.receipt_name,
    )


def _parse_psi_line(line: str, resource: str) -> tuple[str, dict[str, float | int]]:
    fields = line.split()
    if not fields or fields[0] not in {"some", "full"}:
        raise ValueError(f"{resource} PSI line must start with some or full")
    metrics: dict[str, str] = {}
    for field in fields[1:]:
        if "=" not in field:
            raise ValueError(f"{resource} PSI metric is malformed: {field!r}")
        key, value = field.split("=", 1)
        if key in metrics:
            raise ValueError(f"{resource} PSI metric is duplicated: {key}")
        metrics[key] = value
    required = {"avg10", "avg60", "avg300", "total"}
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"{resource} PSI line lacks " + ", ".join(missing))
    parsed: dict[str, float | int] = {}
    for key in ("avg10", "avg60", "avg300"):
        value = float(metrics[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{resource} PSI {key} must be finite and nonnegative")
        parsed[key] = value
    try:
        total = int(metrics["total"])
    except ValueError as error:
        raise ValueError(f"{resource} PSI total must be an integer") from error
    if total < 0:
        raise ValueError(f"{resource} PSI total must be nonnegative")
    parsed["total"] = total
    return fields[0], parsed


def parse_psi_snapshot(text: str, resource: str) -> dict[str, dict[str, float | int] | None]:
    """Parse one Linux PSI file, retaining the optional ``full`` line."""
    parsed: dict[str, dict[str, float | int] | None] = {"some": None, "full": None}
    for line in text.splitlines():
        if not line.strip():
            continue
        category, metrics = _parse_psi_line(line, resource)
        if parsed[category] is not None:
            raise ValueError(f"{resource} PSI has duplicate {category} lines")
        parsed[category] = metrics
    if parsed["some"] is None:
        raise ValueError(f"{resource} PSI lacks a some line")
    return parsed


def linux_psi_snapshot(proc_root: Path) -> dict[str, dict[str, Any]]:
    """Capture CPU, I/O, and memory PSI without turning pressure into a gate."""
    result: dict[str, dict[str, Any]] = {}
    for resource in PSI_RESOURCES:
        source = proc_root / "pressure" / resource
        try:
            parsed = parse_psi_snapshot(source.read_text(encoding="utf-8"), resource)
            result[resource] = {
                "status": "ok",
                "source": str(source),
                "some": parsed["some"],
                "full": parsed["full"],
                "error": None,
            }
        except FileNotFoundError:
            result[resource] = {
                "status": "unavailable",
                "source": str(source),
                "some": None,
                "full": None,
                "error": "PSI file is unavailable",
            }
        except ValueError as error:
            result[resource] = {
                "status": "malformed",
                "source": str(source),
                "some": None,
                "full": None,
                "error": str(error),
            }
        except OSError as error:
            result[resource] = {
                "status": "unavailable",
                "source": str(source),
                "some": None,
                "full": None,
                "error": f"{type(error).__name__}: {error}",
            }
    return result


def parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        fields = raw_value.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError as error:
            raise ValueError(f"meminfo value for {key} is not an integer") from error
        if value < 0:
            raise ValueError(f"meminfo value for {key} must be nonnegative")
        if len(fields) >= 2 and fields[1] not in {"kB", "KB", "kb"}:
            raise ValueError(f"meminfo unit for {key} must be kB")
        values[key] = value * 1024
    if "MemTotal" not in values:
        raise ValueError("meminfo lacks MemTotal")
    return values


def linux_host_memory_snapshot(proc_root: Path) -> dict[str, Any]:
    source = proc_root / "meminfo"
    try:
        values = parse_meminfo(source.read_text(encoding="utf-8"))
        return {
            "status": "ok",
            "source": str(source),
            "mem_total_bytes": values["MemTotal"],
            "mem_available_bytes": values.get("MemAvailable"),
            "mem_free_bytes": values.get("MemFree"),
            "error": None,
        }
    except FileNotFoundError:
        return {
            "status": "unavailable",
            "source": str(source),
            "mem_total_bytes": None,
            "mem_available_bytes": None,
            "mem_free_bytes": None,
            "error": "meminfo is unavailable",
        }
    except ValueError as error:
        return {
            "status": "malformed",
            "source": str(source),
            "mem_total_bytes": None,
            "mem_available_bytes": None,
            "mem_free_bytes": None,
            "error": str(error),
        }
    except OSError as error:
        return {
            "status": "unavailable",
            "source": str(source),
            "mem_total_bytes": None,
            "mem_available_bytes": None,
            "mem_free_bytes": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _parse_mib(raw: str) -> int:
    normalized = raw.strip().removesuffix("MiB").strip()
    try:
        value = Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError(f"GPU memory value is not MiB: {raw!r}") from error
    if not value.is_finite() or value < 0:
        raise ValueError(f"GPU memory value must be finite and nonnegative: {raw!r}")
    return int(value * MIB_BYTES)


def parse_nvidia_smi_gpu0(text: str) -> dict[str, Any]:
    """Parse the fixed CSV query and select physical GPU index zero."""
    devices: dict[int, dict[str, Any]] = {}
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != 5:
            raise ValueError("nvidia-smi row must have index, uuid, name, memory.used, memory.total")
        try:
            index = int(row[0].strip())
        except ValueError as error:
            raise ValueError(f"nvidia-smi GPU index is invalid: {row[0]!r}") from error
        if index in devices:
            raise ValueError(f"nvidia-smi reports GPU index {index} more than once")
        devices[index] = {
            "index": index,
            "uuid": row[1].strip(),
            "name": row[2].strip(),
            "used_bytes": _parse_mib(row[3]),
            "total_bytes": _parse_mib(row[4]),
        }
    if GPU_INDEX not in devices:
        raise ValueError("nvidia-smi did not report physical GPU index 0")
    return devices[GPU_INDEX]


def nvidia_smi_gpu0_snapshot(nvidia_smi: str) -> dict[str, Any]:
    command = [
        nvidia_smi,
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"nvidia-smi exited with code {completed.returncode}: {completed.stderr.strip()}")
        device = parse_nvidia_smi_gpu0(completed.stdout)
        return {
            "status": "ok",
            "gpu_index": GPU_INDEX,
            "command": command,
            "used_bytes": device["used_bytes"],
            "total_bytes": device["total_bytes"],
            "uuid": device["uuid"],
            "name": device["name"],
            "error": None,
        }
    except Exception as error:
        return {
            "status": "unavailable",
            "gpu_index": GPU_INDEX,
            "command": command,
            "used_bytes": None,
            "total_bytes": None,
            "uuid": None,
            "name": None,
            "error": f"{type(error).__name__}: {error}",
        }


def environment_snapshot(config: RepeatControlConfig) -> dict[str, Any]:
    """Capture an observation that can be retained even when a probe fails."""
    return {
        "at_utc": utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "psi": linux_psi_snapshot(config.proc_root),
        "host_memory": linux_host_memory_snapshot(config.proc_root),
        "gpu_0": nvidia_smi_gpu0_snapshot(config.nvidia_smi),
    }


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _log_paths(output_dir: Path, kind: str, index: int) -> tuple[Path, Path]:
    prefix = f"{kind}-{index:03d}"
    return output_dir / f"{prefix}.stdout.log", output_dir / f"{prefix}.stderr.log"


def _planned_paths(config: RepeatControlConfig) -> list[Path]:
    paths = [config.output_dir / config.receipt_name]
    for kind, count in (("warmup", config.warmups), ("timed", config.repeats)):
        for index in range(1, count + 1):
            paths.extend(_log_paths(config.output_dir, kind, index))
    return paths


def prepare_output_dir(config: RepeatControlConfig) -> Path:
    """Create a receipt directory without overwriting any planned evidence."""
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ControlError(f"cannot create output-dir {config.output_dir}: {error}") from error
    if not config.output_dir.is_dir():
        raise ControlError(f"output-dir must be a directory: {config.output_dir}")
    existing = [path for path in _planned_paths(config) if path.exists() or path.is_symlink()]
    if existing:
        rendered = ", ".join(str(path) for path in existing[:3])
        more = "" if len(existing) <= 3 else f" (and {len(existing) - 3} more)"
        raise ControlError(f"refusing to overwrite existing repeat evidence: {rendered}{more}")
    return config.output_dir / config.receipt_name


def execute_attempt(
    config: RepeatControlConfig,
    *,
    kind: str,
    index: int,
) -> dict[str, Any]:
    """Run one warmup or timed attempt and always retain its observations."""
    stdout_path, stderr_path = _log_paths(config.output_dir, kind, index)
    pre = environment_snapshot(config)
    started_at = utc_now()
    started_ns = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    exit_code: int | None = None
    timed_out = False
    status = "failed-to-start"
    error: str | None = None
    try:
        with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
            process = subprocess.Popen(
                list(config.command),
                cwd=str(config.cwd),
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                shell=False,
            )
            try:
                process.wait(timeout=config.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                error = f"command exceeded {config.timeout_seconds:g} seconds"
                _stop_process(process)
            exit_code = process.returncode
            if timed_out:
                status = "timed-out"
            elif exit_code == 0:
                status = "succeeded"
            else:
                status = "failed"
                error = f"command exited with code {exit_code}"
    except KeyboardInterrupt:
        status = "interrupted"
        error = "controller interrupted"
        if process is not None:
            _stop_process(process)
            exit_code = process.returncode
    except BaseException as exception:
        status = "failed-to-start" if process is None else "failed"
        error = f"{type(exception).__name__}: {exception}"
        if process is not None:
            _stop_process(process)
            exit_code = process.returncode
    finally:
        finished_ns = time.monotonic_ns()
        finished_at = utc_now()
        post = environment_snapshot(config)
    elapsed_ns = finished_ns - started_ns
    return {
        "kind": kind,
        "index": index,
        "argv": list(config.command),
        "argv_sha256": sha256_argv(config.command),
        "shell": False,
        "cwd": str(config.cwd),
        "timeout_seconds": config.timeout_seconds,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_time_ns": elapsed_ns,
        "wall_time_ms": round(elapsed_ns / 1_000_000, 6),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "status": status,
        "error": error,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "pre": pre,
        "post": post,
    }


def r7_quantile(values: Iterable[float], probability: float) -> float:
    """Return the Hyndman-Fan type-7 quantile used for bootstrap percentiles."""
    observations = sorted(float(value) for value in values)
    if not observations:
        raise ControlError("quantile requires at least one observation")
    if not 0.0 <= probability <= 1.0:
        raise ControlError("quantile probability must be in [0, 1]")
    if any(not math.isfinite(value) for value in observations):
        raise ControlError("quantile observations must be finite")
    if len(observations) == 1:
        return observations[0]
    position = (len(observations) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    if fraction == 0:
        return observations[lower]
    return observations[lower] + fraction * (observations[lower + 1] - observations[lower])


def bootstrap_median_95_ci(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Calculate a deterministic percentile-bootstrap interval for the median."""
    if not values:
        return {
            "confidence_level": 0.95,
            "method": "percentile-bootstrap-r7",
            "resamples": resamples,
            "seed": seed,
            "lower_ms": None,
            "upper_ms": None,
            "reason": "no successful timed runs",
        }
    random_source = random.Random(seed)
    sample_count = len(values)
    resampled_medians = [
        statistics.median(values[random_source.randrange(sample_count)] for _ in range(sample_count))
        for _ in range(resamples)
    ]
    return {
        "confidence_level": 0.95,
        "method": "percentile-bootstrap-r7",
        "resamples": resamples,
        "seed": seed,
        "lower_ms": r7_quantile(resampled_medians, 0.025),
        "upper_ms": r7_quantile(resampled_medians, 0.975),
        "reason": None,
    }


def summarize_timed_runs(
    timed_runs: Sequence[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize only successful timed walls while exposing every failure count."""
    successful = [
        float(run["wall_time_ms"])
        for run in timed_runs
        if run.get("status") == "succeeded"
    ]
    failures = len(timed_runs) - len(successful)
    if successful:
        median = statistics.median(successful)
        mean = statistics.fmean(successful)
        stddev = statistics.stdev(successful) if len(successful) >= 2 else 0.0
        minimum = min(successful)
        maximum = max(successful)
        reason = None
    else:
        median = mean = stddev = minimum = maximum = None
        reason = "no successful timed runs"
    return {
        "count": len(timed_runs),
        "successful_count": len(successful),
        "failures": failures,
        "statistics_population": "successful timed runs only; failed and timed-out runs remain in timed_runs",
        "median_ms": median,
        "mean_ms": mean,
        "stddev_ms": stddev,
        "stddev_method": "sample standard deviation; 0.0 for one successful run",
        "min_ms": minimum,
        "max_ms": maximum,
        "bootstrap_median_95_ci_ms": bootstrap_median_95_ci(
            successful,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "reason": reason,
    }


def write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    """Write once with exclusive creation; an existing receipt is never replaced."""
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ControlError(f"refusing to overwrite immutable receipt: {path}") from error
    except OSError as error:
        raise ControlError(f"cannot write immutable receipt {path}: {error}") from error


def run_repeat_control(config: RepeatControlConfig) -> dict[str, Any]:
    """Execute all configured repetitions and write exactly one immutable receipt."""
    config = validate_config(config)
    receipt_path = prepare_output_dir(config)
    warmup_runs: list[dict[str, Any]] = []
    timed_runs: list[dict[str, Any]] = []
    interrupted = False
    for index in range(1, config.warmups + 1):
        run = execute_attempt(config, kind="warmup", index=index)
        warmup_runs.append(run)
        if run["status"] == "interrupted":
            interrupted = True
            break
    if not interrupted:
        for index in range(1, config.repeats + 1):
            run = execute_attempt(config, kind="timed", index=index)
            timed_runs.append(run)
            if run["status"] == "interrupted":
                interrupted = True
                break

    summary = summarize_timed_runs(
        timed_runs,
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
    )
    all_runs = [*warmup_runs, *timed_runs]
    failure_count = sum(run["status"] != "succeeded" for run in all_runs)
    if interrupted:
        status = "interrupted"
    elif failure_count:
        status = "completed-with-failures"
    else:
        status = "completed"
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "receipt_path": str(receipt_path),
        "output_dir": str(config.output_dir),
        "controller": {
            "offline_only": True,
            "command_execution": "argv with shell=False",
            "gpu_index": GPU_INDEX,
            "receipt_write_mode": "exclusive-create-no-overwrite",
        },
        "command": {
            "argv": list(config.command),
            "argv_sha256": sha256_argv(config.command),
            "cwd": str(config.cwd),
            "timeout_seconds": config.timeout_seconds,
        },
        "configuration": {
            "warmups": config.warmups,
            "timed_repeats": config.repeats,
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_seed": config.bootstrap_seed,
            "proc_root": str(config.proc_root),
            "nvidia_smi": config.nvidia_smi,
        },
        "retention_policy": {
            "timed_runs": "all configured timed attempts are retained, including non-zero exits and timeouts",
            "host_pressure": "PSI is observed and retained; high-pressure runs are not discarded",
        },
        "limitations": [
            "GPU-0 observations are discrete pre/post nvidia-smi samples, not a continuous high-water measurement.",
            "This receipt does not by itself qualify serving performance, latency, stability, or correctness.",
        ],
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "summary": summary,
        "status": status,
        "failure_count_including_warmups": failure_count,
    }
    write_immutable_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for immutable receipt and per-run logs")
    parser.add_argument("--working-directory", type=Path, default=Path.cwd(), help="CWD for every command attempt")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup attempts; must be at least 1")
    parser.add_argument("--repeats", type=int, default=5, help="Timed attempts; must be at least 1")
    parser.add_argument("--timeout-seconds", type=float, default=300.0, help="Per-attempt wall timeout")
    parser.add_argument("--nvidia-smi", default="nvidia-smi", help="nvidia-smi executable path or command name")
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"), help="Linux procfs root; injectable for tests")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Deterministic bootstrap resample count; must be at least 1",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Deterministic bootstrap random seed",
    )
    parser.add_argument(
        "--receipt-name",
        default=DEFAULT_RECEIPT_NAME,
        help="JSON filename created once inside output-dir",
    )
    return parser


def parse_command_line(argv: Sequence[str] | None = None) -> RepeatControlConfig:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if "-h" in raw_argv or "--help" in raw_argv:
        parser.parse_args(raw_argv)
    if "--" not in raw_argv:
        raise ControlError("command argv must follow -- (for example: -- /path/to/benchmark --flag value)")
    separator = raw_argv.index("--")
    namespace = parser.parse_args(raw_argv[:separator])
    command = tuple(raw_argv[separator + 1 :])
    config = RepeatControlConfig(
        output_dir=namespace.output_dir,
        cwd=namespace.working_directory,
        command=command,
        warmups=namespace.warmups,
        repeats=namespace.repeats,
        timeout_seconds=namespace.timeout_seconds,
        nvidia_smi=namespace.nvidia_smi,
        proc_root=namespace.proc_root,
        bootstrap_resamples=namespace.bootstrap_resamples,
        bootstrap_seed=namespace.bootstrap_seed,
        receipt_name=namespace.receipt_name,
    )
    return validate_config(config)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_command_line(argv)
        receipt = run_repeat_control(config)
    except ControlError as error:
        print(f"n01_repeat_control: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"receipt_path": receipt["receipt_path"], "status": receipt["status"]}, sort_keys=True))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
