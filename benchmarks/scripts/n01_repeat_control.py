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
import re
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
N06A_PARENT_CLEANUP_SCHEMA_VERSION = "riley.n01-n06a-parent-cleanup.v1"
N06A_TIMEOUT_DESCENDANT_SCHEMA_VERSION = "riley.n01-n06a-timeout-descendants.v1"
N06A_DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION = "riley.n01-n06a-docker-container-cleanup.v1"
N06A_DOCKER_CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
MAX_N06A_DOCKER_CLEANUP_COMMAND_SECONDS = 30.0
MAX_N06A_DOCKER_CLEANUP_OUTPUT_BYTES = 64 * 1024
MAX_N06A_DESCENDANT_CLEANUP_SECONDS = 5.0
N06A_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS = "not-started-after-failed-cleanup"
N06A_NOT_STARTED_AFTER_FAILED_CLEANUP_REASON = (
    "N06-A parent cleanup was unproven; fail-stop prevents further launches"
)
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
class N06ATimeoutCleanupConfig:
    """Explicit, narrowly scoped parent cleanup for an N06-A child attempt.

    This is intentionally opt-in.  It derives one deterministic Docker name
    from the N06-A artifact root and outer attempt identity; it never searches
    for, stops, or removes a container by image, port, label, or broad prefix.
    """

    artifact_root: Path
    docker_launcher: str
    command_timeout_seconds: float = MAX_N06A_DOCKER_CLEANUP_COMMAND_SECONDS


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
    n06a_timeout_cleanup: N06ATimeoutCleanupConfig | None = None


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


def _validate_n06a_timeout_cleanup(
    cleanup: N06ATimeoutCleanupConfig | None,
) -> N06ATimeoutCleanupConfig | None:
    """Normalize the only opt-in path that can control an N06-A container.

    N01 must not infer this from a benchmark argv or an environment variable.
    Both the artifact root and Docker launcher have to be supplied by the
    operator, so a normal repeat-controller run retains its legacy behavior.
    """
    if cleanup is None:
        return None
    if not isinstance(cleanup, N06ATimeoutCleanupConfig):
        raise ControlError("n06a-timeout-cleanup must be an N06ATimeoutCleanupConfig")
    if not isinstance(cleanup.artifact_root, Path):
        raise ControlError("n06a-timeout-cleanup artifact-root must be a Path")
    if not isinstance(cleanup.docker_launcher, str) or not cleanup.docker_launcher:
        raise ControlError("n06a-timeout-cleanup Docker launcher must be a nonempty argv[0]")
    if "\x00" in cleanup.docker_launcher:
        raise ControlError("n06a-timeout-cleanup Docker launcher may not contain NUL")
    command_timeout_seconds = _finite_positive(
        cleanup.command_timeout_seconds,
        "n06a-timeout-cleanup command-timeout-seconds",
    )
    if command_timeout_seconds > MAX_N06A_DOCKER_CLEANUP_COMMAND_SECONDS:
        raise ControlError(
            "n06a-timeout-cleanup command-timeout-seconds may not exceed "
            f"{MAX_N06A_DOCKER_CLEANUP_COMMAND_SECONDS:g}"
        )
    try:
        artifact_root = cleanup.artifact_root.expanduser().resolve()
    except (OSError, RuntimeError) as error:
        raise ControlError(
            "n06a-timeout-cleanup artifact-root cannot be resolved: "
            f"{cleanup.artifact_root}"
        ) from error
    if artifact_root.exists() and not artifact_root.is_dir():
        raise ControlError(
            "n06a-timeout-cleanup artifact-root must be a directory or a new directory: "
            f"{cleanup.artifact_root}"
        )
    return N06ATimeoutCleanupConfig(
        artifact_root=artifact_root,
        docker_launcher=cleanup.docker_launcher,
        command_timeout_seconds=command_timeout_seconds,
    )


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
    n06a_timeout_cleanup = _validate_n06a_timeout_cleanup(config.n06a_timeout_cleanup)
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
        n06a_timeout_cleanup=n06a_timeout_cleanup,
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


def n06a_attempt_artifact_directory(
    cleanup: N06ATimeoutCleanupConfig,
    *,
    kind: str,
    index: int,
) -> Path:
    """Reproduce N06-A's create-only artifact directory spelling.

    ``Path.resolve()`` intentionally uses its non-strict default here.  Before
    an N06-A child creates its directory, this is the path N06-A will resolve
    after its mkdir; after a child has run, it is byte-for-byte the same path
    string N06-A hashed for the Docker name.
    """
    if kind not in {"warmup", "timed"}:
        raise ControlError(f"N06-A cleanup phase is invalid: {kind!r}")
    _positive_integer(index, "N06-A cleanup attempt index")
    try:
        return (cleanup.artifact_root / f"{kind}-{index:03d}").resolve()
    except (OSError, RuntimeError) as error:
        raise ControlError(
            f"N06-A cleanup attempt directory cannot be resolved for {kind}-{index:03d}"
        ) from error


def n06a_vllm_container_name(
    cleanup: N06ATimeoutCleanupConfig,
    *,
    kind: str,
    index: int,
) -> tuple[Path, str]:
    """Return the exact vLLM name derived by N06-A for one outer attempt."""
    attempt_dir = n06a_attempt_artifact_directory(cleanup, kind=kind, index=index)
    digest = hashlib.sha256(str(attempt_dir).encode("utf-8")).hexdigest()[:20]
    name = f"riley-n06a-vllm-{kind}-{index:04d}-{digest}"
    if not N06A_DOCKER_CONTAINER_NAME_RE.fullmatch(name):
        raise ControlError("derived N06-A vLLM Docker container name is invalid")
    return attempt_dir, name


def _n06a_cleanup_receipt_path(output_dir: Path, kind: str, index: int) -> Path:
    return output_dir / f"{kind}-{index:03d}.n06a-parent-cleanup.json"


def _read_linux_process_identity(
    proc_root: Path,
    pid: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read the stable pieces of Linux ``/proc/<pid>/stat`` needed for safety.

    A PID alone is not a safe signal target: it can be recycled while cleanup
    is in progress.  The start-time tick is retained with every candidate and
    rechecked before a process-group signal is sent.
    """
    if not isinstance(pid, int) or pid <= 0:
        return None, f"invalid Linux PID {pid!r}"
    source = proc_root / str(pid) / "stat"
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as error:
        return None, f"cannot read {source}: {type(error).__name__}: {error}"
    closing_parenthesis = text.rfind(")")
    if closing_parenthesis < 2:
        return None, f"malformed {source}: missing comm terminator"
    try:
        observed_pid = int(text[: text.find(" ")])
    except ValueError:
        return None, f"malformed {source}: PID field is invalid"
    if observed_pid != pid:
        return None, f"malformed {source}: PID field differs from path"
    fields = text[closing_parenthesis + 1 :].split()
    # The suffix begins at field 3 (state); starttime is field 22.
    if len(fields) <= 19:
        return None, f"malformed {source}: stat has too few fields"
    try:
        parent_pid = int(fields[1])
        process_group = int(fields[2])
        session = int(fields[3])
        starttime_ticks = int(fields[19])
    except ValueError:
        return None, f"malformed {source}: numeric stat field is invalid"
    if parent_pid < 0 or process_group <= 0 or session <= 0 or starttime_ticks < 0:
        return None, f"malformed {source}: stat identity field is out of range"
    return {
        "pid": pid,
        "ppid": parent_pid,
        "pgid": process_group,
        "session": session,
        "starttime_ticks": starttime_ticks,
        "stat_path": str(source),
    }, None


def _linux_child_pids(proc_root: Path, parent_pid: int) -> tuple[list[int] | None, str | None]:
    """Return a task leader's direct children without scanning unrelated PIDs."""
    source = proc_root / str(parent_pid) / "task" / str(parent_pid) / "children"
    try:
        raw = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, f"Linux child list is unavailable for PID {parent_pid}: {source}"
    except OSError as error:
        return None, f"cannot read {source}: {type(error).__name__}: {error}"
    children: list[int] = []
    for token in raw.split():
        try:
            child_pid = int(token)
        except ValueError:
            return None, f"malformed Linux child PID {token!r} in {source}"
        if child_pid <= 0:
            return None, f"invalid Linux child PID {child_pid} in {source}"
        children.append(child_pid)
    if len(children) != len(set(children)):
        return None, f"duplicate Linux child PID in {source}"
    return children, None


def _snapshot_linux_descendants(proc_root: Path, root_pid: int) -> dict[str, Any]:
    """Snapshot only transitive children of the N01-owned outer process."""
    receipt: dict[str, Any] = {
        "schema_version": N06A_TIMEOUT_DESCENDANT_SCHEMA_VERSION,
        "root_pid": root_pid,
        "proc_root": str(proc_root),
        "processes": [],
        "errors": [],
        "snapshot_complete": False,
    }
    pending = [root_pid]
    visited_parents: set[int] = set()
    seen_children: set[int] = set()
    while pending:
        parent_pid = pending.pop()
        if parent_pid in visited_parents:
            continue
        visited_parents.add(parent_pid)
        children, error = _linux_child_pids(proc_root, parent_pid)
        if error is not None:
            # A descendant may exit after its identity was captured but before
            # we descend into its own task directory.  That is already-cleaned
            # evidence, not an incomplete snapshot of a still-live process.
            if parent_pid != root_pid:
                current, current_error = _read_linux_process_identity(proc_root, parent_pid)
                if current_error is None and current is None:
                    continue
            receipt["errors"].append(error)
            continue
        assert children is not None
        for child_pid in children:
            if child_pid in seen_children:
                receipt["errors"].append(
                    f"Linux descendant PID {child_pid} appeared under multiple parents"
                )
                continue
            identity, identity_error = _read_linux_process_identity(proc_root, child_pid)
            if identity_error is not None:
                receipt["errors"].append(identity_error)
                continue
            # A child can exit between the task/children read and stat lookup.
            if identity is None:
                continue
            if identity["ppid"] != parent_pid:
                receipt["errors"].append(
                    "Linux child identity changed during snapshot: "
                    f"expected parent {parent_pid}, observed {identity['ppid']} for PID {child_pid}"
                )
                continue
            seen_children.add(child_pid)
            receipt["processes"].append(identity)
            pending.append(child_pid)
    receipt["processes"].sort(key=lambda item: int(item["pid"]))
    receipt["snapshot_complete"] = not receipt["errors"]
    return receipt


def _same_linux_process_identity(
    current: dict[str, Any] | None,
    expected: dict[str, Any],
) -> bool:
    return current is not None and all(
        current.get(key) == expected.get(key)
        for key in ("pid", "pgid", "session", "starttime_ticks")
    )


def _recorded_group_leaders(
    snapshot: dict[str, Any],
    outer_process_group: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose only descendant-led groups; outer-group members are N01-owned."""
    leaders: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    identities = list(snapshot.get("processes", []))
    known_pids = {int(identity["pid"]) for identity in identities}
    for identity in identities:
        pid = int(identity["pid"])
        process_group = int(identity["pgid"])
        if process_group == outer_process_group:
            skipped.append({"pid": pid, "pgid": process_group, "reason": "covered-by-outer-process-group"})
        elif pid == process_group:
            leaders.append(identity)
        elif process_group not in known_pids:
            skipped.append(
                {
                    "pid": pid,
                    "pgid": process_group,
                    "reason": "group-leader-not-a-recorded-descendant",
                }
            )
    return leaders, skipped


def _signal_recorded_descendant_groups(
    process: subprocess.Popen[bytes],
    *,
    proc_root: Path,
) -> dict[str, Any]:
    """Snapshot then signal N06-A's separate server/Docker process sessions.

    This runs *before* N01 terminates the outer driver process.  It never
    signals a group unless that group leader was a verified descendant with the
    same PID/start-time identity at signal time.
    """
    receipt: dict[str, Any] = {
        "schema_version": N06A_TIMEOUT_DESCENDANT_SCHEMA_VERSION,
        "status": "unproven",
        "root_pid": process.pid,
        "outer_process_group": None,
        "snapshot": None,
        "candidate_groups": [],
        "skipped_processes": [],
        "actions": [],
        "errors": [],
        "cleanup_verified": False,
    }
    if os.name != "posix":
        receipt["errors"].append("Linux descendant cleanup is unavailable on this platform")
        return receipt
    try:
        outer_process_group = os.getpgid(process.pid)
    except OSError as error:
        receipt["errors"].append(
            f"cannot obtain outer process group for PID {process.pid}: {type(error).__name__}: {error}"
        )
        return receipt
    receipt["outer_process_group"] = outer_process_group
    snapshot = _snapshot_linux_descendants(proc_root, process.pid)
    receipt["snapshot"] = snapshot
    if not snapshot["snapshot_complete"]:
        receipt["errors"].append("Linux descendant snapshot was incomplete")
        return receipt
    leaders, skipped = _recorded_group_leaders(snapshot, outer_process_group)
    receipt["candidate_groups"] = [
        {
            "pgid": identity["pgid"],
            "leader_pid": identity["pid"],
            "leader_starttime_ticks": identity["starttime_ticks"],
        }
        for identity in leaders
    ]
    receipt["skipped_processes"] = skipped
    for identity in leaders:
        current, current_error = _read_linux_process_identity(proc_root, int(identity["pid"]))
        action: dict[str, Any] = {
            "signal": "SIGTERM",
            "pgid": identity["pgid"],
            "leader_pid": identity["pid"],
            "leader_starttime_ticks": identity["starttime_ticks"],
        }
        if current_error is not None:
            action["status"] = "not-signalled"
            action["error"] = current_error
            receipt["errors"].append(current_error)
        elif not _same_linux_process_identity(current, identity):
            action["status"] = "already-gone-or-reused"
        else:
            try:
                os.killpg(int(identity["pgid"]), signal.SIGTERM)
                action["status"] = "signalled"
            except ProcessLookupError:
                action["status"] = "already-gone-or-reused"
            except OSError as error:
                action["status"] = "not-signalled"
                action["error"] = f"{type(error).__name__}: {error}"
                receipt["errors"].append(str(action["error"]))
        receipt["actions"].append(action)
    return receipt


def _remaining_recorded_descendants(
    receipt: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    snapshot = receipt.get("snapshot")
    if not isinstance(snapshot, dict):
        return [], ["Linux descendant snapshot is missing"]
    proc_root_raw = snapshot.get("proc_root")
    if not isinstance(proc_root_raw, str) or not proc_root_raw:
        return [], ["Linux descendant snapshot lacks proc_root"]
    proc_root = Path(proc_root_raw)
    remaining: list[dict[str, Any]] = []
    errors: list[str] = []
    processes = snapshot.get("processes")
    if not isinstance(processes, list):
        return [], ["Linux descendant snapshot lacks process records"]
    for expected in processes:
        if not isinstance(expected, dict) or not isinstance(expected.get("pid"), int):
            errors.append("Linux descendant snapshot has an invalid process record")
            continue
        current, current_error = _read_linux_process_identity(proc_root, int(expected["pid"]))
        if current_error is not None:
            errors.append(current_error)
        elif _same_linux_process_identity(current, expected):
            remaining.append(expected)
    return remaining, errors


def _finalize_recorded_descendant_cleanup(
    receipt: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Require the snapshot identities to disappear after N01 stops its child."""
    if receipt.get("snapshot") is None:
        receipt["status"] = "unproven"
        receipt["cleanup_verified"] = False
        return receipt
    grace_seconds = min(MAX_N06A_DESCENDANT_CLEANUP_SECONDS, timeout_seconds)
    deadline = time.monotonic() + grace_seconds
    remaining: list[dict[str, Any]] = []
    while True:
        remaining, errors = _remaining_recorded_descendants(receipt)
        if errors:
            receipt["errors"].extend(error for error in errors if error not in receipt["errors"])
            break
        if not remaining:
            receipt["status"] = "cleaned"
            receipt["cleanup_verified"] = not receipt["errors"]
            return receipt
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)

    # A group is escalated only when its original, recorded leader identity is
    # still live.  Never derive a new PGID from a recycled PID.
    candidate_groups = receipt.get("candidate_groups", [])
    snapshot = receipt.get("snapshot")
    proc_root = Path(snapshot["proc_root"]) if isinstance(snapshot, dict) else Path("/proc")
    expected_by_pid = {
        int(item["pid"]): item
        for item in (snapshot.get("processes", []) if isinstance(snapshot, dict) else [])
        if isinstance(item, dict) and isinstance(item.get("pid"), int)
    }
    for candidate in candidate_groups if isinstance(candidate_groups, list) else []:
        if not isinstance(candidate, dict):
            continue
        leader_pid = candidate.get("leader_pid")
        if not isinstance(leader_pid, int):
            continue
        expected = expected_by_pid.get(leader_pid)
        if expected is None:
            continue
        current, current_error = _read_linux_process_identity(proc_root, leader_pid)
        action: dict[str, Any] = {
            "signal": "SIGKILL",
            "pgid": candidate.get("pgid"),
            "leader_pid": leader_pid,
            "leader_starttime_ticks": candidate.get("leader_starttime_ticks"),
        }
        if current_error is not None:
            action["status"] = "not-signalled"
            action["error"] = current_error
            receipt["errors"].append(current_error)
        elif not _same_linux_process_identity(current, expected):
            action["status"] = "already-gone-or-reused"
        else:
            try:
                os.killpg(int(candidate["pgid"]), signal.SIGKILL)
                action["status"] = "signalled"
            except ProcessLookupError:
                action["status"] = "already-gone-or-reused"
            except OSError as error:
                action["status"] = "not-signalled"
                action["error"] = f"{type(error).__name__}: {error}"
                receipt["errors"].append(str(action["error"]))
        receipt["actions"].append(action)

    kill_deadline = time.monotonic() + min(1.0, grace_seconds)
    while True:
        remaining, errors = _remaining_recorded_descendants(receipt)
        if errors:
            receipt["errors"].extend(error for error in errors if error not in receipt["errors"])
            break
        if not remaining:
            receipt["status"] = "cleaned"
            receipt["cleanup_verified"] = not receipt["errors"]
            return receipt
        if time.monotonic() >= kill_deadline:
            break
        time.sleep(0.02)
    receipt["remaining_processes"] = [
        {
            "pid": item["pid"],
            "pgid": item["pgid"],
            "starttime_ticks": item["starttime_ticks"],
        }
        for item in remaining
    ]
    receipt["errors"].append("recorded N06-A descendant process identities remain after cleanup")
    receipt["status"] = "unproven"
    receipt["cleanup_verified"] = False
    return receipt


def _stop_process_with_receipt(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    """Stop N01's own process group and retain the bounded outcome."""
    receipt: dict[str, Any] = {
        "pid": process.pid,
        "actions": [],
        "errors": [],
        "cleanup_verified": False,
    }
    if process.poll() is not None:
        receipt["actions"].append("already-exited")
        receipt["returncode"] = process.returncode
        receipt["cleanup_verified"] = True
        return receipt
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
            receipt["actions"].append("process-group-sigterm")
        else:
            process.terminate()
            receipt["actions"].append("process-terminate")
    except ProcessLookupError:
        receipt["actions"].append("already-gone-before-term")
        receipt["returncode"] = process.poll()
        receipt["cleanup_verified"] = process.poll() is not None
        return receipt
    except OSError as error:
        receipt["errors"].append(f"term: {type(error).__name__}: {error}")
        receipt["returncode"] = process.poll()
        return receipt
    try:
        process.wait(timeout=5)
        receipt["actions"].append("wait-after-term")
        receipt["returncode"] = process.returncode
        receipt["cleanup_verified"] = True
        return receipt
    except subprocess.TimeoutExpired:
        receipt["actions"].append("term-wait-timeout")
    except Exception as error:
        receipt["errors"].append(f"term-wait: {type(error).__name__}: {error}")
        receipt["returncode"] = process.poll()
        return receipt
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
            receipt["actions"].append("process-group-sigkill")
        else:
            process.kill()
            receipt["actions"].append("process-kill")
    except ProcessLookupError:
        receipt["actions"].append("already-gone-before-kill")
        receipt["returncode"] = process.poll()
        receipt["cleanup_verified"] = process.poll() is not None
        return receipt
    except OSError as error:
        receipt["errors"].append(f"kill: {type(error).__name__}: {error}")
        receipt["returncode"] = process.poll()
        return receipt
    try:
        process.wait(timeout=5)
        receipt["actions"].append("wait-after-kill")
    except Exception as error:
        receipt["errors"].append(f"kill-wait: {type(error).__name__}: {error}")
    receipt["returncode"] = process.poll()
    receipt["cleanup_verified"] = process.poll() is not None and not receipt["errors"]
    return receipt


def _stop_process(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    """Backward-compatible process stopper used by older controller callers."""
    return _stop_process_with_receipt(process)


def _n06a_docker_cleanup_command(
    launcher: str,
    arguments: Sequence[str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one bounded Docker control-plane argv without a shell."""
    argv = [launcher, *arguments]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )
    except Exception as error:
        return {
            "argv": argv,
            "timeout_seconds": timeout_seconds,
            "returncode": None,
            "stdout_bytes": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_bytes": 0,
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "error": f"{type(error).__name__}: {error}",
        }
    stdout = completed.stdout
    stderr = completed.stderr
    command: dict[str, Any] = {
        "argv": argv,
        "timeout_seconds": timeout_seconds,
        "returncode": completed.returncode,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    if (
        len(stdout) > MAX_N06A_DOCKER_CLEANUP_OUTPUT_BYTES
        or len(stderr) > MAX_N06A_DOCKER_CLEANUP_OUTPUT_BYTES
    ):
        command["error"] = (
            "N06-A Docker cleanup command output exceeds "
            f"{MAX_N06A_DOCKER_CLEANUP_OUTPUT_BYTES} bytes"
        )
        return command
    try:
        command["stdout_text"] = stdout.decode("utf-8")
        command["stderr_text"] = stderr.decode("utf-8")
    except UnicodeDecodeError as error:
        command["error"] = f"N06-A Docker cleanup command output is not UTF-8: {error}"
    return command


def _n06a_docker_container_inventory(
    cleanup: N06ATimeoutCleanupConfig,
    container_name: str,
) -> tuple[dict[str, Any], bool | None, str | None]:
    """Ask the daemon for exactly one owned name, never a broad container set."""
    command = _n06a_docker_cleanup_command(
        cleanup.docker_launcher,
        (
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^/{container_name}$",
            "--format",
            "{{.Names}}\\t{{.ID}}",
        ),
        timeout_seconds=cleanup.command_timeout_seconds,
    )
    if command.get("error") is not None:
        return command, None, str(command["error"])
    if command.get("returncode") != 0:
        return command, None, "N06-A Docker exact-name inventory exited non-zero"
    text = command.get("stdout_text")
    if not isinstance(text, str):
        return command, None, "N06-A Docker exact-name inventory lacks UTF-8 stdout"
    lines = [line for line in text.splitlines() if line]
    if not lines:
        return command, False, None
    if len(lines) != 1:
        return command, None, "N06-A Docker exact-name inventory returned multiple rows"
    name, separator, container_id = lines[0].partition("\t")
    if (
        separator != "\t"
        or name != container_name
        or not re.fullmatch(r"[0-9a-f]{12,64}", container_id)
    ):
        return command, None, "N06-A Docker exact-name inventory returned an unexpected row"
    return command, True, None


def _cleanup_n06a_owned_container(
    cleanup: N06ATimeoutCleanupConfig,
    container_name: str,
) -> dict[str, Any]:
    """Stop only the derived N06-A vLLM container and prove daemon absence.

    A successful outer child exit only proves the Python driver process ended.
    The final exact-name inventory is deliberately daemon-backed, and is paired
    with a non-successful inspect so an unavailable daemon cannot be confused
    with an absent container.
    """
    receipt: dict[str, Any] = {
        "schema_version": N06A_DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION,
        "launcher": cleanup.docker_launcher,
        "container_name": container_name,
        "commands": [],
        "notes": [],
        "errors": [],
        "cleanup_verified": False,
    }
    if not N06A_DOCKER_CONTAINER_NAME_RE.fullmatch(container_name):
        receipt["errors"].append("derived N06-A vLLM Docker name is invalid")
        return receipt
    before, present, inventory_error = _n06a_docker_container_inventory(cleanup, container_name)
    receipt["commands"].append({"operation": "inventory-before", **before})
    if inventory_error is not None:
        receipt["errors"].append(inventory_error)
        return receipt
    receipt["initial_state"] = "present" if present else "absent"
    if present:
        stop_seconds = max(
            1,
            math.ceil(
                min(cleanup.command_timeout_seconds, MAX_N06A_DOCKER_CLEANUP_COMMAND_SECONDS)
            ),
        )
        stop = _n06a_docker_cleanup_command(
            cleanup.docker_launcher,
            ("container", "stop", "--time", str(stop_seconds), container_name),
            timeout_seconds=cleanup.command_timeout_seconds,
        )
        receipt["commands"].append({"operation": "stop", **stop})
        if stop.get("error") is not None:
            receipt["errors"].append(str(stop["error"]))
            return receipt
        if stop.get("returncode") != 0:
            receipt["notes"].append("Docker stop returned non-zero; final absence remains required")
        wait = _n06a_docker_cleanup_command(
            cleanup.docker_launcher,
            ("container", "wait", container_name),
            timeout_seconds=cleanup.command_timeout_seconds,
        )
        receipt["commands"].append({"operation": "wait", **wait})
        if wait.get("error") is not None:
            receipt["errors"].append(str(wait["error"]))
            return receipt
        if wait.get("returncode") != 0:
            receipt["notes"].append(
                "Docker wait returned non-zero after stop; final absence remains required"
            )

    after_stop, still_present, inventory_error = _n06a_docker_container_inventory(
        cleanup, container_name
    )
    receipt["commands"].append({"operation": "inventory-after-stop", **after_stop})
    if inventory_error is not None:
        receipt["errors"].append(inventory_error)
        return receipt
    if still_present:
        remove = _n06a_docker_cleanup_command(
            cleanup.docker_launcher,
            ("container", "rm", "--force", container_name),
            timeout_seconds=cleanup.command_timeout_seconds,
        )
        receipt["commands"].append({"operation": "force-remove", **remove})
        if remove.get("error") is not None or remove.get("returncode") != 0:
            receipt["errors"].append(
                str(remove.get("error") or "N06-A Docker force-remove exited non-zero")
            )
            return receipt
        final_inventory, final_present, inventory_error = _n06a_docker_container_inventory(
            cleanup, container_name
        )
        receipt["commands"].append(
            {"operation": "inventory-after-force-remove", **final_inventory}
        )
    else:
        final_inventory, final_present, inventory_error = after_stop, still_present, None
    if inventory_error is not None or final_present is not False:
        receipt["errors"].append(
            inventory_error or "N06-A Docker container remains present after cleanup"
        )
        return receipt
    inspect = _n06a_docker_cleanup_command(
        cleanup.docker_launcher,
        ("container", "inspect", "--format", "{{.Id}}", container_name),
        timeout_seconds=cleanup.command_timeout_seconds,
    )
    receipt["commands"].append({"operation": "inspect-absence", **inspect})
    if inspect.get("error") is not None:
        receipt["errors"].append(str(inspect["error"]))
        return receipt
    if inspect.get("returncode") == 0:
        receipt["errors"].append(
            "N06-A Docker inspect still resolves the owned container after absence inventory"
        )
        return receipt
    receipt["final_state"] = "absent"
    receipt["cleanup_verified"] = True
    return receipt


def _append_attempt_error(existing: str | None, addition: str) -> str:
    return addition if existing is None else f"{existing}; {addition}"


def _post_attempt_n06a_cleanup(
    config: RepeatControlConfig,
    *,
    kind: str,
    index: int,
    timed_out: bool,
    parent_termination_requested: bool,
    timeout_descendants: dict[str, Any] | None,
    outer_process_cleanup: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Write one immutable parent cleanup receipt after every opt-in attempt."""
    cleanup = config.n06a_timeout_cleanup
    if cleanup is None:
        return None
    attempt_dir, container_name = n06a_vllm_container_name(cleanup, kind=kind, index=index)
    receipt: dict[str, Any] = {
        "schema_version": N06A_PARENT_CLEANUP_SCHEMA_VERSION,
        "kind": kind,
        "index": index,
        "attempt_artifact_directory": str(attempt_dir),
        "container_name": container_name,
        "timed_out": timed_out,
        "parent_termination_requested": parent_termination_requested,
        "timeout_descendant_cleanup": timeout_descendants
        if timeout_descendants is not None
        else {
            "status": "not-required",
            "cleanup_verified": True,
            "reason": "outer attempt did not time out",
        },
        "outer_process_cleanup": outer_process_cleanup
        if outer_process_cleanup is not None
        else {
            "status": "not-required",
            "cleanup_verified": True,
            "reason": "outer attempt did not require parent termination",
        },
        "docker_container_cleanup": None,
        "errors": [],
        "cleanup_verified": False,
    }
    if parent_termination_requested and timeout_descendants is not None:
        receipt["timeout_descendant_cleanup"] = _finalize_recorded_descendant_cleanup(
            timeout_descendants,
            timeout_seconds=cleanup.command_timeout_seconds,
        )
    if parent_termination_requested and timeout_descendants is None:
        receipt["errors"].append("parent-terminated N06-A attempt lacks a descendant cleanup receipt")
    if parent_termination_requested and outer_process_cleanup is None:
        receipt["errors"].append("parent-terminated N06-A attempt lacks an outer process cleanup receipt")
    if parent_termination_requested and not bool(receipt["timeout_descendant_cleanup"].get("cleanup_verified")):
        receipt["errors"].append("parent-terminated N06-A descendant cleanup is unproven")
    if parent_termination_requested and not bool(receipt["outer_process_cleanup"].get("cleanup_verified")):
        receipt["errors"].append("parent-terminated N06-A outer process cleanup is unproven")
    try:
        docker_cleanup = _cleanup_n06a_owned_container(cleanup, container_name)
    except Exception as error:
        # Preserve a structurally recognizable, exact-name failed receipt so
        # the offline N03b reader can retain this failed-cleanup attempt
        # without accepting arbitrary Docker control-plane provenance.
        docker_cleanup = {
            "schema_version": N06A_DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION,
            "launcher": cleanup.docker_launcher,
            "container_name": container_name,
            "commands": [],
            "notes": [],
            "errors": [f"{type(error).__name__}: {error}"],
            "cleanup_verified": False,
        }
    receipt["docker_container_cleanup"] = docker_cleanup
    if not bool(docker_cleanup.get("cleanup_verified")):
        receipt["errors"].append("N06-A Docker daemon absence is unproven")
    receipt["cleanup_verified"] = not receipt["errors"]
    receipt_path = _n06a_cleanup_receipt_path(config.output_dir, kind, index)
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        write_immutable_json(receipt_path, receipt)
        receipt["receipt_path"] = str(receipt_path)
        receipt["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    except Exception as error:
        receipt["errors"].append(f"cannot retain N06-A parent cleanup receipt: {type(error).__name__}: {error}")
        receipt["cleanup_verified"] = False
        receipt["receipt_path"] = str(receipt_path)
        receipt["receipt_sha256"] = None
    return receipt


def _log_paths(output_dir: Path, kind: str, index: int) -> tuple[Path, Path]:
    prefix = f"{kind}-{index:03d}"
    return output_dir / f"{prefix}.stdout.log", output_dir / f"{prefix}.stderr.log"


def _planned_paths(config: RepeatControlConfig) -> list[Path]:
    paths = [config.output_dir / config.receipt_name]
    for kind, count in (("warmup", config.warmups), ("timed", config.repeats)):
        for index in range(1, count + 1):
            paths.extend(_log_paths(config.output_dir, kind, index))
            if config.n06a_timeout_cleanup is not None:
                paths.append(_n06a_cleanup_receipt_path(config.output_dir, kind, index))
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
    parent_termination_requested = False
    timeout_descendants: dict[str, Any] | None = None
    outer_process_cleanup: dict[str, Any] | None = None
    n06a_parent_cleanup: dict[str, Any] | None = None
    status = "failed-to-start"
    error: str | None = None
    try:
        with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
            # The child alone receives its outer attempt identity.  This lets a
            # paired benchmark alternate timed lane order from the immutable N01
            # receipt index without mutating the controller's own environment.
            child_environment = os.environ.copy()
            child_environment["N01_REPEAT_CONTROL_PHASE"] = kind
            child_environment["N01_REPEAT_CONTROL_INDEX"] = str(index)
            process = subprocess.Popen(
                list(config.command),
                cwd=str(config.cwd),
                env=child_environment,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                shell=False,
            )
            try:
                process.wait(timeout=config.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                parent_termination_requested = True
                error = f"command exceeded {config.timeout_seconds:g} seconds"
                if config.n06a_timeout_cleanup is not None:
                    timeout_descendants = _signal_recorded_descendant_groups(
                        process,
                        proc_root=config.proc_root,
                    )
                outer_process_cleanup = _stop_process(process)
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
            parent_termination_requested = True
            if config.n06a_timeout_cleanup is not None:
                timeout_descendants = _signal_recorded_descendant_groups(
                    process,
                    proc_root=config.proc_root,
                )
            outer_process_cleanup = _stop_process(process)
            exit_code = process.returncode
    except BaseException as exception:
        status = "failed-to-start" if process is None else "failed"
        error = f"{type(exception).__name__}: {exception}"
        if process is not None:
            parent_termination_requested = True
            if config.n06a_timeout_cleanup is not None:
                timeout_descendants = _signal_recorded_descendant_groups(
                    process,
                    proc_root=config.proc_root,
                )
            outer_process_cleanup = _stop_process(process)
            exit_code = process.returncode
    finally:
        if config.n06a_timeout_cleanup is not None:
            try:
                n06a_parent_cleanup = _post_attempt_n06a_cleanup(
                    config,
                    kind=kind,
                    index=index,
                    timed_out=timed_out,
                    parent_termination_requested=parent_termination_requested,
                    timeout_descendants=timeout_descendants
                    if parent_termination_requested
                    else None,
                    outer_process_cleanup=outer_process_cleanup
                    if parent_termination_requested
                    else None,
                )
            except Exception as cleanup_error:
                n06a_parent_cleanup = {
                    "schema_version": N06A_PARENT_CLEANUP_SCHEMA_VERSION,
                    "kind": kind,
                    "index": index,
                    "cleanup_verified": False,
                    "errors": [
                        (
                            "N06-A parent cleanup controller exception: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    ],
                }
            if n06a_parent_cleanup is None or not n06a_parent_cleanup.get("cleanup_verified"):
                error = _append_attempt_error(error, "N06-A parent cleanup is unproven")
                # A failed child can leave the same owned Riley/Docker residue
                # as a successful child.  Make the cleanup failure explicit so
                # run_repeat_control can fail-stop before it launches another
                # warmup or timed lane on a potentially contaminated GPU.
                if status != "interrupted":
                    status = "failed-cleanup"
        finished_ns = time.monotonic_ns()
        finished_at = utc_now()
        post = environment_snapshot(config)
    elapsed_ns = finished_ns - started_ns
    result = {
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
    if n06a_parent_cleanup is not None:
        result["n06a_parent_cleanup"] = n06a_parent_cleanup
    return result



def _not_started_after_failed_cleanup(
    *,
    kind: str,
    index: int,
    blocking_kind: str,
    blocking_index: int,
) -> dict[str, Any]:
    """Represent a planned attempt deliberately never launched after cleanup failed.

    This has no stdout/stderr, PSI, or cleanup sidecar because N01 has not
    created a child process or an owned Docker name for it.  Its compact,
    exact fields make the fail-stop visible to the offline reader without
    manufacturing evidence for work that never happened.
    """
    if kind not in {"warmup", "timed"} or blocking_kind not in {"warmup", "timed"}:
        raise ControlError("N06-A fail-stop attempt kind is invalid")
    _positive_integer(index, "N06-A fail-stop index")
    _positive_integer(blocking_index, "N06-A fail-stop blocking index")
    return {
        "kind": kind,
        "index": index,
        "status": N06A_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS,
        "blocking_kind": blocking_kind,
        "blocking_index": blocking_index,
        "reason": N06A_NOT_STARTED_AFTER_FAILED_CLEANUP_REASON,
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
    fail_stop: tuple[str, int] | None = None
    for index in range(1, config.warmups + 1):
        if fail_stop is not None:
            warmup_runs.append(
                _not_started_after_failed_cleanup(
                    kind="warmup",
                    index=index,
                    blocking_kind=fail_stop[0],
                    blocking_index=fail_stop[1],
                )
            )
            continue
        run = execute_attempt(config, kind="warmup", index=index)
        warmup_runs.append(run)
        if run["status"] == "interrupted":
            interrupted = True
            break
        if run["status"] == "failed-cleanup":
            fail_stop = ("warmup", index)
    if not interrupted:
        for index in range(1, config.repeats + 1):
            if fail_stop is not None:
                timed_runs.append(
                    _not_started_after_failed_cleanup(
                        kind="timed",
                        index=index,
                        blocking_kind=fail_stop[0],
                        blocking_index=fail_stop[1],
                    )
                )
                continue
            run = execute_attempt(config, kind="timed", index=index)
            timed_runs.append(run)
            if run["status"] == "interrupted":
                interrupted = True
                break
            if run["status"] == "failed-cleanup":
                fail_stop = ("timed", index)

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
            "n06a_timeout_cleanup": None
            if config.n06a_timeout_cleanup is None
            else {
                "artifact_root": str(config.n06a_timeout_cleanup.artifact_root),
                "docker_launcher": config.n06a_timeout_cleanup.docker_launcher,
                "command_timeout_seconds": config.n06a_timeout_cleanup.command_timeout_seconds,
                "scope": "one exact N06-A vLLM name per outer attempt",
            },
        },
        "retention_policy": {
            "timed_runs": "all configured timed attempts are retained, including non-zero exits and timeouts",
            "failed_cleanup_fail_stop": (
                "after failed-cleanup, remaining planned attempts are recorded as "
                "not-started-after-failed-cleanup and no later child is launched"
            ),
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
    parser.add_argument(
        "--n06a-timeout-cleanup-artifact-root",
        type=Path,
        help=(
            "Explicit N06-A artifact root used only to derive the one owned vLLM "
            "container name for parent cleanup"
        ),
    )
    parser.add_argument(
        "--n06a-timeout-cleanup-docker-launcher",
        help=(
            "Explicit Docker argv[0] used only with --n06a-timeout-cleanup-artifact-root"
        ),
    )
    parser.add_argument(
        "--n06a-timeout-cleanup-command-timeout-seconds",
        type=float,
        default=MAX_N06A_DOCKER_CLEANUP_COMMAND_SECONDS,
        help=(
            "Bound for each opt-in N06-A Docker cleanup command (maximum "
            f"{MAX_N06A_DOCKER_CLEANUP_COMMAND_SECONDS:g} seconds)"
        ),
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
    if (namespace.n06a_timeout_cleanup_artifact_root is None) != (
        namespace.n06a_timeout_cleanup_docker_launcher is None
    ):
        raise ControlError(
            "--n06a-timeout-cleanup-artifact-root and "
            "--n06a-timeout-cleanup-docker-launcher must be supplied together"
        )
    n06a_timeout_cleanup = (
        None
        if namespace.n06a_timeout_cleanup_artifact_root is None
        else N06ATimeoutCleanupConfig(
            artifact_root=namespace.n06a_timeout_cleanup_artifact_root,
            docker_launcher=namespace.n06a_timeout_cleanup_docker_launcher,
            command_timeout_seconds=namespace.n06a_timeout_cleanup_command_timeout_seconds,
        )
    )
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
        n06a_timeout_cleanup=n06a_timeout_cleanup,
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
