#!/usr/bin/env python3
"""Summarize repeated N03b/N06-A D128 operator and serving evidence.

This offline reader consumes immutable ``n01_repeat_control`` receipts.  It
never launches a model, server, benchmark client, or Blender process.  Every
configured timed attempt remains in the resulting document, including failed
and timed-out attempts and their CPU/I/O/memory PSI observations.  A failed
attempt cannot contribute a parsed performance observation, but it is never
discarded because the host was busy.

``--operator-receipt`` reads the existing N03a V2 prepared-paged-decode CUDA
control through its strict reader.  That evidence remains explicitly an
operator control.  ``--serving-receipt`` reads a paired full-model serving
command: every successful outer command must emit one
``riley-n06a-d128-serving`` marker for Riley and one for vLLM.  The Riley
marker must point to a fresh, hashed snapshot of Riley's actual startup log,
which proves the D128 ragged-paged implementation was requested and resolved
with no fallback.  The two scopes deliberately remain separate in the output.

The serving marker is a compact, one-line, whitespace-separated key=value
record.  Its exact contract is documented in ``benchmarks/scripts/README.md``
so the Rust-only serving bridge can emit it without introducing Python into
Riley's runtime path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import stat
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import n03a_paged_d128_repeat_summary as n03a
import serving_token_client_v2 as token_client


REPEAT_RECEIPT_SCHEMA_VERSION = "riley.n01-repeat-control-receipt.v1"
SUMMARY_SCHEMA_VERSION = "riley.n03b-n06a-d128-repeat-summary.v1"
SERVING_MARKER_PREFIX = "riley-n06a-d128-serving"
STARTUP_RECEIPT_PREFIX = "RILEY_DECODE_ATTENTION"
REQUESTED_BACKEND_CLI_ID = "native-bf16-paged-split-gqa-d128-two-stage"
ATTEMPT_ARTIFACT_SCHEMA_VERSION = "riley.n06a-paired-serving-attempt.v1"
RETAINED_PHASE_SCHEMA_VERSION = "riley.n06a-streaming-phase.v1"
WORKLOAD_SCHEMA_VERSION = "riley.n06a-d128-serving-workload.v1"
MODEL_IDENTITY_MANIFEST_SCHEMA_VERSION = "riley.n06a-model-identity-manifest.v1"
MODEL_IDENTITY_VALIDATION_SCHEMA_VERSION = "riley.n06a-model-identity-validation.v1"
VLLM_PREFIX_CACHING_DISABLED = "disabled"
VLLM_GPU_MEMORY_UTILIZATION = "0.65"
WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES = 19_000_000_000
VLLM_AUTO_BACKEND_REQUESTED = "vllm-auto"
VLLM_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
DOCKER_CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION = "riley.n06a-docker-container-cleanup.v1"
N01_N06A_PARENT_CLEANUP_SCHEMA_VERSION = "riley.n01-n06a-parent-cleanup.v1"
N01_N06A_DOCKER_CLEANUP_SCHEMA_VERSION = "riley.n01-n06a-docker-container-cleanup.v1"
N01_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS = "not-started-after-failed-cleanup"
N01_NOT_STARTED_AFTER_FAILED_CLEANUP_REASON = (
    "N06-A parent cleanup was unproven; fail-stop prevents further launches"
)

MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_STDOUT_LOG_BYTES = 16 * 1024 * 1024
MAX_STARTUP_SNAPSHOT_BYTES = 8 * 1024 * 1024
MAX_RETAINED_REQUEST_ROWS_BYTES = 128 * 1024 * 1024
MAX_RETAINED_PHASE_BYTES = 4 * 1024 * 1024
MAX_ATTEMPT_CONFIG_BYTES = 4 * 1024 * 1024
MAX_GPU_MEMORY_SAMPLES_BYTES = 4 * 1024 * 1024
MAX_LANE_PROVENANCE_BYTES = 4 * 1024 * 1024
MAX_LANE_PSI_ARTIFACT_BYTES = 1 * 1024 * 1024
MAX_WORKLOAD_BYTES = 4 * 1024 * 1024
MAX_MODEL_IDENTITY_MANIFEST_BYTES = 1 * 1024 * 1024
# Must match N06-A: the pinned Qwen2.5-3B tokenizer.json is about 7 MiB.
# Summary rehashes only bounded metadata files, never checkpoint shards.
MAX_MODEL_IDENTITY_METADATA_BYTES = 8 * 1024 * 1024
MAX_MODEL_IDENTITY_GIT_OUTPUT_BYTES = 1 * 1024 * 1024
MODEL_IDENTITY_GIT_TIMEOUT_SECONDS = 30.0
MAX_RETAINED_REQUEST_ROWS = 10_000
MAX_TIMED_RUNS = 1_000
MAX_NATIVE_D128_BATCH_TOKEN_BUDGET = 32
MAX_NATIVE_D128_MODEL_LENGTH = 32_768
P99_QUALIFIED_MINIMUM_REQUESTS = 1_000
MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS = 0.5
GPU_MEMORY_SAMPLE_SCHEDULING_SLACK_SECONDS = 0.5
GPU_IDLE_CENSUS_SCHEMA_VERSION = "riley.n06a-post-lane-gpu-idle-census.v1"
GPU_IDLE_CENSUS_MAX_USED_BYTES = 512 * 1024 * 1024
GPU_IDLE_CENSUS_COMMAND_TIMEOUT_SECONDS = 5.0
VLLM_BACKEND_RECEIPT_REGEX = re.compile(
    r"Using (?P<backend_resolved>[A-Z0-9_]+) attention backend out of potential backends:"
)
# The reader calculates CIs for every lane and paired metric.  Keep a malformed
# or accidental receipt from turning an offline report into prolonged CPU load
# on the shared serving host.  The N01 default is 10,000 resamples and five
# timed runs, so its normal 50,000 draws stay inside this envelope.
MAX_BOOTSTRAP_RESAMPLES = 100_000
MAX_BOOTSTRAP_DRAWS_PER_METRIC = 250_000
PSI_RESOURCES = ("cpu", "io", "memory")
LANE_PSI_SCHEMA_VERSION = "riley.n06a-lane-psi.v1"
LANE_PSI_POLICY = "observed-only; never used for lane eligibility, sample selection, or performance adjustment"

QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
QWEN_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
OPERATOR_IMPLEMENTATION_ID = (
    "riley.cuda.native-bf16-paged-split-gqa.qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2"
)
SERVING_IMPLEMENTATION_ID = (
    "riley.cuda.ragged-paged-attention.native-bf16-paged-split-gqa."
    "qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SummaryError(ValueError):
    """A receipt, marker, or resolved-backend receipt is malformed."""


@dataclass(frozen=True)
class RepeatReceipt:
    """One validated immutable repeat-control receipt and its timed attempts."""

    path: Path
    sha256: str
    output_dir: Path
    timed_repeats: int
    bootstrap_resamples: int
    bootstrap_seed: int
    warmup_runs: tuple[Mapping[str, Any], ...]
    timed_runs: tuple[Mapping[str, Any], ...]
    n06a_timeout_cleanup: Mapping[str, Any] | None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SummaryError(f"JSON object has duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SummaryError(f"JSON has non-finite constant {value!r}")


def _read_bounded_regular_file(
    path: Path, *, label: str, maximum_bytes: int
) -> tuple[Path, bytes, str]:
    """Read one stable regular file without accepting a final symlink."""
    try:
        before = path.lstat()
    except OSError as error:
        raise SummaryError(f"{label} cannot be inspected: {path}: {error}") from error
    if stat.S_ISLNK(before.st_mode):
        raise SummaryError(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise SummaryError(f"{label} must be a regular file: {path}")
    if before.st_size > maximum_bytes:
        raise SummaryError(
            f"{label} exceeds bounded input limit of {maximum_bytes} bytes: {path}"
        )
    try:
        payload = path.read_bytes()
        after = path.stat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SummaryError(f"{label} cannot be read: {path}: {error}") from error
    if not stat.S_ISREG(after.st_mode):
        raise SummaryError(f"{label} changed type while being read: {path}")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SummaryError(f"{label} changed while being read: {path}")
    if len(payload) != before.st_size:
        raise SummaryError(f"{label} was truncated while being read: {path}")
    return resolved, payload, hashlib.sha256(payload).hexdigest()


def _decode_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise SummaryError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise SummaryError(f"{label} is malformed JSON: {error}") from error
    return _require_mapping(decoded, label)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"{label} must be an object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SummaryError(f"{label} must be a nonempty string")
    return value


def _require_integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SummaryError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise SummaryError(f"{label} must be >= {minimum}")
    return value


def _require_finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SummaryError(f"{label} must be a finite number")
    if positive and parsed <= 0:
        raise SummaryError(f"{label} must be positive")
    return parsed


def _parse_record_integer(value: str, label: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise SummaryError(f"{label} must be a base-10 integer") from error
    if parsed < minimum:
        raise SummaryError(f"{label} must be >= {minimum}")
    return parsed


def _parse_record_float(value: str, label: str, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise SummaryError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise SummaryError(f"{label} must be a finite number")
    if positive and parsed <= 0:
        raise SummaryError(f"{label} must be positive")
    return parsed


def _require_exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, label)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise SummaryError(f"{label} has " + "; ".join(detail))
    return mapping


def _parse_marker_tokens(line: str, *, prefix: str, label: str) -> dict[str, str]:
    if not line.startswith(prefix + " "):
        raise SummaryError(f"{label} must begin with {prefix!r}")
    fields: dict[str, str] = {}
    for token in line[len(prefix) + 1 :].split():
        if "=" not in token:
            raise SummaryError(f"{label} has malformed token {token!r}")
        key, value = token.split("=", 1)
        if not key or not value:
            raise SummaryError(f"{label} has malformed token {token!r}")
        if key in fields:
            raise SummaryError(f"{label} has duplicate field {key!r}")
        fields[key] = value
    return fields


def _r7_quantile(values: Iterable[float], probability: float) -> float:
    observations = sorted(float(value) for value in values)
    if not observations:
        raise SummaryError("quantile requires at least one observation")
    if not 0.0 <= probability <= 1.0:
        raise SummaryError("quantile probability must be in [0, 1]")
    if any(not math.isfinite(value) for value in observations):
        raise SummaryError("quantile observations must be finite")
    if len(observations) == 1:
        return observations[0]
    position = (len(observations) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    if fraction == 0:
        return observations[lower]
    return observations[lower] + fraction * (observations[lower + 1] - observations[lower])


def _bootstrap_median_95_ci(
    values: Sequence[float], *, resamples: int, seed: int
) -> dict[str, Any]:
    if not values:
        raise SummaryError("bootstrap requires at least one observation")
    source = random.Random(seed)
    count = len(values)
    medians = [
        statistics.median(values[source.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    return {
        "confidence_level": 0.95,
        "method": "percentile-bootstrap-r7",
        "resamples": resamples,
        "seed": seed,
        "lower": _r7_quantile(medians, 0.025),
        "upper": _r7_quantile(medians, 0.975),
    }


def _summarize_values(
    values: Sequence[float], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, Any]:
    observations = [float(value) for value in values]
    if not observations or any(not math.isfinite(value) for value in observations):
        raise SummaryError("statistics require finite observations")
    return {
        "count": len(observations),
        "median": statistics.median(observations),
        "mean": statistics.fmean(observations),
        "sample_stddev": statistics.stdev(observations) if len(observations) > 1 else 0.0,
        "sample_stddev_method": "sample standard deviation; 0.0 for one observation",
        "min": min(observations),
        "max": max(observations),
        "deterministic_bootstrap_median_95_ci": _bootstrap_median_95_ci(
            observations,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    }


def _parse_psi_metrics(value: object, label: str) -> dict[str, float | int] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, label)
    result: dict[str, float | int] = {}
    for key in ("avg10", "avg60", "avg300"):
        parsed = _require_finite_number(mapping.get(key), f"{label}.{key}")
        if parsed < 0:
            raise SummaryError(f"{label}.{key} must be nonnegative")
        result[key] = parsed
    result["total"] = _require_integer(mapping.get("total"), f"{label}.total", minimum=0)
    return result


def _validate_psi_resource(
    value: object, *, label: str, expected_source: str | None = None
) -> dict[str, Any]:
    item = _require_mapping(value, label)
    status = _require_string(item.get("status"), f"{label}.status")
    if status not in {"ok", "unavailable", "malformed"}:
        raise SummaryError(f"{label}.status is unsupported: {status!r}")
    if expected_source is not None:
        if set(item) != {"status", "source", "some", "full", "error"}:
            raise SummaryError(f"{label} fields differ from the lane PSI contract")
        source = _require_string(item.get("source"), f"{label}.source")
        if source != expected_source:
            raise SummaryError(f"{label}.source differs from its configured procfs source")
    some = _parse_psi_metrics(item.get("some"), f"{label}.some")
    full = _parse_psi_metrics(item.get("full"), f"{label}.full")
    if status == "ok" and some is None:
        raise SummaryError(f"{label}.some is required when status is ok")
    if status != "ok" and (some is not None or full is not None):
        raise SummaryError(
            f"{label} must not contain numbers when status is {status!r}"
        )
    error = item.get("error")
    if error is not None and not isinstance(error, str):
        raise SummaryError(f"{label}.error must be a string or null")
    result = {"status": status, "some": some, "full": full, "error": error}
    if expected_source is not None:
        result["source"] = expected_source
    return result


def _extract_psi_resource(
    run: Mapping[str, Any], *, phase: str, resource: str, run_label: str
) -> dict[str, Any]:
    observation = _require_mapping(run.get(phase), f"{run_label}.{phase}")
    psi = _require_mapping(observation.get("psi"), f"{run_label}.{phase}.psi")
    return _validate_psi_resource(
        psi.get(resource),
        label=f"{run_label}.{phase}.psi.{resource}",
    )


def _timed_run_covariate(run: Mapping[str, Any]) -> dict[str, Any]:
    index = _require_integer(run.get("index"), "timed run.index", minimum=1)
    label = f"repeat receipt.timed_runs[{index}]"
    status = _require_string(run.get("status"), f"{label}.status")
    if status == N01_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS:
        # Validated structurally by _load_repeat_receipt.  It deliberately has
        # no process, log, PSI, timing, or cleanup evidence because launching
        # another child after cleanup failed would be unsafe.
        return {
            "timed_index": index,
            "status": status,
            "exit_code": None,
            "timed_out": False,
            "wall_time_ms": None,
            "error": run["reason"],
            "attempt_log_paths": None,
            "psi": None,
            "blocking": {
                "kind": run["blocking_kind"],
                "index": run["blocking_index"],
                "reason": run["reason"],
            },
        }
    return {
        "timed_index": index,
        "status": status,
        "exit_code": run.get("exit_code"),
        "timed_out": run.get("timed_out"),
        "wall_time_ms": _require_finite_number(
            run.get("wall_time_ms"), f"{label}.wall_time_ms", positive=True
        ),
        "error": run.get("error"),
        "attempt_log_paths": {
            "stdout_path": _require_string(run.get("stdout_path"), f"{label}.stdout_path"),
            "stderr_path": _require_string(run.get("stderr_path"), f"{label}.stderr_path"),
        },
        "psi": {
            phase: {
                resource: _extract_psi_resource(
                    run,
                    phase=phase,
                    resource=resource,
                    run_label=label,
                )
                for resource in PSI_RESOURCES
            }
            for phase in ("pre", "post")
        },
    }


def _parse_n06a_timeout_cleanup_configuration(value: object) -> Mapping[str, Any] | None:
    if value is None:
        return None
    cleanup = _require_exact_keys(
        value,
        {"artifact_root", "docker_launcher", "command_timeout_seconds", "scope"},
        "repeat receipt.configuration.n06a_timeout_cleanup",
    )
    root_text = _require_string(cleanup.get("artifact_root"), "N06-A timeout cleanup artifact_root")
    launcher = _require_string(cleanup.get("docker_launcher"), "N06-A timeout cleanup docker_launcher")
    if "\x00" in launcher:
        raise SummaryError("N06-A timeout cleanup Docker launcher must not contain NUL")
    root = Path(root_text)
    if not root.is_absolute():
        raise SummaryError("N06-A timeout cleanup artifact_root must be absolute")
    timeout = _require_finite_number(
        cleanup.get("command_timeout_seconds"),
        "N06-A timeout cleanup command_timeout_seconds",
        positive=True,
    )
    if timeout > 30.0:
        raise SummaryError("N06-A timeout cleanup command_timeout_seconds must not exceed 30")
    if cleanup.get("scope") != "one exact N06-A vLLM name per outer attempt":
        raise SummaryError("N06-A timeout cleanup scope differs")
    return {
        "artifact_root": str(root.resolve()),
        "docker_launcher": launcher,
        "command_timeout_seconds": timeout,
        "scope": cleanup["scope"],
    }


def _n06a_parent_container_name(cleanup: Mapping[str, Any], *, kind: str, index: int) -> tuple[Path, str]:
    root = Path(_require_string(cleanup.get("artifact_root"), "N06-A cleanup artifact_root"))
    attempt_dir = (root / f"{kind}-{index:03d}").resolve()
    name = f"riley-n06a-vllm-{kind}-{index:04d}-{hashlib.sha256(str(attempt_dir).encode('utf-8')).hexdigest()[:20]}"
    if not DOCKER_CONTAINER_NAME_RE.fullmatch(name):
        raise SummaryError("derived N06-A parent cleanup Docker container name is invalid")
    return attempt_dir, name


def _validate_n01_docker_cleanup_command(
    value: object,
    *,
    expected_argv: list[str],
    timeout_seconds: float,
    label: str,
) -> Mapping[str, Any]:
    command = _require_mapping(value, label)
    if command.get("argv") != expected_argv:
        raise SummaryError(f"{label}.argv differs from its exact owned-container command")
    if command.get("timeout_seconds") != timeout_seconds:
        raise SummaryError(f"{label}.timeout_seconds differs from N01 configuration")
    returncode = command.get("returncode")
    if returncode is not None:
        _require_integer(returncode, f"{label}.returncode")
    command_error = command.get("error")
    if command_error is not None and (not isinstance(command_error, str) or not command_error):
        raise SummaryError(f"{label}.error must be a nonempty string when present")
    for stream in ("stdout", "stderr"):
        text = command.get(f"{stream}_text")
        if not isinstance(text, str):
            # N01 retains an empty hashed stream even when subprocess execution
            # itself failed before it could obtain UTF-8 output.  It is an
            # incomplete cleanup receipt, never a success proof, but rejecting
            # it wholesale would hide the timed failed-cleanup attempt.
            if command_error is None:
                raise SummaryError(f"{label}.{stream}_text is required")
            raw = b""
        else:
            raw = text.encode("utf-8")
        if len(raw) > 64 * 1024 or (
            command.get(f"{stream}_bytes") != len(raw)
            or command.get(f"{stream}_sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise SummaryError(f"{label}.{stream} receipt differs")
    return command


def _validate_n01_docker_absence(
    value: object, *, cleanup: Mapping[str, Any], container_name: str, label: str
) -> Mapping[str, Any]:
    """Verify N01's parent-side daemon proof without allowing broad Docker argv."""
    receipt = _require_mapping(value, f"{label}.docker_container_cleanup")
    launcher = _require_string(cleanup.get("docker_launcher"), f"{label} configured Docker launcher")
    timeout = float(cleanup["command_timeout_seconds"])
    if (
        receipt.get("schema_version") != N01_N06A_DOCKER_CLEANUP_SCHEMA_VERSION
        or receipt.get("launcher") != launcher
        or receipt.get("container_name") != container_name
        or not isinstance(receipt.get("commands"), list)
        or not isinstance(receipt.get("notes"), list)
        or not isinstance(receipt.get("errors"), list)
    ):
        raise SummaryError(f"{label} N01 Docker cleanup contract differs")
    commands = receipt["commands"]
    if not commands:
        if receipt.get("cleanup_verified") is True or receipt.get("errors") == []:
            raise SummaryError(f"{label} N01 Docker cleanup lacks a retained failure reason")
        return {"cleanup_verified": False, "command_count": 0}
    command_by_operation: dict[str, Mapping[str, Any]] = {}
    allowed = {
        "inventory-before",
        "stop",
        "wait",
        "inventory-after-stop",
        "force-remove",
        "inventory-after-force-remove",
        "inspect-absence",
    }
    expected_inventory = [
        launcher,
        "container",
        "ls",
        "--all",
        "--filter",
        f"name=^/{container_name}$",
        "--format",
        "{{.Names}}\t{{.ID}}",
    ]
    for index, raw_command in enumerate(commands):
        command = _require_mapping(raw_command, f"{label} N01 Docker command[{index}]")
        operation = _require_string(command.get("operation"), f"{label} N01 Docker command[{index}].operation")
        if operation not in allowed or operation in command_by_operation:
            raise SummaryError(f"{label} N01 Docker cleanup operation is invalid")
        if operation in {"inventory-before", "inventory-after-stop", "inventory-after-force-remove"}:
            expected = expected_inventory
        elif operation == "stop":
            argv = command.get("argv")
            if (
                not isinstance(argv, list)
                or len(argv) != 6
                or argv[:4] != [launcher, "container", "stop", "--time"]
                or not isinstance(argv[4], str)
                or not re.fullmatch(r"[1-9][0-9]*", argv[4])
                or not 1 <= int(argv[4], 10) <= 30
                or argv[5] != container_name
            ):
                raise SummaryError(f"{label} N01 Docker stop argv is not bounded to the owned container")
            expected = argv
        elif operation == "wait":
            expected = [launcher, "container", "wait", container_name]
        elif operation == "force-remove":
            expected = [launcher, "container", "rm", "--force", container_name]
        else:
            expected = [launcher, "container", "inspect", "--format", "{{.Id}}", container_name]
        command_by_operation[operation] = _validate_n01_docker_cleanup_command(
            command,
            expected_argv=expected,
            timeout_seconds=timeout,
            label=f"{label} N01 Docker command[{index}]",
        )
    # A false proof is a retained failure, but it must still never contain an
    # arbitrary or broad Docker command.  Successful serving needs the full
    # exact-name absence sequence below.
    if receipt.get("cleanup_verified") is not True:
        return {"cleanup_verified": False, "command_count": len(commands)}
    if receipt.get("errors") != [] or receipt.get("final_state") != "absent":
        raise SummaryError(f"{label} N01 Docker cleanup claims success without daemon absence")
    for operation in ("inventory-before", "inventory-after-stop", "inspect-absence"):
        if operation not in command_by_operation:
            raise SummaryError(f"{label} N01 Docker cleanup lacks {operation}")
    before = command_by_operation["inventory-before"]
    after_stop = command_by_operation["inventory-after-stop"]
    inspect = command_by_operation["inspect-absence"]
    if (
        before.get("returncode") != 0
        or after_stop.get("returncode") != 0
        or after_stop.get("stdout_text") != ""
        or inspect.get("returncode") == 0
        or inspect.get("stdout_text") != ""
    ):
        raise SummaryError(f"{label} N01 Docker cleanup does not prove final exact-name absence")
    before_text = before.get("stdout_text")
    assert isinstance(before_text, str)
    if before_text:
        if not re.fullmatch(re.escape(container_name) + r"\t[0-9a-f]{12,64}\n?", before_text):
            raise SummaryError(f"{label} N01 Docker initial inventory is not only the owned container")
        if "stop" not in command_by_operation or "wait" not in command_by_operation:
            raise SummaryError(f"{label} N01 Docker cleanup did not stop/wait its owned container")
    elif "stop" in command_by_operation or "wait" in command_by_operation:
        raise SummaryError(f"{label} N01 Docker cleanup stopped without an owned inventory")
    after_text = after_stop.get("stdout_text")
    assert isinstance(after_text, str)
    if after_text:
        if not re.fullmatch(re.escape(container_name) + r"\t[0-9a-f]{12,64}\n?", after_text):
            raise SummaryError(f"{label} N01 Docker residual inventory is not only the owned container")
        force = command_by_operation.get("force-remove")
        after_force = command_by_operation.get("inventory-after-force-remove")
        if force is None or after_force is None or force.get("returncode") != 0 or after_force.get("returncode") != 0 or after_force.get("stdout_text") != "":
            raise SummaryError(f"{label} N01 Docker residual was not force-removed and re-inventoried")
    elif "force-remove" in command_by_operation or "inventory-after-force-remove" in command_by_operation:
        raise SummaryError(f"{label} N01 Docker cleanup force-removed without a residual inventory")
    return {"cleanup_verified": True, "command_count": len(commands), "container_name": container_name}


def _validate_n06a_parent_cleanup(
    run: Mapping[str, Any], *, cleanup: Mapping[str, Any], output_dir: Path
) -> Mapping[str, Any]:
    kind = _require_string(run.get("kind"), "N06-A parent cleanup run kind")
    if kind not in {"warmup", "timed"}:
        raise SummaryError("N06-A parent cleanup run kind is invalid")
    index = _require_integer(run.get("index"), f"N06-A parent cleanup {kind} index", minimum=1)
    label = f"repeat receipt.{kind}_runs[{index}].n06a_parent_cleanup"
    parent = _require_mapping(run.get("n06a_parent_cleanup"), label)
    expected_path, expected_name = _n06a_parent_container_name(cleanup, kind=kind, index=index)
    required = {
        "schema_version",
        "kind",
        "index",
        "attempt_artifact_directory",
        "container_name",
        "timed_out",
        "parent_termination_requested",
        "timeout_descendant_cleanup",
        "outer_process_cleanup",
        "docker_container_cleanup",
        "errors",
        "cleanup_verified",
        "receipt_path",
        "receipt_sha256",
    }
    _require_exact_keys(parent, required, label)
    if (
        parent.get("schema_version") != N01_N06A_PARENT_CLEANUP_SCHEMA_VERSION
        or parent.get("kind") != kind
        or parent.get("index") != index
        or parent.get("attempt_artifact_directory") != str(expected_path)
        or parent.get("container_name") != expected_name
        or parent.get("timed_out") != run.get("timed_out")
        or not isinstance(parent.get("parent_termination_requested"), bool)
        or not isinstance(parent.get("cleanup_verified"), bool)
        or not isinstance(parent.get("errors"), list)
        or not all(isinstance(item, str) for item in parent["errors"])
    ):
        raise SummaryError(f"{label} contract differs from N01's exact scope")
    receipt_path = Path(_require_string(parent.get("receipt_path"), f"{label}.receipt_path"))
    if receipt_path.name != f"{kind}-{index:03d}.n06a-parent-cleanup.json" or receipt_path.parent != output_dir:
        raise SummaryError(f"{label}.receipt_path must be its direct immutable N01 sidecar")
    sidecar_path, sidecar_payload, sidecar_sha256 = _read_bounded_regular_file(
        receipt_path,
        label=f"{label} sidecar",
        maximum_bytes=MAX_RECEIPT_BYTES,
    )
    if parent.get("receipt_sha256") != sidecar_sha256 or not SHA256_RE.fullmatch(str(parent.get("receipt_sha256"))):
        raise SummaryError(f"{label}.receipt_sha256 differs from its immutable sidecar")
    sidecar = _decode_json(sidecar_payload, label=f"{label} sidecar")
    if sidecar != {key: value for key, value in parent.items() if key not in {"receipt_path", "receipt_sha256"}}:
        raise SummaryError(f"{label} sidecar content differs from the receipt-bound cleanup object")
    daemon = _validate_n01_docker_absence(
        parent.get("docker_container_cleanup"), cleanup=cleanup, container_name=expected_name, label=label
    )
    parent_termination = bool(parent["parent_termination_requested"])
    descendants = _require_mapping(parent.get("timeout_descendant_cleanup"), f"{label}.timeout_descendant_cleanup")
    outer = _require_mapping(parent.get("outer_process_cleanup"), f"{label}.outer_process_cleanup")
    if parent_termination:
        if descendants.get("cleanup_verified") is not True or outer.get("cleanup_verified") is not True:
            if parent.get("cleanup_verified") is True:
                raise SummaryError(f"{label} claims cleanup success without parent/descendant termination proof")
    else:
        if descendants.get("status") != "not-required" or outer.get("status") != "not-required":
            raise SummaryError(f"{label} non-terminated attempt has inconsistent parent cleanup receipts")
    if parent.get("cleanup_verified") is True:
        if parent.get("errors") != [] or daemon.get("cleanup_verified") is not True:
            raise SummaryError(f"{label} claims cleanup success without exact daemon absence")
    elif run.get("status") == "succeeded":
        raise SummaryError(f"{label} successful timed run lacks proven N06-A parent cleanup")
    return {
        "cleanup_verified": parent["cleanup_verified"],
        "receipt_path": str(sidecar_path),
        "receipt_sha256": sidecar_sha256,
        "attempt_artifact_directory": str(expected_path),
        "container_name": expected_name,
        "daemon": daemon,
    }



def _validate_not_started_after_failed_cleanup(
    value: object,
    *,
    kind: str,
    index: int,
    blocking: tuple[str, int] | None,
    label: str,
) -> Mapping[str, Any]:
    """Validate an explicit non-execution placeholder emitted by N01 fail-stop."""
    if blocking is None:
        raise SummaryError(f"{label} is a stray fail-stop placeholder without an earlier failed-cleanup")
    run = _require_exact_keys(
        value,
        {"kind", "index", "status", "blocking_kind", "blocking_index", "reason"},
        label,
    )
    if (
        run.get("kind") != kind
        or run.get("index") != index
        or run.get("status") != N01_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS
        or run.get("blocking_kind") != blocking[0]
        or run.get("blocking_index") != blocking[1]
        or run.get("reason") != N01_NOT_STARTED_AFTER_FAILED_CLEANUP_REASON
    ):
        raise SummaryError(f"{label} fail-stop placeholder does not bind its blocking failed-cleanup")
    return run


def _validate_executed_n01_run(
    run: Mapping[str, Any],
    *,
    kind: str,
    index: int,
    label: str,
    cleanup: Mapping[str, Any] | None,
    output_dir: Path,
) -> None:
    """Check a run that N01 actually launched before any fail-stop boundary."""
    status = _require_string(run.get("status"), f"{label}.status")
    if status not in {"succeeded", "failed", "timed-out", "failed-to-start", "failed-cleanup"}:
        raise SummaryError(f"{label}.status is unsupported: {status!r}")
    _require_finite_number(run.get("wall_time_ms"), f"{label}.wall_time_ms", positive=True)
    timed_out = run.get("timed_out")
    if not isinstance(timed_out, bool):
        raise SummaryError(f"{label}.timed_out must be a boolean")
    exit_code = run.get("exit_code")
    if exit_code is not None:
        _require_integer(exit_code, f"{label}.exit_code")
    if status == "succeeded" and (exit_code != 0 or timed_out):
        raise SummaryError(f"{label} cannot mark a nonzero or timed-out attempt succeeded")
    if status == "timed-out" and not timed_out:
        raise SummaryError(f"{label} marks a non-timeout attempt timed-out")
    _require_string(run.get("stdout_path"), f"{label}.stdout_path")
    _require_string(run.get("stderr_path"), f"{label}.stderr_path")
    error = run.get("error")
    if error is not None and not isinstance(error, str):
        raise SummaryError(f"{label}.error must be a string or null")
    if cleanup is not None:
        _validate_n06a_parent_cleanup(run, cleanup=cleanup, output_dir=output_dir)

def _load_repeat_receipt(receipt_path: Path) -> RepeatReceipt:
    path, payload, sha256 = _read_bounded_regular_file(
        receipt_path,
        label="repeat receipt",
        maximum_bytes=MAX_RECEIPT_BYTES,
    )
    receipt = _decode_json(payload, label="repeat receipt")
    if receipt.get("schema_version") != REPEAT_RECEIPT_SCHEMA_VERSION:
        raise SummaryError(
            "repeat receipt.schema_version must be "
            f"{REPEAT_RECEIPT_SCHEMA_VERSION!r}"
        )
    declared_path = Path(_require_string(receipt.get("receipt_path"), "repeat receipt.receipt_path"))
    try:
        if declared_path.resolve(strict=True) != path:
            raise SummaryError("repeat receipt.receipt_path does not identify the supplied receipt")
    except OSError as error:
        raise SummaryError("repeat receipt.receipt_path cannot be resolved") from error
    output_dir = Path(_require_string(receipt.get("output_dir"), "repeat receipt.output_dir"))
    try:
        output_dir = output_dir.resolve(strict=True)
    except OSError as error:
        raise SummaryError("repeat receipt.output_dir cannot be resolved") from error
    if not output_dir.is_dir():
        raise SummaryError("repeat receipt.output_dir must be a directory")
    configuration = _require_mapping(receipt.get("configuration"), "repeat receipt.configuration")
    timed_repeats = _require_integer(
        configuration.get("timed_repeats"),
        "repeat receipt.configuration.timed_repeats",
        minimum=1,
    )
    if timed_repeats > MAX_TIMED_RUNS:
        raise SummaryError(f"repeat receipt timed_repeats exceeds {MAX_TIMED_RUNS}")
    bootstrap_resamples = _require_integer(
        configuration.get("bootstrap_resamples"),
        "repeat receipt.configuration.bootstrap_resamples",
        minimum=1,
    )
    if bootstrap_resamples > MAX_BOOTSTRAP_RESAMPLES:
        raise SummaryError(
            "repeat receipt.bootstrap_resamples exceeds the bounded shared-host reader limit "
            f"of {MAX_BOOTSTRAP_RESAMPLES}"
        )
    if timed_repeats * bootstrap_resamples > MAX_BOOTSTRAP_DRAWS_PER_METRIC:
        raise SummaryError(
            "repeat receipt timed_repeats * bootstrap_resamples exceeds the bounded "
            f"shared-host reader limit of {MAX_BOOTSTRAP_DRAWS_PER_METRIC} draws per metric"
        )
    bootstrap_seed = _require_integer(
        configuration.get("bootstrap_seed"),
        "repeat receipt.configuration.bootstrap_seed",
    )
    n06a_timeout_cleanup = _parse_n06a_timeout_cleanup_configuration(
        configuration.get("n06a_timeout_cleanup")
    )
    if receipt.get("status") not in {"completed", "completed-with-failures"}:
        raise SummaryError("repeat receipt must be completed before analysis")
    raw_runs = receipt.get("timed_runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != timed_repeats:
        raise SummaryError("repeat receipt must retain every configured timed attempt")

    # The parent-side cleanup feature makes N01's warmup attempts relevant:
    # an unproven warmup cleanup blocks every later timed launch.  Legacy
    # receipts without this feature retain the historical timed-only reader.
    warmup_runs: list[Mapping[str, Any]] = []
    blocking: tuple[str, int] | None = None
    if n06a_timeout_cleanup is not None:
        warmups = _require_integer(
            configuration.get("warmups"),
            "repeat receipt.configuration.warmups",
            minimum=0,
        )
        raw_warmups = receipt.get("warmup_runs")
        if not isinstance(raw_warmups, list) or len(raw_warmups) != warmups:
            raise SummaryError("repeat receipt must retain every configured N06-A warmup attempt")
        for expected_index, raw_warmup in enumerate(raw_warmups, start=1):
            label = f"repeat receipt.warmup_runs[{expected_index}]"
            run = _require_mapping(raw_warmup, label)
            status = _require_string(run.get("status"), f"{label}.status")
            if status == N01_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS:
                warmup_runs.append(
                    _validate_not_started_after_failed_cleanup(
                        run,
                        kind="warmup",
                        index=expected_index,
                        blocking=blocking,
                        label=label,
                    )
                )
                continue
            if blocking is not None:
                raise SummaryError(f"{label} executed after an earlier failed-cleanup fail-stop")
            if run.get("kind") != "warmup" or run.get("index") != expected_index:
                raise SummaryError(f"{label} must retain its sequential warmup kind/index")
            _validate_executed_n01_run(
                run,
                kind="warmup",
                index=expected_index,
                label=label,
                cleanup=n06a_timeout_cleanup,
                output_dir=output_dir,
            )
            warmup_runs.append(run)
            if status == "failed-cleanup":
                blocking = ("warmup", expected_index)

    timed_runs: list[Mapping[str, Any]] = []
    for expected_index, raw_run in enumerate(raw_runs, start=1):
        label = f"repeat receipt.timed_runs[{expected_index}]"
        run = _require_mapping(raw_run, label)
        status = _require_string(run.get("status"), f"{label}.status")
        if status == N01_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS:
            if n06a_timeout_cleanup is None:
                raise SummaryError(f"{label} has a fail-stop placeholder without N06-A cleanup configuration")
            timed_runs.append(
                _validate_not_started_after_failed_cleanup(
                    run,
                    kind="timed",
                    index=expected_index,
                    blocking=blocking,
                    label=label,
                )
            )
            continue
        if blocking is not None:
            raise SummaryError(f"{label} executed after an earlier failed-cleanup fail-stop")
        if run.get("kind") != "timed" or run.get("index") != expected_index:
            raise SummaryError(f"{label} must retain its sequential timed kind/index")
        _validate_executed_n01_run(
            run,
            kind="timed",
            index=expected_index,
            label=label,
            cleanup=n06a_timeout_cleanup,
            output_dir=output_dir,
        )
        timed_runs.append(run)
        if n06a_timeout_cleanup is not None and status == "failed-cleanup":
            blocking = ("timed", expected_index)
    return RepeatReceipt(
        path=path,
        sha256=sha256,
        output_dir=output_dir,
        timed_repeats=timed_repeats,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        warmup_runs=tuple(sorted(warmup_runs, key=lambda item: int(item["index"]))),
        timed_runs=tuple(sorted(timed_runs, key=lambda item: int(item["index"]))),
        n06a_timeout_cleanup=n06a_timeout_cleanup,
    )


def _load_timed_stdout(receipt: RepeatReceipt, run: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    index = _require_integer(run.get("index"), "timed run.index", minimum=1)
    raw_path = Path(_require_string(run.get("stdout_path"), f"timed run {index}.stdout_path"))
    expected_name = f"timed-{index:03d}.stdout.log"
    if raw_path.name != expected_name:
        raise SummaryError(f"timed run {index}.stdout_path must be named {expected_name!r}")
    path, payload, sha256 = _read_bounded_regular_file(
        raw_path,
        label=f"timed stdout for run {index}",
        maximum_bytes=MAX_STDOUT_LOG_BYTES,
    )
    if path.parent != receipt.output_dir:
        raise SummaryError(f"timed run {index}.stdout_path must be directly inside receipt.output_dir")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SummaryError(f"timed stdout for run {index} is not UTF-8") from error
    return text, {"path": str(path), "sha256": sha256}


def _validate_startup_snapshot(
    *, path_text: str, expected_sha256: str, marker_label: str
) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise SummaryError(f"{marker_label}.startup_log_sha256 must be a lowercase SHA-256")
    path = Path(path_text)
    if not path.is_absolute():
        raise SummaryError(f"{marker_label}.startup_log_path must be absolute")
    resolved, payload, observed_sha256 = _read_bounded_regular_file(
        path,
        label=f"{marker_label} startup log snapshot",
        maximum_bytes=MAX_STARTUP_SNAPSHOT_BYTES,
    )
    if observed_sha256 != expected_sha256:
        raise SummaryError(f"{marker_label} startup log SHA-256 does not match marker")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SummaryError(f"{marker_label} startup log snapshot is not UTF-8") from error
    receipt_fields: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        offset = line.find(STARTUP_RECEIPT_PREFIX)
        if offset < 0:
            continue
        if line.find(STARTUP_RECEIPT_PREFIX, offset + len(STARTUP_RECEIPT_PREFIX)) >= 0:
            raise SummaryError(f"{marker_label} startup log line {line_number} has multiple receipts")
        receipt_fields.append(
            _parse_marker_tokens(
                line[offset:],
                prefix=STARTUP_RECEIPT_PREFIX,
                label=f"{marker_label} startup log line {line_number}",
            )
        )
    if len(receipt_fields) != 1:
        raise SummaryError(
            f"{marker_label} startup log snapshot must contain exactly one {STARTUP_RECEIPT_PREFIX} receipt"
        )
    fields = receipt_fields[0]
    expected_fields = {
        "requested_backend",
        "resolved_ragged_backend",
        "fallback_reason",
        "query_heads",
        "key_value_heads",
        "head_size",
        "page_size",
        "graph",
    }
    if set(fields) != expected_fields:
        missing = sorted(expected_fields - set(fields))
        extra = sorted(set(fields) - expected_fields)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported " + ", ".join(extra))
        raise SummaryError(f"{marker_label} startup receipt has " + "; ".join(details))
    if fields["requested_backend"] != REQUESTED_BACKEND_CLI_ID:
        raise SummaryError(f"{marker_label} startup receipt requested_backend differs")
    if fields["resolved_ragged_backend"] != SERVING_IMPLEMENTATION_ID:
        raise SummaryError(f"{marker_label} startup receipt resolved_ragged_backend differs")
    if fields["fallback_reason"] != "none" or fields["graph"] != "false":
        raise SummaryError(f"{marker_label} startup receipt must record no fallback and graph=false")
    geometry = {
        "query_heads": _parse_record_integer(fields["query_heads"], f"{marker_label}.query_heads", minimum=1),
        "key_value_heads": _parse_record_integer(
            fields["key_value_heads"], f"{marker_label}.key_value_heads", minimum=1
        ),
        "head_size": _parse_record_integer(fields["head_size"], f"{marker_label}.head_size", minimum=1),
        "page_size": _parse_record_integer(fields["page_size"], f"{marker_label}.page_size", minimum=1),
    }
    if geometry != {"query_heads": 16, "key_value_heads": 2, "head_size": 128, "page_size": 16}:
        raise SummaryError(f"{marker_label} startup receipt geometry differs from Qwen D128")
    return {
        "path": str(resolved),
        "sha256": observed_sha256,
        "marker_prefix": STARTUP_RECEIPT_PREFIX,
        "requested_backend": fields["requested_backend"],
        "resolved_ragged_backend": fields["resolved_ragged_backend"],
        "geometry": geometry,
    }


def _read_marked_artifact(
    *, path_text: str, expected_sha256: str, label: str, maximum_bytes: int
) -> tuple[Path, bytes, str]:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise SummaryError(f"{label} SHA-256 must be a lowercase SHA-256")
    path = Path(path_text)
    if not path.is_absolute():
        raise SummaryError(f"{label} path must be absolute")
    resolved, payload, observed_sha256 = _read_bounded_regular_file(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    if observed_sha256 != expected_sha256:
        raise SummaryError(f"{label} SHA-256 does not match marker")
    return resolved, payload, observed_sha256


def _decode_jsonl_rows(payload: bytes, *, label: str) -> list[Mapping[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SummaryError(f"{label} is not UTF-8") from error
    lines = text.splitlines()
    if not lines or len(lines) > MAX_RETAINED_REQUEST_ROWS:
        raise SummaryError(f"{label} must contain 1..{MAX_RETAINED_REQUEST_ROWS} JSONL rows")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise SummaryError(f"{label}:{line_number} must not be blank")
        try:
            decoded = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise SummaryError(f"{label}:{line_number} is malformed JSON") from error
        rows.append(_require_mapping(decoded, f"{label}:{line_number}"))
    return rows


def _argv_option_values(argv: Sequence[str], flag: str, *, label: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == flag:
            if index + 1 >= len(argv):
                raise SummaryError(f"{label} has {flag} without a value")
            values.append(argv[index + 1])
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
    return values



_DOCKER_RUN_VALUE_OPTIONS = frozenset({"--gpus", "--ipc", "--network", "-v", "--volume", "--name"})
_DOCKER_RUN_FLAG_OPTIONS = frozenset({"--rm"})


def _docker_run_image_operand_index(argv: Sequence[str], *, label: str) -> int:
    """Find Docker's image operand without treating an inert digest arg as it."""
    if len(argv) < 3 or argv[1] != "run":
        raise SummaryError(f"{label} must begin with Docker run")
    index = 2
    while index < len(argv):
        token = argv[index]
        if token in _DOCKER_RUN_FLAG_OPTIONS:
            index += 1
            continue
        if token in _DOCKER_RUN_VALUE_OPTIONS:
            if index + 1 >= len(argv) or not argv[index + 1]:
                raise SummaryError(f"{label} Docker option {token!r} lacks a value before its image operand")
            index += 2
            continue
        if any(
            token.startswith(option + "=")
            for option in _DOCKER_RUN_VALUE_OPTIONS
            if option.startswith("--")
        ):
            if token.endswith("="):
                raise SummaryError(f"{label} Docker option {token!r} has an empty value")
            index += 1
            continue
        if token.startswith("-"):
            raise SummaryError(f"{label} has unsupported Docker option before its image operand: {token!r}")
        return index
    raise SummaryError(f"{label} lacks a Docker image operand")


def _require_docker_controls_before_image(
    argv: Sequence[str], *, image_operand_index: int, label: str
) -> None:
    """Reject Docker controls after the image, where Docker would not apply them."""
    if image_operand_index < 2 or image_operand_index >= len(argv):
        raise SummaryError(f"{label} has an invalid Docker image operand index")
    for index, token in enumerate(argv[2:], start=2):
        is_value_option = token in _DOCKER_RUN_VALUE_OPTIONS
        is_equals_value_option = any(
            token.startswith(option + "=")
            for option in _DOCKER_RUN_VALUE_OPTIONS
            if option.startswith("--")
        )
        is_flag_option = token in _DOCKER_RUN_FLAG_OPTIONS or token.startswith("--rm=")
        if not (is_value_option or is_equals_value_option or is_flag_option):
            continue
        if index >= image_operand_index:
            raise SummaryError(
                f"{label} Docker control option {token!r} must occur before its image operand"
            )
        if is_value_option and index + 1 >= image_operand_index:
            raise SummaryError(
                f"{label} Docker control option {token!r} must retain its value before the image operand"
            )

def _load_pinned_n06a_reference(
    *,
    workload_config: Mapping[str, Any],
    record: Mapping[str, Any],
    label: str,
) -> tuple[token_client.TokenReference, Mapping[str, Any]]:
    """Load the exact workload whose path and digest are promoted in a marker.

    The raw completion rows are meaningful only relative to the immutable
    prompt and reference IDs.  A summary therefore fails closed when the
    source workload has been removed or changed, instead of accepting a
    marker's count-only description of it.
    """
    required_config = {
        "path",
        "sha256",
        "case",
        "model_id",
        "model_revision",
        "prompt_tokens",
        "max_output_tokens",
        "reference_sha256",
    }
    _require_exact_keys(workload_config, required_config, f"{label} attempt config.workload")
    path_text = _require_string(workload_config.get("path"), f"{label} workload path")
    expected_sha256 = _require_string(workload_config.get("sha256"), f"{label} workload SHA-256")
    if not SHA256_RE.fullmatch(expected_sha256):
        raise SummaryError(f"{label} workload SHA-256 must be a lowercase SHA-256")
    workload_path = Path(path_text)
    if not workload_path.is_absolute():
        raise SummaryError(f"{label} workload path must be absolute")
    _, payload, observed_sha256 = _read_bounded_regular_file(
        workload_path,
        label=f"{label} pinned workload",
        maximum_bytes=MAX_WORKLOAD_BYTES,
    )
    if observed_sha256 != expected_sha256:
        raise SummaryError(f"{label} pinned workload SHA-256 differs from attempt config")
    document = _decode_json(payload, label=f"{label} pinned workload")
    required_document = {
        "schema_version",
        "case",
        "model_id",
        "model_revision",
        "prompt",
        "prompt_token_ids",
        "output_token_ids",
        "output_text",
        "finish_reason",
        "sampling",
    }
    _require_exact_keys(document, required_document, f"{label} pinned workload")
    if document.get("schema_version") != WORKLOAD_SCHEMA_VERSION:
        raise SummaryError(f"{label} pinned workload schema differs")
    expected_identity = {
        "case": record["case"],
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
    }
    for key, expected in expected_identity.items():
        if document.get(key) != expected or workload_config.get(key) != expected:
            raise SummaryError(f"{label} pinned workload differs at {key!r}")
    prompt = _require_string(document.get("prompt"), f"{label} pinned workload.prompt")
    output_text = document.get("output_text")
    if not isinstance(output_text, str):
        raise SummaryError(f"{label} pinned workload.output_text must be a string")
    if document.get("finish_reason") != "length":
        raise SummaryError(f"{label} pinned workload.finish_reason must be 'length'")
    prompt_ids = document.get("prompt_token_ids")
    output_ids = document.get("output_token_ids")
    if (
        not isinstance(prompt_ids, list)
        or not isinstance(output_ids, list)
        or len(prompt_ids) != record["prompt_tokens"]
        or len(output_ids) != record["max_output_tokens"]
        or any(type(token) is not int or not 0 <= token < 2**32 for token in [*prompt_ids, *output_ids])
    ):
        raise SummaryError(f"{label} pinned workload token IDs differ from its marker shape")
    if (
        workload_config.get("prompt_tokens") != len(prompt_ids)
        or workload_config.get("max_output_tokens") != len(output_ids)
    ):
        raise SummaryError(f"{label} pinned workload token counts differ from attempt config")
    sampling = _require_exact_keys(
        document.get("sampling"), {"temperature", "top_p"}, f"{label} pinned workload.sampling"
    )
    if sampling.get("temperature") != 0.0 or sampling.get("top_p") != 1.0:
        raise SummaryError(f"{label} pinned workload must retain greedy temperature=0.0/top_p=1.0")
    try:
        reference = token_client.TokenReference(
            record["model_id"],
            tuple(prompt_ids),
            tuple(output_ids),
            output_text,
            "length",
        )
    except Exception as error:
        raise SummaryError(f"{label} pinned workload cannot construct a strict token reference: {error}") from error
    if workload_config.get("reference_sha256") != reference.sha256:
        raise SummaryError(f"{label} pinned workload reference SHA-256 differs from attempt config")
    return reference, document


def _validate_pinned_token_client(config: Mapping[str, Any], *, label: str) -> None:
    """Require the current replay parser to be exactly the controller-pinned source."""
    controller = _require_mapping(config.get("controller"), f"{label} attempt config.controller")
    expected_path = _require_string(controller.get("token_client_path"), f"{label} token client path")
    expected_sha256 = _require_string(controller.get("token_client_sha256"), f"{label} token client SHA-256")
    if not SHA256_RE.fullmatch(expected_sha256):
        raise SummaryError(f"{label} token client SHA-256 must be a lowercase SHA-256")
    current_path = Path(token_client.__file__).resolve()
    _, _, current_sha256 = _read_bounded_regular_file(
        current_path,
        label=f"{label} current token client source",
        maximum_bytes=MAX_WORKLOAD_BYTES,
    )
    if expected_path != str(current_path) or expected_sha256 != current_sha256:
        raise SummaryError(
            f"{label} replay token client differs from the source pinned by the serving controller"
        )


def _model_identity_relative_path(value: object, label: str) -> str:
    raw = _require_string(value, label)
    if "\\" in raw:
        raise SummaryError(f"{label} must use a portable relative POSIX path")
    parsed = PurePosixPath(raw)
    if (
        parsed.is_absolute()
        or raw in {".", ".."}
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise SummaryError(f"{label} must be a non-traversing relative path")
    return raw


def _model_identity_file(model_path: Path, relative_path: str, *, label: str) -> Path:
    candidate = model_path.joinpath(*PurePosixPath(relative_path).parts)
    try:
        before = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise SummaryError(f"{label} cannot be inspected: {candidate}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SummaryError(f"{label} must be a regular non-symlink file")
    try:
        resolved.relative_to(model_path)
    except ValueError as error:
        raise SummaryError(f"{label} resolves outside the pinned model path") from error
    return resolved


def _parse_model_identity_lfs_listing(text: str, *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s+[ *-]\s+(.+)", raw_line)
        if match is None:
            raise SummaryError(f"{label} line {line_number} is not a git-lfs long listing")
        oid, raw_path = match.groups()
        relative_path = _model_identity_relative_path(raw_path, f"{label} path on line {line_number}")
        if relative_path in result:
            raise SummaryError(f"{label} repeats path {relative_path!r}")
        result[relative_path] = oid
    return result


def _validate_model_identity_command(
    command: object, *, expected_argv: list[str], label: str
) -> Mapping[str, Any]:
    value = _require_mapping(command, label)
    if value.get("argv") != expected_argv or value.get("returncode") != 0 or "error" in value:
        raise SummaryError(f"{label} does not prove the exact bounded git/LFS command")
    if value.get("timeout_seconds") != MODEL_IDENTITY_GIT_TIMEOUT_SECONDS:
        raise SummaryError(f"{label}.timeout_seconds differs")
    started_ns = _require_integer(value.get("started_ns"), f"{label}.started_ns", minimum=0)
    finished_ns = _require_integer(value.get("finished_ns"), f"{label}.finished_ns", minimum=started_ns)
    if finished_ns < started_ns:
        raise SummaryError(f"{label} timestamps move backwards")
    for stream in ("stdout", "stderr"):
        text = value.get(f"{stream}_text")
        if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_MODEL_IDENTITY_GIT_OUTPUT_BYTES:
            raise SummaryError(f"{label}.{stream}_text is invalid")
        raw = text.encode("utf-8")
        if (
            value.get(f"{stream}_bytes") != len(raw)
            or value.get(f"{stream}_sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise SummaryError(f"{label}.{stream} receipt differs")
    return value


def _validate_model_identity_artifacts(
    *,
    fields: Mapping[str, str],
    config: Mapping[str, Any],
    record: Mapping[str, Any],
    attempt_dir: Path,
    label: str,
) -> dict[str, Any]:
    """Replay the pinned model manifest and its bounded Git/LFS receipt.

    Large shards are checked by regular-file size and a current `git lfs
    ls-files -l` OID listing; the driver never rereads their multi-gigabyte
    payload during a repeated serving campaign.  Small tokenizer/config files
    are rehashed from the manifest on every summary pass.
    """
    manifest_path, manifest_payload, manifest_sha256 = _read_marked_artifact(
        path_text=fields["model_identity_manifest_path"],
        expected_sha256=fields["model_identity_manifest_sha256"],
        label=f"{label} model identity manifest",
        maximum_bytes=MAX_MODEL_IDENTITY_MANIFEST_BYTES,
    )
    validation_path, validation_payload, validation_sha256 = _read_marked_artifact(
        path_text=fields["model_identity_validation_path"],
        expected_sha256=fields["model_identity_validation_sha256"],
        label=f"{label} model identity validation",
        maximum_bytes=MAX_MODEL_IDENTITY_MANIFEST_BYTES,
    )
    if (
        manifest_path.parent != attempt_dir
        or validation_path.parent != attempt_dir
        or manifest_path.name != "model.identity.manifest.json"
        or validation_path.name != "model.identity.validation.json"
    ):
        raise SummaryError(f"{label} model identity artifacts must be dedicated attempt files")
    launch = _require_mapping(config.get("launch_provenance"), f"{label} attempt config.launch_provenance")
    identity = _require_exact_keys(
        launch.get("model_identity"),
        {
            "path",
            "sha256",
            "source_path",
            "source_sha256",
            "validation_path",
            "validation_sha256",
            "git_path",
            "git_sha256",
        },
        f"{label} attempt config.launch_provenance.model_identity",
    )
    if (
        identity.get("path") != str(manifest_path)
        or identity.get("sha256") != manifest_sha256
        or identity.get("validation_path") != str(validation_path)
        or identity.get("validation_sha256") != validation_sha256
        or identity.get("source_sha256") != manifest_sha256
    ):
        raise SummaryError(f"{label} model identity config differs from marker-bound artifacts")
    source_path = _require_string(identity.get("source_path"), f"{label} model identity source path")
    if not Path(source_path).is_absolute():
        raise SummaryError(f"{label} model identity source path must be absolute")
    git_path = _require_string(identity.get("git_path"), f"{label} model identity git path")
    git_sha256 = _require_string(identity.get("git_sha256"), f"{label} model identity git SHA-256")
    if not Path(git_path).is_absolute() or not SHA256_RE.fullmatch(git_sha256):
        raise SummaryError(f"{label} model identity git provenance is invalid")

    manifest = _decode_json(manifest_payload, label=f"{label} model identity manifest")
    _require_exact_keys(
        manifest,
        {"schema_version", "model_id", "model_revision", "model_path", "metadata_files", "shards"},
        f"{label} model identity manifest",
    )
    if (
        manifest.get("schema_version") != MODEL_IDENTITY_MANIFEST_SCHEMA_VERSION
        or manifest.get("model_id") != record["model_id"]
        or manifest.get("model_revision") != record["model_revision"]
    ):
        raise SummaryError(f"{label} model identity manifest does not pin the marker model")
    model_path_text = _require_string(manifest.get("model_path"), f"{label} model identity model_path")
    model_path = Path(model_path_text)
    if not model_path.is_absolute() or not model_path.is_dir() or str(model_path.resolve()) != model_path_text:
        raise SummaryError(f"{label} model identity model_path must be an existing canonical directory")
    metadata_raw = manifest.get("metadata_files")
    shards_raw = manifest.get("shards")
    if (
        not isinstance(metadata_raw, list)
        or not metadata_raw
        or len(metadata_raw) > 128
        or not isinstance(shards_raw, list)
        or not shards_raw
        or len(shards_raw) > 128
    ):
        raise SummaryError(f"{label} model identity manifest file lists are invalid")
    known_paths: set[str] = set()
    metadata: list[tuple[str, int, str]] = []
    for index, item in enumerate(metadata_raw):
        entry = _require_exact_keys(
            item, {"path", "size_bytes", "sha256"}, f"{label} model metadata[{index}]"
        )
        relative_path = _model_identity_relative_path(entry.get("path"), f"{label} model metadata[{index}].path")
        size_bytes = _require_integer(entry.get("size_bytes"), f"{label} model metadata[{index}].size_bytes", minimum=1)
        sha256 = _require_string(entry.get("sha256"), f"{label} model metadata[{index}].sha256")
        if size_bytes > MAX_MODEL_IDENTITY_METADATA_BYTES or not SHA256_RE.fullmatch(sha256) or relative_path in known_paths:
            raise SummaryError(f"{label} model metadata[{index}] identity is invalid")
        known_paths.add(relative_path)
        file_path = _model_identity_file(model_path, relative_path, label=f"{label} model metadata[{index}]")
        _, payload, observed_sha256 = _read_bounded_regular_file(
            file_path,
            label=f"{label} model metadata {relative_path}",
            maximum_bytes=MAX_MODEL_IDENTITY_METADATA_BYTES,
        )
        if len(payload) != size_bytes or observed_sha256 != sha256:
            raise SummaryError(f"{label} model metadata identity differs for {relative_path}")
        metadata.append((relative_path, size_bytes, sha256))
    shards: list[tuple[str, int, str]] = []
    for index, item in enumerate(shards_raw):
        entry = _require_exact_keys(
            item, {"path", "size_bytes", "lfs_oid_sha256"}, f"{label} model shard[{index}]"
        )
        relative_path = _model_identity_relative_path(entry.get("path"), f"{label} model shard[{index}].path")
        size_bytes = _require_integer(entry.get("size_bytes"), f"{label} model shard[{index}].size_bytes", minimum=1)
        oid = _require_string(entry.get("lfs_oid_sha256"), f"{label} model shard[{index}].lfs_oid_sha256")
        if not SHA256_RE.fullmatch(oid) or relative_path in known_paths:
            raise SummaryError(f"{label} model shard[{index}] identity is invalid")
        known_paths.add(relative_path)
        file_path = _model_identity_file(model_path, relative_path, label=f"{label} model shard[{index}]")
        try:
            before = file_path.stat()
            after = file_path.stat()
        except OSError as error:
            raise SummaryError(f"{label} model shard[{index}] cannot be statted") from error
        if (
            before.st_size != size_bytes
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise SummaryError(f"{label} model shard identity differs for {relative_path}")
        shards.append((relative_path, size_bytes, oid))

    validation = _decode_json(validation_payload, label=f"{label} model identity validation")
    _require_exact_keys(
        validation,
        {
            "schema_version",
            "manifest_sha256",
            "model_path",
            "model_revision",
            "git",
            "commands",
            "observed_git_head",
            "observed_lfs_oids",
            "validated",
            "errors",
        },
        f"{label} model identity validation",
    )
    if (
        validation.get("schema_version") != MODEL_IDENTITY_VALIDATION_SCHEMA_VERSION
        or validation.get("manifest_sha256") != manifest_sha256
        or validation.get("model_path") != model_path_text
        or validation.get("model_revision") != record["model_revision"]
        or validation.get("validated") is not True
        or validation.get("errors") != []
        or validation.get("observed_git_head") != record["model_revision"]
        or validation.get("git") != {"path": git_path, "sha256": git_sha256}
    ):
        raise SummaryError(f"{label} model identity validation does not prove the pinned revision")
    commands = validation.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        raise SummaryError(f"{label} model identity validation command list is invalid")
    head = _validate_model_identity_command(
        commands[0],
        expected_argv=[git_path, "-C", model_path_text, "rev-parse", "HEAD"],
        label=f"{label} model identity validation head command",
    )
    lfs = _validate_model_identity_command(
        commands[1],
        expected_argv=[git_path, "-C", model_path_text, "lfs", "ls-files", "-l"],
        label=f"{label} model identity validation LFS command",
    )
    if head.get("stdout_text") != record["model_revision"] + "\n":
        raise SummaryError(f"{label} model identity validation git HEAD stdout differs")
    lfs_text = lfs.get("stdout_text")
    assert isinstance(lfs_text, str)
    observed_listing = _parse_model_identity_lfs_listing(
        lfs_text, label=f"{label} model identity validation LFS stdout"
    )
    expected_oids = {path: oid for path, _, oid in shards}
    expected_observed = {path: expected_oids[path] for path in sorted(expected_oids)}
    if validation.get("observed_lfs_oids") != expected_observed:
        raise SummaryError(f"{label} model identity validation observed LFS OIDs differ")
    if any(observed_listing.get(path) != oid for path, oid in expected_oids.items()):
        raise SummaryError(f"{label} model identity validation LFS stdout differs from the manifest")
    return {
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "validation": {"path": str(validation_path), "sha256": validation_sha256},
        "model_path": model_path_text,
        "metadata_file_count": len(metadata),
        "shard_count": len(shards),
    }


def _overlap_peak_from_rows(
    rows: Sequence[Mapping[str, Any]], *, start_key: str, end_key: str, label: str
) -> int:
    events: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        start = _require_integer(row.get(start_key), f"{label}[{row_index}].{start_key}", minimum=0)
        end = _require_integer(row.get(end_key), f"{label}[{row_index}].{end_key}", minimum=0)
        if end <= start:
            raise SummaryError(f"{label}[{row_index}] has an invalid {start_key}/{end_key} interval")
        events.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise SummaryError(f"{label} request intervals are unbalanced")
        peak = max(peak, active)
    if active != 0:
        raise SummaryError(f"{label} request intervals are unbalanced")
    return peak


def _replay_retained_row(
    *,
    row: Mapping[str, Any],
    row_label: str,
    reference: token_client.TokenReference,
    workload_document: Mapping[str, Any],
    request_timeout_seconds: float,
) -> Mapping[str, Any]:
    """Reparse one persisted strict SSE exchange and return replayed metrics."""
    expected_request = {
        "model": reference.model,
        "prompt": workload_document["prompt"],
        "max_tokens": len(reference.output_token_ids),
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    required_truth = {
        "schema_version": token_client.SCHEMA,
        "status": "success",
        "mode": "strict",
        "streaming": True,
        "protocol_valid": True,
        "parser_protocol_valid": True,
        "transport_complete": True,
        "reference_match": True,
        "observation_only": False,
        "correctness_qualified": False,
        "performance_qualified": False,
        "error": None,
        "cleanup_errors": [],
        "owned_connection_closed": True,
        "http_status": 200,
        "reference_sha256": reference.sha256,
    }
    for key, expected in required_truth.items():
        if row.get(key) != expected:
            raise SummaryError(f"{row_label}.{key} differs from the strict retained-row contract")
    if row.get("request") != expected_request:
        raise SummaryError(f"{row_label}.request differs from the pinned strict completion request")
    expected_request_sha256 = hashlib.sha256(
        json.dumps(expected_request, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    if row.get("request_body_sha256") != expected_request_sha256:
        raise SummaryError(f"{row_label}.request_body_sha256 differs from the pinned completion request")
    if _require_finite_number(
        row.get("total_deadline_seconds"), f"{row_label}.total_deadline_seconds", positive=True
    ) != request_timeout_seconds:
        raise SummaryError(f"{row_label}.total_deadline_seconds differs from attempt config")
    call_started_ns = _require_integer(row.get("call_started_ns"), f"{row_label}.call_started_ns", minimum=0)
    started_ns = _require_integer(row.get("started_ns"), f"{row_label}.started_ns", minimum=0)
    headers_received_ns = _require_integer(
        row.get("headers_received_ns"), f"{row_label}.headers_received_ns", minimum=0
    )
    finished_ns = _require_integer(row.get("finished_ns"), f"{row_label}.finished_ns", minimum=0)
    call_finished_ns = _require_integer(
        row.get("call_finished_ns"), f"{row_label}.call_finished_ns", minimum=0
    )
    if not call_started_ns <= started_ns <= headers_received_ns <= finished_ns <= call_finished_ns:
        raise SummaryError(f"{row_label} request timestamps are not ordered")
    frames = row.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SummaryError(f"{row_label}.frames must retain the complete nonempty SSE exchange")
    parser = token_client.TokenResponseParser(reference, streaming=True, started_ns=started_ns, mode="strict")
    try:
        for frame_index, frame in enumerate(frames):
            frame_mapping = _require_exact_keys(
                frame, {"arrived_ns", "data"}, f"{row_label}.frames[{frame_index}]"
            )
            arrived_ns = _require_integer(
                frame_mapping.get("arrived_ns"), f"{row_label}.frames[{frame_index}].arrived_ns", minimum=0
            )
            data = frame_mapping.get("data")
            if not isinstance(data, str) or not headers_received_ns <= arrived_ns <= finished_ns:
                raise SummaryError(f"{row_label}.frames[{frame_index}] is outside the completed SSE transport interval")
            parser.feed_sse(data.encode("utf-8"), arrived_ns)
        replayed = parser.finish()
    except SummaryError:
        raise
    except Exception as error:
        raise SummaryError(f"{row_label} saved SSE frames cannot be replayed strictly: {error}") from error
    for key, expected in replayed.items():
        if row.get(key) != expected:
            raise SummaryError(f"{row_label}.{key} differs from replayed strict SSE evidence")
    if row.get("token_ids") != list(reference.output_token_ids) or row.get("prompt_token_ids") != list(reference.prompt_token_ids):
        raise SummaryError(f"{row_label} replayed token IDs differ from the pinned workload")
    groups = _require_mapping(row.get("token_delivery_groups"), f"{row_label}.token_delivery_groups")
    if groups.get("frame_token_counts") != [1] * len(reference.output_token_ids):
        raise SummaryError(f"{row_label} did not retain exactly one generated token per SSE event")
    identity = _require_mapping(row.get("response_identity"), f"{row_label}.response_identity")
    _require_string(identity.get("id"), f"{row_label}.response_identity.id")
    metrics = _require_mapping(replayed.get("metrics"), f"{row_label} replay metrics")
    for metric_name in ("e2e_ns", "token_ttft_ns", "token_tpot_ns"):
        _require_finite_number(metrics.get(metric_name), f"{row_label}.metrics.{metric_name}", positive=True)
    return {
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "call_started_ns": call_started_ns,
        "call_finished_ns": call_finished_ns,
        "response_id": identity["id"],
        "metrics": metrics,
    }


def _validate_server_warmup_artifacts(
    *,
    lane: str,
    fields: Mapping[str, str],
    attempt_dir: Path,
    record: Mapping[str, Any],
    configuration: Mapping[str, Any],
    reference: token_client.TokenReference,
    workload_document: Mapping[str, Any],
    request_timeout_seconds: float,
    label: str,
) -> dict[str, Any]:
    """Replay the completed per-server warmup bound into each promoted marker."""
    request_path, request_payload, request_sha256 = _read_marked_artifact(
        path_text=fields["server_warmup_request_rows_path"],
        expected_sha256=fields["server_warmup_request_rows_sha256"],
        label=f"{label} server warmup request rows",
        maximum_bytes=MAX_RETAINED_REQUEST_ROWS_BYTES,
    )
    phase_path, phase_payload, phase_sha256 = _read_marked_artifact(
        path_text=fields["server_warmup_phase_path"],
        expected_sha256=fields["server_warmup_phase_sha256"],
        label=f"{label} server warmup phase",
        maximum_bytes=MAX_RETAINED_PHASE_BYTES,
    )
    if (
        request_path.parent != attempt_dir
        or phase_path.parent != attempt_dir
        or request_path.name != f"{lane}.server-warmup.requests.jsonl"
        or phase_path.name != f"{lane}.server-warmup.phase.json"
    ):
        raise SummaryError(f"{label} server warmup artifacts must be dedicated lane files in the attempt directory")
    warmup_requests = _require_integer(
        configuration.get("warmup_requests"), f"{label} attempt config configuration.warmup_requests", minimum=1
    )
    rows = _decode_jsonl_rows(request_payload, label=f"{label} server warmup request rows")
    if len(rows) != warmup_requests:
        raise SummaryError(f"{label} server warmup row count differs from attempt config")
    replayed_rows: list[Mapping[str, Any]] = []
    seen_indices: set[int] = set()
    for row_index, row in enumerate(rows):
        row_label = f"{label} server warmup request rows[{row_index}]"
        if row.get("phase") != "server-warmup" or row.get("warmup") is not True:
            raise SummaryError(f"{row_label} is not a strict per-server warmup row")
        index = _require_integer(row.get("index"), f"{row_label}.index", minimum=0)
        if index in seen_indices:
            raise SummaryError(f"{row_label}.index is duplicated")
        seen_indices.add(index)
        replayed_rows.append(
            _replay_retained_row(
                row=row,
                row_label=row_label,
                reference=reference,
                workload_document=workload_document,
                request_timeout_seconds=request_timeout_seconds,
            )
        )
    if seen_indices != set(range(warmup_requests)):
        raise SummaryError(f"{label} server warmup indices must cover the completed request range")
    if len({str(row["response_id"]) for row in replayed_rows}) != len(replayed_rows):
        raise SummaryError(f"{label} server warmup rows repeat a response identity")
    phase = _decode_json(phase_payload, label=f"{label} server warmup phase")
    expected_phase = {
        "schema_version": RETAINED_PHASE_SCHEMA_VERSION,
        "phase": "server-warmup",
        "warmup": True,
        "offered_concurrency": record["offered_concurrency"],
        "requested": warmup_requests,
        "attempted": warmup_requests,
        "succeeded": warmup_requests,
        "failed": 0,
        "unresolved_attempts": 0,
        "not_started": 0,
        "duplicate_response_ids": [],
        "worker_errors": [],
        "completed": True,
        "failure_policy": "stop refill, drain owned in-flight requests, preserve all started rows, no retries",
        "timing_scope": "client-observed single-token SSE delivery; not CUDA or scheduler commit time",
    }
    for key, expected in expected_phase.items():
        if phase.get(key) != expected:
            raise SummaryError(f"{label} server warmup phase differs at {key!r}")
    phase_started_ns = _require_integer(
        phase.get("phase_started_ns"), f"{label} server warmup phase.phase_started_ns", minimum=0
    )
    phase_finished_ns = _require_integer(
        phase.get("phase_finished_ns"), f"{label} server warmup phase.phase_finished_ns", minimum=phase_started_ns
    )
    phase_wall_ns = _require_integer(
        phase.get("phase_wall_ns"), f"{label} server warmup phase.phase_wall_ns", minimum=1
    )
    if phase_finished_ns <= phase_started_ns or phase_finished_ns - phase_started_ns != phase_wall_ns:
        raise SummaryError(f"{label} server warmup phase timestamps do not bind its wall interval")
    if not all(
        phase_started_ns <= row["call_started_ns"] <= row["started_ns"]
        <= row["finished_ns"] <= row["call_finished_ns"] <= phase_finished_ns
        for row in replayed_rows
    ):
        raise SummaryError(f"{label} server warmup phase does not enclose its strict rows")
    observed_overlap = _overlap_peak_from_rows(
        replayed_rows,
        start_key="started_ns",
        end_key="finished_ns",
        label=f"{label} server warmup request rows",
    )
    if (
        phase.get("observed_max_request_in_flight") != observed_overlap
        or observed_overlap != record["offered_concurrency"]
    ):
        raise SummaryError(f"{label} server warmup observed concurrency differs from strict rows")
    return {
        "request_rows": {"path": str(request_path), "sha256": request_sha256, "row_count": len(rows)},
        "phase": {
            "path": str(phase_path),
            "sha256": phase_sha256,
            "phase_started_ns": phase_started_ns,
            "phase_finished_ns": phase_finished_ns,
            "phase_wall_ns": phase_wall_ns,
        },
    }


def _validate_retained_artifacts(
    *, lane: str, fields: Mapping[str, str], record: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """Verify the immutable raw evidence named by a successful serving marker."""
    request_path, request_payload, request_sha256 = _read_marked_artifact(
        path_text=fields["request_rows_path"],
        expected_sha256=fields["request_rows_sha256"],
        label=f"{label} retained request rows",
        maximum_bytes=MAX_RETAINED_REQUEST_ROWS_BYTES,
    )
    phase_path, phase_payload, phase_sha256 = _read_marked_artifact(
        path_text=fields["phase_path"],
        expected_sha256=fields["phase_sha256"],
        label=f"{label} retained phase",
        maximum_bytes=MAX_RETAINED_PHASE_BYTES,
    )
    config_path, config_payload, config_sha256 = _read_marked_artifact(
        path_text=fields["attempt_config_path"],
        expected_sha256=fields["attempt_config_sha256"],
        label=f"{label} attempt config",
        maximum_bytes=MAX_ATTEMPT_CONFIG_BYTES,
    )
    if request_path.parent != phase_path.parent or phase_path.parent != config_path.parent:
        raise SummaryError(f"{label} retained artifacts must share one attempt directory")
    if request_path.name != f"{lane}.requests.jsonl":
        raise SummaryError(f"{label} request_rows_path must name {lane}.requests.jsonl")
    if phase_path.name != f"{lane}.phase.json":
        raise SummaryError(f"{label} phase_path must name {lane}.phase.json")
    if config_path.name != "attempt.config.json":
        raise SummaryError(f"{label} attempt_config_path must name attempt.config.json")

    phase = _decode_json(phase_payload, label=f"{label} retained phase")
    if phase.get("schema_version") != RETAINED_PHASE_SCHEMA_VERSION:
        raise SummaryError(f"{label} retained phase schema differs")
    expected_phase = {
        "phase": "retained",
        "warmup": False,
        "offered_concurrency": record["offered_concurrency"],
        "requested": record["attempted_requests"],
        "attempted": record["attempted_requests"],
        "succeeded": record["successful_requests"],
        "failed": record["failed_requests"],
        "completed": True,
    }
    for key, expected in expected_phase.items():
        if phase.get(key) != expected:
            raise SummaryError(f"{label} retained phase differs at {key!r}")
    phase_wall_ns = _require_integer(
        phase.get("phase_wall_ns"), f"{label} retained phase.phase_wall_ns", minimum=1
    )

    config = _decode_json(config_payload, label=f"{label} attempt config")
    if config.get("schema_version") != ATTEMPT_ARTIFACT_SCHEMA_VERSION:
        raise SummaryError(f"{label} attempt config schema differs")
    if config.get("phase") != "timed":
        raise SummaryError(f"{label} attempt config must be a timed outer attempt")
    if config.get("pair_order") != record["pair_order"]:
        raise SummaryError(f"{label} attempt config pair_order differs from marker")
    config_index = _require_integer(config.get("index"), f"{label} attempt config.index", minimum=1)
    workload = _require_mapping(config.get("workload"), f"{label} attempt config.workload")
    configuration = _require_mapping(config.get("configuration"), f"{label} attempt config.configuration")
    expected_workload = {
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "prompt_tokens": record["prompt_tokens"],
        "max_output_tokens": record["max_output_tokens"],
    }
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            raise SummaryError(f"{label} attempt config workload differs at {key!r}")
    for key in ("concurrency", "batch_token_budget", "max_model_len"):
        expected = record["offered_concurrency"] if key == "concurrency" else record[key]
        if configuration.get(key) != expected:
            raise SummaryError(f"{label} attempt config configuration differs at {key!r}")
    if configuration.get("cuda_visible_devices") != "0":
        raise SummaryError(
            f"{label} attempt config must bind Riley and the physical-GPU sampler to CUDA_VISIBLE_DEVICES=0"
        )
    whole_gpu_sampling = _require_mapping(
        configuration.get("whole_gpu_memory_sampling"),
        f"{label} attempt config.configuration.whole_gpu_memory_sampling",
    )
    if whole_gpu_sampling.get("gpu_index") != 0:
        raise SummaryError(f"{label} attempt config whole-GPU sampling must use physical GPU 0")
    if whole_gpu_sampling.get("peak_limit_bytes") != record["whole_gpu_sampled_peak_limit_bytes"]:
        raise SummaryError(f"{label} attempt config whole-GPU peak limit differs from marker")
    interval = _require_finite_number(
        whole_gpu_sampling.get("sampling_interval_seconds"),
        f"{label} attempt config whole-GPU sampling interval",
    )
    if not 0.25 <= interval <= 0.5:
        raise SummaryError(f"{label} attempt config whole-GPU sampling interval must be 0.25..0.5 seconds")
    if whole_gpu_sampling.get("max_sample_duration_seconds") != MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS:
        raise SummaryError(f"{label} attempt config whole-GPU sample-duration bound differs")
    expected_max_start_gap = interval + GPU_MEMORY_SAMPLE_SCHEDULING_SLACK_SECONDS
    if not math.isclose(
        _require_finite_number(
            whole_gpu_sampling.get("max_start_gap_seconds"),
            f"{label} attempt config whole-GPU max start gap",
            positive=True,
        ),
        expected_max_start_gap,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SummaryError(f"{label} attempt config whole-GPU cadence bound differs")
    launch_provenance = _require_mapping(
        config.get("launch_provenance"), f"{label} attempt config.launch_provenance"
    )
    if launch_provenance.get("vllm_image_digest") != record["vllm_image_digest"]:
        raise SummaryError(f"{label} attempt config vLLM image digest differs from marker")
    model_identity = _validate_model_identity_artifacts(
        fields=fields,
        config=config,
        record=record,
        attempt_dir=config_path.parent,
        label=label,
    )
    raw_vllm_argv = config.get("vllm_argv")
    if not isinstance(raw_vllm_argv, list) or not raw_vllm_argv or not all(
        isinstance(item, str) and item for item in raw_vllm_argv
    ):
        raise SummaryError(f"{label} attempt config.vllm_argv must be a nonempty argv list")
    vllm_argv = list(raw_vllm_argv)
    if len(vllm_argv) < 4 or vllm_argv[1] != "run":
        raise SummaryError(f"{label} attempt config vLLM argv must use an owned Docker run command")
    if vllm_argv.count("--rm") != 1:
        raise SummaryError(f"{label} attempt config must retain exactly one Docker --rm")
    if any(token in {"-d", "--detach"} or token.startswith("--detach=") for token in vllm_argv):
        raise SummaryError(f"{label} attempt config must not detach its owned vLLM Docker container")
    if any(token.startswith("--name=") for token in vllm_argv):
        raise SummaryError(f"{label} attempt config must retain Docker --name as a separate owned argv pair")
    if vllm_argv.count("--no-enable-prefix-caching") != 1:
        raise SummaryError(f"{label} attempt config must disable vLLM prefix caching exactly once")
    if vllm_argv.count("--enable-chunked-prefill") != 1:
        raise SummaryError(f"{label} attempt config must enable vLLM chunked prefill exactly once")
    if any(token == "--attention-backend" or token.startswith("--attention-backend=") for token in vllm_argv):
        raise SummaryError(f"{label} attempt config must leave the vLLM attention backend auto-selected")
    if (
        vllm_argv.count("--network") != 1
        or any(token.startswith("--network=") for token in vllm_argv)
        or _argv_option_values(vllm_argv, "--network", label=f"{label} attempt config.vllm_argv") != ["host"]
    ):
        raise SummaryError(f"{label} attempt config must use exactly one separate Docker --network host argv pair")
    expected_options = {
        "--dtype": ["bfloat16"],
        "--kv-cache-dtype": ["bfloat16"],
        "--ipc": ["host"],
        "--gpus": ["device=0"],
        "--model": ["/model"],
        "--served-model-name": [record["model_id"]],
        "--gpu-memory-utilization": [record["vllm_gpu_memory_utilization"]],
        "--max-model-len": [str(record["max_model_len"])],
        "--max-num-seqs": [str(record["offered_concurrency"])],
        "--max-num-batched-tokens": [str(record["batch_token_budget"])],
    }
    for flag, expected in expected_options.items():
        if _argv_option_values(vllm_argv, flag, label=f"{label} attempt config.vllm_argv") != expected:
            raise SummaryError(f"{label} attempt config vLLM argv differs at {flag!r}")
    expected_image = f"vllm/vllm-openai@{record['vllm_image_digest']}"
    if [item for item in vllm_argv if record["vllm_image_digest"] in item] != [expected_image]:
        raise SummaryError(f"{label} attempt config vLLM argv does not use the exact pinned image argv")
    image_operand_index = _docker_run_image_operand_index(
        vllm_argv, label=f"{label} attempt config.vllm_argv"
    )
    _require_docker_controls_before_image(
        vllm_argv,
        image_operand_index=image_operand_index,
        label=f"{label} attempt config.vllm_argv",
    )
    if vllm_argv[image_operand_index] != expected_image:
        raise SummaryError(
            f"{label} attempt config must place the pinned vLLM image as Docker's first non-option operand"
        )
    mount_values = [
        *_argv_option_values(vllm_argv, "-v", label=f"{label} attempt config.vllm_argv"),
        *_argv_option_values(vllm_argv, "--volume", label=f"{label} attempt config.vllm_argv"),
    ]
    if mount_values != [model_identity["model_path"] + ":/model:ro"]:
        raise SummaryError(f"{label} attempt config vLLM argv does not mount exactly the pinned model read-only")
    raw_riley_argv = config.get("riley_argv")
    if not isinstance(raw_riley_argv, list) or not raw_riley_argv:
        raise SummaryError(f"{label} attempt config.riley_argv is invalid")
    if _argv_option_values(raw_riley_argv, "--model", label=f"{label} attempt config.riley_argv") != [model_identity["model_path"]]:
        raise SummaryError(f"{label} attempt config Riley argv does not use the pinned model path")
    if _argv_option_values(raw_riley_argv, "--model-id", label=f"{label} attempt config.riley_argv") != [record["model_id"]]:
        raise SummaryError(f"{label} attempt config Riley argv does not use the pinned served model id")
    docker_container = _require_exact_keys(
        launch_provenance.get("vllm_docker_container"),
        {"name", "detached", "cleanup_schema_version"},
        f"{label} attempt config.vllm_docker_container",
    )
    container_name = _require_string(docker_container.get("name"), f"{label} vLLM Docker container name")
    expected_name = (
        f"riley-n06a-vllm-timed-{config_index:04d}-"
        f"{hashlib.sha256(str(config_path.parent).encode('utf-8')).hexdigest()[:20]}"
    )
    if (
        not DOCKER_CONTAINER_NAME_RE.fullmatch(container_name)
        or container_name != expected_name
        or docker_container.get("detached") is not False
        or docker_container.get("cleanup_schema_version") != DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION
    ):
        raise SummaryError(f"{label} attempt config vLLM Docker container contract differs")
    if _argv_option_values(vllm_argv, "--name", label=f"{label} attempt config.vllm_argv") != [container_name]:
        raise SummaryError(f"{label} attempt config must bind exactly one deterministic Docker --name")
    vllm_fairness = _require_mapping(
        launch_provenance.get("vllm_fixed_prompt_fairness"),
        f"{label} attempt config.vllm_fixed_prompt_fairness",
    )
    expected_fairness = {
        "prefix_caching": record["vllm_prefix_caching"],
        "docker_gpu_binding": "device=0",
        "gpu_memory_utilization": record["vllm_gpu_memory_utilization"],
        "whole_gpu_sampled_peak_limit_bytes": record["whole_gpu_sampled_peak_limit_bytes"],
    }
    for key, expected in expected_fairness.items():
        if vllm_fairness.get(key) != expected:
            raise SummaryError(f"{label} attempt config vLLM fairness differs at {key!r}")

    _validate_pinned_token_client(config, label=label)
    reference, workload_document = _load_pinned_n06a_reference(
        workload_config=workload,
        record=record,
        label=label,
    )
    request_timeout_seconds = _require_finite_number(
        configuration.get("request_timeout_seconds"),
        f"{label} attempt config configuration.request_timeout_seconds",
        positive=True,
    )

    rows = _decode_jsonl_rows(request_payload, label=f"{label} retained request rows")
    if len(rows) != record["successful_requests"]:
        raise SummaryError(f"{label} retained request row count differs from marker")
    seen_indices: set[int] = set()
    replayed_rows: list[Mapping[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_label = f"{label} retained request rows[{row_index}]"
        if row.get("phase") != "retained" or row.get("warmup") is not False:
            raise SummaryError(f"{row_label} is not a retained successful request")
        index = _require_integer(row.get("index"), f"{row_label}.index", minimum=0)
        if index in seen_indices:
            raise SummaryError(f"{row_label}.index is duplicated")
        seen_indices.add(index)
        replayed_rows.append(
            _replay_retained_row(
                row=row,
                row_label=row_label,
                reference=reference,
                workload_document=workload_document,
                request_timeout_seconds=request_timeout_seconds,
            )
        )
    if seen_indices != set(range(record["successful_requests"])):
        raise SummaryError(f"{label} retained request indices must cover the completed request range")
    response_ids = [str(row["response_id"]) for row in replayed_rows]
    if len(set(response_ids)) != len(response_ids):
        raise SummaryError(f"{label} retained request rows repeat a response identity")

    phase_started_ns = _require_integer(
        phase.get("phase_started_ns"), f"{label} retained phase.phase_started_ns", minimum=0
    )
    phase_finished_ns = _require_integer(
        phase.get("phase_finished_ns"), f"{label} retained phase.phase_finished_ns", minimum=phase_started_ns
    )
    if phase_finished_ns <= phase_started_ns or phase_finished_ns - phase_started_ns != phase_wall_ns:
        raise SummaryError(f"{label} retained phase wall interval differs from phase timestamps")
    if phase.get("unresolved_attempts") != 0 or phase.get("not_started") != 0:
        raise SummaryError(f"{label} retained phase must account for every requested request")
    if phase.get("duplicate_response_ids") != [] or phase.get("worker_errors") != []:
        raise SummaryError(f"{label} retained phase records duplicate identities or worker errors")
    if not all(
        phase_started_ns <= row["call_started_ns"] <= row["started_ns"]
        <= row["finished_ns"] <= row["call_finished_ns"] <= phase_finished_ns
        for row in replayed_rows
    ):
        raise SummaryError(f"{label} retained phase does not enclose its replayed request intervals")
    observed_overlap = _overlap_peak_from_rows(
        replayed_rows,
        start_key="started_ns",
        end_key="finished_ns",
        label=f"{label} retained request rows",
    )
    if (
        phase.get("observed_max_request_in_flight") != observed_overlap
        or observed_overlap != record["offered_concurrency"]
    ):
        raise SummaryError(f"{label} retained phase observed concurrency differs from replayed request intervals")
    if phase.get("timing_scope") != "client-observed single-token SSE delivery; not CUDA or scheduler commit time":
        raise SummaryError(f"{label} retained phase timing scope differs from the N06-A client contract")
    if phase.get("failure_policy") != "stop refill, drain owned in-flight requests, preserve all started rows, no retries":
        raise SummaryError(f"{label} retained phase failure policy differs from the N06-A client contract")
    if fields["retained_wall_ms"] != f"{phase_wall_ns / 1_000_000.0:.6f}":
        raise SummaryError(f"{label} retained phase wall time differs from marker")
    replayed_output_tokens = len(replayed_rows) * len(reference.output_token_ids)
    if replayed_output_tokens != record["output_tokens"]:
        raise SummaryError(f"{label} replayed output token total differs from marker")
    replayed_throughput = replayed_output_tokens * 1_000_000_000.0 / phase_wall_ns
    if fields["output_tokens_per_second"] != f"{replayed_throughput:.6f}":
        raise SummaryError(f"{label} replayed throughput differs from marker")
    for prefix, metric_key in (
        ("token_ttft", "token_ttft_ns"),
        ("token_tpot", "token_tpot_ns"),
        ("e2e", "e2e_ns"),
    ):
        values_ms = [float(row["metrics"][metric_key]) / 1_000_000.0 for row in replayed_rows]
        for statistic, probability in (("median", 0.5), ("p95", 0.95), ("p99", 0.99)):
            marker_key = f"{prefix}_{statistic}_ms"
            expected_text = f"{_r7_quantile(values_ms, probability):.6f}"
            if fields[marker_key] != expected_text:
                raise SummaryError(
                    f"{label} {marker_key} differs from R7 replay of retained strict rows"
                )
    expected_p99_status = "qualified" if len(replayed_rows) >= P99_QUALIFIED_MINIMUM_REQUESTS else "descriptive"
    if fields["p99_status"] != expected_p99_status:
        raise SummaryError(f"{label}.p99_status differs from the replayed retained-row count")
    server_warmup = _validate_server_warmup_artifacts(
        lane=lane,
        fields=fields,
        attempt_dir=config_path.parent,
        record=record,
        configuration=configuration,
        reference=reference,
        workload_document=workload_document,
        request_timeout_seconds=request_timeout_seconds,
        label=label,
    )
    return {
        "request_rows": {
            "path": str(request_path),
            "sha256": request_sha256,
            "row_count": len(rows),
            "reference_sha256": reference.sha256,
        },
        "phase": {
            "path": str(phase_path),
            "sha256": phase_sha256,
            "phase_started_ns": phase_started_ns,
            "phase_finished_ns": phase_finished_ns,
            "phase_wall_ns": phase_wall_ns,
        },
        "server_warmup": server_warmup,
        "model_identity": model_identity,
        "attempt_config": {
            "path": str(config_path),
            "sha256": config_sha256,
            "timed_index": config_index,
            "pair_order": config["pair_order"],
        },
    }


def _validate_whole_gpu_peak_receipt(
    *, lane: str, fields: Mapping[str, str], record: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """Verify the lane's independent whole-GPU sampled peak receipt."""
    sample_path, sample_payload, sample_sha256 = _read_marked_artifact(
        path_text=fields["whole_gpu_sample_path"],
        expected_sha256=fields["whole_gpu_sample_sha256"],
        label=f"{label} whole-GPU samples",
        maximum_bytes=MAX_GPU_MEMORY_SAMPLES_BYTES,
    )
    peak_path, peak_payload, peak_sha256 = _read_marked_artifact(
        path_text=fields["whole_gpu_peak_receipt_path"],
        expected_sha256=fields["whole_gpu_peak_receipt_sha256"],
        label=f"{label} whole-GPU peak receipt",
        maximum_bytes=MAX_RETAINED_PHASE_BYTES,
    )
    if sample_path.parent != peak_path.parent:
        raise SummaryError(f"{label} whole-GPU samples and peak receipt must share one lane directory")
    if sample_path.name != f"{lane}.whole-gpu-memory.samples.jsonl":
        raise SummaryError(f"{label} whole_gpu_sample_path has an unexpected lane name")
    if peak_path.name != f"{lane}.whole-gpu-memory.peak.json":
        raise SummaryError(f"{label} whole_gpu_peak_receipt_path has an unexpected lane name")
    samples = _decode_jsonl_rows(sample_payload, label=f"{label} whole-GPU samples")
    peak = _decode_json(peak_payload, label=f"{label} whole-GPU peak receipt")
    if len(samples) < 2:
        raise SummaryError(f"{label} whole-GPU sampled peak requires at least two samples")
    memory_values: list[int] = []
    previous_started_ns: int | None = None
    previous_finished_ns: int | None = None
    expected_interval_seconds = _require_finite_number(
        peak.get("sampling_interval_seconds"),
        f"{label} whole-GPU peak receipt.sampling_interval_seconds",
        positive=True,
    )
    if not 0.25 <= expected_interval_seconds <= 0.5:
        raise SummaryError(f"{label} whole-GPU peak receipt interval must be 0.25..0.5 seconds")
    if peak.get("max_sample_duration_seconds") != MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS:
        raise SummaryError(f"{label} whole-GPU peak receipt sample-duration bound differs")
    expected_max_start_gap_seconds = expected_interval_seconds + GPU_MEMORY_SAMPLE_SCHEDULING_SLACK_SECONDS
    if not math.isclose(
        _require_finite_number(
            peak.get("max_start_gap_seconds"),
            f"{label} whole-GPU peak receipt.max_start_gap_seconds",
            positive=True,
        ),
        expected_max_start_gap_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SummaryError(f"{label} whole-GPU peak receipt cadence bound differs")
    maximum_sample_duration_ns = int(MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS * 1_000_000_000)
    maximum_start_gap_ns = int(expected_max_start_gap_seconds * 1_000_000_000)
    for index, sample in enumerate(samples):
        started_ns = _require_integer(
            sample.get("started_ns"), f"{label} whole-GPU samples[{index}].started_ns", minimum=0
        )
        finished_ns = _require_integer(
            sample.get("finished_ns"), f"{label} whole-GPU samples[{index}].finished_ns", minimum=0
        )
        timestamp = _require_integer(
            sample.get("monotonic_ns"), f"{label} whole-GPU samples[{index}].monotonic_ns", minimum=0
        )
        used_bytes = _require_integer(
            sample.get("memory_used_bytes"), f"{label} whole-GPU samples[{index}].memory_used_bytes", minimum=0
        )
        if finished_ns < started_ns or timestamp != finished_ns:
            raise SummaryError(f"{label} whole-GPU sample {index} has an invalid observation interval")
        if finished_ns - started_ns > maximum_sample_duration_ns:
            raise SummaryError(f"{label} whole-GPU sample {index} exceeded its duration bound")
        if previous_started_ns is not None:
            if started_ns < previous_started_ns or (
                previous_finished_ns is not None and started_ns < previous_finished_ns
            ):
                raise SummaryError(f"{label} whole-GPU sample clocks overlap or move backwards")
            if started_ns - previous_started_ns > maximum_start_gap_ns:
                raise SummaryError(f"{label} whole-GPU sample cadence has an unobserved gap")
        previous_started_ns = started_ns
        previous_finished_ns = finished_ns
        memory_values.append(used_bytes)
    observed_peak = max(memory_values)
    if observed_peak != record["whole_gpu_sampled_peak_bytes"]:
        raise SummaryError(f"{label} whole-GPU sample peak differs from marker")
    if observed_peak > record["whole_gpu_sampled_peak_limit_bytes"]:
        raise SummaryError(f"{label} whole-GPU sample peak exceeds the strict limit")
    if peak.get("schema_version") != "riley.n06a-whole-gpu-memory-peak.v1":
        raise SummaryError(f"{label} whole-GPU peak receipt schema differs")
    lifecycle = _require_mapping(peak.get("lifecycle"), f"{label} whole-GPU peak receipt.lifecycle")
    server_started_ns = _require_integer(
        lifecycle.get("server_started_ns"), f"{label} whole-GPU peak receipt lifecycle.server_started_ns", minimum=0
    )
    server_cleanup_completed_ns = _require_integer(
        lifecycle.get("server_cleanup_completed_ns"),
        f"{label} whole-GPU peak receipt lifecycle.server_cleanup_completed_ns",
        minimum=server_started_ns,
    )
    actual_lifecycle = {
        "sample_started_before_server_launch": samples[0]["started_ns"] <= server_started_ns,
        "sample_overlapped_server_lifetime": any(
            sample["started_ns"] <= server_cleanup_completed_ns
            and sample["finished_ns"] >= server_started_ns
            for sample in samples
        ),
        "sample_finished_after_server_cleanup": samples[-1]["finished_ns"] >= server_cleanup_completed_ns,
    }
    for key, expected in actual_lifecycle.items():
        if lifecycle.get(key) is not expected:
            raise SummaryError(f"{label} whole-GPU peak receipt lifecycle differs at {key!r}")
        if not expected:
            raise SummaryError(f"{label} whole-GPU samples do not cover the owned server lifecycle")
    expected_peak = {
        "lane": lane,
        "gpu_index": 0,
        "sample_count": len(samples),
        "sample_path": str(sample_path),
        "sample_sha256": sample_sha256,
        "peak_used_bytes": observed_peak,
        "peak_limit_bytes": record["whole_gpu_sampled_peak_limit_bytes"],
        "compliant": True,
        "errors": [],
    }
    for key, expected in expected_peak.items():
        if peak.get(key) != expected:
            raise SummaryError(f"{label} whole-GPU peak receipt differs at {key!r}")
    return {
        "sample_path": str(sample_path),
        "sample_sha256": sample_sha256,
        "sample_count": len(samples),
        "final_sample_finished_ns": samples[-1]["finished_ns"],
        "peak_path": str(peak_path),
        "peak_sha256": peak_sha256,
        "peak_used_bytes": observed_peak,
        "peak_limit_bytes": record["whole_gpu_sampled_peak_limit_bytes"],
        "sampling_interval_seconds": expected_interval_seconds,
        "max_sample_duration_seconds": MAX_GPU_MEMORY_SAMPLE_DURATION_SECONDS,
        "max_start_gap_seconds": expected_max_start_gap_seconds,
        "lifecycle": {
            "server_started_ns": server_started_ns,
            "server_cleanup_completed_ns": server_cleanup_completed_ns,
            **actual_lifecycle,
        },
    }



def _validate_post_lane_gpu_idle_census(
    *,
    lane: str,
    provenance: Mapping[str, Any],
    configuration: Mapping[str, Any],
    cleanup_completed_ns: int,
    label: str,
) -> dict[str, Any]:
    """Replay the post-lane GPU-0 census retained inside the hashed provenance.

    This is deliberately separate from the in-lane sampled peak.  The latter
    proves the cap during an owned server lifetime; this bounded census proves
    that the lane did not leave an allocation or compute process that could
    bias the next AB/BA lane.
    """
    sampling = _require_mapping(
        configuration.get("whole_gpu_memory_sampling"),
        f"{label} attempt config configuration.whole_gpu_memory_sampling",
    )
    nvidia_smi = _require_string(
        sampling.get("nvidia_smi"),
        f"{label} attempt config whole-GPU nvidia_smi",
    )
    receipt = _require_mapping(
        provenance.get("post_lane_gpu_idle"),
        f"{label} lane provenance.post_lane_gpu_idle",
    )
    expected = {
        "schema_version": GPU_IDLE_CENSUS_SCHEMA_VERSION,
        "gpu_index": 0,
        "max_idle_memory_bytes": GPU_IDLE_CENSUS_MAX_USED_BYTES,
        "compute_process_pids": [],
        "compliant": True,
        "errors": [],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise SummaryError(f"{label} post-lane GPU idle census differs at {key!r}")
    memory_used_bytes = _require_integer(
        receipt.get("memory_used_bytes"),
        f"{label} post-lane GPU idle census.memory_used_bytes",
        minimum=0,
    )
    if memory_used_bytes > GPU_IDLE_CENSUS_MAX_USED_BYTES:
        raise SummaryError(f"{label} post-lane GPU idle census exceeds the 512MiB bound")
    commands = receipt.get("commands")
    if not isinstance(commands, list) or len(commands) != 2:
        raise SummaryError(f"{label} post-lane GPU idle census requires exactly two commands")
    expected_suffixes = (
        ["--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        ["--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    )
    parsed_processes: list[int] | None = None
    parsed_memory_bytes: int | None = None
    previous_finished_ns = cleanup_completed_ns
    command_summaries: list[dict[str, Any]] = []
    for index, (command, suffix) in enumerate(zip(commands, expected_suffixes, strict=True)):
        command_mapping = _require_mapping(
            command, f"{label} post-lane GPU idle command[{index}]"
        )
        if command_mapping.get("argv") != [nvidia_smi, *suffix]:
            raise SummaryError(f"{label} post-lane GPU idle command[{index}] argv differs")
        if command_mapping.get("returncode") != 0 or "error" in command_mapping:
            raise SummaryError(f"{label} post-lane GPU idle command[{index}] did not succeed")
        timeout = _require_finite_number(
            command_mapping.get("timeout_seconds"),
            f"{label} post-lane GPU idle command[{index}].timeout_seconds",
            positive=True,
        )
        if timeout > GPU_IDLE_CENSUS_COMMAND_TIMEOUT_SECONDS:
            raise SummaryError(f"{label} post-lane GPU idle command[{index}] timeout is unbounded")
        started_ns = _require_integer(
            command_mapping.get("started_ns"),
            f"{label} post-lane GPU idle command[{index}].started_ns",
            minimum=cleanup_completed_ns,
        )
        finished_ns = _require_integer(
            command_mapping.get("finished_ns"),
            f"{label} post-lane GPU idle command[{index}].finished_ns",
            minimum=started_ns,
        )
        if started_ns < previous_finished_ns:
            raise SummaryError(f"{label} post-lane GPU idle commands overlap or precede cleanup")
        previous_finished_ns = finished_ns
        streams: dict[str, str] = {}
        for stream in ("stdout", "stderr"):
            text = command_mapping.get(f"{stream}_text")
            if not isinstance(text, str):
                raise SummaryError(f"{label} post-lane GPU idle command[{index}] lacks {stream} text")
            raw = text.encode("utf-8")
            if (
                command_mapping.get(f"{stream}_bytes") != len(raw)
                or command_mapping.get(f"{stream}_sha256") != hashlib.sha256(raw).hexdigest()
            ):
                raise SummaryError(
                    f"{label} post-lane GPU idle command[{index}] {stream} receipt differs"
                )
            streams[stream] = text
        if index == 0:
            rows = [item.strip() for item in streams["stdout"].splitlines() if item.strip()]
            if not all(re.fullmatch(r"[1-9][0-9]*", item) for item in rows):
                raise SummaryError(f"{label} post-lane GPU idle compute-process output is malformed")
            parsed_processes = [int(item, 10) for item in rows]
        else:
            rows = [item.strip() for item in streams["stdout"].splitlines() if item.strip()]
            if len(rows) != 1 or not re.fullmatch(r"[0-9]+(?:\s+MiB)?", rows[0]):
                raise SummaryError(f"{label} post-lane GPU idle memory output is malformed")
            parsed_memory_bytes = int(rows[0].split()[0], 10) * 1024 * 1024
        command_summaries.append(
            {"argv": list(command_mapping["argv"]), "started_ns": started_ns, "finished_ns": finished_ns}
        )
    if parsed_processes != [] or parsed_memory_bytes != memory_used_bytes:
        raise SummaryError(f"{label} post-lane GPU idle summary differs from raw nvidia-smi output")
    return {
        "gpu_index": 0,
        "memory_used_bytes": memory_used_bytes,
        "commands": command_summaries,
    }

def _validate_lane_psi_configuration(
    configuration: Mapping[str, Any], *, label: str
) -> str:
    value = _require_mapping(configuration.get("lane_psi"), f"{label}.lane_psi")
    expected_keys = {"schema_version", "proc_root", "resources", "policy"}
    if set(value) != expected_keys:
        raise SummaryError(f"{label}.lane_psi fields differ from the N06-A contract")
    if value.get("schema_version") != LANE_PSI_SCHEMA_VERSION:
        raise SummaryError(f"{label}.lane_psi schema version differs")
    if value.get("policy") != LANE_PSI_POLICY:
        raise SummaryError(f"{label}.lane_psi policy differs")
    proc_root = _require_string(value.get("proc_root"), f"{label}.lane_psi.proc_root")
    if proc_root != "/proc":
        raise SummaryError(f"{label}.lane_psi.proc_root must be /proc")
    resources = value.get("resources")
    if resources != list(PSI_RESOURCES):
        raise SummaryError(f"{label}.lane_psi.resources must be CPU/I/O/memory in contract order")
    return proc_root


def _validate_lane_psi_snapshot(
    value: object,
    *,
    phase: str,
    proc_root: str,
    label: str,
) -> dict[str, Any]:
    snapshot = _require_exact_keys(
        value,
        {
            "schema_version",
            "policy",
            "snapshot_started_ns",
            "snapshot_finished_ns",
            "psi",
        },
        label,
    )
    if snapshot.get("schema_version") != LANE_PSI_SCHEMA_VERSION:
        raise SummaryError(f"{label}.schema_version differs")
    if snapshot.get("policy") != LANE_PSI_POLICY:
        raise SummaryError(f"{label}.policy differs")
    started_ns = _require_integer(snapshot.get("snapshot_started_ns"), f"{label}.snapshot_started_ns", minimum=0)
    finished_ns = _require_integer(
        snapshot.get("snapshot_finished_ns"),
        f"{label}.snapshot_finished_ns",
        minimum=started_ns,
    )
    psi = _require_mapping(snapshot.get("psi"), f"{label}.psi")
    if set(psi) != set(PSI_RESOURCES):
        raise SummaryError(f"{label}.psi must contain CPU/I/O/memory exactly once")
    return {
        "snapshot_started_ns": started_ns,
        "snapshot_finished_ns": finished_ns,
        "psi": {
            resource: _validate_psi_resource(
                psi.get(resource),
                label=f"{label}.psi.{resource}",
                expected_source=str(Path(proc_root) / "pressure" / resource),
            )
            for resource in PSI_RESOURCES
        },
        "phase": phase,
    }


def _validate_lane_pressure_artifact(
    *,
    lane: str,
    provenance: Mapping[str, Any],
    attempt_dir: Path,
    attempt_config: Mapping[str, Any],
    configuration: Mapping[str, Any],
    server_started_ns: int,
    cleanup_completed_ns: int,
    final_whole_gpu_sample_finished_ns: int,
    first_post_lane_gpu_idle_started_ns: int,
    label: str,
) -> dict[str, Any]:
    """Replay the marker-bound PSI artifact without treating pressure as a gate."""
    proc_root = _validate_lane_psi_configuration(configuration, label=label)
    reference = _require_exact_keys(
        provenance.get("lane_pressure"),
        {"path", "sha256", "policy"},
        f"{label} lane provenance.lane_pressure",
    )
    if reference.get("policy") != LANE_PSI_POLICY:
        raise SummaryError(f"{label} lane provenance PSI policy differs")
    artifact_path, artifact_payload, artifact_sha256 = _read_marked_artifact(
        path_text=_require_string(reference.get("path"), f"{label} lane pressure path"),
        expected_sha256=_require_string(reference.get("sha256"), f"{label} lane pressure SHA-256"),
        label=f"{label} lane PSI artifact",
        maximum_bytes=MAX_LANE_PSI_ARTIFACT_BYTES,
    )
    if artifact_path.parent != attempt_dir or artifact_path.name != f"{lane}.lane-psi.json":
        raise SummaryError(f"{label} lane PSI artifact must be the dedicated lane file in its attempt directory")
    artifact = _require_exact_keys(
        _decode_json(artifact_payload, label=f"{label} lane PSI artifact"),
        {"schema_version", "policy", "lane", "attempt", "proc_root", "pre", "post"},
        f"{label} lane PSI artifact",
    )
    if (
        artifact.get("schema_version") != LANE_PSI_SCHEMA_VERSION
        or artifact.get("policy") != LANE_PSI_POLICY
        or artifact.get("lane") != lane
    ):
        raise SummaryError(f"{label} lane PSI artifact identity differs")
    if artifact.get("proc_root") != proc_root:
        raise SummaryError(f"{label} lane PSI artifact proc_root differs from its attempt configuration")
    expected_attempt = {
        "phase": attempt_config.get("phase"),
        "index": attempt_config.get("index"),
        "pair_order": attempt_config.get("pair_order"),
    }
    if artifact.get("attempt") != expected_attempt:
        raise SummaryError(f"{label} lane PSI artifact attempt identity differs from its attempt config")
    pre = _validate_lane_psi_snapshot(
        artifact.get("pre"), phase="pre", proc_root=proc_root, label=f"{label} lane PSI pre"
    )
    post = _validate_lane_psi_snapshot(
        artifact.get("post"), phase="post", proc_root=proc_root, label=f"{label} lane PSI post"
    )
    if pre["snapshot_finished_ns"] > server_started_ns:
        raise SummaryError(f"{label} lane PSI pre snapshot falls after server launch")
    if post["snapshot_started_ns"] < cleanup_completed_ns:
        raise SummaryError(f"{label} lane PSI post snapshot falls before owned cleanup")
    if post["snapshot_started_ns"] < final_whole_gpu_sample_finished_ns:
        raise SummaryError(f"{label} lane PSI post snapshot falls before final whole-GPU sampler completion")
    if post["snapshot_finished_ns"] > first_post_lane_gpu_idle_started_ns:
        raise SummaryError(f"{label} lane PSI post snapshot extends past post-lane GPU idle census")
    if post["snapshot_started_ns"] < pre["snapshot_finished_ns"]:
        raise SummaryError(f"{label} lane PSI snapshots move backwards")
    snapshot_window_us = (post["snapshot_started_ns"] - pre["snapshot_finished_ns"]) / 1_000.0
    if snapshot_window_us <= 0.0:
        raise SummaryError(f"{label} lane PSI snapshot window must be positive")
    derived: dict[str, dict[str, Any]] = {}
    for resource in PSI_RESOURCES:
        resource_derived: dict[str, Any] = {}
        for category in ("some", "full"):
            pre_metrics = pre["psi"][resource][category]
            post_metrics = post["psi"][resource][category]
            if pre_metrics is None or post_metrics is None:
                resource_derived[category] = None
                continue
            total_delta_us = int(post_metrics["total"]) - int(pre_metrics["total"])
            if total_delta_us < 0:
                raise SummaryError(f"{label} lane PSI {resource}.{category}.total moves backwards")
            resource_derived[category] = {
                "pre_avg10_percent": float(pre_metrics["avg10"]),
                "post_avg10_percent": float(post_metrics["avg10"]),
                "total_delta_us": total_delta_us,
                "snapshot_window_us": snapshot_window_us,
                "stall_percent": total_delta_us * 100.0 / snapshot_window_us,
            }
        derived[resource] = resource_derived
    return {
        "path": str(artifact_path),
        "sha256": artifact_sha256,
        "pre": pre,
        "post": post,
        "derived": derived,
    }


def _validate_lane_provenance(
    *, lane: str, fields: Mapping[str, str], record: Mapping[str, Any], label: str
) -> dict[str, Any]:
    """Bind marker metrics to the owned process cleanup and startup receipts."""
    retained = _require_mapping(record.get("retained_artifacts"), f"{label} retained artifacts")
    attempt_config = _require_mapping(retained.get("attempt_config"), f"{label} retained attempt config")
    attempt_dir = Path(_require_string(attempt_config.get("path"), f"{label} attempt config path")).parent
    provenance_path, provenance_payload, provenance_sha256 = _read_marked_artifact(
        path_text=fields["lane_provenance_path"],
        expected_sha256=fields["lane_provenance_sha256"],
        label=f"{label} lane provenance",
        maximum_bytes=MAX_LANE_PROVENANCE_BYTES,
    )
    if provenance_path.parent != attempt_dir or provenance_path.name != f"{lane}.provenance.json":
        raise SummaryError(f"{label} lane provenance must be the lane receipt in its attempt directory")
    provenance = _decode_json(provenance_payload, label=f"{label} lane provenance")
    config_path, config_payload, _ = _read_marked_artifact(
        path_text=fields["attempt_config_path"],
        expected_sha256=fields["attempt_config_sha256"],
        label=f"{label} attempt config for lane provenance",
        maximum_bytes=MAX_ATTEMPT_CONFIG_BYTES,
    )
    if config_path.parent != attempt_dir:
        raise SummaryError(f"{label} lane provenance must use its marked attempt config")
    config = _decode_json(config_payload, label=f"{label} attempt config for lane provenance")
    configuration = _require_mapping(
        config.get("configuration"), f"{label} attempt config configuration for lane provenance"
    )
    expected_argv = config.get(f"{lane}_argv")
    if not isinstance(expected_argv, list) or not expected_argv or not all(
        isinstance(item, str) and item for item in expected_argv
    ):
        raise SummaryError(f"{label} attempt config {lane}_argv is invalid")
    if provenance.get("argv") != expected_argv:
        raise SummaryError(f"{label} lane provenance argv differs from its attempt config")
    _require_integer(provenance.get("pid"), f"{label} lane provenance.pid", minimum=1)
    started_ns = _require_integer(
        provenance.get("started_ns"), f"{label} lane provenance.started_ns", minimum=0
    )
    cleanup_completed_ns = _require_integer(
        provenance.get("cleanup_completed_ns"),
        f"{label} lane provenance.cleanup_completed_ns",
        minimum=started_ns,
    )
    ready = _require_mapping(provenance.get("ready"), f"{label} lane provenance.ready")
    ready_ns = _require_integer(
        ready.get("ready_ns"), f"{label} lane provenance.ready.ready_ns", minimum=started_ns
    )
    if ready_ns > cleanup_completed_ns:
        raise SummaryError(f"{label} readiness receipt falls after owned cleanup")
    cleanup = _require_mapping(provenance.get("cleanup"), f"{label} lane provenance.cleanup")
    if cleanup.get("lane") != lane or cleanup.get("cleanup_verified") is not True:
        raise SummaryError(f"{label} lane provenance does not prove owned-process cleanup")
    _require_integer(cleanup.get("pid"), f"{label} lane provenance.cleanup.pid", minimum=1)
    _require_integer(cleanup.get("returncode"), f"{label} lane provenance.cleanup.returncode")
    if cleanup.get("errors") != []:
        raise SummaryError(f"{label} lane provenance cleanup has recorded errors")
    docker_container_cleanup: dict[str, Any] | None = None
    if lane == "vllm":
        launch_provenance = _require_mapping(
            config.get("launch_provenance"), f"{label} attempt config.launch_provenance"
        )
        docker_container = _require_mapping(
            launch_provenance.get("vllm_docker_container"),
            f"{label} attempt config.vllm_docker_container",
        )
        expected_container_name = _require_string(
            docker_container.get("name"), f"{label} expected vLLM Docker container name"
        )
        container = _require_mapping(cleanup.get("container"), f"{label} vLLM Docker cleanup receipt")
        if (
            container.get("schema_version") != DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION
            or container.get("container_name") != expected_container_name
            or container.get("launcher") != expected_argv[0]
            or container.get("cleanup_verified") is not True
            or container.get("final_state") != "absent"
            or container.get("errors") != []
        ):
            raise SummaryError(f"{label} vLLM Docker cleanup receipt does not prove daemon container absence")
        commands = container.get("commands")
        if not isinstance(commands, list) or not commands:
            raise SummaryError(f"{label} vLLM Docker cleanup receipt lacks bounded control-plane commands")
        if not isinstance(container.get("notes"), list) or not all(
            isinstance(note, str) for note in container["notes"]
        ):
            raise SummaryError(f"{label} vLLM Docker cleanup receipt notes are invalid")
        command_by_operation: dict[str, Mapping[str, Any]] = {}
        allowed_operations = {
            "inventory-before",
            "stop",
            "wait",
            "inventory-after-stop",
            "force-remove",
            "inventory-after-force-remove",
            "inspect-absence",
        }
        for command_index, command in enumerate(commands):
            command_mapping = _require_mapping(
                command, f"{label} vLLM Docker cleanup command[{command_index}]"
            )
            operation = _require_string(
                command_mapping.get("operation"),
                f"{label} vLLM Docker cleanup command[{command_index}].operation",
            )
            if operation in command_by_operation:
                raise SummaryError(f"{label} vLLM Docker cleanup receipt repeats {operation!r}")
            if operation not in allowed_operations:
                raise SummaryError(f"{label} vLLM Docker cleanup receipt has unsupported operation {operation!r}")
            command_by_operation[operation] = command_mapping
            argv = command_mapping.get("argv")
            if (
                not isinstance(argv, list)
                or not argv
                or argv[0] != expected_argv[0]
                or not all(isinstance(item, str) and item for item in argv)
            ):
                raise SummaryError(f"{label} vLLM Docker cleanup command[{command_index}] argv differs")
            expected_inventory_argv = [
                expected_argv[0],
                "container",
                "ls",
                "--all",
                "--filter",
                f"name=^/{expected_container_name}$",
                "--format",
                "{{.Names}}\t{{.ID}}",
            ]
            expected_inspect_argv = [
                expected_argv[0],
                "container",
                "inspect",
                "--format",
                "{{.Id}}",
                expected_container_name,
            ]
            if operation in {"inventory-before", "inventory-after-stop", "inventory-after-force-remove"}:
                if argv != expected_inventory_argv:
                    raise SummaryError(f"{label} vLLM Docker {operation} argv is not the exact owned-name inventory")
            elif operation == "stop":
                if (
                    len(argv) != 6
                    or argv[:4] != [expected_argv[0], "container", "stop", "--time"]
                    or not re.fullmatch(r"[1-9][0-9]*", argv[4])
                    or not 1 <= int(argv[4], 10) <= 30
                    or argv[5] != expected_container_name
                ):
                    raise SummaryError(f"{label} vLLM Docker stop argv is not bounded to the owned container")
            elif operation == "wait":
                if argv != [expected_argv[0], "container", "wait", expected_container_name]:
                    raise SummaryError(f"{label} vLLM Docker wait argv is not the exact owned container")
            elif operation == "force-remove":
                if argv != [expected_argv[0], "container", "rm", "--force", expected_container_name]:
                    raise SummaryError(f"{label} vLLM Docker force-remove argv is not the exact owned container")
            elif operation == "inspect-absence" and argv != expected_inspect_argv:
                raise SummaryError(f"{label} vLLM Docker inspect argv is not the exact owned container")
            timeout = _require_finite_number(
                command_mapping.get("timeout_seconds"),
                f"{label} vLLM Docker cleanup command[{command_index}].timeout_seconds",
                positive=True,
            )
            if timeout > 30.0:
                raise SummaryError(f"{label} vLLM Docker cleanup command[{command_index}] is not bounded to 30 seconds")
            _require_integer(
                command_mapping.get("returncode"),
                f"{label} vLLM Docker cleanup command[{command_index}].returncode",
            )
            for stream in ("stdout", "stderr"):
                text = command_mapping.get(f"{stream}_text")
                if not isinstance(text, str):
                    raise SummaryError(f"{label} vLLM Docker cleanup command[{command_index}] lacks {stream} text")
                raw = text.encode("utf-8")
                if (
                    command_mapping.get(f"{stream}_bytes") != len(raw)
                    or command_mapping.get(f"{stream}_sha256") != hashlib.sha256(raw).hexdigest()
                ):
                    raise SummaryError(
                        f"{label} vLLM Docker cleanup command[{command_index}] {stream} receipt differs"
                    )
            if "error" in command_mapping:
                raise SummaryError(f"{label} vLLM Docker cleanup command[{command_index}] has an unproven command error")
        for required_operation in ("inventory-before", "inventory-after-stop", "inspect-absence"):
            if required_operation not in command_by_operation:
                raise SummaryError(f"{label} vLLM Docker cleanup receipt lacks {required_operation}")
        before = command_by_operation["inventory-before"]
        after_stop = command_by_operation["inventory-after-stop"]
        inspect = command_by_operation["inspect-absence"]
        if (
            before.get("returncode") != 0
            or after_stop.get("returncode") != 0
            or after_stop.get("stdout_text") != ""
            or inspect.get("returncode") == 0
        ):
            raise SummaryError(f"{label} vLLM Docker cleanup receipt does not prove final exact-name absence")
        before_text = before.get("stdout_text")
        if not isinstance(before_text, str):
            raise SummaryError(f"{label} vLLM Docker inventory-before lacks stdout")
        if before_text:
            if not re.fullmatch(
                re.escape(expected_container_name) + r"\t[0-9a-f]{12,64}\n?", before_text
            ):
                raise SummaryError(f"{label} vLLM Docker inventory-before does not identify only the owned container")
            for required_operation in ("stop", "wait"):
                if required_operation not in command_by_operation:
                    raise SummaryError(
                        f"{label} vLLM Docker cleanup receipt saw a live container but did not attempt {required_operation}"
                    )
        elif "stop" in command_by_operation or "wait" in command_by_operation:
            raise SummaryError(f"{label} vLLM Docker cleanup attempted stop/wait without an owned container inventory")
        after_stop_text = after_stop.get("stdout_text")
        if after_stop_text:
            if not re.fullmatch(
                re.escape(expected_container_name) + r"\t[0-9a-f]{12,64}\n?", after_stop_text
            ):
                raise SummaryError(f"{label} vLLM Docker inventory-after-stop does not identify only the owned container")
            force_remove = command_by_operation.get("force-remove")
            after_force_remove = command_by_operation.get("inventory-after-force-remove")
            if (
                force_remove is None
                or after_force_remove is None
                or force_remove.get("returncode") != 0
                or after_force_remove.get("returncode") != 0
                or after_force_remove.get("stdout_text") != ""
            ):
                raise SummaryError(f"{label} vLLM Docker residue was not force-removed and re-inventoried")
        elif "force-remove" in command_by_operation or "inventory-after-force-remove" in command_by_operation:
            raise SummaryError(f"{label} vLLM Docker cleanup recorded force-remove without a residual container")
        if inspect.get("stdout_text") != "":
            raise SummaryError(f"{label} vLLM Docker inspect-absence must have empty stdout")
        docker_container_cleanup = {
            "container_name": expected_container_name,
            "command_count": len(commands),
            "final_state": "absent",
        }
    gpu = _require_mapping(provenance.get("whole_gpu_memory"), f"{label} lane provenance.whole_gpu_memory")
    expected_gpu = {
        "sample_path": fields["whole_gpu_sample_path"],
        "sample_sha256": fields["whole_gpu_sample_sha256"],
        "peak_path": fields["whole_gpu_peak_receipt_path"],
        "peak_sha256": fields["whole_gpu_peak_receipt_sha256"],
        "peak_used_bytes": record["whole_gpu_sampled_peak_bytes"],
        "peak_limit_bytes": record["whole_gpu_sampled_peak_limit_bytes"],
        "compliant": True,
        "errors": [],
    }
    for key, expected in expected_gpu.items():
        if gpu.get(key) != expected:
            raise SummaryError(f"{label} lane provenance whole-GPU receipt differs at {key!r}")
    gpu_lifecycle = _require_mapping(gpu.get("lifecycle"), f"{label} lane provenance whole-GPU lifecycle")
    verified_whole_gpu_memory = _require_mapping(
        record.get("whole_gpu_memory"), f"{label} whole-GPU memory"
    )
    verified_lifecycle = _require_mapping(
        verified_whole_gpu_memory.get("lifecycle"),
        f"{label} verified whole-GPU lifecycle",
    )
    if gpu_lifecycle != verified_lifecycle:
        raise SummaryError(f"{label} lane provenance whole-GPU lifecycle differs from its peak receipt")
    if (
        verified_lifecycle.get("server_started_ns") != started_ns
        or verified_lifecycle.get("server_cleanup_completed_ns") != cleanup_completed_ns
    ):
        raise SummaryError(f"{label} lane provenance lifecycle does not match the sampled peak receipt")
    post_lane_gpu_idle = _validate_post_lane_gpu_idle_census(
        lane=lane,
        provenance=provenance,
        configuration=configuration,
        cleanup_completed_ns=cleanup_completed_ns,
        label=label,
    )
    final_whole_gpu_sample_finished_ns = _require_integer(
        verified_whole_gpu_memory.get("final_sample_finished_ns"),
        f"{label} final whole-GPU sample finished_ns",
        minimum=cleanup_completed_ns,
    )
    idle_commands = post_lane_gpu_idle.get("commands")
    if not isinstance(idle_commands, list) or not idle_commands:
        raise SummaryError(f"{label} post-lane GPU idle census lacks its first command")
    first_post_lane_gpu_idle_started_ns = _require_integer(
        _require_mapping(
            idle_commands[0], f"{label} post-lane GPU idle first command"
        ).get("started_ns"),
        f"{label} post-lane GPU idle first command.started_ns",
        minimum=final_whole_gpu_sample_finished_ns,
    )
    lane_pressure = _validate_lane_pressure_artifact(
        lane=lane,
        provenance=provenance,
        attempt_dir=attempt_dir,
        attempt_config=config,
        configuration=configuration,
        server_started_ns=started_ns,
        cleanup_completed_ns=cleanup_completed_ns,
        final_whole_gpu_sample_finished_ns=final_whole_gpu_sample_finished_ns,
        first_post_lane_gpu_idle_started_ns=first_post_lane_gpu_idle_started_ns,
        label=label,
    )
    retained_phase = _require_mapping(retained.get("phase"), f"{label} retained phase lifecycle")
    warmup_artifacts = _require_mapping(retained.get("server_warmup"), f"{label} server warmup lifecycle")
    warmup_phase = _require_mapping(warmup_artifacts.get("phase"), f"{label} server warmup phase lifecycle")
    for phase_name, phase in (("server warmup", warmup_phase), ("retained", retained_phase)):
        phase_started_ns = _require_integer(
            phase.get("phase_started_ns"), f"{label} {phase_name} phase_started_ns", minimum=0
        )
        phase_finished_ns = _require_integer(
            phase.get("phase_finished_ns"), f"{label} {phase_name} phase_finished_ns", minimum=phase_started_ns
        )
        if not (
            started_ns <= ready_ns <= phase_started_ns < phase_finished_ns <= cleanup_completed_ns
        ):
            raise SummaryError(
                f"{label} {phase_name} phase is outside the owned server lifecycle/readiness interval"
            )

    vllm_snapshot_paths: dict[str, str] | None = None
    if lane == "riley":
        snapshot = _require_mapping(provenance.get("riley_startup_snapshot"), f"{label} Riley startup provenance")
        if (
            snapshot.get("path") != fields["startup_log_path"]
            or snapshot.get("sha256") != fields["startup_log_sha256"]
        ):
            raise SummaryError(f"{label} Riley startup provenance differs from its marker snapshot")
        riley_snapshot_path = Path(fields["startup_log_path"])
        if riley_snapshot_path.parent != attempt_dir or riley_snapshot_path.name != "riley.startup.stderr.log":
            raise SummaryError(f"{label} Riley startup snapshot must be the dedicated frozen lane stderr log")
    else:
        snapshot = _require_mapping(provenance.get("vllm_startup_snapshot"), f"{label} vLLM startup provenance")
        expected_snapshot = {
            "stdout_path": fields["startup_stdout_log_path"],
            "stdout_sha256": fields["startup_stdout_log_sha256"],
            "stderr_path": fields["startup_stderr_log_path"],
            "stderr_sha256": fields["startup_stderr_log_sha256"],
            "backend_requested": fields["backend_requested"],
            "backend_resolved": fields["backend_resolved"],
        }
        for key, expected in expected_snapshot.items():
            if snapshot.get(key) != expected:
                raise SummaryError(f"{label} vLLM startup provenance differs at {key!r}")
        stdout_path, stdout_payload, _ = _read_marked_artifact(
            path_text=fields["startup_stdout_log_path"],
            expected_sha256=fields["startup_stdout_log_sha256"],
            label=f"{label} vLLM startup stdout snapshot",
            maximum_bytes=MAX_STARTUP_SNAPSHOT_BYTES,
        )
        stderr_path, stderr_payload, _ = _read_marked_artifact(
            path_text=fields["startup_stderr_log_path"],
            expected_sha256=fields["startup_stderr_log_sha256"],
            label=f"{label} vLLM startup stderr snapshot",
            maximum_bytes=MAX_STARTUP_SNAPSHOT_BYTES,
        )
        if (
            stdout_path.parent != attempt_dir
            or stderr_path.parent != attempt_dir
            or stdout_path.name != "vllm.startup.stdout.log"
            or stderr_path.name != "vllm.startup.stderr.log"
        ):
            raise SummaryError(f"{label} vLLM startup snapshots must be the frozen lane logs")
        try:
            startup_text = (stdout_payload + b"\n" + stderr_payload).decode("utf-8")
        except UnicodeDecodeError as error:
            raise SummaryError(f"{label} vLLM startup snapshots are not UTF-8") from error
        launch_provenance = _require_mapping(
            config.get("launch_provenance"), f"{label} attempt config.launch_provenance"
        )
        attestation = _require_mapping(
            launch_provenance.get("vllm_backend_attestation"),
            f"{label} attempt config.vllm_backend_attestation",
        )
        if (
            attestation.get("backend_requested") != VLLM_AUTO_BACKEND_REQUESTED
            or attestation.get("startup_receipt_regex") != VLLM_BACKEND_RECEIPT_REGEX.pattern
            or attestation.get("required_match_count") != 1
        ):
            raise SummaryError(f"{label} attempt config vLLM backend attestation contract differs")
        fragments = attestation.get("startup_required_fragments")
        if not isinstance(fragments, list) or not fragments or not all(
            isinstance(fragment, str) and fragment and len(fragment) <= 1_024
            for fragment in fragments
        ):
            raise SummaryError(f"{label} attempt config vLLM startup fragments are invalid")
        if any(fragment not in startup_text for fragment in fragments):
            raise SummaryError(f"{label} vLLM startup snapshots lack a required attestation fragment")
        matches = list(VLLM_BACKEND_RECEIPT_REGEX.finditer(startup_text))
        if len(matches) != 1 or matches[0].group("backend_resolved") != fields["backend_resolved"]:
            raise SummaryError(f"{label} vLLM startup backend receipt does not match its marker")
        vllm_snapshot_paths = {
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    result = {
        "path": str(provenance_path),
        "sha256": provenance_sha256,
        "pid": provenance["pid"],
        "started_ns": started_ns,
        "ready_ns": ready_ns,
        "cleanup_completed_ns": cleanup_completed_ns,
        "post_lane_gpu_idle": post_lane_gpu_idle,
        "lane_pressure": lane_pressure,
    }
    if vllm_snapshot_paths is not None:
        result["vllm_startup_snapshot"] = vllm_snapshot_paths
    if docker_container_cleanup is not None:
        result["vllm_docker_container_cleanup"] = docker_container_cleanup
    return result


SERVING_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "lane",
        "case",
        "pair_order",
        "model_id",
        "model_revision",
        "prompt_tokens",
        "max_output_tokens",
        "offered_concurrency",
        "batch_token_budget",
        "max_model_len",
        "model_identity_manifest_path",
        "model_identity_manifest_sha256",
        "model_identity_validation_path",
        "model_identity_validation_sha256",
        "vllm_prefix_caching",
        "vllm_image_digest",
        "vllm_gpu_memory_utilization",
        "whole_gpu_sampled_peak_limit_bytes",
        "whole_gpu_sampled_peak_bytes",
        "whole_gpu_sample_path",
        "whole_gpu_sample_sha256",
        "whole_gpu_peak_receipt_path",
        "whole_gpu_peak_receipt_sha256",
        "request_rows_path",
        "request_rows_sha256",
        "phase_path",
        "phase_sha256",
        "server_warmup_request_rows_path",
        "server_warmup_request_rows_sha256",
        "server_warmup_phase_path",
        "server_warmup_phase_sha256",
        "attempt_config_path",
        "attempt_config_sha256",
        "lane_provenance_path",
        "lane_provenance_sha256",
        "attempted_requests",
        "successful_requests",
        "failed_requests",
        "output_tokens",
        "retained_wall_ms",
        "output_tokens_per_second",
        "token_timing_scope",
        "token_ttft_median_ms",
        "token_ttft_p95_ms",
        "token_ttft_p99_ms",
        "token_tpot_median_ms",
        "token_tpot_p95_ms",
        "token_tpot_p99_ms",
        "e2e_median_ms",
        "e2e_p95_ms",
        "e2e_p99_ms",
        "latency_sample_count",
        "p99_status",
        "quality_status",
        "status",
    }
)

RILEY_SERVING_FIELDS = SERVING_COMMON_FIELDS | {
    "backend_requested",
    "backend_resolved",
    "fallback_reason",
    "startup_log_path",
    "startup_log_sha256",
    "query_heads",
    "key_value_heads",
    "head_size",
    "page_size",
    "graph_capture_enabled",
}

VLLM_SERVING_FIELDS = SERVING_COMMON_FIELDS | {
    "backend_requested",
    "backend_resolved",
    "fallback_reason",
    "startup_stdout_log_path",
    "startup_stdout_log_sha256",
    "startup_stderr_log_path",
    "startup_stderr_log_sha256",
}


def _validate_latency_order(record: Mapping[str, Any], *, label: str, prefix: str) -> None:
    median = float(record[f"{prefix}_median_ms"])
    p95 = float(record[f"{prefix}_p95_ms"])
    p99 = float(record[f"{prefix}_p99_ms"])
    if p95 < median or p99 < p95:
        raise SummaryError(f"{label}.{prefix} percentiles must satisfy median <= p95 <= p99")


def _parse_serving_record(
    fields: Mapping[str, str], *, label: str
) -> tuple[str, dict[str, Any]]:
    lane = fields.get("lane")
    if lane not in {"riley", "vllm"}:
        raise SummaryError(f"{label}.lane must be riley or vllm")
    required = RILEY_SERVING_FIELDS if lane == "riley" else VLLM_SERVING_FIELDS
    actual = set(fields)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        parts: list[str] = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("unsupported " + ", ".join(extra))
        raise SummaryError(f"{label} serving marker has " + "; ".join(parts))
    if fields["schema_version"] != "1":
        raise SummaryError(f"{label}.schema_version must be '1'")
    if fields["pair_order"] not in {"riley-vllm", "vllm-riley"}:
        raise SummaryError(f"{label}.pair_order must be riley-vllm or vllm-riley")
    if fields["model_id"] != QWEN_MODEL_ID or fields["model_revision"] != QWEN_MODEL_REVISION:
        raise SummaryError(f"{label} must use the pinned Qwen2.5-3B identity")
    if fields["token_timing_scope"] != "client-observed-single-token-sse":
        raise SummaryError(f"{label}.token_timing_scope must identify unambiguous token SSE timing")
    if fields["quality_status"] != "passed" or fields["status"] != "passed":
        raise SummaryError(f"{label} must record passed quality and status")
    prompt_tokens = _parse_record_integer(fields["prompt_tokens"], f"{label}.prompt_tokens", minimum=1)
    max_output_tokens = _parse_record_integer(
        fields["max_output_tokens"], f"{label}.max_output_tokens", minimum=2
    )
    offered_concurrency = _parse_record_integer(
        fields["offered_concurrency"], f"{label}.offered_concurrency", minimum=1
    )
    batch_token_budget = _parse_record_integer(
        fields["batch_token_budget"], f"{label}.batch_token_budget", minimum=1
    )
    max_model_len = _parse_record_integer(
        fields["max_model_len"], f"{label}.max_model_len", minimum=1
    )
    if offered_concurrency > batch_token_budget:
        raise SummaryError(f"{label}.offered_concurrency must not exceed batch_token_budget")
    if batch_token_budget > MAX_NATIVE_D128_BATCH_TOKEN_BUDGET:
        raise SummaryError(
            f"{label}.batch_token_budget must not exceed the native D128 bound "
            f"{MAX_NATIVE_D128_BATCH_TOKEN_BUDGET}"
        )
    if max_model_len > MAX_NATIVE_D128_MODEL_LENGTH:
        raise SummaryError(
            f"{label}.max_model_len must not exceed the native D128 bound "
            f"{MAX_NATIVE_D128_MODEL_LENGTH}"
        )
    if fields["vllm_prefix_caching"] != VLLM_PREFIX_CACHING_DISABLED:
        raise SummaryError(f"{label}.vllm_prefix_caching must be disabled for the fixed prompt")
    if not VLLM_IMAGE_DIGEST_RE.fullmatch(fields["vllm_image_digest"]):
        raise SummaryError(f"{label}.vllm_image_digest must be an immutable lowercase Docker digest")
    if fields["vllm_gpu_memory_utilization"] != VLLM_GPU_MEMORY_UTILIZATION:
        raise SummaryError(
            f"{label}.vllm_gpu_memory_utilization must be {VLLM_GPU_MEMORY_UTILIZATION}"
        )
    whole_gpu_sampled_peak_limit_bytes = _parse_record_integer(
        fields["whole_gpu_sampled_peak_limit_bytes"],
        f"{label}.whole_gpu_sampled_peak_limit_bytes",
        minimum=1,
    )
    if whole_gpu_sampled_peak_limit_bytes != WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES:
        raise SummaryError(
            f"{label}.whole_gpu_sampled_peak_limit_bytes must be "
            f"{WHOLE_GPU_SAMPLED_PEAK_LIMIT_BYTES}"
        )
    whole_gpu_sampled_peak_bytes = _parse_record_integer(
        fields["whole_gpu_sampled_peak_bytes"],
        f"{label}.whole_gpu_sampled_peak_bytes",
        minimum=0,
    )
    if whole_gpu_sampled_peak_bytes > whole_gpu_sampled_peak_limit_bytes:
        raise SummaryError(f"{label}.whole_gpu_sampled_peak_bytes exceeds its strict limit")
    attempted = _parse_record_integer(fields["attempted_requests"], f"{label}.attempted_requests", minimum=1)
    successful = _parse_record_integer(fields["successful_requests"], f"{label}.successful_requests", minimum=0)
    failed = _parse_record_integer(fields["failed_requests"], f"{label}.failed_requests", minimum=0)
    output_tokens = _parse_record_integer(fields["output_tokens"], f"{label}.output_tokens", minimum=0)
    if attempted != successful + failed or failed != 0 or successful != attempted:
        raise SummaryError(f"{label} must retain a complete zero-failure fixed workload")
    if output_tokens != successful * max_output_tokens:
        raise SummaryError(f"{label}.output_tokens must equal successful_requests * max_output_tokens")
    if prompt_tokens + max_output_tokens > max_model_len:
        raise SummaryError(f"{label}.max_model_len must fit prompt_tokens + max_output_tokens")
    latency_sample_count = _parse_record_integer(
        fields["latency_sample_count"], f"{label}.latency_sample_count", minimum=1
    )
    if latency_sample_count != successful:
        raise SummaryError(f"{label}.latency_sample_count must equal successful_requests")
    p99_status = fields["p99_status"]
    if p99_status not in {"descriptive", "qualified"}:
        raise SummaryError(f"{label}.p99_status must be descriptive or qualified")
    if p99_status == "qualified" and latency_sample_count < P99_QUALIFIED_MINIMUM_REQUESTS:
        raise SummaryError(
            f"{label}.p99_status cannot be qualified with fewer than "
            f"{P99_QUALIFIED_MINIMUM_REQUESTS} samples"
        )
    result: dict[str, Any] = {
        "lane": lane,
        "case": fields["case"],
        "pair_order": fields["pair_order"],
        "model_id": fields["model_id"],
        "model_revision": fields["model_revision"],
        "prompt_tokens": prompt_tokens,
        "max_output_tokens": max_output_tokens,
        "offered_concurrency": offered_concurrency,
        "batch_token_budget": batch_token_budget,
        "max_model_len": max_model_len,
        "model_identity_manifest_path": fields["model_identity_manifest_path"],
        "model_identity_manifest_sha256": fields["model_identity_manifest_sha256"],
        "model_identity_validation_path": fields["model_identity_validation_path"],
        "model_identity_validation_sha256": fields["model_identity_validation_sha256"],
        "vllm_prefix_caching": fields["vllm_prefix_caching"],
        "vllm_image_digest": fields["vllm_image_digest"],
        "vllm_gpu_memory_utilization": fields["vllm_gpu_memory_utilization"],
        "whole_gpu_sampled_peak_limit_bytes": whole_gpu_sampled_peak_limit_bytes,
        "whole_gpu_sampled_peak_bytes": whole_gpu_sampled_peak_bytes,
        "attempted_requests": attempted,
        "successful_requests": successful,
        "failed_requests": failed,
        "output_tokens": output_tokens,
        "retained_wall_ms": _parse_record_float(
            fields["retained_wall_ms"], f"{label}.retained_wall_ms", positive=True
        ),
        "output_tokens_per_second": _parse_record_float(
            fields["output_tokens_per_second"],
            f"{label}.output_tokens_per_second",
            positive=True,
        ),
        "token_timing_scope": fields["token_timing_scope"],
        "latency_sample_count": latency_sample_count,
        "p99_status": p99_status,
    }
    for prefix in ("token_ttft", "token_tpot", "e2e"):
        for statistic in ("median", "p95", "p99"):
            key = f"{prefix}_{statistic}_ms"
            result[key] = _parse_record_float(fields[key], f"{label}.{key}", positive=True)
        _validate_latency_order(result, label=label, prefix=prefix)
    expected_throughput = output_tokens * 1_000.0 / result["retained_wall_ms"]
    if not math.isclose(
        result["output_tokens_per_second"],
        expected_throughput,
        rel_tol=1e-3,
        abs_tol=1e-6,
    ):
        raise SummaryError(f"{label}.output_tokens_per_second is inconsistent with output_tokens and wall time")
    result["retained_artifacts"] = _validate_retained_artifacts(
        lane=lane,
        fields=fields,
        record=result,
        label=label,
    )
    result["whole_gpu_memory"] = _validate_whole_gpu_peak_receipt(
        lane=lane,
        fields=fields,
        record=result,
        label=label,
    )
    if lane == "riley":
        # These differ in the actual `riley serve` receipt: the CLI selects a
        # mode and the runtime resolves that mode to its concrete implementation.
        if (
            fields["backend_requested"] != REQUESTED_BACKEND_CLI_ID
            or fields["backend_resolved"] != SERVING_IMPLEMENTATION_ID
        ):
            raise SummaryError(
                f"{label} must request the native D128 CLI backend and resolve its V2 implementation"
            )
        if fields["fallback_reason"] != "none":
            raise SummaryError(f"{label}.fallback_reason must be none")
        geometry_values = {
            "query_heads": _parse_record_integer(fields["query_heads"], f"{label}.query_heads", minimum=1),
            "key_value_heads": _parse_record_integer(
                fields["key_value_heads"], f"{label}.key_value_heads", minimum=1
            ),
            "head_size": _parse_record_integer(fields["head_size"], f"{label}.head_size", minimum=1),
            "page_size": _parse_record_integer(fields["page_size"], f"{label}.page_size", minimum=1),
        }
        if geometry_values != {"query_heads": 16, "key_value_heads": 2, "head_size": 128, "page_size": 16}:
            raise SummaryError(f"{label} geometry must be QH16/KVH2/D128/page16")
        if fields["graph_capture_enabled"] != "false":
            raise SummaryError(f"{label}.graph_capture_enabled must be false")
        startup_snapshot = _validate_startup_snapshot(
            path_text=fields["startup_log_path"],
            expected_sha256=fields["startup_log_sha256"],
            marker_label=label,
        )
        if (
            fields["backend_requested"] != startup_snapshot["requested_backend"]
            or fields["backend_resolved"] != startup_snapshot["resolved_ragged_backend"]
        ):
            raise SummaryError(f"{label} backend fields differ from its startup receipt")
        result.update(
            backend_requested=fields["backend_requested"],
            backend_resolved=fields["backend_resolved"],
            fallback_reason=fields["fallback_reason"],
            geometry=geometry_values,
            startup_snapshot=startup_snapshot,
        )
    else:
        if fields["backend_requested"] != VLLM_AUTO_BACKEND_REQUESTED:
            raise SummaryError(f"{label}.backend_requested must record the vLLM auto attention selector")
        if fields["fallback_reason"] != "none":
            raise SummaryError(f"{label}.fallback_reason must be none for the paired vLLM control")
        result.update(
            backend_requested=fields["backend_requested"],
            backend_resolved=fields["backend_resolved"],
            fallback_reason=fields["fallback_reason"],
        )
    result["lane_provenance"] = _validate_lane_provenance(
        lane=lane,
        fields=fields,
        record=result,
        label=label,
    )
    return lane, result


def _parse_serving_stdout(text: str, *, source: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    marker_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        offset = raw_line.find(SERVING_MARKER_PREFIX)
        if offset < 0:
            continue
        if raw_line.find(SERVING_MARKER_PREFIX, offset + len(SERVING_MARKER_PREFIX)) >= 0:
            raise SummaryError(f"{source}:{line_number} contains multiple serving marker prefixes")
        marker_count += 1
        label = f"{source}:{line_number}"
        fields = _parse_marker_tokens(raw_line[offset:].strip(), prefix=SERVING_MARKER_PREFIX, label=label)
        lane, record = _parse_serving_record(fields, label=label)
        if lane in records:
            raise SummaryError(f"{label} duplicates serving marker for lane {lane!r}")
        records[lane] = {**record, "record_line_number": line_number}
    if set(records) != {"riley", "vllm"} or marker_count != 2:
        raise SummaryError(f"{source} must contain exactly one Riley and one vLLM serving marker")
    left, right = records["riley"], records["vllm"]
    common_identity = (
        "case",
        "pair_order",
        "model_id",
        "model_revision",
        "prompt_tokens",
        "max_output_tokens",
        "offered_concurrency",
        "batch_token_budget",
        "max_model_len",
        "model_identity_manifest_path",
        "model_identity_manifest_sha256",
        "model_identity_validation_path",
        "model_identity_validation_sha256",
        "vllm_prefix_caching",
        "vllm_image_digest",
        "vllm_gpu_memory_utilization",
        "whole_gpu_sampled_peak_limit_bytes",
        "attempted_requests",
        "successful_requests",
        "failed_requests",
        "output_tokens",
        "token_timing_scope",
        "latency_sample_count",
        "p99_status",
    )
    for key in common_identity:
        if left[key] != right[key]:
            raise SummaryError(f"{source} paired lanes differ at {key!r}")
    left_config = left["retained_artifacts"]["attempt_config"]
    right_config = right["retained_artifacts"]["attempt_config"]
    if left_config != right_config:
        raise SummaryError(f"{source} paired lanes must bind the same attempt configuration artifact")
    riley_before_vllm = left["record_line_number"] < right["record_line_number"]
    expected_riley_before_vllm = left["pair_order"] == "riley-vllm"
    if riley_before_vllm != expected_riley_before_vllm:
        raise SummaryError(
            f"{source} serving marker line order differs from its declared pair_order"
        )
    return records


def _operator_summary(receipt_path: Path) -> dict[str, Any]:
    """Use N03a's current strict V2 reader and attach all PSI resources."""
    # Preflight N03b's resource bounds before delegating to the existing N03a
    # reader, whose compatible input schema otherwise permits arbitrary
    # bootstrap counts.
    receipt = _load_repeat_receipt(receipt_path)
    try:
        n03a_summary = n03a.summarize_receipt(receipt_path)
    except n03a.SummaryError as error:
        raise SummaryError(f"N03a operator receipt is invalid: {error}") from error
    return {
        "classification": "prepared-paged-decode-operator-control",
        "expected_operator_implementation_id": OPERATOR_IMPLEMENTATION_ID,
        "full_model_serving": False,
        "vllm_comparison": False,
        "n03a_summary": n03a_summary,
        "timed_run_pressure_covariates": [
            _timed_run_covariate(run) for run in receipt.timed_runs
        ],
        "limitations": [
            "The N03a V2 control proves only the single-query prepared paged operator selection and timing scope.",
            "It does not prove batched runtime dispatch, Qwen full-model quality, HTTP serving, or a vLLM comparison.",
            "CPU, I/O, and memory PSI are covariates retained for every timed attempt and are never used to filter a run.",
        ],
    }


SERVING_METRICS = (
    "output_tokens_per_second",
    "token_ttft_median_ms",
    "token_ttft_p95_ms",
    "token_ttft_p99_ms",
    "token_tpot_median_ms",
    "token_tpot_p95_ms",
    "token_tpot_p99_ms",
    "e2e_median_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
)


def _paired_metrics(riley: Mapping[str, Any], vllm: Mapping[str, Any]) -> dict[str, float]:
    """Return one outer-pair effect for every reported latency percentile."""
    result: dict[str, float] = {
        "throughput_riley_over_vllm": riley["output_tokens_per_second"]
        / vllm["output_tokens_per_second"],
    }
    for timing in ("token_ttft", "token_tpot", "e2e"):
        for percentile in ("median", "p95", "p99"):
            key = f"{timing}_{percentile}_ms"
            result[f"{timing}_{percentile}_riley_over_vllm"] = riley[key] / vllm[key]
            result[f"{timing}_{percentile}_vllm_minus_riley_ms"] = vllm[key] - riley[key]
    return result


def _summarize_lane_metrics(
    observations: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, Any] | None:
    if not observations:
        return None
    return {
        lane: {
            "outer_process_metrics": {
                metric: _summarize_values(
                    [float(observation[lane][metric]) for observation in observations],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                )
                for metric in SERVING_METRICS
            }
        }
        for lane in ("riley", "vllm")
    }


def _summarize_paired_metrics(
    observations: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, Any] | None:
    if not observations:
        return None
    metric_names = tuple(observations[0]["paired_metrics"])
    return {
        metric: _summarize_values(
            [float(observation["paired_metrics"][metric]) for observation in observations],
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        )
        for metric in metric_names
    }


def _order_stratum(
    *, order: str, observations: Sequence[Mapping[str, Any]], receipt: RepeatReceipt
) -> dict[str, Any]:
    if not observations:
        return {
            "pair_order": order,
            "successful_outer_pairs": 0,
            "status": "incomplete",
            "reason": "no successful outer pairs in this order stratum",
            "per_lane": None,
            "paired_outer_process_metrics": None,
        }
    return {
        "pair_order": order,
        "successful_outer_pairs": len(observations),
        "status": "descriptive",
        "reason": "order-stratified repeated outer-pair summary; not a promotion decision",
        "per_lane": _summarize_lane_metrics(
            observations,
            bootstrap_resamples=receipt.bootstrap_resamples,
            bootstrap_seed=receipt.bootstrap_seed,
        ),
        "paired_outer_process_metrics": _summarize_paired_metrics(
            observations,
            bootstrap_resamples=receipt.bootstrap_resamples,
            bootstrap_seed=receipt.bootstrap_seed,
        ),
    }


def _descriptive_pressure_values(values: Sequence[float]) -> dict[str, Any]:
    """Summarize an observed PSI series without inference or reweighting."""
    observations = [float(value) for value in values]
    if not observations:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "sample_stddev": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(observations),
        "median": statistics.median(observations),
        "mean": statistics.fmean(observations),
        "sample_stddev": statistics.stdev(observations) if len(observations) > 1 else 0.0,
        "min": min(observations),
        "max": max(observations),
    }


def _lane_pressure_covariate(
    *, timed_index: int, pair_order: str, lane: str, pressure: Mapping[str, Any]
) -> dict[str, Any]:
    if pair_order not in {"riley-vllm", "vllm-riley"}:
        raise SummaryError("lane pressure covariate has an unsupported pair order")
    if lane not in {"riley", "vllm"}:
        raise SummaryError("lane pressure covariate has an unsupported lane")
    return {
        "timed_index": timed_index,
        "pair_order": pair_order,
        "lane": lane,
        "lane_position": "first" if pair_order.split("-", 1)[0] == lane else "second",
        "artifact": {"path": pressure["path"], "sha256": pressure["sha256"]},
        "pre": pressure["pre"],
        "post": pressure["post"],
        "derived": pressure["derived"],
    }


def _order_by_lane_pressure_sensitivity(
    covariates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose order/position pressure differences without altering any metric."""
    by_order: dict[str, list[Mapping[str, Any]]] = {
        order: [item for item in covariates if item["pair_order"] == order]
        for order in ("riley-vllm", "vllm-riley")
    }
    order_summaries: dict[str, Any] = {}
    for order, records in by_order.items():
        lanes: dict[str, Any] = {}
        for lane in ("riley", "vllm"):
            lane_records = [item for item in records if item["lane"] == lane]
            resource_summaries: dict[str, Any] = {}
            for resource in PSI_RESOURCES:
                pre_statuses = [item["pre"]["psi"][resource]["status"] for item in lane_records]
                post_statuses = [item["post"]["psi"][resource]["status"] for item in lane_records]
                categories: dict[str, Any] = {}
                for category in ("some", "full"):
                    available = [
                        item
                        for item in lane_records
                        if item["pre"]["psi"][resource][category] is not None
                        and item["post"]["psi"][resource][category] is not None
                        and item["derived"][resource][category] is not None
                    ]
                    categories[category] = {
                        "available_pair_count": len(available),
                        "pre_avg10_percent": _descriptive_pressure_values(
                            [item["pre"]["psi"][resource][category]["avg10"] for item in available]
                        ),
                        "post_avg10_percent": _descriptive_pressure_values(
                            [item["post"]["psi"][resource][category]["avg10"] for item in available]
                        ),
                        "total_delta_us": _descriptive_pressure_values(
                            [item["derived"][resource][category]["total_delta_us"] for item in available]
                        ),
                        "stall_percent": _descriptive_pressure_values(
                            [item["derived"][resource][category]["stall_percent"] for item in available]
                        ),
                    }
                resource_summaries[resource] = {
                    "pre_status_counts": {
                        status: pre_statuses.count(status)
                        for status in ("ok", "unavailable", "malformed")
                    },
                    "post_status_counts": {
                        status: post_statuses.count(status)
                        for status in ("ok", "unavailable", "malformed")
                    },
                    "categories": categories,
                }
            lanes[lane] = {
                "lane_position": "first" if order.split("-", 1)[0] == lane else "second",
                "completed_lane_observations": len(lane_records),
                "resources": resource_summaries,
            }
        order_summaries[order] = {
            "completed_outer_pairs": len(records) // 2,
            "status": "descriptive",
            "lanes": lanes,
        }
    return {
        "status": "descriptive",
        "policy": (
            "Lane PSI is shown by AB/BA order and lane position only; it is never used "
            "to select, filter, pair, weight, adjust, or promote a performance result."
        ),
        "total_delta_unit": "microseconds from Linux PSI cumulative total",
        "by_pair_order": order_summaries,
    }


def _serving_summary(receipt_path: Path) -> dict[str, Any]:
    receipt = _load_repeat_receipt(receipt_path)
    covariates: list[dict[str, Any]] = []
    lane_pressure_covariates: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    startup_snapshot_paths: set[str] = set()
    vllm_startup_snapshot_paths: set[str] = set()
    lane_provenance_paths: set[str] = set()
    retained_artifact_paths: set[str] = set()
    attempt_config_paths: set[str] = set()
    failed_pairs: list[dict[str, Any]] = []
    expected_identity: dict[str, Any] | None = None
    for run in receipt.timed_runs:
        covariate = _timed_run_covariate(run)
        run_status = _require_string(run.get("status"), "timed serving run.status")
        parent_cleanup: Mapping[str, Any] | None = None
        if (
            receipt.n06a_timeout_cleanup is not None
            and run_status != N01_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS
        ):
            parent_cleanup = _validate_n06a_parent_cleanup(
                run,
                cleanup=receipt.n06a_timeout_cleanup,
                output_dir=receipt.output_dir,
            )
            covariate["n06a_parent_cleanup"] = dict(parent_cleanup)
        if run_status == "succeeded":
            text, stdout = _load_timed_stdout(receipt, run)
            records = _parse_serving_stdout(text, source=stdout["path"])
            riley = records["riley"]
            vllm = records["vllm"]
            startup_path = riley["startup_snapshot"]["path"]
            if startup_path in startup_snapshot_paths:
                raise SummaryError(
                    "every successful serving outer process must have its own immutable startup log snapshot"
            )
            startup_snapshot_paths.add(startup_path)
            vllm_snapshots = _require_mapping(
                vllm["lane_provenance"].get("vllm_startup_snapshot"),
                "validated vLLM startup snapshot paths",
            )
            for path_key in ("stdout_path", "stderr_path"):
                snapshot_path = _require_string(
                    vllm_snapshots.get(path_key), f"validated vLLM startup snapshot {path_key}"
                )
                if snapshot_path in vllm_startup_snapshot_paths:
                    raise SummaryError(
                        "every successful serving outer process must have fresh vLLM startup snapshots"
                    )
                vllm_startup_snapshot_paths.add(snapshot_path)
            vllm_provenance_path = vllm["lane_provenance"]["path"]
            if vllm_provenance_path in lane_provenance_paths:
                raise SummaryError("every successful serving lane must have a unique provenance receipt")
            lane_provenance_paths.add(vllm_provenance_path)
            riley_provenance_path = riley["lane_provenance"]["path"]
            if riley_provenance_path in lane_provenance_paths:
                raise SummaryError("every successful serving lane must have a unique provenance receipt")
            lane_provenance_paths.add(riley_provenance_path)
            current_config_path = riley["retained_artifacts"]["attempt_config"]["path"]
            if parent_cleanup is not None:
                if parent_cleanup.get("cleanup_verified") is not True:
                    raise SummaryError("successful serving outer process lacks proven N01 parent cleanup")
                if parent_cleanup.get("attempt_artifact_directory") != str(Path(current_config_path).parent):
                    raise SummaryError("N01 parent cleanup attempt directory differs from marker-bound N06 artifacts")
                vllm_config = _decode_json(
                    _read_bounded_regular_file(
                        Path(current_config_path),
                        label="marker-bound N06 attempt config for N01 cleanup",
                        maximum_bytes=MAX_ATTEMPT_CONFIG_BYTES,
                    )[1],
                    label="marker-bound N06 attempt config for N01 cleanup",
                )
                launch = _require_mapping(
                    vllm_config.get("launch_provenance"), "marker-bound N06 launch provenance for N01 cleanup"
                )
                container = _require_mapping(
                    launch.get("vllm_docker_container"), "marker-bound N06 vLLM container for N01 cleanup"
                )
                if parent_cleanup.get("container_name") != container.get("name"):
                    raise SummaryError("N01 parent cleanup container name differs from marker-bound N06 vLLM name")
            if current_config_path in attempt_config_paths:
                raise SummaryError("attempt configuration artifact was reused by a different outer process")
            attempt_config_paths.add(current_config_path)
            for lane_record in (riley, vllm):
                artifacts = lane_record["retained_artifacts"]
                for artifact_kind in ("request_rows", "phase"):
                    artifact_path = artifacts[artifact_kind]["path"]
                    if artifact_path in retained_artifact_paths:
                        raise SummaryError(
                            "every successful serving lane must bind its own retained raw artifacts"
                        )
                    retained_artifact_paths.add(artifact_path)
                if artifacts["attempt_config"]["timed_index"] != int(run["index"]):
                    raise SummaryError("attempt configuration index differs from its N01 timed run")
                if artifacts["attempt_config"]["pair_order"] != riley["pair_order"]:
                    raise SummaryError("attempt configuration pair_order differs from its N01 timed run")
            identity = {
                key: riley[key]
                for key in (
                    "case",
                    "model_id",
                    "model_revision",
                    "prompt_tokens",
                    "max_output_tokens",
                    "offered_concurrency",
                    "batch_token_budget",
                    "max_model_len",
                    "vllm_prefix_caching",
                    "vllm_image_digest",
                    "vllm_gpu_memory_utilization",
                    "whole_gpu_sampled_peak_limit_bytes",
                    "attempted_requests",
                    "successful_requests",
                    "failed_requests",
                    "output_tokens",
                    "token_timing_scope",
                    "latency_sample_count",
                    "p99_status",
                )
            }
            expected_order = "riley-vllm" if int(run["index"]) % 2 else "vllm-riley"
            if riley["pair_order"] != expected_order:
                raise SummaryError(
                    "successful serving outer attempts must alternate Riley/vLLM order by timed index"
                )
            if expected_identity is None:
                expected_identity = identity
            elif identity != expected_identity:
                raise SummaryError("successful serving outer attempts do not use one identical workload")
            observation = {
                "timed_index": int(run["index"]),
                "pair_order": riley["pair_order"],
                "stdout": stdout,
                "riley": riley,
                "vllm": vllm,
                "paired_metrics": _paired_metrics(riley, vllm),
            }
            observations.append(observation)
            lane_pressure = {
                lane: _lane_pressure_covariate(
                    timed_index=int(run["index"]),
                    pair_order=riley["pair_order"],
                    lane=lane,
                    pressure=_require_mapping(
                        _require_mapping(
                            (riley if lane == "riley" else vllm).get("lane_provenance"),
                            f"validated {lane} lane provenance",
                        ).get("lane_pressure"),
                        f"validated {lane} lane PSI provenance",
                    ),
                )
                for lane in ("riley", "vllm")
            }
            covariate["lane_pressure"] = lane_pressure
            lane_pressure_covariates.extend(lane_pressure[lane] for lane in ("riley", "vllm"))
            covariate["stdout"] = stdout
        else:
            covariate["lane_pressure"] = None
            if run_status == N01_NOT_STARTED_AFTER_FAILED_CLEANUP_STATUS:
                failed_pairs.append(
                    {
                        "timed_index": int(run["index"]),
                        "status": run_status,
                        "exit_code": None,
                        "timed_out": False,
                        "reason": run["reason"],
                        "blocking": {
                            "kind": run["blocking_kind"],
                            "index": run["blocking_index"],
                        },
                    }
                )
            else:
                failed_pairs.append(
                    {
                        "timed_index": int(run["index"]),
                        "status": run_status,
                        "exit_code": run["exit_code"],
                        "timed_out": run["timed_out"],
                        "reason": run["error"] or f"N01 timed attempt status={run_status}",
                    }
                )
        covariates.append(covariate)
    completed_pair_count = len(observations)
    planned_order_counts = {
        "riley-vllm": sum(index % 2 for index in range(1, receipt.timed_repeats + 1)),
        "vllm-riley": sum(index % 2 == 0 for index in range(1, receipt.timed_repeats + 1)),
    }
    by_order = {
        order: [item for item in observations if item["pair_order"] == order]
        for order in ("riley-vllm", "vllm-riley")
    }
    successful_order_counts = {order: len(items) for order, items in by_order.items()}
    pair_completion = {
        "completed": completed_pair_count,
        "planned": receipt.timed_repeats,
        "completed_over_planned": f"{completed_pair_count}/{receipt.timed_repeats}",
        "non_successful": receipt.timed_repeats - completed_pair_count,
        "non_successful_reasons": failed_pairs,
    }
    all_planned_pairs_completed = completed_pair_count == receipt.timed_repeats
    planned_orders_balanced = (
        receipt.timed_repeats % 2 == 0
        and planned_order_counts["riley-vllm"] == planned_order_counts["vllm-riley"]
    )
    successful_orders_balanced = successful_order_counts == planned_order_counts
    if not all_planned_pairs_completed:
        promotion_status = {
            "status": "incomplete",
            "reason": "one or more planned timed pairs failed, timed out, or did not start",
        }
    elif not planned_orders_balanced:
        promotion_status = {
            "status": "descriptive",
            "reason": "planned timed pair count is odd, so AB/BA order strata are unbalanced",
        }
    else:
        promotion_status = {
            "status": "descriptive",
            "reason": "offline paired serving evidence is descriptive and requires separate promotion review",
        }
    promotion_status["pair_completion"] = pair_completion
    order_balance = {
        "planned_by_pair_order": planned_order_counts,
        "successful_by_pair_order": successful_order_counts,
        "status": "balanced" if planned_orders_balanced and successful_orders_balanced else "unbalanced",
        "reason": (
            "planned and successful timed pairs are balanced across AB/BA order"
            if planned_orders_balanced and successful_orders_balanced
            else "use an even planned repeat count and retain every timed pair for balanced AB/BA effects"
        ),
    }
    per_lane = _summarize_lane_metrics(
        observations,
        bootstrap_resamples=receipt.bootstrap_resamples,
        bootstrap_seed=receipt.bootstrap_seed,
    )
    paired = _summarize_paired_metrics(
        observations,
        bootstrap_resamples=receipt.bootstrap_resamples,
        bootstrap_seed=receipt.bootstrap_seed,
    )
    tail_metrics = (
        {
            name: value
            for name, value in paired.items()
            if "_p95_" in name or "_p99_" in name
        }
        if paired is not None
        else None
    )
    if not observations:
        tail_effect_status = {
            "status": "incomplete",
            "reason": "0 retained successful pairs; no tail effect can be estimated",
        }
    elif not all_planned_pairs_completed:
        tail_effect_status = {
            "status": "incomplete",
            "reason": "one or more planned timed pairs are absent, so tail effects are incomplete",
        }
    elif expected_identity is None or expected_identity["p99_status"] != "qualified":
        tail_effect_status = {
            "status": "descriptive",
            "reason": f"P99 requires at least {P99_QUALIFIED_MINIMUM_REQUESTS} retained requests per lane/cell",
        }
    else:
        tail_effect_status = {
            "status": "qualified",
            "reason": "paired P95/P99 ratios and deltas include outer-pair bootstrap confidence intervals",
        }
    return {
        "classification": "paired-full-model-serving-d128-v2-versus-vllm",
        "expected_riley_backend": {
            "startup_receipt_prefix": STARTUP_RECEIPT_PREFIX,
            "requested_backend": REQUESTED_BACKEND_CLI_ID,
            "resolved_implementation_id": SERVING_IMPLEMENTATION_ID,
            "fallback_reason": "none",
            "geometry": {
                "query_heads": 16,
                "key_value_heads": 2,
                "head_size": 128,
                "page_size": 16,
            },
            "execution": {
                "graph_capture_enabled": False,
            },
        },
        "receipt": {
            "path": str(receipt.path),
            "sha256": receipt.sha256,
            "schema_version": REPEAT_RECEIPT_SCHEMA_VERSION,
            "timed_repeats_configured": receipt.timed_repeats,
            "timed_runs_retained": len(receipt.timed_runs),
            "successful_timed_runs": len(observations),
            "non_successful_timed_runs": len(receipt.timed_runs) - len(observations),
            "bootstrap_resamples": receipt.bootstrap_resamples,
            "bootstrap_seed": receipt.bootstrap_seed,
        },
        "workload": expected_identity,
        "paired_order_policy": "timed odd indexes are riley-vllm; even indexes are vllm-riley",
        "promotion_status": promotion_status,
        "pair_order_balance": order_balance,
        "paired_tail_effect_status": tail_effect_status,
        "timed_run_pressure_covariates": covariates,
        "lane_pressure_covariates": lane_pressure_covariates,
        "order_by_lane_pressure_sensitivity": _order_by_lane_pressure_sensitivity(
            lane_pressure_covariates
        ),
        "successful_outer_pair_observations": observations,
        "per_lane": per_lane,
        "paired_outer_process_metrics": paired,
        "paired_tail_outer_process_metrics": tail_metrics,
        "order_stratified_effects": {
            order: _order_stratum(order=order, observations=items, receipt=receipt)
            for order, items in by_order.items()
        },
        "limitations": [
            "Only completed zero-failure outer attempts contribute numerical serving statistics; any missing planned pair makes promotion_status incomplete and remains in pair_completion and timed_run_pressure_covariates.",
            "Outer and lane CPU/I/O/memory PSI are observed covariates only and are never used as a selection, filtering, pairing, weighting, adjustment, or promotion rule.",
            "Lane PSI artifacts are captured around each owned lane; order-by-pressure summaries are descriptive context, not causal or adjusted performance effects.",
            "Client token timestamps are delivery observations, not CUDA or scheduler-commit timestamps.",
            "P95/P99 paired ratios and deltas are reported with outer-pair bootstrap intervals; tail_effect_status remains descriptive unless every planned pair completes and each lane records at least 1000 retained requests.",
            "The paired result is a Qwen2.5-3B D128 Riley graph-disabled serving comparison for its exact workload, not a general vLLM superiority claim.",
            "The N06-A wrapper binds an immutable vLLM image digest and its fixed execution/KV/prefix argv; the log-derived auto-selected backend remains reviewable in each retained lane provenance.",
        ],
    }


def summarize(
    *, operator_receipt: Path | None = None, serving_receipt: Path | None = None
) -> dict[str, Any]:
    """Read one or both evidence scopes without combining their performance claims."""
    if operator_receipt is None and serving_receipt is None:
        raise SummaryError("at least one of --operator-receipt or --serving-receipt is required")
    result: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "scope": {
            "operator_control_and_serving_are_separate": True,
            "operator_implementation_id": OPERATOR_IMPLEMENTATION_ID,
            "serving_implementation_id": SERVING_IMPLEMENTATION_ID,
            "host_pressure_policy": "all attempts and CPU/I/O/memory PSI covariates are retained; no pressure-based filtering",
        },
        "operator_control": None,
        "serving": None,
    }
    if operator_receipt is not None:
        result["operator_control"] = _operator_summary(operator_receipt)
    if serving_receipt is not None:
        result["serving"] = _serving_summary(serving_receipt)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator-receipt",
        type=Path,
        help="Immutable N03a V2 n01-repeat-control receipt; no artifact is written",
    )
    parser.add_argument(
        "--serving-receipt",
        type=Path,
        help="Immutable paired N06-A n01-repeat-control receipt; no artifact is written",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = summarize(
            operator_receipt=arguments.operator_receipt,
            serving_receipt=arguments.serving_receipt,
        )
    except SummaryError as error:
        print(f"n03b_n06a_d128_repeat_summary: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
