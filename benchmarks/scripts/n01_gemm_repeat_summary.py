#!/usr/bin/env python3
"""Summarize repeated Qwen2.5-3B native-GEMM control receipts.

The input is a completed immutable ``n01_repeat_control`` receipt.  This tool
does not launch a process, modify an artifact, or make a serving comparison.
It only reads the successful timed-run stdout logs referenced by that receipt,
validates the eight canonical synthetic BF16 projection-control records, and
prints a JSON summary to stdout.

Linux I/O PSI is deliberately retained as a per-run covariate.  It is never
used as a filter, so observations from an operating host remain available for
interpretation instead of being silently discarded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import stat
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


REPEAT_RECEIPT_SCHEMA_VERSION = "riley.n01-repeat-control-receipt.v1"
SUMMARY_SCHEMA_VERSION = "riley.n01-gemm-repeat-summary.v1"
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_STDOUT_LOG_BYTES = 8 * 1024 * 1024
MAX_TIMED_RUNS = 1_000


class SummaryError(ValueError):
    """A malformed receipt or operator-control log prevents summary output."""


@dataclass(frozen=True)
class ExpectedCase:
    """One current Qwen2.5-3B synthetic native-control projection shape."""

    label: str
    m: int
    n: int
    k: int


EXPECTED_CASES: tuple[ExpectedCase, ...] = (
    ExpectedCase("qwen2_5_3b-qkv-decode-m1", 1, 2_560, 2_048),
    ExpectedCase("qwen2_5_3b-qkv-decode-m8", 8, 2_560, 2_048),
    ExpectedCase("qwen2_5_3b-qkv-decode-m32", 32, 2_560, 2_048),
    ExpectedCase("qwen2_5_3b-qkv-prefill-m128", 128, 2_560, 2_048),
    ExpectedCase("qwen2_5_3b-gate-up-decode-m1", 1, 22_016, 2_048),
    ExpectedCase("qwen2_5_3b-gate-up-decode-m8", 8, 22_016, 2_048),
    ExpectedCase("qwen2_5_3b-gate-up-decode-m32", 32, 22_016, 2_048),
    ExpectedCase("qwen2_5_3b-gate-up-prefill-m128", 128, 22_016, 2_048),
)
EXPECTED_BY_LABEL = {case.label: case for case in EXPECTED_CASES}

REQUIRED_GEMM_FIELDS = {
    "case",
    "m",
    "n",
    "k",
    "gemm_reduction_policy",
    "latency_scope",
    "latency_median_ms",
    "latency_p95_ms",
    "effective_median_tflops",
    "temporary_bytes",
    "implementation_id",
    "explicit_stream",
    "python_free",
}
EXPECTED_CONTROL_RECORD = {
    "schema_version": "1",
    "input_weight_fixture": "deterministic-synthetic",
    "cases": "qkv:1,8,32,128;gate-up:1,8,32,128",
    "reduction_policy": "strict-no-split-v1",
    "full_model_fusion": "false",
    "full_model_serving": "false",
    "status": "passed",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SummaryError(f"receipt JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SummaryError(f"receipt JSON has non-finite constant {value!r}")


def _read_bounded_regular_file(path: Path, *, label: str, maximum_bytes: int) -> tuple[Path, bytes, str]:
    """Read one immutable evidence file without following a final symlink."""
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
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SummaryError(f"{label} cannot be resolved: {path}: {error}") from error
    return resolved, payload, hashlib.sha256(payload).hexdigest()


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
    converted = float(value)
    if not math.isfinite(converted):
        raise SummaryError(f"{label} must be a finite number")
    if positive and converted <= 0:
        raise SummaryError(f"{label} must be positive")
    return converted


def _parse_tokens(line: str, *, prefix: str, label: str) -> dict[str, str]:
    if not line.startswith(prefix + " "):
        raise SummaryError(f"{label} must begin with {prefix!r}")
    values: dict[str, str] = {}
    for token in line[len(prefix) + 1 :].split():
        if "=" not in token:
            raise SummaryError(f"{label} has malformed token {token!r}")
        key, value = token.split("=", 1)
        if not key or not value:
            raise SummaryError(f"{label} has malformed token {token!r}")
        if key in values:
            raise SummaryError(f"{label} has duplicate field {key!r}")
        values[key] = value
    return values


def _parse_record_integer(value: str, label: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise SummaryError(f"{label} must be a base-10 integer") from error
    if parsed < 0:
        raise SummaryError(f"{label} must be nonnegative")
    return parsed


def _parse_record_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise SummaryError(f"{label} must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise SummaryError(f"{label} must be a finite positive number")
    return parsed


def parse_qwen_control_stdout(text: str, *, source: str) -> dict[str, dict[str, Any]]:
    """Parse exactly one canonical record for every required control shape."""
    records: dict[str, dict[str, Any]] = {}
    control_markers: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line.startswith("riley-cuda-gemm"):
            fields = _parse_tokens(
                line,
                prefix="riley-cuda-gemm",
                label=f"{source}:{line_number}",
            )
            missing = sorted(REQUIRED_GEMM_FIELDS - set(fields))
            if missing:
                raise SummaryError(
                    f"{source}:{line_number} lacks required GEMM fields: {', '.join(missing)}"
                )
            case_label = fields["case"]
            expected = EXPECTED_BY_LABEL.get(case_label)
            if expected is None:
                raise SummaryError(
                    f"{source}:{line_number} has unexpected GEMM case {case_label!r}"
                )
            if case_label in records:
                raise SummaryError(
                    f"{source}:{line_number} duplicates expected GEMM case {case_label!r}"
                )
            dimensions = {
                "m": _parse_record_integer(fields["m"], f"{source}:{line_number}.m"),
                "n": _parse_record_integer(fields["n"], f"{source}:{line_number}.n"),
                "k": _parse_record_integer(fields["k"], f"{source}:{line_number}.k"),
            }
            expected_dimensions = {"m": expected.m, "n": expected.n, "k": expected.k}
            if dimensions != expected_dimensions:
                raise SummaryError(
                    f"{source}:{line_number} dimensions for {case_label!r} are "
                    f"{dimensions}, expected {expected_dimensions}"
                )
            if fields["gemm_reduction_policy"] != "strict-no-split-v1":
                raise SummaryError(
                    f"{source}:{line_number} must use strict-no-split-v1 reduction"
                )
            if fields["latency_scope"] != "ffi_execute_sync":
                raise SummaryError(
                    f"{source}:{line_number} must report ffi_execute_sync latency scope"
                )
            if fields["explicit_stream"] != "true" or fields["python_free"] != "true":
                raise SummaryError(
                    f"{source}:{line_number} must retain explicit_stream=true and python_free=true"
                )
            if not fields["implementation_id"].startswith("cublaslt:"):
                raise SummaryError(
                    f"{source}:{line_number}.implementation_id must identify a cuBLASLt plan"
                )
            temporary_bytes = _parse_record_integer(
                fields["temporary_bytes"], f"{source}:{line_number}.temporary_bytes"
            )
            records[case_label] = {
                "case": case_label,
                **dimensions,
                "latency_median_ms": _parse_record_float(
                    fields["latency_median_ms"],
                    f"{source}:{line_number}.latency_median_ms",
                ),
                "latency_p95_ms": _parse_record_float(
                    fields["latency_p95_ms"],
                    f"{source}:{line_number}.latency_p95_ms",
                ),
                "effective_median_tflops": _parse_record_float(
                    fields["effective_median_tflops"],
                    f"{source}:{line_number}.effective_median_tflops",
                ),
                "temporary_bytes": temporary_bytes,
                "implementation_id": fields["implementation_id"],
                "record_line_number": line_number,
            }
        elif line.startswith("riley-cuda-qwen2_5-3b-native-control"):
            marker = _parse_tokens(
                line,
                prefix="riley-cuda-qwen2_5-3b-native-control",
                label=f"{source}:{line_number}",
            )
            control_markers.append(marker)

    missing_cases = [case.label for case in EXPECTED_CASES if case.label not in records]
    if missing_cases:
        raise SummaryError(
            f"{source} is incomplete: missing expected GEMM records: {', '.join(missing_cases)}"
        )
    if len(control_markers) != 1:
        raise SummaryError(
            f"{source} must contain exactly one Qwen native-control status record; found {len(control_markers)}"
        )
    marker = control_markers[0]
    for key, expected_value in EXPECTED_CONTROL_RECORD.items():
        actual = marker.get(key)
        if actual != expected_value:
            raise SummaryError(
                f"{source} native-control status field {key!r} is {actual!r}, expected {expected_value!r}"
            )
    return records


def r7_quantile(values: Iterable[float], probability: float) -> float:
    """Return the Hyndman-Fan type-7 quantile used for bootstrap percentiles."""
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


def bootstrap_median_95_ci(
    values: Sequence[float], *, resamples: int, seed: int
) -> dict[str, Any]:
    """Produce a deterministic percentile-bootstrap interval for the median."""
    if not values:
        raise SummaryError("bootstrap requires at least one observation")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise SummaryError("bootstrap resamples must be an integer >= 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SummaryError("bootstrap seed must be an integer")
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
        "lower": r7_quantile(medians, 0.025),
        "upper": r7_quantile(medians, 0.975),
    }


def summarize_values(
    values: Sequence[float], *, bootstrap_resamples: int, bootstrap_seed: int, bootstrap: bool
) -> dict[str, Any]:
    """Summarize a nonempty observed metric vector without filtering values."""
    if not values:
        raise SummaryError("statistics require at least one observation")
    observations = [float(value) for value in values]
    if any(not math.isfinite(value) for value in observations):
        raise SummaryError("statistics observations must be finite")
    result: dict[str, Any] = {
        "count": len(observations),
        "median": statistics.median(observations),
        "mean": statistics.fmean(observations),
        "sample_stddev": statistics.stdev(observations) if len(observations) >= 2 else 0.0,
        "sample_stddev_method": "sample standard deviation; 0.0 for one observation",
        "min": min(observations),
        "max": max(observations),
    }
    if bootstrap:
        result["deterministic_bootstrap_median_95_ci"] = bootstrap_median_95_ci(
            observations,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
    return result


def _parse_psi_metrics(value: object, label: str) -> dict[str, float | int] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, label)
    result: dict[str, float | int] = {}
    for name in ("avg10", "avg60", "avg300"):
        parsed = _require_finite_number(mapping.get(name), f"{label}.{name}")
        if parsed < 0:
            raise SummaryError(f"{label}.{name} must be nonnegative")
        result[name] = parsed
    result["total"] = _require_integer(mapping.get("total"), f"{label}.total", minimum=0)
    return result


def extract_io_psi(run: Mapping[str, Any], *, phase: str, run_label: str) -> dict[str, Any]:
    """Extract a retained I/O PSI snapshot without treating pressure as a gate."""
    observation = _require_mapping(run.get(phase), f"{run_label}.{phase}")
    psi = _require_mapping(observation.get("psi"), f"{run_label}.{phase}.psi")
    io = _require_mapping(psi.get("io"), f"{run_label}.{phase}.psi.io")
    status = _require_string(io.get("status"), f"{run_label}.{phase}.psi.io.status")
    if status not in {"ok", "unavailable", "malformed"}:
        raise SummaryError(f"{run_label}.{phase}.psi.io.status is unsupported: {status!r}")
    some = _parse_psi_metrics(io.get("some"), f"{run_label}.{phase}.psi.io.some")
    full = _parse_psi_metrics(io.get("full"), f"{run_label}.{phase}.psi.io.full")
    if status == "ok" and some is None:
        raise SummaryError(f"{run_label}.{phase}.psi.io.some is required when status is ok")
    if status != "ok" and (some is not None or full is not None):
        raise SummaryError(
            f"{run_label}.{phase}.psi.io must not have numeric PSI data when status is {status!r}"
        )
    error = io.get("error")
    if error is not None and not isinstance(error, str):
        raise SummaryError(f"{run_label}.{phase}.psi.io.error must be a string or null")
    return {"status": status, "some": some, "full": full, "error": error}


def _load_receipt(receipt_path: Path) -> tuple[Path, Mapping[str, Any], str]:
    resolved, payload, sha256 = _read_bounded_regular_file(
        receipt_path,
        label="repeat receipt",
        maximum_bytes=MAX_RECEIPT_BYTES,
    )
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise SummaryError(f"repeat receipt is not UTF-8: {resolved}") from error
    except json.JSONDecodeError as error:
        raise SummaryError(f"repeat receipt is malformed JSON: {resolved}: {error}") from error
    return resolved, _require_mapping(decoded, "repeat receipt"), sha256


def _validate_receipt_identity(
    receipt: Mapping[str, Any], *, receipt_path: Path
) -> tuple[Path, int, int, int]:
    schema = receipt.get("schema_version")
    if schema != REPEAT_RECEIPT_SCHEMA_VERSION:
        raise SummaryError(
            f"repeat receipt.schema_version must be {REPEAT_RECEIPT_SCHEMA_VERSION!r}, got {schema!r}"
        )
    declared_receipt_path = Path(_require_string(receipt.get("receipt_path"), "repeat receipt.receipt_path"))
    try:
        declared_receipt_path = declared_receipt_path.resolve(strict=True)
    except OSError as error:
        raise SummaryError("repeat receipt.receipt_path cannot be resolved") from error
    if declared_receipt_path != receipt_path:
        raise SummaryError(
            "repeat receipt.receipt_path does not identify the supplied immutable receipt"
        )
    output_dir = Path(_require_string(receipt.get("output_dir"), "repeat receipt.output_dir"))
    try:
        output_dir = output_dir.resolve(strict=True)
    except OSError as error:
        raise SummaryError("repeat receipt.output_dir cannot be resolved") from error
    if not output_dir.is_dir():
        raise SummaryError("repeat receipt.output_dir must be a directory")
    configuration = _require_mapping(receipt.get("configuration"), "repeat receipt.configuration")
    timed_repeats = _require_integer(
        configuration.get("timed_repeats"), "repeat receipt.configuration.timed_repeats", minimum=1
    )
    if timed_repeats > MAX_TIMED_RUNS:
        raise SummaryError(f"repeat receipt.configured timed repeats exceeds {MAX_TIMED_RUNS}")
    bootstrap_resamples = _require_integer(
        configuration.get("bootstrap_resamples"),
        "repeat receipt.configuration.bootstrap_resamples",
        minimum=1,
    )
    bootstrap_seed = _require_integer(
        configuration.get("bootstrap_seed"),
        "repeat receipt.configuration.bootstrap_seed",
    )
    return output_dir, timed_repeats, bootstrap_resamples, bootstrap_seed


def _validate_timed_runs(
    receipt: Mapping[str, Any], *, timed_repeats: int
) -> list[Mapping[str, Any]]:
    receipt_status = receipt.get("status")
    if receipt_status not in {"completed", "completed-with-failures"}:
        raise SummaryError(
            "repeat receipt must be completed before analysis; interrupted receipts are incomplete"
        )
    timed_runs = receipt.get("timed_runs")
    if not isinstance(timed_runs, list):
        raise SummaryError("repeat receipt.timed_runs must be an array")
    if len(timed_runs) != timed_repeats:
        raise SummaryError(
            "repeat receipt.timed_runs must retain every configured timed attempt "
            f"({timed_repeats} expected, {len(timed_runs)} present)"
        )
    result: list[Mapping[str, Any]] = []
    indices: set[int] = set()
    for offset, raw_run in enumerate(timed_runs, start=1):
        run = _require_mapping(raw_run, f"repeat receipt.timed_runs[{offset - 1}]")
        label = f"repeat receipt.timed_runs[{offset - 1}]"
        if run.get("kind") != "timed":
            raise SummaryError(f"{label}.kind must be 'timed'")
        index = _require_integer(run.get("index"), f"{label}.index", minimum=1)
        if index > timed_repeats or index in indices:
            raise SummaryError(f"{label}.index must be a unique value in 1..{timed_repeats}")
        indices.add(index)
        status = _require_string(run.get("status"), f"{label}.status")
        if status not in {"succeeded", "failed", "timed-out", "failed-to-start"}:
            raise SummaryError(f"{label}.status is unsupported: {status!r}")
        _require_finite_number(run.get("wall_time_ms"), f"{label}.wall_time_ms", positive=True)
        timed_out = run.get("timed_out")
        if not isinstance(timed_out, bool):
            raise SummaryError(f"{label}.timed_out must be a boolean")
        exit_code = run.get("exit_code")
        if exit_code is not None:
            _require_integer(exit_code, f"{label}.exit_code")
        if status == "succeeded" and (exit_code != 0 or timed_out):
            raise SummaryError(f"{label} marks a non-zero or timed-out attempt as succeeded")
        result.append(run)
    if indices != set(range(1, timed_repeats + 1)):
        raise SummaryError("repeat receipt.timed_runs indices are incomplete")
    return result


def _load_successful_run_records(
    run: Mapping[str, Any], *, output_dir: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    index = int(run["index"])
    label = f"repeat receipt.timed_runs[{index}]"
    stdout_raw = Path(_require_string(run.get("stdout_path"), f"{label}.stdout_path"))
    expected_name = f"timed-{index:03d}.stdout.log"
    if stdout_raw.name != expected_name:
        raise SummaryError(f"{label}.stdout_path must be named {expected_name!r}")
    resolved, payload, sha256 = _read_bounded_regular_file(
        stdout_raw,
        label=f"timed stdout for run {index}",
        maximum_bytes=MAX_STDOUT_LOG_BYTES,
    )
    if resolved.parent != output_dir:
        raise SummaryError(f"{label}.stdout_path must be directly inside repeat receipt.output_dir")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SummaryError(f"{label}.stdout_path is not UTF-8") from error
    records = parse_qwen_control_stdout(text, source=str(resolved))
    return records, {
        "timed_index": index,
        "stdout_path": str(resolved),
        "stdout_sha256": sha256,
    }


def summarize_receipt(receipt_path: Path) -> dict[str, Any]:
    """Validate one immutable repeat receipt and derive its operator-control summary."""
    resolved_receipt, receipt, receipt_sha256 = _load_receipt(receipt_path)
    output_dir, timed_repeats, bootstrap_resamples, bootstrap_seed = _validate_receipt_identity(
        receipt,
        receipt_path=resolved_receipt,
    )
    timed_runs = _validate_timed_runs(receipt, timed_repeats=timed_repeats)
    run_covariates: list[dict[str, Any]] = []
    observations_by_case: dict[str, list[dict[str, Any]]] = {
        case.label: [] for case in EXPECTED_CASES
    }
    successful_log_count = 0
    for run in timed_runs:
        index = int(run["index"])
        run_label = f"repeat receipt.timed_runs[{index}]"
        covariate = {
            "timed_index": index,
            "status": run["status"],
            "exit_code": run.get("exit_code"),
            "timed_out": run["timed_out"],
            "wall_time_ms": _require_finite_number(
                run["wall_time_ms"], f"{run_label}.wall_time_ms", positive=True
            ),
            "io_psi": {
                "pre": extract_io_psi(run, phase="pre", run_label=run_label),
                "post": extract_io_psi(run, phase="post", run_label=run_label),
            },
        }
        if run["status"] == "succeeded":
            records, log_identity = _load_successful_run_records(run, output_dir=output_dir)
            covariate["stdout"] = log_identity
            successful_log_count += 1
            for case_label, record in records.items():
                observations_by_case[case_label].append(
                    {
                        "timed_index": index,
                        "stdout_sha256": log_identity["stdout_sha256"],
                        "record_line_number": record["record_line_number"],
                        "latency_median_ms": record["latency_median_ms"],
                        "latency_p95_ms": record["latency_p95_ms"],
                        "effective_median_tflops": record["effective_median_tflops"],
                        "temporary_bytes": record["temporary_bytes"],
                        "implementation_id": record["implementation_id"],
                    }
                )
        run_covariates.append(covariate)
    if successful_log_count == 0:
        raise SummaryError("repeat receipt has no successful timed runs to summarize")

    per_case: list[dict[str, Any]] = []
    for expected in EXPECTED_CASES:
        observations = observations_by_case[expected.label]
        if len(observations) != successful_log_count:
            raise SummaryError(
                f"case {expected.label!r} is incomplete across successful timed runs "
                f"({len(observations)} of {successful_log_count})"
            )
        per_case.append(
            {
                "case": expected.label,
                "m": expected.m,
                "n": expected.n,
                "k": expected.k,
                "observations": observations,
                "latency_median_ms": summarize_values(
                    [float(item["latency_median_ms"]) for item in observations],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap=True,
                ),
                "latency_p95_ms": summarize_values(
                    [float(item["latency_p95_ms"]) for item in observations],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap=False,
                ),
                "effective_median_tflops": summarize_values(
                    [float(item["effective_median_tflops"]) for item in observations],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap=False,
                ),
            }
        )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "scope": {
            "classification": "synthetic-native-bf16-projection-operator-control",
            "full_model_serving": False,
            "vllm_comparison": False,
            "limitations": [
                "This analysis summarizes prepared FFI execute-and-synchronize GEMM controls only.",
                "It is not a full-model, serving, latency, throughput, correctness, or vLLM comparison result.",
                "I/O PSI is retained as an operating-host covariate and is not used to filter runs.",
            ],
        },
        "receipt": {
            "path": str(resolved_receipt),
            "sha256": receipt_sha256,
            "schema_version": REPEAT_RECEIPT_SCHEMA_VERSION,
            "timed_repeats_configured": timed_repeats,
            "timed_runs_retained": len(timed_runs),
            "successful_timed_runs": successful_log_count,
            "non_successful_timed_runs": len(timed_runs) - successful_log_count,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
        },
        "expected_cases": [
            {"case": case.label, "m": case.m, "n": case.n, "k": case.k}
            for case in EXPECTED_CASES
        ],
        "timed_run_io_psi_covariates": run_covariates,
        "per_case": per_case,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        required=True,
        type=Path,
        help="Immutable n01-repeat-control receipt to read; no artifact is written",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_receipt(args.receipt)
    except SummaryError as error:
        print(f"n01_gemm_repeat_summary: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
