"""Summarize repeated N03a paged-D128 paired CUDA-event controls.

This reader consumes one completed immutable ``n01_repeat_control`` receipt.
Every timed attempt remains visible in the output, including failed and
timed-out attempts, together with its pre/post I/O PSI observations.  Only a
successful process can contribute a parsed operator observation, because a
failed process did not complete the paired CUDA-event control.  I/O pressure
is never a selection criterion.

The successful process log contract is intentionally narrow: it must contain
exactly the two ``riley-cuda-n03a-paged-gqa`` records emitted by the current
GPU control, for Qwen2.5-3B GQA geometry at contexts 2,048 and 16,384.  The
result is an operator-control summary, not a full-model, serving, or vLLM
comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import stat
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPEAT_RECEIPT_SCHEMA_VERSION = "riley.n01-repeat-control-receipt.v1"
SUMMARY_SCHEMA_VERSION = "riley.n03a-paged-d128-repeat-summary.v1"
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_STDOUT_LOG_BYTES = 8 * 1024 * 1024
MAX_TIMED_RUNS = 1_000
MARKER_PREFIX = "riley-cuda-n03a-paged-gqa"
IMPLEMENTATION_ID = (
    "riley.cuda.native-bf16-paged-split-gqa.qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2"
)


class SummaryError(ValueError):
    """A malformed receipt or N03a operator-control log prevents a summary."""


@dataclass(frozen=True)
class ExpectedCase:
    """One exact CUDA-event control record emitted by the N03a GPU test."""

    label: str
    logical_tokens: int


EXPECTED_CASES: tuple[ExpectedCase, ...] = (
    ExpectedCase("b1-c2048-qh16-kvh2-d128", 2_048),
    ExpectedCase("b1-c16384-qh16-kvh2-d128", 16_384),
)
EXPECTED_BY_LABEL = {case.label: case for case in EXPECTED_CASES}

REQUIRED_MARKER_FIELDS = frozenset(
    {
        "schema_version",
        "case",
        "logical_tokens",
        "batch",
        "query_heads",
        "key_value_heads",
        "head_size",
        "page_size",
        "fixture",
        "shuffled_page_ids",
        "partial_last_page",
        "timing_scope",
        "internal_warmups_per_backend",
        "paired_rounds",
        "paired_order",
        "native_median_ms",
        "native_p95_ms",
        "reference_median_ms",
        "reference_p95_ms",
        "paired_speedup_ratio",
        "paired_delta_ms",
        "native_workspace_bytes",
        "native_v2_state_prefix_bytes",
        "native_v2_transition_step_bytes",
        "native_v2_normalizer_bytes",
        "reference_workspace_bytes",
        "implementation_id",
        "implementation_version",
        "graph_capture_supported",
        "operator_parity",
        "allocation_delta",
        "python_free",
        "full_model_serving",
        "vllm_comparison",
        "status",
    }
)

EXPECTED_STATIC_MARKER_FIELDS = {
    "schema_version": "2",
    "batch": "1",
    "query_heads": "16",
    "key_value_heads": "2",
    "head_size": "128",
    "page_size": "16",
    "fixture": "patterned",
    "shuffled_page_ids": "true",
    "partial_last_page": "false",
    "timing_scope": "prepared_paged_decode_execute_cuda_event",
    "internal_warmups_per_backend": "8",
    "paired_rounds": "24",
    "paired_order": "ABBA",
    "implementation_id": IMPLEMENTATION_ID,
    "implementation_version": "2",
    "graph_capture_supported": "false",
    "operator_parity": "passed",
    "allocation_delta": "0",
    "python_free": "true",
    "full_model_serving": "false",
    "vllm_comparison": "false",
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
    """Read a stable regular evidence file without following a final symlink."""
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
        raise SummaryError(f"{label} cannot be resolved: {path}") from error
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


def _validate_paired_metric_relation(record: Mapping[str, Any], *, source: str) -> None:
    """Check the displayed ratio and delta remain coherent after six-place formatting."""
    expected_speedup = float(record["reference_median_ms"]) / float(record["native_median_ms"])
    expected_delta = float(record["reference_median_ms"]) - float(record["native_median_ms"])
    if not math.isclose(
        float(record["paired_speedup_ratio"]),
        expected_speedup,
        rel_tol=1e-3,
        abs_tol=1e-4,
    ):
        raise SummaryError(f"{source}.paired_speedup_ratio is inconsistent with displayed medians")
    if not math.isclose(
        float(record["paired_delta_ms"]),
        expected_delta,
        rel_tol=0.0,
        abs_tol=2e-6,
    ):
        raise SummaryError(f"{source}.paired_delta_ms is inconsistent with displayed medians")


def parse_n03a_paged_d128_stdout(text: str, *, source: str) -> dict[str, dict[str, Any]]:
    """Parse exactly one current N03a marker for each required context length."""
    records: dict[str, dict[str, Any]] = {}
    marker_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        marker_offset = line.find(MARKER_PREFIX)
        if marker_offset < 0:
            continue
        if line.find(MARKER_PREFIX, marker_offset + len(MARKER_PREFIX)) >= 0:
            raise SummaryError(f"{source}:{line_number} contains multiple N03a marker prefixes")
        marker_count += 1
        label = f"{source}:{line_number}"
        fields = _parse_tokens(line[marker_offset:], prefix=MARKER_PREFIX, label=label)
        missing = sorted(REQUIRED_MARKER_FIELDS - set(fields))
        if missing:
            raise SummaryError(f"{label} lacks required marker fields: {', '.join(missing)}")
        unknown = sorted(set(fields) - REQUIRED_MARKER_FIELDS)
        if unknown:
            raise SummaryError(f"{label} has unsupported marker fields: {', '.join(unknown)}")
        case_label = fields["case"]
        expected_case = EXPECTED_BY_LABEL.get(case_label)
        if expected_case is None:
            raise SummaryError(f"{label} has unexpected N03a case {case_label!r}")
        if case_label in records:
            raise SummaryError(f"{label} duplicates expected N03a case {case_label!r}")
        for field, expected_value in EXPECTED_STATIC_MARKER_FIELDS.items():
            actual = fields[field]
            if actual != expected_value:
                raise SummaryError(
                    f"{label}.{field} is {actual!r}, expected {expected_value!r}"
                )
        logical_tokens = _parse_record_integer(
            fields["logical_tokens"], f"{label}.logical_tokens", minimum=1
        )
        if logical_tokens != expected_case.logical_tokens:
            raise SummaryError(
                f"{label}.logical_tokens is {logical_tokens}, "
                f"expected {expected_case.logical_tokens} for {case_label!r}"
            )
        native_median_ms = _parse_record_float(
            fields["native_median_ms"], f"{label}.native_median_ms", positive=True
        )
        native_p95_ms = _parse_record_float(
            fields["native_p95_ms"], f"{label}.native_p95_ms", positive=True
        )
        reference_median_ms = _parse_record_float(
            fields["reference_median_ms"], f"{label}.reference_median_ms", positive=True
        )
        reference_p95_ms = _parse_record_float(
            fields["reference_p95_ms"], f"{label}.reference_p95_ms", positive=True
        )
        if native_p95_ms < native_median_ms:
            raise SummaryError(f"{label}.native_p95_ms must be >= native_median_ms")
        if reference_p95_ms < reference_median_ms:
            raise SummaryError(f"{label}.reference_p95_ms must be >= reference_median_ms")
        record = {
            "case": case_label,
            "logical_tokens": logical_tokens,
            "native_median_ms": native_median_ms,
            "native_p95_ms": native_p95_ms,
            "reference_median_ms": reference_median_ms,
            "reference_p95_ms": reference_p95_ms,
            "paired_speedup_ratio": _parse_record_float(
                fields["paired_speedup_ratio"],
                f"{label}.paired_speedup_ratio",
                positive=True,
            ),
            "paired_delta_ms": _parse_record_float(
                fields["paired_delta_ms"], f"{label}.paired_delta_ms"
            ),
            "native_workspace_bytes": _parse_record_integer(
                fields["native_workspace_bytes"],
                f"{label}.native_workspace_bytes",
                minimum=0,
            ),
            "native_v2_state_prefix_bytes": _parse_record_integer(
                fields["native_v2_state_prefix_bytes"],
                f"{label}.native_v2_state_prefix_bytes",
                minimum=1,
            ),
            "native_v2_transition_step_bytes": _parse_record_integer(
                fields["native_v2_transition_step_bytes"],
                f"{label}.native_v2_transition_step_bytes",
                minimum=1,
            ),
            "native_v2_normalizer_bytes": _parse_record_integer(
                fields["native_v2_normalizer_bytes"],
                f"{label}.native_v2_normalizer_bytes",
                minimum=1,
            ),
            "reference_workspace_bytes": _parse_record_integer(
                fields["reference_workspace_bytes"],
                f"{label}.reference_workspace_bytes",
                minimum=0,
            ),
            "implementation_id": fields["implementation_id"],
            "implementation_version": fields["implementation_version"],
            "internal_warmups_per_backend": _parse_record_integer(
                fields["internal_warmups_per_backend"],
                f"{label}.internal_warmups_per_backend",
                minimum=1,
            ),
            "paired_rounds": _parse_record_integer(
                fields["paired_rounds"], f"{label}.paired_rounds", minimum=1
            ),
            "record_line_number": line_number,
        }
        expected_native_workspace_bytes = (
            record["native_v2_state_prefix_bytes"]
            + record["native_v2_transition_step_bytes"]
            + record["native_v2_normalizer_bytes"]
        )
        if record["native_workspace_bytes"] != expected_native_workspace_bytes:
            raise SummaryError(
                f"{label}.native_workspace_bytes is {record['native_workspace_bytes']}, "
                "expected the sum of native V2 workspace partitions "
                f"({expected_native_workspace_bytes})"
            )
        _validate_paired_metric_relation(record, source=label)
        records[case_label] = record
    missing_cases = [case.label for case in EXPECTED_CASES if case.label not in records]
    if missing_cases:
        raise SummaryError(
            f"{source} is incomplete: missing expected N03a records: {', '.join(missing_cases)}"
        )
    if marker_count != len(EXPECTED_CASES):
        raise SummaryError(
            f"{source} must contain exactly {len(EXPECTED_CASES)} N03a markers; found {marker_count}"
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
    """Produce a deterministic percentile-bootstrap interval for an outer median."""
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
    values: Sequence[float], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, Any]:
    """Summarize every parsed successful outer-process observation."""
    if not values:
        raise SummaryError("statistics require at least one observation")
    observations = [float(value) for value in values]
    if any(not math.isfinite(value) for value in observations):
        raise SummaryError("statistics observations must be finite")
    return {
        "count": len(observations),
        "median": statistics.median(observations),
        "mean": statistics.fmean(observations),
        "sample_stddev": statistics.stdev(observations) if len(observations) >= 2 else 0.0,
        "sample_stddev_method": "sample standard deviation; 0.0 for one observation",
        "min": min(observations),
        "max": max(observations),
        "deterministic_bootstrap_median_95_ci": bootstrap_median_95_ci(
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
    for name in ("avg10", "avg60", "avg300"):
        parsed = _require_finite_number(mapping.get(name), f"{label}.{name}")
        if parsed < 0:
            raise SummaryError(f"{label}.{name} must be nonnegative")
        result[name] = parsed
    result["total"] = _require_integer(mapping.get("total"), f"{label}.total", minimum=0)
    return result


def extract_io_psi(run: Mapping[str, Any], *, phase: str, run_label: str) -> dict[str, Any]:
    """Retain one I/O PSI observation; it is deliberately not a quality gate."""
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
    declared_receipt_path = Path(
        _require_string(receipt.get("receipt_path"), "repeat receipt.receipt_path")
    )
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
        configuration.get("timed_repeats"),
        "repeat receipt.configuration.timed_repeats",
        minimum=1,
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
        if status == "timed-out" and not timed_out:
            raise SummaryError(f"{label} marks a non-timeout attempt as timed-out")
        _require_string(run.get("stdout_path"), f"{label}.stdout_path")
        _require_string(run.get("stderr_path"), f"{label}.stderr_path")
        error = run.get("error")
        if error is not None and not isinstance(error, str):
            raise SummaryError(f"{label}.error must be a string or null")
        result.append(run)
    if indices != set(range(1, timed_repeats + 1)):
        raise SummaryError("repeat receipt.timed_runs indices are incomplete")
    return sorted(result, key=lambda run: int(run["index"]))


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
    records = parse_n03a_paged_d128_stdout(text, source=str(resolved))
    return records, {
        "timed_index": index,
        "stdout_path": str(resolved),
        "stdout_sha256": sha256,
    }


def _run_covariate(run: Mapping[str, Any]) -> dict[str, Any]:
    index = int(run["index"])
    label = f"repeat receipt.timed_runs[{index}]"
    return {
        "timed_index": index,
        "status": run["status"],
        "exit_code": run.get("exit_code"),
        "timed_out": run["timed_out"],
        "wall_time_ms": _require_finite_number(
            run["wall_time_ms"], f"{label}.wall_time_ms", positive=True
        ),
        "error": run.get("error"),
        "attempt_log_paths": {
            "stdout_path": run["stdout_path"],
            "stderr_path": run["stderr_path"],
        },
        "io_psi": {
            "pre": extract_io_psi(run, phase="pre", run_label=label),
            "post": extract_io_psi(run, phase="post", run_label=label),
        },
    }


def summarize_receipt(receipt_path: Path) -> dict[str, Any]:
    """Validate an immutable repeat receipt and aggregate N03a outer attempts."""
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
        covariate = _run_covariate(run)
        if run["status"] == "succeeded":
            records, log_identity = _load_successful_run_records(run, output_dir=output_dir)
            covariate["stdout"] = log_identity
            successful_log_count += 1
            for case_label, record in records.items():
                observations_by_case[case_label].append(
                    {
                        "timed_index": int(run["index"]),
                        "stdout_sha256": log_identity["stdout_sha256"],
                        "record_line_number": record["record_line_number"],
                        "native_median_ms": record["native_median_ms"],
                        "native_p95_ms": record["native_p95_ms"],
                        "reference_median_ms": record["reference_median_ms"],
                        "reference_p95_ms": record["reference_p95_ms"],
                        "paired_speedup_ratio": record["paired_speedup_ratio"],
                        "paired_delta_ms": record["paired_delta_ms"],
                        "native_workspace_bytes": record["native_workspace_bytes"],
                        "native_v2_state_prefix_bytes": record[
                            "native_v2_state_prefix_bytes"
                        ],
                        "native_v2_transition_step_bytes": record[
                            "native_v2_transition_step_bytes"
                        ],
                        "native_v2_normalizer_bytes": record[
                            "native_v2_normalizer_bytes"
                        ],
                        "reference_workspace_bytes": record["reference_workspace_bytes"],
                        "implementation_id": record["implementation_id"],
                        "implementation_version": record["implementation_version"],
                    }
                )
        run_covariates.append(covariate)
    if successful_log_count == 0:
        raise SummaryError("repeat receipt has no successful timed runs to summarize")

    per_case: list[dict[str, Any]] = []
    metric_names = (
        "native_median_ms",
        "native_p95_ms",
        "reference_median_ms",
        "reference_p95_ms",
        "paired_speedup_ratio",
        "paired_delta_ms",
    )
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
                "logical_tokens": expected.logical_tokens,
                "aggregation_unit": (
                    "one parsed marker from one successful outer repeat-control process; "
                    "each marker's timings are that process's internally paired ABBA CUDA-event control"
                ),
                "observations": observations,
                "outer_process_metrics": {
                    metric: summarize_values(
                        [float(item[metric]) for item in observations],
                        bootstrap_resamples=bootstrap_resamples,
                        bootstrap_seed=bootstrap_seed,
                    )
                    for metric in metric_names
                },
            }
        )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "scope": {
            "classification": "paired-native-bf16-paged-split-gqa-d128-operator-control",
            "geometry": {
                "batch": 1,
                "query_heads": 16,
                "key_value_heads": 2,
                "head_size": 128,
                "page_size": 16,
            },
            "full_model_serving": False,
            "vllm_comparison": False,
            "limitations": [
                "This analysis aggregates only prepared paged-decode paired CUDA-event operator controls.",
                "It is not a full-model, serving, latency, throughput, stability, correctness, or vLLM comparison result.",
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
            {
                "case": case.label,
                "logical_tokens": case.logical_tokens,
                "marker_prefix": MARKER_PREFIX,
            }
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
        print(f"n03a_paged_d128_repeat_summary: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
