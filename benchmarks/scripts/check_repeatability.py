#!/usr/bin/env python3
"""Evaluate the PR-01 repeatability gate from canonical JSONL trials.

`matrix.yaml` is deliberately JSON syntax, so this script has no YAML or
statistics-package dependency.  It prints one JSON report and exits nonzero
for malformed/non-comparable input as well as for a threshold failure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CELL_KEYS = ("concurrency", "prompt_tokens", "output_tokens", "warm_state")
HASH_FIELDS = ("matrix_sha256", "prompts_sha256", "lane_manifest_sha256")
COMPARABILITY_FIELDS = (
    "scope",
    "matrix_id",
    "matrix_sha256",
    "prompts_sha256",
    "lane_manifest_sha256",
    "environment_id",
    "model_revision",
    "engine_revision",
)
THRESHOLD_KEYS = (
    "warm_p50_cv_max",
    "warm_p95_cv_max",
    "throughput_cv_max",
    "cold_model_load_p50_cv_max",
    "peak_vram_relative_range_max",
    "failure_count_max",
)
SUCCESS_STATUSES = {"ok", "pass", "passed", "success", "succeeded"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_MATRIX = REPOSITORY_ROOT / "benchmarks/matrix.yaml"
CANONICAL_PYTHON_VERSION = "3.13.15"
CANONICAL_PYTHON_LINUX_X86_64_SHA256 = (
    "ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866"
)
REPORT_CONTRACT_VERSION = "rustinfer.repeatability.v2"


class InputError(ValueError):
    """The matrix or a trial does not satisfy the canonical input contract."""


class ComparabilityError(InputError):
    """Well-formed trials cannot participate in one direct comparison."""


def r7(values: Iterable[float], probability: float) -> float:
    """Return a Hyndman-Fan type-7 quantile.

    For sorted samples x[0..n-1], h=(n-1)*p and the result linearly
    interpolates x[floor(h)] and x[ceil(h)].
    """

    try:
        ordered = sorted(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError("R7 observations must be finite JSON numbers") from error
    if not ordered:
        raise InputError("R7 quantile requires at least one observation")
    if not 0.0 <= probability <= 1.0:
        raise InputError("R7 probability must be in [0, 1]")
    if any(not math.isfinite(value) for value in ordered):
        raise InputError("R7 observations must be finite")
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * probability
    lower = math.floor(h)
    fraction = h - lower
    if fraction == 0.0:
        return ordered[lower]
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def sample_cv(values: Iterable[float]) -> float:
    """Return sample standard deviation divided by the absolute mean."""

    try:
        observations = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError("sample CV observations must be finite JSON numbers") from error
    if len(observations) < 2:
        raise InputError("sample CV requires at least two independent runs")
    if any(not math.isfinite(value) for value in observations):
        raise InputError("sample CV observations must be finite")
    mean = statistics.fmean(observations)
    if mean == 0.0:
        if all(value == 0.0 for value in observations):
            return 0.0
        raise InputError("sample CV is undefined when the mean is zero")
    return statistics.stdev(observations) / abs(mean)


def relative_range(values: Iterable[float]) -> float:
    """Return (maximum - minimum) / R7 median."""

    try:
        observations = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError("relative-range observations must be finite JSON numbers") from error
    if not observations:
        raise InputError("relative range requires at least one observation")
    median = r7(observations, 0.5)
    spread = max(observations) - min(observations)
    if median == 0.0:
        if spread == 0.0:
            return 0.0
        raise InputError("relative range is undefined when the median is zero")
    if median < 0.0:
        raise InputError("relative range requires a non-negative median")
    return spread / median


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if not _is_number(value):
        raise InputError(f"{label} must be a JSON number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise InputError(f"{label} must be a finite JSON number") from error
    if not math.isfinite(result):
        raise InputError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise InputError(f"{label} must be >= {minimum:g}")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise InputError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be a JSON object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{label} must be a non-empty string")
    return value


def _load_matrix(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise InputError(f"cannot read matrix {path}: {error}") from error
    try:
        matrix = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputError(
            f"{path} must contain UTF-8 JSON (YAML-only syntax is not accepted): {error}"
        ) from error
    if not isinstance(matrix, dict):
        raise InputError("matrix root must be a JSON object")
    return matrix, hashlib.sha256(raw).hexdigest()


def _gate_contract(
    matrix: Mapping[str, Any],
) -> tuple[
    dict[str, float],
    int,
    list[tuple[int, int, int, str]],
    dict[str, int],
]:
    gate = _mapping(matrix.get("repeatability_gate"), "matrix.repeatability_gate")
    thresholds_raw = _mapping(
        gate.get("thresholds"), "matrix.repeatability_gate.thresholds"
    )
    thresholds: dict[str, float] = {}
    for key in THRESHOLD_KEYS:
        thresholds[key] = _finite_number(
            thresholds_raw.get(key),
            f"matrix.repeatability_gate.thresholds.{key}",
            minimum=0.0,
        )

    cells = gate.get("cells")
    if not isinstance(cells, list) or not cells:
        raise InputError(
            "matrix.repeatability_gate.cells must be a non-empty list of exact cells"
        )

    measurement = _mapping(matrix.get("measurement"), "matrix.measurement")
    required_runs = _integer(
        measurement.get("independent_runs"),
        "matrix.measurement.independent_runs",
        minimum=2,
    )
    trials_per_state: dict[str, int] = {}
    for warm_state in ("warm", "cold"):
        state_measurement = _mapping(
            measurement.get(warm_state), f"matrix.measurement.{warm_state}"
        )
        trials_per_state[warm_state] = _integer(
            state_measurement.get("measured_iterations_per_run"),
            f"matrix.measurement.{warm_state}.measured_iterations_per_run",
            minimum=1,
        )

    expected: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int, int, str]] = set()
    for index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell, f"repeatability_gate cell {index}")
        nested = cell.get("workload")
        if nested is not None:
            combined = dict(_mapping(nested, f"repeatability_gate cell {index}.workload"))
            if "warm_state" in cell:
                combined["warm_state"] = cell["warm_state"]
            cell = combined
        key = _cell_key(cell, f"repeatability_gate cell {index}")
        if key in seen:
            raise InputError(f"duplicate repeatability gate cell {_cell_dict(key)}")
        seen.add(key)
        expected.append(key)
    return thresholds, required_runs, expected, trials_per_state


def _expand_trial_paths(paths: Sequence[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            expanded.append(path)
        else:
            raise InputError(f"trial input does not exist: {path}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise InputError("no trial JSONL files were found")
    return unique


def _load_trials(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    files = _expand_trial_paths(paths)
    trials: list[dict[str, Any]] = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise InputError(f"cannot read trial file {path}: {error}") from error
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise InputError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise InputError(f"{path}:{line_number}: trial row must be an object")
            row = dict(row)
            row["__source"] = f"{path}:{line_number}"
            trials.append(row)
    if not trials:
        raise InputError("trial JSONL inputs contain no rows")
    return trials, [str(path) for path in files]


def _load_contract_validator() -> Any:
    path = Path(__file__).with_name("validate_contract.py")
    spec = importlib.util.spec_from_file_location(
        "rustinfer_repeatability_contract_validator", path
    )
    if spec is None or spec.loader is None:
        raise InputError(f"cannot load benchmark contract validator {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (OSError, ImportError) as error:
        raise InputError(f"cannot load benchmark contract validator: {error}") from error
    return module


def _validate_matrix_contract(
    matrix: dict[str, Any], matrix_path: Path, *, allow_noncanonical_matrix: bool
) -> str:
    canonical = matrix_path.expanduser().resolve() == CANONICAL_MATRIX.resolve()
    if not canonical:
        if allow_noncanonical_matrix:
            return "noncanonical-test-bypass"
        raise InputError(
            "repeatability checker requires the canonical benchmarks/matrix.yaml; "
            "--allow-noncanonical-matrix is test-only"
        )
    validator = _load_contract_validator()
    try:
        validator.validate_matrix(matrix, REPOSITORY_ROOT)
    except (validator.ContractError, OSError) as error:
        raise InputError(f"matrix contract validation failed: {error}") from error
    if platform.python_version() != CANONICAL_PYTHON_VERSION:
        raise InputError(
            "canonical repeatability checker requires Python "
            f"{CANONICAL_PYTHON_VERSION}, found {platform.python_version()}"
        )
    try:
        python_sha256 = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    except OSError as error:
        raise InputError(f"cannot hash canonical checker Python executable: {error}") from error
    if (
        sys.implementation.name != "cpython"
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
        or python_sha256 != CANONICAL_PYTHON_LINUX_X86_64_SHA256
    ):
        raise InputError(
            "canonical repeatability checker requires the pinned CPython Linux "
            "x86_64 executable"
        )
    return "passed"


def _validate_raw_contract(
    matrix: dict[str, Any], matrix_path: Path, trial_files: Sequence[Path]
) -> None:
    """Apply the shared result schema and cross-file checks before statistics."""

    validator = _load_contract_validator()
    schema_path = REPOSITORY_ROOT / "benchmarks/schemas/result.schema.json"
    prompts_path = REPOSITORY_ROOT / "benchmarks/prompts.jsonl"
    try:
        schema = validator.validate_schema_document(
            json.loads(schema_path.read_text(encoding="utf-8")),
            "result",
            str(schema_path),
        )
        lane_paths: dict[str, Path] = {}
        lanes_by_implementation: dict[str, dict[str, Any]] = {}
        manifests = matrix.get("lane_manifests")
        if not isinstance(manifests, list) or not manifests:
            raise InputError("matrix.lane_manifests must be a non-empty string array")
        for relative in manifests:
            if not isinstance(relative, str):
                raise InputError("matrix.lane_manifests must contain only strings")
            lane_path = (REPOSITORY_ROOT / relative).resolve()
            if REPOSITORY_ROOT not in lane_path.parents:
                raise InputError(f"lane manifest escapes repository: {relative!r}")
            lane = validator.validate_lane_manifest(
                json.loads(lane_path.read_text(encoding="utf-8")),
                matrix,
                str(lane_path),
            )
            validator.validate_dependency_project(REPOSITORY_ROOT, lane)
            lane_paths[lane["lane_id"]] = lane_path
            lanes_by_implementation[lane["implementation_id"]] = lane
        for trial_file in trial_files:
            validator.validate_result_file(
                trial_file,
                schema,
                matrix,
                matrix_path,
                prompts_path,
                lane_paths,
                lanes_by_implementation,
            )
    except InputError:
        raise
    except validator.ComparabilityContractError as error:
        raise ComparabilityError(
            f"raw result is incomparable with the supplied contract: {error}"
        ) from error
    except (validator.ContractError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InputError(f"raw result contract validation failed: {error}") from error


def _cell_key(
    source: Mapping[str, Any], label: str
) -> tuple[int, int, int, str]:
    concurrency = _integer(source.get("concurrency"), f"{label}.concurrency", minimum=1)
    prompt_tokens = _integer(
        source.get("prompt_tokens"), f"{label}.prompt_tokens", minimum=1
    )
    output_tokens = _integer(
        source.get("output_tokens"), f"{label}.output_tokens", minimum=1
    )
    warm_state = source.get("warm_state")
    if warm_state not in {"warm", "cold"}:
        raise InputError(f"{label}.warm_state must be 'warm' or 'cold'")
    return concurrency, prompt_tokens, output_tokens, warm_state


def _cell_dict(key: tuple[int, int, int, str]) -> dict[str, Any]:
    return dict(zip(CELL_KEYS, key))


def _validate_and_group(
    trials: list[dict[str, Any]], matrix: Mapping[str, Any], matrix_sha256: str
) -> tuple[
    dict[tuple[int, int, int, str], dict[Any, list[dict[str, Any]]]],
    dict[str, Any],
]:
    baseline: dict[str, Any] = {}
    groups: dict[
        tuple[int, int, int, str], dict[Any, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    identities: set[tuple[tuple[int, int, int, str], Any, int]] = set()
    expected_matrix_id = matrix.get("matrix_id")
    expected_scope = matrix.get("benchmark_scope", "end-to-end")
    if expected_scope != "end-to-end":
        raise InputError(
            "repeatability checker only supports an end-to-end benchmark matrix"
        )
    expected_model_revision = None
    if isinstance(matrix.get("model"), dict):
        expected_model_revision = matrix["model"].get("revision")
    axes = matrix.get("axes") if isinstance(matrix.get("axes"), dict) else {}

    for row in trials:
        source = row["__source"]
        for field in COMPARABILITY_FIELDS:
            value = _nonempty_string(row.get(field), f"{source}: {field}")
            if field in HASH_FIELDS and (
                len(value) != 64
                or value.lower() != value
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise InputError(
                    f"{source}: {field} must be 64 lowercase hexadecimal characters"
                )
            if field not in baseline:
                baseline[field] = value
            elif value != baseline[field]:
                raise ComparabilityError(
                    f"{source}: incomparable {field}: {value!r} != {baseline[field]!r}"
                )

        if row["scope"] != expected_scope:
            raise ComparabilityError(
                f"{source}: repeatability gate requires scope {expected_scope!r}, "
                f"got {row['scope']!r}"
            )
        if row["matrix_sha256"] != matrix_sha256:
            raise ComparabilityError(
                f"{source}: matrix_sha256 does not match the bytes of the supplied matrix"
            )
        if expected_matrix_id is not None and row["matrix_id"] != expected_matrix_id:
            raise ComparabilityError(
                f"{source}: matrix_id {row['matrix_id']!r} does not match matrix.matrix_id "
                f"{expected_matrix_id!r}"
            )
        if (
            expected_model_revision is not None
            and row["model_revision"] != expected_model_revision
        ):
            raise ComparabilityError(
                f"{source}: model_revision {row['model_revision']!r} does not match "
                f"matrix.model.revision {expected_model_revision!r}"
            )

        provenance = _mapping(row.get("provenance"), f"{source}: provenance")
        if provenance.get("git_dirty") is not False:
            raise ComparabilityError(f"{source}: provenance.git_dirty must be false")
        git_revision = _nonempty_string(
            provenance.get("git_revision"), f"{source}: provenance.git_revision"
        )
        if "git_revision" not in baseline:
            baseline["git_revision"] = git_revision
        elif git_revision != baseline["git_revision"]:
            raise ComparabilityError(
                f"{source}: incomparable provenance.git_revision: "
                f"{git_revision!r} != {baseline['git_revision']!r}"
            )

        run_id = _nonempty_string(row.get("run_id"), f"{source}: run_id")
        trial_index = _integer(
            row.get("trial_index"), f"{source}: trial_index", minimum=1
        )
        failure_count = _integer(row.get("failure_count"), f"{source}: failure_count")
        metrics = _mapping(row.get("metrics"), f"{source}: metrics")
        workload = _mapping(row.get("workload"), f"{source}: workload")
        merged_workload = dict(workload)
        merged_workload["warm_state"] = row.get("warm_state")
        if "warm_state" in workload and workload["warm_state"] != row.get("warm_state"):
            raise InputError(
                f"{source}: workload.warm_state and row warm_state must agree"
            )
        key = _cell_key(merged_workload, f"{source}: workload")
        for axis_name, value in zip(CELL_KEYS, key):
            allowed = axes.get(axis_name)
            if isinstance(allowed, list) and value not in allowed:
                raise ComparabilityError(
                    f"{source}: workload.{axis_name}={value!r} is outside matrix axes"
                )
        identity = (key, run_id, trial_index)
        if identity in identities:
            raise InputError(
                f"{source}: duplicate trial_index {trial_index} for run {run_id!r} "
                f"and cell {_cell_dict(key)}"
            )
        identities.add(identity)
        row["__failure_count"] = failure_count
        row["__metrics"] = metrics
        groups[key][run_id].append(row)

    baseline["git_dirty"] = False
    return groups, baseline


def _trial_succeeded(row: Mapping[str, Any]) -> bool:
    if row["__failure_count"] != 0:
        return False
    status = row.get("status")
    if status is None:
        return True
    return isinstance(status, str) and status.lower() in SUCCESS_STATUSES


def _request_succeeded(request: Mapping[str, Any]) -> bool:
    failure_reason = request.get("failure_reason")
    if failure_reason is not None and failure_reason != "":
        return False
    status = request.get("status")
    if status is None:
        return True
    return isinstance(status, str) and status.lower() in SUCCESS_STATUSES


def _metric(row: Mapping[str, Any], name: str, *, minimum: float = 0.0) -> float:
    return _finite_number(
        row["__metrics"].get(name), f"{row['__source']}: metrics.{name}", minimum=minimum
    )


def _run_summary(
    run_id: Any, rows: Sequence[dict[str, Any]], warm_state: str
) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: row["trial_index"])
    successful = [row for row in ordered_rows if _trial_succeeded(row)]
    summary: dict[str, Any] = {
        "run_id": run_id,
        "trial_count": len(ordered_rows),
        "successful_trial_count": len(successful),
        "failure_count": sum(row["__failure_count"] for row in ordered_rows),
    }
    if not successful:
        raise InputError(f"run {run_id!r} has no successful trials")

    throughput = [
        _metric(row, "output_tokens_per_second", minimum=0.0) for row in successful
    ]
    peak_vram = [_metric(row, "peak_vram_bytes", minimum=0.0) for row in successful]
    summary["throughput_tokens_per_second"] = {
        "observation_count": len(throughput),
        "r7_p50": r7(throughput, 0.5),
    }
    summary["peak_vram_bytes"] = {
        "observation_count": len(peak_vram),
        "maximum": max(peak_vram),
    }

    if warm_state == "warm":
        end_to_end: list[float] = []
        request_mean_tpot: list[float] = []
        pooled_itl: list[float] = []
        for row in successful:
            requests = row.get("requests")
            if not isinstance(requests, list):
                raise InputError(f"{row['__source']}: requests must be a JSON list")
            for request_index, request_raw in enumerate(requests):
                request = _mapping(
                    request_raw, f"{row['__source']}: requests[{request_index}]"
                )
                if _request_succeeded(request):
                    end_to_end.append(
                        _finite_number(
                            request.get("end_to_end_ms"),
                            f"{row['__source']}: requests[{request_index}].end_to_end_ms",
                            minimum=0.0,
                        )
                    )
                    request_mean_tpot.append(
                        _finite_number(
                            request.get("mean_tpot_ms"),
                            f"{row['__source']}: requests[{request_index}].mean_tpot_ms",
                            minimum=0.0,
                        )
                    )
                    raw_itl = request.get("itl_ms")
                    if not isinstance(raw_itl, list):
                        raise InputError(
                            f"{row['__source']}: requests[{request_index}].itl_ms "
                            "must be an array"
                        )
                    pooled_itl.extend(
                        _finite_number(
                            value,
                            f"{row['__source']}: requests[{request_index}]"
                            f".itl_ms[{itl_index}]",
                            minimum=0.0,
                        )
                        for itl_index, value in enumerate(raw_itl)
                    )
        if not end_to_end:
            raise InputError(f"warm run {run_id!r} has no successful request observations")
        if not pooled_itl:
            raise InputError(f"warm run {run_id!r} has no per-token ITL observations")
        summary["end_to_end_ms"] = {
            "observation_count": len(end_to_end),
            "r7_p50": r7(end_to_end, 0.5),
            "r7_p95": r7(end_to_end, 0.95),
        }
        summary["request_mean_tpot_ms"] = {
            "observation_count": len(request_mean_tpot),
            "r7_p50": r7(request_mean_tpot, 0.5),
            "r7_p95": r7(request_mean_tpot, 0.95),
        }
        summary["pooled_itl_ms"] = {
            "observation_count": len(pooled_itl),
            "r7_p50": r7(pooled_itl, 0.5),
            "r7_p95": r7(pooled_itl, 0.95),
        }
    else:
        model_load = [_metric(row, "model_load_ms", minimum=0.0) for row in successful]
        summary["model_load_ms"] = {
            "observation_count": len(model_load),
            "r7_p50": r7(model_load, 0.5),
        }
    return summary


def _check(name: str, value: float, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "operator": "<=",
        "threshold": threshold,
        "passed": value <= threshold,
    }


def _token_identity_errors(
    runs: Mapping[Any, Sequence[dict[str, Any]]],
) -> list[str]:
    expected: dict[tuple[int, int], tuple[str, str, str]] = {}
    errors: list[str] = []
    for run_id in sorted(runs):
        for row in runs[run_id]:
            if not _trial_succeeded(row):
                continue
            requests = row["requests"]
            for request_index, request in enumerate(requests):
                if not _request_succeeded(request):
                    continue
                position = (row["trial_index"], request_index)
                identity = (
                    request["prompt_id"],
                    request["prompt_token_ids_sha256"],
                    request["generated_token_ids_sha256"],
                )
                prior = expected.setdefault(position, identity)
                if identity != prior:
                    errors.append(
                        "independent runs differ at trial_index "
                        f"{position[0]}, request position {position[1]}: "
                        "prompt/generated token identity is not stable "
                        f"(run {run_id!r})"
                    )
    return errors


def _evaluate_cell(
    key: tuple[int, int, int, str],
    runs: Mapping[Any, Sequence[dict[str, Any]]],
    thresholds: Mapping[str, float],
    required_runs: int,
    required_trials_per_run: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workload": _cell_dict(key),
        "required_independent_runs": required_runs,
        "independent_run_count": len(runs),
        "required_trials_per_run": required_trials_per_run,
        "run_summaries": [],
        "statistics": {},
        "checks": [],
        "errors": [],
    }
    if len(runs) != required_runs:
        result["errors"].append(
            f"requires exactly {required_runs} independent runs, found {len(runs)}"
        )
    result["errors"].extend(_token_identity_errors(runs))

    for run_id in sorted(runs):
        trial_indices = sorted(row["trial_index"] for row in runs[run_id])
        expected_indices = list(range(1, required_trials_per_run + 1))
        if trial_indices != expected_indices:
            result["errors"].append(
                f"run {run_id!r} requires trial_index 1..{required_trials_per_run}; "
                f"found {trial_indices}"
            )
        try:
            result["run_summaries"].append(_run_summary(run_id, runs[run_id], key[3]))
        except InputError as error:
            result["errors"].append(str(error))

    summaries = result["run_summaries"]
    if len(summaries) == len(runs) and len(summaries) >= 2:
        try:
            throughput_cv = sample_cv(
                item["throughput_tokens_per_second"]["r7_p50"] for item in summaries
            )
            peak_range = relative_range(
                item["peak_vram_bytes"]["maximum"] for item in summaries
            )
            result["statistics"].update(
                {
                    "throughput_p50_sample_cv": throughput_cv,
                    "peak_vram_max_relative_range": peak_range,
                }
            )
            if key[3] == "warm":
                p50_cv = sample_cv(item["end_to_end_ms"]["r7_p50"] for item in summaries)
                p95_cv = sample_cv(item["end_to_end_ms"]["r7_p95"] for item in summaries)
                result["statistics"].update(
                    {
                        "warm_end_to_end_r7_p50_sample_cv": p50_cv,
                        "warm_end_to_end_r7_p95_sample_cv": p95_cv,
                        "warm_request_mean_tpot_r7_p50_sample_cv": sample_cv(
                            item["request_mean_tpot_ms"]["r7_p50"]
                            for item in summaries
                        ),
                        "warm_request_mean_tpot_r7_p95_sample_cv": sample_cv(
                            item["request_mean_tpot_ms"]["r7_p95"]
                            for item in summaries
                        ),
                        "warm_pooled_itl_r7_p50_sample_cv": sample_cv(
                            item["pooled_itl_ms"]["r7_p50"]
                            for item in summaries
                        ),
                        "warm_pooled_itl_r7_p95_sample_cv": sample_cv(
                            item["pooled_itl_ms"]["r7_p95"]
                            for item in summaries
                        ),
                    }
                )
                result["checks"].extend(
                    [
                        _check(
                            "throughput_cv_max",
                            throughput_cv,
                            thresholds["throughput_cv_max"],
                        ),
                        _check(
                            "peak_vram_relative_range_max",
                            peak_range,
                            thresholds["peak_vram_relative_range_max"],
                        ),
                        _check("warm_p50_cv_max", p50_cv, thresholds["warm_p50_cv_max"]),
                        _check("warm_p95_cv_max", p95_cv, thresholds["warm_p95_cv_max"]),
                    ]
                )
            else:
                result["checks"].append(
                    _check(
                        "peak_vram_relative_range_max",
                        peak_range,
                        thresholds["peak_vram_relative_range_max"],
                    )
                )
                load_cv = sample_cv(
                    item["model_load_ms"]["r7_p50"] for item in summaries
                )
                result["statistics"]["cold_model_load_r7_p50_sample_cv"] = load_cv
                result["checks"].append(
                    _check(
                        "cold_model_load_p50_cv_max",
                        load_cv,
                        thresholds["cold_model_load_p50_cv_max"],
                    )
                )
        except InputError as error:
            result["errors"].append(str(error))

    failures = sum(
        row["__failure_count"] for run_rows in runs.values() for row in run_rows
    )
    result["statistics"]["failure_count"] = failures
    result["checks"].append(
        _check("failure_count_max", float(failures), thresholds["failure_count_max"])
    )
    result["passed"] = not result["errors"] and all(
        check["passed"] for check in result["checks"]
    )
    return result


def _definitions() -> dict[str, str]:
    return {
        "r7": "sort x; h=(n-1)*p; linearly interpolate x[floor(h)] and x[ceil(h)]",
        "warm_latency": (
            "within each run and warm cell, flatten successful "
            "requests[].end_to_end_ms across trials and compute R7 p50/p95"
        ),
        "request_mean_tpot": (
            "within each run and warm cell, flatten successful request-level "
            "requests[].mean_tpot_ms observations and compute R7 p50/p95"
        ),
        "pooled_itl": (
            "within each run and warm cell, pool every actual per-token value from "
            "successful requests[].itl_ms arrays and compute R7 p50/p95"
        ),
        "throughput": (
            "within each run and cell, compute R7 p50 of successful-trial "
            "metrics.output_tokens_per_second; report its sample CV for every cell, "
            "but apply throughput_cv_max only to warm cells because a cold run has "
            "one diagnostic first-request observation"
        ),
        "cold_model_load": (
            "within each run and cold cell, compute R7 p50 of successful-trial "
            "metrics.model_load_ms"
        ),
        "sample_cv": "across independent runs, sample standard deviation / abs(mean)",
        "peak_vram_relative_range": (
            "take each run's maximum successful-trial metrics.peak_vram_bytes, then "
            "(max-min)/R7 median across runs"
        ),
        "failures": "sum root failure_count across all trials in a cell",
        "token_identity": (
            "for each cell, trial_index, and request position, require identical "
            "prompt_token_ids_sha256 and generated_token_ids_sha256 across runs"
        ),
        "coverage": (
            "every configured exact cell must contain the same run_ids; each run must "
            "contain trial_index 1..measurement.<warm_state>.measured_iterations_per_run"
        ),
    }


def evaluate(
    matrix_path: Path,
    trial_paths: Sequence[Path],
    *,
    allow_noncanonical_matrix: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable repeatability report."""

    report: dict[str, Any] = {
        "contract_version": REPORT_CONTRACT_VERSION,
        "status": "error",
        "passed": False,
        "definitions": _definitions(),
        "matrix": str(matrix_path),
        "trial_files": [],
        "comparability": {},
        "cells": [],
        "errors": [],
    }
    try:
        matrix, matrix_sha256 = _load_matrix(matrix_path)
        report["matrix_sha256"] = matrix_sha256
        report["matrix_contract_validation"] = _validate_matrix_contract(
            matrix,
            matrix_path,
            allow_noncanonical_matrix=allow_noncanonical_matrix,
        )
        thresholds, required_runs, expected_cells, trials_per_state = _gate_contract(
            matrix
        )
        expanded_trial_files = _expand_trial_paths(trial_paths)
        report["trial_files"] = [str(path) for path in expanded_trial_files]
        _validate_raw_contract(matrix, matrix_path, expanded_trial_files)
        trials, _ = _load_trials(expanded_trial_files)
        groups, comparability = _validate_and_group(trials, matrix, matrix_sha256)
        report["comparability"] = comparability
        report["thresholds"] = thresholds

        selected_keys = expected_cells
        expected_set = set(expected_cells)
        report["ignored_trial_count"] = sum(
            len(rows)
            for key, runs in groups.items()
            if key not in expected_set
            for rows in runs.values()
        )
        if report["ignored_trial_count"]:
            report["errors"].append(
                "trial inputs contain rows outside the exact repeatability gate cells: "
                f"{report['ignored_trial_count']}"
            )
        present_run_sets = [set(groups[key]) for key in selected_keys if key in groups]
        if present_run_sets and any(
            run_ids != present_run_sets[0] for run_ids in present_run_sets[1:]
        ):
            report["errors"].append(
                "all repeatability cells must contain the same independent run_ids"
            )
        for key in selected_keys:
            if key not in groups:
                report["errors"].append(
                    f"matrix repeatability cell has no trials: {_cell_dict(key)}"
                )
                continue
            report["cells"].append(
                _evaluate_cell(
                    key,
                    groups[key],
                    thresholds,
                    required_runs,
                    trials_per_state[key[3]],
                )
            )
        report["passed"] = (
            not report["errors"]
            and bool(report["cells"])
            and all(cell["passed"] for cell in report["cells"])
        )
        report["status"] = "passed" if report["passed"] else "failed"
    except ComparabilityError as error:
        report["status"] = "incomparable"
        report["errors"].append(str(error))
    except InputError as error:
        report["errors"].append(str(error))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", required=True, type=Path, help="JSON-syntax benchmark matrix.yaml"
    )
    parser.add_argument(
        "trials",
        nargs="+",
        type=Path,
        help="trial JSONL files or directories searched recursively for *.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, help="write the JSON report here instead of stdout"
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument(
        "--allow-noncanonical-matrix",
        action="store_true",
        help="allow a partial synthetic matrix for offline unit tests only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.matrix,
        args.trials,
        allow_noncanonical_matrix=args.allow_noncanonical_matrix,
    )
    rendered = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=None if args.compact else 2,
    )
    if args.output:
        try:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as error:
            fallback = {
                "contract_version": REPORT_CONTRACT_VERSION,
                "status": "error",
                "passed": False,
                "errors": [f"cannot write report {args.output}: {error}"],
            }
            print(json.dumps(fallback, sort_keys=True), file=sys.stderr)
            return 1
    else:
        print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
