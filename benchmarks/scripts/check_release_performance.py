#!/usr/bin/env python3
"""Check a release candidate against the immutable PR15 performance baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
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
BASELINE_SHA256 = "38ac9581c68ef1b229849529574755326f21d94a0b6787bc1e9f69c2cb9f6209"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


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


def _load_json_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise InputError(f"cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(raw, object_pairs_hook=_pairs_no_duplicates)
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


def _literal(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        raise InputError(f"{path}: expected {expected!r}, got {value!r}")


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest_file(path: Path, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise InputError(f"cannot hash {label} {path}: {error}") from error
    return digest.hexdigest()


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
        "correctness_report_sha256",
    ]:
        _sha256(binding[field], f"baseline.measurement_binding.{field}")
    _literal(binding["semantic_class"], "E0", "baseline.semantic_class")
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
        "candidate_id": _string(row["candidate_id"], "candidate.candidate_id"),
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


def _load_raw_runs(
    paths: Sequence[Path | str], candidate: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    run_paths = [Path(path) for path in paths]
    if len(run_paths) != 5:
        raise InputError(
            f"candidate: expected exactly 5 independent run files, got {len(run_paths)}"
        )
    loaded: list[tuple[Path, dict[str, Any], str]] = []
    try:
        for path in run_paths:
            run, raw = _load_json_bytes(path, "raw native profile run")
            native_profile._validate_run(run, str(path))
            if run["role"] != "candidate":
                raise InputError(
                    f"{path}.role: expected 'candidate', got {run['role']!r}"
                )
            loaded.append((path, run, _digest_bytes(raw)))
        if sorted(run["pair_index"] for _, run, _ in loaded) != list(range(1, 6)):
            raise InputError("candidate: pair_index values must be exactly 1..5")
        loaded.sort(key=lambda row: row[1]["pair_index"])
        runs = [run for _, run, _ in loaded]
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
        native_profile._require_equal(
            [native_profile._request_identity(run) for run in runs],
            "release candidate raw request identities",
        )
    except native_profile.ComparabilityError as error:
        raise ComparabilityError(str(error)) from error
    except native_profile.InputError as error:
        raise InputError(str(error)) from error

    declared_by_pair = {
        binding["pair_index"]: binding for binding in candidate["raw_runs"]
    }
    for path, run, actual_digest in loaded:
        pair_index = run["pair_index"]
        if declared_by_pair.get(pair_index) != {
            "pair_index": pair_index,
            "run_id": run["run_id"],
            "sha256": actual_digest,
        }:
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
        if source[field] != expected:
            raise InputError(
                f"raw source.{field} does not match candidate source binding"
            )
    if source["runtime_flag"] != {
        "name": "execution_completion",
        "value": "iteration-batch",
    }:
        raise ComparabilityError(
            "raw source.runtime_flag must select execution_completion=iteration-batch"
        )

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
    for name, raw_value in [
        ("model", raw_model),
        ("environment", raw_environment),
        ("workload", raw_workload),
    ]:
        if candidate[name] != raw_value:
            raise ComparabilityError(
                f"candidate {name} does not match its raw native profile runs"
            )
    if environment["software"]["container_image_sha256"] != candidate_source[
        "profile_image_sha256"
    ]:
        raise InputError(
            "raw environment producer image does not match profile_image_sha256"
        )

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
    if candidate["run_summary"] != derived_summary:
        raise InputError("candidate.run_summary does not equal raw-derived summary")
    if candidate["metrics"] != derived_metrics:
        raise InputError("candidate.metrics do not equal raw-derived R7 metrics")
    return runs, derived_summary, derived_metrics


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

        if not profile_image_id.startswith("sha256:"):
            raise InputError("--profile-image-id: expected sha256:<lowercase digest>")
        if not release_image_id.startswith("sha256:"):
            raise InputError("--release-image-id: expected sha256:<lowercase digest>")
        profile_image_digest = profile_image_id.removeprefix("sha256:")
        release_image_digest = release_image_id.removeprefix("sha256:")
        _sha256(profile_image_digest, "--profile-image-id")
        _sha256(release_image_digest, "--release-image-id")
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
        weights_digest = _digest_file(Path(weights), "model weights")
        tokenizer_digest = _digest_file(Path(tokenizer), "tokenizer")
        if candidate["model"]["weights_sha256"] != weights_digest:
            raise InputError("candidate.model.weights_sha256 does not match --weights")
        if candidate["model"]["tokenizer_sha256"] != tokenizer_digest:
            raise InputError("candidate.model.tokenizer_sha256 does not match --tokenizer")

        _runs, raw_summary, raw_metrics = _load_raw_runs(run_paths, candidate)

        for field in ["model", "environment", "workload"]:
            if candidate[field] != baseline[field]:
                raise ComparabilityError(f"candidate {field} differs from baseline lane")
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
