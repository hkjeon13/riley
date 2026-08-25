#!/usr/bin/env python3
"""Check paired native baseline/candidate profiling evidence.

The checker is deliberately independent of the PR-01 matrix and lane contract.
It uses only the Python standard library, validates a closed v1 input shape,
prints one strict v1 report, and fails closed on malformed or incomparable data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence


RUN_SCHEMA_VERSION = "rustinfer.native-profile-run.v1"
REPORT_SCHEMA_VERSION = "rustinfer.native-profile-pair-report.v1"
REQUIRED_RUNS = 5
TTFT_P95_RATIO_MAX = 1.05
TPOT_P95_RATIO_MAX = 1.05
THROUGHPUT_RATIO_MIN = 0.95
PRIMARY_IMPROVEMENT_MIN = 0.05
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_CONCURRENCY = 8
MAX_PROMPT_TOKENS = 8_192
MIN_OUTPUT_TOKENS = 2
MAX_OUTPUT_TOKENS = 512
MAX_WARMUPS = 100
MAX_MEASURED_ITERATIONS = 100
U64_MAX = 18_446_744_073_709_551_615
CANONICAL_PRIMARY_METRIC = "aggregate.host.execute_ns"
CORRECTNESS_GATES = {
    "residual_rmsnorm": "pr15-fused-residual-rmsnorm-exact-v1",
    "execution_completion": "pr15-iteration-command-batch-exact-v1",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

REQUEST_PRIMARY_METRICS = {
    "request.ttft_median_ms": ("ttft_ms", 0.50, "lower"),
    "request.ttft_p95_ms": ("ttft_ms", 0.95, "lower"),
    "request.tpot_median_ms": ("tpot_ms", 0.50, "lower"),
    "request.tpot_p95_ms": ("tpot_ms", 0.95, "lower"),
    "request.e2e_median_ms": ("e2e_ms", 0.50, "lower"),
    "request.e2e_p95_ms": ("e2e_ms", 0.95, "lower"),
}
AGGREGATE_PRIMARY_METRICS = {
    "aggregate.throughput_output_tokens_per_second": (
        ("throughput_output_tokens_per_second",),
        "higher",
    ),
    "aggregate.host.plan_ns": (("host", "plan_ns"), "lower"),
    "aggregate.host.execute_ns": (("host", "execute_ns"), "lower"),
    "aggregate.host.sampling_ns": (("host", "sampling_ns"), "lower"),
    "aggregate.host.commit_ns": (("host", "commit_ns"), "lower"),
    "aggregate.cuda.stream_span_ns": (("cuda", "stream_span_ns"), "lower"),
    "aggregate.cuda.idle_ns": (("cuda", "idle_ns"), "lower"),
    "aggregate.counters.iterations": (("counters", "iterations"), "lower"),
    "aggregate.counters.kernel_launches": (
        ("counters", "kernel_launches"),
        "lower",
    ),
    "aggregate.counters.copies.h2d_calls": (
        ("counters", "copies", "h2d_calls"),
        "lower",
    ),
    "aggregate.counters.copies.h2d_bytes": (
        ("counters", "copies", "h2d_bytes"),
        "lower",
    ),
    "aggregate.counters.copies.d2h_calls": (
        ("counters", "copies", "d2h_calls"),
        "lower",
    ),
    "aggregate.counters.copies.d2h_bytes": (
        ("counters", "copies", "d2h_bytes"),
        "lower",
    ),
    "aggregate.counters.allocations.device_allocations": (
        ("counters", "allocations", "device_allocations"),
        "lower",
    ),
    "aggregate.counters.allocations.device_frees": (
        ("counters", "allocations", "device_frees"),
        "lower",
    ),
    "aggregate.counters.allocations.pinned_allocations": (
        ("counters", "allocations", "pinned_allocations"),
        "lower",
    ),
    "aggregate.counters.allocations.pinned_frees": (
        ("counters", "allocations", "pinned_frees"),
        "lower",
    ),
    "aggregate.counters.allocations.peak_device_bytes": (
        ("counters", "allocations", "peak_device_bytes"),
        "lower",
    ),
}
PRIMARY_METRICS = set(REQUEST_PRIMARY_METRICS) | set(AGGREGATE_PRIMARY_METRICS)


class InputError(ValueError):
    """Input is not a valid native profile run set."""


class ComparabilityError(InputError):
    """Valid runs do not describe one exact paired experiment."""


def _fail(path: str, message: str) -> NoReturn:
    raise InputError(f"{path}: {message}")


def _incomparable(path: str, message: str) -> NoReturn:
    raise ComparabilityError(f"{path}: {message}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> NoReturn:
    raise InputError(f"non-finite JSON number {value!r} is forbidden")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_EVIDENCE_BYTES:
            raise InputError(
                f"{path}: exceeds the {MAX_EVIDENCE_BYTES}-byte evidence bound"
            )
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
    except FileNotFoundError:
        raise InputError(f"{path}: file does not exist") from None
    except (OSError, UnicodeDecodeError) as error:
        raise InputError(f"{path}: cannot read UTF-8 JSON: {error}") from error
    except json.JSONDecodeError as error:
        raise InputError(
            f"{path}:{error.lineno}:{error.colno}: invalid JSON: {error.msg}"
        ) from None
    if not isinstance(value, dict):
        raise InputError(f"{path}: root must be a JSON object")
    return value


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _exact_keys(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    result = _object(value, path)
    actual = set(result)
    missing = sorted(keys - actual)
    extra = sorted(actual - keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        _fail(path, "; ".join(details))
    return result


def _string(value: Any, path: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, "has an invalid format")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(path, "must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        _fail(path, f"must be {bounds}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(path, "must be a JSON number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(path, "must be representable as a finite IEEE-754 number")
    if not math.isfinite(result):
        _fail(path, "must be finite")
    if strictly_positive and result <= 0.0:
        _fail(path, "must be > 0")
    if not strictly_positive and result < minimum:
        _fail(path, f"must be >= {minimum:g}")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _timestamp(value: Any, path: str) -> str:
    result = _string(value, path)
    if not result.endswith("Z"):
        _fail(path, "must be an RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError:
        _fail(path, "must be a valid RFC 3339 timestamp")
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        _fail(path, "must use UTC")
    return result


def _validate_runtime_flag(value: Any, path: str) -> None:
    flag = _exact_keys(value, {"name", "value"}, path)
    _string(flag["name"], f"{path}.name", pattern=ID_RE)
    _string(flag["value"], f"{path}.value")


def _validate_source(value: Any, path: str) -> None:
    source = _exact_keys(
        value,
        {
            "git_commit",
            "git_dirty",
            "executable_sha256",
            "implementation_id",
            "runtime_flag",
            "semantic_class",
            "correctness_gate_id",
            "correctness_report_sha256",
        },
        path,
    )
    _string(source["git_commit"], f"{path}.git_commit", pattern=GIT_COMMIT_RE)
    if _boolean(source["git_dirty"], f"{path}.git_dirty"):
        _fail(f"{path}.git_dirty", "must be false for source-bound evidence")
    _string(
        source["executable_sha256"],
        f"{path}.executable_sha256",
        pattern=SHA256_RE,
    )
    _string(source["implementation_id"], f"{path}.implementation_id", pattern=ID_RE)
    _validate_runtime_flag(source["runtime_flag"], f"{path}.runtime_flag")
    if source["semantic_class"] != "E0":
        _fail(f"{path}.semantic_class", "must equal 'E0'")
    _string(source["correctness_gate_id"], f"{path}.correctness_gate_id", pattern=ID_RE)
    _string(
        source["correctness_report_sha256"],
        f"{path}.correctness_report_sha256",
        pattern=SHA256_RE,
    )


def _validate_environment(value: Any, path: str) -> None:
    environment = _exact_keys(value, {"gpu", "host", "software"}, path)
    gpu = _exact_keys(
        environment["gpu"],
        {"model", "uuid", "device_index", "pci_bus_id", "compute_capability", "vram_bytes"},
        f"{path}.gpu",
    )
    for key in ("model", "uuid", "pci_bus_id"):
        _string(gpu[key], f"{path}.gpu.{key}")
    _integer(gpu["device_index"], f"{path}.gpu.device_index")
    capability = _string(gpu["compute_capability"], f"{path}.gpu.compute_capability")
    if re.fullmatch(r"[0-9]+\.[0-9]+", capability) is None:
        _fail(f"{path}.gpu.compute_capability", "must have major.minor form")
    _integer(gpu["vram_bytes"], f"{path}.gpu.vram_bytes", minimum=1)

    host = _exact_keys(
        environment["host"],
        {
            "environment_id",
            "cpu_model",
            "physical_core_count",
            "logical_core_count",
            "ram_bytes",
            "os_release",
            "kernel_release",
            "architecture",
        },
        f"{path}.host",
    )
    _string(host["environment_id"], f"{path}.host.environment_id", pattern=ID_RE)
    for key in ("cpu_model", "os_release", "kernel_release", "architecture"):
        _string(host[key], f"{path}.host.{key}")
    physical = _integer(
        host["physical_core_count"], f"{path}.host.physical_core_count", minimum=1
    )
    logical = _integer(
        host["logical_core_count"], f"{path}.host.logical_core_count", minimum=1
    )
    if logical < physical:
        _fail(f"{path}.host.logical_core_count", "must be >= physical_core_count")
    _integer(host["ram_bytes"], f"{path}.host.ram_bytes", minimum=1)

    software = _exact_keys(
        environment["software"],
        {
            "nvidia_driver_version",
            "cuda_runtime_version",
            "cuda_toolkit_version",
            "cublas_version",
            "container_image_sha256",
        },
        f"{path}.software",
    )
    for key in (
        "nvidia_driver_version",
        "cuda_runtime_version",
        "cuda_toolkit_version",
        "cublas_version",
    ):
        _string(software[key], f"{path}.software.{key}")
    _string(
        software["container_image_sha256"],
        f"{path}.software.container_image_sha256",
        pattern=SHA256_RE,
    )


def _validate_workload(value: Any, path: str) -> None:
    workload = _exact_keys(
        value,
        {
            "workload_id",
            "model_id",
            "model_revision",
            "weights_sha256",
            "tokenizer_sha256",
            "dtype",
            "concurrency",
            "prompt_tokens",
            "output_tokens",
            "warmups",
            "measured_iterations",
            "sampling_id",
            "seed",
        },
        path,
    )
    _string(workload["workload_id"], f"{path}.workload_id", pattern=ID_RE)
    for key in ("model_id", "model_revision", "dtype"):
        _string(workload[key], f"{path}.{key}")
    for key in ("weights_sha256", "tokenizer_sha256"):
        _string(workload[key], f"{path}.{key}", pattern=SHA256_RE)
    _integer(
        workload["concurrency"],
        f"{path}.concurrency",
        minimum=1,
        maximum=MAX_CONCURRENCY,
    )
    _integer(
        workload["prompt_tokens"],
        f"{path}.prompt_tokens",
        minimum=1,
        maximum=MAX_PROMPT_TOKENS,
    )
    _integer(
        workload["output_tokens"],
        f"{path}.output_tokens",
        minimum=MIN_OUTPUT_TOKENS,
        maximum=MAX_OUTPUT_TOKENS,
    )
    _integer(
        workload["warmups"],
        f"{path}.warmups",
        minimum=5,
        maximum=MAX_WARMUPS,
    )
    _integer(
        workload["measured_iterations"],
        f"{path}.measured_iterations",
        minimum=30,
        maximum=MAX_MEASURED_ITERATIONS,
    )
    if workload["sampling_id"] != "greedy":
        _fail(f"{path}.sampling_id", "must be canonical greedy sampling")
    if workload["seed"] is not None:
        _fail(f"{path}.seed", "must be null for canonical greedy sampling")


def _validate_trace(value: Any, path: str) -> None:
    trace = _exact_keys(
        value,
        {"capacity", "retained_records", "dropped_records"},
        path,
    )
    capacity = _integer(
        trace["capacity"], f"{path}.capacity", minimum=1, maximum=U64_MAX
    )
    retained = _integer(
        trace["retained_records"], f"{path}.retained_records", maximum=U64_MAX
    )
    _integer(
        trace["dropped_records"], f"{path}.dropped_records", maximum=U64_MAX
    )
    if retained > capacity:
        _fail(f"{path}.retained_records", "must be <= trace capacity")


def _validate_measurement(value: Any, path: str, *, count: bool) -> None:
    measurement = _exact_keys(value, {"validity", "value"}, path)
    validity = measurement["validity"]
    if validity == "measured":
        if count:
            _integer(measurement["value"], f"{path}.value", maximum=U64_MAX)
        else:
            _number(measurement["value"], f"{path}.value")
    elif validity == "unmeasured":
        if measurement["value"] is not None:
            _fail(f"{path}.value", "must be null when validity is unmeasured")
    else:
        _fail(f"{path}.validity", "must be 'measured' or 'unmeasured'")


def _validate_aggregate(value: Any, path: str) -> None:
    aggregate = _exact_keys(
        value,
        {"host", "cuda", "counters", "throughput_output_tokens_per_second"},
        path,
    )
    host = _exact_keys(
        aggregate["host"],
        {"plan_ns", "execute_ns", "sampling_ns", "commit_ns"},
        f"{path}.host",
    )
    for key in host:
        _validate_measurement(host[key], f"{path}.host.{key}", count=False)
    cuda = _exact_keys(
        aggregate["cuda"], {"stream_span_ns", "idle_ns"}, f"{path}.cuda"
    )
    for key in cuda:
        _validate_measurement(cuda[key], f"{path}.cuda.{key}", count=False)
    counters = _exact_keys(
        aggregate["counters"],
        {"iterations", "kernel_launches", "copies", "allocations"},
        f"{path}.counters",
    )
    for key in ("iterations", "kernel_launches"):
        _validate_measurement(
            counters[key], f"{path}.counters.{key}", count=True
        )
    copies = _exact_keys(
        counters["copies"],
        {"h2d_calls", "h2d_bytes", "d2h_calls", "d2h_bytes"},
        f"{path}.counters.copies",
    )
    for key in copies:
        _validate_measurement(
            copies[key], f"{path}.counters.copies.{key}", count=True
        )
    allocations = _exact_keys(
        counters["allocations"],
        {
            "device_allocations",
            "device_frees",
            "pinned_allocations",
            "pinned_frees",
            "peak_device_bytes",
        },
        f"{path}.counters.allocations",
    )
    for key in allocations:
        _validate_measurement(
            allocations[key], f"{path}.counters.allocations.{key}", count=True
        )
    _number(
        aggregate["throughput_output_tokens_per_second"],
        f"{path}.throughput_output_tokens_per_second",
        strictly_positive=True,
    )


def _validate_request(
    value: Any,
    path: str,
    *,
    prompt_tokens: int,
    output_tokens: int,
) -> None:
    request = _exact_keys(
        value,
        {
            "input_index",
            "prompt_u32le_sha256",
            "generated_u32le_sha256",
            "prompt_token_count",
            "requested_output_token_count",
            "generated_token_count",
            "ttft_ms",
            "tpot_ms",
            "e2e_ms",
        },
        path,
    )
    _integer(request["input_index"], f"{path}.input_index")
    for key in ("prompt_u32le_sha256", "generated_u32le_sha256"):
        _string(request[key], f"{path}.{key}", pattern=SHA256_RE)
    if _integer(request["prompt_token_count"], f"{path}.prompt_token_count", minimum=1) != prompt_tokens:
        _fail(f"{path}.prompt_token_count", "must equal workload.prompt_tokens")
    requested = _integer(
        request["requested_output_token_count"],
        f"{path}.requested_output_token_count",
        minimum=1,
    )
    if requested != output_tokens:
        _fail(
            f"{path}.requested_output_token_count",
            "must equal workload.output_tokens",
        )
    generated = _integer(
        request["generated_token_count"], f"{path}.generated_token_count", minimum=1
    )
    if generated != requested:
        _fail(f"{path}.generated_token_count", "must equal requested count")
    ttft = _number(request["ttft_ms"], f"{path}.ttft_ms", strictly_positive=True)
    tpot = _number(request["tpot_ms"], f"{path}.tpot_ms", strictly_positive=True)
    e2e = _number(request["e2e_ms"], f"{path}.e2e_ms", strictly_positive=True)
    minimum_e2e = ttft + tpot * (generated - 1)
    if not math.isfinite(minimum_e2e):
        _fail(f"{path}.e2e_ms", "latency composition must remain finite")
    tolerance = max(1.0e-9, abs(minimum_e2e) * 1.0e-12)
    if e2e + tolerance < minimum_e2e:
        _fail(
            f"{path}.e2e_ms",
            "must cover TTFT plus every reported inter-token interval",
        )


def _validate_run(run: dict[str, Any], source_path: str) -> None:
    row = _exact_keys(
        run,
        {
            "schema_version",
            "role",
            "pair_index",
            "run_id",
            "recorded_at_utc",
            "status",
            "failure_count",
            "source",
            "environment",
            "workload",
            "trace",
            "primary_metric",
            "aggregate",
            "requests",
        },
        source_path,
    )
    if row["schema_version"] != RUN_SCHEMA_VERSION:
        _fail(
            f"{source_path}.schema_version",
            f"must equal {RUN_SCHEMA_VERSION!r}",
        )
    if row["role"] not in {"baseline", "candidate"}:
        _fail(f"{source_path}.role", "must be 'baseline' or 'candidate'")
    _integer(row["pair_index"], f"{source_path}.pair_index", minimum=1, maximum=5)
    _string(row["run_id"], f"{source_path}.run_id", pattern=ID_RE)
    _timestamp(row["recorded_at_utc"], f"{source_path}.recorded_at_utc")
    if row["status"] not in {"success", "failure"}:
        _fail(f"{source_path}.status", "must be 'success' or 'failure'")
    failure_count = _integer(
        row["failure_count"],
        f"{source_path}.failure_count",
        maximum=U64_MAX,
    )
    if row["status"] == "success" and failure_count != 0:
        _fail(f"{source_path}.failure_count", "must be zero for success status")
    if row["status"] == "failure" and failure_count == 0:
        _fail(f"{source_path}.failure_count", "must be nonzero for failure status")
    _validate_source(row["source"], f"{source_path}.source")
    _validate_environment(row["environment"], f"{source_path}.environment")
    _validate_workload(row["workload"], f"{source_path}.workload")
    _validate_trace(row["trace"], f"{source_path}.trace")
    if row["primary_metric"] != CANONICAL_PRIMARY_METRIC:
        _fail(
            f"{source_path}.primary_metric",
            f"must be the preregistered metric {CANONICAL_PRIMARY_METRIC!r}",
        )
    _validate_aggregate(row["aggregate"], f"{source_path}.aggregate")
    requests = row["requests"]
    if not isinstance(requests, list):
        _fail(f"{source_path}.requests", "must be an array")
    workload = row["workload"]
    expected_count = workload["concurrency"] * workload["measured_iterations"]
    if len(requests) != expected_count:
        _fail(
            f"{source_path}.requests",
            f"must contain concurrency * measured_iterations = {expected_count} observations",
        )
    for index, request in enumerate(requests):
        _validate_request(
            request,
            f"{source_path}.requests[{index}]",
            prompt_tokens=workload["prompt_tokens"],
            output_tokens=workload["output_tokens"],
        )
    indices = sorted(request["input_index"] for request in requests)
    if indices != list(range(expected_count)):
        _fail(
            f"{source_path}.requests",
            "input_index values must be unique and exactly 0..request_count-1",
        )
    iteration_measurement = row["aggregate"]["counters"]["iterations"]
    if iteration_measurement["validity"] != "measured":
        _fail(
            f"{source_path}.aggregate.counters.iterations",
            "must be measured to prove complete trace retention",
        )
    iteration_count = _integer(
        iteration_measurement["value"],
        f"{source_path}.aggregate.counters.iterations.value",
        minimum=1,
    )
    expected_retained = expected_count + iteration_count
    if row["trace"]["retained_records"] != expected_retained:
        _fail(
            f"{source_path}.trace.retained_records",
            f"must equal request_count + iteration_count = {expected_retained}",
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _request_identity(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "input_index",
        "prompt_u32le_sha256",
        "generated_u32le_sha256",
        "prompt_token_count",
        "requested_output_token_count",
        "generated_token_count",
    )
    requests = sorted(run["requests"], key=lambda request: request["input_index"])
    return [{key: request[key] for key in fields} for request in requests]


def _source_common(run: Mapping[str, Any]) -> dict[str, Any]:
    source = run["source"]
    fields = (
        "git_commit",
        "executable_sha256",
        "semantic_class",
        "correctness_gate_id",
        "correctness_report_sha256",
    )
    return {key: source[key] for key in fields}


def _runtime_binding(run: Mapping[str, Any]) -> dict[str, Any]:
    source = run["source"]
    return {
        "implementation_id": source["implementation_id"],
        "runtime_flag": source["runtime_flag"],
    }


def _load_role(paths: Sequence[Path], role: str) -> list[dict[str, Any]]:
    if len(paths) != REQUIRED_RUNS:
        raise InputError(
            f"{role}: expected exactly {REQUIRED_RUNS} independent run files, got {len(paths)}"
        )
    runs: list[dict[str, Any]] = []
    for path in paths:
        run = _read_json(path)
        _validate_run(run, str(path))
        if run["role"] != role:
            raise InputError(f"{path}.role: expected {role!r}, got {run['role']!r}")
        runs.append(run)
    indices = [run["pair_index"] for run in runs]
    if sorted(indices) != list(range(1, REQUIRED_RUNS + 1)):
        raise InputError(
            f"{role}: pair_index values must be exactly 1..{REQUIRED_RUNS}"
        )
    return sorted(runs, key=lambda run: run["pair_index"])


def _require_equal(values: Sequence[Any], path: str) -> Any:
    reference = values[0]
    for index, value in enumerate(values[1:], 1):
        if value != reference:
            _incomparable(path, f"run position {index + 1} differs")
    return reference


def _bind_pair(
    baseline: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    all_runs = list(baseline) + list(candidate)
    run_ids = [run["run_id"] for run in all_runs]
    if len(set(run_ids)) != len(run_ids):
        _incomparable("run_id", "all ten independent process run IDs must be unique")

    common_source = _require_equal(
        [_source_common(run) for run in all_runs], "source"
    )
    environment = _require_equal(
        [run["environment"] for run in all_runs], "environment"
    )
    workload = _require_equal([run["workload"] for run in all_runs], "workload")
    primary_metric = _require_equal(
        [run["primary_metric"] for run in all_runs], "primary_metric"
    )
    request_identity = _require_equal(
        [_request_identity(run) for run in all_runs],
        "requests token counts and u32-LE hashes",
    )
    baseline_runtime = _require_equal(
        [_runtime_binding(run) for run in baseline], "baseline runtime binding"
    )
    candidate_runtime = _require_equal(
        [_runtime_binding(run) for run in candidate], "candidate runtime binding"
    )
    baseline_flag = baseline_runtime["runtime_flag"]
    candidate_flag = candidate_runtime["runtime_flag"]
    flag_pairs = {
        "residual_rmsnorm": ("separate", "fused"),
        "execution_completion": ("per-operation", "iteration-batch"),
    }
    baseline_name = baseline_flag["name"]
    candidate_name = candidate_flag["name"]
    if baseline_name != candidate_name or baseline_name not in flag_pairs:
        _incomparable(
            "source.runtime_flag",
            "both arms must bind one supported runtime flag",
        )
    expected_baseline, expected_candidate = flag_pairs[baseline_name]
    if baseline_flag["value"] != expected_baseline:
        _incomparable(
            "source.runtime_flag",
            f"baseline must bind {baseline_name}={expected_baseline}",
        )
    if candidate_flag["value"] != expected_candidate:
        _incomparable(
            "source.runtime_flag",
            f"candidate must bind {baseline_name}={expected_candidate}",
        )
    expected_correctness_gate = CORRECTNESS_GATES[baseline_name]
    if common_source["correctness_gate_id"] != expected_correctness_gate:
        _incomparable(
            "source.correctness_gate_id",
            f"{baseline_name} requires {expected_correctness_gate}",
        )
    if baseline_runtime["implementation_id"] == candidate_runtime["implementation_id"]:
        _incomparable(
            "source.implementation_id",
            "baseline and candidate implementation IDs must differ",
        )

    return {
        **common_source,
        "environment_sha256": _sha256_json(environment),
        "workload_sha256": _sha256_json(workload),
        "request_identity_sha256": _sha256_json(request_identity),
        "baseline_runtime": baseline_runtime,
        "candidate_runtime": candidate_runtime,
        "primary_metric": primary_metric,
    }


def r7(values: Sequence[float], probability: float) -> float:
    """Return the Hyndman-Fan type-7 quantile used by the evidence report."""

    observations = sorted(float(value) for value in values)
    if not observations:
        raise InputError("R7 quantile requires at least one observation")
    if any(not math.isfinite(value) for value in observations):
        raise InputError("R7 observations must be finite")
    if not 0.0 <= probability <= 1.0:
        raise InputError("R7 probability must be in [0, 1]")
    if len(observations) == 1:
        return observations[0]
    h = (len(observations) - 1) * probability
    lower = math.floor(h)
    fraction = h - lower
    if fraction == 0.0:
        return observations[lower]
    return observations[lower] + fraction * (
        observations[lower + 1] - observations[lower]
    )


def _ratio(candidate: float, baseline: float, path: str) -> float:
    if baseline <= 0.0:
        raise InputError(f"{path}: baseline must be > 0 for a ratio")
    result = candidate / baseline
    if not math.isfinite(result):
        raise InputError(f"{path}: ratio must be finite")
    return result


def _run_request_values(run: Mapping[str, Any], field: str) -> list[float]:
    return [float(request[field]) for request in run["requests"]]


def _run_metric_pair(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], field: str
) -> dict[str, float]:
    baseline_values = _run_request_values(baseline, field)
    candidate_values = _run_request_values(candidate, field)
    baseline_median = r7(baseline_values, 0.50)
    candidate_median = r7(candidate_values, 0.50)
    baseline_p95 = r7(baseline_values, 0.95)
    candidate_p95 = r7(candidate_values, 0.95)
    return {
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "baseline_p95": baseline_p95,
        "candidate_p95": candidate_p95,
        "median_ratio": _ratio(candidate_median, baseline_median, field),
        "p95_ratio": _ratio(candidate_p95, baseline_p95, field),
    }


def _metric_comparison(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, float]:
    baseline_values = [
        value for run in baseline for value in _run_request_values(run, field)
    ]
    candidate_values = [
        value for run in candidate for value in _run_request_values(run, field)
    ]
    baseline_median = r7(baseline_values, 0.50)
    candidate_median = r7(candidate_values, 0.50)
    baseline_p95 = r7(baseline_values, 0.95)
    candidate_p95 = r7(candidate_values, 0.95)
    pairs = [
        _run_metric_pair(left, right, field)
        for left, right in zip(baseline, candidate, strict=True)
    ]
    return {
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "baseline_p95": baseline_p95,
        "candidate_p95": candidate_p95,
        "median_ratio": _ratio(candidate_median, baseline_median, field),
        "p95_ratio": _ratio(candidate_p95, baseline_p95, field),
        "paired_median_change_fraction": r7(
            [pair["median_ratio"] - 1.0 for pair in pairs], 0.50
        ),
        "paired_p95_change_fraction": r7(
            [pair["p95_ratio"] - 1.0 for pair in pairs], 0.50
        ),
    }


def _throughput(run: Mapping[str, Any]) -> float:
    return float(run["aggregate"]["throughput_output_tokens_per_second"])


def _throughput_comparison(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    baseline_values = [_throughput(run) for run in baseline]
    candidate_values = [_throughput(run) for run in candidate]
    pair_ratios = [
        _ratio(right, left, "throughput")
        for left, right in zip(baseline_values, candidate_values, strict=True)
    ]
    baseline_median = r7(baseline_values, 0.50)
    candidate_median = r7(candidate_values, 0.50)
    baseline_p95 = r7(baseline_values, 0.95)
    candidate_p95 = r7(candidate_values, 0.95)
    return {
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "baseline_p95": baseline_p95,
        "candidate_p95": candidate_p95,
        "median_ratio": _ratio(candidate_median, baseline_median, "throughput median"),
        "p95_ratio": _ratio(candidate_p95, baseline_p95, "throughput p95"),
        "paired_median_change_fraction": r7(
            [ratio - 1.0 for ratio in pair_ratios], 0.50
        ),
        "paired_p95_change_fraction": r7(
            [ratio - 1.0 for ratio in pair_ratios], 0.95
        ),
    }


def _dig(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def _primary_value(run: Mapping[str, Any], name: str) -> tuple[float, str]:
    request_metric = REQUEST_PRIMARY_METRICS.get(name)
    if request_metric is not None:
        field, probability, direction = request_metric
        return r7(_run_request_values(run, field), probability), direction
    aggregate_path, direction = AGGREGATE_PRIMARY_METRICS[name]
    value = _dig(run["aggregate"], aggregate_path)
    if aggregate_path == ("throughput_output_tokens_per_second",):
        return _number(value, name, strictly_positive=True), direction
    measurement = _object(value, name)
    if measurement["validity"] != "measured":
        raise InputError(f"{name}: declared primary metric is unmeasured")
    return _number(measurement["value"], f"{name}.value"), direction


def _improvement(
    baseline: float, candidate: float, direction: str, path: str
) -> float:
    ratio = _ratio(candidate, baseline, path)
    return 1.0 - ratio if direction == "lower" else ratio - 1.0


def _primary_comparison(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    name: str,
) -> tuple[dict[str, Any], list[tuple[float, float, float]]]:
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    improvements: list[float] = []
    direction: str | None = None
    for left, right in zip(baseline, candidate, strict=True):
        left_value, left_direction = _primary_value(left, name)
        right_value, right_direction = _primary_value(right, name)
        if left_direction != right_direction:
            raise InputError(f"{name}: inconsistent metric direction")
        direction = left_direction
        baseline_values.append(left_value)
        candidate_values.append(right_value)
        improvements.append(
            _improvement(left_value, right_value, direction, name)
        )
    assert direction is not None
    baseline_median = r7(baseline_values, 0.50)
    candidate_median = r7(candidate_values, 0.50)
    comparison = {
        "name": name,
        "direction": direction,
        "baseline_median": baseline_median,
        "candidate_median": candidate_median,
        "improvement_fraction": _improvement(
            baseline_median, candidate_median, direction, name
        ),
        "paired_improvement_fraction_median": r7(improvements, 0.50),
    }
    pairs = list(zip(baseline_values, candidate_values, improvements, strict=True))
    return comparison, pairs


def _check(name: str, observed: float, operator: str, limit: float) -> dict[str, Any]:
    if operator == "<=":
        passed = observed <= limit
    elif operator == ">=":
        passed = observed >= limit
    else:  # pragma: no cover - internal contract
        raise AssertionError(operator)
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "operator": operator,
        "limit": limit,
    }


def _thresholds() -> dict[str, float | int]:
    return {
        "required_independent_runs": REQUIRED_RUNS,
        "ttft_p95_candidate_to_baseline_max": TTFT_P95_RATIO_MAX,
        "tpot_p95_candidate_to_baseline_max": TPOT_P95_RATIO_MAX,
        "throughput_candidate_to_baseline_min": THROUGHPUT_RATIO_MIN,
        "primary_improvement_fraction_min": PRIMARY_IMPROVEMENT_MIN,
    }


def _empty_report(baseline_count: int, candidate_count: int) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "error",
        "passed": False,
        "primary_metric": None,
        "run_counts": {
            "required": REQUIRED_RUNS,
            "baseline": baseline_count,
            "candidate": candidate_count,
        },
        "thresholds": _thresholds(),
        "bindings": None,
        "metrics": None,
        "run_pairs": [],
        "checks": [],
        "errors": [],
    }


def evaluate(
    baseline_paths: Sequence[Path | str], candidate_paths: Sequence[Path | str]
) -> dict[str, Any]:
    """Return a strict paired-profile report for five run files per role."""

    baseline_files = [Path(path) for path in baseline_paths]
    candidate_files = [Path(path) for path in candidate_paths]
    report = _empty_report(len(baseline_files), len(candidate_files))
    try:
        baseline = _load_role(baseline_files, "baseline")
        candidate = _load_role(candidate_files, "candidate")
        binding = _bind_pair(baseline, candidate)
        primary_metric = binding.pop("primary_metric")

        ttft = _metric_comparison(baseline, candidate, "ttft_ms")
        tpot = _metric_comparison(baseline, candidate, "tpot_ms")
        e2e = _metric_comparison(baseline, candidate, "e2e_ms")
        throughput = _throughput_comparison(baseline, candidate)
        primary, primary_pairs = _primary_comparison(
            baseline, candidate, primary_metric
        )

        run_pairs = []
        for pair_offset, (left, right) in enumerate(
            zip(baseline, candidate, strict=True)
        ):
            left_primary, right_primary, improvement = primary_pairs[pair_offset]
            left_throughput = _throughput(left)
            right_throughput = _throughput(right)
            run_pairs.append(
                {
                    "pair_index": left["pair_index"],
                    "baseline_run_id": left["run_id"],
                    "candidate_run_id": right["run_id"],
                    "ttft_ms": _run_metric_pair(left, right, "ttft_ms"),
                    "tpot_ms": _run_metric_pair(left, right, "tpot_ms"),
                    "e2e_ms": _run_metric_pair(left, right, "e2e_ms"),
                    "throughput_output_tokens_per_second": {
                        "baseline": left_throughput,
                        "candidate": right_throughput,
                        "ratio": _ratio(
                            right_throughput, left_throughput, "throughput"
                        ),
                    },
                    "primary": {
                        "baseline": left_primary,
                        "candidate": right_primary,
                        "improvement_fraction": improvement,
                    },
                }
            )

        total_failures = sum(run["failure_count"] for run in baseline + candidate)
        total_dropped = sum(
            run["trace"]["dropped_records"] for run in baseline + candidate
        )
        effective_primary_improvement = min(
            primary["improvement_fraction"],
            primary["paired_improvement_fraction_median"],
        )
        checks = [
            _check("zero_failures", total_failures, "<=", 0.0),
            _check(
                "zero_dropped_trace_records", total_dropped, "<=", 0.0
            ),
            _check(
                "ttft_p95_regression",
                ttft["p95_ratio"],
                "<=",
                TTFT_P95_RATIO_MAX,
            ),
            _check(
                "tpot_p95_regression",
                tpot["p95_ratio"],
                "<=",
                TPOT_P95_RATIO_MAX,
            ),
            _check(
                "throughput_regression",
                throughput["median_ratio"],
                ">=",
                THROUGHPUT_RATIO_MIN,
            ),
            _check(
                "primary_effective_improvement",
                effective_primary_improvement,
                ">=",
                PRIMARY_IMPROVEMENT_MIN,
            ),
        ]
        passed = all(check["passed"] for check in checks)
        report.update(
            {
                "status": "passed" if passed else "failed",
                "passed": passed,
                "primary_metric": primary_metric,
                "bindings": binding,
                "metrics": {
                    "ttft_ms": ttft,
                    "tpot_ms": tpot,
                    "e2e_ms": e2e,
                    "throughput_output_tokens_per_second": throughput,
                    "primary": primary,
                },
                "run_pairs": run_pairs,
                "checks": checks,
            }
        )
    except ComparabilityError as error:
        report["status"] = "incomparable"
        report["errors"] = [str(error)]
    except InputError as error:
        report["status"] = "error"
        report["errors"] = [str(error)]
    return report


def _flatten(values: Sequence[Sequence[str]]) -> list[str]:
    return [item for group in values for item in group]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        nargs="+",
        required=True,
        metavar="RUN.json",
        help="five independent baseline run documents (option may be repeated)",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs="+",
        required=True,
        metavar="RUN.json",
        help="five independent candidate run documents (option may be repeated)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="also create this report path; an existing path is never overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(_flatten(args.baseline), _flatten(args.candidate))
    encoded = json.dumps(
        report,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if args.report is not None:
        try:
            with args.report.open("x", encoding="utf-8") as handle:
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
