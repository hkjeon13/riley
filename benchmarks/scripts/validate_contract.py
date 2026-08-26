#!/usr/bin/env python3
"""Validate the PR 01 benchmark contract with the Python standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NoReturn


CONTRACT_VERSION = "1.0.0"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
WEIGHTS_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
CONFIG_SHA256 = "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
TOKENIZER_SHA256 = "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db"
TOKENIZER_FILES_SHA256 = {
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}
MAX_CONTEXT_TOKENS = 8192
GPU_MODEL = "NVIDIA GeForce RTX 4090"
COMPUTE_CAPABILITY = "8.9"
DTYPE = "bf16"
PRIMARY_ENVIRONMENT_ID = "rtx4090-ubuntu22-driver580-v1"
PRIMARY_DRIVER_VERSION = "580.173.02"
PRIMARY_RAM_BYTES = 67_185_598_464
PYTHON_VERSION_FILE_SHA256 = "861b3dd8083d28f336ef70f6755bc399538ddad627b1d095820ca34cb953cf14"
EMPTY_TOKEN_IDS_SHA256 = hashlib.sha256(b"").hexdigest()

PROMPT_CATEGORIES = {
    "short",
    "multilingual",
    "symbols-code",
    "long-repetition",
    "context-boundary",
    "minimal",
    "early-eos",
}
LANE_IDS = {"hf-transformers", "vllm", "rustinfer-native"}
SINGLE_CELL_BENCHMARK_FLAGS = {
    "--matrix",
    "--prompts",
    "--result-dir",
    "--run-index",
    "--run-id",
    "--warm-state",
    "--concurrency",
    "--prompt-tokens",
    "--output-tokens",
}
REFERENCE_FIXTURE_SOURCE_PATHS = {
    "matrix": "benchmarks/matrix.yaml",
    "prompts": "benchmarks/prompts.jsonl",
    "environment": "benchmarks/environment.md",
    "lane_manifest": "benchmarks/lanes/hf-transformers.json",
    "correctness_gate": "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json",
    "prompt_schema": "benchmarks/schemas/prompt.schema.json",
    "fixture_schema": "benchmarks/schemas/reference-fixture.schema.json",
    "contract_validator": "benchmarks/scripts/validate_contract.py",
    "dependency_manifest": "tools/python/reference/pyproject.toml",
    "dependency_lock": "tools/python/reference/uv.lock",
    "python_version_file": "tools/python/reference/.python-version",
    "constants": "tools/python/reference/rustinfer_reference/constants.py",
    "environment_probe": "tools/python/reference/rustinfer_reference/environment.py",
    "fixture_generator": "tools/python/reference/rustinfer_reference/fixture.py",
    "hf_backend": "tools/python/reference/rustinfer_reference/hf_backend.py",
    "cli": "tools/python/reference/rustinfer_reference/cli.py",
}


class ContractError(ValueError):
    """Raised when a version-controlled benchmark contract is inconsistent."""


class ComparabilityContractError(ContractError):
    """A schema-valid result refers to a different benchmark contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _error(path: str, message: str) -> NoReturn:
    raise ContractError(f"{path}: {message}")


def _comparison_error(path: str, message: str) -> NoReturn:
    raise ComparabilityContractError(f"{path}: {message}")


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except FileNotFoundError:
        raise ContractError(f"{path}: file does not exist") from None
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_threshold_calibration_evidence(
    root: Path,
    correctness_gate: dict[str, Any],
    gate_path: Path,
) -> None:
    evidence = correctness_gate["numeric"]["threshold_activation"][
        "calibration_evidence"
    ]
    evidence_path = f"{gate_path}.numeric.threshold_activation.calibration_evidence"
    relative_report_path = Path(evidence["report_path"])
    if relative_report_path.is_absolute():
        _error(f"{evidence_path}.report_path", "must be repository-relative")
    report_path = (root / relative_report_path).resolve()
    if root not in report_path.parents or not report_path.is_file():
        _error(
            f"{evidence_path}.report_path",
            "missing or repository-external calibration report",
        )
    _expect(
        report_path.stat().st_size,
        evidence["report_size_bytes"],
        f"{evidence_path}.report_size_bytes",
    )
    _expect(
        _sha256(report_path),
        evidence["report_sha256"],
        f"{evidence_path}.report_sha256",
    )

    report = _read_json(report_path)
    _expect(report["report_kind"], "hf-oracle-calibration", str(report_path))
    _expect(report["gate_id"], evidence["report_gate_id"], str(report_path))
    _expect(report["status"], evidence["report_status"], str(report_path))
    _expect(report["e0_candidate_evidence"], False, str(report_path))
    _expect(
        report["bindings"]["git_revision"],
        evidence["git_revision"],
        f"{report_path}.bindings.git_revision",
    )
    summary = report["summary"]
    _expect(summary["case_count"], evidence["case_count"], str(report_path))
    _expect(summary["failure_count"], evidence["failure_count"], str(report_path))
    _expect(
        summary["semantic_self_check_pass"],
        evidence["semantic_self_check_pass"],
        str(report_path),
    )
    _expect(len(report["cases"]), evidence["case_count"], f"{report_path}.cases")
    _expect(
        sum(case["pass"] is False for case in report["cases"]),
        evidence["failure_count"],
        f"{report_path}.cases",
    )
    for tensor_name, expected_metrics in evidence[
        "observed_aggregate_metrics"
    ].items():
        _expect(
            summary["aggregate_numeric"][tensor_name]["metrics"],
            expected_metrics,
            f"{report_path}.summary.aggregate_numeric.{tensor_name}.metrics",
        )


def _exact_keys(value: Any, expected: set[str], path: str) -> None:
    if not isinstance(value, dict):
        _error(path, "must be an object")
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        _error(path, "; ".join(details))


def _expect(actual: Any, expected: Any, path: str) -> None:
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        _error(path, f"expected {expected!r}, got {actual!r}")


def _expect_comparable(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        _comparison_error(path, f"expected {expected!r}, got {actual!r}")


def _json_identity(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    raise ContractError(f"schema uses unsupported type {type_name!r}")


def _resolve_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ContractError(f"only local JSON Schema references are supported: {ref!r}")
    value: Any = root_schema
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"unresolved JSON Schema reference: {ref!r}")
        value = value[part]
    if not isinstance(value, dict):
        raise ContractError(f"JSON Schema reference is not an object: {ref!r}")
    return value


def _is_valid(instance: Any, schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    try:
        _validate_instance(instance, schema, root_schema, "$")
    except ContractError:
        return False
    return True


def _validate_instance(
    instance: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str,
) -> None:
    """Validate the JSON Schema subset used by the two repository schemas."""

    if "$ref" in schema:
        _validate_instance(instance, _resolve_ref(schema["$ref"], root_schema), root_schema, path)
        schema = {key: value for key, value in schema.items() if key != "$ref"}

    if "const" in schema and _json_identity(instance) != _json_identity(schema["const"]):
        _error(path, f"must equal {schema['const']!r}")
    if "enum" in schema:
        encoded = _json_identity(instance)
        if all(encoded != _json_identity(candidate) for candidate in schema["enum"]):
            _error(path, f"must be one of {schema['enum']!r}")

    declared_type = schema.get("type")
    if declared_type is not None:
        allowed_types = [declared_type] if isinstance(declared_type, str) else declared_type
        if not isinstance(allowed_types, list) or not all(
            isinstance(item, str) for item in allowed_types
        ):
            raise ContractError(f"schema at {path} has an invalid type declaration")
        if not any(_matches_type(instance, item) for item in allowed_types):
            _error(path, f"must have type {allowed_types!r}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise ContractError(f"schema at {path} has a non-array required keyword")
        missing = [key for key in required if key not in instance]
        if missing:
            _error(path, f"missing required properties {missing!r}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ContractError(f"schema at {path} has non-object properties")
        for key, subschema in properties.items():
            if key in instance:
                _validate_instance(instance[key], subschema, root_schema, f"{path}.{key}")
        additional = set(instance) - set(properties)
        if additional and schema.get("additionalProperties") is False:
            _error(path, f"unexpected properties {sorted(additional)!r}")
        additional_schema = schema.get("additionalProperties")
        if additional and isinstance(additional_schema, dict):
            for key in additional:
                _validate_instance(
                    instance[key], additional_schema, root_schema, f"{path}.{key}"
                )

    if isinstance(instance, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(instance) < minimum:
            _error(path, f"must contain at least {minimum} items")
        if maximum is not None and len(instance) > maximum:
            _error(path, f"must contain at most {maximum} items")
        if schema.get("uniqueItems"):
            identities = [_json_identity(item) for item in instance]
            if len(identities) != len(set(identities)):
                _error(path, "must contain unique items")

        prefix_items = schema.get("prefixItems", [])
        if not isinstance(prefix_items, list) or not all(
            isinstance(subschema, dict) for subschema in prefix_items
        ):
            raise ContractError(f"schema at {path} has invalid prefixItems")
        for index, subschema in enumerate(prefix_items[: len(instance)]):
            _validate_instance(
                instance[index], subschema, root_schema, f"{path}[{index}]"
            )

        items = schema.get("items")
        trailing_start = len(prefix_items)
        if items is False and len(instance) > trailing_start:
            _error(path, "must not contain items after the prefixItems inventory")
        if isinstance(items, dict):
            for index, item in enumerate(
                instance[trailing_start:], start=trailing_start
            ):
                _validate_instance(item, items, root_schema, f"{path}[{index}]")
        elif items is not None and not isinstance(items, bool):
            raise ContractError(f"schema at {path} has invalid items")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            _error(path, f"must contain at least {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            _error(path, f"must contain at most {schema['maxLength']} characters")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            _error(path, f"does not match {schema['pattern']!r}")
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(instance.replace("Z", "+00:00"))
            except ValueError:
                _error(path, "must be an RFC 3339 date-time")
            if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
                _error(path, "must include a UTC offset")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if not math.isfinite(instance):
            _error(path, "must be finite")
        if "minimum" in schema and instance < schema["minimum"]:
            _error(path, f"must be >= {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            _error(path, f"must be <= {schema['maximum']}")

    for subschema in schema.get("allOf", []):
        _validate_instance(instance, subschema, root_schema, path)
    if "anyOf" in schema and not any(
        _is_valid(instance, subschema, root_schema) for subschema in schema["anyOf"]
    ):
        _error(path, "does not satisfy any anyOf branch")
    if "oneOf" in schema:
        matches = sum(
            _is_valid(instance, subschema, root_schema) for subschema in schema["oneOf"]
        )
        if matches != 1:
            _error(path, f"must satisfy exactly one oneOf branch, matched {matches}")
    if "if" in schema:
        branch = "then" if _is_valid(instance, schema["if"], root_schema) else "else"
        if branch in schema:
            _validate_instance(instance, schema[branch], root_schema, path)


def validate_instance(instance: Any, schema: dict[str, Any]) -> None:
    """Public wrapper used by tests and result-producing tools."""

    _validate_instance(instance, schema, schema, "$")


def _walk_schema_strictness(value: Any, path: str) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            if value.get("additionalProperties") is not False:
                _error(path, "object schemas must set additionalProperties to false")
            properties = value.get("properties")
            required = value.get("required")
            if not isinstance(properties, dict) or not isinstance(required, list):
                _error(path, "strict object schemas need properties and required")
            if set(properties) != set(required) or len(required) != len(set(required)):
                _error(path, "every declared property must be required exactly once")
        for key, child in value.items():
            _walk_schema_strictness(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_schema_strictness(child, f"{path}[{index}]")


def validate_schema_document(schema: Any, kind: str, path: str) -> dict[str, Any]:
    if not isinstance(schema, dict):
        _error(path, "schema must be an object")
    _expect(schema.get("$schema"), SCHEMA_DIALECT, f"{path}.$schema")
    _expect(schema.get("type"), "object", f"{path}.type")
    _expect(schema.get("additionalProperties"), False, f"{path}.additionalProperties")
    _walk_schema_strictness(schema, path)

    properties = schema.get("properties", {})
    if kind == "result":
        expected = {
            "contract_version",
            "trial_id",
            "run_id",
            "trial_index",
            "scope",
            "matrix_id",
            "matrix_sha256",
            "prompts_sha256",
            "lane_manifest_sha256",
            "environment_id",
            "semantic_class",
            "implementation_id",
            "reference_implementation",
            "runtime_dependency_class",
            "approximation_enabled",
            "error_budget",
            "seed",
            "warm_state",
            "model_revision",
            "engine_revision",
            "failure_count",
            "speculative",
            "sparse_attention",
            "quantization",
            "metrics",
            "requests",
        }
        missing = expected - set(properties)
        if missing:
            _error(path, f"result schema lacks contract fields {sorted(missing)!r}")
    elif kind == "prompt":
        expected = {
            "contract_version",
            "prompt_id",
            "category",
            "language",
            "text",
            "target_prompt_tokens",
            "boundary_kind",
            "expected_behavior",
            "contains_sensitive_data",
        }
        _expect(set(properties), expected, f"{path}.properties")
        _expect(set(properties["category"].get("enum", [])), PROMPT_CATEGORIES, path)
    elif kind == "reference-fixture":
        _expect(
            set(properties),
            {
                "schema_version",
                "created_at",
                "generator",
                "provenance",
                "contract",
                "corpus",
                "generation",
                "rng",
                "cases",
            },
            f"{path}.properties",
        )
    else:
        raise ContractError(f"unknown schema kind {kind!r}")
    return schema


def validate_matrix(matrix: Any, root: Path) -> dict[str, Any]:
    path = "benchmarks/matrix.yaml"
    _exact_keys(
        matrix,
        {
            "contract_version",
            "matrix_id",
            "benchmark_scope",
            "tokenization",
            "primary_hardware",
            "model",
            "correctness_gate",
            "lane_manifests",
            "allowed_semantic_classes",
            "axes",
            "measurement",
            "cache_policy",
            "repeatability_gate",
            "expected_counts",
        },
        path,
    )
    _expect(matrix["contract_version"], CONTRACT_VERSION, f"{path}.contract_version")
    _expect(matrix["matrix_id"], "smollm2-135m-rtx4090-bf16-v1", f"{path}.matrix_id")
    _expect(matrix["benchmark_scope"], "end-to-end", f"{path}.benchmark_scope")
    _expect(
        matrix["tokenization"],
        {
            "input": "pretokenized",
            "latency_included": False,
            "add_special_tokens": True,
        },
        f"{path}.tokenization",
    )

    hardware = matrix["primary_hardware"]
    _exact_keys(hardware, {"gpu_model", "compute_capability", "dtype"}, f"{path}.primary_hardware")
    _expect(hardware["gpu_model"], GPU_MODEL, f"{path}.primary_hardware.gpu_model")
    _expect(
        hardware["compute_capability"],
        COMPUTE_CAPABILITY,
        f"{path}.primary_hardware.compute_capability",
    )
    _expect(hardware["dtype"], DTYPE, f"{path}.primary_hardware.dtype")

    model = matrix["model"]
    _exact_keys(
        model,
        {"id", "revision", "weights_file", "weights_sha256", "max_context_tokens"},
        f"{path}.model",
    )
    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_file": "model.safetensors",
        "weights_sha256": WEIGHTS_SHA256,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }
    _expect(model, expected_model, f"{path}.model")

    _expect(
        matrix["correctness_gate"],
        {
            "gate_id": "smollm2-fp32-bf16-native-e0-v2",
            "path": "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json",
        },
        f"{path}.correctness_gate",
    )
    correctness_gate_path = (root / matrix["correctness_gate"]["path"]).resolve()
    if root.resolve() not in correctness_gate_path.parents or not correctness_gate_path.is_file():
        _error(f"{path}.correctness_gate.path", "missing or repository-external gate")

    lane_paths = matrix["lane_manifests"]
    expected_lane_paths = [
        "benchmarks/lanes/hf-transformers.json",
        "benchmarks/lanes/vllm.json",
        "benchmarks/lanes/rustinfer-native.json",
    ]
    _expect(lane_paths, expected_lane_paths, f"{path}.lane_manifests")
    for relative in lane_paths:
        resolved = (root / relative).resolve()
        if root.resolve() not in resolved.parents:
            _error(f"{path}.lane_manifests", f"path escapes repository: {relative!r}")
        if not resolved.is_file():
            _error(f"{path}.lane_manifests", f"missing manifest {relative!r}")

    _expect(matrix["allowed_semantic_classes"], ["reference", "E0"], path)
    axes = matrix["axes"]
    _exact_keys(
        axes,
        {
            "concurrency",
            "prompt_tokens",
            "output_tokens",
            "sampling",
            "warm_state",
            "approximation_enabled",
        },
        f"{path}.axes",
    )
    _expect(axes["concurrency"], [1, 2, 4, 8], f"{path}.axes.concurrency")
    _expect(axes["prompt_tokens"], [128, 1024, 4096], f"{path}.axes.prompt_tokens")
    _expect(axes["output_tokens"], [32, 128], f"{path}.axes.output_tokens")
    _expect(axes["warm_state"], ["cold", "warm"], f"{path}.axes.warm_state")
    _expect(axes["approximation_enabled"], [False], f"{path}.axes.approximation_enabled")
    _expect(
        axes["sampling"],
        [
            {
                "id": "greedy",
                "strategy": "greedy",
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "seed": None,
                "ignore_eos": True,
                "fixed_output_length": True,
            }
        ],
        f"{path}.axes.sampling",
    )
    for prompt_tokens in axes["prompt_tokens"]:
        for output_tokens in axes["output_tokens"]:
            if prompt_tokens + output_tokens > MAX_CONTEXT_TOKENS:
                _error(path, "a prompt/output cell exceeds the pinned context length")

    measurement = matrix["measurement"]
    _exact_keys(
        measurement,
        {
            "independent_runs",
            "run_index_origin",
            "thermal_stabilization",
            "cold",
            "warm",
            "reported_percentiles",
        },
        f"{path}.measurement",
    )
    _expect(measurement["independent_runs"], 5, f"{path}.measurement.independent_runs")
    _expect(measurement["run_index_origin"], 1, f"{path}.measurement.run_index_origin")
    _expect(
        measurement["thermal_stabilization"],
        {
            "temperature_limit_c": 50,
            "retry_interval_seconds": 30,
            "maximum_wait_seconds": 1200,
            "retry_only_on_temperature_limit": True,
            "final_full_preflight_required": True,
        },
        f"{path}.measurement.thermal_stabilization",
    )
    _expect(measurement["reported_percentiles"], [50, 95], path)
    state_keys = {
        "warmup_iterations",
        "measured_iterations_per_run",
        "fresh_process_per_independent_run",
        "reset_model_state_per_independent_run",
        "reuse_model_within_run",
    }
    for state in ("cold", "warm"):
        _exact_keys(measurement[state], state_keys, f"{path}.measurement.{state}")
    _expect(
        measurement["cold"],
        {
            "warmup_iterations": 0,
            "measured_iterations_per_run": 1,
            "fresh_process_per_independent_run": True,
            "reset_model_state_per_independent_run": True,
            "reuse_model_within_run": False,
        },
        f"{path}.measurement.cold",
    )
    _expect(
        measurement["warm"],
        {
            "warmup_iterations": 5,
            "measured_iterations_per_run": 30,
            "fresh_process_per_independent_run": True,
            "reset_model_state_per_independent_run": True,
            "reuse_model_within_run": True,
        },
        f"{path}.measurement.warm",
    )

    _expect(
        matrix["cache_policy"],
        {
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
            "external_cache_environment": [
                "UV_CACHE_DIR",
                "UV_PYTHON_INSTALL_DIR",
                "HF_HOME",
                "VLLM_CACHE_ROOT",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "CUDA_CACHE_PATH",
            ],
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            "lane_prime_cells": {
                lane: [
                    {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32, "warm_state": "warm"},
                    {"concurrency": 1, "prompt_tokens": 4096, "output_tokens": 128, "warm_state": "warm"},
                    {"concurrency": 8, "prompt_tokens": 128, "output_tokens": 32, "warm_state": "warm"},
                ]
                for lane in ("hf-transformers", "vllm")
            },
            "reject_measured_cache_mutation": True,
        },
        f"{path}.cache_policy",
    )

    gate = matrix["repeatability_gate"]
    _exact_keys(gate, {"cells", "thresholds"}, f"{path}.repeatability_gate")
    _expect(
        gate["cells"],
        [
            {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32, "warm_state": "warm"},
            {"concurrency": 1, "prompt_tokens": 4096, "output_tokens": 128, "warm_state": "warm"},
            {"concurrency": 8, "prompt_tokens": 128, "output_tokens": 32, "warm_state": "warm"},
            {"concurrency": 1, "prompt_tokens": 128, "output_tokens": 32, "warm_state": "cold"},
        ],
        f"{path}.repeatability_gate.cells",
    )
    expected_thresholds = {
        "warm_p50_cv_max": 0.05,
        "warm_p95_cv_max": 0.10,
        "throughput_cv_max": 0.05,
        "cold_model_load_p50_cv_max": 0.10,
        "peak_vram_relative_range_max": 0.01,
        "failure_count_max": 0,
    }
    _expect(gate["thresholds"], expected_thresholds, f"{path}.repeatability_gate.thresholds")

    cell_count = math.prod(len(axes[key]) for key in axes)
    cells_per_state = cell_count // len(axes["warm_state"])
    raw_trials = measurement["independent_runs"] * cells_per_state * sum(
        measurement[state]["measured_iterations_per_run"] for state in axes["warm_state"]
    )
    expected_counts = {
        "cells_per_lane": cell_count,
        "independent_runs_per_cell": measurement["independent_runs"],
        "raw_trials_per_lane": raw_trials,
    }
    _expect(matrix["expected_counts"], expected_counts, f"{path}.expected_counts")
    return matrix


def _validate_command(command: Any, path: str) -> None:
    if not isinstance(command, dict):
        _error(path, "must be an object")
    allowed = {"status", "argv", "environment", "protocol", "output_contract"}
    if set(command) - allowed:
        _error(path, f"unexpected fields {sorted(set(command) - allowed)!r}")
    if not {"status", "argv"}.issubset(command):
        _error(path, "status and argv are required")
    if command["status"] not in {
        "available",
        "not-required",
        "adapter-required",
        "contract-only",
    }:
        _error(f"{path}.status", "unknown command status")
    argv = command["argv"]
    if argv is not None and (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
    ):
        _error(f"{path}.argv", "must be null or a non-empty string array")
    if command["status"] == "available" and argv is None:
        _error(path, "available commands need argv")
    if command["status"] in {"not-required", "adapter-required"} and argv is not None:
        _error(path, f"{command['status']} commands must have null argv")
    environment = command.get("environment")
    if environment is not None and (
        not isinstance(environment, dict)
        or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in environment.items()
        )
    ):
        _error(f"{path}.environment", "must be an object of string values")
    if "output_contract" in command:
        _expect(command["output_contract"], "benchmarks/schemas/result.schema.json", path)
    if "protocol" in command and not command["protocol"]:
        _error(f"{path}.protocol", "must not be empty")


def validate_lane_manifest(lane: Any, matrix: dict[str, Any], path: str) -> dict[str, Any]:
    _exact_keys(
        lane,
        {
            "contract_version",
            "lane_id",
            "implementation_id",
            "reference_implementation",
            "runtime_dependency_class",
            "semantic_class",
            "availability",
            "dependency_manifest",
            "engine",
            "model",
            "dtype",
            "sampling_id",
            "commands",
            "dependency_policy",
        },
        path,
    )
    _expect(lane["contract_version"], CONTRACT_VERSION, f"{path}.contract_version")
    if lane["lane_id"] not in LANE_IDS:
        _error(f"{path}.lane_id", "unknown lane")
    if not isinstance(lane["implementation_id"], str) or not lane["implementation_id"]:
        _error(f"{path}.implementation_id", "must be a non-empty string")
    _expect(lane["reference_implementation"], "hf-transformers-eager", path)
    if lane["semantic_class"] not in matrix["allowed_semantic_classes"]:
        _error(f"{path}.semantic_class", "not enabled by the matrix")
    _expect(lane["dtype"], DTYPE, f"{path}.dtype")
    _expect(lane["sampling_id"], "greedy", f"{path}.sampling_id")

    engine = lane["engine"]
    _exact_keys(
        engine,
        {"name", "version", "revision", "backend", "dependencies"},
        f"{path}.engine",
    )
    if not isinstance(engine["dependencies"], dict):
        _error(f"{path}.engine.dependencies", "must be an object")
    for name, version in engine["dependencies"].items():
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            _error(f"{path}.engine.dependencies", "dependency versions must be strings")

    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
    }
    _expect(lane["model"], expected_model, f"{path}.model")

    commands = lane["commands"]
    _exact_keys(commands, {"golden_fixture", "serve", "benchmark"}, f"{path}.commands")
    for name, command in commands.items():
        _validate_command(command, f"{path}.commands.{name}")

    policy = lane["dependency_policy"]
    _exact_keys(policy, {"allowed", "forbidden"}, f"{path}.dependency_policy")
    for name in ("allowed", "forbidden"):
        values = policy[name]
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            _error(f"{path}.dependency_policy.{name}", "must be a string array")
        if len(values) != len(set(values)):
            _error(f"{path}.dependency_policy.{name}", "must not contain duplicates")
    overlap = set(policy["allowed"]) & set(policy["forbidden"])
    if overlap:
        _error(path, f"dependencies cannot be both allowed and forbidden: {sorted(overlap)!r}")

    lane_id = lane["lane_id"]
    if lane_id == "hf-transformers":
        _expect(lane["dependency_manifest"], "tools/python/reference/pyproject.toml", path)
        _expect(lane["runtime_dependency_class"], "python-reference", path)
        _expect(lane["availability"], "available", path)
        _expect(lane["implementation_id"], "hf-transformers-eager", path)
        _expect(
            engine,
            {
                "name": "transformers",
                "version": "5.15.1",
                "revision": "transformers-5.15.1+torch-2.13.0",
                "backend": "eager",
                "dependencies": {
                    "nvidia-ml-py": "13.610.43",
                    "psutil": "7.2.2",
                    "safetensors": "0.8.0",
                    "torch": "2.13.0",
                    "transformers": "5.15.1",
                },
            },
            f"{path}.engine",
        )
        golden_argv = commands["golden_fixture"]["argv"]
        if (
            not golden_argv
            or "generate" not in golden_argv
            or not {"--offline", "--no-sync", "--prompts", "--repo-root", "--output"}.issubset(
                golden_argv
            )
        ):
            _error(path, "HF golden command lacks prompt/repository/output bindings")
        for flag, expected in (
            ("--prompts", "benchmarks/prompts.jsonl"),
            ("--repo-root", "."),
            (
                "--output",
                "/var/tmp/rustinfer-reference/smollm2-135m-bf16.json",
            ),
        ):
            if golden_argv.count(flag) != 1:
                _error(path, f"HF golden command must contain {flag} exactly once")
            position = golden_argv.index(flag)
            if position + 1 >= len(golden_argv):
                _error(path, f"HF golden command {flag} lacks a value")
            _expect(golden_argv[position + 1], expected, f"{path}.commands.golden_fixture")
        argv = commands["benchmark"]["argv"]
        _expect(
            argv[:8] if argv else None,
            [
                "uv", "run", "--frozen", "--offline", "--no-sync", "--project",
                "tools/python/reference", "rustinfer-reference",
            ],
            f"{path}.commands.benchmark.argv",
        )
        if "benchmark" not in argv or not SINGLE_CELL_BENCHMARK_FLAGS.issubset(argv):
            _error(path, "HF benchmark command is not bound to the benchmark protocol")
    elif lane_id == "vllm":
        _expect(lane["dependency_manifest"], "benchmarks/lanes/vllm/pyproject.toml", path)
        _expect(lane["runtime_dependency_class"], "python-reference", path)
        _expect(lane["availability"], "available", path)
        _expect(
            engine,
            {
                "name": "vllm",
                "version": "0.27.1",
                "revision": "vllm-0.27.1",
                "backend": "native",
                "dependencies": {
                    "nvidia-ml-py": "13.610.43",
                    "psutil": "7.2.2",
                    "vllm": "0.27.1",
                },
            },
            f"{path}.engine",
        )
        _expect(
            commands["serve"]["argv"][:9],
            [
                "uv",
                "run",
                "--frozen",
                "--offline",
                "--no-sync",
                "--project",
                "benchmarks/lanes/vllm",
                "vllm",
                "serve",
            ],
            path,
        )
        expected_vllm_environment = {
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_DO_NOT_TRACK": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
        _expect(
            commands["serve"].get("environment"),
            expected_vllm_environment,
            f"{path}.commands.serve.environment",
        )
        benchmark = commands["benchmark"]
        _expect(benchmark["status"], "available", path)
        _expect(
            benchmark.get("environment"),
            expected_vllm_environment,
            f"{path}.commands.benchmark.environment",
        )
        expected_prefix = [
            "uv",
            "run",
            "--frozen",
            "--offline",
            "--no-sync",
            "--project",
            "benchmarks/lanes/vllm",
            "rustinfer-vllm-benchmark",
        ]
        _expect(benchmark["argv"][:8], expected_prefix, path)
        if not SINGLE_CELL_BENCHMARK_FLAGS.issubset(benchmark["argv"]):
            _error(path, "vLLM benchmark command lacks a required single-cell flag")
    else:
        _expect(lane["dependency_manifest"], None, path)
        _expect(lane["runtime_dependency_class"], "native-production", path)
        _expect(lane["availability"], "contract-only", path)
        _expect(lane["semantic_class"], "E0", path)
        _expect(engine["name"], "rustinfer", path)
        _expect(engine["version"], None, path)
        required_forbidden = {
            "python",
            "pytorch",
            "transformers",
            "python-subprocess",
            "triton-python-jit",
        }
        if not required_forbidden.issubset(policy["forbidden"]):
            _error(path, "native lane does not forbid every Python fallback")
        lowered_allowed = " ".join(policy["allowed"]).lower()
        if any(name in lowered_allowed for name in ("python", "torch", "transformers", "triton")):
            _error(path, "native lane allows a Python reference dependency")
        for command_name in ("serve", "benchmark"):
            command = commands[command_name]
            _expect(command["status"], "contract-only", path)
            _expect(command["argv"][0], "rustinfer", path)
        if not SINGLE_CELL_BENCHMARK_FLAGS.issubset(commands["benchmark"]["argv"]):
            _error(path, "native benchmark command lacks a required single-cell flag")
    return lane


def validate_dependency_project(root: Path, lane: Mapping[str, Any]) -> None:
    manifest = lane["dependency_manifest"]
    if manifest is None:
        return
    manifest_path = (root / manifest).resolve()
    if root not in manifest_path.parents or not manifest_path.is_file():
        _error(f"lane {lane['lane_id']}", f"missing dependency manifest {manifest!r}")
    try:
        with manifest_path.open("rb") as handle:
            project_file = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        _error(str(manifest_path), f"invalid TOML: {exc}")
    project = project_file.get("project")
    if not isinstance(project, dict):
        _error(str(manifest_path), "missing [project] table")
    _expect(project.get("requires-python"), ">=3.13,<3.14", str(manifest_path))
    expected_dependencies = {
        "hf-transformers": [
            "nvidia-ml-py==13.610.43",
            "psutil==7.2.2",
            "safetensors==0.8.0",
            "torch==2.13.0",
            "transformers==5.15.1",
        ],
        "vllm": [
            "nvidia-ml-py==13.610.43",
            "psutil==7.2.2",
            "vllm==0.27.1",
        ],
    }
    _expect(
        project.get("dependencies"),
        expected_dependencies[lane["lane_id"]],
        f"{manifest_path}: project.dependencies",
    )
    python_version_path = manifest_path.parent / ".python-version"
    try:
        python_version = python_version_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        _error(str(python_version_path), f"cannot read Python pin: {exc}")
    _expect(python_version, "3.13.15", str(python_version_path))

    lock_path = manifest_path.parent / "uv.lock"
    try:
        with lock_path.open("rb") as handle:
            lock = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        _error(str(lock_path), f"invalid or missing uv lock: {exc}")
    _expect(lock.get("requires-python"), "==3.13.*", f"{lock_path}: requires-python")
    _expect(
        lock.get("options"),
        {"exclude-newer": "2026-08-24T23:59:59Z"},
        f"{lock_path}: options",
    )
    packages = lock.get("package")
    if not isinstance(packages, list) or not all(
        isinstance(package, dict) for package in packages
    ):
        _error(str(lock_path), "lock package list is invalid")
    packages_by_name: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        name = package.get("name")
        if isinstance(name, str):
            packages_by_name.setdefault(name, []).append(package)
    project_names = {
        "hf-transformers": "rustinfer-reference",
        "vllm": "rustinfer-vllm-benchmark-lane",
    }
    project_name = project_names[lane["lane_id"]]
    project_packages = packages_by_name.get(project_name, [])
    if len(project_packages) != 1:
        _error(str(lock_path), f"lock must contain one editable {project_name!r} package")
    locked_project = project_packages[0]
    _expect(locked_project.get("source"), {"editable": "."}, str(lock_path))
    expected_direct = {
        "hf-transformers": {
            "nvidia-ml-py": "13.610.43",
            "psutil": "7.2.2",
            "safetensors": "0.8.0",
            "torch": "2.13.0",
            "transformers": "5.15.1",
        },
        "vllm": {
            "nvidia-ml-py": "13.610.43",
            "psutil": "7.2.2",
            "vllm": "0.27.1",
        },
    }[lane["lane_id"]]
    locked_metadata = locked_project.get("metadata")
    if not isinstance(locked_metadata, dict):
        _error(str(lock_path), "editable package metadata must be an object")
    requires_dist = locked_metadata.get("requires-dist")
    if not isinstance(requires_dist, list):
        _error(str(lock_path), "editable package lacks requires-dist metadata")
    actual_direct = {
        requirement.get("name"): str(requirement.get("specifier", "")).removeprefix("==")
        for requirement in requires_dist
        if isinstance(requirement, dict) and isinstance(requirement.get("name"), str)
    }
    _expect(actual_direct, expected_direct, f"{lock_path}: direct requirements")
    for package_name, expected_version in expected_direct.items():
        locked = packages_by_name.get(package_name, [])
        if len(locked) != 1:
            _error(str(lock_path), f"expected one locked {package_name!r} package")
        _expect(
            locked[0].get("version"),
            expected_version,
            f"{lock_path}: {package_name} version",
        )
    if lane["lane_id"] == "vllm":
        scripts = project.get("scripts")
        if scripts != {
            "rustinfer-vllm-benchmark": "rustinfer_vllm_benchmark.cli:main"
        }:
            _error(str(manifest_path), "vLLM project does not expose the adapter console script")


def validate_prompts(path: Path, schema: dict[str, Any]) -> int:
    prompt_ids: set[str] = set()
    categories: set[str] = set()
    languages: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                _error(f"{path}:{line_number}", "blank JSONL lines are forbidden")
            try:
                prompt = json.loads(
                    raw_line, object_pairs_hook=_reject_duplicate_keys
                )
            except json.JSONDecodeError as exc:
                _error(f"{path}:{line_number}:{exc.colno}", f"invalid JSON: {exc.msg}")
            except ContractError as exc:
                _error(f"{path}:{line_number}", str(exc))
            try:
                validate_instance(prompt, schema)
            except ContractError as exc:
                _error(f"{path}:{line_number}", str(exc))
            prompt_id = prompt["prompt_id"]
            if prompt_id in prompt_ids:
                _error(f"{path}:{line_number}", f"duplicate prompt_id {prompt_id!r}")
            prompt_ids.add(prompt_id)
            categories.add(prompt["category"])
            languages.add(prompt["language"])
            if prompt["category"] == "early-eos" and (
                not prompt["text"]
                or prompt["language"] != "en"
                or prompt["target_prompt_tokens"] is not None
                or prompt["expected_behavior"]
                != "greedy-eos-at-first-output-token"
            ):
                _error(
                    f"{path}:{line_number}",
                    "early-eos must be a non-empty English prompt with null target "
                    "and greedy-eos-at-first-output-token behavior",
                )
            count += 1
    if count == 0:
        _error(str(path), "prompt corpus must not be empty")
    missing_categories = PROMPT_CATEGORIES - categories
    if missing_categories:
        _error(str(path), f"missing golden categories {sorted(missing_categories)!r}")
    if "ko" not in languages and "mixed" not in languages:
        _error(str(path), "corpus must exercise Korean")
    if "en" not in languages and "mixed" not in languages:
        _error(str(path), "corpus must exercise English")
    return count


def _fixture_git(root: Path, arguments: list[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        _error(
            "benchmarks/reference",
            f"cannot replay fixture Git provenance ({' '.join(arguments)}): {exc}",
        )


def validate_reference_fixture(
    root: Path,
    fixture_path: Path,
    fixture_schema: dict[str, Any],
    prompts_path: Path,
) -> int:
    """Validate the checked-in golden fixture and replay every source binding."""

    fixture = _read_json(fixture_path)
    try:
        validate_instance(fixture, fixture_schema)
    except ContractError as exc:
        _error(str(fixture_path), str(exc))
    _expect(
        fixture["generator"]["local_files_only"],
        True,
        f"{fixture_path}.generator.local_files_only",
    )
    _expect(fixture["generator"]["device"], "cuda:0", f"{fixture_path}.generator.device")
    _expect(
        fixture["generator"]["device_name"],
        GPU_MODEL,
        f"{fixture_path}.generator.device_name",
    )
    contract = fixture["contract"]
    _expect(contract["config_sha256"], CONFIG_SHA256, f"{fixture_path}.contract")
    _expect(
        contract["tokenizer_sha256"], TOKENIZER_SHA256, f"{fixture_path}.contract"
    )
    _expect(
        contract["tokenizer_files_sha256"],
        TOKENIZER_FILES_SHA256,
        f"{fixture_path}.contract.tokenizer_files_sha256",
    )

    provenance = fixture["provenance"]
    sources = provenance["sources"]
    _expect(
        set(sources),
        set(REFERENCE_FIXTURE_SOURCE_PATHS),
        f"{fixture_path}.provenance.sources",
    )
    revision = provenance["git_revision"]
    recorded_tree = provenance["git_tree"]
    actual_tree = _fixture_git(root, ["rev-parse", f"{revision}^{{tree}}"])
    try:
        actual_tree_text = actual_tree.decode("ascii").strip()
    except UnicodeDecodeError:
        _error(f"{fixture_path}.provenance.git_tree", "Git returned non-ASCII tree SHA")
    _expect(
        actual_tree_text,
        recorded_tree,
        f"{fixture_path}.provenance.git_tree",
    )
    for name, relative in REFERENCE_FIXTURE_SOURCE_PATHS.items():
        source = sources[name]
        _expect(source["path"], relative, f"{fixture_path}.provenance.sources.{name}.path")
        resolved = (root / relative).resolve()
        if root not in resolved.parents or not resolved.is_file():
            _error(
                f"{fixture_path}.provenance.sources.{name}",
                "bound source is missing or escapes the repository",
            )
        digest = source["sha256"]
        # A checked-in golden remains bound to the producer at its recorded
        # immutable revision. Later additive producer changes must not require
        # rewriting historical provenance or pretending that they generated
        # the old fixture; replay the exact Git object below instead.
        committed = _fixture_git(root, ["show", f"{revision}:{relative}"])
        _expect(
            hashlib.sha256(committed).hexdigest(),
            digest,
            f"{fixture_path}.provenance.sources.{name}.recorded_commit",
        )
    _expect(
        provenance["sources"]["python_version_file"]["sha256"],
        PYTHON_VERSION_FILE_SHA256,
        f"{fixture_path}.provenance.sources.python_version_file.sha256",
    )

    try:
        raw_prompts = prompts_path.read_bytes()
        prompt_text = raw_prompts.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _error(str(prompts_path), f"cannot read canonical UTF-8 prompt bytes: {exc}")
    prompt_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(prompt_text.splitlines(), start=1):
        if not line.strip():
            _error(f"{prompts_path}:{line_number}", "blank JSONL lines are forbidden")
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            _error(f"{prompts_path}:{line_number}", f"invalid JSON: {exc.msg}")
        except ContractError as exc:
            _error(f"{prompts_path}:{line_number}", str(exc))
        if not isinstance(row, dict):
            _error(f"{prompts_path}:{line_number}", "prompt must be an object")
        prompt_rows.append(row)
    corpus_digest = hashlib.sha256(raw_prompts).hexdigest()
    _expect(
        fixture["corpus"]["sha256"],
        corpus_digest,
        f"{fixture_path}.corpus.sha256",
    )
    _expect(
        fixture["corpus"]["prompt_count"],
        len(prompt_rows),
        f"{fixture_path}.corpus.prompt_count",
    )
    _expect(
        sources["prompts"]["sha256"],
        corpus_digest,
        f"{fixture_path}.provenance.sources.prompts.sha256",
    )
    cases = fixture["cases"]
    _expect(len(cases), len(prompt_rows), f"{fixture_path}.cases")
    eos_token_ids = set(fixture["generation"]["eos_token_ids"])
    for index, (case, prompt) in enumerate(zip(cases, prompt_rows, strict=True)):
        case_path = f"{fixture_path}.cases[{index}]"
        _expect(case["prompt_id"], prompt["prompt_id"], f"{case_path}.prompt_id")
        _expect(
            case["prompt_text_sha256"],
            hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest(),
            f"{case_path}.prompt_text_sha256",
        )
        metadata = {
            "category": prompt["category"],
            "language": prompt["language"],
            "target_prompt_tokens": prompt["target_prompt_tokens"],
            "boundary_kind": prompt["boundary_kind"],
            "expected_behavior": prompt["expected_behavior"],
        }
        _expect(case["prompt_metadata"], metadata, f"{case_path}.prompt_metadata")
        _expect(
            case["input"]["token_count"],
            len(case["input"]["token_ids"]),
            f"{case_path}.input.token_count",
        )
        if prompt["target_prompt_tokens"] is not None:
            _expect(
                case["input"]["token_count"],
                prompt["target_prompt_tokens"],
                f"{case_path}.input.token_count",
            )
        greedy = case["greedy"]
        _expect(
            greedy["cache_on_token_ids"],
            greedy["cache_off_token_ids"],
            f"{case_path}.greedy.cache_parity",
        )
        if prompt["category"] == "early-eos":
            _expect(greedy["stop_reason"], "eos", f"{case_path}.greedy.stop_reason")
            emitted = greedy["cache_on_token_ids"]
            if len(emitted) != 1 or emitted[0] not in eos_token_ids:
                _error(case_path, "early-eos must emit a configured EOS at output index 0")
    return len(cases)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                _error(f"{path}:{line_number}", "blank JSONL lines are forbidden")
            try:
                yield line_number, json.loads(
                    raw_line, object_pairs_hook=_reject_duplicate_keys
                )
            except json.JSONDecodeError as exc:
                _error(f"{path}:{line_number}:{exc.colno}", f"invalid JSON: {exc.msg}")
            except ContractError as exc:
                _error(f"{path}:{line_number}", str(exc))


def validate_result_file(
    path: Path,
    schema: dict[str, Any],
    matrix: dict[str, Any],
    matrix_path: Path,
    prompts_path: Path | None,
    lane_paths: dict[str, Path],
    lanes_by_implementation: dict[str, dict[str, Any]],
) -> int:
    count = 0
    expected_matrix_hash = _sha256(matrix_path)
    expected_prompts_hash = _sha256(prompts_path) if prompts_path else None
    for line_number, row in _iter_jsonl(path):
        row_path = f"{path}:{line_number}"
        try:
            validate_instance(row, schema)
        except ContractError as exc:
            _error(row_path, str(exc))
        _expect_comparable(
            row["scope"], matrix["benchmark_scope"], f"{row_path}.scope"
        )
        _expect_comparable(
            row["matrix_id"], matrix["matrix_id"], f"{row_path}.matrix_id"
        )
        _expect_comparable(
            row["matrix_sha256"], expected_matrix_hash, f"{row_path}.matrix_sha256"
        )
        if expected_prompts_hash is not None:
            _expect_comparable(
                row["prompts_sha256"],
                expected_prompts_hash,
                f"{row_path}.prompts_sha256",
            )
        _expect_comparable(row["model_id"], MODEL_ID, f"{row_path}.model_id")
        _expect_comparable(
            row["model_revision"], MODEL_REVISION, f"{row_path}.model_revision"
        )
        _expect_comparable(row["dtype"], DTYPE, f"{row_path}.dtype")
        max_trial_index = matrix["measurement"][row["warm_state"]][
            "measured_iterations_per_run"
        ]
        if row["trial_index"] > max_trial_index:
            _error(row_path, "trial_index exceeds the predeclared per-run iteration count")
        if row["semantic_class"] not in matrix["allowed_semantic_classes"]:
            _error(row_path, "semantic class is not enabled by this matrix")
        if row["semantic_class"] == "E0" and row["status"] == "success":
            expected_gate_id = matrix["correctness_gate"]["gate_id"]
            _expect(
                row["correctness_gate_id"],
                expected_gate_id,
                f"{row_path}.correctness_gate_id",
            )
            _error(
                row_path,
                "PR 01 native lane is contract-only: successful E0 results are "
                "fail-closed until the complete oracle/candidate manifest, sidecar, "
                "and executable bundle is approved by raw evidence replay with "
                "rustinfer-reference calibrate-validate-report",
            )
        else:
            _expect(
                row["correctness_gate_id"],
                None,
                f"{row_path}.correctness_gate_id",
            )
            _expect(
                row["correctness_report_sha256"],
                None,
                f"{row_path}.correctness_report_sha256",
            )
        _expect(row["approximation_enabled"], False, f"{row_path}.approximation_enabled")
        _expect(row["error_budget"], None, f"{row_path}.error_budget")
        _expect(row["seed"], None, f"{row_path}.seed")
        workload = row["workload"]
        if workload is None:
            _error(row_path, "the PR 01 matrix accepts only end-to-end rows")
        for key in ("concurrency", "prompt_tokens", "output_tokens", "warm_state"):
            if workload[key] not in matrix["axes"][key]:
                _error(f"{row_path}.workload.{key}", "value is outside the matrix")
        _expect(workload["sampling_id"], "greedy", f"{row_path}.workload.sampling_id")
        _expect(row["warm_state"], workload["warm_state"], f"{row_path}.warm_state")
        _expect_comparable(
            row["environment_id"],
            PRIMARY_ENVIRONMENT_ID,
            f"{row_path}.environment_id",
        )
        _expect_comparable(
            row["environment"]["gpu_model"],
            GPU_MODEL,
            f"{row_path}.environment.gpu_model",
        )
        _expect_comparable(
            row["environment"]["compute_capability"],
            COMPUTE_CAPABILITY,
            f"{row_path}.environment.compute_capability",
        )
        _expect_comparable(
            row["environment"]["gpu_count"],
            1,
            f"{row_path}.environment.gpu_count",
        )
        _expect_comparable(
            row["environment"]["nvidia_driver_version"],
            PRIMARY_DRIVER_VERSION,
            f"{row_path}.environment.nvidia_driver_version",
        )
        _expect_comparable(
            row["environment"]["ram_bytes"],
            PRIMARY_RAM_BYTES,
            f"{row_path}.environment.ram_bytes",
        )
        if "Ubuntu 22.04" not in row["environment"]["os"]:
            _comparison_error(
                f"{row_path}.environment.os", "primary OS must contain 'Ubuntu 22.04'"
            )
        if "i7-13700K" not in row["environment"]["cpu_model"]:
            _comparison_error(
                f"{row_path}.environment.cpu_model",
                "primary CPU must contain 'i7-13700K'",
            )

        implementation = row["implementation_id"]
        lane = lanes_by_implementation.get(implementation)
        if lane is None:
            _error(row_path, f"unknown implementation_id {implementation!r}")
        _expect_comparable(
            row["semantic_class"], lane["semantic_class"], f"{row_path}.semantic_class"
        )
        _expect_comparable(
            row["reference_implementation"],
            lane["reference_implementation"],
            row_path,
        )
        _expect_comparable(
            row["runtime_dependency_class"], lane["runtime_dependency_class"], row_path
        )
        _expect_comparable(
            row["engine_revision"], lane["engine"]["revision"], row_path
        )
        lane_path = lane_paths[lane["lane_id"]]
        _expect_comparable(
            row["lane_manifest_sha256"], _sha256(lane_path), row_path
        )

        requests = row["requests"]
        if len(requests) != workload["concurrency"]:
            _error(row_path, "request count must equal concurrency")
        request_failures = sum(request["status"] == "failure" for request in requests)
        if request_failures != row["failure_count"]:
            _error(row_path, "failure_count does not match failed request observations")
        if (row["status"] == "failure") != (request_failures > 0):
            _error(row_path, "row status does not match request failure observations")
        for request in requests:
            _expect(
                request["prompt_tokens"],
                workload["prompt_tokens"],
                f"{row_path}.requests[].prompt_tokens",
            )
            _expect(
                request["requested_output_tokens"],
                workload["output_tokens"],
                f"{row_path}.requests[].requested_output_tokens",
            )
            if request["status"] == "success":
                _expect(
                    request["generated_tokens"],
                    workload["output_tokens"],
                    f"{row_path}.requests[].generated_tokens",
                )
                generated_tokens = request["generated_tokens"]
                itl_ms = request["itl_ms"]
                if not isinstance(itl_ms, list):
                    _error(row_path, "successful request itl_ms must be an array")
                if len(itl_ms) != max(0, generated_tokens - 1):
                    _error(
                        row_path,
                        "successful request itl_ms length must equal generated_tokens - 1",
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                    for value in itl_ms
                ):
                    _error(row_path, "successful request ITLs must be finite and nonnegative")
                mean_tpot_ms = request["mean_tpot_ms"]
                expected_mean = (
                    math.fsum(float(value) for value in itl_ms) / len(itl_ms)
                    if itl_ms
                    else 0.0
                )
                if not math.isclose(
                    float(mean_tpot_ms),
                    expected_mean,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    _error(row_path, "mean_tpot_ms must equal the mean of itl_ms")
                ttft_ms = float(request["ttft_ms"])
                end_to_end_ms = float(request["end_to_end_ms"])
                if ttft_ms > end_to_end_ms:
                    _error(row_path, "request ttft_ms cannot exceed end_to_end_ms")
                if ttft_ms + math.fsum(float(value) for value in itl_ms) > (
                    end_to_end_ms + max(1e-9, end_to_end_ms * 1e-9)
                ):
                    _error(
                        row_path,
                        "request end_to_end_ms cannot precede its final token observation",
                    )
            else:
                _expect(
                    request["generated_tokens"],
                    0,
                    f"{row_path}.requests[].generated_tokens",
                )
                _expect(
                    request["generated_token_ids_sha256"],
                    EMPTY_TOKEN_IDS_SHA256,
                    f"{row_path}.requests[].generated_token_ids_sha256",
                )
                for field in ("ttft_ms", "end_to_end_ms", "mean_tpot_ms", "itl_ms"):
                    _expect(
                        request[field],
                        None,
                        f"{row_path}.requests[].{field}",
                    )
        count += 1
    if count == 0:
        _error(str(path), "result JSONL must contain at least one trial")
    return count


def validate_contract(root: Path, explicit_results: Iterable[Path] = ()) -> dict[str, int]:
    root = root.resolve()
    matrix_path = root / "benchmarks/matrix.yaml"
    result_schema_path = root / "benchmarks/schemas/result.schema.json"
    prompt_schema_path = root / "benchmarks/schemas/prompt.schema.json"
    fixture_schema_path = root / "benchmarks/schemas/reference-fixture.schema.json"
    correctness_gate_schema_path = root / "benchmarks/schemas/correctness-gate.schema.json"
    prompts_path = root / "benchmarks/prompts.jsonl"
    fixture_path = root / "benchmarks/reference/smollm2-135m-bf16.json"

    matrix = validate_matrix(_read_json(matrix_path), root)
    result_schema = validate_schema_document(
        _read_json(result_schema_path), "result", str(result_schema_path)
    )
    prompt_schema = validate_schema_document(
        _read_json(prompt_schema_path), "prompt", str(prompt_schema_path)
    )
    fixture_schema = validate_schema_document(
        _read_json(fixture_schema_path),
        "reference-fixture",
        str(fixture_schema_path),
    )
    correctness_gate_schema = _read_json(correctness_gate_schema_path)
    _expect(
        correctness_gate_schema.get("$schema"),
        SCHEMA_DIALECT,
        f"{correctness_gate_schema_path}.$schema",
    )
    _walk_schema_strictness(correctness_gate_schema, str(correctness_gate_schema_path))
    correctness_gate_path = root / matrix["correctness_gate"]["path"]
    correctness_gate = _read_json(correctness_gate_path)
    validate_instance(correctness_gate, correctness_gate_schema)
    _expect(
        correctness_gate["gate_id"],
        matrix["correctness_gate"]["gate_id"],
        f"{correctness_gate_path}.gate_id",
    )
    _expect(correctness_gate["model"]["id"], MODEL_ID, str(correctness_gate_path))
    _expect(
        correctness_gate["model"]["revision"],
        MODEL_REVISION,
        str(correctness_gate_path),
    )
    _expect(
        correctness_gate["model"]["weights_sha256"],
        WEIGHTS_SHA256,
        str(correctness_gate_path),
    )
    validate_threshold_calibration_evidence(
        root,
        correctness_gate,
        correctness_gate_path,
    )

    lane_paths: dict[str, Path] = {}
    lanes_by_implementation: dict[str, dict[str, Any]] = {}
    for relative in matrix["lane_manifests"]:
        lane_path = root / relative
        lane = validate_lane_manifest(_read_json(lane_path), matrix, str(lane_path))
        validate_dependency_project(root, lane)
        lane_id = lane["lane_id"]
        if lane_id in lane_paths:
            _error(str(lane_path), f"duplicate lane_id {lane_id!r}")
        if lane["implementation_id"] in lanes_by_implementation:
            _error(str(lane_path), "duplicate implementation_id")
        if lane_path.stem != lane_id:
            _error(str(lane_path), "manifest filename must match lane_id")
        lane_paths[lane_id] = lane_path
        lanes_by_implementation[lane["implementation_id"]] = lane
    _expect(set(lane_paths), LANE_IDS, "benchmarks/lanes")
    if "hf-transformers-eager" not in lanes_by_implementation:
        _error("benchmarks/lanes", "reference implementation has no manifest")

    prompt_count = validate_prompts(prompts_path, prompt_schema) if prompts_path.is_file() else 0
    if not fixture_path.is_file():
        _error(str(fixture_path), "canonical golden fixture is required")
    reference_case_count = validate_reference_fixture(
        root,
        fixture_path,
        fixture_schema,
        prompts_path,
    )
    result_paths = {path.resolve() for path in explicit_results}
    result_paths.update((root / "benchmarks/results").glob("**/raw.jsonl"))
    result_count = 0
    for result_path in sorted(result_paths):
        result_count += validate_result_file(
            result_path,
            result_schema,
            matrix,
            matrix_path,
            prompts_path if prompts_path.is_file() else None,
            lane_paths,
            lanes_by_implementation,
        )
    return {
        "lanes": len(lane_paths),
        "prompts": prompt_count,
        "reference_cases": reference_case_count,
        "trials": result_count,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="repository root (defaults to the root containing this script)",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        type=Path,
        help="additional raw result JSONL to validate; may be repeated",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result_paths = [
        path if path.is_absolute() else args.root / path for path in args.result
    ]
    try:
        counts = validate_contract(args.root, result_paths)
    except (ContractError, OSError) as exc:
        print(f"benchmark contract validation failed: {exc}", file=sys.stderr)
        return 1
    prompt_note = str(counts["prompts"]) if counts["prompts"] else "absent (allowed)"
    print(
        "benchmark contract valid: "
        f"{counts['lanes']} lanes, {prompt_note} prompts, "
        f"{counts['reference_cases']} golden cases, {counts['trials']} result trials"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
