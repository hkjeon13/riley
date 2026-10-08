#!/usr/bin/env python3
"""Validate the closed PR 17 extension-admission registry.

The checker intentionally uses only the Python standard library. JSON Schema
files document the portable artifact format; this program is the mandatory CI
gate and repeats the security- and policy-sensitive cross-file checks without
requiring a runtime dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib
from typing import Any


REGISTRY_PATH = "deploy/extensions/registry.json"
PROPOSAL_ROOT = "deploy/extensions/proposals"
PLAN_ROOT = "deploy/extensions/plans"
CONTRACT_ROOT = "benchmarks/extensions/contracts"
IMPLEMENTATION_ROOT = "deploy/extensions/implementations"

REGISTRY_SCHEMA = "deploy/extensions/registry.schema.json"
PROPOSAL_SCHEMA = "deploy/extensions/proposal.schema.json"
CONTRACT_SCHEMA = "benchmarks/extensions/benchmark-contract.schema.json"
IMPLEMENTATION_SCHEMA = "deploy/extensions/implementation.schema.json"
SCHEMA_SEMANTIC_SHA256 = {
    REGISTRY_SCHEMA: "c66d72df9782b6733ddfc5050c1e77da547007f058f1fba98aa1cc60d5e0a0eb",
    PROPOSAL_SCHEMA: "497b119e0da7521d921e4ab52ec68026dfcfafc22385e1a48749c5b0c627b986",
    CONTRACT_SCHEMA: "69897cb86aa847c21e56e340ca3c7709dee7c016c3ad67182ad3eb4c92153b24",
    IMPLEMENTATION_SCHEMA: "11f77429ffa7eea134cd8fd2e0db2e1a8c99bb208061879d6e159f3ff2e8caff",
}

SEMANTIC_CLASSES = {"reference", "E0", "E1", "A1", "M1"}
EXTENSION_TRACKS = {
    "quantization",
    "low-rank-weight-compression",
    "kv-compression",
    "prefix-cache",
    "kv-offload-prefetch",
    "query-aware-kv-selection",
    "moe",
    "mamba-ssm",
    "multimodal",
    "speculative-decoding",
    "jacobi-lookahead",
}
TRACK_SEMANTIC_CLASSES = {
    "quantization": {"E0", "A1"},
    "low-rank-weight-compression": {"A1"},
    "kv-compression": {"A1"},
    "prefix-cache": {"reference"},
    "kv-offload-prefetch": {"reference"},
    "query-aware-kv-selection": {"A1"},
    "moe": {"E0", "M1"},
    "mamba-ssm": {"E0"},
    "multimodal": {"E0", "M1"},
    "speculative-decoding": {"E0", "E1"},
    "jacobi-lookahead": {"E0", "A1", "M1"},
}
EXTENSION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
RUNTIME_FLAG = re.compile(r"^RILEY_EXPERIMENTAL_[A-Z0-9_]+$")
RUNTIME_FLAG_BYTES = re.compile(rb"RILEY_EXPERIMENTAL_[A-Z0-9_]+")
RUST_TEST_ID = re.compile(r"^[a-z_][a-z0-9_]{0,127}$")
REPOSITORY_PATH = re.compile(
    r"^(?!/)(?!.*\\)(?!(?:.*/)?\.{1,2}(?:/|$))"
    r"(?!(?:.*/)?(?:\.git|\.gitignore|\.gitattributes|\.gitmodules|\.mailmap)(?:/|$))"
    r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MODEL_REVISION = re.compile(r"^[0-9a-f]{40}$")
VALIDATION_TEST_PATH = re.compile(
    r"^crates/[A-Za-z0-9_-]+/tests/[A-Za-z0-9_][A-Za-z0-9_.-]*\.rs$"
)
GIT_CONTROL_NAMES = {".git", ".gitignore", ".gitattributes", ".gitmodules", ".mailmap"}
RUNTIME_SOURCE_SUFFIXES = {
    ".rs",
    ".toml",
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
}

# Every admission metric must name a scalar numeric observation that the
# immutable PR 01 result schema can actually carry. Extensions needing another
# metric must first version that common result contract.
COMMON_END_TO_END_METRICS = {
    "metrics.batch_wall_ms",
    "metrics.output_tokens_per_second",
    "metrics.cpu_utilization_percent",
    "metrics.gpu_utilization_percent",
    "metrics.peak_vram_bytes",
    "requests[].ttft_ms",
    "requests[].end_to_end_ms",
    "requests[].mean_tpot_ms",
}
QUANTIZATION_METRICS = {
    "metrics.model_load_ms",
    "quantization.transform_runtime_ms",
    "quantization.weight_bytes",
    "quantization.kv_bytes",
    "quantization.gemm_throughput_tflops",
}
SPECULATIVE_METRICS = {
    "speculative.lookahead",
    "speculative.acceptance_rate",
    "speculative.accepted_tokens_per_verify",
    "speculative.target_calls_per_output_token",
    "speculative.draft_latency_ms",
    "speculative.verification_latency_ms",
    "speculative.rejected_suffix_tokens",
    "speculative.rollback_count",
}
SPARSE_ATTENTION_METRICS = {
    "sparse_attention.selected_pages",
    "sparse_attention.total_pages",
    "sparse_attention.page_metadata_bytes",
    "sparse_attention.page_bound_time_ms",
    "sparse_attention.exact_fallback_rate",
}
TRACK_PRIMARY_METRICS = {
    "quantization": COMMON_END_TO_END_METRICS | QUANTIZATION_METRICS,
    "low-rank-weight-compression": COMMON_END_TO_END_METRICS
    | {"metrics.model_load_ms"},
    "kv-compression": COMMON_END_TO_END_METRICS | {"quantization.kv_bytes"},
    "prefix-cache": COMMON_END_TO_END_METRICS,
    "kv-offload-prefetch": COMMON_END_TO_END_METRICS,
    "query-aware-kv-selection": COMMON_END_TO_END_METRICS
    | SPARSE_ATTENTION_METRICS,
    "moe": COMMON_END_TO_END_METRICS | {"metrics.model_load_ms"},
    "mamba-ssm": COMMON_END_TO_END_METRICS,
    "multimodal": COMMON_END_TO_END_METRICS | {"metrics.model_load_ms"},
    "speculative-decoding": COMMON_END_TO_END_METRICS | SPECULATIVE_METRICS,
    "jacobi-lookahead": COMMON_END_TO_END_METRICS
    | {
        "speculative.lookahead",
        "speculative.target_calls_per_output_token",
        "speculative.rejected_suffix_tokens",
        "speculative.rollback_count",
    },
}
PERFORMANCE_METRIC_PATHS = set().union(*TRACK_PRIMARY_METRICS.values())
TRACK_REQUIRED_METRICS = {
    "quantization": {
        "metrics.output_tokens_per_second",
        "metrics.peak_vram_bytes",
        "quantization.transform_runtime_ms",
        "requests[].mean_tpot_ms",
        "requests[].ttft_ms",
    },
    "low-rank-weight-compression": {
        "metrics.output_tokens_per_second",
        "metrics.peak_vram_bytes",
        "requests[].mean_tpot_ms",
    },
    "kv-compression": {
        "metrics.output_tokens_per_second",
        "metrics.peak_vram_bytes",
        "quantization.kv_bytes",
        "requests[].mean_tpot_ms",
    },
    "prefix-cache": {
        "metrics.output_tokens_per_second",
        "metrics.peak_vram_bytes",
        "requests[].ttft_ms",
    },
    "kv-offload-prefetch": {
        "metrics.cpu_utilization_percent",
        "metrics.gpu_utilization_percent",
        "metrics.output_tokens_per_second",
        "metrics.peak_vram_bytes",
        "requests[].mean_tpot_ms",
    },
    "query-aware-kv-selection": COMMON_END_TO_END_METRICS
    | SPARSE_ATTENTION_METRICS,
    "moe": {
        "metrics.gpu_utilization_percent",
        "metrics.output_tokens_per_second",
        "metrics.peak_vram_bytes",
        "requests[].mean_tpot_ms",
    },
    "mamba-ssm": {
        "metrics.batch_wall_ms",
        "metrics.gpu_utilization_percent",
        "metrics.output_tokens_per_second",
        "metrics.peak_vram_bytes",
    },
    "multimodal": {
        "metrics.model_load_ms",
        "metrics.output_tokens_per_second",
        "metrics.peak_vram_bytes",
        "requests[].ttft_ms",
    },
    "speculative-decoding": SPECULATIVE_METRICS
    | {
        "metrics.output_tokens_per_second",
        "requests[].mean_tpot_ms",
        "requests[].ttft_ms",
    },
    "jacobi-lookahead": {
        "metrics.gpu_utilization_percent",
        "metrics.output_tokens_per_second",
        "requests[].mean_tpot_ms",
        "speculative.lookahead",
        "speculative.rejected_suffix_tokens",
        "speculative.rollback_count",
        "speculative.target_calls_per_output_token",
    },
}
QUALITY_METRIC_PATHS = {
    "failure_count",
    "sparse_attention.omitted_mass_bound",
}
RESULT_METRIC_PATHS = PERFORMANCE_METRIC_PATHS | QUALITY_METRIC_PATHS

TRACK_CLASS_QUALITY_METRICS = {
    (track, semantic_class): {"failure_count"}
    for track, classes in TRACK_SEMANTIC_CLASSES.items()
    for semantic_class in classes
    if semantic_class in {"reference", "E0"}
}
TRACK_CLASS_QUALITY_METRICS[("query-aware-kv-selection", "A1")] = {
    "sparse_attention.omitted_mass_bound",
}

REGISTRY_KEYS = {"$schema", "schema_version", "extensions"}
ENTRY_KEYS = {
    "extension_id",
    "status",
    "track",
    "semantic_class",
    "proposal_path",
    "deploy_document_path",
    "benchmark_contract_path",
    "implementation_link_path",
}
PROPOSAL_KEYS = {
    "$schema",
    "schema_version",
    "extension_id",
    "status",
    "track",
    "title",
    "semantic_class",
    "problem_statement",
    "implementation_boundary",
    "reference_path",
    "reference_sha256",
    "fallback_path",
    "fallback_sha256",
    "primary_metric",
    "required_metrics",
    "quality_or_error_metric",
    "runtime_flag",
    "default_enabled",
    "stable_default",
    "result_disclosure",
    "rollback",
    "deploy_document_path",
    "benchmark_contract_path",
    "class_gate",
    "approval_answers",
}
CONTRACT_KEYS = {
    "$schema",
    "schema_version",
    "extension_id",
    "status",
    "track",
    "semantic_class",
    "proposal_path",
    "deploy_document_path",
    "reference_path",
    "reference_sha256",
    "fallback_path",
    "fallback_sha256",
    "runtime_flag",
    "primary_metric",
    "required_metrics",
    "quality_or_error_metric",
    "workloads",
    "comparison_environment",
    "measurement",
    "class_gate",
}
IMPLEMENTATION_KEYS = {
    "$schema",
    "schema_version",
    "extension_id",
    "status",
    "proposal_path",
    "deploy_document_path",
    "benchmark_contract_path",
    "runtime_flag",
    "runtime_flag_source_path",
    "implementation_paths",
    "validation_tests",
    "default_enabled",
    "stable_default",
}

APPROVAL_ANSWER_KEYS = {
    "user_workload_bottleneck",
    "semantic_class_rationale",
    "existing_ir_expression",
    "implementation_location_rationale",
    "correctness_reference",
    "error_or_distribution_contract",
    "memory_and_operational_complexity",
    "fallback_and_rollback",
    "expected_resource_reduction",
    "end_to_end_benefit_hypothesis",
}
CLASS_GATE_KEYS = {
    "reference": {
        "kind",
        "behavioral_parity",
        "token_parity",
        "stable_fallback",
        "lifetime_resource_regression",
    },
    "E0": {
        "kind",
        "reference_parity",
        "dtype_tolerances",
        "extreme_value_cases",
        "token_level_regression",
    },
    "E1": {
        "kind",
        "distribution_contract",
        "statistical_test",
        "rng_isolation",
        "rng_snapshot_restore",
        "greedy_exact",
        "fixed_seed_definition",
    },
    "A1": {
        "kind",
        "error_budget",
        "exact_fallback",
        "opt_in",
        "usage_disclosure",
        "quality_latency_curve",
    },
    "M1": {
        "kind",
        "research_track",
        "calibration_or_training_provenance",
        "production_core_isolated",
        "opt_in",
        "usage_disclosure",
        "quality_latency_curve",
    },
}


class ExtensionGateError(RuntimeError):
    """Raised when extension metadata fails closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExtensionGateError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ExtensionGateError(f"{label}: cannot read {path}: {error}") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except ExtensionGateError as error:
        raise ExtensionGateError(f"{label}: {error}") from error
    except json.JSONDecodeError as error:
        raise ExtensionGateError(f"{label}: invalid JSON: {error}") from error


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExtensionGateError(f"{label}: expected an object")
    return value


def _closed_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    obj = _object(value, label)
    missing = sorted(keys - obj.keys())
    unknown = sorted(obj.keys() - keys)
    if missing:
        raise ExtensionGateError(f"{label}: missing fields: {', '.join(missing)}")
    if unknown:
        raise ExtensionGateError(f"{label}: unknown fields: {', '.join(unknown)}")
    return obj


def _string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ExtensionGateError(f"{label}: expected one non-empty trimmed line")
    return value


def _constant(value: Any, expected: Any, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ExtensionGateError(f"{label}: expected {expected!r}, got {value!r}")


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare parsed JSON values without Python's bool/int coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtensionGateError(f"{label}: expected a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ExtensionGateError(f"{label}: expected a finite non-negative number")
    return result


def _identifier(value: Any, label: str) -> str:
    result = _string(value, label)
    if not EXTENSION_ID.fullmatch(result):
        raise ExtensionGateError(f"{label}: invalid extension id {result!r}")
    return result


def _metric(value: Any, label: str) -> str:
    result = _string(value, label)
    if result not in RESULT_METRIC_PATHS:
        raise ExtensionGateError(
            f"{label}: metric must be a scalar path in the common result schema"
        )
    return result


def _sha256(value: Any, label: str) -> str:
    result = _string(value, label)
    if not SHA256.fullmatch(result):
        raise ExtensionGateError(f"{label}: expected lowercase SHA-256")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ExtensionGateError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _rust_code_without_comments_or_literals(raw: bytes, label: str) -> str:
    try:
        source = raw.decode("utf-8")
    except UnicodeError as error:
        raise ExtensionGateError(f"{label}: expected UTF-8 Rust source") from error
    cleaned = list(source)

    def blank(start: int, end: int) -> None:
        for position in range(start, end):
            if cleaned[position] not in {"\n", "\r"}:
                cleaned[position] = " "

    index = 0
    while index < len(source):
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            if end < 0:
                end = len(source)
            blank(index, end)
            index = end
            continue
        if source.startswith("/*", index):
            depth = 1
            end = index + 2
            while end < len(source) and depth:
                if source.startswith("/*", end):
                    depth += 1
                    end += 2
                elif source.startswith("*/", end):
                    depth -= 1
                    end += 2
                else:
                    end += 1
            if depth:
                raise ExtensionGateError(f"{label}: unterminated Rust block comment")
            blank(index, end)
            index = end
            continue

        raw_prefix_length = 0
        for prefix in ("br", "cr", "r"):
            if source.startswith(prefix, index):
                cursor = index + len(prefix)
                while cursor < len(source) and source[cursor] == "#":
                    cursor += 1
                if cursor < len(source) and source[cursor] == '"':
                    raw_prefix_length = cursor - index
                    break
        if raw_prefix_length:
            quote = index + raw_prefix_length
            hashes = source[index:quote].count("#")
            terminator = '"' + ("#" * hashes)
            end = source.find(terminator, quote + 1)
            if end < 0:
                raise ExtensionGateError(f"{label}: unterminated Rust raw string")
            end += len(terminator)
            blank(index, end)
            index = end
            continue

        string_prefix = 1 if source.startswith(("b\"", "c\""), index) else 0
        if source[index] == '"' or string_prefix:
            quote = index + string_prefix
            end = quote + 1
            escaped = False
            while end < len(source):
                character = source[end]
                if character == '"' and not escaped:
                    end += 1
                    break
                if character == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
                end += 1
            else:
                raise ExtensionGateError(f"{label}: unterminated Rust string")
            blank(index, end)
            index = end
            continue

        char_start = index + 1 if source.startswith("b'", index) else index
        if char_start < len(source) and source[char_start] == "'":
            cursor = char_start + 1
            if cursor < len(source) and source[cursor] == "\\":
                cursor += 2
                while cursor < len(source) and source[cursor] not in {"'", "\n", "\r"}:
                    cursor += 1
            else:
                cursor += 1
            if cursor < len(source) and source[cursor] == "'":
                end = cursor + 1
                blank(index, end)
                index = end
                continue
        index += 1
    return "".join(cleaned)


def _validate_registered_rust_test(raw: bytes, test_id: str, label: str) -> None:
    if not RUST_TEST_ID.fullmatch(test_id):
        raise ExtensionGateError(f"{label}.test_id: expected a Rust test identifier")
    code = _rust_code_without_comments_or_literals(raw, f"{label}.path")
    if re.search(r"#\s*!?\s*\[\s*(?:cfg|cfg_attr|ignore)\b", code):
        raise ExtensionGateError(
            f"{label}.path: validation tests cannot be cfg-gated or ignored"
        )
    declaration = re.compile(
        rf"#\s*\[\s*test\s*\]\s*fn\s+{re.escape(test_id)}\s*\(\s*\)"
    )
    matches = list(declaration.finditer(code))
    if len(matches) != 1:
        raise ExtensionGateError(
            f"{label}.test_id: expected exactly one direct #[test] function"
        )
    depths = {"{": 0, "(": 0, "[": 0}
    closing = {"}": "{", ")": "(", "]": "["}
    for character in code[: matches[0].start()]:
        if character in depths:
            depths[character] += 1
        elif character in closing:
            opener = closing[character]
            depths[opener] -= 1
            if depths[opener] < 0:
                raise ExtensionGateError(f"{label}.path: unbalanced Rust delimiters")
    if any(depths.values()):
        raise ExtensionGateError(
            f"{label}.test_id: #[test] function must be declared at target top level"
        )


def _git_tracked_files(root: Path) -> set[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "-z", "--"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise ExtensionGateError(f"cannot enumerate Git-tracked files: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExtensionGateError(
            f"cannot enumerate Git-tracked files under repository root: {detail}"
        )
    try:
        names = completed.stdout.decode("utf-8").split("\0")
    except UnicodeError as error:
        raise ExtensionGateError("Git index contains a non-UTF-8 path") from error
    return {name for name in names if name}


def _git_bytes(root: Path, arguments: list[str], label: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise ExtensionGateError(f"{label}: cannot execute Git: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExtensionGateError(f"{label}: Git command failed: {detail}")
    return completed.stdout


def _git_show_optional(root: Path, revision: str, relative: str) -> bytes | None:
    object_name = f"{revision}:{relative}"
    try:
        probe = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", object_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise ExtensionGateError(f"transition: cannot execute Git: {error}") from error
    if probe.returncode != 0:
        return None
    return _git_bytes(root, ["show", object_name], f"transition {relative}")


def _is_runtime_source(relative: str) -> bool:
    return relative.startswith("crates/") and Path(relative).suffix in RUNTIME_SOURCE_SUFFIXES


def _runtime_flags_in_bytes(raw: bytes) -> set[str]:
    return {match.decode("ascii") for match in RUNTIME_FLAG_BYTES.findall(raw)}


def _base_runtime_flags(root: Path, revision: str) -> set[str]:
    raw_paths = _git_bytes(
        root,
        ["ls-tree", "-r", "--name-only", "-z", revision, "--", "crates"],
        "transition base runtime source inventory",
    )
    try:
        paths = raw_paths.decode("utf-8").split("\0")
    except UnicodeError as error:
        raise ExtensionGateError(
            "transition base runtime source inventory contains a non-UTF-8 path"
        ) from error
    flags: set[str] = set()
    for relative in paths:
        if not relative or not _is_runtime_source(relative):
            continue
        raw = _git_show_optional(root, revision, relative)
        if raw is None:
            raise ExtensionGateError(
                f"transition base runtime source disappeared: {relative}"
            )
        flags.update(_runtime_flags_in_bytes(raw))
    return flags


def _current_runtime_flags(root: Path) -> set[str]:
    tracked_files = _git_tracked_files(root)
    flags: set[str] = set()
    for relative in tracked_files:
        if not _is_runtime_source(relative):
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ExtensionGateError(
                f"transition current runtime source must be a regular file: {relative}"
            )
        try:
            flags.update(_runtime_flags_in_bytes(path.read_bytes()))
        except OSError as error:
            raise ExtensionGateError(
                f"transition cannot read current runtime source {relative}: {error}"
            ) from error
    return flags


def _load_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ExtensionGateError(f"{label}: expected UTF-8 JSON") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ExtensionGateError) as error:
        raise ExtensionGateError(f"{label}: invalid JSON: {error}") from error


def _relative_parts(value: Any, label: str) -> tuple[str, ...]:
    path = _string(value, label)
    if "\\" in path or path.startswith("/"):
        raise ExtensionGateError(f"{label}: path must be repository-relative POSIX")
    raw_parts = path.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ExtensionGateError(f"{label}: path traversal or non-canonical path")
    if any(part in GIT_CONTROL_NAMES for part in raw_parts):
        raise ExtensionGateError(f"{label}: Git control paths are forbidden")
    if PurePosixPath(path).as_posix() != path:
        raise ExtensionGateError(f"{label}: path is not canonical")
    if not REPOSITORY_PATH.fullmatch(path):
        raise ExtensionGateError(
            f"{label}: path must use portable ASCII repository segments"
        )
    return tuple(raw_parts)


def _checked_file(
    root: Path,
    value: Any,
    label: str,
    *,
    required_prefix: str | None = None,
    required_suffix: str | None = None,
    tracked_files: set[str] | None = None,
) -> tuple[str, Path]:
    parts = _relative_parts(value, label)
    relative = "/".join(parts)
    if required_prefix is not None and parts[:-1] != tuple(required_prefix.split("/")):
        raise ExtensionGateError(f"{label}: must be directly under {required_prefix}/")
    if required_suffix is not None and not parts[-1].endswith(required_suffix):
        raise ExtensionGateError(f"{label}: must end in {required_suffix}")

    candidate = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ExtensionGateError(f"{label}: symlinks are forbidden: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ExtensionGateError(f"{label}: file does not exist: {relative}") from error
    root_resolved = root.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved):
        raise ExtensionGateError(f"{label}: path escapes repository: {relative}")
    if required_prefix is not None:
        allowed = (root / required_prefix).resolve(strict=True)
        if not resolved.is_relative_to(allowed):
            raise ExtensionGateError(f"{label}: path escapes {required_prefix}/")
    if not resolved.is_file():
        raise ExtensionGateError(f"{label}: expected a regular file: {relative}")
    if tracked_files is not None and relative not in tracked_files:
        raise ExtensionGateError(f"{label}: file must be Git-tracked: {relative}")
    return relative, resolved


def _inventory(root: Path, directory: str, suffix: str) -> set[str]:
    base = root / directory
    if base.is_symlink():
        raise ExtensionGateError(f"{directory}: symlink forbidden")
    if not base.exists():
        return set()
    if not base.is_dir():
        raise ExtensionGateError(f"{directory}: expected a real directory")

    inventory: set[str] = set()
    for current_text, directories, files in os.walk(base, followlinks=False):
        current = Path(current_text)
        for name in directories:
            path = current / name
            if path.is_symlink():
                raise ExtensionGateError(f"{path.relative_to(root)}: symlink forbidden")
        for name in files:
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ExtensionGateError(f"{relative}: symlink forbidden")
            if path.suffix != suffix:
                raise ExtensionGateError(f"{relative}: unknown extension artifact type")
            inventory.add(relative)
    return inventory


def _validate_schema_files(root: Path) -> None:
    expected = {
        REGISTRY_SCHEMA: "https://riley.invalid/schemas/extension-registry-v1.schema.json",
        PROPOSAL_SCHEMA: "https://riley.invalid/schemas/extension-proposal-v1.schema.json",
        CONTRACT_SCHEMA: "https://riley.invalid/schemas/extension-benchmark-contract-v1.schema.json",
        IMPLEMENTATION_SCHEMA: "https://riley.invalid/schemas/extension-implementation-v1.schema.json",
    }
    for relative, schema_id in expected.items():
        _, resolved = _checked_file(root, relative, relative)
        schema = _object(_load_json(resolved, relative), relative)
        _constant(schema.get("$schema"), "https://json-schema.org/draft/2020-12/schema", f"{relative}.$schema")
        _constant(schema.get("$id"), schema_id, f"{relative}.$id")
        _constant(schema.get("type"), "object", f"{relative}.type")
        try:
            canonical = json.dumps(
                schema,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ExtensionGateError(f"{relative}: cannot canonicalize schema: {error}") from error
        actual_digest = hashlib.sha256(canonical).hexdigest()
        if actual_digest != SCHEMA_SEMANTIC_SHA256[relative]:
            raise ExtensionGateError(
                f"{relative}: closed schema contract digest mismatch"
            )


def _validate_tolerances(value: Any, label: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ExtensionGateError(f"{label}: expected at least one dtype tolerance")
    dtypes: set[str] = set()
    for index, raw in enumerate(value):
        item_label = f"{label}[{index}]"
        item = _closed_object(raw, {"dtype", "atol", "rtol"}, item_label)
        dtype = _string(item["dtype"], f"{item_label}.dtype")
        if dtype in dtypes:
            raise ExtensionGateError(f"{label}: duplicate dtype {dtype!r}")
        dtypes.add(dtype)
        atol = _number(item["atol"], f"{item_label}.atol")
        rtol = _number(item["rtol"], f"{item_label}.rtol")
        if atol >= 1 or rtol >= 1:
            raise ExtensionGateError(
                f"{item_label}: atol and rtol must each be less than 1"
            )
    return dtypes


def _validate_class_gate(
    value: Any,
    semantic_class: str,
    quality_metric: str,
    implementation_boundary: str,
    label: str,
) -> dict[str, Any]:
    keys = CLASS_GATE_KEYS[semantic_class]
    gate = _closed_object(value, keys, label)
    _constant(gate["kind"], semantic_class, f"{label}.kind")

    if semantic_class == "reference":
        for field in (
            "behavioral_parity",
            "token_parity",
            "stable_fallback",
            "lifetime_resource_regression",
        ):
            _constant(gate[field], True, f"{label}.{field}")
    elif semantic_class == "E0":
        _constant(gate["reference_parity"], True, f"{label}.reference_parity")
        _validate_tolerances(gate["dtype_tolerances"], f"{label}.dtype_tolerances")
        _constant(gate["extreme_value_cases"], True, f"{label}.extreme_value_cases")
        _constant(gate["token_level_regression"], True, f"{label}.token_level_regression")
    elif semantic_class == "E1":
        _string(gate["distribution_contract"], f"{label}.distribution_contract")
        _string(gate["statistical_test"], f"{label}.statistical_test")
        _constant(gate["rng_isolation"], True, f"{label}.rng_isolation")
        _constant(gate["rng_snapshot_restore"], True, f"{label}.rng_snapshot_restore")
        _constant(gate["greedy_exact"], True, f"{label}.greedy_exact")
        _string(gate["fixed_seed_definition"], f"{label}.fixed_seed_definition")
    elif semantic_class == "A1":
        budget = _closed_object(gate["error_budget"], {"metric", "unit", "maximum"}, f"{label}.error_budget")
        if _metric(budget["metric"], f"{label}.error_budget.metric") != quality_metric:
            raise ExtensionGateError(f"{label}.error_budget.metric: must equal quality_or_error_metric")
        _constant(budget["unit"], "fraction", f"{label}.error_budget.unit")
        maximum = _number(budget["maximum"], f"{label}.error_budget.maximum")
        if maximum >= 1:
            raise ExtensionGateError(
                f"{label}.error_budget.maximum: fraction must be less than 1"
            )
        for field in ("exact_fallback", "opt_in", "usage_disclosure", "quality_latency_curve"):
            _constant(gate[field], True, f"{label}.{field}")
    elif semantic_class == "M1":
        if implementation_boundary != "research":
            raise ExtensionGateError(f"{label}: M1 requires implementation_boundary='research'")
        _string(gate["research_track"], f"{label}.research_track")
        _string(gate["calibration_or_training_provenance"], f"{label}.calibration_or_training_provenance")
        for field in ("production_core_isolated", "opt_in", "usage_disclosure", "quality_latency_curve"):
            _constant(gate[field], True, f"{label}.{field}")
    return gate


def _validate_approval_answers(
    value: Any,
    reference_path: str,
    label: str,
) -> None:
    answers = _closed_object(value, APPROVAL_ANSWER_KEYS, label)
    for field in (
        "user_workload_bottleneck",
        "semantic_class_rationale",
        "implementation_location_rationale",
        "error_or_distribution_contract",
        "memory_and_operational_complexity",
        "fallback_and_rollback",
        "end_to_end_benefit_hypothesis",
    ):
        _string(answers[field], f"{label}.{field}")

    ir = _closed_object(
        answers["existing_ir_expression"],
        {"disposition", "rationale"},
        f"{label}.existing_ir_expression",
    )
    disposition = _string(
        ir["disposition"], f"{label}.existing_ir_expression.disposition"
    )
    if disposition not in {
        "existing-ir",
        "additive-ir-change",
        "separate-mixer-or-backend",
    }:
        raise ExtensionGateError(
            f"{label}.existing_ir_expression.disposition: unknown value {disposition!r}"
        )
    _string(ir["rationale"], f"{label}.existing_ir_expression.rationale")

    _constant(
        answers["correctness_reference"],
        reference_path,
        f"{label}.correctness_reference",
    )
    reductions = answers["expected_resource_reduction"]
    allowed_reductions = {"flops", "serial-depth", "hbm-traffic"}
    if (
        not isinstance(reductions, list)
        or not reductions
        or any(not isinstance(item, str) for item in reductions)
        or len(reductions) != len(set(reductions))
        or not set(reductions) <= allowed_reductions
    ):
        raise ExtensionGateError(
            f"{label}.expected_resource_reduction: use unique flops, serial-depth, "
            "or hbm-traffic values"
        )


def _validate_proposal(
    root: Path,
    path: Path,
    entry: dict[str, Any],
    tracked_files: set[str],
    label: str,
) -> dict[str, Any]:
    proposal = _closed_object(_load_json(path, label), PROPOSAL_KEYS, label)
    _constant(proposal["$schema"], "../proposal.schema.json", f"{label}.$schema")
    _constant(proposal["schema_version"], "riley.extension-proposal.v1", f"{label}.schema_version")
    extension_id = _identifier(proposal["extension_id"], f"{label}.extension_id")
    if extension_id != entry["extension_id"]:
        raise ExtensionGateError(f"{label}.extension_id: registry mismatch")
    _constant(proposal["status"], entry["status"], f"{label}.status")
    _constant(proposal["track"], entry["track"], f"{label}.track")
    semantic_class = _string(proposal["semantic_class"], f"{label}.semantic_class")
    if semantic_class != entry["semantic_class"]:
        raise ExtensionGateError(f"{label}.semantic_class: registry mismatch")
    _string(proposal["title"], f"{label}.title")
    _string(proposal["problem_statement"], f"{label}.problem_statement")
    boundary = _string(proposal["implementation_boundary"], f"{label}.implementation_boundary")
    if boundary not in {"core", "backend", "plugin", "research"}:
        raise ExtensionGateError(f"{label}.implementation_boundary: unknown value {boundary!r}")
    reference_path, reference_file = _checked_file(
        root,
        proposal["reference_path"],
        f"{label}.reference_path",
        tracked_files=tracked_files,
    )
    reference_sha256 = _sha256(
        proposal["reference_sha256"], f"{label}.reference_sha256"
    )
    if _file_sha256(reference_file) != reference_sha256:
        raise ExtensionGateError(f"{label}.reference_sha256: file content mismatch")
    fallback_path, fallback_file = _checked_file(
        root,
        proposal["fallback_path"],
        f"{label}.fallback_path",
        tracked_files=tracked_files,
    )
    fallback_sha256 = _sha256(
        proposal["fallback_sha256"], f"{label}.fallback_sha256"
    )
    if _file_sha256(fallback_file) != fallback_sha256:
        raise ExtensionGateError(f"{label}.fallback_sha256: file content mismatch")
    if fallback_path == reference_path:
        raise ExtensionGateError(
            f"{label}: reference_path and fallback_path must be different files"
        )
    primary_metric = _metric(proposal["primary_metric"], f"{label}.primary_metric")
    if primary_metric not in PERFORMANCE_METRIC_PATHS:
        raise ExtensionGateError(
            f"{label}.primary_metric: expected a performance/resource result path"
        )
    allowed_primary = TRACK_PRIMARY_METRICS[proposal["track"]]
    if primary_metric not in allowed_primary:
        allowed = ", ".join(sorted(allowed_primary))
        raise ExtensionGateError(
            f"{label}.primary_metric: {proposal['track']} requires one of {allowed}"
        )
    raw_required_metrics = proposal["required_metrics"]
    if (
        not isinstance(raw_required_metrics, list)
        or not raw_required_metrics
        or any(not isinstance(item, str) for item in raw_required_metrics)
        or raw_required_metrics != sorted(set(raw_required_metrics))
    ):
        raise ExtensionGateError(
            f"{label}.required_metrics: expected a sorted unique non-empty list"
        )
    required_metrics = {
        _metric(item, f"{label}.required_metrics[{index}]")
        for index, item in enumerate(raw_required_metrics)
    }
    expected_required_metrics = TRACK_REQUIRED_METRICS[proposal["track"]]
    if required_metrics != expected_required_metrics:
        raise ExtensionGateError(
            f"{label}.required_metrics: must equal the closed {proposal['track']} metric set"
        )
    if primary_metric not in required_metrics:
        raise ExtensionGateError(
            f"{label}.primary_metric: must also appear in required_metrics"
        )
    quality_metric = _metric(proposal["quality_or_error_metric"], f"{label}.quality_or_error_metric")
    if quality_metric == primary_metric:
        raise ExtensionGateError(
            f"{label}: primary_metric and quality_or_error_metric must differ"
        )
    _validate_class_gate(
        proposal["class_gate"],
        semantic_class,
        quality_metric,
        boundary,
        f"{label}.class_gate",
    )
    allowed_quality = TRACK_CLASS_QUALITY_METRICS.get(
        (proposal["track"], semantic_class), set()
    )
    if not allowed_quality:
        raise ExtensionGateError(
            f"{label}.quality_or_error_metric: {proposal['track']}/{semantic_class} "
            "requires a future common result schema version"
        )
    if quality_metric not in allowed_quality:
        allowed = ", ".join(sorted(allowed_quality))
        raise ExtensionGateError(
            f"{label}.quality_or_error_metric: expected one of {allowed}"
        )
    runtime_flag = _string(proposal["runtime_flag"], f"{label}.runtime_flag")
    if not RUNTIME_FLAG.fullmatch(runtime_flag):
        raise ExtensionGateError(f"{label}.runtime_flag: must use RILEY_EXPERIMENTAL_* namespace")
    _constant(proposal["default_enabled"], False, f"{label}.default_enabled")
    _constant(proposal["stable_default"], False, f"{label}.stable_default")
    _string(proposal["result_disclosure"], f"{label}.result_disclosure")
    _string(proposal["rollback"], f"{label}.rollback")
    _constant(proposal["deploy_document_path"], entry["deploy_document_path"], f"{label}.deploy_document_path")
    _constant(proposal["benchmark_contract_path"], entry["benchmark_contract_path"], f"{label}.benchmark_contract_path")
    _validate_approval_answers(
        proposal["approval_answers"], reference_path, f"{label}.approval_answers"
    )
    return proposal


def _validate_measurement(value: Any, label: str) -> None:
    keys = {
        "independent_process_runs",
        "measured_iterations_per_run",
        "required_statistics",
        "end_to_end",
        "fallback_comparison",
        "environment_dimensions",
    }
    measurement = _closed_object(value, keys, label)
    for field in ("independent_process_runs", "measured_iterations_per_run"):
        count = measurement[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 5:
            raise ExtensionGateError(f"{label}.{field}: expected integer >= 5")
    statistics = measurement["required_statistics"]
    if not isinstance(statistics, list) or len(statistics) != len(set(statistics)):
        raise ExtensionGateError(f"{label}.required_statistics: expected unique list")
    if not {"median", "p95"}.issubset(statistics) or not set(statistics) <= {"median", "p95", "p99"}:
        raise ExtensionGateError(f"{label}.required_statistics: median and p95 are mandatory")
    _constant(measurement["end_to_end"], True, f"{label}.end_to_end")
    _constant(measurement["fallback_comparison"], True, f"{label}.fallback_comparison")
    dimensions = measurement["environment_dimensions"]
    expected = {
        "gpu",
        "driver",
        "cuda",
        "model_id",
        "model_revision",
        "dtype",
        "concurrency",
        "prompt_output_lengths",
        "sampling",
        "warm_state",
    }
    if not isinstance(dimensions, list) or len(dimensions) != len(set(dimensions)) or set(dimensions) != expected:
        raise ExtensionGateError(f"{label}.environment_dimensions: closed environment set required")


def _validate_positive_integer_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ExtensionGateError(f"{label}: expected unique positive integers")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in value
    ):
        raise ExtensionGateError(f"{label}: expected unique positive integers")
    if len(value) != len(set(value)):
        raise ExtensionGateError(f"{label}: expected unique positive integers")


def _validate_comparison_environment(value: Any, label: str) -> dict[str, Any]:
    keys = {
        "gpu_count",
        "gpu",
        "driver",
        "cuda",
        "model_id",
        "model_revision",
        "dtype",
        "concurrency",
        "prompt_tokens",
        "output_tokens",
        "sampling_configs",
        "warm_states",
    }
    environment = _closed_object(value, keys, label)
    _constant(environment["gpu_count"], 1, f"{label}.gpu_count")
    for field in ("gpu", "driver", "cuda", "model_id", "model_revision", "dtype"):
        _string(environment[field], f"{label}.{field}")
    if not MODEL_REVISION.fullmatch(environment["model_revision"]):
        raise ExtensionGateError(
            f"{label}.model_revision: expected a pinned 40-character lowercase revision"
        )
    for field in ("concurrency", "prompt_tokens", "output_tokens"):
        _validate_positive_integer_list(environment[field], f"{label}.{field}")
    warm_states = environment["warm_states"]
    if (
        not isinstance(warm_states, list)
        or len(warm_states) != len(set(warm_states))
        or set(warm_states) != {"cold", "warm"}
    ):
        raise ExtensionGateError(f"{label}.warm_states: cold and warm are mandatory")

    sampling_configs = environment["sampling_configs"]
    if not isinstance(sampling_configs, list) or not sampling_configs:
        raise ExtensionGateError(f"{label}.sampling_configs: expected non-empty list")
    sampling_ids: set[str] = set()
    for index, raw in enumerate(sampling_configs):
        sampling_label = f"{label}.sampling_configs[{index}]"
        sampling = _closed_object(
            raw,
            {
                "id",
                "strategy",
                "temperature",
                "top_p",
                "top_k",
                "seed",
                "ignore_eos",
                "fixed_output_length",
            },
            sampling_label,
        )
        sampling_id = _string(sampling["id"], f"{sampling_label}.id")
        if sampling_id in sampling_ids:
            raise ExtensionGateError(f"{label}.sampling_configs: duplicate id")
        sampling_ids.add(sampling_id)
        strategy = _string(sampling["strategy"], f"{sampling_label}.strategy")
        if strategy not in {"greedy", "sampling"}:
            raise ExtensionGateError(f"{sampling_label}.strategy: unknown strategy")
        for field in ("temperature", "top_p"):
            item = sampling[field]
            if item is not None:
                number = _number(item, f"{sampling_label}.{field}")
                if field == "top_p" and (number <= 0 or number > 1):
                    raise ExtensionGateError(
                        f"{sampling_label}.top_p: expected 0 < top_p <= 1"
                    )
        top_k = sampling["top_k"]
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ExtensionGateError(f"{sampling_label}.top_k: expected null or integer >= 1")
        seed = sampling["seed"]
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ExtensionGateError(f"{sampling_label}.seed: expected null or integer >= 0")
        for field in ("ignore_eos", "fixed_output_length"):
            if not isinstance(sampling[field], bool):
                raise ExtensionGateError(f"{sampling_label}.{field}: expected boolean")
        if strategy == "greedy":
            for field in ("temperature", "top_p", "top_k", "seed"):
                _constant(sampling[field], None, f"{sampling_label}.{field}")
        else:
            if sampling["temperature"] is None or sampling["temperature"] <= 0:
                raise ExtensionGateError(
                    f"{sampling_label}.temperature: sampling requires a positive value"
                )
            if sampling["seed"] is None:
                raise ExtensionGateError(f"{sampling_label}.seed: sampling requires a fixed seed")
    return environment


def _validate_contract(
    root: Path,
    path: Path,
    entry: dict[str, Any],
    proposal: dict[str, Any],
    tracked_files: set[str],
    label: str,
) -> dict[str, Any]:
    contract = _closed_object(_load_json(path, label), CONTRACT_KEYS, label)
    _constant(contract["$schema"], "../benchmark-contract.schema.json", f"{label}.$schema")
    _constant(contract["schema_version"], "riley.extension-benchmark-contract.v1", f"{label}.schema_version")
    _validate_class_gate(
        contract["class_gate"],
        proposal["semantic_class"],
        proposal["quality_or_error_metric"],
        proposal["implementation_boundary"],
        f"{label}.class_gate",
    )
    cross_fields = (
        "extension_id",
        "status",
        "track",
        "semantic_class",
        "deploy_document_path",
        "reference_path",
        "reference_sha256",
        "fallback_path",
        "fallback_sha256",
        "runtime_flag",
        "primary_metric",
        "required_metrics",
        "quality_or_error_metric",
        "class_gate",
    )
    for field in cross_fields:
        if not _same_json_value(contract[field], proposal[field]):
            raise ExtensionGateError(f"{label}.{field}: proposal mismatch")
    _constant(contract["proposal_path"], entry["proposal_path"], f"{label}.proposal_path")
    workloads = contract["workloads"]
    if not isinstance(workloads, list) or not workloads:
        raise ExtensionGateError(f"{label}.workloads: expected a non-empty list")
    workload_paths: set[str] = set()
    for index, raw in enumerate(workloads):
        workload_label = f"{label}.workloads[{index}]"
        workload = _closed_object(raw, {"path", "sha256"}, workload_label)
        relative, workload_file = _checked_file(
            root,
            workload["path"],
            f"{workload_label}.path",
            tracked_files=tracked_files,
        )
        if relative in workload_paths:
            raise ExtensionGateError(f"{label}.workloads: duplicate workload path")
        workload_paths.add(relative)
        expected_sha256 = _sha256(
            workload["sha256"], f"{workload_label}.sha256"
        )
        if _file_sha256(workload_file) != expected_sha256:
            raise ExtensionGateError(f"{workload_label}.sha256: file content mismatch")
    environment = _validate_comparison_environment(
        contract["comparison_environment"], f"{label}.comparison_environment"
    )
    if proposal["semantic_class"] == "E0":
        tolerance_dtypes = _validate_tolerances(
            contract["class_gate"]["dtype_tolerances"],
            f"{label}.class_gate.dtype_tolerances",
        )
        if tolerance_dtypes != {environment["dtype"]}:
            raise ExtensionGateError(
                f"{label}.class_gate.dtype_tolerances: must match comparison_environment.dtype exactly"
            )
    _validate_measurement(contract["measurement"], f"{label}.measurement")
    return contract


def _validate_implementation(
    root: Path,
    path: Path,
    entry: dict[str, Any],
    proposal: dict[str, Any],
    tracked_files: set[str],
    label: str,
) -> None:
    implementation = _closed_object(
        _load_json(path, label), IMPLEMENTATION_KEYS, label
    )
    _constant(
        implementation["$schema"],
        "../implementation.schema.json",
        f"{label}.$schema",
    )
    _constant(
        implementation["schema_version"],
        "riley.extension-implementation.v1",
        f"{label}.schema_version",
    )
    _constant(
        implementation["status"],
        "experimental-implementation",
        f"{label}.status",
    )
    cross_values = {
        "extension_id": entry["extension_id"],
        "proposal_path": entry["proposal_path"],
        "deploy_document_path": entry["deploy_document_path"],
        "benchmark_contract_path": entry["benchmark_contract_path"],
        "runtime_flag": proposal["runtime_flag"],
    }
    for field, expected in cross_values.items():
        _constant(implementation[field], expected, f"{label}.{field}")
    _constant(implementation["default_enabled"], False, f"{label}.default_enabled")
    _constant(implementation["stable_default"], False, f"{label}.stable_default")

    raw_paths = implementation["implementation_paths"]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ExtensionGateError(f"{label}.implementation_paths: expected non-empty list")
    implementation_paths: dict[str, Path] = {}
    for index, raw_path in enumerate(raw_paths):
        relative, resolved = _checked_file(
            root,
            raw_path,
            f"{label}.implementation_paths[{index}]",
            tracked_files=tracked_files,
        )
        if relative in implementation_paths:
            raise ExtensionGateError(
                f"{label}.implementation_paths: duplicate source path"
            )
        if relative.startswith(f"{IMPLEMENTATION_ROOT}/") or relative.startswith(
            "deploy/extensions/"
        ) or relative.startswith("benchmarks/extensions/"):
            raise ExtensionGateError(
                f"{label}.implementation_paths[{index}]: metadata is not implementation source"
            )
        implementation_paths[relative] = resolved

    flag_source, flag_source_file = _checked_file(
        root,
        implementation["runtime_flag_source_path"],
        f"{label}.runtime_flag_source_path",
        tracked_files=tracked_files,
    )
    if flag_source not in implementation_paths:
        raise ExtensionGateError(
            f"{label}.runtime_flag_source_path: must also appear in implementation_paths"
        )
    try:
        flag_source_bytes = flag_source_file.read_bytes()
    except OSError as error:
        raise ExtensionGateError(
            f"{label}.runtime_flag_source_path: cannot read source: {error}"
        ) from error
    if proposal["runtime_flag"].encode("ascii") not in flag_source_bytes:
        raise ExtensionGateError(
            f"{label}.runtime_flag_source_path: declared runtime flag literal is absent"
        )

    _, workspace_manifest = _checked_file(
        root, "Cargo.toml", f"{label}.validation_tests.workspace", tracked_files=tracked_files
    )
    try:
        with workspace_manifest.open("rb") as source:
            workspace_document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExtensionGateError(f"{label}.validation_tests: invalid workspace Cargo.toml") from error
    workspace = workspace_document.get("workspace")
    members = workspace.get("members") if isinstance(workspace, dict) else None
    if not isinstance(members, list) or any(not isinstance(item, str) for item in members):
        raise ExtensionGateError(f"{label}.validation_tests: workspace members are unavailable")
    workspace_members = set(members)

    raw_tests = implementation["validation_tests"]
    if not isinstance(raw_tests, list) or not raw_tests:
        raise ExtensionGateError(f"{label}.validation_tests: expected non-empty list")
    test_paths: set[str] = set()
    test_ids: set[str] = set()
    for index, raw_test in enumerate(raw_tests):
        test_label = f"{label}.validation_tests[{index}]"
        test = _closed_object(raw_test, {"path", "sha256", "test_id"}, test_label)
        test_path, test_file = _checked_file(
            root,
            test["path"],
            f"{test_label}.path",
            tracked_files=tracked_files,
        )
        if not VALIDATION_TEST_PATH.fullmatch(test_path):
            raise ExtensionGateError(
                f"{test_label}.path: test must be an auto-discovered Rust integration test"
            )
        crate_root = "/".join(test_path.split("/")[:2])
        if crate_root not in workspace_members:
            raise ExtensionGateError(
                f"{test_label}.path: crate is not an explicit Cargo workspace member"
            )
        _, crate_manifest_file = _checked_file(
            root,
            f"{crate_root}/Cargo.toml",
            f"{test_label}.crate_manifest",
            tracked_files=tracked_files,
        )
        try:
            with crate_manifest_file.open("rb") as source:
                crate_document = tomllib.load(source)
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ExtensionGateError(
                f"{test_label}.crate_manifest: invalid Cargo.toml"
            ) from error
        package = crate_document.get("package")
        if not isinstance(package, dict):
            raise ExtensionGateError(
                f"{test_label}.crate_manifest: package table is required"
            )
        if package.get("autotests", True) is not True:
            raise ExtensionGateError(
                f"{test_label}.crate_manifest: autotests must remain enabled"
            )
        crate_relative_test = test_path.removeprefix(f"{crate_root}/")
        explicit_targets = crate_document.get("test", [])
        if not isinstance(explicit_targets, list):
            raise ExtensionGateError(
                f"{test_label}.crate_manifest: invalid explicit test targets"
            )
        for target_index, target in enumerate(explicit_targets):
            if not isinstance(target, dict):
                raise ExtensionGateError(
                    f"{test_label}.crate_manifest.test[{target_index}]: expected table"
                )
            target_path = target.get("path")
            target_name = target.get("name")
            if target_path is None and isinstance(target_name, str):
                target_path = f"tests/{target_name}.rs"
            elif target_path is not None:
                if (
                    not isinstance(target_path, str)
                    or "\\" in target_path
                    or target_path.startswith("/")
                    or any(part in {"", ".."} for part in target_path.split("/"))
                ):
                    raise ExtensionGateError(
                        f"{test_label}.crate_manifest.test[{target_index}].path: "
                        "expected a canonical crate-relative path"
                    )
                target_path = PurePosixPath(target_path).as_posix()
            if target_path != crate_relative_test:
                continue
            if target.get("test", True) is not True or target.get("harness", True) is not True:
                raise ExtensionGateError(
                    f"{test_label}.crate_manifest: validation target must use the test harness"
                )
            required_features = target.get("required-features", [])
            if required_features not in (None, []):
                raise ExtensionGateError(
                    f"{test_label}.crate_manifest: validation target cannot require features"
                )
        if test_path not in implementation_paths:
            raise ExtensionGateError(
                f"{test_label}.path: must also appear in implementation_paths"
            )
        if test_path in test_paths:
            raise ExtensionGateError(f"{label}.validation_tests: duplicate path")
        test_paths.add(test_path)
        test_id = _string(test["test_id"], f"{test_label}.test_id")
        if not RUST_TEST_ID.fullmatch(test_id):
            raise ExtensionGateError(
                f"{test_label}.test_id: expected a Rust test identifier"
            )
        if test_id in test_ids:
            raise ExtensionGateError(f"{label}.validation_tests: duplicate test_id")
        test_ids.add(test_id)
        expected_sha256 = _sha256(test["sha256"], f"{test_label}.sha256")
        if _file_sha256(test_file) != expected_sha256:
            raise ExtensionGateError(f"{test_label}.sha256: file content mismatch")
        try:
            test_bytes = test_file.read_bytes()
        except OSError as error:
            raise ExtensionGateError(f"{test_label}.path: cannot read test: {error}") from error
        _validate_registered_rust_test(test_bytes, test_id, test_label)


def _validate_transition(
    root: Path,
    base_revision: str,
    entries: list[dict[str, Any]],
    runtime_flags_by_id: dict[str, str],
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", base_revision):
        raise ExtensionGateError("--base-revision must be a full lowercase Git SHA-1")
    _git_bytes(root, ["cat-file", "-e", f"{base_revision}^{{commit}}"], "transition base")
    base_registry_bytes = _git_show_optional(root, base_revision, REGISTRY_PATH)
    if base_registry_bytes is None:
        if entries:
            raise ExtensionGateError(
                "transition bootstrap: the first registry revision must remain empty"
            )
        return

    base_registry = _closed_object(
        _load_json_bytes(base_registry_bytes, f"{base_revision}:{REGISTRY_PATH}"),
        REGISTRY_KEYS,
        f"{base_revision}:{REGISTRY_PATH}",
    )
    base_raw_entries = base_registry["extensions"]
    if not isinstance(base_raw_entries, list):
        raise ExtensionGateError("transition base registry extensions must be a list")
    base_entries: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(base_raw_entries):
        label = f"transition base registry.extensions[{index}]"
        entry = _closed_object(raw, ENTRY_KEYS, label)
        extension_id = _identifier(entry["extension_id"], f"{label}.extension_id")
        if extension_id in base_entries:
            raise ExtensionGateError("transition base registry has duplicate IDs")
        base_entries[extension_id] = entry
    current_entries = {entry["extension_id"]: entry for entry in entries}

    removed = sorted(base_entries.keys() - current_entries.keys())
    if removed:
        raise ExtensionGateError(
            f"transition: v1 registry entries are append-only; removed={removed}"
        )
    linked = sorted(
        extension_id
        for extension_id in base_entries.keys() & current_entries.keys()
        if base_entries[extension_id]["implementation_link_path"] is None
        and current_entries[extension_id]["implementation_link_path"] is not None
    )
    if len(linked) > 1:
        raise ExtensionGateError(
            "transition: one implementation PR may link only one extension; "
            f"linked={linked}"
        )
    new_runtime_flags = _current_runtime_flags(root) - _base_runtime_flags(
        root, base_revision
    )
    if linked:
        expected_flag = runtime_flags_by_id[linked[0]]
        if new_runtime_flags != {expected_flag}:
            raise ExtensionGateError(
                "transition: an implementation link must introduce exactly its approved "
                f"runtime flag; expected={expected_flag!r}, new={sorted(new_runtime_flags)}"
            )
    elif new_runtime_flags:
        raise ExtensionGateError(
            "transition: new experimental runtime flags require exactly one approved "
            f"implementation link; new={sorted(new_runtime_flags)}"
        )
    immutable_entry_fields = ENTRY_KEYS - {"implementation_link_path"}
    for extension_id in sorted(base_entries.keys() & current_entries.keys()):
        base_entry = base_entries[extension_id]
        current_entry = current_entries[extension_id]
        for field in immutable_entry_fields:
            if not _same_json_value(base_entry[field], current_entry[field]):
                raise ExtensionGateError(
                    f"transition: existing {extension_id}.{field} is immutable in v1"
                )
        old_link = base_entry["implementation_link_path"]
        new_link = current_entry["implementation_link_path"]
        if old_link is not None and old_link != new_link:
            raise ExtensionGateError(
                f"transition: existing {extension_id} implementation link cannot change or be removed"
            )
        if old_link is not None:
            base_link_bytes = _git_show_optional(root, base_revision, old_link)
            if base_link_bytes is None:
                raise ExtensionGateError(
                    f"transition: base implementation artifact is missing for "
                    f"{extension_id}: {old_link}"
                )
            try:
                current_link_bytes = (root / old_link).read_bytes()
            except OSError as error:
                raise ExtensionGateError(
                    f"transition: cannot read current implementation artifact "
                    f"{old_link}: {error}"
                ) from error
            if current_link_bytes != base_link_bytes:
                raise ExtensionGateError(
                    f"transition: linked implementation artifact is immutable: {old_link}"
                )
        for field in (
            "proposal_path",
            "deploy_document_path",
            "benchmark_contract_path",
        ):
            relative = base_entry[field]
            base_bytes = _git_show_optional(root, base_revision, relative)
            if base_bytes is None:
                raise ExtensionGateError(
                    f"transition: base artifact is missing for {extension_id}: {relative}"
                )
            try:
                current_bytes = (root / relative).read_bytes()
            except OSError as error:
                raise ExtensionGateError(
                    f"transition: cannot read current artifact {relative}: {error}"
                ) from error
            if current_bytes != base_bytes:
                raise ExtensionGateError(
                    f"transition: admitted {extension_id} artifact is immutable: {relative}"
                )

    added = sorted(current_entries.keys() - base_entries.keys())
    if len(added) > 1:
        raise ExtensionGateError(
            f"transition: one admission PR may add only one extension; added={added}"
        )
    if added:
        extension_id = added[0]
        entry = current_entries[extension_id]
        _constant(
            entry["implementation_link_path"],
            None,
            f"transition {extension_id}.implementation_link_path",
        )
        changed_raw = _git_bytes(
            root,
            ["diff", "--no-renames", "--name-only", "-z", base_revision, "--"],
            "transition diff",
        )
        try:
            changed = {
                item
                for item in changed_raw.decode("utf-8").split("\0")
                if item
            }
        except UnicodeError as error:
            raise ExtensionGateError("transition diff contains a non-UTF-8 path") from error
        expected = {
            REGISTRY_PATH,
            entry["proposal_path"],
            entry["deploy_document_path"],
            entry["benchmark_contract_path"],
        }
        if changed != expected:
            raise ExtensionGateError(
                "transition: admission-only PR must change exactly registry, proposal, "
                f"plan, and benchmark contract; expected={sorted(expected)}, "
                f"changed={sorted(changed)}"
            )


def validate_repository(root: Path, base_revision: str | None = None) -> int:
    """Validate extension admission metadata and return the approved count."""
    root = root.resolve(strict=True)
    _validate_schema_files(root)
    _, registry_file = _checked_file(root, REGISTRY_PATH, REGISTRY_PATH)
    registry = _closed_object(_load_json(registry_file, REGISTRY_PATH), REGISTRY_KEYS, REGISTRY_PATH)
    _constant(registry["$schema"], "registry.schema.json", f"{REGISTRY_PATH}.$schema")
    _constant(registry["schema_version"], "riley.extension-registry.v1", f"{REGISTRY_PATH}.schema_version")
    raw_entries = registry["extensions"]
    if not isinstance(raw_entries, list):
        raise ExtensionGateError(f"{REGISTRY_PATH}.extensions: expected a list")

    entries: list[dict[str, Any]] = []
    ids: set[str] = set()
    used_paths: set[str] = set()
    runtime_flags: set[str] = set()
    runtime_flags_by_id: dict[str, str] = {}
    proposal_paths: set[str] = set()
    plan_paths: set[str] = set()
    contract_paths: set[str] = set()
    implementation_paths: set[str] = set()

    for index, raw in enumerate(raw_entries):
        label = f"{REGISTRY_PATH}.extensions[{index}]"
        entry = _closed_object(raw, ENTRY_KEYS, label)
        extension_id = _identifier(entry["extension_id"], f"{label}.extension_id")
        if extension_id in ids:
            raise ExtensionGateError(f"{label}: duplicate extension_id {extension_id!r}")
        ids.add(extension_id)
        _constant(
            entry["status"],
            "approved-for-implementation",
            f"{label}.status",
        )
        track = _string(entry["track"], f"{label}.track")
        if track not in EXTENSION_TRACKS:
            raise ExtensionGateError(f"{label}.track: unknown value {track!r}")
        semantic_class = _string(entry["semantic_class"], f"{label}.semantic_class")
        if semantic_class not in SEMANTIC_CLASSES:
            raise ExtensionGateError(f"{label}.semantic_class: unknown value {semantic_class!r}")
        if semantic_class not in TRACK_SEMANTIC_CLASSES[track]:
            allowed = ", ".join(sorted(TRACK_SEMANTIC_CLASSES[track]))
            raise ExtensionGateError(
                f"{label}.semantic_class: track {track!r} requires one of {allowed}"
            )

        specs = (
            ("proposal_path", PROPOSAL_ROOT, ".json", proposal_paths),
            ("deploy_document_path", PLAN_ROOT, ".md", plan_paths),
            ("benchmark_contract_path", CONTRACT_ROOT, ".json", contract_paths),
        )
        for field, prefix, suffix, inventory in specs:
            relative, _ = _checked_file(root, entry[field], f"{label}.{field}", required_prefix=prefix, required_suffix=suffix)
            if Path(relative).stem != extension_id:
                raise ExtensionGateError(f"{label}.{field}: filename must be {extension_id}{suffix}")
            if relative in used_paths:
                raise ExtensionGateError(f"{label}.{field}: duplicate registered path {relative}")
            used_paths.add(relative)
            inventory.add(relative)
        implementation_link = entry["implementation_link_path"]
        if implementation_link is not None:
            relative, _ = _checked_file(
                root,
                implementation_link,
                f"{label}.implementation_link_path",
                required_prefix=IMPLEMENTATION_ROOT,
                required_suffix=".json",
            )
            if Path(relative).stem != extension_id:
                raise ExtensionGateError(
                    f"{label}.implementation_link_path: filename must be {extension_id}.json"
                )
            if relative in used_paths:
                raise ExtensionGateError(
                    f"{label}.implementation_link_path: duplicate registered path {relative}"
                )
            used_paths.add(relative)
            implementation_paths.add(relative)
        entry = dict(entry)
        entry["extension_id"] = extension_id
        entry["track"] = track
        entry["semantic_class"] = semantic_class
        entries.append(entry)

    if [entry["extension_id"] for entry in entries] != sorted(ids):
        raise ExtensionGateError(f"{REGISTRY_PATH}.extensions: entries must be sorted by extension_id")

    expected_inventories = (
        (PROPOSAL_ROOT, ".json", proposal_paths),
        (PLAN_ROOT, ".md", plan_paths),
        (CONTRACT_ROOT, ".json", contract_paths),
        (IMPLEMENTATION_ROOT, ".json", implementation_paths),
    )
    for directory, suffix, registered in expected_inventories:
        actual = _inventory(root, directory, suffix)
        if actual != registered:
            missing = sorted(registered - actual)
            unknown = sorted(actual - registered)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unregistered={unknown}")
            raise ExtensionGateError(f"{directory}: registry completeness failure ({'; '.join(details)})")

    tracked_files = _git_tracked_files(root) if entries else set()
    for entry in entries:
        extension_id = entry["extension_id"]
        _, proposal_file = _checked_file(root, entry["proposal_path"], entry["proposal_path"], required_prefix=PROPOSAL_ROOT, required_suffix=".json")
        proposal = _validate_proposal(
            root, proposal_file, entry, tracked_files, entry["proposal_path"]
        )
        flag = proposal["runtime_flag"]
        if flag in runtime_flags:
            raise ExtensionGateError(f"{entry['proposal_path']}.runtime_flag: duplicate flag {flag}")
        runtime_flags.add(flag)
        runtime_flags_by_id[extension_id] = flag
        _, contract_file = _checked_file(root, entry["benchmark_contract_path"], entry["benchmark_contract_path"], required_prefix=CONTRACT_ROOT, required_suffix=".json")
        _validate_contract(
            root,
            contract_file,
            entry,
            proposal,
            tracked_files,
            entry["benchmark_contract_path"],
        )
        if extension_id != Path(entry["deploy_document_path"]).stem:
            raise ExtensionGateError(f"{entry['deploy_document_path']}: plan filename mismatch")
        implementation_link = entry["implementation_link_path"]
        if implementation_link is not None:
            _, implementation_file = _checked_file(
                root,
                implementation_link,
                implementation_link,
                required_prefix=IMPLEMENTATION_ROOT,
                required_suffix=".json",
            )
            _validate_implementation(
                root,
                implementation_file,
                entry,
                proposal,
                tracked_files,
                implementation_link,
            )

    if base_revision:
        _validate_transition(root, base_revision, entries, runtime_flags_by_id)
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of ci/)",
    )
    parser.add_argument(
        "--base-revision",
        default=None,
        help="optional full base commit SHA for append-only PR transition checks",
    )
    arguments = parser.parse_args(argv)
    try:
        count = validate_repository(arguments.root, arguments.base_revision or None)
    except ExtensionGateError as error:
        print(f"extension gate failed: {error}", file=sys.stderr)
        return 1
    print(f"extension gate passed: {count} approved extension(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
