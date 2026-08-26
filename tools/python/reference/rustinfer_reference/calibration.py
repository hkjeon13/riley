"""Versioned FP32 numeric and BF16 semantic calibration contracts.

Manifests intentionally contain no producer-computed error scalars. A gate
report is accepted only when this module re-opens the bound safetensors files
and replays the comparison, so a self-consistent forged JSON report is not
correctness evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .constants import (
    ATTENTION_BACKEND,
    MAX_CONTEXT_TOKENS,
    MODEL_CONFIG_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PRIMARY_ENVIRONMENT_ID,
    PYTHON_EXECUTABLE_SHA256,
    PYTHON_PLATFORM_MACHINE,
    PYTHON_PLATFORM_SYSTEM,
    PYTHON_VERSION,
    PYTHON_VERSION_FILE_SHA256,
    RUNTIME_DEPENDENCY_CLASS,
    SAFETENSORS_VERSION,
    TORCH_VERSION,
    TOKENIZER_ARTIFACT_FILENAMES,
    TOKENIZER_FILES_SHA256,
    TOKENIZER_SHA256,
    TRANSFORMERS_VERSION,
)
from .environment import (
    EnvironmentContractError,
    environment_comparability_signature,
    validate_environment_snapshot,
)

CALIBRATION_SCHEMA_VERSION = "1.0.0"
CALIBRATION_GATE_ID = "smollm2-fp32-bf16-native-e0-v2"
FP32_ORACLE_KIND = "fp32-numeric-oracle"
BF16_ORACLE_KIND = "bf16-semantic-oracle"
CANDIDATE_KIND = "candidate"
CALIBRATION_KINDS = {FP32_ORACLE_KIND, BF16_ORACLE_KIND, CANDIDATE_KIND}
SEMANTIC_GENERATION_STEPS = 32
CROSS_CACHE_EXACT_WINDOW = 16
CALIBRATION_TOP_K = 10
CALIBRATION_PROMPT_COUNT = 31
LOG_PROB_PIPELINE = "log-softmax-fp32-v1"
MODEL_EOS_TOKEN_IDS = (0,)
NATIVE_EXECUTABLE_FILENAME = "rustinfer-native"
NATIVE_ENGINE_REVISION = "rustinfer-native-contract-v2"
NATIVE_BUILD_ARGV = (
    "cargo",
    "build",
    "--locked",
    "--release",
    "--package",
    "rustinfer-native",
    "--no-default-features",
    "--features",
    "cuda",
    "--bin",
    NATIVE_EXECUTABLE_FILENAME,
)
_NUMERIC_METRIC_CHUNK_ELEMENTS = 262_144
_F32_STRUCT = struct.Struct("<f")

HF_ORACLE_REDUCTION_VARIANT: dict[str, object] = {
    "variant_id": "hf-eager-default-v1",
    "partition_kind": "hf-eager-runtime-default",
    "chunk_elements": None,
    "remainder_policy": "runtime-default",
    "merge_order": "runtime-default",
}
CANONICAL_CANDIDATE_REDUCTION_VARIANT: dict[str, object] = {
    "variant_id": "canonical-v1",
    "partition_kind": "production-default",
    "chunk_elements": None,
    "remainder_policy": "implementation-default",
    "merge_order": "implementation-default",
}
ALTERNATE_CANDIDATE_REDUCTION_VARIANT: dict[str, object] = {
    "variant_id": "fixed-contiguous-37-balanced-v1",
    "partition_kind": "fixed-contiguous",
    "chunk_elements": 37,
    "remainder_policy": "last-short-chunk",
    "merge_order": "deterministic-balanced-binary-tree-by-chunk-index",
}
REQUIRED_CANDIDATE_REDUCTION_VARIANTS = (
    CANONICAL_CANDIDATE_REDUCTION_VARIANT,
    ALTERNATE_CANDIDATE_REDUCTION_VARIANT,
)

# Version 2 was predeclared once from the reviewed, failing full-31 v1 report by
# applying uniform 15% outward headroom to every observed aggregate metric.
# It becomes usable for E0 only after an independent, replay-validated passing
# full-31 v2 HF oracle report; later data-dependent adjustment requires a bump.
CALIBRATION_THRESHOLDS: dict[str, dict[str, float]] = {
    "first_layer_hidden": {
        "max_abs_max": 0.3884272575378418,
        "mean_abs_max": 0.008509292567237658,
        "max_relative_max": 0.13578447438776492,
        "mean_relative_max": 0.005414661057131772,
        "cosine_min": 0.999983706829855,
    },
    "final_logits": {
        "max_abs_max": 5.852936458587647,
        "mean_abs_max": 1.151280319263363,
        "max_relative_max": 1.1707394897937775,
        "mean_relative_max": 0.13616598220459955,
        "cosine_min": 0.9979035305495393,
    },
    "final_log_probs": {
        "max_abs_max": 4.998420619964599,
        "mean_abs_max": 0.6007178144163239,
        "max_relative_max": 0.5767027348279953,
        "mean_relative_max": 0.04668832837569344,
        "cosine_min": 0.9987779663298298,
    },
}

TENSOR_NAMES = ("first_layer_hidden", "final_logits", "final_log_probs")
HF_SOURCE_PATHS = {
    "matrix": "benchmarks/matrix.yaml",
    "prompts": "benchmarks/prompts.jsonl",
    "gate_manifest": "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json",
    "dependency_lock": "tools/python/reference/uv.lock",
    "python_version_file": "tools/python/reference/.python-version",
    "lane_manifest": "benchmarks/lanes/hf-transformers.json",
    "environment": "benchmarks/environment.md",
    "environment_probe": "tools/python/reference/rustinfer_reference/environment.py",
}
NATIVE_SOURCE_PATHS = {
    "matrix": "benchmarks/matrix.yaml",
    "prompts": "benchmarks/prompts.jsonl",
    "gate_manifest": "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json",
    "dependency_lock": "Cargo.lock",
    "python_version_file": "tools/python/reference/.python-version",
    "lane_manifest": "benchmarks/lanes/rustinfer-native.json",
    "environment": "benchmarks/environment.md",
    "environment_probe": "tools/python/reference/rustinfer_reference/environment.py",
}
SOURCE_NAMES = tuple(HF_SOURCE_PATHS)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROMPT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class CalibrationError(ValueError):
    """A calibration artifact is malformed or cannot be compared."""


def gate_contract_document() -> dict[str, object]:
    """Return the exact language-neutral gate manifest represented by this tool."""

    return {
        "contract_version": CALIBRATION_SCHEMA_VERSION,
        "gate_id": CALIBRATION_GATE_ID,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "config_sha256": MODEL_CONFIG_SHA256,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_files_sha256": dict(TOKENIZER_FILES_SHA256),
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "eos_token_ids": list(MODEL_EOS_TOKEN_IDS),
        },
        "roles": {
            "fp32_oracle": "numeric-only",
            "bf16_oracle": "semantic-path-oracle-and-numeric-calibration",
            "native_candidate": "numeric-vs-fp32-and-semantic-vs-bf16",
        },
        "corpus": {
            "source": "benchmarks/prompts.jsonl",
            "expected_prompt_count": CALIBRATION_PROMPT_COUNT,
            "binding": "ordered-id-text-category-language-target-boundary-behavior",
        },
        "numeric": {
            "relative_error_denominator": "max(abs(fp32),1)",
            "comparison_scope": "each-full-tensor-per-prompt-per-candidate-variant",
            "gate_aggregation": "worst-prompt-metric-and-all-prompts-must-pass",
            "threshold_activation": {
                "status": "predeclared-requires-passing-full-corpus-oracle-report",
                "predeclaration_basis": "reviewed-failing-full-31-v1-calibration",
                "headroom_policy": "uniform-15-percent-outward-from-observed-aggregate",
                "headroom_formula": "upper=observed*1.15;cosine_min=1-(1-observed)*1.15",
                "calibration_evidence": {
                    "gate_id": "smollm2-fp32-bf16-native-e0-v1",
                    "git_revision": "8ab7490bfdf9efd1d7c7d831204b8e67c0c7c5b9",
                    "report_gate_id": "smollm2-hf-fp32-bf16-calibration-v1",
                    "report_path": "benchmarks/correctness/evidence/smollm2-fp32-bf16-native-e0-v1-failed-oracle-report.json",
                    "report_sha256": "ca13c033af2ddce5cfbf280fc1f4d2f95d0cba0e242bda8c59f2592946cec726",
                    "report_size_bytes": 48625,
                    "report_status": "fail",
                    "case_count": 31,
                    "failure_count": 12,
                    "semantic_self_check_pass": True,
                    "observed_aggregate_metrics": {
                        "first_layer_hidden": {
                            "max_abs": 0.33776283264160156,
                            "mean_abs": 0.007399384841076224,
                            "max_relative": 0.11807345598936081,
                            "mean_relative": 0.004708400919245019,
                            "cosine_similarity": 0.9999858320259609,
                        },
                        "final_logits": {
                            "max_abs": 5.089509963989258,
                            "mean_abs": 1.0011133210985765,
                            "max_relative": 1.0180343389511108,
                            "mean_relative": 0.11840520191704308,
                            "cosine_similarity": 0.9981769830865559,
                        },
                        "final_log_probs": {
                            "max_abs": 4.346452713012695,
                            "mean_abs": 0.5223633168837599,
                            "max_relative": 0.5014806389808655,
                            "mean_relative": 0.04059854641364647,
                            "cosine_similarity": 0.998937362025939,
                        },
                    },
                },
                "activation_evidence": "independent-replayed-passing-full-31-v2-corpus-report",
                "required_oracle_report_kind": "hf-oracle-calibration",
                "required_oracle_gate_id": "smollm2-hf-fp32-bf16-calibration-v2",
                "required_oracle_case_count": CALIBRATION_PROMPT_COUNT,
                "required_oracle_report_status": "pass",
                "failure_policy": "gate-fail-requires-version-bump-no-data-dependent-adjustment",
            },
            "tensors": {
                "first_layer_hidden": {
                    "definition": "full-first-transformer-layer-output-all-valid-token-positions",
                    "capture_cache_path": "off",
                    "fp32_sidecar_dtype": "float32",
                    "candidate_sidecar_dtype": "bfloat16",
                    "thresholds": CALIBRATION_THRESHOLDS["first_layer_hidden"],
                },
                "final_logits": {
                    "definition": "full-vocabulary-last-valid-position-logits",
                    "capture_cache_path": "off",
                    "fp32_sidecar_dtype": "float32",
                    "candidate_sidecar_dtype": "bfloat16",
                    "thresholds": CALIBRATION_THRESHOLDS["final_logits"],
                },
                "final_log_probs": {
                    "definition": "full-vocabulary-fp32-log-softmax-of-final-logits",
                    "pipeline_id": LOG_PROB_PIPELINE,
                    "capture_cache_path": "off",
                    "fp32_sidecar_dtype": "float32",
                    "candidate_sidecar_dtype": "float32",
                    "thresholds": CALIBRATION_THRESHOLDS["final_log_probs"],
                },
            },
        },
        "semantic": {
            "generation_steps": SEMANTIC_GENERATION_STEPS,
            "cross_cache_exact_window": CROSS_CACHE_EXACT_WINDOW,
            "divergence_step_index_origin": 0,
            "path_matching": {
                "native-cache-on": "hf-bf16-cache-on",
                "native-cache-off": "hf-bf16-cache-off",
            },
            "generated_token_ids_comparison": "ordered-exact",
            "top_1_comparison": "ordered-exact",
            "top_k": {"k": CALIBRATION_TOP_K, "comparison": "set-exact"},
        },
        "reduction_variants": {
            "oracle": HF_ORACLE_REDUCTION_VARIANT,
            "required_candidate": list(REQUIRED_CANDIDATE_REDUCTION_VARIANTS),
            "each_candidate_variant_compared_independently": True,
            "all_required_variants_must_pass": True,
            "alternate_profile_definition": {
                "applies_to": "all-floating-point-reductions-contributing-to-captured-tensors-or-greedy-logits",
                "included_reductions": [
                    "matmul-dot-product-sums",
                    "rmsnorm-sum-of-squares",
                    "attention-softmax-max-and-sum",
                    "final-log-softmax-max-and-sum",
                ],
                "reduction_axis_order": "logical-reduction-index-ascending",
                "chunk_local_order": "left-fold-in-logical-index-order",
                "chunk_elements": 37,
                "remainder_policy": "single-short-final-chunk",
                "balanced_merge_order": "merge-adjacent-chunk-partials-by-ascending-index-each-level-carry-unpaired-last",
                "accumulator_dtype_policy": "same-as-canonical-operator-contract",
            },
        },
        "provenance": {
            "required_hash_bindings": [
                "candidate-manifest",
                "matrix",
                "prompts",
                "model-config",
                "model-weights",
                "tokenizer-artifacts",
                "dependency-locks",
                "git-revisions",
                "environment",
                "lane-manifests",
            ],
            "candidate_runtime_dependency_class": "native-production",
            "candidate_git_dirty": False,
            "candidate_execution_bindings": [
                "bundled-executable-sha256",
                "clean-git-revision",
                "cargo-lock-sha256",
                "build-argv",
                "capture-argv",
            ],
            "e0_result_acceptance": {
                "pr01_native_lane_status": "contract-only-reject-e0-success",
                "future_activation": "raw-evidence-bundle-replay-required",
                "required_bundle_members": [
                    "fp32-manifest-and-sidecar",
                    "bf16-manifest-and-sidecar",
                    "passing-oracle-calibration-report",
                    "candidate-manifest-sidecar-and-executable",
                    "correctness-report",
                ],
                "approval_command": "rustinfer-reference calibrate-validate-report",
            },
            "result_correctness_report_sha256": "sha256-of-raw-report-file-bytes",
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise CalibrationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"value is not canonical JSON: {error}") from error


def aggregate_tokenizer_sha256(file_digests: Mapping[str, str]) -> str:
    """Hash the canonical JSON map of the five immutable tokenizer file hashes."""

    if set(file_digests) != set(TOKENIZER_ARTIFACT_FILENAMES):
        raise CalibrationError("tokenizer artifact set differs from immutable contract")
    normalized = {
        filename: _expect_sha(file_digests[filename], f"tokenizer.{filename}")
        for filename in TOKENIZER_ARTIFACT_FILENAMES
    }
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise CalibrationError("calibration timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CalibrationError("timestamp must be canonical UTC text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise CalibrationError("timestamp is invalid") from error
    if utc_text(parsed) != value:
        raise CalibrationError("timestamp is not canonical to whole seconds")
    return parsed


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= 0xFFFFFFFF
        ):
            raise CalibrationError("token ID cannot be encoded as canonical u32")
        digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()


def first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    """Return the zero-based first differing output token, including length mismatch."""

    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def ranked_top_k(values: Sequence[float], count: int) -> list[int]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise CalibrationError("top-k count must be positive")
    if len(values) < count:
        raise CalibrationError("top-k count exceeds tensor length")
    normalized: list[float] = []
    for value in values:
        converted = float(value)
        if not math.isfinite(converted):
            raise CalibrationError("top-k tensor contains a non-finite value")
        normalized.append(converted)
    return sorted(range(len(normalized)), key=lambda index: (-normalized[index], index))[
        :count
    ]


def top_k_token_set(values: Sequence[float], count: int) -> list[int]:
    """Return canonical ascending token IDs; the gate compares a set, not rank order."""

    return sorted(ranked_top_k(values, count))


def recompute_numeric_metrics(
    fp32_values: Sequence[float], candidate_values: Sequence[float]
) -> dict[str, float]:
    if len(fp32_values) != len(candidate_values) or not fp32_values:
        raise CalibrationError("numeric tensors must have equal non-zero lengths")
    chunks = (
        (
            fp32_values[start : start + _NUMERIC_METRIC_CHUNK_ELEMENTS],
            candidate_values[start : start + _NUMERIC_METRIC_CHUNK_ELEMENTS],
        )
        for start in range(0, len(fp32_values), _NUMERIC_METRIC_CHUNK_ELEMENTS)
    )
    return _recompute_numeric_metrics_from_chunks(chunks, len(fp32_values))


def _finite_f32(value: object, operation: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CalibrationError(
            f"numeric tensor {operation} is not a float32 value"
        ) from error
    if not math.isfinite(converted):
        raise CalibrationError("numeric tensor contains a non-finite value")
    try:
        rounded = _F32_STRUCT.unpack(_F32_STRUCT.pack(converted))[0]
    except (OverflowError, struct.error) as error:
        raise CalibrationError(
            f"numeric tensor {operation} exceeds the finite float32 range"
        ) from error
    if not math.isfinite(rounded):
        raise CalibrationError(
            f"numeric tensor {operation} exceeds the finite float32 range"
        )
    return rounded


def _recompute_numeric_metrics_from_chunks(
    chunks: Iterable[tuple[Sequence[object], Sequence[object]]], expected_count: int
) -> dict[str, float]:
    """Apply the portable F32 metric arithmetic to fixed-size value chunks."""

    maximum_absolute = 0.0
    maximum_relative = 0.0
    absolute_sums: list[float] = []
    relative_sums: list[float] = []
    products: list[float] = []
    reference_squares: list[float] = []
    candidate_squares: list[float] = []
    observed_count = 0
    for reference_raw, candidate_raw in chunks:
        if len(reference_raw) != len(candidate_raw) or not reference_raw:
            raise CalibrationError("numeric tensors must have equal non-zero lengths")
        reference = [_finite_f32(value, "value") for value in reference_raw]
        candidate = [_finite_f32(value, "value") for value in candidate_raw]
        absolute: list[float] = []
        relative: list[float] = []
        for reference_value, candidate_value in zip(
            reference, candidate, strict=True
        ):
            difference = abs(
                _finite_f32(reference_value - candidate_value, "difference")
            )
            absolute.append(difference)
            relative.append(
                _finite_f32(
                    difference / max(abs(reference_value), 1.0),
                    "relative difference",
                )
            )
        maximum_absolute = max(maximum_absolute, max(absolute))
        maximum_relative = max(maximum_relative, max(relative))
        absolute_sums.append(math.fsum(absolute))
        relative_sums.append(math.fsum(relative))
        products.append(
            math.fsum(
                reference_value * candidate_value
                for reference_value, candidate_value in zip(
                    reference, candidate, strict=True
                )
            )
        )
        reference_squares.append(math.fsum(value * value for value in reference))
        candidate_squares.append(math.fsum(value * value for value in candidate))
        observed_count += len(reference)
    if observed_count != expected_count or observed_count <= 0:
        raise CalibrationError("numeric tensors must have equal non-zero lengths")

    reference_norm = math.sqrt(math.fsum(reference_squares))
    candidate_norm = math.sqrt(math.fsum(candidate_squares))
    if reference_norm == 0.0 and candidate_norm == 0.0:
        cosine = 1.0
    elif reference_norm == 0.0 or candidate_norm == 0.0:
        cosine = 0.0
    else:
        cosine = math.fsum(products) / (reference_norm * candidate_norm)
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "max_abs": maximum_absolute,
        "mean_abs": math.fsum(absolute_sums) / observed_count,
        "max_relative": maximum_relative,
        "mean_relative": math.fsum(relative_sums) / observed_count,
        "cosine_similarity": cosine,
    }


def _flat_float_tensor(value: object) -> object:
    """Normalize a real tensor without materializing a Python-float list."""

    normalized = value
    for method in ("detach", "cpu", "float", "contiguous"):
        operation = getattr(normalized, method, None)
        if not callable(operation):
            raise TypeError(f"tensor has no callable {method}()")
        normalized = operation()
    reshape = getattr(normalized, "reshape", None)
    numel = getattr(normalized, "numel", None)
    if not callable(reshape) or not callable(numel):
        raise TypeError("tensor does not support reshape()/numel()")
    normalized = reshape(-1)
    if not callable(getattr(normalized, "numel", None)):
        raise TypeError("flattened tensor does not support numel()")
    return normalized


def recompute_numeric_metrics_from_tensors(
    reference_tensor: object,
    candidate_tensor: object,
) -> dict[str, float]:
    """Recompute metrics from raw tensors with bounded temporary memory.

    Safetensors yields real ``torch.Tensor`` objects.  The full 8,064-token
    hidden capture is deliberately not converted to millions of Python float
    objects. Both tensor and sequence loaders use the same fixed-size chunks,
    F32 element arithmetic, and binary64 ``math.fsum`` reductions.
    """

    try:
        reference = _flat_float_tensor(reference_tensor)
        candidate = _flat_float_tensor(candidate_tensor)
        reference_count = int(reference.numel())
        candidate_count = int(candidate.numel())
    except (AttributeError, TypeError, ValueError):
        return recompute_numeric_metrics(
            tensor_values(reference_tensor), tensor_values(candidate_tensor)
        )
    if reference_count != candidate_count or reference_count <= 0:
        raise CalibrationError("numeric tensors must have equal non-zero lengths")

    try:
        chunks = (
            (
                tensor_values(reference[start:stop]),
                tensor_values(candidate[start:stop]),
            )
            for start in range(0, reference_count, _NUMERIC_METRIC_CHUNK_ELEMENTS)
            for stop in (
                min(start + _NUMERIC_METRIC_CHUNK_ELEMENTS, reference_count),
            )
        )
        return _recompute_numeric_metrics_from_chunks(chunks, reference_count)
    except CalibrationError:
        raise
    except Exception as error:
        raise CalibrationError(f"cannot reduce calibration tensors: {error}") from error


def metrics_pass(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> bool:
    return (
        metrics["max_abs"] <= thresholds["max_abs_max"]
        and metrics["mean_abs"] <= thresholds["mean_abs_max"]
        and metrics["max_relative"] <= thresholds["max_relative_max"]
        and metrics["mean_relative"] <= thresholds["mean_relative_max"]
        and metrics["cosine_similarity"] >= thresholds["cosine_min"]
    )


def _expect_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CalibrationError(f"{path}: expected object")
    return value


def _expect_exact_keys(value: Mapping[str, object], keys: set[str], path: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise CalibrationError(f"{path}: keys differ; missing={missing}, extra={extra}")


def _expect_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationError(f"{path}: expected non-empty string")
    return value


def _expect_sha(value: object, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CalibrationError(f"{path}: expected lowercase SHA-256")
    return value


def _expect_int(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibrationError(f"{path}: expected integer >= {minimum}")
    return value


def _expect_finite(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{path}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError(f"{path}: expected finite number")
    return result


def _validate_token_ids(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CalibrationError(f"{path}: expected token ID array")
    result = tuple(value)
    for token_id in result:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= 0xFFFFFFFF
        ):
            raise CalibrationError(f"{path}: invalid token ID")
    return result


def _variant_map(variants: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    return {str(variant["variant_id"]): dict(variant) for variant in variants}


def expected_variant_configs(kind: object) -> dict[str, dict[str, object]]:
    if kind in {FP32_ORACLE_KIND, BF16_ORACLE_KIND}:
        return _variant_map((HF_ORACLE_REDUCTION_VARIANT,))
    if kind == CANDIDATE_KIND:
        return _variant_map(REQUIRED_CANDIDATE_REDUCTION_VARIANTS)
    raise CalibrationError("unsupported artifact kind")


def _validate_variant_config(value: object, expected: Mapping[str, object], path: str) -> None:
    config = _expect_object(value, path)
    _expect_exact_keys(
        config,
        {"variant_id", "partition_kind", "chunk_elements", "remainder_policy", "merge_order"},
        path,
    )
    if config != expected:
        raise CalibrationError(f"{path}: reduction execution profile differs")


def _validate_source_ref(value: object, expected_path: str, path: str) -> None:
    source = _expect_object(value, path)
    _expect_exact_keys(source, {"path", "sha256"}, path)
    if source["path"] != expected_path:
        raise CalibrationError(f"{path}.path: expected {expected_path!r}")
    _expect_sha(source["sha256"], f"{path}.sha256")


def _validate_tensor_ref(value: object, path: str) -> dict[str, object]:
    tensor = _expect_object(value, path)
    _expect_exact_keys(tensor, {"key", "shape", "dtype", "cache_path"}, path)
    _expect_string(tensor["key"], f"{path}.key")
    shape = tensor["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        )
    ):
        raise CalibrationError(f"{path}.shape: expected positive dimensions")
    if tensor["dtype"] not in {"float32", "bfloat16"}:
        raise CalibrationError(f"{path}.dtype: unsupported dtype")
    if tensor["cache_path"] != "off":
        raise CalibrationError(f"{path}.cache_path: tensor capture must be cache-off")
    return tensor


def _validate_semantic_path(value: object, path: str) -> dict[str, object]:
    semantic_path = _expect_object(value, path)
    _expect_exact_keys(semantic_path, {"generated_token_ids", "stop_reason"}, path)
    token_ids = _validate_token_ids(
        semantic_path["generated_token_ids"], f"{path}.generated_token_ids"
    )
    if not token_ids:
        raise CalibrationError(f"{path}.generated_token_ids: must be non-empty")
    if semantic_path["stop_reason"] not in {"eos", "max_new_tokens"}:
        raise CalibrationError(f"{path}.stop_reason: unsupported value")
    if (
        semantic_path["stop_reason"] == "max_new_tokens"
        and len(token_ids) != SEMANTIC_GENERATION_STEPS
    ):
        raise CalibrationError(f"{path}: max_new_tokens path has the wrong length")
    return semantic_path


def _validate_semantic(value: object, path: str) -> dict[str, object]:
    semantic = _expect_object(value, path)
    _expect_exact_keys(
        semantic,
        {
            "top_1_token_id",
            "top_k_token_id_set",
            "cache_on",
            "cache_off",
            "cross_cache_first_divergence_step",
            "cross_cache_exact_window_match",
        },
        path,
    )
    _expect_int(semantic["top_1_token_id"], f"{path}.top_1_token_id")
    top_k = _validate_token_ids(semantic["top_k_token_id_set"], f"{path}.top_k_token_id_set")
    if len(top_k) != CALIBRATION_TOP_K or list(top_k) != sorted(set(top_k)):
        raise CalibrationError(f"{path}.top_k_token_id_set: expected sorted exact set")
    cache_on = _validate_semantic_path(semantic["cache_on"], f"{path}.cache_on")
    cache_off = _validate_semantic_path(semantic["cache_off"], f"{path}.cache_off")
    on_tokens = tuple(cache_on["generated_token_ids"])
    off_tokens = tuple(cache_off["generated_token_ids"])
    if len(on_tokens) > SEMANTIC_GENERATION_STEPS or len(off_tokens) > SEMANTIC_GENERATION_STEPS:
        raise CalibrationError(f"{path}: generated length exceeds contract")
    divergence = first_divergence(on_tokens, off_tokens)
    if semantic["cross_cache_first_divergence_step"] != divergence:
        raise CalibrationError(f"{path}: divergence step was not recomputed")
    window_match = divergence is None or divergence >= CROSS_CACHE_EXACT_WINDOW
    if semantic["cross_cache_exact_window_match"] is not window_match:
        raise CalibrationError(f"{path}: exact-window flag mismatch")
    return semantic


def _validate_argv(value: object, path: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(argument, str) or not argument for argument in value)
    ):
        raise CalibrationError(f"{path}: expected non-empty argv strings")
    return value


def _flag_values(argv: Sequence[str], flag: str, path: str) -> list[str]:
    values: list[str] = []
    for index, argument in enumerate(argv):
        if argument != flag:
            continue
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise CalibrationError(f"{path}: {flag} lacks a value")
        values.append(argv[index + 1])
    return values


def _normalized_absolute_posix_path(value: object, path: str) -> str:
    text = _expect_string(value, path)
    components = text.split("/")
    if (
        not text.startswith("/")
        or text == "/"
        or "\\" in text
        or "\x00" in text
        or components[0] != ""
        or any(component in {"", ".", ".."} for component in components[1:])
    ):
        raise CalibrationError(f"{path}: expected normalized absolute POSIX path")
    return text


def _normalized_sibling(value: object, path: str, suffix: str) -> str:
    text = _expect_string(value, path)
    if (
        text in {".", ".."}
        or "/" in text
        or "\\" in text
        or "\x00" in text
        or len(text) <= len(suffix)
        or not text.endswith(suffix)
        or Path(text).name != text
    ):
        raise CalibrationError(
            f"{path}: expected normalized sibling filename ending in {suffix}"
        )
    return text


def _validate_candidate_execution(value: object, path: str) -> dict[str, object]:
    execution = _expect_object(value, path)
    _expect_exact_keys(execution, {"executable", "build_argv", "capture_argv"}, path)
    executable = _expect_object(execution["executable"], f"{path}.executable")
    _expect_exact_keys(executable, {"path", "sha256"}, f"{path}.executable")
    if executable["path"] != NATIVE_EXECUTABLE_FILENAME:
        raise CalibrationError(f"{path}.executable.path: canonical sibling name required")
    _expect_sha(executable["sha256"], f"{path}.executable.sha256")
    build_argv = _validate_argv(execution["build_argv"], f"{path}.build_argv")
    if build_argv != list(NATIVE_BUILD_ARGV):
        raise CalibrationError(f"{path}.build_argv: canonical locked release build required")
    capture_argv = _validate_argv(execution["capture_argv"], f"{path}.capture_argv")
    if capture_argv[:2] != [NATIVE_EXECUTABLE_FILENAME, "calibrate"]:
        raise CalibrationError(f"{path}.capture_argv: canonical native calibrate ABI required")
    expected_flags = [
        "--repository-root",
        "--model",
        "--gate-manifest",
        "--prompts",
        "--manifest",
        "--sidecar",
        "--reduction-variant",
        "--reduction-variant",
    ]
    if len(capture_argv) != 18 or capture_argv[2::2] != expected_flags:
        raise CalibrationError(
            f"{path}.capture_argv: exact ordered contract-v2 flag inventory required"
        )
    values = capture_argv[3::2]
    _normalized_absolute_posix_path(values[0], f"{path}.capture_argv.--repository-root")
    _normalized_absolute_posix_path(values[1], f"{path}.capture_argv.--model")
    if values[2] != HF_SOURCE_PATHS["gate_manifest"]:
        raise CalibrationError(f"{path}.capture_argv: --gate-manifest binding differs")
    if values[3] != HF_SOURCE_PATHS["prompts"]:
        raise CalibrationError(f"{path}.capture_argv: --prompts binding differs")
    _normalized_sibling(values[4], f"{path}.capture_argv.--manifest", ".json")
    _normalized_sibling(values[5], f"{path}.capture_argv.--sidecar", ".safetensors")
    expected_variants = [
        str(variant["variant_id"])
        for variant in REQUIRED_CANDIDATE_REDUCTION_VARIANTS
    ]
    if values[6:] != expected_variants:
        raise CalibrationError(f"{path}.capture_argv: both ordered variants are required")
    return execution


def validate_calibration_manifest(manifest: Mapping[str, object]) -> None:
    root = _expect_object(manifest, "manifest")
    _expect_exact_keys(
        root,
        {
            "schema_version",
            "artifact_kind",
            "created_at",
            "producer",
            "candidate_execution",
            "contract",
            "provenance",
            "corpus",
            "sidecar",
            "cases",
        },
        "manifest",
    )
    if root["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationError("manifest.schema_version: unsupported version")
    kind = root["artifact_kind"]
    if kind not in CALIBRATION_KINDS:
        raise CalibrationError("manifest.artifact_kind: unsupported kind")
    parse_utc(root["created_at"])

    producer = _expect_object(root["producer"], "manifest.producer")
    _expect_exact_keys(
        producer,
        {
            "implementation_id",
            "engine_revision",
            "runtime_dependency_class",
            "python_version",
            "python_executable_sha256",
            "python_platform_system",
            "python_platform_machine",
            "torch_version",
            "transformers_version",
            "safetensors_version",
        },
        "manifest.producer",
    )
    _expect_string(producer["implementation_id"], "manifest.producer.implementation_id")
    _expect_string(producer["engine_revision"], "manifest.producer.engine_revision")
    if kind in {FP32_ORACLE_KIND, BF16_ORACLE_KIND}:
        if producer != {
            "implementation_id": "hf-transformers-eager",
            "engine_revision": f"transformers-{TRANSFORMERS_VERSION}+torch-{TORCH_VERSION}",
            "runtime_dependency_class": RUNTIME_DEPENDENCY_CLASS,
            "python_version": PYTHON_VERSION,
            "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
            "python_platform_system": PYTHON_PLATFORM_SYSTEM,
            "python_platform_machine": PYTHON_PLATFORM_MACHINE,
            "torch_version": TORCH_VERSION,
            "transformers_version": TRANSFORMERS_VERSION,
            "safetensors_version": SAFETENSORS_VERSION,
        }:
            raise CalibrationError("manifest.producer: HF oracle runtime contract mismatch")
        _expect_string(producer["python_version"], "manifest.producer.python_version")
    else:
        if producer != {
            "implementation_id": "rustinfer-native",
            "engine_revision": NATIVE_ENGINE_REVISION,
            "runtime_dependency_class": "native-production",
            "python_version": None,
            "python_executable_sha256": None,
            "python_platform_system": None,
            "python_platform_machine": None,
            "torch_version": None,
            "transformers_version": None,
            "safetensors_version": None,
        }:
            raise CalibrationError("manifest.producer: candidate must be native-production")

    candidate_execution: dict[str, object] | None = None
    if kind == CANDIDATE_KIND:
        candidate_execution = _validate_candidate_execution(
            root["candidate_execution"], "manifest.candidate_execution"
        )
    elif root["candidate_execution"] is not None:
        raise CalibrationError("manifest.candidate_execution: HF oracle must use null")

    contract = _expect_object(root["contract"], "manifest.contract")
    _expect_exact_keys(
        contract,
        {
            "model_id",
            "gate_id",
            "model_revision",
            "config_sha256",
            "weights_sha256",
            "tokenizer_sha256",
            "tokenizer_files_sha256",
            "dtype",
            "attention_backend",
            "tensor_capture_cache_path",
            "log_prob_pipeline",
            "trust_remote_code",
            "max_context_tokens",
            "eos_token_ids",
            "semantic_generation_steps",
            "cross_cache_exact_window",
            "top_k",
            "oracle_reduction_variant",
            "required_candidate_reduction_variants",
        },
        "manifest.contract",
    )
    if contract["gate_id"] != CALIBRATION_GATE_ID:
        raise CalibrationError("manifest.contract.gate_id: immutable mismatch")
    if (
        contract["model_id"] != MODEL_ID
        or contract["model_revision"] != MODEL_REVISION
    ):
        raise CalibrationError("manifest.contract: model identity mismatch")
    if contract["weights_sha256"] != MODEL_WEIGHTS_SHA256:
        raise CalibrationError("manifest.contract.weights_sha256: immutable mismatch")
    if contract["config_sha256"] != MODEL_CONFIG_SHA256:
        raise CalibrationError("manifest.contract.config_sha256: immutable mismatch")
    if contract["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise CalibrationError("manifest.contract.tokenizer_sha256: immutable mismatch")
    tokenizer_files = _expect_object(
        contract["tokenizer_files_sha256"], "manifest.contract.tokenizer_files_sha256"
    )
    if set(tokenizer_files) != set(TOKENIZER_ARTIFACT_FILENAMES):
        raise CalibrationError("manifest.contract.tokenizer_files_sha256: file set mismatch")
    for filename, digest in tokenizer_files.items():
        _expect_sha(digest, f"manifest.contract.tokenizer_files_sha256.{filename}")
    if tokenizer_files != TOKENIZER_FILES_SHA256:
        raise CalibrationError(
            "manifest.contract.tokenizer_files_sha256: immutable artifact mismatch"
        )
    if contract["tokenizer_sha256"] != aggregate_tokenizer_sha256(tokenizer_files):
        raise CalibrationError("manifest.contract.tokenizer_sha256: aggregate mismatch")
    expected_dtype = "float32" if kind == FP32_ORACLE_KIND else "bfloat16"
    if contract["dtype"] != expected_dtype:
        raise CalibrationError("manifest.contract.dtype: artifact role mismatch")
    if (
        contract["attention_backend"] != ATTENTION_BACKEND
        or contract["tensor_capture_cache_path"] != "off"
        or contract["log_prob_pipeline"] != LOG_PROB_PIPELINE
        or contract["trust_remote_code"] is not False
        or contract["max_context_tokens"] != MAX_CONTEXT_TOKENS
        or contract["semantic_generation_steps"] != SEMANTIC_GENERATION_STEPS
        or contract["cross_cache_exact_window"] != CROSS_CACHE_EXACT_WINDOW
        or contract["top_k"] != CALIBRATION_TOP_K
        or contract["oracle_reduction_variant"] != HF_ORACLE_REDUCTION_VARIANT
        or contract["required_candidate_reduction_variants"]
        != list(REQUIRED_CANDIDATE_REDUCTION_VARIANTS)
    ):
        raise CalibrationError("manifest.contract: immutable execution contract mismatch")
    eos_token_ids = _validate_token_ids(
        contract["eos_token_ids"], "manifest.contract.eos_token_ids"
    )
    if eos_token_ids != MODEL_EOS_TOKEN_IDS:
        raise CalibrationError("manifest.contract.eos_token_ids: immutable model mismatch")

    provenance = _expect_object(root["provenance"], "manifest.provenance")
    _expect_exact_keys(
        provenance,
        {
            "sources",
            "git_revision",
            "git_dirty",
            "git_status_sha256",
            "environment_id",
            "observed_environment",
        },
        "manifest.provenance",
    )
    source_paths = NATIVE_SOURCE_PATHS if kind == CANDIDATE_KIND else HF_SOURCE_PATHS
    sources = _expect_object(provenance["sources"], "manifest.provenance.sources")
    if set(sources) != set(SOURCE_NAMES):
        raise CalibrationError("manifest.provenance.sources: source set mismatch")
    for name in SOURCE_NAMES:
        _validate_source_ref(
            sources[name], source_paths[name], f"manifest.provenance.sources.{name}"
        )
    if sources["python_version_file"]["sha256"] != PYTHON_VERSION_FILE_SHA256:
        raise CalibrationError(
            "manifest.provenance.sources.python_version_file.sha256: immutable mismatch"
        )
    revision = _expect_string(provenance["git_revision"], "manifest.provenance.git_revision")
    if not _GIT_REVISION_RE.fullmatch(revision):
        raise CalibrationError("manifest.provenance.git_revision: invalid")
    if provenance["git_dirty"] is not False or provenance["git_status_sha256"] != _EMPTY_SHA256:
        raise CalibrationError("manifest.provenance: correctness evidence requires clean tracked Git")
    if provenance["environment_id"] != PRIMARY_ENVIRONMENT_ID:
        raise CalibrationError("manifest.provenance.environment_id: target mismatch")
    try:
        validate_environment_snapshot(
            provenance["observed_environment"],
            "manifest.provenance.observed_environment",
        )
    except EnvironmentContractError as error:
        raise CalibrationError(str(error)) from error

    corpus = _expect_object(root["corpus"], "manifest.corpus")
    _expect_exact_keys(corpus, {"prompt_count"}, "manifest.corpus")
    prompt_count = _expect_int(corpus["prompt_count"], "manifest.corpus.prompt_count", minimum=1)
    if prompt_count != CALIBRATION_PROMPT_COUNT:
        raise CalibrationError(
            f"manifest.corpus.prompt_count: expected full {CALIBRATION_PROMPT_COUNT}-prompt corpus"
        )

    sidecar = _expect_object(root["sidecar"], "manifest.sidecar")
    _expect_exact_keys(sidecar, {"path", "sha256", "format", "tensor_count"}, "manifest.sidecar")
    sidecar_path = Path(_expect_string(sidecar["path"], "manifest.sidecar.path"))
    if sidecar_path.is_absolute() or len(sidecar_path.parts) != 1:
        raise CalibrationError("manifest.sidecar.path: must be a sibling filename")
    _expect_sha(sidecar["sha256"], "manifest.sidecar.sha256")
    if sidecar["format"] != "safetensors":
        raise CalibrationError("manifest.sidecar.format: must be safetensors")
    if candidate_execution is not None:
        sidecar_values = _flag_values(
            candidate_execution["capture_argv"],
            "--sidecar",
            "manifest.candidate_execution.capture_argv",
        )
        if sidecar_values != [sidecar_path.name]:
            raise CalibrationError(
                "manifest.candidate_execution.capture_argv: sidecar output differs"
            )
    variant_count = len(expected_variant_configs(kind))
    if sidecar["tensor_count"] != prompt_count * len(TENSOR_NAMES) * variant_count:
        raise CalibrationError("manifest.sidecar.tensor_count: inconsistent")

    cases = root["cases"]
    if not isinstance(cases, list) or len(cases) != prompt_count:
        raise CalibrationError("manifest.cases: count mismatch")
    expected_variants = expected_variant_configs(kind)
    seen_ids: set[str] = set()
    tensor_keys: set[str] = set()
    for index, raw_case in enumerate(cases):
        path = f"manifest.cases[{index}]"
        case = _expect_object(raw_case, path)
        _expect_exact_keys(
            case,
            {
                "prompt_id",
                "prompt_text_sha256",
                "prompt_metadata",
                "input_token_ids_sha256",
                "input_first_token_id",
                "input_token_count",
                "hidden_anchor_positions",
                "variants",
            },
            path,
        )
        prompt_id = _expect_string(case["prompt_id"], f"{path}.prompt_id")
        if not _PROMPT_ID_RE.fullmatch(prompt_id) or prompt_id in seen_ids:
            raise CalibrationError(f"{path}.prompt_id: invalid or duplicate")
        seen_ids.add(prompt_id)
        _expect_sha(case["prompt_text_sha256"], f"{path}.prompt_text_sha256")
        prompt_metadata = _expect_object(case["prompt_metadata"], f"{path}.prompt_metadata")
        _expect_exact_keys(
            prompt_metadata,
            {
                "category",
                "language",
                "target_prompt_tokens",
                "boundary_kind",
                "expected_behavior",
            },
            f"{path}.prompt_metadata",
        )
        for metadata_key in ("category", "language", "boundary_kind", "expected_behavior"):
            _expect_string(prompt_metadata[metadata_key], f"{path}.prompt_metadata.{metadata_key}")
        if prompt_metadata["target_prompt_tokens"] is not None:
            _expect_int(
                prompt_metadata["target_prompt_tokens"],
                f"{path}.prompt_metadata.target_prompt_tokens",
                minimum=1,
            )
        _expect_sha(case["input_token_ids_sha256"], f"{path}.input_token_ids_sha256")
        _expect_int(case["input_first_token_id"], f"{path}.input_first_token_id")
        token_count = _expect_int(case["input_token_count"], f"{path}.input_token_count", minimum=1)
        if token_count + SEMANTIC_GENERATION_STEPS > MAX_CONTEXT_TOKENS:
            raise CalibrationError(f"{path}.input_token_count: lacks semantic headroom")
        anchors = _expect_object(case["hidden_anchor_positions"], f"{path}.hidden_anchor_positions")
        expected_anchors = {"first": 0, "middle": (token_count - 1) // 2, "last": token_count - 1}
        if anchors != expected_anchors:
            raise CalibrationError(f"{path}.hidden_anchor_positions: canonical positions mismatch")

        variants = _expect_object(case["variants"], f"{path}.variants")
        if set(variants) != set(expected_variants):
            raise CalibrationError(f"{path}.variants: required reduction profiles differ")
        for variant_id, expected_config in expected_variants.items():
            variant_path = f"{path}.variants.{variant_id}"
            variant = _expect_object(variants[variant_id], variant_path)
            _expect_exact_keys(variant, {"config", "tensors", "semantic"}, variant_path)
            _validate_variant_config(variant["config"], expected_config, f"{variant_path}.config")
            tensors = _expect_object(variant["tensors"], f"{variant_path}.tensors")
            if set(tensors) != set(TENSOR_NAMES):
                raise CalibrationError(f"{variant_path}.tensors: tensor set mismatch")
            validated = {
                name: _validate_tensor_ref(tensors[name], f"{variant_path}.tensors.{name}")
                for name in TENSOR_NAMES
            }
            hidden_shape = validated["first_layer_hidden"]["shape"]
            logits_shape = validated["final_logits"]["shape"]
            log_probs_shape = validated["final_log_probs"]["shape"]
            if len(hidden_shape) != 2 or hidden_shape[0] != token_count:
                raise CalibrationError(f"{variant_path}: hidden must cover every valid token")
            if len(logits_shape) != 1 or logits_shape != log_probs_shape:
                raise CalibrationError(f"{variant_path}: logits/log-prob shape mismatch")
            activation_dtype = "float32" if kind == FP32_ORACLE_KIND else "bfloat16"
            if (
                validated["first_layer_hidden"]["dtype"] != activation_dtype
                or validated["final_logits"]["dtype"] != activation_dtype
                or validated["final_log_probs"]["dtype"] != "float32"
            ):
                raise CalibrationError(f"{variant_path}: role tensor dtype mismatch")
            for tensor in validated.values():
                key = str(tensor["key"])
                if key in tensor_keys:
                    raise CalibrationError(f"{variant_path}: duplicate sidecar tensor key")
                tensor_keys.add(key)
            if kind == FP32_ORACLE_KIND:
                if variant["semantic"] is not None:
                    raise CalibrationError(f"{variant_path}.semantic: FP32 is numeric-only")
            else:
                semantic = _validate_semantic(
                    variant["semantic"], f"{variant_path}.semantic"
                )
                if prompt_metadata["category"] == "early-eos":
                    for cache_path in ("cache_on", "cache_off"):
                        generated = semantic[cache_path]["generated_token_ids"]
                        if (
                            generated[0] not in eos_token_ids
                            or semantic[cache_path]["stop_reason"] != "eos"
                            or len(generated) != 1
                        ):
                            raise CalibrationError(
                                f"{variant_path}.semantic.{cache_path}: "
                                "early-eos must stop on EOS at output step zero"
                            )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read {label} {path}: {error}") from error
    return _expect_object(value, str(path))


def load_calibration_manifest(path: Path) -> dict[str, object]:
    manifest = _load_json_object(path, "calibration manifest")
    validate_calibration_manifest(manifest)
    return manifest


def verify_manifest_sources(manifest: Mapping[str, object], repo_root: Path) -> None:
    root = repo_root.resolve()
    for name in SOURCE_NAMES:
        source = manifest["provenance"]["sources"][name]
        path = (root / source["path"]).resolve()
        if root != path and root not in path.parents:
            raise CalibrationError(f"source {name} escapes repository root")
        if sha256_file(path) != source["sha256"]:
            raise CalibrationError(f"source {name} SHA-256 differs from manifest")
    revision = manifest["provenance"]["git_revision"]
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        for name in SOURCE_NAMES:
            source = manifest["provenance"]["sources"][name]
            committed = subprocess.run(
                ["git", "show", f"{revision}:{source['path']}"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
            if hashlib.sha256(committed).hexdigest() != source["sha256"]:
                raise CalibrationError(
                    f"source {name} bytes differ from claimed Git revision"
                )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CalibrationError(
            f"cannot verify source files at claimed Git revision {revision}: {error}"
        ) from error
    gate_path = root / manifest["provenance"]["sources"]["gate_manifest"]["path"]
    if _load_json_object(gate_path, "correctness gate manifest") != gate_contract_document():
        raise CalibrationError("language-neutral correctness gate manifest differs from tool")
    from .fixture import load_prompts

    prompts_path = root / manifest["provenance"]["sources"]["prompts"]["path"]
    prompts, corpus_sha256 = load_prompts(prompts_path)
    if corpus_sha256 != manifest["provenance"]["sources"]["prompts"]["sha256"]:
        raise CalibrationError("prompt corpus digest differs after parsing")
    cases = manifest["cases"]
    if len(prompts) != len(cases):
        raise CalibrationError("ordered prompt corpus and manifest case count differ")
    for index, (prompt, case) in enumerate(zip(prompts, cases, strict=True)):
        expected_metadata = prompt.metadata
        if (
            case["prompt_id"] != prompt.prompt_id
            or case["prompt_text_sha256"]
            != hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
            or case["prompt_metadata"] != expected_metadata
        ):
            raise CalibrationError(
                f"manifest.cases[{index}]: does not bind the ordered prompt row"
            )
        if (
            prompt.target_prompt_tokens is not None
            and case["input_token_count"] != prompt.target_prompt_tokens
        ):
            raise CalibrationError(
                f"manifest.cases[{index}]: target_prompt_tokens was not materialized exactly"
            )
    lane_path = root / manifest["provenance"]["sources"]["lane_manifest"]["path"]
    lane = _load_json_object(lane_path, "lane manifest")
    if manifest["artifact_kind"] == CANDIDATE_KIND:
        if (
            lane.get("lane_id") != "rustinfer-native"
            or lane.get("implementation_id") != "rustinfer-native"
            or lane.get("runtime_dependency_class") != "native-production"
        ):
            raise CalibrationError("candidate lane manifest is not rustinfer-native")
        engine = _expect_object(lane.get("engine"), "candidate lane.engine")
        if engine.get("revision") != manifest["producer"]["engine_revision"]:
            raise CalibrationError("candidate engine revision differs from lane manifest")
    elif (
        lane.get("lane_id") != "hf-transformers"
        or lane.get("implementation_id") != "hf-transformers-eager"
        or lane.get("runtime_dependency_class") != RUNTIME_DEPENDENCY_CLASS
    ):
        raise CalibrationError("oracle lane manifest is not hf-transformers")


def _canonical_dtype(tensor: object) -> str:
    dtype = str(getattr(tensor, "dtype", ""))
    mapping = {
        "torch.float32": "float32",
        "float32": "float32",
        "F32": "float32",
        "torch.bfloat16": "bfloat16",
        "bfloat16": "bfloat16",
        "BF16": "bfloat16",
    }
    if dtype not in mapping:
        raise CalibrationError(f"sidecar tensor has unsupported dtype {dtype!r}")
    return mapping[dtype]


def _tensor_shape(tensor: object) -> list[int]:
    try:
        shape = [int(dimension) for dimension in tensor.shape]
    except Exception as error:
        raise CalibrationError(f"sidecar tensor has invalid shape: {error}") from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise CalibrationError("sidecar tensor has non-positive shape")
    return shape


def _flatten_nested(value: object) -> list[float]:
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for child in value:
            result.extend(_flatten_nested(child))
        return result
    return [float(value)]


def tensor_values(tensor: object) -> list[float]:
    value = tensor
    for method in ("detach", "cpu", "float", "contiguous"):
        operation = getattr(value, method, None)
        if callable(operation):
            value = operation()
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        value = reshape(-1)
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    try:
        result = _flatten_nested(raw)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"sidecar tensor cannot be converted to floats: {error}") from error
    if not result:
        raise CalibrationError("sidecar tensor is empty")
    return result


SidecarLoader = Callable[[Path], Mapping[str, object]]


class _SafeTensorMapping(Mapping[str, object]):
    """Open one safetensors value at a time instead of retaining all 31 cases."""

    def __init__(self, path: Path, opener: Callable[..., object]) -> None:
        self._path = path
        self._context: object | None = None
        self._handle: object | None = None
        try:
            self._context = opener(str(self._path), framework="pt", device="cpu")
            self._handle = self._context.__enter__()
            self._keys = tuple(self._handle.keys())
        except Exception as error:
            self.close()
            raise CalibrationError(f"cannot index safetensors sidecar {path}: {error}") from error

    def __getitem__(self, key: str) -> object:
        if key not in self._keys:
            raise KeyError(key)
        try:
            if self._handle is None:
                raise CalibrationError("safetensors sidecar is closed")
            return self._handle.get_tensor(key)
        except Exception as error:
            raise CalibrationError(
                f"cannot read safetensors tensor {key!r}: {error}"
            ) from error

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def close(self) -> None:
        context, self._context, self._handle = self._context, None, None
        if context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()


def _default_sidecar_loader(path: Path) -> Mapping[str, object]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise CalibrationError("install pinned safetensors to compare calibration") from error
    return _SafeTensorMapping(path, safe_open)


def _load_verified_sidecar(
    manifest: Mapping[str, object], manifest_path: Path, loader: SidecarLoader
) -> Mapping[str, object]:
    if manifest["artifact_kind"] == CANDIDATE_KIND:
        execution = manifest["candidate_execution"]
        executable = execution["executable"]
        executable_path = manifest_path.resolve().parent / executable["path"]
        if sha256_file(executable_path) != executable["sha256"]:
            raise CalibrationError(
                f"candidate executable SHA-256 differs for {manifest_path}"
            )
        manifest_outputs = _flag_values(
            execution["capture_argv"],
            "--manifest",
            "manifest.candidate_execution.capture_argv",
        )
        if manifest_outputs != [manifest_path.name]:
            raise CalibrationError("candidate capture argv manifest output differs")
    sidecar = manifest["sidecar"]
    sidecar_path = manifest_path.resolve().parent / sidecar["path"]
    if sha256_file(sidecar_path) != sidecar["sha256"]:
        raise CalibrationError(f"sidecar SHA-256 differs for {manifest_path}")
    tensors = loader(sidecar_path)
    if not isinstance(tensors, Mapping):
        raise CalibrationError("sidecar loader did not return a tensor mapping")
    expected_refs = {
        variant["tensors"][name]["key"]: variant["tensors"][name]
        for case in manifest["cases"]
        for variant in case["variants"].values()
        for name in TENSOR_NAMES
    }
    if set(tensors) != set(expected_refs):
        raise CalibrationError("sidecar tensor keys differ from manifest")
    for key, reference in expected_refs.items():
        tensor = tensors[key]
        if _tensor_shape(tensor) != reference["shape"]:
            raise CalibrationError(f"sidecar tensor {key!r} shape differs from manifest")
        if _canonical_dtype(tensor) != reference["dtype"]:
            raise CalibrationError(f"sidecar tensor {key!r} dtype differs from manifest")
    return tensors


def verify_calibration_artifact(
    *,
    manifest: Mapping[str, object],
    manifest_path: Path,
    repo_root: Path,
    expected_kind: str | None = None,
    sidecar_loader: SidecarLoader | None = None,
) -> int:
    """Verify manifest, repository bindings, sidecar hash/keys/shapes/dtypes/top-k."""

    validate_calibration_manifest(manifest)
    if expected_kind is not None and manifest["artifact_kind"] != expected_kind:
        raise CalibrationError(
            f"artifact kind {manifest['artifact_kind']!r} != {expected_kind!r}"
        )
    verify_manifest_sources(manifest, repo_root)
    tensors = _load_verified_sidecar(
        manifest, manifest_path, sidecar_loader or _default_sidecar_loader
    )
    if manifest["artifact_kind"] != FP32_ORACLE_KIND:
        for case in manifest["cases"]:
            for variant_id, variant in case["variants"].items():
                semantic = variant["semantic"]
                logits = tensor_values(
                    tensors[variant["tensors"]["final_logits"]["key"]]
                )
                ranked = ranked_top_k(logits, CALIBRATION_TOP_K)
                if (
                    semantic["top_1_token_id"] != ranked[0]
                    or semantic["top_k_token_id_set"] != sorted(ranked)
                ):
                    raise CalibrationError(
                        f"{case['prompt_id']}/{variant_id}: top-k metadata is not tensor-derived"
                    )
    return len(tensors)


def _case_map(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {case["prompt_id"]: case for case in manifest["cases"]}


def _common_bindings(
    fp32: Mapping[str, object], bf16: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    for left, right, label in (
        (fp32, bf16, "FP32/BF16 oracle"),
        (fp32, candidate, "oracle/candidate"),
    ):
        for key in (
            "model_id",
            "model_revision",
            "config_sha256",
            "weights_sha256",
            "tokenizer_sha256",
            "tokenizer_files_sha256",
            "max_context_tokens",
            "eos_token_ids",
        ):
            if left["contract"][key] != right["contract"][key]:
                raise CalibrationError(f"{label} {key} binding differs")
        for source_name in (
            "matrix",
            "prompts",
            "gate_manifest",
            "environment",
            "environment_probe",
        ):
            if (
                left["provenance"]["sources"][source_name]["sha256"]
                != right["provenance"]["sources"][source_name]["sha256"]
            ):
                raise CalibrationError(f"{label} {source_name} binding differs")
        if left["provenance"]["environment_id"] != right["provenance"]["environment_id"]:
            raise CalibrationError(f"{label} environment_id differs")
        if environment_comparability_signature(
            left["provenance"]["observed_environment"]
        ) != environment_comparability_signature(
            right["provenance"]["observed_environment"]
        ):
            raise CalibrationError(f"{label} observed environment differs")
    for key in ("git_revision", "git_status_sha256"):
        if fp32["provenance"][key] != bf16["provenance"][key]:
            raise CalibrationError(f"FP32/BF16 oracle {key} differs")
    if (
        fp32["provenance"]["sources"]["dependency_lock"]["sha256"]
        != bf16["provenance"]["sources"]["dependency_lock"]["sha256"]
        or fp32["provenance"]["sources"]["lane_manifest"]["sha256"]
        != bf16["provenance"]["sources"]["lane_manifest"]["sha256"]
    ):
        raise CalibrationError("FP32/BF16 oracle dependency provenance differs")
    return {
        "model_id": fp32["contract"]["model_id"],
        "model_revision": fp32["contract"]["model_revision"],
        "config_sha256": fp32["contract"]["config_sha256"],
        "weights_sha256": fp32["contract"]["weights_sha256"],
        "tokenizer_sha256": fp32["contract"]["tokenizer_sha256"],
        "matrix_sha256": fp32["provenance"]["sources"]["matrix"]["sha256"],
        "prompts_sha256": fp32["provenance"]["sources"]["prompts"]["sha256"],
        "gate_manifest_sha256": fp32["provenance"]["sources"]["gate_manifest"]["sha256"],
        "environment_sha256": fp32["provenance"]["sources"]["environment"]["sha256"],
        "environment_id": fp32["provenance"]["environment_id"],
        "oracle_git_revision": fp32["provenance"]["git_revision"],
        "oracle_git_status_sha256": fp32["provenance"]["git_status_sha256"],
        "candidate_git_revision": candidate["provenance"]["git_revision"],
        "candidate_git_status_sha256": candidate["provenance"]["git_status_sha256"],
        "candidate_executable_sha256": candidate["candidate_execution"]["executable"][
            "sha256"
        ],
        "candidate_build_argv_sha256": hashlib.sha256(
            canonical_json_bytes(candidate["candidate_execution"]["build_argv"])
        ).hexdigest(),
        "candidate_capture_argv_sha256": hashlib.sha256(
            canonical_json_bytes(candidate["candidate_execution"]["capture_argv"])
        ).hexdigest(),
    }


def _metric_record(reference: object, candidate: object, tensor_name: str) -> dict[str, object]:
    metrics = recompute_numeric_metrics_from_tensors(reference, candidate)
    return {
        "metrics": metrics,
        "pass": metrics_pass(metrics, CALIBRATION_THRESHOLDS[tensor_name]),
    }


def _semantic_record(
    *,
    prompt_id: str,
    oracle_case: Mapping[str, object],
    oracle_tensors: Mapping[str, object],
    candidate_variant: Mapping[str, object],
    candidate_tensors: Mapping[str, object],
) -> dict[str, object]:
    oracle_variant = oracle_case["variants"][HF_ORACLE_REDUCTION_VARIANT["variant_id"]]
    oracle_semantic = oracle_variant["semantic"]
    candidate_semantic = candidate_variant["semantic"]
    oracle_logits = tensor_values(
        oracle_tensors[oracle_variant["tensors"]["final_logits"]["key"]]
    )
    candidate_logits = tensor_values(
        candidate_tensors[candidate_variant["tensors"]["final_logits"]["key"]]
    )
    oracle_ranked = ranked_top_k(oracle_logits, CALIBRATION_TOP_K)
    candidate_ranked = ranked_top_k(candidate_logits, CALIBRATION_TOP_K)
    oracle_set = sorted(oracle_ranked)
    candidate_set = sorted(candidate_ranked)
    if (
        oracle_semantic["top_1_token_id"] != oracle_ranked[0]
        or oracle_semantic["top_k_token_id_set"] != oracle_set
    ):
        raise CalibrationError(f"{prompt_id}: BF16 top-k metadata is not tensor-derived")
    if (
        candidate_semantic["top_1_token_id"] != candidate_ranked[0]
        or candidate_semantic["top_k_token_id_set"] != candidate_set
    ):
        raise CalibrationError(f"{prompt_id}: candidate top-k metadata is not tensor-derived")
    cache_on_exact = candidate_semantic["cache_on"] == oracle_semantic["cache_on"]
    cache_off_exact = candidate_semantic["cache_off"] == oracle_semantic["cache_off"]
    top_1_exact = candidate_ranked[0] == oracle_ranked[0]
    top_k_set_exact = set(candidate_set) == set(oracle_set)
    oracle_divergence = first_divergence(
        oracle_semantic["cache_on"]["generated_token_ids"],
        oracle_semantic["cache_off"]["generated_token_ids"],
    )
    candidate_divergence = first_divergence(
        candidate_semantic["cache_on"]["generated_token_ids"],
        candidate_semantic["cache_off"]["generated_token_ids"],
    )
    window_match = (
        (oracle_divergence is None or oracle_divergence >= CROSS_CACHE_EXACT_WINDOW)
        and (candidate_divergence is None or candidate_divergence >= CROSS_CACHE_EXACT_WINDOW)
    )
    result: dict[str, object] = {
        "cache_on_exact": cache_on_exact,
        "cache_off_exact": cache_off_exact,
        "top_1_exact": top_1_exact,
        "top_k_set_exact": top_k_set_exact,
        "hf_cross_cache_first_divergence_step": oracle_divergence,
        "candidate_cross_cache_first_divergence_step": candidate_divergence,
        "cross_cache_exact_window": CROSS_CACHE_EXACT_WINDOW,
        "cross_cache_exact_window_match": window_match,
    }
    result["pass"] = all(
        bool(result[key])
        for key in (
            "cache_on_exact",
            "cache_off_exact",
            "top_1_exact",
            "top_k_set_exact",
            "cross_cache_exact_window_match",
        )
    )
    return result


def compare_calibrations(
    *,
    fp32_manifest: Mapping[str, object],
    fp32_manifest_path: Path,
    bf16_manifest: Mapping[str, object],
    bf16_manifest_path: Path,
    oracle_calibration_report: Mapping[str, object],
    oracle_calibration_report_path: Path,
    candidate_manifest: Mapping[str, object],
    candidate_manifest_path: Path,
    repo_root: Path,
    created_at: datetime,
    sidecar_loader: SidecarLoader | None = None,
) -> dict[str, object]:
    for manifest in (fp32_manifest, bf16_manifest, candidate_manifest):
        validate_calibration_manifest(manifest)
        verify_manifest_sources(manifest, repo_root)
    if fp32_manifest["artifact_kind"] != FP32_ORACLE_KIND:
        raise CalibrationError("FP32 input is not the numeric oracle")
    if bf16_manifest["artifact_kind"] != BF16_ORACLE_KIND:
        raise CalibrationError("BF16 input is not the semantic oracle")
    if candidate_manifest["artifact_kind"] != CANDIDATE_KIND:
        raise CalibrationError("candidate input is not a native candidate")
    from .oracle_calibration import replay_validate_oracle_report

    loader = sidecar_loader or _default_sidecar_loader
    replay_validate_oracle_report(
        report=oracle_calibration_report,
        fp32_manifest=fp32_manifest,
        fp32_manifest_path=fp32_manifest_path,
        bf16_manifest=bf16_manifest,
        bf16_manifest_path=bf16_manifest_path,
        repo_root=repo_root,
        sidecar_loader=loader,
    )
    if (
        oracle_calibration_report["status"] != "pass"
        or oracle_calibration_report["summary"]["case_count"]
        != CALIBRATION_PROMPT_COUNT
        or oracle_calibration_report["e0_candidate_evidence"] is not False
    ):
        raise CalibrationError(
            "candidate gate requires passing replayed full-corpus HF oracle calibration"
        )
    bindings = _common_bindings(fp32_manifest, bf16_manifest, candidate_manifest)
    fp32_tensors = _load_verified_sidecar(fp32_manifest, fp32_manifest_path, loader)
    bf16_tensors = _load_verified_sidecar(bf16_manifest, bf16_manifest_path, loader)
    candidate_tensors = _load_verified_sidecar(candidate_manifest, candidate_manifest_path, loader)
    fp32_cases = _case_map(fp32_manifest)
    bf16_cases = _case_map(bf16_manifest)
    candidate_cases = _case_map(candidate_manifest)
    if list(fp32_cases) != list(bf16_cases) or list(fp32_cases) != list(candidate_cases):
        raise CalibrationError("oracle/candidate prompt order differs")

    candidate_variant_ids = [
        str(config["variant_id"]) for config in REQUIRED_CANDIDATE_REDUCTION_VARIANTS
    ]
    aggregate_metrics: dict[str, dict[str, dict[str, float]]] = {
        variant_id: {
            tensor_name: {
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "max_relative": 0.0,
                "mean_relative": 0.0,
                "cosine_similarity": 1.0,
            }
            for tensor_name in TENSOR_NAMES
        }
        for variant_id in candidate_variant_ids
    }
    case_reports: list[dict[str, object]] = []
    oracle_variant_id = str(HF_ORACLE_REDUCTION_VARIANT["variant_id"])
    for prompt_id, fp32_case in fp32_cases.items():
        bf16_case = bf16_cases[prompt_id]
        candidate_case = candidate_cases[prompt_id]
        for key in (
            "prompt_text_sha256",
            "prompt_metadata",
            "input_token_ids_sha256",
            "input_first_token_id",
            "input_token_count",
            "hidden_anchor_positions",
        ):
            if fp32_case[key] != bf16_case[key] or fp32_case[key] != candidate_case[key]:
                raise CalibrationError(f"{prompt_id}: {key} binding differs")
        fp32_variant = fp32_case["variants"][oracle_variant_id]
        variant_reports: dict[str, object] = {}
        for variant_id in candidate_variant_ids:
            candidate_variant = candidate_case["variants"][variant_id]
            numeric: dict[str, object] = {}
            for tensor_name in TENSOR_NAMES:
                fp32_ref = fp32_variant["tensors"][tensor_name]
                candidate_ref = candidate_variant["tensors"][tensor_name]
                if candidate_ref["shape"] != fp32_ref["shape"]:
                    raise CalibrationError(f"{prompt_id}/{variant_id}: {tensor_name} shape differs")
                record = _metric_record(
                    fp32_tensors[fp32_ref["key"]],
                    candidate_tensors[candidate_ref["key"]],
                    tensor_name,
                )
                numeric[tensor_name] = record
                aggregate = aggregate_metrics[variant_id][tensor_name]
                metrics = record["metrics"]
                for metric_name in ("max_abs", "mean_abs", "max_relative", "mean_relative"):
                    aggregate[metric_name] = max(aggregate[metric_name], metrics[metric_name])
                aggregate["cosine_similarity"] = min(
                    aggregate["cosine_similarity"], metrics["cosine_similarity"]
                )
            semantic = _semantic_record(
                prompt_id=prompt_id,
                oracle_case=bf16_case,
                oracle_tensors=bf16_tensors,
                candidate_variant=candidate_variant,
                candidate_tensors=candidate_tensors,
            )
            variant_pass = all(record["pass"] for record in numeric.values()) and semantic["pass"]
            variant_reports[variant_id] = {
                "numeric": numeric,
                "semantic": semantic,
                "pass": variant_pass,
            }
        case_reports.append(
            {
                "prompt_id": prompt_id,
                "variants": variant_reports,
                "pass": all(record["pass"] for record in variant_reports.values()),
            }
        )

    variant_summaries: dict[str, object] = {}
    for variant_id in candidate_variant_ids:
        aggregate = {
            tensor_name: {
                "metrics": metrics,
                "pass": metrics_pass(metrics, CALIBRATION_THRESHOLDS[tensor_name]),
            }
            for tensor_name, metrics in aggregate_metrics[variant_id].items()
        }
        case_variants = [case["variants"][variant_id] for case in case_reports]
        numeric_pass = all(item["pass"] for item in aggregate.values()) and all(
            all(item["pass"] for item in case["numeric"].values()) for case in case_variants
        )
        semantic_pass = all(case["semantic"]["pass"] for case in case_variants)
        variant_summaries[variant_id] = {
            "case_count": len(case_variants),
            "failure_count": sum(not case["pass"] for case in case_variants),
            "numeric_pass": numeric_pass,
            "semantic_pass": semantic_pass,
            "aggregate_numeric": aggregate,
            "pass": numeric_pass and semantic_pass,
        }
    numeric_pass = all(summary["numeric_pass"] for summary in variant_summaries.values())
    semantic_pass = all(summary["semantic_pass"] for summary in variant_summaries.values())
    report: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "gate_id": CALIBRATION_GATE_ID,
        "created_at": utc_text(created_at),
        "status": "pass" if numeric_pass and semantic_pass else "fail",
        "roles": {
            "fp32": "numeric-only",
            "bf16": "semantic-only",
            "candidate_numeric_reference": "fp32",
            "candidate_semantic_reference": "hf-bf16-path-matched",
        },
        "gate_contract": {
            "thresholds": CALIBRATION_THRESHOLDS,
            "oracle_reduction_variant": HF_ORACLE_REDUCTION_VARIANT,
            "required_candidate_reduction_variants": list(REQUIRED_CANDIDATE_REDUCTION_VARIANTS),
            "cross_cache_exact_window": CROSS_CACHE_EXACT_WINDOW,
            "top_k_comparison": "set-exact",
            "top_1_comparison": "ordered-exact",
            "threshold_activation_evidence": "replayed-passing-full-31-hf-oracle-calibration-report-v2",
        },
        "inputs": {
            "fp32_manifest_sha256": sha256_file(fp32_manifest_path),
            "bf16_manifest_sha256": sha256_file(bf16_manifest_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
            "oracle_calibration_report_sha256": sha256_file(
                oracle_calibration_report_path
            ),
            "fp32_sidecar_sha256": fp32_manifest["sidecar"]["sha256"],
            "bf16_sidecar_sha256": bf16_manifest["sidecar"]["sha256"],
            "candidate_sidecar_sha256": candidate_manifest["sidecar"]["sha256"],
        },
        "bindings": {
            **bindings,
            "dependency_locks": {
                "fp32": fp32_manifest["provenance"]["sources"]["dependency_lock"]["sha256"],
                "bf16": bf16_manifest["provenance"]["sources"]["dependency_lock"]["sha256"],
                "candidate": candidate_manifest["provenance"]["sources"]["dependency_lock"]["sha256"],
            },
            "lane_manifests": {
                "fp32": fp32_manifest["provenance"]["sources"]["lane_manifest"]["sha256"],
                "bf16": bf16_manifest["provenance"]["sources"]["lane_manifest"]["sha256"],
                "candidate": candidate_manifest["provenance"]["sources"]["lane_manifest"]["sha256"],
            },
        },
        "summary": {
            "case_count": len(case_reports),
            "candidate_variant_count": len(candidate_variant_ids),
            "failure_count": sum(
                not case["variants"][variant_id]["pass"]
                for case in case_reports
                for variant_id in candidate_variant_ids
            ),
            "numeric_pass": numeric_pass,
            "semantic_pass": semantic_pass,
            "variants": variant_summaries,
        },
        "cases": case_reports,
    }
    _validate_report_structure(report)
    return report


def _validate_metrics(value: object, tensor_name: str, path: str) -> dict[str, float]:
    del tensor_name
    metrics = _expect_object(value, path)
    _expect_exact_keys(
        metrics,
        {"max_abs", "mean_abs", "max_relative", "mean_relative", "cosine_similarity"},
        path,
    )
    normalized = {
        key: _expect_finite(metric, f"{path}.{key}") for key, metric in metrics.items()
    }
    if any(
        normalized[key] < 0.0
        for key in ("max_abs", "mean_abs", "max_relative", "mean_relative")
    ):
        raise CalibrationError(f"{path}: error metrics must be non-negative")
    if not -1.0 <= normalized["cosine_similarity"] <= 1.0:
        raise CalibrationError(f"{path}.cosine_similarity: outside [-1, 1]")
    return normalized


def _validate_numeric_record(value: object, tensor_name: str, path: str) -> dict[str, object]:
    record = _expect_object(value, path)
    _expect_exact_keys(record, {"metrics", "pass"}, path)
    metrics = _validate_metrics(record["metrics"], tensor_name, f"{path}.metrics")
    expected = metrics_pass(metrics, CALIBRATION_THRESHOLDS[tensor_name])
    if record["pass"] is not expected:
        raise CalibrationError(f"{path}.pass: inconsistent with predeclared threshold")
    return record


def _validate_report_semantic(value: object, path: str) -> dict[str, object]:
    semantic = _expect_object(value, path)
    _expect_exact_keys(
        semantic,
        {
            "cache_on_exact",
            "cache_off_exact",
            "top_1_exact",
            "top_k_set_exact",
            "hf_cross_cache_first_divergence_step",
            "candidate_cross_cache_first_divergence_step",
            "cross_cache_exact_window",
            "cross_cache_exact_window_match",
            "pass",
        },
        path,
    )
    for key in (
        "cache_on_exact",
        "cache_off_exact",
        "top_1_exact",
        "top_k_set_exact",
        "cross_cache_exact_window_match",
        "pass",
    ):
        if not isinstance(semantic[key], bool):
            raise CalibrationError(f"{path}.{key}: expected boolean")
    for key in (
        "hf_cross_cache_first_divergence_step",
        "candidate_cross_cache_first_divergence_step",
    ):
        if semantic[key] is not None:
            _expect_int(semantic[key], f"{path}.{key}")
    if semantic["cross_cache_exact_window"] != CROSS_CACHE_EXACT_WINDOW:
        raise CalibrationError(f"{path}.cross_cache_exact_window: contract changed")
    expected = all(
        semantic[key]
        for key in (
            "cache_on_exact",
            "cache_off_exact",
            "top_1_exact",
            "top_k_set_exact",
            "cross_cache_exact_window_match",
        )
    )
    if semantic["pass"] is not expected:
        raise CalibrationError(f"{path}.pass: inconsistent")
    return semantic


def _worst_metrics(records: Sequence[Mapping[str, object]], tensor_name: str) -> dict[str, float]:
    metrics = [record["numeric"][tensor_name]["metrics"] for record in records]
    return {
        "max_abs": max(metric["max_abs"] for metric in metrics),
        "mean_abs": max(metric["mean_abs"] for metric in metrics),
        "max_relative": max(metric["max_relative"] for metric in metrics),
        "mean_relative": max(metric["mean_relative"] for metric in metrics),
        "cosine_similarity": min(metric["cosine_similarity"] for metric in metrics),
    }


def _validate_report_structure(report: Mapping[str, object]) -> None:
    root = _expect_object(report, "report")
    _expect_exact_keys(
        root,
        {
            "schema_version",
            "gate_id",
            "created_at",
            "status",
            "roles",
            "gate_contract",
            "inputs",
            "bindings",
            "summary",
            "cases",
        },
        "report",
    )
    if root["schema_version"] != CALIBRATION_SCHEMA_VERSION or root["gate_id"] != CALIBRATION_GATE_ID:
        raise CalibrationError("report: schema or gate ID mismatch")
    parse_utc(root["created_at"])
    if root["roles"] != {
        "fp32": "numeric-only",
        "bf16": "semantic-only",
        "candidate_numeric_reference": "fp32",
        "candidate_semantic_reference": "hf-bf16-path-matched",
    }:
        raise CalibrationError("report.roles: immutable role split changed")
    if root["gate_contract"] != {
        "thresholds": CALIBRATION_THRESHOLDS,
        "oracle_reduction_variant": HF_ORACLE_REDUCTION_VARIANT,
        "required_candidate_reduction_variants": list(REQUIRED_CANDIDATE_REDUCTION_VARIANTS),
        "cross_cache_exact_window": CROSS_CACHE_EXACT_WINDOW,
        "top_k_comparison": "set-exact",
        "top_1_comparison": "ordered-exact",
        "threshold_activation_evidence": "replayed-passing-full-31-hf-oracle-calibration-report-v2",
    }:
        raise CalibrationError("report.gate_contract: frozen gate changed")
    inputs = _expect_object(root["inputs"], "report.inputs")
    _expect_exact_keys(
        inputs,
        {
            "fp32_manifest_sha256",
            "bf16_manifest_sha256",
            "candidate_manifest_sha256",
            "oracle_calibration_report_sha256",
            "fp32_sidecar_sha256",
            "bf16_sidecar_sha256",
            "candidate_sidecar_sha256",
        },
        "report.inputs",
    )
    for key, digest in inputs.items():
        _expect_sha(digest, f"report.inputs.{key}")
    bindings = _expect_object(root["bindings"], "report.bindings")
    _expect_exact_keys(
        bindings,
        {
            "model_id",
            "model_revision",
            "config_sha256",
            "weights_sha256",
            "tokenizer_sha256",
            "matrix_sha256",
            "prompts_sha256",
            "gate_manifest_sha256",
            "environment_sha256",
            "environment_id",
            "oracle_git_revision",
            "oracle_git_status_sha256",
            "candidate_git_revision",
            "candidate_git_status_sha256",
            "candidate_executable_sha256",
            "candidate_build_argv_sha256",
            "candidate_capture_argv_sha256",
            "dependency_locks",
            "lane_manifests",
        },
        "report.bindings",
    )
    if bindings["model_id"] != MODEL_ID or bindings["model_revision"] != MODEL_REVISION:
        raise CalibrationError("report.bindings: model identity mismatch")
    if (
        bindings["config_sha256"] != MODEL_CONFIG_SHA256
        or bindings["weights_sha256"] != MODEL_WEIGHTS_SHA256
        or bindings["tokenizer_sha256"] != TOKENIZER_SHA256
    ):
        raise CalibrationError("report.bindings: immutable model artifact mismatch")
    if bindings["environment_id"] != PRIMARY_ENVIRONMENT_ID:
        raise CalibrationError("report.bindings.environment_id: mismatch")
    for key in (
        "config_sha256",
        "tokenizer_sha256",
        "matrix_sha256",
        "prompts_sha256",
        "gate_manifest_sha256",
        "environment_sha256",
        "oracle_git_status_sha256",
        "candidate_git_status_sha256",
        "candidate_executable_sha256",
        "candidate_build_argv_sha256",
        "candidate_capture_argv_sha256",
    ):
        _expect_sha(bindings[key], f"report.bindings.{key}")
    for key in ("oracle_git_revision", "candidate_git_revision"):
        revision = _expect_string(bindings[key], f"report.bindings.{key}")
        if not _GIT_REVISION_RE.fullmatch(revision):
            raise CalibrationError(f"report.bindings.{key}: invalid")
    for group_name in ("dependency_locks", "lane_manifests"):
        group = _expect_object(bindings[group_name], f"report.bindings.{group_name}")
        _expect_exact_keys(group, {"fp32", "bf16", "candidate"}, f"report.bindings.{group_name}")
        for role, digest in group.items():
            _expect_sha(digest, f"report.bindings.{group_name}.{role}")

    cases = root["cases"]
    if not isinstance(cases, list) or len(cases) != CALIBRATION_PROMPT_COUNT:
        raise CalibrationError(
            f"report.cases: expected full {CALIBRATION_PROMPT_COUNT}-prompt corpus"
        )
    candidate_variant_ids = [str(item["variant_id"]) for item in REQUIRED_CANDIDATE_REDUCTION_VARIANTS]
    seen_prompts: set[str] = set()
    for case_index, raw_case in enumerate(cases):
        path = f"report.cases[{case_index}]"
        case = _expect_object(raw_case, path)
        _expect_exact_keys(case, {"prompt_id", "variants", "pass"}, path)
        prompt_id = _expect_string(case["prompt_id"], f"{path}.prompt_id")
        if not _PROMPT_ID_RE.fullmatch(prompt_id) or prompt_id in seen_prompts:
            raise CalibrationError(f"{path}.prompt_id: invalid or duplicate")
        seen_prompts.add(prompt_id)
        variants = _expect_object(case["variants"], f"{path}.variants")
        if set(variants) != set(candidate_variant_ids):
            raise CalibrationError(f"{path}.variants: required variants differ")
        for variant_id in candidate_variant_ids:
            variant_path = f"{path}.variants.{variant_id}"
            variant = _expect_object(variants[variant_id], variant_path)
            _expect_exact_keys(variant, {"numeric", "semantic", "pass"}, variant_path)
            numeric = _expect_object(variant["numeric"], f"{variant_path}.numeric")
            if set(numeric) != set(TENSOR_NAMES):
                raise CalibrationError(f"{variant_path}.numeric: tensor set differs")
            for tensor_name in TENSOR_NAMES:
                _validate_numeric_record(
                    numeric[tensor_name], tensor_name, f"{variant_path}.numeric.{tensor_name}"
                )
            semantic = _validate_report_semantic(variant["semantic"], f"{variant_path}.semantic")
            expected = all(numeric[name]["pass"] for name in TENSOR_NAMES) and semantic["pass"]
            if variant["pass"] is not expected:
                raise CalibrationError(f"{variant_path}.pass: inconsistent")
        expected_case = all(variants[variant_id]["pass"] for variant_id in candidate_variant_ids)
        if case["pass"] is not expected_case:
            raise CalibrationError(f"{path}.pass: inconsistent")

    summary = _expect_object(root["summary"], "report.summary")
    _expect_exact_keys(
        summary,
        {
            "case_count",
            "candidate_variant_count",
            "failure_count",
            "numeric_pass",
            "semantic_pass",
            "variants",
        },
        "report.summary",
    )
    if summary["case_count"] != len(cases) or summary["candidate_variant_count"] != len(candidate_variant_ids):
        raise CalibrationError("report.summary: case/variant count mismatch")
    summary_variants = _expect_object(summary["variants"], "report.summary.variants")
    if set(summary_variants) != set(candidate_variant_ids):
        raise CalibrationError("report.summary.variants: required variants differ")
    expected_failures = 0
    for variant_id in candidate_variant_ids:
        path = f"report.summary.variants.{variant_id}"
        variant_summary = _expect_object(summary_variants[variant_id], path)
        _expect_exact_keys(
            variant_summary,
            {
                "case_count",
                "failure_count",
                "numeric_pass",
                "semantic_pass",
                "aggregate_numeric",
                "pass",
            },
            path,
        )
        records = [case["variants"][variant_id] for case in cases]
        failures = sum(not record["pass"] for record in records)
        expected_failures += failures
        if variant_summary["case_count"] != len(records) or variant_summary["failure_count"] != failures:
            raise CalibrationError(f"{path}: counts inconsistent")
        aggregates = _expect_object(variant_summary["aggregate_numeric"], f"{path}.aggregate_numeric")
        if set(aggregates) != set(TENSOR_NAMES):
            raise CalibrationError(f"{path}.aggregate_numeric: tensor set differs")
        for tensor_name in TENSOR_NAMES:
            aggregate = _validate_numeric_record(
                aggregates[tensor_name], tensor_name, f"{path}.aggregate_numeric.{tensor_name}"
            )
            if aggregate["metrics"] != _worst_metrics(records, tensor_name):
                raise CalibrationError(f"{path}.aggregate_numeric.{tensor_name}: not recomputed")
        numeric_pass = all(
            all(record["numeric"][name]["pass"] for name in TENSOR_NAMES)
            for record in records
        ) and all(aggregates[name]["pass"] for name in TENSOR_NAMES)
        semantic_pass = all(record["semantic"]["pass"] for record in records)
        if (
            variant_summary["numeric_pass"] is not numeric_pass
            or variant_summary["semantic_pass"] is not semantic_pass
            or variant_summary["pass"] is not (numeric_pass and semantic_pass)
        ):
            raise CalibrationError(f"{path}: gate booleans inconsistent")
    numeric_pass = all(summary_variants[variant_id]["numeric_pass"] for variant_id in candidate_variant_ids)
    semantic_pass = all(summary_variants[variant_id]["semantic_pass"] for variant_id in candidate_variant_ids)
    if (
        summary["failure_count"] != expected_failures
        or summary["numeric_pass"] is not numeric_pass
        or summary["semantic_pass"] is not semantic_pass
    ):
        raise CalibrationError("report.summary: aggregate gate inconsistent")
    expected_status = "pass" if numeric_pass and semantic_pass else "fail"
    if root["status"] != expected_status:
        raise CalibrationError("report.status: inconsistent with all candidate variants")


def load_correctness_report(path: Path) -> dict[str, object]:
    report = _load_json_object(path, "correctness report")
    _validate_report_structure(report)
    return report


def replay_validate_correctness_report(
    *,
    report: Mapping[str, object],
    fp32_manifest: Mapping[str, object],
    fp32_manifest_path: Path,
    bf16_manifest: Mapping[str, object],
    bf16_manifest_path: Path,
    oracle_calibration_report: Mapping[str, object],
    oracle_calibration_report_path: Path,
    candidate_manifest: Mapping[str, object],
    candidate_manifest_path: Path,
    repo_root: Path,
    sidecar_loader: SidecarLoader | None = None,
) -> None:
    """Approve a report only by replaying its bound raw tensor comparison."""

    _validate_report_structure(report)
    expected = compare_calibrations(
        fp32_manifest=fp32_manifest,
        fp32_manifest_path=fp32_manifest_path,
        bf16_manifest=bf16_manifest,
        bf16_manifest_path=bf16_manifest_path,
        oracle_calibration_report=oracle_calibration_report,
        oracle_calibration_report_path=oracle_calibration_report_path,
        candidate_manifest=candidate_manifest,
        candidate_manifest_path=candidate_manifest_path,
        repo_root=repo_root,
        created_at=parse_utc(report["created_at"]),
        sidecar_loader=sidecar_loader,
    )
    if canonical_json_bytes(report) != canonical_json_bytes(expected):
        raise CalibrationError("correctness report differs from comparator replay")


def write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise CalibrationError(f"refusing to overwrite existing artifact: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
