#!/usr/bin/env python3
"""Collect one Qwen2.5-3B native-D128 numerical trace from Rust test stdout.

The marker is an offline numerical diagnostic. It is bound to a cache-on
teacher-forced Hugging Face reference and its cache-off control, but it never
represents a serving-performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, NoReturn


MAX_TEST_STDOUT_BYTES = 64 * 1024 * 1024
MARKER_PREFIX = "RILEY_QWEN3B_NATIVE_D128_LOGIT_TRACE="
TRACE_SCHEMA_VERSION = "riley.qwen3b-native-d128-teacher-forced-logit-trace.v2"
COLLECTED_SCHEMA_VERSION = (
    "riley.qwen3b-native-d128-teacher-forced-logit-trace-artifact.v2"
)
TRACE_ARTIFACT_KIND = (
    "qwen2.5-3b-native-d128-scheduler-committed-teacher-forced-logit-trace"
)
COLLECTED_ARTIFACT_KIND = f"{TRACE_ARTIFACT_KIND}-artifact"
HF_TEACHER_FORCED_ORACLE_SCHEMA = "riley.qwen3b-hf-eager-teacher-forced-generation.v1"
HF_TEACHER_FORCED_ARTIFACT_KIND = (
    "qwen2.5-3b-hf-eager-bf16-p2048-teacher-forced-generation"
)
QWEN3B_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
QWEN3B_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
QWEN3B_WORKLOAD_SHA256 = (
    "7a0a8fec31d45e397e1ec57335fa1c9de2d3da7daa9a32e9002a62763c05261e"
)
QWEN3B_WORKLOAD_CASE = "qwen3b-c8-p2048-o128"
NATIVE_D128_BACKEND_ID = (
    "riley.cuda.ragged-paged-attention.native-bf16-paged-split-gqa."
    "qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2"
)
STRICT_PROJECTION_BIAS_BACKEND = "strict-staged-v1"
FUSED_PROJECTION_BIAS_BACKEND = "cublaslt-bias-epilogue-experimental-v1"
TRACE_OUTPUT_ROWS, TOP_K = 9, 32
PROMPT_TOKEN_COUNT, ADDRESSABLE_TOKEN_COUNT, VOCABULARY_SIZE = 2_048, 151_665, 151_936
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TRACE_KEYS = {
    "schema_version",
    "artifact_kind",
    "performance_claim_eligible",
    "trace_contract",
    "model",
    "workload",
    "hf_teacher_forced_oracle",
    "modes",
}
CONTRACT_KEYS = {
    "scheduler_executor_replay",
    "http_transport_replayed",
    "sampling",
    "cache_on_scheduler_reference",
    "cache_off_control_reference",
    "projection_bias_modes",
    "native_d128_backend",
    "max_active_sequences",
    "batch_token_budget",
    "prefill_chunk_tokens",
    "max_sequence_tokens",
    "physical_kv_blocks",
    "trace_output_rows",
    "submitted_request_count",
    "scheduled_batch_size",
    "capacity_configuration_only",
    "request_max_new_tokens",
    "execution_graph_policy",
    "residual_rmsnorm",
    "execution_completion",
    "metadata_transport",
    "batch_shape_policy",
    "reduction_profile",
    "top32_order_comparison",
}
WORKLOAD_KEYS = {
    "sha256",
    "case",
    "prompt_token_count",
    "teacher_token_ids",
    "teacher_token_ids_le_u32_sha256",
}
ORACLE_KEYS = {
    "schema_version",
    "artifact_kind",
    "artifact_sha256",
    "teacher_token_ids_le_u32_sha256",
    "cache_off_sidecar",
    "cache_on_sidecar",
}
SIDECAR_BINDING_KEYS = {"basename", "sha256"}
MODE_KEYS = {
    "projection_bias_backend",
    "prefill_iteration_count",
    "decode_iteration_count",
    "first_hf_cache_off_selected_token_mismatch",
    "first_hf_cache_on_selected_token_mismatch",
    "rows",
}
ROW_KEYS = {
    "step",
    "scheduler_iteration_id",
    "kind",
    "iteration_input_token_count",
    "target_logical_length",
    "row_bf16_le_sha256",
    "addressable_bf16_le_sha256",
    "raw_argmax_token_id",
    "selected_token_id",
    "selected_logit_bf16_as_f32",
    "top_token_ids",
    "top_values_bf16_as_f32",
    "hf_cache_off",
    "hf_cache_on",
}
HF_ROW_KEYS = {
    "mode",
    "step",
    "call_input_token_count",
    "call_input_token_ids_le_u32_sha256",
    "context_token_count",
    "attention_mask_token_count",
    "position_start",
    "position_end",
    "teacher_input_token_id",
    "cache_length_before",
    "cache_length_after",
    "logits_bf16_le_sha256",
    "addressable_bf16_le_sha256",
    "raw_argmax_token_id",
    "selected_token_id",
    "cache_off_teacher_token_id",
    "selection_matches_cache_off_teacher",
    "selected_logit_bf16_as_f32",
    "top_token_ids",
    "top_values_bf16_as_f32",
    "raw_hash_matches",
    "selected_token_matches",
    "raw_argmax_matches",
    "top32_order_matches_bf16_numeric_tie_break",
    "top32_set_overlap",
}


class TraceCollectorError(ValueError):
    """A source log or marker payload violates the diagnostic contract."""


def _fail(path: str, message: str) -> NoReturn:
    raise TraceCollectorError(f"{path}: {message}")


def _duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceCollectorError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    raise TraceCollectorError(f"JSON non-finite value {value!r} is forbidden")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TraceCollectorError(f"JSON non-finite value {value!r} is forbidden")
    return result


def _object(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    missing, extra = expected - set(value), set(value) - expected
    if missing or extra:
        _fail(path, f"missing {sorted(missing)}; unexpected {sorted(extra)}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _sha256(value: Any, path: str) -> str:
    result = _string(value, path)
    if SHA256_RE.fullmatch(result) is None:
        _fail(path, "must be 64 lowercase hexadecimal characters")
    return result


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a boolean")
    return value


def _integer(value: Any, path: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, "must be a non-negative integer")
    if maximum is not None and value > maximum:
        _fail(path, f"must be <= {maximum}")
    return value


def _optional_integer(
    value: Any, path: str, *, maximum: int | None = None
) -> int | None:
    if value is None:
        return None
    return _integer(value, path, maximum=maximum)


def _number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(path, "must be a finite JSON number")
    return float(value)


def _equal(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        _fail(path, f"must equal {expected!r}")


def _u32_le_sha256(token_ids: list[int]) -> str:
    payload = b"".join(token_id.to_bytes(4, "little") for token_id in token_ids)
    return hashlib.sha256(payload).hexdigest()


def _top_k(
    ids_value: Any, values_value: Any, path: str, *, upper_bound: int
) -> tuple[list[int], list[float]]:
    if not isinstance(ids_value, list) or len(ids_value) != TOP_K:
        _fail(f"{path}.top_token_ids", f"must contain exactly {TOP_K} IDs")
    ids = [
        _integer(value, f"{path}.top_token_ids[{index}]", maximum=upper_bound - 1)
        for index, value in enumerate(ids_value)
    ]
    if len(set(ids)) != TOP_K:
        _fail(f"{path}.top_token_ids", "must be unique")
    if not isinstance(values_value, list) or len(values_value) != TOP_K:
        _fail(f"{path}.top_values_bf16_as_f32", f"must contain exactly {TOP_K} values")
    values = [
        _number(value, f"{path}.top_values_bf16_as_f32[{index}]")
        for index, value in enumerate(values_value)
    ]
    if any(left < right for left, right in zip(values, values[1:])):
        _fail(f"{path}.top_values_bf16_as_f32", "must be descending")
    if any(
        left_value == right_value and left_id >= right_id
        for (left_id, left_value), (right_id, right_value) in zip(
            zip(ids, values), zip(ids[1:], values[1:])
        )
    ):
        _fail(
            f"{path}.top_token_ids",
            "must order equal BF16 numeric values by ascending token ID",
        )
    return ids, values


def _validate_sidecar_binding(value: Any, path: str) -> None:
    binding = _object(value, SIDECAR_BINDING_KEYS, path)
    basename = _string(binding["basename"], f"{path}.basename")
    if Path(basename).name != basename or not basename.endswith(".safetensors"):
        _fail(f"{path}.basename", "must be a safetensors basename")
    _sha256(binding["sha256"], f"{path}.sha256")


def _validate_hf_row(
    hf_value: Any,
    native: dict[str, Any],
    step: int,
    teacher_token: int,
    previous_teacher_token: int | None,
    expected_mode: str,
    path: str,
) -> int:
    hf = _object(hf_value, HF_ROW_KEYS, path)
    _equal(_string(hf["mode"], f"{path}.mode"), expected_mode, f"{path}.mode")
    _equal(_integer(hf["step"], f"{path}.step"), step, f"{path}.step")
    context = PROMPT_TOKEN_COUNT + step
    if expected_mode == "cache-off":
        expected = {
            "call_input_token_count": context,
            "context_token_count": context,
            "attention_mask_token_count": context,
            "position_start": 0,
            "position_end": context - 1,
            "teacher_input_token_id": None,
            "cache_length_before": None,
            "cache_length_after": None,
        }
    else:
        if step == 0:
            expected = {
                "call_input_token_count": PROMPT_TOKEN_COUNT,
                "context_token_count": PROMPT_TOKEN_COUNT,
                "attention_mask_token_count": PROMPT_TOKEN_COUNT,
                "position_start": 0,
                "position_end": PROMPT_TOKEN_COUNT - 1,
                "teacher_input_token_id": None,
                "cache_length_before": 0,
                "cache_length_after": PROMPT_TOKEN_COUNT,
            }
        else:
            expected = {
                "call_input_token_count": 1,
                "context_token_count": context,
                "attention_mask_token_count": context,
                "position_start": context - 1,
                "position_end": context - 1,
                "teacher_input_token_id": previous_teacher_token,
                "cache_length_before": context - 1,
                "cache_length_after": context,
            }
            if previous_teacher_token is None:
                _fail(path, "cache-on decode lacks a prior teacher token")
    for key in (
        "call_input_token_count",
        "context_token_count",
        "attention_mask_token_count",
        "position_start",
        "position_end",
    ):
        _equal(_integer(hf[key], f"{path}.{key}"), expected[key], f"{path}.{key}")
    call_input_token_ids_sha256 = _sha256(
        hf["call_input_token_ids_le_u32_sha256"],
        f"{path}.call_input_token_ids_le_u32_sha256",
    )
    if expected_mode == "cache-on" and step > 0:
        if previous_teacher_token is None:
            _fail(path, "cache-on decode lacks a prior teacher token")
        _equal(
            call_input_token_ids_sha256,
            _u32_le_sha256([previous_teacher_token]),
            f"{path}.call_input_token_ids_le_u32_sha256",
        )
    _equal(
        _optional_integer(
            hf["teacher_input_token_id"],
            f"{path}.teacher_input_token_id",
            maximum=ADDRESSABLE_TOKEN_COUNT - 1,
        ),
        expected["teacher_input_token_id"],
        f"{path}.teacher_input_token_id",
    )
    _equal(
        _optional_integer(hf["cache_length_before"], f"{path}.cache_length_before"),
        expected["cache_length_before"],
        f"{path}.cache_length_before",
    )
    _equal(
        _optional_integer(hf["cache_length_after"], f"{path}.cache_length_after"),
        expected["cache_length_after"],
        f"{path}.cache_length_after",
    )
    hf_hash = _sha256(hf["logits_bf16_le_sha256"], f"{path}.logits_bf16_le_sha256")
    _sha256(hf["addressable_bf16_le_sha256"], f"{path}.addressable_bf16_le_sha256")
    hf_raw_argmax = _integer(
        hf["raw_argmax_token_id"],
        f"{path}.raw_argmax_token_id",
        maximum=VOCABULARY_SIZE - 1,
    )
    hf_selected = _integer(
        hf["selected_token_id"],
        f"{path}.selected_token_id",
        maximum=ADDRESSABLE_TOKEN_COUNT - 1,
    )
    _equal(
        _integer(
            hf["cache_off_teacher_token_id"],
            f"{path}.cache_off_teacher_token_id",
            maximum=ADDRESSABLE_TOKEN_COUNT - 1,
        ),
        teacher_token,
        f"{path}.cache_off_teacher_token_id",
    )
    selected_matches_teacher = _bool(
        hf["selection_matches_cache_off_teacher"],
        f"{path}.selection_matches_cache_off_teacher",
    )
    _equal(
        selected_matches_teacher,
        hf_selected == teacher_token,
        f"{path}.selection_matches_cache_off_teacher",
    )
    if expected_mode == "cache-off":
        _equal(hf_selected, teacher_token, f"{path}.selected_token_id")
    hf_logit = _number(
        hf["selected_logit_bf16_as_f32"], f"{path}.selected_logit_bf16_as_f32"
    )
    hf_ids, hf_values = _top_k(
        hf["top_token_ids"],
        hf["top_values_bf16_as_f32"],
        path,
        upper_bound=ADDRESSABLE_TOKEN_COUNT,
    )
    _equal(hf_selected, hf_ids[0], f"{path}.selected_token_id")
    _equal(hf_logit, hf_values[0], f"{path}.selected_logit_bf16_as_f32")
    expected_flags = {
        "raw_hash_matches": native["row_bf16_le_sha256"] == hf_hash,
        "selected_token_matches": native["selected_token_id"] == hf_selected,
        "raw_argmax_matches": native["raw_argmax_token_id"] == hf_raw_argmax,
        "top32_order_matches_bf16_numeric_tie_break": native["top_token_ids"] == hf_ids,
        "top32_set_overlap": len(set(native["top_token_ids"]) & set(hf_ids)),
    }
    for key, expected_value in expected_flags.items():
        actual = (
            _integer(hf[key], f"{path}.{key}", maximum=TOP_K)
            if key == "top32_set_overlap"
            else _bool(hf[key], f"{path}.{key}")
        )
        _equal(actual, expected_value, f"{path}.{key}")
    return hf_selected


def _validate_mode(
    value: Any, expected_backend: str, teacher_tokens: list[int], path: str
) -> None:
    mode = _object(value, MODE_KEYS, path)
    _equal(
        _string(mode["projection_bias_backend"], f"{path}.projection_bias_backend"),
        expected_backend,
        f"{path}.projection_bias_backend",
    )
    _equal(
        _integer(mode["prefill_iteration_count"], f"{path}.prefill_iteration_count"),
        64,
        f"{path}.prefill_iteration_count",
    )
    _equal(
        _integer(mode["decode_iteration_count"], f"{path}.decode_iteration_count"),
        TRACE_OUTPUT_ROWS - 1,
        f"{path}.decode_iteration_count",
    )
    rows = mode["rows"]
    if not isinstance(rows, list) or len(rows) != TRACE_OUTPUT_ROWS:
        _fail(f"{path}.rows", f"must contain exactly {TRACE_OUTPUT_ROWS} rows")
    cache_off_mismatch: int | None = None
    cache_on_mismatch: int | None = None
    last_iteration_id = 0
    for step, value in enumerate(rows):
        row_path = f"{path}.rows[{step}]"
        row = _object(value, ROW_KEYS, row_path)
        _equal(_integer(row["step"], f"{row_path}.step"), step, f"{row_path}.step")
        iteration = _integer(
            row["scheduler_iteration_id"], f"{row_path}.scheduler_iteration_id"
        )
        if iteration <= last_iteration_id:
            _fail(f"{row_path}.scheduler_iteration_id", "must be strictly increasing")
        last_iteration_id = iteration
        _equal(
            _string(row["kind"], f"{row_path}.kind"),
            "prefill" if step == 0 else "decode",
            f"{row_path}.kind",
        )
        _equal(
            _integer(
                row["iteration_input_token_count"],
                f"{row_path}.iteration_input_token_count",
            ),
            32 if step == 0 else 1,
            f"{row_path}.iteration_input_token_count",
        )
        _equal(
            _integer(row["target_logical_length"], f"{row_path}.target_logical_length"),
            PROMPT_TOKEN_COUNT + step,
            f"{row_path}.target_logical_length",
        )
        _sha256(row["row_bf16_le_sha256"], f"{row_path}.row_bf16_le_sha256")
        _sha256(
            row["addressable_bf16_le_sha256"], f"{row_path}.addressable_bf16_le_sha256"
        )
        _integer(
            row["raw_argmax_token_id"],
            f"{row_path}.raw_argmax_token_id",
            maximum=VOCABULARY_SIZE - 1,
        )
        selected = _integer(
            row["selected_token_id"],
            f"{row_path}.selected_token_id",
            maximum=ADDRESSABLE_TOKEN_COUNT - 1,
        )
        selected_logit = _number(
            row["selected_logit_bf16_as_f32"], f"{row_path}.selected_logit_bf16_as_f32"
        )
        ids, values = _top_k(
            row["top_token_ids"],
            row["top_values_bf16_as_f32"],
            row_path,
            upper_bound=ADDRESSABLE_TOKEN_COUNT,
        )
        _equal(selected, ids[0], f"{row_path}.selected_token_id")
        _equal(selected_logit, values[0], f"{row_path}.selected_logit_bf16_as_f32")
        cache_off_selected = _validate_hf_row(
            row["hf_cache_off"],
            row,
            step,
            teacher_tokens[step],
            None,
            "cache-off",
            f"{row_path}.hf_cache_off",
        )
        cache_on_selected = _validate_hf_row(
            row["hf_cache_on"],
            row,
            step,
            teacher_tokens[step],
            None if step == 0 else teacher_tokens[step - 1],
            "cache-on",
            f"{row_path}.hf_cache_on",
        )
        if cache_off_mismatch is None and selected != cache_off_selected:
            cache_off_mismatch = step
        if cache_on_mismatch is None and selected != cache_on_selected:
            cache_on_mismatch = step
    for field, expected in (
        ("first_hf_cache_off_selected_token_mismatch", cache_off_mismatch),
        ("first_hf_cache_on_selected_token_mismatch", cache_on_mismatch),
    ):
        reported = mode[field]
        if reported is not None:
            reported = _integer(
                reported, f"{path}.{field}", maximum=TRACE_OUTPUT_ROWS - 1
            )
        _equal(reported, expected, f"{path}.{field}")


def validate_trace(document: Any) -> dict[str, Any]:
    """Validate the non-performance numerical trace marker without mutation."""

    trace = _object(document, TRACE_KEYS, "trace")
    _equal(
        _string(trace["schema_version"], "trace.schema_version"),
        TRACE_SCHEMA_VERSION,
        "trace.schema_version",
    )
    _equal(
        _string(trace["artifact_kind"], "trace.artifact_kind"),
        TRACE_ARTIFACT_KIND,
        "trace.artifact_kind",
    )
    if _bool(trace["performance_claim_eligible"], "trace.performance_claim_eligible"):
        _fail(
            "trace.performance_claim_eligible",
            "must be false for a numerical diagnostic",
        )
    contract = _object(trace["trace_contract"], CONTRACT_KEYS, "trace.trace_contract")
    expected_contract: dict[str, Any] = {
        "scheduler_executor_replay": True,
        "http_transport_replayed": False,
        "sampling": "teacher-forced-cache-off-addressable-greedy-argmax-after-scheduler-commit",
        "cache_on_scheduler_reference": "step0-p2048-prefill;step>0-teacher_token_ids[step-1]-at-position-2048+step-1",
        "cache_off_control_reference": "full-prefix-cache-off-at-position-0-through-2047+step",
        "projection_bias_modes": [
            STRICT_PROJECTION_BIAS_BACKEND,
            FUSED_PROJECTION_BIAS_BACKEND,
        ],
        "native_d128_backend": NATIVE_D128_BACKEND_ID,
        "max_active_sequences": 8,
        "batch_token_budget": 32,
        "prefill_chunk_tokens": 32,
        "max_sequence_tokens": 2_176,
        "physical_kv_blocks": 1_088,
        "trace_output_rows": TRACE_OUTPUT_ROWS,
        "submitted_request_count": 1,
        "scheduled_batch_size": 1,
        "capacity_configuration_only": True,
        "request_max_new_tokens": TRACE_OUTPUT_ROWS,
        "execution_graph_policy": "disabled",
        "residual_rmsnorm": "separate",
        "execution_completion": "iteration-batch",
        "metadata_transport": "synchronous",
        "batch_shape_policy": "fixed-maximum",
        "reduction_profile": "canonical-v1",
        "top32_order_comparison": "bf16-numeric-descending-token-id-ascending-tie-break",
    }
    for key, expected in expected_contract.items():
        field_path = f"trace.trace_contract.{key}"
        if isinstance(expected, bool):
            actual = _bool(contract[key], field_path)
        elif isinstance(expected, int):
            actual = _integer(contract[key], field_path)
        elif isinstance(expected, str):
            actual = _string(contract[key], field_path)
        else:
            actual = contract[key]
        _equal(actual, expected, field_path)
    model = _object(trace["model"], {"id", "revision"}, "trace.model")
    _equal(_string(model["id"], "trace.model.id"), QWEN3B_MODEL_ID, "trace.model.id")
    _equal(
        _string(model["revision"], "trace.model.revision"),
        QWEN3B_MODEL_REVISION,
        "trace.model.revision",
    )
    workload = _object(trace["workload"], WORKLOAD_KEYS, "trace.workload")
    _equal(
        _sha256(workload["sha256"], "trace.workload.sha256"),
        QWEN3B_WORKLOAD_SHA256,
        "trace.workload.sha256",
    )
    _equal(
        _string(workload["case"], "trace.workload.case"),
        QWEN3B_WORKLOAD_CASE,
        "trace.workload.case",
    )
    _equal(
        _integer(workload["prompt_token_count"], "trace.workload.prompt_token_count"),
        PROMPT_TOKEN_COUNT,
        "trace.workload.prompt_token_count",
    )
    tokens_value = workload["teacher_token_ids"]
    if not isinstance(tokens_value, list) or len(tokens_value) != TRACE_OUTPUT_ROWS:
        _fail(
            "trace.workload.teacher_token_ids",
            f"must contain exactly {TRACE_OUTPUT_ROWS} IDs",
        )
    teacher_tokens = [
        _integer(
            token,
            f"trace.workload.teacher_token_ids[{index}]",
            maximum=ADDRESSABLE_TOKEN_COUNT - 1,
        )
        for index, token in enumerate(tokens_value)
    ]
    _equal(
        _sha256(
            workload["teacher_token_ids_le_u32_sha256"],
            "trace.workload.teacher_token_ids_le_u32_sha256",
        ),
        _u32_le_sha256(teacher_tokens),
        "trace.workload.teacher_token_ids_le_u32_sha256",
    )
    oracle = _object(
        trace["hf_teacher_forced_oracle"], ORACLE_KEYS, "trace.hf_teacher_forced_oracle"
    )
    _equal(
        _string(
            oracle["schema_version"], "trace.hf_teacher_forced_oracle.schema_version"
        ),
        HF_TEACHER_FORCED_ORACLE_SCHEMA,
        "trace.hf_teacher_forced_oracle.schema_version",
    )
    _equal(
        _string(
            oracle["artifact_kind"], "trace.hf_teacher_forced_oracle.artifact_kind"
        ),
        HF_TEACHER_FORCED_ARTIFACT_KIND,
        "trace.hf_teacher_forced_oracle.artifact_kind",
    )
    _sha256(oracle["artifact_sha256"], "trace.hf_teacher_forced_oracle.artifact_sha256")
    _sha256(
        oracle["teacher_token_ids_le_u32_sha256"],
        "trace.hf_teacher_forced_oracle.teacher_token_ids_le_u32_sha256",
    )
    _validate_sidecar_binding(
        oracle["cache_off_sidecar"], "trace.hf_teacher_forced_oracle.cache_off_sidecar"
    )
    _validate_sidecar_binding(
        oracle["cache_on_sidecar"], "trace.hf_teacher_forced_oracle.cache_on_sidecar"
    )
    modes = trace["modes"]
    if not isinstance(modes, list) or len(modes) != 2:
        _fail("trace.modes", "must contain strict and fused modes")
    _validate_mode(
        modes[0], STRICT_PROJECTION_BIAS_BACKEND, teacher_tokens, "trace.modes[0]"
    )
    _validate_mode(
        modes[1], FUSED_PROJECTION_BIAS_BACKEND, teacher_tokens, "trace.modes[1]"
    )
    return trace


def _read_stdout(path: Path) -> bytes:
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            raise TraceCollectorError(f"test stdout {path} must be a regular file")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise TraceCollectorError(
            f"could not open test stdout {path}: {error}"
        ) from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            after = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise TraceCollectorError(
                    f"test stdout {path} changed while it was opened"
                )
            payload = handle.read(MAX_TEST_STDOUT_BYTES + 1)
    except OSError as error:
        raise TraceCollectorError(
            f"could not read test stdout {path}: {error}"
        ) from error
    if len(payload) > MAX_TEST_STDOUT_BYTES:
        raise TraceCollectorError(
            f"test stdout {path} exceeds {MAX_TEST_STDOUT_BYTES} bytes"
        )
    return payload


def extract_trace_from_stdout(stdout: bytes) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TraceCollectorError("test stdout is not UTF-8") from error
    markers = [
        line[len(MARKER_PREFIX) :]
        for line in text.splitlines()
        if line.startswith(MARKER_PREFIX)
    ]
    if len(markers) != 1:
        raise TraceCollectorError(
            f"test stdout must contain exactly one {MARKER_PREFIX!r} marker; found {len(markers)}"
        )
    payload = markers[0]
    if not payload or payload != payload.strip():
        raise TraceCollectorError(
            "trace marker must contain compact JSON without outer whitespace"
        )
    try:
        return validate_trace(
            json.loads(
                payload,
                object_pairs_hook=_duplicate_key,
                parse_constant=_nonfinite,
                parse_float=_finite_float,
            )
        )
    except (json.JSONDecodeError, TraceCollectorError) as error:
        raise TraceCollectorError(
            f"trace marker is not strict JSON: {error}"
        ) from error


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_exclusive(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
        )
    except FileExistsError as error:
        raise TraceCollectorError(
            f"refusing to overwrite existing output: {path}"
        ) from error
    except OSError as error:
        raise TraceCollectorError(f"could not create output {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(document))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def collect_trace(*, test_stdout_path: Path, output_path: Path) -> dict[str, Any]:
    stdout = _read_stdout(test_stdout_path)
    trace = extract_trace_from_stdout(stdout)
    document = {
        "schema_version": COLLECTED_SCHEMA_VERSION,
        "artifact_kind": COLLECTED_ARTIFACT_KIND,
        "performance_claim_eligible": False,
        "source_log_sha256": hashlib.sha256(stdout).hexdigest(),
        "trace": trace,
    }
    _write_exclusive(output_path, document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect one native-D128 numerical trace artifact."
    )
    parser.add_argument("--test-stdout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        document = collect_trace(
            test_stdout_path=args.test_stdout.expanduser(),
            output_path=args.output.expanduser(),
        )
    except TraceCollectorError as error:
        print(f"collect_qwen3b_native_d128_trace: error: {error}", file=sys.stderr)
        return 2
    print(
        f"collected native-D128 diagnostic trace to {args.output} (source_log_sha256={document['source_log_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
