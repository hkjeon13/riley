"""HF FP32-versus-BF16 oracle calibration evidence (never an E0 candidate gate)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from .calibration import (
    BF16_ORACLE_KIND,
    CALIBRATION_PROMPT_COUNT,
    CALIBRATION_SCHEMA_VERSION,
    CALIBRATION_THRESHOLDS,
    CALIBRATION_TOP_K,
    CROSS_CACHE_EXACT_WINDOW,
    FP32_ORACLE_KIND,
    HF_ORACLE_REDUCTION_VARIANT,
    TENSOR_NAMES,
    CalibrationError,
    SidecarLoader,
    _default_sidecar_loader,
    _expect_exact_keys,
    _expect_finite,
    _expect_int,
    _expect_object,
    _expect_sha,
    _expect_string,
    _load_verified_sidecar,
    _metric_record,
    _validate_report_semantic,
    canonical_json_bytes,
    first_divergence,
    load_calibration_manifest,
    metrics_pass,
    parse_utc,
    ranked_top_k,
    sha256_file,
    tensor_values,
    utc_text,
    validate_calibration_manifest,
    verify_manifest_sources,
)
from .constants import (
    MODEL_CONFIG_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PRIMARY_ENVIRONMENT_ID,
    TOKENIZER_SHA256,
)
from .environment import environment_comparability_signature

ORACLE_CALIBRATION_REPORT_KIND = "hf-oracle-calibration"
ORACLE_CALIBRATION_GATE_ID = "smollm2-hf-fp32-bf16-calibration-v2"


def _case_map(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {case["prompt_id"]: case for case in manifest["cases"]}


def _oracle_bindings(
    fp32: Mapping[str, object], bf16: Mapping[str, object]
) -> dict[str, object]:
    for key in (
        "model_id",
        "model_revision",
        "config_sha256",
        "weights_sha256",
        "tokenizer_sha256",
        "tokenizer_files_sha256",
        "max_context_tokens",
        "eos_token_ids",
        "oracle_reduction_variant",
        "required_candidate_reduction_variants",
    ):
        if fp32["contract"][key] != bf16["contract"][key]:
            raise CalibrationError(f"FP32/BF16 oracle {key} differs")
    for key in (
        "sources",
        "git_revision",
        "git_dirty",
        "git_status_sha256",
        "environment_id",
    ):
        if fp32["provenance"][key] != bf16["provenance"][key]:
            raise CalibrationError(f"FP32/BF16 oracle provenance {key} differs")
    if environment_comparability_signature(
        fp32["provenance"]["observed_environment"]
    ) != environment_comparability_signature(
        bf16["provenance"]["observed_environment"]
    ):
        raise CalibrationError(
            "FP32/BF16 oracle provenance observed environment differs"
        )
    return {
        "model_id": fp32["contract"]["model_id"],
        "model_revision": fp32["contract"]["model_revision"],
        "config_sha256": fp32["contract"]["config_sha256"],
        "weights_sha256": fp32["contract"]["weights_sha256"],
        "tokenizer_sha256": fp32["contract"]["tokenizer_sha256"],
        "matrix_sha256": fp32["provenance"]["sources"]["matrix"]["sha256"],
        "prompts_sha256": fp32["provenance"]["sources"]["prompts"]["sha256"],
        "gate_manifest_sha256": fp32["provenance"]["sources"]["gate_manifest"]["sha256"],
        "dependency_lock_sha256": fp32["provenance"]["sources"]["dependency_lock"]["sha256"],
        "lane_manifest_sha256": fp32["provenance"]["sources"]["lane_manifest"]["sha256"],
        "environment_sha256": fp32["provenance"]["sources"]["environment"]["sha256"],
        "environment_id": fp32["provenance"]["environment_id"],
        "git_revision": fp32["provenance"]["git_revision"],
        "git_status_sha256": fp32["provenance"]["git_status_sha256"],
    }


def _oracle_semantic_self_check(
    prompt_id: str,
    case: Mapping[str, object],
    tensors: Mapping[str, object],
) -> dict[str, object]:
    variant_id = str(HF_ORACLE_REDUCTION_VARIANT["variant_id"])
    variant = case["variants"][variant_id]
    semantic = variant["semantic"]
    logits = tensor_values(tensors[variant["tensors"]["final_logits"]["key"]])
    ranked = ranked_top_k(logits, CALIBRATION_TOP_K)
    metadata_derived = (
        semantic["top_1_token_id"] == ranked[0]
        and semantic["top_k_token_id_set"] == sorted(ranked)
    )
    cache_on = semantic["cache_on"]["generated_token_ids"]
    cache_off = semantic["cache_off"]["generated_token_ids"]
    top_1_matches_paths = bool(cache_on and cache_off) and (
        cache_on[0] == ranked[0] and cache_off[0] == ranked[0]
    )
    divergence = first_divergence(cache_on, cache_off)
    divergence_recomputed = (
        semantic["cross_cache_first_divergence_step"] == divergence
    )
    exact_window = divergence is None or divergence >= CROSS_CACHE_EXACT_WINDOW
    result = {
        "top_k_metadata_tensor_derived": metadata_derived,
        "top_1_matches_cache_paths": top_1_matches_paths,
        "cross_cache_first_divergence_step": divergence,
        "cross_cache_exact_window": CROSS_CACHE_EXACT_WINDOW,
        "cross_cache_exact_window_match": exact_window,
        "divergence_metadata_recomputed": divergence_recomputed,
    }
    result["pass"] = all(
        bool(result[key])
        for key in (
            "top_k_metadata_tensor_derived",
            "top_1_matches_cache_paths",
            "cross_cache_exact_window_match",
            "divergence_metadata_recomputed",
        )
    )
    if not result["pass"] and not prompt_id:
        raise CalibrationError("oracle semantic self-check requires prompt identity")
    return result


def _worst_metrics(records: Sequence[Mapping[str, object]], tensor_name: str) -> dict[str, float]:
    metrics = [record["numeric"][tensor_name]["metrics"] for record in records]
    return {
        "max_abs": max(metric["max_abs"] for metric in metrics),
        "mean_abs": max(metric["mean_abs"] for metric in metrics),
        "max_relative": max(metric["max_relative"] for metric in metrics),
        "mean_relative": max(metric["mean_relative"] for metric in metrics),
        "cosine_similarity": min(metric["cosine_similarity"] for metric in metrics),
    }


def compare_hf_oracles(
    *,
    fp32_manifest: Mapping[str, object],
    fp32_manifest_path: Path,
    bf16_manifest: Mapping[str, object],
    bf16_manifest_path: Path,
    repo_root: Path,
    created_at: datetime,
    sidecar_loader: SidecarLoader | None = None,
) -> dict[str, object]:
    for manifest in (fp32_manifest, bf16_manifest):
        validate_calibration_manifest(manifest)
        verify_manifest_sources(manifest, repo_root)
    if fp32_manifest["artifact_kind"] != FP32_ORACLE_KIND:
        raise CalibrationError("FP32 input is not the numeric oracle")
    if bf16_manifest["artifact_kind"] != BF16_ORACLE_KIND:
        raise CalibrationError("BF16 input is not the semantic oracle")
    bindings = _oracle_bindings(fp32_manifest, bf16_manifest)
    loader = sidecar_loader or _default_sidecar_loader
    fp32_tensors = _load_verified_sidecar(fp32_manifest, fp32_manifest_path, loader)
    bf16_tensors = _load_verified_sidecar(bf16_manifest, bf16_manifest_path, loader)
    fp32_cases = _case_map(fp32_manifest)
    bf16_cases = _case_map(bf16_manifest)
    if list(fp32_cases) != list(bf16_cases):
        raise CalibrationError("FP32/BF16 prompt order differs")
    variant_id = str(HF_ORACLE_REDUCTION_VARIANT["variant_id"])
    case_reports: list[dict[str, object]] = []
    for prompt_id, fp32_case in fp32_cases.items():
        bf16_case = bf16_cases[prompt_id]
        for key in (
            "prompt_text_sha256",
            "prompt_metadata",
            "input_token_ids_sha256",
            "input_first_token_id",
            "input_token_count",
            "hidden_anchor_positions",
        ):
            if fp32_case[key] != bf16_case[key]:
                raise CalibrationError(f"{prompt_id}: FP32/BF16 {key} differs")
        fp32_variant = fp32_case["variants"][variant_id]
        bf16_variant = bf16_case["variants"][variant_id]
        numeric: dict[str, object] = {}
        for tensor_name in TENSOR_NAMES:
            fp32_ref = fp32_variant["tensors"][tensor_name]
            bf16_ref = bf16_variant["tensors"][tensor_name]
            if fp32_ref["shape"] != bf16_ref["shape"]:
                raise CalibrationError(f"{prompt_id}: {tensor_name} shape differs")
            numeric[tensor_name] = _metric_record(
                fp32_tensors[fp32_ref["key"]],
                bf16_tensors[bf16_ref["key"]],
                tensor_name,
            )
        semantic = _oracle_semantic_self_check(prompt_id, bf16_case, bf16_tensors)
        case_reports.append(
            {
                "prompt_id": prompt_id,
                "numeric": numeric,
                "semantic_self_check": semantic,
                "pass": all(item["pass"] for item in numeric.values()) and semantic["pass"],
            }
        )
    aggregate = {
        tensor_name: {
            "metrics": _worst_metrics(case_reports, tensor_name),
            "pass": metrics_pass(
                _worst_metrics(case_reports, tensor_name),
                CALIBRATION_THRESHOLDS[tensor_name],
            ),
        }
        for tensor_name in TENSOR_NAMES
    }
    numeric_pass = all(case["numeric"][name]["pass"] for case in case_reports for name in TENSOR_NAMES)
    semantic_pass = all(case["semantic_self_check"]["pass"] for case in case_reports)
    report: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "report_kind": ORACLE_CALIBRATION_REPORT_KIND,
        "gate_id": ORACLE_CALIBRATION_GATE_ID,
        "created_at": utc_text(created_at),
        "status": "pass" if numeric_pass and semantic_pass else "fail",
        "e0_candidate_evidence": False,
        "roles": {
            "fp32": "numeric-reference",
            "bf16": "numeric-calibration-and-semantic-oracle-self-check",
        },
        "thresholds": CALIBRATION_THRESHOLDS,
        "inputs": {
            "fp32_manifest_sha256": sha256_file(fp32_manifest_path),
            "bf16_manifest_sha256": sha256_file(bf16_manifest_path),
            "fp32_sidecar_sha256": fp32_manifest["sidecar"]["sha256"],
            "bf16_sidecar_sha256": bf16_manifest["sidecar"]["sha256"],
        },
        "bindings": bindings,
        "summary": {
            "case_count": len(case_reports),
            "failure_count": sum(not case["pass"] for case in case_reports),
            "numeric_pass": numeric_pass,
            "semantic_self_check_pass": semantic_pass,
            "aggregate_numeric": aggregate,
        },
        "cases": case_reports,
    }
    _validate_oracle_report_structure(report)
    return report


def _validate_metric_record(value: object, tensor_name: str, path: str) -> dict[str, object]:
    record = _expect_object(value, path)
    _expect_exact_keys(record, {"metrics", "pass"}, path)
    metrics = _expect_object(record["metrics"], f"{path}.metrics")
    _expect_exact_keys(
        metrics,
        {"max_abs", "mean_abs", "max_relative", "mean_relative", "cosine_similarity"},
        f"{path}.metrics",
    )
    normalized = {
        key: _expect_finite(raw, f"{path}.metrics.{key}")
        for key, raw in metrics.items()
    }
    expected = metrics_pass(normalized, CALIBRATION_THRESHOLDS[tensor_name])
    if record["pass"] is not expected:
        raise CalibrationError(f"{path}.pass: inconsistent")
    return record


def _validate_semantic_self(value: object, path: str) -> dict[str, object]:
    record = _expect_object(value, path)
    _expect_exact_keys(
        record,
        {
            "top_k_metadata_tensor_derived",
            "top_1_matches_cache_paths",
            "cross_cache_first_divergence_step",
            "cross_cache_exact_window",
            "cross_cache_exact_window_match",
            "divergence_metadata_recomputed",
            "pass",
        },
        path,
    )
    for key in (
        "top_k_metadata_tensor_derived",
        "top_1_matches_cache_paths",
        "cross_cache_exact_window_match",
        "divergence_metadata_recomputed",
        "pass",
    ):
        if not isinstance(record[key], bool):
            raise CalibrationError(f"{path}.{key}: expected boolean")
    if record["cross_cache_first_divergence_step"] is not None:
        _expect_int(record["cross_cache_first_divergence_step"], f"{path}.cross_cache_first_divergence_step")
    if record["cross_cache_exact_window"] != CROSS_CACHE_EXACT_WINDOW:
        raise CalibrationError(f"{path}.cross_cache_exact_window: changed")
    expected = all(
        record[key]
        for key in (
            "top_k_metadata_tensor_derived",
            "top_1_matches_cache_paths",
            "cross_cache_exact_window_match",
            "divergence_metadata_recomputed",
        )
    )
    if record["pass"] is not expected:
        raise CalibrationError(f"{path}.pass: inconsistent")
    return record


def _validate_oracle_report_structure(report: Mapping[str, object]) -> None:
    root = _expect_object(report, "oracle_report")
    _expect_exact_keys(
        root,
        {
            "schema_version",
            "report_kind",
            "gate_id",
            "created_at",
            "status",
            "e0_candidate_evidence",
            "roles",
            "thresholds",
            "inputs",
            "bindings",
            "summary",
            "cases",
        },
        "oracle_report",
    )
    if root["gate_id"] != ORACLE_CALIBRATION_GATE_ID:
        raise CalibrationError("oracle_report.gate_id: immutable mismatch")
    if (
        root["schema_version"] != CALIBRATION_SCHEMA_VERSION
        or root["report_kind"] != ORACLE_CALIBRATION_REPORT_KIND
        or root["e0_candidate_evidence"] is not False
    ):
        raise CalibrationError("oracle_report: identity or E0 role mismatch")
    parse_utc(root["created_at"])
    if root["roles"] != {
        "fp32": "numeric-reference",
        "bf16": "numeric-calibration-and-semantic-oracle-self-check",
    } or root["thresholds"] != CALIBRATION_THRESHOLDS:
        raise CalibrationError("oracle_report: roles or predeclared thresholds changed")
    inputs = _expect_object(root["inputs"], "oracle_report.inputs")
    _expect_exact_keys(
        inputs,
        {"fp32_manifest_sha256", "bf16_manifest_sha256", "fp32_sidecar_sha256", "bf16_sidecar_sha256"},
        "oracle_report.inputs",
    )
    for key, value in inputs.items():
        _expect_sha(value, f"oracle_report.inputs.{key}")
    bindings = _expect_object(root["bindings"], "oracle_report.bindings")
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
            "dependency_lock_sha256",
            "lane_manifest_sha256",
            "environment_sha256",
            "environment_id",
            "git_revision",
            "git_status_sha256",
        },
        "oracle_report.bindings",
    )
    if (
        bindings["model_id"] != MODEL_ID
        or bindings["model_revision"] != MODEL_REVISION
        or bindings["config_sha256"] != MODEL_CONFIG_SHA256
        or bindings["weights_sha256"] != MODEL_WEIGHTS_SHA256
        or bindings["tokenizer_sha256"] != TOKENIZER_SHA256
        or bindings["environment_id"] != PRIMARY_ENVIRONMENT_ID
    ):
        raise CalibrationError("oracle_report.bindings: immutable identity differs")
    for key in (
        "config_sha256",
        "tokenizer_sha256",
        "matrix_sha256",
        "prompts_sha256",
        "gate_manifest_sha256",
        "dependency_lock_sha256",
        "lane_manifest_sha256",
        "environment_sha256",
        "git_status_sha256",
    ):
        _expect_sha(bindings[key], f"oracle_report.bindings.{key}")
    git_revision = _expect_string(
        bindings["git_revision"], "oracle_report.bindings.git_revision"
    )
    if not re.fullmatch(r"[0-9a-f]{40,64}", git_revision):
        raise CalibrationError("oracle_report.bindings.git_revision: invalid")
    cases = root["cases"]
    if not isinstance(cases, list) or len(cases) != CALIBRATION_PROMPT_COUNT:
        raise CalibrationError(
            "oracle_report.cases: expected full "
            f"{CALIBRATION_PROMPT_COUNT}-prompt corpus"
        )
    seen: set[str] = set()
    for index, raw_case in enumerate(cases):
        path = f"oracle_report.cases[{index}]"
        case = _expect_object(raw_case, path)
        _expect_exact_keys(case, {"prompt_id", "numeric", "semantic_self_check", "pass"}, path)
        prompt_id = _expect_string(case["prompt_id"], f"{path}.prompt_id")
        if prompt_id in seen:
            raise CalibrationError(f"{path}.prompt_id: duplicate")
        seen.add(prompt_id)
        numeric = _expect_object(case["numeric"], f"{path}.numeric")
        if set(numeric) != set(TENSOR_NAMES):
            raise CalibrationError(f"{path}.numeric: tensor set differs")
        for name in TENSOR_NAMES:
            _validate_metric_record(numeric[name], name, f"{path}.numeric.{name}")
        semantic = _validate_semantic_self(case["semantic_self_check"], f"{path}.semantic_self_check")
        expected = all(numeric[name]["pass"] for name in TENSOR_NAMES) and semantic["pass"]
        if case["pass"] is not expected:
            raise CalibrationError(f"{path}.pass: inconsistent")
    summary = _expect_object(root["summary"], "oracle_report.summary")
    _expect_exact_keys(
        summary,
        {"case_count", "failure_count", "numeric_pass", "semantic_self_check_pass", "aggregate_numeric"},
        "oracle_report.summary",
    )
    if summary["case_count"] != len(cases) or summary["failure_count"] != sum(not case["pass"] for case in cases):
        raise CalibrationError("oracle_report.summary: counts inconsistent")
    aggregate = _expect_object(summary["aggregate_numeric"], "oracle_report.summary.aggregate_numeric")
    if set(aggregate) != set(TENSOR_NAMES):
        raise CalibrationError("oracle_report.summary.aggregate_numeric: tensor set differs")
    for name in TENSOR_NAMES:
        record = _validate_metric_record(aggregate[name], name, f"oracle_report.summary.aggregate_numeric.{name}")
        if record["metrics"] != _worst_metrics(cases, name):
            raise CalibrationError(f"oracle_report.summary.aggregate_numeric.{name}: not recomputed")
    numeric_pass = all(case["numeric"][name]["pass"] for case in cases for name in TENSOR_NAMES)
    semantic_pass = all(case["semantic_self_check"]["pass"] for case in cases)
    if summary["numeric_pass"] is not numeric_pass or summary["semantic_self_check_pass"] is not semantic_pass:
        raise CalibrationError("oracle_report.summary: booleans inconsistent")
    expected_status = "pass" if numeric_pass and semantic_pass else "fail"
    if root["status"] != expected_status:
        raise CalibrationError("oracle_report.status: inconsistent")


def load_oracle_calibration_report(path: Path) -> dict[str, object]:
    from .calibration import _load_json_object

    report = _load_json_object(path, "oracle calibration report")
    _validate_oracle_report_structure(report)
    return report


def replay_validate_oracle_report(
    *,
    report: Mapping[str, object],
    fp32_manifest: Mapping[str, object],
    fp32_manifest_path: Path,
    bf16_manifest: Mapping[str, object],
    bf16_manifest_path: Path,
    repo_root: Path,
    sidecar_loader: SidecarLoader | None = None,
) -> None:
    _validate_oracle_report_structure(report)
    expected = compare_hf_oracles(
        fp32_manifest=fp32_manifest,
        fp32_manifest_path=fp32_manifest_path,
        bf16_manifest=bf16_manifest,
        bf16_manifest_path=bf16_manifest_path,
        repo_root=repo_root,
        created_at=parse_utc(report["created_at"]),
        sidecar_loader=sidecar_loader,
    )
    if canonical_json_bytes(report) != canonical_json_bytes(expected):
        raise CalibrationError("oracle calibration report differs from comparator replay")
