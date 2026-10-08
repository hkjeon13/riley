#!/usr/bin/env python3
"""Pure final-candidate contract for the optimizer canonical-E0 report.

This module deliberately has no release-tool imports.  It is shared by the
legacy final candidate checker and the held-FD Gate E optimizer component, so
the component cannot grant optimizer-E0 status to a report that the final
candidate checker would reject.  Raw evidence replay remains owned by
``check_optimization_evidence``; this module only validates the closed report
semantic contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, NoReturn


CONTRACT_VERSION = "riley.optimizer-e0-final-report-contract.v1"
GATE_ID = "pr15-iteration-command-batch-exact-v1"
FIXED37_PRODUCTION_BATCH_GATE_ID = "pr16-fixed37-production-batch-e0-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
PLACEHOLDER_RE = re.compile(
    r"(?:placeholder|replace[-_ ]?me|sha256[-_ ]?of|\btodo\b|<[^>]+>)",
    re.IGNORECASE,
)

EXPECTED_TOKENS = (
    4052, 2025, 284, 965, 6497, 288, 1492, 418,
    260, 16438, 30, 198, 198, 504, 16438, 314,
)
TEST_IDS = (
    "cuda-compile-only",
    "workspace-all-features-all-targets",
    "command-batch-lifecycle",
    "command-batch-resource-ledger",
    "smollm2-multi-step-greedy-exact",
    "fixed37-production-batch-e0",
)
EXPECTED_MODEL = {
    "model_id": "HuggingFaceTB/SmolLM2-135M",
    "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "dtype": "bf16",
    "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    "tokenizer_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
}
EXPECTED_IMPLEMENTATIONS = {
    "baseline": "per-operation",
    "candidate": "iteration-batch",
    "residual_rmsnorm": "separate",
    "rollback": "--execution-completion per-operation",
}
EXPECTED_FIXED37_FIXTURE_SHA256 = (
    "87333a1859be45a2f8e7563d898dde5e64256ccc03ca4da3cab90def07dd3c95"
)
EXPECTED_FIXED37_TOKEN_IDS_SHA256 = (
    "9e38488c0d41dae4a28e7e262baf772f2c643e9f8a9c57941a9e47aaec77ac5c"
)
FIXED37_CACHED_GROWING_COSINE_MIN = 0.997_903_530_549_539_3
FIXED37_CACHED_GROWING_MAX_ABS_MAX = 5.852_936_458_587_647
FIXED37_CACHED_GROWING_MEAN_ABS_MAX = 1.151_280_319_263_363

_POLICY = {
    "contract_version": CONTRACT_VERSION,
    "gate_id": GATE_ID,
    "fixed37_gate_id": FIXED37_PRODUCTION_BATCH_GATE_ID,
    "test_ids": TEST_IDS,
    "expected_tokens": EXPECTED_TOKENS,
    "expected_model": EXPECTED_MODEL,
    "expected_implementations": EXPECTED_IMPLEMENTATIONS,
    "fixed37": {
        "fixture_sha256": EXPECTED_FIXED37_FIXTURE_SHA256,
        "token_ids_sha256": EXPECTED_FIXED37_TOKEN_IDS_SHA256,
        "cosine_min": FIXED37_CACHED_GROWING_COSINE_MIN,
        "max_abs_max": FIXED37_CACHED_GROWING_MAX_ABS_MAX,
        "mean_abs_max": FIXED37_CACHED_GROWING_MEAN_ABS_MAX,
    },
}
POLICY_SHA256 = hashlib.sha256(
    json.dumps(_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class OptimizerE0SemanticContractError(ValueError):
    """The optimizer report does not satisfy the immutable final contract."""


def _fail(path: str, message: str) -> NoReturn:
    raise OptimizerE0SemanticContractError(f"{path}: {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _reject_placeholders(value: Any, path: str) -> None:
    """Match the legacy final checker's recursive placeholder rejection."""

    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if PLACEHOLDER_RE.search(value):
            _fail(path, "contains a placeholder marker")
        if value in {"0" * 40, "0" * 64, f"sha256:{'0' * 64}"}:
            _fail(path, "all-zero placeholder value is forbidden")


def _exact(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    result = _object(value, path)
    missing = sorted(keys - set(result))
    extra = sorted(set(result) - keys)
    if missing or extra:
        _fail(path, f"closed object mismatch; missing={missing}, unexpected={extra}")
    return result


def _string(value: Any, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    if PLACEHOLDER_RE.search(value):
        _fail(path, "contains a placeholder marker")
    if pattern is not None and pattern.fullmatch(value) is None:
        _fail(path, "has invalid format")
    return value


def _sha256(value: Any, path: str) -> str:
    digest = _string(value, path, SHA256_RE)
    if digest == "0" * 64:
        _fail(path, "all-zero placeholder digest is forbidden")
    return digest


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be a finite number")
    if minimum is not None and result < minimum:
        _fail(path, f"must be >= {minimum}")
    return result


def _test(value: Any, path: str, test_id: str, expected: Mapping[str, Any]) -> str:
    row = _exact(value, {"id", "result", "log_sha256", *expected}, path)
    if row["id"] != test_id or row["result"] != "passed":
        _fail(path, "test id/result mismatch")
    for key, expected_value in expected.items():
        if row[key] != expected_value:
            _fail(f"{path}.{key}", f"must be {expected_value!r}")
    return _sha256(row["log_sha256"], f"{path}.log_sha256")


def validate_final_candidate_report(
    report: Mapping[str, Any],
    *,
    source_revision: str,
    source_archive_sha256: str,
) -> str:
    """Validate the closed optimizer-E0 report and return its image digest.

    The caller supplies identity values from an independently pinned source
    boundary.  They are not inferred from the self-authored report.
    """

    _reject_placeholders(report, "optimizer report")
    row = _exact(
        report,
        {
            "schema_version", "gate_id", "recorded_at_utc", "status", "semantic_class",
            "source", "build", "gpu", "model", "implementations", "tests",
        },
        "optimizer report",
    )
    if row["schema_version"] != 1 or row["gate_id"] != GATE_ID:
        _fail("optimizer report", "optimizer equivalence schema/gate mismatch")
    if row["status"] != "passed" or row["semantic_class"] != "E0":
        _fail("optimizer report", "optimizer equivalence must be a passed E0 gate")
    _string(row["recorded_at_utc"], "optimizer report.recorded_at_utc")
    source = _exact(
        row["source"],
        {"git_commit", "git_dirty", "archive_sha256"},
        "optimizer report.source",
    )
    if source != {
        "git_commit": source_revision,
        "git_dirty": False,
        "archive_sha256": source_archive_sha256,
    }:
        _fail("optimizer report.source", "does not exactly match candidate source")
    build = _exact(
        row["build"],
        {
            "container_image_sha256", "network", "cargo_locked", "cargo_offline",
            "rustc", "cuda_toolkit", "cuda_architecture",
        },
        "optimizer report.build",
    )
    profile_image_sha256 = _sha256(
        build["container_image_sha256"],
        "optimizer report.build.container_image_sha256",
    )
    for key, expected in {
        "network": "none",
        "cargo_locked": True,
        "cargo_offline": True,
        "cuda_architecture": "89",
    }.items():
        if build[key] != expected:
            _fail(f"optimizer report.build.{key}", f"must be {expected!r}")
    _string(build["rustc"], "optimizer report.build.rustc")
    _string(build["cuda_toolkit"], "optimizer report.build.cuda_toolkit")
    gpu = _exact(
        row["gpu"],
        {"model", "uuid", "pci_bus_id", "compute_capability", "vram_mib", "driver_version"},
        "optimizer report.gpu",
    )
    for key in ("model", "uuid", "pci_bus_id", "compute_capability", "driver_version"):
        _string(gpu[key], f"optimizer report.gpu.{key}")
    if (
        not isinstance(gpu["vram_mib"], int)
        or isinstance(gpu["vram_mib"], bool)
        or gpu["vram_mib"] <= 0
    ):
        _fail("optimizer report.gpu.vram_mib", "must be a positive integer")
    model = _exact(
        row["model"],
        {"model_id", "revision", "dtype", "manifest_sha256", "weights_sha256", "tokenizer_sha256"},
        "optimizer report.model",
    )
    for key, expected in EXPECTED_MODEL.items():
        if model[key] != expected:
            _fail(f"optimizer report.model.{key}", f"must be {expected!r}")
    _sha256(model["manifest_sha256"], "optimizer report.model.manifest_sha256")
    implementations = _exact(
        row["implementations"],
        {"baseline", "candidate", "residual_rmsnorm", "rollback"},
        "optimizer report.implementations",
    )
    if implementations != EXPECTED_IMPLEMENTATIONS:
        _fail("optimizer report.implementations", "runtime flag/rollback contract mismatch")
    tests = row["tests"]
    if not isinstance(tests, list) or len(tests) != len(TEST_IDS):
        _fail("optimizer report.tests", "exact optimizer test inventory is required")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tests):
        test = _object(raw, f"optimizer report.tests[{index}]")
        test_id = _string(test.get("id"), f"optimizer report.tests[{index}].id", ID_RE)
        if test_id in by_id:
            _fail(f"optimizer report.tests[{index}].id", "duplicate test id")
        by_id[test_id] = test
    if set(by_id) != set(TEST_IDS):
        _fail("optimizer report.tests", f"test id set mismatch: {sorted(by_id)}")
    fixed37_row = by_id["fixed37-production-batch-e0"]
    fixed37_compile_log_sha256 = _sha256(
        fixed37_row.get("compile_log_sha256"),
        "optimizer report.tests.fixed37-production-batch-e0.compile_log_sha256",
    )
    fixed37_test_binary_sha256 = _sha256(
        fixed37_row.get("test_binary_sha256"),
        "optimizer report.tests.fixed37-production-batch-e0.test_binary_sha256",
    )
    fixed37_worst_cosine = _finite_number(
        fixed37_row.get("fixed_cached_growing_worst_cosine"),
        "optimizer report.tests.fixed37-production-batch-e0.fixed_cached_growing_worst_cosine",
        minimum=0.0,
    )
    fixed37_worst_max_abs = _finite_number(
        fixed37_row.get("fixed_cached_growing_worst_max_abs"),
        "optimizer report.tests.fixed37-production-batch-e0.fixed_cached_growing_worst_max_abs",
        minimum=0.0,
    )
    fixed37_worst_mean_abs = _finite_number(
        fixed37_row.get("fixed_cached_growing_worst_mean_abs"),
        "optimizer report.tests.fixed37-production-batch-e0.fixed_cached_growing_worst_mean_abs",
        minimum=0.0,
    )
    if (
        fixed37_worst_cosine < FIXED37_CACHED_GROWING_COSINE_MIN
        or fixed37_worst_max_abs > FIXED37_CACHED_GROWING_MAX_ABS_MAX
        or fixed37_worst_mean_abs > FIXED37_CACHED_GROWING_MEAN_ABS_MAX
    ):
        _fail(
            "optimizer report.tests.fixed37-production-batch-e0",
            "cached/growing metrics exceed the immutable E0 bounds",
        )
    _ = {
        "cuda-compile-only": _test(
            by_id["cuda-compile-only"],
            "optimizer report.tests.cuda-compile-only",
            "cuda-compile-only",
            {},
        ),
        "workspace-all-features-all-targets": _test(
            by_id["workspace-all-features-all-targets"],
            "optimizer report.tests.workspace-all-features-all-targets",
            "workspace-all-features-all-targets",
            {},
        ),
        "command-batch-lifecycle": _test(
            by_id["command-batch-lifecycle"],
            "optimizer report.tests.command-batch-lifecycle",
            "command-batch-lifecycle",
            {"one_shot_finish": True, "drop_restores_stream": True},
        ),
        "command-batch-resource-ledger": _test(
            by_id["command-batch-resource-ledger"],
            "optimizer report.tests.command-batch-resource-ledger",
            "command-batch-resource-ledger",
            {
                "validation_fail_closed": True,
                "queued_chain_raw_byte_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "stream_reuse_after_finish": True,
                "owner_close_live_allocation_count": 0,
            },
        ),
        "smollm2-multi-step-greedy-exact": _test(
            by_id["smollm2-multi-step-greedy-exact"],
            "optimizer report.tests.smollm2-multi-step-greedy-exact",
            "smollm2-multi-step-greedy-exact",
            {
                "decode_steps": 16,
                "committed_iterations": 16,
                "raw_logit_mismatches": 0,
                "generated_token_ids": list(EXPECTED_TOKENS),
                "token_id_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "owner_close_live_allocation_count": 0,
            },
        ),
        "fixed37-production-batch-e0": _test(
            fixed37_row,
            "optimizer report.tests.fixed37-production-batch-e0",
            "fixed37-production-batch-e0",
            {
                "gate_id": FIXED37_PRODUCTION_BATCH_GATE_ID,
                "fixture_sha256": EXPECTED_FIXED37_FIXTURE_SHA256,
                "generated_token_ids_sha256": EXPECTED_FIXED37_TOKEN_IDS_SHA256,
                "cases": 31,
                "compared_steps": 481,
                "exact_window": 16,
                "fixed_profile": "fixed-contiguous-37-balanced-v1",
                "canonical_profile": "canonical-v1",
                "residual_rmsnorm": "separate",
                "execution_completion": "iteration-batch",
                "fixed_prefill_raw_logit_mismatches": 0,
                "fixed_cached_growing_token_id_mismatches": 0,
                "fixed_cached_growing_cosine_min": FIXED37_CACHED_GROWING_COSINE_MIN,
                "fixed_cached_growing_max_abs_max": FIXED37_CACHED_GROWING_MAX_ABS_MAX,
                "fixed_cached_growing_mean_abs_max": FIXED37_CACHED_GROWING_MEAN_ABS_MAX,
                "fixed_cached_growing_worst_cosine": fixed37_worst_cosine,
                "fixed_cached_growing_worst_max_abs": fixed37_worst_max_abs,
                "fixed_cached_growing_worst_mean_abs": fixed37_worst_mean_abs,
                "fixed_cached_growing_threshold_violations": 0,
                "fixed_golden_token_id_mismatches": 0,
                "canonical_golden_token_id_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "owner_close_live_allocation_count": 0,
                "compile_command_id": "compile-fixed37-production-batch-e0",
                "execute_command_id": "fixed37-production-batch-e0",
                "compile_log_sha256": fixed37_compile_log_sha256,
                "test_binary_sha256": fixed37_test_binary_sha256,
            },
        ),
    }
    return profile_image_sha256
