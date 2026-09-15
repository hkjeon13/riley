#!/usr/bin/env python3
"""Validate the N01 model, numerical, and 20 GB planning contracts.

This validator intentionally does not inspect or run a model checkpoint.  It
checks the pinned planning inputs and rejects a descriptor that represents an
estimate, an unsupported execution shape, or pending numerical criteria as a
measured or qualified serving result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


MODELS_SCHEMA_VERSION = "n01-model-descriptors-v1"
NUMERICAL_SCHEMA_VERSION = "n01-numerical-profiles-v1"
PEAK_LIMIT_BYTES = 20_000_000_000
MINIMUM_RESERVE_BYTES = 1_000_000_000
BYTES_PER_BF16 = 2
KV_TENSORS = 2
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_PEAK_COMPONENTS = [
    "weights",
    "resident_kv",
    "activations_and_scratch",
    "cuda_graph_pool",
    "cuda_and_library_context",
    "weight_packing_and_loading_duplicates",
    "external_gpu_usage",
]

STRICT_TENSOR_THRESHOLDS = {
    "first_layer_hidden": {
        "max_abs_max": 0.3884272575378418,
        "max_relative_max": 0.13578447438776492,
        "mean_abs_max": 0.008509292567237658,
        "mean_relative_max": 0.005414661057131772,
        "cosine_min": 0.999983706829855,
    },
    "final_logits": {
        "max_abs_max": 5.852936458587647,
        "max_relative_max": 1.1707394897937775,
        "mean_abs_max": 1.151280319263363,
        "mean_relative_max": 0.13616598220459955,
        "cosine_min": 0.9979035305495393,
    },
    "final_log_probs": {
        "max_abs_max": 4.998420619964599,
        "max_relative_max": 0.5767027348279953,
        "mean_abs_max": 0.6007178144163239,
        "mean_relative_max": 0.04668832837569344,
        "cosine_min": 0.9987779663298298,
    },
}

EXPECTED_MODELS: dict[str, dict[str, Any]] = {
    "smollm2-135m-strict-regression": {
        "role": "strict-regression",
        "source": {
            "model_id": "HuggingFaceTB/SmolLM2-135M",
            "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        },
        "asset_status": "locked",
        "locked_hashes": {
            "config_sha256": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
            "weights_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
            "tokenizer_aggregate_sha256": "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db",
            "tokenizer_json_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        },
        "architecture": {
            "family": "llama",
            "source_architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "activation": "silu",
            "hidden_size": 576,
            "intermediate_size": 1536,
            "layer_count": 30,
            "query_heads": 9,
            "kv_heads": 3,
            "head_dim": 64,
            "max_context_tokens": 8192,
            "rope": {"theta": 100000, "scaling": "none", "interleaved": False},
            "rms_norm_epsilon": 0.00001,
            "attention_bias": {"q": False, "k": False, "v": False, "o": False},
            "attention_window": {"use_sliding_window": False, "sliding_window_tokens": None, "max_window_layers": None},
            "mlp_bias": False,
            "tied_embeddings": True,
            "vocab_size": 49152,
            "special_tokens": {"bos": 0, "eos": [0], "pad": None},
        },
        "tokenizer": {
            "profile_id": "smollm2-135m-locked-v1",
            "status": "locked",
            "source": "locked-tokenizer-artifact",
        },
        "parameter_count_estimate": 135_000_000,
        "planning_weight_ceiling_bytes": 400_000_000,
        "scenario_scope": {
            "c32-1024-plus-32": (1024, 32, (1, 8, 32)),
        },
        "numerical_profiles": ["strict-smollm2-135m-e0-v3", "native-bf16-r0-v1"],
        "serving_support": ("supported-existing-strict-path", "none"),
        "acceptance_status": "strict-regression-only",
    },
    "smollm2-1.7b-instruct-connection": {
        "role": "connection-validation",
        "source": {
            "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "revision": "57aa3c6599c53705406c648e7acca7e11dc45ea3",
        },
        "asset_status": "materialization-required",
        "architecture": {
            "family": "llama",
            "source_architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "activation": "silu",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "layer_count": 24,
            "query_heads": 32,
            "kv_heads": 32,
            "head_dim": 64,
            "max_context_tokens": 8192,
            "rope": {"theta": 130000, "scaling": "none", "interleaved": False},
            "rms_norm_epsilon": 0.00001,
            "attention_bias": {"q": False, "k": False, "v": False, "o": False},
            "attention_window": {"use_sliding_window": False, "sliding_window_tokens": None, "max_window_layers": None},
            "mlp_bias": False,
            "tied_embeddings": True,
            "vocab_size": 49152,
            "special_tokens": {"bos": 1, "eos": [2], "pad": 2},
        },
        "tokenizer": {
            "profile_id": "smollm2-1.7b-source-pinned-v1",
            "status": "materialization-required",
            "source": "pinned-source-revision",
        },
        "parameter_count_estimate": 1_700_000_000,
        "planning_weight_ceiling_bytes": 4_000_000_000,
        "scenario_scope": {
            "connection-c8-512-plus-128": (512, 128, (1, 8)),
            "connection-c4-7168-plus-1024": (7168, 1024, (1, 4)),
        },
        "numerical_profiles": ["native-bf16-r0-v1"],
        "serving_support": ("not-yet-qualified", "full-model-loader-and-d64-session-shape"),
        "acceptance_status": "not-qualified",
    },
    "qwen2.5-3b-instruct-primary": {
        "role": "primary-serving-evaluation",
        "source": {
            "model_id": "Qwen/Qwen2.5-3B-Instruct",
            "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        },
        "asset_status": "checkpoint-required",
        "architecture": {
            "family": "qwen2",
            "source_architecture": "Qwen2ForCausalLM",
            "model_type": "qwen2",
            "activation": "silu",
            "hidden_size": 2048,
            "intermediate_size": 11008,
            "layer_count": 36,
            "query_heads": 16,
            "kv_heads": 2,
            "head_dim": 128,
            "max_context_tokens": 32768,
            "rope": {"theta": 1000000, "scaling": "none", "interleaved": False},
            "rms_norm_epsilon": 0.000001,
            "attention_bias": {"q": True, "k": True, "v": True, "o": False},
            "attention_window": {"use_sliding_window": False, "sliding_window_tokens": 32768, "max_window_layers": 70},
            "mlp_bias": False,
            "tied_embeddings": True,
            "vocab_size": 151936,
            "special_tokens": {"bos": 151643, "eos": [151645], "pad": None},
        },
        "tokenizer": {
            "profile_id": "qwen2.5-3b-checkpoint-required-v1",
            "status": "checkpoint-required",
            "source": "pinned-source-revision",
            "must_not_reuse": "qwen2.5-0.5b-instruct-bf16",
        },
        "parameter_count_estimate": 3_090_000_000,
        "planning_weight_ceiling_bytes": 7_000_000_000,
        "scenario_scope": {
            "short-c32-512-plus-128": (512, 128, (1, 8, 32)),
            "general-c32-2048-plus-256": (2048, 256, (1, 8, 32)),
            "long-context-c8-8192-plus-512": (8192, 512, (1, 4, 8)),
            "long-generation-c4-16384-plus-1024": (16384, 1024, (1, 2, 4)),
        },
        "numerical_profiles": ["native-bf16-r0-v1"],
        "serving_support": ("unsupported", "head-dimension-128-execution"),
        "acceptance_status": "not-qualified",
    },
}


class ContractError(ValueError):
    """A descriptor violates the N01 contract."""


class DuplicateJsonKeyError(ValueError):
    """A JSON object repeated a key and would otherwise be silently overwritten."""


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        value[key] = member
    return value


def _fail(path: str, message: str) -> None:
    raise ContractError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return value


def _exact_keys(value: Any, keys: Iterable[str], path: str) -> Mapping[str, Any]:
    mapping = _mapping(value, path)
    expected = set(keys)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        _fail(path, "; ".join(details))
    return mapping


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(path, "must be a finite number")
    result = float(value)
    if positive and result <= 0.0:
        _fail(path, "must be positive")
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
        raise ContractError(f"{path}: cannot load JSON: {error}") from error
    return _mapping(payload, str(path))


def _validate_global_memory_contract(value: Any) -> Mapping[str, Any]:
    contract = _exact_keys(
        value,
        {
            "peak_limit_bytes",
            "minimum_uncertainty_reserve_bytes",
            "dtype",
            "kv_formula",
            "required_peak_components",
            "measurement_policy",
        },
        "global_memory_contract",
    )
    if _integer(contract["peak_limit_bytes"], "global_memory_contract.peak_limit_bytes") != PEAK_LIMIT_BYTES:
        _fail("global_memory_contract.peak_limit_bytes", "must equal the 20,000,000,000-byte RTX 4090 limit")
    reserve = _integer(
        contract["minimum_uncertainty_reserve_bytes"],
        "global_memory_contract.minimum_uncertainty_reserve_bytes",
    )
    if reserve < MINIMUM_RESERVE_BYTES:
        _fail("global_memory_contract.minimum_uncertainty_reserve_bytes", "must reserve at least 1,000,000,000 bytes")
    dtype = _exact_keys(
        contract["dtype"],
        {"weights", "kv_cache", "bytes_per_element"},
        "global_memory_contract.dtype",
    )
    if dtype != {"weights": "bf16", "kv_cache": "bf16", "bytes_per_element": BYTES_PER_BF16}:
        _fail("global_memory_contract.dtype", "must retain BF16 weights/KV and two bytes per element")
    if contract["kv_formula"] != "2 * layers * kv_heads * head_dim * bytes_per_element * resident_tokens":
        _fail("global_memory_contract.kv_formula", "must bind both K and V BF16 payloads")
    components = _list(contract["required_peak_components"], "global_memory_contract.required_peak_components")
    if components != REQUIRED_PEAK_COMPONENTS:
        _fail("global_memory_contract.required_peak_components", "must include every lifecycle peak component in contract order")
    _string(contract["measurement_policy"], "global_memory_contract.measurement_policy")
    return contract


def _expected_tensor_shapes(architecture: Mapping[str, Any]) -> dict[str, Any]:
    hidden = architecture["hidden_size"]
    intermediate = architecture["intermediate_size"]
    vocab = architecture["vocab_size"]
    query_projection = architecture["query_heads"] * architecture["head_dim"]
    kv_projection = architecture["kv_heads"] * architecture["head_dim"]
    return {
        "embedding": [vocab, hidden],
        "attention": {
            "q_proj": [query_projection, hidden],
            "k_proj": [kv_projection, hidden],
            "v_proj": [kv_projection, hidden],
            "o_proj": [hidden, hidden],
        },
        "mlp": {
            "gate_proj": [intermediate, hidden],
            "up_proj": [intermediate, hidden],
            "down_proj": [hidden, intermediate],
        },
        "final_norm": [hidden],
        "lm_head": {"shape": [vocab, hidden], "storage": "logical-tied-to-embedding"},
    }


def _validate_asset(asset: Any, expected: Mapping[str, Any], path: str) -> None:
    status = expected["asset_status"]
    if status == "locked":
        object_value = _exact_keys(asset, {"status", "locked_hashes"}, path)
        if object_value["status"] != status:
            _fail(f"{path}.status", f"must be {status!r}")
        hashes = _exact_keys(
            object_value["locked_hashes"],
            {"config_sha256", "weights_sha256", "tokenizer_aggregate_sha256", "tokenizer_json_sha256"},
            f"{path}.locked_hashes",
        )
        for key, digest in hashes.items():
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                _fail(f"{path}.locked_hashes.{key}", "must be a lowercase SHA-256 digest")
        if hashes != expected["locked_hashes"]:
            _fail(f"{path}.locked_hashes", "does not match the existing locked 135M artifact")
        return

    object_value = _exact_keys(asset, {"status", "materialization_gate"}, path)
    if object_value["status"] != status:
        _fail(f"{path}.status", f"must be {status!r}")
    _string(object_value["materialization_gate"], f"{path}.materialization_gate")


def _validate_weight_accounting(value: Any, expected: Mapping[str, Any], path: str) -> int:
    accounting = _exact_keys(
        value,
        {
            "parameter_count_estimate",
            "bf16_parameter_payload_estimate_bytes",
            "estimate_status",
            "observed_resident_bytes",
            "planning_resident_ceiling_bytes",
            "materialization_gate",
        },
        path,
    )
    parameters = _integer(accounting["parameter_count_estimate"], f"{path}.parameter_count_estimate", minimum=1)
    if parameters != expected["parameter_count_estimate"]:
        _fail(f"{path}.parameter_count_estimate", "does not match the pinned planning estimate")
    payload = _integer(
        accounting["bf16_parameter_payload_estimate_bytes"],
        f"{path}.bf16_parameter_payload_estimate_bytes",
        minimum=1,
    )
    if payload != parameters * BYTES_PER_BF16:
        _fail(f"{path}.bf16_parameter_payload_estimate_bytes", "must equal parameter_count_estimate * 2")
    if accounting["estimate_status"] != "planning-estimate-not-measured-device-residency":
        _fail(f"{path}.estimate_status", "must not present a planning estimate as measured device residency")
    if accounting["observed_resident_bytes"] is not None:
        _fail(f"{path}.observed_resident_bytes", "requires a lifecycle receipt and must remain null in this planning artifact")
    ceiling = _integer(
        accounting["planning_resident_ceiling_bytes"],
        f"{path}.planning_resident_ceiling_bytes",
        minimum=1,
    )
    if ceiling != expected["planning_weight_ceiling_bytes"] or ceiling < payload:
        _fail(f"{path}.planning_resident_ceiling_bytes", "must be the pinned conservative planning ceiling")
    _string(accounting["materialization_gate"], f"{path}.materialization_gate")
    return ceiling


def _validate_memory_scenarios(
    scenarios_value: Any,
    architecture: Mapping[str, Any],
    weight_ceiling: int,
    expected: Mapping[str, Any],
    path: str,
) -> None:
    scenarios = _list(scenarios_value, path)
    if not scenarios:
        _fail(path, "must contain at least one scenario")
    seen: set[str] = set()
    expected_scope = expected["scenario_scope"]
    for index, raw_scenario in enumerate(scenarios):
        item_path = f"{path}[{index}]"
        scenario = _exact_keys(
            raw_scenario,
            {
                "id",
                "prompt_tokens",
                "max_new_tokens",
                "active_capacity_candidates",
                "max_active_sequences",
                "resident_tokens_per_sequence",
                "resident_tokens_total",
                "kv_payload_bytes",
                "remaining_non_model_budget_bytes",
                "accounting_status",
            },
            item_path,
        )
        scenario_id = _string(scenario["id"], f"{item_path}.id")
        if scenario_id in seen:
            _fail(f"{item_path}.id", "must be unique")
        seen.add(scenario_id)
        if scenario_id not in expected_scope:
            _fail(f"{item_path}.id", "is not part of the pinned N01 workload scope")
        prompt = _integer(scenario["prompt_tokens"], f"{item_path}.prompt_tokens", minimum=1)
        output = _integer(scenario["max_new_tokens"], f"{item_path}.max_new_tokens", minimum=1)
        capacities = _list(scenario["active_capacity_candidates"], f"{item_path}.active_capacity_candidates")
        if not capacities:
            _fail(f"{item_path}.active_capacity_candidates", "must not be empty")
        parsed_capacities = tuple(
            _integer(value, f"{item_path}.active_capacity_candidates[{offset}]", minimum=1)
            for offset, value in enumerate(capacities)
        )
        if parsed_capacities != tuple(sorted(set(parsed_capacities))):
            _fail(f"{item_path}.active_capacity_candidates", "must be strictly ascending and unique")
        if (prompt, output, parsed_capacities) != expected_scope[scenario_id]:
            _fail(item_path, "does not match the pinned N01 workload dimensions")
        max_active = _integer(scenario["max_active_sequences"], f"{item_path}.max_active_sequences", minimum=1)
        if max_active != parsed_capacities[-1]:
            _fail(f"{item_path}.max_active_sequences", "must equal the largest active-capacity candidate")
        resident_per_sequence = _integer(
            scenario["resident_tokens_per_sequence"], f"{item_path}.resident_tokens_per_sequence", minimum=1
        )
        if resident_per_sequence != prompt + output:
            _fail(f"{item_path}.resident_tokens_per_sequence", "must include prompt and maximum output tokens")
        if resident_per_sequence > architecture["max_context_tokens"]:
            _fail(f"{item_path}.resident_tokens_per_sequence", "exceeds the model context limit")
        resident_total = _integer(scenario["resident_tokens_total"], f"{item_path}.resident_tokens_total", minimum=1)
        if resident_total != resident_per_sequence * max_active:
            _fail(f"{item_path}.resident_tokens_total", "must not assume shared-prefix savings in admission accounting")
        expected_kv = (
            KV_TENSORS
            * architecture["layer_count"]
            * architecture["kv_heads"]
            * architecture["head_dim"]
            * BYTES_PER_BF16
            * resident_total
        )
        if _integer(scenario["kv_payload_bytes"], f"{item_path}.kv_payload_bytes", minimum=1) != expected_kv:
            _fail(f"{item_path}.kv_payload_bytes", "does not match the BF16 K/V payload formula")
        remaining = _integer(
            scenario["remaining_non_model_budget_bytes"],
            f"{item_path}.remaining_non_model_budget_bytes",
            minimum=0,
        )
        expected_remaining = PEAK_LIMIT_BYTES - MINIMUM_RESERVE_BYTES - weight_ceiling - expected_kv
        if remaining != expected_remaining or remaining <= 0:
            _fail(f"{item_path}.remaining_non_model_budget_bytes", "must exactly preserve the 20GB limit and 1GB reserve")
        if scenario["accounting_status"] != "planning-only-requires-whole-gpu-lifecycle-receipt":
            _fail(f"{item_path}.accounting_status", "must not claim a measured full-GPU peak")
    if set(expected_scope) != seen:
        _fail(path, f"must contain exactly the pinned scenario ids {sorted(expected_scope)}")


def _validate_model(raw_model: Any, index: int) -> Mapping[str, Any]:
    path = f"models[{index}]"
    model = _exact_keys(
        raw_model,
        {
            "id",
            "role",
            "source",
            "asset",
            "architecture",
            "tensor_shapes",
            "tokenizer",
            "weight_accounting",
            "memory_scenarios",
            "numerical_profiles",
            "serving_support",
            "acceptance_status",
        },
        path,
    )
    model_id = _string(model["id"], f"{path}.id")
    expected = EXPECTED_MODELS.get(model_id)
    if expected is None:
        _fail(f"{path}.id", "is not a pinned N01 model descriptor")
    if model["role"] != expected["role"]:
        _fail(f"{path}.role", "does not match the pinned model role")
    source = _exact_keys(model["source"], {"model_id", "revision"}, f"{path}.source")
    if not isinstance(source["revision"], str) or REVISION_RE.fullmatch(source["revision"]) is None:
        _fail(f"{path}.source.revision", "must be a full lowercase 40-hex source revision")
    if source != expected["source"]:
        _fail(f"{path}.source", "does not match the pinned source model and revision")
    _validate_asset(model["asset"], expected, f"{path}.asset")
    architecture = _exact_keys(
        model["architecture"],
        {
            "family",
            "source_architecture",
            "model_type",
            "activation",
            "hidden_size",
            "intermediate_size",
            "layer_count",
            "query_heads",
            "kv_heads",
            "head_dim",
            "max_context_tokens",
            "rope",
            "rms_norm_epsilon",
            "attention_bias",
            "attention_window",
            "mlp_bias",
            "tied_embeddings",
            "vocab_size",
            "special_tokens",
        },
        f"{path}.architecture",
    )
    if architecture != expected["architecture"]:
        _fail(f"{path}.architecture", "does not match the pinned architecture, RoPE, bias, tie, or token contract")
    for numeric_name in (
        "hidden_size",
        "intermediate_size",
        "layer_count",
        "query_heads",
        "kv_heads",
        "head_dim",
        "max_context_tokens",
        "vocab_size",
    ):
        _integer(architecture[numeric_name], f"{path}.architecture.{numeric_name}", minimum=1)
    if architecture["hidden_size"] != architecture["query_heads"] * architecture["head_dim"]:
        _fail(f"{path}.architecture.hidden_size", "must equal query_heads * head_dim")
    if architecture["query_heads"] % architecture["kv_heads"] != 0:
        _fail(f"{path}.architecture.kv_heads", "must divide query_heads")
    if architecture["head_dim"] % 2 != 0:
        _fail(f"{path}.architecture.head_dim", "must be even for RoPE")
    _finite_number(architecture["rms_norm_epsilon"], f"{path}.architecture.rms_norm_epsilon", positive=True)
    if model["tensor_shapes"] != _expected_tensor_shapes(architecture):
        _fail(f"{path}.tensor_shapes", "does not derive from the pinned architecture")
    tokenizer = _mapping(model["tokenizer"], f"{path}.tokenizer")
    if tokenizer != expected["tokenizer"]:
        _fail(f"{path}.tokenizer", "does not match the pinned tokenizer profile and availability state")
    weight_ceiling = _validate_weight_accounting(model["weight_accounting"], expected, f"{path}.weight_accounting")
    _validate_memory_scenarios(
        model["memory_scenarios"], architecture, weight_ceiling, expected, f"{path}.memory_scenarios"
    )
    profile_ids = _list(model["numerical_profiles"], f"{path}.numerical_profiles")
    if profile_ids != expected["numerical_profiles"] or any(not isinstance(value, str) for value in profile_ids):
        _fail(f"{path}.numerical_profiles", "does not match the pinned numerical-profile linkage")
    support = _exact_keys(model["serving_support"], {"status", "reason", "required_capability"}, f"{path}.serving_support")
    if (support["status"], support["required_capability"]) != expected["serving_support"]:
        _fail(f"{path}.serving_support", "does not match the pinned implementation support state")
    _string(support["reason"], f"{path}.serving_support.reason")
    if model["acceptance_status"] != expected["acceptance_status"]:
        _fail(f"{path}.acceptance_status", "must not present an unmeasured or unsupported model as qualified")
    return model


def _validate_pending_gate(gate: Mapping[str, Any], path: str) -> None:
    if gate["status"] != "criteria-pending":
        _fail(f"{path}.status", "must be criteria-pending")
    binding = _exact_keys(gate["gate_binding"], {"status", "gate_id", "path"}, f"{path}.gate_binding")
    if binding != {"status": "criteria-pending", "gate_id": None, "path": None}:
        _fail(f"{path}.gate_binding", "must not invent a frozen reference gate")
    corpus = _exact_keys(
        gate["reference_corpus"], {"status", "source", "expected_prompt_count", "binding"}, f"{path}.reference_corpus"
    )
    if corpus != {"status": "criteria-pending", "source": None, "expected_prompt_count": None, "binding": None}:
        _fail(f"{path}.reference_corpus", "must remain pending until a corpus is frozen")
    seed = _exact_keys(gate["seed_policy"], {"status", "mode", "seed", "stochastic_policy"}, f"{path}.seed_policy")
    if seed != {"status": "criteria-pending", "mode": None, "seed": None, "stochastic_policy": None}:
        _fail(f"{path}.seed_policy", "must remain pending until seed policy is frozen")
    replay = _exact_keys(gate["replay_policy"], {"status", "policy", "candidate_bindings"}, f"{path}.replay_policy")
    if replay != {"status": "criteria-pending", "policy": None, "candidate_bindings": None}:
        _fail(f"{path}.replay_policy", "must remain pending until replay policy is frozen")
    tensor = _exact_keys(
        gate["tensor_accuracy"], {"status", "relative_error_denominator", "aggregation", "tensors"}, f"{path}.tensor_accuracy"
    )
    if tensor != {"status": "criteria-pending", "relative_error_denominator": None, "aggregation": None, "tensors": None}:
        _fail(f"{path}.tensor_accuracy", "must not invent tensor thresholds")
    teacher = _exact_keys(
        gate["teacher_forced"], {"status", "logits", "distribution", "cache_paths"}, f"{path}.teacher_forced"
    )
    if teacher != {"status": "criteria-pending", "logits": None, "distribution": None, "cache_paths": None}:
        _fail(f"{path}.teacher_forced", "must remain pending until teacher-forced criteria are frozen")
    for name in ("perplexity", "workload_quality"):
        metric = _exact_keys(gate[name], {"status", "threshold", "evaluation_tokens", "reason"}, f"{path}.{name}")
        if metric["status"] != "criteria-pending" or metric["threshold"] is not None or metric["evaluation_tokens"] is not None:
            _fail(f"{path}.{name}", "must not invent a quality threshold")
        _string(metric["reason"], f"{path}.{name}.reason")
    generation = _exact_keys(
        gate["free_generation"], {"status", "generation_steps", "generated_token_ids", "top_1", "top_k", "cross_cache_exact_window"}, f"{path}.free_generation"
    )
    if generation != {
        "status": "criteria-pending",
        "generation_steps": None,
        "generated_token_ids": None,
        "top_1": None,
        "top_k": None,
        "cross_cache_exact_window": None,
    }:
        _fail(f"{path}.free_generation", "must remain pending until free-generation criteria are frozen")


def _validate_strict_gate(gate: Mapping[str, Any], path: str) -> None:
    if gate["status"] != "frozen":
        _fail(f"{path}.status", "must be frozen")
    binding = _exact_keys(gate["gate_binding"], {"status", "gate_id", "path"}, f"{path}.gate_binding")
    if binding != {
        "status": "frozen",
        "gate_id": "smollm2-fp32-bf16-native-e0-v3",
        "path": "benchmarks/correctness/smollm2-fp32-bf16-native-e0-v3.json",
    }:
        _fail(f"{path}.gate_binding", "must bind the existing frozen E0 v3 gate")
    corpus = _exact_keys(
        gate["reference_corpus"], {"status", "source", "expected_prompt_count", "binding"}, f"{path}.reference_corpus"
    )
    if corpus != {
        "status": "frozen",
        "source": "benchmarks/prompts.jsonl",
        "expected_prompt_count": 31,
        "binding": "ordered-id-text-category-language-target-boundary-behavior",
    }:
        _fail(f"{path}.reference_corpus", "does not bind the existing frozen corpus")
    seed = _exact_keys(gate["seed_policy"], {"status", "mode", "seed", "stochastic_policy"}, f"{path}.seed_policy")
    if seed["status"] != "frozen" or seed["mode"] != "deterministic-greedy" or seed["seed"] is not None:
        _fail(f"{path}.seed_policy", "must preserve the deterministic greedy seed policy")
    _string(seed["stochastic_policy"], f"{path}.seed_policy.stochastic_policy")
    replay = _exact_keys(gate["replay_policy"], {"status", "policy", "candidate_bindings"}, f"{path}.replay_policy")
    if replay["status"] != "frozen" or replay["policy"] != "raw-evidence-bundle-replay-required":
        _fail(f"{path}.replay_policy", "must preserve the frozen replay policy")
    expected_bindings = [
        "bundled-executable-sha256",
        "clean-git-revision",
        "cargo-lock-sha256",
        "build-argv",
        "capture-argv",
    ]
    if replay["candidate_bindings"] != expected_bindings:
        _fail(f"{path}.replay_policy.candidate_bindings", "does not bind all required candidate provenance")
    tensor = _exact_keys(
        gate["tensor_accuracy"], {"status", "relative_error_denominator", "aggregation", "tensors"}, f"{path}.tensor_accuracy"
    )
    if tensor["status"] != "frozen" or tensor["relative_error_denominator"] != "max(abs(fp32),1)" or tensor["aggregation"] != "worst-prompt-metric-and-all-prompts-must-pass":
        _fail(f"{path}.tensor_accuracy", "must preserve the frozen tensor comparison semantics")
    tensors = _mapping(tensor["tensors"], f"{path}.tensor_accuracy.tensors")
    if tensors != STRICT_TENSOR_THRESHOLDS:
        _fail(f"{path}.tensor_accuracy.tensors", "does not match the frozen E0 v3 thresholds")
    for tensor_name, thresholds in tensors.items():
        for metric_name, metric_value in _mapping(thresholds, f"{path}.tensor_accuracy.tensors.{tensor_name}").items():
            positive = metric_name != "cosine_min"
            parsed = _finite_number(metric_value, f"{path}.tensor_accuracy.tensors.{tensor_name}.{metric_name}", positive=positive)
            if metric_name == "cosine_min" and not 0.0 < parsed <= 1.0:
                _fail(f"{path}.tensor_accuracy.tensors.{tensor_name}.{metric_name}", "must be in (0, 1]")
    teacher = _exact_keys(
        gate["teacher_forced"], {"status", "logits", "distribution", "cache_paths"}, f"{path}.teacher_forced"
    )
    if teacher != {
        "status": "frozen",
        "logits": "full-vocabulary-last-valid-position-logits",
        "distribution": "full-vocabulary-fp32-log-softmax-of-final-logits",
        "cache_paths": ["off", "on"],
    }:
        _fail(f"{path}.teacher_forced", "does not preserve the frozen teacher-forced logit/distribution contract")
    for name in ("perplexity", "workload_quality"):
        metric = _exact_keys(gate[name], {"status", "threshold", "evaluation_tokens", "reason"}, f"{path}.{name}")
        if metric["status"] != "not-gated-in-e0-v3" or metric["threshold"] is not None or metric["evaluation_tokens"] is not None:
            _fail(f"{path}.{name}", "must not fabricate a missing E0 v3 quality threshold")
        _string(metric["reason"], f"{path}.{name}.reason")
    generation = _exact_keys(
        gate["free_generation"], {"status", "generation_steps", "generated_token_ids", "top_1", "top_k", "cross_cache_exact_window"}, f"{path}.free_generation")
    if generation != {
        "status": "frozen",
        "generation_steps": 32,
        "generated_token_ids": "ordered-exact",
        "top_1": "ordered-exact",
        "top_k": {"k": 10, "comparison": "set-exact"},
        "cross_cache_exact_window": 16,
    }:
        _fail(f"{path}.free_generation", "does not preserve the frozen free-generation contract")


def _validate_profile(raw_profile: Any, index: int) -> Mapping[str, Any]:
    path = f"profiles[{index}]"
    profile = _exact_keys(
        raw_profile,
        {"id", "applies_to", "status", "performance_eligibility", "execution", "correctness_gate"},
        path,
    )
    profile_id = _string(profile["id"], f"{path}.id")
    applies_to = _list(profile["applies_to"], f"{path}.applies_to")
    if any(not isinstance(model_id, str) for model_id in applies_to) or len(set(applies_to)) != len(applies_to):
        _fail(f"{path}.applies_to", "must contain unique model ids")
    execution = _exact_keys(
        profile["execution"],
        {
            "weight_dtype",
            "kv_dtype",
            "activation_dtype",
            "semantics_status",
            "accumulator_dtype",
            "reduction_order",
            "softmax_policy",
            "cast_boundaries",
            "rmsnorm_policy",
            "bias_placement",
            "rope_placement",
        },
        f"{path}.execution",
    )
    for dtype_name in ("weight_dtype", "kv_dtype", "activation_dtype"):
        if execution[dtype_name] != "bf16":
            _fail(f"{path}.execution.{dtype_name}", "must remain BF16")
    semantic_fields = (
        "accumulator_dtype",
        "reduction_order",
        "softmax_policy",
        "cast_boundaries",
        "rmsnorm_policy",
        "bias_placement",
        "rope_placement",
    )
    gate = _exact_keys(
        profile["correctness_gate"],
        {
            "status",
            "gate_binding",
            "reference_corpus",
            "seed_policy",
            "replay_policy",
            "tensor_accuracy",
            "teacher_forced",
            "perplexity",
            "workload_quality",
            "free_generation",
        },
        f"{path}.correctness_gate",
    )
    if profile_id == "strict-smollm2-135m-e0-v3":
        if applies_to != ["smollm2-135m-strict-regression"] or profile["status"] != "locked" or profile["performance_eligibility"] != "baseline-only":
            _fail(path, "does not match the pinned strict baseline profile")
        if execution["semantics_status"] != "frozen":
            _fail(f"{path}.execution.semantics_status", "must be frozen for the strict profile")
        for policy_name in semantic_fields:
            _string(execution[policy_name], f"{path}.execution.{policy_name}")
        _validate_strict_gate(gate, f"{path}.correctness_gate")
    elif profile_id == "native-bf16-r0-v1":
        if applies_to != list(EXPECTED_MODELS) or profile["status"] != "criteria-pending" or profile["performance_eligibility"] != "blocked_on_reference":
            _fail(path, "must remain the unqualified native BF16 R0 profile")
        if execution["semantics_status"] != "criteria-pending" or any(execution[field] is not None for field in semantic_fields):
            _fail(f"{path}.execution", "must make every native BF16 arithmetic-placement policy explicitly pending")
        _validate_pending_gate(gate, f"{path}.correctness_gate")
    else:
        _fail(f"{path}.id", "is not a pinned N01 numerical profile")
    return profile


def validate_documents(models_document: Mapping[str, Any], numerical_document: Mapping[str, Any]) -> None:
    models_root = _exact_keys(models_document, {"schema_version", "global_memory_contract", "models"}, "models-document")
    if models_root["schema_version"] != MODELS_SCHEMA_VERSION:
        _fail("models-document.schema_version", f"must equal {MODELS_SCHEMA_VERSION!r}")
    _validate_global_memory_contract(models_root["global_memory_contract"])
    raw_models = _list(models_root["models"], "models-document.models")
    if len(raw_models) != len(EXPECTED_MODELS):
        _fail("models-document.models", "must contain exactly the pinned N01 models")
    models_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_model in enumerate(raw_models):
        model = _validate_model(raw_model, index)
        model_id = model["id"]
        if model_id in models_by_id:
            _fail(f"models[{index}].id", "must be unique")
        models_by_id[model_id] = model
    if set(models_by_id) != set(EXPECTED_MODELS):
        _fail("models-document.models", "does not contain the complete pinned model set")

    numerical_root = _exact_keys(numerical_document, {"schema_version", "profiles"}, "numerical-document")
    if numerical_root["schema_version"] != NUMERICAL_SCHEMA_VERSION:
        _fail("numerical-document.schema_version", f"must equal {NUMERICAL_SCHEMA_VERSION!r}")
    raw_profiles = _list(numerical_root["profiles"], "numerical-document.profiles")
    if len(raw_profiles) != 2:
        _fail("numerical-document.profiles", "must contain strict and native-BF16 profiles")
    profiles_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_profile in enumerate(raw_profiles):
        profile = _validate_profile(raw_profile, index)
        profile_id = profile["id"]
        if profile_id in profiles_by_id:
            _fail(f"profiles[{index}].id", "must be unique")
        profiles_by_id[profile_id] = profile
    if set(profiles_by_id) != {"strict-smollm2-135m-e0-v3", "native-bf16-r0-v1"}:
        _fail("numerical-document.profiles", "does not contain the complete pinned profile set")

    for model_id, model in models_by_id.items():
        for profile_id in model["numerical_profiles"]:
            profile = profiles_by_id.get(profile_id)
            if profile is None:
                _fail(f"model {model_id}.numerical_profiles", f"references unknown profile {profile_id!r}")
            if model_id not in profile["applies_to"]:
                _fail(f"model {model_id}.numerical_profiles", f"profile {profile_id!r} does not apply to the model")
            if profile["status"] == "criteria-pending" and model["acceptance_status"] not in {"not-qualified", "strict-regression-only"}:
                _fail(f"model {model_id}.acceptance_status", f"cannot qualify while {profile_id!r} is pending a reference gate")
    for profile_id, profile in profiles_by_id.items():
        for model_id in profile["applies_to"]:
            model = models_by_id.get(model_id)
            if model is None or profile_id not in model["numerical_profiles"]:
                _fail(f"profile {profile_id}.applies_to", f"is not linked from {model_id!r}")


def validate_paths(models_path: Path, numerical_path: Path) -> None:
    validate_documents(_load_json(models_path), _load_json(numerical_path))


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=here / "model-descriptors-v1.json")
    parser.add_argument("--numerical-profiles", type=Path, default=here / "numerical-profiles-v1.json")
    args = parser.parse_args(argv)
    try:
        validate_paths(args.models, args.numerical_profiles)
    except ContractError as error:
        print(f"N01 stage contract invalid: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "valid-planning-contract",
                "models": sorted(EXPECTED_MODELS),
                "peak_limit_bytes": PEAK_LIMIT_BYTES,
                "minimum_reserve_bytes": MINIMUM_RESERVE_BYTES,
                "native_bf16": "criteria-pending-blocked-on-reference",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
