#!/usr/bin/env python3
"""Read-only checker for C03-A V1 routing-fuzz diagnostic receipts.

This checker accepts only a compact canonical diagnostic document emitted by
the test-only Rust receipt writer. A successful check establishes structural
binding of the descriptor, test configuration, and source revision only. It
does not replay a scheduler failure, preserve a panic signature/root cause,
prove a general/global minimum, produce GPU evidence, or qualify C02/C03-B.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, NoReturn


MAX_RECEIPT_BYTES = 64 * 1024
RECEIPT_FORMAT = "riley.scheduler.routing-fuzz-receipt"
RECEIPT_FORMAT_VERSION = 1
RECEIPT_SCOPE = "diagnostic-only"
TRACE_KIND = "general-mixed-operation-v1"
TEST_TARGET = "riley-scheduler::general_mixed_operation_routing"
FAILURE_PREDICATE = "inner-replayer-panicked-only"
REDUCER_SCOPE = "v1-selector-local"
MINIMIZED_CASE_ID = "failing-minimized"
SOURCE_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
CASE_ID_RE = re.compile(r"[a-z0-9-]+\Z")
TOP_LEVEL_KEYS = (
    "format",
    "format_version",
    "scope",
    "trace_kind",
    "test_target",
    "source_revision",
    "source_case_id",
    "failure_predicate",
    "reducer_scope",
    "source_descriptor_json",
    "minimized_descriptor_json",
    "source_operations",
    "minimized_operations",
    "source_scheduler_config",
    "minimized_scheduler_config",
    "symbolic_kv_layout",
    "replay_timeline_ns",
    "not_established",
)
DESCRIPTOR_KEYS = (
    "format",
    "format_version",
    "trace_kind",
    "case_id",
    "source_seed",
    "decoder_count",
    "final_prefill_count",
    "prime_slot_order",
    "mixed_slot_order",
    "cancel_decoder_index",
    "settlement",
)
SCHEDULER_CONFIG_KEYS = (
    "max_waiting_requests",
    "max_waiting_prompt_tokens",
    "max_active_sequences",
    "max_sequence_tokens",
    "iteration_token_budget",
    "max_prefill_chunk_tokens",
    "aging_threshold_ns",
    "overload_policy",
    "admission_timeout_ns",
    "max_promised_kv_blocks",
    "metrics_window_samples",
)
KV_LAYOUT_KEYS = (
    "layer_count",
    "physical_block_count",
    "key_value_head_count",
    "head_dimension",
    "block_size_tokens",
)
TIMELINE_KEYS = (
    "decoder_submit_and_prime_ns",
    "final_prefill_submit_and_mixed_ns",
    "close_ns",
)
NOT_ESTABLISHED = [
    "c02_qualification",
    "c03_b_gpu_evidence",
    "general_or_global_minimum",
    "panic_site_payload_signature_root_cause",
    "scheduler_reexecution",
]


class InputError(ValueError):
    """Receipt input is malformed, noncanonical, or has inconsistent bindings."""


def _raise_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    object_value: dict[str, Any] = {}
    for key, value in pairs:
        if key in object_value:
            raise InputError(f"JSON object repeats key {key!r}")
        object_value[key] = value
    return object_value


def _reject_nonfinite(value: str) -> NoReturn:
    raise InputError(f"JSON non-finite value {value!r} is forbidden")


def _reject_float(value: str) -> NoReturn:
    raise InputError(f"JSON floating-point value {value!r} is forbidden")


def _canonical_document(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"


def _parse_object(document: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            document,
            object_pairs_hook=_raise_duplicate_key,
            parse_constant=_reject_nonfinite,
            parse_float=_reject_float,
        )
    except (json.JSONDecodeError, InputError) as error:
        raise InputError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise InputError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: tuple[str, ...], label: str) -> None:
    if tuple(value) != expected:
        raise InputError(f"{label} has a missing, unknown, or noncanonical field order")


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise InputError(f"{label} must be a string")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{label} must be an integer")
    return value


def _matches_exact_json_value(actual: Any, expected: Any, label: str) -> bool:
    if isinstance(expected, int):
        return _require_int(actual, label) == expected
    if expected is None:
        return actual is None
    if isinstance(expected, str):
        return _require_string(actual, label) == expected
    raise AssertionError(f"unsupported expected JSON scalar for {label}")


def _validate_source_revision(value: Any, label: str) -> str:
    revision = _require_string(value, label)
    if SOURCE_REVISION_RE.fullmatch(revision) is None or revision == "0" * 40:
        raise InputError(f"{label} must be 40 lowercase nonzero hexadecimal digits")
    return revision


def _validate_case_id(value: Any, label: str) -> str:
    case_id = _require_string(value, label)
    if (
        not 1 <= len(case_id) <= 96
        or case_id.startswith("-")
        or case_id.endswith("-")
        or CASE_ID_RE.fullmatch(case_id) is None
    ):
        raise InputError(f"{label} must be a bounded lowercase identifier")
    return case_id


def _validate_slot_order(value: Any, slot_count: int, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != slot_count:
        raise InputError(f"{label} must contain exactly {slot_count} slots")
    slots = [_require_int(slot, f"{label}[{index}]") for index, slot in enumerate(value)]
    if sorted(slots) != list(range(slot_count)):
        raise InputError(f"{label} must be a dense slot permutation")
    return slots


def _validate_descriptor_document(document: Any, label: str) -> dict[str, Any]:
    encoded = _require_string(document, label)
    if "\r" in encoded:
        raise InputError(f"{label} must use LF-only canonical JSON")
    descriptor = _parse_object(encoded, label)
    if encoded != _canonical_document(descriptor):
        raise InputError(f"{label} is not exact canonical JSON")
    _require_exact_keys(descriptor, DESCRIPTOR_KEYS, label)
    if descriptor["format"] != "riley.scheduler.general-mixed-operation":
        raise InputError(f"{label}.format is unsupported")
    if _require_int(descriptor["format_version"], f"{label}.format_version") != 1:
        raise InputError(f"{label}.format_version is unsupported")
    if descriptor["trace_kind"] != TRACE_KIND:
        raise InputError(f"{label}.trace_kind is unsupported")
    _validate_case_id(descriptor["case_id"], f"{label}.case_id")
    source_seed = _require_string(descriptor["source_seed"], f"{label}.source_seed")
    if re.fullmatch(r"0x[0-9a-f]{16}", source_seed) is None:
        raise InputError(f"{label}.source_seed must be a fixed lowercase hexadecimal seed")
    decoder_count = _require_int(descriptor["decoder_count"], f"{label}.decoder_count")
    final_prefill_count = _require_int(
        descriptor["final_prefill_count"], f"{label}.final_prefill_count"
    )
    if not 1 <= decoder_count <= 3:
        raise InputError(f"{label}.decoder_count must be in 1..=3")
    if not 1 <= final_prefill_count <= 3:
        raise InputError(f"{label}.final_prefill_count must be in 1..=3")
    _validate_slot_order(descriptor["prime_slot_order"], decoder_count, f"{label}.prime_slot_order")
    _validate_slot_order(
        descriptor["mixed_slot_order"],
        decoder_count + final_prefill_count,
        f"{label}.mixed_slot_order",
    )
    cancellation = descriptor["cancel_decoder_index"]
    if cancellation is not None:
        cancellation = _require_int(cancellation, f"{label}.cancel_decoder_index")
        if not 0 <= cancellation < decoder_count:
            raise InputError(f"{label}.cancel_decoder_index must select a decoder")
    settlement = _require_string(descriptor["settlement"], f"{label}.settlement")
    if settlement not in {"commit", "abort_not_dispatched"}:
        raise InputError(f"{label}.settlement is unsupported")
    return descriptor


def _inversion_count(order: list[int]) -> int:
    return sum(
        int(order[left] > order[right])
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )


def _reducer_rank(descriptor: dict[str, Any]) -> tuple[int, int, int, int]:
    cancellation = descriptor["cancel_decoder_index"]
    return (
        descriptor["decoder_count"] + descriptor["final_prefill_count"],
        int(cancellation is not None),
        cancellation if cancellation is not None else 0,
        _inversion_count(descriptor["prime_slot_order"])
        + _inversion_count(descriptor["mixed_slot_order"]),
    )


def _expected_operations(descriptor: dict[str, Any]) -> str:
    cancellation = descriptor["cancel_decoder_index"]
    cancellation_text = (
        "none" if cancellation is None else f"cancel decoder[{cancellation}]"
    )
    settlement_text = (
        "complete" if descriptor["settlement"] == "commit" else "abort(not-dispatched)"
    )
    return (
        f"submit decoder[0..{descriptor['decoder_count']}) -> plan-prime -> "
        f"complete-prime(order={descriptor['prime_slot_order']}) -> "
        f"submit final-prefill[0..{descriptor['final_prefill_count']}) -> plan-mixed -> "
        f"{cancellation_text} -> {settlement_text}(order={descriptor['mixed_slot_order']}) "
        "-> close"
    )


def _expected_scheduler_config(descriptor: dict[str, Any]) -> dict[str, Any]:
    width = descriptor["decoder_count"] + descriptor["final_prefill_count"]
    return {
        "max_waiting_requests": width,
        "max_waiting_prompt_tokens": width,
        "max_active_sequences": width,
        "max_sequence_tokens": 3,
        "iteration_token_budget": width,
        "max_prefill_chunk_tokens": 1,
        "aging_threshold_ns": 2,
        "overload_policy": "wait",
        "admission_timeout_ns": None,
        "max_promised_kv_blocks": width,
        "metrics_window_samples": 8,
    }


def _validate_scheduler_config(
    value: Any, descriptor: dict[str, Any], label: str
) -> None:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be an object")
    _require_exact_keys(value, SCHEDULER_CONFIG_KEYS, label)
    for field, expected in _expected_scheduler_config(descriptor).items():
        if not _matches_exact_json_value(value[field], expected, f"{label}.{field}"):
            raise InputError(f"{label} does not bind the descriptor-derived harness configuration")


def _validate_kv_layout(value: Any) -> None:
    if not isinstance(value, dict):
        raise InputError("symbolic_kv_layout must be an object")
    _require_exact_keys(value, KV_LAYOUT_KEYS, "symbolic_kv_layout")
    expected = {
        "layer_count": 1,
        "physical_block_count": 64,
        "key_value_head_count": 1,
        "head_dimension": 8,
        "block_size_tokens": 16,
    }
    for field, expected_value in expected.items():
        if not _matches_exact_json_value(
            value[field], expected_value, f"symbolic_kv_layout.{field}"
        ):
            raise InputError("symbolic_kv_layout differs from the fixed host replay layout")


def _validate_replay_timeline(value: Any) -> None:
    if not isinstance(value, dict):
        raise InputError("replay_timeline_ns must be an object")
    _require_exact_keys(value, TIMELINE_KEYS, "replay_timeline_ns")
    expected = {
        "decoder_submit_and_prime_ns": 0,
        "final_prefill_submit_and_mixed_ns": 1,
        "close_ns": 2,
    }
    for field, expected_value in expected.items():
        if not _matches_exact_json_value(
            value[field], expected_value, f"replay_timeline_ns.{field}"
        ):
            raise InputError("replay_timeline_ns differs from the fixed host replay timeline")


def expected_receipt_filename(receipt: dict[str, Any], source: dict[str, Any]) -> str:
    return (
        f"{TRACE_KIND}-{receipt['source_case_id']}-{source['source_seed'][2:]}.json"
    )


def validate_receipt(
    document: str, expected_source_revision: str, receipt_name: str | None = None
) -> dict[str, Any]:
    """Validates one exact canonical receipt and returns its parsed object."""
    _validate_source_revision(expected_source_revision, "expected source revision")
    receipt = _parse_object(document, "routing fuzz receipt")
    if document != _canonical_document(receipt):
        raise InputError("routing fuzz receipt is not exact canonical JSON")
    _require_exact_keys(receipt, TOP_LEVEL_KEYS, "routing fuzz receipt")
    if receipt["format"] != RECEIPT_FORMAT:
        raise InputError("routing fuzz receipt format is unsupported")
    if _require_int(receipt["format_version"], "routing fuzz receipt.format_version") != RECEIPT_FORMAT_VERSION:
        raise InputError("routing fuzz receipt format_version is unsupported")
    if receipt["scope"] != RECEIPT_SCOPE:
        raise InputError("routing fuzz receipt must remain diagnostic-only")
    if receipt["trace_kind"] != TRACE_KIND:
        raise InputError("routing fuzz receipt trace_kind is unsupported")
    if receipt["test_target"] != TEST_TARGET:
        raise InputError("routing fuzz receipt test_target is unsupported")
    source_revision = _validate_source_revision(
        receipt["source_revision"], "routing fuzz receipt.source_revision"
    )
    if source_revision != expected_source_revision:
        raise InputError("routing fuzz receipt source_revision does not match the expected tree")
    source_case_id = _validate_case_id(
        receipt["source_case_id"], "routing fuzz receipt.source_case_id"
    )
    if receipt["failure_predicate"] != FAILURE_PREDICATE:
        raise InputError("routing fuzz receipt failure_predicate is unsupported")
    if receipt["reducer_scope"] != REDUCER_SCOPE:
        raise InputError("routing fuzz receipt reducer_scope is unsupported")
    source = _validate_descriptor_document(
        receipt["source_descriptor_json"], "routing fuzz receipt.source_descriptor_json"
    )
    minimized = _validate_descriptor_document(
        receipt["minimized_descriptor_json"],
        "routing fuzz receipt.minimized_descriptor_json",
    )
    if source["case_id"] != source_case_id:
        raise InputError("routing fuzz receipt source_case_id does not bind the source descriptor")
    if receipt_name is not None and receipt_name != expected_receipt_filename(receipt, source):
        raise InputError("routing fuzz receipt filename does not bind source case ID and seed")
    if minimized["case_id"] != MINIMIZED_CASE_ID:
        raise InputError("routing fuzz receipt minimized descriptor case_id is unsupported")
    if source["source_seed"] != minimized["source_seed"]:
        raise InputError("routing fuzz receipt minimized descriptor changed source_seed")
    if source["settlement"] != minimized["settlement"]:
        raise InputError("routing fuzz receipt minimized descriptor changed settlement")
    if _reducer_rank(minimized) > _reducer_rank(source):
        raise InputError("routing fuzz receipt minimized descriptor increased reducer rank")
    if receipt["source_operations"] != _expected_operations(source):
        raise InputError("routing fuzz receipt source_operations does not bind its descriptor")
    if receipt["minimized_operations"] != _expected_operations(minimized):
        raise InputError("routing fuzz receipt minimized_operations does not bind its descriptor")
    _validate_scheduler_config(
        receipt["source_scheduler_config"], source, "routing fuzz receipt.source_scheduler_config"
    )
    _validate_scheduler_config(
        receipt["minimized_scheduler_config"],
        minimized,
        "routing fuzz receipt.minimized_scheduler_config",
    )
    _validate_kv_layout(receipt["symbolic_kv_layout"])
    _validate_replay_timeline(receipt["replay_timeline_ns"])
    if receipt["not_established"] != NOT_ESTABLISHED:
        raise InputError("routing fuzz receipt not_established boundary changed")
    return receipt


def _read_receipt(path: Path) -> str:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise InputError(f"could not inspect receipt {path}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise InputError(f"receipt {path} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise InputError(f"could not open receipt {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            after = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise InputError(f"receipt {path} changed while it was opened")
            payload = stream.read(MAX_RECEIPT_BYTES + 1)
    except OSError as error:
        raise InputError(f"could not read receipt {path}: {error}") from error
    if len(payload) > MAX_RECEIPT_BYTES:
        raise InputError(f"receipt {path} exceeds {MAX_RECEIPT_BYTES} bytes")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InputError(f"receipt {path} is not UTF-8") from error


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one C03-A V1 diagnostic routing-fuzz receipt."
    )
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-source-revision", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_arguments()
    try:
        document = _read_receipt(args.receipt)
        receipt = validate_receipt(document, args.expected_source_revision, args.receipt.name)
    except InputError as error:
        print(f"routing fuzz receipt rejected: {error}", file=sys.stderr)
        return 2
    print(
        "routing fuzz diagnostic receipt is structurally valid "
        f"for source revision {receipt['source_revision']}; "
        "scheduler replay, root cause, GPU evidence, and qualification remain not established"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
