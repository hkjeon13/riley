#!/usr/bin/env python3
"""Fail-closed RC3 qualification-input policy.

This source tree does not yet contain an authenticated same-stack producer for
a durable RC3 semantic receipt.  In particular, raw/structural receipts and
the narrow held-FD semantic replays are useful diagnostics, but are not
qualification evidence.  Until that durable receipt exists, there are no
admitted inputs.

The module deliberately has no CLI, filesystem API, output report, or
operational imports.  A future outer qualification finalizer may call the
bytes API before it opens or invokes any evidence replayer; this version will
always reject.  Adding an allow-list entry requires a separately reviewed
durable receipt and same-stack normal-return lineage contract in the same
change.
"""

from __future__ import annotations

import json
import math
from types import MappingProxyType
from typing import Any, Mapping, NoReturn


POLICY_VERSION = "riley.rc3-qualification-input-policy.v2"
SCOPE = "qualification-input-denial-only"
AUTHORITY = "qualification-input-denial-only"
MAX_INPUT_BYTES = 64 * 1024
MAX_JSON_NESTING = 32
MAX_JSON_NODES = 4_096

# This is intentionally empty.  A version suffix, a `passed` status, or a
# caller-supplied authority string must never become an admission mechanism.
ADMITTED_INPUTS: frozenset[str] = frozenset()

# Keep exact current-tree schemas explicit so a historical or narrow result
# receives a useful denial reason before the generic fail-closed path.  An
# unlisted schema is still rejected below.
REJECTED_SCHEMA_REASONS: Mapping[str, str] = MappingProxyType(
    {
        "riley.soak-v2-receipt.v1": "historical-soak-v1-rejected",
        "riley.soak-v2-raw-provenance.v1": "historical-soak-v1-rejected",
        "riley.soak-v2-raw-provenance.v2": "raw-soak-v2-rejected",
        "riley.soak-v2-raw-provenance.v3": "raw-soak-v3-rejected",
        "riley.soak-v2-raw-provenance.v4": "raw-soak-v4-rejected",
        "riley.soak-v2-raw-provenance.v5": "raw-soak-v5-rejected",
        "riley.soak-v2-provenance-check.v2": "raw-soak-v2-rejected",
        "riley.soak-v2-provenance-check.v3": "raw-soak-v3-rejected",
        "riley.soak-v2-provenance-check.v4": "raw-soak-v4-rejected",
        "riley.soak-v2-provenance-check.v5": "raw-soak-v5-rejected",
        "riley.soak-v2-semantic-replay-precheck.v1": "soak-structural-precheck-rejected",
        "riley.soak-v2-semantic-replay.v2": "soak-semantic-replay-not-durable",
        "riley.rc3-rollback-receipt.v1": "historical-rollback-v1-rejected",
        "riley.rc3-rollback-raw-provenance.v1": "historical-rollback-v1-rejected",
        "riley.rc3-rollback-raw-provenance.v2": "raw-rollback-v2-rejected",
        "riley.rc3-rollback-raw-provenance.v3": "raw-rollback-v3-rejected",
        "riley.rc3-rollback-raw-structural-precheck.v1": "rollback-structural-precheck-rejected",
        "riley.rc3-rollback-operational-semantics.v1": "rollback-held-fd-diagnostic-rejected",
        "riley.rc3-rollback-finalizer-receipt.v1": "rollback-finalizer-receipt-not-semantic",
        "riley.rc3-rollback-finalizer-receipt-complete.v1": "rollback-finalizer-receipt-not-semantic",
        "riley.rc3-rollback-terminal-provenance.v4": "rollback-terminal-provenance-not-semantic",
        "riley.rc3-rollback-terminal-provenance-check.v4": "rollback-terminal-provenance-not-semantic",
        "riley.rc3-gate-e-aggregate-semantic-replay.v1": "gate-e-aggregate-replay-not-durable",
        "riley.rc3-gate-e-aggregate-replay-record.v1": "gate-e-aggregate-record-not-durable",
        "riley.rc3-gate-e-aggregate-replay-record-complete.v1": "gate-e-aggregate-record-not-durable",
        "riley.rc3-gate-e-input-inventory-replay.v1": "gate-e-input-inventory-not-semantic",
        "riley.rc3-gate-e-native-e0-semantic-replay.v1": "gate-e-component-replay-not-aggregate",
        "riley.rc3-gate-e-optimizer-e0-semantic-replay.v1": "gate-e-component-replay-not-aggregate",
        "riley.rc3-gate-e-python-free-semantic-replay.v1": "gate-e-component-replay-not-aggregate",
        "riley.rc3-gate-e-performance-semantic-replay.v1": "gate-e-component-replay-not-aggregate",
        "riley.rc3-gate-e-soak-semantic-replay.v1": "gate-e-component-replay-not-aggregate",
        "riley.rc3-gate-e-native-root-bundle-preflight.v1": "native-root-bundle-preflight-not-qualification",
        "riley.rc3-gate-e-root-bundle.v1": "native-root-bundle-manifest-not-qualification",
        "riley.rc3-freeze-input-admission.v1": "freeze-input-admission-not-frozen-candidate",
        "riley.rc3-frozen-candidate.v1": "frozen-candidate-identity-not-semantic",
        "riley.rc3-frozen-candidate-replay.v1": "frozen-candidate-identity-not-semantic",
        "riley.reconstructed-prior-baseline-content-bridge.v1": "reconstructed-content-bridge-not-qualification",
        "riley.reconstructed-runtime-a-b-materialization.v1": "reconstructed-materialization-not-qualification",
        "riley.reconstructed-runtime-python-prerequisite.v1": "runtime-python-prerequisite-not-materialization",
        "riley.release-candidate-manifest.v1": "legacy-release-candidate-rejected",
        "riley.release-candidate-manifest.v2": "legacy-release-candidate-rejected",
        "riley.release-candidate-report.v2": "legacy-release-candidate-rejected",
        "riley.reliability-soak-report.v1": "legacy-release-candidate-rejected",
        "riley.reliability-soak-report.v2": "legacy-release-candidate-rejected",
    }
)

REJECTED_AUTHORITY_REASONS: Mapping[str, str] = MappingProxyType(
    {
        "raw-structural-only": "raw-structural-input-rejected",
        "soak-v2-semantic-replay-only": "soak-semantic-replay-not-durable",
        "raw-operational-semantics-only": "rollback-held-fd-diagnostic-rejected",
        "raw-finalizer-normal-return-only": "rollback-finalizer-receipt-not-semantic",
        "gate-e-aggregate-semantic-replay-only": "gate-e-aggregate-replay-not-durable",
        "gate-e-input-inventory-replay-only": "gate-e-input-inventory-not-semantic",
        "freeze-input-structural-only": "freeze-input-admission-not-frozen-candidate",
        "frozen-candidate-input-identity-only": "frozen-candidate-identity-not-semantic",
        "cross-root-content-bridge-only": "reconstructed-content-bridge-not-qualification",
        "held-fd-a-b-runtime-assembly-content-closure-only": "reconstructed-materialization-not-qualification",
        "interpreter-readiness-only": "runtime-python-prerequisite-not-materialization",
        "not-authoritative": "non-authoritative-input-rejected",
    }
)


class QualificationInputPolicyError(ValueError):
    """A proposed RC3 qualification input is not admissible."""


def _fail(code: str, message: str) -> NoReturn:
    error = QualificationInputPolicyError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate-json-key", f"qualification input repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _fail("non-finite-json-number", f"qualification input contains non-finite JSON number {value!r}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the exact JSON byte form required at this denial boundary."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        _fail("unencodable-canonical-json", f"cannot encode qualification input canonically: {error}")


def _validate_json_budget(value: Any) -> None:
    """Bound decoded JSON without recursive traversal or path interpretation."""

    nodes = 0
    pending: list[tuple[Any, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            _fail("json-node-budget-exceeded", "qualification input exceeds the JSON node budget")
        if depth > MAX_JSON_NESTING:
            _fail("json-nesting-too-deep", "qualification input exceeds the JSON nesting bound")
        if type(current) is dict:
            for key, child in current.items():
                if type(key) is not str:
                    _fail("invalid-json-object-key", "qualification input has a non-string JSON key")
                pending.append((child, depth + 1))
        elif type(current) is list:
            pending.extend((child, depth + 1) for child in current)
        elif type(current) is float and not math.isfinite(current):
            _fail("non-finite-json-number", "qualification input contains a non-finite JSON number")


def parse_canonical_qualification_input(raw: bytes, *, label: str = "qualification input") -> dict[str, Any]:
    """Parse one bounded canonical JSON document without any I/O.

    The future outer finalizer must read and hold a candidate input through its
    own reviewed descriptor boundary before passing the bytes here.  This
    module intentionally does not provide an alternate path-based reader.
    """

    if type(raw) is not bytes or not raw:
        _fail("invalid-json-byte-length", f"{label} must be nonempty bytes")
    if len(raw) > MAX_INPUT_BYTES:
        _fail("json-byte-budget-exceeded", f"{label} exceeds {MAX_INPUT_BYTES} bytes")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except QualificationInputPolicyError:
        raise
    except UnicodeDecodeError as error:
        _fail("invalid-json", f"{label} is not UTF-8 JSON: {error}")
    except json.JSONDecodeError as error:
        _fail("invalid-json", f"{label} is not JSON: {error}")
    except RecursionError as error:
        _fail("json-nesting-too-deep", f"{label} exceeds the parser nesting bound: {error}")
    if type(decoded) is not dict:
        _fail("invalid-json-root", f"{label} must have a JSON object root")
    _validate_json_budget(decoded)
    if raw != canonical_json_bytes(decoded):
        _fail("noncanonical-json", f"{label} must use exact canonical JSON bytes")
    return decoded


def qualification_input_denial_reason(document: Mapping[str, Any]) -> str:
    """Classify a document's fail-closed denial without admitting anything."""

    if not isinstance(document, Mapping):
        _fail("invalid-qualification-input", "qualification input must be a JSON object mapping")

    schema_version = document.get("schema_version")
    if schema_version is not None:
        if type(schema_version) is not str or not schema_version or len(schema_version) > 256:
            _fail("invalid-schema-version", "qualification input schema_version must be bounded text")
        known_reason = REJECTED_SCHEMA_REASONS.get(schema_version)
        if known_reason is not None:
            return known_reason

    authority = document.get("authority")
    if authority is not None:
        if type(authority) is not str or not authority or len(authority) > 256:
            _fail("invalid-qualification-authority", "qualification input authority must be bounded text")
        known_reason = REJECTED_AUTHORITY_REASONS.get(authority)
        if known_reason is not None:
            return known_reason

    if document.get("qualification_status") == "not-run":
        return "qualification-not-run-input-rejected"
    if document.get("status") == "passed":
        return "unrecognized-passed-qualification-input"
    return "unsupported-qualification-input"


def reject_qualification_input_document(
    document: Mapping[str, Any],
    *,
    label: str = "qualification input",
) -> NoReturn:
    """Reject a parsed proposed input; there is no success path in v2."""

    reason = qualification_input_denial_reason(document)
    schema_version = document.get("schema_version") if isinstance(document, Mapping) else None
    description = schema_version if type(schema_version) is str else "unversioned document"
    _fail(reason, f"{label} ({description}) is not an RC3 qualification input under {POLICY_VERSION}")


def reject_qualification_input_bytes(raw: bytes, *, label: str = "qualification input") -> NoReturn:
    """Parse and reject one proposed input before any replayer is invoked."""

    reject_qualification_input_document(
        parse_canonical_qualification_input(raw, label=label),
        label=label,
    )
