#!/usr/bin/env python3
"""Pure canonical-byte contract for a future Gate E execution closure.

This module parses supplied bytes only. It does not open an audit path, inspect
ELF, invoke a loader, execute Python, resolve a dynamic dependency, install a
bundle, or contact a GPU/Docker/evidence/qualification surface. A later static
native guardian must authenticate held objects and prove the actual loader and
runtime closure before it may rely on this declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Final, NoReturn


SCHEMA_VERSION: Final = "riley.rc3-gate-e-execution-closure-manifest.v1"
MAX_MANIFEST_BYTES: Final = 64 * 1024
MAX_JSON_NESTING: Final = 16
MAX_JSON_NODES: Final = 1_024
MAX_INTEGER_LITERAL_DIGITS: Final = 19
MAX_AUDIT_PATH_BYTES: Final = 512
MAX_RUNTIME_LEAVES: Final = 128
MAX_LEAF_BYTES: Final = 512 * 1024 * 1024
MAX_CLOSURE_BYTES: Final = 2 * 1024 * 1024 * 1024
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
AUDIT_PATH_RE: Final = re.compile(
    r"^/(?:[A-Za-z0-9_+@%:=,-][A-Za-z0-9._+@%:=,-]*)(?:/[A-Za-z0-9_+@%:=,-][A-Za-z0-9._+@%:=,-]*)*$"
)


class ExecutionClosureContractError(ValueError):
    """A proposed Gate E execution-closure declaration is unsafe."""


def _fail(code: str, message: str) -> NoReturn:
    error = ExecutionClosureContractError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


@dataclass(frozen=True)
class ClosureLeaf:
    """One declared future execution-closure object, not an opened object."""

    audit_path: str
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class ExecutionClosureManifest:
    """Validated declaration plus the canonical raw-byte closure identity."""

    interpreter: ClosureLeaf
    dynamic_loader: ClosureLeaf
    runtime_leaves: tuple[ClosureLeaf, ...]
    runtime_closure_sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("duplicate-json-key", "execution-closure manifest repeats a JSON key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _fail("non-finite-json-number", f"execution-closure manifest has non-finite JSON number {value!r}")


def _bounded_integer_literal(value: str) -> int:
    if len(value.lstrip("-")) > MAX_INTEGER_LITERAL_DIGITS:
        _fail("integer-literal-too-large", "execution-closure manifest has an oversized integer literal")
    try:
        return int(value)
    except ValueError as error:
        _fail("invalid-json-number", f"execution-closure manifest has an invalid integer literal: {error}")


def _reject_float(value: str) -> NoReturn:
    _fail("invalid-json-number", f"execution-closure manifest cannot contain decimal JSON number {value!r}")


def canonical_execution_closure_manifest_bytes(value: object) -> bytes:
    """Encode the sole v1 manifest byte form, including one terminal newline."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        _fail("unencodable-canonical-json", f"cannot canonically encode execution closure: {error}")


def _validate_json_budget(value: object) -> None:
    nodes = 0
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            _fail("json-node-budget-exceeded", "execution-closure manifest exceeds its JSON node budget")
        if depth > MAX_JSON_NESTING:
            _fail("json-nesting-too-deep", "execution-closure manifest exceeds its JSON nesting bound")
        if type(current) is dict:
            for key, child in current.items():
                if type(key) is not str:
                    _fail("invalid-json-object-key", "execution-closure manifest has a non-string JSON key")
                pending.append((child, depth + 1))
        elif type(current) is list:
            pending.extend((child, depth + 1) for child in current)


def _mapping(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        _fail("unexpected-field-set", f"{label} has an unexpected field set")
    return value


def _positive(value: object, label: str, maximum: int) -> int:
    if type(value) is not int or value <= 0 or value > maximum:
        _fail("invalid-byte-length", f"{label} must be a positive bounded integer")
    return value


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        _fail("invalid-sha256", f"{label} must be lowercase 64-hex SHA-256")
    if value == "0" * 64:
        _fail("zero-sha256", f"{label} cannot be the all-zero digest")
    return value


def _audit_path(value: object, label: str) -> str:
    if type(value) is not str or not value.isascii() or len(value.encode("ascii")) > MAX_AUDIT_PATH_BYTES:
        _fail("invalid-audit-path", f"{label} must be a bounded ASCII audit path")
    if AUDIT_PATH_RE.fullmatch(value) is None:
        _fail("invalid-audit-path", f"{label} must be an absolute canonical audit path")
    return value


def _leaf(value: object, label: str) -> ClosureLeaf:
    item = _mapping(value, {"audit_path", "byte_length", "sha256"}, label)
    return ClosureLeaf(
        audit_path=_audit_path(item["audit_path"], f"{label}.audit_path"),
        byte_length=_positive(item["byte_length"], f"{label}.byte_length", MAX_LEAF_BYTES),
        sha256=_sha256(item["sha256"], f"{label}.sha256"),
    )


def parse_execution_closure_manifest_v1(raw: bytes) -> ExecutionClosureManifest:
    """Parse one canonical v1 declaration without any filesystem or process I/O.

    ``runtime_closure_sha256`` in the result is SHA-256 of the supplied exact
    canonical manifest bytes, including the terminal newline. It is only a
    declaration identity: this function does not claim the listed objects are
    present, complete, owned, held, executable, or loader-resolved.
    """

    if type(raw) is not bytes or not raw:
        _fail("invalid-manifest-byte-length", "execution-closure manifest must be nonempty bytes")
    if len(raw) > MAX_MANIFEST_BYTES:
        _fail("manifest-byte-budget-exceeded", "execution-closure manifest exceeds its byte budget")
    if not raw.endswith(b"\n") or raw[:-1].endswith(b"\n"):
        _fail("invalid-terminal-newline", "execution-closure manifest needs exactly one terminal newline")
    encoded = raw[:-1]
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_bounded_integer_literal,
            parse_float=_reject_float,
            parse_constant=_reject_nonfinite,
        )
    except ExecutionClosureContractError:
        raise
    except UnicodeDecodeError as error:
        _fail("invalid-json", f"execution-closure manifest is not UTF-8 JSON: {error}")
    except json.JSONDecodeError as error:
        _fail("invalid-json", f"execution-closure manifest is not JSON: {error}")
    except ValueError as error:
        _fail("invalid-json", f"execution-closure manifest has an invalid JSON value: {error}")
    except RecursionError as error:
        _fail("json-nesting-too-deep", f"execution-closure manifest exceeds parser nesting: {error}")
    if type(value) is not dict:
        _fail("invalid-json-root", "execution-closure manifest must have an object root")
    _validate_json_budget(value)
    if canonical_execution_closure_manifest_bytes(value) != raw:
        _fail("noncanonical-json", "execution-closure manifest must use exact canonical JSON bytes")

    item = _mapping(
        value,
        {"dynamic_loader", "interpreter", "runtime_leaves", "schema_version"},
        "execution-closure manifest",
    )
    if item["schema_version"] != SCHEMA_VERSION:
        _fail("unsupported-schema-version", "execution-closure manifest has an unsupported schema version")
    interpreter = _leaf(item["interpreter"], "interpreter")
    dynamic_loader = _leaf(item["dynamic_loader"], "dynamic_loader")
    raw_runtime_leaves = item["runtime_leaves"]
    if type(raw_runtime_leaves) is not list or not raw_runtime_leaves:
        _fail("invalid-runtime-leaves", "runtime_leaves must be a nonempty JSON array")
    if len(raw_runtime_leaves) > MAX_RUNTIME_LEAVES:
        _fail("runtime-leaf-budget-exceeded", "runtime_leaves exceeds its entry budget")

    runtime_leaves: list[ClosureLeaf] = []
    previous_path: bytes | None = None
    total_bytes = interpreter.byte_length + dynamic_loader.byte_length
    paths = {interpreter.audit_path, dynamic_loader.audit_path}
    if len(paths) != 2:
        _fail("duplicate-audit-path", "interpreter and dynamic loader cannot share an audit path")
    for index, raw_leaf in enumerate(raw_runtime_leaves):
        leaf = _leaf(raw_leaf, f"runtime_leaves[{index}]")
        path_key = leaf.audit_path.encode("ascii")
        if previous_path is not None and path_key <= previous_path:
            _fail("runtime-leaves-not-strictly-sorted", "runtime_leaves must be bytewise sorted and unique")
        if leaf.audit_path in paths:
            _fail("duplicate-audit-path", "runtime_leaves cannot repeat interpreter or loader audit paths")
        total_bytes += leaf.byte_length
        if total_bytes > MAX_CLOSURE_BYTES:
            _fail("closure-byte-budget-exceeded", "execution closure exceeds its total byte budget")
        paths.add(leaf.audit_path)
        runtime_leaves.append(leaf)
        previous_path = path_key

    return ExecutionClosureManifest(
        interpreter=interpreter,
        dynamic_loader=dynamic_loader,
        runtime_leaves=tuple(runtime_leaves),
        runtime_closure_sha256=hashlib.sha256(raw).hexdigest(),
    )
