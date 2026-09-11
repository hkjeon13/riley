#!/usr/bin/env python3
"""Pure canonical-byte review-input contract for a future Gate E guardian.

This module validates a supplied *unapproved* design-review input only.  It
does not read an audit path, inspect an ELF file, authenticate a bundle,
install a service, open a cgroup or ledger, launch a child, or contact a GPU,
Docker, evidence, receipt, freeze, or qualification surface.  In particular,
an input accepted here is not an administrator approval and cannot authorize
an implementation or an installed no-GPU acceptance run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Final, NoReturn


SCHEMA_VERSION: Final = "riley.rc3-gate-e-native-guardian-review.v1"
SCOPE: Final = "native-guardian-review-only"
AUTHORITY: Final = "not-authoritative"
REVIEW_STATUS: Final = "unapproved-design-input"
INSTALLATION_STATUS: Final = "not-installed"
OPERATIONAL_STATUS: Final = "not-authorized"
BUNDLE_SCHEMA_VERSION: Final = "riley.rc3-gate-e-native-guardian-bundle.v2"

MAX_DOCUMENT_BYTES: Final = 64 * 1024
MAX_JSON_NESTING: Final = 16
MAX_JSON_NODES: Final = 256
MAX_INTEGER_LITERAL_DIGITS: Final = 19
MAX_OBJECT_BYTES: Final = 16 * 1024 * 1024 * 1024
MAX_SIDECAR_BYTES: Final = 64 * 1024
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")

STATIC_STRATEGY: Final = "static-interpreter"
DYNAMIC_STRATEGY: Final = "same-object-dynamic-loader"
REQUIRED_ARTIFACTS: Final = frozenset(
    {
        "bundle_schema",
        "cpu_hostile_path_matrix",
        "fd_syscall_state_table",
        "installed_no_gpu_acceptance_plan",
        "static_build_recipe",
        "static_elf_inspection_policy",
        "threat_model",
    }
)
BOOTSTRAP_FDS: Final = (0, 1, 2, 31, 32)
WORKER_FDS: Final = (0, 1, 2)


class NativeGuardianReviewContractError(ValueError):
    """A proposed native-guardian review input is unsafe or incomplete."""


def _fail(code: str, message: str) -> NoReturn:
    error = NativeGuardianReviewContractError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


@dataclass(frozen=True)
class DigestBinding:
    """A bounded declaration identity, never an opened or authenticated object."""

    byte_length: int
    sha256: str


@dataclass(frozen=True)
class NativeGuardianReviewInput:
    """Validated review input with no installation or execution meaning."""

    bundle_manifest: DigestBinding
    execution_closure_sidecar: DigestBinding
    execution_strategy: str
    review_input_sha256: str


def canonical_native_guardian_review_bytes(value: object) -> bytes:
    """Encode the sole v1 review-input byte form, including one newline."""

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
        _fail("unencodable-canonical-json", f"cannot canonically encode review input: {error}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("duplicate-json-key", "native guardian review input repeats a JSON key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _fail("non-finite-json-number", f"review input has non-finite JSON number {value!r}")


def _bounded_integer_literal(value: str) -> int:
    if len(value.lstrip("-")) > MAX_INTEGER_LITERAL_DIGITS:
        _fail("integer-literal-too-large", "review input has an oversized integer literal")
    try:
        return int(value)
    except ValueError as error:
        _fail("invalid-json-number", f"review input has an invalid integer literal: {error}")


def _reject_float(value: str) -> NoReturn:
    _fail("invalid-json-number", f"review input cannot contain decimal JSON number {value!r}")


def _validate_json_budget(value: object) -> None:
    nodes = 0
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            _fail("json-node-budget-exceeded", "review input exceeds its JSON node budget")
        if depth > MAX_JSON_NESTING:
            _fail("json-nesting-too-deep", "review input exceeds its JSON nesting bound")
        if type(current) is dict:
            for key, child in current.items():
                if type(key) is not str:
                    _fail("invalid-json-object-key", "review input has a non-string JSON key")
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


def _binding(value: object, label: str, *, maximum: int = MAX_OBJECT_BYTES) -> DigestBinding:
    item = _mapping(value, {"byte_length", "sha256"}, label)
    return DigestBinding(
        byte_length=_positive(item["byte_length"], f"{label}.byte_length", maximum),
        sha256=_sha256(item["sha256"], f"{label}.sha256"),
    )


def _parse_bundle(value: object) -> tuple[DigestBinding, DigestBinding]:
    item = _mapping(
        value,
        {
            "bundle_schema_version",
            "execution_closure_sidecar",
            "manifest",
            "v1_compatibility",
        },
        "bundle",
    )
    if item["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION:
        _fail("invalid-bundle-schema-version", "review input must name the v2 native guardian bundle")
    if item["v1_compatibility"] is not False:
        _fail("legacy-v1-reuse", "review input cannot silently reuse the v1 bundle grammar")
    sidecar = _binding(
        item["execution_closure_sidecar"],
        "bundle.execution_closure_sidecar",
        maximum=MAX_SIDECAR_BYTES,
    )
    manifest_item = _mapping(
        item["manifest"],
        {
            "byte_length",
            "execution_closure_sidecar_byte_length",
            "execution_closure_sidecar_sha256",
            "sha256",
        },
        "bundle.manifest",
    )
    manifest = DigestBinding(
        byte_length=_positive(
            manifest_item["byte_length"],
            "bundle.manifest.byte_length",
            MAX_DOCUMENT_BYTES,
        ),
        sha256=_sha256(manifest_item["sha256"], "bundle.manifest.sha256"),
    )
    sidecar_length = _positive(
        manifest_item["execution_closure_sidecar_byte_length"],
        "bundle.manifest.execution_closure_sidecar_byte_length",
        MAX_SIDECAR_BYTES,
    )
    sidecar_sha256 = _sha256(
        manifest_item["execution_closure_sidecar_sha256"],
        "bundle.manifest.execution_closure_sidecar_sha256",
    )
    if sidecar_length != sidecar.byte_length or sidecar_sha256 != sidecar.sha256:
        _fail(
            "sidecar-manifest-binding-mismatch",
            "bundle manifest must repeat the exact raw execution-closure sidecar identity",
        )
    return manifest, sidecar


def _parse_execution_strategy(value: object) -> str:
    if type(value) is not dict:
        _fail("invalid-execution-strategy", "execution_strategy must be an object")
    kind = value.get("kind")
    if kind == STATIC_STRATEGY:
        item = _mapping(
            value,
            {
                "dependency_resolution",
                "kind",
                "pt_interp",
                "static_elf_inspection_policy_sha256",
            },
            "execution_strategy",
        )
        if item["pt_interp"] != "absent" or item["dependency_resolution"] != "none":
            _fail(
                "invalid-execution-strategy",
                "static strategy must reject PT_INTERP and dynamic dependency resolution",
            )
        _sha256(
            item["static_elf_inspection_policy_sha256"],
            "execution_strategy.static_elf_inspection_policy_sha256",
        )
        return STATIC_STRATEGY
    if kind == DYNAMIC_STRATEGY:
        item = _mapping(
            value,
            {
                "dynamic_loader_binding",
                "kind",
                "loader_resolution_proof_sha256",
                "pt_interp",
                "rejection_policy_sha256",
            },
            "execution_strategy",
        )
        if item["pt_interp"] != "reviewed-held-object":
            _fail(
                "invalid-execution-strategy",
                "dynamic strategy must name a reviewed held PT_INTERP object",
            )
        _binding(item["dynamic_loader_binding"], "execution_strategy.dynamic_loader_binding")
        _sha256(
            item["loader_resolution_proof_sha256"],
            "execution_strategy.loader_resolution_proof_sha256",
        )
        _sha256(
            item["rejection_policy_sha256"],
            "execution_strategy.rejection_policy_sha256",
        )
        return DYNAMIC_STRATEGY
    _fail("invalid-execution-strategy", "review input must select exactly one reviewed execution strategy")


def _parse_fd_abi(value: object) -> None:
    item = _mapping(
        value,
        {
            "bootstrap_inherited_fds",
            "capabilities",
            "core_fd",
            "environment",
            "no_new_privs",
            "worker_inherited_fds",
        },
        "fd_abi",
    )
    if (
        type(item["bootstrap_inherited_fds"]) is not list
        or tuple(item["bootstrap_inherited_fds"]) != BOOTSTRAP_FDS
        or type(item["worker_inherited_fds"]) is not list
        or tuple(item["worker_inherited_fds"]) != WORKER_FDS
        or item["core_fd"] != 32
        or item["environment"] != "empty"
        or item["no_new_privs"] is not True
        or item["capabilities"] != "cleared"
    ):
        _fail(
            "invalid-fd-abi",
            "review input must retain only 0/1/2/31/32 for bootstrap and 0/1/2 for worker",
        )
    if any(type(fd) is not int for fd in item["bootstrap_inherited_fds"]) or any(
        type(fd) is not int for fd in item["worker_inherited_fds"]
    ):
        _fail("invalid-fd-abi", "FD ABI entries must be exact integer descriptors")


def _parse_required_artifacts(value: object) -> None:
    item = _mapping(value, set(REQUIRED_ARTIFACTS), "required_artifacts")
    for name in REQUIRED_ARTIFACTS:
        _sha256(item[name], f"required_artifacts.{name}")


def parse_native_guardian_review_v1(raw: bytes) -> NativeGuardianReviewInput:
    """Parse one unapproved C02-P2 review input without filesystem/process I/O.

    The returned SHA-256 identifies only the supplied canonical JSON bytes.  It
    does not authenticate a manifest, sidecar, artifact, static ELF result,
    dynamic-loader proof, service unit, or installed object.  Those objects
    require the separate administrator/reviewer process described in
    ``RC3_GATE_E_NATIVE_GUARDIAN_REVIEW.md``.
    """

    if type(raw) is not bytes or not raw:
        _fail("invalid-review-byte-length", "review input must be nonempty bytes")
    if len(raw) > MAX_DOCUMENT_BYTES:
        _fail("review-byte-budget-exceeded", "review input exceeds its byte budget")
    if not raw.endswith(b"\n") or raw[:-1].endswith(b"\n"):
        _fail("invalid-terminal-newline", "review input needs exactly one terminal newline")
    encoded = raw[:-1]
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_bounded_integer_literal,
            parse_float=_reject_float,
            parse_constant=_reject_nonfinite,
        )
    except NativeGuardianReviewContractError:
        raise
    except UnicodeDecodeError as error:
        _fail("invalid-json", f"review input is not UTF-8 JSON: {error}")
    except json.JSONDecodeError as error:
        _fail("invalid-json", f"review input is not JSON: {error}")
    except ValueError as error:
        _fail("invalid-json", f"review input has an invalid JSON value: {error}")
    except RecursionError as error:
        _fail("json-nesting-too-deep", f"review input exceeds parser nesting: {error}")
    if type(value) is not dict:
        _fail("invalid-json-root", "review input must have an object root")
    _validate_json_budget(value)
    if canonical_native_guardian_review_bytes(value) != raw:
        _fail("noncanonical-json", "review input must use exact canonical JSON bytes")

    item = _mapping(
        value,
        {
            "authority",
            "bundle",
            "execution_strategy",
            "fd_abi",
            "installation_status",
            "operational_status",
            "required_artifacts",
            "review_status",
            "schema_version",
            "scope",
        },
        "review input",
    )
    if item["schema_version"] != SCHEMA_VERSION:
        _fail("unsupported-schema-version", "review input has an unsupported schema version")
    if item["scope"] != SCOPE:
        _fail("invalid-scope", "review input has an unsafe scope")
    if item["authority"] != AUTHORITY:
        _fail("invalid-authority", "review input cannot claim authority")
    if item["review_status"] != REVIEW_STATUS:
        _fail("invalid-review-status", "review input cannot claim administrator approval")
    if item["installation_status"] != INSTALLATION_STATUS:
        _fail("invalid-installation-status", "review input cannot claim an installed guardian")
    if item["operational_status"] != OPERATIONAL_STATUS:
        _fail("invalid-operational-status", "review input cannot authorize operations")

    manifest, sidecar = _parse_bundle(item["bundle"])
    strategy = _parse_execution_strategy(item["execution_strategy"])
    _parse_fd_abi(item["fd_abi"])
    _parse_required_artifacts(item["required_artifacts"])
    return NativeGuardianReviewInput(
        bundle_manifest=manifest,
        execution_closure_sidecar=sidecar,
        execution_strategy=strategy,
        review_input_sha256=hashlib.sha256(raw).hexdigest(),
    )
