#!/usr/bin/env python3
"""Shared, dependency-free primitives for the competitive benchmark contract.

The competitive lane deliberately keeps its evidence tooling outside Riley's
production dependency graph.  This module uses only the Python standard
library and treats every JSON document as an untrusted input: duplicate keys,
unknown fields in execution evidence, non-finite numbers, and incomplete
identity records are rejected before any statistic is calculated.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = "riley.competitive.contract.v1"
MATRIX_SCHEMA_VERSION = "riley.competitive.matrix.v1"
LANE_SCHEMA_VERSION = "riley.competitive.lane.v1"
REQUESTS_SCHEMA_VERSION = "riley.competitive.requests.v1"
PLAN_SCHEMA_VERSION = "riley.competitive.execution-plan.v1"
RAW_SCHEMA_VERSION = "riley.competitive.raw.v1"
REPORT_SCHEMA_VERSION = "riley.competitive.report.v1"

# A syntactically valid manifest is not necessarily the reviewed C01
# contract.  The values below are the content digests of the reviewed v1
# assets.  They deliberately live in the checker, rather than in a manifest
# supplied by a campaign, so an alternate contract cannot lower a threshold
# and declare itself authoritative.  Updating one of these values is an
# intentional C01 contract revision and must be reviewed with the asset it
# names.
CANONICAL_CONTRACT_RELATIVE_PATH = "benchmarks/competitive/contract-v1.json"
CANONICAL_CONTRACT_ID = "riley-vllm-competitive-v1"
CANONICAL_PREFLIGHT_RELATIVE_PATH = "benchmarks/scripts/preflight.sh"
# Generated plans, materialized lane inputs/outputs, and raw journals must
# stay out of the reviewed source tree.  This exact ignored
# workspace is deliberately narrower than a generic ``campaigns/`` directory:
# source cleanliness remains a claim gate for every other untracked or dirty
# path in the checkout.
CAMPAIGN_ARTIFACT_WORKSPACE_RELATIVE_PATH = "benchmarks/competitive/.campaign-work"
CANONICAL_ASSET_SHA256: dict[str, str] = {
    CANONICAL_CONTRACT_RELATIVE_PATH: "4852ae4711aa2e02331995533babea0a21db07205620ccfc7fe94be3d7fa4194",
    "benchmarks/competitive/matrices/diagnostic-sm89-bf16-v1.json": "7288245ac4dd5d6fecd89f57a03207449e9fbd2905e8bc873e1aae233419077a",
    "benchmarks/competitive/matrices/latency-sm89-bf16-v1.json": "03eb884c35b458b5e61526f039ec51212104797f105008a4049842ab7b048fe6",
    "benchmarks/competitive/matrices/serving-sm89-bf16-v1.json": "9b414dc177b097c65f4fd44c95cb5169b15a8a9aa139f36a50e0f5bc01d8ae34",
    "benchmarks/competitive/lanes/riley.json": "9035fa499d6dec60a29668199d754ff7d677b2c4794900a2fd98be0dbd4623fa",
    "benchmarks/competitive/lanes/vllm-current.json": "8b1ce0ac66c7a8f7631f126c59b33a4a13d48b9d5ffa332fafcb76e6056b1047",
}
CANONICAL_PREFLIGHT_SHA256 = "2371a6291b6b47b89e960867a1c3ae814ffc1e10115ea74eb02e0792db9f42e4"

# `benchmarks/scripts/preflight.sh` emits this closed receipt.  C01 accepts
# the complete snapshot only: accepting a hand-selected subset would let a
# campaign manufacture a seemingly-ready environment by omitting failed
# checks.
PREFLIGHT_REQUIRED_KEYS = frozenset(
    {
        "environment_id",
        "os_id",
        "os_version_id",
        "kernel_release",
        "machine",
        "cpu_model",
        "physical_cpu_cores",
        "logical_cpu_threads",
        "ram_bytes",
        "git_revision",
        "gpu_name",
        "compute_capability",
        "memory_total_mib",
        "memory_used_mib",
        "driver_version",
        "persistence_mode",
        "temperature_c",
        "power_limit_w",
        "graphics_clock_mhz",
        "memory_clock_mhz",
        "cpu_governor",
        "cpu_governor_policy_count",
        "clock_synchronized",
        "staging_available_bytes",
        "staging_minimum_bytes",
    }
)
PREFLIGHT_EXACT_VALUES = {
    "environment_id": "rtx4090-ubuntu22-driver580-v1",
    "os_id": "ubuntu",
    "os_version_id": "22.04",
    "kernel_release": "6.8.0-138-generic",
    "machine": "x86_64",
    "cpu_model": "Intel Core i7-13700K",
    "physical_cpu_cores": "16",
    "logical_cpu_threads": "24",
    "ram_bytes": "67185598464",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "compute_capability": "8.9",
    "memory_total_mib": "24564",
    "driver_version": "580.173.02",
    "persistence_mode": "Disabled",
    "cpu_governor": "powersave",
    "cpu_governor_policy_count": "24",
    "clock_synchronized": "yes",
    "staging_minimum_bytes": str(20 * 1024**3),
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
TEMPLATE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:{}-]*$")

REQUEST_WORKLOAD_FIELDS = (
    "request_id",
    "prompt_token_ids_sha256",
    "prompt_tokens",
    "requested_output_tokens",
    "sampling",
    "seed",
    "eos_policy",
    "cache_policy",
    "arrival_schedule_id",
)
WORKLOAD_EXECUTION_FIELDS = (
    "model_identity_sha256",
    "measurement_mode",
    "warm_state",
    "arrival_mode",
    "client_behavior",
    "cancellation_rate_percent",
    "slo_profile",
)


class ContractError(ValueError):
    """Raised for malformed or internally inconsistent contract input."""


class ComparabilityError(ContractError):
    """Raised when valid evidence cannot be directly compared."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON with duplicate-key rejection."""

    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"{path}: JSON input must be a regular file, not a link or device")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except FileNotFoundError as error:
        raise ContractError(f"{path}: file does not exist") from error
    except OSError as error:
        raise ContractError(f"{path}: cannot read JSON: {error}") from error
    except json.JSONDecodeError as error:
        raise ContractError(
            f"{path}:{error.lineno}:{error.colno}: invalid JSON: {error.msg}"
        ) from error


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a report deterministically and reject unsupported values."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"cannot serialize canonical JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"{path}: hash input must be a regular file, not a link or device")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractError(f"{path}: cannot hash file: {error}") from error
    return digest.hexdigest()


def _relative_path_inside_root(root: Path, path: Path, label: str) -> str:
    """Return a normalized repository-relative path without accepting escape."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ContractError(f"{label} must stay inside repository root") from error


def _canonical_asset_kind(relative_path: str) -> str | None:
    if relative_path == CANONICAL_CONTRACT_RELATIVE_PATH:
        return "contract"
    if relative_path.startswith("benchmarks/competitive/matrices/"):
        return "matrix"
    if relative_path.startswith("benchmarks/competitive/lanes/"):
        return "lane"
    return None


def canonical_asset_reference(root: Path, path: Path, *, kind: str, label: str) -> str | None:
    """Return the canonical relative path if *path* is an exact C01 asset.

    A file with the same JSON shape at another path is deliberately not
    canonical.  Campaign-specific assets may be used only through the
    explicit parent bindings validated below.
    """

    relative_path = _relative_path_inside_root(root, path, label)
    expected_hash = CANONICAL_ASSET_SHA256.get(relative_path)
    if expected_hash is None:
        return None
    actual_kind = _canonical_asset_kind(relative_path)
    if actual_kind != kind:
        raise ContractError(f"{label} names a canonical {actual_kind}, not a {kind}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ContractError(
            f"{label} no longer matches the reviewed canonical {kind} SHA-256"
        )
    return relative_path


def require_canonical_contract(root: Path, path: Path, contract: Mapping[str, Any]) -> None:
    """Require the reviewed C01 contract, never a campaign-authored one."""

    relative_path = canonical_asset_reference(
        root,
        path,
        kind="contract",
        label="contract path",
    )
    if relative_path != CANONICAL_CONTRACT_RELATIVE_PATH:
        raise ContractError("C01 campaign contract must be the reviewed canonical contract-v1.json")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractError("canonical contract schema version drift")
    if contract.get("contract_id") != CANONICAL_CONTRACT_ID:
        raise ContractError("canonical contract ID drift")


def _parent_contract_receipt(value: Any, label: str) -> Mapping[str, Any]:
    receipt = _object(value, label)
    expect_keys(
        receipt,
        label,
        required=("path", "sha256", "schema_version", "contract_id"),
    )
    if receipt["path"] != CANONICAL_CONTRACT_RELATIVE_PATH:
        raise ContractError(f"{label}.path must name the canonical C01 contract")
    if receipt["sha256"] != CANONICAL_ASSET_SHA256[CANONICAL_CONTRACT_RELATIVE_PATH]:
        raise ContractError(f"{label}.sha256 must pin the canonical C01 contract")
    if receipt["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ContractError(f"{label}.schema_version must pin {CONTRACT_SCHEMA_VERSION}")
    if receipt["contract_id"] != CANONICAL_CONTRACT_ID:
        raise ContractError(f"{label}.contract_id must pin {CANONICAL_CONTRACT_ID}")
    return receipt


def validate_parent_contract_receipt(value: Any, label: str) -> Mapping[str, Any]:
    """Validate the immutable parent contract receipt carried by concrete input."""

    return _parent_contract_receipt(value, label)


def verify_parent_contract_receipt(root: Path, value: Any, label: str) -> Mapping[str, Any]:
    receipt = _parent_contract_receipt(value, label)
    canonical_path = root / CANONICAL_CONTRACT_RELATIVE_PATH
    if sha256_file(canonical_path) != receipt["sha256"]:
        raise ContractError(f"{label} canonical contract file/hash drift")
    contract = validate_contract(load_json(canonical_path), str(canonical_path))
    require_canonical_contract(root, canonical_path, contract)
    return receipt


def _parent_asset_receipt(value: Any, label: str) -> Mapping[str, Any]:
    receipt = _object(value, label)
    expect_keys(
        receipt,
        label,
        required=("path", "sha256", "schema_version", "asset_id"),
    )
    _string(receipt["path"], f"{label}.path")
    validate_sha256(receipt["sha256"], f"{label}.sha256")
    _string(receipt["schema_version"], f"{label}.schema_version")
    validate_identifier(receipt["asset_id"], f"{label}.asset_id")
    return receipt


def validate_parent_asset_receipt(value: Any, label: str) -> Mapping[str, Any]:
    return _parent_asset_receipt(value, label)


def verify_parent_asset_receipt(
    root: Path,
    value: Any,
    *,
    kind: str,
    label: str,
) -> Mapping[str, Any]:
    """Verify a concrete asset's exact canonical matrix/lane parent."""

    receipt = _parent_asset_receipt(value, label)
    parent_path = root / str(receipt["path"])
    relative_path = canonical_asset_reference(
        root,
        parent_path,
        kind=kind,
        label=f"{label}.path",
    )
    if relative_path is None:
        raise ContractError(f"{label}.path must name a reviewed canonical {kind}")
    if receipt["sha256"] != CANONICAL_ASSET_SHA256[relative_path]:
        raise ContractError(f"{label}.sha256 does not match canonical {kind}")
    parent = load_json(parent_path)
    if kind == "matrix":
        parent = validate_matrix(parent, str(parent_path))
        expected_schema = MATRIX_SCHEMA_VERSION
        expected_id = parent["matrix_id"]
    elif kind == "lane":
        parent = validate_lane(parent, str(parent_path))
        expected_schema = LANE_SCHEMA_VERSION
        expected_id = parent["lane_id"]
    else:  # pragma: no cover - defensive helper contract
        raise ContractError(f"unsupported parent asset kind {kind!r}")
    if receipt["schema_version"] != expected_schema:
        raise ContractError(f"{label}.schema_version does not match canonical {kind}")
    if receipt["asset_id"] != expected_id:
        raise ContractError(f"{label}.asset_id does not match canonical {kind}")
    return parent


def verify_campaign_matrix_binding(root: Path, path: Path, matrix: Mapping[str, Any]) -> None:
    """Require an exact canonical matrix or a concrete parent-bound derivative."""

    if canonical_asset_reference(root, path, kind="matrix", label="matrix path") is not None:
        return
    if "parent_contract" not in matrix or "parent_asset" not in matrix:
        raise ContractError(
            "a noncanonical campaign matrix must carry canonical parent_contract and parent_asset receipts"
        )
    verify_parent_contract_receipt(root, matrix["parent_contract"], "matrix.parent_contract")
    parent = verify_parent_asset_receipt(
        root,
        matrix["parent_asset"],
        kind="matrix",
        label="matrix.parent_asset",
    )
    # A concrete matrix may materialize cells, but cannot quietly relax the
    # host, model class, or preflight envelope selected by its reviewed parent.
    for field in ("tier", "measurement_mode", "hardware_requirement", "model", "preflight"):
        if matrix.get(field) != parent.get(field):
            raise ContractError(f"concrete campaign matrix {field} drifts from its canonical parent")


def verify_campaign_lane_binding(root: Path, path: Path, lane: Mapping[str, Any]) -> None:
    """Require an exact canonical lane or an available pinned derivative."""

    if canonical_asset_reference(root, path, kind="lane", label="lane path") is not None:
        return
    if "parent_contract" not in lane or "parent_asset" not in lane:
        raise ContractError(
            "a noncanonical campaign lane must carry canonical parent_contract and parent_asset receipts"
        )
    verify_parent_contract_receipt(root, lane["parent_contract"], "lane.parent_contract")
    parent = verify_parent_asset_receipt(
        root,
        lane["parent_asset"],
        kind="lane",
        label="lane.parent_asset",
    )
    if lane.get("lane_id") != parent.get("lane_id"):
        raise ContractError("concrete campaign lane_id must preserve its canonical lane identity")
    if lane.get("role") != parent.get("role"):
        raise ContractError("concrete campaign lane role must preserve its canonical lane role")
    lane_engine = _object(lane.get("engine"), "concrete campaign lane.engine")
    parent_engine = _object(parent.get("engine"), "canonical parent lane.engine")
    for field in ("name", "backend"):
        if lane_engine.get(field) != parent_engine.get(field):
            raise ContractError(f"concrete campaign lane engine.{field} drifts from canonical parent")


def verify_campaign_request_binding(root: Path, manifest: Mapping[str, Any]) -> None:
    """Request artifacts are inherently concrete and must bind to C01 v1."""

    if "parent_contract" not in manifest:
        raise ContractError("campaign request manifest must carry a canonical parent_contract receipt")
    verify_parent_contract_receipt(root, manifest["parent_contract"], "request_manifest.parent_contract")


def canonical_preflight_script_receipt(root: Path) -> dict[str, str]:
    path = root / CANONICAL_PREFLIGHT_RELATIVE_PATH
    if sha256_file(path) != CANONICAL_PREFLIGHT_SHA256:
        raise ContractError("canonical preflight script SHA-256 drift")
    return {
        "path": CANONICAL_PREFLIGHT_RELATIVE_PATH,
        "sha256": CANONICAL_PREFLIGHT_SHA256,
    }


def verify_canonical_preflight_script_receipt(root: Path, value: Any, label: str) -> Mapping[str, Any]:
    receipt = _object(value, label)
    expect_keys(receipt, label, required=("path", "sha256"))
    if receipt["path"] != CANONICAL_PREFLIGHT_RELATIVE_PATH:
        raise ContractError(f"{label}.path must name benchmarks/scripts/preflight.sh")
    if receipt["sha256"] != CANONICAL_PREFLIGHT_SHA256:
        raise ContractError(f"{label}.sha256 must pin the reviewed preflight script")
    canonical_preflight_script_receipt(root)
    return receipt


def load_preflight_receipt(path: Path, source: Mapping[str, Any]) -> dict[str, str]:
    """Parse the full reviewed preflight stdout, with no optional fields.

    This validates a snapshot, not an assertion in a campaign plan.  Callers
    must still bind its file digest and re-read it before claiming a result.
    """

    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"{path}: preflight receipt must be a regular file")
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ContractError(f"cannot read preflight receipt {path}: {error}") from error
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or "=" not in line:
            raise ContractError(f"{path}:{line_number}: preflight receipt must contain key=value rows")
        key, item = line.split("=", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or not item or key in values:
            raise ContractError(f"{path}:{line_number}: invalid or duplicate preflight receipt key")
        values[key] = item
    missing = sorted(PREFLIGHT_REQUIRED_KEYS - set(values))
    extra = sorted(set(values) - PREFLIGHT_REQUIRED_KEYS)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("extra=" + ", ".join(extra))
        raise ContractError(f"{path}: preflight receipt has closed-field drift: {'; '.join(details)}")
    revision = source.get("git_revision")
    if not isinstance(revision, str) or not GIT_REVISION_RE.fullmatch(revision):
        raise ContractError("preflight source receipt must contain a full lowercase Git revision")
    if values["git_revision"] != revision:
        raise ContractError(f"{path}: preflight Git revision differs from campaign source")
    for key, expected in PREFLIGHT_EXACT_VALUES.items():
        if values[key] != expected:
            raise ContractError(f"{path}: preflight {key} must be {expected!r}")
    integer_limits = {
        "memory_used_mib": (0, 256),
        "temperature_c": (0, 50),
        "staging_available_bytes": (20 * 1024**3, None),
    }
    for key, (minimum, maximum) in integer_limits.items():
        try:
            parsed = int(values[key])
        except ValueError as error:
            raise ContractError(f"{path}: preflight {key} must be an integer") from error
        if parsed < minimum or (maximum is not None and parsed > maximum):
            raise ContractError(f"{path}: preflight {key} is outside its contract bound")
    try:
        power_limit = float(values["power_limit_w"])
    except ValueError as error:
        raise ContractError(f"{path}: preflight power_limit_w must be numeric") from error
    if not math.isfinite(power_limit) or power_limit <= 0.0:
        raise ContractError(f"{path}: preflight power_limit_w must be finite and positive")
    if int(values["staging_available_bytes"]) < int(values["staging_minimum_bytes"]):
        raise ContractError(f"{path}: preflight staging filesystem is below its contract minimum")
    return values


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a JSON array")
    return value


def _string(value: Any, label: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    if pattern is not None and not pattern.fullmatch(value):
        raise ContractError(f"{label} has invalid format")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ContractError(f"{label} must be >= {minimum:g}")
    return result


def expect_keys(
    value: Mapping[str, Any],
    label: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise ContractError(f"{label} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise ContractError(f"{label} has unknown field(s): {', '.join(unknown)}")


def validate_sha256(value: Any, label: str) -> str:
    return _string(value, label, pattern=SHA256_RE)


def validate_identifier(value: Any, label: str) -> str:
    return _string(value, label, pattern=IDENTIFIER_RE)


def path_for_plan(root: Path, path: Path) -> str:
    """Return a repository-relative path, rejecting paths outside the checkout."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise ContractError(f"{path} must be inside repository root {root}") from error


def campaign_artifact_workspace(root: Path) -> Path:
    """Return the one declared ignored workspace without following a link.

    The workspace is an execution-evidence boundary, not a generic temporary
    directory.  A symlink at any existing component would weaken the
    repository containment promise, so reject it before consumers read or
    create an artifact below the workspace.
    """

    resolved_root = root.resolve()
    workspace = resolved_root / CAMPAIGN_ARTIFACT_WORKSPACE_RELATIVE_PATH
    cursor = resolved_root
    for component in Path(CAMPAIGN_ARTIFACT_WORKSPACE_RELATIVE_PATH).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ContractError("campaign artifact workspace must not be a symbolic link")
    resolved_workspace = workspace.resolve(strict=False)
    try:
        resolved_workspace.relative_to(resolved_root)
    except ValueError as error:  # defensive for unusual filesystem races
        raise ContractError("campaign artifact workspace must stay inside repository root") from error
    return workspace


def require_campaign_artifact_path(root: Path, path: Path, label: str) -> Path:
    """Require a path strictly below the campaign workspace without escape.

    Explicit ``.``/``..`` components and a link escape are rejected.  The
    returned path is physically resolved after the containment checks,
    suitable for subsequent regular-file validation or create-only output.
    """

    resolved_root = root.resolve()
    workspace = campaign_artifact_workspace(resolved_root)
    raw = path if path.is_absolute() else resolved_root / path
    if any(component in {".", ".."} for component in raw.parts):
        raise ContractError(f"{label} must not contain dot path components")
    if raw.is_symlink():
        raise ContractError(f"{label} must not be a symbolic link")
    # Normalize filesystem aliases outside the repository first (macOS often
    # presents temporary paths as both /var and /private/var), then enforce
    # the physical workspace boundary.  A link that escapes the workspace is
    # therefore rejected rather than merely passing a lexical prefix check.
    resolved = raw.resolve(strict=False)
    resolved_workspace = workspace.resolve(strict=False)
    try:
        relative = resolved.relative_to(resolved_workspace)
    except ValueError as error:
        raise ContractError(
            f"{label} must stay inside {CAMPAIGN_ARTIFACT_WORKSPACE_RELATIVE_PATH}"
        ) from error
    if not relative.parts:
        raise ContractError(f"{label} must name a file below the campaign artifact workspace")
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ContractError(f"{label} must stay inside repository root") from error
    return resolved


def create_only_write(path: Path, data: bytes) -> None:
    """Write exactly once without replacing a prior campaign artifact."""

    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite existing artifact {path}")
    if not path.parent.is_dir():
        raise ContractError(f"output parent does not exist: {path.parent}")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise ContractError(f"cannot create artifact {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A concurrent replacement or post-create I/O failure must leave the
        # observed path for inspection.  Unlinking by pathname here could
        # delete another process's replacement artifact.
        raise


def _copy_json(value: Any) -> Any:
    """Canonical round-trip used only for plan values that originate in JSON."""

    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def validate_contract(value: Any, label: str = "contract") -> Mapping[str, Any]:
    contract = _object(value, label)
    expect_keys(
        contract,
        label,
        required=(
            "schema_version",
            "contract_id",
            "required_independent_runs",
            "orders",
            "m4",
            "m5",
            "required_environment_keys",
        ),
        optional=(
            "$schema",
            "status",
            "purpose",
            "description",
            "historical_baseline",
            "current_competitor",
            "identity_requirements",
            "measurement_requirements",
            "preflight",
            "tiers",
            "matrix_manifests",
            "lane_manifests",
            "campaign_admission",
            "result_states",
        ),
    )
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ContractError(f"{label}.schema_version must be {CONTRACT_SCHEMA_VERSION}")
    validate_identifier(contract["contract_id"], f"{label}.contract_id")
    _integer(
        contract["required_independent_runs"],
        f"{label}.required_independent_runs",
        minimum=2,
    )
    orders = _array(contract["orders"], f"{label}.orders")
    if orders != ["AB", "BA"]:
        raise ContractError(f"{label}.orders must be exactly ['AB', 'BA']")
    environment_keys = _array(
        contract["required_environment_keys"],
        f"{label}.required_environment_keys",
    )
    if not environment_keys or len(environment_keys) != len(set(environment_keys)):
        raise ContractError(
            f"{label}.required_environment_keys must be a non-empty unique list"
        )
    for index, key in enumerate(environment_keys):
        validate_identifier(key, f"{label}.required_environment_keys[{index}]")

    for verdict, lower_goodput in (("m4", 0.0), ("m5", 1.0)):
        rule = _object(contract[verdict], f"{label}.{verdict}")
        expect_keys(
            rule,
            f"{label}.{verdict}",
            required=("ttft_p95_ratio_max", "tpot_p95_ratio_max"),
            optional=(
                "slo_goodput_ratio_min",
                "peak_vram_ratio_max",
                "primary_cell_ids",
                "geometric_mean_required",
                "failure_count_max",
                "token_mismatch_count_max",
                "required_cell_selector",
                "aggregate",
            ),
        )
        finite_number(rule["ttft_p95_ratio_max"], f"{label}.{verdict}.ttft_p95_ratio_max", minimum=0.0)
        finite_number(rule["tpot_p95_ratio_max"], f"{label}.{verdict}.tpot_p95_ratio_max", minimum=0.0)
        if "slo_goodput_ratio_min" in rule:
            finite_number(
                rule["slo_goodput_ratio_min"],
                f"{label}.{verdict}.slo_goodput_ratio_min",
                minimum=lower_goodput,
            )
        if "peak_vram_ratio_max" in rule:
            finite_number(
                rule["peak_vram_ratio_max"],
                f"{label}.{verdict}.peak_vram_ratio_max",
                minimum=0.0,
            )
    if "identity_requirements" in contract:
        identity = _object(contract["identity_requirements"], f"{label}.identity_requirements")
        expect_keys(
            identity,
            f"{label}.identity_requirements",
            required=(
                "candidate_lane_id",
                "baseline_lane_id",
                "same_campaign_fields",
                "token_equivalence_policy",
            ),
        )
        validate_identifier(identity["candidate_lane_id"], f"{label}.identity_requirements.candidate_lane_id")
        validate_identifier(identity["baseline_lane_id"], f"{label}.identity_requirements.baseline_lane_id")
        fields = _array(identity["same_campaign_fields"], f"{label}.identity_requirements.same_campaign_fields")
        if not fields or len(fields) != len(set(fields)):
            raise ContractError(f"{label}.identity_requirements.same_campaign_fields must be non-empty and unique")
        for index, field in enumerate(fields):
            validate_identifier(field, f"{label}.identity_requirements.same_campaign_fields[{index}]")
        _string(identity["token_equivalence_policy"], f"{label}.identity_requirements.token_equivalence_policy")
    if "measurement_requirements" in contract:
        measurement = _object(contract["measurement_requirements"], f"{label}.measurement_requirements")
        expect_keys(
            measurement,
            f"{label}.measurement_requirements",
            required=(
                "percentile_method",
                "statistics",
                "warmup_excluded",
                "engine_and_http_separate",
                "failure_samples_excluded_from_success_percentiles",
            ),
        )
        if measurement["percentile_method"] not in {"r7", "nearest-rank"}:
            raise ContractError(f"{label}.measurement_requirements.percentile_method is unsupported")
        statistics = _array(measurement["statistics"], f"{label}.measurement_requirements.statistics")
        if not statistics or not all(isinstance(item, str) and item for item in statistics):
            raise ContractError(f"{label}.measurement_requirements.statistics must be a non-empty string array")
        for field in (
            "warmup_excluded",
            "engine_and_http_separate",
            "failure_samples_excluded_from_success_percentiles",
        ):
            if measurement[field] is not True:
                raise ContractError(f"{label}.measurement_requirements.{field} must be true")
    for field in ("matrix_manifests", "lane_manifests"):
        if field in contract:
            manifests = _array(contract[field], f"{label}.{field}")
            if not manifests or len(manifests) != len(set(manifests)):
                raise ContractError(f"{label}.{field} must be a non-empty unique path list")
            for index, path in enumerate(manifests):
                _string(path, f"{label}.{field}[{index}]")
    if "campaign_admission" in contract:
        admission = _object(contract["campaign_admission"], f"{label}.campaign_admission")
        expect_keys(
            admission,
            f"{label}.campaign_admission",
            required=(
                "current_competitor_pin_required",
                "model_identity_pin_required",
                "request_manifest_schema_version",
                "execution_plan_schema_version",
                "raw_record_schema_version",
                "unknown_fields_rejected",
            ),
        )
        for field in (
            "current_competitor_pin_required",
            "model_identity_pin_required",
            "unknown_fields_rejected",
        ):
            if admission[field] is not True:
                raise ContractError(f"{label}.campaign_admission.{field} must be true")
        for field in (
            "request_manifest_schema_version",
            "execution_plan_schema_version",
            "raw_record_schema_version",
        ):
            _string(admission[field], f"{label}.campaign_admission.{field}")
    if "result_states" in contract:
        states = _array(contract["result_states"], f"{label}.result_states")
        if states != ["passed", "partial-win", "failed", "incomparable"]:
            raise ContractError(f"{label}.result_states must be the closed result-state sequence")
    return contract


def validate_lane(value: Any, label: str = "lane") -> Mapping[str, Any]:
    lane = _object(value, label)
    expect_keys(
        lane,
        label,
        required=("schema_version", "lane_id", "availability", "engine", "command"),
        optional=(
            "$schema",
            "description",
            "role",
            "model_support",
            "environment",
            "artifact_requirements",
            "artifact_receipts",
            "parent_contract",
            "parent_asset",
            "materialization",
        ),
    )
    if lane["schema_version"] != LANE_SCHEMA_VERSION:
        raise ContractError(f"{label}.schema_version must be {LANE_SCHEMA_VERSION}")
    validate_identifier(lane["lane_id"], f"{label}.lane_id")
    if lane["availability"] not in {
        "available",
        "contract-only",
        "campaign-pin-required",
    }:
        raise ContractError(f"{label}.availability is unsupported")
    engine = _object(lane["engine"], f"{label}.engine")
    if not engine:
        raise ContractError(f"{label}.engine must not be empty")
    command = _object(lane["command"], f"{label}.command")
    if command.get("status") not in {
        "available",
        "contract-only",
        "campaign-pin-required",
        "contract-template",
    }:
        raise ContractError(f"{label}.command.status is unsupported")
    argv = command.get("argv")
    if argv is not None:
        argv_array = _array(argv, f"{label}.command.argv")
        if not argv_array or not all(isinstance(part, str) and part for part in argv_array):
            raise ContractError(f"{label}.command.argv must be a non-empty string array")
    has_parent_contract = "parent_contract" in lane
    has_parent_asset = "parent_asset" in lane
    if has_parent_contract != has_parent_asset:
        raise ContractError(f"{label} must carry parent_contract and parent_asset together")
    if has_parent_contract:
        validate_parent_contract_receipt(lane["parent_contract"], f"{label}.parent_contract")
        validate_parent_asset_receipt(lane["parent_asset"], f"{label}.parent_asset")
    if "artifact_requirements" in lane:
        engine_required = ("name", "backend", "pin_status", "version", "revision", "dependency_lock_sha256")
        expect_keys(engine, f"{label}.engine", required=engine_required)
        expect_keys(
            command,
            f"{label}.command",
            required=("status", "argv", "required_placeholders", "output_format"),
        )
        placeholders = _array(command["required_placeholders"], f"{label}.command.required_placeholders")
        fully_materialized = lane["availability"] == "available" and "materialization" in lane
        if (not fully_materialized and not placeholders) or len(placeholders) != len(set(placeholders)):
            raise ContractError(
                f"{label}.command.required_placeholders must be "
                f"{'empty' if fully_materialized else 'non-empty'} and unique"
            )
        for index, placeholder in enumerate(placeholders):
            validate_identifier(placeholder, f"{label}.command.required_placeholders[{index}]")
        _string(command["output_format"], f"{label}.command.output_format")
        artifacts = _array(lane["artifact_requirements"], f"{label}.artifact_requirements")
        if not artifacts or not all(isinstance(item, str) and item for item in artifacts):
            raise ContractError(f"{label}.artifact_requirements must be a non-empty string array")
    if lane["availability"] == "available":
        engine_required = (
            "name",
            "backend",
            "pin_status",
            "version",
            "revision",
            "dependency_lock_sha256",
        )
        expect_keys(engine, f"{label}.engine", required=engine_required)
        if engine["pin_status"] != "pinned":
            raise ContractError(f"{label}.engine.pin_status must be pinned when available")
        for field in ("version", "revision"):
            _string(engine[field], f"{label}.engine.{field}")
        validate_sha256(engine["dependency_lock_sha256"], f"{label}.engine.dependency_lock_sha256")
        if command.get("status") != "available":
            raise ContractError(f"{label}.command.status must be available when lane is available")
        if "artifact_receipts" not in lane:
            raise ContractError(f"{label}.artifact_receipts are required for an available lane")
        receipts = _object(lane["artifact_receipts"], f"{label}.artifact_receipts")
        receipt_keys = (
            "executable_sha256",
            "source_or_wheel_sha256",
            "dependency_lock_sha256",
            "runtime_options_sha256",
            "model_identity_sha256",
            "tokenizer_identity_sha256",
        )
        expect_keys(receipts, f"{label}.artifact_receipts", required=receipt_keys)
        for field in receipt_keys:
            validate_sha256(receipts[field], f"{label}.artifact_receipts.{field}")
        if receipts["dependency_lock_sha256"] != engine["dependency_lock_sha256"]:
            raise ContractError(
                f"{label}.artifact_receipts.dependency_lock_sha256 must match engine pin"
            )
    elif "artifact_receipts" in lane:
        raise ContractError(f"{label}.artifact_receipts require availability: available")
    if "materialization" in lane:
        materialization = _object(lane["materialization"], f"{label}.materialization")
        expect_keys(
            materialization,
            f"{label}.materialization",
            required=(
                "schema_version",
                "campaign_id",
                "source_git_revision",
                "immutable_input_path",
                "immutable_input_sha256",
                "expanded_argv_sha256",
                "template_path",
                "template_sha256",
            ),
        )
        if materialization["schema_version"] != "riley.competitive.lane-materialization.v1":
            raise ContractError(f"{label}.materialization.schema_version is unsupported")
        validate_identifier(materialization["campaign_id"], f"{label}.materialization.campaign_id")
        revision = _string(materialization["source_git_revision"], f"{label}.materialization.source_git_revision")
        if not GIT_REVISION_RE.fullmatch(revision):
            raise ContractError(f"{label}.materialization.source_git_revision must be full lowercase 40-hex")
        _string(materialization["immutable_input_path"], f"{label}.materialization.immutable_input_path")
        validate_sha256(
            materialization["immutable_input_sha256"],
            f"{label}.materialization.immutable_input_sha256",
        )
        validate_sha256(
            materialization["expanded_argv_sha256"],
            f"{label}.materialization.expanded_argv_sha256",
        )
        _string(materialization["template_path"], f"{label}.materialization.template_path")
        validate_sha256(materialization["template_sha256"], f"{label}.materialization.template_sha256")
        if lane["availability"] != "available":
            raise ContractError(f"{label}.materialization requires availability: available")
    return lane


def normalise_sampling_identity(
    value: Any,
    label: str,
    *,
    allow_unparameterized_seeded: bool = False,
) -> dict[str, Any]:
    """Return a closed sampling profile suitable for exact identity checks."""

    if isinstance(value, str):
        profile: Mapping[str, Any] = {"id": value}
    else:
        profile = _object(value, label)
    expect_keys(profile, label, required=("id",), optional=("temperature", "top_p", "top_k"))
    sampling_id = _string(profile["id"], f"{label}.id")
    if sampling_id == "greedy":
        if len(profile) != 1:
            raise ContractError(f"{label}: greedy sampling must not carry tunable parameters")
        return {"id": "greedy"}
    if sampling_id != "seeded-top-p-v1":
        raise ContractError(f"{label}.id is unsupported")
    if "temperature" not in profile or "top_p" not in profile:
        if allow_unparameterized_seeded and set(profile) == {"id"}:
            return {"id": "seeded-top-p-v1"}
        raise ContractError(f"{label}: seeded-top-p-v1 requires temperature and top_p")
    temperature = finite_number(profile["temperature"], f"{label}.temperature", minimum=0.0)
    top_p = finite_number(profile["top_p"], f"{label}.top_p", minimum=0.0)
    if top_p <= 0.0 or top_p > 1.0:
        raise ContractError(f"{label}.top_p must be in (0, 1]")
    result: dict[str, Any] = {
        "id": "seeded-top-p-v1",
        "temperature": temperature,
        "top_p": top_p,
    }
    if "top_k" in profile:
        result["top_k"] = _integer(profile["top_k"], f"{label}.top_k", minimum=0)
    return result


def validate_sampling_seed(value: Any, seed: Any, label: str) -> dict[str, Any]:
    profile = normalise_sampling_identity(value, f"{label}.sampling")
    if profile["id"] == "greedy":
        if seed is not None:
            raise ContractError(f"{label}.seed must be null for greedy sampling")
    else:
        if seed is None:
            raise ContractError(f"{label}.seed is required for seeded-top-p-v1")
        _integer(seed, f"{label}.seed", minimum=0)
    return profile


def slo_profiles_by_id(manifest: Mapping[str, Any], label: str = "request manifest") -> dict[str, Mapping[str, Any]]:
    """Validate and index immutable per-cell SLO profiles."""

    if "slo_profiles" not in manifest:
        return {}
    profiles = _array(manifest["slo_profiles"], f"{label}.slo_profiles")
    if not profiles:
        raise ContractError(f"{label}.slo_profiles must not be empty when supplied")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_profile in enumerate(profiles):
        profile = _object(raw_profile, f"{label}.slo_profiles[{index}]")
        expect_keys(
            profile,
            f"{label}.slo_profiles[{index}]",
            required=("slo_profile_id", "ttft_ms_max", "tpot_ms_max"),
        )
        profile_id = validate_identifier(profile["slo_profile_id"], f"{label}.slo_profiles[{index}].slo_profile_id")
        if profile_id in result:
            raise ContractError(f"{label}.slo_profiles has duplicate ID {profile_id!r}")
        finite_number(profile["ttft_ms_max"], f"{label}.slo_profiles[{index}].ttft_ms_max", minimum=0.0)
        finite_number(profile["tpot_ms_max"], f"{label}.slo_profiles[{index}].tpot_ms_max", minimum=0.0)
        if profile["ttft_ms_max"] <= 0.0 or profile["tpot_ms_max"] <= 0.0:
            raise ContractError(f"{label}.slo_profiles[{index}] limits must be positive")
        result[profile_id] = profile
    return result


def request_workload_identity(request: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Copy every request-side workload control into canonical comparison form."""

    result = {
        "request_id": validate_identifier(request["request_id"], f"{label}.request_id"),
        "prompt_token_ids_sha256": validate_sha256(
            request["prompt_token_ids_sha256"], f"{label}.prompt_token_ids_sha256"
        ),
        "prompt_tokens": _integer(request["prompt_tokens"], f"{label}.prompt_tokens", minimum=0),
        "requested_output_tokens": _integer(
            request["requested_output_tokens"], f"{label}.requested_output_tokens", minimum=1
        ),
        "sampling": validate_sampling_seed(request["sampling"], request["seed"], label),
        "seed": request["seed"],
        "eos_policy": request["eos_policy"],
        "cache_policy": request["cache_policy"],
        "arrival_schedule_id": validate_identifier(
            request["arrival_schedule_id"], f"{label}.arrival_schedule_id"
        ),
    }
    if result["eos_policy"] not in {"ignore-eos", "stop-on-eos"}:
        raise ContractError(f"{label}.eos_policy is unsupported")
    if result["cache_policy"] not in {
        "cache-off",
        "controlled-prefix-hit-50",
        "controlled-prefix-hit-90",
    }:
        raise ContractError(f"{label}.cache_policy is unsupported")
    return result


def workload_execution_receipt(workload: Mapping[str, Any]) -> dict[str, Any]:
    """The behavior that an adapter must repeat for every independent run."""

    return {field: _copy_json(workload[field]) for field in WORKLOAD_EXECUTION_FIELDS}


def build_workload_identity(
    cell: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Freeze a complete concrete workload for one cell.

    The plan carries this value and raw rows carry its execution projection.
    Request-side fields remain on individual raw requests so the checker can
    reject an adapter that reports the correct hash while actually using a
    cheaper prompt, sampling profile, or arrival schedule.
    """

    profiles = slo_profiles_by_id(manifest, label)
    profile_id = cell["slo_profile_id"]
    if profile_id is None:
        slo_profile: Any = None
    else:
        if profile_id == "campaign-pinned-required":
            raise ContractError(f"{label}: executable cell has an unpinned SLO profile")
        slo_profile = profiles.get(profile_id)
        if slo_profile is None:
            raise ContractError(f"{label}: cell references unknown SLO profile {profile_id!r}")
        slo_profile = _copy_json(slo_profile)
    identities = [
        request_workload_identity(request, f"{label}.requests[{index}]")
        for index, request in enumerate(requests)
    ]
    if len({str(request["request_id"]) for request in identities}) != len(identities):
        raise ContractError(f"{label}: workload has duplicate request IDs")
    model_identity = _object(manifest["model_identity"], f"{label}.model_identity")
    return {
        "model_identity_sha256": sha256_bytes(canonical_json_bytes(model_identity)),
        "measurement_mode": cell["measurement_mode"],
        "warm_state": cell["warm_state"],
        "arrival_mode": cell["arrival_mode"],
        "client_behavior": cell["client_behavior"],
        "cancellation_rate_percent": cell["cancellation_rate_percent"],
        "slo_profile": slo_profile,
        "requests": sorted(identities, key=lambda request: str(request["request_id"])),
    }


def validate_matrix(value: Any, label: str = "matrix") -> Mapping[str, Any]:
    matrix = _object(value, label)
    expect_keys(
        matrix,
        label,
        required=("schema_version", "matrix_id", "tier", "cells"),
        optional=(
            "$schema",
            "description",
            "model_identity_status",
            "defaults",
            "measurement_mode",
            "hardware_requirement",
            "model",
            "preflight",
            "parent_contract",
            "parent_asset",
        ),
    )
    if matrix["schema_version"] != MATRIX_SCHEMA_VERSION:
        raise ContractError(f"{label}.schema_version must be {MATRIX_SCHEMA_VERSION}")
    validate_identifier(matrix["matrix_id"], f"{label}.matrix_id")
    if matrix["tier"] not in {
        "D",
        "C",
        "S",
        "P",
        "diagnostic",
        "competitive-latency",
        "serving-slo",
        "prefix-reuse",
    }:
        raise ContractError(f"{label}.tier is unsupported")
    cells = _array(matrix["cells"], f"{label}.cells")
    if not cells:
        raise ContractError(f"{label}.cells must not be empty")
    seen: set[str] = set()
    for index, raw_cell in enumerate(cells):
        cell = _object(raw_cell, f"{label}.cells[{index}]")
        common_fields = {
            "cell_id",
            "required_for",
            "measurement_mode",
            "warm_state",
            "sampling",
            "eos_policy",
            "cache_policy",
            "arrival_mode",
            "arrival_schedule_id",
            "client_behavior",
            "cancellation_rate_percent",
            "slo_profile_id",
            "primary",
        }
        shape_fields = {
            "template",
            "axes",
            "concurrency",
            "prompt_tokens",
            "requested_output_tokens",
        }
        if set(cell) - (common_fields | shape_fields):
            unknown = ", ".join(sorted(set(cell) - (common_fields | shape_fields)))
            raise ContractError(f"{label}.cells[{index}] has unknown field(s): {unknown}")
        for required in (
            "cell_id",
            "required_for",
            "measurement_mode",
            "warm_state",
            "sampling",
            "eos_policy",
            "cache_policy",
            "arrival_mode",
            "arrival_schedule_id",
            "client_behavior",
            "cancellation_rate_percent",
            "slo_profile_id",
            "primary",
        ):
            if required not in cell:
                raise ContractError(f"{label}.cells[{index}] is missing {required}")
        cell_id = _string(
            cell["cell_id"],
            f"{label}.cells[{index}].cell_id",
            pattern=TEMPLATE_IDENTIFIER_RE if cell.get("template") else IDENTIFIER_RE,
        )
        if cell_id in seen:
            raise ContractError(f"{label}.cells has duplicate cell_id {cell_id!r}")
        seen.add(cell_id)
        if cell["measurement_mode"] not in {"engine-only", "http-streaming"}:
            raise ContractError(
                f"{label}.cells[{index}].measurement_mode is unsupported"
            )
        required_for = _array(cell["required_for"], f"{label}.cells[{index}].required_for")
        if not required_for or len(required_for) != len(set(required_for)):
            raise ContractError(f"{label}.cells[{index}].required_for must be non-empty and unique")
        if any(entry not in {"diagnostic", "m4", "m5", "s1"} for entry in required_for):
            raise ContractError(f"{label}.cells[{index}].required_for is unsupported")
        if cell["warm_state"] not in {"cold", "warm"}:
            raise ContractError(f"{label}.cells[{index}].warm_state is unsupported")
        normalise_sampling_identity(
            cell["sampling"],
            f"{label}.cells[{index}].sampling",
            allow_unparameterized_seeded=True,
        )
        if cell["eos_policy"] not in {"ignore-eos", "stop-on-eos"}:
            raise ContractError(f"{label}.cells[{index}].eos_policy is unsupported")
        if cell["cache_policy"] not in {
            "cache-off",
            "controlled-prefix-hit-50",
            "controlled-prefix-hit-90",
        }:
            raise ContractError(f"{label}.cells[{index}].cache_policy is unsupported")
        if cell["arrival_mode"] not in {"closed-loop", "open-loop"}:
            raise ContractError(f"{label}.cells[{index}].arrival_mode is unsupported")
        _string(
            cell["arrival_schedule_id"],
            f"{label}.cells[{index}].arrival_schedule_id",
            pattern=TEMPLATE_IDENTIFIER_RE if cell.get("template") else IDENTIFIER_RE,
        )
        if cell["client_behavior"] not in {"normal", "disconnect-after-first-chunk", "backpressure"}:
            raise ContractError(f"{label}.cells[{index}].client_behavior is unsupported")
        cancellation_rate = _integer(
            cell["cancellation_rate_percent"],
            f"{label}.cells[{index}].cancellation_rate_percent",
            minimum=0,
        )
        if cancellation_rate > 100:
            raise ContractError(f"{label}.cells[{index}].cancellation_rate_percent must be <= 100")
        if cell["slo_profile_id"] is not None:
            _string(
                cell["slo_profile_id"],
                f"{label}.cells[{index}].slo_profile_id",
                pattern=IDENTIFIER_RE,
            )
        if not isinstance(cell["primary"], bool):
            raise ContractError(f"{label}.cells[{index}].primary must be boolean")
        if "template" in cell:
            if cell["template"] is not True:
                raise ContractError(f"{label}.cells[{index}].template must be true")
            axes = _object(cell.get("axes"), f"{label}.cells[{index}].axes")
            if not axes:
                raise ContractError(f"{label}.cells[{index}].axes must not be empty")
            for axis, values in axes.items():
                validate_identifier(axis, f"{label}.cells[{index}].axes key")
                array = _array(values, f"{label}.cells[{index}].axes.{axis}")
                if not array:
                    raise ContractError(f"{label}.cells[{index}].axes.{axis} must not be empty")
        elif "axes" in cell:
            raise ContractError(f"{label}.cells[{index}].axes requires template: true")
        else:
            for field in ("concurrency", "prompt_tokens", "requested_output_tokens"):
                if field not in cell:
                    raise ContractError(f"{label}.cells[{index}] concrete workload is missing {field}")
                _integer(cell[field], f"{label}.cells[{index}].{field}", minimum=1)
            if matrix["tier"] in {"S", "serving-slo"} and cell["slo_profile_id"] in {
                None,
                "campaign-pinned-required",
            }:
                raise ContractError(
                    f"{label}.cells[{index}] serving workload requires a concrete SLO profile"
                )
    if "hardware_requirement" in matrix:
        hardware = _object(matrix["hardware_requirement"], f"{label}.hardware_requirement")
        expect_keys(
            hardware,
            f"{label}.hardware_requirement",
            required=("gpu_count", "gpu_architecture", "gpu_model", "dtype", "execution_host_requirement"),
        )
        if hardware["gpu_count"] != 1 or hardware["gpu_architecture"] != "sm89":
            raise ContractError(f"{label}.hardware_requirement must require one sm89 GPU")
        if hardware["gpu_model"] != "NVIDIA GeForce RTX 4090" or hardware["dtype"] != "bf16":
            raise ContractError(f"{label}.hardware_requirement must require RTX 4090 BF16")
    if "preflight" in matrix:
        preflight = _object(matrix["preflight"], f"{label}.preflight")
        expect_keys(
            preflight,
            f"{label}.preflight",
            required=(
                "fresh_process_per_independent_run",
                "clean_git_required",
                "thermal_stabilization_required",
                "exclusive_gpu_required",
                "warmup_excluded_from_measurements",
            ),
        )
        if any(value is not True for value in preflight.values()):
            raise ContractError(f"{label}.preflight requires all safety gates to be true")
    has_parent_contract = "parent_contract" in matrix
    has_parent_asset = "parent_asset" in matrix
    if has_parent_contract != has_parent_asset:
        raise ContractError(f"{label} must carry parent_contract and parent_asset together")
    if has_parent_contract:
        validate_parent_contract_receipt(matrix["parent_contract"], f"{label}.parent_contract")
        validate_parent_asset_receipt(matrix["parent_asset"], f"{label}.parent_asset")
    return matrix


def matrix_cells(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return concrete cells; templates remain explicitly non-executable.

    C01's latency and serving matrices intentionally describe model classes
    before a campaign chooses a concrete checkpoint.  A plan cannot silently
    turn such a template into a workload, so we retain it and make the caller
    reject it until a campaign-specific matrix is supplied.
    """

    result: list[dict[str, Any]] = []
    for raw_cell in matrix["cells"]:
        cell = _copy_json(raw_cell)
        if cell.get("template"):
            cell["executable"] = False
        else:
            cell["executable"] = True
        result.append(cell)
    return result


def validate_request_manifest(value: Any, label: str = "request manifest") -> Mapping[str, Any]:
    manifest = _object(value, label)
    expect_keys(
        manifest,
        label,
        required=("schema_version", "manifest_id", "model_identity", "request_sets"),
        optional=("description", "slo_profiles", "parent_contract"),
    )
    if manifest["schema_version"] != REQUESTS_SCHEMA_VERSION:
        raise ContractError(f"{label}.schema_version must be {REQUESTS_SCHEMA_VERSION}")
    validate_identifier(manifest["manifest_id"], f"{label}.manifest_id")
    if "parent_contract" in manifest:
        validate_parent_contract_receipt(manifest["parent_contract"], f"{label}.parent_contract")
    model = _object(manifest["model_identity"], f"{label}.model_identity")
    expect_keys(
        model,
        f"{label}.model_identity",
        required=("model_id", "model_revision", "tokenizer_revision", "tokenizer_files_sha256"),
        optional=("weights_sha256", "model_weights_sha256", "tokenizer_aggregate_sha256"),
    )
    for key in ("model_id", "model_revision", "tokenizer_revision"):
        _string(model[key], f"{label}.model_identity.{key}")
    weight_fields = [field for field in ("weights_sha256", "model_weights_sha256") if field in model]
    if len(weight_fields) != 1:
        raise ContractError(
            f"{label}.model_identity must contain exactly one of weights_sha256 or model_weights_sha256"
        )
    validate_sha256(model[weight_fields[0]], f"{label}.model_identity.{weight_fields[0]}")
    if "tokenizer_aggregate_sha256" in model:
        validate_sha256(
            model["tokenizer_aggregate_sha256"],
            f"{label}.model_identity.tokenizer_aggregate_sha256",
        )
    tokenizer_files = _object(
        model["tokenizer_files_sha256"],
        f"{label}.model_identity.tokenizer_files_sha256",
    )
    if not tokenizer_files:
        raise ContractError(f"{label}.model_identity.tokenizer_files_sha256 must not be empty")
    for filename, file_hash in tokenizer_files.items():
        _string(filename, f"{label}.model_identity.tokenizer_files_sha256 key")
        validate_sha256(file_hash, f"{label}.model_identity.tokenizer_files_sha256.{filename}")

    slo_profiles_by_id(manifest, label)

    request_sets = _array(manifest["request_sets"], f"{label}.request_sets")
    if not request_sets:
        raise ContractError(f"{label}.request_sets must not be empty")
    seen_cells: set[str] = set()
    for index, raw_set in enumerate(request_sets):
        request_set = _object(raw_set, f"{label}.request_sets[{index}]")
        expect_keys(request_set, f"{label}.request_sets[{index}]", required=("cell_id", "requests"))
        cell_id = validate_identifier(request_set["cell_id"], f"{label}.request_sets[{index}].cell_id")
        if cell_id in seen_cells:
            raise ContractError(f"{label}.request_sets has duplicate cell_id {cell_id!r}")
        seen_cells.add(cell_id)
        requests = _array(request_set["requests"], f"{label}.request_sets[{index}].requests")
        if not requests:
            raise ContractError(f"{label}.request_sets[{index}].requests must not be empty")
        seen_requests: set[str] = set()
        for request_index, raw_request in enumerate(requests):
            request = _object(raw_request, f"{label}.request_sets[{index}].requests[{request_index}]")
            expect_keys(
                request,
                f"{label}.request_sets[{index}].requests[{request_index}]",
                required=(
                    "request_id",
                    "prompt_token_ids_sha256",
                    "prompt_tokens",
                    "requested_output_tokens",
                    "sampling",
                    "seed",
                    "eos_policy",
                    "cache_policy",
                    "arrival_schedule_id",
                ),
            )
            request_id = validate_identifier(
                request["request_id"],
                f"{label}.request_sets[{index}].requests[{request_index}].request_id",
            )
            if request_id in seen_requests:
                raise ContractError(
                    f"{label}.request_sets[{index}].requests has duplicate request_id {request_id!r}"
                )
            seen_requests.add(request_id)
            request_workload_identity(
                request,
                f"{label}.request_sets[{index}].requests[{request_index}]",
            )
    return manifest


def request_sets_by_cell(manifest: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        str(request_set["cell_id"]): list(request_set["requests"])
        for request_set in manifest["request_sets"]
    }


def r7(values: Iterable[float], probability: float) -> float:
    """Hyndman-Fan type-7 quantile with finite input validation."""

    observations = sorted(finite_number(value, "quantile observation") for value in values)
    if not observations:
        raise ContractError("R7 quantile requires at least one observation")
    if not 0.0 <= probability <= 1.0:
        raise ContractError("R7 probability must be in [0, 1]")
    if len(observations) == 1:
        return observations[0]
    index = (len(observations) - 1) * probability
    lower = math.floor(index)
    fraction = index - lower
    if fraction == 0.0:
        return observations[lower]
    return observations[lower] + fraction * (observations[lower + 1] - observations[lower])


def nearest_rank(values: Iterable[float], probability: float) -> float:
    """Return the contract's nearest-rank percentile for finite observations."""

    observations = sorted(finite_number(value, "quantile observation") for value in values)
    if not observations:
        raise ContractError("nearest-rank quantile requires at least one observation")
    if not 0.0 < probability <= 1.0:
        raise ContractError("nearest-rank probability must be in (0, 1]")
    rank = max(1, math.ceil(probability * len(observations)))
    return observations[rank - 1]


def geometric_mean(values: Iterable[float], label: str) -> float:
    observations = [finite_number(value, label, minimum=0.0) for value in values]
    if not observations:
        raise ContractError(f"{label} requires at least one observation")
    if any(value == 0.0 for value in observations):
        return 0.0
    return math.exp(math.fsum(math.log(value) for value in observations) / len(observations))


def expand_jsonl_paths(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_symlink():
            raise ContractError(f"raw evidence input must not be a symbolic link: {path}")
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            for candidate in sorted(path.rglob("*.jsonl")):
                if candidate.is_symlink() or not candidate.is_file():
                    raise ContractError(f"raw evidence must be a regular non-link JSONL file: {candidate}")
                result.append(candidate)
        else:
            raise ContractError(f"raw evidence input does not exist: {path}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in result:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ContractError("no raw JSONL evidence files were found")
    return unique


def load_jsonl(paths: Sequence[Path]) -> list[tuple[Path, int, Mapping[str, Any]]]:
    rows: list[tuple[Path, int, Mapping[str, Any]]] = []
    for path in expand_jsonl_paths(paths):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ContractError(f"{path}:{line_number}: blank JSONL rows are forbidden")
                    try:
                        parsed = json.loads(line, object_pairs_hook=reject_duplicate_keys)
                    except json.JSONDecodeError as error:
                        raise ContractError(
                            f"{path}:{line_number}: invalid JSON: {error.msg}"
                        ) from error
                    rows.append((path, line_number, _object(parsed, f"{path}:{line_number}")))
        except OSError as error:
            raise ContractError(f"{path}: cannot read JSONL: {error}") from error
    if not rows:
        raise ContractError("raw JSONL evidence is empty")
    return rows
