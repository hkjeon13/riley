#!/usr/bin/env python3
"""Prepare one new private evidence root for native-fallback lifecycle v5.

This program has no GPU, network, process-launch, or qualification authority.
It creates the fresh trusted evidence root, creates the source-owned audit
child, validates exactly one canonical native-fallback contract-v2 scenario,
and copies that exact contract into the root through a create-only descriptor.
The future authenticated lifecycle-v5 runner subsequently passes only this
frozen copy to the raw scenario producer.  This helper creates no bind request,
terminal marker, receipt, GPU workload, or qualification result.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

import provenance_v2_common as common


CONTRACT_VERSION = "riley.c02-raw-soak-runner-contract.v2"
DEFAULT_AUDIT_DIRECTORY_NAME = "source-audit"
DEFAULT_CONTRACT_COPY_NAME = "fallback-lifecycle-scenario-contract.json"
# This frozen copy must remain directly consumable by the v2 raw producer.
MAX_CONTRACT_BYTES = 1024 * 1024
CANDIDATE_RE = re.compile(
    r"^riley-(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)-rc[1-9][0-9]*$"
)
SCENARIO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
PROFILE = "max-performance-exact"
FALLBACK_SCENARIO_ID = "exact-backend-fallback"


class LifecycleV5EvidencePreparationError(ValueError):
    """A native-fallback lifecycle-v5 root or contract is unsafe."""


def _fail(code: str, message: str) -> NoReturn:
    error = LifecycleV5EvidencePreparationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail("unexpected-field-set", f"{label} must contain exactly {sorted(fields)}")
    return value


def _absolute_regular_bytes(path: Path, label: str) -> bytes:
    raw_path = os.fspath(path)
    if (
        not os.path.isabs(raw_path)
        or "\x00" in raw_path
        or "\\" in raw_path
        or raw_path.startswith("//")
        or raw_path != os.path.normpath(raw_path)
    ):
        _fail("invalid-absolute-path", f"{label} must be a normalized absolute path")
    parent_text, name = os.path.split(raw_path)
    if not parent_text or not name:
        _fail("invalid-contract-path", f"{label} must name one regular file")
    parent_fd = _common(
        lambda: common.open_absolute_directory(Path(parent_text), f"{label} parent")
    )
    try:
        return _common(
            lambda: common.read_bounded_regular_relative(
                parent_fd,
                name,
                label,
                maximum_bytes=MAX_CONTRACT_BYTES,
            )
        )
    finally:
        os.close(parent_fd)


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    raw = os.fspath(evidence_root)
    if not os.path.isabs(raw) or raw != os.path.normpath(raw):
        _fail("invalid-absolute-path", "--evidence-root must be a normalized absolute path")
    source_root = Path(__file__).resolve().parents[2]
    try:
        Path(raw).relative_to(source_root)
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be outside the source checkout",
    )


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        _fail("invalid-contract-value", f"{label} must be a finite JSON number")
    number = float(value)
    if number < minimum or number > maximum:
        _fail(
            "invalid-contract-value",
            f"{label} must be at least {minimum} and at most {maximum}",
        )
    return number


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> None:
    if type(value) is not int or value < minimum or value > maximum:
        _fail(
            "invalid-contract-value",
            f"{label} must be an integer from {minimum} through {maximum}",
        )


def validate_one_fallback_scenario_contract(
    raw: bytes,
    *,
    candidate_id: str,
    configuration_profile: str,
) -> str:
    """Validate the closed v2 native-fallback producer contract before root creation."""

    if CANDIDATE_RE.fullmatch(candidate_id) is None:
        _fail("invalid-candidate-id", "--candidate-id is not canonical")
    if configuration_profile != PROFILE:
        _fail("invalid-configuration-profile", f"--configuration-profile must be {PROFILE}")
    row = _common(
        lambda: common.parse_canonical_json(
            raw,
            "scenario contract",
            maximum_bytes=MAX_CONTRACT_BYTES,
        )
    )
    top = _exact(
        row,
        {"schema_version", "candidate_id", "configuration_profile", "scenarios"},
        "scenario contract",
    )
    if top["schema_version"] != CONTRACT_VERSION:
        _fail("unsupported-contract-version", "scenario contract must use native fallback v2")
    if top["candidate_id"] != candidate_id:
        _fail("contract-candidate-mismatch", "scenario contract candidate differs from --candidate-id")
    if top["configuration_profile"] != configuration_profile:
        _fail(
            "contract-profile-mismatch",
            "scenario contract profile differs from --configuration-profile",
        )
    scenarios = top["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        _fail(
            "lifecycle-scenario-count",
            "native fallback lifecycle v5 requires exactly one scenario",
        )
    scenario = _exact(scenarios[0], {"scenario_id", "completion_request"}, "scenario contract.scenarios[0]")
    scenario_id = scenario["scenario_id"]
    if (
        type(scenario_id) is not str
        or len(scenario_id) > 128
        or SCENARIO_ID_RE.fullmatch(scenario_id) is None
        or scenario_id != FALLBACK_SCENARIO_ID
    ):
        _fail("invalid-scenario-id", "scenario contract must contain exact-backend-fallback")
    completion = _exact(
        scenario["completion_request"],
        {"model", "prompt", "max_tokens", "temperature", "top_p", "seed", "stream"},
        "scenario contract.scenarios[0].completion_request",
    )
    for field, maximum in (("model", 256), ("prompt", 1_048_576)):
        value = completion[field]
        if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
            _fail("invalid-contract-value", f"completion_request.{field} has an invalid length")
    _integer(completion["max_tokens"], "completion_request.max_tokens", minimum=1, maximum=65_536)
    if completion["max_tokens"] != 1:
        _fail("invalid-contract-value", "completion_request.max_tokens must be exactly 1")
    if _number(completion["temperature"], "completion_request.temperature", minimum=0, maximum=2) != 1.0:
        _fail("invalid-contract-value", "completion_request.temperature must be exactly 1.0")
    if _number(completion["top_p"], "completion_request.top_p", minimum=0, maximum=1) != 1.0:
        _fail("invalid-contract-value", "completion_request.top_p must be exactly 1.0")
    _integer(
        completion["seed"],
        "completion_request.seed",
        minimum=0,
        maximum=18_446_744_073_709_551_615,
    )
    if completion["stream"] is not False:
        _fail("streaming-not-supported", "native fallback lifecycle v5 permits only stream:false")
    return FALLBACK_SCENARIO_ID


def prepare_lifecycle_evidence(
    evidence_root: Path,
    *,
    scenario_contract: Path,
    candidate_id: str,
    configuration_profile: str,
) -> dict[str, Any]:
    """Create a fresh root/audit topology and freeze the one fallback v2 contract."""

    _assert_external_to_source_checkout(evidence_root)
    contract_raw = _absolute_regular_bytes(scenario_contract, "--scenario-contract")
    scenario_id = validate_one_fallback_scenario_contract(
        contract_raw,
        candidate_id=candidate_id,
        configuration_profile=configuration_profile,
    )
    root_fd = _common(
        lambda: common.create_private_evidence_directory(evidence_root, "--evidence-root")
    )
    try:
        audit_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                DEFAULT_AUDIT_DIRECTORY_NAME,
                "source audit directory",
            )
        )
        os.close(audit_fd)
        created = _common(
            lambda: common.write_create_only(
                root_fd,
                DEFAULT_CONTRACT_COPY_NAME,
                contract_raw,
                "frozen lifecycle scenario contract",
            )
        )
        return {
            "schema_version": "riley.c02-lifecycle-evidence-preparation.v5",
            "status": "prepared",
            "qualification_status": "not-run",
            "scenario_id": scenario_id,
            "scenario_contract": created.descriptor(
                DEFAULT_CONTRACT_COPY_NAME,
                "frozen lifecycle scenario contract",
            ).as_json(),
        }
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--scenario-contract", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--configuration-profile", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prepared = prepare_lifecycle_evidence(
            args.evidence_root,
            scenario_contract=args.scenario_contract,
            candidate_id=args.candidate_id,
            configuration_profile=args.configuration_profile,
        )
    except (LifecycleV5EvidencePreparationError, OSError) as error:
        print(f"C02 lifecycle evidence preparation refused: {error}", file=sys.stderr)
        return 2
    print(common.canonical_json_bytes(prepared).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
