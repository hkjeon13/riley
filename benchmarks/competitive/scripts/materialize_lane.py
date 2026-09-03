#!/usr/bin/env python3
"""Materialize a reviewed C01 lane template into an executable lane.

The checked-in Riley and vLLM lane manifests are deliberately templates: they
must never be handed directly to an executor.  This module accepts a closed
immutable input document and derives a campaign-local ``available`` lane.  It
does not select a real candidate, version, host, or credential; those values
are supplied by the external campaign-admission workflow.

The emitted lane remains a normal ``riley.competitive.lane.v1`` manifest, so
``run_campaign.py`` can hash and bind it without a second plan format.  Its
``materialization`` receipt makes the campaign and reviewed template binding
auditable by the execution adapter.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from competitive_common import (
    CANONICAL_ASSET_SHA256,
    CANONICAL_CONTRACT_ID,
    CANONICAL_CONTRACT_RELATIVE_PATH,
    CONTRACT_SCHEMA_VERSION,
    LANE_SCHEMA_VERSION,
    ContractError,
    canonical_asset_reference,
    canonical_json_bytes,
    create_only_write,
    expect_keys,
    load_json,
    path_for_plan,
    require_campaign_artifact_path,
    sha256_bytes,
    sha256_file,
    validate_identifier,
    validate_lane,
    validate_sha256,
    verify_parent_asset_receipt,
    verify_parent_contract_receipt,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parents[2]

MATERIALIZATION_SCHEMA_VERSION = "riley.competitive.lane-materialization.v1"
MATERIALIZATION_INPUT_SCHEMA_VERSION = "riley.competitive.lane-input.v1"
ARTIFACT_RECEIPT_FIELDS = (
    "executable_sha256",
    "source_or_wheel_sha256",
    "dependency_lock_sha256",
    "runtime_options_sha256",
    "model_identity_sha256",
    "tokenizer_identity_sha256",
)
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ContractError(f"{label} must be a non-empty single-line string")
    return value


def _git_revision(value: Any, label: str) -> str:
    revision = _nonempty_string(value, label)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ContractError(f"{label} must be a full lowercase 40-hex revision")
    return revision


def _path_inside_root(root: Path, path: Path, label: str) -> Path:
    """Resolve an artifact path without accepting a root/symlink escape."""

    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(f"{label} must stay inside repository root") from error
    return resolved


def _template_placeholders(template: Mapping[str, Any]) -> tuple[str, ...]:
    command = _mapping(template["command"], "template.command")
    argv = command.get("argv")
    if not isinstance(argv, list):  # validate_lane gives the more precise error first.
        raise ContractError("template.command.argv must be a JSON array")
    declared = command.get("required_placeholders")
    if not isinstance(declared, list):
        raise ContractError("template.command.required_placeholders must be a JSON array")
    if len(declared) != len(set(declared)):
        raise ContractError("template.command.required_placeholders repeats a placeholder")
    for value in declared:
        validate_identifier(value, "template.command.required_placeholders entry")

    observed: set[str] = set()
    for index, token in enumerate(argv):
        if not isinstance(token, str) or not token:
            raise ContractError(f"template.command.argv[{index}] must be a non-empty string")
        matches = list(PLACEHOLDER_RE.finditer(token))
        observed.update(match.group(1) for match in matches)
        remainder = PLACEHOLDER_RE.sub("", token)
        if "{" in remainder or "}" in remainder:
            raise ContractError(f"template.command.argv[{index}] has malformed placeholder syntax")
    if observed != set(declared):
        missing = sorted(set(declared) - observed)
        surplus = sorted(observed - set(declared))
        details: list[str] = []
        if missing:
            details.append("declared but unused: " + ", ".join(missing))
        if surplus:
            details.append("used but undeclared: " + ", ".join(surplus))
        raise ContractError("template command placeholder contract drift: " + "; ".join(details))
    return tuple(str(value) for value in declared)


def validate_materialization_input(
    value: Any,
    *,
    template: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the closed input used to derive an executable lane.

    The input intentionally carries only hashes and command substitutions.
    It cannot carry extra command arguments, environment overrides, or a
    mutable matrix/cell definition.
    """

    document = _mapping(value, "lane materialization input")
    expect_keys(
        document,
        "lane materialization input",
        required=(
            "schema_version",
            "campaign_id",
            "lane_id",
            "role",
            "source",
            "engine",
            "artifact_receipts",
            "command_bindings",
        ),
    )
    if document["schema_version"] != MATERIALIZATION_INPUT_SCHEMA_VERSION:
        raise ContractError(
            "lane materialization input.schema_version must be "
            f"{MATERIALIZATION_INPUT_SCHEMA_VERSION}"
        )
    validate_identifier(document["campaign_id"], "lane materialization input.campaign_id")
    if document["lane_id"] != template["lane_id"]:
        raise ContractError("lane materialization input.lane_id drifts from template")
    if document["role"] != template.get("role"):
        raise ContractError("lane materialization input.role drifts from template")

    source = _mapping(document["source"], "lane materialization input.source")
    expect_keys(source, "lane materialization input.source", required=("git_revision", "git_dirty"))
    _git_revision(source["git_revision"], "lane materialization input.source.git_revision")
    if source["git_dirty"] is not False:
        raise ContractError("lane materialization input.source.git_dirty must be false")

    engine = _mapping(document["engine"], "lane materialization input.engine")
    expect_keys(
        engine,
        "lane materialization input.engine",
        required=("version", "revision", "dependency_lock_sha256"),
    )
    _nonempty_string(engine["version"], "lane materialization input.engine.version")
    _git_revision(engine["revision"], "lane materialization input.engine.revision")
    validate_sha256(
        engine["dependency_lock_sha256"],
        "lane materialization input.engine.dependency_lock_sha256",
    )

    receipts = _mapping(document["artifact_receipts"], "lane materialization input.artifact_receipts")
    expect_keys(receipts, "lane materialization input.artifact_receipts", required=ARTIFACT_RECEIPT_FIELDS)
    for field in ARTIFACT_RECEIPT_FIELDS:
        validate_sha256(receipts[field], f"lane materialization input.artifact_receipts.{field}")
    if receipts["dependency_lock_sha256"] != engine["dependency_lock_sha256"]:
        raise ContractError(
            "lane materialization input artifact dependency_lock_sha256 must match engine"
        )

    bindings = _mapping(document["command_bindings"], "lane materialization input.command_bindings")
    required_placeholders = set(_template_placeholders(template))
    if set(bindings) != required_placeholders:
        missing = sorted(required_placeholders - set(bindings))
        surplus = sorted(set(bindings) - required_placeholders)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if surplus:
            details.append("unexpected: " + ", ".join(surplus))
        raise ContractError("lane materialization input.command_bindings must exactly match template: " + "; ".join(details))
    for key, item in bindings.items():
        _nonempty_string(item, f"lane materialization input.command_bindings.{key}")
    return document


def _expand_command(template: Mapping[str, Any], bindings: Mapping[str, Any]) -> list[str]:
    argv = _mapping(template["command"], "template.command")["argv"]
    assert isinstance(argv, list)  # checked by validate_lane/_template_placeholders

    def replace(match: re.Match[str]) -> str:
        return str(bindings[match.group(1)])

    expanded = [PLACEHOLDER_RE.sub(replace, str(token)) for token in argv]
    if any("{" in token or "}" in token for token in expanded):
        raise ContractError("materialized command still contains a placeholder")
    return expanded


def _materialization_receipt(
    *,
    root: Path,
    template_path: Path,
    immutable_input: Mapping[str, Any],
    immutable_input_path: Path,
    immutable_input_sha256: str,
    expanded_argv: Sequence[str],
) -> dict[str, str]:
    relative_path = canonical_asset_reference(
        root,
        template_path,
        kind="lane",
        label="lane template path",
    )
    if relative_path is None:
        raise ContractError("lane materialization template must be a reviewed canonical lane")
    return {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "campaign_id": str(immutable_input["campaign_id"]),
        "source_git_revision": str(immutable_input["source"]["git_revision"]),
        # This is the exact create-only input file digest, not merely a hash
        # of a deserialized object.  Whitespace or duplicate-key rewrites are
        # therefore evidence drift too.
        "immutable_input_path": path_for_plan(root, immutable_input_path),
        "immutable_input_sha256": immutable_input_sha256,
        "expanded_argv_sha256": sha256_bytes(canonical_json_bytes(list(expanded_argv))),
        "template_path": relative_path,
        "template_sha256": CANONICAL_ASSET_SHA256[relative_path],
    }


def _build_materialized_lane(
    *,
    root: Path,
    template_path: Path,
    immutable_input: Mapping[str, Any],
    immutable_input_path: Path,
    immutable_input_sha256: str,
) -> dict[str, Any]:
    """Derive a lane without reading or accepting a mutable output artifact."""

    template = validate_lane(load_json(template_path), str(template_path))
    if canonical_asset_reference(root, template_path, kind="lane", label="lane template path") is None:
        raise ContractError("lane materialization template must be a reviewed canonical lane")
    input_document = validate_materialization_input(immutable_input, template=template)
    command = _mapping(template["command"], "template.command")
    engine = _mapping(template["engine"], "template.engine")
    input_engine = _mapping(input_document["engine"], "lane materialization input.engine")
    receipts = _mapping(input_document["artifact_receipts"], "lane materialization input.artifact_receipts")

    parent_contract = {
        "path": CANONICAL_CONTRACT_RELATIVE_PATH,
        "sha256": CANONICAL_ASSET_SHA256[CANONICAL_CONTRACT_RELATIVE_PATH],
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CANONICAL_CONTRACT_ID,
    }
    template_relative = canonical_asset_reference(
        root,
        template_path,
        kind="lane",
        label="lane template path",
    )
    assert template_relative is not None
    expanded_argv = _expand_command(
        template,
        _mapping(input_document["command_bindings"], "bindings"),
    )
    lane = {
        "$schema": template.get("$schema", "../contract-v1.schema.json"),
        "schema_version": LANE_SCHEMA_VERSION,
        "lane_id": template["lane_id"],
        "role": template.get("role"),
        "availability": "available",
        "engine": {
            "name": engine["name"],
            "backend": engine["backend"],
            "pin_status": "pinned",
            "version": input_engine["version"],
            "revision": input_engine["revision"],
            "dependency_lock_sha256": input_engine["dependency_lock_sha256"],
        },
        "command": {
            "status": "available",
            "argv": expanded_argv,
            "required_placeholders": [],
            "output_format": command["output_format"],
        },
        "artifact_requirements": list(template["artifact_requirements"]),
        "artifact_receipts": dict(receipts),
        "parent_contract": parent_contract,
        "parent_asset": {
            "path": template_relative,
            "sha256": CANONICAL_ASSET_SHA256[template_relative],
            "schema_version": LANE_SCHEMA_VERSION,
            "asset_id": template["lane_id"],
        },
        "materialization": _materialization_receipt(
            root=root,
            template_path=template_path,
            immutable_input=input_document,
            immutable_input_path=immutable_input_path,
            immutable_input_sha256=immutable_input_sha256,
            expanded_argv=expanded_argv,
        ),
    }
    validate_lane(lane, "materialized lane")
    return lane


def materialize_lane(
    *,
    root: Path,
    template_path: Path,
    immutable_input_path: Path,
) -> dict[str, Any]:
    """Return a fully substituted, pinned lane from one closed input file.

    A claim-ready lane is never constructed from an in-memory dictionary: the
    receipt must name and hash a create-only input in the declared campaign
    workspace so a later verifier can derive the same argv exactly.
    """

    root = root.resolve()
    template_path = _path_inside_root(root, template_path, "lane template path")
    immutable_input_path = require_campaign_artifact_path(
        root,
        immutable_input_path,
        "lane materialization immutable input",
    )
    immutable_input_sha256 = sha256_file(immutable_input_path)
    immutable_input = load_json(immutable_input_path)
    lane = _build_materialized_lane(
        root=root,
        template_path=template_path,
        immutable_input=immutable_input,
        immutable_input_path=immutable_input_path,
        immutable_input_sha256=immutable_input_sha256,
    )
    # The generated value has no output path yet, but this verifies the same
    # source/template/input/argv provenance rules used by plan consumers.
    verify_campaign_lane_binding_value(root=root, lane=lane)
    return lane


def verify_campaign_lane_binding_value(*, root: Path, lane: Mapping[str, Any]) -> None:
    """Verify parent bindings for an in-memory materialized lane.

    ``verify_campaign_lane_binding`` operates on a path so it can distinguish
    canonical assets from derivatives.  Materialized lanes are necessarily
    derivatives, and their receipt pins the canonical template explicitly.
    """

    root = root.resolve()
    validate_lane(lane, "materialized lane")
    verify_parent_contract_receipt(root, lane["parent_contract"], "materialized lane.parent_contract")
    parent = verify_parent_asset_receipt(
        root,
        lane["parent_asset"],
        kind="lane",
        label="materialized lane.parent_asset",
    )
    materialization = _mapping(lane.get("materialization"), "materialized lane.materialization")
    expect_keys(
        materialization,
        "materialized lane.materialization",
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
    if materialization["schema_version"] != MATERIALIZATION_SCHEMA_VERSION:
        raise ContractError("materialized lane.materialization schema version drift")
    validate_identifier(materialization["campaign_id"], "materialized lane.materialization.campaign_id")
    _git_revision(materialization["source_git_revision"], "materialized lane.materialization.source_git_revision")
    immutable_input_path = require_campaign_artifact_path(
        root,
        root / str(materialization["immutable_input_path"]),
        "materialized lane.materialization.immutable_input_path",
    )
    validate_sha256(
        materialization["immutable_input_sha256"],
        "materialized lane.materialization.immutable_input_sha256",
    )
    validate_sha256(
        materialization["expanded_argv_sha256"],
        "materialized lane.materialization.expanded_argv_sha256",
    )
    if sha256_file(immutable_input_path) != materialization["immutable_input_sha256"]:
        raise ContractError("materialized lane immutable input receipt drift")
    template_path = root / str(materialization["template_path"])
    template_relative = canonical_asset_reference(
        root,
        template_path,
        kind="lane",
        label="materialized lane.materialization.template_path",
    )
    if template_relative is None or materialization["template_sha256"] != CANONICAL_ASSET_SHA256[template_relative]:
        raise ContractError("materialized lane.materialization template receipt drift")
    parent_asset = _mapping(lane["parent_asset"], "materialized lane.parent_asset")
    if (
        parent_asset.get("path") != template_relative
        or parent_asset.get("sha256") != materialization["template_sha256"]
        or parent.get("lane_id") != lane.get("lane_id")
        or parent.get("role") != lane.get("role")
    ):
        raise ContractError("materialized lane parent/template receipt drift")
    command = _mapping(lane["command"], "materialized lane.command")
    if command.get("status") != "available" or command.get("required_placeholders") != []:
        raise ContractError("materialized lane command must be fully substituted and available")
    argv = command.get("argv")
    if not isinstance(argv, list) or any("{" in str(token) or "}" in str(token) for token in argv):
        raise ContractError("materialized lane command must not contain placeholders")
    if sha256_bytes(canonical_json_bytes(argv)) != materialization["expanded_argv_sha256"]:
        raise ContractError("materialized lane expanded argv receipt drift")

    # Re-materialize the complete closed lane from the reviewed template and
    # hashed input.  Comparing canonical values catches semantic drift even if
    # somebody rebuilds a plan around a hand-edited executable, model argv, or
    # artifact receipt.
    expected_lane = _build_materialized_lane(
        root=root,
        template_path=template_path,
        immutable_input=load_json(immutable_input_path),
        immutable_input_path=immutable_input_path,
        immutable_input_sha256=str(materialization["immutable_input_sha256"]),
    )
    if canonical_json_bytes(expected_lane) != canonical_json_bytes(lane):
        raise ContractError("materialized lane contents drift from immutable input/template")


def write_materialized_lane(
    *,
    root: Path,
    template_path: Path,
    immutable_input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create a lane exactly once; never replace a campaign artifact."""

    root = root.resolve()
    template_path = _path_inside_root(root, template_path, "lane template path")
    immutable_input_path = require_campaign_artifact_path(
        root,
        immutable_input_path,
        "lane materialization immutable input",
    )
    output_path = require_campaign_artifact_path(root, output_path, "lane materialization output")
    lane = materialize_lane(
        root=root,
        template_path=template_path,
        immutable_input_path=immutable_input_path,
    )
    create_only_write(output_path, canonical_json_bytes(lane))
    return lane


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--immutable-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        lane = write_materialized_lane(
            root=arguments.repo_root.resolve(),
            template_path=arguments.template.absolute(),
            immutable_input_path=arguments.immutable_input.absolute(),
            output_path=arguments.output.absolute(),
        )
    except ContractError as error:
        print(f"materialize_lane: {error}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(lane).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
