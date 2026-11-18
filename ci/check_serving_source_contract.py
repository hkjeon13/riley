#!/usr/bin/env python3
"""Check the reviewed 0.1.0 serving CLI/profile and distribution source pins.

This is a source-drift guard, not GPU qualification or a historical RC bundle
attestation. Pins are literals reviewed with their source changes, never
discovered or silently refreshed while validating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path

CONTRACT_PATH = Path("ci/serving-source-contract.json")
SOURCE_PATHS = {
    "crates/riley-server/src/main.rs",
    "crates/riley-server/src/inference_profile.rs",
    "deploy/docker/Dockerfile",
    "deploy/docker/entrypoint.sh",
}
MAX_SOURCE_BYTES = 4 * 1024 * 1024


class SourceContractError(ValueError):
    """A reviewed source contract is missing, malformed or stale."""


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SourceContractError(f"duplicate contract key: {key}")
        result[key] = value
    return result


def _read_regular(root: Path, relative: Path) -> bytes:
    path = root
    for index, part in enumerate(relative.parts):
        path /= part
        info = path.lstat()
        expected = stat.S_ISREG if index == len(relative.parts) - 1 else stat.S_ISDIR
        if not expected(info.st_mode):
            raise SourceContractError(f"contract path must be regular and not a link: {relative}")
    if info.st_size > MAX_SOURCE_BYTES:
        raise SourceContractError(f"contract source exceeds size bound: {relative}")
    with path.open("rb") as handle:
        raw = handle.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise SourceContractError(f"contract source exceeds size bound: {relative}")
    return raw


def validate(root: Path) -> dict[str, object]:
    contract = json.loads(_read_regular(root, CONTRACT_PATH), object_pairs_hook=_object)
    if not isinstance(contract, dict) or set(contract) != {
        "schema_version", "reviewed_base_revision", "distribution", "sources"
    }:
        raise SourceContractError("contract fields differ from serving-source-contract.v1")
    if contract["schema_version"] != "riley.serving-source-contract.v1":
        raise SourceContractError("unsupported serving source contract version")
    if not isinstance(contract["reviewed_base_revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", contract["reviewed_base_revision"]
    ):
        raise SourceContractError("reviewed base revision must be a full Git SHA")
    if contract["distribution"] != "docker-cu124-sm80-sm86-sm89":
        raise SourceContractError("distribution identity changed without contract review")
    sources = contract["sources"]
    if not isinstance(sources, dict) or set(sources) != SOURCE_PATHS:
        raise SourceContractError("serving source inventory differs from the reviewed paths")
    for relative, expected in sorted(sources.items()):
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise SourceContractError(f"invalid source SHA-256: {relative}")
        actual = hashlib.sha256(_read_regular(root, Path(relative))).hexdigest()
        if actual != expected:
            raise SourceContractError(
                f"serving source drift: {relative}: {actual} != reviewed {expected}; "
                "review the behavior and update the contract with the source"
            )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        validate(args.repo_root.resolve())
    except (OSError, ValueError) as error:
        print(f"serving source contract failed: {error}", file=sys.stderr)
        return 1
    print("serving source contract passed (source pins only; no GPU qualification)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
