#!/usr/bin/env python3
"""Stdin-only checker for an unapproved Gate E guardian review input.

The checker consumes one bounded canonical JSON byte stream from standard
input.  It does not accept a path, install/approve/execute option, or any
operational argument, and it emits no success artifact.  A zero exit status
means only that the input has the narrow non-authoritative review shape.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence


def _load_sibling_contract() -> object:
    """Load only the parser next to this checker after resolving symlinks."""

    contract_path = Path(__file__).resolve().with_name(
        "gate_e_native_guardian_review_contract_v1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_gate_e_native_guardian_review_contract_v1",
        contract_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the sibling guardian review parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_sibling_contract()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check one unapproved native-guardian review input from stdin only.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--review-input-contract-check",
        action="store_true",
        help="require the non-authoritative canonical-byte review-input check",
    )
    return parser


def _isolated_python_runtime() -> bool:
    flags = sys.flags
    return bool(
        flags.isolated
        and flags.no_site
        and flags.ignore_environment
        and flags.dont_write_bytecode
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the opt-in stdin-only contract check without producing an artifact."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.review_input_contract_check:
        parser.error("--review-input-contract-check is required")
    if not _isolated_python_runtime():
        parser.error("requires Python -I -S -E -B")
    raw = sys.stdin.buffer.read(contract.MAX_DOCUMENT_BYTES + 1)
    try:
        contract.parse_native_guardian_review_v1(raw)
    except contract.NativeGuardianReviewContractError as error:
        reason = getattr(error, "reason_code", "invalid-review-input")
        parser.error(f"review input rejected: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
