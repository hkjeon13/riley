#!/usr/bin/env python3
"""Create one canonical, external, immutable RC3 candidate freeze document.

The input is a reviewed, fully filled candidate object.  This writer does not
discover hashes, build code, or run GPU work: it validates the closed C02
shape, canonicalizes it, and publishes it exactly once.  The output must stay
outside the candidate checkout so recording its own revision cannot create a
self-referential Git candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import check_rc3_qualification as qualification


@dataclass(frozen=True)
class FreezeWriteResult:
    candidate_id: str
    sha256: str
    path: Path


def _outside_repository(output: Path, repository_root: Path) -> None:
    try:
        root = repository_root.resolve(strict=True)
        candidate = output.resolve(strict=False)
    except OSError as error:
        raise qualification.QualificationError(
            f"cannot resolve repository/output path safely: {error}"
        ) from error
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    raise qualification.QualificationError(
        "freeze output must be outside the candidate checkout; a tracked freeze "
        "would be self-referential"
    )


def write_freeze(
    input_path: Path,
    output_path: Path,
    *,
    repository_root: Path | None = None,
) -> FreezeWriteResult:
    raw = qualification._read_regular_path(input_path, "freeze input")
    document = qualification._parse_document(raw, "freeze input")
    frozen = qualification._validate_freeze(document)
    if repository_root is not None:
        _outside_repository(output_path, repository_root)
        qualification._validate_repository(repository_root, frozen)
    canonical = qualification.canonical_json_bytes(document)
    qualification._write_create_only_bytes(output_path, canonical)
    return FreezeWriteResult(
        candidate_id=frozen.candidate_id,
        sha256=hashlib.sha256(canonical).hexdigest(),
        path=output_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="reviewed filled freeze object")
    parser.add_argument("--output", required=True, type=Path, help="new create-only external freeze path")
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="optional clean candidate checkout to validate and exclude from output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = write_freeze(
            args.input,
            args.output,
            repository_root=args.repository_root,
        )
    except (OSError, qualification.QualificationError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "candidate_id": result.candidate_id,
                "path": str(result.path),
                "sha256": result.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
