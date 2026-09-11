#!/usr/bin/env python3
"""Package five checked native profile runs as PR-16 performance evidence."""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path
from typing import Sequence

from check_release_performance import (
    ComparabilityError,
    InputError,
    package_release_performance_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--recorded-at-utc", required=True)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--profile-binary", required=True, type=Path)
    parser.add_argument("--release-binary", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--correctness-report", required=True, type=Path)
    parser.add_argument("--profile-image-id", required=True)
    parser.add_argument("--release-image-id", required=True)
    parser.add_argument("--run", required=True, nargs=5, type=Path)
    parser.add_argument("--runner-receipt-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = package_release_performance_evidence(
            args.baseline,
            args.output_directory,
            candidate_id=args.candidate_id,
            recorded_at_utc=args.recorded_at_utc,
            source_archive=args.source_archive,
            profile_binary=args.profile_binary,
            release_binary=args.release_binary,
            weights=args.weights,
            tokenizer=args.tokenizer,
            correctness_report=args.correctness_report,
            profile_image_id=args.profile_image_id,
            release_image_id=args.release_image_id,
            run_paths=args.run,
            runner_receipt_root=args.runner_receipt_root,
        )
    except (
        ComparabilityError,
        FileExistsError,
        InputError,
        OSError,
        tarfile.TarError,
    ) as error:
        print(
            f"cannot package release performance evidence: {error}",
            file=os.sys.stderr,
        )
        return 2
    return 0 if result["report"]["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
