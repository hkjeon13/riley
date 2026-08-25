#!/usr/bin/env python3
"""Package a checked PR-16 soak run as deterministic raw release evidence."""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path
from typing import Sequence

from check_reliability_soak import InputError, package_raw_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        package_raw_evidence(args.manifest, args.run_directory, args.output)
    except (InputError, FileExistsError, OSError, tarfile.TarError) as error:
        print(f"cannot package reliability soak evidence: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
