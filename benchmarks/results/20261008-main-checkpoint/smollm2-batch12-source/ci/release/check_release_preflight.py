#!/usr/bin/env python3
"""Fail before a release build when immutable metadata or licensing is absent."""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

from release_common import (
    ReleaseContractError,
    read_workspace_version,
    release_manifest,
    validate_license,
    validate_license_metadata,
    validate_server_defaults_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-date-epoch", required=True, type=int)
    return parser.parse_args()


def check_preflight(repository_root: Path, source_revision: str, source_date_epoch: int) -> None:
    repository_root = repository_root.resolve()
    license_path = repository_root / "LICENSE"
    try:
        metadata = license_path.lstat()
    except OSError as error:
        raise ReleaseContractError(
            "root LICENSE is required before release packaging; the repository owner must choose it"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ReleaseContractError("root LICENSE must be a regular file, not a link or device")
    validate_license(license_path.read_bytes())
    validate_license_metadata(repository_root)
    validate_server_defaults_source(repository_root)
    version = read_workspace_version(repository_root)
    release_manifest(version, source_revision, source_date_epoch)


def main() -> int:
    args = parse_args()
    try:
        check_preflight(args.repository_root, args.source_revision, args.source_date_epoch)
    except (OSError, ReleaseContractError) as error:
        print(f"release preflight failed: {error}", file=os.sys.stderr)
        return 1
    print("release preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
