#!/usr/bin/env python3
"""Build the deterministic, fail-closed riley release bundle."""

from __future__ import annotations

import argparse
import gzip
import os
import stat
import tarfile
import tempfile
from pathlib import Path

from release_common import (
    ReleaseContractError,
    canonical_json_bytes,
    native_manifest_bytes,
    read_workspace_version,
    release_manifest,
    release_root,
    sha256_bytes,
    validate_binary,
    validate_license,
    validate_license_metadata,
    validate_server_defaults_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-date-epoch", required=True, type=int)
    return parser.parse_args()


def _regular_file(path: Path, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReleaseContractError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ReleaseContractError(f"{label} must be a regular file, not a link or device: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ReleaseContractError(f"cannot read {label} {path}: {error}") from error


def _write(path: Path, contents: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    path.chmod(mode)


def _tar_info(path: Path, archive_name: str, source_date_epoch: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(archive_name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = source_date_epoch
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    else:
        info.type = tarfile.REGTYPE
        info.mode = 0o755 if archive_name.endswith("/bin/riley") else 0o644
        info.size = path.stat().st_size
    return info


def _write_archive(staging_root: Path, output: Path, source_date_epoch: int) -> None:
    names = sorted(
        [staging_root.name]
        + [
            path.relative_to(staging_root.parent).as_posix()
            for path in staging_root.rglob("*")
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary_output.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                compresslevel=9,
                mtime=source_date_epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as archive:
                    for name in names:
                        path = staging_root.parent / name
                        info = _tar_info(path, name, source_date_epoch)
                        if path.is_dir():
                            archive.addfile(info)
                        else:
                            with path.open("rb") as source:
                                archive.addfile(info, source)
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)


def build_bundle(
    *,
    binary_path: Path,
    output: Path,
    repository_root: Path,
    source_revision: str,
    source_date_epoch: int,
) -> Path:
    repository_root = repository_root.resolve()
    version = read_workspace_version(repository_root)
    root_name = release_root(version)

    license_contents = _regular_file(repository_root / "LICENSE", "root LICENSE")
    validate_license(license_contents)
    validate_license_metadata(repository_root)
    validate_server_defaults_source(repository_root)
    binary_contents = _regular_file(binary_path, "release CLI binary")
    if not os.access(binary_path, os.X_OK):
        raise ReleaseContractError("release CLI binary is not executable")
    dependencies = validate_binary(binary_contents)

    manifest = release_manifest(version, source_revision, source_date_epoch)
    with tempfile.TemporaryDirectory(prefix="riley-release-") as temporary_directory:
        staging_root = Path(temporary_directory) / root_name
        _write(staging_root / "bin/riley", binary_contents, 0o755)
        _write(
            staging_root / "manifest/native-dependencies.txt",
            native_manifest_bytes(dependencies),
            0o644,
        )
        _write(
            staging_root / "manifest/release.json",
            canonical_json_bytes(manifest),
            0o644,
        )
        _write(staging_root / "LICENSE", license_contents, 0o644)
        notice_path = repository_root / "NOTICE"
        if notice_path.exists():
            _write(staging_root / "NOTICE", _regular_file(notice_path, "root NOTICE"), 0o644)

        checksummed_files = sorted(
            path for path in staging_root.rglob("*") if path.is_file()
        )
        checksum_lines = [
            f"{sha256_bytes(path.read_bytes())}  {path.relative_to(staging_root).as_posix()}"
            for path in checksummed_files
        ]
        _write(
            staging_root / "SHA256SUMS",
            ("\n".join(checksum_lines) + "\n").encode("ascii"),
            0o644,
        )
        _write_archive(staging_root, output, source_date_epoch)

    # A producer never emits an archive that its consumer would reject.
    from verify_release_bundle import verify_bundle

    verify_bundle(output)
    return output


def main() -> int:
    args = parse_args()
    try:
        output = build_bundle(
            binary_path=args.binary,
            output=args.output,
            repository_root=args.repository_root,
            source_revision=args.source_revision,
            source_date_epoch=args.source_date_epoch,
        )
    except (OSError, ReleaseContractError, tarfile.TarError) as error:
        print(f"release bundle build failed: {error}", file=os.sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
