#!/usr/bin/env python3
"""Package one clean-container release build into closed raw evidence."""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

from check_reproducible_build import (
    GATE_ID,
    MAX_BINARY_SIZE,
    MAX_METADATA_SIZE,
    RELATIVE_FILES,
    SCHEMA_VERSION,
    _bundle_details,
    _regular_path,
    _sha256_file,
    _validate_builder_image_inspect,
    _validate_completion_receipt,
    _validate_container_inspect,
    _validate_logs,
    _validate_source_archive,
    expected_commands,
    expected_environment,
    invocation_bytes,
    validate_single_evidence,
)
from release_common import (
    ReleaseContractError,
    canonical_json_bytes,
    parse_native_manifest,
    validate_binary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", choices=("A", "B"), required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--build-image-id", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--profile-binary", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--toolchain-log", type=Path, required=True)
    parser.add_argument("--builder-image-inspect", type=Path, required=True)
    parser.add_argument("--container-inspect", type=Path, required=True)
    parser.add_argument("--post-container-inspect", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--preflight-log", type=Path, required=True)
    parser.add_argument("--cargo-build-log", type=Path, required=True)
    parser.add_argument("--profile-build-log", type=Path, required=True)
    parser.add_argument("--bundle-build-log", type=Path, required=True)
    parser.add_argument("--bundle-verify-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _copy_regular(source: Path, destination: Path, label: str, mode: int = 0o644) -> None:
    _regular_path(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
    destination.chmod(mode)


def _write(path: Path, contents: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(contents)
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
        info.mode = (
            0o755
            if archive_name.endswith(("/bin/riley", "/bin/riley-profile"))
            else 0o644
        )
        info.size = path.stat().st_size
    return info


def _write_raw_tar(staging_root: Path, output: Path, source_date_epoch: int) -> None:
    names = sorted(
        [staging_root.name]
        + [path.relative_to(staging_root.parent).as_posix() for path in staging_root.rglob("*")]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise ReleaseContractError(f"refusing to overwrite reproducibility evidence: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as raw:
            with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name in names:
                    path = staging_root.parent / name
                    info = _tar_info(path, name, source_date_epoch)
                    if path.is_dir():
                        archive.addfile(info)
                    else:
                        with path.open("rb") as source:
                            archive.addfile(info, source)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ReleaseContractError(
                f"refusing to overwrite reproducibility evidence: {output}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def package_evidence(
    *,
    build_id: str,
    source_archive: Path,
    source_revision: str,
    source_date_epoch: int,
    build_image_id: str,
    binary: Path,
    profile_binary: Path,
    bundle: Path,
    native_manifest: Path,
    toolchain_log: Path,
    builder_image_inspect: Path,
    container_inspect: Path,
    post_container_inspect: Path,
    completion_receipt: Path,
    preflight_log: Path,
    cargo_build_log: Path,
    profile_build_log: Path,
    bundle_build_log: Path,
    bundle_verify_log: Path,
    output: Path,
) -> None:
    if build_id not in {"A", "B"}:
        raise ReleaseContractError("build_id must be A or B")
    _validate_source_archive(source_archive, source_revision, source_date_epoch)
    for path, label in (
        (binary, "release binary"),
        (profile_binary, "release profile binary"),
        (bundle, "release bundle"),
        (native_manifest, "native dependency manifest"),
        (toolchain_log, "toolchain log"),
        (builder_image_inspect, "Docker builder image inspect evidence"),
        (container_inspect, "Docker container inspect evidence"),
        (post_container_inspect, "post-run Docker container inspect evidence"),
        (completion_receipt, "in-container completion receipt"),
        (preflight_log, "preflight log"),
        (cargo_build_log, "Cargo build log"),
        (profile_build_log, "profile Cargo build log"),
        (bundle_build_log, "bundle build log"),
        (bundle_verify_log, "bundle verification log"),
    ):
        _regular_path(path, label)
    if binary.stat().st_size > MAX_BINARY_SIZE:
        raise ReleaseContractError("release binary exceeds its evidence size bound")
    if profile_binary.stat().st_size > MAX_BINARY_SIZE:
        raise ReleaseContractError("release profile binary exceeds its evidence size bound")
    if native_manifest.stat().st_size > MAX_METADATA_SIZE:
        raise ReleaseContractError("native dependency manifest exceeds its evidence size bound")
    if container_inspect.stat().st_size > MAX_METADATA_SIZE:
        raise ReleaseContractError("Docker container inspect evidence exceeds its size bound")
    if post_container_inspect.stat().st_size > MAX_METADATA_SIZE:
        raise ReleaseContractError("post-run Docker inspect evidence exceeds its size bound")
    if completion_receipt.stat().st_size > MAX_METADATA_SIZE:
        raise ReleaseContractError("in-container completion receipt exceeds its size bound")
    if builder_image_inspect.stat().st_size > MAX_METADATA_SIZE:
        raise ReleaseContractError("Docker builder image inspect evidence exceeds its size bound")

    binary_bytes = binary.read_bytes()
    profile_binary_bytes = profile_binary.read_bytes()
    native_bytes = native_manifest.read_bytes()
    validate_binary(binary_bytes)
    validate_binary(profile_binary_bytes)
    parse_native_manifest(native_bytes)
    details = _bundle_details(bundle)
    if details.source_revision != source_revision or details.source_date_epoch != source_date_epoch:
        raise ReleaseContractError("release bundle provenance differs from the evidence source")
    if details.binary != binary_bytes:
        raise ReleaseContractError("release bundle binary differs from the raw release binary")
    if details.native_manifest != native_bytes:
        raise ReleaseContractError("release bundle native manifest differs from the raw manifest")
    builder_environment = _validate_builder_image_inspect(
        builder_image_inspect.read_bytes(),
        build_image_id,
    )
    pre_container = _validate_container_inspect(
        container_inspect.read_bytes(),
        build_id=build_id,
        build_image_id=build_image_id,
        source_revision=source_revision,
        source_archive_sha256=_sha256_file(source_archive),
        source_date_epoch=source_date_epoch,
        builder_environment=builder_environment,
    )
    post_container = _validate_container_inspect(
        post_container_inspect.read_bytes(),
        build_id=build_id,
        build_image_id=build_image_id,
        source_revision=source_revision,
        source_archive_sha256=_sha256_file(source_archive),
        source_date_epoch=source_date_epoch,
        builder_environment=builder_environment,
        expected_phase="exited",
    )
    if (
        pre_container.container_id != post_container.container_id
        or pre_container.workspace_volume != post_container.workspace_volume
        or pre_container.workspace_source != post_container.workspace_source
    ):
        raise ReleaseContractError(
            "pre-start and post-run Docker receipts describe different containers"
        )
    completion_files = {
        "bin/riley": binary,
        "bin/riley-profile": profile_binary,
        "bundle/riley.tar.gz": bundle,
        "manifest/native-dependencies.txt": native_manifest,
        "logs/toolchain.txt": toolchain_log,
        "logs/preflight.log": preflight_log,
        "logs/cargo-build.log": cargo_build_log,
        "logs/profile-build.log": profile_build_log,
        "logs/bundle-build.log": bundle_build_log,
        "logs/bundle-verify.log": bundle_verify_log,
    }
    _validate_completion_receipt(
        completion_receipt.read_bytes(),
        build_id=build_id,
        build_image_id=build_image_id,
        source_revision=source_revision,
        source_archive_sha256=_sha256_file(source_archive),
        source_date_epoch=source_date_epoch,
        container_id=pre_container.container_id,
        file_paths=completion_files,
    )

    root_name = f"riley-repro-build-{build_id.lower()}"
    with tempfile.TemporaryDirectory(prefix="riley-repro-package-") as temporary:
        root = Path(temporary) / root_name
        root.mkdir()
        _copy_regular(source_archive, root / "source.tar", "source archive")
        _copy_regular(binary, root / "bin/riley", "release binary", 0o755)
        _copy_regular(
            profile_binary,
            root / "bin/riley-profile",
            "release profile binary",
            0o755,
        )
        _copy_regular(bundle, root / "bundle/riley.tar.gz", "release bundle")
        _copy_regular(
            native_manifest,
            root / "manifest/native-dependencies.txt",
            "native dependency manifest",
        )
        _copy_regular(toolchain_log, root / "logs/toolchain.txt", "toolchain log")
        _copy_regular(
            builder_image_inspect,
            root / "logs/builder-image-inspect.json",
            "Docker builder image inspect evidence",
        )
        _copy_regular(
            container_inspect,
            root / "logs/container-inspect.json",
            "Docker container inspect evidence",
        )
        _copy_regular(
            post_container_inspect,
            root / "logs/container-inspect-post.json",
            "post-run Docker container inspect evidence",
        )
        _copy_regular(
            completion_receipt,
            root / "logs/build-completion.json",
            "in-container completion receipt",
        )
        _copy_regular(preflight_log, root / "logs/preflight.log", "preflight log")
        _copy_regular(cargo_build_log, root / "logs/cargo-build.log", "Cargo build log")
        _copy_regular(
            profile_build_log,
            root / "logs/profile-build.log",
            "profile Cargo build log",
        )
        _copy_regular(bundle_build_log, root / "logs/bundle-build.log", "bundle build log")
        _copy_regular(
            bundle_verify_log,
            root / "logs/bundle-verify.log",
            "bundle verification log",
        )
        _write(
            root / "logs/container-invocation.txt",
            invocation_bytes(build_id, build_image_id),
        )

        build_manifest = {
            "schema_version": SCHEMA_VERSION,
            "gate_id": GATE_ID,
            "status": "passed",
            "build_id": build_id,
            "source": {
                "archive_sha256": _sha256_file(source_archive),
                "revision": source_revision,
                "source_date_epoch": source_date_epoch,
                "workspace": "fresh-git-archive",
            },
            "environment": expected_environment(build_image_id),
            "commands": expected_commands(source_revision, source_date_epoch),
            "artifacts": {
                "binary_sha256": _sha256_file(binary),
                "binary_size": binary.stat().st_size,
                "profile_binary_sha256": _sha256_file(profile_binary),
                "profile_binary_size": profile_binary.stat().st_size,
                "bundle_sha256": _sha256_file(bundle),
                "bundle_size": bundle.stat().st_size,
                "native_manifest_sha256": _sha256_file(native_manifest),
                "native_manifest_size": native_manifest.stat().st_size,
            },
        }
        _write(root / "manifest/build.json", canonical_json_bytes(build_manifest))

        checksum_paths = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
        expected_without_checksum = sorted(RELATIVE_FILES - {"SHA256SUMS"})
        if checksum_paths != expected_without_checksum:
            raise ReleaseContractError("producer staging inventory differs from the closed contract")
        checksums = "".join(
            f"{_sha256_file(root / relative)}  {relative}\n" for relative in checksum_paths
        ).encode("ascii")
        _write(root / "SHA256SUMS", checksums)

        evidence = SimpleNamespace(
            build_id=build_id,
            files={
                path.relative_to(root).as_posix(): path
                for path in root.rglob("*")
                if path.is_file()
            },
        )
        _validate_logs(evidence, build_image_id)
        _write_raw_tar(root, output, source_date_epoch)

    validate_single_evidence(
        output,
        expected_build_id=build_id,
        source_archive=source_archive,
        source_revision=source_revision,
        source_date_epoch=source_date_epoch,
        build_image_id=build_image_id,
    )


def main() -> int:
    args = parse_args()
    try:
        package_evidence(
            build_id=args.build_id,
            source_archive=args.source_archive,
            source_revision=args.source_revision,
            source_date_epoch=args.source_date_epoch,
            build_image_id=args.build_image_id,
            binary=args.binary,
            profile_binary=args.profile_binary,
            bundle=args.bundle,
            native_manifest=args.native_manifest,
            toolchain_log=args.toolchain_log,
            builder_image_inspect=args.builder_image_inspect,
            container_inspect=args.container_inspect,
            post_container_inspect=args.post_container_inspect,
            completion_receipt=args.completion_receipt,
            preflight_log=args.preflight_log,
            cargo_build_log=args.cargo_build_log,
            profile_build_log=args.profile_build_log,
            bundle_build_log=args.bundle_build_log,
            bundle_verify_log=args.bundle_verify_log,
            output=args.output,
        )
    except (OSError, ReleaseContractError, tarfile.TarError) as error:
        print(f"reproducibility evidence packaging failed: {error}", file=os.sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
