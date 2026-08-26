#!/usr/bin/env python3
"""CPU-only tests for the PR16 independent release reproducibility gate."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent))

from build_release_bundle import build_bundle  # noqa: E402
from check_reproducible_build import (  # noqa: E402
    BUILDER_PATH,
    CONTAINER_COMMAND,
    GATE_ID,
    PROXY_ENVIRONMENT,
    RUSTUP_TOOLCHAIN,
    _validate_source_archive,
    check_reproducible_build,
)
from package_reproducible_build_evidence import package_evidence  # noqa: E402
from release_common import (  # noqa: E402
    MIT_LICENSE_BYTES,
    ReleaseContractError,
    canonical_json_bytes,
)
from test_release import (  # noqa: E402
    fixture_elf,
    install_reviewed_server_defaults_source,
)
from write_reproducible_build_completion import (  # noqa: E402
    CompletionReceiptError,
    write_completion_receipt,
)

REVISION = "a" * 40
EPOCH = 1_700_000_000
IMAGE_ID = "sha256:" + "b" * 64
SOURCE_SHA_PLACEHOLDER = ""
TOOLCHAIN_LOG = (
    b"rustc_version=rustc 1.85.0 (4d91de4e4 2025-02-17)\n"
    b"cargo_version=cargo 1.85.0 (d73d2caf9 2024-12-31)\n"
    b"nvcc_version=Cuda compilation tools, release 12.8, V12.8.93\n"
)
CARGO_LOG = (
    b"   Compiling rustinfer-server v0.1.0 (/workspace/crates/rustinfer-server)\n"
    b"    Finished `release` profile [optimized] target(s) in 1.23s\n"
)
PROFILE_CARGO_LOG = (
    b"   Compiling rustinfer-server v0.1.0 (/workspace/crates/rustinfer-server)\n"
    b"    Finished `release` profile [optimized] target(s) in 0.42s\n"
)


def builder_image_inspect() -> list[dict[str, object]]:
    return [
        {
            "Id": IMAGE_ID,
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {
                "WorkingDir": "/workspace",
                "Volumes": None,
                "Env": [
                    f"PATH={BUILDER_PATH}",
                    "LD_LIBRARY_PATH=/usr/local/cuda/lib64",
                    "LIBRARY_PATH=/usr/local/cuda/lib64/stubs",
                    "DEBIAN_FRONTEND=noninteractive",
                    "CARGO_HOME=/usr/local/cargo",
                    "RUSTUP_HOME=/usr/local/rustup",
                    f"RUSTUP_TOOLCHAIN={RUSTUP_TOOLCHAIN}",
                    "CUDA_HOME=/usr/local/cuda",
                    "CUDAToolkit_ROOT=/usr/local/cuda",
                    "RUSTINFER_CUDA_ARCHITECTURES=89",
                    "CARGO_INCREMENTAL=0",
                    "CARGO_NET_OFFLINE=true",
                    "CUDA_VERSION=12.8.1",
                ],
            },
        }
    ]


def container_inspect(build_id: str) -> list[dict[str, object]]:
    lower = build_id.lower()
    baseline_environment = list(builder_image_inspect()[0]["Config"]["Env"])
    injected_environment = [
        f"RUSTINFER_REPRO_BUILD_ID={build_id}",
        f"RUSTINFER_SOURCE_REVISION={REVISION}",
        f"RUSTINFER_SOURCE_ARCHIVE_SHA256={SOURCE_SHA_PLACEHOLDER}",
        f"RUSTINFER_BUILD_IMAGE_ID={IMAGE_ID}",
        f"SOURCE_DATE_EPOCH={EPOCH}",
        *(f"{key}={value}" for key, value in PROXY_ENVIRONMENT.items()),
    ]
    return [
        {
            "Id": ("1" if build_id == "A" else "2") * 64,
            "Image": IMAGE_ID,
            "Platform": "linux",
            "State": {
                "Status": "created",
                "Running": False,
                "Paused": False,
                "Restarting": False,
                "OOMKilled": False,
                "Dead": False,
                "Pid": 0,
                "ExitCode": 0,
                "Error": "",
                "StartedAt": "0001-01-01T00:00:00Z",
                "FinishedAt": "0001-01-01T00:00:00Z",
            },
            "RestartCount": 0,
            "Path": "/bin/bash",
            "Args": ["-ceu", CONTAINER_COMMAND],
            "Config": {
                "Image": IMAGE_ID,
                "Entrypoint": ["/bin/bash"],
                "User": "0:0",
                "WorkingDir": "/workspace",
                "Healthcheck": {"Test": ["NONE"]},
                "Cmd": ["-ceu", CONTAINER_COMMAND],
                "Env": baseline_environment + injected_environment,
            },
            "HostConfig": {
                "NetworkMode": "none",
                "Runtime": "runc",
                "Privileged": False,
                "ReadonlyRootfs": False,
                "CapDrop": ["ALL"],
                "CapAdd": None,
                "SecurityOpt": ["no-new-privileges"],
                "PidsLimit": 4096,
                "Devices": [],
                "DeviceRequests": [],
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                "Binds": None,
                "VolumeDriver": "",
                "PidMode": "",
                "IpcMode": "private",
                "UTSMode": "",
                "UsernsMode": "",
                "CgroupnsMode": "private",
                "VolumesFrom": None,
                "DeviceCgroupRules": None,
                "GroupAdd": None,
                "Links": None,
                "Sysctls": None,
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": "/host/source.tar",
                        "Target": "/input/source.tar",
                        "ReadOnly": True,
                    },
                    {
                        "Type": "bind",
                        "Source": f"/host/container-inspect-{lower}.json",
                        "Target": "/input/container-inspect.json",
                        "ReadOnly": True,
                    },
                    {
                        "Type": "bind",
                        "Source": "/host/builder-image-inspect.json",
                        "Target": "/input/builder-image-inspect.json",
                        "ReadOnly": True,
                    },
                    {
                        "Type": "bind",
                        "Source": f"/host/run-{lower}",
                        "Target": "/evidence",
                        "ReadOnly": False,
                    },
                    {
                        "Type": "volume",
                        "Source": "",
                        "Target": "/workspace",
                        "ReadOnly": False,
                        "VolumeOptions": {"NoCopy": True, "DriverConfig": {}},
                    },
                ],
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": "/host/source.tar",
                    "Destination": "/input/source.tar",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": f"/host/container-inspect-{lower}.json",
                    "Destination": "/input/container-inspect.json",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": "/host/builder-image-inspect.json",
                    "Destination": "/input/builder-image-inspect.json",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": f"/host/run-{lower}",
                    "Destination": "/evidence",
                    "RW": True,
                },
                {
                    "Type": "volume",
                    "Name": f"anonymous-workspace-{lower}",
                    "Driver": "local",
                    "Source": f"/var/lib/docker/volumes/anonymous-workspace-{lower}/_data",
                    "Destination": "/workspace",
                    "RW": True,
                },
            ],
        }
    ]


def post_container_inspect(build_id: str) -> list[dict[str, object]]:
    document = container_inspect(build_id)
    suffix = "1" if build_id == "A" else "3"
    document[0]["State"].update(
        {
            "Status": "exited",
            "ExitCode": 0,
            "StartedAt": f"2026-08-26T00:00:00.00000000{suffix}Z",
            "FinishedAt": f"2026-08-26T00:00:00.00000000{int(suffix) + 1}Z",
        }
    )
    return document


def _source_member(name: str, contents: bytes | None) -> tuple[tarfile.TarInfo, bytes | None]:
    member = tarfile.TarInfo(name)
    member.uid = 0
    member.gid = 0
    member.uname = "root"
    member.gname = "root"
    member.mtime = EPOCH
    if contents is None:
        member.type = tarfile.DIRTYPE
        member.mode = 0o775
        member.size = 0
    else:
        member.type = tarfile.REGTYPE
        member.mode = 0o664
        member.size = len(contents)
    return member, contents


def write_source_archive(path: Path, revision: str = REVISION) -> None:
    entries = [
        _source_member("Cargo.lock", b"# fixture lock\n"),
        _source_member("Cargo.toml", b"[workspace]\nmembers=[]\n"),
        _source_member("benchmarks/", None),
        _source_member("benchmarks/lanes/", None),
        _source_member("benchmarks/lanes/vllm.json", b"{}\n"),
        _source_member("benchmarks/lanes/vllm/", None),
        _source_member("benchmarks/lanes/vllm/README.md", b"fixture\n"),
        _source_member("ci/", None),
        _source_member("ci/release/", None),
        _source_member("ci/release/Dockerfile", b"FROM scratch\n"),
    ]
    with tarfile.open(
        path,
        mode="w",
        format=tarfile.PAX_FORMAT,
        pax_headers={"comment": revision},
    ) as archive:
        for member, contents in entries:
            archive.addfile(member, io.BytesIO(contents) if contents is not None else None)


def rewrite_evidence(
    source: Path,
    destination: Path,
    mutate: Callable[[dict[str, tuple[tarfile.TarInfo, bytes | None]]], None],
) -> None:
    entries: dict[str, tuple[tarfile.TarInfo, bytes | None]] = {}
    with tarfile.open(source, mode="r:") as archive:
        for member in archive:
            contents = archive.extractfile(member).read() if member.isreg() else None
            entries[member.name] = (copy.copy(member), contents)
    mutate(entries)
    with tarfile.open(destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(entries):
            member, contents = entries[name]
            archive.addfile(member, io.BytesIO(contents) if contents is not None else None)


def update_checksum(
    entries: dict[str, tuple[tarfile.TarInfo, bytes | None]],
    root: str,
    relative: str,
    contents: bytes,
) -> None:
    checksum_name = f"{root}/SHA256SUMS"
    checksum_member, checksum_contents = entries[checksum_name]
    assert checksum_contents is not None
    digest = hashlib.sha256(contents).hexdigest()
    lines = checksum_contents.decode("ascii").splitlines()
    prefix = f"  {relative}"
    rewritten = [f"{digest}{prefix}" if line.endswith(prefix) else line for line in lines]
    replacement = ("\n".join(rewritten) + "\n").encode("ascii")
    checksum_member.size = len(replacement)
    entries[checksum_name] = (checksum_member, replacement)


class ReproducibleBuildGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        (self.repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\n'
            'license = "MIT"\n',
            encoding="utf-8",
        )
        (self.repository / "LICENSE").write_bytes(MIT_LICENSE_BYTES)
        install_reviewed_server_defaults_source(self.repository)
        self.source_archive = self.root / "source.tar"
        write_source_archive(self.source_archive)
        global SOURCE_SHA_PLACEHOLDER
        SOURCE_SHA_PLACEHOLDER = hashlib.sha256(self.source_archive.read_bytes()).hexdigest()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        (self.logs / "toolchain.txt").write_bytes(TOOLCHAIN_LOG)
        (self.logs / "preflight.log").write_bytes(b"release preflight passed\n")
        (self.logs / "cargo-build.log").write_bytes(CARGO_LOG)
        (self.logs / "profile-build.log").write_bytes(PROFILE_CARGO_LOG)
        (self.logs / "bundle-build.log").write_bytes(
            b"/workspace/release/rustinfer.tar.gz\n"
        )
        (self.logs / "bundle-verify.log").write_bytes(
            b"verified /workspace/release/rustinfer.tar.gz\n"
        )
        (self.logs / "builder-image-inspect.json").write_bytes(
            canonical_json_bytes(builder_image_inspect())
        )
        for build_id in ("A", "B"):
            (self.logs / f"container-inspect-{build_id.lower()}.json").write_bytes(
                canonical_json_bytes(container_inspect(build_id))
            )
            (self.logs / f"container-inspect-{build_id.lower()}-post.json").write_bytes(
                canonical_json_bytes(post_container_inspect(build_id))
            )
        self.binary, self.bundle, self.native = self.make_release("first", fixture_elf())
        self.profile = self.root / "rustinfer-profile"
        self.profile.write_bytes(fixture_elf() + b"profile binary subject")
        self.profile.chmod(0o755)
        self.receipt_index = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_release(self, name: str, binary_contents: bytes) -> tuple[Path, Path, Path]:
        binary = self.root / f"{name}-rustinfer"
        binary.write_bytes(binary_contents)
        binary.chmod(0o755)
        bundle = self.root / f"{name}-rustinfer.tar.gz"
        build_bundle(
            binary_path=binary,
            output=bundle,
            repository_root=self.repository,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
        )
        native = self.root / f"{name}-native-dependencies.txt"
        with tarfile.open(bundle, mode="r:gz") as archive:
            member = next(
                item
                for item in archive
                if item.name.endswith("/manifest/native-dependencies.txt")
            )
            native.write_bytes(archive.extractfile(member).read())
        return binary, bundle, native

    def package(
        self,
        build_id: str,
        *,
        binary: Path | None = None,
        profile: Path | None = None,
        bundle: Path | None = None,
        native: Path | None = None,
        output_name: str | None = None,
    ) -> Path:
        output = self.root / (output_name or f"repro-build-{build_id.lower()}.tar")
        selected_binary = binary or self.binary
        selected_profile = profile or self.profile
        selected_bundle = bundle or self.bundle
        selected_native = native or self.native
        self.receipt_index += 1
        completion_receipt = self.logs / (
            f"build-completion-{build_id.lower()}-{self.receipt_index}.json"
        )
        write_completion_receipt(
            build_id=build_id,
            source_revision=REVISION,
            source_archive_sha256=SOURCE_SHA_PLACEHOLDER,
            source_date_epoch=EPOCH,
            build_image_id=IMAGE_ID,
            container_inspect=self.logs / f"container-inspect-{build_id.lower()}.json",
            outputs={
                "binary": selected_binary,
                "profile_binary": selected_profile,
                "bundle": selected_bundle,
                "native_manifest": selected_native,
                "toolchain_log": self.logs / "toolchain.txt",
                "preflight_log": self.logs / "preflight.log",
                "cargo_build_log": self.logs / "cargo-build.log",
                "profile_build_log": self.logs / "profile-build.log",
                "bundle_build_log": self.logs / "bundle-build.log",
                "bundle_verify_log": self.logs / "bundle-verify.log",
            },
            output=completion_receipt,
        )
        package_evidence(
            build_id=build_id,
            source_archive=self.source_archive,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
            build_image_id=IMAGE_ID,
            binary=selected_binary,
            profile_binary=selected_profile,
            bundle=selected_bundle,
            native_manifest=selected_native,
            toolchain_log=self.logs / "toolchain.txt",
            builder_image_inspect=self.logs / "builder-image-inspect.json",
            container_inspect=self.logs / f"container-inspect-{build_id.lower()}.json",
            post_container_inspect=(
                self.logs / f"container-inspect-{build_id.lower()}-post.json"
            ),
            completion_receipt=completion_receipt,
            preflight_log=self.logs / "preflight.log",
            cargo_build_log=self.logs / "cargo-build.log",
            profile_build_log=self.logs / "profile-build.log",
            bundle_build_log=self.logs / "bundle-build.log",
            bundle_verify_log=self.logs / "bundle-verify.log",
            output=output,
        )
        return output

    def check(self, evidence_a: Path, evidence_b: Path) -> dict[str, object]:
        return check_reproducible_build(
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            source_archive=self.source_archive,
            expected_source_archive_sha256=SOURCE_SHA_PLACEHOLDER,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
            build_image_id=IMAGE_ID,
            final_binary=self.binary,
            final_profile_binary=self.profile,
            final_bundle=self.bundle,
            final_native_manifest=self.native,
        )

    def test_two_identical_independent_builds_pass(self) -> None:
        report = self.check(self.package("A"), self.package("B"))
        self.assertEqual(report["gate_id"], GATE_ID)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["build"]["independent_clean_containers"], 2)
        self.assertTrue(report["comparisons"]["bundle_a_b_final_byte_exact"])
        self.assertTrue(
            report["comparisons"]["profile_binary_a_b_final_byte_exact"]
        )
        self.assertEqual(
            report["artifacts"]["profile_binary_sha256"],
            hashlib.sha256(self.profile.read_bytes()).hexdigest(),
        )
        self.assertLess(
            report["evidence"]["a_started_at"],
            report["evidence"]["a_finished_at"],
        )
        self.assertEqual(
            canonical_json_bytes(report),
            canonical_json_bytes(json.loads(canonical_json_bytes(report))),
        )

    def test_git_archive_file_then_same_prefix_directory_order_is_accepted(self) -> None:
        _validate_source_archive(self.source_archive, REVISION, EPOCH)

    def test_source_archive_rejects_trailing_tar_record(self) -> None:
        changed = self.root / "source-with-trailing-record.tar"
        changed.write_bytes(self.source_archive.read_bytes() + b"\0" * 10_240)
        with self.assertRaisesRegex(ReleaseContractError, "trailing records"):
            _validate_source_archive(changed, REVISION, EPOCH)

    def test_structurally_valid_b_build_with_different_bytes_fails(self) -> None:
        second_binary, second_bundle, second_native = self.make_release(
            "second", fixture_elf() + b"different application bytes"
        )
        evidence_a = self.package("A")
        evidence_b = self.package(
            "B",
            binary=second_binary,
            bundle=second_bundle,
            native=second_native,
        )
        with self.assertRaisesRegex(ReleaseContractError, "byte-exact"):
            self.check(evidence_a, evidence_b)

    def test_profile_binary_must_match_a_b_and_final(self) -> None:
        second_profile = self.root / "second-rustinfer-profile"
        second_profile.write_bytes(fixture_elf() + b"different profile bytes")
        second_profile.chmod(0o755)
        evidence_a = self.package("A")
        evidence_b = self.package("B", profile=second_profile)
        with self.assertRaisesRegex(
            ReleaseContractError,
            "release profile binary A/B/final is not byte-exact",
        ):
            self.check(evidence_a, evidence_b)

    def test_post_run_receipt_must_prove_successful_exit(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "failed-post-run-receipt.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            relative = "logs/container-inspect-post.json"
            name = f"{root}/{relative}"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document[0]["State"]["ExitCode"] = 1
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, relative, replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(
            ReleaseContractError,
            "exited receipt has invalid State.ExitCode",
        ):
            self.check(changed, self.package("B"))

    def test_completion_receipt_binds_container_identity(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "forged-completion-receipt.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            relative = "logs/build-completion.json"
            name = f"{root}/{relative}"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document["container_id"] = "f" * 64
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, relative, replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(
            ReleaseContractError,
            "completion receipt differs for container_id",
        ):
            self.check(changed, self.package("B"))

    def test_completion_receipt_binds_output_bytes(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "forged-completion-output.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            relative = "logs/build-completion.json"
            name = f"{root}/{relative}"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document["outputs"]["binary"]["sha256"] = "e" * 64
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, relative, replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(
            ReleaseContractError,
            "completion output bytes differ for binary",
        ):
            self.check(changed, self.package("B"))

    def test_changed_build_command_fails_even_with_valid_checksum(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "changed-command.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/manifest/build.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document["commands"]["build"].remove("--offline")
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "manifest/build.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "locked/offline"):
            self.check(changed, self.package("B"))

    def test_changed_profile_build_command_fails_even_with_valid_checksum(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "changed-profile-command.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/manifest/build.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document["commands"]["profile_build"].remove("--offline")
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "manifest/build.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "locked/offline"):
            self.check(changed, self.package("B"))

    def test_docker_network_claim_is_checked_against_raw_inspect(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "changed-network-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document[0]["HostConfig"]["NetworkMode"] = "bridge"
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "network=none"):
            self.check(changed, self.package("B"))

    def test_container_environment_is_closed_against_rustflags(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "changed-environment-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document[0]["Config"]["Env"].append("RUSTFLAGS=-Cmetadata=unreviewed")
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "environment differs"):
            self.check(changed, self.package("B"))

    def test_builder_and_container_cannot_share_injected_environment(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "injected-builder-environment.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            for relative in (
                "logs/builder-image-inspect.json",
                "logs/container-inspect.json",
            ):
                name = f"{root}/{relative}"
                member, contents = entries[name]
                assert contents is not None
                document = json.loads(contents)
                document[0]["Config"]["Env"].extend(
                    (
                        "BASH_ENV=/opt/repro-hook.sh",
                        "LD_PRELOAD=/opt/repro-hook.so",
                        "PYTHONPATH=/opt/repro-python",
                    )
                )
                replacement = canonical_json_bytes(document)
                member.size = len(replacement)
                entries[name] = (member, replacement)
                update_checksum(entries, root, relative, replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "unreviewed environment"):
            self.check(changed, self.package("B"))

    def test_builder_path_cannot_precede_the_pinned_toolchains(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "injected-builder-path.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            for relative in (
                "logs/builder-image-inspect.json",
                "logs/container-inspect.json",
            ):
                name = f"{root}/{relative}"
                member, contents = entries[name]
                assert contents is not None
                document = json.loads(contents)
                environment = document[0]["Config"]["Env"]
                environment[environment.index(f"PATH={BUILDER_PATH}")] = (
                    f"PATH=/opt/repro-bin:{BUILDER_PATH}"
                )
                replacement = canonical_json_bytes(document)
                member.size = len(replacement)
                entries[name] = (member, replacement)
                update_checksum(entries, root, relative, replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "differs for PATH"):
            self.check(changed, self.package("B"))

    def test_consistent_but_wrong_rustup_toolchain_is_rejected(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "wrong-rustup-toolchain.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            for relative in (
                "logs/builder-image-inspect.json",
                "logs/container-inspect.json",
            ):
                name = f"{root}/{relative}"
                member, contents = entries[name]
                assert contents is not None
                document = json.loads(contents)
                environment = document[0]["Config"]["Env"]
                pinned = f"RUSTUP_TOOLCHAIN={RUSTUP_TOOLCHAIN}"
                environment[environment.index(pinned)] = (
                    "RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu"
                )
                replacement = canonical_json_bytes(document)
                member.size = len(replacement)
                entries[name] = (member, replacement)
                update_checksum(entries, root, relative, replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(
            ReleaseContractError,
            "differs for RUSTUP_TOOLCHAIN",
        ):
            self.check(changed, self.package("B"))

    def test_container_rejects_host_pid_namespace(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "host-pid-namespace-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document[0]["HostConfig"]["PidMode"] = "host"
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "host namespace: PidMode"):
            self.check(changed, self.package("B"))

    def test_container_security_options_are_closed(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "extra-security-option-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document[0]["HostConfig"]["SecurityOpt"].append("seccomp=unconfined")
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "no-new-privileges"):
            self.check(changed, self.package("B"))

    def test_container_command_must_be_exact(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "changed-command-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document[0]["Config"]["Cmd"][1] = "echo injected; " + CONTAINER_COMMAND
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "one-build driver"):
            self.check(changed, self.package("B"))

    def test_container_entrypoint_cannot_wrap_the_build_driver(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "changed-entrypoint-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document[0]["Config"]["Entrypoint"] = ["/unreviewed-wrapper"]
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "entrypoint"):
            self.check(changed, self.package("B"))

    def test_workspace_request_must_be_anonymous_and_nocopy(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "named-workspace-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            workspace = next(
                mount
                for mount in document[0]["HostConfig"]["Mounts"]
                if mount["Target"] == "/workspace"
            )
            workspace["Source"] = "preseeded-named-volume"
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "anonymous volume"):
            self.check(changed, self.package("B"))

    def test_workspace_rejects_custom_shared_volume_driver(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "custom-workspace-driver-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            host = document[0]["HostConfig"]
            host["VolumeDriver"] = "unreviewed-shared-plugin"
            workspace_request = next(
                mount for mount in host["Mounts"] if mount["Target"] == "/workspace"
            )
            workspace_request["VolumeOptions"]["DriverConfig"] = {
                "Name": "unreviewed-shared-plugin",
                "Options": {"share": "same"},
            }
            workspace = next(
                mount
                for mount in document[0]["Mounts"]
                if mount["Destination"] == "/workspace"
            )
            workspace["Driver"] = "unreviewed-shared-plugin"
            workspace["Source"] = "/plugin/shared/same-workspace"
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "local volume driver"):
            self.check(changed, self.package("B"))

    def test_container_inspect_receipt_must_be_pre_start(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "started-container-inspect.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            name = f"{root}/logs/container-inspect.json"
            member, contents = entries[name]
            assert contents is not None
            document = json.loads(contents)
            document[0]["State"]["Status"] = "exited"
            document[0]["State"]["StartedAt"] = "2026-08-26T00:00:00Z"
            replacement = canonical_json_bytes(document)
            member.size = len(replacement)
            entries[name] = (member, replacement)
            update_checksum(entries, root, "logs/container-inspect.json", replacement)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "created receipt"):
            self.check(changed, self.package("B"))

    def test_source_archive_requires_trusted_external_digest(self) -> None:
        with self.assertRaisesRegex(ReleaseContractError, "trusted expected SHA-256"):
            check_reproducible_build(
                evidence_a=self.package("A"),
                evidence_b=self.package("B"),
                source_archive=self.source_archive,
                expected_source_archive_sha256="c" * 64,
                source_revision=REVISION,
                source_date_epoch=EPOCH,
                build_image_id=IMAGE_ID,
                final_binary=self.binary,
                final_profile_binary=self.profile,
                final_bundle=self.bundle,
                final_native_manifest=self.native,
            )

    def test_a_b_must_use_distinct_anonymous_workspaces(self) -> None:
        for name, document in (
            ("container-inspect-b.json", container_inspect("B")),
            ("container-inspect-b-post.json", post_container_inspect("B")),
        ):
            workspace = next(
                mount
                for mount in document[0]["Mounts"]
                if mount["Destination"] == "/workspace"
            )
            workspace["Name"] = "anonymous-workspace-a"
            workspace["Source"] = "/var/lib/docker/volumes/anonymous-workspace-a/_data"
            (self.logs / name).write_bytes(canonical_json_bytes(document))
        with self.assertRaisesRegex(ReleaseContractError, "same Docker workspace volume"):
            self.check(self.package("A"), self.package("B"))

    def test_self_consistent_claims_cannot_hide_bundle_binary_mismatch(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "changed-binary.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = "rustinfer-repro-build-a"
            binary_name = f"{root}/bin/rustinfer"
            binary_member, binary_contents = entries[binary_name]
            assert binary_contents is not None
            replacement_binary = binary_contents + b"self-consistent fake claim"
            binary_member.size = len(replacement_binary)
            entries[binary_name] = (binary_member, replacement_binary)

            manifest_name = f"{root}/manifest/build.json"
            manifest_member, manifest_contents = entries[manifest_name]
            assert manifest_contents is not None
            document = json.loads(manifest_contents)
            document["artifacts"]["binary_sha256"] = hashlib.sha256(
                replacement_binary
            ).hexdigest()
            document["artifacts"]["binary_size"] = len(replacement_binary)
            replacement_manifest = canonical_json_bytes(document)
            manifest_member.size = len(replacement_manifest)
            entries[manifest_name] = (manifest_member, replacement_manifest)

            update_checksum(entries, root, "bin/rustinfer", replacement_binary)
            update_checksum(entries, root, "manifest/build.json", replacement_manifest)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(
            ReleaseContractError,
            "completion output bytes differ for binary",
        ):
            self.check(changed, self.package("B"))

    def test_symlink_member_is_rejected_before_artifact_checks(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "symlink.tar"

        def mutate(entries: dict[str, tuple[tarfile.TarInfo, bytes | None]]) -> None:
            name = "rustinfer-repro-build-a/source.tar"
            member, _ = entries[name]
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            member.size = 0
            entries[name] = (member, None)

        rewrite_evidence(evidence_a, changed, mutate)
        with self.assertRaisesRegex(ReleaseContractError, "link or special"):
            self.check(changed, self.package("B"))

    def test_evidence_rejects_trailing_tar_record(self) -> None:
        evidence_a = self.package("A")
        changed = self.root / "evidence-with-trailing-record.tar"
        changed.write_bytes(evidence_a.read_bytes() + b"\0" * 10_240)
        with self.assertRaisesRegex(ReleaseContractError, "trailing records"):
            self.check(changed, self.package("B"))

    def test_source_archive_must_embed_exact_git_revision(self) -> None:
        wrong_source = self.root / "wrong-source.tar"
        write_source_archive(wrong_source, revision="c" * 40)
        with self.assertRaisesRegex(ReleaseContractError, "exact git archive revision"):
            check_reproducible_build(
                evidence_a=self.package("A"),
                evidence_b=self.package("B"),
                source_archive=wrong_source,
                expected_source_archive_sha256=SOURCE_SHA_PLACEHOLDER,
                source_revision=REVISION,
                source_date_epoch=EPOCH,
                build_image_id=IMAGE_ID,
                final_binary=self.binary,
                final_profile_binary=self.profile,
                final_bundle=self.bundle,
                final_native_manifest=self.native,
            )

    def test_package_rejects_linked_binary_input(self) -> None:
        linked = self.root / "linked-binary"
        linked.symlink_to(self.binary)
        with self.assertRaisesRegex(CompletionReceiptError, "regular file"):
            self.package("A", binary=linked)


class ReproducibilityRunnerStaticTests(unittest.TestCase):
    def test_host_runner_uses_two_inspected_networkless_containers(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        contents = (repository / "ci/run_release_reproducibility.sh").read_text(
            encoding="utf-8"
        )
        for marker in (
            "docker create",
            "git -c tar.umask=0002 archive",
            "require_exact_clean_checkout",
            "git status --porcelain=v1 --untracked-files=all",
            "output directory must be outside the source checkout",
            "--runtime runc",
            "--cgroupns private",
            "--restart no",
            "--entrypoint /bin/bash",
            "--user 0:0",
            "--workdir /workspace",
            "--no-healthcheck",
            "--network none",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--env HTTPS_PROXY=",
            "type=volume,destination=/workspace,volume-nocopy",
            "destination=/input/builder-image-inspect.json,readonly",
            'docker inspect "${container_id}"',
            'docker start --attach "${container_id}"',
            'post_inspect_path="${output_dir}/container-inspect-${lower_id}-post.json"',
            '--completion-receipt "${run_dir}/logs/build-completion.json"',
            '--profile-binary "${run_dir}/artifacts/rustinfer-profile"',
            '--final-profile-binary "${output_dir}/final/rustinfer-profile"',
            'run_one A',
            'run_one B',
            "check_reproducible_build.py",
            'test "${source_archive_sha256}" = "${expected_source_archive_sha256}"',
            '--expected-source-archive-sha256 "${expected_source_archive_sha256}"',
        ):
            self.assertIn(marker, contents)
        self.assertNotIn(
            '--expected-source-archive-sha256 "${source_archive_sha256}"',
            contents,
        )
        self.assertNotIn("--gpus", contents)

    def test_container_build_command_is_locked_and_offline(self) -> None:
        contents = Path(__file__).with_name("run_reproducible_build_once.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "cargo build --locked --offline --release --features cuda,server",
            contents,
        )
        self.assertIn(
            "cargo build --locked --offline --release --features bench,cuda",
            contents,
        )
        self.assertIn("--bin rustinfer-profile", contents)
        self.assertIn("CARGO_NET_OFFLINE=true", contents)
        self.assertIn("target/release/rustinfer-profile", contents)
        self.assertIn("write_reproducible_build_completion.py", contents)
        self.assertLess(
            contents.index("install -m 0755 target/release/rustinfer-profile"),
            contents.index("write_reproducible_build_completion.py"),
        )
        for forbidden in (
            "llvm-strip",
            "objcopy --strip",
            "CARGO_PROFILE_RELEASE_STRIP",
            "-Cstrip=",
        ):
            self.assertNotIn(forbidden, contents)

    def test_builder_environment_is_digest_pinned_and_only_prefetches(self) -> None:
        contents = Path(__file__).with_name("ReproducibleBuild.Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "rust:1.85.0-bookworm@sha256:16a7f242108de02f10fe4a392991679bafa7694e59f5b40a54d5af1be9b40d03",
            contents,
        )
        self.assertIn(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7",
            contents,
        )
        self.assertIn("cargo fetch --locked", contents)
        self.assertIn("python3-tomli", contents)
        self.assertIn(
            f"ENV RUSTUP_TOOLCHAIN={RUSTUP_TOOLCHAIN}",
            contents,
        )
        self.assertNotIn("cargo build", contents)
        self.assertNotIn("COPY .", contents)


if __name__ == "__main__":
    unittest.main()
