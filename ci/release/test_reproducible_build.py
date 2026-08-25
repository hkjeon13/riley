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
    GATE_ID,
    _validate_source_archive,
    check_reproducible_build,
)
from package_reproducible_build_evidence import package_evidence  # noqa: E402
from release_common import ReleaseContractError, canonical_json_bytes  # noqa: E402
from test_release import fixture_elf  # noqa: E402

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


def container_inspect(build_id: str) -> list[dict[str, object]]:
    lower = build_id.lower()
    command = (
        'test -z "$(find /workspace -mindepth 1 -print -quit)"\n'
        "tar --extract --file /input/source.tar --directory /workspace\n"
        "cd /workspace\n"
        "exec /bin/bash ci/release/run_reproducible_build_once.sh\n"
    )
    return [
        {
            "Id": ("1" if build_id == "A" else "2") * 64,
            "Image": IMAGE_ID,
            "Platform": "linux",
            "Config": {
                "Image": IMAGE_ID,
                "WorkingDir": "/workspace",
                "Cmd": ["/bin/bash", "-ceu", command],
                "Env": [
                    f"RUSTINFER_REPRO_BUILD_ID={build_id}",
                    f"RUSTINFER_SOURCE_REVISION={REVISION}",
                    f"RUSTINFER_SOURCE_ARCHIVE_SHA256={SOURCE_SHA_PLACEHOLDER}",
                    f"RUSTINFER_BUILD_IMAGE_ID={IMAGE_ID}",
                    f"SOURCE_DATE_EPOCH={EPOCH}",
                ],
            },
            "HostConfig": {
                "NetworkMode": "none",
                "Runtime": "runc",
                "Privileged": False,
                "ReadonlyRootfs": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "PidsLimit": 4096,
                "Devices": [],
                "DeviceRequests": [],
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
                    "Source": f"/host/run-{lower}",
                    "Destination": "/evidence",
                    "RW": True,
                },
                {
                    "Type": "volume",
                    "Name": f"anonymous-workspace-{lower}",
                    "Source": f"/var/lib/docker/volumes/anonymous-workspace-{lower}/_data",
                    "Destination": "/workspace",
                    "RW": True,
                },
            ],
        }
    ]


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
            'license = "LicenseRef-Test-Fixture"\n',
            encoding="utf-8",
        )
        (self.repository / "LICENSE").write_text(
            "Owner-approved fixture license for reproducibility unit tests.\n"
            "Permission is limited to this isolated temporary test fixture.\n",
            encoding="utf-8",
        )
        self.source_archive = self.root / "source.tar"
        write_source_archive(self.source_archive)
        global SOURCE_SHA_PLACEHOLDER
        SOURCE_SHA_PLACEHOLDER = hashlib.sha256(self.source_archive.read_bytes()).hexdigest()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        (self.logs / "toolchain.txt").write_bytes(TOOLCHAIN_LOG)
        (self.logs / "preflight.log").write_bytes(b"release preflight passed\n")
        (self.logs / "cargo-build.log").write_bytes(CARGO_LOG)
        (self.logs / "bundle-build.log").write_bytes(
            b"/workspace/release/rustinfer.tar.gz\n"
        )
        (self.logs / "bundle-verify.log").write_bytes(
            b"verified /workspace/release/rustinfer.tar.gz\n"
        )
        for build_id in ("A", "B"):
            (self.logs / f"container-inspect-{build_id.lower()}.json").write_bytes(
                canonical_json_bytes(container_inspect(build_id))
            )
        self.binary, self.bundle, self.native = self.make_release("first", fixture_elf())

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
        bundle: Path | None = None,
        native: Path | None = None,
        output_name: str | None = None,
    ) -> Path:
        output = self.root / (output_name or f"repro-build-{build_id.lower()}.tar")
        package_evidence(
            build_id=build_id,
            source_archive=self.source_archive,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
            build_image_id=IMAGE_ID,
            binary=binary or self.binary,
            bundle=bundle or self.bundle,
            native_manifest=native or self.native,
            toolchain_log=self.logs / "toolchain.txt",
            container_inspect=self.logs / f"container-inspect-{build_id.lower()}.json",
            preflight_log=self.logs / "preflight.log",
            cargo_build_log=self.logs / "cargo-build.log",
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
            source_revision=REVISION,
            source_date_epoch=EPOCH,
            build_image_id=IMAGE_ID,
            final_binary=self.binary,
            final_bundle=self.bundle,
            final_native_manifest=self.native,
        )

    def test_two_identical_independent_builds_pass(self) -> None:
        report = self.check(self.package("A"), self.package("B"))
        self.assertEqual(report["gate_id"], GATE_ID)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["build"]["independent_clean_containers"], 2)
        self.assertTrue(report["comparisons"]["bundle_a_b_final_byte_exact"])
        self.assertEqual(
            canonical_json_bytes(report),
            canonical_json_bytes(json.loads(canonical_json_bytes(report))),
        )

    def test_git_archive_file_then_same_prefix_directory_order_is_accepted(self) -> None:
        _validate_source_archive(self.source_archive, REVISION, EPOCH)

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

    def test_a_b_must_use_distinct_anonymous_workspaces(self) -> None:
        document = container_inspect("B")
        workspace = next(
            mount
            for mount in document[0]["Mounts"]
            if mount["Destination"] == "/workspace"
        )
        workspace["Name"] = "anonymous-workspace-a"
        (self.logs / "container-inspect-b.json").write_bytes(
            canonical_json_bytes(document)
        )
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
        with self.assertRaisesRegex(ReleaseContractError, "bundle binary differs"):
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

    def test_source_archive_must_embed_exact_git_revision(self) -> None:
        wrong_source = self.root / "wrong-source.tar"
        write_source_archive(wrong_source, revision="c" * 40)
        with self.assertRaisesRegex(ReleaseContractError, "exact git archive revision"):
            check_reproducible_build(
                evidence_a=self.package("A"),
                evidence_b=self.package("B"),
                source_archive=wrong_source,
                source_revision=REVISION,
                source_date_epoch=EPOCH,
                build_image_id=IMAGE_ID,
                final_binary=self.binary,
                final_bundle=self.bundle,
                final_native_manifest=self.native,
            )

    def test_package_rejects_linked_binary_input(self) -> None:
        linked = self.root / "linked-binary"
        linked.symlink_to(self.binary)
        with self.assertRaisesRegex(ReleaseContractError, "regular file"):
            self.package("A", binary=linked)


class ReproducibilityRunnerStaticTests(unittest.TestCase):
    def test_host_runner_uses_two_inspected_networkless_containers(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        contents = (repository / "ci/run_release_reproducibility.sh").read_text(
            encoding="utf-8"
        )
        for marker in (
            "docker create",
            "--runtime runc",
            "--network none",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "type=volume,destination=/workspace,volume-nocopy",
            'docker inspect "${container_id}"',
            'run_one A',
            'run_one B',
            "check_reproducible_build.py",
        ):
            self.assertIn(marker, contents)
        self.assertNotIn("--gpus", contents)

    def test_container_build_command_is_locked_and_offline(self) -> None:
        contents = Path(__file__).with_name("run_reproducible_build_once.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "cargo build --locked --offline --release --features cuda,server",
            contents,
        )
        self.assertIn("CARGO_NET_OFFLINE=true", contents)

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
        self.assertNotIn("cargo build", contents)
        self.assertNotIn("COPY .", contents)


if __name__ == "__main__":
    unittest.main()
