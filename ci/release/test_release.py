#!/usr/bin/env python3
"""CPU-only unit tests for release packaging and static runtime contracts."""

from __future__ import annotations

import gzip
import io
import struct
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent))

from build_release_bundle import build_bundle  # noqa: E402
from release_common import (  # noqa: E402
    ALLOWED_NATIVE_DEPENDENCIES,
    ReleaseContractError,
    validate_binary,
)
from verify_release_bundle import verify_bundle  # noqa: E402
from verify_runtime_dockerfile import verify_dockerfile  # noqa: E402

EPOCH = 1_700_000_000
REVISION = "a" * 40
DEPENDENCIES = sorted(
    {
        "libc.so.6",
        "libcublasLt.so.12",
        "libcuda.so.1",
        "libcudart.so.12",
        "libgcc_s.so.1",
    }
)
assert set(DEPENDENCIES) <= ALLOWED_NATIVE_DEPENDENCIES


def fixture_elf(dependencies: list[str] = DEPENDENCIES) -> bytes:
    strings = bytearray(b"\0")
    needed_offsets: list[int] = []
    for dependency in dependencies:
        needed_offsets.append(len(strings))
        strings.extend(dependency.encode("ascii") + b"\0")

    dynamic_offset = 0x200
    string_offset = 0x300
    virtual_base = 0x400000
    dynamic_entries = [(5, virtual_base + string_offset), (10, len(strings))]
    dynamic_entries.extend((1, offset) for offset in needed_offsets)
    dynamic_entries.append((0, 0))
    dynamic = b"".join(struct.pack("<qQ", *entry) for entry in dynamic_entries)
    total_size = string_offset + len(strings)
    binary = bytearray(total_size)
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        binary,
        0,
        ident,
        3,
        62,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        2,
        0,
        0,
        0,
    )
    struct.pack_into(
        "<IIQQQQQQ",
        binary,
        64,
        1,
        5,
        0,
        virtual_base,
        virtual_base,
        total_size,
        total_size,
        0x1000,
    )
    struct.pack_into(
        "<IIQQQQQQ",
        binary,
        120,
        2,
        4,
        dynamic_offset,
        virtual_base + dynamic_offset,
        virtual_base + dynamic_offset,
        len(dynamic),
        len(dynamic),
        8,
    )
    binary[dynamic_offset : dynamic_offset + len(dynamic)] = dynamic
    binary[string_offset : string_offset + len(strings)] = strings
    return bytes(binary)


def rewrite_archive(
    source: Path,
    destination: Path,
    mutate: Callable[[list[tuple[tarfile.TarInfo, bytes | None]]], None],
) -> None:
    with tarfile.open(source, "r:gz") as archive:
        entries = []
        for member in archive.getmembers():
            contents = archive.extractfile(member).read() if member.isreg() else None
            entries.append((member, contents))
    mutate(entries)
    entries.sort(key=lambda entry: entry[0].name)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=EPOCH) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for member, contents in entries:
                    archive.addfile(member, io.BytesIO(contents) if contents is not None else None)


class ReleaseBundleTests(unittest.TestCase):
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
            "Owner-approved fixture license for release contract unit tests.\n"
            "Permission is granted only inside this temporary test fixture.\n",
            encoding="utf-8",
        )
        self.binary = self.root / "rustinfer"
        self.binary.write_bytes(fixture_elf())
        self.binary.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build(self, name: str = "release.tar.gz") -> Path:
        output = self.root / name
        build_bundle(
            binary_path=self.binary,
            output=output,
            repository_root=self.repository,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
        )
        return output

    def test_bundle_is_deterministic_and_verifies(self) -> None:
        first = self.build("first.tar.gz")
        second = self.build("second.tar.gz")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        verify_bundle(first)

    def test_safe_application_strings_are_not_runtime_dependencies(self) -> None:
        binary = fixture_elf() + b"transformers_version ExperimentalTriton"
        self.assertEqual(validate_binary(binary), DEPENDENCIES)

    def test_nvcc_process_named_temporary_symbol_is_rejected(self) -> None:
        markers = (
            b"\0tmpxft_000002ab_00000000-6_batch_primitives.cudafe1.cpp\0",
            b"\0/tmp/tmpxft_000002ab_00000000-6_gemm.cudafe1.cpp\0",
            b"\0tmpxft_ABCDEF01_0000000A-12_memory.cudafe1.cpp\0",
            b"\0tmpxft_000002ab_00000000-6_version.cudafe1.stub.c\0",
        )
        for marker in markers:
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(
                    ReleaseContractError,
                    "nondeterministic nvcc temporary symbol",
                ):
                    validate_binary(fixture_elf() + marker)
        self.assertEqual(
            validate_binary(fixture_elf() + b"\0tmpxft files are discussed here\0"),
            DEPENDENCIES,
        )

    def test_missing_owner_selected_license_fails_preflight(self) -> None:
        (self.repository / "LICENSE").unlink()
        with self.assertRaisesRegex(ReleaseContractError, "LICENSE"):
            self.build()

    def test_missing_cargo_license_metadata_fails_preflight(self) -> None:
        (self.repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ReleaseContractError, "workspace.package.license"):
            self.build()

    def test_path_traversal_is_rejected(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "traversal.tar.gz"

        def add_traversal(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            member = tarfile.TarInfo("../escape")
            member.size = 1
            member.mode = 0o644
            member.mtime = EPOCH
            entries.append((member, b"x"))

        rewrite_archive(valid, invalid, add_traversal)
        with self.assertRaisesRegex(ReleaseContractError, "unsafe archive member path"):
            verify_bundle(invalid)

    def test_python_artifact_path_is_rejected(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "python-artifact.tar.gz"

        def add_python(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = entries[0][0].name.split("/", 1)[0]
            member = tarfile.TarInfo(f"{root}/bootstrap.py")
            member.size = 1
            member.mode = 0o644
            member.mtime = EPOCH
            entries.append((member, b"x"))

        rewrite_archive(valid, invalid, add_python)
        with self.assertRaisesRegex(ReleaseContractError, "forbidden Python runtime artifact"):
            verify_bundle(invalid)

    def test_symlink_is_rejected(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "symlink.tar.gz"

        def replace_binary(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            for index, (member, _) in enumerate(entries):
                if member.name.endswith("/bin/rustinfer"):
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/bin/true"
                    member.size = 0
                    entries[index] = (member, None)

        rewrite_archive(valid, invalid, replace_binary)
        with self.assertRaisesRegex(ReleaseContractError, "links and special files"):
            verify_bundle(invalid)

    def test_extra_file_is_rejected(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "extra.tar.gz"

        def add_extra(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            root = entries[0][0].name.split("/", 1)[0]
            member = tarfile.TarInfo(f"{root}/extra.txt")
            member.size = 1
            member.mode = 0o644
            member.mtime = EPOCH
            entries.append((member, b"x"))

        rewrite_archive(valid, invalid, add_extra)
        with self.assertRaisesRegex(ReleaseContractError, "unreviewed extra members"):
            verify_bundle(invalid)

    def test_tampered_file_is_rejected_by_checksum(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "tampered.tar.gz"

        def tamper_license(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            for index, (member, contents) in enumerate(entries):
                if member.name.endswith("/LICENSE"):
                    replacement = contents + b"tampered\n"
                    member.size = len(replacement)
                    entries[index] = (member, replacement)

        rewrite_archive(valid, invalid, tamper_license)
        with self.assertRaisesRegex(ReleaseContractError, "SHA-256 mismatch"):
            verify_bundle(invalid)

    def test_native_manifest_must_match_elf(self) -> None:
        valid = self.build("valid.tar.gz")
        invalid = self.root / "dependency-mismatch.tar.gz"

        def tamper_manifest(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
            for index, (member, contents) in enumerate(entries):
                if member.name.endswith("/manifest/native-dependencies.txt"):
                    replacement = contents.replace(b"dependency=libgcc_s.so.1\n", b"")
                    member.size = len(replacement)
                    entries[index] = (member, replacement)

        rewrite_archive(valid, invalid, tamper_manifest)
        with self.assertRaisesRegex(ReleaseContractError, "does not match"):
            verify_bundle(invalid)


class RuntimeDockerfileTests(unittest.TestCase):
    def test_runtime_dockerfile_contract(self) -> None:
        verify_dockerfile()


if __name__ == "__main__":
    unittest.main()
