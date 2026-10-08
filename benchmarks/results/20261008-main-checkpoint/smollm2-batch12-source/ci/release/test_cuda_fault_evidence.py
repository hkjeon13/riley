#!/usr/bin/env python3
"""CPU-only adversarial tests for the CUDA fault raw-evidence checker."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_release_bundle import build_bundle  # noqa: E402
from check_cuda_fault_evidence import (  # noqa: E402
    ATTESTATION_VERSION,
    BASE_EVIDENCE_FILES,
    CHECK_IDS,
    FAULT_CASES,
    FAULT_PREFIX,
    GATE,
    HOST_RUNTIME_TESTS,
    MARKER_PREFIX,
    MEMORY_TESTS,
    SANITIZER_FILES,
    SUMMARY,
    CudaFaultEvidenceError,
    load_raw_evidence_archive,
    produce,
    replay_raw_evidence,
    validate,
)
from release_common import MIT_LICENSE_BYTES  # noqa: E402
from test_release import (  # noqa: E402
    DEPENDENCIES,
    EPOCH,
    fixture_elf,
    install_reviewed_server_defaults_source,
)


REVISION = "1a2b3c4d5e6f78901234567890abcdef12345678"
BUILD_IMAGE_ID = "sha256:" + hashlib.sha256(b"CUDA build image").hexdigest()
RELEASE_IMAGE_ID = "sha256:" + hashlib.sha256(b"minimal release image").hexdigest()


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "gpu-evidence"
        self.evidence.mkdir()
        self.source_archive = root / "source.tar"
        self.release_binary = root / "riley"
        self.release_bundle = root / "riley.tar.gz"
        self.host_path = "target/debug/deps/host_runtime_gpu-0123456789abcdef"
        self.memory_path = "target/debug/deps/memory_gpu-1234567890abcdef"
        self.fault_path = (
            "target/debug/deps/memory_fault_injection_gpu-234567890abcdef1"
        )
        self.counter = 0

        with tarfile.open(
            self.source_archive,
            "w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": REVISION},
        ) as archive:
            contents = b"source archive fixture\n"
            member = tarfile.TarInfo("README.md")
            member.size = len(contents)
            member.mtime = EPOCH
            archive.addfile(member, io.BytesIO(contents))

        self.release_binary.write_bytes(fixture_elf())
        self.release_binary.chmod(0o755)
        repository = root / "repository"
        repository.mkdir()
        (repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\n'
            'license = "MIT"\n',
            encoding="utf-8",
        )
        (repository / "LICENSE").write_bytes(MIT_LICENSE_BYTES)
        install_reviewed_server_defaults_source(repository)
        build_bundle(
            binary_path=self.release_binary,
            output=self.release_bundle,
            repository_root=repository,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
        )
        self._write_base_evidence()

    def _test_elf(self, required_strings: set[str], *, fault: bool = False) -> bytes:
        binary = bytearray(fixture_elf())
        struct.pack_into("<Q", binary, 24, 0x400000)
        binary.extend(b"\0")
        for value in sorted(required_strings):
            binary.extend(value.encode("ascii") + b"\0")
        if fault:
            binary.extend(FAULT_PREFIX.encode("ascii") + b"arm\0")
        return bytes(binary)

    def _test_list(self, source: str, artifact_path: str, tests: set[str]) -> bytes:
        lines = [
            "Finished `test` profile [unoptimized + debuginfo] target(s) in 0.01s",
            f"Running tests/{source}.rs ({artifact_path})",
            *(f"{name}: test" for name in sorted(tests)),
            "",
        ]
        return "\n".join(lines).encode()

    def _host_log(self) -> bytes:
        lines = [
            "Finished `test` profile [unoptimized + debuginfo] target(s) in 0.01s",
            f"Running tests/host_runtime_gpu.rs ({self.host_path})",
            "",
            "running 8 tests",
        ]
        for name in sorted(HOST_RUNTIME_TESTS):
            if name == "device_metadata_is_reported":
                lines.extend(
                    [
                        f"test {name} ... riley-cuda-device-metadata ordinal=0 "
                        "name=NVIDIA GeForce RTX 4090 compute_capability=8.9 "
                        "total_memory_bytes=25250627584 multiprocessor_count=128 "
                        "driver_version=13000 runtime_version=12080",
                        "ok",
                    ]
                )
            elif name == "repeated_create_drop_has_no_resource_leak":
                lines.extend(
                    [
                        f"test {name} ... riley-cuda-leak-smoke iterations=128 "
                        "before_free_bytes=24594284544 after_free_bytes=24594284544",
                        "ok",
                    ]
                )
            else:
                lines.append(f"test {name} ... ok")
        lines.extend(
            [
                "",
                "test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; "
                "0 filtered out; finished in 0.19s",
                "",
            ]
        )
        return "\n".join(lines).encode()

    def _memory_log(self) -> bytes:
        lines = [
            "Finished `test` profile [unoptimized + debuginfo] target(s) in 0.01s",
            f"Running tests/memory_gpu.rs ({self.memory_path})",
            "",
            "running 5 tests",
        ]
        for name in sorted(MEMORY_TESTS):
            if name == "allocation_accounting_returns_to_zero":
                lines.extend(
                    [
                        f"test {name} ...",
                        "riley-cuda-memory-accounting device_live_bytes=0 "
                        "device_live_allocations=0 pinned_host_live_bytes=0 "
                        "pinned_host_live_allocations=0",
                        "ok",
                    ]
                )
            else:
                lines.append(f"test {name} ... ok")
        lines.extend(
            [
                "",
                "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; "
                "0 filtered out; finished in 0.08s",
                "",
            ]
        )
        return "\n".join(lines).encode()

    def _fault_log(self) -> bytes:
        lines = [
            "Finished `test` profile [unoptimized + debuginfo] target(s) in 0.01s",
            f"Running tests/memory_fault_injection_gpu.rs ({self.fault_path})",
            "running 1 test",
        ]
        parent_pid = 4000
        for index, case in enumerate(FAULT_CASES):
            child_pid = 4100 + index
            lines.extend(
                [
                    f"riley-cuda-memory-fault-case case={case} event=spawn "
                    f"parent_pid={parent_pid} child_pid={child_pid}",
                    f"riley-cuda-memory-fault-case case={case} event=start "
                    f"child_pid={child_pid}",
                    "running 1 test",
                    f"riley-cuda-memory-fault-case case={case} event=passed "
                    f"child_pid={child_pid}",
                    "test memory_fault_subprocess ... ok",
                    SUMMARY,
                    f"riley-cuda-memory-fault-case case={case} event=joined "
                    f"parent_pid={parent_pid} child_pid={child_pid} exit_code=0",
                ]
            )
        lines.extend(
            [
                "test memory_fault_cases_are_subprocess_isolated ... ok",
                SUMMARY,
                "",
            ]
        )
        return "\n".join(lines).encode()

    def _elf_evidence(
        self,
        prefix: str,
        artifact_path: str,
        binary_sha256: str,
        *,
        fault: bool = False,
    ) -> None:
        headers = f"artifact={artifact_path}\nsha256={binary_sha256}\n"
        ldd = headers + "".join(
            f"{dependency} => /usr/lib/x86_64-linux-gnu/{dependency}\n"
            for dependency in DEPENDENCIES
        )
        readelf = headers + "".join(
            "0x0000000000000001 (NEEDED) Shared library: "
            f"[{dependency}]\n"
            for dependency in DEPENDENCIES
        )
        nm_symbol = (
            "0000000000002000 T riley_cuda_test_memory_fault_arm\n"
            if fault
            else "                 U cudaGetDeviceCount\n"
        )
        (self.evidence / f"{prefix}-ldd.txt").write_text(ldd, encoding="utf-8")
        (self.evidence / f"{prefix}-readelf.txt").write_text(
            readelf,
            encoding="utf-8",
        )
        (self.evidence / f"{prefix}-nm.txt").write_text(
            headers + nm_symbol,
            encoding="utf-8",
        )

    def _environment(self, *, sanitizer: bool = False) -> bytes:
        return (
            f"source_revision={REVISION}\n"
            "source_archive_command=git archive --format=tar HEAD\n"
            f"source_archive_sha256={digest(self.source_archive.read_bytes())}\n"
            f"gpu_image_id={BUILD_IMAGE_ID}\n"
            "cuda_visible_devices=all\n"
            "nvidia_visible_devices=all\n"
            "leak_iterations=128\n"
            f"compute_sanitizer={int(sanitizer)}\n"
            "Linux fixture 6.8.0 x86_64 GNU/Linux\n"
            "rustc 1.85.0 (4d91de4e4 2025-02-17)\n"
            "cargo 1.85.0 (d73d2caf9 2024-12-31)\n"
            "Cuda compilation tools, release 12.8, V12.8.93\n"
            "riley 0.1.0 (server=true, cuda=true, cuda_abi=1)\n"
        ).encode()

    def _write_base_evidence(self) -> None:
        placeholder = b"fixture evidence\n"
        for name in sorted(BASE_EVIDENCE_FILES - {"SHA256SUMS"}):
            (self.evidence / name).write_bytes(placeholder)
        (self.evidence / "environment.txt").write_bytes(self._environment())
        (self.evidence / "nvidia-smi-list.txt").write_text(
            "GPU 0: NVIDIA GeForce RTX 4090 "
            "(UUID: GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0)\n",
            encoding="ascii",
        )
        (self.evidence / "nvidia-smi-device-metadata.csv").write_text(
            "0, GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0, "
            "NVIDIA GeForce RTX 4090, 8.9, 24564, 580.173.02\n",
            encoding="ascii",
        )
        (self.evidence / "cuda-driver-libraries.txt").write_text(
            "libcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1\n"
            "libcudart.so.12 (libc6,x86-64) => /usr/local/cuda/lib64/libcudart.so.12\n",
            encoding="utf-8",
        )
        (self.evidence / "host-runtime-test-list.txt").write_bytes(
            self._test_list("host_runtime_gpu", self.host_path, HOST_RUNTIME_TESTS)
        )
        (self.evidence / "memory-test-list.txt").write_bytes(
            self._test_list("memory_gpu", self.memory_path, MEMORY_TESTS)
        )
        (self.evidence / "memory-fault-test-list.txt").write_bytes(
            self._test_list(
                "memory_fault_injection_gpu",
                self.fault_path,
                {"memory_fault_cases_are_subprocess_isolated", "memory_fault_subprocess"},
            )
        )
        (self.evidence / "host-runtime-tests.log").write_bytes(self._host_log())
        (self.evidence / "memory-tests.log").write_bytes(self._memory_log())
        (self.evidence / "memory-fault-tests.log").write_bytes(self._fault_log())

        host_binary = self._test_elf(
            HOST_RUNTIME_TESTS
            | {"riley-cuda-device-metadata", "riley-cuda-leak-smoke"}
        )
        memory_binary = self._test_elf(
            MEMORY_TESTS | {"riley-cuda-memory-accounting"}
        )
        fault_binary = self._test_elf(
            {"memory_fault_cases_are_subprocess_isolated", "memory_fault_subprocess"}
            | set(FAULT_CASES)
            | {MARKER_PREFIX, "RILEY_CUDA_MEMORY_FAULT_CHILD"},
            fault=True,
        )
        (self.evidence / "host-runtime-test-binary").write_bytes(host_binary)
        (self.evidence / "memory-test-binary").write_bytes(memory_binary)
        (self.evidence / "memory-fault-test-binary").write_bytes(fault_binary)
        (self.evidence / "host-runtime-test-binary.sha256").write_text(
            f"{digest(host_binary)}  {self.host_path}\n",
            encoding="ascii",
        )
        (self.evidence / "memory-test-binary.sha256").write_text(
            f"{digest(memory_binary)}  {self.memory_path}\n",
            encoding="ascii",
        )
        (self.evidence / "memory-fault-test-binary.sha256").write_text(
            f"{digest(fault_binary)}  {self.fault_path}\n",
            encoding="ascii",
        )
        self._elf_evidence("host-runtime", self.host_path, digest(host_binary))
        self._elf_evidence("memory", self.memory_path, digest(memory_binary))
        self._elf_evidence(
            "memory-fault",
            self.fault_path,
            digest(fault_binary),
            fault=True,
        )
        binary_sha256 = digest(self.release_binary.read_bytes())
        (self.evidence / "release-binary.sha256").write_text(
            f"{binary_sha256}  target/release/riley\n",
            encoding="ascii",
        )
        self._elf_evidence("release", "target/release/riley", binary_sha256)
        self.refresh_checksums()

    def refresh_checksums(self) -> None:
        names = sorted(path.name for path in self.evidence.iterdir() if path.name != "SHA256SUMS")
        contents = "".join(
            f"{digest((self.evidence / name).read_bytes())}  {name}\n" for name in names
        )
        (self.evidence / "SHA256SUMS").write_text(contents, encoding="ascii")

    def replace_test_binary(
        self,
        evidence_name: str,
        receipt_name: str,
        log_prefix: str,
        contents: bytes,
    ) -> None:
        receipt_path = self.evidence / receipt_name
        old_digest, artifact_path = receipt_path.read_text(encoding="ascii").rstrip("\n").split(
            "  ",
            1,
        )
        new_digest = digest(contents)
        (self.evidence / evidence_name).write_bytes(contents)
        receipt_path.write_text(
            f"{new_digest}  {artifact_path}\n",
            encoding="ascii",
        )
        for suffix in ("ldd.txt", "readelf.txt", "nm.txt"):
            path = self.evidence / f"{log_prefix}-{suffix}"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    f"sha256={old_digest}\n",
                    f"sha256={new_digest}\n",
                    1,
                ),
                encoding="utf-8",
            )
        self.refresh_checksums()

    def enable_sanitizer(self) -> None:
        (self.evidence / "environment.txt").write_bytes(self._environment(sanitizer=True))
        suffix = b"ERROR SUMMARY: 0 errors\nLEAK SUMMARY: 0 bytes leaked\n"
        (self.evidence / "compute-sanitizer-memcheck.log").write_bytes(
            self._host_log() + suffix
        )
        (self.evidence / "compute-sanitizer-memory-memcheck.log").write_bytes(
            self._memory_log() + suffix
        )
        self.refresh_checksums()

    def validate(self, **overrides: object) -> tuple[dict[str, bytes], dict[str, object]]:
        arguments: dict[str, object] = {
            "source_revision": REVISION,
            "source_archive": self.source_archive,
            "build_image_id": BUILD_IMAGE_ID,
            "release_binary": self.release_binary,
            "release_bundle": self.release_bundle,
            "release_image_id": RELEASE_IMAGE_ID,
        }
        arguments.update(overrides)
        return validate(self.evidence, **arguments)  # type: ignore[arg-type]

    def produce(self) -> tuple[dict[str, object], Path, Path]:
        self.counter += 1
        raw = self.root / f"raw-{self.counter}.tar"
        report = self.root / f"report-{self.counter}.json"
        attestation = produce(
            self.evidence,
            source_revision=REVISION,
            source_archive=self.source_archive,
            build_image_id=BUILD_IMAGE_ID,
            release_binary=self.release_binary,
            release_bundle=self.release_bundle,
            release_image_id=RELEASE_IMAGE_ID,
            raw_evidence=raw,
            report=report,
        )
        return attestation, raw, report


class CudaFaultEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_evidence_produces_closed_attestation_and_deterministic_raw_tar(self) -> None:
        first, first_raw, first_report = self.fixture.produce()
        second, second_raw, second_report = self.fixture.produce()
        self.assertEqual(first, second)
        self.assertEqual(first_raw.read_bytes(), second_raw.read_bytes())
        self.assertEqual(first_report.read_bytes(), second_report.read_bytes())
        self.assertEqual(first["schema_version"], ATTESTATION_VERSION)
        self.assertEqual(first["gate"], GATE)
        self.assertEqual(first["status"], "passed")
        self.assertEqual(
            {row["id"] for row in first["checks"]},  # type: ignore[index]
            set(CHECK_IDS),
        )
        self.assertEqual(first["raw_evidence_sha256"], digest(first_raw.read_bytes()))
        self.assertEqual(json.loads(first_report.read_bytes()), first)
        with tarfile.open(first_raw, "r:") as archive:
            self.assertEqual(
                [member.name for member in archive.getmembers()],
                sorted(BASE_EVIDENCE_FILES),
            )
            for member in archive.getmembers():
                self.assertEqual((member.uid, member.gid, member.mtime, member.mode), (0, 0, 0, 0o644))

    def test_preserved_raw_archive_replays_to_exact_attestation(self) -> None:
        expected, raw, _ = self.fixture.produce()
        files, environment, raw_sha256 = load_raw_evidence_archive(raw)
        self.assertEqual(set(files), BASE_EVIDENCE_FILES)
        self.assertEqual(environment["gpu_image_id"], BUILD_IMAGE_ID)
        self.assertEqual(raw_sha256, digest(raw.read_bytes()))
        replayed = replay_raw_evidence(
            raw,
            source_revision=REVISION,
            source_archive=self.fixture.source_archive,
            build_image_id=BUILD_IMAGE_ID,
            release_binary=self.fixture.release_binary,
            release_bundle=self.fixture.release_bundle,
            release_image_id=RELEASE_IMAGE_ID,
        )
        self.assertEqual(replayed, expected)

    def test_preserved_raw_archive_rejects_payload_tampering(self) -> None:
        _, raw, _ = self.fixture.produce()
        contents = raw.read_bytes()
        self.assertIn(b"NVIDIA GeForce RTX 4090", contents)
        raw.write_bytes(
            contents.replace(b"NVIDIA GeForce RTX 4090", b"NVIDIB GeForce RTX 4090", 1)
        )
        with self.assertRaisesRegex(CudaFaultEvidenceError, "digest mismatch"):
            replay_raw_evidence(
                raw,
                source_revision=REVISION,
                source_archive=self.fixture.source_archive,
                build_image_id=BUILD_IMAGE_ID,
                release_binary=self.fixture.release_binary,
                release_bundle=self.fixture.release_bundle,
                release_image_id=RELEASE_IMAGE_ID,
            )

    def test_preserved_raw_archive_rejects_noncanonical_metadata(self) -> None:
        _, raw, _ = self.fixture.produce()
        files, _, _ = load_raw_evidence_archive(raw)
        noncanonical = self.fixture.root / "noncanonical.tar"
        with tarfile.open(noncanonical, "w", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(files):
                member = tarfile.TarInfo(name)
                member.size = len(files[name])
                member.mode = 0o644
                member.uid = 1
                member.gid = 0
                member.uname = "root"
                member.gname = "root"
                member.mtime = 0
                archive.addfile(member, io.BytesIO(files[name]))
        with self.assertRaisesRegex(CudaFaultEvidenceError, "non-canonical metadata"):
            load_raw_evidence_archive(noncanonical)

    def test_preserved_raw_archive_rejects_trailing_tar_record(self) -> None:
        _, raw, _ = self.fixture.produce()
        raw.write_bytes(raw.read_bytes() + b"\0" * 10240)
        with self.assertRaisesRegex(CudaFaultEvidenceError, "end-of-archive padding"):
            load_raw_evidence_archive(raw)

    def test_preserved_raw_archive_rebinds_build_image(self) -> None:
        _, raw, _ = self.fixture.produce()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "gpu_image_id"):
            replay_raw_evidence(
                raw,
                source_revision=REVISION,
                source_archive=self.fixture.source_archive,
                build_image_id="sha256:" + "f" * 64,
                release_binary=self.fixture.release_binary,
                release_bundle=self.fixture.release_bundle,
                release_image_id=RELEASE_IMAGE_ID,
            )

    def test_checksum_tampering_is_rejected(self) -> None:
        (self.fixture.evidence / "memory-fault-tests.log").write_bytes(b"tampered\n")
        with self.assertRaisesRegex(CudaFaultEvidenceError, "digest mismatch"):
            self.fixture.validate()

    def test_forged_receipt_and_placeholder_runtime_logs_are_rejected(self) -> None:
        (self.fixture.evidence / "host-runtime-test-binary.sha256").write_text(
            f"{'f' * 64}  {self.fixture.host_path}\n",
            encoding="ascii",
        )
        (self.fixture.evidence / "host-runtime-tests.log").write_text(
            "all CUDA tests passed\n",
            encoding="utf-8",
        )
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "executed test inventory|running 8 tests"):
            self.fixture.validate()

    def test_preserved_test_binary_must_match_its_checksum_receipt(self) -> None:
        path = self.fixture.evidence / "memory-test-binary"
        path.write_bytes(path.read_bytes() + b"tampered-after-run\0")
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "checksum receipt"):
            self.fixture.validate()

    def test_preserved_test_binary_must_be_executable_linux_x86_64_elf(self) -> None:
        self.fixture.replace_test_binary(
            "host-runtime-test-binary",
            "host-runtime-test-binary.sha256",
            "host-runtime",
            b"\x7fELFsynthetic-placeholder",
        )
        with self.assertRaisesRegex(CudaFaultEvidenceError, "64-bit little-endian|ELF"):
            self.fixture.validate()

    def test_preserved_test_binary_must_contain_reviewed_marker_strings(self) -> None:
        path = self.fixture.evidence / "host-runtime-test-binary"
        contents = path.read_bytes().replace(
            b"riley-cuda-leak-smoke",
            b"riley-cuda-leak-smokf",
            1,
        )
        self.fixture.replace_test_binary(
            "host-runtime-test-binary",
            "host-runtime-test-binary.sha256",
            "host-runtime",
            contents,
        )
        with self.assertRaisesRegex(CudaFaultEvidenceError, "omits reviewed test/marker strings"):
            self.fixture.validate()

    def test_elf_inspection_header_must_bind_preserved_binary_digest(self) -> None:
        path = self.fixture.evidence / "memory-fault-readelf.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(f"sha256={text.splitlines()[1].removeprefix('sha256=')}", f"sha256={'f' * 64}"),
            encoding="utf-8",
        )
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "exact inspected ELF path and SHA-256"):
            self.fixture.validate()

    def test_host_runtime_test_inventory_cannot_be_self_declared(self) -> None:
        path = self.fixture.evidence / "host-runtime-test-list.txt"
        path.write_text(
            path.read_text().replace(
                "command_batch_proxy_is_one_shot_and_drop_restores_stream_use: test\n",
                "",
            )
        )
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "reviewed test inventory mismatch"):
            self.fixture.validate()

    def test_nvidia_inventory_must_be_the_reviewed_sm89_gpu(self) -> None:
        path = self.fixture.evidence / "nvidia-smi-device-metadata.csv"
        path.write_text(path.read_text().replace(", 8.9,", ", 9.0,"), encoding="ascii")
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "RTX 4090/sm89"):
            self.fixture.validate()

    def test_environment_must_prove_pinned_cuda_toolchain(self) -> None:
        path = self.fixture.evidence / "environment.txt"
        path.write_text(path.read_text().replace("release 12.8", "release 12.7"))
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "CUDA 12.8 compiler"):
            self.fixture.validate()

    def test_closed_inventory_rejects_extra_file(self) -> None:
        (self.fixture.evidence / "self-attested-result.txt").write_text("passed\n")
        with self.assertRaisesRegex(CudaFaultEvidenceError, "closed inventory mismatch"):
            self.fixture.validate()

    def test_symlinked_evidence_member_is_rejected(self) -> None:
        target = self.fixture.evidence / "release-nm.txt"
        target.unlink()
        target.symlink_to("environment.txt")
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "regular file"):
            self.fixture.validate()

    def test_source_archive_binding_is_recomputed(self) -> None:
        environment = (self.fixture.evidence / "environment.txt").read_text()
        environment = environment.replace(
            f"source_archive_sha256={digest(self.fixture.source_archive.read_bytes())}",
            f"source_archive_sha256={digest(b'claimed archive')}",
        )
        (self.fixture.evidence / "environment.txt").write_text(environment)
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "source_archive_sha256"):
            self.fixture.validate()

    def test_source_archive_must_carry_git_commit(self) -> None:
        other = self.fixture.root / "wrong-source.tar"
        with tarfile.open(
            other,
            "w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": "0" * 40},
        ) as archive:
            contents = b"wrong\n"
            member = tarfile.TarInfo("README.md")
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
        with self.assertRaisesRegex(CudaFaultEvidenceError, "PAX git archive commit"):
            self.fixture.validate(source_archive=other)

    def test_build_image_binding_must_match_environment(self) -> None:
        with self.assertRaisesRegex(CudaFaultEvidenceError, "gpu_image_id"):
            self.fixture.validate(build_image_id="sha256:" + "f" * 64)

    def test_fault_test_inventory_is_exact(self) -> None:
        (self.fixture.evidence / "memory-fault-test-list.txt").write_text(
            "memory_fault_cases_are_subprocess_isolated: test\n",
            encoding="utf-8",
        )
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "two reviewed tests"):
            self.fixture.validate()

    def test_missing_case_marker_is_rejected(self) -> None:
        log = (self.fixture.evidence / "memory-fault-tests.log").read_text()
        line = (
            "riley-cuda-memory-fault-case case=explicit-close-ambiguous "
            "event=passed child_pid=4101\n"
        )
        (self.fixture.evidence / "memory-fault-tests.log").write_text(log.replace(line, ""))
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "four markers"):
            self.fixture.validate()

    def test_child_pid_reuse_is_rejected(self) -> None:
        path = self.fixture.evidence / "memory-fault-tests.log"
        path.write_text(path.read_text().replace("child_pid=4101", "child_pid=4100"))
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "reused a child process"):
            self.fixture.validate()

    def test_nonzero_child_join_is_rejected(self) -> None:
        path = self.fixture.evidence / "memory-fault-tests.log"
        path.write_text(path.read_text().replace("child_pid=4102 exit_code=0", "child_pid=4102 exit_code=86"))
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "did not exit zero"):
            self.fixture.validate()

    def test_parent_and_child_summary_count_is_exact(self) -> None:
        path = self.fixture.evidence / "memory-fault-tests.log"
        path.write_text(path.read_text().replace(SUMMARY, "", 1))
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "four child and one parent"):
            self.fixture.validate()

    def test_production_checksum_must_match_supplied_binary(self) -> None:
        path = self.fixture.evidence / "release-binary.sha256"
        path.write_text(f"{'f' * 64}  target/release/riley\n")
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "inspected production release binary"):
            self.fixture.validate()

    def test_production_nm_fault_symbol_is_rejected(self) -> None:
        path = self.fixture.evidence / "release-nm.txt"
        path.write_text(path.read_text() + "0000000000002000 T riley_cuda_test_memory_fault_arm\n")
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "fault-injection symbol"):
            self.fixture.validate()

    def test_supplied_production_binary_fault_symbol_is_rejected(self) -> None:
        binary = self.fixture.root / "faulty-riley"
        binary.write_bytes(self.fixture.release_binary.read_bytes() + b"riley_cuda_test_memory_fault_arm\0")
        with self.assertRaisesRegex(CudaFaultEvidenceError, "fault-injection symbol"):
            self.fixture.validate(release_binary=binary)

    def test_sanitizer_inventory_and_zero_results_are_checked(self) -> None:
        self.fixture.enable_sanitizer()
        self.fixture.validate()
        path = self.fixture.evidence / "compute-sanitizer-memcheck.log"
        path.write_text("ERROR SUMMARY: 1 error\nLEAK SUMMARY: 0 bytes leaked\n")
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "zero-error/zero-leak"):
            self.fixture.validate()

    def test_existing_report_is_rejected_before_raw_output(self) -> None:
        raw = self.fixture.root / "new-raw.tar"
        report = self.fixture.root / "existing-report.json"
        report.write_text("owner data\n")
        with self.assertRaisesRegex(CudaFaultEvidenceError, "refusing to replace"):
            produce(
                self.fixture.evidence,
                source_revision=REVISION,
                source_archive=self.fixture.source_archive,
                build_image_id=BUILD_IMAGE_ID,
                release_binary=self.fixture.release_binary,
                release_bundle=self.fixture.release_bundle,
                release_image_id=RELEASE_IMAGE_ID,
                raw_evidence=raw,
                report=report,
            )
        self.assertFalse(raw.exists())

    def test_outputs_inside_raw_evidence_directory_are_rejected(self) -> None:
        with self.assertRaisesRegex(CudaFaultEvidenceError, "outside --evidence-dir"):
            produce(
                self.fixture.evidence,
                source_revision=REVISION,
                source_archive=self.fixture.source_archive,
                build_image_id=BUILD_IMAGE_ID,
                release_binary=self.fixture.release_binary,
                release_bundle=self.fixture.release_bundle,
                release_image_id=RELEASE_IMAGE_ID,
                raw_evidence=self.fixture.evidence / "recursive-raw.tar",
                report=self.fixture.root / "report.json",
            )


class CudaFaultRunnerStaticTests(unittest.TestCase):
    def test_python_free_runner_emits_the_checker_inventory(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        runner = (repository / "ci/verify_python_free_gpu_runtime.sh").read_text()
        for name in sorted(BASE_EVIDENCE_FILES):
            self.assertIn(name, runner)
        self.assertIn(
            "sha256sum $evidence_files | LC_ALL=C sort -k2 >SHA256SUMS",
            runner,
        )
        self.assertIn("nm -a --defined-only \"$release_binary\"", runner)
        self.assertIn(
            "grep -aFq 'riley_cuda_test_memory_fault_' \"$release_binary\"",
            runner,
        )

    def test_fault_harness_emits_pid_bound_child_markers(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        harness = (
            repository
            / "crates/riley-cuda/tests/memory_fault_injection_gpu.rs"
        ).read_text()
        self.assertIn(".spawn()?", harness)
        self.assertIn("let child_pid = child.id();", harness)
        for event in ("spawn", "start", "passed", "joined"):
            self.assertIn(f"event={event}", harness)
        for case in FAULT_CASES:
            self.assertIn(f'"{case}"', harness)


if __name__ == "__main__":
    unittest.main()
