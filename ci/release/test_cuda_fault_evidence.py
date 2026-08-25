#!/usr/bin/env python3
"""CPU-only adversarial tests for the CUDA fault raw-evidence checker."""

from __future__ import annotations

import hashlib
import io
import json
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
    GATE,
    SANITIZER_FILES,
    SUMMARY,
    CudaFaultEvidenceError,
    produce,
    validate,
)
from test_release import EPOCH, fixture_elf  # noqa: E402


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
        self.release_binary = root / "rustinfer"
        self.release_bundle = root / "rustinfer.tar.gz"
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
            'license = "LicenseRef-Test-Fixture"\n',
            encoding="utf-8",
        )
        (repository / "LICENSE").write_text(
            "Owner-approved fixture license for release evidence unit tests.\n"
            "This fixture text is intentionally longer than the release minimum.\n",
            encoding="utf-8",
        )
        build_bundle(
            binary_path=self.release_binary,
            output=self.release_bundle,
            repository_root=repository,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
        )
        self._write_base_evidence()

    def _fault_log(self) -> bytes:
        lines = ["running 1 test"]
        parent_pid = 4000
        for index, case in enumerate(FAULT_CASES):
            child_pid = 4100 + index
            lines.extend(
                [
                    f"rustinfer-cuda-memory-fault-case case={case} event=spawn "
                    f"parent_pid={parent_pid} child_pid={child_pid}",
                    f"rustinfer-cuda-memory-fault-case case={case} event=start "
                    f"child_pid={child_pid}",
                    f"rustinfer-cuda-memory-fault-case case={case} event=passed "
                    f"child_pid={child_pid}",
                    "test memory_fault_subprocess ... ok",
                    SUMMARY,
                    f"rustinfer-cuda-memory-fault-case case={case} event=joined "
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
            "Linux fixture 6.8.0 x86_64\n"
            "rustc 1.85.0\n"
            "nvcc release 12.8\n"
        ).encode()

    def _write_base_evidence(self) -> None:
        placeholder = b"fixture evidence\n"
        for name in sorted(BASE_EVIDENCE_FILES - {"SHA256SUMS"}):
            (self.evidence / name).write_bytes(placeholder)
        (self.evidence / "environment.txt").write_bytes(self._environment())
        (self.evidence / "memory-fault-test-list.txt").write_text(
            "memory_fault_cases_are_subprocess_isolated: test\n"
            "memory_fault_subprocess: test\n",
            encoding="utf-8",
        )
        (self.evidence / "memory-fault-tests.log").write_bytes(self._fault_log())
        (self.evidence / "host-runtime-test-binary.sha256").write_text(
            f"{digest(b'host test binary')}  target/debug/deps/host_runtime_gpu-0123456789abcdef\n",
            encoding="ascii",
        )
        (self.evidence / "memory-test-binary.sha256").write_text(
            f"{digest(b'memory test binary')}  target/debug/deps/memory_gpu-1234567890abcdef\n",
            encoding="ascii",
        )
        (self.evidence / "memory-fault-test-binary.sha256").write_text(
            f"{digest(b'fault test binary')}  target/debug/deps/memory_fault_injection_gpu-234567890abcdef1\n",
            encoding="ascii",
        )
        binary_sha256 = digest(self.release_binary.read_bytes())
        (self.evidence / "release-binary.sha256").write_text(
            f"{binary_sha256}  target/release/rustinfer\n",
            encoding="ascii",
        )
        (self.evidence / "release-ldd.txt").write_text(
            "artifact=target/release/rustinfer\n"
            "libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12\n"
            "libcuda.so.1 => /usr/lib/x86_64-linux-gnu/libcuda.so.1\n",
            encoding="utf-8",
        )
        (self.evidence / "release-readelf.txt").write_text(
            "artifact=target/release/rustinfer\n"
            "0x0000000000000001 (NEEDED) Shared library: [libcudart.so.12]\n"
            "0x0000000000000001 (NEEDED) Shared library: [libcuda.so.1]\n",
            encoding="utf-8",
        )
        (self.evidence / "release-nm.txt").write_text(
            "artifact=target/release/rustinfer\n0000000000001000 T main\n",
            encoding="utf-8",
        )
        self.refresh_checksums()

    def refresh_checksums(self) -> None:
        names = sorted(path.name for path in self.evidence.iterdir() if path.name != "SHA256SUMS")
        contents = "".join(
            f"{digest((self.evidence / name).read_bytes())}  {name}\n" for name in names
        )
        (self.evidence / "SHA256SUMS").write_text(contents, encoding="ascii")

    def enable_sanitizer(self) -> None:
        (self.evidence / "environment.txt").write_bytes(self._environment(sanitizer=True))
        for name in SANITIZER_FILES:
            (self.evidence / name).write_text(
                "ERROR SUMMARY: 0 errors\nLEAK SUMMARY: 0 bytes leaked\n",
                encoding="utf-8",
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

    def test_checksum_tampering_is_rejected(self) -> None:
        (self.fixture.evidence / "memory-fault-tests.log").write_bytes(b"tampered\n")
        with self.assertRaisesRegex(CudaFaultEvidenceError, "digest mismatch"):
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
            "rustinfer-cuda-memory-fault-case case=explicit-close-ambiguous "
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
        path.write_text(f"{'f' * 64}  target/release/rustinfer\n")
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "inspected production release binary"):
            self.fixture.validate()

    def test_production_nm_fault_symbol_is_rejected(self) -> None:
        path = self.fixture.evidence / "release-nm.txt"
        path.write_text(path.read_text() + "0000000000002000 T rustinfer_cuda_test_memory_fault_arm\n")
        self.fixture.refresh_checksums()
        with self.assertRaisesRegex(CudaFaultEvidenceError, "fault-injection symbol"):
            self.fixture.validate()

    def test_supplied_production_binary_fault_symbol_is_rejected(self) -> None:
        binary = self.fixture.root / "faulty-rustinfer"
        binary.write_bytes(self.fixture.release_binary.read_bytes() + b"rustinfer_cuda_test_memory_fault_arm\0")
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
            "grep -aFq 'rustinfer_cuda_test_memory_fault_' \"$release_binary\"",
            runner,
        )

    def test_fault_harness_emits_pid_bound_child_markers(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        harness = (
            repository
            / "crates/rustinfer-cuda/tests/memory_fault_injection_gpu.rs"
        ).read_text()
        self.assertIn(".spawn()?", harness)
        self.assertIn("let child_pid = child.id();", harness)
        for event in ("spawn", "start", "passed", "joined"):
            self.assertIn(f"event={event}", harness)
        for case in FAULT_CASES:
            self.assertIn(f'"{case}"', harness)


if __name__ == "__main__":
    unittest.main()
