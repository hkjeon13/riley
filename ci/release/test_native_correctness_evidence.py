#!/usr/bin/env python3
"""CPU-only tests for native correctness raw tensor replay."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools/python/reference"))

import check_native_correctness_evidence as checker  # noqa: E402
from test_release import fixture_elf  # noqa: E402
from rustinfer_reference import calibration, oracle_calibration  # noqa: E402


CALIBRATION_TEST = (
    REPOSITORY_ROOT / "tools/python/reference/tests/test_calibration.py"
)
CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "native_correctness_calibration_fixture", CALIBRATION_TEST
)
assert CALIBRATION_SPEC is not None and CALIBRATION_SPEC.loader is not None
calibration_fixture_module = importlib.util.module_from_spec(CALIBRATION_SPEC)
sys.modules[CALIBRATION_SPEC.name] = calibration_fixture_module
CALIBRATION_SPEC.loader.exec_module(calibration_fixture_module)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def write_safetensors(path: Path, tensors: dict[str, object]) -> None:
    metadata: dict[str, object] = {}
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        dtype = "F32" if str(tensor.dtype) == "float32" else "BF16"
        raw = b"".join(
            struct.pack("<f", float(value)) if dtype == "F32" else _bf16(float(value))
            for value in tensor.values
        )
        metadata[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads.append(raw)
        offset += len(raw)
    header = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    padding = (-len(header)) % 8
    header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(payloads))


class NativeFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repository = root / "repository"
        self.repository.mkdir()
        self.calibration = calibration_fixture_module.CalibrationFixture(
            self.repository
        )
        self.fp32, self.fp32_path = self.calibration.make(
            calibration.FP32_ORACLE_KIND
        )
        self.bf16, self.bf16_path = self.calibration.make(
            calibration.BF16_ORACLE_KIND
        )
        self.candidate, self.candidate_path = self.calibration.make(
            calibration.CANDIDATE_KIND
        )
        candidate_executable = (
            self.repository / calibration.NATIVE_EXECUTABLE_FILENAME
        )
        candidate_executable.write_bytes(
            fixture_elf() + b"\0" + b"\0".join(checker.CANDIDATE_BINARY_MARKERS) + b"\0"
        )
        self.candidate["candidate_execution"]["executable"]["sha256"] = sha256(
            candidate_executable
        )
        for manifest, path in (
            (self.fp32, self.fp32_path),
            (self.bf16, self.bf16_path),
            (self.candidate, self.candidate_path),
        ):
            sidecar = path.parent / manifest["sidecar"]["path"]
            write_safetensors(sidecar, self.calibration.sidecars[sidecar.name])
            manifest["sidecar"]["sha256"] = sha256(sidecar)
            path.write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        self.executable = self.repository / calibration.NATIVE_EXECUTABLE_FILENAME
        self.executable.chmod(0o755)

        mappings: list[checker._PureSafeTensorMapping] = []

        def loader(path: Path):
            mapping = checker._PureSafeTensorMapping(path)
            mappings.append(mapping)
            return mapping

        try:
            oracle_document = oracle_calibration.compare_hf_oracles(
                fp32_manifest=self.fp32,
                fp32_manifest_path=self.fp32_path,
                bf16_manifest=self.bf16,
                bf16_manifest_path=self.bf16_path,
                repo_root=self.repository,
                created_at=calibration_fixture_module.FIXED_TIME,
                sidecar_loader=loader,
            )
            self.oracle_report = root / "oracle-report.json"
            self.oracle_report.write_text(
                json.dumps(oracle_document, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            correctness = calibration.compare_calibrations(
                fp32_manifest=self.fp32,
                fp32_manifest_path=self.fp32_path,
                bf16_manifest=self.bf16,
                bf16_manifest_path=self.bf16_path,
                oracle_calibration_report=oracle_document,
                oracle_calibration_report_path=self.oracle_report,
                candidate_manifest=self.candidate,
                candidate_manifest_path=self.candidate_path,
                repo_root=self.repository,
                created_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
                sidecar_loader=loader,
            )
        finally:
            for mapping in reversed(mappings):
                mapping.close()
        self.correctness_report = root / "correctness-report.json"
        self.correctness_report.write_text(
            json.dumps(correctness, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self.candidate_source = root / "candidate-source.tar"
        self.oracle_source = root / "oracle-source.tar"
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={self.candidate_source}",
                self.calibration.candidate_revision,
            ],
            cwd=self.repository,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={self.oracle_source}",
                self.calibration.oracle_revision,
            ],
            cwd=self.repository,
            check=True,
        )
        self.raw = root / "native-correctness-evidence.tar"
        self.result = checker.build_raw_evidence(
            candidate_source_archive=self.candidate_source,
            oracle_source_archive=self.oracle_source,
            fp32_manifest=self.fp32_path,
            bf16_manifest=self.bf16_path,
            oracle_report=self.oracle_report,
            candidate_manifest=self.candidate_path,
            correctness_report=self.correctness_report,
            output=self.raw,
        )

    def payloads(self) -> dict[str, bytes]:
        with tarfile.open(self.raw, "r:") as archive:
            return {
                member.name: archive.extractfile(member).read()
                for member in archive.getmembers()
            }

    def rewrite(self, payloads: dict[str, bytes]) -> None:
        checksums = b"".join(
            f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n".encode(
                "ascii"
            )
            for name in checker.PAYLOAD_NAMES
        )
        values: dict[str, bytes] = {
            name: payloads[name] for name in checker.PAYLOAD_NAMES
        }
        values["SHA256SUMS"] = checksums
        checker._write_canonical_tar(self.raw, values, exclusive=False)


class NativeCorrectnessEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = NativeFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_closed_raw_sidecars_replay_without_torch(self) -> None:
        result = checker.replay_raw_evidence(
            self.fixture.raw,
            source_revision=self.fixture.calibration.candidate_revision,
            source_archive=self.fixture.candidate_source,
            correctness_report=self.fixture.correctness_report,
            candidate_executable=self.fixture.executable,
        )
        self.assertEqual(result.schema_version, checker.SCHEMA_VERSION)
        self.assertEqual(result.case_count, 31)
        self.assertEqual(result.failure_count, 0)
        self.assertEqual(result.candidate_executable_sha256, sha256(self.fixture.executable))

    def test_archive_is_byte_reproducible(self) -> None:
        second = self.fixture.root / "second.tar"
        checker.build_raw_evidence(
            candidate_source_archive=self.fixture.candidate_source,
            oracle_source_archive=self.fixture.oracle_source,
            fp32_manifest=self.fixture.fp32_path,
            bf16_manifest=self.fixture.bf16_path,
            oracle_report=self.fixture.oracle_report,
            candidate_manifest=self.fixture.candidate_path,
            correctness_report=self.fixture.correctness_report,
            output=second,
        )
        self.assertEqual(self.fixture.raw.read_bytes(), second.read_bytes())

    def test_arbitrary_legacy_tar_is_rejected(self) -> None:
        arbitrary = self.fixture.root / "arbitrary.tar"
        with tarfile.open(arbitrary, "w") as archive:
            member = tarfile.TarInfo("replay-summary.json")
            raw = b'{"status":"passed"}\n'
            member.size = len(raw)
            archive.addfile(member, checker._BytesReader(raw))
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError, "closed member inventory"
        ):
            checker.replay_raw_evidence(arbitrary)

    def test_self_declared_report_metrics_cannot_replace_tensor_replay(self) -> None:
        payloads = self.fixture.payloads()
        report = json.loads(payloads["correctness-report.json"])
        report["cases"][0]["variants"]["canonical-v1"]["numeric"][
            "first_layer_hidden"
        ]["metrics"]["max_abs"] = 0.0
        payloads["correctness-report.json"] = (
            json.dumps(report, sort_keys=True, indent=2) + "\n"
        ).encode()
        self.fixture.rewrite(payloads)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError, "comparator replay"
        ):
            checker.replay_raw_evidence(self.fixture.raw)

    def test_sidecar_tamper_is_rejected_after_internal_rehash(self) -> None:
        payloads = self.fixture.payloads()
        changed = bytearray(payloads["candidate-sidecar.safetensors"])
        changed[-1] ^= 0x40
        payloads["candidate-sidecar.safetensors"] = bytes(changed)
        self.fixture.rewrite(payloads)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "sidecar SHA-256 differs|comparator replay",
        ):
            checker.replay_raw_evidence(self.fixture.raw)

    def test_external_executable_must_equal_replayed_executable(self) -> None:
        other = self.fixture.root / "other-executable"
        other.write_bytes(b"not the candidate executable\n")
        other.chmod(0o755)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "candidate_executable: bytes differ",
        ):
            checker.replay_raw_evidence(
                self.fixture.raw, candidate_executable=other
            )

    def test_external_report_must_equal_replayed_report(self) -> None:
        other = self.fixture.root / "other-report.json"
        other.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "correctness_report: bytes differ",
        ):
            checker.replay_raw_evidence(
                self.fixture.raw, correctness_report=other
            )

    def test_arbitrary_candidate_executable_is_not_hash_only_evidence(self) -> None:
        payloads = self.fixture.payloads()
        payloads["candidate-executable"] = b"arbitrary executable bytes\n"
        self.fixture.rewrite(payloads)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "invalid Linux x86_64 native ELF",
        ):
            checker.replay_raw_evidence(self.fixture.raw)

    def test_candidate_capture_argv_rejects_unreviewed_arguments(self) -> None:
        payloads = self.fixture.payloads()
        manifest = json.loads(payloads["candidate-manifest.json"])
        manifest["candidate_execution"]["capture_argv"].extend(
            ["--unreviewed-flag", "unreviewed-value"]
        )
        payloads["candidate-manifest.json"] = (
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        ).encode()
        self.fixture.rewrite(payloads)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "exact ordered contract-v2 flag inventory",
        ):
            checker.replay_raw_evidence(self.fixture.raw)

    def test_candidate_source_revision_is_not_self_declared(self) -> None:
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "source_revision: does not match",
        ):
            checker.replay_raw_evidence(
                self.fixture.raw, source_revision="f" * 40
            )


if __name__ == "__main__":
    unittest.main()
