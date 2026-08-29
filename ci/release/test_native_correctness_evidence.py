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
from release_common import (  # noqa: E402
    ReleaseContractError,
    validate_binary,
    validate_calibration_binary,
)
from test_release import DEPENDENCIES, fixture_elf  # noqa: E402
from riley_reference import calibration, oracle_calibration  # noqa: E402


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
CALIBRATION_DEPENDENCIES = sorted({*DEPENDENCIES, "libnvidia-ml.so.1"})


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
            fixture_elf(CALIBRATION_DEPENDENCIES)
            + b"\0"
            + b"\0".join(checker.CANDIDATE_BINARY_MARKERS)
            + b"\0"
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
            self.oracle_document = oracle_document
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
        self.oracle_trust_anchors = checker.OracleTrustAnchors(
            source_revision=self.calibration.oracle_revision,
            fp32_manifest_sha256=sha256(self.fp32_path),
            fp32_sidecar_sha256=sha256(
                self.fp32_path.parent / self.fp32["sidecar"]["path"]
            ),
            bf16_manifest_sha256=sha256(self.bf16_path),
            bf16_sidecar_sha256=sha256(
                self.bf16_path.parent / self.bf16["sidecar"]["path"]
            ),
            historical_report_sha256=hashlib.sha256(
                b"historical torch oracle report fixture\n"
            ).hexdigest(),
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
            oracle_trust_anchors=self.oracle_trust_anchors,
        )

    def replay(self, **bindings: object) -> checker.NativeReplayResult:
        return checker.replay_raw_evidence(
            self.raw,
            oracle_trust_anchors=self.oracle_trust_anchors,
            **bindings,
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

    def rewrite_as_replay_valid_failure(self) -> dict[str, object]:
        case = self.candidate["cases"][0]
        tensor_key = case["variants"][checker.REQUIRED_VARIANTS[0]]["tensors"][
            "first_layer_hidden"
        ]["key"]
        candidate_sidecar = self.candidate_path.parent / self.candidate["sidecar"][
            "path"
        ]
        tensors = self.calibration.sidecars[candidate_sidecar.name]
        tensors[tensor_key].values[0] += 1_000.0
        write_safetensors(candidate_sidecar, tensors)
        self.candidate["sidecar"]["sha256"] = sha256(candidate_sidecar)
        self.candidate_path.write_text(
            json.dumps(self.candidate, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        mappings: list[checker._PureSafeTensorMapping] = []

        def loader(path: Path):
            mapping = checker._PureSafeTensorMapping(path)
            mappings.append(mapping)
            return mapping

        try:
            report = calibration.compare_calibrations(
                fp32_manifest=self.fp32,
                fp32_manifest_path=self.fp32_path,
                bf16_manifest=self.bf16,
                bf16_manifest_path=self.bf16_path,
                oracle_calibration_report=self.oracle_document,
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
        self.correctness_report.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        payloads = self.payloads()
        payloads["candidate-manifest.json"] = self.candidate_path.read_bytes()
        payloads["candidate-sidecar.safetensors"] = candidate_sidecar.read_bytes()
        payloads["correctness-report.json"] = self.correctness_report.read_bytes()
        self.rewrite(payloads)
        return report

    def rewrite_as_historical_v2(self) -> dict[str, object]:
        candidate, candidate_path = self.calibration.make(
            calibration.CANDIDATE_KIND,
            candidate_gate_id=calibration.ORACLE_MANIFEST_GATE_ID,
        )
        executable = self.repository / calibration.NATIVE_EXECUTABLE_FILENAME
        executable.write_bytes(
            fixture_elf(CALIBRATION_DEPENDENCIES)
            + b"\0"
            + b"\0".join(checker.CANDIDATE_BINARY_COMMON_MARKERS)
            + b"\0"
            + calibration.ORACLE_MANIFEST_GATE_ID.encode("ascii")
            + b"\0"
        )
        executable.chmod(0o755)
        candidate["candidate_execution"]["executable"]["sha256"] = sha256(
            executable
        )
        sidecar = candidate_path.parent / candidate["sidecar"]["path"]
        write_safetensors(sidecar, self.calibration.sidecars[sidecar.name])
        candidate["sidecar"]["sha256"] = sha256(sidecar)
        candidate_path.write_text(
            json.dumps(candidate, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        mappings: list[checker._PureSafeTensorMapping] = []

        def loader(path: Path):
            mapping = checker._PureSafeTensorMapping(path)
            mappings.append(mapping)
            return mapping

        try:
            report = calibration.compare_calibrations(
                fp32_manifest=self.fp32,
                fp32_manifest_path=self.fp32_path,
                bf16_manifest=self.bf16,
                bf16_manifest_path=self.bf16_path,
                oracle_calibration_report=self.oracle_document,
                oracle_calibration_report_path=self.oracle_report,
                candidate_manifest=candidate,
                candidate_manifest_path=candidate_path,
                repo_root=self.repository,
                created_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
                sidecar_loader=loader,
            )
        finally:
            for mapping in reversed(mappings):
                mapping.close()
        payloads = self.payloads()
        payloads["candidate-manifest.json"] = candidate_path.read_bytes()
        payloads["candidate-sidecar.safetensors"] = sidecar.read_bytes()
        payloads["candidate-executable"] = executable.read_bytes()
        payloads["correctness-report.json"] = (
            json.dumps(report, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        self.rewrite(payloads)
        return report


class NativeCorrectnessEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = NativeFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_calibration_binary_has_a_separate_required_nvml_abi(self) -> None:
        calibration_binary = fixture_elf(CALIBRATION_DEPENDENCIES)
        self.assertEqual(
            validate_calibration_binary(calibration_binary),
            CALIBRATION_DEPENDENCIES,
        )
        with self.assertRaisesRegex(
            ReleaseContractError, "unreviewed libraries: libnvidia-ml.so.1"
        ):
            validate_binary(calibration_binary)
        with self.assertRaisesRegex(
            ReleaseContractError, "missing reviewed CUDA/NVML libraries.*libnvidia-ml.so.1"
        ):
            validate_calibration_binary(fixture_elf())

    def test_closed_raw_sidecars_replay_without_torch(self) -> None:
        result = self.fixture.replay(
            source_revision=self.fixture.calibration.candidate_revision,
            source_archive=self.fixture.candidate_source,
            correctness_report=self.fixture.correctness_report,
            candidate_executable=self.fixture.executable,
        )
        self.assertEqual(result.schema_version, checker.SCHEMA_VERSION)
        self.assertEqual(result.case_count, 31)
        self.assertEqual(result.failure_count, 0)
        self.assertEqual(result.candidate_executable_sha256, sha256(self.fixture.executable))
        self.assertEqual(
            result.source_archive_byte_length,
            self.fixture.candidate_source.stat().st_size,
        )
        self.assertEqual(
            result.correctness_report_byte_length,
            self.fixture.correctness_report.stat().st_size,
        )
        self.assertEqual(
            result.candidate_executable_byte_length,
            self.fixture.executable.stat().st_size,
        )

    def test_historical_v2_two_variant_raw_bundle_still_replays(self) -> None:
        report = self.fixture.rewrite_as_historical_v2()
        self.assertEqual(report["gate_id"], calibration.ORACLE_MANIFEST_GATE_ID)
        self.assertEqual(report["summary"]["candidate_variant_count"], 2)
        result = self.fixture.replay()
        self.assertEqual(result.case_count, 31)
        self.assertEqual(result.failure_count, 0)

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
            oracle_trust_anchors=self.fixture.oracle_trust_anchors,
        )
        self.assertEqual(self.fixture.raw.read_bytes(), second.read_bytes())

    def test_rejected_report_leaves_no_create_only_archive_but_remains_replayable(self) -> None:
        report = self.fixture.rewrite_as_replay_valid_failure()
        self.assertEqual(report["status"], "fail")
        diagnostic = self.fixture.replay()
        self.assertGreater(diagnostic.failure_count, 0)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "status: must be pass for evidence packaging",
        ):
            self.fixture.replay(require_passing_report=True)

        rejected = self.fixture.root / "rejected.tar"
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "status: must be pass for evidence packaging",
        ):
            checker.build_raw_evidence(
                candidate_source_archive=self.fixture.candidate_source,
                oracle_source_archive=self.fixture.oracle_source,
                fp32_manifest=self.fixture.fp32_path,
                bf16_manifest=self.fixture.bf16_path,
                oracle_report=self.fixture.oracle_report,
                candidate_manifest=self.fixture.candidate_path,
                correctness_report=self.fixture.correctness_report,
                output=rejected,
                oracle_trust_anchors=self.fixture.oracle_trust_anchors,
            )
        self.assertFalse(rejected.exists())

    def test_packaging_requires_every_case_variant_and_tensor_pass(self) -> None:
        report = json.loads(self.fixture.correctness_report.read_text(encoding="utf-8"))
        report["cases"][0]["variants"][checker.REQUIRED_VARIANTS[0]]["numeric"][
            calibration.TENSOR_NAMES[0]
        ]["pass"] = False
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            r"cases\[0\].*\.pass: must be true",
        ):
            checker._require_passing_native_e0_report(report)

    def test_historical_report_hash_is_provenance_not_portable_report_identity(self) -> None:
        self.assertEqual(
            checker.PRODUCTION_ORACLE_TRUST_ANCHORS.historical_report_sha256,
            "1fd064d780868ed76202b9adbd773f2ef76cc54a35551a92145f882d779871ea",
        )
        self.assertNotEqual(
            self.fixture.oracle_trust_anchors.historical_report_sha256,
            sha256(self.fixture.oracle_report),
        )
        result = self.fixture.replay()
        self.assertEqual(result.failure_count, 0)

    def test_portable_oracle_report_requires_exact_raw_sidecar_replay(self) -> None:
        payloads = self.fixture.payloads()
        report = json.loads(payloads["oracle-report.json"])
        report["cases"][0]["numeric"]["first_layer_hidden"]["metrics"][
            "max_abs"
        ] += 0.125
        payloads["oracle-report.json"] = (
            json.dumps(report, sort_keys=True, indent=2) + "\n"
        ).encode()
        self.fixture.rewrite(payloads)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "comparator replay failed",
        ):
            self.fixture.replay()

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
            self.fixture.replay()

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
            self.fixture.replay()

    def test_external_executable_must_equal_replayed_executable(self) -> None:
        other = self.fixture.root / "other-executable"
        other.write_bytes(b"not the candidate executable\n")
        other.chmod(0o755)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "candidate_executable: bytes differ",
        ):
            self.fixture.replay(candidate_executable=other)

    def test_external_report_must_equal_replayed_report(self) -> None:
        other = self.fixture.root / "other-report.json"
        other.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "correctness_report: bytes differ",
        ):
            self.fixture.replay(correctness_report=other)

    def test_arbitrary_candidate_executable_is_not_hash_only_evidence(self) -> None:
        payloads = self.fixture.payloads()
        payloads["candidate-executable"] = b"arbitrary executable bytes\n"
        self.fixture.rewrite(payloads)
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "invalid Linux x86_64 native ELF",
        ):
            self.fixture.replay()

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
            "exact ordered smollm2-fp32-bf16-native-e0-v3 flag inventory",
        ):
            self.fixture.replay()

    def test_candidate_source_revision_is_not_self_declared(self) -> None:
        with self.assertRaisesRegex(
            checker.NativeCorrectnessEvidenceError,
            "source_revision: does not match",
        ):
            self.fixture.replay(source_revision="f" * 40)


if __name__ == "__main__":
    unittest.main()
