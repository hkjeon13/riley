#!/usr/bin/env python3
"""CPU-only adversarial tests for optimizer raw-evidence replay."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from check_optimization_evidence import (  # noqa: E402
    CHECKSUM_FILE,
    EXPECTED_COMMANDS,
    EXPECTED_TOKENS,
    GATE_ID,
    INPUT_FILES,
    LOG_FILES,
    RAW_FILES,
    RECEIPT_FILE,
    RECEIPT_VERSION,
    REPORT_FILE,
    TEST_BINARIES,
    OptimizationEvidenceError,
    _json,
    load_raw_evidence_archive,
    produce,
    replay_raw_evidence,
)
from release_common import canonical_json_bytes  # noqa: E402
from test_release import fixture_elf  # noqa: E402


REVISION = "1a2b3c4d5e6f78901234567890abcdef12345678"
SOURCE_SHA256 = hashlib.sha256(b"source.tar").hexdigest()
BUILD_IMAGE_ID = "sha256:" + hashlib.sha256(b"builder image").hexdigest()


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def executable_fixture_elf() -> bytes:
    binary = bytearray(fixture_elf())
    struct.pack_into("<Q", binary, 24, 0x400040)
    return bytes(binary)


SUMMARY = (
    "test result: ok. {passed} passed; 0 failed; {ignored} ignored; "
    "0 measured; {filtered} filtered out; finished in 0.01s\n"
)


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.report_path = root / "report.json"
        self.profile_binary = root / "rustinfer-profile"
        self.counter = 0

        self.profile_binary.write_bytes(
            executable_fixture_elf()
            + b"\0pr15-iteration-command-batch-exact-v1\0per-operation\0iteration-batch\0"
        )
        self.profile_binary.chmod(0o755)
        binary_markers = {
            "host-runtime-gpu-test": (
                "command_batch_proxy_is_one_shot_and_drop_restores_stream_use",
                "pr16-command-batch-lifecycle",
            ),
            "primitives-gpu-test": (
                "command_batch_releases_multi_primitive_resource_ledger_after_validation_error",
                "pr16-command-batch-resource-ledger",
            ),
            "llama-batch-gpu-test": (
                "iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly",
                "pr15-execution-completion-parity",
            ),
        }
        for name, markers in binary_markers.items():
            (self.evidence / name).write_bytes(
                executable_fixture_elf()
                + b"\0"
                + b"\0".join(marker.encode() for marker in markers)
                + b"\0"
            )

        self.logs = self._logs()
        for test_id, name in LOG_FILES.items():
            (self.evidence / name).write_bytes(self.logs[test_id])
        self.report = self._report()
        self._write_report()
        self.receipt = self._receipt()
        self._write_receipt()

    def _logs(self) -> dict[str, bytes]:
        compile_log = (
            "rustc 1.85.0 (fixture)\n"
            "cargo 1.85.0 (fixture)\n"
            "Cuda compilation tools, release 12.8, V12.8.93\n"
            "Finished `release` profile [optimized + debuginfo]\n"
            "test native_symbols_link_without_device_initialization ... ok\n"
            + SUMMARY.format(passed=1, ignored=0, filtered=0)
            + "rustinfer 0.1.0 (server=true, cuda=true, cuda_abi=1)\n"
            "error: rustinfer-cuda native build failed: "
            "CUDAToolkit_ROOT=/definitely/missing/rustinfer-cuda is not a directory\n"
            "artifact=target/release/rustinfer\n"
            "artifact=target/release/rustinfer-profile\n"
            "Python-free CUDA production/profile compile, C ABI link, tensor memory, "
            "version, and dependency smoke passed\n"
        ).encode()
        workspace = (
            "Finished `test` profile [unoptimized + debuginfo]\n"
            "test command_batch_proxy_is_one_shot_and_drop_restores_stream_use ... ignored\n"
            "test command_batch_releases_multi_primitive_resource_ledger_after_validation_error ... ignored\n"
            "test iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly ... ignored\n"
            + SUMMARY.format(passed=1, ignored=0, filtered=0) * 20
        ).encode()
        lifecycle = (
            "Running tests/host_runtime_gpu.rs (target/debug/deps/host_runtime_gpu-fixture)\n"
            "running 1 test\n"
            "test command_batch_proxy_is_one_shot_and_drop_restores_stream_use ... "
            "pr16-command-batch-lifecycle schema_version=1 one_shot_finish=true "
            "drop_restores_stream=true status=passed\n"
            "ok\n"
            + SUMMARY.format(passed=1, ignored=0, filtered=7)
        ).encode()
        ledger = (
            "Running tests/primitives_gpu.rs (target/debug/deps/primitives_gpu-fixture)\n"
            "running 1 test\n"
            "test command_batch_releases_multi_primitive_resource_ledger_after_validation_error ... "
            "pr16-command-batch-resource-ledger schema_version=1 "
            "validation_fail_closed=true queued_chain_raw_byte_mismatches=0 "
            "cuda_live_allocation_delta=0 stream_reuse_after_finish=true "
            "owner_close_live_allocation_count=0 status=passed\n"
            "ok\n"
            + SUMMARY.format(passed=1, ignored=0, filtered=5)
        ).encode()
        token_text = ", ".join(str(value) for value in EXPECTED_TOKENS)
        parity = (
            "Running tests/llama_batch_gpu.rs (target/debug/deps/llama_batch_gpu-fixture)\n"
            "running 1 test\n"
            "test iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly ... "
            "pr15-execution-completion-parity schema_version=1 decode_steps=16 "
            "committed_iterations=16 raw_logit_mismatches=0 token_id_mismatches=0 "
            "cuda_live_allocation_delta=0 owner_close_live_allocation_count=0 "
            f"generated_token_ids=[{token_text}] status=passed\n"
            "ok\n"
            + SUMMARY.format(passed=1, ignored=0, filtered=6)
        ).encode()
        return {
            "cuda-compile-only": compile_log,
            "workspace-all-features-all-targets": workspace,
            "command-batch-lifecycle": lifecycle,
            "command-batch-resource-ledger": ledger,
            "smollm2-multi-step-greedy-exact": parity,
        }

    def _report(self) -> dict[str, object]:
        tests: list[dict[str, object]] = [
            {
                "id": "cuda-compile-only",
                "result": "passed",
                "log_sha256": digest(self.logs["cuda-compile-only"]),
            },
            {
                "id": "workspace-all-features-all-targets",
                "result": "passed",
                "log_sha256": digest(self.logs["workspace-all-features-all-targets"]),
            },
            {
                "id": "command-batch-lifecycle",
                "result": "passed",
                "one_shot_finish": True,
                "drop_restores_stream": True,
                "log_sha256": digest(self.logs["command-batch-lifecycle"]),
            },
            {
                "id": "command-batch-resource-ledger",
                "result": "passed",
                "validation_fail_closed": True,
                "queued_chain_raw_byte_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "stream_reuse_after_finish": True,
                "owner_close_live_allocation_count": 0,
                "log_sha256": digest(self.logs["command-batch-resource-ledger"]),
            },
            {
                "id": "smollm2-multi-step-greedy-exact",
                "result": "passed",
                "decode_steps": 16,
                "committed_iterations": 16,
                "raw_logit_mismatches": 0,
                "generated_token_ids": EXPECTED_TOKENS,
                "token_id_mismatches": 0,
                "cuda_live_allocation_delta": 0,
                "owner_close_live_allocation_count": 0,
                "log_sha256": digest(self.logs["smollm2-multi-step-greedy-exact"]),
            },
        ]
        return {
            "schema_version": 1,
            "gate_id": GATE_ID,
            "recorded_at_utc": "2026-08-26T00:00:00Z",
            "status": "passed",
            "semantic_class": "E0",
            "source": {
                "git_commit": REVISION,
                "git_dirty": False,
                "archive_sha256": SOURCE_SHA256,
            },
            "build": {
                "container_image_sha256": BUILD_IMAGE_ID.removeprefix("sha256:"),
                "network": "none",
                "cargo_locked": True,
                "cargo_offline": True,
                "rustc": "1.85.0",
                "cuda_toolkit": "12.8.93",
                "cuda_architecture": "89",
            },
            "gpu": {
                "model": "NVIDIA GeForce RTX 4090",
                "uuid": "GPU-fixture",
                "pci_bus_id": "00000000:01:00.0",
                "compute_capability": "8.9",
                "vram_mib": 24564,
                "driver_version": "580.173.02",
            },
            "model": {
                "model_id": "HuggingFaceTB/SmolLM2-135M",
                "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "dtype": "bf16",
                "manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
                "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
                "tokenizer_sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            },
            "implementations": {
                "baseline": "per-operation",
                "candidate": "iteration-batch",
                "residual_rmsnorm": "separate",
                "rollback": "--execution-completion per-operation",
            },
            "tests": tests,
        }

    def _receipt(self) -> dict[str, object]:
        commands = []
        for command_id, argv in EXPECTED_COMMANDS.items():
            environment = {
                "CARGO_NET_OFFLINE": "true",
                "CARGO_TERM_COLOR": "never",
                "RUSTINFER_CUDA_ARCHITECTURES": "89",
            }
            if command_id == "smollm2-multi-step-greedy-exact":
                environment["RUSTINFER_REAL_CHECKPOINT"] = "/model"
            commands.append(
                {
                    "id": command_id,
                    "argv": argv,
                    "environment": environment,
                    "exit_code": 0,
                    "log": LOG_FILES[command_id],
                    "test_binary": TEST_BINARIES.get(command_id),
                }
            )
        return {
            "schema_version": RECEIPT_VERSION,
            "status": "completed",
            "source": copy.deepcopy(self.report["source"]),
            "build": copy.deepcopy(self.report["build"]),
            "gpu": copy.deepcopy(self.report["gpu"]),
            "model": copy.deepcopy(self.report["model"]),
            "profile_binary_sha256": digest(self.profile_binary.read_bytes()),
            "subjects": {
                name: {
                    "sha256": digest((self.evidence / name).read_bytes()),
                    "size": (self.evidence / name).stat().st_size,
                }
                for name in sorted(TEST_BINARIES.values())
            },
            "commands": commands,
        }

    def _write_report(self) -> None:
        self.report_path.write_bytes(canonical_json_bytes(self.report))

    def _write_receipt(self) -> None:
        (self.evidence / RECEIPT_FILE).write_bytes(canonical_json_bytes(self.receipt))

    def refresh_log(self, test_id: str, contents: bytes) -> None:
        self.logs[test_id] = contents
        (self.evidence / LOG_FILES[test_id]).write_bytes(contents)
        for test in self.report["tests"]:  # type: ignore[index]
            if test["id"] == test_id:
                test["log_sha256"] = digest(contents)
        self._write_report()

    def produce(self) -> tuple[dict[str, object], Path]:
        self.counter += 1
        raw = self.root / f"raw-{self.counter}.tar"
        result = produce(
            self.evidence,
            report=self.report_path,
            source_revision=REVISION,
            source_archive_sha256=SOURCE_SHA256,
            build_image_id=BUILD_IMAGE_ID,
            profile_binary=self.profile_binary,
            raw_evidence=raw,
        )
        return result, raw

    def replay(self, raw: Path, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "report": self.report_path,
            "source_revision": REVISION,
            "source_archive_sha256": SOURCE_SHA256,
            "build_image_id": BUILD_IMAGE_ID,
            "profile_binary": self.profile_binary,
        }
        arguments.update(overrides)
        return replay_raw_evidence(raw, **arguments)  # type: ignore[arg-type]


class OptimizationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_evidence_is_deterministic_and_replays(self) -> None:
        first, first_raw = self.fixture.produce()
        second, second_raw = self.fixture.produce()
        self.assertEqual(first, second)
        self.assertEqual(first_raw.read_bytes(), second_raw.read_bytes())
        self.assertEqual(self.fixture.replay(first_raw), first)
        files, raw_sha = load_raw_evidence_archive(first_raw)
        self.assertEqual(set(files), RAW_FILES)
        self.assertEqual(raw_sha, digest(first_raw.read_bytes()))
        self.assertEqual(first["profile_binary_sha256"], digest(self.fixture.profile_binary.read_bytes()))

    def test_overflowing_json_float_is_rejected(self) -> None:
        with self.assertRaisesRegex(OptimizationEvidenceError, "non-finite"):
            _json(b'{"value":1e309}', "fixture")

    def test_empty_synthetic_logs_are_rejected(self) -> None:
        self.fixture.refresh_log("command-batch-lifecycle", b"\n")
        with self.assertRaisesRegex(OptimizationEvidenceError, "marker|Cargo test summary"):
            self.fixture.produce()

    def test_missing_semantic_marker_is_rejected_even_when_hash_matches(self) -> None:
        contents = self.fixture.logs["command-batch-resource-ledger"].replace(
            b"validation_fail_closed=true", b"validation_fail_closed=false"
        )
        self.fixture.refresh_log("command-batch-resource-ledger", contents)
        with self.assertRaisesRegex(OptimizationEvidenceError, "must contain exactly one"):
            self.fixture.produce()

    def test_parity_mismatch_is_derived_from_raw_log(self) -> None:
        contents = self.fixture.logs["smollm2-multi-step-greedy-exact"].replace(
            b"raw_logit_mismatches=0", b"raw_logit_mismatches=1"
        )
        self.fixture.refresh_log("smollm2-multi-step-greedy-exact", contents)
        with self.assertRaisesRegex(OptimizationEvidenceError, "reviewed E0 result"):
            self.fixture.produce()

    def test_failed_workspace_summary_is_rejected(self) -> None:
        contents = self.fixture.logs["workspace-all-features-all-targets"].replace(
            b"0 failed", b"1 failed", 1
        )
        self.fixture.refresh_log("workspace-all-features-all-targets", contents)
        with self.assertRaisesRegex(OptimizationEvidenceError, "failed Cargo test summary"):
            self.fixture.produce()

    def test_test_binary_bytes_must_match_subject_receipt(self) -> None:
        path = self.fixture.evidence / "host-runtime-gpu-test"
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(OptimizationEvidenceError, "subject differs"):
            self.fixture.produce()

    def test_test_binary_must_be_elf_with_embedded_test_marker(self) -> None:
        path = self.fixture.evidence / "host-runtime-gpu-test"
        path.write_bytes(b"not an ELF but has command_batch_proxy_is_one_shot_and_drop_restores_stream_use")
        self.fixture.receipt = self.fixture._receipt()
        self.fixture._write_receipt()
        with self.assertRaisesRegex(OptimizationEvidenceError, "valid Linux x86_64"):
            self.fixture.produce()

    def test_profile_binary_substitution_is_rejected(self) -> None:
        _, raw = self.fixture.produce()
        substitute = self.fixture.root / "substitute"
        substitute.write_bytes(
            executable_fixture_elf()
            + b"\0pr15-iteration-command-batch-exact-v1\0per-operation\0iteration-batch\0substitute"
        )
        with self.assertRaisesRegex(OptimizationEvidenceError, "profile binary differs"):
            self.fixture.replay(raw, profile_binary=substitute)

    def test_command_exit_receipt_must_be_zero(self) -> None:
        self.fixture.receipt["commands"][2]["exit_code"] = 1  # type: ignore[index]
        self.fixture._write_receipt()
        with self.assertRaisesRegex(OptimizationEvidenceError, "reviewed invocation"):
            self.fixture.produce()

    def test_command_argv_receipt_is_closed(self) -> None:
        self.fixture.receipt["commands"][0]["argv"] = ["true"]  # type: ignore[index]
        self.fixture._write_receipt()
        with self.assertRaisesRegex(OptimizationEvidenceError, "reviewed invocation"):
            self.fixture.produce()

    def test_command_order_is_closed(self) -> None:
        commands = self.fixture.receipt["commands"]  # type: ignore[index]
        commands[0], commands[1] = commands[1], commands[0]
        self.fixture._write_receipt()
        with self.assertRaisesRegex(OptimizationEvidenceError, "execution order"):
            self.fixture.produce()

    def test_zero_entry_test_binary_is_rejected(self) -> None:
        path = self.fixture.evidence / "host-runtime-gpu-test"
        path.write_bytes(
            fixture_elf()
            + b"\0command_batch_proxy_is_one_shot_and_drop_restores_stream_use"
            + b"\0pr16-command-batch-lifecycle\0"
        )
        self.fixture.receipt = self.fixture._receipt()
        self.fixture._write_receipt()
        with self.assertRaisesRegex(OptimizationEvidenceError, "executable Linux x86-64"):
            self.fixture.produce()

    def test_source_and_build_image_are_external_bindings(self) -> None:
        _, raw = self.fixture.produce()
        with self.assertRaisesRegex(OptimizationEvidenceError, "source"):
            self.fixture.replay(raw, source_archive_sha256="f" * 64)
        with self.assertRaisesRegex(OptimizationEvidenceError, "build image"):
            self.fixture.replay(raw, build_image_id="sha256:" + "e" * 64)

    def test_raw_payload_tampering_breaks_internal_checksums(self) -> None:
        _, raw = self.fixture.produce()
        contents = raw.read_bytes()
        marker = b"one_shot_finish=true"
        self.assertIn(marker, contents)
        raw.write_bytes(contents.replace(marker, b"one_shot_finish=fals", 1))
        with self.assertRaisesRegex(OptimizationEvidenceError, "digest mismatch"):
            self.fixture.replay(raw)

    def test_trailing_tar_record_is_rejected(self) -> None:
        _, raw = self.fixture.produce()
        raw.write_bytes(raw.read_bytes() + b"\0" * 10240)
        with self.assertRaisesRegex(OptimizationEvidenceError, "end-of-archive padding"):
            load_raw_evidence_archive(raw)

    def test_noncanonical_metadata_is_rejected(self) -> None:
        _, raw = self.fixture.produce()
        files, _ = load_raw_evidence_archive(raw)
        changed = self.fixture.root / "changed.tar"
        with tarfile.open(changed, "w", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(files):
                member = tarfile.TarInfo(name)
                member.size = len(files[name])
                member.mode = 0o755 if name in TEST_BINARIES.values() else 0o644
                member.uid = 1
                member.gid = 0
                member.uname = "root"
                member.gname = "root"
                member.mtime = 0
                archive.addfile(member, io.BytesIO(files[name]))
        with self.assertRaisesRegex(OptimizationEvidenceError, "non-canonical metadata"):
            load_raw_evidence_archive(changed)

    def test_evidence_directory_rejects_extra_and_symlink(self) -> None:
        (self.fixture.evidence / "extra").write_text("pass")
        with self.assertRaisesRegex(OptimizationEvidenceError, "closed input inventory"):
            self.fixture.produce()
        (self.fixture.evidence / "extra").unlink()
        target = self.fixture.evidence / RECEIPT_FILE
        target.unlink()
        target.symlink_to(LOG_FILES["cuda-compile-only"])
        with self.assertRaisesRegex(OptimizationEvidenceError, "regular file"):
            self.fixture.produce()

    def test_submitted_report_must_equal_embedded_canonical_report(self) -> None:
        _, raw = self.fixture.produce()
        self.fixture.report["recorded_at_utc"] = "2026-08-26T00:00:01Z"
        self.fixture._write_report()
        with self.assertRaisesRegex(OptimizationEvidenceError, "exact submitted report"):
            self.fixture.replay(raw)


if __name__ == "__main__":
    unittest.main()
