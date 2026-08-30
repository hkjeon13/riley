from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


writer = _load("write_hf_golden_oracle_receipt_test", "write_hf_golden_oracle_receipt.py")
checker = _load("check_hf_golden_oracle_receipts_test", "check_hf_golden_oracle_receipts.py")


MODEL_TREE_SHA256 = hashlib.sha256(b"reviewed model tree").hexdigest()
TEST_TEXT = "fixture completion"
TEST_TEXT_SHA256 = hashlib.sha256(TEST_TEXT.encode("utf-8")).hexdigest()


def _receipt(nonce_character: str, pid: int, start_ticks: int) -> dict:
    nonce = nonce_character * 64
    observation = {
        "cache_mode": "on",
        "generated_text": TEST_TEXT,
        "generated_text_utf8_sha256": TEST_TEXT_SHA256,
        "generated_token_ids": list(checker.EXPECTED_TOKEN_IDS),
        "generated_token_ids_u32le_sha256": checker.EXPECTED_TOKEN_IDS_U32LE_SHA256,
    }
    return {
        "gate_id": checker.GATE_ID,
        "invocation": {
            "normalized_argv": checker._expected_normalized_argv(
                MODEL_TREE_SHA256, checker.DEPENDENCY_LOCK_SHA256
            )
        },
        "model": {
            "config_sha256": checker.MODEL_CONFIG_SHA256,
            "file_count": 9,
            "id": checker.MODEL_ID,
            "revision": checker.MODEL_REVISION,
            "tokenizer_aggregate_sha256": checker.TOKENIZER_AGGREGATE_SHA256,
            "tokenizer_files_sha256": dict(checker.TOKENIZER_FILES_SHA256),
            "tree_sha256": MODEL_TREE_SHA256,
            "weights_sha256": checker.MODEL_WEIGHTS_SHA256,
        },
        "observations": [
            observation,
            {**copy.deepcopy(observation), "cache_mode": "off"},
        ],
        "oracle": {
            "dependency_lock": {
                "name": "uv.lock",
                "sha256": checker.DEPENDENCY_LOCK_SHA256,
            },
            "implementation_id": "hf-transformers-eager",
            "provenance_kind": "dependency-lock",
        },
        "process": {
            "boot_id": "12345678-1234-1234-1234-123456789abc",
            "pid": pid,
            "start_time_ticks": start_ticks,
        },
        "prompt": {
            "text": checker.PROMPT,
            "utf8_base64": "RXhwbGFpbiB3aHkgZGV0ZXJtaW5pc3RpYyBiZW5jaG1hcmtzIG5lZWQgaW1tdXRhYmxlIGlucHV0cy4=",
            "utf8_sha256": checker.PROMPT_SHA256,
        },
        "run_id": f"hf-golden-oracle-{nonce}",
        "run_nonce": nonce,
        "runtime": {
            "cuda_runtime_version": checker.CUDA_RUNTIME_VERSION,
            "dependencies": {
                "tokenizers": checker.TOKENIZERS_VERSION,
                "torch": checker.TORCH_VERSION,
                "transformers": checker.TRANSFORMERS_VERSION,
            },
            "driver_version": checker.DRIVER_VERSION,
            "environment": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_VISIBLE_DEVICES": checker.GPU_UUID,
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            },
            "gpu": {
                "compute_capability": checker.GPU_COMPUTE_CAPABILITY,
                "name": checker.GPU_NAME,
                "uuid": checker.GPU_UUID,
            },
            "python": {
                "executable": "/opt/cpython-3.13.15/bin/python3.13",
                "executable_sha256": checker.PYTHON_EXECUTABLE_SHA256,
                "ignore_environment": True,
                "isolated": True,
                "no_user_site": True,
                "platform_machine": "x86_64",
                "platform_system": "linux",
                "version": checker.PYTHON_VERSION,
            },
        },
        "schema_version": checker.RECEIPT_SCHEMA,
        "settings": {
            "add_special_tokens": True,
            "attention_implementation": "eager",
            "cache_modes": ["on", "off"],
            "clean_up_tokenization_spaces": False,
            "device": "cuda:0",
            "do_sample": False,
            "dtype": "bfloat16",
            "local_files_only": True,
            "max_new_tokens": 8,
            "num_beams": 1,
            "skip_special_tokens": True,
            "trust_remote_code": False,
        },
        "status": "passed",
        "timing": {
            "ended_at_utc": "2026-08-27T01:02:04.000000Z",
            "started_at_utc": "2026-08-27T01:02:03.000000Z",
        },
    }


class ReceiptFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.receipts = [root / "receipt-1.json", root / "receipt-2.json"]
        self.documents = [_receipt("a", 101, 1001), _receipt("b", 202, 2002)]
        self.write(0)
        self.write(1)

    def write(self, index: int) -> None:
        self.receipts[index].write_bytes(checker._canonical_json_bytes(self.documents[index]))

    def approve(self, output: Path | None = None):
        return checker.approve(
            receipts=self.receipts,
            output=output or self.root / "approval.json",
            expected_model_tree_sha256=MODEL_TREE_SHA256,
            expected_dependency_lock_sha256=checker.DEPENDENCY_LOCK_SHA256,
            expected_python_executable_sha256=checker.PYTHON_EXECUTABLE_SHA256,
            expected_gpu_uuid=checker.GPU_UUID,
        )


class HfGoldenOracleCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = ReceiptFixture(self.root)
        self.text_patch = mock.patch.object(
            checker, "EXPECTED_TEXT_UTF8_SHA256", TEST_TEXT_SHA256
        )
        self.text_patch.start()

    def tearDown(self) -> None:
        self.text_patch.stop()
        self.temporary.cleanup()

    def test_two_fresh_receipts_create_closed_approval(self) -> None:
        output = self.root / "approval.json"
        approval = self.fixture.approve(output)
        self.assertEqual(approval["status"], "approved")
        self.assertEqual(len(approval["receipt_bindings"]), 2)
        self.assertEqual(len(approval["cache_paths_verified"]), 4)
        self.assertEqual(
            approval["expected_generation"]["generated_token_ids"],
            checker.EXPECTED_TOKEN_IDS,
        )
        self.assertEqual(
            output.read_bytes(), checker._canonical_json_bytes(approval)
        )
        for index, binding in enumerate(approval["receipt_bindings"]):
            self.assertEqual(
                binding["receipt_sha256"],
                hashlib.sha256(self.fixture.receipts[index].read_bytes()).hexdigest(),
            )

    def test_missing_receipt_fails_closed(self) -> None:
        self.fixture.receipts[1] = self.root / "missing.json"
        with self.assertRaisesRegex(checker.OracleApprovalError, "cannot stat"):
            self.fixture.approve()

    def test_tampered_token_id_fails(self) -> None:
        self.fixture.documents[1]["observations"][1]["generated_token_ids"][0] += 1
        self.fixture.write(1)
        with self.assertRaisesRegex(checker.OracleApprovalError, "reviewed IDs"):
            self.fixture.approve()

    def test_spliced_model_anchor_fails(self) -> None:
        self.fixture.documents[1]["model"]["tree_sha256"] = "f" * 64
        self.fixture.write(1)
        with self.assertRaisesRegex(checker.OracleApprovalError, "reviewer anchor"):
            self.fixture.approve()

    def test_spliced_dependency_lock_fails(self) -> None:
        self.fixture.documents[1]["oracle"]["dependency_lock"]["sha256"] = "f" * 64
        self.fixture.write(1)
        with self.assertRaisesRegex(checker.OracleApprovalError, "reviewer anchor"):
            self.fixture.approve()

    def test_same_run_nonce_fails_fresh_process_gate(self) -> None:
        self.fixture.documents[1]["run_nonce"] = "a" * 64
        self.fixture.documents[1]["run_id"] = "hf-golden-oracle-" + "a" * 64
        self.fixture.write(1)
        with self.assertRaisesRegex(checker.OracleApprovalError, "fresh run IDs"):
            self.fixture.approve()

    def test_same_process_identity_fails(self) -> None:
        self.fixture.documents[1]["process"] = copy.deepcopy(
            self.fixture.documents[0]["process"]
        )
        self.fixture.write(1)
        with self.assertRaisesRegex(checker.OracleApprovalError, "fresh process identities"):
            self.fixture.approve()

    def test_duplicate_json_key_is_rejected(self) -> None:
        raw = self.fixture.receipts[0].read_text(encoding="utf-8")
        raw = raw.replace('"gate_id":', '"gate_id":"spliced","gate_id":', 1)
        self.fixture.receipts[0].write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(checker.OracleApprovalError, "duplicate object key"):
            self.fixture.approve()

    def test_nonfinite_json_number_is_rejected(self) -> None:
        raw = self.fixture.receipts[0].read_text(encoding="utf-8")
        raw = raw.replace('"pid":101', '"pid":NaN', 1)
        self.fixture.receipts[0].write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(checker.OracleApprovalError, "non-finite"):
            self.fixture.approve()

    def test_unsafe_python_path_is_rejected(self) -> None:
        self.fixture.documents[0]["runtime"]["python"]["executable"] = "/opt/../bin/python"
        self.fixture.write(0)
        with self.assertRaisesRegex(checker.OracleApprovalError, "invalid syntax|normalized safe"):
            self.fixture.approve()

    def test_noncanonical_receipt_bytes_are_rejected(self) -> None:
        document = self.fixture.documents[0]
        self.fixture.receipts[0].write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(checker.OracleApprovalError, "canonical JSON"):
            self.fixture.approve()

    def test_symlink_receipt_is_rejected(self) -> None:
        link = self.root / "receipt-link.json"
        link.symlink_to(self.fixture.receipts[0])
        self.fixture.receipts[0] = link
        with self.assertRaisesRegex(checker.OracleApprovalError, "non-symlink"):
            self.fixture.approve()

    def test_approval_overwrite_is_rejected(self) -> None:
        output = self.root / "approval.json"
        self.fixture.approve(output)
        with self.assertRaisesRegex(checker.OracleApprovalError, "already exists"):
            self.fixture.approve(output)

    def test_approval_writer_requires_nofollow_before_creating_output(self) -> None:
        output = self.root / "approval.json"
        with mock.patch.object(checker.os, "O_NOFOLLOW", 0):
            with self.assertRaisesRegex(checker.OracleApprovalError, "os\\.O_NOFOLLOW"):
                self.fixture.approve(output)
        self.assertFalse(output.exists())
        self.assertNotIn(
            'hasattr(os, "O_NOFOLLOW")',
            Path(checker.__file__).read_text(encoding="utf-8"),
        )

    def test_exactly_two_receipts_are_required(self) -> None:
        with self.assertRaisesRegex(checker.OracleApprovalError, "exactly twice"):
            checker.approve(
                receipts=self.fixture.receipts[:1],
                output=self.root / "approval.json",
                expected_model_tree_sha256=MODEL_TREE_SHA256,
                expected_dependency_lock_sha256=checker.DEPENDENCY_LOCK_SHA256,
                expected_python_executable_sha256=checker.PYTHON_EXECUTABLE_SHA256,
                expected_gpu_uuid=checker.GPU_UUID,
            )


class HfGoldenOracleWriterUnitTests(unittest.TestCase):
    class FakeBackend:
        metadata = {}

        def __init__(self, *, mismatch: bool = False) -> None:
            self.calls: list[bool] = []
            self.mismatch = mismatch

        def generate(self, *, use_cache: bool):
            self.calls.append(use_cache)
            token_ids = list(writer.EXPECTED_TOKEN_IDS)
            if self.mismatch and not use_cache:
                token_ids[-1] += 1
            return token_ids, TEST_TEXT

    def test_fake_backend_exercises_cache_on_then_off(self) -> None:
        backend = self.FakeBackend()
        with mock.patch.object(writer, "EXPECTED_TEXT_UTF8_SHA256", TEST_TEXT_SHA256):
            observations = writer._run_cache_pair(backend)
        self.assertEqual(backend.calls, [True, False])
        self.assertEqual([row["cache_mode"] for row in observations], ["on", "off"])

    def test_fake_backend_cache_mismatch_fails(self) -> None:
        backend = self.FakeBackend(mismatch=True)
        with mock.patch.object(writer, "EXPECTED_TEXT_UTF8_SHA256", TEST_TEXT_SHA256):
            with self.assertRaisesRegex(writer.OracleReceiptError, "cache-on and cache-off"):
                writer._run_cache_pair(backend)

    def test_receipt_writer_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            writer._write_create_only(output, b"first\n")
            with self.assertRaisesRegex(writer.OracleReceiptError, "create-only"):
                writer._write_create_only(output, b"second\n")
            self.assertEqual(output.read_bytes(), b"first\n")

    def test_receipt_writer_requires_nofollow_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            with mock.patch.object(writer.os, "O_NOFOLLOW", 0):
                with self.assertRaisesRegex(writer.OracleReceiptError, "os\\.O_NOFOLLOW"):
                    writer._write_create_only(output, b"receipt\n")
            self.assertFalse(output.exists())
            self.assertNotIn(
                'hasattr(os, "O_NOFOLLOW")',
                Path(writer.__file__).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
