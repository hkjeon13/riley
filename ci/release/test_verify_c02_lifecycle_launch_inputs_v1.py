#!/usr/bin/env python3
"""CPU-only hostile-path tests for lifecycle host launch-input verification."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import verify_c02_lifecycle_launch_inputs_v1 as verifier


class VerifyC02LifecycleLaunchInputsV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.binary = self.base / "riley"
        self.binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o700)
        self.model = self.base / "model"
        self.model.mkdir(mode=0o700)
        (self.model / "config.json").write_bytes(b'{"model":"fixture"}')
        nested = self.model / "weights"
        nested.mkdir(mode=0o700)
        (nested / "part-00.bin").write_bytes(b"weights")
        for path in (self.model / "config.json", nested / "part-00.bin"):
            path.chmod(0o600)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _binary_digest(self) -> str:
        return hashlib.sha256(self.binary.read_bytes()).hexdigest()

    def _model_digest(self) -> str:
        rows = []
        for path in sorted((path for path in self.model.rglob("*") if path.is_file()), key=lambda item: item.as_posix()):
            rows.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(self.model).as_posix()}\n".encode("ascii")
            )
        return hashlib.sha256(b"".join(rows)).hexdigest()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def test_verifies_binary_and_deterministic_model_tree(self) -> None:
        result = verifier.verify_launch_inputs(
            binary=self.binary,
            binary_sha256=self._binary_digest(),
            model_dir=self.model,
            model_tree_sha256=self._model_digest(),
            phase="pre-launch",
        )
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["phase"], "pre-launch")
        self.assertEqual(result["binary"]["sha256"], self._binary_digest())
        self.assertEqual(result["model"]["tree_sha256"], self._model_digest())
        self.assertEqual(result["model"]["file_count"], 2)

    def test_uses_global_relative_path_sort_matching_the_host_contract(self) -> None:
        tokenizer = self.model / "tokenizer"
        tokenizer.mkdir(mode=0o700)
        (tokenizer / "merges.txt").write_bytes(b"merges")
        (tokenizer / "merges.txt").chmod(0o600)
        (self.model / "tokenizer.json").write_bytes(b"tokenizer")
        (self.model / "tokenizer.json").chmod(0o600)

        result = verifier.verify_model_tree(self.model, self._model_digest())

        self.assertEqual(result["tree_sha256"], self._model_digest())
        self.assertEqual(result["file_count"], 4)

    def test_rejects_digest_drift_mutable_entries_and_links(self) -> None:
        with self.assertRaises(verifier.LaunchInputVerificationError) as raised:
            verifier.verify_binary(self.binary, "a" * 64)
        self.assert_reason(raised, "binary-sha256-mismatch")

        (self.model / "config.json").chmod(0o620)
        with self.assertRaises(verifier.LaunchInputVerificationError) as raised:
            verifier.verify_model_tree(self.model, self._model_digest())
        self.assert_reason(raised, "unsafe-input-mode")
        (self.model / "config.json").chmod(0o600)

        (self.model / "link.bin").symlink_to("config.json")
        with self.assertRaises(verifier.LaunchInputVerificationError) as raised:
            verifier.verify_model_tree(self.model, self._model_digest())
        self.assert_reason(raised, "unsafe-model-path")

    def test_rejects_noncanonical_paths_and_invalid_phase(self) -> None:
        with self.assertRaises(verifier.LaunchInputVerificationError) as raised:
            verifier.verify_launch_inputs(
                binary=Path("relative-riley"),
                binary_sha256=self._binary_digest(),
                model_dir=self.model,
                model_tree_sha256=self._model_digest(),
                phase="pre-launch",
            )
        self.assert_reason(raised, "invalid-absolute-path")
        with self.assertRaises(verifier.LaunchInputVerificationError) as raised:
            verifier.verify_launch_inputs(
                binary=self.binary,
                binary_sha256=self._binary_digest(),
                model_dir=self.model,
                model_tree_sha256=self._model_digest(),
                phase="afterwards",
            )
        self.assert_reason(raised, "invalid-phase")


if __name__ == "__main__":
    unittest.main()
