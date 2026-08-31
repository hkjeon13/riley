#!/usr/bin/env python3
"""CPU-only CLI tests for the stdin-only guardian review-input checker."""

from __future__ import annotations

import hashlib
import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RELEASE_DIRECTORY = Path(__file__).resolve().parent
CHECKER = RELEASE_DIRECTORY / "check_gate_e_native_guardian_review_v1.py"
sys.path.insert(0, str(RELEASE_DIRECTORY))

import check_gate_e_native_guardian_review_v1 as checker  # noqa: E402
import gate_e_native_guardian_review_contract_v1 as contract  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def binding(label: str, length: int = 1_024) -> dict[str, object]:
    return {"byte_length": length, "sha256": digest(label)}


def review_input() -> bytes:
    sidecar = binding("execution-closure-sidecar", 4_096)
    return contract.canonical_native_guardian_review_bytes(
        {
            "schema_version": contract.SCHEMA_VERSION,
            "scope": contract.SCOPE,
            "authority": contract.AUTHORITY,
            "review_status": contract.REVIEW_STATUS,
            "installation_status": contract.INSTALLATION_STATUS,
            "operational_status": contract.OPERATIONAL_STATUS,
            "bundle": {
                "bundle_schema_version": contract.BUNDLE_SCHEMA_VERSION,
                "v1_compatibility": False,
                "execution_closure_sidecar": sidecar,
                "manifest": {
                    "byte_length": 8_192,
                    "sha256": digest("bundle-manifest"),
                    "execution_closure_sidecar_byte_length": sidecar["byte_length"],
                    "execution_closure_sidecar_sha256": sidecar["sha256"],
                },
            },
            "execution_strategy": {
                "kind": contract.STATIC_STRATEGY,
                "pt_interp": "absent",
                "dependency_resolution": "none",
                "static_elf_inspection_policy_sha256": digest("static-elf-policy"),
            },
            "fd_abi": {
                "bootstrap_inherited_fds": list(contract.BOOTSTRAP_FDS),
                "worker_inherited_fds": list(contract.WORKER_FDS),
                "core_fd": 32,
                "environment": "empty",
                "no_new_privs": True,
                "capabilities": "cleared",
            },
            "required_artifacts": {
                name: digest(f"artifact:{name}")
                for name in sorted(contract.REQUIRED_ARTIFACTS)
            },
        }
    )


def run_checker(
    arguments: list[str],
    raw: bytes = b"",
    *,
    isolated: bool = True,
    checker_path: Path = CHECKER,
) -> subprocess.CompletedProcess[bytes]:
    command = [sys.executable]
    if isolated:
        command.extend(["-I", "-S", "-E", "-B"])
    command.extend([str(checker_path), *arguments])
    return subprocess.run(command, input=raw, capture_output=True, check=False)


class NativeGuardianReviewCheckerTests(unittest.TestCase):
    def test_checker_has_no_caller_controlled_path_or_operational_surface(self) -> None:
        source = inspect.getsource(checker)
        for forbidden in (
            "import os",
            "import subprocess",
            "import socket",
            "open(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("sys.stdin.buffer", source)
        self.assertNotIn("sys.path.insert", source)
        self.assertIn("Path(__file__).resolve()", source)

    def test_valid_input_requires_explicit_opt_in_and_emits_nothing(self) -> None:
        completed = run_checker(["--review-input-contract-check"], review_input())
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8"))
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(completed.stderr, b"")

    def test_help_missing_flag_unknown_and_pathlike_options_fail(self) -> None:
        help_result = run_checker(["--help"])
        self.assertEqual(help_result.returncode, 0)
        self.assertIn(b"--review-input-contract-check", help_result.stdout)

        missing = run_checker([], review_input())
        self.assertEqual(missing.returncode, 2)
        self.assertIn(b"--review-input-contract-check is required", missing.stderr)

        for option in ("--id=0", "--record", "--install", "--approve"):
            with self.subTest(option=option):
                completed = run_checker([option], review_input())
                self.assertEqual(completed.returncode, 2)
                self.assertIn(b"unrecognized arguments", completed.stderr)

    def test_non_isolated_runtime_and_invalid_or_oversized_stdin_fail_closed(self) -> None:
        non_isolated = run_checker(["--review-input-contract-check"], review_input(), isolated=False)
        self.assertEqual(non_isolated.returncode, 2)
        self.assertIn(b"requires Python -I -S -E -B", non_isolated.stderr)

        malformed = run_checker(["--review-input-contract-check"], b"{}\n")
        self.assertEqual(malformed.returncode, 2)
        self.assertIn(b"unexpected-field-set", malformed.stderr)

        oversized = run_checker(
            ["--review-input-contract-check"],
            b"x" * (contract.MAX_DOCUMENT_BYTES + 1),
        )
        self.assertEqual(oversized.returncode, 2)
        self.assertIn(b"review-byte-budget-exceeded", oversized.stderr)

    def test_symlinked_checker_ignores_a_neighbor_contract_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            shadow_directory = Path(temporary_directory)
            shadow_contract = shadow_directory / "gate_e_native_guardian_review_contract_v1.py"
            shadow_contract.write_text("raise RuntimeError('shadow contract loaded')\n", encoding="utf-8")
            checker_link = shadow_directory / "review-checker-link.py"
            checker_link.symlink_to(CHECKER)

            completed = run_checker(
                ["--review-input-contract-check"],
                review_input(),
                checker_path=checker_link,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8"))
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(completed.stderr, b"")


if __name__ == "__main__":
    unittest.main()
