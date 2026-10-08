#!/usr/bin/env python3
"""CPU-only hostile-path tests for the immutable Gate E execution anchor."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ISOLATED_PYTHON = ("/usr/bin/python3.10", "-I", "-S", "-E", "-B")
sys.path.insert(0, os.fspath(REPOSITORY_ROOT / "ci" / "release"))

import verify_rc3_gate_e_execution_anchor_v1 as anchor  # noqa: E402


class ExecutionAnchorTests(unittest.TestCase):
    def _make_anchor(self, root: Path) -> tuple[Path, Path]:
        anchor_root = root / "anchor"
        lock_directory = root / "lock"
        anchor_root.mkdir(mode=0o755)
        lock_directory.mkdir(mode=0o700)
        os.chmod(anchor_root, 0o755)
        os.chmod(lock_directory, 0o700)
        bootstrap = anchor_root / anchor.BOOTSTRAP_NAME
        core = anchor_root / anchor.CORE_NAME
        bootstrap.write_bytes(b"#!/usr/bin/python3.10\n# installed bootstrap fixture\n")
        core.write_bytes(b"# installed private raw-core fixture\n")
        os.chmod(bootstrap, 0o555)
        os.chmod(core, 0o444)
        manifest = {
            "schema_version": anchor.ANCHOR_SCHEMA_VERSION,
            "bootstrap": {
                "filename": anchor.BOOTSTRAP_NAME,
                "sha256": hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
                "byte_length": bootstrap.stat().st_size,
            },
            "core": {
                "filename": anchor.CORE_NAME,
                "sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
                "byte_length": core.stat().st_size,
            },
            "lock_directory": os.fspath(lock_directory),
        }
        (anchor_root / anchor.MANIFEST_NAME).write_bytes(
            anchor._canonical_json_bytes(manifest) + b"\n"
        )
        os.chmod(anchor_root / anchor.MANIFEST_NAME, 0o444)
        return anchor_root, lock_directory

    def _verify_fixture(self, root: Path) -> dict[str, object]:
        anchor_root, lock_directory = self._make_anchor(root)
        return anchor._verify_anchor(
            anchor_root,
            lock_directory,
            owner_uid=os.geteuid(),
            trusted_prefix=root,
        )

    def test_private_fixture_verifies_only_the_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._verify_fixture(root)
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["scope"], anchor.ANCHOR_SCOPE)
        self.assertEqual(result["authority"], anchor.ANCHOR_AUTHORITY)
        self.assertEqual(
            result["not_established"],
            {
                "bootstrap_execution": "not-established",
                "private_core_execution": "not-established",
                "verifier_source_integrity": "not-established",
                "host_mount_namespace_identity": "not-established",
                "acl_write_prohibition": "not-established",
                "gpu_lock_acquired": "not-established",
                "gpu_query": "not-established",
                "docker_execution": "not-established",
                "evidence_created": "not-established",
                "actual_gate_e_capture": "not-established",
                "semantic_receipt": "not-established",
                "qualification": "not-established",
            },
        )

    def test_production_policy_requires_root_owned_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor_root, lock_directory = self._make_anchor(root)
            with self.assertRaisesRegex(anchor.ExecutionAnchorError, "unsafe-anchor-owner"):
                anchor._verify_anchor(
                    anchor_root,
                    lock_directory,
                    owner_uid=0,
                    trusted_prefix=root,
                )

    def test_link_mode_digest_and_lock_mutations_fail_closed(self) -> None:
        mutations = {
            "bootstrap-symlink": lambda anchor_root, _lock: (
                (anchor_root / anchor.BOOTSTRAP_NAME).unlink(),
                (anchor_root / anchor.BOOTSTRAP_NAME).symlink_to(anchor.CORE_NAME),
            ),
            "group-writable-core": lambda anchor_root, _lock: os.chmod(
                anchor_root / anchor.CORE_NAME, 0o664
            ),
            "setuid-bootstrap": lambda anchor_root, _lock: os.chmod(
                anchor_root / anchor.BOOTSTRAP_NAME, 0o4555
            ),
            "core-digest-mismatch": lambda anchor_root, _lock: (
                os.chmod(anchor_root / anchor.CORE_NAME, 0o644),
                (anchor_root / anchor.CORE_NAME).write_bytes(b"different private core bytes\n"),
            ),
            "lock-mode": lambda _anchor_root, lock: os.chmod(lock, 0o755),
        }
        for name, mutation in mutations.items():
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                anchor_root, lock_directory = self._make_anchor(root)
                mutation(anchor_root, lock_directory)
                with self.assertRaises(anchor.ExecutionAnchorError):
                    anchor._verify_anchor(
                        anchor_root,
                        lock_directory,
                        owner_uid=os.geteuid(),
                        trusted_prefix=root,
                    )

    def test_noncanonical_or_widened_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor_root, lock_directory = self._make_anchor(root)
            manifest_path = anchor_root / anchor.MANIFEST_NAME
            os.chmod(manifest_path, 0o644)
            manifest_path.write_text(
                '{"schema_version":"first","schema_version":"second"}\n',
                encoding="utf-8",
            )
            os.chmod(manifest_path, 0o444)
            with self.assertRaisesRegex(anchor.ExecutionAnchorError, "noncanonical-json"):
                anchor._verify_anchor(
                    anchor_root,
                    lock_directory,
                    owner_uid=os.geteuid(),
                    trusted_prefix=root,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor_root, lock_directory = self._make_anchor(root)
            manifest_path = anchor_root / anchor.MANIFEST_NAME
            manifest = anchor._parse_manifest(manifest_path.read_bytes())
            manifest["unexpected"] = "widened"  # type: ignore[index]
            os.chmod(manifest_path, 0o644)
            manifest_path.write_bytes(anchor._canonical_json_bytes(manifest) + b"\n")
            os.chmod(manifest_path, 0o444)
            with self.assertRaisesRegex(anchor.ExecutionAnchorError, "invalid-anchor-manifest"):
                anchor._verify_anchor(
                    anchor_root,
                    lock_directory,
                    owner_uid=os.geteuid(),
                    trusted_prefix=root,
                )

    def test_public_cli_has_no_path_override_or_execution_path(self) -> None:
        runner = REPOSITORY_ROOT / "ci" / "release" / "verify_rc3_gate_e_execution_anchor_v1.py"
        help_result = subprocess.run(
            [*ISOLATED_PYTHON, os.fspath(runner), "--help"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--anchor-contract-probe", help_result.stdout)

        for arguments in (("--id=0",), ("--anchor-root", "/tmp/unsafe")):
            with self.subTest(arguments=arguments):
                rejected = subprocess.run(
                    [*ISOLATED_PYTHON, os.fspath(runner), *arguments],
                    text=True,
                    capture_output=True,
                    check=False,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
                )
                self.assertEqual(rejected.returncode, 2)
                self.assertIn("unrecognized arguments", rejected.stderr)

        fixed_probe = subprocess.run(
            [*ISOLATED_PYTHON, os.fspath(runner), "--anchor-contract-probe"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertIn(fixed_probe.returncode, {0, 2})
        if fixed_probe.returncode == 0:
            self.assertIn('"status":"checked"', fixed_probe.stdout)
        else:
            self.assertIn("RC3 Gate E execution anchor:", fixed_probe.stderr)

        unisolated = subprocess.run(
            ["/usr/bin/python3.10", "-B", os.fspath(runner), "--anchor-contract-probe"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(unisolated.returncode, 2)
        self.assertIn("unsafe-reviewed-python", unisolated.stderr)

    def test_static_surface_is_verification_only(self) -> None:
        source = (
            REPOSITORY_ROOT / "ci" / "release" / "verify_rc3_gate_e_execution_anchor_v1.py"
        ).read_text(encoding="utf-8")
        for required in (
            '"O_NOFOLLOW"',
            '"O_DIRECTORY"',
            '"O_CLOEXEC"',
            "_stable_identity",
            "root-owned",
            "sys.flags.isolated",
            "PYTHON_SHA256",
            "verify_fixed_execution_anchor",
            "--anchor-contract-probe",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "fcntl.flock",
            "subprocess.",
            "os.exec",
            "os.fork",
            "nvidia-smi",
            "docker run",
            "write_rc3_gate_e",
            "replay_rc3_gate_e",
            "check_rc3_qualification",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
