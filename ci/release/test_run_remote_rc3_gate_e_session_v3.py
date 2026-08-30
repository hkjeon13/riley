#!/usr/bin/env python3
"""CPU-only contract tests for the root-bound RC3 Gate E v3 bootstrap.

The public entrypoint is deliberately not exercised from a checkout as an
execution path: it must fail before any anchor, lock, socket, or child action.
Positive coverage uses an isolated current-UID fixture and the bootstrap's
private test seam.  It validates only the sealed no-action core handshake; no
GPU, Docker, evidence, replay, receipt, or qualification operation is used.
"""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = REPOSITORY_ROOT / "ci/release/run_remote_rc3_gate_e_session_v3.py"
CORE = REPOSITORY_ROOT / "ci/release/rc3_gate_e_private_raw_core_v1.py"
PYTHON = "/usr/bin/python3.10"


ISOLATED_DRIVER = r'''
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


BOOTSTRAP_NAME = "run_remote_rc3_gate_e_session_v3.py"
CORE_NAME = "rc3_gate_e_private_raw_core_v1.py"
MANIFEST_NAME = "execution-anchor.json"
LOCK_NAME = "gate-e-v3.lock"
SCHEMA = "riley.rc3-gate-e-execution-anchor.v1"


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_bytes(path, payload, mode):
    path.write_bytes(payload)
    os.chmod(path, mode)


def manifest_entry(path, name):
    payload = path.read_bytes()
    return {
        "filename": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_length": len(payload),
    }


def load_fixture_bootstrap(path):
    spec = importlib.util.spec_from_file_location("rc3_gate_e_v3_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct fixture bootstrap import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def close_fd_seven():
    try:
        os.close(7)
    except OSError:
        pass


def lock_is_available(lock_path):
    descriptor = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def run(mode):
    bootstrap_source = Path(sys.argv[1])
    core_source = Path(sys.argv[2])
    with tempfile.TemporaryDirectory(prefix="rc3-gate-e-v3-bootstrap-") as raw_temp:
        temp_root = Path(raw_temp)
        anchor_root = temp_root / "anchor"
        lock_directory = temp_root / "lock"
        anchor_root.mkdir(mode=0o755)
        lock_directory.mkdir(mode=0o700)
        bootstrap = anchor_root / BOOTSTRAP_NAME
        core = anchor_root / CORE_NAME
        manifest = anchor_root / MANIFEST_NAME
        lock_path = lock_directory / LOCK_NAME
        write_bytes(bootstrap, bootstrap_source.read_bytes(), 0o755)
        write_bytes(core, core_source.read_bytes(), 0o644)
        write_bytes(lock_path, b"", 0o600)
        anchor_manifest = {
            "schema_version": SCHEMA,
            "bootstrap": manifest_entry(bootstrap, BOOTSTRAP_NAME),
            "core": manifest_entry(core, CORE_NAME),
            "lock_directory": os.fspath(lock_directory),
        }
        write_bytes(manifest, canonical(anchor_manifest) + b"\n", 0o644)
        if mode == "core_tamper":
            with open(core, "ab", buffering=0) as handle:
                handle.write(b"x")
        elif mode == "manifest_noncanonical":
            write_bytes(manifest, canonical(anchor_manifest), 0o644)
        elif mode == "lock_wrong_mode":
            os.chmod(lock_path, 0o640)
        elif mode == "lock_symlink":
            target = temp_root / "other-lock"
            write_bytes(target, b"", 0o600)
            os.unlink(lock_path)
            os.symlink(target, lock_path)
        module = load_fixture_bootstrap(bootstrap)
        close_fd_seven()

        if mode == "stdio_cloexec":
            for descriptor in (0, 1, 2):
                os.set_inheritable(descriptor, False)

        def invoke():
            return module._run_no_action_for_test(
                anchor_root,
                lock_directory,
                owner_uid=os.getuid(),
                owner_gid=os.getgid(),
                trusted_prefix=temp_root,
            )

        held_descriptor = None
        try:
            if mode == "lock_held":
                held_descriptor = os.open(
                    lock_path,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                )
                fcntl.flock(held_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if mode in {"success", "stdio_cloexec"}:
                result = invoke()
                lock_is_available(lock_path)
                print(json.dumps({"status": "success", "result": result}, sort_keys=True))
                return
            try:
                invoke()
            except module.BootstrapError as error:
                print(
                    json.dumps(
                        {
                            "status": "expected-failure",
                            "error": str(error),
                        },
                        sort_keys=True,
                    )
                )
                return
            raise AssertionError("hostile fixture unexpectedly completed")
        finally:
            if held_descriptor is not None:
                try:
                    fcntl.flock(held_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(held_descriptor)


run(sys.argv[3])
'''


class GateEV3BootstrapTests(unittest.TestCase):
    maxDiff = None

    def _isolated(self, mode: str) -> dict[str, object]:
        completed = subprocess.run(
            [
                PYTHON,
                "-I",
                "-S",
                "-E",
                "-B",
                "-c",
                ISOLATED_DRIVER,
                os.fspath(BOOTSTRAP),
                os.fspath(CORE),
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f"isolated fixture mode={mode!r} failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            self.fail(
                f"isolated fixture mode={mode!r} did not emit one JSON result: "
                f"{error}; stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )

    def test_private_fixture_performs_only_the_sealed_no_action_handoff(self) -> None:
        for mode in ("success", "stdio_cloexec"):
            with self.subTest(mode=mode):
                record = self._isolated(mode)
                self.assertEqual(record["status"], "success")
                result = record["result"]
                self.assertIsInstance(result, dict)
                self.assertEqual(result["scope"], "bootstrap-core-no-action-smoke-test-only")
                self.assertEqual(result["status"], "no-action-complete")
                self.assertEqual(result["parent_lock_fd"], 7)
                self.assertEqual(
                    result["core_sha256"],
                    "953de3d1cffa78d38317505b85334337293a54e30f946bf8e690913e6e75815c",
                )
                self.assertEqual(
                    result["guarantees"],
                    {
                        "gpu_lock_acquired": False,
                        "gpu_queried": False,
                        "docker_invoked": False,
                        "evidence_created": False,
                        "semantic_replay_run": False,
                        "receipt_published": False,
                        "qualification_decided": False,
                    },
                )

    def test_anchor_and_lock_tampering_fail_closed_before_handoff(self) -> None:
        for mode in (
            "core_tamper",
            "manifest_noncanonical",
            "lock_wrong_mode",
            "lock_symlink",
            "lock_held",
        ):
            with self.subTest(mode=mode):
                record = self._isolated(mode)
                self.assertEqual(record["status"], "expected-failure")
                self.assertIsInstance(record["error"], str)
                self.assertTrue(record["error"])

    def test_checkout_entrypoint_rejects_before_fixed_anchor_work(self) -> None:
        protected_paths = (
            Path("/opt/riley/rc3-gate-e-v1"),
            Path("/var/lib/riley/rc3-gate-e/lock/gate-e-v3.lock"),
        )
        before = {path: path.exists() for path in protected_paths}
        completed = subprocess.run(
            [
                PYTHON,
                "-I",
                "-S",
                "-E",
                "-B",
                os.fspath(BOOTSTRAP),
                "--bootstrap-core-smoke-test",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("fixed root-installed bootstrap path", completed.stderr)
        self.assertEqual(before, {path: path.exists() for path in protected_paths})

    def test_unrecognized_cli_override_is_rejected(self) -> None:
        completed = subprocess.run(
            [
                PYTHON,
                "-I",
                "-S",
                "-E",
                "-B",
                os.fspath(BOOTSTRAP),
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("only --bootstrap-core-smoke-test is accepted", completed.stderr)

    def test_source_retains_its_fixed_no_action_contract(self) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn('ANCHOR_ROOT: Final = Path("/opt/riley/rc3-gate-e-v1")', source)
        self.assertIn('LOCK_DIRECTORY: Final = Path("/var/lib/riley/rc3-gate-e/lock")', source)
        self.assertIn('LOCK_NAME: Final = "gate-e-v3.lock"', source)
        self.assertIn('PARENT_LOCK_FD: Final = 7', source)
        self.assertIn('CORE_FD: Final = 8', source)
        self.assertIn('CONFIG_FD: Final = 9', source)
        self.assertIn('CONTROL_FD: Final = 10', source)
        self.assertIn("COMPILED_CORE_SHA256", source)
        self.assertIn("COMPILED_CORE_BYTE_LENGTH", source)
        self.assertIn("os.memfd_create", source)
        self.assertIn("F_SEAL_WRITE", source)
        self.assertIn("socket.SOCK_SEQPACKET", source)
        self.assertIn("socket.SO_PASSCRED", source)
        self.assertIn("os.execve", source)
        self.assertIn("fcntl.flock", source)
        self.assertIn("ALLOWED_ANCHOR_FILESYSTEMS", source)
        self.assertIn("require_local_filesystem=True", source)
        self.assertIn("require_success=True", source)
        self.assertIn("os.set_inheritable(descriptor, True)", source)
        self.assertNotIn("O_CREAT", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("write_rc3", source)


if __name__ == "__main__":
    unittest.main()
