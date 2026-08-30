#!/usr/bin/env python3
"""CPU-only tests for the source-bound RC3 Gate E v2 probe."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "ci/release/run_remote_rc3_gate_e_session_v2.py"
PYTHON = "/usr/bin/python3.10"


class RunRemoteRc3GateESessionV2Tests(unittest.TestCase):
    def _run(
        self,
        *arguments: str,
        runner: Path = RUNNER,
        isolated: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [PYTHON]
        if isolated:
            command.extend(["-I", "-S", "-E"])
        command.extend([str(runner), *arguments])
        return subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )

    def test_help_and_bad_invocations_are_no_action(self) -> None:
        help_result = self._run("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--performance-source-contract-probe", help_result.stdout)
        self.assertIn("does not open a GPU lock", help_result.stdout)

        missing = self._run()
        self.assertEqual(missing.returncode, 2)
        self.assertIn("usage:", missing.stderr)

        unknown = self._run("--id=0")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("usage:", unknown.stderr)

    def test_probe_is_isolated_and_reports_only_sealed_source_metadata(self) -> None:
        unisolated = self._run(
            "--performance-source-contract-probe",
            isolated=False,
        )
        self.assertEqual(unisolated.returncode, 2)
        self.assertIn("must run under", unisolated.stderr)

        completed = self._run("--performance-source-contract-probe")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(
            payload["schema_version"],
            "riley.rc3-gate-e-session-source-contract.v2",
        )
        self.assertEqual(payload["status"], "source-bound-no-action")
        self.assertEqual(
            payload["guarantees"],
            {
                "gpu_lock_opened": False,
                "bash_invoked": False,
                "docker_invoked": False,
                "evidence_created": False,
                "receipt_published": False,
                "qualification_decided": False,
            },
        )
        source = payload["source"]
        self.assertEqual(
            source["body_path"],
            "ci/run_remote_release_performance.sh",
        )
        self.assertEqual(source["repository_root"], str(REPOSITORY_ROOT))
        self.assertGreater(source["repository_root_dev"], 0)
        self.assertGreater(source["repository_root_ino"], 0)
        self.assertGreater(source["sealed_snapshot_size"], 0)
        self.assertRegex(source["sealed_snapshot_sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(source["body_source_dev"], 0)
        self.assertGreater(source["body_source_ino"], 0)
        self.assertGreater(source["sealed_snapshot_dev"], 0)
        self.assertGreater(source["sealed_snapshot_ino"], 0)

    def test_relative_and_symlinked_launcher_paths_fail_before_source_probe(self) -> None:
        relative = subprocess.run(
            [
                PYTHON,
                "-I",
                "-S",
                "-E",
                str(RUNNER.relative_to(REPOSITORY_ROOT)),
                "--performance-source-contract-probe",
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(relative.returncode, 2)
        self.assertIn("fixed trusted launcher path", relative.stderr)

        with tempfile.TemporaryDirectory() as directory:
            symlink = Path(directory) / "session-v2.py"
            os.symlink(RUNNER, symlink)
            linked = self._run(
                "--performance-source-contract-probe",
                runner=symlink,
            )
        self.assertEqual(linked.returncode, 2)
        self.assertIn("fixed trusted launcher path", linked.stderr)

        with tempfile.TemporaryDirectory() as directory:
            linked_root = Path(directory) / "linked-root"
            os.symlink(REPOSITORY_ROOT, linked_root)
            ancestor_linked = self._run(
                "--performance-source-contract-probe",
                runner=(
                    linked_root / "ci/release/run_remote_rc3_gate_e_session_v2.py"
                ),
            )
        self.assertEqual(ancestor_linked.returncode, 2)
        self.assertIn("fixed trusted launcher path", ancestor_linked.stderr)

    def test_static_contract_has_no_action_capability_and_seals_before_future_use(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for fragment in (
            "/usr/bin/python3.10 -I -S -E",
            "sys.flags.isolated",
            "sys.flags.no_site",
            "sys.flags.ignore_environment",
            "os.O_NOFOLLOW",
            "os.O_NONBLOCK",
            "os.O_CLOEXEC",
            "os.memfd_create",
            "os.MFD_ALLOW_SEALING",
            "fcntl.F_ADD_SEALS",
            "fcntl.F_GET_SEALS",
            'REMOTE_REPOSITORY_ROOT = "/home/psyche/rustinfer-vllm-roadmap-serial"',
            "_open_directory_at(repository_root_fd",
            'dir_fd=ci_fd',
            "SOURCE_SNAPSHOT_FD = 8",
            "reserved source descriptor is already open",
            "source-bound-no-action",
        ):
            self.assertIn(fragment, source)
        self.assertLess(
            source.index("os.memfd_create"),
            source.index("fcntl.fcntl(snapshot_fd, fcntl.F_ADD_SEALS"),
        )
        self.assertLess(
            source.index("fcntl.fcntl(snapshot_fd, fcntl.F_ADD_SEALS"),
            source.index("source-bound-no-action"),
        )
        self.assertNotIn("Path(__file__)", source)
        self.assertNotIn("Path.resolve", source)
        for forbidden in (
            "fcntl.flock",
            "nvidia-smi",
            "/usr/bin/docker",
            "subprocess.",
            "os.exec",
            "socket.",
            "ssh ",
            "write_rc3",
            "replay_rc3",
            "check_rc3_qualification",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
