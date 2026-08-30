#!/usr/bin/env python3
"""Static, CPU-only guards for the RC3 Gate E supervisor smoke-test runner."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "ci/release/run_remote_rc3_gate_e_session_v1.sh"


class RunRemoteRc3GateESessionTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(RUNNER), *arguments],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )

    def test_bash_syntax_and_help_need_no_shared_lock(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--supervisor-smoke-test", completed.stdout)
        self.assertIn("no Gate E action was run", completed.stdout)
        self.assertIn("does not select or query a GPU", completed.stdout)
        self.assertIn("absolute path", completed.stdout)

    def test_bad_invocations_fail_before_authenticated_gpu_supervision(self) -> None:
        missing = self._run()
        self.assertEqual(missing.returncode, 2)
        self.assertIn("usage:", missing.stderr)

        duplicate = self._run(
            "--supervisor-smoke-test",
            "--supervisor-smoke-test",
        )
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("exactly one", duplicate.stderr)

        trailing = self._run("--supervisor-smoke-test", "unexpected")
        self.assertEqual(trailing.returncode, 2)
        self.assertIn("exactly one", trailing.stderr)

        unknown = self._run("--unrecognized-option")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown option", unknown.stderr)

        legacy = self._run("--id=0")
        self.assertEqual(legacy.returncode, 2)
        self.assertIn("unknown option", legacy.stderr)

        invented_gpu = self._run("--gpu-index", "0")
        self.assertEqual(invented_gpu.returncode, 2)
        self.assertIn("exactly one", invented_gpu.stderr)

        internal = self._run("--gpu-lock-supervised")
        self.assertEqual(internal.returncode, 2)
        self.assertIn("supervisor PID was not authenticated", internal.stderr)

        relative = subprocess.run(
            ["bash", str(RUNNER.relative_to(REPOSITORY_ROOT)), "--supervisor-smoke-test"],
            text=True,
            capture_output=True,
            check=False,
            cwd=REPOSITORY_ROOT,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(relative.returncode, 2)
        self.assertIn("must be invoked by absolute path", relative.stderr)

        sourced = subprocess.run(
            ["bash", "-c", 'source "$1"', "bash", str(RUNNER)],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(sourced.returncode, 2)
        self.assertIn("must be executed, not sourced", sourced.stderr)

    def test_runner_is_a_no_action_authenticated_supervisor_only(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        required_fragments = (
            "set -euo pipefail",
            "set -o noclobber",
            "umask 077",
            "BASH_SOURCE[0]",
            '[[ ${BASH_SOURCE[0]:-} != /* ]]',
            "return 2 2>/dev/null || exit 2",
            "--gpu-lock-supervised",
            "--supervisor-smoke-test",
            "os.O_NOFOLLOW",
            "os.O_NONBLOCK",
            "os.O_CLOEXEC",
            "fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)",
            "PR_SET_PDEATHSIG",
            "os.setsid()",
            'os.listdir("/proc/self/fd")',
            "if fd > 2 and fd != lock_fd",
            "RILEY_RC3_GATE_E_SESSION_SUPERVISOR_PID",
            "RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_FD",
            "RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_ID",
            "/proc/$PPID/fd/$SUPERVISOR_LOCK_FD",
            "/proc/$$/fd/$SUPERVISOR_LOCK_FD",
            "observed_lock_id=$(/usr/bin/stat -Lc '%d:%i'",
            "inherited supervisor lock identity differs from parent",
            "FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE",
            "exec 9>&-",
            "env -i",
            "authenticated supervisor smoke test completed; no Gate E action was run",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, source)

        for forbidden in (
            "nvidia-smi",
            "docker ",
            "podman ",
            "ssh ",
            "curl ",
            "systemctl ",
            "sudo ",
            "eval ",
            "bash -c",
            "--server-command",
            "--id=",
            "0.0.0.0:",
            "write_rc3_gate_e",
            "write_rc3_frozen_candidate",
            "replay_rc3_gate_e_aggregate",
            "check_release_candidate.py",
            "check_rc3_qualification.py",
            'getattr(os, "O_',
            "SUPERVISOR_TOKEN",
        ):
            self.assertNotIn(forbidden, source)

    def test_authentication_precedes_the_only_diagnostic(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertLess(
            source.index('preflight_invocation "$@"'),
            source.index("opened_fd = os.open(LOCK_PATH"),
        )
        authentication = source.index(
            '[[ ${RILEY_RC3_GATE_E_SESSION_SUPERVISOR_PID:-} =~'
        )
        diagnostic = source.rindex(
            "authenticated supervisor smoke test completed; no Gate E action was run"
        )
        self.assertLess(authentication, source.rindex("exec 9>&-"))
        self.assertLess(source.rindex("exec 9>&-"), diagnostic)


if __name__ == "__main__":
    unittest.main()
