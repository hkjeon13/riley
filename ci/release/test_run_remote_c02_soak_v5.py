#!/usr/bin/env python3
"""Static, CPU-only guards for the authenticated C02 raw-v5 runner."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "ci/release/run_remote_c02_soak_v5.sh"
RELEASE_DIRECTORY = REPOSITORY_ROOT / "ci/release"


class RunRemoteC02SoakV5Tests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(RUNNER), *arguments],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )

    def test_bash_syntax_and_help_need_no_gpu_lock_or_evidence_root(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--binary-sha256", completed.stdout)
        self.assertIn("--model-tree-sha256", completed.stdout)
        self.assertIn("--scenario-contract", completed.stdout)
        self.assertIn("--freeze-sha256", completed.stdout)
        self.assertIn("raw v5 manifest", completed.stdout)

    def test_bad_invocations_fail_before_the_authenticated_gpu_supervisor(self) -> None:
        missing = self._run()
        self.assertEqual(missing.returncode, 2)
        self.assertIn("usage:", missing.stderr)

        duplicate = self._run("--binary", "/tmp/one", "--binary", "/tmp/two")
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("may occur only once", duplicate.stderr)

        unknown = self._run("--unrecognized-option")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown option", unknown.stderr)

        # --id is an old out-of-contract shortcut. It must be rejected by
        # the outer parser, before a filesystem, GPU, or evidence root opens.
        old_shortcut = self._run("--id=0")
        self.assertEqual(old_shortcut.returncode, 2)
        self.assertIn("unknown option", old_shortcut.stderr)

        ignored_timeout = self._run("--request-timeout-seconds", "1")
        self.assertEqual(ignored_timeout.returncode, 2)
        self.assertIn("unknown option", ignored_timeout.stderr)

        internal = self._run("--gpu-lock-supervised")
        self.assertEqual(internal.returncode, 2)
        self.assertIn("supervisor PID was not authenticated", internal.stderr)

    def test_isolated_python_bridge_resolves_checked_sibling_modules(self) -> None:
        bridge = (
            "import os,runpy,stat,sys\n"
            "script=sys.argv[1]\n"
            "metadata=os.lstat(script)\n"
            "assert stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)\n"
            "directory=os.path.dirname(script)\n"
            "assert os.path.isabs(script) and directory\n"
            "sys.path.insert(0,directory)\n"
            "sys.argv=[script,*sys.argv[2:]]\n"
            "runpy.run_path(script,run_name='__main__')\n"
        )
        for name in (
            "prepare_c02_lifecycle_evidence_v5.py",
            "verify_c02_lifecycle_launch_inputs_v1.py",
            "verify_c02_lifecycle_shutdown_v1.py",
        ):
            completed = subprocess.run(
                ["/usr/bin/python3", "-B", "-I", "-S", "-c", bridge, str(RELEASE_DIRECTORY / name), "--help"],
                text=True,
                capture_output=True,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            )
            self.assertEqual(completed.returncode, 0, f"{name}: {completed.stderr}")

    def test_runner_retains_the_narrow_authenticated_raw_v5_contract(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        required_fragments = (
            "set -euo pipefail",
            "set -o noclobber",
            "umask 077",
            "--gpu-lock-supervised",
            "os.O_NOFOLLOW",
            "os.O_NONBLOCK",
            "os.O_CLOEXEC",
            "fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)",
            "PR_SET_PDEATHSIG",
            "os.setsid()",
            'os.listdir("/proc/self/fd")',
            "if fd > 2 and fd != lock_fd",
            "RILEY_C02_LIFECYCLE_SUPERVISOR_LOCK_FD",
            "/proc/${PPID}/fd/${SUPERVISOR_LOCK_FD}",
            "FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE",
            "exec 9>&-",
            "runpy.run_path",
            "sys.path.insert(0, directory)",
            'nvidia-smi -i "$gpu_index"',
            "--query-gpu=uuid,memory.used",
            "observed_memory <= max_gpu_memory_mib",
            "prepare_c02_lifecycle_evidence_v5.py",
            "verify_c02_lifecycle_launch_inputs_v1.py",
            "run_remote_c02_config_endpoint_observation_v1.sh",
            "check_c02_config_bridge_v1.py",
            "run_remote_c02_raw_soak_scenarios_v1.sh",
            "run_remote_c02_observations_v2.sh",
            "verify_c02_lifecycle_shutdown_v1.py",
            "c02_lifecycle_process_guard_v1.py",
            "signal_server_if_current",
            "FALLBACK_CAPTURE_NAME='fallback-capture'",
            "FALLBACK_OBSERVATION_NAME='fallback-observation'",
            "FROZEN_CONTRACT_NAME='fallback-lifecycle-scenario-contract.json'",
            "--capture-name \"$FALLBACK_CAPTURE_NAME\"",
            "--capture-name \"$FALLBACK_OBSERVATION_NAME\"",
            "--scenario-contract \"$evidence_dir/$FROZEN_CONTRACT_NAME\"",
            "--configuration-profile must be max-performance-exact",
            "--sampling-backend gpu-greedy",
            "--sampling-backend|--sampling-backend=*",
            "run_private_v5_raw_finalizer",
            "finalize_c02_lifecycle_v5_raw.py",
            "_finalize_authenticated_v5_raw_once",
            "cannot publish a lifecycle receipt",
            "--shutdown-on-stdin=*|--c02-*",
            "--c02-audit-dir",
            "--c02-shutdown-artifact",
            "--noproxy '*'",
            "--sample-count 1",
            "qualification_status=not-run",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, source)
        for forbidden in (
            "docker ",
            "podman ",
            "ssh ",
            "systemctl ",
            "sudo ",
            "--server-command",
            "--id=",
            "0.0.0.0:",
            "bash -c",
            "eval ",
            "RILEY_C02_CAPTURE_CLEAN_ENV",
            "/bin/kill",
            "kill -0",
            "write_c02_lifecycle_bind_request_v1.py",
            "write_c02_lifecycle_supervisor_receipt_v1.py",
            "write_c02_lifecycle_bind_request_v5.py",
            "run_bind_raw_c02_soak_v5.sh",
            "bind_raw_c02_soak_v5.py",
        ):
            self.assertNotIn(forbidden, source)

    def test_capture_order_is_fixed_and_private_raw_finalization_is_last(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        sequence = (
            "prepare_c02_lifecycle_evidence_v5.py",
            "run_remote_c02_config_endpoint_observation_v1.sh",
            "check_c02_config_bridge_v1.py",
            "run_remote_c02_raw_soak_scenarios_v1.sh",
            "run_remote_c02_observations_v2.sh",
            "shutdown_server_successfully\n",
            "verify_c02_lifecycle_shutdown_v1.py",
            "run_private_v5_raw_finalizer >",
        )
        positions = [source.index(fragment) for fragment in sequence]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("publishes at most one raw v5 manifest", source)
        finalizer_call = source.index("run_private_v5_raw_finalizer >")
        success_output = source.index(
            'printf \'%s\\n\' "C02 native-fallback raw evidence completed',
            finalizer_call,
        )
        self.assertLess(finalizer_call, success_output)

    def test_runner_owns_the_gpu_greedy_sampling_selection(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        launch_sampling = source.index("--sampling-backend gpu-greedy")
        loaded_arguments = source.index('"${loaded_arguments[@]}"', launch_sampling)
        self.assertLess(launch_sampling, loaded_arguments)
        args_file = source[source.index("load_arguments() {") : source.index("load_environment() {")]
        self.assertIn("--sampling-backend|--sampling-backend=*", args_file)

    def test_readiness_and_shutdown_recheck_process_identity_after_races(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        ready_start = source.index("wait_for_ready() {")
        ready_end = source.index("\nwait_for_ready\n", ready_start)
        ready_block = source[ready_start:ready_end]
        first_identity = ready_block.index(
            'server_identity_state "$server_pid" "$server_start_ticks"'
        )
        curl_probe = ready_block.index("/usr/bin/curl --noproxy '*'")
        ready_return = ready_block.index("return 0")
        second_identity = ready_block.index(
            'server_identity_state "$server_pid" "$server_start_ticks"',
            first_identity + 1,
        )
        self.assertLess(first_identity, curl_probe)
        self.assertLess(curl_probe, second_identity)
        self.assertLess(second_identity, ready_return)
        self.assertIn(
            'if server_identity_state "$server_pid" "$server_start_ticks"; then\n'
            "                    return 0\n"
            "                else\n"
            "                    identity_status=$?\n"
            "                fi",
            ready_block,
        )

        cleanup_start = source.index("stop_server_after_failure() {")
        cleanup_end = source.index("\ncleanup() {", cleanup_start)
        cleanup_block = source[cleanup_start:cleanup_end]
        self.assertIn(
            "A child can become a zombie during the final sleep above.", cleanup_block
        )
        self.assertIn(
            'if server_identity_state "$pid" "$ticks"; then\n                        state=0',
            cleanup_block,
        )

        shutdown_start = source.index("shutdown_server_successfully() {")
        shutdown_end = source.index("\nshutdown_server_successfully\n", shutdown_start)
        shutdown_block = source[shutdown_start:shutdown_end]
        self.assertIn(
            "Avoid acting on the state from before the final sleep", shutdown_block
        )
        self.assertIn(
            'if server_identity_state "$pid" "$ticks"; then\n            identity_status=0',
            shutdown_block,
        )


if __name__ == "__main__":
    unittest.main()
