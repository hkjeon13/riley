#!/usr/bin/env python3
"""Static, CPU-only guards for the authenticated RC3 rollback raw runner."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "ci/release/run_remote_rc3_rollback_capture.sh"
RELEASE_DIRECTORY = REPOSITORY_ROOT / "ci/release"


class RunRemoteRc3RollbackCaptureTests(unittest.TestCase):
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
        for flag in (
            "--baseline-manifest-path",
            "--candidate-scenario-contract",
            "--rollback-generation-request",
            "--candidate-image-inspect",
            "--rollback-image-inspect",
        ):
            self.assertIn(flag, completed.stdout)
        self.assertIn("raw RC3-to-reconstructed-RC2 rollback", completed.stdout)

    def test_bad_invocations_fail_before_authenticated_gpu_supervision(self) -> None:
        missing = self._run()
        self.assertEqual(missing.returncode, 2)
        self.assertIn("usage:", missing.stderr)

        duplicate = self._run("--candidate-id", "one", "--candidate-id", "two")
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("may occur only once", duplicate.stderr)

        unknown = self._run("--unrecognized-option")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown option", unknown.stderr)

        legacy = self._run("--id=0")
        self.assertEqual(legacy.returncode, 2)
        self.assertIn("unknown option", legacy.stderr)

        profile_override = self._run("--configuration-profile", "stable-default")
        self.assertEqual(profile_override.returncode, 2)
        self.assertIn("unknown option", profile_override.stderr)

        internal = self._run("--gpu-lock-supervised")
        self.assertEqual(internal.returncode, 2)
        self.assertIn("supervisor PID was not authenticated", internal.stderr)

    def test_checked_isolated_python_helpers_offer_their_public_help(self) -> None:
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
            "prepare_rc3_rollback_evidence_v1.py",
            "capture_c02_config_endpoint_observation_v1.py",
            "capture_rc3_rollback_phase_v1.py",
            "capture_c02_raw_soak_scenarios_v1.py",
            "verify_c02_lifecycle_launch_inputs_v1.py",
        ):
            completed = subprocess.run(
                ["/usr/bin/python3", "-B", "-I", "-S", "-c", bridge, str(RELEASE_DIRECTORY / name), "--help"],
                text=True,
                capture_output=True,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            )
            self.assertEqual(completed.returncode, 0, f"{name}: {completed.stderr}")

    def test_runner_retains_authenticated_raw_only_boundary(self) -> None:
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
            "RILEY_RC3_ROLLBACK_SUPERVISOR_LOCK_FD",
            "/proc/$PPID/fd/$SUPERVISOR_LOCK_FD",
            "FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE",
            "exec 9>&-",
            "runpy.run_path",
            "sys.path.insert(0, directory)",
            'nvidia-smi -i "$gpu_index"',
            "--query-gpu=uuid,memory.used",
            "observed_memory <= max_gpu_memory_mib",
            "prepare_rc3_rollback_evidence_v1.py",
            "materialize_rc3_rollback_candidate_config_v1.py",
            "_initialize_candidate_config_directory",
            "_materialize_candidate_config_bridge",
            "run_remote_c02_config_endpoint_observation_v1.sh",
            "capture_rc3_rollback_phase_v1.py",
            "run_remote_c02_raw_soak_scenarios_v1.sh",
            "validate_candidate_scenario_contract",
            "finalize_rc3_rollback_finalizer_receipt_v1.py",
            "PreparationRequest(",
            "_finalize_authenticated_rollback_raw_once",
            "os._exit(0)",
            "require_current_server 'configuration observation'",
            "require_current_server 'candidate phase capture'",
            "require_current_server 'candidate serial capture'",
            "require_current_server 'rollback phase capture'",
            "--c02-configuration-profile \"$STABLE_DEFAULT_PROFILE\"",
            "--c02-startup-artifact \"$evidence_root/config/startup.json\"",
            "--c02-audit-dir \"$evidence_root/$SOURCE_AUDIT_DIRECTORY\"",
            "--c02-shutdown-artifact \"$evidence_root/$SOURCE_AUDIT_DIRECTORY/shutdown.json\"",
            "--noproxy '*'",
            "qualification verdict",
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
            "verify_c02_lifecycle_shutdown_v1.py",
            "write_rc3_rollback",
            "bind_raw_rc3_rollback",
            "check_rc3_qualification.py",
            "check_release_candidate.py",
        ):
            self.assertNotIn(forbidden, source)

    def test_capture_order_and_phase_grammar_are_fixed(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        static_prepare = source.index(
            'run_python "$repo_root/ci/release/prepare_rc3_rollback_evidence_v1.py"'
        )
        config_init = source.index("\nrun_private_config_initializer\n", static_prepare)
        candidate_launch = source.index("\nlaunch_candidate_server\n", config_init)
        config_observation = source.index(
            'run_remote_c02_config_endpoint_observation_v1.sh',
            candidate_launch,
        )
        config_materialization = source.index(
            "configuration_sha256=$(run_private_config_materializer)",
            config_observation,
        )
        candidate_phase = source.index(
            '--capture-name "$CANDIDATE_PHASE_CAPTURE_NAME"',
            config_materialization,
        )
        serial_capture = source.index(
            'run_remote_c02_raw_soak_scenarios_v1.sh',
            candidate_phase,
        )
        candidate_shutdown = source.index(
            "\nshutdown_server_successfully\n",
            serial_capture,
        )
        rollback_launch = source.index("\nlaunch_rollback_server\n", candidate_shutdown)
        rollback_phase = source.index(
            '--capture-name "$ROLLBACK_PHASE_CAPTURE_NAME"',
            rollback_launch,
        )
        rollback_shutdown = source.index(
            "\nshutdown_server_successfully\n",
            rollback_phase,
        )
        terminal_finalizer = source.index(
            "\nrun_private_rollback_finalizer\n",
            rollback_shutdown,
        )
        self.assertEqual(
            [
                static_prepare,
                config_init,
                candidate_launch,
                config_observation,
                config_materialization,
                candidate_phase,
                serial_capture,
                candidate_shutdown,
                rollback_launch,
                rollback_phase,
                rollback_shutdown,
                terminal_finalizer,
            ],
            sorted(
                [
                    static_prepare,
                    config_init,
                    candidate_launch,
                    config_observation,
                    config_materialization,
                    candidate_phase,
                    serial_capture,
                    candidate_shutdown,
                    rollback_launch,
                    rollback_phase,
                    rollback_shutdown,
                    terminal_finalizer,
                ]
            ),
        )
        candidate_block = source[candidate_phase:serial_capture]
        self.assertNotIn("--generation-request", candidate_block)
        rollback_block = source[rollback_phase:rollback_shutdown]
        self.assertIn("--generation-request \"$rollback_generation_request\"", rollback_block)
        self.assertIn("remove_scratch_for_terminal\nrun_private_rollback_finalizer", source)
        self.assertLess(
            source.index("validate_candidate_scenario_contract\n\npreflight_gpu"),
            static_prepare,
        )

    def test_runner_owns_config_profile_and_argument_environment_controls(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("readonly STABLE_DEFAULT_PROFILE='stable-default'", source)
        self.assertIn("--candidate-port and --rollback-port must differ", source)
        candidate_arguments = source[
            source.index("load_candidate_arguments() {") : source.index("load_rollback_arguments() {")
        ]
        rollback_arguments = source[
            source.index("load_rollback_arguments() {") : source.index("load_environment_file() {")
        ]
        self.assertIn("--c02-*", candidate_arguments)
        self.assertIn("--c02-*", rollback_arguments)
        self.assertIn("RILEY_C02_*", source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)


if __name__ == "__main__":
    unittest.main()
