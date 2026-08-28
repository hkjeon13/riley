#!/usr/bin/env python3
"""Static contract checks for the C02-P0 remote raw-capture producer."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "ci/release/run_remote_c02_runtime_config_capture.sh"


class RemoteC02RuntimeConfigCaptureTests(unittest.TestCase):
    def test_runner_has_valid_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    def test_help_is_available_without_gpu_or_model_inputs(self) -> None:
        environment = dict(os.environ)
        environment["RILEY_C02_CAPTURE_CLEAN_ENV"] = "1"
        completed = subprocess.run(
            ["bash", str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--container-image", completed.stdout)
        self.assertIn("--container-binary", completed.stdout)
        self.assertIn("--stable-args-file", completed.stdout)
        self.assertIn("--max-env-file", completed.stdout)
        self.assertIn("--max-gpu-memory-mib 256", completed.stdout)

    def test_missing_inputs_fail_before_any_gpu_or_evidence_action(self) -> None:
        environment = dict(os.environ)
        environment["RILEY_C02_CAPTURE_CLEAN_ENV"] = "1"
        completed = subprocess.run(
            ["bash", str(RUNNER)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("usage:", completed.stderr)

    def test_mutually_exclusive_launch_modes_fail_before_gpu_or_evidence_action(self) -> None:
        environment = dict(os.environ)
        environment["RILEY_C02_CAPTURE_CLEAN_ENV"] = "1"
        completed = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--binary",
                "/tmp/riley-host",
                "--container-image",
                "riley-native-cuda:test",
                "--container-binary",
                "/workspace/target/release/riley",
            ],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("choose exactly one launch mode", completed.stderr)

    def test_incomplete_container_mode_fails_before_gpu_or_evidence_action(self) -> None:
        environment = dict(os.environ)
        environment["RILEY_C02_CAPTURE_CLEAN_ENV"] = "1"
        completed = subprocess.run(
            [
                "bash",
                str(RUNNER),
                "--container-image",
                "riley-native-cuda:test",
            ],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("choose exactly one launch mode", completed.stderr)

    def test_runner_retains_the_c02_raw_capture_safety_contract(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        required_fragments = (
            "DEFAULT_MAX_GPU_MEMORY_MIB=256",
            'RILEY_C02_CAPTURE_CLEAN_ENV=1',
            "/usr/bin/env -i",
            "GPU_LOCK_PATH='/var/tmp/riley-server-4096-gpu-evidence.lock'",
            '-i "$gpu_index"',
            "observed_gpu_memory <= max_gpu_memory_mib",
            "--evidence-dir must be outside the source tree",
            "copy_create_only",
            "os.O_EXCL",
            "O_NOFOLLOW",
            "/usr/bin/python3 -B -I -S",
            "validate_raw_c02_runtime_config.py",
            "--server-startup-artifact",
            "--noproxy '*'",
            "/v1/config",
            "--c02-candidate-id",
            "--c02-configuration-profile",
            "--c02-startup-artifact",
            "two C02-P0 arms resolved the same effective_config",
            "no C02 qualification was run",
            "--container-image",
            "--container-binary",
            "docker image inspect --format '{{.Id}}'",
            "--pull=never",
            "in-image --container-binary SHA-256 does not match --binary-sha256",
            "write_container_runtime_receipt",
            "container_image_id",
            "--gpus \"device=${gpu_index}\"",
            "--network host",
            "type=bind,src=$model_dir,dst=$CONTAINER_MODEL_DIR,readonly",
            "type=bind,src=$server_output_dir,dst=$CONTAINER_SERVER_OUTPUT_DIR",
            "--entrypoint /usr/bin/env",
            "-i \"${loaded_environment[@]}\"",
            "--device \"$CONTAINER_VISIBLE_GPU_INDEX\"",
            "--bind \"127.0.0.1:${port}\"",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, source)
        self.assertNotIn("sudo ", source)
        self.assertNotIn("systemctl ", source)
        self.assertNotIn("--network bridge", source)
        self.assertNotIn("--publish ", source)
        self.assertNotIn("0.0.0.0:${port}", source)
        self.assertNotIn("check_rc3_qualification.py", source)
        self.assertNotIn("import check_effective_runtime_config_receipt", source)


if __name__ == "__main__":
    unittest.main()
