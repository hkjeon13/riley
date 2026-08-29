#!/usr/bin/env python3
"""Static contract checks for the C02-P0 remote raw-capture producer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "ci/release/run_remote_c02_runtime_config_capture.sh"


class RemoteC02RuntimeConfigCaptureTests(unittest.TestCase):
    @staticmethod
    def _embedded_python(function_name: str) -> str:
        source = RUNNER.read_text(encoding="utf-8")
        function_start = source.index(f"{function_name}() {{")
        heredoc_start = source.index("<<'PY'\n", function_start) + len("<<'PY'\n")
        heredoc_end = source.index("\nPY\n}", heredoc_start)
        return source[heredoc_start:heredoc_end]

    def _run_embedded_python(self, function_name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", "-I", "-S", "-", *arguments],
            input=self._embedded_python(function_name),
            text=True,
            capture_output=True,
            check=False,
        )

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

    def test_embedded_copy_is_private_create_only_and_rejects_unsafe_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            source = root / "source.json"
            destination = root / "captured.json"
            source.write_bytes(b'{"captured":true}')

            completed = self._run_embedded_python(
                "copy_create_only", str(source), str(destination)
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            metadata = destination.lstat()
            self.assertEqual(metadata.st_mode & 0o777, 0o600)
            self.assertEqual(metadata.st_nlink, 1)

            collision = self._run_embedded_python(
                "copy_create_only", str(source), str(destination)
            )
            self.assertNotEqual(collision.returncode, 0)
            self.assertEqual(destination.read_bytes(), b'{"captured":true}')

            linked_source = root / "source-link.json"
            linked_destination = root / "linked-captured.json"
            os.symlink(source, linked_source)
            linked = self._run_embedded_python(
                "copy_create_only", str(linked_source), str(linked_destination)
            )
            self.assertNotEqual(linked.returncode, 0)
            self.assertFalse(linked_destination.exists())

            unsafe_parent = root / "unsafe-parent"
            unsafe_parent.mkdir(mode=0o700)
            os.chmod(unsafe_parent, 0o755)
            unsafe = self._run_embedded_python(
                "copy_create_only", str(source), str(unsafe_parent / "captured.json")
            )
            self.assertNotEqual(unsafe.returncode, 0)

    def test_embedded_container_receipt_is_private_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            destination = root / "container-runtime.json"
            arguments = (
                str(destination),
                "riley-native-cuda:test",
                "sha256:" + "a" * 64,
                "/workspace/target/release/riley",
                "b" * 64,
                "c" * 64,
                "3",
                "GPU-01234567-89ab-cdef-0123-456789abcdef",
                "1000:1000",
            )

            completed = self._run_embedded_python(
                "write_container_runtime_receipt", *arguments
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(destination.read_text(encoding="ascii"))
            self.assertEqual(payload["container_image_id"], arguments[2])
            self.assertEqual(payload["host_gpu_index"], 3)
            metadata = destination.lstat()
            self.assertEqual(metadata.st_mode & 0o777, 0o600)
            self.assertEqual(metadata.st_nlink, 1)

            collision = self._run_embedded_python(
                "write_container_runtime_receipt", *arguments
            )
            self.assertNotEqual(collision.returncode, 0)

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
            "O_NONBLOCK",
            "O_DIRECTORY",
            "MAX_RAW_CAPTURE_BYTES=$((8 * 1024 * 1024))",
            "--max-filesize \"$MAX_RAW_CAPTURE_BYTES\"",
            "MAX_CAPTURE_BYTES = 8 * 1024 * 1024",
            "source_fd = os.open(source, source_flags)",
            "os.read(source_fd",
            "source_after = os.fstat(source_fd)",
            "source_path_after = os.lstat(source)",
            "dir_fd=destination_parent_fd",
            "os.lstat(destination_name, dir_fd=destination_parent_fd)",
            "os.fsync(destination_parent_fd)",
            "dir_fd=parent_fd",
            "os.lstat(destination_name, dir_fd=parent_fd)",
            "os.fsync(parent_fd)",
            "os.fchmod(destination_fd, 0o600)",
            "os.fchmod(fd, 0o600)",
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
        self.assertNotIn('getattr(os, "O_NOFOLLOW", 0)', source)
        self.assertNotIn('with open(source, "rb", buffering=0)', source)


if __name__ == "__main__":
    unittest.main()
