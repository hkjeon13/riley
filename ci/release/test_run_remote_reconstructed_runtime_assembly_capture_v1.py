#!/usr/bin/env python3
"""Static, CPU-only guards for the raw source-free assembly host runner."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "ci/release/run_remote_reconstructed_runtime_assembly_capture_v1.sh"
RELEASE_DIRECTORY = REPOSITORY_ROOT / "ci/release"


class RunRemoteReconstructedRuntimeAssemblyCaptureTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(RUNNER), *arguments],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )

    def test_bash_syntax_and_help_have_no_docker_or_lock_side_effect(self) -> None:
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for fragment in (
            "--reconstruction-id",
            "--source-revision",
            "--expected-source-archive-sha256",
            "--repro-build-inputs-sha256",
            "--release-binary",
            "--release-bundle",
            "--evidence-dir",
            "never start",
            "Python 3.11+",
        ):
            self.assertIn(fragment, completed.stdout)

    def test_bad_invocations_fail_before_the_authenticated_docker_supervisor(self) -> None:
        missing = self._run()
        self.assertEqual(missing.returncode, 2)
        self.assertIn("usage:", missing.stderr)

        duplicate = self._run("--reconstruction-id", "a", "--reconstruction-id", "b")
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("may occur only once", duplicate.stderr)

        unknown = self._run("--unrecognized-option")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown option", unknown.stderr)

        legacy_shortcut = self._run("--id=0")
        self.assertEqual(legacy_shortcut.returncode, 2)
        self.assertIn("unknown option", legacy_shortcut.stderr)

        forged_sentinel = self._run("--assembly-lock-supervised")
        self.assertEqual(forged_sentinel.returncode, 2)
        self.assertIn("supervisor PID was not authenticated", forged_sentinel.stderr)

    def test_isolated_python_helpers_are_python_310_compatible_at_help_time(self) -> None:
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
            "compose_reconstructed_runtime_assembly_capture_v1.py",
            "prepare_reconstructed_runtime_image_export_oci_normalization_v1.py",
            "verify_reconstructed_runtime_assembly_dockerfile.py",
        ):
            completed = subprocess.run(
                [sys.executable, "-B", "-I", "-S", "-c", bridge, str(RELEASE_DIRECTORY / name), "--help"],
                text=True,
                capture_output=True,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(completed.returncode, 0, f"{name}: {completed.stderr}")

    def test_runner_retains_the_closed_never_started_docker_contract(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        required_fragments = (
            "set -euo pipefail",
            "set -o noclobber",
            "umask 077",
            "--assembly-lock-supervised",
            "os.O_NOFOLLOW",
            "os.O_NONBLOCK",
            "os.O_CLOEXEC",
            "fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)",
            "PR_SET_PDEATHSIG",
            "os.setsid()",
            "os.pipe2(os.O_CLOEXEC)",
            "os.killpg(child_pid, signum)",
            'os.listdir("/proc/self/fd")',
            "if fd > 2 and fd not in {lock_fd, ready_write}",
            "RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_LOCK_FD",
            "/proc/${PPID}/fd/${SUPERVISOR_LOCK_FD}",
            "FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE",
            "exec 9>&-",
            "env -i",
            "PYTHONDONTWRITEBYTECODE=1",
            "terminate_direct_children",
            "/proc/$$/task/$$/children",
            "initialize_reconstructed_runtime_assembly_evidence_v1.py",
            'run_python "$evidence_initializer" --evidence-dir',
            "PINNED_RUNTIME=",
            "reviewed pinned runtime base is not present locally",
            "--print-source-sha256",
            "--dockerfile-sha256",
            "docker build",
            "--file Dockerfile",
            "--platform linux/amd64",
            "--network none",
            "--pull=false",
            "--no-cache",
            "--iidfile",
            "RILEY_RECONSTRUCTION_ID",
            "RILEY_SOURCE_REVISION",
            "RILEY_SOURCE_ARCHIVE_SHA256",
            "RILEY_REPRO_BUILD_INPUTS_SHA256",
            "RILEY_RELEASE_BINARY_SHA256",
            "RILEY_RELEASE_BUNDLE_SHA256",
            "RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256",
            "docker image inspect",
            "docker image save",
            "prepare_reconstructed_runtime_image_export_oci_normalization_v1.py",
            "docker create --network none --restart no",
            "docker container inspect",
            "docker cp",
            "container-inspect-after.json",
            "cmp --silent",
            "compose_reconstructed_runtime_assembly_capture_v1.py",
            " read-id --kind image ",
            " read-id --kind container ",
            " stream ",
            "MAX_IMAGE_EXPORT_ARCHIVE_BYTES",
            "runtime-tree",
            "assembly-capture.tar",
            "verify_release_inputs pre-build",
            "verify_release_inputs post-capture",
            "qualification passed",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, source)
        # Build logs intentionally preserve merged stdout/stderr. Structured
        # JSON/tar/ID leaves must receive Docker stdout only, otherwise a
        # benign warning corrupts bytes consumed by the strict replayers.
        self.assertEqual(source.count("2>&1 | run_python"), 1)
        for forbidden in (
            "docker run",
            "docker start",
            "docker exec",
            "--gpus",
            "--privileged",
            "--mount",
            "--secret",
            "--ssh",
            "--network host",
            "--pid host",
            "--ipc host",
            "--uts host",
            "--userns host",
            "docker tag",
            "docker pull ",
            "docker image rm",
            "nvidia-smi",
            "ssh ",
            "sudo ",
            "systemctl ",
            "eval ",
            "bash -c",
            "--id=",
            'image_id=$(/bin/cat',
            "prepare_reconstructed_runtime_assembly_capture_v1.py",
            "prepare_reconstructed_runtime_oci_inputs_v1.py",
            "prepare_reconstructed_runtime_image_export_assembly_content_bridge_v1.py",
        ):
            self.assertNotIn(forbidden, source)

    def test_build_export_normalization_create_and_composition_order_is_fixed(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        initialization = source.index('run_python "$evidence_initializer" --evidence-dir')
        base_preflight = source.index('docker image inspect "$PINNED_RUNTIME"')
        dockerfile_digest = source.index("dockerfile_sha256=$(verify_static_recipe)")
        context = source.index('run_python "$composer" context')
        build = source.index("docker build")
        image_inspect = source.index('docker image inspect "$image_id"', build)
        image_save = source.index("docker image save", image_inspect)
        normalization = source.index('run_python "$normalizer"', image_save)
        create = source.index("docker create --network none --restart no", normalization)
        first_inspect = source.index('docker container inspect "$container_id"', create)
        copy = source.index("docker cp", first_inspect)
        second_inspect = source.index('docker container inspect "$container_id"', copy)
        compare = source.index("cmp --silent", second_inspect)
        runtime_tree = source.index('run_python "$composer" runtime-tree', compare)
        capture = source.index('run_python "$composer" capture', runtime_tree)
        post_inputs = source.index("verify_release_inputs post-capture", capture)
        removal = source.index("docker container rm", post_inputs)
        self.assertEqual(
            [
                dockerfile_digest,
                initialization,
                base_preflight,
                context,
                build,
                image_inspect,
                image_save,
                normalization,
                create,
                first_inspect,
                copy,
                second_inspect,
                compare,
                runtime_tree,
                capture,
                post_inputs,
                removal,
            ],
            sorted(
                [
                    dockerfile_digest,
                    initialization,
                    base_preflight,
                    context,
                    build,
                    image_inspect,
                    image_save,
                    normalization,
                    create,
                    first_inspect,
                    copy,
                    second_inspect,
                    compare,
                    runtime_tree,
                    capture,
                    post_inputs,
                    removal,
                ]
            ),
        )
        self.assertEqual(source.count("--build-arg \"RILEY_"), 7)

    def test_recipe_hash_literal_tracks_the_reviewed_static_recipe_contract(self) -> None:
        sys.path.insert(0, str(RELEASE_DIRECTORY))
        import verify_reconstructed_runtime_assembly_dockerfile as recipe  # noqa: PLC0415

        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(f"RECIPE_NORMALIZED_INSTRUCTIONS_SHA256='{recipe.EXPECTED_NORMALIZED_INSTRUCTION_SHA256}'", source)
        self.assertIn(f"PINNED_RUNTIME='{recipe.PINNED_RUNTIME}'", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
