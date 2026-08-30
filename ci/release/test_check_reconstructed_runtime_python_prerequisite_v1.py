#!/usr/bin/env python3
"""CPU-only hostile-path tests for the reconstructed-runtime Python preflight."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIRECTORY = Path(__file__).resolve().parent
ISOLATED_PYTHON = ("/usr/bin/python3.10", "-I", "-S", "-E", "-B")
sys.path.insert(0, os.fspath(RELEASE_DIRECTORY))

import check_reconstructed_runtime_python_prerequisite_v1 as checker  # noqa: E402


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_probe(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "dont_write_bytecode": True,
        "ignore_environment": True,
        "implementation": checker.EXPECTED_IMPLEMENTATION,
        "isolated": True,
        "machine": checker.EXPECTED_MACHINE,
        "no_site": True,
        "no_user_site": True,
        "platform": checker.EXPECTED_PLATFORM,
        "tomllib": True,
        "version": checker.EXPECTED_VERSION,
    }
    result.update(overrides)
    return result


class ReconstructedRuntimePythonPrerequisiteTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        # Synthetic executable fixtures use tempfile.  The production policy
        # rejects that location; the dedicated hostile test below restores the
        # actual volatile-prefix list to cover that boundary.
        self._production_unstable_prefixes = checker.UNSTABLE_PREFIXES
        self._fixture_prefix_patch = mock.patch.object(checker, "UNSTABLE_PREFIXES", ())
        self._fixture_prefix_patch.start()

    def tearDown(self) -> None:
        self._fixture_prefix_patch.stop()

    def _make_python(self, root: Path, *, mode: int = 0o500) -> tuple[Path, str]:
        directory = root / "toolchain" / "cpython-3.13.15" / "bin"
        directory.mkdir(parents=True)
        path = directory / "python3.13"
        contents = b"pinned-cpython-fixture\n"
        path.write_bytes(contents)
        os.chmod(path, mode)
        return path, digest(contents)

    def _completed_probe(self, document: dict[str, object] | None = None, *, stderr: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout=checker.canonical_json_bytes(valid_probe() if document is None else document),
            stderr=stderr,
        )

    def _check_with_probe(
        self,
        path: Path,
        expected_hash: str,
        completed: subprocess.CompletedProcess[bytes],
    ) -> tuple[dict[str, object], mock.Mock]:
        with mock.patch.object(checker, "EXPECTED_PYTHON_SHA256", expected_hash), mock.patch.object(
            checker, "_launch_held_probe", return_value=object()
        ) as launch, mock.patch.object(
            checker,
            "_collect_probe_output",
            return_value=(completed.returncode, completed.stdout, completed.stderr),
        ):
            result = checker.check_reconstructed_runtime_python_prerequisite(path)
        return result, launch

    def assert_reason(self, expected: str, action: object) -> None:
        with self.assertRaises(checker.ReconstructedRuntimePythonPrerequisiteError) as raised:
            action()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), expected)

    def test_pinned_held_python_runs_descriptor_path_isolated_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, expected_hash = self._make_python(Path(directory))
            result, launch = self._check_with_probe(path, expected_hash, self._completed_probe())
        self.assertEqual(result["schema_version"], checker.PREREQUISITE_VERSION)
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["python"]["sha256"], expected_hash)  # type: ignore[index]
        self.assertEqual(result["not_established"], checker.NOT_ESTABLISHED)
        launch.assert_called_once()
        descriptor = launch.call_args.args[0]
        self.assertIsInstance(descriptor, int)
        argv = checker._probe_argv(descriptor)
        self.assertRegex(argv[0], r"^/proc/self/fd/[0-9]+$")
        self.assertEqual(argv[1:5], ["-I", "-S", "-E", "-B"])
        self.assertEqual(argv[5], "-c")
        self.assertEqual(argv[6], checker.PROBE_PROGRAM)
        self.assertIn("sys.stdout.write", checker.PROBE_PROGRAM)
        self.assertNotIn("print(", checker.PROBE_PROGRAM)

    def test_held_probe_launches_with_only_fixed_isolated_inputs(self) -> None:
        process = object()
        with mock.patch.object(checker.subprocess, "Popen", return_value=process) as popen:
            self.assertIs(checker._launch_held_probe(37), process)
        invoked = popen.call_args.kwargs
        self.assertEqual(popen.call_args.args[0], checker._probe_argv(37))
        self.assertIs(invoked["stdin"], checker.subprocess.DEVNULL)
        self.assertIs(invoked["stdout"], checker.subprocess.PIPE)
        self.assertIs(invoked["stderr"], checker.subprocess.PIPE)
        self.assertEqual(invoked["env"], checker.PROBE_ENVIRONMENT)
        self.assertEqual(invoked["cwd"], "/")
        self.assertTrue(invoked["close_fds"])
        self.assertEqual(invoked["pass_fds"], (37,))
        self.assertTrue(invoked["start_new_session"])

    def test_held_executable_descriptor_never_collides_with_child_stdio(self) -> None:
        with mock.patch.object(checker.fcntl, "fcntl", return_value=37) as duplicate, mock.patch.object(
            checker, "_close_quietly"
        ) as close:
            self.assertEqual(checker._move_held_descriptor_above_standard_streams(0), 37)
        duplicate.assert_called_once_with(0, checker.fcntl.F_DUPFD_CLOEXEC, 3)
        close.assert_called_once_with(0)

        with mock.patch.object(checker.fcntl, "fcntl") as duplicate:
            self.assertEqual(checker._move_held_descriptor_above_standard_streams(3), 3)
        duplicate.assert_not_called()

    def test_hash_mismatch_rejects_before_any_probe_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _expected_hash = self._make_python(Path(directory))
            with mock.patch.object(checker, "_launch_held_probe") as launch:
                self.assert_reason(
                    "python-sha256-mismatch",
                    lambda: checker.check_reconstructed_runtime_python_prerequisite(path),
                )
        launch.assert_not_called()

    def test_oversized_executable_rejects_before_hash_or_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _expected_hash = self._make_python(Path(directory))
            with mock.patch.object(checker, "MAX_PYTHON_EXECUTABLE_BYTES", 1), mock.patch.object(
                checker, "_sha256_held_file"
            ) as hash_file, mock.patch.object(checker, "_launch_held_probe") as launch:
                self.assert_reason(
                    "python-executable-too-large",
                    lambda: checker.check_reconstructed_runtime_python_prerequisite(path),
                )
        hash_file.assert_not_called()
        launch.assert_not_called()

    def test_symlink_nonregular_mode_and_hardlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, expected_hash = self._make_python(root)
            link = root / "toolchain-link"
            link.symlink_to(path)
            self.assert_reason(
                "unsafe-python-executable",
                lambda: checker.check_reconstructed_runtime_python_prerequisite(link),
            )

            nonexecutable, nonexecutable_hash = self._make_python(root / "noexec", mode=0o400)
            self.assert_reason(
                "nonexecutable-python",
                lambda: self._check_with_probe(nonexecutable, nonexecutable_hash, self._completed_probe()),
            )

            os.link(path, root / "second-name")
            self.assert_reason(
                "nonunique-python-executable",
                lambda: self._check_with_probe(path, expected_hash, self._completed_probe()),
            )

    def test_invalid_path_and_volatile_path_fail_closed(self) -> None:
        self.assert_reason(
            "invalid-python-path",
            lambda: checker.check_reconstructed_runtime_python_prerequisite(Path("relative/python")),
        )
        with mock.patch.object(checker, "UNSTABLE_PREFIXES", self._production_unstable_prefixes):
            self.assert_reason(
                "unstable-python-path",
                lambda: checker.check_reconstructed_runtime_python_prerequisite(Path("/tmp/python3.13")),
            )
        self.assert_reason(
            "python-path-inside-source-checkout",
            lambda: checker.check_reconstructed_runtime_python_prerequisite(Path(checker.__file__)),
        )

    def test_timeout_nonzero_stderr_and_probe_contract_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, expected_hash = self._make_python(Path(directory))
            timeout = checker.ReconstructedRuntimePythonPrerequisiteError("fixture timeout")
            timeout.reason_code = "python-probe-timeout"  # type: ignore[attr-defined]
            with mock.patch.object(checker, "EXPECTED_PYTHON_SHA256", expected_hash), mock.patch.object(
                checker, "_launch_held_probe", return_value=object()
            ), mock.patch.object(checker, "_collect_probe_output", side_effect=timeout):
                self.assert_reason(
                    "python-probe-timeout",
                    lambda: checker.check_reconstructed_runtime_python_prerequisite(path),
                )
            for name, completed, expected in (
                ("nonzero", self._completed_probe(returncode=1), "python-probe-failed"),
                ("stderr", self._completed_probe(stderr=b"warning\n"), "unexpected-probe-stderr"),
                ("version", self._completed_probe(valid_probe(version="3.10.12")), "python-probe-mismatch"),
                ("tomllib", self._completed_probe(valid_probe(tomllib=False)), "python-probe-mismatch"),
            ):
                with self.subTest(name=name):
                    with mock.patch.object(checker, "EXPECTED_PYTHON_SHA256", expected_hash), mock.patch.object(
                        checker, "_launch_held_probe", return_value=object()
                    ), mock.patch.object(
                        checker,
                        "_collect_probe_output",
                        return_value=(completed.returncode, completed.stdout, completed.stderr),
                    ):
                        self.assert_reason(
                            expected,
                            lambda: checker.check_reconstructed_runtime_python_prerequisite(path),
                        )

    def test_probe_json_must_be_bounded_closed_and_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, expected_hash = self._make_python(Path(directory))
            malformed = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"{}", stderr=b"")
            with mock.patch.object(checker, "EXPECTED_PYTHON_SHA256", expected_hash), mock.patch.object(
                checker, "_launch_held_probe", return_value=object()
            ), mock.patch.object(
                checker,
                "_collect_probe_output",
                return_value=(malformed.returncode, malformed.stdout, malformed.stderr),
            ):
                self.assert_reason(
                    "invalid-probe-schema",
                    lambda: checker.check_reconstructed_runtime_python_prerequisite(path),
                )
            noncanonical = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b'{"version":"3.13.15"}', stderr=b""
            )
            with mock.patch.object(checker, "EXPECTED_PYTHON_SHA256", expected_hash), mock.patch.object(
                checker, "_launch_held_probe", return_value=object()
            ), mock.patch.object(
                checker,
                "_collect_probe_output",
                return_value=(noncanonical.returncode, noncanonical.stdout, noncanonical.stderr),
            ):
                self.assert_reason(
                    "invalid-probe-schema",
                    lambda: checker.check_reconstructed_runtime_python_prerequisite(path),
                )
            trailing_newline = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=checker.canonical_json_bytes(valid_probe()) + b"\n",
                stderr=b"",
            )
            with mock.patch.object(checker, "EXPECTED_PYTHON_SHA256", expected_hash), mock.patch.object(
                checker, "_launch_held_probe", return_value=object()
            ), mock.patch.object(
                checker,
                "_collect_probe_output",
                return_value=(trailing_newline.returncode, trailing_newline.stdout, trailing_newline.stderr),
            ):
                self.assert_reason(
                    "noncanonical-probe-json",
                    lambda: checker.check_reconstructed_runtime_python_prerequisite(path),
                )

    def test_probe_reader_bounds_pipes_kills_groups_and_rejects_deep_json(self) -> None:
        oversized = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-E",
                "-B",
                "-c",
                "import sys;sys.stdout.write('x'*20000)",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.assert_reason("python-probe-output-too-large", lambda: checker._collect_probe_output(oversized))

        delayed = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-E",
                "-B",
                "-c",
                "import time;time.sleep(60)",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        with mock.patch.object(checker, "PROBE_TIMEOUT_SECONDS", 0.25):
            self.assert_reason("python-probe-timeout", lambda: checker._collect_probe_output(delayed))

        exited_leader = mock.Mock()
        exited_leader.pid = 31415
        exited_leader.poll.return_value = 0
        exited_leader.wait.return_value = 0
        with mock.patch.object(checker.os, "killpg") as killpg:
            checker._terminate_probe_group(exited_leader)
        killpg.assert_called_once_with(31415, checker.signal.SIGKILL)

        interrupted_process = mock.Mock()
        interrupted_process.stdout = mock.Mock()
        interrupted_process.stderr = mock.Mock()
        interrupted_selector = mock.Mock()
        interrupted_selector.get_map.return_value = {"stdout": object()}
        interrupted_selector.select.side_effect = KeyboardInterrupt
        with mock.patch.object(checker.os, "set_blocking"), mock.patch.object(
            checker.selectors, "DefaultSelector", return_value=interrupted_selector
        ), mock.patch.object(checker, "_terminate_probe_group") as terminate:
            with self.assertRaises(KeyboardInterrupt):
                checker._collect_probe_output(interrupted_process)
        terminate.assert_called_once_with(interrupted_process)
        interrupted_selector.close.assert_called_once()

        bounded_process = mock.Mock()
        bounded_process.stdout = mock.Mock()
        bounded_process.stdout.fileno.return_value = 41
        bounded_process.stderr = mock.Mock()
        bounded_process.stderr.fileno.return_value = 42
        bounded_key = mock.Mock()
        bounded_key.fileobj = bounded_process.stdout
        bounded_key.data = "stdout"
        bounded_selector = mock.Mock()
        bounded_selector.get_map.return_value = {"stdout": object()}
        bounded_selector.select.return_value = [(bounded_key, checker.selectors.EVENT_READ)]
        with mock.patch.object(checker, "MAX_PROBE_BYTES", 4), mock.patch.object(
            checker.os, "set_blocking"
        ), mock.patch.object(
            checker.selectors, "DefaultSelector", return_value=bounded_selector
        ), mock.patch.object(checker.os, "read", return_value=b"xxxxx") as read, mock.patch.object(
            checker, "_terminate_probe_group"
        ) as terminate:
            self.assert_reason("python-probe-output-too-large", lambda: checker._collect_probe_output(bounded_process))
        read.assert_called_once_with(41, 5)
        terminate.assert_called_once_with(bounded_process)

        deep_json = b"[" * 8000 + b"]" * 8000
        self.assert_reason("probe-json-nesting-too-deep", lambda: checker._parse_probe(deep_json))

    def test_main_writes_canonical_stdout_without_a_trailing_newline(self) -> None:
        expected = {"checked": True}
        output = io.StringIO()
        with mock.patch.object(checker, "_require_controller_isolation"), mock.patch.object(
            checker, "check_reconstructed_runtime_python_prerequisite", return_value=expected
        ), mock.patch.object(checker.sys, "stdout", output):
            self.assertEqual(checker.main(["--python", "/external/python3.13"]), 0)
        self.assertEqual(output.getvalue(), checker.canonical_json_bytes(expected).decode("utf-8"))

    def test_public_cli_is_isolated_and_never_executes_ambient_python(self) -> None:
        runner = REPOSITORY_ROOT / "ci" / "release" / "check_reconstructed_runtime_python_prerequisite_v1.py"
        help_result = subprocess.run(
            [*ISOLATED_PYTHON, os.fspath(runner), "--help"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--python", help_result.stdout)

        unrecognized = subprocess.run(
            [*ISOLATED_PYTHON, os.fspath(runner), "--python", "/usr/bin/python3.10", "--id=0"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(unrecognized.returncode, 2)
        self.assertIn("unrecognized arguments", unrecognized.stderr)

        ambient = subprocess.run(
            [*ISOLATED_PYTHON, os.fspath(runner), "--python", "/usr/bin/python3.10"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(ambient.returncode, 2)
        self.assertIn("python-sha256-mismatch", ambient.stderr)

    def test_static_surface_has_no_install_or_operational_actions(self) -> None:
        source = (REPOSITORY_ROOT / "ci" / "release" / "check_reconstructed_runtime_python_prerequisite_v1.py").read_text(
            encoding="utf-8"
        )
        for required in (
            '"O_NOFOLLOW"',
            '"O_DIRECTORY"',
            '"O_CLOEXEC"',
            '"O_NONBLOCK"',
            "/proc/self/fd/",
            '"-I"',
            '"-S"',
            '"-E"',
            '"-B"',
            "tomllib",
            "EXPECTED_PYTHON_SHA256",
            "MAX_PYTHON_EXECUTABLE_BYTES",
            "F_DUPFD_CLOEXEC",
            "remaining_capacity + 1",
            "sys.stdout.write",
            "start_new_session=True",
            "MAX_PROBE_STDERR_BYTES",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "urllib",
            "requests",
            "uv python install",
            "pip install",
            "docker ",
            "nvidia-smi",
            "socket.",
            "subprocess.run",
            "check_reconstructed_runtime_a_b_materialization",
        ):
            self.assertNotIn(forbidden, source)
if __name__ == "__main__":
    unittest.main()
