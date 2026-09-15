from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

import n01_repeat_control as control


FAKE_COMMAND = r"""
import os
from pathlib import Path
import sys
import time

counter_path = Path(os.environ["N01_REPEAT_TEST_COMMAND_COUNTER"])
count = int(counter_path.read_text(encoding="utf-8") if counter_path.exists() else "0") + 1
counter_path.write_text(str(count), encoding="utf-8")
proc_root = Path(os.environ["N01_REPEAT_TEST_PROC_ROOT"])
for offset, resource in enumerate(("cpu", "io", "memory")):
    value = count * 10 + offset
    (proc_root / "pressure" / resource).write_text(
        "some avg10={0}.00 avg60={0}.10 avg300={0}.20 total={1}\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total={2}\n".format(value, value * 100, value * 10),
        encoding="utf-8",
    )
print("fake stdout attempt {}".format(count))
print("fake stderr attempt {}".format(count), file=sys.stderr)
time.sleep(float(os.environ.get("N01_REPEAT_TEST_SLEEP_SECONDS", "0.002")))
failures = {int(value) for value in os.environ.get("N01_REPEAT_TEST_FAILURES", "").split(",") if value}
if count in failures:
    raise SystemExit(17)
"""


FAKE_NVIDIA_SMI = r"""
import os
from pathlib import Path
import sys

expected = ["--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits"]
if sys.argv[1:] != expected:
    print("unexpected nvidia-smi argv: {!r}".format(sys.argv[1:]), file=sys.stderr)
    raise SystemExit(71)
counter_path = Path(os.environ["N01_REPEAT_TEST_NVIDIA_COUNTER"])
count = int(counter_path.read_text(encoding="utf-8") if counter_path.exists() else "0") + 1
counter_path.write_text(str(count), encoding="utf-8")
print("0, GPU-test-uuid, Test GPU, {}, 24576".format(100 + count))
"""


SLEEP_COMMAND = r"""
import time
time.sleep(1)
"""


ENVIRONMENT_COMMAND = r"""
import json
import os
print(json.dumps({
    "phase": os.environ.get("N01_REPEAT_CONTROL_PHASE"),
    "index": os.environ.get("N01_REPEAT_CONTROL_INDEX"),
}, sort_keys=True))
"""


N06A_TIMEOUT_PARENT_COMMAND = r"""
import os
from pathlib import Path
import subprocess
import sys
import time

child = Path(os.environ["N01_N06A_TIMEOUT_CHILD"])
child_pid_path = Path(os.environ["N01_N06A_TIMEOUT_CHILD_PID"])
process = subprocess.Popen([sys.executable, str(child)], start_new_session=True)
child_pid_path.write_text(str(process.pid), encoding="utf-8")
while True:
    time.sleep(1)
"""


N06A_TIMEOUT_CHILD_COMMAND = r"""
import os
from pathlib import Path
import signal
import time

signal_path = Path(os.environ["N01_N06A_TIMEOUT_CHILD_SIGNAL"])

def handle_term(_signal, _frame):
    signal_path.write_text("SIGTERM", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_term)
while True:
    time.sleep(1)
"""


N06A_ABSENT_DOCKER = r"""
import json
import os
from pathlib import Path
import sys

Path(os.environ["N01_N06A_DOCKER_LOG"]).open("a", encoding="utf-8").write(
    json.dumps(sys.argv[1:]) + "\n"
)
arguments = sys.argv[1:]
if arguments[:3] == ["container", "ls", "--all"]:
    raise SystemExit(0)
if arguments[:2] == ["container", "inspect"]:
    raise SystemExit(1)
raise SystemExit(71)
"""


class N01RepeatControlTests(unittest.TestCase):
    def write_proc_snapshot(self, proc_root: Path, *, value: int = 1) -> None:
        (proc_root / "pressure").mkdir(parents=True)
        for offset, resource in enumerate(("cpu", "io", "memory")):
            resource_value = value + offset
            (proc_root / "pressure" / resource).write_text(
                "some avg10={0}.00 avg60={0}.10 avg300={0}.20 total={1}\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total={2}\n".format(
                    resource_value,
                    resource_value * 100,
                    resource_value * 10,
                ),
                encoding="utf-8",
            )
        (proc_root / "meminfo").write_text(
            "MemTotal:       64000 kB\nMemFree:        12000 kB\nMemAvailable:   30000 kB\n",
            encoding="utf-8",
        )

    def write_program(self, directory: Path, name: str, source: str) -> Path:
        program = directory / name
        program.write_text("#!/usr/bin/env python3\n" + source, encoding="utf-8")
        program.chmod(0o755)
        return program

    def make_config(
        self,
        directory: Path,
        *,
        command: tuple[str, ...],
        nvidia_smi: Path,
        proc_root: Path,
        output_name: str = "receipt",
        warmups: int = 1,
        repeats: int = 3,
        timeout_seconds: float = 5.0,
        n06a_timeout_cleanup: control.N06ATimeoutCleanupConfig | None = None,
    ) -> control.RepeatControlConfig:
        return control.RepeatControlConfig(
            output_dir=directory / output_name,
            cwd=directory,
            command=command,
            warmups=warmups,
            repeats=repeats,
            timeout_seconds=timeout_seconds,
            nvidia_smi=str(nvidia_smi),
            proc_root=proc_root,
            bootstrap_resamples=400,
            bootstrap_seed=99,
            receipt_name="receipt.json",
            n06a_timeout_cleanup=n06a_timeout_cleanup,
        )

    def test_retains_every_timed_attempt_with_pressure_and_gpu_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            proc_root = directory / "proc"
            self.write_proc_snapshot(proc_root)
            command = self.write_program(directory, "fake_command.py", FAKE_COMMAND)
            nvidia_smi = self.write_program(directory, "fake_nvidia_smi.py", FAKE_NVIDIA_SMI)
            environment = {
                "N01_REPEAT_TEST_COMMAND_COUNTER": str(directory / "command-counter.txt"),
                "N01_REPEAT_TEST_NVIDIA_COUNTER": str(directory / "nvidia-counter.txt"),
                "N01_REPEAT_TEST_PROC_ROOT": str(proc_root),
                "N01_REPEAT_TEST_FAILURES": "3",
            }
            config = self.make_config(
                directory,
                command=(sys.executable, str(command)),
                nvidia_smi=nvidia_smi,
                proc_root=proc_root,
            )
            with mock.patch.dict(os.environ, environment, clear=False):
                receipt = control.run_repeat_control(config)

            persisted = json.loads(Path(receipt["receipt_path"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt, persisted)
            self.assertEqual(receipt["status"], "completed-with-failures")
            self.assertEqual(len(receipt["warmup_runs"]), 1)
            self.assertEqual(len(receipt["timed_runs"]), 3)
            self.assertEqual(
                [run["status"] for run in receipt["timed_runs"]],
                ["succeeded", "failed", "succeeded"],
            )
            self.assertEqual(receipt["summary"]["count"], 3)
            self.assertEqual(receipt["summary"]["successful_count"], 2)
            self.assertEqual(receipt["summary"]["failures"], 1)
            self.assertIsNotNone(receipt["summary"]["median_ms"])
            self.assertEqual(
                receipt["summary"]["bootstrap_median_95_ci_ms"]["seed"],
                99,
            )
            self.assertIn("high-pressure runs are not discarded", receipt["retention_policy"]["host_pressure"])
            self.assertTrue(any("not a continuous high-water" in item for item in receipt["limitations"]))

            failed = receipt["timed_runs"][1]
            self.assertEqual(failed["exit_code"], 17)
            self.assertFalse(failed["timed_out"])
            self.assertIn("fake stdout attempt 3", Path(failed["stdout_path"]).read_text(encoding="utf-8"))
            self.assertIn("fake stderr attempt 3", Path(failed["stderr_path"]).read_text(encoding="utf-8"))

            for run in receipt["timed_runs"]:
                self.assertFalse(run["shell"])
                self.assertGreater(run["wall_time_ns"], 0)
                self.assertTrue(Path(run["stdout_path"]).is_file())
                self.assertTrue(Path(run["stderr_path"]).is_file())
                for observation in (run["pre"], run["post"]):
                    self.assertEqual(set(observation["psi"]), {"cpu", "io", "memory"})
                    self.assertTrue(all(item["status"] == "ok" for item in observation["psi"].values()))
                    self.assertEqual(observation["host_memory"]["status"], "ok")
                    self.assertEqual(observation["host_memory"]["mem_available_bytes"], 30_000 * 1024)
                    self.assertEqual(observation["gpu_0"]["status"], "ok")
                    self.assertEqual(observation["gpu_0"]["gpu_index"], 0)
                    self.assertIsNotNone(observation["gpu_0"]["used_bytes"])
                    self.assertEqual(observation["gpu_0"]["total_bytes"], 24_576 * control.MIB_BYTES)
            self.assertEqual(receipt["timed_runs"][0]["pre"]["psi"]["cpu"]["some"]["avg10"], 10.0)
            self.assertEqual(receipt["timed_runs"][0]["post"]["psi"]["cpu"]["some"]["avg10"], 20.0)

    def test_timeout_runs_are_retained_and_statistics_remain_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            proc_root = directory / "proc"
            self.write_proc_snapshot(proc_root)
            nvidia_smi = self.write_program(directory, "fake_nvidia_smi.py", FAKE_NVIDIA_SMI)
            sleeper = self.write_program(directory, "sleep.py", SLEEP_COMMAND)
            environment = {
                "N01_REPEAT_TEST_NVIDIA_COUNTER": str(directory / "nvidia-counter.txt"),
            }
            config = self.make_config(
                directory,
                command=(sys.executable, str(sleeper)),
                nvidia_smi=nvidia_smi,
                proc_root=proc_root,
                warmups=1,
                repeats=2,
                timeout_seconds=0.03,
            )
            with mock.patch.dict(os.environ, environment, clear=False):
                receipt = control.run_repeat_control(config)

            self.assertEqual(receipt["status"], "completed-with-failures")
            self.assertEqual(len(receipt["timed_runs"]), 2)
            self.assertTrue(all(run["status"] == "timed-out" for run in receipt["timed_runs"]))
            self.assertTrue(all(run["timed_out"] for run in receipt["timed_runs"]))
            self.assertEqual(receipt["summary"]["count"], 2)
            self.assertEqual(receipt["summary"]["successful_count"], 0)
            self.assertEqual(receipt["summary"]["failures"], 2)
            self.assertIsNone(receipt["summary"]["median_ms"])
            self.assertEqual(
                receipt["summary"]["bootstrap_median_95_ci_ms"]["reason"],
                "no successful timed runs",
            )

    def test_summary_bootstrap_is_deterministic_and_excludes_no_records(self) -> None:
        timed_runs = [
            {"status": "succeeded", "wall_time_ms": 10.0},
            {"status": "failed", "wall_time_ms": 99.0},
            {"status": "succeeded", "wall_time_ms": 20.0},
            {"status": "succeeded", "wall_time_ms": 40.0},
        ]
        first = control.summarize_timed_runs(
            timed_runs,
            bootstrap_resamples=600,
            bootstrap_seed=1234,
        )
        second = control.summarize_timed_runs(
            timed_runs,
            bootstrap_resamples=600,
            bootstrap_seed=1234,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["count"], 4)
        self.assertEqual(first["successful_count"], 3)
        self.assertEqual(first["failures"], 1)
        self.assertEqual(first["median_ms"], 20.0)
        self.assertEqual(first["mean_ms"], (10.0 + 20.0 + 40.0) / 3.0)
        self.assertGreater(first["stddev_ms"], 0.0)
        confidence = first["bootstrap_median_95_ci_ms"]
        self.assertLessEqual(confidence["lower_ms"], first["median_ms"])
        self.assertGreaterEqual(confidence["upper_ms"], first["median_ms"])

    def test_receipt_and_log_names_are_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            proc_root = directory / "proc"
            self.write_proc_snapshot(proc_root)
            command = self.write_program(directory, "fake_command.py", FAKE_COMMAND)
            nvidia_smi = self.write_program(directory, "fake_nvidia_smi.py", FAKE_NVIDIA_SMI)
            environment = {
                "N01_REPEAT_TEST_COMMAND_COUNTER": str(directory / "command-counter.txt"),
                "N01_REPEAT_TEST_NVIDIA_COUNTER": str(directory / "nvidia-counter.txt"),
                "N01_REPEAT_TEST_PROC_ROOT": str(proc_root),
            }
            config = self.make_config(
                directory,
                command=(sys.executable, str(command)),
                nvidia_smi=nvidia_smi,
                proc_root=proc_root,
                repeats=1,
            )
            with mock.patch.dict(os.environ, environment, clear=False):
                receipt = control.run_repeat_control(config)
                first_bytes = Path(receipt["receipt_path"]).read_bytes()
                with self.assertRaisesRegex(control.ControlError, "refusing to overwrite"):
                    control.run_repeat_control(config)
                symlink_config = self.make_config(
                    directory,
                    command=(sys.executable, str(command)),
                    nvidia_smi=nvidia_smi,
                    proc_root=proc_root,
                    output_name="symlink-output",
                    repeats=1,
                )
                symlink_config.output_dir.mkdir()
                (symlink_config.output_dir / symlink_config.receipt_name).symlink_to(
                    directory / "outside-receipt.json"
                )
                with self.assertRaisesRegex(control.ControlError, "refusing to overwrite"):
                    control.run_repeat_control(symlink_config)
            self.assertEqual(Path(receipt["receipt_path"]).read_bytes(), first_bytes)
            self.assertEqual((directory / "command-counter.txt").read_text(encoding="utf-8"), "2")

    def test_child_receives_attempt_context_without_mutating_controller_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            proc_root = directory / "proc"
            self.write_proc_snapshot(proc_root)
            command = self.write_program(directory, "environment_command.py", ENVIRONMENT_COMMAND)
            nvidia_smi = self.write_program(directory, "fake_nvidia_smi.py", FAKE_NVIDIA_SMI)
            config = self.make_config(
                directory,
                command=(sys.executable, str(command)),
                nvidia_smi=nvidia_smi,
                proc_root=proc_root,
                warmups=1,
                repeats=2,
            )
            environment = {"N01_REPEAT_TEST_NVIDIA_COUNTER": str(directory / "nvidia-counter.txt")}
            with mock.patch.dict(os.environ, environment, clear=False):
                self.assertNotIn("N01_REPEAT_CONTROL_PHASE", os.environ)
                self.assertNotIn("N01_REPEAT_CONTROL_INDEX", os.environ)
                receipt = control.run_repeat_control(config)
                self.assertNotIn("N01_REPEAT_CONTROL_PHASE", os.environ)
                self.assertNotIn("N01_REPEAT_CONTROL_INDEX", os.environ)
            self.assertEqual(receipt["status"], "completed")
            observed = []
            for kind, index in (("warmup", 1), ("timed", 1), ("timed", 2)):
                stdout = directory / "receipt" / f"{kind}-{index:03d}.stdout.log"
                observed.append(json.loads(stdout.read_text(encoding="utf-8")))
            self.assertEqual(
                observed,
                [
                    {"index": "1", "phase": "warmup"},
                    {"index": "1", "phase": "timed"},
                    {"index": "2", "phase": "timed"},
                ],
            )

    def test_n06a_parent_cleanup_name_is_exact_and_cli_requires_explicit_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            artifact_root = directory / "n06a-artifacts"
            cleanup = control.N06ATimeoutCleanupConfig(
                artifact_root=artifact_root,
                docker_launcher="/usr/bin/docker",
                command_timeout_seconds=3.0,
            )
            normalized = control._validate_n06a_timeout_cleanup(cleanup)
            assert normalized is not None
            attempt_dir, name = control.n06a_vllm_container_name(
                normalized,
                kind="timed",
                index=2,
            )
            expected_attempt = (artifact_root.resolve() / "timed-002").resolve()
            expected_digest = hashlib.sha256(str(expected_attempt).encode("utf-8")).hexdigest()[:20]
            self.assertEqual(attempt_dir, expected_attempt)
            self.assertEqual(name, f"riley-n06a-vllm-timed-0002-{expected_digest}")

            with self.assertRaisesRegex(control.ControlError, "must be supplied together"):
                control.parse_command_line(
                    [
                        "--output-dir",
                        str(directory / "output"),
                        "--n06a-timeout-cleanup-artifact-root",
                        str(artifact_root),
                        "--",
                        sys.executable,
                        "-V",
                    ]
                )
            parsed = control.parse_command_line(
                [
                    "--output-dir",
                    str(directory / "output"),
                    "--n06a-timeout-cleanup-artifact-root",
                    str(artifact_root),
                    "--n06a-timeout-cleanup-docker-launcher",
                    "/usr/bin/docker",
                    "--n06a-timeout-cleanup-command-timeout-seconds",
                    "2.5",
                    "--",
                    sys.executable,
                    "-V",
                ]
            )
            self.assertIsNotNone(parsed.n06a_timeout_cleanup)
            assert parsed.n06a_timeout_cleanup is not None
            self.assertEqual(parsed.n06a_timeout_cleanup.artifact_root, artifact_root.resolve())
            self.assertEqual(parsed.n06a_timeout_cleanup.docker_launcher, "/usr/bin/docker")
            self.assertEqual(parsed.n06a_timeout_cleanup.command_timeout_seconds, 2.5)

    def test_n06a_daemon_cleanup_touches_only_the_exact_derived_container(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            cleanup = control.N06ATimeoutCleanupConfig(
                artifact_root=directory / "artifacts",
                docker_launcher="/usr/bin/docker",
                command_timeout_seconds=1.0,
            )
            cleanup = control._validate_n06a_timeout_cleanup(cleanup)
            assert cleanup is not None
            _, container_name = control.n06a_vllm_container_name(
                cleanup,
                kind="timed",
                index=1,
            )
            invocations: list[list[str]] = []
            inventory_count = 0

            def fake_run(argv, **_kwargs):
                nonlocal inventory_count
                rendered = list(argv)
                invocations.append(rendered)
                arguments = rendered[1:]
                if arguments[:3] == ["container", "ls", "--all"]:
                    inventory_count += 1
                    stdout = (
                        f"{container_name}\t{'a' * 12}\n".encode()
                        if inventory_count == 1
                        else b""
                    )
                    return subprocess.CompletedProcess(rendered, 0, stdout, b"")
                if arguments[:2] == ["container", "stop"]:
                    self.assertEqual(arguments[-1], container_name)
                    return subprocess.CompletedProcess(rendered, 0, b"", b"")
                if arguments[:2] == ["container", "wait"]:
                    self.assertEqual(arguments[-1], container_name)
                    return subprocess.CompletedProcess(rendered, 0, b"0\n", b"")
                if arguments[:2] == ["container", "inspect"]:
                    self.assertEqual(arguments[-1], container_name)
                    return subprocess.CompletedProcess(rendered, 1, b"", b"No such container")
                self.fail(f"unexpected Docker cleanup argv: {arguments!r}")

            with mock.patch.object(control.subprocess, "run", side_effect=fake_run):
                receipt = control._cleanup_n06a_owned_container(cleanup, container_name)

            self.assertTrue(receipt["cleanup_verified"])
            self.assertEqual(receipt["initial_state"], "present")
            self.assertEqual(receipt["final_state"], "absent")
            operations = [command["operation"] for command in receipt["commands"]]
            self.assertEqual(
                operations,
                ["inventory-before", "stop", "wait", "inventory-after-stop", "inspect-absence"],
            )
            self.assertTrue(invocations)
            self.assertFalse(any("prune" in argv for argv in invocations))
            inventory = invocations[0]
            self.assertIn(f"name=^/{container_name}$", inventory)
            for argv in invocations:
                arguments = argv[1:]
                if arguments[:3] == ["container", "ls", "--all"]:
                    self.assertIn(f"name=^/{container_name}$", arguments)
                else:
                    self.assertEqual(arguments[-1], container_name)

    def test_n06a_timeout_snapshot_signals_only_recorded_separate_group_leader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            proc_root = directory / "proc"

            def write_stat(pid: int, parent_pid: int, process_group: int, session: int) -> None:
                task_directory = proc_root / str(pid) / "task" / str(pid)
                task_directory.mkdir(parents=True)
                # /proc/<pid>/stat fields after comm start at field 3.  Field
                # 22 (zero-based suffix index 19) is the immutable start tick.
                suffix = [
                    "S",
                    str(parent_pid),
                    str(process_group),
                    str(session),
                    *(["0"] * 15),
                    str(pid * 10),
                ]
                (proc_root / str(pid) / "stat").write_text(
                    f"{pid} (n06a-test) {' '.join(suffix)}\n",
                    encoding="utf-8",
                )
                (task_directory / "children").write_text("", encoding="utf-8")

            write_stat(100, 1, 100, 100)
            write_stat(101, 100, 100, 100)  # N06-A driver remains in N01's group.
            write_stat(200, 101, 200, 200)  # Separate Riley/Docker session leader.
            write_stat(201, 200, 200, 200)
            (proc_root / "100" / "task" / "100" / "children").write_text("101", encoding="utf-8")
            (proc_root / "101" / "task" / "101" / "children").write_text("200", encoding="utf-8")
            (proc_root / "200" / "task" / "200" / "children").write_text("201", encoding="utf-8")
            signals: list[tuple[int, int]] = []
            process = type("FakeProcess", (), {"pid": 100})()
            with mock.patch.object(control.os, "name", "posix"), mock.patch.object(
                control.os,
                "getpgid",
                return_value=100,
            ), mock.patch.object(
                control.os,
                "killpg",
                side_effect=lambda pgid, signal_number: signals.append((pgid, signal_number)),
            ):
                receipt = control._signal_recorded_descendant_groups(
                    process,
                    proc_root=proc_root,
                )

            self.assertTrue(receipt["snapshot"]["snapshot_complete"])
            self.assertEqual(signals, [(200, control.signal.SIGTERM)])
            self.assertEqual(
                receipt["candidate_groups"],
                [{"pgid": 200, "leader_pid": 200, "leader_starttime_ticks": 2000}],
            )
            self.assertIn(
                {"pid": 101, "pgid": 100, "reason": "covered-by-outer-process-group"},
                receipt["skipped_processes"],
            )
            for pid in (101, 200, 201):
                shutil.rmtree(proc_root / str(pid))
            finalized = control._finalize_recorded_descendant_cleanup(
                receipt,
                timeout_seconds=0.1,
            )
            self.assertTrue(finalized["cleanup_verified"])
            self.assertEqual(finalized["status"], "cleaned")

    def test_n06a_unproven_post_attempt_cleanup_turns_success_into_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            proc_root = directory / "proc"
            self.write_proc_snapshot(proc_root)
            command = self.write_program(directory, "environment_command.py", ENVIRONMENT_COMMAND)
            nvidia_smi = self.write_program(directory, "fake_nvidia_smi.py", FAKE_NVIDIA_SMI)
            cleanup = control.N06ATimeoutCleanupConfig(
                artifact_root=directory / "n06a-artifacts",
                docker_launcher="/usr/bin/docker",
                command_timeout_seconds=1.0,
            )
            config = self.make_config(
                directory,
                command=(sys.executable, str(command)),
                nvidia_smi=nvidia_smi,
                proc_root=proc_root,
                warmups=1,
                repeats=1,
                n06a_timeout_cleanup=cleanup,
            )
            unproven = {
                "schema_version": control.N06A_DOCKER_CONTAINER_CLEANUP_SCHEMA_VERSION,
                "cleanup_verified": False,
                "errors": ["fake daemon unavailable"],
            }
            environment = {"N01_REPEAT_TEST_NVIDIA_COUNTER": str(directory / "nvidia-counter.txt")}
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                control,
                "_cleanup_n06a_owned_container",
                return_value=unproven,
            ):
                receipt = control.run_repeat_control(config)

            self.assertEqual(receipt["status"], "completed-with-failures")
            self.assertEqual(
                [run["status"] for run in [*receipt["warmup_runs"], *receipt["timed_runs"]]],
                ["failed-cleanup", "not-started-after-failed-cleanup"],
            )
            failed_cleanup = receipt["warmup_runs"][0]
            parent_cleanup = failed_cleanup["n06a_parent_cleanup"]
            self.assertFalse(parent_cleanup["cleanup_verified"])
            self.assertIn("N06-A parent cleanup is unproven", failed_cleanup["error"])
            evidence = Path(parent_cleanup["receipt_path"])
            self.assertTrue(evidence.is_file())
            self.assertEqual(
                parent_cleanup["receipt_sha256"],
                hashlib.sha256(evidence.read_bytes()).hexdigest(),
            )
            self.assertFalse(json.loads(evidence.read_text(encoding="utf-8"))["cleanup_verified"])
            blocked = receipt["timed_runs"][0]
            self.assertEqual(
                blocked,
                {
                    "kind": "timed",
                    "index": 1,
                    "status": "not-started-after-failed-cleanup",
                    "blocking_kind": "warmup",
                    "blocking_index": 1,
                    "reason": control.N06A_NOT_STARTED_AFTER_FAILED_CLEANUP_REASON,
                },
            )
            self.assertFalse((directory / "receipt" / "timed-001.stdout.log").exists())
            self.assertFalse((directory / "receipt" / "timed-001.n06a-parent-cleanup.json").exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux /proc process ancestry")
    def test_n06a_timeout_cleans_separate_descendant_session_before_outer_driver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            proc_root = Path("/proc")
            parent = self.write_program(
                directory,
                "n06a_timeout_parent.py",
                N06A_TIMEOUT_PARENT_COMMAND,
            )
            child = self.write_program(
                directory,
                "n06a_timeout_child.py",
                N06A_TIMEOUT_CHILD_COMMAND,
            )
            docker = self.write_program(directory, "n06a_absent_docker.py", N06A_ABSENT_DOCKER)
            nvidia_smi = self.write_program(directory, "fake_nvidia_smi.py", FAKE_NVIDIA_SMI)
            cleanup = control.N06ATimeoutCleanupConfig(
                artifact_root=directory / "n06a-artifacts",
                docker_launcher=str(docker),
                command_timeout_seconds=1.0,
            )
            config = self.make_config(
                directory,
                command=(sys.executable, str(parent)),
                nvidia_smi=nvidia_smi,
                proc_root=proc_root,
                warmups=1,
                repeats=1,
                timeout_seconds=0.5,
                n06a_timeout_cleanup=cleanup,
            )
            child_signal = directory / "child-signal.txt"
            environment = {
                "N01_REPEAT_TEST_NVIDIA_COUNTER": str(directory / "nvidia-counter.txt"),
                "N01_N06A_TIMEOUT_CHILD": str(child),
                "N01_N06A_TIMEOUT_CHILD_PID": str(directory / "child.pid"),
                "N01_N06A_TIMEOUT_CHILD_SIGNAL": str(child_signal),
                "N01_N06A_DOCKER_LOG": str(directory / "docker.jsonl"),
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                receipt = control.run_repeat_control(config)

            self.assertEqual(receipt["status"], "completed-with-failures")
            self.assertEqual(
                [run["status"] for run in [*receipt["warmup_runs"], *receipt["timed_runs"]]],
                ["timed-out", "timed-out"],
            )
            self.assertEqual(child_signal.read_text(encoding="utf-8"), "SIGTERM")
            for run in [*receipt["warmup_runs"], *receipt["timed_runs"]]:
                parent_cleanup = run["n06a_parent_cleanup"]
                self.assertTrue(parent_cleanup["cleanup_verified"])
                self.assertTrue(parent_cleanup["docker_container_cleanup"]["cleanup_verified"])
                descendants = parent_cleanup["timeout_descendant_cleanup"]
                self.assertTrue(descendants["cleanup_verified"])
                self.assertEqual(descendants["status"], "cleaned")
                self.assertTrue(
                    any(action["status"] == "signalled" for action in descendants["actions"])
                )
                self.assertTrue(parent_cleanup["outer_process_cleanup"]["cleanup_verified"])
            docker_commands = [
                json.loads(line)
                for line in (directory / "docker.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(docker_commands)
            self.assertFalse(any("prune" in command for command in docker_commands))

    def test_malformed_cli_and_optional_linux_fields_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            with self.assertRaisesRegex(control.ControlError, "must follow --"):
                control.parse_command_line(["--output-dir", str(output_dir)])
            with self.assertRaisesRegex(control.ControlError, "warmups must be an integer >= 1"):
                control.parse_command_line(
                    ["--output-dir", str(output_dir), "--warmups", "0", "--", sys.executable, "-V"]
                )
            with self.assertRaisesRegex(control.ControlError, "Blender"):
                control.parse_command_line(
                    ["--output-dir", str(output_dir), "--", "/usr/local/bin/blender"]
                )

        psi = control.parse_psi_snapshot("some avg10=1.00 avg60=2.00 avg300=3.00 total=42\n", "cpu")
        self.assertEqual(psi["some"]["total"], 42)
        self.assertIsNone(psi["full"])
        self.assertEqual(
            control.parse_meminfo("MemTotal: 100 kB\nMemFree: 10 kB\n")["MemTotal"],
            100 * 1024,
        )


if __name__ == "__main__":
    unittest.main()
