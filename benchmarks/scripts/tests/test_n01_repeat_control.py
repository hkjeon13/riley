from __future__ import annotations

import json
import os
from pathlib import Path
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
