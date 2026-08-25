from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

import run_repeatability_gate as runner  # noqa: E402


PREFLIGHT = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

counter_path = Path(os.environ["FAKE_PREFLIGHT_COUNTER"])
count = int(counter_path.read_text() if counter_path.exists() else "0") + 1
counter_path.write_text(str(count))
with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"stage": "preflight", "count": count}) + "\n")
if count <= int(os.environ.get("FAKE_PREFLIGHT_THERMAL_FAILURES", "0")):
    print("preflight: start temperature 61 C exceeds 50 C", file=sys.stderr)
    raise SystemExit(2)
mode = os.environ.get("FAKE_PREFLIGHT_CONTRACT", "valid")
power_limit = "451.00" if int(os.environ.get("FAKE_PREFLIGHT_DRIFT_AT", "0")) == count else "450.00"
values = {
    "environment_id": "rtx4090-ubuntu22-driver580-v1",
    "os_id": "ubuntu",
    "os_version_id": "22.04",
    "kernel_release": "6.8.0-138-generic",
    "machine": "x86_64",
    "cpu_model": "Intel Core i7-13700K",
    "physical_cpu_cores": "16",
    "logical_cpu_threads": "24",
    "ram_bytes": "67185598464",
    "driver_version": "580.173.02",
    "persistence_mode": "Disabled",
    "power_limit_w": power_limit,
    "graphics_clock_mhz": "2520",
    "memory_clock_mhz": "10501",
    "cpu_governor": "powersave",
    "cpu_governor_policy_count": "24",
    "memory_total_mib": "24564",
    "clock_synchronized": "yes",
    "staging_available_bytes": str(50 * 1024 * 1024 * 1024),
    "staging_minimum_bytes": str(20 * 1024 * 1024 * 1024),
}
if mode == "missing":
    del values["cpu_governor"]
elif mode == "clock-unsynchronized":
    values["clock_synchronized"] = "no"
elif mode == "disk-low":
    values["staging_available_bytes"] = str(19 * 1024 * 1024 * 1024)
elif mode == "persistence-enabled":
    values["persistence_mode"] = "Enabled"
elif mode == "performance-governor":
    values["cpu_governor"] = "performance"
elif mode == "wrong-host":
    values["kernel_release"] = "6.8.0-139-generic"
for key, value in values.items():
    print(f"{key}={value}")
if mode == "duplicate":
    print("driver_version=580.173.02")
print(f"preflight stderr {count}", file=sys.stderr)
if int(os.environ.get("FAKE_PREFLIGHT_FAIL_AT", "0")) == count:
    print("injected preflight failure", file=sys.stderr)
    raise SystemExit(23)
"""


FAKE_UV = r"""#!/usr/bin/env python3
import json
import os
import shlex
import sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("uv 0.12.5-fake")
    raise SystemExit(0)
if args == ["python", "find", "3.13.15"]:
    managed = (
        Path(os.environ["UV_PYTHON_INSTALL_DIR"])
        / "cpython-3.13.15-fake"
        / "bin/python3.13"
    )
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " \"$@\"\n",
        encoding="utf-8",
    )
    managed.chmod(0o755)
    with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": "python-find", "path": str(managed)}) + "\n")
    print(managed)
    raise SystemExit(0)
if args[:1] == ["sync"]:
    with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "stage": "sync",
            "argv": args,
            "project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
        }, sort_keys=True) + "\n")
    print("sync stdout")
    print("sync stderr", file=sys.stderr)
    if os.environ.get("FAKE_SYNC_FAIL") == "1":
        raise SystemExit(27)
    (Path(os.environ["UV_CACHE_DIR"]) / "synced.marker").write_text("frozen\n")
    project_environment = Path(os.environ["UV_PROJECT_ENVIRONMENT"])
    (project_environment / "bin").mkdir(parents=True, exist_ok=True)
    managed = (
        Path(os.environ["UV_PYTHON_INSTALL_DIR"])
        / "cpython-3.13.15-fake"
        / "bin/python3.13"
    )
    (project_environment / "bin/python").symlink_to(managed)
    (project_environment / "synced.marker").write_text("frozen\n")
    raise SystemExit(0)
def value(flag):
    return args[args.index(flag) + 1]

result_dir = Path(value("--result-dir"))
if "preparation" in result_dir.parts:
    prime_counter = Path(os.environ["FAKE_PRIME_COUNTER"])
    prime_count = int(prime_counter.read_text() if prime_counter.exists() else "0") + 1
    prime_counter.write_text(str(prime_count))
    event = {
        "stage": "prime",
        "count": prime_count,
        "concurrency": int(value("--concurrency")),
        "prompt_tokens": int(value("--prompt-tokens")),
        "output_tokens": int(value("--output-tokens")),
        "project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
    }
    with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    print(f"prime stdout {prime_count}")
    print(f"prime stderr {prime_count}", file=sys.stderr)
    if int(os.environ.get("FAKE_PRIME_FAIL_AT", "0")) == prime_count:
        raise SystemExit(28)
    marker = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / (
        f"c{event['concurrency']}-p{event['prompt_tokens']}-o{event['output_tokens']}.compiled"
    )
    marker.write_text("primed\n")
    result_dir.mkdir(parents=True, exist_ok=False)
    workload = {
        "concurrency": event["concurrency"],
        "prompt_tokens": event["prompt_tokens"],
        "output_tokens": event["output_tokens"],
        "warm_state": "warm",
    }
    (result_dir / "raw.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "prime": prime_count,
                    "run_id": value("--run-id"),
                    "trial_index": trial_index,
                    "status": "success",
                    "failure_count": 0,
                    "workload": workload,
                }
            ) + "\n"
            for trial_index in range(1, 31)
        )
    )
    raise SystemExit(0)

counter_path = Path(os.environ["FAKE_BENCHMARK_COUNTER"])
count = int(counter_path.read_text() if counter_path.exists() else "0") + 1
counter_path.write_text(str(count))
event = {
    "stage": "benchmark",
    "count": count,
    "run_id": value("--run-id"),
    "run_index": int(value("--run-index")),
    "warm_state": value("--warm-state"),
    "concurrency": int(value("--concurrency")),
    "prompt_tokens": int(value("--prompt-tokens")),
    "output_tokens": int(value("--output-tokens")),
    "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    "project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
}
with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(event, sort_keys=True) + "\n")
print(f"benchmark stdout {count}")
print(f"benchmark stderr {count}", file=sys.stderr)
if int(os.environ.get("FAKE_BENCHMARK_FAIL_AT", "0")) == count:
    print("injected benchmark failure", file=sys.stderr)
    raise SystemExit(29)
if os.environ.get("FAKE_MEASUREMENT_CACHE_WRITE") == "1" and count == 1:
    (Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "unexpected.compiled").write_text("drift\n")
result_dir.mkdir(parents=True, exist_ok=False)
(result_dir / "raw.jsonl").write_text(json.dumps({"fake": count}) + "\n")
"""


CHECKER = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
trials = [item for item in args if item.endswith("raw.jsonl")]
with Path(os.environ["FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"stage": "checker", "trial_count": len(trials)}) + "\n")
passed = os.environ.get("FAKE_CHECKER_PASS", "1") == "1"
output.write_text(json.dumps({
    "contract_version": os.environ.get(
        "FAKE_CHECKER_CONTRACT_VERSION", "rustinfer.repeatability.v2"
    ),
    "status": "passed" if passed else "failed",
    "passed": passed,
}) + "\n")
print("checker stdout")
print("checker stderr", file=sys.stderr)
raise SystemExit(0 if passed else 31)
"""


class FakePrograms:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.event_log = root / "fake-events.jsonl"
        self.preflight_counter = root / "preflight-counter.txt"
        self.benchmark_counter = root / "benchmark-counter.txt"
        self.prime_counter = root / "prime-counter.txt"
        self.cache_roots = {
            "UV_CACHE_DIR": root / "cache" / "uv",
            "UV_PYTHON_INSTALL_DIR": root / "cache" / "uv-python",
            "HF_HOME": root / "cache" / "huggingface",
            "VLLM_CACHE_ROOT": root / "cache" / "vllm",
            "TORCHINDUCTOR_CACHE_DIR": root / "cache" / "torchinductor",
            "TRITON_CACHE_DIR": root / "cache" / "triton",
            "CUDA_CACHE_PATH": root / "cache" / "cuda",
        }
        for path in self.cache_roots.values():
            path.mkdir(parents=True)
        self.preflight = self._program("preflight.py", PREFLIGHT)
        self.uv = self._program("uv", FAKE_UV)
        self.checker = self._program("checker.py", CHECKER)

    def _program(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    def environment(self, **extra: str) -> dict[str, str]:
        result = {
            "FAKE_EVENT_LOG": str(self.event_log),
            "FAKE_PREFLIGHT_COUNTER": str(self.preflight_counter),
            "FAKE_BENCHMARK_COUNTER": str(self.benchmark_counter),
            "FAKE_PRIME_COUNTER": str(self.prime_counter),
            "FAKE_CHECKER_PASS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **{key: str(path) for key, path in self.cache_roots.items()},
        }
        result.update(extra)
        return result

    def argv(
        self,
        output_root: Path,
        lane: str = "vllm",
        finalize_to: Path | None = None,
        *,
        allow_noncanonical_tools: bool = True,
    ) -> list[str]:
        argv = [
            "--lane",
            lane,
            "--output-root",
            str(output_root),
            "--preflight",
            str(self.preflight),
            "--checker",
            str(self.checker),
            "--uv",
            str(self.uv),
        ]
        if allow_noncanonical_tools:
            argv.append("--allow-noncanonical-tools")
        if finalize_to is not None:
            argv.extend(("--finalize-to", str(finalize_to)))
        return argv

    def events(self) -> list[dict[str, object]]:
        if not self.event_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.event_log.read_text(encoding="utf-8").splitlines()
        ]


class RepeatabilityRunnerTests(unittest.TestCase):
    def test_gzip_json_artifact_is_deterministic_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = {"z": [3, 2, 1], "a": {"unicode": "재현성"}}
            first = root / "first.json.gz"
            second = root / "second.json.gz"

            runner._write_new_gzip_json(first, value)
            runner._write_new_gzip_json(second, value)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first.read_bytes()[4:8], b"\x00\x00\x00\x00")
            self.assertEqual(
                json.loads(gzip.decompress(first.read_bytes()).decode("utf-8")),
                value,
            )

    def test_sensitive_runtime_and_preflight_overrides_are_rejected(self) -> None:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "32",
            "RUSTINFER_MAX_IDLE_MEMORY_MIB": "999999",
            "TRITON_HOME": "/tmp/untracked-triton-home",
            "UV_PROJECT_ENVIRONMENT": "/tmp/untracked-project-env",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(runner.RunnerError) as captured:
                runner._sanitized_child_environment(allow_test_environment=False)
        message = str(captured.exception)
        self.assertIn("CUDA_VISIBLE_DEVICES", message)
        self.assertIn("OMP_NUM_THREADS", message)
        self.assertIn("RUSTINFER_MAX_IDLE_MEMORY_MIB", message)
        self.assertIn("TRITON_HOME", message)
        self.assertIn("UV_PROJECT_ENVIRONMENT", message)

    def test_canonical_plan_requires_exact_inputs_and_full_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            copied_matrix = root / "matrix.yaml"
            copied_matrix.write_bytes(
                (REPOSITORY_ROOT / "benchmarks/matrix.yaml").read_bytes()
            )
            args = runner._parser().parse_args(
                [
                    "--lane",
                    "vllm",
                    "--output-root",
                    str(root / "artifacts"),
                    "--matrix",
                    str(copied_matrix),
                    "--preflight",
                    str(fakes.preflight),
                    "--checker",
                    str(fakes.checker),
                    "--uv",
                    str(fakes.uv),
                ]
            )
            args.runner_argv = ["runner-test"]
            with mock.patch.dict(os.environ, fakes.environment(), clear=False):
                with self.assertRaisesRegex(runner.RunnerError, "noncanonical --matrix"):
                    runner._build_plan(args, root / "artifacts")

            canonical_args = runner._parser().parse_args(
                [
                    "--lane",
                    "vllm",
                    "--output-root",
                    str(root / "canonical-artifacts"),
                    "--preflight",
                    str(fakes.preflight),
                    "--checker",
                    str(fakes.checker),
                    "--uv",
                    str(fakes.uv),
                ]
            )
            canonical_args.runner_argv = ["runner-test"]
            counts = {"lanes": 3, "prompts": 31, "reference_cases": 31, "trials": 0}
            with (
                mock.patch.object(runner, "CANONICAL_PREFLIGHT", fakes.preflight),
                mock.patch.object(runner, "CANONICAL_CHECKER", fakes.checker),
                mock.patch.object(
                    runner,
                    "_toolchain_evidence",
                    return_value={
                        "uv": {
                            "path": str(fakes.uv),
                            "sha256": runner.CANONICAL_UV_LINUX_X86_64_SHA256,
                            "version": runner.CANONICAL_UV_VERSION,
                        },
                        "python": {
                            "path": sys.executable,
                            "sha256": runner.CANONICAL_PYTHON_LINUX_X86_64_SHA256,
                            "implementation": "cpython",
                            "version": runner.CANONICAL_PYTHON_VERSION,
                            "platform": "linux",
                            "machine": "x86_64",
                        },
                    },
                ),
                mock.patch.object(
                    runner, "_validate_canonical_contract", return_value=counts
                ) as validate,
                mock.patch.dict(os.environ, fakes.environment(), clear=False),
            ):
                plan = runner._build_plan(
                    canonical_args, root / "canonical-artifacts"
                )
            validate.assert_called_once_with()
            self.assertTrue(plan["canonical_execution"])
            self.assertEqual(plan["contract_validation"]["counts"], counts)

    def test_prime_raw_uses_shared_validator_and_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw.jsonl"
            cell = dict(runner.PRIME_CELLS[0])
            run_id = "prime-run"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "trial_index": index,
                            "status": "success",
                            "failure_count": 0,
                            "workload": cell,
                        }
                    )
                    + "\n"
                    for index in range(1, 31)
                ),
                encoding="utf-8",
            )

            class FakeValidator:
                ContractError = ValueError

                def __init__(self) -> None:
                    self.calls = 0

                def validate_result_file(self, *args: object) -> int:
                    self.calls += 1
                    return 30

            validator = FakeValidator()
            context = (validator, {}, {}, {}, {})
            count = runner._validate_prime_raw(
                path,
                cell,
                run_id,
                context,
                matrix_path=runner.CANONICAL_MATRIX,
                prompts_path=runner.CANONICAL_PROMPTS,
            )
            self.assertEqual(count, 30)
            self.assertEqual(validator.calls, 1)

    def test_runs_exact_four_cells_by_five_fresh_invocations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            with mock.patch.dict(os.environ, fakes.environment(), clear=False):
                returncode = runner.main(fakes.argv(output))

            self.assertEqual(returncode, 0)
            plan = json.loads(
                (output / "execution-plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["contract_version"], runner.RUNNER_CONTRACT_VERSION)
            self.assertEqual(plan["invocation_count"], 20)
            self.assertEqual(plan["cells_per_run"], 4)
            self.assertEqual(plan["independent_runs"], 5)
            self.assertEqual(len(set(plan["run_ids"])), 5)
            self.assertEqual(
                [item["ordinal"] for item in plan["invocations"]],
                list(range(1, 21)),
            )
            for run_index in range(1, 6):
                group = [
                    item
                    for item in plan["invocations"]
                    if item["independent_run_index"] == run_index
                ]
                self.assertEqual(len(group), 4)
                self.assertEqual({item["run_id"] for item in group}, {plan["run_ids"][run_index - 1]})
                self.assertEqual(
                    [item["cell"] for item in group], list(runner.EXPECTED_CELLS)
                )
            for invocation in plan["invocations"]:
                self.assertEqual(
                    invocation["benchmark_argv"][0], str(fakes.uv.resolve())
                )
                self.assertEqual(
                    invocation["benchmark_argv"][1:6],
                    ["run", "--frozen", "--offline", "--no-sync", "--project"],
                )
                self.assertEqual(invocation["environment"]["UV_OFFLINE"], "1")
                self.assertEqual(invocation["environment"]["UV_PYTHON"], "3.13.15")
                self.assertEqual(
                    invocation["environment"]["UV_PYTHON_DOWNLOADS"], "never"
                )
                self.assertEqual(
                    invocation["environment"]["PYTHONDONTWRITEBYTECODE"], "1"
                )
                self.assertEqual(
                    invocation["environment"]["CUDA_CACHE_MAXSIZE"], "4294967296"
                )
                self.assertEqual(invocation["environment"]["PYTHONHASHSEED"], "0")
                self.assertEqual(
                    invocation["environment"]["HF_HUB_DISABLE_TELEMETRY"], "1"
                )
                self.assertEqual(
                    invocation["environment"]["VLLM_USE_FLASHINFER_SAMPLER"],
                    "0",
                )
                self.assertNotIn("CUDA_VISIBLE_DEVICES", invocation["environment"])
                self.assertEqual(
                    invocation["environment"]["UV_PROJECT_ENVIRONMENT"],
                    plan["runtime_environment_policy"]["project_environment"]["path"],
                )
                result_dir = Path(invocation["result_dir"])
                self.assertTrue((result_dir / "raw.jsonl").is_file())
                self.assertTrue(
                    (Path(invocation["artifact_dir"]) / "preflight.stdout.txt").is_file()
                )

            fake_events = fakes.events()
            self.assertEqual(
                [event["stage"] for event in fake_events],
                ["python-find", "sync", "prime", "prime", "prime"]
                + [stage for _ in range(20) for stage in ("preflight", "benchmark")]
                + ["checker"],
            )
            benchmarks = [event for event in fake_events if event["stage"] == "benchmark"]
            self.assertEqual(len(benchmarks), 20)
            self.assertTrue(
                all(event["flashinfer_sampler"] == "0" for event in benchmarks)
            )
            project_environments = {
                event["project_environment"]
                for event in fake_events
                if event["stage"] in {"sync", "prime", "benchmark"}
            }
            self.assertEqual(len(project_environments), 1)
            project_environment = Path(project_environments.pop())
            self.assertTrue(project_environment.is_dir())
            self.assertEqual(
                project_environment.parent.parent.resolve(),
                fakes.cache_roots["UV_PYTHON_INSTALL_DIR"].resolve(),
            )
            self.assertNotIn(REPOSITORY_ROOT, project_environment.parents)
            self.assertEqual(fake_events[-1]["trial_count"], 20)
            self.assertTrue((output / "repeatability-report.json").is_file())
            report = json.loads(
                (output / "repeatability-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["contract_version"],
                runner.EXPECTED_REPEATABILITY_REPORT_CONTRACT,
            )
            self.assertTrue((output / "completion.json").is_file())
            completion = json.loads(
                (output / "completion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                completion["contract_version"], runner.RUNNER_CONTRACT_VERSION
            )
            baseline = json.loads(
                (output / "preflight-baseline.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                baseline["comparability"]["driver_version"], "580.173.02"
            )
            self.assertFalse((output / "failure.json").exists())
            runner_events = (
                output / "execution-events.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(runner_events), 48)
            preparation = json.loads(
                (output / "preparation/summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                preparation["contract_version"], runner.PREPARATION_CONTRACT_VERSION
            )
            self.assertEqual(len(preparation["prime_invocations"]), 3)
            self.assertTrue(preparation["prime_results_excluded_from_checker"])
            self.assertTrue(preparation["python_evidence"]["same_binary_sha256"])
            self.assertEqual(
                preparation["cache_inventory_artifact"]["contract_version"],
                runner.CACHE_INVENTORY_ARTIFACT_CONTRACT,
            )
            for key in ("cache_inventory_before", "cache_inventory_after"):
                inventory_path = Path(plan["preparation"][key])
                self.assertEqual(inventory_path.suffixes[-2:], [".json", ".gz"])
                self.assertFalse(inventory_path.with_suffix("").exists())
                with gzip.open(inventory_path, "rt", encoding="utf-8") as stream:
                    inventory = json.load(stream)
                self.assertEqual(
                    inventory["contract_version"], "rustinfer.cache-inventory.v1"
                )
            self.assertEqual(
                plan["reproducibility_environment"]["allowlisted_values"]
                ["HF_HUB_OFFLINE"],
                "1",
            )

            expected_hashes = {
                "matrix": REPOSITORY_ROOT / "benchmarks/matrix.yaml",
                "prompts": REPOSITORY_ROOT / "benchmarks/prompts.jsonl",
                "lane_manifest": REPOSITORY_ROOT / "benchmarks/lanes/vllm.json",
                "dependency_lock": REPOSITORY_ROOT / "benchmarks/lanes/vllm/uv.lock",
            }
            for key, path in expected_hashes.items():
                self.assertEqual(
                    plan["inputs"][key]["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            lock_prefix = plan["inputs"]["dependency_lock"]["sha256"][:16]
            self.assertRegex(
                Path(
                    plan["runtime_environment_policy"]["project_environment"]["path"]
                ).name,
                rf"^vllm-{lock_prefix}-[0-9a-f]{{12}}$",
            )

    def test_hf_lane_uses_the_same_three_profile_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            with mock.patch.dict(os.environ, fakes.environment(), clear=False):
                returncode = runner.main(fakes.argv(output, lane="hf-transformers"))
            self.assertEqual(returncode, 0)
            plan = json.loads((output / "execution-plan.json").read_text())
            self.assertEqual(
                [item["cell"] for item in plan["preparation"]["prime_invocations"]],
                list(runner.PRIME_CELLS),
            )
            self.assertEqual(
                [event["stage"] for event in fakes.events()].count("prime"), 3
            )

    def test_thermal_limit_retries_are_bounded_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            environment = fakes.environment(
                FAKE_PREFLIGHT_THERMAL_FAILURES="2",
                FAKE_BENCHMARK_FAIL_AT="1",
            )
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(runner.time, "sleep") as sleep,
            ):
                returncode = runner.main(fakes.argv(output))
            self.assertEqual(returncode, 2)
            self.assertEqual(sleep.call_args_list, [mock.call(30), mock.call(30)])
            plan = json.loads((output / "execution-plan.json").read_text())
            invocation = plan["invocations"][0]
            attempts = Path(invocation["artifacts"]["preflight_attempts"])
            self.assertEqual(
                sorted(path.name for path in attempts.iterdir()),
                [
                    f"attempt-{attempt:03d}.{suffix}"
                    for attempt in range(1, 4)
                    for suffix in ("snapshot.json", "stderr.txt", "stdout.txt")
                ],
            )
            first = json.loads((attempts / "attempt-001.snapshot.json").read_text())
            third = json.loads((attempts / "attempt-003.snapshot.json").read_text())
            self.assertTrue(first["retryable_temperature_limit"])
            self.assertEqual(first["observed_temperature_c"], 61)
            self.assertFalse(third["retryable_temperature_limit"])
            self.assertEqual(third["returncode"], 0)
            self.assertIn(
                "driver_version=580.173.02",
                Path(invocation["artifacts"]["preflight_stdout"]).read_text(),
            )

    def test_project_environment_is_fresh_and_cache_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            destination = runner._derive_project_environment(
                uv_python_install_root=str(
                    fakes.cache_roots["UV_PYTHON_INSTALL_DIR"]
                ),
                lane_id="vllm",
                dependency_lock_sha256="a" * 64,
                execution_nonce="b" * 12,
            )
            destination.mkdir(parents=True)
            with self.assertRaisesRegex(runner.RunnerError, "already exists"):
                runner._derive_project_environment(
                    uv_python_install_root=str(
                        fakes.cache_roots["UV_PYTHON_INSTALL_DIR"]
                    ),
                    lane_id="vllm",
                    dependency_lock_sha256="a" * 64,
                    execution_nonce="b" * 12,
                )

            outside = root / "outside"
            outside.mkdir()
            escaping = fakes.cache_roots["UV_PYTHON_INSTALL_DIR"] / "escaping"
            escaping.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(runner.RunnerError, "symlink escapes"):
                runner._cache_inventory(fakes.environment())

    def test_canonical_toolchain_rejects_unpinned_uv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fakes = FakePrograms(Path(directory))
            with self.assertRaisesRegex(runner.RunnerError, "canonical uv version"):
                runner._toolchain_evidence(fakes.uv, canonical=True)
            with (
                mock.patch.object(
                    runner, "_executable_version", return_value=runner.CANONICAL_UV_VERSION
                ),
                mock.patch.object(
                    runner,
                    "_artifact",
                    return_value={"path": str(fakes.uv), "sha256": "0" * 64},
                ),
            ):
                with self.assertRaisesRegex(runner.RunnerError, "uv Linux x86_64"):
                    runner._toolchain_evidence(fakes.uv, canonical=True)

    def test_preflight_configuration_drift_stops_before_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            environment = fakes.environment(FAKE_PREFLIGHT_DRIFT_AT="3")
            with mock.patch.dict(os.environ, environment, clear=False):
                returncode = runner.main(fakes.argv(output))

            self.assertEqual(returncode, 2)
            self.assertEqual(
                [event["stage"] for event in fakes.events()],
                ["python-find", "sync", "prime", "prime", "prime", "preflight", "benchmark", "preflight", "benchmark", "preflight"],
            )
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "preflight-comparability")
            self.assertEqual(failure["ordinal"], 3)
            self.assertIn("power_limit_w", failure["message"])
            plan = json.loads((output / "execution-plan.json").read_text())
            drift_snapshot = Path(
                plan["invocations"][2]["artifacts"]["preflight_snapshot"]
            )
            self.assertTrue(drift_snapshot.is_file())

    def test_preflight_required_key_missing_or_duplicate_is_rejected(self) -> None:
        for mode in ("missing", "duplicate"):
            with self.subTest(mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fakes = FakePrograms(root)
                output = root / "artifacts"
                environment = fakes.environment(FAKE_PREFLIGHT_CONTRACT=mode)
                with mock.patch.dict(os.environ, environment, clear=False):
                    returncode = runner.main(fakes.argv(output))
                self.assertEqual(returncode, 2)
                failure = json.loads(
                    (output / "failure.json").read_text(encoding="utf-8")
                )
                self.assertEqual(failure["stage"], "preflight-contract")
                self.assertEqual(failure["ordinal"], 1)
                self.assertEqual(
                    [event["stage"] for event in fakes.events()],
                    ["python-find", "sync", "prime", "prime", "prime", "preflight"],
                )

    def test_primary_clock_disk_and_power_policy_fail_closed(self) -> None:
        modes = (
            "clock-unsynchronized",
            "disk-low",
            "persistence-enabled",
            "performance-governor",
            "wrong-host",
        )
        for mode in modes:
            with self.subTest(mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fakes = FakePrograms(root)
                output = root / "artifacts"
                environment = fakes.environment(FAKE_PREFLIGHT_CONTRACT=mode)
                with mock.patch.dict(os.environ, environment, clear=False):
                    returncode = runner.main(fakes.argv(output))
                self.assertEqual(returncode, 2)
                failure = json.loads(
                    (output / "failure.json").read_text(encoding="utf-8")
                )
                self.assertEqual(failure["stage"], "preflight-contract")
                self.assertEqual(
                    [event["stage"] for event in fakes.events()],
                    ["python-find", "sync", "prime", "prime", "prime", "preflight"],
                )

    def test_preflight_failure_stops_and_preserves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            environment = fakes.environment(FAKE_PREFLIGHT_FAIL_AT="3")
            with mock.patch.dict(os.environ, environment, clear=False):
                returncode = runner.main(fakes.argv(output))

            self.assertEqual(returncode, 2)
            self.assertEqual(
                [event["stage"] for event in fakes.events()],
                ["python-find", "sync", "prime", "prime", "prime", "preflight", "benchmark", "preflight", "benchmark", "preflight"],
            )
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "preflight")
            self.assertEqual(failure["ordinal"], 3)
            self.assertEqual(failure["returncode"], 23)
            failed_dir = Path(
                json.loads((output / "execution-plan.json").read_text())["invocations"][2][
                    "artifact_dir"
                ]
            )
            self.assertIn(
                "injected preflight failure",
                (failed_dir / "preflight.stderr.txt").read_text(encoding="utf-8"),
            )
            self.assertFalse((failed_dir / "benchmark.stdout.txt").exists())
            self.assertFalse((output / "repeatability-report.json").exists())

    def test_benchmark_failure_stops_and_preserves_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            environment = fakes.environment(FAKE_BENCHMARK_FAIL_AT="2")
            with mock.patch.dict(os.environ, environment, clear=False):
                returncode = runner.main(fakes.argv(output))

            self.assertEqual(returncode, 2)
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "benchmark")
            self.assertEqual(failure["ordinal"], 2)
            self.assertEqual(failure["returncode"], 29)
            plan = json.loads((output / "execution-plan.json").read_text())
            failed_dir = Path(plan["invocations"][1]["artifact_dir"])
            self.assertIn(
                "injected benchmark failure",
                (failed_dir / "benchmark.stderr.txt").read_text(encoding="utf-8"),
            )
            self.assertFalse(Path(plan["invocations"][1]["result_dir"]).exists())
            self.assertEqual(
                [event["stage"] for event in fakes.events()],
                ["python-find", "sync", "prime", "prime", "prime", "preflight", "benchmark", "preflight", "benchmark"],
            )

    def test_nonpassing_checker_report_is_preserved_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            environment = fakes.environment(FAKE_CHECKER_PASS="0")
            with mock.patch.dict(os.environ, environment, clear=False):
                returncode = runner.main(fakes.argv(output))

            self.assertEqual(returncode, 1)
            report = json.loads(
                (output / "repeatability-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report,
                {
                    "contract_version": "rustinfer.repeatability.v2",
                    "status": "failed",
                    "passed": False,
                },
            )
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "checker")
            self.assertEqual(failure["returncode"], 31)

    def test_passing_v1_checker_report_is_rejected_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            environment = fakes.environment(
                FAKE_CHECKER_CONTRACT_VERSION="rustinfer.repeatability.v1"
            )
            with mock.patch.dict(os.environ, environment, clear=False):
                returncode = runner.main(fakes.argv(output))

            self.assertEqual(returncode, 1)
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "checker-report")
            self.assertIn("contract_version", failure["message"])
            self.assertFalse((output / "completion.json").exists())

    def test_preparation_failure_is_preserved_before_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            with mock.patch.dict(
                os.environ,
                fakes.environment(FAKE_PRIME_FAIL_AT="2"),
                clear=False,
            ):
                returncode = runner.main(fakes.argv(output))

            self.assertEqual(returncode, 2)
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "preparation")
            self.assertIn("prime invocation 2", failure["message"])
            self.assertEqual(
                [event["stage"] for event in fakes.events()],
                ["python-find", "sync", "prime", "prime"],
            )
            self.assertIn(
                "prime stderr 2",
                (output / "preparation/prime-02-warm-c1-p4096-o128/benchmark.stderr.txt")
                .read_text(encoding="utf-8"),
            )

    def test_measured_invocation_cannot_fill_external_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            with mock.patch.dict(
                os.environ,
                fakes.environment(FAKE_MEASUREMENT_CACHE_WRITE="1"),
                clear=False,
            ):
                returncode = runner.main(fakes.argv(output))

            self.assertEqual(returncode, 2)
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "cache-drift")
            self.assertEqual(failure["ordinal"], 1)
            plan = json.loads((output / "execution-plan.json").read_text())
            drift = Path(
                plan["invocations"][0]["artifacts"]["cache_drift_inventory"]
            )
            self.assertTrue(drift.is_file())
            self.assertNotEqual(
                json.loads(drift.read_text())["aggregate_sha256"],
                json.loads(
                    (output / "preparation/summary.json").read_text()
                )["measured_cache_baseline_sha256"],
            )

    def test_canonical_gate_requires_offline_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "artifacts"
            with mock.patch.dict(
                os.environ,
                fakes.environment(HF_HUB_OFFLINE="0"),
                clear=False,
            ):
                returncode = runner.main(fakes.argv(output))
            self.assertEqual(returncode, 2)
            self.assertFalse(output.exists())
            self.assertEqual(fakes.events(), [])

    def test_passing_staging_tree_can_be_atomically_finalized_with_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            output = root / "staging"
            results_root = root / "results"
            results_root.mkdir()
            destination = results_root / "20260824T000000Z-vllm-repeatability-test"
            fake_child_environment = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "UV_OFFLINE": "1",
                **fakes.environment(),
            }

            def fake_python_evidence(
                path: Path,
                *,
                environment: object,
                stdout_path: Path,
                stderr_path: Path,
                canonical: bool,
            ) -> dict[str, object]:
                evidence = {
                    "implementation": "cpython",
                    "version": runner.CANONICAL_PYTHON_VERSION,
                    "platform": "linux",
                    "machine": "x86_64",
                    "reported_executable": str(path),
                    "path": str(path.resolve()),
                    "sha256": runner.CANONICAL_PYTHON_LINUX_X86_64_SHA256,
                }
                stdout_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                return evidence

            with (
                mock.patch.object(runner, "RESULTS_ROOT", results_root),
                mock.patch.object(runner, "CANONICAL_PREFLIGHT", fakes.preflight),
                mock.patch.object(runner, "CANONICAL_CHECKER", fakes.checker),
                mock.patch.object(
                    runner,
                    "_toolchain_evidence",
                    return_value={
                        "uv": {
                            "path": str(fakes.uv),
                            "sha256": runner.CANONICAL_UV_LINUX_X86_64_SHA256,
                            "version": runner.CANONICAL_UV_VERSION,
                        },
                        "python": {
                            "path": sys.executable,
                            "sha256": runner.CANONICAL_PYTHON_LINUX_X86_64_SHA256,
                            "implementation": "cpython",
                            "version": runner.CANONICAL_PYTHON_VERSION,
                            "platform": "linux",
                            "machine": "x86_64",
                        },
                    },
                ),
                mock.patch.object(
                    runner,
                    "_inspect_python_interpreter",
                    side_effect=fake_python_evidence,
                ),
                mock.patch.object(
                    runner,
                    "_validate_canonical_contract",
                    return_value={"lanes": 3, "prompts": 31, "reference_cases": 31, "trials": 0},
                ),
                mock.patch.object(
                    runner, "_shared_result_validation_context", return_value=None
                ),
                mock.patch.object(
                    runner,
                    "_sanitized_child_environment",
                    return_value=fake_child_environment,
                ),
                mock.patch.dict(os.environ, fakes.environment(), clear=False),
            ):
                returncode = runner.main(
                    fakes.argv(
                        output,
                        finalize_to=destination,
                        allow_noncanonical_tools=False,
                    )
                )

            self.assertEqual(returncode, 0)
            self.assertTrue(output.is_dir())
            self.assertTrue(destination.is_dir())
            manifest = json.loads(
                (destination / "finalize-manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreater(manifest["file_count_excluding_this_manifest"], 80)
            self.assertEqual(manifest["destination"], str(destination.resolve()))
            finalized_paths = {
                item["path"] for item in manifest["files_excluding_this_manifest"]
            }
            self.assertIn(
                "preparation/cache.inventory.before.json.gz", finalized_paths
            )
            self.assertIn(
                "preparation/cache.inventory.after.json.gz", finalized_paths
            )
            self.assertNotIn(
                "preparation/cache.inventory.before.json", finalized_paths
            )
            self.assertNotIn(
                "preparation/cache.inventory.after.json", finalized_paths
            )
            for item in manifest["files_excluding_this_manifest"]:
                copied = destination / item["path"]
                self.assertEqual(copied.stat().st_size, item["bytes"])
                self.assertEqual(
                    hashlib.sha256(copied.read_bytes()).hexdigest(), item["sha256"]
                )
            plan = json.loads(
                (destination / "execution-plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["finalize_to"], str(destination.resolve()))
            self.assertTrue((destination / "README.md").is_file())
            self.assertTrue((destination / "metadata.json").is_file())
            self.assertTrue((destination / "raw.jsonl").is_file())
            metadata = json.loads(
                (destination / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["combined_raw"]["nonempty_lines"], 20)

    def test_finalize_refuses_symlinked_staging_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root = root / "results"
            results_root.mkdir()
            staging = root / "staging"
            staging.mkdir()
            (staging / "artifact.txt").write_text("artifact", encoding="utf-8")
            (staging / "link.txt").symlink_to(staging / "artifact.txt")
            destination = results_root / "20260824T000000Z-vllm-repeatability-symlink"
            with mock.patch.object(runner, "RESULTS_ROOT", results_root):
                with self.assertRaisesRegex(runner.RunnerError, "symlink"):
                    runner._finalize_staging(
                        staging,
                        destination,
                        {
                            "git_revision": "a" * 40,
                            "lane_id": "vllm",
                            "implementation_id": "vllm",
                        },
                    )
            self.assertFalse(destination.exists())

    def test_finalize_name_is_bound_to_lane_and_workload(self) -> None:
        valid = Path(
            "benchmarks/results/20260824T000000Z-vllm-repeatability-gate-a"
        )
        with mock.patch.object(runner, "RESULTS_ROOT", REPOSITORY_ROOT / "benchmarks/results"):
            destination = runner._safe_finalize_destination(valid)
            runner._validate_finalize_name(destination, "vllm")
            with self.assertRaisesRegex(runner.RunnerError, "YYYYMMDD"):
                runner._validate_finalize_name(destination, "hf-transformers-eager")

    def test_refuses_existing_or_repository_internal_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fakes = FakePrograms(root)
            existing = root / "existing"
            existing.mkdir()
            with mock.patch.dict(os.environ, fakes.environment(), clear=False):
                self.assertEqual(runner.main(fakes.argv(existing)), 2)
            self.assertEqual(fakes.events(), [])

            internal = REPOSITORY_ROOT / f"runner-output-must-not-exist-{uuid.uuid4().hex}"
            self.assertFalse(internal.exists())
            with mock.patch.dict(os.environ, fakes.environment(), clear=False):
                self.assertEqual(runner.main(fakes.argv(internal)), 2)
            self.assertFalse(internal.exists())
            self.assertEqual(fakes.events(), [])

    def test_manifest_placeholders_are_strict(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "benchmarks/lanes/vllm.json").read_text(
                encoding="utf-8"
            )
        )
        cases = {
            "unknown": ("{run_id}", "{surprise}"),
            "missing": ("{run_id}", "run-static"),
            "embedded": ("{run_id}", "prefix-{run_id}"),
        }
        for label, (old, new) in cases.items():
            with self.subTest(label):
                changed = json.loads(json.dumps(manifest))
                argv = changed["commands"]["benchmark"]["argv"]
                argv[argv.index(old)] = new
                with self.assertRaises(runner.RunnerError):
                    runner._manifest_command(changed)
        changed = json.loads(json.dumps(manifest))
        changed["commands"]["benchmark"]["environment"]["HF_HUB_OFFLINE"] = "0"
        with self.assertRaisesRegex(runner.RunnerError, "cannot override"):
            runner._manifest_command(changed)


if __name__ == "__main__":
    unittest.main()
