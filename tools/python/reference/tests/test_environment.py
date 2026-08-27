from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from benchmarks.scripts.validate_contract import ContractError, validate_instance
from riley_reference.calibration import FP32_ORACLE_KIND, CalibrationError
from riley_reference.cli import main
from riley_reference.environment import (
    PRIMARY_ENVIRONMENT_SNAPSHOT,
    EnvironmentContractError,
    probe_primary_environment,
    validate_environment_snapshot,
)
from riley_reference.hf_calibration import produce_hf_oracle

from .support import fixture_provenance, write_prompts


def _snapshot_with(path: tuple[str, ...], value: object) -> dict[str, object]:
    snapshot = copy.deepcopy(PRIMARY_ENVIRONMENT_SNAPSHOT)
    cursor = snapshot
    for key in path[:-1]:
        child = cursor[key]
        if not isinstance(child, dict):
            raise AssertionError(f"test path {path!r} does not select an object")
        cursor = child
    cursor[path[-1]] = value
    return snapshot


def _cpuinfo() -> str:
    records = []
    for thread in range(24):
        core = thread if thread < 16 else thread - 16
        records.append(
            "\n".join(
                (
                    f"processor : {thread}",
                    "model name : 13th Gen Intel(R) Core(TM) i7-13700K",
                    "physical id : 0",
                    f"core id : {core}",
                )
            )
        )
    return "\n\n".join(records) + "\n"


class EnvironmentContractTests(unittest.TestCase):
    @property
    def repository(self) -> Path:
        return Path(__file__).resolve().parents[4]

    def test_stdlib_probe_collects_exact_host_and_transient_preflight(self) -> None:
        def command(arguments) -> str:
            joined = " ".join(arguments)
            if "--query-gpu=" in joined:
                return (
                    "0, NVIDIA GeForce RTX 4090, 580.173.02, 8.9, 24564, "
                    "12, 42, Disabled, 450.00, [N/A], [N/A]\n"
                )
            if "--query-compute-apps=" in joined:
                return ""
            if arguments[0] == "timedatectl":
                return "yes\n"
            return "NVIDIA-SMI 580.173.02   CUDA Version: 13.0\n"

        def read(path: Path) -> str:
            return {
                Path("/etc/os-release"): 'ID=ubuntu\nVERSION_ID="22.04"\n',
                Path("/proc/cpuinfo"): _cpuinfo(),
                Path("/proc/meminfo"): "MemTotal:       65610936 kB\n",
            }[path]

        observed = probe_primary_environment(
            command_runner=command,
            text_reader=read,
            uname_probe=lambda: SimpleNamespace(
                release="6.8.0-138-generic", machine="x86_64"
            ),
            which_probe=lambda _name: None,
            governor_probe=lambda: ["powersave"] * 24,
        )
        self.assertEqual(observed["accelerator"]["memory_used_mib"], 12)
        self.assertEqual(observed["accelerator"]["temperature_c"], 42)
        self.assertEqual(observed["host"]["cpu_governor_policy_count"], 24)
        validate_environment_snapshot(observed)

    def test_exact_identity_and_preflight_mismatches_fail_closed(self) -> None:
        mismatches = {
            "driver": (("accelerator", "nvidia_driver_version"), "999.0"),
            "ram": (("host", "ram_bytes"), 67_185_598_465),
            "os": (("host", "os_version_id"), "24.04"),
            "cpu": (("host", "cpu_model"), "Different CPU"),
            "gpu count": (("accelerator", "gpu_count"), 2),
            "persistence": (("accelerator", "persistence_mode"), "Enabled"),
            "compute process": (("accelerator", "compute_process_count"), 1),
            "idle memory": (("accelerator", "memory_used_mib"), 257),
            "temperature": (("accelerator", "temperature_c"), 51),
            "governor": (("host", "cpu_governor"), "performance"),
            "governor count": (("host", "cpu_governor_policy_count"), 23),
            "clock sync": (("host", "clock_synchronized"), False),
            "application clock sentinel": (
                ("accelerator", "application_graphics_clock_mhz"),
                2520,
            ),
        }
        for label, (path, value) in mismatches.items():
            with self.subTest(label=label):
                with self.assertRaises(EnvironmentContractError):
                    validate_environment_snapshot(_snapshot_with(path, value))

    def test_fixture_cli_injected_probe_rejects_before_provenance_or_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repo = workspace / "repo"
            prompts = repo / "benchmarks/prompts.jsonl"
            prompts.parent.mkdir(parents=True)
            write_prompts(prompts)
            calls: list[str] = []

            def provenance(_root: Path, *, observed_environment):
                calls.append("provenance")
                value = fixture_provenance()
                value["observed_environment"] = observed_environment
                value["sources"]["prompts"]["sha256"] = __import__(
                    "hashlib"
                ).sha256(prompts.read_bytes()).hexdigest()
                return value

            for index, (path, value) in enumerate(
                (
                    (("accelerator", "nvidia_driver_version"), "wrong"),
                    (("host", "ram_bytes"), 1),
                    (("host", "os_id"), "wrong"),
                    (("host", "cpu_model"), "wrong"),
                    (("accelerator", "gpu_count"), 2),
                )
            ):
                with self.subTest(path=path), contextlib.redirect_stderr(io.StringIO()):
                    result = main(
                        [
                            "generate",
                            "--prompts",
                            str(prompts),
                            "--output",
                            str(workspace / f"fixture-{index}.json"),
                            "--repo-root",
                            str(repo),
                            "--max-new-tokens",
                            "1",
                        ],
                        backend_factory=lambda **_kwargs: calls.append("backend"),
                        provenance_factory=provenance,
                        environment_probe=lambda path=path, value=value: _snapshot_with(
                            path, value
                        ),
                    )
                    self.assertEqual(result, 2)
            self.assertEqual(calls, [])

    def test_oracle_producer_injected_probe_rejects_before_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repo = workspace / "repo"
            repo.mkdir()
            calls: list[str] = []
            with self.assertRaisesRegex(CalibrationError, "environment preflight"):
                produce_hf_oracle(
                    artifact_kind=FP32_ORACLE_KIND,
                    prompts_path=repo / "benchmarks/prompts.jsonl",
                    manifest_path=workspace / "fp32.json",
                    sidecar_path=workspace / "fp32.safetensors",
                    repo_root=repo,
                    device="fake",
                    local_files_only=True,
                    created_at=__import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ),
                    backend_factory=lambda **_kwargs: calls.append("backend"),
                    environment_probe=lambda: _snapshot_with(
                        ("accelerator", "compute_process_count"), 1
                    ),
                )
            self.assertEqual(calls, [])

    def test_snapshot_schema_rejects_unknown_fields(self) -> None:
        changed = copy.deepcopy(PRIMARY_ENVIRONMENT_SNAPSHOT)
        changed["host"]["unknown"] = True
        for schema_name in (
            "reference-fixture.schema.json",
            "correctness-calibration-manifest.schema.json",
        ):
            with self.subTest(schema=schema_name):
                schema = json.loads(
                    (self.repository / "benchmarks/schemas" / schema_name).read_text(
                        encoding="utf-8"
                    )
                )
                with self.assertRaises(ContractError):
                    validate_instance(changed, schema["$defs"]["environmentSnapshot"])


if __name__ == "__main__":
    unittest.main()
