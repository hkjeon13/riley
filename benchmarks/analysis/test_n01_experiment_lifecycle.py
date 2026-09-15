import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

ANALYSIS_DIRECTORY = Path(__file__).resolve().parent
if str(ANALYSIS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIRECTORY))

import n01_experiment_lifecycle as lifecycle

NEXT_STAGE_DIRECTORY = ANALYSIS_DIRECTORY.parent / "next_stage"

from n01_experiment_lifecycle import (
    GLOBAL_GPU_PEAK_CEILING_BYTES,
    MIB_BYTES,
    PHASES,
    UNCERTAINTY_RESERVE_BYTES,
    ManifestError,
    parse_nvidia_smi,
    parse_proc_meminfo,
    parse_proc_status_rss,
    run_manifest,
    sha256_bytes,
    validate_manifest,
)


class N01ExperimentLifecycleTests(unittest.TestCase):
    def make_fake_nvidia_smi(self, directory: Path, values: list[int]) -> tuple[Path, dict[str, str]]:
        values_path = directory / "fake-nvidia-smi-values.txt"
        values_path.write_text("\n".join(str(value) for value in values) + "\n", encoding="utf-8")
        counter_path = directory / "fake-nvidia-smi-counter.txt"
        executable = directory / "fake-nvidia-smi"
        executable.write_text(
            "#!/bin/sh\n"
            "count=0\n"
            "if [ -f \"$N01_FAKE_SMI_COUNTER\" ]; then count=$(sed -n '1p' \"$N01_FAKE_SMI_COUNTER\"); fi\n"
            "count=$((count + 1))\n"
            "printf '%s\\n' \"$count\" > \"$N01_FAKE_SMI_COUNTER\"\n"
            "value=$(sed -n \"${count}p\" \"$N01_FAKE_SMI_VALUES\")\n"
            "if [ -z \"$value\" ]; then value=$(tail -n 1 \"$N01_FAKE_SMI_VALUES\"); fi\n"
            "printf '0, GPU-test, Test GPU, %s, 24576\\n' \"$value\"\n",
            encoding="utf-8",
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return executable, {
            "N01_FAKE_SMI_COUNTER": str(counter_path),
            "N01_FAKE_SMI_VALUES": str(values_path),
        }

    def manifest(self, nvidia_smi: Path, phases: list[dict[str, object]] | None = None) -> dict[str, object]:
        return {
            "schema_version": "riley.n01-experiment-manifest.v1",
            "experiment_id": "n01-lifecycle-test",
            "lifecycle_policy": "leave_stopped_no_restore",
            "contract_bindings": {
                "model_descriptor": {
                    "path": "benchmarks/next_stage/model-descriptors-v1.json",
                    "sha256": "a" * 64,
                },
                "numerical_profile": {
                    "id": "native-bf16-r0-v1",
                    "path": "benchmarks/next_stage/numerical-profiles-v1.json",
                    "sha256": "b" * 64,
                },
                "checkpoint_provenance": {"status": "materialization-pending"},
            },
            "target": {
                "model_id": "qwen2.5-3b-instruct-primary",
                "evidence_scope": "operator-control",
            },
            "gpu": {
                "nvidia_smi": str(nvidia_smi),
                "gpu_indices": [0],
                "poll_interval_ms": 10,
            },
            "memory_budget": {
                "global_peak_ceiling_bytes": GLOBAL_GPU_PEAK_CEILING_BYTES,
                "uncertainty_reserve_bytes": UNCERTAINTY_RESERVE_BYTES,
            },
            "phases": phases
            or [
                {
                    "name": name,
                    "argv": [sys.executable, "-c", "import time; time.sleep(0.04)"],
                    "timeout_seconds": 5,
                }
                for name in PHASES
            ],
        }

    def bind_contract_files(self, directory: Path, manifest: dict[str, object]) -> None:
        """Bind relative manifest paths to concrete, immutable fixture bytes."""
        descriptor = directory / "model-descriptor.json"
        profile = directory / "numerical-profile.json"
        descriptor.write_bytes((NEXT_STAGE_DIRECTORY / "model-descriptors-v1.json").read_bytes())
        profile.write_bytes((NEXT_STAGE_DIRECTORY / "numerical-profiles-v1.json").read_bytes())
        bindings = manifest["contract_bindings"]
        assert isinstance(bindings, dict)
        model_descriptor = bindings["model_descriptor"]
        numerical_profile = bindings["numerical_profile"]
        assert isinstance(model_descriptor, dict)
        assert isinstance(numerical_profile, dict)
        model_descriptor["path"] = descriptor.name
        model_descriptor["sha256"] = sha256_bytes(descriptor.read_bytes())
        numerical_profile["path"] = profile.name
        numerical_profile["sha256"] = sha256_bytes(profile.read_bytes())

    def test_full_lifecycle_records_global_bytes_and_each_required_phase(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [400, 550, 700, 850, 900])
            manifest = self.manifest(executable)
            self.bind_contract_files(directory, manifest)
            manifest["phases"][0]["cwd"] = "."  # type: ignore[index]
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 0, receipt)
            self.assertTrue(receipt["terminal"])
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["lifecycle_policy"], "leave_stopped_no_restore")
            self.assertEqual(receipt["contract_bindings"], manifest["contract_bindings"])
            verification = receipt["contract_binding_verification"]
            self.assertEqual(
                verification["model_descriptor"]["resolved_path"],
                str((directory / "model-descriptor.json").resolve()),
            )
            self.assertEqual(
                verification["numerical_profile"]["verified_sha256"],
                manifest["contract_bindings"]["numerical_profile"]["sha256"],  # type: ignore[index]
            )
            self.assertEqual(verification["checkpoint_provenance"], {"status": "materialization-pending"})
            self.assertEqual(
                verification["target"],
                {
                    "model_id": "qwen2.5-3b-instruct-primary",
                    "numerical_profile_id": "native-bf16-r0-v1",
                    "evidence_scope": "operator-control",
                    "semantic_contract": "validated-n01-v1",
                },
            )
            self.assertEqual(receipt["phase_order"], list(PHASES))
            self.assertEqual([phase["name"] for phase in receipt["phases"]], list(PHASES))
            self.assertTrue(all(phase["state"] == "completed" for phase in receipt["phases"]))
            self.assertTrue(all(phase["memory"]["sample_count"] >= 2 for phase in receipt["phases"]))
            self.assertTrue(
                all(
                    phase["cwd"] == str(directory.resolve())
                    for phase in receipt["phases"]
                )
            )
            self.assertEqual(receipt["baseline"]["global_gpu_used_bytes"], 400 * MIB_BYTES)
            self.assertEqual(receipt["overall"]["overall_global_gpu_peak_bytes"], 900 * MIB_BYTES)
            self.assertEqual(receipt["memory_budget"]["global_peak_ceiling_bytes"], GLOBAL_GPU_PEAK_CEILING_BYTES)
            self.assertEqual(receipt["memory_budget"]["uncertainty_reserve_bytes"], UNCERTAINTY_RESERVE_BYTES)
            self.assertTrue(receipt["overall"]["reserve_satisfied"])
            self.assertEqual(receipt["overall"]["sampled_peak_evidence_status"], "sampled-observed-not-continuous")
            self.assertEqual(receipt["overall"]["missing_in_process_sample_phases"], [])
            self.assertTrue(
                all(
                    phase["memory"]["live_poll_sample_count"] >= 1
                    for phase in receipt["phases"]
                    if phase["name"] in {"warmup", "timed-serving"}
                )
            )
            self.assertEqual({sample["phase"] for sample in receipt["samples"] if sample["phase"]}, set(PHASES))
            self.assertTrue(all(sample["global_gpu_used_bytes"] is not None for sample in receipt["samples"]))
            self.assertTrue(
                all(
                    {"controller_rss_bytes", "host_memory_total_bytes", "host_memory_available_bytes"} <= set(sample)
                    for sample in receipt["samples"]
                )
            )
            self.assertIn("controller_rss_peak_bytes", receipt["overall"])
            self.assertIn("host_memory_available_min_bytes", receipt["overall"])

    def test_failed_command_skips_workload_but_runs_shutdown_and_writes_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100] * 32)
            shutdown_marker = directory / "shutdown-ran.txt"
            phases = [
                {"name": "asset-preparation", "argv": ["/usr/bin/true"], "timeout_seconds": 5},
                {"name": "gpu-initialization", "argv": ["/usr/bin/false"], "timeout_seconds": 5},
                {"name": "warmup", "argv": ["/usr/bin/true"], "timeout_seconds": 5},
                {"name": "timed-serving", "argv": ["/usr/bin/true"], "timeout_seconds": 5},
                {
                    "name": "shutdown",
                    "argv": ["/bin/sh", "-c", f"printf shutdown > {shutdown_marker}"],
                    "timeout_seconds": 5,
                },
            ]
            manifest = self.manifest(executable, phases)
            self.bind_contract_files(directory, manifest)
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 1)
            self.assertTrue(receipt["terminal"])
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(
                [phase["state"] for phase in receipt["phases"]],
                ["completed", "failed", "skipped", "skipped", "completed"],
                receipt,
            )
            self.assertEqual(shutdown_marker.read_text(encoding="utf-8"), "shutdown")
            self.assertIn("gpu-initialization failed: command exited with code 1", receipt["failure_reasons"])
            self.assertEqual(receipt["phases"][-1]["name"], "shutdown")

    def test_restore_or_blender_command_is_rejected_before_any_command_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100])
            phases = [
                {"name": name, "argv": ["/usr/bin/true"], "timeout_seconds": 5}
                for name in PHASES
            ]
            phases[0]["argv"] = ["/bin/echo", "restore-blender"]
            manifest = self.manifest(executable, phases)
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 2)
            self.assertTrue(receipt["terminal"])
            self.assertEqual(receipt["status"], "invalid-manifest")
            self.assertIn("may not invoke Blender or a Blender restoration helper", receipt["manifest"]["validation_error"])
            self.assertFalse((directory / "fake-nvidia-smi-counter.txt").exists())
            self.assertTrue(all(phase["state"] == "skipped" for phase in receipt["phases"]))

    def test_historical_blender_receipt_filename_is_not_treated_as_a_blender_command(self):
        manifest = self.manifest(Path("/usr/bin/true"))
        manifest["phases"][0]["argv"] = [  # type: ignore[index]
            "/usr/bin/echo",
            "/data/archive/blender-restored.json",
            "/data/archive/restore-summary.json",
        ]
        validate_manifest(manifest)
        manifest["phases"][0]["argv"] = ["/usr/bin/blender"]  # type: ignore[index]
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)

    def test_materialized_checkpoint_provenance_requires_its_path_and_sha256(self):
        manifest = self.manifest(Path("/usr/bin/true"))
        manifest["contract_bindings"]["checkpoint_provenance"] = {  # type: ignore[index]
            "status": "materialized",
            "path": "artifacts/qwen2.5-3b/riley-checkpoint.json",
            "sha256": "c" * 64,
        }
        validate_manifest(manifest)
        manifest["contract_bindings"]["checkpoint_provenance"]["sha256"] = "not-a-sha256"  # type: ignore[index]
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)

    def test_reserve_is_a_separate_gate_from_the_absolute_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100, 18_200])
            manifest = self.manifest(executable)
            self.bind_contract_files(directory, manifest)
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 1)
            self.assertTrue(receipt["overall"]["within_global_peak_ceiling"])
            self.assertFalse(receipt["overall"]["reserve_satisfied"])
            self.assertLess(receipt["overall"]["overall_global_gpu_peak_bytes"], GLOBAL_GPU_PEAK_CEILING_BYTES)
            self.assertIn("global GPU peak did not retain the 1,000,000,000-byte reserve", receipt["failure_reasons"])

    def test_contract_binding_digest_mismatch_rejects_before_gpu_or_phase_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100])
            marker = directory / "phase-command-ran.txt"
            phases = [
                {"name": name, "argv": ["/usr/bin/true"], "timeout_seconds": 5}
                for name in PHASES
            ]
            phases[0]["argv"] = ["/bin/sh", "-c", f"touch {marker}"]
            manifest = self.manifest(executable, phases)
            self.bind_contract_files(directory, manifest)
            bindings = manifest["contract_bindings"]
            assert isinstance(bindings, dict)
            model_descriptor = bindings["model_descriptor"]
            assert isinstance(model_descriptor, dict)
            model_descriptor["sha256"] = "0" * 64
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 2)
            self.assertEqual(receipt["status"], "invalid-manifest")
            self.assertIsNone(receipt["contract_binding_verification"])
            self.assertIn("does not match", receipt["manifest"]["validation_error"])
            self.assertFalse(marker.exists())
            self.assertFalse((directory / "fake-nvidia-smi-counter.txt").exists())

    def test_hash_matched_but_semantically_invalid_contract_rejects_before_gpu_or_phase_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100])
            marker = directory / "phase-command-ran.txt"
            phases = [
                {"name": name, "argv": ["/usr/bin/true"], "timeout_seconds": 5}
                for name in PHASES
            ]
            phases[0]["argv"] = ["/bin/sh", "-c", f"touch {marker}"]
            manifest = self.manifest(executable, phases)
            self.bind_contract_files(directory, manifest)
            bindings = manifest["contract_bindings"]
            assert isinstance(bindings, dict)
            model_descriptor = bindings["model_descriptor"]
            assert isinstance(model_descriptor, dict)
            descriptor = directory / str(model_descriptor["path"])
            descriptor.write_text("{}\n", encoding="utf-8")
            model_descriptor["sha256"] = sha256_bytes(descriptor.read_bytes())
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 2)
            self.assertEqual(receipt["status"], "invalid-manifest")
            self.assertIn("N01 model/numerical contract is invalid", receipt["manifest"]["validation_error"])
            self.assertFalse(marker.exists())
            self.assertFalse((directory / "fake-nvidia-smi-counter.txt").exists())

    def test_materialized_checkpoint_digest_is_verified_before_execution_and_receipted_when_valid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100] * 64)
            manifest = self.manifest(executable)
            self.bind_contract_files(directory, manifest)
            checkpoint = directory / "checkpoint-provenance.json"
            checkpoint.write_text('{"checkpoint":"fixture"}\n', encoding="utf-8")
            bindings = manifest["contract_bindings"]
            assert isinstance(bindings, dict)
            bindings["checkpoint_provenance"] = {
                "status": "materialized",
                "path": checkpoint.name,
                "sha256": "0" * 64,
            }
            manifest_path = directory / "manifest.json"
            bad_receipt_path = directory / "bad-receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, bad_receipt_path)
            bad_receipt = json.loads(bad_receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 2)
            self.assertIn("checkpoint_provenance.sha256 does not match", bad_receipt["manifest"]["validation_error"])
            self.assertFalse((directory / "fake-nvidia-smi-counter.txt").exists())

            bindings["checkpoint_provenance"]["sha256"] = sha256_bytes(checkpoint.read_bytes())  # type: ignore[index]
            good_receipt_path = directory / "good-receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, good_receipt_path)
            receipt = json.loads(good_receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 0, receipt)
            self.assertEqual(
                receipt["contract_binding_verification"]["checkpoint_provenance"],
                {
                    "status": "materialized",
                    "resolved_path": str(checkpoint.resolve()),
                    "verified_sha256": sha256_bytes(checkpoint.read_bytes()),
                },
            )

    def test_duplicate_manifest_key_is_rejected_before_gpu_or_phase_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100])
            manifest = self.manifest(executable)
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            encoded = json.dumps(manifest)
            manifest_path.write_text(encoded[:-1] + ',"experiment_id":"duplicate"}', encoding="utf-8")

            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 2)
            self.assertEqual(receipt["status"], "invalid-manifest")
            self.assertIn("duplicate JSON key 'experiment_id'", receipt["manifest"]["validation_error"])
            self.assertFalse((directory / "fake-nvidia-smi-counter.txt").exists())

    def test_unexpected_controller_failure_still_attempts_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100] * 32)
            shutdown_marker = directory / "shutdown-ran.txt"
            phases = [
                {"name": name, "argv": ["/usr/bin/true"], "timeout_seconds": 5}
                for name in PHASES
            ]
            phases[-1]["argv"] = ["/bin/sh", "-c", f"printf shutdown > {shutdown_marker}"]
            manifest = self.manifest(executable, phases)
            self.bind_contract_files(directory, manifest)
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            real_execute_phase = lifecycle.execute_phase

            def fail_asset_preparation(phase, *args, **kwargs):
                if phase["name"] == "asset-preparation":
                    raise RuntimeError("injected controller failure")
                return real_execute_phase(phase, *args, **kwargs)

            with mock.patch.dict(os.environ, environment), mock.patch.object(
                lifecycle,
                "execute_phase",
                side_effect=fail_asset_preparation,
            ):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 1)
            self.assertEqual(shutdown_marker.read_text(encoding="utf-8"), "shutdown")
            self.assertEqual(receipt["phases"][-1]["state"], "completed")
            self.assertIn("controller failure: RuntimeError: injected controller failure", receipt["failure_reasons"])

    def test_missing_in_process_samples_rejects_sampled_peak_as_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            executable, environment = self.make_fake_nvidia_smi(directory, [100] * 32)
            phases = [
                {"name": name, "argv": ["/usr/bin/true"], "timeout_seconds": 5}
                for name in PHASES
            ]
            manifest = self.manifest(executable, phases)
            self.bind_contract_files(directory, manifest)
            manifest_path = directory / "manifest.json"
            receipt_path = directory / "receipt.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.dict(os.environ, environment):
                result = run_manifest(manifest_path, receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(result, 1)
            self.assertEqual(receipt["overall"]["sampled_peak_evidence_status"], "insufficient-in-process-sampling")
            self.assertEqual(receipt["overall"]["missing_in_process_sample_phases"], ["warmup", "timed-serving"])
            self.assertIn("sampled GPU peak lacks an in-process poll for warmup, timed-serving", receipt["failure_reasons"])

    def test_n01_manifest_rejects_multi_gpu_before_per_device_budget_schema_exists(self):
        manifest = self.manifest(Path("/usr/bin/true"))
        manifest["gpu"]["gpu_indices"] = [0, 1]  # type: ignore[index]
        with self.assertRaisesRegex(ManifestError, "only physical GPU index"):
            validate_manifest(manifest)

    def test_nvidia_smi_parser_selects_physical_gpus_and_converts_mib_to_bytes(self):
        devices = parse_nvidia_smi(
            "0, GPU-zero, Test GPU 0, 400, 24576\n1, GPU-one, Test GPU 1, 600.5, 40960\n",
            [1, 0],
        )
        self.assertEqual([device["index"] for device in devices], [1, 0])
        self.assertEqual(devices[0]["used_bytes"], int(600.5 * MIB_BYTES))
        self.assertEqual(sum(device["used_bytes"] for device in devices), int(1000.5 * MIB_BYTES))

    def test_linux_proc_parsers_preserve_controller_rss_and_host_memory_units(self):
        self.assertEqual(parse_proc_status_rss("Name:\tpython\nVmRSS:\t  1234 kB\n"), 1234 * 1024)
        self.assertIsNone(parse_proc_status_rss("Name:\tpython\n"))
        memory = parse_proc_meminfo("MemTotal:       4000 kB\nMemAvailable:  1250 kB\n")
        self.assertEqual(memory["MemTotal"], 4000 * 1024)
        self.assertEqual(memory["MemAvailable"], 1250 * 1024)

    def test_schemas_bind_the_fixed_policy_budgets_and_phase_order(self):
        schema_root = Path(__file__).parents[1] / "schemas"
        manifest_schema = json.loads((schema_root / "n01-experiment-manifest-v1.schema.json").read_text(encoding="utf-8"))
        receipt_schema = json.loads((schema_root / "n01-lifecycle-receipt-v1.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_schema["properties"]["lifecycle_policy"]["const"], "leave_stopped_no_restore")
        self.assertEqual(manifest_schema["properties"]["memory_budget"]["properties"]["global_peak_ceiling_bytes"]["const"], GLOBAL_GPU_PEAK_CEILING_BYTES)
        self.assertEqual(manifest_schema["properties"]["memory_budget"]["properties"]["uncertainty_reserve_bytes"]["const"], UNCERTAINTY_RESERVE_BYTES)
        self.assertEqual(receipt_schema["properties"]["phase_order"]["const"], list(PHASES))
        self.assertEqual(receipt_schema["properties"]["lifecycle_policy"]["const"], "leave_stopped_no_restore")
        self.assertIn("contract_bindings", manifest_schema["required"])
        self.assertIn("target", manifest_schema["required"])
        self.assertIn("contract_bindings", receipt_schema["required"])
        self.assertIn("contract_binding_verification", receipt_schema["required"])
        self.assertEqual(manifest_schema["properties"]["gpu"]["properties"]["gpu_indices"]["const"], [0])
        self.assertEqual(
            manifest_schema["$defs"]["checkpointProvenance"]["oneOf"][1]["properties"]["status"]["const"],
            "materialization-pending",
        )


if __name__ == "__main__":
    unittest.main()
