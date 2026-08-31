from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_campaign  # noqa: E402
import competitive_common  # noqa: E402
import run_campaign  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(competitive_common.canonical_json_bytes(value))


class CampaignFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_root = SCRIPTS.parents[2]
        self.contract_path = self.root / competitive_common.CANONICAL_CONTRACT_RELATIVE_PATH
        self.matrix_path = self.root / "campaigns" / "latency-concrete.json"
        self.riley_lane_path = self.root / "campaigns" / "riley-pinned.json"
        self.competitor_lane_path = self.root / "campaigns" / "vllm-pinned.json"
        self.requests_path = self.root / "campaigns" / "requests.json"
        self.preflight_path = self.root / "campaigns" / "preflight.txt"
        self.plan_path = self.root / "campaigns" / "plan.json"
        self.raw_path = self.root / "campaigns" / "raw.jsonl"
        self._write_assets()
        self.plan = self._build_plan()
        _write_json(self.plan_path, self.plan)
        self.plan_sha256 = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        self.rows = self._rows()
        self.write_rows()

    def close(self) -> None:
        self.temporary.cleanup()

    def _write_assets(self) -> None:
        for relative_path in competitive_common.CANONICAL_ASSET_SHA256:
            source = self.source_root / relative_path
            target = self.root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        source_preflight = self.source_root / competitive_common.CANONICAL_PREFLIGHT_RELATIVE_PATH
        target_preflight = self.root / competitive_common.CANONICAL_PREFLIGHT_RELATIVE_PATH
        target_preflight.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_preflight, target_preflight)
        self.matrix_path.parent.mkdir(parents=True, exist_ok=True)

        parent_contract = {
            "path": competitive_common.CANONICAL_CONTRACT_RELATIVE_PATH,
            "sha256": competitive_common.CANONICAL_ASSET_SHA256[
                competitive_common.CANONICAL_CONTRACT_RELATIVE_PATH
            ],
            "schema_version": competitive_common.CONTRACT_SCHEMA_VERSION,
            "contract_id": competitive_common.CANONICAL_CONTRACT_ID,
        }
        latency_parent_path = "benchmarks/competitive/matrices/latency-sm89-bf16-v1.json"
        latency_parent = competitive_common.load_json(self.root / latency_parent_path)
        self.preflight_path.write_text(
            "\n".join(
                (
                    "environment_id=rtx4090-ubuntu22-driver580-v1",
                    "os_id=ubuntu",
                    "os_version_id=22.04",
                    "kernel_release=6.8.0-138-generic",
                    "machine=x86_64",
                    "cpu_model=Intel Core i7-13700K",
                    "physical_cpu_cores=16",
                    "logical_cpu_threads=24",
                    "ram_bytes=67185598464",
                    "git_revision=" + "d" * 40,
                    "gpu_name=NVIDIA GeForce RTX 4090",
                    "compute_capability=8.9",
                    "memory_total_mib=24564",
                    "memory_used_mib=0",
                    "driver_version=580.173.02",
                    "persistence_mode=Disabled",
                    "temperature_c=35",
                    "power_limit_w=450.00",
                    "graphics_clock_mhz=2550",
                    "memory_clock_mhz=10501",
                    "cpu_governor=powersave",
                    "cpu_governor_policy_count=24",
                    "clock_synchronized=yes",
                    "staging_available_bytes=32212254720",
                    "staging_minimum_bytes=21474836480",
                    "",
                )
            ),
            encoding="utf-8",
        )
        _write_json(
            self.matrix_path,
            {
                "schema_version": competitive_common.MATRIX_SCHEMA_VERSION,
                "matrix_id": "campaign-latency-v1",
                "tier": latency_parent["tier"],
                "measurement_mode": latency_parent["measurement_mode"],
                "hardware_requirement": latency_parent["hardware_requirement"],
                "model": latency_parent["model"],
                "preflight": latency_parent["preflight"],
                "parent_contract": parent_contract,
                "parent_asset": {
                    "path": latency_parent_path,
                    "sha256": competitive_common.CANONICAL_ASSET_SHA256[latency_parent_path],
                    "schema_version": competitive_common.MATRIX_SCHEMA_VERSION,
                    "asset_id": latency_parent["matrix_id"],
                },
                "cells": [
                    {
                        "cell_id": "c1-p128-o32",
                        "measurement_mode": "engine-only",
                        "warm_state": "warm",
                        "sampling": "greedy",
                        "eos_policy": "ignore-eos",
                        "cache_policy": "cache-off",
                        "arrival_mode": "closed-loop",
                        "arrival_schedule_id": "closed-loop-v1",
                        "client_behavior": "normal",
                        "cancellation_rate_percent": 0,
                        "slo_profile_id": "test-slo-v1",
                        "required_for": ["m4", "m5"],
                        "primary": True,
                        "concurrency": 2,
                        "prompt_tokens": 128,
                        "requested_output_tokens": 32,
                    }
                ],
            },
        )
        for path, parent_path in (
            (self.riley_lane_path, "benchmarks/competitive/lanes/riley.json"),
            (self.competitor_lane_path, "benchmarks/competitive/lanes/vllm-current.json"),
        ):
            parent_lane = competitive_common.load_json(self.root / parent_path)
            engine = parent_lane["engine"]["name"]
            _write_json(
                path,
                {
                    "schema_version": competitive_common.LANE_SCHEMA_VERSION,
                    "lane_id": parent_lane["lane_id"],
                    "role": parent_lane["role"],
                    "availability": "available",
                    "engine": {
                        "name": engine,
                        "backend": parent_lane["engine"]["backend"],
                        "pin_status": "pinned",
                        "version": "test-1.0.0",
                        "revision": "a" * 40,
                        "dependency_lock_sha256": "7" * 64,
                    },
                    "command": {
                        "status": "available",
                        "argv": [engine, "benchmark", "--result-dir", "{result_dir}"],
                        "required_placeholders": ["result_dir"],
                        "output_format": competitive_common.RAW_SCHEMA_VERSION,
                    },
                    "artifact_requirements": parent_lane["artifact_requirements"],
                    "artifact_receipts": {
                        "executable_sha256": "8" * 64,
                        "source_or_wheel_sha256": "9" * 64,
                        "dependency_lock_sha256": "7" * 64,
                        "runtime_options_sha256": "a" * 64,
                        "model_identity_sha256": "b" * 64,
                        "tokenizer_identity_sha256": "c" * 64,
                    },
                    "parent_contract": parent_contract,
                    "parent_asset": {
                        "path": parent_path,
                        "sha256": competitive_common.CANONICAL_ASSET_SHA256[parent_path],
                        "schema_version": competitive_common.LANE_SCHEMA_VERSION,
                        "asset_id": parent_lane["lane_id"],
                    },
                },
            )
        _write_json(
            self.requests_path,
            {
                "schema_version": competitive_common.REQUESTS_SCHEMA_VERSION,
                "manifest_id": "test-requests-v1",
                "parent_contract": parent_contract,
                "model_identity": {
                    "model_id": "test/model",
                    "model_revision": "a" * 40,
                    "weights_sha256": "1" * 64,
                    "tokenizer_revision": "b" * 40,
                    "tokenizer_files_sha256": {"tokenizer.json": "2" * 64},
                    "tokenizer_aggregate_sha256": "3" * 64,
                },
                "slo_profiles": [
                    {
                        "slo_profile_id": "test-slo-v1",
                        "ttft_ms_max": 25,
                        "tpot_ms_max": 5,
                    }
                ],
                "request_sets": [
                    {
                        "cell_id": "c1-p128-o32",
                        "requests": [
                            {
                                "request_id": "request-0",
                                "prompt_token_ids_sha256": "4" * 64,
                                "prompt_tokens": 128,
                                "requested_output_tokens": 32,
                                "sampling": "greedy",
                                "seed": None,
                                "eos_policy": "ignore-eos",
                                "cache_policy": "cache-off",
                                "arrival_schedule_id": "closed-loop-v1",
                            },
                            {
                                "request_id": "request-1",
                                "prompt_token_ids_sha256": "6" * 64,
                                "prompt_tokens": 128,
                                "requested_output_tokens": 32,
                                "sampling": "greedy",
                                "seed": None,
                                "eos_policy": "ignore-eos",
                                "cache_policy": "cache-off",
                                "arrival_schedule_id": "closed-loop-v1",
                            }
                        ],
                    }
                ],
            },
        )

    def _build_plan(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "root": self.root,
            "contract_path": self.contract_path,
            "matrix_paths": [self.matrix_path],
            "riley_lane_path": self.riley_lane_path,
            "competitor_lane_path": self.competitor_lane_path,
            "request_manifest_path": self.requests_path,
            "preflight_receipt_path": self.preflight_path,
            "campaign_id": "test-campaign-v1",
            "created_at_utc": "2026-08-28T00:00:00Z",
            "allow_dirty_source": False,
            "require_executable_lanes": True,
        }
        arguments.update(overrides)
        with patch.object(
            run_campaign,
            "source_receipt",
            return_value={
                "git_revision": "d" * 40,
                "git_dirty": False,
                "development_only": False,
            },
        ):
            return run_campaign.build_plan(**arguments)  # type: ignore[arg-type]

    def _rows(self) -> list[dict[str, object]]:
        request_sha = hashlib.sha256(self.requests_path.read_bytes()).hexdigest()
        workload_receipts = {
            str(receipt["cell_id"]): receipt
            for receipt in self.plan["workloads"]  # type: ignore[index]
        }
        rows: list[dict[str, object]] = []
        for invocation in self.plan["invocations"]:  # type: ignore[index]
            role = invocation["role"]  # type: ignore[index]
            riley = role == "riley"
            workload_receipt = workload_receipts[str(invocation["cell_id"])]
            rows.append(
                {
                    "schema_version": competitive_common.RAW_SCHEMA_VERSION,
                    "campaign_id": self.plan["campaign_id"],
                    "campaign_plan_sha256": self.plan_sha256,
                    "invocation_id": invocation["invocation_id"],
                    "lane_id": invocation["lane_id"],
                    "role": role,
                    "execution_id": invocation["execution_id"],
                    "cell_id": invocation["cell_id"],
                    "run_index": invocation["run_index"],
                    "order": invocation["order"],
                    "position": invocation["position"],
                    "measurement_mode": "engine-only",
                    "request_manifest_sha256": request_sha,
                    "workload_sha256": workload_receipt["sha256"],
                    "workload": competitive_common.workload_execution_receipt(
                        workload_receipt["value"]  # type: ignore[arg-type]
                    ),
                    "recorded_at_utc": "2026-08-28T00:00:01Z",
                    "source": self.plan["source"],
                    "environment": {
                        "gpu_uuid": "GPU-test-0001",
                        "gpu_model": "NVIDIA GeForce RTX 4090",
                        "compute_capability": "8.9",
                        "gpu_count": 1,
                        "driver_version": "580.173.02",
                        "cuda_runtime_version": "12.8",
                        "cuda_toolkit_version": "12.8",
                        "container_image_digest": "sha256:" + "1" * 64,
                        "git_commit": "d" * 40,
                        "source_archive_sha256": "2" * 64,
                        "lane_command_sha256": ("3" if riley else "4") * 64,
                        "engine_version": "test-1.0.0",
                        "engine_revision": "a" * 40,
                        "engine_options_sha256": ("5" if riley else "6") * 64,
                        "model_id": "test/model",
                        "model_revision": "a" * 40,
                        "model_weights_sha256": "1" * 64,
                        "tokenizer_revision": "b" * 40,
                        "tokenizer_files_sha256": "3" * 64,
                        "dtype": "bf16",
                        "clock_policy": "application-clocks-pinned",
                        "power_limit_watts": 450,
                    },
                    "phase": "measured",
                    "status": "success",
                    "failure_reason": None,
                    "metrics": {
                        "output_tokens_per_second": 110.0 if riley else 100.0,
                        "slo_goodput_tokens_per_second": 110.0 if riley else 100.0,
                        "peak_vram_bytes": 1040.0 if riley else 1000.0,
                        "usable_kv_bytes": 10000.0,
                    },
                    "requests": [
                        {
                            "request_id": "request-0",
                            "prompt_token_ids_sha256": "4" * 64,
                            "prompt_tokens": 128,
                            "generated_token_ids_sha256": "5" * 64,
                            "status": "success",
                            "failure_reason": None,
                            "requested_output_tokens": 32,
                            "sampling": "greedy",
                            "seed": None,
                            "eos_policy": "ignore-eos",
                            "cache_policy": "cache-off",
                            "arrival_schedule_id": "closed-loop-v1",
                            "generated_tokens": 32,
                            "ttft_ms": 9.0 if riley else 10.0,
                            "tpot_ms": 1.8 if riley else 2.0,
                            "end_to_end_ms": 66.0 if riley else 72.0,
                            "terminal_event_count": 1,
                        },
                        {
                            "request_id": "request-1",
                            "prompt_token_ids_sha256": "6" * 64,
                            "prompt_tokens": 128,
                            "generated_token_ids_sha256": "7" * 64,
                            "status": "success",
                            "failure_reason": None,
                            "requested_output_tokens": 32,
                            "sampling": "greedy",
                            "seed": None,
                            "eos_policy": "ignore-eos",
                            "cache_policy": "cache-off",
                            "arrival_schedule_id": "closed-loop-v1",
                            "generated_tokens": 32,
                            "ttft_ms": 9.0 if riley else 10.0,
                            "tpot_ms": 1.8 if riley else 2.0,
                            "end_to_end_ms": 66.0 if riley else 72.0,
                            "terminal_event_count": 1,
                        }
                    ],
                }
            )
        return rows

    def write_rows(self) -> None:
        self.raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=True) + "\n" for row in self.rows),
            encoding="utf-8",
        )


class CampaignCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = CampaignFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def report(self, *, current_source: dict[str, object] | None = None) -> dict[str, object]:
        if current_source is None:
            current_source = {"git_revision": "d" * 40, "git_dirty": False}
        with patch.object(check_campaign, "current_source_receipt", return_value=current_source):
            return check_campaign.check_campaign(
                plan_path=self.fixture.plan_path,
                raw_paths=[self.fixture.raw_path],
                root=self.fixture.root,
            )

    def test_passed_report_is_byte_identical_on_replay(self) -> None:
        first = self.report()
        second = self.report()
        self.assertEqual(first["status"], "passed")
        self.assertEqual(
            competitive_common.canonical_json_bytes(first),
            competitive_common.canonical_json_bytes(second),
        )

    def test_generated_token_mismatch_fails_without_statistics(self) -> None:
        competitor = next(row for row in self.fixture.rows if row["role"] == "competitor")
        competitor["requests"][0]["generated_token_ids_sha256"] = "6" * 64  # type: ignore[index]
        self.fixture.write_rows()
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cells"], [])

    def test_lane_failure_is_failed_without_success_percentiles(self) -> None:
        failed = self.fixture.rows[0]
        failed["status"] = "failure"
        failed["failure_reason"] = "synthetic launch failure"
        failed["metrics"] = None
        failed["requests"] = []
        self.fixture.write_rows()
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["cells"], [])

    def test_partial_win_is_a_closed_distinct_outcome(self) -> None:
        for row in self.fixture.rows:
            if row["role"] != "riley":
                continue
            row["metrics"]["slo_goodput_tokens_per_second"] = 105.0  # type: ignore[index]
            row["requests"][0]["tpot_ms"] = 2.0  # type: ignore[index]
        self.fixture.write_rows()
        report = self.report()
        self.assertEqual(report["status"], "partial-win")
        self.assertTrue(report["m4"]["passed"])  # type: ignore[index]
        self.assertFalse(report["m5"]["passed"])  # type: ignore[index]

    def test_cheap_or_different_request_workload_cannot_pass(self) -> None:
        request = self.fixture.rows[0]["requests"][0]  # type: ignore[index]
        request["prompt_tokens"] = 1
        request["requested_output_tokens"] = 1
        request["sampling"] = {
            "id": "seeded-top-p-v1",
            "temperature": 0.7,
            "top_p": 0.9,
        }
        request["seed"] = 17
        request["eos_policy"] = "stop-on-eos"
        request["cache_policy"] = "controlled-prefix-hit-90"
        request["arrival_schedule_id"] = "cheap-arrival-v1"
        self.fixture.write_rows()

        report = self.report()
        self.assertEqual(report["status"], "failed")
        failures = report["evidence"]["semantic_failures"]  # type: ignore[index]
        self.assertTrue(any("prompt_tokens differs" in item for item in failures))
        self.assertTrue(any("sampling differs" in item for item in failures))
        self.assertTrue(any("seed differs" in item for item in failures))
        self.assertTrue(any("arrival_schedule_id differs" in item for item in failures))

    def test_different_warm_client_cancellation_or_slo_cannot_pass(self) -> None:
        workload = self.fixture.rows[0]["workload"]  # type: ignore[index]
        workload["warm_state"] = "cold"
        workload["arrival_mode"] = "open-loop"
        workload["client_behavior"] = "backpressure"
        workload["cancellation_rate_percent"] = 100
        workload["slo_profile"]["ttft_ms_max"] = 9999  # type: ignore[index]
        self.fixture.write_rows()

        report = self.report()
        self.assertEqual(report["status"], "incomparable")
        self.assertIn("model/warm/arrival/client/SLO behavior", report["comparability"]["reasons"][0])  # type: ignore[index]

    def test_self_authored_plan_workload_receipt_is_rederived(self) -> None:
        plan = json.loads(self.fixture.plan_path.read_text(encoding="utf-8"))
        receipt = plan["workloads"][0]
        receipt["value"]["requests"][0]["prompt_tokens"] = 1
        receipt["sha256"] = competitive_common.sha256_bytes(
            competitive_common.canonical_json_bytes(receipt["value"])
        )
        self._write_plan_and_rebind_raw(plan)

        report = self.report()
        self.assertEqual(report["status"], "incomparable")
        self.assertIn("immutable matrix/request workload", report["comparability"]["reasons"][0])  # type: ignore[index]

    def test_each_independent_run_requires_full_manifest_coverage(self) -> None:
        run = next(
            row
            for row in self.fixture.rows
            if row["role"] == "riley" and row["run_index"] == 1
        )
        run["requests"].pop()  # type: ignore[index]
        self.fixture.write_rows()

        report = self.report()
        self.assertEqual(report["status"], "failed")
        failures = report["evidence"]["semantic_failures"]  # type: ignore[index]
        self.assertTrue(any("request set differs" in item for item in failures))

    def test_p95_is_median_of_per_run_p95_summaries(self) -> None:
        divergent_run = next(
            row
            for row in self.fixture.rows
            if row["role"] == "riley" and row["run_index"] == 1
        )
        for request in divergent_run["requests"]:  # type: ignore[index]
            request["ttft_ms"] = 1000.0
            request["tpot_ms"] = 100.0
        self.fixture.write_rows()

        report = self.report()
        self.assertEqual(report["status"], "passed")
        riley = report["cells"][0]["riley"]  # type: ignore[index]
        self.assertEqual(riley["ttft_p95_ms"], 9.0)
        self.assertEqual(riley["tpot_p95_ms"], 1.8)
        summaries = riley["per_run_summaries"]
        self.assertEqual(len(summaries), 5)
        self.assertEqual(summaries[0]["run_index"], 1)
        self.assertEqual(summaries[0]["ttft_p95_ms"], 1000.0)

    def test_environment_mixing_is_incomparable(self) -> None:
        self.fixture.rows[-1]["environment"]["gpu_uuid"] = "GPU-other"  # type: ignore[index]
        self.fixture.write_rows()
        report = self.report()
        self.assertEqual(report["status"], "incomparable")
        self.assertIn("GPU UUID/driver/CUDA environment differs", report["comparability"]["reasons"][0])  # type: ignore[index]

    def test_warmup_rows_and_unknown_fields_are_rejected(self) -> None:
        self.fixture.rows[0]["phase"] = "warmup"
        self.fixture.write_rows()
        self.assertEqual(self.report()["status"], "incomparable")
        self.fixture.rows[0]["phase"] = "measured"
        self.fixture.rows[0]["not_in_schema"] = True
        self.fixture.write_rows()
        self.assertEqual(self.report()["status"], "incomparable")

    def test_nan_is_incomparable(self) -> None:
        self.fixture.rows[0]["metrics"]["peak_vram_bytes"] = float("nan")  # type: ignore[index]
        self.fixture.write_rows()
        self.assertEqual(self.report()["status"], "incomparable")

    def test_duplicate_keys_and_invalid_hashes_are_incomparable(self) -> None:
        self.fixture.rows[0]["campaign_plan_sha256"] = "not-a-sha"
        self.fixture.write_rows()
        self.assertEqual(self.report()["status"], "incomparable")
        self.fixture.rows[0]["campaign_plan_sha256"] = self.fixture.plan_sha256
        self.fixture.write_rows()
        with self.fixture.raw_path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema_version":"x","schema_version":"x"}\n')
        self.assertEqual(self.report()["status"], "incomparable")

    def test_insufficient_independent_runs_are_incomparable(self) -> None:
        plan = json.loads(self.fixture.plan_path.read_text(encoding="utf-8"))
        plan["execution"] = [entry for entry in plan["execution"] if entry["run_index"] != 5]
        plan["invocations"] = [entry for entry in plan["invocations"] if entry["run_index"] != 5]
        _write_json(self.fixture.plan_path, plan)
        self.assertEqual(self.report()["status"], "incomparable")

    def test_role_inversion_is_incomparable(self) -> None:
        self.fixture.rows[0]["role"] = "competitor"
        self.fixture.write_rows()
        self.assertEqual(self.report()["status"], "incomparable")

    def test_default_plan_creation_rejects_dirty_source(self) -> None:
        with patch.object(run_campaign, "_git_output", side_effect=["d" * 40 + "\n", " M benchmark"]):
            with self.assertRaisesRegex(competitive_common.ContractError, "dirty source"):
                run_campaign.source_receipt(self.fixture.root, allow_dirty_source=False)

    def test_missing_or_failed_preflight_cannot_make_a_ready_plan(self) -> None:
        with patch.object(
            run_campaign,
            "source_receipt",
            return_value={
                "git_revision": "d" * 40,
                "git_dirty": False,
                "development_only": False,
            },
        ):
            plan = run_campaign.build_plan(
                root=self.fixture.root,
                contract_path=self.fixture.contract_path,
                matrix_paths=[self.fixture.matrix_path],
                riley_lane_path=self.fixture.riley_lane_path,
                competitor_lane_path=self.fixture.competitor_lane_path,
                request_manifest_path=self.fixture.requests_path,
                preflight_receipt_path=None,
                campaign_id="no-preflight-v1",
                created_at_utc="2026-08-28T00:00:00Z",
                allow_dirty_source=False,
                require_executable_lanes=False,
            )
        self.assertEqual(plan["readiness"]["state"], "blocked")
        self.fixture.preflight_path.write_text(
            self.fixture.preflight_path.read_text(encoding="utf-8").replace(
                "memory_used_mib=0", "memory_used_mib=257"
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(competitive_common.ContractError, "memory_used_mib"):
            run_campaign.load_preflight_receipt(
                self.fixture.preflight_path,
                {"git_revision": "d" * 40},
            )

    def _write_plan_and_rebind_raw(self, plan: dict[str, object]) -> None:
        self.fixture.plan = plan
        _write_json(self.fixture.plan_path, plan)
        self.fixture.plan_sha256 = hashlib.sha256(self.fixture.plan_path.read_bytes()).hexdigest()
        workload_receipts = {
            str(receipt["cell_id"]): receipt
            for receipt in plan["workloads"]  # type: ignore[index]
        }
        for row in self.fixture.rows:
            row["campaign_plan_sha256"] = self.fixture.plan_sha256
            row["source"] = plan["source"]
            receipt = workload_receipts[str(row["cell_id"])]
            row["workload_sha256"] = receipt["sha256"]
            row["workload"] = competitive_common.workload_execution_receipt(
                receipt["value"]  # type: ignore[arg-type]
            )
        self.fixture.write_rows()

    def test_weak_self_authored_contract_is_not_a_campaign_contract(self) -> None:
        weak_contract = competitive_common.load_json(self.fixture.contract_path)
        weak_contract["m5"]["ttft_p95_ratio_max"] = 99.0
        path = self.fixture.root / "campaigns" / "weak-contract.json"
        _write_json(path, weak_contract)
        with self.assertRaisesRegex(competitive_common.ContractError, "canonical contract"):
            self.fixture._build_plan(contract_path=path)

    def test_unparented_self_authored_matrix_and_lane_are_rejected(self) -> None:
        weak_matrix = competitive_common.load_json(self.fixture.matrix_path)
        weak_matrix.pop("parent_contract")
        weak_matrix.pop("parent_asset")
        matrix_path = self.fixture.root / "campaigns" / "weak-matrix.json"
        _write_json(matrix_path, weak_matrix)
        with self.assertRaisesRegex(competitive_common.ContractError, "parent_contract"):
            self.fixture._build_plan(matrix_paths=[matrix_path])

        weak_lane = competitive_common.load_json(self.fixture.riley_lane_path)
        weak_lane.pop("parent_contract")
        weak_lane.pop("parent_asset")
        lane_path = self.fixture.root / "campaigns" / "weak-riley.json"
        _write_json(lane_path, weak_lane)
        with self.assertRaisesRegex(competitive_common.ContractError, "parent_contract"):
            self.fixture._build_plan(riley_lane_path=lane_path)

    def test_forged_ready_state_and_mutated_preflight_cannot_pass(self) -> None:
        blocked = self.fixture._build_plan(
            preflight_receipt_path=None,
            require_executable_lanes=False,
        )
        blocked["readiness"] = {"state": "ready", "blocked_reasons": []}
        self._write_plan_and_rebind_raw(blocked)
        report = self.report()
        self.assertEqual(report["status"], "incomparable")
        self.assertIn("no passed reviewed", report["comparability"]["reasons"][0])  # type: ignore[index]

        plan = self.fixture._build_plan()
        self.fixture.preflight_path.write_text(
            self.fixture.preflight_path.read_text(encoding="utf-8").replace(
                "memory_used_mib=0", "memory_used_mib=257"
            ),
            encoding="utf-8",
        )
        plan["preflight"]["sha256"] = hashlib.sha256(self.fixture.preflight_path.read_bytes()).hexdigest()  # type: ignore[index]
        plan["preflight"]["values"]["memory_used_mib"] = "257"  # type: ignore[index]
        self._write_plan_and_rebind_raw(plan)
        report = self.report()
        self.assertEqual(report["status"], "incomparable")
        self.assertIn("preflight", report["comparability"]["reasons"][0])  # type: ignore[index]

    def test_wrong_parent_receipt_is_rejected_even_when_asset_is_well_formed(self) -> None:
        weak_matrix = competitive_common.load_json(self.fixture.matrix_path)
        weak_matrix["parent_asset"]["sha256"] = "0" * 64
        matrix_path = self.fixture.root / "campaigns" / "wrong-parent-matrix.json"
        _write_json(matrix_path, weak_matrix)
        with self.assertRaisesRegex(competitive_common.ContractError, "does not match canonical"):
            self.fixture._build_plan(matrix_paths=[matrix_path])

    def test_unpinned_lane_cannot_pass_even_if_plan_claims_ready(self) -> None:
        unpinned_lane = competitive_common.load_json(self.fixture.riley_lane_path)
        unpinned_lane["availability"] = "campaign-pin-required"
        unpinned_lane["engine"]["pin_status"] = "campaign-pinned-required"
        unpinned_lane["engine"]["version"] = None
        unpinned_lane["engine"]["revision"] = None
        unpinned_lane["engine"]["dependency_lock_sha256"] = None
        unpinned_lane["command"]["status"] = "campaign-pin-required"
        unpinned_lane.pop("artifact_receipts")
        path = self.fixture.root / "campaigns" / "riley-unpinned.json"
        _write_json(path, unpinned_lane)
        blocked = self.fixture._build_plan(
            riley_lane_path=path,
            require_executable_lanes=False,
        )
        self.assertEqual(blocked["readiness"]["state"], "blocked")  # type: ignore[index]
        blocked["readiness"] = {"state": "ready", "blocked_reasons": []}
        self._write_plan_and_rebind_raw(blocked)
        report = self.report()
        self.assertEqual(report["status"], "incomparable")
        self.assertIn("not actually available", report["comparability"]["reasons"][0])  # type: ignore[index]

    def test_dirty_or_head_drift_at_claim_time_cannot_pass(self) -> None:
        dirty = self.report(current_source={"git_revision": "d" * 40, "git_dirty": True})
        self.assertEqual(dirty["status"], "incomparable")
        self.assertIn("current source tree is dirty", dirty["comparability"]["reasons"][0])  # type: ignore[index]

        drifted = self.report(current_source={"git_revision": "e" * 40, "git_dirty": False})
        self.assertEqual(drifted["status"], "incomparable")
        self.assertIn("current Git HEAD differs", drifted["comparability"]["reasons"][0])  # type: ignore[index]


class StaticManifestTests(unittest.TestCase):
    ROOT = SCRIPTS.parents[2]
    COMPETITIVE = ROOT / "benchmarks" / "competitive"

    def test_all_versioned_static_manifests_validate(self) -> None:
        competitive_common.validate_contract(
            competitive_common.load_json(self.COMPETITIVE / "contract-v1.json")
        )
        for path in sorted((self.COMPETITIVE / "matrices").glob("*.json")):
            competitive_common.validate_matrix(competitive_common.load_json(path), str(path))
        for path in sorted((self.COMPETITIVE / "lanes").glob("*.json")):
            competitive_common.validate_lane(competitive_common.load_json(path), str(path))

    def test_canonical_asset_ledger_matches_reviewed_content(self) -> None:
        for relative_path, expected_sha256 in competitive_common.CANONICAL_ASSET_SHA256.items():
            self.assertEqual(
                competitive_common.sha256_file(self.ROOT / relative_path),
                expected_sha256,
                relative_path,
            )
        self.assertEqual(
            competitive_common.sha256_file(
                self.ROOT / competitive_common.CANONICAL_PREFLIGHT_RELATIVE_PATH
            ),
            competitive_common.CANONICAL_PREFLIGHT_SHA256,
        )

    def test_static_manifest_unknown_fields_are_rejected(self) -> None:
        contract = competitive_common.load_json(self.COMPETITIVE / "contract-v1.json")
        contract = deepcopy(contract)
        contract["unreviewed"] = True
        with self.assertRaises(competitive_common.ContractError):
            competitive_common.validate_contract(contract)

        matrix = competitive_common.load_json(
            self.COMPETITIVE / "matrices" / "diagnostic-sm89-bf16-v1.json"
        )
        matrix = deepcopy(matrix)
        matrix["cells"][0]["unreviewed"] = True
        with self.assertRaises(competitive_common.ContractError):
            competitive_common.validate_matrix(matrix)

        lane = competitive_common.load_json(self.COMPETITIVE / "lanes" / "riley.json")
        lane = deepcopy(lane)
        lane["command"]["unreviewed"] = True
        with self.assertRaises(competitive_common.ContractError):
            competitive_common.validate_lane(lane)

    def test_diagnostic_prompt_artifact_hash_is_current(self) -> None:
        matrix = competitive_common.load_json(
            self.COMPETITIVE / "matrices" / "diagnostic-sm89-bf16-v1.json"
        )
        source = matrix["model"]["prompt_token_ids_source"]
        actual = hashlib.sha256((self.ROOT / source["path"]).read_bytes()).hexdigest()
        self.assertEqual(actual, source["sha256"])


if __name__ == "__main__":
    unittest.main()
