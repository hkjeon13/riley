#!/usr/bin/env python3
"""CPU-only adversarial tests for the C02 full-duration soak-v2 receipt."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import check_rc3_qualification as qualification
import check_soak_v2_receipt as soak


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class SoakV2Fixture:
    """A complete local immutable-evidence tree with a mocked raw replay edge."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.freeze_path = root / "riley-0.1.0-rc3.freeze.json"
        self.candidate_id = "riley-0.1.0-rc3"
        self.revision = "a" * 40
        self.base_relative = "reports/final.json"
        self.manifest_relative = "candidates/final-release-candidate.json"
        self.semantic_report_relative = "receipts/soak_v2.json"
        self.receipt_relative = "soak-v2/raw-receipt.json"
        self.trace_relative = "soak-v2/scenario-trace.json"
        self.gate_report_relative = "gate-e/reliability-soak-report.json"
        self.raw_archive_relative = "gate-e/reliability-soak.evidence.tar"
        self.golden_relative = "gate-e/python-free-correctness-golden.json"
        self.native_relative = "gate-e/native-correctness.json"
        self.receipt_path = self.evidence / self.receipt_relative
        self.trace_path = self.evidence / self.trace_relative
        self.gate_report_path = self.evidence / self.gate_report_relative
        self.raw_archive_path = self.evidence / self.raw_archive_relative
        self.golden_path = self.evidence / self.golden_relative
        self.native_path = self.evidence / self.native_relative
        self.contract = soak._load_contract()

        self.release = {
            "binary_sha256": digest("release binary"),
            "bundle_sha256": digest("release bundle"),
            "image_id": "sha256:" + digest("release image"),
            "cuda_c_abi_version": "12.8.1",
        }
        self.images = {
            "reproducible": "sha256:" + digest("reproducible image"),
            "cuda": "sha256:" + digest("cuda image"),
            "optimization": "sha256:" + digest("optimization image"),
        }
        stable_input = {
            "argv": ["serve", "--execution-completion", "iteration-batch"],
            "environment": {"RILEY_CONFIG_RECEIPT": "1"},
        }
        maximum_input = {
            "argv": ["serve", "--execution-completion", "per-operation"],
            "environment": {"RILEY_EXACT": "1"},
        }
        self.arms = {
            "stable_default": {
                **stable_input,
                "configuration_sha256": digest(qualification.canonical_json_bytes(stable_input)),
            },
            "max_performance_exact": {
                **maximum_input,
                "configuration_sha256": digest(qualification.canonical_json_bytes(maximum_input)),
            },
        }
        self.freeze: dict[str, object] = {
            "schema_version": qualification.FREEZE_VERSION,
            "candidate_id": self.candidate_id,
            "created_at_utc": "2026-08-28T00:00:00Z",
            "status": "frozen",
            "source": {
                "git_revision": self.revision,
                "archive_sha256": digest("source archive"),
                "cargo_lock_sha256": digest("cargo lock"),
                "extension_registry_sha256": digest("extension registry"),
                "correctness_golden_sha256": digest("correctness golden"),
            },
            "release": self.release,
            "images": self.images,
            "toolchain": {
                "rustc": "rustc 1.85.0",
                "nvcc": "Cuda compilation tools, release 12.8, V12.8.93",
                "driver": "580.173.02",
                "cuda_runtime": "12.8.1",
                "cuda_toolkit": "12.8.93",
                "cublas": "12.8.4.1",
            },
            "models": {
                name: {
                    "model_id": f"fixture/{name}",
                    "model_revision": "b" * 40,
                    "config_sha256": digest(f"{name} config"),
                    "weights_sha256": digest(f"{name} weights"),
                    "tokenizer_revision": "c" * 40,
                    "tokenizer_files_sha256": digest(f"{name} tokenizer"),
                }
                for name in ("smollm2", "qwen")
            },
            "arms": self.arms,
            "rollback": {
                "binary_sha256": digest("rollback binary"),
                "bundle_sha256": digest("rollback bundle"),
                "image_id": "sha256:" + digest("rollback image"),
            },
            "outputs": {
                "final_release_candidate_manifest": {"path": self.manifest_relative},
                "final_release_candidate": {"path": self.base_relative},
                "receipts": {
                    gate: {
                        "path": (
                            self.semantic_report_relative if gate == "soak_v2" else f"receipts/{gate}.json"
                        )
                    }
                    for gate in qualification.REQUIRED_GATES
                },
            },
            "required_gates": list(qualification.REQUIRED_GATES),
        }

    def write_canonical(self, path: Path, document: object) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = qualification.canonical_json_bytes(document)
        path.write_bytes(raw)
        return raw

    def write_bytes(self, path: Path, contents: bytes) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return contents

    def descriptor(self, relative: str, raw: bytes) -> dict[str, str]:
        return {"path": relative, "sha256": digest(raw)}

    def bindings(self) -> dict[str, str]:
        stable = self.arms["stable_default"]
        return {
            "freeze_sha256": self.freeze_sha,
            "base_release_candidate_report_sha256": self.base_sha,
            "configuration_profile": soak.STABLE_DEFAULT_PROFILE,
            "configuration_sha256": stable["configuration_sha256"],
        }

    def _gate_report(self) -> dict[str, object]:
        return {
            "schema_version": qualification.release_candidate.reliability_soak.REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "bindings": {
                "contract_id": qualification.release_candidate.SOAK_CONTRACT_ID,
                "reviewed_manifest_template_canonical_sha256": digest("template"),
                "manifest_sha256": digest("soak manifest"),
                "binding_sha256": digest("soak binding"),
                "trusted_correctness": {},
                "runtime_provenance": {},
                "source": {},
            },
            "scenario_summaries": [
                {"scenario_id": scenario_id}
                for scenario_id, _kind, _duration in soak.V1_SCENARIOS
            ],
            "observations": {},
            "checks": [],
            "errors": [],
        }

    def _base_report(self, evidence_hashes: dict[str, str]) -> dict[str, object]:
        return {
            "schema_version": qualification.release_candidate.REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "candidate_id": self.candidate_id,
            "manifest_sha256": digest("base manifest"),
            "bindings": {
                "git_revision": self.revision,
                "source_archive_sha256": self.freeze["source"]["archive_sha256"],
                "release_binary_sha256": self.release["binary_sha256"],
                "release_bundle_sha256": self.release["bundle_sha256"],
                "release_image_sha256": self.release["image_id"].removeprefix("sha256:"),
                "build_image_ids": {
                    "reproducible_build": self.images["reproducible"],
                    "cuda_fault": self.images["cuda"],
                    "optimization_correctness": self.images["optimization"],
                },
                "native_correctness_executable_sha256": digest("native executable"),
                "profile_binary_sha256": digest("profile binary"),
                "reproducibility_report_sha256": digest("repro report"),
                "correctness_golden_sha256": self.freeze["source"]["correctness_golden_sha256"],
                "evidence_sha256": evidence_hashes,
            },
            "checks": [{"name": name, "passed": True} for name in qualification.BASE_CHECKS],
            "errors": [],
        }

    def _scenario_results(self) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        now = 1
        for scenario in self.contract.scenarios:
            duration = scenario["duration_seconds"]
            attempted = max(100, scenario["minimum_requests"])
            cancellation_percent = scenario["cancellation_percent"]
            cancelled = attempted * cancellation_percent // 100
            capacity_rejections = scenario["minimum_capacity_rejections"]
            completed = attempted - cancelled - capacity_rejections
            target_kv_utilization = scenario["minimum_kv_utilization_percent"]
            kv_blocks = {
                "free": 100 - target_kv_utilization,
                "reserved": 0,
                "active": target_kv_utilization,
            }
            observations: list[dict[str, object]] = []
            for elapsed in range(0, duration, soak.EXPECTED_MAXIMUM_INTERVAL_SECONDS):
                terminal = attempted * elapsed // duration
                observed_completed = min(completed, terminal)
                after_completed = terminal - observed_completed
                observed_cancelled = min(cancelled, after_completed)
                observed_rejections = after_completed - observed_cancelled
                observations.append(
                    {
                        "elapsed_seconds": elapsed,
                        "rss_bytes": 1_024 + elapsed,
                        "pinned_bytes": 2_048,
                        "vram_bytes": 4_096 + elapsed,
                        "kv_blocks": copy.deepcopy(kv_blocks),
                        "request_states": {
                            "active": attempted - terminal,
                            "completed": observed_completed,
                            "cancelled": observed_cancelled,
                            "capacity_rejections": observed_rejections,
                        },
                        "terminal_events_total": terminal,
                    }
                )
            observations.append(
                {
                    "elapsed_seconds": duration,
                    "rss_bytes": 1_024 + duration,
                    "pinned_bytes": 2_048,
                    "vram_bytes": 4_096 + duration,
                    "kv_blocks": copy.deepcopy(kv_blocks),
                    "request_states": {
                        "active": 0,
                        "completed": completed,
                        "cancelled": cancelled,
                        "capacity_rejections": capacity_rejections,
                    },
                    "terminal_events_total": attempted,
                }
            )
            lifecycle = [
                {"cycle": cycle, "events": list(soak.MODEL_LIFECYCLE_EVENTS)}
                for cycle in range(1, scenario["minimum_model_load_unload_cycles"] + 1)
            ]
            results.append(
                {
                    "scenario_id": scenario["id"],
                    "started_monotonic_ns": now,
                    "ended_monotonic_ns": now + duration * 1_000_000_000,
                    "observed_duration_seconds": duration,
                    "attempted_requests": attempted,
                    "completed_requests": completed,
                    "cancelled_requests": cancelled,
                    "capacity_rejections": capacity_rejections,
                    "terminal_events": attempted,
                    "max_kv_utilization_percent": scenario["minimum_kv_utilization_percent"],
                    "exact_backend_fallbacks": scenario["minimum_exact_backend_fallbacks"],
                    "backend_events": copy.deepcopy(scenario["required_backend_events"]),
                    "model_lifecycle_cycles": lifecycle,
                    "interval_observations": observations,
                }
            )
            now += duration * 1_000_000_000 + 1
        return results

    def materialize(self) -> str:
        self.freeze_sha = digest(self.write_canonical(self.freeze_path, self.freeze))
        self.write_canonical(self.evidence / self.manifest_relative, {"fixture": "Gate E manifest"})
        self.gate_report_document = self._gate_report()
        self.gate_report_raw = self.write_canonical(self.gate_report_path, self.gate_report_document)
        self.raw_archive_raw = self.write_bytes(self.raw_archive_path, b"synthetic canonical Gate E archive")
        self.golden_raw = self.write_canonical(self.golden_path, {"fixture": "correctness golden"})
        self.native_raw = self.write_canonical(self.native_path, {"fixture": "native report"})
        evidence_hashes = {
            key: digest(f"base evidence {key}") for key in qualification.BASE_EVIDENCE_SHA256_KEYS
        }
        evidence_hashes.update(
            {
                "reliability_soak": digest(self.gate_report_raw),
                "reliability_soak_raw": digest(self.raw_archive_raw),
                "python_free_e2e_correctness_golden_raw": digest(self.golden_raw),
                "native_correctness_report": digest(self.native_raw),
            }
        )
        self.base_raw = self.write_canonical(
            self.evidence / self.base_relative, self._base_report(evidence_hashes)
        )
        self.base_sha = digest(self.base_raw)
        self.trace_document: dict[str, object] = {
            "schema_version": soak.TRACE_VERSION,
            "candidate_id": self.candidate_id,
            "bindings": self.bindings(),
            "scenario_contract": {
                "path": self.contract.descriptor.path,
                "sha256": self.contract.descriptor.sha256,
            },
            "gate_e_soak_raw_sha256": digest(self.raw_archive_raw),
            "scenario_results": self._scenario_results(),
        }
        self.trace_raw = self.write_canonical(self.trace_path, self.trace_document)
        self.receipt_document: dict[str, object] = {
            "schema_version": soak.RECEIPT_VERSION,
            "candidate_id": self.candidate_id,
            "bindings": self.bindings(),
            "scenario_contract": {
                "path": self.contract.descriptor.path,
                "sha256": self.contract.descriptor.sha256,
            },
            "gate_e_soak": {
                "report": self.descriptor(self.gate_report_relative, self.gate_report_raw),
                "raw_archive": self.descriptor(self.raw_archive_relative, self.raw_archive_raw),
                "correctness_golden": self.descriptor(self.golden_relative, self.golden_raw),
                "native_correctness_report": self.descriptor(self.native_relative, self.native_raw),
            },
            "scenario_trace": self.descriptor(self.trace_relative, self.trace_raw),
        }
        self.receipt_raw = self.write_canonical(self.receipt_path, self.receipt_document)
        return self.freeze_sha

    def rewrite_trace_and_receipt(self) -> None:
        self.trace_raw = self.write_canonical(self.trace_path, self.trace_document)
        self.receipt_document["scenario_trace"] = self.descriptor(self.trace_relative, self.trace_raw)
        self.receipt_raw = self.write_canonical(self.receipt_path, self.receipt_document)

    def replay_output(self) -> dict[str, object]:
        reliability = qualification.release_candidate.reliability_soak
        return {
            "report": copy.deepcopy(self.gate_report_document),
            "archive_sha256": digest(self.raw_archive_raw),
            **{f"{name}_sha256": digest(f"raw member {name}") for name in reliability.RAW_ARCHIVE_PAYLOADS},
        }


class SoakV2ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SoakV2Fixture(Path(self.temporary.name).resolve())
        self.freeze_sha = self.fixture.materialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _evaluate_call(self, receipt_path: str | None = None) -> dict[str, object]:
        return soak.evaluate(
            self.fixture.freeze_path,
            self.fixture.evidence,
            receipt_path or self.fixture.receipt_relative,
            expected_freeze_sha256=self.freeze_sha,
        )

    def evaluate(self, receipt_path: str | None = None) -> dict[str, object]:
        with (
            mock.patch.object(
                qualification,
                "revalidate_base_release_candidate",
                return_value=(self.fixture.base_raw, self.fixture.base_sha),
            ),
            mock.patch.object(
                qualification.release_candidate.reliability_soak,
                "replay_raw_evidence_archive",
                return_value=self.fixture.replay_output(),
            ),
        ):
            return self._evaluate_call(receipt_path)

    def test_valid_receipt_replays_gate_e_archive_and_full_contract(self) -> None:
        with (
            mock.patch.object(
                qualification,
                "revalidate_base_release_candidate",
                return_value=(self.fixture.base_raw, self.fixture.base_sha),
            ) as gate_e,
            mock.patch.object(
                qualification.release_candidate.reliability_soak,
                "replay_raw_evidence_archive",
                return_value=self.fixture.replay_output(),
            ) as archive_replay,
        ):
            report = self._evaluate_call()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["passed"])
        self.assertEqual(report["scenario_contract"], {
            "path": soak.CONTRACT_RELATIVE_PATH,
            "sha256": soak.CONTRACT_SHA256,
        })
        self.assertEqual(
            [result["scenario_id"] for result in report["scenario_results"]],
            list(soak.EXPECTED_SCENARIO_IDS),
        )
        self.assertEqual(len(report["checks"]), len(soak.CHECK_NAMES))
        parsed = soak.validate_check_report(report)
        self.assertEqual(parsed.receipt.path, self.fixture.receipt_relative)
        self.assertEqual(parsed.scenario_trace.path, self.fixture.trace_relative)
        self.assertEqual(len(parsed.scenario_results), len(soak.EXPECTED_SCENARIO_IDS))
        gate_e.assert_called_once()
        archive_replay.assert_called_once()
        replay_path = archive_replay.call_args.args[0]
        self.assertNotEqual(replay_path, self.fixture.raw_archive_path)
        self.assertFalse(str(replay_path).startswith(str(self.fixture.evidence)))

    def test_gate_e_failure_blocks_raw_soak_receipt(self) -> None:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            side_effect=qualification.GateFailure("Gate E failed"),
        ):
            report = self._evaluate_call()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["gate-failed"])

    def test_raw_receipt_has_no_generic_passed_envelope(self) -> None:
        self.fixture.receipt_document["passed"] = True
        self.fixture.receipt_raw = self.fixture.write_canonical(
            self.fixture.receipt_path, self.fixture.receipt_document
        )
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_raw_receipt_cannot_alias_freeze_declared_semantic_output(self) -> None:
        report = self.evaluate(self.fixture.semantic_report_relative)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["reserved-output-path-collision"])

    def test_raw_trace_descriptor_cannot_alias_freeze_declared_semantic_output(self) -> None:
        self.fixture.receipt_document["scenario_trace"] = {
            "path": self.fixture.semantic_report_relative,
            "sha256": digest("semantic output is not raw evidence"),
        }
        self.fixture.receipt_raw = self.fixture.write_canonical(
            self.fixture.receipt_path, self.fixture.receipt_document
        )
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["reserved-output-path-collision"])

    def test_receipt_descriptor_paths_must_be_distinct(self) -> None:
        gate = self.fixture.receipt_document["gate_e_soak"]
        assert isinstance(gate, dict)
        gate["raw_archive"] = copy.deepcopy(gate["report"])
        self.fixture.receipt_raw = self.fixture.write_canonical(
            self.fixture.receipt_path, self.fixture.receipt_document
        )
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["duplicate-evidence-path"])

    def test_hard_link_evidence_alias_is_rejected(self) -> None:
        alias_relative = "soak-v2/trace-hard-link.json"
        alias_path = self.fixture.evidence / alias_relative
        os.link(self.fixture.receipt_path, alias_path)
        self.fixture.receipt_document["scenario_trace"] = {
            "path": alias_relative,
            "sha256": digest("not reached because inode alias is rejected"),
        }
        self.fixture.receipt_raw = self.fixture.write_canonical(
            self.fixture.receipt_path, self.fixture.receipt_document
        )
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["hard-link-evidence-alias"])

    def test_gate_e_report_must_equal_raw_archive_replay(self) -> None:
        replay = self.fixture.replay_output()
        replayed_report = replay["report"]
        assert isinstance(replayed_report, dict)
        bindings = replayed_report["bindings"]
        assert isinstance(bindings, dict)
        bindings["manifest_sha256"] = digest("different replayed manifest")
        with (
            mock.patch.object(
                qualification,
                "revalidate_base_release_candidate",
                return_value=(self.fixture.base_raw, self.fixture.base_sha),
            ),
            mock.patch.object(
                qualification.release_candidate.reliability_soak,
                "replay_raw_evidence_archive",
                return_value=replay,
            ),
        ):
            report = self._evaluate_call()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["gate-e-soak-report-replay-mismatch"])

    def test_gate_e_raw_archive_hash_must_match_replay(self) -> None:
        replay = self.fixture.replay_output()
        replay["archive_sha256"] = digest("substituted archive")
        with (
            mock.patch.object(
                qualification,
                "revalidate_base_release_candidate",
                return_value=(self.fixture.base_raw, self.fixture.base_sha),
            ),
            mock.patch.object(
                qualification.release_candidate.reliability_soak,
                "replay_raw_evidence_archive",
                return_value=replay,
            ),
        ):
            report = self._evaluate_call()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["gate-e-soak-archive-hash-mismatch"])

    def test_cancellation_rate_10_must_match_actual_counts(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("cancellation-10")]
        assert isinstance(result, dict)
        result["cancelled_requests"] = 0
        result["completed_requests"] = result["attempted_requests"]
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["cancellation-rate-mismatch"])

    def test_cancellation_zero_arm_rejects_a_single_cancelled_request(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("cancellation-0")]
        assert isinstance(result, dict)
        result["cancelled_requests"] = 1
        result["completed_requests"] = int(result["completed_requests"]) - 1
        observations = result["interval_observations"]
        assert isinstance(observations, list)
        final = observations[-1]
        assert isinstance(final, dict)
        request_states = final["request_states"]
        assert isinstance(request_states, dict)
        request_states["completed"] = int(request_states["completed"]) - 1
        request_states["cancelled"] = 1
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["cancellation-rate-mismatch"])

    def test_cancellation_fifty_arm_rejects_an_inexact_rate(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("cancellation-50")]
        assert isinstance(result, dict)
        result["cancelled_requests"] = 49
        result["completed_requests"] = 51
        observations = result["interval_observations"]
        assert isinstance(observations, list)
        final = observations[-1]
        assert isinstance(final, dict)
        request_states = final["request_states"]
        assert isinstance(request_states, dict)
        request_states["completed"] = 51
        request_states["cancelled"] = 49
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["cancellation-rate-mismatch"])

    def test_kv_90_arm_must_reach_its_closed_range(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("kv-90")]
        assert isinstance(result, dict)
        result["max_kv_utilization_percent"] = 89
        observations = result["interval_observations"]
        assert isinstance(observations, list)
        for observation in observations:
            assert isinstance(observation, dict)
            kv_blocks = observation["kv_blocks"]
            assert isinstance(kv_blocks, dict)
            kv_blocks["free"] = 11
            kv_blocks["active"] = 89
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["kv-utilization-mismatch"])

    def test_kv_summary_must_equal_raw_block_inventory(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("kv-90")]
        assert isinstance(result, dict)
        result["max_kv_utilization_percent"] = 89
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["kv-utilization-evidence-mismatch"])

    def test_kv_capacity_arm_must_observe_capacity_rejection(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("kv-capacity")]
        assert isinstance(result, dict)
        rejected = result["capacity_rejections"]
        assert isinstance(rejected, int)
        result["capacity_rejections"] = 0
        result["completed_requests"] = int(result["completed_requests"]) + rejected
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["kv-capacity-boundary-missing"])

    def test_exact_backend_fallback_requires_ordered_raw_events(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("exact-backend-fallback")]
        assert isinstance(result, dict)
        result["backend_events"] = ["exact-backend-fallback-selected"]
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["exact-backend-fallback-missing"])

    def test_exact_backend_fallback_rejects_unreviewed_extra_fallback(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("exact-backend-fallback")]
        assert isinstance(result, dict)
        result["exact_backend_fallbacks"] = 2
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["exact-backend-fallback-missing"])

    def test_repeated_model_lifecycle_requires_every_cycle(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[soak.EXPECTED_SCENARIO_IDS.index("repeated-model-load-unload")]
        assert isinstance(result, dict)
        cycles = result["model_lifecycle_cycles"]
        assert isinstance(cycles, list)
        result["model_lifecycle_cycles"] = cycles[:-1]
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["model-lifecycle-coverage-missing"])

    def test_trace_binds_the_same_gate_e_raw_archive(self) -> None:
        self.fixture.trace_document["gate_e_soak_raw_sha256"] = digest("different raw archive")
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])

    def test_raw_receipt_must_bind_candidate_freeze_base_and_stable_arm(self) -> None:
        bindings = self.fixture.receipt_document["bindings"]
        assert isinstance(bindings, dict)
        variants = {
            "candidate": (self.fixture.receipt_document, "candidate_id", "riley-0.1.0-rc4"),
            "freeze": (bindings, "freeze_sha256", digest("another freeze")),
            "base": (bindings, "base_release_candidate_report_sha256", digest("another Gate E report")),
            "stable-arm": (bindings, "configuration_sha256", digest("another stable arm")),
        }
        for label, (target, field, replacement) in variants.items():
            with self.subTest(label=label):
                original = target[field]
                target[field] = replacement
                self.fixture.receipt_raw = self.fixture.write_canonical(
                    self.fixture.receipt_path, self.fixture.receipt_document
                )
                report = self.evaluate()
                self.assertEqual(report["status"], "incomparable")
                self.assertEqual(report["reason_codes"], ["incomparable-binding"])
                target[field] = original
        self.fixture.receipt_raw = self.fixture.write_canonical(
            self.fixture.receipt_path, self.fixture.receipt_document
        )

    def test_interval_telemetry_must_cover_the_full_scenario(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[0]
        assert isinstance(result, dict)
        observations = result["interval_observations"]
        assert isinstance(observations, list)
        result["interval_observations"] = observations[:-1]
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["interval-observation-coverage-missing"])

    def test_interval_final_request_totals_must_equal_scenario_outcome(self) -> None:
        rows = self.fixture.trace_document["scenario_results"]
        assert isinstance(rows, list)
        result = rows[0]
        assert isinstance(result, dict)
        observations = result["interval_observations"]
        assert isinstance(observations, list)
        final = observations[-1]
        assert isinstance(final, dict)
        request_states = final["request_states"]
        assert isinstance(request_states, dict)
        request_states["active"] = 1
        request_states["completed"] = int(request_states["completed"]) - 1
        self.fixture.rewrite_trace_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["request-state-accounting-mismatch"])

    def test_source_contract_current_hash_is_required(self) -> None:
        with mock.patch.object(soak, "CONTRACT_SHA256", digest("different source contract")):
            report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["source-contract-hash-mismatch"])

    def test_check_report_parser_rejects_raw_descriptor_alias(self) -> None:
        report = copy.deepcopy(self.evaluate())
        report["scenario_trace"] = copy.deepcopy(report["receipt"])
        with self.assertRaises(qualification.QualificationError) as raised:
            soak.validate_check_report(report)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "duplicate-evidence-path")

    def test_cli_report_is_create_only(self) -> None:
        output = self.fixture.root / "soak-v2-check.json"
        arguments = [
            "--freeze",
            str(self.fixture.freeze_path),
            "--expected-freeze-sha256",
            self.freeze_sha,
            "--evidence-root",
            str(self.fixture.evidence),
            "--receipt",
            self.fixture.receipt_relative,
            "--report",
            str(output),
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                qualification,
                "revalidate_base_release_candidate",
                return_value=(self.fixture.base_raw, self.fixture.base_sha),
            ),
            mock.patch.object(
                qualification.release_candidate.reliability_soak,
                "replay_raw_evidence_archive",
                return_value=self.fixture.replay_output(),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            self.assertEqual(soak.main(arguments), 0)
        original = output.read_bytes()
        with (
            mock.patch.object(
                qualification,
                "revalidate_base_release_candidate",
                return_value=(self.fixture.base_raw, self.fixture.base_sha),
            ),
            mock.patch.object(
                qualification.release_candidate.reliability_soak,
                "replay_raw_evidence_archive",
                return_value=self.fixture.replay_output(),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            self.assertEqual(soak.main(arguments), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_static_contract_covers_v1_and_new_c02_arms_at_full_duration(self) -> None:
        contract_path = Path(__file__).parents[2] / soak.CONTRACT_RELATIVE_PATH
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [scenario["id"] for scenario in contract["scenarios"]],
            list(soak.EXPECTED_SCENARIO_IDS),
        )
        self.assertEqual(contract["total_duration_seconds"], soak.EXPECTED_TOTAL_DURATION_SECONDS)
        self.assertEqual(contract["maximum_interval_seconds"], soak.EXPECTED_MAXIMUM_INTERVAL_SECONDS)
        self.assertEqual(hashlib.sha256(contract_path.read_bytes()).hexdigest(), soak.CONTRACT_SHA256)
        self.assertEqual(
            [
                scenario["cancellation_percent"]
                for scenario in contract["scenarios"]
                if scenario["id"] in {"cancellation-0", "cancellation-10", "cancellation-50"}
            ],
            [0, 10, 50],
        )


if __name__ == "__main__":
    unittest.main()
