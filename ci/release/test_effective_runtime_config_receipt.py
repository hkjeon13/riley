#!/usr/bin/env python3
"""CPU-only adversarial tests for the C02 dual-arm configuration receipt."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import check_effective_runtime_config_receipt as receipt
import check_rc3_qualification as qualification
import write_effective_runtime_config_startup_artifact as artifact_writer


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class ConfigReceiptFixture:
    """A complete local dual-arm evidence tree; Gate E replay is mocked."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.freeze_path = root / "riley-0.1.0-rc3.freeze.json"
        self.candidate_id = "riley-0.1.0-rc3"
        self.revision = "a" * 40
        self.base_relative = "reports/final.json"
        self.manifest_relative = "candidates/final-release-candidate.json"
        self.semantic_report_relative = "receipts/startup_configuration.json"
        self.endpoint_relative = {
            receipt.STABLE_DEFAULT_PROFILE: "startup/stable-v1-config.json",
            receipt.MAX_PERFORMANCE_EXACT_PROFILE: "startup/max-v1-config.json",
        }
        self.artifact_relative = {
            receipt.STABLE_DEFAULT_PROFILE: "startup/stable-startup-config.json",
            receipt.MAX_PERFORMANCE_EXACT_PROFILE: "startup/max-startup-config.json",
        }
        self.endpoint_path = {
            profile: self.evidence / relative
            for profile, relative in self.endpoint_relative.items()
        }
        self.artifact_path = {
            profile: self.evidence / relative
            for profile, relative in self.artifact_relative.items()
        }

        release = {
            "binary_sha256": digest("release binary"),
            "bundle_sha256": digest("release bundle"),
            "image_id": "sha256:" + digest("release image"),
            "cuda_c_abi_version": "12.8.1",
        }
        images = {
            "reproducible": "sha256:" + digest("reproducible image"),
            "cuda": "sha256:" + digest("cuda image"),
            "optimization": "sha256:" + digest("optimization image"),
        }
        stable_input = {
            "argv": [
                "serve",
                "--execution-completion",
                "iteration-batch",
                "--batch-token-budget",
                "64",
                "--metadata-transport",
                "packed-async",
            ],
            "environment": {"RILEY_CONFIG_RECEIPT": "1"},
        }
        maximum_input = {
            "argv": [
                "serve",
                "--execution-completion",
                "per-operation",
                "--batch-token-budget",
                "96",
                "--metadata-transport",
                "synchronous",
            ],
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
            "release": release,
            "images": images,
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
                            self.semantic_report_relative
                            if gate == "startup_configuration"
                            else f"receipts/{gate}.json"
                        )
                    }
                    for gate in qualification.REQUIRED_GATES
                },
            },
            "required_gates": list(qualification.REQUIRED_GATES),
        }

    def effective_config(self, profile: str) -> dict[str, object]:
        if profile == receipt.STABLE_DEFAULT_PROFILE:
            return {
                "execution_completion_mode": "iteration-batch",
                "batch_shape": {"policy": "power-of-two", "buckets": [1, 8, 64]},
                "metadata_transport": "packed-async",
                "sampling_backend": "gpu-greedy",
                "attention_backend": {
                    "prefill": "riley.attention.prefill-v1",
                    "decode": "riley.attention.decode-v1",
                },
                "gemm_reduction_policy": "strict-no-split-v1",
                "experimental_flags": {"residual-rmsnorm": "separate"},
                "fallback_policy": {
                    "cross_profile_fallback": "forbidden",
                    "runtime_selection": "exact-fallback-allowed",
                },
                "batch_token_budget": 64,
                "kv_geometry": {
                    "layout": "paged",
                    "block_tokens": 16,
                    "physical_blocks": 512,
                },
            }
        if profile == receipt.MAX_PERFORMANCE_EXACT_PROFILE:
            return {
                "execution_completion_mode": "per-operation",
                "batch_shape": {"policy": "fixed-max", "buckets": [96]},
                "metadata_transport": "synchronous",
                "sampling_backend": "gpu-greedy",
                "attention_backend": {
                    "prefill": "riley.attention.exact-prefill-v1",
                    "decode": "riley.attention.exact-decode-v1",
                },
                "gemm_reduction_policy": "exact-no-split-v1",
                "experimental_flags": {"exact-mode": "on", "residual-rmsnorm": "fused"},
                "fallback_policy": {
                    "cross_profile_fallback": "forbidden",
                    "runtime_selection": "fail-closed",
                },
                "batch_token_budget": 96,
                "kv_geometry": {
                    "layout": "contiguous",
                    "block_tokens": 32,
                    "physical_blocks": 128,
                },
            }
        raise AssertionError(f"unknown fixture profile: {profile}")

    def _frozen_arm(self, profile: str) -> dict[str, object]:
        arm = self.arms[receipt.PROFILE_TO_FREEZE_ARM[profile]]
        assert isinstance(arm, dict)
        return arm

    def endpoint_payload(self, profile: str) -> dict[str, object]:
        effective_config = self.effective_config(profile)
        return {
            "schema_version": receipt.ENDPOINT_VERSION,
            "candidate_id": self.candidate_id,
            "runtime_identity": {
                "configuration_profile": profile,
                "configuration_sha256": self._frozen_arm(profile)["configuration_sha256"],
            },
            "effective_config": effective_config,
            "effective_config_sha256": digest(qualification.canonical_json_bytes(effective_config)),
        }

    def write_canonical(self, path: Path, document: object) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = qualification.canonical_json_bytes(document)
        path.write_bytes(raw)
        return raw

    def materialize(self) -> str:
        self.freeze_sha = digest(self.write_canonical(self.freeze_path, self.freeze))
        # A real release binary writes its after-cold-prepare endpoint capture
        # and create-only startup artifact after candidate freeze, but before
        # the C02 Gate E report (a future output) exists.  Keep that order in
        # the fixture so the raw contract cannot regain that dependency.
        self.endpoint_documents: dict[str, dict[str, object]] = {}
        self.artifact_results: dict[str, receipt.StartupArtifactWriteResult] = {}
        for index, profile in enumerate(receipt.ARM_PROFILES, start=1):
            endpoint = self.endpoint_payload(profile)
            self.endpoint_documents[profile] = endpoint
            self.write_canonical(self.endpoint_path[profile], endpoint)
            self.artifact_results[profile] = artifact_writer.write_startup_artifact(
                self.endpoint_path[profile],
                self.artifact_path[profile],
                created_at_utc=f"2026-08-28T00:00:0{index}Z",
            )

        # The config tests mock the Gate E replay boundary (which has its own
        # exhaustive outer tests), but retain exactly the freeze-declared paths.
        self.base_raw = self.write_canonical(
            self.evidence / self.base_relative, {"replayed": "gate-e-fixture"}
        )
        self.base_sha = digest(self.base_raw)
        self.write_canonical(
            self.evidence / self.manifest_relative, {"fixture": "gate-e-manifest"}
        )
        return self.freeze_sha

    def rewrite_endpoint(self, profile: str, document: dict[str, object]) -> bytes:
        self.endpoint_documents[profile] = document
        return self.write_canonical(self.endpoint_path[profile], document)

    def rewrite_artifact(self, profile: str, document: dict[str, object]) -> bytes:
        return self.write_canonical(self.artifact_path[profile], document)

    def rewrite_coherent_endpoint_and_artifact(
        self, profile: str, endpoint: dict[str, object]
    ) -> None:
        """Rewrite temporary evidence to test a semantic conflict, not staleness."""

        endpoint_raw = self.rewrite_endpoint(profile, endpoint)
        artifact = json.loads(self.artifact_path[profile].read_text(encoding="utf-8"))
        artifact["runtime_identity"] = copy.deepcopy(endpoint["runtime_identity"])
        artifact["endpoint_payload_sha256"] = digest(endpoint_raw)
        artifact["endpoint_payload"] = copy.deepcopy(endpoint)
        self.rewrite_artifact(profile, artifact)


class EffectiveRuntimeConfigReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ConfigReceiptFixture(Path(self.temporary.name).resolve())
        self.freeze_sha = self.fixture.materialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _evaluate_call(
        self,
        *,
        stable_endpoint: str | None = None,
        stable_artifact: str | None = None,
        max_endpoint: str | None = None,
        max_artifact: str | None = None,
    ) -> dict[str, object]:
        return receipt.evaluate(
            self.fixture.freeze_path,
            self.fixture.evidence,
            stable_endpoint or self.fixture.endpoint_relative[receipt.STABLE_DEFAULT_PROFILE],
            stable_artifact or self.fixture.artifact_relative[receipt.STABLE_DEFAULT_PROFILE],
            max_endpoint or self.fixture.endpoint_relative[receipt.MAX_PERFORMANCE_EXACT_PROFILE],
            max_artifact or self.fixture.artifact_relative[receipt.MAX_PERFORMANCE_EXACT_PROFILE],
            expected_freeze_sha256=self.freeze_sha,
        )

    def evaluate(self, **overrides: str) -> dict[str, object]:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ):
            return self._evaluate_call(**overrides)

    def test_valid_receipt_replays_two_distinct_arms_and_parser_is_isomorphic(self) -> None:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ) as replay:
            report = self._evaluate_call()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["passed"])
        self.assertEqual(report["stable_promotion_profile"], receipt.STABLE_DEFAULT_PROFILE)
        self.assertEqual(tuple(report["arms"]), receipt.ARM_PROFILES)
        self.assertEqual(len(report["checks"]), len(receipt.CHECK_NAMES))
        parsed = receipt.validate_check_report(report)
        self.assertEqual(parsed.candidate_id, self.fixture.candidate_id)
        self.assertEqual(parsed.freeze_sha256, self.freeze_sha)
        self.assertEqual(parsed.base_release_candidate_report.sha256, self.fixture.base_sha)
        self.assertEqual(parsed.stable_promotion_profile, receipt.STABLE_DEFAULT_PROFILE)
        self.assertEqual(tuple(parsed.arms), receipt.ARM_PROFILES)
        for profile in receipt.ARM_PROFILES:
            arm = parsed.arms[profile]
            self.assertEqual(arm.configuration_profile, profile)
            self.assertEqual(arm.endpoint_payload.path, self.fixture.endpoint_relative[profile])
            self.assertEqual(arm.startup_artifact.path, self.fixture.artifact_relative[profile])
            endpoint = json.loads(self.fixture.endpoint_path[profile].read_text(encoding="utf-8"))
            self.assertEqual(set(endpoint["effective_config"]), set(receipt.CONFIG_DIMENSIONS))
        stable = parsed.arms[receipt.STABLE_DEFAULT_PROFILE]
        maximum = parsed.arms[receipt.MAX_PERFORMANCE_EXACT_PROFILE]
        self.assertNotEqual(stable.endpoint_payload.sha256, maximum.endpoint_payload.sha256)
        self.assertNotEqual(stable.startup_artifact.sha256, maximum.startup_artifact.sha256)
        self.assertNotEqual(stable.effective_config_sha256, maximum.effective_config_sha256)
        replay.assert_called_once()
        replay_frozen, replay_freeze_sha, replay_evidence_root = replay.call_args.args
        self.assertEqual(replay_frozen.candidate_id, self.fixture.candidate_id)
        self.assertEqual(replay_freeze_sha, self.freeze_sha)
        self.assertEqual(replay_evidence_root, self.fixture.evidence)

    def test_raw_endpoint_and_startup_artifact_are_feasible_after_freeze_before_gate_e(self) -> None:
        """The binary can publish raw startup facts before future Gate E hashes exist."""

        with tempfile.TemporaryDirectory() as temporary:
            before = ConfigReceiptFixture(Path(temporary).resolve())
            profile = receipt.STABLE_DEFAULT_PROFILE
            before.write_canonical(before.freeze_path, before.freeze)
            endpoint = before.endpoint_payload(profile)
            endpoint_path = before.endpoint_path[profile]
            artifact_path = before.artifact_path[profile]
            before.write_canonical(endpoint_path, endpoint)

            self.assertTrue(before.freeze_path.exists())
            self.assertFalse((before.evidence / before.base_relative).exists())
            result = artifact_writer.write_startup_artifact(
                endpoint_path,
                artifact_path,
                created_at_utc="2026-08-28T00:00:01Z",
            )

            self.assertEqual(result.configuration_profile, profile)
            raw_endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
            raw_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            for document in (raw_endpoint, raw_artifact):
                identity = document["runtime_identity"]
                self.assertEqual(
                    set(identity), {"configuration_profile", "configuration_sha256"}
                )
                self.assertNotIn("freeze_sha256", identity)
                self.assertNotIn("base_release_candidate_report_sha256", identity)
            self.assertEqual(raw_artifact["endpoint_payload"], raw_endpoint)

    def test_stable_only_legacy_report_fails_closed(self) -> None:
        report = copy.deepcopy(self.evaluate())
        arms = report["arms"]
        assert isinstance(arms, dict)
        arms.pop(receipt.MAX_PERFORMANCE_EXACT_PROFILE)
        with self.assertRaises(qualification.QualificationError) as raised:
            receipt.validate_check_report(report)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "unknown-or-missing-field")

    def test_report_parser_rejects_cross_arm_descriptor_alias(self) -> None:
        report = copy.deepcopy(self.evaluate())
        arms = report["arms"]
        assert isinstance(arms, dict)
        maximum = arms[receipt.MAX_PERFORMANCE_EXACT_PROFILE]
        stable = arms[receipt.STABLE_DEFAULT_PROFILE]
        assert isinstance(maximum, dict) and isinstance(stable, dict)
        maximum["endpoint_payload"] = copy.deepcopy(stable["endpoint_payload"])
        with self.assertRaises(qualification.QualificationError) as raised:
            receipt.validate_check_report(report)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "duplicate-config-evidence-path")

    def test_report_parser_rejects_same_effective_config_hash_for_both_arms(self) -> None:
        report = copy.deepcopy(self.evaluate())
        arms = report["arms"]
        assert isinstance(arms, dict)
        maximum = arms[receipt.MAX_PERFORMANCE_EXACT_PROFILE]
        stable = arms[receipt.STABLE_DEFAULT_PROFILE]
        assert isinstance(maximum, dict) and isinstance(stable, dict)
        maximum["effective_config_sha256"] = stable["effective_config_sha256"]
        with self.assertRaises(qualification.QualificationError) as raised:
            receipt.validate_check_report(report)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "indistinguishable-effective-config")

    def test_writer_is_create_only_and_canonical_for_each_arm(self) -> None:
        for index, profile in enumerate(receipt.ARM_PROFILES, start=3):
            output = self.fixture.root / f"{profile}-second.json"
            result = artifact_writer.write_startup_artifact(
                self.fixture.endpoint_path[profile],
                output,
                created_at_utc=f"2026-08-28T00:00:0{index}Z",
            )
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(output.read_bytes(), qualification.canonical_json_bytes(document))
            self.assertEqual(result.configuration_profile, profile)
            self.assertEqual(result.startup_artifact_sha256, digest(output.read_bytes()))
            self.assertEqual(
                set(document["runtime_identity"]),
                {"configuration_profile", "configuration_sha256"},
            )
            self.assertNotIn("freeze_sha256", document["runtime_identity"])
            self.assertNotIn(
                "base_release_candidate_report_sha256", document["runtime_identity"]
            )
            with self.assertRaises(qualification.QualificationError):
                artifact_writer.write_startup_artifact(
                    self.fixture.endpoint_path[profile],
                    output,
                    created_at_utc=f"2026-08-28T00:00:0{index}Z",
                )

    def test_max_endpoint_artifact_digest_mismatch_fails(self) -> None:
        profile = receipt.MAX_PERFORMANCE_EXACT_PROFILE
        artifact = json.loads(self.fixture.artifact_path[profile].read_text(encoding="utf-8"))
        artifact["endpoint_payload_sha256"] = digest("a different endpoint")
        self.fixture.rewrite_artifact(profile, artifact)
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["endpoint-artifact-digest-mismatch"])

    def test_max_artifact_runtime_identity_drift_is_incomparable(self) -> None:
        profile = receipt.MAX_PERFORMANCE_EXACT_PROFILE
        artifact = json.loads(self.fixture.artifact_path[profile].read_text(encoding="utf-8"))
        artifact["runtime_identity"]["configuration_sha256"] = digest("different arm")
        self.fixture.rewrite_artifact(profile, artifact)
        report = self.evaluate()
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])

    def test_legacy_freeze_and_gate_e_fields_are_rejected_from_raw_endpoint(self) -> None:
        profile = receipt.STABLE_DEFAULT_PROFILE
        endpoint = copy.deepcopy(self.fixture.endpoint_documents[profile])
        identity = endpoint["runtime_identity"]
        assert isinstance(identity, dict)
        identity["freeze_sha256"] = self.freeze_sha
        identity["base_release_candidate_report_sha256"] = self.fixture.base_sha
        self.fixture.rewrite_endpoint(profile, endpoint)

        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_legacy_bindings_envelope_is_rejected_from_raw_endpoint(self) -> None:
        """v1 drafts placed future decision hashes in a raw `bindings` object."""

        profile = receipt.STABLE_DEFAULT_PROFILE
        endpoint = copy.deepcopy(self.fixture.endpoint_documents[profile])
        identity = endpoint.pop("runtime_identity")
        assert isinstance(identity, dict)
        endpoint["bindings"] = {
            "freeze_sha256": self.freeze_sha,
            "base_release_candidate_report_sha256": self.fixture.base_sha,
            **identity,
        }
        with self.assertRaises(qualification.QualificationError) as raised:
            receipt.validate_endpoint_payload(endpoint)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "unknown-or-missing-field")

    def test_legacy_freeze_and_gate_e_fields_are_rejected_from_raw_startup_artifact(self) -> None:
        profile = receipt.STABLE_DEFAULT_PROFILE
        artifact = json.loads(self.fixture.artifact_path[profile].read_text(encoding="utf-8"))
        identity = artifact["runtime_identity"]
        assert isinstance(identity, dict)
        identity["freeze_sha256"] = self.freeze_sha
        identity["base_release_candidate_report_sha256"] = self.fixture.base_sha
        self.fixture.rewrite_artifact(profile, artifact)

        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_legacy_bindings_envelope_is_rejected_from_raw_startup_artifact(self) -> None:
        profile = receipt.STABLE_DEFAULT_PROFILE
        artifact = json.loads(self.fixture.artifact_path[profile].read_text(encoding="utf-8"))
        identity = artifact.pop("runtime_identity")
        assert isinstance(identity, dict)
        artifact["bindings"] = {
            "freeze_sha256": self.freeze_sha,
            "base_release_candidate_report_sha256": self.fixture.base_sha,
            **identity,
        }
        with self.assertRaises(qualification.QualificationError) as raised:
            receipt.validate_startup_artifact(artifact)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "unknown-or-missing-field")

    def test_failed_gate_e_replay_blocks_both_arm_receipt(self) -> None:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            side_effect=qualification.GateFailure("replayed Gate E failed"),
        ) as replay:
            report = self._evaluate_call()
        replay.assert_called_once()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["gate-failed"])

    def test_replayed_base_digest_is_a_post_capture_semantic_binding(self) -> None:
        replayed_raw = b'{"replayed":"different"}'
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(replayed_raw, digest(replayed_raw)),
        ):
            report = self._evaluate_call()
        self.assertEqual(report["status"], "passed")
        base = report["base_release_candidate_report"]
        assert isinstance(base, dict)
        self.assertEqual(base["sha256"], digest(replayed_raw))
        for profile in receipt.ARM_PROFILES:
            endpoint = json.loads(self.fixture.endpoint_path[profile].read_text(encoding="utf-8"))
            self.assertNotIn("freeze_sha256", endpoint["runtime_identity"])
            self.assertNotIn(
                "base_release_candidate_report_sha256", endpoint["runtime_identity"]
            )

    def test_max_profile_must_bind_its_own_frozen_arm(self) -> None:
        profile = receipt.MAX_PERFORMANCE_EXACT_PROFILE
        endpoint = copy.deepcopy(self.fixture.endpoint_documents[profile])
        runtime_identity = endpoint["runtime_identity"]
        assert isinstance(runtime_identity, dict)
        runtime_identity["configuration_profile"] = receipt.STABLE_DEFAULT_PROFILE
        self.fixture.rewrite_endpoint(profile, endpoint)
        report = self.evaluate()
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])

    def test_max_profile_configuration_hash_must_bind_its_own_frozen_arm(self) -> None:
        profile = receipt.MAX_PERFORMANCE_EXACT_PROFILE
        endpoint = copy.deepcopy(self.fixture.endpoint_documents[profile])
        runtime_identity = endpoint["runtime_identity"]
        assert isinstance(runtime_identity, dict)
        runtime_identity["configuration_sha256"] = digest("some other frozen command")
        self.fixture.rewrite_endpoint(profile, endpoint)
        report = self.evaluate()
        self.assertEqual(report["status"], "incomparable")
        self.assertEqual(report["reason_codes"], ["incomparable-binding"])

    def test_effective_config_must_be_distinct_across_frozen_arms(self) -> None:
        profile = receipt.MAX_PERFORMANCE_EXACT_PROFILE
        endpoint = copy.deepcopy(self.fixture.endpoint_documents[profile])
        stable_config = self.fixture.effective_config(receipt.STABLE_DEFAULT_PROFILE)
        endpoint["effective_config"] = stable_config
        endpoint["effective_config_sha256"] = digest(qualification.canonical_json_bytes(stable_config))
        self.fixture.rewrite_coherent_endpoint_and_artifact(profile, endpoint)
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["indistinguishable-effective-config"])

    def test_unknown_effective_dimension_fails_closed(self) -> None:
        profile = receipt.MAX_PERFORMANCE_EXACT_PROFILE
        endpoint = copy.deepcopy(self.fixture.endpoint_documents[profile])
        effective = endpoint["effective_config"]
        assert isinstance(effective, dict)
        effective["undisclosed_mode"] = "on"
        endpoint["effective_config_sha256"] = digest(qualification.canonical_json_bytes(effective))
        self.fixture.rewrite_endpoint(profile, endpoint)
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_batch_shape_must_end_at_effective_budget(self) -> None:
        profile = receipt.STABLE_DEFAULT_PROFILE
        endpoint = copy.deepcopy(self.fixture.endpoint_documents[profile])
        effective = endpoint["effective_config"]
        assert isinstance(effective, dict)
        shape = effective["batch_shape"]
        assert isinstance(shape, dict)
        shape["buckets"] = [1, 8, 32]
        endpoint["effective_config_sha256"] = digest(qualification.canonical_json_bytes(effective))
        self.fixture.rewrite_endpoint(profile, endpoint)
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["invalid-effective-config"])

    def test_packed_async_requires_iteration_batch_completion(self) -> None:
        profile = receipt.STABLE_DEFAULT_PROFILE
        endpoint = copy.deepcopy(self.fixture.endpoint_documents[profile])
        effective = endpoint["effective_config"]
        assert isinstance(effective, dict)
        effective["execution_completion_mode"] = "per-operation"
        endpoint["effective_config_sha256"] = digest(qualification.canonical_json_bytes(effective))
        self.fixture.rewrite_endpoint(profile, endpoint)
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["invalid-effective-config"])

    def test_noncanonical_max_endpoint_bytes_fail_before_digest_comparison(self) -> None:
        profile = receipt.MAX_PERFORMANCE_EXACT_PROFILE
        endpoint = self.fixture.endpoint_documents[profile]
        self.fixture.endpoint_path[profile].write_text(json.dumps(endpoint, indent=2), encoding="utf-8")
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["noncanonical-endpoint-payload"])

    def test_duplicate_endpoint_key_is_rejected(self) -> None:
        profile = receipt.MAX_PERFORMANCE_EXACT_PROFILE
        self.fixture.endpoint_path[profile].write_text(
            '{"schema_version":"x","schema_version":"x"}', encoding="utf-8"
        )
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["duplicate-json-key"])

    def test_raw_arm_paths_must_all_be_distinct(self) -> None:
        report = self.evaluate(
            stable_artifact=self.fixture.endpoint_relative[receipt.STABLE_DEFAULT_PROFILE]
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["duplicate-config-evidence-path"])

    def test_raw_arm_paths_cannot_collide_with_frozen_outputs(self) -> None:
        report = self.evaluate(max_artifact=self.fixture.semantic_report_relative)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["reserved-output-path-collision"])

    def test_endpoint_path_must_stay_below_evidence_root(self) -> None:
        report = self.evaluate(stable_endpoint="../v1-config.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["reason_codes"], ["invalid-relative-path"])

    def test_cli_report_is_create_only_with_all_four_inputs(self) -> None:
        output = self.fixture.root / "config-check.json"
        arguments = [
            "--freeze",
            str(self.fixture.freeze_path),
            "--expected-freeze-sha256",
            self.freeze_sha,
            "--evidence-root",
            str(self.fixture.evidence),
            "--stable-endpoint-payload",
            self.fixture.endpoint_relative[receipt.STABLE_DEFAULT_PROFILE],
            "--stable-startup-artifact",
            self.fixture.artifact_relative[receipt.STABLE_DEFAULT_PROFILE],
            "--max-endpoint-payload",
            self.fixture.endpoint_relative[receipt.MAX_PERFORMANCE_EXACT_PROFILE],
            "--max-startup-artifact",
            self.fixture.artifact_relative[receipt.MAX_PERFORMANCE_EXACT_PROFILE],
            "--report",
            str(output),
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(receipt.main(arguments), 0)
        original = output.read_bytes()
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(self.fixture.base_raw, self.fixture.base_sha),
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(receipt.main(arguments), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_schema_declares_exact_ten_dimensions_and_closed_dual_arm_inventory(self) -> None:
        schema_path = (
            Path(__file__).parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "effective-runtime-config-receipt-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        declared = tuple(schema["$defs"]["effectiveConfig"]["required"])
        self.assertEqual(declared, receipt.CONFIG_DIMENSIONS)
        self.assertFalse(schema["$defs"]["effectiveConfig"]["additionalProperties"])
        self.assertEqual(
            tuple(schema["$defs"]["configurationArms"]["required"]),
            receipt.ARM_PROFILES,
        )
        identity = schema["$defs"]["runtimeConfigIdentity"]
        self.assertEqual(
            tuple(identity["properties"]["configuration_profile"]["enum"]), receipt.ARM_PROFILES
        )
        self.assertEqual(
            set(identity["required"]), {"configuration_profile", "configuration_sha256"}
        )
        self.assertNotIn("freeze_sha256", identity["properties"])
        self.assertNotIn("base_release_candidate_report_sha256", identity["properties"])
        self.assertIn("freeze_sha256", schema["$defs"]["checkReport"]["required"])
        self.assertIn("base_release_candidate_report", schema["$defs"]["checkReport"]["required"])


if __name__ == "__main__":
    unittest.main()
