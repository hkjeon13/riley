#!/usr/bin/env python3
"""CPU-only hostile tests for RC3 static-to-effective config replay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import check_rc3_rollback_provenance_v3 as rollback
import check_rc3_static_effective_config_v1 as static_config
import prepare_rc3_rollback_evidence_v1 as preparation
import provenance_v2_common as common
import test_c02_provenance_v2 as c02_fixtures
from test_check_reconstructed_prior_baseline_v2 import BaselineV2Fixture


class StaticEffectiveConfigReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.c02 = c02_fixtures.C02ProvenanceV2Tests()
        self.environment = self._new_environment("evidence")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _effective_config(self) -> dict[str, object]:
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
            "experimental_flags": {},
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

    def _static_document(
        self,
        *,
        candidate_id: str | None = None,
        configuration_profile: str = "stable-default",
        effective_config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        expected = self._effective_config() if effective_config is None else effective_config
        return {
            "schema_version": static_config.STATIC_EFFECTIVE_CONFIG_VERSION,
            "candidate_id": self.c02.candidate if candidate_id is None else candidate_id,
            "configuration_profile": configuration_profile,
            "expected_effective_config": expected,
            "expected_effective_config_sha256": hashlib.sha256(
                common.canonical_json_bytes(expected)
            ).hexdigest(),
        }

    def _new_environment(
        self,
        name: str,
        *,
        static_document: dict[str, object] | None = None,
        bridge_bindings: dict[str, str] | None = None,
    ) -> dict[str, object]:
        root = self.base / name
        root.mkdir(mode=0o700)
        os.chmod(root, 0o700)
        BaselineV2Fixture(
            root,
            target_commit_sha1=rollback.RECONSTRUCTED_ROLLBACK_TARGET,
            tag_object_sha1=rollback.RECONSTRUCTED_ROLLBACK_TAG_OBJECT,
        )
        inputs = self.base / f"{name}-inputs"
        inputs.mkdir(mode=0o700)
        os.chmod(inputs, 0o700)
        freeze = inputs / "freeze.raw"
        base_report = inputs / "base-report.raw"
        configuration = inputs / "stable-default-config.json"
        freeze.write_bytes(b'{"freeze":"future"}\n')
        base_report.write_bytes(b'{"base_report":"future"}\n')
        configuration.write_bytes(
            common.canonical_json_bytes(
                self._static_document() if static_document is None else static_document
            )
        )
        for source in (freeze, base_report, configuration):
            os.chmod(source, 0o644)
        preparation.prepare_rollback_evidence(
            preparation.EvidencePreparationRequest(
                evidence_root=root,
                baseline_manifest_path="baseline.json",
                candidate_id=self.c02.candidate,
                freeze_input=freeze,
                base_release_candidate_report_input=base_report,
                stable_default_configuration_input=configuration,
            )
        )
        tree = c02_fixtures.EvidenceTree(root)
        bridge = self.c02.configuration_evidence(
            tree,
            self.c02.bindings if bridge_bindings is None else bridge_bindings,
            bridge_prefix="config-bridge",
        )
        return {"root": root, "bridge": bridge}

    def _replay(
        self,
        environment: dict[str, object] | None = None,
        *,
        used_paths: set[str] | None = None,
    ) -> static_config.ReplayedStaticEffectiveConfig:
        environment = self.environment if environment is None else environment
        root = environment["root"]
        bridge = environment["bridge"]
        assert isinstance(root, Path)
        assert isinstance(bridge, dict)
        root_fd = common.open_private_evidence_directory(root, "test evidence root")
        try:
            return static_config.replay_static_effective_config_v1_fd(
                root_fd,
                endpoint_path=bridge["endpoint"]["path"],
                startup_artifact_path=bridge["startup_artifact"]["path"],
                session_path=bridge["endpoint_observation"]["path"],
                used_paths=used_paths,
            )
        finally:
            os.close(root_fd)

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(static_config.StaticEffectiveConfigError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_replays_static_intent_against_a_distinct_runtime_identity_hash(self) -> None:
        replayed = self._replay()
        bridge = self.environment["bridge"]
        assert isinstance(bridge, dict)
        self.assertEqual(replayed.candidate_id, self.c02.candidate)
        self.assertEqual(replayed.configuration_profile, "stable-default")
        self.assertEqual(replayed.expected_effective_config, self._effective_config())
        self.assertEqual(
            replayed.expected_effective_config_sha256,
            hashlib.sha256(common.canonical_json_bytes(self._effective_config())).hexdigest(),
        )
        self.assertEqual(replayed.runtime_configuration_sha256, self.c02.bindings["configuration_sha256"])
        self.assertNotEqual(
            replayed.runtime_configuration_sha256,
            replayed.static_bindings.configuration.sha256,
        )
        self.assertEqual(
            replayed.config_bridge.endpoint,
            common.parse_descriptor(bridge["endpoint"], "fixture config endpoint"),
        )
        self.assertEqual(replayed.config_bridge.target.listener_port, 8080)
        self.assertEqual(replayed.config_bridge.target.target.gpu_index, 0)

    def test_rejects_static_schema_identity_hash_and_effective_config_drift(self) -> None:
        unknown = self._static_document()
        unknown["unexpected"] = True
        self.assert_reason(
            "unexpected-field-set",
            lambda: self._replay(self._new_environment("unknown", static_document=unknown)),
        )

        candidate = self._static_document(candidate_id="riley-0.1.0-rc4")
        self.assert_reason(
            "static-config-candidate-mismatch",
            lambda: self._replay(self._new_environment("candidate", static_document=candidate)),
        )

        bad_hash = self._static_document()
        bad_hash["expected_effective_config_sha256"] = "f" * 64
        self.assert_reason(
            "expected-effective-config-hash-mismatch",
            lambda: self._replay(self._new_environment("hash", static_document=bad_hash)),
        )

        different = self._effective_config()
        different["sampling_backend"] = "cpu"
        self.assert_reason(
            "effective-config-mismatch",
            lambda: self._replay(
                self._new_environment(
                    "effective-drift",
                    static_document=self._static_document(effective_config=different),
                )
            ),
        )

    def test_rejects_bridge_profile_drift_and_reserved_path_reuse(self) -> None:
        self.assert_reason(
            "runtime-config-profile-mismatch",
            lambda: self._replay(
                self._new_environment(
                    "profile-drift",
                    bridge_bindings=self.c02.max_performance_bindings,
                )
            ),
        )
        self.assert_reason(
            "duplicate-evidence-path",
            lambda: self._replay(
                used_paths={
                    "rollback-v3-evidence-inputs/stable-default-configuration.raw"
                }
            ),
        )

    def test_rejects_mutated_static_snapshot_and_is_nonoperational(self) -> None:
        root = self.environment["root"]
        assert isinstance(root, Path)
        snapshot = (
            root
            / preparation.INPUTS_DIRECTORY_NAME
            / preparation.CONFIGURATION_NAME
        )
        snapshot.write_bytes(b"{}")
        with self.assertRaises(static_config.StaticEffectiveConfigError):
            self._replay()

        source = Path(static_config.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import argparse",
            "def main(",
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
        ):
            self.assertNotIn(forbidden, source)

    def test_schema_document_is_closed_and_matches_the_static_contract(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-static-effective-config-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(schema["required"]),
            {
                "schema_version",
                "candidate_id",
                "configuration_profile",
                "expected_effective_config",
                "expected_effective_config_sha256",
            },
        )
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            static_config.STATIC_EFFECTIVE_CONFIG_VERSION,
        )
        self.assertEqual(
            schema["properties"]["configuration_profile"]["const"],
            "stable-default",
        )


if __name__ == "__main__":
    unittest.main()
