#!/usr/bin/env python3
"""Focused P0 tests for the pre-freeze runtime-configuration contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import effective_runtime_config_contract as contract
import validate_raw_c02_runtime_config as raw_validator
import write_effective_runtime_config_startup_artifact as artifact_writer


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ContractFixture:
    candidate_id = "riley-0.1.0-rc99"
    profile = contract.STABLE_DEFAULT_PROFILE

    def __init__(self, root: Path) -> None:
        self.root = root
        self.endpoint_path = root / "endpoint.json"
        self.artifact_path = root / "startup.json"
        self.effective_config: dict[str, object] = {
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
            "kv_geometry": {"layout": "paged", "block_tokens": 16, "physical_blocks": 512},
        }
        self.endpoint: dict[str, object] = {
            "schema_version": contract.ENDPOINT_VERSION,
            "candidate_id": self.candidate_id,
            "runtime_identity": {
                "configuration_profile": self.profile,
                "configuration_sha256": "a" * 64,
            },
            "effective_config": self.effective_config,
            "effective_config_sha256": digest(contract.canonical_json_bytes(self.effective_config)),
        }
        self.write_endpoint()

    def write_endpoint(self) -> bytes:
        encoded = contract.canonical_json_bytes(self.endpoint)
        self.endpoint_path.write_bytes(encoded)
        return encoded


class EffectiveRuntimeConfigContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ContractFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_endpoint_and_writer_round_trip_without_freeze_or_gate_e(self) -> None:
        endpoint_raw = self.fixture.endpoint_path.read_bytes()
        endpoint_document, endpoint = contract.validate_endpoint_bytes(endpoint_raw)
        result = artifact_writer.write_startup_artifact(
            self.fixture.endpoint_path,
            self.fixture.artifact_path,
            created_at_utc="2026-08-28T06:25:48Z",
        )
        artifact_raw = self.fixture.artifact_path.read_bytes()
        artifact_document, artifact = contract.validate_startup_artifact_bytes(artifact_raw)

        self.assertEqual(endpoint_document, self.fixture.endpoint)
        self.assertEqual(artifact.candidate_id, endpoint.candidate_id)
        self.assertEqual(artifact.endpoint_payload, endpoint)
        self.assertEqual(artifact.endpoint_payload_sha256, digest(endpoint_raw))
        self.assertEqual(artifact_document["endpoint_payload"], endpoint_document)
        self.assertEqual(artifact_raw, contract.canonical_json_bytes(artifact_document))
        self.assertEqual(result.path, self.fixture.artifact_path)
        self.assertEqual(result.startup_artifact_sha256, digest(artifact_raw))

    def test_public_effective_config_validator_preserves_endpoint_normalization(self) -> None:
        endpoint_raw = self.fixture.endpoint_path.read_bytes()
        _document, endpoint = contract.validate_endpoint_bytes(endpoint_raw)
        self.assertEqual(
            contract.validate_effective_config(self.fixture.effective_config),
            endpoint.effective_config,
        )

    def test_raw_endpoint_rejects_future_freeze_binding(self) -> None:
        endpoint = copy.deepcopy(self.fixture.endpoint)
        identity = endpoint["runtime_identity"]
        assert isinstance(identity, dict)
        identity["freeze_sha256"] = "b" * 64
        self.fixture.endpoint = endpoint
        self.fixture.write_endpoint()

        with self.assertRaises(contract.EffectiveRuntimeConfigError) as raised:
            contract.validate_endpoint_bytes(self.fixture.endpoint_path.read_bytes())
        self.assertEqual(raised.exception.reason_code, "unknown-or-missing-field")

    def test_host_raw_validator_and_p0_contract_accept_the_same_canonical_fact(self) -> None:
        endpoint_raw = self.fixture.endpoint_path.read_bytes()
        artifact_path = self.fixture.root / "raw-artifact.json"
        artifact_writer.write_startup_artifact(
            self.fixture.endpoint_path,
            artifact_path,
            created_at_utc="2026-08-28T06:25:48Z",
        )

        _document, endpoint = contract.validate_endpoint_bytes(endpoint_raw)
        _document, artifact = contract.validate_startup_artifact_bytes(artifact_path.read_bytes())
        self.assertEqual(artifact.endpoint_payload, endpoint)
        self.assertEqual(
            raw_validator.validate_raw_capture(
                profile=self.fixture.profile,
                candidate_id=self.fixture.candidate_id,
                endpoint_path=self.fixture.endpoint_path,
                artifact_path=artifact_path,
                server_artifact_path=artifact_path,
            ),
            endpoint.effective_config_sha256,
        )

    def test_writer_is_create_only_and_refuses_a_noncanonical_source(self) -> None:
        artifact_writer.write_startup_artifact(
            self.fixture.endpoint_path,
            self.fixture.artifact_path,
            created_at_utc="2026-08-28T06:25:48Z",
        )
        with self.assertRaises(contract.EffectiveRuntimeConfigError) as raised:
            artifact_writer.write_startup_artifact(
                self.fixture.endpoint_path,
                self.fixture.artifact_path,
                created_at_utc="2026-08-28T06:25:49Z",
            )
        self.assertEqual(raised.exception.reason_code, "create-only-write-failed")

        self.fixture.endpoint_path.write_bytes(self.fixture.endpoint_path.read_bytes() + b"\n")
        with self.assertRaises(contract.EffectiveRuntimeConfigError) as raised:
            artifact_writer.write_startup_artifact(
                self.fixture.endpoint_path,
                self.fixture.root / "noncanonical.json",
                created_at_utc="2026-08-28T06:25:50Z",
            )
        self.assertEqual(raised.exception.reason_code, "noncanonical-endpoint-payload")

    def test_writer_rejects_existing_symlink_without_replacement(self) -> None:
        output = self.fixture.root / "artifact-link.json"
        target = self.fixture.root / "target.json"
        target.write_bytes(b"untouched")
        try:
            os.symlink(target, output)
        except OSError as error:  # pragma: no cover - platforms without symlink support
            self.skipTest(f"symlink unavailable: {error}")

        with self.assertRaises(contract.EffectiveRuntimeConfigError) as raised:
            artifact_writer.write_startup_artifact(
                self.fixture.endpoint_path,
                output,
                created_at_utc="2026-08-28T06:25:48Z",
            )
        self.assertEqual(raised.exception.reason_code, "create-only-write-failed")
        self.assertTrue(output.is_symlink())
        self.assertEqual(target.read_bytes(), b"untouched")

    def test_writer_fails_closed_when_nofollow_is_unavailable(self) -> None:
        with mock.patch.object(contract.os, "O_NOFOLLOW", None):
            with self.assertRaises(contract.EffectiveRuntimeConfigError) as raised:
                artifact_writer.write_startup_artifact(
                    self.fixture.endpoint_path,
                    self.fixture.artifact_path,
                    created_at_utc="2026-08-28T06:25:48Z",
                )
        self.assertEqual(raised.exception.reason_code, "unsafe-platform")
        self.assertFalse(self.fixture.artifact_path.exists())

    def test_writer_fails_closed_when_directory_open_is_unavailable(self) -> None:
        with mock.patch.object(contract.os, "O_DIRECTORY", None):
            with self.assertRaises(contract.EffectiveRuntimeConfigError) as raised:
                artifact_writer.write_startup_artifact(
                    self.fixture.endpoint_path,
                    self.fixture.artifact_path,
                    created_at_utc="2026-08-28T06:25:48Z",
                )
        self.assertEqual(raised.exception.reason_code, "unsafe-platform")
        self.assertFalse(self.fixture.artifact_path.exists())

    def test_p0_contract_and_writer_do_not_import_c02_finalizer_or_release_common(self) -> None:
        release_dir = Path(__file__).resolve().parent
        for name in (
            "effective_runtime_config_contract.py",
            "write_effective_runtime_config_startup_artifact.py",
        ):
            source = (release_dir / name).read_text(encoding="utf-8")
            self.assertNotIn("check_rc3_qualification", source)
            self.assertNotIn("release_common", source)
            self.assertNotIn("tomllib", source)

        raw_source = (release_dir / "validate_raw_c02_runtime_config.py").read_text(encoding="utf-8")
        self.assertNotIn("import effective_runtime_config_contract", raw_source)


if __name__ == "__main__":
    unittest.main()
