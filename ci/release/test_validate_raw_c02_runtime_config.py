#!/usr/bin/env python3
"""Python-3.10-only tests for the raw C02 remote-capture validator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
VALIDATOR = HERE / "validate_raw_c02_runtime_config.py"
SPEC = importlib.util.spec_from_file_location("raw_c02_runtime_config_test", VALIDATOR)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load raw validator: {VALIDATOR}")
raw = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = raw
SPEC.loader.exec_module(raw)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class RawCaptureFixture:
    candidate_id = "riley-0.1.0-rc99"
    profile = raw.STABLE_DEFAULT_PROFILE

    def __init__(self, root: Path) -> None:
        self.root = root
        self.endpoint_path = root / "endpoint-payload.json"
        self.artifact_path = root / "startup-artifact.json"
        self.server_artifact_path = root / "server-startup-artifact.json"
        self.endpoint = self._endpoint()
        self.artifact = self._artifact(self.endpoint)
        self.write_canonical()

    @staticmethod
    def effective_config() -> dict[str, object]:
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

    def _endpoint(self) -> dict[str, object]:
        effective_config = self.effective_config()
        return {
            "schema_version": raw.ENDPOINT_VERSION,
            "candidate_id": self.candidate_id,
            "runtime_identity": {
                "configuration_profile": self.profile,
                "configuration_sha256": "a" * 64,
            },
            "effective_config": effective_config,
            "effective_config_sha256": sha256(raw.canonical_json_bytes(effective_config)),
        }

    def _artifact(self, endpoint: dict[str, object]) -> dict[str, object]:
        endpoint_raw = raw.canonical_json_bytes(endpoint)
        return {
            "schema_version": raw.STARTUP_ARTIFACT_VERSION,
            "created_at_utc": "2026-08-28T06:25:48Z",
            "candidate_id": self.candidate_id,
            "endpoint_path": "/v1/config",
            "runtime_identity": copy.deepcopy(endpoint["runtime_identity"]),
            "endpoint_payload_sha256": sha256(endpoint_raw),
            "endpoint_payload": copy.deepcopy(endpoint),
        }

    def write_canonical(self) -> None:
        endpoint_raw = raw.canonical_json_bytes(self.endpoint)
        artifact_raw = raw.canonical_json_bytes(self.artifact)
        self.endpoint_path.write_bytes(endpoint_raw)
        self.artifact_path.write_bytes(artifact_raw)
        self.server_artifact_path.write_bytes(artifact_raw)

    def validate(self) -> str:
        return raw.validate_raw_capture(
            profile=self.profile,
            candidate_id=self.candidate_id,
            endpoint_path=self.endpoint_path,
            artifact_path=self.artifact_path,
            server_artifact_path=self.server_artifact_path,
        )


class RawC02RuntimeConfigValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = RawCaptureFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _assert_fails(self, reason_code: str) -> None:
        with self.assertRaises(raw.RawC02ValidationError) as raised:
            self.fixture.validate()
        self.assertEqual(raised.exception.reason_code, reason_code)

    def test_valid_capture_is_accepted_and_cli_uses_only_the_stdlib(self) -> None:
        expected = self.fixture.endpoint["effective_config_sha256"]
        self.assertEqual(self.fixture.validate(), expected)
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-I",
                "-S",
                str(VALIDATOR),
                "--profile",
                self.fixture.profile,
                "--candidate-id",
                self.fixture.candidate_id,
                "--endpoint",
                str(self.fixture.endpoint_path),
                "--startup-artifact",
                str(self.fixture.artifact_path),
                "--server-startup-artifact",
                str(self.fixture.server_artifact_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, f"{expected}\n")

    def test_noncanonical_endpoint_bytes_fail_closed(self) -> None:
        self.fixture.endpoint_path.write_bytes(raw.canonical_json_bytes(self.fixture.endpoint) + b"\n")
        self._assert_fails("noncanonical-endpoint-payload")

    def test_noncanonical_startup_artifact_bytes_fail_closed(self) -> None:
        noncanonical = raw.canonical_json_bytes(self.fixture.artifact) + b"\n"
        self.fixture.artifact_path.write_bytes(noncanonical)
        self.fixture.server_artifact_path.write_bytes(noncanonical)
        self._assert_fails("noncanonical-startup-artifact")

    def test_duplicate_json_key_fails_before_schema_parsing(self) -> None:
        self.fixture.endpoint_path.write_bytes(
            b'{"candidate_id":"riley-0.1.0-rc99","candidate_id":"riley-0.1.0-rc99"}'
        )
        self._assert_fails("duplicate-json-key")

    def test_unknown_effective_config_dimension_fails_closed(self) -> None:
        endpoint = copy.deepcopy(self.fixture.endpoint)
        effective_config = endpoint["effective_config"]
        assert isinstance(effective_config, dict)
        effective_config["undisclosed_mode"] = "on"
        endpoint["effective_config_sha256"] = sha256(raw.canonical_json_bytes(effective_config))
        self.fixture.endpoint = endpoint
        self.fixture.artifact = self.fixture._artifact(endpoint)
        self.fixture.write_canonical()
        self._assert_fails("unknown-or-missing-field")

    def test_packed_async_requires_iteration_batch_completion(self) -> None:
        endpoint = copy.deepcopy(self.fixture.endpoint)
        effective_config = endpoint["effective_config"]
        assert isinstance(effective_config, dict)
        effective_config["execution_completion_mode"] = "per-operation"
        endpoint["effective_config_sha256"] = sha256(raw.canonical_json_bytes(effective_config))
        self.fixture.endpoint = endpoint
        self.fixture.artifact = self.fixture._artifact(endpoint)
        self.fixture.write_canonical()
        self._assert_fails("invalid-effective-config")

    def test_endpoint_candidate_must_bind_runner_candidate(self) -> None:
        with self.assertRaises(raw.RawC02ValidationError) as raised:
            raw.validate_raw_capture(
                profile=self.fixture.profile,
                candidate_id="riley-0.1.0-rc100",
                endpoint_path=self.fixture.endpoint_path,
                artifact_path=self.fixture.artifact_path,
                server_artifact_path=self.fixture.server_artifact_path,
            )
        self.assertEqual(raised.exception.reason_code, "incomparable-binding")

    def test_artifact_identity_must_bind_embedded_endpoint(self) -> None:
        artifact = copy.deepcopy(self.fixture.artifact)
        identity = artifact["runtime_identity"]
        assert isinstance(identity, dict)
        identity["configuration_profile"] = raw.MAX_PERFORMANCE_EXACT_PROFILE
        self.fixture.artifact = artifact
        self.fixture.write_canonical()
        self._assert_fails("incomparable-binding")

    def test_startup_artifact_schema_is_closed(self) -> None:
        artifact = copy.deepcopy(self.fixture.artifact)
        artifact["undisclosed_binding"] = "not allowed"
        self.fixture.artifact = artifact
        self.fixture.write_canonical()
        self._assert_fails("unknown-or-missing-field")

    def test_artifact_endpoint_digest_must_hash_captured_endpoint_bytes(self) -> None:
        artifact = copy.deepcopy(self.fixture.artifact)
        artifact["endpoint_payload_sha256"] = "b" * 64
        self.fixture.artifact = artifact
        self.fixture.write_canonical()
        self._assert_fails("endpoint-artifact-digest-mismatch")

    def test_server_artifact_copy_must_be_byte_identical(self) -> None:
        self.fixture.server_artifact_path.write_bytes(b"{}")
        self._assert_fails("startup-artifact-copy-mismatch")

    def test_source_is_self_contained_and_tracks_raw_contract_inventory(self) -> None:
        source = VALIDATOR.read_text(encoding="utf-8")
        for required in (
            "effective_runtime_config_contract.py",
            "CONFIG_DIMENSIONS = (",
            'ENDPOINT_VERSION = "riley.effective-runtime-config.v1"',
            'STARTUP_ARTIFACT_VERSION = "riley.effective-runtime-config-startup-artifact.v1"',
            "canonical_json_bytes",
            "duplicate-json-key",
            "validate_raw_capture",
        ):
            self.assertIn(required, source)
        self.assertNotIn("import check_rc3_qualification", source)
        self.assertNotIn("import check_effective_runtime_config_receipt", source)
        self.assertNotIn("import tomllib", source)


if __name__ == "__main__":
    unittest.main()
