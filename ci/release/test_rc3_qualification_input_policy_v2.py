#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 qualification-input denial policy."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


RELEASE_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(RELEASE_DIRECTORY))

import rc3_qualification_input_policy_v2 as policy  # noqa: E402


def canonical(value: object) -> bytes:
    return policy.canonical_json_bytes(value)


class QualificationInputPolicyTests(unittest.TestCase):
    maxDiff = None

    def assert_reason(self, expected: str, raw: bytes) -> None:
        with self.assertRaises(policy.QualificationInputPolicyError) as raised:
            policy.reject_qualification_input_bytes(raw)
        self.assertEqual(getattr(raised.exception, "reason_code", None), expected)

    def test_policy_has_no_admitted_inputs_or_operational_surface(self) -> None:
        self.assertEqual(policy.ADMITTED_INPUTS, frozenset())
        source = inspect.getsource(policy)
        for forbidden in (
            "import os",
            "import pathlib",
            "import subprocess",
            "import socket",
            "open(",
            "__main__",
        ):
            self.assertNotIn(forbidden, source)

    def test_exact_historical_and_narrow_schemas_are_rejected_before_status(self) -> None:
        cases = {
            "riley.soak-v2-receipt.v1": "historical-soak-v1-rejected",
            "riley.soak-v2-raw-provenance.v5": "raw-soak-v5-rejected",
            "riley.soak-v2-semantic-replay.v2": "soak-semantic-replay-not-durable",
            "riley.rc3-rollback-receipt.v1": "historical-rollback-v1-rejected",
            "riley.rc3-rollback-raw-structural-precheck.v1": "rollback-structural-precheck-rejected",
            "riley.rc3-rollback-operational-semantics.v1": "rollback-held-fd-diagnostic-rejected",
            "riley.rc3-rollback-finalizer-receipt.v1": "rollback-finalizer-receipt-not-semantic",
            "riley.rc3-gate-e-aggregate-semantic-replay.v1": "gate-e-aggregate-replay-not-durable",
            "riley.rc3-gate-e-aggregate-replay-record.v1": "gate-e-aggregate-record-not-durable",
            "riley.rc3-freeze-input-admission.v1": "freeze-input-admission-not-frozen-candidate",
            "riley.rc3-frozen-candidate.v1": "frozen-candidate-identity-not-semantic",
            "riley.reconstructed-runtime-a-b-materialization.v1": "reconstructed-materialization-not-qualification",
            "riley.reconstructed-runtime-python-prerequisite.v1": "runtime-python-prerequisite-not-materialization",
            "riley.rc3-gate-e-root-bundle.v1": "native-root-bundle-manifest-not-qualification",
            "riley.rc3-gate-e-execution-closure-manifest.v1": "execution-closure-manifest-not-qualification",
            "riley.rc3-gate-e-native-guardian-review.v1": "native-guardian-review-input-not-qualification",
            "riley.release-candidate-report.v2": "legacy-release-candidate-rejected",
        }
        for schema_version, expected in cases.items():
            with self.subTest(schema_version=schema_version):
                self.assert_reason(
                    expected,
                    canonical(
                        {
                            "schema_version": schema_version,
                            "authority": "invented-admission-authority",
                            "qualification_status": "passed",
                            "status": "passed",
                        }
                    ),
                )

    def test_native_root_bundle_preflight_is_rejected_before_authority_or_status(self) -> None:
        self.assert_reason(
            "native-root-bundle-preflight-not-qualification",
            canonical(
                {
                    "actual_gate_e_producer": "not-established",
                    "authority": "not-authoritative",
                    "docker_execution": "not-established",
                    "evidence": "not-established",
                    "execution_authority": "not-established",
                    "gpu_execution": "not-run",
                    "guardian_lease": "not-established",
                    "host_initial_namespace": "not-established",
                    "installation": "not-installed",
                    "interpreter_runtime_closure": "not-established",
                    "object_observation_status": "checked",
                    "pre_python_loader_boundary": "not-established",
                    "qualification_status": "not-run",
                    "reason_code": None,
                    "same_object_exec": "not-established",
                    "schema_version": "riley.rc3-gate-e-native-root-bundle-preflight.v1",
                    "scope": "root-bundle-object-observation-only",
                    "status": "not-established",
                }
            ),
        )

    def test_native_root_bundle_json_line_is_noncanonical_before_schema_dispatch(self) -> None:
        self.assert_reason(
            "noncanonical-json",
            b'{"schema_version":"riley.rc3-gate-e-native-root-bundle-preflight.v1",'
            b'"status":"not-established","object_observation_status":"not-established",'
            b'"scope":"root-bundle-object-observation-only",'
            b'"authority":"not-authoritative","installation":"not-installed",'
            b'"host_initial_namespace":"not-established",'
            b'"pre_python_loader_boundary":"not-established",'
            b'"same_object_exec":"not-established",'
            b'"interpreter_runtime_closure":"not-established",'
            b'"guardian_lease":"not-established",'
            b'"execution_authority":"not-established",'
            b'"actual_gate_e_producer":"not-established",'
            b'"gpu_execution":"not-run","docker_execution":"not-established",'
            b'"evidence":"not-established","qualification_status":"not-run",'
            b'"reason_code":"effective-uid-gid-not-root"}\n',
        )

    def test_execution_closure_manifest_line_is_noncanonical_before_schema_dispatch(self) -> None:
        self.assert_reason(
            "noncanonical-json",
            b'{"schema_version":"riley.rc3-gate-e-execution-closure-manifest.v1"}\n',
        )

    def test_exact_narrow_authorities_are_rejected_without_schema_allowlist(self) -> None:
        cases = {
            "raw-structural-only": "raw-structural-input-rejected",
            "soak-v2-semantic-replay-only": "soak-semantic-replay-not-durable",
            "raw-operational-semantics-only": "rollback-held-fd-diagnostic-rejected",
            "gate-e-aggregate-semantic-replay-only": "gate-e-aggregate-replay-not-durable",
            "freeze-input-structural-only": "freeze-input-admission-not-frozen-candidate",
            "held-fd-a-b-runtime-assembly-content-closure-only": "reconstructed-materialization-not-qualification",
            "interpreter-readiness-only": "runtime-python-prerequisite-not-materialization",
            "not-authoritative": "non-authoritative-input-rejected",
        }
        for authority, expected in cases.items():
            with self.subTest(authority=authority):
                self.assert_reason(
                    expected,
                    canonical(
                        {
                            "schema_version": "riley.future-durable-receipt.v2",
                            "authority": authority,
                            "status": "passed",
                        }
                    ),
                )

    def test_v2_suffix_and_passed_text_never_create_admission(self) -> None:
        self.assert_reason(
            "unrecognized-passed-qualification-input",
            canonical(
                {
                    "schema_version": "riley.future-qualification-receipt.v2",
                    "authority": "future-qualification-authority",
                    "qualification_status": "passed",
                    "status": "passed",
                }
            ),
        )
        self.assert_reason(
            "qualification-not-run-input-rejected",
            canonical(
                {
                    "schema_version": "riley.future-qualification-receipt.v99",
                    "qualification_status": "not-run",
                    "status": "bound",
                }
            ),
        )
        self.assert_reason(
            "unsupported-qualification-input",
            canonical({"schema_version": "riley.unknown.v2", "status": "bound"}),
        )

    def test_malformed_duplicate_noncanonical_and_oversized_documents_fail_closed(self) -> None:
        self.assert_reason("invalid-json-byte-length", b"")
        self.assert_reason("duplicate-json-key", b'{"schema_version":"one","schema_version":"two"}')
        self.assert_reason("noncanonical-json", b'{"schema_version": "riley.unknown.v2"}')
        self.assert_reason("invalid-json-root", canonical(["not-an-object"]))
        self.assert_reason("non-finite-json-number", b'{"schema_version":NaN}')
        self.assert_reason("non-finite-json-number", b'{"schema_version":"riley.unknown.v2","value":1e9999}')
        self.assert_reason("json-byte-budget-exceeded", b"x" * (policy.MAX_INPUT_BYTES + 1))

    def test_deep_and_invalid_headers_fail_before_generic_rejection(self) -> None:
        nested: object = {"schema_version": "riley.unknown.v2"}
        for _ in range(policy.MAX_JSON_NESTING + 1):
            nested = {"nested": nested}
        self.assert_reason("json-nesting-too-deep", canonical(nested))
        self.assert_reason(
            "json-node-budget-exceeded",
            canonical({"schema_version": "riley.unknown.v2", "nodes": [0] * policy.MAX_JSON_NODES}),
        )
        self.assert_reason("invalid-schema-version", canonical({"schema_version": 2}))
        self.assert_reason(
            "invalid-qualification-authority",
            canonical({"schema_version": "riley.unknown.v2", "authority": []}),
        )


if __name__ == "__main__":
    unittest.main()
