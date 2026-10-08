#!/usr/bin/env python3
"""CPU-only hostile-path tests for the native guardian review-input contract."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path


RELEASE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_DIRECTORY = RELEASE_DIRECTORY.parents[1]
SCHEMA_PATH = (
    REPOSITORY_DIRECTORY
    / "benchmarks"
    / "release"
    / "candidates"
    / "gate-e-native-guardian-review-v1.schema.json"
)
sys.path.insert(0, str(RELEASE_DIRECTORY))

import gate_e_native_guardian_review_contract_v1 as contract  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def binding(label: str, length: int = 1_024) -> dict[str, object]:
    return {"byte_length": length, "sha256": digest(label)}


def review_document(*, dynamic: bool = False) -> dict[str, object]:
    sidecar = binding("execution-closure-sidecar", 4_096)
    strategy: dict[str, object]
    if dynamic:
        strategy = {
            "kind": contract.DYNAMIC_STRATEGY,
            "pt_interp": "reviewed-held-object",
            "dynamic_loader_binding": binding("dynamic-loader", 250_000),
            "loader_resolution_proof_sha256": digest("loader-resolution-proof"),
            "rejection_policy_sha256": digest("dynamic-rejection-policy"),
        }
    else:
        strategy = {
            "kind": contract.STATIC_STRATEGY,
            "pt_interp": "absent",
            "dependency_resolution": "none",
            "static_elf_inspection_policy_sha256": digest("static-elf-policy"),
        }
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "scope": contract.SCOPE,
        "authority": contract.AUTHORITY,
        "review_status": contract.REVIEW_STATUS,
        "installation_status": contract.INSTALLATION_STATUS,
        "operational_status": contract.OPERATIONAL_STATUS,
        "bundle": {
            "bundle_schema_version": contract.BUNDLE_SCHEMA_VERSION,
            "v1_compatibility": False,
            "execution_closure_sidecar": sidecar,
            "manifest": {
                "byte_length": 8_192,
                "sha256": digest("bundle-manifest"),
                "execution_closure_sidecar_byte_length": sidecar["byte_length"],
                "execution_closure_sidecar_sha256": sidecar["sha256"],
            },
        },
        "execution_strategy": strategy,
        "fd_abi": {
            "bootstrap_inherited_fds": list(contract.BOOTSTRAP_FDS),
            "worker_inherited_fds": list(contract.WORKER_FDS),
            "core_fd": 32,
            "environment": "empty",
            "no_new_privs": True,
            "capabilities": "cleared",
        },
        "required_artifacts": {
            name: digest(f"artifact:{name}") for name in sorted(contract.REQUIRED_ARTIFACTS)
        },
    }


def canonical(value: object) -> bytes:
    return contract.canonical_native_guardian_review_bytes(value)


class NativeGuardianReviewContractTests(unittest.TestCase):
    maxDiff = None

    def assert_reason(self, expected: str, raw: bytes) -> None:
        with self.assertRaises(contract.NativeGuardianReviewContractError) as raised:
            contract.parse_native_guardian_review_v1(raw)
        self.assertEqual(getattr(raised.exception, "reason_code", None), expected)

    def test_schema_declares_exact_non_authoritative_review_input_shape(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["schema_version"]["const"], contract.SCHEMA_VERSION
        )
        self.assertEqual(schema["properties"]["authority"]["const"], contract.AUTHORITY)
        self.assertEqual(
            schema["properties"]["review_status"]["const"], contract.REVIEW_STATUS
        )
        self.assertEqual(
            schema["properties"]["installation_status"]["const"],
            contract.INSTALLATION_STATUS,
        )
        self.assertEqual(
            schema["properties"]["operational_status"]["const"],
            contract.OPERATIONAL_STATUS,
        )
        self.assertEqual(len(schema["properties"]["execution_strategy"]["oneOf"]), 2)
        self.assertEqual(
            schema["properties"]["fd_abi"]["properties"]["bootstrap_inherited_fds"]["const"],
            list(contract.BOOTSTRAP_FDS),
        )
        self.assertEqual(
            set(schema["properties"]["required_artifacts"]["required"]),
            set(contract.REQUIRED_ARTIFACTS),
        )

    def test_contract_has_no_filesystem_process_or_operational_surface(self) -> None:
        source = inspect.getsource(contract)
        for forbidden in (
            "import os",
            "import pathlib",
            "import subprocess",
            "import socket",
            "open(",
            "__main__",
            "exec(",
            "eval(",
        ):
            self.assertNotIn(forbidden, source)

    def test_static_input_returns_only_opaque_declared_identities(self) -> None:
        document = review_document()
        raw = canonical(document)

        parsed = contract.parse_native_guardian_review_v1(raw)

        self.assertEqual(parsed.execution_strategy, contract.STATIC_STRATEGY)
        self.assertEqual(parsed.bundle_manifest.byte_length, 8_192)
        self.assertEqual(parsed.execution_closure_sidecar.byte_length, 4_096)
        self.assertEqual(parsed.review_input_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(raw, canonical(json.loads(raw[:-1].decode("utf-8"))))

    def test_dynamic_input_requires_the_explicit_alternative_shape(self) -> None:
        document = review_document(dynamic=True)
        parsed = contract.parse_native_guardian_review_v1(canonical(document))
        self.assertEqual(parsed.execution_strategy, contract.DYNAMIC_STRATEGY)

    def test_exact_canonical_bytes_duplicate_keys_and_terminal_newline_are_required(self) -> None:
        raw = canonical(review_document())

        self.assert_reason("invalid-terminal-newline", raw[:-1])
        self.assert_reason("invalid-terminal-newline", raw + b"\n")
        self.assert_reason("noncanonical-json", raw.replace(b":", b": ", 1))
        self.assert_reason(
            "duplicate-json-key",
            b'{"schema_version":"one","schema_version":"two"}\n',
        )
        self.assert_reason("non-finite-json-number", b'{"schema_version":NaN}\n')
        self.assert_reason(
            "integer-literal-too-large",
            b'{"byte_length":' + (b"9" * 5_000) + b"}\n",
        )
        self.assert_reason("invalid-json-number", b'{"byte_length":1.0}\n')
        self.assert_reason("invalid-json-number", b'{"byte_length":1e3}\n')
        self.assert_reason("invalid-json", b"\xff\n")

    def test_root_status_and_field_set_never_claim_authority_or_approval(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        extra = review_document()
        extra["unexpected"] = True
        cases.append(("unexpected-field-set", extra))

        unsupported = review_document()
        unsupported["schema_version"] = "riley.rc3-gate-e-native-guardian-review.v2"
        cases.append(("unsupported-schema-version", unsupported))

        authority = review_document()
        authority["authority"] = "administrator-approved"
        cases.append(("invalid-authority", authority))

        review_status = review_document()
        review_status["review_status"] = "approved"
        cases.append(("invalid-review-status", review_status))

        installed = review_document()
        installed["installation_status"] = "installed"
        cases.append(("invalid-installation-status", installed))

        operational = review_document()
        operational["operational_status"] = "gpu-authorized"
        cases.append(("invalid-operational-status", operational))

        for expected, document in cases:
            with self.subTest(expected=expected):
                self.assert_reason(expected, canonical(document))

    def test_v2_bundle_and_raw_sidecar_manifest_binding_are_closed(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        legacy = review_document()
        legacy["bundle"] = copy.deepcopy(legacy["bundle"])
        legacy["bundle"]["v1_compatibility"] = True  # type: ignore[index]
        cases.append(("legacy-v1-reuse", legacy))

        wrong_schema = review_document()
        wrong_schema["bundle"] = copy.deepcopy(wrong_schema["bundle"])
        wrong_schema["bundle"]["bundle_schema_version"] = "riley.rc3-gate-e-v1"  # type: ignore[index]
        cases.append(("invalid-bundle-schema-version", wrong_schema))

        sidecar_mismatch = review_document()
        sidecar_mismatch["bundle"] = copy.deepcopy(sidecar_mismatch["bundle"])
        sidecar_mismatch["bundle"]["manifest"]["execution_closure_sidecar_sha256"] = digest("other-sidecar")  # type: ignore[index]
        cases.append(("sidecar-manifest-binding-mismatch", sidecar_mismatch))

        zero = review_document()
        zero["bundle"] = copy.deepcopy(zero["bundle"])
        zero["bundle"]["execution_closure_sidecar"]["sha256"] = "0" * 64  # type: ignore[index]
        cases.append(("zero-sha256", zero))

        too_large = review_document()
        too_large["bundle"] = copy.deepcopy(too_large["bundle"])
        too_large["bundle"]["execution_closure_sidecar"]["byte_length"] = contract.MAX_SIDECAR_BYTES + 1  # type: ignore[index]
        cases.append(("invalid-byte-length", too_large))

        for expected, document in cases:
            with self.subTest(expected=expected):
                self.assert_reason(expected, canonical(document))

    def test_strategy_is_exactly_one_closed_static_or_dynamic_choice(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        static_interp = review_document()
        static_interp["execution_strategy"] = copy.deepcopy(static_interp["execution_strategy"])
        static_interp["execution_strategy"]["pt_interp"] = "reviewed-held-object"  # type: ignore[index]
        cases.append(("invalid-execution-strategy", static_interp))

        static_extra = review_document()
        static_extra["execution_strategy"] = copy.deepcopy(static_extra["execution_strategy"])
        static_extra["execution_strategy"]["dynamic_loader_binding"] = binding("unexpected")  # type: ignore[index]
        cases.append(("unexpected-field-set", static_extra))

        dynamic_missing = review_document(dynamic=True)
        dynamic_missing["execution_strategy"] = copy.deepcopy(dynamic_missing["execution_strategy"])
        del dynamic_missing["execution_strategy"]["loader_resolution_proof_sha256"]  # type: ignore[index]
        cases.append(("unexpected-field-set", dynamic_missing))

        dynamic_bad_loader = review_document(dynamic=True)
        dynamic_bad_loader["execution_strategy"] = copy.deepcopy(dynamic_bad_loader["execution_strategy"])
        dynamic_bad_loader["execution_strategy"]["dynamic_loader_binding"]["sha256"] = "0" * 64  # type: ignore[index]
        cases.append(("zero-sha256", dynamic_bad_loader))

        unknown = review_document()
        unknown["execution_strategy"] = {"kind": "infer-from-host"}
        cases.append(("invalid-execution-strategy", unknown))

        for expected, document in cases:
            with self.subTest(expected=expected):
                self.assert_reason(expected, canonical(document))

    def test_fd_abi_and_artifact_digest_set_are_closed(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        fd_order = review_document()
        fd_order["fd_abi"] = copy.deepcopy(fd_order["fd_abi"])
        fd_order["fd_abi"]["bootstrap_inherited_fds"] = [0, 1, 2, 32, 31]  # type: ignore[index]
        cases.append(("invalid-fd-abi", fd_order))

        worker_leak = review_document()
        worker_leak["fd_abi"] = copy.deepcopy(worker_leak["fd_abi"])
        worker_leak["fd_abi"]["worker_inherited_fds"] = [0, 1, 2, 32]  # type: ignore[index]
        cases.append(("invalid-fd-abi", worker_leak))

        fd_boolean = review_document()
        fd_boolean["fd_abi"] = copy.deepcopy(fd_boolean["fd_abi"])
        fd_boolean["fd_abi"]["bootstrap_inherited_fds"][0] = False  # type: ignore[index]
        cases.append(("invalid-fd-abi", fd_boolean))

        worker_boolean = review_document()
        worker_boolean["fd_abi"] = copy.deepcopy(worker_boolean["fd_abi"])
        worker_boolean["fd_abi"]["worker_inherited_fds"] = [False, True, 2]  # type: ignore[index]
        cases.append(("invalid-fd-abi", worker_boolean))

        missing_artifact = review_document()
        missing_artifact["required_artifacts"] = copy.deepcopy(missing_artifact["required_artifacts"])
        del missing_artifact["required_artifacts"]["threat_model"]  # type: ignore[index]
        cases.append(("unexpected-field-set", missing_artifact))

        zero_artifact = review_document()
        zero_artifact["required_artifacts"] = copy.deepcopy(zero_artifact["required_artifacts"])
        zero_artifact["required_artifacts"]["threat_model"] = "0" * 64  # type: ignore[index]
        cases.append(("zero-sha256", zero_artifact))

        for expected, document in cases:
            with self.subTest(expected=expected):
                self.assert_reason(expected, canonical(document))

    def test_json_depth_node_and_document_byte_budgets_fail_closed(self) -> None:
        nested: object = {"schema_version": contract.SCHEMA_VERSION}
        for _ in range(contract.MAX_JSON_NESTING + 1):
            nested = {"nested": nested}
        self.assert_reason("json-nesting-too-deep", canonical(nested))

        node_heavy = {"schema_version": contract.SCHEMA_VERSION, "nodes": [0] * contract.MAX_JSON_NODES}
        self.assert_reason("json-node-budget-exceeded", canonical(node_heavy))
        self.assert_reason("review-byte-budget-exceeded", b"x" * (contract.MAX_DOCUMENT_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
