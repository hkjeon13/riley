#!/usr/bin/env python3
"""CPU-only adversarial tests for the fixed C02 routing receipt checker."""

from __future__ import annotations

import copy
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import check_rc3_qualification as qualification
import check_rc3_routing_receipt as routing


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class RoutingReceiptFixture:
    """A complete local evidence tree; Gate E is mocked only at its boundary."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.freeze_path = root / "riley-0.1.0-rc3.freeze.json"
        self.candidate_id = "riley-0.1.0-rc3"
        self.base_relative = "reports/final.json"
        self.manifest_relative = "candidates/final-release-candidate.json"
        self.semantic_report_relative = "receipts/routing.json"
        self.raw_receipt_relative = "routing/raw-receipt.json"
        self.trace_manifest_relative = "routing/traces.json"
        self.executable_relative = "routing/riley-release"
        self.executable_bytes = b"\x7fELF\x02\x01\x01\x00fixed-routing-release-binary"
        self.corpus = routing._load_corpus()
        self.trace_documents: dict[str, dict[str, object]] = {}
        self.trace_paths: dict[str, str] = {}
        self.trace_raws: dict[str, bytes] = {}

        release = {
            "binary_sha256": digest(self.executable_bytes),
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
            "argv": ["serve", "--execution-completion", "iteration-batch"],
            "environment": {"RILEY_ROUTING_RECEIPT": "1"},
        }
        maximum_input = {
            "argv": ["serve", "--execution-completion", "per-operation"],
            "environment": {"RILEY_EXACT": "1"},
        }
        arms = {
            "stable_default": {
                **stable_input,
                "configuration_sha256": digest(qualification.canonical_json_bytes(stable_input)),
            },
            "max_performance_exact": {
                **maximum_input,
                "configuration_sha256": digest(qualification.canonical_json_bytes(maximum_input)),
            },
        }
        models = {
            name: {
                "model_id": f"fixture/{name}",
                "model_revision": "b" * 40,
                "config_sha256": digest(f"{name} config"),
                "weights_sha256": digest(f"{name} weights"),
                "tokenizer_revision": "c" * 40,
                "tokenizer_files_sha256": digest(f"{name} tokenizer"),
            }
            for name in ("smollm2", "qwen")
        }
        self.freeze: dict[str, object] = {
            "schema_version": qualification.FREEZE_VERSION,
            "candidate_id": self.candidate_id,
            "created_at_utc": "2026-08-28T00:00:00Z",
            "status": "frozen",
            "source": {
                "git_revision": "a" * 40,
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
            "models": models,
            "arms": arms,
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
                            self.semantic_report_relative if gate == "routing" else f"receipts/{gate}.json"
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

    def descriptor(self, relative: str, raw: bytes) -> dict[str, str]:
        return {"path": relative, "sha256": digest(raw)}

    def bindings(self) -> dict[str, str]:
        arms = self.freeze["arms"]
        assert isinstance(arms, dict)
        stable = arms["stable_default"]
        assert isinstance(stable, dict)
        return {
            "freeze_sha256": self.freeze_sha,
            "base_release_candidate_report_sha256": self.base_sha,
            "configuration_profile": routing.STABLE_DEFAULT_PROFILE,
            "configuration_sha256": str(stable["configuration_sha256"]),
        }

    def model(self) -> dict[str, str]:
        models = self.freeze["models"]
        assert isinstance(models, dict)
        model = models["smollm2"]
        assert isinstance(model, dict)
        return copy.deepcopy(model)  # type: ignore[return-value]

    def execution_trace(self) -> dict[str, str]:
        source = self.freeze["source"]
        release = self.freeze["release"]
        images = self.freeze["images"]
        assert isinstance(source, dict) and isinstance(release, dict) and isinstance(images, dict)
        return {
            "source_revision": str(source["git_revision"]),
            "source_archive_sha256": str(source["archive_sha256"]),
            "release_binary_sha256": str(release["binary_sha256"]),
            "release_bundle_sha256": str(release["bundle_sha256"]),
            "release_image_id": str(release["image_id"]),
            "test_image_id": str(images["cuda"]),
            "test_executable_sha256": str(release["binary_sha256"]),
        }

    def execution_receipt(self) -> dict[str, object]:
        result: dict[str, object] = copy.deepcopy(self.execution_trace())
        executable_sha256 = result.pop("test_executable_sha256")
        result["test_executable"] = {
            "path": self.executable_relative,
            "sha256": executable_sha256,
        }
        return result

    def materialize(self) -> str:
        self.freeze_sha = digest(self.write_canonical(self.freeze_path, self.freeze))
        self.base_raw = self.write_canonical(
            self.evidence / self.base_relative, {"replayed": "gate-e-fixture"}
        )
        self.base_sha = digest(self.base_raw)
        self.write_canonical(self.evidence / self.manifest_relative, {"fixture": "gate-e-manifest"})
        executable_path = self.evidence / self.executable_relative
        executable_path.parent.mkdir(parents=True, exist_ok=True)
        executable_path.write_bytes(self.executable_bytes)
        for case in self.corpus.cases:
            case_id = str(case["case_id"])
            self.trace_paths[case_id] = f"routing/traces/{case_id}.json"
            self.trace_documents[case_id] = {
                "schema_version": routing.TRACE_VERSION,
                "candidate_id": self.candidate_id,
                "bindings": self.bindings(),
                "model": self.model(),
                "execution": self.execution_trace(),
                "corpus": {"path": self.corpus.descriptor.path, "sha256": self.corpus.descriptor.sha256},
                "case_id": case_id,
                "trace": copy.deepcopy(case["trace"]),
            }
        self.rebuild()
        return self.freeze_sha

    def rebuild(self) -> None:
        for case in self.corpus.cases:
            case_id = str(case["case_id"])
            self.trace_raws[case_id] = self.write_canonical(
                self.evidence / self.trace_paths[case_id], self.trace_documents[case_id]
            )
        self.manifest_document: dict[str, object] = {
            "schema_version": routing.TRACE_MANIFEST_VERSION,
            "candidate_id": self.candidate_id,
            "bindings": self.bindings(),
            "model": self.model(),
            "execution": self.execution_trace(),
            "corpus": {"path": self.corpus.descriptor.path, "sha256": self.corpus.descriptor.sha256},
            "traces": [
                {
                    "case_id": case_id,
                    "trace": self.descriptor(self.trace_paths[case_id], self.trace_raws[case_id]),
                }
                for case_id in routing.CASE_IDS
            ],
        }
        manifest_raw = self.write_canonical(self.evidence / self.trace_manifest_relative, self.manifest_document)
        self.receipt_document: dict[str, object] = {
            "schema_version": routing.RECEIPT_VERSION,
            "status": "passed",
            "passed": True,
            "candidate_id": self.candidate_id,
            "bindings": self.bindings(),
            "model": self.model(),
            "execution": self.execution_receipt(),
            "corpus": {"path": self.corpus.descriptor.path, "sha256": self.corpus.descriptor.sha256},
            "trace_manifest": self.descriptor(self.trace_manifest_relative, manifest_raw),
        }
        self.receipt_raw = self.write_canonical(
            self.evidence / self.raw_receipt_relative, self.receipt_document
        )

    def rewrite_trace(self, case_id: str) -> None:
        self.rebuild()


class RoutingReceiptTests(unittest.TestCase):
    def fixture(self) -> RoutingReceiptFixture:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture = RoutingReceiptFixture(Path(temporary.name))
        fixture.materialize()
        return fixture

    def evaluate(self, fixture: RoutingReceiptFixture) -> dict[str, object]:
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(fixture.base_raw, fixture.base_sha),
        ):
            return routing.evaluate(
                fixture.freeze_path,
                fixture.evidence,
                fixture.raw_receipt_relative,
                expected_freeze_sha256=fixture.freeze_sha,
            )

    def assert_rejected(self, report: dict[str, object]) -> None:
        self.assertFalse(report["passed"])
        self.assertIn(report["status"], {"failed", "incomparable"})
        self.assertTrue(report["reason_codes"])

    def test_valid_fixed_release_routing_receipt_replays_and_exposes_outer_api(self) -> None:
        fixture = self.fixture()
        report = self.evaluate(fixture)
        self.assertEqual("passed", report["status"])
        self.assertTrue(report["passed"])
        self.assertEqual(list(routing.CASE_IDS), [trace["case_id"] for trace in report["traces"]])
        parsed = routing.validate_check_report(report)
        self.assertEqual(fixture.raw_receipt_relative, parsed.receipt.path)
        self.assertEqual(fixture.executable_relative, parsed.execution["test_executable"]["path"])

    def test_generic_passed_or_hash_only_receipt_is_not_a_gate(self) -> None:
        fixture = self.fixture()
        fixture.write_canonical(
            fixture.evidence / fixture.raw_receipt_relative,
            {"schema_version": routing.RECEIPT_VERSION, "status": "passed", "passed": True},
        )
        self.assert_rejected(self.evaluate(fixture))

    def test_source_controlled_corpus_and_frozen_release_binding_cannot_drift(self) -> None:
        fixture = self.fixture()
        with mock.patch.object(routing, "CORPUS_SHA256", "f" * 64):
            report = self.evaluate(fixture)
        self.assert_rejected(report)
        self.assertIn("routing-corpus-sha-mismatch", report["reason_codes"])

        fixture = self.fixture()
        execution = fixture.receipt_document["execution"]
        assert isinstance(execution, dict)
        execution["test_image_id"] = "sha256:" + digest("unfrozen test image")
        fixture.write_canonical(fixture.evidence / fixture.raw_receipt_relative, fixture.receipt_document)
        report = self.evaluate(fixture)
        self.assert_rejected(report)
        self.assertEqual("incomparable", report["status"])

    def test_slot_request_token_mutation_is_rejected_even_when_descriptor_hashes_are_rebuilt(self) -> None:
        fixture = self.fixture()
        trace = fixture.trace_documents["routing-c5-permuted-mixed"]
        body = trace["trace"]
        assert isinstance(body, dict)
        iterations = body["iterations"]
        assert isinstance(iterations, list) and isinstance(iterations[0], dict)
        published = iterations[0]["published_tokens"]
        assert isinstance(published, list) and isinstance(published[0], dict)
        published[0]["request_id"] = "c5-b"
        fixture.rewrite_trace("routing-c5-permuted-mixed")
        self.assert_rejected(self.evaluate(fixture))

    def test_precommit_cancellation_cannot_publish_a_token(self) -> None:
        fixture = self.fixture()
        trace = fixture.trace_documents["routing-c8-cancel-precommit"]
        body = trace["trace"]
        assert isinstance(body, dict)
        iteration = body["iterations"][0]
        assert isinstance(iteration, dict)
        published = iteration["published_tokens"]
        assert isinstance(published, list)
        published.append({"sequence": 7, "slot": 1, "request_id": "c8-b", "token_id": 802})
        terminal_events = body["terminal_events"]
        assert isinstance(terminal_events, list)
        for index, event in enumerate(terminal_events):
            assert isinstance(event, dict)
            event["sequence"] = index + 8
        fixture.rewrite_trace("routing-c8-cancel-precommit")
        self.assert_rejected(self.evaluate(fixture))

    def test_malformed_plan_must_be_rejected_before_dispatch(self) -> None:
        fixture = self.fixture()
        trace = fixture.trace_documents["routing-malformed-pre-dispatch"]
        body = trace["trace"]
        assert isinstance(body, dict)
        body["dispatch_count"] = 1
        fixture.rewrite_trace("routing-malformed-pre-dispatch")
        self.assert_rejected(self.evaluate(fixture))

    def test_commit_failure_must_not_publish_or_leak_kv(self) -> None:
        fixture = self.fixture()
        trace = fixture.trace_documents["routing-commit-failure-contained"]
        body = trace["trace"]
        assert isinstance(body, dict)
        iteration = body["iterations"][0]
        assert isinstance(iteration, dict)
        iteration["published_tokens"] = [
            {"sequence": 0, "slot": 0, "request_id": "commit-b", "token_id": 902}
        ]
        terminal_events = body["terminal_events"]
        assert isinstance(terminal_events, list)
        for index, event in enumerate(terminal_events, start=1):
            assert isinstance(event, dict)
            event["sequence"] = index
        fixture.rewrite_trace("routing-commit-failure-contained")
        self.assert_rejected(self.evaluate(fixture))

    def test_symlink_and_descriptor_aliases_are_rejected(self) -> None:
        fixture = self.fixture()
        trace_path = fixture.evidence / fixture.trace_paths["routing-c1-basic"]
        outside = fixture.root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        trace_path.unlink()
        trace_path.symlink_to(outside)
        self.assert_rejected(self.evaluate(fixture))

        fixture = self.fixture()
        traces = fixture.manifest_document["traces"]
        assert isinstance(traces, list) and isinstance(traces[0], dict) and isinstance(traces[1], dict)
        traces[1]["trace"] = copy.deepcopy(traces[0]["trace"])
        manifest_raw = fixture.write_canonical(fixture.evidence / fixture.trace_manifest_relative, fixture.manifest_document)
        fixture.receipt_document["trace_manifest"] = fixture.descriptor(fixture.trace_manifest_relative, manifest_raw)
        fixture.write_canonical(fixture.evidence / fixture.raw_receipt_relative, fixture.receipt_document)
        self.assert_rejected(self.evaluate(fixture))

    def test_create_only_cli_report(self) -> None:
        fixture = self.fixture()
        output = fixture.root / "routing-check.json"
        argv = [
            "--freeze",
            str(fixture.freeze_path),
            "--expected-freeze-sha256",
            fixture.freeze_sha,
            "--evidence-root",
            str(fixture.evidence),
            "--receipt",
            fixture.raw_receipt_relative,
            "--report",
            str(output),
        ]
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(fixture.base_raw, fixture.base_sha),
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(0, routing.main(argv))
        self.assertTrue(output.exists())
        with mock.patch.object(
            qualification,
            "revalidate_base_release_candidate",
            return_value=(fixture.base_raw, fixture.base_sha),
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(2, routing.main(argv))


if __name__ == "__main__":
    unittest.main()
