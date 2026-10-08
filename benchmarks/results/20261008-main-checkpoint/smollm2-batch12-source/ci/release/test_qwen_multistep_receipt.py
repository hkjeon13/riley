#!/usr/bin/env python3
"""CPU-only adversarial tests for the native C02 Qwen v2 receipt checker."""

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

import check_qwen_multistep_receipt as qwen
import check_rc3_qualification as qualification


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class QwenReceiptFixture:
    """A complete native public/audit evidence tree; Gate E is mocked only at replay."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.freeze_path = root / "riley-0.1.0-rc3.freeze.json"
        self.candidate_id = "riley-0.1.0-rc3"
        self.revision = "a" * 40
        self.golden = qwen._load_golden()
        self.wire = qwen._load_wire(self.golden)
        self.raw_receipt_relative = "qwen/raw-receipt-v2.json"
        self.raw_receipt_path = self.evidence / self.raw_receipt_relative
        self.semantic_report_relative = "receipts/qwen_multistep.json"
        self.base_relative = "reports/final.json"
        self.manifest_relative = "candidates/final-release-candidate.json"
        self.case_paths: dict[str, dict[str, dict[str, str]]] = {}
        self.case_documents: dict[str, dict[str, dict[str, object]]] = {}
        self.manifest_document: dict[str, object] = {}
        self.receipt_document: dict[str, object] = {}
        self.freeze: dict[str, object] = self._freeze_document()

    def _freeze_document(self) -> dict[str, object]:
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
            "argv": ["serve", "--execution-completion", "iteration-batch"],
            "environment": {"RILEY_CONFIG_RECEIPT": "1"},
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
        return {
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
                "smollm2": {
                    "model_id": "fixture/smollm2",
                    "model_revision": "b" * 40,
                    "config_sha256": digest("smollm config"),
                    "weights_sha256": digest("smollm weights"),
                    "tokenizer_revision": "c" * 40,
                    "tokenizer_files_sha256": digest("smollm tokenizer"),
                },
                "qwen": copy.deepcopy(self.golden.model),
            },
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
                    gate: {"path": self.semantic_report_relative if gate == "qwen_multistep" else f"receipts/{gate}.json"}
                    for gate in qualification.REQUIRED_GATES
                },
            },
            "required_gates": list(qualification.REQUIRED_GATES),
        }

    def canonical(self, value: object) -> bytes:
        return qualification.canonical_json_bytes(value)

    def write_bytes(self, relative: str, raw: bytes) -> bytes:
        path = self.evidence / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def write_canonical(self, path: Path, document: object) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.canonical(document)
        path.write_bytes(raw)
        return raw

    def descriptor(self, relative: str, raw: bytes) -> dict[str, str]:
        return {"path": relative, "sha256": digest(raw)}

    def write_freeze(self) -> str:
        return digest(self.write_canonical(self.freeze_path, self.freeze))

    def bindings(self) -> dict[str, str]:
        arms = self.freeze["arms"]
        assert isinstance(arms, dict)
        stable = arms["stable_default"]
        assert isinstance(stable, dict)
        return {
            "freeze_sha256": self.freeze_sha,
            "base_release_candidate_report_sha256": self.base_sha,
            "configuration_profile": qwen.STABLE_DEFAULT_PROFILE,
            "configuration_sha256": str(stable["configuration_sha256"]),
        }

    def _artifact(self, case_id: str, mode: str, role: str, raw: bytes, document: dict[str, object] | None = None) -> tuple[str, bytes]:
        relative = f"qwen/{case_id}/{mode}/{role}"
        self.write_bytes(relative, raw)
        self.case_paths.setdefault(case_id, {}).setdefault(mode, {})[role] = relative
        if document is not None:
            self.case_documents.setdefault(case_id, {}).setdefault(mode, {})[role] = document
        return relative, raw

    @staticmethod
    def _headers(*, stream: bool, body: bytes) -> bytes:
        lines = (
            [
                "HTTP/1.1 200 OK",
                "Content-Type: text/event-stream; charset=utf-8",
                "Cache-Control: no-cache",
                "Connection: close",
                "X-Content-Type-Options: nosniff",
                "",
                "",
            ]
            if stream
            else [
                "HTTP/1.1 200 OK",
                "Content-Type: application/json; charset=utf-8",
                f"Content-Length: {len(body)}",
                "Connection: close",
                "X-Content-Type-Options: nosniff",
                "",
                "",
            ]
        )
        return "\r\n".join(lines).encode("ascii")

    @staticmethod
    def _sse_payload(request_id: str, created: int, model: str, text: str, finish_reason: str | None) -> dict[str, object]:
        return {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
        }

    @staticmethod
    def encode_sse(payloads: list[dict[str, object]]) -> bytes:
        frames = [
            b"data: " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
            for payload in payloads
        ]
        frames.append(b"data: [DONE]")
        return b"\n\n".join(frames) + b"\n\n"

    def _audit_document(self, golden_case: qwen.GoldenCase, request_id: str, delivery_mode: str) -> dict[str, object]:
        pieces = [{"token_id": piece.token_id, "text": piece.text} for piece in golden_case.expected_committed_output_tokens]
        return {
            "schema_version": qwen.AUDIT_VERSION,
            "committed_output_tokens": pieces,
            "delivery_mode": delivery_mode,
            "finish_reason": golden_case.finish_reason,
            "prompt_token_ids": list(golden_case.prompt_token_ids),
            "server_request_id": request_id,
            "usage": {
                "prompt_tokens": len(golden_case.prompt_token_ids),
                "completion_tokens": len(pieces),
                "total_tokens": len(golden_case.prompt_token_ids) + len(pieces),
            },
        }

    def _capture_mode(self, golden_case: qwen.GoldenCase, wire_case: qwen.WireCase, mode: str, serial: int) -> dict[str, dict[str, str]]:
        streaming = mode == "stream"
        request_id = f"cmpl-{golden_case.case_id}-{mode}"
        request = {
            "model": self.golden.model["model_id"],
            "prompt": wire_case.rendered_prompt,
            "max_tokens": golden_case.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": streaming,
        }
        request_relative, request_raw = self._artifact(golden_case.case_id, mode, "request-body.json", self.canonical(request), request)
        audit = self._audit_document(golden_case, request_id, "stream" if streaming else "non-stream")
        audit_relative, audit_raw = self._artifact(golden_case.case_id, mode, "audit-record.json", self.canonical(audit), audit)
        expected_text = "".join(piece.text for piece in golden_case.expected_committed_output_tokens)
        created = 1_787_900_000 + serial
        if streaming:
            payloads = [
                self._sse_payload(request_id, created, self.golden.model["model_id"], piece.text, None)
                for piece in golden_case.expected_committed_output_tokens
                if piece.text
            ]
            payloads.append(self._sse_payload(request_id, created, self.golden.model["model_id"], "", golden_case.finish_reason))
            response_raw = self.encode_sse(payloads)
            response_document = {"payloads": payloads}
            response_name = "response.sse"
        else:
            response_document = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": self.golden.model["model_id"],
                "choices": [{"text": expected_text, "index": 0, "logprobs": None, "finish_reason": golden_case.finish_reason}],
                "usage": {
                    "prompt_tokens": len(golden_case.prompt_token_ids),
                    "completion_tokens": len(golden_case.expected_committed_output_tokens),
                    "total_tokens": len(golden_case.prompt_token_ids) + len(golden_case.expected_committed_output_tokens),
                },
            }
            response_raw = self.canonical(response_document)
            response_name = "response.json"
        response_relative, response_raw = self._artifact(golden_case.case_id, mode, response_name, response_raw, response_document)
        header_relative, header_raw = self._artifact(golden_case.case_id, mode, "response.headers", self._headers(stream=streaming, body=response_raw))
        return {
            "request_body": self.descriptor(request_relative, request_raw),
            "response_headers": self.descriptor(header_relative, header_raw),
            "response_body": self.descriptor(response_relative, response_raw),
            "audit_record": self.descriptor(audit_relative, audit_raw),
        }

    def materialize(self) -> str:
        self.freeze_sha = self.write_freeze()
        self.base_raw = self.write_canonical(self.evidence / self.base_relative, {"replayed": "gate-e-fixture"})
        self.base_sha = digest(self.base_raw)
        self.write_canonical(self.evidence / self.manifest_relative, {"fixture": "gate-e-manifest"})
        cases: list[dict[str, object]] = []
        for index, (golden_case, wire_case) in enumerate(zip(self.golden.cases, self.wire.cases, strict=True)):
            cases.append(
                {
                    "case_id": golden_case.case_id,
                    "non_stream": self._capture_mode(golden_case, wire_case, "non-stream", index * 2 + 1),
                    "stream": self._capture_mode(golden_case, wire_case, "stream", index * 2 + 2),
                }
            )
        self.manifest_document = {
            "schema_version": qwen.CASE_MANIFEST_VERSION,
            "candidate_id": self.candidate_id,
            "bindings": self.bindings(),
            "model": copy.deepcopy(self.golden.model),
            "golden": {"path": self.golden.descriptor.path, "sha256": self.golden.descriptor.sha256},
            "wire": {"path": self.wire.descriptor.path, "sha256": self.wire.descriptor.sha256},
            "cases": cases,
        }
        self.rewrite_manifest_and_receipt()
        return self.freeze_sha

    def rewrite_manifest_and_receipt(self) -> None:
        manifest_raw = self.write_canonical(self.evidence / "qwen/cases-v2.json", self.manifest_document)
        self.receipt_document = {
            "schema_version": qwen.RECEIPT_VERSION,
            "status": "passed",
            "passed": True,
            "candidate_id": self.candidate_id,
            "bindings": self.bindings(),
            "model": copy.deepcopy(self.golden.model),
            "golden_id": self.golden.golden_id,
            "golden": {"path": self.golden.descriptor.path, "sha256": self.golden.descriptor.sha256},
            "wire": {"path": self.wire.descriptor.path, "sha256": self.wire.descriptor.sha256},
            "case_manifest": self.descriptor("qwen/cases-v2.json", manifest_raw),
        }
        self.write_canonical(self.raw_receipt_path, self.receipt_document)

    def manifest_case(self, index: int) -> dict[str, object]:
        cases = self.manifest_document["cases"]
        assert isinstance(cases, list)
        result = cases[index]
        assert isinstance(result, dict)
        return result

    def capture_document(self, index: int, mode: str, role: str) -> dict[str, object]:
        case_id = self.golden.cases[index].case_id
        document = self.case_documents[case_id][mode][role]
        assert isinstance(document, dict)
        return document

    def rewrite_capture_json(self, index: int, mode: str, role: str, document: dict[str, object]) -> None:
        case_id = self.golden.cases[index].case_id
        relative = self.case_paths[case_id][mode][role]
        raw = self.write_bytes(relative, self.canonical(document))
        self.case_documents[case_id][mode][role] = document
        capture = self.manifest_case(index)["stream" if mode == "stream" else "non_stream"]
        assert isinstance(capture, dict)
        manifest_role = "response_body" if role.startswith("response.") else "audit_record" if role == "audit-record.json" else "request_body"
        capture[manifest_role] = self.descriptor(relative, raw)
        self.rewrite_manifest_and_receipt()

    def rewrite_capture_raw(self, index: int, mode: str, role: str, raw: bytes) -> None:
        case_id = self.golden.cases[index].case_id
        relative = self.case_paths[case_id][mode][role]
        self.write_bytes(relative, raw)
        capture = self.manifest_case(index)["stream" if mode == "stream" else "non_stream"]
        assert isinstance(capture, dict)
        manifest_role = "response_headers" if role == "response.headers" else "response_body"
        capture[manifest_role] = self.descriptor(relative, raw)
        self.rewrite_manifest_and_receipt()


class QwenMultistepReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = QwenReceiptFixture(Path(self.temporary.name).resolve())
        self.freeze_sha = self.fixture.materialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evaluate(self) -> dict[str, object]:
        with mock.patch.object(qualification, "revalidate_base_release_candidate", return_value=(self.fixture.base_raw, self.fixture.base_sha)):
            return qwen.evaluate(self.fixture.freeze_path, self.fixture.evidence, self.fixture.raw_receipt_relative, expected_freeze_sha256=self.freeze_sha)

    def test_valid_native_capture_replays_exactly_and_report_parser_is_isomorphic(self) -> None:
        with mock.patch.object(qualification, "revalidate_base_release_candidate", return_value=(self.fixture.base_raw, self.fixture.base_sha)) as replay:
            report = qwen.evaluate(self.fixture.freeze_path, self.fixture.evidence, self.fixture.raw_receipt_relative, expected_freeze_sha256=self.freeze_sha)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["passed"])
        self.assertEqual(report["golden"], {"path": qwen.GOLDEN_RELATIVE_PATH, "sha256": qwen.GOLDEN_SHA256})
        self.assertEqual(report["wire"], {"path": qwen.WIRE_RELATIVE_PATH, "sha256": qwen.WIRE_SHA256})
        self.assertEqual([case["case_id"] for case in report["cases"]], [case.case_id for case in self.fixture.golden.cases])
        self.assertEqual(len(report["checks"]), len(qwen.CHECK_NAMES))
        parsed = qwen.validate_check_report(report)
        self.assertEqual(parsed.golden.path, qwen.GOLDEN_RELATIVE_PATH)
        self.assertEqual(parsed.wire.path, qwen.WIRE_RELATIVE_PATH)
        self.assertEqual(len(parsed.cases), qwen.GOLDEN_CASE_COUNT)
        replay.assert_called_once()

    def test_korean_empty_committed_piece_is_not_a_public_delta(self) -> None:
        korean = self.fixture.golden.cases[2]
        self.assertEqual(korean.expected_committed_output_tokens[-1].text, "")
        raw = (self.fixture.evidence / self.fixture.case_paths[korean.case_id]["stream"]["response.sse"]).read_bytes()
        frames = raw[:-2].split(b"\n\n")
        self.assertEqual(len(frames), len([piece for piece in korean.expected_committed_output_tokens if piece.text]) + 2)
        self.assertEqual(self.evaluate()["status"], "passed")

    def test_audit_text_piece_must_match_static_detokenization_golden(self) -> None:
        audit = copy.deepcopy(self.fixture.capture_document(2, "non-stream", "audit-record.json"))
        pieces = audit["committed_output_tokens"]
        assert isinstance(pieces, list) and isinstance(pieces[-1], dict)
        pieces[-1]["text"] = "unexpected"
        self.fixture.rewrite_capture_json(2, "non-stream", "audit-record.json", audit)
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["audit-committed-token-mismatch"])

    def test_public_sse_cannot_expose_a_token_id(self) -> None:
        case = self.fixture.golden.cases[0]
        raw = (self.fixture.evidence / self.fixture.case_paths[case.case_id]["stream"]["response.sse"]).read_bytes()
        frames = raw[:-2].split(b"\n\n")
        first = json.loads(frames[0][6:])
        first["token_id"] = case.expected_committed_output_tokens[0].token_id
        frames[0] = b"data: " + json.dumps(first, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.fixture.rewrite_capture_raw(0, "stream", "response.sse", b"\n\n".join(frames) + b"\n\n")
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["unknown-or-missing-field"])

    def test_sse_empty_delta_is_rejected_even_when_audit_has_an_empty_piece(self) -> None:
        case = self.fixture.golden.cases[2]
        raw = (self.fixture.evidence / self.fixture.case_paths[case.case_id]["stream"]["response.sse"]).read_bytes()
        frames = raw[:-2].split(b"\n\n")
        terminal = json.loads(frames[-2][6:])
        empty_delta = copy.deepcopy(terminal)
        choice = empty_delta["choices"][0]
        assert isinstance(choice, dict)
        choice["finish_reason"] = None
        frames.insert(-2, b"data: " + json.dumps(empty_delta, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        self.fixture.rewrite_capture_raw(2, "stream", "response.sse", b"\n\n".join(frames) + b"\n\n")
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["stream-event-count-mismatch"])

    def test_sse_terminal_must_be_empty_then_done(self) -> None:
        case = self.fixture.golden.cases[1]
        raw = (self.fixture.evidence / self.fixture.case_paths[case.case_id]["stream"]["response.sse"]).read_bytes()
        frames = raw[:-2].split(b"\n\n")
        terminal = json.loads(frames[-2][6:])
        choice = terminal["choices"][0]
        assert isinstance(choice, dict)
        choice["text"] = "late"
        frames[-2] = b"data: " + json.dumps(terminal, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.fixture.rewrite_capture_raw(1, "stream", "response.sse", b"\n\n".join(frames) + b"\n\n")
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["public-text-mismatch"])

    def test_modes_must_bind_distinct_server_request_ids(self) -> None:
        stream_audit = copy.deepcopy(self.fixture.capture_document(0, "stream", "audit-record.json"))
        non_audit = self.fixture.capture_document(0, "non-stream", "audit-record.json")
        stream_audit["server_request_id"] = non_audit["server_request_id"]
        self.fixture.rewrite_capture_json(0, "stream", "audit-record.json", stream_audit)
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["cross-mode-request-id-collision"])

    def test_public_non_stream_response_must_bind_its_audit_id(self) -> None:
        response = copy.deepcopy(self.fixture.capture_document(1, "non-stream", "response.json"))
        response["id"] = "cmpl-not-the-audit-id"
        self.fixture.rewrite_capture_json(1, "non-stream", "response.json", response)
        case = self.fixture.golden.cases[1]
        response_path = self.fixture.evidence / self.fixture.case_paths[case.case_id]["non-stream"]["response.json"]
        self.fixture.rewrite_capture_raw(
            1,
            "non-stream",
            "response.headers",
            self.fixture._headers(stream=False, body=response_path.read_bytes()),
        )
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["non-stream-audit-request-id-mismatch"])

    def test_raw_request_literal_prompt_must_equal_wire_contract(self) -> None:
        request = copy.deepcopy(self.fixture.capture_document(1, "non-stream", "request-body.json"))
        request["prompt"] = "not the exact Qwen prompt"
        self.fixture.rewrite_capture_json(1, "non-stream", "request-body.json", request)
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["public-request-prompt-mismatch"])

    def test_raw_header_content_length_is_checked_against_public_bytes(self) -> None:
        case = self.fixture.golden.cases[0]
        raw = (self.fixture.evidence / self.fixture.case_paths[case.case_id]["non-stream"]["response.headers"]).read_bytes()
        self.fixture.rewrite_capture_raw(0, "non-stream", "response.headers", raw.replace(b"Content-Length: ", b"Content-Length: 999"))
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["http-header-contract-mismatch"])

    def test_duplicate_raw_descriptor_is_rejected(self) -> None:
        case = self.fixture.manifest_case(0)
        non_stream = case["non_stream"]
        stream = case["stream"]
        assert isinstance(non_stream, dict) and isinstance(stream, dict)
        stream["audit_record"] = copy.deepcopy(non_stream["audit_record"])
        self.fixture.rewrite_manifest_and_receipt()
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["duplicate-evidence-path"])

    def test_raw_receipt_must_be_canonical_v2_json(self) -> None:
        raw = self.fixture.raw_receipt_path.read_bytes()
        self.fixture.raw_receipt_path.write_bytes(b" " + raw)
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["noncanonical-evidence"])

    def test_v1_receipt_version_is_not_silently_reinterpreted_as_v2(self) -> None:
        receipt = copy.deepcopy(self.fixture.receipt_document)
        receipt["schema_version"] = "riley.qwen-multistep-receipt.v1"
        self.fixture.write_canonical(self.fixture.raw_receipt_path, receipt)
        report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["unsupported-qwen-receipt-version"])

    def test_source_golden_digest_drift_blocks_the_gate(self) -> None:
        with mock.patch.object(qwen, "GOLDEN_SHA256", digest("different source golden")):
            report = self.evaluate()
        self.assertEqual(report["reason_codes"], ["qwen-golden-drift"])

    def test_check_report_parser_rejects_extra_case_field(self) -> None:
        report = self.evaluate()
        cases = report["cases"]
        assert isinstance(cases, list) and isinstance(cases[0], dict)
        cases[0]["token_id"] = 1
        with self.assertRaises(qualification.QualificationError) as error:
            qwen.validate_check_report(report)
        self.assertEqual(getattr(error.exception, "reason_code"), "unknown-or-missing-field")

    def test_cli_report_is_create_only(self) -> None:
        output = self.fixture.root / "qwen-check-v2.json"
        arguments = [
            "--freeze", str(self.fixture.freeze_path),
            "--expected-freeze-sha256", self.freeze_sha,
            "--evidence-root", str(self.fixture.evidence),
            "--receipt", self.fixture.raw_receipt_relative,
            "--report", str(output),
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(qualification, "revalidate_base_release_candidate", return_value=(self.fixture.base_raw, self.fixture.base_sha)):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(qwen.main(arguments), 0)
        original = output.read_bytes()
        with mock.patch.object(qualification, "revalidate_base_release_candidate", return_value=(self.fixture.base_raw, self.fixture.base_sha)):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(qwen.main(arguments), 2)
        self.assertEqual(output.read_bytes(), original)

    def test_v2_schema_declares_native_public_and_audit_descriptors(self) -> None:
        schema_path = Path(__file__).parents[2] / "benchmarks/release/candidates/qwen-multistep-receipt-v2.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertIn({"$ref": "#/$defs/qwenWire"}, schema["oneOf"])
        capture = schema["$defs"]["modeCapture"]
        self.assertEqual(set(capture["required"]), {"request_body", "response_headers", "response_body", "audit_record"})
        golden_case = schema["$defs"]["goldenCase"]
        self.assertIn("expected_committed_output_tokens", golden_case["required"])
        self.assertEqual(schema["$defs"]["checkReport"]["properties"]["schema_version"]["const"], qwen.CHECK_REPORT_VERSION)


if __name__ == "__main__":
    unittest.main()
