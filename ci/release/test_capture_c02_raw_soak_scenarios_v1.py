#!/usr/bin/env python3
"""CPU-only hostile tests for the serial C02 raw scenario producer."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import capture_c02_raw_soak_scenarios_v1 as capture


def canonical(value: object) -> bytes:
    return capture._canonical(value)


def proc_stat(pid: int, ticks: int) -> bytes:
    fields = ["S", *("0" for _ in range(18)), str(ticks), "0"]
    return f"{pid} (riley worker) {' '.join(fields)}\n".encode("ascii")


class FakeSocket:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.sent = b""

    def __enter__(self) -> "FakeSocket":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def settimeout(self, _seconds: float) -> None:
        return None

    def sendall(self, value: bytes) -> None:
        self.sent += value

    def recv(self, _size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""


class CaptureRawSoakScenariosTests(unittest.TestCase):
    candidate_id = "riley-0.1.0-rc3"
    profile = "max-performance-exact"
    server_pid = 123
    start_ticks = 456
    configuration_sha256 = "a" * 64

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.evidence = self.base / "evidence"
        self.evidence.mkdir(mode=0o700)
        os.chmod(self.evidence, 0o700)
        self.audit_dir = self.evidence / "source-audit"
        self.audit_dir.mkdir(mode=0o700)
        os.chmod(self.audit_dir, 0o700)
        self.contract_path = self.base / "serial-contract.json"
        self.contract_path.write_bytes(canonical(self._contract()))
        self.endpoint = capture.parse_endpoint("http://127.0.0.1:18080/v1/completions")
        self.listener = capture.Listener(
            tcp=(
                b"  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
                b"   0: 0100007F:46A0 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 42 1\n"
            ),
            inode=42,
            sockets=(42, 99),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _contract(self, *, scenario_id: str = "smoke", stream: bool = False) -> dict:
        return {
            "schema_version": capture.CONTRACT_VERSION,
            "candidate_id": self.candidate_id,
            "configuration_profile": self.profile,
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "completion_request": {
                        "model": "fixture-model",
                        "prompt": "hello",
                        "max_tokens": 2,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": 7,
                        "stream": stream,
                    },
                }
            ],
        }

    def _write_audit(self, request_id: str = "cmpl-safe") -> tuple[bytes, bytes]:
        record = canonical(
            {
                "schema_version": capture.AUDIT_VERSION,
                "candidate_id": self.candidate_id,
                "runtime_identity": {
                    "configuration_profile": self.profile,
                    "configuration_sha256": self.configuration_sha256,
                },
                "process_identity": {"pid": self.server_pid, "start_ticks": self.start_ticks},
                "server_request_id": request_id,
                "delivery_mode": "non-stream",
                "prompt_token_ids": [1],
                "committed_output_tokens": [{"emitted_text_delta": "x", "token_id": 2}],
                "sampling_selections": [
                    {
                        "committed": True,
                        "configured_backend": "gpu-greedy",
                        "ineligibility_reason": None,
                        "iteration_id": 1,
                        "selected_backend": "gpu-greedy",
                    }
                ],
                "finish_reason": "length",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        name = f"{request_id}.json"
        marker = canonical(
            {
                "schema_version": capture.AUDIT_COMPLETION_VERSION,
                "artifact_filename": name,
                "artifact_sha256": hashlib.sha256(record).hexdigest(),
            }
        )
        (self.audit_dir / name).write_bytes(record)
        (self.audit_dir / f"{name}.complete").write_bytes(marker)
        return record, marker

    def _request(self) -> capture.CaptureRequest:
        return capture.CaptureRequest(
            endpoint=self.endpoint,
            server_pid=self.server_pid,
            candidate_id=self.candidate_id,
            configuration_profile=self.profile,
            configuration_sha256=self.configuration_sha256,
            evidence_root=self.evidence,
            capture_name="serial-capture",
            audit_dir_name="source-audit",
            scenario_contract=self.contract_path,
            audit_wait_seconds=0.2,
        )

    def test_capture_preserves_exact_http_and_source_audit_without_gpu_or_service_calls(self) -> None:
        record, marker = self._write_audit()
        response = canonical({"id": "cmpl-safe", "object": "text_completion"})
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(response)}\r\n"
            "Content-Type: application/json; charset=utf-8\r\nConnection: close\r\n\r\n"
        ).encode("ascii")
        fake = FakeSocket([head + response, b""])
        with mock.patch.object(capture.socket, "create_connection", return_value=fake), mock.patch.object(
            capture, "_server_stat", return_value=(proc_stat(self.server_pid, self.start_ticks), self.start_ticks)
        ), mock.patch.object(capture, "_bound_listener", return_value=self.listener):
            session = capture.capture_raw_scenarios(self._request(), repository_root=self.repository)
        self.assertEqual(session["schema_version"], capture.CAPTURE_VERSION)
        self.assertEqual(session["capture_status"], "captured")
        self.assertEqual(session["qualification_status"], "not-run")
        self.assertEqual(
            session["target"],
            {
                "server_pid": self.server_pid,
                "server_start_ticks": self.start_ticks,
                "listener_port": 18080,
                "listener_inode": 42,
            },
        )
        self.assertEqual(session["scenarios"][0]["runtime_event_log"]["path"], "source-audit/cmpl-safe.json")
        capture_dir = self.evidence / "serial-capture"
        self.assertTrue((capture_dir / "session.json").is_file())
        self.assertFalse((capture_dir / capture.INCOMPLETE_MARKER_NAME).exists())
        self.assertEqual((capture_dir / "raw/000000.response-body.json").read_bytes(), response)
        self.assertEqual((self.audit_dir / "cmpl-safe.json").read_bytes(), record)
        self.assertEqual((self.audit_dir / "cmpl-safe.json.complete").read_bytes(), marker)
        self.assertIn(b"POST /v1/completions HTTP/1.1\r\n", fake.sent)
        self.assertIn(b"Connection: close\r\n\r\n", fake.sent)
        self.assertEqual(stat.S_IMODE(capture_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((capture_dir / "session.json").stat().st_mode), 0o600)

    def test_contract_rejects_fallback_stream_and_binding_drift(self) -> None:
        fallback = canonical(self._contract(scenario_id="exact-backend-fallback"))
        streaming = canonical(self._contract(stream=True))
        for raw in (fallback, streaming):
            with self.subTest(raw=raw):
                with self.assertRaises(capture.RawScenarioCaptureError):
                    capture.validate_contract(raw, candidate_id=self.candidate_id, configuration_profile=self.profile)
        with self.assertRaises(capture.RawScenarioCaptureError):
            capture.validate_contract(canonical(self._contract()), candidate_id="riley-0.1.0-rc4", configuration_profile=self.profile)

    def test_response_capture_rejects_trailing_or_truncated_bytes(self) -> None:
        body = canonical({"id": "cmpl-safe"})
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        for chunks in ([head + body + b"x", b""], [head + body, b"x"], [head + body[:1], b""]):
            with self.subTest(chunks=chunks):
                with mock.patch.object(capture.socket, "create_connection", return_value=FakeSocket(chunks)):
                    with self.assertRaises(capture.RawScenarioCaptureError):
                        capture._capture_completion(self.endpoint, body)

    def test_response_headers_reject_obs_fold_invalid_media_type_and_duplicates(self) -> None:
        for head in (
            b"HTTP/1.1 200 OK\r\n Content-Length: 2\r\nContent-Type: application/json\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/jsonp\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length : 2\r\nContent-Type: application/json\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length:\x0b2\r\nContent-Type: application/json\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\ncontent-length: 2\r\nContent-Type: application/json\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n",
        ):
            with self.subTest(head=head):
                with self.assertRaises(capture.RawScenarioCaptureError):
                    capture._response_length(head)

    def test_proc_stat_rejects_an_internal_pid_mismatch(self) -> None:
        with mock.patch.object(capture, "_read_absolute_regular", return_value=proc_stat(999, self.start_ticks)):
            with self.assertRaises(capture.RawScenarioCaptureError):
                capture._server_stat(self.server_pid)

    def test_failed_directory_sync_restores_nonterminal_marker(self) -> None:
        capture_dir = self.evidence / "marker-test"
        capture_dir.mkdir(mode=0o700)
        os.chmod(capture_dir, 0o700)
        descriptor = os.open(capture_dir, capture._open_flags(directory=True))
        marker = canonical({"capture_name": "marker-test", "schema_version": capture.INCOMPLETE_MARKER_VERSION})
        try:
            capture._write_new(descriptor, capture.INCOMPLETE_MARKER_NAME, marker)
            with mock.patch.object(capture, "_fsync", side_effect=capture.RawScenarioCaptureError("forced fsync failure")):
                with self.assertRaises(capture.RawScenarioCaptureError):
                    capture._remove_incomplete_marker(descriptor, marker)
        finally:
            os.close(descriptor)
        self.assertTrue((capture_dir / capture.INCOMPLETE_MARKER_NAME).is_file())

    def test_listener_drift_during_completion_keeps_capture_nonterminal(self) -> None:
        self._write_audit()
        response = canonical({"id": "cmpl-safe"})
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(response)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        drifted = capture.Listener(tcp=self.listener.tcp.replace(b" 42 1", b" 43 1"), inode=43, sockets=(43,))
        with mock.patch.object(capture.socket, "create_connection", return_value=FakeSocket([head + response, b""])), mock.patch.object(
            capture, "_server_stat", return_value=(proc_stat(self.server_pid, self.start_ticks), self.start_ticks)
        ), mock.patch.object(capture, "_bound_listener", side_effect=[self.listener, drifted]):
            with self.assertRaises(capture.RawScenarioCaptureError):
                capture.capture_raw_scenarios(self._request(), repository_root=self.repository)
        self.assertTrue((self.evidence / "serial-capture" / capture.INCOMPLETE_MARKER_NAME).is_file())

    def test_source_audit_mismatch_leaves_the_nonterminal_marker(self) -> None:
        self._write_audit()
        response = canonical({"id": "cmpl-safe"})
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(response)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        bad_record = json.loads((self.audit_dir / "cmpl-safe.json").read_text(encoding="utf-8"))
        bad_record["candidate_id"] = "riley-0.1.0-rc4"
        raw_bad = canonical(bad_record)
        (self.audit_dir / "cmpl-safe.json").write_bytes(raw_bad)
        (self.audit_dir / "cmpl-safe.json.complete").write_bytes(
            canonical(
                {
                    "schema_version": capture.AUDIT_COMPLETION_VERSION,
                    "artifact_filename": "cmpl-safe.json",
                    "artifact_sha256": hashlib.sha256(raw_bad).hexdigest(),
                }
            )
        )
        with mock.patch.object(capture.socket, "create_connection", return_value=FakeSocket([head + response, b""])), mock.patch.object(
            capture, "_server_stat", return_value=(proc_stat(self.server_pid, self.start_ticks), self.start_ticks)
        ), mock.patch.object(capture, "_bound_listener", return_value=self.listener):
            with self.assertRaises(capture.RawScenarioCaptureError):
                capture.capture_raw_scenarios(self._request(), repository_root=self.repository)
        self.assertTrue((self.evidence / "serial-capture" / capture.INCOMPLETE_MARKER_NAME).is_file())

    def test_rejects_nonloopback_and_source_tree_evidence_and_missing_open_flag(self) -> None:
        for endpoint in (
            "http://localhost:18080/v1/completions",
            "http://127.0.0.1:80/v1/completions",
            "http://127.0.0.1:65536/v1/completions",
            "http://127.0.0.1:18080/v1/config",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(capture.RawScenarioCaptureError):
                    capture.parse_endpoint(endpoint)
        inside = self._request()
        inside = capture.CaptureRequest(**{**inside.__dict__, "evidence_root": self.repository})
        with self.assertRaises(capture.RawScenarioCaptureError):
            capture.capture_raw_scenarios(inside, repository_root=self.repository)
        with mock.patch.object(capture.os, "O_NOFOLLOW", 0):
            with self.assertRaises(capture.RawScenarioCaptureError):
                capture._open_flags(directory=True)

    def test_rejects_zero_configuration_sha256(self) -> None:
        with self.assertRaises(capture.RawScenarioCaptureError):
            capture._sha256("0" * 64, "fixture configuration SHA-256")

    def test_cli_help_is_available_without_a_capture(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                capture.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_published_contract_schema_matches_the_closed_runtime_contract(self) -> None:
        candidate_root = Path(__file__).resolve().parents[2] / "benchmarks/release/candidates"
        schema = json.loads(
            (candidate_root / "c02-raw-soak-runner-contract-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["properties"]["schema_version"]["const"], capture.CONTRACT_VERSION)
        scenario_id = schema["$defs"]["scenario"]["properties"]["scenario_id"]
        self.assertIn("exact-backend-fallback", scenario_id["pattern"])
        self.assertEqual(
            schema["$defs"]["completionRequest"]["properties"]["stream"]["const"],
            False,
        )
        capture_schema = json.loads(
            (candidate_root / "c02-raw-scenario-capture-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            capture_schema["properties"]["schema_version"]["const"],
            capture.CAPTURE_VERSION,
        )
        self.assertEqual(
            set(capture_schema["required"]),
            {
                "schema_version",
                "capture_status",
                "qualification_status",
                "endpoint",
                "contract",
                "runtime_identity",
                "target",
                "scenarios",
            },
        )
        self.assertEqual(
            capture_schema["$defs"]["sha256"]["not"]["const"],
            "0" * 64,
        )
        endpoint_pattern = capture_schema["properties"]["endpoint"]["pattern"]
        for endpoint in (
            "http://127.0.0.1:1024/v1/completions",
            "http://127.0.0.1:65535/v1/completions",
        ):
            self.assertIsNotNone(re.fullmatch(endpoint_pattern, endpoint))
        for endpoint in (
            "http://127.0.0.1:1023/v1/completions",
            "http://127.0.0.1:65536/v1/completions",
        ):
            self.assertIsNone(re.fullmatch(endpoint_pattern, endpoint))
        audit_schema = json.loads(
            (candidate_root / "c02-generation-audit-v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        completion_schema = json.loads(
            (candidate_root / "c02-generation-audit-completion-v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(audit_schema["$defs"]["sha256"]["not"]["const"], "0" * 64)
        self.assertEqual(
            completion_schema["properties"]["artifact_sha256"]["not"]["const"],
            "0" * 64,
        )


if __name__ == "__main__":
    unittest.main()
