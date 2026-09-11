#!/usr/bin/env python3
"""CPU-only tests for the self-contained C02 /v1/config bridge producer."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import capture_c02_config_endpoint_observation_v1 as capture


def canonical(value: object) -> bytes:
    return capture._canonical(value)


def proc_stat(pid: int = 123, ticks: int = 456) -> bytes:
    fields = ["S", *("0" for _ in range(18)), str(ticks)]
    return f"{pid} (riley worker) {' '.join(fields)}\n".encode("ascii")


def proc_status(pid: int = 123) -> bytes:
    return f"Name:\triley\nPid:\t{pid}\n".encode("ascii")


class CaptureConfigEndpointObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.evidence = self.base / "evidence"
        self.evidence.mkdir(mode=0o700)
        os.chmod(self.evidence, 0o700)
        self.endpoint = capture.parse_endpoint("http://127.0.0.1:18080/v1/config")
        self.request = capture.CaptureRequest(
            endpoint=self.endpoint,
            server_pid=123,
            gpu_index=0,
            evidence_root=self.evidence,
            capture_name="config-bridge",
        )
        self.listener = capture.Listener(
            tcp=(
                b"  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
                b"   0: 0100007F:46A0 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 42 1\n"
            ),
            inode=42,
            sockets=(42, 99),
        )
        self.selection = b"0, GPU-deadbeef\n"
        self.apps = b"123, 8\n"
        self.body = canonical({"literal": "한국어", "runtime": "fixture"})
        self.http_request = (
            b"GET /v1/config HTTP/1.1\r\nHost: 127.0.0.1:18080\r\n"
            b"Accept: application/json\r\nConnection: close\r\n\r\n"
        )
        self.http_head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(self.body)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _patch_capture(self) -> mock._patch:
        return mock.patch.multiple(
            capture,
            _capture_server_stat=mock.Mock(return_value=(proc_stat(), 456)),
            _bound_listener=mock.Mock(return_value=self.listener),
            _capture_gpu=mock.Mock(return_value=(self.selection, self.apps, "GPU-deadbeef")),
            _capture_endpoint=mock.Mock(return_value=(self.http_request, self.http_head, self.body)),
            _capture_status=mock.Mock(return_value=proc_status()),
        )

    def test_endpoint_is_literal_loopback_config_only(self) -> None:
        self.assertEqual(self.endpoint.port, 18080)
        for value in (
            "https://127.0.0.1:18080/v1/config",
            "http://localhost:18080/v1/config",
            "http://127.0.0.1:18080/v1/c02/metrics",
            "http://127.0.0.1:80/v1/config",
            "http://127.0.0.1:18080/v1/config?x=1",
        ):
            with self.subTest(value=value):
                with self.assertRaises(capture.ConfigEndpointObservationError):
                    capture.parse_endpoint(value)

    def test_capture_writes_raw_body_and_same_process_bridge_without_gpu_calls(self) -> None:
        with self._patch_capture():
            session = capture.capture_config_endpoint_observation(
                self.request, repository_root=self.repository
            )
        self.assertEqual(session["schema_version"], capture.BRIDGE_VERSION)
        self.assertEqual(session["capture_status"], "captured")
        self.assertEqual(session["qualification_status"], "not-run")
        self.assertEqual(session["target"]["listener_inode"], 42)
        self.assertEqual(session["endpoint"]["body_sha256"], hashlib.sha256(self.body).hexdigest())
        capture_dir = self.evidence / "config-bridge"
        self.assertTrue((capture_dir / "session.json").is_file())
        self.assertFalse((capture_dir / capture.INCOMPLETE_MARKER_NAME).exists())
        self.assertEqual((capture_dir / "raw/config-endpoint.json").read_bytes(), self.body)
        self.assertEqual(stat.S_IMODE((capture_dir / "session.json").stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(capture_dir.stat().st_mode), 0o700)

    def test_response_parser_rejects_transfer_encoding_and_duplicate_length(self) -> None:
        transfer = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
        duplicate = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\n"
        for head in (transfer, duplicate):
            with self.subTest(head=head):
                with self.assertRaises(capture.ConfigEndpointObservationError):
                    capture._response_length(head)

    def test_socket_capture_rejects_extra_or_truncated_body_bytes(self) -> None:
        class FakeSocket:
            def __init__(self, chunks: list[bytes]) -> None:
                self.chunks = list(chunks)

            def __enter__(self) -> "FakeSocket":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def settimeout(self, _value: float) -> None:
                return None

            def sendall(self, _value: bytes) -> None:
                return None

            def recv(self, _size: int) -> bytes:
                return self.chunks.pop(0) if self.chunks else b""

        head = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\n"
        for chunks in ([head + b"{}x", b""], [head + b"{}", b"x", b""], [head + b"{", b""]):
            with self.subTest(chunks=chunks):
                with mock.patch.object(capture.socket, "create_connection", return_value=FakeSocket(chunks)):
                    with self.assertRaises(capture.ConfigEndpointObservationError):
                        capture._capture_endpoint(self.endpoint)

    def test_socket_capture_accepts_an_exact_bound_body_coalesced_with_its_head(self) -> None:
        class FakeSocket:
            def __init__(self, chunks: list[bytes]) -> None:
                self.chunks = list(chunks)

            def __enter__(self) -> "FakeSocket":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def settimeout(self, _value: float) -> None:
                return None

            def sendall(self, _value: bytes) -> None:
                return None

            def recv(self, _size: int) -> bytes:
                return self.chunks.pop(0) if self.chunks else b""

        body = b'{"a":1}'
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        with mock.patch.object(capture, "MAX_HTTP_BYTES", len(body)):
            with mock.patch.object(
                capture.socket,
                "create_connection",
                return_value=FakeSocket([head + body, b""]),
            ):
                _request, captured_head, captured_body = capture._capture_endpoint(self.endpoint)
        self.assertEqual(captured_head, head)
        self.assertEqual(captured_body, body)

    def test_refuses_source_tree_and_missing_open_flags(self) -> None:
        inside = capture.CaptureRequest(
            endpoint=self.endpoint, server_pid=123, gpu_index=0,
            evidence_root=self.repository, capture_name="inside",
        )
        with self._patch_capture():
            with self.assertRaises(capture.ConfigEndpointObservationError):
                capture.capture_config_endpoint_observation(inside, repository_root=self.repository)
        with mock.patch.object(capture.os, "O_NOFOLLOW", 0):
            with self.assertRaises(capture.ConfigEndpointObservationError):
                capture._open_flags(directory=True)

    def test_isolated_cli_help_loads_without_a_capture(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                capture.main(["--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
