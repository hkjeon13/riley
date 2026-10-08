#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3/RC2 raw phase producer."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import capture_rc3_rollback_phase_v1 as capture  # noqa: E402


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def proc_stat(pid: int = 123, ticks: int = 456) -> bytes:
    fields = ["S", *("0" for _ in range(18)), str(ticks)]
    return f"{pid} (riley worker) {' '.join(fields)}\n".encode("ascii")


def proc_status(pid: int = 123) -> bytes:
    return f"Name:\triley\nPid:\t{pid}\n".encode("ascii")


def tcp_table(port: int = 18080, inode: int = 42, address: str = "0100007F") -> bytes:
    return (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        f"   0: {address}:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 {inode} 1\n"
    ).encode("ascii")


def bound_listener(port: int = 18080, inode: int = 42) -> object:
    return capture.c02.BoundListener(
        proc_net_tcp=tcp_table(port, inode), socket_inode=inode, server_socket_inodes=(inode, 99)
    )


def response(body: bytes, content_type: str = "application/json") -> tuple[bytes, bytes]:
    return (
        (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("ascii"),
        body,
    )


class RollbackPhaseCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "evidence"
        self.root.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.repository_patch = mock.patch.object(
            capture, "_repository_root", return_value=self.repository
        )
        self.repository_patch.start()
        self.endpoint = capture.parse_endpoint("http://127.0.0.1:18080")
        self.request = capture.CaptureRequest(
            endpoint=self.endpoint,
            server_pid=123,
            gpu_index=0,
            evidence_root=self.root,
            capture_name="reconstructed-phase",
            generation_body=canonical(
                {
                    "model": "fixture",
                    "prompt": "hello",
                    "max_tokens": 1,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 0,
                    "stream": False,
                }
            ),
        )
        self.target = capture.TargetIdentity(123, 456, 18080, 42, 0, "GPU-deadbeef")

    def tearDown(self) -> None:
        self.repository_patch.stop()
        self.temporary.cleanup()

    def _runtime_patches(self) -> list[object]:
        sockets = canonical(
            {
                "schema_version": capture.c02.SOCKET_SNAPSHOT_VERSION,
                "server_pid": 123,
                "socket_inodes": [42, 99],
            }
        )
        health_head, health_body = response(b"ready\n", "text/plain")
        generation_head, generation_body = response(canonical({"id": "cmpl-fixture"}))
        assert self.request.generation_body is not None
        return [
            mock.patch.object(capture, "_preflight_target", return_value=self.target),
            mock.patch.object(
                capture.c02,
                "_capture_server_stat",
                side_effect=[(proc_stat(), 456), (proc_stat(), 456)],
            ),
            mock.patch.object(
                capture.c02,
                "_capture_bound_listener",
                side_effect=[bound_listener(), bound_listener()],
            ),
            mock.patch.object(capture.c02, "_socket_snapshot_raw", side_effect=[sockets, sockets]),
            mock.patch.object(
                capture.c02,
                "_capture_gpu",
                return_value=(b"0, GPU-deadbeef\n", b"123, 0\n", "GPU-deadbeef"),
            ),
            mock.patch.object(capture.c02, "_capture_server_status", return_value=proc_status()),
            mock.patch.object(
                capture,
                "_capture_exchange",
                side_effect=[
                    (
                        capture._request_bytes("GET", self.endpoint, "/readyz", b""),
                        health_head,
                        health_body,
                    ),
                    (
                        capture._request_bytes(
                            "POST",
                            self.endpoint,
                            "/v1/completions",
                            self.request.generation_body,
                        ),
                        generation_head,
                        generation_body,
                    ),
                ],
            ),
        ]

    def _capture_completed(self, request: capture.CaptureRequest | None = None) -> dict[str, object]:
        patches = self._runtime_patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            return capture.capture_phase(request or self.request)

    def _replay(self, capture_name: str) -> capture.ReplayedPhaseCapture:
        root_fd = capture.common.open_private_evidence_directory(self.root, "fixture evidence root")
        try:
            return capture.replay_rc3_rollback_phase_v1_fd(root_fd, capture_name)
        finally:
            os.close(root_fd)

    def _replace_session(self, capture_name: str, document: object) -> None:
        session_path = self.root / capture_name / "session.json"
        session_path.write_bytes(canonical(document))
        os.chmod(session_path, 0o600)

    def test_endpoint_is_literal_loopback_base_only(self) -> None:
        self.assertEqual(self.endpoint.url, "http://127.0.0.1:18080")
        for value in (
            "https://127.0.0.1:18080",
            "http://localhost:18080",
            "http://127.0.0.1:18080/readyz",
            "http://127.0.0.1:18080?q=1",
            "http://127.0.0.1:80",
            "http://127.0.0.1",
        ):
            with self.subTest(value=value), self.assertRaises(capture.RollbackPhaseCaptureError):
                capture.parse_endpoint(value)

    def test_http_framing_refuses_chunking_duplicate_length_and_non_200(self) -> None:
        good, _ = response(b"{}")
        self.assertEqual(capture._response_length(good, "fixture", require_json=True), 2)
        for head in (
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
            b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\ncontent-length: 2\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
        ):
            with self.subTest(head=head), self.assertRaises(capture.RollbackPhaseCaptureError):
                capture._response_length(head, "fixture", require_json=False)

    def test_reused_no_follow_private_root_open_fails_closed(self) -> None:
        with mock.patch.object(capture.c02.os, "O_NOFOLLOW", None):
            with self.assertRaises(capture.RollbackPhaseCaptureError):
                capture._open_capture_directories(self.request, capture._marker())

    def test_capture_writes_closed_leaf_inventory_and_derives_target(self) -> None:
        session = self._capture_completed()
        directory = self.root / self.request.capture_name
        expected = {
            "pre-stat",
            "post-stat",
            "pre-tcp",
            "post-tcp",
            "pre-fd-sockets.json",
            "post-fd-sockets.json",
            "status",
            "gpu-selection.csv",
            "gpu-compute-apps.csv",
            "health-request.http",
            "health-response-head.http",
            "health-response-body.bin",
            "generation-request.http",
            "generation-response-head.http",
            "generation-response-body.bin",
        }
        self.assertEqual({item.name for item in (directory / "raw").iterdir()}, expected)
        self.assertTrue((directory / "session.json").is_file())
        self.assertFalse((directory / capture.INCOMPLETE_MARKER_NAME).exists())
        self.assertEqual(session["schema_version"], capture.SESSION_VERSION)
        self.assertEqual(session["qualification_status"], "not-run")
        self.assertEqual(session["target"], self.target.as_json())
        self.assertIsNotNone(session["generation"])
        for group in ("process_evidence", "health", "generation"):
            assert session[group] is not None
            for descriptor in session[group].values():
                raw = (self.root / descriptor["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), descriptor["sha256"])
                self.assertEqual(len(raw), descriptor["byte_length"])
        for private_directory in (self.root, directory, directory / "raw"):
            self.assertEqual(stat.S_IMODE(private_directory.stat().st_mode), 0o700)

    def test_held_fd_replay_returns_closed_terminal_phase(self) -> None:
        session = self._capture_completed()

        replayed = self._replay(self.request.capture_name)

        self.assertEqual(replayed.capture_name, self.request.capture_name)
        self.assertEqual(replayed.endpoint, self.endpoint)
        self.assertEqual(replayed.target, self.target)
        self.assertEqual(
            replayed.process_evidence["pre_stat"].as_json(),
            session["process_evidence"]["pre_stat"],  # type: ignore[index]
        )
        self.assertEqual(
            replayed.health["request"].as_json(),
            session["health"]["request"],  # type: ignore[index]
        )
        assert replayed.generation is not None
        self.assertEqual(
            replayed.generation["response_body"].as_json(),
            session["generation"]["response_body"],  # type: ignore[index]
        )

    def test_held_fd_replay_derives_target_from_the_consumed_raw_snapshot(self) -> None:
        self._capture_completed()

        with mock.patch.object(
            capture.rollback_v3,
            "derive_phase_target_from_raw_evidence_fd",
            side_effect=AssertionError("root-relative raw reopen must not occur"),
        ):
            replayed = self._replay(self.request.capture_name)

        self.assertEqual(replayed.target, self.target)

    def test_held_fd_replay_allows_candidate_host_phase_without_generation(self) -> None:
        request = capture.CaptureRequest(
            **{**self.request.__dict__, "capture_name": "candidate-host-replay", "generation_body": None}
        )
        self._capture_completed(request)

        replayed = self._replay(request.capture_name)

        self.assertEqual(replayed.target, self.target)
        self.assertIsNone(replayed.generation)

    def test_held_fd_replay_rejects_incomplete_or_extra_inventory(self) -> None:
        incomplete_request = capture.CaptureRequest(
            **{**self.request.__dict__, "capture_name": "incomplete-replay"}
        )
        self._capture_completed(incomplete_request)
        marker = self.root / incomplete_request.capture_name / capture.INCOMPLETE_MARKER_NAME
        marker.write_bytes(capture._marker())
        os.chmod(marker, 0o600)
        with self.assertRaises(capture.RollbackPhaseCaptureError):
            self._replay(incomplete_request.capture_name)

        extra_request = capture.CaptureRequest(
            **{**self.request.__dict__, "capture_name": "extra-raw-replay"}
        )
        self._capture_completed(extra_request)
        extra = self.root / extra_request.capture_name / "raw" / "unexpected"
        extra.write_bytes(b"unexpected\n")
        os.chmod(extra, 0o600)
        with self.assertRaises(capture.RollbackPhaseCaptureError):
            self._replay(extra_request.capture_name)

    def test_held_fd_replay_rejects_declared_target_or_canonical_request_drift(self) -> None:
        self._capture_completed()
        session_path = self.root / self.request.capture_name / "session.json"
        target_drift = json.loads(session_path.read_text(encoding="utf-8"))
        target_drift["target"]["listener_inode"] = 43
        self._replace_session(self.request.capture_name, target_drift)
        with self.assertRaises(capture.RollbackPhaseCaptureError) as target_error:
            self._replay(self.request.capture_name)
        self.assertEqual(getattr(target_error.exception, "reason_code", None), "phase-target-raw-mismatch")

        request_drift = capture.CaptureRequest(
            **{**self.request.__dict__, "capture_name": "request-drift"}
        )
        session = self._capture_completed(request_drift)
        raw_path = self.root / request_drift.capture_name / "raw" / "health-request.http"
        raw = capture._request_bytes("GET", self.endpoint, "/wrong", b"")
        raw_path.write_bytes(raw)
        os.chmod(raw_path, 0o600)
        request_document = json.loads(canonical(session))
        request_document["health"]["request"] = capture._descriptor(
            f"{request_drift.capture_name}/raw/health-request.http", raw
        )
        self._replace_session(request_drift.capture_name, request_document)
        with self.assertRaises(capture.RollbackPhaseCaptureError) as request_error:
            self._replay(request_drift.capture_name)
        self.assertEqual(getattr(request_error.exception, "reason_code", None), "invalid-http-request")

    def test_candidate_host_snapshot_has_no_extra_unaudited_generation(self) -> None:
        request = capture.CaptureRequest(
            **{**self.request.__dict__, "capture_name": "candidate-host", "generation_body": None}
        )
        sockets = canonical(
            {
                "schema_version": capture.c02.SOCKET_SNAPSHOT_VERSION,
                "server_pid": 123,
                "socket_inodes": [42, 99],
            }
        )
        health_head, health_body = response(b"ready\n", "text/plain")
        with mock.patch.object(capture, "_preflight_target", return_value=self.target), mock.patch.object(
            capture.c02,
            "_capture_server_stat",
            side_effect=[(proc_stat(), 456), (proc_stat(), 456)],
        ), mock.patch.object(
            capture.c02, "_capture_bound_listener", side_effect=[bound_listener(), bound_listener()]
        ), mock.patch.object(capture.c02, "_socket_snapshot_raw", return_value=sockets), mock.patch.object(
            capture.c02,
            "_capture_gpu",
            return_value=(b"0, GPU-deadbeef\n", b"123, 0\n", "GPU-deadbeef"),
        ), mock.patch.object(capture.c02, "_capture_server_status", return_value=proc_status()), mock.patch.object(
            capture,
            "_capture_exchange",
            return_value=(b"GET /readyz HTTP/1.1\r\n\r\n", health_head, health_body),
        ) as exchanges:
            session = capture.capture_phase(request)
        self.assertEqual(exchanges.call_count, 1)
        self.assertIsNone(session["generation"])
        self.assertFalse((self.root / request.capture_name / "raw" / "generation-request.http").exists())

    def test_direct_api_rejects_noncanonical_or_arbitrary_generation_bytes(self) -> None:
        bad_request = capture.CaptureRequest(
            **{**self.request.__dict__, "generation_body": b'{"arbitrary":true}'}
        )
        with self.assertRaises(capture.RollbackPhaseCaptureError):
            capture.capture_phase(bad_request)
        noncanonical_request = capture.CaptureRequest(
            **{**self.request.__dict__, "generation_body": self.request.generation_body + b"\n"}
        )
        with self.assertRaises(capture.RollbackPhaseCaptureError):
            capture.capture_phase(noncanonical_request)

    def test_failure_keeps_marker_and_never_publishes_session(self) -> None:
        with mock.patch.object(capture, "_preflight_target", return_value=self.target), mock.patch.object(
            capture.c02,
            "_capture_server_stat",
            return_value=(proc_stat(), 456),
        ), mock.patch.object(capture.c02, "_capture_bound_listener", return_value=bound_listener()), mock.patch.object(
            capture.c02,
            "_socket_snapshot_raw",
            return_value=canonical(
                {"schema_version": capture.c02.SOCKET_SNAPSHOT_VERSION, "server_pid": 123, "socket_inodes": [42, 99]}
            ),
        ), mock.patch.object(capture, "_capture_exchange", side_effect=capture.RollbackPhaseCaptureError("fixture failure")):
            with self.assertRaisesRegex(capture.RollbackPhaseCaptureError, "fixture failure"):
                capture.capture_phase(self.request)
        directory = self.root / self.request.capture_name
        self.assertTrue((directory / capture.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((directory / "session.json").exists())

    def test_existing_name_root_symlink_and_marker_sync_ambiguity_fail_closed(self) -> None:
        marker = capture._marker()
        directories = capture._open_capture_directories(self.request, marker)
        capture.c02._close_quietly(directories.raw_fd)
        capture.c02._close_quietly(directories.capture_fd)
        capture.c02._close_quietly(directories.root_fd)
        with self.assertRaisesRegex(capture.RollbackPhaseCaptureError, "already exists"):
            capture._open_capture_directories(self.request, marker)
        link = self.base / "evidence-link"
        os.symlink(self.root, link)
        link_request = capture.CaptureRequest(
            **{**self.request.__dict__, "evidence_root": link, "capture_name": "linked"}
        )
        with self.assertRaises(capture.RollbackPhaseCaptureError):
            capture._open_capture_directories(link_request, marker)

    def test_cli_help_and_source_remain_raw_only(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", "-S", str(Path(capture.__file__)), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--generation-request", completed.stdout)
        wrapper = Path(capture.__file__).with_name("run_capture_rc3_rollback_phase_v1.sh")
        wrapped = subprocess.run(
            ["bash", str(wrapper), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(wrapped.returncode, 0, wrapped.stderr)
        self.assertIn("--capture-name", wrapped.stdout)
        wrapper_source = wrapper.read_text(encoding="utf-8")
        for required in ("/usr/bin/env -i", "-B -I -S", "os.lstat(script)", "sys.path.insert"):
            self.assertIn(required, wrapper_source)
        source = Path(capture.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "docker ",
            "podman ",
            "ssh ",
            "systemctl ",
            "renameat2",
            "subprocess.run",
            "qualification_status\": \"passed",
        ):
            self.assertNotIn(forbidden, source)
        replay_source = source[
            source.index("def replay_rc3_rollback_phase_v1_fd(") : source.index(
                "\ndef capture_phase(", source.index("def replay_rc3_rollback_phase_v1_fd(")
            )
        ]
        for forbidden in (
            "argparse",
            "socket.",
            "subprocess",
            "docker",
            "ssh ",
            "_capture_exchange",
            "_preflight_target",
            "derive_phase_target_from_raw_evidence_fd",
        ):
            self.assertNotIn(forbidden, replay_source)
        self.assertIn("derive_phase_target_from_raw_bytes", replay_source)


if __name__ == "__main__":
    unittest.main()
