#!/usr/bin/env python3
"""CPU-only tests for the self-contained C02 v2 raw observation producer."""

from __future__ import annotations

import contextlib
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

import capture_c02_observations_v2 as capture  # noqa: E402
import effective_runtime_config_contract as runtime_config  # noqa: E402


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def metrics_raw(*, failed: int = 0, empty_kv: bool = False) -> bytes:
    return canonical(
        {
            "schema_version": capture.METRICS_VERSION,
            "request_states": {
                "active": 1,
                "pending_requests": 2,
                "completed": 3,
                "failed": failed,
                "cancelled": 4,
                "capacity_rejections": 5,
            },
            "kv_blocks": {"free": 0 if empty_kv else 60, "reserved": 20, "active": 20},
            "allocation": {
                "device_live_count": 3,
                "device_live_bytes": 768,
                "pinned_live_count": 2,
                "pinned_live_bytes": 256,
            },
            "quiescence": {
                "completion_outbox": 0,
                "outstanding_iterations": 0,
                "riley_owned_live_allocations": 5,
                "worker_accepting": False,
                "scheduler_accepting": False,
            },
        }
    )


def proc_stat(pid: int = 123, start_ticks: int = 456) -> bytes:
    fields = ["S", *("0" for _ in range(18)), str(start_ticks)]
    return f"{pid} (riley worker) {' '.join(fields)}\n".encode("ascii")


def proc_status(pid: int = 123) -> bytes:
    return f"Name:\triley\nPid:\t{pid}\n".encode("ascii")


def tcp_listener_table(*, port: int, inode: int, address: str = "0100007F") -> bytes:
    return (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        f"   0: {address}:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 {inode} 1\n"
    ).encode("ascii")


def listener(*, port: int = 18080, inode: int = 42) -> capture.BoundListener:
    return capture.BoundListener(
        proc_net_tcp=tcp_listener_table(port=port, inode=inode),
        socket_inode=inode,
        server_socket_inodes=(inode, 99),
    )


class CaptureC02ObservationsV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.evidence_root = self.base / "evidence-root"
        self.evidence_root.mkdir(mode=0o700)
        os.chmod(self.evidence_root, 0o700)
        self.endpoint = capture.parse_endpoint("http://127.0.0.1:18080/v1/c02/metrics")
        self.request = capture.CaptureRequest(
            endpoint=self.endpoint,
            server_pid=123,
            gpu_index=0,
            evidence_root=self.evidence_root,
            capture_name="soak-a",
            interval_seconds=1,
            sample_count=2,
        )
        self.target = capture.TargetIdentity(
            server_pid=123,
            server_start_ticks=456,
            gpu_index=0,
            gpu_uuid="GPU-deadbeef",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fixture_sample(self, sequence: int, elapsed: int) -> capture.CapturedSample:
        raw_prefix = f"{self.request.capture_name}/raw/{sequence:06d}"
        metric = metrics_raw(failed=sequence)
        before_stat = proc_stat()
        after_stat = proc_stat()
        before_listener = tcp_listener_table(port=self.endpoint.port, inode=42)
        after_listener = tcp_listener_table(port=self.endpoint.port, inode=42)
        before_sockets = canonical(
            {
                "schema_version": capture.SOCKET_SNAPSHOT_VERSION,
                "server_pid": 123,
                "socket_inodes": [42, 99],
            }
        )
        after_sockets = before_sockets
        status = proc_status()
        selection = b"0, GPU-deadbeef\n"
        apps = b"123, 0\n"
        document = {
            "schema_version": capture.SAMPLE_VERSION,
            "sequence": sequence,
            "elapsed_monotonic_millis": elapsed,
            "endpoint": {
                "http_status": 200,
                "body": capture._descriptor(f"{raw_prefix}.metrics.json", metric),
                "listener": {
                    "address": "127.0.0.1",
                    "port": self.endpoint.port,
                    "socket_inode": 42,
                    "before_proc_net_tcp": capture._descriptor(
                        f"{raw_prefix}.proc-net-tcp-before", before_listener
                    ),
                    "after_proc_net_tcp": capture._descriptor(
                        f"{raw_prefix}.proc-net-tcp-after", after_listener
                    ),
                    "before_server_fd_sockets": capture._descriptor(
                        f"{raw_prefix}.proc-fd-sockets-before.json", before_sockets
                    ),
                    "after_server_fd_sockets": capture._descriptor(
                        f"{raw_prefix}.proc-fd-sockets-after.json", after_sockets
                    ),
                },
            },
            "process": {
                "pid": 123,
                "start_ticks": 456,
                "present": True,
                "pre_endpoint_stat": capture._descriptor(
                    f"{raw_prefix}.proc-stat-before", before_stat
                ),
                "stat": capture._descriptor(f"{raw_prefix}.proc-stat", after_stat),
                "status": capture._descriptor(f"{raw_prefix}.proc-status", status),
            },
            "gpu": {
                "index": 0,
                "uuid": "GPU-deadbeef",
                "selection_query": capture._descriptor(
                    f"{raw_prefix}.gpu-selection.csv", selection
                ),
                "compute_apps": capture._descriptor(
                    f"{raw_prefix}.gpu-compute-apps.csv", apps
                ),
            },
        }
        return capture.CapturedSample(
            raw_files=(
                (f"{sequence:06d}.metrics.json", metric),
                (f"{sequence:06d}.proc-stat-before", before_stat),
                (f"{sequence:06d}.proc-net-tcp-before", before_listener),
                (f"{sequence:06d}.proc-fd-sockets-before.json", before_sockets),
                (f"{sequence:06d}.proc-net-tcp-after", after_listener),
                (f"{sequence:06d}.proc-fd-sockets-after.json", after_sockets),
                (f"{sequence:06d}.proc-stat", after_stat),
                (f"{sequence:06d}.proc-status", status),
                (f"{sequence:06d}.gpu-selection.csv", selection),
                (f"{sequence:06d}.gpu-compute-apps.csv", apps),
            ),
            document=document,
        )

    def _capture_with_fixtures(self, **kwargs: object) -> dict[str, object]:
        request = kwargs.pop("request", self.request)
        assert isinstance(request, capture.CaptureRequest)
        with mock.patch.object(capture, "_preflight_target", return_value=self.target), mock.patch.object(
            capture,
            "_capture_one",
            side_effect=[self._fixture_sample(0, 0), self._fixture_sample(1, 1000)],
        ):
            return capture.capture_observations(
                request,
                repository_root=self.repository,
                sleep=mock.Mock(),
                monotonic_ns=kwargs.pop("monotonic_ns", iter([0, 0, 1_000_000]).__next__),
            )

    def test_endpoint_is_literal_loopback_only(self) -> None:
        self.assertEqual(self.endpoint.port, 18080)
        for value in (
            "https://127.0.0.1:18080/v1/c02/metrics",
            "http://localhost:18080/v1/c02/metrics",
            "http://127.0.0.1:18080/metrics",
            "http://127.0.0.1:18080/v1/c02/metrics?x=1",
            "http://127.0.0.1/v1/c02/metrics",
        ):
            with self.subTest(value=value):
                with self.assertRaises(capture.ObservationCaptureError):
                    capture.parse_endpoint(value)

    def test_metrics_validation_preserves_raw_schema_without_qualification_thresholds(self) -> None:
        capture.validate_metrics_raw(metrics_raw(failed=17, empty_kv=True))
        with self.assertRaisesRegex(capture.ObservationCaptureError, "canonical JSON"):
            capture.validate_metrics_raw(metrics_raw() + b"\n")
        with self.assertRaisesRegex(capture.ObservationCaptureError, "exactly"):
            capture.validate_metrics_raw(canonical({"schema_version": capture.METRICS_VERSION}))

    def test_missing_no_follow_or_directory_open_flag_fails_closed(self) -> None:
        for name in ("O_NOFOLLOW", "O_DIRECTORY"):
            with self.subTest(name=name), mock.patch.object(capture.os, name, None):
                with self.assertRaisesRegex(capture.ObservationCaptureError, name):
                    capture._open_flags(directory=True)

    def test_gpu_selection_is_index_and_uuid_and_compute_apps_binds_server_pid(self) -> None:
        selection_result = mock.Mock(stdout=b"0, GPU-deadbeef\n")
        apps_result = mock.Mock(stdout=b"123, 0\n")
        with mock.patch.object(capture.subprocess, "run", side_effect=[selection_result, apps_result]) as runner:
            selection, apps, uuid = capture._capture_gpu(0, 123, None)
        self.assertEqual((selection, apps, uuid), (b"0, GPU-deadbeef\n", b"123, 0\n", "GPU-deadbeef"))
        first = runner.call_args_list[0].args[0]
        second = runner.call_args_list[1].args[0]
        self.assertEqual(first[:3], ["/usr/bin/nvidia-smi", "-i", "0"])
        self.assertIn("--query-gpu=index,uuid", first)
        self.assertIn("--query-compute-apps=pid,used_gpu_memory", second)
        self.assertNotIn("--id=0", first)
        with self.assertRaisesRegex(capture.ObservationCaptureError, "index differs"):
            capture._parse_gpu_selection(b"1, GPU-deadbeef\n", 0)
        with self.assertRaisesRegex(capture.ObservationCaptureError, "exactly one server PID"):
            capture._validate_compute_apps(b"999, 1\n", 123)

    def test_listener_parser_refuses_wildcard_or_ambiguous_listener(self) -> None:
        table = tcp_listener_table(port=self.endpoint.port, inode=42)
        with mock.patch.object(capture, "_read_absolute_regular", return_value=table):
            self.assertEqual(capture._capture_loopback_listener(self.endpoint.port), (table, 42))
        wildcard = tcp_listener_table(port=self.endpoint.port, inode=42, address="00000000")
        with mock.patch.object(capture, "_read_absolute_regular", return_value=wildcard):
            with self.assertRaisesRegex(capture.ObservationCaptureError, "wildcard"):
                capture._capture_loopback_listener(self.endpoint.port)
        second = (
            f"   1: 0100007F:{self.endpoint.port:04X} 00000000:0000 0A "
            "00000000:00000000 00:00000000 00000000 1000 0 43 1\n"
        ).encode("ascii")
        with mock.patch.object(capture, "_read_absolute_regular", return_value=table + second):
            with self.assertRaisesRegex(capture.ObservationCaptureError, "exactly one"):
                capture._capture_loopback_listener(self.endpoint.port)

    def test_capture_one_binds_pre_and_post_listener_process_and_gpu_raw_leaves(self) -> None:
        metric = metrics_raw()
        with mock.patch.object(
            capture,
            "_capture_server_stat",
            side_effect=[(proc_stat(), 456), (proc_stat(), 456)],
        ), mock.patch.object(
            capture,
            "_capture_bound_listener",
            side_effect=[listener(), listener()],
        ), mock.patch.object(capture, "_capture_endpoint", return_value=metric), mock.patch.object(
            capture,
            "_capture_server_status",
            return_value=proc_status(),
        ), mock.patch.object(
            capture,
            "_capture_gpu",
            return_value=(b"0, GPU-deadbeef\n", b"123, 0\n", "GPU-deadbeef"),
        ):
            result = capture._capture_one(self.request, self.target, sequence=0, elapsed_monotonic_millis=7)
        document = result.document
        self.assertEqual(document["schema_version"], capture.SAMPLE_VERSION)
        self.assertEqual(document["elapsed_monotonic_millis"], 7)
        self.assertEqual(
            set(document["endpoint"]["listener"]),
            {
                "address",
                "port",
                "socket_inode",
                "before_proc_net_tcp",
                "after_proc_net_tcp",
                "before_server_fd_sockets",
                "after_server_fd_sockets",
            },
        )
        self.assertEqual(set(document["process"]), {"pid", "start_ticks", "present", "pre_endpoint_stat", "stat", "status"})
        self.assertEqual(set(document["gpu"]), {"index", "uuid", "selection_query", "compute_apps"})
        self.assertEqual(len(result.raw_files), 10)
        for descriptor in (
            document["endpoint"]["body"],
            document["process"]["pre_endpoint_stat"],
            document["gpu"]["selection_query"],
        ):
            self.assertEqual(set(descriptor), {"path", "sha256", "byte_length"})

    def test_capture_writes_fresh_private_child_and_removes_marker_last(self) -> None:
        session = self._capture_with_fixtures()
        capture_root = self.evidence_root / self.request.capture_name
        self.assertTrue((capture_root / "session.json").is_file())
        self.assertFalse((capture_root / capture.INCOMPLETE_MARKER_NAME).exists())
        for directory in (self.evidence_root, capture_root, capture_root / "raw", capture_root / "samples"):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(session["schema_version"], capture.SESSION_VERSION)
        self.assertEqual(session["target"], self.target.as_json())
        self.assertEqual(len(session["samples"]), 2)
        for sequence, descriptor in enumerate(session["samples"]):
            raw = (self.evidence_root / descriptor["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), descriptor["sha256"])
            self.assertEqual(len(raw), descriptor["byte_length"])
            self.assertEqual(json.loads(raw)["sequence"], sequence)
        self.assertTrue((capture_root / "raw" / "000000.gpu-selection.csv").is_file())
        self.assertTrue((capture_root / "raw" / "000001.proc-fd-sockets-after.json").is_file())

    def test_completion_sync_failure_restores_incomplete_marker(self) -> None:
        original = capture._fsync_checked

        def fail_marker_removal(descriptor: int, label: str) -> None:
            if label == "capture directory after completion marker removal":
                raise capture.ObservationCaptureError("fixture marker sync failure")
            original(descriptor, label)

        with mock.patch.object(capture, "_preflight_target", return_value=self.target), mock.patch.object(
            capture,
            "_capture_one",
            side_effect=[self._fixture_sample(0, 0), self._fixture_sample(1, 1000)],
        ), mock.patch.object(capture, "_fsync_checked", side_effect=fail_marker_removal):
            with self.assertRaisesRegex(capture.ObservationCaptureError, "marker sync failure"):
                capture.capture_observations(
                    self.request,
                    repository_root=self.repository,
                    sleep=mock.Mock(),
                    monotonic_ns=iter([0, 0, 1_000_000]).__next__,
                )
        capture_root = self.evidence_root / self.request.capture_name
        self.assertTrue((capture_root / "session.json").is_file())
        self.assertTrue((capture_root / capture.INCOMPLETE_MARKER_NAME).is_file())

    def test_marker_exists_if_capture_directory_initialization_fails_after_creation(self) -> None:
        original = capture._require_new_private_directory

        def fail_raw_directory(parent_fd: int, name: str, label: str) -> int:
            if name == "raw":
                raise capture.ObservationCaptureError("fixture raw directory failure")
            return original(parent_fd, name, label)

        with mock.patch.object(capture, "_preflight_target", return_value=self.target), mock.patch.object(
            capture,
            "_require_new_private_directory",
            side_effect=fail_raw_directory,
        ):
            with self.assertRaisesRegex(capture.ObservationCaptureError, "raw directory failure"):
                capture.capture_observations(
                    self.request,
                    repository_root=self.repository,
                    sleep=mock.Mock(),
                    monotonic_ns=iter([0]).__next__,
                )
        capture_root = self.evidence_root / self.request.capture_name
        self.assertTrue((capture_root / capture.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((capture_root / "session.json").exists())

    def test_existing_capture_source_root_and_root_symlink_are_refused(self) -> None:
        self._capture_with_fixtures()
        with mock.patch.object(capture, "_preflight_target", return_value=self.target):
            with self.assertRaisesRegex(capture.ObservationCaptureError, "already exists"):
                capture.capture_observations(
                    self.request,
                    repository_root=self.repository,
                    sleep=mock.Mock(),
                    monotonic_ns=iter([0]).__next__,
                )
        source_root = self.repository / "evidence"
        source_root.mkdir(mode=0o700)
        source_request = capture.CaptureRequest(
            **{**self.request.__dict__, "evidence_root": source_root, "capture_name": "source"}
        )
        with mock.patch.object(capture, "_preflight_target", return_value=self.target):
            with self.assertRaisesRegex(capture.ObservationCaptureError, "outside the source checkout"):
                capture.capture_observations(
                    source_request,
                    repository_root=self.repository,
                    sleep=mock.Mock(),
                    monotonic_ns=iter([0]).__next__,
                )
        link = self.base / "evidence-link"
        os.symlink(self.evidence_root, link)
        with self.assertRaisesRegex(capture.ObservationCaptureError, "without following links"):
            capture._open_private_evidence_root(link, "symlink root")

    def test_v2_output_round_trips_through_current_raw_binder(self) -> None:
        session = self._capture_with_fixtures()
        opaque = {
            "scenario-contract.json": b"contract\n",
            "request-ledger.json": b"ledger\n",
            "runtime-event.log": b"runtime\n",
            "generation-audit-index.json": b"audit\n",
        }
        for name, raw in opaque.items():
            (self.evidence_root / name).write_bytes(raw)
        effective_config = {
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
            "kv_geometry": {"layout": "paged", "block_tokens": 16, "physical_blocks": 512},
        }
        endpoint_document = {
            "schema_version": runtime_config.ENDPOINT_VERSION,
            "candidate_id": "riley-0.1.0-rc2",
            "runtime_identity": {
                "configuration_profile": "stable-default",
                "configuration_sha256": "c" * 64,
            },
            "effective_config": effective_config,
            "effective_config_sha256": hashlib.sha256(
                runtime_config.canonical_json_bytes(effective_config)
            ).hexdigest(),
        }
        endpoint_raw = runtime_config.canonical_json_bytes(endpoint_document)
        startup_document = {
            "schema_version": runtime_config.STARTUP_ARTIFACT_VERSION,
            "created_at_utc": "2026-08-29T00:00:00Z",
            "candidate_id": "riley-0.1.0-rc2",
            "endpoint_path": "/v1/config",
            "runtime_identity": endpoint_document["runtime_identity"],
            "endpoint_payload_sha256": hashlib.sha256(endpoint_raw).hexdigest(),
            "endpoint_payload": endpoint_document,
        }
        (self.evidence_root / "config-endpoint.json").write_bytes(endpoint_raw)
        startup_raw = runtime_config.canonical_json_bytes(startup_document)
        (self.evidence_root / "config-startup.json").write_bytes(startup_raw)
        session_raw = (self.evidence_root / self.request.capture_name / "session.json").read_bytes()
        manifest = {
            "schema_version": "riley.soak-v2-raw-provenance.v2",
            "capture_status": "captured",
            "qualification_status": "not-run",
            "candidate_id": "riley-0.1.0-rc2",
            "bindings": {
                "freeze_sha256": "a" * 64,
                "base_release_candidate_report_sha256": "b" * 64,
                "configuration_profile": "stable-default",
                "configuration_sha256": "c" * 64,
            },
            "configuration_evidence": {
                "endpoint": capture._descriptor("config-endpoint.json", endpoint_raw),
                "startup_artifact": capture._descriptor("config-startup.json", startup_raw),
            },
            "scenario_contract": capture._descriptor("scenario-contract.json", opaque["scenario-contract.json"]),
            "scenarios": [
                {
                    "scenario_id": "normal",
                    "target": session["target"],
                    "observation_session": capture._descriptor(
                        f"{self.request.capture_name}/session.json", session_raw
                    ),
                    "request_ledger": capture._descriptor("request-ledger.json", opaque["request-ledger.json"]),
                    "runtime_event_log": capture._descriptor("runtime-event.log", opaque["runtime-event.log"]),
                    "generation_audit_index": capture._descriptor(
                        "generation-audit-index.json", opaque["generation-audit-index.json"]
                    ),
                    "fallback_event_log": None,
                }
            ],
        }
        (self.evidence_root / "soak-manifest.json").write_bytes(canonical(manifest))
        import check_c02_provenance_v2 as binder

        report = binder.verify_soak_provenance(self.evidence_root, "soak-manifest.json")
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["targets"][0]["target"], self.target.as_json())

    def test_isolated_python_can_load_cli_help_and_source_has_no_semantic_checker(self) -> None:
        script = Path(capture.__file__)
        completed = subprocess.run(
            [sys.executable, "-B", "-I", "-S", str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--evidence-root", completed.stdout)
        self.assertIn("--capture-name", completed.stdout)
        source = script.read_text(encoding="utf-8")
        for forbidden in (
            "check_soak_v2_receipt",
            "check_rc3_rollback_receipt",
            "check_rc3_qualification",
            "candidate_id",
            "freeze_sha256",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
