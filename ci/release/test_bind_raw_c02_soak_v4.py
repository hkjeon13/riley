#!/usr/bin/env python3
"""CPU-only integration tests for the closed C02 serial raw binder v4."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bind_raw_c02_soak_v2 as v3_binder
import bind_raw_c02_soak_v4 as binder
import capture_c02_raw_soak_scenarios_v1 as capture
import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common


class EvidenceTree:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, relative: str, value: bytes | dict) -> bytes:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = common.canonical_json_bytes(value) if isinstance(value, dict) else value
        path.write_bytes(raw)
        return raw

    def descriptor(self, relative: str, raw: bytes) -> dict:
        return common.descriptor_for_bytes(relative, raw, relative).as_json()


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


def _proc_stat(pid: int, ticks: int) -> bytes:
    fields = ["S", *("0" for _ in range(18)), str(ticks), "0"]
    return f"{pid} (riley server) {' '.join(fields)}\n".encode("ascii")


def _proc_tcp(port: int, inode: int) -> bytes:
    return (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        f"   0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 {inode} 1\n"
    ).encode("ascii")


def _metrics() -> dict:
    return {
        "schema_version": checker.METRICS_VERSION,
        "request_states": {
            "active": 0,
            "pending_requests": 0,
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
            "capacity_rejections": 0,
        },
        "kv_blocks": {"free": 10, "reserved": 0, "active": 0},
        "allocation": {
            "device_live_count": 0,
            "device_live_bytes": 0,
            "pinned_live_count": 0,
            "pinned_live_bytes": 0,
        },
        "quiescence": {
            "completion_outbox": 0,
            "outstanding_iterations": 0,
            "riley_owned_live_allocations": 0,
            "worker_accepting": False,
            "scheduler_accepting": False,
        },
    }


class BindRawC02SoakV4Tests(unittest.TestCase):
    candidate_id = "riley-0.1.0-rc3"
    profile = checker.STABLE_DEFAULT_PROFILE
    configuration_sha256 = "c" * 64
    gpu_uuid = "GPU-12345678-abcd-efab-cdef-1234567890ab"
    pid = 1111
    ticks = 2222
    port = 18080
    inode = 7001

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.root = self.base / "evidence"
        self.root.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        self.tree = EvidenceTree(self.root)
        self.audit_dir = self.root / "source-audit"
        self.audit_dir.mkdir(mode=0o700)
        os.chmod(self.audit_dir, 0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _target(self) -> dict:
        return {
            "server_pid": self.pid,
            "server_start_ticks": self.ticks,
            "gpu_index": 0,
            "gpu_uuid": self.gpu_uuid,
        }

    def _configuration_evidence(self) -> dict:
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
        identity = {
            "configuration_profile": self.profile,
            "configuration_sha256": self.configuration_sha256,
        }
        endpoint = {
            "schema_version": runtime_config.ENDPOINT_VERSION,
            "candidate_id": self.candidate_id,
            "runtime_identity": identity,
            "effective_config": effective_config,
            "effective_config_sha256": hashlib.sha256(
                runtime_config.canonical_json_bytes(effective_config)
            ).hexdigest(),
        }
        endpoint_raw = self.tree.put("config/endpoint.json", endpoint)
        self.tree.put(
            "config/startup.json",
            {
                "schema_version": runtime_config.STARTUP_ARTIFACT_VERSION,
                "created_at_utc": "2026-08-29T00:00:00Z",
                "candidate_id": self.candidate_id,
                "endpoint_path": "/v1/config",
                "runtime_identity": identity,
                "endpoint_payload_sha256": hashlib.sha256(endpoint_raw).hexdigest(),
                "endpoint_payload": endpoint,
            },
        )
        base = "config-bridge"
        leaves = {
            "request": self.tree.put(
                f"{base}/raw/config-request.http",
                (
                    f"GET /v1/config HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n"
                    "Accept: application/json\r\nConnection: close\r\n\r\n"
                ).encode("ascii"),
            ),
            "head": self.tree.put(
                f"{base}/raw/config-response-head.http",
                (
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(endpoint_raw)}\r\n"
                    "Content-Type: application/json\r\n\r\n"
                ).encode("ascii"),
            ),
            "tcp_before": self.tree.put(f"{base}/raw/proc-tcp-before", _proc_tcp(self.port, self.inode)),
            "tcp_after": self.tree.put(f"{base}/raw/proc-tcp-after", _proc_tcp(self.port, self.inode)),
            "sockets_before": self.tree.put(f"{base}/raw/fds-before.json", {"schema_version": checker.SOCKET_SNAPSHOT_VERSION, "server_pid": self.pid, "socket_inodes": [self.inode]}),
            "sockets_after": self.tree.put(f"{base}/raw/fds-after.json", {"schema_version": checker.SOCKET_SNAPSHOT_VERSION, "server_pid": self.pid, "socket_inodes": [self.inode]}),
            "stat_before": self.tree.put(f"{base}/raw/stat-before", _proc_stat(self.pid, self.ticks)),
            "stat_after": self.tree.put(f"{base}/raw/stat-after", _proc_stat(self.pid, self.ticks)),
            "status": self.tree.put(f"{base}/raw/status", f"Name:\triley\nPid:\t{self.pid}\n".encode("ascii")),
            "selection": self.tree.put(f"{base}/raw/gpu-selection.csv", f"0, {self.gpu_uuid}\n".encode("ascii")),
            "apps": self.tree.put(f"{base}/raw/gpu-apps.csv", f"{self.pid}, 42\n".encode("ascii")),
        }
        descriptors = {key: self.tree.descriptor(
            f"{base}/raw/" + {
                "request": "config-request.http", "head": "config-response-head.http",
                "tcp_before": "proc-tcp-before", "tcp_after": "proc-tcp-after",
                "sockets_before": "fds-before.json", "sockets_after": "fds-after.json",
                "stat_before": "stat-before", "stat_after": "stat-after", "status": "status",
                "selection": "gpu-selection.csv", "apps": "gpu-apps.csv",
            }[key], raw)
            for key, raw in leaves.items()
        }
        self.tree.put(
            f"{base}/session.json",
            {
                "schema_version": checker.CONFIG_ENDPOINT_OBSERVATION_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "target": {**self._target(), "listener_port": self.port, "listener_inode": self.inode},
                "endpoint": {
                    "method": "GET", "request_target": "/v1/config", "http_status": 200,
                    "request": descriptors["request"], "response_head": descriptors["head"],
                    "body_sha256": hashlib.sha256(endpoint_raw).hexdigest(), "body_byte_length": len(endpoint_raw),
                    "listener": {"address": "127.0.0.1", "port": self.port, "socket_inode": self.inode, "before_proc_net_tcp": descriptors["tcp_before"], "after_proc_net_tcp": descriptors["tcp_after"], "before_server_fd_sockets": descriptors["sockets_before"], "after_server_fd_sockets": descriptors["sockets_after"]},
                },
                "process": {"server_pid": self.pid, "server_start_ticks": self.ticks, "pre_endpoint_stat": descriptors["stat_before"], "post_endpoint_stat": descriptors["stat_after"], "status": descriptors["status"]},
                "gpu": {"index": 0, "uuid": self.gpu_uuid, "selection_query": descriptors["selection"], "compute_apps": descriptors["apps"]},
            },
        )
        return {
            "endpoint_path": "config/endpoint.json",
            "startup_artifact_path": "config/startup.json",
            "endpoint_observation_path": f"{base}/session.json",
        }

    def _observation_session(
        self,
        name: str,
        *,
        port: int | None = None,
        gpu_index: int | None = None,
        gpu_uuid: str | None = None,
    ) -> str:
        observed_port = self.port if port is None else port
        observed_gpu_index = 0 if gpu_index is None else gpu_index
        observed_gpu_uuid = self.gpu_uuid if gpu_uuid is None else gpu_uuid
        base = name
        leaves = {
            "metrics": self.tree.put(f"{base}/raw/metrics.json", _metrics()),
            "tcp_before": self.tree.put(f"{base}/raw/tcp-before", _proc_tcp(observed_port, self.inode)),
            "tcp_after": self.tree.put(f"{base}/raw/tcp-after", _proc_tcp(observed_port, self.inode)),
            "sockets_before": self.tree.put(f"{base}/raw/fds-before.json", {"schema_version": checker.SOCKET_SNAPSHOT_VERSION, "server_pid": self.pid, "socket_inodes": [self.inode]}),
            "sockets_after": self.tree.put(f"{base}/raw/fds-after.json", {"schema_version": checker.SOCKET_SNAPSHOT_VERSION, "server_pid": self.pid, "socket_inodes": [self.inode]}),
            "stat_before": self.tree.put(f"{base}/raw/stat-before", _proc_stat(self.pid, self.ticks)),
            "stat": self.tree.put(f"{base}/raw/stat", _proc_stat(self.pid, self.ticks)),
            "status": self.tree.put(f"{base}/raw/status", f"Name:\triley\nPid:\t{self.pid}\n".encode("ascii")),
            "selection": self.tree.put(f"{base}/raw/gpu-selection.csv", f"{observed_gpu_index}, {observed_gpu_uuid}\n".encode("ascii")),
            "apps": self.tree.put(f"{base}/raw/gpu-apps.csv", f"{self.pid}, 42\n".encode("ascii")),
        }
        descriptors = {key: self.tree.descriptor(
            f"{base}/raw/" + {
                "metrics": "metrics.json", "tcp_before": "tcp-before", "tcp_after": "tcp-after",
                "sockets_before": "fds-before.json", "sockets_after": "fds-after.json",
                "stat_before": "stat-before", "stat": "stat", "status": "status",
                "selection": "gpu-selection.csv", "apps": "gpu-apps.csv",
            }[key], raw)
            for key, raw in leaves.items()
        }
        sample_path = f"{base}/samples/000000.json"
        sample_raw = self.tree.put(
            sample_path,
            {
                "schema_version": checker.OBSERVATION_SAMPLE_VERSION,
                "sequence": 0,
                "elapsed_monotonic_millis": 0,
                "endpoint": {
                    "http_status": 200,
                    "body": descriptors["metrics"],
                    "listener": {"address": "127.0.0.1", "port": observed_port, "socket_inode": self.inode, "before_proc_net_tcp": descriptors["tcp_before"], "after_proc_net_tcp": descriptors["tcp_after"], "before_server_fd_sockets": descriptors["sockets_before"], "after_server_fd_sockets": descriptors["sockets_after"]},
                },
                "process": {"pid": self.pid, "start_ticks": self.ticks, "present": True, "pre_endpoint_stat": descriptors["stat_before"], "stat": descriptors["stat"], "status": descriptors["status"]},
                "gpu": {"index": observed_gpu_index, "uuid": observed_gpu_uuid, "selection_query": descriptors["selection"], "compute_apps": descriptors["apps"]},
            },
        )
        self.tree.put(
            f"{base}/session.json",
            {
                "schema_version": checker.OBSERVATION_SESSION_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "endpoint": {"url": f"http://127.0.0.1:{observed_port}/v1/c02/metrics", "expected_schema_version": checker.METRICS_VERSION},
                "target": {**self._target(), "gpu_index": observed_gpu_index, "gpu_uuid": observed_gpu_uuid},
                "samples": [self.tree.descriptor(sample_path, sample_raw)],
            },
        )
        return f"{base}/session.json"

    def _audit(self, request_id: str) -> None:
        raw = common.canonical_json_bytes(
            {
                "schema_version": capture.AUDIT_VERSION,
                "candidate_id": self.candidate_id,
                "runtime_identity": {"configuration_profile": self.profile, "configuration_sha256": self.configuration_sha256},
                "process_identity": {"pid": self.pid, "start_ticks": self.ticks},
                "server_request_id": request_id,
                "delivery_mode": "non-stream",
                "prompt_token_ids": [1],
                "committed_output_tokens": [{"emitted_text_delta": "x", "token_id": 2}],
                "sampling_selections": [{"committed": True, "configured_backend": "gpu-greedy", "ineligibility_reason": None, "iteration_id": 1, "selected_backend": "gpu-greedy"}],
                "finish_reason": "length",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        name = f"{request_id}.json"
        (self.audit_dir / name).write_bytes(raw)
        (self.audit_dir / f"{name}.complete").write_bytes(
            common.canonical_json_bytes({"schema_version": capture.AUDIT_COMPLETION_VERSION, "artifact_filename": name, "artifact_sha256": hashlib.sha256(raw).hexdigest()})
        )

    def _serial_capture(self, scenario_ids: tuple[str, ...] = ("smoke",)) -> str:
        contract = {
            "schema_version": capture.CONTRACT_VERSION,
            "candidate_id": self.candidate_id,
            "configuration_profile": self.profile,
            "scenarios": [
                {"scenario_id": scenario_id, "completion_request": {"model": "fixture-model", "prompt": f"hello-{scenario_id}", "max_tokens": 2, "temperature": 0.0, "top_p": 1.0, "seed": index + 1, "stream": False}}
                for index, scenario_id in enumerate(scenario_ids)
            ],
        }
        contract_path = self.base / "serial-contract.json"
        contract_path.write_bytes(common.canonical_json_bytes(contract))
        sockets: list[FakeSocket] = []
        for index, _scenario_id in enumerate(scenario_ids):
            request_id = f"cmpl-{index + 1}"
            self._audit(request_id)
            body = common.canonical_json_bytes({"id": request_id, "object": "text_completion"})
            head = (f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nContent-Type: application/json\r\n\r\n").encode("ascii")
            sockets.append(FakeSocket([head + body, b""]))
        listener = capture.Listener(tcp=_proc_tcp(self.port, self.inode), inode=self.inode, sockets=(self.inode,))
        request = capture.CaptureRequest(
            endpoint=capture.parse_endpoint(f"http://127.0.0.1:{self.port}/v1/completions"),
            server_pid=self.pid,
            candidate_id=self.candidate_id,
            configuration_profile=self.profile,
            configuration_sha256=self.configuration_sha256,
            evidence_root=self.root,
            capture_name="serial-capture",
            audit_dir_name="source-audit",
            scenario_contract=contract_path,
            audit_wait_seconds=0.2,
        )
        with mock.patch.object(capture.socket, "create_connection", side_effect=sockets), mock.patch.object(
            capture, "_server_stat", return_value=(_proc_stat(self.pid, self.ticks), self.ticks)
        ), mock.patch.object(capture, "_bound_listener", return_value=listener):
            capture.capture_raw_scenarios(request, repository_root=self.repository)
        return "serial-capture/session.json"

    def _request(self, *, scenario_ids: tuple[str, ...] = ("smoke",), observation_ports: tuple[int, ...] | None = None) -> tuple[str, dict]:
        configuration = self._configuration_evidence()
        capture_path = self._serial_capture(scenario_ids)
        ports = (self.port,) * len(scenario_ids) if observation_ports is None else observation_ports
        request = {
            "schema_version": binder.BIND_REQUEST_VERSION,
            "candidate_id": self.candidate_id,
            "binding_inputs": {"freeze_sha256": "a" * 64, "base_release_candidate_report_sha256": "b" * 64, "configuration_profile": self.profile},
            "configuration_evidence": configuration,
            "scenario_capture_session_path": capture_path,
            "scenarios": [
                {"scenario_id": scenario_id, "observation_session_path": self._observation_session(f"obs-{index}", port=ports[index])}
                for index, scenario_id in enumerate(scenario_ids)
            ],
        }
        path = "requests/v4-bind-request.json"
        self.tree.put(path, request)
        return path, request

    def _redirect_source_audit(self, scenario_index: int, audit_directory: str) -> None:
        """Move one producer audit pair and rebind its v1 session descriptors."""

        session_path = "serial-capture/session.json"
        session = json.loads((self.root / session_path).read_text(encoding="utf-8"))
        scenario = session["scenarios"][scenario_index]
        index_path = scenario["generation_audit_index"]["path"]
        index = json.loads((self.root / index_path).read_text(encoding="utf-8"))
        record = index["audit_record"]
        marker = index["audit_completion_marker"]
        record_name = Path(record["path"]).name
        marker_name = Path(marker["path"]).name
        record_raw = (self.root / record["path"]).read_bytes()
        marker_raw = (self.root / marker["path"]).read_bytes()
        new_record_path = f"{audit_directory}/{record_name}"
        new_marker_path = f"{audit_directory}/{marker_name}"
        self.tree.put(new_record_path, record_raw)
        self.tree.put(new_marker_path, marker_raw)
        index["audit_record"] = self.tree.descriptor(new_record_path, record_raw)
        index["audit_completion_marker"] = self.tree.descriptor(new_marker_path, marker_raw)
        index_raw = self.tree.put(index_path, index)
        scenario["runtime_event_log"] = self.tree.descriptor(new_record_path, record_raw)
        scenario["generation_audit_index"] = self.tree.descriptor(index_path, index_raw)
        self.tree.put(session_path, session)

    def _write_marker_pair(self, name: str, marker: dict) -> None:
        """Install a deliberately manual v4 pair for hostile verifier tests."""

        staging = self.root / f"{name}.marker-staging"
        staging.write_bytes(common.canonical_json_bytes(marker))
        os.chmod(staging, 0o600)
        os.link(staging, self.root / f"{name}.complete")
        os.link(staging, self.root / f"{name}.intent")
        os.unlink(staging)

    def test_binds_actual_serial_producer_capture_and_publishes_v4_marker(self) -> None:
        request_path, _request = self._request(scenario_ids=("smoke", "short"))
        report = binder.bind_raw_soak_manifest(self.root, request_path, "soak-v4.json")
        manifest_raw = (self.root / "soak-v4.json").read_bytes()
        manifest = common.parse_canonical_json(manifest_raw, "v4 manifest")
        marker = common.parse_canonical_json((self.root / "soak-v4.json.complete").read_bytes(), "v4 marker")
        final_stat = os.lstat(self.root / "soak-v4.json.complete")
        intent_stat = os.lstat(self.root / "soak-v4.json.intent")
        self.assertEqual(manifest["schema_version"], checker.SOAK_V4_MANIFEST_VERSION)
        self.assertEqual(manifest["scenario_capture_session"]["path"], "serial-capture/session.json")
        self.assertEqual(manifest["scenario_contract"]["path"], "serial-capture/raw/scenario-contract.json")
        self.assertEqual([row["scenario_id"] for row in manifest["scenarios"]], ["smoke", "short"])
        self.assertEqual(manifest["scenarios"][0]["runtime_event_log"]["path"], "source-audit/cmpl-1.json")
        self.assertEqual(marker["schema_version"], checker.SOAK_V4_COMPLETION_MARKER_VERSION)
        self.assertEqual(marker["artifact_sha256"], hashlib.sha256(manifest_raw).hexdigest())
        self.assertEqual((final_stat.st_dev, final_stat.st_ino), (intent_stat.st_dev, intent_stat.st_ino))
        self.assertEqual(final_stat.st_nlink, 2)
        self.assertEqual(intent_stat.st_nlink, 2)
        self.assertEqual(report["schema_version"], checker.SOAK_V4_REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(checker.verify_completed_soak_provenance_v4(self.root, "soak-v4.json"), report)

    def test_v1_replay_exposes_only_verified_completion_descriptors(self) -> None:
        self._request(scenario_ids=("smoke",))
        session_path = "serial-capture/session.json"
        session_raw = (self.root / session_path).read_bytes()
        session = common.parse_canonical_json(session_raw, "serial capture session")
        assert isinstance(session, dict)
        ledger_path = session["scenarios"][0]["request_ledger"]["path"]
        ledger = common.parse_canonical_json(
            (self.root / ledger_path).read_bytes(),
            "serial capture request ledger",
        )
        assert isinstance(ledger, dict)

        root_fd = common.open_private_evidence_directory(self.root, "test evidence root")
        try:
            replay = checker.replay_raw_scenario_capture_v1_fd(
                root_fd,
                common.descriptor_for_bytes(
                    session_path,
                    session_raw,
                    "serial capture session",
                ),
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
                configuration_sha256=self.configuration_sha256,
                used_paths=set(),
            )
        finally:
            os.close(root_fd)

        scenario = replay.scenarios[0]
        self.assertEqual(
            scenario.request,
            common.parse_descriptor(ledger["request"], "expected request"),
        )
        self.assertEqual(
            scenario.response_head,
            common.parse_descriptor(ledger["response_head"], "expected response head"),
        )
        self.assertEqual(
            scenario.response_body,
            common.parse_descriptor(ledger["response_body"], "expected response body"),
        )

    def test_rejects_legacy_fields_versions_and_reordered_inventory(self) -> None:
        request_path, request = self._request(scenario_ids=("first", "second"))
        request["schema_version"] = "riley.soak-v2-bind-request.v3"
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "legacy.json")
        self.assert_reason(raised, "unsupported-bind-request-version")
        request["schema_version"] = binder.BIND_REQUEST_VERSION
        request["scenario_contract_path"] = "serial-capture/raw/scenario-contract.json"
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "legacy-field.json")
        self.assert_reason(raised, "unexpected-field-set")
        request.pop("scenario_contract_path")
        request["scenarios"].reverse()
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "reordered.json")
        self.assert_reason(raised, "scenario-capture-inventory-mismatch")

    def test_rejects_retained_capture_marker_and_observation_listener_drift(self) -> None:
        request_path, request = self._request()
        self.tree.put("serial-capture/capture-incomplete.json", {"schema_version": "riley.c02-raw-scenario-capture-incomplete.v1", "capture_name": "serial-capture"})
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "incomplete.json")
        self.assert_reason(raised, "incomplete-capture")
        (self.root / "serial-capture/capture-incomplete.json").unlink()
        request["scenarios"][0]["observation_session_path"] = self._observation_session(
            "obs-drift", port=18081
        )
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "drift.json")
        self.assert_reason(raised, "scenario-capture-observation-target-mismatch")

    def test_replays_the_source_audit_link_instead_of_accepting_an_opaque_event_leaf(self) -> None:
        request_path, _request = self._request()
        session_path = self.root / "serial-capture/session.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        scenario = session["scenarios"][0]
        scenario["runtime_event_log"] = scenario["generation_audit_index"]
        self.tree.put("serial-capture/session.json", session)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "opaque-event.json")
        self.assert_reason(raised, "scenario-capture-runtime-event-mismatch")

    def test_rejects_nested_session_paths_and_gpu_drift_before_publication(self) -> None:
        request_path, request = self._request()
        request["scenarios"][0]["observation_session_path"] = "nested/observation/session.json"
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "nested-path.json")
        self.assert_reason(raised, "invalid-session-path")
        self.assertFalse((self.root / "nested-path.json").exists())

        request["scenarios"][0]["observation_session_path"] = self._observation_session(
            "obs-gpu-drift",
            gpu_index=1,
            gpu_uuid="GPU-87654321-abcd-efab-cdef-1234567890ab",
        )
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "gpu-drift.json")
        self.assert_reason(raised, "configuration-scenario-target-mismatch")
        self.assertFalse((self.root / "gpu-drift.json").exists())

    def test_v4_verifier_rejects_manually_authored_nested_sessions(self) -> None:
        request_path, _request = self._request()
        binder.bind_raw_soak_manifest(self.root, request_path, "valid-v4.json")
        manifest = json.loads((self.root / "valid-v4.json").read_text(encoding="utf-8"))

        nested_config = json.loads(json.dumps(manifest))
        nested_config["configuration_evidence"]["endpoint_observation"][
            "path"
        ] = "nested/config/session.json"
        self.tree.put("nested-config.json", nested_config)
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_soak_provenance_v4(self.root, "nested-config.json")
        self.assert_reason(raised, "invalid-session-path")

        nested_observation = json.loads(json.dumps(manifest))
        nested_observation["scenarios"][0]["observation_session"][
            "path"
        ] = "nested/observation/session.json"
        self.tree.put("nested-observation.json", nested_observation)
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_soak_provenance_v4(self.root, "nested-observation.json")
        self.assert_reason(raised, "invalid-session-path")

    def test_preexisting_exact_completion_marker_cannot_create_a_terminal_pair(self) -> None:
        request_path, _request = self._request()
        name = "preexisting-marker.json"
        root_fd = common.open_private_evidence_directory(self.root, "test evidence root")
        try:
            manifest = binder._manifest_from_request(root_fd, request_path, name)
        finally:
            os.close(root_fd)
        manifest_raw = common.canonical_json_bytes(manifest)
        self.tree.put(
            f"{name}.complete",
            {
                "schema_version": checker.SOAK_V4_COMPLETION_MARKER_VERSION,
                "artifact_filename": name,
                "artifact_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            },
        )
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, name)
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / name).exists())
        with self.assertRaises(checker.C02ProvenanceError):
            checker.verify_completed_soak_provenance_v4(self.root, name)

        intent_only_name = "preexisting-intent.json"
        self.tree.put(f"{intent_only_name}.intent", {"stale": True})
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, intent_only_name)
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / intent_only_name).exists())

    def test_v4_verifier_rejects_unpaired_or_aliased_completion_markers(self) -> None:
        request_path, _request = self._request()
        name = "paired-marker-shape.json"
        binder.bind_raw_soak_manifest(self.root, request_path, name)
        final = self.root / f"{name}.complete"
        intent = self.root / f"{name}.intent"

        os.unlink(intent)
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance_v4(self.root, name)
        self.assert_reason(raised, "missing-soak-v4-completion-marker")

        os.unlink(final)
        marker = {
            "schema_version": checker.SOAK_V4_COMPLETION_MARKER_VERSION,
            "artifact_filename": name,
            "artifact_sha256": hashlib.sha256((self.root / name).read_bytes()).hexdigest(),
        }
        self._write_marker_pair(name, marker)
        os.link(final, self.root / "marker-third-link")
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance_v4(self.root, name)
        self.assert_reason(raised, "invalid-paired-hardlink")
        os.unlink(self.root / "marker-third-link")

        os.unlink(final)
        os.unlink(intent)
        final_stage = self.root / "marker-final-stage"
        intent_stage = self.root / "marker-intent-stage"
        final_stage.write_bytes(common.canonical_json_bytes(marker))
        intent_stage.write_bytes(common.canonical_json_bytes(marker))
        os.chmod(final_stage, 0o600)
        os.chmod(intent_stage, 0o600)
        os.link(final_stage, final)
        os.link(final_stage, self.root / "marker-final-peer")
        os.link(intent_stage, intent)
        os.link(intent_stage, self.root / "marker-intent-peer")
        os.unlink(final_stage)
        os.unlink(intent_stage)
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance_v4(self.root, name)
        self.assert_reason(raised, "invalid-paired-hardlink")

    def test_v4_verifier_rejects_symlink_and_content_mismatched_completion_markers(self) -> None:
        request_path, _request = self._request()
        name = "paired-marker-content.json"
        binder.bind_raw_soak_manifest(self.root, request_path, name)
        final = self.root / f"{name}.complete"
        intent = self.root / f"{name}.intent"
        os.unlink(final)
        os.symlink(intent.name, final)
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance_v4(self.root, name)
        self.assert_reason(raised, "invalid-paired-hardlink")

        os.unlink(final)
        os.unlink(intent)
        self._write_marker_pair(
            name,
            {
                "schema_version": checker.SOAK_V4_COMPLETION_MARKER_VERSION,
                "artifact_filename": name,
                "artifact_sha256": "f" * 64,
            },
        )
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance_v4(self.root, name)
        self.assert_reason(raised, "soak-v4-completion-marker-mismatch")

    def test_v4_maximum_manifest_name_leaves_room_for_marker_and_intent(self) -> None:
        request_path, _request = self._request()
        name = f"{'m' * (checker.MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES - len('.json'))}.json"
        self.assertEqual(len(name), checker.MAX_SOAK_TERMINAL_MANIFEST_NAME_BYTES)
        report = binder.bind_raw_soak_manifest(self.root, request_path, name)
        self.assertEqual(report["status"], "bound")
        self.assertTrue((self.root / name).is_file())
        self.assertTrue((self.root / f"{name}.complete").is_file())
        self.assertTrue((self.root / f"{name}.intent").is_file())

    def test_rejects_capture_owned_and_mixed_source_audit_directories(self) -> None:
        request_path, _request = self._request(scenario_ids=("first", "second"))
        self._redirect_source_audit(0, "serial-capture")
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "capture-owned-audit.json")
        self.assert_reason(raised, "source-audit-layout-mismatch")
        self.assertFalse((self.root / "capture-owned-audit.json").exists())

    def test_rejects_mixed_source_audit_directories(self) -> None:
        request_path, _request = self._request(scenario_ids=("first", "second"))
        self._redirect_source_audit(1, "other-source-audit")
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "mixed-audit.json")
        self.assert_reason(raised, "source-audit-layout-mismatch")
        self.assertFalse((self.root / "mixed-audit.json").exists())

    def test_v3_rejects_v1_serial_capture_before_and_after_publication(self) -> None:
        request_path, request = self._request()
        capture_session = json.loads(
            (self.root / "serial-capture/session.json").read_text(encoding="utf-8")
        )
        captured = capture_session["scenarios"][0]
        self.tree.put(
            "serial-capture/capture-incomplete.json",
            {
                "schema_version": "riley.c02-raw-scenario-capture-incomplete.v1",
                "capture_name": "serial-capture",
            },
        )
        v3_request = {
            "schema_version": v3_binder.BIND_REQUEST_VERSION,
            "candidate_id": self.candidate_id,
            "binding_inputs": request["binding_inputs"],
            "configuration_evidence": request["configuration_evidence"],
            "scenario_contract_path": capture_session["contract"]["path"],
            "scenarios": [
                {
                    "scenario_id": captured["scenario_id"],
                    "target": self._target(),
                    "observation_session_path": request["scenarios"][0][
                        "observation_session_path"
                    ],
                    "request_ledger_path": captured["request_ledger"]["path"],
                    "runtime_event_log_path": captured["runtime_event_log"]["path"],
                    "generation_audit_index_path": captured["generation_audit_index"][
                        "path"
                    ],
                    "fallback_event_log_path": None,
                }
            ],
        }
        v3_request_path = "requests/v3-serial-bypass.json"
        self.tree.put(v3_request_path, v3_request)
        with self.assertRaises(v3_binder.RawSoakBindError) as raised:
            v3_binder.bind_raw_soak_manifest(self.root, v3_request_path, "v3-bypass.json")
        self.assert_reason(raised, "v3-serial-capture-v1-rejected")
        self.assertFalse((self.root / "v3-bypass.json").exists())

        def descriptor(path: str) -> dict:
            raw = (self.root / path).read_bytes()
            return self.tree.descriptor(path, raw)

        configuration = request["configuration_evidence"]
        legacy_manifest = {
            "schema_version": checker.SOAK_MANIFEST_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "candidate_id": self.candidate_id,
            "bindings": {
                **request["binding_inputs"],
                "configuration_sha256": self.configuration_sha256,
            },
            "configuration_evidence": {
                "endpoint": descriptor(configuration["endpoint_path"]),
                "startup_artifact": descriptor(configuration["startup_artifact_path"]),
                "endpoint_observation": descriptor(
                    configuration["endpoint_observation_path"]
                ),
            },
            "scenario_contract": descriptor(capture_session["contract"]["path"]),
            "scenarios": [
                {
                    "scenario_id": captured["scenario_id"],
                    "target": self._target(),
                    "observation_session": descriptor(
                        request["scenarios"][0]["observation_session_path"]
                    ),
                    "request_ledger": captured["request_ledger"],
                    "runtime_event_log": captured["runtime_event_log"],
                    "generation_audit_index": captured["generation_audit_index"],
                    "fallback_event_log": None,
                }
            ],
        }
        raw_manifest = self.tree.put("legacy-v3.json", legacy_manifest)
        self.tree.put(
            "legacy-v3.json.complete",
            {
                "schema_version": checker.SOAK_COMPLETION_MARKER_VERSION,
                "artifact_filename": "legacy-v3.json",
                "artifact_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            },
        )
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance(self.root, "legacy-v3.json")
        self.assert_reason(raised, "v3-serial-capture-v1-rejected")

    def test_marker_intent_file_sync_failure_leaves_a_nonterminal_nonreplaceable_v4_manifest(self) -> None:
        request_path, _request = self._request()
        original = common._fsync_checked

        def fail_only_intent(descriptor: int, label: str) -> None:
            if label == "soak raw manifest completion marker intent":
                error = common.ProvenanceV2Error("fixture marker fsync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_only_intent):
            with self.assertRaises(binder.RawSoakBindError) as raised:
                binder.bind_raw_soak_manifest(self.root, request_path, "v4-marker-failure.json")
        self.assert_reason(raised, "durability-failure")
        self.assertTrue((self.root / "v4-marker-failure.json").is_file())
        self.assertTrue((self.root / "v4-marker-failure.json.intent").is_file())
        self.assertFalse((self.root / "v4-marker-failure.json.complete").exists())
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance_v4(self.root, "v4-marker-failure.json")
        self.assert_reason(raised, "missing-soak-v4-completion-marker")
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "v4-marker-failure.json")
        self.assert_reason(raised, "output-name-collision")

    def test_final_marker_directory_sync_failure_is_ambiguous_not_successful(self) -> None:
        request_path, _request = self._request()
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "soak raw manifest completion marker parent directory":
                error = common.ProvenanceV2Error("fixture final marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(binder.RawSoakBindError) as raised:
                binder.bind_raw_soak_manifest(self.root, request_path, "v4-ambiguous-marker.json")
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / "v4-ambiguous-marker.json.complete").is_file())
        self.assertTrue((self.root / "v4-ambiguous-marker.json.intent").is_file())
        self.assertEqual(
            checker.verify_completed_soak_provenance_v4(self.root, "v4-ambiguous-marker.json")["status"],
            "bound",
        )
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "v4-ambiguous-marker.json")
        self.assert_reason(raised, "output-name-collision")

    def test_final_marker_link_failure_leaves_only_nonterminal_intent(self) -> None:
        request_path, _request = self._request()

        def fail_link(*_args: object, **_kwargs: object) -> None:
            error = common.ProvenanceV2Error("fixture marker link failure")
            error.reason_code = "link-publication-failure"  # type: ignore[attr-defined]
            raise error

        with mock.patch.object(common, "publish_create_only_hardlink", side_effect=fail_link):
            with self.assertRaises(binder.RawSoakBindError) as raised:
                binder.bind_raw_soak_manifest(self.root, request_path, "v4-link-failure.json")
        self.assert_reason(raised, "link-publication-failure")
        self.assertTrue((self.root / "v4-link-failure.json.intent").is_file())
        self.assertFalse((self.root / "v4-link-failure.json.complete").exists())
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance_v4(self.root, "v4-link-failure.json")
        self.assert_reason(raised, "missing-soak-v4-completion-marker")

    def test_schema_and_binder_are_closed_and_nonoperational(self) -> None:
        schema_path = Path(__file__).resolve().parents[2] / "benchmarks/release/candidates/soak-v2-bind-request-v4.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], binder.BIND_REQUEST_VERSION)
        self.assertEqual(set(schema["required"]), {"schema_version", "candidate_id", "binding_inputs", "configuration_evidence", "scenario_capture_session_path", "scenarios"})
        scenario = schema["$defs"]["scenario"]
        self.assertEqual(set(scenario["required"]), {"scenario_id", "observation_session_path"})
        self.assertNotIn("target", scenario["properties"])
        capture_schema = json.loads(
            (Path(__file__).resolve().parents[2] / "benchmarks/release/candidates/c02-raw-scenario-capture-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(capture_schema["properties"]["schema_version"]["const"], capture.CAPTURE_VERSION)
        self.assertEqual(
            set(capture_schema["required"]),
            {"schema_version", "capture_status", "qualification_status", "endpoint", "contract", "runtime_identity", "target", "scenarios"},
        )
        source = Path(binder.__file__).read_text(encoding="utf-8")
        for forbidden in ("import socket", "import subprocess", "nvidia-smi", "docker", "podman", "ssh "):
            self.assertNotIn(forbidden, source)
        wrapper = Path(__file__).with_name("run_bind_raw_c02_soak_v4.sh").read_text(encoding="utf-8")
        self.assertIn("bind_raw_c02_soak_v4.py", wrapper)
        for forbidden in ("nvidia-smi", "docker", "podman", "ssh", "curl", "wget", "systemctl"):
            self.assertNotIn(forbidden, wrapper)


if __name__ == "__main__":
    unittest.main()
