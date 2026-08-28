#!/usr/bin/env python3
"""Focused hostile tests for the raw-only C02 provenance v2 proposal."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common


class EvidenceTree:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, relative: str, value: bytes | dict) -> common.EvidenceDescriptor:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = common.canonical_json_bytes(value) if isinstance(value, dict) else value
        path.write_bytes(raw)
        return common.descriptor_for_bytes(relative, raw, relative)


def metrics() -> dict:
    return {
        "schema_version": checker.METRICS_VERSION,
        "request_states": {
            "active": 0,
            "pending_requests": 0,
            "completed": 2,
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


def proc_stat(pid: int, start_ticks: int) -> bytes:
    # The parser starts at Linux field 3 after the parenthesized comm; field
    # 22 (starttime) is index 19 in this suffix.
    fields = ["S", *("0" for _ in range(18)), str(start_ticks), "0"]
    return f"{pid} (riley server) {' '.join(fields)}\n".encode("ascii")


def proc_net_tcp(port: int, inode: int) -> bytes:
    return (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        f"   0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 {inode} 1 0000000000000000 100 0 0 10 0\n"
    ).encode("ascii")


class C02ProvenanceV2Tests(unittest.TestCase):
    candidate = "riley-0.1.0-rc3"
    bindings = {
        "freeze_sha256": "1" * 64,
        "base_release_candidate_report_sha256": "2" * 64,
        "configuration_profile": "stable-default",
        "configuration_sha256": "3" * 64,
    }
    max_performance_bindings = {
        **bindings,
        "configuration_profile": checker.MAX_PERFORMANCE_EXACT_PROFILE,
        "configuration_sha256": "4" * 64,
    }
    gpu_uuid = "GPU-12345678-abcd-efab-cdef-1234567890ab"

    def target(self, pid: int, ticks: int) -> dict:
        return {
            "server_pid": pid,
            "server_start_ticks": ticks,
            "gpu_index": 0,
            "gpu_uuid": self.gpu_uuid,
        }

    def configuration_evidence(
        self,
        tree: EvidenceTree,
        bindings: dict[str, str],
        *,
        pid: int = 1111,
        ticks: int = 2222,
        port: int = 8080,
    ) -> dict:
        profile = bindings["configuration_profile"]
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
        endpoint = {
            "schema_version": runtime_config.ENDPOINT_VERSION,
            "candidate_id": self.candidate,
            "runtime_identity": {
                "configuration_profile": profile,
                "configuration_sha256": bindings["configuration_sha256"],
            },
            "effective_config": effective_config,
            "effective_config_sha256": hashlib.sha256(
                runtime_config.canonical_json_bytes(effective_config)
            ).hexdigest(),
        }
        endpoint_raw = runtime_config.canonical_json_bytes(endpoint)
        startup = {
            "schema_version": runtime_config.STARTUP_ARTIFACT_VERSION,
            "created_at_utc": "2026-08-29T00:00:00Z",
            "candidate_id": self.candidate,
            "endpoint_path": "/v1/config",
            "runtime_identity": endpoint["runtime_identity"],
            "endpoint_payload_sha256": hashlib.sha256(endpoint_raw).hexdigest(),
            "endpoint_payload": endpoint,
        }
        prefix = f"configuration/{profile}"
        inode = 7000 + pid
        bridge_prefix = "captures/config-bridge"
        request_raw = (
            f"GET /v1/config HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            "Accept: application/json\r\nConnection: close\r\n\r\n"
        ).encode("ascii")
        response_head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(endpoint_raw)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        leaves = {
            "request": tree.put(f"{bridge_prefix}/raw/config-request.http", request_raw),
            "response_head": tree.put(f"{bridge_prefix}/raw/config-response-head.http", response_head),
            "tcp_before": tree.put(f"{bridge_prefix}/raw/proc-net-tcp-before", proc_net_tcp(port, inode)),
            "tcp_after": tree.put(f"{bridge_prefix}/raw/proc-net-tcp-after", proc_net_tcp(port, inode)),
            "sockets_before": tree.put(f"{bridge_prefix}/raw/proc-fd-sockets-before.json", {"schema_version": checker.SOCKET_SNAPSHOT_VERSION, "server_pid": pid, "socket_inodes": [inode]}),
            "sockets_after": tree.put(f"{bridge_prefix}/raw/proc-fd-sockets-after.json", {"schema_version": checker.SOCKET_SNAPSHOT_VERSION, "server_pid": pid, "socket_inodes": [inode]}),
            "stat_before": tree.put(f"{bridge_prefix}/raw/proc-stat-before", proc_stat(pid, ticks)),
            "stat_after": tree.put(f"{bridge_prefix}/raw/proc-stat-after", proc_stat(pid, ticks)),
            "status": tree.put(f"{bridge_prefix}/raw/proc-status", f"Name:\triley\nPid:\t{pid}\n".encode("ascii")),
            "selection": tree.put(f"{bridge_prefix}/raw/gpu-selection.csv", f"0, {self.gpu_uuid}\n".encode("ascii")),
            "apps": tree.put(f"{bridge_prefix}/raw/gpu-compute-apps.csv", f"{pid}, 42\n".encode("ascii")),
        }
        bridge = tree.put(
            f"{bridge_prefix}/session.json",
            {
                "schema_version": checker.CONFIG_ENDPOINT_OBSERVATION_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "target": {"server_pid": pid, "server_start_ticks": ticks, "listener_port": port, "listener_inode": inode, "gpu_index": 0, "gpu_uuid": self.gpu_uuid},
                "endpoint": {
                    "method": "GET", "request_target": "/v1/config", "http_status": 200,
                    "request": leaves["request"].as_json(), "response_head": leaves["response_head"].as_json(),
                    "body_sha256": hashlib.sha256(endpoint_raw).hexdigest(), "body_byte_length": len(endpoint_raw),
                    "listener": {"address": "127.0.0.1", "port": port, "socket_inode": inode, "before_proc_net_tcp": leaves["tcp_before"].as_json(), "after_proc_net_tcp": leaves["tcp_after"].as_json(), "before_server_fd_sockets": leaves["sockets_before"].as_json(), "after_server_fd_sockets": leaves["sockets_after"].as_json()},
                },
                "process": {"server_pid": pid, "server_start_ticks": ticks, "pre_endpoint_stat": leaves["stat_before"].as_json(), "post_endpoint_stat": leaves["stat_after"].as_json(), "status": leaves["status"].as_json()},
                "gpu": {"index": 0, "uuid": self.gpu_uuid, "selection_query": leaves["selection"].as_json(), "compute_apps": leaves["apps"].as_json()},
            },
        )
        return {
            "endpoint": tree.put(f"{prefix}-endpoint.json", endpoint_raw).as_json(),
            "startup_artifact": tree.put(f"{prefix}-startup.json", startup).as_json(),
            "endpoint_observation": bridge.as_json(),
        }

    def complete_soak_manifest(
        self, tree: EvidenceTree, manifest: common.EvidenceDescriptor
    ) -> common.EvidenceDescriptor:
        return tree.put(
            f"{manifest.path}.complete",
            {
                "schema_version": checker.SOAK_COMPLETION_MARKER_VERSION,
                "artifact_filename": Path(manifest.path).name,
                "artifact_sha256": manifest.sha256,
            },
        )

    def session(self, tree: EvidenceTree, name: str, *, pid: int, ticks: int, port: int) -> common.EvidenceDescriptor:
        base = f"captures/{name}"
        inode = 7000 + pid
        leaves = {
            "body": tree.put(f"{base}/raw/000000.metrics.json", metrics()),
            "tcp_before": tree.put(f"{base}/raw/000000.proc-net-tcp-before", proc_net_tcp(port, inode)),
            "tcp_after": tree.put(f"{base}/raw/000000.proc-net-tcp-after", proc_net_tcp(port, inode)),
            "sockets_before": tree.put(
                f"{base}/raw/000000.proc-fd-sockets-before.json",
                {"schema_version": checker.SOCKET_SNAPSHOT_VERSION, "server_pid": pid, "socket_inodes": [inode]},
            ),
            "sockets_after": tree.put(
                f"{base}/raw/000000.proc-fd-sockets-after.json",
                {"schema_version": checker.SOCKET_SNAPSHOT_VERSION, "server_pid": pid, "socket_inodes": [inode]},
            ),
            "stat_before": tree.put(f"{base}/raw/000000.proc-stat-before", proc_stat(pid, ticks)),
            "stat": tree.put(f"{base}/raw/000000.proc-stat", proc_stat(pid, ticks)),
            "status": tree.put(
                f"{base}/raw/000000.proc-status",
                f"Name:\triley\nPid:\t{pid}\n".encode("ascii"),
            ),
            "selection": tree.put(
                f"{base}/raw/000000.gpu-index-uuid.csv",
                f"0, {self.gpu_uuid}\n".encode("ascii"),
            ),
            "apps": tree.put(f"{base}/raw/000000.gpu-compute-apps.csv", f"{pid}, 42\n".encode("ascii")),
        }
        sample = {
            "schema_version": checker.OBSERVATION_SAMPLE_VERSION,
            "sequence": 0,
            "elapsed_monotonic_millis": 0,
            "endpoint": {
                "http_status": 200,
                "body": leaves["body"].as_json(),
                "listener": {
                    "address": "127.0.0.1",
                    "port": port,
                    "socket_inode": inode,
                    "before_proc_net_tcp": leaves["tcp_before"].as_json(),
                    "after_proc_net_tcp": leaves["tcp_after"].as_json(),
                    "before_server_fd_sockets": leaves["sockets_before"].as_json(),
                    "after_server_fd_sockets": leaves["sockets_after"].as_json(),
                },
            },
            "process": {
                "pid": pid,
                "start_ticks": ticks,
                "present": True,
                "pre_endpoint_stat": leaves["stat_before"].as_json(),
                "stat": leaves["stat"].as_json(),
                "status": leaves["status"].as_json(),
            },
            "gpu": {
                "index": 0,
                "uuid": self.gpu_uuid,
                "selection_query": leaves["selection"].as_json(),
                "compute_apps": leaves["apps"].as_json(),
            },
        }
        sample_descriptor = tree.put(f"{base}/samples/000000.json", sample)
        return tree.put(
            f"{base}/session.json",
            {
                "schema_version": checker.OBSERVATION_SESSION_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "endpoint": {
                    "url": f"http://127.0.0.1:{port}/v1/c02/metrics",
                    "expected_schema_version": checker.METRICS_VERSION,
                },
                "target": self.target(pid, ticks),
                "samples": [sample_descriptor.as_json()],
            },
        )

    def phase(self, tree: EvidenceTree, name: str, *, pid: int, ticks: int, port: int) -> dict:
        session = self.session(tree, name, pid=pid, ticks=ticks, port=port)
        base = f"{name}-phase"
        return {
            "target": self.target(pid, ticks),
            "observation_session": session.as_json(),
            "request_ledger": tree.put(f"{base}/request-ledger.ndjson", b"raw request bytes\n").as_json(),
            "runtime_event_log": tree.put(f"{base}/runtime-events.ndjson", b"raw audit bytes\n").as_json(),
            "generation_audit_index": tree.put(f"{base}/generation-audit-index.json", {"index": []}).as_json(),
        }

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def test_soak_binds_max_performance_exact_fallback_raw_leaves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            contract = tree.put("contracts/soak-v2.json", b"reviewed contract bytes\n")
            fallback = tree.put("soak-phase/fallback-events.ndjson", b"native fallback event bytes\n")
            scenario = {
                "scenario_id": "exact-backend-fallback",
                **phase,
                "fallback_event_log": fallback.as_json(),
            }
            manifest = tree.put(
                "soak/raw-manifest.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.max_performance_bindings,
                    "configuration_evidence": self.configuration_evidence(
                        tree, self.max_performance_bindings
                    ),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [scenario],
                },
            )
            report = checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
            self.assertEqual(report["schema_version"], checker.SOAK_REPORT_VERSION)
            self.assertEqual(report["status"], "bound")
            self.assertEqual(report["qualification_status"], "not-run")
            self.assertEqual(
                report["bindings"]["configuration_profile"],
                checker.MAX_PERFORMANCE_EXACT_PROFILE,
            )
            self.assertEqual(report["targets"][0]["target"], self.target(1111, 2222))

    def test_config_bridge_rejects_body_tuple_and_cross_capture_drift(self) -> None:
        def check(mutator: object, reason: str, *, phase_pid: int = 1111) -> None:
            with tempfile.TemporaryDirectory() as temporary:
                tree = EvidenceTree(Path(temporary))
                phase = self.phase(tree, "soak", pid=phase_pid, ticks=2222, port=8080)
                config = self.configuration_evidence(tree, self.bindings)
                bridge_path = config["endpoint_observation"]["path"]
                assert isinstance(bridge_path, str)
                bridge = json.loads((tree.root / bridge_path).read_bytes())
                assert callable(mutator)
                mutator(tree, bridge)
                config["endpoint_observation"] = tree.put(bridge_path, bridge).as_json()
                contract = tree.put("contracts/soak-v2.json", b"contract")
                manifest = tree.put(
                    "soak/bridge-drift.json",
                    {
                        "schema_version": checker.SOAK_MANIFEST_VERSION,
                        "capture_status": "captured",
                        "qualification_status": "not-run",
                        "candidate_id": self.candidate,
                        "bindings": self.bindings,
                        "configuration_evidence": config,
                        "scenario_contract": contract.as_json(),
                        "scenarios": [{"scenario_id": "bridge", **phase, "fallback_event_log": None}],
                    },
                )
                with self.assertRaises(checker.C02ProvenanceError) as raised:
                    checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
                self.assert_reason(raised, reason)

        check(lambda _tree, bridge: bridge["endpoint"].__setitem__("body_sha256", "f" * 64), "config-endpoint-body-hash-mismatch")
        check(lambda _tree, bridge: bridge["endpoint"].__setitem__("body_byte_length", 1), "config-endpoint-body-length-mismatch")
        check(lambda _tree, bridge: bridge["process"].__setitem__("server_pid", 9999), "config-process-target-mismatch")
        check(lambda _tree, bridge: bridge["endpoint"]["listener"].__setitem__("socket_inode", 9999), "config-listener-target-mismatch")
        check(lambda _tree, bridge: bridge["endpoint"]["listener"].__setitem__("port", 8081), "config-listener-target-mismatch")
        check(lambda _tree, bridge: bridge["gpu"].__setitem__("uuid", "GPU-deadbeef"), "config-gpu-target-mismatch")
        check(lambda _tree, bridge: bridge["target"].__setitem__("listener_port", 80), "invalid-listener-port")
        check(lambda _tree, bridge: bridge["target"].__setitem__("listener_port", 1023), "invalid-listener-port")
        check(lambda _tree, bridge: bridge["endpoint"]["listener"].__setitem__("port", 80), "invalid-listener-port")
        check(lambda _tree, bridge: bridge["endpoint"]["listener"].__setitem__("port", 1023), "invalid-listener-port")
        check(lambda _tree, _bridge: None, "configuration-scenario-target-mismatch", phase_pid=1112)

        def inject_other_capture(tree: EvidenceTree, bridge: dict) -> None:
            raw = (tree.root / bridge["endpoint"]["request"]["path"]).read_bytes()
            bridge["endpoint"]["request"] = tree.put("other-capture/raw/request.http", raw).as_json()

        check(inject_other_capture, "config-observation-layout-mismatch")

    def test_config_bridge_rejects_transfer_encoding_and_incomplete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            config = self.configuration_evidence(tree, self.bindings)
            bridge_path = config["endpoint_observation"]["path"]
            assert isinstance(bridge_path, str)
            bridge = json.loads((tree.root / bridge_path).read_bytes())
            head_path = bridge["endpoint"]["response_head"]["path"]
            assert isinstance(head_path, str)
            original_head = (tree.root / head_path).read_bytes()
            head = original_head.replace(
                b"Content-Type: application/json\r\n", b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n"
            )
            bridge["endpoint"]["response_head"] = tree.put(head_path, head).as_json()
            config["endpoint_observation"] = tree.put(bridge_path, bridge).as_json()
            contract = tree.put("contracts/soak-v2.json", b"contract")
            manifest = tree.put("soak/transfer.json", {"schema_version": checker.SOAK_MANIFEST_VERSION, "capture_status": "captured", "qualification_status": "not-run", "candidate_id": self.candidate, "bindings": self.bindings, "configuration_evidence": config, "scenario_contract": contract.as_json(), "scenarios": [{"scenario_id": "bridge", **phase, "fallback_event_log": None}]})
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
            self.assert_reason(raised, "invalid-config-response-head")

            bridge["endpoint"]["response_head"] = tree.put(head_path, original_head).as_json()
            config["endpoint_observation"] = tree.put(bridge_path, bridge).as_json()
            manifest = tree.put("soak/incomplete.json", {**json.loads((tree.root / manifest.path).read_bytes()), "configuration_evidence": config})
            marker = tree.root / "captures/config-bridge/capture-incomplete.json"
            marker.write_bytes(common.canonical_json_bytes({"capture_status": "incomplete"}))
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
            self.assert_reason(raised, "incomplete-capture")

    def test_metrics_are_structural_but_validate_all_declared_value_types(self) -> None:
        raw_metrics = metrics()
        raw_metrics["request_states"]["failed"] = 7
        raw_metrics["kv_blocks"] = {"free": 0, "reserved": 0, "active": 0}
        raw_metrics["allocation"] = {
            "device_live_count": 1,
            "device_live_bytes": 2,
            "pinned_live_count": 3,
            "pinned_live_bytes": 4,
        }
        raw_metrics["quiescence"] = {
            "completion_outbox": 5,
            "outstanding_iterations": 6,
            "riley_owned_live_allocations": 7,
            "worker_accepting": True,
            "scheduler_accepting": True,
        }
        checker._metrics(common.canonical_json_bytes(raw_metrics), "raw metrics")

        invalid_allocation = metrics()
        invalid_allocation["allocation"]["device_live_count"] = True
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker._metrics(common.canonical_json_bytes(invalid_allocation), "invalid allocation")
        self.assert_reason(raised, "invalid-integer")

        invalid_quiescence = metrics()
        invalid_quiescence["quiescence"]["worker_accepting"] = 1
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker._metrics(common.canonical_json_bytes(invalid_quiescence), "invalid quiescence")
        self.assert_reason(raised, "invalid-boolean")

    def test_rejects_empty_descriptor_and_invalid_scenario_identifier(self) -> None:
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker._descriptor(
                {"path": "raw/empty", "sha256": "1" * 64, "byte_length": 0},
                "empty descriptor",
            )
        self.assert_reason(raised, "empty-evidence-leaf")

        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            contract = tree.put("contracts/soak-v2.json", b"contract")
            manifest = tree.put(
                "soak/invalid-scenario.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.bindings,
                    "configuration_evidence": self.configuration_evidence(tree, self.bindings),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [{"scenario_id": "Not_A_Scenario", **phase, "fallback_event_log": None}],
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
            self.assert_reason(raised, "invalid-scenario-inventory")

    def test_soak_rejects_status_pid_and_elapsed_monotonic_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            session_path = tree.root / phase["observation_session"]["path"]
            session_document = json.loads(session_path.read_bytes())
            sample_path = tree.root / session_document["samples"][0]["path"]
            sample_document = json.loads(sample_path.read_bytes())
            bad_status = tree.put(
                "captures/soak/raw/000000.proc-status-drift",
                b"Name:\triley\nPid:\t9999\n",
            )
            sample_document["process"]["status"] = bad_status.as_json()
            bad_sample = tree.put("captures/soak/samples/000000-status-drift.json", sample_document)
            session_document["samples"] = [bad_sample.as_json()]
            bad_session = tree.put("captures/soak/session-status-drift.json", session_document)
            phase["observation_session"] = bad_session.as_json()
            contract = tree.put("contracts/soak-v2.json", b"contract")
            manifest = tree.put(
                "soak/status-drift.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.bindings,
                    "configuration_evidence": self.configuration_evidence(tree, self.bindings),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [{"scenario_id": "status-pid", **phase, "fallback_event_log": None}],
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
            self.assert_reason(raised, "pid-start-tick-mismatch")

            phase = self.phase(tree, "elapsed", pid=2222, ticks=3333, port=8081)
            session_path = tree.root / phase["observation_session"]["path"]
            session_document = json.loads(session_path.read_bytes())
            first_sample_path = tree.root / session_document["samples"][0]["path"]
            first_sample = json.loads(first_sample_path.read_bytes())

            def clone_raw_descriptors(value: object) -> object:
                if isinstance(value, dict):
                    if set(value) == {"path", "sha256", "byte_length"}:
                        source_path = value["path"]
                        assert isinstance(source_path, str)
                        copied_path = source_path.replace("000000", "000001", 1)
                        self.assertNotEqual(copied_path, source_path)
                        return tree.put(copied_path, (tree.root / source_path).read_bytes()).as_json()
                    return {key: clone_raw_descriptors(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [clone_raw_descriptors(item) for item in value]
                return value

            second_sample = clone_raw_descriptors(first_sample)
            assert isinstance(second_sample, dict)
            second_sample["sequence"] = 1
            second_sample["elapsed_monotonic_millis"] = 0
            second_descriptor = tree.put("captures/elapsed/samples/000001.json", second_sample)
            session_document["samples"].append(second_descriptor.as_json())
            elapsed_session = tree.put("captures/elapsed/session-elapsed-drift.json", session_document)
            phase["observation_session"] = elapsed_session.as_json()
            elapsed_manifest = tree.put(
                "soak/elapsed-drift.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.bindings,
                    "configuration_evidence": self.configuration_evidence(
                        tree, self.bindings, pid=2222, ticks=3333, port=8081
                    ),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [{"scenario_id": "elapsed-order", **phase, "fallback_event_log": None}],
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), elapsed_manifest.path)
            self.assert_reason(raised, "sample-elapsed-not-increasing")

    def test_cli_emits_canonical_bound_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            contract = tree.put("contracts/soak-v2.json", b"contract")
            manifest = tree.put(
                "raw-manifest.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.bindings,
                    "configuration_evidence": self.configuration_evidence(tree, self.bindings),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [{"scenario_id": "cli-output", **phase, "fallback_event_log": None}],
                },
            )
            self.complete_soak_manifest(tree, manifest)
            output_bytes = io.BytesIO()
            output = io.TextIOWrapper(output_bytes, encoding="utf-8")
            with mock.patch.object(checker.sys, "stdout", output):
                self.assertEqual(
                    checker.main(
                        [
                            "--evidence-root",
                            str(tree.root.resolve()),
                            "--kind",
                            "soak",
                            "--manifest",
                            manifest.path,
                        ]
                    ),
                    0,
                )
            output.flush()
            raw_report = output_bytes.getvalue()
            self.assertEqual(raw_report, common.canonical_json_bytes(json.loads(raw_report)) + b"\n")

    def test_soak_rejects_raw_gpu_index_uuid_tuple_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            session_path = tree.root / phase["observation_session"]["path"]
            session_document = json.loads(session_path.read_bytes())
            sample_path = tree.root / session_document["samples"][0]["path"]
            sample_document = json.loads(sample_path.read_bytes())
            bad_selection = tree.put(
                "captures/soak/raw/000000.gpu-index-uuid-drift.csv",
                f"1, {self.gpu_uuid}\n".encode("ascii"),
            )
            sample_document["gpu"]["selection_query"] = bad_selection.as_json()
            bad_sample = tree.put("captures/soak/samples/000000-drift.json", sample_document)
            session_document["samples"] = [bad_sample.as_json()]
            bad_session = tree.put("captures/soak/session-drift.json", session_document)
            phase["observation_session"] = bad_session.as_json()
            contract = tree.put("contracts/soak-v2.json", b"contract")
            manifest = tree.put(
                "soak/gpu-drift.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.bindings,
                    "configuration_evidence": self.configuration_evidence(tree, self.bindings),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [{"scenario_id": "gpu-tuple", **phase, "fallback_event_log": None}],
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
            self.assert_reason(raised, "gpu-tuple-mismatch")

    def test_soak_rejects_historical_v1_and_missing_fallback_raw_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            contract = tree.put("contracts/soak-v2.json", b"contract")
            v1 = tree.put(
                "soak/v1.json",
                {
                    "schema_version": "riley.soak-v2-receipt.v1",
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.bindings,
                    "configuration_evidence": self.configuration_evidence(tree, self.bindings),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [],
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), v1.path)
            self.assert_reason(raised, "historical-soak-v1-rejected")

            historical_v2 = tree.put(
                "soak/historical-v2.json",
                {
                    **json.loads((tree.root / v1.path).read_bytes()),
                    "schema_version": "riley.soak-v2-raw-provenance.v2",
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), historical_v2.path)
            self.assert_reason(raised, "historical-soak-v2-rejected")

            missing_fallback = tree.put(
                "soak/v2.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.max_performance_bindings,
                    "configuration_evidence": self.configuration_evidence(
                        tree, self.max_performance_bindings
                    ),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [{"scenario_id": "exact-backend-fallback", **phase, "fallback_event_log": None}],
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), missing_fallback.path)
            self.assert_reason(raised, "fallback-raw-leaf-missing")

    def test_soak_rejects_exact_fallback_under_stable_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            contract = tree.put("contracts/soak-v2.json", b"contract")
            fallback = tree.put("soak-phase/fallback.ndjson", b"native fallback event")
            manifest = tree.put(
                "soak/stable-fallback.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.bindings,
                    "configuration_evidence": self.configuration_evidence(tree, self.bindings),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [
                        {
                            "scenario_id": "exact-backend-fallback",
                            **phase,
                            "fallback_event_log": fallback.as_json(),
                        }
                    ],
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
            self.assert_reason(raised, "fallback-profile-mismatch")

    def test_rejects_nonprivate_evidence_root_before_reading_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            os.chmod(root, 0o755)
            try:
                with self.assertRaises(checker.C02ProvenanceError) as raised:
                    checker.verify_soak_provenance(root, "soak/raw-manifest.json")
                self.assert_reason(raised, "unsafe-evidence-root-mode")
            finally:
                os.chmod(root, 0o700)

    def test_soak_rejects_an_unremoved_capture_marker_and_session_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            phase = self.phase(tree, "soak", pid=1111, ticks=2222, port=8080)
            contract = tree.put("contracts/soak-v2.json", b"contract")
            fallback = tree.put("soak-phase/fallback.ndjson", b"event")
            manifest = tree.put(
                "soak/v2.json",
                {
                    "schema_version": checker.SOAK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.max_performance_bindings,
                    "configuration_evidence": self.configuration_evidence(
                        tree, self.max_performance_bindings
                    ),
                    "scenario_contract": contract.as_json(),
                    "scenarios": [{"scenario_id": "exact-backend-fallback", **phase, "fallback_event_log": fallback.as_json()}],
                },
            )
            tree.put("captures/soak/capture-incomplete.json", {"capture_status": "incomplete"})
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_soak_provenance(tree.root.resolve(), manifest.path)
            self.assert_reason(raised, "incomplete-capture")

    def test_rollback_binds_shutdown_marker_and_rejects_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            candidate_phase = self.phase(tree, "candidate", pid=1111, ticks=2222, port=8080)
            shutdown = tree.put(
                "candidate-phase/shutdown.json",
                {
                    "schema_version": checker.SHUTDOWN_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "server_pid": 1111,
                    "server_start_ticks": 2222,
                    "worker_ready": False,
                    "final_metrics": metrics(),
                },
            )
            marker = tree.put(
                "candidate-phase/shutdown.json.complete",
                {
                    "schema_version": checker.SHUTDOWN_MARKER_VERSION,
                    "artifact_filename": "shutdown.json",
                    "artifact_sha256": shutdown.sha256,
                },
            )
            candidate_phase["shutdown_artifact"] = shutdown.as_json()
            candidate_phase["shutdown_marker"] = marker.as_json()
            rollback_phase = self.phase(tree, "rollback", pid=3333, ticks=4444, port=8081)
            candidate_artifacts = {
                key: tree.put(f"candidate-artifacts/{key}", f"candidate {key}".encode("ascii")).as_json()
                for key in ("binary", "bundle", "image_inspect")
            }
            rollback_artifacts = {
                key: tree.put(f"rollback-artifacts/{key}", f"rollback {key}".encode("ascii")).as_json()
                for key in ("binary", "bundle", "image_inspect")
            }
            atomic_switch = {
                key: tree.put(f"atomic/{key}.raw", f"raw {key}".encode("ascii")).as_json()
                for key in ("pre_active_stat", "post_active_stat", "candidate_staged_stat", "rollback_staged_stat", "rename_transcript")
            }
            manifest_document = {
                "schema_version": checker.ROLLBACK_MANIFEST_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "candidate_id": self.candidate,
                "bindings": self.bindings,
                "candidate": candidate_phase,
                "rollback": rollback_phase,
                "candidate_artifacts": candidate_artifacts,
                "rollback_artifacts": rollback_artifacts,
                "atomic_switch": atomic_switch,
            }
            manifest = tree.put("rollback/raw-manifest.json", manifest_document)
            report = checker.verify_rollback_provenance(tree.root.resolve(), manifest.path)
            self.assertEqual(report["schema_version"], checker.ROLLBACK_REPORT_VERSION)
            self.assertEqual(report["targets"][0]["target"], self.target(1111, 2222))

            manifest_document["schema_version"] = "riley.rc3-rollback-receipt.v1"
            historical = tree.put("rollback/historical-v1.json", manifest_document)
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_rollback_provenance(tree.root.resolve(), historical.path)
            self.assert_reason(raised, "historical-rollback-v1-rejected")

            max_profile = tree.put(
                "rollback/max-profile.json",
                {
                    **manifest_document,
                    "schema_version": checker.ROLLBACK_MANIFEST_VERSION,
                    "bindings": self.max_performance_bindings,
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_rollback_provenance(tree.root.resolve(), max_profile.path)
            self.assert_reason(raised, "invalid-configuration-profile")

    def test_shutdown_marker_descriptor_path_and_filename_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            candidate_phase = self.phase(tree, "candidate", pid=1111, ticks=2222, port=8080)
            rollback_phase = self.phase(tree, "rollback", pid=3333, ticks=4444, port=8081)
            candidate_artifacts = {
                key: tree.put(f"candidate-artifacts/{key}", f"candidate {key}".encode("ascii")).as_json()
                for key in ("binary", "bundle", "image_inspect")
            }
            rollback_artifacts = {
                key: tree.put(f"rollback-artifacts/{key}", f"rollback {key}".encode("ascii")).as_json()
                for key in ("binary", "bundle", "image_inspect")
            }
            atomic_switch = {
                key: tree.put(f"atomic/{key}.raw", f"raw {key}".encode("ascii")).as_json()
                for key in ("pre_active_stat", "post_active_stat", "candidate_staged_stat", "rollback_staged_stat", "rename_transcript")
            }
            shutdown_document = {
                "schema_version": checker.SHUTDOWN_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "server_pid": 1111,
                "server_start_ticks": 2222,
                "worker_ready": False,
                "final_metrics": metrics(),
            }

            def completion_marker(
                path: str,
                artifact: common.EvidenceDescriptor,
                artifact_filename: str,
            ) -> common.EvidenceDescriptor:
                return tree.put(
                    path,
                    {
                        "schema_version": checker.SHUTDOWN_MARKER_VERSION,
                        "artifact_filename": artifact_filename,
                        "artifact_sha256": artifact.sha256,
                    },
                )

            def manifest(
                name: str,
                artifact: common.EvidenceDescriptor,
                marker: common.EvidenceDescriptor,
            ) -> common.EvidenceDescriptor:
                phase = {
                    **candidate_phase,
                    "shutdown_artifact": artifact.as_json(),
                    "shutdown_marker": marker.as_json(),
                }
                return tree.put(
                    f"rollback/{name}.json",
                    {
                        "schema_version": checker.ROLLBACK_MANIFEST_VERSION,
                        "capture_status": "captured",
                        "qualification_status": "not-run",
                        "candidate_id": self.candidate,
                        "bindings": self.bindings,
                        "candidate": phase,
                        "rollback": rollback_phase,
                        "candidate_artifacts": candidate_artifacts,
                        "rollback_artifacts": rollback_artifacts,
                        "atomic_switch": atomic_switch,
                    },
                )

            shutdown = tree.put("candidate-phase/shutdown.json", shutdown_document)
            nested_marker = completion_marker(
                "candidate-phase/nested/shutdown.json.complete",
                shutdown,
                "shutdown.json",
            )
            hidden_shutdown = tree.put("candidate-phase/.shutdown.json", shutdown_document)
            hidden_marker = completion_marker(
                "candidate-phase/.shutdown.json.complete",
                hidden_shutdown,
                ".shutdown.json",
            )
            filename_shutdown = tree.put("candidate-phase/shutdown-filename.json", shutdown_document)
            filename_marker = completion_marker(
                "candidate-phase/shutdown-filename.json.complete",
                filename_shutdown,
                "other-shutdown.json",
            )

            for name, artifact, marker, reason in (
                ("marker-nested", shutdown, nested_marker, "shutdown-marker-path-mismatch"),
                ("marker-hidden", hidden_shutdown, hidden_marker, "shutdown-marker-path-mismatch"),
                ("marker-filename", filename_shutdown, filename_marker, "shutdown-marker-mismatch"),
            ):
                with self.subTest(name=name):
                    with self.assertRaises(checker.C02ProvenanceError) as raised:
                        checker.verify_rollback_provenance(
                            tree.root.resolve(), manifest(name, artifact, marker).path
                        )
                    self.assert_reason(raised, reason)

    def test_shutdown_marker_mismatch_and_symlink_manifest_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tree = EvidenceTree(Path(temporary))
            candidate_phase = self.phase(tree, "candidate", pid=1111, ticks=2222, port=8080)
            shutdown = tree.put(
                "candidate-phase/shutdown.json",
                {
                    "schema_version": checker.SHUTDOWN_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "server_pid": 1111,
                    "server_start_ticks": 2222,
                    "worker_ready": False,
                    "final_metrics": metrics(),
                },
            )
            marker = tree.put(
                "candidate-phase/shutdown.json.complete",
                {
                    "schema_version": checker.SHUTDOWN_MARKER_VERSION,
                    "artifact_filename": "shutdown.json",
                    "artifact_sha256": "f" * 64,
                },
            )
            candidate_phase.update({"shutdown_artifact": shutdown.as_json(), "shutdown_marker": marker.as_json()})
            rollback_phase = self.phase(tree, "rollback", pid=3333, ticks=4444, port=8081)
            artifact_map = lambda prefix: {
                key: tree.put(f"{prefix}/{key}", key.encode("ascii")).as_json()
                for key in ("binary", "bundle", "image_inspect")
            }
            atomic_switch = {
                key: tree.put(f"atomic/{key}", key.encode("ascii")).as_json()
                for key in ("pre_active_stat", "post_active_stat", "candidate_staged_stat", "rollback_staged_stat", "rename_transcript")
            }
            bad = tree.put(
                "rollback/bad-marker.json",
                {
                    "schema_version": checker.ROLLBACK_MANIFEST_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "candidate_id": self.candidate,
                    "bindings": self.bindings,
                    "candidate": candidate_phase,
                    "rollback": rollback_phase,
                    "candidate_artifacts": artifact_map("candidate-artifacts"),
                    "rollback_artifacts": artifact_map("rollback-artifacts"),
                    "atomic_switch": atomic_switch,
                },
            )
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_rollback_provenance(tree.root.resolve(), bad.path)
            self.assert_reason(raised, "shutdown-marker-mismatch")

            link = tree.root / "rollback" / "symlink.json"
            link.symlink_to("bad-marker.json")
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_rollback_provenance(tree.root.resolve(), "rollback/symlink.json")
            self.assertEqual(getattr(raised.exception, "reason_code", None), "unsafe-evidence-path")

    def test_schema_documents_are_valid_json(self) -> None:
        directory = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
        )
        for name in (
            "soak-v2-receipt-v2.schema.json",
            "soak-v2-receipt-v3.schema.json",
            "soak-v2-receipt-v4.schema.json",
            "soak-v2-bind-request-v4.schema.json",
            "c02-raw-scenario-capture-v1.schema.json",
            "rollback-receipt-v2.schema.json",
            "c02-config-endpoint-observation-v1.schema.json",
        ):
            with self.subTest(name=name):
                document = json.loads((directory / name).read_text(encoding="utf-8"))
                self.assertEqual(document["$schema"], "https://json-schema.org/draft/2020-12/schema")
                if "descriptor" in document["$defs"]:
                    self.assertEqual(
                        document["$defs"]["descriptor"]["properties"]["byte_length"]["minimum"],
                        1,
                    )

        soak_schema = json.loads(
            (directory / "soak-v2-receipt-v3.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            soak_schema["$defs"]["bindings"]["properties"]["configuration_profile"]["enum"],
            [checker.STABLE_DEFAULT_PROFILE, checker.MAX_PERFORMANCE_EXACT_PROFILE],
        )
        self.assertEqual(
            soak_schema["$defs"]["rawManifest"]["properties"]["configuration_evidence"],
            {"$ref": "#/$defs/configurationEvidence"},
        )
        self.assertEqual(
            soak_schema["$defs"]["configurationEvidence"]["required"],
            ["endpoint", "startup_artifact", "endpoint_observation"],
        )
        self.assertEqual(
            soak_schema["$defs"]["completion"]["properties"][
                "schema_version"
            ],
            {"const": checker.SOAK_COMPLETION_MARKER_VERSION},
        )
        fallback_condition = soak_schema["$defs"]["scenario"]["allOf"][0]
        self.assertEqual(
            fallback_condition["then"]["properties"]["fallback_event_log"],
            {"$ref": "#/$defs/descriptor"},
        )
        profile_condition = soak_schema["$defs"]["rawManifest"]["allOf"][0]
        self.assertEqual(
            profile_condition["then"]["properties"]["bindings"]["properties"][
                "configuration_profile"
            ],
            {"const": checker.MAX_PERFORMANCE_EXACT_PROFILE},
        )
        self.assertEqual(
            soak_schema["$defs"]["configurationEvidence"]["properties"]["endpoint"],
            {"$ref": "#/$defs/configEndpointDescriptor"},
        )
        bind_request_schema = json.loads(
            (directory / "soak-v2-bind-request-v3.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            bind_request_schema["properties"]["schema_version"],
            {"const": "riley.soak-v2-bind-request.v3"},
        )
        self.assertEqual(
            bind_request_schema["$defs"]["configurationEvidence"]["required"],
            ["endpoint_path", "startup_artifact_path", "endpoint_observation_path"],
        )
        self.assertEqual(
            bind_request_schema["$defs"]["relativePath"]["maxLength"],
            checker.MAX_RELATIVE_PATH_BYTES,
        )
        self.assertEqual(
            bind_request_schema["$defs"]["scenario"]["properties"][
                "fallback_event_log_path"
            ]["oneOf"][0],
            {"type": "null"},
        )
        for schema in (soak_schema, bind_request_schema):
            relative_path_pattern = schema["$defs"]["relativePath"]["pattern"]
            self.assertIsNone(re.fullmatch(relative_path_pattern, "a/."))
            self.assertIsNone(re.fullmatch(relative_path_pattern, "a/.."))
            self.assertIsNotNone(re.fullmatch(relative_path_pattern, "a/.hidden"))

        soak_v4_schema = json.loads(
            (directory / "soak-v2-receipt-v4.schema.json").read_text(encoding="utf-8")
        )
        bind_request_v4_schema = json.loads(
            (directory / "soak-v2-bind-request-v4.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            soak_v4_schema["$defs"]["rawManifest"]["properties"]["schema_version"],
            {"const": checker.SOAK_V4_MANIFEST_VERSION},
        )
        self.assertEqual(
            soak_v4_schema["$defs"]["completion"]["properties"]["schema_version"],
            {"const": checker.SOAK_V4_COMPLETION_MARKER_VERSION},
        )
        self.assertEqual(
            bind_request_v4_schema["properties"]["schema_version"],
            {"const": "riley.soak-v2-bind-request.v4"},
        )
        self.assertEqual(
            set(bind_request_v4_schema["required"]),
            {
                "schema_version",
                "candidate_id",
                "binding_inputs",
                "configuration_evidence",
                "scenario_capture_session_path",
                "scenarios",
            },
        )
        self.assertEqual(
            set(bind_request_v4_schema["$defs"]["scenario"]["required"]),
            {"scenario_id", "observation_session_path"},
        )


if __name__ == "__main__":
    unittest.main()
