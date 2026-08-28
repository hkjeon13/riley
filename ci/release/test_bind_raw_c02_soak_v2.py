#!/usr/bin/env python3
"""Hostile, CPU-only tests for the raw-only C02 soak manifest binder."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bind_raw_c02_soak_v2 as binder
import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common


_UNSET = object()


class EvidenceTree:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, relative: str, value: bytes | dict) -> bytes:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = common.canonical_json_bytes(value) if isinstance(value, dict) else value
        path.write_bytes(raw)
        return raw


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


def _proc_stat(pid: int, ticks: int) -> bytes:
    fields = ["S", *("0" for _ in range(18)), str(ticks), "0"]
    return f"{pid} (riley server) {' '.join(fields)}\n".encode("ascii")


def _proc_tcp(port: int, inode: int) -> bytes:
    return (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        f"   0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 {inode} 1\n"
    ).encode("ascii")


class BindRawC02SoakV2Tests(unittest.TestCase):
    candidate_id = "riley-0.1.0-rc3"
    gpu_uuid = "GPU-12345678-abcd-efab-cdef-1234567890ab"
    configuration_sha256 = "c" * 64

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "evidence"
        self.root.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        self.tree = EvidenceTree(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _target(self) -> dict:
        return {
            "server_pid": 1111,
            "server_start_ticks": 2222,
            "gpu_index": 0,
            "gpu_uuid": self.gpu_uuid,
        }

    def _session(self, name: str, *, port: int = 18080) -> str:
        target = self._target()
        pid = target["server_pid"]
        ticks = target["server_start_ticks"]
        inode = 7001
        base = f"captures/{name}"
        raw = {
            "metrics": self.tree.put(f"{base}/raw/000000.metrics.json", _metrics()),
            "tcp_before": self.tree.put(
                f"{base}/raw/000000.proc-net-tcp-before", _proc_tcp(port, inode)
            ),
            "tcp_after": self.tree.put(
                f"{base}/raw/000000.proc-net-tcp-after", _proc_tcp(port, inode)
            ),
            "sockets_before": self.tree.put(
                f"{base}/raw/000000.proc-fd-sockets-before.json",
                {
                    "schema_version": checker.SOCKET_SNAPSHOT_VERSION,
                    "server_pid": pid,
                    "socket_inodes": [inode],
                },
            ),
            "sockets_after": self.tree.put(
                f"{base}/raw/000000.proc-fd-sockets-after.json",
                {
                    "schema_version": checker.SOCKET_SNAPSHOT_VERSION,
                    "server_pid": pid,
                    "socket_inodes": [inode],
                },
            ),
            "stat_before": self.tree.put(
                f"{base}/raw/000000.proc-stat-before", _proc_stat(pid, ticks)
            ),
            "stat": self.tree.put(f"{base}/raw/000000.proc-stat", _proc_stat(pid, ticks)),
            "status": self.tree.put(
                f"{base}/raw/000000.proc-status",
                f"Name:\triley\nPid:\t{pid}\n".encode("ascii"),
            ),
            "selection": self.tree.put(
                f"{base}/raw/000000.gpu-index-uuid.csv",
                f"0, {self.gpu_uuid}\n".encode("ascii"),
            ),
            "apps": self.tree.put(
                f"{base}/raw/000000.gpu-compute-apps.csv", f"{pid}, 42\n".encode("ascii")
            ),
        }
        descriptor = lambda relative, value: common.descriptor_for_bytes(
            relative, value, relative
        ).as_json()
        sample_path = f"{base}/samples/000000.json"
        sample = {
            "schema_version": checker.OBSERVATION_SAMPLE_VERSION,
            "sequence": 0,
            "elapsed_monotonic_millis": 0,
            "endpoint": {
                "http_status": 200,
                "body": descriptor(f"{base}/raw/000000.metrics.json", raw["metrics"]),
                "listener": {
                    "address": "127.0.0.1",
                    "port": port,
                    "socket_inode": inode,
                    "before_proc_net_tcp": descriptor(
                        f"{base}/raw/000000.proc-net-tcp-before", raw["tcp_before"]
                    ),
                    "after_proc_net_tcp": descriptor(
                        f"{base}/raw/000000.proc-net-tcp-after", raw["tcp_after"]
                    ),
                    "before_server_fd_sockets": descriptor(
                        f"{base}/raw/000000.proc-fd-sockets-before.json",
                        raw["sockets_before"],
                    ),
                    "after_server_fd_sockets": descriptor(
                        f"{base}/raw/000000.proc-fd-sockets-after.json",
                        raw["sockets_after"],
                    ),
                },
            },
            "process": {
                "pid": pid,
                "start_ticks": ticks,
                "present": True,
                "pre_endpoint_stat": descriptor(
                    f"{base}/raw/000000.proc-stat-before", raw["stat_before"]
                ),
                "stat": descriptor(f"{base}/raw/000000.proc-stat", raw["stat"]),
                "status": descriptor(f"{base}/raw/000000.proc-status", raw["status"]),
            },
            "gpu": {
                "index": 0,
                "uuid": self.gpu_uuid,
                "selection_query": descriptor(
                    f"{base}/raw/000000.gpu-index-uuid.csv", raw["selection"]
                ),
                "compute_apps": descriptor(
                    f"{base}/raw/000000.gpu-compute-apps.csv", raw["apps"]
                ),
            },
        }
        sample_raw = self.tree.put(sample_path, sample)
        session_path = f"{base}/session.json"
        self.tree.put(
            session_path,
            {
                "schema_version": checker.OBSERVATION_SESSION_VERSION,
                "capture_status": "captured",
                "qualification_status": "not-run",
                "endpoint": {
                    "url": f"http://127.0.0.1:{port}/v1/c02/metrics",
                    "expected_schema_version": checker.METRICS_VERSION,
                },
                "target": target,
                "samples": [descriptor(sample_path, sample_raw)],
            },
        )
        return session_path

    def _write_configuration(
        self,
        *,
        profile: str,
        candidate_id: str | None = None,
        configuration_sha256: str | None = None,
        startup_candidate_id: str | None = None,
        startup_identity: dict[str, str] | None = None,
        endpoint_payload_sha256: str | None = None,
        effective_hash: str | None = None,
    ) -> dict[str, str]:
        candidate = self.candidate_id if candidate_id is None else candidate_id
        config_sha = (
            self.configuration_sha256
            if configuration_sha256 is None
            else configuration_sha256
        )
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
        expected_effective_hash = hashlib.sha256(
            runtime_config.canonical_json_bytes(effective_config)
        ).hexdigest()
        identity = {
            "configuration_profile": profile,
            "configuration_sha256": config_sha,
        }
        endpoint = {
            "schema_version": runtime_config.ENDPOINT_VERSION,
            "candidate_id": candidate,
            "runtime_identity": identity,
            "effective_config": effective_config,
            "effective_config_sha256": (
                expected_effective_hash if effective_hash is None else effective_hash
            ),
        }
        endpoint_path = "config/endpoint.json"
        endpoint_raw = self.tree.put(endpoint_path, endpoint)
        startup = {
            "schema_version": runtime_config.STARTUP_ARTIFACT_VERSION,
            "created_at_utc": "2026-08-29T00:00:00Z",
            "candidate_id": candidate if startup_candidate_id is None else startup_candidate_id,
            "endpoint_path": "/v1/config",
            "runtime_identity": identity if startup_identity is None else startup_identity,
            "endpoint_payload_sha256": (
                hashlib.sha256(endpoint_raw).hexdigest()
                if endpoint_payload_sha256 is None
                else endpoint_payload_sha256
            ),
            "endpoint_payload": endpoint,
        }
        startup_path = "config/startup.json"
        self.tree.put(startup_path, startup)
        return {
            "endpoint_path": endpoint_path,
            "startup_artifact_path": startup_path,
        }

    def _write_bind_request(
        self,
        *,
        profile: str = checker.STABLE_DEFAULT_PROFILE,
        config_profile: str | None = None,
        include_fallback: bool = False,
        fallback_path: object = _UNSET,
        configuration: dict[str, str] | None = None,
    ) -> tuple[str, dict]:
        actual_profile = profile if config_profile is None else config_profile
        configuration_evidence = (
            self._write_configuration(profile=actual_profile)
            if configuration is None
            else configuration
        )
        session_path = self._session("normal")
        contract_path = "contracts/soak-v2.json"
        request_ledger_path = "workload/request-ledger.ndjson"
        runtime_event_path = "workload/runtime-events.ndjson"
        audit_index_path = "workload/generation-audit-index.json"
        self.tree.put(contract_path, b"reviewed scenario contract\n")
        self.tree.put(request_ledger_path, b"request raw bytes\n")
        self.tree.put(runtime_event_path, b"runtime raw bytes\n")
        self.tree.put(audit_index_path, b"generation audit raw bytes\n")
        if fallback_path is _UNSET:
            fallback_path = "workload/fallback-events.ndjson" if include_fallback else None
        if fallback_path is not None:
            assert isinstance(fallback_path, str)
            self.tree.put(fallback_path, b"native fallback raw event\n")
        scenario_id = "exact-backend-fallback" if include_fallback else "normal"
        request = {
            "schema_version": binder.BIND_REQUEST_VERSION,
            "candidate_id": self.candidate_id,
            "binding_inputs": {
                "freeze_sha256": "a" * 64,
                "base_release_candidate_report_sha256": "b" * 64,
                "configuration_profile": profile,
            },
            "configuration_evidence": configuration_evidence,
            "scenario_contract_path": contract_path,
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "target": self._target(),
                    "observation_session_path": session_path,
                    "request_ledger_path": request_ledger_path,
                    "runtime_event_log_path": runtime_event_path,
                    "generation_audit_index_path": audit_index_path,
                    "fallback_event_log_path": fallback_path,
                }
            ],
        }
        request_path = "requests/bind-request.json"
        self.tree.put(request_path, request)
        return request_path, request

    def _write_unmarked_manifest(self, manifest_name: str) -> None:
        request_path, _request = self._write_bind_request()
        root_fd = common.open_private_evidence_directory(self.root, "test evidence root")
        try:
            manifest = binder._manifest_from_request(root_fd, request_path, manifest_name)
            common.write_create_only_json(root_fd, manifest_name, manifest, "test manifest")
        finally:
            os.close(root_fd)

    def test_binds_stable_raw_config_and_writes_canonical_manifest_and_marker(self) -> None:
        request_path, _request = self._write_bind_request()
        report = binder.bind_raw_soak_manifest(self.root, request_path, "soak-manifest.json")

        manifest_raw = (self.root / "soak-manifest.json").read_bytes()
        manifest = common.parse_canonical_json(manifest_raw, "manifest")
        marker_raw = (self.root / "soak-manifest.json.complete").read_bytes()
        marker = common.parse_canonical_json(marker_raw, "marker")
        self.assertEqual(manifest_raw, common.canonical_json_bytes(manifest))
        self.assertEqual(marker_raw, common.canonical_json_bytes(marker))
        self.assertEqual(manifest["bindings"]["configuration_sha256"], self.configuration_sha256)
        self.assertEqual(
            manifest["configuration_evidence"]["endpoint"]["path"], "config/endpoint.json"
        )
        self.assertEqual(marker["schema_version"], checker.SOAK_COMPLETION_MARKER_VERSION)
        self.assertEqual(marker["artifact_filename"], "soak-manifest.json")
        self.assertEqual(marker["artifact_sha256"], hashlib.sha256(manifest_raw).hexdigest())
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(
            checker.verify_completed_soak_provenance(self.root, "soak-manifest.json"), report
        )

    def test_binds_max_performance_exact_fallback_without_a_semantic_verdict(self) -> None:
        request_path, _request = self._write_bind_request(
            profile=checker.MAX_PERFORMANCE_EXACT_PROFILE,
            include_fallback=True,
        )
        report = binder.bind_raw_soak_manifest(self.root, request_path, "max-fallback.json")
        manifest = common.parse_canonical_json(
            (self.root / "max-fallback.json").read_bytes(), "max fallback manifest"
        )
        self.assertEqual(
            manifest["bindings"]["configuration_profile"],
            checker.MAX_PERFORMANCE_EXACT_PROFILE,
        )
        self.assertIsNotNone(manifest["scenarios"][0]["fallback_event_log"])
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")

    def test_completed_verifier_rejects_absent_wrong_hidden_and_nested_markers(self) -> None:
        self._write_unmarked_manifest("absent.json")
        self.assertEqual(
            checker.verify_soak_provenance(self.root, "absent.json")["status"], "bound"
        )
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance(self.root, "absent.json")
        self.assert_reason(raised, "missing-soak-completion-marker")

        self._write_unmarked_manifest("wrong.json")
        root_fd = common.open_private_evidence_directory(self.root, "test evidence root")
        try:
            common.write_create_only_json(
                root_fd,
                "wrong.json.complete",
                {
                    "schema_version": checker.SOAK_COMPLETION_MARKER_VERSION,
                    "artifact_filename": "wrong.json",
                    "artifact_sha256": "f" * 64,
                },
                "wrong marker",
            )
            self.tree.put(
                ".hidden.json.complete",
                {
                    "schema_version": checker.SOAK_COMPLETION_MARKER_VERSION,
                    "artifact_filename": "hidden.json",
                    "artifact_sha256": "f" * 64,
                },
            )
            os.mkdir("markers", dir_fd=root_fd)
            markers_fd = os.open("markers", os.O_RDONLY | os.O_DIRECTORY, dir_fd=root_fd)
            try:
                common.write_create_only_json(
                    markers_fd,
                    "nested.json.complete",
                    {
                        "schema_version": checker.SOAK_COMPLETION_MARKER_VERSION,
                        "artifact_filename": "nested.json",
                        "artifact_sha256": "f" * 64,
                    },
                    "nested marker",
                )
            finally:
                os.close(markers_fd)
        finally:
            os.close(root_fd)
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance(self.root, "wrong.json")
        self.assert_reason(raised, "soak-completion-marker-mismatch")

        self._write_unmarked_manifest("hidden.json")
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance(self.root, "hidden.json")
        self.assert_reason(raised, "missing-soak-completion-marker")

        self._write_unmarked_manifest("nested.json")
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance(self.root, "nested.json")
        self.assert_reason(raised, "missing-soak-completion-marker")

    def test_rejects_request_config_and_fallback_drift_before_publication(self) -> None:
        request_path, request = self._write_bind_request(
            profile=checker.MAX_PERFORMANCE_EXACT_PROFILE,
            config_profile=checker.STABLE_DEFAULT_PROFILE,
            include_fallback=True,
        )
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "profile-drift.json")
        self.assert_reason(raised, "runtime-config-profile-mismatch")
        self.assertFalse((self.root / "profile-drift.json").exists())

        configuration = self._write_configuration(
            profile=checker.MAX_PERFORMANCE_EXACT_PROFILE
        )
        request_path, request = self._write_bind_request(
            profile=checker.MAX_PERFORMANCE_EXACT_PROFILE,
            include_fallback=True,
            configuration=configuration,
        )
        request["scenarios"][0]["scenario_id"] = "exact-backend-fallback"
        request["scenarios"][0]["fallback_event_log_path"] = None
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "missing-fallback.json")
        self.assert_reason(raised, "fallback-raw-leaf-missing")

        request["binding_inputs"]["configuration_sha256"] = "d" * 64
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "caller-hash.json")
        self.assert_reason(raised, "unexpected-field-set")

    def test_rejects_runtime_config_candidate_identity_digest_and_effective_hash_drift(self) -> None:
        configuration = self._write_configuration(
            profile=checker.STABLE_DEFAULT_PROFILE,
            candidate_id="riley-0.1.0-rc4",
        )
        request_path, _request = self._write_bind_request(configuration=configuration)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "candidate-drift.json")
        self.assert_reason(raised, "runtime-config-candidate-mismatch")

        configuration = self._write_configuration(
            profile=checker.STABLE_DEFAULT_PROFILE,
            startup_identity={
                "configuration_profile": checker.STABLE_DEFAULT_PROFILE,
                "configuration_sha256": "d" * 64,
            },
        )
        request_path, _request = self._write_bind_request(configuration=configuration)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "identity-drift.json")
        self.assert_reason(raised, "incomparable-binding")

        configuration = self._write_configuration(
            profile=checker.STABLE_DEFAULT_PROFILE,
            endpoint_payload_sha256="f" * 64,
        )
        request_path, _request = self._write_bind_request(configuration=configuration)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "digest-drift.json")
        self.assert_reason(raised, "startup-endpoint-hash-mismatch")

        configuration = self._write_configuration(
            profile=checker.STABLE_DEFAULT_PROFILE,
            effective_hash="f" * 64,
        )
        request_path, _request = self._write_bind_request(configuration=configuration)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "effective-drift.json")
        self.assert_reason(raised, "effective-config-hash-mismatch")

    def test_rejects_duplicate_paths_invalid_manifest_names_and_create_only_collisions(self) -> None:
        request_path, request = self._write_bind_request()
        request["scenario_contract_path"] = request["configuration_evidence"]["endpoint_path"]
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "duplicate.json")
        self.assert_reason(raised, "duplicate-evidence-path")

        long_path = "/".join(["a" * 100] * 5 + ["a" * 8])
        self.assertEqual(len(long_path), checker.MAX_RELATIVE_PATH_BYTES + 1)
        request_path, request = self._write_bind_request()
        request["configuration_evidence"]["endpoint_path"] = long_path
        self.tree.put(request_path, request)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "too-long-path.json")
        self.assert_reason(raised, "invalid-relative-path")
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker._descriptor(
                {"path": long_path, "sha256": "d" * 64, "byte_length": 1},
                "too-long descriptor",
            )
        self.assert_reason(raised, "invalid-relative-path")

        request_path, _request = self._write_bind_request()
        for name in ("nested/name.json", ".hidden.json", "x" * 247 + ".json"):
            with self.subTest(name=name):
                with self.assertRaises(binder.RawSoakBindError) as raised:
                    binder.bind_raw_soak_manifest(self.root, request_path, name)
                self.assert_reason(raised, "invalid-manifest-name")

        self.tree.put("collision.json", b"preexisting")
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "collision.json")
        self.assert_reason(raised, "create-only-collision")

        target = self.root / "link-target.json"
        target.write_bytes(b"untouched")
        link = self.root / "link.json"
        try:
            os.symlink(target.name, link)
        except OSError as error:  # pragma: no cover - platform capability
            self.skipTest(f"symlink unavailable: {error}")
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "link.json")
        self.assert_reason(raised, "create-only-collision")
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_bytes(), b"untouched")

        hard_target = self.root / "hard-target.json"
        hard_target.write_bytes(b"untouched")
        hard = self.root / "hard.json"
        try:
            os.link(hard_target, hard)
        except OSError as error:  # pragma: no cover - platform capability
            self.skipTest(f"hard link unavailable: {error}")
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "hard.json")
        self.assert_reason(raised, "create-only-collision")

    def test_marker_publication_failure_leaves_an_incomplete_nonreplaceable_manifest(self) -> None:
        request_path, _request = self._write_bind_request()
        original = common.write_create_only_json

        def fail_only_marker(
            root_fd: int, name: str, value: object, label: str
        ) -> common.CreatedEvidence:
            if name == "marker-failure.json.complete":
                error = common.ProvenanceV2Error("fixture marker fsync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            return original(root_fd, name, value, label)

        with mock.patch.object(common, "write_create_only_json", side_effect=fail_only_marker):
            with self.assertRaises(binder.RawSoakBindError) as raised:
                binder.bind_raw_soak_manifest(self.root, request_path, "marker-failure.json")
        self.assert_reason(raised, "durability-failure")
        self.assertTrue((self.root / "marker-failure.json").is_file())
        self.assertFalse((self.root / "marker-failure.json.complete").exists())
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance(self.root, "marker-failure.json")
        self.assert_reason(raised, "missing-soak-completion-marker")
        with self.assertRaises(binder.RawSoakBindError) as raised:
            binder.bind_raw_soak_manifest(self.root, request_path, "marker-failure.json")
        self.assert_reason(raised, "create-only-collision")

    def test_completed_verifier_rejects_a_manifest_replaced_after_marker_read(self) -> None:
        request_path, _request = self._write_bind_request()
        binder.bind_raw_soak_manifest(self.root, request_path, "replacement-race.json")
        original = checker._read_soak_completion_marker

        def read_marker_then_replace(
            root_fd: int, manifest: common.EvidenceDescriptor
        ) -> None:
            original(root_fd, manifest)
            replacement = json.loads((self.root / "replacement-race.json").read_bytes())
            replacement["candidate_id"] = "riley-0.1.0-rc4"
            self.tree.put("replacement-race.json", replacement)

        with mock.patch.object(
            checker, "_read_soak_completion_marker", side_effect=read_marker_then_replace
        ):
            with self.assertRaises(checker.C02ProvenanceError) as raised:
                checker.verify_completed_soak_provenance(self.root, "replacement-race.json")
        self.assert_reason(raised, "soak-manifest-changed-during-completion-verification")

    def test_binder_keeps_one_held_root_fd_when_the_path_is_swapped(self) -> None:
        request_path, _request = self._write_bind_request()
        root_fd = common.open_private_evidence_directory(self.root, "test evidence root")
        moved_root = self.base / "moved-evidence"
        os.rename(self.root, moved_root)
        self.root.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        try:
            report = binder.bind_raw_soak_manifest_fd(root_fd, request_path, "held-fd.json")
        finally:
            os.close(root_fd)
        self.assertEqual(report["status"], "bound")
        self.assertTrue((moved_root / "held-fd.json").is_file())
        self.assertTrue((moved_root / "held-fd.json.complete").is_file())
        self.assertFalse((self.root / "held-fd.json").exists())

    def test_path_wrapper_opens_the_private_root_once_and_source_uses_no_path_reader(self) -> None:
        request_path, _request = self._write_bind_request()
        original = common.open_private_evidence_directory
        with mock.patch.object(common, "open_private_evidence_directory", wraps=original) as opened:
            binder.bind_raw_soak_manifest(self.root, request_path, "one-open.json")
        self.assertEqual(opened.call_count, 1)
        source = Path(binder.__file__).read_text(encoding="utf-8")
        self.assertNotIn("read_regular_path", source)
        self.assertNotIn("verify_soak_provenance(", source)
        for forbidden in ("subprocess", "socket", "urllib", "requests", "nvidia-smi"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
