#!/usr/bin/env python3
"""Hostile-input tests for reconstructed-baseline rollback raw provenance v3."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_c02_provenance_v2 as c02
import check_rc3_rollback_provenance_v3 as checker
import provenance_v2_common as common
from test_check_reconstructed_prior_baseline import BaselineFixture


def metrics() -> dict[str, object]:
    return {
        "schema_version": c02.METRICS_VERSION,
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


class RollbackV3Fixture:
    candidate_id = "riley-0.1.0-rc3"
    gpu_uuid = "GPU-12345678-abcd-efab-cdef-1234567890ab"
    bindings = {
        "freeze_sha256": "1" * 64,
        "base_release_candidate_report_sha256": "2" * 64,
        "configuration_profile": checker.STABLE_DEFAULT_PROFILE,
        "configuration_sha256": "3" * 64,
    }

    def __init__(self, root: Path) -> None:
        self.root = root
        self.baseline = BaselineFixture(root)

    def put(self, relative: str, value: bytes | dict[str, object]) -> dict[str, object]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = common.canonical_json_bytes(value) if isinstance(value, dict) else value
        path.write_bytes(raw)
        return common.descriptor_for_bytes(relative, raw, relative).as_json()

    def descriptor(self, relative: str) -> dict[str, object]:
        raw = (self.root / relative).read_bytes()
        return common.descriptor_for_bytes(relative, raw, relative).as_json()

    def target(self, pid: int, ticks: int, *, port: int, inode: int) -> dict[str, object]:
        return {
            "server_pid": pid,
            "server_start_ticks": ticks,
            "listener_port": port,
            "listener_inode": inode,
            "gpu_index": 0,
            "gpu_uuid": self.gpu_uuid,
        }

    def exchange(self, phase: str, kind: str) -> dict[str, object]:
        prefix = f"capture/{phase}/{kind}"
        return {
            "request": self.put(
                f"{prefix}-request.http",
                f"GET /{kind} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("ascii"),
            ),
            "response_head": self.put(
                f"{prefix}-response-head.http",
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n",
            ),
            "response_body": self.put(
                f"{prefix}-response-body.json",
                b'{"status":"ok"}',
            ),
        }

    @staticmethod
    def proc_stat(pid: int, ticks: int) -> bytes:
        fields = ["S", *("0" for _ in range(18)), str(ticks), "0"]
        return f"{pid} (riley-server) {' '.join(fields)}\n".encode("ascii")

    @staticmethod
    def proc_tcp(port: int, inode: int) -> bytes:
        return (
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
            f"   0: 0100007F:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 {inode} 1\n"
        ).encode("ascii")

    def process_evidence(
        self,
        phase: str,
        *,
        pid: int,
        ticks: int,
        port: int,
        inode: int,
    ) -> dict[str, object]:
        raw = {
            "pre_stat": self.proc_stat(pid, ticks),
            "post_stat": self.proc_stat(pid, ticks),
            "pre_tcp": self.proc_tcp(port, inode),
            "post_tcp": self.proc_tcp(port, inode),
            "pre_fd_sockets": {
                "schema_version": c02.SOCKET_SNAPSHOT_VERSION,
                "server_pid": pid,
                "socket_inodes": [inode],
            },
            "post_fd_sockets": {
                "schema_version": c02.SOCKET_SNAPSHOT_VERSION,
                "server_pid": pid,
                "socket_inodes": [inode],
            },
            "status": f"Name:\triley-server\nPid:\t{pid}\n".encode("ascii"),
            "gpu_selection": f"0,{self.gpu_uuid}\n".encode("ascii"),
            "gpu_compute_apps": f"{pid},0\n".encode("ascii"),
        }
        return {
            name: self.put(f"capture/{phase}/process-{name}.raw", value)
            for name, value in raw.items()
        }

    def phase(
        self,
        phase: str,
        *,
        pid: int,
        ticks: int,
        port: int,
        inode: int,
        candidate: bool,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "target": self.target(pid, ticks, port=port, inode=inode),
            "process_evidence": self.process_evidence(
                phase,
                pid=pid,
                ticks=ticks,
                port=port,
                inode=inode,
            ),
            "health": self.exchange(phase, "health"),
            "generation": self.exchange(phase, "generation"),
        }
        if candidate:
            row["audit"] = {
                "availability": "source-owned",
                "generation_audit_index": self.put(
                    f"capture/{phase}/generation-audit-index.json",
                    {"schema_version": "fixture.audit-index.v1", "records": []},
                ),
            }
            shutdown = self.put(
                f"capture/{phase}/shutdown.json",
                {
                    "schema_version": c02.SHUTDOWN_VERSION,
                    "capture_status": "captured",
                    "qualification_status": "not-run",
                    "server_pid": pid,
                    "server_start_ticks": ticks,
                    "worker_ready": False,
                    "final_metrics": metrics(),
                },
            )
            row["shutdown_artifact"] = shutdown
            row["shutdown_marker"] = self.put(
                f"capture/{phase}/shutdown.json.complete",
                {
                    "schema_version": c02.SHUTDOWN_MARKER_VERSION,
                    "artifact_filename": "shutdown.json",
                    "artifact_sha256": shutdown["sha256"],
                },
            )
        else:
            row["audit"] = {"availability": "not-supported"}
        return row

    def artifacts(self, phase: str) -> dict[str, object]:
        result: dict[str, object] = {}
        for name in sorted(checker.ARTIFACT_FIELDS):
            if phase == "rollback" and name == "bundle":
                raw = (
                    self.root / self.baseline.a_artifacts["bundle"]["path"]
                ).read_bytes()
            elif phase == "rollback" and name == "image_inspect":
                raw = (
                    self.root
                    / self.baseline.a_runtime_image_inspect_raw["path"]
                ).read_bytes()
            else:
                raw = f"{phase}:{name}\n".encode("ascii")
            result[name] = self.put(
                f"capture/{phase}/artifact-{name}.raw",
                raw,
            )
        return result

    def document(self) -> dict[str, object]:
        candidate = self.phase(
            "candidate",
            pid=1111,
            ticks=2222,
            port=8080,
            inode=7001,
            candidate=True,
        )
        rollback = self.phase(
            "rollback",
            pid=3333,
            ticks=4444,
            port=8081,
            inode=7002,
            candidate=False,
        )
        atomic_switch = {
            name: self.put(
                f"capture/switch/{name}.raw",
                f"switch:{name}\n".encode("ascii"),
            )
            for name in sorted(checker.ATOMIC_SWITCH_FIELDS)
        }
        return {
            "schema_version": checker.ROLLBACK_V3_MANIFEST_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "candidate_id": self.candidate_id,
            "bindings": copy.deepcopy(self.bindings),
            "reconstructed_baseline": {"manifest": self.descriptor("baseline.json")},
            "candidate": candidate,
            "rollback": rollback,
            "candidate_artifacts": self.artifacts("candidate"),
            "rollback_artifacts": self.artifacts("rollback"),
            "atomic_switch": atomic_switch,
        }

    def write_manifest(self, document: dict[str, object]) -> str:
        self.put("rollback/manifest.json", document)
        return "rollback/manifest.json"


class RollbackV3ProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name) / "evidence"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        self.root = root.resolve(strict=True)
        self.fixture = RollbackV3Fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext[BaseException], code: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def pinned_baseline(self) -> mock._patch:
        return mock.patch.object(
            checker,
            "RECONSTRUCTED_ROLLBACK_TARGET",
            self.fixture.baseline.target_commit_sha1,
        )

    def test_binds_reconstructed_baseline_with_explicit_legacy_audit_boundary(self) -> None:
        manifest = self.fixture.write_manifest(self.fixture.document())
        with self.pinned_baseline():
            report = checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assertEqual(report["schema_version"], checker.ROLLBACK_V3_REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(
            report["reconstructed_baseline"]["baseline_kind"],
            "reconstructed-tag-baseline",
        )
        self.assertEqual(
            report["raw_evidence"]["candidate"]["audit"]["availability"],
            "source-owned",
        )
        self.assertEqual(
            report["raw_evidence"]["rollback"]["audit"]["availability"],
            "not-supported",
        )
        self.assertIn(
            "reconstructed-baseline-a-b-replay-binding",
            [row["name"] for row in report["checks"]],
        )
        self.assertIn(
            "active-baseline-bundle-and-image-binding",
            [row["name"] for row in report["checks"]],
        )

    def test_fd_api_replays_full_baseline_without_using_path_wrapper(self) -> None:
        manifest = self.fixture.write_manifest(self.fixture.document())
        with self.pinned_baseline():
            expected = checker.verify_rollback_provenance_v3(self.root, manifest)
        root_fd = common.open_private_evidence_directory(
            self.root,
            "rollback v3 held evidence root",
        )
        try:
            with self.pinned_baseline(), mock.patch.object(
                checker.baseline,
                "validate_file",
                side_effect=AssertionError("path wrapper must not be used"),
            ):
                actual = checker.verify_rollback_provenance_v3_fd(root_fd, manifest)
        finally:
            os.close(root_fd)
        self.assertEqual(actual, expected)

    def test_bytes_core_and_raw_target_deriver_match_the_file_replay(self) -> None:
        document = self.fixture.document()
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline():
            expected = checker.verify_rollback_provenance_v3(self.root, manifest)
        raw = (self.root / manifest).read_bytes()
        descriptor = common.descriptor_for_bytes(manifest, raw, "rollback manifest")
        process_evidence = {
            name: common.parse_descriptor(value, f"candidate process {name}")
            for name, value in document["candidate"]["process_evidence"].items()
        }
        root_fd = common.open_private_evidence_directory(
            self.root,
            "rollback v3 held evidence root",
        )
        try:
            with self.pinned_baseline():
                actual = checker.verify_rollback_provenance_v3_bytes_fd(
                    root_fd,
                    descriptor,
                    raw,
                )
                derived = checker.derive_phase_target_from_raw_evidence_fd(
                    root_fd,
                    process_evidence,
                    "candidate raw target",
                )
        finally:
            os.close(root_fd)
        self.assertEqual(actual, expected)
        self.assertEqual(derived.as_json(), document["candidate"]["target"])

    def test_bytes_core_rejects_a_laundered_manifest_descriptor(self) -> None:
        document = self.fixture.document()
        manifest = self.fixture.write_manifest(document)
        raw = (self.root / manifest).read_bytes()
        descriptor = common.descriptor_for_bytes(
            manifest,
            raw,
            "laundered manifest descriptor",
        )
        changed = copy.deepcopy(document)
        changed["candidate_id"] = "riley-9.9.9-rc9"
        root_fd = common.open_private_evidence_directory(
            self.root,
            "rollback v3 held evidence root",
        )
        try:
            with self.pinned_baseline(), self.assertRaises(
                checker.RollbackV3ProvenanceError
            ) as raised:
                checker.verify_rollback_provenance_v3_bytes_fd(
                    root_fd,
                    descriptor,
                    common.canonical_json_bytes(changed),
                )
        finally:
            os.close(root_fd)
        self.assert_reason(raised, "manifest-document-descriptor-mismatch")

    def test_rejects_nonprivate_held_root_before_manifest_read(self) -> None:
        self.root.chmod(0o755)
        try:
            root_fd = common.open_absolute_directory(
                self.root,
                "nonprivate rollback v3 evidence root",
            )
            try:
                with self.assertRaises(checker.RollbackV3ProvenanceError) as raised:
                    checker.verify_rollback_provenance_v3_fd(root_fd, "missing.json")
                self.assert_reason(raised, "unsafe-evidence-root-mode")
            finally:
                os.close(root_fd)
        finally:
            self.root.chmod(0o700)

    def test_rejects_source_checkout_as_an_evidence_root_before_opening_it(self) -> None:
        source_root = Path(checker.__file__).resolve().parents[2]
        with self.assertRaises(checker.RollbackV3ProvenanceError) as raised:
            checker.verify_rollback_provenance_v3(source_root, "ignored.json")
        self.assert_reason(raised, "evidence-root-inside-source-checkout")

    def test_rejects_target_drift_from_raw_process_socket_and_gpu_leaves(self) -> None:
        document = self.fixture.document()
        document["candidate"]["target"]["server_pid"] = 9999
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "phase-target-raw-mismatch")

        document = self.fixture.document()
        document["rollback"]["process_evidence"]["pre_tcp"] = self.fixture.put(
            "capture/rollback/process-pre_tcp.raw",
            self.fixture.proc_tcp(8081, 7999),
        )
        document["rollback"]["process_evidence"]["pre_fd_sockets"] = self.fixture.put(
            "capture/rollback/process-pre_fd_sockets.raw",
            {
                "schema_version": c02.SOCKET_SNAPSHOT_VERSION,
                "server_pid": 3333,
                "socket_inodes": [7999],
            },
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "listener-proof-mismatch")

        document = self.fixture.document()
        document["rollback"]["process_evidence"]["post_fd_sockets"] = self.fixture.put(
            "capture/rollback/process-post_fd_sockets.raw",
            {
                "schema_version": c02.SOCKET_SNAPSHOT_VERSION,
                "server_pid": 3333,
                "socket_inodes": [7999],
            },
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "listener-proof-missing")

        document = self.fixture.document()
        document["candidate"]["process_evidence"]["gpu_selection"] = self.fixture.put(
            "capture/candidate/process-gpu_selection.raw",
            b"0,GPU-ffffffff-ffff-ffff-ffff-ffffffffffff\n",
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "phase-target-raw-mismatch")

        document = self.fixture.document()
        document["candidate"]["process_evidence"]["status"] = self.fixture.put(
            "capture/candidate/process-status.raw",
            b"Name:\triley-server\nPid:\t9999\n",
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "pid-start-tick-mismatch")

        document = self.fixture.document()
        document["rollback"]["process_evidence"]["gpu_compute_apps"] = self.fixture.put(
            "capture/rollback/process-gpu_compute_apps.raw",
            b"9999,0\n",
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "gpu-process-binding-mismatch")

    def test_rejects_candidate_or_baseline_audit_availability_drift(self) -> None:
        document = self.fixture.document()
        document["candidate"]["audit"] = {"availability": "not-supported"}
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "invalid-audit-availability")

        document = self.fixture.document()
        document["rollback"]["audit"] = {
            "availability": "source-owned",
            "generation_audit_index": self.fixture.put(
                "capture/rollback/forged-audit-index.json",
                {"schema_version": "fixture.audit-index.v1", "records": []},
            ),
        }
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "invalid-audit-availability")

    def test_rejects_reused_process_identity_and_cross_role_leaf_alias(self) -> None:
        document = self.fixture.document()
        document["rollback"] = self.fixture.phase(
            "rollback-reused",
            pid=1111,
            ticks=2222,
            port=8080,
            inode=7001,
            candidate=False,
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "reused-candidate-process")

        document = self.fixture.document()
        document["atomic_switch"]["post_active_stat"] = copy.deepcopy(
            document["candidate"]["health"]["request"]
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "duplicate-evidence-path")

    def test_rejects_nonstable_profile_and_candidate_shutdown_marker_drift(self) -> None:
        document = self.fixture.document()
        document["bindings"]["configuration_profile"] = "max-performance-exact"
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "invalid-configuration-profile")

        document = self.fixture.document()
        document["candidate"]["shutdown_marker"] = self.fixture.put(
            "capture/candidate/shutdown.json.complete",
            {
                "schema_version": c02.SHUTDOWN_MARKER_VERSION,
                "artifact_filename": "shutdown.json",
                "artifact_sha256": "f" * 64,
            },
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "shutdown-marker-mismatch")

    def test_rejects_historical_baseline_claim_before_phase_authority(self) -> None:
        changed = copy.deepcopy(self.fixture.baseline.manifest)
        changed["historical_distribution"] = "attested"
        self.fixture.baseline.rewrite_manifest(changed)
        manifest = self.fixture.write_manifest(self.fixture.document())
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "historical-distribution-claim")

    def test_rejects_baseline_bundle_or_image_identity_drift(self) -> None:
        document = self.fixture.document()
        document["rollback_artifacts"]["bundle"] = self.fixture.put(
            "capture/rollback/forged-bundle.raw",
            b"not the reconstructed bundle",
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "baseline-bundle-binding-mismatch")

        document = self.fixture.document()
        document["rollback_artifacts"]["image_inspect"] = self.fixture.put(
            "capture/rollback/forged-image-inspect.json",
            b'[{"Id":"sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}]',
        )
        manifest = self.fixture.write_manifest(document)
        with self.pinned_baseline(), self.assertRaises(
            checker.RollbackV3ProvenanceError
        ) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "baseline-image-binding-mismatch")

    def test_rejects_another_reconstructed_tag_target(self) -> None:
        manifest = self.fixture.write_manifest(self.fixture.document())
        with self.assertRaises(checker.RollbackV3ProvenanceError) as raised:
            checker.verify_rollback_provenance_v3(self.root, manifest)
        self.assert_reason(raised, "unsupported-reconstructed-baseline")

    def test_published_schema_preserves_the_v3_legacy_boundary(self) -> None:
        repository_schema = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "rollback-receipt-v3.schema.json"
        )
        schema_path = (
            repository_schema
            if repository_schema.is_file()
            else Path(__file__).with_name("rollback-receipt-v3.schema.json")
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema",
        )
        self.assertEqual(
            schema["$defs"]["rawManifest"]["properties"]["schema_version"],
            {"const": checker.ROLLBACK_V3_MANIFEST_VERSION},
        )
        self.assertEqual(
            schema["$defs"]["reconstructedBaselinePhase"]["properties"]["audit"],
            {"$ref": "#/$defs/unsupportedAudit"},
        )
        self.assertNotIn(
            "shutdown_artifact",
            schema["$defs"]["reconstructedBaselinePhase"]["properties"],
        )
        self.assertEqual(
            schema["$defs"]["provenanceReport"]["properties"][
                "qualification_status"
            ],
            {"const": "not-run"},
        )
        self.assertEqual(
            schema["$defs"]["provenanceReport"]["properties"][
                "reconstructed_baseline"
            ]["properties"]["baseline_id"],
            {"const": checker.RECONSTRUCTED_ROLLBACK_BASELINE_ID},
        )


if __name__ == "__main__":
    unittest.main()
