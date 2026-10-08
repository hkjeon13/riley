#!/usr/bin/env python3
"""Hostile-input tests for the local-only RC3 rollback raw v3 binder."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import bind_raw_rc3_rollback_capture as binder
import check_rc3_rollback_provenance_v3 as checker
import provenance_v2_common as common
from test_check_rc3_rollback_provenance_v3 import RollbackV3Fixture


class BindRawRc3RollbackCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name) / "evidence"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        self.root = root.resolve(strict=True)
        self.fixture = RollbackV3Fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext[BaseException],
        code: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def pinned_baseline(self) -> mock._patch:
        return mock.patch.multiple(
            checker,
            RECONSTRUCTED_ROLLBACK_TARGET=self.fixture.baseline.target_commit_sha1,
            RECONSTRUCTED_ROLLBACK_TAG_OBJECT=self.fixture.baseline.tag_object_sha1,
        )

    @staticmethod
    def _path(descriptor: MappingLike) -> str:
        return str(descriptor["path"])

    def _path_map(self, descriptors: MappingLike, fields: frozenset[str]) -> dict[str, str]:
        return {
            f"{name}_path": self._path(descriptors[name])
            for name in sorted(fields)
        }

    def _phase_request(
        self,
        phase: MappingLike,
        *,
        candidate_phase: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "process_evidence": self._path_map(
                phase["process_evidence"],
                checker.RAW_PROCESS_FIELDS,
            ),
            "health": self._path_map(phase["health"], checker.HTTP_EXCHANGE_FIELDS),
            "generation": self._path_map(
                phase["generation"],
                checker.HTTP_EXCHANGE_FIELDS,
            ),
        }
        if candidate_phase:
            result.update(
                {
                    "generation_audit_index_path": self._path(
                        phase["audit"]["generation_audit_index"]
                    ),
                    "shutdown_artifact_path": self._path(phase["shutdown_artifact"]),
                    "shutdown_marker_path": self._path(phase["shutdown_marker"]),
                }
            )
        return result

    def _request(self, *, request_path: str = "requests/rollback-bind.json") -> tuple[str, dict[str, Any]]:
        document: MappingLike = self.fixture.document()
        binding_leaves = {
            "freeze_path": ("bindings/freeze.raw", b"freeze-v3\\n"),
            "base_release_candidate_report_path": (
                "bindings/base-release-report.raw",
                b"base-release-report-v3\\n",
            ),
            "configuration_path": (
                "bindings/stable-default-config.raw",
                b"stable-default-config-v3\\n",
            ),
        }
        for path, raw in binding_leaves.values():
            self.fixture.put(path, raw)
        request: dict[str, Any] = {
            "schema_version": binder.BIND_REQUEST_VERSION,
            "candidate_id": document["candidate_id"],
            "binding_evidence": {
                name: path for name, (path, _raw) in binding_leaves.items()
            },
            "reconstructed_baseline": {
                "manifest_path": self._path(document["reconstructed_baseline"]["manifest"]),
            },
            "candidate": self._phase_request(document["candidate"], candidate_phase=True),
            "rollback": self._phase_request(document["rollback"], candidate_phase=False),
            "candidate_artifacts": self._path_map(
                document["candidate_artifacts"],
                checker.ARTIFACT_FIELDS,
            ),
            "rollback_artifacts": self._path_map(
                document["rollback_artifacts"],
                checker.ARTIFACT_FIELDS,
            ),
            "atomic_switch": self._path_map(
                document["atomic_switch"],
                checker.ATOMIC_SWITCH_FIELDS,
            ),
        }
        self.fixture.put(request_path, request)
        return request_path, request

    def _rewrite_request(self, request_path: str, request: dict[str, Any]) -> None:
        self.fixture.put(request_path, request)

    def _bind(self, request_path: str, manifest_name: str) -> dict[str, Any]:
        with self.pinned_baseline():
            return binder.bind_raw_rollback_manifest(self.root, request_path, manifest_name)

    def test_binds_closed_path_only_request_and_self_verifies_nonterminal_output(self) -> None:
        request_path, request = self._request()
        name = "rollback-bound.json"
        report = self._bind(request_path, name)

        output = self.root / name
        raw = output.read_bytes()
        manifest = json.loads(raw)
        self.assertEqual(raw, common.canonical_json_bytes(manifest))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assertEqual(output.stat().st_nlink, 1)
        self.assertFalse((self.root / f"{name}.complete").exists())
        self.assertFalse((self.root / f"{name}.intent").exists())
        self.assertEqual(report["schema_version"], checker.ROLLBACK_V3_REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(manifest["bindings"]["configuration_profile"], checker.STABLE_DEFAULT_PROFILE)
        self.assertEqual(
            manifest["bindings"]["freeze_sha256"],
            hashlib.sha256(b"freeze-v3\\n").hexdigest(),
        )
        self.assertNotIn("target", request["candidate"])
        self.assertNotIn("target", request["rollback"])
        self.assertNotIn("audit", request["candidate"])
        target_by_phase = {row["phase"]: row["target"] for row in report["targets"]}
        self.assertEqual(target_by_phase["candidate"]["server_pid"], 1111)
        self.assertEqual(target_by_phase["rollback"]["server_pid"], 3333)
        with self.pinned_baseline():
            standalone = checker.verify_rollback_provenance_v3(self.root, name)
        self.assertEqual(report, standalone)

    def test_fd_entry_uses_held_baseline_replay_not_path_wrapper(self) -> None:
        request_path, _request = self._request()
        root_fd = common.open_private_evidence_directory(self.root, "rollback bind test root")
        try:
            with self.pinned_baseline(), mock.patch.object(
                checker.baseline,
                "validate_file",
                side_effect=AssertionError("path wrapper must not be used"),
            ):
                report = binder.bind_raw_rollback_manifest_fd(
                    root_fd,
                    request_path,
                    "held-fd-bound.json",
                )
        finally:
            os.close(root_fd)
        self.assertEqual(report["status"], "bound")

    def test_held_locked_entry_neither_reopens_nor_relocks_the_root(self) -> None:
        request_path, _request = self._request()
        root_fd = common.open_private_evidence_directory(self.root, "held rollback v3 bind root")
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        real_flock = fcntl.flock

        def reject_root_relock(descriptor: int, operation: int) -> None:
            if descriptor == root_fd:
                raise AssertionError("held v3 root must not be relocked")
            real_flock(descriptor, operation)

        try:
            with self.pinned_baseline(), mock.patch.object(
                binder.fcntl,
                "flock",
                side_effect=reject_root_relock,
            ):
                report = binder._bind_raw_rollback_manifest_held_locked_fd(  # noqa: SLF001
                    root_fd,
                    request_path,
                    "held-locked-bound.json",
                )
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)
        self.assertEqual(report["status"], "bound")

    def test_rejects_caller_declared_target_descriptors_and_audit_before_output(self) -> None:
        mutations = (
            (
                "target",
                lambda request: request["candidate"].update(
                    {"target": {"server_pid": 1}}
                ),
            ),
            (
                "binding-sha",
                lambda request: request["binding_evidence"].update(
                    {"freeze_sha256": "f" * 64}
                ),
            ),
            (
                "audit-availability",
                lambda request: request["candidate"].update(
                    {"audit": {"availability": "not-supported"}}
                ),
            ),
        )
        for suffix, mutate in mutations:
            with self.subTest(suffix=suffix):
                request_path, request = self._request(
                    request_path=f"requests/forged-{suffix}.json"
                )
                mutate(request)
                self._rewrite_request(request_path, request)
                name = f"forged-{suffix}.json"
                with self.assertRaises(binder.RollbackBindError) as raised:
                    self._bind(request_path, name)
                self.assert_reason(raised, "unexpected-field-set")
                self.assertFalse((self.root / name).exists())

    def test_derives_process_identities_and_rejects_drift_or_reuse_prepublication(self) -> None:
        request_path, request = self._request()
        status_path = request["candidate"]["process_evidence"]["status_path"]
        self.fixture.put(status_path, b"Name:\triley-server\\nPid:\t9999\\n")
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "status-drift.json")
        self.assert_reason(raised, "pid-start-tick-mismatch")
        self.assertFalse((self.root / "status-drift.json").exists())

        request_path, request = self._request(request_path="requests/reused-process.json")
        reused = self.fixture.process_evidence(
            "rollback-reused",
            pid=1111,
            ticks=2222,
            port=8081,
            inode=7011,
        )
        request["rollback"]["process_evidence"] = self._path_map(
            reused,
            checker.RAW_PROCESS_FIELDS,
        )
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "reused-process.json")
        self.assert_reason(raised, "reused-candidate-process")
        self.assertFalse((self.root / "reused-process.json").exists())

    def test_rejects_raw_listener_and_gpu_identity_inconsistencies_prepublication(self) -> None:
        request_path, request = self._request(request_path="requests/post-tcp-drift.json")
        post_tcp_path = request["candidate"]["process_evidence"]["post_tcp_path"]
        self.fixture.put(post_tcp_path, self.fixture.proc_tcp(8082, 7001))
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "post-tcp-drift.json")
        self.assert_reason(raised, "listener-proof-mismatch")
        self.assertFalse((self.root / "post-tcp-drift.json").exists())

        request_path, request = self._request(request_path="requests/gpu-pid-drift.json")
        compute_apps_path = request["candidate"]["process_evidence"][
            "gpu_compute_apps_path"
        ]
        self.fixture.put(compute_apps_path, b"9999,0\n")
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "gpu-pid-drift.json")
        self.assert_reason(raised, "gpu-process-binding-mismatch")
        self.assertFalse((self.root / "gpu-pid-drift.json").exists())

        request_path, request = self._request(request_path="requests/multiple-listeners.json")
        header, first_row = self.fixture.proc_tcp(8080, 7001).split(b"\n", 1)
        second_row = self.fixture.proc_tcp(8082, 7003).split(b"\n", 1)[1]
        two_listeners = header + b"\n" + first_row + second_row
        for name in ("pre_tcp_path", "post_tcp_path"):
            self.fixture.put(
                request["candidate"]["process_evidence"][name],
                two_listeners,
            )
        for name in ("pre_fd_sockets_path", "post_fd_sockets_path"):
            self.fixture.put(
                request["candidate"]["process_evidence"][name],
                {
                    "schema_version": checker.c02.SOCKET_SNAPSHOT_VERSION,
                    "server_pid": 1111,
                    "socket_inodes": [7001, 7003],
                },
            )
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "multiple-listeners.json")
        self.assert_reason(raised, "listener-proof-missing")
        self.assertFalse((self.root / "multiple-listeners.json").exists())

    def test_replays_candidate_shutdown_pair_before_output(self) -> None:
        request_path, request = self._request()
        marker_path = request["candidate"]["shutdown_marker_path"]
        self.fixture.put(
            marker_path,
            {
                "schema_version": checker.c02.SHUTDOWN_MARKER_VERSION,
                "artifact_filename": "shutdown.json",
                "artifact_sha256": "f" * 64,
            },
        )
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "shutdown-marker-drift.json")
        self.assert_reason(raised, "shutdown-marker-mismatch")
        self.assertFalse((self.root / "shutdown-marker-drift.json").exists())

    def test_rejects_duplicate_direct_and_transitive_baseline_paths_before_output(self) -> None:
        request_path, request = self._request()
        request["atomic_switch"]["post_active_stat_path"] = request["candidate"]["health"][
            "request_path"
        ]
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "duplicate-direct.json")
        self.assert_reason(raised, "duplicate-evidence-path")
        self.assertFalse((self.root / "duplicate-direct.json").exists())

        request_path, request = self._request(request_path="requests/baseline-alias.json")
        request["binding_evidence"]["freeze_path"] = self.fixture.baseline.a_artifacts[
            "bundle"
        ]["path"]
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "baseline-alias.json")
        self.assert_reason(raised, "duplicate-evidence-path")
        self.assertFalse((self.root / "baseline-alias.json").exists())

    def test_replays_binary_and_bundle_baseline_bindings_before_publication(self) -> None:
        request_path, request = self._request(request_path="requests/baseline-binary-drift.json")
        self.fixture.put("capture/rollback/forged-binary.raw", b"forged rollback binary")
        request["rollback_artifacts"]["binary_path"] = "capture/rollback/forged-binary.raw"
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "baseline-binary-drift.json")
        self.assert_reason(raised, "baseline-binary-binding-mismatch")
        self.assertFalse((self.root / "baseline-binary-drift.json").exists())

        request_path, request = self._request()
        self.fixture.put("capture/rollback/forged-bundle.raw", b"forged rollback bundle")
        request["rollback_artifacts"]["bundle_path"] = "capture/rollback/forged-bundle.raw"
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "baseline-bundle-drift.json")
        self.assert_reason(raised, "baseline-bundle-binding-mismatch")
        self.assertFalse((self.root / "baseline-bundle-drift.json").exists())

    def test_rejects_legacy_or_malformed_bind_requests_before_output(self) -> None:
        for version in (
            "riley.rc3-rollback-bind-request.v1",
            "riley.rc3-rollback-bind-request.v2",
        ):
            with self.subTest(version=version):
                request_path, request = self._request(
                    request_path=f"requests/{version.rsplit('.', 1)[-1]}.json"
                )
                request["schema_version"] = version
                self._rewrite_request(request_path, request)
                name = f"legacy-{version.rsplit('.', 1)[-1]}.json"
                with self.assertRaises(binder.RollbackBindError) as raised:
                    self._bind(request_path, name)
                self.assert_reason(raised, "unsupported-bind-request-version")
                self.assertFalse((self.root / name).exists())

        duplicate_request = self.root / "requests/duplicate-key.json"
        duplicate_request.write_bytes(
            b'{"schema_version":"riley.rc3-rollback-bind-request.v3",'
            b'"schema_version":"riley.rc3-rollback-bind-request.v3"}'
        )
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind("requests/duplicate-key.json", "duplicate-key.json")
        self.assert_reason(raised, "duplicate-json-key")
        self.assertFalse((self.root / "duplicate-key.json").exists())

        oversized_request = self.root / "requests/oversized.json"
        oversized_request.write_bytes(b"x" * (binder.MAX_BIND_REQUEST_BYTES + 1))
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind("requests/oversized.json", "oversized.json")
        self.assert_reason(raised, "input-too-large")
        self.assertFalse((self.root / "oversized.json").exists())

    def test_rejects_unsafe_raw_paths_and_private_root_failures(self) -> None:
        request_path, request = self._request()
        request["candidate_artifacts"]["binary_path"] = "../outside"
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "traversal.json")
        self.assert_reason(raised, "invalid-relative-path")
        self.assertFalse((self.root / "traversal.json").exists())

        request_path, request = self._request(request_path="requests/hard-link.json")
        original = self.root / request["candidate_artifacts"]["binary_path"]
        alias = self.root / "aliases/candidate-binary.raw"
        alias.parent.mkdir(parents=True, exist_ok=True)
        os.link(original, alias)
        request["candidate_artifacts"]["binary_path"] = "aliases/candidate-binary.raw"
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "hard-link.json")
        self.assert_reason(raised, "nonunique-evidence-inode")
        self.assertFalse((self.root / "hard-link.json").exists())

        request_path, _request = self._request(request_path="requests/nonprivate.json")
        self.root.chmod(0o755)
        try:
            root_fd = common.open_absolute_directory(self.root, "nonprivate rollback bind root")
            try:
                with self.assertRaises(binder.RollbackBindError) as raised:
                    binder.bind_raw_rollback_manifest_fd(
                        root_fd,
                        request_path,
                        "nonprivate.json",
                    )
                self.assert_reason(raised, "unsafe-evidence-root-mode")
            finally:
                os.close(root_fd)
        finally:
            self.root.chmod(0o700)

    def test_rejects_final_and_intermediate_symlink_inputs(self) -> None:
        request_path, request = self._request(request_path="requests/final-symlink.json")
        binary_path = self.root / request["candidate_artifacts"]["binary_path"]
        binary_path.unlink()
        binary_path.symlink_to(self.root / "outside-binary.raw")
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "final-symlink.json")
        self.assert_reason(raised, "unsafe-evidence-path")
        self.assertFalse((self.root / "final-symlink.json").exists())

        request_path, request = self._request(
            request_path="requests/intermediate-symlink.json"
        )
        real_directory = self.root / "real-capture"
        real_directory.mkdir()
        (real_directory / "binary.raw").write_bytes(b"captured binary")
        (self.root / "linked-capture").symlink_to("real-capture", target_is_directory=True)
        request["candidate_artifacts"]["binary_path"] = "linked-capture/binary.raw"
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "intermediate-symlink.json")
        self.assert_reason(raised, "unsafe-evidence-directory")
        self.assertFalse((self.root / "intermediate-symlink.json").exists())

    def test_rejects_source_root_and_reserved_output_siblings(self) -> None:
        source_root = Path(binder.__file__).resolve().parents[2]
        with self.assertRaises(binder.RollbackBindError) as raised:
            binder.bind_raw_rollback_manifest(source_root, "ignored.json", "ignored.json")
        self.assert_reason(raised, "evidence-root-inside-source-checkout")

        request_path, _request = self._request()
        sidecar = self.root / "reserved.json.complete"
        sidecar.write_bytes(b"unrelated terminal-looking sidecar")
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "reserved.json")
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / "reserved.json").exists())

        self.fixture.put("existing.json", b"already published")
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "existing.json")
        self.assert_reason(raised, "output-name-collision")

    def test_fd_entry_remains_pinned_after_the_root_path_is_replaced(self) -> None:
        request_path, _request = self._request()
        root_fd = common.open_private_evidence_directory(self.root, "pinned rollback bind root")
        relocated = self.root.parent / "relocated-evidence"
        try:
            self.root.rename(relocated)
            self.root.mkdir(mode=0o700)
            self.root.chmod(0o700)
            with self.pinned_baseline():
                report = binder.bind_raw_rollback_manifest_fd(
                    root_fd,
                    request_path,
                    "pinned-root.json",
                )
        finally:
            os.close(root_fd)
        self.assertEqual(report["status"], "bound")
        self.assertTrue((relocated / "pinned-root.json").is_file())
        self.assertFalse((self.root / "pinned-root.json").exists())

    def test_lock_and_postpublication_replay_failure_never_signal_completion(self) -> None:
        request_path, _request = self._request()
        root_fd = common.open_private_evidence_directory(self.root, "rollback bind test root")
        try:
            with mock.patch.object(binder.fcntl, "flock", side_effect=OSError("lock unavailable")):
                with self.assertRaises(binder.RollbackBindError) as raised:
                    binder.bind_raw_rollback_manifest_fd(root_fd, request_path, "locked.json")
                self.assert_reason(raised, "output-lock-unavailable")
        finally:
            os.close(root_fd)
        self.assertFalse((self.root / "locked.json").exists())

        error = checker.RollbackV3ProvenanceError("simulated post-publication replay failure")
        error.reason_code = "simulated-post-publication-replay"  # type: ignore[attr-defined]
        with self.pinned_baseline(), mock.patch.object(
            checker,
            "verify_rollback_provenance_v3_fd",
            side_effect=error,
        ):
            with self.assertRaises(binder.RollbackBindError) as raised:
                binder.bind_raw_rollback_manifest(
                    self.root,
                    request_path,
                    "postwrite-failure.json",
                )
        self.assert_reason(raised, "simulated-post-publication-replay")
        self.assertTrue((self.root / "postwrite-failure.json").is_file())
        self.assertFalse((self.root / "postwrite-failure.json.complete").exists())
        self.assertFalse((self.root / "postwrite-failure.json.intent").exists())

    def test_streams_large_artifact_without_bounded_raw_reader(self) -> None:
        request_path, request = self._request()
        binary_path = request["candidate_artifacts"]["binary_path"]
        large = b"x" * (checker.MAX_RAW_LEAF_BYTES + 1)
        (self.root / binary_path).write_bytes(large)
        original = common.read_bounded_regular_relative

        def forbid_large_raw_read(
            root_fd: int,
            relative_path: str,
            label: str,
            *,
            maximum_bytes: int = common.DEFAULT_MAX_ARTIFACT_BYTES,
        ) -> bytes:
            if relative_path == binary_path:
                raise AssertionError("large binary must be stream-described, not materialized")
            return original(
                root_fd,
                relative_path,
                label,
                maximum_bytes=maximum_bytes,
            )

        with self.pinned_baseline(), mock.patch.object(
            common,
            "read_bounded_regular_relative",
            side_effect=forbid_large_raw_read,
        ):
            report = binder.bind_raw_rollback_manifest(
                self.root,
                request_path,
                "streamed-artifact.json",
            )
        descriptor = report["raw_evidence"]["candidate_artifacts"]["binary"]
        self.assertEqual(descriptor["byte_length"], len(large))
        self.assertEqual(descriptor["sha256"], hashlib.sha256(large).hexdigest())

    def test_request_is_canonical_and_binder_has_no_operational_import_or_terminal_publish(self) -> None:
        request_path, request = self._request()
        (self.root / request_path).write_bytes(common.canonical_json_bytes(request) + b"\n")
        with self.assertRaises(binder.RollbackBindError) as raised:
            self._bind(request_path, "noncanonical-request.json")
        self.assert_reason(raised, "noncanonical-json")
        self.assertFalse((self.root / "noncanonical-request.json").exists())

        source = Path(binder.__file__).read_text(encoding="utf-8")
        imports: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.add(node.module.split(".", 1)[0])
        self.assertFalse(imports.intersection({"socket", "subprocess", "asyncio"}))
        self.assertNotIn("publish_create_only_hardlink", source)

    def test_published_bind_request_schema_is_path_only_and_closed(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "rollback-bind-request-v3.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema_version"], {"const": binder.BIND_REQUEST_VERSION})
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("sha256", schema["$defs"])
        self.assertEqual(schema["$defs"]["candidateId"]["maxLength"], 128)
        self.assertNotIn("target", schema["$defs"]["candidatePhase"]["properties"])
        self.assertNotIn("audit", schema["$defs"]["candidatePhase"]["properties"])
        self.assertEqual(
            set(schema["$defs"]["candidatePhase"]["required"]),
            {
                "process_evidence",
                "health",
                "generation",
                "generation_audit_index_path",
                "shutdown_artifact_path",
                "shutdown_marker_path",
            },
        )
        self.assertEqual(
            set(schema["$defs"]["rollbackPhase"]["required"]),
            {"process_evidence", "health", "generation"},
        )
        pattern = schema["$defs"]["relativePath"]["pattern"]
        self.assertIsNotNone(re.fullmatch(pattern, "capture/a.raw"))
        self.assertIsNone(re.fullmatch(pattern, "capture/../a.raw"))


MappingLike = dict[str, Any]


if __name__ == "__main__":
    unittest.main()
