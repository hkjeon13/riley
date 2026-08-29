#!/usr/bin/env python3
"""CPU-only hostile tests for the single-chain rollback terminal v4 producer."""

from __future__ import annotations

import ast
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest import mock

import bind_raw_rc3_rollback_capture as v3_binder
import bind_raw_rc3_rollback_terminal_v4 as binder
import capture_rc3_rollback_atomic_switch_v1 as atomic
import capture_rc3_rollback_atomic_transaction_v1 as transaction
import check_rc3_rollback_provenance_v3 as v3
import check_rc3_rollback_provenance_v4 as checker
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common
from test_check_rc3_rollback_provenance_v3 import RollbackV3Fixture


def fake_exchange(directory_fd: int) -> None:
    """Portable test double; production capture always uses renameat2."""

    temporary = "v4-test-exchange-temporary"
    os.rename(atomic.ACTIVE_NAME, temporary, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.rename(
        atomic.ROLLBACK_STAGED_NAME,
        atomic.ACTIVE_NAME,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    os.rename(temporary, atomic.ROLLBACK_STAGED_NAME, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)


class RollbackV4TerminalTests(unittest.TestCase):
    v3_name = "rollback-v3.json"
    v4_name = "rollback-v4.json"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "evidence"
        self.root.mkdir(mode=0o700)
        self.root.chmod(0o700)
        self.inputs = self.base / "inputs"
        self.inputs.mkdir(mode=0o700)
        self.inputs.chmod(0o700)
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.source_patches = [
            mock.patch.object(prepare, "_source_root", return_value=self.repository),
            mock.patch.object(atomic, "_source_root", return_value=self.repository),
            mock.patch.object(transaction, "_source_root", return_value=self.repository),
            mock.patch.object(checker, "_source_root", return_value=self.repository),
            mock.patch.object(binder, "_source_root", return_value=self.repository),
        ]
        for patch in self.source_patches:
            patch.start()
        self.fixture = RollbackV3Fixture(self.root)
        self.document = self.fixture.document()
        self.preparation_request = self._preparation_request()
        self.request_path, self.bind_request = self._bind_request()
        self.pinned_target = mock.patch.object(
            v3,
            "RECONSTRUCTED_ROLLBACK_TARGET",
            self.fixture.baseline.target_commit_sha1,
        )
        self.pinned_target.start()

    def tearDown(self) -> None:
        self.pinned_target.stop()
        for patch in reversed(self.source_patches):
            patch.stop()
        self.temporary.cleanup()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext[BaseException],
        code: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    @staticmethod
    def _path(descriptor: Mapping[str, Any]) -> str:
        value = descriptor["path"]
        assert isinstance(value, str)
        return value

    def _path_map(
        self,
        descriptors: Mapping[str, Mapping[str, Any]],
        fields: frozenset[str],
    ) -> dict[str, str]:
        return {f"{name}_path": self._path(descriptors[name]) for name in sorted(fields)}

    def _phase_request(
        self,
        phase: Mapping[str, Any],
        *,
        candidate_phase: bool,
    ) -> dict[str, Any]:
        process_evidence = phase["process_evidence"]
        health = phase["health"]
        generation = phase["generation"]
        assert isinstance(process_evidence, Mapping)
        assert isinstance(health, Mapping)
        assert isinstance(generation, Mapping)
        result: dict[str, Any] = {
            "process_evidence": self._path_map(process_evidence, v3.RAW_PROCESS_FIELDS),
            "health": self._path_map(health, v3.HTTP_EXCHANGE_FIELDS),
            "generation": self._path_map(generation, v3.HTTP_EXCHANGE_FIELDS),
        }
        if candidate_phase:
            audit = phase["audit"]
            assert isinstance(audit, Mapping)
            result.update(
                {
                    "generation_audit_index_path": self._path(audit["generation_audit_index"]),
                    "shutdown_artifact_path": self._path(phase["shutdown_artifact"]),
                    "shutdown_marker_path": self._path(phase["shutdown_marker"]),
                }
            )
        return result

    def _prepared_artifact_paths(self, arm: str) -> dict[str, str]:
        filenames = {
            "binary": f"{arm}-binary",
            "bundle": f"{arm}-bundle",
            "image_inspect": f"{arm}-image-inspect.json",
        }
        return {
            f"{name}_path": f"{prepare.ARTIFACT_DIRECTORY_NAME}/{filenames[name]}"
            for name in sorted(v3.ARTIFACT_FIELDS)
        }

    def _prepared_atomic_paths(self) -> dict[str, str]:
        filenames = {
            "pre_active_stat": "pre-active-stat.json",
            "post_active_stat": "post-active-stat.json",
            "candidate_staged_stat": "candidate-staged-stat.json",
            "rollback_staged_stat": "rollback-staged-stat.json",
            "rename_transcript": "rename-transcript.json",
        }
        return {
            f"{name}_path": f"{transaction.ATOMIC_CAPTURE_DIRECTORY_NAME}/{filenames[name]}"
            for name in sorted(v3.ATOMIC_SWITCH_FIELDS)
        }

    def _bind_request(
        self,
        *,
        request_path: str = "requests/rollback-v3-bind.json",
    ) -> tuple[str, dict[str, Any]]:
        candidate = self.document["candidate"]
        rollback = self.document["rollback"]
        baseline = self.document["reconstructed_baseline"]
        assert isinstance(candidate, Mapping)
        assert isinstance(rollback, Mapping)
        assert isinstance(baseline, Mapping)
        binding_leaves = {
            "freeze_path": ("bindings/freeze.raw", b"freeze-v4\n"),
            "base_release_candidate_report_path": (
                "bindings/base-release-report.raw",
                b"base-release-report-v4\n",
            ),
            "configuration_path": (
                "bindings/stable-default-config.raw",
                b"stable-default-config-v4\n",
            ),
        }
        for path, raw in binding_leaves.values():
            self.fixture.put(path, raw)
        request: dict[str, Any] = {
            "schema_version": v3_binder.BIND_REQUEST_VERSION,
            "candidate_id": self.document["candidate_id"],
            "binding_evidence": {name: path for name, (path, _raw) in binding_leaves.items()},
            "reconstructed_baseline": {"manifest_path": self._path(baseline["manifest"])},
            "candidate": self._phase_request(candidate, candidate_phase=True),
            "rollback": self._phase_request(rollback, candidate_phase=False),
            "candidate_artifacts": self._prepared_artifact_paths("candidate"),
            "rollback_artifacts": self._prepared_artifact_paths("rollback"),
            "atomic_switch": self._prepared_atomic_paths(),
        }
        self.fixture.put(request_path, request)
        return request_path, request

    def _rewrite_request(self, path: str, document: dict[str, Any]) -> None:
        self.fixture.put(path, document)

    def _artifact_bytes(self, descriptor: Mapping[str, Any]) -> bytes:
        return (self.root / self._path(descriptor)).read_bytes()

    def _input(self, name: str, raw: bytes, mode: int) -> Path:
        path = self.inputs / name
        path.write_bytes(raw)
        path.chmod(mode)
        return path

    def _preparation_request(self) -> prepare.PreparationRequest:
        candidate = self.document["candidate_artifacts"]
        rollback = self.document["rollback_artifacts"]
        assert isinstance(candidate, Mapping)
        assert isinstance(rollback, Mapping)
        return prepare.PreparationRequest(
            evidence_root=self.root,
            candidate_binary=self._input("candidate-bin", self._artifact_bytes(candidate["binary"]), 0o700),
            candidate_bundle=self._input("candidate-bundle", self._artifact_bytes(candidate["bundle"]), 0o600),
            candidate_image_inspect=self._input(
                "candidate-image.json", self._artifact_bytes(candidate["image_inspect"]), 0o600
            ),
            rollback_binary=self._input("rollback-bin", self._artifact_bytes(rollback["binary"]), 0o700),
            rollback_bundle=self._input("rollback-bundle", self._artifact_bytes(rollback["bundle"]), 0o600),
            rollback_image_inspect=self._input(
                "rollback-image.json", self._artifact_bytes(rollback["image_inspect"]), 0o600
            ),
        )

    def _compose(
        self,
        *,
        request_path: str | None = None,
        v3_name: str | None = None,
        v4_name: str | None = None,
    ) -> dict[str, Any]:
        with mock.patch.object(atomic, "_rename_exchange", side_effect=fake_exchange):
            return binder.capture_and_bind_rollback_terminal_v4(
                self.preparation_request,
                request_path or self.request_path,
                v3_name or self.v3_name,
                v4_name or self.v4_name,
            )

    def test_public_producer_closes_preparation_transaction_v3_and_v4(self) -> None:
        report = self._compose()
        self.assertEqual(report["schema_version"], checker.ROLLBACK_V4_REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(
            checker.verify_rollback_provenance_v4(
                self.root,
                self.v4_name,
                require_completion=True,
            ),
            report,
        )
        for name in (self.v4_name, f"{self.v4_name}.intent", f"{self.v4_name}.complete"):
            self.assertTrue((self.root / name).is_file())
        final = self.root / f"{self.v4_name}.complete"
        intent = self.root / f"{self.v4_name}.intent"
        self.assertEqual((final.stat().st_dev, final.stat().st_ino), (intent.stat().st_dev, intent.stat().st_ino))
        for marker in (final, intent):
            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            self.assertEqual(marker.stat().st_nlink, 2)
        self.assertTrue((self.root / prepare.SNAPSHOT_DIRECTORY_NAME / prepare.COMPLETE_MARKER_NAME).is_file())
        self.assertTrue((self.root / transaction.TRANSACTION_DIRECTORY_NAME / transaction.COMPLETE_MARKER_NAME).is_file())

    def test_ambiguous_preparation_stops_transaction_and_terminal_chain(self) -> None:
        original = common._fsync_checked

        def fail_preparation_parent(descriptor: int, label: str) -> None:
            if label == "artifact preparation completion marker parent directory":
                common._fail("durability-failure", "fixture preparation final marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_preparation_parent):
            with self.assertRaises(binder.RollbackV4TerminalBindError) as raised:
                self._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / prepare.SNAPSHOT_DIRECTORY_NAME / prepare.COMPLETE_MARKER_NAME).is_file())
        self.assertFalse((self.root / transaction.TRANSACTION_DIRECTORY_NAME).exists())
        self.assertFalse((self.root / self.v3_name).exists())
        self.assertFalse((self.root / self.v4_name).exists())
        with self.assertRaises(binder.RollbackV4TerminalBindError) as retry:
            self._compose(v4_name="retry-v4.json")
        self.assert_reason(retry, "create-only-collision")
        self.assertFalse((self.root / "retry-v4.json").exists())

    def test_ambiguous_transaction_stops_v3_and_v4_chain(self) -> None:
        original = common._fsync_checked

        def fail_transaction_parent(descriptor: int, label: str) -> None:
            if label == "atomic transaction completion marker parent directory":
                common._fail("durability-failure", "fixture transaction final marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_transaction_parent):
            with self.assertRaises(binder.RollbackV4TerminalBindError) as raised:
                self._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / transaction.TRANSACTION_DIRECTORY_NAME / transaction.COMPLETE_MARKER_NAME).is_file())
        self.assertFalse((self.root / self.v3_name).exists())
        self.assertFalse((self.root / self.v4_name).exists())
        with self.assertRaises(binder.RollbackV4TerminalBindError) as retry:
            self._compose(v4_name="retry-v4.json")
        self.assert_reason(retry, "create-only-collision")
        self.assertFalse((self.root / "retry-v4.json").exists())

    def test_ambiguous_v4_pair_is_structural_only_and_cannot_restart_chain(self) -> None:
        original = common._fsync_checked

        def fail_v4_parent(descriptor: int, label: str) -> None:
            if label == "rollback v4 completion marker parent directory":
                common._fail("durability-failure", "fixture v4 final marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_v4_parent):
            with self.assertRaises(binder.RollbackV4TerminalBindError) as raised:
                self._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / self.v3_name).is_file())
        self.assertTrue((self.root / self.v4_name).is_file())
        self.assertEqual(
            checker.verify_rollback_provenance_v4(
                self.root,
                self.v4_name,
                require_completion=True,
            )["status"],
            "bound",
        )
        with self.assertRaises(binder.RollbackV4TerminalBindError) as retry:
            self._compose(v4_name="retry-v4.json")
        self.assert_reason(retry, "create-only-collision")
        self.assertFalse((self.root / "retry-v4.json").exists())

    def test_existing_preparation_and_transaction_cannot_be_reopened_into_v4(self) -> None:
        with mock.patch.object(atomic, "_rename_exchange", side_effect=fake_exchange):
            prepare.prepare_artifacts(self.preparation_request)
            transaction.capture_atomic_transaction(self.root)
        self.assertTrue((self.root / transaction.TRANSACTION_DIRECTORY_NAME / "session.json").is_file())
        with self.assertRaises(binder.RollbackV4TerminalBindError) as raised:
            self._compose()
        self.assert_reason(raised, "create-only-collision")
        self.assertFalse((self.root / self.v3_name).exists())
        self.assertFalse((self.root / self.v4_name).exists())

    def test_v3_alternate_artifact_path_fails_before_v4_publication(self) -> None:
        candidate_descriptor = self.document["candidate_artifacts"]
        assert isinstance(candidate_descriptor, Mapping)
        request_path, request = self._bind_request(request_path="requests/candidate-artifact.json")
        candidate_artifacts = request["candidate_artifacts"]
        assert isinstance(candidate_artifacts, dict)
        candidate_artifacts["binary_path"] = self._path(
            self.fixture.put(
                "alternate/candidate-binary.raw",
                self._artifact_bytes(candidate_descriptor["binary"]),
            )
        )
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackV4TerminalBindError) as raised:
            self._compose(
                request_path=request_path,
                v3_name="candidate-artifact-v3.json",
                v4_name="candidate-artifact-v4.json",
            )
        self.assert_reason(raised, "candidate-artifact-transaction-mismatch")
        self.assertTrue((self.root / "candidate-artifact-v3.json").is_file())
        self.assertFalse((self.root / "candidate-artifact-v4.json").exists())

    def test_v3_alternate_atomic_path_fails_before_v4_publication(self) -> None:
        atomic_descriptor = self.document["atomic_switch"]
        assert isinstance(atomic_descriptor, Mapping)
        request_path, request = self._bind_request(request_path="requests/atomic-switch.json")
        atomic_paths = request["atomic_switch"]
        assert isinstance(atomic_paths, dict)
        atomic_paths["pre_active_stat_path"] = self._path(
            self.fixture.put(
                "alternate/pre-active-stat.json",
                self._artifact_bytes(atomic_descriptor["pre_active_stat"]),
            )
        )
        self._rewrite_request(request_path, request)
        with self.assertRaises(binder.RollbackV4TerminalBindError) as raised:
            self._compose(
                request_path=request_path,
                v3_name="atomic-switch-v3.json",
                v4_name="atomic-switch-v4.json",
            )
        self.assert_reason(raised, "atomic-switch-transaction-mismatch")
        self.assertTrue((self.root / "atomic-switch-v3.json").is_file())
        self.assertFalse((self.root / "atomic-switch-v4.json").exists())

    def test_v3_terminal_looking_sidecar_is_rejected_before_transaction(self) -> None:
        (self.root / f"{self.v3_name}.intent").write_bytes(b"unrelated")
        with self.assertRaises(binder.RollbackV4TerminalBindError) as raised:
            self._compose()
        self.assert_reason(raised, "output-name-collision")
        self.assertTrue((self.root / prepare.SNAPSHOT_DIRECTORY_NAME).is_dir())
        self.assertFalse((self.root / transaction.TRANSACTION_DIRECTORY_NAME).exists())
        self.assertFalse((self.root / self.v3_name).exists())
        self.assertFalse((self.root / self.v4_name).exists())

    def test_schema_name_limit_and_public_surface_match_the_contract(self) -> None:
        schema_candidates = (
            Path(__file__).with_name("rollback-receipt-v4.schema.json"),
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rollback-receipt-v4.schema.json",
        )
        schema_path = next(path for path in schema_candidates if path.is_file())
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$defs"]["rootJsonLeaf"]["maxLength"], 128)
        self.assertEqual(len("a" * 123 + ".json"), 128)
        self.assertEqual(binder._manifest_name("a" * 123 + ".json", "fixture"), "a" * 123 + ".json")
        with self.assertRaises(binder.RollbackV4TerminalBindError) as raised:
            binder._manifest_name("a" * 124 + ".json", "fixture")
        self.assert_reason(raised, "invalid-manifest-name")

        source = Path(binder.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        public_functions = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        }
        self.assertEqual(public_functions, {"capture_and_bind_rollback_terminal_v4"})
        self.assertNotIn("def _publish_rollback_terminal_v4", source)
        self.assertNotIn("def _capture_and_bind_rollback_terminal_v4", source)
        self.assertNotIn("def capture_and_bind_rollback_terminal_v4_on_held_switch_fd", source)
        self.assertFalse(hasattr(binder, "_publish_rollback_terminal_v4_after_transaction_on_held_switch_fd"))
        self.assertFalse(hasattr(binder, "_capture_and_bind_rollback_terminal_v4_on_held_switch_fd"))
        self.assertFalse(hasattr(binder, "capture_and_bind_rollback_terminal_v4_on_held_switch_fd"))
        for forbidden in ("argparse", "subprocess.run", "socket.", "docker ", "ssh ", "systemctl ", "nvidia"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
