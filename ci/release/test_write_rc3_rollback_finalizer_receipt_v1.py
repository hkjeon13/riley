#!/usr/bin/env python3
"""CPU-only hostile tests for the private rollback finalizer receipt v1."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import inspect
import json
import os
import stat
import unittest
from pathlib import Path
from typing import Any, Callable, TypeVar
from unittest import mock

import check_c02_provenance_v2 as c02
import finalize_rc3_rollback_candidate_source_v4 as finalizer
import prepare_rc3_rollback_artifacts_v1 as artifact_prepare
import provenance_v2_common as common
import replay_rc3_rollback_candidate_source_v1 as candidate_source
import replay_rc3_rollback_operational_semantics_v1 as operational_semantics
import test_write_rc3_rollback_candidate_source_bind_request_v1 as writer_fixtures
import write_rc3_rollback_finalizer_receipt_v1 as receipt


T = TypeVar("T")


class RollbackFinalizerReceiptV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.writer_fixture = writer_fixtures.WriteRollbackCandidateSourceBindRequestTests(
            methodName="runTest"
        )
        self.writer_fixture.setUp()
        self.root = self.writer_fixture.root

    def tearDown(self) -> None:
        self.writer_fixture.tearDown()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext[BaseException],
        code: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def _with_held_fds(self, call: Callable[[int, int], T]) -> T:
        root_fd = common.open_private_evidence_directory(
            self.root,
            "rollback finalizer receipt test root",
        )
        switch_fd: int | None = None
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            switch_fd = common.open_private_child_directory(
                root_fd,
                artifact_prepare.SWITCH_DIRECTORY_NAME,
                "rollback finalizer receipt test switch",
            )
            fcntl.flock(switch_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return call(root_fd, switch_fd)
        finally:
            if switch_fd is not None:
                fcntl.flock(switch_fd, fcntl.LOCK_UN)
                os.close(switch_fd)
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

    def _write(self) -> dict[str, Any]:
        return self._with_held_fds(
            receipt._finalize_and_write_rollback_receipt_on_held_root_switch_fds  # noqa: SLF001
        )

    def _rewrite_candidate_shutdown(self, mutate: Callable[[dict[str, Any]], None]) -> None:
        artifact_path = self.root / candidate_source.SHUTDOWN_ARTIFACT_PATH
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
        mutate(document)
        raw = common.canonical_json_bytes(document)
        artifact_path.write_bytes(raw)
        marker = {
            "schema_version": c02.SHUTDOWN_MARKER_VERSION,
            "artifact_filename": "shutdown.json",
            "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        }
        (self.root / candidate_source.SHUTDOWN_MARKER_PATH).write_bytes(
            common.canonical_json_bytes(marker)
        )

    def test_publishes_one_same_stack_completed_raw_receipt(self) -> None:
        document = self._write()
        self.assertEqual(document["schema_version"], receipt.RECEIPT_VERSION)
        self.assertEqual(document["status"], "completed")
        self.assertEqual(document["qualification_status"], "not-run")
        self.assertEqual(
            document["authority"], receipt.RAW_FINALIZER_NORMAL_RETURN_AUTHORITY
        )
        self.assertEqual(document["reason_codes"], [])
        self.assertNotIn("operational_semantics", document)
        self.assertNotIn("semantic_receipt", document)
        self.assertEqual(
            document["finalizer_outputs"]["v4_manifest"]["path"],
            finalizer.ROLLBACK_V4_MANIFEST_NAME,
        )
        consumed_paths = document["candidate_source"]["consumed_paths"]
        self.assertEqual(consumed_paths, sorted(set(consumed_paths)))
        self.assertIn("source-audit/shutdown.json.complete", consumed_paths)
        self.assertIn("config-bridge/session.json", consumed_paths)
        receipt_path = self.root / receipt.RECEIPT_NAME
        self.assertEqual(receipt_path.read_bytes(), common.canonical_json_bytes(document))
        intent = self.root / f"{receipt.RECEIPT_NAME}.intent"
        complete = self.root / f"{receipt.RECEIPT_NAME}.complete"
        self.assertEqual(stat.S_IMODE(intent.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(complete.stat().st_mode), 0o600)
        self.assertEqual(intent.stat().st_nlink, 2)
        self.assertEqual(complete.stat().st_nlink, 2)
        self.assertEqual(
            (intent.stat().st_dev, intent.stat().st_ino),
            (complete.stat().st_dev, complete.stat().st_ino),
        )
        self.assertTrue((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).is_file())

    def test_operational_semantics_is_an_ephemeral_prepublication_veto(self) -> None:
        original = (
            receipt.operational_semantics._replay_rc3_rollback_operational_semantics_on_held_root_switch_fds  # noqa: SLF001
        )

        def snapshot_tree() -> tuple[str, ...]:
            return tuple(
                sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*"))
            )

        def replay_without_persistence(root_fd: int, switch_fd: int) -> dict[str, Any]:
            before = snapshot_tree()
            with mock.patch.object(
                common,
                "write_create_only",
                side_effect=AssertionError("operational veto must not write evidence"),
            ), mock.patch.object(
                common,
                "write_create_only_json",
                side_effect=AssertionError("operational veto must not write JSON evidence"),
            ), mock.patch.object(
                common,
                "publish_create_only_hardlink",
                side_effect=AssertionError("operational veto must not publish evidence"),
            ):
                report = original(root_fd, switch_fd)
            self.assertEqual(snapshot_tree(), before)
            return report

        with mock.patch.object(
            receipt.operational_semantics,
            "_replay_rc3_rollback_operational_semantics_on_held_root_switch_fds",
            side_effect=replay_without_persistence,
            autospec=True,
        ) as replay:
            document = self._write()
        self.assertEqual(replay.call_count, 1)
        self.assertEqual(document["authority"], receipt.RAW_FINALIZER_NORMAL_RETURN_AUTHORITY)

    def test_operational_veto_requires_the_typed_finalizer_closure(self) -> None:
        with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
            receipt._require_operational_semantics_veto(-1, -1, object())  # noqa: SLF001
        self.assert_reason(raised, "invalid-finalizer-result")

    def test_operational_veto_precedes_receipt_closure_replay_and_receipt_leaves(self) -> None:
        original_veto = receipt._require_operational_semantics_veto  # noqa: SLF001
        original_document = receipt._receipt_document_from_closure  # noqa: SLF001
        events: list[str] = []

        def record_veto(root_fd: int, switch_fd: int, closure: Any) -> None:
            self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())
            events.append("veto")
            original_veto(root_fd, switch_fd, closure)

        def record_document(*args: Any, **kwargs: Any) -> Any:
            self.assertEqual(events[0], "veto")
            self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())
            events.append("receipt-document")
            return original_document(*args, **kwargs)

        with mock.patch.object(
            receipt,
            "_require_operational_semantics_veto",
            side_effect=record_veto,
        ), mock.patch.object(
            receipt,
            "_receipt_document_from_closure",
            side_effect=record_document,
        ):
            document = self._write()
        self.assertEqual(events, ["veto", "receipt-document", "receipt-document"])
        self.assertEqual(document["status"], "completed")

    def test_operational_semantics_failure_stops_before_receipt_publication(self) -> None:
        error = operational_semantics.RollbackOperationalSemanticsError("fixture semantic veto")
        error.reason_code = "fixture-operational-semantics-failure"  # type: ignore[attr-defined]
        with mock.patch.object(
            receipt.operational_semantics,
            "_replay_rc3_rollback_operational_semantics_on_held_root_switch_fds",
            side_effect=error,
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "fixture-operational-semantics-failure")
        self.assertTrue((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).is_file())
        self.assertTrue((self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete").is_file())
        for name in (
            receipt.RECEIPT_NAME,
            f"{receipt.RECEIPT_NAME}.intent",
            f"{receipt.RECEIPT_NAME}.complete",
        ):
            self.assertFalse((self.root / name).exists(), name)

    def test_operational_semantics_veto_failure_cannot_be_resumed(self) -> None:
        error = operational_semantics.RollbackOperationalSemanticsError("fixture semantic veto")
        error.reason_code = "fixture-operational-semantics-failure"  # type: ignore[attr-defined]
        with mock.patch.object(
            receipt.operational_semantics,
            "_replay_rc3_rollback_operational_semantics_on_held_root_switch_fds",
            side_effect=error,
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError):
                self._write()
        with mock.patch.object(
            receipt.operational_semantics,
            "_replay_rc3_rollback_operational_semantics_on_held_root_switch_fds",
            side_effect=AssertionError("veto must not run on a resumed finalizer"),
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def test_real_non_drained_shutdown_veto_stops_before_receipt_publication(self) -> None:
        def make_non_drained(document: dict[str, Any]) -> None:
            document["final_metrics"]["request_states"]["active"] = 1

        self._rewrite_candidate_shutdown(make_non_drained)
        with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
            self._write()
        self.assert_reason(raised, "candidate-shutdown-not-drained")
        self.assertTrue((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).is_file())
        self.assertTrue((self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete").is_file())
        for name in (
            receipt.RECEIPT_NAME,
            f"{receipt.RECEIPT_NAME}.intent",
            f"{receipt.RECEIPT_NAME}.complete",
        ):
            self.assertFalse((self.root / name).exists(), name)

    def test_operational_semantics_closure_drift_stops_before_receipt_publication(self) -> None:
        original = (
            receipt.operational_semantics._replay_rc3_rollback_operational_semantics_on_held_root_switch_fds  # noqa: SLF001
        )

        def replay_then_drift(root_fd: int, switch_fd: int) -> dict[str, Any]:
            report = original(root_fd, switch_fd)
            report["candidate_id"] = "riley-0.1.0-rc3-drifted"
            return report

        with mock.patch.object(
            receipt.operational_semantics,
            "_replay_rc3_rollback_operational_semantics_on_held_root_switch_fds",
            side_effect=replay_then_drift,
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "operational-semantics-closure-mismatch")
        self.assertTrue((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).is_file())
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def test_receipt_output_collision_stops_before_finalizer(self) -> None:
        (self.root / receipt.RECEIPT_NAME).write_bytes(b"occupied\n")
        with mock.patch.object(
            receipt.finalizer,
            "_finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds",
            side_effect=AssertionError("finalizer must not run after receipt collision"),
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).exists())

    def test_receipt_reserved_sidecar_collision_stops_before_finalizer(self) -> None:
        (self.root / f"{receipt.RECEIPT_NAME}.complete").write_bytes(b"stale\n")
        with mock.patch.object(
            receipt.finalizer,
            "_finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds",
            side_effect=AssertionError("finalizer must not run after receipt sidecar collision"),
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).exists())

    def test_v4_final_link_ambiguity_never_creates_a_receipt(self) -> None:
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "fixed rollback v4 completion marker parent directory":
                common._fail("durability-failure", "fixture v4 final marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).is_file())
        self.assertTrue((self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.intent").is_file())
        self.assertTrue((self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete").is_file())
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def test_receipt_final_link_ambiguity_returns_no_successful_receipt(self) -> None:
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "rollback finalizer receipt completion marker parent directory":
                common._fail("durability-failure", "fixture receipt final marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / receipt.RECEIPT_NAME).is_file())
        self.assertTrue((self.root / f"{receipt.RECEIPT_NAME}.intent").is_file())
        self.assertTrue((self.root / f"{receipt.RECEIPT_NAME}.complete").is_file())

    def test_final_prepublication_closure_recheck_leaves_no_receipt_outputs(self) -> None:
        original = receipt._receipt_document_from_closure  # noqa: SLF001
        calls = 0

        def fail_final_recheck(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                receipt._fail(  # noqa: SLF001
                    "fixture-final-prepublication-recheck",
                    "fixture rejects the last closure replay before publication",
                )
            return original(*args, **kwargs)

        with mock.patch.object(
            receipt,
            "_receipt_document_from_closure",
            side_effect=fail_final_recheck,
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "fixture-final-prepublication-recheck")
        for name in (
            receipt.RECEIPT_NAME,
            f"{receipt.RECEIPT_NAME}.intent",
            f"{receipt.RECEIPT_NAME}.complete",
        ):
            self.assertFalse((self.root / name).exists(), name)

    def test_successful_terminal_publish_has_no_post_publish_closure_replay(self) -> None:
        original_publish = common.publish_create_only_hardlink
        original_document = receipt._receipt_document_from_closure  # noqa: SLF001
        original_operational = (
            receipt.operational_semantics._replay_rc3_rollback_operational_semantics_on_held_root_switch_fds  # noqa: SLF001
        )
        published = False

        def mark_terminal_publish(*args: Any, **kwargs: Any) -> None:
            nonlocal published
            original_publish(*args, **kwargs)
            destination = (
                args[2]
                if len(args) >= 3
                else kwargs.get("destination_name")
            )
            if destination == f"{receipt.RECEIPT_NAME}.complete":
                published = True

        def reject_post_publish_replay(*args: Any, **kwargs: Any) -> Any:
            if published:
                raise AssertionError("closure must not be replayed after terminal publication")
            return original_document(*args, **kwargs)

        def reject_post_publish_operational(*args: Any, **kwargs: Any) -> Any:
            if published:
                raise AssertionError("operational semantics must not run after terminal publication")
            return original_operational(*args, **kwargs)

        with mock.patch.object(
            common,
            "publish_create_only_hardlink",
            side_effect=mark_terminal_publish,
        ), mock.patch.object(
            receipt,
            "_receipt_document_from_closure",
            side_effect=reject_post_publish_replay,
        ), mock.patch.object(
            receipt.operational_semantics,
            "_replay_rc3_rollback_operational_semantics_on_held_root_switch_fds",
            side_effect=reject_post_publish_operational,
        ):
            document = self._write()
        self.assertEqual(document["status"], "completed")

    def test_post_finalizer_source_session_drift_stops_before_receipt_publication(self) -> None:
        original = receipt.finalizer._finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds

        def finalize_then_mutate(root_fd: int, switch_fd: int) -> Any:
            closure = original(root_fd, switch_fd)
            path = closure.written.candidate_source.source_capture.session.path
            (self.root / path).write_bytes(b"{}")
            return closure

        with mock.patch.object(
            receipt.finalizer,
            "_finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds",
            side_effect=finalize_then_mutate,
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError):
                self._write()
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def _assert_post_finalizer_drift_stops(
        self,
        select_path: Callable[[Any], str],
    ) -> None:
        original = receipt.finalizer._finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds

        def finalize_then_mutate(root_fd: int, switch_fd: int) -> Any:
            closure = original(root_fd, switch_fd)
            (self.root / select_path(closure)).write_bytes(b"{}")
            return closure

        with mock.patch.object(
            receipt.finalizer,
            "_finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds",
            side_effect=finalize_then_mutate,
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError):
                self._write()
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def test_post_finalizer_static_drift_stops_receipt(self) -> None:
        self._assert_post_finalizer_drift_stops(
            lambda closure: closure.written.static_bindings.freeze.path
        )

    def test_post_finalizer_rollback_phase_drift_stops_receipt(self) -> None:
        self._assert_post_finalizer_drift_stops(
            lambda closure: closure.written.rollback_phase.process_evidence["pre_stat"].path
        )

    def test_post_finalizer_transaction_drift_stops_receipt(self) -> None:
        self._assert_post_finalizer_drift_stops(
            lambda closure: closure.written.atomic_transaction.session_descriptor.path
        )

    def test_post_finalizer_bind_request_drift_stops_before_receipt_publication(self) -> None:
        original = receipt.finalizer._finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds

        def finalize_then_mutate(root_fd: int, switch_fd: int) -> Any:
            closure = original(root_fd, switch_fd)
            (self.root / closure.written.request_descriptor.path).write_bytes(b"{}")
            return closure

        with mock.patch.object(
            receipt.finalizer,
            "_finalize_rollback_candidate_source_v4_with_closure_on_held_root_switch_fds",
            side_effect=finalize_then_mutate,
        ):
            with self.assertRaises(receipt.RollbackFinalizerReceiptError) as raised:
                self._write()
        self.assert_reason(raised, "bind-request-descriptor-drift")
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def test_held_descriptors_are_not_reopened_or_relocked(self) -> None:
        def write_without_reopen(root_fd: int, switch_fd: int) -> dict[str, Any]:
            with mock.patch.object(
                common,
                "open_private_evidence_directory",
                side_effect=AssertionError("receipt must not reopen the evidence-root path"),
            ):
                return receipt._finalize_and_write_rollback_receipt_on_held_root_switch_fds(  # noqa: SLF001
                    root_fd,
                    switch_fd,
                )

        document = self._with_held_fds(write_without_reopen)
        self.assertEqual(document["status"], "completed")

    def test_schema_is_closed_and_receipt_surface_has_no_cli_or_operational_imports(self) -> None:
        source = Path(receipt.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        public_functions = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        }
        self.assertEqual(public_functions, set())
        for forbidden in (
            "import argparse",
            "def main(",
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "def resume",
            "def retry",
            "import fcntl",
            "flock(",
            "open_private_evidence_directory(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertEqual(
            list(
                inspect.signature(
                    receipt._finalize_and_write_rollback_receipt_on_held_root_switch_fds  # noqa: SLF001
                ).parameters
            ),
            ["root_fd", "switch_fd"],
        )
        schema_path = (
            Path(receipt.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-rollback-finalizer-receipt-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        receipt_schema = schema["$defs"]["receipt"]
        self.assertFalse(receipt_schema["additionalProperties"])
        self.assertEqual(
            receipt_schema["properties"]["authority"]["const"],
            receipt.RAW_FINALIZER_NORMAL_RETURN_AUTHORITY,
        )
        self.assertIn("never proves host rollback", schema["description"])


if __name__ == "__main__":
    unittest.main()
