#!/usr/bin/env python3
"""CPU-only hostile tests for the fixed RC3 rollback receipt compositor."""

from __future__ import annotations

import ast
import fcntl
import inspect
import os
import stat
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import capture_rc3_rollback_atomic_switch_v1 as atomic
import capture_rc3_rollback_atomic_transaction_v1 as transaction
import compose_rc3_rollback_finalizer_receipt_v1 as composer
import finalize_rc3_rollback_candidate_source_v4 as fixed_finalizer
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common
import replay_rc3_rollback_candidate_source_v1 as candidate_source
import test_replay_rc3_rollback_candidate_source_v1 as candidate_fixtures
import test_write_rc3_rollback_candidate_source_bind_request_v1 as writer_fixtures
import write_rc3_rollback_finalizer_receipt_v1 as receipt


class RollbackFinalizerReceiptFixture:
    """Build fixed dynamic evidence but leave preparation/transaction absent."""

    def _set_up_rollback_receipt_fixture(self) -> None:
        self.candidate_fixture = candidate_fixtures.CandidateSourceJoinTests(
            methodName="runTest"
        )
        self.candidate_fixture.setUp()
        environment = self.candidate_fixture.environment
        fixture = environment["fixture"]
        self.fixture = fixture
        self.root = fixture.root

        # Reuse the existing CPU-only rollback-phase capturer, but deliberately
        # do not call its setup: that setup would pre-create the preparation and
        # transaction this compositor must own.
        phase_support = writer_fixtures.WriteRollbackCandidateSourceBindRequestTests(
            methodName="runTest"
        )
        phase_support.fixture = fixture
        phase_support.root = self.root
        phase_support._capture_rollback_phase()  # noqa: SLF001
        self.request = self._preparation_request()

    def _tear_down_rollback_receipt_fixture(self) -> None:
        self.candidate_fixture.doCleanups()

    def _preparation_request(self) -> prepare.PreparationRequest:
        baseline = self.root / "reproductions" / "a"
        inputs = self.fixture.base / "terminal-compositor-artifact-inputs"
        inputs.mkdir(mode=0o700, exist_ok=True)
        inputs.chmod(0o700)

        def input_file(name: str, raw: bytes, mode: int) -> Path:
            path = inputs / name
            path.write_bytes(raw)
            path.chmod(mode)
            return path

        rollback_binary = baseline / "riley-server"
        rollback_bundle = baseline / "riley.bundle.tar.zst"
        rollback_image = baseline / "docker-image-inspect.json"
        return prepare.PreparationRequest(
            evidence_root=self.root,
            candidate_binary=input_file("candidate-binary", b"candidate compositor binary\n", 0o700),
            candidate_bundle=input_file("candidate-bundle", b"candidate compositor bundle\n", 0o600),
            candidate_image_inspect=input_file(
                "candidate-image-inspect.json",
                rollback_image.read_bytes(),
                0o600,
            ),
            rollback_binary=input_file("rollback-binary", rollback_binary.read_bytes(), 0o700),
            rollback_bundle=input_file("rollback-bundle", rollback_bundle.read_bytes(), 0o600),
            rollback_image_inspect=input_file(
                "rollback-image-inspect.json",
                rollback_image.read_bytes(),
                0o600,
            ),
        )

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext[BaseException],
        code: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)


class ComposeRollbackFinalizerReceiptTests(
    unittest.TestCase,
    RollbackFinalizerReceiptFixture,
):
    def setUp(self) -> None:
        self._set_up_rollback_receipt_fixture()

    def tearDown(self) -> None:
        self._tear_down_rollback_receipt_fixture()

    def _compose(self) -> dict[str, Any]:
        root_fd = common.open_private_evidence_directory(
            self.root,
            "rollback receipt compositor test root",
        )
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(
                atomic,
                "_rename_exchange",
                side_effect=writer_fixtures._fake_exchange,
            ):
                return composer._prepare_transaction_and_write_fixed_receipt_on_held_root_fd(  # noqa: SLF001
                    root_fd,
                    self.request,
                )
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

    def test_composes_one_fixed_completed_raw_receipt(self) -> None:
        document = self._compose()
        self.assertEqual(document["schema_version"], receipt.RECEIPT_VERSION)
        self.assertEqual(document["status"], "completed")
        self.assertEqual(document["qualification_status"], "not-run")
        self.assertEqual(document["authority"], receipt.RAW_FINALIZER_NORMAL_RETURN_AUTHORITY)
        for name in (
            prepare.SNAPSHOT_DIRECTORY_NAME,
            prepare.ARTIFACT_DIRECTORY_NAME,
            prepare.SWITCH_DIRECTORY_NAME,
            transaction.ATOMIC_CAPTURE_DIRECTORY_NAME,
            transaction.TRANSACTION_DIRECTORY_NAME,
            fixed_finalizer.ROLLBACK_V3_MANIFEST_NAME,
            fixed_finalizer.ROLLBACK_V4_MANIFEST_NAME,
            receipt.RECEIPT_NAME,
        ):
            self.assertTrue((self.root / name).exists(), name)
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

    def test_fixed_receipt_collision_stops_before_artifact_preparation(self) -> None:
        (self.root / receipt.RECEIPT_NAME).write_bytes(b"occupied\n")
        with mock.patch.object(
            prepare,
            "_prepare_artifacts_then_terminal_success_held_root_fd",
            side_effect=AssertionError("preparation must not run after final output collision"),
        ):
            with self.assertRaises(composer.RollbackFinalizerReceiptComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / prepare.SNAPSHOT_DIRECTORY_NAME).exists())
        self.assertFalse((self.root / transaction.TRANSACTION_DIRECTORY_NAME).exists())

    def test_dynamic_preflight_stops_before_artifact_preparation(self) -> None:
        failure = candidate_source.CandidateSourceJoinError("fixture dynamic evidence failure")
        failure.reason_code = "fixture-dynamic-preflight"  # type: ignore[attr-defined]
        with mock.patch.object(
            candidate_source,
            "_replay_candidate_source_join_on_held_root_fd",
            side_effect=failure,
        ), mock.patch.object(
            prepare,
            "_prepare_artifacts_then_terminal_success_held_root_fd",
            side_effect=AssertionError("preparation must not run after dynamic preflight failure"),
        ):
            with self.assertRaises(composer.RollbackFinalizerReceiptComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "fixture-dynamic-preflight")
        self.assertFalse((self.root / prepare.SNAPSHOT_DIRECTORY_NAME).exists())
        self.assertFalse((self.root / transaction.TRANSACTION_DIRECTORY_NAME).exists())

    def test_preparation_terminal_failure_never_invokes_transaction(self) -> None:
        original = common._fsync_checked

        def fail_preparation_marker(descriptor: int, label: str) -> None:
            if label == "artifact preparation completion marker parent directory":
                common._fail("durability-failure", "fixture preparation marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_preparation_marker), mock.patch.object(
            transaction,
            "_capture_atomic_transaction_then_terminal_success_held_switch_fd",
            side_effect=AssertionError("transaction must not run after preparation failure"),
        ):
            with self.assertRaises(composer.RollbackFinalizerReceiptComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertFalse((self.root / transaction.TRANSACTION_DIRECTORY_NAME).exists())
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def test_transaction_terminal_failure_never_invokes_fixed_finalizer(self) -> None:
        original = common._fsync_checked

        def fail_transaction_marker(descriptor: int, label: str) -> None:
            if label == "atomic transaction completion marker parent directory":
                common._fail("durability-failure", "fixture transaction marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_transaction_marker), mock.patch.object(
            receipt,
            "_finalize_and_write_rollback_receipt_on_held_root_switch_fds",
            side_effect=AssertionError("receipt finalizer must not run after transaction failure"),
        ):
            with self.assertRaises(composer.RollbackFinalizerReceiptComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / transaction.TRANSACTION_DIRECTORY_NAME).exists())
        self.assertFalse((self.root / fixed_finalizer.ROLLBACK_V4_MANIFEST_NAME).exists())
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def test_v4_terminal_ambiguity_never_writes_receipt(self) -> None:
        original = common._fsync_checked

        def fail_v4_marker(descriptor: int, label: str) -> None:
            if label == "fixed rollback v4 completion marker parent directory":
                common._fail("durability-failure", "fixture v4 marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_v4_marker):
            with self.assertRaises(composer.RollbackFinalizerReceiptComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / fixed_finalizer.ROLLBACK_V4_MANIFEST_NAME).is_file())
        self.assertTrue((self.root / f"{fixed_finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete").is_file())
        self.assertFalse((self.root / receipt.RECEIPT_NAME).exists())

    def test_receipt_terminal_ambiguity_is_not_retried(self) -> None:
        original = common._fsync_checked

        def fail_receipt_marker(descriptor: int, label: str) -> None:
            if label == "rollback finalizer receipt completion marker parent directory":
                common._fail("durability-failure", "fixture receipt marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_receipt_marker):
            with self.assertRaises(composer.RollbackFinalizerReceiptComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / receipt.RECEIPT_NAME).is_file())
        self.assertTrue((self.root / f"{receipt.RECEIPT_NAME}.complete").is_file())
        with self.assertRaises(composer.RollbackFinalizerReceiptComposeError) as retry:
            self._compose()
        self.assert_reason(retry, "output-name-collision")

    def test_terminal_receipt_has_no_post_publication_switch_replay(self) -> None:
        original_publish = common.publish_create_only_hardlink
        original_require = transaction._require_held_switch_fd
        published = False

        def mark_receipt_publish(*args: Any, **kwargs: Any) -> None:
            nonlocal published
            original_publish(*args, **kwargs)
            destination = args[2] if len(args) >= 3 else kwargs.get("destination_name")
            if destination == f"{receipt.RECEIPT_NAME}.complete":
                published = True

        def reject_post_publication_require(root_fd: int, switch_fd: int) -> None:
            if published:
                raise AssertionError("terminal receipt must not be followed by a switch replay")
            original_require(root_fd, switch_fd)

        with mock.patch.object(
            common,
            "publish_create_only_hardlink",
            side_effect=mark_receipt_publish,
        ), mock.patch.object(
            transaction,
            "_require_held_switch_fd",
            side_effect=reject_post_publication_require,
        ):
            document = self._compose()
        self.assertEqual(document["status"], "completed")

    def test_terminal_receipt_ignores_late_preparation_fd_close_error(self) -> None:
        original_publish = common.publish_create_only_hardlink
        original_close = os.close
        published = False

        class PreparationOsProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(os, name)

            @staticmethod
            def close(descriptor: int) -> None:
                if published:
                    raise OSError("fixture late preparation close failure")
                original_close(descriptor)

        def mark_receipt_publish(*args: Any, **kwargs: Any) -> None:
            nonlocal published
            original_publish(*args, **kwargs)
            destination = args[2] if len(args) >= 3 else kwargs.get("destination_name")
            if destination == f"{receipt.RECEIPT_NAME}.complete":
                published = True

        with mock.patch.object(
            common,
            "publish_create_only_hardlink",
            side_effect=mark_receipt_publish,
        ), mock.patch.object(prepare, "os", PreparationOsProxy()):
            document = self._compose()
        self.assertEqual(document["status"], "completed")

    def test_private_surface_exposes_no_cli_or_dynamic_terminal_inputs(self) -> None:
        source = Path(composer.__file__).read_text(encoding="utf-8")
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
            "bind_raw_rc3_rollback_terminal_v4",
            "capture_and_bind_rollback_terminal_v4",
        ):
            self.assertNotIn(forbidden, source)
        parameters = inspect.signature(
            composer._prepare_transaction_and_write_fixed_receipt_on_held_root_fd  # noqa: SLF001
        ).parameters
        self.assertEqual(list(parameters), ["root_fd", "request"])
        for forbidden_parameter in (
            "candidate_id",
            "configuration_profile",
            "target",
            "descriptor",
            "output_name",
            "manifest_name",
            "receipt_name",
            "continuation",
        ):
            self.assertNotIn(forbidden_parameter, parameters)


if __name__ == "__main__":
    unittest.main()
