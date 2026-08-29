#!/usr/bin/env python3
"""CPU-only guards for the private authenticated rollback receipt entry."""

from __future__ import annotations

import ast
import fcntl
import inspect
import os
import unittest
from pathlib import Path
from unittest import mock

import capture_rc3_rollback_atomic_switch_v1 as atomic
import compose_rc3_rollback_finalizer_receipt_v1 as composer
import finalize_rc3_rollback_finalizer_receipt_v1 as finalizer
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common
import test_write_rc3_rollback_candidate_source_bind_request_v1 as writer_fixtures
import write_rc3_rollback_finalizer_receipt_v1 as receipt
from test_compose_rc3_rollback_finalizer_receipt_v1 import (
    RollbackFinalizerReceiptFixture,
)


class FinalizeRollbackFinalizerReceiptTests(
    unittest.TestCase,
    RollbackFinalizerReceiptFixture,
):
    def setUp(self) -> None:
        self._set_up_rollback_receipt_fixture()

    def tearDown(self) -> None:
        self._tear_down_rollback_receipt_fixture()

    def _finalize(self) -> dict:
        with mock.patch.object(
            atomic,
            "_rename_exchange",
            side_effect=writer_fixtures._fake_exchange,
        ):
            return finalizer._finalize_authenticated_rollback_raw_once(self.request)  # noqa: SLF001

    def test_locks_root_once_then_enters_private_compositor(self) -> None:
        original_flock = fcntl.flock
        calls: list[int] = []

        def record_flock(descriptor: int, operation: int) -> None:
            calls.append(operation)
            original_flock(descriptor, operation)

        expected = {"status": "completed", "qualification_status": "not-run"}
        with mock.patch.object(finalizer.fcntl, "flock", side_effect=record_flock), mock.patch.object(
            composer,
            "_prepare_transaction_and_write_fixed_receipt_on_held_root_fd",
            return_value=expected,
        ) as composed:
            self.assertEqual(
                finalizer._finalize_authenticated_rollback_raw_once(self.request),  # noqa: SLF001
                expected,
            )
        self.assertEqual(calls, [fcntl.LOCK_EX | fcntl.LOCK_NB, fcntl.LOCK_UN])
        self.assertEqual(composed.call_count, 1)
        self.assertEqual(composed.call_args.args, (mock.ANY, self.request))

    def test_compositor_failure_preserves_reason_without_second_attempt(self) -> None:
        error = composer.RollbackFinalizerReceiptComposeError("fixture terminal ambiguity")
        error.reason_code = "ambiguous-terminal-publication"  # type: ignore[attr-defined]
        with mock.patch.object(
            composer,
            "_prepare_transaction_and_write_fixed_receipt_on_held_root_fd",
            side_effect=error,
        ) as composed:
            with self.assertRaises(finalizer.AuthenticatedRollbackFinalizationError) as raised:
                finalizer._finalize_authenticated_rollback_raw_once(self.request)  # noqa: SLF001
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertEqual(composed.call_count, 1)

    def test_real_finalizer_creates_one_fixed_completed_receipt(self) -> None:
        document = self._finalize()
        self.assertEqual(document["schema_version"], receipt.RECEIPT_VERSION)
        self.assertEqual(document["status"], "completed")
        self.assertTrue((self.root / receipt.RECEIPT_NAME).is_file())
        self.assertTrue((self.root / f"{receipt.RECEIPT_NAME}.complete").is_file())

    def test_rejects_invalid_request_before_opening_evidence(self) -> None:
        with mock.patch.object(
            common,
            "open_private_evidence_directory",
            side_effect=AssertionError("invalid request must fail before opening evidence"),
        ):
            with self.assertRaises(finalizer.AuthenticatedRollbackFinalizationError) as raised:
                finalizer._finalize_authenticated_rollback_raw_once(object())  # type: ignore[arg-type] # noqa: SLF001
        self.assert_reason(raised, "invalid-preparation-request")

    def test_private_surface_has_no_cli_or_dynamic_terminal_inputs(self) -> None:
        source = Path(finalizer.__file__).read_text(encoding="utf-8")
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
            "add_argument",
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "def resume",
            "def retry",
            "publish_create_only_hardlink",
        ):
            self.assertNotIn(forbidden, source)
        parameters = inspect.signature(
            finalizer._finalize_authenticated_rollback_raw_once  # noqa: SLF001
        ).parameters
        self.assertEqual(list(parameters), ["request"])
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
