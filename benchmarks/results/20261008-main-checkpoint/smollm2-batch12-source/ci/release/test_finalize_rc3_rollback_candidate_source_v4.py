#!/usr/bin/env python3
"""CPU-only hostile tests for the fixed same-stack rollback v3/v4 finalizer."""

from __future__ import annotations

import ast
import dataclasses
import fcntl
import inspect
import os
import stat
import unittest
from pathlib import Path
from typing import Any, Callable, TypeVar
from unittest import mock

import check_rc3_rollback_provenance_v4 as v4_checker
import finalize_rc3_rollback_candidate_source_v4 as finalizer
import prepare_rc3_rollback_artifacts_v1 as artifact_prepare
import provenance_v2_common as common
import test_write_rc3_rollback_candidate_source_bind_request_v1 as writer_fixtures
import write_rc3_rollback_candidate_source_bind_request_v1 as writer


T = TypeVar("T")


class FinalizeRollbackCandidateSourceV4Tests(unittest.TestCase):
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
            "fixed rollback finalizer test root",
        )
        switch_fd: int | None = None
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            switch_fd = common.open_private_child_directory(
                root_fd,
                artifact_prepare.SWITCH_DIRECTORY_NAME,
                "fixed rollback finalizer test switch",
            )
            fcntl.flock(switch_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return call(root_fd, switch_fd)
        finally:
            if switch_fd is not None:
                fcntl.flock(switch_fd, fcntl.LOCK_UN)
                os.close(switch_fd)
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

    def _finalize(self) -> dict[str, Any]:
        return self._with_held_fds(
            finalizer._finalize_rollback_candidate_source_v4_on_held_root_switch_fds  # noqa: SLF001
        )

    def test_finalizes_fixed_request_v3_and_v4_from_one_held_stack(self) -> None:
        report = self._finalize()
        self.assertEqual(report["schema_version"], v4_checker.ROLLBACK_V4_REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(
            v4_checker.verify_rollback_provenance_v4(
                self.root,
                finalizer.ROLLBACK_V4_MANIFEST_NAME,
                require_completion=True,
            ),
            report,
        )
        for name in (
            writer.BIND_REQUEST_NAME,
            finalizer.ROLLBACK_V3_MANIFEST_NAME,
            finalizer.ROLLBACK_V4_MANIFEST_NAME,
            f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.intent",
            f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete",
        ):
            self.assertTrue((self.root / name).is_file())
        for name in (
            f"{writer.BIND_REQUEST_NAME}.intent",
            f"{writer.BIND_REQUEST_NAME}.complete",
            f"{finalizer.ROLLBACK_V3_MANIFEST_NAME}.intent",
            f"{finalizer.ROLLBACK_V3_MANIFEST_NAME}.complete",
        ):
            self.assertFalse((self.root / name).exists())
        intent = self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.intent"
        complete = self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete"
        self.assertEqual(stat.S_IMODE(intent.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(complete.stat().st_mode), 0o600)
        self.assertEqual(intent.stat().st_nlink, 2)
        self.assertEqual(complete.stat().st_nlink, 2)
        self.assertEqual((intent.stat().st_dev, intent.stat().st_ino), (complete.stat().st_dev, complete.stat().st_ino))

    def test_request_descriptor_drift_stops_before_v3(self) -> None:
        original = writer._write_fixed_candidate_source_bind_request_on_held_root_switch_fds

        def write_then_mutate(
            root_fd: int,
            switch_fd: int,
        ) -> writer.WrittenCandidateSourceBindRequest:
            written = original(root_fd, switch_fd)
            (self.root / writer.BIND_REQUEST_NAME).write_bytes(b"{}")
            return written

        with mock.patch.object(
            finalizer.writer,
            "_write_fixed_candidate_source_bind_request_on_held_root_switch_fds",
            side_effect=write_then_mutate,
        ):
            with self.assertRaises(
                finalizer.RollbackCandidateSourceFinalizerError
            ) as raised:
                self._finalize()
        self.assert_reason(raised, "bind-request-descriptor-drift")
        self.assertFalse((self.root / finalizer.ROLLBACK_V3_MANIFEST_NAME).exists())
        self.assertFalse((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).exists())

    def test_terminal_recheck_stops_after_v4_nonterminal_write(self) -> None:
        original = common.write_create_only_json

        def write_then_mutate(*args: Any, **kwargs: Any) -> Any:
            created = original(*args, **kwargs)
            if args[1] == finalizer.ROLLBACK_V4_MANIFEST_NAME:
                (self.root / writer.BIND_REQUEST_NAME).write_bytes(b"{}")
            return created

        with mock.patch.object(
            finalizer.common,
            "write_create_only_json",
            side_effect=write_then_mutate,
        ):
            with self.assertRaises(
                finalizer.RollbackCandidateSourceFinalizerError
            ) as raised:
                self._finalize()
        self.assert_reason(raised, "bind-request-descriptor-drift")
        self.assertTrue((self.root / finalizer.ROLLBACK_V3_MANIFEST_NAME).is_file())
        self.assertTrue((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).is_file())
        self.assertFalse((self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.intent").exists())
        self.assertFalse((self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete").exists())

    def test_recheck_rejects_each_typed_writer_component_drift(self) -> None:
        def check(root_fd: int, switch_fd: int) -> None:
            written = writer._write_fixed_candidate_source_bind_request_on_held_root_switch_fds(  # noqa: SLF001
                root_fd,
                switch_fd,
            )
            replayed = writer._replay_inputs(root_fd, switch_fd)  # noqa: SLF001
            static_drift = dataclasses.replace(
                replayed,
                candidate_source=dataclasses.replace(
                    replayed.candidate_source,
                    static_effective=dataclasses.replace(
                        replayed.candidate_source.static_effective,
                        static_bindings=dataclasses.replace(
                            replayed.candidate_source.static_effective.static_bindings,
                            candidate_id="riley-0.1.0-rc4",
                        ),
                    ),
                ),
            )
            candidate_drift = dataclasses.replace(
                replayed,
                candidate_source=dataclasses.replace(
                    replayed.candidate_source,
                    consumed_paths=frozenset(
                        set(replayed.candidate_source.consumed_paths) | {"synthetic-source-drift"}
                    ),
                ),
            )
            phase_drift = dataclasses.replace(
                replayed,
                rollback_phase=dataclasses.replace(
                    replayed.rollback_phase,
                    capture_name="synthetic-rollback-phase-drift",
                ),
            )
            transaction_descriptor = replayed.atomic_transaction.session_descriptor
            transaction_drift = dataclasses.replace(
                replayed,
                atomic_transaction=dataclasses.replace(
                    replayed.atomic_transaction,
                    session_descriptor=common.EvidenceDescriptor(
                        path=transaction_descriptor.path,
                        sha256="f" * 64,
                        byte_length=transaction_descriptor.byte_length,
                    ),
                ),
            )
            for drift, code in (
                (static_drift, "static-preparation-replay-drift"),
                (candidate_drift, "candidate-source-replay-drift"),
                (phase_drift, "rollback-phase-replay-drift"),
                (transaction_drift, "atomic-transaction-replay-drift"),
            ):
                with self.subTest(code=code), mock.patch.object(
                    finalizer.writer,
                    "_replay_inputs",
                    return_value=drift,
                ):
                    with self.assertRaises(
                        finalizer.RollbackCandidateSourceFinalizerError
                    ) as raised:
                        finalizer._recheck_written_state(root_fd, switch_fd, written)  # noqa: SLF001
                    self.assert_reason(raised, code)

        self._with_held_fds(check)

    def test_visible_v4_pair_after_final_link_sync_failure_is_ambiguous(self) -> None:
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "fixed rollback v4 completion marker parent directory":
                common._fail("durability-failure", "fixture final marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(
                finalizer.RollbackCandidateSourceFinalizerError
            ) as raised:
                self._finalize()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).is_file())
        self.assertTrue((self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.intent").is_file())
        self.assertTrue((self.root / f"{finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete").is_file())

    def test_later_output_collision_stops_before_fixed_request_writer(self) -> None:
        (self.root / finalizer.ROLLBACK_V4_MANIFEST_NAME).write_bytes(b"occupied\n")
        with mock.patch.object(
            finalizer.writer,
            "_write_fixed_candidate_source_bind_request_on_held_root_switch_fds",
            side_effect=AssertionError("writer must not run after finalizer output collision"),
        ):
            with self.assertRaises(
                finalizer.RollbackCandidateSourceFinalizerError
            ) as raised:
                self._finalize()
        self.assert_reason(raised, "output-name-collision")
        self.assertFalse((self.root / writer.BIND_REQUEST_NAME).exists())

    def test_held_fds_are_not_reopened_or_relocked(self) -> None:
        def finalize_without_reopen(root_fd: int, switch_fd: int) -> dict[str, Any]:
            with mock.patch.object(
                common,
                "open_private_evidence_directory",
                side_effect=AssertionError("finalizer must not reopen the root path"),
            ):
                return finalizer._finalize_rollback_candidate_source_v4_on_held_root_switch_fds(  # noqa: SLF001
                    root_fd,
                    switch_fd,
                )

        report = self._with_held_fds(finalize_without_reopen)
        self.assertEqual(report["status"], "bound")

    def test_private_surface_has_no_cli_or_injected_inputs(self) -> None:
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
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "def resume",
            "def retry",
            "capture_and_bind_rollback_terminal_v4",
        ):
            self.assertNotIn(forbidden, source)
        self.assertEqual(
            list(
                inspect.signature(
                    finalizer._finalize_rollback_candidate_source_v4_on_held_root_switch_fds  # noqa: SLF001
                ).parameters
            ),
            ["root_fd", "switch_fd"],
        )


if __name__ == "__main__":
    unittest.main()
