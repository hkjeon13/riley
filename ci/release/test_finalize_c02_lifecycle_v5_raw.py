#!/usr/bin/env python3
"""CPU-only guards for the authenticated runner's private raw-v5 finalizer."""

from __future__ import annotations

import ast
import fcntl
import inspect
import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import check_c02_provenance_v2 as checker
import compose_c02_lifecycle_v5_raw as composer
import finalize_c02_lifecycle_v5_raw as finalizer
import provenance_v2_common as common
import test_write_c02_lifecycle_bind_request_v5 as writer_fixtures


class FinalizeC02LifecycleV5RawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.writer_fixture = writer_fixtures.WriteC02LifecycleBindRequestV5Tests(
            methodName="runTest"
        )
        self.writer_fixture.setUp()
        self.root = self.writer_fixture.root
        self.bridge_report = self.writer_fixture.bridge_report
        self.candidate_id = self.writer_fixture.fixture.candidate_id

    def tearDown(self) -> None:
        self.writer_fixture.tearDown()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _finalize(self) -> dict:
        return finalizer._finalize_authenticated_v5_raw_once(  # noqa: SLF001
            evidence_root=self.root,
            bridge_report_path=self.bridge_report,
            candidate_id=self.candidate_id,
            freeze_sha256="a" * 64,
            base_release_candidate_report_sha256="b" * 64,
        )

    def test_locks_once_then_invokes_private_compositor(self) -> None:
        original_flock = fcntl.flock
        calls: list[int] = []

        def record_flock(descriptor: int, operation: int) -> None:
            calls.append(operation)
            original_flock(descriptor, operation)

        expected = {"status": "bound", "qualification_status": "not-run"}
        with mock.patch.object(finalizer.fcntl, "flock", side_effect=record_flock):
            with mock.patch.object(
                composer,
                "_write_and_bind_v5_held_locked_root_fd",
                return_value=expected,
            ) as composed:
                self.assertEqual(self._finalize(), expected)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual(calls[1], fcntl.LOCK_UN)
        self.assertEqual(composed.call_count, 1)
        self.assertEqual(composed.call_args.args, (mock.ANY,))
        self.assertEqual(composed.call_args.kwargs["evidence_root"], self.root)

    def test_compositor_failure_preserves_reason_without_a_second_attempt(self) -> None:
        error = composer.C02LifecycleV5RawComposeError("fixture ambiguity")
        error.reason_code = "ambiguous-terminal-publication"  # type: ignore[attr-defined]
        with mock.patch.object(
            composer,
            "_write_and_bind_v5_held_locked_root_fd",
            side_effect=error,
        ) as composed:
            with self.assertRaises(finalizer.C02LifecycleV5RawFinalizationError) as raised:
                self._finalize()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertEqual(composed.call_count, 1)

    def test_real_compositor_creates_one_verified_fixed_raw_pair(self) -> None:
        report = self._finalize()
        self.assertEqual(report["schema_version"], checker.SOAK_V5_REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertTrue((self.root / composer.BIND_REQUEST_NAME).is_file())
        for name in (
            composer.MANIFEST_NAME,
            f"{composer.MANIFEST_NAME}.intent",
            f"{composer.MANIFEST_NAME}.complete",
        ):
            self.assertTrue((self.root / name).is_file())
        self.assertEqual(
            checker.verify_completed_soak_provenance_v5(self.root, composer.MANIFEST_NAME),
            report,
        )

    def test_finalizer_holds_the_root_lock_while_calling_the_compositor(self) -> None:
        contender = (
            "import fcntl,os,sys\n"
            "flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC\n"
            "fd=os.open(sys.argv[1],flags)\n"
            "try:\n"
            "    try:\n"
            "        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "    except BlockingIOError:\n"
            "        raise SystemExit(0)\n"
            "    raise SystemExit(1)\n"
            "finally:\n"
            "    os.close(fd)\n"
        )

        def assert_exclusive_lock(root_fd: int, **_kwargs: object) -> dict:
            completed = subprocess.run(
                ["/usr/bin/python3", "-B", "-S", "-c", contender, str(self.root)],
                text=True,
                capture_output=True,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertGreaterEqual(root_fd, 0)
            return {"status": "bound", "qualification_status": "not-run"}

        with mock.patch.object(
            composer,
            "_write_and_bind_v5_held_locked_root_fd",
            side_effect=assert_exclusive_lock,
        ):
            self.assertEqual(self._finalize()["status"], "bound")

    def test_real_ambiguous_marker_failure_cannot_restart_the_chain(self) -> None:
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "v5 soak raw manifest completion marker parent directory":
                error = common.ProvenanceV2Error("fixture final marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(finalizer.C02LifecycleV5RawFinalizationError) as raised:
                self._finalize()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / composer.BIND_REQUEST_NAME).is_file())
        self.assertTrue((self.root / composer.MANIFEST_NAME).is_file())
        self.assertTrue((self.root / f"{composer.MANIFEST_NAME}.intent").is_file())
        self.assertTrue((self.root / f"{composer.MANIFEST_NAME}.complete").is_file())

        with self.assertRaises(finalizer.C02LifecycleV5RawFinalizationError) as retry:
            self._finalize()
        self.assert_reason(retry, "output-name-collision")

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
            "argparse",
            "def main",
            "add_argument",
            "import socket",
            "import subprocess",
            "def resume",
            "def retry",
            "def finalize",
            "publish_create_only_hardlink",
        ):
            self.assertNotIn(forbidden, source)
        parameters = inspect.signature(
            finalizer._finalize_authenticated_v5_raw_once  # noqa: SLF001
        ).parameters
        for forbidden_parameter in (
            "output_name",
            "manifest_name",
            "configuration_profile",
            "target",
            "descriptor",
            "continuation",
        ):
            self.assertNotIn(forbidden_parameter, parameters)


if __name__ == "__main__":
    unittest.main()
