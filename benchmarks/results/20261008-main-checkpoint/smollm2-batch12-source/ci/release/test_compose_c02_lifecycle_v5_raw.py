#!/usr/bin/env python3
"""CPU-only hostile tests for the private C02 native-fallback v5 raw compositor."""

from __future__ import annotations

import ast
import fcntl
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bind_raw_c02_soak_v5 as v5_binder
import check_c02_provenance_v2 as checker
import compose_c02_lifecycle_v5_raw as composer
import provenance_v2_common as common
import test_write_c02_lifecycle_bind_request_v5 as writer_fixtures


class ComposeC02LifecycleV5RawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.writer_fixture = writer_fixtures.WriteC02LifecycleBindRequestV5Tests(
            methodName="runTest"
        )
        self.writer_fixture.setUp()
        self.root = self.writer_fixture.root
        self.tree = self.writer_fixture.tree
        self.bridge_report = self.writer_fixture.bridge_report
        self.candidate_id = self.writer_fixture.fixture.candidate_id

    def tearDown(self) -> None:
        self.writer_fixture.tearDown()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _compose(self, *, bridge_report_path: Path | None = None) -> dict:
        root_fd = common.open_private_evidence_directory(self.root, "test root")
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._compose_on_held_fd(
                root_fd,
                self.root,
                bridge_report_path=bridge_report_path or self.bridge_report,
            )
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

    def _compose_on_held_fd(
        self,
        root_fd: int,
        evidence_root: Path,
        *,
        bridge_report_path: Path,
    ) -> dict:
        return composer._write_and_bind_v5_held_locked_root_fd(  # noqa: SLF001
            root_fd,
            evidence_root=evidence_root,
            bridge_report_path=bridge_report_path,
            candidate_id=self.candidate_id,
            freeze_sha256="a" * 64,
            base_release_candidate_report_sha256="b" * 64,
        )

    def test_held_lock_chain_creates_only_fixed_raw_outputs(self) -> None:
        root_fd = common.open_private_evidence_directory(self.root, "test root")
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(
                v5_binder.fcntl,
                "flock",
                side_effect=AssertionError("private v5 chain must not change the root lock"),
            ):
                report = composer._write_and_bind_v5_held_locked_root_fd(  # noqa: SLF001
                    root_fd,
                    evidence_root=self.root,
                    bridge_report_path=self.bridge_report,
                    candidate_id=self.candidate_id,
                    freeze_sha256="a" * 64,
                    base_release_candidate_report_sha256="b" * 64,
                )
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

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

    def test_writer_failure_never_calls_binder_or_creates_fixed_outputs(self) -> None:
        self.bridge_report.write_bytes(b"untrusted bridge stdout\n")
        with mock.patch.object(
            v5_binder,
            "_bind_raw_soak_manifest_held_locked_fd",
            side_effect=AssertionError("binder must not run after a request failure"),
        ):
            with self.assertRaises(composer.C02LifecycleV5RawComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "invalid-json")
        self.assertFalse((self.root / composer.BIND_REQUEST_NAME).exists())
        self.assertFalse((self.root / composer.MANIFEST_NAME).exists())
        self.assertFalse((self.root / f"{composer.MANIFEST_NAME}.complete").exists())

    def test_fixed_request_collision_refuses_before_terminal_publication(self) -> None:
        self.tree.put(composer.BIND_REQUEST_NAME, {"occupied": True})
        with self.assertRaises(composer.C02LifecycleV5RawComposeError) as request_raised:
            self._compose()
        self.assert_reason(request_raised, "output-name-collision")
        self.assertFalse((self.root / composer.MANIFEST_NAME).exists())

    def test_fixed_terminal_sibling_refuses_before_request_publication(self) -> None:
        self.tree.put(f"{composer.MANIFEST_NAME}.intent", {"occupied": True})
        with self.assertRaises(composer.C02LifecycleV5RawComposeError) as marker_raised:
            self._compose()
        self.assert_reason(marker_raised, "output-name-collision")
        self.assertFalse((self.root / composer.BIND_REQUEST_NAME).exists())
        self.assertFalse((self.root / composer.MANIFEST_NAME).exists())

    def test_ambiguous_terminal_pair_cannot_be_retried_as_a_new_chain(self) -> None:
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "v5 soak raw manifest completion marker parent directory":
                error = common.ProvenanceV2Error("fixture final marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(composer.C02LifecycleV5RawComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.root / composer.BIND_REQUEST_NAME).is_file())
        self.assertTrue((self.root / composer.MANIFEST_NAME).is_file())
        self.assertTrue((self.root / f"{composer.MANIFEST_NAME}.intent").is_file())
        self.assertTrue((self.root / f"{composer.MANIFEST_NAME}.complete").is_file())
        self.assertEqual(
            checker.verify_completed_soak_provenance_v5(self.root, composer.MANIFEST_NAME)["status"],
            "bound",
        )

        with self.assertRaises(composer.C02LifecycleV5RawComposeError) as retry:
            self._compose()
        self.assert_reason(retry, "output-name-collision")

    def test_binder_prepublication_failure_leaves_only_the_fixed_request(self) -> None:
        error = v5_binder.RawSoakBindError("fixture v5 binder refusal")
        error.reason_code = "fixture-binder-refusal"  # type: ignore[attr-defined]
        with mock.patch.object(
            v5_binder,
            "_bind_raw_soak_manifest_held_locked_fd",
            side_effect=error,
        ):
            with self.assertRaises(composer.C02LifecycleV5RawComposeError) as raised:
                self._compose()
        self.assert_reason(raised, "fixture-binder-refusal")
        self.assertTrue((self.root / composer.BIND_REQUEST_NAME).is_file())
        self.assertFalse((self.root / composer.MANIFEST_NAME).exists())
        self.assertFalse((self.root / f"{composer.MANIFEST_NAME}.intent").exists())
        self.assertFalse((self.root / f"{composer.MANIFEST_NAME}.complete").exists())

    def test_secure_reopen_requires_held_fd_to_match_evidence_path(self) -> None:
        other_root = self.writer_fixture.external_path / "other-private-evidence"
        other_root.mkdir(mode=0o700)
        other_root.chmod(0o700)
        root_fd = common.open_private_evidence_directory(other_root, "other test root")
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(composer.C02LifecycleV5RawComposeError) as raised:
                self._compose_on_held_fd(
                    root_fd,
                    self.root,
                    bridge_report_path=self.bridge_report,
                )
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)
        self.assert_reason(raised, "evidence-root-fd-path-mismatch")
        self.assertFalse((other_root / composer.BIND_REQUEST_NAME).exists())
        self.assertFalse((self.root / composer.BIND_REQUEST_NAME).exists())

    def test_secure_reopen_rejects_unsafe_ancestor_and_intermediate_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            unsafe_parent = Path(temporary) / "unsafe-parent"
            unsafe_parent.mkdir(mode=0o700)
            unsafe_parent.chmod(0o777)
            unsafe_root = unsafe_parent / "evidence"
            unsafe_root.mkdir(mode=0o700)
            unsafe_root.chmod(0o700)
            flags = (
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | os.O_CLOEXEC
            )
            root_fd = os.open(unsafe_root, flags)
            try:
                fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(composer.C02LifecycleV5RawComposeError) as unsafe_raised:
                    self._compose_on_held_fd(
                        root_fd,
                        unsafe_root,
                        bridge_report_path=self.bridge_report,
                    )
            finally:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
                os.close(root_fd)
            self.assert_reason(unsafe_raised, "unsafe-evidence-ancestor")
            self.assertFalse((unsafe_root / composer.BIND_REQUEST_NAME).exists())
            self.assertFalse((unsafe_root / composer.MANIFEST_NAME).exists())

            alias = Path(temporary) / "evidence-alias"
            alias.symlink_to(self.root, target_is_directory=True)
            root_fd = common.open_private_evidence_directory(self.root, "test root")
            try:
                fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(composer.C02LifecycleV5RawComposeError) as link_raised:
                    self._compose_on_held_fd(
                        root_fd,
                        alias,
                        bridge_report_path=self.bridge_report,
                    )
            finally:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
                os.close(root_fd)
            self.assert_reason(link_raised, "unsafe-evidence-directory")
            self.assertFalse((self.root / composer.BIND_REQUEST_NAME).exists())
            self.assertFalse((self.root / composer.MANIFEST_NAME).exists())

    def test_private_surface_has_no_cli_or_resume_parameters(self) -> None:
        source = Path(composer.__file__).read_text(encoding="utf-8")
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
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "def resume",
            "def retry",
            "def finalize",
            "add_argument",
            "publish_create_only_hardlink",
        ):
            self.assertNotIn(forbidden, source)
        parameters = inspect.signature(
            composer._write_and_bind_v5_held_locked_root_fd  # noqa: SLF001
        ).parameters
        for forbidden_parameter in (
            "configuration_profile",
            "output_name",
            "manifest_name",
            "target",
            "descriptor",
            "continuation",
        ):
            self.assertNotIn(forbidden_parameter, parameters)


if __name__ == "__main__":
    unittest.main()
