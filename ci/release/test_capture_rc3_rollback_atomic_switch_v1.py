#!/usr/bin/env python3
"""CPU-only tests for the isolated ``renameat2`` rollback switch producer."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import capture_rc3_rollback_atomic_switch_v1 as capture  # noqa: E402


def fake_exchange(directory_fd: int) -> None:
    """Portable test double for exchange; production always uses renameat2."""

    temporary = "test-exchange-temporary"
    os.rename(capture.ACTIVE_NAME, temporary, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.rename(
        capture.ROLLBACK_STAGED_NAME,
        capture.ACTIVE_NAME,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    os.rename(temporary, capture.ROLLBACK_STAGED_NAME, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)


class AtomicSwitchCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.root = self.base / "evidence"
        self.root.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        self.switch = self.root / "switch-workspace"
        self.switch.mkdir(mode=0o700)
        os.chmod(self.switch, 0o700)
        self._write_staged(capture.ACTIVE_NAME, b"candidate")
        self._write_staged(capture.ROLLBACK_STAGED_NAME, b"rollback")
        self.request = capture.SwitchRequest(self.root, "switch-workspace", "atomic-switch")
        self.root_patch = mock.patch.object(capture, "_source_root", return_value=self.repository)
        self.root_patch.start()

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.temporary.cleanup()

    def _write_staged(self, name: str, value: bytes) -> Path:
        path = self.switch / name
        path.write_bytes(value)
        os.chmod(path, 0o700)
        return path

    def test_closed_leaf_and_renameat2_availability_validation(self) -> None:
        for value in ("", ".", "..", "../switch", "nested/path", ".hidden"):
            with self.subTest(value=value), self.assertRaises(capture.AtomicSwitchCaptureError):
                capture._leaf(value, "fixture")
        with mock.patch.object(capture.sys, "platform", "linux"), mock.patch.object(
            capture.ctypes, "CDLL", return_value=object()
        ):
            with self.assertRaises(capture.AtomicSwitchCaptureError):
                capture._rename_exchange(7)

    def test_capture_exchanges_exact_single_link_staged_files_and_writes_five_leaves(self) -> None:
        candidate_inode = (self.switch / capture.ACTIVE_NAME).stat().st_ino
        rollback_inode = (self.switch / capture.ROLLBACK_STAGED_NAME).stat().st_ino
        with mock.patch.object(capture, "_rename_exchange", side_effect=fake_exchange):
            session = capture.capture_atomic_switch(self.request)
        capture_root = self.root / self.request.capture_name
        self.assertEqual((self.switch / capture.ACTIVE_NAME).stat().st_ino, rollback_inode)
        self.assertEqual((self.switch / capture.ROLLBACK_STAGED_NAME).stat().st_ino, candidate_inode)
        self.assertTrue((capture_root / "session.json").is_file())
        self.assertFalse((capture_root / capture.INCOMPLETE_MARKER_NAME).exists())
        self.assertEqual(session["schema_version"], capture.SESSION_VERSION)
        self.assertEqual(session["qualification_status"], "not-run")
        self.assertEqual(
            set(session["atomic_switch"]),
            {
                "pre_active_stat",
                "post_active_stat",
                "candidate_staged_stat",
                "rollback_staged_stat",
                "rename_transcript",
            },
        )
        for descriptor in session["atomic_switch"].values():
            raw = (self.root / descriptor["path"]).read_bytes()
            self.assertEqual(len(raw), descriptor["byte_length"])
            self.assertEqual(json.loads(raw)["schema_version"], capture.STAT_VERSION if "stat" in descriptor["path"] else capture.SWITCH_VERSION)
        transcript = json.loads((capture_root / "rename-transcript.json").read_bytes())
        self.assertEqual(transcript["operation"], "renameat2-rename-exchange")
        for path in (self.root, self.switch, capture_root):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)

    @unittest.skipUnless(sys.platform == "linux", "renameat2 integration requires Linux")
    def test_linux_renameat2_exchange_maps_the_two_inodes_without_a_fallback(self) -> None:
        candidate_inode = (self.switch / capture.ACTIVE_NAME).stat().st_ino
        rollback_inode = (self.switch / capture.ROLLBACK_STAGED_NAME).stat().st_ino
        session = capture.capture_atomic_switch(self.request)
        self.assertEqual((self.switch / capture.ACTIVE_NAME).stat().st_ino, rollback_inode)
        self.assertEqual((self.switch / capture.ROLLBACK_STAGED_NAME).stat().st_ino, candidate_inode)
        transcript = json.loads((self.root / self.request.capture_name / "rename-transcript.json").read_bytes())
        self.assertEqual(transcript["post_active"]["inode"], rollback_inode)
        self.assertEqual(transcript["post_candidate_staged"]["inode"], candidate_inode)
        self.assertEqual(session["atomic_switch"]["rename_transcript"]["path"], "atomic-switch/rename-transcript.json")

    def test_rejects_hardlink_mode_drift_or_cross_device_before_exchange(self) -> None:
        active = self.switch / capture.ACTIVE_NAME
        os.chmod(active, 0o600)
        with self.assertRaisesRegex(capture.AtomicSwitchCaptureError, "mode 0700"):
            capture.capture_atomic_switch(self.request)
        os.chmod(active, 0o700)
        os.link(active, self.switch / "active-alias")
        hardlink_request = capture.SwitchRequest(self.root, "switch-workspace", "atomic-switch-hardlink")
        with self.assertRaisesRegex(capture.AtomicSwitchCaptureError, "single-link"):
            capture.capture_atomic_switch(hardlink_request)

    def test_rename_failure_retains_marker_and_does_not_publish_session(self) -> None:
        with mock.patch.object(
            capture,
            "_rename_exchange",
            side_effect=capture.AtomicSwitchCaptureError("fixture exchange failure"),
        ):
            with self.assertRaisesRegex(capture.AtomicSwitchCaptureError, "fixture exchange failure"):
                capture.capture_atomic_switch(self.request)
        capture_root = self.root / self.request.capture_name
        self.assertTrue((capture_root / capture.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((capture_root / "session.json").exists())

    def test_existing_capture_root_and_switch_symlink_are_refused(self) -> None:
        directory = self.root / self.request.capture_name
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        with self.assertRaisesRegex(capture.AtomicSwitchCaptureError, "already exists"):
            capture.capture_atomic_switch(self.request)
        directory.rmdir()
        source = self.root / "switch-real"
        self.switch.rename(source)
        os.symlink(source, self.switch)
        with self.assertRaises(capture.AtomicSwitchCaptureError):
            capture.capture_atomic_switch(self.request)

    def test_new_capture_child_rejects_same_uid_path_replacement_before_open(self) -> None:
        root_fd = capture.common.open_private_evidence_directory(self.root, "fixture root")
        original_open = capture._open_private_child
        replacement = self.root / "replacement-child"
        replacement.mkdir(mode=0o700)
        os.chmod(replacement, 0o700)

        def replace_then_open(parent_fd: int, name: str, label: str) -> int:
            os.rmdir(name, dir_fd=parent_fd)
            os.rename("replacement-child", name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            return original_open(parent_fd, name, label)

        try:
            with mock.patch.object(capture, "_open_private_child", side_effect=replace_then_open):
                with self.assertRaisesRegex(capture.AtomicSwitchCaptureError, "not created"):
                    capture._new_private_child(root_fd, "raced-capture", "raced capture directory")
        finally:
            capture._close_quietly(root_fd)

    def test_marker_sync_failure_restores_nonterminal_marker(self) -> None:
        original = capture._fsync

        def fail_completion_sync(descriptor: int, label: str) -> None:
            if label == "capture directory after incomplete marker removal":
                raise capture.AtomicSwitchCaptureError("fixture marker sync failure")
            original(descriptor, label)

        with mock.patch.object(capture, "_rename_exchange", side_effect=fake_exchange), mock.patch.object(
            capture, "_fsync", side_effect=fail_completion_sync
        ):
            with self.assertRaisesRegex(capture.AtomicSwitchCaptureError, "marker sync failure"):
                capture.capture_atomic_switch(self.request)
        capture_root = self.root / self.request.capture_name
        self.assertTrue((capture_root / "session.json").is_file())
        self.assertTrue((capture_root / capture.INCOMPLETE_MARKER_NAME).is_file())

    def test_cli_help_and_source_require_linux_renameat2_without_fallback(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", "-S", str(Path(capture.__file__)), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--switch-dir-name", completed.stdout)
        wrapper = Path(capture.__file__).with_name("run_capture_rc3_rollback_atomic_switch_v1.sh")
        wrapped = subprocess.run(
            ["bash", str(wrapper), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(wrapped.returncode, 0, wrapped.stderr)
        self.assertIn("--capture-name", wrapped.stdout)
        wrapper_source = wrapper.read_text(encoding="utf-8")
        for required in ("/usr/bin/env -i", "-B -I -S", "os.lstat(script)", "sys.path.insert"):
            self.assertIn(required, wrapper_source)
        source = Path(capture.__file__).read_text(encoding="utf-8")
        for required in ("renameat2(RENAME_EXCHANGE)", "RENAME_EXCHANGE", "ctypes.CDLL", "os.fsync"):
            self.assertIn(required, source)
        for forbidden in ("subprocess.run", "os.rename(", "docker ", "ssh ", "systemctl ", "qualification_status\": \"passed"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
