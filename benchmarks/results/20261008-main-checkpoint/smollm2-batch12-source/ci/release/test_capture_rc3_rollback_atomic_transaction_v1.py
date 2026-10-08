#!/usr/bin/env python3
"""CPU-only tests for the fixed held-FD rollback exchange transaction."""

from __future__ import annotations

import fcntl
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

import capture_rc3_rollback_atomic_switch_v1 as atomic  # noqa: E402
import capture_rc3_rollback_atomic_transaction_v1 as transaction  # noqa: E402
import prepare_rc3_rollback_artifacts_v1 as prepare  # noqa: E402
import provenance_v2_common as common  # noqa: E402


def fake_exchange(directory_fd: int) -> None:
    """Portable test double; production always invokes Linux renameat2."""

    temporary = "test-exchange-temporary"
    os.rename(atomic.ACTIVE_NAME, temporary, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.rename(
        atomic.ROLLBACK_STAGED_NAME,
        atomic.ACTIVE_NAME,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    os.rename(temporary, atomic.ROLLBACK_STAGED_NAME, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)


class AtomicTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.root = self.base / "evidence"
        self.root.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        self.inputs = self.base / "inputs"
        self.inputs.mkdir(mode=0o700)
        os.chmod(self.inputs, 0o700)
        self.candidate_binary = self._write("candidate-bin", b"candidate executable\n", 0o700)
        self.candidate_bundle = self._write("candidate-bundle", b"candidate bundle\n", 0o600)
        self.candidate_image = self._write("candidate-image.json", b"[{\"Id\":\"candidate\"}]", 0o600)
        self.rollback_binary = self._write("rollback-bin", b"rollback executable\n", 0o700)
        self.rollback_bundle = self._write("rollback-bundle", b"rollback bundle\n", 0o600)
        self.rollback_image = self._write("rollback-image.json", b"[{\"Id\":\"rollback\"}]", 0o600)
        self.request = prepare.PreparationRequest(
            evidence_root=self.root,
            candidate_binary=self.candidate_binary,
            candidate_bundle=self.candidate_bundle,
            candidate_image_inspect=self.candidate_image,
            rollback_binary=self.rollback_binary,
            rollback_bundle=self.rollback_bundle,
            rollback_image_inspect=self.rollback_image,
        )
        self.patches = [
            mock.patch.object(prepare, "_source_root", return_value=self.repository),
            mock.patch.object(atomic, "_source_root", return_value=self.repository),
            mock.patch.object(transaction, "_source_root", return_value=self.repository),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def _write(self, name: str, raw: bytes, mode: int) -> Path:
        path = self.inputs / name
        path.write_bytes(raw)
        os.chmod(path, mode)
        return path

    def _runtime(self, name: str) -> Path:
        return self.root / prepare.SWITCH_DIRECTORY_NAME / name

    def _prepare(self) -> dict[str, object]:
        return prepare.prepare_artifacts(self.request)

    def test_same_held_switch_fd_replays_pre_exchange_and_post_without_relocking_it(self) -> None:
        prepared = self._prepare()
        root_fd = common.open_private_evidence_directory(self.root, "fixture root")
        switch_fd = common.open_private_child_directory(
            root_fd,
            prepare.SWITCH_DIRECTORY_NAME,
            "fixture switch directory",
        )
        fcntl.flock(switch_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        real_flock = fcntl.flock

        def reject_switch_relock(descriptor: int, operation: int) -> None:
            if descriptor == switch_fd:
                raise AssertionError("held switch FD was reopened or relocked")
            real_flock(descriptor, operation)

        try:
            with mock.patch.object(atomic, "_rename_exchange", side_effect=fake_exchange), mock.patch.object(
                fcntl,
                "flock",
                side_effect=reject_switch_relock,
            ):
                pre_replay = prepare.replay_artifact_preparation_on_held_switch_fd(
                    root_fd,
                    switch_fd,
                    runtime_layout="pre-switch",
                )
                self.assertEqual(dict(pre_replay.session), prepared)
                atomic.capture_atomic_switch_on_held_switch_fd(
                    root_fd,
                    switch_fd,
                    prepare.SWITCH_DIRECTORY_NAME,
                    transaction.ATOMIC_CAPTURE_DIRECTORY_NAME,
                )
                replay = atomic.replay_atomic_switch_capture_on_held_switch_fd(
                    root_fd,
                    switch_fd,
                    prepare.SWITCH_DIRECTORY_NAME,
                    transaction.ATOMIC_CAPTURE_DIRECTORY_NAME,
                )
                post_replay = prepare.replay_artifact_preparation_on_held_switch_fd(
                    root_fd,
                    switch_fd,
                    runtime_layout="post-switch",
                )
                self.assertEqual(dict(post_replay.session), prepared)
            transaction._cross_bind_preparation_and_exchange(
                prepared,  # type: ignore[arg-type]
                (pre_replay.candidate_runtime, pre_replay.rollback_runtime),
                replay,
                (post_replay.candidate_runtime, post_replay.rollback_runtime),
            )
        finally:
            fcntl.flock(switch_fd, fcntl.LOCK_UN)
            os.close(switch_fd)
            os.close(root_fd)

    def test_transaction_captures_fixed_terminal_join_and_remains_not_run(self) -> None:
        prepared = self._prepare()
        candidate_inode = self._runtime(prepare.ACTIVE_NAME).stat().st_ino
        rollback_inode = self._runtime(prepare.ROLLBACK_STAGED_NAME).stat().st_ino
        with mock.patch.object(atomic, "_rename_exchange", side_effect=fake_exchange):
            session = transaction.capture_atomic_transaction(self.root)
        self.assertEqual(session["schema_version"], transaction.SESSION_VERSION)
        self.assertEqual(session["capture_status"], "captured")
        self.assertEqual(session["qualification_status"], "not-run")
        self.assertEqual(transaction.verify_atomic_transaction(self.root), session)
        self.assertEqual(self._runtime(prepare.ACTIVE_NAME).stat().st_ino, rollback_inode)
        self.assertEqual(self._runtime(prepare.ROLLBACK_STAGED_NAME).stat().st_ino, candidate_inode)
        transaction_root = self.root / transaction.TRANSACTION_DIRECTORY_NAME
        self.assertFalse((transaction_root / transaction.INCOMPLETE_MARKER_NAME).exists())
        completion = transaction_root / transaction.COMPLETE_MARKER_NAME
        intent = transaction_root / transaction.COMPLETE_INTENT_NAME
        self.assertTrue(completion.is_file())
        self.assertTrue(intent.is_file())
        self.assertEqual(
            (completion.stat().st_dev, completion.stat().st_ino),
            (intent.stat().st_dev, intent.stat().st_ino),
        )
        for marker in (completion, intent):
            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            self.assertEqual(marker.stat().st_nlink, 2)
        self.assertEqual(stat.S_IMODE(transaction_root.stat().st_mode), 0o700)
        document = json.loads((transaction_root / "session.json").read_bytes())
        self.assertEqual(document, session)
        self.assertEqual(
            document["preparation_session"]["path"],
            f"{prepare.SNAPSHOT_DIRECTORY_NAME}/session.json",
        )
        self.assertEqual(
            document["atomic_switch_session"]["path"],
            f"{transaction.ATOMIC_CAPTURE_DIRECTORY_NAME}/session.json",
        )
        self.assertEqual(prepared["qualification_status"], "not-run")

    def test_transaction_failure_keeps_its_nonterminal_marker_and_no_session(self) -> None:
        self._prepare()
        with mock.patch.object(
            atomic,
            "_rename_exchange",
            side_effect=atomic.AtomicSwitchCaptureError("fixture exchange failure"),
        ):
            with self.assertRaisesRegex(transaction.RollbackAtomicTransactionError, "fixture exchange failure"):
                transaction.capture_atomic_transaction(self.root)
        transaction_root = self.root / transaction.TRANSACTION_DIRECTORY_NAME
        self.assertTrue((transaction_root / transaction.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((transaction_root / "session.json").exists())

    def test_in_place_runtime_mutation_between_atomic_hash_and_exchange_is_rejected(self) -> None:
        self._prepare()

        def mutate_then_exchange(directory_fd: int) -> None:
            before = os.lstat(atomic.ACTIVE_NAME, dir_fd=directory_fd)
            descriptor = os.open(atomic.ACTIVE_NAME, os.O_RDWR, dir_fd=directory_fd)
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                os.write(descriptor, b"candidate altered!!!\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            after = os.lstat(atomic.ACTIVE_NAME, dir_fd=directory_fd)
            self.assertEqual(
                (after.st_dev, after.st_ino, after.st_size),
                (before.st_dev, before.st_ino, before.st_size),
            )
            self.assertNotEqual(self._runtime(atomic.ACTIVE_NAME).read_bytes(), b"candidate executable\n")
            fake_exchange(directory_fd)

        self.assertEqual(len(b"candidate altered!!!\n"), len(b"candidate executable\n"))
        with mock.patch.object(atomic, "_rename_exchange", side_effect=mutate_then_exchange):
            with self.assertRaisesRegex(transaction.RollbackAtomicTransactionError, "did not place candidate active inode"):
                transaction.capture_atomic_transaction(self.root)
        transaction_root = self.root / transaction.TRANSACTION_DIRECTORY_NAME
        self.assertTrue((transaction_root / transaction.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((transaction_root / "session.json").exists())

    def test_terminal_verifier_rejects_missing_completion_receipt(self) -> None:
        self._prepare()
        with mock.patch.object(atomic, "_rename_exchange", side_effect=fake_exchange):
            transaction.capture_atomic_transaction(self.root)
        transaction_root = self.root / transaction.TRANSACTION_DIRECTORY_NAME
        (transaction_root / transaction.COMPLETE_MARKER_NAME).unlink()
        self.assertTrue((transaction_root / transaction.COMPLETE_INTENT_NAME).is_file())
        with self.assertRaisesRegex(transaction.RollbackAtomicTransactionError, "paired marker"):
            transaction.verify_atomic_transaction(self.root)

    def test_marker_sync_and_restore_failure_never_becomes_terminal(self) -> None:
        self._prepare()
        original = transaction._publish_completion_marker

        def fail_completion_sync(
            transaction_fd: int,
            marker: object,
            session: common.CreatedEvidence,
        ) -> None:
            with mock.patch.object(transaction.os, "fsync", side_effect=OSError("fixture marker sync failure")), mock.patch.object(
                transaction,
                "_restore_marker",
                return_value=None,
            ):
                original(transaction_fd, marker, session)  # type: ignore[arg-type]

        with mock.patch.object(atomic, "_rename_exchange", side_effect=fake_exchange), mock.patch.object(
            transaction,
            "_publish_completion_marker",
            side_effect=fail_completion_sync,
        ):
            with self.assertRaisesRegex(transaction.RollbackAtomicTransactionError, "marker removal"):
                transaction.capture_atomic_transaction(self.root)
        transaction_root = self.root / transaction.TRANSACTION_DIRECTORY_NAME
        self.assertTrue((transaction_root / "session.json").is_file())
        self.assertFalse((transaction_root / transaction.INCOMPLETE_MARKER_NAME).exists())
        self.assertFalse((transaction_root / transaction.COMPLETE_INTENT_NAME).exists())
        self.assertFalse((transaction_root / transaction.COMPLETE_MARKER_NAME).exists())
        with self.assertRaisesRegex(transaction.RollbackAtomicTransactionError, "paired marker"):
            transaction.verify_atomic_transaction(self.root)

    def test_completion_pair_sync_failure_is_explicitly_ambiguous(self) -> None:
        self._prepare()
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "atomic transaction completion marker parent directory":
                common._fail("durability-failure", "fixture final marker directory sync failure")
            original(descriptor, label)

        with mock.patch.object(atomic, "_rename_exchange", side_effect=fake_exchange), mock.patch.object(
            common,
            "_fsync_checked",
            side_effect=fail_final_parent,
        ):
            with self.assertRaises(transaction.RollbackAtomicTransactionError) as raised:
                transaction.capture_atomic_transaction(self.root)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "ambiguous-terminal-publication")
        self.assertIn("ambiguous-terminal-publication", str(raised.exception))
        transaction_root = self.root / transaction.TRANSACTION_DIRECTORY_NAME
        completion = transaction_root / transaction.COMPLETE_MARKER_NAME
        intent = transaction_root / transaction.COMPLETE_INTENT_NAME
        self.assertEqual((completion.stat().st_dev, completion.stat().st_ino), (intent.stat().st_dev, intent.stat().st_ino))
        self.assertEqual(completion.stat().st_nlink, 2)
        self.assertEqual(transaction.verify_atomic_transaction(self.root)["capture_status"], "captured")

    def test_terminal_replay_rejects_transaction_session_mode_marker_and_cross_bind_drift(self) -> None:
        prepared = self._prepare()
        with mock.patch.object(atomic, "_rename_exchange", side_effect=fake_exchange):
            transaction_session = transaction.capture_atomic_transaction(self.root)
        transaction_root = self.root / transaction.TRANSACTION_DIRECTORY_NAME
        os.chmod(transaction_root / "session.json", 0o644)
        with self.assertRaisesRegex(transaction.RollbackAtomicTransactionError, "mode 0600"):
            transaction.verify_atomic_transaction(self.root)
        os.chmod(transaction_root / "session.json", 0o600)
        (transaction_root / transaction.INCOMPLETE_MARKER_NAME).write_bytes(b"incomplete")
        os.chmod(transaction_root / transaction.INCOMPLETE_MARKER_NAME, 0o600)
        with self.assertRaisesRegex(transaction.RollbackAtomicTransactionError, "incomplete marker"):
            transaction.verify_atomic_transaction(self.root)
        (transaction_root / transaction.INCOMPLETE_MARKER_NAME).unlink()

        root_fd = common.open_private_evidence_directory(self.root, "fixture root")
        switch_fd = common.open_private_child_directory(
            root_fd,
            prepare.SWITCH_DIRECTORY_NAME,
            "fixture switch directory",
        )
        try:
            replay = atomic.replay_atomic_switch_capture_on_held_switch_fd(
                root_fd,
                switch_fd,
                prepare.SWITCH_DIRECTORY_NAME,
                transaction.ATOMIC_CAPTURE_DIRECTORY_NAME,
            )
            forged = json.loads(json.dumps(prepared))
            forged["runtime_materializations"]["candidate"]["inode"] += 1
            pre_layout = transaction._preparation_layout(
                transaction_session["preparation_pre_switch"],
                "fixture pre-switch layout",
            )
            post_replay = prepare.replay_artifact_preparation_on_held_switch_fd(
                root_fd,
                switch_fd,
                runtime_layout="post-switch",
            )
            with self.assertRaisesRegex(transaction.RollbackAtomicTransactionError, "pre-switch candidate runtime"):
                transaction._cross_bind_preparation_and_exchange(
                    forged,
                    pre_layout,
                    replay,
                    (post_replay.candidate_runtime, post_replay.rollback_runtime),
                )
        finally:
            os.close(switch_fd)
            os.close(root_fd)

    @unittest.skipUnless(sys.platform == "linux", "renameat2 transaction integration requires Linux")
    def test_linux_transaction_uses_the_real_exchange_without_a_fallback(self) -> None:
        self._prepare()
        candidate_inode = self._runtime(prepare.ACTIVE_NAME).stat().st_ino
        rollback_inode = self._runtime(prepare.ROLLBACK_STAGED_NAME).stat().st_ino
        session = transaction.capture_atomic_transaction(self.root)
        self.assertEqual(session["qualification_status"], "not-run")
        self.assertEqual(self._runtime(prepare.ACTIVE_NAME).stat().st_ino, rollback_inode)
        self.assertEqual(self._runtime(prepare.ROLLBACK_STAGED_NAME).stat().st_ino, candidate_inode)

    def test_cli_wrapper_and_source_expose_only_the_fixed_evidence_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", "-S", str(Path(transaction.__file__)), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--evidence-root", completed.stdout)
        self.assertNotIn("--switch", completed.stdout)
        wrapper = Path(transaction.__file__).with_name("run_capture_rc3_rollback_atomic_transaction_v1.sh")
        wrapped = subprocess.run(
            ["bash", str(wrapper), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(wrapped.returncode, 0, wrapped.stderr)
        self.assertIn("--evidence-root", wrapped.stdout)
        wrapper_source = wrapper.read_text(encoding="utf-8")
        for required in ("/usr/bin/env -i", "-B -I -S", "os.lstat(script)", "sys.path.insert"):
            self.assertIn(required, wrapper_source)
        source = Path(transaction.__file__).read_text(encoding="utf-8")
        for required in (
            "runtime_layout=\"pre-switch\"",
            "capture_atomic_switch_on_held_switch_fd",
            "runtime_layout=\"post-switch\"",
            "qualification_status\": \"not-run",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "subprocess.run",
            "socket.",
            "docker ",
            "ssh ",
            "systemctl ",
            "qualification_status\": \"passed",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
