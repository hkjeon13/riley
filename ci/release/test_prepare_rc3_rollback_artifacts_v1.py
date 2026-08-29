#!/usr/bin/env python3
"""CPU-only hostile-path tests for rollback artifact preparation."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import capture_rc3_rollback_atomic_switch_v1 as atomic_switch  # noqa: E402
import prepare_rc3_rollback_artifacts_v1 as prepare  # noqa: E402
import provenance_v2_common as common  # noqa: E402


def fake_exchange(directory_fd: int) -> None:
    """Portable test double; production atomic helper remains renameat2-only."""

    temporary = "test-exchange-temporary"
    os.rename(atomic_switch.ACTIVE_NAME, temporary, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.rename(
        atomic_switch.ROLLBACK_STAGED_NAME,
        atomic_switch.ACTIVE_NAME,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    os.rename(temporary, atomic_switch.ROLLBACK_STAGED_NAME, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)


class ArtifactPreparationTests(unittest.TestCase):
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
        self.prepare_root_patch = mock.patch.object(prepare, "_source_root", return_value=self.repository)
        self.atomic_root_patch = mock.patch.object(atomic_switch, "_source_root", return_value=self.repository)
        self.prepare_root_patch.start()
        self.atomic_root_patch.start()

    def tearDown(self) -> None:
        self.atomic_root_patch.stop()
        self.prepare_root_patch.stop()
        self.temporary.cleanup()

    def _write(self, name: str, raw: bytes, mode: int) -> Path:
        path = self.inputs / name
        path.write_bytes(raw)
        os.chmod(path, mode)
        return path

    def _snapshot(self, name: str) -> Path:
        return self.root / prepare.ARTIFACT_DIRECTORY_NAME / name

    def _runtime(self, name: str) -> Path:
        return self.root / prepare.SWITCH_DIRECTORY_NAME / name

    def test_prepare_creates_immutable_snapshots_and_distinct_switchable_runtime_files(self) -> None:
        session = prepare.prepare_artifacts(self.request)
        self.assertEqual(session["schema_version"], prepare.SESSION_VERSION)
        self.assertEqual(session["capture_status"], "captured")
        self.assertEqual(session["qualification_status"], "not-run")
        self.assertEqual(prepare.verify_artifact_preparation(self.root), session)
        snapshot_dir = self.root / prepare.SNAPSHOT_DIRECTORY_NAME
        self.assertTrue((snapshot_dir / "session.json").is_file())
        self.assertFalse((snapshot_dir / prepare.INCOMPLETE_MARKER_NAME).exists())
        completion = snapshot_dir / prepare.COMPLETE_MARKER_NAME
        intent = snapshot_dir / prepare.COMPLETE_INTENT_NAME
        self.assertTrue(completion.is_file())
        self.assertTrue(intent.is_file())
        self.assertEqual(
            (completion.stat().st_dev, completion.stat().st_ino),
            (intent.stat().st_dev, intent.stat().st_ino),
        )
        for marker in (completion, intent):
            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            self.assertEqual(marker.stat().st_nlink, 2)
        for directory in (
            self.root,
            snapshot_dir,
            self.root / prepare.ARTIFACT_DIRECTORY_NAME,
            self.root / prepare.SWITCH_DIRECTORY_NAME,
        ):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        snapshots = {
            "candidate-binary": self.candidate_binary.read_bytes(),
            "candidate-bundle": self.candidate_bundle.read_bytes(),
            "candidate-image-inspect.json": self.candidate_image.read_bytes(),
            "rollback-binary": self.rollback_binary.read_bytes(),
            "rollback-bundle": self.rollback_bundle.read_bytes(),
            "rollback-image-inspect.json": self.rollback_image.read_bytes(),
        }
        for name, expected in snapshots.items():
            path = self._snapshot(name)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(path.stat().st_nlink, 1)
        active = self._runtime(prepare.ACTIVE_NAME)
        staged = self._runtime(prepare.ROLLBACK_STAGED_NAME)
        self.assertEqual(active.read_bytes(), self.candidate_binary.read_bytes())
        self.assertEqual(staged.read_bytes(), self.rollback_binary.read_bytes())
        self.assertEqual(stat.S_IMODE(active.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o700)
        self.assertEqual(active.stat().st_nlink, 1)
        self.assertEqual(staged.stat().st_nlink, 1)
        self.assertNotEqual(active.stat().st_ino, self._snapshot("candidate-binary").stat().st_ino)
        self.assertNotEqual(staged.stat().st_ino, self._snapshot("rollback-binary").stat().st_ino)
        self.assertNotEqual(active.stat().st_ino, staged.stat().st_ino)

    def test_runtime_copies_only_the_immutable_snapshot_not_the_external_source_reopened_later(self) -> None:
        original = common.materialize_descriptor_runtime_copy

        def mutate_external_then_materialize(*args: object, **kwargs: object) -> common.RuntimeMaterialization:
            if len(args) >= 4 and args[3] == prepare.ACTIVE_NAME:
                self.candidate_binary.write_bytes(b"candidate source changed after snapshot\n")
                os.chmod(self.candidate_binary, 0o700)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with mock.patch.object(common, "materialize_descriptor_runtime_copy", side_effect=mutate_external_then_materialize):
            prepare.prepare_artifacts(self.request)
        self.assertEqual(self._snapshot("candidate-binary").read_bytes(), b"candidate executable\n")
        self.assertEqual(self._runtime(prepare.ACTIVE_NAME).read_bytes(), b"candidate executable\n")
        self.assertEqual(self.candidate_binary.read_bytes(), b"candidate source changed after snapshot\n")

    def test_snapshot_inode_swap_before_runtime_materialization_is_rejected(self) -> None:
        original = common.materialize_descriptor_runtime_copy
        swapped = False

        def replace_snapshot_then_materialize(*args: object, **kwargs: object) -> common.RuntimeMaterialization:
            nonlocal swapped
            if len(args) >= 4 and args[3] == prepare.ACTIVE_NAME and not swapped:
                swapped = True
                snapshot = self._snapshot("candidate-binary")
                replacement = snapshot.with_name("candidate-binary-replacement")
                replacement.write_bytes(snapshot.read_bytes())
                os.chmod(replacement, 0o600)
                os.rename(replacement, snapshot)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with mock.patch.object(common, "materialize_descriptor_runtime_copy", side_effect=replace_snapshot_then_materialize):
            with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "expected inode"):
                prepare.prepare_artifacts(self.request)
        capture = self.root / prepare.SNAPSHOT_DIRECTORY_NAME
        self.assertTrue((capture / prepare.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((capture / "session.json").exists())

    def test_atomic_switch_helper_accepts_the_prepared_fixed_switch_directory(self) -> None:
        session = prepare.prepare_artifacts(self.request)
        candidate_inode = self._runtime(prepare.ACTIVE_NAME).stat().st_ino
        rollback_inode = self._runtime(prepare.ROLLBACK_STAGED_NAME).stat().st_ino
        request = atomic_switch.SwitchRequest(
            self.root,
            prepare.SWITCH_DIRECTORY_NAME,
            "atomic-switch-capture",
        )
        with mock.patch.object(atomic_switch, "_rename_exchange", side_effect=fake_exchange):
            capture = atomic_switch.capture_atomic_switch(request)
        self.assertEqual(self._runtime(prepare.ACTIVE_NAME).stat().st_ino, rollback_inode)
        self.assertEqual(self._runtime(prepare.ROLLBACK_STAGED_NAME).stat().st_ino, candidate_inode)
        self.assertEqual(capture["qualification_status"], "not-run")
        with self.assertRaises(prepare.RollbackArtifactPreparationError):
            prepare.verify_artifact_preparation(self.root)
        self.assertEqual(
            prepare.verify_artifact_preparation(self.root, runtime_layout="post-switch"),
            session,
        )

    def test_terminal_replay_rejects_snapshot_mode_drift_and_runtime_mode_race(self) -> None:
        session = prepare.prepare_artifacts(self.request)
        snapshot = self._snapshot("candidate-binary")
        os.chmod(snapshot, 0o644)
        with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "mode 0600"):
            prepare.verify_artifact_preparation(self.root)
        os.chmod(snapshot, 0o600)
        self.assertEqual(prepare.verify_artifact_preparation(self.root), session)
        original = common.verify_private_runtime_descriptor_file
        changed = False

        def verify_then_change(*args: object, **kwargs: object) -> None:
            nonlocal changed
            original(*args, **kwargs)  # type: ignore[arg-type]
            if not changed:
                changed = True
                os.chmod(self._runtime(prepare.ACTIVE_NAME), 0o600)

        with mock.patch.object(common, "verify_private_runtime_descriptor_file", side_effect=verify_then_change):
            with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "identity changed"):
                prepare.verify_artifact_preparation(self.root)

    def test_terminal_replay_rechecks_marker_after_descriptor_replay(self) -> None:
        prepare.prepare_artifacts(self.request)
        original = common.verify_private_runtime_descriptor_file
        published = False

        def verify_then_publish_marker(*args: object, **kwargs: object) -> None:
            nonlocal published
            original(*args, **kwargs)  # type: ignore[arg-type]
            if not published:
                published = True
                root_fd = common.open_private_evidence_directory(self.root, "fixture root")
                snapshot_fd = common.open_private_child_directory(
                    root_fd,
                    prepare.SNAPSHOT_DIRECTORY_NAME,
                    "fixture capture directory",
                )
                try:
                    common.write_create_only_json(
                        snapshot_fd,
                        prepare.INCOMPLETE_MARKER_NAME,
                        {
                            "schema_version": prepare.INCOMPLETE_MARKER_VERSION,
                            "capture_status": "incomplete",
                            "qualification_status": "not-run",
                        },
                        "injected incomplete marker",
                    )
                finally:
                    os.close(snapshot_fd)
                    os.close(root_fd)

        with mock.patch.object(common, "verify_private_runtime_descriptor_file", side_effect=verify_then_publish_marker):
            with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "incomplete marker"):
                prepare.verify_artifact_preparation(self.root)

    def test_terminal_replay_rejects_missing_completion_receipt(self) -> None:
        prepare.prepare_artifacts(self.request)
        snapshot_dir = self.root / prepare.SNAPSHOT_DIRECTORY_NAME
        (snapshot_dir / prepare.COMPLETE_MARKER_NAME).unlink()
        self.assertTrue((snapshot_dir / prepare.COMPLETE_INTENT_NAME).is_file())
        with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "paired marker"):
            prepare.verify_artifact_preparation(self.root)

    def test_marker_sync_and_restore_failure_never_becomes_terminal(self) -> None:
        original = prepare._publish_completion_marker

        def fail_completion_sync(
            snapshot_fd: int,
            marker: object,
            session: common.CreatedEvidence,
        ) -> None:
            with mock.patch.object(prepare.os, "fsync", side_effect=OSError("fixture marker sync failure")), mock.patch.object(
                prepare,
                "_restore_marker",
                return_value=None,
            ):
                original(snapshot_fd, marker, session)  # type: ignore[arg-type]

        with mock.patch.object(prepare, "_publish_completion_marker", side_effect=fail_completion_sync):
            with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "marker removal"):
                prepare.prepare_artifacts(self.request)
        snapshot_dir = self.root / prepare.SNAPSHOT_DIRECTORY_NAME
        self.assertTrue((snapshot_dir / "session.json").is_file())
        self.assertFalse((snapshot_dir / prepare.INCOMPLETE_MARKER_NAME).exists())
        self.assertFalse((snapshot_dir / prepare.COMPLETE_INTENT_NAME).exists())
        self.assertFalse((snapshot_dir / prepare.COMPLETE_MARKER_NAME).exists())
        with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "paired marker"):
            prepare.verify_artifact_preparation(self.root)

    def test_completion_pair_sync_failure_is_explicitly_ambiguous(self) -> None:
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "artifact preparation completion marker parent directory":
                common._fail("durability-failure", "fixture final marker directory sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(prepare.RollbackArtifactPreparationError) as raised:
                prepare.prepare_artifacts(self.request)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "ambiguous-terminal-publication")
        self.assertIn("ambiguous-terminal-publication", str(raised.exception))
        snapshot_dir = self.root / prepare.SNAPSHOT_DIRECTORY_NAME
        completion = snapshot_dir / prepare.COMPLETE_MARKER_NAME
        intent = snapshot_dir / prepare.COMPLETE_INTENT_NAME
        self.assertEqual((completion.stat().st_dev, completion.stat().st_ino), (intent.stat().st_dev, intent.stat().st_ino))
        self.assertEqual(completion.stat().st_nlink, 2)
        self.assertEqual(prepare.verify_artifact_preparation(self.root)["capture_status"], "captured")

    def test_terminal_replay_rejects_private_session_mode_and_held_child_replacement(self) -> None:
        prepare.prepare_artifacts(self.request)
        session_path = self.root / prepare.SNAPSHOT_DIRECTORY_NAME / "session.json"
        os.chmod(session_path, 0o644)
        with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "mode 0600"):
            prepare.verify_artifact_preparation(self.root)
        os.chmod(session_path, 0o600)

        original = common.verify_private_snapshot_descriptor_file
        swapped = False

        def verify_then_replace_visible_artifacts(*args: object, **kwargs: object) -> None:
            nonlocal swapped
            original(*args, **kwargs)  # type: ignore[arg-type]
            if not swapped:
                swapped = True
                artifacts = self.root / prepare.ARTIFACT_DIRECTORY_NAME
                moved = self.root / "old-artifacts"
                os.rename(artifacts, moved)
                artifacts.mkdir(mode=0o700)
                os.chmod(artifacts, 0o700)

        with mock.patch.object(
            common,
            "verify_private_snapshot_descriptor_file",
            side_effect=verify_then_replace_visible_artifacts,
        ):
            with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "held immutable artifact"):
                prepare.verify_artifact_preparation(self.root)

    def test_held_switch_replacement_is_rejected_after_runtime_replay(self) -> None:
        prepare.prepare_artifacts(self.request)
        original = common.verify_private_runtime_descriptor_file
        swapped = False

        def verify_then_replace_visible_switch(*args: object, **kwargs: object) -> None:
            nonlocal swapped
            original(*args, **kwargs)  # type: ignore[arg-type]
            if not swapped:
                swapped = True
                switch = self.root / prepare.SWITCH_DIRECTORY_NAME
                moved = self.root / "old-switch"
                os.rename(switch, moved)
                switch.mkdir(mode=0o700)
                os.chmod(switch, 0o700)

        with mock.patch.object(
            common,
            "verify_private_runtime_descriptor_file",
            side_effect=verify_then_replace_visible_switch,
        ):
            with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "held rollback switch"):
                prepare.verify_artifact_preparation(self.root)

    def test_nonexecutable_binary_fails_closed_with_incomplete_marker(self) -> None:
        os.chmod(self.candidate_binary, 0o600)
        with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "owner must be able to execute"):
            prepare.prepare_artifacts(self.request)
        capture = self.root / prepare.SNAPSHOT_DIRECTORY_NAME
        self.assertTrue((capture / prepare.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((capture / "session.json").exists())
        self.assertFalse((self.root / prepare.SWITCH_DIRECTORY_NAME / prepare.ACTIVE_NAME).exists())

    def test_source_mutation_during_stream_copy_leaves_marker_and_never_publishes_session(self) -> None:
        original = common._write_all
        mutated = False

        def mutate_source(descriptor: int, raw: bytes, label: str) -> None:
            nonlocal mutated
            original(descriptor, raw, label)
            if label == "candidate binary destination" and not mutated:
                mutated = True
                self.candidate_binary.write_bytes(b"changed during snapshot\n")
                os.chmod(self.candidate_binary, 0o700)

        with mock.patch.object(common, "_write_all", side_effect=mutate_source):
            with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "(grew|changed) while it was read"):
                prepare.prepare_artifacts(self.request)
        capture = self.root / prepare.SNAPSHOT_DIRECTORY_NAME
        self.assertTrue((capture / prepare.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((capture / "session.json").exists())

    def test_fixed_root_child_collision_and_symlink_source_are_refused_without_following(self) -> None:
        collision = self.root / prepare.ARTIFACT_DIRECTORY_NAME
        collision.mkdir(mode=0o700)
        os.chmod(collision, 0o700)
        with self.assertRaisesRegex(prepare.RollbackArtifactPreparationError, "already exists"):
            prepare.prepare_artifacts(self.request)
        self.assertFalse((self.root / prepare.SNAPSHOT_DIRECTORY_NAME).exists())
        collision.rmdir()
        real = self.inputs / "real-candidate"
        self.candidate_binary.rename(real)
        os.symlink(real, self.candidate_binary)
        with self.assertRaises(prepare.RollbackArtifactPreparationError):
            prepare.prepare_artifacts(self.request)
        capture = self.root / prepare.SNAPSHOT_DIRECTORY_NAME
        self.assertTrue((capture / prepare.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((capture / "session.json").exists())

    def test_cli_wrapper_and_source_remain_raw_only_without_operational_commands(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", "-S", str(Path(prepare.__file__)), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--candidate-binary", completed.stdout)
        wrapper = Path(prepare.__file__).with_name("run_prepare_rc3_rollback_artifacts_v1.sh")
        wrapped = subprocess.run(
            ["bash", str(wrapper), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(wrapped.returncode, 0, wrapped.stderr)
        self.assertIn("--rollback-image-inspect", wrapped.stdout)
        wrapper_source = wrapper.read_text(encoding="utf-8")
        for required in ("/usr/bin/env -i", "-B -I -S", "os.lstat(script)", "sys.path.insert"):
            self.assertIn(required, wrapper_source)
        source = Path(prepare.__file__).read_text(encoding="utf-8")
        for forbidden in ("subprocess", "docker", "ssh", "os.rename", "os.replace", "os.link", "qualification_status\": \"passed"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
