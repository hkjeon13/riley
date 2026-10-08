#!/usr/bin/env python3
"""CPU-only hostile-path tests for static RC3 rollback evidence preparation."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import check_rc3_rollback_provenance_v3 as rollback  # noqa: E402
import prepare_rc3_rollback_evidence_v1 as prepare  # noqa: E402
import provenance_v2_common as common  # noqa: E402
from test_check_reconstructed_prior_baseline import BaselineFixture as LegacyBaselineFixture  # noqa: E402
from test_check_reconstructed_prior_baseline_v2 import BaselineV2Fixture  # noqa: E402


class RollbackEvidencePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.root = self._new_baseline_root("evidence")
        self.inputs = self.base / "inputs"
        self.inputs.mkdir(mode=0o700)
        os.chmod(self.inputs, 0o700)
        self.freeze = self._write_input("freeze.raw", b'{"freeze":"future-external-input"}\n')
        self.base_report = self._write_input(
            "base-release-candidate-report.raw",
            b'{"base_release_candidate_report":"future-external-input"}\n',
        )
        self.configuration = self._write_input(
            "stable-default-configuration.raw",
            b'{"configuration":"opaque-static-input"}\n',
        )
        self.request = self._request(self.root)
        self.prepare_root_patch = mock.patch.object(prepare, "_source_root", return_value=self.repository)
        self.prepare_root_patch.start()

    def tearDown(self) -> None:
        self.prepare_root_patch.stop()
        self.temporary.cleanup()

    def _new_baseline_root(
        self,
        name: str,
        *,
        reviewed_target: bool = True,
        reviewed_tag_object: bool = True,
    ) -> Path:
        root = self.base / name
        root.mkdir(mode=0o700)
        os.chmod(root, 0o700)
        BaselineV2Fixture(
            root,
            target_commit_sha1=(rollback.RECONSTRUCTED_ROLLBACK_TARGET if reviewed_target else "b" * 40),
            tag_object_sha1=(
                rollback.RECONSTRUCTED_ROLLBACK_TAG_OBJECT
                if reviewed_tag_object
                else "a" * 40
            ),
        )
        return root

    def _write_input(self, name: str, raw: bytes, mode: int = 0o644) -> Path:
        path = self.inputs / name
        path.write_bytes(raw)
        os.chmod(path, mode)
        return path

    def _request(self, root: Path, **changes: object) -> prepare.EvidencePreparationRequest:
        request = prepare.EvidencePreparationRequest(
            evidence_root=root,
            baseline_manifest_path="baseline.json",
            candidate_id="riley-0.1.0-rc3",
            freeze_input=self.freeze,
            base_release_candidate_report_input=self.base_report,
            stable_default_configuration_input=self.configuration,
        )
        return replace(request, **changes)

    def _snapshot(self, name: str) -> Path:
        return self.root / prepare.INPUTS_DIRECTORY_NAME / name

    def _preparation(self) -> Path:
        return self.root / prepare.PREPARATION_DIRECTORY_NAME

    def _baseline_bytes(self, root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
            and path.relative_to(root).parts[0]
            not in {prepare.PREPARATION_DIRECTORY_NAME, prepare.INPUTS_DIRECTORY_NAME}
        }

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(prepare.RollbackEvidencePreparationError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_prepares_verified_baseline_and_three_private_static_snapshots(self) -> None:
        baseline_before = self._baseline_bytes(self.root)
        session = prepare.prepare_rollback_evidence(self.request)
        self.assertEqual(session["schema_version"], prepare.SESSION_VERSION)
        self.assertEqual(session["capture_status"], "captured")
        self.assertEqual(session["qualification_status"], "not-run")
        self.assertEqual(session["authority"], prepare.AUTHORITY)
        self.assertEqual(session["candidate_id"], "riley-0.1.0-rc3")
        self.assertEqual(session["configuration_profile"], "stable-default")
        self.assertEqual(
            session["reconstructed_baseline"]["target_commit_sha1"],
            rollback.RECONSTRUCTED_ROLLBACK_TARGET,
        )
        self.assertEqual(prepare.verify_rollback_evidence_preparation(self.root), session)
        self.assertEqual(baseline_before, self._baseline_bytes(self.root))
        encoded = common.canonical_json_bytes(session).decode("utf-8")
        for source in (self.freeze, self.base_report, self.configuration):
            self.assertNotIn(os.fspath(source), encoded)
        expected = {
            prepare.FREEZE_NAME: self.freeze.read_bytes(),
            prepare.BASE_REPORT_NAME: self.base_report.read_bytes(),
            prepare.CONFIGURATION_NAME: self.configuration.read_bytes(),
        }
        for name, raw in expected.items():
            snapshot = self._snapshot(name)
            self.assertEqual(snapshot.read_bytes(), raw)
            self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o600)
            self.assertEqual(snapshot.stat().st_nlink, 1)
        preparation = self._preparation()
        self.assertFalse((preparation / prepare.INCOMPLETE_MARKER_NAME).exists())
        complete = preparation / prepare.COMPLETE_MARKER_NAME
        intent = preparation / prepare.COMPLETE_INTENT_NAME
        self.assertEqual((complete.stat().st_dev, complete.stat().st_ino), (intent.stat().st_dev, intent.stat().st_ino))
        for marker in (complete, intent):
            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            self.assertEqual(marker.stat().st_nlink, 2)
        for directory in (self.root, preparation, self.root / prepare.INPUTS_DIRECTORY_NAME):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)

    def test_rejects_unreviewed_baseline_target_before_any_output(self) -> None:
        bad_root = self._new_baseline_root("unreviewed", reviewed_target=False)
        self.assert_reason(
            "unsupported-reconstructed-baseline",
            lambda: prepare.prepare_rollback_evidence(self._request(bad_root)),
        )
        self.assertFalse((bad_root / prepare.PREPARATION_DIRECTORY_NAME).exists())
        self.assertFalse((bad_root / prepare.INPUTS_DIRECTORY_NAME).exists())

    def test_rejects_unreviewed_baseline_tag_object_before_any_output(self) -> None:
        bad_root = self._new_baseline_root("unreviewed-tag-object", reviewed_tag_object=False)
        self.assert_reason(
            "reviewed-reconstructed-tag-object-mismatch",
            lambda: prepare.prepare_rollback_evidence(self._request(bad_root)),
        )
        self.assertFalse((bad_root / prepare.PREPARATION_DIRECTORY_NAME).exists())
        self.assertFalse((bad_root / prepare.INPUTS_DIRECTORY_NAME).exists())

    def test_rejects_legacy_binary_unbound_baseline_before_any_output(self) -> None:
        legacy_root = self.base / "legacy"
        legacy_root.mkdir(mode=0o700)
        os.chmod(legacy_root, 0o700)
        LegacyBaselineFixture(
            legacy_root,
            target_commit_sha1=rollback.RECONSTRUCTED_ROLLBACK_TARGET,
        )
        self.assert_reason(
            "rollback-binary-provenance-required",
            lambda: prepare.prepare_rollback_evidence(self._request(legacy_root)),
        )
        self.assertFalse((legacy_root / prepare.PREPARATION_DIRECTORY_NAME).exists())
        self.assertFalse((legacy_root / prepare.INPUTS_DIRECTORY_NAME).exists())

    def test_rejects_nonadjacent_or_different_version_candidate_before_opening_root(self) -> None:
        for candidate in ("riley-0.1.0-rc2", "riley-0.1.0-rc4", "riley-0.1.1-rc3"):
            with self.subTest(candidate=candidate):
                self.assert_reason(
                    "candidate-not-immediate-prior-baseline-successor",
                    lambda candidate=candidate: prepare.prepare_rollback_evidence(
                        self._request(self.root, candidate_id=candidate)
                    ),
                )
                self.assertFalse((self.root / prepare.PREPARATION_DIRECTORY_NAME).exists())

    def test_rejects_altered_baseline_leaf_before_static_output(self) -> None:
        leaf = self.root / "source" / "riley-0.1.0-rc2.tar.zst"
        leaf.write_bytes(b"altered reconstructed source archive")
        with self.assertRaises(prepare.RollbackEvidencePreparationError):
            prepare.prepare_rollback_evidence(self.request)
        self.assertFalse((self.root / prepare.PREPARATION_DIRECTORY_NAME).exists())
        self.assertFalse((self.root / prepare.INPUTS_DIRECTORY_NAME).exists())

    def test_rejects_unsafe_root_source_tree_and_fixed_child_collision(self) -> None:
        os.chmod(self.root, 0o755)
        with self.assertRaisesRegex(prepare.RollbackEvidencePreparationError, "mode must be exactly 0700"):
            prepare.prepare_rollback_evidence(self.request)
        os.chmod(self.root, 0o700)
        inside_source = self.repository / "evidence"
        inside_source.mkdir(mode=0o700)
        os.chmod(inside_source, 0o700)
        self.assert_reason(
            "evidence-root-inside-source-checkout",
            lambda: prepare.prepare_rollback_evidence(self._request(inside_source)),
        )
        collision = self.root / prepare.INPUTS_DIRECTORY_NAME
        collision.mkdir(mode=0o700)
        os.chmod(collision, 0o700)
        self.assert_reason("create-only-collision", lambda: prepare.prepare_rollback_evidence(self.request))
        self.assertFalse((self.root / prepare.PREPARATION_DIRECTORY_NAME).exists())

    def test_rejects_relative_root_internal_and_duplicate_input_paths_before_output(self) -> None:
        with self.subTest("relative"):
            self.assert_reason(
                "invalid-absolute-path",
                lambda: prepare.prepare_rollback_evidence(
                    self._request(self.root, freeze_input=Path("relative-freeze.raw"))
                ),
            )
        with self.subTest("root internal"):
            self.assert_reason(
                "input-inside-evidence-root",
                lambda: prepare.prepare_rollback_evidence(
                    self._request(self.root, freeze_input=self.root / "baseline.json")
                ),
            )
        with self.subTest("duplicate"):
            self.assert_reason(
                "duplicate-input-path",
                lambda: prepare.prepare_rollback_evidence(
                    self._request(self.root, base_release_candidate_report_input=self.freeze)
                ),
            )
        self.assertFalse((self.root / prepare.PREPARATION_DIRECTORY_NAME).exists())

    def test_rejects_symlink_static_source_without_following(self) -> None:
        target = self.inputs / "real-freeze.raw"
        self.freeze.rename(target)
        os.symlink(target, self.freeze)
        with self.assertRaises(prepare.RollbackEvidencePreparationError):
            prepare.prepare_rollback_evidence(self.request)
        self.assertTrue((self.root / prepare.PREPARATION_DIRECTORY_NAME / prepare.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((self._preparation() / "session.json").exists())

    def test_rejects_hardlinked_static_source(self) -> None:
        hard = self.inputs / "hard-freeze.raw"
        os.link(self.freeze, hard)
        with self.assertRaises(prepare.RollbackEvidencePreparationError):
            prepare.prepare_rollback_evidence(self._request(self.root, freeze_input=hard))

    def test_rejects_group_writable_static_source(self) -> None:
        os.chmod(self.freeze, 0o664)
        with self.assertRaises(prepare.RollbackEvidencePreparationError):
            prepare.prepare_rollback_evidence(self.request)

    def test_rejects_empty_static_source(self) -> None:
        self.freeze.write_bytes(b"")
        os.chmod(self.freeze, 0o644)
        with self.assertRaises(prepare.RollbackEvidencePreparationError):
            prepare.prepare_rollback_evidence(self.request)

    def test_rejects_directory_fifo_symlinked_parent_and_oversized_static_sources(self) -> None:
        directory = self.inputs / "freeze-directory"
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        fifo = self.inputs / "freeze.fifo"
        os.mkfifo(fifo, 0o600)
        oversized = self._write_input("freeze-oversized.raw", b"x" * (prepare.MAX_INPUT_BYTES + 1))
        real_parent = self.inputs / "real-parent"
        real_parent.mkdir(mode=0o700)
        os.chmod(real_parent, 0o700)
        parent_leaf = real_parent / "freeze.raw"
        parent_leaf.write_bytes(b"via symlinked parent\n")
        os.chmod(parent_leaf, 0o644)
        alias_parent = self.inputs / "alias-parent"
        os.symlink(real_parent, alias_parent)
        cases = {
            "directory": directory,
            "fifo": fifo,
            "oversized": oversized,
            "symlinked-parent": alias_parent / "freeze.raw",
        }
        for name, source in cases.items():
            with self.subTest(kind=name):
                root = self._new_baseline_root(f"invalid-{name}")
                with self.assertRaises(prepare.RollbackEvidencePreparationError):
                    prepare.prepare_rollback_evidence(self._request(root, freeze_input=source))
                self.assertTrue(
                    (root / prepare.PREPARATION_DIRECTORY_NAME / prepare.INCOMPLETE_MARKER_NAME).is_file()
                )
                self.assertFalse((root / prepare.PREPARATION_DIRECTORY_NAME / "session.json").exists())

    def test_rejects_source_mutation_during_snapshot_and_leaves_incomplete_marker(self) -> None:
        original = common._write_all
        changed = False

        def mutate_source(descriptor: int, raw: bytes, label: str) -> None:
            nonlocal changed
            original(descriptor, raw, label)
            if label == "freeze static input destination" and not changed:
                changed = True
                self.freeze.write_bytes(b'{"freeze":"changed-during-copy"}\n')
                os.chmod(self.freeze, 0o644)

        with mock.patch.object(common, "_write_all", side_effect=mutate_source):
            with self.assertRaises(prepare.RollbackEvidencePreparationError):
                prepare.prepare_rollback_evidence(self.request)
        self.assertTrue((self._preparation() / prepare.INCOMPLETE_MARKER_NAME).is_file())
        self.assertFalse((self._preparation() / "session.json").exists())

    def test_terminal_replay_rejects_snapshot_and_completion_drift(self) -> None:
        session = prepare.prepare_rollback_evidence(self.request)
        snapshot = self._snapshot(prepare.FREEZE_NAME)
        os.chmod(snapshot, 0o644)
        with self.assertRaisesRegex(prepare.RollbackEvidencePreparationError, "mode 0600"):
            prepare.verify_rollback_evidence_preparation(self.root)
        os.chmod(snapshot, 0o600)
        self.assertEqual(prepare.verify_rollback_evidence_preparation(self.root), session)
        (self._preparation() / prepare.COMPLETE_MARKER_NAME).unlink()
        with self.assertRaisesRegex(prepare.RollbackEvidencePreparationError, "entries differ|paired marker"):
            prepare.verify_rollback_evidence_preparation(self.root)

    def test_terminal_replay_rejects_extra_entry_and_held_input_directory_replacement(self) -> None:
        prepare.prepare_rollback_evidence(self.request)
        extra = self._preparation() / "unexpected.raw"
        extra.write_bytes(b"unexpected")
        os.chmod(extra, 0o600)
        with self.assertRaisesRegex(prepare.RollbackEvidencePreparationError, "entries differ"):
            prepare.verify_rollback_evidence_preparation(self.root)
        extra.unlink()
        original = common.verify_private_snapshot_descriptor_file
        swapped = False

        def verify_then_swap(*args: object, **kwargs: object) -> None:
            nonlocal swapped
            original(*args, **kwargs)  # type: ignore[arg-type]
            if not swapped:
                swapped = True
                inputs = self.root / prepare.INPUTS_DIRECTORY_NAME
                moved = self.root / "old-static-inputs"
                os.rename(inputs, moved)
                inputs.mkdir(mode=0o700)
                os.chmod(inputs, 0o700)

        with mock.patch.object(common, "verify_private_snapshot_descriptor_file", side_effect=verify_then_swap):
            with self.assertRaisesRegex(prepare.RollbackEvidencePreparationError, "held static rollback evidence inputs"):
                prepare.verify_rollback_evidence_preparation(self.root)

    def test_cli_wrapper_rejects_legacy_id_before_root_access_and_stays_raw_only(self) -> None:
        script = Path(prepare.__file__)
        environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"}
        help_result = subprocess.run(
            [sys.executable, "-B", "-S", str(script), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--baseline-manifest-path", help_result.stdout)
        legacy = subprocess.run(
            [
                sys.executable,
                "-B",
                "-S",
                str(script),
                "--evidence-root",
                "/no/such/evidence-root",
                "--baseline-manifest-path",
                "baseline.json",
                "--candidate-id",
                "riley-0.1.0-rc3",
                "--freeze-input",
                "/no/such/freeze",
                "--base-release-candidate-report-input",
                "/no/such/report",
                "--stable-default-configuration-input",
                "/no/such/configuration",
                "--id=0",
            ],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertNotEqual(legacy.returncode, 0)
        self.assertIn("unrecognized arguments: --id=0", legacy.stderr)
        wrapper = script.with_name("run_prepare_rc3_rollback_evidence_v1.sh")
        wrapped = subprocess.run(
            ["bash", str(wrapper), "--help"],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(wrapped.returncode, 0, wrapped.stderr)
        self.assertIn("--freeze-input", wrapped.stdout)
        wrapper_source = wrapper.read_text(encoding="utf-8")
        for required in ("/usr/bin/env -i", "-B -I -S", "os.lstat(script)", "sys.path.insert"):
            self.assertIn(required, wrapper_source)
        source = script.read_text(encoding="utf-8")
        for forbidden in (
            "subprocess",
            "docker",
            "ssh",
            "os.rename",
            "os.replace",
            "os.link",
            "systemctl",
            "nvidia-smi",
            "qualification_status\": \"passed",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
