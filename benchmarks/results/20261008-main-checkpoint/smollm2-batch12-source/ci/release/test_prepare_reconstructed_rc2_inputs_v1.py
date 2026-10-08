#!/usr/bin/env python3
"""CPU-only hostile-path tests for reviewed reconstructed RC2 source inputs."""

from __future__ import annotations

import hashlib
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

import check_reconstructed_prior_baseline_v2 as baseline  # noqa: E402
import prepare_reconstructed_rc2_inputs_v1 as prepare  # noqa: E402
import provenance_v2_common as common  # noqa: E402


class ReconstructedRc2SourceInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.repository = self.base / "repository"
        self.repository.mkdir(mode=0o700)
        os.chmod(self.repository, 0o700)
        self._git("init")
        self._git("config", "user.email", "codex@example.invalid")
        self._git("config", "user.name", "Codex Test")
        (self.repository / "ci" / "release").mkdir(parents=True)
        (self.repository / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        (self.repository / "Cargo.lock").write_text("# fixture lock\n", encoding="utf-8")
        (self.repository / "ci" / "release" / "Dockerfile").write_text(
            "FROM scratch\n", encoding="utf-8"
        )
        self._git("add", "Cargo.toml", "Cargo.lock", "ci/release/Dockerfile")
        self._git(
            "commit",
            "-m",
            "fixture RC2 source",
            environment={
                "GIT_AUTHOR_DATE": "2026-08-26T09:15:43+0000",
                "GIT_COMMITTER_DATE": "2026-08-26T09:15:43+0000",
            },
        )
        self.target = self._git_stdout("rev-parse", "HEAD").strip()
        self._git("tag", "-a", "riley-0.1.0-rc2", "-m", "fixture reviewed RC2")
        self.tag_object = self._git_stdout("rev-parse", "refs/tags/riley-0.1.0-rc2").strip()
        self.expected_archive_sha256 = self._archive_sha256(self.target)
        self.patches = [
            mock.patch.object(prepare, "_repository_root", return_value=self.repository),
            mock.patch.object(prepare, "RECONSTRUCTED_RC2_TAG_OBJECT", self.tag_object),
            mock.patch.object(prepare, "RECONSTRUCTED_RC2_TARGET", self.target),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def _git(
        self,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> None:
        merged = dict(os.environ)
        if environment is not None:
            merged.update(environment)
        subprocess.run(
            ["git", "-C", os.fspath(self.repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged,
        )

    def _git_stdout(self, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", os.fspath(self.repository), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout

    def _archive_sha256(self, revision: str) -> str:
        completed = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(self.repository),
                "-c",
                "tar.umask=0002",
                "archive",
                "--format=tar",
                revision,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return hashlib.sha256(completed.stdout).hexdigest()

    def _root(self, name: str = "evidence") -> Path:
        return self.base / name

    def _verify(self, root: Path) -> dict[str, object]:
        return prepare.verify_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(prepare.ReconstructedRc2InputsError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_prepares_a_v2_compatible_reviewed_source_closure(self) -> None:
        root = self._root()
        receipt = prepare.prepare_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )
        self.assertEqual(receipt["schema_version"], prepare.SOURCE_INPUTS_VERSION)
        self.assertEqual(receipt["status"], "prepared")
        self.assertEqual(receipt["qualification_status"], "not-run")
        self.assertEqual(receipt["baseline_id"], prepare.RECONSTRUCTED_RC2_BASELINE_ID)
        self.assertEqual(receipt["expected_source_archive_sha256"], self.expected_archive_sha256)
        self.assertEqual(receipt["git_identity"]["tag_object_sha1"], self.tag_object)
        self.assertEqual(receipt["git_identity"]["target_commit_sha1"], self.target)
        self.assertNotIn("source_date_epoch", receipt["git_identity"])
        self.assertEqual(self._verify(root), receipt)
        source = baseline._source(receipt["source"], "fixture source")
        self.assertEqual(source.tag_name, "riley-0.1.0-rc2")
        self.assertEqual(source.archive.sha256, self.expected_archive_sha256)
        self.assertFalse((root / "baseline.json").exists())
        self.assertFalse((root / "reconstructed-prior-baseline-v2.json").exists())
        self.assertEqual(set(os.listdir(root)), {prepare.SOURCE_DIRECTORY_NAME, prepare.SOURCE_INPUTS_NAME})
        self.assertEqual(
            set(os.listdir(root / prepare.SOURCE_DIRECTORY_NAME)),
            {"git-tag-object.json", "git-tag-target.json", prepare.SOURCE_ARCHIVE_NAME},
        )
        for directory in (root, root / prepare.SOURCE_DIRECTORY_NAME):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        for path in (root / prepare.SOURCE_DIRECTORY_NAME).iterdir():
            metadata = path.stat()
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(metadata.st_nlink, 1)
        tag_object = json.loads(
            (root / prepare.SOURCE_DIRECTORY_NAME / "git-tag-object.json").read_text("utf-8")
        )
        self.assertEqual(tag_object["schema_version"], baseline.GIT_TAG_OBJECT_VERSION)
        self.assertEqual(tag_object["object_sha1"], self.tag_object)

    def test_rejects_an_unreviewed_source_archive_digest_before_output(self) -> None:
        root = self._root()
        self.assert_reason(
            "source-archive-digest-mismatch",
            lambda: prepare.prepare_reconstructed_rc2_inputs(
                root,
                expected_source_archive_sha256="a" * 64,
            ),
        )
        self.assertFalse(root.exists())

    def test_rejects_a_tag_object_pin_mismatch_before_output(self) -> None:
        root = self._root()
        with mock.patch.object(prepare, "RECONSTRUCTED_RC2_TAG_OBJECT", "a" * 40):
            self.assert_reason(
                "reviewed-tag-object-mismatch",
                lambda: prepare.prepare_reconstructed_rc2_inputs(
                    root,
                    expected_source_archive_sha256=self.expected_archive_sha256,
                ),
            )
        self.assertFalse(root.exists())

    def test_rejects_a_lightweight_tag_before_output(self) -> None:
        lightweight = self.base / "lightweight-repository"
        lightweight.mkdir(mode=0o700)
        os.chmod(lightweight, 0o700)
        subprocess.run(["git", "-C", os.fspath(lightweight), "init"], check=True, stdout=subprocess.PIPE)
        subprocess.run(
            ["git", "-C", os.fspath(lightweight), "config", "user.email", "codex@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", os.fspath(lightweight), "config", "user.name", "Codex Test"],
            check=True,
        )
        (lightweight / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        (lightweight / "Cargo.lock").write_text("# fixture lock\n", encoding="utf-8")
        (lightweight / "ci" / "release").mkdir(parents=True)
        (lightweight / "ci" / "release" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        subprocess.run(["git", "-C", os.fspath(lightweight), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", os.fspath(lightweight), "commit", "-m", "lightweight fixture"],
            check=True,
            stdout=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-C", os.fspath(lightweight), "tag", "riley-0.1.0-rc2"],
            check=True,
        )
        target = subprocess.run(
            ["git", "-C", os.fspath(lightweight), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        with mock.patch.object(prepare, "_repository_root", return_value=lightweight), mock.patch.object(
            prepare, "RECONSTRUCTED_RC2_TAG_OBJECT", target
        ), mock.patch.object(prepare, "RECONSTRUCTED_RC2_TARGET", target):
            self.assert_reason(
                "reviewed-tag-not-annotated",
                lambda: prepare.prepare_reconstructed_rc2_inputs(
                    self._root("lightweight-evidence"),
                    expected_source_archive_sha256="a" * 64,
                ),
            )
        self.assertFalse(self._root("lightweight-evidence").exists())

    def test_rejects_an_output_root_inside_the_source_checkout(self) -> None:
        self.assert_reason(
            "evidence-root-inside-source",
            lambda: prepare.prepare_reconstructed_rc2_inputs(
                self.repository / "evidence",
                expected_source_archive_sha256=self.expected_archive_sha256,
            ),
        )
        self.assertFalse((self.repository / "evidence").exists())

    def test_rejects_a_non_normalized_output_root_before_git(self) -> None:
        root = Path(os.fspath(self.base) + "/nested/../evidence")
        self.assert_reason(
            "non-normalized-absolute-path",
            lambda: prepare.prepare_reconstructed_rc2_inputs(
                root,
                expected_source_archive_sha256=self.expected_archive_sha256,
            ),
        )
        self.assertFalse(self._root().exists())

    def test_rejects_an_oversized_git_archive_before_output(self) -> None:
        root = self._root()
        with mock.patch.object(prepare, "MAX_SOURCE_ARCHIVE_BYTES", 1):
            self.assert_reason(
                "source-archive-size",
                lambda: prepare.prepare_reconstructed_rc2_inputs(
                    root,
                    expected_source_archive_sha256=self.expected_archive_sha256,
                ),
            )
        self.assertFalse(root.exists())

    def test_replay_rejects_archive_mutation_and_extra_entries(self) -> None:
        root = self._root()
        prepare.prepare_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )
        archive = root / prepare.SOURCE_DIRECTORY_NAME / prepare.SOURCE_ARCHIVE_NAME
        archive.write_bytes(b"mutated source archive")
        os.chmod(archive, 0o600)
        with self.assertRaises(prepare.ReconstructedRc2InputsError):
            self._verify(root)

        root = self._root("extra-entry")
        prepare.prepare_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )
        extra = root / "unexpected"
        extra.write_bytes(b"unexpected")
        os.chmod(extra, 0o600)
        self.assert_reason(
            "unexpected-evidence-entry",
            lambda: self._verify(root),
        )

    def test_replay_requires_a_reviewer_anchor_and_fixed_source_paths(self) -> None:
        root = self._root()
        receipt = prepare.prepare_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )
        self.assert_reason(
            "reviewed-source-archive-digest-mismatch",
            lambda: prepare.verify_reconstructed_rc2_inputs(
                root,
                expected_source_archive_sha256="a" * 64,
            ),
        )
        receipt["source"]["archive"]["path"] = "source/other.tar"  # type: ignore[index]
        receipt_path = root / prepare.SOURCE_INPUTS_NAME
        receipt_path.write_bytes(common.canonical_json_bytes(receipt))
        os.chmod(receipt_path, 0o600)
        self.assert_reason("source-leaf-path-mismatch", lambda: self._verify(root))

    def test_replay_wraps_a_malformed_source_descriptor(self) -> None:
        root = self._root()
        receipt = prepare.prepare_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )
        receipt["source"]["archive"]["path"] = "source//other.tar"  # type: ignore[index]
        receipt_path = root / prepare.SOURCE_INPUTS_NAME
        receipt_path.write_bytes(common.canonical_json_bytes(receipt))
        os.chmod(receipt_path, 0o600)
        self.assert_reason("invalid-relative-path", lambda: self._verify(root))

    def test_replay_rejects_a_non_string_declared_digest(self) -> None:
        root = self._root()
        receipt = prepare.prepare_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )
        receipt["expected_source_archive_sha256"] = 7
        receipt_path = root / prepare.SOURCE_INPUTS_NAME
        receipt_path.write_bytes(common.canonical_json_bytes(receipt))
        os.chmod(receipt_path, 0o600)
        self.assert_reason("invalid-expected-source-archive-sha256", lambda: self._verify(root))

    def test_replay_rejects_an_unbound_source_epoch_claim(self) -> None:
        root = self._root()
        receipt = prepare.prepare_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )
        receipt["git_identity"]["source_date_epoch"] = 1  # type: ignore[index]
        receipt_path = root / prepare.SOURCE_INPUTS_NAME
        receipt_path.write_bytes(common.canonical_json_bytes(receipt))
        os.chmod(receipt_path, 0o600)
        self.assert_reason("unknown-or-missing-field", lambda: self._verify(root))

    def test_create_only_root_refuses_reuse(self) -> None:
        root = self._root()
        receipt = prepare.prepare_reconstructed_rc2_inputs(
            root,
            expected_source_archive_sha256=self.expected_archive_sha256,
        )
        with self.assertRaises(prepare.ReconstructedRc2InputsError):
            prepare.prepare_reconstructed_rc2_inputs(
                root,
                expected_source_archive_sha256=self.expected_archive_sha256,
            )
        self.assertEqual(self._verify(root), receipt)

    def test_cli_help_needs_no_git_repository_or_release_inputs(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(prepare.__file__).resolve()), "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--expected-source-archive-sha256", completed.stdout)

    def test_schema_is_a_closed_source_only_receipt_contract(self) -> None:
        schema_path = (
            Path(prepare.__file__).resolve().parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "reconstructed-rc2-source-inputs-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"],
            "https://riley.invalid/benchmarks/release/candidates/"
            "reconstructed-rc2-source-inputs-v1.schema.json",
        )
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(schema["properties"]["schema_version"]["const"], prepare.SOURCE_INPUTS_VERSION)
        self.assertEqual(schema["properties"]["status"]["const"], "prepared")
        self.assertEqual(schema["properties"]["qualification_status"]["const"], "not-run")
        self.assertEqual(
            schema["$defs"]["archiveDescriptor"]["allOf"][1]["properties"]["path"]["const"],
            f"source/{prepare.SOURCE_ARCHIVE_NAME}",
        )
        self.assertNotIn("source_date_epoch", schema["$defs"]["gitIdentity"]["properties"])
        self.assertEqual(schema["$defs"]["gitSha1"]["allOf"][1]["not"]["const"], "0" * 40)
        self.assertEqual(schema["$defs"]["sha256"]["allOf"][1]["not"]["const"], "0" * 64)
        self.assertIn("not a reconstructed baseline", schema["description"])

    def test_producer_is_git_and_source_only(self) -> None:
        source = Path(prepare.__file__).read_text(encoding="utf-8")
        self.assertIn('"git", "-C"', source)
        self.assertIn('"archive",', source)
        self.assertIn("GIT_NO_REPLACE_OBJECTS", source)
        self.assertIn("rebase_descriptor_to_held_leaf", source)
        self.assertIn("snapshot_absolute_regular_create_only", source)
        self.assertNotIn('"docker"', source)
        self.assertNotIn('"cargo"', source)
        self.assertNotIn('"nvidia-smi"', source)
        self.assertNotIn('"ssh"', source)


if __name__ == "__main__":
    unittest.main()
