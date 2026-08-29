#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 source pre-freeze checker."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import check_rc3_prefreeze as checker  # noqa: E402


CANDIDATE_ID = "riley-0.1.0-rc3"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class PrefreezeFixture:
    def __init__(self, base: Path) -> None:
        self.root = base / "checkout"
        self.root.mkdir(parents=True)
        self.defaults = b"fixture reviewed Rust serve defaults\n"
        self._write_source_tree()
        self._git("init", "--quiet")
        self.commit_all("initial source")

    def _write_source_tree(self) -> None:
        (self.root / "Cargo.toml").write_text(
            "[workspace]\n"
            'members = ["crates/riley-server"]\n'
            '\n[workspace.package]\n'
            'version = "0.1.0"\n'
            'license = "MIT"\n',
            encoding="utf-8",
        )
        (self.root / "Cargo.lock").write_text(
            "# fixture lockfile\nversion = 4\n",
            encoding="utf-8",
        )
        registry = self.root / "deploy/extensions"
        registry.mkdir(parents=True)
        (registry / "registry.json").write_text(
            '{"$schema":"registry.schema.json","schema_version":"riley.extension-registry.v1","extensions":[]}\n',
            encoding="utf-8",
        )
        server = self.root / "crates/riley-server"
        server.mkdir(parents=True)
        (server / "Cargo.toml").write_text(
            "[package]\n"
            'name = "riley-server"\n'
            'license.workspace = true\n',
            encoding="utf-8",
        )
        (server / "src").mkdir()
        (server / "src/main.rs").write_bytes(self.defaults)

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", "-C", os.fspath(self.root), *arguments],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def commit_all(self, message: str) -> None:
        self._git("add", "--all")
        self._git(
            "-c",
            "user.name=RC3 fixture",
            "-c",
            "user.email=rc3-fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            message,
        )

    @property
    def revision(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.decode("ascii").strip()

    def run(
        self,
        *,
        repository_root: Path | None = None,
        expected_revision: str | None = None,
        candidate_id: str = CANDIDATE_ID,
        defaults_sha256: str | None = None,
    ) -> dict[str, object]:
        with mock.patch.object(
            checker,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            defaults_sha256 or _sha256(self.defaults),
        ):
            return checker.check_prefreeze(
                repository_root or self.root,
                expected_revision or self.revision,
                candidate_id,
            )


class Rc3PrefreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        # Use the physical spelling: the production checker deliberately
        # refuses lexical symlink components before it asks Git anything.
        self.base = Path(self.temporary.name).resolve(strict=True)
        self.fixture = PrefreezeFixture(self.base)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext,
        reason: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def test_clean_source_returns_only_not_frozen_source_facts(self) -> None:
        before = {
            relative: (self.fixture.root / relative).read_bytes()
            for relative in (
                "Cargo.toml",
                "Cargo.lock",
                "deploy/extensions/registry.json",
                "crates/riley-server/Cargo.toml",
                "crates/riley-server/src/main.rs",
            )
        }
        report = self.fixture.run()

        self.assertEqual(report["schema_version"], checker.PREFREEZE_REPORT_VERSION)
        self.assertEqual(report["scope"], "source-pre-freeze-only")
        self.assertEqual(report["candidate_status"], "not-frozen")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["candidate_id"], CANDIDATE_ID)
        self.assertEqual(report["source_revision"], self.fixture.revision)
        self.assertEqual(report["workspace_version"], "0.1.0")
        self.assertNotIn("freeze_sha256", report)
        self.assertNotIn("passed", report)
        self.assertNotIn("archive", report)

        inputs = report["source_inputs"]
        self.assertIsInstance(inputs, dict)
        assert isinstance(inputs, dict)
        self.assertEqual(inputs["cargo_lock"]["path"], "Cargo.lock")
        self.assertEqual(
            inputs["extension_registry"]["path"],
            "deploy/extensions/registry.json",
        )
        self.assertEqual(
            inputs["server_defaults_source"]["path"],
            "crates/riley-server/src/main.rs",
        )
        self.assertEqual(
            [entry["path"] for entry in inputs["workspace_manifests"]],
            ["Cargo.toml", "crates/riley-server/Cargo.toml"],
        )
        self.assertEqual(
            self._git_status(),
            b"",
            "the checker must not stage, alter, or create source paths",
        )
        self.assertEqual(
            {
                relative: (self.fixture.root / relative).read_bytes()
                for relative in before
            },
            before,
        )

    def test_cli_prints_canonical_not_frozen_json_without_an_output_path(self) -> None:
        stdout = io.StringIO()
        with mock.patch.object(
            checker,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            _sha256(self.fixture.defaults),
        ), contextlib.redirect_stdout(stdout):
            result = checker.main(
                [
                    "--repository-root",
                    os.fspath(self.fixture.root),
                    "--expected-revision",
                    self.fixture.revision,
                    "--candidate-id",
                    CANDIDATE_ID,
                ]
            )
        self.assertEqual(result, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["candidate_status"], "not-frozen")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(self._git_status(), b"")

    def test_rejects_aliases_invalid_candidate_ids_and_version_mismatch(self) -> None:
        for invalid_revision in ("HEAD", "1" * 39, "A" * 40, "0" * 40):
            with self.subTest(invalid_revision=invalid_revision), self.assertRaises(
                checker.Rc3PrefreezeError
            ) as raised:
                self.fixture.run(expected_revision=invalid_revision)
            self.assert_reason(raised, "invalid-expected-revision")

        for invalid_candidate in (
            "riley-01.1.0-rc1",
            "riley-0.1.0-rc0",
            "riley-0.1-rc1",
            "riley-0.1.0-rc01",
        ):
            with self.subTest(invalid_candidate=invalid_candidate), self.assertRaises(
                checker.Rc3PrefreezeError
            ) as raised:
                self.fixture.run(candidate_id=invalid_candidate)
            self.assert_reason(raised, "invalid-candidate-id")

        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run(candidate_id="riley-0.1.1-rc1")
        self.assert_reason(raised, "candidate-workspace-version-mismatch")

    def test_rejects_head_root_and_checkout_drift(self) -> None:
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run(expected_revision="1" * 40)
        self.assert_reason(raised, "head-mismatch")

        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run(repository_root=self.fixture.root / "crates")
        self.assert_reason(raised, "repository-root-mismatch")

        (self.fixture.root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "checkout-not-clean")
        (self.fixture.root / "untracked.txt").unlink()

        (self.fixture.root / "Cargo.lock").write_text("dirty\n", encoding="utf-8")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "checkout-not-clean")

    def test_rejects_index_flags_that_hide_tracked_source_drift(self) -> None:
        for flag, expected_tag in (
            ("--assume-unchanged", b"h "),
            ("--skip-worktree", b"S "),
        ):
            with self.subTest(flag=flag):
                fixture = PrefreezeFixture(self.base / flag.removeprefix("--"))
                fixture._git("update-index", flag, "Cargo.lock")
                (fixture.root / "Cargo.lock").write_text(
                    "hidden tracked drift\n",
                    encoding="utf-8",
                )
                self.assertEqual(
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            os.fspath(fixture.root),
                            "status",
                            "--porcelain=v1",
                            "--untracked-files=all",
                            "--ignored=no",
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    ).stdout,
                    b"",
                )
                visibility = fixture._git("ls-files", "-v", "-z").stdout
                self.assertIn(expected_tag + b"Cargo.lock\0", visibility)
                with self.assertRaises(checker.Rc3PrefreezeError) as raised:
                    fixture.run()
                self.assert_reason(raised, "unsafe-index-flags")

    def test_rejects_file_mode_drift_hidden_by_repo_local_git_config(self) -> None:
        script = self.fixture.root / "ci-check.sh"
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(0o644)
        self.fixture.commit_all("tracked mode fixture")
        self.fixture._git("config", "core.fileMode", "false")
        script.chmod(0o755)
        self.assertEqual(
            self._git_status(),
            b"",
            "the fixture proves unpinned repository config hides mode drift",
        )
        self.assertIn(b"H ci-check.sh\0", self.fixture._git("ls-files", "-v", "-z").stdout)
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "checkout-not-clean")

    def test_rejects_linked_source_inputs_before_hash_binding(self) -> None:
        outside = self.base / "outside.lock"
        outside.write_text("outside\n", encoding="utf-8")
        lockfile = self.fixture.root / "Cargo.lock"
        lockfile.unlink()
        lockfile.symlink_to(outside)
        self.fixture.commit_all("symlink lockfile")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run()
        self.assertIn(
            getattr(raised.exception, "reason_code", ""),
            {"unsafe-evidence-path", "unsafe-evidence-directory"},
        )

    def test_rejects_hardlink_and_missing_safe_open_flag(self) -> None:
        defaults = self.fixture.root / "crates/riley-server/src/main.rs"
        alias = self.fixture.root / "defaults-alias.rs"
        os.link(defaults, alias)
        self.fixture.commit_all("hardlink defaults")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "nonunique-evidence-inode")

        second = PrefreezeFixture(self.base / "second")
        with mock.patch.object(checker.common.os, "O_NOFOLLOW", 0), self.assertRaises(
            checker.Rc3PrefreezeError
        ) as raised:
            second.run()
        self.assert_reason(raised, "missing-open-safety-flag")

    def test_rejects_release_metadata_and_reviewed_defaults_drift(self) -> None:
        cargo = self.fixture.root / "Cargo.toml"
        cargo.write_text(
            cargo.read_text(encoding="utf-8").replace('license = "MIT"', 'license = "Apache-2.0"'),
            encoding="utf-8",
        )
        self.fixture.commit_all("wrong license")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "invalid-release-metadata")

        third = PrefreezeFixture(self.base / "third")
        cargo = third.root / "Cargo.toml"
        cargo.write_text(
            cargo.read_text(encoding="utf-8").replace(
                'license = "MIT"',
                'license = "MIT"\n"license-file" = "COPYING"',
            ),
            encoding="utf-8",
        )
        third.commit_all("quoted license file")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            third.run()
        self.assert_reason(raised, "invalid-release-metadata")

        escaped_key = PrefreezeFixture(self.base / "escaped-key")
        cargo = escaped_key.root / "Cargo.toml"
        cargo.write_text(
            cargo.read_text(encoding="utf-8").replace(
                'license = "MIT"',
                'license = "MIT"\n"license\\u002dfile" = "COPYING"',
            ),
            encoding="utf-8",
        )
        escaped_key.commit_all("escaped license file key")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            escaped_key.run()
        self.assert_reason(raised, "invalid-release-metadata")

        second = PrefreezeFixture(self.base / "second")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            second.run(defaults_sha256="f" * 64)
        self.assert_reason(raised, "server-defaults-source-mismatch")

    def test_checked_in_reviewed_server_defaults_pin_matches_source(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        source = repository_root / checker.SERVER_DEFAULTS_SOURCE_PATH
        self.assertEqual(
            _sha256(source.read_bytes()),
            checker.SERVER_DEFAULTS_SOURCE_SHA256,
            "changing reviewed server defaults requires an explicit "
            "release-contract pin update",
        )

    def test_rejects_table_shaped_metadata_hidden_inside_a_multiline_string(self) -> None:
        cargo = self.fixture.root / "Cargo.toml"
        cargo.write_text(
            "[workspace]\n"
            'members = ["crates/riley-server"]\n'
            'note = """\n'
            "[workspace.package]\n"
            'version = "0.1.0"\n'
            'license = "MIT"\n'
            '"""\n',
            encoding="utf-8",
        )
        self.fixture.commit_all("fake workspace package in multiline string")
        with self.assertRaises(checker.Rc3PrefreezeError) as raised:
            self.fixture.run()
        self.assert_reason(raised, "unsupported-toml-syntax")

    def test_only_stdlib_310_and_read_only_git_surface_are_used(self) -> None:
        source = Path(checker.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("tomllib", imported)
        self.assertNotIn("release_common", imported)
        self.assertNotIn("ssh", source.lower())
        self.assertNotIn("docker", source.lower())
        self.assertNotIn("nvidia", source.lower())
        self.assertNotIn("systemctl", source.lower())
        for prohibited_git_action in ('"tag"', '"commit"', '"push"', '"archive"'):
            self.assertNotIn(prohibited_git_action, source)
        self.assertIn('"rev-parse"', source)
        self.assertIn('"status"', source)
        self.assertIn('"ls-files"', source)
        self.assertIn("GIT_OPTIONAL_LOCKS", source)
        self.assertIn("core.fsmonitor=false", source)
        self.assertIn("core.fileMode=true", source)
        self.assertIn("--ignore-submodules=none", source)
        self.assertIn("pass_fds", source)

    def _git_status(self) -> bytes:
        return subprocess.run(
            [
                "git",
                "-C",
                os.fspath(self.fixture.root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=no",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout


if __name__ == "__main__":
    unittest.main()
