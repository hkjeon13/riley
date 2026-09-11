#!/usr/bin/env python3
"""CPU-only hostile-path tests for raw assembly evidence initialization."""

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

import initialize_reconstructed_runtime_assembly_evidence_v1 as initialize  # noqa: E402


class RuntimeAssemblyEvidenceInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.parent = self.base / "external"
        self.parent.mkdir(mode=0o700)
        self.parent.chmod(0o700)
        self.source = self.base / "source"
        self.source.mkdir(mode=0o700)
        self.source.chmod(0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _initialize(self, path: Path) -> dict[str, object]:
        return initialize.initialize_reconstructed_runtime_assembly_evidence(
            path,
            source_root=self.source,
        )

    def test_creates_exact_new_private_root_and_raw_child(self) -> None:
        root = self.parent / "evidence"
        report = self._initialize(root)
        self.assertEqual(report["status"], "initialized")
        self.assertEqual(report["raw_directory"], "raw")
        raw = root / "raw"
        for directory in (root, raw):
            metadata = directory.stat()
            self.assertTrue(stat.S_ISDIR(metadata.st_mode))
            self.assertEqual(metadata.st_uid, os.geteuid())
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o700)
            self.assertGreaterEqual(metadata.st_nlink, 2)

    def test_rejects_nonsticky_writable_parent_before_create(self) -> None:
        unsafe = self.base / "unsafe"
        unsafe.mkdir(mode=0o700)
        unsafe.chmod(0o777)
        root = unsafe / "evidence"
        with self.assertRaises(initialize.AssemblyEvidenceInitializationError) as raised:
            self._initialize(root)
        self.assert_reason(raised, "unsafe-evidence-ancestor")
        self.assertFalse(root.exists())

    def test_rejects_source_tree_and_symlink_aliases_before_create(self) -> None:
        direct = self.source / "evidence"
        with self.assertRaises(initialize.AssemblyEvidenceInitializationError) as direct_raised:
            self._initialize(direct)
        self.assert_reason(direct_raised, "evidence-root-inside-source-checkout")
        self.assertFalse(direct.exists())

        alias = self.base / "source-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        aliased = alias / "evidence"
        with self.assertRaises(initialize.AssemblyEvidenceInitializationError) as alias_raised:
            self._initialize(aliased)
        self.assertIn(
            getattr(alias_raised.exception, "reason_code", None),
            {"unsafe-evidence-directory", "evidence-root-inside-source-checkout"},
        )
        self.assertFalse((self.source / "evidence").exists())

    def test_rejects_a_descendant_bind_mount_alias_from_mountinfo(self) -> None:
        mountinfo = "\n".join(
            (
                "21 1 0:42 / / rw,relatime - ext4 /dev/root rw",
                "22 21 0:42 /workspace/repo/ci /mnt/repo-ci rw,relatime - ext4 /dev/root rw",
            )
        )
        with mock.patch.object(initialize.sys, "platform", "linux"), mock.patch.object(
            Path,
            "read_text",
            return_value=mountinfo,
        ):
            with self.assertRaises(initialize.AssemblyEvidenceInitializationError) as raised:
                initialize._assert_not_source_bind_alias(  # noqa: SLF001
                    Path("/mnt/repo-ci/evidence"),
                    Path("/workspace/repo"),
                )
        self.assert_reason(raised, "evidence-root-inside-source-checkout")

    def test_rejects_a_nested_checkout_mount_alias_from_mountinfo(self) -> None:
        """A nested checkout mount can have a different backing device."""

        mountinfo = "\n".join(
            (
                "21 1 0:42 / / rw,relatime - ext4 /dev/root rw",
                "22 21 0:43 /repo-ci-root /workspace/repo/ci rw,relatime - ext4 /dev/work rw",
                "23 21 0:43 /repo-ci-root /mnt/repo-ci rw,relatime - ext4 /dev/work rw",
            )
        )
        with mock.patch.object(initialize.sys, "platform", "linux"), mock.patch.object(
            Path,
            "read_text",
            return_value=mountinfo,
        ):
            with self.assertRaises(initialize.AssemblyEvidenceInitializationError) as raised:
                initialize._assert_not_source_bind_alias(  # noqa: SLF001
                    Path("/mnt/repo-ci/evidence"),
                    Path("/workspace/repo"),
                )
        self.assert_reason(raised, "evidence-root-inside-source-checkout")

    def test_ignores_an_unrelated_namespace_mountinfo_root(self) -> None:
        """Real hosts commonly expose unrelated nsfs roots such as mnt:[id]."""

        mountinfo = "\n".join(
            (
                "21 1 0:42 / / rw,relatime - ext4 /dev/root rw",
                "22 21 0:4 mnt:[4026533116] /run/snapd/ns/example.mnt rw - nsfs nsfs rw",
            )
        )
        with mock.patch.object(initialize.sys, "platform", "linux"), mock.patch.object(
            Path,
            "read_text",
            return_value=mountinfo,
        ):
            initialize._assert_not_source_bind_alias(  # noqa: SLF001
                Path("/var/tmp/evidence"),
                Path("/workspace/repo"),
            )

    def test_rejects_a_relevant_nonpath_mountinfo_root(self) -> None:
        mountinfo = "\n".join(
            (
                "21 1 0:42 / / rw,relatime - ext4 /dev/root rw",
                "22 21 0:4 mnt:[4026533116] /workspace/repo rw - nsfs nsfs rw",
            )
        )
        with mock.patch.object(initialize.sys, "platform", "linux"), mock.patch.object(
            Path,
            "read_text",
            return_value=mountinfo,
        ):
            with self.assertRaises(initialize.AssemblyEvidenceInitializationError) as raised:
                initialize._assert_not_source_bind_alias(  # noqa: SLF001
                    Path("/var/tmp/evidence"),
                    Path("/workspace/repo"),
                )
        self.assert_reason(raised, "unsafe-evidence-directory")

    def test_rejects_existing_or_symlink_output_leaf(self) -> None:
        collision = self.parent / "collision"
        collision.mkdir(mode=0o700)
        collision.chmod(0o700)
        with self.assertRaises(initialize.AssemblyEvidenceInitializationError) as collision_raised:
            self._initialize(collision)
        self.assert_reason(collision_raised, "create-only-collision")

        target = self.parent / "target"
        target.mkdir(mode=0o700)
        target.chmod(0o700)
        alias = self.parent / "alias"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(initialize.AssemblyEvidenceInitializationError) as alias_raised:
            self._initialize(alias)
        self.assertIn(getattr(alias_raised.exception, "reason_code", None), {"create-only-collision", "unsafe-evidence-directory"})

    def test_help_and_source_are_operationally_inert(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(initialize.__file__).resolve()), "--help"],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source = Path(initialize.__file__).read_text(encoding="utf-8")
        for forbidden in ("subprocess", "socket", "requests", "urllib", "os.system", "Popen", "docker", "podman"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
