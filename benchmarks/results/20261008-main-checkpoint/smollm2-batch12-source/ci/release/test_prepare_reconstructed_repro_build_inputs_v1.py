#!/usr/bin/env python3
"""CPU-only hostile-input tests for reconstructed PR16 A/B input closure v1."""

from __future__ import annotations

import ast
import hashlib
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import prepare_reconstructed_repro_build_inputs_v1 as prepare  # noqa: E402
import prepare_reconstructed_rc2_inputs_v1 as source_inputs  # noqa: E402
import provenance_v2_common as common  # noqa: E402
import test_reproducible_build as repro_fixture  # noqa: E402


class ReconstructedReproBuildInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_tempdir = tempfile.tempdir
        tempfile.tempdir = os.fspath(Path(tempfile.gettempdir()).resolve())
        self.fixture = repro_fixture.ReproducibleBuildGateTests(methodName="runTest")
        self.fixture.setUp()
        self.base = self.fixture.root.resolve()
        self.expected_source_sha256 = hashlib.sha256(
            self.fixture.source_archive.read_bytes()
        ).hexdigest()
        self.source_root = self.base / "source-inputs"
        self.source_directory = self.source_root / source_inputs.SOURCE_DIRECTORY_NAME
        self.source_root.mkdir(mode=0o700)
        self.source_directory.mkdir(mode=0o700)
        os.chmod(self.source_root, 0o700)
        os.chmod(self.source_directory, 0o700)
        tag_object_raw = common.canonical_json_bytes({"fixture": "tag-object"})
        tag_target_raw = common.canonical_json_bytes({"fixture": "tag-target"})
        archive_raw = self.fixture.source_archive.read_bytes()
        for name, raw in (
            ("git-tag-object.json", tag_object_raw),
            ("git-tag-target.json", tag_target_raw),
            (source_inputs.SOURCE_ARCHIVE_NAME, archive_raw),
        ):
            path = self.source_directory / name
            path.write_bytes(raw)
            os.chmod(path, 0o600)
        source = {
            "tag_name": source_inputs.RECONSTRUCTED_RC2_TAG,
            "tag_object": common.descriptor_for_bytes(
                "source/git-tag-object.json", tag_object_raw, "fixture tag object"
            ).as_json(),
            "tag_target": common.descriptor_for_bytes(
                "source/git-tag-target.json", tag_target_raw, "fixture tag target"
            ).as_json(),
            "archive": common.descriptor_for_bytes(
                f"source/{source_inputs.SOURCE_ARCHIVE_NAME}", archive_raw, "fixture source archive"
            ).as_json(),
        }
        self.source_row = {
            "schema_version": source_inputs.SOURCE_INPUTS_VERSION,
            "status": "prepared",
            "qualification_status": "not-run",
            "baseline_id": source_inputs.RECONSTRUCTED_RC2_BASELINE_ID,
            "source": source,
            "git_identity": {
                "tag_ref": source_inputs.RECONSTRUCTED_RC2_TAG_REF,
                "tag_object_sha1": source_inputs.RECONSTRUCTED_RC2_TAG_OBJECT,
                "target_commit_sha1": repro_fixture.REVISION,
            },
            "expected_source_archive_sha256": self.expected_source_sha256,
            "archive_generation": source_inputs.ARCHIVE_GENERATION,
        }
        receipt_path = self.source_root / source_inputs.SOURCE_INPUTS_NAME
        receipt_path.write_bytes(common.canonical_json_bytes(self.source_row))
        os.chmod(receipt_path, 0o600)
        self.patches = [
            mock.patch.object(
                prepare.source_inputs,
                "verify_reconstructed_rc2_inputs_fd",
                side_effect=self._verify_source_inputs,
            ),
            mock.patch.object(
                prepare.source_inputs,
                "RECONSTRUCTED_RC2_TARGET",
                repro_fixture.REVISION,
            ),
            # The hostile fixture uses a synthetic active release bundle.  The
            # historical verifier itself is exercised in test_release; keep
            # this closure test focused on held-FD input behavior.
            mock.patch.object(
                prepare.reproducibility,
                "validate_reconstructed_rc2_reproducibility_inputs",
                side_effect=prepare.reproducibility.validate_reproducibility_inputs,
            ),
        ]
        self.reconstructed_validator = self.patches[-1].start()
        for patch in self.patches[:-1]:
            patch.start()
        self.repro_a = self.fixture.package("A")
        self.repro_b = self.fixture.package("B")
        # The preparer deliberately rejects mutable host inputs.  Do not let
        # the invoking user's umask change this fixture's intended contract.
        os.chmod(self.repro_a, 0o600)
        os.chmod(self.repro_b, 0o600)

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.fixture.tearDown()
        tempfile.tempdir = self.previous_tempdir

    def _verify_source_inputs(
        self,
        _root_fd: int,
        *,
        expected_source_archive_sha256: str,
    ) -> dict[str, object]:
        if expected_source_archive_sha256 != self.expected_source_sha256:
            raise source_inputs.ReconstructedRc2InputsError("fixture reviewer SHA mismatch")
        return dict(self.source_row)

    def _root(self, name: str = "repro-inputs") -> Path:
        return self.base / name

    def _prepare(self, root: Path | None = None) -> dict[str, object]:
        return prepare.prepare_reconstructed_repro_build_inputs(
            root or self._root(),
            source_input_root=self.source_root,
            expected_source_archive_sha256=self.expected_source_sha256,
            expected_build_image_id=repro_fixture.IMAGE_ID,
            repro_build_a=self.repro_a,
            repro_build_b=self.repro_b,
        )

    def _verify(self, root: Path) -> dict[str, object]:
        return prepare.verify_reconstructed_repro_build_inputs(
            root,
            source_input_root=self.source_root,
            expected_source_archive_sha256=self.expected_source_sha256,
            expected_build_image_id=repro_fixture.IMAGE_ID,
        )

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(prepare.ReproBuildInputsError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_prepares_and_replays_a_closed_a_b_input_root(self) -> None:
        root = self._root()
        receipt = self._prepare(root)
        self.assertEqual(receipt["schema_version"], prepare.REPRO_BUILD_INPUTS_VERSION)
        self.assertEqual(receipt["status"], "prepared")
        self.assertEqual(receipt["qualification_status"], "not-run")
        self.assertEqual(receipt["capture_scope"], prepare.CAPTURE_SCOPE)
        self.assertEqual(receipt["reproducibility_contract"]["source_revision"], repro_fixture.REVISION)
        self.assertEqual(receipt["reproducibility_contract"]["build_image_id"], repro_fixture.IMAGE_ID)
        self.assertGreaterEqual(self.reconstructed_validator.call_count, 1)
        self.assertEqual(self._verify(root), receipt)
        self.assertEqual(set(os.listdir(root)), {prepare.REPRO_BUILDS_DIRECTORY_NAME, prepare.REPRO_BUILD_INPUTS_NAME})
        self.assertEqual(set(os.listdir(root / prepare.REPRO_BUILDS_DIRECTORY_NAME)), {"a", "b"})
        for arm in prepare.RECONSTRUCTION_IDS:
            arm_root = root / prepare.REPRO_BUILDS_DIRECTORY_NAME / arm
            self.assertEqual(
                set(os.listdir(arm_root)),
                {f"repro-build-{arm}.tar", "build.json", "riley", "riley.tar.gz"},
            )
            self.assertEqual(stat.S_IMODE(arm_root.stat().st_mode), 0o700)
            for leaf in arm_root.iterdir():
                metadata = leaf.stat()
                self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
                self.assertEqual(metadata.st_nlink, 1)
        self.assertEqual(receipt["equality"]["binary"]["sha256"], receipt["builds"]["a"]["binary"]["sha256"])
        self.assertEqual(receipt["not_established"], prepare.NOT_ESTABLISHED)

    def test_rejects_wrong_reviewer_source_sha_before_output(self) -> None:
        root = self._root()
        self.assert_reason(
            "invalid-source-inputs",
            lambda: prepare.prepare_reconstructed_repro_build_inputs(
                root,
                source_input_root=self.source_root,
                expected_source_archive_sha256="a" * 64,
                expected_build_image_id=repro_fixture.IMAGE_ID,
                repro_build_a=self.repro_a,
                repro_build_b=self.repro_b,
            ),
        )
        self.assertFalse(root.exists())

    def test_rejects_wrong_reviewed_builder_image_before_output(self) -> None:
        root = self._root()
        self.assert_reason(
            "invalid-pr16-reproducibility-evidence",
            lambda: prepare.prepare_reconstructed_repro_build_inputs(
                root,
                source_input_root=self.source_root,
                expected_source_archive_sha256=self.expected_source_sha256,
                expected_build_image_id="sha256:" + "c" * 64,
                repro_build_a=self.repro_a,
                repro_build_b=self.repro_b,
            ),
        )
        self.assertTrue(root.exists())
        self.assertFalse((root / prepare.REPRO_BUILD_INPUTS_NAME).exists())

    def test_rejects_arm_swap(self) -> None:
        root = self._root()
        self.assert_reason(
            "invalid-pr16-reproducibility-evidence",
            lambda: prepare.prepare_reconstructed_repro_build_inputs(
                root,
                source_input_root=self.source_root,
                expected_source_archive_sha256=self.expected_source_sha256,
                expected_build_image_id=repro_fixture.IMAGE_ID,
                repro_build_a=self.repro_b,
                repro_build_b=self.repro_a,
            ),
        )
        self.assertFalse((root / prepare.REPRO_BUILD_INPUTS_NAME).exists())

    def test_rejects_same_input_inode_before_output(self) -> None:
        root = self._root()
        self.assert_reason(
            "input-alias",
            lambda: prepare.prepare_reconstructed_repro_build_inputs(
                root,
                source_input_root=self.source_root,
                expected_source_archive_sha256=self.expected_source_sha256,
                expected_build_image_id=repro_fixture.IMAGE_ID,
                repro_build_a=self.repro_a,
                repro_build_b=self.repro_a,
            ),
        )
        self.assertFalse(root.exists())

    def test_rejects_a_structurally_valid_b_binary_mismatch(self) -> None:
        second_binary, second_bundle, second_native = self.fixture.make_release(
            "different", repro_fixture.fixture_elf() + b"different server bytes"
        )
        changed_b = self.fixture.package(
            "B",
            binary=second_binary,
            bundle=second_bundle,
            native=second_native,
            output_name="changed-b.tar",
        )
        os.chmod(changed_b, 0o600)
        root = self._root()
        self.assert_reason(
            "invalid-pr16-reproducibility-evidence",
            lambda: prepare.prepare_reconstructed_repro_build_inputs(
                root,
                source_input_root=self.source_root,
                expected_source_archive_sha256=self.expected_source_sha256,
                expected_build_image_id=repro_fixture.IMAGE_ID,
                repro_build_a=self.repro_a,
                repro_build_b=changed_b,
            ),
        )
        self.assertFalse((root / prepare.REPRO_BUILD_INPUTS_NAME).exists())

    def test_rejects_tampered_selected_binary_on_replay(self) -> None:
        root = self._root()
        self._prepare(root)
        selected = root / prepare.REPRO_BUILDS_DIRECTORY_NAME / "a" / "riley"
        selected.write_bytes(b"tampered")
        os.chmod(selected, 0o600)
        self.assert_reason("evidence-length-mismatch", lambda: self._verify(root))

    def test_rejects_extra_evidence_entry_on_replay(self) -> None:
        root = self._root()
        self._prepare(root)
        extra = root / prepare.REPRO_BUILDS_DIRECTORY_NAME / "a" / "extra"
        extra.write_bytes(b"extra")
        os.chmod(extra, 0o600)
        self.assert_reason("unexpected-evidence-entry", lambda: self._verify(root))

    def test_rejects_source_root_overlap_before_output(self) -> None:
        root = self.source_root / "nested-output"
        self.assert_reason(
            "output-source-overlap",
            lambda: self._prepare(root),
        )
        self.assertFalse(root.exists())

    def test_source_file_does_not_import_operational_clients(self) -> None:
        source = Path(prepare.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_imports = {"subprocess", "socket", "urllib", "requests"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({alias.name.split(".")[0] for alias in node.names} & forbidden_imports)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden_imports)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
