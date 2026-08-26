#!/usr/bin/env python3
"""CPU-only tests for the deterministic final-candidate manifest writer."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import write_release_candidate_manifest as writer  # noqa: E402
from build_release_bundle import build_bundle  # noqa: E402
from release_common import MIT_LICENSE_BYTES, canonical_json_bytes  # noqa: E402
from test_release import (  # noqa: E402
    EPOCH,
    fixture_elf,
    install_reviewed_server_defaults_source,
)


REVISION = "1a2b3c4d5e6f78901234567890abcdef12345678"
CANDIDATE_ID = "rustinfer-0.1.0-rc1"


def _digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


class WriterFixture:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "submitted-evidence"
        self.root.mkdir()
        artifact_directory = self.root / "artifacts"
        artifact_directory.mkdir()
        self.repository = base / "repository"
        self.repository.mkdir()
        (self.repository / "Cargo.toml").write_text(
            "[workspace]\n"
            "members = []\n"
            "[workspace.package]\n"
            'version = "0.1.0"\n'
            'license = "MIT"\n',
            encoding="utf-8",
        )
        (self.repository / "LICENSE").write_bytes(MIT_LICENSE_BYTES)
        install_reviewed_server_defaults_source(self.repository)

        self.relative_paths: dict[str, str] = {}
        self.paths: dict[str, Path] = {}
        for ordinal, spec in enumerate(writer.ARTIFACT_SPECS, 1):
            relative = f"artifacts/{ordinal:02d}-{spec.key}.bin"
            path = self.root / relative
            path.write_bytes(f"fixture for {spec.key}\n".encode("ascii"))
            self.relative_paths[spec.key] = relative
            self.paths[spec.key] = path

        self.paths["source_archive"].write_bytes(b"trusted source archive\n")
        self.paths["release_binary"].write_bytes(fixture_elf())
        self.paths["release_binary"].chmod(0o755)
        build_bundle(
            binary_path=self.paths["release_binary"],
            output=self.paths["release_bundle"],
            repository_root=self.repository,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
        )
        self.paths["python_free_e2e_correctness_golden"].write_bytes(
            canonical_json_bytes({"fixture": "trusted correctness golden"})
        )
        for key in (
            "native_correctness_candidate_executable",
            "reproducible_profile_binary",
        ):
            self.paths[key].write_bytes(fixture_elf() + key.encode("ascii"))
            self.paths[key].chmod(0o755)

        self.release_image_id = "sha256:" + _digest(b"release image")
        self.reproducible_image_id = "sha256:" + _digest(
            b"reproducible-build image"
        )
        self.cuda_image_id = "sha256:" + _digest(b"cuda-build image")
        self.optimization_image_id = "sha256:" + _digest(
            b"optimization-build image"
        )

    @property
    def anchors(self) -> dict[str, str]:
        return {
            "expected_candidate_id": CANDIDATE_ID,
            "expected_revision": REVISION,
            "expected_source_archive_sha256": _digest(
                self.paths["source_archive"].read_bytes()
            ),
            "expected_release_image_id": self.release_image_id,
            "expected_reproducible_build_image_id": self.reproducible_image_id,
            "expected_cuda_build_image_id": self.cuda_image_id,
            "expected_optimization_build_image_id": self.optimization_image_id,
            "expected_correctness_golden_sha256": _digest(
                self.paths[
                    "python_free_e2e_correctness_golden"
                ].read_bytes()
            ),
        }

    def passed_report(
        self,
        manifest_path: Path,
        _evidence_root: Path,
        **anchors: object,
    ) -> dict[str, Any]:
        manifest_fd = anchors["manifest_fd"]
        assert isinstance(manifest_fd, int)
        raw = os.pread(manifest_fd, os.fstat(manifest_fd).st_size, 0)
        return {
            "schema_version": writer.REPORT_VERSION,
            "status": "passed",
            "passed": True,
            "candidate_id": anchors["expected_candidate_id"],
            "manifest_sha256": _digest(raw),
            "bindings": {
                "git_revision": anchors["expected_revision"],
                "source_archive_sha256": anchors[
                    "expected_source_archive_sha256"
                ],
                "release_image_sha256": anchors[
                    "expected_release_image_id"
                ].removeprefix("sha256:"),
                "build_image_ids": {
                    "reproducible_build": anchors[
                        "expected_reproducible_build_image_id"
                    ],
                    "cuda_fault": anchors["expected_cuda_build_image_id"],
                    "optimization_correctness": anchors[
                        "expected_optimization_build_image_id"
                    ],
                },
                "correctness_golden_sha256": anchors[
                    "expected_correctness_golden_sha256"
                ],
            },
            "checks": [
                {"name": name, "passed": True}
                for name in writer.FINAL_CHECK_NAMES
            ],
            "errors": [],
        }

    def write(
        self,
        output: Path,
        *,
        evaluator: Callable[..., dict[str, Any]] | None = None,
        artifact_paths: dict[str, str] | None = None,
        anchors: dict[str, str] | None = None,
    ) -> tuple[writer.WriteResult, mock.Mock]:
        selected_evaluator = evaluator or self.passed_report
        with mock.patch.object(
            writer.candidate_checker,
            "evaluate",
            side_effect=selected_evaluator,
        ) as checker:
            result = writer.write_manifest(
                self.root,
                output,
                artifact_paths=(artifact_paths or self.relative_paths),
                **(anchors or self.anchors),
            )
        return result, checker


class ManifestWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        # macOS exposes the temporary root through /var, which is itself a
        # symlink.  The writer deliberately rejects every lexical symlink
        # component, so tests use the physical /private/var spelling.
        self.base = Path(self.temporary.name).resolve()
        self.fixture = WriterFixture(self.base)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_no_staging_files(self, output: Path) -> None:
        self.assertEqual(
            list(output.parent.glob(f".{output.name}.tmp-*")),
            [],
        )

    def test_writes_canonical_v2_manifest_and_supplies_held_fd_to_checker(
        self,
    ) -> None:
        output = self.base / "candidate.json"
        result, checker = self.fixture.write(output)

        raw = output.read_bytes()
        document = json.loads(raw)
        self.assertEqual(raw, canonical_json_bytes(document))
        self.assertEqual(document, result.manifest)
        self.assertEqual(document["schema_version"], writer.MANIFEST_VERSION)
        self.assertEqual(document["candidate_id"], CANDIDATE_ID)
        self.assertEqual(
            document["evidence"]["reproducible_build"]["source_date_epoch"],
            EPOCH,
        )
        self.assertEqual(result.manifest_sha256, _digest(raw))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o644)
        self.assert_no_staging_files(output)

        descriptors: list[dict[str, str]] = []

        def collect(value: object) -> None:
            if isinstance(value, dict):
                if set(value) == {"path", "sha256"}:
                    descriptors.append(value)
                else:
                    for child in value.values():
                        collect(child)

        collect(document)
        self.assertEqual(len(descriptors), 21)
        self.assertEqual(
            {descriptor["path"] for descriptor in descriptors},
            set(self.fixture.relative_paths.values()),
        )

        checker.assert_called_once()
        checker_manifest, checker_root = checker.call_args.args
        self.assertNotEqual(checker_manifest, output)
        checker_fd = checker.call_args.kwargs["manifest_fd"]
        self.assertIsInstance(checker_fd, int)
        self.assertIn(str(checker_fd), checker_manifest.parts)
        self.assertTrue(
            str(checker_manifest).startswith(("/proc/self/fd/", "/dev/fd/"))
        )
        self.assertEqual(checker_root, self.fixture.root.resolve())
        checker_anchors = dict(checker.call_args.kwargs)
        checker_anchors.pop("manifest_fd")
        self.assertEqual(checker_anchors, self.fixture.anchors)
        self.assertEqual(result.published_path, output)

    def test_same_inputs_produce_byte_identical_manifests(self) -> None:
        first = self.base / "candidate-a.json"
        second = self.base / "candidate-b.json"
        self.fixture.write(first)
        self.fixture.write(second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_existing_regular_output_is_preserved(self) -> None:
        output = self.base / "candidate.json"
        output.write_bytes(b"do not replace\n")
        with mock.patch.object(writer.candidate_checker, "evaluate") as checker:
            with self.assertRaisesRegex(
                writer.ManifestWriterError,
                "refusing to replace an existing path",
            ):
                writer.write_manifest(
                    self.fixture.root,
                    output,
                    artifact_paths=self.fixture.relative_paths,
                    **self.fixture.anchors,
                )
        self.assertEqual(output.read_bytes(), b"do not replace\n")
        checker.assert_not_called()

    def test_existing_symlink_output_is_preserved(self) -> None:
        protected = self.base / "protected.json"
        protected.write_bytes(b"protected\n")
        output = self.base / "candidate.json"
        output.symlink_to(protected)
        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "refusing to replace an existing path",
        ):
            self.fixture.write(output)
        self.assertTrue(output.is_symlink())
        self.assertEqual(protected.read_bytes(), b"protected\n")

    def test_repeated_rejected_self_checks_leave_no_staging_residue(self) -> None:
        output = self.base / "candidate.json"

        def reject(*_args: object, **_kwargs: object) -> dict[str, Any]:
            return {
                "schema_version": writer.REPORT_VERSION,
                "status": "failed",
                "passed": False,
                "candidate_id": CANDIDATE_ID,
                "manifest_sha256": "0" * 64,
                "bindings": {},
                "checks": [],
                "errors": ["fixture rejection"],
            }

        for _ in range(3):
            with self.assertRaisesRegex(
                writer.ManifestWriterError, "fixture rejection"
            ):
                self.fixture.write(output, evaluator=reject)
            self.assertFalse(output.exists())
            self.assert_no_staging_files(output)

    def test_passed_report_with_wrong_binding_is_rejected(self) -> None:
        output = self.base / "candidate.json"

        def wrong_binding(*args: object, **kwargs: str) -> dict[str, Any]:
            report = self.fixture.passed_report(*args, **kwargs)
            report["bindings"]["correctness_golden_sha256"] = _digest(b"other")
            return report

        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "does not equal the writer input",
        ):
            self.fixture.write(output, evaluator=wrong_binding)
        self.assertFalse(output.exists())
        self.assert_no_staging_files(output)

    def test_passed_report_cannot_omit_fixed37_production_batch_check(self) -> None:
        output = self.base / "candidate.json"

        def omit_fixed37(*args: object, **kwargs: str) -> dict[str, Any]:
            report = self.fixture.passed_report(*args, **kwargs)
            report["checks"] = [
                check
                for check in report["checks"]
                if check["name"] != "fixed37_production_batch_e0"
            ]
            return report

        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "did not pass every closed gate",
        ):
            self.fixture.write(output, evaluator=omit_fixed37)
        self.assertFalse(output.exists())
        self.assert_no_staging_files(output)

    def test_source_and_golden_must_match_external_anchors(self) -> None:
        for anchor in (
            "expected_source_archive_sha256",
            "expected_correctness_golden_sha256",
        ):
            with self.subTest(anchor=anchor):
                anchors = self.fixture.anchors
                anchors[anchor] = _digest(f"wrong {anchor}".encode("ascii"))
                output = self.base / f"{anchor}.json"
                with mock.patch.object(
                    writer.candidate_checker, "evaluate"
                ) as checker:
                    with self.assertRaises(writer.ManifestWriterError):
                        writer.write_manifest(
                            self.fixture.root,
                            output,
                            artifact_paths=self.fixture.relative_paths,
                            **anchors,
                        )
                checker.assert_not_called()
                self.assertFalse(output.exists())

    def test_candidate_base_version_must_match_verified_bundle(self) -> None:
        anchors = self.fixture.anchors
        anchors["expected_candidate_id"] = "rustinfer-0.2.0-rc1"
        output = self.base / "candidate.json"
        with mock.patch.object(writer.candidate_checker, "evaluate") as checker:
            with self.assertRaisesRegex(
                writer.ManifestWriterError,
                "release version differs from the verified bundle",
            ):
                writer.write_manifest(
                    self.fixture.root,
                    output,
                    artifact_paths=self.fixture.relative_paths,
                    **anchors,
                )
        checker.assert_not_called()
        self.assertFalse(output.exists())

    def test_artifact_paths_are_closed_and_strictly_relative(self) -> None:
        invalid_paths = (
            "/absolute",
            "../traversal",
            "artifacts//alias",
            "artifacts/./alias",
            "artifacts\\windows-alias",
        )
        for ordinal, invalid in enumerate(invalid_paths):
            with self.subTest(path=invalid):
                paths = dict(self.fixture.relative_paths)
                paths["source_archive"] = invalid
                output = self.base / f"invalid-{ordinal}.json"
                with self.assertRaisesRegex(
                    writer.ManifestWriterError,
                    "POSIX relative path|normalization aliases",
                ):
                    self.fixture.write(output, artifact_paths=paths)
                self.assertFalse(output.exists())

        missing = dict(self.fixture.relative_paths)
        missing.pop("performance_report")
        with self.assertRaisesRegex(writer.ManifestWriterError, "missing="):
            self.fixture.write(self.base / "missing.json", artifact_paths=missing)
        extra = dict(self.fixture.relative_paths)
        extra["undocumented"] = "artifacts/undocumented"
        with self.assertRaisesRegex(writer.ManifestWriterError, "unexpected="):
            self.fixture.write(self.base / "extra.json", artifact_paths=extra)

    def test_lexical_and_inode_aliases_are_rejected(self) -> None:
        lexical = dict(self.fixture.relative_paths)
        lexical["performance_raw_evidence"] = lexical["performance_report"]
        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "duplicates another artifact path",
        ):
            self.fixture.write(self.base / "lexical.json", artifact_paths=lexical)

        aliased_path = self.fixture.paths["performance_raw_evidence"]
        aliased_path.unlink()
        os.link(self.fixture.paths["performance_report"], aliased_path)
        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "hard-link aliases artifact",
        ):
            self.fixture.write(self.base / "inode.json")
        self.assertFalse((self.base / "inode.json").exists())

    def test_symlink_leaf_and_component_are_rejected_without_following(self) -> None:
        leaf_paths = dict(self.fixture.relative_paths)
        leaf = self.fixture.paths["source_archive"]
        leaf.unlink()
        leaf.symlink_to(self.fixture.paths["performance_report"])
        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "cannot open without following links",
        ):
            self.fixture.write(self.base / "leaf.json", artifact_paths=leaf_paths)

        leaf.unlink()
        leaf.write_bytes(b"trusted source archive\n")
        linked_component = self.fixture.root / "linked-artifacts"
        linked_component.symlink_to(self.fixture.root / "artifacts", target_is_directory=True)
        component_paths = dict(self.fixture.relative_paths)
        component_paths["source_archive"] = (
            "linked-artifacts/" + Path(leaf_paths["source_archive"]).name
        )
        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "cannot open without following links",
        ):
            self.fixture.write(
                self.base / "component.json",
                artifact_paths=component_paths,
            )

    def test_post_check_path_replacement_is_detected(self) -> None:
        output = self.base / "candidate.json"
        target = self.fixture.paths["performance_report"]

        def replace_after_check(
            manifest_path: Path,
            evidence_root: Path,
            **anchors: str,
        ) -> dict[str, Any]:
            report = self.fixture.passed_report(
                manifest_path,
                evidence_root,
                **anchors,
            )
            target.unlink()
            target.write_bytes(b"replacement after self-check\n")
            return report

        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "path was replaced during the final self-check",
        ):
            self.fixture.write(output, evaluator=replace_after_check)
        self.assertFalse(output.exists())
        self.assert_no_staging_files(output)

    def test_post_check_staged_manifest_mutation_is_detected(self) -> None:
        output = self.base / "candidate.json"

        def mutate_staged_manifest(
            manifest_path: Path,
            evidence_root: Path,
            **anchors: object,
        ) -> dict[str, Any]:
            report = self.fixture.passed_report(
                manifest_path,
                evidence_root,
                **anchors,
            )
            manifest_fd = anchors["manifest_fd"]
            assert isinstance(manifest_fd, int)
            os.ftruncate(manifest_fd, 0)
            os.pwrite(manifest_fd, b"{}\n", 0)
            return report

        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "staged manifest: changed during the final self-check",
        ):
            self.fixture.write(output, evaluator=mutate_staged_manifest)
        self.assertFalse(output.exists())
        self.assert_no_staging_files(output)

    def test_unrelated_staging_name_cannot_replace_anonymous_source(self) -> None:
        output = self.base / "candidate.json"
        replacement = b"unrelated staging replacement\n"
        fake_staging = output.parent / f".{output.name}.tmp-attacker"

        def create_unrelated_staging_name(
            manifest_path: Path,
            evidence_root: Path,
            **anchors: object,
        ) -> dict[str, Any]:
            report = self.fixture.passed_report(
                manifest_path,
                evidence_root,
                **anchors,
            )
            fake_staging.write_bytes(replacement)
            return report

        result, _ = self.fixture.write(
            output, evaluator=create_unrelated_staging_name
        )
        self.assertEqual(output.read_bytes(), canonical_json_bytes(result.manifest))
        self.assertNotEqual(output.read_bytes(), replacement)
        self.assertEqual(fake_staging.read_bytes(), replacement)

    def test_output_must_be_outside_evidence_root(self) -> None:
        output = self.fixture.root / "candidate.json"
        with mock.patch.object(writer.candidate_checker, "evaluate") as checker:
            with self.assertRaisesRegex(
                writer.ManifestWriterError,
                "outside the read-only evidence root",
            ):
                writer.write_manifest(
                    self.fixture.root,
                    output,
                    artifact_paths=self.fixture.relative_paths,
                    **self.fixture.anchors,
                )
        checker.assert_not_called()
        self.assertFalse(output.exists())

    def test_destination_created_after_preflight_is_never_replaced(self) -> None:
        output = self.base / "candidate.json"
        sentinel = b"concurrent destination\n"

        def create_destination(
            manifest_path: Path,
            evidence_root: Path,
            **anchors: str,
        ) -> dict[str, Any]:
            report = self.fixture.passed_report(
                manifest_path,
                evidence_root,
                **anchors,
            )
            output.write_bytes(sentinel)
            return report

        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "refusing to replace an existing path",
        ):
            self.fixture.write(output, evaluator=create_destination)
        self.assertEqual(output.read_bytes(), sentinel)
        self.assert_no_staging_files(output)

    def test_lexical_symlink_in_output_parent_is_rejected_before_check(self) -> None:
        physical_parent = self.base / "physical-parent"
        physical_parent.mkdir()
        linked_parent = self.base / "linked-parent"
        linked_parent.symlink_to(physical_parent, target_is_directory=True)
        output = linked_parent / "candidate.json"

        with mock.patch.object(writer.candidate_checker, "evaluate") as checker:
            with self.assertRaisesRegex(
                writer.ManifestWriterError,
                "cannot traverse lexical parent without following symlinks",
            ):
                writer.write_manifest(
                    self.fixture.root,
                    output,
                    artifact_paths=self.fixture.relative_paths,
                    **self.fixture.anchors,
                )
        checker.assert_not_called()
        self.assertFalse((physical_parent / output.name).exists())

    def test_output_parent_ancestor_rebind_after_check_is_rejected(self) -> None:
        original_ancestor = self.base / "publish-tree"
        original_parent = original_ancestor / "nested"
        original_parent.mkdir(parents=True)
        moved_ancestor = self.base / "moved-publish-tree"
        output = original_parent / "candidate.json"

        def rebind_parent_path(
            manifest_path: Path,
            evidence_root: Path,
            **anchors: object,
        ) -> dict[str, Any]:
            report = self.fixture.passed_report(
                manifest_path,
                evidence_root,
                **anchors,
            )
            original_ancestor.rename(moved_ancestor)
            original_parent.mkdir(parents=True)
            return report

        with self.assertRaisesRegex(
            writer.ManifestWriterError,
            "lexical parent path changed during the final self-check",
        ):
            self.fixture.write(output, evaluator=rebind_parent_path)
        self.assertFalse(output.exists())
        self.assertFalse((moved_ancestor / "nested" / output.name).exists())
        self.assert_no_staging_files(output)

    def test_output_name_replacement_during_final_validation_is_preserved(
        self,
    ) -> None:
        output = self.base / "candidate.json"
        replacement = b"concurrent output replacement\n"
        real_revalidate = writer._revalidate_published_output

        def replace_output(
            binding: writer.OutputParentBinding,
            parent_fd: int,
            published_fd: int,
            expected_sha256: str,
        ) -> None:
            output.unlink()
            output.write_bytes(replacement)
            real_revalidate(
                binding,
                parent_fd,
                published_fd,
                expected_sha256,
            )

        with mock.patch.object(
            writer,
            "_revalidate_published_output",
            side_effect=replace_output,
        ):
            with self.assertRaisesRegex(
                writer.ManifestWriterError,
                "published path does not name the held manifest",
            ):
                self.fixture.write(output)
        self.assertEqual(output.read_bytes(), replacement)
        self.assert_no_staging_files(output)

    def test_direct_copy_failure_preserves_partial_create_only_residue(self) -> None:
        output = self.base / "candidate.json"
        partial = b"partial manifest publication\n"
        real_create = writer._create_staged_manifest

        def force_direct_publication(parent_fd: int) -> writer.StagedManifest:
            staged = real_create(parent_fd)
            return writer.StagedManifest(
                file_fd=staged.file_fd,
                checker_path=staged.checker_path,
                linkable_tmpfile=False,
            )

        def fail_copy(_source_fd: int, destination_fd: int) -> None:
            os.write(destination_fd, partial)
            raise OSError("fixture copy failure")

        with mock.patch.object(
            writer,
            "_create_staged_manifest",
            side_effect=force_direct_publication,
        ), mock.patch.object(writer, "_copy_fd", side_effect=fail_copy):
            with self.assertRaisesRegex(
                writer.ManifestWriterError,
                "fixture copy failure",
            ):
                self.fixture.write(output)
        self.assertEqual(output.read_bytes(), partial)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        self.assert_no_staging_files(output)

    def test_linux_link_publication_falls_back_to_proc_bound_fd(self) -> None:
        class FakeLinkat:
            argtypes: object = None
            restype: object = None

            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def __call__(self, *arguments: object) -> int:
                self.calls.append(arguments)
                if len(self.calls) == 1:
                    writer.ctypes.set_errno(writer.errno.ENOENT)
                    return -1
                return 0

        fake_linkat = FakeLinkat()
        fake_libc = mock.Mock(linkat=fake_linkat)
        with mock.patch.object(writer.sys, "platform", "linux"), mock.patch.object(
            writer.ctypes,
            "CDLL",
            return_value=fake_libc,
        ):
            writer._link_tmpfile_noreplace(41, "candidate.json", 42)

        self.assertEqual(len(fake_linkat.calls), 2)
        self.assertEqual(fake_linkat.calls[0][0:2], (41, b""))
        self.assertEqual(
            fake_linkat.calls[0][-1], writer.LINUX_AT_EMPTY_PATH
        )
        self.assertEqual(
            fake_linkat.calls[1][0:2],
            (writer.LINUX_AT_FDCWD, b"/proc/self/fd/41"),
        )
        self.assertEqual(
            fake_linkat.calls[1][-1], writer.LINUX_AT_SYMLINK_FOLLOW
        )

    def test_directory_fsync_failure_preserves_published_residue(self) -> None:
        output = self.base / "candidate.json"
        real_fsync = os.fsync
        failed_directory_sync = False

        def fail_first_directory_sync(file_fd: int) -> None:
            nonlocal failed_directory_sync
            if stat.S_ISDIR(os.fstat(file_fd).st_mode) and not failed_directory_sync:
                failed_directory_sync = True
                raise OSError("fixture directory fsync failure")
            real_fsync(file_fd)

        with mock.patch.object(writer.os, "fsync", side_effect=fail_first_directory_sync):
            with self.assertRaisesRegex(OSError, "fixture directory fsync failure"):
                self.fixture.write(output)
        self.assertTrue(failed_directory_sync)
        self.assertTrue(output.is_file())
        raw = output.read_bytes()
        self.assertEqual(raw, canonical_json_bytes(json.loads(raw)))
        self.assert_no_staging_files(output)

    def test_actual_candidate_evaluate_passes_closed_final_check_inventory(
        self,
    ) -> None:
        # Import lazily because this fixture assembles all replayable gate
        # evidence at module load time.
        from test_release_candidate import CandidateFixture

        evidence_root = self.base / "actual-candidate-evidence"
        evidence_root.mkdir()
        candidate = CandidateFixture(evidence_root)
        path_keys = {
            "source_archive": "source",
            "release_binary": "binary",
            "release_bundle": "bundle",
            "python_free_e2e_report": "python_report",
            "python_free_e2e_raw_evidence": "python_raw",
            "python_free_e2e_correctness_golden": "correctness_golden",
            "cuda_fault_report": "cuda_report",
            "cuda_fault_raw_evidence": "cuda_raw",
            "native_correctness_report": "native_correctness",
            "native_correctness_raw_replay": "native_replay",
            "native_correctness_candidate_executable": "native_executable",
            "reproducible_build_a": "repro_build_a",
            "reproducible_build_b": "repro_build_b",
            "reproducible_profile_binary": "profile_binary",
            "reproducible_native_manifest": "native_manifest",
            "optimization_correctness_report": "optimization_correctness",
            "optimization_correctness_raw_evidence": "optimization_raw",
            "performance_report": "performance",
            "performance_raw_evidence": "performance_raw",
            "reliability_soak_report": "soak",
            "reliability_soak_raw_evidence": "soak_raw",
        }
        artifact_paths = {
            manifest_key: candidate.paths[fixture_key]
            .relative_to(evidence_root)
            .as_posix()
            for manifest_key, fixture_key in path_keys.items()
        }
        anchors = {
            "expected_candidate_id": CANDIDATE_ID,
            "expected_revision": candidate.revision,
            "expected_source_archive_sha256": candidate.trusted_source_sha256,
            "expected_release_image_id": f"sha256:{candidate.image_sha}",
            "expected_reproducible_build_image_id": (
                candidate.reproducible_build_image_id
            ),
            "expected_cuda_build_image_id": candidate.cuda_build_image_id,
            "expected_optimization_build_image_id": (
                candidate.optimization_build_image_id
            ),
            "expected_correctness_golden_sha256": (
                candidate.correctness_golden_sha256
            ),
        }

        def actual_entrypoint(
            manifest_path: Path,
            checker_root: Path,
            **received: object,
        ) -> dict[str, object]:
            self.assertEqual(checker_root, evidence_root.resolve())
            manifest_fd = received.pop("manifest_fd")
            assert isinstance(manifest_fd, int)
            return candidate.evaluate(
                manifest_path=manifest_path,
                manifest_fd=manifest_fd,
                **received,
            )

        output = self.base / "actual-candidate.json"
        with mock.patch.object(
            writer.candidate_checker,
            "evaluate",
            side_effect=actual_entrypoint,
        ) as checker:
            result = writer.write_manifest(
                evidence_root,
                output,
                artifact_paths=artifact_paths,
                **anchors,
            )

        checker.assert_called_once()
        self.assertTrue(result.self_check_report["passed"])
        self.assertEqual(
            [check["name"] for check in result.self_check_report["checks"]],
            list(writer.FINAL_CHECK_NAMES),
        )
        self.assertEqual(output.read_bytes(), canonical_json_bytes(result.manifest))
        self.assert_no_staging_files(output)

    def test_cli_has_eight_anchors_and_twenty_one_explicit_artifacts(self) -> None:
        parser = writer._parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        artifact_options = {spec.option for spec in writer.ARTIFACT_SPECS}
        anchor_options = {
            "--expected-candidate-id",
            "--expected-revision",
            "--expected-source-archive-sha256",
            "--expected-release-image-id",
            "--expected-reproducible-build-image-id",
            "--expected-cuda-build-image-id",
            "--expected-optimization-build-image-id",
            "--expected-correctness-golden-sha256",
        }
        self.assertEqual(len(artifact_options), 21)
        self.assertTrue(artifact_options <= options)
        self.assertEqual(len(anchor_options), 8)
        self.assertTrue(anchor_options <= options)
        self.assertNotIn("--source-date-epoch", options)
        self.assertNotIn("--final-report", options)
        self.assertNotIn("--reproducibility-report", options)


if __name__ == "__main__":
    unittest.main()
