#!/usr/bin/env python3
"""Focused hostile-input tests for binary-bound C02-P1 baseline v2 evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import check_reconstructed_prior_baseline_v2 as checker
import provenance_v2_common as common


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


class BaselineV2Fixture:
    def __init__(
        self,
        root: Path,
        *,
        target_commit_sha1: str = "b" * 40,
        tag_object_sha1: str = "a" * 40,
        target_output_commit_sha1: str | None = None,
        recipe_binary_mismatch: bool = False,
        recipe_bundle_mismatch: bool = False,
        binary_b_mismatch: bool = False,
        image_b_mismatch: bool = False,
        raw_docker_b_mismatch: bool = False,
        raw_docker_duplicate_id: bool = False,
        oci_archive_b_mismatch: bool = False,
        image_raw_descriptor_mismatch: bool = False,
        image_archive_descriptor_mismatch: bool = False,
        binary_aliases_bundle: bool = False,
        share_a_b_binary: bool = False,
        share_a_b_bundle: bool = False,
        duplicate_recipe_path: bool = False,
    ) -> None:
        self.root = root
        self.oci_archive_b_mismatch = oci_archive_b_mismatch
        self.baseline_id = "reconstructed-riley-0.1.0-rc2"
        self.tag_object_sha1 = tag_object_sha1
        self.target_commit_sha1 = target_commit_sha1
        self.tag_name = "riley-0.1.0-rc2"
        self.tag_ref = f"refs/tags/{self.tag_name}"
        self.tag_object = self.write_json(
            "source/git-tag-object.json",
            {
                "schema_version": checker.GIT_TAG_OBJECT_VERSION,
                "tag_ref": self.tag_ref,
                "object_type": "tag",
                "object_sha1": self.tag_object_sha1,
                "target_object_type": "commit",
                "target_object_sha1": self.target_commit_sha1,
            },
        )
        self.tag_target = self.write_json(
            "source/git-tag-target.json",
            {
                "schema_version": checker.GIT_TAG_TARGET_VERSION,
                "tag_ref": self.tag_ref,
                "tag_object_sha1": self.tag_object_sha1,
                "target_commit_sha1": target_output_commit_sha1 or self.target_commit_sha1,
            },
        )
        self.archive = self.write_file("source/riley-0.1.0-rc2.tar.zst", b"prior tagged source")
        self.source = {
            "tag_name": self.tag_name,
            "tag_object": self.tag_object,
            "tag_target": self.tag_target,
            "archive": self.archive,
        }
        self.a_image_id = "sha256:" + digest("reconstructed OCI image")
        self.b_image_id = (
            "sha256:" + digest("other OCI image")
            if image_b_mismatch
            else self.a_image_id
        )
        self.a_artifacts = self.artifacts("a", binary_b_mismatch=False)
        self.b_artifacts = self.artifacts("b", binary_b_mismatch=binary_b_mismatch)
        if binary_aliases_bundle:
            self.a_artifacts["binary"] = copy.deepcopy(self.a_artifacts["bundle"])
            self.b_artifacts["binary"] = copy.deepcopy(self.b_artifacts["bundle"])
        if share_a_b_binary:
            self.b_artifacts["binary"] = copy.deepcopy(self.a_artifacts["binary"])
        if share_a_b_bundle:
            self.b_artifacts["bundle"] = copy.deepcopy(self.a_artifacts["bundle"])
        self.a_recipe = self.write_file("reproductions/a/build-recipe.txt", b"recipe a")
        self.b_recipe = self.write_file("reproductions/b/build-recipe.txt", b"recipe b")
        if duplicate_recipe_path:
            self.a_recipe = copy.deepcopy(self.a_artifacts["bundle"])
        self.a_runtime_image_inspect_raw = self.write_raw_docker_inspect(
            "a", self.a_image_id
        )
        raw_b_image_id = (
            "sha256:" + digest("raw Docker inspect disagrees")
            if raw_docker_b_mismatch
            else self.b_image_id
        )
        self.b_runtime_image_inspect_raw = self.write_raw_docker_inspect(
            "b", raw_b_image_id, duplicate_id=raw_docker_duplicate_id
        )
        self.a_recipe_inspect = self.write_json(
            "reproductions/a/recipe-inspect.json",
            self.recipe_inspect_document(
                "a",
                self.a_recipe,
                self.archive if recipe_binary_mismatch else self.a_artifacts["binary"],
                self.archive if recipe_bundle_mismatch else self.a_artifacts["bundle"],
            ),
        )
        self.b_recipe_inspect = self.write_json(
            "reproductions/b/recipe-inspect.json",
            self.recipe_inspect_document(
                "b",
                self.b_recipe,
                self.b_artifacts["binary"],
                self.b_artifacts["bundle"],
            ),
        )
        self.a_image_inspect = self.write_json(
            "reproductions/a/image-inspect.json",
            self.image_inspect_document(
                "a",
                self.a_artifacts,
                self.b_runtime_image_inspect_raw
                if image_raw_descriptor_mismatch
                else self.a_runtime_image_inspect_raw,
                self.a_image_id,
                self.archive if image_archive_descriptor_mismatch else None,
            ),
        )
        self.b_image_inspect = self.write_json(
            "reproductions/b/image-inspect.json",
            self.image_inspect_document(
                "b", self.b_artifacts, self.b_runtime_image_inspect_raw, self.b_image_id
            ),
        )
        self.a_receipt_path = "reproductions/a/build-receipt.json"
        self.b_receipt_path = "reproductions/b/build-receipt.json"
        self.a_receipt = self.build_receipt(
            "a",
            self.a_recipe_inspect,
            self.a_image_inspect,
            self.a_runtime_image_inspect_raw,
            self.a_artifacts,
        )
        self.b_receipt = self.build_receipt(
            "b",
            self.b_recipe_inspect,
            self.b_image_inspect,
            self.b_runtime_image_inspect_raw,
            self.b_artifacts,
        )
        self.a_receipt_descriptor = self.write_json(self.a_receipt_path, self.a_receipt)
        self.b_receipt_descriptor = self.write_json(self.b_receipt_path, self.b_receipt)
        self.manifest = self.manifest_document()
        self.write_json("baseline.json", self.manifest)

    def write_file(self, relative: str, contents: bytes) -> dict[str, Any]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return common.descriptor_for_bytes(relative, contents, relative).as_json()

    def write_json(self, relative: str, document: object) -> dict[str, Any]:
        return self.write_file(relative, common.canonical_json_bytes(document))

    def artifacts(self, reconstruction: str, *, binary_b_mismatch: bool) -> dict[str, Any]:
        return {
            "binary": self.write_file(
                f"reproductions/{reconstruction}/riley-server",
                b"different reconstructed server binary"
                if binary_b_mismatch and reconstruction == "b"
                else b"reconstructed server binary",
            ),
            "bundle": self.write_file(
                f"reproductions/{reconstruction}/riley.bundle.tar.zst", b"reconstructed bundle"
            ),
            "oci": {
                "archive": self.write_file(
                    f"reproductions/{reconstruction}/oci-image.tar",
                    b"different OCI archive"
                    if self.oci_archive_b_mismatch and reconstruction == "b"
                    else b"OCI archive",
                ),
                "layout": self.write_file(
                    f"reproductions/{reconstruction}/oci-layout.tar", b"OCI layout"
                ),
                "manifest": self.write_file(
                    f"reproductions/{reconstruction}/oci-manifest.json", b"OCI manifest"
                ),
            },
        }

    def recipe_inspect_document(
        self,
        reconstruction: str,
        recipe: dict[str, Any],
        binary: dict[str, Any],
        bundle: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": checker.RECIPE_INSPECT_VERSION,
            "baseline_id": self.baseline_id,
            "reconstruction_id": reconstruction,
            "source": copy.deepcopy(self.source),
            "recipe": copy.deepcopy(recipe),
            "binary": copy.deepcopy(binary),
            "bundle": copy.deepcopy(bundle),
        }

    def write_raw_docker_inspect(
        self, reconstruction: str, image_id: str, *, duplicate_id: bool = False
    ) -> dict[str, Any]:
        if duplicate_id:
            raw = (
                b'[\n  {"Id":"'
                + image_id.encode("ascii")
                + b'","Id":"'
                + image_id.encode("ascii")
                + b'"}\n]\n'
            )
        else:
            raw = (
                b'[\n  {\n    "Id": "'
                + image_id.encode("ascii")
                + b'",\n    "RepoTags": ["fixture/riley:reconstructed"]\n  }\n]\n'
            )
        return self.write_file(f"reproductions/{reconstruction}/docker-image-inspect.json", raw)

    def image_inspect_document(
        self,
        reconstruction: str,
        artifacts: dict[str, Any],
        runtime_image_inspect_raw: dict[str, Any],
        image_id: str,
        archive_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": checker.IMAGE_INSPECT_VERSION,
            "baseline_id": self.baseline_id,
            "reconstruction_id": reconstruction,
            "source": copy.deepcopy(self.source),
            "runtime_image_inspect_raw": copy.deepcopy(runtime_image_inspect_raw),
            "oci_archive": copy.deepcopy(archive_override or artifacts["oci"]["archive"]),
            "oci_layout": copy.deepcopy(artifacts["oci"]["layout"]),
            "oci_manifest": copy.deepcopy(artifacts["oci"]["manifest"]),
            "image_id": image_id,
        }

    def build_receipt(
        self,
        reconstruction: str,
        recipe_inspect: dict[str, Any],
        image_inspect: dict[str, Any],
        runtime_image_inspect_raw: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": checker.BUILD_RECEIPT_VERSION,
            "baseline_id": self.baseline_id,
            "reconstruction_id": reconstruction,
            "source": copy.deepcopy(self.source),
            "recipe_inspect": copy.deepcopy(recipe_inspect),
            "image_inspect": copy.deepcopy(image_inspect),
            "runtime_image_inspect_raw": copy.deepcopy(runtime_image_inspect_raw),
            "artifacts": copy.deepcopy(artifacts),
        }

    @staticmethod
    def equality_descriptor(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return {"a": copy.deepcopy(a), "b": copy.deepcopy(b), "sha256": a["sha256"]}

    def manifest_document(self) -> dict[str, Any]:
        return {
            "schema_version": checker.MANIFEST_VERSION,
            "baseline_id": self.baseline_id,
            "baseline_kind": checker.BASELINE_KIND,
            "provenance_class": checker.PROVENANCE_CLASS,
            "historical_distribution": checker.HISTORICAL_DISTRIBUTION,
            "historical_stable_artifact_status": checker.HISTORICAL_STABLE_ARTIFACT_STATUS,
            "was_previously_shipped": checker.WAS_PREVIOUSLY_SHIPPED,
            "source": copy.deepcopy(self.source),
            "reproductions": {
                "a": copy.deepcopy(self.a_receipt_descriptor),
                "b": copy.deepcopy(self.b_receipt_descriptor),
            },
            "equality": {
                "binary": self.equality_descriptor(
                    self.a_artifacts["binary"], self.b_artifacts["binary"]
                ),
                "bundle": self.equality_descriptor(
                    self.a_artifacts["bundle"], self.b_artifacts["bundle"]
                ),
                "oci_archive": self.equality_descriptor(
                    self.a_artifacts["oci"]["archive"], self.b_artifacts["oci"]["archive"]
                ),
                "oci_layout": self.equality_descriptor(
                    self.a_artifacts["oci"]["layout"], self.b_artifacts["oci"]["layout"]
                ),
                "oci_manifest": self.equality_descriptor(
                    self.a_artifacts["oci"]["manifest"], self.b_artifacts["oci"]["manifest"]
                ),
                "oci_image": {
                    "a": self.a_image_id,
                    "b": self.b_image_id,
                    "image_id": self.a_image_id,
                },
            },
        }

    def rewrite_manifest(self, document: dict[str, Any]) -> None:
        self.manifest = document
        self.write_json("baseline.json", document)


class ReconstructedPriorBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name) / "evidence"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        self.root = root.resolve(strict=True)
        self.fixture = BaselineV2Fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_failure(self, code: str, *, root: Path | None = None) -> None:
        with self.assertRaises(common.ProvenanceV2Error) as raised:
            checker.validate_file(root or self.root, "baseline.json")
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_accepts_raw_git_recipe_and_oci_inspect_bound_a_b_reconstruction(self) -> None:
        report = checker.validate_file(self.root, "baseline.json")
        self.assertTrue(report["passed"])
        self.assertEqual(report["baseline_kind"], "reconstructed-tag-baseline")
        self.assertEqual(report["provenance_class"], "reconstructed-from-source")
        self.assertEqual(report["historical_distribution"], "not-attested")
        self.assertEqual(report["historical_stable_artifact_status"], "unavailable")
        self.assertFalse(report["was_previously_shipped"])
        self.assertEqual(report["source_archive_content_binding"], "not-validated")
        self.assertEqual(report["oci_archive_content_binding"], "not-validated")
        self.assertEqual(report["git_identity"]["target_commit_sha1"], "b" * 40)
        self.assertIn("binary", report["reproductions"]["a"]["artifacts"])
        self.assertIn("archive", report["reproductions"]["a"]["artifacts"]["oci"])
        self.assertIn("runtime_image_inspect_raw", report["reproductions"]["a"])
        self.assertEqual([row["name"] for row in report["checks"]], list(checker.CHECK_NAMES))

    def test_descriptor_preflight_sees_exact_closure_before_raw_streaming(self) -> None:
        root_fd = common.open_private_evidence_directory(self.root, "fixture evidence")
        seen: list[tuple[common.EvidenceDescriptor, ...]] = []

        class StopBeforeRawStreaming(Exception):
            pass

        def preflight(descriptors: tuple[common.EvidenceDescriptor, ...]) -> None:
            seen.append(descriptors)
            raise StopBeforeRawStreaming()

        try:
            with mock.patch.object(
                checker,
                "_verify_raw_descriptor",
                side_effect=AssertionError("preflight must precede raw streaming"),
            ), self.assertRaises(StopBeforeRawStreaming):
                checker.evaluate(
                    root_fd,
                    self.fixture.manifest,
                    descriptor_preflight=preflight,
                )
        finally:
            os.close(root_fd)
        self.assertEqual(len(seen), 1)
        descriptors = seen[0]
        self.assertEqual(len(descriptors), 23)
        self.assertEqual(len({descriptor.path for descriptor in descriptors}), 23)
        self.assertEqual(
            {descriptor.path for descriptor in descriptors},
            {
                "source/git-tag-object.json",
                "source/git-tag-target.json",
                "source/riley-0.1.0-rc2.tar.zst",
                "reproductions/a/build-receipt.json",
                "reproductions/b/build-receipt.json",
                "reproductions/a/recipe-inspect.json",
                "reproductions/a/image-inspect.json",
                "reproductions/a/docker-image-inspect.json",
                "reproductions/a/build-recipe.txt",
                "reproductions/a/riley-server",
                "reproductions/a/riley.bundle.tar.zst",
                "reproductions/a/oci-image.tar",
                "reproductions/a/oci-layout.tar",
                "reproductions/a/oci-manifest.json",
                "reproductions/b/recipe-inspect.json",
                "reproductions/b/image-inspect.json",
                "reproductions/b/docker-image-inspect.json",
                "reproductions/b/build-recipe.txt",
                "reproductions/b/riley-server",
                "reproductions/b/riley.bundle.tar.zst",
                "reproductions/b/oci-image.tar",
                "reproductions/b/oci-layout.tar",
                "reproductions/b/oci-manifest.json",
            },
        )

    def test_rejects_historical_distribution_or_shipment_claim(self) -> None:
        changed = copy.deepcopy(self.fixture.manifest)
        changed["historical_distribution"] = "attested"
        self.fixture.rewrite_manifest(changed)
        self.assert_failure("historical-distribution-claim")

        changed = copy.deepcopy(self.fixture.manifest)
        changed["historical_distribution"] = checker.HISTORICAL_DISTRIBUTION
        changed["was_previously_shipped"] = True
        self.fixture.rewrite_manifest(changed)
        self.assert_failure("historical-shipped-claim")

    def test_checker_requires_private_evidence_root_and_rejects_symlink_alias(self) -> None:
        self.root.chmod(0o755)
        try:
            self.assert_failure("unsafe-evidence-root-mode")
        finally:
            self.root.chmod(0o700)

        with mock.patch.object(common.os, "geteuid", return_value=os.geteuid() + 1):
            self.assert_failure("unsafe-evidence-root-owner")

        link = self.root.parent / "evidence-link"
        link.symlink_to(self.root.name, target_is_directory=True)
        self.assert_failure("unsafe-evidence-directory", root=link)

    def test_rejects_raw_tag_target_that_disagrees_with_raw_tag_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = (Path(temporary) / "evidence")
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, target_output_commit_sha1="c" * 40)
            self.assert_failure("git-tag-object-target-mismatch", root=root)

    def test_rejects_recipe_inspect_that_does_not_bind_receipt_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, recipe_bundle_mismatch=True)
            self.assert_failure("descriptor-binding-mismatch", root=root)

    def test_rejects_recipe_inspect_that_does_not_bind_receipt_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, recipe_binary_mismatch=True)
            self.assert_failure("descriptor-binding-mismatch", root=root)

    def test_rejects_image_inspect_that_does_not_bind_same_oci_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, image_b_mismatch=True)
            self.assert_failure("a-b-equality-mismatch", root=root)

    def test_rejects_raw_docker_image_id_that_disagrees_with_canonical_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, raw_docker_b_mismatch=True)
            self.assert_failure("runtime-image-inspect-id-mismatch", root=root)

    def test_rejects_canonical_image_inspect_with_another_runtime_raw_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, image_raw_descriptor_mismatch=True)
            self.assert_failure("descriptor-binding-mismatch", root=root)

    def test_rejects_canonical_image_inspect_with_another_oci_archive_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, image_archive_descriptor_mismatch=True)
            self.assert_failure("descriptor-binding-mismatch", root=root)

    def test_rejects_duplicate_key_in_original_docker_inspect_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, raw_docker_duplicate_id=True)
            self.assert_failure("duplicate-json-key", root=root)

    def test_rejects_oci_archive_a_b_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, oci_archive_b_mismatch=True)
            self.assert_failure("a-b-equality-mismatch", root=root)

    def test_rejects_server_binary_a_b_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, binary_b_mismatch=True)
            self.assert_failure("a-b-equality-mismatch", root=root)

    def test_stream_verifies_opaque_oci_archive_leaf(self) -> None:
        archive = self.root / self.fixture.b_artifacts["oci"]["archive"]["path"]
        archive.write_bytes(b"BAD archive")
        self.assert_failure("evidence-hash-mismatch")

    def test_stream_verifies_server_binary_leaf(self) -> None:
        binary = self.root / self.fixture.b_artifacts["binary"]["path"]
        binary.write_bytes(b"x" * len(binary.read_bytes()))
        self.assert_failure("evidence-hash-mismatch")

    def test_rejects_shared_a_b_bundle_path_even_with_equal_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, share_a_b_bundle=True)
            self.assert_failure("non-independent-reconstruction", root=root)

    def test_rejects_shared_a_b_binary_path_even_with_equal_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, share_a_b_binary=True)
            self.assert_failure("non-independent-reconstruction", root=root)

    def test_rejects_binary_aliasing_bundle_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, binary_aliases_bundle=True)
            self.assert_failure("duplicate-evidence-path", root=root)

    def test_rejects_global_descriptor_path_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            root = root.resolve(strict=True)
            BaselineV2Fixture(root, duplicate_recipe_path=True)
            self.assert_failure("duplicate-evidence-path", root=root)

    def test_rejects_symlinked_bundle_via_strong_common_fd_reader(self) -> None:
        bundle = self.root / self.fixture.b_artifacts["bundle"]["path"]
        bundle.unlink()
        bundle.symlink_to("../a/riley.bundle.tar.zst")
        self.assert_failure("unsafe-evidence-path")

    def test_cli_create_only_report_uses_pinned_root_fd(self) -> None:
        result = checker.main(
            [
                "--evidence-root",
                str(self.root),
                "--manifest",
                "baseline.json",
                "--report-name",
                "baseline-check.json",
            ]
        )
        self.assertEqual(result, 0)
        report = common.parse_canonical_json(
            (self.root / "baseline-check.json").read_bytes(), "check report"
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            checker.main(
                [
                    "--evidence-root",
                    str(self.root),
                    "--manifest",
                    "baseline.json",
                    "--report-name",
                    "baseline-check.json",
                ]
            ),
            1,
        )

    def test_published_v2_schema_requires_binary_closure(self) -> None:
        repository_schema = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "reconstructed-prior-baseline-v2.schema.json"
        )
        schema_path = (
            repository_schema
            if repository_schema.is_file()
            else Path(__file__).with_name("reconstructed-prior-baseline-v2.schema.json")
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"],
            "https://riley.invalid/benchmarks/release/candidates/"
            "reconstructed-prior-baseline-v2.schema.json",
        )
        self.assertEqual(
            schema["$defs"]["baseline"]["properties"]["schema_version"],
            {"const": checker.MANIFEST_VERSION},
        )
        self.assertIn("binary", schema["$defs"]["artifacts"]["required"])
        self.assertIn("binary", schema["$defs"]["recipeInspect"]["required"])
        self.assertIn("binary", schema["$defs"]["equality"]["required"])
        self.assertEqual(
            schema["$defs"]["checkReport"]["properties"]["schema_version"],
            {"const": checker.CHECK_REPORT_VERSION},
        )
        self.assertEqual(schema["$defs"]["checkReport"]["properties"]["checks"]["minItems"], 17)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
