#!/usr/bin/env python3
"""CPU-only hostile-input tests for the C02-P1 cross-root content bridge."""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
import stat
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


import check_reconstructed_prior_baseline_v2 as baseline
import prepare_reconstructed_prior_baseline_content_bridge_v1 as bridge
import prepare_reconstructed_rc2_inputs_v1 as source_inputs
import prepare_reconstructed_runtime_oci_inputs_v1 as runtime_oci
import provenance_v2_common as common


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _oci_digest(raw: bytes) -> str:
    return "sha256:" + _sha256(raw)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class BridgeFixture:
    """Build independently replayable source/OCI roots and a v2 projection."""

    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.source_root, self.source_sha256 = self._make_source_root("source-inputs", b"reviewed RC2 source\n")
        self.oci_a_root, self.oci_a_receipt = self._make_oci_root("runtime-oci-a", "a", b"same layer\n")
        self.oci_b_root, self.oci_b_receipt = self._make_oci_root("runtime-oci-b", "b", b"same layer\n")
        self.baseline_root = self.base / "baseline"
        self._make_baseline_root()

    def close(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _private_directory(path: Path) -> None:
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)

    @staticmethod
    def _write(path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for parent in (path.parent,):
            os.chmod(parent, 0o700)
        path.write_bytes(raw)
        os.chmod(path, 0o600)

    def _write_root_file(self, root: Path, relative: str, raw: bytes) -> dict[str, Any]:
        self._write(root / relative, raw)
        return common.descriptor_for_bytes(relative, raw, relative).as_json()

    def _make_source_root(self, name: str, archive: bytes) -> tuple[Path, str]:
        root = self.base / name
        self._private_directory(root)
        source = root / source_inputs.SOURCE_DIRECTORY_NAME
        self._private_directory(source)
        tag_object_document = {
            "schema_version": baseline.GIT_TAG_OBJECT_VERSION,
            "tag_ref": source_inputs.RECONSTRUCTED_RC2_TAG_REF,
            "object_type": "tag",
            "object_sha1": source_inputs.RECONSTRUCTED_RC2_TAG_OBJECT,
            "target_object_type": "commit",
            "target_object_sha1": source_inputs.RECONSTRUCTED_RC2_TARGET,
        }
        tag_target_document = {
            "schema_version": baseline.GIT_TAG_TARGET_VERSION,
            "tag_ref": source_inputs.RECONSTRUCTED_RC2_TAG_REF,
            "tag_object_sha1": source_inputs.RECONSTRUCTED_RC2_TAG_OBJECT,
            "target_commit_sha1": source_inputs.RECONSTRUCTED_RC2_TARGET,
        }
        tag_object_raw = common.canonical_json_bytes(tag_object_document)
        tag_target_raw = common.canonical_json_bytes(tag_target_document)
        self._write(source / "git-tag-object.json", tag_object_raw)
        self._write(source / "git-tag-target.json", tag_target_raw)
        self._write(source / source_inputs.SOURCE_ARCHIVE_NAME, archive)
        archive_sha256 = _sha256(archive)
        source_binding = {
            "tag_name": source_inputs.RECONSTRUCTED_RC2_TAG,
            "tag_object": common.descriptor_for_bytes(
                "source/git-tag-object.json", tag_object_raw, "tag object"
            ).as_json(),
            "tag_target": common.descriptor_for_bytes(
                "source/git-tag-target.json", tag_target_raw, "tag target"
            ).as_json(),
            "archive": common.descriptor_for_bytes(
                f"source/{source_inputs.SOURCE_ARCHIVE_NAME}", archive, "source archive"
            ).as_json(),
        }
        receipt = {
            "schema_version": source_inputs.SOURCE_INPUTS_VERSION,
            "status": "prepared",
            "qualification_status": "not-run",
            "baseline_id": source_inputs.RECONSTRUCTED_RC2_BASELINE_ID,
            "source": source_binding,
            "git_identity": {
                "tag_ref": source_inputs.RECONSTRUCTED_RC2_TAG_REF,
                "tag_object_sha1": source_inputs.RECONSTRUCTED_RC2_TAG_OBJECT,
                "target_commit_sha1": source_inputs.RECONSTRUCTED_RC2_TARGET,
            },
            "expected_source_archive_sha256": archive_sha256,
            "archive_generation": source_inputs.ARCHIVE_GENERATION,
        }
        self._write(root / source_inputs.SOURCE_INPUTS_NAME, common.canonical_json_bytes(receipt))
        self.assert_source_replay(root, archive_sha256)
        return root, archive_sha256

    @staticmethod
    def assert_source_replay(root: Path, expected_sha256: str) -> None:
        source_inputs.verify_reconstructed_rc2_inputs(root, expected_source_archive_sha256=expected_sha256)

    @staticmethod
    def _tar_file(archive: tarfile.TarFile, name: str, raw: bytes) -> None:
        member = tarfile.TarInfo(name)
        member.size = len(raw)
        member.mode = 0o600
        member.mtime = 0
        archive.addfile(member, io.BytesIO(raw))

    @staticmethod
    def _tar_directory(archive: tarfile.TarFile, name: str) -> None:
        member = tarfile.TarInfo(name)
        member.type = tarfile.DIRTYPE
        member.mode = 0o700
        member.mtime = 0
        archive.addfile(member)

    def _make_oci_root(
        self,
        name: str,
        arm: str,
        layer: bytes,
    ) -> tuple[Path, dict[str, Any]]:
        config = _canonical(
            {
                "architecture": "amd64",
                "config": {"Entrypoint": ["/usr/local/bin/riley-server"]},
                "os": "linux",
                "rootfs": {"diff_ids": [], "type": "layers"},
            }
        )
        config_descriptor = {
            "mediaType": runtime_oci.OCI_CONFIG_MEDIA_TYPE,
            "digest": _oci_digest(config),
            "size": len(config),
        }
        layer_descriptor = {
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            "digest": _oci_digest(layer),
            "size": len(layer),
        }
        manifest = _canonical(
            {
                "schemaVersion": 2,
                "mediaType": runtime_oci.OCI_MANIFEST_MEDIA_TYPE,
                "config": config_descriptor,
                "layers": [layer_descriptor],
            }
        )
        index = _canonical(
            {
                "schemaVersion": 2,
                "mediaType": runtime_oci.OCI_INDEX_MEDIA_TYPE,
                "manifests": [
                    {
                        "mediaType": runtime_oci.OCI_MANIFEST_MEDIA_TYPE,
                        "digest": _oci_digest(manifest),
                        "size": len(manifest),
                        "platform": dict(runtime_oci.PLATFORM),
                    }
                ],
            }
        )
        layout = _canonical({"imageLayoutVersion": "1.0.0"})
        archive_path = self.base / f"{name}.tar"
        with tarfile.open(archive_path, "w", format=tarfile.USTAR_FORMAT) as archive:
            self._tar_directory(archive, "blobs")
            self._tar_directory(archive, "blobs/sha256")
            self._tar_file(archive, "oci-layout", layout)
            self._tar_file(archive, "index.json", index)
            self._tar_file(archive, f"blobs/sha256/{_sha256(manifest)}", manifest)
            self._tar_file(archive, f"blobs/sha256/{_sha256(config)}", config)
            self._tar_file(archive, f"blobs/sha256/{_sha256(layer)}", layer)
        os.chmod(archive_path, 0o600)
        inspect_path = self.base / f"{name}-inspect.json"
        inspect_path.write_bytes(
            _canonical(
                [
                    {
                        "Architecture": "amd64",
                        "Id": _oci_digest(config),
                        "Os": "linux",
                        "RepoTags": [],
                    }
                ]
            )
        )
        os.chmod(inspect_path, 0o600)
        root = self.base / name
        receipt = runtime_oci.prepare_reconstructed_runtime_oci_inputs(
            root,
            image_inspect=inspect_path,
            oci_archive=archive_path,
            reconstruction_id=arm,
        )
        if runtime_oci.verify_reconstructed_runtime_oci_inputs(root) != receipt:
            raise AssertionError("runtime OCI fixture did not replay exactly")
        return root, receipt

    def _make_baseline_root(self) -> None:
        root = self.baseline_root
        self._private_directory(root)
        source_directory = self.source_root / source_inputs.SOURCE_DIRECTORY_NAME
        source = {
            "tag_name": source_inputs.RECONSTRUCTED_RC2_TAG,
            "tag_object": self._write_root_file(
                root, "source/git-tag-object.json", (source_directory / "git-tag-object.json").read_bytes()
            ),
            "tag_target": self._write_root_file(
                root, "source/git-tag-target.json", (source_directory / "git-tag-target.json").read_bytes()
            ),
            "archive": self._write_root_file(
                root,
                f"source/riley-0.1.0-rc2.tar",
                (source_directory / source_inputs.SOURCE_ARCHIVE_NAME).read_bytes(),
            ),
        }
        image_id = self.oci_a_receipt["image_id"]
        artifacts: dict[str, dict[str, Any]] = {}
        raw_inspects: dict[str, dict[str, Any]] = {}
        recipes: dict[str, dict[str, Any]] = {}
        recipe_inspects: dict[str, dict[str, Any]] = {}
        image_inspects: dict[str, dict[str, Any]] = {}
        receipts: dict[str, dict[str, Any]] = {}
        receipt_descriptors: dict[str, dict[str, Any]] = {}
        for arm, oci_root in (("a", self.oci_a_root), ("b", self.oci_b_root)):
            runtime_directory = oci_root / runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME
            artifacts[arm] = {
                "binary": self._write_root_file(root, f"reproductions/{arm}/riley-server", b"same server binary\n"),
                "bundle": self._write_root_file(root, f"reproductions/{arm}/riley.bundle.tar.zst", b"same bundle\n"),
                "oci": {
                    "archive": self._write_root_file(
                        root,
                        f"reproductions/{arm}/oci-image.tar",
                        (runtime_directory / runtime_oci.OCI_ARCHIVE_NAME).read_bytes(),
                    ),
                    "layout": self._write_root_file(
                        root,
                        f"reproductions/{arm}/oci-layout",
                        (runtime_directory / runtime_oci.OCI_LAYOUT_NAME).read_bytes(),
                    ),
                    "manifest": self._write_root_file(
                        root,
                        f"reproductions/{arm}/oci-manifest.json",
                        (runtime_directory / runtime_oci.OCI_MANIFEST_NAME).read_bytes(),
                    ),
                },
            }
            raw_inspects[arm] = self._write_root_file(
                root,
                f"reproductions/{arm}/docker-image-inspect.json",
                (runtime_directory / runtime_oci.IMAGE_INSPECT_NAME).read_bytes(),
            )
            recipes[arm] = self._write_root_file(root, f"reproductions/{arm}/build-recipe.txt", f"recipe {arm}\n".encode())
            recipe_inspect_document = {
                "schema_version": baseline.RECIPE_INSPECT_VERSION,
                "baseline_id": source_inputs.RECONSTRUCTED_RC2_BASELINE_ID,
                "reconstruction_id": arm,
                "source": copy.deepcopy(source),
                "recipe": copy.deepcopy(recipes[arm]),
                "binary": copy.deepcopy(artifacts[arm]["binary"]),
                "bundle": copy.deepcopy(artifacts[arm]["bundle"]),
            }
            recipe_inspects[arm] = self._write_root_file(
                root,
                f"reproductions/{arm}/recipe-inspect.json",
                common.canonical_json_bytes(recipe_inspect_document),
            )
            image_inspect_document = {
                "schema_version": baseline.IMAGE_INSPECT_VERSION,
                "baseline_id": source_inputs.RECONSTRUCTED_RC2_BASELINE_ID,
                "reconstruction_id": arm,
                "source": copy.deepcopy(source),
                "runtime_image_inspect_raw": copy.deepcopy(raw_inspects[arm]),
                "oci_archive": copy.deepcopy(artifacts[arm]["oci"]["archive"]),
                "oci_layout": copy.deepcopy(artifacts[arm]["oci"]["layout"]),
                "oci_manifest": copy.deepcopy(artifacts[arm]["oci"]["manifest"]),
                "image_id": image_id,
            }
            image_inspects[arm] = self._write_root_file(
                root,
                f"reproductions/{arm}/image-inspect.json",
                common.canonical_json_bytes(image_inspect_document),
            )
            receipts[arm] = {
                "schema_version": baseline.BUILD_RECEIPT_VERSION,
                "baseline_id": source_inputs.RECONSTRUCTED_RC2_BASELINE_ID,
                "reconstruction_id": arm,
                "source": copy.deepcopy(source),
                "recipe_inspect": copy.deepcopy(recipe_inspects[arm]),
                "image_inspect": copy.deepcopy(image_inspects[arm]),
                "runtime_image_inspect_raw": copy.deepcopy(raw_inspects[arm]),
                "artifacts": copy.deepcopy(artifacts[arm]),
            }
            receipt_descriptors[arm] = self._write_root_file(
                root,
                f"reproductions/{arm}/build-receipt.json",
                common.canonical_json_bytes(receipts[arm]),
            )
        def equality_descriptor(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
            return {"a": copy.deepcopy(left), "b": copy.deepcopy(right), "sha256": left["sha256"]}

        manifest = {
            "schema_version": baseline.MANIFEST_VERSION,
            "baseline_id": source_inputs.RECONSTRUCTED_RC2_BASELINE_ID,
            "baseline_kind": baseline.BASELINE_KIND,
            "provenance_class": baseline.PROVENANCE_CLASS,
            "historical_distribution": baseline.HISTORICAL_DISTRIBUTION,
            "historical_stable_artifact_status": baseline.HISTORICAL_STABLE_ARTIFACT_STATUS,
            "was_previously_shipped": baseline.WAS_PREVIOUSLY_SHIPPED,
            "source": copy.deepcopy(source),
            "reproductions": {"a": receipt_descriptors["a"], "b": receipt_descriptors["b"]},
            "equality": {
                "binary": equality_descriptor(artifacts["a"]["binary"], artifacts["b"]["binary"]),
                "bundle": equality_descriptor(artifacts["a"]["bundle"], artifacts["b"]["bundle"]),
                "oci_archive": equality_descriptor(
                    artifacts["a"]["oci"]["archive"], artifacts["b"]["oci"]["archive"]
                ),
                "oci_layout": equality_descriptor(
                    artifacts["a"]["oci"]["layout"], artifacts["b"]["oci"]["layout"]
                ),
                "oci_manifest": equality_descriptor(
                    artifacts["a"]["oci"]["manifest"], artifacts["b"]["oci"]["manifest"]
                ),
                "oci_image": {"a": image_id, "b": image_id, "image_id": image_id},
            },
        }
        self._write(root / "baseline.json", common.canonical_json_bytes(manifest))
        self.baseline_manifest = manifest
        if baseline.validate_file(root, "baseline.json")["passed"] is not True:
            raise AssertionError("baseline fixture did not replay exactly")


class ReconstructedPriorBaselineContentBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = BridgeFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def _bridge_root(self, name: str = "content-bridge") -> Path:
        return self.fixture.base / name

    def _arguments(self, **changes: Any) -> dict[str, Any]:
        values = {
            "baseline_root": self.fixture.baseline_root,
            "baseline_manifest": "baseline.json",
            "source_input_root": self.fixture.source_root,
            "expected_source_archive_sha256": self.fixture.source_sha256,
            "runtime_oci_a_root": self.fixture.oci_a_root,
            "runtime_oci_b_root": self.fixture.oci_b_root,
        }
        values.update(changes)
        return values

    def _prepare(self, root: Path | None = None, **changes: Any) -> dict[str, Any]:
        return bridge.prepare_reconstructed_prior_baseline_content_bridge(
            root or self._bridge_root(), **self._arguments(**changes)
        )

    def _verify(self, root: Path | None = None, **changes: Any) -> dict[str, Any]:
        return bridge.verify_reconstructed_prior_baseline_content_bridge(
            root or self._bridge_root(), **self._arguments(**changes)
        )

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(bridge.ReconstructedPriorBaselineContentBridgeError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_prepares_and_replays_a_narrow_cross_root_content_bridge(self) -> None:
        root = self._bridge_root()
        receipt = self._prepare(root)
        self.assertEqual(receipt["schema_version"], bridge.BRIDGE_VERSION)
        self.assertEqual(receipt["status"], bridge.BRIDGE_STATUS)
        self.assertEqual(receipt["qualification_status"], "not-run")
        self.assertEqual(receipt["authority"], bridge.BRIDGE_AUTHORITY)
        self.assertEqual(receipt["binding_status"]["v2_report_oci_archive_content_binding"], "not-validated")
        self.assertEqual(
            receipt["binding_status"]["source_archive_content_binding"],
            bridge.SOURCE_ARCHIVE_CONTENT_BINDING,
        )
        self.assertEqual(
            receipt["binding_status"]["oci_archive_content_binding"],
            bridge.OCI_ARCHIVE_CONTENT_BINDING,
        )
        self.assertEqual(receipt["not_established"]["source_to_runtime_image_binding"], "not-established")
        self.assertEqual(receipt["not_established"]["qualification"], "not-run")
        self.assertEqual(self._verify(root), receipt)
        self.assertEqual(set(os.listdir(root)), {bridge.BRIDGE_RECEIPT_NAME})
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((root / bridge.BRIDGE_RECEIPT_NAME).stat().st_mode), 0o600)
        self.assertEqual((root / bridge.BRIDGE_RECEIPT_NAME).stat().st_nlink, 1)
        legacy = baseline.validate_file(self.fixture.baseline_root, "baseline.json")
        self.assertEqual(legacy["source_archive_content_binding"], "not-validated")
        self.assertEqual(legacy["oci_archive_content_binding"], "not-validated")

    def test_requires_the_caller_reviewer_sha_on_each_replay(self) -> None:
        root = self._bridge_root()
        self._prepare(root)
        self.assert_reason(
            "reviewed-source-archive-digest-mismatch",
            lambda: self._verify(root, expected_source_archive_sha256="a" * 64),
        )

    def test_rejects_a_different_valid_source_closure(self) -> None:
        alternative_root, alternative_sha = self.fixture._make_source_root("other-source-inputs", b"different valid source\n")
        root = self._bridge_root()
        self.assert_reason(
            "cross-root-content-mismatch",
            lambda: self._prepare(
                root,
                source_input_root=alternative_root,
                expected_source_archive_sha256=alternative_sha,
            ),
        )
        self.assertFalse(root.exists())

    def test_rejects_a_different_valid_oci_archive_even_when_config_image_id_matches(self) -> None:
        alternate_root, _alternate_receipt = self.fixture._make_oci_root(
            "other-runtime-oci-a", "a", b"different layer with same config\n"
        )
        root = self._bridge_root()
        self.assert_reason(
            "cross-root-content-mismatch",
            lambda: self._prepare(root, runtime_oci_a_root=alternate_root),
        )
        self.assertFalse(root.exists())

    def test_rejects_arm_swaps_and_overlapping_root_roles_before_output(self) -> None:
        root = self._bridge_root()
        self.assert_reason(
            "reconstruction-id-mismatch",
            lambda: self._prepare(
                root,
                runtime_oci_a_root=self.fixture.oci_b_root,
                runtime_oci_b_root=self.fixture.oci_a_root,
            ),
        )
        self.assertFalse(root.exists())
        self.assert_reason(
            "bridge-root-overlap",
            lambda: self._prepare(self.fixture.baseline_root / "bridge"),
        )
        self.assert_reason(
            "bridge-root-overlap",
            lambda: self._prepare(
                self._bridge_root("alias"), runtime_oci_b_root=self.fixture.oci_a_root
            ),
        )

    def test_rejects_nonprivate_or_symlinked_external_roots(self) -> None:
        self.fixture.oci_a_root.chmod(0o755)
        try:
            self.assert_reason("unsafe-evidence-root-mode", lambda: self._prepare())
        finally:
            self.fixture.oci_a_root.chmod(0o700)
        symlink = self.fixture.base / "oci-a-link"
        symlink.symlink_to(self.fixture.oci_a_root.name, target_is_directory=True)
        self.assert_reason(
            "unsafe-evidence-directory",
            lambda: self._prepare(self._bridge_root("symlink"), runtime_oci_a_root=symlink),
        )

    def test_rejects_tampered_bridge_receipt_and_create_only_collision(self) -> None:
        root = self._bridge_root()
        receipt = self._prepare(root)
        self.assert_reason("create-only-collision", lambda: self._prepare(root))
        altered = copy.deepcopy(receipt)
        altered["not_established"]["source_to_runtime_image_binding"] = "validated"
        (root / bridge.BRIDGE_RECEIPT_NAME).write_bytes(common.canonical_json_bytes(altered))
        os.chmod(root / bridge.BRIDGE_RECEIPT_NAME, 0o600)
        self.assert_reason("invalid-bridge-receipt", lambda: self._verify(root))

    def test_replay_binds_the_caller_relative_baseline_manifest_and_rechecks_before_publish(self) -> None:
        root = self._bridge_root()
        self._prepare(root)
        alias = self.fixture.baseline_root / "baseline-copy.json"
        alias.write_bytes((self.fixture.baseline_root / "baseline.json").read_bytes())
        os.chmod(alias, 0o600)
        self.assertTrue(baseline.validate_file(self.fixture.baseline_root, "baseline-copy.json")["passed"])
        self.assert_reason(
            "bridge-replay-mismatch",
            lambda: self._verify(root, baseline_manifest="baseline-copy.json"),
        )

        unstable_root = self._bridge_root("unstable")
        original = bridge._replay_external_inputs
        calls = 0

        def unstable(**kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            value = original(**kwargs)
            if calls == 2:
                value = copy.deepcopy(value)
                value["baseline"]["baseline_id"] = "reconstructed-riley-0.1.0-rc2-raced"
            return value

        with mock.patch.object(bridge, "_replay_external_inputs", side_effect=unstable):
            self.assert_reason("bridge-replay-mismatch", lambda: self._prepare(unstable_root))
        self.assertTrue((unstable_root / bridge.BRIDGE_RECEIPT_NAME).is_file())

    def test_rejects_extra_bridge_entries(self) -> None:
        root = self._bridge_root()
        self._prepare(root)
        extra = root / "extra"
        extra.write_bytes(b"unexpected")
        os.chmod(extra, 0o600)
        self.assert_reason("unexpected-evidence-entry", lambda: self._verify(root))

    def test_descriptor_comparison_includes_byte_length_not_just_digest(self) -> None:
        descriptor = {
            "path": "left",
            "sha256": "a" * 64,
            "byte_length": 1,
        }
        other = dict(descriptor)
        other["path"] = "right"
        other["byte_length"] = 2
        self.assert_reason(
            "cross-root-content-mismatch",
            lambda: bridge._assert_same_fingerprint(descriptor, other, "fixture descriptor"),
        )

    def test_published_schema_and_source_only_implementation_contract(self) -> None:
        repository_schema = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "reconstructed-prior-baseline-content-bridge-v1.schema.json"
        )
        schema_path = repository_schema if repository_schema.is_file() else Path(__file__).with_name(
            "reconstructed-prior-baseline-content-bridge-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$id"],
            "https://riley.invalid/benchmarks/release/candidates/"
            "reconstructed-prior-baseline-content-bridge-v1.schema.json",
        )
        self.assertEqual(schema["properties"]["schema_version"], {"const": bridge.BRIDGE_VERSION})
        self.assertEqual(schema["properties"]["authority"], {"const": bridge.BRIDGE_AUTHORITY})
        self.assertEqual(
            schema["$defs"]["bindingStatus"]["properties"]["oci_archive_content_binding"],
            {"const": bridge.OCI_ARCHIVE_CONTENT_BINDING},
        )
        self.assertEqual(
            schema["$defs"]["notEstablished"]["properties"]["source_to_runtime_image_binding"],
            {"const": "not-established"},
        )
        source = Path(bridge.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        self.assertFalse({"subprocess", "socket", "urllib", "requests"} & imported)
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse({"system", "popen", "Popen", "run", "check_call", "check_output"} & calls)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
