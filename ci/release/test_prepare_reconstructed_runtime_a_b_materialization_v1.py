#!/usr/bin/env python3
"""CPU-only hostile-path tests for A/B runtime content materialization v1.

The remote source host deliberately has Python 3.10 while the upstream PR16
reproducibility replayer requires Python 3.11+.  These tests exercise this
module's held-FD/root/output contract by replacing only the already-reviewed
per-arm capture replayer.  They never need Docker, a compiler, a GPU, a
container, a network connection, or real release evidence.
"""

from __future__ import annotations

import ast
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import prepare_reconstructed_runtime_a_b_materialization_v1 as materializer  # noqa: E402
import provenance_v2_common as common  # noqa: E402


SOURCE_SHA = "a" * 64
BUILD_IMAGE_ID = "sha256:" + "b" * 64
TAG_OBJECT_SHA = "c" * 40
TARGET_COMMIT_SHA = "d" * 40
BINARY_SHA = "1" * 64
BUNDLE_SHA = "2" * 64
TREE_SHA = "3" * 64


def _descriptor(path: str, digest: str, length: int = 1) -> dict[str, object]:
    return {"path": path, "sha256": digest, "byte_length": length}


class RuntimeABMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="riley-runtime-ab-materialization-")
        self.base = Path(self.temporary.name).resolve()
        self.source_root = self._private_root("source")
        self.repro_root = self._private_root("repro")
        self.oci_a_root = self._private_root("oci-a")
        self.oci_b_root = self._private_root("oci-b")
        self.capture_a_root = self._private_root("capture-a")
        self.capture_b_root = self._private_root("capture-b")
        self._write_capture_receipt_leaf(self.capture_a_root, "a")
        self._write_capture_receipt_leaf(self.capture_b_root, "b")
        self.capture_by_inode = {
            self._identity(self.capture_a_root): "a",
            self._identity(self.capture_b_root): "b",
        }
        self.row_mutators: dict[str, object] = {}
        self.calls: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _private_root(self, name: str) -> Path:
        path = self.base / name
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
        return path

    def _write_capture_receipt_leaf(self, root: Path, arm: str) -> None:
        leaf = root / materializer.assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_NAME
        leaf.write_bytes(common.canonical_json_bytes({"fixture_capture_arm": arm}))
        os.chmod(leaf, 0o600)

    @staticmethod
    def _identity(path: Path) -> tuple[int, int]:
        metadata = path.stat()
        return metadata.st_dev, metadata.st_ino

    @staticmethod
    def _fd_identity(descriptor: int) -> tuple[int, int]:
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino

    def _capture_row(self, arm: str) -> dict[str, object]:
        image_digest = "4" * 64 if arm == "a" else "5" * 64
        oci_digest = "6" * 64 if arm == "a" else "7" * 64
        capture_digest = "8" * 64 if arm == "a" else "9" * 64
        selected = {
            "reconstruction_id": arm,
            "evidence_build_id": arm.upper(),
            "binary": _descriptor(f"repro-builds/{arm}/riley", BINARY_SHA, 17),
            "bundle": _descriptor(f"repro-builds/{arm}/riley.tar.gz", BUNDLE_SHA, 31),
        }
        return {
            "schema_version": materializer.assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_VERSION,
            "status": materializer.assembly_capture.STATUS,
            "qualification_status": "not-run",
            "authority": materializer.assembly_capture.AUTHORITY,
            "capture_scope": materializer.assembly_capture.CAPTURE_SCOPE,
            "baseline_id": materializer.BASELINE_ID,
            "reconstruction_id": arm,
            "platform": dict(materializer.PLATFORM),
            "source_inputs": {
                "receipt": _descriptor(materializer.SOURCE_INPUT_RECEIPT_NAME, "a" * 64, 9),
                "expected_source_archive_sha256": SOURCE_SHA,
                "git_identity": {
                    "tag_ref": "refs/tags/riley-0.1.0-rc2",
                    "tag_object_sha1": TAG_OBJECT_SHA,
                    "target_commit_sha1": TARGET_COMMIT_SHA,
                },
                "source": {},
            },
            "reproducibility_inputs": {
                "receipt": _descriptor(materializer.REPRO_INPUT_RECEIPT_NAME, "b" * 64, 11),
                "reproducibility_contract": {
                    "schema_version": 1,
                    "gate_id": "pr16-release-build-reproducibility-v1",
                    "source_revision": TARGET_COMMIT_SHA,
                    "source_date_epoch": 1,
                    "build_image_id": BUILD_IMAGE_ID,
                    "platform": dict(materializer.PLATFORM),
                    "network": "none",
                    "independent_clean_containers": 2,
                },
                "selected_build": selected,
            },
            "runtime_oci_inputs": {
                "receipt": _descriptor(materializer.RUNTIME_OCI_INPUT_RECEIPT_NAME, oci_digest, 13),
                "reconstruction_id": arm,
                "image_id": "sha256:" + image_digest,
                "image_inspect": _descriptor("runtime-image/docker-image-inspect.json", oci_digest, 19),
                "archive": _descriptor("runtime-image/oci-image-layout.tar", oci_digest, 23),
                "layout": _descriptor("runtime-image/oci-layout", oci_digest, 3),
                "index": _descriptor("runtime-image/index.json", oci_digest, 5),
                "manifest": _descriptor("runtime-image/manifest.json", oci_digest, 7),
                "config": _descriptor("runtime-image/config.json", oci_digest, 11),
            },
            "assembly_recipe": {},
            "capture": {
                "archive": _descriptor(
                    "runtime-assembly-capture/assembly-capture.tar",
                    capture_digest,
                    29,
                ),
                "members": {
                    name: {"sha256": capture_digest, "byte_length": 0 if name == "build.log" else 1}
                    for name in materializer.assembly_capture.CAPTURE_MEMBER_NAMES
                },
                "context": {},
                "image_id": "sha256:" + image_digest,
                "container_id": "f" * 64,
                "runtime_tree": {"sha256": TREE_SHA, "entry_count": 3, "byte_length": 47},
            },
            "binding_status": {},
            "not_established": {},
        }

    def _fake_capture_replay(self, root_fd: int, **kwargs: object) -> dict[str, object]:
        identity = self._fd_identity(root_fd)
        arm = self.capture_by_inode.get(identity)
        if arm is None:
            raise AssertionError("materializer passed an unknown capture root")
        requested = kwargs["reconstruction_id"]
        self.calls.append((str(requested), identity, self._fd_identity(kwargs["runtime_oci_input_root_fd"])))  # type: ignore[arg-type]
        row = self._capture_row(arm)
        mutator = self.row_mutators.get(arm)
        if mutator is not None:
            mutator(row)  # type: ignore[operator]
        return row

    def _arguments(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "source_input_root": self.source_root,
            "repro_build_input_root": self.repro_root,
            "runtime_oci_input_root_a": self.oci_a_root,
            "runtime_oci_input_root_b": self.oci_b_root,
            "assembly_capture_root_a": self.capture_a_root,
            "assembly_capture_root_b": self.capture_b_root,
            "expected_source_archive_sha256": SOURCE_SHA,
            "expected_build_image_id": BUILD_IMAGE_ID,
        }
        values.update(overrides)
        return values

    def _patch_capture_replay(self) -> mock._patch[object]:
        return mock.patch.object(
            materializer.assembly_capture,
            "verify_reconstructed_runtime_assembly_capture_fd",
            side_effect=self._fake_capture_replay,
        )

    def _prepare(self, root: Path, **overrides: object) -> dict[str, object]:
        with self._patch_capture_replay():
            return materializer.prepare_reconstructed_runtime_a_b_materialization(
                root,
                **self._arguments(**overrides),  # type: ignore[arg-type]
            )

    def _verify(self, root: Path, **overrides: object) -> dict[str, object]:
        with self._patch_capture_replay():
            return materializer.verify_reconstructed_runtime_a_b_materialization(
                root,
                **self._arguments(**overrides),  # type: ignore[arg-type]
            )

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(materializer.RuntimeABMaterializationError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_prepares_and_replays_closed_content_only_receipt(self) -> None:
        root = self.base / "materialization"
        receipt = self._prepare(root)
        self.assertEqual(receipt["schema_version"], materializer.MATERIALIZATION_VERSION)
        self.assertEqual(receipt["status"], "bound")
        self.assertEqual(receipt["qualification_status"], "not-run")
        self.assertEqual(receipt["materialization_scope"], materializer.MATERIALIZATION_SCOPE)
        arms = receipt["arms"]
        assert isinstance(arms, dict)
        self.assertNotEqual(arms["a"]["runtime_oci_inputs"]["image_id"], arms["b"]["runtime_oci_inputs"]["image_id"])
        self.assertEqual(receipt["equality"]["release_binary"], {"sha256": BINARY_SHA, "byte_length": 17})
        self.assertEqual(receipt["equality"]["captured_runtime_tree"], {"sha256": TREE_SHA, "byte_length": 47, "entry_count": 3})
        self.assertEqual(self._verify(root), receipt)
        self.assertEqual(set(os.listdir(root)), {materializer.MATERIALIZATION_RECEIPT_NAME})
        leaf = root / materializer.MATERIALIZATION_RECEIPT_NAME
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(leaf.stat().st_mode), 0o600)
        self.assertEqual(leaf.stat().st_nlink, 1)
        schema_path = (
            Path(materializer.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/reconstructed-runtime-a-b-materialization-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], materializer.MATERIALIZATION_VERSION)
        self.assertEqual(schema["properties"]["binding_status"]["const"], materializer.BINDING_STATUS)
        self.assertEqual(schema["properties"]["not_established"]["const"], materializer.NOT_ESTABLISHED)

    def test_rejects_arm_swap_root_alias_and_output_overlap(self) -> None:
        self.capture_by_inode[self._identity(self.capture_a_root)] = "b"
        self.assert_reason("reconstruction-id-mismatch", lambda: self._prepare(self.base / "arm-swap"))
        self.capture_by_inode[self._identity(self.capture_a_root)] = "a"
        self.assert_reason(
            "materialization-root-overlap",
            lambda: self._prepare(self.source_root),
        )
        root = self.base / "alias-output"
        self._prepare(root)
        root_fd = common.open_private_evidence_directory(root, "materialization root")
        source_fd = common.open_private_evidence_directory(self.source_root, "source root")
        repro_fd = common.open_private_evidence_directory(self.repro_root, "repro root")
        oci_a_fd = common.open_private_evidence_directory(self.oci_a_root, "OCI a root")
        oci_b_fd = os.dup(oci_a_fd)
        capture_a_fd = common.open_private_evidence_directory(self.capture_a_root, "capture a root")
        capture_b_fd = common.open_private_evidence_directory(self.capture_b_root, "capture b root")
        try:
            with self._patch_capture_replay():
                self.assert_reason(
                    "input-root-alias",
                    lambda: materializer.verify_reconstructed_runtime_a_b_materialization_fd(
                        root_fd,
                        source_input_root_fd=source_fd,
                        repro_build_input_root_fd=repro_fd,
                        runtime_oci_input_root_a_fd=oci_a_fd,
                        runtime_oci_input_root_b_fd=oci_b_fd,
                        assembly_capture_root_a_fd=capture_a_fd,
                        assembly_capture_root_b_fd=capture_b_fd,
                        expected_source_archive_sha256=SOURCE_SHA,
                        expected_build_image_id=BUILD_IMAGE_ID,
                    ),
                )
        finally:
            for descriptor in (capture_b_fd, capture_a_fd, oci_b_fd, oci_a_fd, repro_fd, source_fd, root_fd):
                os.close(descriptor)

    def test_rejects_cross_arm_artifact_and_runtime_tree_mismatch(self) -> None:
        def binary_drift(row: dict[str, object]) -> None:
            row["reproducibility_inputs"]["selected_build"]["binary"]["sha256"] = "e" * 64  # type: ignore[index]

        self.row_mutators["b"] = binary_drift
        self.assert_reason("a-b-content-mismatch", lambda: self._prepare(self.base / "binary-drift"))

        def tree_drift(row: dict[str, object]) -> None:
            row["capture"]["runtime_tree"]["sha256"] = "e" * 64  # type: ignore[index]

        self.row_mutators["b"] = tree_drift
        self.assert_reason("a-b-runtime-tree-mismatch", lambda: self._prepare(self.base / "tree-drift"))

    def test_replay_rejects_claim_escalation_extra_output_and_input_drift(self) -> None:
        root = self.base / "mutated"
        self._prepare(root)
        receipt_path = root / materializer.MATERIALIZATION_RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["not_established"]["runtime_build_execution"] = "validated"
        receipt_path.write_bytes(common.canonical_json_bytes(receipt))
        os.chmod(receipt_path, 0o600)
        self.assert_reason("invalid-materialization-receipt", lambda: self._verify(root))

        root = self.base / "extra-output"
        self._prepare(root)
        extra = root / "extra"
        extra.write_bytes(b"extra")
        os.chmod(extra, 0o600)
        self.assert_reason("unexpected-evidence-entry", lambda: self._verify(root))

        root = self.base / "input-drift"
        calls = 0

        def drift_after_first_replay(_root_fd: int, **kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            row = self._capture_row(str(kwargs["reconstruction_id"]))
            if calls > 2 and kwargs["reconstruction_id"] == "b":
                row["capture"]["runtime_tree"]["sha256"] = "e" * 64  # type: ignore[index]
            return row

        with mock.patch.object(
            materializer.assembly_capture,
            "verify_reconstructed_runtime_assembly_capture_fd",
            side_effect=drift_after_first_replay,
        ):
            self.assert_reason(
                "a-b-runtime-tree-mismatch",
                lambda: materializer.prepare_reconstructed_runtime_a_b_materialization(
                    root,
                    **self._arguments(),  # type: ignore[arg-type]
                ),
            )

    def test_never_imports_or_copies_operational_clients(self) -> None:
        source = Path(materializer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_imports = {"subprocess", "socket", "urllib", "requests", "docker", "torch", "pynvml"}
        forbidden_attributes = {"extract", "extractall", "system", "popen", "run", "materialize_descriptor_runtime_copy"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({alias.name.split(".")[0] for alias in node.names} & forbidden_imports)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden_imports)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, forbidden_attributes)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
