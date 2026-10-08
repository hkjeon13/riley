#!/usr/bin/env python3
"""CPU-only hostile-path tests for the image-export/assembly content bridge."""

from __future__ import annotations

import ast
import json
import os
import stat
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

import prepare_reconstructed_runtime_image_export_assembly_content_bridge_v1 as bridge  # noqa: E402
import prepare_reconstructed_runtime_image_export_oci_normalization_v1 as normalize  # noqa: E402
import prepare_reconstructed_runtime_oci_inputs_v1 as runtime_oci  # noqa: E402
import provenance_v2_common as common  # noqa: E402
import test_prepare_reconstructed_runtime_assembly_capture_v1 as capture_fixture  # noqa: E402
import test_prepare_reconstructed_runtime_oci_inputs_v1 as oci_fixture  # noqa: E402
import test_reproducible_build as reproducibility_fixture  # noqa: E402


class BridgeFixture:
    """Construct N -> O -> C evidence with the existing CPU-only fixtures."""

    def __init__(self) -> None:
        self.capture_test = capture_fixture.RuntimeAssemblyCaptureTests(methodName="runTest")
        self.capture_test.setUp()
        self.base = self.capture_test.base
        self.source_root = self.capture_test.repro_fixture.source_root
        self.repro_root = self.capture_test.repro_root
        self.expected_source_sha256 = self.capture_test.expected_source_sha
        self.expected_build_image_id = reproducibility_fixture.IMAGE_ID

        raw_fixture = self.capture_test.oci_fixture
        self.normalization_root = self.base / "image-export-normalization"
        self.normalization_receipt = normalize.prepare_reconstructed_runtime_image_export_oci_normalization(
            self.normalization_root,
            image_inspect=raw_fixture.inspect,
            image_export_archive=raw_fixture.archive,
            reconstruction_id="a",
        )
        normalized_inspect = self.normalization_root / normalize.RAW_DIRECTORY_NAME / normalize.IMAGE_INSPECT_NAME
        normalized_archive = self.normalization_root / normalize.NORMALIZED_DIRECTORY_NAME / normalize.OCI_ARCHIVE_NAME
        self.runtime_oci_root = self.base / "runtime-oci-from-normalized-export"
        self.runtime_oci_receipt = runtime_oci.prepare_reconstructed_runtime_oci_inputs(
            self.runtime_oci_root,
            image_inspect=normalized_inspect,
            oci_archive=normalized_archive,
            reconstruction_id="a",
        )
        self.capture_test.oci_fixture = oci_fixture.OciFixture(
            inspect=normalized_inspect,
            archive=normalized_archive,
            image_id=self.normalization_receipt["image_id"],
        )
        self.capture_test.oci_root = self.runtime_oci_root
        self.capture_test.oci_receipt = self.runtime_oci_receipt
        self.capture_test._write_capture(self.capture_test.raw_capture)
        self.capture_root = self.base / "runtime-assembly-capture-from-normalized-export"
        self.capture_receipt = self.capture_test._prepare(self.capture_root)

    def close(self) -> None:
        self.capture_test.tearDown()

    def arguments(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "source_input_root": self.source_root,
            "repro_build_input_root": self.repro_root,
            "image_export_normalization_root": self.normalization_root,
            "runtime_oci_input_root": self.runtime_oci_root,
            "assembly_capture_root": self.capture_root,
            "expected_source_archive_sha256": self.expected_source_sha256,
            "expected_build_image_id": self.expected_build_image_id,
            "reconstruction_id": "a",
        }
        values.update(overrides)
        return values

    def make_alternate_normalization(self, name: str, *, layer: bytes) -> Path:
        fixture = self.capture_test.oci_fixture_test._write_fixture(name, actual_layer=layer)
        root = self.base / f"{name}-normalization"
        normalize.prepare_reconstructed_runtime_image_export_oci_normalization(
            root,
            image_inspect=fixture.inspect,
            image_export_archive=fixture.archive,
            reconstruction_id="a",
        )
        return root


class RuntimeImageExportAssemblyContentBridgeTests(unittest.TestCase):
    fixture: BridgeFixture

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = BridgeFixture()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.close()

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(bridge.RuntimeImageExportAssemblyContentBridgeError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def _prepare(self, root: Path, **overrides: object) -> dict[str, object]:
        return bridge.prepare_reconstructed_runtime_image_export_assembly_content_bridge(
            root,
            **self.fixture.arguments(**overrides),  # type: ignore[arg-type]
        )

    def _verify(self, root: Path, **overrides: object) -> dict[str, object]:
        return bridge.verify_reconstructed_runtime_image_export_assembly_content_bridge(
            root,
            **self.fixture.arguments(**overrides),  # type: ignore[arg-type]
        )

    def test_prepares_and_replays_one_closed_content_bridge(self) -> None:
        root = self.fixture.base / "content-bridge"
        receipt = self._prepare(root)
        self.assertEqual(receipt["schema_version"], bridge.BRIDGE_VERSION)
        self.assertEqual(receipt["status"], "bound")
        self.assertEqual(receipt["qualification_status"], "not-run")
        self.assertEqual(receipt["capture_scope"], bridge.BRIDGE_CAPTURE_SCOPE)
        self.assertEqual(receipt["image_id"], self.fixture.normalization_receipt["image_id"])
        self.assertEqual(self._verify(root), receipt)
        self.assertEqual(set(os.listdir(root)), {bridge.BRIDGE_RECEIPT_NAME})
        leaf = root / bridge.BRIDGE_RECEIPT_NAME
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(leaf.stat().st_mode), 0o600)
        self.assertEqual(leaf.stat().st_nlink, 1)
        normalization = receipt["image_export_normalization"]
        runtime = receipt["runtime_oci_inputs"]
        capture = receipt["assembly_capture"]
        assert isinstance(normalization, dict) and isinstance(runtime, dict) and isinstance(capture, dict)
        self.assertEqual(normalization["image_export_archive"]["path"], "image-export/runtime-image-export.tar")
        self.assertEqual(normalization["oci_archive"]["sha256"], runtime["archive"]["sha256"])
        self.assertEqual(capture["capture_members"]["oci-image-layout.tar"]["sha256"], runtime["archive"]["sha256"])
        schema_path = (
            Path(bridge.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/reconstructed-runtime-image-export-assembly-content-bridge-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], bridge.BRIDGE_VERSION)
        self.assertEqual(schema["properties"]["capture_scope"]["const"], bridge.BRIDGE_CAPTURE_SCOPE)
        self.assertEqual(schema["properties"]["binding_status"]["const"], bridge.BINDING_STATUS)
        self.assertEqual(schema["properties"]["not_established"]["const"], bridge.NOT_ESTABLISHED_STATUS)

    def test_rejects_arm_swap_and_root_overlap_before_publish(self) -> None:
        self.assert_reason(
            "reconstruction-id-mismatch",
            lambda: self._prepare(self.fixture.base / "wrong-arm", reconstruction_id="b"),
        )
        self.assert_reason(
            "bridge-root-overlap",
            lambda: self._prepare(self.fixture.normalization_root),
        )

    def test_rejects_same_image_id_with_a_different_normalized_layer(self) -> None:
        alternate = self.fixture.make_alternate_normalization("same-id-different-layer", layer=b"different layer\n")
        self.assert_reason(
            "cross-root-content-mismatch",
            lambda: self._prepare(
                self.fixture.base / "different-layer-bridge",
                image_export_normalization_root=alternate,
            ),
        )

    def test_replay_rejects_claim_escalation_and_extra_output(self) -> None:
        root = self.fixture.base / "mutated-bridge"
        self._prepare(root)
        receipt_path = root / bridge.BRIDGE_RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["not_established"]["docker_image_export_execution"] = "validated"
        receipt_path.write_bytes(common.canonical_json_bytes(receipt))
        os.chmod(receipt_path, 0o600)
        self.assert_reason("invalid-bridge-receipt", lambda: self._verify(root))

        root = self.fixture.base / "extra-output-bridge"
        self._prepare(root)
        extra = root / "extra"
        extra.write_bytes(b"extra")
        os.chmod(extra, 0o600)
        self.assert_reason("unexpected-evidence-entry", lambda: self._verify(root))

    def test_replay_rejects_unhashable_closed_contract_fields(self) -> None:
        for name, mutate in (
            (
                "unhashable-arm",
                lambda receipt: receipt.__setitem__("reconstruction_id", []),
            ),
            (
                "unhashable-source-layout",
                lambda receipt: receipt["image_export_normalization"].__setitem__("source_layout", []),  # type: ignore[index]
            ),
        ):
            root = self.fixture.base / name
            self._prepare(root)
            receipt_path = root / bridge.BRIDGE_RECEIPT_NAME
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            mutate(receipt)
            receipt_path.write_bytes(common.canonical_json_bytes(receipt))
            os.chmod(receipt_path, 0o600)
            self.assert_reason("invalid-bridge-receipt", lambda root=root: self._verify(root))

    def test_rejects_external_root_alias_and_never_imports_operational_clients(self) -> None:
        self.assert_reason(
            "bridge-root-overlap",
            lambda: self._prepare(
                self.fixture.base / "aliased-roots",
                runtime_oci_input_root=self.fixture.normalization_root,
            ),
        )
        root_fd = common.open_private_evidence_directory(self.fixture.normalization_root, "normalization as bridge root")
        source_fd = common.open_private_evidence_directory(self.fixture.source_root, "source inputs root")
        repro_fd = common.open_private_evidence_directory(self.fixture.repro_root, "reproducibility inputs root")
        normalization_fd = os.dup(root_fd)
        oci_fd = common.open_private_evidence_directory(self.fixture.runtime_oci_root, "runtime OCI inputs root")
        capture_fd = common.open_private_evidence_directory(self.fixture.capture_root, "runtime assembly capture root")
        try:
            self.assert_reason(
                "input-root-alias",
                lambda: bridge.verify_reconstructed_runtime_image_export_assembly_content_bridge_fd(
                    root_fd,
                    source_input_root_fd=source_fd,
                    repro_build_input_root_fd=repro_fd,
                    image_export_normalization_root_fd=normalization_fd,
                    runtime_oci_input_root_fd=oci_fd,
                    assembly_capture_root_fd=capture_fd,
                    expected_source_archive_sha256=self.fixture.expected_source_sha256,
                    expected_build_image_id=self.fixture.expected_build_image_id,
                    reconstruction_id="a",
                ),
            )
        finally:
            for descriptor in (capture_fd, oci_fd, normalization_fd, repro_fd, source_fd, root_fd):
                os.close(descriptor)
        source = Path(bridge.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_imports = {"subprocess", "socket", "urllib", "requests"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({alias.name.split(".")[0] for alias in node.names} & forbidden_imports)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden_imports)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {"extract", "extractall"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
