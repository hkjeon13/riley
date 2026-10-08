#!/usr/bin/env python3
"""CPU-only hostile-path tests for reconstructed runtime OCI inputs v1."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import prepare_reconstructed_runtime_oci_inputs_v1 as prepare  # noqa: E402
import provenance_v2_common as common  # noqa: E402


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _raw_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class OciFixture:
    inspect: Path
    archive: Path
    image_id: str


class RuntimeOciInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _root(self, name: str = "evidence") -> Path:
        return self.base / name

    def _add_file(self, archive: tarfile.TarFile, name: str, raw: bytes) -> None:
        member = tarfile.TarInfo(name)
        member.size = len(raw)
        member.mode = 0o600
        member.mtime = 0
        archive.addfile(member, io.BytesIO(raw))

    def _add_directory(self, archive: tarfile.TarFile, name: str, payload: bytes = b"") -> None:
        member = tarfile.TarInfo(name)
        member.type = tarfile.DIRTYPE
        member.size = len(payload)
        member.mode = 0o700
        member.mtime = 0
        archive.addfile(member, io.BytesIO(payload) if payload else None)

    def _write_fixture(
        self,
        name: str = "fixture",
        *,
        inspect_id: str | None = None,
        inspect_raw: bytes | None = None,
        index_platform: dict[str, str] | None = None,
        duplicate_index_manifest: bool = False,
        include_layer: bool = True,
        actual_layer: bytes = b"fixture runtime layer\n",
        declared_layer: bytes | None = None,
        extra_file: tuple[str, bytes] | None = None,
        hardlink: bool = False,
        directory_payload: bytes = b"",
        config_padding: int = 0,
        omit_index_media_type: bool = False,
        omit_manifest_media_type: bool = False,
        omit_index_platform: bool = False,
    ) -> OciFixture:
        config_document: dict[str, object] = {
            "architecture": "amd64",
            "config": {"Entrypoint": ["/usr/local/bin/riley-server"]},
            "os": "linux",
            "rootfs": {"diff_ids": [], "type": "layers"},
        }
        if config_padding:
            config_document["padding"] = "x" * config_padding
        config = _raw_json(config_document)
        declared_layer = actual_layer if declared_layer is None else declared_layer
        config_descriptor = {
            "mediaType": prepare.OCI_CONFIG_MEDIA_TYPE,
            "digest": _digest(config),
            "size": len(config),
        }
        layer_descriptor = {
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
            "digest": _digest(declared_layer),
            "size": len(declared_layer),
        }
        manifest_document: dict[str, object] = {
            "schemaVersion": 2,
            "config": config_descriptor,
            "layers": [layer_descriptor],
        }
        if not omit_manifest_media_type:
            manifest_document["mediaType"] = prepare.OCI_MANIFEST_MEDIA_TYPE
        manifest = _raw_json(manifest_document)
        manifest_descriptor = {
            "mediaType": prepare.OCI_MANIFEST_MEDIA_TYPE,
            "digest": _digest(manifest),
            "size": len(manifest),
        }
        if not omit_index_platform:
            manifest_descriptor["platform"] = dict(prepare.PLATFORM if index_platform is None else index_platform)
        manifests = [manifest_descriptor]
        if duplicate_index_manifest:
            manifests.append(dict(manifest_descriptor))
        index_document: dict[str, object] = {"schemaVersion": 2, "manifests": manifests}
        if not omit_index_media_type:
            index_document["mediaType"] = prepare.OCI_INDEX_MEDIA_TYPE
        index = _raw_json(index_document)
        layout = _raw_json({"imageLayoutVersion": "1.0.0"})
        archive_path = self.base / f"{name}.tar"
        with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            self._add_directory(archive, "blobs", directory_payload)
            self._add_directory(archive, "blobs/sha256")
            self._add_file(archive, "oci-layout", layout)
            self._add_file(archive, "index.json", index)
            self._add_file(archive, "blobs/sha256/" + _digest(manifest)[7:], manifest)
            self._add_file(archive, "blobs/sha256/" + _digest(config)[7:], config)
            if include_layer:
                self._add_file(archive, "blobs/sha256/" + _digest(actual_layer)[7:], actual_layer)
            if extra_file is not None:
                self._add_file(archive, extra_file[0], extra_file[1])
            if hardlink:
                link = tarfile.TarInfo("blobs/sha256/" + "f" * 64)
                link.type = tarfile.LNKTYPE
                link.linkname = "oci-layout"
                link.mode = 0o600
                link.mtime = 0
                archive.addfile(link)
        os.chmod(archive_path, 0o600)
        image_id = _digest(config) if inspect_id is None else inspect_id
        inspect_path = self.base / f"{name}-inspect.json"
        raw = inspect_raw
        if raw is None:
            raw = _raw_json(
                [{"Architecture": "amd64", "Id": image_id, "Os": "linux", "RepoTags": []}]
            )
        inspect_path.write_bytes(raw)
        os.chmod(inspect_path, 0o600)
        return OciFixture(inspect=inspect_path, archive=archive_path, image_id=image_id)

    def _prepare(self, fixture: OciFixture, root: Path, arm: str = "a") -> dict[str, object]:
        return prepare.prepare_reconstructed_runtime_oci_inputs(
            root,
            image_inspect=fixture.inspect,
            oci_archive=fixture.archive,
            reconstruction_id=arm,
        )

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(prepare.RuntimeOciInputsError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_prepares_and_replays_a_single_runtime_oci_content_closure(self) -> None:
        fixture = self._write_fixture()
        root = self._root()
        receipt = self._prepare(fixture, root)
        self.assertEqual(receipt["schema_version"], prepare.RUNTIME_OCI_INPUTS_VERSION)
        self.assertEqual(receipt["status"], "prepared")
        self.assertEqual(receipt["qualification_status"], "not-run")
        self.assertEqual(receipt["reconstruction_id"], "a")
        self.assertEqual(receipt["image_id"], fixture.image_id)
        self.assertEqual(receipt["content_binding"], "validated")
        self.assertEqual(receipt["source_binding"], "not-established")
        self.assertEqual(prepare.verify_reconstructed_runtime_oci_inputs(root), receipt)
        self.assertEqual(set(os.listdir(root)), {prepare.RUNTIME_IMAGE_DIRECTORY_NAME, prepare.RUNTIME_OCI_INPUTS_NAME})
        image_directory = root / prepare.RUNTIME_IMAGE_DIRECTORY_NAME
        self.assertEqual(
            set(os.listdir(image_directory)),
            {name for _field, name, _label, _maximum in prepare.RUNTIME_LEAVES},
        )
        for directory in (root, image_directory):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        for path in image_directory.iterdir():
            metadata = path.stat()
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(metadata.st_nlink, 1)

    def test_held_snapshot_callback_returns_only_after_archive_digest_validation(self) -> None:
        fixture = self._write_fixture()
        root = self._root()
        receipt = self._prepare(fixture, root)
        root_fd = common.open_private_evidence_directory(root, "test root")
        try:
            image_fd = common.open_private_child_directory(root_fd, prepare.RUNTIME_IMAGE_DIRECTORY_NAME, "test image")
            try:
                descriptor = common.parse_descriptor(receipt["archive"], "archive descriptor")
                held = common.rebase_descriptor_to_held_leaf(
                    descriptor,
                    expected_root_relative_path=f"{prepare.RUNTIME_IMAGE_DIRECTORY_NAME}/{prepare.OCI_ARCHIVE_NAME}",
                    leaf_name=prepare.OCI_ARCHIVE_NAME,
                    label="archive descriptor",
                )
                prefix = common.consume_private_snapshot_descriptor_file(
                    image_fd,
                    held,
                    "test archive",
                    lambda source: source.read(16),
                    maximum_bytes=prepare.MAX_OCI_ARCHIVE_BYTES,
                )
            finally:
                os.close(image_fd)
        finally:
            os.close(root_fd)
        self.assertEqual(prefix, fixture.archive.read_bytes()[:16])

    def test_rejects_inspect_config_id_mismatch_without_a_receipt(self) -> None:
        fixture = self._write_fixture(inspect_id="sha256:" + "1" * 64)
        root = self._root()
        self.assert_reason("runtime-image-id-mismatch", lambda: self._prepare(fixture, root))
        self.assertFalse((root / prepare.RUNTIME_OCI_INPUTS_NAME).exists())

    def test_rejects_arm64_or_multi_manifest_indexes_without_a_receipt(self) -> None:
        arm = self._write_fixture("arm", index_platform={"os": "linux", "architecture": "arm64"})
        self.assert_reason("oci-platform-mismatch", lambda: self._prepare(arm, self._root("arm-root")))
        multi = self._write_fixture("multi", duplicate_index_manifest=True)
        self.assert_reason("invalid-oci-index", lambda: self._prepare(multi, self._root("multi-root")))

    def test_accepts_optional_oci_index_and_manifest_metadata_but_checks_declared_platform(self) -> None:
        optional = self._write_fixture(
            "optional-metadata",
            omit_index_media_type=True,
            omit_manifest_media_type=True,
            omit_index_platform=True,
        )
        receipt = self._prepare(optional, self._root("optional-root"))
        self.assertEqual(receipt["image_id"], optional.image_id)
        declared_arm = self._write_fixture(
            "declared-arm", index_platform={"os": "linux", "architecture": "arm64"}
        )
        self.assert_reason("oci-platform-mismatch", lambda: self._prepare(declared_arm, self._root("declared-arm-root")))

    def test_rejects_missing_and_digest_mismatched_layer_blobs(self) -> None:
        missing = self._write_fixture("missing", include_layer=False)
        self.assert_reason("oci-blob-closure-mismatch", lambda: self._prepare(missing, self._root("missing-root")))
        mismatch = self._write_fixture(
            "mismatch", actual_layer=b"actual layer", declared_layer=b"declared layer"
        )
        self.assert_reason("oci-blob-closure-mismatch", lambda: self._prepare(mismatch, self._root("mismatch-root")))

    def test_rejects_docker_save_masquerade_and_unsafe_tar_members(self) -> None:
        fixture = self._write_fixture("save")
        with tarfile.open(fixture.archive, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            self._add_file(archive, "manifest.json", b"[]")
        os.chmod(fixture.archive, 0o600)
        self.assert_reason("unsafe-oci-member", lambda: self._prepare(fixture, self._root("save-root")))

        traversal = self._write_fixture("traversal", extra_file=("../escape", b"escape"))
        self.assert_reason("unsafe-oci-member", lambda: self._prepare(traversal, self._root("traversal-root")))
        hardlink = self._write_fixture("hardlink", hardlink=True)
        self.assert_reason(
            "unsupported-oci-tar-extension", lambda: self._prepare(hardlink, self._root("hardlink-root"))
        )

        compressed = self._write_fixture("compressed")
        compressed.archive.write_bytes(gzip.compress(compressed.archive.read_bytes()))
        os.chmod(compressed.archive, 0o600)
        self.assert_reason("invalid-oci-tar", lambda: self._prepare(compressed, self._root("compressed-root")))
        trailing = self._write_fixture("trailing")
        with trailing.archive.open("ab") as destination:
            destination.write(b"x" * 512)
        os.chmod(trailing.archive, 0o600)
        self.assert_reason("invalid-oci-tar", lambda: self._prepare(trailing, self._root("trailing-root")))

    def test_tarfile_never_receives_gnu_or_pax_extension_headers(self) -> None:
        for name, type_flag in (("pax", ord("x")), ("gnu-longname", ord("L")), ("sparse", ord("S"))):
            with self.subTest(name=name):
                fixture = self._write_fixture(name)
                raw = bytearray(fixture.archive.read_bytes())
                raw[prepare.TAR_TYPE_OFFSET] = type_flag
                fixture.archive.write_bytes(raw)
                os.chmod(fixture.archive, 0o600)
                with mock.patch.object(
                    prepare.tarfile,
                    "open",
                    side_effect=AssertionError("extension headers must fail before tarfile parses them"),
                ):
                    self.assert_reason(
                        "unsupported-oci-tar-extension",
                        lambda fixture=fixture, name=name: self._prepare(
                            fixture, self._root(f"{name}-root")
                        ),
                    )

    def test_rejects_excess_members_before_accepting_a_large_tar_inventory(self) -> None:
        fixture = self._write_fixture("many-members")
        with mock.patch.object(prepare, "MAX_OCI_MEMBERS", 3):
            self.assert_reason("invalid-oci-tar", lambda: self._prepare(fixture, self._root("many-root")))
        payload_directory = self._write_fixture("payload-directory", directory_payload=b"not empty")
        self.assert_reason(
            "invalid-oci-tar",
            lambda: self._prepare(payload_directory, self._root("payload-directory-root")),
        )

    def test_rejects_manifest_and_config_json_before_retaining_oversized_members(self) -> None:
        manifest_fixture = self._write_fixture("large-manifest")
        with tarfile.open(manifest_fixture.archive, mode="r:") as archive:
            index = json.loads(archive.extractfile("index.json").read())  # type: ignore[union-attr]
        manifest_limit = index["manifests"][0]["size"] - 1
        with mock.patch.object(prepare, "MAX_RECEIPT_BYTES", manifest_limit):
            self.assert_reason(
                "oci-json-size",
                lambda: self._prepare(manifest_fixture, self._root("large-manifest-root")),
            )

        config_fixture = self._write_fixture("large-config", config_padding=4096)
        with tarfile.open(config_fixture.archive, mode="r:") as archive:
            index = json.loads(archive.extractfile("index.json").read())  # type: ignore[union-attr]
            manifest_name = "blobs/sha256/" + index["manifests"][0]["digest"][7:]
            manifest = json.loads(archive.extractfile(manifest_name).read())  # type: ignore[union-attr]
        config_limit = manifest["config"]["size"] - 1
        with mock.patch.object(prepare, "MAX_RECEIPT_BYTES", config_limit):
            self.assert_reason(
                "oci-json-size",
                lambda: self._prepare(config_fixture, self._root("large-config-root")),
            )

    def test_rejects_duplicate_json_keys_before_a_receipt(self) -> None:
        fixture = self._write_fixture(
            "duplicate-json",
            inspect_raw=(
                b'[{"Id":"sha256:' + b"2" * 64 + b'","Id":"sha256:' + b"2" * 64
                + b'","Os":"linux","Architecture":"amd64"}]'
            ),
        )
        self.assert_reason("duplicate-json-key", lambda: self._prepare(fixture, self._root("duplicate-root")))

    def test_replay_rejects_archive_and_derived_snapshot_mutation(self) -> None:
        fixture = self._write_fixture()
        root = self._root()
        self._prepare(fixture, root)
        archive = root / prepare.RUNTIME_IMAGE_DIRECTORY_NAME / prepare.OCI_ARCHIVE_NAME
        archive.write_bytes(b"not an OCI archive")
        os.chmod(archive, 0o600)
        with self.assertRaises(prepare.RuntimeOciInputsError):
            prepare.verify_reconstructed_runtime_oci_inputs(root)

        fixture = self._write_fixture("derived")
        derived_root = self._root("derived-root")
        self._prepare(fixture, derived_root)
        config = derived_root / prepare.RUNTIME_IMAGE_DIRECTORY_NAME / prepare.OCI_CONFIG_NAME
        config.write_bytes(b"{}")
        os.chmod(config, 0o600)
        with self.assertRaises(prepare.RuntimeOciInputsError):
            prepare.verify_reconstructed_runtime_oci_inputs(derived_root)

    def test_replay_rejects_extra_entries_and_fixed_descriptor_path_drift(self) -> None:
        fixture = self._write_fixture()
        root = self._root()
        receipt = self._prepare(fixture, root)
        extra = root / "extra"
        extra.write_bytes(b"extra")
        os.chmod(extra, 0o600)
        self.assert_reason("unexpected-evidence-entry", lambda: prepare.verify_reconstructed_runtime_oci_inputs(root))

        drift_root = self._root("drift")
        receipt = self._prepare(fixture, drift_root)
        receipt["config"]["path"] = "runtime-image/other.json"  # type: ignore[index]
        receipt_path = drift_root / prepare.RUNTIME_OCI_INPUTS_NAME
        receipt_path.write_bytes(common.canonical_json_bytes(receipt))
        os.chmod(receipt_path, 0o600)
        self.assert_reason("runtime-oci-leaf-path-mismatch", lambda: prepare.verify_reconstructed_runtime_oci_inputs(drift_root))

    def test_create_only_and_input_output_alias_rules(self) -> None:
        fixture = self._write_fixture()
        root = self._root()
        receipt = self._prepare(fixture, root)
        with self.assertRaises(prepare.RuntimeOciInputsError):
            self._prepare(fixture, root)
        self.assertEqual(prepare.verify_reconstructed_runtime_oci_inputs(root), receipt)
        self.assert_reason(
            "output-input-alias",
            lambda: prepare.prepare_reconstructed_runtime_oci_inputs(
                fixture.inspect,
                image_inspect=fixture.inspect,
                oci_archive=fixture.archive,
                reconstruction_id="a",
            ),
        )

    def test_cli_help_and_schema_define_a_closed_source_only_contract(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(prepare.__file__).resolve()), "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--reconstruction-id", completed.stdout)
        schema_path = (
            Path(prepare.__file__).resolve().parents[2]
            / "benchmarks"
            / "release"
            / "candidates"
            / "reconstructed-runtime-oci-inputs-v1.schema.json"
        )
        if not schema_path.is_file():
            schema_path = Path(prepare.__file__).with_name("reconstructed-runtime-oci-inputs-v1.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(schema["properties"]["schema_version"]["const"], prepare.RUNTIME_OCI_INPUTS_VERSION)
        self.assertEqual(schema["properties"]["qualification_status"]["const"], "not-run")
        self.assertEqual(schema["properties"]["content_binding"]["const"], "validated")
        self.assertEqual(
            schema["$defs"]["archiveDescriptor"]["allOf"][1]["properties"]["path"]["const"],
            f"runtime-image/{prepare.OCI_ARCHIVE_NAME}",
        )
        self.assertIn("not a reconstructed baseline", schema["description"])

    def test_producer_is_capture_only_and_never_extracts_or_runs_workloads(self) -> None:
        source = Path(prepare.__file__).read_text(encoding="utf-8")
        self.assertIn("consume_private_snapshot_descriptor_file", source)
        self.assertIn('mode="r:"', source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)
        self.assertNotIn("os.system(", source)
        self.assertNotIn("extractall(", source)
        self.assertNotIn(".extract(", source)
        self.assertNotIn('"cargo"', source)
        self.assertNotIn('"nvidia-smi"', source)
        self.assertNotIn('"ssh"', source)


if __name__ == "__main__":
    unittest.main()
