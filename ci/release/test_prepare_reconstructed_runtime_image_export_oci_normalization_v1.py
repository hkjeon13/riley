#!/usr/bin/env python3
"""CPU-only hostile-path tests for runtime image-export OCI normalization v1."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parent))

import prepare_reconstructed_runtime_image_export_oci_normalization_v1 as normalize  # noqa: E402
import prepare_reconstructed_runtime_oci_inputs_v1 as oci_inputs  # noqa: E402
import provenance_v2_common as common  # noqa: E402


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _raw_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class Fixture:
    inspect: Path
    archive: Path
    image_id: str


class RuntimeImageExportNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _add_file(self, archive: tarfile.TarFile, name: str, raw: bytes, *, mtime: int = 0) -> None:
        member = tarfile.TarInfo(name)
        member.size = len(raw)
        member.mode = 0o640
        member.uid = 123
        member.gid = 456
        member.uname = "input-user"
        member.gname = "input-group"
        member.mtime = mtime
        archive.addfile(member, io.BytesIO(raw))

    def _add_directory(self, archive: tarfile.TarFile, name: str, *, mtime: int = 0) -> None:
        member = tarfile.TarInfo(name)
        member.type = tarfile.DIRTYPE
        member.mode = 0o750
        member.uid = 123
        member.gid = 456
        member.uname = "input-user"
        member.gname = "input-group"
        member.mtime = mtime
        archive.addfile(member)

    def _write_inspect(self, name: str, image_id: str) -> Path:
        path = self.base / f"{name}-inspect.json"
        path.write_bytes(_raw_json([{"Id": image_id, "Os": "linux", "Architecture": "amd64", "RepoTags": []}]))
        os.chmod(path, 0o600)
        return path

    def _write_legacy(
        self,
        name: str = "legacy",
        *,
        inspect_id: str | None = None,
        layer_raws: tuple[bytes, ...] = (b"legacy layer one\n", b"legacy layer two\n"),
        diff_ids: list[str] | None = None,
        multi_manifest: bool = False,
        reverse_order: bool = False,
        include_directories: bool = True,
    ) -> Fixture:
        layer_ids = tuple(f"{index + 1:064x}" for index in range(len(layer_raws)))
        config = _raw_json(
            {
                "architecture": "amd64",
                "config": {"Entrypoint": ["/opt/riley/bin/riley"]},
                "os": "linux",
                "rootfs": {
                    "type": "layers",
                    "diff_ids": list(diff_ids if diff_ids is not None else [_digest(raw) for raw in layer_raws]),
                },
            }
        )
        image_id = _digest(config)
        config_name = image_id[7:] + ".json"
        manifest_row = {
            "Config": config_name,
            "RepoTags": None,
            "Layers": [f"{layer_id}/layer.tar" for layer_id in layer_ids],
        }
        manifest = _raw_json([manifest_row] * (2 if multi_manifest else 1))
        entries: list[tuple[str, str, bytes]] = [
            ("file", "manifest.json", manifest),
            ("file", config_name, config),
            ("file", "repositories", b"{}"),
        ]
        for layer_id, layer_raw in zip(layer_ids, layer_raws):
            if include_directories:
                entries.append(("directory", layer_id, b""))
            entries.extend(
                [
                    ("file", f"{layer_id}/VERSION", b"1.0"),
                    ("file", f"{layer_id}/json", _raw_json({"id": layer_id})),
                    ("file", f"{layer_id}/layer.tar", layer_raw),
                ]
            )
        if reverse_order:
            entries.reverse()
        archive = self.base / f"{name}.tar"
        with tarfile.open(archive, mode="w", format=tarfile.USTAR_FORMAT) as output:
            for index, (kind, path, raw) in enumerate(entries):
                if kind == "directory":
                    self._add_directory(output, path, mtime=index + 1)
                else:
                    self._add_file(output, path, raw, mtime=index + 1)
        os.chmod(archive, 0o600)
        return Fixture(
            inspect=self._write_inspect(name, image_id if inspect_id is None else inspect_id),
            archive=archive,
            image_id=image_id,
        )

    def _write_oci(
        self,
        name: str = "oci",
        *,
        compatibility_sidecars: bool = False,
        moby_compatibility_sidecars: bool = False,
        omit_layer_blob: bool = False,
        layer_media_type: str = "application/vnd.oci.image.layer.v1.tar+gzip",
    ) -> Fixture:
        layer = b"OCI source layer bytes\n"
        config = _raw_json(
            {
                "architecture": "amd64",
                "config": {"Entrypoint": ["/opt/riley/bin/riley"]},
                "os": "linux",
                "rootfs": {"type": "layers", "diff_ids": []},
            }
        )
        image_id = _digest(config)
        layer_descriptor = {"mediaType": layer_media_type, "digest": _digest(layer), "size": len(layer)}
        manifest = _raw_json(
            {
                "schemaVersion": 2,
                "mediaType": oci_inputs.OCI_MANIFEST_MEDIA_TYPE,
                "config": {
                    "mediaType": oci_inputs.OCI_CONFIG_MEDIA_TYPE,
                    "digest": image_id,
                    "size": len(config),
                },
                "layers": [layer_descriptor],
            }
        )
        index = _raw_json(
            {
                "schemaVersion": 2,
                "mediaType": oci_inputs.OCI_INDEX_MEDIA_TYPE,
                "manifests": [
                    {
                        "mediaType": oci_inputs.OCI_MANIFEST_MEDIA_TYPE,
                        "digest": _digest(manifest),
                        "size": len(manifest),
                        "platform": dict(normalize.PLATFORM),
                    }
                ],
            }
        )
        archive = self.base / f"{name}.tar"
        with tarfile.open(archive, mode="w", format=tarfile.USTAR_FORMAT) as output:
            self._add_directory(output, "blobs")
            self._add_directory(output, "blobs/sha256")
            self._add_file(output, "oci-layout", _raw_json({"imageLayoutVersion": "1.0.0"}))
            self._add_file(output, "index.json", index)
            self._add_file(output, "blobs/sha256/" + _digest(manifest)[7:], manifest)
            self._add_file(output, "blobs/sha256/" + image_id[7:], config)
            if not omit_layer_blob:
                self._add_file(output, "blobs/sha256/" + _digest(layer)[7:], layer)
            if compatibility_sidecars:
                if moby_compatibility_sidecars:
                    self._add_file(
                        output,
                        "manifest.json",
                        _raw_json(
                            [
                                {
                                    "Config": "blobs/sha256/" + image_id[7:],
                                    "RepoTags": ["riley:runtime-amd64"],
                                    "Layers": ["blobs/sha256/" + _digest(layer)[7:]],
                                }
                            ]
                        ),
                    )
                    self._add_file(output, "repositories", _raw_json({"riley": {"runtime-amd64": image_id[7:]}}))
                else:
                    self._add_file(output, "manifest.json", b"[]")
                    self._add_file(output, "repositories", b"{}")
        os.chmod(archive, 0o600)
        return Fixture(inspect=self._write_inspect(name, image_id), archive=archive, image_id=image_id)

    def _prepare(self, fixture: Fixture, root: Path, arm: str = "a") -> dict[str, object]:
        return normalize.prepare_reconstructed_runtime_image_export_oci_normalization(
            root,
            image_inspect=fixture.inspect,
            image_export_archive=fixture.archive,
            reconstruction_id=arm,
        )

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(normalize.RuntimeImageExportNormalizationError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_prepares_replays_and_integrates_legacy_docker_save(self) -> None:
        fixture = self._write_legacy()
        root = self.base / "normalization"
        receipt = self._prepare(fixture, root)
        self.assertEqual(receipt["schema_version"], normalize.NORMALIZATION_VERSION)
        self.assertEqual(receipt["status"], "prepared")
        self.assertEqual(receipt["qualification_status"], "not-run")
        self.assertEqual(receipt["authority"], normalize.AUTHORITY)
        self.assertEqual(receipt["source_layout"], normalize.SOURCE_LAYOUT_LEGACY)
        self.assertEqual(receipt["image_id"], fixture.image_id)
        self.assertEqual(normalize.verify_reconstructed_runtime_image_export_oci_normalization(root), receipt)
        self.assertEqual(
            set(os.listdir(root)),
            {normalize.RAW_DIRECTORY_NAME, normalize.NORMALIZED_DIRECTORY_NAME, normalize.NORMALIZATION_NAME},
        )
        for directory in (root, root / normalize.RAW_DIRECTORY_NAME, root / normalize.NORMALIZED_DIRECTORY_NAME):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        for directory in (root / normalize.RAW_DIRECTORY_NAME, root / normalize.NORMALIZED_DIRECTORY_NAME):
            for path in directory.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual(path.stat().st_nlink, 1)

        downstream_root = self.base / "oci-inputs"
        downstream = oci_inputs.prepare_reconstructed_runtime_oci_inputs(
            downstream_root,
            image_inspect=root / normalize.RAW_DIRECTORY_NAME / normalize.IMAGE_INSPECT_NAME,
            oci_archive=root / normalize.NORMALIZED_DIRECTORY_NAME / normalize.OCI_ARCHIVE_NAME,
            reconstruction_id="a",
        )
        self.assertEqual(downstream["image_id"], fixture.image_id)

    def test_is_deterministic_and_writes_canonical_ustar_metadata(self) -> None:
        first = self._write_legacy("first", include_directories=False)
        second = self._write_legacy("second", reverse_order=True, include_directories=True)
        first_root = self.base / "first-normalization"
        second_root = self.base / "second-normalization"
        self._prepare(first, first_root)
        self._prepare(second, second_root)
        first_archive = (first_root / normalize.NORMALIZED_DIRECTORY_NAME / normalize.OCI_ARCHIVE_NAME).read_bytes()
        second_archive = (second_root / normalize.NORMALIZED_DIRECTORY_NAME / normalize.OCI_ARCHIVE_NAME).read_bytes()
        self.assertEqual(first_archive, second_archive)
        with tarfile.open(fileobj=io.BytesIO(first_archive), mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name.rstrip("/") for member in members]
            self.assertEqual(names[:4], ["oci-layout", "index.json", "blobs", "blobs/sha256"])
            self.assertEqual(len(names), 8)
            for member in members:
                self.assertEqual(member.uid, 0)
                self.assertEqual(member.gid, 0)
                self.assertEqual(member.uname, "")
                self.assertEqual(member.gname, "")
                self.assertEqual(member.mtime, 0)
                self.assertEqual(member.mode, 0o755 if member.isdir() else 0o644)
        schema_path = (
            Path(normalize.__file__).resolve().parents[2]
            / "benchmarks/release/candidates/reconstructed-runtime-image-export-oci-normalization-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$defs"]["ociArchiveDescriptor"]["allOf"][1]["properties"]["byte_length"]["maximum"],
            normalize.MAX_NORMALIZED_OCI_ARCHIVE_BYTES,
        )
        self.assertEqual(
            schema["$defs"]["imageExportArchiveDescriptor"]["allOf"][1]["properties"]["byte_length"]["maximum"],
            normalize.MAX_IMAGE_EXPORT_ARCHIVE_BYTES,
        )

    def test_accepts_clean_and_nonempty_opaque_sidecar_oci_exports_without_exporting_sidecars(self) -> None:
        clean = self._write_oci("clean")
        clean_receipt = self._prepare(clean, self.base / "clean-normalization")
        self.assertEqual(clean_receipt["source_layout"], normalize.SOURCE_LAYOUT_OCI)

        hybrid = self._write_oci("hybrid", compatibility_sidecars=True, moby_compatibility_sidecars=True)
        root = self.base / "hybrid-normalization"
        hybrid_receipt = self._prepare(hybrid, root)
        self.assertEqual(hybrid_receipt["source_layout"], normalize.SOURCE_LAYOUT_OCI_SIDECARS)
        with tarfile.open(root / normalize.NORMALIZED_DIRECTORY_NAME / normalize.OCI_ARCHIVE_NAME, mode="r:") as archive:
            names = {member.name.rstrip("/") for member in archive.getmembers()}
        self.assertNotIn("manifest.json", names)
        self.assertNotIn("repositories", names)

    def test_rejects_missing_referenced_oci_blob_before_an_output_receipt(self) -> None:
        fixture = self._write_oci("missing-oci-blob", omit_layer_blob=True)
        root = self.base / "missing-oci-blob-root"
        self.assert_reason("missing-oci-blob", lambda: self._prepare(fixture, root))
        self.assertFalse((root / normalize.NORMALIZATION_NAME).exists())

    def test_rejects_identity_multi_manifest_and_legacy_diff_id_mismatch(self) -> None:
        wrong_id = self._write_legacy("wrong-id", inspect_id="sha256:" + "f" * 64)
        self.assert_reason("runtime-image-id-mismatch", lambda: self._prepare(wrong_id, self.base / "wrong-id-root"))

        multi = self._write_legacy("multi", multi_manifest=True)
        self.assert_reason("invalid-docker-save-manifest", lambda: self._prepare(multi, self.base / "multi-root"))

        mismatch = self._write_legacy("diff", diff_ids=["sha256:" + "e" * 64, "sha256:" + "d" * 64])
        self.assert_reason("legacy-layer-diff-id-mismatch", lambda: self._prepare(mismatch, self.base / "diff-root"))

    def test_rejects_missing_selected_legacy_content_before_an_output_receipt(self) -> None:
        config_name = "a" * 64 + ".json"
        layer_path = "b" * 64 + "/layer.tar"
        archive = self.base / "missing-selected-content.tar"
        with tarfile.open(archive, mode="w", format=tarfile.USTAR_FORMAT) as output:
            self._add_file(
                output,
                "manifest.json",
                _raw_json([{"Config": config_name, "RepoTags": [], "Layers": [layer_path]}]),
            )
        os.chmod(archive, 0o600)
        fixture = Fixture(
            inspect=self._write_inspect("missing-selected-content", "sha256:" + "a" * 64),
            archive=archive,
            image_id="sha256:" + "a" * 64,
        )
        root = self.base / "missing-selected-content-root"
        self.assert_reason("image-export-closure-mismatch", lambda: self._prepare(fixture, root))
        self.assertFalse((root / normalize.NORMALIZATION_NAME).exists())

    def test_rejects_archive_extensions_before_tarfile_and_unsafe_members(self) -> None:
        fixture = self._write_legacy("pax")
        raw = bytearray(fixture.archive.read_bytes())
        raw[normalize.TAR_TYPE_OFFSET] = ord("x")
        fixture.archive.write_bytes(raw)
        os.chmod(fixture.archive, 0o600)
        with mock.patch.object(
            normalize.tarfile,
            "open",
            side_effect=AssertionError("PAX header must fail before tarfile is reached"),
        ):
            self.assert_reason("unsupported-image-export-tar-extension", lambda: self._prepare(fixture, self.base / "pax-root"))

        unsafe = self._write_legacy("unsafe")
        with tarfile.open(unsafe.archive, mode="w", format=tarfile.USTAR_FORMAT) as output:
            link = tarfile.TarInfo("manifest.json")
            link.type = tarfile.LNKTYPE
            link.linkname = "elsewhere"
            output.addfile(link)
        os.chmod(unsafe.archive, 0o600)
        self.assert_reason("unsupported-image-export-tar-extension", lambda: self._prepare(unsafe, self.base / "unsafe-root"))

    def test_rejects_oversized_zero_trailer_before_writing_output(self) -> None:
        fixture = self._write_legacy("trailer")
        with fixture.archive.open("ab") as archive:
            archive.write(b"\0" * (normalize.MAX_CANONICAL_TAR_TRAILER_BYTES + normalize.TAR_BLOCK_BYTES))
        os.chmod(fixture.archive, 0o600)
        root = self.base / "trailer-root"
        self.assert_reason("tar-trailer-size", lambda: self._prepare(fixture, root))
        self.assertFalse((root / normalize.NORMALIZATION_NAME).exists())

    def test_rejects_output_alias_and_replay_mutation_or_extra_leaf(self) -> None:
        fixture = self._write_legacy("alias")
        self.assert_reason(
            "output-input-alias",
            lambda: normalize.prepare_reconstructed_runtime_image_export_oci_normalization(
                fixture.inspect,
                image_inspect=fixture.inspect,
                image_export_archive=fixture.archive,
                reconstruction_id="a",
            ),
        )
        root = self.base / "replay-root"
        self._prepare(fixture, root)
        archive = root / normalize.NORMALIZED_DIRECTORY_NAME / normalize.OCI_ARCHIVE_NAME
        archive.write_bytes(b"not an OCI archive")
        os.chmod(archive, 0o600)
        self.assert_reason("evidence-length-mismatch", lambda: normalize.verify_reconstructed_runtime_image_export_oci_normalization(root))

        fixture = self._write_legacy("extra")
        root = self.base / "extra-root"
        self._prepare(fixture, root)
        extra = root / normalize.NORMALIZED_DIRECTORY_NAME / "extra"
        extra.write_bytes(b"extra")
        os.chmod(extra, 0o600)
        self.assert_reason("unexpected-evidence-entry", lambda: normalize.verify_reconstructed_runtime_image_export_oci_normalization(root))

    def test_replay_rejects_rehashed_noncanonical_oci_tar_header(self) -> None:
        fixture = self._write_legacy("noncanonical-header")
        root = self.base / "noncanonical-header-root"
        self._prepare(fixture, root)
        archive = root / normalize.NORMALIZED_DIRECTORY_NAME / normalize.OCI_ARCHIVE_NAME
        raw = bytearray(archive.read_bytes())
        raw[100:108] = b"0000600\0"
        raw[148:156] = b" " * 8
        raw[148:156] = f"{sum(raw[:normalize.TAR_BLOCK_BYTES]):06o}\0 ".encode("ascii")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as parsed:
            self.assertEqual(parsed.getmembers()[0].mode, 0o600)
        archive.write_bytes(raw)
        os.chmod(archive, 0o600)
        receipt_path = root / normalize.NORMALIZATION_NAME
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["oci_archive"]["sha256"] = hashlib.sha256(raw).hexdigest()
        receipt["oci_archive"]["byte_length"] = len(raw)
        receipt_path.write_bytes(common.canonical_json_bytes(receipt))
        os.chmod(receipt_path, 0o600)
        self.assert_reason("noncanonical-oci-tar", lambda: normalize.verify_reconstructed_runtime_image_export_oci_normalization(root))

    def test_source_does_not_invoke_operational_clients_or_extract_archives(self) -> None:
        source = Path(normalize.__file__).read_text(encoding="utf-8")
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
