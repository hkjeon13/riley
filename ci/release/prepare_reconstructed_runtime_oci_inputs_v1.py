#!/usr/bin/env python3
"""Prepare one source-only OCI runtime-image content input closure.

This tool never invokes a container runtime, compiler, GPU, service, or
qualification gate.  It accepts two already-captured raw files: a one-image
runtime inspect response and one uncompressed OCI image-layout tar.  It
creates a fresh private evidence root, snapshots those raw inputs, verifies
the OCI index/manifest/config/layer content binding through a held descriptor,
and stores the exact selected JSON members for a later A/B materializer.

Docker-save archives are deliberately outside this contract.  A later
consumer may combine two independently prepared roots, but this per-image
receipt does not claim a source, bundle, build invocation, independence, or
qualification result.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

sys.dont_write_bytecode = True

import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, NoReturn, Sequence

import provenance_v2_common as common


RUNTIME_OCI_INPUTS_VERSION = "riley.reconstructed-runtime-oci-inputs.v1"
RUNTIME_OCI_INPUTS_NAME = "reconstructed-runtime-oci-inputs.json"
RUNTIME_IMAGE_DIRECTORY_NAME = "runtime-image"
IMAGE_INSPECT_NAME = "docker-image-inspect.json"
OCI_ARCHIVE_NAME = "oci-image-layout.tar"
OCI_LAYOUT_NAME = "oci-layout"
OCI_INDEX_NAME = "index.json"
OCI_MANIFEST_NAME = "manifest.json"
OCI_CONFIG_NAME = "config.json"
ARCHIVE_FORMAT = "oci-image-layout-tar.v1"
CAPTURE_SCOPE = "single-runtime-oci-layout-content-binding"
CONTENT_BINDING = "validated"
NOT_ESTABLISHED = "not-established"
PLATFORM = {"os": "linux", "architecture": "amd64"}

OCI_INDEX_MEDIA_TYPE = "application/vnd.oci.image.index.v1+json"
OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json"
OCI_LAYER_MEDIA_TYPES = {
    "application/vnd.oci.image.layer.v1.tar",
    "application/vnd.oci.image.layer.v1.tar+gzip",
    "application/vnd.oci.image.layer.v1.tar+zstd",
    "application/vnd.oci.image.layer.nondistributable.v1.tar",
    "application/vnd.oci.image.layer.nondistributable.v1.tar+gzip",
    "application/vnd.oci.image.layer.nondistributable.v1.tar+zstd",
}

MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES
MAX_OCI_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
MAX_OCI_MEMBER_BYTES = 32 * 1024 * 1024 * 1024
MAX_OCI_TOTAL_MEMBER_BYTES = 60 * 1024 * 1024 * 1024
MAX_OCI_MEMBERS = 4096
TAR_BLOCK_BYTES = 512
MIN_TAR_END_BYTES = 2 * TAR_BLOCK_BYTES
TAR_TYPE_OFFSET = 156
TAR_SIZE_START = 124
TAR_SIZE_END = 136
SHA256_DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
BLOB_PATH_RE = re.compile(r"^blobs/sha256/([0-9a-f]{64})$")

RUNTIME_LEAVES = (
    ("image_inspect", IMAGE_INSPECT_NAME, "raw runtime image inspect", MAX_RECEIPT_BYTES),
    ("archive", OCI_ARCHIVE_NAME, "raw OCI image-layout archive", MAX_OCI_ARCHIVE_BYTES),
    ("layout", OCI_LAYOUT_NAME, "raw OCI layout header", MAX_RECEIPT_BYTES),
    ("index", OCI_INDEX_NAME, "raw OCI index", MAX_RECEIPT_BYTES),
    ("manifest", OCI_MANIFEST_NAME, "raw OCI manifest", MAX_RECEIPT_BYTES),
    ("config", OCI_CONFIG_NAME, "raw OCI config", MAX_RECEIPT_BYTES),
)


class RuntimeOciInputsError(common.ProvenanceV2Error):
    """The captured runtime OCI inputs cannot be prepared or replayed."""


def _fail(code: str, message: str) -> NoReturn:
    error = RuntimeOciInputsError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Any) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _normalized_absolute_path(value: Path, label: str) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\x00" in raw or not os.path.isabs(raw):
        _fail("invalid-absolute-path", f"{label} must be a normalized absolute path")
    if "\n" in raw or "\r" in raw:
        _fail("invalid-absolute-path", f"{label} must be a single-line path")
    if raw.startswith("//") or os.path.normpath(raw) != raw or raw == os.path.sep:
        _fail("non-normalized-absolute-path", f"{label} must be a normalized non-root absolute path")
    path = Path(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail("non-normalized-absolute-path", f"{label} must not contain traversal components")
    return path


def _require_external_output(
    evidence_root: Path,
    image_inspect: Path,
    oci_archive: Path,
) -> None:
    if image_inspect == oci_archive:
        _fail("input-alias", "runtime inspect and OCI archive inputs must be distinct files")
    if evidence_root in {image_inspect, oci_archive}:
        _fail("output-input-alias", "--evidence-root must not name an input file")
    for source, label in ((image_inspect, "runtime inspect"), (oci_archive, "OCI archive")):
        if evidence_root in source.parents:
            _fail("output-contains-input", f"--evidence-root must not contain the {label} input")


def _exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else []
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; expected={sorted(expected)}, actual={actual}",
        )
    return value


def _sha256_digest(value: Any, label: str) -> str:
    if type(value) is not str:
        _fail("invalid-oci-digest", f"{label} must be a lowercase sha256 digest")
    match = SHA256_DIGEST_RE.fullmatch(value)
    if match is None or match.group(1) == "0" * 64:
        _fail("invalid-oci-digest", f"{label} must be a non-zero lowercase sha256 digest")
    return value


def _descriptor_size(value: Any, label: str) -> int:
    if type(value) is not int or value < 1 or value > MAX_OCI_MEMBER_BYTES:
        _fail("invalid-oci-descriptor", f"{label} must be a bounded positive integer")
    return value


@dataclass(frozen=True)
class OciBlobDescriptor:
    media_type: str
    digest: str
    byte_length: int


@dataclass(frozen=True)
class ImageInspect:
    image_id: str


@dataclass(frozen=True)
class OciArchiveFacts:
    image_id: str
    layout_raw: bytes
    index_raw: bytes
    manifest_raw: bytes
    config_raw: bytes


def _parse_runtime_image_inspect(raw: bytes, label: str) -> ImageInspect:
    document = _common(
        lambda: common.parse_strict_json(
            raw,
            label,
            maximum_bytes=MAX_RECEIPT_BYTES,
            require_object=False,
        )
    )
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        _fail("invalid-runtime-image-inspect", f"{label} must be a one-image inspect array")
    row = document[0]
    image_id = _sha256_digest(row.get("Id"), f"{label}[0].Id")
    if row.get("Os") != PLATFORM["os"] or row.get("Architecture") != PLATFORM["architecture"]:
        _fail("runtime-image-platform-mismatch", f"{label} must describe linux/amd64")
    return ImageInspect(image_id=image_id)


def _safe_oci_member_name(name: str, label: str, *, directory: bool) -> str:
    if type(name) is not str or not name or "\x00" in name or name.startswith("/"):
        _fail("unsafe-oci-member", f"{label} has an unsafe archive path")
    if "\\" in name or "//" in name:
        _fail("unsafe-oci-member", f"{label} has a non-normalized archive path")
    normalized = name.rstrip("/") if directory else name
    if not normalized or (not directory and name.endswith("/")):
        _fail("unsafe-oci-member", f"{label} has an invalid archive path")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        _fail("unsafe-oci-member", f"{label} has a traversal archive path")
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail("unsafe-oci-member", f"{label} has a traversal archive path")
    return normalized


def _tar_data_end(member: tarfile.TarInfo) -> int:
    if member.offset_data < 0 or member.size < 0:
        _fail("invalid-oci-tar", "OCI archive has an invalid member offset or size")
    padded = ((member.size + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES) * TAR_BLOCK_BYTES
    return member.offset_data + padded


def _parse_tar_size(raw: bytes, label: str) -> int:
    if len(raw) != TAR_SIZE_END - TAR_SIZE_START:
        _fail("invalid-oci-tar", f"{label} has an invalid tar size field")
    if raw and raw[0] & 0x80:
        _fail("unsupported-oci-tar-extension", f"{label} uses a non-octal tar size field")
    text = raw.rstrip(b"\x00 ").lstrip(b" ")
    if not text:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in text):
        _fail("invalid-oci-tar", f"{label} has an invalid tar size field")
    return int(text, 8)


def _preflight_oci_tar(stream: BinaryIO) -> None:
    """Reject extension headers before :mod:`tarfile` can materialize them.

    ``tarfile`` resolves GNU long names and PAX headers inside ``next()`` and
    may retain their payloads before a caller sees a ``TarInfo``.  This raw,
    bounded header pass accepts only ordinary regular files and zero-payload
    directories, so those unbounded extension payloads cannot reach tarfile.
    Tarfile subsequently validates checksums and exposes the same safe member
    grammar for the semantic OCI checks below.
    """

    try:
        stream.seek(0, os.SEEK_END)
        archive_size = stream.tell()
        stream.seek(0, os.SEEK_SET)
    except OSError as error:
        _fail("invalid-oci-tar", f"cannot inspect OCI archive before parsing: {error}")
    if archive_size < MIN_TAR_END_BYTES or archive_size > MAX_OCI_ARCHIVE_BYTES:
        _fail("invalid-oci-tar", "OCI archive has an invalid byte length")
    if archive_size % TAR_BLOCK_BYTES:
        _fail("invalid-oci-tar", "OCI archive size must be tar-block aligned")
    offset = 0
    member_count = 0
    total_size = 0
    zero_block = b"\x00" * TAR_BLOCK_BYTES
    while offset < archive_size:
        try:
            header = stream.read(TAR_BLOCK_BYTES)
        except OSError as error:
            _fail("invalid-oci-tar", f"cannot read OCI tar header: {error}")
        if len(header) != TAR_BLOCK_BYTES:
            _fail("truncated-oci-tar", "OCI archive header is truncated")
        offset += TAR_BLOCK_BYTES
        if header == zero_block:
            try:
                second = stream.read(TAR_BLOCK_BYTES)
            except OSError as error:
                _fail("invalid-oci-tar", f"cannot read OCI tar end marker: {error}")
            if second != zero_block:
                _fail("invalid-oci-tar", "OCI archive has an invalid tar end marker")
            offset += TAR_BLOCK_BYTES
            remaining = archive_size - offset
            while remaining:
                try:
                    trailing = stream.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                except OSError as error:
                    _fail("invalid-oci-tar", f"cannot read OCI tar trailer: {error}")
                if not trailing:
                    _fail("truncated-oci-tar", "OCI archive trailer is truncated")
                if any(trailing):
                    _fail("invalid-oci-tar", "OCI archive has nonzero trailing bytes")
                remaining -= len(trailing)
            return
        member_type = header[TAR_TYPE_OFFSET : TAR_TYPE_OFFSET + 1]
        if member_type not in {b"\x00", b"0", b"5"}:
            _fail("unsupported-oci-tar-extension", "OCI archive has a non-regular tar member type")
        member_size = _parse_tar_size(header[TAR_SIZE_START:TAR_SIZE_END], "OCI tar member")
        if member_type == b"5" and member_size != 0:
            _fail("invalid-oci-tar", "OCI archive directories must not carry payload bytes")
        if member_type != b"5":
            if member_size > MAX_OCI_MEMBER_BYTES:
                _fail("oci-member-size", "OCI archive member exceeds its byte bound")
            total_size += member_size
            if total_size > MAX_OCI_TOTAL_MEMBER_BYTES:
                _fail("oci-total-size", "OCI archive members exceed their total byte bound")
        member_count += 1
        if member_count > MAX_OCI_MEMBERS:
            _fail("invalid-oci-tar", "OCI archive contains too many members")
        padded_size = ((member_size + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES) * TAR_BLOCK_BYTES
        if padded_size > archive_size - offset:
            _fail("truncated-oci-tar", "OCI archive member payload is truncated")
        try:
            stream.seek(padded_size, os.SEEK_CUR)
        except OSError as error:
            _fail("invalid-oci-tar", f"cannot skip OCI tar member payload: {error}")
        offset += padded_size
    _fail("truncated-oci-tar", "OCI archive has no two-block tar end marker")


def _read_member(
    stream: BinaryIO,
    member: tarfile.TarInfo,
    label: str,
    *,
    maximum_bytes: int,
    retain: bool,
) -> tuple[bytes | None, str]:
    if member.size < 1 or member.size > maximum_bytes:
        _fail("oci-member-size", f"{label} is empty or exceeds its byte bound")
    try:
        stream.seek(member.offset_data, os.SEEK_SET)
    except OSError as error:
        _fail("invalid-oci-tar", f"cannot seek {label}: {error}")
    digest = hashlib.sha256()
    remaining = member.size
    chunks: list[bytes] | None = [] if retain else None
    while remaining:
        try:
            chunk = stream.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
        except OSError as error:
            _fail("invalid-oci-tar", f"cannot read {label}: {error}")
        if not chunk:
            _fail("truncated-oci-tar", f"{label} is truncated")
        digest.update(chunk)
        if chunks is not None:
            chunks.append(chunk)
        remaining -= len(chunk)
    if chunks is None:
        return None, digest.hexdigest()
    return b"".join(chunks), digest.hexdigest()


def _hash_member(
    stream: BinaryIO,
    member: tarfile.TarInfo,
    expected: OciBlobDescriptor,
    label: str,
    *,
    retain: bool,
) -> bytes | None:
    if member.size != expected.byte_length:
        _fail("oci-descriptor-size-mismatch", f"{label} size differs from its OCI descriptor")
    raw, actual_digest = _read_member(
        stream,
        member,
        label,
        maximum_bytes=expected.byte_length,
        retain=retain,
    )
    if actual_digest != expected.digest[7:]:
        _fail("oci-descriptor-digest-mismatch", f"{label} digest differs from its OCI descriptor")
    return raw


def _parse_oci_descriptor(
    value: Any,
    label: str,
    *,
    media_types: set[str],
) -> OciBlobDescriptor:
    if not isinstance(value, dict):
        _fail("invalid-oci-descriptor", f"{label} must be an object")
    if set(("mediaType", "digest", "size")) - set(value):
        _fail("invalid-oci-descriptor", f"{label} must contain mediaType, digest, and size")
    media_type = value["mediaType"]
    if type(media_type) is not str or media_type not in media_types:
        _fail("invalid-oci-media-type", f"{label}.mediaType is not allowed by this contract")
    return OciBlobDescriptor(
        media_type=media_type,
        digest=_sha256_digest(value["digest"], f"{label}.digest"),
        byte_length=_descriptor_size(value["size"], f"{label}.size"),
    )


def _blob_path(descriptor: OciBlobDescriptor) -> str:
    return "blobs/sha256/" + descriptor.digest[7:]


def _require_bounded_json_descriptor(descriptor: OciBlobDescriptor, label: str) -> OciBlobDescriptor:
    if descriptor.byte_length > MAX_RECEIPT_BYTES:
        _fail("oci-json-size", f"{label} exceeds the bounded JSON-member size")
    return descriptor


def _require_platform(value: Any, label: str) -> None:
    if value != PLATFORM:
        _fail("oci-platform-mismatch", f"{label} must be exactly linux/amd64")


def _parse_oci_archive(stream: BinaryIO, expected_image_id: str) -> OciArchiveFacts:
    """Validate one uncompressed, single-image OCI layout without extraction."""

    _preflight_oci_tar(stream)
    try:
        stream.seek(0, os.SEEK_SET)
        archive = tarfile.open(fileobj=stream, mode="r:")
    except (OSError, tarfile.TarError) as error:
        _fail("invalid-oci-tar", f"OCI archive must be an uncompressed tar: {error}")
    try:
        if archive.pax_headers:
            _fail("unsupported-oci-tar-extension", "OCI archive must not use global PAX headers")
        members: list[tarfile.TarInfo] = []
        while True:
            member = archive.next()
            if member is None:
                break
            if len(members) >= MAX_OCI_MEMBERS:
                _fail("invalid-oci-tar", "OCI archive contains too many members")
            members.append(member)
        if not members:
            _fail("invalid-oci-tar", "OCI archive has no members")
        files: dict[str, tarfile.TarInfo] = {}
        directories: set[str] = set()
        total_size = 0
        last_data_end = 0
        for member in members:
            if member.pax_headers or member.sparse is not None:
                _fail("unsupported-oci-tar-extension", "OCI archive must not use PAX or sparse members")
            if member.isdir():
                name = _safe_oci_member_name(member.name, "OCI directory", directory=True)
                if name not in {"blobs", "blobs/sha256"} or name in directories or name in files:
                    _fail("unsafe-oci-member", "OCI archive has an unexpected or duplicate directory")
                if member.size != 0:
                    _fail("invalid-oci-tar", "OCI archive directories must not carry payload bytes")
                directories.add(name)
                last_data_end = max(last_data_end, _tar_data_end(member))
                continue
            if not member.isreg():
                _fail("unsafe-oci-member", "OCI archive contains a link or special member")
            name = _safe_oci_member_name(member.name, "OCI file", directory=False)
            if name in files or name in directories:
                _fail("duplicate-oci-member", f"OCI archive repeats {name!r}")
            if name not in {OCI_LAYOUT_NAME, OCI_INDEX_NAME} and BLOB_PATH_RE.fullmatch(name) is None:
                _fail("unsafe-oci-member", f"OCI archive has an unexpected member {name!r}")
            if member.size < 1 or member.size > MAX_OCI_MEMBER_BYTES:
                _fail("oci-member-size", f"OCI archive member {name!r} exceeds its byte bound")
            total_size += member.size
            if total_size > MAX_OCI_TOTAL_MEMBER_BYTES:
                _fail("oci-total-size", "OCI archive members exceed their total byte bound")
            files[name] = member
            last_data_end = max(last_data_end, _tar_data_end(member))
        if OCI_LAYOUT_NAME not in files or OCI_INDEX_NAME not in files:
            _fail("not-oci-image-layout", "OCI archive must contain oci-layout and index.json")
        try:
            stream.seek(0, os.SEEK_END)
            archive_size = stream.tell()
        except OSError as error:
            _fail("invalid-oci-tar", f"cannot inspect OCI archive size: {error}")
        if archive_size < last_data_end + MIN_TAR_END_BYTES or archive_size > MAX_OCI_ARCHIVE_BYTES:
            _fail("invalid-oci-tar", "OCI archive has an invalid end-of-archive size")
        if archive_size % TAR_BLOCK_BYTES:
            _fail("invalid-oci-tar", "OCI archive size must be tar-block aligned")
        try:
            stream.seek(last_data_end, os.SEEK_SET)
        except OSError as error:
            _fail("invalid-oci-tar", f"cannot inspect OCI archive trailer: {error}")
        trailing = archive_size - last_data_end
        while trailing:
            try:
                chunk = stream.read(min(common.DEFAULT_READ_CHUNK_BYTES, trailing))
            except OSError as error:
                _fail("invalid-oci-tar", f"cannot read OCI archive trailer: {error}")
            if not chunk:
                _fail("truncated-oci-tar", "OCI archive trailer is truncated")
            if any(chunk):
                _fail("invalid-oci-tar", "OCI archive has nonzero trailing bytes")
            trailing -= len(chunk)

        layout_raw, _layout_digest = _read_member(
            stream, files[OCI_LAYOUT_NAME], "OCI layout header", maximum_bytes=MAX_RECEIPT_BYTES, retain=True
        )
        index_raw, _index_digest = _read_member(
            stream, files[OCI_INDEX_NAME], "OCI index", maximum_bytes=MAX_RECEIPT_BYTES, retain=True
        )
        assert layout_raw is not None and index_raw is not None
        layout = _common(lambda: common.parse_strict_json(layout_raw, "OCI layout header"))
        if not isinstance(layout, dict) or layout.get("imageLayoutVersion") != "1.0.0":
            _fail("invalid-oci-layout", "OCI layout header must declare imageLayoutVersion 1.0.0")
        index = _common(lambda: common.parse_strict_json(index_raw, "OCI index"))
        if (
            not isinstance(index, dict)
            or index.get("schemaVersion") != 2
            or not isinstance(index.get("manifests"), list)
            or len(index["manifests"]) != 1
        ):
            _fail("invalid-oci-index", "OCI index must contain exactly one OCI image manifest")
        if "mediaType" in index and index["mediaType"] != OCI_INDEX_MEDIA_TYPE:
            _fail("invalid-oci-media-type", "OCI index mediaType is not an OCI image index")
        index_descriptor = _require_bounded_json_descriptor(
            _parse_oci_descriptor(
                index["manifests"][0],
                "OCI index manifest descriptor",
                media_types={OCI_MANIFEST_MEDIA_TYPE},
            ),
            "OCI manifest descriptor",
        )
        if "platform" in index["manifests"][0]:
            _require_platform(index["manifests"][0]["platform"], "OCI index manifest platform")
        manifest_path = _blob_path(index_descriptor)
        if manifest_path not in files:
            _fail("missing-oci-blob", "OCI index manifest blob is absent")
        manifest_raw = _hash_member(
            stream, files[manifest_path], index_descriptor, "OCI manifest blob", retain=True
        )
        assert manifest_raw is not None
        manifest = _common(lambda: common.parse_strict_json(manifest_raw, "OCI manifest blob"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schemaVersion") != 2
            or "config" not in manifest
            or not isinstance(manifest.get("layers"), list)
            or not manifest["layers"]
        ):
            _fail("invalid-oci-manifest", "OCI manifest must contain a config and one or more layers")
        if "mediaType" in manifest and manifest["mediaType"] != OCI_MANIFEST_MEDIA_TYPE:
            _fail("invalid-oci-media-type", "OCI manifest mediaType is not an OCI image manifest")
        config_descriptor = _require_bounded_json_descriptor(
            _parse_oci_descriptor(
                manifest["config"],
                "OCI manifest config descriptor",
                media_types={OCI_CONFIG_MEDIA_TYPE},
            ),
            "OCI config descriptor",
        )
        layers = [
            _parse_oci_descriptor(layer, f"OCI manifest layer[{index}]", media_types=OCI_LAYER_MEDIA_TYPES)
            for index, layer in enumerate(manifest["layers"])
        ]
        expected_blob_paths = {manifest_path, _blob_path(config_descriptor)}
        if _blob_path(config_descriptor) == manifest_path:
            _fail("oci-cross-binding-alias", "OCI config must not alias the manifest blob")
        for layer in layers:
            layer_path = _blob_path(layer)
            if layer_path in expected_blob_paths:
                _fail("oci-cross-binding-alias", "OCI layers must use distinct config and manifest blobs")
            expected_blob_paths.add(layer_path)
        actual_blob_paths = {name for name in files if BLOB_PATH_RE.fullmatch(name) is not None}
        if actual_blob_paths != expected_blob_paths:
            _fail("oci-blob-closure-mismatch", "OCI archive blobs must exactly close the selected image")
        config_path = _blob_path(config_descriptor)
        config_raw = _hash_member(stream, files[config_path], config_descriptor, "OCI config blob", retain=True)
        assert config_raw is not None
        config = _common(lambda: common.parse_strict_json(config_raw, "OCI config blob"))
        if not isinstance(config, dict):  # defensive; strict parser already requires an object.
            _fail("invalid-oci-config", "OCI config must be an object")
        if config.get("os") != PLATFORM["os"] or config.get("architecture") != PLATFORM["architecture"]:
            _fail("oci-platform-mismatch", "OCI config must describe linux/amd64")
        config_image_id = "sha256:" + hashlib.sha256(config_raw).hexdigest()
        if config_image_id != config_descriptor.digest or config_image_id != expected_image_id:
            _fail("runtime-image-id-mismatch", "runtime inspect Id must equal the OCI config digest")
        for index, layer in enumerate(layers):
            _hash_member(
                stream,
                files[_blob_path(layer)],
                layer,
                f"OCI layer blob {index}",
                retain=False,
            )
        return OciArchiveFacts(
            image_id=config_image_id,
            layout_raw=layout_raw,
            index_raw=index_raw,
            manifest_raw=manifest_raw,
            config_raw=config_raw,
        )
    except RuntimeOciInputsError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail("invalid-oci-tar", f"cannot parse OCI archive: {error}")
    finally:
        archive.close()


def _assert_entries(directory_fd: int, expected: set[str], label: str) -> None:
    try:
        entries = set(os.listdir(directory_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot list {label}: {error}")
    if entries != expected:
        _fail(
            "unexpected-evidence-entry",
            f"{label} entries differ; expected={sorted(expected)}, actual={sorted(entries)}",
        )


def _rebase_runtime_descriptor(
    descriptor: common.EvidenceDescriptor,
    field: str,
    label: str,
) -> common.EvidenceDescriptor:
    leaf = next(name for candidate, name, _leaf_label, _maximum in RUNTIME_LEAVES if candidate == field)
    return _common(
        lambda: common.rebase_descriptor_to_held_leaf(
            descriptor,
            expected_root_relative_path=f"{RUNTIME_IMAGE_DIRECTORY_NAME}/{leaf}",
            leaf_name=leaf,
            label=label,
        )
    )


def _consume_runtime_bytes(
    runtime_fd: int,
    descriptor: common.EvidenceDescriptor,
    field: str,
    label: str,
) -> bytes:
    maximum = next(limit for candidate, _name, _leaf_label, limit in RUNTIME_LEAVES if candidate == field)
    held = _rebase_runtime_descriptor(descriptor, field, label)

    def read_all(source: BinaryIO) -> bytes:
        chunks: list[bytes] = []
        remaining = maximum
        while True:
            try:
                chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining + 1))
            except OSError as error:
                _fail("unreadable-input", f"cannot read {label}: {error}")
            if not chunk:
                break
            if len(chunk) > remaining:
                _fail("input-too-large", f"{label} exceeds its byte bound")
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw:
            _fail("empty-input", f"{label} must not be empty")
        return raw

    return _common(
        lambda: common.consume_private_snapshot_descriptor_file(
            runtime_fd,
            held,
            label,
            read_all,
            maximum_bytes=maximum,
        )
    )


def _consume_runtime_archive(
    runtime_fd: int,
    descriptor: common.EvidenceDescriptor,
    expected_image_id: str,
) -> OciArchiveFacts:
    held = _rebase_runtime_descriptor(descriptor, "archive", "raw OCI image-layout archive")
    return _common(
        lambda: common.consume_private_snapshot_descriptor_file(
            runtime_fd,
            held,
            "raw OCI image-layout archive",
            lambda source: _parse_oci_archive(source, expected_image_id),
            maximum_bytes=MAX_OCI_ARCHIVE_BYTES,
        )
    )


def _parse_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(
        dict(value),
        {
            "schema_version",
            "status",
            "qualification_status",
            "capture_scope",
            "reconstruction_id",
            "archive_format",
            "platform",
            "image_id",
            "content_binding",
            "source_binding",
            "bundle_binding",
            "build_invocation_binding",
            "independence_binding",
            "image_inspect",
            "archive",
            "layout",
            "index",
            "manifest",
            "config",
        },
        "runtime OCI inputs receipt",
    )
    if (
        row["schema_version"] != RUNTIME_OCI_INPUTS_VERSION
        or row["status"] != "prepared"
        or row["qualification_status"] != "not-run"
        or row["capture_scope"] != CAPTURE_SCOPE
        or row["reconstruction_id"] not in {"a", "b"}
        or row["archive_format"] != ARCHIVE_FORMAT
        or row["platform"] != PLATFORM
        or row["content_binding"] != CONTENT_BINDING
        or any(row[field] != NOT_ESTABLISHED for field in (
            "source_binding", "bundle_binding", "build_invocation_binding", "independence_binding"
        ))
    ):
        _fail("invalid-runtime-oci-receipt", "runtime OCI inputs receipt is not the exact v1 contract")
    image_id = _sha256_digest(row["image_id"], "runtime OCI inputs receipt.image_id")
    descriptors: dict[str, common.EvidenceDescriptor] = {}
    for field, leaf, leaf_label, _maximum in RUNTIME_LEAVES:
        descriptor = _common(lambda field=field, leaf_label=leaf_label: common.parse_descriptor(row[field], leaf_label))
        expected = f"{RUNTIME_IMAGE_DIRECTORY_NAME}/{leaf}"
        if descriptor.path != expected:
            _fail("runtime-oci-leaf-path-mismatch", f"{leaf_label} must use the fixed path {expected!r}")
        descriptors[field] = descriptor
    _common(lambda: common.require_unique_descriptors(tuple(descriptors.values()), "runtime OCI receipt descriptors"))
    return {**row, "image_id": image_id, "_descriptors": descriptors}


def verify_reconstructed_runtime_oci_inputs_fd(root_fd: int) -> dict[str, Any]:
    """Replay one already-held OCI input root without creating output."""

    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "runtime OCI inputs root"))
    _assert_entries(root_fd, {RUNTIME_IMAGE_DIRECTORY_NAME, RUNTIME_OCI_INPUTS_NAME}, "runtime OCI inputs root")
    receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            RUNTIME_OCI_INPUTS_NAME,
            "runtime OCI inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    parsed = _parse_receipt(receipt)
    descriptors = parsed["_descriptors"]
    runtime_fd = _common(
        lambda: common.open_private_child_directory(
            root_fd, RUNTIME_IMAGE_DIRECTORY_NAME, "runtime OCI evidence directory"
        )
    )
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                runtime_fd,
                RUNTIME_IMAGE_DIRECTORY_NAME,
                "held runtime OCI evidence directory",
            )
        )
        _assert_entries(
            runtime_fd,
            {name for _field, name, _label, _maximum in RUNTIME_LEAVES},
            "runtime OCI evidence directory",
        )
        inspect = _parse_runtime_image_inspect(
            _consume_runtime_bytes(runtime_fd, descriptors["image_inspect"], "image_inspect", "raw runtime image inspect"),
            "raw runtime image inspect",
        )
        if inspect.image_id != parsed["image_id"]:
            _fail("runtime-image-id-mismatch", "receipt image_id differs from raw runtime inspect")
        facts = _consume_runtime_archive(runtime_fd, descriptors["archive"], inspect.image_id)
        if facts.image_id != parsed["image_id"]:
            _fail("runtime-image-id-mismatch", "receipt image_id differs from OCI config")
        expected_raw = {
            "layout": facts.layout_raw,
            "index": facts.index_raw,
            "manifest": facts.manifest_raw,
            "config": facts.config_raw,
        }
        for field, raw in expected_raw.items():
            replayed = _consume_runtime_bytes(runtime_fd, descriptors[field], field, f"runtime OCI {field} snapshot")
            if replayed != raw:
                _fail("oci-derived-snapshot-mismatch", f"runtime OCI {field} snapshot differs from the held archive")
        _assert_entries(
            runtime_fd,
            {name for _field, name, _label, _maximum in RUNTIME_LEAVES},
            "runtime OCI evidence directory",
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                runtime_fd,
                RUNTIME_IMAGE_DIRECTORY_NAME,
                "held runtime OCI evidence directory",
            )
        )
    finally:
        os.close(runtime_fd)
    _assert_entries(root_fd, {RUNTIME_IMAGE_DIRECTORY_NAME, RUNTIME_OCI_INPUTS_NAME}, "runtime OCI inputs root")
    terminal_receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            RUNTIME_OCI_INPUTS_NAME,
            "runtime OCI inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    if terminal_receipt != receipt:
        _fail("raced-input", "runtime OCI inputs receipt changed during replay")
    return {key: value for key, value in parsed.items() if key != "_descriptors"}


def verify_reconstructed_runtime_oci_inputs(evidence_root: Path) -> dict[str, Any]:
    evidence_root = _normalized_absolute_path(evidence_root, "runtime OCI inputs root")
    root_fd = _common(
        lambda: common.open_private_evidence_directory(evidence_root, "runtime OCI inputs root")
    )
    try:
        return verify_reconstructed_runtime_oci_inputs_fd(root_fd)
    finally:
        os.close(root_fd)


def prepare_reconstructed_runtime_oci_inputs(
    evidence_root: Path,
    *,
    image_inspect: Path,
    oci_archive: Path,
    reconstruction_id: str,
) -> dict[str, Any]:
    """Create one fresh OCI content-input root and self-replay it."""

    evidence_root = _normalized_absolute_path(evidence_root, "--evidence-root")
    image_inspect = _normalized_absolute_path(image_inspect, "--image-inspect")
    oci_archive = _normalized_absolute_path(oci_archive, "--oci-archive")
    if reconstruction_id not in {"a", "b"}:
        _fail("invalid-reconstruction-id", "--reconstruction-id must be exactly a or b")
    _require_external_output(evidence_root, image_inspect, oci_archive)
    root_fd = _common(
        lambda: common.create_private_evidence_directory(evidence_root, "runtime OCI inputs root")
    )
    runtime_fd: int | None = None
    try:
        runtime_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd, RUNTIME_IMAGE_DIRECTORY_NAME, "runtime OCI evidence directory"
            )
        )
        inspect_snapshot = _common(
            lambda: common.snapshot_absolute_regular_create_only(
                image_inspect,
                runtime_fd,
                IMAGE_INSPECT_NAME,
                "raw runtime image inspect",
                maximum_bytes=MAX_RECEIPT_BYTES,
                minimum_bytes=1,
            )
        )
        archive_snapshot = _common(
            lambda: common.snapshot_absolute_regular_create_only(
                oci_archive,
                runtime_fd,
                OCI_ARCHIVE_NAME,
                "raw OCI image-layout archive",
                maximum_bytes=MAX_OCI_ARCHIVE_BYTES,
                minimum_bytes=1,
            )
        )
        descriptors = {
            "image_inspect": inspect_snapshot.descriptor(
                f"{RUNTIME_IMAGE_DIRECTORY_NAME}/{IMAGE_INSPECT_NAME}", "raw runtime image inspect"
            ),
            "archive": archive_snapshot.descriptor(
                f"{RUNTIME_IMAGE_DIRECTORY_NAME}/{OCI_ARCHIVE_NAME}", "raw OCI image-layout archive"
            ),
        }
        inspect = _parse_runtime_image_inspect(
            _consume_runtime_bytes(runtime_fd, descriptors["image_inspect"], "image_inspect", "raw runtime image inspect"),
            "raw runtime image inspect",
        )
        facts = _consume_runtime_archive(runtime_fd, descriptors["archive"], inspect.image_id)
        for field, leaf, raw, label in (
            ("layout", OCI_LAYOUT_NAME, facts.layout_raw, "raw OCI layout header"),
            ("index", OCI_INDEX_NAME, facts.index_raw, "raw OCI index"),
            ("manifest", OCI_MANIFEST_NAME, facts.manifest_raw, "raw OCI manifest"),
            ("config", OCI_CONFIG_NAME, facts.config_raw, "raw OCI config"),
        ):
            created = _common(lambda leaf=leaf, raw=raw, label=label: common.write_create_only(runtime_fd, leaf, raw, label))
            descriptors[field] = created.descriptor(f"{RUNTIME_IMAGE_DIRECTORY_NAME}/{leaf}", label)
        receipt = {
            "schema_version": RUNTIME_OCI_INPUTS_VERSION,
            "status": "prepared",
            "qualification_status": "not-run",
            "capture_scope": CAPTURE_SCOPE,
            "reconstruction_id": reconstruction_id,
            "archive_format": ARCHIVE_FORMAT,
            "platform": dict(PLATFORM),
            "image_id": facts.image_id,
            "content_binding": CONTENT_BINDING,
            "source_binding": NOT_ESTABLISHED,
            "bundle_binding": NOT_ESTABLISHED,
            "build_invocation_binding": NOT_ESTABLISHED,
            "independence_binding": NOT_ESTABLISHED,
            **{field: descriptor.as_json() for field, descriptor in descriptors.items()},
        }
        _common(
            lambda: common.write_create_only_json(
                root_fd, RUNTIME_OCI_INPUTS_NAME, receipt, "runtime OCI inputs receipt"
            )
        )
        replayed = verify_reconstructed_runtime_oci_inputs_fd(root_fd)
        if replayed != receipt:
            _fail("prepublication-replay-drift", "held runtime OCI replay differs from draft receipt")
        return receipt
    finally:
        if runtime_fd is not None:
            os.close(runtime_fd)
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--image-inspect", type=Path, required=True)
    parser.add_argument("--oci-archive", type=Path, required=True)
    parser.add_argument("--reconstruction-id", choices=("a", "b"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = prepare_reconstructed_runtime_oci_inputs(
            args.evidence_root,
            image_inspect=args.image_inspect,
            oci_archive=args.oci_archive,
            reconstruction_id=args.reconstruction_id,
        )
    except RuntimeOciInputsError as error:
        print(f"runtime OCI inputs: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
