#!/usr/bin/env python3
"""Normalize one already-captured runtime image export into canonical OCI bytes.

This is a source-only content transformer.  It neither invokes Docker nor
builds an image, starts a container or service, accesses a GPU, or runs a
qualification gate.  It snapshots an existing raw image inspect response and
runtime image export tar into a fresh private root, selects exactly one linux/amd64
image by its raw config digest, and writes a deterministic uncompressed OCI
image-layout USTAR containing only that image's config and layers.

The resulting receipt proves only the selected raw-content conversion.  It
does not prove that Docker save/build ran, that the raw records came from the
same invocation as an assembly capture, or any source, bundle, runtime, GPU,
rollback, freeze, or qualification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import stat
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Mapping, NoReturn, Sequence

sys.dont_write_bytecode = True

import prepare_reconstructed_runtime_oci_inputs_v1 as runtime_oci
import provenance_v2_common as common


NORMALIZATION_VERSION = "riley.reconstructed-runtime-image-export-oci-normalization.v1"
NORMALIZATION_NAME = "reconstructed-runtime-image-export-oci-normalization.json"
RAW_DIRECTORY_NAME = "image-export"
NORMALIZED_DIRECTORY_NAME = "normalized-oci"
IMAGE_INSPECT_NAME = "runtime-image-inspect.json"
IMAGE_EXPORT_ARCHIVE_NAME = "runtime-image-export.tar"
OCI_ARCHIVE_NAME = "oci-image-layout.tar"
OCI_LAYOUT_NAME = "oci-layout"
OCI_INDEX_NAME = "index.json"
OCI_MANIFEST_NAME = "manifest.json"
OCI_CONFIG_NAME = "config.json"

CAPTURE_SCOPE = "single-runtime-image-export-to-canonical-oci-content-normalization"
AUTHORITY = "runtime-image-export-to-canonical-oci-content-normalization-only"
STATUS = "prepared"
QUALIFICATION_STATUS = "not-run"
RAW_ARCHIVE_FORMAT = "runtime-image-export-tar.v1"
OCI_ARCHIVE_FORMAT = "oci-image-layout-tar.v1"
PLATFORM = {"os": "linux", "architecture": "amd64"}
CONTENT_BINDING = "validated"
NOT_ESTABLISHED = "not-established"

SOURCE_LAYOUT_LEGACY = "docker-save-v1"
SOURCE_LAYOUT_OCI = "oci-layout-v1"
SOURCE_LAYOUT_OCI_SIDECARS = "oci-layout-v1-with-opaque-sidecars"
SOURCE_LAYOUTS = {SOURCE_LAYOUT_LEGACY, SOURCE_LAYOUT_OCI, SOURCE_LAYOUT_OCI_SIDECARS}

MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES
# The normalizer's OCI output is intentionally bounded to the same maximum
# single USTAR member that runtime-assembly-capture v1 can later embed.  A
# future larger-image contract must use a directory snapshot rather than PAX.
MAX_USTAR_REGULAR_MEMBER_BYTES = 8 ** 11 - 1
MAX_NORMALIZED_OCI_ARCHIVE_BYTES = MAX_USTAR_REGULAR_MEMBER_BYTES
# A Docker save contains the same selected raw layers plus small legacy or
# compatibility metadata.  Permit a bounded metadata allowance while refusing
# an unbounded archival scratch-volume commitment.
MAX_IMAGE_EXPORT_ARCHIVE_BYTES = MAX_NORMALIZED_OCI_ARCHIVE_BYTES + 64 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = MAX_USTAR_REGULAR_MEMBER_BYTES
MAX_ARCHIVE_MEMBERS = 4096
MAX_CANONICAL_TAR_TRAILER_BYTES = 20 * 512
TAR_BLOCK_BYTES = 512
TAR_RECORD_BYTES = 20 * TAR_BLOCK_BYTES
MIN_TAR_END_BYTES = 2 * TAR_BLOCK_BYTES
TAR_SIZE_START = 124
TAR_SIZE_END = 136
TAR_TYPE_OFFSET = 156

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
OCI_BLOB_PATH_RE = re.compile(r"^blobs/sha256/([0-9a-f]{64})$")
LEGACY_CONFIG_NAME_RE = re.compile(r"^([0-9a-f]{64})\.json$")
LEGACY_LAYER_PATH_RE = re.compile(r"^([0-9a-f]{64})/layer\.tar$")


class RuntimeImageExportNormalizationError(common.ProvenanceV2Error):
    """The raw runtime image export cannot be normalized safely."""


def _fail(code: str, message: str) -> NoReturn:
    error = RuntimeImageExportNormalizationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _oci(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except runtime_oci.RuntimeOciInputsError as error:
        _fail(getattr(error, "reason_code", "invalid-oci-layout"), str(error))


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


def _require_external_output(evidence_root: Path, image_inspect: Path, image_export_archive: Path) -> None:
    if image_inspect == image_export_archive:
        _fail("input-alias", "runtime inspect and image export archive inputs must be distinct files")
    if evidence_root in {image_inspect, image_export_archive}:
        _fail("output-input-alias", "--evidence-root must not name an input file")
    for source, label in ((image_inspect, "runtime inspect"), (image_export_archive, "image export archive")):
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


def _digest(value: Any, label: str) -> str:
    if type(value) is not str:
        _fail("invalid-oci-digest", f"{label} must be a lowercase sha256 digest")
    match = OCI_DIGEST_RE.fullmatch(value)
    if match is None or match.group(1) == "0" * 64:
        _fail("invalid-oci-digest", f"{label} must be a non-zero lowercase sha256 digest")
    return value


def _digest_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _descriptor_size(value: Any, label: str) -> int:
    if type(value) is not int or value < 1 or value > MAX_ARCHIVE_MEMBER_BYTES:
        _fail("invalid-oci-descriptor", f"{label} must be a bounded positive integer")
    return value


def _safe_archive_member_name(name: str, label: str, *, directory: bool) -> str:
    if type(name) is not str or not name or "\x00" in name or name.startswith("/"):
        _fail("unsafe-image-export-member", f"{label} has an unsafe archive path")
    if "\\" in name or "//" in name:
        _fail("unsafe-image-export-member", f"{label} has a non-normalized archive path")
    normalized = name.rstrip("/") if directory else name
    if not normalized or (not directory and name.endswith("/")):
        _fail("unsafe-image-export-member", f"{label} has an invalid archive path")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        _fail("unsafe-image-export-member", f"{label} has a traversal archive path")
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail("unsafe-image-export-member", f"{label} has a traversal archive path")
    return normalized


def _parse_tar_size(raw: bytes, label: str) -> int:
    if len(raw) != TAR_SIZE_END - TAR_SIZE_START:
        _fail("invalid-image-export-tar", f"{label} has an invalid tar size field")
    if raw and raw[0] & 0x80:
        _fail("unsupported-image-export-tar-extension", f"{label} uses a non-octal tar size field")
    text = raw.rstrip(b"\x00 ").lstrip(b" ")
    if not text:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in text):
        _fail("invalid-image-export-tar", f"{label} has an invalid tar size field")
    return int(text, 8)


def _padded_size(size: int) -> int:
    return ((size + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES) * TAR_BLOCK_BYTES


def _preflight_image_export_tar(stream: BinaryIO) -> None:
    """Reject extension payloads before :mod:`tarfile` can parse them."""

    try:
        stream.seek(0, os.SEEK_END)
        archive_size = stream.tell()
        stream.seek(0, os.SEEK_SET)
    except OSError as error:
        _fail("invalid-image-export-tar", f"cannot inspect runtime image export archive before parsing: {error}")
    if archive_size < MIN_TAR_END_BYTES or archive_size > MAX_IMAGE_EXPORT_ARCHIVE_BYTES:
        _fail("invalid-image-export-tar", "runtime image export archive has an invalid byte length")
    if archive_size % TAR_BLOCK_BYTES:
        _fail("invalid-image-export-tar", "runtime image export archive size must be tar-block aligned")
    offset = 0
    member_count = 0
    total_size = 0
    zero_block = b"\x00" * TAR_BLOCK_BYTES
    while offset < archive_size:
        try:
            header = stream.read(TAR_BLOCK_BYTES)
        except OSError as error:
            _fail("invalid-image-export-tar", f"cannot read runtime image export tar header: {error}")
        if len(header) != TAR_BLOCK_BYTES:
            _fail("truncated-image-export-tar", "runtime image export archive header is truncated")
        offset += TAR_BLOCK_BYTES
        if header == zero_block:
            try:
                second = stream.read(TAR_BLOCK_BYTES)
            except OSError as error:
                _fail("invalid-image-export-tar", f"cannot read runtime image export tar end marker: {error}")
            if second != zero_block:
                _fail("invalid-image-export-tar", "runtime image export archive has an invalid tar end marker")
            offset += TAR_BLOCK_BYTES
            remaining = archive_size - offset
            if remaining > MAX_CANONICAL_TAR_TRAILER_BYTES:
                _fail("tar-trailer-size", "runtime image export archive has an oversized zero trailer")
            while remaining:
                try:
                    trailing = stream.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
                except OSError as error:
                    _fail("invalid-image-export-tar", f"cannot read runtime image export tar trailer: {error}")
                if not trailing:
                    _fail("truncated-image-export-tar", "runtime image export archive trailer is truncated")
                if any(trailing):
                    _fail("invalid-image-export-tar", "runtime image export archive has nonzero trailing bytes")
                remaining -= len(trailing)
            return
        member_type = header[TAR_TYPE_OFFSET : TAR_TYPE_OFFSET + 1]
        if member_type not in {b"\x00", b"0", b"5"}:
            _fail("unsupported-image-export-tar-extension", "runtime image export archive has a non-regular tar member type")
        member_size = _parse_tar_size(header[TAR_SIZE_START:TAR_SIZE_END], "runtime image export tar member")
        if member_type == b"5" and member_size != 0:
            _fail("invalid-image-export-tar", "runtime image export archive directories must not carry payload bytes")
        if member_type != b"5":
            if member_size > MAX_ARCHIVE_MEMBER_BYTES:
                _fail("image-export-member-size", "runtime image export archive member exceeds its byte bound")
            total_size += member_size
            if total_size > MAX_IMAGE_EXPORT_ARCHIVE_BYTES:
                _fail("image-export-total-size", "runtime image export archive members exceed their total byte bound")
        member_count += 1
        if member_count > MAX_ARCHIVE_MEMBERS:
            _fail("invalid-image-export-tar", "runtime image export archive contains too many members")
        padded = _padded_size(member_size)
        if padded > archive_size - offset:
            _fail("truncated-image-export-tar", "runtime image export archive member payload is truncated")
        try:
            stream.seek(padded, os.SEEK_CUR)
        except OSError as error:
            _fail("invalid-image-export-tar", f"cannot skip runtime image export tar member payload: {error}")
        offset += padded
    _fail("truncated-image-export-tar", "runtime image export archive has no two-block tar end marker")


@dataclass(frozen=True)
class ArchiveMember:
    name: str
    byte_length: int
    offset_data: int


@dataclass(frozen=True)
class ArchiveInventory:
    files: Mapping[str, ArchiveMember]
    directories: frozenset[str]


@dataclass(frozen=True)
class SelectedLayer:
    member: ArchiveMember
    digest: str
    media_type: str


@dataclass(frozen=True)
class SelectedImage:
    source_layout: str
    config_raw: bytes
    config_digest: str
    layers: tuple[SelectedLayer, ...]


@dataclass(frozen=True)
class NormalizationFacts:
    source_layout: str
    oci_archive: common.CreatedEvidence


def _read_archive_inventory(stream: BinaryIO) -> ArchiveInventory:
    _preflight_image_export_tar(stream)
    try:
        stream.seek(0, os.SEEK_SET)
        archive = tarfile.open(fileobj=stream, mode="r:")
    except (OSError, tarfile.TarError) as error:
        _fail("invalid-image-export-tar", f"runtime image export archive must be an uncompressed tar: {error}")
    try:
        if archive.pax_headers:
            _fail("unsupported-image-export-tar-extension", "runtime image export archive must not use global PAX headers")
        files: dict[str, ArchiveMember] = {}
        directories: set[str] = set()
        total_size = 0
        while True:
            member = archive.next()
            if member is None:
                break
            if len(files) + len(directories) >= MAX_ARCHIVE_MEMBERS:
                _fail("invalid-image-export-tar", "runtime image export archive contains too many members")
            if member.pax_headers or member.sparse is not None:
                _fail("unsupported-image-export-tar-extension", "runtime image export archive must not use PAX or sparse members")
            if member.isdir():
                name = _safe_archive_member_name(member.name, "runtime image export directory", directory=True)
                if member.size != 0 or name in directories or name in files:
                    _fail("unsafe-image-export-member", "runtime image export archive has an unsafe duplicate directory")
                directories.add(name)
                continue
            if not member.isreg():
                _fail("unsafe-image-export-member", "runtime image export archive contains a link or special member")
            name = _safe_archive_member_name(member.name, "runtime image export file", directory=False)
            if name in files or name in directories:
                _fail("duplicate-image-export-member", f"runtime image export archive repeats {name!r}")
            if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES or member.offset_data < 0:
                _fail("image-export-member-size", f"runtime image export member {name!r} exceeds its byte bound")
            total_size += member.size
            if total_size > MAX_IMAGE_EXPORT_ARCHIVE_BYTES:
                _fail("image-export-total-size", "runtime image export archive members exceed their total byte bound")
            files[name] = ArchiveMember(name=name, byte_length=member.size, offset_data=member.offset_data)
        if not files:
            _fail("invalid-image-export-tar", "runtime image export archive has no regular members")
        return ArchiveInventory(files=files, directories=frozenset(directories))
    except RuntimeImageExportNormalizationError:
        raise
    except (OSError, tarfile.TarError) as error:
        _fail("invalid-image-export-tar", f"cannot parse runtime image export archive: {error}")
    finally:
        archive.close()


def _read_member(
    stream: BinaryIO,
    member: ArchiveMember,
    label: str,
    *,
    maximum_bytes: int,
    retain: bool,
    minimum_bytes: int = 1,
) -> tuple[bytes | None, str]:
    if member.byte_length < minimum_bytes or member.byte_length > maximum_bytes:
        _fail("image-export-member-size", f"{label} has an invalid byte length")
    try:
        stream.seek(member.offset_data, os.SEEK_SET)
    except OSError as error:
        _fail("invalid-image-export-tar", f"cannot seek {label}: {error}")
    digest = hashlib.sha256()
    remaining = member.byte_length
    chunks: list[bytes] | None = [] if retain else None
    while remaining:
        try:
            chunk = stream.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
        except OSError as error:
            _fail("invalid-image-export-tar", f"cannot read {label}: {error}")
        if not chunk:
            _fail("truncated-image-export-tar", f"{label} is truncated")
        digest.update(chunk)
        if chunks is not None:
            chunks.append(chunk)
        remaining -= len(chunk)
    return (b"".join(chunks) if chunks is not None else None), digest.hexdigest()


def _read_json_member(stream: BinaryIO, member: ArchiveMember, label: str) -> tuple[bytes, Any]:
    raw, _digest_value = _read_member(
        stream,
        member,
        label,
        maximum_bytes=MAX_RECEIPT_BYTES,
        retain=True,
    )
    assert raw is not None
    document = _common(
        lambda: common.parse_strict_json(raw, label, maximum_bytes=MAX_RECEIPT_BYTES, require_object=False)
    )
    return raw, document


def _parse_runtime_image_inspect(raw: bytes) -> str:
    document = _common(
        lambda: common.parse_strict_json(
            raw,
            "raw runtime image inspect",
            maximum_bytes=MAX_RECEIPT_BYTES,
            require_object=False,
        )
    )
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        _fail("invalid-runtime-image-inspect", "raw runtime image inspect must be a one-image array")
    row = document[0]
    image_id = _digest(row.get("Id"), "raw runtime image inspect[0].Id")
    if row.get("Os") != PLATFORM["os"] or row.get("Architecture") != PLATFORM["architecture"]:
        _fail("runtime-image-platform-mismatch", "raw runtime image inspect must describe linux/amd64")
    return image_id


def _parse_oci_descriptor(value: Any, label: str, *, media_types: set[str]) -> tuple[str, str, int]:
    if not isinstance(value, dict):
        _fail("invalid-oci-descriptor", f"{label} must be an object")
    if set(("mediaType", "digest", "size")) - set(value):
        _fail("invalid-oci-descriptor", f"{label} must contain mediaType, digest, and size")
    media_type = value["mediaType"]
    if type(media_type) is not str or media_type not in media_types:
        _fail("invalid-oci-media-type", f"{label}.mediaType is not allowed by this contract")
    return media_type, _digest(value["digest"], f"{label}.digest"), _descriptor_size(value["size"], f"{label}.size")


def _blob_path(digest: str) -> str:
    return "blobs/sha256/" + digest[7:]


def _require_platform(value: Any, label: str) -> None:
    if value != PLATFORM:
        _fail("oci-platform-mismatch", f"{label} must be exactly linux/amd64")


def _parse_config(config_raw: bytes, expected_image_id: str, label: str) -> dict[str, Any]:
    config = _common(lambda: common.parse_strict_json(config_raw, label, maximum_bytes=MAX_RECEIPT_BYTES))
    if config.get("os") != PLATFORM["os"] or config.get("architecture") != PLATFORM["architecture"]:
        _fail("oci-platform-mismatch", f"{label} must describe linux/amd64")
    if _digest_bytes(config_raw) != expected_image_id:
        _fail("runtime-image-id-mismatch", "runtime inspect Id must equal the selected config digest")
    return config


def _parse_legacy_docker_save(
    stream: BinaryIO,
    inventory: ArchiveInventory,
    expected_image_id: str,
) -> SelectedImage:
    files = inventory.files
    if "manifest.json" not in files:
        _fail("not-image-export-layout", "legacy runtime image export archive must contain manifest.json")
    manifest_raw, document = _read_json_member(stream, files["manifest.json"], "legacy Docker-save manifest")
    del manifest_raw
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        _fail("invalid-docker-save-manifest", "legacy Docker-save manifest must contain exactly one image")
    manifest = document[0]
    if set(manifest) != {"Config", "RepoTags", "Layers"}:
        _fail("invalid-docker-save-manifest", "legacy Docker-save manifest has unexpected fields")
    config_name = manifest["Config"]
    if type(config_name) is not str or LEGACY_CONFIG_NAME_RE.fullmatch(config_name) is None:
        _fail("invalid-docker-save-manifest", "legacy Docker-save manifest Config must be a lowercase digest JSON name")
    repo_tags = manifest["RepoTags"]
    if repo_tags is not None and (not isinstance(repo_tags, list) or any(type(tag) is not str for tag in repo_tags)):
        _fail("invalid-docker-save-manifest", "legacy Docker-save manifest RepoTags must be a string array")
    layers_value = manifest["Layers"]
    if not isinstance(layers_value, list) or not layers_value or any(type(path) is not str for path in layers_value):
        _fail("invalid-docker-save-manifest", "legacy Docker-save manifest Layers must be a nonempty string array")
    if len(layers_value) > MAX_ARCHIVE_MEMBERS:
        _fail("invalid-docker-save-manifest", "legacy Docker-save manifest has too many layers")
    layer_paths = tuple(layers_value)
    if len(set(layer_paths)) != len(layer_paths):
        _fail("duplicate-image-export-member", "legacy Docker-save manifest repeats a layer path")
    layer_ids: list[str] = []
    for path in layer_paths:
        match = LEGACY_LAYER_PATH_RE.fullmatch(path)
        if match is None:
            _fail("invalid-docker-save-manifest", "legacy Docker-save layer path is not normalized")
        layer_ids.append(match.group(1))
    if len(set(layer_ids)) != len(layer_ids):
        _fail("duplicate-image-export-member", "legacy Docker-save manifest repeats a layer identity")
    required_files = {"manifest.json", config_name}
    for layer_id, layer_path in zip(layer_ids, layer_paths):
        required_files.update({layer_path, f"{layer_id}/VERSION", f"{layer_id}/json"})
    optional_files = {"repositories"}
    if set(files) - required_files - optional_files:
        _fail("image-export-closure-mismatch", "legacy runtime image export archive has unexpected files")
    if required_files - set(files):
        _fail("image-export-closure-mismatch", "legacy runtime image export archive is missing required files")
    if inventory.directories - set(layer_ids):
        _fail("image-export-closure-mismatch", "legacy runtime image export archive has unexpected directories")
    if "repositories" in files:
        _repositories_raw, repositories = _read_json_member(stream, files["repositories"], "legacy Docker-save repositories")
        if not isinstance(repositories, dict):
            _fail("invalid-docker-save-repositories", "legacy Docker-save repositories must be an object")
    for layer_id in layer_ids:
        version_raw, _version_digest = _read_member(
            stream,
            files[f"{layer_id}/VERSION"],
            f"legacy Docker-save layer {layer_id} VERSION",
            maximum_bytes=64,
            retain=True,
        )
        if version_raw not in {b"1.0", b"1.0\n"}:
            _fail("invalid-docker-save-layer", "legacy Docker-save layer VERSION must be 1.0")
        _metadata_raw, metadata = _read_json_member(
            stream,
            files[f"{layer_id}/json"],
            f"legacy Docker-save layer {layer_id} metadata",
        )
        if not isinstance(metadata, dict):
            _fail("invalid-docker-save-layer", "legacy Docker-save layer metadata must be an object")
    config_member = files[config_name]
    config_raw, config_digest = _read_member(
        stream,
        config_member,
        "legacy Docker-save config",
        maximum_bytes=MAX_RECEIPT_BYTES,
        retain=True,
    )
    assert config_raw is not None
    config_digest = "sha256:" + config_digest
    if config_digest != expected_image_id or LEGACY_CONFIG_NAME_RE.fullmatch(config_name).group(1) != expected_image_id[7:]:
        _fail("runtime-image-id-mismatch", "legacy Docker-save config does not match raw image inspect Id")
    config = _parse_config(config_raw, expected_image_id, "legacy Docker-save config")
    rootfs = config.get("rootfs")
    if not isinstance(rootfs, dict) or rootfs.get("type") != "layers" or not isinstance(rootfs.get("diff_ids"), list):
        _fail("invalid-docker-save-config", "legacy Docker-save config must declare rootfs layers diff_ids")
    diff_ids = rootfs["diff_ids"]
    if len(diff_ids) != len(layer_paths) or any(type(digest) is not str for digest in diff_ids):
        _fail("legacy-layer-diff-id-mismatch", "legacy Docker-save rootfs diff_ids must match selected layers")
    selected_layers: list[SelectedLayer] = []
    observed_digests: set[str] = {config_digest}
    for index, (layer_path, expected_diff_id) in enumerate(zip(layer_paths, diff_ids)):
        expected_digest = _digest(expected_diff_id, f"legacy Docker-save rootfs.diff_ids[{index}]")
        member = files[layer_path]
        _raw, actual_digest = _read_member(
            stream,
            member,
            f"legacy Docker-save layer {index}",
            maximum_bytes=MAX_ARCHIVE_MEMBER_BYTES,
            retain=False,
        )
        actual = "sha256:" + actual_digest
        if actual != expected_digest:
            _fail("legacy-layer-diff-id-mismatch", "legacy Docker-save layer differs from config rootfs.diff_ids")
        if actual in observed_digests:
            _fail("oci-cross-binding-alias", "legacy Docker-save layers must have distinct config/blob digests")
        observed_digests.add(actual)
        selected_layers.append(
            SelectedLayer(
                member=member,
                digest=actual,
                media_type="application/vnd.oci.image.layer.v1.tar",
            )
        )
    return SelectedImage(
        source_layout=SOURCE_LAYOUT_LEGACY,
        config_raw=config_raw,
        config_digest=config_digest,
        layers=tuple(selected_layers),
    )


def _parse_oci_layout(
    stream: BinaryIO,
    inventory: ArchiveInventory,
    expected_image_id: str,
) -> SelectedImage:
    files = inventory.files
    required_headers = {OCI_LAYOUT_NAME, OCI_INDEX_NAME}
    if not required_headers <= set(files):
        _fail("not-oci-image-layout", "OCI runtime image export archive must contain oci-layout and index.json")
    allowed_sidecars = {"manifest.json", "repositories"}
    unknown_names = {
        name
        for name in files
        if name not in required_headers and name not in allowed_sidecars and OCI_BLOB_PATH_RE.fullmatch(name) is None
    }
    if unknown_names:
        _fail("image-export-closure-mismatch", "OCI runtime image export archive has unexpected files")
    if inventory.directories not in {frozenset(), frozenset({"blobs", "blobs/sha256"})}:
        _fail("image-export-closure-mismatch", "OCI runtime image export archive has unexpected directories")
    for sidecar in allowed_sidecars & set(files):
        if files[sidecar].byte_length > MAX_RECEIPT_BYTES:
            _fail("image-export-member-size", f"OCI runtime image export compatibility sidecar {sidecar!r} exceeds its byte bound")
        _read_member(
            stream,
            files[sidecar],
            f"OCI runtime image export compatibility sidecar {sidecar}",
            maximum_bytes=MAX_RECEIPT_BYTES,
            retain=False,
            minimum_bytes=0,
        )
    layout_raw, layout = _read_json_member(stream, files[OCI_LAYOUT_NAME], "OCI runtime image export layout header")
    del layout_raw
    if not isinstance(layout, dict) or layout.get("imageLayoutVersion") != "1.0.0":
        _fail("invalid-oci-layout", "OCI runtime image export layout header must declare imageLayoutVersion 1.0.0")
    index_raw, index = _read_json_member(stream, files[OCI_INDEX_NAME], "OCI runtime image export index")
    del index_raw
    if (
        not isinstance(index, dict)
        or index.get("schemaVersion") != 2
        or not isinstance(index.get("manifests"), list)
        or len(index["manifests"]) != 1
    ):
        _fail("invalid-oci-index", "OCI runtime image export index must contain exactly one manifest")
    if "mediaType" in index and index["mediaType"] != runtime_oci.OCI_INDEX_MEDIA_TYPE:
        _fail("invalid-oci-media-type", "OCI runtime image export index mediaType is not an OCI image index")
    manifest_media_type, manifest_digest, manifest_size = _parse_oci_descriptor(
        index["manifests"][0],
        "OCI runtime image export index manifest descriptor",
        media_types={runtime_oci.OCI_MANIFEST_MEDIA_TYPE},
    )
    del manifest_media_type
    if manifest_size > MAX_RECEIPT_BYTES:
        _fail("oci-json-size", "OCI runtime image export manifest exceeds the JSON byte bound")
    if "platform" in index["manifests"][0]:
        _require_platform(index["manifests"][0]["platform"], "OCI runtime image export index manifest platform")
    manifest_path = _blob_path(manifest_digest)
    if manifest_path not in files:
        _fail("missing-oci-blob", "OCI runtime image export manifest blob is absent")
    manifest_member = files[manifest_path]
    if manifest_member.byte_length != manifest_size:
        _fail("oci-descriptor-size-mismatch", "OCI runtime image export manifest size differs from its descriptor")
    manifest_raw, manifest_actual = _read_member(
        stream,
        manifest_member,
        "OCI runtime image export manifest blob",
        maximum_bytes=MAX_RECEIPT_BYTES,
        retain=True,
    )
    assert manifest_raw is not None
    if "sha256:" + manifest_actual != manifest_digest:
        _fail("oci-descriptor-digest-mismatch", "OCI runtime image export manifest digest differs from its descriptor")
    manifest = _common(
        lambda: common.parse_strict_json(manifest_raw, "OCI runtime image export manifest blob", maximum_bytes=MAX_RECEIPT_BYTES)
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 2
        or "config" not in manifest
        or not isinstance(manifest.get("layers"), list)
        or not manifest["layers"]
    ):
        _fail("invalid-oci-manifest", "OCI runtime image export manifest must contain a config and layers")
    if "mediaType" in manifest and manifest["mediaType"] != runtime_oci.OCI_MANIFEST_MEDIA_TYPE:
        _fail("invalid-oci-media-type", "OCI runtime image export manifest mediaType is not an OCI image manifest")
    config_media_type, config_digest, config_size = _parse_oci_descriptor(
        manifest["config"],
        "OCI runtime image export config descriptor",
        media_types={runtime_oci.OCI_CONFIG_MEDIA_TYPE},
    )
    del config_media_type
    if config_size > MAX_RECEIPT_BYTES:
        _fail("oci-json-size", "OCI runtime image export config exceeds the JSON byte bound")
    selected_layers: list[SelectedLayer] = []
    expected_blob_paths = {manifest_path, _blob_path(config_digest)}
    if _blob_path(config_digest) == manifest_path:
        _fail("oci-cross-binding-alias", "OCI runtime image export config must not alias the manifest")
    for index, layer_value in enumerate(manifest["layers"]):
        media_type, digest, byte_length = _parse_oci_descriptor(
            layer_value,
            f"OCI runtime image export layer descriptor[{index}]",
            media_types=runtime_oci.OCI_LAYER_MEDIA_TYPES,
        )
        path = _blob_path(digest)
        if path in expected_blob_paths:
            _fail("oci-cross-binding-alias", "OCI runtime image export layers must use distinct config and manifest blobs")
        expected_blob_paths.add(path)
        if path not in files:
            _fail("missing-oci-blob", "OCI runtime image export layer blob is absent")
        selected_layers.append(SelectedLayer(member=files[path], digest=digest, media_type=media_type))
        if files[path].byte_length != byte_length:
            _fail("oci-descriptor-size-mismatch", "OCI runtime image export layer size differs from its descriptor")
    actual_blob_paths = {name for name in files if OCI_BLOB_PATH_RE.fullmatch(name) is not None}
    if actual_blob_paths != expected_blob_paths:
        _fail("oci-blob-closure-mismatch", "OCI runtime image export blobs must exactly close the selected image")
    config_path = _blob_path(config_digest)
    if config_path not in files or files[config_path].byte_length != config_size:
        _fail("oci-descriptor-size-mismatch", "OCI runtime image export config size differs from its descriptor")
    config_raw, config_actual = _read_member(
        stream,
        files[config_path],
        "OCI runtime image export config blob",
        maximum_bytes=MAX_RECEIPT_BYTES,
        retain=True,
    )
    assert config_raw is not None
    if "sha256:" + config_actual != config_digest:
        _fail("oci-descriptor-digest-mismatch", "OCI runtime image export config digest differs from its descriptor")
    _parse_config(config_raw, expected_image_id, "OCI runtime image export config blob")
    if config_digest != expected_image_id:
        _fail("runtime-image-id-mismatch", "OCI runtime image export config descriptor differs from raw image inspect Id")
    for index, layer in enumerate(selected_layers):
        _raw, actual = _read_member(
            stream,
            layer.member,
            f"OCI runtime image export layer blob {index}",
            maximum_bytes=MAX_ARCHIVE_MEMBER_BYTES,
            retain=False,
        )
        if "sha256:" + actual != layer.digest:
            _fail("oci-descriptor-digest-mismatch", "OCI runtime image export layer digest differs from its descriptor")
    source_layout = SOURCE_LAYOUT_OCI_SIDECARS if allowed_sidecars & set(files) else SOURCE_LAYOUT_OCI
    return SelectedImage(
        source_layout=source_layout,
        config_raw=config_raw,
        config_digest=config_digest,
        layers=tuple(selected_layers),
    )


def _select_image(stream: BinaryIO, expected_image_id: str) -> SelectedImage:
    inventory = _read_archive_inventory(stream)
    has_layout = OCI_LAYOUT_NAME in inventory.files
    has_index = OCI_INDEX_NAME in inventory.files
    if has_layout != has_index:
        _fail("not-image-export-layout", "runtime image export archive must contain both OCI layout headers or neither")
    if has_layout:
        return _parse_oci_layout(stream, inventory, expected_image_id)
    return _parse_legacy_docker_save(stream, inventory, expected_image_id)


def _canonical_oci_json(selected: SelectedImage) -> tuple[bytes, bytes, bytes]:
    manifest_document = {
        "schemaVersion": 2,
        "mediaType": runtime_oci.OCI_MANIFEST_MEDIA_TYPE,
        "config": {
            "mediaType": runtime_oci.OCI_CONFIG_MEDIA_TYPE,
            "digest": selected.config_digest,
            "size": len(selected.config_raw),
        },
        "layers": [
            {
                "mediaType": layer.media_type,
                "digest": layer.digest,
                "size": layer.member.byte_length,
            }
            for layer in selected.layers
        ],
    }
    manifest_raw = common.canonical_json_bytes(manifest_document)
    manifest_digest = _digest_bytes(manifest_raw)
    index_raw = common.canonical_json_bytes(
        {
            "schemaVersion": 2,
            "mediaType": runtime_oci.OCI_INDEX_MEDIA_TYPE,
            "manifests": [
                {
                    "mediaType": runtime_oci.OCI_MANIFEST_MEDIA_TYPE,
                    "digest": manifest_digest,
                    "size": len(manifest_raw),
                    "platform": dict(PLATFORM),
                }
            ],
        }
    )
    layout_raw = common.canonical_json_bytes({"imageLayoutVersion": "1.0.0"})
    return layout_raw, index_raw, manifest_raw


def _estimated_canonical_oci_size(selected: SelectedImage) -> int:
    members = _canonical_oci_members(selected)
    file_sizes = [byte_length for _name, byte_length, directory in members if not directory]
    if any(size < 1 or size > MAX_ARCHIVE_MEMBER_BYTES for size in file_sizes):
        _fail("oci-member-size", "canonical OCI output has a member outside the USTAR byte bound")
    member_bytes = sum(TAR_BLOCK_BYTES + _padded_size(byte_length) for _name, byte_length, _directory in members)
    # ``tarfile`` writes two mandatory zero blocks then fills out its 20-block
    # record.  The selected members fully determine the padding length, so the
    # resulting USTAR transport bytes—not merely its OCI content—are replayed.
    trailer_bytes = MIN_TAR_END_BYTES + (-(member_bytes + MIN_TAR_END_BYTES) % TAR_RECORD_BYTES)
    total = member_bytes + trailer_bytes
    if total > MAX_NORMALIZED_OCI_ARCHIVE_BYTES:
        _fail("normalized-oci-too-large", "canonical OCI output exceeds the runtime assembly v1 USTAR bound")
    return total


class _MemberReader:
    """A bounded, non-extracting view of one held source tar member."""

    def __init__(self, stream: BinaryIO, member: ArchiveMember, label: str) -> None:
        self._stream = stream
        self._remaining = member.byte_length
        self._label = label
        try:
            stream.seek(member.offset_data, os.SEEK_SET)
        except OSError as error:
            _fail("invalid-image-export-tar", f"cannot seek {label}: {error}")

    def read(self, size: int = -1) -> bytes:
        if self._remaining == 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        try:
            raw = self._stream.read(size)
        except OSError as error:
            _fail("invalid-image-export-tar", f"cannot read {self._label}: {error}")
        if not raw:
            _fail("truncated-image-export-tar", f"{self._label} is truncated")
        self._remaining -= len(raw)
        return raw

    def require_complete(self) -> None:
        if self._remaining != 0:
            _fail("truncated-image-export-tar", f"{self._label} was not copied completely")


def _canonical_tar_info(name: str, byte_length: int, *, directory: bool) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.mode = 0o755 if directory else 0o644
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    if directory:
        member.type = tarfile.DIRTYPE
        member.size = 0
    else:
        member.size = byte_length
    return member


def _canonical_oci_members(selected: SelectedImage) -> tuple[tuple[str, int, bool], ...]:
    layout_raw, index_raw, manifest_raw = _canonical_oci_json(selected)
    rows: list[tuple[str, int, bool]] = [
        (OCI_LAYOUT_NAME, len(layout_raw), False),
        (OCI_INDEX_NAME, len(index_raw), False),
        ("blobs", 0, True),
        ("blobs/sha256", 0, True),
        (_blob_path(_digest_bytes(manifest_raw)), len(manifest_raw), False),
        (_blob_path(selected.config_digest), len(selected.config_raw), False),
    ]
    rows.extend((_blob_path(layer.digest), layer.member.byte_length, False) for layer in selected.layers)
    return tuple(rows)


def _validate_canonical_oci_ustar(stream: BinaryIO, selected: SelectedImage) -> None:
    """Require the exact output USTAR grammar before semantic OCI parsing.

    Descriptor hashes alone do not constrain tar headers, member order, or
    trailers: a writer could preserve OCI bytes while changing those raw
    transport facts.  Derive each expected USTAR header from the same closed
    metadata used by the producer, then compare it byte-for-byte before
    ``tarfile`` sees the output.
    """

    try:
        stream.seek(0, os.SEEK_END)
        archive_size = stream.tell()
        stream.seek(0, os.SEEK_SET)
    except OSError as error:
        _fail("invalid-oci-tar", f"cannot inspect canonical OCI archive: {error}")
    expected_archive_size = _estimated_canonical_oci_size(selected)
    if archive_size != expected_archive_size:
        _fail("noncanonical-oci-tar", "canonical OCI archive byte length differs from the exact USTAR form")
    offset = 0
    for name, byte_length, directory in _canonical_oci_members(selected):
        try:
            header = stream.read(TAR_BLOCK_BYTES)
        except OSError as error:
            _fail("invalid-oci-tar", f"cannot read canonical OCI header for {name!r}: {error}")
        if len(header) != TAR_BLOCK_BYTES:
            _fail("truncated-oci-tar", "canonical OCI archive header is truncated")
        expected_header = _canonical_tar_info(name, byte_length, directory=directory).tobuf(
            format=tarfile.USTAR_FORMAT,
            encoding="utf-8",
            errors="surrogateescape",
        )
        if header != expected_header:
            _fail("noncanonical-oci-tar", f"canonical OCI archive header/order differs at {name!r}")
        offset += TAR_BLOCK_BYTES
        padded = _padded_size(byte_length)
        if padded > archive_size - offset:
            _fail("truncated-oci-tar", f"canonical OCI archive member {name!r} is truncated")
        try:
            stream.seek(byte_length, os.SEEK_CUR)
        except OSError as error:
            _fail("invalid-oci-tar", f"cannot skip canonical OCI member {name!r}: {error}")
        offset += byte_length
        padding_bytes = padded - byte_length
        if padding_bytes:
            try:
                padding = stream.read(padding_bytes)
            except OSError as error:
                _fail("invalid-oci-tar", f"cannot read canonical OCI padding for {name!r}: {error}")
            if len(padding) != padding_bytes:
                _fail("truncated-oci-tar", f"canonical OCI archive padding is truncated at {name!r}")
            if any(padding):
                _fail("noncanonical-oci-tar", f"canonical OCI archive padding is nonzero at {name!r}")
        offset += padding_bytes
    remaining = archive_size - offset
    expected_trailer_bytes = archive_size - sum(
        TAR_BLOCK_BYTES + _padded_size(byte_length)
        for _name, byte_length, _directory in _canonical_oci_members(selected)
    )
    if remaining != expected_trailer_bytes:
        _fail("noncanonical-oci-tar", "canonical OCI archive has an invalid exact zero trailer length")
    while remaining:
        try:
            trailing = stream.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
        except OSError as error:
            _fail("invalid-oci-tar", f"cannot read canonical OCI trailer: {error}")
        if not trailing:
            _fail("truncated-oci-tar", "canonical OCI archive trailer is truncated")
        if any(trailing):
            _fail("noncanonical-oci-tar", "canonical OCI archive has nonzero trailing bytes")
        remaining -= len(trailing)


def _write_stream_create_only(
    directory_fd: int,
    name: str,
    label: str,
    writer: Callable[[BinaryIO], None],
    *,
    maximum_bytes: int,
) -> common.CreatedEvidence:
    """Create one private, bounded output leaf without a replace/rename step."""

    _common(lambda: common.require_private_evidence_directory_fd(directory_fd, f"{label} parent"))
    _common(lambda: common.validate_relative_path(name, f"{label} name"))
    if PurePosixPath(name).parts != (name,):
        _fail("invalid-evidence-name", f"{label} name must be a direct leaf")
    if not callable(writer):
        _fail("invalid-evidence-writer", f"{label} writer must be callable")
    try:
        nofollow, _directory, cloexec, _nonblock = common.require_safe_open_flags()
        descriptor = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow | cloexec, 0o600, dir_fd=directory_fd)
    except FileExistsError as error:
        _fail("create-only-collision", f"cannot create new {label}: {error}")
    except OSError as error:
        _fail("unwritable-output", f"cannot create new {label}: {error}")
    stable: os.stat_result | None = None
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except OSError as error:
            _fail("unsafe-output-mode", f"cannot make {label} private: {error}")
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600:
            _fail("unsafe-output-mode", f"{label} was not created as a private regular leaf")
        try:
            with os.fdopen(os.dup(descriptor), "w+b", buffering=0) as output:
                writer(output)
                output.flush()
        except RuntimeImageExportNormalizationError:
            raise
        except (OSError, tarfile.TarError) as error:
            _fail("unwritable-output", f"cannot write {label}: {error}")
        try:
            os.fsync(descriptor)
        except OSError as error:
            _fail("durability-failure", f"cannot durably synchronize {label}: {error}")
        stable = os.fstat(descriptor)
        if (
            not stat.S_ISREG(stable.st_mode)
            or stable.st_nlink != 1
            or stat.S_IMODE(stable.st_mode) != 0o600
            or stable.st_size < 1
            or stable.st_size > maximum_bytes
            or (stable.st_dev, stable.st_ino) != (before.st_dev, before.st_ino)
        ):
            _fail("unsafe-output-mode", f"{label} changed or exceeded its byte bound while it was written")
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
        except OSError as error:
            _fail("unreadable-output", f"cannot rewind {label}: {error}")
        digest = hashlib.sha256()
        remaining = stable.st_size
        while remaining:
            try:
                chunk = os.read(descriptor, min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
            except OSError as error:
                _fail("unreadable-output", f"cannot read {label}: {error}")
            if not chunk:
                _fail("unreadable-output", f"cannot read complete {label}")
            digest.update(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    assert stable is not None
    try:
        visible = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        _fail("raced-output", f"cannot re-inspect {label}: {error}")
    if (
        not stat.S_ISREG(visible.st_mode)
        or visible.st_nlink != 1
        or stat.S_IMODE(visible.st_mode) != 0o600
        or (visible.st_dev, visible.st_ino) != (stable.st_dev, stable.st_ino)
        or visible.st_size != stable.st_size
    ):
        _fail("raced-output", f"{label} changed before it could be published")
    try:
        os.fsync(directory_fd)
    except OSError as error:
        _fail("durability-failure", f"cannot durably synchronize {label} parent directory: {error}")
    return common.CreatedEvidence(
        name=name,
        sha256=digest.hexdigest(),
        byte_length=stable.st_size,
        device=stable.st_dev,
        inode=stable.st_ino,
    )


def _write_canonical_oci(stream: BinaryIO, selected: SelectedImage, normalized_fd: int) -> common.CreatedEvidence:
    _estimated_canonical_oci_size(selected)
    layout_raw, index_raw, manifest_raw = _canonical_oci_json(selected)

    def writer(destination: BinaryIO) -> None:
        with tarfile.open(fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            archive.addfile(_canonical_tar_info(OCI_LAYOUT_NAME, len(layout_raw), directory=False), io.BytesIO(layout_raw))
            archive.addfile(_canonical_tar_info(OCI_INDEX_NAME, len(index_raw), directory=False), io.BytesIO(index_raw))
            archive.addfile(_canonical_tar_info("blobs", 0, directory=True))
            archive.addfile(_canonical_tar_info("blobs/sha256", 0, directory=True))
            archive.addfile(
                _canonical_tar_info(_blob_path(_digest_bytes(manifest_raw)), len(manifest_raw), directory=False),
                io.BytesIO(manifest_raw),
            )
            archive.addfile(
                _canonical_tar_info(_blob_path(selected.config_digest), len(selected.config_raw), directory=False),
                io.BytesIO(selected.config_raw),
            )
            for index, layer in enumerate(selected.layers):
                reader = _MemberReader(stream, layer.member, f"selected image export layer {index}")
                archive.addfile(
                    _canonical_tar_info(_blob_path(layer.digest), layer.member.byte_length, directory=False),
                    reader,
                )
                reader.require_complete()

    return _write_stream_create_only(
        normalized_fd,
        OCI_ARCHIVE_NAME,
        "canonical normalized OCI archive",
        writer,
        maximum_bytes=MAX_NORMALIZED_OCI_ARCHIVE_BYTES,
    )


def _normalize_archive(stream: BinaryIO, expected_image_id: str, normalized_fd: int) -> NormalizationFacts:
    selected = _select_image(stream, expected_image_id)
    created = _write_canonical_oci(stream, selected, normalized_fd)
    return NormalizationFacts(source_layout=selected.source_layout, oci_archive=created)


RAW_LEAVES = (
    ("image_inspect", IMAGE_INSPECT_NAME, "raw runtime image inspect", MAX_RECEIPT_BYTES),
    ("image_export_archive", IMAGE_EXPORT_ARCHIVE_NAME, "raw runtime image export archive", MAX_IMAGE_EXPORT_ARCHIVE_BYTES),
)
NORMALIZED_LEAVES = (
    ("oci_archive", OCI_ARCHIVE_NAME, "canonical normalized OCI archive", MAX_NORMALIZED_OCI_ARCHIVE_BYTES),
    ("layout", OCI_LAYOUT_NAME, "canonical OCI layout header", MAX_RECEIPT_BYTES),
    ("index", OCI_INDEX_NAME, "canonical OCI index", MAX_RECEIPT_BYTES),
    ("manifest", OCI_MANIFEST_NAME, "canonical OCI manifest", MAX_RECEIPT_BYTES),
    ("config", OCI_CONFIG_NAME, "selected OCI config", MAX_RECEIPT_BYTES),
)


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


def _rebase_descriptor(
    descriptor: common.EvidenceDescriptor,
    *,
    directory_name: str,
    leaf_name: str,
    label: str,
) -> common.EvidenceDescriptor:
    return _common(
        lambda: common.rebase_descriptor_to_held_leaf(
            descriptor,
            expected_root_relative_path=f"{directory_name}/{leaf_name}",
            leaf_name=leaf_name,
            label=label,
        )
    )


def _consume_bytes(
    directory_fd: int,
    descriptor: common.EvidenceDescriptor,
    *,
    directory_name: str,
    leaf_name: str,
    label: str,
    maximum_bytes: int,
) -> bytes:
    held = _rebase_descriptor(
        descriptor,
        directory_name=directory_name,
        leaf_name=leaf_name,
        label=label,
    )

    def read_all(source: BinaryIO) -> bytes:
        chunks: list[bytes] = []
        remaining = maximum_bytes
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
            directory_fd,
            held,
            label,
            read_all,
            maximum_bytes=maximum_bytes,
        )
    )


def _consume_archive(
    directory_fd: int,
    descriptor: common.EvidenceDescriptor,
    *,
    directory_name: str,
    leaf_name: str,
    label: str,
    maximum_bytes: int,
    consumer: Callable[[BinaryIO], Any],
) -> Any:
    held = _rebase_descriptor(
        descriptor,
        directory_name=directory_name,
        leaf_name=leaf_name,
        label=label,
    )
    return _common(
        lambda: common.consume_private_snapshot_descriptor_file(
            directory_fd,
            held,
            label,
            consumer,
            maximum_bytes=maximum_bytes,
        )
    )


def _parse_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(
        dict(value),
        {
            "schema_version",
            "status",
            "qualification_status",
            "authority",
            "capture_scope",
            "reconstruction_id",
            "raw_archive_format",
            "normalized_archive_format",
            "source_layout",
            "platform",
            "image_id",
            "content_binding",
            "docker_invocation_binding",
            "source_binding",
            "bundle_binding",
            "build_invocation_binding",
            "runtime_capture_binding",
            "independence_binding",
            "image_inspect",
            "image_export_archive",
            "oci_archive",
            "layout",
            "index",
            "manifest",
            "config",
        },
        "runtime image-export OCI normalization receipt",
    )
    if (
        row["schema_version"] != NORMALIZATION_VERSION
        or row["status"] != STATUS
        or row["qualification_status"] != QUALIFICATION_STATUS
        or row["authority"] != AUTHORITY
        or row["capture_scope"] != CAPTURE_SCOPE
        or row["reconstruction_id"] not in {"a", "b"}
        or row["raw_archive_format"] != RAW_ARCHIVE_FORMAT
        or row["normalized_archive_format"] != OCI_ARCHIVE_FORMAT
        or row["source_layout"] not in SOURCE_LAYOUTS
        or row["platform"] != PLATFORM
        or row["content_binding"] != CONTENT_BINDING
        or any(
            row[field] != NOT_ESTABLISHED
            for field in (
                "docker_invocation_binding",
                "source_binding",
                "bundle_binding",
                "build_invocation_binding",
                "runtime_capture_binding",
                "independence_binding",
            )
        )
    ):
        _fail("invalid-normalization-receipt", "runtime image-export normalization receipt is not the exact v1 contract")
    image_id = _digest(row["image_id"], "runtime image-export normalization receipt.image_id")
    descriptors: dict[str, common.EvidenceDescriptor] = {}
    for field, leaf, leaf_label, _maximum in RAW_LEAVES:
        descriptor = _common(lambda field=field, leaf_label=leaf_label: common.parse_descriptor(row[field], leaf_label))
        expected = f"{RAW_DIRECTORY_NAME}/{leaf}"
        if descriptor.path != expected:
            _fail("normalization-leaf-path-mismatch", f"{leaf_label} must use the fixed path {expected!r}")
        descriptors[field] = descriptor
    for field, leaf, leaf_label, _maximum in NORMALIZED_LEAVES:
        descriptor = _common(lambda field=field, leaf_label=leaf_label: common.parse_descriptor(row[field], leaf_label))
        expected = f"{NORMALIZED_DIRECTORY_NAME}/{leaf}"
        if descriptor.path != expected:
            _fail("normalization-leaf-path-mismatch", f"{leaf_label} must use the fixed path {expected!r}")
        descriptors[field] = descriptor
    _common(lambda: common.require_unique_descriptors(tuple(descriptors.values()), "runtime image-export normalization descriptors"))
    return {**row, "image_id": image_id, "_descriptors": descriptors}


def _verify_normalized_archive(
    stream: BinaryIO,
    expected_image_id: str,
    source_selected: SelectedImage,
) -> tuple[SelectedImage, runtime_oci.OciArchiveFacts]:
    # Keep the existing OCI v1 consumer as an independently implemented
    # compatibility oracle. Its clean-layout parser deliberately rejects the
    # opaque root sidecars that this normalizer retained only in the raw root.
    _validate_canonical_oci_ustar(stream, source_selected)
    facts = _oci(lambda: runtime_oci._parse_oci_archive(stream, expected_image_id))
    selected = _select_image(stream, expected_image_id)
    if selected.source_layout != SOURCE_LAYOUT_OCI:
        _fail("normalized-oci-layout", "normalized OCI archive must be a clean OCI layout without sidecars")
    if selected.config_raw != source_selected.config_raw or selected.config_digest != source_selected.config_digest:
        _fail("normalized-oci-content-mismatch", "normalized OCI config differs from the selected image export config")
    source_layers = tuple((layer.digest, layer.member.byte_length, layer.media_type) for layer in source_selected.layers)
    normalized_layers = tuple((layer.digest, layer.member.byte_length, layer.media_type) for layer in selected.layers)
    if normalized_layers != source_layers:
        _fail("normalized-oci-content-mismatch", "normalized OCI layers differ from selected image export layers")
    expected_layout, expected_index, expected_manifest = _canonical_oci_json(source_selected)
    if (
        facts.layout_raw != expected_layout
        or facts.index_raw != expected_index
        or facts.manifest_raw != expected_manifest
        or facts.config_raw != source_selected.config_raw
    ):
        _fail("normalized-oci-nondeterministic", "normalized OCI JSON members differ from the canonical selected-content form")
    return selected, facts


def verify_reconstructed_runtime_image_export_oci_normalization_fd(root_fd: int) -> dict[str, Any]:
    """Replay one already-held normalization root without creating output."""

    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "runtime image export normalization root"))
    expected_root_entries = {RAW_DIRECTORY_NAME, NORMALIZED_DIRECTORY_NAME, NORMALIZATION_NAME}
    _assert_entries(root_fd, expected_root_entries, "runtime image export normalization root")
    receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            NORMALIZATION_NAME,
            "runtime image export normalization receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    parsed = _parse_receipt(receipt)
    descriptors = parsed["_descriptors"]
    raw_fd = _common(
        lambda: common.open_private_child_directory(root_fd, RAW_DIRECTORY_NAME, "raw runtime image export evidence directory")
    )
    normalized_fd = _common(
        lambda: common.open_private_child_directory(root_fd, NORMALIZED_DIRECTORY_NAME, "normalized OCI evidence directory")
    )
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd, raw_fd, RAW_DIRECTORY_NAME, "held raw runtime image export evidence directory"
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd, normalized_fd, NORMALIZED_DIRECTORY_NAME, "held normalized OCI evidence directory"
            )
        )
        _assert_entries(raw_fd, {leaf for _field, leaf, _label, _maximum in RAW_LEAVES}, "raw runtime image export evidence directory")
        _assert_entries(
            normalized_fd,
            {leaf for _field, leaf, _label, _maximum in NORMALIZED_LEAVES},
            "normalized OCI evidence directory",
        )
        image_id = _parse_runtime_image_inspect(
            _consume_bytes(
                raw_fd,
                descriptors["image_inspect"],
                directory_name=RAW_DIRECTORY_NAME,
                leaf_name=IMAGE_INSPECT_NAME,
                label="raw runtime image inspect",
                maximum_bytes=MAX_RECEIPT_BYTES,
            )
        )
        if image_id != parsed["image_id"]:
            _fail("runtime-image-id-mismatch", "normalization receipt image_id differs from raw image inspect")
        source_selected = _consume_archive(
            raw_fd,
            descriptors["image_export_archive"],
            directory_name=RAW_DIRECTORY_NAME,
            leaf_name=IMAGE_EXPORT_ARCHIVE_NAME,
            label="raw runtime image export archive",
            maximum_bytes=MAX_IMAGE_EXPORT_ARCHIVE_BYTES,
            consumer=lambda source: _select_image(source, image_id),
        )
        if source_selected.source_layout != parsed["source_layout"]:
            _fail("source-layout-mismatch", "normalization receipt source_layout differs from raw runtime image export archive")
        _normalized_selected, facts = _consume_archive(
            normalized_fd,
            descriptors["oci_archive"],
            directory_name=NORMALIZED_DIRECTORY_NAME,
            leaf_name=OCI_ARCHIVE_NAME,
            label="canonical normalized OCI archive",
            maximum_bytes=MAX_NORMALIZED_OCI_ARCHIVE_BYTES,
            consumer=lambda source: _verify_normalized_archive(source, image_id, source_selected),
        )
        expected_raw = {
            "layout": facts.layout_raw,
            "index": facts.index_raw,
            "manifest": facts.manifest_raw,
            "config": facts.config_raw,
        }
        for field, raw in expected_raw.items():
            leaf = next(name for candidate, name, _label, _maximum in NORMALIZED_LEAVES if candidate == field)
            replayed = _consume_bytes(
                normalized_fd,
                descriptors[field],
                directory_name=NORMALIZED_DIRECTORY_NAME,
                leaf_name=leaf,
                label=f"normalized OCI {field} snapshot",
                maximum_bytes=MAX_RECEIPT_BYTES,
            )
            if replayed != raw:
                _fail("normalized-oci-derived-snapshot-mismatch", f"normalized OCI {field} snapshot differs from held archive")
        _assert_entries(raw_fd, {leaf for _field, leaf, _label, _maximum in RAW_LEAVES}, "raw runtime image export evidence directory")
        _assert_entries(
            normalized_fd,
            {leaf for _field, leaf, _label, _maximum in NORMALIZED_LEAVES},
            "normalized OCI evidence directory",
        )
    finally:
        os.close(normalized_fd)
        os.close(raw_fd)
    _assert_entries(root_fd, expected_root_entries, "runtime image export normalization root")
    terminal_receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            NORMALIZATION_NAME,
            "runtime image export normalization receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    if terminal_receipt != receipt:
        _fail("raced-input", "runtime image export normalization receipt changed during replay")
    return {key: value for key, value in parsed.items() if key != "_descriptors"}


def verify_reconstructed_runtime_image_export_oci_normalization(evidence_root: Path) -> dict[str, Any]:
    evidence_root = _normalized_absolute_path(evidence_root, "runtime image export normalization root")
    root_fd = _common(
        lambda: common.open_private_evidence_directory(evidence_root, "runtime image export normalization root")
    )
    try:
        return verify_reconstructed_runtime_image_export_oci_normalization_fd(root_fd)
    finally:
        os.close(root_fd)


def prepare_reconstructed_runtime_image_export_oci_normalization(
    evidence_root: Path,
    *,
    image_inspect: Path,
    image_export_archive: Path,
    reconstruction_id: str,
) -> dict[str, Any]:
    """Create one fresh runtime image export normalization root and self-replay it."""

    evidence_root = _normalized_absolute_path(evidence_root, "--evidence-root")
    image_inspect = _normalized_absolute_path(image_inspect, "--image-inspect")
    image_export_archive = _normalized_absolute_path(image_export_archive, "--image-export-archive")
    if reconstruction_id not in {"a", "b"}:
        _fail("invalid-reconstruction-id", "--reconstruction-id must be exactly a or b")
    _require_external_output(evidence_root, image_inspect, image_export_archive)
    root_fd = _common(
        lambda: common.create_private_evidence_directory(evidence_root, "runtime image export normalization root")
    )
    raw_fd: int | None = None
    normalized_fd: int | None = None
    try:
        raw_fd = _common(
            lambda: common.create_private_child_directory(root_fd, RAW_DIRECTORY_NAME, "raw runtime image export evidence directory")
        )
        normalized_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd, NORMALIZED_DIRECTORY_NAME, "normalized OCI evidence directory"
            )
        )
        inspect_snapshot = _common(
            lambda: common.snapshot_absolute_regular_create_only(
                image_inspect,
                raw_fd,
                IMAGE_INSPECT_NAME,
                "raw runtime image inspect",
                maximum_bytes=MAX_RECEIPT_BYTES,
                minimum_bytes=1,
            )
        )
        image_export_snapshot = _common(
            lambda: common.snapshot_absolute_regular_create_only(
                image_export_archive,
                raw_fd,
                IMAGE_EXPORT_ARCHIVE_NAME,
                "raw runtime image export archive",
                maximum_bytes=MAX_IMAGE_EXPORT_ARCHIVE_BYTES,
                minimum_bytes=MIN_TAR_END_BYTES,
            )
        )
        descriptors: dict[str, common.EvidenceDescriptor] = {
            "image_inspect": inspect_snapshot.descriptor(
                f"{RAW_DIRECTORY_NAME}/{IMAGE_INSPECT_NAME}", "raw runtime image inspect"
            ),
            "image_export_archive": image_export_snapshot.descriptor(
                f"{RAW_DIRECTORY_NAME}/{IMAGE_EXPORT_ARCHIVE_NAME}", "raw runtime image export archive"
            ),
        }
        image_id = _parse_runtime_image_inspect(
            _consume_bytes(
                raw_fd,
                descriptors["image_inspect"],
                directory_name=RAW_DIRECTORY_NAME,
                leaf_name=IMAGE_INSPECT_NAME,
                label="raw runtime image inspect",
                maximum_bytes=MAX_RECEIPT_BYTES,
            )
        )
        normalization = _consume_archive(
            raw_fd,
            descriptors["image_export_archive"],
            directory_name=RAW_DIRECTORY_NAME,
            leaf_name=IMAGE_EXPORT_ARCHIVE_NAME,
            label="raw runtime image export archive",
            maximum_bytes=MAX_IMAGE_EXPORT_ARCHIVE_BYTES,
            consumer=lambda source: _normalize_archive(source, image_id, normalized_fd),
        )
        descriptors["oci_archive"] = normalization.oci_archive.descriptor(
            f"{NORMALIZED_DIRECTORY_NAME}/{OCI_ARCHIVE_NAME}", "canonical normalized OCI archive"
        )
        source_selected = _consume_archive(
            raw_fd,
            descriptors["image_export_archive"],
            directory_name=RAW_DIRECTORY_NAME,
            leaf_name=IMAGE_EXPORT_ARCHIVE_NAME,
            label="raw runtime image export archive for normalization binding",
            maximum_bytes=MAX_IMAGE_EXPORT_ARCHIVE_BYTES,
            consumer=lambda source: _select_image(source, image_id),
        )
        if source_selected.source_layout != normalization.source_layout:
            _fail("source-layout-mismatch", "raw runtime image export archive changed its selected layout during normalization")
        _normalized_selected, facts = _consume_archive(
            normalized_fd,
            descriptors["oci_archive"],
            directory_name=NORMALIZED_DIRECTORY_NAME,
            leaf_name=OCI_ARCHIVE_NAME,
            label="canonical normalized OCI archive",
            maximum_bytes=MAX_NORMALIZED_OCI_ARCHIVE_BYTES,
            consumer=lambda source: _verify_normalized_archive(source, image_id, source_selected),
        )
        for field, leaf, raw, label in (
            ("layout", OCI_LAYOUT_NAME, facts.layout_raw, "canonical OCI layout header"),
            ("index", OCI_INDEX_NAME, facts.index_raw, "canonical OCI index"),
            ("manifest", OCI_MANIFEST_NAME, facts.manifest_raw, "canonical OCI manifest"),
            ("config", OCI_CONFIG_NAME, facts.config_raw, "selected OCI config"),
        ):
            created = _common(lambda leaf=leaf, raw=raw, label=label: common.write_create_only(normalized_fd, leaf, raw, label))
            descriptors[field] = created.descriptor(f"{NORMALIZED_DIRECTORY_NAME}/{leaf}", label)
        receipt = {
            "schema_version": NORMALIZATION_VERSION,
            "status": STATUS,
            "qualification_status": QUALIFICATION_STATUS,
            "authority": AUTHORITY,
            "capture_scope": CAPTURE_SCOPE,
            "reconstruction_id": reconstruction_id,
            "raw_archive_format": RAW_ARCHIVE_FORMAT,
            "normalized_archive_format": OCI_ARCHIVE_FORMAT,
            "source_layout": normalization.source_layout,
            "platform": dict(PLATFORM),
            "image_id": image_id,
            "content_binding": CONTENT_BINDING,
            "docker_invocation_binding": NOT_ESTABLISHED,
            "source_binding": NOT_ESTABLISHED,
            "bundle_binding": NOT_ESTABLISHED,
            "build_invocation_binding": NOT_ESTABLISHED,
            "runtime_capture_binding": NOT_ESTABLISHED,
            "independence_binding": NOT_ESTABLISHED,
            **{field: descriptor.as_json() for field, descriptor in descriptors.items()},
        }
        _common(
            lambda: common.write_create_only_json(
                root_fd,
                NORMALIZATION_NAME,
                receipt,
                "runtime image export normalization receipt",
            )
        )
        replayed = verify_reconstructed_runtime_image_export_oci_normalization_fd(root_fd)
        if replayed != receipt:
            _fail("prepublication-replay-drift", "held runtime image export normalization replay differs from draft receipt")
        return receipt
    finally:
        if normalized_fd is not None:
            os.close(normalized_fd)
        if raw_fd is not None:
            os.close(raw_fd)
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--image-inspect", type=Path, required=True)
    parser.add_argument("--image-export-archive", type=Path, required=True)
    parser.add_argument("--reconstruction-id", choices=("a", "b"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = prepare_reconstructed_runtime_image_export_oci_normalization(
            args.evidence_root,
            image_inspect=args.image_inspect,
            image_export_archive=args.image_export_archive,
            reconstruction_id=args.reconstruction_id,
        )
    except RuntimeImageExportNormalizationError as error:
        print(f"runtime image-export OCI normalization: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
