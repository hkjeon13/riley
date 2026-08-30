#!/usr/bin/env python3
"""Bind one already-captured reconstructed runtime-image assembly, safely.

This is a post-capture, source-only preparer.  It neither invokes Docker nor
builds an image, starts a container or service, accesses a GPU, or runs a
qualification gate.  A future reviewed host runner supplies one per-arm raw
USTAR capture.  This tool snapshots that file into a fresh private evidence
root and replays the reviewed source, reproducibility, OCI, recipe, context,
image, and never-started-container bindings through held descriptors.

The resulting receipt is deliberately a *single-arm structural binding*.  It
does not establish independent A/B runtime captures, runtime execution, GPU
behaviour, rollback, freeze, historical distribution, or qualification.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import os
import re
import stat
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterator, Mapping, NoReturn, Sequence

sys.dont_write_bytecode = True

import provenance_v2_common as common
import verify_reconstructed_runtime_assembly_dockerfile as assembly_recipe


RUNTIME_ASSEMBLY_CAPTURE_VERSION = "riley.reconstructed-runtime-assembly-capture.v1"
RUNTIME_ASSEMBLY_CAPTURE_NAME = "reconstructed-runtime-assembly-capture.json"
RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY = "runtime-assembly-capture"
RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE = "assembly-capture.tar"
CAPTURE_SCOPE = "single-arm-source-free-runtime-assembly-and-never-started-container-capture"
AUTHORITY = "raw-runtime-assembly-capture-structural-only"
STATUS = "bound"
QUALIFICATION_STATUS = "not-run"
PLATFORM = {"os": "linux", "architecture": "amd64"}

CAPTURE_INVOCATION_VERSION = "riley.reconstructed-runtime-assembly-build-invocation.v1"
OCI_EXPORT_INVOCATION_VERSION = "riley.reconstructed-runtime-assembly-oci-export-invocation.v1"
CAPTURE_COMPLETION_VERSION = "riley.reconstructed-runtime-assembly-capture-completion.v1"
CONTEXT_ARCHIVE_FORMAT = "ustar-v1"
OCI_ARCHIVE_FORMAT = "oci-image-layout-tar.v1"

MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES
# POSIX USTAR stores an ordinary-file size in eleven octal digits.  The OCI
# input closure itself permits larger archives, but this deliberately USTAR
# capture contract must reject any one nested OCI member above this representable
# bound rather than silently pretending a PAX/GNU extension is acceptable.
MAX_USTAR_REGULAR_MEMBER_BYTES = 8 ** 11 - 1
MAX_CONTEXT_ARCHIVE_BYTES = 3 * 1024 * 1024 * 1024
MAX_RUNTIME_TREE_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_OCI_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
MAX_CAPTURE_OCI_ARCHIVE_BYTES = MAX_USTAR_REGULAR_MEMBER_BYTES
MAX_BINARY_BYTES = 512 * 1024 * 1024
MAX_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
MAX_BUILD_LOG_BYTES = 16 * 1024 * 1024
MAX_DOCKERFILE_BYTES = 1024 * 1024
MAX_CANONICAL_TAR_TRAILER_BYTES = 20 * 512
# This is the exact sum of the fixed member ceilings after USTAR block padding,
# eleven 512-byte headers, the two required end blocks, and at most one
# conventional 20-block all-zero trailer.  Keeping the snapshot bound tied to
# the closed inventory prevents a mostly-zero 128-GiB input from consuming the
# private evidence volume before its raw grammar is rejected.
MAX_CAPTURE_ARCHIVE_BYTES = (
    MAX_RECEIPT_BYTES  # SHA256SUMS
    + 512  # build.iid rounds up to one USTAR block
    + MAX_BUILD_LOG_BYTES
    + 5 * MAX_RECEIPT_BYTES  # completion, invocation, container/image/export JSON
    + MAX_RUNTIME_TREE_ARCHIVE_BYTES
    + MAX_CONTEXT_ARCHIVE_BYTES
    + MAX_CAPTURE_OCI_ARCHIVE_BYTES + 1  # 8 GiB - 1 rounds up to 8 GiB
    + 11 * 512  # fixed member headers
    + 2 * 512  # required USTAR end marker
    + MAX_CANONICAL_TAR_TRAILER_BYTES
)
MAX_CAPTURE_MEMBERS = 11
MAX_CONTEXT_MEMBERS = 3
MAX_RUNTIME_TREE_MEMBERS = 8
TAR_BLOCK_BYTES = 512
MIN_TAR_END_BYTES = 2 * TAR_BLOCK_BYTES
TAR_SIZE_START = 124
TAR_SIZE_END = 136
TAR_MODE_START = 100
TAR_MODE_END = 108
TAR_UID_START = 108
TAR_UID_END = 116
TAR_GID_START = 116
TAR_GID_END = 124
TAR_MTIME_START = 136
TAR_MTIME_END = 148
TAR_CHECKSUM_START = 148
TAR_CHECKSUM_END = 156
TAR_TYPE_OFFSET = 156
TAR_MAGIC_START = 257
TAR_MAGIC_END = 263
TAR_VERSION_START = 263
TAR_VERSION_END = 265
TAR_LINKNAME_START = 157
TAR_LINKNAME_END = 257
TAR_PREFIX_START = 345
TAR_PREFIX_END = 500
TAR_UNAME_START = 265
TAR_UNAME_END = 297
TAR_GNAME_START = 297
TAR_GNAME_END = 329

CAPTURE_MEMBER_NAMES = (
    "SHA256SUMS",
    "build.iid",
    "build.log",
    "capture-completion.json",
    "capture-invocation.json",
    "container-inspect.json",
    "container-opt-riley.tar",
    "context.tar",
    "image-inspect.json",
    "oci-export-invocation.json",
    "oci-image-layout.tar",
)
CAPTURE_COMPLETION_MEMBER_NAMES = tuple(
    name for name in CAPTURE_MEMBER_NAMES if name not in {"SHA256SUMS", "capture-completion.json"}
)
CONTEXT_MEMBER_NAMES = ("Dockerfile", "input/riley", "input/riley.tar.gz")

SHA256SUM_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
# The final image is a closed runtime recipe. A blacklist is not sufficient:
# an inherited loader, allocator, interpreter, shell, or application setting
# can alter behavior without using one of the currently known names. Require
# the recipe's exact three entries instead.
EXPECTED_IMAGE_ENVIRONMENT = {
    "PATH": "/opt/riley/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
}
EXPECTED_RUNTIME_USER = "65532:65532"
EXPECTED_RUNTIME_ENTRYPOINT = ("/opt/riley/bin/riley",)
EXPECTED_RUNTIME_COMMAND = ("--help",)
CONTAINER_EMPTY_HOST_FIELDS = (
    "Binds",
    "VolumesFrom",
    "Tmpfs",
    "Devices",
    "DeviceRequests",
    "Mounts",
    "CapAdd",
    "CapDrop",
    "SecurityOpt",
    "DeviceCgroupRules",
    "ExtraHosts",
    "PortBindings",
)
# Docker represents its default private namespace modes differently across
# daemon releases (empty string versus ``private``). Both are harmless here;
# the host and another container's namespaces are not.
CONTAINER_SAFE_NAMESPACE_MODES = {
    "PidMode": (None, "", "private"),
    "IpcMode": (None, "", "private"),
    "UTSMode": (None, "", "private"),
    "UsernsMode": (None, "", "private"),
    "CgroupnsMode": (None, "", "private"),
}


class RuntimeAssemblyCaptureError(common.ProvenanceV2Error):
    """One runtime assembly capture or its evidence roots are unsafe."""


def _fail(code: str, message: str) -> NoReturn:
    error = RuntimeAssemblyCaptureError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _dependencies() -> tuple[Any, Any, Any]:
    """Import PR16-dependent adapters only when a full replay is requested.

    The remote source host intentionally still has Python 3.10 whereas the
    PR16 release checker requires ``tomllib``.  Keeping this import lazy lets
    ``--help`` and static parsing remain usable there; CI uses Python 3.13.
    """

    try:
        source_inputs = importlib.import_module("prepare_reconstructed_rc2_inputs_v1")
        repro_inputs = importlib.import_module("prepare_reconstructed_repro_build_inputs_v1")
        runtime_oci = importlib.import_module("prepare_reconstructed_runtime_oci_inputs_v1")
    except ModuleNotFoundError as error:
        if error.name == "tomllib":
            _fail(
                "unsupported-python-runtime",
                "runtime assembly capture replay requires Python 3.11+ because the reviewed PR16 checker uses tomllib",
            )
        raise
    return source_inputs, repro_inputs, runtime_oci


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


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_disjoint_paths(paths: Mapping[str, Path]) -> None:
    values = tuple(paths.items())
    for index, (left_label, left) in enumerate(values):
        for right_label, right in values[index + 1 :]:
            if _paths_overlap(left, right):
                _fail(
                    "capture-root-overlap",
                    f"{left_label} and {right_label} must be disjoint normalized paths",
                )


def _root_identity(directory_fd: int, label: str) -> tuple[int, int]:
    try:
        metadata = os.fstat(directory_fd)
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot inspect {label}: {error}")
    return metadata.st_dev, metadata.st_ino


def _require_distinct_root_fds(roots: Mapping[str, int]) -> None:
    observed: dict[tuple[int, int], str] = {}
    for label, directory_fd in roots.items():
        identity = _root_identity(directory_fd, label)
        prior = observed.get(identity)
        if prior is not None:
            _fail("input-root-alias", f"{label} aliases the already-held {prior}")
        observed[identity] = label


def _close_fds(fds: Mapping[str, int]) -> None:
    for descriptor in reversed(tuple(fds.values())):
        os.close(descriptor)


def _open_external_roots(
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root: Path,
) -> dict[str, int]:
    opened: dict[str, int] = {}
    try:
        for label, path in (
            ("source inputs root", source_input_root),
            ("reproducibility inputs root", repro_build_input_root),
            ("runtime OCI inputs root", runtime_oci_input_root),
        ):
            opened[label] = _common(lambda path=path, label=label: common.open_private_evidence_directory(path, label))
        _require_distinct_root_fds(opened)
        return opened
    except BaseException:
        _close_fds(opened)
        raise


def _exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else []
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; expected={sorted(expected)}, actual={actual}",
        )
    return value


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail("invalid-sha256", f"{label} must be a non-zero lowercase SHA-256")
    return value


def _image_id(value: Any, label: str) -> str:
    if type(value) is not str or IMAGE_ID_RE.fullmatch(value) is None or value == "sha256:" + "0" * 64:
        _fail("invalid-image-id", f"{label} must be a non-zero lowercase sha256 image ID")
    return value


def _container_id(value: Any, label: str) -> str:
    if type(value) is not str or CONTAINER_ID_RE.fullmatch(value) is None or value == "0" * 64:
        _fail("invalid-container-id", f"{label} must be a non-zero lowercase 64-hex container ID")
    return value


def _descriptor(value: Any, label: str, *, expected_path: str | None = None, maximum_bytes: int | None = None) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if expected_path is not None and descriptor.path != expected_path:
        _fail("descriptor-path-mismatch", f"{label} must have fixed path {expected_path!r}")
    if descriptor.byte_length < 1:
        _fail("empty-descriptor", f"{label} must describe nonempty evidence")
    if maximum_bytes is not None and descriptor.byte_length > maximum_bytes:
        _fail("descriptor-size", f"{label} exceeds its byte bound")
    return descriptor


@dataclass(frozen=True)
class ExternalFacts:
    reconstruction_id: str
    source_inputs: dict[str, Any]
    repro_inputs: dict[str, Any]
    runtime_oci_inputs: dict[str, Any]
    source_revision: str
    expected_source_archive_sha256: str
    repro_receipt: common.EvidenceDescriptor
    binary: common.EvidenceDescriptor
    bundle: common.EvidenceDescriptor
    oci_image_inspect: common.EvidenceDescriptor
    oci_archive: common.EvidenceDescriptor
    image_id: str
    repro_root_fd: int


def _replay_external_inputs(
    *,
    source_root_fd: int,
    repro_root_fd: int,
    runtime_oci_root_fd: int,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    reconstruction_id: str,
) -> ExternalFacts:
    if reconstruction_id not in {"a", "b"}:
        _fail("invalid-reconstruction-id", "--reconstruction-id must be a or b")
    source_inputs, repro_inputs, runtime_oci = _dependencies()
    _require_distinct_root_fds(
        {
            "source inputs root": source_root_fd,
            "reproducibility inputs root": repro_root_fd,
            "runtime OCI inputs root": runtime_oci_root_fd,
        }
    )
    source_row = _common(
        lambda: source_inputs.verify_reconstructed_rc2_inputs_fd(
            source_root_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
        )
    )
    repro_row = _common(
        lambda: repro_inputs.verify_reconstructed_repro_build_inputs_fd(
            repro_root_fd,
            source_root_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
        )
    )
    oci_row = _common(lambda: runtime_oci.verify_reconstructed_runtime_oci_inputs_fd(runtime_oci_root_fd))
    if oci_row.get("reconstruction_id") != reconstruction_id:
        _fail("reconstruction-id-mismatch", "runtime OCI inputs must match --reconstruction-id")
    if oci_row.get("platform") != PLATFORM:
        _fail("runtime-oci-platform-mismatch", "runtime OCI inputs must be linux/amd64")

    source_receipt = _common(
        lambda: common.describe_regular_relative(
            source_root_fd,
            source_inputs.SOURCE_INPUTS_NAME,
            "reviewed source inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    source_projection = {
        "receipt": source_receipt.as_json(),
        "expected_source_archive_sha256": expected_source_archive_sha256,
        "git_identity": source_row["git_identity"],
        "source": source_row["source"],
    }
    if repro_row.get("source_inputs") != source_projection:
        _fail("source-inputs-binding-mismatch", "reproducibility inputs no longer bind the held source inputs")
    source_revision = source_row["git_identity"].get("target_commit_sha1")
    if type(source_revision) is not str or source_revision != repro_row["reproducibility_contract"].get("source_revision"):
        _fail("source-revision-mismatch", "reproducibility contract differs from reviewed source target")

    repro_receipt = _common(
        lambda: common.describe_regular_relative(
            repro_root_fd,
            repro_inputs.REPRO_BUILD_INPUTS_NAME,
            "reproducibility inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    selected = _exact(
        repro_row.get("builds", {}).get(reconstruction_id),
        {"reconstruction_id", "evidence_build_id", "evidence_archive", "build_manifest", "binary", "bundle"},
        f"reproducibility build {reconstruction_id}",
    )
    if selected["reconstruction_id"] != reconstruction_id:
        _fail("reconstruction-id-mismatch", "selected reproducibility build has the wrong arm")
    binary = _descriptor(
        selected["binary"],
        "selected release binary",
        expected_path=f"repro-builds/{reconstruction_id}/riley",
        maximum_bytes=MAX_BINARY_BYTES,
    )
    bundle = _descriptor(
        selected["bundle"],
        "selected release bundle",
        expected_path=f"repro-builds/{reconstruction_id}/riley.tar.gz",
        maximum_bytes=MAX_BUNDLE_BYTES,
    )
    oci_receipt = _common(
        lambda: common.describe_regular_relative(
            runtime_oci_root_fd,
            runtime_oci.RUNTIME_OCI_INPUTS_NAME,
            "runtime OCI inputs receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    oci_image_inspect = _descriptor(
        oci_row["image_inspect"],
        "runtime OCI raw image inspect",
        expected_path="runtime-image/docker-image-inspect.json",
        maximum_bytes=MAX_RECEIPT_BYTES,
    )
    oci_archive = _descriptor(
        oci_row["archive"],
        "runtime OCI archive",
        expected_path="runtime-image/oci-image-layout.tar",
        maximum_bytes=MAX_CAPTURE_OCI_ARCHIVE_BYTES,
    )
    image_id = _image_id(oci_row.get("image_id"), "runtime OCI image ID")
    repro_projection = {
        "receipt": repro_receipt.as_json(),
        "reproducibility_contract": repro_row["reproducibility_contract"],
        "selected_build": {
            "reconstruction_id": selected["reconstruction_id"],
            "evidence_build_id": selected["evidence_build_id"],
            "binary": binary.as_json(),
            "bundle": bundle.as_json(),
        },
    }
    runtime_oci_projection = {
        "receipt": oci_receipt.as_json(),
        "reconstruction_id": reconstruction_id,
        "image_id": image_id,
        **{
            name: _descriptor(oci_row[name], f"runtime OCI {name}").as_json()
            for name in ("image_inspect", "archive", "layout", "index", "manifest", "config")
        },
    }
    return ExternalFacts(
        reconstruction_id=reconstruction_id,
        source_inputs=source_projection,
        repro_inputs=repro_projection,
        runtime_oci_inputs=runtime_oci_projection,
        source_revision=source_revision,
        expected_source_archive_sha256=expected_source_archive_sha256,
        repro_receipt=repro_receipt,
        binary=binary,
        bundle=bundle,
        oci_image_inspect=oci_image_inspect,
        oci_archive=oci_archive,
        image_id=image_id,
        repro_root_fd=repro_root_fd,
    )


@dataclass(frozen=True)
class UstarMember:
    name: str
    kind: str
    mode: int
    uid: int
    gid: int
    mtime: int
    uname: str
    gname: str
    byte_length: int
    offset_data: int


def _read_exact(stream: BinaryIO, count: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        try:
            chunk = stream.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
        except OSError as error:
            _fail("unreadable-capture", f"cannot read {label}: {error}")
        if not chunk:
            _fail("truncated-capture", f"{label} is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _seek(stream: BinaryIO, offset: int, label: str) -> None:
    try:
        stream.seek(offset, os.SEEK_SET)
    except OSError as error:
        _fail("unreadable-capture", f"cannot seek {label}: {error}")


def _tar_octal(raw: bytes, label: str) -> int:
    if not raw or raw[0] & 0x80:
        _fail("unsupported-tar-extension", f"{label} uses a non-octal tar field")
    text = raw.rstrip(b"\x00 ").lstrip(b" ")
    if not text:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in text):
        _fail("invalid-tar", f"{label} is not a valid octal tar field")
    return int(text, 8)


def _tar_text(raw: bytes, label: str) -> str:
    marker = raw.find(b"\x00")
    if marker >= 0:
        if any(raw[marker + 1 :]):
            _fail("invalid-tar", f"{label} has non-zero bytes after its NUL terminator")
        raw = raw[:marker]
    if not raw:
        return ""
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("invalid-tar", f"{label} must be ASCII: {error}")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        _fail("invalid-tar", f"{label} contains a control character")
    return value


def _safe_member_name(name: str, label: str, *, directory: bool) -> str:
    if not name or "\x00" in name or name.startswith("/") or "\\" in name or "//" in name:
        _fail("unsafe-tar-member", f"{label} has an unsafe archive path")
    normalized = name.rstrip("/") if directory else name
    if not normalized or (not directory and name.endswith("/")):
        _fail("unsafe-tar-member", f"{label} has an invalid archive path")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        _fail("unsafe-tar-member", f"{label} has a traversal archive path")
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        _fail("unsafe-tar-member", f"{label} has a traversal archive path")
    return normalized


def _verify_tar_checksum(header: bytes, label: str) -> None:
    stored = _tar_octal(header[TAR_CHECKSUM_START:TAR_CHECKSUM_END], f"{label} checksum")
    candidate = header[:TAR_CHECKSUM_START] + b" " * (TAR_CHECKSUM_END - TAR_CHECKSUM_START) + header[TAR_CHECKSUM_END:]
    if sum(candidate) != stored:
        _fail("invalid-tar", f"{label} has an invalid tar checksum")


def _all_zero(stream: BinaryIO, remaining: int, label: str) -> None:
    while remaining:
        chunk = _read_exact(stream, min(common.DEFAULT_READ_CHUNK_BYTES, remaining), label)
        if any(chunk):
            _fail("invalid-tar", f"{label} has non-zero trailing bytes")
        remaining -= len(chunk)


def _scan_ustar(
    stream: BinaryIO,
    *,
    start: int,
    byte_length: int,
    label: str,
    maximum_bytes: int,
    maximum_members: int,
    maximum_total_member_bytes: int,
    allow_directories: bool,
    maximum_trailer_bytes: int = MAX_CANONICAL_TAR_TRAILER_BYTES,
) -> list[UstarMember]:
    """Preflight a bounded USTAR grammar without letting tarfile parse it.

    GNU/PAX/sparse/link/device headers are rejected from raw header bytes
    before any parser can materialize their extension payload.  The capture
    grammars use only short USTAR names, ordinary files, and (for the runtime
    tree only) zero-payload directories.
    """

    if byte_length < MIN_TAR_END_BYTES or byte_length > maximum_bytes or byte_length % TAR_BLOCK_BYTES:
        _fail("invalid-tar", f"{label} has an invalid bounded tar length")
    _seek(stream, start, label)
    offset = 0
    total_member_bytes = 0
    members: list[UstarMember] = []
    zero_block = b"\x00" * TAR_BLOCK_BYTES
    while offset < byte_length:
        header = _read_exact(stream, TAR_BLOCK_BYTES, f"{label} header")
        offset += TAR_BLOCK_BYTES
        if header == zero_block:
            second = _read_exact(stream, TAR_BLOCK_BYTES, f"{label} end marker")
            offset += TAR_BLOCK_BYTES
            if second != zero_block:
                _fail("invalid-tar", f"{label} has an invalid two-block end marker")
            trailer_bytes = byte_length - offset
            if trailer_bytes > maximum_trailer_bytes:
                _fail("tar-trailer-size", f"{label} has an excessive all-zero tar trailer")
            _all_zero(stream, trailer_bytes, f"{label} trailer")
            return members
        _verify_tar_checksum(header, f"{label} member {len(members)}")
        if header[TAR_MAGIC_START:TAR_MAGIC_END] != b"ustar\x00" or header[TAR_VERSION_START:TAR_VERSION_END] != b"00":
            _fail("unsupported-tar-extension", f"{label} must use the USTAR v00 header format")
        if any(header[TAR_PREFIX_START:TAR_PREFIX_END]):
            _fail("unsupported-tar-extension", f"{label} must not use USTAR path prefixes")
        if any(header[TAR_LINKNAME_START:TAR_LINKNAME_END]):
            _fail("unsupported-tar-extension", f"{label} must not contain tar link targets")
        member_type = header[TAR_TYPE_OFFSET:TAR_TYPE_OFFSET + 1]
        if member_type in {b"\x00", b"0"}:
            kind = "file"
        elif member_type == b"5" and allow_directories:
            kind = "directory"
        else:
            _fail("unsupported-tar-extension", f"{label} contains a non-regular tar member type")
        name = _safe_member_name(
            _tar_text(header[:100], f"{label} member name"),
            f"{label} member",
            directory=kind == "directory",
        )
        byte_count = _tar_octal(header[TAR_SIZE_START:TAR_SIZE_END], f"{label} member size")
        if byte_count > MAX_USTAR_REGULAR_MEMBER_BYTES:
            _fail("ustar-member-size", f"{label} member {name!r} exceeds the USTAR regular-file size limit")
        if kind == "directory" and byte_count != 0:
            _fail("invalid-tar", f"{label} directory {name!r} has payload bytes")
        mode = _tar_octal(header[TAR_MODE_START:TAR_MODE_END], f"{label} member mode")
        uid = _tar_octal(header[TAR_UID_START:TAR_UID_END], f"{label} member uid")
        gid = _tar_octal(header[TAR_GID_START:TAR_GID_END], f"{label} member gid")
        mtime = _tar_octal(header[TAR_MTIME_START:TAR_MTIME_END], f"{label} member mtime")
        uname = _tar_text(header[TAR_UNAME_START:TAR_UNAME_END], f"{label} member uname")
        gname = _tar_text(header[TAR_GNAME_START:TAR_GNAME_END], f"{label} member gname")
        padded = ((byte_count + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES) * TAR_BLOCK_BYTES
        if padded > byte_length - offset:
            _fail("truncated-tar", f"{label} member {name!r} payload is truncated")
        if len(members) >= maximum_members:
            _fail("tar-member-count", f"{label} exceeds its member-count bound")
        total_member_bytes += byte_count
        if total_member_bytes > maximum_total_member_bytes:
            _fail("tar-total-size", f"{label} exceeds its total member-byte bound")
        members.append(
            UstarMember(
                name=name,
                kind=kind,
                mode=mode,
                uid=uid,
                gid=gid,
                mtime=mtime,
                uname=uname,
                gname=gname,
                byte_length=byte_count,
                offset_data=start + offset,
            )
        )
        try:
            stream.seek(padded, os.SEEK_CUR)
        except OSError as error:
            _fail("unreadable-capture", f"cannot skip {label} member {name!r}: {error}")
        offset += padded
    _fail("truncated-tar", f"{label} has no two-block tar end marker")


def _hash_member(stream: BinaryIO, member: UstarMember, label: str) -> str:
    _seek(stream, member.offset_data, label)
    digest = hashlib.sha256()
    remaining = member.byte_length
    while remaining:
        chunk = _read_exact(stream, min(common.DEFAULT_READ_CHUNK_BYTES, remaining), label)
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _read_member(stream: BinaryIO, member: UstarMember, label: str, *, maximum_bytes: int) -> bytes:
    if member.byte_length > maximum_bytes:
        _fail("tar-member-size", f"{label} exceeds its byte bound")
    _seek(stream, member.offset_data, label)
    return _read_exact(stream, member.byte_length, label)


def _member_map(members: Sequence[UstarMember], expected_names: Sequence[str], label: str) -> dict[str, UstarMember]:
    names = tuple(member.name for member in members)
    if names != tuple(expected_names):
        _fail("tar-inventory", f"{label} members differ from the fixed canonical inventory")
    if any(member.kind != "file" for member in members):
        _fail("tar-member-type", f"{label} must contain only ordinary regular files")
    return {member.name: member for member in members}


def _require_canonical_capture_metadata(member: UstarMember, label: str) -> None:
    if (
        member.mode != 0o644
        or member.uid != 0
        or member.gid != 0
        or member.mtime != 0
        or member.uname
        or member.gname
    ):
        _fail("noncanonical-capture-tar", f"{label} must use mode 0644, uid/gid 0, mtime 0, and empty names")


def _capture_member_bound(name: str) -> int:
    limits = {
        "SHA256SUMS": MAX_RECEIPT_BYTES,
        "build.iid": 80,
        "build.log": MAX_BUILD_LOG_BYTES,
        "capture-completion.json": MAX_RECEIPT_BYTES,
        "capture-invocation.json": MAX_RECEIPT_BYTES,
        "container-inspect.json": MAX_RECEIPT_BYTES,
        "container-opt-riley.tar": MAX_RUNTIME_TREE_ARCHIVE_BYTES,
        "context.tar": MAX_CONTEXT_ARCHIVE_BYTES,
        "image-inspect.json": MAX_RECEIPT_BYTES,
        "oci-export-invocation.json": MAX_RECEIPT_BYTES,
        "oci-image-layout.tar": MAX_CAPTURE_OCI_ARCHIVE_BYTES,
    }
    return limits[name]


def _parse_checksums(raw: bytes, expected_names: Sequence[str]) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        _fail("invalid-checksums", f"SHA256SUMS must be ASCII: {error}")
    if not text.endswith("\n"):
        _fail("invalid-checksums", "SHA256SUMS must end with a newline")
    entries: dict[str, str] = {}
    ordered: list[str] = []
    for line in text.splitlines():
        match = SHA256SUM_LINE.fullmatch(line)
        if match is None:
            _fail("invalid-checksums", "SHA256SUMS has an invalid line")
        digest, name = match.groups()
        if name in entries:
            _fail("invalid-checksums", "SHA256SUMS contains a duplicate member")
        entries[name] = digest
        ordered.append(name)
    if tuple(ordered) != tuple(expected_names):
        _fail("invalid-checksums", "SHA256SUMS must list exactly the canonical member order")
    return entries


def _strict_json(raw: bytes, label: str, *, canonical: bool) -> Any:
    if canonical:
        return _common(lambda: common.parse_canonical_json(raw, label, maximum_bytes=MAX_RECEIPT_BYTES))
    return _common(
        lambda: common.parse_strict_json(
            raw,
            label,
            maximum_bytes=MAX_RECEIPT_BYTES,
            require_object=False,
        )
    )


def _validate_captured_recipe(raw: bytes) -> None:
    if not raw or len(raw) > MAX_DOCKERFILE_BYTES:
        _fail("recipe-size", "captured Dockerfile must be nonempty and bounded")
    with tempfile.TemporaryDirectory(prefix="riley-runtime-recipe-") as directory:
        path = Path(directory) / "Dockerfile"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(descriptor, raw[offset:])
                    if written <= 0:
                        _fail("temporary-write", "cannot materialize captured Dockerfile")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            _fail("temporary-write", f"cannot materialize captured Dockerfile: {error}")
        try:
            assembly_recipe.verify_reconstructed_runtime_assembly_dockerfile(path)
        except assembly_recipe.RuntimeAssemblyContractError as error:
            _fail("invalid-assembly-recipe", str(error))


@dataclass(frozen=True)
class TreeEntry:
    name: str
    kind: str
    mode: int
    sha256: str | None
    byte_length: int


def _hash_tar_stream_member(archive: tarfile.TarFile, member: tarfile.TarInfo, label: str) -> str:
    source = archive.extractfile(member)
    if source is None:
        _fail("invalid-bundle", f"cannot read private bundle member {label}")
    digest = hashlib.sha256()
    remaining = member.size
    while remaining:
        chunk = source.read(min(common.DEFAULT_READ_CHUNK_BYTES, remaining))
        if not chunk:
            _fail("invalid-bundle", f"private bundle member {label} is truncated")
        digest.update(chunk)
        remaining -= len(chunk)
    if source.read(1):
        _fail("invalid-bundle", f"private bundle member {label} has unexpected trailing data")
    return digest.hexdigest()


@contextlib.contextmanager
def _materialized_private_bundle(
    repro_root_fd: int,
    bundle: common.EvidenceDescriptor,
) -> Iterator[Path]:
    """Materialize a held snapshot only into a fresh private checker root."""

    temporary_parent = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="riley-runtime-bundle-", dir=temporary_parent) as directory:
        root = Path(directory).resolve()
        try:
            os.chmod(root, 0o700)
        except OSError as error:
            _fail("temporary-mode", f"cannot secure private bundle checker directory: {error}")
        root_fd = _common(lambda: common.open_private_evidence_directory(root, "private bundle checker root"))
        try:
            _common(
                lambda: common.materialize_descriptor_runtime_copy(
                    repro_root_fd,
                    bundle,
                    root_fd,
                    "selected-bundle.tar.gz",
                    "selected release bundle",
                    maximum_bytes=MAX_BUNDLE_BYTES,
                )
            )
            yield root / "selected-bundle.tar.gz"
        finally:
            os.close(root_fd)


def _bundle_runtime_tree(repro_root_fd: int, bundle: common.EvidenceDescriptor) -> dict[str, TreeEntry]:
    """Return the bundle's strip-root runtime tree after private verification."""

    try:
        verify_release_bundle = importlib.import_module("verify_reconstructed_rc2_pr16_bundle_v1")
    except ModuleNotFoundError as error:
        if error.name == "tomllib":
            _fail("unsupported-python-runtime", "release bundle replay requires Python 3.11+")
        raise
    with _materialized_private_bundle(repro_root_fd, bundle) as path:
        try:
            verify_release_bundle.verify_reconstructed_rc2_pr16_bundle(path)
        except Exception as error:  # The private copy is only useful if the reviewed bundle verifier accepts it.
            _fail("invalid-bundle", f"selected private release bundle does not replay: {error}")
        try:
            archive = tarfile.open(path, mode="r|gz")
        except (OSError, tarfile.TarError) as error:
            _fail("invalid-bundle", f"cannot reopen selected private release bundle: {error}")
        tree: dict[str, TreeEntry] = {}
        root_name: str | None = None
        try:
            for member in archive:
                raw_name = member.name.rstrip("/") if member.isdir() else member.name
                if not raw_name or "\\" in raw_name or raw_name.startswith("/"):
                    _fail("invalid-bundle", "selected private release bundle has an unsafe tree path")
                parts = PurePosixPath(raw_name).parts
                if any(part in {"", ".", ".."} for part in parts):
                    _fail("invalid-bundle", "selected private release bundle has a traversal tree path")
                if root_name is None:
                    root_name = parts[0]
                if parts[0] != root_name:
                    _fail("invalid-bundle", "selected private release bundle has more than one root")
                if len(parts) == 1:
                    if not member.isdir():
                        _fail("invalid-bundle", "selected private release bundle root is not a directory")
                    continue
                relative = "/".join(parts[1:])
                if relative in tree:
                    _fail("invalid-bundle", "selected private release bundle has duplicate runtime paths")
                if member.isdir():
                    if member.size != 0:
                        _fail("invalid-bundle", "selected private release bundle directory has payload bytes")
                    tree[relative] = TreeEntry(relative, "directory", member.mode, None, 0)
                elif member.isreg():
                    tree[relative] = TreeEntry(
                        relative,
                        "file",
                        member.mode,
                        _hash_tar_stream_member(archive, member, relative),
                        member.size,
                    )
                else:
                    _fail("invalid-bundle", "selected private release bundle has a non-regular runtime member")
        except tarfile.TarError as error:
            _fail("invalid-bundle", f"cannot parse selected private release bundle: {error}")
        finally:
            archive.close()
    if not tree:
        _fail("invalid-bundle", "selected private release bundle has no runtime tree")
    return tree


def _tree_summary(tree: Mapping[str, TreeEntry]) -> dict[str, Any]:
    rows = [
        {
            "path": entry.name,
            "kind": entry.kind,
            "mode": entry.mode,
            "sha256": entry.sha256,
            "byte_length": entry.byte_length,
        }
        for _name, entry in sorted(tree.items())
    ]
    raw = common.canonical_json_bytes(rows)
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entry_count": len(rows),
        "byte_length": sum(entry.byte_length for entry in tree.values()),
    }


def _validate_context(
    stream: BinaryIO,
    member: UstarMember,
    external: ExternalFacts,
) -> dict[str, Any]:
    members = _scan_ustar(
        stream,
        start=member.offset_data,
        byte_length=member.byte_length,
        label="captured build context",
        maximum_bytes=MAX_CONTEXT_ARCHIVE_BYTES,
        maximum_members=MAX_CONTEXT_MEMBERS,
        maximum_total_member_bytes=MAX_BINARY_BYTES + MAX_BUNDLE_BYTES + MAX_DOCKERFILE_BYTES,
        allow_directories=False,
    )
    indexed = _member_map(members, CONTEXT_MEMBER_NAMES, "captured build context")
    for context_member in members:
        _require_canonical_capture_metadata(context_member, f"captured build context {context_member.name}")
    dockerfile = _read_member(stream, indexed["Dockerfile"], "captured Dockerfile", maximum_bytes=MAX_DOCKERFILE_BYTES)
    _validate_captured_recipe(dockerfile)
    binary = indexed["input/riley"]
    bundle = indexed["input/riley.tar.gz"]
    binary_sha = _hash_member(stream, binary, "captured context binary")
    bundle_sha = _hash_member(stream, bundle, "captured context bundle")
    if binary.byte_length != external.binary.byte_length or binary_sha != external.binary.sha256:
        _fail("context-binary-mismatch", "captured build context binary differs from the selected reproducibility binary")
    if bundle.byte_length != external.bundle.byte_length or bundle_sha != external.bundle.sha256:
        _fail("context-bundle-mismatch", "captured build context bundle differs from the selected reproducibility bundle")
    return {
        "dockerfile_normalized_instructions_sha256": assembly_recipe.EXPECTED_NORMALIZED_INSTRUCTION_SHA256,
        "binary": {"sha256": binary_sha, "byte_length": binary.byte_length},
        "bundle": {"sha256": bundle_sha, "byte_length": bundle.byte_length},
    }


def _expected_build_argv(external: ExternalFacts, context_member: UstarMember, context_sha256: str) -> list[str]:
    build_args = (
        ("RILEY_RECONSTRUCTION_ID", external.reconstruction_id),
        ("RILEY_SOURCE_REVISION", external.source_revision),
        ("RILEY_SOURCE_ARCHIVE_SHA256", external.expected_source_archive_sha256),
        ("RILEY_REPRO_BUILD_INPUTS_SHA256", external.repro_receipt.sha256),
        ("RILEY_RELEASE_BINARY_SHA256", external.binary.sha256),
        ("RILEY_RELEASE_BUNDLE_SHA256", external.bundle.sha256),
        ("RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256", assembly_recipe.EXPECTED_NORMALIZED_INSTRUCTION_SHA256),
    )
    argv = [
        "docker",
        "build",
        "--file",
        "Dockerfile",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--pull=false",
        "--no-cache",
        "--iidfile",
        "build.iid",
    ]
    for name, value in build_args:
        argv.extend(("--build-arg", f"{name}={value}"))
    argv.append("-")
    return argv


def _validate_invocation(
    raw: bytes,
    external: ExternalFacts,
    context_member: UstarMember,
    context_sha256: str,
) -> None:
    row = _exact(
        _strict_json(raw, "captured build invocation", canonical=True),
        {"schema_version", "argv", "stdin"},
        "captured build invocation",
    )
    if row["schema_version"] != CAPTURE_INVOCATION_VERSION:
        _fail("invalid-invocation", "captured build invocation has an unexpected schema version")
    if row["argv"] != _expected_build_argv(external, context_member, context_sha256):
        _fail("build-invocation-mismatch", "captured build invocation differs from the closed source-free command")
    stdin = _exact(row["stdin"], {"member", "format", "sha256", "byte_length"}, "captured build invocation.stdin")
    if stdin != {
        "member": "context.tar",
        "format": CONTEXT_ARCHIVE_FORMAT,
        "sha256": context_sha256,
        "byte_length": context_member.byte_length,
    }:
        _fail("build-context-mismatch", "captured build invocation does not bind the canonical context tar")


def _validate_oci_export_invocation(raw: bytes, image_id: str) -> None:
    row = _exact(
        _strict_json(raw, "captured OCI export invocation", canonical=True),
        {"schema_version", "source_image_id", "output_member", "format", "platform"},
        "captured OCI export invocation",
    )
    if row != {
        "schema_version": OCI_EXPORT_INVOCATION_VERSION,
        "source_image_id": image_id,
        "output_member": "oci-image-layout.tar",
        "format": OCI_ARCHIVE_FORMAT,
        "platform": PLATFORM,
    }:
        _fail("oci-export-invocation-mismatch", "captured OCI export invocation differs from the closed capture contract")


def _string_list(value: Any, label: str, *, invalid_code: str = "invalid-image-inspect") -> list[str]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        _fail(invalid_code, f"{label} must be a JSON string array")
    return value


def _empty_container_option(value: Any) -> bool:
    return value is None or value == [] or value == {}


def _validate_expected_runtime_environment(
    value: Any,
    label: str,
    *,
    invalid_code: str,
    mismatch_code: str,
) -> None:
    environment = _string_list(value, label, invalid_code=invalid_code)
    parsed_environment: dict[str, str] = {}
    for item in environment:
        if "=" not in item:
            _fail(invalid_code, f"{label} contains a malformed entry")
        name, environment_value = item.split("=", 1)
        if not name or name in parsed_environment:
            _fail(invalid_code, f"{label} has an empty or duplicate name")
        parsed_environment[name] = environment_value
    if parsed_environment != EXPECTED_IMAGE_ENVIRONMENT:
        missing_environment = sorted(set(EXPECTED_IMAGE_ENVIRONMENT) - set(parsed_environment))
        extra_environment = sorted(set(parsed_environment) - set(EXPECTED_IMAGE_ENVIRONMENT))
        mismatched_environment = sorted(
            name
            for name in set(EXPECTED_IMAGE_ENVIRONMENT) & set(parsed_environment)
            if parsed_environment[name] != EXPECTED_IMAGE_ENVIRONMENT[name]
        )
        _fail(
            mismatch_code,
            "captured runtime config must contain exactly the reviewed runtime environment; "
            f"missing={missing_environment}, extra={extra_environment}, mismatched={mismatched_environment}",
        )


def _validate_expected_runtime_config(
    config: Mapping[str, Any],
    label: str,
    *,
    invalid_code: str,
    mismatch_code: str,
    environment_mismatch_code: str,
) -> None:
    if config.get("User") != EXPECTED_RUNTIME_USER:
        _fail(mismatch_code, f"{label}.User must retain the recipe's non-root user")
    if _string_list(config.get("Entrypoint"), f"{label}.Entrypoint", invalid_code=invalid_code) != list(
        EXPECTED_RUNTIME_ENTRYPOINT
    ):
        _fail(mismatch_code, f"{label}.Entrypoint differs from the reviewed recipe")
    if _string_list(config.get("Cmd"), f"{label}.Cmd", invalid_code=invalid_code) != list(EXPECTED_RUNTIME_COMMAND):
        _fail(mismatch_code, f"{label}.Cmd differs from the reviewed recipe")
    _validate_expected_runtime_environment(
        config.get("Env"),
        f"{label}.Env",
        invalid_code=invalid_code,
        mismatch_code=environment_mismatch_code,
    )
    if config.get("WorkingDir") not in (None, ""):
        _fail(mismatch_code, f"{label}.WorkingDir must not add a working directory")
    if not _empty_container_option(config.get("Volumes")):
        _fail(mismatch_code, f"{label}.Volumes must not declare volumes")
    # A healthcheck executes independently of the reviewed entrypoint. The
    # source-free image and its never-started capture must not retain one.
    if config.get("Healthcheck") is not None:
        _fail(mismatch_code, f"{label}.Healthcheck must be absent")
    # The recipe does not use deferred parent-image build instructions either.
    # Accept an absent/empty Docker representation only.
    if config.get("OnBuild") not in (None, []):
        _fail(mismatch_code, f"{label}.OnBuild must be absent")


def _image_labels(external: ExternalFacts) -> dict[str, str]:
    return {
        "org.riley.reconstructed-runtime-assembly.version": "v1",
        "org.riley.reconstructed-runtime-assembly.reconstruction-id": external.reconstruction_id,
        "org.riley.reconstructed-runtime-assembly.source-revision": external.source_revision,
        "org.riley.reconstructed-runtime-assembly.source-archive-sha256": external.expected_source_archive_sha256,
        "org.riley.reconstructed-runtime-assembly.repro-build-inputs-sha256": external.repro_receipt.sha256,
        "org.riley.reconstructed-runtime-assembly.release-binary-sha256": external.binary.sha256,
        "org.riley.reconstructed-runtime-assembly.release-bundle-sha256": external.bundle.sha256,
        "org.riley.reconstructed-runtime-assembly.recipe-normalized-instructions-sha256": assembly_recipe.EXPECTED_NORMALIZED_INSTRUCTION_SHA256,
    }


def _validate_image_inspect(raw: bytes, external: ExternalFacts, image_id: str) -> None:
    document = _strict_json(raw, "captured image inspect", canonical=False)
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        _fail("invalid-image-inspect", "captured image inspect must be a one-image array")
    row = document[0]
    if _image_id(row.get("Id"), "captured image inspect.Id") != image_id:
        _fail("image-id-mismatch", "captured image inspect differs from iidfile/runtime OCI image ID")
    if row.get("Os") != PLATFORM["os"] or row.get("Architecture") != PLATFORM["architecture"]:
        _fail("image-platform-mismatch", "captured image inspect must describe linux/amd64")
    config = row.get("Config")
    if not isinstance(config, dict):
        _fail("invalid-image-inspect", "captured image inspect.Config must be an object")
    labels = config.get("Labels")
    if not isinstance(labels, dict) or any(type(key) is not str or type(value) is not str for key, value in labels.items()):
        _fail("invalid-image-inspect", "captured image inspect.Config.Labels must be a string map")
    for key, expected in _image_labels(external).items():
        if labels.get(key) != expected:
            _fail("image-label-mismatch", f"captured image label {key!r} differs from the closed recipe inputs")
    _validate_expected_runtime_config(
        config,
        "captured image inspect.Config",
        invalid_code="invalid-image-inspect",
        mismatch_code="image-config-mismatch",
        environment_mismatch_code="image-environment-mismatch",
    )


def _validate_container_inspect(raw: bytes, image_id: str) -> str:
    document = _strict_json(raw, "captured container inspect", canonical=False)
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        _fail("invalid-container-inspect", "captured container inspect must be a one-container array")
    row = document[0]
    container_id = _container_id(row.get("Id"), "captured container inspect.Id")
    if _image_id(row.get("Image"), "captured container inspect.Image") != image_id:
        _fail("container-image-mismatch", "captured container was not created from the captured image ID")
    state = row.get("State")
    if not isinstance(state, dict):
        _fail("invalid-container-inspect", "captured container inspect.State must be an object")
    expected_state = {
        "Status": "created",
        "Running": False,
        "Paused": False,
        "Restarting": False,
        "OOMKilled": False,
        "Dead": False,
        "Pid": 0,
        "ExitCode": 0,
        "Error": "",
        "StartedAt": "0001-01-01T00:00:00Z",
        "FinishedAt": "0001-01-01T00:00:00Z",
    }
    for name, expected in expected_state.items():
        if state.get(name) != expected:
            _fail("container-started-or-invalid", f"captured container state {name!r} is not the never-started value")
    if row.get("Mounts") != []:
        _fail("container-mount-mismatch", "captured container must not have mounts")
    host = row.get("HostConfig")
    if not isinstance(host, dict) or host.get("NetworkMode") != "none" or host.get("Privileged") is not False:
        _fail("container-host-config-mismatch", "captured container must be unprivileged and network-disabled")
    for field in CONTAINER_EMPTY_HOST_FIELDS:
        if not _empty_container_option(host.get(field)):
            _fail("container-host-config-mismatch", f"captured container HostConfig.{field} must be empty")
    for field, safe_values in CONTAINER_SAFE_NAMESPACE_MODES.items():
        if host.get(field) not in safe_values:
            _fail(
                "container-host-config-mismatch",
                f"captured container HostConfig.{field} must retain a private namespace mode",
            )
    config = row.get("Config")
    if not isinstance(config, dict):
        _fail("invalid-container-inspect", "captured container inspect.Config must be an object")
    _validate_expected_runtime_config(
        config,
        "captured container inspect.Config",
        invalid_code="invalid-container-inspect",
        mismatch_code="container-config-mismatch",
        environment_mismatch_code="container-environment-mismatch",
    )
    return container_id


def _validate_runtime_tree(
    stream: BinaryIO,
    member: UstarMember,
    external: ExternalFacts,
) -> dict[str, Any]:
    expected_tree = _bundle_runtime_tree(external.repro_root_fd, external.bundle)
    members = _scan_ustar(
        stream,
        start=member.offset_data,
        byte_length=member.byte_length,
        label="captured /opt/riley filesystem",
        maximum_bytes=MAX_RUNTIME_TREE_ARCHIVE_BYTES,
        maximum_members=MAX_RUNTIME_TREE_MEMBERS,
        # Keep parsing bounded independently of the expected tree so a hostile
        # extra leaf reaches the closed-inventory check below instead of being
        # mistaken for an arithmetic overflow of otherwise valid evidence.
        maximum_total_member_bytes=MAX_RUNTIME_TREE_ARCHIVE_BYTES - MIN_TAR_END_BYTES,
        allow_directories=True,
    )
    names = tuple(item.name for item in members)
    expected_names = tuple(sorted(expected_tree))
    if names != expected_names:
        _fail("runtime-tree-inventory", "captured /opt/riley filesystem differs from the verified bundle tree")
    actual_tree: dict[str, TreeEntry] = {}
    for item in members:
        expected = expected_tree[item.name]
        if item.kind != expected.kind or item.mode != expected.mode or item.byte_length != expected.byte_length:
            _fail("runtime-tree-metadata-mismatch", f"captured runtime member {item.name!r} differs from the verified bundle tree")
        if item.uid != 65532 or item.gid != 65532:
            _fail("runtime-tree-owner-mismatch", f"captured runtime member {item.name!r} is not owned by the final non-root image user")
        if item.mode & 0o6000:
            _fail("runtime-tree-setid", f"captured runtime member {item.name!r} has set-ID bits")
        digest = None if item.kind == "directory" else _hash_member(stream, item, f"captured runtime member {item.name}")
        if digest != expected.sha256:
            _fail("runtime-tree-content-mismatch", f"captured runtime member {item.name!r} differs from the verified bundle tree")
        actual_tree[item.name] = TreeEntry(item.name, item.kind, item.mode, digest, item.byte_length)
    return _tree_summary(actual_tree)


def _validate_completion(
    raw: bytes,
    *,
    external: ExternalFacts,
    image_id: str,
    container_id: str,
    member_digests: Mapping[str, str],
    members: Mapping[str, UstarMember],
) -> None:
    row = _exact(
        _strict_json(raw, "captured completion record", canonical=True),
        {"schema_version", "reconstruction_id", "image_id", "container_id", "container_state", "container_started", "members"},
        "captured completion record",
    )
    if (
        row["schema_version"] != CAPTURE_COMPLETION_VERSION
        or row["reconstruction_id"] != external.reconstruction_id
        or row["image_id"] != image_id
        or row["container_id"] != container_id
        or row["container_state"] != "created"
        or row["container_started"] is not False
    ):
        _fail("completion-binding-mismatch", "captured completion record differs from raw derived facts")
    recorded = row["members"]
    if not isinstance(recorded, dict) or set(recorded) != set(CAPTURE_COMPLETION_MEMBER_NAMES):
        _fail("completion-member-mismatch", "captured completion record has the wrong member inventory")
    for name in CAPTURE_COMPLETION_MEMBER_NAMES:
        expected = {"sha256": member_digests[name], "byte_length": members[name].byte_length}
        if recorded[name] != expected:
            _fail("completion-member-mismatch", f"captured completion record differs for {name!r}")


@dataclass(frozen=True)
class CaptureFacts:
    members: dict[str, dict[str, Any]]
    context: dict[str, Any]
    image_id: str
    container_id: str
    runtime_tree: dict[str, Any]


def _parse_capture_archive(stream: BinaryIO, external: ExternalFacts) -> CaptureFacts:
    outer = _scan_ustar(
        stream,
        start=0,
        byte_length=_stream_length(stream, "runtime assembly capture archive"),
        label="runtime assembly capture archive",
        maximum_bytes=MAX_CAPTURE_ARCHIVE_BYTES,
        maximum_members=MAX_CAPTURE_MEMBERS,
        maximum_total_member_bytes=MAX_CAPTURE_ARCHIVE_BYTES - MIN_TAR_END_BYTES,
        allow_directories=False,
    )
    members = _member_map(outer, CAPTURE_MEMBER_NAMES, "runtime assembly capture archive")
    for name, member in members.items():
        _require_canonical_capture_metadata(member, f"runtime assembly capture member {name}")
        if member.byte_length > _capture_member_bound(name):
            _fail("capture-member-size", f"runtime assembly capture member {name!r} exceeds its byte bound")
        if name != "build.log" and member.byte_length < 1:
            _fail("empty-capture-member", f"runtime assembly capture member {name!r} must be nonempty")
    member_digests = {name: _hash_member(stream, member, f"runtime assembly capture {name}") for name, member in members.items()}
    checksums = _parse_checksums(
        _read_member(stream, members["SHA256SUMS"], "runtime assembly SHA256SUMS", maximum_bytes=MAX_RECEIPT_BYTES),
        CAPTURE_MEMBER_NAMES[1:],
    )
    for name in CAPTURE_MEMBER_NAMES[1:]:
        if checksums[name] != member_digests[name]:
            _fail("capture-checksum-mismatch", f"runtime assembly SHA256SUMS differs for {name!r}")

    context = _validate_context(stream, members["context.tar"], external)
    _validate_invocation(
        _read_member(stream, members["capture-invocation.json"], "captured build invocation", maximum_bytes=MAX_RECEIPT_BYTES),
        external,
        members["context.tar"],
        member_digests["context.tar"],
    )
    iid_raw = _read_member(stream, members["build.iid"], "captured build iidfile", maximum_bytes=80)
    # Docker's --iidfile contains the immutable image ID bytes themselves,
    # without a line terminator. Preserve that raw Docker format rather than
    # inventing a canonical newline that a real host runner would not emit.
    expected_iid = external.image_id.encode("ascii")
    if iid_raw != expected_iid:
        _fail("iidfile-image-mismatch", "captured iidfile differs from the runtime OCI image ID")
    if (
        members["image-inspect.json"].byte_length != external.oci_image_inspect.byte_length
        or member_digests["image-inspect.json"] != external.oci_image_inspect.sha256
    ):
        _fail("image-inspect-closure-mismatch", "captured image inspect differs from the verified runtime OCI input")
    if (
        members["oci-image-layout.tar"].byte_length != external.oci_archive.byte_length
        or member_digests["oci-image-layout.tar"] != external.oci_archive.sha256
    ):
        _fail("oci-archive-closure-mismatch", "captured OCI archive differs from the verified runtime OCI input")
    _validate_image_inspect(
        _read_member(stream, members["image-inspect.json"], "captured image inspect", maximum_bytes=MAX_RECEIPT_BYTES),
        external,
        external.image_id,
    )
    _validate_oci_export_invocation(
        _read_member(stream, members["oci-export-invocation.json"], "captured OCI export invocation", maximum_bytes=MAX_RECEIPT_BYTES),
        external.image_id,
    )
    container_id = _validate_container_inspect(
        _read_member(stream, members["container-inspect.json"], "captured container inspect", maximum_bytes=MAX_RECEIPT_BYTES),
        external.image_id,
    )
    runtime_tree = _validate_runtime_tree(stream, members["container-opt-riley.tar"], external)
    _validate_completion(
        _read_member(stream, members["capture-completion.json"], "captured completion record", maximum_bytes=MAX_RECEIPT_BYTES),
        external=external,
        image_id=external.image_id,
        container_id=container_id,
        member_digests=member_digests,
        members=members,
    )
    return CaptureFacts(
        members={
            name: {"sha256": member_digests[name], "byte_length": members[name].byte_length}
            for name in CAPTURE_MEMBER_NAMES
        },
        context=context,
        image_id=external.image_id,
        container_id=container_id,
        runtime_tree=runtime_tree,
    )


def _stream_length(stream: BinaryIO, label: str) -> int:
    try:
        stream.seek(0, os.SEEK_END)
        byte_length = stream.tell()
        stream.seek(0, os.SEEK_SET)
    except OSError as error:
        _fail("unreadable-capture", f"cannot inspect {label}: {error}")
    return byte_length


def _capture_archive_descriptor_from_receipt(receipt: Mapping[str, Any]) -> common.EvidenceDescriptor:
    capture = _exact(
        receipt.get("capture"),
        {"archive", "members", "context", "image_id", "container_id", "runtime_tree"},
        "runtime assembly capture receipt.capture",
    )
    return _descriptor(
        capture["archive"],
        "runtime assembly capture archive descriptor",
        expected_path=f"{RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY}/{RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}",
        maximum_bytes=MAX_CAPTURE_ARCHIVE_BYTES,
    )


def _parse_receipt(receipt: Any, reconstruction_id: str) -> dict[str, Any]:
    row = _exact(
        receipt,
        {
            "schema_version",
            "status",
            "qualification_status",
            "authority",
            "capture_scope",
            "baseline_id",
            "reconstruction_id",
            "platform",
            "source_inputs",
            "reproducibility_inputs",
            "runtime_oci_inputs",
            "assembly_recipe",
            "capture",
            "binding_status",
            "not_established",
        },
        "runtime assembly capture receipt",
    )
    if (
        row["schema_version"] != RUNTIME_ASSEMBLY_CAPTURE_VERSION
        or row["status"] != STATUS
        or row["qualification_status"] != QUALIFICATION_STATUS
        or row["authority"] != AUTHORITY
        or row["capture_scope"] != CAPTURE_SCOPE
        or row["reconstruction_id"] != reconstruction_id
        or row["platform"] != PLATFORM
    ):
        _fail("invalid-capture-receipt", "runtime assembly capture receipt has an unexpected fixed field")
    _capture_archive_descriptor_from_receipt(row)
    return row


def _receipt(
    *,
    external: ExternalFacts,
    archive: common.EvidenceDescriptor,
    facts: CaptureFacts,
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_ASSEMBLY_CAPTURE_VERSION,
        "status": STATUS,
        "qualification_status": QUALIFICATION_STATUS,
        "authority": AUTHORITY,
        "capture_scope": CAPTURE_SCOPE,
        "baseline_id": "reconstructed-riley-0.1.0-rc2",
        "reconstruction_id": external.reconstruction_id,
        "platform": dict(PLATFORM),
        "source_inputs": external.source_inputs,
        "reproducibility_inputs": external.repro_inputs,
        "runtime_oci_inputs": external.runtime_oci_inputs,
        "assembly_recipe": {
            "normalized_instructions_sha256": assembly_recipe.EXPECTED_NORMALIZED_INSTRUCTION_SHA256,
            "pinned_runtime": assembly_recipe.PINNED_RUNTIME,
        },
        "capture": {
            "archive": archive.as_json(),
            "members": facts.members,
            "context": facts.context,
            "image_id": facts.image_id,
            "container_id": facts.container_id,
            "runtime_tree": facts.runtime_tree,
        },
        "binding_status": {
            "source_inputs": "replayed-reviewed-rc2-source-inputs-v1",
            "reproducibility_inputs": "replayed-pr16-release-build-reproducibility-v1",
            "assembly_recipe": "validated-static-source-free-recipe-v1",
            "canonical_context": "validated",
            "runtime_build_invocation": "raw-record-structurally-validated-v1",
            "runtime_oci_content": "validated-via-runtime-oci-inputs-v1",
            "created_container_filesystem": "raw-record-structurally-validated-never-started-v1",
        },
        "not_established": {
            "runtime_build_execution": "not-established",
            "container_filesystem_capture_provenance": "not-established",
            "bundle_to_runtime_image": "not-established",
            "source_to_runtime_image": "not-established",
            "runtime_capture_independence": "not-established",
            "a_b_runtime_image_equality": "not-established",
            "rollback": "not-established",
            "freeze": "not-established",
            "qualification": "not-run",
            "service_execution": "not-run",
            "gpu_execution": "not-run",
            "historical_distribution": "not-attested",
        },
    }


def _consume_capture(
    capture_fd: int,
    archive: common.EvidenceDescriptor,
    external: ExternalFacts,
) -> CaptureFacts:
    held = _common(
        lambda: common.rebase_descriptor_to_held_leaf(
            archive,
            expected_root_relative_path=f"{RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY}/{RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}",
            leaf_name=RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE,
            label="runtime assembly capture archive",
        )
    )
    return _common(
        lambda: common.consume_private_snapshot_descriptor_file(
            capture_fd,
            held,
            "runtime assembly capture archive",
            lambda stream: _parse_capture_archive(stream, external),
            maximum_bytes=MAX_CAPTURE_ARCHIVE_BYTES,
        )
    )


def verify_reconstructed_runtime_assembly_capture_fd(
    root_fd: int,
    *,
    source_input_root_fd: int,
    repro_build_input_root_fd: int,
    runtime_oci_input_root_fd: int,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    reconstruction_id: str,
) -> dict[str, Any]:
    """Replay a published capture root and all caller-held external roots."""

    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "runtime assembly capture root"))
    _require_distinct_root_fds(
        {
            "runtime assembly capture root": root_fd,
            "source inputs root": source_input_root_fd,
            "reproducibility inputs root": repro_build_input_root_fd,
            "runtime OCI inputs root": runtime_oci_input_root_fd,
        }
    )
    try:
        entries = set(os.listdir(root_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot list runtime assembly capture root: {error}")
    if entries != {RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY, RUNTIME_ASSEMBLY_CAPTURE_NAME}:
        _fail("unexpected-evidence-entry", "runtime assembly capture root has an unexpected inventory")
    receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            RUNTIME_ASSEMBLY_CAPTURE_NAME,
            "runtime assembly capture receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    parsed = _parse_receipt(receipt, reconstruction_id)
    capture_fd = _common(
        lambda: common.open_private_child_directory(
            root_fd,
            RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY,
            "runtime assembly capture archive directory",
        )
    )
    try:
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                capture_fd,
                RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY,
                "held runtime assembly capture archive directory",
            )
        )
        if set(os.listdir(capture_fd)) != {RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}:
            _fail("unexpected-evidence-entry", "runtime assembly capture archive directory has an unexpected inventory")
        external = _replay_external_inputs(
            source_root_fd=source_input_root_fd,
            repro_root_fd=repro_build_input_root_fd,
            runtime_oci_root_fd=runtime_oci_input_root_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id=reconstruction_id,
        )
        archive = _capture_archive_descriptor_from_receipt(parsed)
        facts = _consume_capture(capture_fd, archive, external)
        expected = _receipt(external=external, archive=archive, facts=facts)
        if parsed != expected:
            _fail("capture-replay-mismatch", "runtime assembly capture receipt differs from freshly replayed evidence")
        if set(os.listdir(capture_fd)) != {RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}:
            _fail("raced-input", "runtime assembly capture archive directory changed during replay")
    finally:
        os.close(capture_fd)
    terminal = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            RUNTIME_ASSEMBLY_CAPTURE_NAME,
            "runtime assembly capture receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    if terminal != receipt or set(os.listdir(root_fd)) != {RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY, RUNTIME_ASSEMBLY_CAPTURE_NAME}:
        _fail("raced-input", "runtime assembly capture root changed during replay")
    return parsed


def _normalize_inputs(
    evidence_root: Path,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root: Path,
    assembly_capture: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    paths = {
        "runtime assembly capture root": _normalized_absolute_path(evidence_root, "--evidence-root"),
        "source inputs root": _normalized_absolute_path(source_input_root, "--source-input-root"),
        "reproducibility inputs root": _normalized_absolute_path(repro_build_input_root, "--repro-build-input-root"),
        "runtime OCI inputs root": _normalized_absolute_path(runtime_oci_input_root, "--runtime-oci-input-root"),
        "raw assembly capture": _normalized_absolute_path(assembly_capture, "--assembly-capture"),
    }
    _require_disjoint_paths(paths)
    return (
        paths["runtime assembly capture root"],
        paths["source inputs root"],
        paths["reproducibility inputs root"],
        paths["runtime OCI inputs root"],
        paths["raw assembly capture"],
    )


def _normalize_verify_inputs(
    evidence_root: Path,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root: Path,
) -> tuple[Path, Path, Path, Path]:
    paths = {
        "runtime assembly capture root": _normalized_absolute_path(evidence_root, "--evidence-root"),
        "source inputs root": _normalized_absolute_path(source_input_root, "--source-input-root"),
        "reproducibility inputs root": _normalized_absolute_path(repro_build_input_root, "--repro-build-input-root"),
        "runtime OCI inputs root": _normalized_absolute_path(runtime_oci_input_root, "--runtime-oci-input-root"),
    }
    _require_disjoint_paths(paths)
    return (
        paths["runtime assembly capture root"],
        paths["source inputs root"],
        paths["reproducibility inputs root"],
        paths["runtime OCI inputs root"],
    )


def verify_reconstructed_runtime_assembly_capture(
    evidence_root: Path,
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root: Path,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    reconstruction_id: str,
) -> dict[str, Any]:
    """Open verified roots safely and replay one existing capture receipt."""

    evidence_root, source_input_root, repro_build_input_root, runtime_oci_input_root = _normalize_verify_inputs(
        evidence_root,
        source_input_root,
        repro_build_input_root,
        runtime_oci_input_root,
    )
    root_fd = _common(lambda: common.open_private_evidence_directory(evidence_root, "runtime assembly capture root"))
    external: dict[str, int] = {}
    try:
        external = _open_external_roots(
            source_input_root=source_input_root,
            repro_build_input_root=repro_build_input_root,
            runtime_oci_input_root=runtime_oci_input_root,
        )
        return verify_reconstructed_runtime_assembly_capture_fd(
            root_fd,
            source_input_root_fd=external["source inputs root"],
            repro_build_input_root_fd=external["reproducibility inputs root"],
            runtime_oci_input_root_fd=external["runtime OCI inputs root"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id=reconstruction_id,
        )
    finally:
        _close_fds(external)
        os.close(root_fd)


def prepare_reconstructed_runtime_assembly_capture(
    evidence_root: Path,
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root: Path,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    reconstruction_id: str,
    assembly_capture: Path,
) -> dict[str, Any]:
    """Snapshot and self-replay one raw per-arm runtime assembly capture."""

    (
        evidence_root,
        source_input_root,
        repro_build_input_root,
        runtime_oci_input_root,
        assembly_capture,
    ) = _normalize_inputs(
        evidence_root,
        source_input_root,
        repro_build_input_root,
        runtime_oci_input_root,
        assembly_capture,
    )
    external_roots = _open_external_roots(
        source_input_root=source_input_root,
        repro_build_input_root=repro_build_input_root,
        runtime_oci_input_root=runtime_oci_input_root,
    )
    root_fd: int | None = None
    capture_fd: int | None = None
    try:
        external = _replay_external_inputs(
            source_root_fd=external_roots["source inputs root"],
            repro_root_fd=external_roots["reproducibility inputs root"],
            runtime_oci_root_fd=external_roots["runtime OCI inputs root"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id=reconstruction_id,
        )
        root_fd = _common(lambda: common.create_private_evidence_directory(evidence_root, "runtime assembly capture root"))
        _require_distinct_root_fds({"runtime assembly capture root": root_fd, **external_roots})
        capture_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY,
                "runtime assembly capture archive directory",
            )
        )
        snapshot = _common(
            lambda: common.snapshot_absolute_regular_create_only(
                assembly_capture,
                capture_fd,
                RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE,
                "raw runtime assembly capture",
                maximum_bytes=MAX_CAPTURE_ARCHIVE_BYTES,
                minimum_bytes=MIN_TAR_END_BYTES,
            )
        )
        archive = snapshot.descriptor(
            f"{RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY}/{RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}",
            "raw runtime assembly capture",
        )
        facts = _consume_capture(capture_fd, archive, external)
        receipt = _receipt(external=external, archive=archive, facts=facts)
        _common(
            lambda: common.write_create_only_json(
                root_fd,
                RUNTIME_ASSEMBLY_CAPTURE_NAME,
                receipt,
                "runtime assembly capture receipt",
            )
        )
        replayed = verify_reconstructed_runtime_assembly_capture_fd(
            root_fd,
            source_input_root_fd=external_roots["source inputs root"],
            repro_build_input_root_fd=external_roots["reproducibility inputs root"],
            runtime_oci_input_root_fd=external_roots["runtime OCI inputs root"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id=reconstruction_id,
        )
        if replayed != receipt:
            _fail("prepublication-replay-drift", "held runtime assembly capture replay differs from the draft receipt")
        return receipt
    finally:
        if capture_fd is not None:
            os.close(capture_fd)
        if root_fd is not None:
            os.close(root_fd)
        _close_fds(external_roots)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-input-root", type=Path, required=True)
    parser.add_argument("--repro-build-input-root", type=Path, required=True)
    parser.add_argument("--runtime-oci-input-root", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--expected-build-image-id", required=True)
    parser.add_argument("--reconstruction-id", choices=("a", "b"), required=True)
    parser.add_argument("--assembly-capture", type=Path)
    parser.add_argument("--verify", action="store_true", help="replay an existing evidence root instead of creating one")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        arguments = {
            "source_input_root": args.source_input_root,
            "repro_build_input_root": args.repro_build_input_root,
            "runtime_oci_input_root": args.runtime_oci_input_root,
            "expected_source_archive_sha256": args.expected_source_archive_sha256,
            "expected_build_image_id": args.expected_build_image_id,
            "reconstruction_id": args.reconstruction_id,
        }
        if args.verify:
            receipt = verify_reconstructed_runtime_assembly_capture(args.evidence_root, **arguments)
        else:
            if args.assembly_capture is None:
                _fail("missing-assembly-capture", "--assembly-capture is required unless --verify is used")
            receipt = prepare_reconstructed_runtime_assembly_capture(
                args.evidence_root,
                assembly_capture=args.assembly_capture,
                **arguments,
            )
    except RuntimeAssemblyCaptureError as error:
        payload = {
            "schema_version": RUNTIME_ASSEMBLY_CAPTURE_VERSION,
            "status": "failed",
            "qualification_status": QUALIFICATION_STATUS,
            "reason_codes": [getattr(error, "reason_code", "invalid-runtime-assembly-capture")],
            "error": str(error),
        }
        sys.stderr.buffer.write(common.canonical_json_bytes(payload) + b"\n")
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
