#!/usr/bin/env python3
"""Bind one normalized runtime image export to its OCI and capture closures.

This source-only bridge replays already-published private evidence roots.  It
binds one image-export normalization root, one matching runtime-OCI-inputs
root, and one matching runtime-assembly-capture root for the requested arm.
The assembly-capture verifier also requires the reviewed source and
reproducibility-input roots, so this bridge replays those inputs rather than
trusting a nested capture receipt.

It proves byte-content equality only.  It does not invoke Docker, build an
image, start a container or service, access a GPU, establish same-invocation
provenance, or run a qualification gate.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

# This source-only verifier must not dirty the checkout when invoked directly.
sys.dont_write_bytecode = True

import prepare_reconstructed_runtime_assembly_capture_v1 as assembly_capture
import prepare_reconstructed_runtime_image_export_oci_normalization_v1 as image_export
import prepare_reconstructed_runtime_oci_inputs_v1 as runtime_oci
import provenance_v2_common as common


BRIDGE_VERSION = "riley.reconstructed-runtime-image-export-assembly-content-bridge.v1"
BRIDGE_RECEIPT_NAME = "reconstructed-runtime-image-export-assembly-content-bridge.json"
BRIDGE_AUTHORITY = "cross-root-runtime-image-export-assembly-content-bridge-only"
BRIDGE_STATUS = "bound"
QUALIFICATION_STATUS = "not-run"
MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES
PLATFORM = {"os": "linux", "architecture": "amd64"}
NOT_ESTABLISHED = "not-established"
BRIDGE_CAPTURE_SCOPE = "single-arm-normalized-runtime-oci-and-assembly-content-binding"

BINDING_STATUS = {
    "image_export_normalization": "replayed-runtime-image-export-oci-normalization-v1",
    "runtime_oci_inputs": "replayed-runtime-oci-inputs-v1",
    "assembly_capture": "replayed-runtime-assembly-capture-v1",
    "normalized_oci_to_runtime_oci_content": "validated-sha256-and-byte-length-equality",
    "runtime_oci_to_assembly_capture_content": "validated-sha256-and-byte-length-equality",
}
NOT_ESTABLISHED_STATUS = {
    "image_export_and_assembly_capture_same_invocation": NOT_ESTABLISHED,
    "docker_image_export_execution": NOT_ESTABLISHED,
    "runtime_build_execution": NOT_ESTABLISHED,
    "container_filesystem_capture_provenance": NOT_ESTABLISHED,
    "bundle_to_runtime_image": NOT_ESTABLISHED,
    "source_to_runtime_image": NOT_ESTABLISHED,
    "runtime_capture_independence": NOT_ESTABLISHED,
    "a_b_runtime_image_equality": NOT_ESTABLISHED,
    "rollback": NOT_ESTABLISHED,
    "freeze": NOT_ESTABLISHED,
    "qualification": QUALIFICATION_STATUS,
    "service_execution": QUALIFICATION_STATUS,
    "gpu_execution": QUALIFICATION_STATUS,
    "historical_distribution": "not-attested",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_T = TypeVar("_T")


class RuntimeImageExportAssemblyContentBridgeError(common.ProvenanceV2Error):
    """A bridge receipt or one of its held input roots is unsafe."""


def _fail(code: str, message: str) -> NoReturn:
    error = RuntimeImageExportAssemblyContentBridgeError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _bridge(call: Callable[[], _T]) -> _T:
    """Translate imported verifier failures without dropping their reason code."""

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


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_disjoint_paths(paths: Mapping[str, Path]) -> None:
    rows = tuple(paths.items())
    for index, (left_label, left) in enumerate(rows):
        for right_label, right in rows[index + 1 :]:
            if _paths_overlap(left, right):
                _fail(
                    "bridge-root-overlap",
                    f"{left_label} and {right_label} must be disjoint normalized paths",
                )


def _root_identity(directory_fd: int, label: str) -> tuple[int, int]:
    try:
        metadata = os.fstat(directory_fd)
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot inspect {label}: {error}")
    return metadata.st_dev, metadata.st_ino


def _require_distinct_root_fds(roots: Mapping[str, int]) -> None:
    seen: dict[tuple[int, int], str] = {}
    for label, directory_fd in roots.items():
        identity = _root_identity(directory_fd, label)
        previous = seen.get(identity)
        if previous is not None:
            _fail("input-root-alias", f"{label} aliases the already-held {previous}")
        seen[identity] = label


def _close_fds(fds: Mapping[str, int]) -> None:
    for descriptor in reversed(tuple(fds.values())):
        os.close(descriptor)


def _open_external_roots(
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    image_export_normalization_root: Path,
    runtime_oci_input_root: Path,
    assembly_capture_root: Path,
) -> dict[str, int]:
    opened: dict[str, int] = {}
    try:
        for label, path in (
            ("source inputs root", source_input_root),
            ("reproducibility inputs root", repro_build_input_root),
            ("image export normalization root", image_export_normalization_root),
            ("runtime OCI inputs root", runtime_oci_input_root),
            ("runtime assembly capture root", assembly_capture_root),
        ):
            opened[label] = _bridge(lambda path=path, label=label: common.open_private_evidence_directory(path, label))
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


def _descriptor(value: Any, label: str, *, expected_path: str | None = None) -> common.EvidenceDescriptor:
    descriptor = _bridge(lambda: common.parse_descriptor(value, label))
    if expected_path is not None and descriptor.path != expected_path:
        _fail("bridge-receipt-path-mismatch", f"{label} must have fixed path {expected_path!r}")
    return descriptor


def _member_fingerprint(value: Any, label: str) -> dict[str, Any]:
    row = _exact(value, {"sha256", "byte_length"}, label)
    byte_length = row["byte_length"]
    if type(byte_length) is not int or byte_length < 1:
        _fail("invalid-member-fingerprint", f"{label}.byte_length must be a positive integer")
    return {
        "sha256": _sha256(row["sha256"], f"{label}.sha256"),
        "byte_length": byte_length,
    }


def _same_content(
    left: common.EvidenceDescriptor | Mapping[str, Any],
    right: common.EvidenceDescriptor | Mapping[str, Any],
    label: str,
) -> None:
    left_row = left.as_json() if isinstance(left, common.EvidenceDescriptor) else dict(left)
    right_row = right.as_json() if isinstance(right, common.EvidenceDescriptor) else dict(right)
    left_sha = _sha256(left_row.get("sha256"), f"{label} left SHA-256")
    right_sha = _sha256(right_row.get("sha256"), f"{label} right SHA-256")
    left_length = left_row.get("byte_length")
    right_length = right_row.get("byte_length")
    if (
        type(left_length) is not int
        or type(right_length) is not int
        or left_length < 1
        or right_length < 1
        or left_sha != right_sha
        or left_length != right_length
    ):
        _fail("cross-root-content-mismatch", f"{label} differs by SHA-256 or byte length")


def _read_receipt(
    root_fd: int,
    name: str,
    label: str,
) -> tuple[dict[str, Any], common.EvidenceDescriptor]:
    receipt = _bridge(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            name,
            label,
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    descriptor = _bridge(
        lambda: common.describe_regular_relative(
            root_fd,
            name,
            label,
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    return receipt, descriptor


def _require_terminal_receipt(
    root_fd: int,
    name: str,
    label: str,
    expected_receipt: Mapping[str, Any],
    expected_descriptor: common.EvidenceDescriptor,
) -> None:
    terminal, descriptor = _read_receipt(root_fd, name, label)
    if terminal != expected_receipt or descriptor.as_json() != expected_descriptor.as_json():
        _fail("raced-input", f"{label} changed during bridge replay")


def _normalization_projection(
    receipt: Mapping[str, Any],
    receipt_descriptor: common.EvidenceDescriptor,
) -> dict[str, Any]:
    row = _exact(
        dict(receipt),
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
        "image export normalization receipt",
    )
    if (
        type(row["reconstruction_id"]) is not str
        or row["reconstruction_id"] not in {"a", "b"}
        or row["platform"] != PLATFORM
    ):
        _fail("invalid-normalization-receipt", "image export normalization receipt has invalid arm or platform")
    _image_id(row["image_id"], "image export normalization receipt.image_id")
    if type(row["source_layout"]) is not str or row["source_layout"] not in image_export.SOURCE_LAYOUTS:
        _fail("invalid-normalization-receipt", "image export normalization receipt source layout is unknown")
    expected_paths = {
        "receipt": image_export.NORMALIZATION_NAME,
        "image_inspect": f"{image_export.RAW_DIRECTORY_NAME}/{image_export.IMAGE_INSPECT_NAME}",
        "image_export_archive": f"{image_export.RAW_DIRECTORY_NAME}/{image_export.IMAGE_EXPORT_ARCHIVE_NAME}",
        "oci_archive": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_ARCHIVE_NAME}",
        "layout": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_LAYOUT_NAME}",
        "index": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_INDEX_NAME}",
        "manifest": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_MANIFEST_NAME}",
        "config": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_CONFIG_NAME}",
    }
    projection = {
        "receipt": _descriptor(receipt_descriptor.as_json(), "image export normalization receipt descriptor", expected_path=expected_paths["receipt"]).as_json(),
        "reconstruction_id": row["reconstruction_id"],
        "image_id": row["image_id"],
        "source_layout": row["source_layout"],
    }
    for field in ("image_inspect", "image_export_archive", "oci_archive", "layout", "index", "manifest", "config"):
        projection[field] = _descriptor(
            row[field],
            f"image export normalization {field}",
            expected_path=expected_paths[field],
        ).as_json()
    return projection


def _runtime_oci_projection(
    receipt: Mapping[str, Any],
    receipt_descriptor: common.EvidenceDescriptor,
) -> dict[str, Any]:
    row = _exact(
        dict(receipt),
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
        type(row["reconstruction_id"]) is not str
        or row["reconstruction_id"] not in {"a", "b"}
        or row["platform"] != PLATFORM
    ):
        _fail("invalid-runtime-oci-receipt", "runtime OCI receipt has invalid arm or platform")
    _image_id(row["image_id"], "runtime OCI inputs receipt.image_id")
    expected_paths = {
        "receipt": runtime_oci.RUNTIME_OCI_INPUTS_NAME,
        "image_inspect": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.IMAGE_INSPECT_NAME}",
        "archive": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_ARCHIVE_NAME}",
        "layout": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_LAYOUT_NAME}",
        "index": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_INDEX_NAME}",
        "manifest": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_MANIFEST_NAME}",
        "config": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_CONFIG_NAME}",
    }
    projection = {
        "receipt": _descriptor(receipt_descriptor.as_json(), "runtime OCI receipt descriptor", expected_path=expected_paths["receipt"]).as_json(),
        "reconstruction_id": row["reconstruction_id"],
        "image_id": row["image_id"],
    }
    for field in ("image_inspect", "archive", "layout", "index", "manifest", "config"):
        projection[field] = _descriptor(
            row[field],
            f"runtime OCI {field}",
            expected_path=expected_paths[field],
        ).as_json()
    return projection


def _assembly_capture_projection(
    receipt: Mapping[str, Any],
    receipt_descriptor: common.EvidenceDescriptor,
) -> dict[str, Any]:
    row = _exact(
        dict(receipt),
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
        type(row["reconstruction_id"]) is not str
        or row["reconstruction_id"] not in {"a", "b"}
        or row["platform"] != PLATFORM
    ):
        _fail("invalid-capture-receipt", "runtime assembly capture receipt has invalid arm or platform")
    capture = _exact(
        row["capture"],
        {"archive", "members", "context", "image_id", "container_id", "runtime_tree"},
        "runtime assembly capture receipt.capture",
    )
    members = capture["members"]
    if not isinstance(members, dict) or set(members) != set(assembly_capture.CAPTURE_MEMBER_NAMES):
        _fail("invalid-capture-receipt", "runtime assembly capture member inventory is not exact")
    image_member = _member_fingerprint(members["image-inspect.json"], "captured image inspect member")
    archive_member = _member_fingerprint(members["oci-image-layout.tar"], "captured OCI archive member")
    return {
        "receipt": _descriptor(
            receipt_descriptor.as_json(),
            "runtime assembly capture receipt descriptor",
            expected_path=assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_NAME,
        ).as_json(),
        "reconstruction_id": row["reconstruction_id"],
        "image_id": _image_id(capture["image_id"], "runtime assembly capture image_id"),
        "capture_archive": _descriptor(
            capture["archive"],
            "runtime assembly capture archive",
            expected_path=(
                f"{assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY}/"
                f"{assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}"
            ),
        ).as_json(),
        "capture_members": {
            "image-inspect.json": image_member,
            "oci-image-layout.tar": archive_member,
        },
    }


def _replay_external_inputs(
    *,
    source_input_root_fd: int,
    repro_build_input_root_fd: int,
    image_export_normalization_root_fd: int,
    runtime_oci_input_root_fd: int,
    assembly_capture_root_fd: int,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    reconstruction_id: str,
) -> dict[str, Any]:
    if type(reconstruction_id) is not str or reconstruction_id not in {"a", "b"}:
        _fail("invalid-reconstruction-id", "--reconstruction-id must be exactly a or b")
    expected_source_archive_sha256 = _sha256(
        expected_source_archive_sha256,
        "--expected-source-archive-sha256",
    )
    expected_build_image_id = _image_id(expected_build_image_id, "--expected-build-image-id")
    _require_distinct_root_fds(
        {
            "source inputs root": source_input_root_fd,
            "reproducibility inputs root": repro_build_input_root_fd,
            "image export normalization root": image_export_normalization_root_fd,
            "runtime OCI inputs root": runtime_oci_input_root_fd,
            "runtime assembly capture root": assembly_capture_root_fd,
        }
    )

    normalization_verified = _bridge(
        lambda: image_export.verify_reconstructed_runtime_image_export_oci_normalization_fd(
            image_export_normalization_root_fd
        )
    )
    normalization_receipt, normalization_descriptor = _read_receipt(
        image_export_normalization_root_fd,
        image_export.NORMALIZATION_NAME,
        "image export normalization receipt",
    )
    if normalization_receipt != normalization_verified:
        _fail("raced-input", "image export normalization receipt changed during held-root replay")
    normalization = _normalization_projection(normalization_verified, normalization_descriptor)

    runtime_oci_verified = _bridge(
        lambda: runtime_oci.verify_reconstructed_runtime_oci_inputs_fd(runtime_oci_input_root_fd)
    )
    runtime_oci_receipt, runtime_oci_descriptor = _read_receipt(
        runtime_oci_input_root_fd,
        runtime_oci.RUNTIME_OCI_INPUTS_NAME,
        "runtime OCI inputs receipt",
    )
    if runtime_oci_receipt != runtime_oci_verified:
        _fail("raced-input", "runtime OCI receipt changed during held-root replay")
    oci = _runtime_oci_projection(runtime_oci_verified, runtime_oci_descriptor)
    if normalization["reconstruction_id"] != reconstruction_id or oci["reconstruction_id"] != reconstruction_id:
        _fail("reconstruction-id-mismatch", "normalized export and runtime OCI inputs must match --reconstruction-id")

    capture_verified = _bridge(
        lambda: assembly_capture.verify_reconstructed_runtime_assembly_capture_fd(
            assembly_capture_root_fd,
            source_input_root_fd=source_input_root_fd,
            repro_build_input_root_fd=repro_build_input_root_fd,
            runtime_oci_input_root_fd=runtime_oci_input_root_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id=reconstruction_id,
        )
    )
    capture_receipt, capture_descriptor = _read_receipt(
        assembly_capture_root_fd,
        assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_NAME,
        "runtime assembly capture receipt",
    )
    if capture_receipt != capture_verified:
        _fail("raced-input", "runtime assembly capture receipt changed during held-root replay")
    capture = _assembly_capture_projection(capture_verified, capture_descriptor)

    if capture["reconstruction_id"] != reconstruction_id:
        _fail("reconstruction-id-mismatch", "runtime assembly capture must match --reconstruction-id")
    image_id = _image_id(normalization["image_id"], "normalized image export image_id")
    if oci["image_id"] != image_id or capture["image_id"] != image_id:
        _fail("cross-root-image-id-mismatch", "normalized export, runtime OCI, and capture image IDs must match")

    for normalization_field, oci_field, label in (
        ("image_inspect", "image_inspect", "normalized export and runtime OCI image inspect"),
        ("oci_archive", "archive", "normalized export and runtime OCI OCI archive"),
        ("layout", "layout", "normalized export and runtime OCI layout"),
        ("index", "index", "normalized export and runtime OCI index"),
        ("manifest", "manifest", "normalized export and runtime OCI manifest"),
        ("config", "config", "normalized export and runtime OCI config"),
    ):
        _same_content(normalization[normalization_field], oci[oci_field], label)
    capture_members = capture["capture_members"]
    _same_content(oci["image_inspect"], capture_members["image-inspect.json"], "runtime OCI and captured image inspect")
    _same_content(oci["archive"], capture_members["oci-image-layout.tar"], "runtime OCI and captured OCI archive")
    if capture_verified["runtime_oci_inputs"] != oci:
        _fail("capture-runtime-oci-binding-mismatch", "capture receipt does not bind the held runtime OCI input receipt")

    _require_terminal_receipt(
        image_export_normalization_root_fd,
        image_export.NORMALIZATION_NAME,
        "image export normalization receipt",
        normalization_receipt,
        normalization_descriptor,
    )
    _require_terminal_receipt(
        runtime_oci_input_root_fd,
        runtime_oci.RUNTIME_OCI_INPUTS_NAME,
        "runtime OCI inputs receipt",
        runtime_oci_receipt,
        runtime_oci_descriptor,
    )
    _require_terminal_receipt(
        assembly_capture_root_fd,
        assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_NAME,
        "runtime assembly capture receipt",
        capture_receipt,
        capture_descriptor,
    )
    return {
        "schema_version": BRIDGE_VERSION,
        "status": BRIDGE_STATUS,
        "qualification_status": QUALIFICATION_STATUS,
        "authority": BRIDGE_AUTHORITY,
        "capture_scope": BRIDGE_CAPTURE_SCOPE,
        "reconstruction_id": reconstruction_id,
        "platform": dict(PLATFORM),
        "image_id": image_id,
        "assembly_replay_inputs": {
            "expected_source_archive_sha256": expected_source_archive_sha256,
            "expected_build_image_id": expected_build_image_id,
        },
        "image_export_normalization": normalization,
        "runtime_oci_inputs": oci,
        "assembly_capture": capture,
        "binding_status": dict(BINDING_STATUS),
        "not_established": dict(NOT_ESTABLISHED_STATUS),
    }


def _parse_bridge_receipt(value: Any) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "schema_version",
            "status",
            "qualification_status",
            "authority",
            "capture_scope",
            "reconstruction_id",
            "platform",
            "image_id",
            "assembly_replay_inputs",
            "image_export_normalization",
            "runtime_oci_inputs",
            "assembly_capture",
            "binding_status",
            "not_established",
        },
        "runtime image export assembly content bridge receipt",
    )
    if (
        row["schema_version"] != BRIDGE_VERSION
        or row["status"] != BRIDGE_STATUS
        or row["qualification_status"] != QUALIFICATION_STATUS
        or row["authority"] != BRIDGE_AUTHORITY
        or row["capture_scope"] != BRIDGE_CAPTURE_SCOPE
        or type(row["reconstruction_id"]) is not str
        or row["reconstruction_id"] not in {"a", "b"}
        or row["platform"] != PLATFORM
    ):
        _fail("invalid-bridge-receipt", "runtime image export assembly content bridge has invalid fixed fields")
    _image_id(row["image_id"], "bridge receipt.image_id")
    replay_inputs = _exact(
        row["assembly_replay_inputs"],
        {"expected_source_archive_sha256", "expected_build_image_id"},
        "bridge receipt assembly replay inputs",
    )
    _sha256(replay_inputs["expected_source_archive_sha256"], "bridge receipt.expected_source_archive_sha256")
    _image_id(replay_inputs["expected_build_image_id"], "bridge receipt.expected_build_image_id")

    normalization = _exact(
        row["image_export_normalization"],
        {"receipt", "reconstruction_id", "image_id", "source_layout", "image_inspect", "image_export_archive", "oci_archive", "layout", "index", "manifest", "config"},
        "bridge receipt image export normalization",
    )
    if (
        normalization["reconstruction_id"] != row["reconstruction_id"]
        or _image_id(normalization["image_id"], "bridge normalization image_id") != row["image_id"]
        or type(normalization["source_layout"]) is not str
        or normalization["source_layout"] not in image_export.SOURCE_LAYOUTS
    ):
        _fail("invalid-bridge-receipt", "bridge normalization projection does not match the top-level identity")
    normalization_paths = {
        "receipt": image_export.NORMALIZATION_NAME,
        "image_inspect": f"{image_export.RAW_DIRECTORY_NAME}/{image_export.IMAGE_INSPECT_NAME}",
        "image_export_archive": f"{image_export.RAW_DIRECTORY_NAME}/{image_export.IMAGE_EXPORT_ARCHIVE_NAME}",
        "oci_archive": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_ARCHIVE_NAME}",
        "layout": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_LAYOUT_NAME}",
        "index": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_INDEX_NAME}",
        "manifest": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_MANIFEST_NAME}",
        "config": f"{image_export.NORMALIZED_DIRECTORY_NAME}/{image_export.OCI_CONFIG_NAME}",
    }
    for field, expected_path in normalization_paths.items():
        _descriptor(normalization[field], f"bridge normalization {field}", expected_path=expected_path)

    oci = _exact(
        row["runtime_oci_inputs"],
        {"receipt", "reconstruction_id", "image_id", "image_inspect", "archive", "layout", "index", "manifest", "config"},
        "bridge receipt runtime OCI inputs",
    )
    if oci["reconstruction_id"] != row["reconstruction_id"] or _image_id(oci["image_id"], "bridge runtime OCI image_id") != row["image_id"]:
        _fail("invalid-bridge-receipt", "bridge runtime OCI projection does not match the top-level identity")
    oci_paths = {
        "receipt": runtime_oci.RUNTIME_OCI_INPUTS_NAME,
        "image_inspect": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.IMAGE_INSPECT_NAME}",
        "archive": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_ARCHIVE_NAME}",
        "layout": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_LAYOUT_NAME}",
        "index": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_INDEX_NAME}",
        "manifest": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_MANIFEST_NAME}",
        "config": f"{runtime_oci.RUNTIME_IMAGE_DIRECTORY_NAME}/{runtime_oci.OCI_CONFIG_NAME}",
    }
    for field, expected_path in oci_paths.items():
        _descriptor(oci[field], f"bridge runtime OCI {field}", expected_path=expected_path)

    capture = _exact(
        row["assembly_capture"],
        {"receipt", "reconstruction_id", "image_id", "capture_archive", "capture_members"},
        "bridge receipt assembly capture",
    )
    _descriptor(
        capture["receipt"],
        "bridge assembly capture receipt",
        expected_path=assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_NAME,
    )
    _descriptor(
        capture["capture_archive"],
        "bridge assembly capture archive",
        expected_path=(
            f"{assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY}/"
            f"{assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}"
        ),
    )
    if (
        capture["reconstruction_id"] != row["reconstruction_id"]
        or _image_id(capture["image_id"], "bridge assembly capture image_id") != row["image_id"]
    ):
        _fail("invalid-bridge-receipt", "bridge capture projection does not match the top-level identity")
    capture_members = _exact(
        capture["capture_members"],
        {"image-inspect.json", "oci-image-layout.tar"},
        "bridge capture members",
    )
    _member_fingerprint(capture_members["image-inspect.json"], "bridge captured image inspect")
    _member_fingerprint(capture_members["oci-image-layout.tar"], "bridge captured OCI archive")

    if row["binding_status"] != BINDING_STATUS or row["not_established"] != NOT_ESTABLISHED_STATUS:
        _fail("invalid-bridge-receipt", "bridge binding or not-established status is not the exact v1 contract")
    return row


def verify_reconstructed_runtime_image_export_assembly_content_bridge_fd(
    root_fd: int,
    *,
    source_input_root_fd: int,
    repro_build_input_root_fd: int,
    image_export_normalization_root_fd: int,
    runtime_oci_input_root_fd: int,
    assembly_capture_root_fd: int,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    reconstruction_id: str,
) -> dict[str, Any]:
    """Replay a bridge receipt against caller-held external evidence roots."""

    _bridge(lambda: common.require_private_evidence_directory_fd(root_fd, "runtime image export assembly content bridge root"))
    _require_distinct_root_fds(
        {
            "runtime image export assembly content bridge root": root_fd,
            "source inputs root": source_input_root_fd,
            "reproducibility inputs root": repro_build_input_root_fd,
            "image export normalization root": image_export_normalization_root_fd,
            "runtime OCI inputs root": runtime_oci_input_root_fd,
            "runtime assembly capture root": assembly_capture_root_fd,
        }
    )
    try:
        entries = set(os.listdir(root_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot list runtime image export assembly content bridge root: {error}")
    if entries != {BRIDGE_RECEIPT_NAME}:
        _fail("unexpected-evidence-entry", "runtime image export assembly content bridge root must contain only its receipt")
    receipt, descriptor = _read_receipt(root_fd, BRIDGE_RECEIPT_NAME, "runtime image export assembly content bridge receipt")
    _descriptor(descriptor.as_json(), "runtime image export assembly content bridge receipt descriptor", expected_path=BRIDGE_RECEIPT_NAME)
    parsed = _parse_bridge_receipt(receipt)
    expected = _replay_external_inputs(
        source_input_root_fd=source_input_root_fd,
        repro_build_input_root_fd=repro_build_input_root_fd,
        image_export_normalization_root_fd=image_export_normalization_root_fd,
        runtime_oci_input_root_fd=runtime_oci_input_root_fd,
        assembly_capture_root_fd=assembly_capture_root_fd,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_build_image_id=expected_build_image_id,
        reconstruction_id=reconstruction_id,
    )
    if parsed != expected:
        _fail("bridge-replay-mismatch", "bridge receipt differs from freshly replayed held-root inputs")
    _require_terminal_receipt(
        root_fd,
        BRIDGE_RECEIPT_NAME,
        "runtime image export assembly content bridge receipt",
        receipt,
        descriptor,
    )
    try:
        terminal_entries = set(os.listdir(root_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot re-list runtime image export assembly content bridge root: {error}")
    if terminal_entries != {BRIDGE_RECEIPT_NAME}:
        _fail("raced-input", "runtime image export assembly content bridge root changed during replay")
    return parsed


def _normalize_inputs(
    evidence_root: Path,
    source_input_root: Path,
    repro_build_input_root: Path,
    image_export_normalization_root: Path,
    runtime_oci_input_root: Path,
    assembly_capture_root: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = {
        "runtime image export assembly content bridge root": _normalized_absolute_path(evidence_root, "--evidence-root"),
        "source inputs root": _normalized_absolute_path(source_input_root, "--source-input-root"),
        "reproducibility inputs root": _normalized_absolute_path(repro_build_input_root, "--repro-build-input-root"),
        "image export normalization root": _normalized_absolute_path(
            image_export_normalization_root,
            "--image-export-normalization-root",
        ),
        "runtime OCI inputs root": _normalized_absolute_path(runtime_oci_input_root, "--runtime-oci-input-root"),
        "runtime assembly capture root": _normalized_absolute_path(assembly_capture_root, "--assembly-capture-root"),
    }
    _require_disjoint_paths(paths)
    return (
        paths["runtime image export assembly content bridge root"],
        paths["source inputs root"],
        paths["reproducibility inputs root"],
        paths["image export normalization root"],
        paths["runtime OCI inputs root"],
        paths["runtime assembly capture root"],
    )


def verify_reconstructed_runtime_image_export_assembly_content_bridge(
    evidence_root: Path,
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    image_export_normalization_root: Path,
    runtime_oci_input_root: Path,
    assembly_capture_root: Path,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    reconstruction_id: str,
) -> dict[str, Any]:
    """Open disjoint private roots and replay an existing bridge receipt."""

    (
        evidence_root,
        source_input_root,
        repro_build_input_root,
        image_export_normalization_root,
        runtime_oci_input_root,
        assembly_capture_root,
    ) = _normalize_inputs(
        evidence_root,
        source_input_root,
        repro_build_input_root,
        image_export_normalization_root,
        runtime_oci_input_root,
        assembly_capture_root,
    )
    root_fd = _bridge(
        lambda: common.open_private_evidence_directory(evidence_root, "runtime image export assembly content bridge root")
    )
    external: dict[str, int] = {}
    try:
        external = _open_external_roots(
            source_input_root=source_input_root,
            repro_build_input_root=repro_build_input_root,
            image_export_normalization_root=image_export_normalization_root,
            runtime_oci_input_root=runtime_oci_input_root,
            assembly_capture_root=assembly_capture_root,
        )
        return verify_reconstructed_runtime_image_export_assembly_content_bridge_fd(
            root_fd,
            source_input_root_fd=external["source inputs root"],
            repro_build_input_root_fd=external["reproducibility inputs root"],
            image_export_normalization_root_fd=external["image export normalization root"],
            runtime_oci_input_root_fd=external["runtime OCI inputs root"],
            assembly_capture_root_fd=external["runtime assembly capture root"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id=reconstruction_id,
        )
    finally:
        _close_fds(external)
        os.close(root_fd)


def prepare_reconstructed_runtime_image_export_assembly_content_bridge(
    evidence_root: Path,
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    image_export_normalization_root: Path,
    runtime_oci_input_root: Path,
    assembly_capture_root: Path,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
    reconstruction_id: str,
) -> dict[str, Any]:
    """Create one receipt-only bridge and self-replay it with held roots."""

    (
        evidence_root,
        source_input_root,
        repro_build_input_root,
        image_export_normalization_root,
        runtime_oci_input_root,
        assembly_capture_root,
    ) = _normalize_inputs(
        evidence_root,
        source_input_root,
        repro_build_input_root,
        image_export_normalization_root,
        runtime_oci_input_root,
        assembly_capture_root,
    )
    external = _open_external_roots(
        source_input_root=source_input_root,
        repro_build_input_root=repro_build_input_root,
        image_export_normalization_root=image_export_normalization_root,
        runtime_oci_input_root=runtime_oci_input_root,
        assembly_capture_root=assembly_capture_root,
    )
    root_fd: int | None = None
    try:
        draft = _replay_external_inputs(
            source_input_root_fd=external["source inputs root"],
            repro_build_input_root_fd=external["reproducibility inputs root"],
            image_export_normalization_root_fd=external["image export normalization root"],
            runtime_oci_input_root_fd=external["runtime OCI inputs root"],
            assembly_capture_root_fd=external["runtime assembly capture root"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id=reconstruction_id,
        )
        root_fd = _bridge(
            lambda: common.create_private_evidence_directory(evidence_root, "runtime image export assembly content bridge root")
        )
        _require_distinct_root_fds({"runtime image export assembly content bridge root": root_fd, **external})
        _bridge(
            lambda: common.write_create_only_json(
                root_fd,
                BRIDGE_RECEIPT_NAME,
                draft,
                "runtime image export assembly content bridge receipt",
            )
        )
        replayed = verify_reconstructed_runtime_image_export_assembly_content_bridge_fd(
            root_fd,
            source_input_root_fd=external["source inputs root"],
            repro_build_input_root_fd=external["reproducibility inputs root"],
            image_export_normalization_root_fd=external["image export normalization root"],
            runtime_oci_input_root_fd=external["runtime OCI inputs root"],
            assembly_capture_root_fd=external["runtime assembly capture root"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id=reconstruction_id,
        )
        if replayed != draft:
            _fail("prepublication-replay-drift", "held bridge replay differs from the draft receipt")
        return draft
    finally:
        if root_fd is not None:
            os.close(root_fd)
        _close_fds(external)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-input-root", type=Path, required=True)
    parser.add_argument("--repro-build-input-root", type=Path, required=True)
    parser.add_argument("--image-export-normalization-root", type=Path, required=True)
    parser.add_argument("--runtime-oci-input-root", type=Path, required=True)
    parser.add_argument("--assembly-capture-root", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--expected-build-image-id", required=True)
    parser.add_argument("--reconstruction-id", choices=("a", "b"), required=True)
    parser.add_argument("--verify", action="store_true", help="replay an existing bridge root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    arguments = {
        "source_input_root": args.source_input_root,
        "repro_build_input_root": args.repro_build_input_root,
        "image_export_normalization_root": args.image_export_normalization_root,
        "runtime_oci_input_root": args.runtime_oci_input_root,
        "assembly_capture_root": args.assembly_capture_root,
        "expected_source_archive_sha256": args.expected_source_archive_sha256,
        "expected_build_image_id": args.expected_build_image_id,
        "reconstruction_id": args.reconstruction_id,
    }
    try:
        if args.verify:
            receipt = verify_reconstructed_runtime_image_export_assembly_content_bridge(args.evidence_root, **arguments)
        else:
            receipt = prepare_reconstructed_runtime_image_export_assembly_content_bridge(args.evidence_root, **arguments)
    except RuntimeImageExportAssemblyContentBridgeError as error:
        payload = {
            "schema_version": BRIDGE_VERSION,
            "status": "failed",
            "qualification_status": QUALIFICATION_STATUS,
            "reason_codes": [getattr(error, "reason_code", "invalid-runtime-image-export-assembly-content-bridge")],
            "error": str(error),
        }
        sys.stderr.buffer.write(common.canonical_json_bytes(payload) + b"\n")
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
