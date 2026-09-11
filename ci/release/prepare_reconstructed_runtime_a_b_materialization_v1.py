#!/usr/bin/env python3
"""Bind two reconstructed runtime captures without claiming execution.

This CPU-only, receipt-only compositor replays two already-published
``reconstructed-runtime-assembly-capture.v1`` roots against one held RC2
source-input root, one held PR16 A/B reproducibility root, and separate held
OCI-input roots.  It publishes exactly one canonical JSON receipt in a fresh
private root.  The only new cross-arm conclusions are equality of the already
replayed release binary, release bundle, and captured ``/opt/riley`` tree.

It deliberately does *not* invoke Docker, build or export an image, start a
container or service, contact a network endpoint, access a GPU, or establish
same-invocation provenance or independent runtime capture execution.  In
particular image IDs and OCI config/archive bytes are recorded per arm but are
not required to match: the reviewed assembly recipe deliberately labels each
arm with its reconstruction ID.

The delegated capture verifier replays the existing PR16 checker, whose
``tomllib`` dependency requires Python 3.11+.  A Python 3.10 host therefore
rejects a full materialization with ``unsupported-python-runtime`` rather than
accepting a partial replay.  This module's CPU-only contract tests replace that
already-reviewed nested verifier solely to test this wrapper's FD and receipt
boundary.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

sys.dont_write_bytecode = True

import prepare_reconstructed_runtime_assembly_capture_v1 as assembly_capture
import provenance_v2_common as common


MATERIALIZATION_VERSION = "riley.reconstructed-runtime-a-b-materialization.v1"
MATERIALIZATION_RECEIPT_NAME = "reconstructed-runtime-a-b-materialization.json"
MATERIALIZATION_AUTHORITY = "held-fd-a-b-runtime-assembly-content-closure-only"
MATERIALIZATION_SCOPE = "two-arm-held-fd-runtime-content-comparison"
MATERIALIZATION_STATUS = "bound"
QUALIFICATION_STATUS = "not-run"
BASELINE_ID = "reconstructed-riley-0.1.0-rc2"
PLATFORM = {"os": "linux", "architecture": "amd64"}
MAX_RECEIPT_BYTES = common.DEFAULT_MAX_JSON_BYTES
MAX_TREE_ENTRIES = 65_536
MAX_TREE_BYTES = 2 * 1024 * 1024 * 1024

SOURCE_INPUT_RECEIPT_NAME = "reconstructed-rc2-source-inputs.json"
REPRO_INPUT_RECEIPT_NAME = "reconstructed-repro-build-inputs.json"
RUNTIME_OCI_INPUT_RECEIPT_NAME = "reconstructed-runtime-oci-inputs.json"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

BINDING_STATUS = {
    "source_inputs": "replayed-reviewed-rc2-source-inputs-v1",
    "reproducibility_inputs": "replayed-pr16-release-build-reproducibility-v1",
    "runtime_oci_inputs_a": "replayed-runtime-oci-inputs-v1",
    "runtime_oci_inputs_b": "replayed-runtime-oci-inputs-v1",
    "runtime_assembly_capture_a": "replayed-runtime-assembly-capture-v1",
    "runtime_assembly_capture_b": "replayed-runtime-assembly-capture-v1",
    "release_binary_a_b": "validated-sha256-and-byte-length-equality",
    "release_bundle_a_b": "validated-sha256-and-byte-length-equality",
    "captured_runtime_tree_a_b": "validated-sha256-entry-count-and-byte-length-equality",
}

NOT_ESTABLISHED = {
    "docker_image_export_execution": "not-established",
    "image_export_and_assembly_capture_same_invocation": "not-established",
    "runtime_build_execution": "not-established",
    "container_filesystem_capture_provenance": "not-established",
    "source_to_runtime_image": "not-established",
    "bundle_to_runtime_image": "not-established",
    "runtime_capture_independence": "not-established",
    "a_b_runtime_image_equality": "not-established",
    "a_b_oci_image_identity": "not-established",
    "rollback": "not-established",
    "freeze": "not-established",
    "qualification": QUALIFICATION_STATUS,
    "service_execution": QUALIFICATION_STATUS,
    "gpu_execution": QUALIFICATION_STATUS,
    "historical_distribution": "not-attested",
}

_T = TypeVar("_T")


class RuntimeABMaterializationError(common.ProvenanceV2Error):
    """One held input root or A/B materialization receipt is unsafe."""


def _fail(code: str, message: str) -> NoReturn:
    error = RuntimeABMaterializationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


def _common(call: Callable[[], _T]) -> _T:
    """Preserve a nested verifier's fail-closed reason code."""

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
    if raw.startswith("//") or raw == os.path.sep or os.path.normpath(raw) != raw:
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
                    "materialization-root-overlap",
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


def _sha1(value: Any, label: str) -> str:
    if type(value) is not str or SHA1_RE.fullmatch(value) is None or value == "0" * 40:
        _fail("invalid-sha1", f"{label} must be a non-zero lowercase SHA-1")
    return value


def _image_id(value: Any, label: str) -> str:
    if type(value) is not str or IMAGE_ID_RE.fullmatch(value) is None or value == "sha256:" + "0" * 64:
        _fail("invalid-image-id", f"{label} must be a non-zero lowercase sha256 image ID")
    return value


def _descriptor(value: Any, label: str, *, expected_path: str | None = None) -> common.EvidenceDescriptor:
    descriptor = _common(lambda: common.parse_descriptor(value, label))
    if descriptor.byte_length < 1:
        _fail("empty-descriptor", f"{label} must describe nonempty evidence")
    if expected_path is not None and descriptor.path != expected_path:
        _fail("descriptor-path-mismatch", f"{label} must have fixed path {expected_path!r}")
    return descriptor


def _fingerprint(value: Any, label: str, *, tree: bool = False, allow_zero_length: bool = False) -> dict[str, Any]:
    expected = {"sha256", "entry_count", "byte_length"} if tree else {"sha256", "byte_length"}
    row = _exact(value, expected, label)
    digest = _sha256(row["sha256"], f"{label}.sha256")
    length = row["byte_length"]
    if type(length) is not int or length < 0 or (not allow_zero_length and length < 1):
        _fail("invalid-fingerprint", f"{label}.byte_length must be a bounded integer")
    if tree:
        count = row["entry_count"]
        if type(count) is not int or count < 1 or count > MAX_TREE_ENTRIES or length > MAX_TREE_BYTES:
            _fail("invalid-runtime-tree", f"{label} must have a bounded positive entry count and byte length")
    result = {"sha256": digest, "byte_length": length}
    if tree:
        result["entry_count"] = row["entry_count"]
    return result


def _same_content(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> dict[str, Any]:
    left_descriptor = _descriptor(dict(left), f"{label} arm a")
    right_descriptor = _descriptor(dict(right), f"{label} arm b")
    left_fingerprint = {"sha256": left_descriptor.sha256, "byte_length": left_descriptor.byte_length}
    right_fingerprint = {"sha256": right_descriptor.sha256, "byte_length": right_descriptor.byte_length}
    if left_fingerprint != right_fingerprint:
        _fail("a-b-content-mismatch", f"{label} differs by SHA-256 or byte length")
    return left_fingerprint


def _same_tree(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_tree = _fingerprint(dict(left), "captured runtime tree arm a", tree=True)
    right_tree = _fingerprint(dict(right), "captured runtime tree arm b", tree=True)
    if left_tree != right_tree:
        _fail("a-b-runtime-tree-mismatch", "captured /opt/riley runtime trees differ")
    return left_tree


def _read_receipt(root_fd: int, name: str, label: str) -> tuple[dict[str, Any], common.EvidenceDescriptor]:
    receipt = _common(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            name,
            label,
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    descriptor = _common(
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
        _fail("raced-input", f"{label} changed during replay")


def _compact_capture_projection(
    capture_receipt: Any,
    *,
    capture_root_fd: int,
    reconstruction_id: str,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate one nested replay and retain only this receipt's authority."""

    row = _exact(
        capture_receipt,
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
        f"runtime assembly capture {reconstruction_id}",
    )
    if row["reconstruction_id"] != reconstruction_id:
        _fail("reconstruction-id-mismatch", "runtime assembly capture must match its requested arm")
    if (
        row["schema_version"] != assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_VERSION
        or row["status"] != assembly_capture.STATUS
        or row["qualification_status"] != QUALIFICATION_STATUS
        or row["authority"] != assembly_capture.AUTHORITY
        or row["capture_scope"] != assembly_capture.CAPTURE_SCOPE
        or row["baseline_id"] != BASELINE_ID
        or row["platform"] != PLATFORM
    ):
        _fail("invalid-capture-receipt", f"runtime assembly capture {reconstruction_id} has invalid fixed fields")

    source = _exact(
        row["source_inputs"],
        {"receipt", "expected_source_archive_sha256", "git_identity", "source"},
        f"runtime assembly capture {reconstruction_id} source inputs",
    )
    source_receipt = _descriptor(
        source["receipt"],
        f"runtime assembly capture {reconstruction_id} source receipt",
        expected_path=SOURCE_INPUT_RECEIPT_NAME,
    )
    source_sha = _sha256(source["expected_source_archive_sha256"], "capture source archive SHA-256")
    identity = _exact(
        source["git_identity"],
        {"tag_ref", "tag_object_sha1", "target_commit_sha1"},
        f"runtime assembly capture {reconstruction_id} source identity",
    )
    if identity["tag_ref"] != "refs/tags/riley-0.1.0-rc2":
        _fail("invalid-source-identity", "runtime assembly capture source tag must be the reviewed RC2 tag")
    _sha1(identity["tag_object_sha1"], "capture source tag object SHA-1")
    target_commit = _sha1(identity["target_commit_sha1"], "capture source target commit SHA-1")
    if not isinstance(source["source"], dict):
        _fail("invalid-source-inputs", "runtime assembly capture source projection must include an object source closure")
    if source_sha != expected_source_archive_sha256:
        _fail("reviewed-source-archive-digest-mismatch", "capture source SHA differs from the caller-reviewed SHA")
    source_projection = {
        "receipt": source_receipt.as_json(),
        "expected_source_archive_sha256": source_sha,
        "target_commit_sha1": target_commit,
    }

    repro = _exact(
        row["reproducibility_inputs"],
        {"receipt", "reproducibility_contract", "selected_build"},
        f"runtime assembly capture {reconstruction_id} reproducibility inputs",
    )
    repro_receipt = _descriptor(
        repro["receipt"],
        f"runtime assembly capture {reconstruction_id} reproducibility receipt",
        expected_path=REPRO_INPUT_RECEIPT_NAME,
    )
    contract = _exact(
        repro["reproducibility_contract"],
        {
            "schema_version",
            "gate_id",
            "source_revision",
            "source_date_epoch",
            "build_image_id",
            "platform",
            "network",
            "independent_clean_containers",
        },
        f"runtime assembly capture {reconstruction_id} reproducibility contract",
    )
    if (
        contract["schema_version"] != 1
        or contract["gate_id"] != "pr16-release-build-reproducibility-v1"
        or contract["platform"] != PLATFORM
        or contract["network"] != "none"
        or contract["independent_clean_containers"] != 2
        or type(contract["source_date_epoch"]) is not int
        or contract["source_date_epoch"] < 0
    ):
        _fail("invalid-reproducibility-contract", "capture reproducibility contract has invalid fixed fields")
    source_revision = _sha1(contract["source_revision"], "capture reproducibility source revision")
    build_image_id = _image_id(contract["build_image_id"], "capture reproducibility build image ID")
    if source_revision != target_commit or build_image_id != expected_build_image_id:
        _fail("reproducibility-input-mismatch", "capture reproducibility contract differs from held reviewed inputs")
    selected = _exact(
        repro["selected_build"],
        {"reconstruction_id", "evidence_build_id", "binary", "bundle"},
        f"runtime assembly capture {reconstruction_id} selected build",
    )
    if selected["reconstruction_id"] != reconstruction_id or selected["evidence_build_id"] != reconstruction_id.upper():
        _fail("reconstruction-id-mismatch", "capture selected build must match its arm")
    binary = _descriptor(
        selected["binary"],
        f"runtime assembly capture {reconstruction_id} selected binary",
        expected_path=f"repro-builds/{reconstruction_id}/riley",
    )
    bundle = _descriptor(
        selected["bundle"],
        f"runtime assembly capture {reconstruction_id} selected bundle",
        expected_path=f"repro-builds/{reconstruction_id}/riley.tar.gz",
    )
    repro_projection = {
        "receipt": repro_receipt.as_json(),
        "source_revision": source_revision,
        "build_image_id": build_image_id,
        "platform": dict(PLATFORM),
        "network": "none",
        "independent_clean_containers": 2,
    }

    oci = _exact(
        row["runtime_oci_inputs"],
        {"receipt", "reconstruction_id", "image_id", "image_inspect", "archive", "layout", "index", "manifest", "config"},
        f"runtime assembly capture {reconstruction_id} runtime OCI inputs",
    )
    if oci["reconstruction_id"] != reconstruction_id:
        _fail("reconstruction-id-mismatch", "capture runtime OCI inputs must match its arm")
    oci_projection = {
        "receipt": _descriptor(
            oci["receipt"],
            f"runtime assembly capture {reconstruction_id} OCI receipt",
            expected_path=RUNTIME_OCI_INPUT_RECEIPT_NAME,
        ).as_json(),
        "image_id": _image_id(oci["image_id"], f"runtime assembly capture {reconstruction_id} OCI image ID"),
        "image_inspect": _descriptor(
            oci["image_inspect"],
            f"runtime assembly capture {reconstruction_id} OCI image inspect",
            expected_path="runtime-image/docker-image-inspect.json",
        ).as_json(),
        "archive": _descriptor(
            oci["archive"],
            f"runtime assembly capture {reconstruction_id} OCI archive",
            expected_path="runtime-image/oci-image-layout.tar",
        ).as_json(),
        "layout": _descriptor(
            oci["layout"],
            f"runtime assembly capture {reconstruction_id} OCI layout",
            expected_path="runtime-image/oci-layout",
        ).as_json(),
        "index": _descriptor(
            oci["index"],
            f"runtime assembly capture {reconstruction_id} OCI index",
            expected_path="runtime-image/index.json",
        ).as_json(),
        "manifest": _descriptor(
            oci["manifest"],
            f"runtime assembly capture {reconstruction_id} OCI manifest",
            expected_path="runtime-image/manifest.json",
        ).as_json(),
        "config": _descriptor(
            oci["config"],
            f"runtime assembly capture {reconstruction_id} OCI config",
            expected_path="runtime-image/config.json",
        ).as_json(),
    }

    capture = _exact(
        row["capture"],
        {"archive", "members", "context", "image_id", "container_id", "runtime_tree"},
        f"runtime assembly capture {reconstruction_id} capture facts",
    )
    if _image_id(capture["image_id"], f"runtime assembly capture {reconstruction_id} image ID") != oci_projection["image_id"]:
        _fail("capture-runtime-oci-binding-mismatch", "capture image ID differs from its runtime OCI inputs")
    members = _exact(
        capture["members"],
        set(assembly_capture.CAPTURE_MEMBER_NAMES),
        f"runtime assembly capture {reconstruction_id} member inventory",
    )
    for name in assembly_capture.CAPTURE_MEMBER_NAMES:
        _fingerprint(
            members[name],
            f"runtime assembly capture {reconstruction_id} member {name}",
            allow_zero_length=name == "build.log",
        )
    if not isinstance(capture["context"], dict):
        _fail("invalid-capture-receipt", "capture context projection must be an object")
    capture_archive = _descriptor(
        capture["archive"],
        f"runtime assembly capture {reconstruction_id} archive",
        expected_path=(
            f"{assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY}/"
            f"{assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}"
        ),
    )
    runtime_tree = _fingerprint(capture["runtime_tree"], f"runtime assembly capture {reconstruction_id} runtime tree", tree=True)
    capture_receipt_descriptor = _common(
        lambda: common.describe_regular_relative(
            capture_root_fd,
            assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_NAME,
            f"runtime assembly capture {reconstruction_id} receipt",
            maximum_bytes=MAX_RECEIPT_BYTES,
        )
    )
    arm_projection = {
        "capture_receipt": capture_receipt_descriptor.as_json(),
        "reproducibility_artifacts": {"binary": binary.as_json(), "bundle": bundle.as_json()},
        "runtime_oci_inputs": oci_projection,
        "capture": {"archive": capture_archive.as_json(), "runtime_tree": runtime_tree},
    }
    return source_projection, repro_projection, arm_projection


def _replay_external_inputs(
    *,
    source_input_root_fd: int,
    repro_build_input_root_fd: int,
    runtime_oci_input_root_a_fd: int,
    runtime_oci_input_root_b_fd: int,
    assembly_capture_root_a_fd: int,
    assembly_capture_root_b_fd: int,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
) -> dict[str, Any]:
    expected_source_archive_sha256 = _sha256(
        expected_source_archive_sha256,
        "--expected-source-archive-sha256",
    )
    expected_build_image_id = _image_id(expected_build_image_id, "--expected-build-image-id")
    held_roots = {
        "source inputs root": source_input_root_fd,
        "reproducibility inputs root": repro_build_input_root_fd,
        "runtime OCI inputs root a": runtime_oci_input_root_a_fd,
        "runtime OCI inputs root b": runtime_oci_input_root_b_fd,
        "runtime assembly capture root a": assembly_capture_root_a_fd,
        "runtime assembly capture root b": assembly_capture_root_b_fd,
    }
    for label, descriptor in held_roots.items():
        _common(lambda descriptor=descriptor, label=label: common.require_private_evidence_directory_fd(descriptor, label))
    _require_distinct_root_fds(held_roots)

    capture_a = _common(
        lambda: assembly_capture.verify_reconstructed_runtime_assembly_capture_fd(
            assembly_capture_root_a_fd,
            source_input_root_fd=source_input_root_fd,
            repro_build_input_root_fd=repro_build_input_root_fd,
            runtime_oci_input_root_fd=runtime_oci_input_root_a_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id="a",
        )
    )
    source_a, repro_a, arm_a = _compact_capture_projection(
        capture_a,
        capture_root_fd=assembly_capture_root_a_fd,
        reconstruction_id="a",
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_build_image_id=expected_build_image_id,
    )
    capture_b = _common(
        lambda: assembly_capture.verify_reconstructed_runtime_assembly_capture_fd(
            assembly_capture_root_b_fd,
            source_input_root_fd=source_input_root_fd,
            repro_build_input_root_fd=repro_build_input_root_fd,
            runtime_oci_input_root_fd=runtime_oci_input_root_b_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
            reconstruction_id="b",
        )
    )
    source_b, repro_b, arm_b = _compact_capture_projection(
        capture_b,
        capture_root_fd=assembly_capture_root_b_fd,
        reconstruction_id="b",
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_build_image_id=expected_build_image_id,
    )
    if source_a != source_b:
        _fail("shared-input-binding-mismatch", "A/B captures do not bind the same reviewed source input closure")
    if repro_a != repro_b:
        _fail("shared-input-binding-mismatch", "A/B captures do not bind the same reproducibility input closure")
    binary = _same_content(
        arm_a["reproducibility_artifacts"]["binary"],
        arm_b["reproducibility_artifacts"]["binary"],
        "reconstructed release binary",
    )
    bundle = _same_content(
        arm_a["reproducibility_artifacts"]["bundle"],
        arm_b["reproducibility_artifacts"]["bundle"],
        "reconstructed release bundle",
    )
    runtime_tree = _same_tree(arm_a["capture"]["runtime_tree"], arm_b["capture"]["runtime_tree"])
    return {
        "schema_version": MATERIALIZATION_VERSION,
        "status": MATERIALIZATION_STATUS,
        "qualification_status": QUALIFICATION_STATUS,
        "authority": MATERIALIZATION_AUTHORITY,
        "materialization_scope": MATERIALIZATION_SCOPE,
        "baseline_id": BASELINE_ID,
        "platform": dict(PLATFORM),
        "replay_inputs": {
            "expected_source_archive_sha256": expected_source_archive_sha256,
            "expected_build_image_id": expected_build_image_id,
        },
        "source_inputs": source_a,
        "reproducibility_inputs": repro_a,
        "arms": {"a": arm_a, "b": arm_b},
        "equality": {
            "release_binary": binary,
            "release_bundle": bundle,
            "captured_runtime_tree": runtime_tree,
        },
        "binding_status": dict(BINDING_STATUS),
        "not_established": dict(NOT_ESTABLISHED),
    }


def _validate_arm_projection(value: Any, reconstruction_id: str) -> dict[str, Any]:
    row = _exact(
        value,
        {"capture_receipt", "reproducibility_artifacts", "runtime_oci_inputs", "capture"},
        f"materialization arm {reconstruction_id}",
    )
    _descriptor(
        row["capture_receipt"],
        f"materialization arm {reconstruction_id} capture receipt",
        expected_path=assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_NAME,
    )
    artifacts = _exact(
        row["reproducibility_artifacts"],
        {"binary", "bundle"},
        f"materialization arm {reconstruction_id} reproducibility artifacts",
    )
    _descriptor(
        artifacts["binary"],
        f"materialization arm {reconstruction_id} release binary",
        expected_path=f"repro-builds/{reconstruction_id}/riley",
    )
    _descriptor(
        artifacts["bundle"],
        f"materialization arm {reconstruction_id} release bundle",
        expected_path=f"repro-builds/{reconstruction_id}/riley.tar.gz",
    )
    oci = _exact(
        row["runtime_oci_inputs"],
        {"receipt", "image_id", "image_inspect", "archive", "layout", "index", "manifest", "config"},
        f"materialization arm {reconstruction_id} runtime OCI inputs",
    )
    _descriptor(oci["receipt"], f"materialization arm {reconstruction_id} OCI receipt", expected_path=RUNTIME_OCI_INPUT_RECEIPT_NAME)
    _image_id(oci["image_id"], f"materialization arm {reconstruction_id} OCI image ID")
    for field, path in (
        ("image_inspect", "runtime-image/docker-image-inspect.json"),
        ("archive", "runtime-image/oci-image-layout.tar"),
        ("layout", "runtime-image/oci-layout"),
        ("index", "runtime-image/index.json"),
        ("manifest", "runtime-image/manifest.json"),
        ("config", "runtime-image/config.json"),
    ):
        _descriptor(oci[field], f"materialization arm {reconstruction_id} OCI {field}", expected_path=path)
    capture = _exact(row["capture"], {"archive", "runtime_tree"}, f"materialization arm {reconstruction_id} capture")
    _descriptor(
        capture["archive"],
        f"materialization arm {reconstruction_id} capture archive",
        expected_path=f"{assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_DIRECTORY}/{assembly_capture.RUNTIME_ASSEMBLY_CAPTURE_ARCHIVE}",
    )
    _fingerprint(capture["runtime_tree"], f"materialization arm {reconstruction_id} runtime tree", tree=True)
    return row


def _parse_materialization_receipt(value: Any) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "schema_version",
            "status",
            "qualification_status",
            "authority",
            "materialization_scope",
            "baseline_id",
            "platform",
            "replay_inputs",
            "source_inputs",
            "reproducibility_inputs",
            "arms",
            "equality",
            "binding_status",
            "not_established",
        },
        "runtime A/B materialization receipt",
    )
    if (
        row["schema_version"] != MATERIALIZATION_VERSION
        or row["status"] != MATERIALIZATION_STATUS
        or row["qualification_status"] != QUALIFICATION_STATUS
        or row["authority"] != MATERIALIZATION_AUTHORITY
        or row["materialization_scope"] != MATERIALIZATION_SCOPE
        or row["baseline_id"] != BASELINE_ID
        or row["platform"] != PLATFORM
    ):
        _fail("invalid-materialization-receipt", "runtime A/B materialization has invalid fixed fields")
    replay_inputs = _exact(
        row["replay_inputs"],
        {"expected_source_archive_sha256", "expected_build_image_id"},
        "materialization replay inputs",
    )
    _sha256(replay_inputs["expected_source_archive_sha256"], "materialization expected source SHA-256")
    _image_id(replay_inputs["expected_build_image_id"], "materialization expected build image ID")
    source = _exact(
        row["source_inputs"],
        {"receipt", "expected_source_archive_sha256", "target_commit_sha1"},
        "materialization source inputs",
    )
    _descriptor(source["receipt"], "materialization source receipt", expected_path=SOURCE_INPUT_RECEIPT_NAME)
    _sha256(source["expected_source_archive_sha256"], "materialization source archive SHA-256")
    _sha1(source["target_commit_sha1"], "materialization source target commit SHA-1")
    repro = _exact(
        row["reproducibility_inputs"],
        {"receipt", "source_revision", "build_image_id", "platform", "network", "independent_clean_containers"},
        "materialization reproducibility inputs",
    )
    if repro["platform"] != PLATFORM or repro["network"] != "none" or repro["independent_clean_containers"] != 2:
        _fail("invalid-materialization-receipt", "materialization reproducibility projection has invalid fixed fields")
    _descriptor(repro["receipt"], "materialization reproducibility receipt", expected_path=REPRO_INPUT_RECEIPT_NAME)
    _sha1(repro["source_revision"], "materialization reproducibility source revision")
    _image_id(repro["build_image_id"], "materialization reproducibility build image ID")
    arms = _exact(row["arms"], {"a", "b"}, "materialization arms")
    _validate_arm_projection(arms["a"], "a")
    _validate_arm_projection(arms["b"], "b")
    equality = _exact(row["equality"], {"release_binary", "release_bundle", "captured_runtime_tree"}, "materialization equality")
    _fingerprint(equality["release_binary"], "materialization binary equality")
    _fingerprint(equality["release_bundle"], "materialization bundle equality")
    _fingerprint(equality["captured_runtime_tree"], "materialization runtime tree equality", tree=True)
    if row["binding_status"] != BINDING_STATUS or row["not_established"] != NOT_ESTABLISHED:
        _fail("invalid-materialization-receipt", "materialization status fields differ from the exact v1 contract")
    return row


def verify_reconstructed_runtime_a_b_materialization_fd(
    root_fd: int,
    *,
    source_input_root_fd: int,
    repro_build_input_root_fd: int,
    runtime_oci_input_root_a_fd: int,
    runtime_oci_input_root_b_fd: int,
    assembly_capture_root_a_fd: int,
    assembly_capture_root_b_fd: int,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
) -> dict[str, Any]:
    """Replay one materialization receipt against caller-held input roots."""

    _common(lambda: common.require_private_evidence_directory_fd(root_fd, "runtime A/B materialization root"))
    all_roots = {
        "runtime A/B materialization root": root_fd,
        "source inputs root": source_input_root_fd,
        "reproducibility inputs root": repro_build_input_root_fd,
        "runtime OCI inputs root a": runtime_oci_input_root_a_fd,
        "runtime OCI inputs root b": runtime_oci_input_root_b_fd,
        "runtime assembly capture root a": assembly_capture_root_a_fd,
        "runtime assembly capture root b": assembly_capture_root_b_fd,
    }
    _require_distinct_root_fds(all_roots)
    try:
        entries = set(os.listdir(root_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot list runtime A/B materialization root: {error}")
    if entries != {MATERIALIZATION_RECEIPT_NAME}:
        _fail("unexpected-evidence-entry", "runtime A/B materialization root must contain only its receipt")
    receipt, descriptor = _read_receipt(root_fd, MATERIALIZATION_RECEIPT_NAME, "runtime A/B materialization receipt")
    _descriptor(descriptor.as_json(), "runtime A/B materialization receipt descriptor", expected_path=MATERIALIZATION_RECEIPT_NAME)
    parsed = _parse_materialization_receipt(receipt)
    expected = _replay_external_inputs(
        source_input_root_fd=source_input_root_fd,
        repro_build_input_root_fd=repro_build_input_root_fd,
        runtime_oci_input_root_a_fd=runtime_oci_input_root_a_fd,
        runtime_oci_input_root_b_fd=runtime_oci_input_root_b_fd,
        assembly_capture_root_a_fd=assembly_capture_root_a_fd,
        assembly_capture_root_b_fd=assembly_capture_root_b_fd,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_build_image_id=expected_build_image_id,
    )
    if parsed != expected:
        _fail("materialization-replay-mismatch", "materialization receipt differs from freshly replayed held-root inputs")
    _require_terminal_receipt(
        root_fd,
        MATERIALIZATION_RECEIPT_NAME,
        "runtime A/B materialization receipt",
        receipt,
        descriptor,
    )
    try:
        terminal_entries = set(os.listdir(root_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot re-list runtime A/B materialization root: {error}")
    if terminal_entries != {MATERIALIZATION_RECEIPT_NAME}:
        _fail("raced-input", "runtime A/B materialization root changed during replay")
    return parsed


def _normalize_inputs(
    evidence_root: Path,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root_a: Path,
    runtime_oci_input_root_b: Path,
    assembly_capture_root_a: Path,
    assembly_capture_root_b: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    paths = {
        "runtime A/B materialization root": _normalized_absolute_path(evidence_root, "--evidence-root"),
        "source inputs root": _normalized_absolute_path(source_input_root, "--source-input-root"),
        "reproducibility inputs root": _normalized_absolute_path(repro_build_input_root, "--repro-build-input-root"),
        "runtime OCI inputs root a": _normalized_absolute_path(runtime_oci_input_root_a, "--runtime-oci-input-root-a"),
        "runtime OCI inputs root b": _normalized_absolute_path(runtime_oci_input_root_b, "--runtime-oci-input-root-b"),
        "runtime assembly capture root a": _normalized_absolute_path(assembly_capture_root_a, "--assembly-capture-root-a"),
        "runtime assembly capture root b": _normalized_absolute_path(assembly_capture_root_b, "--assembly-capture-root-b"),
    }
    _require_disjoint_paths(paths)
    return tuple(paths.values())  # type: ignore[return-value]


def _open_external_roots(
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root_a: Path,
    runtime_oci_input_root_b: Path,
    assembly_capture_root_a: Path,
    assembly_capture_root_b: Path,
) -> dict[str, int]:
    opened: dict[str, int] = {}
    try:
        for label, path in (
            ("source inputs root", source_input_root),
            ("reproducibility inputs root", repro_build_input_root),
            ("runtime OCI inputs root a", runtime_oci_input_root_a),
            ("runtime OCI inputs root b", runtime_oci_input_root_b),
            ("runtime assembly capture root a", assembly_capture_root_a),
            ("runtime assembly capture root b", assembly_capture_root_b),
        ):
            opened[label] = _common(lambda path=path, label=label: common.open_private_evidence_directory(path, label))
        _require_distinct_root_fds(opened)
        return opened
    except BaseException:
        _close_fds(opened)
        raise


def verify_reconstructed_runtime_a_b_materialization(
    evidence_root: Path,
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root_a: Path,
    runtime_oci_input_root_b: Path,
    assembly_capture_root_a: Path,
    assembly_capture_root_b: Path,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
) -> dict[str, Any]:
    """Open disjoint private roots and replay an existing materialization."""

    (
        evidence_root,
        source_input_root,
        repro_build_input_root,
        runtime_oci_input_root_a,
        runtime_oci_input_root_b,
        assembly_capture_root_a,
        assembly_capture_root_b,
    ) = _normalize_inputs(
        evidence_root,
        source_input_root,
        repro_build_input_root,
        runtime_oci_input_root_a,
        runtime_oci_input_root_b,
        assembly_capture_root_a,
        assembly_capture_root_b,
    )
    root_fd = _common(lambda: common.open_private_evidence_directory(evidence_root, "runtime A/B materialization root"))
    external: dict[str, int] = {}
    try:
        external = _open_external_roots(
            source_input_root=source_input_root,
            repro_build_input_root=repro_build_input_root,
            runtime_oci_input_root_a=runtime_oci_input_root_a,
            runtime_oci_input_root_b=runtime_oci_input_root_b,
            assembly_capture_root_a=assembly_capture_root_a,
            assembly_capture_root_b=assembly_capture_root_b,
        )
        return verify_reconstructed_runtime_a_b_materialization_fd(
            root_fd,
            source_input_root_fd=external["source inputs root"],
            repro_build_input_root_fd=external["reproducibility inputs root"],
            runtime_oci_input_root_a_fd=external["runtime OCI inputs root a"],
            runtime_oci_input_root_b_fd=external["runtime OCI inputs root b"],
            assembly_capture_root_a_fd=external["runtime assembly capture root a"],
            assembly_capture_root_b_fd=external["runtime assembly capture root b"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
        )
    finally:
        _close_fds(external)
        os.close(root_fd)


def prepare_reconstructed_runtime_a_b_materialization(
    evidence_root: Path,
    *,
    source_input_root: Path,
    repro_build_input_root: Path,
    runtime_oci_input_root_a: Path,
    runtime_oci_input_root_b: Path,
    assembly_capture_root_a: Path,
    assembly_capture_root_b: Path,
    expected_source_archive_sha256: str,
    expected_build_image_id: str,
) -> dict[str, Any]:
    """Create one receipt-only A/B content materialization and self-replay it."""

    (
        evidence_root,
        source_input_root,
        repro_build_input_root,
        runtime_oci_input_root_a,
        runtime_oci_input_root_b,
        assembly_capture_root_a,
        assembly_capture_root_b,
    ) = _normalize_inputs(
        evidence_root,
        source_input_root,
        repro_build_input_root,
        runtime_oci_input_root_a,
        runtime_oci_input_root_b,
        assembly_capture_root_a,
        assembly_capture_root_b,
    )
    external = _open_external_roots(
        source_input_root=source_input_root,
        repro_build_input_root=repro_build_input_root,
        runtime_oci_input_root_a=runtime_oci_input_root_a,
        runtime_oci_input_root_b=runtime_oci_input_root_b,
        assembly_capture_root_a=assembly_capture_root_a,
        assembly_capture_root_b=assembly_capture_root_b,
    )
    root_fd: int | None = None
    try:
        draft = _replay_external_inputs(
            source_input_root_fd=external["source inputs root"],
            repro_build_input_root_fd=external["reproducibility inputs root"],
            runtime_oci_input_root_a_fd=external["runtime OCI inputs root a"],
            runtime_oci_input_root_b_fd=external["runtime OCI inputs root b"],
            assembly_capture_root_a_fd=external["runtime assembly capture root a"],
            assembly_capture_root_b_fd=external["runtime assembly capture root b"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
        )
        root_fd = _common(lambda: common.create_private_evidence_directory(evidence_root, "runtime A/B materialization root"))
        _require_distinct_root_fds({"runtime A/B materialization root": root_fd, **external})
        _common(
            lambda: common.write_create_only_json(
                root_fd,
                MATERIALIZATION_RECEIPT_NAME,
                draft,
                "runtime A/B materialization receipt",
            )
        )
        replayed = verify_reconstructed_runtime_a_b_materialization_fd(
            root_fd,
            source_input_root_fd=external["source inputs root"],
            repro_build_input_root_fd=external["reproducibility inputs root"],
            runtime_oci_input_root_a_fd=external["runtime OCI inputs root a"],
            runtime_oci_input_root_b_fd=external["runtime OCI inputs root b"],
            assembly_capture_root_a_fd=external["runtime assembly capture root a"],
            assembly_capture_root_b_fd=external["runtime assembly capture root b"],
            expected_source_archive_sha256=expected_source_archive_sha256,
            expected_build_image_id=expected_build_image_id,
        )
        if replayed != draft:
            _fail("prepublication-replay-drift", "held A/B materialization replay differs from the draft receipt")
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
    parser.add_argument("--runtime-oci-input-root-a", type=Path, required=True)
    parser.add_argument("--runtime-oci-input-root-b", type=Path, required=True)
    parser.add_argument("--assembly-capture-root-a", type=Path, required=True)
    parser.add_argument("--assembly-capture-root-b", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--expected-build-image-id", required=True)
    parser.add_argument("--verify", action="store_true", help="replay an existing evidence root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    arguments = {
        "source_input_root": args.source_input_root,
        "repro_build_input_root": args.repro_build_input_root,
        "runtime_oci_input_root_a": args.runtime_oci_input_root_a,
        "runtime_oci_input_root_b": args.runtime_oci_input_root_b,
        "assembly_capture_root_a": args.assembly_capture_root_a,
        "assembly_capture_root_b": args.assembly_capture_root_b,
        "expected_source_archive_sha256": args.expected_source_archive_sha256,
        "expected_build_image_id": args.expected_build_image_id,
    }
    try:
        if args.verify:
            receipt = verify_reconstructed_runtime_a_b_materialization(args.evidence_root, **arguments)
        else:
            receipt = prepare_reconstructed_runtime_a_b_materialization(args.evidence_root, **arguments)
    except RuntimeABMaterializationError as error:
        payload = {
            "schema_version": MATERIALIZATION_VERSION,
            "status": "failed",
            "qualification_status": QUALIFICATION_STATUS,
            "reason_codes": [getattr(error, "reason_code", "invalid-runtime-a-b-materialization")],
            "error": str(error),
        }
        sys.stderr.buffer.write(common.canonical_json_bytes(payload) + b"\n")
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
