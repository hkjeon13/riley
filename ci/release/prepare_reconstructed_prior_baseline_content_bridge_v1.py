#!/usr/bin/env python3
"""Create and replay a narrow cross-root C02-P1 content-binding bridge.

This tool deliberately does *not* materialize a new reconstructed baseline.
It only binds the opaque source and OCI leaves already accepted by the v2
baseline checker to the reviewed RC2 source-inputs v1 closure and to the
per-arm runtime-OCI-inputs v1 closures.  In particular, it does not establish
that a runtime image was assembled from either reconstruction's binary or
bundle, nor does it establish an independent runtime capture.

No Docker, GPU, service, network, shell, or build invocation occurs here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import check_reconstructed_prior_baseline_v2 as baseline
import prepare_reconstructed_rc2_inputs_v1 as source_inputs
import prepare_reconstructed_runtime_oci_inputs_v1 as runtime_oci
import provenance_v2_common as common


BRIDGE_VERSION = "riley.reconstructed-prior-baseline-content-bridge.v1"
BRIDGE_RECEIPT_NAME = "reconstructed-prior-baseline-content-bridge.json"
BRIDGE_AUTHORITY = "cross-root-content-bridge-only"
BRIDGE_STATUS = "bound"
QUALIFICATION_STATUS = "not-run"
MAX_DOCUMENT_BYTES = common.DEFAULT_MAX_JSON_BYTES

SOURCE_ARCHIVE_CONTENT_BINDING = "reviewed-source-inputs-v1-replayed"
OCI_ARCHIVE_CONTENT_BINDING = "validated-via-runtime-oci-inputs-v1"
OCI_A_B_CONTENT_EQUALITY = "validated"
NOT_ESTABLISHED = "not-established"

_T = TypeVar("_T")


class ReconstructedPriorBaselineContentBridgeError(common.ProvenanceV2Error):
    """A bridge receipt or one of its pinned external inputs is unsafe."""


def _fail(code: str, message: str) -> NoReturn:
    error = ReconstructedPriorBaselineContentBridgeError(message)
    setattr(error, "reason_code", code)
    raise error


def _bridge(call: Callable[[], _T]) -> _T:
    """Preserve an imported verifier's reason code under the bridge API."""

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


def _path_overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_disjoint_paths(paths: Mapping[str, Path]) -> None:
    values = tuple(paths.items())
    for index, (left_label, left) in enumerate(values):
        for right_label, right in values[index + 1 :]:
            if _path_overlaps(left, right):
                _fail(
                    "bridge-root-overlap",
                    f"{left_label} and {right_label} must be disjoint normalized paths",
                )


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = sorted(value) if isinstance(value, dict) else []
        _fail(
            "unknown-or-missing-field",
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
        )
    return value


def _string(value: Any, label: str, *, maximum: int = 1024) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        _fail("invalid-string", f"{label} must be a bounded non-empty string")
    return value


def _descriptor(value: Any, label: str) -> dict[str, Any]:
    return _bridge(lambda: common.parse_descriptor(value, label)).as_json()


def _source(value: Any, label: str) -> dict[str, Any]:
    return _bridge(lambda: baseline._source(value, label)).as_json()


def _identity(value: Any, label: str) -> dict[str, str]:
    row = _exact(value, {"tag_ref", "tag_object_sha1", "target_commit_sha1"}, label)
    tag_ref = _string(row["tag_ref"], f"{label}.tag_ref", maximum=256)
    tag_object = _bridge(lambda: baseline._git_sha1(row["tag_object_sha1"], f"{label}.tag_object_sha1"))
    target = _bridge(lambda: baseline._git_sha1(row["target_commit_sha1"], f"{label}.target_commit_sha1"))
    return {
        "tag_ref": tag_ref,
        "tag_object_sha1": tag_object,
        "target_commit_sha1": target,
    }


def _image_id(value: Any, label: str) -> str:
    return _bridge(lambda: runtime_oci._sha256_digest(value, label))


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


def _open_external_roots(
    *,
    baseline_root: Path,
    source_input_root: Path,
    runtime_oci_a_root: Path,
    runtime_oci_b_root: Path,
) -> dict[str, int]:
    opened: dict[str, int] = {}
    try:
        for label, path in (
            ("baseline root", baseline_root),
            ("source inputs root", source_input_root),
            ("runtime OCI A root", runtime_oci_a_root),
            ("runtime OCI B root", runtime_oci_b_root),
        ):
            opened[label] = _bridge(lambda path=path, label=label: common.open_private_evidence_directory(path, label))
        _require_distinct_root_fds(opened)
        return opened
    except BaseException:
        for directory_fd in reversed(tuple(opened.values())):
            os.close(directory_fd)
        raise


def _close_roots(roots: Mapping[str, int]) -> None:
    for directory_fd in reversed(tuple(roots.values())):
        os.close(directory_fd)


def _read_baseline_manifest(
    root_fd: int,
    manifest_relative: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _bridge(lambda: common.require_private_evidence_directory_fd(root_fd, "baseline root"))
    raw = _bridge(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            manifest_relative,
            "baseline manifest",
            maximum_bytes=MAX_DOCUMENT_BYTES,
        )
    )
    document = _bridge(lambda: common.parse_canonical_json(raw, "baseline manifest", maximum_bytes=MAX_DOCUMENT_BYTES))
    assert isinstance(document, dict)
    report = _bridge(lambda: baseline.evaluate(root_fd, document))
    terminal_raw = _bridge(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            manifest_relative,
            "baseline manifest",
            maximum_bytes=MAX_DOCUMENT_BYTES,
        )
    )
    if terminal_raw != raw:
        _fail("raced-input", "baseline manifest changed during its held-root replay")
    return (
        common.descriptor_for_bytes(manifest_relative, raw, "baseline manifest").as_json(),
        document,
        report,
    )


def _read_private_receipt(
    root_fd: int,
    name: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _bridge(
        lambda: common.read_private_canonical_json_leaf(
            root_fd,
            name,
            label,
            maximum_bytes=MAX_DOCUMENT_BYTES,
        )
    )
    raw = common.canonical_json_bytes(document)
    return document, common.descriptor_for_bytes(name, raw, label).as_json()


def _fingerprint(value: Any, label: str) -> tuple[str, int]:
    parsed = _bridge(lambda: common.parse_descriptor(value, label))
    return parsed.sha256, parsed.byte_length


def _assert_same_fingerprint(left: Any, right: Any, label: str) -> None:
    if _fingerprint(left, f"{label} left") != _fingerprint(right, f"{label} right"):
        _fail("cross-root-content-mismatch", f"{label} SHA-256 or byte length differs across roots")


def _assert_same_source(
    baseline_source: Mapping[str, Any],
    source_input_source: Mapping[str, Any],
) -> None:
    left = _source(baseline_source, "baseline source")
    right = _source(source_input_source, "source inputs source")
    if left["tag_name"] != right["tag_name"]:
        _fail("cross-root-source-binding-mismatch", "baseline and source inputs tag names differ")
    for field, label in (
        ("tag_object", "Git tag object"),
        ("tag_target", "Git tag target"),
        ("archive", "source archive"),
    ):
        _assert_same_fingerprint(left[field], right[field], label)


def _assert_source_identity(
    baseline_report: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
) -> None:
    if baseline_report.get("baseline_id") != source_receipt.get("baseline_id"):
        _fail("cross-root-source-binding-mismatch", "baseline IDs differ across source roots")
    report_identity = _exact(
        baseline_report.get("git_identity"),
        {"tag_name", "tag_ref", "tag_object_sha1", "target_commit_sha1"},
        "baseline report.git_identity",
    )
    source_identity = _identity(source_receipt.get("git_identity"), "source inputs receipt.git_identity")
    if (
        report_identity["tag_ref"] != source_identity["tag_ref"]
        or report_identity["tag_object_sha1"] != source_identity["tag_object_sha1"]
        or report_identity["target_commit_sha1"] != source_identity["target_commit_sha1"]
    ):
        _fail("cross-root-source-binding-mismatch", "baseline Git identity differs from source inputs")


def _arm_view(
    baseline_report: Mapping[str, Any],
    arm: str,
    image_id: str,
) -> dict[str, Any]:
    reproductions = _exact(baseline_report.get("reproductions"), {"a", "b"}, "baseline report.reproductions")
    row = _exact(
        reproductions.get(arm),
        {"receipt", "recipe_inspect", "image_inspect", "runtime_image_inspect_raw", "artifacts"},
        f"baseline report reproduction {arm}",
    )
    artifacts = _exact(row["artifacts"], {"binary", "bundle", "oci"}, f"baseline report reproduction {arm}.artifacts")
    oci = _exact(artifacts["oci"], {"archive", "layout", "manifest"}, f"baseline report reproduction {arm}.artifacts.oci")
    return {
        "runtime_image_inspect_raw": _descriptor(
            row["runtime_image_inspect_raw"], f"baseline report reproduction {arm}.runtime_image_inspect_raw"
        ),
        "oci": {
            field: _descriptor(value, f"baseline report reproduction {arm}.oci.{field}")
            for field, value in oci.items()
        },
        "image_id": image_id,
    }


def _assert_arm_binding(
    arm: str,
    baseline_arm: Mapping[str, Any],
    oci_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if oci_receipt.get("reconstruction_id") != arm:
        _fail("reconstruction-id-mismatch", f"runtime OCI receipt does not belong to reconstruction {arm}")
    image_id = _image_id(oci_receipt.get("image_id"), f"runtime OCI {arm} image_id")
    if baseline_arm["image_id"] != image_id:
        _fail("cross-root-image-id-mismatch", f"reconstruction {arm} image ID differs across roots")
    _assert_same_fingerprint(
        baseline_arm["runtime_image_inspect_raw"],
        oci_receipt.get("image_inspect"),
        f"reconstruction {arm} raw runtime image inspect",
    )
    for field in ("archive", "layout", "manifest"):
        _assert_same_fingerprint(
            baseline_arm["oci"][field],
            oci_receipt.get(field),
            f"reconstruction {arm} OCI {field}",
        )
    return {
        "reconstruction_id": arm,
        "image_id": image_id,
        "image_inspect": _descriptor(oci_receipt.get("image_inspect"), f"runtime OCI {arm}.image_inspect"),
        "archive": _descriptor(oci_receipt.get("archive"), f"runtime OCI {arm}.archive"),
        "layout": _descriptor(oci_receipt.get("layout"), f"runtime OCI {arm}.layout"),
        "index": _descriptor(oci_receipt.get("index"), f"runtime OCI {arm}.index"),
        "manifest": _descriptor(oci_receipt.get("manifest"), f"runtime OCI {arm}.manifest"),
        "config": _descriptor(oci_receipt.get("config"), f"runtime OCI {arm}.config"),
    }


def _assert_oci_a_b_equality(oci_a: Mapping[str, Any], oci_b: Mapping[str, Any]) -> None:
    # The v2 baseline proves equality for the OCI content leaves and image ID,
    # but intentionally does not require byte-identical raw Docker inspect
    # captures.  Repo tags or other non-ID diagnostic fields may differ by arm.
    for field in ("archive", "layout", "manifest"):
        _assert_same_fingerprint(oci_a[field], oci_b[field], f"runtime OCI A/B {field}")
    if oci_a["image_id"] != oci_b["image_id"]:
        _fail("a-b-equality-mismatch", "runtime OCI A/B image IDs differ")


def _replay_external_inputs(
    *,
    baseline_root_fd: int,
    baseline_manifest: str,
    source_input_root_fd: int,
    expected_source_archive_sha256: str,
    runtime_oci_a_root_fd: int,
    runtime_oci_b_root_fd: int,
) -> dict[str, Any]:
    """Replay every external closure through its already-held root FD."""

    _require_distinct_root_fds(
        {
            "baseline root": baseline_root_fd,
            "source inputs root": source_input_root_fd,
            "runtime OCI A root": runtime_oci_a_root_fd,
            "runtime OCI B root": runtime_oci_b_root_fd,
        }
    )
    manifest_descriptor, _manifest, baseline_report = _read_baseline_manifest(
        baseline_root_fd, baseline_manifest
    )
    if (
        baseline_report.get("source_archive_content_binding") != "not-validated"
        or baseline_report.get("oci_archive_content_binding") != "not-validated"
    ):
        _fail("unexpected-v2-content-binding-status", "this bridge only accepts the exact v2 opaque-content report")
    source_verified = _bridge(
        lambda: source_inputs.verify_reconstructed_rc2_inputs_fd(
            source_input_root_fd,
            expected_source_archive_sha256=expected_source_archive_sha256,
        )
    )
    source_receipt, source_receipt_descriptor = _read_private_receipt(
        source_input_root_fd,
        source_inputs.SOURCE_INPUTS_NAME,
        "source inputs receipt",
    )
    if source_receipt != source_verified:
        _fail("raced-input", "source inputs receipt changed during its held-root replay")
    oci_a_verified = _bridge(lambda: runtime_oci.verify_reconstructed_runtime_oci_inputs_fd(runtime_oci_a_root_fd))
    oci_a_receipt, oci_a_receipt_descriptor = _read_private_receipt(
        runtime_oci_a_root_fd,
        runtime_oci.RUNTIME_OCI_INPUTS_NAME,
        "runtime OCI A receipt",
    )
    if oci_a_receipt != oci_a_verified:
        _fail("raced-input", "runtime OCI A receipt changed during its held-root replay")
    oci_b_verified = _bridge(lambda: runtime_oci.verify_reconstructed_runtime_oci_inputs_fd(runtime_oci_b_root_fd))
    oci_b_receipt, oci_b_receipt_descriptor = _read_private_receipt(
        runtime_oci_b_root_fd,
        runtime_oci.RUNTIME_OCI_INPUTS_NAME,
        "runtime OCI B receipt",
    )
    if oci_b_receipt != oci_b_verified:
        _fail("raced-input", "runtime OCI B receipt changed during its held-root replay")

    _assert_same_source(baseline_report["source"], source_receipt["source"])
    _assert_source_identity(baseline_report, source_receipt)
    equality = _exact(
        baseline_report.get("equality"),
        {"binary", "bundle", "oci_archive", "oci_layout", "oci_manifest", "oci_image"},
        "baseline report.equality",
    )
    image_equality = _exact(equality["oci_image"], {"a", "b", "image_id"}, "baseline report.equality.oci_image")
    image_id = _image_id(image_equality["image_id"], "baseline report OCI image ID")
    if image_equality["a"] != image_id or image_equality["b"] != image_id:
        _fail("a-b-equality-mismatch", "baseline report does not retain one A/B OCI image ID")
    baseline_a = _arm_view(baseline_report, "a", image_id)
    baseline_b = _arm_view(baseline_report, "b", image_id)
    oci_a = _assert_arm_binding("a", baseline_a, oci_a_receipt)
    oci_b = _assert_arm_binding("b", baseline_b, oci_b_receipt)
    _assert_oci_a_b_equality(oci_a, oci_b)

    return {
        "schema_version": BRIDGE_VERSION,
        "status": BRIDGE_STATUS,
        "qualification_status": QUALIFICATION_STATUS,
        "authority": BRIDGE_AUTHORITY,
        "baseline": {
            "manifest": manifest_descriptor,
            "baseline_id": baseline_report["baseline_id"],
            "source": _source(baseline_report["source"], "baseline report.source"),
            "reproductions": {"a": baseline_a, "b": baseline_b},
        },
        "source_inputs": {
            "receipt": source_receipt_descriptor,
            "expected_source_archive_sha256": expected_source_archive_sha256,
            "git_identity": _identity(source_receipt["git_identity"], "source inputs receipt.git_identity"),
            "source": _source(source_receipt["source"], "source inputs receipt.source"),
        },
        "runtime_oci_inputs": {
            "a": {"receipt": oci_a_receipt_descriptor, **oci_a},
            "b": {"receipt": oci_b_receipt_descriptor, **oci_b},
        },
        "binding_status": {
            "v2_report_source_archive_content_binding": "not-validated",
            "v2_report_oci_archive_content_binding": "not-validated",
            "source_archive_content_binding": SOURCE_ARCHIVE_CONTENT_BINDING,
            "oci_archive_content_binding": OCI_ARCHIVE_CONTENT_BINDING,
            "oci_a_b_content_equality": OCI_A_B_CONTENT_EQUALITY,
        },
        "not_established": {
            "runtime_build_invocation_binding": NOT_ESTABLISHED,
            "bundle_to_runtime_image_binding": NOT_ESTABLISHED,
            "source_to_runtime_image_binding": NOT_ESTABLISHED,
            "runtime_capture_independence_binding": NOT_ESTABLISHED,
            "rollback_binding": NOT_ESTABLISHED,
            "freeze_binding": NOT_ESTABLISHED,
            "qualification": QUALIFICATION_STATUS,
        },
    }


def _parse_bridge_receipt(value: Any) -> dict[str, Any]:
    """Reject field aliases before the fresh external replay compares facts."""

    row = _exact(
        value,
        {
            "schema_version",
            "status",
            "qualification_status",
            "authority",
            "baseline",
            "source_inputs",
            "runtime_oci_inputs",
            "binding_status",
            "not_established",
        },
        "content bridge receipt",
    )
    if (
        row["schema_version"] != BRIDGE_VERSION
        or row["status"] != BRIDGE_STATUS
        or row["qualification_status"] != QUALIFICATION_STATUS
        or row["authority"] != BRIDGE_AUTHORITY
    ):
        _fail("invalid-bridge-receipt", "content bridge receipt is not the exact v1 contract")
    baseline_row = _exact(
        row["baseline"],
        {"manifest", "baseline_id", "source", "reproductions"},
        "content bridge receipt.baseline",
    )
    _descriptor(baseline_row["manifest"], "content bridge baseline manifest")
    _string(baseline_row["baseline_id"], "content bridge baseline_id", maximum=160)
    _source(baseline_row["source"], "content bridge baseline source")
    reproductions = _exact(baseline_row["reproductions"], {"a", "b"}, "content bridge baseline reproductions")
    for arm in ("a", "b"):
        arm_row = _exact(
            reproductions[arm],
            {"runtime_image_inspect_raw", "oci", "image_id"},
            f"content bridge baseline reproduction {arm}",
        )
        _descriptor(arm_row["runtime_image_inspect_raw"], f"content bridge baseline {arm} raw inspect")
        oci_row = _exact(arm_row["oci"], {"archive", "layout", "manifest"}, f"content bridge baseline {arm} OCI")
        for field, descriptor in oci_row.items():
            _descriptor(descriptor, f"content bridge baseline {arm} OCI {field}")
        _image_id(arm_row["image_id"], f"content bridge baseline {arm} image_id")

    source_row = _exact(
        row["source_inputs"],
        {"receipt", "expected_source_archive_sha256", "git_identity", "source"},
        "content bridge source inputs",
    )
    receipt_descriptor = _descriptor(source_row["receipt"], "content bridge source receipt")
    if receipt_descriptor["path"] != source_inputs.SOURCE_INPUTS_NAME:
        _fail("bridge-receipt-path-mismatch", "content bridge source receipt path is not fixed")
    _bridge(lambda: source_inputs._expected_sha256(source_row["expected_source_archive_sha256"]))
    _identity(source_row["git_identity"], "content bridge source Git identity")
    _source(source_row["source"], "content bridge source binding")

    runtime_rows = _exact(row["runtime_oci_inputs"], {"a", "b"}, "content bridge runtime OCI inputs")
    for arm in ("a", "b"):
        runtime_row = _exact(
            runtime_rows[arm],
            {"receipt", "reconstruction_id", "image_id", "image_inspect", "archive", "layout", "index", "manifest", "config"},
            f"content bridge runtime OCI {arm}",
        )
        receipt_descriptor = _descriptor(runtime_row["receipt"], f"content bridge runtime OCI {arm} receipt")
        if receipt_descriptor["path"] != runtime_oci.RUNTIME_OCI_INPUTS_NAME:
            _fail("bridge-receipt-path-mismatch", f"content bridge runtime OCI {arm} receipt path is not fixed")
        if runtime_row["reconstruction_id"] != arm:
            _fail("reconstruction-id-mismatch", f"content bridge runtime OCI {arm} reconstruction ID is wrong")
        _image_id(runtime_row["image_id"], f"content bridge runtime OCI {arm} image_id")
        for field in ("image_inspect", "archive", "layout", "index", "manifest", "config"):
            _descriptor(runtime_row[field], f"content bridge runtime OCI {arm}.{field}")

    binding = _exact(
        row["binding_status"],
        {
            "v2_report_source_archive_content_binding",
            "v2_report_oci_archive_content_binding",
            "source_archive_content_binding",
            "oci_archive_content_binding",
            "oci_a_b_content_equality",
        },
        "content bridge binding status",
    )
    if binding != {
        "v2_report_source_archive_content_binding": "not-validated",
        "v2_report_oci_archive_content_binding": "not-validated",
        "source_archive_content_binding": SOURCE_ARCHIVE_CONTENT_BINDING,
        "oci_archive_content_binding": OCI_ARCHIVE_CONTENT_BINDING,
        "oci_a_b_content_equality": OCI_A_B_CONTENT_EQUALITY,
    }:
        _fail("invalid-bridge-receipt", "content bridge binding status is not the exact v1 contract")
    not_established = _exact(
        row["not_established"],
        {
            "runtime_build_invocation_binding",
            "bundle_to_runtime_image_binding",
            "source_to_runtime_image_binding",
            "runtime_capture_independence_binding",
            "rollback_binding",
            "freeze_binding",
            "qualification",
        },
        "content bridge not-established status",
    )
    expected_not_established = {
        "runtime_build_invocation_binding": NOT_ESTABLISHED,
        "bundle_to_runtime_image_binding": NOT_ESTABLISHED,
        "source_to_runtime_image_binding": NOT_ESTABLISHED,
        "runtime_capture_independence_binding": NOT_ESTABLISHED,
        "rollback_binding": NOT_ESTABLISHED,
        "freeze_binding": NOT_ESTABLISHED,
        "qualification": QUALIFICATION_STATUS,
    }
    if not_established != expected_not_established:
        _fail("invalid-bridge-receipt", "content bridge must retain every unestablished boundary")
    return row


def verify_reconstructed_prior_baseline_content_bridge_fd(
    bridge_root_fd: int,
    *,
    baseline_root_fd: int,
    baseline_manifest: str,
    source_input_root_fd: int,
    expected_source_archive_sha256: str,
    runtime_oci_a_root_fd: int,
    runtime_oci_b_root_fd: int,
) -> dict[str, Any]:
    """Replay one bridge receipt and all its external held-root inputs."""

    _bridge(lambda: common.require_private_evidence_directory_fd(bridge_root_fd, "content bridge root"))
    _require_distinct_root_fds(
        {
            "content bridge root": bridge_root_fd,
            "baseline root": baseline_root_fd,
            "source inputs root": source_input_root_fd,
            "runtime OCI A root": runtime_oci_a_root_fd,
            "runtime OCI B root": runtime_oci_b_root_fd,
        }
    )
    try:
        entries = set(os.listdir(bridge_root_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot list content bridge root: {error}")
    if entries != {BRIDGE_RECEIPT_NAME}:
        _fail("unexpected-evidence-entry", "content bridge root must contain exactly its receipt")
    receipt = _bridge(
        lambda: common.read_private_canonical_json_leaf(
            bridge_root_fd,
            BRIDGE_RECEIPT_NAME,
            "content bridge receipt",
            maximum_bytes=MAX_DOCUMENT_BYTES,
        )
    )
    parsed = _parse_bridge_receipt(receipt)
    expected_sha256 = _bridge(lambda: source_inputs._expected_sha256(expected_source_archive_sha256))
    expected = _replay_external_inputs(
        baseline_root_fd=baseline_root_fd,
        baseline_manifest=baseline_manifest,
        source_input_root_fd=source_input_root_fd,
        expected_source_archive_sha256=expected_sha256,
        runtime_oci_a_root_fd=runtime_oci_a_root_fd,
        runtime_oci_b_root_fd=runtime_oci_b_root_fd,
    )
    if parsed != expected:
        _fail("bridge-replay-mismatch", "content bridge receipt differs from freshly replayed external inputs")
    terminal = _bridge(
        lambda: common.read_private_canonical_json_leaf(
            bridge_root_fd,
            BRIDGE_RECEIPT_NAME,
            "content bridge receipt",
            maximum_bytes=MAX_DOCUMENT_BYTES,
        )
    )
    if terminal != receipt:
        _fail("raced-input", "content bridge receipt changed during replay")
    try:
        terminal_entries = set(os.listdir(bridge_root_fd))
    except OSError as error:
        _fail("unsafe-evidence-directory", f"cannot re-list content bridge root: {error}")
    if terminal_entries != {BRIDGE_RECEIPT_NAME}:
        _fail("raced-input", "content bridge root entries changed during replay")
    return parsed


def _normalize_inputs(
    evidence_root: Path,
    baseline_root: Path,
    baseline_manifest: str,
    source_input_root: Path,
    runtime_oci_a_root: Path,
    runtime_oci_b_root: Path,
) -> tuple[Path, Path, str, Path, Path, Path]:
    normalized = {
        "content bridge root": _normalized_absolute_path(evidence_root, "--evidence-root"),
        "baseline root": _normalized_absolute_path(baseline_root, "--baseline-root"),
        "source inputs root": _normalized_absolute_path(source_input_root, "--source-input-root"),
        "runtime OCI A root": _normalized_absolute_path(runtime_oci_a_root, "--runtime-oci-a-root"),
        "runtime OCI B root": _normalized_absolute_path(runtime_oci_b_root, "--runtime-oci-b-root"),
    }
    _require_disjoint_paths(normalized)
    manifest = _bridge(lambda: common.validate_relative_path(baseline_manifest, "--baseline-manifest"))
    return (
        normalized["content bridge root"],
        normalized["baseline root"],
        manifest,
        normalized["source inputs root"],
        normalized["runtime OCI A root"],
        normalized["runtime OCI B root"],
    )


def verify_reconstructed_prior_baseline_content_bridge(
    evidence_root: Path,
    *,
    baseline_root: Path,
    baseline_manifest: str,
    source_input_root: Path,
    expected_source_archive_sha256: str,
    runtime_oci_a_root: Path,
    runtime_oci_b_root: Path,
) -> dict[str, Any]:
    """Open all roots safely and replay an existing bridge receipt."""

    (
        evidence_root,
        baseline_root,
        baseline_manifest,
        source_input_root,
        runtime_oci_a_root,
        runtime_oci_b_root,
    ) = _normalize_inputs(
        evidence_root,
        baseline_root,
        baseline_manifest,
        source_input_root,
        runtime_oci_a_root,
        runtime_oci_b_root,
    )
    expected_sha256 = _bridge(lambda: source_inputs._expected_sha256(expected_source_archive_sha256))
    bridge_fd = _bridge(lambda: common.open_private_evidence_directory(evidence_root, "content bridge root"))
    external: dict[str, int] = {}
    try:
        external = _open_external_roots(
            baseline_root=baseline_root,
            source_input_root=source_input_root,
            runtime_oci_a_root=runtime_oci_a_root,
            runtime_oci_b_root=runtime_oci_b_root,
        )
        return verify_reconstructed_prior_baseline_content_bridge_fd(
            bridge_fd,
            baseline_root_fd=external["baseline root"],
            baseline_manifest=baseline_manifest,
            source_input_root_fd=external["source inputs root"],
            expected_source_archive_sha256=expected_sha256,
            runtime_oci_a_root_fd=external["runtime OCI A root"],
            runtime_oci_b_root_fd=external["runtime OCI B root"],
        )
    finally:
        _close_roots(external)
        os.close(bridge_fd)


def prepare_reconstructed_prior_baseline_content_bridge(
    evidence_root: Path,
    *,
    baseline_root: Path,
    baseline_manifest: str,
    source_input_root: Path,
    expected_source_archive_sha256: str,
    runtime_oci_a_root: Path,
    runtime_oci_b_root: Path,
) -> dict[str, Any]:
    """Create one fresh bridge receipt after a full held-FD replay of inputs."""

    (
        evidence_root,
        baseline_root,
        baseline_manifest,
        source_input_root,
        runtime_oci_a_root,
        runtime_oci_b_root,
    ) = _normalize_inputs(
        evidence_root,
        baseline_root,
        baseline_manifest,
        source_input_root,
        runtime_oci_a_root,
        runtime_oci_b_root,
    )
    expected_sha256 = _bridge(lambda: source_inputs._expected_sha256(expected_source_archive_sha256))
    external = _open_external_roots(
        baseline_root=baseline_root,
        source_input_root=source_input_root,
        runtime_oci_a_root=runtime_oci_a_root,
        runtime_oci_b_root=runtime_oci_b_root,
    )
    bridge_fd: int | None = None
    try:
        draft = _replay_external_inputs(
            baseline_root_fd=external["baseline root"],
            baseline_manifest=baseline_manifest,
            source_input_root_fd=external["source inputs root"],
            expected_source_archive_sha256=expected_sha256,
            runtime_oci_a_root_fd=external["runtime OCI A root"],
            runtime_oci_b_root_fd=external["runtime OCI B root"],
        )
        bridge_fd = _bridge(lambda: common.create_private_evidence_directory(evidence_root, "content bridge root"))
        _bridge(
            lambda: common.write_create_only_json(
                bridge_fd,
                BRIDGE_RECEIPT_NAME,
                draft,
                "content bridge receipt",
            )
        )
        replayed = verify_reconstructed_prior_baseline_content_bridge_fd(
            bridge_fd,
            baseline_root_fd=external["baseline root"],
            baseline_manifest=baseline_manifest,
            source_input_root_fd=external["source inputs root"],
            expected_source_archive_sha256=expected_sha256,
            runtime_oci_a_root_fd=external["runtime OCI A root"],
            runtime_oci_b_root_fd=external["runtime OCI B root"],
        )
        if replayed != draft:
            _fail("prepublication-replay-drift", "held bridge replay differs from the draft receipt")
        return draft
    finally:
        if bridge_fd is not None:
            os.close(bridge_fd)
        _close_roots(external)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--source-input-root", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--runtime-oci-a-root", type=Path, required=True)
    parser.add_argument("--runtime-oci-b-root", type=Path, required=True)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="replay an existing bridge root instead of creating one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        arguments = {
            "baseline_root": args.baseline_root,
            "baseline_manifest": args.baseline_manifest,
            "source_input_root": args.source_input_root,
            "expected_source_archive_sha256": args.expected_source_archive_sha256,
            "runtime_oci_a_root": args.runtime_oci_a_root,
            "runtime_oci_b_root": args.runtime_oci_b_root,
        }
        if args.verify:
            receipt = verify_reconstructed_prior_baseline_content_bridge(args.evidence_root, **arguments)
        else:
            receipt = prepare_reconstructed_prior_baseline_content_bridge(args.evidence_root, **arguments)
    except ReconstructedPriorBaselineContentBridgeError as error:
        payload = {
            "schema_version": BRIDGE_VERSION,
            "status": "failed",
            "qualification_status": QUALIFICATION_STATUS,
            "reason_codes": [getattr(error, "reason_code", "invalid-content-bridge")],
            "error": str(error),
        }
        sys.stderr.buffer.write(common.canonical_json_bytes(payload) + b"\n")
        return 1
    sys.stdout.buffer.write(common.canonical_json_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
